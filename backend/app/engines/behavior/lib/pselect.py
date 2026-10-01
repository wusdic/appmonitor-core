"""Attribute and feature selection of the progressive core (docs/lib3/progressive.md §6.4).

STATUS: implemented (W-P2, P05 maths; the engine is engines/behavior/attr_select.py).
Pure functions and bounded structures; no store access except `selection_for`,
the one accessor every learner (P04, P02, P01) uses to read the current
selection (model.attrsel) with the bootstrap fallback.

Which metrics are used, kept, and at which detail is *learned*, never listed
in code (PPC-4, requirement S16/S17):

  StratifiedProbe  R_p learned events per (tree, kind), stratified by the tree's
                   own stratum key (§6.2.2) with allocation ~ sqrt(stratum mass)
                   and >= min(32, stratum size) rows per stratum, time-decayed
                   (H_m) priority keys; every row keeps its HT mass. Rows are
                   stored as (interned key tuple, value tuple) so memory is
                   O(R_p x attributes per event), not a dict per row.
  evaluate()       per attribute on the probe (mass-weighted, Miller-Madow):
                   level = finest level with <= 64 distinct values; H(a);
                   CR(a) = max_c (1 - H(a|c)/H(a)) over context candidates C0;
                   U_t(a) = cov S (H(a) - min_c H(a|c)) - lambda_c cost_us;
                   U_s(a, l) = sum_b I(b; gen(a, l)) - (card - 1) K log2(n) / (2n);
                   g3(a -> b) redundancy among kept attributes.
  assign_roles()   invariant / split / target / shape / redundant / dropped with
                   hysteresis (promote at u_hi = 0.05 bits/event, demote below
                   u_lo = 0.02 for 3 consecutive runs), `net.src` removed from
                   split candidates when it carries no information at any IP
                   level (the "直接 IP 不作为特征" case, who_mode = none).
  node_overrides() per node, the targets re-ranked by node-local H x coverage
                   from the node summaries (constants become node invariants and
                   free their slot), top m_t per node.
  value_groups()   categorical level-1 value groups: agglomerative merge of values
                   whose conditional target distributions are within JSD 0.02 bits.

model.attrsel@(tree key, '__system__') =
    {version, t, roles{a: role}, levels{a: [l...]}, targets_sys{kind: [a...]},
     split_cands{kind: [(a, l)...]}, redundant{b: a}, node_overrides{kind: {nid: [a...]}},
     who_mode, stats{a: {...}}, runs}
"""
from __future__ import annotations

import heapq
import math
import zlib
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import pmdl
from . import psketch as PS
from .pevent import ABSENT, KIND_TXN, KIND_WIN

ATTRSEL = "model.attrsel"
R_P = 4096                 # probe rows per (tree, kind)
R_P_MIN_STRATUM = 32       # at least min(32, size) rows per stratum
STRATA_MAX = 256           # strata tracked per probe (lowest decayed mass evicted)
A_PROBE = 64               # rotating slice of attributes evaluated per run
M_T = 8                    # targets per node
M_SYS = 32                 # system target list length (node overrides draw from it)
U_HI = 0.05                # bits / event: promote
U_LO = 0.02                # bits / event: demote (after DEMOTE_RUNS consecutive runs)
DEMOTE_RUNS = 3
LAMBDA_C = 0.001           # bits per microsecond
LEVEL_CARD = 64            # the evaluation level is the finest with <= 64 distinct values
INV_H = 0.05               # invariant: H <= 0.05 bits ...
INV_COV = 0.99             # ... and coverage >= 0.99
INV_COV_EXIT = 0.97        # an invariant leaves below this coverage (or above 2 INV_H bits)
TARGET_COV = 0.05
LOCAL_TARGET_COV = 0.5     # node-local attributes must cover half of a node's rows
TARGET_STAB = 0.7
SHAPE_DISTINCT = 0.5
SHAPE_CR = 0.05
CR_LOCAL = 0.10           # predictability where present that makes a rare attribute a target
CR_LOCAL_LO = 0.05
CLOSED_CARD = 16           # a categorical with <= this many values where present is a closed-set target
CLOSED_COV_MAX = 0.25      # ... when it is a field of some actions (present on <= 25 % of the events)
N_LOCAL = 8                # probe rows it must be present in (the penalised gain carries the small-sample cost)
RED_G3 = 0.01
RED_HB = 0.1
RED_RHO = 0.99             # numeric pairs: |Spearman rho| where both are present
IP_INFO_MIN = 0.02         # CR(net.src) below this at every level: who = none
REPROBE_S = 7 * PS.DAY     # dropped attributes are re-probed every 7 days
VG_JSD = 0.02              # value-group merge threshold (bits)
SPLIT_MAX_LEVELS = {"ip": 2, "tod": 2, "when": 2, "route": 2, "path": 2}
WHO_ATTRS = frozenset({"net.src", "net.peer_src"})
CONTEXT_SEEDS = ("http.route", "net.src", "ctx.tod_min", "ev.ch", "ctx.daytype", "ctx.when")
WHEN_ATTRS = frozenset({"ctx.tod_min", "ctx.when"})
# bookkeeping attributes: never targets or split candidates (ids, running positions)
NON_TARGET_PREFIX = ("ev.",)
NON_TARGET = frozenset({"ctx.sid", "ctx.sess_age_s", "ctx.think_s", "sess.key", "ctx.sess_pos"})
# attributes derived from the same source field (a split on one trivially predicts the other)
DERIVED_GROUPS: Tuple[frozenset, ...] = (
    frozenset({"http.route", "http.path", "http.method", "http.host", "hdr.host"}),
    frozenset({"net.src", "net.peer_src"}),
    frozenset({"ctx.tod_min", "ctx.when"}),
    frozenset({"ctx.daytype", "ctx.dayclass", "ctx.dow", "ctx.when"}),
    frozenset({"http.status", "http.sclass"}),
    frozenset({"net.dst", "net.dport"}),
    # R3's client stack token is a function of JA3, user agent, TTL and TCP window
    # (lib/stack.stack_token): a split on one of them trivially "predicts" the others
    frozenset({"client.stack", "http.ua", "hdr.user-agent", "tls.ja3", "net.ttl", "net.win"}),
)
# calendar context (§5.4.3): split candidates of the when facet, never m_t targets
# (when is summarised separately in every node)
CALENDAR_ATTRS = frozenset({"ctx.dow", "ctx.daytype", "ctx.dayclass", "ctx.dom", "ctx.mend"})
MISSING = "\x00unrecorded"   # probe: the row did not record this attribute
ROLE_ORDER = ("invariant", "redundant", "split", "target", "shape", "dropped", "probe")


# ================================================================== helpers
def same_source(a: str, b: str) -> bool:
    """True when b is derived from the same source field as a (the split
    attribute's own hierarchy and its derivations are not targets, §6.5.3)."""
    if a == b:
        return True
    if b.startswith(a + ".") or a.startswith(b + "."):
        return True
    def _in(x: str, g: frozenset) -> bool:           # x or a derivation of it (x.len, x.keys)
        return x in g or any(x.startswith(y + ".") for y in g)
    for g in DERIVED_GROUPS:
        if _in(a, g) and _in(b, g):
            return True
    # X.keys vs X.kv.<k> (the key set is the presence pattern of the values)
    for x, y in ((a, b), (b, a)):
        if x.endswith(".keys") and y.startswith(x[:-5] + ".kv."):
            return True
        # X.len vs its parts (X.kv.<k>, X.kv.<k>.len, X.keys): the length of a body
        # is the sum of its fields' lengths, so a split on the body-size bin
        # "predicts" the length of its padding field trivially (pack O: the OA
        # login children split on body.len paid by body.kv.viewstate.len)
        if x.endswith(".len") and (y.startswith(x[:-4] + ".kv.") or y == x[:-4] + ".keys"):
            return True
    return False


def targetable(a: str) -> bool:
    """Attributes that may be m_t targets (who and when are always summarised
    separately and are not counted in m_t; bookkeeping ids never are)."""
    if a in WHO_ATTRS or a in WHEN_ATTRS or a in NON_TARGET or a in CALENDAR_ATTRS:
        return False
    return not a.startswith(NON_TARGET_PREFIX)


def _hashable(v: Any) -> Hashable:
    try:
        hash(v)
        return v
    except TypeError:
        return repr(v)


def codes_of(values: Sequence[Any]) -> Tuple[np.ndarray, int]:
    """Integer codes of a value column (ABSENT is a value). Returns (codes, K)."""
    idx: Dict[Hashable, int] = {}
    out = np.empty(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        k = _hashable(v)
        j = idx.get(k)
        if j is None:
            j = idx[k] = len(idx)
        out[i] = j
    return out, len(idx)


def codes_uniq(values: Sequence[Any]) -> Tuple[np.ndarray, List[Any]]:
    """Integer codes of a value column and the distinct raw values in code order."""
    try:                                                       # fast path: hashable values
        d: Dict[Any, int] = {}
        codes = [d.setdefault(v, len(d)) for v in values]
        return np.asarray(codes, dtype=np.int64), list(d.keys())
    except TypeError:
        pass
    idx: Dict[Hashable, int] = {}
    uniq: List[Any] = []
    out = np.empty(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        k = _hashable(v)
        j = idx.get(k)
        if j is None:
            j = idx[k] = len(uniq)
            uniq.append(v)
        out[i] = j
    return out, uniq


def _num_levels(hier: Any, a: str, level: int, uniq: Sequence[Any]) -> Optional[List[Any]]:
    """Vectorised numeric bins (levels 1-3 of a numeric hierarchy) for the
    distinct values; None when not applicable (the generic path is used)."""
    if level not in (1, 2, 3) or hier.kind(a) != "num" or not hasattr(hier, "_num_edges"):
        return None
    edges, lg = hier._num_edges(a)
    if edges is None:
        return None
    x = np.full(len(uniq), np.nan)
    for i, v in enumerate(uniq):
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            x[i] = float(v)
    ok = np.isfinite(x)
    if lg:
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), -np.inf)
    b = np.searchsorted(edges, x, side="right") >> (level - 1)
    out: List[Any] = [int(v) for v in b.tolist()]
    for i in np.flatnonzero(~ok).tolist():
        out[i] = hier.gen(a, level, uniq[i])
    return out


_GEN_CACHE: Dict[Tuple, Dict[Any, Any]] = {}
_GEN_CACHE_VALUES = 8192           # values per (attribute, level, model fingerprint)
_GEN_CACHE_KEYS = 4096


def _gen_fingerprint(hier: Any, a: str, level: int) -> Tuple:
    """What gen(a, level, .) depends on besides the value: the registry record's
    version (types, bins, groups, set templates) and the learned who / region /
    window models (their sizes)."""
    rec = hier._rec(a) if hasattr(hier, "_rec") else None
    ver = getattr(rec, "version", None) if rec is not None and not isinstance(rec, Mapping) else \
        (id(rec) if rec is not None else None)
    return (a, int(level), ver, hier.kind(a), len(getattr(hier, "ip2g", {}) or {}),
            getattr(hier, "n_groups", 0), len(getattr(hier, "regions", ()) or ()),
            len(getattr(hier, "windows", ()) or ()))


def gen_codes(hier: Any, a: str, level: int, codes0: np.ndarray, uniq: Sequence[Any]) -> Tuple[np.ndarray, int]:
    """Codes of gen(a, level, .) computed on the distinct values only (memoised
    across runs per model fingerprint: the probe keeps most rows between runs)."""
    fast = _num_levels(hier, a, level, uniq)
    if fast is not None:
        m0: Dict[Hashable, int] = {}
        lut0 = np.asarray([m0.setdefault(_hashable(g), len(m0)) for g in fast] or [0], dtype=np.int64)
        return (lut0[codes0] if codes0.size else codes0), len(m0)
    fp = _gen_fingerprint(hier, a, level)
    memo = _GEN_CACHE.get(fp)
    if memo is None:
        if len(_GEN_CACHE) >= _GEN_CACHE_KEYS:
            _GEN_CACHE.clear()
        memo = _GEN_CACHE[fp] = {}
    elif len(memo) > _GEN_CACHE_VALUES:
        memo.clear()
    m: Dict[Hashable, int] = {}
    lut = np.empty(max(1, len(uniq)), dtype=np.int64)
    for i, v in enumerate(uniq):
        hv = _hashable(v)
        g = memo.get(hv, memo)
        if g is memo:
            g = memo[hv] = _hashable(hier.gen(a, level, v))
        lut[i] = m.setdefault(g, len(m))
    return (lut[codes0] if codes0.size else codes0), len(m)


def level_codes(hier: Any, a: str, col: Sequence[Any], text_values: bool = False,
                raw: Optional[Tuple[np.ndarray, List[Any]]] = None) -> Tuple[int, np.ndarray, int]:
    """(level, codes, K) at the finest level with <= LEVEL_CARD distinct values
    (numeric at bins, text at shape at the finest unless text_values)."""
    c0, uniq = raw if raw is not None else codes_uniq(col)
    kind = hier.kind(a)
    L = hier.n_levels(a)
    start = 1 if kind == "num" or (kind == "text" and not text_values) else 0
    best = None
    for l in range(start, L - 1):
        if l == 0 and len(uniq) > LEVEL_CARD:
            continue
        cc, k = gen_codes(hier, a, l, c0, uniq)
        if k <= LEVEL_CARD:
            return l, cc, k
        best = (l, cc, k)
    if best is None:
        cc, k = gen_codes(hier, a, L - 1, c0, uniq)
        return L - 1, cc, k
    return best


def w_entropy(codes: np.ndarray, w: np.ndarray, n_rows: Optional[float] = None) -> float:
    """Mass-weighted plug-in entropy + Miller-Madow correction (K - 1)/(2 n ln 2)
    with n = rows (the evidence the probe holds, not the mass)."""
    if codes.size == 0:
        return 0.0
    tot = np.bincount(codes, weights=w)
    s = tot.sum()
    if s <= 0:
        return 0.0
    p = tot[tot > 0] / s
    h = float(-(p * np.log2(p)).sum())
    n = float(n_rows if n_rows is not None else codes.size)
    k = int((tot > 0).sum())
    return max(0.0, h + (k - 1) / (2.0 * max(n, 1.0) * math.log(2.0)))


def w_cond_entropy(a: np.ndarray, ka: int, c: np.ndarray, kc: int, w: np.ndarray) -> float:
    """H(a | c) = H(a, c) - H(c), mass-weighted, Miller-Madow on both terms."""
    joint = a * max(kc, 1) + c
    return max(0.0, w_entropy(joint, w) - w_entropy(c, w))


def w_plugin(codes: np.ndarray, w: np.ndarray) -> Tuple[float, int]:
    """(mass-weighted plug-in entropy in bits, number of occupied cells)."""
    if codes.size == 0:
        return 0.0, 0
    tot = np.bincount(codes, weights=w)
    s = tot.sum()
    if s <= 0:
        return 0.0, 0
    nz = tot[tot > 0]
    p = nz / s
    return float(-(p * np.log2(p)).sum()), int(nz.size)


def penalised_gain(b: np.ndarray, kb: int, g: np.ndarray, kg: int, w: np.ndarray, n: int,
                   hb: Optional[float] = None, hg: Optional[Tuple[float, int]] = None) -> float:
    """Information the grouping g carries about b beyond chance:
    I_plugin(b; g) - (K_b - 1)(K_g - 1) log2(n) / (2 n) bits (the BIC / chi-square
    scale of the plug-in bias; K = occupied values). Selecting the best of many
    candidates on the Miller-Madow estimate alone admits noise attributes (the
    maximum of hundreds of null estimates), so every gain in §6.4 is penalised
    this way: U_t uses the best context's penalised gain, U_s sums the
    per-target penalised gains clipped at 0."""
    if n <= 1:
        return 0.0
    if hb is None:
        h_b, kb_occ = w_plugin(b, w)
    elif isinstance(hb, tuple):
        h_b, kb_occ = hb
    else:
        h_b, kb_occ = hb, int(np.unique(b).size)
    h_g, kg_occ = w_plugin(g, w) if hg is None else hg
    h_bg, _ = w_plugin(b * max(kg, 1) + g, w)
    mi = max(0.0, h_b + h_g - h_bg)
    return mi - max(0, kb_occ - 1) * max(0, kg_occ - 1) * math.log2(n) / (2.0 * n)


def w_g3(a: np.ndarray, b: np.ndarray, kb: int, w: np.ndarray) -> float:
    """g3(a -> b) = 1 - sum_x max_y c(x, y) / n (mass-weighted)."""
    if a.size == 0:
        return 1.0
    joint = a * max(kb, 1) + b
    tot = np.bincount(joint, weights=w)
    nz = np.flatnonzero(tot > 0)
    xs = nz // max(kb, 1)
    best: Dict[int, float] = {}
    for x, c in zip(xs.tolist(), tot[nz].tolist()):
        if c > best.get(x, 0.0):
            best[x] = c
    s = float(tot.sum())
    return 1.0 - sum(best.values()) / s if s > 0 else 1.0


# ============================================================ stratified probe
class StratifiedProbe:
    """Probe sample of learned events (§6.4): per stratum a reservoir with
    time-decayed priority keys (Efraimidis-Spirakis, w = 2^((t - L)/H_m), so
    recent rows are preferred), capacity R_k = max(min(32, n_k), R_p sqrt(m_k) /
    sum_j sqrt(m_j)) where m_k is the stratum's H_m-decayed mass. Every row keeps
    its HT mass; `rows()` returns probe weights = stratum mass x row mass share."""

    def __init__(self, R: int = R_P, strata_max: int = STRATA_MAX) -> None:
        self.R = int(R)
        self.strata_max = int(strata_max)
        self.L: Optional[float] = None
        self.strata: Dict[Hashable, List[Tuple[float, int, int, tuple, float, float]]] = {}
        self.smass: Dict[Hashable, PS.DecayedVector] = {}
        self.cap: Dict[Hashable, int] = {}
        self._schemas: Dict[tuple, int] = {}
        self._schema_list: List[Tuple[tuple, Dict[str, int]]] = []
        self._sid_want: List[Optional[int]] = []
        self._tcache: Optional[Tuple[Any, Any]] = None
        self._wants: List[frozenset] = []
        self._want_ix: Dict[frozenset, int] = {}
        self._seq = 0
        self._size = 0
        self._mut = 0                  # mutation counter (rows / codes caches)
        self._rcache: Optional[Tuple[Any, Any]] = None
        self._ccache: Optional[Tuple[Any, Dict[str, Any]]] = None
        self.n_offered = 0

    def _schema(self, keys: tuple, wid: Optional[int] = None) -> int:
        k = (keys, wid)
        sid = self._schemas.get(k)
        if sid is None:
            if len(self._schema_list) > 8192:          # bounded intern table
                self._gc_schemas()
            sid = self._schemas[k] = len(self._schema_list)
            self._schema_list.append((keys, {x: i for i, x in enumerate(keys)}))
            self._sid_want.append(wid)
        return sid

    def want_id(self, want: Optional[Iterable[str]]) -> Optional[int]:
        """Intern the set of attributes a row records (None = all of the event's)."""
        if want is None:
            return None
        fw = frozenset(want)
        wid = self._want_ix.get(fw)
        if wid is None:
            if len(self._wants) > 1024:
                return None
            wid = self._want_ix[fw] = len(self._wants)
            self._wants.append(fw)
        return wid

    def _gc_schemas(self) -> None:
        used = {r[2] for rows in self.strata.values() for r in rows}
        old = self._schema_list
        old_w = self._sid_want
        self._schemas, self._schema_list, self._sid_want = {}, [], []
        remap = {}
        for sid in sorted(used):
            keys = old[sid][0]
            remap[sid] = len(self._schema_list)
            self._schemas[(keys, old_w[sid])] = remap[sid]
            self._schema_list.append(old[sid])
            self._sid_want.append(old_w[sid])
        for s, rows in self.strata.items():
            self.strata[s] = [(a, b, remap[c], d, e, f) for a, b, c, d, e, f in rows]

    def _key(self, t: float, u: float) -> float:
        if self.L is None:
            self.L = float(t)
        w = 2.0 ** ((float(t) - self.L) / PS.H_M)
        return math.log(max(min(u, 1.0 - 1e-16), 1e-300)) / w

    def offer(self, stratum: Hashable, row: Mapping[str, Any], mass: float, t: float, u: float,
              want_id: Optional[int] = None) -> None:
        self.n_offered += 1
        self._mut += 1
        sm = self.smass.get(stratum)
        if sm is None:
            if len(self.smass) >= self.strata_max:
                self._evict_stratum(t)
            sm = self.smass[stratum] = PS.DecayedVector([PS.H_M])
            self.strata[stratum] = []
            self.cap[stratum] = R_P_MIN_STRATUM
        sm.add(t, float(mass))
        keys = tuple(sorted(row.keys()))
        sid = self._schema(keys, want_id)
        vals = tuple(row[k] for k in keys)
        k = self._key(t, u)
        heap = self.strata[stratum]
        self._seq += 1
        item = (k, self._seq, sid, vals, float(mass), float(t))
        cap = self.cap.get(stratum, R_P_MIN_STRATUM)
        if len(heap) < cap or self._size < self.R:
            heapq.heappush(heap, item)
            self._size += 1
        elif k > heap[0][0]:
            heapq.heapreplace(heap, item)
        if self._size > self.R + R_P_MIN_STRATUM * len(self.cap) + 1:
            self.rebalance(t)                  # bounded memory between hourly runs

    def _evict_stratum(self, t: float) -> None:
        worst = min(self.smass, key=lambda s: self.smass[s].get(0, t))
        self.smass.pop(worst, None)
        self._size -= len(self.strata.pop(worst, None) or ())
        self.cap.pop(worst, None)

    def rebalance(self, t: float, R: Optional[int] = None) -> None:
        """Recompute stratum capacities ~ sqrt(mass) and trim the reservoirs."""
        if R is not None:
            self.R = int(R)
        if not self.smass:
            return
        self._mut += 1
        m = {s: max(0.0, v.get(0, t)) for s, v in self.smass.items()}
        sq = {s: math.sqrt(x) for s, x in m.items()}
        tot = sum(sq.values()) or 1.0
        for s in self.smass:
            share = int(self.R * sq[s] / tot)
            self.cap[s] = max(R_P_MIN_STRATUM, share)
        # overall bound: shrink the largest strata while the sum exceeds R + floor slack
        total_cap = sum(self.cap.values())
        limit = self.R + R_P_MIN_STRATUM * len(self.cap)
        while total_cap > limit:
            s = max(self.cap, key=self.cap.get)
            if self.cap[s] <= R_P_MIN_STRATUM:
                break
            dec = min(self.cap[s] - R_P_MIN_STRATUM, total_cap - limit)
            self.cap[s] -= dec
            total_cap -= dec
        for s, heap in self.strata.items():
            c = self.cap[s]
            while len(heap) > c:
                heapq.heappop(heap)
        self._size = sum(len(h) for h in self.strata.values())

    def __len__(self) -> int:
        return sum(len(h) for h in self.strata.values())

    def rows(self, t: float) -> Tuple[List[Tuple[int, tuple]], np.ndarray, List[Hashable]]:
        """([(schema id, values)], probe weights, stratum per row). The result
        is cached until the probe changes, so the column transposition and the
        value codes of one evaluation are shared by every function of the run
        (evaluate, ip_information, who_proxies, redundancy, node targets).
        Callers must not modify the returned arrays."""
        key = (float(t), self._mut)
        if self._rcache is not None and self._rcache[0] == key:
            return self._rcache[1]
        res = self._rows(t)
        self._rcache = (key, res)
        return res

    def codes(self, rows: Sequence[Tuple[int, tuple]], name: str) -> Tuple[np.ndarray, List[Any]]:
        """codes_uniq of a column of `rows`, cached per rows object."""
        c = self._ccache
        if c is None or c[0] is not rows:
            c = self._ccache = (rows, {})
        r = c[1].get(name)
        if r is None:
            r = c[1][name] = codes_uniq(self.column(rows, name))
        return r

    def _rows(self, t: float) -> Tuple[List[Tuple[int, tuple]], np.ndarray, List[Hashable]]:
        out: List[Tuple[int, tuple]] = []
        wts: List[float] = []
        strata: List[Hashable] = []
        for s, heap in self.strata.items():
            if not heap:
                continue
            ms = max(0.0, self.smass[s].get(0, t))
            tot = sum(r[4] for r in heap) or 1.0
            for r in heap:
                out.append((r[2], r[3]))
                wts.append(ms * r[4] / tot)
                strata.append(s)
        return out, np.asarray(wts, dtype=np.float64), strata

    def column(self, rows: Sequence[Tuple[int, tuple]], name: str) -> List[Any]:
        """Values of `name` per row: the value, ABSENT when the event lacked it,
        MISSING when the row did not record it (it was not wanted then). Rows are
        transposed once per schema (cached for the `rows` list object)."""
        if self._tcache is None or self._tcache[0] is not rows:
            groups: Dict[int, List[int]] = {}
            for p, (sid, _) in enumerate(rows):
                groups.setdefault(sid, []).append(p)
            tg = []
            for sid, pos in groups.items():
                cols = list(zip(*[rows[p][1] for p in pos])) if self._schema_list[sid][0] else []
                tg.append((sid, pos, cols))
            self._tcache = (rows, tg)
        sl = self._schema_list
        wants = self._wants
        out: List[Any] = [None] * len(rows)
        for sid, pos, cols in self._tcache[1]:
            i = sl[sid][1].get(name)
            if i is not None:
                for p, v in zip(pos, cols[i]):
                    out[p] = v
            else:
                wid = self._sid_want[sid]
                fill = ABSENT if wid is None or name in wants[wid] else MISSING
                for p in pos:
                    out[p] = fill
        return out

    def release(self) -> None:
        """Drop the transposition, rows and codes caches (after an evaluation)."""
        self._tcache = None
        self._rcache = None
        self._ccache = None

    def names(self) -> List[str]:
        used = {r[2] for rows in self.strata.values() for r in rows}
        out: set = set()
        for sid in used:
            out.update(self._schema_list[sid][0])
        return sorted(out)

    def nbytes(self) -> int:
        b = 400
        for heap in self.strata.values():
            for r in heap:
                b += 120 + 8 * len(r[3])
        b += sum(64 + 8 * len(k) for k, _ in self._schema_list)
        return int(b)


# ================================================================ evaluation
def eval_level(hier: Any, a: str, col: Sequence[Any]) -> Tuple[int, List[Any]]:
    """The finest level with <= LEVEL_CARD distinct values on the probe
    (numeric at bins, text at shape at the finest), and the generalised column."""
    kind = hier.kind(a)
    L = hier.n_levels(a)
    start = 1 if kind in ("num", "text") else 0
    best = None
    for l in range(start, L - 1):
        g = [hier.gen(a, l, v) for v in col]
        k = len({_hashable(x) for x in g})
        if k <= LEVEL_CARD:
            return l, g
        best = (l, g)
    if best is None:
        return L - 1, [hier.gen(a, L - 1, v) for v in col]
    return best


class AttrStats(dict):
    """Per-attribute evaluation record (plain dict for storage)."""


TIME_TARGET = "ctx.tod_min"   # the time of day is behaviour: a split candidate's utility counts it


def evaluate(probe: StratifiedProbe, t: float, hier: Any, names: Sequence[str],
             registry: Any = None, targets_prev: Sequence[str] = (),
             splits_prev: Sequence[str] = (), coverage: Optional[Callable[[str], float]] = None,
             stability: Optional[Callable[[str], float]] = None,
             cost_us: Optional[Callable[[str], float]] = None,
             proxies: Sequence[str] = ()) -> Dict[str, Dict[str, Any]]:
    """Evaluate attributes `names` on the probe (§6.4). Returns {a: stats} with
    keys level, H, H0, distinct0, CR, U_t, U_s{l: bits}, best_levels, card{l},
    cov, S, n."""
    rows, w, _ = probe.rows(t)
    n = len(rows)
    out: Dict[str, Dict[str, Any]] = {}
    if n == 0:
        return out
    wsum = float(w.sum()) or 1.0
    colcache: Dict[str, List[Any]] = {}

    def col(a: str) -> List[Any]:
        c = colcache.get(a)
        if c is None:
            c = colcache[a] = probe.column(rows, a)
        return c
    levcache: Dict[str, Tuple[int, np.ndarray, int]] = {}

    rawcache: Dict[str, Tuple[np.ndarray, List[Any]]] = {}

    def raw(a: str) -> Tuple[np.ndarray, List[Any]]:
        r = rawcache.get(a)
        if r is None:
            r = rawcache[a] = probe.codes(rows, a)
        return r

    def lev(a: str) -> Tuple[int, np.ndarray, int]:
        r = levcache.get(a)
        if r is None:
            r = levcache[a] = level_codes(hier, a, col(a), raw=raw(a))
        return r

    # context candidates C0: previous split attributes (top 5) + seeds present
    present = set(probe.names())
    seeds: List[Tuple[str, int]] = []
    if "http.route" in present:
        seeds.append(("http.route", 0))
    if "net.src" in present:
        seeds.append(("net.src", 3 if getattr(hier, "ip2g", None) else 1))
    if "ctx.tod_min" in present:
        seeds.append(("ctx.tod_min", 1))
    if "ev.ch" in present:
        seeds.append(("ev.ch", 0))
    ctxs: List[Tuple[str, int]] = [(a, -1) for a in list(splits_prev)[:5] if a in present]
    for s in seeds:
        if s[0] not in {c[0] for c in ctxs}:
            ctxs.append(s)
    ctx_codes: List[Tuple[str, np.ndarray, int]] = []
    for a, l in ctxs:
        if l < 0:
            ll, cc, k = lev(a)
        else:
            cc, k = gen_codes(hier, a, l, *raw(a))
        ctx_codes.append((a, cc, k))
    ctx_h = [w_plugin(cc, w) for _, cc, _ in ctx_codes]      # H(context) on all rows, once
    # target codes for U_s: the BEHAVIOUR a split would explain (§6.5.3, P04 M26):
    # the system targets except source properties (a client stack is predicted
    # by every who level and says nothing about what the sources do), plus the
    # time of day - P04 codes it as the @when target of every split, and it is
    # often the only behaviour that tells groups apart (measured on pack O's
    # mail, opaque TLS: the system target list was the TCP-window class alone,
    # so no who or time level ever got the split role and the departments' mail
    # windows were never separated)
    prox = set(proxies)
    tgt = [b for b in targets_prev if b in present and b not in prox][:2 * M_T]
    if TIME_TARGET in present and TIME_TARGET not in tgt:
        tgt.append(TIME_TARGET)
    tgt_codes = [(b,) + lev(b)[1:] for b in tgt]
    tknown = {b: np.fromiter((v is not MISSING for v in col(b)), dtype=bool, count=n) for b in tgt}
    Hb = {b: w_plugin(cc, w)[0] for b, cc, _ in tgt_codes}
    Kb = {b: int(np.unique(cc).size) for b, cc, _ in tgt_codes}
    for a in names:
        if a not in present:
            continue
        c_full = col(a)
        # rows where the attribute was recorded (a probe row keeps only the
        # attributes wanted when it was taken; unrecorded is not absent)
        known = np.fromiter((v is not MISSING for v in c_full), dtype=bool, count=n)
        na = int(known.sum())
        if na < N_LOCAL:
            continue
        if na == n:
            ix = None
            c0, wa = c_full, w
            l, cc, k = lev(a)
            c0codes, uniq0 = raw(a)
        else:
            ix = np.flatnonzero(known)
            c0 = [c_full[i] for i in ix]
            wa = w[ix]
            c0codes, uniq0 = codes_uniq(c0)
            l, cc, k = level_codes(hier, a, c0, raw=(c0codes, uniq0))

        def sub(arr: np.ndarray) -> np.ndarray:
            return arr if ix is None else arr[ix]
        wsa = float(wa.sum()) or 1.0
        H = w_entropy(cc, wa, na)
        k0 = len(uniq0)
        H0 = w_entropy(c0codes, wa, na)
        pres = np.asarray([v is not ABSENT for v in c0])
        cov_probe = float(wa[pres].sum() / wsa)
        cov = min(1.0, float(coverage(a))) if coverage is not None else cov_probe
        S = float(stability(a)) if stability is not None else 1.0
        cost = float(cost_us(a)) if cost_us is not None else 0.0
        # predictability: the best context's penalised information gain
        hp, _ = w_plugin(cc, wa)
        hp0, _ = w_plugin(c0codes, wa)
        gain = 0.0
        gain0 = 0.0
        for (ca, ccodes, ck), hc in zip(ctx_codes, ctx_h):
            if same_source(ca, a):
                continue
            cs = sub(ccodes)
            hg = hc if ix is None else w_plugin(cs, wa)
            gain = max(gain, penalised_gain(cc, k, cs, ck, wa, na, hp, hg))
            if hp0 > 0:
                gain0 = max(gain0, penalised_gain(c0codes, k0, cs, ck, wa, na, hp0, hg))
        CR = min(1.0, gain / hp) if hp > 1e-9 else 0.0
        CR0 = min(1.0, gain0 / hp0) if hp0 > 1e-9 else 0.0
        U_t = cov * S * gain - LAMBDA_C * cost
        # the same gain on the rows where the attribute is present, weighted by its
        # share of PROBE ROWS: the probe is stratified ~ sqrt(stratum mass), so an
        # attribute of a rare action (a login body among health checks) is not
        # hidden by the dominant stratum's mass (targets are re-ranked per node)
        n_p = int(pres.sum())
        cov_rows = n_p / max(na, 1)
        U_tc = 0.0
        CR_p = CR
        H_p = H
        if N_LOCAL <= n_p < na:
            wp = wa[pres]
            # the level is chosen on the present values (a closed set of user names
            # is kept as values, not only as their common shape)
            _, ccp, kp = level_codes(hier, a, [v for v, p_ in zip(c0, pres) if p_], text_values=True)
            hpp, _ = w_plugin(ccp, wp)
            H_p = hpp
            gp = 0.0
            for ca, ccodes, ck in ctx_codes:
                if same_source(ca, a):
                    continue
                gp = max(gp, penalised_gain(ccp, kp, sub(ccodes)[pres], ck, wp, n_p, hpp))
            U_tc = cov_rows * S * gp - LAMBDA_C * cost
            CR_p = min(1.0, gp / hpp) if hpp > 1e-9 else 0.0
        elif n_p == na:
            U_tc = U_t
        # split utility per level
        U_s: Dict[int, float] = {}
        card: Dict[int, int] = {}
        L = hier.n_levels(a)
        # each target on the rows where both it and a were recorded
        tsub = []
        for b, bc, bk in tgt_codes:
            if same_source(a, b):
                continue
            kb_ = sub(tknown[b])
            if kb_.all():
                m_b = None
                bcs = sub(bc)
                hb_ = (Hb[b], Kb[b]) if ix is None else w_plugin(bcs, wa)
            else:
                m_b = kb_
                if m_b.sum() < N_LOCAL:
                    continue
                bcs = sub(bc)[m_b]
                hb_ = w_plugin(bcs, wa[m_b])
            tsub.append((b, bcs, bk, m_b, hb_))
        for lv in range(0, L - 1):
            if lv == 0 and len(uniq0) > LEVEL_CARD:
                card[0] = len(uniq0)
                continue
            gc, gk = gen_codes(hier, a, lv, c0codes, uniq0)
            card[lv] = gk
            if gk > LEVEL_CARD or gk < 2:
                continue
            s_ = 0.0
            hg = w_plugin(gc, wa)
            for b, bc, bk, m_b, hb_ in tsub:
                if m_b is None:
                    s_ += max(0.0, penalised_gain(bc, bk, gc, gk, wa, na, hb_, hg))
                else:
                    s_ += max(0.0, penalised_gain(bc, bk, gc[m_b], gk, wa[m_b], int(m_b.sum()), hb_))
            U_s[lv] = s_
        kind = hier.kind(a)
        nbest = SPLIT_MAX_LEVELS.get(kind, 1)
        ranked = sorted(U_s.items(), key=lambda kv: -kv[1])
        best_levels = [lv for lv, u in ranked[:nbest] if u >= U_LO]
        out[a] = {"level": int(l), "H": H, "H0": H0, "distinct0": k0 / max(na, 1),
                  "CR": CR, "CR0": CR0, "U_t": U_t, "U_s": {int(k_): float(v) for k_, v in U_s.items()},
                  "U_s_max": max(U_s.values()) if U_s else -math.inf,
                  "best_levels": best_levels, "card": card, "cov": cov, "S": S, "n": na,
                  "kind": kind, "U_tc": U_tc, "cov_rows": cov_rows, "CR_p": CR_p, "n_p": n_p,
                  "H_p": H_p}
    return out


def ip_information(probe: StratifiedProbe, t: float, hier: Any, targets: Sequence[str]) -> Dict[int, float]:
    """CR(net.src @ l) for l in /32, /24, grp, reg: how much the IP at that level
    predicts the targets (max over targets of 1 - H(b|ip)/H(b)); the input of
    the 'IP is not a feature' decision (§6.4) and of P12's ip_info."""
    rows, w, _ = probe.rows(t)
    n = len(rows)
    out: Dict[int, float] = {}
    if n == 0:
        return out
    ipcol = probe.column(rows, "net.src")
    tg = []
    c_ip, u_ip = probe.codes(rows, "net.src")
    for b in targets:
        if same_source("net.src", b):
            continue
        l, cc, k = level_codes(hier, b, probe.column(rows, b), raw=probe.codes(rows, b))
        hb = w_entropy(cc, w, n)
        if hb > 0.05:
            tg.append((cc, k, hb))
    for lv in (0, 1, 3, 4):
        gc, gk = gen_codes(hier, "net.src", lv, c_ip, u_ip)
        if gk < 2:
            out[lv] = 0.0
            continue
        best = 0.0
        for cc, k, hb in tg:
            best = max(best, penalised_gain(cc, k, gc, gk, w, n) / hb)
        # penalise levels whose values are nearly all distinct (a per-event id predicts
        # everything in-sample): MM-corrected conditional entropy handles part; require
        # the level's items to recur (<= 50 % singletons)
        cnt = np.bincount(gc)
        if (cnt == 1).sum() > 0.5 * gk:
            best = 0.0
        out[lv] = float(max(0.0, best))
    return out


PROXY_COV = 0.5            # a source property is present on >= half of the probe rows


def who_proxies(probe: StratifiedProbe, t: float, hier: Any, attrs: Sequence[str],
                g3_max: float = 0.05, min_h: float = 0.1, min_cov: float = PROXY_COV) -> List[str]:
    """Source properties ("identity proxies"): attributes that are a function
    of the source (g3(net.src -> a) <= g3_max, each IP shows one value), shared
    by several sources (>= 2 sources per value) and carried by the source's
    events whatever the action (present on >= min_cov of the probe rows): a
    department's client stack, TCP window class, TTL, user agent. They describe
    WHO the client is, not what it does, so P04 treats them as context: they
    are never split targets (a split is paid for by the behaviour it explains,
    §6.5.3) and, as split candidates, they yield to the who levels they stand
    in for unless they explain the behaviour better (P04 _check_valid_first);
    a split on one may also be revised into a who level (§6.6).

    Action fields are not source properties even when they are bound to the
    source: a login form's username is a function of the IP, but it is present
    on one action only (coverage << min_cov) and it is what the action submits.
    Time context (tod / calendar) is not a property of the source either,
    although a source active on workdays only shows one day type."""
    rows, w, _ = probe.rows(t)
    if not rows:
        return []
    # unweighted: one heavy automated source (a health monitor) would otherwise make
    # every attribute look like a function of the IP
    w = np.ones(len(rows))
    ip = probe.column(rows, "net.src")
    cip, uip = probe.codes(rows, "net.src")
    out = []
    for a in attrs:
        if a in WHO_ATTRS or not targetable(a):
            continue                                    # who, time / calendar context, bookkeeping
        col = probe.column(rows, a)
        known = np.fromiter((v is not MISSING and v is not ABSENT for v in col), dtype=bool, count=len(col))
        if known.sum() < N_LOCAL:
            continue
        rec = np.fromiter((v is not MISSING for v in col), dtype=bool, count=len(col))
        if known.sum() < min_cov * max(1, int(rec.sum())):
            continue                                    # an action's field, not a source property
        l, cc, k = level_codes(hier, a, [col[i] for i in np.flatnonzero(known)])
        ci = cip[known]
        wk = w[known]
        if k < 2 or w_entropy(cc, wk) < min_h:
            continue
        if w_g3(ci, cc, k, wk) <= g3_max and len(np.unique(ci)) >= 2 * k:
            out.append(a)
    return out


def redundancy(probe: StratifiedProbe, t: float, hier: Any, kept: Sequence[str],
               cost: Callable[[str], float], cov: Callable[[str], float],
               prev: Optional[Mapping[str, str]] = None,
               distinct: Optional[Callable[[str], float]] = None) -> Dict[str, str]:
    """{b: a}: b is redundant given a when g3(a -> b) <= 0.01 and H(b) >= 0.1
    (and a -> b holds; the member of the pair with lower cost and higher
    coverage is kept). Keeper order, so that the choice is stable from run to
    run (measured: re-choosing among equivalent attributes every hour flipped
    client.stack / http.ua / net.ttl and http.route / http.path between split
    and redundant on every run): an attribute that was not redundant in the
    previous run first, then lower cost, higher coverage, fewer distinct values
    (the more general of two equivalent attributes: a route template rather
    than raw paths), name."""
    rows, w, _ = probe.rows(t)
    n = len(rows)
    if n == 0 or len(kept) < 2:
        return {}
    codes = {}
    for a in kept:
        l, cc, k = level_codes(hier, a, probe.column(rows, a), raw=probe.codes(rows, a))
        codes[a] = (cc, k)
    H = {a: w_entropy(codes[a][0], w, n) for a in kept}
    # numeric pairs: the same quantity under two names (response size and bytes
    # down) is redundant when the rank correlation is ~1 where both are present
    nums: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for a in kept:
        if hier.kind(a) != "num":
            continue
        col = probe.column(rows, a)
        x = np.asarray([float(v) if isinstance(v, (int, float, np.integer, np.floating))
                        and not isinstance(v, bool) else np.nan for v in col])
        nums[a] = (x, np.isfinite(x))

    def ranks(v: np.ndarray) -> np.ndarray:
        o = np.argsort(v, kind="stable")
        r = np.empty(v.size)
        r[o] = np.arange(v.size)
        return r
    red: Dict[str, str] = {}
    was_red = set((prev or {}).keys())
    order = sorted(kept, key=lambda a: (a in was_red, round(cost(a), 1), -round(cov(a), 2),
                                        distinct(a) if distinct is not None else 0.0, a))
    for i, a in enumerate(order):
        if a in red:
            continue
        ca, ka = codes[a]
        for b in order[i + 1:]:
            if b in red or H[b] < RED_HB:
                continue
            cb, kb = codes[b]
            e_ab = w_g3(ca, cb, kb, w)
            if e_ab <= RED_G3:
                e_ba = w_g3(cb, ca, ka, w)
                if e_ba <= RED_G3:
                    # keep the member the other is (more nearly) a function of:
                    # a route template determines its path exactly, the path
                    # determines the route only on the probe (GET / POST of one
                    # path), so the route is the keeper (stickiness first)
                    if e_ba + 1e-9 < e_ab and (a in was_red) == (b in was_red) and H[a] >= RED_HB:
                        red[a] = b
                        break
                    red[b] = a
                    continue
            if a in nums and b in nums:
                (xa, ma), (xb, mb) = nums[a], nums[b]
                both = ma & mb
                if both.sum() >= 32 and both.sum() >= 0.9 * max(ma.sum(), mb.sum()):
                    ra, rb = ranks(xa[both]), ranks(xb[both])
                    if ra.std() > 0 and rb.std() > 0 and abs(np.corrcoef(ra, rb)[0, 1]) >= RED_RHO:
                        red[b] = a
    # resolve chains (b -> a -> c): every redundant attribute points at a keeper
    for b in list(red):
        k, hops = red[b], 0
        while k in red and hops < len(red):
            k, hops = red[k], hops + 1
        if k == b:
            del red[b]
        else:
            red[b] = k
    return red


# ==================================================================== roles
def _card0(st: Mapping[str, Any]) -> int:
    """Distinct values of an evaluated attribute at level 0 (its raw values)."""
    c = st.get("card")
    if isinstance(c, Mapping):
        v = c.get(0, c.get("0"))
        return int(v) if v is not None else 10 ** 9
    try:
        return int(c)
    except (TypeError, ValueError):
        return 10 ** 9


def assign_roles(stats: Mapping[str, Mapping[str, Any]], prev: Mapping[str, Any],
                 redundant: Mapping[str, str], ip_info: Mapping[int, float], t: float,
                 kinds_of: Callable[[str], Iterable[int]]) -> Dict[str, Any]:
    """Roles with hysteresis (§6.4). prev = the previous model.attrsel (or {}).
    Returns {roles, levels, targets_sys, split_cands, redundant, who_mode, low}."""
    roles: Dict[str, str] = dict(prev.get("roles") or {})
    low: Dict[str, int] = dict(prev.get("low") or {})
    levels: Dict[str, List[int]] = dict(prev.get("levels") or {})
    dropped_at: Dict[str, float] = dict(prev.get("dropped_at") or {})
    ustat: Dict[str, Dict[str, Any]] = {a: dict(v) for a, v in (prev.get("ustat") or {}).items()}
    for a, st in stats.items():
        ustat[a] = {"U_t": max(float(st["U_t"]), float(st.get("U_tc", st["U_t"]))),
                    "CR_p": float(st.get("CR_p", 0.0)), "n_p": int(st.get("n_p", 0)),
                    "H_p": float(st.get("H_p", st["H"])),
                    "U_s": float(st["U_s_max"]),
                    "cov": max(float(st["cov"]), float(st.get("cov_rows", st["cov"]))),
                    "H": float(st["H"]), "level": int(st["level"]), "CR": float(st["CR"])}
        old = roles.get(a, "probe")
        H, cov = st["H"], st["cov"]
        U_t, U_s = max(st["U_t"], st.get("U_tc", st["U_t"])), st["U_s_max"]
        best_lv = list(st["best_levels"])
        new = None
        # entry / exit band: an attribute enters `invariant` at H <= INV_H and
        # leaves it only above 2 INV_H or below INV_COV_EXIT coverage (a value
        # near the edge flapped between invariant and dropped every hour)
        if (H <= INV_H and cov >= INV_COV) or (old == "invariant" and H <= 2 * INV_H
                                               and cov >= INV_COV_EXIT):
            new = "invariant"
        elif a in redundant:
            new = "redundant"
        else:
            is_split = bool(best_lv) and U_s >= (U_LO if old == "split" else U_HI)
            U_te = max(U_t, float(st.get("U_tc", U_t)))
            cov_e = max(cov, float(st.get("cov_rows", cov)))
            # informative for the system (bits / event) or informative where it is
            # present (a rare action's attributes; P05 re-ranks per node)
            local = (float(st.get("CR_p", 0.0)) >= (CR_LOCAL_LO if old == "target" else CR_LOCAL)
                     and int(st.get("n_p", 0)) >= N_LOCAL)
            is_target = (targetable(a) and st["S"] >= TARGET_STAB
                         and ((U_te >= (U_LO if old == "target" else U_HI) and cov_e >= TARGET_COV) or local))
            # a small closed value set where present (an approval's opinion 同意 /
            # 退回 / ...) is a constraint to state and check even when no context
            # predicts WHICH value comes: it was `dropped` (no predictive gain),
            # so no pattern ever stated or enforced its closed set (measured on
            # pack O: FIN approval opinion never fitted)
            # (local fields only: an attribute present on most of the system's
            # events that no context predicts is noise, not an action's field)
            closed_small = (targetable(a) and st.get("S", 0.0) >= TARGET_STAB
                            and float(st.get("cov", 1.0)) <= CLOSED_COV_MAX
                            and 2 <= _card0(st) <= CLOSED_CARD
                            and float(st.get("H_p", 0.0)) >= INV_H
                            and int(st.get("n_p", 0)) >= N_LOCAL
                            and st.get("kind") in ("categorical", "text", "cat"))
            if is_split:
                new = "split"
            elif is_target or closed_small:
                new = "target"
            elif st["distinct0"] >= SHAPE_DISTINCT and st["CR0"] < SHAPE_CR:
                new = "shape"
            else:
                new = "dropped"
        # hysteresis: a kept role demotes only after DEMOTE_RUNS consecutive low runs
        # (the kept role's own test already uses the low threshold u_lo, so a
        # role changes here only when its utility fell below u_lo, or on an upgrade)
        demotion = ((old in ("split", "target") and new in ("dropped", "shape", "redundant"))
                    or (old == "split" and new == "target")
                    or (old == "redundant" and new != "invariant" and a not in redundant))
        if demotion:
            low[a] = low.get(a, 0) + 1
            if low[a] < DEMOTE_RUNS:
                new = old
            else:
                low[a] = 0
        else:
            low[a] = 0
        if new == "split":
            levels[a] = best_lv or levels.get(a, [st["level"]])
        elif new in ("target", "shape"):
            levels[a] = [int(st["level"])]
        if new == "dropped" and old != "dropped":
            dropped_at[a] = float(t)
        roles[a] = new
    # "IP is not a feature": remove net.src from split candidates when it carries
    # no information at any level
    who_mode = prev.get("who_mode") or "ip"
    if ip_info:
        best_lv = max(ip_info, key=ip_info.get)
        best = ip_info[best_lv]
        if best < IP_INFO_MIN:
            who_mode = "none"
            if roles.get("net.src") == "split":
                roles["net.src"] = "dropped"
        else:
            who_mode = {0: "ip", 1: "prefix", 3: "grp", 4: "reg"}.get(best_lv, "ip")
    # system lists per kind
    targets_sys: Dict[int, List[str]] = {}
    split_cands: Dict[int, List[Tuple[str, int]]] = {}
    for kind in (KIND_TXN, KIND_WIN):
        def _ok(a: str) -> bool:
            u = ustat.get(a, {})
            if u.get("H_p", 1.0) < INV_H:
                return False                   # constant where present: a node invariant
            return ((u.get("U_t", 0.0) >= U_LO and u.get("cov", 0.0) >= TARGET_COV)
                    or (u.get("CR_p", 0.0) >= CR_LOCAL_LO and u.get("n_p", 0) >= N_LOCAL))
        tg = [a for a, r in roles.items() if r in ("target", "split", "shape") and targetable(a)
              and kind in set(kinds_of(a)) and _ok(a)]
        tg.sort(key=lambda a: (-ustat.get(a, {}).get("U_t", 0.0), -ustat.get(a, {}).get("CR_p", 0.0)))
        targets_sys[kind] = tg[:M_SYS]
        sc = []
        for a, r in roles.items():
            if r != "split" or kind not in set(kinds_of(a)):
                continue
            if who_mode == "none" and a in WHO_ATTRS:
                continue
            for lv in levels.get(a, []):
                sc.append((a, int(lv), ustat.get(a, {}).get("U_s", 0.0)))
        sc.sort(key=lambda x: -x[2])
        split_cands[kind] = [(a, lv) for a, lv, _ in sc]
    return {"roles": roles, "levels": levels, "targets_sys": targets_sys,
            "split_cands": split_cands, "redundant": dict(redundant), "who_mode": who_mode,
            "low": low, "dropped_at": dropped_at, "ustat": ustat}


def names_to_evaluate(registry_names: Sequence[str], roles: Mapping[str, str],
                      dropped_at: Mapping[str, float], t: float, run_index: int,
                      a_probe: int = A_PROBE) -> List[str]:
    """Every attribute holding a role other than `dropped`, dropped ones due
    for their 7-day re-probe, plus a rotating crc32 slice of A_probe others."""
    kept = [a for a in registry_names if roles.get(a, "probe") not in ("dropped", "probe")]
    reprobe = [a for a in registry_names if roles.get(a) == "dropped"
               and t - float(dropped_at.get(a, t)) >= REPROBE_S]
    rest = [a for a in registry_names if roles.get(a, "probe") in ("probe", "dropped")
            and a not in set(reprobe)]
    if len(rest) <= a_probe:
        slice_ = rest
    else:
        K = max(1, math.ceil(len(rest) / a_probe))
        slice_ = [a for a in rest if (zlib.crc32(a.encode("utf-8")) + run_index) % K == 0][:a_probe]
    seen = set()
    out = []
    for a in kept + reprobe + slice_:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


# ============================================================ node overrides
def summary_entropy(s: Any, t: float, hier: Any = None, attr: Optional[str] = None) -> Tuple[float, float]:
    """(entropy bits, total H_m mass) of a pnode target summary."""
    ss = getattr(s, "ss", None) or getattr(s, "values", None) or getattr(s, "shapes", None) \
        or getattr(s, "tpl", None)
    if ss is not None and hasattr(ss, "distribution"):
        keys, sh, other = ss.distribution(t)
        p = np.r_[np.asarray(sh, dtype=float), max(0.0, other)]
        p = p[p > 0]
        h = float(-(p * np.log2(p)).sum()) if p.size else 0.0
        return h, float(ss.total(t))
    td = getattr(s, "td", None)
    if td is not None:
        tot = td.total(t)
        if tot <= 0 or td.n_centroids() < 2:
            return 0.0, float(tot)
        qs = np.asarray([td.quantile(q) for q in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)])
        span = float(td.vmax - td.vmin) if hasattr(td, "vmax") else 0.0
        if span <= 1e-12:
            return 0.0, float(tot)
        # entropy of the octile spacing relative to the range: 0 for a constant
        return 3.0 * min(1.0, float(np.ptp(qs)) / span + 0.05), float(tot)
    return 0.0, 0.0


def node_targets_from_probe(tree: Any, probe: StratifiedProbe, t: float, hier: Any,
                            targets_sys: Sequence[str], gone: Iterable[str] = (), m_t: int = M_T,
                            n_min: int = 32, local_pool: Sequence[str] = (),
                            local_cov: float = LOCAL_TARGET_COV) -> Dict[int, List[str]]:
    """{nid: [a...]}: the probe rows are routed through the tree (the same
    routing P03 / P04 use); at every node holding >= n_min probe rows the system
    targets are re-ranked by node-local H(a) x cov(a) on those rows (mass-
    weighted), constants (H < INV_H: node invariants, detected by P04) and
    absent attributes free their slot, top m_t (§6.4 'per node'). Nodes with
    fewer rows inherit from their nearest ancestor in P04.

    `local_pool`: attributes outside the system list (split / target / shape
    roles) that are node-local: present on a small share of the system's
    events (a login form's username, a report's key set), so the system list
    (ranked by system-wide coverage) never holds them. They compete for a
    node's slots when they cover >= local_cov of the node's rows (§6.5.2
    item 3: content attributes are split candidates and targets). Measured on
    pack O: without it the OA login node's targets were client / size
    attributes only, so no who split could pay for itself by predicting the
    usernames or the login minute."""
    rows, w, _ = probe.rows(t)
    if not rows or not targets_sys:
        return {}
    sys_set = set(targets_sys)
    pool = list(targets_sys) + [a for a in local_pool if a not in sys_set]
    sl = probe._schema_list
    by_node: Dict[int, List[int]] = {}
    gone_s = set(gone)
    for i, (sid, vals) in enumerate(rows):
        ix = sl[sid][1]

        def get(nm: str, ix: Dict[str, int] = ix, vals: tuple = vals) -> Any:
            j = ix.get(nm)
            return ABSENT if j is None else vals[j]
        try:
            path = tree.route(get, hier, gone_s, t)
        except Exception:
            continue
        for nid in path:
            by_node.setdefault(int(nid), []).append(i)
    cols: Dict[str, List[Any]] = {a: probe.column(rows, a) for a in pool}
    out: Dict[int, List[str]] = {}
    default = list(targets_sys[:m_t])
    for nid, idx in by_node.items():
        if len(idx) < n_min:
            continue
        ii = np.asarray(idx)
        wn = w[ii]
        tot = float(wn.sum()) or 1.0
        scored = []
        for a in pool:
            col = cols[a]
            vals = [col[i] for i in idx]
            pres = np.asarray([v is not ABSENT for v in vals])
            if not pres.any():
                continue
            l, cc, k = level_codes(hier, a, [v for v, p in zip(vals, pres) if p], text_values=True)
            h, _ = w_plugin(cc, wn[pres])
            if h < INV_H:
                continue
            cov = float(wn[pres].sum()) / tot
            if a not in sys_set and cov < local_cov:
                continue
            scored.append((h * cov, a))
        scored.sort(key=lambda x: (-x[0], x[1]))
        keep = [a for _, a in scored[:m_t]]
        if keep and keep != default:
            out[nid] = keep
    return out


def node_overrides(tree: Any, targets_sys: Sequence[str], t: float, m_t: int = M_T,
                   n_min: float = 30.0) -> Dict[int, List[str]]:
    """{nid: [a...]}: per node with >= n_min evidence, its tracked targets
    re-ranked by H_node x cov_node; constants (H < INV_H) free their slot,
    which the next untracked system targets fill (§6.4)."""
    out: Dict[int, List[str]] = {}
    for nid, nd in tree.nodes.items():
        if nd.n_m(t) < n_min or not nd.targets:
            continue
        mass = nd.mass_at(t) or 1.0
        scored = []
        for a, s in nd.targets.items():
            h, m = summary_entropy(s, t)
            if h < INV_H:
                continue
            scored.append((h * min(1.0, m / mass), a))
        scored.sort(reverse=True)
        keep = [a for _, a in scored[:m_t]]
        for a in targets_sys:
            if len(keep) >= m_t:
                break
            if a not in nd.targets and a not in keep and a not in nd.inv:
                keep.append(a)
        default = list(targets_sys[:m_t])
        if keep != default:
            out[int(nid)] = keep
    return out


# ============================================================== value groups
def value_groups(probe: StratifiedProbe, t: float, hier: Any, a: str, targets: Sequence[str],
                 thr: float = VG_JSD, max_values: int = LEVEL_CARD) -> Dict[Hashable, int]:
    """Categorical level-1 value groups (§5.4.6): agglomerative merge of values
    whose conditional target distributions are within JSD thr bits (averaged
    over targets). Returns {value: group id} for values in groups of >= 2."""
    rows, w, _ = probe.rows(t)
    if not rows:
        return {}
    col = probe.column(rows, a)
    vc, vk = codes_of(col)
    if vk < 3 or vk > max_values:
        return {}
    inv = {}
    for v, c in zip(col, vc.tolist()):
        inv.setdefault(c, v)
    dists = []
    for b in targets:
        if same_source(a, b):
            continue
        l, bc, bk = level_codes(hier, b, probe.column(rows, b), raw=probe.codes(rows, b))
        tab = np.zeros((vk, bk))
        np.add.at(tab, (vc, bc), w)
        dists.append(tab)
    if not dists:
        return {}
    clusters = [[i] for i in range(vk)]
    tabs = [[d[i].copy() for d in dists] for i in range(vk)]

    def dist(x: List[np.ndarray], y: List[np.ndarray]) -> float:
        return float(np.mean([pmdl.jsd(p, q) for p, q in zip(x, y)]))
    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = dist(tabs[i], tabs[j])
                if best is None or d < best[0]:
                    best = (d, i, j)
        if best is None or best[0] > thr:
            break
        _, i, j = best
        clusters[i] += clusters[j]
        tabs[i] = [x + y for x, y in zip(tabs[i], tabs[j])]
        del clusters[j], tabs[j]
    out: Dict[Hashable, int] = {}
    gid = 0
    for cl in clusters:
        if len(cl) < 2:
            continue
        for c in cl:
            out[_hashable(inv[c])] = gid
        gid += 1
    return out


# ================================================================ accessor
def bootstrap_selection(registry: Any, kind: int, present: Optional[Iterable[str]] = None
                        ) -> Dict[str, Any]:
    """Selection before P05's first run: targets = registered attributes of the
    kind with coverage >= 0.05 ranked by entropy x coverage (bookkeeping excluded),
    split candidates = the bootstrap seeds of §6.4 that exist (seeds only; P05
    replaces them within the hour)."""
    names = list(present) if present is not None else []
    recs = {}
    if registry is not None and hasattr(registry, "records"):
        recs = registry.records
        if not names:
            names = [n for n, r in recs.items() if kind in r.kinds]
    tg = []
    for a in names:
        if not targetable(a):
            continue
        r = recs.get(a)
        cov = registry.coverage(a) if r is not None else 0.5
        if r is not None and (cov < TARGET_COV or r.type == "unknown"):
            continue
        ent = float(getattr(r, "entropy", 1.0) or 0.0) if r is not None else 1.0
        if r is not None and r.type == "text" and r.card_estimate() > 1000 and ent > 8:
            continue
        tg.append((ent * cov, a))
    tg.sort(reverse=True)
    seeds = []
    have = set(names)
    for a, l in (("http.route", 0), ("net.src", 1), ("net.src", 0), ("ctx.tod_min", 1),
                 ("ev.ch", 0), ("net.dst", 0)):
        if a in have:
            seeds.append((a, l))
    return {"targets_sys": {kind: [a for _, a in tg[:M_SYS]]}, "split_cands": {kind: seeds},
            "roles": {}, "node_overrides": {}, "who_mode": "ip", "bootstrap": True}


def selection_for(store: Any, key: str, kind: int, registry: Any = None,
                  present: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """The current selection of a tree for an event kind: model.attrsel when P05
    has published targets for the kind, else the bootstrap."""
    from ....models.schema import SYSTEM_ENTITY
    sel = store.get_model(key, SYSTEM_ENTITY, ATTRSEL)
    if isinstance(sel, Mapping) and (sel.get("targets_sys") or {}).get(kind):
        return dict(sel)
    boot = bootstrap_selection(registry, kind, present)
    if isinstance(sel, Mapping):
        out = dict(sel)
        out.setdefault("targets_sys", {})
        out["targets_sys"] = dict(out["targets_sys"])
        out["targets_sys"][kind] = boot["targets_sys"][kind]
        sc = dict(out.get("split_cands") or {})
        if not sc.get(kind):
            sc[kind] = boot["split_cands"][kind]
        out["split_cands"] = sc
        return out
    return boot
