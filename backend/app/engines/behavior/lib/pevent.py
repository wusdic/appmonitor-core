"""Behaviour events: the open-attribute event batch of the progressive core
(docs/lib3/progressive.md §5.2, §6.2.2, §13.2).

STATUS: implemented (W-P0). One `EventBatch` per (system, kind, tick) is written
by P00 (`evt.batch`, kind txn) and P01 (`evt.win`, kind win) through the store's
batch series; P01's `evt.ctx` and P03's `pat.assign` are *aligned* batches
(same rows, own columns, `aligned()`).

Batch layout (columnar, sparse):
    ts[n] float64, ip[n] int32 -> ips[], w[n] float32 row mass before HT
    (aggregation share x sensor sample rate), pi[n] float32 learning inclusion
    probability, learn[n] bool, flags[n] uint8 (bit0 approx, bit1 body_trunc),
    rid[n] int32 original row id (stable across compaction),
    cols{name: Col(rows int32 sorted, vals float64 | object)}, meta{}.
A learned row stands for mass w / pi; its evidence (<= 1) is computed by the
learner (psketch.BurstEvidence), never from mass (PPC-9).

Absence is a value: `ABSENT` (⊥) is returned for an attribute a row does not
carry, and every hierarchy maps it to itself (phier).

Also here: attribute-name normalisation, JSON flattening, learning-sample
selection (threshold sampling, Duffield-Lund-Thorup), priority sampling, the
store series names and the `progressive` configuration defaults.
"""
from __future__ import annotations

import copy
import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------- constants
KIND_TXN = 0
KIND_WIN = 1
KIND_NAMES = {KIND_TXN: "txn", KIND_WIN: "win"}
FLAG_APPROX = 1
FLAG_TRUNC = 2
ABSENT = "⊥"                       # ⊥ : reserved "attribute absent" value
NAME_MAX = 96
SET_MAX = 16                            # scalar elements kept per JSON array
JSON_LEAVES = 64                        # leaves flattened per JSON body
JSON_DEPTH = 4

# store names (§5.6)
EVT_BATCH = "evt.batch"
EVT_WIN = "evt.win"
EVT_CTX = "evt.ctx"
PAT_ASSIGN = "pat.assign"
PAT_RATE = "pat.rate"
BATCH_SERIES = (EVT_BATCH, EVT_WIN, EVT_CTX, PAT_ASSIGN, PAT_RATE)

# configuration defaults (§13.2); core/engine.DEFAULT_CONFIG may carry the same
# key (owned by W-P9); readers always go through pconfig(), which deep-merges
PROGRESSIVE_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "value_policy": {
        "rules": [],                    # [(glob, 'clear' | 'hmac' | 'shape')], first match wins
        "secret_globs": ["body.kv.*pass*", "body.kv.*pwd*", "*token*", "*secret*", "*otp*",
                         "*captcha*", "hdr.authorization", "hdr.cookie", "hdr.set-cookie",
                         "q.kv.*pass*", "q.kv.*pwd*"],
        "v_len": 64,
        "random_bits": 3.5,             # bits/char at length >= random_len -> shape
        "random_len": 16,
        "hmac_key": "appmon-default-deployment-key",
    },
    "trusted_proxies": [],
    "client_ip_headers": ["x-forwarded-for", "forwarded", "x-real-ip"],
    "session_cookies": ["JSESSIONID", "PHPSESSID", "ASP.NET_SessionId", "sid", "session"],
    "type_hints": {"code": ["status", "code", "port", "qtype", "rcode", "method"]},
    "system_families": [],
    "groups_as_classes": False,
    "lib4_inputs": False,
    "budget": {"pcore_cpu_share": 0.25, "lib3_cpu_share": None, "mem_mb_total": 2048,
               "per_system": {}},
    "defaults": {
        "e_rate": 10.0,                 # learned events / s per tree
        "r_tick": 1_000_000,            # rows per tick (P00 subsamples ev_sample rows beyond)
        "w_max": 512,                   # window events per tree and grain
        "a_win": 96,                    # metric attributes per window event
        "k_body": 64,                   # body keys per event
        "body_cap": 4096,
        "ev_sample_max": 64,
        "d_min_s": 600.0,               # learning delay D = max(4 ticks, d_min_s)
        "s_sess": 65536,
        "session_gap_s": 1800.0,
    },
}


def _deep_merge(base: Dict[str, Any], over: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


_PCFG_CACHE: Dict[int, Tuple[Any, Dict[str, Any]]] = {}


def pconfig(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """ctx.config['progressive'] deep-merged over PROGRESSIVE_DEFAULTS (cached
    per config object identity and content)."""
    over = (config or {}).get("progressive") or {}
    key = id(over)
    hit = _PCFG_CACHE.get(key)
    if hit is not None and hit[0] is over and hit[1].get("_src") == repr(over):
        return hit[1]
    merged = _deep_merge(PROGRESSIVE_DEFAULTS, over)
    merged["_src"] = repr(over)
    if len(_PCFG_CACHE) > 64:
        _PCFG_CACHE.clear()
    _PCFG_CACHE[key] = (over, merged)
    return merged


def enabled(config: Optional[Mapping[str, Any]]) -> bool:
    return bool(((config or {}).get("progressive") or {}).get("enabled", False))


def learn_delay_s(dt: float, config: Optional[Mapping[str, Any]] = None) -> float:
    """D = max(4 ticks, d_min_s) (the lib-3 learner rule, §6.9.3)."""
    d_min = float(pconfig(config)["defaults"]["d_min_s"])
    return max(4.0 * float(dt), d_min)


# ------------------------------------------------------------- attribute names
_NAME_BAD = re.compile(r"[^a-z0-9_.\-\[\]:/@*]+")
_INDEX = re.compile(r"\[\d+\]")


def attr_name(*parts: Any) -> str:
    """Normalised attribute name: lower-case parts joined with '.', illegal
    characters replaced by '_', at most NAME_MAX characters (longer names are
    cut and suffixed with '~' + 8 hex of blake2b of the full name)."""
    raw = ".".join(str(p) for p in parts if p is not None and str(p) != "")
    s = _NAME_BAD.sub("_", raw.strip().lower())
    s = re.sub(r"\.{2,}", ".", s).strip(".")
    if len(s) > NAME_MAX:
        h = hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=4).hexdigest()
        s = s[:NAME_MAX - 9] + "~" + h
    return s


def template_key(key: str) -> str:
    """'items[3].name' -> 'items[].name' (array indices removed)."""
    return _INDEX.sub("[]", key)


def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def flatten(prefix: str, obj: Any, out: Dict[str, Any], max_leaves: int = JSON_LEAVES,
            max_depth: int = JSON_DEPTH) -> int:
    """Flatten nested dict / list structure into out[prefix.key...]. Arrays of
    scalars become '<key>[]' = frozenset of up to SET_MAX elements (as str)
    plus '<key>[].n' = length; arrays of objects contribute '<key>[].<sub>'
    = frozenset of the sub-values across elements. Returns leaves written."""
    budget = [max_leaves]

    def put(name: str, v: Any) -> None:
        if budget[0] <= 0:
            return
        budget[0] -= 1
        out[attr_name(name)] = v

    def rec(path: str, v: Any, depth: int) -> None:
        if budget[0] <= 0:
            return
        if isinstance(v, Mapping):
            if depth >= max_depth:
                put(path + ".{}", len(v))
                return
            for k in list(v.keys())[:max_leaves]:
                rec(f"{path}.{template_key(str(k))}", v[k], depth + 1)
        elif isinstance(v, (list, tuple)):
            scal = [x for x in v if _is_scalar(x)]
            objs = [x for x in v if isinstance(x, Mapping)]
            put(path + "[].n", len(v))
            if scal:
                put(path + "[]", frozenset(str(x) for x in scal[:SET_MAX]))
            if objs and depth < max_depth:
                sub: Dict[str, set] = {}
                for o in objs[:SET_MAX]:
                    for k, x in o.items():
                        if _is_scalar(x):
                            sub.setdefault(template_key(str(k)), set()).add(str(x))
                for k, xs in sub.items():
                    put(f"{path}[].{k}", frozenset(list(xs)[:SET_MAX]))
        elif isinstance(v, bool):
            put(path, int(v))
        elif isinstance(v, (int, float)):
            put(path, v)
        elif v is not None:
            put(path, str(v))

    rec(prefix, obj, 0)
    return max_leaves - budget[0]


# ------------------------------------------------------------------- batches
@dataclass(slots=True)
class Col:
    rows: np.ndarray                    # int32[k], sorted row indices where present
    vals: np.ndarray                    # float64[k] or object[k]

    @property
    def numeric(self) -> bool:
        return self.vals.dtype != object

    def nbytes(self) -> int:
        b = self.rows.nbytes + self.vals.nbytes
        if not self.numeric:
            b += 16 * len(self.vals)
        return int(b)


@dataclass
class EventBatch:
    system: str
    kind: int
    t0: float
    t1: float
    n: int
    ts: np.ndarray
    ip: np.ndarray
    ips: List[str]
    w: np.ndarray
    pi: np.ndarray
    learn: np.ndarray
    flags: np.ndarray
    cols: Dict[str, Col]
    rid: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    _pos: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _lst: Dict[str, List[Any]] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.rid is None:
            self.rid = np.arange(self.n, dtype=np.int32)

    # ------------------------------------------------------------- reading
    def names(self) -> List[str]:
        return list(self.cols.keys())

    def has(self, name: str) -> bool:
        return name in self.cols

    def _position(self, name: str) -> Optional[np.ndarray]:
        pos = self._pos.get(name)
        if pos is None:
            c = self.cols.get(name)
            if c is None:
                return None
            pos = np.full(self.n, -1, dtype=np.int32)
            pos[c.rows] = np.arange(len(c.rows), dtype=np.int32)
            self._pos[name] = pos
        return pos

    def _values(self, name: str) -> Optional[List[Any]]:
        """Row-aligned Python list of the column (ABSENT where a row lacks
        it; numpy floats as Python floats), built once per batch and name:
        get() is called per row by every engine (P04, P05, P03: ~6 M calls
        on 3 days of pack O), where numpy scalar indexing dominated."""
        lst = self._lst.get(name)
        if lst is None:
            c = self.cols.get(name)
            if c is None:
                return None
            lst = [ABSENT] * self.n
            if c.numeric:
                vals = c.vals.tolist()
            else:
                vals = [float(v) if isinstance(v, np.floating) else v for v in c.vals.tolist()]
            for r, v in zip(c.rows.tolist(), vals):
                lst[r] = v
            self._lst[name] = lst
        return lst

    def get(self, name: str, row: int, default: Any = ABSENT) -> Any:
        """Value of attribute `name` at row (ABSENT when the row lacks it). O(1)."""
        lst = self._lst.get(name)
        if lst is None:
            lst = self._values(name)
            if lst is None:
                return default
        v = lst[row]
        return default if v is ABSENT else v

    def dense(self, name: str, fill: Any = ABSENT) -> np.ndarray:
        """Row-aligned array of the column (object dtype when fill is not a
        number or the column is not numeric)."""
        c = self.cols.get(name)
        numeric_fill = isinstance(fill, (int, float)) and not isinstance(fill, bool)
        if c is not None and c.numeric and numeric_fill:
            out = np.full(self.n, float(fill))
        else:
            out = np.empty(self.n, dtype=object)
            out[:] = [fill] * self.n if self.n else []
        if c is not None:
            out[c.rows] = c.vals
        return out

    def row(self, i: int, names: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        """{name: value} of the attributes row i carries (restricted to names)."""
        out: Dict[str, Any] = {}
        for nm in (names if names is not None else self.cols.keys()):
            v = self.get(nm, i)
            if v is not ABSENT:
                out[nm] = v
        return out

    def ip_of(self, i: int) -> str:
        return self.ips[int(self.ip[i])]

    def mass(self) -> np.ndarray:
        """HT mass w / pi per row (0 for rows not learned)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.where(self.learn & (self.pi > 0), self.w / np.maximum(self.pi, 1e-12), 0.0)
        return m.astype(np.float64)

    def learned_rows(self) -> np.ndarray:
        return np.flatnonzero(self.learn)

    def event_id(self, i: int) -> Tuple[str, int, float, int]:
        """Global event id (system, kind, t1, original row)."""
        return (self.system, self.kind, self.t1, int(self.rid[i]))

    # ------------------------------------------------------------- deriving
    def select(self, rows: Sequence[int], names: Optional[Iterable[str]] = None) -> "EventBatch":
        """Sub-batch of `rows` (sorted), restricted to `names` (all if None).
        Row ids (`rid`) are kept, so event ids stay valid."""
        r = np.asarray(sorted(int(x) for x in rows), dtype=np.int64)
        remap = np.full(self.n, -1, dtype=np.int64)
        remap[r] = np.arange(r.size)
        keep = set(self.cols) if names is None else set(names) & set(self.cols)
        cols: Dict[str, Col] = {}
        for nm in keep:
            c = self.cols[nm]
            m = remap[c.rows] >= 0
            if m.any():
                cols[nm] = Col(remap[c.rows[m]].astype(np.int32), c.vals[m].copy())
        used = np.unique(self.ip[r]) if r.size else np.zeros(0, dtype=np.int64)
        ipmap = np.full(len(self.ips), -1, dtype=np.int64)
        ipmap[used] = np.arange(used.size)
        return EventBatch(self.system, self.kind, self.t0, self.t1, int(r.size),
                          self.ts[r].copy(), ipmap[self.ip[r]].astype(np.int32),
                          [self.ips[int(k)] for k in used], self.w[r].copy(), self.pi[r].copy(),
                          self.learn[r].copy(), self.flags[r].copy(), cols,
                          self.rid[r].copy(), dict(self.meta))

    def aligned(self, cols: Dict[str, Col], meta: Optional[Dict[str, Any]] = None) -> "EventBatch":
        """A batch row-aligned with this one (shares the row arrays) carrying
        its own columns (evt.ctx, pat.assign)."""
        return EventBatch(self.system, self.kind, self.t0, self.t1, self.n, self.ts, self.ip,
                          self.ips, self.w, self.pi, self.learn, self.flags, cols, self.rid,
                          dict(meta or {}))

    def nbytes(self) -> int:
        b = sum(a.nbytes for a in (self.ts, self.ip, self.w, self.pi, self.learn, self.flags, self.rid))
        b += sum(c.nbytes() + 80 for c in self.cols.values())
        b += sum(len(s) + 50 for s in self.ips)
        return int(b + 400)


class BatchBuilder:
    """Accumulates rows (dict attributes) and builds a columnar EventBatch."""

    def __init__(self, system: str, kind: int = KIND_TXN) -> None:
        self.system = system
        self.kind = int(kind)
        self._ts: List[float] = []
        self._ip: List[int] = []
        self._w: List[float] = []
        self._flags: List[int] = []
        self._ips: Dict[str, int] = {}
        self._cols: Dict[str, Tuple[List[int], List[Any]]] = {}
        self.meta: Dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self._ts)

    def add(self, ts: float, ip: str, attrs: Mapping[str, Any], w: float = 1.0, flags: int = 0) -> int:
        i = len(self._ts)
        self._ts.append(float(ts))
        k = self._ips.get(ip)
        if k is None:
            k = self._ips[ip] = len(self._ips)
        self._ip.append(k)
        self._w.append(float(w))
        self._flags.append(int(flags))
        for nm, v in attrs.items():
            if v is None or v is ABSENT:
                continue
            c = self._cols.get(nm)
            if c is None:
                c = self._cols[nm] = ([], [])
            c[0].append(i)
            c[1].append(v)
        return i

    def build(self, t0: float, t1: float) -> EventBatch:
        n = len(self._ts)
        cols: Dict[str, Col] = {}
        for nm, (rows, vals) in self._cols.items():
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
                arr = np.asarray(vals, dtype=np.float64)
            else:
                arr = np.empty(len(vals), dtype=object)
                arr[:] = vals
            cols[nm] = Col(np.asarray(rows, dtype=np.int32), arr)
        ips = [None] * len(self._ips)
        for s, k in self._ips.items():
            ips[k] = s
        return EventBatch(self.system, self.kind, float(t0), float(t1), n,
                          np.asarray(self._ts, dtype=np.float64),
                          np.asarray(self._ip, dtype=np.int32), ips,
                          np.asarray(self._w, dtype=np.float32),
                          np.ones(n, dtype=np.float32), np.ones(n, dtype=bool),
                          np.asarray(self._flags, dtype=np.uint8), cols, meta=dict(self.meta))


def cols_from_rows(n: int, rows_attrs: Sequence[Mapping[str, Any]]) -> Dict[str, Col]:
    """Columns for an aligned batch from one attribute dict per row."""
    b = BatchBuilder("", 0)
    for i, a in enumerate(rows_attrs):
        b.add(0.0, "", a)
    return b.build(0.0, 0.0).cols


def compact_batch(batch: EventBatch, keep_cols: Optional[Iterable[str]] = None,
                  extra_rows: Optional[Iterable[int]] = None) -> EventBatch:
    """The learned rows (plus extra_rows, e.g. held events) restricted to
    keep_cols (§5.2.3): what a batch is retained as after the tick."""
    rows = set(np.flatnonzero(batch.learn).tolist())
    if extra_rows is not None:
        rows.update(int(r) for r in extra_rows)
    return batch.select(sorted(rows), keep_cols)


# ------------------------------------------------------------------ sampling
def threshold_tau(counts: Sequence[float], budget: float) -> float:
    """Water-filling threshold: tau with sum_k min(c_k, tau) = budget
    (inf when sum c_k <= budget). Duffield, Lund & Thorup 2005."""
    c = np.sort(np.asarray([float(x) for x in counts if x > 0], dtype=np.float64))
    if c.size == 0 or c.sum() <= budget:
        return math.inf
    if budget <= 0:
        return 0.0
    k = c.size
    cum = 0.0
    for i in range(k):
        rest = k - i
        tau = (budget - cum) / rest
        if tau <= c[i]:
            return float(tau)
        cum += c[i]
    return math.inf


def select_learning_sample(batch: EventBatch, strata: Sequence[Hashable], budget: float,
                           u: Sequence[float]) -> float:
    """Stratified threshold sampling (§6.2.2): stratum k with c_k rows keeps
    each row with pi = min(1, tau / c_k); sets batch.pi and batch.learn from
    the uniforms u (one per row, e.g. seeded_uniform). Returns tau. Rare
    strata are always learned in full; HT mass w / pi keeps totals unbiased."""
    n = batch.n
    if n == 0:
        return math.inf
    keys = list(strata)
    counts: Dict[Hashable, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    tau = threshold_tau(list(counts.values()), budget)
    if math.isinf(tau):
        batch.pi[:] = 1.0
        batch.learn[:] = True
        return tau
    pi = np.asarray([min(1.0, tau / counts[k]) for k in keys], dtype=np.float32)
    uu = np.asarray(u, dtype=np.float64)
    batch.pi[:] = pi
    batch.learn[:] = uu < pi
    return tau


def priority_sample(weights: Sequence[float], k: int, u: Sequence[float]
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Priority sampling (Duffield, Lund & Thorup, J. ACM 2007): priority
    q_i = w_i / u_i, keep the k largest; with tau the (k+1)-th largest
    priority, each kept item's unbiased weight is max(w_i, tau). Returns
    (kept indices sorted, their HT weights). All kept at their own weight
    when there are <= k items."""
    w = np.asarray(weights, dtype=np.float64)
    n = w.size
    if n <= k:
        return np.arange(n), w.copy()
    uu = np.clip(np.asarray(u, dtype=np.float64), 1e-300, 1.0)
    q = w / uu
    order = np.argsort(-q, kind="stable")
    kept = np.sort(order[:k])
    tau = float(q[order[k]])
    return kept, np.maximum(w[kept], tau)


def bootstrap_stratum(batch: EventBatch, i: int) -> Tuple[str, str]:
    """Bootstrap stratum key (ev.ch, route | sni | qname | dst) used before a
    tree's root has split (§6.2.2); ev.ch alone when none is present."""
    ch = batch.get("ev.ch", i, "")
    for nm in ("http.route", "tls.sni", "dns.qname", "net.dst"):
        v = batch.get(nm, i)
        if v is not ABSENT:
            return (str(ch), str(v))
    return (str(ch), "")
