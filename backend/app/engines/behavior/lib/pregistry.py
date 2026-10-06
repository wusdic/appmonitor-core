"""Dynamic attribute registry: schema inference, per-attribute statistics,
hierarchy models, schema-change detection (docs/lib3/progressive.md §5.3, §5.4, §6.3).

STATUS: implemented (W-P0 foundation for P02). No code lists which attributes
exist (PPC-4): any name an event carries is registered on first sight, typed
from decayed evidence, summarised with bounded sketches, and given a
generalisation hierarchy model (lib/phier reads `record.type`, `record.hier`,
`record.policy`, `record.card_estimate()`).

One AttrRegistry per tree (system or family), stored as the model
`model.attr@(tree key, '__system__')` (lib/m_ptree accessors). The registry
object itself is the model (the store keeps objects by reference);
`to_dict()` gives a checkpointable form.

Type inference (§5.3, evaluated on H_m-decayed evidence, in this order):
  1 ip        >= 99 % of values parse as IP addresses
  2 time      name ends with '_ts' / '.ts' and >= 99 % finite floats in [1e9, 4e9]
  3 set       >= 99 % frozensets / lists
  4 numeric   >= 98 % parse as finite floats and distinct >= 16; ordinal if distinct < 16;
              a configured code hint (name suffix in type_hints.code) or P05's data test
              (`code_override`) makes it categorical
  5 categorical  composite tuple values (e.g. ctx.when); string values of the payload
              namespaces body / q / hdr are text (§6.11 grammar + closed set);
              else categorical when distinct <= 256 or distinct / n <= 0.05
  6 text      otherwise; kv share >= 0.8 -> parse_as 'form', json share >= 0.8 -> 'json'
A type is locked after n >= 500 and >= 1 day since first seen.

Caps: A_max registered names (512; 128 at tier XS); a new name beyond the cap
is counted in `overflow` (HLL of names + count) and admitted when a
dropped / gone / 30-d-unseen attribute can be evicted (lowest coverage first).
"""
from __future__ import annotations

import fnmatch
import json
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import pmdl
from . import psketch as PS
from .pevent import ABSENT, KIND_TXN
from .phier import _ip_parse, shape

PAYLOAD_TEXT_NS = ("body", "q", "hdr")        # string payload values are text (§6.11)
TYPES = ("categorical", "numeric", "ordinal", "ip", "time", "set", "text", "unknown")
A_MAX = 512
TOP_K = 32
HLL_P = 10                 # distinct-count precision of an attribute's HLL (sigma ~ 3 %)
HLL_P_DROPPED = 6          # ... of a dropped (registry-only) attribute (sigma ~ 13 %)
TOP_K_DROPPED = 8          # a dropped attribute is registry-only (§6.4): presence, HLL, a small top
DAY = PS.DAY
UNSEEN_EVICT_S = 30 * DAY
LOCK_N = 500.0
GONE_RATIO = 0.05
GONE_MIN_DAYS = 2          # normal days of the same day type before an attribute can be declared gone
GONE_REF_DAYS = 14.0       # running-mean horizon of the per-day-type reference coverage
GONE_REF_MIN = 0.05        # a day type where the attribute covers < 5 % of events carries no schema signal
BIN_JSD = 0.05
N_BINS = 8
SET_RARE = 0.01
# type-evidence fields (decayed at H_m)
_TE = ("n", "num", "int", "ip", "set", "kv", "json", "time", "len", "tup", "str")
_TE_IX = {k: i for i, k in enumerate(_TE)}
_KV_RE = re.compile(r"^[^=&;\s]{1,64}=[^&;]*(?:[&;][^=&;\s]{1,64}=[^&;]*)*$")


def _as_list(x: Any, n: int) -> List[Any]:
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist() if x.ndim else [x.item()] * n
    return [x] * n


def _ns(name: str) -> str:
    return name.split(".", 1)[0] if "." in name else name


_DEC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def _num(v: Any) -> Optional[float]:
    """A finite float from a number or a plain decimal string (no exponent,
    no inf / nan: hex digests such as '5e150...' are not numbers)."""
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v)
        return x if math.isfinite(x) else None
    if isinstance(v, str) and 0 < len(v) <= 32 and _DEC_RE.match(v):
        x = float(v)
        return x if math.isfinite(x) else None
    return None


_PLAIN_KEY = frozenset({str, float, int, bool, frozenset})   # AttrRecord.key(v) is v (non-text)


def _classify(v: Any, is_time_name: bool) -> Tuple[Optional[float], Tuple[float, ...], Tuple[int, ...]]:
    """(numeric value or None, type-evidence contributions of one row with
    unit evidence, indices of the non-zero ones): the per-row branch of
    AttrRegistry.observe's type evidence (_observe_rows_ref)."""
    te = [0.0] * len(_TE)
    te[0] = 1.0
    x = _num(v)
    if x is not None:
        te[1] = 1.0
        if x.is_integer():
            te[2] = 1.0
        if is_time_name and 1e9 <= x <= 4e9:
            te[7] = 1.0
    if isinstance(v, str):
        te[8] = float(len(v))
        if x is None:
            te[10] = 1.0
            if _ip_parse(v) is not None:
                te[3] = 1.0
            elif "=" in v and len(v) <= 4096 and _KV_RE.match(v):
                te[5] = 1.0
            elif v[:1] in "{[" and len(v) <= 4096:
                try:
                    json.loads(v)
                    te[6] = 1.0
                except ValueError:
                    pass
    elif isinstance(v, (frozenset, set, list)):
        te[4] = 1.0
    elif isinstance(v, tuple):
        te[9] = 1.0
    nz = tuple(i for i, a in enumerate(te) if a != 0.0 or i == 8 and isinstance(v, str))
    return x, tuple(te), nz


def _all_str(v: Any) -> bool:
    """A hashable collection of str only (a memo keyed by such a value is
    exact: 1 == 1.0 would merge two sets whose str forms differ)."""
    if not isinstance(v, (frozenset, tuple)):
        return False
    return all(type(x) is str for x in v)


def _distributions(ss: Any, t: float, chans: Sequence[int]) -> List[Tuple[List[Any], np.ndarray, float]]:
    """[ss.distribution(t, ch) for ch in chans] of a psketch.DecayedSpaceSaving
    in one pass over its slots (the same per-slot expression and sums)."""
    if not all(hasattr(ss, a) for a in ("_keys", "_m", "_err", "_tot_m")):
        return [ss.distribution(t, ch) for ch in chans]
    keys, M, ERR, tot_m = ss._keys, ss._m, ss._err, ss._tot_m
    tots = [tot_m[c] for c in chans]
    if not keys:
        return [([], np.zeros(0), 1.0 if tot > 0 else 0.0) for tot in tots]
    lists: List[List[float]] = [[] for _ in chans]
    live = [(j, c, tot) for j, (c, tot) in enumerate(zip(chans, tots)) if tot > 0]
    for i in range(len(keys)):
        m, e = M[i], ERR[i]
        for j, c, tot in live:
            lists[j].append(max(0.0, m[c] - e[c]) / tot)
    out = []
    for j, tot in enumerate(tots):
        if tot <= 0:
            out.append(([], np.zeros(0), 0.0))
            continue
        sh = np.asarray(lists[j])
        out.append((list(keys), sh, float(max(0.0, 1.0 - sh.sum()))))
    return out


_CLS_MEMO: Tuple[Dict[Any, Any], Dict[Any, Any]] = ({}, {})     # by is_time_name
_CLS_MAX = 1 << 14            # memoised value classifications (bounded, ~2 MB)
_CLS_STR = 256


_SS_ATTRS = ("_lm", "_idx", "_keys", "_m", "_err", "_e", "_tot_m", "_tot_e", "_evict_e", "_mh", "_eh",
             "primary", "k", "_rescale_to")


def ss_add_rows(ss: Any, keys: Sequence[Any], t: float, ws: Sequence[float], evs: Sequence[float]) -> None:
    """`ss.add(keys[j], t, ws[j], evs[j])` for every j in order, at one time t,
    for a psketch.DecayedSpaceSaving: the same float operations in the same
    order (the growth factors 2^((t - L) / h) are computed once instead of per
    row; a rescale can only happen at the first accepted row). Falls back to
    the per-row adds for any other sketch."""
    if not all(hasattr(ss, a) for a in _SS_ATTRS):
        for key, w, ev in zip(keys, ws, evs):
            ss.add(key, t, w, ev)
        return
    t = float(t)
    mh, eh = ss._mh, ss._eh
    nm, ne = len(mh), len(eh)
    fm: Optional[List[float]] = None
    fe: List[float] = []
    tm, te_ = ss._tot_m, ss._tot_e
    idx, kl, M, ERR, E, EE = ss._idx, ss._keys, ss._m, ss._err, ss._e, ss._evict_e
    cap, p = ss.k, ss.primary
    unrolled = nm == 3 and ne == 2
    inf = math.inf
    for key, w, ev in zip(keys, ws, evs):
        w = float(w)
        ev = float(ev)
        if not (0.0 <= w < inf and 0.0 <= ev < inf) or (w == 0.0 and ev == 0.0):
            continue
        if fm is None:
            L = ss._lm.L
            if L is None:
                ss._lm.L = L = t
            elif (t - L) / ss._lm._hmin > PS.RESCALE_EXP:
                ss._rescale_to(t)
                L = t
            dlt = t - L
            fm = [2.0 ** (dlt / h) for h in mh]
            fe = [2.0 ** (dlt / h) for h in eh]
            if unrolled:
                f0, f1, f2 = fm
                e0, e1 = fe
        if unrolled:
            g0, g1, g2 = w * f0, w * f1, w * f2
            h0, h1 = ev * e0, ev * e1
            tm[0] += g0
            tm[1] += g1
            tm[2] += g2
            te_[0] += h0
            te_[1] += h1
            i = idx.get(key)
            if i is not None:
                row = M[i]
                row[0] += g0
                row[1] += g1
                row[2] += g2
                er = E[i]
                er[0] += h0
                er[1] += h1
                continue
            gm = [g0, g1, g2]
            ge = [h0, h1]
        else:
            gm = [w * f for f in fm]
            ge = [ev * f for f in fe]
            for c in range(nm):
                tm[c] += gm[c]
            for c in range(ne):
                te_[c] += ge[c]
            i = idx.get(key)
            if i is not None:
                row = M[i]
                for c in range(nm):
                    row[c] += gm[c]
                er = E[i]
                for c in range(ne):
                    er[c] += ge[c]
                continue
        if len(kl) < cap:
            idx[key] = len(kl)
            kl.append(key)
            M.append(gm)
            ERR.append([0.0] * nm)
            E.append(ge)
            continue
        col = [r[p] for r in M]
        i = col.index(min(col))                # the first slot of least count, as min(range, key)
        del idx[kl[i]]
        kl[i] = key
        idx[key] = i
        old = M[i]
        ERR[i] = list(old)
        M[i] = [old[c] + gm[c] for c in range(nm)]
        E[i] = ge
        for c in range(ne):
            EE[c] += ge[c]


_H64: Dict[str, int] = {}
_H64_MAX = 1 << 15            # memoised item hashes (items <= _CLS_STR characters; ~4 MB at most)


def _hll_add_items(card: Any, items: Iterable[str], t: float) -> None:
    """`card.add(item, t)` for every item (psketch.EpochHLL): one epoch
    rotation at t, then each item's register max; the blake2b hash of an item
    is memoised (bounded) across calls."""
    if not items:
        return
    cur = getattr(card, "cur", None)
    if not hasattr(card, "_rotate") or cur is None or not hasattr(cur, "reg"):
        for it in items:
            card.add(it, t)
        return
    card._rotate(float(t))
    hll = card.cur
    reg = hll.reg
    wbits = 64 - hll.p
    mask = (1 << wbits) - 1
    cache = _H64
    for it in items:
        x = cache.get(it)
        if x is None:
            x = PS._h64(it)
            if len(it) <= _CLS_STR:
                if len(cache) >= _H64_MAX:
                    cache.clear()
                cache[it] = x
        i = x >> wbits
        rho = wbits + 1 - (x & mask).bit_length()
        if rho > reg[i]:
            reg[i] = rho


class AttrRecord:
    """Registry record of one attribute (§5.3)."""

    __slots__ = ("name", "ns", "kinds", "first_seen", "last_seen", "version", "type", "te",
                 "locked", "parse_as", "policy", "pres", "approx", "card", "top", "elem", "num",
                 "mom", "entropy", "stability", "approx_share", "hier", "role_sys", "cost_us",
                 "state", "low_since", "code_hint", "code_override", "gone_at", "day_pres",
                 "prev_day_pres", "cov_dt")

    def __init__(self, name: str, t: float, kind: int = KIND_TXN) -> None:
        self.name = name
        self.ns = _ns(name)
        self.kinds = {int(kind)}
        self.first_seen = float(t)
        self.last_seen = float(t)
        self.version = 1
        self.type = "unknown"
        self.te = PS.DecayedVector([PS.H_M] * len(_TE))
        self.locked = False
        self.parse_as: Optional[str] = None
        self.policy = "clear"
        self.pres = PS.DecayedVector(PS.HALF_LIVES)          # present mass (H_s, H_m, H_l)
        self.approx = PS.DecayedVector([PS.H_M])             # approx-flagged mass
        self.card = PS.EpochHLL(p=HLL_P)
        self.top = PS.DecayedSpaceSaving(TOP_K)
        self.elem: Optional[PS.DecayedSpaceSaving] = None    # set elements (set type)
        self.num: Optional[PS.TDigest] = None
        self.mom: Optional[PS.DecayedVector] = None          # n, s1, s2, s3 of v and of log v; n_pos
        self.entropy = 0.0
        self.stability = 1.0
        self.approx_share = 0.0
        self.hier: Dict[str, Any] = {}
        self.role_sys = "probe"
        self.cost_us = 0.0
        self.state = "active"
        self.low_since: Optional[float] = None
        self.code_hint = False
        self.code_override = False
        self.gone_at: Optional[float] = None
        self.day_pres = 0.0                                  # mass present in the current day
        self.prev_day_pres = 0.0                             # ... in the previous full day
        # reference coverage per day type (0 workday / makeup, 1 weekend and other):
        # [running mean of the daily coverage over normal days, days counted]
        self.cov_dt: List[List[float]] = [[0.0, 0.0], [0.0, 0.0]]

    # ------------------------------------------------------------------ keys
    def key(self, v: Any) -> Any:
        """The level-1 key tracked by `top` (shape for text, the value else;
        a value the policy already shaped is its own shape)."""
        if self.type == "text" and isinstance(v, str):
            return shape(v)
        if isinstance(v, list):
            return frozenset(str(x) for x in v)
        if isinstance(v, np.floating):
            return float(v)
        return v

    def card_estimate(self) -> float:
        return float(self.card.count())

    def coverage(self, sys_mass: np.ndarray, ch: int = PS.CH_M, t: Optional[float] = None) -> float:
        den = float(sys_mass[ch])
        return float(self.pres.read(t)[ch] / den) if den > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        out = {}
        for k in self.__slots__:
            v = getattr(self, k)
            fn = getattr(v, "to_dict", None)
            out[k] = fn() if callable(fn) else (sorted(v) if isinstance(v, set) else v)
        return out


class AttrRegistry:
    """Registry of one tree (§5.3, §6.3). Methods take `t` explicitly."""

    def __init__(self, system: str, a_max: int = A_MAX,
                 code_hints: Sequence[str] = ("status", "code", "port", "qtype", "rcode", "method"),
                 day_offset_s: float = 8 * 3600.0) -> None:
        self.system = system
        self.day_offset_s = float(day_offset_s)             # local-day boundary (tz offset)
        self.cur_day: Optional[int] = None
        self.day_mass: Dict[int, float] = {}
        self.prev_day_mass: Dict[int, float] = {}
        self.gone_checked_day: Optional[int] = None
        self.a_max = int(a_max)
        self.code_hints = tuple(code_hints)
        self.records: Dict[str, AttrRecord] = {}
        self.ev_mass: Dict[int, PS.DecayedVector] = {}
        self.overflow = PS.HLL(p=10)
        self.overflow_n = 0.0
        self.version = 1
        self.last_refresh: Optional[float] = None

    # ----------------------------------------------------------- mapping
    def get(self, name: str, default: Any = None) -> Any:
        return self.records.get(name, default)

    def __contains__(self, name: str) -> bool:
        return name in self.records

    def __len__(self) -> int:
        return len(self.records)

    def names(self, state: Optional[str] = None) -> List[str]:
        return [n for n, r in self.records.items() if state is None or r.state == state]

    # ---------------------------------------------------------- registration
    def register(self, name: str, t: float, kind: int = KIND_TXN) -> str:
        """'known' | 'new' | 'revived' | 'overflow'. Registration is done on the
        current tick (statistics on t - D)."""
        rec = self.records.get(name)
        if rec is not None:
            rec.kinds.add(int(kind))
            if rec.state == "gone":
                rec.state = "active"
                rec.low_since = None
                rec.gone_at = None
                return "revived"
            return "known"
        if len(self.records) >= self.a_max and not self._evict_one(t):
            self.overflow.add(name)
            self.overflow_n += 1.0
            return "overflow"
        rec = AttrRecord(name, t, kind)
        rec.code_hint = any(name.endswith(h) for h in self.code_hints)
        self.records[name] = rec
        self.version += 1
        return "new"

    def _evict_one(self, t: float) -> bool:
        cands = []
        for n, r in self.records.items():
            if r.state == "gone" or r.role_sys == "dropped" or t - r.last_seen >= UNSEEN_EVICT_S:
                cands.append((float(r.pres.read(t)[PS.CH_L]), n))
        if not cands:
            return False
        cands.sort()
        del self.records[cands[0][1]]
        self.version += 1
        return True

    # ----------------------------------------------------------- statistics
    def _day(self, t: float) -> int:
        return int(math.floor((float(t) + self.day_offset_s) / DAY))

    def _roll(self, t: float) -> None:
        d = self._day(t)
        if self.cur_day is None:
            self.cur_day = d
            return
        if d <= self.cur_day:
            return
        adjacent = d == self.cur_day + 1
        self.prev_day_mass = dict(self.day_mass) if adjacent else {}
        self.day_mass = {}
        for r in self.records.values():
            r.prev_day_pres = r.day_pres if adjacent else 0.0
            r.day_pres = 0.0
        self.cur_day = d

    def observe_events(self, kind: int, t: float, mass: float) -> None:
        """System event mass of a learned batch (coverage denominator)."""
        self._roll(t)
        self.day_mass[int(kind)] = self.day_mass.get(int(kind), 0.0) + float(mass)
        dv = self.ev_mass.get(int(kind))
        if dv is None:
            dv = self.ev_mass[int(kind)] = PS.DecayedVector(PS.HALF_LIVES)
        dv.add(float(t), float(mass))

    def sys_mass(self, rec: AttrRecord, t: Optional[float] = None) -> np.ndarray:
        out = np.zeros(3)
        for k in rec.kinds:
            dv = self.ev_mass.get(k)
            if dv is not None:
                out += dv.read(t)
        return out

    def coverage(self, name: str, t: Optional[float] = None, ch: int = PS.CH_M) -> float:
        rec = self.records.get(name)
        if rec is None:
            return 0.0
        den = self.sys_mass(rec, t)[ch]
        return float(rec.pres.read(t)[ch] / den) if den > 0 else 0.0

    def observe(self, name: str, values: Sequence[Any], t: float, mass: Any = 1.0,
                evidence: Any = None, approx: Any = None, kind: int = KIND_TXN,
                policy: Optional[str] = None) -> int:
        """Update one attribute from the learned rows that carry it (values,
        mass per row, evidence per row (default = 1 per row, capped at 1),
        approx flag per row). Registers the name when unknown. Returns rows used."""
        st = self.register(name, t, kind)
        if st == "overflow":
            return 0
        rec = self.records[name]
        N = len(values)
        ms = _as_list(mass, N)
        es = _as_list(1.0 if evidence is None else evidence, N)
        aps = _as_list(False if approx is None else approx, N)
        vals, m, ev, apm = [], [], [], 0.0
        for v, mm, ee, aa in zip(values, ms, es, aps):
            if v is None or v is ABSENT:
                continue
            vals.append(v)
            m.append(float(mm))
            ev.append(min(1.0, float(ee)))
            if aa:
                apm += float(mm)
        n = len(vals)
        if n == 0:
            return 0
        if policy:
            rec.policy = policy
        t = float(t)
        self._roll(t)
        rec.last_seen = max(rec.last_seen, t)
        tot_m = sum(m)
        rec.pres.add(t, tot_m)
        rec.day_pres += tot_m
        if apm > 0:
            rec.approx.add(t, apm)
        # type evidence (evidence-weighted: typing is about what was observed).
        # Batched per call (bit-identical to the per-row form, which the
        # equivalence test keeps as its reference, tests/lib/
        # test_pregistry_batch_equivalence.py, on recorded pack O streams):
        # each distinct value is classified once, and with unit evidence (P02's
        # case) the fields are integer counts, so summing per distinct value
        # gives the same floats as the per-row sums.
        is_time_name = rec.name.endswith("_ts") or rec.name.endswith(".ts")
        # (memo across calls, bounded: the classification is a function of the
        # value and of is_time_name; values recur from tick to tick)
        memo = _CLS_MEMO[1 if is_time_name else 0]
        if len(memo) >= _CLS_MAX:
            memo.clear()
        classes = []
        for v in vals:
            try:
                # (not for a float zero: 0.0 == -0.0, and the row's own value is
                # kept; not for long strings)
                cls = v.__class__
                k = (cls, v) if ((cls is not float or v != 0.0)
                                 and (cls is not str or len(v) <= _CLS_STR)) else None
                c = memo.get(k) if k is not None else None
            except TypeError:                              # unhashable (a list)
                k = c = None
            if c is None:
                c = _classify(v, is_time_name)
                if k is not None:
                    memo[k] = c
            classes.append(c)
        te = [0.0] * len(_TE)
        if all(e == 1.0 for e in ev):
            cnt: Dict[int, List[Any]] = {}
            for c in classes:
                hit = cnt.get(id(c))
                if hit is None:
                    cnt[id(c)] = [c, 1]
                else:
                    hit[1] += 1
            for c, k_ in cnt.values():
                vec = c[1]
                for i in c[2]:
                    te[i] += k_ * vec[i]
        else:
            for c, e in zip(classes, ev):
                vec = c[1]
                for i in c[2]:
                    te[i] += e * vec[i] if i == 8 else e
        rec.te.add(t, te)
        nums = [c[0] for c in classes if c[0] is not None]
        num_m = [mm for c, mm in zip(classes, m) if c[0] is not None]
        # distinct values: the HLL registers are a max over the hashed items,
        # so each distinct item is hashed and folded in once (same registers)
        card = rec.card
        items = set()
        jm: Dict[Any, str] = {}
        for v in vals[:4096]:
            if isinstance(v, (frozenset, set, list, tuple)):
                try:
                    it = jm.get(v)
                except TypeError:
                    it = None
                if it is None:
                    it = "|".join(sorted(str(x) for x in v))
                    if _all_str(v):                        # (equal sets of str: equal joins)
                        jm[v] = it
            else:
                it = v if type(v) is str else str(v)
            items.add(it)
        _hll_add_items(card, items, t)
        # level-1 keys and set elements (mass / evidence per row, in row order)
        dropped = rec.role_sys == "dropped"
        if rec.type == "text":
            km: Dict[Any, Any] = {}
            keys = []
            for v in vals:
                if isinstance(v, str):
                    kk = (v.__class__, v)
                    x = km.get(kk)
                    if x is None:
                        x = km[kk] = shape(v)
                    keys.append(x)
                else:
                    keys.append(rec.key(v))
        else:
            keys = [v if v.__class__ in _PLAIN_KEY else rec.key(v) for v in vals]
        ss_add_rows(rec.top, keys, t, m, ev)
        if not dropped:
            ek, ew, ee = [], [], []
            em: Dict[Any, List[str]] = {}
            for v, mm, e in zip(vals, m, ev):
                if isinstance(v, (frozenset, set, list)):
                    if rec.elem is None:
                        rec.elem = PS.DecayedSpaceSaving(TOP_K)
                    try:
                        xs = em.get(v)
                    except TypeError:
                        xs = None
                    if xs is None:
                        xs = [str(x) for x in set_elements(v)[:32]]
                        if _all_str(v):
                            em[v] = xs
                    ek.extend(xs)
                    ew.extend([mm] * len(xs))
                    ee.extend([e] * len(xs))
            if ek:
                ss_add_rows(rec.elem, ek, t, ew, ee)
        # numeric summaries (not for a dropped attribute: registry-only)
        if nums and not dropped:
            if rec.num is None:
                rec.num = PS.TDigest(50.0, PS.H_M)
                rec.mom = PS.DecayedVector([PS.H_M] * 9)
            xv = np.asarray(nums)
            mv = np.maximum(np.asarray(num_m), 1e-12)
            vmin = float(xv.min())
            # moments only (skewness test): values clipped to +-1e30 (np.clip
            # costs ~5 us a call; nothing to clip is the rule)
            xm = xv if (vmin >= -1e30 and float(xv.max()) <= 1e30) else np.clip(xv, -1e30, 1e30)
            rec.num.add_many(xv, t, mv)
            pos = xv > 0
            lv = np.log(np.where(pos, xv, 1.0))
            # the 8 weighted moment sums as one row-wise reduction of a C-ordered
            # (8, n) array: numpy sums each contiguous row pairwise exactly as
            # the 1-d .sum() of that row (bit-identical, test)
            M = np.empty((8, xv.size))
            M[0] = mv
            M[1] = mv * xm
            M[2] = mv * xm ** 2
            M[3] = mv * xm ** 3
            M[4] = mv * pos
            M[5] = mv * lv * pos
            M[6] = mv * lv ** 2 * pos
            M[7] = mv * lv ** 3 * pos
            mo = np.zeros(9)
            M.sum(axis=1, out=mo[:8])
            rec.mom.add(t, mo)
            rec.hier["vmin"] = min(rec.hier.get("vmin", vmin), vmin)
        return n

    def observe_batch(self, batch: Any, t: Optional[float] = None,
                      names: Optional[Iterable[str]] = None, rows: Optional[np.ndarray] = None,
                      evidence: Optional[np.ndarray] = None) -> int:
        """Convenience: observe every column (or `names`) of an EventBatch over
        its learned rows (or `rows`), mass = w / pi, approx from flags."""
        t = float(batch.t1 if t is None else t)
        rr = batch.learned_rows() if rows is None else np.asarray(rows, dtype=np.int64)
        if rr.size == 0:
            return 0
        mass_all = batch.mass() if rows is None else (batch.w / np.maximum(batch.pi, 1e-12)).astype(float)
        self.observe_events(batch.kind, t, float(mass_all[rr].sum()))
        sel = np.zeros(batch.n, dtype=bool)
        sel[rr] = True
        ev_all = np.ones(batch.n) if evidence is None else np.asarray(evidence, dtype=float)
        policy = (batch.meta or {}).get("policy", {})
        used = 0
        for nm in (batch.names() if names is None else names):
            c = batch.cols.get(nm)
            if c is None:
                continue
            m = sel[c.rows]
            if not m.any():
                self.register(nm, t, batch.kind)
                continue
            r = c.rows[m]
            vals = c.vals[m]
            used += self.observe(nm, list(vals), t, mass_all[r], ev_all[r],
                                 (batch.flags[r] & 1) > 0, batch.kind, policy.get(nm))
        return used

    # ------------------------------------------------------------- typing
    def infer_type(self, rec: AttrRecord, t: float) -> str:
        te = rec.te.read(t)
        n = te[0]
        if n <= 0:
            return "unknown"
        f = te / n
        distinct = rec.card.count()
        if f[_TE_IX["ip"]] >= 0.99:
            return "ip"
        if (rec.name.endswith("_ts") or rec.name.endswith(".ts")) and f[_TE_IX["time"]] >= 0.99:
            return "time"
        if f[_TE_IX["set"]] >= 0.99:
            return "set"
        if f[_TE_IX["tup"]] >= 0.99:
            return "categorical"                        # composite keys, e.g. ctx.when
        if f[_TE_IX["num"]] >= 0.98:
            if rec.code_hint or rec.code_override:
                return "categorical"
            return "numeric" if distinct >= 16 else "ordinal"
        if rec.ns in PAYLOAD_TEXT_NS and f[_TE_IX["str"]] >= 0.98:
            return "text"                               # payload values: grammar + closed set (§6.11)
        if distinct <= 256 or (distinct / max(n, 1.0)) <= 0.05:
            return "categorical"
        return "text"

    def update_types(self, t: float) -> List[Tuple[str, str, str]]:
        """Re-infer unlocked types; returns [(name, old, new)] for changes."""
        out = []
        for nm, rec in self.records.items():
            if rec.locked:
                continue
            new = self.infer_type(rec, t)
            te = rec.te.read(t)
            n = te[0]
            if new == "text" and n > 0:
                if te[_TE_IX["kv"]] / n >= 0.8:
                    rec.parse_as = "form"
                elif te[_TE_IX["json"]] / n >= 0.8:
                    rec.parse_as = "json"
            if new != rec.type:
                out.append((nm, rec.type, new))
                rec.type = new
                rec.version += 1
                rec.top = PS.DecayedSpaceSaving(TOP_K_DROPPED if rec.role_sys == "dropped" else TOP_K)
                if new == "set" and rec.elem is None and rec.role_sys != "dropped":
                    rec.elem = PS.DecayedSpaceSaving(TOP_K)
            if n >= LOCK_N and t - rec.first_seen >= DAY and rec.type != "unknown":
                rec.locked = True
        if out:
            self.version += 1
        return out

    def update_stats(self, t: float) -> None:
        """Entropy (Chao-Shen on evidence), stability 1 - JSD(p_Hs, p_Hl),
        approx share, log flag."""
        for rec in self.records.values():
            if len(rec.top):                       # (tracked keys: what top.items() tested)
                (keys, sh, other), (_, sh_s, o_s), (_, sh_l, o_l) = \
                    _distributions(rec.top, t, (PS.CH_M, PS.CH_S, PS.CH_L))
                rec.entropy = entropy_from_top(sh, other, rec.card.count(), len(keys))
                rec.stability = 1.0 - pmdl.jsd(np.append(sh_s, o_s), np.append(sh_l, o_l))
            pm = float(rec.pres.read(t)[PS.CH_M])
            rec.approx_share = float(rec.approx.read(t)[0] / pm) if pm > 0 else 0.0
            if rec.mom is not None:
                mo = rec.mom.read(t)
                rec.hier["log"] = bool(_log_better(mo, rec.hier.get("vmin", 0.0)))

    # --------------------------------------------------------- hierarchies
    def refresh_hierarchies(self, t: float, force: bool = False) -> List[str]:
        """Numeric bin edges (8 bins at the t-digest's 1/8..7/8 quantiles, on
        log scale when `log`), replaced only when the JSD between the old and
        new bin occupancy exceeds 0.05; ordinal medians; set templates
        (elements present in >= 1 % of events); status flag. At most daily
        unless force. Returns the names whose hierarchy version changed."""
        if not force and self.last_refresh is not None and t - self.last_refresh < DAY:
            return []
        self.last_refresh = float(t)
        changed = []
        for nm, rec in self.records.items():
            h = rec.hier
            if rec.type in ("numeric", "time") and rec.num is not None and rec.num.total() > 0:
                lg = bool(h.get("log", False))
                qs = [rec.num.quantile(k / N_BINS) for k in range(1, N_BINS)]
                if lg:
                    qs = [math.log(q) if q > 0 else -math.inf for q in qs]
                new = np.unique(np.asarray([q for q in qs if math.isfinite(q)]))
                old = h.get("edges")
                if new.size and (old is None or h.get("edges_log") != lg
                                 or _occupancy_jsd(rec.num, old, new, lg) > BIN_JSD):
                    h["edges"] = new
                    h["edges_log"] = lg
                    h["hver"] = int(h.get("hver", 0)) + 1
                    changed.append(nm)
            elif rec.type == "ordinal" and rec.num is not None:
                med = rec.num.quantile(0.5)
                if h.get("median") != med:
                    h["median"] = med
                    changed.append(nm)
            elif rec.type == "set" and rec.elem is not None:
                tot = rec.top.total(t)
                keep = frozenset(k for k, c, _, _ in rec.elem.items(t)
                                 if tot > 0 and c / tot >= SET_RARE)
                if keep != h.get("set_keep"):
                    h["set_keep"] = keep
                    changed.append(nm)
            if rec.type in ("categorical", "ordinal") and (rec.code_hint or nm.endswith("status")):
                h["status"] = nm.endswith("status")
        for nm in changed:
            self.records[nm].version += 1
        if changed:
            self.version += 1
        return changed

    def set_value_groups(self, name: str, groups: Mapping[Any, Any]) -> None:
        """Categorical level-1 value groups (P05, §5.4.6)."""
        rec = self.records.get(name)
        if rec is not None:
            rec.hier["groups"] = dict(groups)
            rec.version += 1

    def set_role(self, name: str, role: str) -> None:
        """P05's system role. A `dropped` attribute is registry-only (§6.4 role
        table: presence, HLL): its value summaries (top values, numeric digest
        and moments, set elements) are released and kept at TOP_K_DROPPED; they
        are rebuilt from the next events when P05 gives it a role again (its
        weekly re-probe evaluates it on P05's own probe rows). Measured on the
        PG4 attribute axis: every registered attribute kept a 32-value top,
        a t-digest and moments, so the registry grew with the number of
        attributes although 2/3 of pack O-scale's synthetic ones are noise."""
        rec = self.records.get(name)
        if rec is None:
            return
        old = rec.role_sys
        rec.role_sys = role
        if role == "dropped" and old != "dropped":
            rec.top = PS.DecayedSpaceSaving(TOP_K_DROPPED)
            rec.num = None
            rec.mom = None
            rec.elem = None
            # the distinct count at registry precision (M42): the HLL folds
            # exactly to 2^HLL_P_DROPPED registers (sigma ~ 13 %), 2 KB -> 0.1 KB
            # per attribute; the epochs started after a re-promotion are full size
            rec.card.set_precision(HLL_P_DROPPED)
        elif old == "dropped" and role != "dropped":
            rec.top = PS.DecayedSpaceSaving(TOP_K)
            rec.card.set_precision(HLL_P)

    # ------------------------------------------------------- schema change
    def check_gone(self, t: float, normal_day: bool = True, daytype: Optional[int] = None) -> List[str]:
        """Declare `gone` an attribute whose coverage over the previous full
        local day fell below 5 % of its reference coverage, when that day was a
        normal day (the caller passes normal_day for the PREVIOUS day, from
        P01's calendar / volume flag; holidays never count) and the system had
        events that day. Evaluated once per day. Returns names newly gone.

        With `daytype` (0 workday / makeup workday, 1 weekend / other) the
        reference is the attribute's mean daily coverage over past normal days
        of the SAME day type (>= GONE_MIN_DAYS of them): an attribute that only
        workday actions carry (request bodies of logins and approvals) has no
        coverage on a normal weekend, which is not a schema change (measured on
        pack O: every body.* attribute was declared gone each weekend against the
        all-day H_l coverage, collapsing the tree's splits on them). Without
        `daytype` the H_l coverage is the reference (the original rule)."""
        self._roll(t)
        if self.cur_day is None or self.gone_checked_day == self.cur_day:
            return []
        self.gone_checked_day = self.cur_day
        if not normal_day:
            return []
        out = []
        for nm, rec in self.records.items():
            if rec.state != "active":
                continue
            den = sum(self.prev_day_mass.get(k, 0.0) for k in rec.kinds)
            sm = self.sys_mass(rec, t)
            if den <= 0 or sm[PS.CH_L] <= 0:
                continue
            cov_day = rec.prev_day_pres / den
            if daytype is None:
                ref = rec.pres.read(t)[PS.CH_L] / sm[PS.CH_L]
                ref_ok = True
            else:
                cd = getattr(rec, "cov_dt", None)
                if cd is None:
                    cd = rec.cov_dt = [[0.0, 0.0], [0.0, 0.0]]
                ref, n_ref = cd[int(daytype)]
                # (late rows of the previous day type spill over midnight through the
                # learning delay: a residual coverage of a few % is not a signal)
                ref_ok = n_ref >= GONE_MIN_DAYS and ref >= GONE_REF_MIN
            if ref_ok and ref > 0 and cov_day < GONE_RATIO * ref:
                rec.state = "gone"
                rec.gone_at = float(t)
                rec.low_since = float(t)
                out.append(nm)
            elif daytype is not None:
                c = rec.cov_dt[int(daytype)]
                c[1] += 1.0
                c[0] += (cov_day - c[0]) / min(c[1], GONE_REF_DAYS)
        if out:
            self.version += 1
        return out

    # -------------------------------------------------------------- misc
    def nbytes(self) -> int:
        tot = 2048
        for r in self.records.values():
            tot += 600 + r.top.nbytes() + r.card.nbytes() + r.te.nbytes() + r.pres.nbytes()
            if r.num is not None:
                tot += r.num.nbytes() + 200
            if r.elem is not None:
                tot += r.elem.nbytes()
        return int(tot)

    def to_dict(self) -> Dict[str, Any]:
        return {"fmt": 1, "system": self.system, "a_max": self.a_max, "version": self.version,
                "records": {n: r.to_dict() for n, r in self.records.items()},
                "ev_mass": {k: v.to_dict() for k, v in self.ev_mass.items()},
                "overflow_n": self.overflow_n, "overflow": self.overflow.to_dict(),
                "last_refresh": self.last_refresh}


def entropy_from_top(shares: np.ndarray, other: float, distinct: float, k: int) -> float:
    """Entropy (bits) from a heavy-hitter summary: the tracked shares plus the
    untracked mass spread uniformly over the HLL-estimated remaining distinct
    values: H(shares, other) + other * log2(max(1, distinct - k))."""
    p = np.r_[np.asarray(shares, dtype=float), max(0.0, float(other))]
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    h = float(-(p * np.log2(p)).sum())
    return h + max(0.0, float(other)) * math.log2(max(1.0, float(distinct) - k))


def _log_better(mo: np.ndarray, vmin: float) -> bool:
    """`log` flag: all values > 0 and |skewness(log v)| < |skewness(v)|."""
    n, s1, s2, s3, npos, l1, l2, l3 = mo[:8]
    if n <= 0 or npos < n * 0.999 or vmin <= 0 or not np.all(np.isfinite(mo[:8])):
        return False

    def skew(k, a, b, c):
        mu = a / k
        var = b / k - mu * mu
        if var <= 1e-18:
            return 0.0
        m3 = c / k - 3 * mu * b / k + 2 * mu ** 3
        return m3 / var ** 1.5
    return abs(skew(npos, l1, l2, l3)) < abs(skew(n, s1, s2, s3))


def _occupancy_jsd(td: PS.TDigest, old: np.ndarray, new: np.ndarray, lg: bool) -> float:
    """JSD between the current distribution's occupancy of the OLD bins and
    the occupancy the NEW (quantile) bins give, which is ~uniform by
    construction; the two vectors may differ in length when edges merge, so
    the new side is its own occupancy only when the lengths agree."""
    def occ(edges: np.ndarray) -> np.ndarray:
        e = np.exp(edges) if lg else edges
        cdf = np.asarray([td.cdf(float(x)) for x in e])
        return np.maximum(np.diff(np.r_[0.0, cdf, 1.0]), 0.0)
    o_old = occ(np.asarray(old, dtype=float))
    o_new = occ(np.asarray(new, dtype=float))
    if o_new.size != o_old.size:
        o_new = np.full(o_old.size, 1.0 / o_old.size)
    return pmdl.jsd(o_old, o_new)


def set_elements(v: Any) -> List[Any]:
    """The elements of a set-typed value in a process-independent order: a
    (frozen)set iterates in str-hash order, which Python salts per process
    (PYTHONHASHSEED), so which 32 elements a large set contributes and the
    order of the element sketch's slots (its eviction ties) differed between
    two identical runs (§16.12.6). Sets are sorted by (str, type name); a
    list keeps its own order."""
    if isinstance(v, (set, frozenset)):
        return sorted(v, key=_elem_key)
    return list(v)


def _elem_key(x: Any) -> Tuple[str, str]:
    return str(x), type(x).__name__


def value_policy_matches(name: str, globs: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, g) for g in globs)
