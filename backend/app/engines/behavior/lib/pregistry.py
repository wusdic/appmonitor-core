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
DAY = PS.DAY
UNSEEN_EVICT_S = 30 * DAY
LOCK_N = 500.0
GONE_RATIO = 0.05
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


class AttrRecord:
    """Registry record of one attribute (§5.3)."""

    __slots__ = ("name", "ns", "kinds", "first_seen", "last_seen", "version", "type", "te",
                 "locked", "parse_as", "policy", "pres", "approx", "card", "top", "elem", "num",
                 "mom", "entropy", "stability", "approx_share", "hier", "role_sys", "cost_us",
                 "state", "low_since", "code_hint", "code_override", "gone_at", "day_pres",
                 "prev_day_pres")

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
        self.card = PS.EpochHLL(p=10)
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
        # type evidence (evidence-weighted: typing is about what was observed)
        te = [0.0] * len(_TE)
        nums: List[float] = []
        num_m: List[float] = []
        is_time_name = rec.name.endswith("_ts") or rec.name.endswith(".ts")
        for v, e, mm in zip(vals, ev, m):
            te[0] += e
            x = _num(v)
            if x is not None:
                nums.append(x)
                num_m.append(mm)
                te[1] += e
                if x.is_integer():
                    te[2] += e
                if is_time_name and 1e9 <= x <= 4e9:
                    te[7] += e
            if isinstance(v, str):
                te[8] += e * len(v)
                if x is None:
                    te[10] += e
                    if _ip_parse(v) is not None:
                        te[3] += e
                    elif "=" in v and len(v) <= 4096 and _KV_RE.match(v):
                        te[5] += e
                    elif v[:1] in "{[" and len(v) <= 4096:
                        try:
                            json.loads(v)
                            te[6] += e
                        except ValueError:
                            pass
            elif isinstance(v, (frozenset, set, list)):
                te[4] += e
            elif isinstance(v, tuple):
                te[9] += e
        rec.te.add(t, te)
        # distinct values and level-1 keys
        card = rec.card
        for v in vals[:4096]:
            card.add(v if not isinstance(v, (frozenset, set, list, tuple))
                     else "|".join(sorted(str(x) for x in v)), t)
        top = rec.top
        key = rec.key
        for v, mm, e in zip(vals, m, ev):
            top.add(key(v), t, mm, e)
            if isinstance(v, (frozenset, set, list)):
                if rec.elem is None:
                    rec.elem = PS.DecayedSpaceSaving(TOP_K)
                for x in list(v)[:32]:
                    rec.elem.add(str(x), t, mm, e)
        # numeric summaries
        if nums:
            if rec.num is None:
                rec.num = PS.TDigest(50.0, PS.H_M)
                rec.mom = PS.DecayedVector([PS.H_M] * 9)
            xv = np.asarray(nums)
            mv = np.maximum(np.asarray(num_m), 1e-12)
            xm = np.clip(xv, -1e30, 1e30)             # moments only (skewness test)
            rec.num.add_many(xv, t, mv)
            pos = xv > 0
            lv = np.log(np.where(pos, xv, 1.0))
            rec.mom.add(t, np.asarray([
                mv.sum(), (mv * xm).sum(), (mv * xm ** 2).sum(), (mv * xm ** 3).sum(),
                (mv * pos).sum(), (mv * lv * pos).sum(), (mv * lv ** 2 * pos).sum(),
                (mv * lv ** 3 * pos).sum(), 0.0]))
            vmin = float(xv.min())
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
                rec.top = PS.DecayedSpaceSaving(TOP_K)      # keys change meaning with the type
                if new == "set" and rec.elem is None:
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
            items = rec.top.items(t, PS.CH_M)
            if items:
                keys, sh, other = rec.top.distribution(t, PS.CH_M)
                rec.entropy = entropy_from_top(sh, other, rec.card.count(), len(keys))
                keys, sh_s, o_s = rec.top.distribution(t, PS.CH_S)
                _, sh_l, o_l = rec.top.distribution(t, PS.CH_L)
                rec.stability = 1.0 - pmdl.jsd(np.r_[sh_s, o_s], np.r_[sh_l, o_l])
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
        rec = self.records.get(name)
        if rec is not None:
            rec.role_sys = role

    # ------------------------------------------------------- schema change
    def check_gone(self, t: float, normal_day: bool = True) -> List[str]:
        """Declare `gone` an attribute whose coverage over the previous full
        local day fell below 5 % of its H_l coverage, when that day was a
        normal day (the caller passes normal_day for the PREVIOUS day, from
        P01's calendar / volume flag; holidays never count) and the system had
        events that day. Evaluated once per day. Returns names newly gone."""
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
            cov_l = rec.pres.read(t)[PS.CH_L] / sm[PS.CH_L]
            if cov_l > 0 and cov_day < GONE_RATIO * cov_l:
                rec.state = "gone"
                rec.gone_at = float(t)
                rec.low_since = float(t)
                out.append(nm)
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


def value_policy_matches(name: str, globs: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, g) for g in globs)
