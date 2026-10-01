"""Pattern-node data model of the progressive core (docs/lib3/progressive.md §5.5).

STATUS: implemented (W-P0). Pure data structures; P04 owns when they are
updated, P03/P06-P14 read them. Every summary is bounded, forward-decayed,
mergeable (prune / sibling merge, §6.6) and separates mass from evidence
(PPC-9); the evidence of confidence-bearing counters lives on the confidence
channel (the last evidence channel, psketch.EV_L) with `reset_confidence`.

Summaries
    WhoSummary      per IP level (/32, /24, /16, grp, reg) a DecayedSpaceSaving(8),
                    an EpochHLL(p=6) of distinct /32, per-level prequential two-part
                    code lengths (§6.18.2); closedness and heavy sets (§5.5.3, §6.16.2)
    WhenSummary     hist96[daytype][96] (H_m mass) + evidence per day type on the
                    confidence channel + optional minute reservoir (§5.5.4)
    CatSummary      categorical / ordinal / ip targets: DecayedSpaceSaving(16)
    NumSummary      t-digest (H_m), moments at H_s/H_m/H_l, daily (min, max, n_obs)
                    ring of 30 days, lower / upper exceedance reservoirs (64 each)
    TextSummary     shapes SS(8), exact values SS(16) (policy clear / hmac), log2
                    length histogram (16), charset-class counts (8)
    SetSummary      templates SS(8), element presence SS(32)
    PairSketch      binding pair counts: SS(64) over x, each with SS(4) over y (§6.12)
Node                one pattern (§5.5.2)

Counts in every p-value / bound are evidence on the confidence channel.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import psketch as PS
from .phier import Shaped, charset, shape, shape_length

STATES = ("candidate", "confirmed", "stable", "evolving", "stale", "retired", "dormant")
CONFIDENT_STATES = frozenset({"confirmed", "stable", "evolving", "stale"})
WHO_LEVELS = 5                     # /32, /24, /16, grp, reg
WHO_K = 8
WHO_HLL_P = 6
CAT_K = 16
TEXT_SHAPES_K = 8
TEXT_VALUES_K = 16
SET_TPL_K = 8
SET_ELEM_K = 32
RING_DAYS = 30
EXC_RES = 64
PAIR_KX = 64
PAIR_KY = 4
DAYTYPES = ("wd", "nwd")
CLOSED_U = 0.05                    # who closed: unseen mass <= 0.05
HEAVY_COVER = 0.95                 # heavy set covers >= 95 % of mass
HEAVY_MAX = 8
CLOSED_DAYS = 5
DAY = PS.DAY
SUS_MAX = 16                       # suspect sources remembered per who summary
SUS_KEEP_S = 30 * DAY              # a suspect source is forgotten after 30 d without a row


def _conf_ss(k: int) -> PS.DecayedSpaceSaving:
    return PS.DecayedSpaceSaving(k, PS.HALF_LIVES, PS.EV_HALF_LIVES, PS.CH_M)


# ================================================================== who
class WhoSummary:
    """Who summary of a node (§5.5.3). `update(keys, ...)` takes the event's
    generalised IP per level (keys[l] for l in 0..4, ABSENT / None to skip a
    level, e.g. grp when the IP has no group) and the raw /32 for the HLL."""

    __slots__ = ("levels", "hll", "code", "code_n", "sus")

    def __init__(self, n_levels: int = WHO_LEVELS, k: int = WHO_K) -> None:
        self.levels = [_conf_ss(k) for _ in range(n_levels)]
        self.hll = PS.EpochHLL(p=WHO_HLL_P)
        self.code = np.zeros(n_levels)        # prequential code length per level (bits)
        self.code_n = 0.0                     # evidence those code lengths cover
        # suspect sources: {ip: [last row ts, rows]} - sources P03 learned damped as
        # foreign here; their rows never enter the levels (mass, evidence, U, heavy
        # sets, rendering), §6.9.2-§6.9.3
        self.sus: Dict[str, List[float]] = {}

    # ------------------------------------------------------- suspects
    def _sus(self) -> Dict[str, List[float]]:
        d = getattr(self, "sus", None)
        if d is None:
            d = self.sus = {}
        return d

    def is_suspect(self, ip: str, t: float) -> bool:
        r = self._sus().get(ip)
        return r is not None and float(t) - r[0] <= SUS_KEEP_S

    def mark_suspect(self, ip: str, t: float) -> None:
        """Record a row of a suspect source (its first damped-as-foreign row, or any
        later row while it is suspect: persistence alone never makes it a member,
        §6.9.2). Bounded: the source with the oldest last row is forgotten first."""
        d = self._sus()
        r = d.get(ip)
        if r is None:
            if len(d) >= SUS_MAX:
                del d[min(d, key=lambda k: d[k][0])]
            d[ip] = [float(t), 1.0]
        else:
            r[0] = max(r[0], float(t))
            r[1] += 1.0

    def suspects(self, t: float) -> List[str]:
        return [ip for ip, r in self._sus().items() if float(t) - r[0] <= SUS_KEEP_S]

    def code_lengths(self, keys: Sequence[Any], t: float, space_bits: Sequence[float],
                     escape_bits: Sequence[float], alpha: float = 2.0) -> np.ndarray:
        """Two-part code length of the event's IP at every level under the
        current (pre-update) summaries (§6.18.2):
        seen: -log2 p(g) + log2|g|; unseen: -log2 U + escape bits + log2|g|;
        p(g) = (share(g) n + alpha / (k + 1)) / (n + alpha), n = confidence evidence."""
        out = np.zeros(len(self.levels))
        for l, ss in enumerate(self.levels):
            g = keys[l] if l < len(keys) else None
            if g is None:
                out[l] = np.nan
                continue
            n = ss.total_evidence(t)
            U = ss.unseen(t)
            sh = ss.share(g, t) if g in ss else 0.0
            if sh > 0:
                p = (sh * n + alpha / (ss.k + 1)) / (n + alpha)
                out[l] = -math.log2(p) + float(space_bits[l])
            else:
                out[l] = -math.log2(max(U, 1e-12)) + float(escape_bits[l]) + float(space_bits[l])
        return out

    def update(self, keys: Sequence[Any], ip: str, t: float, mass: float, evidence: float,
               code: Optional[np.ndarray] = None) -> None:
        for l, ss in enumerate(self.levels):
            g = keys[l] if l < len(keys) else None
            if g is not None:
                ss.add(g, t, mass, evidence)
        self.hll.add(ip, t)
        if code is not None:
            c = np.where(np.isfinite(code), code, 0.0)
            self.code += evidence * c
            self.code_n += evidence

    def heavy_set(self, level: int, t: float, cover: float = HEAVY_COVER
                  ) -> Tuple[List[Hashable], float]:
        """Smallest set of items (by mass) covering >= cover of the node's
        mass at the level; returns (items, covered share). Uses guaranteed
        counts, so the covered share is a lower bound."""
        ss = self.levels[level]
        tot = ss.total(t)
        if tot <= 0:
            return [], 0.0
        items, acc = [], 0.0
        for key, _, guar, _ in ss.items(t):
            if acc >= cover * tot:
                break
            items.append(key)
            acc += guar
        return items, acc / tot

    def closed_level(self, t: float, n_days: int, u_max: float = CLOSED_U,
                     heavy_max: int = HEAVY_MAX, min_days: int = CLOSED_DAYS,
                     levels: Optional[Sequence[int]] = None) -> Optional[int]:
        """Finest level at which the who summary is closed (§6.16.2): unseen
        mass U <= u_max on the confidence channel, heavy set <= heavy_max
        items covering >= 95 %, evidence over >= min_days normal days
        (n_days is supplied by the node's date bitmap). None when open."""
        if n_days < min_days:
            return None
        for l in (levels if levels is not None else range(len(self.levels))):
            ss = self.levels[l]
            if ss.total(t) <= 0:
                continue
            if ss.unseen(t) > u_max:
                continue
            items, cov = self.heavy_set(l, t)
            if len(items) <= heavy_max and cov >= HEAVY_COVER - 1e-9:
                return l
        return None

    def distinct(self, t: Optional[float] = None) -> float:
        return self.hll.count(t)

    def merge(self, other: "WhoSummary") -> "WhoSummary":
        for a, b in zip(self.levels, other.levels):
            a.merge(b)
        self.hll.merge(other.hll)
        self.code += other.code
        self.code_n += other.code_n
        for ip, r in other._sus().items():
            self.mark_suspect(ip, r[0])
        return self

    def reset_confidence(self, t: float) -> None:
        for ss in self.levels:
            ss.reset_confidence(t)

    def nbytes(self) -> int:
        return int(sum(s.nbytes() for s in self.levels) + self.hll.nbytes() + self.code.nbytes + 64
                   + 96 * len(self._sus()))


# ================================================================= when
class WhenSummary:
    """hist96[daytype][96] mass at H_m, evidence per day type (H_m, H_l with
    confidence reset), optional minute reservoir (R_t = 256) (§5.5.4)."""

    __slots__ = ("hist", "_L", "ev", "res")

    def __init__(self) -> None:
        self.hist = np.zeros((2, 96))
        self._L: Optional[float] = None
        self.ev = [PS.DecayedVector(PS.EV_HALF_LIVES) for _ in DAYTYPES]
        self.res: Optional[PS.WeightedReservoir] = None

    def want_minutes(self, on: bool = True, R: int = 256, seed: int = 0) -> None:
        if on and self.res is None:
            self.res = PS.WeightedReservoir(R, PS.H_M, seed)
        elif not on:
            self.res = None

    def _rescale(self, t: float) -> None:
        if self._L is None:
            self._L = t
        elif (t - self._L) / PS.H_M > PS.RESCALE_EXP:
            self.hist *= 2.0 ** (-(t - self._L) / PS.H_M)
            self._L = t

    def update(self, daytype: int, minute: float, t: float, mass: float, evidence: float,
               u: Optional[float] = None, src: Optional[str] = None) -> None:
        d = 1 if daytype else 0
        t = float(t)
        self._rescale(t)
        slot = int(float(minute) // 15) % 96
        self.hist[d, slot] += mass * 2.0 ** ((t - self._L) / PS.H_M)
        self.ev[d].add(t, evidence)
        if self.res is not None:
            # (day type, minute, source): the source lets a view fit the window of
            # one learned group's part of the node (lib/pwindows.part_when)
            self.res.offer((d, float(minute)) if src is None else (d, float(minute), str(src)),
                           max(mass, 1e-12), t, u)

    def density(self, daytype: int, t: Optional[float] = None) -> np.ndarray:
        """Mass shares over the 96 slots of a day type (zeros if empty)."""
        h = self.hist[1 if daytype else 0]
        s = h.sum()
        return h / s if s > 0 else np.zeros(96)

    def evidence(self, daytype: int, t: float, conf: bool = True) -> float:
        return float(self.ev[1 if daytype else 0].read(t)[-1 if conf else 0])

    def mass(self, daytype: int, t: float) -> float:
        if self._L is None:
            return 0.0
        return float(self.hist[1 if daytype else 0].sum() * 2.0 ** (-(t - self._L) / PS.H_M))

    def merge(self, other: "WhenSummary") -> "WhenSummary":
        if other._L is None:
            return self
        if self._L is None:
            self._L = other._L
        L = max(self._L, other._L)
        self.hist = self.hist * 2.0 ** (-(L - self._L) / PS.H_M) + \
            other.hist * 2.0 ** (-(L - other._L) / PS.H_M)
        self._L = L
        for a, b in zip(self.ev, other.ev):
            a.merge(b)
        return self

    def reset_confidence(self, t: float) -> None:
        for dv in self.ev:
            dv.set_entry(1, dv.get(0, t), t)

    def nbytes(self) -> int:
        b = self.hist.nbytes + 2 * 100
        if self.res is not None:
            b += 64 * len(self.res)
        return int(b)


# ============================================================== targets
class CatSummary:
    """Categorical / ordinal / ip target: DecayedSpaceSaving(16) with mass at
    H_s, H_m, H_l, evidence at H_m and on the confidence channel."""
    kind = "cat"
    __slots__ = ("ss",)

    def __init__(self, k: int = CAT_K) -> None:
        self.ss = _conf_ss(k)

    def update(self, v: Any, t: float, mass: float, evidence: float) -> None:
        self.ss.add(v, t, mass, evidence)

    def top(self, t: float, n: Optional[int] = None) -> List[Tuple[Hashable, float, float, float]]:
        return self.ss.items(t, PS.CH_M, n)

    def invariant(self, t: float, share: float = 0.995, n_min: float = 30.0) -> Optional[Hashable]:
        """The node invariant value (top value >= 99.5 % of mass with >= 30
        confidence-channel evidence, §6.4), else None."""
        items = self.ss.items(t, PS.CH_M, 1)
        if not items or self.ss.total_evidence(t) < n_min:
            return None
        k, c, g, _ = items[0]
        tot = self.ss.total(t)
        return k if tot > 0 and g / tot >= share else None

    def merge(self, other: "CatSummary") -> "CatSummary":
        self.ss.merge(other.ss)
        return self

    def reset_confidence(self, t: float) -> None:
        self.ss.reset_confidence(t)

    def nbytes(self) -> int:
        return self.ss.nbytes()


class NumSummary:
    """Numeric target (§5.5.5, §6.10): t-digest (H_m) of the transformed value
    y (log v when `log`), moments (w, wy, wy^2) at H_s, H_m, H_l, a daily ring
    of (day, min, max, n_obs) over RING_DAYS days (n_obs = evidence units), and
    lower / upper exceedance reservoirs (values beyond the current Q(0.10) /
    Q(0.90)) for GPD tails."""
    kind = "num"
    __slots__ = ("log", "td", "mom", "ring", "lo_res", "hi_res", "seg_day", "_thr", "_thr_n")
    THR_EVERY = 64                     # refresh the exceedance thresholds every 64 updates

    def __init__(self, log: bool = False, seed: int = 0) -> None:
        self.log = bool(log)
        self.td = PS.TDigest(50.0, PS.H_M)
        self.mom = PS.DecayedVector([PS.H_S] * 3 + [PS.H_M] * 3 + [PS.H_L] * 3)
        self.ring = np.full((RING_DAYS, 4), np.nan)       # day, min, max, n_obs (y scale)
        self.lo_res = PS.WeightedReservoir(EXC_RES, PS.H_M, seed)
        self.hi_res = PS.WeightedReservoir(EXC_RES, PS.H_M, seed + 1)
        self.seg_day: Optional[int] = None                  # first day of the confidence segment
        self._thr: Optional[Tuple[float, float]] = None     # cached (Q(0.10), Q(0.90))
        self._thr_n = 0

    def y(self, v: float) -> float:
        x = float(v)
        if self.log:
            return math.log(x) if x > 0 else -math.inf
        return x

    def update(self, v: Any, t: float, mass: float, evidence: float, day: Optional[int] = None,
               extreme: bool = True) -> None:
        """`extreme=False` (a row learned with outlier damping < 1 or from a
        low-trust source) updates the distribution (digest, moments) but not
        the hard-range ring or the exceedance reservoirs: one damped 12 KB
        login must not become the pattern's stated maximum."""
        try:
            y = self.y(v)
        except (TypeError, ValueError):
            return
        if not math.isfinite(y):
            return
        if not extreme:
            self.td.add(y, t, max(mass, 1e-12))
            self.mom.add(t, [mass, mass * y, mass * y * y] * 3)   # list: DecayedVector fast path
            return
        if self._thr_n % self.THR_EVERY == 0 and self.td.total() > 0:
            self._thr = (self.td.quantile(0.1), self.td.quantile(0.9)) \
                if self.td.n_centroids() > 4 else None
        self._thr_n += 1
        if self._thr is not None:
            q10, q90 = self._thr
            if y > q90:
                self.hi_res.offer(y, max(mass, 1e-12), t)
            elif y < q10:
                self.lo_res.offer(y, max(mass, 1e-12), t)
        self.td.add(y, t, max(mass, 1e-12))
        self.mom.add(t, [mass, mass * y, mass * y * y] * 3)   # list: DecayedVector fast path
        if day is not None:
            self._ring_add(int(day), y, evidence)

    def _ring_add(self, day: int, y: float, ev: float) -> None:
        if self.seg_day is None:
            self.seg_day = day
        i = day % RING_DAYS
        r = self.ring[i]
        if not (r[0] == day):
            r[:] = [day, y, y, 0.0]
        r[1] = min(r[1], y)
        r[2] = max(r[2], y)
        r[3] += ev

    def moments(self, t: float, ch: int = PS.CH_M) -> Tuple[float, float, float]:
        """(weight, mean, variance) of y at half-life channel ch."""
        v = self.mom.read(t)[3 * ch:3 * ch + 3]
        w = v[0]
        if w <= 0:
            return 0.0, math.nan, math.nan
        mu = v[1] / w
        return float(w), float(mu), float(max(0.0, v[2] / w - mu * mu))

    def observed_range(self, day_now: int) -> Tuple[float, float, float]:
        """(min, max, n_rng) over the ring's days of the current confidence
        segment within the last RING_DAYS days (§6.10)."""
        r = self.ring
        ok = np.isfinite(r[:, 0]) & (r[:, 0] > day_now - RING_DAYS) & (r[:, 0] <= day_now)
        if self.seg_day is not None:
            ok &= r[:, 0] >= self.seg_day
        if not ok.any():
            return math.nan, math.nan, 0.0
        return float(r[ok, 1].min()), float(r[ok, 2].max()), float(r[ok, 3].sum())

    def exceedances(self, upper: bool = True) -> np.ndarray:
        res = self.hi_res if upper else self.lo_res
        return np.asarray([it for it, _, _ in res.items()], dtype=np.float64)

    def merge(self, other: "NumSummary") -> "NumSummary":
        self.td.merge(other.td)
        self.mom.merge(other.mom)
        for i in range(RING_DAYS):
            a, b = self.ring[i], other.ring[i]
            if not np.isfinite(b[0]):
                continue
            if not np.isfinite(a[0]) or b[0] > a[0]:
                self.ring[i] = b
            elif b[0] == a[0]:
                a[1], a[2], a[3] = min(a[1], b[1]), max(a[2], b[2]), a[3] + b[3]
        for it, w, t in other.hi_res.items():
            self.hi_res.offer(it, w, t)
        for it, w, t in other.lo_res.items():
            self.lo_res.offer(it, w, t)
        return self

    def reset_confidence(self, t: float, day: Optional[int] = None) -> None:
        """New confidence segment: the observed range restarts (§6.10)."""
        self.seg_day = day

    def nbytes(self) -> int:
        return int(self.td.nbytes() + self.mom.nbytes() + self.ring.nbytes
                   + 2 * EXC_RES * 48 + 64)


class TextSummary:
    """Text target (§5.5.5, §6.11): shapes SS(8), exact values SS(16) (clear /
    hmac values only), log2 length histogram (16), charset-class counts (8).
    `update` takes the stored value; a phier.Shaped value (the policy kept
    only its shape) counts as its own shape and never as an exact value."""
    kind = "text"
    __slots__ = ("shapes", "values", "lens", "chars", "policy")
    CLASSES = ("L", "U", "D", "X", "sp", "punct", "quote", "other")

    def __init__(self, policy: str = "clear") -> None:
        self.policy = policy
        self.shapes = _conf_ss(TEXT_SHAPES_K)
        self.values = _conf_ss(TEXT_VALUES_K) if policy != "shape" else None
        self.lens = PS.DecayedVector([PS.H_M] * 16)
        self.chars = PS.DecayedVector([PS.H_M] * 8)

    @staticmethod
    def _char_vec(s: str) -> np.ndarray:
        v = np.zeros(8)
        cs = charset(s)
        for c in cs:
            if c in "LUDX":
                v["LUDX".index(c)] = 1
            elif c == " ":
                v[4] = 1
            elif c in "'\"`":
                v[6] = 1
            elif c.isprintable():
                v[5] = 1
            else:
                v[7] = 1
        return v

    def update(self, v: Any, t: float, mass: float, evidence: float) -> None:
        s = v if isinstance(v, str) else str(v)
        shaped = isinstance(s, Shaped) or self.policy == "shape"
        shp = s if shaped else shape(s)
        self.shapes.add(shp, t, mass, evidence)
        if self.values is not None and not shaped:
            self.values.add(s, t, mass, evidence)
        n = shape_length(s) if shaped else len(s)
        b = min(15, int(math.log2(n + 1)))
        lv = np.zeros(16)
        lv[b] = mass
        self.lens.add(t, lv)
        if not shaped:
            self.chars.add(t, mass * self._char_vec(s))

    def merge(self, other: "TextSummary") -> "TextSummary":
        self.shapes.merge(other.shapes)
        if self.values is not None and other.values is not None:
            self.values.merge(other.values)
        self.lens.merge(other.lens)
        self.chars.merge(other.chars)
        return self

    def reset_confidence(self, t: float) -> None:
        self.shapes.reset_confidence(t)
        if self.values is not None:
            self.values.reset_confidence(t)

    def nbytes(self) -> int:
        return int(self.shapes.nbytes() + (self.values.nbytes() if self.values else 0)
                   + self.lens.nbytes() + self.chars.nbytes())


class SetSummary:
    """Set target (§5.5.5, §6.11): SS(8) of templates (the caller passes the
    level-1 template) and SS(32) of element presence, plus the node's event
    mass / evidence for presence shares."""
    kind = "set"
    __slots__ = ("tpl", "elem", "n")

    def __init__(self) -> None:
        self.tpl = _conf_ss(SET_TPL_K)
        self.elem = _conf_ss(SET_ELEM_K)
        self.n = PS.DecayedVector(PS.HALF_LIVES)

    def update(self, v: Any, t: float, mass: float, evidence: float, template: Any = None) -> None:
        try:
            st = frozenset(v)
        except TypeError:
            return
        self.tpl.add(template if template is not None else st, t, mass, evidence)
        for x in list(st)[:64]:
            self.elem.add(x, t, mass, evidence)
        self.n.add(t, mass)

    def presence(self, t: float) -> Dict[Hashable, float]:
        tot = self.n.read(t)[PS.CH_M]
        if tot <= 0:
            return {}
        return {k: g / tot for k, _, g, _ in self.elem.items(t)}

    def merge(self, other: "SetSummary") -> "SetSummary":
        self.tpl.merge(other.tpl)
        self.elem.merge(other.elem)
        self.n.merge(other.n)
        return self

    def reset_confidence(self, t: float) -> None:
        self.tpl.reset_confidence(t)
        self.elem.reset_confidence(t)

    def nbytes(self) -> int:
        return int(self.tpl.nbytes() + self.elem.nbytes() + self.n.nbytes())


def new_summary(kind: str, policy: str = "clear", log: bool = False) -> Any:
    """Target summary for a registry type: 'num'/'numeric'/'time', 'text',
    'set', else categorical."""
    if kind in ("num", "numeric", "time"):
        return NumSummary(log)
    if kind == "text":
        return TextSummary(policy)
    if kind == "set":
        return SetSummary()
    return CatSummary()


# ======================================================= hold record
HOLD_N_REF = 300                   # "holds": coverage within the 3-sigma error of a 300-event check
HOLD_EPS_MIN = 0.02
HOLD_HALF_DAYS = 30.0              # the record forgets at H_l (the confidence channel's half-life)


def hold_eps(nominal: float) -> float:
    """Tolerance of "the constraint holds": its coverage on new data is at least
    nominal - eps, eps = max(0.02, 3 sqrt(nom (1 - nom) / 300)) - the sampling
    error of checking the statement on a few hundred fresh events (0.05 for a
    90 % band, 0.02 for a 99 % set)."""
    nom = min(max(float(nominal), 0.0), 1.0)
    return max(HOLD_EPS_MIN, 3.0 * math.sqrt(max(nom * (1.0 - nom), 1e-4) / HOLD_N_REF))


HOLD_BATCH_N = 100.0               # a held-out test is a batch of >= 100 checked events ...
HOLD_BATCH_S = 7 * DAY             # ... or of a week, whichever comes first (M46)
HOLD_TEST_HALF_DAYS = 14.0         # the test record forgets at two weeks


class HoldRecord:
    """Prequential record of how a node's statement held on NEW data (§6.9.4,
    the statement's confidence). Every learned, undamped event of a confident
    node is checked against the node's reference statement of the previous day
    (built from earlier data only, so every check is held-out), per constraint
    key (`who`, `when`, a band, a closed set, a grammar, a binding) with the
    nominal coverage the statement gives it.

    The checks are grouped into held-out TESTS (M46): a batch of >= HOLD_BATCH_N
    checked events, or a week of them. A test passes when every constraint
    checked in it holds - its coverage on the batch is >= nominal - hold_eps
    (3 sigma of the check a reader would make on 300 fresh events). The
    stated confidence p_hold() is the predictive probability that the next test
    passes, (passes + 1) / (tests + 2) over the tests of the last weeks
    (forward-decayed at HOLD_TEST_HALF_DAYS): a frequency of the statement
    holding on held-out data - calibrated by construction (Beta-Bernoulli
    predictive), rising with every test that passes, falling with every one
    that fails. Before M46 p_hold was the product over the constraints of each
    one's posterior tail P(theta_c >= nom_c - eps): with 10-20 constraints the
    product was ~0 for nearly every statement (pack O median 0.005 on day 21,
    ECE 0.37-0.41) although 35-45 % of them held; that product is kept as
    p_constraints() (diagnosis). O(constraints) floats per node."""

    __slots__ = ("c", "L", "b", "bt0", "bn", "blast", "T")

    def __init__(self) -> None:
        self.c: Dict[str, List[float]] = {}          # key -> [n, hits, nominal] (forward-decayed)
        self.L: Optional[float] = None
        self.b: Dict[str, List[float]] = {}          # current batch: key -> [n, hits, nominal]
        self.bt0: Optional[float] = None             # batch start
        self.bn = 0.0                                # evidence units checked in the batch
        self.blast: Optional[float] = None           # time of the batch's last event
        self.T = [0.0, 0.0]                          # [passes, tests] (forward-decayed, landmark L)

    def _f(self, t: float) -> float:
        if self.L is None:
            self.L = float(t)
        x = (float(t) - self.L) / (HOLD_HALF_DAYS * DAY)
        if x > 60.0:                                  # rescale the landmark
            g = 2.0 ** (-x)
            for v in self.c.values():
                v[0] *= g
                v[1] *= g
            gt = 2.0 ** (-(float(t) - self.L) / (HOLD_TEST_HALF_DAYS * DAY))
            T = self._T()
            T[0] *= gt
            T[1] *= gt
            self.L = float(t)
            x = 0.0
        return 2.0 ** x

    def _T(self) -> List[float]:
        T = getattr(self, "T", None)
        if T is None:
            T = self.T = [0.0, 0.0]
            self.b, self.bt0, self.bn, self.blast = {}, None, 0.0, None
        return T

    def _close(self) -> None:
        """Close the current batch: one held-out test of the statement."""
        T = self._T()
        if self.b and self.bn > 0 and self.blast is not None:
            # the batch's coverage against the threshold of a HOLD_N_REF-event check
            # (the evaluator's / a reader's test): a lenient small-batch tolerance
            # would make the many small tests of a rare pattern pass by default
            ok = all(h / n >= nom - hold_eps(nom) for n, h, nom in self.b.values() if n > 0)
            g = 2.0 ** ((self.blast - (self.L if self.L is not None else self.blast)) / (HOLD_TEST_HALF_DAYS * DAY))
            T[1] += g
            if ok:
                T[0] += g
        self.b, self.bt0, self.bn, self.blast = {}, None, 0.0, None

    def add(self, key: str, hit: bool, nominal: float, t: float, w: float = 1.0) -> None:
        t = float(t)
        self._T()
        # a new event (later time) may close the batch first: the keys checked on
        # one event always belong to one test
        if self.bt0 is not None and self.blast is not None and t > self.blast and (
                self.bn >= HOLD_BATCH_N or t - self.bt0 >= HOLD_BATCH_S):
            self._close()
        f = self._f(t) * float(w)
        v = self.c.get(key)
        if v is None:
            v = self.c[key] = [0.0, 0.0, float(nominal)]
        v[0] += f
        if hit:
            v[1] += f
        v[2] = float(nominal)
        if self.bt0 is None:
            self.bt0 = t
        if self.blast is None or t > self.blast:
            self.bn += float(w)                       # one unit per event, not per key
            self.blast = t
        bv = self.b.get(key)
        if bv is None:
            bv = self.b[key] = [0.0, 0.0, float(nominal)]
        bv[0] += float(w)
        if hit:
            bv[1] += float(w)
        bv[2] = float(nominal)

    def counts(self, t: float) -> Dict[str, Tuple[float, float, float]]:
        if self.L is None:
            return {}
        g = 2.0 ** (-(float(t) - self.L) / (HOLD_HALF_DAYS * DAY))
        return {k: (v[0] * g, v[1] * g, v[2]) for k, v in self.c.items()}

    def tests(self, t: float) -> Tuple[float, float]:
        """(passes, tests), decayed to t."""
        T = self._T()
        if self.L is None:
            return 0.0, 0.0
        g = 2.0 ** (-(float(t) - self.L) / (HOLD_TEST_HALF_DAYS * DAY))
        return T[0] * g, T[1] * g

    def p_hold(self, t: float) -> float:
        """Predictive probability that the statement holds on the next held-out
        test: (passes + 1) / (tests + 2); NaN before the first test."""
        ps, n = self.tests(t)
        if n <= 0:
            return float("nan")
        return float((ps + 1.0) / (n + 2.0))

    def p_constraints(self, t: float, eps: Optional[float] = None, keys: Optional[Iterable[str]] = None) -> float:
        """P(every constraint's coverage on new data >= nominal - eps) under
        independent Beta posteriors (eps per constraint: hold_eps(nominal));
        NaN before any check. The M30 confidence, kept for diagnosis."""
        from scipy.special import betainc
        cs = self.counts(t)
        if keys is not None:
            ks = set(keys)
            cs = {k: v for k, v in cs.items() if k in ks}
        if not cs:
            return float("nan")
        p = 1.0
        for n, h, nom in cs.values():
            x = min(1.0, max(0.0, nom - (hold_eps(nom) if eps is None else eps)))
            a, b = h + 1.0, max(0.0, n - h) + 1.0
            p *= float(1.0 - betainc(a, b, x)) if x > 0 else 1.0
        return float(p)

    def drop(self, keys: Optional[Iterable[str]] = None) -> None:
        """Forget the record (all of it, or the given constraints): an accepted
        change (§6.9.2) or a structural change starts a new confidence segment
        (all of it: the tests too); a re-stated constraint (M45) restarts its
        own counts and leaves the current batch's test."""
        if keys is None:
            self.c.clear()
            self.L = None
            self.b, self.bt0, self.bn, self.blast = {}, None, 0.0, None
            self.T = [0.0, 0.0]
            return
        self._T()
        for k in keys:
            self.c.pop(k, None)
            self.b.pop(k, None)

    def nbytes(self) -> int:
        return int(64 + 80 * len(self.c) + 80 * len(getattr(self, "b", None) or ()) + 64)


# ============================================================ bindings
class PairSketch:
    """Binding pair counts (X, Y) at a node (§6.12): SpaceSaving(64) over x
    (mass at H_l, evidence at H_m and on the confidence channel), each tracked
    x with a SpaceSaving(4) over y. An evicted x loses its y table."""

    __slots__ = ("x", "y")

    def __init__(self, k_x: int = PAIR_KX, k_y: int = PAIR_KY) -> None:
        self.x = PS.DecayedSpaceSaving(k_x, (PS.H_L,), PS.EV_HALF_LIVES, 0)
        self.y: Dict[Hashable, PS.DecayedSpaceSaving] = {}

    def update(self, xv: Hashable, yv: Hashable, t: float, mass: float, evidence: float) -> None:
        before = set(self.y.keys())
        self.x.add(xv, t, mass, evidence)
        if xv not in self.x:
            return
        tab = self.y.get(xv)
        if tab is None:
            for gone in before - set(self.x.keys()):
                self.y.pop(gone, None)
            tab = self.y[xv] = PS.DecayedSpaceSaving(PAIR_KY, (PS.H_L,), PS.EV_HALF_LIVES, 0)
        tab.add(yv, t, mass, evidence)

    def table(self, t: float) -> Dict[Hashable, List[Tuple[Hashable, float, float]]]:
        """{x: [(y, evidence_conf, guaranteed mass)]} for tracked x."""
        out = {}
        for xv in self.x.keys():
            tab = self.y.get(xv)
            if tab is None:
                continue
            out[xv] = [(k, e, g) for k, _, g, e in tab.items(t, 0)]
        return out

    def reset_confidence(self, t: float) -> None:
        self.x.reset_confidence(t)
        for tab in self.y.values():
            tab.reset_confidence(t)

    def merge(self, other: "PairSketch") -> "PairSketch":
        self.x.merge(other.x)
        for xv, tab in other.y.items():
            if xv in self.x:
                if xv in self.y:
                    self.y[xv].merge(tab)
                else:
                    self.y[xv] = tab.copy()
        for xv in list(self.y):
            if xv not in self.x:
                del self.y[xv]
        return self

    def nbytes(self) -> int:
        return int(self.x.nbytes() + sum(t.nbytes() for t in self.y.values()))


# ================================================================ node
class Node:
    """One pattern (§5.5.2). Summaries are created lazily; everything not
    listed as a summary is a plain field for P04 / P03 / fitters.

    ctx      tuple of (attr, level, values frozenset, negated) from the root
    split    ptree.Split or None;  exc {ip: nid};  is_exc
    state    candidate | confirmed | stable | evolving | stale | retired | dormant
    days     64-bit bitmap of the last 64 local dates with events + total distinct dates
    mass     forward-decayed mass at H_s, H_m, H_l;  n_eff evidence at H_s, H_m, H_l
             (n_eff alone = H_m; n_c = the H_l entry = confidence channel)
    targets  {attr: Summary} (<= m_t);  inv {attr: (level, value)}
    xstats   pevalue.ExcTracker | None;  split_stats pevalue.SplitStats | None
    pairs    {(X, Y): PairSketch};  adwin psketch.ADWIN | None;  ph {attr: PageHinkley}
    """

    __slots__ = ("id", "parent", "depth", "kind", "ctx", "split", "exc", "is_exc", "state",
                 "created", "first_seen", "last_seen", "days_bits", "days_total", "last_day",
                 "version", "cver", "mass", "n_eff", "seg_start", "who", "when", "targets",
                 "rate_iph", "inv", "split_stats", "xstats", "pairs", "adwin", "ph", "alt", "ref",
                 "since_check", "meta")

    def __init__(self, nid: int, parent: Optional[int], depth: int, kind: int,
                 ctx: Tuple[Tuple[str, int, frozenset, bool], ...], t: float,
                 is_exc: bool = False) -> None:
        self.id = int(nid)
        self.parent = parent
        self.depth = int(depth)
        self.kind = int(kind)
        self.ctx = tuple(ctx)
        self.split = None
        self.exc: Dict[str, int] = {}
        self.is_exc = bool(is_exc)
        self.state = "candidate"
        self.created = float(t)
        self.first_seen: Optional[float] = None
        self.last_seen: Optional[float] = None
        self.days_bits = 0
        self.days_total = 0
        self.last_day: Optional[int] = None
        self.version = 1
        self.cver = 0
        self.mass = PS.DecayedVector(PS.HALF_LIVES)
        self.n_eff = PS.DecayedVector(PS.HALF_LIVES)
        self.seg_start = float(t)
        self.who = WhoSummary()
        self.when = WhenSummary()
        self.targets: Dict[str, Any] = {}
        self.rate_iph: Optional[PS.TDigest] = None
        self.inv: Dict[str, Tuple[int, Any]] = {}
        self.split_stats = None
        self.xstats = None
        self.pairs: Dict[Tuple[str, str], PairSketch] = {}
        self.adwin: Optional[PS.ADWIN] = None
        self.ph: Dict[str, PS.PageHinkley] = {}
        self.alt: Optional[int] = None
        self.ref: Optional[Dict[str, Any]] = None
        self.since_check = 0.0
        self.meta: Dict[str, Any] = {}

    # ------------------------------------------------------- plain data
    def to_plain(self) -> Dict[str, Any]:
        """A light JSON-able view for snapshots and reports (eval runner
        _jsonable uses it: Node has __slots__, so it has no __dict__). Working
        state (split statistics, exception trackers, pair sketches, drift
        detectors) is left out on purpose."""
        sp = self.split
        split = None
        if sp is not None:
            split = {"attr": sp.attr, "level": int(sp.level),
                     "groups": [sorted(map(str, g)) for g in sp.groups],
                     "children": [int(c) for c in sp.children], "other": sp.other}
        return {"id": self.id, "parent": self.parent, "depth": self.depth, "kind": self.kind,
                "ctx": [[a, int(l), sorted(map(str, vs)), bool(neg)] for a, l, vs, neg in self.ctx],
                "split": split, "exc": dict(self.exc), "is_exc": self.is_exc, "state": self.state,
                "created": self.created, "first_seen": self.first_seen, "last_seen": self.last_seen,
                "days_total": self.days_total, "version": self.version, "cver": self.cver,
                "targets": sorted(self.targets), "inv": {k: [int(v[0]), str(v[1])] for k, v in self.inv.items()},
                "alt": self.alt, "p_hold": self.p_hold(),
                "hold": {k: [round(n, 2), round(h, 2), round(nom, 4)]
                         for k, (n, h, nom) in (self.hold.counts(self.last_seen or self.created).items()
                                                if self.hold is not None else ())},
                "hold_tests": ([round(x, 2) for x in self.hold.tests(self.last_seen or self.created)]
                               if self.hold is not None else None)}

    # ------------------------------------------------------- confidence
    @property
    def hold(self) -> Optional[HoldRecord]:
        return self.meta.get("hold") if self.meta else None

    def p_hold(self, t: Optional[float] = None) -> Optional[float]:
        """Calibrated confidence of the node's statement (HoldRecord.p_hold):
        the probability that all of its constraints hold on new data; None
        before the first held-out check."""
        hr = self.hold
        if hr is None:
            return None
        p = hr.p_hold(self.last_seen or self.created if t is None else t)
        return None if p != p else round(p, 4)

    # ----------------------------------------------------------- updates
    def touch_day(self, day: int) -> None:
        """Record activity on local date ordinal `day` (64-day bitmap)."""
        if self.last_day is None:
            self.days_bits = 1
            self.days_total = 1
            self.last_day = day
            return
        if day == self.last_day:
            return
        if day > self.last_day:
            sh = day - self.last_day
            self.days_bits = ((self.days_bits << sh) | 1) & ((1 << 64) - 1) if sh < 64 else 1
            self.days_total += 1
            self.last_day = day
        else:                                           # late event of an earlier day
            back = self.last_day - day
            if back < 64 and not (self.days_bits >> back) & 1:
                self.days_bits |= 1 << back
                self.days_total += 1

    def n_days(self, window: int = 64) -> int:
        """Distinct active dates among the last `window` (<= 64) of the bitmap."""
        mask = (1 << min(64, window)) - 1
        return bin(self.days_bits & mask).count("1")

    def update_core(self, t: float, mass: float, evidence: float, who_keys: Sequence[Any],
                    ip: str, daytype: Optional[int] = None, minute: Optional[float] = None,
                    day: Optional[int] = None, who_code: Optional[np.ndarray] = None,
                    u: Optional[float] = None, suspicious: bool = False) -> None:
        """Mass, evidence, who, when and dates: every node on the path (§6.5.1).
        A suspicious row (P03 learned it damped as a foreign source), and every
        row of a source already suspect at this node, is kept out of the who
        summary (mass, evidence, unseen-source estimate, heavy sets): a damped
        source must not become part of the pattern's who by repetition."""
        t = float(t)
        self.mass.add(t, mass)
        self.n_eff.add(t, evidence)
        self.first_seen = t if self.first_seen is None else min(self.first_seen, t)
        self.last_seen = t if self.last_seen is None else max(self.last_seen, t)
        if suspicious or self.who.is_suspect(ip, t):
            self.who.mark_suspect(ip, t)
        else:
            self.who.update(who_keys, ip, t, mass, evidence, who_code)
        if daytype is not None and minute is not None:
            self.when.update(daytype, minute, t, mass, evidence, u, ip)
        if day is not None:
            self.touch_day(int(day))

    def target(self, attr: str, kind: str = "cat", policy: str = "clear", log: bool = False) -> Any:
        s = self.targets.get(attr)
        if s is None:
            s = self.targets[attr] = new_summary(kind, policy, log)
        return s

    def update_target(self, attr: str, v: Any, t: float, mass: float, evidence: float,
                      kind: str = "cat", policy: str = "clear", log: bool = False,
                      day: Optional[int] = None, template: Any = None, extreme: bool = True) -> None:
        s = self.target(attr, kind, policy, log)
        if isinstance(s, NumSummary):
            s.update(v, t, mass, evidence, day, extreme)
        elif isinstance(s, SetSummary):
            s.update(v, t, mass, evidence, template)
        else:
            s.update(v, t, mass, evidence)

    def pair(self, x_attr: str, y_attr: str) -> PairSketch:
        k = (x_attr, y_attr)
        p = self.pairs.get(k)
        if p is None:
            p = self.pairs[k] = PairSketch()
        return p

    # ------------------------------------------------------------- reads
    def n_c(self, t: float) -> float:
        """Evidence on the confidence channel (H_l entry of n_eff)."""
        return self.n_eff.get(PS.CH_L, t)

    def n_m(self, t: float) -> float:
        return self.n_eff.get(PS.CH_M, t)

    def mass_at(self, t: float, ch: int = PS.CH_M) -> float:
        return self.mass.get(ch, t)

    def rate_hs(self, t: float) -> float:
        """H_s-decayed mass per second (§6.5.1 r_target comparison)."""
        return self.mass.get(PS.CH_S, t) * math.log(2.0) / PS.H_S

    @property
    def confident(self) -> bool:
        return self.state in CONFIDENT_STATES

    # ------------------------------------------------------ confidence
    def reset_confidence(self, t: float, attrs: Optional[Iterable[str]] = None,
                         day: Optional[int] = None) -> None:
        """§6.9.4: reset the confidence channel to the H_m state for the
        changed attributes (all, and the node's own evidence, when attrs is
        None, i.e. a structural change)."""
        hr = self.hold
        if attrs is None:
            self.n_eff.set_entry(PS.CH_L, self.n_eff.get(PS.CH_M, t), t)
            self.who.reset_confidence(t)
            self.when.reset_confidence(t)
            self.seg_start = float(t)
            attrs = list(self.targets.keys())
            if hr is not None:
                hr.drop()
        elif hr is not None:
            hr.drop([("when" if a in ("@when", "ctx.tod_min", "ctx.when") else a) for a in attrs])
        for a in attrs:
            s = self.targets.get(a)
            if isinstance(s, NumSummary):
                s.reset_confidence(t, day)
            elif s is not None:
                s.reset_confidence(t)

    # --------------------------------------------------------- merging
    def absorb(self, other: "Node") -> None:
        """Merge another node's summaries into this one (prune / sibling merge)."""
        self.mass.merge(other.mass)
        self.n_eff.merge(other.n_eff)
        self.who.merge(other.who)
        self.when.merge(other.when)
        for a, s in other.targets.items():
            mine = self.targets.get(a)
            if mine is None:
                self.targets[a] = s
            elif type(mine) is type(s):
                mine.merge(s)
        for k, p in other.pairs.items():
            if k in self.pairs:
                self.pairs[k].merge(p)
            else:
                self.pairs[k] = p
        self.days_bits |= other.days_bits
        self.days_total = max(self.days_total, other.days_total)
        for tt in (other.first_seen,):
            if tt is not None:
                self.first_seen = tt if self.first_seen is None else min(self.first_seen, tt)
        if other.last_seen is not None:
            self.last_seen = other.last_seen if self.last_seen is None else max(self.last_seen,
                                                                               other.last_seen)

    def nbytes(self) -> int:
        b = 400 + self.who.nbytes() + self.when.nbytes() + self.mass.nbytes() + self.n_eff.nbytes()
        b += sum(s.nbytes() for s in self.targets.values())
        b += sum(p.nbytes() for p in self.pairs.values())
        if self.split_stats is not None:
            b += self.split_stats.nbytes()
        if self.xstats is not None:
            b += self.xstats.nbytes()
        if self.rate_iph is not None:
            b += self.rate_iph.nbytes()
        if self.adwin is not None:
            b += self.adwin.nbytes()
        return int(b)


def pattern_id(tree_key: str, kind: int, nid: int, version: int, cver: int) -> str:
    """p:<tree key>:<kind>:<nid>@<version>.<cver> (§6.8.2)."""
    return f"p:{tree_key}:{int(kind)}:{int(nid)}@{int(version)}.{int(cver)}"


def parse_pattern_id(pid: str) -> Tuple[str, int, int, int, int]:
    body = pid[2:] if pid.startswith("p:") else pid
    head, ver = body.rsplit("@", 1)
    key, kind, nid = head.rsplit(":", 2)
    v, c = ver.split(".", 1)
    return key, int(kind), int(nid), int(v), int(c)
