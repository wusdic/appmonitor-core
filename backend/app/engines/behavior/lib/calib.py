"""Conformal calibration rings with EVT tails (B24 detector rings, B25 meta rings).

STATUS: implemented. Dataclass layouts, signatures and maths are frozen
(docs/lib3/helpers_api.md).

A Ring is the per (key, detector, stratum) calibration sample: up to M = 256
null scores, kept sorted for O(log M) conformal p-values, each carrying the
tick ts it came from so that governance can delete everything after a
rollback onset (contract H: 'ring entries carry ts'). Beyond the empirical
range (s > q_0.90 with >= 10 exceedances) a PWM-GPD tail extrapolates, which
is what lets a 256-entry ring produce p ~ 1e-6 honestly.

Strata (Mondrian): daypart(4) x cadence class {60, 300, 900, 3600}; identity
uses (daypart, regime tercile). New strata (< 64 entries) logit-blend with a
model p (behavior.pm) so a cadence switch does not produce garbage p-values.

Serialisation: Ring.to_dict() -> {'scores': list[float32], 'ts': list[float],
'gpd': {...} | None} so model.calib stays JSON-like for put_model.

Why copy-on-write arrays: MetricStore keeps models *by reference* and
learners checkpoint their state, so a Ring that mutated its arrays in place
would silently rewrite every snapshot that shares them (replay, rollback).
Every mutation therefore builds fresh arrays (~1.5 us each at M = 256, one
searchsorted plus slice copies, vs ~8 us for np.insert): a shallow copy of a
Ring, or a reference to its arrays, is an immutable snapshot.

Why float32 rounding on both sides: the randomised conformal p counts exact
ties with ==, so a score must be rounded the same way when stored and when
scored (combine.randomized_conformal_p). Rounding uses struct (~0.2 us)
rather than np.float32 because it is on the per-tick hot path.

Why a scalar GPD survival function here: p_from_ring runs per (key,
detector) per tick on one value, where evt.gpd_sf's ndarray path costs more
in numpy overhead than the arithmetic; the maths is identical (tested
against evt.gpd_sf). The tail *fit* runs every GPD_REFIT_TICKS and uses
evt.gpd_pwm_fit directly.

Two guards on the plug-in tail (measured on a streaming M = 256 ring,
refit every 16 ticks, 2e5 null ticks; realised rate / nominal):
  * xi >= 0 for extrapolation. With ~26 exceedances the PWM xi has sd
    ~0.23, so an exponential-tailed null (-log p scores, CUSUMs: xi = 0)
    often fits xi < 0, i.e. a finite end point u + sigma/|xi| that the
    next null tick can cross -> p = 1e-300 (a CRITICAL from pure noise).
    Spec-exact: 4.7x at p <= 3e-4 and 44x at 2e-5 for Exp(1). Replacing a
    negative xi by the exponential tail with the same mean excess gives
    1.5x / 2.5x; Gaussian-like (truly bounded-ish) nulls become
    conservative instead. Heavy tails are unchanged (xi > 0 is kept) and
    remain ~3x anti-conservative at 3e-4 from xi sampling noise.
  * p <= 1/(n+1) beyond the ring maximum: the ring already witnesses that
    s exceeds all n null draws (conformal p < 1/(n+1)), so the tail must
    not report more. Matters for bounded scores (JSD) where the clipped
    xi = -0.5 end point lies beyond the true bound.
  * (round 4, evaluator) tail p >= rate 10^-(s - u), i.e. an exponential
    floor of scale 1/ln 10 (P_SCORE_SIGMA), when the score
    is -log10 of the detector's own p-value pm (m_calib.tail_sigma_min):
    the issued p never decays faster beyond u than pm itself, i.e. B24 may
    make a detector p more conservative in the tail, never more extreme
    than its own model says relative to u. Why: the rings of the H-stream
    detectors turn over in weeks (24 rows a day over 4 dayparts, M = 256),
    so live scores are compared with warm-up scores for the whole of a
    pack, and those were compressed - a young model's honest predictive
    (B06's cross-fitted null, B16's held-out typicality) is heavier than
    the mature one's. The PWM tail of a compressed ring is steep and was
    extrapolated beyond the ring maximum: pack A seed 0, clean control
    ticks, x nominal at p < 1e-3, pm -> issued p: t2 2.7 -> 29, spe 2.1 ->
    12, timing 8.2 -> 8.7 (58 of 102 tail hits beyond the ring maximum);
    replayed with the floor: t2 2.7, spe 1.4, timing 3.2, budget_vol 1.8.
    For an exact null (pm uniform) the scale of the excess is exactly
    1/ln 10, so the floor binds only through sampling noise
    (tests/lib/test_calib_sigma_floor.py).

Round 4: B24 and B25 fit robust_tail - at most TRIM_EPS (2 %) of the ring
trimmed as contamination (trim_count) and the predictive shape floor
xi >= 1/n_u - because their rings now admit every row of a trusted period
whatever its own score (lib/gating.period_weight). fit_tail (xi >= 0,
optional winsorisation) is kept for other callers.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import combine, evt

RING_M = 256            # max entries per ring
SMALL_N = 64            # below this a ring blends with the model / pooled p
TAIL_Q = 0.90           # GPD threshold u = q_0.90(C)
MIN_EXCEED = 10         # exceedances needed before the GPD tail is used
GPD_REFIT_TICKS = 16    # refit stride
KS_MAX_D = 0.05         # health: KS D above this halves the detector weight
RATE_RATIO_BAND = (0.5, 2.0)

XI_FLOOR = 0.0          # fit_tail: xi below this -> exponential tail (module docstring)
P_FLOOR = 1e-300        # tail p floor (same as combine.P_FLOOR)
P_SCORE_SIGMA = 1.0 / math.log(10.0)   # tail scale of -log10 U: floor for p-scores (docstring)
_XI_ZERO = 1e-9         # |xi| below this is the exponential limit (as evt.gpd_sf)
_KEY_SEP = "@"          # ring_key separator: '<detector>@<stratum>'
_STRATUM_SEP = "|"      # stratum separator: 'daypart|cc'
_EMPTY = np.empty(0, dtype=np.float64)
_EMPTY.setflags(write=False)
_F32 = struct.Struct("f")


def _f(x: Any) -> float:
    """float(x) with None -> NaN."""
    return math.nan if x is None else float(x)


def _r32(x: Any) -> float:
    """x rounded to float32 and widened back to a Python float.

    NaN / None -> NaN; magnitudes beyond the float32 range -> +-inf (struct
    raises OverflowError where numpy would warn).
    """
    if x is None:
        return math.nan
    x = float(x)
    try:
        return _F32.unpack(_F32.pack(x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def _r32_array(a: Any) -> np.ndarray:
    """Vectorised _r32 for normalisation paths (quiet on overflow)."""
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(a, dtype=np.float64).ravel().astype(np.float32).astype(np.float64)


# ------------------------------------------------------------------ tail
@dataclass(slots=True)
class GPDTail:
    """A fitted tail: P(S > s) = rate * (1 + xi (s - u)/sigma)^(-1/xi) for s > u."""
    u: float
    xi: float
    sigma: float
    rate: float             # N_u / N (exceedance fraction at fit time)
    n: int                  # ring size at fit time
    fitted_ts: float = float("nan")

    def valid(self) -> bool:
        """All parameters usable: finite u, xi, sigma > 0 and rate in (0, 1]."""
        return (math.isfinite(self.u) and math.isfinite(self.xi)
                and math.isfinite(self.sigma) and self.sigma > 0.0
                and 0.0 < self.rate <= 1.0)

    def sf(self, s: float) -> float:
        """rate * gpd_sf(s - u), floored at P_FLOOR; s <= u returns rate (the body edge).

        NaN s or an unusable tail (not valid()) -> NaN: a degraded input must
        never come out as the floor p (max(P_FLOOR, nan) would be 1e-300).
        """
        s = _f(s)
        if s != s or not self.valid():
            return math.nan
        return max(P_FLOOR, self.rate * _gpd_sf(s - self.u, self.xi, self.sigma))

    def to_dict(self) -> Dict[str, Any]:
        ft = self.fitted_ts
        return {"u": float(self.u), "xi": float(self.xi), "sigma": float(self.sigma),
                "rate": float(self.rate), "n": int(self.n),
                "fitted_ts": None if ft != ft else float(ft)}   # NaN is not JSON

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GPDTail":
        ft = d.get("fitted_ts")
        return cls(u=float(d["u"]), xi=float(d["xi"]), sigma=float(d["sigma"]),
                   rate=float(d["rate"]), n=int(d.get("n", 0)),
                   fitted_ts=math.nan if ft is None else float(ft))


def _gpd_sf(y: float, xi: float, sigma: float) -> float:
    """Scalar evt.gpd_sf: (1 + xi y/sigma)^(-1/xi) in log space; y <= 0 -> 1;
    |xi| < 1e-9 -> exp(-y/sigma); 0 beyond the upper end point when xi < 0.
    Invalid parameters (non-finite xi, sigma <= 0 or non-finite) give NaN for
    y > 0, as evt.gpd_sf does."""
    if y != y:
        return math.nan
    if y <= 0.0:
        return 1.0
    if not (math.isfinite(xi) and 0.0 < sigma < math.inf):
        return math.nan
    z = y / sigma
    if abs(xi) < _XI_ZERO:
        return math.exp(-z)
    t = xi * z
    if t <= -1.0:
        return 0.0
    return math.exp(-math.log1p(t) / xi)


# ------------------------------------------------------------------ ring
@dataclass(slots=True)
class Ring:
    """Sorted calibration ring. Invariant: scores ascending, ts aligned, len <= cap.

    Eviction on overflow removes the entry with the OLDEST ts (FIFO in time,
    not in score) so the ring tracks the recent null. Scores are stored as
    float32-rounded float64 so ties compare consistently.

    Among entries with equal oldest ts the lowest score goes first, which
    makes bulk eviction (from_dict, seed_ring) and repeated add() agree.
    Arrays are replaced, never written in place (see module docstring).
    Construction normalises its inputs: float32 rounding, non-finite scores
    or ts dropped, sorted by score, oldest-first eviction down to cap.
    """
    cap: int = RING_M
    scores: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    ts: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    gpd: Optional[GPDTail] = None

    def __post_init__(self) -> None:
        cap = int(self.cap)
        if cap < 1:
            raise ValueError(f"Ring: cap={self.cap!r} must be >= 1")
        self.cap = cap
        if np.size(self.scores) or np.size(self.ts):     # lists are accepted too
            self.scores, self.ts = _normalise(self.scores, self.ts, cap)
        else:
            self.scores, self.ts = _EMPTY, _EMPTY

    def __len__(self) -> int:
        return int(self.scores.size)

    def add(self, score: float, ts: float) -> None:
        """Insert (score, ts) keeping order (np.searchsorted + insert); NaN is ignored.
        Evicts the oldest-ts entry when len > cap. O(M).

        Non-finite scores (NaN, +-inf, or beyond the float32 range) and a
        non-finite ts are ignored: they would poison quantiles / the GPD fit
        or could never be evicted / rolled back. An entry older than every
        entry of a full ring is itself the one evicted (no change). The GPD
        fit is kept: it is refitted on its own stride (GPD_REFIT_TICKS).
        """
        s = _r32(score)
        if not math.isfinite(s):
            return
        t = _f(ts)
        if not math.isfinite(t):
            return
        sc = self.scores
        tt = self.ts
        n = sc.size
        i = int(sc.searchsorted(s, "right"))
        if n < self.cap:
            new_s = np.empty(n + 1, dtype=np.float64)
            new_t = np.empty(n + 1, dtype=np.float64)
            new_s[:i] = sc[:i]
            new_s[i] = s
            new_s[i + 1:] = sc[i:]
            new_t[:i] = tt[:i]
            new_t[i] = t
            new_t[i + 1:] = tt[i:]
            self.scores, self.ts = new_s, new_t
            return
        if n > self.cap:            # cap lowered after construction: bulk path
            self.scores, self.ts = _evict_to_cap(np.insert(sc, i, s), np.insert(tt, i, t), self.cap)
            return
        # Full: conceptually insert at i then evict the first index with the
        # minimum ts. Old j sits at j (j < i) or j + 1 (j >= i) after insertion.
        j = int(tt.argmin())
        tj = float(tt[j])
        if t < tj or (t == tj and j >= i):
            return                  # the new entry is the oldest: evicted at once
        new_s = np.empty(n, dtype=np.float64)
        new_t = np.empty(n, dtype=np.float64)
        if j < i:                   # drop j, shift (j, i) left, new at i - 1
            new_s[:j] = sc[:j]
            new_s[j:i - 1] = sc[j + 1:i]
            new_s[i - 1] = s
            new_s[i:] = sc[i:]
            new_t[:j] = tt[:j]
            new_t[j:i - 1] = tt[j + 1:i]
            new_t[i - 1] = t
            new_t[i:] = tt[i:]
        else:                       # new at i, shift [i, j) right, drop j
            new_s[:i] = sc[:i]
            new_s[i] = s
            new_s[i + 1:j + 1] = sc[i:j]
            new_s[j + 1:] = sc[j + 1:]
            new_t[:i] = tt[:i]
            new_t[i] = t
            new_t[i + 1:j + 1] = tt[i:j]
            new_t[j + 1:] = tt[j + 1:]
        self.scores, self.ts = new_s, new_t

    def remove_after(self, ts: float) -> int:
        """Delete every entry with entry.ts > ts (rollback); returns the number removed.
        Drops the GPD fit if anything was removed.

        NaN ts removes nothing (a malformed onset must not wipe a ring).
        """
        t = _f(ts)
        if t != t or self.ts.size == 0:
            return 0
        keep = self.ts <= t
        n_removed = int(keep.size - np.count_nonzero(keep))
        if n_removed:
            self.scores = self.scores[keep]
            self.ts = self.ts[keep]
            self.gpd = None
        return n_removed

    def p_value(self, score: float, u: float) -> float:
        """Randomised conformal p (combine.randomized_conformal_p) on this ring only.

        score is float32-rounded first, exactly as add() stores it. NaN score
        or u -> NaN; empty ring -> u.
        """
        return _conformal_p(self.scores, _r32(score), u)

    def quantile(self, q: float) -> float:
        """Empirical quantile (numpy 'linear'); NaN on an empty ring.

        NaN q -> NaN; q outside [0, 1] raises ValueError (numpy's).
        """
        q = float(q)
        if self.scores.size == 0 or q != q:
            return math.nan
        return float(np.quantile(self.scores, q))

    def reset(self) -> None:
        """Empty the ring (model.control version change)."""
        self.scores = _EMPTY
        self.ts = _EMPTY
        self.gpd = None

    def to_dict(self) -> Dict[str, Any]:
        """{'scores', 'ts', 'gpd'} of Python floats; 'cap' only when it is not RING_M."""
        d: Dict[str, Any] = {
            "scores": self.scores.tolist(),       # already float32-rounded
            "ts": self.ts.tolist(),
            "gpd": None if self.gpd is None else self.gpd.to_dict(),
        }
        if self.cap != RING_M:
            d["cap"] = int(self.cap)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Ring":
        """Inverse of to_dict; missing keys give an empty ring. The stored
        entries are re-normalised (rounded, sorted, capped), so a hand-edited
        or older-layout dict still yields a valid ring."""
        g = d.get("gpd")
        tail = g if isinstance(g, GPDTail) else (None if g is None else GPDTail.from_dict(g))
        return cls(cap=int(d.get("cap", RING_M)),
                   scores=np.asarray(d.get("scores", ()), dtype=np.float64),
                   ts=np.asarray(d.get("ts", ()), dtype=np.float64),
                   gpd=tail)


def _normalise(scores: Any, ts: Any, cap: int) -> Tuple[np.ndarray, np.ndarray]:
    """Enforce the Ring invariant on raw arrays (fresh arrays, never views of the input)."""
    s = _r32_array(scores)
    t = np.asarray(ts, dtype=np.float64).ravel()
    if s.size != t.size:
        raise ValueError(f"Ring: len(scores)={s.size} != len(ts)={t.size}")
    ok = np.isfinite(s) & np.isfinite(t)
    if not ok.all():
        s, t = s[ok], t[ok]
    if s.size > cap:
        s, t = _evict_to_cap(s, t, cap)
    order = np.argsort(s, kind="stable")
    return s[order], t[order]


def _evict_to_cap(s: np.ndarray, t: np.ndarray, cap: int) -> Tuple[np.ndarray, np.ndarray]:
    """Keep the `cap` newest entries: evict by ascending (ts, score), the same
    total order repeated Ring.add eviction follows. Output order is by score."""
    n = s.size
    if n <= cap:
        return s, t
    keep = np.lexsort((s, t))[n - cap:]      # last key (t) is primary
    keep = keep[np.argsort(s[keep], kind="stable")]
    return s[keep], t[keep]


# ------------------------------------------------------------------ keys
def _check_part(name: str, value: str, forbidden: str) -> str:
    v = str(value)
    if not v or any(c in v for c in forbidden):
        raise ValueError(f"{name}={value!r} must be non-empty and not contain {forbidden!r}")
    return v


def _as_int(x: Any) -> Optional[int]:
    """x as an int when it is an integral real number (900 or 900.0), else
    None; NaN, +-inf, None, strings and bools are rejected (so key builders
    raise ValueError, never OverflowError / TypeError)."""
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.integer, np.floating)):
        return None
    if isinstance(x, (float, np.floating)) and not (math.isfinite(x) and float(x).is_integer()):
        return None
    return int(x)


def stratum_key(daypart: str, cc: int) -> str:
    """Mondrian stratum label 'daypart|cc', e.g. 'wd_day|900'.

    cc must be a positive integral cadence (900 and 900.0 give the same key).
    """
    dp = _check_part("daypart", daypart, _KEY_SEP + _STRATUM_SEP)
    c = _as_int(cc)
    if c is None or c <= 0:
        raise ValueError(f"stratum_key: cadence class {cc!r} must be a positive integer")
    return f"{dp}{_STRATUM_SEP}{c}"


def identity_stratum_key(daypart: str, regime_tercile: int, cc: Optional[int] = None) -> str:
    """Identity stratum 'daypart|r<k>|cc' (k in 0..2), or 'daypart|r<k>'
    when no cadence class is given.

    The identity score is cadence dependent like every other detector's
    (B16 scores windows of K active ticks: 4 h of evidence at 3600-s ticks,
    1 h at 900 s, 4 min at 60 s, so pi_self is far more certain at coarse
    cadences), hence the cadence class joins daypart x regime tercile, as
    architecture 6 requires of every conformal ring. B24 always passes cc."""
    dp = _check_part("daypart", daypart, _KEY_SEP + _STRATUM_SEP)
    k = _as_int(regime_tercile)
    if k is None or not 0 <= k <= 2:
        raise ValueError(f"identity_stratum_key: regime tercile {regime_tercile!r} not in 0..2")
    if cc is None:
        return f"{dp}{_STRATUM_SEP}r{k}"
    c = _as_int(cc)
    if c is None or c <= 0:
        raise ValueError(f"identity_stratum_key: cadence class {cc!r} must be a positive integer")
    return f"{dp}{_STRATUM_SEP}r{k}{_STRATUM_SEP}{c}"


def grain_stratum_key(daypart: str, grain: str, prov: Optional[int] = None,
                      tercile: Optional[int] = None) -> str:
    """spec v2.1 (cadence.md §5.3, §9.1): stratum of an H / Q stream detector
    in canonical mode: 'daypart|g:h', 'daypart|g:q|p:<prov>' and, for
    identity, 'daypart|r<k>|g:h'. H rings fill at 24 per day at every cadence
    and survive a cadence switch; provisional Q scores have their own ring."""
    dp = _check_part("daypart", daypart, _KEY_SEP + _STRATUM_SEP)
    if grain not in ("h", "q"):
        raise ValueError(f"grain_stratum_key: grain {grain!r} not in ('h', 'q')")
    parts = [dp]
    if tercile is not None:
        k = _as_int(tercile)
        if k is None or not 0 <= k <= 2:
            raise ValueError(f"grain_stratum_key: regime tercile {tercile!r} not in 0..2")
        parts.append(f"r{k}")
    parts.append(f"g:{grain}")
    if grain == "q":
        parts.append(f"p:{1 if prov else 0}")
    return _STRATUM_SEP.join(parts)


def meta_stratum_key(daypart: str, tau: str, cc: Optional[int] = None) -> str:
    """spec v2.1: B25 meta-ring stratum 'daypart|t:<tau>' (+ '|<cc>' for the
    per-tick type t, whose null still depends on the tick length)."""
    dp = _check_part("daypart", daypart, _KEY_SEP + _STRATUM_SEP)
    if tau not in ("h", "q", "t"):
        raise ValueError(f"meta_stratum_key: tick type {tau!r} not in ('h', 'q', 't')")
    if tau == "t":
        c = _as_int(cc)
        if c is None or c <= 0:
            raise ValueError(f"meta_stratum_key: cadence class {cc!r} must be a positive integer")
        return f"{dp}{_STRATUM_SEP}t:t{_STRATUM_SEP}{c}"
    return f"{dp}{_STRATUM_SEP}t:{tau}"


def ring_key(detector: str, stratum: str) -> str:
    """Key inside model.calib: '<detector>@<stratum>'; meta rings use
    detector in {'meta_inst', 'meta_all'} (owned by B25)."""
    d = _check_part("detector", detector, _KEY_SEP)
    st = _check_part("stratum", stratum, _KEY_SEP)
    return f"{d}{_KEY_SEP}{st}"


def split_ring_key(key: str) -> Tuple[str, str]:
    """Inverse of ring_key -> (detector, stratum)."""
    d, sep, st = str(key).partition(_KEY_SEP)
    if not sep or not d or not st or _KEY_SEP in st:
        raise ValueError(f"split_ring_key: {key!r} is not '<detector>@<stratum>'")
    return d, st


# ------------------------------------------------------------ tail fit / p
def _exp_order_means(n: int) -> np.ndarray:
    """E[Y_(m)] / s for the m-th smallest of n Exp(s) draws, m = 1..n:
    H_n - H_{n-m} (H = harmonic numbers)."""
    return np.cumsum(1.0 / np.arange(n, 0, -1, dtype=np.float64))


def winsorise_exceedances(y: np.ndarray, alpha: float) -> np.ndarray:
    """Exceedances y (sorted ascending, >= 0) with outliers pulled in.

    Model: under the null the excesses are exponential with scale s (the
    tail of -log10 of a valid p). The largest of n such excesses is above
    b = s (ln n + ln(1/alpha)) with probability ~alpha, so entries above b
    are contamination. s is estimated from ranks, which the contaminating
    top entries cannot move: with k entries above b, the median inlier
    y_(m), m = ceil((n - k) / 2), is the m-th smallest of all n draws, whose
    mean is s (H_n - H_{n-m}); s = y_(m) / (H_n - H_{n-m}), iterated with b
    until k is stable (start: k = 0, the plain median). Each outlier is then
    replaced by the expected value of its order statistic, s (H_n - H_{r-1})
    for the r-th largest: clipping at b would leave k values far beyond
    every other excess and keep the PWM xi heavy. On a clean exponential
    tail nothing exceeds b with probability ~1 - alpha and y is returned
    unchanged; six warm-up extremes (-log10 p = 21 .. 37 among excesses of
    ~0.4) no longer drag a meta ring's xi to 0.3 - 0.5 (engines.md B25,
    integration §8). Returns y unchanged when alpha is not in (0, 1) or the
    scale is not positive."""
    n = int(y.size)
    if n < 2 or not 0.0 < float(alpha) < 1.0:
        return y
    em = _exp_order_means(n)
    lnb = math.log(n) + math.log(1.0 / float(alpha))
    k = 0
    sc = 0.0
    for _ in range(4):
        m = max(1, (n - k + 1) // 2)
        sc = float(y[m - 1]) / float(em[m - 1])
        if not sc > 0.0:
            return y
        k_new = n - int(np.searchsorted(y, sc * lnb, side="right"))
        if k_new == k:
            break
        k = min(k_new, n - 1)
    if k <= 0:
        return y
    out = np.array(y, dtype=np.float64, copy=True)
    # r-th largest, r = k .. 1 (ascending positions n-k .. n-1): s (H_n - H_{r-1})
    out[n - k:] = np.minimum(out[n - k:], sc * em[n - k:])
    return np.sort(out)


def trim_count(y: np.ndarray, alpha: float, k_max: int) -> int:
    """Number of top exceedances to drop as contamination (robust_tail).

    y: exceedances sorted ascending (>= 0). The null model of the outlier
    test is the one of winsorise_exceedances: excesses exponential with a
    scale s estimated from ranks (the median inlier y_(m), m = ceil((n - k)
    / 2), is the m-th smallest of n draws: s = y_(m) / (H_n - H_{n-m})),
    iterated with the count k of entries above the bound s (ln n + ln
    1/alpha), beyond which the largest of n clean excesses lies with
    probability ~alpha. k is capped at k_max, the contamination bound: at
    most that many entries are ever treated as foreign, so a heavy but clean
    tail cannot be trimmed away wholesale. alpha not in (0, 1), k_max <= 0
    or a non-positive scale -> 0."""
    n = int(y.size)
    k_max = int(k_max)
    if n < 2 or k_max <= 0 or not 0.0 < float(alpha) < 1.0:
        return 0
    em = _exp_order_means(n)
    lnb = math.log(n) + math.log(1.0 / float(alpha))
    k = 0
    for _ in range(6):
        m = max(1, (n - k + 1) // 2)
        sc = float(y[m - 1]) / float(em[m - 1])
        if not sc > 0.0:
            return 0
        k_new = min(k_max, n - int(np.searchsorted(y, sc * lnb, side="right")))
        if k_new == k:
            break
        k = k_new
    return max(0, k)


# robust_tail: the contamination-bounded tail used by B24 and B25 (round 4)
TRIM_ALPHA = 0.01       # outlier test level (a clean 26-exceedance tail is trimmed in ~3.5 % of fits)
TRIM_EPS = 0.02         # contamination bound: at most ceil(0.02 n) entries dropped (6 at M = 256)


def robust_tail(ring: Ring, now_ts: float = float("nan"), alpha: float = TRIM_ALPHA,
                eps: float = TRIM_EPS) -> Optional[GPDTail]:
    """The tail of a NULL ring whose admission does not look at the row's own
    score (lib/gating.period_weight): PWM-GPD over u = q_0.90 fitted to the
    exceedances left after trimming at most ceil(eps n) contaminating top
    entries (trim_count), with the predictive shape floor xi >= 1 / n_u.

    Why trimming instead of score-based exclusion (integration round 4): a
    ring that refuses rows by their own score truncates the tail it fits
    (4.5 - 11x anti-conservative at 1e-3 .. 1e-4 on an exact null, see
    gating.period_weight). Admitting every row of a trusted period instead
    lets a few foreign rows in (an attack released by the governor, a warm-up
    extreme); up to the bound they are removed from the fit and from the
    counts: rate = (n_u - k) / (n - k). Measured on a streaming M = 256
    ring, Exp null, refit every 16 admissions, 3 x 3e4 ticks (realised /
    nominal at p <= 1e-3 / 3e-4 / 1e-4): clean 1.07 / 0.93 / 0.89; with 1 %
    attack-level contamination admitted 0.92 / 1.16 / 1.35, where the
    untrimmed fits give 0.08 - 0.35 (the tail bloats) - tests/lib/
    test_calib_admission.py. The shape floor 1 / n_u is the Bayesian
    predictive of an exponential tail whose scale is estimated from n_u
    excesses (Lomax; engines.md B25 "Tail shape"): the plug-in xi >= 0 was
    1.2 - 1.4x at the same levels. None when fewer than MIN_EXCEED
    exceedances remain."""
    sc = ring.scores
    n = int(sc.size)
    if n < MIN_EXCEED:
        return None
    u = float(np.quantile(sc, TAIL_Q))
    j = int(sc.searchsorted(u, "right"))
    y = sc[j:] - u
    n_u = int(y.size)
    if n_u < MIN_EXCEED:
        return None
    k_max = min(int(math.ceil(float(eps) * n)), n_u - MIN_EXCEED)
    k = trim_count(y, alpha, k_max) if k_max > 0 else 0
    if k:
        y = y[:n_u - k]
    m = n_u - k
    xi, sigma = evt.gpd_pwm_fit(y)
    xi, sigma = float(xi), float(sigma)
    xi_min = 1.0 / m
    if xi < xi_min:
        xi = xi_min
        sigma = float(np.mean(y)) * (1.0 - xi)
    tail = GPDTail(u=u, xi=xi, sigma=sigma, rate=m / (n - k), n=n, fitted_ts=float(now_ts))
    return tail if tail.valid() else None


def fit_tail(ring: Ring, now_ts: float = float("nan"),
             xi_min: float = XI_FLOOR, winsor_alpha: Optional[float] = None
             ) -> Optional[GPDTail]:
    """PWM-GPD fit to exceedances over u = q_0.90(ring) (evt.gpd_pwm_fit).

    Returns None with fewer than MIN_EXCEED exceedances. xi is clipped to
    [-0.5, 0.5] by the fitter. rate = N_u / N.

    xi below `xi_min` (default XI_FLOOR = 0) is raised to xi_min keeping the
    mean excess, sigma = mean(y) (1 - xi_min): at 0 the exponential tail (see
    the module docstring); xi_min = -0.5 reproduces the raw fitter. Exceedances are the
    entries strictly above u (ties at u belong to the body). A degenerate fit
    (sigma not finite and > 0) also returns None, so p_from_ring falls back
    to the conformal p. Pure: the caller assigns ring.gpd. O(M) (the ring is
    already sorted).

    winsor_alpha (None = off): the exceedances are winsorised first
    (winsorise_exceedances) - for scores whose null tail is exponential by
    construction (-log10 of a valid p: B25's meta rings), where a heavier
    fitted tail can only come from contamination.
    """
    sc = ring.scores
    n = int(sc.size)
    if n < MIN_EXCEED:
        return None
    u = float(np.quantile(sc, TAIL_Q))
    k = int(sc.searchsorted(u, "right"))
    n_u = n - k
    if n_u < MIN_EXCEED:
        return None
    y = sc[k:] - u
    if winsor_alpha is not None:
        y = winsorise_exceedances(y, winsor_alpha)
    xi, sigma = evt.gpd_pwm_fit(y)
    xi, sigma = float(xi), float(sigma)
    if xi < xi_min:
        # Keep the mean excess sigma / (1 - xi) = mean(y) (which PWM matches
        # exactly); at xi_min = 0 this is the exponential MLE scale.
        xi = float(xi_min)
        sigma = float(np.mean(y)) * (1.0 - xi)
    tail = GPDTail(u=u, xi=xi, sigma=sigma, rate=n_u / n, n=n, fitted_ts=float(now_ts))
    return tail if tail.valid() else None


def p_from_ring(ring: Ring, s: float, u: float, tail: Optional[GPDTail] = None,
                sigma_min: float = 0.0) -> float:
    """Calibrated p of score s.

    1. s NaN -> NaN.
    2. If `tail` (or ring.gpd) exists and s > tail.u:
         p = tail.rate * evt.gpd_sf(s - tail.u, tail.xi, tail.sigma)
       and, with sigma_min > 0, at least tail.rate * exp(-(s - tail.u) / sigma_min)
       floored at 1e-300 (this is how p < 1/(M+1) is reached), and capped
       at 1/(n+1) when s is above every ring entry (see module docstring).
    3. Otherwise the randomised conformal p on the ring.
    Complexity O(log M).

    `sigma_min` (round 4, evaluator): a lower bound on the tail scale for a
    score that is -log10 of the detector's own p-value (P_SCORE_SIGMA,
    see m_calib.tail_sigma_min): beyond u the issued p then never decays
    faster than the detector's model p itself, rate 10^-(s - u) (a floor
    on the tail function, not on the fitted sigma: with xi > 0 a raised
    sigma would over-shoot). Default 0: off.

    s is float32-rounded first (as stored), for both branches. A tail with
    unusable parameters (see GPDTail.valid) is ignored. u is only used by the
    conformal branch (NaN u -> NaN there; outside [0, 1] raises).
    """
    x = _r32(s)
    if x != x:
        return math.nan
    t = ring.gpd if tail is None else tail
    if t is not None and x > t.u and t.valid():
        p = t.sf(x)
        if sigma_min > 0.0:
            # never faster than the exponential tail of scale sigma_min
            q = t.rate * math.exp(-(x - t.u) / sigma_min)
            if q > p:
                p = q
        n = ring.scores.size
        if n and x > ring.scores[-1]:
            cap = 1.0 / (n + 1)
            if p > cap:
                p = cap
        return p
    return _conformal_p(ring.scores, x, u)


def _conformal_p(sc: np.ndarray, x: float, u: float) -> float:
    """combine.randomized_conformal_p for a ring that holds its invariant
    (float64, sorted, finite), bit-identical to it but ~3x cheaper: it skips
    the generic dtype / NaN handling and uses the ndarray.searchsorted method
    (np.searchsorted's dispatch alone costs ~0.7 us). This is the per (key,
    detector) per tick hot path (B24 Perf: 40 x 31 bisects ~ 2 ms). Anything
    else (e.g. arrays assigned by hand) goes through combine unchanged.
    """
    if x != x:
        return math.nan
    n = sc.size
    if not n or sc.dtype != np.float64 or sc[-1] != sc[-1]:
        return combine.randomized_conformal_p(sc, x, u)
    u = _f(u)
    if u != u:
        return math.nan
    if not 0.0 <= u <= 1.0:
        raise ValueError(f"randomized_conformal_p: u={u!r} outside [0, 1]")
    lo = int(sc.searchsorted(x, "left"))
    hi = int(sc.searchsorted(x, "right"))
    return ((n - hi) + u * ((hi - lo) + 1)) / (n + 1)


def blend_small_sample(p_conf: float, p_model: float, n: int, n0: int = SMALL_N) -> float:
    """Small-ring blend: w = n / (n + n0); p = sigmoid(w logit(p_conf) + (1 - w) logit(p_model)).

    p_model is behavior.pm[d] (exposure-exact model p), the seq stationary
    p_eq for accumulators, or the class-pooled ring p. NaN p_model -> p_conf;
    NaN p_conf -> p_model. p clipped to [1e-300, 1 - 1e-16] inside the logits.

    n <= 0 (or NaN) gives w = 0 (p_model alone); n0 <= 0 gives w = 1.
    Delegates to combine.logit_blend.
    """
    nn = float(n)
    nn = nn if nn > 0.0 else 0.0            # NaN -> 0
    n0f = float(n0)
    if not n0f > 0.0:
        w = 1.0
    elif nn == math.inf:
        w = 1.0
    else:
        w = nn / (nn + n0f)
    return combine.logit_blend(p_conf, p_model, w)


# ------------------------------------------------------------------ health
def ks_uniform(ps: np.ndarray) -> float:
    """One-sample KS statistic D of finite ps against U(0, 1) (health check).

    Non-finite entries are dropped; none left -> NaN. O(n log n).
    """
    p = np.asarray(ps, dtype=np.float64).ravel()
    p = np.sort(p[np.isfinite(p)])
    n = p.size
    if n == 0:
        return math.nan
    i = np.arange(1, n + 1, dtype=np.float64)
    d_plus = float(np.max(i / n - p))
    d_minus = float(np.max(p - (i - 1.0) / n))
    return max(d_plus, d_minus)


def health_weight(ks_d: float, rate_ratio: float) -> float:
    """weight_mult for behavior.calib_health: 0.5 if ks_d > KS_MAX_D or rate_ratio
    outside RATE_RATIO_BAND, else 1.0 (NaN inputs -> 1.0).

    Each NaN only neutralises its own check: (NaN, 5.0) is still 0.5.
    """
    kd = _f(ks_d)
    rr = _f(rate_ratio)
    lo, hi = RATE_RATIO_BAND
    bad_ks = kd > KS_MAX_D                      # False for NaN
    bad_rate = rr == rr and not (lo <= rr <= hi)
    return 0.5 if (bad_ks or bad_rate) else 1.0


# ------------------------------------------------------------------ seeding
def seed_ring(own: Ring, other: Ring, frac: float = 0.5) -> Ring:
    """Link seeding (contract H): own plus the most recent floor(M * frac) entries of
    `other`, oldest-first eviction applied. Returns a new Ring.

    M is own.cap; frac is clipped to [0, 1] (NaN raises ValueError). Seeded
    entries keep other's ts, so a later rollback on the seeded ring removes
    them like its own. Among other's entries with equal ts the higher
    scores are taken first (the same (ts, score) order as eviction). The
    result keeps own.gpd only when nothing was added; otherwise it has no
    tail until the next refit. Neither input is modified.
    """
    f = float(frac)
    if f != f:
        raise ValueError("seed_ring: frac is NaN")
    f = 1.0 if f > 1.0 else (0.0 if f < 0.0 else f)
    k = min(int(math.floor(own.cap * f)), len(other))
    if k <= 0:
        return Ring(cap=own.cap, scores=own.scores, ts=own.ts, gpd=own.gpd)
    pick = np.lexsort((other.scores, other.ts))[-k:]     # k newest of other
    s = np.concatenate((own.scores, other.scores[pick]))
    t = np.concatenate((own.ts, other.ts[pick]))
    return Ring(cap=own.cap, scores=s, ts=t, gpd=None)
