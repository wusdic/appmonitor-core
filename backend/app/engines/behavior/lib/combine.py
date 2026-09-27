"""Evidence combination primitives (B04, B24, B25, B26, B27, B28).

STATUS: implemented. Signatures and maths are frozen (docs/lib3/helpers_api.md).

Why weighted HMP (and not Fisher / ACAT): detector p-values are strongly
dependent (rho ~ 0.64 measured) and many sit at p ~ 1. Fisher inflates false
alarms 32x under that dependence; ACAT is masked by a single p near 1
(ACAT([1e-6, 1.0]) ~ 1). The harmonic mean Sum w / Sum (w/p) is dominated by
the smallest p, insensitive to p ~ 1, and measured at 1.095x nominal at 1e-3
under rho = 0.64; residual miscalibration is removed by per-entity
meta-calibration in B25.

Why randomised conformal p: sparse / accumulator scores have a point mass at
their minimum (silence 0, CUSUM 0) which gives deterministic conformal p = 1
ties that break KS health. The tie-breaking U is *seeded* per
(system, entity, detector, ts) so replay (B29) is bit-reproducible.

Why scalar pure-Python bodies: every function here runs per (entity,
detector) per tick on a handful of values, where numpy call overhead (~15 us
for a 3-element whmp) dwarfs the arithmetic (~1 us in plain floats). Only a
large ndarray input to whmp takes the vectorised path; both paths are tested
to agree.
"""
from __future__ import annotations

import hashlib
import math
from typing import Optional, Sequence

import numpy as np

P_FLOOR = 1e-300            # p is clipped to [P_FLOOR, 1] before 1/p
SECONDS_PER_DAY = 86400.0

# e_day severity ladder (architecture section 4). The thresholds are
# multiplied by the feedback alpha_mult (B23) before comparison.
SEVERITY_E_DAY = (("critical", 3e-6), ("high", 3e-4), ("medium", 3e-3), ("low", 0.03))

_P_CEIL_LOGIT = 1.0 - 1e-16     # logit_blend upper clip (rounds to 1 - 2**-53)
_U_MAX = 1.0 - 2.0 ** -53       # largest float64 below 1
_TWO64 = 2.0 ** 64
_F32_INF_EDGE = 3.4028235677973366e38   # smallest float64 that rounds to float32 inf
_VEC_MIN = 64                   # whmp: ndarray inputs larger than this go vectorised


def _f(x: object) -> float:
    """float(x) with None -> NaN (a missing store value is 'unscored', not an error)."""
    return math.nan if x is None else float(x)  # type: ignore[arg-type]


# ------------------------------------------------------------------- wHMP
def whmp(ps: Sequence[float], ws: Optional[Sequence[float]] = None) -> float:
    """Weighted harmonic mean p: Sum w_i / Sum (w_i / p_i) over finite p_i only.

    NaN p (degraded / unscored) are dropped together with their weights; p
    is clipped to [P_FLOOR, 1]; weights must be >= 0 (zero-weight entries are
    ignored). Returns NaN when no valid (p, w > 0) pair remains. Result is
    clipped to <= 1. Scale-invariant in w. O(len(ps)).
    Example: whmp([1e-3, 0.999, 0.5]) = 0.003 (equal weights).

    Non-finite p (NaN, +-inf) and None count as dropped. A weight kept with a
    finite p must be finite and >= 0, otherwise ValueError (a NaN weight is a
    bug upstream, not evidence to drop silently); weights of dropped p are not
    inspected. len(ws) != len(ps) raises ValueError. Weights are rescaled by
    their max before summing, so huge weights with p = P_FLOOR cannot overflow.
    """
    if isinstance(ps, np.ndarray):
        if ps.size > _VEC_MIN:
            return _whmp_np(ps, ws)
        p_list = ps.ravel().tolist()
    else:
        p_list = ps if isinstance(ps, (list, tuple)) else list(ps)
    if ws is not None:
        if isinstance(ws, np.ndarray):
            w_list = ws.ravel().tolist()
        else:
            w_list = ws if isinstance(ws, (list, tuple)) else list(ws)
        if len(w_list) != len(p_list):
            raise ValueError(f"whmp: len(ws)={len(w_list)} != len(ps)={len(p_list)}")
    vp: list = []
    vw: list = []
    wmax = 0.0
    for i, p in enumerate(p_list):
        p = _f(p)
        if not math.isfinite(p):
            continue
        if ws is None:
            w = 1.0
        else:
            w = _f(w_list[i])
            if not (w >= 0.0 and w != math.inf):
                raise ValueError(f"whmp: weight {w!r} at {i} must be finite and >= 0")
            if w == 0.0:
                continue
        vp.append(1.0 if p > 1.0 else (P_FLOOR if p < P_FLOOR else p))
        vw.append(w)
        if w > wmax:
            wmax = w
    if not vp:
        return math.nan
    # Scale weights into (0, 1] first: Sum w cannot overflow and the largest
    # weight's term keeps Sum (w/p) >= 1, so the quotient is in (0, 1].
    sw = 0.0
    s = 0.0
    for p, w in zip(vp, vw):
        w /= wmax
        sw += w
        s += w / p
    return _clip_p(sw / s)


def _whmp_np(ps: np.ndarray, ws: Optional[Sequence[float]]) -> float:
    """Vectorised whmp for large inputs; same semantics as the scalar path."""
    p = np.asarray(ps, dtype=np.float64).ravel()
    if ws is None:
        w = np.ones_like(p)
    else:
        w = np.asarray(ws, dtype=np.float64).ravel()
        if w.size != p.size:
            raise ValueError(f"whmp: len(ws)={w.size} != len(ps)={p.size}")
    keep = np.isfinite(p)
    wk = w[keep]
    if not np.all((wk >= 0.0) & (wk != np.inf)):
        raise ValueError("whmp: weights of finite p must be finite and >= 0")
    keep &= w > 0.0
    if not keep.any():
        return math.nan
    pk = np.clip(p[keep], P_FLOOR, 1.0)
    wk = w[keep] / w[keep].max()
    return _clip_p(float(wk.sum()) / float(np.sum(wk / pk)))


def _clip_p(p: float) -> float:
    return 1.0 if p > 1.0 else (P_FLOOR if p < P_FLOOR else p)


# ------------------------------------------------------ randomised conformal
def randomized_conformal_p(ring_sorted: np.ndarray, s: float, u: float) -> float:
    """p = (#{c > s} + u (#{c == s} + 1)) / (|C| + 1).

    ring_sorted: ascending float array of calibration scores (higher score =
    more anomalous); u in (0, 1) from seeded_uniform. Two np.searchsorted
    calls: O(log |C|). s NaN -> NaN. Empty ring -> u (uniform under the null).
    Exact ties are compared with ==, so callers must round scores to float32
    consistently before both storing and scoring.

    Robustness (all O(log |C|)): trailing NaN in the ring (where np.sort puts
    them) are not counted as calibration scores; a float32 ring compares s at
    float32 precision; u NaN -> NaN; u outside [0, 1] raises ValueError.
    Result is in (0, 1) for u in (0, 1): never 0, never 1.
    """
    s = _f(s)
    if s != s:
        return math.nan
    u = _f(u)
    if u != u:
        return math.nan
    if not 0.0 <= u <= 1.0:
        raise ValueError(f"randomized_conformal_p: u={u!r} outside [0, 1]")
    c = ring_sorted if isinstance(ring_sorted, np.ndarray) else np.asarray(ring_sorted, dtype=np.float64)
    n = int(c.size)
    if n and c[-1] != c[-1]:                    # NaN sorts last; drop them
        n = int(np.searchsorted(c, np.nan, side="left"))
        c = c[:n]
    if n == 0:
        return u
    if c.dtype == np.float32:
        # Compare in the ring's precision. |s| past the float32 range rounds
        # to +-inf; do that by hand, since np.float32(s) would emit an
        # overflow RuntimeWarning (an exception under -W error).
        s = float(np.float32(s)) if abs(s) < _F32_INF_EDGE else math.copysign(math.inf, s)
    lo = int(np.searchsorted(c, s, side="left"))
    hi = int(np.searchsorted(c, s, side="right"))
    return ((n - hi) + u * ((hi - lo) + 1)) / (n + 1)


# ------------------------------------------------------------------ e_day
def e_day(p: float, dt_s: float) -> float:
    """Expected number of equally extreme null ticks per entity-day: p * 86400 / dt_s.

    Cadence-free severity; NaN -> NaN.
    NaN dt_s -> NaN; a non-positive or infinite dt_s raises ValueError.
    """
    dt = _check_dt(dt_s)
    return _f(p) * SECONDS_PER_DAY / dt


def p_from_e_day(e: float, dt_s: float) -> float:
    """Inverse of e_day: min(1, e * dt_s / 86400).

    Clipped to [0, 1]; NaN e or dt_s -> NaN; dt_s as in e_day.
    """
    dt = _check_dt(dt_s)
    p = _f(e) * dt / SECONDS_PER_DAY
    if p != p:
        return math.nan
    return 1.0 if p > 1.0 else (0.0 if p < 0.0 else p)


def _check_dt(dt_s: float) -> float:
    dt = _f(dt_s)
    if dt != dt:
        return math.nan
    if not (0.0 < dt < math.inf):
        raise ValueError(f"dt_s={dt!r} must be a positive finite cadence in seconds")
    return dt


# --------------------------------------------------------- seeded uniform
def seeded_uniform(*keys: object) -> float:
    """Deterministic U(0, 1) from arbitrary keys (e.g. system, entity, detector, ts).

    u = (int.from_bytes(blake2b('|'.join(map(repr_key, keys)), digest_size=8)) + 0.5) / 2**64
    where repr_key renders floats with repr(float(x)) so 1.0 and 1 differ
    from 'x'. Never 0 or 1; stable across processes and Python versions (no
    built-in hash()). ~1 us.

    repr_key normalises numpy scalars to their Python counterparts (numpy 2
    reprs them as 'np.float64(1.0)'), so np.float64(1.0) == 1.0 as a key while
    1, 1.0 and '1' are three different keys: pass ts consistently as float.
    Strings are repr-quoted, so ('a|b',) and ('a', 'b') differ; tuples and
    lists recurse. The digest is read big-endian (the int.from_bytes default)
    from UTF-8 bytes. The one float rounding that could reach 1.0 (a digest
    within 2**11 of 2**64) is mapped to the largest float below 1.
    """
    msg = "|".join(map(_repr_key, keys)).encode("utf-8")
    h = int.from_bytes(hashlib.blake2b(msg, digest_size=8).digest(), "big")
    return _u_from_int(h)


def _u_from_int(h: int) -> float:
    u = (h + 0.5) / _TWO64
    return u if u < 1.0 else _U_MAX


def _repr_key(k: object) -> str:
    t = type(k)
    if t is str or t is float or t is int:      # hot path: exact builtins
        return repr(k)
    # Order matters: bool subclasses int; np.str_ subclasses str.
    if isinstance(k, str):
        return repr(str(k))
    if isinstance(k, (bool, np.bool_)):
        return repr(bool(k))
    if isinstance(k, (int, np.integer)):
        return repr(int(k))
    if isinstance(k, (float, np.floating)):
        return repr(float(k))
    if isinstance(k, (tuple, list)):
        return "(" + ",".join(map(_repr_key, k)) + ")"
    return repr(k)


# --------------------------------------------------------------- severity
def e_day_severity(e: float, alpha_mult: float = 1.0) -> Optional[str]:
    """Severity *candidate* from e_day: the first (name, thr) in SEVERITY_E_DAY
    with e <= thr * alpha_mult, else None. Corroboration rules (HIGH needs 2
    axes, CRITICAL 3 or lib-4) are applied by the caller (B25).

    NaN e -> None (nothing to report). alpha_mult must be finite and > 0,
    otherwise ValueError (a NaN multiplier would silently mute every alarm).
    """
    a = _f(alpha_mult)
    if not (0.0 < a < math.inf):
        raise ValueError(f"e_day_severity: alpha_mult={a!r} must be finite and > 0")
    e = _f(e)
    if e != e:
        return None
    for name, thr in SEVERITY_E_DAY:
        if e <= thr * a:
            return name
    return None


# ------------------------------------------------------------ logit blend
def logit_blend(p1: float, p2: float, w1: float) -> float:
    """sigmoid(w1 logit(p1) + (1 - w1) logit(p2)), p clipped to [1e-300, 1 - 1e-16];
    NaN in either input returns the other (both NaN -> NaN).

    The passthrough value is returned as given (not clipped). w1 is clipped
    to [0, 1]; NaN w1 raises ValueError. Log-space logit (log p - log1p(-p))
    and a sign-split sigmoid keep p ~ 1e-300 exact to ~1e-15 relative.
    """
    p1 = _f(p1)
    p2 = _f(p2)
    if p1 != p1:
        return p2
    if p2 != p2:
        return p1
    w = _f(w1)
    if w != w:
        raise ValueError("logit_blend: w1 is NaN")
    w = 1.0 if w > 1.0 else (0.0 if w < 0.0 else w)
    x = w * _logit(p1) + (1.0 - w) * _logit(p2)
    return _sigmoid(x)


def _logit(p: float) -> float:
    p = _P_CEIL_LOGIT if p > _P_CEIL_LOGIT else (P_FLOOR if p < P_FLOOR else p)
    return math.log(p) - math.log1p(-p)


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)
