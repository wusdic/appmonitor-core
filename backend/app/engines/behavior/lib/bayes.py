"""Exact, exposure-aware predictive tail probabilities (B03, B04, B07, B18).

Why hand-rolled instead of scipy.stats: scipy.stats.nbinom.cdf costs ~60-290 us
per call against ~0.2-3 us for scipy.special.betainc on the same quantity
(measured), and B04 evaluates ~100 tails per entity per tick with two
anchors. All functions are vectorised over numpy arrays (broadcasting), work
in float64, never return p = 0 or 1 exactly where a z-score is taken
(phi_inv clips), and propagate NaN inputs to NaN outputs (never to p = 1).
Scalar calls (the common case inside engine loops) take a fast path through
scipy.special.cython_special, which skips ufunc dispatch (~0.2 us instead of
~1 us per special-function call); scalar in -> Python float out.

Discrete tails use the *mid-distribution* value
    u = P(X < x) + 0.5 * P(X = x)
which is uniform-ish under the null for discrete X, so z = phi_inv(u) is
~N(0, 1) and the two-sided mid-p is min(1, 2 * min(u, 1 - u)). Each tail is
computed on its own side (sf, not 1 - cdf), so p down to ~1e-300 is exact;
two-sided p is floored at P_FLOOR so downstream logs stay finite. Observed
counts (k, n) are rounded to the nearest integer in pmf / mid-p functions
(they may carry float32 ring noise); cdf / sf take floor(k) as P(X <= k)
is defined for real k. An observation outside the support (k < 0, k > n)
is invalid data and scores NaN, never p = 0.

Conventions
    NB(mean m, size r): P(X = k) = C(k+r-1, k) q^r (1-q)^k with q = r/(r+m);
        var = m + m^2 / r; r -> inf is Poisson(m).
    BB(n, a, b): Beta-Binomial with pmf C(n,k) B(k+a, n-k+b) / B(a, b).
    NIG(m, kappa, alpha, beta): x | mu, s2 ~ N(mu, s2); mu | s2 ~ N(m, s2/kappa);
        s2 ~ InvGamma(alpha, beta). Posterior predictive for one new x is
        Student-t(df = 2 alpha, loc = m, scale = sqrt(beta (kappa + 1) / (alpha kappa))).

Beta-binomial tails: a normal approximation is unusable for ratios (a 1 %
error rate at n = 1000 with c = 50 is strongly right-skewed; the normal tail is
off by >100 orders of magnitude, i.e. constant false alarms). The tail on the
far side of k from the mean is therefore summed exactly from k outward with
the pmf ratio recurrence
    pmf(j+1) / pmf(j) = (n-j)(j+a) / ((j+1)(n-j-1+b))
(log-space cumsum, a handful of vector ops per call), which is exact for every
n <= BB_EXACT_MAX_N and for any n whose tail decays within BB_EXACT_MAX_N
terms. Only the part of a tail lying more than BB_EXACT_MAX_N counts beyond k
(where the pmf is smooth on the integer grid) comes from a Beta moment-matched
to the jittered count (K + U) / (n + 1), which keeps the skewness. Measured
worst |log10 error| of the tails: 0.13 over n <= 1e5, mean in [1e-4, 0.999],
c in [2, 1e6], p in [1e-10, 0.5]; exact (~1e-10 relative) whenever n <= 2048.

Negative-binomial precision: the incomplete beta is fed q or 1 - q, each
formed directly, never a rounded near-1 argument at large r or m (see
_nb_le1), and ln C(k+r-1, k) uses a Stirling difference
once max(r, k+1) >= 10, so pmf / cdf / sf are exact to ~1e-12 relative for
every size up to the Poisson switch (scipy.stats drifts to 1e-8 at r = 1e6).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple, Union

import numpy as np
from scipy import special as sp

try:  # scalar entry points without ufunc dispatch; float args only
    from scipy.special import cython_special as _cs
    _c_betainc = _cs.betainc
    _c_betaincc = _cs.betaincc
    _c_pdtr = _cs.pdtr
    _c_pdtrc = _cs.pdtrc
    _c_ndtr = _cs.ndtr
    _c_ndtri = _cs.ndtri
    _c_stdtr = _cs.stdtr
except (ImportError, AttributeError):  # pragma: no cover - very old scipy
    _c_betainc, _c_betaincc = sp.betainc, sp.betaincc
    _c_pdtr, _c_pdtrc = sp.pdtr, sp.pdtrc
    _c_ndtr, _c_ndtri, _c_stdtr = sp.ndtr, sp.ndtri, sp.stdtr

ArrayLike = Union[float, np.ndarray]

PHI_CLIP = 1e-15            # phi_inv clips u to [PHI_CLIP, 1 - PHI_CLIP] -> |z| <= 7.94
BB_EXACT_MAX_N = 2048       # BB tails summed exactly over up to this many terms (all n <= it)
NB_KAPPA_CLIP = (0.5, 1e3)  # overdispersion estimate clip (B04)
BB_PHI_CLIP = (20.0, 1000.0)
P_FLOOR = 1e-300            # two-sided p floor (finite -log10 p, finite 1/p in wHMP)
NB_R_MIN = 1e-6             # NB size clip ...
NB_R_POISSON = 1e12         # ... and the size at/above which NB is evaluated as Poisson
BB_P_CLIP = 1e-6            # bb_params clips p_hat to [BB_P_CLIP, 1 - BB_P_CLIP]

_KAPPA_POISSON = NB_KAPPA_CLIP[1]  # "no evidence of overdispersion"
_PHI_NO_INFO = BB_PHI_CLIP[1]      # "no evidence of between-row dispersion"
_SCALAR = (int, float, np.integer, np.floating)
_NAN = float("nan")
_INF = float("inf")
_BB_TAIL_REL = 1e-17        # a truncated tail sum stops once its last term is below this share
_INT_CAP = float(2 ** 52)   # integer search ceiling (exact float integers)
_BB_PPF_GRID_MAX = 1024     # bb_ppf sums the whole support up to this n, bisects above
_BB_WIN0 = 256.0            # first BB tail window (terms); then one pass of BB_EXACT_MAX_N
_BB_PY_TERMS = 16           # BB tail sides this short are summed in plain Python
_BIG_ARG_MAX = 1e4          # NB: betainc may take a near-1 argument while max(r, m) <= this
_STIRLING_MIN = 10.0     # max(r, k+1) above which ln C(k+r-1, k) uses a Stirling difference


@dataclass(slots=True)
class NIG:
    """Normal-Inverse-Gamma posterior on a transformed feature value."""
    m: float
    kappa: float
    alpha: float
    beta: float


@dataclass(slots=True)
class GammaRate:
    """Gamma(shape a, rate b) posterior on a per-minute rate; b is in minutes."""
    a: float
    b: float

    @property
    def mean(self) -> float:
        return self.a / self.b


# ----------------------------------------------------------------- plumbing
def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _out(x):
    """0-d result -> Python float, otherwise the ndarray."""
    x = np.asarray(x)
    return float(x) if x.ndim == 0 else x


def _count(x: np.ndarray) -> np.ndarray:
    """Nearest integer (half up), for observed counts that may carry float32 noise."""
    return np.floor(x + 0.5)


def _clip_p(p):
    return np.clip(p, P_FLOOR, 1.0)            # np.clip keeps NaN


def _p2_1(u: float, v: float) -> Tuple[float, float]:
    """Scalar (u, two-sided p) from the two mid-values. Python's min / max
    silently drop NaN (max(P_FLOOR, nan) is P_FLOOR, a p = 1e-300 false alarm),
    so NaN on either side is returned as (NaN, NaN) explicitly."""
    if u != u or v != v:
        return _NAN, _NAN
    return u, min(1.0, max(P_FLOOR, 2.0 * min(u, v)))


def _fl1(x: float) -> float:
    """Scalar floor that passes inf / NaN through (math.floor raises on inf)."""
    return float(math.floor(x)) if math.isfinite(x) else x


def _rd1(x: float) -> float:
    """Scalar nearest integer (half up), inf / NaN passed through."""
    return float(math.floor(x + 0.5)) if math.isfinite(x) else x


# ---------------------------------------------------------------- normal
def phi_inv(u: ArrayLike, clip: float = PHI_CLIP) -> ArrayLike:
    """Standard normal quantile of u clipped to [clip, 1 - clip] (scipy.special.ndtri).

    NaN -> NaN. O(1) per element.
    """
    if isinstance(u, _SCALAR):
        u = float(u)
        if u != u:
            return _NAN
        return float(_c_ndtri(min(max(u, clip), 1.0 - clip)))
    return _out(sp.ndtri(np.clip(_arr(u), clip, 1.0 - clip)))


def phi_sf(z: ArrayLike) -> ArrayLike:
    """Upper normal tail P(Z > z) (scipy.special.ndtr(-z)); NaN -> NaN."""
    if isinstance(z, _SCALAR):
        return float(_c_ndtr(-float(z)))
    return _out(sp.ndtr(-_arr(z)))


def two_sided_midp(u: ArrayLike) -> ArrayLike:
    """Two-sided p from a mid-distribution value: min(1, 2 * min(u, 1 - u)); NaN -> NaN.

    Floored at P_FLOOR (a z-score / log is always taken downstream).
    """
    if isinstance(u, _SCALAR):
        u = float(u)
        if u != u:
            return _NAN
        return min(1.0, max(P_FLOOR, 2.0 * min(u, 1.0 - u)))
    u = _arr(u)
    return _out(_clip_p(2.0 * np.minimum(u, 1.0 - u)))


def combine_anchors(p_cur: ArrayLike, p_ref: ArrayLike) -> ArrayLike:
    """Dual-anchor p (B04): min(1, 2 * min(p_cur, p_ref)).

    If one of the two is NaN the other is used alone (no factor 2); both
    NaN -> NaN.
    """
    if isinstance(p_cur, _SCALAR) and isinstance(p_ref, _SCALAR):
        a, b = float(p_cur), float(p_ref)
        if a != a:
            return b
        if b != b:
            return a
        return min(1.0, 2.0 * min(a, b))
    a, b = np.broadcast_arrays(_arr(p_cur), _arr(p_ref))
    both = ~np.isnan(a) & ~np.isnan(b)
    m = np.fmin(a, b)                          # the non-NaN one when only one is NaN
    return _out(np.where(both, np.minimum(1.0, 2.0 * m), m))


# ------------------------------------------------------------ negative binomial
# Scalar kernels. Preconditions: k integer-valued float >= 0, m > 0 (may be inf),
# r already clipped to [NB_R_MIN, NB_R_POISSON].
def _stirling_tail(x):
    """lnGamma(x) - [(x - 0.5) ln x - x + 0.5 ln(2 pi)] for x >= 9 (error < 1e-12)."""
    x2 = x * x
    return (1.0 / 12.0 - (1.0 / 360.0 - (1.0 / 1260.0 - 1.0 / (1680.0 * x2)) / x2) / x2) / x


def _lncoef1(r: float, k: float) -> float:
    """ln C(k+r-1, k) = lnGamma(k+r) - lnGamma(r) - lnGamma(k+1).

    With x = max(r, k+1) >= 10 the large pair is taken as the Stirling
    difference lnGamma(x+d) - lnGamma(x) = (x - 0.5) log1p(d/x) + d ln(x+d) - d
    + tail(x+d) - tail(x), exact to ~1e-15 |d| ln x; the plain gammaln
    difference loses ~eps x ln x (1e-9 relative on the pmf at r = 1e6).
    """
    if k + 1.0 >= r:
        x, d, rest = k + 1.0, r - 1.0, math.lgamma(r)
    else:
        x, d, rest = r, k, math.lgamma(k + 1.0)
    if x < _STIRLING_MIN:
        return math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1.0)
    return ((x - 0.5) * math.log1p(d / x) + d * math.log(x + d) - d
            + _stirling_tail(x + d) - _stirling_tail(x) - rest)


def _nb_lpmf1(k: float, m: float, r: float) -> float:
    """log pmf = ln C(k+r-1, k) + r ln q + k ln(1-q) with ln q = -log1p(m/r)
    and ln(1-q) = -log1p(r/m), both exact at any r/m."""
    if r >= NB_R_POISSON:
        if m == _INF:
            return -_INF
        return -m if k == 0.0 else k * math.log(m) - m - math.lgamma(k + 1.0)
    lp = -r * math.log1p(m / r)
    if k == 0.0:
        return lp
    return lp - k * math.log1p(r / m) + _lncoef1(r, k)


# Tails come from the regularised incomplete beta I_x. Its argument is formed
# directly (q = r/(r+m) or 1 - q = m/(r+m)); an argument near 1 is rounded
# before the call and that error is amplified ~3 max(r, m) times, so near-1
# arguments are only used while max(r, m) <= _BIG_ARG_MAX (<= 3e-12 relative);
# beyond that the small argument goes to betaincc (exact at any r, m, but ~5x
# slower than betainc in scipy 1.17). betainc(r, k+1, q) alone is off by 1e-6
# at r = 1e11. The upper tail also needs m <= _BIG_ARG_MAX * r: rounding
# y = 1 - q costs eps / q = eps (r + m) / r relative in q, and at small size
# P(X > k) ~ r ln(1/q) inherits it (1e-8 relative at r = 1e-6, m = 1e3).
# The lower tail needs no such rule: q near 1 with m << r means P(X <= k) ~ 1.
def _nb_le1(k: float, m: float, r: float) -> float:
    """P(X <= k) = I_q(r, k+1)."""
    if r >= NB_R_POISSON:
        return float(_c_pdtr(k, m))
    q = 1.0 / (1.0 + m / r)
    if q <= 0.5 or r <= _BIG_ARG_MAX:
        return float(_c_betainc(r, k + 1.0, q))
    return float(_c_betaincc(k + 1.0, r, 1.0 / (1.0 + r / m)))


def _nb_lt1(k: float, m: float, r: float) -> float:
    """P(X < k) = P(X <= k-1)."""
    return _nb_le1(k - 1.0, m, r) if k > 0.0 else 0.0


def _nb_gt1(k: float, m: float, r: float) -> float:
    """P(X > k) = I_{1-q}(k+1, r)."""
    if r >= NB_R_POISSON:
        return float(_c_pdtrc(k, m))
    y = 1.0 / (1.0 + r / m)
    if y <= 0.5 or (m <= _BIG_ARG_MAX and m <= _BIG_ARG_MAX * r):
        return float(_c_betainc(k + 1.0, r, y))
    return float(_c_betaincc(r, k + 1.0, 1.0 / (1.0 + m / r)))


# Array kernels (same preconditions, element-wise; `pois` marks r >= NB_R_POISSON).
def _nb_prep(k, mean, r):
    k, m, r = np.broadcast_arrays(_arr(k), _arr(mean), _arr(r))
    bad = np.isnan(k) | np.isnan(m) | np.isnan(r)
    rr = np.clip(r, NB_R_MIN, NB_R_POISSON)
    return k, m, rr, bad, rr >= NB_R_POISSON


def _lncoef_a(r, k):
    """Array _lncoef1."""
    with np.errstate(all="ignore"):
        kx = k + 1.0
        first = kx >= r
        x = np.where(first, kx, r)
        d = np.where(first, r - 1.0, k)
        big = x >= _STIRLING_MIN
        xb = np.where(big, x, _STIRLING_MIN)
        db = np.where(big, d, 0.0)
        st = ((xb - 0.5) * np.log1p(db / xb) + db * np.log(xb + db) - db
              + _stirling_tail(xb + db) - _stirling_tail(xb) - sp.gammaln(np.where(first, r, kx)))
        return np.where(big, st, sp.gammaln(k + r) - sp.gammaln(r) - sp.gammaln(kx))


def _nb_lpmf_a(k, m, rr, pois):
    with np.errstate(all="ignore"):
        lp = -rr * np.log1p(m / rr)
        lp = np.where(k > 0, lp - k * np.log1p(rr / m) + _lncoef_a(rr, k), lp)
        if pois.any():
            lpp = sp.xlogy(k, m) - m - sp.gammaln(k + 1.0)
            lp = np.where(pois, np.where(np.isposinf(m), -np.inf, lpp), lp)
    return lp


def _nb_tail_a(k, m, rr, pois, upper):
    """P(X > k) where `upper` (bool or bool array) else P(X <= k), for integer
    k >= -1, with the argument rule of _nb_le1 / _nb_gt1: betainc on the small
    argument (or on a near-1 one while max(r, m) <= _BIG_ARG_MAX), betaincc on
    the small argument otherwise."""
    upper = np.broadcast_to(upper, k.shape)
    out = np.empty(k.shape)
    with np.errstate(all="ignore"):
        q = 1.0 / (1.0 + m / rr)
        lo_q = ~upper & ((q <= 0.5) | (rr <= _BIG_ARG_MAX))      # I_q(r, k+1)
        up_y = upper & ((q >= 0.5) | ((m <= _BIG_ARG_MAX) & (m <= _BIG_ARG_MAX * rr)))
        lo_c = ~upper & ~lo_q                                    # 1 - I_{1-q}(k+1, r)
        up_c = upper & ~up_y                                     # 1 - I_q(r, k+1)
        if lo_q.any():
            out[lo_q] = sp.betainc(rr[lo_q], k[lo_q] + 1.0, q[lo_q])
        if up_y.any():
            out[up_y] = sp.betainc(k[up_y] + 1.0, rr[up_y], 1.0 / (1.0 + rr[up_y] / m[up_y]))
        if lo_c.any():
            out[lo_c] = sp.betaincc(k[lo_c] + 1.0, rr[lo_c], 1.0 / (1.0 + rr[lo_c] / m[lo_c]))
        if up_c.any():
            out[up_c] = sp.betaincc(rr[up_c], k[up_c] + 1.0, q[up_c])
        if pois.any():
            pu, pl = pois & upper, pois & ~upper
            out[pu] = sp.pdtrc(k[pu], m[pu])
            out[pl] = sp.pdtr(k[pl], m[pl])
    return np.where(k < 0, np.where(upper, 1.0, 0.0), out)


def _nb_le_a(k, m, rr, pois):
    """P(X <= k) for integer k >= -1."""
    return _nb_tail_a(k, m, rr, pois, False)


def _nb_gt_a(k, m, rr, pois):
    """P(X > k) for integer k >= -1."""
    return _nb_tail_a(k, m, rr, pois, True)


def nb_pmf(k: ArrayLike, mean: ArrayLike, r: ArrayLike) -> ArrayLike:
    """NB(mean, size r) pmf via gammaln (log-space), k integer >= 0 (k < 0 -> 0).

    k is rounded to the nearest integer; mean <= 0 is a point mass at 0.
    """
    if isinstance(k, _SCALAR) and isinstance(mean, _SCALAR) and isinstance(r, _SCALAR):
        k, m, r = float(k), float(mean), float(r)
        if k != k or m != m or r != r:
            return _NAN
        k = _rd1(k)
        if k < 0.0 or k == _INF:
            return 0.0
        if m <= 0.0:
            return 1.0 if k == 0.0 else 0.0
        return math.exp(_nb_lpmf1(k, m, min(max(r, NB_R_MIN), NB_R_POISSON)))
    k, m, rr, bad, pois = _nb_prep(k, mean, r)
    k = _count(k)
    out = np.exp(_nb_lpmf_a(k, m, rr, pois))
    out = np.where(m <= 0, (k == 0).astype(np.float64), out)
    out = np.where((k < 0) | np.isinf(k), 0.0, out)
    return _out(np.where(bad, np.nan, out))


def nb_cdf(k: ArrayLike, mean: ArrayLike, r: ArrayLike) -> ArrayLike:
    """P(X <= k) = betainc(r, floor(k) + 1, q), q = r / (r + mean); k < 0 -> 0.

    mean <= 0 -> point mass at 0 (cdf = 1 for k >= 0). r is clipped to
    [1e-6, 1e12]; r >= 1e12 is treated as Poisson (scipy.special.pdtr).
    For q > 0.5 and r > 1e4 the same value is taken as
    betaincc(floor(k) + 1, r, 1 - q) (exact at large r, see _nb_le1).
    """
    if isinstance(k, _SCALAR) and isinstance(mean, _SCALAR) and isinstance(r, _SCALAR):
        k, m, r = float(k), float(mean), float(r)
        if k != k or m != m or r != r:
            return _NAN
        if k < 0.0:
            return 0.0
        if m <= 0.0 or k == _INF:
            return 1.0
        return _nb_le1(float(math.floor(k)), m, min(max(r, NB_R_MIN), NB_R_POISSON))
    k, m, rr, bad, pois = _nb_prep(k, mean, r)
    kf = np.floor(k)
    out = _nb_le_a(np.where(np.isfinite(kf), kf, -1.0), m, rr, pois)
    out = np.where(np.isposinf(k) | (m <= 0), 1.0, out)
    out = np.where(kf < 0, 0.0, out)
    return _out(np.where(bad, np.nan, out))


def nb_sf(k: ArrayLike, mean: ArrayLike, r: ArrayLike) -> ArrayLike:
    """P(X > k) = betainc(floor(k) + 1, r, 1 - q) computed directly (no 1 - cdf cancellation)."""
    if isinstance(k, _SCALAR) and isinstance(mean, _SCALAR) and isinstance(r, _SCALAR):
        k, m, r = float(k), float(mean), float(r)
        if k != k or m != m or r != r:
            return _NAN
        if k < 0.0:
            return 1.0
        if m <= 0.0 or k == _INF:
            return 0.0
        return _nb_gt1(float(math.floor(k)), m, min(max(r, NB_R_MIN), NB_R_POISSON))
    k, m, rr, bad, pois = _nb_prep(k, mean, r)
    kf = np.floor(k)
    out = _nb_gt_a(np.where(np.isfinite(kf), kf, -1.0), m, rr, pois)
    out = np.where(np.isposinf(k) | (m <= 0), 0.0, out)
    out = np.where(kf < 0, 1.0, out)
    return _out(np.where(bad, np.nan, out))


def _nb_mid1(k, mean, r) -> Tuple[float, float]:
    """Scalar (u, 1 - u), each side from its own tails:
    u = P(X<k) + P(X=k)/2 = (F(k-1) + F(k))/2 and 1 - u = (S(k-1) + S(k))/2,
    sums of same-side tails (no pmf, no subtraction). Invalid k -> NaN."""
    k, m, r = float(k), float(mean), float(r)
    if k != k or m != m or r != r:
        return _NAN, _NAN
    k = _rd1(k)
    if k < 0.0 or k == _INF:
        return _NAN, _NAN
    if m <= 0.0:                               # point mass at 0
        return (0.5, 0.5) if k == 0.0 else (1.0, 0.0)
    r = min(max(r, NB_R_MIN), NB_R_POISSON)
    if k == 0.0:
        return 0.5 * _nb_le1(0.0, m, r), 0.5 * (1.0 + _nb_gt1(0.0, m, r))
    return (0.5 * (_nb_le1(k - 1.0, m, r) + _nb_le1(k, m, r)),
            0.5 * (_nb_gt1(k - 1.0, m, r) + _nb_gt1(k, m, r)))


def _nb_mid_a(k, mean, r):
    """Array _nb_mid1: the four tails of every element in one kernel call."""
    k, m, rr, bad, pois = _nb_prep(k, mean, r)
    k = _count(k)
    bad = bad | (k < 0) | np.isinf(k)
    ks = np.where(bad, 0.0, k)
    kk = np.stack((ks - 1.0, ks, ks - 1.0, ks))
    up = np.zeros(kk.shape, dtype=bool)
    up[2:] = True
    t = _nb_tail_a(kk, np.broadcast_to(m, kk.shape), np.broadcast_to(rr, kk.shape),
                   np.broadcast_to(pois, kk.shape), up)
    u = 0.5 * (t[0] + t[1])
    v = 0.5 * (t[2] + t[3])
    zero = m <= 0
    u = np.where(zero, np.where(ks == 0, 0.5, 1.0), u)
    v = np.where(zero, np.where(ks == 0, 0.5, 0.0), v)
    return np.where(bad, np.nan, u), np.where(bad, np.nan, v)


def nb_mid_u(k: ArrayLike, mean: ArrayLike, r: ArrayLike) -> ArrayLike:
    """Mid-distribution value u = P(X < k) + 0.5 P(X = k) = nb_cdf(k-1) + 0.5 nb_pmf(k).

    For the upper tail use 1 - u computed as nb_sf(k) + 0.5 nb_pmf(k) (stable).
    Evaluated as (nb_cdf(k-1) + nb_cdf(k)) / 2 (and 1 - u as (nb_sf(k-1) +
    nb_sf(k)) / 2): same-side tails only, so both are exact down to ~1e-300.
    k is rounded to the nearest integer first; k < 0 (an impossible count) -> NaN.
    """
    if isinstance(k, _SCALAR) and isinstance(mean, _SCALAR) and isinstance(r, _SCALAR):
        return _nb_mid1(k, mean, r)[0]
    return _out(_nb_mid_a(k, mean, r)[0])


def nb_midp(k: ArrayLike, mean: ArrayLike, r: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """(u, p_two_sided) for an observed count k under NB(mean, r).

    Both tails are computed from their own betainc so p down to ~1e-300 is
    exact (B04 test: Poisson mean 22, k = 1 -> P(X <= 1) = 6.4e-9).
    p is floored at P_FLOOR; invalid k (NaN, < 0) -> (NaN, NaN).
    """
    if isinstance(k, _SCALAR) and isinstance(mean, _SCALAR) and isinstance(r, _SCALAR):
        return _p2_1(*_nb_mid1(k, mean, r))
    u, v = _nb_mid_a(k, mean, r)
    return _out(u), _out(_clip_p(2.0 * np.minimum(u, v)))


def nb_size(kappa_hat: float, a_post: float) -> float:
    """Predictive NB size r = 1 / (1/kappa_hat + 1/a_post) (B04).

    kappa_hat: overdispersion (clipped to NB_KAPPA_CLIP); a_post: Gamma
    posterior shape. Combines extra-Poisson variance with parameter
    uncertainty. a_post <= 0 -> NaN; a_post = inf -> kappa_hat.
    """
    if isinstance(kappa_hat, _SCALAR) and isinstance(a_post, _SCALAR):
        kap, a = float(kappa_hat), float(a_post)
        if kap != kap or a != a or a <= 0.0:
            return _NAN
        kap = min(max(kap, NB_KAPPA_CLIP[0]), NB_KAPPA_CLIP[1])
        return 1.0 / (1.0 / kap + 1.0 / a)
    kap, a = np.broadcast_arrays(_arr(kappa_hat), _arr(a_post))
    with np.errstate(all="ignore"):
        r = 1.0 / (1.0 / np.clip(kap, *NB_KAPPA_CLIP) + 1.0 / a)
    return _out(np.where(a > 0, r, np.nan))


def _int_search(pred: Callable[[np.ndarray], np.ndarray], guess: np.ndarray,
                step: np.ndarray, lo_min: np.ndarray, hi_max: np.ndarray) -> np.ndarray:
    """Smallest integer k in (lo_min, hi_max] with pred(k) True, pred monotone
    (False ... True), given pred(lo_min) False and pred(hi_max) True (hi_max may
    be _INT_CAP, returned as inf when even that is False). Starts from the
    bracket [guess - step, guess + step - 1], widens it with doubling steps,
    then bisects: 2 vectorised pred calls when the guess is right."""
    lo = np.maximum(guess - step, lo_min)
    hi = np.minimum(np.maximum(guess + step - 1.0, lo + 1.0), hi_max)
    s = step.copy()
    for _ in range(64):                        # widen down until pred(lo) is False
        need = (lo > lo_min) & pred(lo)
        if not need.any():
            break
        hi = np.where(need, lo, hi)
        lo = np.where(need, np.maximum(lo - s, lo_min), lo)
        s = np.where(need, 2.0 * s, s)
    s = step.copy()
    ph = pred(hi)
    for _ in range(64):                        # widen up until pred(hi) is True
        need = (hi < hi_max) & ~ph
        if not need.any():
            break
        lo = np.where(need, hi, lo)
        hi = np.where(need, np.minimum(hi + s, hi_max), hi)
        s = np.where(need, 2.0 * s, s)
        ph = pred(hi)
    stuck = ~ph                                # only possible at hi == _INT_CAP
    for _ in range(64):
        gap = hi - lo > 1.0
        if not gap.any():
            break
        mid = np.floor(0.5 * (lo + hi))
        pm = pred(mid) & gap
        hi = np.where(pm, mid, hi)
        lo = np.where(gap & ~pm, mid, lo)
    return np.where(stuck, np.inf, hi)


def nb_ppf(q: ArrayLike, mean: ArrayLike, r: ArrayLike) -> ArrayLike:
    """Smallest integer k with nb_cdf(k) >= q (bisection on nb_cdf; model_state p5/p50/p95).

    q in [0, 1]: q <= 0 or mean <= 0 -> 0; q = 1 -> inf; q outside [0, 1] or
    NaN -> NaN. For q > 0.5 the test is nb_sf(k) <= 1 - q (upper tail on its
    own side; 1 - q is exact there). The search starts from the continuous
    inverse (scipy nbdtrik / pdtrik), so it usually costs 2 tail evaluations.
    """
    q, m, rr, bad, pois = _nb_prep(q, mean, r)
    bad = bad | (q < 0) | (q > 1)
    out = np.full(q.shape, np.nan)
    zero = ~bad & ((q <= 0) | (m <= 0))
    full = ~bad & ~zero & (q >= 1)
    out[zero] = 0.0
    out[full] = np.inf
    work = ~bad & ~zero & ~full
    if work.any():
        qw, mw, rw, pw = q[work], m[work], rr[work], pois[work]
        low = qw <= 0.5

        def pred(kk):                           # one tail per element, on its own side
            t = _nb_tail_a(kk, mw, rw, pw, ~low)
            return np.where(low, t >= qw, t <= 1.0 - qw)

        with np.errstate(all="ignore"):
            g = np.empty(qw.shape)
            g[pw] = sp.pdtrik(qw[pw], mw[pw])
            g[~pw] = sp.nbdtrik(qw[~pw], rw[~pw], 1.0 / (1.0 + mw[~pw] / rw[~pw]))
            sd = np.sqrt(mw + mw * mw / rw)    # Cornish-Fisher fallback if cdflib fails
            z = sp.ndtri(qw)
            cf = mw + sd * (z + (1.0 + 2.0 * mw / rw) / sd * (z * z - 1.0) / 6.0)
            g = np.ceil(np.where(np.isfinite(g), g, cf))
        g = np.clip(np.nan_to_num(g, nan=0.0, posinf=_INT_CAP), 0.0, _INT_CAP)
        out[work] = _int_search(pred, g, np.ones(g.shape), np.full(g.shape, -1.0),
                                np.full(g.shape, _INT_CAP))
    return _out(out)


# -------------------------------------------------------------- beta-binomial
# BB log pmf as rising-factorial differences D(x, d) = lnGamma(x+d) - lnGamma(x):
#     ln C(n,k) + D(a, k) + D(b, n-k) - D(a+b, n),   ln C(n,k) = D(n-j+1, j) - lnGamma(j+1)
# with j = min(k, n-k). betaln(k+a, n-k+b) - betaln(a, b) cancels two terms of
# size ~(a+b) ln(a+b): garbage once a + b > ~1e12 (bb_sf(5, 10, 1e300, 1e300)
# came out 1.53), and lnGamma(n+1) costs eps n ln n absolute at large n.
def _lgd1(x: float, d: float) -> float:
    """lnGamma(x + d) - lnGamma(x) for x > 0, d >= 0 (Stirling difference once x >= 10)."""
    if d == 0.0:
        return 0.0
    if x < _STIRLING_MIN:
        return math.lgamma(x + d) - math.lgamma(x)
    return ((x - 0.5) * math.log1p(d / x) + d * math.log(x + d) - d
            + _stirling_tail(x + d) - _stirling_tail(x))


def _lgd_a(x, d):
    """Array _lgd1 (x > 0, d >= 0 element-wise)."""
    with np.errstate(all="ignore"):
        big = x >= _STIRLING_MIN
        xb = np.where(big, x, _STIRLING_MIN)
        st = ((xb - 0.5) * np.log1p(d / xb) + d * np.log(xb + d) - d
              + _stirling_tail(xb + d) - _stirling_tail(xb))
        out = np.where(big, st, sp.gammaln(x + d) - sp.gammaln(x))
    return np.where(d == 0, 0.0, out)


def _bb_lpmf_a(k, n, a, b):
    """Array BB log pmf for integer 0 <= k <= n, a, b > 0 with a + b finite."""
    j = np.minimum(k, n - k)
    return (_lgd_a(n - j + 1.0, j) - sp.gammaln(j + 1.0)
            + _lgd_a(a, k) + _lgd_a(b, n - k) - _lgd_a(a + b, n))


def bb_logpmf(k: ArrayLike, n: ArrayLike, a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """log BB pmf = lnC(n,k) + lnB(k+a, n-k+b) - lnB(a,b), as rising-factorial
    differences (exact at any a + b, see _lgd1).

    k and n are rounded to the nearest integer; k outside [0, n] -> -inf;
    n < 0 or a, b <= 0 or NaN -> NaN.
    """
    k, n, a, b = np.broadcast_arrays(_count(_arr(k)), _count(_arr(n)), _arr(a), _arr(b))
    with np.errstate(over="ignore"):
        bad = (np.isnan(k) | np.isnan(n) | np.isnan(a) | np.isnan(b) | (n < 0) | (a <= 0)
               | (b <= 0) | ~np.isfinite(a + b))
    inside = (k >= 0) & (k <= n) & ~bad & np.isfinite(n)
    ks = np.where(inside, k, 0.0)
    ns = np.where(inside, n, 0.0)
    lp = _bb_lpmf_a(ks, ns, np.where(bad, 1.0, a), np.where(bad, 1.0, b))
    lp = np.where(inside, lp, -np.inf)
    return _out(np.where(bad, np.nan, lp))


def _bb_lpmf1(k: float, n: float, a: float, b: float) -> float:
    j = min(k, n - k)
    return (_lgd1(n - j + 1.0, j) - math.lgamma(j + 1.0)
            + _lgd1(a, k) + _lgd1(b, n - k) - _lgd1(a + b, n))


def _bb_side_sum(k: float, n: float, a: float, b: float, lpk: float, upper: bool,
                 n_terms: float) -> Tuple[float, float, bool]:
    """Exact sum of the pmf strictly beyond k on one side, from log pmf(k) by
    the ratio recurrence pmf(j+1)/pmf(j) = (n-j)(j+a)/((j+1)(n-j-1+b)).

    Returns (sum, L, converged) over the L terms nearest k. Converged when the
    whole side was summed, or the last term is below _BB_TAIL_REL of the sum
    while still falling. One pass of <= _BB_WIN0 terms, then one of <=
    BB_EXACT_MAX_N; the caller adds a Beta remainder beyond L if not converged.
    """
    if n_terms <= _BB_PY_TERMS:                # tiny side: plain Python beats numpy dispatch
        s, lp, i = 0.0, lpk, k if upper else k - 1.0
        for _ in range(int(n_terms)):
            if upper:
                x = (n - i) / (i + 1.0) * ((i + a) / (n - i - 1.0 + b))
                i += 1.0
            else:
                x = (i + 1.0) / (n - i) * ((n - i - 1.0 + b) / (i + a))
                i -= 1.0
            lp += math.log(x) if x > 0.0 else -_INF    # underflow (a ~ 1e-320): term is 0
            s += math.exp(lp)
        return s, n_terms, True
    L = min(n_terms, _BB_WIN0)
    while True:
        if upper:                              # ratios i -> i+1, i = k .. k+L-1
            i = np.arange(k, k + L)
            x = (n - i) / (i + 1.0) * ((i + a) / (n - i - 1.0 + b))
        else:                                  # pmf(i)/pmf(i+1), i = k-1 .. k-L
            i = np.arange(k - 1.0, k - 1.0 - L, -1.0)
            x = (i + 1.0) / (n - i) * ((n - i - 1.0 + b) / (i + a))
        with np.errstate(divide="ignore"):
            lr = np.log(x)
        t = np.exp(lpk + np.cumsum(lr))
        s = float(t.sum())
        if L >= n_terms or (t[-1] <= _BB_TAIL_REL * s and lr[-1] < 0.0):
            return s, L, True
        if L >= BB_EXACT_MAX_N:
            return s, L, False
        L = min(n_terms, float(BB_EXACT_MAX_N))


def _bb_jitter_beta(n: float, a: float, b: float) -> Tuple[float, float]:
    """Beta(al, be) moment-matched to the jittered count (K + U)/(n + 1),
    U ~ U(0,1): the cells [i, i+1)/(n+1) tile [0, 1] exactly."""
    c = a + b
    mu = a / c
    var_k = n * mu * (1.0 - mu) * (c + n) / (c + 1.0)
    mm = (n * mu + 0.5) / (n + 1.0)
    vv = (var_k + 1.0 / 12.0) / (n + 1.0) ** 2
    cc = max(mm * (1.0 - mm) / vv - 1.0, 1e-9)
    return mm * cc, (1.0 - mm) * cc


def _bb_beta_tail1(j: float, n: float, a: float, b: float, upper: bool) -> float:
    """P(K > j) (upper) or P(K < j) from the jittered Beta: P(K < j) = F(j/(n+1))
    and P(K > j) = 1 - F((j+1)/(n+1)), each evaluated on its own side. Keeps
    the BB skewness; only used where the pmf is smooth on the integer grid (a
    tail wider than BB_EXACT_MAX_N)."""
    al, be = _bb_jitter_beta(n, a, b)
    if upper:
        return float(_c_betainc(be, al, (n - j) / (n + 1.0)))
    return float(_c_betainc(al, be, j / (n + 1.0)))


def _bb_parts1(k: float, n: float, a: float, b: float) -> Tuple[float, float, float]:
    """(P(K < k), P(K = k), P(K > k)) for integer 0 <= k <= n, n >= 1, a, b > 0.

    The tail away from the mean (the small side) is summed exactly term by
    term near k; only the part more than BB_EXACT_MAX_N terms beyond k, if
    still non-negligible, comes from the Beta approximation. The other side is
    the complement (never small, so no cancellation).
    """
    lpk = _bb_lpmf1(k, n, a, b)
    eq = math.exp(lpk)
    upper = k >= n * (a / (a + b))
    side, L, ok = _bb_side_sum(k, n, a, b, lpk, upper, n - k if upper else k)
    if not ok:
        side += _bb_beta_tail1(k + L if upper else k - L, n, a, b, upper)
    other = max(0.0, 1.0 - side - eq)
    return (other, eq, side) if upper else (side, eq, other)


def _bb_prep(k, n, a, b, *, k_round: bool):
    k, n, a, b = np.broadcast_arrays(_arr(k), _count(_arr(n)), _arr(a), _arr(b))
    k = _count(k) if k_round else np.floor(k)
    with np.errstate(over="ignore"):           # a + b overflowing is itself invalid
        bad = (np.isnan(k) | np.isnan(n) | np.isnan(a) | np.isnan(b) | ~np.isfinite(n)
               | (n <= 0) | (a <= 0) | (b <= 0) | ~np.isfinite(a + b))
    return k, n, a, b, bad


def _bb_map(k, n, a, b, fn, mask) -> Tuple[np.ndarray, ...]:
    """Evaluate the scalar kernel _bb_parts1 on the masked elements."""
    lt = np.full(k.shape, np.nan)
    eq = np.full(k.shape, np.nan)
    gt = np.full(k.shape, np.nan)
    for idx in np.ndindex(k.shape):
        if mask[idx]:
            lt[idx], eq[idx], gt[idx] = fn(float(k[idx]), float(n[idx]), float(a[idx]),
                                           float(b[idx]))
    return lt, eq, gt


def _bb_is_scalar(k, n, a, b) -> bool:
    return (isinstance(k, _SCALAR) and isinstance(n, _SCALAR) and isinstance(a, _SCALAR)
            and isinstance(b, _SCALAR))


def _bb_bad1(k: float, n: float, a: float, b: float) -> bool:
    return (k != k or not (0.0 < n < _INF) or not (0.0 < a < _INF)
            or not (0.0 < b < _INF) or a + b == _INF)


def bb_cdf(k: float, n: float, a: float, b: float) -> float:
    """P(K <= floor(k)). Exact (summed from k outward over at most
    BB_EXACT_MAX_N terms, so always for n <= BB_EXACT_MAX_N); a tail wider
    than that uses the moment-matched Beta approximation (module docstring:
    mean = n a/(a+b), var = n a b (a+b+n) / ((a+b)^2 (a+b+1))).
    n <= 0 -> NaN (never p = 1). Vectorised (element loop).
    """
    if _bb_is_scalar(k, n, a, b):
        k, n, a, b = _fl1(float(k)), _rd1(float(n)), float(a), float(b)
        if _bb_bad1(k, n, a, b):
            return _NAN
        if k < 0.0:
            return 0.0
        if k >= n:
            return 1.0
        lt, eq, _ = _bb_parts1(k, n, a, b)
        return min(1.0, lt + eq)
    k, n, a, b, bad = _bb_prep(k, n, a, b, k_round=False)
    body = ~bad & (k >= 0) & (k < n)
    lt, eq, _ = _bb_map(k, n, a, b, _bb_parts1, body)
    out = np.where(k >= n, 1.0, np.where(k < 0, 0.0, np.minimum(1.0, lt + eq)))
    return _out(np.where(bad, np.nan, out))


def bb_sf(k: float, n: float, a: float, b: float) -> float:
    """P(K > floor(k)), summed over the upper tail directly for accuracy (see bb_cdf)."""
    if _bb_is_scalar(k, n, a, b):
        k, n, a, b = _fl1(float(k)), _rd1(float(n)), float(a), float(b)
        if _bb_bad1(k, n, a, b):
            return _NAN
        if k < 0.0:
            return 1.0
        if k >= n:
            return 0.0
        return min(1.0, _bb_parts1(k, n, a, b)[2])
    k, n, a, b, bad = _bb_prep(k, n, a, b, k_round=False)
    body = ~bad & (k >= 0) & (k < n)
    _, _, gt = _bb_map(k, n, a, b, _bb_parts1, body)
    out = np.where(k >= n, 0.0, np.where(k < 0, 1.0, np.minimum(1.0, gt)))
    return _out(np.where(bad, np.nan, out))


def bb_midp(k: float, n: float, a: float, b: float) -> Tuple[float, float]:
    """(u, p_two_sided) with u = P(K < k) + 0.5 P(K = k); n = 0 -> (NaN, NaN).

    B04 checks: k=1,n=2 vs mean 0.1, c=50 -> upper sf 0.188; k=100,n=200 vs
    0.1: phi=50 -> sf 1.3e-8, phi=20 -> sf 7.7e-5.
    k, n rounded to the nearest integer; k outside [0, n] -> (NaN, NaN).
    """
    if _bb_is_scalar(k, n, a, b):
        k, n, a, b = _rd1(float(k)), _rd1(float(n)), float(a), float(b)
        if _bb_bad1(k, n, a, b) or not (0.0 <= k <= n):
            return _NAN, _NAN
        lt, eq, gt = _bb_parts1(k, n, a, b)
        return _p2_1(lt + 0.5 * eq, gt + 0.5 * eq)
    k, n, a, b, bad = _bb_prep(k, n, a, b, k_round=True)
    bad = bad | (k < 0) | (k > n)
    lt, eq, gt = _bb_map(k, n, a, b, _bb_parts1, ~bad)
    u, v = lt + 0.5 * eq, gt + 0.5 * eq
    return _out(u), _out(_clip_p(2.0 * np.minimum(u, v)))


def bb_params(p_hat: float, c: float) -> Tuple[float, float]:
    """(a, b) = (p_hat c, (1 - p_hat) c), with p_hat clipped to [1e-6, 1 - 1e-6].

    c <= 0 or NaN -> (NaN, NaN).
    """
    if isinstance(p_hat, _SCALAR) and isinstance(c, _SCALAR):
        p, c = float(p_hat), float(c)
        if p != p or c != c or c <= 0.0:
            return _NAN, _NAN
        p = min(max(p, BB_P_CLIP), 1.0 - BB_P_CLIP)
        return p * c, (1.0 - p) * c
    p, c = np.broadcast_arrays(np.clip(_arr(p_hat), BB_P_CLIP, 1.0 - BB_P_CLIP), _arr(c))
    c = np.where(c > 0, c, np.nan)
    return _out(p * c), _out((1.0 - p) * c)


def _bb_ppf_group(qs: np.ndarray, n: float, a: float, b: float) -> np.ndarray:
    """bb_ppf for several q in (0, 1) sharing one (n, a, b)."""
    if n <= _BB_PPF_GRID_MAX:                   # whole support once: exact cumulative sums
        j = np.arange(n + 1.0)
        pm = np.exp(_bb_lpmf_a(j, n, a, b))
        cdf = np.cumsum(pm)
        sf = np.concatenate((np.cumsum(pm[::-1])[::-1][1:], [0.0]))   # P(K > j), from the top
        out = np.empty(qs.shape)
        for i, q in enumerate(qs):
            hit = cdf >= q if q <= 0.5 else sf <= 1.0 - q
            out[i] = float(np.argmax(hit)) if hit.any() else n
        return out
    # large n: search on the hybrid cdf / sf from the jittered-Beta quantile
    al, be = _bb_jitter_beta(n, a, b)
    c = a + b
    mu = a / c
    sd = math.sqrt(n * mu * (1.0 - mu) * (c + n) / (c + 1.0))
    step = np.array([max(1.0, math.ceil(0.05 * sd))])
    out = np.empty(qs.shape)
    for i, q in enumerate(qs):
        def pred(kk, q=q):
            return np.array([bb_cdf(float(x), n, a, b) >= q if q <= 0.5
                             else bb_sf(float(x), n, a, b) <= 1.0 - q for x in kk])
        x = (n + 1.0) * float(sp.betaincinv(al, be, q)) - 1.0
        g = float(min(max(math.ceil(x), 0), n)) if math.isfinite(x) else float(math.floor(n * mu))
        out[i] = _int_search(pred, np.array([g]), step, np.array([-1.0]), np.array([n]))[0]
    return out


def bb_ppf(q: float, n: float, a: float, b: float) -> float:
    """Smallest k in [0, n] with bb_cdf(k) >= q (q > 0.5: bb_sf(k) <= 1 - q).

    q <= 0 -> 0, q >= 1 -> n; q outside [0, 1], n <= 0 or NaN -> NaN.
    Quantiles sharing (n, a, b) (e.g. q = [.05, .5, .95]) reuse one pmf pass.
    """
    q_, n_, a_, b_ = np.broadcast_arrays(_arr(q), _count(_arr(n)), _arr(a), _arr(b))
    with np.errstate(over="ignore"):
        bad = (np.isnan(q_) | (q_ < 0) | (q_ > 1) | np.isnan(n_) | ~np.isfinite(n_) | (n_ <= 0)
               | ~(a_ > 0) | ~(b_ > 0) | ~np.isfinite(a_ + b_))
    out = np.full(q_.shape, np.nan)
    groups: dict = {}
    for idx in np.ndindex(q_.shape):
        if bad[idx]:
            continue
        qi, ni = float(q_[idx]), float(n_[idx])
        if qi <= 0.0:
            out[idx] = 0.0
        elif qi >= 1.0:
            out[idx] = ni
        else:
            groups.setdefault((ni, float(a_[idx]), float(b_[idx])), []).append(idx)
    for (ni, ai, bi), idxs in groups.items():
        res = _bb_ppf_group(np.array([float(q_[i]) for i in idxs]), ni, ai, bi)
        for i, v in zip(idxs, res):
            out[i] = v
    return _out(out)


# ----------------------------------------------------------------- student-t
def nig_predictive(post: NIG) -> Tuple[float, float, float]:
    """(df, loc, scale) of the one-step Student-t predictive:
    df = 2 alpha, loc = m, scale = sqrt(beta (kappa + 1) / (alpha kappa)).

    alpha <= 0, kappa <= 0 or beta < 0 -> scale NaN. Fields may be arrays.
    """
    if (isinstance(post.alpha, _SCALAR) and isinstance(post.kappa, _SCALAR)
            and isinstance(post.beta, _SCALAR) and isinstance(post.m, _SCALAR)):
        al, ka, be = float(post.alpha), float(post.kappa), float(post.beta)
        if al > 0.0 and ka > 0.0 and be >= 0.0:
            scale = math.sqrt(be * (ka + 1.0) / (al * ka)) if ka != _INF else math.sqrt(be / al)
        else:
            scale = _NAN
        return 2.0 * al, float(post.m), scale
    al, ka, be = _arr(post.alpha), _arr(post.kappa), _arr(post.beta)
    with np.errstate(all="ignore"):
        s2 = be * (1.0 + 1.0 / ka) / al
    scale = np.sqrt(np.where((al > 0) & (ka > 0) & (be >= 0), s2, np.nan))
    return _out(2.0 * al), _out(_arr(post.m)), _out(scale)


def t_cdf(x: ArrayLike, df: float, loc: float, scale: float) -> ArrayLike:
    """Student-t cdf via scipy.special.stdtr(df, (x - loc)/scale); NaN -> NaN.

    scale <= 0 -> NaN. df = inf is the normal cdf.
    """
    if (isinstance(x, _SCALAR) and isinstance(df, _SCALAR) and isinstance(loc, _SCALAR)
            and isinstance(scale, _SCALAR)):
        s = float(scale)
        if not s > 0.0:
            return _NAN
        return float(_c_stdtr(float(df), (float(x) - float(loc)) / s))
    x, df, loc, s = np.broadcast_arrays(_arr(x), _arr(df), _arr(loc), _arr(scale))
    with np.errstate(all="ignore"):
        out = sp.stdtr(df, (x - loc) / s)
    return _out(np.where(s > 0, out, np.nan))


def t_midp(x: ArrayLike, post: NIG) -> Tuple[ArrayLike, ArrayLike]:
    """(u, p_two_sided) of x under the NIG predictive; continuous so u = cdf.

    The upper tail is taken as stdtr(df, -(x-loc)/scale) to avoid cancellation.
    p is floored at P_FLOOR; NaN x or an invalid posterior -> (NaN, NaN).
    """
    df, loc, scale = nig_predictive(post)
    if isinstance(x, _SCALAR) and isinstance(df, float) and isinstance(scale, float):
        if not scale > 0.0 or x != x:
            return _NAN, _NAN
        z = (float(x) - loc) / scale          # NaN loc -> NaN z -> (NaN, NaN), never p = 1e-300
        return _p2_1(float(_c_stdtr(df, z)), float(_c_stdtr(df, -z)))
    x, df, loc, s = np.broadcast_arrays(_arr(x), _arr(df), _arr(loc), _arr(scale))
    with np.errstate(all="ignore"):
        z = (x - loc) / s
        u = sp.stdtr(df, z)
        v = sp.stdtr(df, -z)
    ok = s > 0
    u = np.where(ok, u, np.nan)
    p = np.where(ok, _clip_p(2.0 * np.minimum(u, v)), np.nan)
    return _out(u), _out(p)


def t_ppf(q: ArrayLike, post: NIG) -> ArrayLike:
    """Predictive quantile loc + scale * stdtrit(df, q).

    q = 0 -> -inf, q = 1 -> +inf (stdtrit returns +inf at 0), q outside [0, 1] -> NaN.
    """
    df, loc, scale = nig_predictive(post)
    q, df, loc, s = np.broadcast_arrays(_arr(q), _arr(df), _arr(loc), _arr(scale))
    with np.errstate(all="ignore"):
        t = sp.stdtrit(df, np.clip(q, 0.0, 1.0))
        t = np.where(q <= 0, -np.inf, np.where(q >= 1, np.inf, t))
        out = loc + s * t
    out = np.where((q < 0) | (q > 1) | ~(s > 0), np.nan, out)
    return _out(out)


# ------------------------------------------------------- posterior updates
def gamma_rate_posterior(a0: float, b0: float, sum_x: float, sum_e: float) -> GammaRate:
    """Gamma-Poisson conjugate update on weighted sufficient stats.

    a = a0 + sum_x (weighted events), b = b0 + sum_e (weighted exposure,
    minutes). Predictive for a tick of exposure e minutes: NB(mean = a/b * e,
    size = a) before overdispersion (see nb_size). Sums are floored at 0
    (decayed float stats can dip to -1e-17); NaN propagates. Vectorised.
    """
    sx, se = np.maximum(_arr(sum_x), 0.0), np.maximum(_arr(sum_e), 0.0)
    return GammaRate(_out(_arr(a0) + sx), _out(_arr(b0) + se))


def count_overdispersion(W: float, sx: float, se: float, sxx: float, sxe: float,
                         see: float) -> float:
    """Moment estimate kappa_hat of NB size from the B03 count stats
    [W, sum x, sum e, sum x^2, sum x e, sum e^2] (weights already applied).

        mu = sx / se                           (rate per minute)
        resid = (sxx - 2 mu sxe + mu^2 see) / W  (mean squared residual)
        excess = resid - mu * se / W             (variance beyond Poisson)
        kappa_hat = mu^2 (see / W) / excess      if excess > 0 else 1e3
    clipped to NB_KAPPA_CLIP. W <= 1 or se <= 0 -> 1e3 (Poisson); mu = 0 ->
    1e3 (no evidence of overdispersion); NaN stats -> NaN. Vectorised.
    """
    W, sx, se, sxx, sxe, see = np.broadcast_arrays(*(_arr(v) for v in (W, sx, se, sxx, sxe, see)))
    bad = np.isnan(W) | np.isnan(sx) | np.isnan(se) | np.isnan(sxx) | np.isnan(sxe) | np.isnan(see)
    with np.errstate(all="ignore"):
        mu = sx / se
        resid = np.maximum((sxx - 2.0 * mu * sxe + mu * mu * see) / W, 0.0)
        excess = resid - mu * se / W
        kap = mu * mu * (see / W) / excess
    ok = (W > 1) & (se > 0) & (mu > 0) & (excess > 0) & np.isfinite(kap)
    kap = np.clip(np.where(ok, kap, _KAPPA_POISSON), *NB_KAPPA_CLIP)
    return _out(np.where(bad, np.nan, kap))


def ratio_posterior(a0: float, b0: float, W: float, sk: float, sn: float,
                    skk_n: float, sr2: float) -> Tuple[float, float]:
    """(p_hat, phi_hat) from the B03 ratio stats [W, sum k, sum n, sum k^2/n, sum (k/n)^2].

        p_hat = (sk + a0) / (sn + a0 + b0)
        m = sk / sn;  S = skk_n - m sk          (n-weighted between-row SS)
        nbar = sn / W
        rho = (S / ((W - 1) m (1 - m)) - 1) / (nbar - 1)   (intra-class corr.)
        phi_hat = 1/rho - 1, clipped to BB_PHI_CLIP; rho <= 0 or W < 3 -> 1000.
    The caller uses concentration c = min(sn + 2, phi_hat).
    No information (sn <= 0, m in {0, 1}, nbar <= 1) also gives 1000; sr2 is
    carried for the unweighted variant and unused; NaN stats -> (NaN, NaN).
    """
    a0, b0, W, sk, sn, skk_n = np.broadcast_arrays(*(_arr(v) for v in (a0, b0, W, sk, sn, skk_n)))
    bad = (np.isnan(a0) | np.isnan(b0) | np.isnan(W) | np.isnan(sk) | np.isnan(sn)
           | np.isnan(skk_n))
    with np.errstate(all="ignore"):
        p_hat = (np.maximum(sk, 0.0) + a0) / (np.maximum(sn, 0.0) + a0 + b0)
        m = sk / sn
        S = np.maximum(skk_n - m * sk, 0.0)
        nbar = sn / W
        rho = (S / ((W - 1.0) * m * (1.0 - m)) - 1.0) / (nbar - 1.0)
        phi = 1.0 / rho - 1.0
    ok = ((W >= 3) & (sn > 0) & (m > 0) & (m < 1) & (nbar > 1) & (rho > 0) & ~np.isnan(phi))
    phi = np.clip(np.where(ok, phi, _PHI_NO_INFO), *BB_PHI_CLIP)
    return _out(np.where(bad, np.nan, p_hat)), _out(np.where(bad, np.nan, phi))


def nig_posterior(prior: NIG, W: float, sx: float, sxx: float) -> NIG:
    """Weighted NIG conjugate update from (W = sum w, sx = sum w x, sxx = sum w x^2).

        kappa = kappa0 + W;  m = (kappa0 m0 + sx) / kappa
        alpha = alpha0 + W / 2
        beta  = beta0 + 0.5 (sxx - sx^2 / W) + kappa0 W (sx/W - m0)^2 / (2 kappa)
    W = 0 returns the prior. The within term is floored at 0 (float error).
    W < 0 (float residue of decay/rollback) is treated as 0. Vectorised.
    """
    m0, k0, a0, b0 = _arr(prior.m), _arr(prior.kappa), _arr(prior.alpha), _arr(prior.beta)
    W, sx, sxx = _arr(W), _arr(sx), _arr(sxx)
    has = W > 0
    Ws = np.where(has, W, 1.0)
    sxs = np.where(has, sx, 0.0)
    with np.errstate(all="ignore"):
        kappa = k0 + Ws
        m = (k0 * m0 + sxs) / kappa
        within = np.maximum(np.where(has, sxx, 0.0) - sxs * sxs / Ws, 0.0)
        dev = sxs - Ws * m0                    # = W (xbar - m0)
        beta = b0 + 0.5 * within + k0 * dev * dev / (2.0 * Ws * kappa)
    nan = np.isnan(W) | np.isnan(sx) | np.isnan(sxx)

    def pick(post_v, prior_v):
        return _out(np.where(nan, np.nan, np.where(has, post_v, prior_v)))

    return NIG(pick(m, m0), pick(kappa, k0), pick(a0 + 0.5 * Ws, a0), pick(beta, b0))


def _nig_stats(post: NIG, prior: NIG) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert nig_posterior: (W, sx, sxx) of `post` relative to `prior`."""
    m0, k0, b0 = _arr(prior.m), _arr(prior.kappa), _arr(prior.beta)
    W = np.maximum(_arr(post.kappa) - k0, 0.0)
    has = W > 0
    Ws = np.where(has, W, 1.0)
    with np.errstate(all="ignore"):
        sx = np.where(has, _arr(post.kappa) * _arr(post.m) - k0 * m0, 0.0)
        dev = sx - Ws * m0
        between = k0 * dev * dev / (2.0 * Ws * _arr(post.kappa))
        within = np.maximum(2.0 * (_arr(post.beta) - b0 - between), 0.0)
        sxx = np.where(has, within + sx * sx / Ws, 0.0)
    return W, sx, sxx


def nig_merge(a: NIG, b: NIG, prior: NIG, w_b: float = 1.0) -> NIG:
    """Posterior from the sufficient stats of a plus w_b x the stats of b
    (both relative to `prior`). Used for link seeding (B := B_own + 0.5 A)
    and hierarchical pooling. Stats are recovered by inverting nig_posterior
    (W = kappa - kappa0, sx = kappa m - kappa0 m0, sxx from beta).
    """
    Wa, sxa, sxxa = _nig_stats(a, prior)
    Wb, sxb, sxxb = _nig_stats(b, prior)
    w = _arr(w_b)
    return nig_posterior(prior, _out(Wa + w * Wb), _out(sxa + w * sxb), _out(sxxa + w * sxxb))
