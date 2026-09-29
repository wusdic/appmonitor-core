"""Sequential statistics with wall-clock thresholds (B07, B14, B16, B25).

STATUS: implemented. Signatures, constants layout and maths are frozen
(docs/lib3/helpers_api.md).

Why wall-clock: a fixed per-tick CUSUM threshold changes its false-alarm
rate 15-60x between 60 s, 900 s and 3600 s cadences. Every threshold here is
derived from a target ARL in *days*, converted to ticks with arl_ticks(), so
the null alarm budget per entity-day is cadence independent.

Per-feature charts are AR(1)-prewhitened because residual autocorrelation
shortens real run lengths; a round-robin block-bootstrap audit (B14)
corrects what the Gaussian formula misses (at most x1.5 on h).

Threshold families and how accurate they are (measured by simulation in
tests/lib/test_seq.py):
  * h_gauss: Siegmund's corrected-diffusion ARL for N(0, 1) increments. The
    realised zero-state ARL is within ~5% of target for k in [0.25, 1] and
    ARL >= 3e2 (a little long at k = 0.25, a little short at k = 1).
  * h_evidence: exact for S = max(0, S + Exp(1) - 3) (q ~ U(0, 1) under the
    null): the adjustment coefficient of Exp(1) - 3 is 0.94, the constant
    3.07 comes from the integral equation.
  * h_llr, bernoulli_threshold: Wald / Lorden bounds, conservative (for
    Gaussian LLRs ~6x, for rare-activity Bernoulli bins 3-10x the target:
    the discrete +log2(p1/p0) jumps overshoot h).
  * mcusum_h: Monte-Carlo table (below), within ~10% on ARL up to 1e6.

Why the solve is in log space: at 60 s ticks the in-control ARL is ~3.5e6
and exp(2 k b) overflows float64 for 2 k b > 709 (k = 1, b > 355, reachable
by a misconfigured ARL); ln(expm1(y) - y) is evaluated stably for tiny y
(series) and huge y (y + log1p(-(1 + y) e^-y)).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

SECONDS_PER_DAY = 86400.0
SIEGMUND_RHO = 0.583                # overshoot correction; 2 rho = 1.166
EVIDENCE_A, EVIDENCE_B = 3.07, 0.94  # h = (ln ARL - 3.07) / 0.94 for S = max(0, S - ln q - 3)
EVIDENCE_DRIFT = 3.0
AR1_PHI_CLIP = (0.0, 0.8)
PSI_CLIP = 3.0                      # prewhitened innovations are clipped to +-3
RHYTHM_P0_MAX = 0.3                 # offhours CUSUM only over bins with p0 <= 0.3
MCUSUM_K = 0.5

H_MAX = 200.0                       # h_gauss search interval is [0, H_MAX]
P_FLOOR = 1e-300                    # same floor as combine / calib
BERN_P_CLIP = (1e-6, 1.0 - 1e-6)
RHYTHM_P1_BOUNDS = (0.5, 0.95)
AR1_MIN_PAIRS = 10

# ---- Crosier MCUSUM thresholds (B14) --------------------------------------
# h depends on the dimension d (1..16) and the in-control ARL in ticks.
# Filled by Monte-Carlo (_mcusum_arl_mc: N(0, I_d) inputs, k = 0.5, zero-state
# ARL, alarm when ||S'|| > h): 2000 chains x 1.2e5 ticks per d, first-passage
# times for a 0.05-spaced h grid read off each chain's running maximum, ARL(h)
# by the censored-exponential estimator sum(min(T, Tmax)) / #crossed. Columns
# 1e2..1e5 come from a local quadratic fit of h against ln ARL (+-0.6 around
# the grid point); 1e6 and 1e7 extrapolate with the radial asymptote
#     ln ARL(h) = a + 2 k h - beta ln h      (k = 0.5, slope fixed at 2k),
# (a, beta) fitted on the simulated ARL in [5e3, ~2e5] and a re-anchored on
# the 1e5 column. Why not a straight line in ln ARL: the radial part of
# Crosier's statistic is a random walk with drift (d - 1)/(2 r) - k, so its
# stationary law is ~Gamma(d, 2k) and ln ARL bends like h - (d - 1) ln h (the
# fit gives beta = d - 1 +- 0.4 for every d); a straight line through the
# 1e4-1e5 slope overstates h at 1e7 by ~1 at d = 16. Validation with 1000
# chains x 1.5e6 ticks: the realised ARL at the 1e6 column is 0.96e6 (d = 4)
# and 1.08e6 (d = 12). MC noise is ~2-5% in ARL (up to ~0.1 in h at ARL 1e2,
# large d, where h is flat in ln ARL). Sanity: d = 2, h = 5.5 gives ARL ~ 200
# (Crosier 1988). mcusum_h interpolates linearly in ln ARL between columns;
# h(ln ARL) is concave, so the chord is slightly low: realised ARL 0.89-1.02x
# target between columns (worst mid 1e2-1e3 at d >= 10), 0.93-1.01x at the
# 100-day operating points 2400 / 9600 / 1.44e5 ticks (3600 / 900 / 60 s).
MCUSUM_ARL_GRID: Tuple[float, ...] = (1e2, 1e3, 1e4, 1e5, 1e6, 1e7)
MCUSUM_H: np.ndarray = np.array([
    # ARL: 1e2    1e3    1e4    1e5    1e6    1e7
    [3.23, 5.45, 7.73, 10.07, 12.39, 14.70],     # d = 1
    [4.66, 7.39, 9.96, 12.52, 15.02, 17.49],     # d = 2
    [5.95, 9.05, 11.89, 14.65, 17.32, 19.93],    # d = 3
    [7.10, 10.57, 13.68, 16.55, 19.32, 22.02],   # d = 4
    [8.26, 12.05, 15.33, 18.41, 21.33, 24.15],   # d = 5
    [9.18, 13.39, 16.95, 20.14, 23.16, 26.06],   # d = 6
    [10.23, 14.68, 18.48, 21.80, 24.93, 27.94],  # d = 7
    [11.26, 15.98, 19.92, 23.36, 26.59, 29.68],  # d = 8
    [12.25, 17.35, 21.41, 24.98, 28.32, 31.50],  # d = 9
    [13.08, 18.50, 22.80, 26.52, 29.97, 33.24],  # d = 10
    [13.93, 19.78, 24.24, 28.08, 31.62, 34.97],  # d = 11
    [14.97, 20.96, 25.58, 29.56, 33.21, 36.65],  # d = 12
    [15.82, 22.18, 26.96, 31.02, 34.74, 38.24],  # d = 13
    [16.60, 23.36, 28.28, 32.54, 36.40, 40.02],  # d = 14
    [17.42, 24.58, 29.59, 33.90, 37.79, 41.44],  # d = 15
    [18.28, 25.74, 30.95, 35.33, 39.33, 43.07],  # d = 16
], dtype=np.float64)
MCUSUM_H.setflags(write=False)                  # shared constant: never mutate
_LN_MCUSUM_GRID = np.log(np.asarray(MCUSUM_ARL_GRID, dtype=np.float64))


# ------------------------------------------------------------- thresholds
def arl_ticks(arl_days: float, dt_s: float) -> float:
    """In-control ARL in ticks for a wall-clock ARL: arl_days * 86400 / dt_s."""
    dt = float(dt_s)
    if dt <= 0.0:
        raise ValueError(f"seq.arl_ticks: dt_s must be > 0, got {dt_s!r}")
    return float(arl_days) * SECONDS_PER_DAY / dt


def _ln_expm1_minus(y: float) -> float:
    """ln(e^y - 1 - y) for y > 0, stable at both ends."""
    if y < 1e-2:
        # e^y - 1 - y = y^2/2 (1 + y/3 + y^2/12 + y^3/60 + ...)
        return 2.0 * math.log(y) - math.log(2.0) + math.log1p(
            y / 3.0 + y * y / 12.0 + y ** 3 / 60.0 + y ** 4 / 360.0)
    if y > 30.0:
        return y + math.log1p(-(1.0 + y) * math.exp(-y))
    return math.log(math.expm1(y) - y)


def _dln_expm1_minus(y: float) -> float:
    """d/dy ln(e^y - 1 - y) = expm1(y) / (e^y - 1 - y)."""
    if y > 30.0:
        return 1.0 / (1.0 - (1.0 + y) * math.exp(-y))
    if y < 1e-2:
        # expm1(y)/(y^2/2 (1 + y/3 + ...)) ~ 2/y (1 + y/2 + y^2/6)/(1 + y/3 + y^2/12)
        return (2.0 / y) * (1.0 + y / 2.0 + y * y / 6.0) / (1.0 + y / 3.0 + y * y / 12.0)
    return math.expm1(y) / (math.expm1(y) - y)


def _solve_expm1_minus(ln_c: float, lo: float, hi: float) -> float:
    """Root y in [lo, hi] of ln(e^y - 1 - y) = ln_c (g increasing, concave).

    Safeguarded Newton: a step leaving the bracket is replaced by bisection.
    Converges in ~5-8 iterations over the whole working range.
    """
    y = 0.5 * (lo + hi)
    for _ in range(200):
        g = _ln_expm1_minus(y) - ln_c
        if g > 0.0:
            hi = y
        else:
            lo = y
        if abs(g) < 1e-13 or hi - lo < 1e-13 * max(1.0, hi):
            break
        step = g / _dln_expm1_minus(y)
        y_new = y - step
        if not (lo < y_new < hi):
            y_new = 0.5 * (lo + hi)
        y = y_new
    return y


def h_gauss(k: float, arl_ticks: float) -> float:
    """One-sided CUSUM threshold for N(0, 1) inputs, reference k, from Siegmund:

        ARL0(h) = (exp(2 k b) - 2 k b - 1) / (2 k^2),  b = h + 1.166
    solved for h by Newton / bisection on [0, 200] (monotone). Checks
    (B14): arl = 2400 d at 900 s -> h = 19.4 (k = .25), 5.35 (k = 1); at 60 s
    24.8 / 6.7; at 3600 s 16.6 / 4.66. O(1).

    k = 0 uses the k -> 0 limit ARL0 = b^2. ARL at or below ARL0(h = 0)
    returns 0; a NaN input returns NaN; k < 0 raises ValueError.
    """
    k = float(k)
    arl = float(arl_ticks)
    if math.isnan(k) or math.isnan(arl):
        return float("nan")
    if k < 0.0:
        raise ValueError(f"seq.h_gauss: k must be >= 0, got {k!r}")
    b0 = 2.0 * SIEGMUND_RHO
    if arl <= 0.0:
        return 0.0
    if math.isinf(arl):
        return H_MAX
    if k == 0.0:
        return min(H_MAX, max(0.0, math.sqrt(arl) - b0))
    # Solve e^y - 1 - y = c with y = 2 k b, c = 2 k^2 ARL (in log space).
    ln_c = math.log(2.0 * k * k) + math.log(arl)
    y_lo = 2.0 * k * b0
    y_hi = 2.0 * k * (H_MAX + b0)
    if ln_c <= _ln_expm1_minus(y_lo):
        return 0.0
    if ln_c >= _ln_expm1_minus(y_hi):
        return H_MAX
    y = _solve_expm1_minus(ln_c, y_lo, y_hi)
    return float(min(H_MAX, max(0.0, y / (2.0 * k) - b0)))


def h_evidence(arl_ticks: float) -> float:
    """Evidence CUSUM threshold (B25): h = (ln ARL - 3.07) / 0.94, floored at 0."""
    arl = float(arl_ticks)
    if math.isnan(arl):
        return float("nan")
    if arl <= 1.0:
        return 0.0
    return max(0.0, (math.log(arl) - EVIDENCE_A) / EVIDENCE_B)


def h_llr(arl_ticks: float) -> float:
    """Threshold for a CUSUM of (calibrated) log-likelihood ratios whose null
    increments satisfy E0[exp(lambda)] <= 1 (identity CUSUMs, B16):
    Lorden / Wald bound h = ln(ARL). Conservative."""
    arl = float(arl_ticks)
    if math.isnan(arl):
        return float("nan")
    return math.log(arl) if arl > 1.0 else 0.0


def bernoulli_threshold(arl_slots: float) -> float:
    """Bernoulli CUSUM threshold in bits (B07): h = log2(ARL_slots) + 0.5.
    100 days of 15-min slots (9600) -> 13.73 bits. ARL < 1 counts as 1."""
    arl = float(arl_slots)
    if math.isnan(arl):
        return float("nan")
    return math.log2(max(arl, 1.0)) + 0.5


def h_for(kind: str, arl_days: float, dt_s: float, k: float = 0.5, d: int = 1) -> float:
    """Dispatcher used by every engine: threshold for a wall-clock ARL.

    kind: 'gauss' (h_gauss(k, .)), 'evidence' (h_evidence), 'llr' (h_llr),
    'bernoulli' (bernoulli_threshold, with dt_s the slot length 900),
    'mcusum' (mcusum_h(d, .)). The ARL is arl_ticks(arl_days, dt_s) in every case.
    Unknown kind -> ValueError.
    """
    arl = arl_ticks(arl_days, dt_s)
    if kind == "gauss":
        return h_gauss(k, arl)
    if kind == "evidence":
        return h_evidence(arl)
    if kind == "llr":
        return h_llr(arl)
    if kind == "bernoulli":
        return bernoulli_threshold(arl)
    if kind == "mcusum":
        return mcusum_h(d, arl)
    raise ValueError(f"seq.h_for: unknown kind {kind!r}")


# ------------------------------------------------------------------ steps
def cusum_step(S: np.ndarray, x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Vectorised one-sided CUSUM: S' = max(0, S + x - k). NaN x contributes 0
    (S unchanged, not reset). Lower-side charts pass -x.

    A NaN state entry (e.g. a stored cusum_state that round-tripped a null)
    restarts at 0, as the scalar steps do; otherwise max() would propagate it
    and the chart would stay NaN, silently dead, forever."""
    S = np.asarray(S, dtype=np.float64)
    S = np.where(np.isnan(S), 0.0, S)
    x = np.asarray(x, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    new = np.maximum(S + x - k, 0.0)
    return np.where(np.isnan(x), S, new)


def cusum_stationary_p(S: np.ndarray, k: float, n_charts: int = 1) -> np.ndarray:
    """Stationary tail of a Gaussian CUSUM as an equivalent p (B14, B24 small samples):
    p_eq = min(1, n_charts * exp(-2 k (S + 0.583))).

    Computed in log space and floored at P_FLOOR (1e-300) so -log10 p stays
    finite; NaN S -> NaN. k may also be an array broadcasting against S (a
    bank with per-chart k)."""
    S = np.asarray(S, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    log_p = math.log(max(float(n_charts), 1.0)) - 2.0 * k * (S + SIEGMUND_RHO)
    return np.clip(np.exp(np.minimum(log_p, 0.0)), P_FLOOR, 1.0)


EVIDENCE_CAP_MULT = 2.0     # B25 evidence CUSUMs are bounded at 2 h (round 4)


def evidence_cusum_step(S: float, q_inst: float, cap: float = math.inf) -> float:
    """B25: S' = min(cap, max(0, S - ln q_inst - 3)). NaN q -> S unchanged.
    q is clipped to [1e-300, 1] before the log. A NaN S restarts at 0.
    cap (round 4, B25 passes EVIDENCE_CAP_MULT x h): the first passage of h,
    hence every alarm onset under the null, is unchanged by any cap >= h;
    it bounds what an excursion keeps after the evidence stops (pack B:
    S = 415 / 2146 against h ~ 6 - 12 after T14 / T15 drained at the null
    drift -2 per tick for days, re-emitting an evidence alarm every tick)."""
    S = float(S)
    if math.isnan(S):               # corrupt stored state: restart, keep this tick
        S = 0.0
    q = float(q_inst)
    if math.isnan(q):
        return S
    q = min(1.0, max(P_FLOOR, q))
    return min(cap, max(0.0, S - math.log(q) - EVIDENCE_DRIFT))


def rhythm_p1(p0: float) -> float:
    """Off-hours alternative (B07): p1 = min(0.95, max(0.5, 5 p0)); only defined for
    p0 <= RHYTHM_P0_MAX (callers skip other bins; returns NaN there)."""
    p0 = float(p0)
    if math.isnan(p0) or p0 > RHYTHM_P0_MAX:
        return float("nan")
    lo, hi = RHYTHM_P1_BOUNDS
    return min(hi, max(lo, 5.0 * p0))


def bernoulli_cusum_step(W: float, a: int, p0: float, p1: float) -> float:
    """Bernoulli CUSUM in bits, once per slot (B07):
    W' = max(0, W + a log2(p1/p0) + (1 - a) log2((1 - p1)/(1 - p0))).
    p0, p1 clipped to [1e-6, 1 - 1e-6]. At p0 = 0.02 an active slot adds 4.64 bits.

    a is the slot activity in {0, 1} (clipped to [0, 1]). A NaN a, p0 or p1
    (e.g. rhythm_p1 of a bin with p0 > 0.3) contributes 0: W unchanged.
    A NaN W restarts at 0 (and this slot's increment still counts).
    """
    W = float(W)
    if math.isnan(W):               # corrupt stored state: restart, keep this slot
        W = 0.0
    a = float(a)
    p0 = float(p0)
    p1 = float(p1)
    if math.isnan(a) or math.isnan(p0) or math.isnan(p1):
        return W
    lo, hi = BERN_P_CLIP
    p0 = min(hi, max(lo, p0))
    p1 = min(hi, max(lo, p1))
    a = min(1.0, max(0.0, a))
    inc = a * math.log2(p1 / p0) + (1.0 - a) * math.log2((1.0 - p1) / (1.0 - p0))
    return max(0.0, W + inc)


# ------------------------------------------------------------ prewhitening
def ar1_phi(x: Sequence[float]) -> float:
    """Lag-1 autocorrelation of x over pairs where both x_t and x_{t-1} are finite,
    clipped to AR1_PHI_CLIP. Fewer than 10 pairs or zero variance -> 0.0.

    Pearson correlation of the (x_t, x_{t-1}) pairs, each side centred on its
    own mean, so NaN gaps drop pairs instead of biasing the estimate.
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < AR1_MIN_PAIRS + 1:
        return 0.0
    cur, prev = v[1:], v[:-1]
    ok = np.isfinite(cur) & np.isfinite(prev)
    if int(ok.sum()) < AR1_MIN_PAIRS:
        return 0.0
    cur, prev = cur[ok], prev[ok]
    # exact constancy test: a mean of equal floats is not always equal to
    # them, so a variance test would see ~1e-34 "variance" and return r = 1
    if np.ptp(cur) == 0.0 or np.ptp(prev) == 0.0:
        return 0.0
    dc = cur - cur.mean()
    dp = prev - prev.mean()
    denom = math.sqrt(float(np.dot(dc, dc)) * float(np.dot(dp, dp)))
    if not denom > 0.0:
        return 0.0
    r = float(np.dot(dc, dp)) / denom
    if not math.isfinite(r):
        return 0.0
    return float(min(AR1_PHI_CLIP[1], max(AR1_PHI_CLIP[0], r)))


def prewhiten(x_t: np.ndarray, x_prev: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """u = (x_t - phi x_prev) / sqrt(1 - phi^2), elementwise.

    x_t NaN -> NaN (the chart ignores it); x_prev NaN -> u = x_t (no
    whitening, stationary variance 1 assumed). Callers clip u to +-PSI_CLIP.
    phi is clipped to AR1_PHI_CLIP (what ar1_phi returns; keeps 1 - phi^2
    away from 0) and a NaN phi means no whitening (phi = 0).
    """
    x_t = np.asarray(x_t, dtype=np.float64)
    x_prev = np.asarray(x_prev, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    phi = np.clip(np.where(np.isnan(phi), 0.0, phi), AR1_PHI_CLIP[0], AR1_PHI_CLIP[1])
    u = (x_t - phi * x_prev) / np.sqrt(1.0 - phi * phi)
    return np.where(np.isnan(x_prev), x_t, u)


# ----------------------------------------------------------------- MCUSUM
def mcusum_step(S: np.ndarray, x: np.ndarray, k: float = MCUSUM_K) -> Tuple[np.ndarray, float]:
    """Crosier MCUSUM on a whitened d-vector x (NaN entries -> 0):

        C = ||S + x||;  S' = 0 if C <= k else (S + x)(1 - k / C);  stat = ||S'||
    Returns (S', stat). O(d).

    Works on the last axis, so a batch S, x of shape [..., d] returns
    (S' [..., d], stat [...] ndarray); the 1-D case returns a float stat.
    ||S'|| = C - k exactly when C > k. S is never modified in place.

    A state vector with any NaN entry restarts at 0 (the whole vector: a
    partly corrupt direction is meaningless). Without this C is NaN, the chart
    takes the C <= k branch every tick and reports stat = 0 forever.
    """
    S = np.asarray(S, dtype=np.float64)
    S = np.where(np.isnan(S).any(axis=-1, keepdims=True), 0.0, S)
    x = np.asarray(x, dtype=np.float64)
    Y = S + np.where(np.isnan(x), 0.0, x)
    C = np.sqrt(np.sum(Y * Y, axis=-1))
    k = float(k)
    live = C > k
    fac = np.where(live, 1.0 - k / np.where(live, C, 1.0), 0.0)
    S_new = Y * fac[..., None]
    stat = np.where(live, C - k, 0.0)
    if np.ndim(stat) == 0:
        return S_new, float(stat)
    return S_new, stat


def mcusum_h(d: int, arl_ticks: float) -> float:
    """Threshold from MCUSUM_H: row d (1..16; d > 16 uses 16), interpolate
    linearly in ln ARL over MCUSUM_ARL_GRID, extrapolate linearly beyond.

    d < 1 uses row 1; ARL <= 1 -> 0; NaN ARL -> NaN; the result is floored at 0.
    """
    arl = float(arl_ticks)
    if math.isnan(arl):
        return float("nan")
    if arl <= 1.0:
        return 0.0
    row = MCUSUM_H[min(max(int(d), 1), MCUSUM_H.shape[0]) - 1]
    if math.isinf(arl):
        return float("inf")
    lx = math.log(arl)
    g = _LN_MCUSUM_GRID
    j = int(np.searchsorted(g, lx, side="right")) - 1
    j = min(max(j, 0), g.size - 2)
    slope = (row[j + 1] - row[j]) / (g[j + 1] - g[j])
    return max(0.0, float(row[j] + slope * (lx - g[j])))


def _mcusum_arl_mc(d: int, h_grid: Sequence[float], n_chains: int = 2000,
                   t_max: int = 120_000, seed: int = 0,
                   k: float = MCUSUM_K) -> Tuple[np.ndarray, np.ndarray]:
    """Offline generator of MCUSUM_H (not used at run time): zero-state ARL of
    Crosier's chart on N(0, I_d) for every h in the ascending h_grid at once.

    Each chain's running maximum M_t gives the first passage over h_j as the
    first t with stat_t > h_j, so one set of paths serves the whole grid.
    Returns (arl[h], n_crossed[h]); arl uses the censored-exponential MLE
    sum(min(T, t_max)) / #crossed. The committed table came from this
    procedure with n_chains = 2000, t_max = 1.2e5, seed = 1000 + d and
    h_grid = arange(0.25, 12 + 1.6 d, 0.05), then the fit described above.
    """
    rng = np.random.default_rng(seed)
    hg = np.asarray(h_grid, dtype=np.float64)
    S = np.zeros((n_chains, d))
    M = np.zeros(n_chains)
    first = np.full((n_chains, hg.size), -1, dtype=np.int64)
    t = 0
    while t < t_max and not (first[:, -1] >= 0).all():
        X = rng.standard_normal((min(128, t_max - t), n_chains, d), dtype=np.float32)
        for xb in X:
            t += 1
            S, stat = mcusum_step(S, xb, k)
            up = np.nonzero(stat > M)[0]
            if up.size:
                newly = (hg[None, :] >= M[up, None]) & (hg[None, :] < stat[up, None])
                sub = first[up]
                sub[newly] = t
                first[up] = sub
                M[up] = stat[up]
    crossed = first >= 0
    n = crossed.sum(axis=0)
    arl = np.where(crossed, first, t).sum(axis=0) / np.maximum(n, 1)
    return arl, n


# ------------------------------------------------------------------ trend
def mann_kendall(x: Sequence[float]) -> Tuple[float, float]:
    """(S statistic, two-sided p) of the Mann-Kendall trend test on finite x,
    normal approximation with tie-corrected variance; n < 4 -> (0, 1). O(n^2) (n <= 30).

    Var S = [n(n-1)(2n+5) - sum_t t(t-1)(2t+5)] / 18 over tie groups of size t,
    z = (S - sign S) / sqrt(Var S) (continuity correction), p = erfc(|z|/sqrt 2).
    All-tied input (Var S = 0) -> (0, 1).
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    n = v.size
    if n < 4:
        return 0.0, 1.0
    diff = v[None, :] - v[:, None]                     # diff[i, j] = x_j - x_i
    s = float(np.sign(diff[np.triu_indices(n, 1)]).sum())
    _, t = np.unique(v, return_counts=True)
    t = t[t > 1].astype(np.float64)
    var = (n * (n - 1.0) * (2.0 * n + 5.0) - float(np.sum(t * (t - 1.0) * (2.0 * t + 5.0)))) / 18.0
    if var <= 0.0:
        return 0.0, 1.0
    z = (s - math.copysign(1.0, s)) / math.sqrt(var) if s != 0.0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return s, float(min(1.0, max(P_FLOOR, p)))


def sen_slope(x: Sequence[float], t: Optional[Sequence[float]] = None) -> float:
    """Theil-Sen slope: median of (x_j - x_i)/(t_j - t_i) over i < j (t defaults to 0..n-1).

    Points with non-finite x or t are dropped and pairs with t_j == t_i are
    skipped; no usable pair -> NaN. len(t) != len(x) -> ValueError.
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    tt = (np.arange(v.size, dtype=np.float64) if t is None
          else np.asarray(t, dtype=np.float64).ravel())
    if tt.size != v.size:
        raise ValueError("seq.sen_slope: x and t must have the same length")
    ok = np.isfinite(v) & np.isfinite(tt)
    v, tt = v[ok], tt[ok]
    if v.size < 2:
        return float("nan")
    i, j = np.triu_indices(v.size, 1)
    dt = tt[j] - tt[i]
    keep = dt != 0.0
    if not keep.any():
        return float("nan")
    return float(np.median((v[j] - v[i])[keep] / dt[keep]))
