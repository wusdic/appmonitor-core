"""Robust covariance, Hotelling T2 / SPE and exact contributions (B06, B15, B19).

STATUS: implemented. Signatures are frozen (docs/lib3/helpers_api.md); where
the maths departs from the original stub it is marked DEVIATION below and
listed in helpers_api.md.

Why: the old per-run IsolationForest trained on attack rows and cost ~1.1 s.
A shrunk (OAS), C-step-robustified covariance fitted on gated rows gives
calibrated T2 (magnitude) and SPE (correlation break) p-values in
microseconds, exact reconstruction-based contributions for explanations,
and principled handling of missing dimensions (conditional expectation).

All inputs are float64 arrays of standardized residuals (behavior.zi);
callers drop or impute NaN before fitting. p = number of dimensions.

Why OAS rather than the sample covariance: B06 refits on <= 336 rows of up to
52 dimensions, and B15 on pooled windows of 48; at n / p ~ 6 the sample
covariance's small eigenvalues are biased towards 0, which inflates T2 and
fakes correlation breaks. OAS shrinks towards mu_t I with a closed-form rho
(no cross-validation), O(n p^2). Caveat (measured): rho is driven by the
whole spectrum, so ONE strong pair among many independent dimensions is
shrunk hard -- a rho = 0.95 pair fitted at n = 336 keeps a regression slope
(what conditional_impute uses) of 0.94 at p = 2, 0.90 at p = 5, 0.54 at
p = 20 and 0.16 at p = 52 (sample covariance: 0.95 throughout).
Real zi have many correlated features (tr(S^2) >> p) and shrink far less.

Why a C-step fit and how it departs from the stub (DEVIATIONS, measured on
30 fits per case, n = 336, 20 % contamination, KL(fit || truth)):
  * Start. The stub started from the OAS of all winsorised rows. A tight
    20 % cluster (all dims at +6, clipped to +4) masks itself there: it
    inflates the start covariance along its own direction, so its d2 look
    typical and the h-subset keeps it (p = 5: KL 1.17 vs 0.004 clean,
    |mu| 1.95). The start is now the spatial-sign start of DetMCD
    (Hubert, Rousseeuw & Verdonck 2012), whose rows each have bounded pull,
    refined by an OAS fit of its ceil(n/2) innermost rows (DetMCD's h0
    step): cluster KL 0.003, scattered 5-sd outliers 0.004, 3 dims shifted
    by 8 sd 0.005, a (+3, -3) pair break 0.025 (max 0.30). Keeping the OAS
    start as a second candidate and choosing by determinant (the MCD
    objective) does worse: the winsorised cluster is a point mass whose
    zero spread wins the determinant at p = 10 (KL 0.9).
  * Up to 3 C-steps (stopping when the h-subset repeats) instead of one;
    the extra steps shed the last few outliers of a moderate pair break.
  * Consistency of the reweighted fit. The stub stopped at 'oas again', but
    rows kept at d2 <= chi2_{p,0.975} are a truncated sample, so the fit is
    biased low (by chi2_{p+2}.cdf(q) / 0.975 = 0.905 at p = 2 for an
    unshrunk normal). The final Sigma is rescaled so the median d2 of its
    inliers is chi2_{p,0.4875}, the median of the truncated chi2_p. This
    data-driven factor also absorbs the scale that OAS shrinkage moves into
    small eigenvalues, which the fixed factor does not: Hotelling KS D over
    2000 fresh n = 200 fits is 0.016-0.039 for p <= 10 (identity, rho = 0.95
    pair, equicorrelated, random correlation); the stub's pipeline gave up
    to 0.062 uncorrected and up to 0.092 with the fixed factor. At q = 20
    OAS makes T2 tighter than the F prediction assumes: D = 0.11 but
    conservative (0.2 % at the 1 % level) for Sigma = I, D = 0.02 for
    random correlation. Price: OAS lifts small eigenvalues and this
    rescale spreads the error, so a strongly anisotropic Sigma comes out
    ~10 % low along its major axis (rho = 0.95 pair, p = 2, n = 336), i.e.
    T2 of a move along the correlation reads ~10 % high.
  * The h-subset scale uses the stub's median(d2) / chi2_{p,0.5} rule with d2
    recomputed under the h-subset fit (as in MCD).
  * Out-of-sample d2 for the h-subset rows (review fix). Under the h-subset
    fit, its own rows have in-sample d2 while the other rows are
    out-of-sample, inflated by ~(m+1)(m-1)/(m(m-p-2)) = 1.27 at m = 252,
    p = 52. Mixing the two in the median rescale and the chi2_{p,0.975}
    cut flagged ~21 % of CLEAN rows at 336 x 52 (factor-correlated), and
    the final rescale then over-tightened Sigma: Hotelling false alarms of
    2.0-3.1 % at the 1 % level and 3.5x at 1e-3 (8 seeds). The subset rows
    now use closed-form leave-one-out d2 (_loo_d2, Sherman-Morrison, one
    extra factorisation): 4-5 % of clean rows flagged, 0.7-0.9 % false
    alarms at 1 %, <= 1.3x at 1e-3; KS D 0.014-0.039 over 2000 n = 200 fits
    for p <= 20 (identity, pair, equicorrelated, random correlation; the
    previous pipeline gave up to 0.057). Outlier resistance is unchanged
    (20 % cluster / scatter / shifted dims / pair break at p = 5, 20, 52).
Known limitation (measured, not fixable here): TIES. The MCD criterion
exploits point masses: when a column holds one exact value in a large share
of rows (a zero-inflated feature, or mid-PIT normal scores of a low-mean
count), the h-subset concentrates on the tie and that column's variance
implodes. 3 of 5 columns 60 % zeros: 23 % false alarms at the 1 % level
(the plain OAS of the winsorised rows: 4 %); standardised Poisson(0.5):
14 %. Feed continuous scores (randomised PIT) to the robust fit.
Cost (one thread): ~2.8 ms at 336 x 52 and ~6.6 ms at 600 x 64, of which
the LOO step is ~0.2 ms (the stub's single C-step: ~1.2 / ~2.7 ms), i.e.
~0.2 ms per tick at B06's 16-tick refit stride. Scoring: T2 ~6 us warm (~25 us for a new missingness
pattern), imputation ~40 us, RBC ~30 us, SPE ~5 us, p-values ~2 us.

Why the prediction (F-scaled) Hotelling p rather than chi2_q: the fitted mean
and covariance carry estimation error; for a new point the exact null is
T2 n (n - q) / (q (n - 1) (n + 1)) ~ F(q, n - q). At q = 20, n = 200 the
chi2_q p-values are visibly anti-conservative (tested), and B06 scores with
q up to 52 on n <= 336.

Why conditional imputation: replacing a missing z by 0 claims the dimension
sat exactly at its mean, which hides a correlation break in the observed
dimensions and understates SPE. z_m = Sigma_mo Sigma_oo^-1 z_o is the point
on the fitted normal closest to the observed part (minimum Mahalanobis
completion), so the completed vector has the same T2 as z_o on Sigma_oo and
the missing coordinates contribute nothing extra.

Why RBC: reconstruction-based contributions have a known null (chi2_1 per
feature) and do not smear a single fault onto correlated neighbours the way
plain T2 contributions do (Alcala & Qin 2009). With missing entries RBC is
evaluated on the observed sub-model (precision of Sigma_oo, obtained from
Sinv by a Schur complement), so each observed term is still chi2_1.

Numerics: every p is clipped to [1e-300, 1]; NaN in -> NaN out, never 0 or 1.
Tail probabilities are computed on their own side (fdtrc, chdtrc, erfc).
Chi-square quantiles use scipy.special (chdtri), ~100x cheaper than
scipy.stats. Cholesky factors fall back to an eigenvalue floor when a
sub-matrix is numerically indefinite.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy import special
from scipy.linalg.lapack import dpotrf, dpotrs, dtrtri, dtrtrs

WINSOR = 4.0
C_STEP_H = 0.75
EIG_FLOOR_REL = 1e-3
PCA_VAR_FRAC = 0.90
CHOL_CACHE_SIZE = 8

_P_FLOOR = 1e-300
_REWEIGHT_Q = 0.975        # reweighting cut-off quantile of chi2_p
_CUM_TOL = 1e-12           # tolerance on the cumulative variance fraction
_MAD_K = 1.4826            # MAD -> sd at the normal
_H0 = 0.5                  # half-subset that refines the spatial-sign start
_C_STEPS = 3               # max C-steps (stops early once the h-subset repeats)


# ------------------------------------------------------------------ helpers
def _clip_p(p: float) -> float:
    if p != p:
        return math.nan
    return 1.0 if p > 1.0 else (_P_FLOOR if p < _P_FLOOR else p)


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D [n, p] array, got shape {X.shape}")
    return X


def _clean_rows(X: np.ndarray, w: Optional[np.ndarray]
                ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Drop rows with any non-finite value, and rows whose weight is not
    finite and > 0. Returns (rows, weights or None)."""
    keep = np.all(np.isfinite(X), axis=1)
    if w is None:
        return (X if keep.all() else X[keep]), None
    w = np.asarray(w, dtype=np.float64).ravel()
    if w.size != X.shape[0]:
        raise ValueError(f"len(w)={w.size} != n rows={X.shape[0]}")
    keep &= np.isfinite(w) & (w > 0.0)
    return X[keep], w[keep]


def _sym(S: np.ndarray) -> np.ndarray:
    return 0.5 * (S + S.T)


def _chol(A: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor (LAPACK dpotrf: ~12 us at p = 52 vs ~18 us for
    np.linalg.cholesky); an indefinite (rounding) matrix is floored first."""
    A = np.asarray(A, dtype=np.float64)
    if A.size == 0:
        return np.zeros((0, 0))
    L, info = dpotrf(A, lower=1, clean=1)
    if info != 0:
        L, info = dpotrf(eigen_floor(A), lower=1, clean=1)
        if info != 0:
            raise np.linalg.LinAlgError("robustcov: matrix is not positive definite")
    return L


def _tri(L: np.ndarray, b: np.ndarray, trans: int = 0) -> np.ndarray:
    """Solve L x = b (trans = 0) or L^T x = b (trans = 1) for lower-triangular L.
    Raw LAPACK dtrtrs: ~3 us vs ~9 us for scipy.linalg.solve_triangular."""
    x, _ = dtrtrs(L, b, lower=1, trans=trans)
    return x


def _sub(A: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """A[rows][:, cols] via take (~3.5 us vs ~11 us for np.ix_ at p = 52)."""
    return A.take(rows, axis=0).take(cols, axis=1)


def _mahal2(Xc: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distances of centred rows Xc [n, p]. Inverting the
    p x p factor once and using one GEMM is ~2x faster than a triangular
    solve with n right-hand sides."""
    Li, _ = dtrtri(_chol(_sym(Sigma)), lower=1)
    Y = Xc @ Li.T
    return np.einsum("ij,ij->i", Y, Y)


def _median0(A: np.ndarray) -> np.ndarray:
    """Median along axis 0 of finite data via one partition (np.median's
    generic path costs ~15 us more per call, which dominates at small p)."""
    n = A.shape[0]
    k = n // 2
    P = np.partition(A, k, axis=0)
    if n % 2:
        return P[k]
    # Even n: the lower middle is the max of the part below k (a second
    # kth in np.partition costs ~3x more than this).
    return 0.5 * (P[k] + P[:k].max(axis=0))


def _wmedian(x: np.ndarray, w: Optional[np.ndarray]) -> float:
    """Weighted median; equal weights give exactly np.median (so a constant
    trust weight reproduces the unweighted fit), and a cumulative weight that
    lands on one half averages the two neighbours, as np.median does."""
    if w is None or np.all(w == w[0]):
        return float(_median0(x))
    order = np.argsort(x, kind="stable")
    xs = x[order]
    cw = np.cumsum(w[order])
    half = 0.5 * cw[-1]
    tol = 1e-12 * cw[-1]
    i = min(int(np.searchsorted(cw, half, side="left")), x.size - 1)
    # A cumulative weight within rounding of one half may sit on either side
    # of it (cumsum of 0.1s, say): check the previous one as well, or the
    # median jumps a whole order statistic on a 1-ulp change of a weight.
    if i > 0 and abs(cw[i - 1] - half) <= tol:
        return 0.5 * float(xs[i - 1] + xs[i])
    if i + 1 < x.size and abs(cw[i] - half) <= tol:
        return 0.5 * float(xs[i] + xs[i + 1])
    return float(xs[i])


def _chi2_ppf(q: float, df: float) -> float:
    return float(special.chdtri(df, 1.0 - q))


def _robust_loc_scale(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Column-wise (median, 1.4826 MAD); a zero MAD falls back to the sd, then 1."""
    med = _median0(A)
    s = _median0(np.abs(A - med)) * _MAD_K
    bad = ~(s > 0.0)
    if bad.any():
        sd = A.std(axis=0)
        s = np.where(bad, np.where(sd > 0.0, sd, 1.0), s)
    return med, s


def _sscm_d2(X: np.ndarray) -> np.ndarray:
    """d2 of rows under the spatial-sign start of DetMCD (Hubert, Rousseeuw &
    Verdonck 2012): robustly standardise, take the eigenvectors of the
    covariance of the unit-norm rows (each row's pull is bounded), and give
    each eigen-direction its robust (MAD) scale."""
    med, s = _robust_loc_scale(X)
    Z = (X - med) / s
    nrm = np.sqrt(np.einsum("ij,ij->i", Z, Z))
    K = Z / np.where(nrm > 0.0, nrm, 1.0)[:, None]
    _, E = np.linalg.eigh(K.T @ K)
    B = Z @ E
    cb, sb = _robust_loc_scale(B)
    Y = (B - cb) / sb
    return np.einsum("ij,ij->i", Y, Y)


def _loo_d2(Xs: np.ndarray, ws: Optional[np.ndarray], mu: np.ndarray,
            Sigma: np.ndarray, rho: float) -> np.ndarray:
    """Leave-one-out d2 of the rows an OAS fit (mu, Sigma, rho) was fitted on.

    Holding rho and mu_t = tr(Sigma) / p at their full-subset values, dropping
    row i (normalised weight w_i, xc = x_i - mu) gives
      x_i - mu_(-i) = xc / (1 - w_i),
      Sigma_(-i)    = B - a_i xc xc^T,
      B = (Sigma - rho mu_t I) / (1 - w) + rho mu_t I,  a_i = (1 - rho) w_i / (1 - w)^2,
    so by Sherman-Morrison d2_(-i) = g_i / ((1 - w_i)^2 (1 - a_i g_i)) with
    g_i = xc^T B^-1 xc: one extra factorisation for all rows. Exact for
    equal weights (w = w_i = 1/m); with unequal weights B uses the mean
    normalised weight w = 1 / n_eff (0.3 % error at weights in [0.5, 2]).
    A row that alone spans a direction (1 - a_i g_i <= 0) gets +inf.
    """
    m, p = Xs.shape
    if ws is None or np.all(ws == ws[0]):
        wn = np.full(m, 1.0 / m)
        wbar = 1.0 / m
    else:
        wn = ws / ws.max()
        wn = wn / wn.sum()
        wbar = float(np.dot(wn, wn))
    mut = float(np.trace(Sigma)) / p
    B = Sigma / (1.0 - wbar)
    B.flat[:: p + 1] += rho * mut * (1.0 - 1.0 / (1.0 - wbar))
    g = _mahal2(Xs - mu, B)
    den = 1.0 - (1.0 - rho) * wn / (1.0 - wbar) ** 2 * g
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0.0, g / ((1.0 - wn) ** 2 * den), np.inf)


def _inlier_median_q(p: int, q: float = _REWEIGHT_Q) -> float:
    """Median of chi2_p truncated at its q quantile: chi2_p.ppf(q / 2)."""
    return _chi2_ppf(0.5 * q, p)


# ------------------------------------------------------------------ fitting
def winsorize(X: np.ndarray, c: float = WINSOR) -> np.ndarray:
    """Clip to [-c, c] (NaN preserved; +-inf become +-c)."""
    return np.clip(np.asarray(X, dtype=np.float64), -c, c)


def oas(X: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """(mu, Sigma, rho) Oracle Approximating Shrinkage (Chen et al. 2010).

    S = weighted MLE covariance (weights normalised, n_eff = (sum w)^2 / sum w^2),
    mu_t = tr(S)/p, rho = min(1, ((1 - 2/p) tr(S^2) + tr(S)^2) /
    ((n_eff + 1 - 2/p) (tr(S^2) - tr(S)^2 / p))), Sigma = (1 - rho) S + rho mu_t I.
    Rows with any NaN are dropped. n < 2 -> (mean, I, 1.0). O(n p^2).

    Also dropped: rows with +-inf, and rows whose weight is not finite and > 0.
    With no usable row the mean is zeros(p) (the null mean of a standardized
    residual). A non-positive denominator (S proportional to I, or p = 1,
    where shrinkage is a no-op) gives rho = 1.
    """
    Xk, wk = _clean_rows(_as_2d(X), w)
    return _oas_rows(Xk, wk)


def _oas_rows(Xk: np.ndarray, wk: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, float]:
    """oas on rows already cleaned by _clean_rows (the C-step's inner loop)."""
    n, p = Xk.shape
    if n < 2 or p == 0:
        mu = Xk[0].copy() if n == 1 else np.zeros(p)
        return mu, np.eye(p), 1.0
    if wk is None or np.all(wk == wk[0]):
        n_eff = float(n)
        mu = Xk.mean(axis=0)
        Xc = Xk - mu
        S = _sym(Xc.T @ Xc) / n
    else:
        wn = wk / wk.max()              # guard sum(w) against overflow
        wn = wn / wn.sum()
        n_eff = 1.0 / float(np.dot(wn, wn))
        mu = wn @ Xk
        Xc = Xk - mu
        S = _sym((Xc * wn[:, None]).T @ Xc)
    tr_s = float(np.trace(S))
    tr_s2 = float(np.sum(S * S))        # tr(S^2) for symmetric S
    num = (1.0 - 2.0 / p) * tr_s2 + tr_s * tr_s
    den = (n_eff + 1.0 - 2.0 / p) * (tr_s2 - tr_s * tr_s / p)
    rho = min(1.0, num / den) if den > 0.0 else 1.0
    rho = max(0.0, rho)
    Sigma = (1.0 - rho) * S
    Sigma.flat[:: p + 1] += rho * tr_s / p
    return mu, Sigma, float(rho)


def c_step_oas(X: np.ndarray, w: Optional[np.ndarray] = None, h: float = C_STEP_H
               ) -> Tuple[np.ndarray, np.ndarray]:
    """(mu, Sigma) robust fit (B06 refit):
    winsorize(X); start d2; keep the h fraction with the smallest d2 -> oas
    (C-step); rescale Sigma by median(d2) / chi2.ppf(0.5, p); keep rows with
    d2 <= chi2.ppf(0.975, p) -> oas again; consistency rescale; eigen_floor.

    DEVIATIONS from the stub (measured, module docstring):
      * start: the DetMCD spatial-sign d2 refined by an OAS fit of its
        ceil(n/2) innermost rows, instead of the OAS of all winsorised rows;
      * up to _C_STEPS = 3 C-steps, stopping once the h-subset repeats;
      * the reweighted Sigma is rescaled so the median d2 of its inliers is
        chi2.ppf(0.4875, p) (the median of chi2_p truncated at 0.975);
      * the h-subset rows' d2 are leave-one-out (closed form), so every d2
        in the median rescale and the reweighting is out-of-sample.

    Details: rows with any non-finite value are dropped (after winsorising,
    so +-inf count as +-c). The h-subset has ceil(h n) >= 2 rows (by count;
    weights enter every OAS and both medians, not the start). The median
    rescale uses the d2 of all rows under the h-subset fit (LOO for the
    subset's own rows), and the
    reweighting uses d2 under the rescaled fit. Fewer than 2 usable rows ->
    oas's (mean, I). Fewer than 2 reweighted inliers -> the rescaled h-subset
    fit. O(n p^2 + p^3), ~2.8 ms at 336 x 52 (one thread).
    """
    X = winsorize(_as_2d(X))
    p = X.shape[1]
    Xk, wk = _clean_rows(X, w)
    n = Xk.shape[0]
    if n < 2 or p == 0:
        mu0, S0, _ = _oas_rows(Xk, wk)
        return mu0, eigen_floor(S0)

    def fit(rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        return _oas_rows(Xk[rows], None if wk is None else wk[rows])

    # Start: spatial-sign d2, refined on its innermost half (DetMCD).
    d2 = _sscm_d2(Xk)
    m0 = max(2, int(math.ceil(_H0 * n)))
    if m0 < n:
        mu1, S1, _ = fit(np.argpartition(d2, m0 - 1)[:m0])
        d2 = _mahal2(Xk - mu1, S1)
    # C-steps on the h-subset.
    m = min(n, max(2, int(math.ceil(h * n))))
    prev = None
    rho1 = 1.0
    for _ in range(_C_STEPS):
        sub = np.sort(np.argpartition(d2, m - 1)[:m]) if m < n else np.arange(n)
        if prev is not None and np.array_equal(sub, prev):
            break
        mu1, S1, rho1 = fit(sub)
        d2 = _mahal2(Xk - mu1, S1)
        prev = sub
    # Put the h-subset rows on the same (out-of-sample) footing as the rest:
    # their in-sample d2 are shrunk by the fit's own overfit while the other
    # rows' are not, which at n / p ~ 6 pushed ~20 % of clean rows past the
    # reweighting cut-off (DEVIATION, module docstring).
    if m < n:
        d2 = d2.copy()
        d2[prev] = _loo_d2(Xk[prev], None if wk is None else wk[prev], mu1, S1, rho1)
    # Consistency rescale of the h-subset scatter (MCD-style).
    scale = _wmedian(d2, wk) / _chi2_ppf(0.5, p)
    if math.isfinite(scale) and scale > 0.0:
        S1 = S1 * scale
        d2 = d2 / scale
    # Reweight at chi2_{p, 0.975}.
    inl = d2 <= _chi2_ppf(_REWEIGHT_Q, p)
    if int(inl.sum()) < 2:
        return mu1, eigen_floor(S1)
    Xi = Xk[inl]
    wi = None if wk is None else wk[inl]
    mu2, S2, _ = _oas_rows(Xi, wi)
    # The inliers are a sample truncated at chi2_{p,0.975}: match their
    # median d2 to chi2_{p,0.4875}.
    scale = _wmedian(_mahal2(Xi - mu2, S2), wi) / _inlier_median_q(p)
    if math.isfinite(scale) and scale > 0.0:
        S2 = S2 * scale
    return mu2, eigen_floor(S2)


def eigen_floor(Sigma: np.ndarray, rel: float = EIG_FLOOR_REL) -> np.ndarray:
    """Symmetrise and floor eigenvalues at rel * mean(eigenvalues) (eigh).

    A matrix already above the floor is returned symmetrised but otherwise
    untouched (no eigen round-off). A degenerate matrix whose mean eigenvalue
    is not > 0 is floored at rel (unit scale of a standardized residual).
    """
    S = _sym(np.asarray(Sigma, dtype=np.float64))
    if S.size == 0:
        return S.copy()
    # Eigenvalues alone first (~2x cheaper): the OAS fits are usually above
    # the floor already, and then no eigenvectors are needed.
    lam = np.linalg.eigvalsh(S)
    mean = float(lam.mean())
    floor = rel * mean if (math.isfinite(mean) and mean > 0.0) else rel
    if lam[0] >= floor:
        return S
    lam, V = np.linalg.eigh(S)
    lam = np.maximum(lam, floor)
    return _sym((V * lam) @ V.T)


def pca_k(Sigma: np.ndarray, var_frac: float = PCA_VAR_FRAC) -> Tuple[np.ndarray, np.ndarray, int]:
    """(U_k [p, k], lam_k [k], k): smallest k whose eigenvalues explain var_frac.

    Eigenvalues descend; each column's largest-|.| entry is made positive so
    the basis is deterministic. 1 <= k <= p; a zero-trace Sigma gives k = p.
    """
    S = _sym(np.asarray(Sigma, dtype=np.float64))
    p = S.shape[0]
    if p == 0:
        return np.zeros((0, 0)), np.zeros(0), 0
    lam, V = np.linalg.eigh(S)
    lam, V = lam[::-1], V[:, ::-1]
    pos = np.clip(lam, 0.0, None)
    total = float(pos.sum())
    if total > 0.0 and math.isfinite(total):
        cum = np.cumsum(pos) / total
        k = int(np.searchsorted(cum, var_frac - _CUM_TOL, side="left")) + 1
        k = min(max(k, 1), p)
    else:
        k = p
    U = V[:, :k].copy()
    piv = np.argmax(np.abs(U), axis=0)
    sgn = np.sign(U[piv, np.arange(k)])
    sgn[sgn == 0] = 1.0
    return U * sgn, lam[:k].copy(), k


# ------------------------------------------------------------------ scoring
def hotelling_pred_p(t2: float, n: float, q: int) -> float:
    """Prediction-interval Hotelling p: F = t2 * n (n - q) / (q (n - 1) (n + 1)),
    p = f.sf(F, q, n - q) (scipy.special.fdtrc). n <= q + 1 or q < 1 -> NaN.

    n may be fractional (n_eff of a weighted fit). NaN t2 -> NaN; t2 <= 0 -> 1.
    """
    t2 = float(t2)
    n = float(n)
    if t2 != t2 or not (q >= 1) or not math.isfinite(n) or n <= q + 1:
        return math.nan
    if t2 <= 0.0:
        return 1.0
    F = t2 * n * (n - q) / (q * (n - 1.0) * (n + 1.0))
    return _clip_p(float(special.fdtrc(q, n - q, F)))


def wilson_hilferty(t2: float, q: int) -> float:
    """Normal score of a chi2_q value: ((t2/q)^(1/3) - (1 - 2/(9q))) / sqrt(2/(9q)).

    NaN t2 or q < 1 -> NaN; t2 < 0 (round-off) counts as 0.
    """
    t2 = float(t2)
    if t2 != t2 or not (q >= 1):
        return math.nan
    v = 2.0 / (9.0 * q)
    return ((max(t2, 0.0) / q) ** (1.0 / 3.0) - (1.0 - v)) / math.sqrt(v)


def spe(z: np.ndarray, U_k: np.ndarray) -> float:
    """Squared prediction error ||z - U_k U_k^T z||^2 of a completed vector.

    The residual is formed explicitly (not ||z||^2 - ||U_k^T z||^2) so a
    small SPE does not drown in cancellation. Any NaN in z -> NaN.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    U = np.asarray(U_k, dtype=np.float64)
    if U.ndim == 2 and U.shape[1] >= z.size:
        # k = p (pca_k does this for a near-isotropic Sigma at p < 10): there
        # is no residual subspace, and the explicit residual would be ~1e-31
        # round-off that spe_box_params then fits and spe_p turns into
        # spurious p < 0.01 on ~1 % of null points. SPE is exactly 0, which
        # makes the Box parameters (and so spe_p) NaN: no SPE evidence.
        return 0.0 if bool(np.all(np.isfinite(z))) else math.nan
    r = z - U @ (U.T @ z)
    return float(r @ r)


def spe_box_params(spe_train: np.ndarray) -> Tuple[float, float]:
    """Box approximation SPE ~ g chi2_h: g = var / (2 mean), h = 2 mean^2 / var
    over finite training SPE; < 10 values or var == 0 -> (NaN, NaN).

    var is the unbiased (ddof = 1) sample variance.
    """
    x = np.asarray(spe_train, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 10:
        return math.nan, math.nan
    m = float(x.mean())
    v = float(x.var(ddof=1))
    if not (v > 0.0 and m > 0.0):
        return math.nan, math.nan
    return v / (2.0 * m), 2.0 * m * m / v


def spe_p(spe_value: float, g: float, h: float) -> float:
    """p = chi2.sf(spe / g, h); NaN params -> NaN."""
    s, g, h = float(spe_value), float(g), float(h)
    if s != s or not (g > 0.0) or not (h > 0.0) or not math.isfinite(g * h):
        return math.nan
    return _clip_p(float(special.chdtrc(h, max(s, 0.0) / g)))


def rbc_contributions(z: np.ndarray, Sinv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruction-based contributions (Alcala & Qin 2009):
    RBC_f = (Sinv z)_f^2 / Sinv_ff, each ~ chi2_1 under the null;
    returns (rbc[p], p_values[p] = chi2.sf(rbc, 1)). NaN z_f -> NaN.

    With missing entries (non-finite z_f) the observed terms use the
    precision of the observed sub-model, P_oo = inv(Sigma_oo) =
    Sinv_oo - Sinv_om Sinv_mm^-1 Sinv_mo (Schur complement):
    RBC_f = (P_oo z_o)_f^2 / (P_oo)_ff, so each stays chi2_1. The numerators
    equal (Sinv z_c)_o for the conditionally imputed z_c (whose
    (Sinv z_c)_m = 0); the full-model denominators Sinv_ff would not give
    chi2_1 terms.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    P = _sym(np.asarray(Sinv, dtype=np.float64))
    p = z.size
    if P.shape != (p, p):
        raise ValueError(f"Sinv shape {P.shape} does not match len(z)={p}")
    rbc = np.full(p, np.nan)
    obs = np.isfinite(z)
    if obs.any():
        o = np.flatnonzero(obs)
        if obs.all():
            P_oo = P
        else:
            m = np.flatnonzero(~obs)
            B = _tri(_chol(_sub(P, m, m)), _sub(P, m, o))
            P_oo = _sym(_sub(P, o, o) - B.T @ B)
        u = P_oo @ z[o]
        d = np.diag(P_oo)
        with np.errstate(divide="ignore", invalid="ignore"):
            rbc[o] = np.where(d > 0.0, u * u / d, np.nan)
    pv = special.erfc(np.sqrt(np.maximum(rbc, 0.0) * 0.5))   # chi2_1 sf, NaN kept
    pv = np.where(np.isnan(rbc), np.nan, np.clip(pv, _P_FLOOR, 1.0))
    return rbc, pv


def conditional_impute(z: np.ndarray, Sigma: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Complete z: z_m = Sigma_mo Sigma_oo^{-1} z_o (Nelson, Taylor & MacGregor 1996).
    observed is a bool mask; all missing -> zeros; none missing -> z.

    Entries marked observed but non-finite are treated as missing. Returns a
    new array; observed entries are copied unchanged.
    """
    z = np.array(z, dtype=np.float64).ravel()
    obs = np.asarray(observed, dtype=bool).ravel()
    if obs.size != z.size:
        raise ValueError(f"len(observed)={obs.size} != len(z)={z.size}")
    obs = obs & np.isfinite(z)
    if obs.all():
        return z
    if not obs.any():
        return np.zeros_like(z)
    S = np.asarray(Sigma, dtype=np.float64)
    o = np.flatnonzero(obs)
    m = np.flatnonzero(~obs)
    L = _chol(_sym(_sub(S, o, o)))
    alpha, _ = dpotrs(L, z[o], lower=1)          # Sigma_oo^-1 z_o
    z[m] = _sub(S, m, o) @ alpha
    return z


@dataclass
class CholCache:
    """LRU(8) of Cholesky factors of Sigma_oo keyed by the missingness pattern
    (observed.tobytes()); invalidated by set_sigma. ~20 us per new pattern.

    Sigma is copied (and symmetrised) on construction and in set_sigma, so a
    caller mutating its array cannot desynchronise the cached factors; cached
    (idx_o, L) arrays are read-only because they are shared between calls.
    maxsize <= 0 disables caching.
    """
    Sigma: np.ndarray
    maxsize: int = CHOL_CACHE_SIZE
    _cache: "OrderedDict[bytes, Tuple[np.ndarray, np.ndarray]]" = field(default_factory=OrderedDict)

    def __post_init__(self) -> None:
        self.set_sigma(self.Sigma)

    def set_sigma(self, Sigma: np.ndarray) -> None:
        S = np.array(Sigma, dtype=np.float64)
        if S.ndim != 2 or S.shape[0] != S.shape[1]:
            raise ValueError(f"Sigma must be square, got shape {S.shape}")
        self.Sigma = _sym(S)
        self._cache = OrderedDict()     # fresh dict: never mutate one a copy shares

    def factor(self, observed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(idx_o, L) with L = cholesky(Sigma[idx_o][:, idx_o]) (lower)."""
        obs = np.asarray(observed, dtype=bool).ravel()
        if obs.size != self.Sigma.shape[0]:
            raise ValueError(f"len(observed)={obs.size} != p={self.Sigma.shape[0]}")
        key = obs.tobytes()
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        idx = np.flatnonzero(obs)
        L = _chol(_sub(self.Sigma, idx, idx))
        idx.flags.writeable = False
        L.flags.writeable = False
        entry = (idx, L)
        if self.maxsize > 0:
            self._cache[key] = entry
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
        return entry

    def t2(self, z: np.ndarray) -> Tuple[float, int]:
        """(T2 = z_o^T Sigma_oo^{-1} z_o, q = |o|) over finite entries of z; q = 0 -> (NaN, 0)."""
        z = np.asarray(z, dtype=np.float64).ravel()
        if z.size != self.Sigma.shape[0]:
            raise ValueError(f"len(z)={z.size} != p={self.Sigma.shape[0]}")
        obs = np.isfinite(z)
        q = int(obs.sum())
        if q == 0:
            return math.nan, 0
        idx, L = self.factor(obs)
        y = _tri(L, z[idx])
        return float(y @ y), q
