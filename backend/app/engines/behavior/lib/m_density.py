"""Read accessors and shared scoring maths for model.density / model.groups
(owner: B06 MultivariateEngine; contract C).

Why a density model: the marginal detectors (B04) test each feature against
its own predictive, so they miss two things. A tick where many features are
each a little high is jointly unusual (magnitude: Hotelling T2), and a tick
where two features that always move together move apart is unusual even if
each value is ordinary (correlation break: SPE, the squared distance from the
principal subspace). Both need the correlation structure of the entity's
standardised residuals (behavior.zi), which B06 fits robustly on gated rows.

Consumers call these functions instead of reading the model dict, so the
layout can evolve in one place and B29 (explain) replays exactly what B06
scored: B06 itself scores through `score_model` / `contributions_model`.
Everything here is pure (no store writes, nothing a caller can observe is
mutated; the Cholesky LRU inside the model is a cache).

Signatures (all z are 52-dim FEATURE_SPEC v2 rows of behavior.zi; NaN / inf
entries are missing):
    get(store, s, e) -> dict | None                 fitted model.density or None
    is_fitted(model) -> bool
    score(store, s, e, z) -> Score | None           T2 / SPE / p-values / WH
    score_model(model, z) -> Score                  (pure; B06 and B29)
    contributions(store, s, e, z, top=5) -> [dict] | None
    contributions_model(model, z) -> (rbc[52], p[52])
    ranked(rbc, p, top=5, z=None) -> [{feature, idx, group, rbc, p, z}]
    axes_from_contrib(rbc, p, top=3, alpha=0.01) -> [feature group]
    loglik(store, s, e, z) -> float                 log N(z_o; mu_o, Sigma_oo)
    sigma(store, s, e) -> float64[52, 52] | None    identity on unmodelled dims
    mean(store, s, e) -> float64[52] | None
    describe(store, s, e) / descriptor(model) -> dict   (profile.extra.mv_model, B30)
    groups(store, s, split_by_feature_group=False) -> [[idx]]   dependence groups (B04)
    group_of(store, s) -> int[52]
    assemble(mu, Sigma, cols, n, n_pred, box, ...) -> dict   (B06 builds models with it)

model.density@(s, e | 'class:<rid>') layout (a dict stored by reference):
    fmt, version        layout id, and the fit counter (= store model_version)
    fitted              False until the first fit (cold entity: never scored)
    mu     float64[52]  robust centre (0 on unmodelled dims)
    Sigma  float64[52, 52]  robust (OAS + C-step) covariance, eigen-floored,
                        identity on unmodelled rows / columns
    cols   int[]        modelled dims (finite in most training rows, no ties)
    U_k    float64[52, k]   principal loadings (zero rows on unmodelled dims)
    lam    float64[k]   principal eigenvalues, descending
    k, n                PCs kept (90 % of the modelled variance); own n_eff
    n_pred              n of the Hotelling prediction distribution (n_eff,
                        + CLASS_PRIOR_N while shrunk towards the class)
    theta1              residual trace sum_{j > k} lambda_j (SPE normaliser)
    box    (g, h)       Box approximation SPE / theta1 ~ g chi2_h, fitted on
                        OUT-OF-SAMPLE normalised SPE (NaN: no SPE evidence)
    box_src             'oos' | 'crossfit' | 'class' | 'given' | None
    null                {'spe': (g, d1, d2) | None, 't2': (s, d2) | None} or
                        absent: the cross-fitted F null (below) that replaces
                        the Box / Hotelling p when present
    class_key, class_w  the class model shrunk towards and its weight
    fitted_ts           ts of the fit
    chol_cache          robustcov.CholCache of Sigma[cols, cols] (LRU of 8
                        missingness patterns)
    Sigma_c, U_c        the cols blocks of Sigma and U_k (scoring fast path)
    _*                  engine-internal (learner state, gate, archive, rings)

model.groups@(s, '__system__'):
    {"fmt", "version", "ts", "n_rows", "group_of": int[52],
     "groups": [[idx]], "names": [[feature]], "rho_min": 0.8}
    Dependence groups: average-linkage clusters of 1 - |Spearman rho| cut at
    0.2 over the system's committed zi. Absent -> singletons.

Scoring (all on the modelled dims, centred by mu, observed o = finite):
    T2 = z_o^T Sigma_oo^-1 z_o (Cholesky of Sigma_oo cached per pattern),
    p_t2 = Hotelling prediction p with n = n_pred and q = |o|;
    wh = Wilson-Hilferty normal score of q F, F the prediction-scaled T2 (so a
         finite-n fit does not bias the score; q F -> chi2_q as n grows);
    completion z_m = Sigma_mo Sigma_oo^-1 z_o (Nelson, Taylor & MacGregor
         1996, the same factor as T2), SPE = ||z_c - U U^T z_c||^2 and
         p_spe = chi2_h.sf((SPE / theta1) / g);
    RBC_f = (Sigma_oo^-1 z_o)_f^2 / (Sigma_oo^-1)_ff on the observed sub-model,
         each chi2_1 (Alcala & Qin 2009), equal to robustcov.rbc_contributions.
Why SPE is normalised by theta1: the Box parameters are fitted on SPE of rows
the model had not seen, which are collected across successive refits; theta1
is the model's own expected SPE, so SPE / theta1 has the same scale across
refits whose k differs.

Cross-fitted null (round 4; model["null"], built by B06 with `null_stats` /
`null_params`). The Gaussian p-values above are anti-conservative on live
data (pack A seed 0, clean control H rows: SPE p < 1e-3 at 11x nominal, T2 at
44x; KS 0.29 / 0.17). Two measured reasons: (1) the OOS SPE ring mixed values
of successive, younger models (its mean fell from 19 to 1 over the warm-up)
and a moment-matched g chi2_h has a light tail; (2) the standardised
residuals are not one stationary Gaussian: their scale drifts by +-25 %
between multi-day periods (mean zi^2 per dim 0.50 -> 0.38 -> 0.50 over the
3600-s warm-up, the 900-s warm-up and live), so relative to its median SPE
has a log-sd of 0.64 and q99.9 / q50 = 8-18 where chi2_14 gives 5.5. The
null is therefore taken from the model's own out-of-sample errors on its
training population: the buffer is split into K = 4 contiguous time blocks,
each block is scored under a model refitted without it (same robust fit and
class / H shrink), and the held-out SPE / theta1 and T2 / q are summarised by
their log-moments. A scale mixture of chi2 (the scale varying between rows)
is a scaled F, so
    SPE / theta1 ~ g F(d1, d2),  d1 = theta1^2 / theta2 of the current model
                                 (Box's dof of the residual eigenvalues);
    T2 / q       ~ s F(q, d2'),  s rescaled from the folds' n to the full n
                                 by the Hotelling factor (n^2 - 1)/(n (n - q));
d2 and the scale solve the log-moment equations Var log F = psi1(d1/2) +
psi1(d2/2), E log F = log(d2/d1) + psi(d1/2) - psi(d2/2). d2 -> inf gives
back g chi2_d1 / d1 and the Hotelling F. Pack A seed 0, clean control live
rows, model p before -> after (engine run): SPE KS 0.29 -> 0.04, p < 1e-3 at
11x -> 2.1x nominal; T2 0.17 -> 0.12, 44x -> 4.1x; T2_q 0.33 -> 0.03, 74x ->
7x; SPE_q 0.06 -> 0.05, 2.3x -> 0.7x. Held-out SPE uses the full model's k
and theta1 (see B06 cross_null). Without a null (young entity, < 64 rows) the
Gaussian p-values above are used and B06 marks the detectors provisional.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import special
from scipy.linalg.lapack import dpotrs, dtrtri

from . import robustcov
from .features import FEATURE_DIM, FEATURE_GROUP, FEATURE_NAMES_V2, GROUP_ORDER, GROUPS

MODEL = "model.density"
GROUPS_MODEL = "model.groups"
SYSTEM_KEY = "__system__"
FMT = 1
CLASS_PRIOR_N = 30.0          # young entity: (n S_e + 30 S_class) / (n + 30)
AXES_ALPHA = 0.01
AXES_TOP = 3

_NAN = math.nan
_GROUP_ARR = np.array([FEATURE_GROUP[n] for n in FEATURE_NAMES_V2], dtype=object)


# ================================================================ dataclasses
@dataclass(frozen=True)
class Score:
    """One scored row. z / z_completed are centred (z - mu) and 52-dim:
    z is NaN where unobserved or unmodelled; z_completed is NaN only on
    unmodelled dims (missing modelled dims hold the conditional expectation)."""
    t2: float
    q: int
    p_t2: float
    wh: float
    spe: float
    spe_norm: float
    p_spe: float
    z: np.ndarray
    z_completed: np.ndarray
    n_pred: float
    k: int

    @property
    def scored(self) -> bool:
        return self.q > 0 and self.t2 == self.t2


def _empty_score(model: Optional[Mapping[str, Any]] = None) -> Score:
    nan52 = np.full(FEATURE_DIM, _NAN)
    return Score(_NAN, 0, _NAN, _NAN, _NAN, _NAN, _NAN, nan52, nan52.copy(),
                 float(model.get("n_pred", _NAN)) if model else _NAN,
                 int(model.get("k", 0)) if model else 0)


# ================================================================ model access
def is_fitted(model: Any) -> bool:
    return (isinstance(model, Mapping) and model.get("fmt") == FMT and bool(model.get("fitted"))
            and model.get("chol_cache") is not None)


def get(store: Any, s: str, e: str) -> Optional[Dict[str, Any]]:
    """The fitted model.density@(s, e), else None (absent, other layout, cold)."""
    m = store.get_model(s, e, MODEL)
    return m if is_fitted(m) else None


def sigma(store: Any, s: str, e: str) -> Optional[np.ndarray]:
    """Full 52x52 covariance of the standardised residuals (a copy); unmodelled
    dims carry the null prior (unit variance, uncorrelated). None if unfitted."""
    m = get(store, s, e)
    return None if m is None else np.array(m["Sigma"], dtype=np.float64)


def mean(store: Any, s: str, e: str) -> Optional[np.ndarray]:
    m = get(store, s, e)
    return None if m is None else np.array(m["mu"], dtype=np.float64)


# ================================================================ assembly
def assemble(mu: np.ndarray, Sigma: np.ndarray, cols: Sequence[int], n: float, n_pred: float,
             box: Tuple[float, float] = (_NAN, _NAN), box_src: Optional[str] = None,
             class_key: Optional[str] = None, class_w: float = 0.0, fitted_ts: float = _NAN,
             version: int = 0) -> Dict[str, Any]:
    """Build a fitted model dict from a full 52-dim (mu, Sigma) and the modelled
    dims `cols` (Sigma is eigen-floored on the cols block, and set to identity
    outside it). PCA keeps the smallest k explaining robustcov.PCA_VAR_FRAC."""
    p = FEATURE_DIM
    cols = np.array(sorted({int(c) for c in cols}), dtype=np.intp)
    mu_f = np.zeros(p)
    S_f = np.eye(p)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    S_in = np.asarray(Sigma, dtype=np.float64)
    if cols.size:
        mu_f[cols] = mu[cols]
        Sc = robustcov.eigen_floor(S_in[np.ix_(cols, cols)])
        S_f[np.ix_(cols, cols)] = Sc
    else:
        Sc = np.zeros((0, 0))
    U_c, lam, k = robustcov.pca_k(Sc) if cols.size else (np.zeros((0, 0)), np.zeros(0), 0)
    U = np.zeros((p, k))
    if k:
        U[cols] = U_c
    theta1 = float(np.trace(Sc) - lam.sum()) if cols.size else _NAN
    if not (k < cols.size and theta1 > 1e-12 * max(float(np.trace(Sc)), 1e-300)):
        theta1 = _NAN                         # no residual subspace: no SPE evidence
    g, h = (float(box[0]), float(box[1])) if box is not None else (_NAN, _NAN)
    return {
        "fmt": FMT, "version": int(version), "fitted": bool(cols.size),
        "mu": mu_f, "Sigma": S_f, "cols": cols, "U_k": U, "lam": np.asarray(lam, dtype=np.float64),
        "k": int(k), "n": float(n), "n_pred": float(n_pred), "theta1": theta1,
        "box": (g, h), "box_src": box_src if g == g else None,
        "class_key": class_key, "class_w": float(class_w), "fitted_ts": float(fitted_ts),
        "chol_cache": robustcov.CholCache(Sc) if cols.size else None,
        "Sigma_c": Sc, "U_c": np.asarray(U_c, dtype=np.float64),
    }


# ================================================================ scoring
def _centred(model: Mapping[str, Any], z: Any) -> Tuple[np.ndarray, np.ndarray]:
    """(zc over cols, full 52 centred z with NaN outside cols / unobserved)."""
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if z.size != FEATURE_DIM:
        raise ValueError(f"m_density: expected a {FEATURE_DIM}-dim row, got {z.size}")
    cols = model["cols"]
    zc = z[cols] - model["mu"][cols]
    zc[~np.isfinite(zc)] = _NAN
    full = np.full(FEATURE_DIM, _NAN)
    full[cols] = zc
    return zc, full


def score_model(model: Mapping[str, Any], z: Any) -> Score:
    """T2, SPE, their model p-values and the WH score of one row (see module doc).
    No observed modelled dim -> an unscored Score (q = 0, all NaN)."""
    if not is_fitted(model):
        return _empty_score(model)
    zc, zfull = _centred(model, z)
    obs = np.isfinite(zc)
    q = int(obs.sum())
    if q == 0:
        return _empty_score(model)
    idx, L = model["chol_cache"].factor(obs)
    zo = zc[idx]
    alpha, _ = dpotrs(L, zo, lower=1)                    # Sigma_oo^-1 z_o
    t2 = float(zo @ alpha)
    n_pred = float(model["n_pred"])
    p_t2 = robustcov.hotelling_pred_p(t2, n_pred, q)
    if p_t2 == p_t2:
        qf = t2 * n_pred * (n_pred - q) / ((n_pred - 1.0) * (n_pred + 1.0))
        wh = robustcov.wilson_hilferty(qf, q)
    else:
        wh = _NAN
    # conditional completion with the same factor (Nelson et al. 1996)
    comp = zc.copy()
    if q < zc.size:
        miss = np.flatnonzero(~obs)
        comp[miss] = model["Sigma_c"][np.ix_(miss, idx)] @ alpha
    zcomp = np.full(FEATURE_DIM, _NAN)
    zcomp[model["cols"]] = comp
    spe = spe_n = p_spe = _NAN
    th = float(model["theta1"])
    nl = model.get("null")
    if th == th:
        U = model["U_c"]
        r = comp - U @ (U.T @ comp)
        spe = float(r @ r)
        spe_n = spe / th
        if nl is not None and nl.get("spe") is not None:
            p_spe = f_sf(spe_n, *nl["spe"])
        else:
            g, h = model["box"]
            p_spe = robustcov.spe_p(spe_n, g, h)
    if nl is not None and nl.get("t2") is not None and t2 == t2:
        s_t2, d2 = nl["t2"]
        p_t2 = f_sf(t2 / q, s_t2, float(q), d2)
    return Score(t2, q, p_t2, wh, spe, spe_n, p_spe, zfull, zcomp, n_pred, int(model["k"]))


# ================================================================ null
NULL_D2_MIN = 1.0        # heaviest tail allowed (F with 1 denominator dof)
NULL_D2_MAX = 1e4        # ~ no scale mixing: g chi2_d1 / d1
NULL_MIN_ROWS = 64       # held-out rows needed for a null


def f_sf(x: float, scale: float, d1: float, d2: float) -> float:
    """P(scale F(d1, d2) > x), floored like the other model p-values."""
    x, scale = float(x), float(scale)
    if not (x == x and scale > 0.0 and d1 > 0.0 and d2 > 0.0):
        return _NAN
    if x <= 0.0:
        return 1.0
    return max(float(special.fdtrc(d1, d2, x / scale)), 1e-300)


def _solve_d2(excess: float) -> float:
    """d2 with psi1(d2 / 2) = excess (the log-variance not explained by the
    numerator dof), clipped to [NULL_D2_MIN, NULL_D2_MAX]."""
    if not excess > float(special.polygamma(1, NULL_D2_MAX / 2.0)):
        return NULL_D2_MAX
    if excess >= float(special.polygamma(1, NULL_D2_MIN / 2.0)):
        return NULL_D2_MIN
    lo, hi = NULL_D2_MIN, NULL_D2_MAX           # psi1 is decreasing: bisect in log d2
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if float(special.polygamma(1, mid / 2.0)) > excess:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def null_stats(values: np.ndarray, dof: Optional[np.ndarray] = None) -> Optional[Dict[str, float]]:
    """Log-moment summary of held-out statistics x_i > 0 (SPE / theta1, or
    T2 / q with dof = q_i): {'n', 'mlog', 'vlog', 'a', 'va', 'psi1'} where
    a_i = log x_i + log q_i - psi(q_i / 2) (T2 only). None below
    NULL_MIN_ROWS usable values."""
    x = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(x) & (x > 0.0)
    if dof is not None:
        dof = np.asarray(dof, dtype=np.float64)
        ok &= np.isfinite(dof) & (dof > 0.0)
    if int(ok.sum()) < NULL_MIN_ROWS:
        return None
    lx = np.log(x[ok])
    out = {"n": float(ok.sum()), "mlog": float(lx.mean()), "vlog": float(lx.var(ddof=1))}
    if dof is not None:
        q = dof[ok]
        a = lx + np.log(q) - special.digamma(q / 2.0)
        out.update(a=float(a.mean()), va=float(a.var(ddof=1)),
                   psi1=float(np.mean(special.polygamma(1, q / 2.0))))
    return out



def null_spe(st: Mapping[str, float], d1: float) -> Optional[Tuple[float, float, float]]:
    """(g, d1, d2) of SPE / theta1 ~ g F(d1, d2) from null_stats of held-out
    SPE / theta1 and the scoring model's d1."""
    if st is None or not (d1 > 0.0 and math.isfinite(d1)):
        return None
    d2 = _solve_d2(st["vlog"] - float(special.polygamma(1, d1 / 2.0)))
    lg = (st["mlog"] - math.log(d2 / d1) - float(special.digamma(d1 / 2.0))
          + float(special.digamma(d2 / 2.0)))
    return math.exp(lg), float(d1), d2


def null_t2(st: Mapping[str, float], n_full: float, n_fold: float, qbar: float
            ) -> Optional[Tuple[float, float]]:
    """(s, d2) of T2 / q ~ s F(q, d2) from null_stats (with dof) of held-out
    T2 / q scored under fold models of size n_fold, rescaled to a model of
    size n_full by the ratio of the Hotelling factors (n^2 - 1)/(n (n - q))
    (a fold model with fewer rows inflates held-out T2 more)."""
    if st is None or "a" not in st:
        return None
    d2 = _solve_d2(st["va"] - st["psi1"])
    ls = st["a"] - math.log(d2) + float(special.digamma(d2 / 2.0))

    def c(n: float) -> float:
        return (n * n - 1.0) / (n * (n - qbar))
    ratio = c(n_full) / c(n_fold) if (n_full > qbar + 1.0 and n_fold > qbar + 1.0) else 1.0
    return math.exp(ls) * ratio, d2


def residual_dof(model: Mapping[str, Any]) -> float:
    """Box's dof theta1^2 / theta2 of the residual eigenvalues (NaN without a
    residual subspace)."""
    Sc = model.get("Sigma_c")
    k = int(model.get("k", 0))
    if Sc is None or not np.asarray(Sc).size:
        return _NAN
    lam = np.linalg.eigvalsh(np.asarray(Sc, dtype=np.float64))[::-1][k:]
    lam = lam[lam > 0.0]
    if not lam.size:
        return _NAN
    return float(lam.sum() ** 2 / np.sum(lam * lam))


def score(store: Any, s: str, e: str, z: Any) -> Optional[Score]:
    m = get(store, s, e)
    return None if m is None else score_model(m, z)


def contributions_model(model: Mapping[str, Any], z: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruction-based contributions (rbc[52], p[52]) on the observed
    sub-model; NaN on missing or unmodelled dims. Equal to
    robustcov.rbc_contributions(z - mu, inv(Sigma)) restricted to cols."""
    rbc = np.full(FEATURE_DIM, _NAN)
    pv = np.full(FEATURE_DIM, _NAN)
    if not is_fitted(model):
        return rbc, pv
    zc, _ = _centred(model, z)
    obs = np.isfinite(zc)
    if not obs.any():
        return rbc, pv
    idx, L = model["chol_cache"].factor(obs)
    alpha, _ = dpotrs(L, zc[idx], lower=1)
    Li, _ = dtrtri(L, lower=1)
    d = np.einsum("ij,ij->j", Li, Li)                    # diag(Sigma_oo^-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(d > 0.0, alpha * alpha / d, _NAN)
    at = model["cols"][idx]
    rbc[at] = r
    pr = special.erfc(np.sqrt(np.maximum(r, 0.0) * 0.5))  # chi2_1 sf
    pv[at] = np.where(np.isnan(r), _NAN, np.clip(pr, 1e-300, 1.0))
    return rbc, pv


def ranked(rbc: np.ndarray, p: np.ndarray, top: Optional[int] = 5,
           z: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """Features by descending RBC (finite only): JSON-friendly dicts."""
    rbc = np.asarray(rbc, dtype=np.float64)
    fin = np.flatnonzero(np.isfinite(rbc))
    order = fin[np.argsort(-rbc[fin], kind="stable")]
    if top is not None:
        order = order[:int(top)]
    out = []
    for i in order.tolist():
        d = {"feature": FEATURE_NAMES_V2[i], "idx": i, "group": FEATURE_GROUP[FEATURE_NAMES_V2[i]],
             "rbc": float(rbc[i]), "p": float(p[i])}
        if z is not None:
            zi = float(z[i])
            d["z"] = zi if math.isfinite(zi) else None
        out.append(d)
    return out


def contributions(store: Any, s: str, e: str, z: Any, top: Optional[int] = 5
                  ) -> Optional[List[Dict[str, Any]]]:
    """Ranked RBC contributions of row z under the entity's model (B29)."""
    m = get(store, s, e)
    if m is None:
        return None
    rbc, p = contributions_model(m, z)
    return ranked(rbc, p, top, _centred(m, z)[1])


def axes_from_contrib(rbc: np.ndarray, p: np.ndarray, top: int = AXES_TOP,
                      alpha: float = AXES_ALPHA) -> List[str]:
    """Feature groups of the top-`top` RBC features with p < alpha, in
    GROUP_ORDER; if none reaches alpha, the group of the largest RBC."""
    rbc = np.asarray(rbc, dtype=np.float64)
    fin = np.flatnonzero(np.isfinite(rbc))
    if not fin.size:
        return []
    order = fin[np.argsort(-rbc[fin], kind="stable")]
    sig = [i for i in order[:int(top)].tolist() if p[i] < alpha]
    pick = sig if sig else order[:1].tolist()
    gs = {_GROUP_ARR[i] for i in pick}
    return [g for g in GROUP_ORDER if g in gs]


def loglik(store: Any, s: str, e: str, z: Any) -> float:
    """Gaussian log-density of the observed modelled dims, NaN if unfitted or
    nothing observed: -(T2 + log det Sigma_oo + q log 2 pi) / 2."""
    m = get(store, s, e)
    if m is None:
        return _NAN
    zc, _ = _centred(m, z)
    obs = np.isfinite(zc)
    q = int(obs.sum())
    if q == 0:
        return _NAN
    idx, L = m["chol_cache"].factor(obs)
    alpha, _ = dpotrs(L, zc[idx], lower=1)
    logdet = 2.0 * float(np.log(np.diag(L)).sum())
    return -0.5 * (float(zc[idx] @ alpha) + logdet + q * math.log(2.0 * math.pi))


# ================================================================ descriptors
def descriptor(model: Mapping[str, Any]) -> Dict[str, Any]:
    """profile.extra.mv_model: a JSON-friendly summary of a fitted model (the
    top-3 loading features of the first 3 PCs, variance explained, sizes and
    where the SPE calibration came from)."""
    def _f(x: float, nd: int = 4) -> Optional[float]:
        x = float(x)
        return round(x, nd) if math.isfinite(x) else None
    if not is_fitted(model):
        return {"fitted": False}
    Sc = model["Sigma_c"]
    tot = float(np.trace(Sc))
    lam = model["lam"]
    pcs = []
    for j in range(min(3, int(model["k"]))):
        u = model["U_k"][:, j]
        top = np.argsort(-np.abs(u), kind="stable")[:3]
        pcs.append({"var_frac": _f(lam[j] / tot if tot > 0 else _NAN),
                    "features": [FEATURE_NAMES_V2[i] for i in top.tolist()],
                    "loadings": [_f(u[i], 3) for i in top.tolist()]})
    g, h = model["box"]
    modelled = set(np.asarray(model["cols"]).tolist())
    return {
        "fitted": True, "version": int(model.get("version", 0)),
        "fitted_ts": _f(model.get("fitted_ts", _NAN), 1),
        "p": int(np.asarray(model["cols"]).size), "k": int(model["k"]),
        "var_explained": _f(float(lam.sum()) / tot if tot > 0 else _NAN),
        "n": _f(model["n"], 1), "n_pred": _f(model["n_pred"], 1),
        "young": bool(model.get("class_w", 0.0) > 0.0),
        "class_key": model.get("class_key"), "class_w": _f(model.get("class_w", 0.0)),
        "spe_box": {"g": _f(g), "h": _f(h), "src": model.get("box_src")},
        "null": _null_desc(model.get("null"), _f),
        "pcs": pcs,
        "unmodelled": [FEATURE_NAMES_V2[i] for i in range(FEATURE_DIM) if i not in modelled],
    }


def _null_desc(nl: Any, f: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(nl, Mapping):
        return None
    out: Dict[str, Any] = {"src": "crossfit"}
    if nl.get("spe") is not None:
        out["spe"] = {"g": f(nl["spe"][0]), "d1": f(nl["spe"][1]), "d2": f(nl["spe"][2])}
    if nl.get("t2") is not None:
        out["t2"] = {"s": f(nl["t2"][0]), "d2": f(nl["t2"][1])}
    return out


def describe(store: Any, s: str, e: str) -> Dict[str, Any]:
    m = store.get_model(s, e, MODEL)
    return descriptor(m) if isinstance(m, Mapping) else {"fitted": False}


# ================================================================ groups
def _singletons() -> List[List[int]]:
    return [[i] for i in range(FEATURE_DIM)]


def groups(store: Any, s: str, split_by_feature_group: bool = False) -> List[List[int]]:
    """Dependence groups of system `s` (lists of feature indices, a partition
    of 0..51 ordered by smallest member). Absent model -> singletons. With
    split_by_feature_group each group is intersected with the FEATURE_SPEC
    groups, so an intensity channel (volume) never shares a group with
    shape features (B04 marg_int / marg_shape)."""
    m = store.get_model(s, SYSTEM_KEY, GROUPS_MODEL)
    gs = m.get("groups") if isinstance(m, Mapping) else None
    if not gs:
        gs = _singletons()
    gs = [sorted(int(i) for i in g) for g in gs if len(g)]
    if split_by_feature_group:
        out: List[List[int]] = []
        for g in gs:
            for fg in GROUP_ORDER:
                part = [i for i in g if i in _FG_SETS[fg]]
                if part:
                    out.append(part)
        gs = out
    return sorted(gs, key=lambda g: g[0])


def group_of(store: Any, s: str) -> np.ndarray:
    """int[52]: dependence-group label of each feature (0..G-1)."""
    out = np.zeros(FEATURE_DIM, dtype=np.int64)
    for j, g in enumerate(groups(store, s)):
        out[g] = j
    return out


_FG_SETS = {g: frozenset(GROUPS[g]) for g in GROUP_ORDER}
