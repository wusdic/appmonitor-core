"""MultivariateEngine (B06) — correlation-aware magnitude (Hotelling T2) and
correlation-break (SPE) scoring of the standardised residuals behavior.zi.

Why: the marginal detectors (B04) test one feature at a time. They miss a
tick where many features are each moderately high (jointly unusual: T2) and a
tick where features that always move together move apart, although each value
is ordinary (e.g. bytes_up up while flows is flat: SPE, the distance from the
principal subspace). This replaces the IsolationForest the old AnomalyEngine
refitted on every run, on every row including attack rows, at ~1 s per run:
here the model is a robust covariance fitted only on gated rows, scoring costs
tens of microseconds, and the contributions are exact (RBC).

What is learned, and how (contract H, lib/gating):
  * The learner state is the buffer of committed zi rows: late (D ticks),
    trust-weighted (w_eff = behavior.trust), held while quarantined,
    checkpointed hourly (rows as float16) and reversible under model.control
    (rollback_to / release / rebase_from / frozen). At most CAP = 336 rows,
    one per 15-minute slot (the newest committed row of the slot). DEVIATION
    from "336 rows": at 900 s this is the same thing; at 60 s a plain 336-row
    buffer would span 5.6 h of one daypart, while slots keep ~3.5 days at
    every cadence (and a replay is order-independent: per slot the max-ts
    row, then the newest 336 slots).
  * Refit every 16 ticks or 4 h (whichever comes first, per-entity phase),
    eagerly while the entity is young (n < 64), and at once after a control
    action: winsorise at +-4, OAS + C-step (robustcov.c_step_oas, which also
    reweights at chi2_{p,0.975} and floors eigenvalues at 1e-3 mean), PCA to
    90 % of the variance.
  * Missing dimensions: a column is modelled when it is finite in >= 50 % of
    the buffer and has no tie mass (one value holding half the entries: MCD
    fits collapse onto a point mass, robustcov docstring); a row is used when
    it observes >= 50 % of the modelled columns. Complete buffers take the
    spec path (c_step_oas). Buffers with missing entries (real zi: the app
    group is NaN in every tick without HTTP) are completed by one EM step
    (conditional expectation from a pairwise-complete first pass, plus the
    conditional covariance), screened by the C-step at chi2_{p,0.999}, and
    fitted by OAS: a C-step on imputed rows distorts the shape (1.9 % T2
    false alarms at 1 %, measured) and complete-case fitting kept 13 of 52
    columns at 3 % scattered NaN. Unmodelled dims are left to B04.
  * Young entity: Sigma~ = (n Sigma_e + 30 Sigma_class) / (n + 30) (and mu
    likewise) with model.density@(s, class:<rid>) when the entity's role class
    has >= 3 members in the system (contract L); the Hotelling n becomes
    n + 30. The class model is fitted on the members' committed buffers.
  * SPE Box parameters are fitted OUT OF SAMPLE (fitting them on the training
    rows gives 2.9-4 % false alarms at 1 %): at each refit the rows committed
    since the previous fit are scored under the previous model and their
    SPE / theta1 (theta1 = residual trace, so the scale is comparable across
    refits whose k differs) enters a 256-entry ring. Until the ring holds 30
    values: the class model's parameters, else a two-fold cross-fit of the
    buffer (bootstrap only), else NaN (no SPE evidence).
  * Every 32 ticks or 8 h per system: dependence groups (|Spearman rho| > 0.8,
    average linkage) of the pooled committed rows -> model.groups@(s,
    __system__), which B04 uses to avoid counting one correlated signal twice.

Scoring (lib/m_density.score_model, shared with B29) runs against the model
as of the last commit, before this tick's commits: T2 on the observed dims
(Cholesky of Sigma_oo cached per missingness pattern) with the Hotelling
prediction p (n = n_eff, q = |o|); SPE of the completed vector (missing dims
imputed by conditional expectation); RBC for axes (feature groups of the top
contributions with p < 0.01); behavior.wh the Wilson-Hilferty score of T2
(taken after the prediction scaling, q F, so a finite-n fit does not bias it).
score.t2 / score.spe = -log10 of the model p (comparable across missingness
patterns), pm = the model p.

Absence is data: no zi row at now -> nothing scored. An active entity without
a zi row, an all-NaN zi row, or a failed B05 tick -> NaN + behavior.degraded
(contract M). A cold entity (no own fit, no class) is not scored (NaN, never
p = 1). ctx.training learns as usual (missing trust counts as 1) and B06 emits
no events at all. Exceptions are never swallowed.

Store: reads behavior.zi, feature.active, behavior.trust / trust_prov /
quarantine (gating), model.control, model.link (gating), model.class
(m_class), model.density (own and class); writes model.density@(s, e) and
@(s, class:<rid>), model.groups@(s, __system__), behavior.score / pm [t2, spe]
and behavior.axes / degraded (lib/emit), behavior.wh (1-element float32 vec
ring), profile.extra.mv_model (at refit) and profile.extra.last_contrib (when
p < 0.01), checkpoints 'density'.
"""
from __future__ import annotations

import math
import warnings
from bisect import bisect_left
from collections import OrderedDict
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from scipy import special
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

from ...core.engine import Context, Engine
from ...models.schema import EntityProfile
from .lib import emit, m_class, m_density, robustcov
from .lib import gating as G
from .lib.classkeys import SYSTEM_KEY, class_kind
from .lib.features import FEATURE_DIM, FEATURE_NAMES_V2

ZI = "behavior.zi"
ACTIVE = "feature.active"
WH = "behavior.wh"
DENSITY = m_density.MODEL
GROUPS_MODEL = m_density.GROUPS_MODEL
B05_ENGINE = "behavior.common_mode"
LEARNER = "density"
DETS = ("t2", "spe")

# ---- learner / refit -------------------------------------------------------
CAP = 336                      # committed rows kept (one per slot)
SLOT_S = 900.0                 # one row per 15-minute slot (cadence invariant)
REFIT_TICKS, REFIT_S = 16, 4 * 3600.0
GROUPS_TICKS, GROUPS_S = 32, 8 * 3600.0
N_FIT_MIN = 32                 # complete rows needed for an own fit
EAGER_N = 64                   # below this n_eff every new commit refits
COL_MIN_FRAC = 0.5             # a column is modelled if finite in >= 50 % of rows
ROW_MIN_OBS = 0.5              # a row is used if it observes >= 50 % of the modelled set
SCREEN_Q = 0.999               # outlier screen of the EM path
CLASS_PRIOR_N = m_density.CLASS_PRIOR_N
CLASS_ROWS = 672               # pooled member rows per class fit
CLASS_MIN_MEMBERS = m_class.MIN_MEMBERS
REBASE_OLD_W = 0.25            # ACCEPTED regime: rows before tau keep a quarter weight
ARCH_KEEP_S = G.JOURNAL_MAX_AGE_S   # per-slot row archive for release / replay

# ---- SPE calibration ---------------------------------------------------------
OOS_CAP = 256
BOX_MIN = 30
OOS_MIN_W = 0.5                # only well-trusted rows calibrate the null
CROSSFIT_MIN = 64

# ---- groups ------------------------------------------------------------------
RHO_MIN = 0.8
GROUP_MIN_PAIRS = 30
GROUP_ROWS_PER_ENT = 96
GROUP_ROWS_MAX = 2400
GROUP_MIN_ROWS = 96

# ---- outputs -----------------------------------------------------------------
ALPHA = 0.01                   # axes / last_contrib threshold on the model p
TOP_CONTRIB = 5
_NAN = math.nan
_CARRY = ("_state", "_gate", "_arch", "_oos_ts", "_oos_v", "_sig", "_front")
_CTRL_KEYS = ("rollback_to", "release", "rebase_from", "frozen")


# ============================================================== learner state
# The state holds lists (slot order) of row ts, weights, slots and float32 rows.
# Rows are immutable by convention and shared between states, so a commit
# copies four lists of <= 336 references (~5 us) instead of the 336 x 52 array
# (~100 us with np.insert); arrays are built at refit time only.
def _init() -> Dict[str, list]:
    return {"ts": [], "w": [], "slot": [], "X": []}


def _update(st: Dict[str, list], row: Tuple[float, np.ndarray], w: float) -> Dict[str, list]:
    """Commit one row: per slot keep the newest-ts row, then the newest CAP
    slots. Pure (new lists), deterministic and order-independent, so a
    release of older held rows or a checkpoint replay gives the same buffer."""
    w = float(w)
    if not w > 0.0:
        return st
    ts, x = float(row[0]), row[1]
    slot = int(math.floor(ts / SLOT_S))
    sl = st["slot"]
    i = bisect_left(sl, slot)
    hit = i < len(sl) and sl[i] == slot
    if hit and ts <= st["ts"][i]:
        return st
    out = {"ts": list(st["ts"]), "w": list(st["w"]), "slot": list(sl), "X": list(st["X"])}
    if hit:
        out["ts"][i], out["w"][i], out["X"][i] = ts, w, x
        return out
    out["ts"].insert(i, ts)
    out["w"].insert(i, w)
    out["slot"].insert(i, slot)
    out["X"].insert(i, x)
    if len(out["ts"]) > CAP:
        out = {k: v[-CAP:] for k, v in out.items()}
    return out


def state_arrays(st: Dict[str, list]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ts[m], w[m], X[m, 52] float32) of a learner state, slot order."""
    if not st["ts"]:
        return np.empty(0), np.empty(0), np.empty((0, FEATURE_DIM), dtype=np.float32)
    return (np.asarray(st["ts"], dtype=np.float64), np.asarray(st["w"], dtype=np.float64),
            np.asarray(st["X"], dtype=np.float32))


def _dump(st: Dict[str, list]) -> Dict[str, np.ndarray]:
    """Checkpoint blob: rows as float16 (architecture section 3)."""
    ts, w, X = state_arrays(st)
    return {"ts": ts, "w": w, "slot": np.asarray(st["slot"], dtype=np.int64),
            "X": X.astype(np.float16)}


def _load(blob: Dict[str, np.ndarray]) -> Dict[str, list]:
    X = np.asarray(blob["X"], dtype=np.float32)
    return {"ts": [float(t) for t in blob["ts"]], "w": [float(v) for v in blob["w"]],
            "slot": [int(v) for v in blob["slot"]], "X": list(X)}


def _merge(own: Dict[str, list], other: Dict[str, list], w: float) -> Dict[str, list]:
    """Link seeding B := B_own + w A: A's rows join in the slots B lacks, at
    w times their weight (B's own rows win a shared slot)."""
    out = own
    have = set(own["slot"])
    for j, sl in enumerate(other["slot"]):
        if sl not in have:
            out = _update(out, (other["ts"][j], other["X"][j]), other["w"][j] * w)
    return out


def _on_rebase(st: Dict[str, list], tau: float) -> Dict[str, list]:
    """ACCEPTED regime from tau: the old regime keeps REBASE_OLD_W of its
    weight so the new one dominates the next refit without a cold start."""
    tau = float(tau)
    if not st["ts"] or st["ts"][0] >= tau:
        return st
    out = dict(st)
    out["w"] = [v * REBASE_OLD_W if t < tau else v for t, v in zip(st["ts"], st["w"])]
    return out


# ============================================================== fitting (pure)
class OwnFit(NamedTuple):
    cols: np.ndarray           # modelled dims
    mu: np.ndarray             # [len(cols)]
    Sigma: np.ndarray          # [len(cols), len(cols)]
    n_eff: float
    X: np.ndarray              # complete training rows on cols, float64
    w: np.ndarray
    ts: np.ndarray


def select_columns(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(cols, usable-row mask): columns finite in >= COL_MIN_FRAC of the rows
    (and in >= N_FIT_MIN rows) without tie mass; rows that observe at least
    ROW_MIN_OBS of those columns."""
    m = X.shape[0]
    fin = np.isfinite(X)
    cand = np.flatnonzero(fin.sum(axis=0) >= max(N_FIT_MIN, COL_MIN_FRAC * m))
    if cand.size:
        cand = cand[~_tie_mass(np.where(fin[:, cand], X[:, cand], np.nan))]
    if not cand.size:
        return cand.astype(np.intp), np.zeros(m, dtype=bool)
    rows = fin[:, cand].sum(axis=1) >= ROW_MIN_OBS * cand.size
    return cand.astype(np.intp), rows


def _tie_mass(A: np.ndarray) -> np.ndarray:
    """Per column: one value holds at least half of the finite entries (so the
    MAD is 0 and an MCD-type fit collapses onto it). One sort, no median."""
    Sx = np.sort(A, axis=0)                          # NaN sort last
    c = np.isfinite(A).sum(axis=0)
    h = np.maximum((c + 1) // 2, 1)
    i = np.arange(A.shape[0])[:, None]
    j = np.minimum(i + h - 1, A.shape[0] - 1)
    same = Sx == np.take_along_axis(Sx, j, axis=0)
    return (same & (i <= c - h)).any(axis=0)


def em_complete(X: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One EM step for rows with missing entries: (completed rows, mean
    conditional covariance). The first pass is the pairwise-complete
    (weighted) covariance of the winsorised rows, eigen-floored; each
    row is then completed by its conditional expectation
    x_m = mu_m - P_mm^-1 P_mo (x_o - mu_o) (P = Sigma^-1; |m| x |m| solves,
    stacked per missing count), and the conditional covariance P_mm^-1 is
    averaged (weights w) so the fitted variance of often-missing columns is
    not biased low by the imputation (the E-step's second moment). Observed
    entries are returned winsorised."""
    Xw = robustcov.winsorize(X)
    miss = ~np.isfinite(Xw)
    M = (~miss).astype(np.float64) * w[:, None]
    cnt = M.sum(axis=0)
    tot = np.where(miss, 0.0, Xw * w[:, None]).sum(axis=0)
    mu1 = np.where(cnt > 0, tot / np.maximum(cnt, 1e-300), 0.0)
    D = np.where(miss, 0.0, Xw - mu1)
    num = (D * w[:, None]).T @ D
    den = (~miss).astype(np.float64).T @ M
    with np.errstate(divide="ignore", invalid="ignore"):
        S1 = np.where(den > 0.0, num / den, 0.0)
    P = np.linalg.inv(robustcov.eigen_floor(S1, rel=0.05))
    out, C = _estep(Xw, mu1, P, w)
    return out, C / float(w.sum())


def _estep(X: np.ndarray, mu: np.ndarray, P: np.ndarray, w: Optional[np.ndarray] = None
           ) -> Tuple[np.ndarray, np.ndarray]:
    """Conditional expectation of the missing (non-finite) entries of each row
    under N(mu, P^-1): x_m = mu_m - P_mm^-1 P_mo (x_o - mu_o), batched by the
    number r of missing entries (one stacked r x r solve per r). Returns (the
    completed copy, sum_i w_i P_mm^-1 scattered on the missing blocks). Rows
    with nothing observed get mu."""
    out = np.array(X, dtype=np.float64)
    p = out.shape[1]
    C = np.zeros((p, p))
    miss = ~np.isfinite(out)
    nm = miss.sum(axis=1)
    for r in np.unique(nm[nm > 0]).tolist():
        rows = np.flatnonzero(nm == r)
        if r == p:
            out[rows] = mu
            continue
        Mi = np.nonzero(miss[rows])[1].reshape(rows.size, r)        # missing idx per row
        Dr = np.where(miss[rows], 0.0, out[rows] - mu)                # observed deviations
        Pm = P[Mi]                                                    # [k, r, p]
        Pmm = np.take_along_axis(Pm, Mi[:, None, :], axis=2)          # [k, r, r]
        rhs = -np.einsum("irp,ip->ir", Pm, Dr)
        Cm = np.linalg.inv(Pmm)
        out[rows[:, None], Mi] = mu[Mi] + np.einsum("irs,is->ir", Cm, rhs)
        if w is not None:
            np.add.at(C, (Mi[:, :, None], Mi[:, None, :]), Cm * w[rows][:, None, None])
    return out, C


def _mahal2(Z: np.ndarray, S: np.ndarray) -> np.ndarray:
    L = np.linalg.cholesky(robustcov.eigen_floor(S))
    Y = np.linalg.solve(L, Z.T)
    return np.einsum("ij,ij->j", Y, Y)


def _chi2_ppf(q: float, df: float) -> float:
    return float(special.chdtri(df, 1.0 - q))


def fit_rows(X: np.ndarray, w: np.ndarray, ts: Optional[np.ndarray] = None) -> Optional[OwnFit]:
    """Robust fit on the usable rows of the selected columns. Complete rows:
    robustcov.c_step_oas (winsorise, OAS, C-steps, reweight at chi2_0.975,
    consistency rescale, eigen floor). Rows with missing entries: one EM step
    (em_complete), the C-step as an outlier screen at chi2_{p, SCREEN_Q}, then
    the OAS of the completed survivors plus the E-step conditional covariance.
    None when fewer than N_FIT_MIN rows are usable."""
    X = np.asarray(X)
    w = np.asarray(w, dtype=np.float64)
    if X.shape[0] < N_FIT_MIN:
        return None
    cols, rows = select_columns(X)
    if not cols.size or int(rows.sum()) < N_FIT_MIN:
        return None
    Xc = X[rows][:, cols].astype(np.float64)
    wc = w[rows]
    tsc = np.asarray(ts, dtype=np.float64)[rows] if ts is not None else np.zeros(Xc.shape[0])
    wopt = None if np.all(wc == wc[0]) else wc
    if np.isfinite(Xc).all():
        mu, S = robustcov.c_step_oas(Xc, wopt)
    else:
        # C-step on imputed rows distorts the shape (the imputed rows sit near
        # the centre and crowd its h-subset: T2 false alarms 1.9 % at 1 %,
        # measured); it only screens outliers here, and the fit is the OAS of
        # the completed survivors plus the E-step conditional covariance
        # (1.0 % at 1 % clean, 0.9 % with 10 % contaminated rows).
        Xc, Cc = em_complete(Xc, wc)
        mu_r, S_r = robustcov.c_step_oas(Xc, wopt)
        keep = _mahal2(Xc - mu_r, S_r + Cc) <= _chi2_ppf(SCREEN_Q, Xc.shape[1])
        if int(keep.sum()) < N_FIT_MIN:
            keep[:] = True
        Xc, wc, tsc = Xc[keep], wc[keep], tsc[keep]
        mu, S, _ = robustcov.oas(Xc, None if np.all(wc == wc[0]) else wc)
        S = robustcov.eigen_floor(S + Cc)
    n_eff = float(wc.sum() ** 2 / np.dot(wc, wc))
    return OwnFit(cols, mu, S, n_eff, Xc, wc, tsc)


def _pca_residual(S: np.ndarray) -> Tuple[np.ndarray, float]:
    """(U_k, theta1) of a covariance block; theta1 NaN if there is no residual."""
    U, lam, k = robustcov.pca_k(S)
    th = float(np.trace(S) - lam.sum())
    if not (k < S.shape[0] and th > 0.0):
        return U, _NAN
    return U, th


def crossfit_spe(fit: OwnFit) -> Tuple[np.ndarray, np.ndarray]:
    """Two-fold out-of-sample SPE / theta1 on the fit's own rows (bootstrap of
    the Box calibration): (ts, values). Alternate rows form the folds."""
    n = fit.X.shape[0]
    idx = np.arange(n)
    vals, tss = [], []
    for a, b in ((idx[0::2], idx[1::2]), (idx[1::2], idx[0::2])):
        if a.size < N_FIT_MIN // 2 or not b.size:
            continue
        wa = fit.w[a]
        mu_h, S_h = robustcov.c_step_oas(fit.X[a], None if np.all(wa == wa[0]) else wa)
        U, th = _pca_residual(S_h)
        if th != th:
            continue
        R = fit.X[b] - mu_h
        R = R - (R @ U) @ U.T
        vals.append(np.einsum("ij,ij->i", R, R) / th)
        tss.append(fit.ts[b])
    if not vals:
        return np.empty(0), np.empty(0)
    return np.concatenate(tss), np.concatenate(vals)


def spe_norm_rows(model: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """SPE / theta1 of 52-dim rows under a fitted model, missing modelled dims
    completed by conditional expectation (as m_density.score_model does, but
    batched); rows observing no modelled dim give NaN."""
    out = np.full(X.shape[0], _NAN)
    th = float(model["theta1"])
    if th != th or not X.shape[0]:
        return out
    cols = model["cols"]
    Z = np.asarray(X, dtype=np.float64)[:, cols] - model["mu"][cols]
    Z[~np.isfinite(Z)] = _NAN
    ok = np.isfinite(Z).any(axis=1)
    if not ok.any():
        return out
    Z = Z[ok]
    if not np.isfinite(Z).all():
        Z, _ = _estep(Z, np.zeros(cols.size), np.linalg.inv(model["Sigma_c"]))
    U = model["U_c"]
    R = Z - (Z @ U) @ U.T
    out[ok] = np.einsum("ij,ij->i", R, R) / th
    return out


def combine(own: Optional[OwnFit], cm: Optional[Dict[str, Any]]
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """(mu, Sigma, cols, n, n_pred, class_w) in the full 52-dim embedding
    (identity / 0 on dims a part does not model). With a class model:
    (n Sigma_e + 30 Sigma_class) / (n + 30), mu likewise, n_pred = n + 30."""
    mu = np.zeros(FEATURE_DIM)
    S = np.eye(FEATURE_DIM)
    cols = np.zeros(0, dtype=np.intp)
    n = 0.0
    if own is not None:
        mu[own.cols] = own.mu
        S[np.ix_(own.cols, own.cols)] = own.Sigma
        cols, n = own.cols, own.n_eff
    if cm is None:
        return mu, S, cols, n, n, 0.0
    wc = CLASS_PRIOR_N / (n + CLASS_PRIOR_N)
    mu = (1.0 - wc) * mu + wc * np.asarray(cm["mu"], dtype=np.float64)
    S = (1.0 - wc) * S + wc * np.asarray(cm["Sigma"], dtype=np.float64)
    cols = np.union1d(cols, np.asarray(cm["cols"], dtype=np.intp))
    return mu, S, cols, n, n + CLASS_PRIOR_N, wc


def fit_model(X: np.ndarray, w: Optional[np.ndarray] = None, *, ts: Optional[np.ndarray] = None,
              cm: Optional[Dict[str, Any]] = None, box: Optional[Tuple[float, float]] = None,
              crossfit: bool = True, fitted_ts: float = _NAN, version: int = 1,
              class_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Offline fit of a density model on rows X [m, 52] (tests, class models,
    replay): own robust fit, class shrink, and the SPE Box parameters (given,
    else the class model's, else a two-fold cross-fit). None if unfittable."""
    X = np.asarray(X)
    w = np.ones(X.shape[0]) if w is None else np.asarray(w, dtype=np.float64)
    own = fit_rows(X, w, ts)
    if own is None and cm is None:
        return None
    mu, S, cols, n, n_pred, wc = combine(own, cm)
    src = None
    if box is not None:
        src = "given"
    elif cm is not None and all(math.isfinite(v) for v in cm["box"]):
        box, src = tuple(cm["box"]), "class"
    elif crossfit and own is not None and own.X.shape[0] >= CROSSFIT_MIN:
        _, v = crossfit_spe(own)
        box, src = (robustcov.spe_box_params(v) if v.size >= BOX_MIN else (_NAN, _NAN)), "crossfit"
    return m_density.assemble(mu, S, cols, n, n_pred, box if box is not None else (_NAN, _NAN),
                              src, class_key if cm is not None else None, wc,
                              fitted_ts, version)


# ============================================================== groups (pure)
def dependence_groups(X: np.ndarray, rho_min: float = RHO_MIN,
                      min_pairs: int = GROUP_MIN_PAIRS) -> Tuple[List[List[int]], np.ndarray]:
    """(groups, |rho|): average-linkage clusters of 1 - |Spearman rho| cut at
    1 - rho_min. rho is pairwise-complete on per-column (average) ranks; a
    pair with < min_pairs common rows, or a column with < min_pairs finite
    values, counts as independent. Groups are ordered by smallest member."""
    X = np.asarray(X, dtype=np.float64)
    N, p = X.shape
    fin = np.isfinite(X)
    R = np.zeros((N, p))
    for j in range(p):
        f = fin[:, j]
        if int(f.sum()) >= min_pairs:
            r = rankdata(X[f, j])
            R[f, j] = r - r.mean()
        else:
            fin[:, j] = False
    M = fin.astype(np.float64)
    num = R.T @ R
    A = (R * R).T @ M                              # A_ij = sum_k r_ki^2 m_kj
    den = np.sqrt(A * A.T)
    cnt = M.T @ M
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where((den > 0.0) & (cnt >= min_pairs), num / den, 0.0)
    rho = np.clip(np.abs(rho), 0.0, 1.0)
    np.fill_diagonal(rho, 1.0)
    D = 1.0 - rho
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    lab = fcluster(Z, t=1.0 - rho_min + 1e-12, criterion="distance")
    groups: Dict[int, List[int]] = {}
    for i, l in enumerate(lab.tolist()):
        groups.setdefault(l, []).append(i)
    return sorted(groups.values(), key=lambda g: g[0]), rho


# ============================================================== engine
class MultivariateEngine(Engine):
    name = "behavior.multivariate"
    layer = "behavior"
    consumes = [ZI, ACTIVE, "behavior.trust", "behavior.trust_prov", "behavior.quarantine",
                "model.class", DENSITY, "model.control", "model.link"]
    produces = [DENSITY, GROUPS_MODEL, emit.SCORE, emit.PM, emit.AXES, emit.DEGRADED, WH,
                "profile.extra.mv_model", "profile.extra.last_contrib"]
    description = ("Robust (OAS + C-step) covariance of gated zi rows per entity with class "
                   "shrink for young entities: Hotelling T2 (prediction p, missing-pattern "
                   "Cholesky cache), SPE with conditional imputation and out-of-sample Box "
                   "calibration, exact RBC contributions, dependence groups.")
    interval = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.refit_ticks = int(params.get("refit_ticks", REFIT_TICKS))
        self.refit_s = float(params.get("refit_s", REFIT_S))
        self.groups_ticks = int(params.get("groups_ticks", GROUPS_TICKS))
        self.groups_s = float(params.get("groups_s", GROUPS_S))
        self._learners: Dict[float, G.GatedLearner] = {}
        self._cur_arch: Optional["OrderedDict[int, Tuple[float, np.ndarray]]"] = None

    # ------------------------------------------------------------------ learner
    def _learner(self, d_min_s: float) -> G.GatedLearner:
        lrn = self._learners.get(d_min_s)
        if lrn is None:
            lrn = self._learners[d_min_s] = G.GatedLearner(
                name=LEARNER, init=_init, update=_update, fetch=self._fetch, dump=_dump,
                load=_load, merge=_merge, on_rebase=_on_rebase, d_min_s=d_min_s,
                ckpt_every_s=G.CKPT_EVERY_S, clock=ZI)
        return lrn

    def _fetch(self, store: Any, s: str, e: str, ts: float
               ) -> Optional[Tuple[float, np.ndarray]]:
        """The zi row at ts: the store ring (6 h), else the per-slot archive
        (8 d; only the newest row of each slot, which is the one kept)."""
        row = store.vec_at(s, e, ZI, ts)
        if row is None:
            hit = self._cur_arch.get(int(math.floor(ts / SLOT_S))) if self._cur_arch else None
            if hit is None or hit[0] != ts:
                return None
            row = hit[1]
        row = np.asarray(row, dtype=np.float32)
        if not np.isfinite(row).any():
            return None
        return float(ts), row

    # ---------------------------------------------------------------------- run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        now, dt = float(ctx.now), float(ctx.window_s)
        d_min = ctx.config.get("D_min_s") or G.D_MIN_S
        lrn = self._learner(float(d_min))
        b05_failed = store.engine_failed(B05_ENGINE, now)
        n = 0
        for s in store.systems():
            for e in store.entities(s):
                n += self._entity(ctx, lrn, s, e, now, dt, b05_failed)
            self._system_models(ctx, s, now, dt)
        return n

    # --------------------------------------------------------------- per entity
    def _entity(self, ctx: Context, lrn: G.GatedLearner, s: str, e: str, now: float,
                dt: float, b05_failed: bool) -> int:
        store = ctx.store
        model = store.get_model(s, e, DENSITY)
        if not (isinstance(model, dict) and model.get("fmt") == m_density.FMT
                and "_state" in model):
            model = None
        zi = store.vec_at(s, e, ZI, now)
        if zi is not None and zi.size != FEATURE_DIM:
            raise ValueError(f"B06: {ZI} has dim {zi.size}, expected {FEATURE_DIM}")
        has_obs = zi is not None and bool(np.isfinite(zi).any())
        written = 0
        if (b05_failed or (zi is not None and not has_obs)
                or (zi is None and _active(store, s, e, now))):
            cause = ("producer_error:" + B05_ENGINE if b05_failed
                     else "nan:" + ZI if zi is not None else "stale:" + ZI)
            emit.write_scores(store, s, e, now, {d: None for d in DETS},
                              degraded={d: cause for d in DETS}, window_s=int(dt))
            store.add_vec(s, e, WH, now, [_NAN], window_s=int(dt))
            written = 1
        if model is None:
            if not has_obs:
                return written
            model = _new_model()
        if has_obs:
            _archive(model["_arch"], now, zi)
            if not b05_failed and m_density.is_fitted(model):
                written = self._score(ctx, s, e, model, zi, now, dt)
        ctrl = self._learn(ctx, lrn, s, e, model, now, dt)
        model = self._maybe_refit(ctx, s, e, model, now, dt, ctrl)
        store.put_model(s, e, DENSITY, model, version=model["version"], ts=now)
        return written

    def _score(self, ctx: Context, s: str, e: str, model: Dict[str, Any], zi: np.ndarray,
               now: float, dt: float) -> int:
        store = ctx.store
        sc = m_density.score_model(model, zi)
        if not sc.scored:
            return 0
        scores = {"t2": _neglog10(sc.p_t2), "spe": _neglog10(sc.p_spe)}
        pm = {"t2": _fin(sc.p_t2), "spe": _fin(sc.p_spe)}
        axes: Dict[str, List[str]] = {}
        if sc.p_t2 < ALPHA or sc.p_spe < ALPHA:
            rbc, pv = m_density.contributions_model(model, zi)
            ax = m_density.axes_from_contrib(rbc, pv)
            for d, p in (("t2", sc.p_t2), ("spe", sc.p_spe)):
                if p < ALPHA and ax:
                    axes[d] = ax
            _put_extra(store, s, e, "last_contrib", {
                "ts": now, "t2": _js(sc.t2), "p_t2": _js(sc.p_t2), "spe": _js(sc.spe),
                "p_spe": _js(sc.p_spe), "q": sc.q, "axes": ax,
                "top": m_density.ranked(rbc, pv, TOP_CONTRIB, sc.z),
                "model_version": int(model["version"])})
        emit.write_scores(store, s, e, now, scores, pm=pm, axes=axes or None, window_s=int(dt))
        store.add_vec(s, e, WH, now, [sc.wh], window_s=int(dt))
        return 1

    def _learn(self, ctx: Context, lrn: G.GatedLearner, s: str, e: str,
               model: Dict[str, Any], now: float, dt: float) -> bool:
        """Gated commits (+ control, + link seeding). True when a control
        directive was applied (the caller refits at once)."""
        store = ctx.store
        gate0: G.GateState = model["_gate"]
        before = [gate0.applied.get(k) for k in _CTRL_KEYS]
        self._cur_arch = model["_arch"]
        try:
            st, gate = lrn.step(store, s, e, model["_state"], gate0, now, dt,
                                training=bool(ctx.training))
            st, gate = lrn.seed_from_link(store, s, e, st, gate,
                                          lambda a: _other_state(store, s, a))
        finally:
            self._cur_arch = None
        model["_state"], model["_gate"] = st, gate
        after = [gate.applied.get(k) for k in _CTRL_KEYS]
        if after[0] != before[0] and after[0] is not None:
            tau = float(after[0])                  # rollback: forget later calibration
            keep = [i for i, t in enumerate(model["_oos_ts"]) if t <= tau]
            model["_oos_ts"] = [model["_oos_ts"][i] for i in keep]
            model["_oos_v"] = [model["_oos_v"][i] for i in keep]
            model["_front"] = min(model["_front"], tau)
        return after != before

    # ------------------------------------------------------------------- refit
    def _maybe_refit(self, ctx: Context, s: str, e: str, model: Dict[str, Any], now: float,
                     dt: float, ctrl: bool) -> Dict[str, Any]:
        st = model["_state"]
        sig = (len(st["ts"]), math.fsum(st["ts"]), math.fsum(st["w"]),
               int(model["_gate"].version))
        if sig == model["_sig"] and not ctrl:
            return model
        eager = not model.get("fitted") or float(model.get("n", 0.0)) < EAGER_N
        if not (ctrl or eager or self.entity_due((s, e, "fit"), now,
                                                 min(self.refit_ticks * dt, self.refit_s))):
            return model
        return self._refit(ctx, s, e, model, now, sig)

    def _refit(self, ctx: Context, s: str, e: str, model: Dict[str, Any], now: float,
               sig: Tuple) -> Dict[str, Any]:
        store = ctx.store
        ts_a, w_a, X_a = state_arrays(model["_state"])
        oos_ts, oos_v = list(model["_oos_ts"]), list(model["_oos_v"])
        # 1. rows committed since the last fit, scored under that (older) model
        if m_density.is_fitted(model) and ts_a.size:
            new = (ts_a > model["_front"]) & (w_a >= OOS_MIN_W)
            if new.any():
                v = spe_norm_rows(model, X_a[new])
                ok = np.isfinite(v)
                oos_ts += ts_a[new][ok].tolist()
                oos_v += v[ok].tolist()
        # 2. own fit + class shrink
        own = fit_rows(X_a, w_a, ts_a)
        ck = m_class.class_key(store, s, e, CLASS_MIN_MEMBERS)
        cm = m_density.get(store, s, ck) if ck else None
        front = float(ts_a.max()) if ts_a.size else -math.inf
        if own is None and cm is None:
            if not model.get("fitted"):
                model["_sig"] = sig
                return model
            new_model = _new_model(model)            # lost its fit (e.g. rollback)
            new_model["version"] = int(model["version"]) + 1
        else:
            mu, S, cols, n, n_pred, wc = combine(own, cm)
            # 3. SPE calibration: out-of-sample ring, else class, else cross-fit
            box, src = (_NAN, _NAN), None
            if len(oos_v) < BOX_MIN and own is not None and own.X.shape[0] >= CROSSFIT_MIN \
                    and (cm is None or not all(math.isfinite(x) for x in cm["box"])):
                t_cf, v_cf = crossfit_spe(own)
                if v_cf.size:
                    oos_ts, oos_v = t_cf.tolist() + oos_ts, v_cf.tolist() + oos_v
                    src = "crossfit"
            if len(oos_v) >= BOX_MIN:
                box = robustcov.spe_box_params(np.asarray(oos_v[-OOS_CAP:]))
                src = src or "oos"
            elif cm is not None:
                box, src = tuple(cm["box"]), "class"
            new_model = m_density.assemble(mu, S, cols, n, n_pred, box, src,
                                           ck if cm is not None else None, wc, now,
                                           int(model["version"]) + 1)
            for k in _CARRY:
                new_model[k] = model[k]
        new_model["_oos_ts"], new_model["_oos_v"] = oos_ts[-OOS_CAP:], oos_v[-OOS_CAP:]
        new_model["_sig"] = sig
        new_model["_front"] = front
        _put_extra(store, s, e, "mv_model", m_density.descriptor(new_model))
        return new_model

    # --------------------------------------------------------- system / class
    def _system_models(self, ctx: Context, s: str, now: float, dt: float) -> None:
        store = ctx.store
        per = min(self.groups_ticks * dt, self.groups_s)
        roles = [k for k in m_class.all_class_keys(store, s, CLASS_MIN_MEMBERS)
                 if class_kind(k) == "role"]
        if roles:
            due = self.entity_due((s, "__classes__"), now, per)
            for ck in roles:
                if due or m_density.get(store, s, ck) is None:
                    self._fit_class(ctx, s, ck, now)
        have = store.get_model(s, SYSTEM_KEY, GROUPS_MODEL) is not None
        if self.entity_due((s, "__groups__"), now, per) or not have:
            self._fit_groups(ctx, s, now)

    def _fit_class(self, ctx: Context, s: str, ck: str, now: float) -> None:
        """model.density@(s, class:<rid>) on the members' committed rows (the
        most recent CLASS_ROWS / members per member)."""
        store = ctx.store
        mem = m_class.class_members(store, s, ck)
        if len(mem) < CLASS_MIN_MEMBERS:
            return
        per = max(N_FIT_MIN // 2, CLASS_ROWS // len(mem))
        Xs, ws, tss = [], [], []
        for ip in mem:
            st = _other_state(store, s, ip)
            if st is None or not st["ts"]:
                continue
            t_i, w_i, X_i = state_arrays({k: v[-per:] for k, v in st.items()})
            Xs.append(X_i)
            ws.append(w_i)
            tss.append(t_i)
        if not Xs or sum(x.shape[0] for x in Xs) < N_FIT_MIN:
            return
        X, w, ts = np.vstack(Xs), np.concatenate(ws), np.concatenate(tss)
        prev = store.get_model(s, ck, DENSITY)
        ver = int(prev.get("version", 0)) + 1 if isinstance(prev, dict) else 1
        cm = fit_model(X, w, ts=ts, crossfit=True, fitted_ts=now, version=ver)
        if cm is None:
            return
        cm["class_key"] = None
        cm["members"] = list(mem)
        store.put_model(s, ck, DENSITY, cm, version=ver, ts=now)
        _put_extra(store, s, ck, "mv_model", m_density.descriptor(cm))

    def _fit_groups(self, ctx: Context, s: str, now: float) -> None:
        """model.groups@(s, __system__) from the pooled committed rows, each
        entity centred by its column medians (level differences between
        entities are not dependence)."""
        store = ctx.store
        states = [st for st in (_other_state(store, s, e) for e in store.entities(s))
                  if st is not None and st["ts"]]
        if sum(min(len(st["ts"]), GROUP_ROWS_PER_ENT) for st in states) < GROUP_MIN_ROWS:
            return
        blocks = []
        for st in states:
            B = np.asarray(st["X"][-GROUP_ROWS_PER_ENT:], dtype=np.float64)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(B, axis=0)
            blocks.append(B - np.where(np.isfinite(med), med, 0.0))
        X = np.vstack(blocks)[-GROUP_ROWS_MAX:]
        groups, _ = dependence_groups(X)
        prev = store.get_model(s, SYSTEM_KEY, GROUPS_MODEL)
        if isinstance(prev, dict) and prev.get("groups") == groups:
            return
        ver = int(prev.get("version", 0)) + 1 if isinstance(prev, dict) else 1
        gof = [0] * FEATURE_DIM
        for j, g in enumerate(groups):
            for i in g:
                gof[i] = j
        store.put_model(s, SYSTEM_KEY, GROUPS_MODEL, {
            "fmt": 1, "version": ver, "ts": now, "n_rows": int(X.shape[0]),
            "groups": groups, "names": [[FEATURE_NAMES_V2[i] for i in g] for g in groups],
            "group_of": gof, "rho_min": RHO_MIN}, version=ver, ts=now)


# ================================================================ helpers
def _new_model(prev: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    m: Dict[str, Any] = {"fmt": m_density.FMT, "version": 0, "fitted": False, "n": 0.0}
    if prev is not None:
        for k in _CARRY:
            m[k] = prev[k]
        return m
    m.update({"_state": _init(), "_gate": G.GateState(), "_arch": OrderedDict(),
              "_oos_ts": [], "_oos_v": [], "_sig": None, "_front": -math.inf})
    return m


def _archive(arch: "OrderedDict[int, Tuple[float, np.ndarray]]", now: float,
             zi: np.ndarray) -> None:
    """Newest zi row per slot for ARCH_KEEP_S (release / rollback replays of
    rows the 6 h zi ring no longer holds)."""
    slot = int(math.floor(now / SLOT_S))
    arch[slot] = (now, np.asarray(zi, dtype=np.float32).copy())
    arch.move_to_end(slot)
    while arch:
        k = next(iter(arch))
        if arch[k][0] >= now - ARCH_KEEP_S:
            break
        del arch[k]


def _other_state(store: Any, s: str, e: str) -> Optional[Dict[str, list]]:
    m = store.get_model(s, e, DENSITY)
    if isinstance(m, dict) and m.get("fmt") == m_density.FMT and "_state" in m:
        return m["_state"]
    return None


def _active(store: Any, s: str, e: str, now: float) -> bool:
    a = store.vec_at(s, e, ACTIVE, now)
    return a is not None and a.size > 0 and float(a[0]) > 0.5


def _neglog10(p: float) -> Optional[float]:
    return -math.log10(max(p, 1e-300)) if p == p else None


def _fin(x: float) -> Optional[float]:
    return float(x) if math.isfinite(x) else None


def _js(x: float, nd: int = 6) -> Optional[float]:
    x = float(x)
    return float(f"{x:.{nd}g}") if math.isfinite(x) else None


def _put_extra(store: Any, s: str, e: str, key: str, value: Any) -> None:
    p = store.profile(s, e) or EntityProfile(system=s, entity=e)
    p.extra[key] = value
    store.put_profile(p)
