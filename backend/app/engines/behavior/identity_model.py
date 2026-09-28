"""IdentityModelEngine (B15): how identifiable is each IP and each class, what
distinguishes it, and whom is it confused with.

Why: a behaviour profile is only useful for "is this still the same user?"
if it can actually tell users apart. v1 (fingerprint.py) reported the cosine
distance of a median vector to its nearest neighbour, in-sample, on vectors
z-normalised per system: it could not say how often an entity would be
mistaken for another, and anything scored on behavior.z (normalised to the
entity itself) makes every entity look like itself. This engine measures
identifiability honestly - blocked cross-validation on ABSOLUTE
representations - and publishes the metric B16 (attribution) and B17
(linking) score windows in (lib/m_identity.py holds the shared maths).

What, per system:
  1. Windows (every tick, learning). Each entity's committed ACTIVE ticks are
     grouped into non-overlapping windows of K = 4 (lib/m_identity
     window_vector: median feature.vec, IQR of 8 key features, mean
     feature.sketch, behavior.timing B / M / think_mu, local-hour sin / cos
     and a workday flag). Collection goes through lib/gating.GatedLearner, so
     it is delayed by D, trust-weighted (ticks with trust < 0.5 are skipped),
     held while quarantined, checkpointed and reversible (model.control
     rollback_to / release / rebase_from / frozen; link seeding copies half of
     the linked entity's windows). Vectors live in model.idwin@(s, __system__),
     at most 3000 per system. Collection runs EVERY tick although the fit has
     a 96-tick stride: feature.vec and feature.sketch are retained 1 d, and a
     96 x 900-s stride plus D would read rows already pruned.
  2. Fit (every 96 ticks or 24 h, whichever comes first, with a per-system
     phase; the first fit as soon as >= 2 entities have >= 9 windows).
     NaN -> role-class median (else system median) + a missing mask;
     standardise; PCA to d <= 48 on the pooled windows; WCCN = inverse of the
     role-balanced average of OAS within-entity covariances (so a big role
     does not dictate the metric); LDA as a generalised eigenproblem against
     the OAS-shrunk within scatter (= sklearn 'eigen' with automatic
     shrinkage; the 'lsqr' solver has no transform, and B16 / B17 need an
     LDA space).
  3. Blocked CV: 3 contiguous folds per entity (by window time); the 2 windows
     on each side of the test block are left out of training, so temporally
     adjacent windows cannot leak. Per entity: recall@1, recall@K (K = 4
     ranks), the confusion row, T99 (99th percentile of held-out genuine d^2
     in LDA space), EER_hard = the WORST pairwise EER of its held-out genuine
     scores against the held-out windows of each of its 3 nearest impostors
     (Bhattacharyya distance in the WCCN space; pooling the three would dilute
     a twin into an EER of ~0.25), separability = clip(1 - 2 EER_hard, 0, 1),
     confusable_with (confusion >= 0.05 or pairwise EER >= 0.2) and anonymity
     sets (union-find over pairs with confusion > 0.2).
  4. Modality LLR calibration (a_m, b_m): prior-balanced logistic regression
     of genuine vs impostor window LLRs. gauss comes from the CV scores
     (l_j - l_bg in LDA space). vocab / rhythm / seq / client / timing come
     from live windows: up to 3 active entities at a time accumulate, over K
     active ticks, the LLR of their raw evidence (act.tokens, act.stream,
     client.stack_set, ...) under their own and their 3 nearest impostors'
     models through the owners' accessors (m_vocab, m_rhythm, m_seq, m_client,
     m_timing). The raw sets are retained 1 h, so they are scored the tick
     they are written, before any model has committed them (out-of-sample).
  5. Class identifiability: the same CV with role labels (model.class).
  6. Distinctive traits: Fisher scores of the window medians against role
     peers (reported in natural units) and Monroe-Colaresi-Quinn log-odds of
     the vocabulary (templates, never raw paths) against the peers.
  7. Every 4th fit, modality-drop importance: the CV is repeated with each of
     9 blocks removed; the EER_hard increase per block is the modality share.
Outputs: model.identity@(s, __system__), profile.separability and
profile.extra.identity per IP, class profiles' extra.identity
(identifiability) and INFO low_identifiability when EER_hard > 0.2 (once per
crossing; never in training). No p-values (B24 owns them).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg as sla

from ...core.engine import Context, Engine
from ...models.schema import BehaviorEvent, EntityProfile, Severity
from .lib import gating as G
from .lib import m_class
from .lib import grains as GR
from .lib import m_identity as MI
from .lib import m_seq
from .lib import m_vocab
from .lib import timebins as TB
from .lib.classkeys import SYSTEM_KEY, role_key
from .lib.features import FEATURE_NAMES_V2
from .lib.robustcov import eigen_floor, oas

LEARNER = "identity.win"
ACTIVE = "feature.active"
TCTX = "feature.tctx"

K_WIN = MI.K_WIN
MIN_WINDOWS = 9              # 3 folds x 3 windows: every fold keeps training windows
SYS_CAP = 3000               # windows per system (spec)
ENT_CAP = 1000               # window entries per entity state
W_MIN = 0.5                  # ticks with trust below this are not learned
D_PCA = 48
N_FOLDS = 3
GAP = 2                      # windows left out on each side of a test block
RECALL_K = 4
N_NEAR = 3
LOW_EER = 0.2                # low_identifiability
DROP_EVERY = 4               # modality-drop importance every 4th fit
CAL_OPEN_MAX = 3             # entities accumulating live modality windows at once
CAL_RING = 2000              # samples kept per modality
CAL_STALE_S = 86400.0
CAL_GAP_S = 4 * 3600.0       # an entity opens its next calibration window >= 4 h after its last
MAX_GAUSS_IMP = 20000
NG_MODS = tuple(m for m in MI.MODALITIES if m != "gauss")
EVENT_AXES = ["identity"]
_NAN = math.nan
_LN2PI = math.log(2.0 * math.pi)


# ============================================================ learner state
def new_state() -> Dict[str, Any]:
    """buf: committed active tick rows of the open window [[ts, *row]];
    wins: [[t0, t1, src]] ascending t1 (src '' = own, else the linked entity
    whose window vector it borrows); new: vectors completed since the engine
    last drained them (not checkpointed: they are already in model.idwin)."""
    return {"buf": [], "wins": [], "new": {}}


def wkey(t1: float) -> str:
    return f"{float(t1):.3f}"


def _update(state: Dict[str, Any], row: Tuple[float, List[float]], w: float) -> Dict[str, Any]:
    """Deterministic in (state, row, w): trusted active rows fill the window
    buffer; K rows make one window (vector computed from those rows only)."""
    if not w >= W_MIN:
        return state
    ts, r = row
    buf = state["buf"]
    buf.append([float(ts)] + list(r))
    if len(buf) < K_WIN:
        return state
    buf.sort(key=lambda x: x[0])              # released rows may arrive late
    R = np.asarray([b[1:] for b in buf], dtype=np.float64)
    t0, t1 = buf[0][0], buf[-1][0]
    state["new"][wkey(t1)] = MI.window_vector(R)
    wins = state["wins"]
    wins.append([t0, t1, ""])
    if len(wins) >= 2 and wins[-2][1] > t1:
        wins.sort(key=lambda x: x[1])
    if len(wins) > ENT_CAP:
        del wins[:len(wins) - ENT_CAP]
    state["buf"] = []
    return state


def _dump(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Checkpoint blob as arrays (the gate deep-copies blobs: a list of lists
    costs ~100x more than one array copy)."""
    buf = state.get("buf") or []
    wins = state.get("wins") or []
    return {"buf": (np.asarray(buf, dtype=np.float64) if buf else np.zeros((0, MI.TICK_DIM + 1))),
            "wt": np.asarray([w[:2] for w in wins], dtype=np.float64).reshape(len(wins), 2),
            "src": {i: w[2] for i, w in enumerate(wins) if w[2]}}


def _load(blob: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not blob:
        return new_state()
    if "wt" not in blob:                                   # plain-list layout
        return {"buf": [list(b) for b in blob.get("buf", [])],
                "wins": [list(w) for w in blob.get("wins", [])], "new": {}}
    src = {int(k): v for k, v in (blob.get("src") or {}).items()}
    wt = np.asarray(blob["wt"], dtype=np.float64).reshape(-1, 2)
    return {"buf": np.asarray(blob["buf"], dtype=np.float64).tolist(),
            "wins": [[float(t0), float(t1), src.get(i, "")] for i, (t0, t1) in enumerate(wt)],
            "new": {}}


def _merge(own: Dict[str, Any], other: Dict[str, Any], w: float) -> Dict[str, Any]:
    """Link seeding: B := B_own + w * A, i.e. every round(1/w)-th of A's
    windows (newest first) joins B's, still pointing at A's vectors."""
    src = str(other.get("_from", ""))
    step = max(1, int(round(1.0 / max(float(w), 1e-9))))
    have = {(x[2] or "", wkey(x[1])) for x in own["wins"]}
    for t0, t1, sr in list(reversed(other.get("wins", [])))[::step]:
        owner = sr or src
        if owner and (owner, wkey(t1)) not in have:
            own["wins"].append([t0, t1, owner])
            have.add((owner, wkey(t1)))
    own["wins"].sort(key=lambda x: x[1])
    if len(own["wins"]) > ENT_CAP:
        del own["wins"][:len(own["wins"]) - ENT_CAP]
    return own


def new_idwin() -> Dict[str, Any]:
    return {"fmt": MI.FMT, "version": 0, "ents": {},
            "sched": {"ticks": 0, "last_fit": None, "runs": 0, "low": {}},
            "cal": {"acc": {}, "ring": {m: [] for m in NG_MODS}, "last": {}}}


# ================================================================ the maths
def _sqdist(Z: np.ndarray, M: np.ndarray) -> np.ndarray:
    """||z_i - m_c||^2 via the Gram expansion (NaN rows of M give NaN)."""
    d2 = (np.sum(Z * Z, axis=1)[:, None] - 2.0 * (Z @ np.nan_to_num(M).T)
          + np.sum(M * M, axis=1)[None, :])
    return np.maximum(d2, 0.0, where=np.isfinite(d2), out=d2)


def _group_means(Y: np.ndarray, y: np.ndarray, n_lab: int, present: np.ndarray) -> np.ndarray:
    """[n_lab, d] label means (NaN for labels not in `present`), one matmul."""
    O = np.zeros((n_lab, Y.shape[0]))
    O[y, np.arange(Y.shape[0])] = 1.0
    cnt = O.sum(axis=1)
    M = np.full((n_lab, Y.shape[1]), np.nan)
    M[present] = (O[present] @ Y) / cnt[present, None]
    return M


class Prep:
    """Imputation + mask + standardisation + PCA fitted on pooled windows."""

    def __init__(self, X: np.ndarray, cls_of_row: Sequence[Optional[str]], d_max: int = D_PCA):
        n = X.shape[0]
        fin = np.isfinite(X)
        with np.errstate(all="ignore"):
            fill = MI._nanmedian(np.where(fin, X, np.nan))
        fill = np.where(np.isfinite(fill), fill, 0.0)
        fill_cls: Dict[str, List[float]] = {}
        cls_arr = np.asarray([c if c is not None else "" for c in cls_of_row], dtype=object)
        for ck in sorted({c for c in cls_of_row if c}):
            sel = cls_arr == ck
            f = MI._nanmedian(X[sel])
            fill_cls[ck] = np.where(np.isfinite(f), f, fill).tolist()
        miss_frac = 1.0 - fin[:, :MI.VEC_DIM].mean(axis=0)
        mask_cols = [int(c) for c in np.flatnonzero((miss_frac > 0.0) & (miss_frac < 1.0))]
        self.pca: Dict[str, Any] = {"fill": fill.tolist(), "fill_cls": fill_cls,
                                    "mask_cols": mask_cols, "keep": None}
        A = MI.augment({"pca": self.pca}, X, list(cls_of_row))
        center = A.mean(axis=0)
        scale = A.std(axis=0)
        keep = np.flatnonzero(scale > 1e-8 * np.maximum(1.0, np.abs(center)))
        self.pca["keep"] = keep.tolist()
        self.center, self.scale = center[keep], scale[keep]
        U = (A[:, keep] - self.center) / self.scale if keep.size else np.zeros((n, 0))
        self.U = U
        d = 0
        if keep.size and n >= 3:
            if n < keep.size:                      # Gram trick: eigh of the smaller side
                lam, Q = np.linalg.eigh(U @ U.T / n)
                lam, Q = lam[::-1], Q[:, ::-1]
                d = int(min(d_max, n - 1, np.sum(lam > 1e-10 * max(lam[0], 1e-300))))
                self.V = (U.T @ Q[:, :d]) / np.sqrt(n * lam[:d])
            else:
                lam, V = np.linalg.eigh(U.T @ U / n)
                lam, V = lam[::-1], V[:, ::-1]
                d = int(min(d_max, n - 1, np.sum(lam > 1e-10 * max(lam[0], 1e-300))))
                self.V = V[:, :d]
        else:
            self.V = np.zeros((keep.size, 0))
        self.d = d
        self.Y = U @ self.V


class Metric:
    """WCCN + shrinkage LDA fitted on training rows of the PCA scores."""

    def __init__(self, Y: np.ndarray, y: np.ndarray, n_lab: int, role_of_lab: Sequence[Any]):
        d = Y.shape[1]
        cnt = np.bincount(y, minlength=n_lab)
        self.present = np.flatnonzero(cnt >= 2)
        self.ok = self.present.size >= 2 and d >= 1
        if not self.ok:
            return
        sel = np.isin(y, self.present)
        Y, y = Y[sel], y[sel]
        means = _group_means(Y, y, n_lab, self.present)
        R = Y - means[y]
        # WCCN: role-balanced average of OAS within-entity covariances
        rnames = sorted({str(g) for g in role_of_lab})
        ridx = np.asarray([rnames.index(str(g)) for g in role_of_lab])[y]
        covs = []
        for g in range(len(rnames)):
            Rg = R[ridx == g]
            if Rg.shape[0] >= 3:
                covs.append(oas(Rg)[1])
        Sw = eigen_floor(np.mean(covs, axis=0) if covs else np.eye(d), 1e-3)
        lam, V = np.linalg.eigh(Sw)
        self.A = (V / np.sqrt(np.maximum(lam, 1e-12))) @ V.T
        Rw = R @ self.A
        mw = means[self.present] @ self.A
        # LDA against the OAS-shrunk within scatter (automatic shrinkage)
        Sw2 = eigen_floor(oas(Rw)[1], 1e-3)
        n_c = cnt[self.present].astype(np.float64)
        mbar = (n_c[:, None] * mw).sum(axis=0) / n_c.sum()
        D = mw - mbar
        Sb = (D * n_c[:, None]).T @ D / n_c.sum()
        vals, vecs = sla.eigh(0.5 * (Sb + Sb.T), 0.5 * (Sw2 + Sw2.T))
        order = np.argsort(vals)[::-1]
        vmax = max(float(vals[order[0]]), 1e-300)
        r = int(min(self.present.size - 1, d, max(1, np.sum(vals > 1e-10 * vmax))))
        self.W = vecs[:, order[:r]]
        self.P = self.A @ self.W
        self.r = r
        Z = Y @ self.P
        self.means = _group_means(Z, y, n_lab, self.present)
        mu = Z.mean(axis=0)
        cov = np.atleast_2d(np.cov(Z.T, ddof=1)) + 1e-6 * np.eye(r)
        self.bg_mu = mu
        self.bg_prec = np.linalg.inv(cov)
        self.bg_logdet = float(np.linalg.slogdet(cov)[1])

    def loglik(self, Yt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(ll [n, L] (NaN for labels absent from training), d2 [n, L], ll_bg [n])."""
        Z = Yt @ self.P
        d2 = _sqdist(Z, self.means)
        ll = -0.5 * (d2 + self.r * _LN2PI)
        dz = Z - self.bg_mu
        q = np.sum((dz @ self.bg_prec) * dz, axis=1)
        bg = -0.5 * (q + self.bg_logdet + self.r * _LN2PI)
        return ll, d2, bg


def fold_plan(pos: np.ndarray, n_of: np.ndarray, lab: np.ndarray
              ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Per-row fold (3 contiguous blocks per label sequence) and the training
    mask of each fold (other blocks, minus GAP windows on each side)."""
    n = n_of[lab]
    fold = np.minimum((N_FOLDS * pos) // np.maximum(n, 1), N_FOLDS - 1)
    trains = []
    for f in range(N_FOLDS):
        lo = np.ceil(f * n / N_FOLDS).astype(int)          # first pos of block f
        hi = np.ceil((f + 1) * n / N_FOLDS).astype(int) - 1
        trains.append((fold != f) & ((pos < lo - GAP) | (pos > hi + GAP)))
    return fold, trains


def cross_validate(Y: np.ndarray, lab: np.ndarray, n_lab: int, role_of_lab: Sequence[Any],
                   fold: np.ndarray, trains: Sequence[np.ndarray]
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Held-out (ll [n, L], d2 [n, L], ll_bg [n]); NaN where not scorable."""
    n = Y.shape[0]
    LL = np.full((n, n_lab), np.nan)
    D2 = np.full((n, n_lab), np.nan)
    BG = np.full(n, np.nan)
    for f in range(N_FOLDS):
        test = fold == f
        tr = trains[f]
        if not test.any() or not tr.any():
            continue
        m = Metric(Y[tr], lab[tr], n_lab, role_of_lab)
        if not m.ok:
            continue
        ll, d2, bg = m.loglik(Y[test])
        LL[test], D2[test], BG[test] = ll, d2, bg
    return LL, D2, BG


def label_stats(LL: np.ndarray, D2: np.ndarray, lab: np.ndarray, n_lab: int,
                near: Sequence[Sequence[int]]) -> List[Dict[str, Any]]:
    """Per label: recall@1 / @K, confusion row, T99, pairwise and hard EER."""
    out: List[Dict[str, Any]] = []
    for c in range(n_lab):
        rows = np.flatnonzero((lab == c) & np.isfinite(LL[:, c]))
        st: Dict[str, Any] = {"n_test": int(rows.size)}
        if not rows.size:
            out.append(st)
            continue
        L = LL[rows]
        Lf = np.where(np.isfinite(L), L, -np.inf)
        pred = np.argmax(Lf, axis=1)
        rank = np.sum(Lf > Lf[:, c:c + 1], axis=1)
        st["recall1"] = float(np.mean(pred == c))
        st["recallK"] = float(np.mean(rank < RECALL_K))
        st["confusion"] = {int(j): float(np.mean(pred == j)) for j in np.unique(pred) if j != c}
        st["t99"] = float(np.percentile(D2[rows, c], 99))
        gen = LL[rows, c]
        pair = {}
        for j in near[c]:
            imp = LL[(lab == j), c]
            e = MI.eer(gen, imp[np.isfinite(imp)])
            if math.isfinite(e):
                pair[int(j)] = e
        st["eer_pair"] = pair
        st["eer_hard"] = max(pair.values()) if pair else _NAN
        out.append(st)
    return out


def nearest_labels(Yw: np.ndarray, lab: np.ndarray, n_lab: int, k: int = N_NEAR
                   ) -> List[List[int]]:
    """k nearest other labels by Bhattacharyya distance of diagonal Gaussians."""
    mu, var = [], []
    for c in range(n_lab):
        Z = Yw[lab == c]
        mu.append(Z.mean(axis=0) if Z.shape[0] else np.full(Yw.shape[1], np.nan))
        var.append(Z.var(axis=0) + 1e-3 if Z.shape[0] >= 2 else np.ones(Yw.shape[1]))
    out = []
    for c in range(n_lab):
        d = [(MI.bhattacharyya_diag(mu[c], var[c], mu[j], var[j]), j)
             for j in range(n_lab) if j != c and np.all(np.isfinite(mu[j]))]
        d.sort()
        out.append([j for _, j in d[:k]])
    return out


# =================================================================== engine
class IdentityModelEngine(Engine):
    name = "behavior.identity_model"
    layer = "behavior"
    consumes = ["feature.vec", "feature.sketch", "feature.tctx", "feature.active",
                "behavior.trust", "behavior.trust_prov", "behavior.quarantine",
                "behavior.timing", "act.tokens", "act.stream", "act.stream_frac",
                "client.stack_set", "tls.sni_etld1_set", "dns.qname_etld1_set",
                "l4.dport_set", "model.vocab", "model.rhythm", "model.seq", "model.client",
                "model.timing", "model.class", "model.control", "model.link"]
    produces = ["model.identity", "model.idwin", "profile.separability",
                "profile.extra.identity", "event.low_identifiability"]
    description = ("Identifiability per IP and class: K=4 absolute windows, PCA-48 + WCCN + "
                   "shrinkage LDA, blocked 3-fold CV (recall@1/K, EER_hard vs 3 nearest "
                   "impostors, T99, confusion, anonymity sets), modality LLR calibration, "
                   "distinctive traits; trust-gated reversible window collection.")
    interval = 1                 # collection; the fit has its own 96-tick / 24-h stride
    period_s = None

    def __init__(self, fit_ticks: int = 96, fit_period_s: float = 86400.0,
                 min_windows: int = MIN_WINDOWS, cal_open_max: int = CAL_OPEN_MAX,
                 cal_gap_s: float = CAL_GAP_S, **params: Any) -> None:
        super().__init__(**params)
        self.fit_ticks = int(fit_ticks)
        self.fit_period_s = float(fit_period_s)
        self.min_windows = max(MIN_WINDOWS, int(min_windows))
        self.cal_open_max = int(cal_open_max)
        self.cal_gap_s = float(cal_gap_s)
        self._now, self._dt, self._config = 0.0, 900.0, {}
        self._learner_tick = G.GatedLearner(name=LEARNER, init=new_state, update=_update,
                                            fetch=self._fetch, dump=_dump, load=_load,
                                            merge=_merge)
        # spec v2.1: windows of 4 committed active H rows (clock feature.meta.h)
        self._learner_h = G.GatedLearner(name=LEARNER, init=new_state, update=_update,
                                         fetch=self._fetch_h, dump=_dump, load=_load,
                                         merge=_merge, clock="feature.meta.h", window_s=3600.0)
        self._learner = self._learner_tick
        self._canon = False

    # ------------------------------------------------------------------ run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        self._now, self._dt, self._config = float(ctx.now), float(ctx.window_s), ctx.config
        self._canon = GR.canonical(ctx.config)
        self._learner = self._learner_h if self._canon else self._learner_tick
        self._learner.d_min_s = float(ctx.config.get("D_min_s") or G.D_MIN_S)
        if self._canon and not GR.decision(self._now, self._dt, "h", GR.CANONICAL):
            return 0                     # spec v2.1: collection / calibration on H ticks
        return sum(self._system(ctx, s) for s in store.systems())

    def refit(self, ctx: Context, system: Optional[str] = None) -> int:
        """Run the CV fit now, outside the stride (ops after a relabel, tests).
        Returns the number of entities scored."""
        store, now = ctx.store, float(ctx.now)
        self._now, self._dt, self._config = now, float(ctx.window_s), ctx.config
        n = 0
        for s in ([system] if system is not None else store.systems()):
            idw = store.get_model(s, SYSTEM_KEY, MI.IDWIN)
            if not (isinstance(idw, dict) and idw.get("fmt") == MI.FMT):
                continue
            k = self._fit(ctx, s, idw)
            if k:
                sched = idw["sched"]
                sched["ticks"], sched["last_fit"] = 0, now
                sched["runs"] += 1
                idw["version"] += 1
                store.put_model(s, SYSTEM_KEY, MI.IDWIN, idw, version=idw["version"], ts=now)
            n += k
        return n

    def _system(self, ctx: Context, s: str) -> int:
        store, now = ctx.store, float(ctx.now)
        idw = store.get_model(s, SYSTEM_KEY, MI.IDWIN)
        if not (isinstance(idw, dict) and idw.get("fmt") == MI.FMT):
            idw = new_idwin()
        ents = store.entities(s)
        if not ents and not idw["ents"]:
            return 0
        live = set(ents)
        for e in [e for e in idw["ents"] if e not in live]:
            del idw["ents"][e]                     # entity gone from the store
        added = 0
        for e in ents:
            added += self._collect(ctx, s, e, idw)
        if added:
            self._enforce_cap(idw)
            idw["version"] += 1
        self._calibrate_live(ctx, s, idw, ents)
        n = 0
        sched = idw["sched"]
        sched["ticks"] += 1
        by_clock = self.entity_due((s, "identity_fit"), now, self.fit_period_s)
        by_ticks = sched["ticks"] >= self.fit_ticks and not self._canon   # v2.1: wall clock only
        if sched["last_fit"] is None or by_ticks or by_clock:
            n = self._fit(ctx, s, idw)
            if n:
                sched["ticks"], sched["last_fit"] = 0, now
                sched["runs"] += 1
                idw["version"] += 1
        store.put_model(s, SYSTEM_KEY, MI.IDWIN, idw, version=idw["version"], ts=now)
        return n

    # ----------------------------------------------------------- collection
    def _fetch(self, store, s: str, e: str, ts: float) -> Optional[Tuple[float, List[float]]]:
        a = store.vec_at(s, e, ACTIVE, ts)
        if a is None or not len(a) or not float(a[0]) > 0.5:
            return None
        depth = int(min(20000, max(4, (self._now - ts) / max(self._dt, 1.0) + 4)))
        tc = MI._dict_at(store, s, e, TCTX, ts, depth)
        if tc is None:
            tc = TB.tctx_from_config(ts, self._config, self._dt)
        row = MI.tick_row(store, s, e, ts, tctx=tc, depth=depth)
        return None if row is None else (float(ts), row.tolist())

    def _fetch_h(self, store, s: str, e: str, ts: float) -> Optional[Tuple[float, List[float]]]:
        """spec v2.1: an active H row (m_identity.grain_row)."""
        a = store.vec_at(s, e, "feature.meta.h", ts)
        if a is None or not len(a) or not float(a[0]) > 0.5:
            return None
        row = MI.grain_row(store, s, e, ts, config=self._config, dt=self._dt)
        return None if row is None else (float(ts), row.tolist())

    def _collect(self, ctx: Context, s: str, e: str, idw: Dict[str, Any]) -> int:
        store = ctx.store
        rec = idw["ents"].get(e)
        if rec is None:
            if store.vec_latest(s, e, "feature.meta.h" if self._canon else ACTIVE) is None:
                return 0                           # no feature rows yet
            rec = idw["ents"][e] = {"state": new_state(), "gate": G.GateState(), "vecs": {}}
        gate = G.GateState.from_dict(rec["gate"])

        def load_other(src: str) -> Optional[Dict[str, Any]]:
            r = idw["ents"].get(src)
            if r is None:
                return None
            st = _load(_dump(r["state"]))
            st["_from"] = src
            return st

        state, gate = self._learner.seed_from_link(store, s, e, rec["state"], gate, load_other)
        state, gate = self._learner.step(store, s, e, state, gate, float(ctx.now),
                                         float(ctx.window_s), training=bool(ctx.training))
        new = state.get("new") or {}
        vecs = rec["vecs"]
        for k, v in new.items():
            vecs[k] = np.asarray(v, dtype=np.float32)
        state["new"] = {}
        if len(vecs) > sum(1 for w in state["wins"] if not w[2]):
            own = {wkey(w[1]) for w in state["wins"] if not w[2]}
            for k in [k for k in vecs if k not in own]:
                del vecs[k]                        # rolled back / trimmed
        rec["state"], rec["gate"] = state, gate
        return len(new)

    def _enforce_cap(self, idw: Dict[str, Any]) -> None:
        """At most SYS_CAP window vectors per system: trim the oldest windows of
        the entities above the fair share first."""
        ents = idw["ents"]
        total = sum(len(r["vecs"]) for r in ents.values())
        if total <= SYS_CAP:
            return
        sizes = {e: len(r["vecs"]) for e, r in ents.items() if r["vecs"]}
        # water-filling: small entities keep all, the rest share what is left
        order = sorted(sizes, key=lambda e: sizes[e])
        left, n_left = SYS_CAP, len(order)
        for e in order:
            quota = left // max(1, n_left)
            keep = min(sizes[e], quota)
            if keep < sizes[e]:
                r = ents[e]
                own = [w for w in r["state"]["wins"] if not w[2]]
                drop = {wkey(w[1]) for w in own[:len(own) - keep]}
                for k in drop:
                    r["vecs"].pop(k, None)
                r["state"]["wins"] = [w for w in r["state"]["wins"]
                                      if w[2] or wkey(w[1]) not in drop]
            left -= keep
            n_left -= 1

    # ------------------------------------------------------ live calibration
    def _calibrate_live(self, ctx: Context, s: str, idw: Dict[str, Any], ents: List[str]) -> None:
        """Window LLR samples of the non-gauss modalities (see module doc 4)."""
        store, now, dt = ctx.store, float(ctx.now), float(ctx.window_s)
        cal = idw["cal"]
        acc, last = cal["acc"], cal["last"]
        live = set(ents)
        for e in [e for e, a in acc.items() if e not in live or now - a["t0"] > CAL_STALE_S]:
            del acc[e]
        active = [e for e in ents if _active_now(store, s, e, now,
                                                 "feature.meta.h" if self._canon else ACTIVE)]
        if not active:
            return
        model = MI.get(store, s)
        free = self.cal_open_max - len(acc)
        if free > 0:
            # stride (integration perf): the per-modality LLR calibration moves
            # slowly and is refitted daily, so an entity contributes at most one
            # window per cal_gap_s (~6 windows / day each, MIN_CAL reached in
            # hours) instead of keeping 3 entities sampling PPM / vocab LLRs
            # under 4 candidates on every tick
            waiting = sorted((e for e in active if e not in acc
                              and not now - last.get(e, -math.inf) < self.cal_gap_s),
                             key=lambda e: (last.get(e, -math.inf), e))
            for e in waiting[:free]:
                imps = self._impostors(model, e, ents)
                if imps:
                    acc[e] = {"c": [e] + imps, "t0": now, "k": 0,
                              "s": [[0.0] * len(NG_MODS) for _ in range(len(imps) + 1)],
                              "n": [[0] * len(NG_MODS) for _ in range(len(imps) + 1)]}
        todo = [e for e in active if e in acc and not G.is_quarantined(store, s, e, now, dt)]
        if not todo:
            return
        bg = MI.Background(store, s, now)
        smap = m_seq.SymbolMap.from_store(store, s)
        ring = cal["ring"]
        for e in todo:
            a = acc[e]
            if self._canon:
                data = MI.grain_modal_data(store, s, e, now, smap=smap, config=self._config,
                                           dt=dt)
            else:
                tc = MI._dict_at(store, s, e, TCTX, now, 2)
                data = MI.tick_modal_data(store, s, e, now, tctx=tc, smap=smap)
            llrs = MI.modality_llrs(store, s, a["c"], data, now, bg)
            for ci, c in enumerate(a["c"]):
                for mi, m in enumerate(NG_MODS):
                    v = llrs[c][m]
                    if math.isfinite(v):
                        a["s"][ci][mi] += v
                        a["n"][ci][mi] += 1
            a["k"] += 1
            if a["k"] < K_WIN:
                continue
            for ci, c in enumerate(a["c"]):
                for mi, m in enumerate(NG_MODS):
                    if a["n"][ci][mi]:
                        ring.setdefault(m, []).append([a["s"][ci][mi], 1 if c == e else 0, now])
            for m in NG_MODS:
                r = ring.get(m) or []
                if len(r) > CAL_RING:
                    del r[:len(r) - CAL_RING]
            del acc[e]
            last[e] = now

    @staticmethod
    def _impostors(model: Optional[Mapping], e: str, ents: Sequence[str]) -> List[str]:
        """3 nearest impostors from the last fit, else 3 deterministic others."""
        near = [j for j in MI.nearest(model, e) if j in set(ents)] if model else []
        if len(near) >= N_NEAR:
            return near[:N_NEAR]
        others = [j for j in sorted(ents) if j != e and j not in near]
        if others:
            k = sum(map(ord, e)) % len(others)
            others = others[k:] + others[:k]
        return (near + others)[:N_NEAR]

    # ------------------------------------------------------------------ fit
    def _gather(self, idw: Dict[str, Any]) -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
        ents = idw["ents"]
        names, Xs, T = [], [], []
        for e in sorted(ents):
            rows, ts = [], []
            for t0, t1, src in ents[e]["state"]["wins"]:
                owner = ents.get(src or e)
                v = owner["vecs"].get(wkey(t1)) if owner is not None else None
                if v is not None:
                    rows.append(v)
                    ts.append(t1)
            if len(rows) >= self.min_windows:
                names.append(e)
                Xs.append(np.asarray(rows, dtype=np.float64))
                T.append(np.asarray(ts, dtype=np.float64))
        return names, Xs, T

    def _fit(self, ctx: Context, s: str, idw: Dict[str, Any]) -> int:
        store, now = ctx.store, float(ctx.now)
        names, Xs, _T = self._gather(idw)
        if len(names) < 2:
            return 0
        C = len(names)
        X = np.vstack(Xs)
        lab = np.concatenate([np.full(x.shape[0], i) for i, x in enumerate(Xs)])
        pos = np.concatenate([np.arange(x.shape[0]) for x in Xs])
        n_of = np.asarray([x.shape[0] for x in Xs])
        rid = {e: m_class.role_id(store, s, e) for e in names}
        ck_of = {e: (role_key(rid[e]) if rid[e] is not None else None) for e in names}
        role_of_lab = [rid[e] if rid[e] is not None else "_" for e in names]
        prep = Prep(X, [ck_of[names[i]] for i in lab])
        if prep.d < 1:
            return 0
        Y = prep.Y
        fold, trains = fold_plan(pos, n_of, lab)

        full = Metric(Y, lab, C, role_of_lab)
        if not full.ok:
            return 0
        near = nearest_labels(Y @ full.A, lab, C)
        LL, D2, BG = cross_validate(Y, lab, C, role_of_lab, fold, trains)
        st = label_stats(LL, D2, lab, C, near)

        runs = int(idw["sched"]["runs"])
        prev = MI.get(store, s)
        share = dict((prev or {}).get("modality_share") or {})
        share_ts = (prev or {}).get("modality_share_ts")
        if runs % DROP_EVERY == 0:
            share = self._drop_importance(prep, lab, fold, trains, C, role_of_lab, near, st,
                                          names)
            share_ts = now

        # ------------------------------------------------ per-entity summary
        conf = {names[c]: {names[j]: v for j, v in (st[c].get("confusion") or {}).items()}
                for c in range(C) if st[c].get("n_test")}
        pairs = [(a, b) for a, row in conf.items() for b, v in row.items() if v > MI.ANON_CONFUSION]
        anon = MI.union_find_sets(names, pairs)
        stats: Dict[str, Dict[str, Any]] = {}
        for c, e in enumerate(names):
            sc = st[c]
            eh = _fin(sc.get("eer_hard"))
            row = conf.get(e, {})
            pe = {names[j]: v for j, v in (sc.get("eer_pair") or {}).items()}
            cw = {j for j, v in row.items() if v >= MI.CONFUSABLE_MIN}
            cw |= {j for j, v in pe.items() if v >= MI.PAIR_EER_CONFUSABLE}
            cw_l = sorted(cw, key=lambda j: (-row.get(j, 0.0), -pe.get(j, 0.0), j))
            stats[e] = {
                "recall1": _r(sc.get("recall1")), "recallK": _r(sc.get("recallK")),
                "K": RECALL_K, "eer_hard": _r(eh), "eer_pair": {j: _r(v) for j, v in pe.items()},
                "t99": _r(sc.get("t99")),
                "separability": _r(min(1.0, max(0.0, 1.0 - 2.0 * eh))) if math.isfinite(eh) else None,
                "near": [names[j] for j in near[c]], "n_windows": int(n_of[c]),
                "confusable_with": cw_l, "role": rid[e],
            }

        # ------------------------------------------------ calibration
        calib, cal_n = self._fit_calibration(LL, BG, lab, idw)

        # ------------------------------------------------ class identifiability
        classes = self._class_identifiability(Y, lab, names, rid, fold, trains, full.A)

        # ------------------------------------------------ full-data model
        P = prep.V @ full.P
        Zf = Y @ full.P
        class_means, class_var = {}, {}
        for ck in sorted({v for v in ck_of.values() if v}):
            sel = np.isin(lab, [i for i, e in enumerate(names) if ck_of[e] == ck])
            class_means[ck] = Zf[sel].mean(axis=0).tolist()
            class_var[ck] = (Zf[sel].var(axis=0) + 1e-3).tolist()
        model = {
            "fmt": MI.FMT, "version": MI.version(prev) + 1, "fitted_ts": now, "run": runs,
            "entities": list(names), "roles": dict(rid),
            "pca": {"fill": prep.pca["fill"], "fill_cls": prep.pca["fill_cls"],
                    "mask_cols": prep.pca["mask_cols"], "keep": prep.pca["keep"],
                    "center": prep.center.tolist(), "scale": prep.scale.tolist(),
                    "P": P.tolist(), "d_pca": int(prep.d), "r": int(full.r)},
            "W": full.W.tolist(),
            "means": {e: full.means[c].tolist() for c, e in enumerate(names)},
            "class_means": class_means, "class_var": class_var,
            "bg": {"mu": full.bg_mu.tolist(), "prec": full.bg_prec.tolist(),
                   "logdet": full.bg_logdet},
            "llr_calib": calib, "llr_n": cal_n,
            "confusion": conf, "anonymity_sets": anon, "stats": stats, "classes": classes,
            "distinctive": self._distinctive(store, s, X, lab, names, rid, now),
            "modality_share": share, "modality_share_ts": share_ts,
            "n_windows": int(X.shape[0]),
        }
        store.put_model(s, SYSTEM_KEY, MI.MODEL, model, version=model["version"], ts=now)
        self._write_profiles(ctx, s, model, names, idw)
        return len(names)

    def _fit_calibration(self, LL: np.ndarray, BG: np.ndarray, lab: np.ndarray,
                         idw: Mapping[str, Any]) -> Tuple[Dict[str, List[float]], Dict[str, List[int]]]:
        n, C = LL.shape
        G_ = LL - BG[:, None]
        gen = G_[np.arange(n), lab]
        mask = np.ones_like(G_, dtype=bool)
        mask[np.arange(n), lab] = False
        imp = G_[mask]
        imp = imp[np.isfinite(imp)]
        if imp.size > MAX_GAUSS_IMP:
            imp = imp[:: int(math.ceil(imp.size / MAX_GAUSS_IMP))]
        gen = gen[np.isfinite(gen)]
        x = np.concatenate([gen, imp])
        y = np.concatenate([np.ones(gen.size), np.zeros(imp.size)])
        calib = {"gauss": list(MI.logistic_calibration(x, y))}
        counts = {"gauss": [int(gen.size), int(imp.size)]}
        ring = idw["cal"]["ring"]
        for m in NG_MODS:
            r = np.asarray(ring.get(m) or np.zeros((0, 3)), dtype=np.float64).reshape(-1, 3)
            calib[m] = list(MI.logistic_calibration(r[:, 0], r[:, 1]))
            counts[m] = [int(np.sum(r[:, 1] > 0.5)), int(np.sum(r[:, 1] <= 0.5))]
        return calib, counts

    def _class_identifiability(self, Y: np.ndarray, lab: np.ndarray, names: List[str],
                               rid: Mapping[str, Optional[str]], fold: np.ndarray,
                               trains: Sequence[np.ndarray], A: np.ndarray) -> Dict[str, Any]:
        roles = sorted({r for r in rid.values() if r is not None})
        if len(roles) < 2:
            return {}
        ridx = {r: i for i, r in enumerate(roles)}
        ent_role = np.asarray([ridx.get(rid[e], -1) if rid[e] is not None else -1
                               for e in names])
        rl = ent_role[lab]
        sel = rl >= 0
        if not sel.any():
            return {}
        Ys, ls = Y[sel], rl[sel]
        R = len(roles)
        folds_s, trains_s = fold[sel], [t[sel] for t in trains]
        near = nearest_labels(Ys @ A, ls, R)
        LL, D2, _bg = cross_validate(Ys, ls, R, list(range(R)), folds_s, trains_s)
        st = label_stats(LL, D2, ls, R, near)
        out = {}
        for i, r in enumerate(roles):
            sc = st[i]
            if not sc.get("n_test"):
                continue
            eh = _fin(sc.get("eer_hard"))
            sep = min(1.0, max(0.0, 1.0 - 2.0 * eh)) if math.isfinite(eh) else None
            out[role_key(r)] = {
                "recall1": _r(sc.get("recall1")), "eer_hard": _r(eh), "separability": _r(sep),
                "identifiability": _r(sep),
                "confusable_with": [role_key(roles[j]) for j, v in
                                    sorted((sc.get("confusion") or {}).items(), key=lambda t: -t[1])
                                    if v >= MI.CONFUSABLE_MIN],
                "n_members": int(np.sum(ent_role == i)), "n_windows": int(np.sum(ls == i)),
            }
        return out

    def _drop_importance(self, prep: Prep, lab: np.ndarray, fold: np.ndarray,
                         trains: Sequence[np.ndarray], C: int, role_of_lab: Sequence[Any],
                         near: Sequence[Sequence[int]], st: Sequence[Mapping[str, Any]],
                         names: List[str]) -> Dict[str, Dict[str, float]]:
        """Modality share: EER_hard increase per entity when a block is removed.
        The block's standardised columns (and their missing masks) are zeroed
        before projecting on the pooled PCA basis, so the refitted WCCN / LDA
        of every fold sees no trace of it; the basis itself is not refitted
        (9 eigendecompositions saved, the metric is refitted anyway)."""
        base = np.asarray([_fin(sc.get("eer_hard")) for sc in st])
        delta = np.zeros((C, len(MI.BLOCKS)))
        keep = prep.pca["keep"]
        mcols = prep.pca["mask_cols"]
        for b, cols in enumerate(MI.BLOCKS.values()):
            aug = set(cols) | {MI.RAW_DIM + k for k, c in enumerate(mcols) if c in cols}
            ucols = [i for i, a in enumerate(keep) if a in aug]
            if not ucols:
                continue
            Ub = prep.U.copy()
            Ub[:, ucols] = 0.0
            LL, D2, _ = cross_validate(Ub @ prep.V, lab, C, role_of_lab, fold, trains)
            stb = label_stats(LL, D2, lab, C, near)
            eb = np.asarray([_fin(sc.get("eer_hard")) for sc in stb])
            d = eb - base
            delta[:, b] = np.where(np.isfinite(d), np.maximum(d, 0.0), 0.0)
        out = {}
        keys = list(MI.BLOCKS)
        for c, e in enumerate(names):
            tot = float(delta[c].sum())
            out[e] = {k: _r(float(delta[c, b]) / tot) for b, k in enumerate(keys)} if tot > 0 \
                else {k: 0.0 for k in keys}
        return out

    def _distinctive(self, store, s: str, X: np.ndarray, lab: np.ndarray, names: List[str],
                     rid: Mapping[str, Optional[str]], now: float) -> Dict[str, Any]:
        """Fisher scores of window medians vs role peers (natural units) and
        MCQ log-odds of the vocabulary vs the peers' vocabularies."""
        med = X[:, MI.W_MED]
        out: Dict[str, Any] = {}
        sysv = m_vocab.get(store, s, SYSTEM_KEY)
        vocab = {e: m_vocab.get(store, s, e) for e in names}
        for c, e in enumerate(names):
            peers = [j for j, f in enumerate(names) if j != c and rid[f] is not None
                     and rid[f] == rid[e]]
            if not peers:
                peers = [j for j in range(len(names)) if j != c]
            me = med[lab == c]
            mp = med[np.isin(lab, peers)]
            (ma, va, na), (mb, vb, nb) = _nstats(me), _nstats(mp)
            fs = (ma - mb) ** 2 / (va + vb + 1e-6)
            ok = (na >= 3) & (nb >= 3) & np.isfinite(fs) & (fs >= 0.5)
            qa = MI._col_quantiles(me, (0.5,))[0]
            qb = MI._col_quantiles(mp, (0.5,))[0]
            feats = sorted(((float(fs[f]), int(f), float(qa[f]), float(qb[f]),
                             bool(ma[f] > mb[f])) for f in np.flatnonzero(ok)),
                           key=lambda t: -t[0])
            ftr = [{"feature": FEATURE_NAMES_V2[f], "fisher": _r(x),
                    "self": _r(MI.to_natural(f, a)), "peers": _r(MI.to_natural(f, b)),
                    "dir": "higher" if up else "lower"} for x, f, a, b, up in feats[:5]]
            voc = []
            for dim in ("tmpl", "sni", "dns", "dport"):
                ys = m_vocab.counts(vocab[e], dim, now)
                if not ys:
                    continue
                yp: Dict[str, float] = {}
                for j in peers:
                    for v, n in m_vocab.counts(vocab[names[j]], dim, now).items():
                        yp[v] = yp.get(v, 0.0) + n
                if not yp:
                    continue
                prior = m_vocab.counts(sysv, dim, now) or {k: ys.get(k, 0.0) + yp.get(k, 0.0)
                                                          for k in set(ys) | set(yp)}
                ts, tp = sum(ys.values()), sum(yp.values())
                for v, (dl, z) in MI.mcq_log_odds(ys, yp, prior).items():
                    if z >= 1.96:
                        voc.append({"dim": dim, "value": v, "z": _r(z), "log_odds": _r(dl),
                                    "share_self": _r(ys.get(v, 0.0) / ts),
                                    "share_peers": _r(yp.get(v, 0.0) / tp)})
            voc.sort(key=lambda d: -d["z"])
            out[e] = {"features": ftr, "vocab": voc[:5]}
        return out

    # -------------------------------------------------------------- outputs
    def _write_profiles(self, ctx: Context, s: str, model: Mapping[str, Any], names: List[str],
                        idw: Dict[str, Any]) -> None:
        store, now = ctx.store, float(ctx.now)
        low = idw["sched"].setdefault("low", {})
        for e in names:
            desc = MI.descriptors(model, e)
            p = store.profile(s, e) or EntityProfile(system=s, entity=e)
            sep = desc.get("separability")
            if sep is not None:
                p.separability = float(sep)
            p.extra["identity"] = desc
            store.put_profile(p)
            eh = _fin(desc.get("eer_hard"))
            if math.isfinite(eh) and eh > LOW_EER:
                if e not in low and not ctx.training:     # once per crossing, live only
                    self._emit_low(store, s, e, desc, model, now)
                    low[e] = now
            elif math.isfinite(eh):
                low.pop(e, None)
        for e, rec in idw["ents"].items():                 # collecting, not yet scored
            if e in names:
                continue
            n_w = sum(1 for _ in rec["state"]["wins"])
            p = store.profile(s, e)
            if p is not None:
                p.extra["identity"] = {"status": "insufficient_windows", "n_windows": n_w,
                                       "min_windows": self.min_windows}
                store.put_profile(p)
        for ck, cs in (model.get("classes") or {}).items():
            p = store.profile(s, ck) or EntityProfile(system=s, entity=ck)
            if cs.get("separability") is not None:
                p.separability = float(cs["separability"])
            p.extra["identity"] = dict(cs, version=model["version"], fitted_ts=now)
            store.put_profile(p)

    def _emit_low(self, store, s: str, e: str, desc: Mapping[str, Any],
                  model: Mapping[str, Any], now: float) -> None:
        eh = float(desc["eer_hard"])
        cw = list(desc.get("confusable_with") or [])
        store.add_event(BehaviorEvent(
            system=s, entity=e, ts=now, kind="low_identifiability",
            score=float(min(1.0, max(0.0, 2.0 * eh))), severity=Severity.INFO,
            description=(f"{e} is hard to tell apart (EER_hard {eh:.2f}, separability "
                         f"{desc.get('separability')})"
                         + (f"; confusable with {', '.join(cw[:3])}" if cw else "")),
            extra={"eer_hard": eh, "separability": desc.get("separability"),
                   "recall1": desc.get("recall1"), "confusable_with": cw,
                   "anonymity_set": desc.get("anonymity_set")},
            axes=list(EVENT_AXES), dedupe_key=f"low_identifiability|{s}|{e}",
            model_version=int(model["version"])))


# ================================================================== helpers
def _nstats(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column NaN-aware (mean, population variance, count)."""
    ok = np.isfinite(M)
    n = ok.sum(axis=0)
    nn = np.maximum(n, 1)
    mean = np.where(ok, M, 0.0).sum(axis=0) / nn
    var = np.where(ok, (M - mean) ** 2, 0.0).sum(axis=0) / nn
    return np.where(n > 0, mean, np.nan), np.where(n > 0, var, np.nan), n


def _active_now(store, s: str, e: str, now: float, name: str = ACTIVE) -> bool:
    a = store.vec_at(s, e, name, now)
    return a is not None and len(a) > 0 and float(a[0]) > 0.5


def _fin(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return _NAN
    return v


def _r(x: Any, nd: int = 4) -> Optional[float]:
    v = _fin(x)
    return round(v, nd) if math.isfinite(v) else None
