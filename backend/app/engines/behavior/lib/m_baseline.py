"""Conjugate seasonal baseline maths and read accessors for model.baseline
(owner: B03 BaselineEngine; contract C). B04 (likelihood), B14 (golden
anchor), B18 (its own model.classagg), B02 / B30 (descriptors) and B01 / B24
(n_eff) call these functions instead of reading the model, so the layout can
evolve in one place.

Why this shape
  * Exact, exposure-aware predictives. Each feature has one likelihood family
    (FAMILY): counts are Gamma-Poisson / negative binomial in natural units
    with exposure e = dt/60 minutes, ratios are Beta-Binomial on (k, n) with n
    the matching count feature (== feature.expo channel), everything else is a
    Normal-Inverse-Gamma on the FEATURE_SPEC transform of feature.nat. Values
    are rebuilt from feature.nat (retained 8 d) so replay / rollback never
    depends on feature.vec (retained 1 d).
  * Seasonal buckets with kernel spreading: bin48 (hour x day type), plus
    hour-of-week (bin168) cells that refine the LOCATION once an anchor has
    >= 4 weeks of commits (dispersion stays pooled in bin48, which has ~5x
    the rows; atypical days - holidays on weekdays, make-up workdays - stay
    on bin48). Each commit is spread over the neighbouring hours with von
    Mises weights exp(4(cos(2 pi dh/24) - 1)), |dh| <= 2 (timebins formula).
  * Decay without touching every bucket: stats are stored scaled by
    2^((T - T0)/hl_f) per feature, so a commit only rewrites the <= 5 buckets
    it touches (checkpoints then share untouched bucket blocks) and a row
    older than the clock (release / replay) is folded with its own decay.
    Everything is linear and deterministic: checkpoint + replay == offline fit.
  * Rate cap as a per-day band: inside one band-day (boundaries 12 h away
    from the bucket's hour, so a daily cluster of commits never straddles
    two bands) a mature bucket's DATA mean sum(num)/sum(den) may wander at
    most +-c sigma15 around its value at the band start (c = 0.1 current,
    0.03 reference, plus allow_drift for the reference), sigma15 taken from
    the bucket's own posterior at the band start. A row that would push it
    out is folded with the largest weight that keeps it on the edge. The data
    mean is decay invariant, so only commits move it (the posterior mean also
    drifts slightly towards the hyperprior as data ages). A per-row allowance
    would also clip ordinary noise and freeze learning; the band lets noise
    cancel inside the day while a persistent shift moves at most c sigma15
    per band-day. The cap binds only on bucket-features holding >= CAP_MIN_W
    rows and >= CAP_MIN_E weighted minutes at the band start (half of each
    once capped: hysteresis), so the hyperprior start converges freely; never
    on rebased rows.
  * Batched folds: the gated learners only queue rows on the anchor (queue);
    the engine folds every queued row once per tick in rounds of one row per
    anchor, vectorised across all entities (flush_many / commit_many), which
    divides the numpy call overhead by the number of entities. Checkpoint
    blobs carry the queue, so dump never has to fold.
  * Hierarchy with empirical-Bayes strength: entity -> class (s, class:<rid>,
    >= 3 members) or system -> org -> hyperprior. A tier's effective stats are
    E = S + s * (E_parent - S) (leave-one-out) with s = min(1, kappa / size
    of the parent) and the hyperprior carried with weight h = s * h_parent,
    i.e. the parent is kappa pseudo-rows (15-min exposure rows for counts,
    trials for ratios, rows otherwise) and never more than it knows. A child
    with no data therefore has exactly the parent's predictive mean.
  * Half-life per feature chosen every 96 commits from {7, 14, 28} d by the
    pinball loss of the p5/p95 of global shadow predictives (one set of
    unbucketed stats per candidate half-life).

Families and observation space (observe / make_row):
    nb  count kinds                x = count this tick, e = dt/60 min
    bb  ratio kinds with a count   k = nat * n, n = nat[RATIO_N_IDX]
        feature as exposure        (http, dns, tls, flows)
    t   everything else            y = transform(nat): bytes log1p(v 60/dt),
                                   avg log / identity, bounded + retransmit /
                                   probe ratios logit, window / gauge identity
                                   or log1p, comp CLR rebuilt from nat counts;
                                   weight min(1, n/5) when n is a count feature
Hyperpriors (lib/priors): Gamma(0.5, 0.5 min) on the per-minute rate,
Beta(0.5, 0.5), NIG(m0_f, 0.01, 1, 4).

Anchor state (class Anchor, persisted inside model.baseline). The contract's
stats[B, F, k] is stored per family (k = 6 count, 4 ratio, 3 t) and scaled:
    a48[48, FULL.WIDTH]   per bucket: count [W, Sx, Se, Sxx, Sxe, See] x 10,
                          ratio [W, Sk, Sn, Skk/n] x 10, NIG [W, Sy, Syy] x 32
                          (FULL.L = 196), then the band edges lo[52], hi[52]
                          (NaN: not capped) and the band-day id
    a168[168, LOC.WIDTH]  location cells: count [W, Sx, Se], ratio [W, Sk, Sn],
                          NIG [W, Sy]; band-start mean m_ref[52] and band id
                          (None: no cells, the reference anchor)
    T0, T, hl[52]         scale epoch, clock (newest folded ts), half-lives (s)
    sh[3, FULL.L]         shadow global stats per candidate half-life (+ losses)
    pending               queued (row, w, cap, drift) not yet folded
The ratio's sum (k/n)^2 slot of the spec is not kept: bayes.ratio_posterior
does not use it.
model.baseline@(s, e) (dict, by reference):
    {fmt, tier: 'entity', version, branch, current: Anchor, reference: Anchor,
     gate / gate_ref: GateState, held: [CommitRow(ts, w_eff, w_prov)],
     golden: {week, snaps, stats, n_reset}, n_eff, allow_drift, ...bookkeeping}
model.baseline@(s, class:<rid> | __system__) and @(__org__, __org__):
    {fmt, tier, ts, version, stats[48, L] (sum of member current stats at ts),
     E[48, L], h[48, 52] (effective stats of the chain above), kappa[52]
     (EB strength for entity children), kappa_cls[52] (system: for classes),
     members, n_eff}

Accessor signatures (store / model may be empty: hyperprior defaults):
  observation space
    observe(nat, dt_s) -> (num[52], den[52], wf[52])  family observation, den =
                          exposure (count min, ratio n, t 1), wf > 0 = valid
    values_from_nat(nat, dt_s) -> ndarray[52]          x | k/n | y per family
    make_row(ts, nat, dt_s, tctx, drift=0.0, elig=True) -> Row
  learning (pure; B03 and B18)
    new_anchor(week=True, select=True, hl_days=14.0) -> Anchor
    new_model() -> {'current': Anchor, 'reference': Anchor}      (B18)
    commit(anchor, row, w, cap=CAP_CURRENT, drift=0.0) -> anchor  (fold now, in place)
    commit_many(anchors, rows, ws, caps, drifts)   one row into each of distinct
                                                   anchors, vectorised (B18 classes)
    queue(anchor, row, w, cap, drift) -> anchor;  flush_many(anchors)
    merge(own, other, w) -> own                                   (link seeding)
    on_rebase_current(anchor, tau, until) / on_rebase_reference(anchor, tau)
    dump(anchor, dtype=float32) -> Blob;  load(blob) -> Anchor
    true_stats(anchor, T=None) -> ndarray[48, FULL.L] | None;  true_cells(...)
  predictives
    predictive(store, s, e, tctx, anchor='current'|'reference', tier='entity'|
               'class'|'system'|'org', loo=True, model=None) -> Pred
    predictive_set(store, s, e, tctx, model=None) -> {'current', 'reference', 'class'}
    anchor_predictive(anchor, tctx, parent=None) -> Pred      (B18, no store)
    midp(pred, nat, dt_s) -> (u[52], p[52])          mid-p per family, NaN unscored
    loglik(pred, nat, dt_s) -> ndarray[52]           log pmf / pdf (nats)
    quantiles(pred, qs, dt_s=900, nat=None) -> ndarray[len(qs), 52] natural units
    mean_nat(pred, dt_s=900) -> ndarray[52]          predictive mean, natural units
    vec_median_sd(pred) -> (median[52], sd[52])      FEATURE_SPEC vec space (legacy)
    sd15(pred) -> ndarray[52]                        predictive sd for 15 min, mean units
    bucket_means(anchor, T=None) -> (mean[48, 52], sd15[48, 52])  own stats + hyperprior
    profile_many(store, s, ents, models, tctx) -> (median[n, 52], sd[n, 52])  vec space
  model
    n_eff(model | anchor) -> float;  maturity(model) -> dict;  hl_days(model) -> ndarray
    n_eff_by_bucket(model | anchor, anchor='current') -> float[48]  rows per bin48 bucket
    own_support(model, tctx, anchor='current') -> (W[52], expo_min)  the entity's own
                                  (no backoff) rows / exposure at tctx's bucket
    held(model) -> [ts];  last_commit_ts(model) -> float;  version(model) -> int
    golden(model) -> ndarray[48, L] | None;  has_golden(model) -> bool
    anchor_summary(model, anchor, feature) -> (mean, sigma15)   data-weighted over
                                  bin48 buckets; live model or runner snapshot (eval)
    golden_offset(store, s, e) -> ndarray[52] | None   zr is measured against golden
                                  once one exists (the reference predictive IS golden),
                                  so the offset is 0 then; None before
    descriptors(model) -> dict                       hourly means per day type (B02/B30)
    seasonal_curves(model, day_type, names) -> {name: [24]}  vec space (legacy profile)
  tiers (pure; B03 orchestrates)
    join(S, Ep, hp, kappa, loo=True) -> (E, h);  eb_kappa(children) -> ndarray[52]
    tier_model(store, s, key) -> dict | None;  parent_key(store, s, e) -> str
    median_select(snapshots) -> ndarray[48, L]       golden = cell-wise median
Pred fields ([52], NaN where not applicable): mu (nb rate / min), r (nb size),
p, c (bb: a = p c, b = (1 - p) c), df, loc, scale (t), mean, ebar (mean row
exposure, min), tier, anchor, bucket, mode ('bin48' | 'bin168'). For B04:
count exposure e = dt/60 min, ratio n = nat[RATIO_N_IDX] (the feature.expo
channel), t values = values_from_nat (the FEATURE_SPEC vec transform).
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from scipy import special as sp

from . import bayes
from . import features as F
from . import grains as GR
from . import m_class
from . import priors as PR
from .classkeys import ORG, SYSTEM_KEY, is_pseudo

MODEL = "model.baseline"
FMT = 2                            # spec v2.1: fmt 2 (H anchors + Q current anchor)
FMTS = (1, 2)                      # fmt-1 models load as H-only fmt 2
NF = F.FEATURE_DIM
DAY = 86400.0
WEEK = 7.0 * DAY

HL_CAND_DAYS: Tuple[float, ...] = (7.0, 14.0, 28.0)
HL_CAND_S = np.array(HL_CAND_DAYS) * DAY
HL_DEFAULT_DAYS = 14.0
HL_REF_DAYS = 28.0                 # the slow reference anchor keeps the longest half-life
SELECT_EVERY = 96                  # commits between half-life selections
PINBALL_EVERY = 8                  # the loss is evaluated on every 8th commit
SELECT_MIN_EVALS = 8               # evaluations with a valid value needed to re-select
LOSS_MEMORY = 0.5                  # losses are halved (not cleared) at each selection

CAP_CURRENT = 0.1                  # sigma15 per band-day
CAP_REFERENCE = 0.03
CAP_MIN_W = 8.0                    # rows in a bucket-feature before the cap binds ...
CAP_MIN_E = 120.0                  # ... and weighted exposure minutes in the bucket
WEEK_MODE_S = 28.0 * DAY           # bin168 location cells after 4 weeks of commits
K168 = 4.0                         # bin168 cell prior strength (rows of the bin48 bucket)
KAPPA_REF = 4.0                    # the current anchor as the reference's prior (rows)
KAPPA_MIN, KAPPA_MAX = PR.BACKOFF_KAPPA_MIN, PR.BACKOFF_KAPPA_MAX
KAPPA_DEFAULT = KAPPA_MIN          # EB not estimable (< 2 children): a weak prior
VM_KAPPA = 4.0
VM_MAX_DH = 2
RENORM_EXP = 40.0                  # re-anchor T0 once the scale reaches 2^40
Z95 = 1.6448536269514722
_TINY = 1e-300

TIERS = ("entity", "class", "system", "org")

# ------------------------------------------------------------------ families
FAM_NB, FAM_BB, FAM_T = 0, 1, 2
FAMILY_NAMES = ("nb", "bb", "t")
TX_BYTES, TX_LOG, TX_ID, TX_LOG1P, TX_LOGIT, TX_CLR = range(6)

_COUNT_OF_METRIC: Dict[str, int] = {}
for _n in F.FEATURE_NAMES_V2:
    _src = F.FEATURE_SOURCE[_n]
    if F.FEATURE_KIND[_n] == "count" and isinstance(_src, str):
        _COUNT_OF_METRIC[_src] = F.FEATURE_INDEX[_n]
# D1 writes derived.dns_fail_rate.n from dns.queries (derived/ratio.py)
_N_ALIAS = {"derived.dns_fail_rate.n": "dns.queries"}


def _n_features(nsrc: Any) -> Optional[Tuple[int, ...]]:
    """Count features whose nat sum is the exposure n, or None when n is not
    a count feature (then it is not recoverable from feature.nat)."""
    if nsrc is None:
        return None
    ms = [nsrc] if isinstance(nsrc, str) else list(nsrc)
    out = []
    for m in ms:
        m = _N_ALIAS.get(m, m)
        if m not in _COUNT_OF_METRIC:
            return None
        out.append(_COUNT_OF_METRIC[m])
    return tuple(out)


def _build_families() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fam = np.zeros(NF, dtype=np.intp)
    tx = np.full(NF, -1, dtype=np.intp)
    ratio_n = np.full(NF, -1, dtype=np.intp)
    nmat = np.zeros((NF, NF))
    has_n = np.zeros(NF, dtype=bool)
    for i, name in enumerate(F.FEATURE_NAMES_V2):
        kind = F.FEATURE_KIND[name]
        nf = _n_features(F.FEATURE_NSRC[name])
        if kind == "count":
            fam[i] = FAM_NB
            continue
        if kind == "ratio" and nf is not None and len(nf) == 1:
            fam[i] = FAM_BB
            ratio_n[i] = nf[0]
            continue
        fam[i] = FAM_T
        vtx = F.VEC_TX.get(name)
        if kind == "bytes":
            tx[i] = TX_BYTES
        elif kind == "avg":
            tx[i] = TX_ID if vtx == "identity" else TX_LOG1P if vtx == "log1p" else TX_LOG
        elif kind in ("bounded", "ratio"):
            tx[i] = TX_LOGIT          # ratios whose n is not a count feature (pkts, probes)
        elif kind in ("window", "gauge"):
            tx[i] = TX_LOG1P if vtx == "log1p" else TX_LOG if vtx == "log" else TX_ID
        elif kind == "clr":
            tx[i] = TX_CLR
        else:
            raise ValueError(f"m_baseline: unsupported kind {kind!r} for {name}")
        if nf is not None:
            has_n[i] = True
            nmat[i, list(nf)] = 1.0
    return fam, tx, ratio_n, nmat, has_n


FAMILY, _TX_ALL, _RATIO_N_ALL, _NMAT_ALL, _HAS_N_ALL = _build_families()
CNT = np.flatnonzero(FAMILY == FAM_NB)
RAT = np.flatnonzero(FAMILY == FAM_BB)
NIG = np.flatnonzero(FAMILY == FAM_T)
nC, nR, nN = CNT.size, RAT.size, NIG.size
RATIO_N_IDX = _RATIO_N_ALL[RAT]            # count feature giving each ratio's n
_RAT_NPOS = np.searchsorted(CNT, RATIO_N_IDX)
_TX = _TX_ALL[NIG]
_NMAT = _NMAT_ALL[NIG]                      # (nN, NF): sum of count nats = n
_HAS_N = _HAS_N_ALL[NIG]
_TX_SETS = {c: np.flatnonzero(_TX == c) for c in range(6)}
_CLR_POS = np.array([int(np.flatnonzero(NIG == F.FEATURE_INDEX[n])[0]) for n in F.CLR_FEATURES])
_I = F.FEATURE_INDEX
# CLR counts rebuilt from nat (features.CLR_FEATURES order): get/write/4xx/5xx
# are ratio * http_requests, syn is syn_ratio * flows, the rest are counts
_CLR_SRC = [(_I["http_get_ratio"], _I["http_requests"]), (_I["http_write_ratio"], _I["http_requests"]),
            (_I["http_4xx_rate"], _I["http_requests"]), (_I["http_5xx_rate"], _I["http_requests"]),
            (_I["dns_queries"], -1), (_I["tls_handshakes"], -1), (_I["flows"], -1),
            (_I["syn_ratio"], _I["flows"])]
_FLOWS = _I["flows"]

# hyperpriors
A0C = PR.COUNT_PRIOR.a0
B0C = PR.COUNT_PRIOR.b0
A0R, B0R = PR.RATIO_PRIOR.a0, PR.RATIO_PRIOR.b0
K0N, AL0N, BE0N = PR.KAPPA0, PR.ALPHA0, PR.BETA0


def _nig_m0() -> np.ndarray:
    out = np.zeros(nN)
    for j, f in enumerate(NIG):
        name = F.FEATURE_NAMES_V2[f]
        kind = F.FEATURE_KIND[name]
        if kind == "ratio":
            out[j] = PR.KIND_M0["bounded"]
        else:
            out[j] = PR.FEATURE_M0.get(name, PR.KIND_M0[kind])
    return out


M0N = _nig_m0()
# mean = (P + sum num) / (Q + sum den) for every family
_P = np.zeros(NF)
_Q = np.zeros(NF)
_P[CNT], _Q[CNT] = A0C, B0C
_P[RAT], _Q[RAT] = A0R, A0R + B0R
_P[NIG], _Q[NIG] = K0N * M0N, K0N
_JOIN_UNIT = np.ones(NF)
_JOIN_UNIT[CNT] = 15.0                     # count kappa is in 15-minute exposure rows
_IS_NB = FAMILY == FAM_NB
_IS_BB = FAMILY == FAM_BB


# ------------------------------------------------------------------ layouts
class _Layout:
    """Slot layout of one bucket row. full: every sufficient statistic, then
    the band edges lo[52], hi[52] and the band id; loc: the first-order
    (location) statistics, then the band-start mean m_ref[52] and the band id."""

    def __init__(self, full: bool) -> None:
        self.full = full
        per = {FAM_NB: 6 if full else 3, FAM_BB: 4 if full else 3, FAM_T: 3 if full else 2}
        W = np.zeros(NF, dtype=np.intp)
        NUM = np.zeros(NF, dtype=np.intp)
        DEN = np.zeros(NF, dtype=np.intp)
        slot_f: List[int] = []
        nonneg: List[bool] = []
        moment: List[str] = []
        blocks = {}
        names = {FAM_NB: ("1", "x", "e", "xx", "xe", "ee"), FAM_BB: ("1", "x", "e", "kkn"),
                 FAM_T: ("1", "x", "xx")}
        pos = 0
        for fam, idx in ((FAM_NB, CNT), (FAM_BB, RAT), (FAM_T, NIG)):
            k = per[fam]
            S = np.arange(pos, pos + k * idx.size, dtype=np.intp).reshape(idx.size, k)
            blocks[fam] = S
            for j, f in enumerate(idx):
                W[f], NUM[f] = S[j, 0], S[j, 1]
                DEN[f] = S[j, 0] if fam == FAM_T else S[j, 2]
                slot_f.extend([int(f)] * k)
                nonneg.extend([not (fam == FAM_T and m == 1) for m in range(k)])
                moment.extend(names[fam][:k])
            pos += k * idx.size
        self.L = pos
        if full:
            self.LO, self.HI, self.BAND = pos, pos + NF, pos + 2 * NF
        else:
            self.LO = self.HI = -1
            self.M_REF, self.BAND = pos, pos + NF
        self.WIDTH = self.BAND + 1
        self.C_S, self.R_S, self.N_S = blocks[FAM_NB], blocks[FAM_BB], blocks[FAM_T]
        self.W, self.NUM, self.DEN = W, NUM, DEN
        self.SLOT_F = np.asarray(slot_f, dtype=np.intp)
        self.NONNEG = np.asarray(nonneg, dtype=bool)
        self.MOMENT = moment
        self.EXPO = int(DEN[_FLOWS])       # weighted exposure minutes of the bucket
        self.W_EXPO = int(W[_FLOWS])


FULL = _Layout(True)
LOC = _Layout(False)
L_STATS = FULL.L


def _moment_slots(lay: _Layout, m: str) -> Tuple[np.ndarray, np.ndarray]:
    sl = np.asarray([i for i, x in enumerate(lay.MOMENT) if x == m], dtype=np.intp)
    return sl, lay.SLOT_F[sl]


# unit-weight contribution template of a FULL row: c = C0 with the moment slots filled
_C0 = np.zeros(FULL.L)
_C0[_moment_slots(FULL, "1")[0]] = 1.0
_CX = _moment_slots(FULL, "x")
_CE = _moment_slots(FULL, "e")
_CXX = _moment_slots(FULL, "xx")
_CXE = _moment_slots(FULL, "xe")
_CEE = _moment_slots(FULL, "ee")
_CKK = _moment_slots(FULL, "kkn")
# LOC slots are a subset of FULL slots (same feature and moment)
_LOC_FROM_FULL = np.asarray(
    [next(i for i in range(FULL.L) if FULL.SLOT_F[i] == LOC.SLOT_F[j]
          and FULL.MOMENT[i] == LOC.MOMENT[j]) for j in range(LOC.L)], dtype=np.intp)
_FLOW_W_SLOT = FULL.W[_FLOWS]


# ================================================================ observation
_CLR_A = np.asarray([a for a, _ in _CLR_SRC], dtype=np.intp)
_CLR_B = np.asarray([b if b >= 0 else NF for _, b in _CLR_SRC], dtype=np.intp)   # NF -> 1.0
_NONNEG_OBS = FAMILY != FAM_T              # counts and ratios must be >= 0


def _nig_values(X: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """FEATURE_SPEC vec transform of the t-family columns [n, nN], rebuilt
    from nat rows X[n, 52] (the CLR composition from the counts behind comp_*)."""
    V = X[:, NIG]
    Y = V.copy()                                           # TX_ID
    r60 = (60.0 / dt)[:, None]
    i = _TX_SETS[TX_BYTES]
    Y[:, i] = np.log1p(np.maximum(V[:, i], 0.0) * r60)
    i = _TX_SETS[TX_LOG]
    vi = V[:, i]
    Y[:, i] = np.log(np.where(vi >= 0.0, np.maximum(vi, F.AVG_FLOOR), np.nan))
    i = _TX_SETS[TX_LOG1P]
    vi = V[:, i]
    Y[:, i] = np.log1p(np.where(vi > -1.0, vi, np.nan))
    i = _TX_SETS[TX_LOGIT]
    pp = np.clip(V[:, i], F.BOUNDED_EPS, 1.0 - F.BOUNDED_EPS)
    Y[:, i] = np.log(pp) - np.log1p(-pp)
    XE = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    cnt = XE[:, _CLR_A] * XE[:, _CLR_B]
    cnt = np.where(cnt > 0.0, cnt, 0.0)                   # NaN (stale ratio) -> 0
    ly = np.log(cnt * r60 + F.CLR_PSEUDO)
    Y[:, _CLR_POS] = np.where((cnt.sum(axis=1) > 0.0)[:, None],
                              ly - ly.mean(axis=1, keepdims=True), np.nan)
    return Y


def _observe_many(X: np.ndarray, dt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.errstate(all="ignore"):
        num = X.copy()
        n = X[:, RATIO_N_IDX]
        num[:, RAT] = np.minimum(X[:, RAT], 1.0) * n
        num[:, NIG] = _nig_values(X, dt)
        den = np.ones_like(X)
        den[:, CNT] = (dt / 60.0)[:, None]
        den[:, RAT] = n
        valid = np.isfinite(num) & (den > 0.0) & ((num >= 0.0) | ~_NONNEG_OBS)
        wf = valid.astype(np.float64)
        xz = np.where(X > 0.0, X, 0.0)                     # NaN -> 0
        wf[:, NIG] *= np.where(_HAS_N, np.minimum(1.0, (xz @ _NMAT.T) / 5.0), 1.0)
        valid = wf > 0.0
        return np.where(valid, num, 0.0), np.where(valid, den, 1.0), wf


def _check_dt(dt_s: Any) -> float:
    dt = float(dt_s)
    if not (dt > 0.0 and math.isfinite(dt)):
        raise ValueError(f"m_baseline: bad dt_s {dt_s!r}")
    return dt


def _check_nat(nat: Any) -> np.ndarray:
    x = np.asarray(nat, dtype=np.float64).reshape(-1)
    if x.size != NF:
        raise ValueError(f"m_baseline: nat must have {NF} values, got {x.size}")
    return x


def observe(nat: Sequence[float], dt_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(num, den, wf) of one row in family observation space.

    count: num = x (count this tick), den = dt/60 minutes; ratio: num = k,
    den = n (the matching count feature); t: num = y, den = 1. wf is the
    per-feature weight (0 = not observed: NaN, n = 0, ...), min(1, n/5) for
    t features with a count exposure. Invalid entries are num = 0, den = 1 so
    they can be multiplied safely."""
    x = _check_nat(nat)
    num, den, wf = _observe_many(x[None, :], np.array([_check_dt(dt_s)]))
    return num[0], den[0], wf[0]


def values_from_nat(nat: Sequence[float], dt_s: float) -> np.ndarray:
    """Family value per feature (count x, ratio k/n, t y); NaN if unobserved."""
    num, den, wf = observe(nat, dt_s)
    out = np.where(wf > 0.0, num, np.nan)
    out[RAT] = np.where(wf[RAT] > 0.0, num[RAT] / den[RAT], np.nan)
    return out


def _contrib_many(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Unit-weight contributions of rows [n, 52] to FULL bucket rows [n, L]."""
    c = np.tile(_C0, (num.shape[0], 1))
    c[:, _CX[0]] = num[:, _CX[1]]
    c[:, _CE[0]] = den[:, _CE[1]]
    xx = num[:, _CXX[1]]
    c[:, _CXX[0]] = xx * xx
    c[:, _CXE[0]] = num[:, _CXE[1]] * den[:, _CXE[1]]
    ee = den[:, _CEE[1]]
    c[:, _CEE[0]] = ee * ee
    kk = num[:, _CKK[1]]
    c[:, _CKK[0]] = kk * kk / den[:, _CKK[1]]
    return c


class Row(NamedTuple):
    """One tick to commit: its feature.nat, real exposure dt and time context
    (feature.tctx fields hour_local, dow, day_type). Observation-space values
    are computed when rows are folded, vectorised across anchors."""
    ts: float
    nat: np.ndarray
    dt: float
    tctx: Mapping[str, Any]
    drift: float = 0.0      # reference: allow_drift in force (log-units / day)
    elig: bool = True       # reference: admission decision
    pair: Optional[Tuple[np.ndarray, np.ndarray]] = None   # spec v2.1: (Q nats [M, 52], Q covs [M])


def make_row(ts: float, nat: Sequence[float], dt_s: float, tctx: Mapping[str, Any],
             drift: float = 0.0, elig: bool = True,
             pair: Optional[Tuple[Any, Any]] = None) -> Row:
    """Row of tick ts from its feature.nat[52], real dt and time context.
    spec v2.1: a grain row passes dt_s = cov and the midpoint tctx; `pair`
    = (the M Q rows' nats, their covs) of a paired hour (cadence.md §6.3)."""
    for k in ("hour_local", "dow", "day_type"):
        if k not in tctx:
            raise ValueError(f"m_baseline.make_row: tctx lacks {k!r}")
    if pair is not None:
        qn = np.asarray(pair[0], dtype=np.float64).reshape(-1, NF)
        qc = np.asarray(pair[1], dtype=np.float64).reshape(-1)
        pair = (qn, qc)
    return Row(float(ts), _check_nat(nat), _check_dt(dt_s), tctx, float(drift), bool(elig), pair)


@lru_cache(maxsize=8192)
def _vm(hour6: float) -> Tuple[int, Tuple[int, ...], Tuple[float, ...]]:
    """(floor hour, 5 offsets, 5 weights) of the von Mises spread: bins whose
    centre is within 2 h of the hour (timebins.von_mises_weights formula),
    padded to 5 with a distinct zero-weight bin."""
    base = int(math.floor(hour6))
    offs, ws = [], []
    for off in range(-3, 4):
        dh = base + off + 0.5 - hour6
        if abs(dh) <= VM_MAX_DH + 1e-9:
            s = math.sin(math.pi * dh / 24.0)
            offs.append(off)
            ws.append(math.exp(-2.0 * VM_KAPPA * s * s))
    while len(offs) < 5:
        offs.append(3 if 3 not in offs else -3)
        ws.append(0.0)
    return base, tuple(offs), tuple(ws)


class _Batch(NamedTuple):
    ts: np.ndarray          # [n]
    local_ts: np.ndarray    # ts in local wall time (band ids)
    nwd: np.ndarray         # 1 = nonworkday
    dow: np.ndarray
    typical: np.ndarray     # day type matches the weekday (no holiday / make-up day)
    hours: np.ndarray       # [n, 5] spread hours (unwrapped)
    vm: np.ndarray          # [n, 5] spread weights
    num: np.ndarray         # [n, 52]
    den: np.ndarray
    wf: np.ndarray
    cfull: np.ndarray       # [n, FULL.L]
    cloc: np.ndarray        # [n, LOC.L]


def _batch(rows: Sequence[Row]) -> _Batch:
    n = len(rows)
    X = np.empty((n, NF))
    ts = np.empty(n)
    dt = np.empty(n)
    hour = np.empty(n)
    nwd = np.empty(n, dtype=np.intp)
    dow = np.empty(n, dtype=np.intp)
    hours = np.empty((n, 5), dtype=np.intp)
    vm = np.empty((n, 5))
    for i, r in enumerate(rows):
        X[i] = r.nat
        ts[i], dt[i] = r.ts, r.dt
        tc = r.tctx
        h = float(tc["hour_local"])
        hour[i] = h
        dow[i] = int(tc["dow"])
        nwd[i] = 1 if tc.get("day_type") == "nonworkday" else 0
        base, offs, w = _vm(round(h, 6))
        hours[i] = offs
        hours[i] += base
        vm[i] = w
    num, den, wf = _observe_many(X, dt)
    cf = _contrib_many(num, den)
    # local wall time: the tz offset is a multiple of 15 min
    off = (np.round(((hour * 3600.0 - ts) % DAY) / 900.0) * 900.0) % DAY
    return _Batch(ts, ts + off, nwd, dow, nwd == (dow >= 5), hours, vm, num, den, wf,
                  cf, cf[:, _LOC_FROM_FULL])


# ==================================================================== anchor
class Anchor:
    """Sufficient statistics of one anchor (see module docstring)."""

    __slots__ = ("a48", "a168", "T0", "T", "hl", "t_first", "n_commit", "sh", "sh_T0",
                 "loss", "nloss", "n_eval", "select", "uncap_lo", "uncap_hi", "reset_after",
                 "n_reset", "blk48", "blk168", "dirty48", "dirty168", "blk_code", "pending",
                 "om", "v15")

    def __init__(self, week: bool = True, select: bool = True,
                 hl_days: float = HL_DEFAULT_DAYS) -> None:
        self.a48 = _new_arr(48, FULL)
        self.a168 = _new_arr(168, LOC) if week else None
        self.T0 = math.nan
        self.T = math.nan
        self.hl = np.full(NF, float(hl_days) * DAY)
        self.t_first = math.nan
        self.n_commit = 0
        self.sh = np.zeros((HL_CAND_S.size, FULL.L))
        self.sh_T0 = math.nan
        self.loss = np.zeros((HL_CAND_S.size, NF))
        self.nloss = np.zeros(NF)
        self.n_eval = 0
        self.select = bool(select)
        self.uncap_lo = math.nan
        self.uncap_hi = math.nan
        self.reset_after = math.nan
        self.n_reset = 0
        self.blk48: List[Optional[bytes]] = [None] * 48
        self.blk168: Optional[List[Optional[bytes]]] = [None] * 168 if week else None
        self.dirty48 = set(range(48))
        self.dirty168 = set(range(168)) if week else set()
        self.blk_code = ""
        self.pending: List[Tuple[Row, float, float, float]] = []   # (row, w, cap, drift)
        # spec v2.1: paired-hour sums [A, B, D, W] x 52 (scaled like the stats;
        # None: this anchor does not estimate omega) and the variance factor of
        # the rate cap's sigma15 (1 for tick rows / Q rows, M for H rows)
        self.om: Optional[np.ndarray] = None
        self.v15 = 1.0

    @property
    def empty(self) -> bool:
        return self.T != self.T

    def week_mode(self) -> bool:
        """bin168 location cells are used once commits span >= 4 weeks."""
        return (self.a168 is not None and self.T == self.T
                and self.T - self.t_first >= WEEK_MODE_S)

    def _dirty_all(self) -> None:
        self.dirty48 = set(range(48))
        if self.a168 is not None:
            self.dirty168 = set(range(168))

    def _asdict(self) -> Dict[str, Any]:
        """Compact summary for reports / eval snapshots: per bin48 bucket the
        capped data mean and sigma15 of every feature (mean units)."""
        mean, s15 = bucket_means(self)
        St = true_stats(self)
        W = np.zeros((48, NF)) if St is None else St[:, FULL.W]
        return {"T": self.T, "t_first": self.t_first, "n_commit": self.n_commit,
                "n_eff": n_eff(self), "hl_days": self.hl / DAY, "week_mode": self.week_mode(),
                "pending": len(self.pending), "mean": mean, "sd15": s15, "W": W,
                "om": true_om(self)}


def _new_arr(n: int, lay: _Layout) -> np.ndarray:
    a = np.zeros((n, lay.WIDTH))
    a[:, lay.BAND] = np.nan
    return a


def new_anchor(week: bool = True, select: bool = True, hl_days: float = HL_DEFAULT_DAYS,
               om: bool = False, v15: float = 1.0) -> Anchor:
    """spec v2.1: om=True gives the anchor the paired-hour omega sums (the
    H current anchor in canonical mode); v15 the sigma15 variance factor."""
    a = Anchor(week=week, select=select, hl_days=hl_days)
    if om:
        a.om = np.zeros((4, NF))
    a.v15 = float(v15)
    return a


def true_om(anc: Optional[Anchor], T: Optional[float] = None) -> Optional[np.ndarray]:
    """spec v2.1: the paired-hour sums [A, B, D, W] x 52 decayed to T (default
    the anchor clock); None when the anchor has none, zeros when empty."""
    if anc is None or anc.om is None:
        return None
    if anc.empty:
        return np.zeros((4, NF))
    return anc.om * np.exp2(-((anc.T if T is None else T) - anc.T0) / anc.hl)[None, :]


def new_model() -> Dict[str, Anchor]:
    """A current (bin48 + bin168, half-life selection) and a reference (bin48,
    28-d half-life) anchor: the two-anchor pair B18 keeps in model.classagg."""
    return {"current": new_anchor(True, True), "reference": new_anchor(False, False, HL_REF_DAYS)}


def _scale(anc: Anchor, T: float, lay: _Layout) -> np.ndarray:
    return np.exp2(-(float(T) - anc.T0) / anc.hl)[lay.SLOT_F]


def true_stats(anc: Optional[Anchor], T: Optional[float] = None) -> Optional[np.ndarray]:
    """bin48 statistics [48, FULL.L] at time T (default: the anchor clock); None if empty."""
    if anc is None or anc.empty:
        return None
    return anc.a48[:, :FULL.L] * _scale(anc, anc.T if T is None else T, FULL)


def true_cells(anc: Optional[Anchor], T: Optional[float] = None) -> Optional[np.ndarray]:
    """bin168 location statistics [168, LOC.L]; None if absent or empty."""
    if anc is None or anc.empty or anc.a168 is None:
        return None
    return anc.a168[:, :LOC.L] * _scale(anc, anc.T if T is None else T, LOC)


# -------------------------------------------------------------- lean maths
def _overdisp(W, sx, se, sxx, sxe, see):
    """bayes.count_overdispersion without broadcasting overhead (same formula)."""
    with np.errstate(all="ignore"):
        mu = sx / se
        resid = np.maximum((sxx - 2.0 * mu * sxe + mu * mu * see) / W, 0.0)
        excess = resid - mu * se / W
        kap = mu * mu * (see / W) / excess
    ok = (W > 1.0) & (se > 0.0) & (mu > 0.0) & (excess > 0.0) & np.isfinite(kap)
    return np.clip(np.where(ok, kap, bayes.NB_KAPPA_CLIP[1]), *bayes.NB_KAPPA_CLIP)


def _phi(W, sk, sn, skk):
    """bayes.ratio_posterior's phi-hat (method of moments, clipped [20, 1000])."""
    with np.errstate(all="ignore"):
        m = sk / sn
        S = np.maximum(skk - m * sk, 0.0)
        nbar = sn / W
        rho = (S / ((W - 1.0) * m * (1.0 - m)) - 1.0) / (nbar - 1.0)
        phi = 1.0 / rho - 1.0
    ok = ((W >= 3.0) & (sn > 0.0) & (m > 0.0) & (m < 1.0) & (nbar > 1.0) & (rho > 0.0)
          & np.isfinite(phi))
    return np.clip(np.where(ok, phi, bayes.BB_PHI_CLIP[1]), *bayes.BB_PHI_CLIP)


class _Par(NamedTuple):
    """Predictive parameters by family block: count [nb, nC], ratio [nb, nR],
    t [nb, nN] (feature order CNT / RAT / NIG)."""
    mu: np.ndarray      # count: rate per minute
    r: np.ndarray       # count: NB size
    p: np.ndarray       # ratio: mean
    c: np.ndarray       # ratio: concentration
    df: np.ndarray      # t
    loc: np.ndarray
    scale: np.ndarray


_FAM_ORDER = np.concatenate((CNT, RAT, NIG))
_INV = np.argsort(_FAM_ORDER)          # family-block columns -> feature order
_C_END = FULL.C_S[-1, -1] + 1
_R_END = FULL.R_S[-1, -1] + 1


def _blocks(E: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Views of FULL rows [nb, L] as [nb, nC, 6], [nb, nR, 4], [nb, nN, 3]."""
    nb = E.shape[0]
    return (E[:, :_C_END].reshape(nb, nC, 6), E[:, _C_END:_R_END].reshape(nb, nR, 4),
            E[:, _R_END:FULL.L].reshape(nb, nN, 3))


def _feat(cnt: Any, rat: Any, nig: Any) -> np.ndarray:
    """Family blocks -> feature order [nb, 52]."""
    return np.concatenate((cnt, rat, nig), axis=1)[:, _INV]


def _params(E: np.ndarray, h: np.ndarray) -> _Par:
    """Predictive parameters of effective stats E[nb, FULL.L] with the
    hyperprior carried at weight h[nb, 52] (count: Gamma rate + moment
    overdispersion -> NB(mu e, r = 1/(1/kappa + 1/a)); ratio: Beta mean and
    c = min(sum n + 2, phi); t: NIG posterior -> Student-t)."""
    C, R, N = _blocks(E)
    with np.errstate(all="ignore"):
        hc = h[:, CNT]
        sx = np.maximum(C[..., 1], 0.0)
        a = hc * A0C + sx
        mu = a / (hc * B0C + np.maximum(C[..., 2], 0.0))
        kap = _overdisp(C[..., 0], C[..., 1], C[..., 2], C[..., 3], C[..., 4], C[..., 5])
        r = 1.0 / (1.0 / kap + 1.0 / a)
        hr = h[:, RAT]
        sn = np.maximum(R[..., 2], 0.0)
        p = (hr * A0R + np.maximum(R[..., 1], 0.0)) / (hr * (A0R + B0R) + sn)
        c = np.minimum(sn + 2.0, _phi(R[..., 0], R[..., 1], R[..., 2], R[..., 3]))
        hn = h[:, NIG]
        W, sy, syy = np.maximum(N[..., 0], 0.0), N[..., 1], N[..., 2]
        k0 = K0N * hn
        kappa = k0 + W
        m = (k0 * M0N + sy) / kappa
        alpha = AL0N * hn + 0.5 * W
        Ws = np.where(W > 0.0, W, 1.0)
        within = np.maximum(syy - sy * sy / Ws, 0.0)
        dev = sy - W * M0N
        beta = BE0N * hn + 0.5 * within + k0 * dev * dev / (2.0 * Ws * kappa)
        scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
    return _Par(mu, r, p, c, 2.0 * alpha, m, scale)


def _mean(par: _Par) -> np.ndarray:
    return _feat(par.mu, par.p, par.loc)


def _sd15_par(par: _Par, ebar: np.ndarray, v: Optional[np.ndarray] = None) -> np.ndarray:
    """Predictive sd for a 15-minute exposure, in mean units, [nb, 52].
    count: NB sd of the 15-min count / 15; ratio: BB sd of k/n at the 15-min
    trials of its exposure feature; t: scale * sqrt(ebar/15), ebar = mean row
    exposure (min) of the bucket (a 15-min value averages ebar/15 rows' worth).
    spec v2.1: `v` [nb] is the Q transfer factor of an H anchor (the
    overdispersion 1/r and 1/(1 + c) scaled by v, cadence.md §6.1-§6.2)."""
    with np.errstate(all="ignore"):
        vv = np.ones((par.mu.shape[0], 1)) if v is None else np.asarray(v, dtype=np.float64).reshape(-1, 1)
        m15 = 15.0 * par.mu
        sc = np.sqrt(m15 + vv * m15 * m15 / par.r) / 15.0
        n15 = 15.0 * par.mu[:, _RAT_NPOS]
        n15 = np.where(np.isfinite(n15) & (n15 >= 1.0), n15, 1.0)
        pp = par.p
        cc = np.maximum(GR.C_T_MIN, (1.0 + par.c) / vv - 1.0) if v is not None else par.c
        sr = np.sqrt(pp * (1.0 - pp) * (n15 + cc) / (n15 * (1.0 + cc)))
        sn = par.scale * np.sqrt(ebar / 15.0)[:, None]
    return _feat(sc, sr, sn)


def _ebar(St: np.ndarray, lay: _Layout = FULL) -> np.ndarray:
    """Mean row exposure (minutes) per bucket, 15 when the bucket is empty."""
    Wf = St[:, lay.W[_FLOWS]]
    with np.errstate(all="ignore"):
        e = St[:, lay.EXPO] / Wf
    return np.where((Wf > 0.0) & np.isfinite(e) & (e > 0.0), e, 15.0)


_H1 = np.ones((48, NF))                  # full hyperprior weight (shared, read-only)
_H1.setflags(write=False)


def _sd15(St: np.ndarray, v: Optional[np.ndarray] = None) -> np.ndarray:
    """Own-posterior sigma15 (hyperprior + own stats) of FULL rows [nb, L]."""
    return _sd15_par(_params(St, np.ones((St.shape[0], NF))), _ebar(St), v)


# ------------------------------------------------------------------- commit
def _clip_weights(w0, Aq, Bq, mean0, m1, lo, hi, num, den):
    """Largest weights keeping each violating mean on its band edge (none if
    the mean already sits beyond the edge it is moving towards)."""
    up = m1 > mean0
    bound = np.where(up, hi, lo)
    inside = np.where(up, mean0 < bound, mean0 > bound)
    wc = (bound * Bq - Aq) / (num - bound * den)
    return np.where(inside & np.isfinite(wc), np.clip(wc, 0.0, w0), 0.0)


def _drift_many(mean0: np.ndarray, drift: np.ndarray) -> np.ndarray:
    """allow_drift (log-units / day) in mean units per row: count rate x
    expm1(d), ratio p(1-p) d, t d."""
    d = np.abs(drift)[:, None]
    return np.where(_IS_NB, mean0 * np.expm1(d), np.where(_IS_BB, mean0 * (1.0 - mean0) * d, d))


def _mature(sub: np.ndarray, lay: _Layout, Gi: np.ndarray, capped_before: np.ndarray) -> np.ndarray:
    """Bucket-features [m, 52] the cap binds on at a band start: >= CAP_MIN_W
    rows and >= CAP_MIN_E weighted minutes in the bucket; half of each for
    one that was capped in its previous band (hysteresis: decay alone must
    not uncap a bucket sitting at the threshold)."""
    f = np.where(capped_before, 0.5, 1.0)
    fe = np.where(capped_before.any(axis=1), 0.5, 1.0)
    return ((sub[:, lay.W] >= f * CAP_MIN_W * Gi)
            & (sub[:, lay.EXPO] >= fe * CAP_MIN_E * Gi[:, _FLOWS])[:, None])


def _band_edges(blk: np.ndarray, newb: np.ndarray, mean0: np.ndarray, G: np.ndarray,
                band: np.ndarray, caps: np.ndarray, drifts: np.ndarray,
                v15: Optional[np.ndarray] = None) -> None:
    """Open a new band for the (anchor, bucket) cells newb of bin48 blocks
    [n, 5, W]: edges mean0 -/+ half, half = cap sigma15 (+ allow_drift) with
    sigma15 from the bucket's own posterior at the band start; NaN edges (no
    cap) while a bucket-feature is immature. One _params call for every cell
    of the flush round."""
    ii, jj = np.nonzero(newb)
    sub = blk[ii, jj]
    mature = _mature(sub, FULL, G[ii], np.isfinite(sub[:, FULL.HI:FULL.BAND]))
    Gi = G[ii]
    hw = np.full(mature.shape, np.nan)
    rows = np.flatnonzero(mature.any(axis=1))
    if rows.size:
        St = sub[rows, :FULL.L] / Gi[rows][:, FULL.SLOT_F]
        vr = None
        if v15 is not None and np.any(v15 != 1.0):
            vr = v15[ii[rows]]
        h = caps[ii[rows]][:, None] * _sd15(St, vr)
        d = drifts[ii[rows]]
        if d.any():
            h = h + _drift_many(mean0[ii[rows], jj[rows]], d)
        hw[rows] = np.where(mature[rows], h, np.nan)
    m = mean0[ii, jj]
    blk[ii, jj, FULL.LO:FULL.HI] = m - hw
    blk[ii, jj, FULL.HI:FULL.BAND] = m + hw
    blk[ii, jj, FULL.BAND] = band[ii, jj]


def _fold48(ancs: Sequence[Anchor], idx: np.ndarray, W0: np.ndarray, B: _Batch, G: np.ndarray,
            band: np.ndarray, capped: np.ndarray, caps: np.ndarray,
            drifts: np.ndarray, v15: Optional[np.ndarray] = None) -> np.ndarray:
    """Fold one row per anchor into its bin48 buckets idx[n, 5] with weights
    W0[n, 5, 52] (scaled space) under the band cap: a violating
    bucket-feature is folded with the weight that puts its mean on the band
    edge. Returns the band half-widths [n, 5, 52] (for the bin168 cells)."""
    blk = np.stack([a.a48[idx[i]] for i, a in enumerate(ancs)])
    live = B.vm > 0.0
    # the capped mean is the data mean sum(num) / sum(den): decay scales both
    # sums alike, so only commits move it (the cap binds on mature buckets only)
    Aq = blk[..., FULL.NUM]
    Bq = blk[..., FULL.DEN]
    mean0 = Aq / Bq
    newb = capped[:, None] & live & ~(band <= blk[..., FULL.BAND])   # later band-day or none
    if newb.any():
        _band_edges(blk, newb, mean0, G, band, caps, drifts, v15)
    lo = blk[..., FULL.LO:FULL.HI]
    hi = blk[..., FULL.HI:FULL.BAND]
    num, den = B.num[:, None, :], B.den[:, None, :]
    m1 = (Aq + W0 * num) / (Bq + W0 * den)
    viol = ((m1 > hi) | (m1 < lo)) & capped[:, None, None]
    if viol.any():
        W0 = np.where(viol, _clip_weights(W0, Aq, Bq, mean0, m1, lo, hi, num, den), W0)
    if not capped.all():                        # rebased rows: a fresh band afterwards
        un = ~capped[:, None] & live
        blk[..., FULL.BAND] = np.where(un, np.nan, blk[..., FULL.BAND])
    half = 0.5 * (hi - lo)
    blk[..., :FULL.L] += W0[..., FULL.SLOT_F] * B.cfull[:, None, :]
    for i, a in enumerate(ancs):
        a.a48[idx[i]] = blk[i]
    return half


def _fold168(ancs: Sequence[Anchor], idx: np.ndarray, W0: np.ndarray, B: _Batch, G: np.ndarray,
             band: np.ndarray, capped: np.ndarray, half: np.ndarray, sel: np.ndarray) -> None:
    """Fold rows sel into bin168 location cells: a cell mean may move at most
    the half-width of its bin48 bucket's band around its own band-start mean
    (NaN while the cell is immature)."""
    blk = np.stack([ancs[i].a168[idx[i]] for i in sel])
    G, band, capped, half, W0 = G[sel], band[sel], capped[sel], half[sel], W0[sel]
    live = B.vm[sel] > 0.0
    Aq = blk[..., LOC.NUM]
    Bq = blk[..., LOC.DEN]
    mean0 = Aq / Bq
    newb = capped[:, None] & live & ~(band <= blk[..., LOC.BAND])
    if newb.any():
        ii, jj = np.nonzero(newb)
        sub = blk[ii, jj]
        mature = _mature(sub, LOC, G[ii], np.isfinite(sub[:, LOC.M_REF:LOC.BAND]))
        blk[ii, jj, LOC.M_REF:LOC.BAND] = np.where(mature, mean0[ii, jj], np.nan)
        blk[ii, jj, LOC.BAND] = band[ii, jj]
    mref = blk[..., LOC.M_REF:LOC.BAND]
    lo, hi = mref - half, mref + half
    num, den = B.num[sel][:, None, :], B.den[sel][:, None, :]
    m1 = (Aq + W0 * num) / (Bq + W0 * den)
    viol = ((m1 > hi) | (m1 < lo)) & capped[:, None, None]
    if viol.any():
        W0 = np.where(viol, _clip_weights(W0, Aq, Bq, mean0, m1, lo, hi, num, den), W0)
    if not capped.all():
        blk[..., LOC.BAND] = np.where(~capped[:, None] & live, np.nan, blk[..., LOC.BAND])
    blk[..., :LOC.L] += W0[..., LOC.SLOT_F] * B.cloc[sel][:, None, :]
    for j, i in enumerate(sel):
        ancs[i].a168[idx[i]] = blk[j]


def _hour_bucket(tc: Mapping[str, Any]) -> int:
    h = int(math.floor(float(tc["hour_local"]))) % 24
    return h + (24 if tc.get("day_type") == "nonworkday" else 0)


def paired_hour_stats(St_b: np.ndarray, row: Row) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(a, b, d, valid) [52] of one paired hour (cadence.md §6.3) against the
    bucket's own H predictive St_b [L] (before the row is folded): NB rates,
    BB proportions and t values of the M Q rows vs the H row (lib/grains
    paired_stats; the H excess variance from the predictive's mean)."""
    qn, qc = row.pair
    m = qn.shape[0]
    a = np.zeros(NF)
    b = np.zeros(NF)
    d = np.zeros(NF)
    ok = np.zeros(NF, dtype=bool)
    if m != GR.M or not np.all(qc >= GR.COVER_MIN * GR.GRAIN_S["q"] * (1.0 - 1e-9)):
        return a, b, d, ok
    par = _params(St_b[None, :], np.ones((1, NF)))
    num_q, den_q, wf_q = _observe_many(qn, qc)
    num_h, den_h, wf_h = _observe_many(row.nat[None, :], np.array([row.dt]))
    valid = (wf_q > 0.0).all(axis=0) & (wf_h[0] > 0.0)
    with np.errstate(all="ignore"):
        # counts: rates per minute
        aa, bb, _ = GR.paired_stats("nb", num_q[:, CNT], den_q[:, CNT], num_h[0, CNT],
                                    den_h[0, CNT], {"mean": par.mu[0], "r": par.r[0]})
        a[CNT], b[CNT] = aa, bb
        aa, bb, _ = GR.paired_stats("bb", num_q[:, RAT], den_q[:, RAT], num_h[0, RAT],
                                    den_h[0, RAT], {"mean": par.p[0], "c": par.c[0]})
        a[RAT], b[RAT] = aa, bb
        var = GR.t_variance(par.scale[0], par.df[0])
        aa, bb, dd = GR.paired_stats("t", num_q[:, NIG], np.ones_like(num_q[:, NIG]),
                                     num_h[0, NIG], 1.0, {"var": var})
        a[NIG], b[NIG], d[NIG] = aa, bb, dd
    ok = valid & np.isfinite(a) & np.isfinite(b) & np.isfinite(d) & (b > 0.0)
    return np.where(ok, a, 0.0), np.where(ok, b, 0.0), np.where(ok, d, 0.0), ok


def _fold_omega(ancs: Sequence[Anchor], rows: Sequence[Row], sel: Sequence[int],
                ws: np.ndarray, B: _Batch, T0: np.ndarray, HL: np.ndarray,
                Tn: np.ndarray) -> None:
    """Fold paired-hour (a, b, d, 1) into the anchors' om sums with the row's
    trust weight and decay (spec v2.1, cadence.md §6.3); before the fold of
    the row itself, so the predictive is the bucket's prior to the row."""
    for i in sel:
        anc, row = ancs[i], rows[i]
        bk = _hour_bucket(row.tctx)
        St_b = anc.a48[bk, :FULL.L] * np.exp2(-(Tn[i] - T0[i]) / HL[i])[FULL.SLOT_F] \
            if not anc.empty else np.zeros(FULL.L)
        a, b, d, ok = paired_hour_stats(St_b, row)
        if not ok.any():
            continue
        g = float(ws[i]) * np.exp2((B.ts[i] - T0[i]) / HL[i])
        anc.om[0] += np.where(ok, g * a, 0.0)
        anc.om[1] += np.where(ok, g * b, 0.0)
        anc.om[2] += np.where(ok, g * d, 0.0)
        anc.om[3] += np.where(ok, g, 0.0)


def commit_many(ancs: Sequence[Anchor], rows: Sequence[Row], ws: Sequence[float],
                caps: Sequence[float], drifts: Sequence[float]) -> None:
    """Fold one row into each of several DISTINCT anchors, vectorised across
    anchors (the engine batches all entities of a tick; B18 its classes).

    Deterministic in (state, row, w): replaying the same rows in the same
    order reproduces the statistics. A row older than an anchor's clock is
    folded with its own decay 2^-((T - ts)/hl) (release / replay order)."""
    keep = [i for i, w in enumerate(ws) if float(w) > 0.0 and math.isfinite(float(w))]
    if not keep:
        return
    ancs = [ancs[i] for i in keep]
    B = _batch([rows[i] for i in keep])
    ws_ = np.array([float(ws[i]) for i in keep])
    caps_ = np.array([float(caps[i]) for i in keep])
    drifts_ = np.array([float(drifts[i]) for i in keep])
    n = len(ancs)
    T0 = np.empty(n)
    Tn = np.empty(n)
    HL = np.empty((n, NF))
    capped = np.empty(n, dtype=bool)
    for i, a in enumerate(ancs):
        ts = float(B.ts[i])
        if a.reset_after == a.reset_after and ts >= a.reset_after:
            _reset(a)
        if a.empty:
            a.T0 = a.T = a.t_first = ts
        tn = ts if ts > a.T else a.T
        if (tn - a.T0) / float(a.hl.min()) > RENORM_EXP:
            _renorm(a, tn)
        T0[i], Tn[i], HL[i] = a.T0, tn, a.hl
        capped[i] = not (a.uncap_lo <= ts <= a.uncap_hi)
    with np.errstate(all="ignore"):
        G = np.exp2((Tn - T0)[:, None] / HL)
        # scaled row weight: w x validity x 2^((ts - T0)/hl), spread over the buckets
        WG = (ws_[:, None] * B.wf) * np.exp2((B.ts - T0)[:, None] / HL)
        W0 = B.vm[:, :, None] * WG[:, None, :]
        hours = B.hours
        band = np.floor((B.local_ts[:, None] - ((hours % 24) + 12.5) * 3600.0) / DAY)
        idx48 = hours % 24 + 24 * B.nwd[:, None]
        rows_k = [rows[i] for i in keep]
        om_sel = [i for i, a in enumerate(ancs) if a.om is not None and rows_k[i].pair is not None]
        if om_sel:
            _fold_omega(ancs, rows_k, om_sel, ws_, B, T0, HL, Tn)
        v15 = np.array([float(a.v15) for a in ancs])
        half = _fold48(ancs, idx48, W0, B, G, band, capped, caps_, drifts_,
                       v15 if np.any(v15 != 1.0) else None)
        sel = np.flatnonzero([a.a168 is not None and bool(B.typical[i]) for i, a in enumerate(ancs)])
        idx168 = (B.dow[:, None] * 24 + hours) % 168
        if sel.size:
            _fold168(ancs, idx168, W0, B, G, band, capped, half, sel)
        for i, a in enumerate(ancs):
            a.T = float(Tn[i])
            a.n_commit += 1
            a.dirty48.update(idx48[i].tolist())
        for i in sel:
            ancs[i].dirty168.update(idx168[i].tolist())
        _shadow(ancs, B, ws_)


def commit(anc: Anchor, row: Row, w: float, *, cap: float = CAP_CURRENT,
           drift: float = 0.0) -> Anchor:
    """Fold one row with trust weight w into one anchor, now (pure API; B18).
    Rows queued on the anchor are folded first."""
    flush_many([anc])
    commit_many([anc], [row], [w], [cap], [drift])
    return anc


def queue(anc: Anchor, row: Row, w: float, *, cap: float = CAP_CURRENT,
          drift: float = 0.0) -> Anchor:
    """GatedLearner update: remember the row for the next flush (O(1)). The
    engine flushes every anchor once per tick, vectorised across anchors; a
    checkpoint blob carries the queue, so dump never has to flush."""
    w = float(w)
    if w > 0.0 and math.isfinite(w):
        anc.pending.append((row, w, float(cap), float(drift)))
    return anc


def flush_many(ancs: Sequence[Anchor]) -> None:
    """Fold every queued row, in queue order per anchor, in rounds of one row
    per anchor (distinct anchors in a round)."""
    seen = set()
    live = []
    for a in ancs:
        if a is not None and a.pending and id(a) not in seen:
            seen.add(id(a))
            live.append(a)
    while live:
        items = [a.pending[0] for a in live]
        commit_many(live, [it[0] for it in items], [it[1] for it in items],
                    [it[2] for it in items], [it[3] for it in items])
        for a in live:
            a.pending.pop(0)
        live = [a for a in live if a.pending]


def _renorm(anc: Anchor, T: float) -> None:
    anc.a48[:, :FULL.L] *= _scale(anc, T, FULL)
    if anc.a168 is not None:
        anc.a168[:, :LOC.L] *= _scale(anc, T, LOC)
    if anc.om is not None:
        anc.om = anc.om * np.exp2(-(float(T) - anc.T0) / anc.hl)[None, :]
    anc.T0 = float(T)
    anc._dirty_all()


def _reset(anc: Anchor) -> None:
    """Empty the statistics (reference reset one day into a rebased regime)."""
    anc.a48 = _new_arr(48, FULL)
    if anc.a168 is not None:
        anc.a168 = _new_arr(168, LOC)
    anc.T0 = anc.T = anc.t_first = math.nan
    anc.n_commit = 0
    if anc.om is not None:
        anc.om = np.zeros((4, NF))
    anc.sh = np.zeros_like(anc.sh)
    anc.sh_T0 = math.nan
    anc.loss = np.zeros_like(anc.loss)
    anc.nloss = np.zeros_like(anc.nloss)
    anc.n_eval = 0
    anc.reset_after = math.nan
    anc.n_reset += 1
    anc._dirty_all()


def _set_hl(anc: Anchor, new_hl: np.ndarray) -> None:
    """Change per-feature half-lives, keeping the true stats at the clock."""
    fac = np.exp2((anc.T - anc.T0) * (1.0 / new_hl - 1.0 / anc.hl))
    anc.a48[:, :FULL.L] *= fac[FULL.SLOT_F]
    if anc.a168 is not None:
        anc.a168[:, :LOC.L] *= fac[LOC.SLOT_F]
    if anc.om is not None:
        anc.om = anc.om * fac[None, :]
    anc.hl = np.asarray(new_hl, dtype=np.float64).copy()
    anc._dirty_all()


# ----------------------------------------------------- half-life selection
def _rho(u: np.ndarray, tau: float) -> np.ndarray:
    return np.maximum(tau * u, (tau - 1.0) * u)


def _pinball(par: _Par, num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Pinball loss of the p5 / p95 of each predictive row against the
    observation (num, den) [k, 52] (moment approximations for NB / BB, exact
    t quantiles)."""
    y = num[:, _FAM_ORDER]
    dc = den[:, CNT]
    m = par.mu * dc
    sd = np.sqrt(m + m * m / par.r)
    lo_c, hi_c = np.maximum(m - Z95 * sd, 0.0), m + Z95 * sd
    dr = den[:, RAT]
    m = par.p * dr
    sd = np.sqrt(dr * par.p * (1.0 - par.p) * (dr + par.c) / (1.0 + par.c))
    lo_r, hi_r = np.maximum(m - Z95 * sd, 0.0), np.minimum(m + Z95 * sd, dr)
    t95 = sp.stdtrit(par.df, 0.95) * par.scale
    lo = np.concatenate((lo_c, lo_r, par.loc - t95), axis=1)
    hi = np.concatenate((hi_c, hi_r, par.loc + t95), axis=1)
    out = _rho(y - lo, 0.05) + _rho(y - hi, 0.95)
    return np.where(np.isfinite(out), out, 0.0)[:, _INV]


_NH = HL_CAND_S.size


def _shadow(ancs: Sequence[Anchor], B: _Batch, ws: np.ndarray) -> None:
    """Global (unbucketed, uncapped) stats at each candidate half-life; for
    anchors with `select`, on every PINBALL_EVERY-th commit the p5/p95
    pinball loss of each candidate before the fold, and every SELECT_EVERY
    commits the per-feature half-life with the smallest (decayed) loss."""
    n = len(ancs)
    shT0 = np.empty(n)
    T = np.empty(n)
    for i, a in enumerate(ancs):
        if a.sh_T0 != a.sh_T0:
            a.sh_T0 = a.T
        if (a.T - a.sh_T0) / HL_CAND_S[0] > RENORM_EXP:
            a.sh *= np.exp2(-(a.T - a.sh_T0) / HL_CAND_S)[:, None]
            a.sh_T0 = a.T
        shT0[i], T[i] = a.sh_T0, a.T
    SH = np.stack([a.sh for a in ancs])                              # n, 3, L
    valid = B.wf > 0.0
    ev = [i for i, a in enumerate(ancs)
          if a.select and a.n_commit % PINBALL_EVERY == 0 and valid[i].any()]
    if ev:
        k = len(ev)
        St = (SH[ev] * np.exp2(-(T - shT0)[ev][:, None] / HL_CAND_S)[:, :, None]).reshape(k * _NH, -1)
        par = _params(St, np.ones((k * _NH, NF)))
        loss = _pinball(par, np.repeat(B.num[ev], _NH, axis=0),
                        np.repeat(B.den[ev], _NH, axis=0)).reshape(k, _NH, NF)
        for j, i in enumerate(ev):
            a = ancs[i]
            a.loss += np.where(valid[i], ws[i] * loss[j], 0.0)
            a.nloss += valid[i]
    SH += np.exp2((B.ts - shT0)[:, None] / HL_CAND_S)[:, :, None] * \
        ((ws[:, None] * B.wf)[:, FULL.SLOT_F] * B.cfull)[:, None, :]
    for i, a in enumerate(ancs):
        a.sh = SH[i]
        if a.select:
            a.n_eval += 1
            if a.n_eval >= SELECT_EVERY:
                _select(a)


def _select(anc: Anchor) -> None:
    ok = anc.nloss >= SELECT_MIN_EVALS
    if ok.any():
        cols = np.arange(NF)
        best = np.argmin(anc.loss, axis=0)
        cur = np.argmin(np.abs(anc.hl[None, :] - HL_CAND_S[:, None]), axis=0)
        keep = anc.loss[cur, cols] <= anc.loss[best, cols] * (1.0 + 1e-9)
        new = np.where(ok & ~keep, HL_CAND_S[best], anc.hl)
        if np.any(new != anc.hl):
            _set_hl(anc, new)
    anc.loss *= LOSS_MEMORY
    anc.nloss *= LOSS_MEMORY
    anc.n_eval = 0


# ---------------------------------------------------- merge / rebase hooks
def merge(own: Anchor, other: Optional[Anchor], w: float) -> Anchor:
    """own + w * other in sufficient-statistic space (link seeding), other
    folded at its own clock. Bands and half-lives of `own` are kept."""
    flush_many([own, other])
    if other is None or other.empty or not (float(w) > 0.0):
        return own
    To = other.T
    S48 = true_stats(other)
    S168 = true_cells(other)
    if own.empty:
        own.T0 = own.T = To
        own.t_first = other.t_first
    T_new = max(own.T, To)
    if ((T_new - own.T0) / own.hl).max() > RENORM_EXP:
        _renorm(own, T_new)
    G = np.exp2((To - own.T0) / own.hl)
    own.a48[:, :FULL.L] += float(w) * S48 * G[FULL.SLOT_F]
    if own.a168 is not None and S168 is not None:
        own.a168[:, :LOC.L] += float(w) * S168 * G[LOC.SLOT_F]
    if own.om is not None and other.om is not None:
        own.om = own.om + float(w) * true_om(other) * G[None, :]
    own.T = T_new
    own.t_first = min(own.t_first, other.t_first)
    own._dirty_all()
    return own


def on_rebase_current(anc: Anchor, tau: float, until: float) -> Anchor:
    """ACCEPTED regime: rows with tau <= ts <= until are folded without the cap."""
    anc.uncap_lo = float(tau)
    anc.uncap_hi = float(until)
    return anc


def on_rebase_reference(anc: Anchor, tau: float) -> Anchor:
    """The reference restarts one day into the new regime."""
    anc.reset_after = float(tau) + DAY
    return anc


# -------------------------------------------------------- checkpoint blobs
_ZMIN = 0.25          # compress a block when >= 25 % of its slots are zero


def _pack(row: np.ndarray, dt: np.dtype) -> bytes:
    """One bucket row as tagged bytes: b'z' + zlib level 1 for mostly-zero
    rows (features the entity never shows: ~5x smaller), else b'r' + raw."""
    raw = row.astype(dt).tobytes()
    if row.size - np.count_nonzero(row) >= _ZMIN * row.size:
        return b"z" + zlib.compress(raw, 1)
    return b"r" + raw


def _unpack(block: bytes) -> bytes:
    return zlib.decompress(block[1:]) if block[:1] == b"z" else block[1:]


class Blob(dict):
    """Checkpoint blob. Immutable by construction (dump copies every array,
    load copies out of it, bucket blocks are bytes and queued rows are never
    mutated), so deep copies share it: copying ~220 block references per
    checkpoint would otherwise cost more than the commit itself."""

    def __deepcopy__(self, memo: Any) -> "Blob":
        return self


def dump(anc: Anchor, dtype: Any = np.float32) -> Dict[str, Any]:
    """Checkpoint blob. Bucket rows are immutable bytes blocks (float32 by
    default; float16 would overflow the raw moment sums; zlib for mostly-zero
    rows) rebuilt only for buckets touched since the last dump, so successive
    checkpoints share every untouched block; rows still queued travel with
    the blob."""
    dt = np.dtype(dtype)
    if anc.blk_code != dt.str:
        anc.blk_code = dt.str
        anc._dirty_all()
    for b in anc.dirty48:
        anc.blk48[b] = _pack(anc.a48[b], dt)
    anc.dirty48 = set()
    b168 = None
    if anc.a168 is not None:
        for b in anc.dirty168:
            anc.blk168[b] = _pack(anc.a168[b], dt)
        anc.dirty168 = set()
        b168 = tuple(anc.blk168)
    # half-lives, shadow stats and losses: one lossless compressed float64 buffer
    aux = zlib.compress(np.concatenate((anc.hl, anc.sh.ravel(), anc.loss.ravel(),
                                        anc.nloss)).tobytes(), 1)
    return Blob({"v": FMT, "dtype": dt.str, "b48": tuple(anc.blk48), "b168": b168,
                 "T0": anc.T0, "T": anc.T, "t_first": anc.t_first, "aux": aux,
                 "n_commit": anc.n_commit, "sh_T0": anc.sh_T0, "n_eval": anc.n_eval,
                 "select": anc.select, "uncap": (anc.uncap_lo, anc.uncap_hi),
                 "reset_after": anc.reset_after, "n_reset": anc.n_reset,
                 "pending": tuple(anc.pending),
                 "om": None if anc.om is None else anc.om.astype(np.float64).tobytes(),
                 "v15": float(anc.v15)})


def load(blob: Mapping[str, Any]) -> Anchor:
    dt = np.dtype(blob["dtype"])
    week = blob.get("b168") is not None
    anc = Anchor(week=week, select=bool(blob.get("select", True)))
    anc.a48 = np.frombuffer(b"".join(_unpack(x) for x in blob["b48"]),
                            dtype=dt).reshape(48, FULL.WIDTH).astype(np.float64)
    anc.blk48 = list(blob["b48"])
    anc.dirty48 = set()
    if week:
        anc.a168 = np.frombuffer(b"".join(_unpack(x) for x in blob["b168"]),
                                 dtype=dt).reshape(168, LOC.WIDTH).astype(np.float64)
        anc.blk168 = list(blob["b168"])
        anc.dirty168 = set()
    anc.blk_code = dt.str
    anc.T0, anc.T, anc.t_first = float(blob["T0"]), float(blob["T"]), float(blob["t_first"])
    aux = np.frombuffer(zlib.decompress(blob["aux"]), dtype=np.float64)
    nh, nsh = NF, _NH * FULL.L
    anc.hl = aux[:nh].copy()
    anc.sh = aux[nh:nh + nsh].reshape(_NH, FULL.L).copy()
    anc.loss = aux[nh + nsh:nh + nsh + _NH * NF].reshape(_NH, NF).copy()
    anc.nloss = aux[nh + nsh + _NH * NF:].copy()
    anc.n_commit = int(blob["n_commit"])
    anc.sh_T0 = float(blob["sh_T0"])
    anc.n_eval = int(blob["n_eval"])
    anc.uncap_lo, anc.uncap_hi = (float(x) for x in blob["uncap"])
    anc.reset_after = float(blob["reset_after"])
    anc.n_reset = int(blob["n_reset"])
    anc.pending = list(blob.get("pending", ()))
    om = blob.get("om")
    anc.om = None if om is None else np.frombuffer(om, dtype=np.float64).reshape(4, NF).copy()
    anc.v15 = float(blob.get("v15", 1.0))
    return anc


# ================================================================ hierarchy
def join(S: np.ndarray, Ep: np.ndarray, hp: np.ndarray, kappa: np.ndarray,
         loo: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Effective stats of a child with own stats S under a parent with
    effective stats Ep and hyperprior weight hp (arrays [..., L] / [..., 52]).
    D = Ep - S (leave-one-out) or Ep; s = min(1, kappa_eff / size(D)) with
    size = exposure minutes (count, kappa x 15 min), trials (ratio) or rows;
    E = S + s D, h = s hp. An empty parent (size 0) passes its hyperprior on."""
    S = np.asarray(S, dtype=np.float64)
    D = np.asarray(Ep, dtype=np.float64) - S if loo else np.array(Ep, dtype=np.float64)
    D[..., FULL.NONNEG] = np.maximum(D[..., FULL.NONNEG], 0.0)
    if loo:
        # a feature the parent only knows through the child itself has no
        # leave-one-out information: drop all its slots (the signed NIG sum
        # would otherwise keep a residue that its floored weight does not)
        dead = D[..., FULL.W] <= 1e-9 * (1.0 + np.abs(S[..., FULL.W]))
        D = np.where(dead[..., FULL.SLOT_F], 0.0, D)
    size = D[..., FULL.DEN]
    keff = np.asarray(kappa, dtype=np.float64) * _JOIN_UNIT
    with np.errstate(all="ignore"):
        s = np.where(size > 0.0, np.minimum(1.0, keff / size), 1.0)
    return S + s[..., FULL.SLOT_F] * D, s * np.asarray(hp, dtype=np.float64)


def _nanmoments(v: np.ndarray, ok: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(count, mean, ddof-1 variance) per column of v over rows where ok."""
    n = ok.sum(axis=0)
    x = np.where(ok, v, 0.0)
    mean = x.sum(axis=0) / np.maximum(n, 1)
    var = (np.where(ok, (v - mean) ** 2, 0.0)).sum(axis=0) / np.maximum(n - 1, 1)
    return n, mean, var


def eb_kappa(children: Sequence[np.ndarray]) -> np.ndarray:
    """Empirical-Bayes prior strength per feature from children's marginal
    stats [L] (summed over buckets). Moment estimates with the between-child
    variance corrected for each child's sampling noise:
      ratio  kappa = m(1-m)/var_between - 1                  (trials)
      count  kappa = v15 / var_between, v15 = var of a 15-min row's rate (rows)
      t      kappa = mean within var / var_between           (rows)
    clipped to [2, 50]; 50 when the children agree, 2 when < 2 children."""
    out = np.full(NF, KAPPA_DEFAULT)
    if len(children) < 2:
        return out
    X = np.stack([np.asarray(c, dtype=np.float64) for c in children])
    C, R, N = _blocks(X)
    with np.errstate(all="ignore"):
        # counts: per-minute rates of children with >= 1 h of exposure
        se, sx = C[..., 2], C[..., 1]
        ok = se >= 60.0
        n, mbar, var = _nanmoments(sx / se, ok)
        pooled = np.where(ok[..., None], C, 0.0).sum(axis=0)          # nC, 6
        kap = _overdisp(*(pooled[:, k] for k in range(6)))
        v15 = (15.0 * mbar + (15.0 * mbar) ** 2 / kap) / 225.0
        samp = np.where(ok, v15 * 15.0 / se, 0.0).sum(axis=0) / np.maximum(n, 1)
        vb = var - samp
        kc = np.where(vb > 0.0, v15 / vb, KAPPA_MAX)
        out[CNT] = np.where(n >= 2, kc, KAPPA_DEFAULT)
        # ratios: proportions of children with >= 20 trials
        sk, sn = R[..., 1], R[..., 2]
        ok = sn >= 20.0
        n, m, var = _nanmoments(sk / sn, ok)
        vb = var - np.where(ok, m * (1.0 - m) / sn, 0.0).sum(axis=0) / np.maximum(n, 1)
        kr = np.where(vb > 0.0, m * (1.0 - m) / vb - 1.0, KAPPA_MAX)
        out[RAT] = np.where((n >= 2) & (m > 0.0) & (m < 1.0), kr, KAPPA_DEFAULT)
        # t: means of children with >= 4 rows
        W, sy, syy = N[..., 0], N[..., 1], N[..., 2]
        ok = W >= 4.0
        mi = sy / W
        n, _, var = _nanmoments(mi, ok)
        wv = np.where(ok, np.maximum(syy / W - mi * mi, 0.0), 0.0)
        vw = wv.sum(axis=0) / np.maximum(n, 1)
        vb = var - np.where(ok, vw / W, 0.0).sum(axis=0) / np.maximum(n, 1)
        kn = np.where(vb > 0.0, vw / vb, KAPPA_MAX)
        out[NIG] = np.where(n >= 2, kn, KAPPA_DEFAULT)
    out = np.where(np.isfinite(out), out, KAPPA_DEFAULT)
    return np.clip(out, KAPPA_MIN, KAPPA_MAX)


def tier_model(store: Any, s: str, key: str) -> Optional[Mapping[str, Any]]:
    """Published tier model (key 'class:<rid>' | '__system__' | 'org')."""
    m = (store.get_model(ORG[0], ORG[1], MODEL) if key == "org"
         else store.get_model(s, key, MODEL))
    if isinstance(m, Mapping) and m.get("fmt") in FMTS and m.get("E") is not None:
        return m
    return None


def parent_key(store: Any, s: str, e: str) -> str:
    """Backoff parent of a real entity: its role class (>= 3 members in s,
    contract L) or the system tier."""
    return m_class.class_key(store, s, e) or SYSTEM_KEY


# =============================================================== predictives
@dataclass
class Pred:
    """Per-feature predictive for one bucket (arrays [52]; NaN where a
    parameter does not apply to the feature's family)."""
    mu: np.ndarray          # nb: rate per minute
    r: np.ndarray           # nb: size
    p: np.ndarray           # bb: mean
    c: np.ndarray           # bb: concentration (a = p c, b = (1 - p) c)
    df: np.ndarray          # t
    loc: np.ndarray
    scale: np.ndarray
    mean: np.ndarray        # mu | p | loc
    ebar: float = 15.0      # mean row exposure (min) of the bucket
    tier: str = "entity"
    anchor: str = "current"
    bucket: int = -1
    mode: str = "bin48"
    grain: str = "h"                       # spec v2.1
    prov: Optional[np.ndarray] = None      # Q: pi_nat = W_native / (W_native + KAPPA_T)
    scored: Optional[np.ndarray] = None    # Q: False for a 'none' feature below KAPPA_T rows

    @property
    def family(self) -> np.ndarray:
        return FAMILY


def _pred(E: np.ndarray, h: np.ndarray, cell: Optional[np.ndarray] = None, **kw: Any) -> Pred:
    """Pred of one bucket from effective stats E[1, L] and hyperprior weight
    h[1, 52]; with a bin168 cell the location is refined (_refine_par)."""
    par = _params(E, h)
    if cell is not None:
        par = _refine_par(par, E, cell[None, :], np.array([0]))
        kw["mode"] = "bin168"
    mu = np.full(NF, np.nan)
    r, p, c, df, loc, scale = (mu.copy() for _ in range(6))
    mu[CNT], r[CNT] = par.mu[0], par.r[0]
    p[RAT], c[RAT] = par.p[0], par.c[0]
    df[NIG], loc[NIG], scale[NIG] = par.df[0], par.loc[0], par.scale[0]
    return Pred(mu, r, p, c, df, loc, scale, _mean(par)[0], ebar=float(_ebar(E)[0]), **kw)


def _refine_par(par: _Par, E: np.ndarray, cells: np.ndarray, rows: np.ndarray) -> _Par:
    """Hour-of-week location for rows: the bin168 cell's mean shrunk to the
    bin48 mean with K168 pseudo-rows of the bucket's average row; dispersion
    stays that of the bin48 predictive."""
    mean = _mean(par)[rows]
    Er = E[rows]
    Wb = Er[:, FULL.W]
    with np.errstate(all="ignore"):
        dbar = Er[:, FULL.DEN] / Wb
        dbar = np.where((Wb > 0.0) & np.isfinite(dbar) & (dbar > 0.0), dbar,
                        np.where(_IS_NB, 15.0, 1.0))
        m168 = (cells[:, LOC.NUM] + K168 * dbar * mean) / (cells[:, LOC.DEN] + K168 * dbar)
    m168 = np.where(np.isfinite(m168), m168, mean)
    mu, p, loc = par.mu.copy(), par.p.copy(), par.loc.copy()
    mu[rows], p[rows], loc[rows] = m168[:, CNT], m168[:, RAT], m168[:, NIG]
    return par._replace(mu=mu, p=p, loc=loc)


def _typical(tctx: Mapping[str, Any]) -> bool:
    dow = int(tctx.get("dow", 0))
    nwd = tctx.get("day_type") == "nonworkday"
    return nwd == (dow >= 5)


def _anchor_E(anc: Optional[Anchor], b: int, T: Optional[float] = None) -> np.ndarray:
    """True bin48 stats of one bucket [1, L] (zeros for an empty anchor)."""
    if anc is None or anc.empty:
        return np.zeros((1, FULL.L))
    return anc.a48[b:b + 1, :FULL.L] * _scale(anc, anc.T if T is None else T, FULL)


def _cell(anc: Anchor, c: int) -> np.ndarray:
    return anc.a168[c, :LOC.L] * _scale(anc, anc.T, LOC)


def anchor_predictive(anc: Optional[Anchor], tctx: Mapping[str, Any],
                      parent: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
                      anchor: str = "current") -> Pred:
    """Predictive of an anchor at tctx's bucket without the store (B18):
    parent = (E_p[L], h_p[52], kappa[52]) of a backoff tier (no leave-one-out),
    else the hyperprior. bin168 location refinement once the anchor is in
    week mode and the day is typical."""
    b = int(tctx["bin48"])
    S = _anchor_E(anc, b)
    if parent is not None:
        E, h = join(S, np.asarray(parent[0])[None], np.asarray(parent[1])[None],
                    parent[2], loo=False)
    else:
        E, h = S, _H1[:1]
    cell = (_cell(anc, int(tctx["bin168"])) if anc is not None and anc.week_mode()
            and _typical(tctx) else None)
    return _pred(E, h, cell, anchor=anchor, bucket=b)


def _entity_model(store: Any, s: str, e: str, model: Any) -> Optional[Mapping[str, Any]]:
    m = model if model is not None else store.get_model(s, e, MODEL)
    if isinstance(m, Mapping) and m.get("fmt") in FMTS and m.get("tier", "entity") == "entity":
        return m
    return None


def _chain(store: Any, s: str, e: str, b: int, m: Optional[Mapping[str, Any]],
           loo: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Effective current-anchor stats and hyperprior weight of a real entity
    at bucket b: own stats joined (leave-one-out) with its backoff parent."""
    cur = m.get("current") if m is not None else None
    S = _anchor_E(cur, b)
    tm = tier_model(store, s, parent_key(store, s, e))
    if tm is None:
        return S, _H1[:1]
    Sp = S
    if loo and cur is not None and not cur.empty and float(tm["ts"]) != cur.T:
        Sp = S * np.exp2(-(float(tm["ts"]) - cur.T) / cur.hl)[FULL.SLOT_F]
    E, h = join(Sp, tm["E"][b:b + 1], tm["h"][b:b + 1], tm["kappa"], loo=loo)
    if Sp is not S:                      # own stats at the anchor clock, prior from the tier
        E = E - Sp + S
    return E, h


def _week_cell(m: Optional[Mapping[str, Any]], tctx: Mapping[str, Any]) -> Optional[np.ndarray]:
    """The current anchor's bin168 cell at tctx when it is in week mode and
    the day is typical, else None (bin48 only)."""
    cur = m.get("current") if m is not None else None
    if cur is not None and cur.week_mode() and _typical(tctx):
        return _cell(cur, int(tctx["bin168"]))
    return None


def _reference_base(m: Optional[Mapping[str, Any]], b: int) -> np.ndarray:
    g = golden(m) if m is not None else None
    if g is not None:
        return g[b:b + 1]
    return _anchor_E(m.get("reference") if m is not None else None, b)


def predictive(store: Any, s: str, e: str, tctx: Mapping[str, Any], *,
               anchor: str = "current", tier: str = "entity", loo: bool = True,
               model: Any = None) -> Pred:
    """Predictive at tctx's bucket.

    tier 'entity' : the entity's own anchor with backoff (a pseudo-entity key
                    'class:<rid>' / '__system__' gives that tier's predictive);
    tier 'class'  : the entity's backoff parent (class, else system), without
                    the entity's own data when loo;
    tier 'system' / 'org'.
    anchor 'reference': the golden anchor once one exists, else the reference
    statistics, with the current anchor as a KAPPA_REF-row prior (an empty
    reference is the current anchor). Always finite: an empty store is the
    hyperprior."""
    if tier not in TIERS:
        raise ValueError(f"m_baseline.predictive: unknown tier {tier!r}")
    if anchor not in ("current", "reference"):
        raise ValueError(f"m_baseline.predictive: unknown anchor {anchor!r}")
    b = int(tctx["bin48"])
    if tier in ("system", "org") or (tier == "entity" and is_pseudo(e)):
        key = "org" if tier == "org" or e == ORG[1] else SYSTEM_KEY if tier == "system" else e
        tm = tier_model(store, s, key)
        if tm is None:
            return _pred(np.zeros((1, FULL.L)), _H1[:1], tier=tier, anchor=anchor, bucket=b)
        return _pred(tm["E"][b:b + 1], tm["h"][b:b + 1], tier=tier, anchor=anchor, bucket=b)
    m = _entity_model(store, s, e, model)
    if tier == "class":
        tm = tier_model(store, s, parent_key(store, s, e))
        if tm is None:
            return _pred(np.zeros((1, FULL.L)), _H1[:1], tier=tier, anchor=anchor, bucket=b)
        if loo and m is not None:
            E, h = _loo_parent(tm, b, m)
        else:
            E, h = tm["E"][b:b + 1], tm["h"][b:b + 1]
        return _pred(E, h, tier=tier, anchor=anchor, bucket=b)
    E, h = _chain(store, s, e, b, m, loo)
    if anchor == "current":
        return _pred(E, h, _week_cell(m, tctx), tier=tier, anchor=anchor, bucket=b)
    E2, h2 = join(_reference_base(m, b), E, h, np.full(NF, KAPPA_REF), loo=False)
    return _pred(E2, h2, tier=tier, anchor=anchor, bucket=b)


def _chain_own(m: Mapping[str, Any], b: int, T: float) -> np.ndarray:
    return _anchor_E(m.get("current"), b, T)


def _loo_parent(tm: Mapping[str, Any], b: int, m: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """The parent tier's effective stats without the entity's own data."""
    own = _chain_own(m, b, float(tm["ts"]))
    D = tm["E"][b:b + 1] - own
    D[..., FULL.NONNEG] = np.maximum(D[..., FULL.NONNEG], 0.0)
    dead = D[..., FULL.W] <= 1e-9 * (1.0 + np.abs(own[..., FULL.W]))
    return np.where(dead[..., FULL.SLOT_F], 0.0, D), tm["h"][b:b + 1]


def predictive_set(store: Any, s: str, e: str, tctx: Mapping[str, Any],
                   model: Any = None) -> Dict[str, Pred]:
    """current, reference (both with backoff) and class (leave-one-out
    parent) predictives of a real entity, sharing the chain computation."""
    b = int(tctx["bin48"])
    m = _entity_model(store, s, e, model)
    E, h = _chain(store, s, e, b, m, True)
    cur_pr = _pred(E, h, _week_cell(m, tctx), tier="entity", anchor="current", bucket=b)
    E2, h2 = join(_reference_base(m, b), E, h, np.full(NF, KAPPA_REF), loo=False)
    ref_pr = _pred(E2, h2, tier="entity", anchor="reference", bucket=b)
    tm = tier_model(store, s, parent_key(store, s, e))
    if tm is None:
        cls = _pred(np.zeros((1, FULL.L)), _H1[:1], tier="class", bucket=b)
    elif m is not None:
        D, hp = _loo_parent(tm, b, m)
        cls = _pred(D, hp, tier="class", bucket=b)
    else:
        cls = _pred(tm["E"][b:b + 1], tm["h"][b:b + 1], tier="class", bucket=b)
    return {"current": cur_pr, "reference": ref_pr, "class": cls}


# ------------------------------------------------ batched predictives (perf)
# B04 builds the predictives of every scored entity of a tick at once. The
# assembly of the effective statistics (chain, join, leave-one-out, bin168
# cell) is per entity exactly as above; the parameter maps (_params, _ebar,
# _mean, _refine_par) are element-wise per row, so evaluating the stacked
# rows in one call gives the same numbers as one _pred call per row.
def _preds_from_rows(E: np.ndarray, h: np.ndarray, cells: Sequence[Optional[np.ndarray]],
                     kws: Sequence[Dict[str, Any]]) -> List[Pred]:
    """[_pred(E[i:i+1], h[i:i+1], cells[i], **kws[i]) for every row i]."""
    par = _params(E, h)
    rows = [i for i, c in enumerate(cells) if c is not None]
    if rows:
        par = _refine_par(par, E, np.stack([cells[i] for i in rows]), np.array(rows))
    mean = _mean(par)
    ebar = _ebar(E)
    out = []
    for i, kw in enumerate(kws):
        mu = np.full(NF, np.nan)
        r, p, c, df, loc, scale = (mu.copy() for _ in range(6))
        mu[CNT], r[CNT] = par.mu[i], par.r[i]
        p[RAT], c[RAT] = par.p[i], par.c[i]
        df[NIG], loc[NIG], scale[NIG] = par.df[i], par.loc[i], par.scale[i]
        if cells[i] is not None:
            kw = dict(kw, mode="bin168")
        out.append(Pred(mu, r, p, c, df, loc, scale, mean[i], ebar=float(ebar[i]), **kw))
    return out


def _set_rows(store: Any, s: str, e: str, tctx: Mapping[str, Any]) -> Tuple[Any, ...]:
    """predictive_set's per-entity assembly: (E, h, cell) of current,
    reference and class, and the entity model."""
    b = int(tctx["bin48"])
    m = _entity_model(store, s, e, None)
    E, h = _chain(store, s, e, b, m, True)
    E2, h2 = join(_reference_base(m, b), E, h, np.full(NF, KAPPA_REF), loo=False)
    tm = tier_model(store, s, parent_key(store, s, e))
    if tm is None:
        Ec, hc = np.zeros((1, FULL.L)), _H1[:1]
    elif m is not None:
        Ec, hc = _loo_parent(tm, b, m)
    else:
        Ec, hc = tm["E"][b:b + 1], tm["h"][b:b + 1]
    return b, m, E, h, _week_cell(m, tctx), E2, h2, Ec, hc


def predictive_set_many(store: Any, items: Sequence[Tuple[str, str, Mapping[str, Any]]]
                        ) -> List[Dict[str, Pred]]:
    """[predictive_set(store, s, e, tctx) for (s, e, tctx) in items], with
    the parameter maps of all 3 x len(items) predictives in one pass."""
    if not items:
        return []
    Es, hs, cells, kws = [], [], [], []
    for s, e, tctx in items:
        b, m, E, h, cell, E2, h2, Ec, hc = _set_rows(store, s, e, tctx)
        Es += [E, E2, Ec]
        hs += [h, h2, hc]
        cells += [cell, None, None]
        kws += [dict(tier="entity", anchor="current", bucket=b),
                dict(tier="entity", anchor="reference", bucket=b),
                dict(tier="class", bucket=b)]
    P = _preds_from_rows(np.concatenate(Es), np.concatenate(hs), cells, kws)
    return [{"current": P[3 * i], "reference": P[3 * i + 1], "class": P[3 * i + 2]}
            for i in range(len(items))]


def predictive_q_many(store: Any, items: Sequence[Tuple[str, str, Mapping[str, Any]]]
                      ) -> List[Dict[str, Pred]]:
    """[predictive_q(store, s, e, tctx) for (s, e, tctx) in items], with the
    H predictives and the Q parameter maps of all items in one pass each."""
    if not items:
        return []
    Es, hs, cells, kws, ms, bs = [], [], [], [], [], []
    for s, e, tctx in items:
        b = int(tctx["bin48"])
        m = _entity_model(store, s, e, None)
        E, h = _chain(store, s, e, b, m, True)
        E2, h2 = join(_reference_base(m, b), E, h, np.full(NF, KAPPA_REF), loo=False)
        Es += [E, E2]
        hs += [h, h2]
        cells += [_week_cell(m, tctx), None]
        kws += [dict(tier="entity", anchor="current", bucket=b),
                dict(tier="entity", anchor="reference", bucket=b)]
        ms.append(m)
        bs.append(b)
    PH = _preds_from_rows(np.concatenate(Es), np.concatenate(hs), cells, kws)
    EQ, SQ, t_refs, W_nats = [], [], [], []
    for i, (s, e, tctx) in enumerate(items):
        cur_h, ref_h = PH[2 * i], PH[2 * i + 1]
        v, da, db = omega_chain(store, s, e, ms[i])
        var_c = GR.t_variance(cur_h.scale, cur_h.df)
        delta_c = da + db * np.where(_JENSEN, -(v - 1.0) * var_c / 2.0, 0.0)
        var_r = GR.t_variance(ref_h.scale, ref_h.df)
        delta_r = da + db * np.where(_JENSEN, -(v - 1.0) * var_r / 2.0, 0.0)
        t_cur = transfer_pred(cur_h, v, np.where(np.isfinite(delta_c), delta_c, 0.0))
        t_refs.append(transfer_pred(ref_h, v, np.where(np.isfinite(delta_r), delta_r, 0.0)))
        S_Q = _anchor_E(q_anchor(ms[i]), bs[i])
        W_nats.append(np.maximum(S_Q[0, FULL.W], 0.0))
        EQ.append(S_Q + pseudo_stats(t_cur, _KT))
        SQ.append(S_Q)
    n = len(items)
    par_t = _params(np.concatenate(EQ), np.zeros((n, NF)))
    par_n = _params(np.concatenate(SQ), np.broadcast_to(_H1[:1], (n, NF)))
    nC_, nR_, nN_ = _NONE[CNT][None, :], _NONE[RAT][None, :], _NONE[NIG][None, :]
    par = _Par(np.where(nC_, par_n.mu, par_t.mu), np.where(nC_, par_n.r, par_t.r),
               np.where(nR_, par_n.p, par_t.p), np.where(nR_, par_n.c, par_t.c),
               np.where(nN_, par_n.df, par_t.df), np.where(nN_, par_n.loc, par_t.loc),
               np.where(nN_, par_n.scale, par_t.scale))
    mean = _mean(par)
    span = ~_TRANSFERABLE & ~_NONE
    out = []
    for i in range(n):
        mu = np.full(NF, np.nan)
        r, p, c, df, loc, scale = (mu.copy() for _ in range(6))
        mu[CNT], r[CNT] = par.mu[i], par.r[i]
        p[RAT], c[RAT] = par.p[i], par.c[i]
        df[NIG], loc[NIG], scale[NIG] = par.df[i], par.loc[i], par.scale[i]
        W_nat = W_nats[i]
        prov = W_nat / (W_nat + _KT)
        scored = np.where(_NONE, W_nat >= _KT, ~span)
        cur = Pred(mu, r, p, c, df, loc, scale, mean[i], ebar=15.0, tier="entity",
                   anchor="current", bucket=bs[i], mode="bin48", grain="q", prov=prov,
                   scored=scored)
        t_ref = t_refs[i]
        t_ref.prov, t_ref.scored = prov, scored
        out.append({"current": cur, "reference": t_ref})
    return out


# ------------------------------------------------------ scoring / quantiles
def midp(pred: Pred, nat: Sequence[float], dt_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """(u, p) per feature: mid-distribution u = P(X < x) + P(X = x)/2 and the
    two-sided mid-p (lib/bayes), NaN where the feature is not observed
    (ratio with n = 0, NaN input): never p = 1 for missing data."""
    num, den, wf = observe(nat, dt_s)
    u = np.full(NF, np.nan)
    p = np.full(NF, np.nan)
    ok = wf[CNT] > 0.0
    if ok.any():
        i = CNT[ok]
        uu, pp = bayes.nb_midp(num[i], pred.mu[i] * den[i], pred.r[i])
        u[i], p[i] = uu, pp
    for f in RAT:
        if wf[f] > 0.0:
            a, bb = pred.p[f] * pred.c[f], (1.0 - pred.p[f]) * pred.c[f]
            u[f], p[f] = bayes.bb_midp(float(num[f]), float(den[f]), float(a), float(bb))
    ok = wf[NIG] > 0.0
    if ok.any():
        i = NIG[ok]
        with np.errstate(all="ignore"):
            z = (num[i] - pred.loc[i]) / pred.scale[i]
            lo = sp.stdtr(pred.df[i], z)
            hi = sp.stdtr(pred.df[i], -z)
        good = np.isfinite(z) & (pred.scale[i] > 0.0)
        u[i] = np.where(good, lo, np.nan)
        p[i] = np.where(good, np.clip(2.0 * np.minimum(lo, hi), bayes.P_FLOOR, 1.0), np.nan)
    return u, p


def loglik(pred: Pred, nat: Sequence[float], dt_s: float) -> np.ndarray:
    """Log predictive density (nats) of the observed row per feature (pmf for
    nb / bb, Student-t pdf of the transformed value for t); NaN unobserved."""
    num, den, wf = observe(nat, dt_s)
    out = np.full(NF, np.nan)
    ok = wf[CNT] > 0.0
    if ok.any():
        i = CNT[ok]
        pm = bayes.nb_pmf(num[i], pred.mu[i] * den[i], pred.r[i])
        out[i] = np.log(np.maximum(pm, bayes.P_FLOOR))
    for f in RAT:
        if wf[f] > 0.0:
            a, bb = pred.p[f] * pred.c[f], (1.0 - pred.p[f]) * pred.c[f]
            out[f] = float(bayes.bb_logpmf(np.floor(num[f] + 0.5), den[f], a, bb))
    ok = wf[NIG] > 0.0
    if ok.any():
        i = NIG[ok]
        df, sc = pred.df[i], pred.scale[i]
        with np.errstate(all="ignore"):
            z = (num[i] - pred.loc[i]) / sc
            out[i] = (sp.gammaln((df + 1.0) / 2.0) - sp.gammaln(df / 2.0)
                      - 0.5 * np.log(df * np.pi) - np.log(sc)
                      - (df + 1.0) / 2.0 * np.log1p(z * z / df))
    return out


def _inverse_tx(y: np.ndarray, dt_s: float) -> np.ndarray:
    """t-family values back to natural units (per-dt bytes, ms, shares, ...).

    `y[..., j]` is the j-th t-family column (the NIG block, in NIG order):
    _TX_SETS holds positions inside that block, not feature indices. Use
    _inverse_tx_full for a full 52-column array."""
    out = np.array(y, dtype=np.float64, copy=True)
    with np.errstate(all="ignore"):
        for code, pos in _TX_SETS.items():
            if not pos.size:
                continue
            v = out[..., pos]
            if code == TX_BYTES:
                v = np.expm1(v) * dt_s / 60.0
            elif code == TX_LOG:
                v = np.exp(v)
            elif code == TX_LOG1P:
                v = np.expm1(v)
            elif code == TX_LOGIT:
                v = sp.expit(v)
            out[..., pos] = v                  # TX_ID, TX_CLR: identity
    return out


def _inverse_tx_full(X: np.ndarray, dt_s: float) -> np.ndarray:
    """Full [..., 52] array: the NIG columns back to natural units, every
    other column unchanged."""
    out = np.array(X, dtype=np.float64, copy=True)
    out[..., NIG] = _inverse_tx(out[..., NIG], dt_s)
    return out


def _typical_n(pred: Pred, e_min: float) -> np.ndarray:
    n = pred.mu[RATIO_N_IDX] * e_min
    return np.where(np.isfinite(n) & (n >= 1.0), np.round(n), 1.0)


def quantiles(pred: Pred, qs: Sequence[float], dt_s: float = 900.0,
              nat: Optional[Sequence[float]] = None) -> np.ndarray:
    """Predictive quantiles [len(qs), 52] in natural units for an exposure of
    dt_s: counts per dt, ratios as fractions at n (from nat when given and
    > 0, else the expected trials of its exposure feature), bytes per dt,
    averages in their units, bounded / shares in [0, 1], CLR in log-ratio units."""
    q = np.asarray(qs, dtype=np.float64).reshape(-1)
    e = float(dt_s) / 60.0
    out = np.full((q.size, NF), np.nan)
    mean = pred.mu[CNT] * e
    out[:, CNT] = bayes.nb_ppf(q[:, None], mean[None, :], pred.r[CNT][None, :])
    n = _typical_n(pred, e)
    if nat is not None:
        x = np.asarray(nat, dtype=np.float64).reshape(-1)[RATIO_N_IDX]
        n = np.where(np.isfinite(x) & (x > 0.0), x, n)
    # one call for every (q, ratio feature): bb_ppf groups the quantiles that
    # share (n, a, b) into one pmf pass, and returns NaN where a, b are invalid
    a = pred.p[RAT] * pred.c[RAT]
    b = (1.0 - pred.p[RAT]) * pred.c[RAT]
    out[:, RAT] = np.asarray(bayes.bb_ppf(q[:, None], n[None, :], a[None, :], b[None, :]),
                             dtype=np.float64).reshape(q.size, RAT.size) / n[None, :]
    with np.errstate(all="ignore"):
        t = sp.stdtrit(pred.df[NIG][None, :], np.clip(q, 0.0, 1.0)[:, None])
        y = pred.loc[NIG][None, :] + pred.scale[NIG][None, :] * t
    out[:, NIG] = _inverse_tx(y, dt_s)
    return out


def quantiles_many(preds: Sequence[Pred], qs: Sequence[float],
                   dt_s: Sequence[float]) -> np.ndarray:
    """quantiles(preds[i], qs, dt_s[i]) for every i at once, [len(preds),
    len(qs), 52]: the same element-wise formulas on stacked parameters (the
    NB integer search and the grouped BB pmf passes are per element, so the
    values are those of the one-by-one calls). B04 model_state (perf)."""
    q = np.asarray(qs, dtype=np.float64).reshape(-1)
    m = len(preds)
    out = np.full((m, q.size, NF), np.nan)
    if m == 0:
        return out
    dts = np.asarray(dt_s, dtype=np.float64).reshape(-1)
    e = dts / 60.0
    mu = np.stack([p.mu for p in preds])
    r = np.stack([p.r for p in preds])
    pp = np.stack([p.p for p in preds])
    cc = np.stack([p.c for p in preds])
    df = np.stack([p.df for p in preds])
    loc = np.stack([p.loc for p in preds])
    scale = np.stack([p.scale for p in preds])
    mean = mu[:, CNT] * e[:, None]
    out[:, :, CNT] = bayes.nb_ppf(q[None, :, None], mean[:, None, :], r[:, CNT][:, None, :])
    n = mu[:, RATIO_N_IDX] * e[:, None]
    n = np.where(np.isfinite(n) & (n >= 1.0), np.round(n), 1.0)
    a = pp[:, RAT] * cc[:, RAT]
    b = (1.0 - pp[:, RAT]) * cc[:, RAT]
    out[:, :, RAT] = np.asarray(bayes.bb_ppf(q[None, :, None], n[:, None, :], a[:, None, :],
                                             b[:, None, :]), dtype=np.float64) / n[:, None, :]
    with np.errstate(all="ignore"):
        t = sp.stdtrit(df[:, NIG][:, None, :], np.clip(q, 0.0, 1.0)[None, :, None])
        y = loc[:, NIG][:, None, :] + scale[:, NIG][:, None, :] * t
    out[:, :, NIG] = _inverse_tx(y, dts[:, None, None])
    return out


def mean_nat(pred: Pred, dt_s: float = 900.0) -> np.ndarray:
    """Predictive centre in natural units: count mean per dt, ratio p, t
    features the inverse transform of the location (a median for monotone
    transforms)."""
    out = np.array(pred.mean, dtype=np.float64, copy=True)
    out[CNT] = pred.mu[CNT] * dt_s / 60.0
    out[NIG] = _inverse_tx(pred.loc[NIG], dt_s)
    return out


def sd15(pred: Pred) -> np.ndarray:
    """Predictive sd for a 15-minute exposure in mean units (the cap's sigma15)."""
    par = _Par(pred.mu[None, CNT], pred.r[None, CNT], pred.p[None, RAT], pred.c[None, RAT],
               pred.df[None, NIG], pred.loc[None, NIG], pred.scale[None, NIG])
    return _sd15_par(par, np.array([pred.ebar]))[0]


def _vec_med_sd(par: _Par, ebar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s15 = _sd15_par(par, ebar)
    with np.errstate(all="ignore"):
        mu = par.mu
        pp = np.clip(par.p, 1e-6, 1.0 - 1e-6)
        med = _feat(np.log1p(mu), np.log(pp) - np.log1p(-pp), par.loc)
        sd = _feat(s15[:, CNT] / (1.0 + mu), s15[:, RAT] / (pp * (1.0 - pp)), par.scale)
    return med, sd


def vec_median_sd(pred: Pred) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate median and sd in FEATURE_SPEC vec space (legacy
    profile.baseline_median / baseline_mad): counts log1p(rate per min),
    ratios logit(p) with the sd at the 15-min trials, t features (loc, scale)."""
    par = _Par(pred.mu[None, CNT], pred.r[None, CNT], pred.p[None, RAT], pred.c[None, RAT],
               pred.df[None, NIG], pred.loc[None, NIG], pred.scale[None, NIG])
    med, sd = _vec_med_sd(par, np.array([pred.ebar]))
    return med[0], sd[0]


def profile_many(store: Any, s: str, ents: Sequence[str], models: Sequence[Mapping[str, Any]],
                 tctx: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """vec-space (median, sd) [n, 52] of the current predictive (with
    backoff, leave-one-out) of several entities of system s at tctx's
    bucket, vectorised: what predictive() + vec_median_sd() give per entity."""
    b = int(tctx["bin48"])
    n = len(ents)
    S = np.zeros((n, FULL.L))
    E = np.zeros((n, FULL.L))
    h = np.ones((n, NF))
    curs: List[Optional[Anchor]] = []
    groups: Dict[str, List[int]] = {}
    for i, (e, m) in enumerate(zip(ents, models)):
        cur = _anchor_of(m, "current")
        curs.append(cur)
        S[i] = _anchor_E(cur, b)[0]
        groups.setdefault(parent_key(store, s, e), []).append(i)
    for key, idx in groups.items():
        tm = tier_model(store, s, key)
        if tm is None:
            E[idx] = S[idx]
            continue
        tts = float(tm["ts"])
        Sd = S[idx].copy()
        for j, i in enumerate(idx):
            c = curs[i]
            if c is not None and not c.empty and c.T != tts:
                Sd[j] *= np.exp2(-(tts - c.T) / c.hl)[FULL.SLOT_F]
        Eg, hg = join(Sd, tm["E"][b][None, :], tm["h"][b][None, :], tm["kappa"], loo=True)
        E[idx] = Eg - Sd + S[idx]
        h[idx] = hg
    par = _params(E, h)
    wk = [i for i, c in enumerate(curs) if c is not None and c.week_mode()] if _typical(tctx) else []
    if wk:
        cells = np.stack([_cell(curs[i], int(tctx["bin168"])) for i in wk])
        par = _refine_par(par, E, cells, np.asarray(wk))
    return _vec_med_sd(par, _ebar(E))


def bucket_means(anc: Optional[Anchor], T: Optional[float] = None, v: Optional[float] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """(mean[48, 52], sigma15[48, 52]) of every bin48 bucket from the anchor's
    own statistics and the hyperprior (what the rate cap sees). spec v2.1:
    `v` (e.g. an H anchor's v15) is the Q transfer factor of sigma15; None =
    the anchor's own-grain sigma (v = 1)."""
    St = true_stats(anc, T)
    if St is None:
        St = np.zeros((48, FULL.L))
    par = _params(St, _H1)
    vv = None if v is None or float(v) == 1.0 else np.full(48, float(v))
    return _mean(par), _sd15_par(par, _ebar(St), vv)


# ===================================================================== model
def _anchor_of(x: Any, key: str = "current") -> Optional[Anchor]:
    if isinstance(x, Anchor):
        return x
    if isinstance(x, Mapping):
        a = x.get(key)
        return a if isinstance(a, Anchor) else None
    return None


def n_eff(x: Any, key: str = "current") -> float:
    """Decayed (14-d half-life) trust-weighted count of committed active rows."""
    anc = _anchor_of(x, key)
    if anc is None or anc.empty:
        return 0.0
    k = int(np.argmin(np.abs(HL_CAND_S - HL_DEFAULT_DAYS * DAY)))
    v = anc.sh[k, FULL.W[_FLOWS]] * 2.0 ** (-(anc.T - anc.sh_T0) / HL_CAND_S[k])
    return float(v) if math.isfinite(v) else 0.0


def hl_days(model: Any) -> np.ndarray:
    anc = _anchor_of(model)
    return (anc.hl if anc is not None else np.full(NF, HL_DEFAULT_DAYS * DAY)) / DAY


def n_eff_by_bucket(x: Any, anchor: str = "current", T: Optional[float] = None) -> np.ndarray:
    """float[48]: decayed committed active rows per bin48 bucket (the
    exposure-weight row count of the flows channel, i.e. of every active
    row) of a model.baseline or an Anchor; zeros when empty (B30, R22.1)."""
    anc = _anchor_of(x, anchor)
    St = true_stats(anc, T)
    if St is None:
        return np.zeros(48)
    return np.maximum(St[:, FULL.W_EXPO], 0.0)


def own_support(model: Any, tctx: Mapping[str, Any], anchor: str = "current",
                T: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """The entity's OWN evidence at tctx's bin48 bucket (no backoff):
    (W[52] decayed committed rows per feature, weighted exposure minutes of
    the bucket). Zeros for an empty / missing anchor. Consumers that treat a
    residual as N(0, 1) against the entity's own model (B14's charts) use it
    to tell an identified bucket from a pure backoff / hyperprior one."""
    anc = _anchor_of(model, anchor)
    if anc is None or anc.empty:
        return np.zeros(NF), 0.0
    E = _anchor_E(anc, int(tctx["bin48"]), T)[0]
    return np.maximum(E[FULL.W], 0.0), float(max(E[FULL.EXPO], 0.0))


def version(model: Any) -> int:
    return int(model.get("version", 0)) if isinstance(model, Mapping) else 0


def _gate(model: Any) -> Any:
    return model.get("gate") if isinstance(model, Mapping) else None


def held(model: Any) -> List[float]:
    """Timestamps of the current-anchor rows held for the governor."""
    g = _gate(model)
    return [float(r.ts) for r in getattr(g, "held", [])] if g is not None else []


def last_commit_ts(model: Any) -> float:
    """Newest committed row of the current anchor (NaN if none)."""
    g = _gate(model)
    j = getattr(g, "journal", None) if g is not None else None
    return float(j[-1].ts) if j else math.nan


def golden(model: Any) -> Optional[np.ndarray]:
    """Golden anchor statistics [48, L] (median of up to 4 low-risk weekly
    reference snapshots) or None."""
    if not isinstance(model, Mapping):
        return None
    g = model.get("golden")
    st = g.get("stats") if isinstance(g, Mapping) else None
    return st if isinstance(st, np.ndarray) else None


def has_golden(model: Any) -> bool:
    return golden(model) is not None


def golden_offset(store: Any, s: str, e: str) -> Optional[np.ndarray]:
    """Offset (in reference sigmas) of the anchor zr is measured against
    relative to golden: zeros once a golden exists (the reference predictive
    is the golden anchor), None before (nothing to offset)."""
    return np.zeros(NF) if has_golden(store.get_model(s, e, MODEL)) else None


def anchor_summary(model: Any, anchor: str, feature: Any) -> Tuple[float, float]:
    """(mean, sigma15) of one feature in an anchor ('current' | 'reference' |
    'golden'), averaged over the bin48 buckets weighted by their data (mean
    units: count rate / min, ratio p, t transformed value). Accepts a live
    model.baseline or a plain-data snapshot of it (Anchor._asdict fields, as
    the eval runner stores them). (NaN, NaN) when the anchor holds no data.
    Used by the eval poisoning gate (docs/lib3/eval.md)."""
    f = F.FEATURE_INDEX[feature] if isinstance(feature, str) else int(feature)
    nan = (math.nan, math.nan)
    if not isinstance(model, Mapping):
        return nan
    if anchor == "golden":
        g = model.get("golden")
        st = g.get("stats") if isinstance(g, Mapping) else None
        if st is None:
            return nan
        St = np.asarray(st, dtype=np.float64).reshape(48, FULL.L)
        par = _params(St, _H1)
        mean, sd, W = _mean(par), _sd15_par(par, _ebar(St)), St[:, FULL.W]
    else:
        a = model.get(anchor)
        d = a._asdict() if isinstance(a, Anchor) else a
        if not isinstance(d, Mapping) or d.get("mean") is None:
            return nan
        mean = np.asarray(d["mean"], dtype=np.float64).reshape(48, NF)
        sd = np.asarray(d["sd15"], dtype=np.float64).reshape(48, NF)
        W = (np.asarray(d["W"], dtype=np.float64).reshape(48, NF) if d.get("W") is not None
             else np.ones((48, NF)))
    w = np.where(np.isfinite(mean[:, f]) & np.isfinite(sd[:, f]) & (W[:, f] > 0.0), W[:, f], 0.0)
    if not w.sum() > 0.0:
        return nan
    return (float(np.sum(w * np.nan_to_num(mean[:, f])) / w.sum()),
            float(np.sum(w * np.nan_to_num(sd[:, f])) / w.sum()))


def median_select(snaps: Sequence[np.ndarray]) -> np.ndarray:
    """Cell-wise median of snapshots [k][48, L]: per (bucket, feature) the
    snapshot with the median posterior mean (the two middle ones averaged for
    even k), keeping each cell a coherent set of sufficient statistics."""
    X = np.stack([np.asarray(x, dtype=np.float64) for x in snaps])        # k, 48, L
    k = X.shape[0]
    with np.errstate(all="ignore"):
        mean = (_P + X[..., FULL.NUM]) / (_Q + X[..., FULL.DEN])          # k, 48, 52
    mean = np.where(np.isfinite(mean), mean, 0.0)
    order = np.argsort(mean, axis=0, kind="stable")
    lo = order[(k - 1) // 2][:, FULL.SLOT_F]                              # 48, L
    hi = order[k // 2][:, FULL.SLOT_F]
    a = np.take_along_axis(X, lo[None], axis=0)[0]
    b = np.take_along_axis(X, hi[None], axis=0)[0]
    return 0.5 * (a + b)


def maturity(model: Any) -> Dict[str, Any]:
    """profile.extra.maturity: committed evidence and anchor state."""
    cur = _anchor_of(model, "current")
    ref = _anchor_of(model, "reference")
    ne = n_eff(cur)
    span = (cur.T - cur.t_first) / DAY if cur is not None and not cur.empty else 0.0
    stage = "cold" if ne < 48.0 else "warming" if ne < 96.0 else "mature"
    hl = hl_days(model)
    return {"n_eff": round(ne, 2), "span_days": round(float(span), 3),
            "mode": "bin168" if cur is not None and cur.week_mode() else "bin48",
            "hl_days": {str(int(d)): int(np.sum(hl == d)) for d in HL_CAND_DAYS},
            "ref_n_eff": round(n_eff(ref), 2), "golden": has_golden(model),
            "stage": stage, "version": version(model)}


def seasonal_curves(model: Any, day_type: str, names: Sequence[str]) -> Dict[str, List[float]]:
    """{name: [24 vec-space medians]} for one day type from the entity's own
    current statistics and the hyperprior (legacy profile.seasonal)."""
    cur = _anchor_of(model, "current")
    St = true_stats(cur)
    if St is None:
        return {}
    off = 24 if day_type == "nonworkday" else 0
    M = _mean(_params(St[off:off + 24], _H1[:24]))
    out: Dict[str, List[float]] = {}
    with np.errstate(all="ignore"):
        for name in names:
            f = F.FEATURE_INDEX.get(name)
            if f is None:
                continue
            if FAMILY[f] == FAM_NB:
                v = np.log1p(M[:, f])
            elif FAMILY[f] == FAM_BB:
                pp = np.clip(M[:, f], 1e-6, 1 - 1e-6)
                v = np.log(pp) - np.log1p(-pp)
            else:
                v = M[:, f]
            out[name] = [float(x) if math.isfinite(x) else 0.0 for x in v]
    return out


def descriptors(model: Any, names: Optional[Sequence[str]] = None,
                dt_s: float = 900.0) -> Dict[str, Any]:
    """Portrait / peer descriptors: per feature the 24-hour predictive means
    in natural units for each day type (own statistics + hyperprior)."""
    cur = _anchor_of(model, "current")
    names = list(names) if names is not None else list(F.KEY_FEATURES)
    out: Dict[str, Any] = {"n_eff": n_eff(cur), "features": {}}
    St = true_stats(cur)
    if St is None:
        return out
    M = _mean(_params(St, _H1))
    Mn = _inverse_tx_full(M, dt_s)
    for name in names:
        f = F.FEATURE_INDEX[name]
        v = M[:, f] * dt_s / 60.0 if FAMILY[f] == FAM_NB else Mn[:, f]
        vals = [float(x) if math.isfinite(x) else None for x in v]
        out["features"][name] = {"workday": vals[:24], "nonworkday": vals[24:]}
    return out


# ============================================================================
# spec v2.1: Q grain (docs/lib3/cadence.md §6)
# ============================================================================
Q_KEY = "q"
_TRANSFER = np.array([F.TRANSFER[n] or "-" for n in F.FEATURE_NAMES_V2])
_JENSEN = _TRANSFER == "jensen"
_NONE = _TRANSFER == "none"
_TRANSFERABLE = ~_NONE & (_TRANSFER != "-")
_KT = GR.KAPPA_T


def q_anchor(model: Any) -> Optional[Anchor]:
    """The entity's Q current anchor (model['q']['current']) or None."""
    if not isinstance(model, Mapping):
        return None
    q = model.get(Q_KEY)
    a = q.get("current") if isinstance(q, Mapping) else None
    return a if isinstance(a, Anchor) else None


def omega_chain(store: Any, s: str, e: str, model: Any = None
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(v[52], delta_a[52], delta_b[52]) of a real entity: the EB chain org ->
    system -> class (>= 3 members) -> entity over the paired-hour sums
    (cadence.md §6.3), omega at the root = M (v = M: independent quarters).
    The location shift at bucket b is delta = delta_a + delta_b * default(b)
    (the family default is the chain's root, and the recursion is linear
    in it)."""
    om_nodes: List[np.ndarray] = []
    org = tier_model(store, s, "org")
    for tm in (org, tier_model(store, s, SYSTEM_KEY)):
        if tm is not None and tm.get("omega") is not None:
            om_nodes.append(np.asarray(tm["omega"], dtype=np.float64))
    pk = parent_key(store, s, e)
    if pk != SYSTEM_KEY:
        tm = tier_model(store, s, pk)
        if tm is not None and tm.get("omega") is not None:
            om_nodes.append(np.asarray(tm["omega"], dtype=np.float64))
    m = _entity_model(store, s, e, model)
    own = true_om(m.get("current")) if m is not None and isinstance(m.get("current"), Anchor) else None
    if own is not None:
        om_nodes.append(own)
    omega = np.full(NF, float(GR.M))
    da = np.zeros(NF)
    db = np.ones(NF)
    K = GR.KAPPA_OMEGA
    for om in om_nodes:
        A, Bs, D, W = om[0], om[1], om[2], om[3]
        omega = GR.omega_eb(A, Bs, W, omega)
        Wp = np.maximum(W, 0.0)
        da = (D + K * da) / (Wp + K)
        db = K * db / (Wp + K)
    return GR.v_from_omega(omega), da, db


def transfer_pred(pred: Pred, v: np.ndarray, delta: np.ndarray) -> Pred:
    """The H predictive transferred to the Q grain (cadence.md §6.2): NB
    r / v, BB (1 + c) / v - 1 (>= C_T_MIN), t loc + delta, scale sqrt(v);
    same means. `none` / span features are left as they are."""
    mu, r, p, c = pred.mu.copy(), pred.r.copy(), pred.p.copy(), pred.c.copy()
    df, loc, scale = pred.df.copy(), pred.loc.copy(), pred.scale.copy()
    v = np.asarray(v, dtype=np.float64)
    i = CNT[_TRANSFERABLE[CNT]]
    _, r[i] = GR.transfer_nb(mu[i], r[i], v[i])
    i = RAT[_TRANSFERABLE[RAT]]
    _, c[i] = GR.transfer_bb(p[i], c[i], v[i])
    i = NIG[_TRANSFERABLE[NIG]]
    loc[i], scale[i], _ = GR.transfer_t(loc[i], scale[i], df[i], v[i], np.asarray(delta)[i])
    mean = np.array(pred.mean, dtype=np.float64, copy=True)
    mean[NIG] = loc[NIG]
    return Pred(mu, r, p, c, df, loc, scale, mean, ebar=15.0, tier=pred.tier,
                anchor=pred.anchor, bucket=pred.bucket, mode=pred.mode, grain="q")


def pseudo_stats(pred: Pred, kappa: float = _KT) -> np.ndarray:
    """kappa conjugate pseudo-rows [1, FULL.L] whose statistics reproduce the
    predictive `pred` at a 15-min exposure (cadence.md §6.2): NB exposure 15
    min with the moments of NB(15 mu, r) (so the moment overdispersion is r),
    BB n_bar = 15 mu_n trials with phi-hat = c, t sums whose h = 0 posterior
    has pred's loc and scale."""
    out = np.zeros((1, FULL.L))
    C, R, N = _blocks(out)
    k = float(kappa)
    with np.errstate(all="ignore"):
        m15 = 15.0 * pred.mu[CNT]
        C[0, :, 0] = k
        C[0, :, 1] = k * m15
        C[0, :, 2] = 15.0 * k
        C[0, :, 3] = k * (m15 * m15 + m15 + m15 * m15 / pred.r[CNT])
        C[0, :, 4] = 15.0 * k * m15
        C[0, :, 5] = 225.0 * k
        nbar = 15.0 * pred.mu[RATIO_N_IDX]
        nbar = np.where(np.isfinite(nbar) & (nbar > 1.5), nbar, 1.5)
        pp = np.clip(pred.p[RAT], 1e-6, 1.0 - 1e-6)
        cc = np.clip(pred.c[RAT], bayes.BB_PHI_CLIP[0], bayes.BB_PHI_CLIP[1])
        rho = 1.0 / (cc + 1.0)
        sn = k * nbar
        sk = pp * sn
        Sres = (k - 1.0) * pp * (1.0 - pp) * (1.0 + rho * (nbar - 1.0))
        R[0, :, 0] = k
        R[0, :, 1] = sk
        R[0, :, 2] = sn
        R[0, :, 3] = Sres + pp * sk
        sc = pred.scale[NIG]
        within = sc * sc * k * k / (k + 1.0)
        N[0, :, 0] = k
        N[0, :, 1] = k * pred.loc[NIG]
        N[0, :, 2] = within + k * pred.loc[NIG] ** 2
    return np.where(np.isfinite(out), out, 0.0)


def predictive_q(store: Any, s: str, e: str, tctx: Mapping[str, Any],
                 model: Any = None) -> Dict[str, Pred]:
    """Q-grain predictives of a real entity at tctx's bucket (the Q row's
    midpoint context), cadence.md §6.2:
      current   = native Q current stats (+) KAPPA_T pseudo-rows of the
                  transferred H current predictive; 'none' features (set,
                  map) use the native stats alone and are scored only once
                  W_native >= KAPPA_T;
      reference = the transfer of the H reference (golden) predictive.
    Pred.prov = W_native / (W_native + KAPPA_T); Pred.scored."""
    b = int(tctx["bin48"])
    m = _entity_model(store, s, e, model)
    E, h = _chain(store, s, e, b, m, True)
    cur_h = _pred(E, h, _week_cell(m, tctx), tier="entity", anchor="current", bucket=b)
    E2, h2 = join(_reference_base(m, b), E, h, np.full(NF, KAPPA_REF), loo=False)
    ref_h = _pred(E2, h2, tier="entity", anchor="reference", bucket=b)
    v, da, db = omega_chain(store, s, e, m)
    var_c = GR.t_variance(cur_h.scale, cur_h.df)
    delta_c = da + db * np.where(_JENSEN, -(v - 1.0) * var_c / 2.0, 0.0)
    var_r = GR.t_variance(ref_h.scale, ref_h.df)
    delta_r = da + db * np.where(_JENSEN, -(v - 1.0) * var_r / 2.0, 0.0)
    t_cur = transfer_pred(cur_h, v, np.where(np.isfinite(delta_c), delta_c, 0.0))
    t_ref = transfer_pred(ref_h, v, np.where(np.isfinite(delta_r), delta_r, 0.0))
    qa = q_anchor(m)
    S_Q = _anchor_E(qa, b)
    W_nat = np.maximum(S_Q[0, FULL.W], 0.0)
    E_Q = S_Q + pseudo_stats(t_cur, _KT)
    par_t = _params(E_Q, np.zeros((1, NF)))
    par_n = _params(S_Q, _H1[:1])
    nC_, nR_, nN_ = _NONE[CNT][None, :], _NONE[RAT][None, :], _NONE[NIG][None, :]
    par = _Par(np.where(nC_, par_n.mu, par_t.mu), np.where(nC_, par_n.r, par_t.r),
               np.where(nR_, par_n.p, par_t.p), np.where(nR_, par_n.c, par_t.c),
               np.where(nN_, par_n.df, par_t.df), np.where(nN_, par_n.loc, par_t.loc),
               np.where(nN_, par_n.scale, par_t.scale))
    mu = np.full(NF, np.nan)
    r, p, c, df, loc, scale = (mu.copy() for _ in range(6))
    mu[CNT], r[CNT] = par.mu[0], par.r[0]
    p[RAT], c[RAT] = par.p[0], par.c[0]
    df[NIG], loc[NIG], scale[NIG] = par.df[0], par.loc[0], par.scale[0]
    # span features: no Q predictive (H only)
    span = ~_TRANSFERABLE & ~_NONE
    prov = W_nat / (W_nat + _KT)
    scored = np.where(_NONE, W_nat >= _KT, ~span)
    cur = Pred(mu, r, p, c, df, loc, scale, _mean(par)[0], ebar=15.0, tier="entity",
               anchor="current", bucket=b, mode="bin48", grain="q", prov=prov, scored=scored)
    t_ref.prov, t_ref.scored = prov, scored
    return {"current": cur, "reference": t_ref}
