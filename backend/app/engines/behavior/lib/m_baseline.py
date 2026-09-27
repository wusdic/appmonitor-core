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
    two bands) a mature bucket mean may wander at most +-c sigma15 around its
    value at the band start (c = 0.1 current, 0.03 reference + allow_drift).
    A row that would push it out is folded with the largest weight that keeps
    it on the boundary. A per-row allowance would also clip ordinary noise
    and freeze learning; the band lets noise cancel inside the day while a
    persistent shift moves at most c sigma15 per day. The cap binds only
    after a bucket-feature holds >= CAP_MIN_W rows and >= CAP_MIN_E weighted
    minutes (the hyperprior start must converge freely), never on rebased rows.
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

Anchor state (class Anchor, persisted inside model.baseline):
    a48[48, FULL.WIDTH]   per bucket: count [W, Sx, Se, Sxx, Sxe, See] x 10,
                          ratio [W, Sk, Sn, Skk/n] x 10, NIG [W, Sy, Syy] x 32,
                          then the band mean m_ref[52] and the band id
    a168[168, LOC.WIDTH]  location cells: count [W, Sx, Se], ratio [W, Sk, Sn],
                          NIG [W, Sy]; band m_ref[52] and id (None: no cells)
    T0, T, hl[52]         scale epoch, clock (newest folded ts), half-lives (s)
    sh[3, FULL.L]         shadow global stats per candidate half-life
model.baseline@(s, e) (dict, by reference):
    {fmt, tier: 'entity', version, branch, current: Anchor, reference: Anchor,
     gate / gate_ref: GateState, golden: {week, snaps, stats, n_reset},
     n_eff, allow_drift, maturity, ...engine bookkeeping}
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
    commit(anchor, row, w, cap=CAP_CURRENT, drift=0.0) -> anchor  (in place)
    merge(own, other, w) -> own                                   (link seeding)
    on_rebase_current(anchor, tau, until) / on_rebase_reference(anchor, tau)
    dump(anchor, dtype=float32) -> blob;  load(blob) -> Anchor
    true_stats(anchor, T=None) -> ndarray[48, FULL.L] | None
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
  model
    n_eff(model | anchor) -> float;  maturity(model) -> dict;  hl_days(model) -> ndarray
    held(model) -> [ts];  last_commit_ts(model) -> float;  version(model) -> int
    golden(model) -> ndarray[48, L] | None;  has_golden(model) -> bool
    golden_offset(store, s, e) -> ndarray[52] | None   zr is measured against golden
                                  once one exists (the reference predictive IS golden),
                                  so the offset is 0 then; None before
    descriptors(model) -> dict                       hourly means per day type (B02/B30)
    seasonal_curves(model, day_type, names) -> {name: [24]}  vec space (legacy profile)
  tiers (pure; B03 orchestrates)
    join(S, Ep, hp, kappa, loo=True) -> (E, h);  eb_kappa(children) -> ndarray[52]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from scipy import special as sp

from . import bayes
from . import features as F
from . import m_class
from . import priors as PR
from .classkeys import ORG, SYSTEM_KEY, is_class, is_pseudo

MODEL = "model.baseline"
FMT = 1
NF = F.FEATURE_DIM
DAY = 86400.0
WEEK = 7.0 * DAY

HL_CAND_DAYS: Tuple[float, ...] = (7.0, 14.0, 28.0)
HL_CAND_S = np.array(HL_CAND_DAYS) * DAY
HL_DEFAULT_DAYS = 14.0
HL_REF_DAYS = 28.0                 # the slow reference anchor keeps the longest half-life
SELECT_EVERY = 96                  # commits between half-life selections
PINBALL_EVERY = 4                  # the loss is evaluated on every 4th commit
SELECT_MIN_EVALS = 12              # evaluations with a valid value needed to re-select

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


def _nig_values(x: np.ndarray, dt_s: float) -> np.ndarray:
    """FEATURE_SPEC vec transform of the t-family columns, rebuilt from nat
    (the CLR composition from the counts behind comp_*)."""
    v = x[NIG]
    y = np.empty(nN)
    with np.errstate(all="ignore"):
        i = _TX_SETS[TX_BYTES]
        y[i] = np.log1p(np.maximum(v[i], 0.0) * (60.0 / dt_s))
        i = _TX_SETS[TX_LOG]
        vi = v[i]
        y[i] = np.where(vi >= 0.0, np.log(np.maximum(vi, F.AVG_FLOOR)), np.nan)
        i = _TX_SETS[TX_ID]
        y[i] = v[i]
        i = _TX_SETS[TX_LOG1P]
        vi = v[i]
        y[i] = np.where(vi > -1.0, np.log1p(vi), np.nan)
        i = _TX_SETS[TX_LOGIT]
        pp = np.clip(v[i], F.BOUNDED_EPS, 1.0 - F.BOUNDED_EPS)
        y[i] = np.log(pp) - np.log1p(-pp)
        xe = np.append(x, 1.0)
        cnt = xe[_CLR_A] * xe[_CLR_B]
        cnt = np.where(cnt > 0.0, cnt, 0.0)            # NaN (stale ratio) -> 0
        if cnt.sum() > 0.0:
            ly = np.log(cnt * (60.0 / dt_s) + F.CLR_PSEUDO)
            y[_CLR_POS] = ly - ly.mean()
        else:
            y[_CLR_POS] = np.nan
    return y


def observe(nat: Sequence[float], dt_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(num, den, wf) of one row in family observation space.

    count: num = x (count this tick), den = dt/60 minutes; ratio: num = k,
    den = n (the matching count feature); t: num = y, den = 1. wf is the
    per-feature weight (0 = not observed: NaN, n = 0, ...), min(1, n/5) for
    t features with a count exposure. Invalid entries are num = 0, den = 1 so
    they can be multiplied safely."""
    x = np.asarray(nat, dtype=np.float64).reshape(-1)
    if x.size != NF:
        raise ValueError(f"m_baseline.observe: nat must have {NF} values, got {x.size}")
    dt = float(dt_s)
    if not (dt > 0.0 and math.isfinite(dt)):
        raise ValueError(f"m_baseline.observe: bad dt_s {dt_s!r}")
    num = np.zeros(NF)
    den = np.ones(NF)
    wf = np.zeros(NF)
    xc = x[CNT]
    ok = np.isfinite(xc) & (xc >= 0.0)
    num[CNT] = np.where(ok, xc, 0.0)
    den[CNT] = dt / 60.0
    wf[CNT] = ok
    n = x[RATIO_N_IDX]
    r = x[RAT]
    okr = np.isfinite(r) & np.isfinite(n) & (n > 0.0) & (r >= 0.0)
    nn = np.where(okr, n, 1.0)
    num[RAT] = np.where(okr, np.minimum(np.where(okr, r, 0.0), 1.0) * nn, 0.0)
    den[RAT] = nn
    wf[RAT] = okr
    y = _nig_values(x, dt)
    xz = np.where(np.isfinite(x) & (x > 0.0), x, 0.0)
    wn = np.where(_HAS_N, np.minimum(1.0, (_NMAT @ xz) / 5.0), 1.0)
    okn = np.isfinite(y) & (wn > 0.0)
    num[NIG] = np.where(okn, y, 0.0)
    wf[NIG] = np.where(okn, wn, 0.0)
    return num, den, wf


def values_from_nat(nat: Sequence[float], dt_s: float) -> np.ndarray:
    """Family value per feature (count x, ratio k/n, t y); NaN if unobserved."""
    num, den, wf = observe(nat, dt_s)
    out = np.where(wf > 0.0, num, np.nan)
    out[RAT] = np.where(wf[RAT] > 0.0, num[RAT] / den[RAT], np.nan)
    return out


def _contrib(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Unit-weight contribution of one row to a FULL bucket row."""
    c = _C0.copy()
    c[_CX[0]] = num[_CX[1]]
    c[_CE[0]] = den[_CE[1]]
    xx = num[_CXX[1]]
    c[_CXX[0]] = xx * xx
    c[_CXE[0]] = num[_CXE[1]] * den[_CXE[1]]
    ee = den[_CEE[1]]
    c[_CEE[0]] = ee * ee
    kk = num[_CKK[1]]
    c[_CKK[0]] = kk * kk / den[_CKK[1]]
    return c


class Row(NamedTuple):
    """One committed tick in observation space (built by make_row)."""
    ts: float
    dt: float
    hour: float             # local hour (fractional), tctx.hour_local
    nwd: int                # 1 = nonworkday
    dow: int
    typical: bool           # day type matches the weekday (no holiday / make-up day)
    local_ts: float         # ts shifted to local wall time (band ids)
    num: np.ndarray
    den: np.ndarray
    wf: np.ndarray
    cfull: np.ndarray
    cloc: np.ndarray
    drift: float = 0.0
    elig: bool = True


def make_row(ts: float, nat: Sequence[float], dt_s: float, tctx: Mapping[str, Any],
             drift: float = 0.0, elig: bool = True) -> Row:
    """Row of tick ts from its feature.nat, real dt and time context."""
    num, den, wf = observe(nat, dt_s)
    hour = float(tctx["hour_local"])
    dow = int(tctx["dow"])
    nwd = 1 if tctx.get("day_type") == "nonworkday" else 0
    ts = float(ts)
    # local wall time: the tz offset is a multiple of 15 min
    off = (round(((hour * 3600.0 - ts) % DAY) / 900.0) * 900.0) % DAY
    cf = _contrib(num, den)
    return Row(ts, float(dt_s), hour, nwd, dow, nwd == (1 if dow >= 5 else 0), ts + off,
               num, den, wf, cf, cf[_LOC_FROM_FULL], float(drift), bool(elig))


@lru_cache(maxsize=8192)
def _vm(hour6: float) -> Tuple[int, np.ndarray, np.ndarray]:
    """(floor hour, offsets, weights) of the von Mises spread: bins whose
    centre is within 2 h of the hour (same formula as timebins.von_mises_weights)."""
    base = int(math.floor(hour6))
    offs, ws = [], []
    for off in range(-3, 4):
        dh = base + off + 0.5 - hour6
        if abs(dh) <= VM_MAX_DH + 1e-9:
            s = math.sin(math.pi * dh / 24.0)
            offs.append(off)
            ws.append(math.exp(-2.0 * VM_KAPPA * s * s))
    o = np.asarray(offs, dtype=np.intp)
    w = np.asarray(ws)
    o.setflags(write=False)
    w.setflags(write=False)
    return base, o, w


# ==================================================================== anchor
class Anchor:
    """Sufficient statistics of one anchor (see module docstring)."""

    __slots__ = ("a48", "a168", "T0", "T", "hl", "t_first", "n_commit", "sh", "sh_T0",
                 "loss", "nloss", "n_eval", "select", "uncap_lo", "uncap_hi", "reset_after",
                 "n_reset", "blk48", "blk168", "dirty48", "dirty168", "blk_code")

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


def _new_arr(n: int, lay: _Layout) -> np.ndarray:
    a = np.zeros((n, lay.WIDTH))
    a[:, lay.BAND] = np.nan
    return a


def new_anchor(week: bool = True, select: bool = True, hl_days: float = HL_DEFAULT_DAYS) -> Anchor:
    return Anchor(week=week, select=select, hl_days=hl_days)


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


def _sd15_par(par: _Par, ebar: np.ndarray) -> np.ndarray:
    """Predictive sd for a 15-minute exposure, in mean units, [nb, 52].
    count: NB sd of the 15-min count / 15; ratio: BB sd of k/n at the 15-min
    trials of its exposure feature; t: scale * sqrt(ebar/15), ebar = mean row
    exposure (min) of the bucket (a 15-min value averages ebar/15 rows' worth)."""
    with np.errstate(all="ignore"):
        m15 = 15.0 * par.mu
        sc = np.sqrt(m15 + m15 * m15 / par.r) / 15.0
        n15 = 15.0 * par.mu[:, _RAT_NPOS]
        n15 = np.where(np.isfinite(n15) & (n15 >= 1.0), n15, 1.0)
        pp, cc = par.p, par.c
        sr = np.sqrt(pp * (1.0 - pp) * (n15 + cc) / (n15 * (1.0 + cc)))
        sn = par.scale * np.sqrt(ebar / 15.0)[:, None]
    return _feat(sc, sr, sn)


def _ebar(St: np.ndarray, lay: _Layout = FULL) -> np.ndarray:
    """Mean row exposure (minutes) per bucket, 15 when the bucket is empty."""
    Wf = St[:, lay.W[_FLOWS]]
    with np.errstate(all="ignore"):
        e = St[:, lay.EXPO] / Wf
    return np.where((Wf > 0.0) & np.isfinite(e) & (e > 0.0), e, 15.0)


_H1 = np.ones((48, NF))


def _sd15(St: np.ndarray) -> np.ndarray:
    """Own-posterior sigma15 (hyperprior + own stats) of FULL rows [nb, L]."""
    return _sd15_par(_params(St, _H1[:St.shape[0]]), _ebar(St))


def _drift_term(mean0: np.ndarray, drift: float) -> np.ndarray:
    """allow_drift (log-units / day) in mean units: count rate x expm1(d),
    ratio p(1-p) d, t d."""
    d = abs(float(drift))
    return np.where(_IS_NB, mean0 * math.expm1(d), np.where(_IS_BB, mean0 * (1.0 - mean0) * d, d))


# ------------------------------------------------------------------- commit
def _clip_weights(w0, Aq, Bq, mean0, m1, lo, hi, num, den):
    """Largest weights keeping each violating mean on its band edge (none if
    the mean already sits beyond the edge it is moving towards)."""
    with np.errstate(all="ignore"):
        up = m1 > mean0
        bound = np.where(up, hi, lo)
        inside = np.where(up, mean0 < bound, mean0 > bound)
        wc = (bound * Bq - Aq) / (num - bound * den)
    return np.where(inside & np.isfinite(wc), np.clip(wc, 0.0, w0), 0.0)


def _fold48(A: np.ndarray, idx: np.ndarray, w0: np.ndarray, row: Row, G: np.ndarray,
            Gs: np.ndarray, cap: float, capped: bool, drift: float,
            band: np.ndarray) -> Optional[np.ndarray]:
    """Fold one row into bin48 buckets idx with weights w0[nb, 52] under the
    band cap. The band edges are set once per bucket and band-day from the
    bucket's own sigma15 at the band start. Returns the half-widths of the
    touched buckets' bands (for the bin168 cells), None when uncapped."""
    blk = A[idx]
    half = None
    if capped:
        with np.errstate(all="ignore"):
            Aq = _P + blk[:, FULL.NUM] / G
            Bq = _Q + blk[:, FULL.DEN] / G
            mean0 = Aq / Bq
        old = blk[:, FULL.BAND]
        newb = ~(band <= old)                   # no band yet, or a later band-day
        if newb.any():
            k = np.flatnonzero(newb)
            St = blk[k, :FULL.L] / Gs
            hw = cap * _sd15(St)
            if drift:
                hw = hw + _drift_term(mean0[k], drift)
            blk[k, FULL.LO:FULL.HI] = mean0[k] - hw
            blk[k, FULL.HI:FULL.BAND] = mean0[k] + hw
            blk[k, FULL.BAND] = band[k]
        lo = blk[:, FULL.LO:FULL.HI]
        hi = blk[:, FULL.HI:FULL.BAND]
        half = 0.5 * (hi - lo)
        live = ((w0 > 0.0) & (blk[:, FULL.W] >= CAP_MIN_W * G)
                & (blk[:, FULL.EXPO] >= CAP_MIN_E * G[_FLOWS])[:, None])
        with np.errstate(all="ignore"):
            m1 = (Aq + w0 * row.num) / (Bq + w0 * row.den)
        viol = live & ((m1 > hi) | (m1 < lo))
        if viol.any():
            w0 = np.where(viol, _clip_weights(w0, Aq, Bq, mean0, m1, lo, hi, row.num, row.den), w0)
    else:
        blk[:, FULL.BAND] = np.nan              # rebased: a fresh band afterwards
    blk[:, :FULL.L] += (w0[:, FULL.SLOT_F] * row.cfull) * Gs
    A[idx] = blk
    return half


def _fold168(A: np.ndarray, idx: np.ndarray, w0: np.ndarray, row: Row, G: np.ndarray,
             Gs: np.ndarray, capped: bool, band: np.ndarray, half: Optional[np.ndarray]) -> None:
    """Fold one row into bin168 location cells: the cell mean may move at
    most the half-width of its bin48 bucket's band around its own band-start
    mean."""
    blk = A[idx]
    if capped and half is not None:
        with np.errstate(all="ignore"):
            Aq = _P + blk[:, LOC.NUM] / G
            Bq = _Q + blk[:, LOC.DEN] / G
            mean0 = Aq / Bq
        old = blk[:, LOC.BAND]
        newb = ~(band <= old)
        if newb.any():
            k = np.flatnonzero(newb)
            blk[k, LOC.M_REF:LOC.BAND] = mean0[k]
            blk[k, LOC.BAND] = band[k]
        mref = blk[:, LOC.M_REF:LOC.BAND]
        lo, hi = mref - half, mref + half
        live = ((w0 > 0.0) & (blk[:, LOC.W] >= CAP_MIN_W * G)
                & (blk[:, LOC.EXPO] >= CAP_MIN_E * G[_FLOWS])[:, None])
        with np.errstate(all="ignore"):
            m1 = (Aq + w0 * row.num) / (Bq + w0 * row.den)
        viol = live & ((m1 > hi) | (m1 < lo))
        if viol.any():
            w0 = np.where(viol, _clip_weights(w0, Aq, Bq, mean0, m1, lo, hi, row.num, row.den), w0)
    else:
        blk[:, LOC.BAND] = np.nan
    blk[:, :LOC.L] += (w0[:, LOC.SLOT_F] * row.cloc) * Gs
    A[idx] = blk


def _band_ids(row: Row, hours: np.ndarray) -> np.ndarray:
    """Band-day id per touched bucket: day boundaries 12 h away from the
    bucket's hour centre, so one day's cluster of commits is one band."""
    h = (hours % 24).astype(np.float64) + 12.5
    return np.floor((row.local_ts - h * 3600.0) / DAY)


def commit(anc: Anchor, row: Row, w: float, *, cap: float = CAP_CURRENT,
           drift: float = 0.0) -> Anchor:
    """Fold one row with trust weight w (GatedLearner update; in place).

    Deterministic in (state, row, w): replaying the same rows in the same
    order reproduces the statistics exactly. A row older than the clock is
    folded with its own decay 2^-((T - ts)/hl)."""
    w = float(w)
    if not (w > 0.0 and math.isfinite(w)):
        return anc
    ts = float(row.ts)
    if anc.reset_after == anc.reset_after and ts >= anc.reset_after:
        _reset(anc)
    if anc.empty:
        anc.T0 = anc.T = anc.t_first = ts
    T_new = ts if ts > anc.T else anc.T
    ex = (T_new - anc.T0) / anc.hl
    if ex.max() > RENORM_EXP:
        _renorm(anc, T_new)
        ex = np.zeros(NF)
    G = np.exp2(ex)
    wrow = w * row.wf
    if ts < T_new:
        wrow = wrow * np.exp2(-(T_new - ts) / anc.hl)
    capped = not (anc.uncap_lo <= ts <= anc.uncap_hi)
    base, offs, vm = _vm(round(row.hour, 6))
    hours = base + offs
    band = _band_ids(row, hours)
    w0 = vm[:, None] * wrow
    idx48 = hours % 24 + 24 * row.nwd
    half = _fold48(anc.a48, idx48, w0, row, G, G[FULL.SLOT_F], cap, capped, drift, band)
    anc.dirty48.update(idx48.tolist())
    if anc.a168 is not None and row.typical:
        idx168 = (row.dow * 24 + hours) % 168
        _fold168(anc.a168, idx168, w0, row, G, G[LOC.SLOT_F], capped, band, half)
        anc.dirty168.update(idx168.tolist())
    anc.T = T_new
    anc.n_commit += 1
    _shadow(anc, row, w)
    return anc


def _renorm(anc: Anchor, T: float) -> None:
    anc.a48[:, :FULL.L] *= _scale(anc, T, FULL)
    if anc.a168 is not None:
        anc.a168[:, :LOC.L] *= _scale(anc, T, LOC)
    anc.T0 = float(T)
    anc._dirty_all()


def _reset(anc: Anchor) -> None:
    """Empty the statistics (reference reset one day into a rebased regime)."""
    anc.a48 = _new_arr(48, FULL)
    if anc.a168 is not None:
        anc.a168 = _new_arr(168, LOC)
    anc.T0 = anc.T = anc.t_first = math.nan
    anc.n_commit = 0
    anc.sh[:] = 0.0
    anc.sh_T0 = math.nan
    anc.loss[:] = 0.0
    anc.nloss[:] = 0.0
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
    anc.hl = np.asarray(new_hl, dtype=np.float64).copy()
    anc._dirty_all()


# ----------------------------------------------------- half-life selection
def _rho(u: np.ndarray, tau: float) -> np.ndarray:
    return np.maximum(tau * u, (tau - 1.0) * u)


def _pinball(par: _Par, row: Row) -> np.ndarray:
    """Pinball loss of the p5 / p95 of each candidate predictive [k, 52]
    (moment approximations for NB / BB, exact t quantiles)."""
    y, d = row.num[_FAM_ORDER], row.den
    with np.errstate(all="ignore"):
        dc = d[CNT]
        m = par.mu * dc
        sd = np.sqrt(m + m * m / par.r)
        lo_c, hi_c = np.maximum(m - Z95 * sd, 0.0), m + Z95 * sd
        dr = d[RAT]
        m = par.p * dr
        sd = np.sqrt(dr * par.p * (1.0 - par.p) * (dr + par.c) / (1.0 + par.c))
        lo_r, hi_r = np.maximum(m - Z95 * sd, 0.0), np.minimum(m + Z95 * sd, dr)
        t95 = sp.stdtrit(par.df, 0.95) * par.scale
        lo = np.concatenate((lo_c, lo_r, par.loc - t95), axis=1)
        hi = np.concatenate((hi_c, hi_r, par.loc + t95), axis=1)
        out = _rho(y - lo, 0.05) + _rho(y - hi, 0.95)
    return np.where(np.isfinite(out), out, 0.0)[:, _INV]


_H3 = np.ones((HL_CAND_S.size, NF))


def _shadow(anc: Anchor, row: Row, w: float) -> None:
    """Global (unbucketed, uncapped) stats at each candidate half-life; with
    `select`, the p5/p95 pinball loss of each before the fold, and every
    SELECT_EVERY commits the per-feature half-life with the smallest loss."""
    T = anc.T
    if anc.sh_T0 != anc.sh_T0:
        anc.sh_T0 = T
    ex = (T - anc.sh_T0) / HL_CAND_S
    if ex.max() > RENORM_EXP:
        anc.sh *= np.exp2(-ex)[:, None]
        anc.sh_T0 = T
        ex = np.zeros_like(ex)
    valid = row.wf > 0.0
    if anc.select and anc.n_commit % PINBALL_EVERY == 0 and valid.any():
        St = anc.sh / np.exp2(ex)[:, None]
        loss = _pinball(_params(St, _H3), row)
        anc.loss += np.where(valid, w * loss, 0.0)
        anc.nloss += valid
    anc.sh += np.exp2((row.ts - anc.sh_T0) / HL_CAND_S)[:, None] * \
        (w * row.wf[FULL.SLOT_F] * row.cfull)[None, :]
    if anc.select:
        anc.n_eval += 1
        if anc.n_eval >= SELECT_EVERY:
            _select(anc)


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
    anc.loss[:] = 0.0
    anc.nloss[:] = 0.0
    anc.n_eval = 0


# ---------------------------------------------------- merge / rebase hooks
def merge(own: Anchor, other: Optional[Anchor], w: float) -> Anchor:
    """own + w * other in sufficient-statistic space (link seeding), other
    folded at its own clock. Bands and half-lives of `own` are kept."""
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
def dump(anc: Anchor, dtype: Any = np.float32) -> Dict[str, Any]:
    """Checkpoint blob. Bucket rows are immutable bytes blocks rebuilt only
    for buckets touched since the last dump, so successive checkpoints share
    every untouched block (copy.deepcopy keeps bytes objects shared)."""
    dt = np.dtype(dtype)
    if anc.blk_code != dt.str:
        anc.blk_code = dt.str
        anc._dirty_all()
    for b in anc.dirty48:
        anc.blk48[b] = anc.a48[b].astype(dt).tobytes()
    anc.dirty48 = set()
    b168 = None
    if anc.a168 is not None:
        for b in anc.dirty168:
            anc.blk168[b] = anc.a168[b].astype(dt).tobytes()
        anc.dirty168 = set()
        b168 = tuple(anc.blk168)
    return {"v": FMT, "dtype": dt.str, "b48": tuple(anc.blk48), "b168": b168,
            "T0": anc.T0, "T": anc.T, "hl": anc.hl.copy(), "t_first": anc.t_first,
            "n_commit": anc.n_commit, "sh": anc.sh.copy(), "sh_T0": anc.sh_T0,
            "loss": anc.loss.copy(), "nloss": anc.nloss.copy(), "n_eval": anc.n_eval,
            "select": anc.select, "uncap": (anc.uncap_lo, anc.uncap_hi),
            "reset_after": anc.reset_after, "n_reset": anc.n_reset}


def load(blob: Mapping[str, Any]) -> Anchor:
    dt = np.dtype(blob["dtype"])
    week = blob.get("b168") is not None
    anc = Anchor(week=week, select=bool(blob.get("select", True)))
    anc.a48 = np.frombuffer(b"".join(blob["b48"]), dtype=dt).reshape(48, FULL.WIDTH).astype(np.float64)
    anc.blk48 = list(blob["b48"])
    anc.dirty48 = set()
    if week:
        anc.a168 = np.frombuffer(b"".join(blob["b168"]), dtype=dt).reshape(168, LOC.WIDTH).astype(np.float64)
        anc.blk168 = list(blob["b168"])
        anc.dirty168 = set()
    anc.blk_code = dt.str
    anc.T0, anc.T, anc.t_first = float(blob["T0"]), float(blob["T"]), float(blob["t_first"])
    anc.hl = np.array(blob["hl"], dtype=np.float64)
    anc.n_commit = int(blob["n_commit"])
    anc.sh = np.array(blob["sh"], dtype=np.float64)
    anc.sh_T0 = float(blob["sh_T0"])
    anc.loss = np.array(blob["loss"], dtype=np.float64)
    anc.nloss = np.array(blob["nloss"], dtype=np.float64)
    anc.n_eval = int(blob["n_eval"])
    anc.uncap_lo, anc.uncap_hi = (float(x) for x in blob["uncap"])
    anc.reset_after = float(blob["reset_after"])
    anc.n_reset = int(blob["n_reset"])
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
    size = D[..., FULL.DEN]
    keff = np.asarray(kappa, dtype=np.float64) * _JOIN_UNIT
    with np.errstate(all="ignore"):
        s = np.where(size > 0.0, np.minimum(1.0, keff / size), 1.0)
    return S + s[..., FULL.SLOT_F] * D, s * np.asarray(hp, dtype=np.float64)


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
    with np.errstate(all="ignore"):
        S = FULL.C_S
        for j, f in enumerate(CNT):
            W, sx, se = X[:, S[j, 0]], X[:, S[j, 1]], X[:, S[j, 2]]
            ok = se >= 60.0
            if ok.sum() < 2:
                continue
            mu = sx[ok] / se[ok]
            mbar = float(mu.mean())
            pooled = X[ok][:, S[j]].sum(axis=0)
            kap = float(_overdisp(*pooled))
            v15 = (15.0 * mbar + (15.0 * mbar) ** 2 / kap) / 225.0
            vb = float(mu.var(ddof=1)) - float(np.mean(v15 * 15.0 / se[ok]))
            out[f] = v15 / vb if vb > 0.0 else KAPPA_MAX
        S = FULL.R_S
        for j, f in enumerate(RAT):
            sk, sn = X[:, S[j, 1]], X[:, S[j, 2]]
            ok = sn >= 20.0
            if ok.sum() < 2:
                continue
            pi = sk[ok] / sn[ok]
            m = float(pi.mean())
            if not 0.0 < m < 1.0:
                continue
            vb = float(pi.var(ddof=1)) - float(np.mean(m * (1.0 - m) / sn[ok]))
            out[f] = m * (1.0 - m) / vb - 1.0 if vb > 0.0 else KAPPA_MAX
        S = FULL.N_S
        for j, f in enumerate(NIG):
            W, sy, syy = X[:, S[j, 0]], X[:, S[j, 1]], X[:, S[j, 2]]
            ok = W >= 4.0
            if ok.sum() < 2:
                continue
            mi = sy[ok] / W[ok]
            wv = np.maximum(syy[ok] / W[ok] - mi * mi, 0.0)
            vw = float(wv.mean())
            vb = float(mi.var(ddof=1)) - float(np.mean(vw / W[ok]))
            out[f] = vw / vb if vb > 0.0 else KAPPA_MAX
    out = np.where(np.isfinite(out), out, KAPPA_DEFAULT)
    return np.clip(out, KAPPA_MIN, KAPPA_MAX)


def tier_model(store: Any, s: str, key: str) -> Optional[Mapping[str, Any]]:
    """Published tier model (key 'class:<rid>' | '__system__' | 'org')."""
    m = (store.get_model(ORG[0], ORG[1], MODEL) if key == "org"
         else store.get_model(s, key, MODEL))
    if isinstance(m, Mapping) and m.get("fmt") == FMT and m.get("E") is not None:
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

    @property
    def family(self) -> np.ndarray:
        return FAMILY


def _pred(E: np.ndarray, h: np.ndarray, **kw: Any) -> Pred:
    par = _params(E, h)
    mu = np.full(NF, np.nan)
    r, p, c, df, loc, scale = (mu.copy() for _ in range(6))
    mu[CNT], r[CNT] = par.mu[0], par.r[0]
    p[RAT], c[RAT] = par.p[0], par.c[0]
    df[NIG], loc[NIG], scale[NIG] = par.df[0], par.loc[0], par.scale[0]
    return Pred(mu, r, p, c, df, loc, scale, _mean(par)[0], ebar=float(_ebar(E)[0]), **kw)


def _typical(tctx: Mapping[str, Any]) -> bool:
    dow = int(tctx.get("dow", 0))
    nwd = tctx.get("day_type") == "nonworkday"
    return nwd == (dow >= 5)


def _refine168(pr: Pred, E: np.ndarray, cell: np.ndarray) -> Pred:
    """Hour-of-week location: the bin168 cell's mean shrunk to the bin48
    mean with K168 pseudo-rows of the bucket's average row; dispersion stays
    that of the bin48 predictive."""
    Wb = E[0, FULL.W]
    with np.errstate(all="ignore"):
        dbar = E[0, FULL.DEN] / Wb
        dbar = np.where((Wb > 0.0) & np.isfinite(dbar) & (dbar > 0.0), dbar,
                        np.where(_IS_NB, 15.0, 1.0))
        num = cell[LOC.NUM]
        den = cell[LOC.DEN]
        m168 = (num + K168 * dbar * pr.mean) / (den + K168 * dbar)
    m168 = np.where(np.isfinite(m168), m168, pr.mean)
    pr.mean = m168
    pr.mu = np.where(_IS_NB, m168, pr.mu)
    pr.p = np.where(_IS_BB, m168, pr.p)
    pr.loc = np.where(FAMILY == FAM_T, m168, pr.loc)
    pr.mode = "bin168"
    return pr


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
    pr = _pred(E, h, anchor=anchor, bucket=b)
    if anc is not None and anc.week_mode() and _typical(tctx):
        pr = _refine168(pr, E, _cell(anc, int(tctx["bin168"])))
    return pr


def _entity_model(store: Any, s: str, e: str, model: Any) -> Optional[Mapping[str, Any]]:
    m = model if model is not None else store.get_model(s, e, MODEL)
    if isinstance(m, Mapping) and m.get("fmt") == FMT and m.get("tier", "entity") == "entity":
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
        pr = _pred(E, h, tier=tier, anchor=anchor, bucket=b)
        cur = m.get("current") if m is not None else None
        if cur is not None and cur.week_mode() and _typical(tctx):
            pr = _refine168(pr, E, _cell(cur, int(tctx["bin168"])))
        return pr
    E2, h2 = join(_reference_base(m, b), E, h, np.full(NF, KAPPA_REF), loo=False)
    return _pred(E2, h2, tier=tier, anchor=anchor, bucket=b)


def _chain_own(m: Mapping[str, Any], b: int, T: float) -> np.ndarray:
    return _anchor_E(m.get("current"), b, T)


def _loo_parent(tm: Mapping[str, Any], b: int, m: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """The parent tier's effective stats without the entity's own data."""
    D = tm["E"][b:b + 1] - _chain_own(m, b, float(tm["ts"]))
    D[..., FULL.NONNEG] = np.maximum(D[..., FULL.NONNEG], 0.0)
    return D, tm["h"][b:b + 1]


def predictive_set(store: Any, s: str, e: str, tctx: Mapping[str, Any],
                   model: Any = None) -> Dict[str, Pred]:
    """current, reference (both with backoff) and class (leave-one-out
    parent) predictives of a real entity, sharing the chain computation."""
    b = int(tctx["bin48"])
    m = _entity_model(store, s, e, model)
    E, h = _chain(store, s, e, b, m, True)
    cur_pr = _pred(E, h, tier="entity", anchor="current", bucket=b)
    cur = m.get("current") if m is not None else None
    if cur is not None and cur.week_mode() and _typical(tctx):
        cur_pr = _refine168(cur_pr, E, _cell(cur, int(tctx["bin168"])))
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
    """t-family values back to natural units (per-dt bytes, ms, shares, ...)."""
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
    for j, f in enumerate(RAT):
        a, b = pred.p[f] * pred.c[f], (1.0 - pred.p[f]) * pred.c[f]
        if a > 0.0 and b > 0.0 and np.isfinite(a + b):
            for i, qq in enumerate(q):
                out[i, f] = float(bayes.bb_ppf(float(qq), float(n[j]), float(a), float(b))) / n[j]
    with np.errstate(all="ignore"):
        t = sp.stdtrit(pred.df[NIG][None, :], np.clip(q, 0.0, 1.0)[:, None])
        y = pred.loc[NIG][None, :] + pred.scale[NIG][None, :] * t
    full = np.full((q.size, NF), np.nan)
    full[:, NIG] = y
    out[:, NIG] = _inverse_tx(full, dt_s)[:, NIG]
    return out


def mean_nat(pred: Pred, dt_s: float = 900.0) -> np.ndarray:
    """Predictive centre in natural units: count mean per dt, ratio p, t
    features the inverse transform of the location (a median for monotone
    transforms)."""
    out = np.array(pred.mean, dtype=np.float64, copy=True)
    out[CNT] = pred.mu[CNT] * dt_s / 60.0
    full = np.full(NF, np.nan)
    full[NIG] = pred.loc[NIG]
    out[NIG] = _inverse_tx(full, dt_s)[NIG]
    return out


def sd15(pred: Pred) -> np.ndarray:
    """Predictive sd for a 15-minute exposure in mean units (the cap's sigma15)."""
    par = _Par(pred.mu[None, CNT], pred.r[None, CNT], pred.p[None, RAT], pred.c[None, RAT],
               pred.df[None, NIG], pred.loc[None, NIG], pred.scale[None, NIG])
    return _sd15_par(par, np.array([pred.ebar]))[0]


def vec_median_sd(pred: Pred) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate median and sd in FEATURE_SPEC vec space (legacy
    profile.baseline_median / baseline_mad): counts log1p(rate per min),
    ratios logit(p) at the 15-min trials, t features as is (loc, scale)."""
    s15 = sd15(pred)
    med = np.array(pred.loc, dtype=np.float64, copy=True)
    sd = np.array(pred.scale, dtype=np.float64, copy=True)
    with np.errstate(all="ignore"):
        mu = pred.mu[CNT]
        med[CNT] = np.log1p(mu)
        sd[CNT] = s15[CNT] / (1.0 + mu)
        pp = np.clip(pred.p[RAT], 1e-6, 1.0 - 1e-6)
        med[RAT] = np.log(pp) - np.log1p(-pp)
        sd[RAT] = s15[RAT] / (pp * (1.0 - pp))
    return med, sd


def bucket_means(anc: Optional[Anchor], T: Optional[float] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """(mean[48, 52], sigma15[48, 52]) of every bin48 bucket from the anchor's
    own statistics and the hyperprior (what the rate cap sees)."""
    St = true_stats(anc, T)
    if St is None:
        St = np.zeros((48, FULL.L))
    par = _params(St, _H1)
    return _mean(par), _sd15_par(par, _ebar(St))


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
    Mn = _inverse_tx(M, dt_s)
    for name in names:
        f = F.FEATURE_INDEX[name]
        v = M[:, f] * dt_s / 60.0 if FAMILY[f] == FAM_NB else Mn[:, f]
        vals = [float(x) if math.isfinite(x) else None for x in v]
        out["features"][name] = {"workday": vals[:24], "nonworkday": vals[24:]}
    return out
