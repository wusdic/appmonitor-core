"""LikelihoodEngine (B04) — exact, exposure-aware predictive p-values per
feature against both baseline anchors and the class, plus the normal scores
(z, zr) every downstream multivariate / sequential engine consumes.

Why (replaces the robust-z primary of anomaly.py): a robust z on a
log-transformed rate ignores exposure (3 requests in a 60-s tick and 3 in a
900-s tick are very different evidence), treats ratios of 2 trials like
ratios of 2000, and has no exact tail, so calibration had to guess. Here each
feature is scored under B03's conjugate predictive for its family
(lib/m_baseline):

  * count   NB(mean mu e, size r = 1/(1/kappa + 1/a)), e = dt/60 minutes,
            kappa the moment overdispersion clipped to [0.5, 1e3];
  * ratio   BetaBinomial(n*, p c, (1 - p) c), c = min(sum n + 2, phi) with
            phi clipped to [20, 1000]; n* = 0 is NaN (nothing observed),
            never p = 1;
  * other   Student-t from the NIG posterior on the FEATURE_SPEC transform.

Tails are exact on their own side (lib/bayes: betainc / summed BB tails /
stdtr), scalar fast paths inside the per-feature loop (the numpy array path
costs ~110 us of dispatch for 10 features). The engine scores all rows of a
tick in one batch (score_features_many, m_baseline *_many): bit-identical to
the per-entity path, ~2.3x cheaper (docs/lib3/integration.md §9).

Dual anchor. The two-sided mid-p is taken against the CURRENT anchor (p_cur)
and the REFERENCE / golden anchor (p_ref), and
    p_f = min(1, 2 min(p_cur, p_ref))          (bayes.combine_anchors)
so a current anchor that has crept toward an attack cannot cancel the
evidence: the reference still sees it (Bonferroni over the two anchors keeps
p_f valid).

Normal scores. z = Phi^-1(u_cur), zr = Phi^-1(u_ref) with u the SEEDED
RANDOMISED PIT u = F(k-1) + V P(X = k), V = combine.seeded_uniform(s, e, ts,
'B04', f) (the same V for both anchors, so z and zr describe the same draw).
The mid-distribution value would put a point mass on every low count
(zero-inflated features), and B06's MCD-style covariance collapses onto such
ties (lib/robustcov "known limit"); the randomised PIT is exactly U(0, 1)
under the predictive, so z is exactly N(0, 1), and the seed keeps replay
bit-identical. Each tail is formed on its small side (u = p/2 + (V - 1/2)
P(X = k) below the median, the mirror image above), so |z| up to 7.94 is
exact and never comes from 1 - u. p_f stays the (deterministic) mid-p.

Channels. Dependence groups come from model.groups (B06) through
m_density.groups(split_by_feature_group=True), so no dependence group
straddles two FEATURE_SPEC groups (fallback: singletons). Within a
dependence group the per-feature p_f are combined by the harmonic mean
(weight 1/|group| each), then the group p's by a harmonic mean across
groups (lib/combine.whmp semantics: NaN dropped, all-NaN -> NaN):
    marg_int   = the volume groups          score -log10 p, pm p
    marg_shape = every other group
    peer       = all groups, under the class-tier predictive (the entity's
                 role class with >= 3 members, else the system tier;
                 leave-one-out). NaN while no tier model exists: a bare
                 hyperprior is not a peer group.
Axes: FEATURE_SPEC groups whose combined p < 0.01, per detector.

A new entity is scored from its first tick: m_baseline.predictive_set backs
off entity -> class -> system -> org -> hyperprior. Absence is data but not
ours: B04 runs only on ticks with feature.active == 1 (silence is B07's).
B04 learns nothing and emits no events, so ctx.training changes nothing;
B24 needs the scores during warm-up too.

profile.extra.model_state: predictive p5 / p50 / p95 in natural units for a
15-minute exposure (counts per 15 min, ratios as fractions, averages in
their units) at the current anchor's bucket, refreshed once an hour per
entity at a per-entity phase (the NB / BB ppf searches cost ~1 ms per
entity, so not every tick, and not all entities on the same tick).

spec v2.1 grains (docs/lib3/cadence.md §6.2, §7.1; canonical grain mode).
On an H decision tick the H row (feature.nat.h, exposure = its coverage,
the window-midpoint bucket) is scored exactly as above: behavior.z / zr / pf
and marg_int / marg_shape / peer. On a Q decision tick the Q row is scored
against m_baseline.predictive_q (native Q stats plus KAPPA_T pseudo-rows of
the transferred H predictive; the transferred H reference): behavior.z.q /
zr.q / pf.q and marg_int_q / marg_shape_q, plus behavior.prov {detector:
pi_nat} (the median native share of the features scored). Span features are
scored only on their span decision (H), and set / map features at Q only
once the Q anchor holds KAPPA_T native rows of them. Nothing is scored on a
non-decision tick (NaN = unscored, not degraded).

Degraded (contract M): an active entity without a feature.nat row at now, or
B01 / B03 failing at this tick, gets all-NaN z / zr / pf rows and NaN scores
with behavior.degraded = {detector: cause}.

Store:
  reads   feature.active, feature.nat (vec rings), feature.tctx (dict; the
          config time context when absent), model.baseline (entity current /
          reference, class, system, org via lib/m_baseline), model.class (via
          m_baseline.parent_key), model.groups (via lib/m_density); the ratio
          exposure n is the count feature behind the feature.expo channel
          (m_baseline.RATIO_N_IDX), so feature.expo itself is not re-read
  writes  behavior.z[52], behavior.zr[52], behavior.pf[52] (float32 vec rings),
          behavior.score / pm [marg_int, marg_shape, peer], behavior.axes,
          behavior.degraded (lib/emit), profile.extra.model_state
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from scipy import special as sp

from ...core.engine import Context, Engine
from ...models.schema import EntityProfile
from .lib import bayes
from .lib import pactive as PA
from .lib import psketch as PS_EARN
from .lib import combine
from .lib import emit
from .lib import grains as GR
from .lib import m_baseline as MB
from .lib import m_density
from .lib import timebins as TB
from .lib.features import (FEATURE_DIM, FEATURE_GROUP, FEATURE_KIND, FEATURE_NAMES_V2,
                           GROUP_ORDER, VEC_TX)

Z = "behavior.z"
EARNED_MODEL = PA.EARNED          # model.earned@(s, '__system__'): bounded mode only (§10.2)
EARN_K = 2048                     # candidate IPs per system (heavy hitters of active IP-rows)
EARN_HL = 7 * 86400.0             # H_m
ZR = "behavior.zr"
PF = "behavior.pf"
ACTIVE = "feature.active"
NAT = "feature.nat"
TCTX = "feature.tctx"
B01_ENGINE = "behavior.feature_vector"
B03_ENGINE = "behavior.baseline"

DETS = ("marg_int", "marg_shape", "peer")
DETS_Q = ("marg_int_q", "marg_shape_q")
PROV = "behavior.prov"
PROV_MIN = 0.5
PROV_CAUSE = emit.cause(emit.PROVISIONAL, "q_transfer")
INTENSITY_GROUP = "volume"
AXIS_ALPHA = 0.01
STATE_QS = (0.05, 0.5, 0.95)
STATE_DT_S = 900.0              # model_state exposure: 15 minutes, cadence independent
STATE_PERIOD_S = 3600.0
SEED_TAG = "B04"

NF = FEATURE_DIM
_NAN = math.nan
_P_FLOOR = bayes.P_FLOOR
_CNT = MB.CNT.tolist()
_RAT = MB.RAT.tolist()
_NIG = MB.NIG
_DISCRETE = MB.FAMILY != MB.FAM_T
_UNSET = object()                # _model_state: quantiles not precomputed
_NAN_ROW = np.full(NF, np.nan, dtype=np.float32)
_NAN_ROW.setflags(write=False)


# ================================================================ scoring
class FeatureScores(NamedTuple):
    """Per-feature outputs of one tick ([52] float64, NaN = unscored)."""
    z: np.ndarray           # Phi^-1 of the randomised PIT, current anchor
    zr: np.ndarray          # same, reference anchor
    pf: np.ndarray          # min(1, 2 min(p_cur, p_ref))
    p_cur: np.ndarray       # two-sided mid-p, current anchor
    p_ref: np.ndarray       # two-sided mid-p, reference anchor
    p_cls: np.ndarray       # two-sided mid-p, class tier (NaN without one)


def family_midp(pred: MB.Pred, num: np.ndarray, den: np.ndarray, wf: np.ndarray,
                want_eq: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(u, p, eq) per feature under one predictive: u = P(X < x) + P(X = x)/2,
    p the two-sided mid-p, eq = P(X = x) for the discrete families (0 for t
    features and when want_eq is False). NaN where the feature is not observed
    (wf == 0: NaN input, ratio with n = 0, ...). Same values as
    m_baseline.midp, through bayes' scalar fast paths."""
    u = [_NAN] * NF
    p = [_NAN] * NF
    eq = [0.0] * NF
    numl, denl, wfl = num.tolist(), den.tolist(), wf.tolist()
    mu, r = pred.mu.tolist(), pred.r.tolist()
    for f in _CNT:
        if wfl[f] > 0.0:
            m = mu[f] * denl[f]
            uu, pp = bayes.nb_midp(numl[f], m, r[f])
            u[f], p[f] = uu, pp
            if want_eq and pp == pp:
                eq[f] = bayes.nb_pmf(numl[f], m, r[f])
    pr, cr = pred.p.tolist(), pred.c.tolist()
    for f in _RAT:
        if wfl[f] > 0.0:
            a, b = pr[f] * cr[f], (1.0 - pr[f]) * cr[f]
            uu, pp = bayes.bb_midp(numl[f], denl[f], a, b)
            u[f], p[f] = uu, pp
            if want_eq and pp == pp:
                eq[f] = _bb_pmf1(numl[f], denl[f], a, b)
    ua, pa, eqa = np.array(u), np.array(p), np.array(eq)
    ok = wf[_NIG] > 0.0
    if ok.any():
        i = _NIG[ok]
        with np.errstate(all="ignore"):
            zt = (num[i] - pred.loc[i]) / pred.scale[i]
            lo = sp.stdtr(pred.df[i], zt)
            hi = sp.stdtr(pred.df[i], -zt)
        good = np.isfinite(zt) & (pred.scale[i] > 0.0)
        ua[i] = np.where(good, lo, np.nan)
        pa[i] = np.where(good, np.clip(2.0 * np.minimum(lo, hi), _P_FLOOR, 1.0), np.nan)
    eqa = np.where(np.isfinite(eqa), eqa, 0.0)
    return ua, pa, eqa


def _bb_pmf1(k: float, n: float, a: float, b: float) -> float:
    """BB pmf at the nearest integers k, n (bayes.bb_midp's rounding) with
    plain lgamma differences: ~1e-12 absolute in log, ample for the PIT's
    P(X = x) and ~20x cheaper than the array bayes.bb_logpmf per feature."""
    k, n = math.floor(k + 0.5), math.floor(n + 0.5)
    if not (0.0 <= k <= n):
        return 0.0
    lg = math.lgamma
    return math.exp(lg(n + 1.0) - lg(k + 1.0) - lg(n - k + 1.0) + lg(k + a) - lg(a)
                    + lg(n - k + b) - lg(b) - lg(n + a + b) + lg(a + b))


def pit_z(u: np.ndarray, p: np.ndarray, eq: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Normal score of the randomised PIT F(x-1) + V P(X = x), formed on the
    small side: below the median t = p/2 + (V - 1/2) eq and z = Phi^-1(t);
    above it t = p/2 + (1/2 - V) eq and z = -Phi^-1(t). For continuous
    features eq = 0 and this is Phi^-1(u) without the 1 - u cancellation.
    NaN p -> NaN z."""
    small = 0.5 * p
    lower = u < 0.5
    with np.errstate(invalid="ignore"):
        t = np.where(lower, small + (v - 0.5) * eq, small + (0.5 - v) * eq)
        t = np.clip(t, 0.0, 1.0)
    z = sp.ndtri(np.clip(t, bayes.PHI_CLIP, 1.0 - bayes.PHI_CLIP))
    return np.where(np.isnan(p), np.nan, np.where(lower, z, -z))


def pit_uniforms(keys: Sequence[Any], need: np.ndarray) -> np.ndarray:
    """V per feature: combine.seeded_uniform(*keys, 'B04', f) where needed
    (discrete features with P(X = x) > 0), 0.5 elsewhere (unused)."""
    v = np.full(NF, 0.5)
    for f in np.flatnonzero(need).tolist():
        v[f] = combine.seeded_uniform(*keys, SEED_TAG, f)
    return v


def score_features(pred_cur: MB.Pred, pred_ref: Optional[MB.Pred], pred_cls: Optional[MB.Pred],
                   nat: Sequence[float], dt_s: float, keys: Sequence[Any]) -> FeatureScores:
    """Per-feature scores of one observed row (feature.nat, real exposure dt)
    against the current and reference anchors and the class tier. A missing
    predictive (None) gives NaN for its outputs; p_f then falls back to the
    other anchor alone (bayes.combine_anchors). keys seed the randomised PIT
    (system, entity, ts)."""
    num, den, wf = MB.observe(nat, dt_s)
    u_c, p_c, eq_c = family_midp(pred_cur, num, den, wf)
    if pred_ref is not None:
        u_r, p_r, eq_r = family_midp(pred_ref, num, den, wf)
    else:
        u_r = p_r = np.full(NF, np.nan)
        eq_r = np.zeros(NF)
    if pred_cls is not None:
        _, p_k, _ = family_midp(pred_cls, num, den, wf, want_eq=False)
    else:
        p_k = np.full(NF, np.nan)
    v = pit_uniforms(keys, _DISCRETE & ((eq_c > 0.0) | (eq_r > 0.0)))
    z = pit_z(u_c, p_c, eq_c, v)
    zr = pit_z(u_r, p_r, eq_r, v)
    pf = np.asarray(bayes.combine_anchors(p_c, p_r), dtype=np.float64)
    return FeatureScores(z, zr, pf, p_c, p_r, p_k)


# ======================================================= batched scoring
# The engine scores every entity of a tick in one pass (perf, docs/lib3/
# integration.md §9): the predictives of all rows are built in one pass
# (m_baseline.predictive_set_many / predictive_q_many), and the per-feature
# mid-p of all rows and anchors are evaluated together. The NB mid-p goes
# through the array bayes kernel, which is bit-identical to the scalar path;
# the NB / BB pmf, the BB mid-p (summed tails) and the observation transform
# (whose CLR mean is a row reduction) stay on the scalar per-element paths of
# family_midp / score_features, so every output is bit-identical to the
# per-entity path (tests/engines/test_b04_batched.py). A vectorised BB tail
# sum (gammaln / numpy log-exp instead of math.lgamma and the scalar loop)
# agreed only to ~1e-11: enough to flip a float32 ring value now and then,
# after which B06's T² drifted on pack A, so it was not kept.
_PRED_FIELDS = ("mu", "r", "p", "c", "df", "loc", "scale")


def stack_preds(preds: Sequence[MB.Pred]) -> Dict[str, np.ndarray]:
    """Pred parameters stacked into [m, 52] arrays."""
    return {f: np.stack([np.asarray(getattr(p, f), dtype=np.float64) for p in preds])
            for f in _PRED_FIELDS}


def family_midp_many(P: Mapping[str, np.ndarray], num: np.ndarray, den: np.ndarray,
                     wf: np.ndarray, want_eq: bool = True
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """family_midp for m rows at once: P holds [m, 52] predictive parameters
    (stack_preds), num / den / wf the [m, 52] observation arrays. Returns
    (u, p, eq) [m, 52], bit-identical to family_midp row by row."""
    m = num.shape[0]
    u = np.full((m, NF), np.nan)
    p = np.full((m, NF), np.nan)
    eq = np.zeros((m, NF))
    rr, cc = np.nonzero(wf[:, MB.CNT] > 0.0)
    if rr.size:
        f = MB.CNT[cc]
        k, mm, r = num[rr, f], P["mu"][rr, f] * den[rr, f], P["r"][rr, f]
        uu, vv = bayes._nb_mid_a(k, mm, r)
        with np.errstate(invalid="ignore"):
            pp = np.clip(2.0 * np.minimum(uu, vv), _P_FLOOR, 1.0)
        u[rr, f] = np.where(np.isnan(pp), np.nan, uu)
        p[rr, f] = pp
        if want_eq:
            nb_pmf = bayes.nb_pmf
            for i, j, kk, mj, rj, pj in zip(rr.tolist(), f.tolist(), k.tolist(), mm.tolist(),
                                            r.tolist(), pp.tolist()):
                if pj == pj:
                    eq[i, j] = nb_pmf(kk, mj, rj)
    rr, cc = np.nonzero(wf[:, MB.RAT] > 0.0)
    if rr.size:
        f = MB.RAT[cc]
        pr, cr = P["p"][rr, f].tolist(), P["c"][rr, f].tolist()
        k, n = num[rr, f].tolist(), den[rr, f].tolist()
        bb_midp = bayes.bb_midp
        for t, (i, j) in enumerate(zip(rr.tolist(), f.tolist())):
            a, b = pr[t] * cr[t], (1.0 - pr[t]) * cr[t]
            uu, pp = bb_midp(k[t], n[t], a, b)
            u[i, j], p[i, j] = uu, pp
            if want_eq and pp == pp:
                eq[i, j] = _bb_pmf1(k[t], n[t], a, b)
    ok = wf[:, _NIG] > 0.0
    if ok.any():
        df, loc, scale = P["df"][:, _NIG], P["loc"][:, _NIG], P["scale"][:, _NIG]
        with np.errstate(all="ignore"):
            zt = (num[:, _NIG] - loc) / scale
            lo = sp.stdtr(df, zt)
            hi = sp.stdtr(df, -zt)
        good = np.isfinite(zt) & (scale > 0.0)
        un = np.where(good, lo, np.nan)
        pn = np.where(good, np.clip(2.0 * np.minimum(lo, hi), _P_FLOOR, 1.0), np.nan)
        u[:, _NIG] = np.where(ok, un, u[:, _NIG])
        p[:, _NIG] = np.where(ok, pn, p[:, _NIG])
    eq = np.where(np.isfinite(eq), eq, 0.0)
    return u, p, eq


def pit_uniforms_many(keys: Sequence[Sequence[Any]], need: np.ndarray) -> np.ndarray:
    """pit_uniforms for m rows: V[i, f] = combine.seeded_uniform(*keys[i], 'B04', f)
    where need[i, f], 0.5 elsewhere. The key prefix is rendered once per row
    (the same message bytes, so the same V)."""
    v = np.full(need.shape, 0.5)
    blake = hashlib.blake2b
    rk = combine._repr_key
    for i, row in enumerate(need):
        fs = np.flatnonzero(row).tolist()
        if not fs:
            continue
        pre = "|".join(map(rk, (*keys[i], SEED_TAG))) + "|"
        for f in fs:
            h = int.from_bytes(blake((pre + repr(f)).encode("utf-8"), digest_size=8).digest(),
                               "big")
            v[i, f] = combine._u_from_int(h)
    return v


def score_features_many(cur: Sequence[MB.Pred], ref: Sequence[Optional[MB.Pred]],
                        cls: Sequence[Optional[MB.Pred]], nat: np.ndarray,
                        dt_s: np.ndarray, keys: Sequence[Sequence[Any]]) -> FeatureScores:
    """score_features for m rows at once (fields are [m, 52] arrays): row i is
    scored against cur[i], ref[i] (None: NaN / fallback as in score_features)
    and cls[i] (None: NaN) with exposure dt_s[i] and PIT seed keys[i]."""
    nat = np.asarray(nat, dtype=np.float64).reshape(-1, NF)
    dt = np.asarray(dt_s, dtype=np.float64).reshape(-1)
    if not (np.all(dt > 0.0) and np.all(np.isfinite(dt))):
        raise ValueError(f"m_baseline: bad dt_s {dt_s!r}")
    m = nat.shape[0]
    # per row: the CLR centring is a row mean, whose batched reduction order differs
    obs = [MB.observe(nat[i], float(dt[i])) for i in range(m)]
    num = np.stack([o[0] for o in obs])
    den = np.stack([o[1] for o in obs])
    wf = np.stack([o[2] for o in obs])
    u_c, p_c, eq_c = family_midp_many(stack_preds(cur), num, den, wf)
    u_r = np.full((m, NF), np.nan)
    p_r = np.full((m, NF), np.nan)
    eq_r = np.zeros((m, NF))
    ir = [i for i, x in enumerate(ref) if x is not None]
    if ir:
        a = np.array(ir)
        u_r[a], p_r[a], eq_r[a] = family_midp_many(stack_preds([ref[i] for i in ir]),
                                                   num[a], den[a], wf[a])
    p_k = np.full((m, NF), np.nan)
    ik = [i for i, x in enumerate(cls) if x is not None]
    if ik:
        a = np.array(ik)
        p_k[a] = family_midp_many(stack_preds([cls[i] for i in ik]), num[a], den[a], wf[a],
                                  want_eq=False)[1]
    v = pit_uniforms_many(keys, _DISCRETE[None, :] & ((eq_c > 0.0) | (eq_r > 0.0)))
    z = pit_z(u_c, p_c, eq_c, v)
    zr = pit_z(u_r, p_r, eq_r, v)
    pf = np.asarray(bayes.combine_anchors(p_c, p_r), dtype=np.float64)
    return FeatureScores(z, zr, pf, p_c, p_r, p_k)


# ================================================================ channels
class GroupLayout:
    """Dependence groups (each inside one FEATURE_SPEC group) as membership
    matrices, for the two-stage harmonic-mean combination. Features no group
    mentions are their own singleton groups; a feature in two groups, or an
    index outside 0..51, is a malformed model.groups and raises."""

    def __init__(self, groups: Sequence[Sequence[int]]) -> None:
        gs = [sorted(int(i) for i in g) for g in groups if len(g)]
        seen = [i for g in gs for i in g]
        if len(set(seen)) != len(seen) or any(not 0 <= i < NF for i in seen):
            raise ValueError("B04: dependence groups must be disjoint subsets of 0..51")
        missing = sorted(set(range(NF)) - set(seen))
        gs = sorted(gs + [[i] for i in missing], key=lambda g: g[0])
        self.groups = gs
        G = len(gs)
        self.M = np.zeros((NF, G))
        fg = []
        for j, g in enumerate(gs):
            self.M[g, j] = 1.0
            names = {FEATURE_GROUP[FEATURE_NAMES_V2[i]] for i in g}
            if len(names) != 1:
                raise ValueError(f"B04: dependence group {g} spans feature groups {sorted(names)}")
            fg.append(names.pop())
        self.fg_of_group = fg
        self.fg_names = [n for n in GROUP_ORDER if n in fg]
        self.FG = np.zeros((G, len(self.fg_names)))
        for j, n in enumerate(fg):
            self.FG[j, self.fg_names.index(n)] = 1.0
        self.vol = np.array([n == INTENSITY_GROUP for n in fg], dtype=bool)


def _hmp_cols(p: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Equal-weight harmonic mean of the finite p in each column's member set
    (M: 0/1 [n, G]); NaN for a column with no finite member. p is clipped to
    [P_FLOOR, 1] as in combine.whmp."""
    ok = np.isfinite(p)
    inv = np.where(ok, 1.0 / np.clip(np.where(ok, p, 1.0), _P_FLOOR, 1.0), 0.0)
    n = ok.astype(np.float64) @ M
    s = inv @ M
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n > 0.0, n / s, np.nan)
    return np.minimum(out, 1.0)


def _hmp(p: np.ndarray) -> float:
    ok = np.isfinite(p)
    if not ok.any():
        return _NAN
    return min(1.0, float(ok.sum() / np.sum(1.0 / np.clip(p[ok], _P_FLOOR, 1.0))))


class Channel(NamedTuple):
    p_int: float            # volume dependence groups
    p_shape: float          # every other group
    p_all: float            # all groups
    p_fg: Dict[str, float]  # per FEATURE_SPEC group


def channels(p: np.ndarray, lay: GroupLayout) -> Channel:
    """Two-stage wHMP of per-feature p: within each dependence group (weight
    1/|group|), then across groups (equal weights)."""
    pg = _hmp_cols(p, lay.M)
    pfg = _hmp_cols(pg, lay.FG)
    return Channel(_hmp(pg[lay.vol]), _hmp(pg[~lay.vol]), _hmp(pg),
                   dict(zip(lay.fg_names, pfg.tolist())))


def _neglog10(p: float) -> Optional[float]:
    return -math.log10(max(p, _P_FLOOR)) if p == p else None


def _fin(p: float) -> Optional[float]:
    return float(p) if p == p else None


def _axes(p_fg: Mapping[str, float], names: Sequence[str]) -> List[str]:
    return [n for n in names if p_fg.get(n, _NAN) < AXIS_ALPHA]


def _js(x: float) -> Optional[float]:
    x = float(x)
    return float(f"{x:.6g}") if math.isfinite(x) else None


# ============================================================ model_state
def state_quantiles(pred: MB.Pred, qs: Sequence[float], dt_s: float) -> np.ndarray:
    """Predictive quantiles [len(qs), 52] in natural units for an exposure of
    dt_s: counts per dt, ratios as fractions at the expected trials of their
    exposure feature, t features as the inverse transform of loc + scale
    t_df^-1(q). Delegates to m_baseline.quantiles (its t-family inverse
    transform was fixed and its ratio path vectorised at integration)."""
    return MB.quantiles(pred, qs, dt_s=dt_s)


# ================================================================ engine
class LikelihoodEngine(Engine):
    name = "behavior.likelihood"
    layer = "behavior"
    consumes = [ACTIVE, NAT, TCTX, "feature.expo", MB.MODEL, "model.class",
                m_density.GROUPS_MODEL]
    produces = [Z, ZR, PF, emit.SCORE, emit.PM, emit.AXES, emit.DEGRADED,
                "profile.extra.model_state"]
    description = ("Exact exposure-aware NB / Beta-Binomial / Student-t predictive mid-p per "
                   "feature against the current and reference anchors (p = 2 min) and the "
                   "class tier; randomised-PIT normal scores z / zr; harmonic-mean channels "
                   "marg_int, marg_shape and peer over the dependence groups.")
    interval = 1

    def __init__(self, state_period_s: float = STATE_PERIOD_S, **params: Any) -> None:
        super().__init__(**params)
        self.state_period_s = float(state_period_s)
        self._state_seen: set = set()
        self._bounded = False
        self._layouts: Dict[Tuple[Tuple[int, ...], ...], GroupLayout] = {}

    # ---------------------------------------------------------------- run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        now, dt = float(ctx.now), float(ctx.window_s)
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError(f"LikelihoodEngine: bad ctx.window_s {ctx.window_s!r}")
        self._bounded = PA.bounded(ctx.config)
        # B01 failed: activity itself is unknown, so every entity is degraded
        # (no z rows: we cannot claim it was present). B03 failed: active
        # entities get NaN rows rather than scores against a stale model.
        b01_err = f"producer_error:{B01_ENGINE}" if store.engine_failed(B01_ENGINE, now) else None
        b03_err = f"producer_error:{B03_ENGINE}" if store.engine_failed(B03_ENGINE, now) else None
        if GR.canonical(ctx.config):
            return self._run_grains(ctx, store, now, dt, b01_err, b03_err)
        tctx0: Optional[Dict[str, Any]] = None
        n = 0
        jobs: List[_Job] = []
        for s in store.systems():
            ents = PA.entities(store, s, now, ctx.config)      # bounded mode: active + earned (§10.3)
            if not ents:
                continue
            lay = self._layout(m_density.groups(store, s, split_by_feature_group=True))
            for e in ents:
                if b01_err is not None:
                    self._degraded(store, s, e, now, dt, b01_err, rows=False)
                    n += 1
                    continue
                if not _active(store, s, e, now):
                    continue
                nat = store.vec_at(s, e, NAT, now)
                if nat is None or b03_err is not None:
                    self._degraded(store, s, e, now, dt, b03_err or f"stale:{NAT}", rows=True)
                    n += 1
                    continue
                tctx = _tctx_at(store, s, e, now)
                if tctx is None:
                    if tctx0 is None:
                        tctx0 = TB.tctx_from_config(now, ctx.config, dt)
                    tctx = tctx0
                jobs.append(_Job(s, e, dt, np.asarray(nat, dtype=np.float64), tctx, lay, None,
                                 dt))
        return n + self._score_jobs(store, now, jobs)

    # ------------------------------------------------------ spec v2.1 grains
    def _run_grains(self, ctx: Context, store: Any, now: float, dt: float,
                    b01_err: Optional[str], b03_err: Optional[str]) -> int:
        """Canonical grain mode: the H pass on H decision ticks, the Q pass on
        Q decision ticks (module docstring)."""
        due = GR.due(now, dt, GR.CANONICAL)
        if not (due["h"] or due["q"]):
            return 0
        self._ensure_retention(store)
        masks = {g: GR.scored_mask(now, dt, g, ctx.config) for g in GR.GRAINS if due[g]}
        tcs = {g: GR.row_tctx(now, g, dt, ctx.config) for g in GR.GRAINS if due[g]}
        n = 0
        jobs: List[_Job] = []
        for s in store.systems():
            ents = PA.entities(store, s, now, ctx.config)      # bounded mode: active + earned (§10.3)
            if not ents:
                continue
            lay = self._layout(m_density.groups(store, s, split_by_feature_group=True))
            for e in ents:
                for g in GR.GRAINS:
                    if not due[g]:
                        continue
                    dets = DETS if g == "h" else DETS_Q
                    if b01_err is not None:
                        self._degraded(store, s, e, now, dt, b01_err, rows=False, dets=dets)
                        n += 1
                        continue
                    meta = store.vec_at(s, e, f"feature.meta.{g}", now)
                    if meta is None or not float(meta[0]) > 0.5:
                        continue
                    nat = store.vec_at(s, e, f"feature.nat.{g}", now)
                    if nat is None or b03_err is not None:
                        self._degraded(store, s, e, now, dt, b03_err or f"stale:feature.nat.{g}",
                                       rows=True, dets=dets, g=g)
                        n += 1
                        continue
                    x = np.asarray(nat, dtype=np.float64).copy()
                    x[~masks[g]] = np.nan
                    cov = float(meta[1])
                    jobs.append(_Job(s, e, cov, x, tcs[g], lay, g, dt))
        return n + self._score_jobs(store, now, jobs)

    def _ensure_retention(self, store: Any) -> None:
        """H-grain behaviour rings: 4 d (96 H rows; B14 / B29 replay), raise-only."""
        if getattr(self, "_ret_store", None) is store:
            return
        for name in (Z, ZR, PF):
            store.ensure_retention(name, max_age_s=4 * 86400.0)
        self._ret_store = store

    def _score_jobs(self, store: Any, now: float, jobs: List["_Job"]) -> int:
        """Score every collected row of this tick in one batch (perf): the
        predictives per row (m_baseline), then score_features_many over all
        rows, then the per-row writes of _entity / _entity_q. B04 reads
        nothing it writes, so this is the per-entity loop reordered."""
        if not jobs:
            return 0
        cur: List[MB.Pred] = []
        ref: List[Optional[MB.Pred]] = []
        cls: List[Optional[MB.Pred]] = []
        X = np.empty((len(jobs), NF))
        keys: List[Tuple[Any, ...]] = []
        # predictives of every row, batched per kind (m_baseline *_many)
        iq = [i for i, j in enumerate(jobs) if j.grain == "q"]
        ih = [i for i, j in enumerate(jobs) if j.grain != "q"]
        pset: List[Any] = [None] * len(jobs)
        for idx, fn in ((iq, MB.predictive_q_many), (ih, MB.predictive_set_many)):
            for i, d in zip(idx, fn(store, [(jobs[i].s, jobs[i].e, jobs[i].tctx) for i in idx])):
                pset[i] = d
        for i, j in enumerate(jobs):
            if j.grain == "q":
                preds = pset[i]
                c = preds["current"]
                X[i] = j.nat if c.scored is None else np.where(c.scored, j.nat, np.nan)
                cur.append(c)
                ref.append(preds["reference"])
                cls.append(None)
                keys.append((j.s, j.e, now, "q"))
            else:
                preds = pset[i]
                has_tier = MB.tier_model(store, j.s, MB.parent_key(store, j.s, j.e)) is not None
                X[i] = j.nat
                cur.append(preds["current"])
                ref.append(preds["reference"])
                cls.append(preds["class"] if has_tier else None)
                keys.append((j.s, j.e, now))
        sc = score_features_many(cur, ref, cls, X, np.array([j.dt for j in jobs]), keys)
        if self._bounded:
            self._earned_gain(store, now, jobs, sc, cls)
        # model_state refreshes due at this tick: all quantiles in one batch
        due = [i for i, j in enumerate(jobs) if self._state_take(j.s, j.e, now, j.grain)]
        state_q: List[Optional[np.ndarray]] = [None] * len(jobs)
        if due:
            Q = MB.quantiles_many([cur[i] for i in due], STATE_QS,
                                  [self._state_dt(jobs[i].grain) for i in due])
            for k, i in enumerate(due):
                state_q[i] = Q[k]
        for i, j in enumerate(jobs):
            row = FeatureScores(*(f[i] for f in sc))
            if j.grain == "q":
                self._write_q(store, j.s, j.e, now, row, cur[i], j.lay, j.w, state_q=state_q[i])
            else:
                self._write_h(store, j.s, j.e, now, j.dt, row, cur[i], cls[i] is not None,
                              j.lay, grain=j.grain, state_q=state_q[i])
        return len(jobs)

    def _earned_gain(self, store: Any, now: float, jobs: List["_Job"], sc: Any,
                     cls: List[Optional[MB.Pred]]) -> None:
        """Bounded mode (progressive.md §10.2): per system, the heavy hitters
        of active IP-rows (SpaceSaving, k_sh = EARN_K) keep an earned-gain
        record: the H_m-decayed sum over their scored rows of
            g_row = sum_f clip(log2 p_cur,f - log2 p_cls,f, -8, 8)
        i.e. how many bits of surprisal the IP's own model saves on its rows
        over its class model (the prequential scores of this tick, computed
        before either model learns the row), and the decayed number of rows.
        P15 turns the records into the earned set E_t(s) (g / n >= 2 bits per
        row after >= 48 rows, top E_max). Memory O(EARN_K) per system."""
        by_sys: Dict[str, List[int]] = {}
        for i, j in enumerate(jobs):
            if j.grain != "q" and cls[i] is not None:
                by_sys.setdefault(j.s, []).append(i)
        for s, idx in by_sys.items():
            rec = store.get_model(s, "__system__", EARNED_MODEL)
            if not isinstance(rec, dict) or rec.get("fmt") != 1:
                rec = {"fmt": 1, "ips": {}, "cand": PS_EARN.DecayedSpaceSaving(EARN_K, [EARN_HL], [EARN_HL], 0),
                       "ts": now}
            cand, ips = rec["cand"], rec["ips"]
            f = 2.0 ** (-(now - float(rec.get("ts", now))) / EARN_HL)
            if f < 1.0:
                for r in ips.values():
                    r["g"] *= f
                    r["n"] *= f
            for i in idx:
                e = jobs[i].e
                pc, pk = sc.p_cur[i], sc.p_cls[i]
                ok = np.isfinite(pc) & np.isfinite(pk) & (pc > 0) & (pk > 0)
                if not ok.any():
                    continue
                g = float(np.clip(np.log2(pc[ok]) - np.log2(pk[ok]), -8.0, 8.0).sum())
                cand.add(e, now, 1.0, 1.0)
                r = ips.get(e)
                if r is None:
                    r = ips[e] = {"g": 0.0, "n": 0.0}
                r["g"] += g
                r["n"] += 1.0
            keep = set(cand.keys())
            for e in [e for e in ips if e not in keep]:
                del ips[e]
            rec["ts"] = now
            store.put_model(s, "__system__", EARNED_MODEL, rec, ts=now)

    def _entity_q(self, store: Any, s: str, e: str, now: float, cov: float, nat: np.ndarray,
                  tctx: Mapping[str, Any], lay: "GroupLayout", dt: float) -> int:
        """The Q pass of one entity (m_baseline.predictive_q); the engine
        batches the same steps in _score_jobs."""
        preds = MB.predictive_q(store, s, e, tctx)
        cur = preds["current"]
        x = nat
        if cur.scored is not None:
            x = np.where(cur.scored, nat, np.nan)
        sc = score_features(cur, preds["reference"], None, x, cov, (s, e, now, "q"))
        self._write_q(store, s, e, now, sc, cur, lay, dt)
        return 1

    def _write_q(self, store: Any, s: str, e: str, now: float, sc: FeatureScores,
                 cur: MB.Pred, lay: "GroupLayout", dt: float, state_q: Any = _UNSET) -> None:
        w = int(round(dt))
        store.add_vec(s, e, Z + ".q", now, sc.z.astype(np.float32), window_s=w)
        store.add_vec(s, e, ZR + ".q", now, sc.zr.astype(np.float32), window_s=w)
        store.add_vec(s, e, PF + ".q", now, sc.pf.astype(np.float32), window_s=w)
        marg = channels(sc.pf, lay)
        scores = {"marg_int_q": _neglog10(marg.p_int), "marg_shape_q": _neglog10(marg.p_shape)}
        pm = {"marg_int_q": _fin(marg.p_int), "marg_shape_q": _fin(marg.p_shape)}
        shape_names = [g for g in lay.fg_names if g != INTENSITY_GROUP]
        axes = {"marg_int_q": _axes(marg.p_fg, [INTENSITY_GROUP]),
                "marg_shape_q": _axes(marg.p_fg, shape_names)}
        prov = cur.prov if cur.prov is not None else np.ones(NF)
        used = np.isfinite(sc.pf)
        pi = float(np.median(prov[used])) if used.any() else 1.0
        # contract M: a Q score on the provisional H -> Q transfer (native
        # share pi < 0.5) is a degraded run (B24 calibrates it in its own
        # 'p:1' stratum and writes the same cause; the merge is idempotent)
        dg = ({d: PROV_CAUSE for d in DETS_Q if pm[d] is not None} if pi < PROV_MIN else None)
        emit.write_scores(store, s, e, now, scores, pm=pm,
                          axes={d: a for d, a in axes.items() if a} or None,
                          degraded=dg or None, window_s=w)
        store.upsert_dict(s, e, PROV, now, {d: pi for d in DETS_Q}, w)
        self._model_state(store, s, e, now, cur, grain="q", prov=pi, q=state_q)

    def _layout(self, groups: List[List[int]]) -> GroupLayout:
        """GroupLayout per distinct partition (model.groups changes rarely)."""
        key = tuple(tuple(g) for g in groups)
        lay = self._layouts.get(key)
        if lay is None:
            if len(self._layouts) >= 16:
                self._layouts.clear()
            lay = self._layouts[key] = GroupLayout(groups)
        return lay

    # --------------------------------------------------------- one entity
    def _entity(self, store: Any, s: str, e: str, now: float, dt: float, nat: np.ndarray,
                tctx: Mapping[str, Any], lay: GroupLayout, grain: Optional[str] = None) -> int:
        """One scored row. v2: the tick row (dt = the tick). spec v2.1 H pass:
        the H row with dt = its coverage (grain='h'). The engine batches the
        same steps in _score_jobs."""
        preds = MB.predictive_set(store, s, e, tctx)
        has_tier = MB.tier_model(store, s, MB.parent_key(store, s, e)) is not None
        sc = score_features(preds["current"], preds["reference"],
                            preds["class"] if has_tier else None,
                            np.asarray(nat, dtype=np.float64), dt, (s, e, now))
        self._write_h(store, s, e, now, dt, sc, preds["current"], has_tier, lay, grain=grain)
        return 1

    def _write_h(self, store: Any, s: str, e: str, now: float, dt: float, sc: FeatureScores,
                 cur: MB.Pred, has_tier: bool, lay: GroupLayout,
                 grain: Optional[str] = None, state_q: Any = _UNSET) -> None:
        w = int(round(dt))
        store.add_vec(s, e, Z, now, sc.z.astype(np.float32), window_s=w)
        store.add_vec(s, e, ZR, now, sc.zr.astype(np.float32), window_s=w)
        store.add_vec(s, e, PF, now, sc.pf.astype(np.float32), window_s=w)

        marg = channels(sc.pf, lay)
        peer = channels(sc.p_cls, lay) if has_tier else None
        p_peer = peer.p_all if peer is not None else _NAN
        scores = {"marg_int": _neglog10(marg.p_int), "marg_shape": _neglog10(marg.p_shape),
                  "peer": _neglog10(p_peer)}
        pm = {"marg_int": _fin(marg.p_int), "marg_shape": _fin(marg.p_shape),
              "peer": _fin(p_peer)}
        shape_names = [g for g in lay.fg_names if g != INTENSITY_GROUP]
        axes = {"marg_int": _axes(marg.p_fg, [INTENSITY_GROUP]),
                "marg_shape": _axes(marg.p_fg, shape_names),
                "peer": _axes(peer.p_fg, lay.fg_names) if peer is not None else []}
        emit.write_scores(store, s, e, now, scores, pm=pm,
                          axes={d: a for d, a in axes.items() if a} or None, window_s=w)
        self._model_state(store, s, e, now, cur, grain=grain, q=state_q)

    def _degraded(self, store: Any, s: str, e: str, now: float, dt: float, cause: str,
                  rows: bool, dets: Sequence[str] = DETS, g: Optional[str] = None) -> None:
        w = int(round(dt))
        if rows:
            suf = ".q" if g == "q" else ""
            for name in (Z, ZR, PF):
                store.add_vec(s, e, name + suf, now, _NAN_ROW, window_s=w)
        emit.write_scores(store, s, e, now, {d: None for d in dets},
                          degraded={d: cause for d in dets}, window_s=w)

    # -------------------------------------------------------- model_state
    def _state_take(self, s: str, e: str, now: float, grain: Optional[str]) -> bool:
        """Whether model_state is refreshed for (s, e, grain) at now (marks it)."""
        key = (s, e) if grain is None else (s, e, grain)
        dkey = (s, e, "model_state") + ((grain,) if grain else ())     # v2 key in tick mode
        due = self.entity_due(dkey, now, self.state_period_s)
        if not due and key in self._state_seen:
            return False
        self._state_seen.add(key)
        return True

    @staticmethod
    def _state_dt(grain: Optional[str]) -> float:
        return STATE_DT_S if grain is None else GR.GRAIN_S[grain]

    def _model_state(self, store: Any, s: str, e: str, now: float, pred: MB.Pred,
                     grain: Optional[str] = None, prov: Optional[float] = None,
                     q: Any = _UNSET) -> None:
        """profile.extra.model_state: once per state_period_s per entity, at a
        per-entity phase (entity_due) so refreshes spread over the hour
        instead of all landing on the bucket boundary; at once for an entity
        without one. It describes the bucket current at the refresh.

        spec v2.1: per grain under 'grains' {h: {...}, q: {..., prov}} at
        dt_s = 3600 / 900; the legacy top-level fields are those of Q when
        Q is observable, else H."""
        if q is _UNSET:                  # unbatched call: decide and compute here
            if not self._state_take(s, e, now, grain):
                return
            q = state_quantiles(pred, STATE_QS, self._state_dt(grain))
        elif q is None:                  # batched: _score_jobs found it not due
            return
        dt_s = self._state_dt(grain)
        feats = {name: [_js(q[0, i]), _js(q[1, i]), _js(q[2, i])]
                 for i, name in enumerate(FEATURE_NAMES_V2)}
        block = {"ts": now, "bucket": int(pred.bucket), "mode": pred.mode, "tier": pred.tier,
                 "anchor": pred.anchor, "dt_s": dt_s, "q": list(STATE_QS), "features": feats}
        prof = store.profile(s, e) or EntityProfile(system=s, entity=e, updated=now)
        if grain is None:
            prof.extra["model_state"] = block
        else:
            if prov is not None:
                block["prov"] = float(prov)
                block["provisional"] = bool(prov < PROV_MIN)
            ms = dict(prof.extra.get("model_state") or {})
            grains = dict(ms.get("grains") or {})
            grains[grain] = block
            legacy = grains.get("q") or grains.get("h")
            ms = dict(legacy)
            ms["grains"] = grains
            prof.extra["model_state"] = ms
        store.put_profile(prof)


# ================================================================ helpers
class _Job(NamedTuple):
    """One row to score this tick (collected per entity, scored in a batch)."""
    s: str
    e: str
    dt: float                    # exposure: the tick (v2) or the grain row's coverage
    nat: np.ndarray              # [52] float64, masked for the grain
    tctx: Mapping[str, Any]
    lay: "GroupLayout"
    grain: Optional[str]         # None (tick mode), 'h' or 'q'
    w: float                     # the tick's dt (window_s of the Q writes)


def _active(store: Any, s: str, e: str, now: float) -> bool:
    a = store.vec_at(s, e, ACTIVE, now)
    return a is not None and a.size > 0 and float(a[0]) > 0.5


def _tctx_at(store: Any, s: str, e: str, now: float) -> Optional[Mapping[str, Any]]:
    """feature.tctx written by B01 at exactly now, or None."""
    m = store.latest_derived(s, e, TCTX)
    if m is not None and m.ts == now and isinstance(m.value, Mapping) \
            and m.value.get("bin48") is not None:
        return m.value
    return None
