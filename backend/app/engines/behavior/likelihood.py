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
costs ~110 us of dispatch for 10 features).

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

import math
from collections.abc import Mapping
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from scipy import special as sp

from ...core.engine import Context, Engine
from ...models.schema import EntityProfile
from .lib import bayes
from .lib import combine
from .lib import emit
from .lib import m_baseline as MB
from .lib import m_density
from .lib import timebins as TB
from .lib.features import (FEATURE_DIM, FEATURE_GROUP, FEATURE_KIND, FEATURE_NAMES_V2,
                           GROUP_ORDER, VEC_TX)

Z = "behavior.z"
ZR = "behavior.zr"
PF = "behavior.pf"
ACTIVE = "feature.active"
NAT = "feature.nat"
TCTX = "feature.tctx"
B01_ENGINE = "behavior.feature_vector"
B03_ENGINE = "behavior.baseline"

DETS = ("marg_int", "marg_shape", "peer")
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
def _t_back_code(name: str) -> int:
    """Inverse of the t-family observation transform (m_baseline's TX codes):
    0 identity, 1 exp, 2 expm1, 3 expit, 4 bytes expm1(y) dt/60."""
    kind, tx = FEATURE_KIND[name], VEC_TX.get(name)
    if kind == "bytes":
        return 4
    if kind == "avg":
        return 0 if tx == "identity" else 2 if tx == "log1p" else 1
    if kind in ("bounded", "ratio"):
        return 3
    if kind in ("window", "gauge"):
        return 2 if tx == "log1p" else 1 if tx == "log" else 0
    return 0                                            # clr: log-ratio units


_T_BACK = np.array([_t_back_code(FEATURE_NAMES_V2[f]) for f in _NIG.tolist()])


def state_quantiles(pred: MB.Pred, qs: Sequence[float], dt_s: float) -> np.ndarray:
    """Predictive quantiles [len(qs), 52] in natural units for an exposure of
    dt_s (m_baseline.quantiles semantics): counts per dt (NB ppf), ratios as
    fractions at the expected trials of their exposure feature (BB ppf, one
    pmf pass per feature for all qs), t features as the inverse transform of
    loc + scale t_df^-1(q) (monotone, so quantiles carry over).

    Formed here rather than by m_baseline.quantiles, which (a) evaluates
    bb_ppf once per (q, feature), 3x the passes (~4 ms per entity), and (b)
    currently applies its t-family inverse transforms to NIG-block positions
    of the full 52-column row (reported to its owner)."""
    q = np.clip(np.asarray(qs, dtype=np.float64).reshape(-1), 0.0, 1.0)[:, None]
    e = float(dt_s) / 60.0
    out = np.full((q.shape[0], NF), np.nan)
    cnt = MB.CNT
    out[:, cnt] = bayes.nb_ppf(q, (pred.mu[cnt] * e)[None, :], pred.r[cnt][None, :])
    rat = MB.RAT
    n = pred.mu[MB.RATIO_N_IDX] * e
    n = np.where(np.isfinite(n) & (n >= 1.0), np.round(n), 1.0)
    a, b = pred.p[rat] * pred.c[rat], (1.0 - pred.p[rat]) * pred.c[rat]
    out[:, rat] = np.asarray(bayes.bb_ppf(q, n[None, :], a[None, :], b[None, :])) / n[None, :]
    with np.errstate(all="ignore"):
        y = pred.loc[_NIG][None, :] + pred.scale[_NIG][None, :] * sp.stdtrit(
            pred.df[_NIG][None, :], q)
        v = np.select([_T_BACK == 1, _T_BACK == 2, _T_BACK == 3, _T_BACK == 4],
                      [np.exp(y), np.expm1(y), sp.expit(y), np.expm1(y) * dt_s / 60.0], y)
    out[:, _NIG] = v
    return out


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
        self._layouts: Dict[Tuple[Tuple[int, ...], ...], GroupLayout] = {}

    # ---------------------------------------------------------------- run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        now, dt = float(ctx.now), float(ctx.window_s)
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError(f"LikelihoodEngine: bad ctx.window_s {ctx.window_s!r}")
        # B01 failed: activity itself is unknown, so every entity is degraded
        # (no z rows: we cannot claim it was present). B03 failed: active
        # entities get NaN rows rather than scores against a stale model.
        b01_err = f"producer_error:{B01_ENGINE}" if store.engine_failed(B01_ENGINE, now) else None
        b03_err = f"producer_error:{B03_ENGINE}" if store.engine_failed(B03_ENGINE, now) else None
        tctx0: Optional[Dict[str, Any]] = None
        n = 0
        for s in store.systems():
            ents = store.entities(s)
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
                n += self._entity(store, s, e, now, dt, nat, tctx, lay)
        return n

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
                tctx: Mapping[str, Any], lay: GroupLayout) -> int:
        preds = MB.predictive_set(store, s, e, tctx)
        has_tier = MB.tier_model(store, s, MB.parent_key(store, s, e)) is not None
        sc = score_features(preds["current"], preds["reference"],
                            preds["class"] if has_tier else None,
                            np.asarray(nat, dtype=np.float64), dt, (s, e, now))
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
        self._model_state(store, s, e, now, preds["current"])
        return 1

    def _degraded(self, store: Any, s: str, e: str, now: float, dt: float, cause: str,
                  rows: bool) -> None:
        w = int(round(dt))
        if rows:
            for name in (Z, ZR, PF):
                store.add_vec(s, e, name, now, _NAN_ROW, window_s=w)
        emit.write_scores(store, s, e, now, {d: None for d in DETS},
                          degraded={d: cause for d in DETS}, window_s=w)

    # -------------------------------------------------------- model_state
    def _model_state(self, store: Any, s: str, e: str, now: float, pred: MB.Pred) -> None:
        """profile.extra.model_state: once per state_period_s per entity, at a
        per-entity phase (entity_due) so refreshes spread over the hour
        instead of all landing on the bucket boundary; at once for an entity
        without one. It describes the bucket current at the refresh."""
        key = (s, e)
        due = self.entity_due((s, e, "model_state"), now, self.state_period_s)
        if not due and key in self._state_seen:
            return
        self._state_seen.add(key)
        q = state_quantiles(pred, STATE_QS, STATE_DT_S)
        feats = {name: [_js(q[0, i]), _js(q[1, i]), _js(q[2, i])]
                 for i, name in enumerate(FEATURE_NAMES_V2)}
        prof = store.profile(s, e) or EntityProfile(system=s, entity=e, updated=now)
        prof.extra["model_state"] = {
            "ts": now, "bucket": int(pred.bucket), "mode": pred.mode, "tier": pred.tier,
            "anchor": pred.anchor, "dt_s": STATE_DT_S, "q": list(STATE_QS), "features": feats}
        store.put_profile(prof)


# ================================================================ helpers
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
