"""B04 LikelihoodEngine (docs/lib3/engines.md '## B04'): spec unit tests
(a)-(f) with the known corrections, plus engine-level edge cases.

The tail maths is tested on hand-built m_baseline.Pred objects (exact
parameters, as the spec states them); the engine is tested on a store whose
model.baseline was trained by the real B03 (constant 22 requests per 900-s
tick, so kappa-hat sits at its 1e3 cap).

Spec test mapping (known corrections applied):
  (a) test_a_poisson_22_lower_tail / test_a_nb_r20 / test_a_engine_trained_baseline
  (b) test_b_ratio_1_of_2_not_significant
  (c) test_c_ratio_100_of_200
  (d) test_d_null_draws_randomised_pit   z from the seeded randomised PIT is
      exactly N(0, 1) (KS on the PIT, count / ratio / t); KS on p_f for a
      continuous feature with one anchor (p_f = p there); with two identical
      anchors p_f = min(1, 2p) is conservative, never anti-conservative
  (e) test_e_dual_anchor_reference_drives_pf
  (f) test_f_ratio_n0_is_nan / test_f_engine_ratio_n0_is_nan
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest
from scipy import stats

from helpers import DT, T0, make_store, run_engine, set_trust

from app.core.engine import default_config
from app.engines.behavior.baseline import BaselineEngine
from app.engines.behavior.likelihood import (
    DETS, PF, Z, ZR, GroupLayout, LikelihoodEngine, channels, score_features,
    state_quantiles)
from app.engines.behavior.lib import emit
from app.engines.behavior.lib import features as F
from app.engines.behavior.lib import m_baseline as MB
from app.engines.behavior.lib import m_density
from app.engines.behavior.lib import timebins as TB

S = "erp"
ENTS = ("10.0.0.1", "10.0.0.2", "10.0.0.3")
IDX = F.FEATURE_INDEX
REQ = IDX["http_requests"]
FLOWS = IDX["flows"]
WRITE = IDX["http_write_ratio"]
LAT = IDX["http_latency"]
BYTES = IDX["bytes_up"]
COUNT_IDX = [i for i, n in enumerate(F.FEATURE_NAMES_V2) if F.FEATURE_KIND[n] == "count"]
NF = F.FEATURE_DIM
CFG = default_config({"strict": True})
TRAIN_TICKS = 3 * 96
# T0 is Wednesday 06:13 in Asia/Shanghai: train Monday 06:13 .. Thursday
# 06:13 so every test tick after training falls in trained workday buckets
TRAIN_T0 = T0 - 2 * 86400.0


# ================================================================ helpers
def make_pred(nb=None, bb=None, t=None) -> MB.Pred:
    """Pred with only the given features defined. nb {f: (mean per tick at
    dt=900, r)}, bb {f: (p, c)}, t {f: (df, loc, scale)} (loc/scale in the
    FEATURE_SPEC transform space)."""
    arr = {k: np.full(NF, np.nan) for k in ("mu", "r", "p", "c", "df", "loc", "scale")}
    for f, (m, r) in (nb or {}).items():
        arr["mu"][f], arr["r"][f] = m / 15.0, r
    for f, (p, c) in (bb or {}).items():
        arr["p"][f], arr["c"][f] = p, c
    for f, (df, loc, sc) in (t or {}).items():
        arr["df"][f], arr["loc"][f], arr["scale"][f] = df, loc, sc
    mean = np.where(np.isfinite(arr["mu"]), arr["mu"],
                    np.where(np.isfinite(arr["p"]), arr["p"], arr["loc"]))
    return MB.Pred(mean=mean, **arr)


def nat_of(**vals) -> np.ndarray:
    """feature.nat row with only the named features set (NaN elsewhere)."""
    x = np.full(NF, np.nan)
    for name, v in vals.items():
        x[IDX[name]] = v
    return x


def score1(cur, nat, ref="same", cls=None, keys=(S, ENTS[0], T0)):
    return score_features(cur, cur if ref == "same" else ref, cls, nat, DT, keys)


def base_row(rng, dt=DT, req=22.0):
    """B01-like feature.nat: counts raw per tick (stale -> 0), ratio k/n,
    averages natural units."""
    x = np.full(NF, np.nan)
    x[COUNT_IDX] = 0.0
    x[REQ] = req * dt / 900.0
    x[FLOWS] = float(round(5.0 * dt / 900.0))
    k = rng.binomial(int(x[REQ]), 0.1) if x[REQ] >= 1 else 0
    if x[REQ] >= 1:
        x[WRITE] = k / x[REQ]
        x[LAT] = 100.0 * rng.lognormal(0.0, 0.2)
    x[BYTES] = 1e5 * dt / 900.0 * rng.lognormal(0.0, 0.2)
    return x


def feed(store, t, nat, e=ENTS[0], dt=DT, active=1.0, tctx=True):
    store.add_vec(S, e, "feature.nat", t, np.asarray(nat, dtype=np.float32), window_s=int(dt))
    store.add_vec(S, e, "feature.active", t, np.array([active], dtype=np.float32),
                  window_s=int(dt))
    store.register_entity(S, e)
    if tctx:
        from app.models.schema import DerivedMetric, MetricKind
        store.add_derived(DerivedMetric(name="feature.tctx", value=TB.tctx_from_config(t, CFG, dt),
                                        ts=t, system=S, entity=e, window_s=int(dt),
                                        kind=MetricKind.CATEGORICAL))


class Trained:
    """A store whose model.baseline (entity, reference, system tier) was
    fitted by the real B03 over TRAIN_TICKS of constant-rate traffic."""

    def __init__(self) -> None:
        self.store = make_store()
        b03 = BaselineEngine()
        rng = np.random.default_rng(0)
        for i in range(TRAIN_TICKS):
            t = TRAIN_T0 + i * DT
            for e in ENTS:
                feed(self.store, t, base_row(rng), e=e)
                set_trust(self.store, S, e, [t], 1.0)
            run_engine(b03, self.store, t)
        self.t_next = TRAIN_T0 + TRAIN_TICKS * DT
        self.rng = rng

    def tick(self, dt=DT) -> float:
        """A fresh tick timestamp after everything written so far."""
        t = self.t_next
        self.t_next += dt
        return t


@pytest.fixture(scope="module")
def trained() -> Trained:
    return Trained()


def z_row(store, t, e=ENTS[0], name=Z):
    r = store.vec_at(S, e, name, t)
    return None if r is None else np.asarray(r, dtype=np.float64)


# ================================================================ (a)
def test_a_poisson_22_lower_tail():
    cur = make_pred(nb={REQ: (22.0, 1e3)})          # kappa-hat at its 1e3 cap, a -> inf
    assert stats.poisson.cdf(1, 22) == pytest.approx(6.4e-9, rel=0.02)
    sc1 = score1(cur, nat_of(http_requests=1.0))
    assert sc1.pf[REQ] < 1e-7
    sc3 = score1(cur, nat_of(http_requests=3.0))
    assert sc3.pf[REQ] < 1e-5
    assert sc3.z[REQ] < -4.5 and sc3.zr[REQ] < -4.5
    # z is below -4.5 whatever the PIT draw: F(2) and F(3) both give it
    for ts in range(20):
        assert score1(cur, nat_of(http_requests=3.0), keys=(S, "x", float(ts))).z[REQ] < -4.5
    # a central observation is unremarkable
    assert score1(cur, nat_of(http_requests=22.0)).pf[REQ] > 0.5


def test_a_nb_r20():
    cur = make_pred(nb={REQ: (22.0, 20.0)})
    assert stats.nbinom.cdf(3, 20, 20 / 42) == pytest.approx(1.04e-4, rel=0.02)
    sc = score1(cur, nat_of(http_requests=3.0), ref=None)
    assert sc.p_cur[REQ] < 1e-3
    sc = score1(cur, nat_of(http_requests=3.0))
    assert sc.pf[REQ] < 1e-3


def test_a_engine_trained_baseline(trained):
    """Through the engine against B03's model: a 22 -> 1 collapse is extreme,
    its z is far negative, marg_int fires on the volume axis."""
    st = trained.store
    t = trained.tick()
    x = base_row(trained.rng)
    x[REQ] = 1.0
    x[WRITE] = 0.0
    x[LAT] = 100.0
    feed(st, t, x)
    n = run_engine(LikelihoodEngine(), st, t)
    assert n == 1
    pf = z_row(st, t, name=PF)
    z = z_row(st, t)
    assert pf[REQ] < 1e-6 and z[REQ] < -4.5
    sc = emit.read_row(st, S, ENTS[0], emit.SCORE, t)
    pm = emit.read_row(st, S, ENTS[0], emit.PM, t)
    assert sc["marg_int"] > 5.0
    assert pm["marg_int"] == pytest.approx(10 ** -sc["marg_int"], rel=1e-6)
    assert "volume" in emit.read_dict(st, S, ENTS[0], emit.AXES, t).get("marg_int", [])


# ================================================================ (b), (c)
def test_b_ratio_1_of_2_not_significant():
    cur = make_pred(bb={WRITE: (0.1, 50.0)})
    x = nat_of(http_requests=2.0, http_write_ratio=0.5)
    sc = score1(cur, x)
    a, b = 0.1 * 50, 0.9 * 50
    from app.engines.behavior.lib import bayes
    assert bayes.bb_sf(0, 2, a, b) == pytest.approx(0.188, abs=5e-4)
    assert sc.p_cur[WRITE] > 0.05 and sc.pf[WRITE] > 0.05
    assert np.isfinite(sc.z[WRITE]) and abs(sc.z[WRITE]) < 3.0


@pytest.mark.parametrize("phi, bound, sf", [(50.0, 1e-6, 1.3e-8), (20.0, 1e-3, 7.7e-5)])
def test_c_ratio_100_of_200(phi, bound, sf):
    from app.engines.behavior.lib import bayes
    a, b = 0.1 * phi, 0.9 * phi
    assert bayes.bb_sf(99, 200, a, b) == pytest.approx(sf, rel=0.05)
    cur = make_pred(bb={WRITE: (0.1, phi)})
    sc = score1(cur, nat_of(http_requests=200.0, http_write_ratio=0.5))
    assert sc.p_cur[WRITE] < bound and sc.pf[WRITE] < bound
    assert sc.z[WRITE] > 3.0


# ================================================================ (d)
def test_d_null_draws_randomised_pit():
    """2000 draws from the predictive: z ~ N(0, 1) for count, ratio and t
    features (seeded randomised PIT; the mid-p itself is not uniform for a
    discrete X), p_f uniform for a continuous feature scored against one
    anchor, and conservative (P(p_f <= a) <= a) with two identical anchors."""
    rng = np.random.default_rng(7)
    n_draw = 2000
    df, loc, scale = 30.0, math.log(100.0), 0.2
    cur = make_pred(nb={REQ: (22.0, 1e12)}, bb={WRITE: (0.1, 60.0)}, t={LAT: (df, loc, scale)})
    zs = np.empty((n_draw, 3))
    pf2 = np.empty((n_draw, 3))
    pf1 = np.empty(n_draw)
    for i in range(n_draw):
        k = float(rng.poisson(22.0))
        pw = rng.beta(6.0, 54.0)
        w = rng.binomial(int(k), pw) / k if k > 0 else np.nan
        y = loc + scale * rng.standard_t(df)
        x = nat_of(http_requests=k, http_write_ratio=w, http_latency=math.exp(y))
        sc = score_features(cur, cur, None, x, DT, (S, "null", float(i)))
        zs[i] = sc.z[[REQ, WRITE, LAT]]
        pf2[i] = sc.pf[[REQ, WRITE, LAT]]
        pf1[i] = score_features(cur, None, None, x, DT, (S, "null", float(i))).pf[LAT]
    for j in range(3):
        z = zs[:, j][np.isfinite(zs[:, j])]
        assert z.size > 1900
        assert abs(z.mean()) < 0.1 and abs(z.std() - 1.0) < 0.1
        assert stats.kstest(stats.norm.cdf(z), "uniform").pvalue > 0.01
    assert stats.kstest(pf1, "uniform").pvalue > 0.01
    for j in range(3):
        p = pf2[:, j][np.isfinite(pf2[:, j])]
        for a in (0.01, 0.05, 0.1, 0.3):
            assert np.mean(p <= a) <= a * 1.2 + 0.005


def test_d_pit_is_seeded_and_anchor_coherent():
    cur = make_pred(nb={REQ: (22.0, 1e3)})
    x = nat_of(http_requests=20.0)
    a = score1(cur, x, keys=(S, "e", 5.0))
    b = score1(cur, x, keys=(S, "e", 5.0))
    c = score1(cur, x, keys=(S, "e", 6.0))
    assert a.z[REQ] == b.z[REQ]                     # replay is bit-identical
    assert a.z[REQ] != c.z[REQ]                     # a new tick draws a new V
    assert a.z[REQ] == a.zr[REQ]                    # same V for both anchors


# ================================================================ (e)
def test_e_dual_anchor_reference_drives_pf():
    loc, sd = math.log(100.0), 0.2
    ref = make_pred(t={LAT: (1e4, loc, sd)})
    cur = make_pred(t={LAT: (1e4, loc + sd, sd)})    # current crept +1 sigma
    x = nat_of(http_latency=math.exp(loc + 3.0 * sd), http_requests=22.0)   # n >= 5 requests
    sc = score_features(cur, ref, None, x, DT, (S, "e", T0))
    assert 0.04 < sc.p_cur[LAT] < 0.06
    assert sc.p_ref[LAT] < 0.005
    assert sc.pf[LAT] < 0.01
    assert sc.pf[LAT] == pytest.approx(2.0 * sc.p_ref[LAT])
    assert sc.z[LAT] == pytest.approx(2.0, abs=1e-3)
    assert sc.zr[LAT] == pytest.approx(3.0, abs=1e-3)


# ================================================================ (f)
def test_f_ratio_n0_is_nan():
    cur = make_pred(bb={WRITE: (0.1, 50.0)}, nb={REQ: (22.0, 1e3)})
    for w in (np.nan, 0.0):                          # B01 writes NaN; 0/0 must not score either
        sc = score1(cur, nat_of(http_requests=0.0, http_write_ratio=w), cls=cur)
        for arr in (sc.z, sc.zr, sc.pf, sc.p_cur, sc.p_ref, sc.p_cls):
            assert math.isnan(arr[WRITE])
        assert np.isfinite(sc.pf[REQ])               # the count itself (0) is data


def test_f_engine_ratio_n0_is_nan(trained):
    st = trained.store
    t = trained.tick()
    x = base_row(trained.rng)
    x[REQ] = 0.0
    x[WRITE] = np.nan
    x[LAT] = np.nan
    feed(st, t, x, e=ENTS[1])
    run_engine(LikelihoodEngine(), st, t)
    for name in (Z, ZR, PF):
        r = z_row(st, t, e=ENTS[1], name=name)
        assert math.isnan(r[WRITE]) and math.isnan(r[LAT])
        assert np.isfinite(r[REQ])


# ================================================================ channels
def test_channels_dependence_group_weights():
    vol = F.GROUPS["volume"]
    p = np.ones(NF)
    p[vol[0]] = 1e-6
    single = channels(p, GroupLayout([[i] for i in range(NF)]))
    # the flagged feature alone vs. the other 8 volume features in one group
    rest = [i for i in range(NF) if i not in vol]
    lay = GroupLayout([[vol[0]], list(vol[1:])] + [[i] for i in rest])
    grouped = channels(p, lay)
    assert single.p_int == pytest.approx(len(vol) / (1e6 + len(vol) - 1), rel=1e-9)
    assert grouped.p_int == pytest.approx(2.0 / (1e6 + 1.0), rel=1e-9)
    assert grouped.p_shape == 1.0 and grouped.p_fg["volume"] < 0.01
    # NaN features are dropped; an all-NaN channel is NaN, never 1
    q = np.full(NF, np.nan)
    q[vol] = 0.5
    ch = channels(q, GroupLayout([[i] for i in range(NF)]))
    assert ch.p_int == pytest.approx(0.5) and math.isnan(ch.p_shape)


def test_group_layout_validation():
    with pytest.raises(ValueError):
        GroupLayout([[REQ, WRITE]] + [[i] for i in range(NF) if i not in (REQ, WRITE)])
    with pytest.raises(ValueError):
        GroupLayout([[REQ, FLOWS], [FLOWS]])                      # FLOWS twice
    with pytest.raises(ValueError):
        GroupLayout([[NF]])
    lay = GroupLayout([[REQ, FLOWS]])                             # the rest: singletons
    assert len(lay.groups) == NF - 1 and sorted([REQ, FLOWS]) in lay.groups


def test_engine_uses_model_groups_split_by_feature_group(trained):
    """model.groups joining a volume and an app feature is split, so an app
    shift moves marg_shape (axis app) and leaves marg_int quiet."""
    st = trained.store
    groups = [[REQ, WRITE, FLOWS]] + [[i] for i in range(NF) if i not in (REQ, WRITE, FLOWS)]
    st.put_model(S, "__system__", m_density.GROUPS_MODEL, {"fmt": 1, "version": 1,
                                                           "groups": groups})
    try:
        t = trained.tick()
        x = base_row(trained.rng)
        x[WRITE] = 20.0 / 22.0                      # 20 of 22 requests are writes (usual 10 %)
        feed(st, t, x, e=ENTS[2])
        run_engine(LikelihoodEngine(), st, t)
        sc = emit.read_row(st, S, ENTS[2], emit.SCORE, t)
        ax = emit.read_dict(st, S, ENTS[2], emit.AXES, t)
        assert sc["marg_shape"] > 4.0
        assert sc["marg_int"] < 2.0
        assert "app" in ax.get("marg_shape", []) and "marg_int" not in ax
    finally:
        st.put_model(S, "__system__", m_density.GROUPS_MODEL, None)


# ================================================================ engine edges
def test_empty_store():
    st = make_store()
    assert run_engine(LikelihoodEngine(), st, T0) == 0
    assert run_engine(LikelihoodEngine(), st, T0, training=True) == 0


def test_silent_entity_writes_nothing(trained):
    st = trained.store
    t = trained.tick()
    feed(st, t, base_row(trained.rng), e=ENTS[1], active=0.0)
    run_engine(LikelihoodEngine(), st, t)
    assert z_row(st, t, e=ENTS[1]) is None
    assert emit.read_row(st, S, ENTS[1], emit.SCORE, t) == {}


def test_new_entity_scored_from_first_tick(trained):
    """No own model: the backoff (system tier) predictive scores it; peer is
    the system tier too, so a normal row is unremarkable everywhere."""
    st = trained.store
    t = trained.tick()
    e = "10.0.0.99"
    feed(st, t, base_row(trained.rng), e=e)
    run_engine(LikelihoodEngine(), st, t)
    z = z_row(st, t, e=e)
    assert np.isfinite(z[REQ]) and abs(z[REQ]) < 3.0
    sc = emit.read_row(st, S, e, emit.SCORE, t)
    assert set(DETS) <= set(sc)
    assert sc["marg_int"] < 2.0 and sc["peer"] < 2.0


def test_peer_nan_without_tier_model():
    """Bare hyperprior (no class / system tier yet): marg scored, peer NaN."""
    st = make_store()
    rng = np.random.default_rng(3)
    feed(st, T0, base_row(rng))
    run_engine(LikelihoodEngine(), st, T0)
    sc = emit.read_row(st, S, ENTS[0], emit.SCORE, T0)
    assert "marg_int" in sc and "marg_shape" in sc and "peer" not in sc
    assert math.isnan(float(emit.read_array(st, S, ENTS[0], emit.PM, T0)[2]))


def test_nan_inputs_are_unscored_not_p1():
    st = make_store()
    feed(st, T0, np.full(NF, np.nan))
    run_engine(LikelihoodEngine(), st, T0)
    for name in (Z, ZR, PF):
        assert np.isnan(z_row(st, T0, name=name)).all()
    assert emit.read_row(st, S, ENTS[0], emit.SCORE, T0) == {}
    assert emit.read_row(st, S, ENTS[0], emit.PM, T0) == {}


def test_bad_nat_dimension_raises():
    st = make_store()
    feed(st, T0, np.zeros(10))
    with pytest.raises(ValueError):
        run_engine(LikelihoodEngine(), st, T0)


def test_degraded_when_nat_missing_or_b03_failed():
    st = make_store()
    st.add_vec(S, ENTS[0], "feature.active", T0, np.array([1.0], np.float32), window_s=900)
    st.register_entity(S, ENTS[0])
    run_engine(LikelihoodEngine(), st, T0)
    deg = emit.read_dict(st, S, ENTS[0], emit.DEGRADED, T0)
    assert deg == {d: "stale:feature.nat" for d in DETS}
    assert np.isnan(z_row(st, T0)).all()
    assert emit.read_row(st, S, ENTS[0], emit.SCORE, T0) == {}

    t = T0 + DT
    feed(st, t, base_row(np.random.default_rng(1)))
    st.put_health("behavior.baseline", {"engine": "behavior.baseline", "ok": False,
                                        "last_error_ts": t})
    run_engine(LikelihoodEngine(), st, t)
    deg = emit.read_dict(st, S, ENTS[0], emit.DEGRADED, t)
    assert deg == {d: "producer_error:behavior.baseline" for d in DETS}
    assert np.isnan(z_row(st, t, name=PF)).all()


def test_degraded_when_b01_failed():
    st = make_store()
    st.register_entity(S, ENTS[0])
    st.put_health("behavior.feature_vector", {"ok": False, "last_error_ts": T0})
    run_engine(LikelihoodEngine(), st, T0)
    assert emit.read_dict(st, S, ENTS[0], emit.DEGRADED, T0) == {
        d: "producer_error:behavior.feature_vector" for d in DETS}
    assert z_row(st, T0) is None


def test_training_mode_same_scores_no_events(trained):
    st = trained.store
    t = trained.tick()
    x = base_row(trained.rng)
    x[REQ] = 90.0                                    # a burst: would alarm downstream
    feed(st, t, x)
    n_ev = len(st.events(limit=10_000))
    run_engine(LikelihoodEngine(), st, t, training=True)
    z_train = z_row(st, t).copy()
    sc_train = emit.read_row(st, S, ENTS[0], emit.SCORE, t)
    assert sc_train["marg_int"] > 5.0
    assert len(st.events(limit=10_000)) == n_ev

    t2 = trained.tick()
    feed(st, t2, x)
    run_engine(LikelihoodEngine(), st, t2, training=False)
    assert len(st.events(limit=10_000)) == n_ev      # B04 never emits events
    assert np.isfinite(z_train[REQ]) and z_train[REQ] > 4.0


def test_cadence_900_to_60(trained):
    """Exposure-aware: the usual rate at 60-s ticks is unremarkable, the
    same per-tick count as a 15-min tick (15x the rate) is extreme."""
    st = trained.store
    eng = LikelihoodEngine()
    t = trained.tick(dt=60.0)
    x = base_row(trained.rng, dt=60.0)               # 22 per 15 min -> ~1.5 per minute
    x[REQ] = 1.0
    x[WRITE] = 0.0
    feed(st, t, x, dt=60.0)
    run_engine(eng, st, t, dt=60.0)
    pf = z_row(st, t, name=PF)
    assert pf[REQ] > 0.05 and abs(z_row(st, t)[REQ]) < 3.0
    sc = emit.read_row(st, S, ENTS[0], emit.SCORE, t)
    assert sc["marg_int"] < 2.0

    t2 = trained.tick(dt=60.0)
    x2 = base_row(trained.rng, dt=60.0)
    x2[REQ] = 22.0
    x2[WRITE] = 2.0 / 22.0
    feed(st, t2, x2, dt=60.0)
    run_engine(eng, st, t2, dt=60.0)
    assert z_row(st, t2, name=PF)[REQ] < 1e-6
    assert emit.read_row(st, S, ENTS[0], emit.SCORE, t2)["marg_int"] > 5.0


def test_tctx_fallback_from_config(trained):
    """Without a feature.tctx row the config clock gives the same bucket."""
    st = trained.store
    t = trained.tick()
    x = base_row(trained.rng)
    feed(st, t, x, e=ENTS[1], tctx=False)
    run_engine(LikelihoodEngine(), st, t)
    z_a = z_row(st, t, e=ENTS[1])
    tc = TB.tctx_from_config(t, CFG, DT)
    preds = MB.predictive_set(st, S, ENTS[1], tc)
    stored = st.vec_at(S, ENTS[1], "feature.nat", t).astype(np.float64)
    ref = score_features(preds["current"], preds["reference"], preds["class"], stored, DT,
                         (S, ENTS[1], t))
    np.testing.assert_allclose(z_a, ref.z.astype(np.float32), equal_nan=True)


def test_model_state_profile(trained):
    st = trained.store
    eng = LikelihoodEngine()
    t = trained.tick()
    feed(st, t, base_row(trained.rng), e=ENTS[2])
    run_engine(eng, st, t)
    ms = st.profile(S, ENTS[2]).extra["model_state"]
    assert ms["dt_s"] == 900.0 and ms["q"] == [0.05, 0.5, 0.95] and ms["ts"] == t
    lo, mid, hi = ms["features"]["http_requests"]
    assert lo <= mid <= hi and 18.0 <= mid <= 26.0
    lo, mid, hi = ms["features"]["http_latency"]
    assert lo < 100.0 < hi
    lo, mid, hi = ms["features"]["http_write_ratio"]
    assert 0.0 <= lo <= mid <= hi <= 1.0
    # refreshed about hourly (per-entity phase), not every tick
    stamps = {t}
    for _ in range(8):                               # 2 h at 900 s
        t2 = trained.tick()
        feed(st, t2, base_row(trained.rng), e=ENTS[2])
        run_engine(eng, st, t2)
        stamps.add(st.profile(S, ENTS[2]).extra["model_state"]["ts"])
    assert 2 <= len(stamps) <= 3


def test_state_quantiles_match_accessor_for_nb_bb(trained):
    """Count and ratio columns equal m_baseline.quantiles; t columns are the
    inverse transform of the Student-t quantiles (natural units)."""
    tc = TB.tctx_from_config(trained.t_next, CFG, DT)
    pred = MB.predictive_set(trained.store, S, ENTS[0], tc)["current"]
    qs = (0.05, 0.5, 0.95)
    mine = state_quantiles(pred, qs, 900.0)
    ref = MB.quantiles(pred, qs, dt_s=900.0)
    np.testing.assert_array_equal(mine[:, MB.CNT], ref[:, MB.CNT])
    np.testing.assert_allclose(mine[:, MB.RAT], ref[:, MB.RAT], rtol=1e-12, equal_nan=True)
    y = pred.loc[LAT] + pred.scale[LAT] * stats.t.ppf(qs, pred.df[LAT])
    np.testing.assert_allclose(mine[:, LAT], np.exp(y), rtol=1e-9)
    y = pred.loc[BYTES] + pred.scale[BYTES] * stats.t.ppf(qs, pred.df[BYTES])
    np.testing.assert_allclose(mine[:, BYTES], np.expm1(y) * 15.0, rtol=1e-9)


def test_perf_per_entity(trained):
    st = trained.store
    eng = LikelihoodEngine()
    t = trained.tick()
    for e in ENTS:
        feed(st, t, base_row(trained.rng), e=e)
    run_engine(eng, st, t)                            # warm (model_state refresh)
    t = trained.tick()
    for e in ENTS:
        feed(st, t, base_row(trained.rng), e=e)
    t0 = time.perf_counter()
    run_engine(eng, st, t)
    per = (time.perf_counter() - t0) / len(ENTS)
    assert per < 0.01, f"{per * 1e3:.2f} ms per entity"


# --------------------------------------- round 4: provisional Q -> degraded
@pytest.mark.parametrize("pi,flagged", [(0.2, True), (0.9, False)])
def test_q_score_on_provisional_transfer_writes_degraded(pi, flagged):
    """Contract M: a Q score whose features rest mostly on the H -> Q transfer
    (native share pi < 0.5) is written with behavior.degraded
    'provisional:q_transfer'; a native-dominated one is not."""
    from app.engines.behavior import likelihood as LK
    store = make_store()
    e = "10.9.9.1"
    store.register_entity(S, e)
    nf = F.FEATURE_DIM
    pf = np.random.default_rng(2).uniform(0.05, 1.0, nf)
    sc = LK.FeatureScores(np.zeros(nf), np.zeros(nf), pf, pf, pf, np.full(nf, np.nan))
    nan = np.full(nf, np.nan)
    cur = MB.Pred(nan, nan, nan, nan, nan, nan, nan, nan, prov=np.full(nf, pi))
    eng = LK.LikelihoodEngine()
    t = T0 + 900.0
    eng._write_q(store, S, e, t, sc, cur, LK.GroupLayout([]), 900.0, state_q=None)
    dg = emit.read_dict(store, S, e, emit.DEGRADED, t)
    if flagged:
        assert dg == {d: LK.PROV_CAUSE for d in LK.DETS_Q}
    else:
        assert not dg
    pm = emit.read_row(store, S, e, emit.PM, t)
    assert math.isfinite(pm["marg_shape_q"]) and math.isfinite(pm["marg_int_q"])
