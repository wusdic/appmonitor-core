"""Regression tests for the shared-code changes made at integration
(docs/lib3/integration.md): lib accessors, store, and the cross-engine
contract fixes (warm-up floor / carry-over, link retraction, label queue)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from helpers import DT, T0, make_store, make_tctx, put_model, run_engine, set_trust

from app.core.engine import default_config
from app.engines.behavior import governor as GOV
from app.engines.behavior.changepoint import ChangepointEngine
from app.engines.behavior.feedback import FeedbackEngine
from app.engines.behavior.likelihood import state_quantiles
from app.engines.behavior.risk import RiskEngine
from app.engines.behavior.lib import bayes, detectors, m_class
from app.engines.behavior.lib import m_baseline as MB
from app.engines.behavior.lib import m_cp
from app.engines.behavior.lib import m_governor as MG
from app.engines.behavior.lib import timebins as TB
from app.engines.behavior.lib.features import FEATURE_INDEX, FEATURE_KIND, KEY_FEATURES
from app.models.schema import DerivedMetric, MetricKind

S = "erp"
CFG = default_config({"strict": True})


# ------------------------------------------------------------ m_baseline
def test_quantiles_and_mean_nat_invert_t_family_on_the_right_columns():
    """B04's report: the t-family inverse transform landed on NIG-block
    positions of the full row (http_latency came out as expit(...) ~ 1)."""
    st = make_store()
    tc = TB.tctx_from_config(T0, CFG, DT)
    pred = MB.predictive(st, S, "10.0.0.1", tc)
    lat = FEATURE_INDEX["http_latency"]
    q = MB.quantiles(pred, (0.05, 0.5, 0.95), 900.0)
    assert q[1, lat] == pytest.approx(math.exp(pred.loc[lat]), rel=1e-9)
    assert q[1, lat] > 1.5                                   # ms, not a share
    assert MB.mean_nat(pred)[lat] == pytest.approx(math.exp(pred.loc[lat]), rel=1e-9)
    for name, kind in FEATURE_KIND.items():                  # bounded stay in [0, 1]
        if kind == "bounded":
            i = FEATURE_INDEX[name]
            assert np.all((q[:, i] >= 0.0) & (q[:, i] <= 1.0)), name
    np.testing.assert_array_equal(state_quantiles(pred, (0.05, 0.5, 0.95), 900.0), q)


def test_own_support_and_n_eff_by_bucket_of_an_empty_model():
    m = MB.new_model()
    W, expo = MB.own_support(m, TB.tctx_from_config(T0, CFG, DT))
    assert W.shape == (52,) and not W.any() and expo == 0.0
    assert MB.n_eff_by_bucket(m).shape == (48,) and not MB.n_eff_by_bucket(m).any()


# --------------------------------------------------------------- lib misc
def test_bb_parts_is_the_midp_kernel():
    lt, eq, gt = bayes.bb_parts(1, 2, 5.0, 45.0)
    assert lt + eq + gt == pytest.approx(1.0)
    u, _ = bayes.bb_midp(1, 2, 5.0, 45.0)
    assert u == pytest.approx(lt + 0.5 * eq)
    assert all(math.isnan(x) for x in bayes.bb_parts(3, 2, 1.0, 1.0))


def test_acc_level_scale():
    d = "cusum"
    arl = detectors.arl_days(d) * 86400.0 / 900.0
    assert detectors.acc_level(1.0 / arl, d, 900.0) == pytest.approx(1.0)
    assert detectors.acc_level(1.0, d, 900.0) == 0.0
    assert math.isnan(detectors.acc_level(math.nan, d, 900.0))


def test_m_class_index_follows_copy_on_write_versions():
    st = make_store()
    a = {f"{S}|10.0.0.{i}": {"role": "r1", "prob": 1.0, "static": ["dmz"]} for i in (1, 2, 3)}
    put_model(st, "__org__", "__org__", "model.class", {"version": 1, "assign": a})
    assert m_class.class_key(st, S, "10.0.0.1") == "class:r1"
    assert m_class.members(st, S, "r1") == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
    assert m_class.class_members(st, S, "class:static:dmz") == ["10.0.0.1", "10.0.0.2",
                                                                 "10.0.0.3"]
    b = {k: dict(v) for k, v in a.items()}
    b[f"{S}|10.0.0.3"]["role"] = "r2"
    put_model(st, "__org__", "__org__", "model.class", {"version": 2, "assign": b})
    assert m_class.class_key(st, S, "10.0.0.1") is None       # r1 now has 2 members
    assert m_class.all_class_keys(st, S) == ["class:r1", "class:static:dmz"]
    assert m_class.members(st, S, "r1", min_prob=0.0) == ["10.0.0.1", "10.0.0.2"]


# ------------------------------------------------------------------ store
def test_timeline_lists_risk_from_the_vec_ring():
    st = make_store()
    for i, r in enumerate([1.0, 2.0, 15.0, 16.0, 55.0]):
        st.add_vec(S, "10.0.0.1", "behavior.risk", T0 + i * DT, [r], window_s=int(DT))
    risk = [it for it in st.timeline(S, "10.0.0.1") if it["type"] == "risk"]
    assert sorted(it["item"].value for it in risk) == [1.0, 15.0, 55.0]    # band changes


def test_default_retention_covers_engine_rules():
    st = make_store()
    H, D = 3600.0, 86400.0
    want = {"http.requests": 24 * H, "act.events": 24 * H, "l4.bytes_up": 13 * H,
            "probe.rtt_ms": 13 * H, "behavior.e_day": 8 * D, "behavior.class.agg": 9 * D,
            "behavior.budget": 1 * D, "behavior.calib_health": 8 * D}
    for name, age in want.items():
        space = "vec" if name.startswith("behavior.") else "raw"
        assert st._rule(space, name, 1.0)[1] == age, name
    assert st._rule("derived", "behavior.budget")[0] == 24
    assert st._rule("raw", "http.status_4xx", 1.0)[1] == 6 * H      # other raw scalars


# -------------------------------------------------------------- governor
def test_rollback_never_reaches_into_warm_up_rows():
    st = make_store()
    eng = GOV.GovernorEngine()
    now = T0 + 100 * DT
    model = GOV.new_model(now)
    model["train_end"] = T0 + 80 * DT
    model["episode"] = {"onset": T0 + 10 * DT}               # an onset inside the warm-up
    frontier = now - 4 * DT
    eng._want_rollback(st, S, "10.0.0.1", model, now, DT, frontier)
    ctl = st.get_model(S, "10.0.0.1", MG.CONTROL)
    assert ctl["rollback_to"] == pytest.approx(T0 + 80 * DT)
    # an onset after the warm-up is unaffected
    model2 = GOV.new_model(now)
    model2["train_end"] = T0 + 80 * DT
    model2["episode"] = {"onset": T0 + 90 * DT}
    eng._want_rollback(st, S, "10.0.0.2", model2, now, DT, frontier)
    assert st.get_model(S, "10.0.0.2", MG.CONTROL)["rollback_to"] == pytest.approx(
        T0 + 89 * DT)


def test_retracted_link_rolls_the_seeded_entity_back_and_releases_it():
    st = make_store()
    for e in ("10.0.0.1", "10.0.0.2"):
        st.register_entity(S, e)
    t_link = T0 + 5 * DT
    now = T0 + 20 * DT
    link = {"fmt": 1, "version": 2, "links": [
        {"id": "A>B", "from": "10.0.0.1", "to": "10.0.0.2", "ts": t_link, "status": "retracted",
         "retracted": True, "retracted_ts": now - DT, "rollback_to": t_link}], "actors": []}
    put_model(st, S, "__system__", "model.link", link, version=2)
    eng = GOV.GovernorEngine()
    run_engine(eng, st, now)
    ctl = st.get_model(S, "10.0.0.2", MG.CONTROL)
    assert ctl["rollback_to"] == pytest.approx(t_link)
    assert ctl["release"] == [pytest.approx(t_link), pytest.approx(now)]
    assert st.get_model(S, "10.0.0.1", MG.CONTROL) is None      # A is untouched
    run_engine(eng, st, now + 2 * 3600.0)                       # done once per link
    assert st.get_model(S, "10.0.0.2", MG.CONTROL)["ts"] == now


# -------------------------------------------------------------- feedback
def test_governor_label_queue_keys_are_queued_by_b23():
    st = make_store()
    st.register_entity(S, "10.0.0.1")
    m = GOV.new_model(T0)
    m.update(regime=MG.DRIFTING, type="ramp", label_queue={"since": T0 - 15 * 86400.0,
                                                            "ts": T0})
    put_model(st, S, "10.0.0.1", MG.MODEL, m)
    run_engine(FeedbackEngine(), st, T0)
    q = st.get_model("__org__", "__org__", "model.feedback")["queue"]
    items = [it for it in q if it.get("source") == "governor"]
    assert len(items) == 1 and items[0]["entity"] == "10.0.0.1" and items[0]["reason"] == "held"
    run_engine(FeedbackEngine(), st, T0 + DT)                   # kept while still queued
    m["regime"] = MG.NORMAL
    eng = FeedbackEngine()
    run_engine(eng, st, T0 + 2 * DT)
    q = st.get_model("__org__", "__org__", "model.feedback")["queue"]
    assert not [it for it in q if it.get("source") == "governor"]


# ------------------------------------------------------------ changepoint
def _zr_ticks(st, eng, n, t0, shift, training=False):
    key = [FEATURE_INDEX[k] for k in KEY_FEATURES]
    for i in range(n):
        t = t0 + i * DT
        zr = np.zeros(52, dtype=np.float32)
        zr[key] = shift
        st.add_vec(S, "10.0.0.1", "behavior.zr", t, zr, window_s=int(DT))
        st.add_vec(S, "10.0.0.1", "feature.active", t, [1.0], window_s=int(DT))
        st.register_entity(S, "10.0.0.1")
        set_trust(st, S, "10.0.0.1", [t], 1.0, quarantine=0.0)
        run_engine(eng, st, t, training=training)
    return t0 + n * DT


def test_changepoint_ignores_residuals_the_own_baseline_does_not_support():
    st = make_store()
    put_model(st, S, "10.0.0.1", MB.MODEL, {"fmt": MB.FMT, "tier": "entity",
                                            **MB.new_model()})
    eng = ChangepointEngine()
    _zr_ticks(st, eng, 12, T0, 3.0)
    assert m_cp.level(st, S, "10.0.0.1")["cusum"] == 0.0     # empty own bucket: no input


def test_changepoint_charts_restart_after_warm_up():
    st = make_store()
    eng = ChangepointEngine()
    t = _zr_ticks(st, eng, 12, T0, 3.0, training=True)
    assert m_cp.level(st, S, "10.0.0.1")["cusum"] > 1.0
    _zr_ticks(st, eng, 1, t, 0.0)
    assert m_cp.level(st, S, "10.0.0.1")["cusum"] == 0.0


# ------------------------------------------------------------------ risk
def test_risk_does_not_carry_warm_up_evidence_into_live():
    st = make_store()
    st.register_entity(S, "10.0.0.1")
    eng = RiskEngine()
    for i in range(6):
        t = T0 + i * DT
        st.add_derived(DerivedMetric(name="behavior.p_family", value={"intensity": 1e-9},
                                     ts=t, system=S, entity="10.0.0.1", window_s=int(DT),
                                     kind=MetricKind.CATEGORICAL))
        run_engine(eng, st, t, training=True)
    assert st.vec_at(S, "10.0.0.1", "behavior.risk", T0 + 5 * DT)[0] > 10.0
    run_engine(eng, st, T0 + 6 * DT)                         # first live tick, no evidence
    assert st.vec_at(S, "10.0.0.1", "behavior.risk", T0 + 6 * DT)[0] == 0.0


# ------------------------------------------------------------ calibration
def test_b24_resets_rings_on_a_bare_version_change_from_the_default():
    """B25's report: B24 read applied['version'], which the gate never records
    while model.control's version equals the default 0, so 0 -> 2 was missed."""
    from app.engines.behavior.calibration import CalibrationEngine
    from app.engines.behavior.lib import emit

    st = make_store()
    e = "10.0.0.1"
    st.register_entity(S, e)
    put_model(st, S, e, "model.baseline", {"n_eff": 100.0})
    put_model(st, S, e, "model.control", {"version": 0})
    eng = CalibrationEngine()
    rng = np.random.default_rng(3)
    t = T0
    for _ in range(40):
        emit.write_scores(st, S, e, t, {"marg_int": float(rng.exponential())})
        tc = make_tctx(t, dt=DT)
        tc.pop("daypart_id", None)
        st.add_derived(DerivedMetric(name="feature.tctx", value=tc, ts=t, system=S, entity=e,
                                     window_s=int(DT), kind=MetricKind.CATEGORICAL))
        run_engine(eng, st, t)
        set_trust(st, S, e, [t], 1.0, quarantine=0.0)
        t += DT
    m = st.get_model(S, e, "model.calib")
    assert m["rings"] and m.get("resets", 0) == 0
    put_model(st, S, e, "model.control", {"version": 2})
    emit.write_scores(st, S, e, t, {"marg_int": 1.0})
    run_engine(eng, st, t)
    m = st.get_model(S, e, "model.calib")
    assert m["resets"] == 1 and m["version"] == 2


# ------------------------------------------------- store: raise-only rules
def test_ensure_retention_only_raises():
    """R2.1: D0 and D2 both ask for act.events; neither may lower a longer
    rule another engine (or the store default) set."""
    st = make_store()
    H = 3600.0
    assert st._rule("raw", "act.events", 1.0)[1] == 24 * H          # store default
    assert st.ensure_retention("act.events", max_age_s=6 * H) is False
    assert st._rule("raw", "act.events", 1.0)[1] == 24 * H          # not lowered
    st.ensure_retention("act.events", max_age_s=48 * H)
    assert st._rule("raw", "act.events", 1.0)[1] == 48 * H          # raised
    assert st.ensure_retention("act.events", max_age_s=24 * H) is False
    assert st._rule("raw", "act.events", 1.0)[1] == 48 * H
    # a name without any rule gets the requested one
    st.ensure_retention("x.custom", max_age_s=2 * H)
    assert st._rule("raw", "x.custom", 1.0)[1] == 2 * H
    # the explicit points cap of an engine-owned series is kept
    st.ensure_retention("behavior.budget", None, 1 * 86400.0)
    assert st._rule("derived", "behavior.budget")[0] == 24


# --------------------------------------------------- store: named snapshot
def test_snapshot_restricted_to_names_equals_the_full_snapshot():
    """lib-4 asks only for the metrics its signatures reference; the values
    must equal the full snapshot's (derived wins over raw, NaN skipped)."""
    from helpers import add_obs_tick, add_derived_series
    st = make_store()
    e = "10.0.0.1"
    add_obs_tick(st, S, e, T0, {"http.requests": 10, "l4.flows": 3})
    add_derived_series(st, S, e, "derived.path_entropy", [0.4])
    add_derived_series(st, S, e, "derived.nan_one", [math.nan])
    st.add_vec(S, e, "behavior.risk", T0, [5.0], window_s=int(DT))
    full = st.snapshot(S, e)
    names = ["http.requests", "derived.path_entropy", "derived.nan_one", "nope"]
    part = st.snapshot(S, e, names=names)
    assert part == {k: v for k, v in full.items() if k in names}
    assert st.snapshot(S, e, now=T0 + DT, names=names) == {}


# ----------------------------------------------------- R2: vocab maturity
def test_new_template_ratio_needs_a_mature_system_vocabulary():
    from app.engines.behavior.lib import m_template as MT
    assert not MT.vocab_mature(None, T0)
    assert not MT.vocab_mature({"born": T0}, T0 + MT.NEW_REF_S - 1.0)
    assert MT.vocab_mature({"born": T0}, T0 + MT.NEW_REF_S)


# ------------------------------------------- m_identity: client sketch block
def test_window_vector_zeroes_the_client_stack_sketch_block():
    """R20.2: client stacks are their own capped modality in B16, so the
    client.stack_set namespace of the mean sketch is not part of z."""
    from app.engines.behavior.lib import m_identity as MI
    from app.engines.behavior.lib import sketch as SK
    rng = np.random.default_rng(1)
    rows = rng.normal(size=(4, MI.TICK_DIM))
    w = MI.window_vector(rows)
    assert np.all(w[MI.W_SK_CLIENT] == 0.0)
    i = SK.SKETCH_NAMESPACES.index("client.stack_set")
    blk = slice(60 + 16 * i, 60 + 16 * (i + 1))
    assert MI.W_SK_CLIENT == blk
    other = [c for c in range(60, 140) if not blk.start <= c < blk.stop]
    assert np.all(w[other] != 0.0)
