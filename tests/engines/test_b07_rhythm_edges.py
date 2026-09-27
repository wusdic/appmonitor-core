"""B07 RhythmEngine: edge cases, governance, cadence, calendar, accessors, perf.

Uses the Rig driver of test_b07_rhythm.py (feature.active + act.stream per
tick, as B01 / R2 write them)."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from helpers import make_store, make_tctx, put_model, run_engine, set_trust

from app.engines.behavior.lib import emit
from app.engines.behavior.lib import m_rhythm as R
from app.engines.behavior.lib.classkeys import ORG, SYSTEM_KEY
from app.engines.behavior.rhythm import DETS, SERIES, RhythmEngine
from app.models.schema import RawMetric
from test_b07_rhythm import (DAY, H_OFF, S, T0, TZ, Rig, backup_at, in_window, local, train,
                             worker)

H = 3600.0


def _night(t: float) -> bool:
    return worker(t) or in_window(local(t)[1], 2.0, 2.75)


# ================================================================ metadata
def test_engine_metadata_and_no_arg_constructor():
    eng = RhythmEngine()
    assert eng.name == "behavior.rhythm" and eng.layer == "behavior" and eng.interval == 1
    for r in ("feature.active", "feature.tctx", "act.stream", "behavior.trust", "model.class",
              "model.control"):
        assert r in eng.consumes
    for w in ("model.rhythm", "behavior.score", "behavior.acc_alarm", "behavior.axes", SERIES,
              "profile.extra.rhythm", "event.schedule_shift"):
        assert w in eng.produces
    assert eng.h_off == pytest.approx(R.H_OFF) and eng.h_sil == pytest.approx(R.H_SIL)
    assert R.H_OFF == pytest.approx(math.log2(100 * 96) + 0.5)


# ================================================================== empty
def test_empty_store_and_entity_without_feature_active():
    st = make_store()
    eng = RhythmEngine()
    assert run_engine(eng, st, T0 + 900.0, dt=900.0, config={"tz": TZ}) == 0
    # an entity known only through act.stream (B01 never wrote it): nothing to do
    st.add_raw(RawMetric(name="act.events", value=3.0, ts=T0 + 1800.0, system=S, entity="x"))
    assert run_engine(eng, st, T0 + 1800.0, dt=900.0, config={"tz": TZ}) == 0
    assert st.get_model(S, "x", R.MODEL) is None
    assert emit.read_row(st, S, "x", emit.SCORE, T0 + 1800.0) == {}


# ============================================================ silent entity
def test_silent_entity_absence_is_data():
    rig = Rig({"q": lambda t: False})
    rows = rig.run_until(T0 + 2 * DAY, dt=H, training=True, record=True, entity="q")
    m = rig.model("q")
    st = m["state"]
    assert st["A48"].sum() == 0.0 and st["N48"].sum() > 0.0        # silent slots were learned
    assert st["n_slots"] > 0
    assert R.p_cell(m, R.cell48(3, 0)) < 0.2
    assert all(r["W_off"] == 0.0 for r in rows)
    assert all(r["score"]["offhours"] == 0.0 and r["pm"]["offhours"] == 1.0 for r in rows)
    assert all("silence" not in r["score"] for r in rows)           # never active: no rhythm
    assert not m["machine_like"] and math.isnan(R.entropy168(m))
    assert all(r["acc"] == {"offhours": 0, "silence": 0} for r in rows)
    assert not rig.store.events(S, "q")
    # accessors on an all-silent model: shape undefined, loglik finite
    assert np.isnan(R.shape48(m)).all()
    assert math.isfinite(R.loglik(m, [0.0], [(R.cell48(3, 0), -1)]))


# ================================================================ NaN input
def test_nan_feature_active_is_unobserved_not_inactive():
    rig = Rig({"w": worker})
    train(rig, 3)
    m0 = copy.deepcopy(rig.model("w")["state"])
    # B01 wrote NaN for a day (and no stream): no evidence either way
    t_end = rig.now + DAY
    while rig.now < t_end:
        rig.now += 900.0
        rig.store.add_vec(S, "w", "feature.active", rig.now,
                          np.array([np.nan], dtype=np.float32), window_s=900)
        run_engine(rig.eng, rig.store, rig.now, training=True, dt=900.0, config=rig.cfg)
        r = rig.snap("w")
        assert r["W_off"] == 0.0 and math.isfinite(r["score"]["offhours"])
    m = rig.model("w")
    led = R.slot_history(m, since=t_end - DAY + 3 * H)
    assert led and all(math.isnan(a) for _, a, _, _, _ in led)
    # nothing learned from the NaN day: counts only decayed, no new observations
    assert m["state"]["n_slots"] <= m0["n_slots"] + 16         # the D-lagged tail (4 h) of day 3
    assert m["state"]["A48"].sum() <= m0["A48"].sum() + 1e-9


def test_missing_active_row_and_b01_failure_write_nan_plus_degraded():
    rig = Rig({"w": worker, "v": worker})
    train(rig, 1)
    # tick where B01 wrote nothing for w (stale)
    rig.now += 900.0
    rig.write_tick(rig.now, 900.0, {"v": worker})
    run_engine(rig.eng, rig.store, rig.now, dt=900.0, config=rig.cfg)
    r = rig.snap("w")
    assert "offhours" not in r["score"] and "silence" not in r["score"]
    deg = emit.read_dict(rig.store, S, "w", emit.DEGRADED, rig.now)
    assert deg == {d: "stale:feature.active" for d in DETS}
    assert "offhours" in rig.snap("v")["score"]
    # B01 raised at this tick: everyone is degraded even with a (stale) row
    rig.now += 900.0
    rig.write_tick(rig.now, 900.0, rig.patterns)
    rig.store.put_health("behavior.feature_vector", {"last_error_ts": rig.now, "error_count": 1})
    run_engine(rig.eng, rig.store, rig.now, dt=900.0, config=rig.cfg)
    for e in ("w", "v"):
        deg = emit.read_dict(rig.store, S, e, emit.DEGRADED, rig.now)
        assert deg == {d: "producer_error:behavior.feature_vector" for d in DETS}
        assert "offhours" not in rig.snap(e)["score"]
    # recovery: the next tick scores again
    rig.store.put_health("behavior.feature_vector", {"last_error_ts": None, "error_count": 1})
    rig.step(900.0)
    assert math.isfinite(rig.snap("w")["score"]["offhours"])


# ================================================================ training
def test_training_learns_but_emits_no_alarm_or_event():
    rig = Rig({"w": worker, "sh": backup_at(1.0, 1.67)})
    train(rig, 21)
    rows = rig.run_until(T0 + 21 * DAY + 5 * H, dt=900.0, training=True,
                         patterns={"w": _night, "sh": backup_at(3.0, 3.67)}, record=True,
                         entity="w")
    assert max(r["W_off"] for r in rows) >= H_OFF                  # the evidence is computed
    assert all(r["acc"].get("offhours") == 0 for r in rows)        # ... but never alarms
    assert rig.model("sh")["shifts"]                               # the shift is recorded
    assert not rig.store.events(S)                                 # no events in warm-up
    # the night activity is learned (training: trust = 1)
    assert rig.model("w")["state"]["n_slots"] > 21 * 96 - 8


# =============================================================== governance
def test_trust_weighting_quarantine_and_frozen():
    rig = Rig({"w": worker})
    train(rig, 2)
    n0 = rig.model("w")["state"]["n_commit"]
    N0 = rig.model("w")["state"]["N48"].sum()
    # live, trust 0: rows are processed but carry no weight
    for _ in range(12):
        rig.step(H)
        set_trust(rig.store, S, "w", [rig.now], 0.0)
    m = rig.model("w")
    assert m["state"]["N48"].sum() <= N0 + 1e-9
    # quarantine: rows are held, not committed
    for _ in range(12):
        rig.step(H)
        set_trust(rig.store, S, "w", [rig.now], 1.0, quarantine=1.0)
    g = rig.model("w")["gate"]
    assert len(g.held) >= 8
    n_q = rig.model("w")["state"]["n_commit"]
    # frozen: nothing is committed
    put_model(rig.store, S, "w", "model.control", {"version": 1, "frozen": True})
    for _ in range(8):
        rig.step(H)
        set_trust(rig.store, S, "w", [rig.now], 1.0)
    assert rig.model("w")["state"]["n_commit"] == n_q
    assert n_q >= n0


def test_rollback_to_restores_the_rhythm_counts():
    rig = Rig({"w": worker})
    train(rig, 3)
    tau = rig.now - 6 * H
    snap = None
    # replay target: the state once every row <= tau was committed (D = 4 ticks)
    ref = Rig({"w": worker})
    ref.run_until(tau + 4 * H, dt=H, training=True)
    snap = copy.deepcopy(ref.model("w")["state"])
    assert snap["t_last"] <= tau + 1e-6
    # a night of attack learned in training, then rolled back
    rig.run_until(rig.now + 6 * H, dt=H, training=True, patterns={"w": lambda t: True})
    assert rig.model("w")["state"]["A48"].sum() > snap["A48"].sum() + 3
    # the governor quarantines at or before the tick it writes rollback_to
    set_trust(rig.store, S, "w", [rig.now], 1.0, quarantine=1.0)
    put_model(rig.store, S, "w", "model.control", {"version": 1, "rollback_to": tau})
    rig.step(H, training=True, patterns={"w": lambda t: False})
    st = rig.model("w")["state"]
    assert st["n_slots"] == snap["n_slots"]
    np.testing.assert_allclose(st["A48"], snap["A48"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(st["N48"], snap["N48"], rtol=1e-9, atol=1e-12)
    assert len(rig.model("w")["gate"].held) >= 6


# =========================================================== class / system
def test_class_prior_tier_and_class_rhythm():
    pats = {f"c{i}": worker for i in range(3)}
    rig = Rig(pats)
    assign = {f"{S}|c{i}": {"role": "7", "prob": 0.9} for i in range(3)}
    put_model(rig.store, ORG[0], ORG[1], "model.class",
              {"assign": assign, "roles": {"7": {"name": "office", "members": list(assign)}},
               "version": 1})
    train(rig, 8)
    cm = rig.store.get_model(S, "class:7", R.MODEL)
    assert cm["kind"] == "class" and cm["n_members"] == 3
    # the class rhythm B18 consumes: expected fraction of members active
    t_day = T0 + 7 * DAY + 10.5 * H                                  # Monday 10:30
    t_night = T0 + 7 * DAY + 3 * H
    assert R.class_fraction(cm, make_tctx(t_day, TZ)) > 0.8
    assert R.class_fraction(cm, make_tctx(t_night, TZ)) < 0.1
    m = rig.model("c0")
    assert m["prior"]["tier"] == "class:7" and m["prior"]["s"] == pytest.approx(6.0)
    sm = rig.store.get_model(S, SYSTEM_KEY, R.MODEL)
    assert sm["kind"] == "system" and sm["n_members"] == 3
    # profile.extra.rhythm descriptors
    rh = rig.store.profile(S, "c0").extra["rhythm"]
    assert rh["active_window"]["start"] == "09:00" and rh["active_window"]["end"] == "18:00"
    assert 13.0 <= rh["mu_h"] <= 14.0 and rh["R"] > 0.5
    assert rh["wd_we_ratio"] is None and rh["prior_tier"] == "class:7"
    assert rh["window80"]["len_h"] <= 9.0


def test_small_class_backs_off_to_system_tier():
    rig = Rig({"a": worker, "b": worker, "c": backup_at(2.0, 2.67)})
    assign = {f"{S}|{e}": {"role": "9", "prob": 0.9} for e in ("a", "b")}
    put_model(rig.store, ORG[0], ORG[1], "model.class", {"assign": assign, "version": 1})
    train(rig, 3)
    assert rig.model("a")["prior"]["tier"] == "system"
    assert rig.model("a")["prior"]["s"] == pytest.approx(R.SYSTEM_S)


# ================================================================ accessors
def test_m_rhythm_accessors_loglik_shapes_roundtrip():
    rig = Rig({"w": worker})
    train(rig, 14)
    m = rig.model("w")
    # shapes: normalised, mass in the 09-18 workday hours
    s48, s168 = R.shape48(m), R.shape168(m)
    assert s48.shape == (48,) and s168.shape == (168,)
    assert s48.sum() == pytest.approx(1.0) and s168.sum() == pytest.approx(1.0)
    assert s48[9:18].sum() > 0.8 and s48[9:18].min() > 5 * s48[:9].max()
    assert s168[:5 * 24].sum() > 0.9                                # Mon-Fri
    # loglik of a normal Monday beats the same activity shifted to the night
    t_mon = T0 + 14 * DAY
    tctx = [make_tctx(t_mon + k * 900.0, TZ) for k in range(96)]
    normal = [1.0 if 36 <= k < 72 else 0.0 for k in range(96)]
    night = [1.0 if k < 36 else 0.0 for k in range(96)]
    ll_n, ll_x = R.loglik(m, normal, tctx), R.loglik(m, night, tctx)
    assert ll_n > ll_x + 50.0
    # (c48, c168) pairs give the same answer as tctx dicts; NaN slots are skipped
    pairs = [R.cells_of_tctx(t) for t in tctx]
    assert R.loglik(m, normal, pairs) == pytest.approx(ll_n)
    terms = R.loglik_terms(m, [math.nan] + normal[1:], tctx)
    assert math.isnan(terms[0]) and np.isfinite(terms[1:]).all()
    assert math.isnan(R.loglik(m, [math.nan], tctx[:1]))
    with pytest.raises(ValueError):
        R.loglik(m, [1.0, 0.0], tctx[:1])
    # window_cells agrees with the tctx cells
    wc = R.window_cells(t_mon, t_mon + DAY, TZ)
    assert [(c48, c168) for _, c48, c168 in wc] == pairs
    # JSON round trip preserves every probability
    d = R.to_dict(m)
    import json
    m2 = R.from_dict(json.loads(json.dumps(d)))
    np.testing.assert_allclose(R.profile48(m2), R.profile48(m), rtol=1e-12)
    # bad input: NaN, never an exception
    assert math.isnan(R.p_cell(m, 999)) and math.isnan(R.p_cell({}, 0))
    # descriptors / detector state
    desc = R.descriptors(m)
    assert desc["active_window"]["start"] == "09:00" and desc["machine_like"] in (True, False)
    ds = R.detector_state(m)
    assert ds["W_off"] == 0.0 and ds["alarm"] == {"offhours": 0, "silence": 0}


def test_series_and_profile_written_every_tick():
    rig = Rig({"w": worker})
    train(rig, 2)
    rig.step(900.0)
    v = rig.store.latest_derived(S, "w", SERIES)
    assert v.ts == rig.now and {"p_expected", "W_off", "s_sil"} <= set(v.value)
    assert rig.store.profile(S, "w").extra["rhythm"]["version"] == 0


def test_link_seeding_gives_the_new_ip_half_the_old_rhythm():
    rig = Rig({"old": worker})
    train(rig, 7)
    old = copy.deepcopy(rig.model("old")["state"])
    rig.patterns["new"] = worker
    rig.step(H, training=True)
    put_model(rig.store, S, SYSTEM_KEY, "model.link",
              {"version": 1, "links": [{"from": "old", "to": "new", "ts": rig.now}]})
    rig.step(H, training=True)
    st = rig.model("new")["state"]
    g = R.decay_factor(old["t_ref"], st["t_ref"])
    assert st["N48"].sum() == pytest.approx(0.5 * g * old["N48"].sum(), rel=0.05)
    m = rig.model("new")
    assert R.p_cell(m, R.cell48(10, 0)) > 0.8 and R.p_cell(m, R.cell48(3, 0)) < 0.1
    # once per link version
    n1 = st["N48"].sum()
    rig.step(H, training=True)
    assert rig.model("new")["state"]["N48"].sum() < n1 + 8
