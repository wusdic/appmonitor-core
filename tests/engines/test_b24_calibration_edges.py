"""B24 CalibrationEngine: edge cases, governance (contract H), class pooling,
identity strata, link seeding, calibration health, profile, m_calib accessors
and a perf bound. The `Rig` driver is shared with test_b24_calibration.py."""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest

from helpers import DT, T0, make_store, put_model, run_engine, set_trust
from test_b24_calibration import E, S, Rig

from app.engines.behavior import calibration as B24
from app.engines.behavior.calibration import CalibrationEngine
from app.engines.behavior.lib import calib, emit, gating, m_calib
from app.engines.behavior.lib.detectors import DETECTORS

ST = calib.stratum_key("wd_day", 900)


def ring_n(rig, d="marg_int", st=ST, e=E):
    return m_calib.ring_size(rig.model(e), d, st)


# ------------------------------------------------------------ empty / silent
def test_empty_store_is_a_no_op():
    st = make_store()
    eng = CalibrationEngine()
    assert run_engine(eng, st, T0) == 0
    st.register_entity(S, E)
    assert run_engine(eng, st, T0 + DT) == 0
    assert st.get_model(S, E, m_calib.MODEL) is None
    assert st.latest_derived(S, "__system__", B24.CALIB_HEALTH) is None


def test_silent_entity_gets_no_p_but_keeps_committing():
    rng = np.random.default_rng(1)
    rig = Rig(daypart="wd_day")
    for _ in range(30):
        rig.step({E: {"marg_int": float(rng.exponential())}})
    n0 = ring_n(rig)
    silent = []
    for _ in range(10):                       # no detector scored: absence is data
        ts = rig.t
        silent.append(ts)
        assert rig.step({}) == ts
        assert emit.read_row(rig.store, S, E, emit.P, ts) == {}
    d = gating.commit_delay_ticks(DT)
    assert ring_n(rig) == n0 + d              # the rows before the silence were committed
    m = rig.model()
    assert all(t not in m["pending"] for t in silent)


def test_pending_bookkeeping_is_bounded_by_score_retention():
    rig = Rig(daypart="wd_day")
    for _ in range(200):                      # > 1 d at 900 s
        rig.step({E: {"marg_int": 1.0}})
    pend = rig.model()["pending"]
    assert len(pend) <= 86400 / DT + 1
    assert min(pend) >= rig.t - DT - 86400
    assert len(rig.model()["gate"].journal) <= 86400 / DT + 1


# ------------------------------------------------------------ training / trust
def test_training_learns_with_default_trust_and_emits_no_events():
    rng = np.random.default_rng(2)
    rig = Rig(daypart="wd_day")
    for _ in range(60):                       # no governor output at all
        rig.step({E: {"marg_int": float(rng.exponential())}}, training=True, govern=False)
    assert ring_n(rig) == 60 - gating.commit_delay_ticks(DT)
    assert rig.store.events() == []
    assert rig.store.incidents() == []


def test_live_missing_trust_admits_nothing():
    rng = np.random.default_rng(3)
    rig = Rig(daypart="wd_day")
    ps = []
    for _ in range(60):
        ts = rig.step({E: {"marg_int": float(rng.exponential())}}, govern=False)
        ps.append(rig.p(E, "marg_int", ts))
    assert ring_n(rig) == 0                   # fail safe: trust 0 when B28 is silent
    assert rig.model()["n_admit"] == 0
    assert all(0.0 < p < 1.0 for p in ps)     # empty ring: p = U, never 1


def test_partial_trust_thins_admission_deterministically():
    def run():
        rng = np.random.default_rng(4)
        rig = Rig(daypart="wd_day")
        for _ in range(204):
            rig.step({E: {"marg_int": float(rng.exponential())}}, trust=0.5)
        return rig
    a, b = run(), run()
    n = ring_n(a)
    assert 70 <= n <= 130                     # ~ 200 x 0.5
    ra = m_calib.ring(a.model(), "marg_int", ST)
    rb = m_calib.ring(b.model(), "marg_int", ST)
    assert np.array_equal(ra.scores, rb.scores) and np.array_equal(ra.ts, rb.ts)


def test_immature_entity_is_not_admitted_until_n_eff_48():
    rng = np.random.default_rng(5)
    rig = Rig(daypart="wd_day", mature=False)
    d = gating.commit_delay_ticks(DT)
    for _ in range(48 + d):
        rig.step({E: {"marg_int": float(rng.exponential())}})
    assert rig.model()["n_admit"] == 0 and ring_n(rig) == 0
    for _ in range(10):
        rig.step({E: {"marg_int": float(rng.exponential())}})
    assert rig.model()["n_admit"] > 0 and ring_n(rig) > 0
    # the own count is in 15-minute equivalents (cadence invariant)
    assert rig.model()["n_own"] == pytest.approx(48 + 10, abs=1e-9)


def test_quarantine_holds_rows_and_frozen_drops_them():
    rng = np.random.default_rng(6)
    rig = Rig(daypart="wd_day")
    for _ in range(40):
        rig.step({E: {"marg_int": float(rng.exponential())}})
    n0 = ring_n(rig)
    for _ in range(12):
        rig.step({E: {"marg_int": 50.0}}, quarantine=1.0)
    m = rig.model()
    assert ring_n(rig) <= n0 + 1              # at most the row committed before t-1 was quarantined
    assert len(m["gate"].held) >= 10
    assert m_calib.ring(m, "marg_int", ST).scores.max() < 50.0
    put_model(rig.store, S, E, "model.control", {"version": 0, "frozen": True})
    rig.step({E: {"marg_int": 50.0}}, quarantine=1.0)
    assert rig.model()["gate"].held == [] and rig.model()["gate"].frozen


# ------------------------------------------------------------ version / rebase
def test_version_change_resets_rings():
    rng = np.random.default_rng(7)
    rig = Rig(daypart="wd_day")
    put_model(rig.store, S, E, "model.control", {"version": 1})     # first seen: no reset
    for _ in range(40):
        rig.step({E: {"marg_int": float(rng.exponential())}})
    assert ring_n(rig) > 0 and rig.model()["resets"] == 0
    put_model(rig.store, S, E, "model.control", {"version": 2})
    rig.step({E: {"marg_int": 1.0}})
    m = rig.model()
    assert m["resets"] == 1 and m["version"] == 2
    assert ring_n(rig) <= 1


def test_rebase_resets_then_replays_held_rows_after_onset():
    rng = np.random.default_rng(8)
    rig = Rig(daypart="wd_day")
    put_model(rig.store, S, E, "model.control", {"version": 0})
    for _ in range(40):
        rig.step({E: {"marg_int": float(rng.exponential())}})
    held_ts = []
    for _ in range(15):                       # new regime, quarantined -> held
        held_ts.append(rig.step({E: {"marg_int": 20.0 + float(rng.exponential())}},
                                quarantine=1.0))
    tau = held_ts[3]
    put_model(rig.store, S, E, "model.control", {"version": 1, "rebase_from": tau})
    rig.step({E: {"marg_int": 21.0}})
    m = rig.model()
    r = m_calib.ring(m, "marg_int", ST)
    assert m["resets"] == 1 and m["version"] == 1
    assert len(r) > 0 and r.ts.min() >= tau and r.scores.min() >= 20.0


def test_rollback_is_rate_limited_and_depth_limited():
    rig = Rig(daypart="wd_day")
    ts = [rig.step({E: {"marg_int": float(i % 7)}}) for i in range(30)]
    # B28 quarantines at (or before) the tick it writes rollback_to
    set_trust(rig.store, S, E, [ts[-1]], 1.0, quarantine=1.0)
    put_model(rig.store, S, E, "model.control", {"rollback_to": ts[20]})
    rig.step({E: {"marg_int": 1.0}}, quarantine=1.0)
    assert m_calib.ring(rig.model(), "marg_int", ST).ts.max() <= ts[20]
    put_model(rig.store, S, E, "model.control", {"rollback_to": ts[10]})
    rig.step({E: {"marg_int": 1.0}}, quarantine=1.0)    # within the hour: deferred
    assert m_calib.ring(rig.model(), "marg_int", ST).ts.max() > ts[10]
    for _ in range(4):
        rig.step({E: {"marg_int": 1.0}}, quarantine=1.0)
    assert m_calib.ring(rig.model(), "marg_int", ST).ts.max() <= ts[10]
    # deeper than 7 d: recorded as applied and skipped (nothing deleted)
    n = ring_n(rig)
    put_model(rig.store, S, E, "model.control", {"rollback_to": T0 - 8 * 86400.0})
    rig.step({E: {"marg_int": 1.0}}, quarantine=1.0)
    assert ring_n(rig) == n
    assert rig.model()["gate"].applied["_last_rollback"].get("skipped") == "depth"


# ------------------------------------------------------------ class keys / pooling
def _class_model(members):
    return {"assign": {f"{S}|{m}": {"role": "r1", "prob": 0.9} for m in members},
            "roles": {"r1": {"name": "office", "members": [f"{S}|{m}" for m in members]}}}


def test_class_pseudo_entity_is_calibrated_without_tctx():
    rng = np.random.default_rng(9)
    ck = "class:r1"
    rig = Rig(entities=(E, ck))               # a class key lives under a real system
    for _ in range(80):
        ts = rig.step({ck: {"class_int": float(rng.exponential())}})
        assert 0.0 < rig.p(ck, "class_int", ts) < 1.0
    assert ck in rig.store.pseudo_entities(S)
    rs = m_calib.rings(rig.model(ck))
    assert rs and all(k.startswith("class_int@") for k in rs)


def test_small_ring_blends_with_the_class_pooled_ring():
    rng = np.random.default_rng(10)
    members = ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"]
    rig = Rig(entities=members, daypart="wd_day")
    put_model(rig.store, "__org__", "__org__", "model.class", _class_model(members))
    for _ in range(40):                       # the newcomer is silent for now
        rig.step({m: {"marg_int": float(rng.exponential())} for m in members[:3]})
    new = members[3]
    for _ in range(12):
        x = float(rng.exponential())
        ts = rig.step({m: {"marg_int": float(rng.exponential())} for m in members[:3]}
                      | {new: {"marg_int": x}})
        own = m_calib.ring(rig.model(new), "marg_int", ST)
        n = 0 if own is None else len(own)
        u = m_calib.uniform(S, new, "marg_int", ts)
        pool = [m_calib.ring(rig.model(m), "marg_int", ST) for m in members[:3]]
        pp, npool = m_calib.pooled_p(pool, x, u)
        assert npool >= 64 and pp == pp
        conf = calib.p_from_ring(own if own is not None else calib.Ring(), x, u)
        expect = calib.blend_small_sample(conf, pp, n)
        assert rig.p(new, "marg_int", ts) == m_calib.issued(expect)
        assert m_calib.issued(m_calib.p_from_snapshot(
            rig.model(new), "marg_int", ST, x, u, pooled=pool)) == rig.p(new, "marg_int", ts)


def test_identity_uses_regime_strata_with_settled_fallback():
    from app.models.schema import DerivedMetric, MetricKind
    rng = np.random.default_rng(11)
    rig = Rig(daypart="wd_day")

    def regime(state):
        rig.store.add_derived(DerivedMetric(name="behavior.regime", value={"state": state},
                                            ts=rig.t, system=S, entity=E, window_s=900,
                                            kind=MetricKind.CATEGORICAL))
    for _ in range(90):
        regime("NORMAL")
        rig.step({E: {"identity": float(rng.exponential())}})
    r0 = calib.identity_stratum_key("wd_day", 0, 900)
    assert ring_n(rig, "identity", r0) >= 64
    regime("SUSPECT")
    x = 2.5
    ts = rig.step({E: {"identity": x}})
    m = rig.model()
    assert m_calib.ring_size(m, "identity", calib.identity_stratum_key("wd_day", 1, 900)) == 0
    u = m_calib.uniform(S, E, "identity", ts)
    settled = m_calib.p_value(m_calib.ring(m, "identity", r0), x, u)
    assert rig.p(E, "identity", ts) == pytest.approx(m_calib.issued(settled), rel=1e-6)
    assert m_calib.regime_tercile({"state": "accepted"}) == 2
    assert m_calib.regime_tercile({"state": "normal", "tercile": 1}) == 1
    assert m_calib.regime_tercile(None) == 0


def test_link_seeding_copies_half_the_source_ring():
    rng = np.random.default_rng(12)
    a, b = "10.0.0.1", "10.0.0.9"
    rig = Rig(entities=(a, b), daypart="wd_day")
    for _ in range(300):
        rig.step({a: {"marg_int": float(rng.exponential())}})
    rig.step({a: {"marg_int": 1.0}, b: {"marg_int": 1.0}})
    assert ring_n(rig, e=b) == 0
    put_model(rig.store, S, "__system__", "model.link",
              {"version": 1, "links": [{"from": a, "to": b, "ts": rig.t}]})
    rig.step({a: {"marg_int": 1.0}, b: {"marg_int": 1.0}})
    rb = m_calib.ring(rig.model(b), "marg_int", ST)
    ra = m_calib.ring(rig.model(a), "marg_int", ST)
    assert len(rb) == calib.RING_M // 2
    assert set(rb.ts.tolist()) <= set(ra.ts.tolist()) | set()
    assert rig.model(b)["gate"].link_version == 1


# ------------------------------------------------------------ health
def test_health_flags_a_drifting_detector_only():
    rng = np.random.default_rng(13)
    ents = [f"10.0.1.{i}" for i in range(4)]
    rig = Rig(entities=ents, daypart="wd_day")
    for i in range(300):
        rig.step({e: {"marg_int": float(rng.exponential()),
                      "t2": float(rng.exponential()) + 0.03 * i} for e in ents})
    h = rig.store.latest_derived(S, "__system__", B24.CALIB_HEALTH)
    assert h is not None and h.ts == rig.t - DT
    out = h.value
    assert out["marg_int"]["n"] >= B24.HEALTH_KS_MIN
    assert out["marg_int"]["weight_mult"] == 1.0 and out["marg_int"]["ks"] < 0.05
    assert out["t2"]["weight_mult"] == 0.5 and out["t2"]["ks"] > 0.05
    assert m_calib.weight_mult(out, "t2") == 0.5 and m_calib.weight_mult(out, "seq") == 1.0
    prof = rig.store.profile(S, ents[0]).extra["calibration"]
    assert prof["weight_mult"] == {"t2": 0.5} and prof["healthy"] is False


def test_evaluate_health_rate_ratio_and_neutral_nan():
    n = len(DETECTORS)
    bufs = [[] for _ in range(n)]
    exc, exp_ = np.zeros(n), np.zeros(n)
    i, j, k = DETECTORS.index("marg_int"), DETECTORS.index("t2"), DETECTORS.index("seq")
    exp_[i], exc[i] = 20.0, 50.0              # 2.5x the budget -> unhealthy
    exp_[j], exc[j] = 20.0, 22.0              # fine
    exp_[k], exc[k] = 5.0, 40.0               # too little expectation: neutral
    out = B24.evaluate_health(bufs, exc, exp_)
    assert out["marg_int"]["rate_ratio"] == 2.5 and out["marg_int"]["weight_mult"] == 0.5
    assert out["t2"]["weight_mult"] == 1.0
    assert math.isnan(out["seq"]["rate_ratio"]) and out["seq"]["weight_mult"] == 1.0
    assert math.isnan(out["seq"]["ks"]) and "spe" not in out


def test_health_rate_counts_untrusted_commits_but_ks_does_not():
    hs = B24.new_health()
    p = np.full(len(DETECTORS), np.nan)
    p[0] = 1e-6
    B24.health_add(hs, T0, 900.0, p, trusted=False)
    B24.health_add(hs, T0 + 900.0, 900.0, np.full(len(DETECTORS), np.nan), trusted=True)
    assert hs["exc"][0] == pytest.approx(2.0 ** (-0.0), rel=1e-3)
    assert hs["exp"][0] == pytest.approx(0.03 * 900 / 86400, rel=1e-3)
    assert len(hs["ks"][0]) == 0 and hs["exp"][1] == 0.0


# ------------------------------------------------------------ profile / accessors
def test_profile_extra_calibration():
    rng = np.random.default_rng(14)
    rig = Rig(daypart="wd_day")
    for _ in range(30):
        rig.step({E: {"marg_int": float(rng.exponential()), "silence": 0.0}})
    info = rig.store.profile(S, E).extra["calibration"]
    assert info["stratum"] == ST and info["mature"] is True
    assert set(info["n_current"]) == {"marg_int", "silence"}
    assert info["small_sample"] == ["marg_int", "silence"]
    assert info["healthy"] is True


def test_b25_meta_rings_are_preserved_and_found():
    rng = np.random.default_rng(15)
    rig = Rig(daypart="wd_day")
    rig.step({E: {"marg_int": 1.0}})
    m = rig.model()
    meta = calib.Ring(scores=[1.0, 2.0, 3.0], ts=[1.0, 2.0, 3.0])
    m["meta"] = {"meta_inst@" + ST: meta}           # what B25 adds to the same dict
    for _ in range(20):
        rig.step({E: {"marg_int": float(rng.exponential())}})
    m = rig.model()
    assert m["meta"]["meta_inst@" + ST] is meta
    assert m_calib.ring(m, "meta_inst", ST) is meta
    assert "meta_inst@" + ST not in m_calib.rings(m)
    # a model B25 created first gets the B24 keys added, not replaced
    put_model(rig.store, S, "10.0.0.7", m_calib.MODEL, {"meta": {"x": 1}})
    rig.store.register_entity(S, "10.0.0.7")
    rig.keys.append("10.0.0.7")
    rig.step({"10.0.0.7": {"marg_int": 1.0}})
    m7 = rig.model("10.0.0.7")
    assert m7["meta"] == {"x": 1} and "rings" in m7


def test_m_calib_accessors_and_json():
    rng = np.random.default_rng(16)
    rig = Rig(daypart="wd_day")
    for _ in range(320):
        rig.step({E: {"cusum": float(rng.exponential())}})
    m = rig.model()
    r = m_calib.ring(m, "cusum", ST)
    assert r.gpd is not None
    for p in (0.5, 0.05, 1e-3, 1e-6):
        s = m_calib.score_at_p(m, "cusum", ST, p)
        back = m_calib.p_from_snapshot(m, "cusum", ST, s * (1 + 1e-6), 0.5)
        assert back == pytest.approx(p, rel=0.25, abs=2.0 / (len(r) + 1))
    assert math.isnan(m_calib.score_at_p(m, "cusum", "nwd_night|60", 0.01))
    assert m_calib.quantile(m, "cusum", ST, 0.5) == pytest.approx(math.log(2), rel=0.3)
    desc = m_calib.describe(m)
    assert desc["detectors"]["cusum"][ST] == len(r) and desc["n_tails"] >= 1
    json.dumps(m_calib.to_json(m), allow_nan=False)
    sysm = rig.store.get_model(S, "__system__", m_calib.MODEL)
    json.dumps(m_calib.to_json(sysm), allow_nan=False)
    # dict-form rings are accepted everywhere (a model restored from JSON)
    mj = m_calib.to_json(m)
    assert m_calib.p_from_snapshot(mj, "cusum", ST, 3.0, 0.3) == \
        m_calib.p_from_snapshot(m, "cusum", ST, 3.0, 0.3)


def test_restored_json_model_keeps_calibrating():
    rng = np.random.default_rng(17)
    rig = Rig(daypart="wd_day")
    for _ in range(100):
        rig.step({E: {"marg_int": float(rng.exponential())}})
    n = ring_n(rig)
    blob = m_calib.to_json(rig.model())
    blob["profile_ts"] = None                 # -inf / NaN become null in JSON
    blob["dt"] = None
    rig.store.put_model(S, E, m_calib.MODEL, json.loads(json.dumps(blob)))
    sysm = rig.store.get_model(S, "__system__", m_calib.MODEL)
    rig.store.put_model(S, "__system__", m_calib.MODEL,
                        json.loads(json.dumps(m_calib.to_json(sysm))))
    rig.step({E: {"marg_int": 1.0}})
    m = rig.model()
    assert isinstance(m["gate"], gating.GateState)
    assert all(isinstance(x, calib.Ring) for x in m["rings"].values())
    assert ring_n(rig) >= n
    hs = rig.store.get_model(S, "__system__", m_calib.MODEL)["health"]
    assert isinstance(hs["exc"], np.ndarray) and len(hs["ks"][0]) > 0


# ------------------------------------------------------------ perf
def test_perf_40_keys_31_detectors():
    """40 keys x 31 detectors, every ring full (the worst case: all detectors
    scored at every key): under a generous 80 ms per tick."""
    rng = np.random.default_rng(18)
    ents = [f"10.0.2.{i}" for i in range(40)]
    rig = Rig(entities=ents, daypart="wd_day")
    for e in ents:                           # pre-populated mature rings with tails
        model = B24.new_model()
        for d in DETECTORS:
            key = calib.ring_key(d, calib.identity_stratum_key("wd_day", 0, 900)
                                 if d == "identity" else ST)
            r = calib.Ring(scores=rng.exponential(size=calib.RING_M),
                           ts=T0 - 1e5 + np.arange(calib.RING_M, dtype=float))
            r.gpd = calib.fit_tail(r)
            model["rings"][key] = r
        rig.store.put_model(S, e, m_calib.MODEL, model)
    times = []
    for _ in range(12):
        scores = {e: {d: float(x) for d, x in zip(DETECTORS, rng.exponential(size=31))}
                  for e in ents}
        pm = {e: {d: 0.5 for d in DETECTORS} for e in ents}
        ts = rig.t
        for e, sc in scores.items():
            emit.write_scores(rig.store, S, e, ts, sc, pm=pm[e])
            rig.write_tctx(e, ts, DT)
        t0 = time.perf_counter()
        n = run_engine(rig.eng, rig.store, ts, dt=DT)
        times.append(time.perf_counter() - t0)
        for e in ents:
            set_trust(rig.store, S, e, [ts], 1.0)
        rig.t = ts + DT
        assert n == 40 * 31
    per_tick = float(np.median(times[2:]))
    assert per_tick < 0.080, per_tick         # measured ~25-30 ms on a shared 4-core box


def test_identity_rings_are_per_cadence_class():
    """The identity score depends on the cadence (B16 windows are K active
    ticks: 4 h at 3600 s, 1 h at 900 s), so its rings are stratified by
    cadence class too. After 3600 s -> 900 s a 900-s identity score is not
    judged against the (much tighter) 3600-s ring: the new stratum starts
    empty and blends with pm (pi_self). Eval pack A: every workday identity
    ring held only 3600-s scores ~1e-6..1e-3, the first 900-s score (0.02,
    pi_self = 0.95) hit its GPD tail at p = 1e-38 and opened an incident on
    every enrolled entity."""
    rng = np.random.default_rng(4)
    rig = Rig(daypart="wd_day", dt=3600.0)
    for _ in range(200):
        rig.step({E: {"identity": float(rng.uniform(1e-6, 1e-3))}},
                 pm={E: {"identity": 0.9999}})
    r3600 = calib.identity_stratum_key("wd_day", 0, 3600)
    assert m_calib.ring_size(rig.model(), "identity", r3600) >= 64
    ts = rig.step({E: {"identity": 0.0226}}, pm={E: {"identity": 0.95}}, dt=900.0)
    assert m_calib.ring_size(rig.model(), "identity", calib.identity_stratum_key(
        "wd_day", 0, 900)) == 0
    assert rig.p(E, "identity", ts) == pytest.approx(0.95, rel=1e-6)
