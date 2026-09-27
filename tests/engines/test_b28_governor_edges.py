"""B28 GovernorEngine edges: empty store, silent entity, NaN inputs, a failed
fusion tick, training, the 900 -> 60 s cadence switch, the accumulator guard,
the pure log-odds arithmetic of lib/m_governor and its read accessors, and a
perf bound. Fixtures come from test_b28_governor.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from helpers import DT, T0, make_store, run_engine

from test_b28_governor import (DAY, E, HOUR, NAN, S, Sim, control, put1, put_acc, put_model,
                               regime_events, ring, states)

from app.engines.behavior import governor as G
from app.engines.behavior.governor import GovernorEngine
from app.engines.behavior.lib import emit
from app.engines.behavior.lib import m_governor as MG
from app.engines.behavior.lib.detectors import ACC_DETECTORS
from app.models.schema import Incident


# ================================================================ edges
def test_empty_store_does_nothing():
    store = make_store()
    assert run_engine(GovernorEngine(), store, T0) == 0
    assert run_engine(GovernorEngine(), store, T0, training=True) == 0
    assert store.systems() == []


def test_silent_entity_is_trusted_and_normal():
    sim = Sim()
    ts = sim.run(12)                                   # registered, nothing observed
    for t in ts:
        assert ring(sim.store, E, MG.TRUST, t) == 1.0
        assert ring(sim.store, E, MG.TRUST_PROV, t) == 1.0
        assert ring(sim.store, E, MG.QUARANTINE, t) == 0.0
    assert states(sim.store) == [] and sim.regime() == MG.NORMAL
    # behavior.regime: first tick, then an hourly heartbeat (not every tick)
    pts = sim.store.derived_series(S, E, MG.REGIME)
    assert 3 <= len(pts) <= 4 and all(p.value["state"] == "normal" for p in pts)
    assert sim.store.profile(S, E).extra["regime"]["state"] == "normal"


def test_nan_inputs_are_neutral_never_trusted_as_p1():
    sim = Sim()
    # q_inst unscored -> q_all is the evidence (1e-5 at 900 s: rarer than 1 per 100 d)
    t = sim.tick(lambda st, t: (put1(st, E, "behavior.q_inst", t, NAN),
                                put1(st, E, "behavior.q_all", t, 1e-5)))
    assert ring(sim.store, E, MG.TRUST_PROV, t) == 0.0
    # both unscored -> the other factors decide (absence is not suspicion)
    t = sim.tick(lambda st, t: (put1(st, E, "behavior.q_inst", t, NAN),
                                put1(st, E, "behavior.q_all", t, NAN)))
    assert ring(sim.store, E, MG.TRUST_PROV, t) == 1.0
    # an all-NaN p row and an all-NaN feature row: nothing scored, nothing triggered
    def nan_rows(st, t):
        emit.write_pvalues(st, S, E, t, {d: None for d in ACC_DETECTORS})
        st.add_vec(S, E, "feature.vec", t, np.full(52, np.nan, dtype=np.float32))
    t = sim.tick(nan_rows)
    assert ring(sim.store, E, MG.TRUST, t) == 1.0 and sim.regime() == MG.NORMAL
    # an accumulator alarm whose p is NaN still counts (level 1)
    t = sim.tick(lambda st, t: put_acc(st, E, t, {"cusum": 1}))
    assert ring(sim.store, E, MG.TRUST, t) == 0.0
    assert sim.regime() == MG.SUSPECT


def test_fusion_failure_writes_nan_trust_and_is_never_quiet():
    sim = Sim()
    sim.run(4, lambda st, t: sim.normal(E, t))
    sim.run(2, lambda st, t: sim.shifted(E, t, 7.0))
    assert sim.regime() == MG.SUSPECT
    def failed(st, t):
        st.put_health("behavior.fusion", {"engine": "behavior.fusion", "ok": False,
                                          "last_error_ts": t})
    ts = sim.run(12, failed)                           # 3 h of failed fusion ticks
    for t in ts:
        assert math.isnan(ring(sim.store, E, MG.TRUST, t))
        assert math.isnan(ring(sim.store, E, MG.TRUST_PROV, t))
        assert ring(sim.store, E, MG.QUARANTINE, t) == 1.0
    assert "returned" not in states(sim.store)
    ts = sim.run(9, lambda st, t: sim.normal(E, t))    # quiet counts from the last failure
    assert "returned" in states(sim.store)
    ret = [x for x in regime_events(sim.store) if x.extra["state"] == "returned"][0]
    assert ret.ts - (ts[0] - DT) >= 2 * HOUR


def test_training_emits_no_events_and_keeps_regime_normal():
    sim = Sim()
    sim.store.put_incident(Incident(system=S, entity=E, status="open", opened=T0, last_seen=T0))
    ts = sim.run(10, lambda st, t: (sim.shifted(E, t, 9.0), put1(st, E, "behavior.cp.prob", t,
                                                                 0.99)), training=True)
    assert sim.store.events(system=S, kinds=("regime",)) == []
    assert sim.regime() == MG.NORMAL
    for t in ts:
        assert ring(sim.store, E, MG.TRUST, t) == 1.0
        assert ring(sim.store, E, MG.QUARANTINE, t) == 1.0    # the incident still quarantines
    assert control(sim.store) == {}


def test_cadence_switch_900_to_60_keeps_wall_clock_rules():
    sim = Sim()
    sim.run(96, lambda st, t: sim.normal(E, t))
    t_s = sim.tick(lambda st, t: sim.shifted(E, t, 7.0))     # SUSPECT at 900 s
    assert sim.regime() == MG.SUSPECT
    # switch to 60 s ticks: DRIFTING needs 1 h of wall clock, not 4 ticks
    alarm60 = lambda st, t: sim.shifted(E, t, 7.0, dt=60.0)  # noqa: E731
    ts = sim.run(70, alarm60, dt=60.0)
    drift = [x for x in regime_events(sim.store) if x.extra["state"] == "drifting"][0]
    assert drift.ts - t_s >= HOUR and drift.ts - t_s <= HOUR + 60.0
    # the evidence factor uses the real dt: q = 0.01 is 14 per day at 60 s (trusted),
    # 0.96 per day at 900 s (partly trusted)
    assert G.evidence_factor(0.01, NAN, 60.0) == 1.0
    assert 0.9 < G.evidence_factor(0.01, NAN, 900.0) < 1.0
    # quiet for max(8 ticks, 2 h) = 2 h of wall clock at 60 s
    t_last = ts[-1]
    q = sim.run(125, lambda st, t: sim.normal(E, t, dt=60.0), dt=60.0)
    ret = [x for x in regime_events(sim.store) if x.extra["state"] == "returned"]
    assert len(ret) == 1
    assert 2 * HOUR <= ret[0].ts - t_last <= 2 * HOUR + 60.0
    assert ring(sim.store, E, MG.TRUST, q[-1]) > 0.0


def test_accumulator_half_h_zeroes_trust_but_guarded_suspect():
    sim = Sim()
    sim.run(4, lambda st, t: sim.normal(E, t))
    n_acc = len(ACC_DETECTORS)

    def pvals(p_cusum):
        def f(st, t):
            sim.normal(E, t)
            pv = {d: 0.5 for d in ACC_DETECTORS}
            pv["cusum"] = p_cusum
            emit.write_pvalues(st, S, E, t, pv)
        return f

    lnarl = math.log(G.seq.arl_ticks(G.arl_days("cusum"), DT))
    p_half = math.exp(-0.55 * lnarl)                   # level 0.55 >= h/2
    assert G.acc_level(p_half, lnarl) == pytest.approx(0.55)
    assert G.combine.e_day(p_half, DT) * n_acc > G.ACC_SUSPECT_E_DAY
    t = sim.tick(pvals(p_half))
    assert ring(sim.store, E, MG.TRUST, t) == 0.0      # every accumulator < h/2 fails
    assert ring(sim.store, E, MG.TRUST_PROV, t) > 0.0
    assert sim.regime() == MG.NORMAL                   # the guard: not rare enough to suspect
    p_rare = 1e-5
    assert G.combine.e_day(p_rare, DT) * n_acc <= G.ACC_SUSPECT_E_DAY
    sim.tick(pvals(p_rare))
    assert sim.regime() == MG.SUSPECT
    ev = regime_events(sim.store)
    assert ev[0].extra["trigger"] == "accumulator"


def test_cp_prob_and_creep_event_trigger_suspect():
    sim = Sim(entities=[E, "10.0.0.2"])
    sim.run(2, lambda st, t: (sim.normal(E, t), sim.normal("10.0.0.2", t)))
    sim.tick(lambda st, t: put1(st, E, "behavior.cp.prob", t, 0.8))
    assert sim.regime() == MG.SUSPECT
    assert regime_events(sim.store)[0].extra["trigger"] == "cp_prob"
    from app.models.schema import BehaviorEvent, Severity
    sim.tick(lambda st, t: st.add_event(BehaviorEvent(
        system=S, entity="10.0.0.2", ts=t, kind="baseline_creep", score=0.3,
        severity=Severity.LOW, axes=["volume"])))
    assert sim.regime("10.0.0.2") == MG.SUSPECT
    ev = regime_events(sim.store, "10.0.0.2")[0]
    assert ev.extra["trigger"] == "baseline_creep" and ev.extra["type"] == "ramp"


def test_discrete_finding_medium_zeroes_trust_prov_suppressed_does_not():
    from app.models.schema import BehaviorEvent, Severity
    sim = Sim()
    t = sim.tick(lambda st, t: (sim.normal(E, t), st.add_event(BehaviorEvent(
        system=S, entity=E, ts=t, kind="rare_access", score=0.5, severity=Severity.MEDIUM))))
    assert ring(sim.store, E, MG.TRUST_PROV, t) == 0.0
    assert sim.regime() == MG.NORMAL                   # findings gate trust, not the regime
    t = sim.tick(lambda st, t: (sim.normal(E, t), st.add_event(BehaviorEvent(
        system=S, entity=E, ts=t, kind="rare_access", score=0.5, severity=Severity.MEDIUM,
        status="suppressed"))))
    assert ring(sim.store, E, MG.TRUST_PROV, t) == 1.0
    t = sim.tick(lambda st, t: (sim.normal(E, t), st.add_event(BehaviorEvent(
        system=S, entity=E, ts=t, kind="first_seen", score=0.1, severity=Severity.LOW))))
    assert ring(sim.store, E, MG.TRUST_PROV, t) == 1.0


# ================================================================ arithmetic
def test_spec_log_odds_arithmetic():
    t3 = MG.time_evidence("intensity", 3)
    assert t3 == pytest.approx(math.log(8.0))
    assert MG.time_evidence("intensity", 7) == pytest.approx(math.log(8.0))   # capped
    assert MG.time_evidence("identity", 3) == 0.0
    assert MG.time_evidence("c2", 3) == 0.0
    x = MG.logodds("intensity", {"id_self": 0.69, "dispersion": 0.69, "time": t3})
    assert x == pytest.approx(3.46, abs=0.01) and MG.p_from_logodds(x) == pytest.approx(0.97,
                                                                                        abs=0.01)
    assert MG.decide("intensity", x, DAY, False, False) == "accept"
    assert MG.decide("intensity", x, DAY - 1, False, False) is None
    xs = MG.logodds("shape", {"id_self": 0.69, "dispersion": 0.69, "time": t3})
    assert MG.p_from_logodds(xs) == pytest.approx(0.92, abs=0.01)
    assert MG.decide("shape", xs, 7 * DAY, False, False) == "accept"
    assert MG.decide("shape", xs, 6 * DAY, False, False) is None
    xc = MG.logodds("categorical", {"id_self": 0.69, "dispersion": 0.69, "time": t3})
    assert MG.p_from_logodds(xc) == pytest.approx(0.88, abs=0.01)
    assert MG.decide("categorical", xc, 30 * DAY, False, False) is None      # needs peer / label
    xcp = MG.logodds("categorical", {"peer": MG.LR_PEER, "dispersion": 0.69, "time": t3})
    assert MG.decide("categorical", xcp, 7 * DAY, False, False) == "accept"
    # malicious evidence blocks ACCEPT; low P rejects only with evidence beyond the prior
    assert MG.decide("intensity", 5.0, 9 * DAY, True, False) is None
    assert MG.decide("exfil", MG.logodds("exfil", {}), 9 * DAY, False, False) is None
    x4 = MG.logodds("exfil", {"lib4": MG.LR_LIB4})
    assert MG.decide("exfil", x4, 0.0, True, True) == "reject"
    # ramp prior only for a gentle, stationary ramp; a steep ramp never accepts
    assert MG.prior("ramp", ramp_ok=True) == 0.5 and MG.prior("ramp") == 0.0
    assert MG.decide("ramp", 9.0, 9 * DAY, False, False, ramp_blocked=True) is None
    # new_entity: the 3-day cold-start path
    xn = MG.logodds("new_entity", {"dispersion": 0.69, "time": t3})
    assert MG.p_from_logodds(xn) < 0.9
    assert MG.decide("new_entity", xn, 2.9 * DAY, False, False) is None
    assert MG.decide("new_entity", xn, 3 * DAY, False, False) == "accept"
    assert MG.decide("new_entity", xn, 3 * DAY, False, True) is None
    assert math.isnan(MG.p_from_logodds(NAN))
    assert MG.p_from_logodds(-800.0) == pytest.approx(0.0) and MG.p_from_logodds(800.0) == 1.0


def test_evidence_series_helpers():
    # group means over finite entries; NaN group when none
    row = np.full(52, np.nan)
    row[G.GROUPS["volume"]] = 2.0
    lv = G.group_levels(row)
    assert lv[G._GIDX["volume"]] == 2.0 and np.isnan(lv[G._GIDX["dns"]])
    # EW base moments of a group mean
    base = G.new_base("vec")
    rng = np.random.default_rng(0)
    for i in range(400):
        x = np.full(len(G.GROUP_ORDER), np.nan)
        x[0] = 3.0 + 0.1 * rng.standard_normal()
        G.base_update(base, x, T0 + i * DT)
    m, v = G.base_moments(base, ["volume"])
    assert m == pytest.approx(3.0, abs=0.02) and v == pytest.approx(0.01, rel=0.25)
    assert G.base_moments(base, ["dns"]) == (G._NAN, G._NAN) or math.isnan(
        G.base_moments(base, ["dns"])[0])
    # stationarity: a flat series passes, a trend fails, residuals around the trend pass
    bins_flat = [[T0 + i * 900.0, 1.0, x, x * x] for i, x in
                 enumerate(0.1 * rng.standard_normal(96))]
    now = T0 + 95 * 900.0
    assert G.stationary_steps(bins_flat, now, DAY / 3, T0, 900.0) == 3
    trend = [[b[0], 1.0, b[2] + 2.0 * i / 96, (b[2] + 2.0 * i / 96) ** 2]
             for i, b in enumerate(bins_flat)]
    assert G.stationary_steps(trend, now, DAY / 3, T0, 900.0) == 0
    sl = G.level_slope_per_day(trend)
    assert sl == pytest.approx(2.0, rel=0.1)
    assert G.stationary_steps(trend, now, DAY / 3, T0, 900.0, sl) == 3
    v0, n0 = G.post_variance(bins_flat)
    assert n0 == 96 and v0 == pytest.approx(np.var([b[2] for b in bins_flat], ddof=1))
    v1, _ = G.post_variance(trend, sl)
    assert v1 < 2 * v0
    # the episode's span must lie inside the regime
    assert G.stationary_steps(bins_flat, now, DAY / 3, now - 2 * HOUR, 900.0) == 0


def test_regime_type_precedence():
    rt = G.regime_type
    assert rt({"volume"}, {}, False, False) == "intensity"
    assert rt({"volume", "c2"}, {}, False, False) == "c2"
    assert rt({"volume"}, {"beacon": True}, False, False) == "c2"
    assert rt({"volume"}, {"exfil_budget": True}, False, False) == "exfil"
    assert rt({"identity"}, {}, True, False) == "identity"
    assert rt({"categorical"}, {}, True, False) == "new_entity"
    assert rt({"categorical", "volume"}, {}, False, True) == "categorical"
    assert rt({"volume"}, {}, False, True) == "ramp"
    assert rt({"shape"}, {}, False, False) == "shape"
    assert rt({"temporal"}, {}, False, False) == "rhythm"
    assert rt(set(), {"schedule_shift": True}, False, False) == "rhythm"


# ================================================================ accessors
def test_m_governor_accessors_on_an_episode():
    store = make_store()
    assert MG.regime(store, S, E) == MG.NORMAL
    assert MG.state(store, S, E)["version"] == 0
    assert MG.control(store, S, E)["rollback_to"] is None
    assert MG.is_quarantined(store, S, E) is False
    assert math.isnan(MG.trust(store, S, E))
    assert MG.episodes(store, S, E) == [] and MG.descriptor(store, S, E)["state"] == "normal"
    sim = Sim(seed=11)
    sim.run(8, lambda st, t: sim.normal(E, t))
    t_s = sim.tick(lambda st, t: sim.shifted(E, t, 7.0))
    sim.run(6, lambda st, t: sim.shifted(E, t, 7.0))
    t_now = sim.t
    st = sim.store
    assert MG.regime(st, S, E) == MG.DRIFTING
    assert MG.regime_at(st, S, E, t_s - DT) == MG.NORMAL
    assert MG.regime_at(st, S, E, t_s) == MG.SUSPECT
    assert MG.regime_at(st, S, E, t_now) == MG.DRIFTING
    assert MG.is_quarantined(st, S, E) and MG.is_quarantined(st, S, E, at=t_s)
    assert not MG.is_quarantined(st, S, E, at=t_s - DT)
    assert MG.trust(st, S, E, at=t_now) == 0.0 and MG.trust_prov(st, S, E, at=t_now) == 0.0
    eps = MG.episodes(st, S, E)
    assert len(eps) == 1 and eps[0]["end"] is None and eps[0]["onset"] == t_s - DT
    assert MG.in_regime_window(st, S, E, t_s - DAY + HOUR)
    assert not MG.in_regime_window(st, S, E, t_s - 2 * DAY)
    d = MG.descriptor(st, S, E)
    assert d["state"] == "drifting" and d["type"] == "intensity" and d["history"]
    assert MG.state(st, S, E)["branch"] == 1
    # a NaN ring value (degraded tick) is skipped by is_quarantined
    st.add_vec(S, E, MG.QUARANTINE, t_now + DT, [np.nan])
    assert MG.is_quarantined(st, S, E)
    assert st.profile(S, E).extra["regime"]["state"] == "drifting"


def test_perf_40_entities_4_classes():
    ents = [f"10.0.3.{i}" for i in range(40)]
    sim = Sim(entities=ents, seed=12)
    put_model(sim.store, "__org__", "__org__", "model.class",
              {"assign": {f"{S}|{e}": {"role": str(i % 4), "prob": 1.0}
                          for i, e in enumerate(ents)}})
    movers = set(ents[:4])

    def feed(st, t):
        for e in ents:
            if e in movers and sim.t > T0 + 8 * DT:
                sim.shifted(e, t, 6.0)
            else:
                sim.normal(e, t)

    spent = []
    for _ in range(40):
        sim.t += DT
        feed(sim.store, sim.t)
        t0 = time.perf_counter()
        run_engine(sim.eng, sim.store, sim.t)
        spent.append(time.perf_counter() - t0)
    per_tick = float(np.mean(spent[5:]))
    assert per_tick < 0.03                   # generous: ~1 ms target in the spec
    assert len([k for k in sim.store.pseudo_entities(S) if k.startswith("class:")]) == 4
