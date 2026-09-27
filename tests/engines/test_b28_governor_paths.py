"""B28 GovernorEngine lifecycle paths: no permanent lockout after REJECT,
analyst labels (expected_change / tp through B23's model.feedback records),
identity changes held for a label (14-day label queue), the new_entity
cold-start acceptance, rollback rate limiting, releases (RETURNED and an
incident-only quarantine) replayed by a real lib/gating learner, class-level
fast acceptance, and the profile / profile-version writes.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pytest

from helpers import DT, T0, add_obs_tick

from test_b28_governor import (DAY, E, HOUR, S, Sim, _toy_learner, class_model, control,
                               match, put1, put_acc, put_alarm, put_model, regime_events, ring,
                               states)

from app.engines.behavior.lib import gating
from app.engines.behavior.lib import m_governor as MG
from app.engines.behavior.lib.features import FEATURE_DIM, GROUPS
from app.models.schema import EntityProfile, Incident


def _feedback(accept: Dict = None, freeze: Dict = None, version: int = 1) -> Dict:
    return {"version": version, "accept": accept or {}, "freeze": freeze or {}}


def test_rejected_is_unfrozen_once_quiet_no_permanent_lockout():
    sim = Sim(seed=21)
    sim.run(48, lambda st, t: sim.normal(E, t))
    sim.run(6, lambda st, t: sim.shifted(E, t, 7.0, axes=("exfil",)))
    sim.tick(lambda st, t: (sim.shifted(E, t, 7.0, axes=("exfil",)), match(st, E, t)))
    sim.tick(lambda st, t: sim.shifted(E, t, 7.0, axes=("exfil",)))
    assert sim.regime() == MG.REJECTED and control(sim.store)["frozen"] is True
    t_end = sim.t
    q = sim.run(12, lambda st, t: sim.normal(E, t))       # the attack ends
    ev = regime_events(sim.store)
    ret = [x for x in ev if x.extra["state"] == "returned"]
    assert len(ret) == 1 and ret[0].extra["from"] == "rejected"
    assert ret[0].ts - t_end >= 2 * HOUR
    c = control(sim.store)
    assert c["frozen"] is False and int(c.get("version") or 0) == 0
    assert c.get("release") is None                       # held rows were discarded
    assert sim.regime() == MG.NORMAL
    assert ring(sim.store, E, MG.TRUST, q[-1]) > 0.0
    assert ring(sim.store, E, MG.QUARANTINE, q[-1]) == 0.0
    eps = MG.episodes(sim.store, S, E)
    assert eps[-1]["state"] == "rejected" and eps[-1]["end"] == ret[0].ts


def test_label_expected_change_accepts_and_tp_rejects_with_rollback():
    e2 = "10.0.0.2"
    sim = Sim(entities=[E, e2], seed=22)
    sim.run(96, lambda st, t: (sim.normal(E, t), sim.normal(e2, t)))
    sim.run(4, lambda st, t: (sim.shifted(E, t, 7.0, axes=("categorical",)),
                              sim.normal(e2, t)))
    assert sim.regime(E) == MG.SUSPECT
    t0_label = T0 + 90 * DT                                # the analyst's window start
    t_lab = sim.t + DT

    def labelled(st, t):
        sim.shifted(E, t, 7.0, axes=("categorical",))
        sim.normal(e2, t)
        put_model(st, "__org__", "__org__", "model.feedback", _feedback(
            accept={f"{S}|{E}": [{"ts": t, "seq": 0, "t0": t0_label, "t1": t,
                                  "label_id": "lb1", "scope": "entity"}]},
            freeze={f"{S}|{e2}": [{"ts": t, "seq": 1, "t0": t0_label, "t1": t,
                                   "label_id": "lb2", "scope": "entity"}]}))

    sim.tick(labelled)
    c = control(sim.store, E)
    assert c["version"] == 1 and c["rebase_from"] == pytest.approx(t0_label)
    acc = [x for x in regime_events(sim.store, E) if x.extra["state"] == "accepted"]
    assert len(acc) == 1 and acc[0].extra["reason"] == "label" and acc[0].ts == t_lab
    # tp on an entity that looked normal: reject, freeze, roll back to the window
    c2 = control(sim.store, e2)
    assert c2["frozen"] is True and int(c2.get("version") or 0) == 0
    assert c2["rollback_to"] == pytest.approx(t0_label - DT)
    assert states(sim.store, e2) == ["suspect", "rollback", "rejected"]
    assert ring(sim.store, e2, MG.QUARANTINE, t_lab) == 1.0
    # records apply once: nothing new on the next ticks
    sim.run(3, labelled)
    assert len([x for x in regime_events(sim.store, E) if x.extra["state"] == "accepted"]) == 1
    assert states(sim.store, e2).count("rejected") == 1
    # a label freeze holds while B23 says frozen, even when quiet
    sim.run(12, lambda st, t: (sim.normal(E, t), sim.normal(e2, t)))
    assert sim.regime(e2) == MG.REJECTED
    # ... and is lifted by a later expected_change at the same tier
    def lifted(st, t):
        sim.normal(E, t)
        sim.normal(e2, t)
        fb = st.get_model("__org__", "__org__", "model.feedback")
        fb = dict(fb, version=2, accept={f"{S}|{e2}": [{"ts": t, "seq": 0, "t0": t0_label,
                                                       "t1": t, "label_id": "lb3",
                                                       "scope": "entity"}]})
        st.put_model("__org__", "__org__", "model.feedback", fb)
    sim.tick(lifted)
    c2 = control(sim.store, e2)
    assert c2["frozen"] is False and c2["version"] == 1


def test_identity_change_is_held_for_a_label_and_queued_after_14_days():
    sim = Sim(dt=3600.0, seed=23)
    sim.run(48, lambda st, t: sim.normal(E, t, dt=3600.0))
    t_s = sim.t + 3600.0
    sim.run(15 * 24, lambda st, t: sim.shifted(E, t, 5.2, axes=("identity",), dt=3600.0))
    st = states(sim.store)
    assert "accepted" not in st and "rejected" not in st    # prior -3 holds, never decides
    assert sim.regime() == MG.DRIFTING
    m = MG.get(sim.store, S, E)
    assert m["type"] == "identity" and "time" not in m["evidence"]
    assert m["label_queue"]["ts"] - t_s >= 14 * DAY
    lq = MG.label_queue(sim.store, S)
    assert [r["entity"] for r in lq] == [E]
    q = [x for x in regime_events(sim.store) if x.extra.get("label_queue")]
    assert len(q) == 1 and q[0].severity.value == "low"
    # the periodic DRIFTING update (every 24 ticks)
    ups = [x for x in regime_events(sim.store) if x.extra.get("update")]
    assert len(ups) >= 14


def test_new_entity_accepts_through_the_three_day_path():
    sim = Sim(dt=3600.0, seed=24)
    add_obs_tick(sim.store, S, "10.0.0.9", T0 - 30 * DAY, {"l4.flows": 3.0})  # watched system
    add_obs_tick(sim.store, S, E, T0, {"l4.flows": 3.0})    # E first seen now
    sim.run(24, lambda st, t: sim.normal(E, t, dt=3600.0))
    t_s = sim.t + 3600.0
    sim.run(4 * 24, lambda st, t: sim.shifted(E, t, 5.4, axes=("categorical",), dt=3600.0))
    acc = [x for x in regime_events(sim.store) if x.extra["state"] == "accepted"]
    assert len(acc) == 1
    assert acc[0].extra["type"] == "new_entity"
    assert acc[0].extra["p_legit"] < 0.9                  # not by P: by the 3-day path
    assert 3 * DAY <= acc[0].ts - t_s <= 3 * DAY + 3600.0
    assert control(sim.store)["version"] == 1


def test_rollback_is_rate_limited_and_deferred_not_dropped():
    sim = Sim(seed=25)
    sim.run(96, lambda st, t: sim.normal(E, t))
    tau1 = sim.t - 20 * DT

    def cp(onset, sm=sim):
        def f(st, t):
            sm.shifted(E, t, 6.5)
            put_acc(st, E, t, {"cusum": 1})
            put_model(st, S, E, "model.cp", {"run": {"episode": {"onset": onset},
                                                     "alarm": {"cusum": 1}, "dt": DT}})
        return f

    t_first = sim.tick(cp(tau1))
    assert control(sim.store)["rollback_to"] == pytest.approx(tau1 - DT)
    tau2 = tau1 - 10 * DT                                 # B14 refines the onset earlier
    sim.run(6, cp(tau2))
    rbs = [x for x in regime_events(sim.store) if x.extra["state"] == "rollback"]
    assert len(rbs) == 2
    assert rbs[0].ts == t_first and rbs[1].ts - t_first >= HOUR
    assert rbs[1].ts == t_first + 4 * DT                  # the first tick allowed
    assert control(sim.store)["rollback_to"] == pytest.approx(tau2 - DT)
    assert MG.get(sim.store, S, E)["rollbacks"] == 2
    # an onset beyond 7 d is clamped to the depth limit
    sim2 = Sim(seed=26)
    sim2.run(4, lambda st, t: sim2.normal(E, t))
    t = sim2.tick(cp(T0 - 10 * DAY, sim2))
    rb = control(sim2.store)["rollback_to"]
    assert t - rb <= gating.ROLLBACK_MAX_DEPTH_S and t - rb > gating.ROLLBACK_MAX_DEPTH_S - DT - 1


def test_release_on_returned_is_replayed_by_a_gated_learner():
    sim = Sim(seed=27)
    lr = _toy_learner()
    state, gate = lr.init(), gating.GateState()
    phase = {"alarm": False}

    def before(st, t):
        nonlocal state, gate
        if phase["alarm"]:
            sim.shifted(E, t, 5.5)
        else:
            sim.normal(E, t)
        put1(st, E, "feature.active", t, 1.0)
        state, gate = lr.step(st, S, E, state, gate, t, DT)

    sim.run(40, before)
    phase["alarm"] = True
    sim.run(3, before)
    phase["alarm"] = False
    sim.run(14, before)
    ret = [x for x in regime_events(sim.store) if x.extra["state"] == "returned"]
    assert len(ret) == 1
    rel = control(sim.store)["release"]
    assert rel[1] == ret[0].ts
    assert gate.applied.get("release") == pytest.approx(rel)
    # every row that was held is committed now, with its trust_prov
    assert not [r for r in gate.held if r.ts <= rel[1]]
    frontier_at_return = rel[1] - 4 * DT                  # rows learners had processed
    held = [r for r in gate.journal if rel[0] < r.ts <= frontier_at_return]
    assert len(held) >= 8
    for r in held:
        assert r.w_eff == pytest.approx(ring(sim.store, E, MG.TRUST_PROV, r.ts))
    alarm_rows = [r for r in held if r.w_eff == 0.0]
    assert len(alarm_rows) == 3                           # the alarm ticks carry no weight
    # known limitation: the last D rows before the verdict were not held and
    # are committed with their recorded trust (0 while the regime was open)
    tail = [r for r in gate.journal if frontier_at_return < r.ts <= rel[1]]
    assert tail and all(r.w_eff == ring(sim.store, E, MG.TRUST, r.ts) for r in tail)


def test_incident_only_quarantine_releases_when_the_incident_closes():
    sim = Sim(seed=28)
    sim.run(8, lambda st, t: sim.normal(E, t))
    inc_id = sim.store.put_incident(Incident(system=S, entity=E, status="open",
                                             opened=sim.t + DT, last_seen=sim.t + DT))
    ts = sim.run(5, lambda st, t: sim.normal(E, t))
    assert sim.regime() == MG.NORMAL                      # an incident alone is not a regime
    assert all(ring(sim.store, E, MG.QUARANTINE, t) == 1.0 for t in ts)
    assert all(ring(sim.store, E, MG.TRUST, t) == 0.0 for t in ts)
    inc = sim.store.get_incident(inc_id)
    inc.status, inc.close_reason = "closed", "timeout"    # B27 closes it (quiet)
    sim.store.put_incident(inc)
    t = sim.tick(lambda st, t: sim.normal(E, t))
    assert ring(sim.store, E, MG.QUARANTINE, t) == 0.0
    assert control(sim.store)["release"] == [ts[0] - 4 * DT, t]      # from q_floor
    assert states(sim.store) == []


def _vec_groups(store, e, t, levels: Dict[str, float], rng) -> None:
    row = np.full(FEATURE_DIM, np.nan)
    for g, v in levels.items():
        idx = list(GROUPS[g])
        row[idx] = v + 0.05 * rng.standard_normal(len(idx))
    store.add_vec(S, e, "feature.vec", t, row.astype(np.float32), window_s=int(DT))


def test_class_level_acceptance_carries_concordant_shape_members():
    members = ["10.0.4.1", "10.0.4.2", "10.0.4.3", "10.0.4.4"]
    movers = members[:3]
    ck = "class:r1"
    sim = Sim(entities=members, seed=29)
    put_model(sim.store, "__org__", "__org__", "model.class", class_model(members))

    def feed(st, t, moved):
        for e in members:
            if moved and e in movers and MG.version(st, S, e) == 0:
                put1(st, e, "behavior.q_inst", t, 1e-7)
                put1(st, e, "behavior.q_all", t, 1e-7)
                _vec_groups(st, e, t, {"volume": 5.0, "comp": 0.8}, sim.rng)
                put_alarm(st, e, t, ("shape",))
            else:
                put1(st, e, "behavior.q_inst", t, 0.5)
                put1(st, e, "behavior.q_all", t, 0.5)
                _vec_groups(st, e, t, {"volume": 5.0, "comp": 0.8 if (moved and e in movers)
                                       else 0.0}, sim.rng)
        if moved and MG.version(st, S, ck) == 0:
            put_alarm(st, ck, t, ("shape",), sev="low")

    sim.run(96, lambda st, t: feed(st, t, False))
    onset = sim.t + DT
    sim.run(100, lambda st, t: feed(st, t, True))
    cls = [x for x in regime_events(sim.store, ck) if x.extra["state"] == "accepted"]
    assert len(cls) == 1 and cls[0].extra["reason"] == "class"
    assert DAY - DT <= cls[0].ts - onset <= DAY + 2 * DT
    for e in movers:                                      # shape: T_type 7 d on their own
        acc = [x for x in regime_events(sim.store, e) if x.extra["state"] == "accepted"]
        assert len(acc) == 1 and acc[0].extra["reason"] == "class_member", e
        assert acc[0].ts - cls[0].ts <= DT
        assert control(sim.store, e)["accepted_class_change"] == ck
    assert states(sim.store, members[3]) == []


def test_profile_regime_merged_and_profile_version_on_accept():
    sim = Sim(seed=30)
    p = EntityProfile(system=S, entity=E, updated=T0)
    p.extra["regime"] = {"delta_by_feature": {"bytes_up": {"z_mean": 3.0}}}   # B14's key
    sim.store.put_profile(p)
    sim.run(8, lambda st, t: sim.normal(E, t))
    sim.run(2, lambda st, t: sim.shifted(E, t, 7.0))
    reg = sim.store.profile(S, E).extra["regime"]
    assert reg["state"] == "suspect" and reg["delta_by_feature"]["bytes_up"]["z_mean"] == 3.0
    assert reg["branch"] == 1 and reg["type"] == "intensity"

    def label(st, t):
        sim.shifted(E, t, 7.0)
        put_model(st, "__org__", "__org__", "model.feedback", _feedback(
            accept={f"{S}|{E}": [{"ts": t, "seq": 0, "t0": None, "t1": t, "label_id": "l",
                                  "scope": "entity"}]}))
    t = sim.tick(label)
    reg = sim.store.profile(S, E).extra["regime"]
    assert reg["state"] == "accepted" and reg["version"] == 1
    assert reg["delta_by_feature"]["bytes_up"]["z_mean"] == 3.0
    pv = sim.store.profile_versions(S, E)
    assert len(pv) == 1 and pv[0].version == 1 and pv[0].ts == t
    assert pv[0].obj["prev_version"] == 0 and pv[0].obj["rebase_from"] == pytest.approx(
        control(sim.store)["rebase_from"])
    # behavior.regime carries the transition for B27 / B24
    pts = sim.store.derived_tail(S, E, MG.REGIME, 3)
    assert pts[-1].value["state"] == "accepted" and pts[-1].value["version"] == 1
    t = sim.tick(lambda st, t: sim.normal(E, t, 7.0))
    assert sim.regime() == MG.NORMAL
    assert MG.regime_at(sim.store, S, E, t - DT) == MG.ACCEPTED
    assert MG.regime_at(sim.store, S, E, t) == MG.NORMAL
