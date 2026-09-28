"""B27 IncidentEngine: docs/lib3/engines.md B27 unit tests (a)-(d) plus edges.

The engine is driven in isolation: each test plays B25 (behavior.alarm,
q_inst, e_day, p_family), B24 (behavior.p), B26 (behavior.risk), B28
(behavior.regime, regime events), the discrete producers (store.events),
analysts (store.add_label) and B23 (model.feedback) for a tick, then runs
B27 through tests/helpers.run_engine (strict).
"""
from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pytest

from helpers import DT, T0, make_store, put_model, run_engine

from app.engines.behavior import incident as I
from app.engines.behavior.incident import IncidentEngine
from app.engines.behavior.lib import m_feedback
from app.engines.behavior.lib.detectors import DETECTOR_INDEX, N_DETECTORS
from app.models.schema import (BehaviorEvent, DerivedMetric, Label, MetricKind, Severity,
                               SignatureMatch)

S = "sys"
E = "10.0.0.1"
NAN = float("nan")


# ------------------------------------------------------------------ fixtures
def put_dict(store, e: str, name: str, ts: float, v: Dict, dt: float = DT) -> None:
    store.add_derived(DerivedMetric(name=name, value=dict(v), ts=ts, system=S, entity=e,
                                    window_s=int(dt), kind=MetricKind.CATEGORICAL))


def put_alarm(store, e: str, ts: float, sev: str = "medium", axes: Sequence[str] = ("volume",),
              e_day: float = 1e-3, dt: float = DT, **kw) -> None:
    a = {"path": "single_tick", "severity": sev, "axes": list(axes), "families": ["intensity"],
         "e_day": e_day, "e_day_path": e_day, "acc": []}
    a.update(kw)
    put_dict(store, e, "behavior.alarm", ts, a, dt)


def put1(store, e: str, name: str, ts: float, v: float, dt: float = DT) -> None:
    store.add_vec(S, e, name, ts, np.asarray([v], dtype=np.float32), window_s=int(dt))


def put_p(store, e: str, ts: float, pvals: Dict[str, float], dt: float = DT) -> None:
    row = np.full(N_DETECTORS, np.nan, dtype=np.float32)
    for d, p in pvals.items():
        row[DETECTOR_INDEX[d]] = p
    store.add_vec(S, e, "behavior.p", ts, row, window_s=int(dt))


def finding(store, e: str, ts: float, kind: str = "rare_access",
            sev: Severity = Severity.MEDIUM, **kw) -> str:
    return store.add_event(BehaviorEvent(system=S, entity=e, ts=ts, kind=kind, score=0.8,
                                         severity=sev, **kw))


def inc_events(store, entity: Optional[str] = None) -> List[BehaviorEvent]:
    evs = store.events(system=S, entity=entity, kinds=("incident",), limit=100000)
    return sorted(evs, key=lambda e: (e.ts, e.id))


def notifs(store, entity: Optional[str] = None) -> List[BehaviorEvent]:
    return [e for e in inc_events(store, entity) if e.extra["state"] in ("open", "escalate")]


def states(store, entity: Optional[str] = None) -> List[str]:
    return [e.extra["state"] for e in inc_events(store, entity)]


class Rig:
    def __init__(self, entities: Iterable[str] = (E,), dt: float = DT,
                 config: Optional[Dict] = None) -> None:
        self.store = make_store()
        self.eng = IncidentEngine()
        self.dt = dt
        self.t = T0 - dt
        self.config = config
        for e in entities:
            self.store.register_entity(S, e)

    def tick(self, before=None, after=None, dt: Optional[float] = None,
             training: bool = False) -> float:
        dt = self.dt if dt is None else dt
        self.t += dt
        if before is not None:
            before(self.store, self.t)
        run_engine(self.eng, self.store, self.t, training=training, dt=dt, config=self.config)
        if after is not None:
            after(self.store, self.t)
        return self.t


def class_model(members: Sequence[str], rid: str = "r1") -> Dict:
    return {"assign": {f"{S}|{m}": {"role": rid, "prob": 0.9, "static": [], "pool": None}
                       for m in members},
            "roles": {rid: {"name": "workers", "members": [f"{S}|{m}" for m in members]}},
            "version": 1}


# ================================================================ spec tests
def test_a_40_alarm_ticks_one_incident_at_most_3_notifications():
    rig = Rig()
    sevs = ["low"] * 3 + ["medium", "low"] * 10 + ["medium"] * 17

    for sev in sevs:
        rig.tick(lambda st, t, sev=sev: put_alarm(st, E, t, sev=sev, e_day=1e-3,
                                                   axes=("volume",)))
    incs = rig.store.incidents(system=S)
    assert len(incs) == 1
    inc = incs[0]
    assert inc.status == "open" and inc.severity == Severity.MEDIUM
    assert inc.axes == ["volume"] and "alarm" in inc.kinds
    n = notifs(rig.store)
    assert 1 <= len(n) <= 3
    assert [e.extra["state"] for e in n] == ["open", "escalate"]
    assert all(e.incident_id == inc.id and e.extra["incident_id"] == inc.id for e in n)
    # evidence is throttled: new info or hourly, not one entry per tick
    alarms = [x for x in inc.evidence if x.get("source") == "alarm"]
    assert len(alarms) <= 14


def test_b_common_mode_class_parent_and_silent_children():
    members = [f"10.0.1.{i}" for i in range(8)]
    rig = Rig(members)
    put_model(rig.store, "__org__", "__org__", "model.class", class_model(members))
    alarming = members[:5]

    def before(st, t):
        for m in alarming:
            put_alarm(st, m, t, sev="medium", axes=("volume",))

    for _ in range(3):
        rig.tick(before)
    incs = rig.store.incidents(system=S)
    parents = [i for i in incs if i.entity == "class:r1"]
    assert len(parents) == 1
    parent = parents[0]
    assert "coherent_shift" in parent.kinds and parent.severity == Severity.LOW
    children = [i for i in incs if i.entity in alarming]
    assert len(children) == 5
    assert all(c.parent_id == parent.id and c.status == "suppressed" for c in children)
    assert all(any(x.get("state") == "suppressed_common" for x in c.evidence)
               for c in children)
    # 0 notifying entity incidents; the class parent notifies once
    assert all(not notifs(rig.store, m) for m in members)
    assert [e.extra["state"] for e in notifs(rig.store, "class:r1")] == ["open"]
    # members stay out of the parent's entities (B26 keeps them at 0.25)
    assert not set(parent.entities) & set(members)


def test_b_member_with_other_axes_is_not_parented_and_minority_is_not_common():
    members = [f"10.0.1.{i}" for i in range(8)]
    rig = Rig(members)
    put_model(rig.store, "__org__", "__org__", "model.class", class_model(members))

    def before(st, t):
        for m in members[:4]:
            put_alarm(st, m, t, axes=("volume",))
        put_alarm(st, members[4], t, axes=("volume", "exfil"))

    rig.tick(before)
    incs = {i.entity: i for i in rig.store.incidents(system=S)}
    parent = incs["class:r1"]
    assert all(incs[m].parent_id == parent.id for m in members[:4])
    assert incs[members[4]].parent_id == "" and incs[members[4]].status == "open"
    assert len(notifs(rig.store, members[4])) == 1

    # 3 of 8 (< 50 %): ordinary root incidents that notify
    rig2 = Rig(members)
    put_model(rig2.store, "__org__", "__org__", "model.class", class_model(members))
    rig2.tick(lambda st, t: [put_alarm(st, m, t, axes=("volume",)) for m in members[:3]])
    incs2 = rig2.store.incidents(system=S)
    assert len(incs2) == 3 and all(i.parent_id == "" and i.status == "open" for i in incs2)
    assert sum(len(notifs(rig2.store, m)) for m in members[:3]) == 3


def test_b_child_promoted_when_other_axis_appears():
    members = [f"10.0.1.{i}" for i in range(6)]
    rig = Rig(members)
    put_model(rig.store, "__org__", "__org__", "model.class", class_model(members))
    rig.tick(lambda st, t: [put_alarm(st, m, t, axes=("volume",)) for m in members[:4]])
    child = rig.store.incidents(system=S, entity=members[0])[0]
    assert child.parent_id and child.status == "suppressed"
    rig.tick(lambda st, t: put_alarm(st, members[0], t, axes=("categorical",)))
    child = rig.store.get_incident(child.id)
    assert child.parent_id == "" and child.status == "open"
    assert [e.extra["state"] for e in notifs(rig.store, members[0])] == ["open"]


def test_c_same_new_sni_within_1h_is_one_campaign():
    ents = ["10.0.2.1", "10.0.2.2", "10.0.2.3", "10.0.2.4", "10.0.2.5"]
    rig = Rig(ents)
    sni = {"dim": "sni", "value": "cdn.evil-x.com"}
    plan = {0: [(ents[0], sni)], 1: [(ents[1], sni)],
            3: [(ents[2], sni), (ents[3], {"dim": "sni", "value": "static.benign.org"})],
            12: [(ents[4], sni)]}                        # 2 h 15 m after the last member
    for i in range(14):
        rig.tick(lambda st, t, i=i: [finding(st, e, t, kind="first_seen", extra=dict(x),
                                             axes=["categorical"])
                                     for e, x in plan.get(i, [])])
    incs = {i.entity: i for i in rig.store.incidents(system=S)}
    assert len(incs) == 5
    camp = {incs[e].campaign_id for e in ents[:3]}
    assert len(camp) == 1 and "" not in camp
    assert incs[ents[3]].campaign_id == "" and incs[ents[4]].campaign_id == ""
    # the discrete findings now point at their incidents
    for e in ents:
        ev = rig.store.events(S, e, kinds=("first_seen",))[0]
        assert ev.incident_id == incs[e].id


def test_d_closes_after_attack_with_risk_high_when_accumulators_decay():
    rig = Rig()
    n_attack = 12
    t_end = None
    decay = 3                                            # accumulator loud 3 ticks after

    def before(st, t, i):
        put1(st, E, "behavior.risk", t, 72.0)
        if i < n_attack:
            put_alarm(st, E, t, sev="high", axes=("volume", "exfil"), e_day=1e-5)
            put1(st, E, "behavior.q_inst", t, 1e-6)
            put_p(st, E, t, {"marg_int": 1e-6, "budget_exfil": 1e-6})
        else:
            put1(st, E, "behavior.q_inst", t, 0.5)
            p_acc = 1e-6 if i < n_attack + decay else 0.6
            put_p(st, E, t, {"marg_int": 0.4, "budget_exfil": p_acc, "cusum": 0.5})

    closed_at = None
    for i in range(40):
        t = rig.tick(lambda st, t, i=i: before(st, t, i))
        if i == n_attack - 1:
            t_end = t
        inc = rig.store.incidents(system=S)[0]
        if closed_at is None and inc.status == "closed":
            closed_at = t
    inc = rig.store.incidents(system=S)
    assert len(inc) == 1                                 # decaying risk never reopens it
    inc = inc[0]
    assert inc.status == "closed" and inc.close_reason == "timeout"
    assert closed_at is not None
    assert closed_at - t_end <= max(8 * DT, 7200.0) + 1e-6
    assert closed_at - t_end >= max(8 * DT, 7200.0) - 1e-6
    assert float(rig.store.vec_at(S, E, "behavior.risk", closed_at)[0]) >= 60.0
    assert states(rig.store)[-1] == "close"


def test_d_idle_entity_stale_q_inst_rows_do_not_block_the_quiet_close():
    """The quiet rule reads the entity's last q_inst rows; an entity that went
    idle right after alarming (a weekend, a holiday) kept those alarming rows
    as its latest for days, so its incident never closed and a threat days
    later only escalated it (pack B T18, round 2). Rows older than the quiet
    window are no longer current evidence."""
    rig = Rig()

    def alarming(st, t):
        put_alarm(st, E, t)
        put1(st, E, "behavior.q_inst", t, 1e-6)

    for _ in range(2):
        rig.tick(alarming)
    for _ in range(12):                                  # idle: nothing written for 3 h
        rig.tick()
    inc = rig.store.incidents(system=S)[0]
    assert inc.status == "closed" and inc.close_reason == "timeout"


def test_d_accumulator_above_quarter_h_holds_the_incident_open():
    rig = Rig()

    def before(st, t, i):
        put1(st, E, "behavior.q_inst", t, 0.5)
        if i == 0:
            put_alarm(st, E, t)
        loud = i < 20
        # budget ARL at 900 s: p ~ 1e-3 is well above h/4
        put_p(st, E, t, {"budget_vol": 1e-3 if loud else 0.7})

    closed = []
    for i in range(30):
        t = rig.tick(lambda st, t, i=i: before(st, t, i))
        if rig.store.incidents(system=S)[0].status == "closed":
            closed.append((i, t))
    assert closed and closed[0][0] == 20
    # an alarm flag alone also blocks
    rig2 = Rig()
    for i in range(12):
        rig2.tick(lambda st, t, i=i: (put_alarm(st, E, t) if i == 0 else None,
                                      put_dict(st, E, "behavior.acc_alarm", t, {"beacon": 1})))
    assert rig2.store.incidents(system=S)[0].status == "open"


def test_acc_level_from_p_matches_quarter_h():
    from app.engines.behavior.lib.detectors import arl_days
    arl = arl_days("budget_vol") * 86400.0 / DT
    p = arl ** -0.25
    assert I.acc_level_from_p(p, "budget_vol", DT) == pytest.approx(0.25)
    assert math.isnan(I.acc_level_from_p(NAN, "budget_vol", DT))
    assert I.acc_level_from_p(1.0, "cusum", DT) == 0.0


# ================================================================ lifecycle
def test_regime_returned_closes_even_with_high_risk():
    rig = Rig()

    def before(st, t, i):
        put1(st, E, "behavior.risk", t, 80.0)
        put_dict(st, E, "behavior.acc_alarm", t, {"cusum": 1})     # (d) cannot fire
        if i < 3:
            put_alarm(st, E, t)

    def after(st, t, i):                                  # B28 runs after B27
        state = "suspect" if i < 5 else "returned"
        put_dict(st, E, "behavior.regime", t, {"state": state.upper()})

    for i in range(8):
        rig.tick(lambda st, t, i=i: before(st, t, i), lambda st, t, i=i: after(st, t, i))
    inc = rig.store.incidents(system=S)[0]
    assert inc.status == "closed" and inc.close_reason == "returned"
    assert states(rig.store) == ["open", "close"]


def test_regime_accepted_event_and_persistent_returned_before_open():
    rig = Rig()
    # a regime that was already RETURNED before the incident must not close it
    for i in range(3):
        rig.tick(after=lambda st, t: put_dict(st, E, "behavior.regime", t,
                                              {"state": "returned"}))
    rig.tick(lambda st, t: put_alarm(st, E, t),
             lambda st, t: put_dict(st, E, "behavior.regime", t, {"state": "returned"}))
    for i in range(2):
        rig.tick(lambda st, t: put_dict(st, E, "behavior.acc_alarm", t, {"cusum": 1}),
                 lambda st, t: put_dict(st, E, "behavior.regime", t, {"state": "returned"}))
    inc = rig.store.incidents(system=S)[0]
    assert inc.status == "open"
    rig.tick(after=lambda st, t: st.add_event(BehaviorEvent(
        system=S, entity=E, ts=t, kind="regime", score=0.0, extra={"state": "accepted"})))
    rig.tick()
    inc = rig.store.get_incident(inc.id)
    assert inc.status == "closed" and inc.close_reason == "accepted"


def test_external_close_by_governor_is_notified():
    rig = Rig()
    rig.tick(lambda st, t: put_alarm(st, E, t))
    inc = rig.store.incidents(system=S)[0]
    inc.status, inc.close_reason = "closed", "accepted"      # B28 closes on ACCEPT
    rig.store.put_incident(inc)
    rig.tick()
    assert states(rig.store) == ["open", "close"]
    assert inc_events(rig.store)[-1].extra["close_reason"] == "accepted"


def test_label_closes_and_unsure_does_not():
    rig = Rig(["10.0.0.1", "10.0.0.2"])
    eid = {}

    def before(st, t):
        put_alarm(st, "10.0.0.1", t)
        eid["x"] = finding(st, "10.0.0.2", t, sev=Severity.HIGH)

    rig.tick(before)
    a = rig.store.incidents(system=S, entity="10.0.0.1")[0]
    b = rig.store.incidents(system=S, entity="10.0.0.2")[0]
    rig.store.add_label(Label(system=S, entity=E, target_type="incident", target_id=a.id,
                              verdict="unsure", ts=rig.t))
    rig.tick(lambda st, t: put_dict(st, E, "behavior.acc_alarm", t, {"cusum": 1}))
    assert a.status == "open"
    rig.store.add_label(Label(system=S, entity=E, target_type="incident", target_id=a.id,
                              verdict="fp", ts=rig.t))
    rig.store.add_label(Label(system=S, entity="10.0.0.2", target_type="event",
                              target_id=eid["x"], verdict="tp", ts=rig.t))
    rig.tick()
    assert a.status == "closed" and a.close_reason == "labelled"
    assert b.status == "closed" and b.close_reason == "labelled"
    assert rig.store.get_event(eid["x"]).status == "closed"
    # a labelled incident is final: a new alarm opens a new id
    rig.tick(lambda st, t: put_alarm(st, E, t))
    ids = {i.id for i in rig.store.incidents(system=S, entity=E)}
    assert len(ids) == 2


def test_reopen_within_24h_reuses_id_and_after_24h_does_not():
    rig = Rig(dt=3600.0)
    rig.tick(lambda st, t: put_alarm(st, E, t, dt=3600.0))
    for _ in range(8):                                   # quiet max(8 ticks, 2 h) = 8 h
        rig.tick()
    inc = rig.store.incidents(system=S)[0]
    assert inc.status == "closed" and inc.close_reason == "timeout"
    first_id = inc.id
    for _ in range(5):
        rig.tick()
    rig.tick(lambda st, t: put_alarm(st, E, t, dt=3600.0))
    incs = rig.store.incidents(system=S)
    assert len(incs) == 1 and incs[0].id == first_id and incs[0].status == "open"
    assert incs[0].close_reason is None
    opens = [e for e in inc_events(rig.store) if e.extra["state"] == "open"]
    assert len(opens) == 2 and opens[1].extra.get("reopened") is True
    for _ in range(8 + 25):
        rig.tick()
    rig.tick(lambda st, t: put_alarm(st, E, t, dt=3600.0))
    assert len({i.id for i in rig.store.incidents(system=S)}) == 2


def test_14_day_incident_goes_to_label_queue_not_closed():
    rig = Rig(dt=3600.0)
    rig.tick(lambda st, t: put_alarm(st, E, t, dt=3600.0))
    for _ in range(14 * 24 + 3):
        rig.tick(lambda st, t: put_dict(st, E, "behavior.acc_alarm", t, {"jsd": 1}, 3600.0))
    inc = rig.store.incidents(system=S)[0]
    assert inc.status == "open" and inc.close_reason is None
    held = [x for x in inc.evidence if x.get("state") == "label_queue"]
    assert len(held) == 1
    upd = [e for e in inc_events(rig.store) if e.extra["state"] == "update"]
    assert len(upd) == 1 and upd[0].extra["reason"] == "label_queue"
    assert upd[0].ts - inc.opened > 14 * 86400.0


# ================================================================ openers / joins
def test_findings_medium_open_low_only_join():
    rig = Rig(["10.0.0.1", "10.0.0.2"])
    low_id = {}
    rig.tick(lambda st, t: low_id.setdefault("a", finding(st, "10.0.0.2", t,
                                                           sev=Severity.LOW)))
    assert not rig.store.incidents(system=S)
    assert rig.store.get_event(low_id["a"]).incident_id == ""
    rig.tick(lambda st, t: finding(st, E, t, kind="beacon", sev=Severity.MEDIUM))
    rig.tick(lambda st, t: low_id.setdefault("b", finding(st, E, t, kind="client_change",
                                                           sev=Severity.LOW)))
    incs = rig.store.incidents(system=S)
    assert len(incs) == 1 and incs[0].entity == E
    assert set(incs[0].kinds) == {"beacon", "client_change"}
    assert "c2" in incs[0].axes
    assert rig.store.get_event(low_id["b"]).incident_id == incs[0].id
    assert incs[0].e_day_min == pytest.approx(3e-3)       # MEDIUM-equivalent e_day


def test_risk_opens_only_with_two_ticks_and_fresh_family_evidence():
    dt = DT
    rig = Rig(["10.0.0.1", "10.0.0.2", "10.0.0.3"])
    p_hit = 0.05 * dt / 86400.0                            # e_day 0.05 <= 0.1

    def before(st, t, i):
        # .1: family hit + risk >= 30 twice -> opens at the 2nd risk tick
        if i == 0:
            put_dict(st, "10.0.0.1", "behavior.p_family", t, {"shape": p_hit, "peer": 0.4})
        if i >= 1:
            put1(st, "10.0.0.1", "behavior.risk", t, 35.0)
        # .2: risk high but no family ever at e_day <= 0.1
        put_dict(st, "10.0.0.2", "behavior.p_family", t, {"shape": 0.3})
        put1(st, "10.0.0.2", "behavior.risk", t, 45.0)
        # .3: family hit but risk >= 30 on single, non-consecutive ticks
        put_dict(st, "10.0.0.3", "behavior.p_family", t, {"shape": p_hit})
        put1(st, "10.0.0.3", "behavior.risk", t, 40.0 if i % 2 else 10.0)

    opened = {}
    for i in range(6):
        t = rig.tick(lambda st, t, i=i: before(st, t, i))
        for inc in rig.store.incidents(system=S):
            opened.setdefault(inc.entity, (i, inc))
    assert set(opened) == {"10.0.0.1"}
    i, inc = opened["10.0.0.1"]
    assert i == 2 and "risk" in inc.kinds and inc.severity == Severity.LOW
    assert "shape" in inc.axes and inc.e_day_min == pytest.approx(0.05, rel=1e-6)


def test_alias_and_actor_join_within_gap():
    a, b, c = "10.0.3.1", "10.0.3.2", "10.0.3.3"
    rig = Rig([a, b, c])
    put_model(rig.store, S, "__system__", "model.link",
              {"links": [{"from": a, "to": b, "ts": T0}], "actors": [[b, c]], "version": 1})
    rig.tick(lambda st, t: put_alarm(st, a, t))
    rig.tick(lambda st, t: put_alarm(st, b, t, axes=("categorical",)))
    incs = rig.store.incidents(system=S)
    assert len(incs) == 1 and set(incs[0].entities) == {a, b}
    assert set(incs[0].axes) == {"volume", "categorical"}
    assert [e.extra["state"] for e in notifs(rig.store)] == ["open", "escalate"]
    # beyond max(4 ticks, 1 h) an actor member opens its own incident
    for _ in range(5):
        rig.tick(lambda st, t: put_dict(st, a, "behavior.acc_alarm", t, {"cusum": 1}))
    rig.tick(lambda st, t: put_alarm(st, c, t))
    assert len(rig.store.incidents(system=S)) == 2


def test_lib4_match_joins_as_evidence_but_does_not_open():
    rig = Rig(["10.0.0.1", "10.0.0.2"])

    def before(st, t):
        st.add_match(SignatureMatch(system=S, entity="10.0.0.2", ts=t - DT, signature_id="x",
                                    label="scan", category="recon", confidence=0.9,
                                    severity=Severity.HIGH))
        put_alarm(st, E, t)
        st.add_match(SignatureMatch(system=S, entity=E, ts=t - DT, signature_id="y",
                                    label="exfil", category="exfil", confidence=0.9,
                                    severity=Severity.HIGH))

    rig.tick(before)
    incs = rig.store.incidents(system=S)
    assert len(incs) == 1 and incs[0].entity == E
    assert any(x.get("source") == "lib4" and x["signature_id"] == "y" for x in incs[0].evidence)


def test_repeating_lib4_match_does_not_flood_the_evidence():
    """A poller's routine info match every tick used to add an evidence entry
    per tick; on a week-long incident 477 of the 512 kept entries were lib-4
    info matches and the alarm entries were evicted. Below MEDIUM: one entry
    per signature per opening; >= MEDIUM: at most hourly."""
    rig = Rig()

    def before(st, t):
        put_alarm(st, E, t)
        st.add_match(SignatureMatch(system=S, entity=E, ts=t - DT, signature_id="health",
                                    label="hc", category="maintenance", confidence=1.0,
                                    severity=Severity.INFO))
        st.add_match(SignatureMatch(system=S, entity=E, ts=t - DT, signature_id="c2",
                                    label="c2", category="beacon", confidence=1.0,
                                    severity=Severity.HIGH))

    n = int(3 * 3600 / DT)
    for _ in range(n):
        rig.tick(before)
    ev = rig.store.incidents(system=S)[0].evidence
    info = [x for x in ev if x.get("source") == "lib4" and x["signature_id"] == "health"]
    high = [x for x in ev if x.get("source") == "lib4" and x["signature_id"] == "c2"]
    assert len(info) == 1
    assert 3 <= len(high) <= 4 < n


def test_habitual_lib4_match_does_not_keep_an_incident_open():
    """An integration host matched the MEDIUM signature 'high_error_backend'
    on almost every tick; each match restarted the quiet clock, so an FP
    incident stayed open for days and a later threat only escalated it
    (pack B T4). A habitual match (lib/stages: >= 4 matched ticks, the first
    >= 24 h earlier; same rule as B26) joins as evidence but lets the
    incident close; a new medium signature still restarts the clock."""
    def match(sig):
        def f(st, t):
            st.add_match(SignatureMatch(system=S, entity=E, ts=t - DT, signature_id=sig,
                                        label=sig, category="health", confidence=0.8,
                                        severity=Severity.MEDIUM))
        return f
    rig = Rig()
    for _ in range(int(30 * 3600 / DT)):                  # warm-up: the habit forms
        rig.tick(match("hb"), training=True)
    rig.tick(lambda st, t: (match("hb")(st, t), put_alarm(st, E, t)))
    for _ in range(12):                                  # 3 h of habitual matches only
        rig.tick(match("hb"))
    inc = rig.store.incidents(system=S)[0]
    assert inc.status == "closed" and inc.close_reason == "timeout"
    assert any(x.get("source") == "lib4" and x["signature_id"] == "hb" for x in inc.evidence)
    rig2 = Rig()
    rig2.tick(lambda st, t: (match("new")(st, t), put_alarm(st, E, t)))
    for _ in range(12):                                  # a NEW medium activity is not routine
        rig2.tick(match("new"))
    assert rig2.store.incidents(system=S)[0].status == "open"


def test_class_alarm_opens_class_incident():
    rig = Rig()
    rig.tick(lambda st, t: put_alarm(st, "class:r9", t, sev="low", axes=("categorical",)))
    incs = rig.store.incidents(system=S)
    assert len(incs) == 1 and incs[0].entity == "class:r9" and incs[0].entities == []
    assert notifs(rig.store, "class:r9")


# ================================================================ feedback / budget
def _policy(axes=("volume",), level=1e-6) -> Dict:
    toks, gate = m_feedback.build_tokens(["alarm"], list(axes), {}, [])
    return {"id": "pol1", "label_id": "lb1", "verdict": "fp", "scope": "pattern", "system": S,
            "entity": None, "class_key": None, "tokens": sorted(toks), "gate": sorted(gate),
            "e_day_ref": level * 10, "level": level, "created": T0, "expires": None}


def test_feedback_suppression_and_escape():
    rig = Rig()
    put_model(rig.store, "__org__", "__org__", "model.feedback",
              {"version": 1, "policies": [_policy()]})
    for _ in range(3):
        rig.tick(lambda st, t: put_alarm(st, E, t, e_day=1e-3))
    inc = rig.store.incidents(system=S)[0]
    assert inc.status == "suppressed" and not notifs(rig.store)
    # an exfil axis the analyst never saw escapes the policy
    rig.tick(lambda st, t: put_alarm(st, E, t, axes=("volume", "exfil"), e_day=1e-3))
    assert inc.status == "open"
    assert [e.extra["state"] for e in notifs(rig.store)] == ["open"]


def test_token_buckets_queue_and_release_by_risk():
    ents = ["10.0.4.1", "10.0.4.2", "10.0.4.3", "10.0.4.4"]
    rig = Rig(ents, config={"alert_budget": {"entity_per_hour": 1, "system_per_day": 2}})
    risks = {ents[0]: 20.0, ents[1]: 90.0, ents[2]: 50.0, ents[3]: 70.0}

    def before(st, t):
        for e, r in risks.items():
            put_alarm(st, e, t)
            put1(st, e, "behavior.risk", t, r)

    rig.tick(before)
    sent = [e.entity for e in notifs(rig.store)]
    assert sent == [ents[1], ents[3]]                     # the two highest risks
    # the day budget refills 2 tokens per 24 h: ~12 h per token
    for _ in range(13 * 4):
        rig.tick(lambda st, t: [put_dict(st, e, "behavior.acc_alarm", t, {"cusum": 1})
                                for e in ents] + [put1(st, e, "behavior.risk", t, r)
                                                  for e, r in risks.items()])
    sent = [e.entity for e in notifs(rig.store)]
    assert sent[:3] == [ents[1], ents[3], ents[2]]
    late = notifs(rig.store, ents[2])[0]
    assert late.extra["queued_at"] == T0 and late.ts > T0


def test_entity_bucket_delays_escalation():
    rig = Rig(config={"alert_budget": {"entity_per_hour": 1, "system_per_day": 20}})
    rig.tick(lambda st, t: put_alarm(st, E, t, sev="low"))
    rig.tick(lambda st, t: put_alarm(st, E, t, sev="high", axes=("volume", "exfil")))
    assert [e.extra["state"] for e in notifs(rig.store)] == ["open"]
    for _ in range(4):
        rig.tick(lambda st, t: put_dict(st, E, "behavior.acc_alarm", t, {"cusum": 1}))
    n = notifs(rig.store)
    assert [e.extra["state"] for e in n] == ["open", "escalate"]
    assert n[1].ts - n[0].ts >= 3600.0 and n[1].extra["queued_at"] == T0 + DT


# ================================================================ edges
def test_empty_store_and_silent_entity():
    st = make_store()
    eng = IncidentEngine()
    assert run_engine(eng, st, T0) == 0
    st.register_entity(S, E)
    for i in range(5):
        assert run_engine(eng, st, T0 + i * DT) == 0
    assert not st.incidents() and not st.events()


def test_nan_inputs_are_neutral():
    rig = Rig()

    def before(st, t, i):
        put1(st, E, "behavior.risk", t, NAN)
        put1(st, E, "behavior.q_inst", t, NAN)
        put1(st, E, "behavior.e_day", t, NAN)
        put_p(st, E, t, {})
        if i == 0:
            put_alarm(st, E, t, e_day=NAN, e_day_path=NAN)

    for i in range(4):
        rig.tick(lambda st, t, i=i: before(st, t, i))
    inc = rig.store.incidents(system=S)[0]
    assert inc.status == "open" and inc.e_day_min is None
    assert inc.risk == 0.0
    for i in range(6):                                    # NaN q_inst / p are neutral
        rig.tick(lambda st, t: before(st, t, 1))
    assert inc.status == "closed" and inc.close_reason == "timeout"
    ev = inc_events(rig.store)[0]
    assert ev.e_day is None and ev.extra["risk"] is None


def test_training_emits_nothing_and_opens_nothing():
    rig = Rig()
    for _ in range(10):
        rig.tick(lambda st, t: (put_alarm(st, E, t), finding(st, E, t, sev=Severity.HIGH),
                                put1(st, E, "behavior.risk", t, 90.0)), training=True)
    assert not rig.store.incidents() and not inc_events(rig.store)
    # the first live tick does not replay warm-up findings
    rig.tick()
    assert not rig.store.incidents()


def test_cadence_switch_900_to_60_keeps_wall_clock_windows():
    rig = Rig()
    rig.tick(lambda st, t: put_alarm(st, E, t))
    for _ in range(2):
        rig.tick()
    inc = rig.store.incidents(system=S)[0]
    assert inc.status == "open"
    t_switch = rig.t
    closed_at = None
    for _ in range(200):
        t = rig.tick(dt=60.0)
        if inc.status == "closed":
            closed_at = t
            break
    # 2 h of quiet since the alarm (8 ticks at 60 s would be only 8 min)
    assert closed_at is not None and closed_at - T0 == pytest.approx(7200.0)
    assert closed_at > t_switch
    # join gap at 60 s is still 1 h (alias) and reopening still reuses the id
    rig.tick(lambda st, t: put_alarm(st, E, t, dt=60.0), dt=60.0)
    assert len(rig.store.incidents(system=S)) == 1 and inc.status == "open"


def test_token_bucket_and_campaign_primitives():
    b = I.TokenBucket(3, 3600.0, 0.0)
    for _ in range(3):
        assert b.level(0.0) >= 1.0
        b.take()
    assert b.level(0.0) < 1.0
    assert b.level(1200.0) == pytest.approx(1.0)
    assert b.level(10 ** 6) == pytest.approx(3.0)
    fz = frozenset
    items = [("a", 0.0, fz({"a:categorical", "n:sni=x"})),
             ("b", 1800.0, fz({"a:categorical", "n:sni=x"})),
             ("c", 3000.0, fz({"a:categorical", "n:sni=x", "f:bytes_up+"})),
             ("d", 100.0, fz({"a:volume"})),
             ("e", 200.0, fz({"a:volume"}))]                  # bare axis: never a campaign
    assert I.campaign_groups(items) == [["a", "b", "c"]]


def test_perf_40_entities():
    ents = [f"10.1.0.{i}" for i in range(40)]
    rig = Rig(ents)
    rng = np.random.default_rng(3)

    def before(st, t):
        for e in ents:
            put1(st, e, "behavior.q_inst", t, float(rng.uniform()))
            put1(st, e, "behavior.risk", t, float(rng.uniform(0, 25)))
            if rng.uniform() < 0.02:
                put_alarm(st, e, t, sev="low", axes=(str(rng.choice(["volume", "shape"])),))

    for _ in range(20):
        rig.tick(before)
    spent = 0.0
    n = 100
    for _ in range(n):
        rig.t += DT
        before(rig.store, rig.t)
        t0 = time.perf_counter()
        run_engine(rig.eng, rig.store, rig.t)
        spent += time.perf_counter() - t0
    per_tick_ms = spent / n * 1000.0
    assert rig.store.incidents(system=S)
    assert per_tick_ms < 10.0, per_tick_ms                # spec: < 1 ms; generous for CI


def test_decaying_risk_of_a_closed_incident_does_not_reopen_it_on_a_weak_hit():
    """Evaluator round 3: after a quiet close the key's risk decays over days,
    and a family at e_day <= 0.1 is an ordinary null event (~1 per entity-day
    over the families); old risk + any later weak hit reopened the incident
    within hours (pack A: 58 reopenings of 23 control incidents). Only risk
    the last incident did not cover opens; new risk on top still does."""
    rig = Rig()
    p_hit = 0.05 * DT / 86400.0                            # e_day 0.05 <= 0.1

    def before(st, t, i):
        if i < 3:                                          # the episode: 3 alarm ticks
            put_alarm(st, E, t, sev="medium", axes=("volume",), e_day=1e-4)
            put1(st, E, "behavior.q_inst", t, 1e-5)
            put1(st, E, "behavior.risk", t, 60.0)
            return
        put1(st, E, "behavior.q_inst", t, 0.5)
        put_p(st, E, t, {"marg_int": 0.4, "cusum": 0.5})
        if i < 30:
            put1(st, E, "behavior.risk", t, 58.0 - 0.1 * (i - 3))   # old risk decaying
        else:
            put1(st, E, "behavior.risk", t, 90.0)                   # new evidence on top
        if i in (20, 32):
            put_dict(st, E, "behavior.p_family", t, {"shape": p_hit})

    reopened_at, was_closed = None, False
    for i in range(36):
        rig.tick(lambda st, t, i=i: before(st, t, i))
        inc = rig.store.incidents(system=S)[0]
        if i == 18:
            assert inc.status == "closed"
        was_closed = was_closed or inc.status == "closed"
        if reopened_at is None and was_closed and inc.status == "open":
            reopened_at = i
    assert len(rig.store.incidents(system=S)) == 1        # the same id when it reopens
    # the weak hit at tick 20 on the old risk did not reopen (the v2 rule did,
    # at tick 20); the new risk from tick 30 on did
    assert reopened_at == 30
