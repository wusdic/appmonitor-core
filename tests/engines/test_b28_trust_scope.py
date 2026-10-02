"""B28 trust rule (round 3, groups_views owner): a live incident lowers a
source's trust by its EVIDENCE and SCOPE (the learned patterns it concerns),
not by its existence.

Pack O: 192.168.1.21's approval route was renamed (D3, a legitimate change);
P03 reported the new route as a MEDIUM `new_action` on day 14, the incident
stayed open until its timeout six days later, and trust 0 for those days meant
none of the address's logins, documents or mail - which no finding concerned -
was ever learned (the held rows were released at their recorded trust 0)."""
from __future__ import annotations

from typing import List, Sequence, Tuple

from app.engines.behavior.lib import m_governor as MG
from app.models.schema import BehaviorEvent, Incident, Severity

from test_b28_governor import DT, E, S, Sim, ring


def _finding(sim: Sim, node: int, sev: Severity = Severity.MEDIUM, kind: str = "pattern_violation",
             key: str = "oa") -> Tuple[str, Severity]:
    eid = sim.store.add_event(BehaviorEvent(
        system=S, entity=E, ts=sim.t, kind=kind, score=0.5, severity=sev,
        extra={"type": "novel", "node": node, "tree_key": key, "pattern_id": f"{key}:0:{node}:1.0"}))
    return eid, sev


def _incident(sim: Sim, findings: Sequence[Tuple[str, Severity]], extra: Sequence[dict] = ()) -> str:
    ev: List[dict] = [{"ts": sim.t, "source": "b27", "state": "open"}]
    for eid, sev in findings:
        ev.append({"ts": sim.t, "source": "event", "kind": "pattern_violation", "event_id": eid,
                   "severity": sev.value if hasattr(sev, "value") else str(sev), "axes": ["categorical"]})
    ev.extend(extra)
    sevs = [s for _, s in findings] or [Severity.LOW]
    top = Severity.HIGH if Severity.HIGH in sevs else Severity.MEDIUM
    kinds = ["pattern_violation"] + (["alarm"] if any(x.get("source") == "alarm" for x in extra) else [])
    return sim.store.put_incident(Incident(system=S, entity=E, status="open", opened=sim.t,
                                           last_seen=sim.t, severity=top, kinds=kinds, evidence=ev))


def _run_open(sim: Sim, n: int = 6) -> List[float]:
    return sim.run(n, lambda st, t: sim.normal(E, t))


def test_a_one_pattern_finding_does_not_stop_learning_from_the_source():
    sim = Sim(seed=3)
    _run_open(sim, 8)
    _incident(sim, [_finding(sim, 17)])
    ts = _run_open(sim)
    assert sim.regime() == MG.NORMAL
    assert all(ring(sim.store, E, MG.QUARANTINE, t) == 1.0 for t in ts)     # still held
    tr = [ring(sim.store, E, MG.TRUST, t) for t in ts]
    prov = [ring(sim.store, E, MG.TRUST_PROV, t) for t in ts]
    assert all(t == p for t, p in zip(tr, prov)) and max(tr) > 0.5         # ... at its own trust


def test_b_the_same_pattern_on_several_days_is_still_one_pattern():
    sim = Sim(seed=4)
    _run_open(sim, 8)
    alarm = {"ts": sim.t, "source": "alarm", "path": "evidence_cusum", "severity": "low",
             "families": ["conformity"], "axes": ["categorical"]}
    _incident(sim, [_finding(sim, 17), _finding(sim, 17)], [alarm])
    ts = _run_open(sim)
    assert max(ring(sim.store, E, MG.TRUST, t) for t in ts) > 0.5


def test_c_evidence_about_the_source_keeps_trust_zero():
    cases = {
        "two patterns": lambda sim: _incident(sim, [_finding(sim, 17), _finding(sim, 23)]),
        "high": lambda sim: _incident(sim, [_finding(sim, 17, Severity.HIGH)]),
        "entity alarm": lambda sim: _incident(sim, [_finding(sim, 17)], [
            {"ts": sim.t, "source": "alarm", "path": "cusum", "severity": "low",
             "families": ["volume"], "axes": ["volume"]}]),
        "no evidence": lambda sim: sim.store.put_incident(Incident(
            system=S, entity=E, status="open", opened=sim.t, last_seen=sim.t)),
    }
    for name, make in cases.items():
        sim = Sim(seed=5)
        _run_open(sim, 8)
        make(sim)
        ts = _run_open(sim, 4)
        assert all(ring(sim.store, E, MG.TRUST, t) == 0.0 for t in ts), name
        assert all(ring(sim.store, E, MG.QUARANTINE, t) == 1.0 for t in ts), name
