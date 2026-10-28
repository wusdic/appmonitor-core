"""Round 9 (governor trust rule owner): a borrowed-credential incident
distrusts the (source, credential) PAIR beyond the incident's window; within
the window it counts like any finding.

O-real seed 0 (progressive.md 16.16.9 / 16.16.12 item 15): A2 is 192.168.1.21
logging in to oa as rose on days 17-21. Round 8 lowered trust per source only;
once the incidents were over the address learned rose's logins at full weight.

Now P03's credential finding names the pair (extra 'cred'), B28 publishes the
pairs of a live incident in model.governor['cred_pairs'] for 7 d after its last
live tick, and P03 learns the pair's rows damped (conformity._distrusted)
while the address's other rows keep full weight.

Within the window the trust factor is round 8's (a HIGH finding keeps the
source at 0): trust is per source and tick, read by every learner, and a
borrowed session's later rows do not carry the credential. A first version
credited the source during the window; on pack O seed 2 that released
192.168.1.21's held A2-hour rows at trust 1 and the borrowed session's
approvals displaced GA.oa.approvals#2 from the oa tree (recall@21 .881 ->
.857) - test_a / test_b fail on that version.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import pytest

from app.engines.behavior import governor as GV
from app.engines.behavior.lib import m_governor as MG
from app.engines.behavior.lib import m_ptree as MP
from app.models.schema import BehaviorEvent, Incident, Severity

from test_b28_governor import E, S, Sim, ring
from test_p03_conformity import GA, LOGIN, Fx, workdays

PAIR = "oa|body.kv.username=rose"


# ------------------------------------------------------------------ B28 side
def _cred_finding(sim: Sim, sev: Severity, cred: bool = True,
                  flags: Sequence[str] = ("cross_binding",)) -> Tuple[str, Severity]:
    extra = {"type": "content", "node": 14, "tree_key": "oa", "pattern_id": "oa:0:14:1.0",
             "flags": list(flags)}
    if cred:
        extra["cred"] = {"attr": "body.kv.username", "value": "rose"}
    eid = sim.store.add_event(BehaviorEvent(
        system=S, entity=E, ts=sim.t, kind="pattern_violation", score=0.9, severity=sev,
        axes=["credential"], extra=extra))
    return eid, sev


def _plain_finding(sim: Sim, node: int, sev: Severity) -> Tuple[str, Severity]:
    eid = sim.store.add_event(BehaviorEvent(
        system=S, entity=E, ts=sim.t, kind="pattern_violation", score=0.5, severity=sev,
        extra={"type": "novel", "node": node, "tree_key": "oa", "pattern_id": f"oa:0:{node}:1.0"}))
    return eid, sev


def _incident(sim: Sim, findings, extra: Sequence[dict] = ()) -> str:
    ev: List[dict] = [{"ts": sim.t, "source": "b27", "state": "open"}]
    for eid, sev in findings:
        ev.append({"ts": sim.t, "source": "event", "kind": "pattern_violation", "event_id": eid,
                   "severity": sev.value, "axes": ["credential"]})
    ev.extend(extra)
    top = Severity.HIGH if any(s == Severity.HIGH for _, s in findings) else Severity.MEDIUM
    kinds = ["pattern_violation"] + (["alarm"] if any(x.get("source") == "alarm" for x in extra) else [])
    return sim.store.put_incident(Incident(system=S, entity=E, status="open", opened=sim.t,
                                           last_seen=sim.t, severity=top, kinds=kinds, evidence=ev))


def _run(sim: Sim, n: int) -> List[float]:
    return sim.run(n, lambda st, t: sim.normal(E, t))


def _resolve(sim: Sim, iid: str) -> None:
    inc = sim.store.get_incident(iid)
    inc.status = "resolved"
    inc.last_seen = sim.t
    sim.store.put_incident(inc)


def test_a_a_high_credential_incident_withholds_its_window_and_distrusts_the_pair_after():
    sim = Sim(seed=3)
    _run(sim, 8)
    iid = _incident(sim, [_cred_finding(sim, Severity.MEDIUM), _cred_finding(sim, Severity.HIGH)])
    ts = _run(sim, 6)
    # the window: the source's rows are held and released at trust 0 (round 8)
    assert all(ring(sim.store, E, MG.TRUST, t) == 0.0 for t in ts)
    assert all(ring(sim.store, E, MG.QUARANTINE, t) == 1.0 for t in ts)
    pairs = sim.store.get_model(S, E, MG.MODEL)["cred_pairs"]
    assert set(pairs) == {PAIR} and pairs[PAIR] >= ts[-1] + GV.CRED_PAIR_HOLD_S - 1.0
    # beyond it: the source learns again, the pair stays distrusted for 7 d
    _resolve(sim, iid)
    ts2 = _run(sim, 4)
    assert max(ring(sim.store, E, MG.TRUST, t) for t in ts2) > 0.5
    assert all(ring(sim.store, E, MG.QUARANTINE, t) == 0.0 for t in ts2)
    pairs2 = sim.store.get_model(S, E, MG.MODEL)["cred_pairs"]
    assert pairs2 == pairs                                   # not extended once it is over


def test_b_every_pair_flag_is_published_and_the_factor_is_round_8s():
    for fl in (("cross_binding",), ("foreign_source",), ("concurrent_use", "foreign_source")):
        # HIGH: evidence about the source while live
        sim = Sim(seed=4)
        _run(sim, 8)
        _incident(sim, [_cred_finding(sim, Severity.HIGH, flags=fl)])
        ts = _run(sim, 4)
        assert all(ring(sim.store, E, MG.TRUST, t) == 0.0 for t in ts), fl
        assert set(sim.store.get_model(S, E, MG.MODEL)["cred_pairs"]) == {PAIR}, fl
        # one MEDIUM finding on one pattern: pattern-scoped (round 3), the source keeps learning
        sim = Sim(seed=4)
        _run(sim, 8)
        _incident(sim, [_cred_finding(sim, Severity.MEDIUM, flags=fl)])
        ts = _run(sim, 4)
        assert max(ring(sim.store, E, MG.TRUST, t) for t in ts) > 0.5, fl
        assert set(sim.store.get_model(S, E, MG.MODEL)["cred_pairs"]) == {PAIR}, fl


def test_c_evidence_about_the_source_still_keeps_trust_zero():
    alarm = {"source": "alarm", "path": "cusum", "severity": "low", "families": ["volume"],
             "axes": ["volume"]}
    cases = {   # name: (make, pairs published)
        "no pair": (lambda sim: _incident(sim, [_cred_finding(sim, Severity.HIGH, cred=False)]), False),
        "pair + entity alarm": (lambda sim: _incident(sim, [_cred_finding(sim, Severity.MEDIUM)],
                                                      [dict(alarm, ts=sim.t)]), True),
        "pair + other HIGH finding": (lambda sim: _incident(
            sim, [_cred_finding(sim, Severity.MEDIUM), _plain_finding(sim, 23, Severity.HIGH)]), True),
        "pair + findings on two patterns": (lambda sim: _incident(
            sim, [_cred_finding(sim, Severity.MEDIUM), _plain_finding(sim, 23, Severity.MEDIUM)]), True),
        "a flag that is not about a pair": (lambda sim: _incident(
            sim, [_cred_finding(sim, Severity.HIGH, flags=("unbound_value",))]), False),
    }
    for name, (make, published) in cases.items():
        sim = Sim(seed=5)
        _run(sim, 8)
        make(sim)
        ts = _run(sim, 4)
        assert all(ring(sim.store, E, MG.TRUST, t) == 0.0 for t in ts), name
        got = (sim.store.get_model(S, E, MG.MODEL) or {}).get("cred_pairs") or {}
        assert set(got) == ({PAIR} if published else set()), name


def test_d_credential_pair_helper():
    ex = {"flags": ["cross_binding"], "tree_key": "oa",
          "cred": {"attr": "body.kv.username", "value": "rose"}}
    assert GV.credential_pair(ex) == PAIR
    assert GV.credential_pair(dict(ex, flags=["unbound_value"])) is None
    assert GV.credential_pair({k: v for k, v in ex.items() if k != "cred"}) is None


# ------------------------------------------------------------------ P03 side
def _login_fixture(bound: bool) -> Tuple[Fx, float]:
    fx = Fx()
    users = dict(zip(GA, ["jack", "rose", "mike"]))
    for d in workdays(15):
        for k, ip in enumerate(GA):
            fx.learn("oa", LOGIN, ip, d + (9 * 60 + 3 * k) * 60.0)
    if bound:
        nd = fx.node("oa", LOGIN)
        table = {ip: {"n": 15.0, "top": u, "k": 15.0, "bound": True, "LB": 0.95, "p_viol": 0.02}
                 for ip, u in users.items()}
        fx.put("oa", MP.PBIND, nd.id, pairs={"net.src->body.kv.username": {
            "x": "net.src", "y": "body.kv.username", "dir": "fwd", "fd": {"g3": 0.0, "holds": True},
            "table": table, "bound_values": {u: [ip] for ip, u in users.items()}}})
    return fx, workdays(16)[-1] + 9 * 3600 + 600


def test_e_p03_credential_finding_names_its_pair():
    fx, t = _login_fixture(bound=True)
    fx.score("oa", [(t, "192.168.1.21", {"http.route": LOGIN, "body.kv.username": "rose"})])
    v = [e for e in fx.violations("192.168.1.21") if e.extra["type"] == "content"]
    assert v and v[0].extra["cred"] == {"attr": "body.kv.username", "value": "rose"}
    assert GV.credential_pair(v[0].extra) == PAIR


def test_f_p03_learns_a_distrusted_pairs_rows_damped_and_the_rest_at_full_weight():
    fx, t = _login_fixture(bound=False)        # no binding record: no flag on any login
    _, asg = fx.score("oa", [(t, "192.168.1.21", {"http.route": LOGIN, "body.kv.username": "rose"})])
    assert asg.get("damp", 0) == 1.0
    fx.st.put_model("oa", "192.168.1.21", MG.MODEL, {"regime": MG.NORMAL,
                                                     "cred_pairs": {PAIR: t + 7 * 86400.0}})
    _, asg = fx.score("oa", [(t + 60, "192.168.1.21", {"http.route": LOGIN, "body.kv.username": "rose"})])
    assert asg.get("damp", 0) == pytest.approx(0.1)                     # the pair: damped
    _, asg = fx.score("oa", [(t + 120, "192.168.1.21", {"http.route": LOGIN, "body.kv.username": "jack"})])
    assert asg.get("damp", 0) == 1.0                                    # its own credential
    _, asg = fx.score("oa", [(t + 180, "192.168.1.23", {"http.route": LOGIN, "body.kv.username": "rose"})])
    assert asg.get("damp", 0) == 1.0                                    # rose's own address
    # an expired pair no longer damps
    fx.st.put_model("oa", "192.168.1.21", MG.MODEL, {"regime": MG.NORMAL,
                                                     "cred_pairs": {PAIR: t + 200.0}})
    _, asg = fx.score("oa", [(t + 7200, "192.168.1.21", {"http.route": LOGIN, "body.kv.username": "rose"})])
    assert asg.get("damp", 0) == 1.0
