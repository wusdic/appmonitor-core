"""B02 PeerGroupEngine: role lineage (split, merge, transition, retirement),
eligibility (quarantine, open incident) and org-wide role ids across systems.

Uses the World builder of test_b02_peer_group.py."""
from __future__ import annotations

from helpers import set_trust

from app.engines.behavior.lib import m_class
from app.engines.behavior.peer_group import PeerGroupEngine
from app.models.schema import Incident
from test_b02_peer_group import DT, NOW, ROLES, S, World, events, ip, role_of
from test_b02_peer_group_edges import H6, fit, preset


def test_class_split_event_and_inherited_id():
    w = World(outlier=False)
    both = [ip(0, i) for i in range(4)] + [ip(1, i) for i in range(4)]
    preset(w, {"rX": both, "rB": [ip(2, i) for i in range(4)]})
    mc = fit(w, PeerGroupEngine())
    ev = events(w.store, "class_split")
    assert len(ev) == 1 and ev[0].extra["role"] == "rX" and len(ev[0].extra["into"]) == 2
    roles = {role_of(mc, S, e) for e in both}
    assert len(roles) == 2 and "rX" in roles
    new = (roles - {"rX"}).pop()
    assert any(x["kind"] == "split" and x["from"] == "rX" for x in mc["roles"][new]["lineage"])
    assert {role_of(mc, S, ip(2, i)) for i in range(4)} == {"rB"}
    assert not events(w.store, "class_transition")


def test_class_merge_event():
    w = World(outlier=False)
    preset(w, {"rA": [ip(0, 0), ip(0, 1)], "rC": [ip(0, 2), ip(0, 3)]})
    mc = fit(w, PeerGroupEngine())
    ev = events(w.store, "class_merge")
    assert len(ev) == 1 and sorted(ev[0].extra["from"]) == ["rA", "rC"]
    rid = ev[0].extra["role"]
    assert {role_of(mc, S, ip(0, i)) for i in range(4)} == {rid}
    assert ev[0].entity == f"class:{rid}"


def test_class_transition_needs_three_runs():
    w = World(outlier=False)
    eng = PeerGroupEngine()
    mc = fit(w, eng)
    r_int, r_api = role_of(mc, S, ip(0, 0)), role_of(mc, S, ip(1, 0))
    mover = ip(0, 0)
    w.specs[mover] = ROLES[1]                 # quietly repurposed as an API client
    w.build([mover])
    t = NOW
    for run in range(1, 4):
        t += H6
        mc = fit(w, eng, t)
        tr = events(w.store, "class_transition", mover)
        if run < 3:
            assert not tr and role_of(mc, S, mover) == r_int
            assert mc["assign"][f"{S}|{mover}"]["pend"] == [r_api, run]
    assert len(tr) == 1 and tr[0].extra["from"] == r_int and tr[0].extra["to"] == r_api
    assert tr[0].extra["prob"] >= 0.7 and role_of(mc, S, mover) == r_api
    assert f"{S}|{mover}" in mc["roles"][r_api]["members"]


def test_quarantine_incident_and_retirement():
    w = World(outlier=False)
    eng = PeerGroupEngine()
    mc = fit(w, eng)
    r_int = role_of(mc, S, ip(0, 0))
    # an open incident keeps one member out of the clustering (it keeps its role)
    w.store.put_incident(Incident(system=S, entity=ip(1, 0), status="open", opened=NOW,
                                  last_seen=NOW))
    # the whole interactive role is quarantined
    for i in range(4):
        set_trust(w.store, S, ip(0, i), [NOW + DT], 0.0, quarantine=1.0)
    t = NOW
    for run in range(1, 4):
        t += H6
        mc = fit(w, eng, t)
        if run < 3:
            assert mc["roles"][r_int]["absent"] == run
            assert role_of(mc, S, ip(0, 0)) == r_int          # held, not re-clustered
    assert r_int not in mc["roles"] and role_of(mc, S, ip(0, 0)) is None
    r_api = role_of(mc, S, ip(1, 1))
    assert role_of(mc, S, ip(1, 0)) == r_api
    assert f"{S}|{ip(1, 0)}" not in mc["roles"][r_api]["cluster"]
    assert f"{S}|{ip(1, 0)}" in mc["roles"][r_api]["members"]


def test_cross_system_roles_are_global():
    w = World(outlier=False)
    other = World(seed=8, system="oa", outlier=False)
    for (s, e) in [(other.system, x) for x in other.truth]:
        for name in ("model.baseline", "model.rhythm", "model.vocab", "model.client"):
            w.store.put_model(s, e, name, other.store.get_model(s, e, name))
        w.store.register_entity(s, e)
    mc = fit(w, PeerGroupEngine())
    for r in range(3):
        assert role_of(mc, S, ip(r, 0)) == role_of(mc, "oa", ip(r, 0))
    rid = role_of(mc, "oa", ip(1, 0))
    assert m_class.members(w.store, "oa", rid) == [ip(1, i) for i in range(4)]
    assert w.store.profile("oa", f"class:{rid}").extra["peer_group"]["n_members_org"] == 8
