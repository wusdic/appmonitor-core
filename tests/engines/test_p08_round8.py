"""P08 round 8: a family tree key keeps its members' binding bookkeeping.

P12 adopts the seed member's tree, registry and fitted model.pbind for a new
family (R9: oa-r2 joins oa as 'fam:1' on O-real day 11) but not P08's private
state (probe, value histories, bound values). The family started empty: its
first fit re-screened an empty probe and dropped every pair, and a source that
logs in once a workday needed n_bind days again before a binding was stated,
so 192.168.1.21 -> jack was not bound when A2 used rose's credential from it
on day 17 (O-real A2 missed on seeds 0-2, PG3 bindings 0/3). Fails on the
round-7 code."""
from __future__ import annotations

from app.core.store import MetricStore
from app.engines.behavior.binding import STATE, BindingEngine
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pfd as FD
from app.models.schema import ORG, SYSTEM_ENTITY

DAY = 86400.0
T0 = 1_788_220_800.0
PK = "net.src->body.kv.username"


def _hist(rows):
    h = FD.ValueHistory()
    for x, y, d in rows:
        h.observe(x, y, T0 + d * DAY + 9 * 3600, 700000 + d, True)
    return h


def _member_state(rows, offered):
    probe = FD.ProbeReservoir()
    probe.offered = offered
    return {"probe": probe, "hist": {PK: _hist(rows)}, "last": {}, "bound": {}, "rebound": {},
            "track": {}, "screened_t": T0 + 9 * DAY, "screened_at": offered}


def test_a_new_family_starts_from_its_members_binding_state():
    store = MetricStore()
    oa = _member_state([("192.168.1.21", "jack", d) for d in range(9)]
                       + [("192.168.1.23", "rose", d) for d in range(9)], offered=500)
    oa["last"] = {"oa": T0 + 10 * DAY}
    oa["bound"] = {(0, 14, PK): {"192.168.1.21": "jack"}}
    r2 = _member_state([("192.168.1.21", "jack", 10), ("192.168.2.51", "tom", 10)], offered=40)
    r2["last"] = {"oa-r2": T0 + 10.5 * DAY}
    store.put_model("oa", SYSTEM_ENTITY, STATE, oa)
    store.put_model("oa-r2", SYSTEM_ENTITY, STATE, r2)
    store.put_model(ORG, ORG, MP.SYSFAM, {"version": 1, "member": {"oa": "fam:1", "oa-r2": "fam:1"},
                                          "families": {"fam:1": ["oa", "oa-r2"]}})
    st = BindingEngine._state(store, "fam:1")
    h = st["hist"][PK]
    # the seed's history (9 normal days of jack) plus the replica's day 10
    rec = h.get("192.168.1.21")["v"]["jack"]
    assert rec[2] == 10 and len(rec[3]) == FD.HIST_DAYS and rec[3][-1] == 700010
    assert h.get("192.168.1.23")["v"]["rose"][2] == 9
    assert h.get("192.168.2.51") is not None
    # the seed's screening and bound values carry over: no re-screen of an empty probe
    assert st["probe"].offered == 500 and st["screened_t"] == T0 + 9 * DAY
    assert st["bound"] == {(0, 14, PK): {"192.168.1.21": "jack"}}
    assert st["last"] == {"oa": T0 + 10 * DAY, "oa-r2": T0 + 10.5 * DAY}
    assert FD.history_support(h.get("192.168.1.21"), None) == FD.history_support(
        _hist([("192.168.1.21", "jack", d) for d in range(11)]).get("192.168.1.21"), None)
    # the members' own states are not aliased by the family's
    assert st["hist"][PK] is not oa["hist"][PK]
    assert store.get_model("fam:1", SYSTEM_ENTITY, STATE) is st


def test_a_plain_system_or_a_family_without_member_state_starts_empty():
    store = MetricStore()
    assert not BindingEngine._state(store, "oa")["hist"]
    store.put_model(ORG, ORG, MP.SYSFAM, {"version": 1, "member": {"a": "fam:2", "b": "fam:2"},
                                          "families": {"fam:2": ["a", "b"]}})
    assert not BindingEngine._state(store, "fam:2")["hist"]


def test_value_history_merge():
    a = _hist([("x", "jack", d) for d in range(3)])
    b = _hist([("x", "jack", d) for d in range(2, 5)] + [("x", "rose", 6), ("y", "tom", 1)])
    a.merge(b)
    ja = a.get("x")["v"]["jack"]
    assert ja[2] == 6 and ja[3] == [700000 + d for d in range(5)]
    assert ja[0] == T0 + 9 * 3600 and ja[1] == T0 + 4 * DAY + 9 * 3600
    assert a.get("x")["v"]["rose"][2] == 1 and a.get("y")["v"]["tom"][2] == 1
    # the source's latest values last: rose (day 6, from b) is the most recent
    assert list(a.get("x")["recent"])[-1] == "rose"
    assert a.version(["x"]) == 3 + 4
