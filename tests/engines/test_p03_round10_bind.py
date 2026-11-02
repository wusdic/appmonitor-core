"""P03 round 10 (bounds owner, item 5): a binding pair the covering node's own
record says nothing about is checked by the nearest ancestor's record.

O-real seed 0 day 17 (A2): 192.168.1.21 logs in as rose at the login node's
'other' child (the group split's remainder), whose net.src -> username table
has no entry for .21 (its DEV pool does not hold); the route node binds
rose -> 192.168.1.23. The pair was marked 'checked' at the child although the
child returned no verdict, so the route node's record was never read and A2
produced no event."""
from __future__ import annotations

from app.engines.behavior.lib import m_ptree as MP
from app.models.schema import SYSTEM_ENTITY, Severity

from test_p03_conformity import GA, LOGIN
from test_p03_round3 import _young_department_node

PAIR = "net.src->body.kv.username"
REV = "body.kv.username->net.src"
USERS = dict(zip(GA, ["jack", "rose", "mike"]))
DEV = [f"10.20.0.{i}" for i in range(1, 6)]


def _fwd(src_users, holds=True, extra=None):
    table = {ip: {"n": 15.0, "top": u, "k": 15.0, "bound": True, "LB": 0.95, "p_viol": 0.02}
             for ip, u in src_users.items()}
    table.update(extra or {})
    return {"x": "net.src", "y": "body.kv.username", "dir": "fwd", "fd": {"g3": 0.0, "holds": holds},
            "table": table, "bound_values": {u: [ip] for ip, u in src_users.items()} if holds else {}}


def _rev(user_srcs):
    table = {u: {"n": 15.0, "top": ip, "k": 15.0, "bound": True, "LB": 0.95, "p_viol": 0.02}
             for u, ip in user_srcs.items()}
    return {"x": "body.kv.username", "y": "net.src", "dir": "rev", "fd": {"g3": 0.0, "holds": True},
            "table": table, "bound_values": {}}


def _dev_pool_record():
    """The child's record: a DEV pool whose sources log in as many users -
    the dependency does not hold, its entries are neither bound nor sets."""
    ent = {ip: {"n": 6.0, "top": "u%d" % i, "k": 2.0, "bound": False, "LB": 0.3, "p_viol": 0.5}
           for i, ip in enumerate(DEV)}
    return {"x": "net.src", "y": "body.kv.username", "dir": "fwd", "fd": {"g3": 0.4, "holds": False},
            "table": ent, "bound_values": {}}


def _setup(child_pairs, route_pairs):
    fx, tr, route_nd, child, days = _young_department_node()
    fx.put("oa", MP.PBIND, route_nd.id, pairs=route_pairs)
    m = fx.st.get_model("oa", SYSTEM_ENTITY, MP.PBIND)
    m["nodes"][0][child.id] = {"status": "fitted", "pairs": child_pairs}
    fx.st.put_model("oa", SYSTEM_ENTITY, MP.PBIND, m)
    return fx, child, days


def _login(fx, days, user):
    t = days[46] + 8 * 3600 + 40 * 60
    _, asg = fx.score("oa", [(t, "192.168.1.21", {"http.route": LOGIN, "body.kv.username": user})])
    return asg, [e for e in fx.violations("192.168.1.21") if e.extra["type"] == "content"]


def test_pair_without_a_verdict_at_the_child_is_checked_at_the_route_node():
    fx, child, days = _setup({PAIR: _dev_pool_record()}, {PAIR: _fwd(USERS)})
    asg, v = _login(fx, days, "rose")
    assert int(asg.get("conf", 0)) == child.id
    assert v and "cross_binding" in v[0].extra["flags"] and v[0].severity == Severity.MEDIUM


def test_reverse_pair_without_an_entry_at_the_child_falls_back_to_the_ancestor():
    """The same for the reverse record: the child's username -> source table
    has no entry for rose; the route node's closes rose's sources to {.23}."""
    child_rev = _rev({"u%d" % i: ip for i, ip in enumerate(DEV)})
    fx, child, days = _setup({REV: child_rev}, {REV: _rev({u: ip for ip, u in USERS.items()})})
    t = days[46] + 8 * 3600 + 40 * 60
    fx.score("oa", [(t - 300.0, "192.168.1.23", {"http.route": LOGIN, "body.kv.username": "rose"})])
    _, asg = fx.score("oa", [(t, "192.168.1.21", {"http.route": LOGIN, "body.kv.username": "rose"})])
    v = [e for e in fx.violations("192.168.1.21") if e.extra["type"] == "content"]
    assert int(asg.get("conf", 0)) == child.id
    assert v and {"foreign_source", "concurrent_use"} <= set(v[0].extra["flags"])
    assert v[0].severity in (Severity.MEDIUM, Severity.HIGH)


def test_child_verdict_still_takes_precedence():
    """When the child's own record binds the source, its verdict stands and the
    ancestor's record is not consulted again (.21 -> jack: a clean login)."""
    fx, child, days = _setup({PAIR: _fwd({"192.168.1.21": "jack"})}, {PAIR: _fwd(USERS)})
    asg, v = _login(fx, days, "jack")
    assert int(asg.get("conf", 0)) == child.id
    assert not v
