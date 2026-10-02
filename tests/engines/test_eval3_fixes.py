"""Evaluator round-3 fixes in the engines (docs/lib3/progressive.md §16.11).
Each test fails on the code the four round-3 owners delivered."""
from __future__ import annotations

import pytest

from helpers import make_store

from app.engines.behavior import views as VW
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.lib import ptree as PT
from app.engines.behavior.lib import pwindows as PW
from app.models.schema import ORG, SYSTEM_ENTITY

from test_p14_views import CFG, DAY, T0, _keys


class _Res:
    """A minute reservoir: items() -> ((daytype, minute, src), weight, t)."""

    def __init__(self, rows):
        self.rows = rows

    def items(self):
        return list(self.rows)

    def __len__(self):
        return len(self.rows)


def test_a_part_with_its_own_windows_is_not_more_confident_than_its_node(monkeypatch):
    """A group's part that states its own arrival windows also states the
    node's content constraints; its confidence cannot exceed the node's
    held-out hold rate (P04 p_hold). The views stated the min of the parts'
    NOMINAL coverages (0.85-0.97 on pack O's mail parts against a hold rate of
    ~0.5): the most over-confident statements of PG2."""
    ga = ["192.168.1.21", "192.168.1.23", "10.168.7.121"]
    fin = ["192.168.2.10", "192.168.2.11"]
    m = PT.PTreeModel("oa")
    tr = m.tree(EV.KIND_TXN, T0, create=True)
    sp = tr.split(tr.root, "http.route", 0, [["GET oa.corp /docs"]], T0)
    node, root = tr.nodes[sp.children[0]], tr.nodes[tr.root]
    rows = []
    t = T0
    for d in range(21):
        day0 = T0 + d * DAY
        for k, (ip, g) in enumerate([(x, "G1") for x in ga] + [(x, "G2") for x in fin]):
            t = day0 + (10 * 60 + 7 * k) * 60.0
            for nd in (root, node):
                nd.update_core(t, 1.0, 1.0, _keys(ip, g), ip, 0, (t + 8 * 3600) % DAY / 60.0,
                               int((t + 8 * 3600) // DAY))
            rows.append(((0, (t + 8 * 3600) % DAY / 60.0, ip), 1.0, t))
    for nd in (root, node):
        nd.state = "confirmed"
    real = PW.part_when

    class _When:
        res = _Res(rows)

    monkeypatch.setattr(VW.PW, "part_when", lambda when, members, tz=0.0: real(_When, members, tz))
    orig = PN.Node.p_hold
    monkeypatch.setattr(PN.Node, "p_hold", lambda self, _t: 0.31 if self is node else orig(self, _t))
    m.t_last = t
    st = make_store()
    st.put_model("oa", SYSTEM_ENTITY, MP.PTREE, m, version=1)
    st.put_model(ORG, ORG, MP.WHO_GROUPS, {
        "groups": {"G1": {"id": "G1", "name": "综合部", "members": ga},
                   "G2": {"id": "G2", "name": "财务部", "members": fin}},
        "ip2g": dict({ip: "G1" for ip in ga}, **{ip: "G2" for ip in fin}),
        "mode": {"oa": {"mode": "ip"}}})
    v = VW.system_view(st, "oa", CFG, t + 3600.0)
    parts = [s for s in v["statements"] if (s["evidence"].get("who") or {}).get("part_of")]
    assert len(parts) == 2
    # each part states its own windows (its members' arrivals, not the node's)
    wins = [tuple(map(tuple, (s["evidence"].get("when") or {}).get("workday") or [])) for s in parts]
    assert all(wins) and len(set(wins)) == 2, wins
    own = parts
    for s in own:
        assert s["confidence"] == pytest.approx(0.31), s["confidence"]
    # a weak workflow edge (dependency strength 0.1) is no coverage of the part:
    # with p_hold 0.9 the part states min(0.9, its windows' coverage)
    monkeypatch.setattr(PN.Node, "p_hold", lambda self, _t: 0.9 if self is node else orig(self, _t))
    monkeypatch.setattr(VW.PR, "workflow_block", lambda edges: ([{"from": "a", "to": "b"}], "流程：a → b", "workflow: a -> b", 0.1))
    v = VW.system_view(st, "oa", CFG, t + 3600.0)
    parts = [s for s in v["statements"] if (s["evidence"].get("who") or {}).get("part_of")]
    for s in parts:
        cov = float(s["evidence"]["when"]["coverage"])
        assert s["confidence"] == pytest.approx(min(0.9, cov)), (s["confidence"], cov)


def test_held_out_test_checks_only_the_bindings_the_statement_states():
    """P04's held-out test must check what the statement says. P14 states a
    bound pair only for an identifier-like payload (>= 8 values system-wide)
    or when the bound sources hold different values; a pair binding every
    source to the same client version is a constant of the action and is not
    stated, but P04 checked it (ptree open issue, PG2 calibration)."""
    from app.engines.behavior import pattern_tree as PTE
    from app.engines.behavior.lib import pfd as FD
    nd = PN.Node(1, None, 1, 0, (), 0.0)
    tab_same = {ip: {"bound": True, "top": "5.2.1", "LB": 0.95} for ip in ("192.168.1.21", "192.168.1.23")}
    tab_users = {"192.168.1.21": {"bound": True, "top": "jack", "LB": 0.95},
                 "192.168.1.23": {"bound": True, "top": "rose", "LB": 0.95}}
    fit = {MP.PBIND: {"pairs": {
        "net.src->hdr.x-client-ver": {"x": "net.src", "y": "hdr.x-client-ver", "fd": {"holds": True},
                                      "table": tab_same},
        "net.src->body.kv.username": {"x": "net.src", "y": "body.kv.username", "fd": {"holds": True},
                                      "table": tab_users}}}}
    cards = {"hdr.x-client-ver": 2.0, "body.kv.username": 3.0}
    cons = PTE._hold_constraints(nd, 0.0, {}, fit, lambda rec: cards[rec["y"]])
    binds = sorted(k for k in cons if k.startswith("bind:"))
    assert binds == ["bind:body.kv.username:192.168.1.21", "bind:body.kv.username:192.168.1.23"], binds
    # the views keep the same pairs
    assert not FD.binding_stated(fit[MP.PBIND]["pairs"]["net.src->hdr.x-client-ver"], 2.0)
    assert FD.binding_stated(fit[MP.PBIND]["pairs"]["net.src->body.kv.username"], 3.0)


def test_department_part_of_a_prefix_node_states_its_own_members_and_values():
    """The 销售部 part of a node whose context is an address context
    (net.src in 192.168.2.0/24 + 192.168.3.0/24: the 财务部+销售部 login
    node) lists every member whose own signature holds the action and lies in
    that context - not only the node's heavy hitters - and states the node's
    closed user-name set restricted to its members' bound values (pack O seed
    0, day 14: the part listed 5 of 20 members and the node's 23 names)."""
    from app.engines.behavior import who_groups as WG
    from app.engines.behavior import conformity as CF
    from app.models.schema import ORG as _ORG
    fin = ["192.168.2.10", "192.168.2.11", "192.168.2.12"]
    sales = [f"192.168.3.{20 + i}" for i in range(12)]
    dev = "10.50.0.7"                                   # signature holds the action, outside the context
    users = dict(zip(fin, ["lucy", "tom", "kate"]), **{ip: f"s{i:02d}" for i, ip in enumerate(sales)})
    route = "POST oa.corp /login"
    m = PT.PTreeModel("oa")
    tr = m.tree(EV.KIND_TXN, T0, create=True)
    sp = tr.split(tr.root, "http.route", 0, [[route]], T0)
    node, root = tr.nodes[sp.children[0]], tr.nodes[tr.root]
    node.ctx = tuple(node.ctx) + (("net.src", 1, ("192.168.2.0/24", "192.168.3.0/24"), False),)
    grp = dict({ip: "G6" for ip in fin}, **{ip: "G11" for ip in sales}, **{dev: "G5"})
    ws = WG.WGState()
    t = T0
    for d in range(10):
        day0 = T0 + d * DAY
        k = 0
        for ip, n in [(ip, 1) for ip in sales] + [(ip, 4) for ip in fin] + [(dev, 1)]:
            for _ in range(n):
                t = day0 + (9 * 60 + k) * 60.0
                k += 1
                if ip != dev:
                    for nd in (root, node):
                        nd.update_core(t, 1.0, 1.0, _keys(ip, grp[ip]), ip, 0, (t + 8 * 3600) % DAY / 60.0,
                                       int((t + 8 * 3600) // DAY))
                ws.sigs.add(ip, f"oa|{route}", t, 1.0, 1.0, int((t + 8 * 3600) // DAY))
    for nd in (root, node):
        nd.state = "confirmed"
    m.t_last = t
    seen = {str(ip) for ip, *_ in node.who.levels[0].items(t)}
    assert not set(sales) <= seen                      # precondition: heavy hitters lost most of 销售部
    names = sorted(users.values())
    st = make_store()
    st.put_model("oa", SYSTEM_ENTITY, MP.PTREE, m, version=1)
    st.put_model("oa", SYSTEM_ENTITY, MP.PGRAMMAR, {"nodes": {int(EV.KIND_TXN): {node.id: {"status": "fitted", "attrs": {
        "body.kv.username": {"kind": "text", "grammar": "[a-z0-9]{2,4}", "c_g": 1.0, "U_s": 0.0,
                             "closed": names, "U": 0.01}}}}}})
    st.put_model("oa", SYSTEM_ENTITY, MP.PBIND, {"nodes": {int(EV.KIND_TXN): {node.id: {"pairs": {
        "net.src->body.kv.username": {"x": "net.src", "y": "body.kv.username", "fd": {"holds": True},
                                      "table": {ip: {"bound": True, "top": u, "LB": 0.97}
                                                for ip, u in users.items()}}}}}}})
    st.put_model(_ORG, _ORG, CF.WG_STATE, ws)
    st.put_model(_ORG, _ORG, MP.WHO_GROUPS, {
        "groups": {"G6": {"id": "G6", "name": "财务部", "dept": "财务部", "members": fin},
                   "G11": {"id": "G11", "name": "销售部", "dept": "销售部", "members": sales},
                   "G5": {"id": "G5", "name": "研发", "dept": "研发", "members": [dev]}},
        "ip2g": grp, "mode": {"oa": {"mode": "ip"}}})
    v = VW.system_view(st, "oa", CFG, t + 3600.0)
    parts = [s for s in v["statements"] if (s["evidence"].get("who") or {}).get("part_of")]
    sp_ = [s for s in parts if set(s["evidence"]["who"]["members"]) & set(sales)]
    assert len(sp_) == 1
    assert set(sp_[0]["evidence"]["who"]["members"]) == set(sales)
    assert sorted(sp_[0]["evidence"]["content"]["body.kv.username"]["closed"]) == sorted(users[ip] for ip in sales)
    fp = [s for s in parts if set(s["evidence"]["who"]["members"]) & set(fin)]
    assert sorted(fp[0]["evidence"]["content"]["body.kv.username"]["closed"]) == ["kate", "lucy", "tom"]
    assert not any(dev in s["evidence"]["who"]["members"] for s in parts)
    # the node's own statement keeps the whole closed set
    whole = [s for s in v["statements"] if not (s["evidence"].get("who") or {}).get("part_of")
             and s["evidence"].get("route", "").endswith("/login")]
    assert sorted(whole[0]["evidence"]["content"]["body.kv.username"]["closed"]) == names


def test_a_constant_pair_does_not_measure_the_binding_arm():
    """P12 measures P08's arm from its judged pair records. A pair binding every
    source to one value (net.src -> body.fmt = form) is judged with gain 0 by
    construction; it said 'bindings are worth 0' on pack O's finance system on
    day 10 and P12 switched P08 off while the three users' user-name pair
    (3-4 logins each, n_bind = 5) was still gathering evidence - finance's
    bindings were then missing at day 14 (PG1 bindings 0.67, seed 1)."""
    from types import SimpleNamespace as NS
    from app.engines.behavior.system_profile import bindings_pending, fitted_gain
    node = NS(parent=None, mass_at=lambda t: 100.0)
    leaf = NS(parent=0, mass_at=lambda t: 10.0)
    ptm = NS(kinds={0: NS(root=0, nodes={0: node, 1: leaf})})
    fin = {"192.168.2.10": "lucy", "192.168.2.11": "tom", "192.168.2.12": "kate"}
    const = {"gain": 0.0, "fd": {"judged": 2}, "table": {ip: {"top": "form", "n": 9.0} for ip in fin}}
    user = {"gain": 1.58, "fd": {"judged": 0}, "table": {ip: {"top": u, "n": 3.0} for ip, u in fin.items()}}
    m = {"nodes": {0: {1: {"pairs": {"net.src->body.fmt": const, "net.src->body.kv.username": user}}}}}
    assert fitted_gain(m, ptm, 0.0) is None                 # nothing that could bind was judged
    assert bindings_pending(m)
    # one-off visitors (n = 1 each) are nothing to wait for
    portal = {"gain": 0.0, "fd": {"judged": 0},
              "table": {f"10.60.0.{i}": {"top": f"u{i}", "n": 1.0} for i in range(20)}}
    assert not bindings_pending({"nodes": {0: {1: {"pairs": {"p": portal}}}}})
    # a judged pair that binds different values is a measurement as before
    done = dict(user, fd={"judged": 3}, gain=1.5)
    assert fitted_gain({"nodes": {0: {1: {"pairs": {"a": const, "b": done}}}}}, ptm, 0.0) == 0.15
