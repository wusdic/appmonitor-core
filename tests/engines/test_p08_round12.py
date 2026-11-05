"""P08, round 12 (views / groups / bindings owner): the pair-node budget."""
from __future__ import annotations

from types import SimpleNamespace as NS

from app.engines.behavior import binding as BD


def _node(nid, mass, targets, pairs=()):
    return NS(id=nid, mass_at=lambda t, m=mass: m, targets=set(targets), is_exc=False,
              pairs={p: object() for p in pairs})


def _specs():
    # the screen's rank order: a high-ranked pair on many nodes, then the
    # username pairs (the reverse 'username -> net.src' carried for its
    # stated entries, appended last)
    return [{"x": "client.stack", "y": "http.ua", "dir": "fwd"},
            {"x": "net.src", "y": "body.kv.username", "dir": "fwd"},
            {"x": "body.kv.username", "y": "net.src", "dir": "rev", "carried": True}]


def _tree(n_ua):
    nodes = {i: _node(i, 1000.0 - i, ["http.ua"]) for i in range(n_ua)}
    nodes[100] = _node(100, 50.0, ["body.kv.username"], [("net.src", "body.kv.username")])
    # 综合部's login node: the lightest login node, holding both username sketches
    nodes[101] = _node(101, 10.0, ["body.kv.username"],
                       [("net.src", "body.kv.username"), ("body.kv.username", "net.src")])
    return NS(nodes=nodes, root=0)


def test_the_budget_keeps_stated_and_existing_sketches_before_new_ones():
    """Pack O seed 0 (round 11): the budget (32 pair-nodes per tree) went pair
    by pair in the screen's order, so on a day the higher-ranked pairs held
    more nodes the carried reverse pair lost its sketch at 综合部's login node
    p:oa:0:17 (the lightest); P04 dropped the sketch, and on day 21 it held one
    row - the statement lost 'username=jack 只来自 192.168.1.21' and three more
    reverse clauses. A pair-node whose binding the views state, then one whose
    sketch exists, gets the budget before a new one."""
    tree = _tree(8)
    stated = {(0, "body.kv.username->net.src"): [101]}
    out = BD.BindingEngine._allocate(0, tree, _specs(), stated, 0.0, budget=4)[0]
    assert out[101] == [["net.src", "body.kv.username"], ["body.kv.username", "net.src"]]
    assert out[100] == [["net.src", "body.kv.username"]]
    assert sum(len(v) for v in out.values()) == 4
    # without a stated entry, the existing sketches still come first
    out = BD.BindingEngine._allocate(0, tree, _specs(), {}, 0.0, budget=3)[0]
    assert ["body.kv.username", "net.src"] in out.get(101, [])
    assert ["net.src", "body.kv.username"] in out.get(100, [])


def test_the_budget_is_the_rank_order_when_it_does_not_bind():
    """Below the budget nothing changes: every pair's nodes, in rank order."""
    tree = _tree(3)
    out = BD.BindingEngine._allocate(0, tree, _specs(), {}, 0.0, budget=32)[0]
    assert out[0] == out[1] == out[2] == [["client.stack", "http.ua"]]
    assert out[100] == [["net.src", "body.kv.username"], ["body.kv.username", "net.src"]]
    assert out[101] == [["net.src", "body.kv.username"], ["body.kv.username", "net.src"]]


DAY = 86400.0
T0 = 1_788_220_800.0          # a Monday 00:00 UTC
FAM = [("192.168.1.21", "jack"), ("192.168.1.23", "rose"), ("10.168.7.121", "mike"),
       ("192.168.2.10", "lucy"), ("192.168.2.11", "tom"), ("192.168.2.12", "kate")]


def _family_login():
    """O-real seed 2, the family login node on day 12: a young node sketch
    (two logins per source) under eight days of tree-level history; the
    shared terminal 192.168.1.40 used by jack2 (5 days) and rose2 (3 days),
    both from its first day."""
    from app.engines.behavior.lib import pfd as FD
    from app.engines.behavior.lib import pnode as PN
    ps, h = PN.PairSketch(), FD.ValueHistory()
    for d in range(8):
        t = T0 + d * DAY + 9 * 3600
        for i, (x, y) in enumerate(FAM):
            h.observe(x, y, t + i, 700000 + d, True)
            if d >= 6:
                ps.update(x, y, t + i, 1.0, 1.0)
        y40 = "jack2" if d in (0, 2, 3, 5, 7) else "rose2"
        h.observe("192.168.1.40", y40, t + 50, 700000 + d, True)
        if d >= 6:
            ps.update("192.168.1.40", y40, t + 50, 1.0, 1.0)
    return ps, h, T0 + 8 * DAY


def test_a_shared_terminal_is_not_pooled_as_its_top_user():
    """O-real seed 2 (round 11, days 12-16): pooled on its days, the shared
    terminal 192.168.1.40 counted as jack2 on 5 of 8 days - a counter-example
    (g3 0.059 > FD_G3) that kept the family login node's net.src -> username
    dependency from holding, so no binding of the node was stated (bindings
    @14 2/6). A source whose history holds a second recurring value is judged
    on the node's own counts; the six one-valued users stay pooled and bound."""
    from app.engines.behavior.lib import pfd as FD
    ps, h, t = _family_login()
    xs = list(ps.x.keys())
    assert not BD._one_valued(h.get("192.168.1.40"))
    assert BD._one_valued(h.get("192.168.1.21"))
    old = FD.fit_pair(ps, t, support={str(x): FD.history_support(h.get(str(x))) for x in xs})
    assert not old["fd"]["holds"] and old["fd"]["g3"] > FD.FD_G3          # the round-11 veto
    rec = FD.fit_pair(ps, t, support=BD.pooled_support(h, xs, {}))
    assert rec["fd"]["holds"], rec["fd"]
    assert all(rec["table"][x].get("bound") and rec["table"][x]["top"] == y for x, y in FAM)
    assert not rec["table"]["192.168.1.40"].get("bound")
