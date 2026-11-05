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
