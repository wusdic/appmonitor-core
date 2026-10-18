"""lib/pdfg round 8: accepted edges and required predecessors are retired with
hysteresis (separate accept / retire rules with evidence bounds), so an edge
whose dependency sits at the acceptance cut does not flip every mining pass
(pack O seed 1, GA.oa.approvals list -> item at dependency 0.806, lost on days
19-21)."""
from __future__ import annotations

import pytest

from app.engines.behavior.lib import pdfg as DF

T = 1_790_000_000.0


def _chain(st, g, seq, t, gap=60.0):
    ids = [st.acts.add(k, t, 1, 1)[0] for k in seq]
    for x in ids:
        st.cnt.add((g, x), t, 1, 1)
    st.starts.add((g, ids[0]), t, 1, 1)
    for x, y in zip(ids, ids[1:]):
        st.add_edge(g, x, y, t, 1, 1, DF.delay_bin(gap))
    for j, y in enumerate(ids):
        st.top_b.add(y)
        st.pcnt.add((g, y), t, 1, 1)
        for x in ids[:j]:
            st.add_prec(g, y, x, t, 1, 1)
    return ids


def _pairs(st, sc):
    return {(st.acts.key_of(e["a"]), st.acts.key_of(e["b"])): e for e in sc["edges"]}


def test_dependency_upper_bound_brackets_the_point_value():
    for c_ab, c_ba in ((50, 4), (50, 6), (10, 3), (200, 20)):
        d = DF.dependency(c_ab, c_ba)
        assert DF.dependency_upper(c_ab, c_ba) > d
    # a well-supported edge just under the cut is not confidently below it ...
    assert DF.dependency(50, 6) < DF.DEP_MIN <= DF.dependency_upper(50, 6)
    # ... a weak or reversed one is
    assert DF.dependency_upper(10, 4) < DF.DEP_MIN
    assert DF.dependency_upper(50, 30) < DF.DEP_MIN


def test_an_accepted_edge_at_the_cut_is_held_and_a_new_one_is_not_accepted():
    st = DF.FlowState()
    t = T
    for k in range(50):
        _chain(st, "*", ["list", "item"], t + 600 * k)
    for k in range(5):
        _chain(st, "*", ["item", "list"], t + 600 * (50 + k))
    t1 = t + 600 * 56
    sc = DF.mine_scope(st, "*", t1)
    e = _pairs(st, sc).get(("list", "item"))
    assert e is not None and e["dep"] >= DF.DEP_MIN and not e.get("held")
    # one more reverse succession: the point dependency dips below 0.8
    _chain(st, "*", ["item", "list"], t1 + 60)
    _chain(st, "*", ["item", "list"], t1 + 120)
    sc = DF.mine_scope(st, "*", t1 + 600)
    e = _pairs(st, sc).get(("list", "item"))
    assert e is not None, "an accepted edge flipped out on a dip at the cut"
    assert e["dep"] < DF.DEP_MIN and e["held"] is True
    # the same counts on a fresh state are not accepted (acceptance unchanged)
    st2 = DF.FlowState()
    for k in range(50):
        _chain(st2, "*", ["list", "item"], t + 600 * k)
    for k in range(7):
        _chain(st2, "*", ["item", "list"], t + 600 * (50 + k))
    assert ("list", "item") not in _pairs(st2, DF.mine_scope(st2, "*", t1 + 600))


def test_a_held_edge_is_retired_once_the_evidence_contradicts_it():
    st = DF.FlowState()
    for k in range(50):
        _chain(st, "*", ["a", "b"], T + 600 * k)
    t1 = T + 600 * 50
    assert ("a", "b") in _pairs(st, DF.mine_scope(st, "*", t1))
    for k in range(30):
        _chain(st, "*", ["b", "a"], t1 + 60 * (k + 1))
    t2 = t1 + 60 * 40
    assert DF.dependency_upper(st.edges.ev(("*", st.acts.id_of("a"), st.acts.id_of("b")), t2),
                               st.edges.ev(("*", st.acts.id_of("b"), st.acts.id_of("a")), t2)) < DF.DEP_MIN
    assert ("a", "b") not in _pairs(st, DF.mine_scope(st, "*", t2))
    assert not any(k[1:] == (st.acts.id_of("a"), st.acts.id_of("b")) for k in st.edges_on)


def test_an_accepted_requirement_is_held_through_a_few_misses():
    st = DF.FlowState()
    for k in range(40):
        _chain(st, "*", ["form", "generate"], T + 600 * k)
    t1 = T + 600 * 41
    req = {(st.acts.key_of(r["b"]), st.acts.key_of(r["a"])): r for r in DF.mine_scope(st, "*", t1)["requires"]}
    assert ("generate", "form") in req and not req[("generate", "form")].get("held")
    for k in range(3):
        _chain(st, "*", ["generate"], t1 + 60 * (k + 1))      # generate without form
    t2 = t1 + 600
    sc = DF.mine_scope(st, "*", t2)
    req = {(st.acts.key_of(r["b"]), st.acts.key_of(r["a"])): r for r in sc["requires"]}
    r = req.get(("generate", "form"))
    assert r is not None and r["lb"] < DF.REQ_LB and r["held"] is True
    # the same counts never accepted before: not a requirement
    st2 = DF.FlowState()
    for k in range(40):
        _chain(st2, "*", ["form", "generate"], T + 600 * k)
    for k in range(3):
        _chain(st2, "*", ["generate"], t1 + 60 * (k + 1))
    assert not DF.mine_scope(st2, "*", t2)["requires"]
    # many misses: retired
    for k in range(40):
        _chain(st, "*", ["generate"], t2 + 60 * (k + 1))
    assert not DF.mine_scope(st, "*", t2 + 3600)["requires"]


def test_retiring_an_action_purges_its_accepted_entries():
    st = DF.FlowState()
    for k in range(30):
        _chain(st, "*", ["x", "y"], T + 600 * k)
    DF.mine_scope(st, "*", T + 600 * 31)
    y = st.acts.id_of("y")
    assert any(y in k[1:] for k in st.edges_on) and any(y in k[1:] for k in st.reqs_on)
    st.retire(y)
    assert not any(y in k[1:] for k in st.edges_on) and not any(y in k[1:] for k in st.reqs_on)


def test_a_loop_edge_is_held_while_its_loop_evidence_stays_above_half():
    """Pack O seed 1: the approvals' list -> item edge was kept on the loop
    measure l2 = 0.804 (~4 decayed loop patterns) and vanished whenever the
    count dipped under 4 (days 8-10, 12-13). Accepted at l2 >= 0.8 (n >= 4),
    retired at l2 < 2/3 (n < 2)."""
    assert abs(DF.L2_RETIRE - 2.0 / 3.0) < 1e-12
    st = DF.FlowState()
    g = "*"
    ids = [st.acts.add(k, T, 1, 1)[0] for k in ("item", "approve")]
    a, b = ids
    for x in ids:
        st.cnt.add((g, x), T, 1, 1)
    for k in range(12):
        st.add_edge(g, a, b, T + k, 1, 1, 3)
        st.add_edge(g, b, a, T + k, 1, 1, 3)
    for k in range(5):
        st.add_loop(g, a, b, T + k, 1, 1)
    e = [x for x in DF.mine_scope(st, g, T + 20)["edges"] if (x["a"], x["b"]) == (a, b)]
    assert e and e[0]["loop2"] is not None and not e[0].get("held")
    # the steps go on (a -> b and b -> a every few hours) but the loop count
    # decays under 4 (l2 < 0.8) while staying above 2: held
    t, l2 = T + 20, 1.0
    while l2 >= DF.L2_MIN:
        t += 3 * 3600.0
        st.add_edge(g, a, b, t, 1, 1, 3)
        st.add_edge(g, b, a, t, 1, 1, 3)
        l2 = DF.loop2_measure(st.loop2.ev((g, a, b), t), st.loop2.ev((g, b, a), t))
    assert DF.L2_RETIRE <= l2 < DF.L2_MIN, l2
    e = [x for x in DF.mine_scope(st, g, t)["edges"] if (x["a"], x["b"]) == (a, b)]
    assert e and e[0]["held"] is True and e[0]["dep"] == pytest.approx(l2, abs=1e-4)
    # long after the loops stopped (l2 < 2/3): retired
    while l2 >= DF.L2_RETIRE:
        t += 3 * 3600.0
        st.add_edge(g, a, b, t, 1, 1, 3)
        st.add_edge(g, b, a, t, 1, 1, 3)
        l2 = DF.loop2_measure(st.loop2.ev((g, a, b), t), st.loop2.ev((g, b, a), t))
    assert not [x for x in DF.mine_scope(st, g, t)["edges"] if (x["a"], x["b"]) == (a, b)]
