"""P04 round 9 (docs/lib3/progressive.md §16.16.11-§16.16.12): untested
statements keep the prior they were issued with; a source split is not paid
by the payload of a stated binding (S5 regression on O-real); see also
test_p05_round9."""
from __future__ import annotations

from ptree_sim import DAY, MON

from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.lib import ptree as PT
from app.engines.behavior.pattern_tree import PatternTreeEngine


def _route(i):
    return (("http.route", 0, frozenset({f"GET oa.corp.local /r{i}"}), False),)


def _pool(tr, n_fail_first, base=100):
    """16 tested statements of one kind: `n_fail_first` failed their first
    (and only) test, the others held 20 of 20 from their first test on."""
    cons = {"who": (0, frozenset({"1.1.1.1"}), 0.95)}
    for i in range(16):
        nd = PN.Node(base + i, tr.root, 1, 0, _route(base + i), MON)
        nd.state = "confirmed"
        nd.ref = {"hold": dict(cons)}
        hr = nd.meta["hold"] = PN.HoldRecord()
        if i < n_fail_first:
            hr.T, hr.ti, hr.fl = [0.0, 1.0], 1, [[0, ["when"]]]
        else:
            hr.T, hr.ti, hr.fl = [20.0, 20.0], 20, []
        tr.nodes[nd.id] = nd


def test_an_untested_statement_keeps_the_prior_it_was_issued_with():
    """Its confidence moves on its own evidence only. Taken from the daily
    refit of the pooled prior, every untested statement moved with the other
    statements' tests: pack O seed 4 stated 0.9688 for four untested patterns
    on day 10 and 0.9664 on day 21 (PG2's sign test: 23 lost / 4 gained,
    p = 0.0002). Here the pool's first-test pass rate falls from ~0.9 to ~0.1
    while the statement has no test: it states its issuance value throughout,
    and takes the current prior at its first test of its own."""
    eng = PatternTreeEngine()
    m = PT.PTreeModel("oa")
    tr = m.tree(0, MON)
    _pool(tr, 2)                                   # 14 of 16 passed their first test
    cons = {"who": (0, frozenset({"1.1.1.1"}), 0.95)}
    fresh = PN.Node(99, tr.root, 1, 0, _route(99), MON)
    fresh.state = "confirmed"
    fresh.ref = {"hold": dict(cons)}
    tr.nodes[fresh.id] = fresh
    noref = PN.Node(98, tr.root, 1, 0, _route(98), MON)   # confirmed, not referenced yet
    noref.state = "confirmed"
    tr.nodes[noref.id] = noref
    eng._hold_priors(m, 1, MON)
    eng._hold_priors(m, 2, MON + DAY)
    p0 = fresh.p_hold(MON + DAY)
    q0 = noref.p_hold(MON + DAY)
    assert p0 is not None and p0 > 0.8, p0
    assert q0 is not None and q0 > 0.8, q0
    # the other statements now fail their first tests
    for nid in list(tr.nodes):
        if nid >= 100:
            del tr.nodes[nid]
    _pool(tr, 14, base=200)
    for d in (3, 4, 5):
        eng._hold_priors(m, d, MON + (d - 1) * DAY)
        assert fresh.p_hold(MON + (d - 1) * DAY) == p0, (d, fresh.p_hold(MON + (d - 1) * DAY), p0)
        assert noref.p_hold(MON + (d - 1) * DAY) == q0, (d, noref.p_hold(MON + (d - 1) * DAY), q0)
    assert eng._hp["prior0"]["*"][0] / sum(eng._hp["prior0"]["*"]) < 0.5   # the pool did move
    # its constraints stated: the kind's issuance rate (once), then fixed again
    noref.ref = {"hold": dict(cons)}
    eng._hold_priors(m, 6, MON + 5 * DAY)
    q1 = noref.p_hold(MON + 5 * DAY)
    assert q1 is not None and q1 < q0
    eng._hold_priors(m, 7, MON + 6 * DAY)
    assert noref.p_hold(MON + 6 * DAY) == q1
    # its first test of its own: the current prior with its own record
    hr = fresh.meta["hold"] = PN.HoldRecord()
    hr.T, hr.ti, hr.fl = [1.0, 1.0], 1, []
    eng._hold_priors(m, 8, MON + 7 * DAY)
    a, b = fresh.meta["hold_prior"]
    assert abs(fresh.p_hold(MON + 7 * DAY) - round((1.0 + a) / (1.0 + a + b), 4)) < 1e-9
    assert fresh.p_hold(MON + 7 * DAY) < p0


def test_a_restated_statement_is_issued_anew():
    """A new confidence segment (an accepted change: HoldRecord.drop) is a new
    statement: it takes the prior of the day it is re-stated."""
    eng = PatternTreeEngine()
    m = PT.PTreeModel("oa")
    tr = m.tree(0, MON)
    _pool(tr, 2)
    cons = {"who": (0, frozenset({"1.1.1.1"}), 0.95)}
    nd = PN.Node(99, tr.root, 1, 0, _route(99), MON)
    nd.state = "confirmed"
    nd.ref = {"hold": dict(cons)}
    nd.meta["hold"] = PN.HoldRecord()
    tr.nodes[nd.id] = nd
    eng._hold_priors(m, 1, MON)
    eng._hold_priors(m, 2, MON + DAY)
    p0 = nd.p_hold(MON + DAY)
    for nid in list(tr.nodes):
        if nid >= 100:
            del tr.nodes[nid]
    _pool(tr, 14, base=200)
    eng._hold_priors(m, 3, MON + 2 * DAY)
    assert nd.p_hold(MON + 2 * DAY) == p0
    nd.meta["hold"].drop()
    eng._hold_priors(m, 4, MON + 3 * DAY)
    assert nd.p_hold(MON + 3 * DAY) < p0


def _feed(ss, n_events=600, seed=0):
    """Three sources whose username is a function of the source (a binding)
    and whose body size is the same lottery for all: the who candidate
    predicts the username, not what the sources do."""
    import numpy as np
    rng = np.random.default_rng(seed)
    kb = 8
    leaf = np.ones((2, kb))
    for e in range(n_events):
        src = ["ip1", "ip2", "ip3"][int(rng.integers(0, 3))]
        b = np.array([{"ip1": 0, "ip2": 1, "ip3": 2}[src], int(rng.integers(0, kb))])
        p = leaf / leaf.sum(axis=1, keepdims=True)
        ss.update([src, f"c{int(rng.integers(0, 16))}"][:ss.C], b, p, 1.0, day=e // 40)
        leaf[np.arange(2), b] += 1


def test_a_source_split_is_not_paid_by_a_stated_binding():
    """O-real seed 0 (round 8): the family login node of 综合部 + 财务部 split by
    /24 on day 18, each /24 mixing the departments' members, paid by the
    usernames - a value fixed by the source, which any partition of the
    sources predicts; the 综合部 login statement was lost. Once P08 states the
    binding (P05's who_bound), a source candidate no longer counts the
    payload: here nothing else separates the sources, so no split."""
    import math
    from app.engines.behavior.lib import pevalue as PE
    from app.engines.behavior.pattern_tree import _mask_bound_targets
    targets = ["body.kv.username", "body.len"]
    ss = PE.SplitStats(2, k_b=8, C=2)
    ss.set_candidate(0, ("net.src", 1), card_hint=256)
    ss.set_candidate(1, ("body.kv.captcha", 0), card_hint=16)
    _feed(ss)
    thr = PE.TAU0 + math.log2(ss.C_ever)
    assert ss.log2_e(0) > thr                       # paid by the usernames
    before1 = ss.log2_e(1)
    _mask_bound_targets(ss, targets, ["body.kv.username"])
    assert ss.log2_e(0) < thr
    assert not ss.tmask[0, 0] and ss.tmask[0, 1]
    assert ss.log2_e(1) == before1                  # a non-source candidate keeps its targets
    assert ss.tmask[1].all()
    # nothing stated: nothing masked
    ss2 = PE.SplitStats(2, k_b=8, C=1)
    ss2.set_candidate(0, ("net.src", 1), card_hint=256)
    _feed(ss2)
    _mask_bound_targets(ss2, targets, [])
    assert ss2.tmask.all() and ss2.log2_e(0) > thr


def test_a_tested_statement_takes_the_pools_prior_only_in_its_own_tests_direction():
    """Pack O seed 4, days 10 -> 21: statements that passed every test of
    their own lost confidence at their next passed test because the pool's
    prior had declined meanwhile (19 lost / 8 gained, PG2 sign test p =
    0.026). After a passed test the refit cannot lower a statement's
    confidence; after a failed one it cannot raise it; the round-4 rule (the
    first days' pessimistic prior is left behind as the statement keeps
    passing) still holds."""
    eng = PatternTreeEngine()
    m = PT.PTreeModel("oa")
    tr = m.tree(0, MON)
    _pool(tr, 2)
    cons = {"who": (0, frozenset({"1.1.1.1"}), 0.95)}
    nd = PN.Node(99, tr.root, 1, 0, _route(99), MON)
    nd.state = "confirmed"
    nd.ref = {"hold": dict(cons)}
    hr = nd.meta["hold"] = PN.HoldRecord()
    hr.T, hr.ti, hr.fl = [3.0, 3.0], 3, []
    tr.nodes[nd.id] = nd
    eng._hold_priors(m, 1, MON)
    eng._hold_priors(m, 2, MON + DAY)
    p0 = nd.p_hold(MON + DAY)
    for nid in list(tr.nodes):                     # the pool's prior declines
        if nid >= 100:
            del tr.nodes[nid]
    _pool(tr, 12, base=200)
    eng._hold_priors(m, 3, MON + 2 * DAY)
    hr.T, hr.ti = [4.0, 4.0], 4                    # its own 4th test: passed
    eng._hold_priors(m, 4, MON + 3 * DAY)
    p1 = nd.p_hold(MON + 3 * DAY)
    assert p1 >= p0, (p0, p1)
    hr.T, hr.ti, hr.fl = [4.0, 5.0], 5, [[4, ["who"]]]   # its 5th test: failed
    eng._hold_priors(m, 5, MON + 4 * DAY)
    p2 = nd.p_hold(MON + 4 * DAY)
    assert p2 < p1
    a, b = nd.meta["hold_prior"]
    assert a / (a + b) < 0.5                       # the failure takes the lower current prior
