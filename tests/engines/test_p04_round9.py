"""P04 round 9 (docs/lib3/progressive.md §16.16.11-§16.16.12): untested
statements keep the prior they were issued with; the who split of a small
learned-group node (S5 regression on O-real); see also test_p05_round9."""
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


class _SS:
    """A stub of pevalue.SplitStats with the statistics _check_valid_first reads:
    candidate 0 = the /24 level (valid on an identity it predicts), 1 = the
    learned-group level (larger selective saving, e-process still growing)."""

    def __init__(self, l2e, sg, margin10, W=27.0, n=(60.0, 27.0)):
        import numpy as np
        self.C_ever = 7
        self.n = np.asarray(n, dtype=float)
        self.C = 2
        self.ordinal = [0, 1]
        self._l2e, self._sg = list(l2e), list(sg)
        self.W = np.full((2, 2), W)
        self._m = {(1, 0): margin10, (0, 1): -margin10}
        self.G = np.asarray(sg, dtype=float)

    def roll(self, day):
        pass

    def active(self):
        return [0, 1]

    def log2_e(self, i):
        return self._l2e[i]

    def selective_gain(self, i):
        return self._sg[i]

    def selective_margin(self, c, o):
        return self._m[(c, o)]

    def check(self, tau0, s_mode):
        from app.engines.behavior.lib import pevalue as PE
        c1 = int(max(self.active(), key=lambda i: self.G[i]))
        dec = PE.SplitDecision(c1, None)
        dec.log2_e = self._l2e[c1]
        dec.threshold = tau0 + 2.807
        dec.pass_v = dec.log2_e >= dec.threshold
        dec.pass_g = dec.pass_s = dec.pass_d = dec.pass_m = True
        return dec


def test_a_valid_split_waits_for_a_rival_that_saves_more_on_the_same_events():
    """A candidate that has passed (V) is not taken while a tracked rival with
    growing evidence saves >= tau0 bits more on their common events: on O-real
    seed 0 the login node of 综合部 + 财务部 (learned groups G18 + G19) took the
    /24 split on day 18 (log2 e 13.3, earned on the usernames, an identity)
    while the group level saved 38.9 bits more (log2 e 4.2), and kept it: the
    综合部 login statement was lost. The rival only defers the split."""
    from app.engines.behavior.pattern_tree import _check_valid_first
    # /24 valid (13.3 >= 12.8), group level not yet, beats it by 38.9 bits
    dec = _check_valid_first(_SS([13.3, 4.2], [19.7, 58.6], 38.9), 10.0, "margin", 18)
    assert not dec.accept
    # the rival passes (V) in turn: it is the split taken
    dec = _check_valid_first(_SS([14.0, 13.0], [19.7, 58.6], 38.9), 10.0, "margin", 29)
    assert dec.accept and dec.best == 1
    # a rival without evidence (log2 e <= 0) or within tau0 bits does not hold the split
    assert _check_valid_first(_SS([13.3, -2.0], [19.7, 58.6], 38.9), 10.0, "margin", 18).accept
    assert _check_valid_first(_SS([13.3, 4.2], [19.7, 25.0], 6.0), 10.0, "margin", 18).accept
    # a rival that has counted as much evidence as the valid candidate and still
    # has not passed (V) is the weaker one: the split is taken (bounded wait)
    assert _check_valid_first(_SS([13.3, 4.2], [19.7, 58.6], 38.9, n=(60.0, 60.0)), 10.0, "margin", 40).accept
