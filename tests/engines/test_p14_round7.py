"""P14 views, round 7 (time / workflow / content / views owner): the
confidence of a group's part that states its own windows composes the node's
held-out record WITHOUT the node's 'when' tests with the part's own window
confidence. Fails on round 6 (52e813f: no compose_conf; the part took
min(p_hold, own))."""
from __future__ import annotations

import math

from app.engines.behavior import views as VW


class _Hold:
    def __init__(self, facets):
        self._f = facets

    def facet_tests(self):
        return dict(self._f)


class _Node:
    """A node whose statement failed its 'when' tests (the node's union
    window of three departments) and held every other constraint."""

    def __init__(self, facets, p_hold=0.45, prior=(8.0, 2.0)):
        self.hold = _Hold(facets) if facets is not None else None
        self.meta = {"hold_prior": list(prior)}
        self._p = p_hold

    def p_hold(self, t=None):
        return self._p


def test_a_part_with_its_own_windows_is_not_charged_the_nodes_when_failures():
    nd = _Node({"~when": (19.0, 20.0), "when": (6.0, 20.0), "~net": (6.0, 20.0)})
    rest = VW.hold_without_when(nd)
    assert math.isclose(rest, (19.0 + 8.0) / (20.0 + 10.0))
    # own windows (part_when) stated at 0.88: min(rest 0.9, 0.88), not min(p_hold 0.45, 0.88)
    got = VW.compose_conf(nd, 0.0, 0.3, ("G1", ["10.0.0.1"], None, 1.0, ["G1"]), True, 0.88)
    assert math.isclose(got, 0.88)
    # the node's own statement (its own union windows) keeps the full record
    assert math.isclose(VW.compose_conf(nd, 0.0, 0.3, None, False, 0.88), 0.45)


def test_without_a_when_free_record_the_part_falls_back_to_p_hold():
    nd = _Node(None)
    assert VW.hold_without_when(nd) is None
    assert math.isclose(VW.compose_conf(nd, 0.0, 0.3, ("G1", ["10.0.0.1"], None, 1.0, ["G1"]), True, 0.88), 0.45)
    # no record at all: the min-over-parts value passed in
    class _Bare:
        meta = {}

        def p_hold(self, t=None):
            return None
    assert VW.compose_conf(_Bare(), 0.0, 0.3, None, True, 0.88) == 0.3


def test_a_workflow_clause_takes_its_dependency_not_the_successor_share():
    """Fails on round 6: pack O seed 1 day 9, oa GET /docs and its three group
    parts stated 0.107 (no P04 record yet: the min over the parts) = the
    Jeffreys lower bound of GET /docs/{num} -> GET /docs's share of the
    document's next steps, although the clause ("A -> B") states the
    dependency (0.98) and the statements held 1.0."""
    from app.engines.behavior.lib import prender as PR
    edges = [{"from": "GET oa.corp.local /docs/{num}", "to": "GET oa.corp.local /docs", "dep": 0.9799,
              "band": [141.0, 1796.3], "confidence": 0.1075},
             {"from": "GET oa.corp.local /docs", "to": "GET oa.corp.local /docs/{num}", "dep": 0.9799,
              "band": [10.1, 56.1], "confidence": 0.9799}]
    ev, zh, en, f_c = PR.workflow_block(edges)
    assert math.isclose(f_c, 0.9799)
    # the edge's own share-bounded confidence stays in the evidence
    assert [e["confidence"] for e in ev] == [0.1075, 0.9799]
    # an edge without a dependency (older models) keeps its confidence
    assert math.isclose(PR.workflow_block([{"from": "a", "to": "b", "confidence": 0.4}])[3], 0.4)


def test_a_parts_own_windows_enter_as_the_probability_they_hold():
    """Fails on round 6: a group's part that states its own windows composed
    them as coverage x day stability - pack O seed 2, days 12-20, the mail
    parts of two departments (two windows, 9 sessions on 3 dates, coverage
    0.65, stability 0.67) stated 0.20-0.46 and held (parts stated 0.876,
    held 0.976). The windows enter as the probability that they hold at
    their stated coverage (pwindows.window_hold), the "holds" of P04's record
    and of the statement contract."""
    import numpy as np
    from app.engines.behavior.lib import pnode as PN
    from app.engines.behavior.lib import pwindows as PW
    rec = {"res": "minute", "windows": [[568, 584], [806, 834]], "coverage": 0.65, "w_points": 9.0,
           "cv": [33.8, 52.0], "stability": 0.67, "all_day": False}
    h = PW.window_hold(rec)
    assert 0.6 < h < 0.9 and h > PW.confidence(rec)
    # more sessions at the same coverage: more certain that it holds
    assert PW.window_hold(dict(rec, w_points=90.0, cv=[338.0, 520.0])) > h
    # an all-day record states no time constraint; a drifting one is drift_hold's
    assert PW.window_hold(dict(rec, all_day=True)) == 1.0
    assert PW.window_hold(dict(rec, drift={"workday": {"k": 0, "n": 3}})) is None
    # composition: min(hold of the node's other constraints, the windows' hold)
    nd = _Node({"~when": (19.0, 20.0)}, p_hold=0.45, prior=(8.0, 2.0))
    part = ("G1", ["10.0.0.1"], None, 1.0, ["G1"])
    assert math.isclose(VW.compose_conf(nd, 0.0, 0.3, part, True, PW.confidence(rec), h), min(0.9, h))
    # part_when reports it: two sessions a workday of three members over 6 dates
    OFF, DAY, D0 = 8 * 3600.0, 86400.0, 20341
    node = PN.Node(1, None, 0, 0, (), D0 * DAY - OFF)
    node.when.want_minutes(True, R=4096, seed=0)
    r = np.random.default_rng(5)
    for d in range(6):
        for k, src in enumerate(("10.0.0.1", "10.0.0.2", "10.0.0.3")):
            for lo, hi in ((570.0, 585.0), (805.0, 835.0)):
                m = float(r.uniform(lo, hi))
                node.when.update(0, m, (D0 + d) * DAY - OFF + m * 60.0, 1.0, 1.0, src=src)
    pw = PW.part_when(node.when, ["10.0.0.1", "10.0.0.2", "10.0.0.3"], OFF)
    assert pw is not None and pw.get("hold") is not None
    wd = pw["by_daytype"]["wd"]
    assert math.isclose(pw["hold"], PW.window_hold(wd))
