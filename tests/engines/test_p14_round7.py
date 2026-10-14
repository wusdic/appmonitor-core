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
