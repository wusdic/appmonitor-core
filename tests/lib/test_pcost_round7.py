"""Round 7 (tree owner, PG4 attribute axis): exact rewrites of per-update
work that grew with the number of attributes. Each test runs the old code
(kept here as a reference) and the new one on the same stream and requires
the same state."""
from __future__ import annotations

import math

import numpy as np

from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.lib import psketch as PS


class _RefPair(PN.PairSketch):
    """PairSketch.update as of round 6 (a set of every tracked x per update)."""

    __slots__ = ()

    def update(self, xv, yv, t, mass, evidence):
        before = set(self.y.keys())
        self.x.add(xv, t, mass, evidence)
        if xv not in self.x:
            return
        tab = self.y.get(xv)
        if tab is None:
            for gone in before - set(self.x.keys()):
                self.y.pop(gone, None)
            tab = self.y[xv] = PS.DecayedSpaceSaving(PN.PAIR_KY, (PS.H_L,), PS.EV_HALF_LIVES, 0)
        tab.add(yv, t, mass, evidence)


def _state(ps: PN.PairSketch):
    x = ps.x
    return (list(x._keys), [list(r) for r in x._m], [list(r) for r in x._err], [list(r) for r in x._e],
            list(x._tot_m), list(x._evict_e),
            {k: (list(v._keys), [list(r) for r in v._m], [list(r) for r in v._e])
             for k, v in ((k, v.to_dss() if hasattr(v, "to_dss") else v) for k, v in ps.y.items())},
            list(ps.y.keys()))


def test_pair_sketch_under_churn_equals_the_round6_update():
    """An x with many more values than the 64 tracked (a synthetic attribute
    of the 340-attribute point, a source address) evicts on most updates."""
    rng = np.random.default_rng(7)
    a, b = PN.PairSketch(), _RefPair()
    t = 1_780_000_000.0
    for i in range(6000):
        xv = int(rng.zipf(1.3)) % 400
        yv = int(rng.integers(0, 6))
        m = float(rng.uniform(0.1, 3.0))
        t += float(rng.exponential(30.0))
        a.update(xv, yv, t, m, 1.0)
        b.update(xv, yv, t, m, 1.0)
    assert _state(a) == _state(b)
    assert set(a.y) <= set(a.x.keys())


def test_space_saving_eviction_takes_the_first_least_counter():
    """psketch eviction: the first counter with the least primary mass (ties
    included) - the counter min(range(k), key=...) chose."""
    ss = PS.DecayedSpaceSaving(4, (PS.H_M,), (PS.H_M,), 0)
    for k in ("a", "b", "c", "d"):
        ss.add(k, 0.0, 1.0, 1.0)              # four equal counters
    ss.add("e", 0.0, 1.0, 1.0)                # evicts the first of the tied minima
    assert ss.keys() == ["e", "b", "c", "d"]
    ss.add("b", 0.0, 0.5, 0.0)
    ss.add("f", 0.0, 1.0, 1.0)                # c is now the first least
    assert ss.keys() == ["e", "b", "f", "d"]
    assert math.isclose(ss._err[2][0], 1.0)
