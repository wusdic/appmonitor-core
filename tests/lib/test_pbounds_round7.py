"""lib/pbounds, round 7 (evaluator; the conformity owner's OC7-3): the
reference p_value reads a value at a run of equal grid points at the middle of
its CDF jump, as P03's _NumFast does since FIX9 (pscore.grid_mid_cdf). The
plain interpolation read a value AT the grid's minimum as F = 0, so every value
of a point mass (O-red / O-real: oa login net.pkts_down 2 against the range
2-2) got p = 0. Fails on the round-6 / HEAD code."""
from __future__ import annotations

import numpy as np

from app.engines.behavior import conformity as CF
from app.engines.behavior.lib import pbounds as PB


def test_a_value_of_a_point_mass_is_not_extreme_in_the_reference_p_value():
    rec = {"qgrid": [2.0] * 17, "band98": [2.0, 2.0], "n_c": 300.0}
    p, _ = PB.p_value(rec, 2.0)
    assert p == 1.0
    assert PB.p_value(rec, 3.0)[0] < 0.01
    g = [2.0] * 12 + [3.0] * 5
    rec2 = {"qgrid": g, "band98": [2.0, 3.0], "n_c": 300.0}
    assert PB.p_value(rec2, 2.0)[0] > 0.5


def test_the_reference_p_value_equals_p03s_fast_path_with_and_without_ties():
    rng = np.random.default_rng(4)
    grids = [[2.0] * 17, [2.0] * 12 + [3.0] * 5, [1.0] * 3 + list(np.linspace(1.0, 9.0, 14)),
             sorted(rng.normal(50, 5, 17).tolist())]
    for g in grids:
        rec = {"qgrid": [float(x) for x in g], "band98": [float(g[0]), float(g[-1])], "n_c": 300.0}
        fast = CF._NumFast(rec)
        for v in list(g) + [float(g[0]) - 1.0, float(g[-1]) + 1.0, 0.5 * (g[0] + g[-1])]:
            a = PB.p_value(rec, float(v))[0]
            b = fast.p(float(v))[0]
            assert abs(a - b) <= 1e-12, (g, v, a, b)


def test_without_ties_the_grid_cdf_is_the_linear_interpolation():
    q = [float(x) for x in range(17)]
    ps = np.linspace(0.0, 1.0, 17)
    for y in (0.0, 0.3, 4.0, 4.5, 15.99, 16.0):
        assert abs(PB._grid_cdf(q, y) - float(np.interp(y, q, ps))) <= 1e-15
