"""Round 5: the conformal p of a score at its ring's lowest level (lib/calib
module docstring "Lower atom").

PG5 D4 (pack O seeds 3 / 4, §16.12.15 item 4): the AUTO health monitor
192.168.9.9 scores conf_who / conf_seq = 0 (pm = 1, nothing unusual) on
almost every tick, so its B24 ring held only zeros and a score of 0 got the
randomised p = U - a seeded coin flip - and a perfectly normal source opened
incidents (oa 1758132000, conf_who p = 7.9e-5, on every seed). The issued p of
a score at or below the ring's lowest level is now the upper p of its tie block
(1); randomisation stays inside every genuinely tied level above it; the
calibration monitors keep the exactly uniform randomised p.

Proof by simulation: exchangeable null draws (point mass at the minimum of
mass pi in {0, 0.3, 0.9, 1}, an Exp body, and a Poisson lattice) - the issued
p is pointwise >= the randomised one, which is uniform, so P(p <= a) <= a;
the degenerate all-tie ring gives p = 1 whatever u.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.engines.behavior.lib import calib, combine, m_calib

N_RING = 64
TRIALS = 6000


def _ring(xs) -> calib.Ring:
    r = calib.Ring()
    for i, x in enumerate(xs):
        r.add(float(x), float(i))
    return r


def _draw(rng, kind: str, pi: float, size: int) -> np.ndarray:
    if kind == "lattice":
        return rng.poisson(0.3, size).astype(np.float64)        # P(0) = 0.74, levels 1, 2, ..
    x = rng.exponential(1.0, size) + 0.5
    return np.where(rng.random(size) < pi, 0.0, x)


def _null_ps(kind: str, pi: float, seed: int):
    """(issued p, randomised p) of the (n+1)-th of n+1 exchangeable null draws."""
    rng = np.random.default_rng(seed)
    p_new, p_rnd = np.empty(TRIALS), np.empty(TRIALS)
    for t in range(TRIALS):
        xs = _draw(rng, kind, pi, N_RING + 1)
        sc = np.sort(np.asarray([calib._r32(v) for v in xs[:-1]], dtype=np.float64))
        x = calib._r32(xs[-1])
        u = float(rng.random())
        p_new[t] = calib._conformal_p(sc, x, u)
        p_rnd[t] = calib._conformal_p(sc, x, u, rand_atom=True)
    return p_new, p_rnd


def _sd(a: float, n: int) -> float:
    return math.sqrt(a * (1.0 - a) / n)


@pytest.mark.parametrize("kind,pi", [("mix", 0.0), ("mix", 0.3), ("mix", 0.9), ("mix", 1.0),
                                     ("lattice", 0.0)])
def test_issued_p_is_valid_and_the_randomised_p_uniform_under_the_null(kind, pi):
    p_new, p_rnd = _null_ps(kind, pi, seed=int(pi * 10) + (100 if kind == "lattice" else 0))
    # the randomised p is exactly uniform (the monitors' p)
    assert calib.ks_uniform(p_rnd) < 0.03
    # the issued p is pointwise >= it: valid (super-uniform) at every level
    assert np.all(p_new >= p_rnd)
    for a in (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9):
        assert np.mean(p_new <= a) <= a + 3.0 * _sd(a, TRIALS), a
    if pi == 1.0:
        assert np.all(p_new == 1.0)                     # degenerate: never a coin flip
    elif kind == "mix" and pi == 0.0:
        # a continuous null: only a score below every ring entry (1/(n+1))
        # changes, from ~1 to 1
        assert np.mean(p_new != p_rnd) < 3.0 / (N_RING + 1)
        assert calib.ks_uniform(p_new) < 0.03
    else:
        # the atom's block (its null mass) is piled at 1 ...
        at = p_new == 1.0
        frac = np.mean(at)
        mass = 0.74 if kind == "lattice" else pi
        assert abs(frac - mass) < 0.05
        # ... and well below the atom's block (which starts near 1 - mass) the
        # issued p is exactly uniform: levels ABOVE the atom (the lattice's
        # 1, 2, ..) keep their randomisation, so P(p <= a) = a there
        for a in (0.005, 0.01, 0.25 * (1.0 - mass)):
            assert abs(np.mean(p_new <= a) - a) < 3.5 * _sd(a, TRIALS) + 1e-3, a


def test_degenerate_all_tie_ring_gives_one_for_every_u():
    ring = _ring(np.zeros(256))
    for u in (0.0, 1e-6, 7.9e-5, 0.3, 1.0):
        assert ring.p_value(0.0, u) == 1.0
        assert calib.p_from_ring(ring, 0.0, u) == 1.0
        assert calib.p_from_ring(ring, -1.0, u) == 1.0               # below the atom
        # the monitors' p: the seeded coin flip, exactly uniform under the null
        assert calib.p_from_ring(ring, 0.0, u, rand_atom=True) == pytest.approx(u)
    # B24's seeded U of the D4 incident: p = U = 7.9e-5 before round 5
    u = m_calib.uniform("oa", "192.168.9.9", "conf_who", 1758132000.0)
    assert calib.p_from_ring(ring, 0.0, u) == 1.0
    # a score above the atom is evidence: p <= 1/(n+1)
    assert ring.p_value(0.5, 0.5) == pytest.approx(0.5 / 257)


def test_genuine_ties_above_the_atom_stay_randomised():
    ring = _ring([0.0] * 200 + [1.0] * 50 + [2.0] * 6)
    n = 256
    for u in (0.1, 0.9):
        want = ((n - 250) + u * (50 + 1)) / (n + 1)
        assert ring.p_value(1.0, u) == pytest.approx(want)
        assert ring.p_value(1.0, u) == ring.p_value(1.0, u, rand_atom=True)
    assert ring.p_value(1.0, 0.1) < ring.p_value(1.0, 0.9)
    assert ring.p_value(0.0, 0.1) == ring.p_value(0.0, 0.9) == 1.0


def test_generic_rings_follow_the_same_rule():
    # arrays not holding the Ring invariant go through combine (float32, NaN tail)
    f32 = np.zeros(10, dtype=np.float32)
    assert calib._conformal_p(f32, 0.0, 0.2) == 1.0
    assert calib._conformal_p(f32, 0.0, 0.2, rand_atom=True) == pytest.approx(
        combine.randomized_conformal_p(f32, 0.0, 0.2))
    with_nan = np.array([0.0, 0.0, 1.0, np.nan])
    assert calib._conformal_p(with_nan, 0.0, 0.2) == 1.0
    assert calib._conformal_p(with_nan, 1.0, 0.2) == pytest.approx((0 + 0.2 * 2) / 4)
    assert calib._conformal_p(np.empty(0), 3.0, 0.2) == 1.0          # no history: no evidence
    assert calib._conformal_p(np.empty(0), 3.0, 0.2, rand_atom=True) == 0.2
    assert math.isnan(calib._conformal_p(np.empty(0), 3.0, math.nan))
    assert math.isnan(calib._conformal_p(f32, math.nan, 0.2))


def test_m_calib_paths_issue_one_at_the_lowest_level():
    u = 0.37
    zeros = _ring(np.zeros(100))
    small = _ring(np.zeros(20))
    # pooled class rings
    p, n = m_calib.pooled_p([zeros, small], 0.0, u)
    assert n == 120 and p == 1.0
    pr, _ = m_calib.pooled_p([zeros, small], 0.0, u, rand_atom=True)
    assert pr == pytest.approx(u * 121 / 121)
    p1, _ = m_calib.pooled_p([_ring([0.0] * 60 + [1.0] * 10)], 1.0, u)
    assert p1 == pytest.approx((0 + u * 11) / 71)                      # above the atom
    # pm prior: pm = 1 is -log10 pm = 0, the lowest level
    pm_r = _ring([m_calib.pm_score(1.0)] * 40)
    assert m_calib.pm_prior(pm_r, 1.0, u) == 1.0
    assert m_calib.pm_prior(pm_r, 1.0, u, rand_atom=True) == pytest.approx(u)
    big_pm = _ring([m_calib.pm_score(1.0)] * 150)                      # >= PM_CAL_N: conformal
    assert m_calib.pm_prior(big_pm, 1.0, u) == 1.0
    # B24's small-ring blend: a young all-zero ring with the pm = 1 prior
    assert m_calib.p_value(small, 0.0, u, m_calib.pm_prior(pm_r, 1.0, u)) == pytest.approx(1.0)
    assert m_calib.p_value(None, 0.0, u) == 1.0                        # nothing at all
    # B29's replay reproduces the issued p
    model = {m_calib.RINGS: {calib.ring_key("novelty", calib.stratum_key("wd_day", 900)): zeros}}
    assert m_calib.p_replay(model, "novelty", "wd_day", 900, 0.0, u, pm=1.0) == 1.0
    assert m_calib.p_from_snapshot(model, "novelty", calib.stratum_key("wd_day", 900),
                                   0.0, u) == 1.0
