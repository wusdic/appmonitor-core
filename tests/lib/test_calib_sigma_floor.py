"""Round 4 (evaluator): the tail-scale floor for p-scores (lib/calib
P_SCORE_SIGMA, m_calib.tail_sigma_min).

A detector whose score is -log10 of its own p-value pm has, under an exact
null, excesses over any threshold u distributed Exp(ln 10): GPD scale
1/ln 10. B24's rings of H-stream detectors hold warm-up scores for weeks,
and those were compressed (a young model's predictive is heavier than the
mature one's), so the PWM tail was steep and its extrapolation beyond the
ring turned ordinary live pm into p ~ 1e-9 (pack A seed 0: t2 29x nominal
at 1e-3 where pm was 2.7x). The floor keeps the issued p from decaying
faster than pm beyond u.
"""
from __future__ import annotations

import math

import numpy as np

from app.engines.behavior.lib import calib, m_calib

LV = 1e-3


def _ratio(compress: float, floor: bool, n_rep: int = 40, n_live: int = 4000) -> float:
    """Ring of 256 warm-up p-scores -log10(U^compress) (compress < 1: a
    conservative, compressed warm-up), robust tail; live pm ~ U (valid).
    Returns realised / nominal of issued p <= 1e-3."""
    rng = np.random.default_rng(7)
    hits = n = 0
    for _ in range(n_rep):
        r = calib.Ring()
        for i, x in enumerate(-compress * np.log10(rng.random(calib.RING_M))):
            r.add(float(x), float(i))
        r.gpd = calib.robust_tail(r, 0.0)
        pm = rng.random(n_live)
        for v, uu in zip(pm, rng.random(n_live)):
            x = -math.log10(v)
            sm = m_calib.tail_sigma_min(x, v) if floor else 0.0
            hits += calib.p_from_ring(r, x, float(uu), sigma_min=sm) <= LV
            n += 1
    return hits / (n * LV)


def test_exact_null_stays_calibrated_with_the_floor():
    assert 0.5 <= _ratio(1.0, True) <= 2.0


def test_compressed_ring_is_bounded_by_the_floor():
    """Warm-up scores at half their live scale: without the floor the tail
    extrapolation is ~30x nominal at 1e-3; with it the excess is bounded by
    the ring's own body offset (rate 10^u ~ 3x)."""
    no = _ratio(0.5, False, n_rep=10)
    yes = _ratio(0.5, True, n_rep=10)
    assert no > 15.0, no
    assert yes < 5.0, yes


def test_tail_sigma_min_only_for_rows_whose_score_is_minus_log10_pm():
    assert m_calib.tail_sigma_min(3.0, 1e-3) == calib.P_SCORE_SIGMA
    assert m_calib.tail_sigma_min(0.0, 1.0) == calib.P_SCORE_SIGMA
    assert m_calib.tail_sigma_min(3.0, float("nan")) == 0.0      # bocpd / seq: no pm
    assert m_calib.tail_sigma_min(3.0, 0.2) == 0.0               # a statistic, not -log10 pm
    assert m_calib.tail_sigma_min(float("nan"), 1e-3) == 0.0


def test_floor_never_lowers_p_and_keeps_the_ring_max_cap():
    rng = np.random.default_rng(3)
    r = calib.Ring()
    for i, x in enumerate(-0.5 * np.log10(rng.random(calib.RING_M))):
        r.add(float(x), float(i))
    r.gpd = calib.robust_tail(r, 0.0)
    for x in (float(r.gpd.u) + 0.1, 3.0, 8.0, 30.0):
        p0 = calib.p_from_ring(r, x, 0.5)
        p1 = calib.p_from_ring(r, x, 0.5, sigma_min=calib.P_SCORE_SIGMA)
        assert p1 >= p0
        if x > r.scores[-1]:
            assert p1 <= 1.0 / (len(r) + 1)
    # beyond the steep fit the floor binds: p = rate 10^-(x - u)
    x = 8.0
    t = r.gpd
    want = max(t.sf(x), t.rate * 10.0 ** -(x - t.u))
    assert math.isclose(calib.p_from_ring(r, x, 0.5, sigma_min=calib.P_SCORE_SIGMA), want,
                        rel_tol=1e-9)
