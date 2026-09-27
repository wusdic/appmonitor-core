"""B25 FusionEngine: statistical unit tests (b) and (c) of docs/lib3/engines.md.

(b) Five dependent nulls (equicorrelated Gaussians, rho = 0.64, one-sided
    p) in five families over 2e5 ticks: the realised rate of p_all <= 1e-3
    is within [0.8, 1.3]x nominal for the raw wHMP and within [0.8, 1.2]x
    after the per-entity meta-calibration (the engine's own ring functions:
    one M = 256 ring, commit delay D = 4 ticks, GPD tail refits).
(c) Uniform q_inst null at dt = 900 s and 60 s: evidence-CUSUM alarm onsets
    per entity-day within [0.015, 0.045]. One run of 200 entity-days holds
    only ~6 expected onsets (Poisson sd 2.4, i.e. +-0.012 per entity-day), so
    the band is asserted on 10 replicates of 200 entity-days (pooled and
    median), which is the spec's check with a usable power.
Fixed seeds throughout.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

import helpers  # noqa: F401  (sys.path)

from app.engines.behavior import fusion as F
from app.engines.behavior.lib import calib, combine
from app.engines.behavior.lib.detectors import DETECTOR_INDEX, N_DETECTORS

from test_b25_fusion import E, Rig

FIVE = ["marg_int", "marg_shape", "peer", "novelty", "seq"]     # five families


def test_b_dependent_nulls_rate_before_and_after_meta_calibration():
    rng = np.random.default_rng(0)
    n, rho = 200_000, 0.64
    z = (math.sqrt(rho) * rng.standard_normal((n, 1))
         + math.sqrt(1 - rho) * rng.standard_normal((n, 5)))
    p = norm.sf(z)
    hmp = 5.0 / np.sum(1.0 / p, axis=1)
    # the vectorised HMP is the engine's p_all (one detector per family, unit weights)
    row = np.full(N_DETECTORS, np.nan)
    for i in range(0, n, 20_000):
        for j, d in enumerate(FIVE):
            row[DETECTOR_INDEX[d]] = p[i, j]
        fz = F.fuse(row)
        assert fz.p_all == pytest.approx(hmp[i], rel=1e-12)
        assert fz.p_inst == pytest.approx(hmp[i], rel=1e-12)
    raw = float(np.mean(hmp <= 1e-3)) / 1e-3
    assert 0.8 <= raw <= 1.3, raw

    s = (-np.log10(hmp)).tolist()
    u = rng.random(n).tolist()
    pr = hmp.tolist()
    ring = calib.Ring()
    cnt = 0
    qs = []
    D = 4
    add, pq = F.meta_add, F.meta_q
    for t in range(n):
        if t >= D:
            cnt = add(ring, s[t - D], float(t - D), cnt)
        qs.append(pq(ring, s[t], u[t], pr[t]))
    q = np.asarray(qs)
    meta = float(np.mean(q <= 1e-3)) / 1e-3
    assert 0.8 <= meta <= 1.2, meta
    assert np.all((q > 0) & (q <= 1))


@pytest.mark.parametrize("dt", [900.0, 60.0])
def test_c_uniform_null_evidence_alarm_rate(dt):
    h = F.evidence_h(dt)
    ticks = int(200 * 86400 / dt)
    rates = []
    for seed in range(10):
        q = np.random.default_rng(100 + seed).random(ticks)
        _, onsets = F.evidence_path(q, h)
        rates.append(onsets / 200.0)
    pooled = float(np.mean(rates))
    assert 0.015 <= pooled <= 0.045, rates
    assert 0.015 <= float(np.median(rates)) <= 0.045, rates


def test_c_engine_cusum_matches_vectorised_path():
    """The engine's evidence series on a uniform null equals evidence_path."""
    rig = Rig()
    q = np.random.default_rng(7).random(150)
    S_ = []
    for x in q:
        rig.step({E: {"novelty": float(x)}}, trust=None)      # empty rings: q_inst = p
        S_.append(rig.v(F.EVIDENCE))
    path, _ = F.evidence_path(q, F.evidence_h(900))
    np.testing.assert_allclose(S_, path, rtol=1e-5, atol=1e-5)
    assert combine.e_day(0.5, 900) == pytest.approx(48.0)
