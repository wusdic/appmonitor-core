"""B14 bocpd model p-value (round 4, evaluator): pm = min(1, 1/BF), the
Bayes-factor bound of cp = P(r <= 3 h | data) against the hazard prior, and
score.bocpd = -log10 pm.

bocpd was the only detector without a pm, so B24 extrapolated its score
purely from the ring tail: a machine persona's warm-up cp stayed near 0
(ring max 0.007) and a live cp of 0.54 was issued p = 2e-21 (pack D seed 0:
8 of 14 HIGH+ control incidents were bocpd accumulator alarms). With a pm,
B24's p-score tail floor (lib/calib) bounds the extrapolation by the
model's own scale.
"""
from __future__ import annotations

import math

import numpy as np

from helpers import T0, make_store, run_engine
from test_b14_changepoint import S, feed

from app.engines.behavior import changepoint as CP
from app.engines.behavior.lib import emit, m_cp
from app.engines.behavior.lib.features import FEATURE_DIM


def test_pm_is_the_bayes_factor_bound():
    odds = CP.BOC_PRIOR_ODDS
    assert math.isclose(CP.BOC_PRIOR, 1.0 - (1.0 - CP.HAZARD) ** (CP.BOC_WINDOW_H + 1))
    assert CP.bocpd_pm(0.0) == 1.0
    assert CP.bocpd_pm(CP.BOC_PRIOR) == 1.0                      # posterior = prior: no evidence
    assert math.isclose(CP.bocpd_pm(0.54), odds * 0.46 / 0.54, rel_tol=1e-12)
    cps = np.linspace(0.05, 0.999999, 50)
    pms = [CP.bocpd_pm(c) for c in cps]
    assert all(a >= b for a, b in zip(pms, pms[1:]))              # monotone: ranking unchanged
    assert CP.bocpd_pm(1.0) > 0.0 and math.isnan(CP.bocpd_pm(float("nan")))


def test_pm_is_valid_on_a_null():
    """Hourly N(0,1) and AR(1) 0.3 inputs, 4 x 5000 h each: P(pm <= a) <= a
    (Markov / Ville bound of a Bayes factor; the bound is conservative)."""
    for phi in (0.0, 0.3):
        pms = []
        for seed in range(4):
            rng = np.random.default_rng(seed)
            st, x = CP.bocpd_new(), np.zeros(2)
            for t in range(5000):
                x = phi * x + math.sqrt(1.0 - phi * phi) * rng.standard_normal(2)
                st = CP.bocpd_step(st, x)
                if t >= 168:
                    pms.append(CP.bocpd_pm(CP.bocpd_prob(st)))
        pms = np.asarray(pms)
        for a in (0.1, 0.05, 0.01):
            assert np.mean(pms <= a) <= a, (phi, a, np.mean(pms <= a))


def test_engine_writes_bocpd_pm_and_its_score():
    """Through ChangepointEngine at 3600 s (one BOCPD step per tick): pm
    carries bocpd_pm(cp.prob) and the score is its -log10; a level shift
    drives cp up and pm down with it."""
    store, eng = make_store(), CP.ChangepointEngine()
    rng = np.random.default_rng(3)
    dt = 3600.0
    rows = rng.standard_normal((120, FEATURE_DIM))
    rows[90:, :] += 3.0
    seen, low, low0 = 0, 1.0, 1.0
    for i, zr in enumerate(rows):
        t = T0 + i * dt
        vec = 8.0 + 0.5 * zr
        feed(store, "e", t, zr, dt=dt, vec=vec, nat=np.expm1(vec))
        run_engine(eng, store, t, dt=dt)
        cp = store.vec_at(S, "e", m_cp.CP_PROB, t)
        if cp is None or not math.isfinite(float(cp[0])):
            continue
        sc = emit.read_row(store, S, "e", emit.SCORE, t)
        pm = emit.read_row(store, S, "e", emit.PM, t)
        want = CP.bocpd_pm(float(cp[0]))
        assert math.isclose(pm["bocpd"], want, rel_tol=1e-5, abs_tol=1e-30), (i, pm["bocpd"], want)
        assert math.isclose(sc["bocpd"], -math.log10(want), rel_tol=1e-5, abs_tol=1e-6)
        seen += 1
        if i >= 90:
            low = min(low, pm["bocpd"])
        else:
            low0 = min(low0, pm["bocpd"])
    assert seen > 50 and low < 0.5 * low0, (low, low0)


def test_first_hours_after_a_start_are_unscored():
    """Round 4: while every run is <= BOC_WINDOW_H hours long, P(r <= 3 h) = 1
    by construction; a new IP was issued bocpd p = 1e-38 and a CRITICAL
    alarm on its first H tick (pack A: L6 renumbered IPs, L8 new employee)."""
    st = CP.bocpd_new()
    rng = np.random.default_rng(0)
    for k in range(1, 7):
        st = CP.bocpd_step(st, rng.standard_normal(2))
        assert CP.bocpd_steps(st) == k
    assert CP.bocpd_steps({"r": np.zeros(1)}) == math.inf       # a pre-round-4 state
    store, eng = make_store(), CP.ChangepointEngine()
    dt = 3600.0
    rows = rng.standard_normal((12, FEATURE_DIM))
    for i, zr in enumerate(rows):
        t = T0 + i * dt
        vec = 8.0 + 0.5 * zr
        feed(store, "new", t, zr, dt=dt, vec=vec, nat=np.expm1(vec))
        run_engine(eng, store, t, dt=dt)
        m = store.get_model(S, "new", m_cp.MODEL)
        steps = CP.bocpd_steps(m["run"]["bocpd"])
        sc = emit.read_row(store, S, "new", emit.SCORE, t)
        acc = emit.read_dict(store, S, "new", emit.ACC_ALARM, t)
        if steps <= CP.BOC_WINDOW_H:
            assert not math.isfinite(sc.get("bocpd", math.nan)), (i, steps, sc.get("bocpd"))
            assert int(acc.get("bocpd", 0) or 0) == 0
        elif steps >= CP.BOC_WINDOW_H + 2:
            assert math.isfinite(sc["bocpd"])
