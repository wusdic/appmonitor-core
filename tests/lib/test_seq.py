"""Tests for engines/behavior/lib/seq.py: wall-clock thresholds (Siegmund,
evidence, LLR, Bernoulli, MCUSUM table), the chart steps, AR(1)
prewhitening and the Mann-Kendall / Theil-Sen creep statistics.

Thresholds are checked twice: against the reference numbers quoted in
docs/lib3/engines.md (B07, B14, B25) and by simulation, i.e. the realised
zero-state ARL of a chart run with the library's own step function and
threshold is compared with the target ARL (reduced ARLs for speed). All
simulations are vectorised over independent chains with fixed seeds, so the
file is deterministic and runs in a few seconds.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from app.engines.behavior.lib import seq as Q  # noqa: E402

NAN = float("nan")


# ------------------------------------------------------------------ helpers
def _siegmund_arl(k: float, h: float) -> float:
    """Siegmund's ARL0 for N(0, 1) increments (the formula h_gauss inverts)."""
    y = 2.0 * k * (h + 2.0 * Q.SIEGMUND_RHO)
    return (math.exp(y) - y - 1.0) / (2.0 * k * k)


def _realised_arl(step, draw, h: float, n_chains: int, t_max: int, seed: int,
                  state_shape=()) -> float:
    """Zero-state ARL of a chart: n independent chains from S = 0, alarm when
    stat > h. step(S, x) -> (S', stat). Chains that have not alarmed by t_max
    are censored and handled by the exponential MLE sum(min(T, t_max)) / #alarms
    (exact for geometric run lengths, negligible bias otherwise)."""
    rng = np.random.default_rng(seed)
    S = np.zeros((n_chains,) + tuple(state_shape))
    T = np.full(n_chains, -1, dtype=np.int64)
    t = 0
    while t < t_max:
        t += 1
        S, stat = step(S, draw(rng, n_chains))
        hit = (stat > h) & (T < 0)
        T[hit] = t
        if (T >= 0).all():
            break
    crossed = T >= 0
    assert crossed.sum() > 0.5 * n_chains, "simulation horizon too short"
    return float(np.where(crossed, T, t).sum() / crossed.sum())


def _gauss_step(k):
    def step(S, x):
        S = Q.cusum_step(S, x, k)
        return S, S
    return step


def _normal(rng, n):
    return rng.standard_normal(n)


def _ar1(n: int, phi: float, rng, n_chains: int = 1) -> np.ndarray:
    """Stationary AR(1) with unit marginal variance, shape [n, n_chains]."""
    e = rng.standard_normal((n, n_chains)) * math.sqrt(1.0 - phi * phi)
    x = np.empty((n, n_chains))
    x[0] = rng.standard_normal(n_chains)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


# --------------------------------------------------------------- arl_ticks
def test_arl_ticks_wall_clock():
    assert Q.arl_ticks(2400, 900) == pytest.approx(230400.0)
    assert Q.arl_ticks(2400, 60) == pytest.approx(3456000.0)
    assert Q.arl_ticks(100, 900) == pytest.approx(9600.0)
    assert Q.arl_ticks(33, 3600) == pytest.approx(792.0)
    with pytest.raises(ValueError):
        Q.arl_ticks(10, 0)
    with pytest.raises(ValueError):
        Q.arl_ticks(10, -60)


# ----------------------------------------------------------------- h_gauss
@pytest.mark.parametrize("dt, k, target", [
    (900, 0.25, 19.4), (900, 1.0, 5.35),
    (60, 0.25, 24.8), (60, 1.0, 6.7),
    (3600, 0.25, 16.6), (3600, 1.0, 4.66),
])
def test_h_gauss_matches_b14_reference(dt, k, target):
    h = Q.h_gauss(k, Q.arl_ticks(2400, dt))
    assert abs(h - target) / target < 0.05
    assert abs(h - target) < 0.06                     # in fact within the quoted rounding
    assert Q.h_for("gauss", 2400, dt, k=k) == pytest.approx(h)


@pytest.mark.parametrize("k", [0.05, 0.25, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("arl", [10.0, 1e2, 1e4, 3.456e6, 1e9])
def test_h_gauss_inverts_siegmund(k, arl):
    h = Q.h_gauss(k, arl)
    if h > 0.0:
        assert _siegmund_arl(k, h) == pytest.approx(arl, rel=1e-8)
    else:                                   # ARL below ARL0(h = 0): threshold floored
        assert arl <= _siegmund_arl(k, 0.0) * (1 + 1e-12)


def test_h_gauss_monotone_and_edges():
    arls = np.logspace(1, 8, 30)
    for k in (0.25, 0.5, 1.0):
        hs = [Q.h_gauss(k, a) for a in arls]
        assert all(np.diff(hs) >= 0.0)
    for a in (1e3, 1e6):
        hs = [Q.h_gauss(k, a) for k in (0.1, 0.25, 0.5, 1.0, 2.0)]
        assert all(np.diff(hs) < 0.0)
    assert Q.h_gauss(0.5, 1.0) == 0.0 and Q.h_gauss(0.5, 0.0) == 0.0
    assert Q.h_gauss(0.5, -3.0) == 0.0
    # k -> 0 limit ARL = b^2 is continuous
    assert Q.h_gauss(0.0, 100.0) == pytest.approx(10.0 - 1.166)
    assert Q.h_gauss(1e-9, 100.0) == pytest.approx(Q.h_gauss(0.0, 100.0), rel=1e-6)
    assert Q.h_gauss(1e-3, 100.0) == pytest.approx(Q.h_gauss(0.0, 100.0), rel=1e-2)
    # no overflow; capped at the search interval
    assert Q.h_gauss(1.0, 1e300) == Q.H_MAX
    assert Q.h_gauss(1.0, math.inf) == Q.H_MAX
    assert math.isnan(Q.h_gauss(NAN, 100.0)) and math.isnan(Q.h_gauss(0.5, NAN))
    with pytest.raises(ValueError):
        Q.h_gauss(-0.1, 100.0)


@pytest.mark.parametrize("k, arl", [(0.25, 500.0), (0.5, 300.0), (1.0, 300.0)])
def test_h_gauss_realised_arl_by_simulation(k, arl):
    h = Q.h_gauss(k, arl)
    got = _realised_arl(_gauss_step(k), _normal, h, 1500, int(8 * arl), seed=11)
    assert 0.7 * arl <= got <= 1.3 * arl, (k, arl, h, got)
    # Siegmund is accurate here: within 10%
    assert abs(got / arl - 1.0) < 0.10


def test_b14_family_alarm_rate_is_cadence_independent():
    """B14 (a) in miniature: 48 N(0,1) charts (12 x 2 sides x k in {.25, 1})
    at ARL 2400 d each -> family rate 0.02 per entity-day, at 900 s and 3600 s."""
    rates = {}
    for dt, days, seeds in ((900, 30, 16), (3600, 60, 16)):
        k = np.tile(np.repeat([0.25, 1.0], 24), seeds)          # [seeds * 48]
        h = np.array([Q.h_for("gauss", 2400, dt, k=kk) for kk in k])
        rng = np.random.default_rng(dt)
        S = np.zeros(k.size)
        alarms = 0
        n_ticks = int(days * 86400 / dt)
        for _ in range(n_ticks):
            z = rng.standard_normal(k.size // 2)
            x = np.concatenate([z, -z])                         # upper / lower side
            S = Q.cusum_step(S, x, k)
            hit = S > h
            alarms += int(hit.sum())
            S[hit] = 0.0
        rates[dt] = alarms / (seeds * days)
    assert rates[900] <= 0.03 and rates[3600] <= 0.03
    assert 0.5 <= (rates[3600] + 1e-3) / (rates[900] + 1e-3) <= 2.0


# ------------------------------------------------------------- h_evidence
@pytest.mark.parametrize("dt, target", [(900, 5.31), (60, 8.19), (3600, 3.83)])
def test_h_evidence_reference(dt, target):
    h = Q.h_evidence(Q.arl_ticks(33, dt))
    assert h == pytest.approx(target, abs=0.01)
    assert Q.h_for("evidence", 33, dt) == pytest.approx(h)


def test_h_evidence_edges():
    assert Q.h_evidence(1.0) == 0.0 and Q.h_evidence(0.0) == 0.0
    assert Q.h_evidence(math.exp(Q.EVIDENCE_A)) == pytest.approx(0.0)
    assert Q.h_evidence(10.0) == 0.0                         # ln 10 < 3.07: floored
    assert math.isnan(Q.h_evidence(NAN))


@pytest.mark.parametrize("dt, n_alarm", [(900, 4), (60, 6)])
def test_evidence_cusum_persistent_q_alarm_tick(dt, n_alarm):
    """B25: a persistent q = 0.01 alarms in 4 ticks at 900 s, 6 at 60 s."""
    h = Q.h_for("evidence", 33, dt)
    S, t = 0.0, 0
    while S < h:
        S = Q.evidence_cusum_step(S, 0.01)
        t += 1
    assert t == n_alarm


def test_evidence_cusum_step():
    assert Q.evidence_cusum_step(0.0, 0.01) == pytest.approx(-math.log(0.01) - 3.0)
    assert Q.evidence_cusum_step(2.0, 1.0) == 0.0            # floored at 0
    assert Q.evidence_cusum_step(5.0, 0.5) == pytest.approx(5.0 + math.log(2.0) - 3.0)
    assert Q.evidence_cusum_step(4.2, NAN) == 4.2            # NaN q: unchanged, not reset
    assert Q.evidence_cusum_step(0.0, 0.0) == pytest.approx(-math.log(1e-300) - 3.0)
    assert Q.evidence_cusum_step(1.0, 7.0) == 0.0            # q > 1 clipped to 1


@pytest.mark.parametrize("arl", [300.0, 1000.0])
def test_h_evidence_realised_arl_by_simulation(arl):
    """Null q ~ U(0, 1): the realised ARL matches the exact-integral-equation fit."""
    h = Q.h_evidence(arl)

    def step(S, q):
        S = np.maximum(0.0, S - np.log(q) - Q.EVIDENCE_DRIFT)
        return S, S

    got = _realised_arl(step, lambda r, n: r.random(n), h, 1500, int(8 * arl), seed=5)
    assert 0.7 * arl <= got <= 1.3 * arl, (arl, h, got)


# ------------------------------------------------------------------ h_llr
def test_h_llr_values_and_conservative_by_simulation():
    assert Q.h_llr(math.e ** 4) == pytest.approx(4.0)
    assert Q.h_llr(1.0) == 0.0 and Q.h_llr(0.5) == 0.0
    assert math.isnan(Q.h_llr(NAN))
    assert Q.h_for("llr", 100, 900) == pytest.approx(math.log(9600.0))
    # LLR of N(1,1) vs N(0,1) under H0: lambda = x - 1/2, E0 exp(lambda) = 1
    arl = 50.0
    h = Q.h_llr(arl)

    def step(S, lam):
        S = np.maximum(0.0, S + lam)
        return S, S

    got = _realised_arl(step, lambda r, n: r.standard_normal(n) - 0.5, h, 800, 4000, seed=6)
    assert got >= arl                                        # Lorden bound: conservative


# -------------------------------------------------------------- Bernoulli
def test_bernoulli_threshold_reference():
    assert Q.bernoulli_threshold(9600) == pytest.approx(13.73, abs=0.005)
    assert Q.h_for("bernoulli", 100, 900) == pytest.approx(Q.bernoulli_threshold(9600))
    assert Q.bernoulli_threshold(1.0) == 0.5
    assert Q.bernoulli_threshold(0.1) == 0.5                 # ARL < 1 counts as 1
    assert math.isnan(Q.bernoulli_threshold(NAN))


@pytest.mark.parametrize("p0, p1", [(0.02, 0.5), (0.1, 0.5), (0.15, 0.75), (0.25, 0.95),
                                    (0.3, 0.95), (0.0, 0.5), (-0.1, 0.5)])
def test_rhythm_p1(p0, p1):
    assert Q.rhythm_p1(p0) == pytest.approx(p1)


def test_rhythm_p1_undefined_bins():
    assert math.isnan(Q.rhythm_p1(0.31)) and math.isnan(Q.rhythm_p1(0.9))
    assert math.isnan(Q.rhythm_p1(NAN))


def test_bernoulli_cusum_bits_b07():
    """p0 = 0.02: an active slot adds 4.64 bits, the 3rd active slot alarms."""
    p0 = 0.02
    p1 = Q.rhythm_p1(p0)
    h = Q.h_for("bernoulli", 100, 900)
    w1 = Q.bernoulli_cusum_step(0.0, 1, p0, p1)
    assert w1 == pytest.approx(math.log2(25.0)) and w1 == pytest.approx(4.64, abs=0.005)
    w0 = Q.bernoulli_cusum_step(10.0, 0, p0, p1)
    assert w0 == pytest.approx(10.0 + math.log2(0.5 / 0.98))
    W, n = 0.0, 0
    while W < h:
        W = Q.bernoulli_cusum_step(W, 1, p0, p1)
        n += 1
    assert n == 3
    assert Q.bernoulli_cusum_step(0.5, 0, p0, p1) == 0.0      # floored at 0


def test_bernoulli_cusum_step_nan_and_clipping():
    assert Q.bernoulli_cusum_step(3.0, 1, 0.5, Q.rhythm_p1(0.5)) == 3.0   # p0 > 0.3 bin
    assert Q.bernoulli_cusum_step(3.0, 1, NAN, 0.5) == 3.0
    assert Q.bernoulli_cusum_step(3.0, NAN, 0.02, 0.5) == 3.0
    w = Q.bernoulli_cusum_step(0.0, 1, 0.0, 1.0)              # clipped to 1e-6 / 1-1e-6
    assert math.isfinite(w) and w == pytest.approx(math.log2((1 - 1e-6) / 1e-6))
    assert Q.bernoulli_cusum_step(0.0, True, 0.02, 0.5) == pytest.approx(math.log2(25.0))


def test_bernoulli_threshold_conservative_by_simulation():
    """Wald bound: E0[2^inc] = 1 so ARL >= 2^h; the realised ARL is above target."""
    p0, arl = 0.2, 200.0
    p1 = Q.rhythm_p1(p0)
    h = Q.bernoulli_threshold(arl)
    up, dn = math.log2(p1 / p0), math.log2((1 - p1) / (1 - p0))

    def step(W, a):
        W = np.maximum(0.0, W + np.where(a, up, dn))
        return W, W

    got = _realised_arl(step, lambda r, n: r.random(n) < p0, h, 600, 6000, seed=7)
    assert got >= arl


# ---------------------------------------------------------------- CUSUM step
def test_cusum_step_vectorised_nan_keeps_state():
    S = np.array([0.0, 1.0, 2.0, 5.0])
    x = np.array([2.0, -3.0, NAN, 0.5])
    k = np.array([0.5, 0.5, 0.5, 1.0])
    out = Q.cusum_step(S, x, k)
    np.testing.assert_allclose(out, [1.5, 0.0, 2.0, 4.5])
    assert out is not S and S[2] == 2.0
    # lower side: charts pass -x
    np.testing.assert_allclose(Q.cusum_step(np.zeros(2), -np.array([-2.0, 1.0]), 0.5), [1.5, 0.0])
    # scalar k broadcasts, scalar in -> 0-d out
    assert float(Q.cusum_step(1.0, 0.2, 0.25)) == pytest.approx(0.95)
    # a NaN never resets an accumulated statistic
    S = np.array([7.0])
    for _ in range(5):
        S = Q.cusum_step(S, np.array([NAN]), 0.25)
    assert S[0] == 7.0


def test_cusum_stationary_p():
    S = np.array([0.0, 1.0, 5.0, 20.0])
    p = Q.cusum_stationary_p(S, 0.5)
    np.testing.assert_allclose(p, np.exp(-2 * 0.5 * (S + 0.583)))
    p48 = Q.cusum_stationary_p(S, 0.5, n_charts=48)
    np.testing.assert_allclose(p48, np.minimum(1.0, 48 * np.exp(-(S + 0.583))))
    assert p48[0] == 1.0 and p48[1] == 1.0
    assert Q.cusum_stationary_p(np.array([1e6]), 1.0)[0] == Q.P_FLOOR
    assert math.isnan(Q.cusum_stationary_p(np.array([NAN]), 0.5)[0])
    assert np.all(np.diff(Q.cusum_stationary_p(np.linspace(0, 30, 50), 0.25)) <= 0.0)
    # per-chart k (the B14 bank mixes k = 0.25 and 1.0)
    kk = np.array([0.25, 1.0, 0.25, 1.0])
    np.testing.assert_allclose(Q.cusum_stationary_p(S, kk, 48),
                               np.minimum(1.0, 48 * np.exp(-2 * kk * (S + 0.583))))


def test_cusum_stationary_p_is_calibrated_under_null():
    """Stationary N(0,1) CUSUM: P(p_eq <= a) ~ a (the tail formula is honest)."""
    rng = np.random.default_rng(8)
    k = 0.5
    S = np.zeros(2000)
    ps = []
    for t in range(1200):
        S = Q.cusum_step(S, rng.standard_normal(S.size), k)
        if t >= 200 and t % 5 == 0:
            ps.append(Q.cusum_stationary_p(S, k))
    ps = np.concatenate(ps)
    for a in (0.1, 0.01):
        assert 0.7 <= float((ps <= a).mean()) / a <= 1.3


# ------------------------------------------------------------- prewhitening
def test_ar1_phi_estimates_and_clips():
    rng = np.random.default_rng(9)
    x = _ar1(4000, 0.5, rng)[:, 0]
    assert Q.ar1_phi(x) == pytest.approx(0.5, abs=0.04)
    assert Q.ar1_phi(_ar1(4000, 0.95, rng)[:, 0]) == 0.8               # clipped high
    assert Q.ar1_phi(_ar1(4000, -0.6, rng)[:, 0]) == 0.0               # clipped low
    assert Q.ar1_phi(rng.standard_normal(4000)) == pytest.approx(0.0, abs=0.04)
    assert Q.ar1_phi(list(x[:500])) == pytest.approx(0.5, abs=0.12)   # plain sequence


def test_ar1_phi_nan_pairs_and_degenerate():
    rng = np.random.default_rng(10)
    x = _ar1(4000, 0.6, rng)[:, 0]
    x[rng.random(x.size) < 0.2] = NAN                     # ~64% of pairs survive
    assert Q.ar1_phi(x) == pytest.approx(0.6, abs=0.05)
    assert Q.ar1_phi(np.arange(10.0)) == 0.0             # 9 pairs < 10
    assert Q.ar1_phi(np.arange(11.0)) == 0.8             # 10 pairs: defined (r = 1, clipped)
    y = np.full(40, NAN)
    y[::2] = 1.0                                         # no adjacent finite pair
    assert Q.ar1_phi(y) == 0.0
    assert Q.ar1_phi([0.1] * 50) == 0.0                  # constant: zero variance
    assert Q.ar1_phi([]) == 0.0 and Q.ar1_phi([NAN] * 30) == 0.0
    z = np.where(np.arange(60) % 2 == 0, 1.0, -1.0)
    assert Q.ar1_phi(z) == 0.0                           # perfectly alternating: -1 -> 0
    assert Q.ar1_phi(np.r_[np.zeros(30), np.ones(30)]) == 0.8   # step: r ~ 0.93 -> clip
    assert Q.ar1_phi([1.0, 2.0, math.inf] * 10) == Q.ar1_phi([1.0, 2.0, NAN] * 10)


def test_prewhiten_formula_and_nan_policy():
    x_t = np.array([1.0, 2.0, NAN, 0.5, 1.0])
    x_p = np.array([0.5, NAN, 1.0, NAN, 1.0])
    phi = np.array([0.5, 0.5, 0.5, 0.0, 0.8])
    u = Q.prewhiten(x_t, x_p, phi)
    np.testing.assert_allclose(u[0], (1.0 - 0.25) / math.sqrt(0.75))
    assert u[1] == 2.0                                   # x_prev NaN -> x_t
    assert math.isnan(u[2])                              # x_t NaN -> NaN
    assert u[3] == 0.5
    np.testing.assert_allclose(u[4], 0.2 / 0.6)
    # scalar phi broadcasts; phi = 0 is the identity; NaN phi = no whitening
    np.testing.assert_allclose(Q.prewhiten(x_t[:2], np.zeros(2), 0.0), x_t[:2])
    assert Q.prewhiten(1.0, 1.0, NAN) == pytest.approx(1.0)
    # out-of-contract phi is clipped to AR1_PHI_CLIP (no division by ~0)
    assert Q.prewhiten(1.0, 1.0, 0.999) == pytest.approx(Q.prewhiten(1.0, 1.0, 0.8))


def test_prewhiten_ar1_gives_white_unit_variance():
    rng = np.random.default_rng(12)
    x = _ar1(6000, 0.6, rng)[:, 0]
    phi = Q.ar1_phi(x)
    u = Q.prewhiten(x[1:], x[:-1], np.full(x.size - 1, phi))
    assert np.var(u) == pytest.approx(1.0, abs=0.06)
    assert abs(np.corrcoef(u[1:], u[:-1])[0, 1]) < 0.04
    assert Q.ar1_phi(u) == pytest.approx(0.0, abs=0.04)


def test_prewhitening_restores_cusum_arl_on_ar1_input():
    """Why B14 prewhitens: AR(1) residuals (phi = 0.5) shorten the run length
    of a chart designed for iid N(0,1); after prewhitening + clipping to +-3
    the realised ARL is back within 30% of target."""
    k, arl, phi = 0.5, 300.0, 0.5
    h = Q.h_gauss(k, arl)
    n_chains, t_max = 800, 2400
    rng = np.random.default_rng(13)
    x = _ar1(t_max + 1, phi, rng, n_chains)
    res = {}
    for mode in ("raw", "white"):
        S = np.zeros(n_chains)
        T = np.full(n_chains, -1)
        for t in range(1, t_max + 1):
            if mode == "raw":
                inp = x[t]
            else:
                inp = np.clip(Q.prewhiten(x[t], x[t - 1], phi), -Q.PSI_CLIP, Q.PSI_CLIP)
            S = Q.cusum_step(S, inp, k)
            T[(S > h) & (T < 0)] = t
            if (T >= 0).all():
                break
        crossed = T >= 0
        res[mode] = np.where(crossed, T, t).sum() / crossed.sum()
    assert res["raw"] < 0.5 * arl                         # autocorrelation: ~3x too many alarms
    assert 0.7 * arl <= res["white"] <= 1.3 * arl, res


# ----------------------------------------------------------------- MCUSUM
def test_mcusum_step_maths():
    S = np.zeros(2)
    S1, st = Q.mcusum_step(S, np.array([3.0, 4.0]))           # C = 5
    np.testing.assert_allclose(S1, np.array([3.0, 4.0]) * (1 - 0.5 / 5.0))
    assert st == pytest.approx(4.5) and isinstance(st, float)
    assert st == pytest.approx(float(np.linalg.norm(S1)))
    S2, st2 = Q.mcusum_step(S1, np.array([-2.7, -3.6]))       # S1 + x = 0: C <= k -> reset
    assert st2 == 0.0 and np.all(S2 == 0.0)
    S3, st3 = Q.mcusum_step(np.array([0.2, 0.0]), np.array([0.1, 0.1]))   # C < k -> reset
    assert st3 == 0.0 and np.all(S3 == 0.0)
    # NaN entries contribute 0; input not mutated
    S = np.array([1.0, 1.0, 0.0])
    S4, st4 = Q.mcusum_step(S, np.array([NAN, 1.0, NAN]), k=0.5)
    Y = np.array([1.0, 2.0, 0.0])
    np.testing.assert_allclose(S4, Y * (1 - 0.5 / np.linalg.norm(Y)))
    assert S.tolist() == [1.0, 1.0, 0.0]
    # d = 1 keeps the sign: a symmetric two-sided chart
    Sm, stm = Q.mcusum_step(np.array([0.0]), np.array([-2.0]))
    assert Sm[0] == pytest.approx(-1.5) and stm == pytest.approx(1.5)


def test_mcusum_step_batch_matches_rows():
    rng = np.random.default_rng(14)
    S = rng.standard_normal((5, 4))
    x = rng.standard_normal((5, 4)) * 0.3
    x[2, 1] = NAN
    Sb, sb = Q.mcusum_step(S, x, k=0.7)
    assert Sb.shape == (5, 4) and sb.shape == (5,)
    for i in range(5):
        Si, si = Q.mcusum_step(S[i], x[i], k=0.7)
        np.testing.assert_allclose(Sb[i], Si)
        assert sb[i] == pytest.approx(si)


def test_nan_state_restarts_charts_instead_of_killing_them():
    """Regression: a NaN in a stored chart state (e.g. cusum_state that
    round-tripped a JSON null) used to disable the chart for good. MCUSUM took
    the C <= k branch forever (S' = NaN * 0, stat 0 under any input) and the
    scalar CUSUM stayed NaN. Both now restart from 0, like the scalar steps."""
    S = np.array([NAN, 0.0, 0.0])
    for _ in range(3):
        S, st = Q.mcusum_step(S, np.array([9.0, 9.0, 9.0]))
    assert np.all(np.isfinite(S)) and st > 30.0
    # batch: only the corrupt row restarts, the healthy row is untouched
    Sb = np.array([[NAN, 1.0], [1.0, 1.0]])
    out, stb = Q.mcusum_step(Sb, np.zeros((2, 2)))
    assert np.all(out[0] == 0.0) and stb[0] == 0.0
    np.testing.assert_allclose(out[1], Q.mcusum_step(Sb[1], np.zeros(2))[0])
    # one-sided bank: a NaN state entry restarts, others are unaffected
    S = np.array([NAN, 1.0])
    for _ in range(3):
        S = Q.cusum_step(S, np.array([5.0, 5.0]), np.array([0.25, 0.25]))
    np.testing.assert_allclose(S, [14.25, 15.25])
    assert np.all(np.isfinite(Q.cusum_stationary_p(S, 0.25, 48)))
    # NaN x still leaves a (finite) state unchanged; a NaN state with NaN x is 0
    np.testing.assert_allclose(Q.cusum_step(np.array([NAN, 3.0]), np.array([NAN, NAN]), 0.5), [0.0, 3.0])
    # consistent with the scalar steps
    assert Q.evidence_cusum_step(NAN, 1e-5) == pytest.approx(-math.log(1e-5) - 3.0)
    assert Q.bernoulli_cusum_step(NAN, 1, 0.02, 0.5) == pytest.approx(math.log2(25.0))


def test_mcusum_table_shape_and_monotone():
    H = Q.MCUSUM_H
    assert isinstance(H, np.ndarray) and H.shape == (16, len(Q.MCUSUM_ARL_GRID))
    assert np.all(np.isfinite(H)) and np.all(H > 0.0)
    assert np.all(np.diff(H, axis=1) > 0.0)               # increasing in ARL
    assert np.all(np.diff(H, axis=0) > 0.0)               # increasing in d
    assert not H.flags.writeable                          # shared constant
    # ln ARL(h) grows at most at the asymptotic rate 2k = 1: dh/dlnARL >= 1
    slopes = np.diff(H, axis=1) / np.diff(np.log(Q.MCUSUM_ARL_GRID))
    assert np.all(slopes >= 0.95)
    # Crosier (1988): d = 2, k = 0.5, h = 5.5 -> in-control ARL ~ 200
    assert Q.mcusum_h(2, 200.0) == pytest.approx(5.5, abs=0.3)


def test_mcusum_h_interpolation_and_extrapolation():
    H = Q.MCUSUM_H
    g = Q.MCUSUM_ARL_GRID
    for d in (1, 5, 12, 16):
        for j, a in enumerate(g):
            assert Q.mcusum_h(d, a) == pytest.approx(H[d - 1, j])
        mid = math.sqrt(g[1] * g[2])                    # midpoint in ln ARL
        assert Q.mcusum_h(d, mid) == pytest.approx(0.5 * (H[d - 1, 1] + H[d - 1, 2]))
        hi = g[-1] * 10.0                               # linear beyond the grid
        s = (H[d - 1, -1] - H[d - 1, -2]) / math.log(10.0)
        assert Q.mcusum_h(d, hi) == pytest.approx(H[d - 1, -1] + s * math.log(10.0))
        lo = g[0] / math.e
        s0 = (H[d - 1, 1] - H[d - 1, 0]) / math.log(10.0)
        assert Q.mcusum_h(d, lo) == pytest.approx(max(0.0, H[d - 1, 0] - s0))
    assert Q.mcusum_h(40, 1e4) == Q.mcusum_h(16, 1e4)
    assert Q.mcusum_h(0, 1e4) == Q.mcusum_h(1, 1e4)
    assert Q.mcusum_h(3, 1.0) == 0.0 and Q.mcusum_h(3, 1.5) >= 0.0
    assert math.isnan(Q.mcusum_h(3, NAN))
    assert Q.h_for("mcusum", 100, 900, d=12) == pytest.approx(Q.mcusum_h(12, 9600.0))
    assert Q.h_for("mcusum", 100, 900, d=12) > Q.h_for("mcusum", 100, 3600, d=12)


@pytest.mark.parametrize("d, arl", [(2, 300.0), (4, 2000.0), (12, 300.0)])
def test_mcusum_table_matches_simulation(d, arl):
    h = Q.mcusum_h(d, arl)
    got = _realised_arl(lambda S, x: Q.mcusum_step(S, x, Q.MCUSUM_K),
                        lambda r, n: r.standard_normal((n, d)), h,
                        n_chains=700, t_max=int(4 * arl), seed=100 + d, state_shape=(d,))
    assert 0.7 * arl <= got <= 1.3 * arl, (d, arl, h, got)


def test_mcusum_generator_reproduces_table_low_arl():
    """The offline generator (_mcusum_arl_mc) agrees with the committed table
    at the 1e2 column (small run: 400 chains)."""
    for d in (1, 3, 8):
        h = Q.MCUSUM_H[d - 1, 0]
        arl, n = Q._mcusum_arl_mc(d, [h - 1.0, h, h + 1.0], n_chains=400, t_max=1500,
                                  seed=7 + d)
        assert n[1] == 400
        assert 0.85 * 100.0 <= arl[1] <= 1.15 * 100.0, (d, arl)
        assert arl[0] < arl[1] < arl[2]


# --------------------------------------------------------------- dispatcher
def test_h_for_dispatch():
    assert Q.h_for("gauss", 2400, 900) == pytest.approx(Q.h_gauss(0.5, 230400.0))
    assert Q.h_for("llr", 1, 60) == pytest.approx(math.log(1440.0))
    assert Q.h_for("mcusum", 100, 60, d=3) == pytest.approx(Q.mcusum_h(3, 144000.0))
    with pytest.raises(ValueError):
        Q.h_for("ewma", 100, 900)
    with pytest.raises(ValueError):
        Q.h_for("gauss", 100, 0)


# ------------------------------------------------------------------ trend
def _mk_reference(x):
    """Brute-force Mann-Kendall (independent loops) for cross-checking."""
    x = [v for v in x if math.isfinite(v)]
    n = len(x)
    s = sum(int(x[j] > x[i]) - int(x[j] < x[i]) for i in range(n) for j in range(i + 1, n))
    counts = {}
    for v in x:
        counts[v] = counts.get(v, 0) + 1
    tie = sum(t * (t - 1) * (2 * t + 5) for t in counts.values())
    var = (n * (n - 1) * (2 * n + 5) - tie) / 18.0
    z = 0.0 if s == 0 else (s - (1 if s > 0 else -1)) / math.sqrt(var)
    return float(s), math.erfc(abs(z) / math.sqrt(2.0))


def test_mann_kendall_known_values():
    s, p = Q.mann_kendall([1, 2, 3, 4, 5])
    assert s == 10.0
    assert p == pytest.approx(math.erfc((9 / math.sqrt(50 / 3)) / math.sqrt(2)))
    assert p == pytest.approx(0.0275, abs=5e-4)
    s2, p2 = Q.mann_kendall([5, 4, 3, 2, 1])
    assert s2 == -10.0 and p2 == pytest.approx(p)
    assert Q.mann_kendall([1, 2, 3]) == (0.0, 1.0)            # n < 4
    assert Q.mann_kendall([1, NAN, 2, NAN, 3]) == (0.0, 1.0)  # 3 finite
    assert Q.mann_kendall([2.0] * 10) == (0.0, 1.0)           # all tied
    s3, p3 = Q.mann_kendall([1, 1, 2, 1])                    # one tie group of 3
    assert s3 == 1.0 and p3 == 1.0                           # continuity-corrected z = 0
    s4, p4 = Q.mann_kendall([1, 2, 1, 2])                    # Var = (156 - 2 * 18) / 18
    assert s4 == 2.0 and p4 == pytest.approx(math.erfc(1.0 / math.sqrt(120 / 18) / math.sqrt(2)))


def test_mann_kendall_matches_bruteforce_with_ties_and_nan():
    rng = np.random.default_rng(15)
    for _ in range(20):
        n = int(rng.integers(4, 31))
        x = np.round(rng.standard_normal(n) + 0.05 * np.arange(n), 1)   # ties
        x[rng.random(n) < 0.1] = NAN
        s, p = Q.mann_kendall(x)
        s_ref, p_ref = _mk_reference(list(x))
        if sum(math.isfinite(v) for v in x) < 4:
            assert (s, p) == (0.0, 1.0)
            continue
        assert s == s_ref and p == pytest.approx(p_ref, rel=1e-12)


def test_mann_kendall_null_size_and_power():
    rng = np.random.default_rng(16)
    rej = np.mean([Q.mann_kendall(rng.standard_normal(14))[1] < 0.05 for _ in range(1500)])
    assert 0.025 <= rej <= 0.075
    trend = 0.5 * np.arange(14.0)
    pw = np.mean([Q.mann_kendall(trend + rng.standard_normal(14))[1] < 0.01
                  for _ in range(200)])
    assert pw > 0.9


def test_sen_slope():
    t = np.arange(14.0)
    x = 2.0 * t + 1.0
    x[3] = 100.0
    x[10] = -50.0                                         # outliers do not move the median
    assert Q.sen_slope(x) == pytest.approx(2.0)
    tt = 86400.0 * np.arange(6)
    assert Q.sen_slope(0.05 * tt / 86400.0, tt / 86400.0) == pytest.approx(0.05)
    assert Q.sen_slope([1.0, NAN, 3.0, 4.0]) == pytest.approx(1.0)
    assert Q.sen_slope([1.0, 2.0, 5.0], [0.0, 0.0, 1.0]) == pytest.approx(3.5)   # t tie skipped
    assert math.isnan(Q.sen_slope([1.0]))
    assert math.isnan(Q.sen_slope([1.0, 2.0], [3.0, 3.0]))
    assert math.isnan(Q.sen_slope([]))
    with pytest.raises(ValueError):
        Q.sen_slope([1.0, 2.0, 3.0], [0.0, 1.0])
    # b14 creep rule: |slope| > 0.05 log-units / day over 14 daily values
    rng = np.random.default_rng(17)
    y = 0.08 * np.arange(14) + 0.05 * rng.standard_normal(14)
    s, p = Q.mann_kendall(y)
    assert p < 0.01 and Q.sen_slope(y) > 0.05


def test_constants_contract():
    assert 2 * Q.SIEGMUND_RHO == pytest.approx(1.166)
    assert Q.AR1_PHI_CLIP == (0.0, 0.8) and Q.PSI_CLIP == 3.0
    assert Q.RHYTHM_P0_MAX == 0.3 and Q.MCUSUM_K == 0.5
    assert Q.MCUSUM_ARL_GRID == (1e2, 1e3, 1e4, 1e5, 1e6, 1e7)
