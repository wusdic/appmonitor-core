"""Tests for engines/behavior/lib/evt.py: PWM-GPD / POT tails (B13, B24), the
Gamma renewal LRT and its finite-n null table (B12), the window-corrected
Z^2 scan with its upcrossing trials correction, and Rayleigh (B11).

Statistical checks use fixed seeds (deterministic). The targets come from
docs/lib3/engines.md (B11 (c), B12 (a)-(d)) and the implementing brief:
PWM recovers xi within 0.1 on 2000 samples; +-30% jitter at P = 300 s with
n = 12 intervals gives median p < 1e-6; a Poisson null with n in {12, 20, 40}
has < 0.5% of p < 1e-3 over 1000 streams. The whole file runs in ~2 s.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
from scipy import special, stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from app.engines.behavior.lib import evt as E  # noqa: E402

NAN = float("nan")


# ------------------------------------------------------------------ oracles
def ref_pwm(y):
    """gpd_pwm_fit exactly as its docstring states."""
    y = np.sort(np.asarray(y, dtype=np.float64))
    n = y.size
    p = (np.arange(1, n + 1) - 0.35) / n
    a0 = y.mean()
    a1 = np.mean((1 - p) * y)
    xi = 2 - a0 / (a0 - 2 * a1)
    sigma = 2 * a0 * a1 / (a0 - 2 * a1)
    if not -0.5 <= xi <= 0.5:
        xi = min(0.5, max(-0.5, xi))
        sigma = a0 * (1 - xi)
    return xi, sigma


def ref_z2(t, periods, m=2):
    """Window- and endpoint-corrected Z^2_m by explicit loops (docstring formula)."""
    t = np.asarray(t, dtype=np.float64)
    t = t - t.min()
    n, span = t.size, t.max()
    out = []
    for per in periods:
        tot = 0.0
        for k in range(1, m + 1):
            s = np.exp(2j * np.pi * k * t / per).sum()
            theta = 2 * np.pi * k * span / per
            w = (np.exp(1j * theta) - 1) / (1j * theta)
            tot += abs(s - 1 - np.exp(1j * theta) - (n - 2) * w) ** 2
        out.append(2.0 / max(n - 2, 1) * tot)
    return np.array(out)


def null_p(intervals) -> float:
    kappa, lr = E.gamma_renewal_lrt(intervals)
    return E.beacon_null_p(lr, len(intervals))


# ======================================================================= GPD
@pytest.mark.parametrize("xi", [-0.3, 0.0, 0.2, 0.4])
def test_pwm_recovers_xi_and_sigma_on_2000_samples(xi):
    rng = np.random.default_rng(int(1000 * (xi + 1)))
    y = stats.genpareto.rvs(xi, scale=2.0, size=2000, random_state=rng)
    xi_hat, sigma_hat = E.gpd_pwm_fit(y)
    assert abs(xi_hat - xi) < 0.1
    assert sigma_hat == pytest.approx(2.0, rel=0.1)


def test_pwm_matches_docstring_formula_and_is_order_free():
    rng = np.random.default_rng(3)
    y = rng.exponential(1.7, 57)
    xi, sigma = E.gpd_pwm_fit(y)
    want = ref_pwm(y)
    assert (xi, sigma) == pytest.approx(want, rel=1e-12)
    assert E.gpd_pwm_fit(rng.permutation(y)) == pytest.approx((xi, sigma), rel=1e-12)
    assert E.gpd_pwm_fit(list(y)) == pytest.approx((xi, sigma), rel=1e-12)


def test_pwm_clips_xi_and_reestimates_sigma_from_the_mean():
    rng = np.random.default_rng(5)
    heavy = stats.genpareto.rvs(0.9, size=4000, random_state=rng)
    xi, sigma = E.gpd_pwm_fit(heavy)
    assert xi == 0.5 and sigma == pytest.approx(heavy.mean() * 0.5, rel=1e-12)
    bounded = rng.uniform(0, 1, 4000)            # GPD xi = -1
    xi, sigma = E.gpd_pwm_fit(bounded)
    assert xi == -0.5 and sigma == pytest.approx(bounded.mean() * 1.5, rel=1e-12)


def test_pwm_fallbacks_and_nan_handling():
    assert E.gpd_pwm_fit([]) == (0.0, 1e-12)
    assert E.gpd_pwm_fit([1.0, 2.0]) == (0.0, 1.5)           # n < 3: exponential
    assert E.gpd_pwm_fit([0.0, 0.0, 0.0, 0.0]) == (0.0, 1e-12)
    assert E.gpd_pwm_fit([-1.0, -2.0, 0.5]) == (0.0, 1e-12)  # a0 <= 0
    y = [0.3, 1.2, 0.1, 2.5, 0.7]
    assert E.gpd_pwm_fit(y + [NAN, math.inf, -math.inf]) == E.gpd_pwm_fit(y)
    assert E.gpd_pwm_fit([NAN, 1.0, 2.0, NAN]) == (0.0, 1.5)  # 2 finite left
    xi, sigma = E.gpd_pwm_fit(np.full(10, 3.0))              # all equal: bounded -> clip
    assert xi == -0.5 and sigma == pytest.approx(4.5)


@pytest.mark.parametrize("xi", [-0.45, -0.2, 0.0, 1e-10, 0.1, 0.5])
def test_gpd_sf_matches_scipy(xi):
    y = np.array([0.0, 1e-9, 0.3, 1.0, 1.9, 2.5, 10.0, 1e3])
    got = E.gpd_sf(y, xi, 0.9)
    want = stats.genpareto.sf(y, xi, scale=0.9)
    rel = 1e-9 if abs(xi) >= 1e-9 else 1e-4
    np.testing.assert_allclose(got, want, rtol=rel, atol=1e-300)


def test_gpd_sf_edges_shape_and_invalid_parameters():
    y = np.array([[-1.0, 0.0], [NAN, 5.0]])
    out = E.gpd_sf(y, -0.25, 1.0)                           # end point at 4
    assert out.shape == (2, 2)
    assert out[0, 0] == 1.0 and out[0, 1] == 1.0
    assert math.isnan(out[1, 0]) and out[1, 1] == 0.0
    assert E.gpd_sf(4.0 - 1e-9, -0.25, 1.0) > 0.0
    assert np.ndim(E.gpd_sf(2.0, 0.1, 1.0)) == 0
    assert float(E.gpd_sf(2.0, 0.1, 1.0)) == pytest.approx(stats.genpareto.sf(2.0, 0.1), rel=1e-12)
    bad = E.gpd_sf(np.array([-1.0, 1.0]), 0.1, 0.0)
    assert bad[0] == 1.0 and math.isnan(bad[1])
    assert math.isnan(E.gpd_sf(np.array([1.0]), NAN, 1.0)[0])
    # far tail stays finite and positive in log space
    assert 0.0 < E.gpd_sf(np.array([1e12]), 0.5, 1.0)[0] < 1e-11


def test_gpd_sf_agrees_with_calib_scalar_twin():
    from app.engines.behavior.lib import calib as K
    ys = np.array([0.0, 0.1, 1.0, 3.0, 30.0, 300.0])
    for xi in (-0.5, -0.1, 0.0, 0.2, 0.5):
        got = np.array([K._gpd_sf(float(v), xi, 1.3) for v in ys])
        np.testing.assert_allclose(got, E.gpd_sf(ys, xi, 1.3), rtol=1e-12, atol=1e-300)


# ======================================================================= POT
@pytest.mark.parametrize("xi", [-0.3, 0.0, 0.3])
def test_pot_quantile_inverts_pot_sf(xi):
    u, sigma, rate = 10.0, 2.0, 0.02
    for q in (1e-2, 1e-4, 1e-7):
        z = E.pot_quantile(u, xi, sigma, rate, q)
        assert z > u
        assert float(E.pot_sf(np.array([z]), u, xi, sigma, rate)[0]) == pytest.approx(q, rel=1e-9,
                                                                                    abs=0)
    want = u + sigma * math.log(rate / 1e-4) if xi == 0.0 else \
        u + sigma / xi * ((rate / 1e-4) ** xi - 1)
    assert E.pot_quantile(u, xi, sigma, rate, 1e-4) == pytest.approx(want, rel=1e-12)


def test_pot_quantile_edges():
    assert E.pot_quantile(5.0, 0.2, 1.0, 0.02, 0.02) == 5.0      # q >= rate: body
    assert E.pot_quantile(5.0, 0.2, 1.0, 0.02, 0.5) == 5.0
    assert E.pot_quantile(5.0, -0.25, 1.0, 0.02, 0.0) == pytest.approx(9.0)  # end point
    assert E.pot_quantile(5.0, 0.2, 1.0, 0.02, 0.0) == math.inf
    assert E.pot_quantile(5.0, 0.0, 1.0, 0.02, 0.0) == math.inf
    assert math.isnan(E.pot_quantile(NAN, 0.2, 1.0, 0.02, 1e-3))
    assert math.isnan(E.pot_quantile(5.0, 0.2, 1.0, 0.02, NAN))
    # xi within 1e-9 of 0 uses the exponential formula without blowing up
    got = E.pot_quantile(5.0, 1e-12, 1.0, 0.02, 2e-5)
    assert got == pytest.approx(5.0 + math.log(1e3), rel=1e-9)
    # regression: a large xi * ln(rate/q) raised OverflowError from math.expm1
    assert E.pot_quantile(5.0, 2.0, 1.0, 1.0, 1e-300) == math.inf
    # regression: rate is clipped as in pot_sf, so the two stay inverse
    z = E.pot_quantile(0.0, 0.1, 2.0, 7.0, 1e-3)
    assert E.pot_sf(np.array([z]), 0.0, 0.1, 2.0, 7.0)[0] == pytest.approx(1e-3, rel=1e-9)
    # invalid sigma -> NaN (gpd_sf gives NaN there too), not a finite level
    assert math.isnan(E.pot_quantile(5.0, 0.2, 0.0, 0.02, 1e-3))
    assert math.isnan(E.pot_quantile(5.0, 0.2, -1.0, 0.02, 1e-3))


def test_pot_sf_body_nan_and_rate_clip():
    x = np.array([0.0, 10.0, 12.0, NAN])
    out = E.pot_sf(x, 10.0, 0.1, 2.0, 0.05)
    assert out[0] == 1.0 and out[1] == 1.0                     # x <= u: no tail evidence
    assert out[2] == pytest.approx(0.05 * stats.genpareto.sf(2.0, 0.1, scale=2.0), rel=1e-12)
    assert math.isnan(out[3])
    assert E.pot_sf(np.array([12.0]), 10.0, 0.1, 2.0, 7.0)[0] == pytest.approx(
        stats.genpareto.sf(2.0, 0.1, scale=2.0), rel=1e-12)   # rate clipped to 1
    assert np.all(np.isnan(E.pot_sf(x, NAN, 0.1, 2.0, 0.05)))
    assert np.all(np.isnan(E.pot_sf(x, 10.0, 0.1, 2.0, NAN)))


def test_pot_quantile_budget_threshold_is_calibrated():
    # B13 use: u = P98 of history, PWM fit to the excesses, then z_q. On a
    # 5000-day Gamma history the realised exceedance of z_1e-3 is ~1e-3.
    rng = np.random.default_rng(13)
    hist = rng.gamma(3.0, 2.0, 5000)
    u = float(np.quantile(hist, 0.98))
    exc = hist[hist > u] - u
    xi, sigma = E.gpd_pwm_fit(exc)
    z = E.pot_quantile(u, xi, sigma, exc.size / hist.size, 1e-3)
    realised = stats.gamma.sf(z, 3.0, scale=2.0)
    assert 3e-4 < realised < 3e-3


# ======================================================== Gamma renewal LRT
@pytest.mark.parametrize("kappa", [3.0, 30.0])
def test_lrt_is_the_gamma_mle_and_the_likelihood_ratio(kappa):
    rng = np.random.default_rng(int(kappa))
    x = rng.gamma(kappa, 7.0, 60)
    k_hat, lr = E.gamma_renewal_lrt(x)
    s = math.log(x.mean()) - np.mean(np.log(x))
    assert math.log(k_hat) - special.psi(k_hat) == pytest.approx(s, rel=1e-10)
    ll1 = stats.gamma.logpdf(x, k_hat, scale=x.mean() / k_hat).sum()
    ll0 = stats.expon.logpdf(x, scale=x.mean()).sum()
    assert lr == pytest.approx(2 * (ll1 - ll0), rel=1e-9)
    a, _, _ = stats.gamma.fit(x, floc=0)
    assert k_hat == pytest.approx(a, rel=1e-6)


def test_lrt_is_one_sided_and_scale_free():
    rng = np.random.default_rng(4)
    bursty = rng.lognormal(0.0, 1.5, 50)
    k_hat, lr = E.gamma_renewal_lrt(bursty)
    assert k_hat < 1.0 and lr == 0.0
    x = rng.gamma(5.0, 1.0, 30)
    for c in (1e-2, 1.0, 3600.0):
        assert E.gamma_renewal_lrt(c * x) == pytest.approx(E.gamma_renewal_lrt(x), rel=1e-9)


def test_lrt_degenerate_inputs():
    k, lr = E.gamma_renewal_lrt([300.0] * 12)
    assert k == E.KAPPA_MAX and lr > 10 * 12
    k2, lr2 = E.gamma_renewal_lrt([300.0] * 24)
    assert lr2 == pytest.approx(2 * lr, rel=1e-9)                 # LR ~ n at the cap
    assert all(math.isnan(v) for v in E.gamma_renewal_lrt([1.0, 2.0]))
    assert all(math.isnan(v) for v in E.gamma_renewal_lrt([]))
    assert all(math.isnan(v) for v in E.gamma_renewal_lrt([1.0, NAN, 2.0, math.inf]))
    x = [1.0, 2.0, 3.0, 4.0]
    assert E.gamma_renewal_lrt(x + [NAN]) == E.gamma_renewal_lrt(x)
    # zero / negative intervals are clipped to 1 ms, never log(0)
    k, lr = E.gamma_renewal_lrt([0.0, -5.0, 300.0, 310.0, 290.0])
    assert math.isfinite(k) and math.isfinite(lr)
    assert E.gamma_renewal_lrt([0.0, 1.0, 2.0]) == E.gamma_renewal_lrt([1e-3, 1.0, 2.0])


def test_scalar_and_vector_lr_paths_agree_and_lr_is_monotone_in_s():
    s = np.concatenate([[0.0, 1e-9, E._S_AT_KAPPA_MAX, 2 * E._S_AT_KAPPA_MAX],
                        np.geomspace(1e-6, 3.0, 200)])
    for n in (12, 100):
        k_v, lr_v = E._kappa_lr(s, n)
        sc = np.array([E._kappa_lr_scalar(float(v), n) for v in s])
        np.testing.assert_allclose(sc[:, 0], k_v, rtol=1e-12)
        np.testing.assert_allclose(sc[:, 1], lr_v, rtol=1e-12, atol=1e-12)
        order = np.argsort(s)
        assert np.all(np.diff(lr_v[order]) <= 1e-9)            # decreasing in s
        assert np.all(lr_v[s >= np.euler_gamma] == 0.0)          # kappa <= 1 <=> s >= gamma
    # continuity across the kappa cap
    a = E._kappa_lr_scalar(E._S_AT_KAPPA_MAX * (1 - 1e-9), 20)[1]
    b = E._kappa_lr_scalar(E._S_AT_KAPPA_MAX * (1 + 1e-9), 20)[1]
    assert a == pytest.approx(b, rel=1e-6)


# ============================================================ null table
def test_null_table_layout_and_monotonicity():
    tab = E.BEACON_NULL_LR
    assert tab.shape == (len(E.BEACON_NULL_N), len(E.BEACON_NULL_LOG10P))
    assert np.all(np.isfinite(tab)) and np.all(tab > 0)
    assert np.all(np.diff(tab, axis=1) > 0)                     # rarer p -> larger LR
    assert np.all(np.diff(tab, axis=0) < 0)                     # row rule is conservative
    assert list(E.BEACON_NULL_N) == sorted(E.BEACON_NULL_N)
    assert E.BEACON_NULL_N[0] == 8 and E.BEACON_NULL_N[-1] == 256
    assert {8, 9, 10, 11, 12, 20, 40} <= set(E.BEACON_NULL_N)    # exact rows at small n
    assert E.BEACON_NULL_LOG10P[0] == pytest.approx(math.log10(0.5), abs=1e-5)
    assert np.all(np.diff(E.BEACON_NULL_LOG10P) < 0)
    assert not tab.flags.writeable
    # the finite-n quantile exceeds the chi2_1 one-sided asymptotic (2-5x anti-conservative)
    chi2_q = stats.chi2.isf(2 * 1e-5, 1)
    assert tab[E.BEACON_NULL_N.index(12), E.BEACON_NULL_LOG10P.index(-5.0)] > chi2_q + 2.0


@pytest.mark.parametrize("n, cols", [(8, (1, 24)), (12, (0, 6, 24)), (256, (2, 12))])
def test_null_table_matches_exact_inversion(n, cols):
    lp = [E.BEACON_NULL_LOG10P[j] for j in cols]
    exact = E._exact_null_lr_quantiles(n, lp)
    row = E.BEACON_NULL_LR[E.BEACON_NULL_N.index(n)]
    np.testing.assert_allclose(row[list(cols)], exact, rtol=1e-6, atol=1e-6)


def test_exact_null_cdf_agrees_with_simulation_on_both_sides_of_the_mean():
    n = 16
    rng = np.random.default_rng(21)
    tt = n * E._log_ratio_stat(rng.standard_exponential((60_000, n)))
    mu = E._null_cgf_d1(0.0, n)
    for t in (0.35 * mu, 0.7 * mu, mu * (1 - 1e-9), mu * (1 + 1e-9), 1.4 * mu):
        p = math.exp(E._null_log_cdf_T(t, n))
        emp = np.mean(tt <= t)
        assert abs(emp - p) < 4.5 * math.sqrt(p * (1 - p) / tt.size) + 1e-4, (t, p, emp)
    # continuity where the lower and upper contours hand over
    lo = E._null_log_cdf_T(mu * (1 - 1e-12), n)
    hi = E._null_log_cdf_T(mu * (1 + 1e-12), n)
    assert lo == pytest.approx(hi, abs=1e-9)
    assert E._null_log_cdf_T(0.0, n) == -math.inf


def test_beacon_null_p_knots_rows_and_edges():
    tab = E.BEACON_NULL_LR
    for i in (0, 4, 9, 16):
        for j in (0, 6, 24):
            got = E.beacon_null_p(tab[i, j], E.BEACON_NULL_N[i])
            assert got == pytest.approx(10 ** E.BEACON_NULL_LOG10P[j], rel=1e-9, abs=0)
    assert E.beacon_null_p(0.0, 20) == 1.0 and E.beacon_null_p(-3.0, 20) == 1.0
    assert 0.5 < E.beacon_null_p(1e-6, 20) <= 1.0              # anchor (0, 1): conservative
    assert math.isnan(E.beacon_null_p(NAN, 20))
    assert math.isnan(E.beacon_null_p(20.0, 7))
    assert math.isfinite(E.beacon_null_p(20.0, 11))
    assert math.isnan(E.beacon_null_p(20.0, NAN))
    assert math.isnan(E.beacon_null_p(20.0, None))
    # between grid rows the smaller n' is used (conservative), beyond 256 row 256
    assert E.beacon_null_p(20.0, 13) == E.beacon_null_p(20.0, 12)
    assert E.beacon_null_p(20.0, 13) > E.beacon_null_p(20.0, 14)
    assert E.beacon_null_p(20.0, 5000) == E.beacon_null_p(20.0, 256)
    assert E.beacon_null_p(20.0, 40.9) == E.beacon_null_p(20.0, 40)


def _s_of_lr(lr: float, n: int) -> float:
    from scipy import optimize
    return optimize.brentq(lambda s: E._kappa_lr_scalar(s, n)[1] - lr, 1e-15, np.euler_gamma,
                           xtol=1e-17, rtol=1e-14, maxiter=500)


def test_beacon_null_p_monotone_and_tail_extrapolation():
    lrs = np.linspace(0.0, 120.0, 2401)
    for n in (8, 12, 48, 256):
        ps = np.array([E.beacon_null_p(v, n) for v in lrs])
        assert np.all(np.diff(ps) <= 0) and ps[0] == 1.0
    n = 12
    row = E.BEACON_NULL_LR[E.BEACON_NULL_N.index(n)]
    last, p_last = row[-1], 10 ** E.BEACON_NULL_LOG10P[-1]
    assert E.beacon_null_p(last + 1e-9, n) == pytest.approx(p_last, rel=1e-6, abs=0)  # continuous
    want = p_last * math.exp(-(n - 1) / (2 * n) * 10.0)          # asymptotic slope of ln p
    assert E.beacon_null_p(last + 10, n) == pytest.approx(want, rel=1e-9, abs=0)
    assert E.beacon_null_p(1e5, n) == E.P_FLOOR
    assert E.beacon_null_p(math.inf, n) == E.P_FLOOR


@pytest.mark.parametrize("n", [8, 12, 40, 256])
def test_beacon_null_p_beyond_the_table_is_conservative_and_tight(n):
    # regression: the chi2_1 tail shape used beyond LR(1e-12) decays faster
    # than the exact null (ln p ~ -(n-1)/(2n) LR as s -> 0) and was up to 0.9
    # decades anti-conservative at n = 8, LR_last + 30.
    row = E.BEACON_NULL_LR[E.BEACON_NULL_N.index(n)]
    for d in (5.0, 15.0, 30.0):
        lr = row[-1] + d
        exact = E._null_log_cdf_T(n * _s_of_lr(lr, n), n) / math.log(10)
        got = math.log10(E.beacon_null_p(lr, n))
        assert exact - 1e-6 <= got <= exact + 0.06, (n, d, got, exact)


@pytest.mark.parametrize("n", [12, 20, 40])
def test_poisson_null_validity_1000_streams(n):
    # brief target / B12 (c): fewer than 0.5% of Poisson streams give p < 1e-3
    rng = np.random.default_rng(100 + n)
    ps = np.array([null_p(rng.exponential(300.0, n)) for _ in range(1000)])
    assert np.mean(ps < 1e-3) < 0.005
    assert np.all((ps > 0) & (ps <= 1))


def test_poisson_null_is_calibrated_not_just_conservative():
    # 40k null draws per n: realised P(p <= a) within 4.5 binomial sd of a,
    # from both sides (a conservative table would fail the lower bound).
    for n in (9, 12, 20, 40, 100):
        lr = E._mc_null_lr(n, 40_000, np.random.default_rng([7, n]))
        idx = E.BEACON_NULL_N.index(max(v for v in E.BEACON_NULL_N if v <= n))
        exact_row = n in E.BEACON_NULL_N
        for a, lp in ((0.1, -1.0), (0.01, -2.0), (1e-3, -3.0)):
            hits = int(np.sum(lr >= E.BEACON_NULL_LR[idx, E.BEACON_NULL_LOG10P.index(lp)]))
            sd = math.sqrt(lr.size * a * (1 - a))
            assert hits <= lr.size * a + 4.5 * sd, (n, a, hits)
            if exact_row:
                assert hits >= lr.size * a - 4.5 * sd, (n, a, hits)


@pytest.mark.parametrize("n", [11, 12])
def test_jittered_beacon_power_at_n12(n):
    # brief target / B12: one request every 300 s +- 30% (uniform). n = 12
    # intervals (brief) and n = 11, i.e. 12 events (generator T10: 'p < 1e-6
    # after 12 events'): median p < 1e-6 either way.
    rng = np.random.default_rng(300 + n)
    ps = np.array([null_p(300.0 * rng.uniform(0.7, 1.3, n)) for _ in range(300)])
    assert np.median(ps) < 1e-6


def test_jittered_beacon_power_at_20_events():
    rng = np.random.default_rng(300)
    ps20 = np.array([null_p(300.0 * rng.uniform(0.7, 1.3, 19)) for _ in range(100)])
    assert np.max(ps20) < 1e-6                     # B12 (a): 20 events -> p < 1e-6


def test_human_lognormal_gaps_are_not_beacons():
    # B12 (d): log-normal gaps with sigma = 1: fewer than 0.5% have p < 1e-4
    rng = np.random.default_rng(8)
    ps = np.array([null_p(rng.lognormal(math.log(8.0), 1.0, 39)) for _ in range(1000)])
    assert np.mean(ps < 1e-4) < 0.005


def test_build_beacon_null_table_mc_agrees_and_is_deterministic():
    grid, lps, sims = (12, 40), (-0.30103, -1.0, -2.0, -3.0, -4.0), 20_000
    a = E.build_beacon_null_table(grid, lps, n_sims=sims, seed=5)
    b = E.build_beacon_null_table(grid, lps, n_sims=sims, seed=5)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (2, 5)
    assert np.all(np.isnan(a[:, 4]))                 # 20000 * 1e-4 < 10: unresolvable
    # the row for n = 40 does not depend on the rest of the grid
    np.testing.assert_array_equal(E.build_beacon_null_table((40,), lps, sims, 5)[0], a[1])
    for i, n in enumerate(grid):
        for j, lp in enumerate(lps[:4]):
            p_true = E.beacon_null_p(a[i, j], n)     # exact tail at the MC quantile
            tol = 4.5 / math.sqrt(sims * 10 ** lp) / math.log(10) + 0.03
            assert abs(math.log10(p_true) - lp) < tol, (n, lp, p_true)


# ============================================================ periodicity
def test_z2_trial_periods_grid_and_effective_trials():
    span = 800.0
    per, n_eff = E.z2_trial_periods(span)            # band [span/4, 10 s], 381 trials
    f = 1.0 / per
    assert per[0] == pytest.approx(span / 4) and np.all(np.diff(per) < 0)
    np.testing.assert_allclose(np.diff(f), 1.0 / (5 * span), rtol=1e-9)
    assert f[-1] <= 0.1 + 1e-12 and 0.1 - f[-1] < 1.0 / (5 * span)
    assert n_eff == pytest.approx((f[-1] - f[0]) * span, rel=1e-12)
    assert len(per) == int(round((0.1 - 4 / span) * 5 * span)) + 1 < E.Z2_MAX_TRIALS
    # explicit band and oversampling
    per, n_eff = E.z2_trial_periods(3000.0, p_min_s=250.0, p_max_s=350.0, oversample=10)
    assert per.max() == pytest.approx(350.0) and per.min() >= 250.0
    np.testing.assert_allclose(np.diff(1 / per), 1 / (10 * 3000.0), rtol=1e-9)
    assert n_eff == pytest.approx((1 / per.min() - 1 / 350.0) * 3000.0, rel=1e-12)


def test_z2_trial_periods_cap_coarsens_then_truncates():
    # 196 Fourier frequencies: 5x would need 981 trials -> coarsen to fit 512
    per, n_eff = E.z2_trial_periods(2000.0)
    f = 1.0 / per
    assert len(per) == E.Z2_MAX_TRIALS
    assert f[0] == pytest.approx(4 / 2000.0) and f[-1] == pytest.approx(0.1, rel=1e-9)
    assert n_eff == pytest.approx(196.0, rel=1e-9)
    assert 2.0 <= 1.0 / (np.diff(f).mean() * 2000.0) < 5.0
    # 8636 Fourier frequencies: oversample floors at 2, grid keeps the long-period end
    per, n_eff = E.z2_trial_periods(86400.0)
    f = 1.0 / per
    assert len(per) == E.Z2_MAX_TRIALS
    np.testing.assert_allclose(np.diff(f), 1 / (E.Z2_MIN_OVERSAMPLE * 86400.0), rtol=1e-9)
    assert f[0] == pytest.approx(4 / 86400.0)
    assert n_eff == pytest.approx(511 / E.Z2_MIN_OVERSAMPLE, rel=1e-9)
    # a caller asking for less oversampling than the floor keeps its request
    per, n_eff = E.z2_trial_periods(86400.0, oversample=1)
    np.testing.assert_allclose(np.diff(1 / per), 1 / 86400.0, rtol=1e-9)
    assert n_eff == pytest.approx(511.0, rel=1e-9)


def test_z2_trial_periods_degenerate_inputs():
    for args in ((0.0,), (-5.0,), (NAN,), (30.0,), (1000.0, 10.0, 5.0), (1000.0, 0.0),
                 (1000.0, 10.0, NAN)):
        per, n_eff = E.z2_trial_periods(*args)
        assert per.size == 0 and n_eff == 0.0
    per, n_eff = E.z2_trial_periods(1000.0, 100.0, 100.0)      # single trial
    assert per.size == 1 and n_eff == 0.0


def test_z2_periodogram_matches_loop_reference_on_both_paths():
    rng = np.random.default_rng(2)
    t = 1.7e9 + np.sort(rng.uniform(0, 5000, 23))
    per, _ = E.z2_trial_periods(t.max() - t.min())
    got = E.z2_periodogram(t, per[::7])                           # uniform: factorised path
    np.testing.assert_allclose(got, ref_z2(t, per[::7]), rtol=1e-9, atol=1e-9)
    odd = rng.uniform(20, 900, 17)                                # arbitrary: direct path
    np.testing.assert_allclose(E.z2_periodogram(t, odd, m=3), ref_z2(t, odd, 3),
                               rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(E.z2_periodogram(t, odd, m=1), ref_z2(t, odd, 1),
                               rtol=1e-9, atol=1e-9)


def test_z2_uniform_grid_takes_the_fast_path_and_equals_direct(monkeypatch):
    rng = np.random.default_rng(9)
    t = np.sort(rng.uniform(0, 86400, 256))
    per, _ = E.z2_trial_periods(86400.0)
    direct = np.concatenate([E.z2_periodogram(t, per[a:a + 2]) for a in range(0, per.size, 2)])

    def boom(*_a, **_k):
        raise AssertionError("uniform grid fell back to the O(n L) path")

    monkeypatch.setattr(E, "_z2_sums_direct", boom)
    fast = E.z2_periodogram(t, per)
    np.testing.assert_allclose(fast, direct, rtol=1e-9, atol=1e-9)


def test_z2_periodogram_properties_and_edges():
    rng = np.random.default_rng(6)
    t = np.sort(rng.uniform(0, 3000, 30))
    per = np.array([100.0, 250.0, 333.3])
    base = E.z2_periodogram(t, per)
    np.testing.assert_allclose(E.z2_periodogram(t + 12345.678, per), base, rtol=1e-9)
    np.testing.assert_allclose(E.z2_periodogram(np.r_[t, NAN], per), base, rtol=1e-12)
    # a perfect train covering whole cycles: Z^2_m = 2 (n - 2) m at the true
    # period (the two endpoints carry no information: they define the window)
    train = 300.0 * np.arange(20)
    assert E.z2_periodogram(train, [300.0])[0] == pytest.approx(2 * 18 * 2, rel=1e-9)
    assert E.z2_periodogram(train, [300.0], m=3)[0] == pytest.approx(2 * 18 * 3, rel=1e-9)
    np.testing.assert_allclose(E.z2_periodogram([3.0, 700.0], per), 0.0, atol=1e-12)  # n = 2
    out = E.z2_periodogram(t, [100.0, -1.0, NAN, 0.0])
    assert math.isfinite(out[0]) and np.all(np.isnan(out[1:]))
    assert np.all(np.isnan(E.z2_periodogram([5.0], per)))
    assert E.z2_periodogram(t, []).size == 0
    with pytest.raises(ValueError):
        E.z2_periodogram(t, per, m=0)


def test_z2_window_correction_removes_low_frequency_bias():
    # Poisson times at 4.5 cycles per window: uncorrected Z^2 is biased by
    # ~2 n |w|^2 per harmonic; the corrected one has mean ~ 2m = 4.
    rng = np.random.default_rng(12)
    n, span = 256, 1000.0
    per = [span / 4.5]
    z_corr, z_raw = [], []
    for _ in range(300):
        t = np.sort(rng.uniform(0, span, n))
        z_corr.append(E.z2_periodogram(t, per)[0])
        ph = 2 * np.pi * (t - t[0]) / per[0]
        z_raw.append(2 / n * (abs(np.exp(1j * ph).sum()) ** 2 + abs(np.exp(2j * ph).sum()) ** 2))
    assert 3.4 < np.mean(z_corr) < 4.6
    assert np.mean(z_raw) > np.mean(z_corr) + 1.5


def test_z2_endpoints_do_not_bias_the_null_at_whole_cycles():
    # regression: t_min and t_max define the window, so their phases are 0 and
    # theta by construction; subtracting only n w left a mean of 1 + e^(i theta)
    # per harmonic (2 at whole cycles): Z^2 mean 4.42 and 1.4x the chi2_4 tail
    # at n = 20. Conditioned on the endpoints the null is chi2_4 again.
    rng = np.random.default_rng(3)
    n, draws = 20, 10_000
    q = stats.chi2.isf(0.01, 4)
    z = np.array([E.z2_periodogram(t, [(t[-1] - t[0]) / 7.0])[0]
                  for t in (np.sort(rng.uniform(0, 1000.0, n)) for _ in range(draws))])
    assert abs(z.mean() - 4.0) < 0.12
    assert np.sum(z > q) <= 0.01 * draws + 3.5 * math.sqrt(draws * 0.01 * 0.99)


def test_z2_p_formula_and_edges():
    for m, z, w in ((1, 20.0, 50.0), (2, 35.0, 102.2), (3, 60.0, 7.0)):
        c = math.sqrt(math.pi * (m + 1) * (2 * m + 1) / 18)
        up = w * c * (z / 2) ** (m - 0.5) * math.exp(-z / 2) / math.gamma(m)
        want = stats.chi2.sf(z, 2 * m) + up
        assert E.z2_p(z, m, w) == pytest.approx(want, rel=1e-10)
    assert E.z2_p(30.0, 2, 0.0) == pytest.approx(stats.chi2.sf(30.0, 4), rel=1e-10)
    # with the grid size the union bound L * sf also applies; the smaller wins
    dav = E.z2_p(40.0, 2, 255.5)
    assert E.z2_p(40.0, 2, 255.5, 512) == pytest.approx(512 * stats.chi2.sf(40.0, 4), rel=1e-10)
    assert E.z2_p(40.0, 2, 255.5, 512) < dav / 2
    assert E.z2_p(40.0, 2, 10.0, 10_000) == E.z2_p(40.0, 2, 10.0)       # Davies smaller
    assert E.z2_p(40.0, 2, 255.5, 0) == dav and E.z2_p(40.0, 2, 255.5, None) == dav
    assert E.z2_p(2.0, 2, 100.0) == 1.0
    assert E.z2_p(0.0, 2, 10.0) == 1.0 and E.z2_p(-1.0, 2, 10.0) == 1.0
    assert E.z2_p(5000.0, 2, 100.0) == E.P_FLOOR
    assert math.isnan(E.z2_p(NAN, 2, 10.0)) and math.isnan(E.z2_p(10.0, 2, NAN))
    zs = np.linspace(0.1, 200, 400)
    ps = [E.z2_p(z, 2, 50.0) for z in zs]
    assert np.all(np.diff(ps) <= 0)
    # the upcrossing bound is stricter than the stub's Bonferroni at alarm levels
    assert E.z2_p(45.0, 2, 100.0) > 100.0 * stats.chi2.sf(45.0, 4) * 3


@pytest.mark.parametrize("n, span", [(20, 86400.0), (100, 86400.0), (40, 800.0)])
def test_z2_scan_is_valid_on_poisson_times(n, span):
    # default scans: 2x-oversampled 512-trial grid (long spans) and a 5x one
    # (span 800). The stub's Bonferroni was 3-4x anti-conservative here; the
    # Davies bound alone is valid (conservative on 2x grids) and min(Davies,
    # L sf) is ~calibrated (measured 0.8-1.1x at 1e-2..1e-4 on 4e4 streams).
    rng = np.random.default_rng(40 + n)
    per, n_eff = E.z2_trial_periods(span)
    zmax = [np.nanmax(E.z2_periodogram(np.sort(rng.uniform(0, span, n)), per))
            for _ in range(600)]
    p_dav = np.array([E.z2_p(z, 2, n_eff) for z in zmax])
    p_min = np.array([E.z2_p(z, 2, n_eff, len(per)) for z in zmax])
    assert np.all(p_min <= p_dav)
    for ps in (p_dav, p_min):
        assert np.mean(ps < 0.1) < 0.125
        assert np.mean(ps < 0.01) < 0.022
    assert np.mean(p_min < 0.1) > 0.03          # at most ~3x conservative


def test_z2_detects_missed_beat_beacons_the_renewal_test_misses():
    # a sleeping implant: 100 beats at 300 s (1 s phase jitter), 70% skipped,
    # plus 15 unrelated Poisson events to the same destination. Intervals are
    # multiples of P mixed with noise splits, so the renewal LRT sees ~Exp;
    # the phases stay coherent, so the Z^2 scan (default band) nails it.
    z2_hits = lrt_blind = 0
    for seed in range(20):
        rng = np.random.default_rng(500 + seed)
        beats = 300.0 * np.arange(100) + rng.normal(0, 1.0, 100)
        kept = beats[rng.uniform(size=100) > 0.7]
        t = np.sort(np.r_[kept, rng.uniform(0, beats[-1], 15)])
        per, n_eff = E.z2_trial_periods(t[-1] - t[0])
        z = E.z2_periodogram(t, per)
        j = int(np.nanargmax(z))
        p_z2 = E.z2_p(z[j], 2, n_eff, len(per))
        if p_z2 < 1e-6:
            z2_hits += 1
            assert min(abs(per[j] / 300.0 - 1), abs(per[j] / 150.0 - 1)) < 0.01
        lrt_blind += null_p(np.diff(t)) > 1e-3
    assert z2_hits >= 17 and lrt_blind >= 17


@pytest.mark.parametrize("n, alpha", [(12, 1e-4), (16, 1e-6)])
def test_z2_strict_train_alone(n, alpha):
    # low-jitter train at 300 s +- 0.5 s. Z^2_2 <= 2 (n - 2) m = 4 (n - 2)
    # caps the scan p (~2e-5 at n = 12 over ~255 Fourier frequencies): below
    # that it is the renewal LRT's job (p ~ 1e-12 for such a train).
    rng = np.random.default_rng(77 + n)
    t = 300.0 * np.arange(n) + rng.uniform(-0.5, 0.5, n)
    per, n_eff = E.z2_trial_periods(t[-1] - t[0])
    z = E.z2_periodogram(t, per)
    j = int(np.nanargmax(z))
    assert z[j] <= 4 * (n - 2) + 1e-9
    assert E.z2_p(z[j], 2, n_eff, len(per)) < alpha
    assert per[j] == pytest.approx(300.0, rel=0.02) or per[j] == pytest.approx(150.0, rel=0.02)
    assert null_p(np.diff(t)) < 1e-10


def test_rayleigh_matches_zar_formula_and_edges():
    rng = np.random.default_rng(1)
    t = rng.uniform(0, 5000, 40)
    per = 123.4
    ph = 2 * np.pi * t / per
    r2 = np.cos(ph).sum() ** 2 + np.sin(ph).sum() ** 2
    n = t.size
    want = math.exp(math.sqrt(1 + 4 * n + 4 * (n * n - r2)) - (1 + 2 * n))
    assert E.rayleigh_p(t, per) == pytest.approx(min(1.0, want), rel=1e-9)
    assert E.rayleigh_p(t + 1.7e9, per) == pytest.approx(E.rayleigh_p(t, per), rel=1e-6)
    assert E.rayleigh_p(np.r_[t, NAN], per) == E.rayleigh_p(t, per)
    perfect = E.rayleigh_p(37.0 * np.arange(20), 37.0)
    assert perfect == pytest.approx(math.exp(math.sqrt(81) - 41), rel=1e-6)
    assert math.isnan(E.rayleigh_p([1.0], 10.0))
    assert math.isnan(E.rayleigh_p(t, 0.0)) and math.isnan(E.rayleigh_p(t, NAN))
    assert 0.0 <= E.rayleigh_p([0.0, 5.0], 10.0) <= 1.0          # antipodal pair


def test_rayleigh_strict_train_b11_and_calibration_under_uniform_phases():
    # B11 (c): a 37 s +- 0.3 s train over 2 h -> best period in 35-40 s, p < 1e-4
    rng = np.random.default_rng(37)
    t = np.cumsum(37.0 + rng.uniform(-0.3, 0.3, 195))
    grid = np.arange(30.0, 45.0, 0.01)
    ps = np.array([E.rayleigh_p(t, g) for g in grid])
    best = grid[int(np.argmin(ps))]
    assert 35.0 <= best <= 40.0 and ps.min() < 1e-4
    # uniform phases: P(p < 0.05) ~ 0.05 (Zar's small-n correction)
    null = np.array([E.rayleigh_p(rng.uniform(0, 1e4, 25), 97.0) for _ in range(2000)])
    assert 0.035 < np.mean(null < 0.05) < 0.065


def test_rayleigh_loses_renewal_jittered_beacons():
    # decisions.md: +-30% renewal jitter kills phase coherence (median p ~ 0.14
    # at n = 40) while the renewal LRT sees them at ~1e-12: why B12 is renewal-first
    rng = np.random.default_rng(14)
    ray, lrt = [], []
    for _ in range(200):
        d = 300.0 * rng.uniform(0.7, 1.3, 40)
        ray.append(E.rayleigh_p(np.cumsum(d), 300.0))
        lrt.append(null_p(d))
    assert np.median(ray) > 0.02
    assert np.median(lrt) < 1e-10
