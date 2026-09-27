"""bayes (B03/B04/B07/B18 predictives): tails validated against scipy.stats and
against exact rational arithmetic (fractions) where scipy itself cancels, the
B04 unit-test targets, NaN / edge policy, scalar-vs-array agreement, null
calibration, conjugate updates, and the NB speed contract (>50x scipy.stats)."""
from __future__ import annotations

import math
import timeit
from decimal import Decimal, getcontext
from fractions import Fraction

import numpy as np
import pytest
from scipy import special as sp
from scipy import stats

from app.engines.behavior.lib import bayes as B

NAN = float("nan")


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))


# ------------------------------------------------------------- exact references
def _fbeta(x: int, y: int) -> Fraction:
    """B(x, y) for positive integers, exactly."""
    return Fraction(math.factorial(x - 1) * math.factorial(y - 1), math.factorial(x + y - 1))


def bb_pmf_exact(k: int, n: int, a: int, b: int) -> Fraction:
    return math.comb(n, k) * _fbeta(k + a, n - k + b) / _fbeta(a, b)


def bb_sf_exact(k: int, n: int, a: int, b: int) -> float:
    return float(sum(bb_pmf_exact(j, n, a, b) for j in range(k + 1, n + 1)))


def nb_cdf_exact(k: int, m: Fraction, r: int) -> float:
    q = Fraction(r) / (r + m)
    return float(sum(math.comb(j + r - 1, j) * q ** r * (1 - q) ** j for j in range(k + 1)))


def pois_cdf_exact(k: int, m: int) -> float:
    return float(sum(Fraction(m ** j, math.factorial(j)) for j in range(k + 1))) * math.exp(-m)


def bb_tails_full(n, a, b):
    """Independent full-support log-space BB (cdf, sf) for any n (test reference)."""
    j = np.arange(n + 1.0)
    lp = (sp.gammaln(n + 1) - sp.gammaln(j + 1) - sp.gammaln(n - j + 1)
          + sp.betaln(j + a, n - j + b) - sp.betaln(a, b))
    pm = np.exp(lp)
    cdf = np.cumsum(pm)
    sf = np.concatenate([np.cumsum(pm[::-1])[::-1][1:], [0.0]])
    return cdf, sf


# ------------------------------------------------------------------ normal
def test_phi_inv_clip_nan_and_arrays():
    assert B.phi_inv(0.5) == 0.0
    assert B.phi_inv(0.0) == pytest.approx(-7.941345326170998)
    assert B.phi_inv(1.0) == pytest.approx(7.94, abs=0.01)
    assert abs(B.phi_inv(0.0)) <= 7.95 and np.isfinite(B.phi_inv(1.0))
    assert math.isnan(B.phi_inv(NAN))
    assert B.phi_inv(1e-3, clip=0.01) == pytest.approx(sp.ndtri(0.01))
    u = np.array([0.0, 0.025, 0.5, 0.975, 1.0, np.nan])
    z = B.phi_inv(u)
    assert isinstance(z, np.ndarray) and z.shape == (6,)
    assert z[1] == pytest.approx(-1.959964) and z[3] == pytest.approx(1.959964)
    assert np.isnan(z[-1]) and np.all(np.isfinite(z[:-1]))
    assert isinstance(B.phi_inv(np.float32(0.3)), float)


def test_phi_sf_two_sided_and_anchors():
    assert B.phi_sf(1.959964) == pytest.approx(0.025, rel=1e-5)
    assert B.phi_sf(-40.0) == 1.0 and B.phi_sf(40.0) == pytest.approx(sp.ndtr(-40.0))
    assert math.isnan(B.phi_sf(NAN))
    assert np.allclose(B.phi_sf(np.array([0.0, 1.0])), [0.5, sp.ndtr(-1.0)])
    # two-sided mid-p
    assert B.two_sided_midp(0.5) == 1.0
    assert B.two_sided_midp(0.01) == pytest.approx(0.02)
    assert B.two_sided_midp(0.995) == pytest.approx(0.01)
    assert B.two_sided_midp(0.0) == B.P_FLOOR and B.two_sided_midp(1.0) == B.P_FLOOR
    assert math.isnan(B.two_sided_midp(NAN))
    p = B.two_sided_midp(np.array([0.0, 0.2, np.nan]))
    assert p[0] == B.P_FLOOR and p[1] == pytest.approx(0.4) and np.isnan(p[2])
    # dual anchor (B04): min(1, 2 min); one NaN -> other alone; both NaN -> NaN
    assert B.combine_anchors(0.05, 0.004) == pytest.approx(0.008)
    assert B.combine_anchors(0.6, 0.9) == 1.0
    assert B.combine_anchors(NAN, 0.004) == 0.004 and B.combine_anchors(0.03, NAN) == 0.03
    assert math.isnan(B.combine_anchors(NAN, NAN))
    out = B.combine_anchors(np.array([0.05, np.nan, 0.2, np.nan]),
                            np.array([0.004, 0.3, np.nan, np.nan]))
    assert np.allclose(out[:3], [0.008, 0.3, 0.2]) and np.isnan(out[3])


def test_dual_anchor_reference_drives_pf():
    """B04 (e): current anchor shifted +1 sigma, reference unchanged, x = ref + 3 sigma."""
    prior = B.NIG(0.0, 1e6, 1e6, 1e6)                   # ~N(0, 1) predictive, df huge
    cur = B.NIG(1.0, 1e6, 1e6, 1e6)
    x = 3.0
    _, p_cur = B.t_midp(x, cur)
    _, p_ref = B.t_midp(x, prior)
    assert 0.04 < p_cur < 0.05
    assert B.combine_anchors(p_cur, p_ref) < 0.01


# ------------------------------------------------------------ negative binomial
def test_nb_b04_targets_against_exact():
    # Poisson(22) (r >= 1e12 -> pdtr): P(X<=3) = 5.7e-7, P(X<=1) = 6.4e-9
    for r in (1e12, np.inf, 1e15):
        assert B.nb_cdf(3, 22.0, r) == pytest.approx(pois_cdf_exact(3, 22), rel=1e-12)
        assert B.nb_cdf(1, 22.0, r) == pytest.approx(pois_cdf_exact(1, 22), rel=1e-12)
    assert B.nb_cdf(3, 22.0, 1e12) == pytest.approx(5.689585e-7, rel=1e-6)
    assert B.nb_cdf(1, 22.0, 1e12) == pytest.approx(6.415777e-9, rel=1e-6)
    assert B.nb_cdf(3, 22, 1e12) == pytest.approx(stats.poisson.cdf(3, 22), rel=1e-12)
    # NB r = 20, mean 22: P(X<=3) = 1.04e-4 (exact rational reference)
    assert B.nb_cdf(3, 22.0, 20.0) == pytest.approx(nb_cdf_exact(3, Fraction(22), 20), rel=1e-12)
    assert B.nb_cdf(3, 22.0, 20.0) == pytest.approx(1.043876e-4, rel=1e-6)
    # kappa capped at 1e3 (the B04 "Poisson" baseline): slightly heavier than Poisson
    assert 6.4158e-9 < B.nb_cdf(1, 22.0, 1e3) < 1e-8


def test_nb_midp_b04_unit_test_a():
    # Poisson baseline, both anchors agree -> p_f = min(1, 2 min(p, p))
    u1, p1 = B.nb_midp(1, 22.0, 1e12)
    assert u1 == pytest.approx(pois_cdf_exact(0, 22) + 0.5 * 22 * math.exp(-22), rel=1e-12)
    assert B.combine_anchors(p1, p1) < 1e-7
    u3, p3 = B.nb_midp(3, 22.0, 1e12)
    assert B.combine_anchors(p3, p3) < 1e-5 and B.phi_inv(u3) < -4.5
    # kappa capped at 1e3 with a large posterior shape -> r ~ 1e3: same verdicts
    r = B.nb_size(1e9, 1e7)
    assert r == pytest.approx(999.9, rel=1e-3)
    _, p1k = B.nb_midp(1, 22.0, r)
    _, p3k = B.nb_midp(3, 22.0, r)
    assert B.combine_anchors(p1k, p1k) < 1e-7 and B.combine_anchors(p3k, p3k) < 1e-5
    # NB r = 20: observing 3 gives p < 1e-3
    _, p = B.nb_midp(3, 22.0, 20.0)
    assert p < 1e-3
    assert p == pytest.approx(2 * (B.nb_cdf(2, 22.0, 20.0) + 0.5 * B.nb_pmf(3, 22.0, 20.0)))


@pytest.mark.parametrize("m", [0.01, 0.7, 3.0, 22.0, 450.0, 3e4])
@pytest.mark.parametrize("r", [1e-3, 0.4, 2.0, 20.0, 1e3, 1e6])
def test_nb_pmf_cdf_sf_match_scipy(m, r):
    sd = math.sqrt(m + m * m / r)
    k = np.unique(np.floor(np.linspace(0, m + 12 * sd + 10, 60)))
    ref = stats.nbinom(r, r / (r + m))
    # scipy's own error: its gammaln-difference pmf drifts ~eps * max(r, k) ln (4e-8 at
    # r = 1e6) and nbdtr/nbdtrc form 1 - p (1e-8 at r/m = 1e8); ours is checked to
    # 1e-12 against 60-digit decimals in the next tests
    clean = r <= 1e3 and k.max() < 1e4
    assert rel(B.nb_pmf(k, m, r), ref.pmf(k)) < (1e-9 if clean else 1e-7)
    c, s = B.nb_cdf(k, m, r), B.nb_sf(k, m, r)
    if r <= 1e3:
        sel = ref.cdf(k) > 1e-280
        assert rel(c[sel], ref.cdf(k)[sel]) < 1e-8      # boost betainc vs cephes nbdtr
        sel = ref.sf(k) > 1e-280
        assert rel(s[sel], ref.sf(k)[sel]) < 1e-8
    assert np.allclose(c + s, 1.0, atol=1e-13)


def test_nb_pmf_exact_at_large_size():
    """ln Gamma(k+r) - ln Gamma(r) via the Stirling difference: 1e-12 relative
    against 60-digit decimal arithmetic where the gammaln difference loses 1e-9."""
    getcontext().prec = 60

    def pmf_dec(k, m, r):
        m, r = Decimal(m), Decimal(r)
        q = r / (r + m)
        coef = Decimal(1)
        for j in range(k):
            coef *= (r + j) / (j + 1)
        return float(coef * q ** r * (1 - q) ** k)

    for m, r in ((0.01, 10 ** 6), (22.0, 10 ** 6), (22.0, 20), (3.0, 10 ** 9), (0.5, 11)):
        for k in (1, 2, 5, 10, 40):
            ref = pmf_dec(k, m, r)
            assert B.nb_pmf(k, m, float(r)) == pytest.approx(ref, rel=1e-12)
            assert B.nb_pmf(np.array([k]), m, float(r))[0] == pytest.approx(ref, rel=1e-12)
    # huge k with a small size (the k >> r branch of the Stirling difference)
    k, m, r = 200_000, 3e4, 0.4
    lref = (math.fsum(math.log1p((r - 1.0) / (j + 1.0)) for j in range(k))
            - r * math.log1p(m / r) - k * math.log1p(r / m))
    assert math.log(B.nb_pmf(k, m, r)) == pytest.approx(lref, abs=1e-9)


def test_nb_tails_exact_at_large_size():
    """betainc is fed the small one of q and 1 - q, so NB tails stay exact up to
    the Poisson switch (betainc(r, k+1, q) alone is off by 1e-6 at r = 1e11)."""
    getcontext().prec = 80

    def cdf_dec(k, m, r):
        m, r = Decimal(m), Decimal(r)
        q = r / (r + m)
        qr, coef, tot = (r * q.ln()).exp(), Decimal(1), Decimal(0)
        for j in range(k + 1):
            if j:
                coef *= (r + j - 1) / j
            tot += coef * qr * (1 - q) ** j
        return tot

    for r in (10 ** 6, 10 ** 9, 10 ** 11):
        for m in (22.0, 0.01):
            for k in (0, 1, 10, 30, 60):
                ref = cdf_dec(k, m, r)
                lo, up = float(ref), float(1 - ref)
                ka = np.array([k])
                if lo > 1e-300:
                    assert B.nb_cdf(k, m, float(r)) == pytest.approx(lo, rel=1e-10)
                    assert B.nb_cdf(ka, m, float(r))[0] == pytest.approx(lo, rel=1e-10)
                if up > 1e-300:
                    assert B.nb_sf(k, m, float(r)) == pytest.approx(up, rel=1e-10)
                    assert B.nb_sf(ka, m, float(r))[0] == pytest.approx(up, rel=1e-10)


def test_nb_far_tails_are_exact_on_both_sides():
    # upper tail far beyond 1 - cdf resolution
    assert B.nb_sf(150, 22.0, 1e12) == pytest.approx(stats.poisson.sf(150, 22.0), rel=1e-10)
    assert B.nb_sf(150, 22.0, 1e12) < 1e-60
    assert B.nb_sf(500, 3.0, 5.0) == pytest.approx(stats.nbinom.sf(500, 5.0, 5 / 8), rel=1e-10)
    # lower tail far below
    assert B.nb_cdf(500, 1000.0, 1e12) == pytest.approx(stats.poisson.cdf(500, 1000.0), rel=1e-10)
    assert B.nb_cdf(500, 1000.0, 1e12) < 1e-50
    # mid-p resolves p ~ 1e-100 instead of clipping to 0
    u, p = B.nb_midp(200, 22.0, 1e12)
    assert u == 1.0 or u > 1 - 1e-15
    assert 1e-150 < p < 1e-90
    ref = 2 * (stats.poisson.sf(200, 22.0) + 0.5 * stats.poisson.pmf(200, 22.0))
    assert p == pytest.approx(ref, rel=1e-9)


def test_nb_large_size_is_accurate_and_poisson_limit():
    for r in (1e7, 1e9, 1e11):
        k = np.arange(0, 60.0)
        # NB -> Poisson with relative gap O(m^2 / r)
        assert rel(B.nb_pmf(k, 22.0, r), stats.poisson.pmf(k, 22.0)) < 1e4 / r + 1e-9
        assert rel(B.nb_cdf(k, 22.0, r), stats.poisson.cdf(k, 22.0)) < 1e4 / r + 1e-9
    # tiny r: extreme overdispersion stays finite and normalised
    k = np.arange(0, 2000.0)
    assert abs(B.nb_pmf(k, 5.0, 1e-6).sum() + B.nb_sf(1999, 5.0, 1e-6) - 1.0) < 1e-12
    assert B.nb_cdf(3, 5.0, 0.0) == B.nb_cdf(3, 5.0, 1e-6)       # r clipped to 1e-6


def test_nb_mid_u_identities():
    m, r = 22.0, 20.0
    for k in range(0, 80):
        u = B.nb_mid_u(k, m, r)
        ref = B.nb_cdf(k - 1, m, r) + 0.5 * B.nb_pmf(k, m, r)
        assert u == pytest.approx(ref, rel=1e-12, abs=1e-300)
        up = B.nb_sf(k, m, r) + 0.5 * B.nb_pmf(k, m, r)
        assert u + up == pytest.approx(1.0, abs=1e-13)
        uu, p = B.nb_midp(k, m, r)
        assert uu == u and p == pytest.approx(max(B.P_FLOOR, min(1.0, 2 * min(u, up))))
    # rounding of float32-noisy counts; cdf floors real k
    assert B.nb_midp(2.9999998, m, r) == B.nb_midp(3, m, r)
    assert B.nb_pmf(np.float32(7.0), m, r) == B.nb_pmf(7, m, r)
    assert B.nb_cdf(2.5, m, r) == B.nb_cdf(2, m, r)
    assert B.nb_sf(2.5, m, r) == B.nb_sf(2, m, r)


def test_nb_edge_cases_and_nan_policy():
    # k < 0, point mass at 0, infinite k
    assert B.nb_cdf(-1, 3.0, 2.0) == 0.0 and B.nb_sf(-1, 3.0, 2.0) == 1.0
    assert B.nb_pmf(-2, 3.0, 2.0) == 0.0
    assert B.nb_cdf(0, 0.0, 2.0) == 1.0 and B.nb_sf(0, 0.0, 2.0) == 0.0
    assert B.nb_pmf(0, 0.0, 2.0) == 1.0 and B.nb_pmf(1, 0.0, 2.0) == 0.0
    assert B.nb_cdf(np.inf, 3.0, 2.0) == 1.0 and B.nb_sf(np.inf, 3.0, 2.0) == 0.0
    assert B.nb_midp(0, 0.0, 5.0) == (0.5, 1.0)
    assert B.nb_midp(2, 0.0, 5.0) == (1.0, B.P_FLOOR)
    # mean = inf: all mass at infinity, no NaN
    assert B.nb_cdf(10, np.inf, 5.0) == 0.0 and B.nb_sf(10, np.inf, 5.0) == 1.0
    assert B.nb_pmf(10, np.inf, 5.0) == 0.0 and B.nb_cdf(10, np.inf, 1e12) == 0.0
    # invalid observation -> NaN, never p = 0 / 1
    for bad in (-1, NAN, np.inf):
        u, p = B.nb_midp(bad, 3.0, 2.0)
        assert math.isnan(u) and math.isnan(p)
        assert math.isnan(B.nb_mid_u(bad, 3.0, 2.0))
    # NaN anywhere -> NaN everywhere
    for args in ((NAN, 3.0, 2.0), (1, NAN, 2.0), (1, 3.0, NAN)):
        for f in (B.nb_pmf, B.nb_cdf, B.nb_sf, B.nb_mid_u):
            assert math.isnan(f(*args))
            assert np.isnan(f(*(np.array([a]) for a in args)))[0]
        assert all(math.isnan(v) for v in B.nb_midp(*args))
    # absurd observation: p floored, never exactly 0
    u, p = B.nb_midp(100000, 22.0, 1e12)
    assert p == B.P_FLOOR and u == 1.0
    u, p = B.nb_midp(0, 1e6, 1e12)
    assert p == B.P_FLOOR and np.isfinite(B.phi_inv(u))


def test_nb_scalar_and_array_paths_agree():
    rng = np.random.default_rng(3)
    k = np.concatenate([rng.integers(0, 80, 40).astype(float), [-1.0, 0.0, np.nan, np.inf]])
    m = np.concatenate([rng.uniform(0.01, 60, 40), [3.0, 0.0, 2.0, 5.0]])
    r = np.concatenate([rng.choice([0.5, 5.0, 50.0, 1e12], 40), [2.0, 1.0, 1.0, 1e12]])
    vec = {f: f(k, m, r) for f in (B.nb_pmf, B.nb_cdf, B.nb_sf, B.nb_mid_u)}
    uv, pv = B.nb_midp(k, m, r)
    for i in range(len(k)):
        for f, arr in vec.items():
            s = f(float(k[i]), float(m[i]), float(r[i]))
            same = s == pytest.approx(arr[i], rel=1e-12, abs=0)
            assert (math.isnan(s) and math.isnan(arr[i])) or same, (f, i)
        us, ps = B.nb_midp(float(k[i]), float(m[i]), float(r[i]))
        assert (math.isnan(us) and np.isnan(uv[i])) or (us == pytest.approx(uv[i], rel=1e-12)
                                                         and ps == pytest.approx(pv[i], rel=1e-12))
    # broadcasting and return types
    assert B.nb_cdf(np.arange(5.0), 3.0, 2.0).shape == (5,)
    assert B.nb_cdf(np.arange(6.0).reshape(2, 3), np.array([[1.0], [2.0]]), 4.0).shape == (2, 3)
    assert isinstance(B.nb_cdf(np.array(3.0), 22.0, 20.0), float)
    assert isinstance(B.nb_midp(3, 22.0, 20.0)[1], float)


def test_nb_null_calibration():
    rng = np.random.default_rng(11)
    for m, r in ((22.0, 20.0), (22.0, 1e12), (200.0, 50.0), (3.0, 1.0)):
        x = (rng.negative_binomial(r, r / (r + m), 2000) if r < 1e12
             else rng.poisson(m, 2000)).astype(float)
        u, p = B.nb_midp(x, m, r)
        z = B.phi_inv(u)
        assert abs(z.mean()) < 0.1 and abs(z.std() - 1.0) < 0.1
        assert np.all((p > 0) & (p <= 1))
        # the randomised PIT built from the same cdf / pmf is exactly uniform (the
        # mid-p itself is only uniform-ish for discrete X, so it is not KS-tested)
        pit = B.nb_cdf(x - 1, m, r) + rng.uniform(size=x.size) * B.nb_pmf(x, m, r)
        assert stats.kstest(pit, "uniform").pvalue > 0.01


def test_nb_ppf_matches_scipy_and_definition():
    qs = np.array([1e-9, 1e-4, 0.01, 0.05, 0.3, 0.5, 0.7, 0.95, 0.99, 1 - 1e-6])
    for m in (0.01, 0.5, 3.0, 22.0, 400.0, 1e5):
        for r in (1e-3, 0.3, 2.0, 20.0, 1e3):
            got = B.nb_ppf(qs, m, r)
            ref = stats.nbinom.ppf(qs, r, r / (r + m))
            for q, g, e in zip(qs, got, ref):
                if g != e:          # only allowed at float resolution of the cdf near q
                    near = min(abs(B.nb_cdf(g - 1, m, r) - q), abs(B.nb_cdf(g, m, r) - q))
                    assert near < 1e-13
        assert np.array_equal(B.nb_ppf(qs, m, 1e12), stats.poisson.ppf(qs, m))
    got = B.nb_ppf(np.array([0.05, 0.5, 0.95]), 22.0, 20.0)
    for q, g in zip((0.05, 0.5, 0.95), got):
        assert B.nb_cdf(g, 22.0, 20.0) >= q > B.nb_cdf(g - 1, 22.0, 20.0)
    # edges
    assert B.nb_ppf(0.0, 5.0, 2.0) == 0.0 and B.nb_ppf(1.0, 5.0, 2.0) == np.inf
    assert B.nb_ppf(0.7, 0.0, 2.0) == 0.0
    assert math.isnan(B.nb_ppf(-0.1, 5.0, 2.0)) and math.isnan(B.nb_ppf(1.1, 5.0, 2.0))
    assert math.isnan(B.nb_ppf(NAN, 5.0, 2.0)) and math.isnan(B.nb_ppf(0.5, NAN, 2.0))
    assert isinstance(B.nb_ppf(0.5, 22.0, 20.0), float)


def test_nb_size():
    assert B.nb_size(10.0, 10.0) == pytest.approx(5.0)
    assert B.nb_size(1e9, 1e12) == pytest.approx(1e3, rel=1e-6)        # kappa clipped to 1e3
    assert B.nb_size(0.01, 1e6) == pytest.approx(0.5, rel=1e-5)        # kappa clipped to 0.5
    assert B.nb_size(40.0, np.inf) == 40.0
    assert math.isnan(B.nb_size(40.0, 0.0)) and math.isnan(B.nb_size(NAN, 3.0))
    r = B.nb_size(np.array([10.0, 1e5]), np.array([10.0, 1e9]))
    assert np.allclose(r, [5.0, 1e3], rtol=1e-5)


def test_nb_betainc_is_50x_faster_than_scipy_stats():
    f_ours = lambda: B.nb_cdf(3, 22.0, 20.0)                                   # noqa: E731
    f_ref = lambda: stats.nbinom.cdf(3, 20.0, 20.0 / 42.0)                     # noqa: E731
    assert f_ours() == pytest.approx(f_ref(), rel=1e-12)
    t_ours = min(timeit.repeat(f_ours, number=2000, repeat=7)) / 2000
    t_ref = min(timeit.repeat(f_ref, number=40, repeat=7)) / 40
    assert t_ref / t_ours > 50, (t_ours * 1e6, t_ref * 1e6)


# -------------------------------------------------------------- beta-binomial
def test_bb_b04_targets_against_exact_rationals():
    a50, b50 = B.bb_params(0.1, 50.0)
    a20, b20 = B.bb_params(0.1, 20.0)
    assert (a50, b50) == pytest.approx((5.0, 45.0)) and (a20, b20) == pytest.approx((2.0, 18.0))
    ref50, ref20 = bb_sf_exact(99, 200, 5, 45), bb_sf_exact(99, 200, 2, 18)
    assert ref50 == pytest.approx(1.3424e-8, rel=1e-4)
    assert ref20 == pytest.approx(7.699e-5, rel=1e-4)
    assert B.bb_sf(99, 200, a50, b50) == pytest.approx(ref50, rel=1e-11)
    assert B.bb_sf(99, 200, a20, b20) == pytest.approx(ref20, rel=1e-11)
    assert B.bb_sf(0, 2, 5.0, 45.0) == pytest.approx(16 / 85, rel=1e-13)        # 0.188
    assert B.bb_sf(0, 2, 5.0, 45.0) == pytest.approx(stats.betabinom.sf(0, 2, 5, 45), rel=1e-12)
    # (b) k = 1 of n = 2 vs mean 0.1, c = 50: not an anomaly
    u, p = B.bb_midp(1, 2, a50, b50)
    assert p > 0.05 and 0.5 < u < 1
    # (c) k = 100 of n = 200: phi 50 -> p < 1e-6, phi 20 -> p < 1e-3
    _, p50 = B.bb_midp(100, 200, a50, b50)
    _, p20 = B.bb_midp(100, 200, a20, b20)
    assert p50 < 1e-6 and p20 < 1e-3
    half50 = 0.5 * float(bb_pmf_exact(100, 200, 5, 45))
    assert p50 == pytest.approx(2 * (bb_sf_exact(100, 200, 5, 45) + half50), rel=1e-10)
    # (f) n* = 0 -> NaN, not p = 1
    assert all(math.isnan(v) for v in B.bb_midp(0, 0, a50, b50))


def test_bb_logpmf_matches_scipy_and_support():
    for n, a, b in ((1, 0.3, 0.7), (7, 2.0, 5.0), (200, 5.0, 45.0), (5000, 0.2, 19.8)):
        k = np.unique(np.linspace(0, n, 50).round())
        assert rel(B.bb_logpmf(k, n, a, b), stats.betabinom.logpmf(k, n, a, b)) < 1e-10
    assert B.bb_logpmf(-1, 5, 1.0, 1.0) == -np.inf and B.bb_logpmf(6, 5, 1.0, 1.0) == -np.inf
    assert B.bb_logpmf(0, 0, 1.0, 1.0) == 0.0
    assert math.isnan(B.bb_logpmf(1, -1, 1.0, 1.0)) and math.isnan(B.bb_logpmf(1, 5, 0.0, 1.0))
    assert math.isnan(B.bb_logpmf(NAN, 5, 1.0, 1.0))
    assert B.bb_logpmf(2.9999998, 10, 2.0, 3.0) == B.bb_logpmf(3, 10, 2.0, 3.0)
    assert np.exp(B.bb_logpmf(np.arange(31.0), 30, 0.4, 3.0)).sum() == pytest.approx(1.0, abs=1e-13)


@pytest.mark.parametrize("n", [1, 5, 40, 200, 1500])
@pytest.mark.parametrize("mu,c", [(0.001, 20.0), (0.1, 50.0), (0.5, 2.0), (0.97, 300.0),
                                  (0.3, 1e5)])
def test_bb_cdf_sf_exact_up_to_bb_exact_max_n(n, mu, c):
    a, b = B.bb_params(mu, c)
    cdf, sf = bb_tails_full(n, a, b)
    k = np.unique(np.linspace(0, n, 25).round())
    kk = k.astype(int)
    got_c, got_s = B.bb_cdf(k, n, a, b), B.bb_sf(k, n, a, b)
    sel = cdf[kk] > 1e-280
    assert rel(got_c[sel], cdf[kk][sel]) < 1e-9
    sel = sf[kk] > 1e-280
    assert rel(got_s[sel], sf[kk][sel]) < 1e-9
    assert np.allclose(got_c + got_s, 1.0, atol=1e-12)


@pytest.mark.parametrize("n,mu,c", [(5000, 0.01, 50.0), (20000, 1e-4, 20.0), (20000, 0.1, 20.0),
                                    (100000, 1e-3, 200.0), (20000, 0.5, 1e6)])
def test_bb_large_n_tails_stay_accurate(n, mu, c):
    """Above BB_EXACT_MAX_N: exact near k + Beta remainder; |log10 error| < 0.2
    on p in [1e-10, 0.5] where a normal approximation is off by orders of magnitude."""
    a, b = B.bb_params(mu, c)
    cdf, sf = bb_tails_full(n, a, b)
    for ref, fn in ((sf, B.bb_sf), (cdf, B.bb_cdf)):
        ks = np.nonzero((ref > 1e-10) & (ref < 0.5))[0]
        if len(ks) == 0:
            continue
        ks = ks[np.linspace(0, len(ks) - 1, min(12, len(ks))).astype(int)]
        got = np.array([fn(float(k), n, a, b) for k in ks])
        assert np.max(np.abs(np.log10(got) - np.log10(ref[ks]))) < 0.2, (n, mu, c)


def test_bb_skewed_ratio_is_not_a_false_alarm():
    """1 % error rate, n = 1000, c = 50: 100 errors in a tick is unusual (p ~ 3e-3)
    but the normal approximation with continuity correction says p ~ 4e-10."""
    a, b = B.bb_params(0.01, 50.0)
    cdf, sf = bb_tails_full(1000, a, b)
    mean, var = 1000 * 0.01, 1000 * a * b * (a + b + 1000) / ((a + b) ** 2 * (a + b + 1))
    for k in (40, 100, 150):
        _, p = B.bb_midp(k, 1000, a, b)
        assert p == pytest.approx(2 * (sf[k] + 0.5 * (sf[k - 1] - sf[k])), rel=1e-9)
    _, p = B.bb_midp(100, 1000, a, b)
    assert p > 1e-3 and 2 * sp.ndtr(-(100 - 0.5 - mean) / math.sqrt(var)) < 1e-9     # avoided
    _, p = B.bb_midp(150, 1000, a, b)
    assert p > 1e-4 and 2 * sp.ndtr(-(150 - 0.5 - mean) / math.sqrt(var)) < 1e-20


def test_bb_edge_cases_and_nan_policy():
    a, b = 2.0, 8.0
    assert B.bb_cdf(-1, 10, a, b) == 0.0 and B.bb_sf(-1, 10, a, b) == 1.0
    assert B.bb_cdf(10, 10, a, b) == 1.0 and B.bb_sf(10, 10, a, b) == 0.0
    assert B.bb_cdf(np.inf, 10, a, b) == 1.0 and B.bb_sf(-np.inf, 10, a, b) == 1.0
    assert B.bb_cdf(3.7, 10, a, b) == B.bb_cdf(3, 10, a, b)
    for n in (0, -3, NAN, np.inf):
        assert math.isnan(B.bb_cdf(1, n, a, b)) and math.isnan(B.bb_sf(1, n, a, b))
        assert all(math.isnan(v) for v in B.bb_midp(0, n, a, b))
    for args in ((NAN, 10, a, b), (1, 10, NAN, b), (1, 10, a, 0.0), (1, 10, -1.0, b)):
        assert math.isnan(B.bb_cdf(*args)) and math.isnan(B.bb_sf(*args))
        assert all(math.isnan(v) for v in B.bb_midp(*args))
    # observation outside the support is invalid -> NaN (never p = 0)
    assert all(math.isnan(v) for v in B.bb_midp(11, 10, a, b))
    assert all(math.isnan(v) for v in B.bb_midp(-1, 10, a, b))
    # extremes of the support are valid and finite
    u, p = B.bb_midp(10, 10, 0.01, 50.0)
    assert u > 0.999 and B.P_FLOOR <= p < 1e-10
    u, p = B.bb_midp(0, 10, a, b)
    assert u == pytest.approx(0.5 * math.exp(B.bb_logpmf(0, 10, a, b)))
    assert p == pytest.approx(2 * u)
    # float32-noisy counts are rounded in mid-p
    assert B.bb_midp(np.float32(3.0000002), np.float32(10.0), a, b) == B.bb_midp(3, 10, a, b)


def test_bb_scalar_and_array_paths_agree():
    rng = np.random.default_rng(5)
    n = rng.choice([1, 2, 30, 200, 3000], 30).astype(float)
    k = np.floor(rng.uniform(0, 1, 30) * (n + 1))
    a = rng.uniform(0.1, 20, 30)
    b = rng.uniform(0.1, 200, 30)
    k = np.append(k, [NAN, 3.0, 5.0])
    n = np.append(n, [10.0, 0.0, 4.0])
    a, b = np.append(a, [1.0, 1.0, 1.0]), np.append(b, [1.0, 1.0, 1.0])
    vc, vs = B.bb_cdf(k, n, a, b), B.bb_sf(k, n, a, b)
    vu, vp = B.bb_midp(k, n, a, b)
    for i in range(len(k)):
        args = (float(k[i]), float(n[i]), float(a[i]), float(b[i]))
        for got, arr in ((B.bb_cdf(*args), vc[i]), (B.bb_sf(*args), vs[i]),
                         (B.bb_midp(*args)[0], vu[i]), (B.bb_midp(*args)[1], vp[i])):
            assert (math.isnan(got) and np.isnan(arr)) or got == pytest.approx(arr, rel=1e-13)
    assert np.isnan(vp[-3]) and np.isnan(vp[-2]) and np.isnan(vu[-1])       # NaN, n = 0, k > n
    assert isinstance(B.bb_midp(np.array(3.0), 10, 2.0, 3.0)[0], float)
    assert B.bb_midp(np.arange(4.0), 3.0, 1.0, 1.0)[1].shape == (4,)


def test_bb_null_calibration():
    rng = np.random.default_rng(13)
    for n, mu, c in ((50, 0.1, 50.0), (200, 0.3, 20.0), (1000, 0.02, 100.0)):
        a, b = B.bb_params(mu, c)
        k = rng.binomial(n, rng.beta(a, b, 1500))
        u, p = B.bb_midp(k.astype(float), n, a, b)
        z = B.phi_inv(u)
        assert abs(z.mean()) < 0.1 and abs(z.std() - 1.0) < 0.1, (n, mu, c)


def test_bb_params():
    assert B.bb_params(0.25, 40.0) == pytest.approx((10.0, 30.0))
    a, b = B.bb_params(0.0, 50.0)
    assert a == pytest.approx(5e-5) and b == pytest.approx(50.0 - 5e-5)
    a, b = B.bb_params(1.0, 50.0)
    assert b == pytest.approx(5e-5)
    assert all(math.isnan(v) for v in B.bb_params(0.3, 0.0))
    assert all(math.isnan(v) for v in B.bb_params(NAN, 10.0))
    a, b = B.bb_params(np.array([0.1, 0.5]), np.array([20.0, -1.0]))
    assert a[0] == pytest.approx(2.0) and np.isnan(a[1]) and np.isnan(b[1])


def test_bb_ppf():
    qs = np.array([1e-9, 1e-4, 0.01, 0.05, 0.3, 0.5, 0.7, 0.95, 0.99, 1 - 1e-6])
    for n in (1, 2, 10, 200, 1000):
        for mu, c in ((0.001, 20.0), (0.1, 50.0), (0.5, 2.0), (0.97, 200.0), (0.3, 1e5)):
            a, b = B.bb_params(mu, c)
            ref = stats.betabinom.ppf(qs, n, a, b)
            assert np.array_equal(B.bb_ppf(qs, n, a, b), ref), (n, mu, c)
    # above the grid threshold: the definition holds (bisection on the hybrid cdf / sf)
    n, (a, b) = 5000, B.bb_params(0.01, 50.0)
    for q, g in zip((0.05, 0.5, 0.95), B.bb_ppf(np.array([0.05, 0.5, 0.95]), n, a, b)):
        if q <= 0.5:
            assert B.bb_cdf(g, n, a, b) >= q > B.bb_cdf(g - 1, n, a, b)
        else:
            assert B.bb_sf(g, n, a, b) <= 1 - q < B.bb_sf(g - 1, n, a, b)
    assert B.bb_ppf(0.0, 10, 1.0, 1.0) == 0.0 and B.bb_ppf(1.0, 10, 1.0, 1.0) == 10.0
    assert math.isnan(B.bb_ppf(0.5, 0, 1.0, 1.0)) and math.isnan(B.bb_ppf(1.5, 10, 1.0, 1.0))
    assert math.isnan(B.bb_ppf(NAN, 10, 1.0, 1.0))
    assert isinstance(B.bb_ppf(0.5, 10, 1.0, 1.0), float)


# ----------------------------------------------------------------- student-t
def test_nig_predictive():
    df, loc, scale = B.nig_predictive(B.NIG(1.5, 3.0, 2.5, 4.0))
    assert (df, loc) == (5.0, 1.5)
    assert scale == pytest.approx(math.sqrt(4.0 * 4.0 / (2.5 * 3.0)))
    # the hyperprior: 2 dof, scale ~ 20
    df, _, scale = B.nig_predictive(B.NIG(0.0, 0.01, 1.0, 4.0))
    assert df == 2.0 and scale == pytest.approx(20.1, abs=0.01)
    assert math.isnan(B.nig_predictive(B.NIG(0.0, 0.0, 1.0, 1.0))[2])
    assert math.isnan(B.nig_predictive(B.NIG(0.0, 1.0, -1.0, 1.0))[2])
    df, loc, scale = B.nig_predictive(B.NIG(np.zeros(3), np.ones(3), np.array([1.0, 2.0, 0.0]),
                                            np.ones(3)))
    assert np.allclose(df, [2.0, 4.0, 0.0]) and np.isnan(scale[2]) and np.isfinite(scale[:2]).all()


def test_t_cdf_midp_ppf_match_scipy():
    post = B.NIG(1.0, 3.0, 2.5, 4.0)
    df, loc, sc = B.nig_predictive(post)
    x = np.array([-500.0, -50.0, -3.0, 0.0, 1.0, 2.5, 40.0, 1e3])
    ref_c, ref_s = stats.t.cdf(x, df, loc, sc), stats.t.sf(x, df, loc, sc)
    assert rel(B.t_cdf(x, df, loc, sc), ref_c) < 1e-10
    u, p = B.t_midp(x, post)
    assert rel(u, ref_c) < 1e-10
    assert rel(p, np.minimum(1.0, 2 * np.minimum(ref_c, ref_s))) < 1e-10
    assert p[-1] < 1e-8 and p[0] < 1e-9                    # both far tails resolved
    for xi in x:
        us, ps = B.t_midp(float(xi), post)
        assert us == pytest.approx(stats.t.cdf(xi, df, loc, sc), rel=1e-10)
        assert B.t_cdf(float(xi), df, loc, sc) == pytest.approx(us, rel=1e-12)
    # heavy far tail with large df: exact, not clipped
    big = B.NIG(0.0, 1e6, 5e5, 5e5)
    _, p = B.t_midp(30.0, big)
    df2, loc2, sc2 = B.nig_predictive(big)
    assert p == pytest.approx(2 * stats.t.sf(30.0, df2, loc2, sc2), rel=1e-8) and p < 1e-150
    qs = np.array([1e-9, 0.01, 0.05, 0.5, 0.95, 0.99, 1 - 1e-9])
    assert rel(B.t_ppf(qs, post), stats.t.ppf(qs, df, loc, sc)) < 1e-9
    assert np.allclose(B.t_cdf(B.t_ppf(qs, post), df, loc, sc), qs, rtol=1e-8)
    assert B.t_ppf(0.0, post) == -np.inf and B.t_ppf(1.0, post) == np.inf
    assert math.isnan(B.t_ppf(-0.1, post)) and math.isnan(B.t_ppf(NAN, post))


def test_t_nan_policy_and_null_calibration():
    post = B.NIG(0.0, 10.0, 5.0, 5.0)
    assert all(math.isnan(v) for v in B.t_midp(NAN, post))
    assert all(math.isnan(v) for v in B.t_midp(1.0, B.NIG(0.0, 0.0, 5.0, 5.0)))
    assert math.isnan(B.t_cdf(1.0, 3.0, 0.0, 0.0)) and math.isnan(B.t_cdf(NAN, 3.0, 0.0, 1.0))
    u, p = B.t_midp(np.array([0.0, np.nan]), post)
    assert u[0] == 0.5 and p[0] == 1.0 and np.isnan(u[1]) and np.isnan(p[1])
    assert B.t_midp(1e12, post)[1] >= B.P_FLOOR
    assert B.t_cdf(1.0, np.inf, 0.0, 1.0) == pytest.approx(sp.ndtr(1.0))
    rng = np.random.default_rng(17)
    df, loc, sc = B.nig_predictive(post)
    u, p = B.t_midp(loc + sc * rng.standard_t(df, 2000), post)
    assert stats.kstest(u, "uniform").pvalue > 0.01 and stats.kstest(p, "uniform").pvalue > 0.01


# ------------------------------------------------------- posterior updates
def test_gamma_rate_posterior():
    g = B.gamma_rate_posterior(0.5, 0.5, 300.0, 30.0)
    assert (g.a, g.b) == (300.5, 30.5) and g.mean == pytest.approx(300.5 / 30.5)
    g = B.gamma_rate_posterior(0.5, 0.5, -1e-17, -1e-17)                   # decay residue
    assert (g.a, g.b) == (0.5, 0.5)
    g = B.gamma_rate_posterior(0.5, 0.5, np.array([10.0, 20.0]), np.array([1.0, 2.0]))
    assert np.allclose(g.a, [10.5, 20.5]) and np.allclose(g.mean, [10.5 / 1.5, 20.5 / 2.5])
    assert math.isnan(B.gamma_rate_posterior(0.5, 0.5, NAN, 1.0).a)
    # predictive for exposure e: NB(mean = a/b e, size = a) ~ exposure-scaled
    g = B.gamma_rate_posterior(0.5, 0.5, 10.0 * 200 * 15, 200 * 15.0)
    u, p = B.nb_midp(150, g.mean * 15.0, B.nb_size(1e9, g.a))
    assert p > 0.5 and g.mean == pytest.approx(10.0, rel=1e-3)


def _count_stats(x, e, w=None):
    w = np.ones_like(x) if w is None else w
    return (w.sum(), (w * x).sum(), (w * e).sum(), (w * x * x).sum(), (w * x * e).sum(),
            (w * e * e).sum())


def test_count_overdispersion_recovers_size():
    rng = np.random.default_rng(19)
    e = rng.choice([5.0, 15.0, 60.0], 4000)                               # mixed cadences
    mu = 2.0
    for kappa in (2.0, 20.0):
        x = rng.negative_binomial(kappa, kappa / (kappa + mu * e)).astype(float)
        assert B.count_overdispersion(*_count_stats(x, e)) == pytest.approx(kappa, rel=0.25)
    x = rng.poisson(mu * e).astype(float)
    assert B.count_overdispersion(*_count_stats(x, e)) > 100                # ~Poisson
    # degenerate inputs
    assert B.count_overdispersion(1.0, 5.0, 1.0, 25.0, 5.0, 1.0) == 1e3        # W <= 1
    assert B.count_overdispersion(10.0, 5.0, 0.0, 25.0, 0.0, 0.0) == 1e3       # se <= 0
    assert B.count_overdispersion(10.0, 0.0, 50.0, 0.0, 0.0, 250.0) == 1e3     # mu = 0
    x = np.array([0.0, 0, 0, 0, 500, 0, 0, 0])                                  # wildly bursty
    assert B.count_overdispersion(*_count_stats(x, np.ones(8))) == 0.5
    assert math.isnan(B.count_overdispersion(10.0, NAN, 50.0, 0.0, 0.0, 250.0))
    k = B.count_overdispersion(np.array([1.0, 10.0]), 5.0, np.array([1.0, 0.0]), 25.0, 5.0, 1.0)
    assert np.allclose(k, [1e3, 1e3])
    # weights behave like replication
    x, e = rng.poisson(8.0, 50).astype(float) * rng.integers(1, 4, 50), np.full(50, 4.0)
    w = rng.integers(1, 3, 50).astype(float)
    rep = np.repeat(np.arange(50), w.astype(int))
    assert B.count_overdispersion(*_count_stats(x, e, w)) == pytest.approx(
        B.count_overdispersion(*_count_stats(x[rep], e[rep])), rel=1e-12)


def _ratio_stats(k, n):
    return len(k), k.sum(), n.sum(), (k * k / n).sum(), ((k / n) ** 2).sum()


def test_ratio_posterior_recovers_concentration():
    rng = np.random.default_rng(23)
    n = rng.integers(80, 160, 3000).astype(float)
    k = rng.binomial(n.astype(int), rng.beta(0.1 * 50, 0.9 * 50, 3000)).astype(float)
    p_hat, phi = B.ratio_posterior(0.5, 0.5, *_ratio_stats(k, n))
    assert p_hat == pytest.approx((k.sum() + 0.5) / (n.sum() + 1.0))
    assert phi == pytest.approx(50.0, rel=0.3)
    k = rng.binomial(n.astype(int), 0.1).astype(float)                     # no overdispersion
    assert B.ratio_posterior(0.5, 0.5, *_ratio_stats(k, n))[1] == 1000.0
    k = np.where(rng.uniform(size=3000) < 0.1, n, 0.0)                      # all-or-nothing rows
    assert B.ratio_posterior(0.5, 0.5, *_ratio_stats(k, n))[1] == 20.0
    # degenerate: few rows, no trials, never a success
    assert B.ratio_posterior(0.5, 0.5, 2.0, 3.0, 10.0, 1.0, 0.1)[1] == 1000.0
    p, phi = B.ratio_posterior(0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert p == 0.5 and phi == 1000.0
    p, phi = B.ratio_posterior(0.5, 0.5, 10.0, 0.0, 500.0, 0.0, 0.0)
    assert p == pytest.approx(0.5 / 501) and phi == 1000.0
    assert all(math.isnan(v) for v in B.ratio_posterior(0.5, 0.5, 10.0, NAN, 500.0, 0.0, 0.0))


def _nig_direct(prior, x, w):
    """Textbook NIG update on (weighted) data."""
    W = w.sum()
    xbar = (w * x).sum() / W
    k = prior.kappa + W
    return B.NIG((prior.kappa * prior.m + W * xbar) / k, k, prior.alpha + W / 2,
                 prior.beta + 0.5 * (w * (x - xbar) ** 2).sum()
                 + prior.kappa * W * (xbar - prior.m) ** 2 / (2 * k))


def _stats(x, w):
    return w.sum(), (w * x).sum(), (w * x * x).sum()


def test_nig_posterior_matches_textbook_update():
    prior = B.NIG(math.log(1e4), 0.01, 1.0, 4.0)
    rng = np.random.default_rng(29)
    x = rng.normal(8.0, 0.7, 300)
    w = rng.uniform(0.2, 1.0, 300)
    post = B.nig_posterior(prior, *_stats(x, w))
    ref = _nig_direct(prior, x, w)
    for f in ("m", "kappa", "alpha", "beta"):
        assert getattr(post, f) == pytest.approx(getattr(ref, f), rel=1e-10)
        assert isinstance(getattr(post, f), float)
    assert B.nig_posterior(prior, 0.0, 0.0, 0.0) == prior
    assert B.nig_posterior(prior, -1e-18, 0.0, 0.0) == prior               # float residue
    # within term floored at 0: constant data with rounding error
    post = B.nig_posterior(prior, 3.0, 3.0 * 0.1, 3.0 * 0.1 ** 2 * (1 - 1e-15))
    assert post.beta >= prior.beta
    # the predictive tightens around the data
    df, loc, sc = B.nig_predictive(B.nig_posterior(prior, *_stats(x, np.ones(300))))
    assert loc == pytest.approx(x.mean(), abs=0.01) and sc == pytest.approx(0.7, rel=0.15)
    # vectorised over features
    post = B.nig_posterior(prior, np.array([0.0, 300.0]), np.array([0.0, x.sum()]),
                           np.array([0.0, (x * x).sum()]))
    assert post.m[0] == prior.m and post.m[1] == pytest.approx(ref.m, rel=0.01)
    assert math.isnan(B.nig_posterior(prior, 1.0, NAN, 1.0).m)


def test_nig_merge_is_stats_addition():
    prior = B.NIG(0.0, 0.01, 1.0, 4.0)
    rng = np.random.default_rng(31)
    xa, xb = rng.normal(2.0, 1.0, 120), rng.normal(3.0, 2.0, 80)
    wa, wb = np.ones(120), np.ones(80)
    pa, pb = B.nig_posterior(prior, *_stats(xa, wa)), B.nig_posterior(prior, *_stats(xb, wb))
    merged = B.nig_merge(pa, pb, prior)
    both = B.nig_posterior(prior, *_stats(np.r_[xa, xb], np.r_[wa, wb]))
    for f in ("m", "kappa", "alpha", "beta"):
        assert getattr(merged, f) == pytest.approx(getattr(both, f), rel=1e-9)
    half = B.nig_merge(pa, pb, prior, w_b=0.5)                             # link seeding
    ref = B.nig_posterior(prior, *_stats(np.r_[xa, xb], np.r_[wa, 0.5 * wb]))
    for f in ("m", "kappa", "alpha", "beta"):
        assert getattr(half, f) == pytest.approx(getattr(ref, f), rel=1e-9)
    same = B.nig_merge(pa, prior, prior)                                   # adding nothing
    for f in ("m", "kappa", "alpha", "beta"):
        assert getattr(same, f) == pytest.approx(getattr(pa, f), rel=1e-12)
    assert B.nig_merge(prior, prior, prior) == prior


# ------------------------------------------------- adversarial-review regressions
def test_b04_targets_against_scipy_stats():
    """The B04 tail targets straight from scipy.stats (distribution-level check)."""
    assert B.nb_cdf(3, 22.0, 1e12) == pytest.approx(stats.poisson.cdf(3, 22), rel=1e-10)
    assert B.nb_cdf(1, 22.0, 1e12) == pytest.approx(stats.poisson.cdf(1, 22), rel=1e-10)
    assert B.nb_cdf(3, 22.0, 20.0) == pytest.approx(stats.nbinom.cdf(3, 20, 20 / 42), rel=1e-10)
    assert stats.nbinom.cdf(3, 20, 20 / 42) == pytest.approx(1.04e-4, rel=5e-3)
    # scipy's betabinom.sf is 1 - cdf (~2e-6 relative off here), hence rel=1e-5
    for c, want in ((50.0, 1.3e-8), (20.0, 7.7e-5)):
        a, b = B.bb_params(0.1, c)
        assert B.bb_sf(99, 200, a, b) == pytest.approx(stats.betabinom.sf(99, 200, a, b), rel=1e-5)
        assert B.bb_sf(99, 200, a, b) == pytest.approx(want, rel=0.05)
    assert B.bb_sf(0, 2, 5.0, 45.0) == pytest.approx(stats.betabinom.sf(0, 2, 5, 45), rel=1e-12)


def test_scalar_midp_never_turns_nan_into_p_floor():
    """Regression: Python max(P_FLOOR, nan) is P_FLOOR, so a NaN predictive
    location scored p = 1e-300 (a certain false alarm) on the scalar path."""
    for post in (B.NIG(NAN, 1.0, 2.0, 1.0), B.NIG(0.0, 1.0, 2.0, NAN)):
        assert all(math.isnan(v) for v in B.t_midp(1.0, post))
        u, p = B.t_midp(np.array([1.0]), post)
        assert np.isnan(u[0]) and np.isnan(p[0])
    assert B._p2_1(0.2, NAN) == (pytest.approx(NAN, nan_ok=True), pytest.approx(NAN, nan_ok=True))
    assert all(math.isnan(v) for v in B._p2_1(NAN, 0.3))
    assert B._p2_1(0.2, 0.8) == (0.2, pytest.approx(0.4))


def test_nb_upper_tail_exact_at_small_size():
    """Regression: with m > r the upper tail fed betainc the rounded near-1
    argument y = m/(r+m); at r = 1e-6, m = 1e3 that was 9e-9 relative off.
    Reference: betaincc on the directly formed small argument q = r/(r+m)."""
    for r in (1e-6, 1e-4, 1e-2):
        for m in (1e2, 1e3, 1e4, 3e4):
            k = np.array([0.0, 1.0, 50.0, 5e3, 5e4])
            ref = sp.betaincc(r, k + 1.0, r / (r + m))
            assert rel(B.nb_sf(k, m, r), ref) < 1e-11
            assert rel([B.nb_sf(float(x), m, r) for x in k], ref) < 1e-11
            _, p2 = B.nb_midp(k[1:], m, r)                   # upper side is the small one
            ref_v = 0.5 * (sp.betaincc(r, k[1:], r / (r + m)) + ref[1:])
            assert rel(p2, 2.0 * ref_v) < 1e-11


def test_bb_large_concentration_and_extreme_shapes():
    """Regression: betaln(k+a, n-k+b) - betaln(a, b) cancels at large a + b
    (bb_sf(5, 10, 1e300, 1e300) was 1.53), and the pmf ratio recurrence
    raised ValueError (log of 0 / overflow) for extreme but finite a, b."""
    for c in (1e8, 1e12, 1e15, 2e300):
        a, b = 0.5 * c, 0.5 * c                              # -> Binomial(10, 0.5)
        assert B.bb_sf(5, 10, a, b) == pytest.approx(stats.binom.sf(5, 10, 0.5), rel=1e-6)
        assert B.bb_cdf(5, 10, a, b) == pytest.approx(stats.binom.cdf(5, 10, 0.5), rel=1e-6)
        assert np.allclose(B.bb_logpmf(np.arange(11.0), 10, a, b),
                           stats.binom.logpmf(np.arange(11), 10, 0.5), rtol=1e-6)
    # large n, large c: log pmf against a sum of logs of the rising factorials
    # (fsum, ~1e-12); the betaln form was 4e-9 off here (scipy's betabinom.sf
    # returns 3.4e-9 for this 4.7e-85 tail).
    n, k, a, b = 100000, 12000, 1e5, 9e5
    ref = (math.fsum(math.log(n - i) - math.log(i + 1) for i in range(k))
           + math.fsum(math.log(a + i) for i in range(k))
           + math.fsum(math.log(b + i) for i in range(n - k))
           - math.fsum(math.log(a + b + i) for i in range(n)))
    assert abs(B.bb_logpmf(k, n, a, b) - ref) < 1e-9
    lp, tail = ref, 0.0                                        # exact upper tail from ref
    for j in range(k, k + 600):
        lp += math.log((n - j) * (j + a) / ((j + 1) * (n - j - 1 + b)))
        tail += math.exp(lp)
    assert B.bb_sf(k, n, a, b) == pytest.approx(tail, rel=1e-8)
    # finite but extreme shapes: no exception; valid probabilities or NaN
    for k, n, a, b in ((0, 10, 1e-320, 1e3), (5, 10, 1e307, 1e308), (1, 10, 1e-300, 1.0)):
        u, p = B.bb_midp(k, n, a, b)
        assert 0.0 <= u <= 1.0 and B.P_FLOOR <= p <= 1.0
        assert 0.0 <= B.bb_sf(k, n, a, b) <= 1.0 and 0.0 <= B.bb_cdf(k, n, a, b) <= 1.0
        ua, pa = B.bb_midp(np.array([k]), n, a, b)
        assert ua[0] == pytest.approx(u) and pa[0] == pytest.approx(p)
    # a + b overflowing to inf is an invalid model -> NaN everywhere, never p = 1e-300
    assert all(math.isnan(v) for v in B.bb_midp(1, 10, 1e308, 1e308))
    assert math.isnan(B.bb_sf(1, 10, 1e308, 1e308)) and math.isnan(B.bb_cdf(1, 10, 1e308, 1e308))
    assert math.isnan(B.bb_logpmf(1, 10, 1e308, 1e308))
    assert math.isnan(B.bb_ppf(0.5, 10, 1e308, 1e308))
    assert np.isnan(B.bb_midp(np.array([1.0]), 10, 1e308, 1e308)[1][0])
