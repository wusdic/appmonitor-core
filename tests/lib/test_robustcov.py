"""Tests for engines/behavior/lib/robustcov.py: OAS, the robust C-step fit,
eigen floor / PCA, Hotelling prediction p, SPE with the Box approximation,
reconstruction-based contributions, conditional imputation and the
Cholesky LRU cache.

Scenario tests follow docs/lib3/engines.md B06 (a)-(c): a rho = 0.95 pair,
a correlation break (+2, -2) against a move along the correlation, a missing
dimension, and the null calibration of the prediction p. Statistical checks
use fixed seeds (deterministic). Maths oracles are written from the module
docstrings, independently of the implementation.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from app.engines.behavior.lib import robustcov as R  # noqa: E402

NAN = float("nan")
RHO = 0.95


# ------------------------------------------------------------------ helpers
def _pair_sigma(p: int, rho: float = RHO) -> np.ndarray:
    S = np.eye(p)
    S[0, 1] = S[1, 0] = rho
    return S


def _draw(n: int, Sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((n, Sigma.shape[0])) @ np.linalg.cholesky(Sigma).T


def _ref_oas(X, w=None):
    """oas exactly as its docstring specifies (test oracle)."""
    X = np.asarray(X, float)
    keep = np.all(np.isfinite(X), axis=1)
    if w is not None:
        w = np.asarray(w, float)
        keep &= np.isfinite(w) & (w > 0)
    X = X[keep]
    n, p = X.shape
    wn = np.full(n, 1.0 / n) if w is None else w[keep] / w[keep].sum()
    n_eff = 1.0 / np.sum(wn ** 2)
    mu = wn @ X
    Xc = X - mu
    S = (Xc * wn[:, None]).T @ Xc
    tr, tr2 = np.trace(S), np.trace(S @ S)
    num = (1 - 2 / p) * tr2 + tr ** 2
    den = (n_eff + 1 - 2 / p) * (tr2 - tr ** 2 / p)
    rho = 1.0 if den <= 0 else min(1.0, num / den)
    return mu, (1 - rho) * S + rho * tr / p * np.eye(p), rho


def _kl(S_hat: np.ndarray, S_true: np.ndarray) -> float:
    """KL(N(0, S_true) || N(0, S_hat)): 0 iff equal, scale- and shape-aware."""
    p = S_true.shape[0]
    M = np.linalg.solve(S_hat, S_true)
    return 0.5 * (np.trace(M) - p - np.linalg.slogdet(M)[1])


def _fit_model(X):
    """The B06 pipeline on training rows: robust fit, PCA, Box on training SPE."""
    mu, S = R.c_step_oas(X)
    U, lam, k = R.pca_k(S)
    spe_tr = np.array([R.spe(x - mu, U) for x in X])
    g, h = R.spe_box_params(spe_tr)
    return mu, S, U, k, g, h


def _score(model, x, n):
    mu, S, U, k, g, h = model
    z = np.asarray(x, float) - mu
    t2, q = R.CholCache(S).t2(z)
    return R.hotelling_pred_p(t2, n, q), R.spe_p(R.spe(z, U), g, h), t2


# ------------------------------------------------------------------ winsorize
def test_winsorize_clips_and_preserves_nan():
    X = np.array([[-9.0, 0.5, NAN], [np.inf, -np.inf, 3.9]])
    Y = R.winsorize(X)
    assert Y[0, 0] == -4.0 and Y[0, 1] == 0.5 and math.isnan(Y[0, 2])
    assert Y[1, 0] == 4.0 and Y[1, 1] == -4.0 and Y[1, 2] == 3.9
    assert X[0, 0] == -9.0                               # input untouched
    assert R.winsorize(X, c=1.0)[1, 2] == 1.0
    assert R.winsorize([1, 2, 30]).dtype == np.float64


# ------------------------------------------------------------------ oas
def test_oas_matches_docstring_formula():
    rng = np.random.default_rng(0)
    X = _draw(40, _pair_sigma(6, 0.6), rng)
    mu, S, rho = R.oas(X)
    mu_r, S_r, rho_r = _ref_oas(X)
    assert 0.0 < rho < 1.0
    assert rho == pytest.approx(rho_r, rel=1e-12)
    np.testing.assert_allclose(mu, mu_r, atol=1e-14)
    np.testing.assert_allclose(S, S_r, atol=1e-13)
    np.testing.assert_allclose(S, S.T, atol=0)


def test_oas_weighted_matches_formula_and_weighted_moments():
    rng = np.random.default_rng(1)
    X = _draw(60, _pair_sigma(4, 0.8), rng)
    w = rng.uniform(0.2, 3.0, 60)
    mu, S, rho = R.oas(X, w)
    mu_r, S_r, rho_r = _ref_oas(X, w)
    assert rho == pytest.approx(rho_r, rel=1e-12)
    np.testing.assert_allclose(S, S_r, atol=1e-13)
    np.testing.assert_allclose(mu, np.average(X, axis=0, weights=w), atol=1e-14)
    # Sigma is the weighted MLE covariance shrunk towards mu_t I.
    C = np.cov(X.T, aweights=w, bias=True)
    np.testing.assert_allclose(S, (1 - rho) * C + rho * np.trace(C) / 4 * np.eye(4), atol=1e-13)
    # Uniform weights (any scale) are the unweighted fit; huge weights do not overflow.
    for c in (1.0, 7.5, 1e300):
        m2, S2, r2 = R.oas(X, np.full(60, c))
        m1, S1, r1 = R.oas(X)
        np.testing.assert_allclose(m2, m1, atol=1e-14)
        np.testing.assert_allclose(S2, S1, atol=1e-13)
        assert r2 == pytest.approx(r1, rel=1e-12)


def test_oas_drops_nonfinite_rows_and_bad_weights():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((30, 3))
    Xd = np.vstack([X, [[NAN, 0, 0], [np.inf, 1, 1], [5, 5, 5], [6, 6, 6]]])
    w = np.r_[np.ones(30), 1.0, 1.0, 0.0, NAN]
    mu, S, rho = R.oas(Xd, w)
    mu0, S0, rho0 = R.oas(X)
    np.testing.assert_allclose(mu, mu0, atol=1e-14)
    np.testing.assert_allclose(S, S0, atol=1e-14)
    assert rho == rho0
    mu, S, _ = R.oas(Xd, np.r_[np.ones(33), -1.0])      # negative weight dropped
    mu1, S1, _ = R.oas(np.vstack([X, [[5, 5, 5]]]))
    np.testing.assert_allclose(mu, mu1, atol=1e-14)
    np.testing.assert_allclose(S, S1, atol=1e-14)
    with pytest.raises(ValueError):
        R.oas(X, np.ones(5))
    with pytest.raises(ValueError):
        R.oas(np.ones(5))


def test_oas_small_n_and_p1():
    mu, S, rho = R.oas(np.array([[1.0, 2.0]]))
    np.testing.assert_array_equal(mu, [1.0, 2.0])
    np.testing.assert_array_equal(S, np.eye(2))
    assert rho == 1.0
    mu, S, rho = R.oas(np.full((3, 2), NAN))             # no usable row
    np.testing.assert_array_equal(mu, [0.0, 0.0])
    np.testing.assert_array_equal(S, np.eye(2))
    assert rho == 1.0
    # p = 1: shrinkage is a no-op, Sigma is the MLE variance.
    x = np.array([[1.0], [2.0], [4.0], [7.0]])
    mu, S, rho = R.oas(x)
    assert S.shape == (1, 1) and S[0, 0] == pytest.approx(np.var(x))
    assert mu[0] == pytest.approx(3.5)
    # S proportional to I: zero denominator -> rho = 1 (and Sigma = S).
    _, S, rho = R.oas(np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]))
    assert rho == 1.0
    np.testing.assert_allclose(S, 0.5 * np.eye(2), atol=1e-15)


def test_oas_is_positive_definite_when_n_below_p():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((10, 30))
    _, S, rho = R.oas(X)
    assert rho > 0.3
    assert np.linalg.eigvalsh(S).min() > 0.0


def test_oas_close_to_sklearn_at_large_p():
    # sklearn drops the 2/p terms of Chen et al. eq. 23; at p = 60 they agree closely.
    from sklearn.covariance import OAS
    rng = np.random.default_rng(4)
    X = _draw(100, 0.3 * np.eye(60) + 0.7 * np.ones((60, 60)) * 0.1, rng)
    _, S, rho = R.oas(X)
    sk = OAS().fit(X)
    assert rho == pytest.approx(sk.shrinkage_, rel=0.05)
    assert np.linalg.norm(S - sk.covariance_) / np.linalg.norm(sk.covariance_) < 0.02


# ------------------------------------------------------------------ eigen floor / pca
def test_eigen_floor():
    S = np.diag([10.0, 1.0, 1e-9])
    F = R.eigen_floor(S)
    lam = np.linalg.eigvalsh(F)
    assert lam.min() == pytest.approx(1e-3 * np.mean([10.0, 1.0, 1e-9]), rel=1e-9)
    assert lam.max() == pytest.approx(10.0)
    np.testing.assert_array_equal(F, F.T)
    # Already above the floor: returned symmetrised, values untouched.
    A = np.array([[2.0, 0.5], [0.5 + 1e-12, 1.0]])
    np.testing.assert_array_equal(R.eigen_floor(A), 0.5 * (A + A.T))
    # Indefinite -> PD; degenerate (zero) -> rel * I.
    B = np.array([[1.0, 2.0], [2.0, 1.0]])
    assert np.linalg.eigvalsh(R.eigen_floor(B)).min() > 0
    np.testing.assert_allclose(R.eigen_floor(np.zeros((3, 3))), 1e-3 * np.eye(3), atol=1e-18)
    assert R.eigen_floor(np.zeros((0, 0))).shape == (0, 0)


def test_pca_k():
    # Identity: 9 of 10 equal eigenvalues explain exactly 90 % (tolerance on the sum).
    U, lam, k = R.pca_k(np.eye(10))
    assert k == 9 and U.shape == (10, 9) and lam.shape == (9,)
    rng = np.random.default_rng(5)
    Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    ev = np.array([5.0, 3.0, 1.0, 0.5, 0.5])
    S = (Q * ev) @ Q.T
    U, lam, k = R.pca_k(S)
    assert k == 3
    np.testing.assert_allclose(lam, [5.0, 3.0, 1.0], rtol=1e-10)
    np.testing.assert_allclose(U.T @ U, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(S @ U, U * lam, atol=1e-10)
    piv = np.argmax(np.abs(U), axis=0)
    assert np.all(U[piv, np.arange(3)] > 0)              # deterministic sign
    assert R.pca_k(S, var_frac=1.0)[2] == 5
    assert R.pca_k(S, var_frac=0.0)[2] == 1
    assert R.pca_k(S, var_frac=0.5)[2] == 1
    assert R.pca_k(np.zeros((4, 4)))[2] == 4


# ------------------------------------------------------------------ Hotelling / WH
def test_hotelling_pred_p_formula_and_edges():
    for t2, n, q in [(5.0, 200, 3), (40.0, 336, 20), (0.3, 50, 1), (12.0, 30.5, 4)]:
        F = t2 * n * (n - q) / (q * (n - 1) * (n + 1))
        assert R.hotelling_pred_p(t2, n, q) == pytest.approx(stats.f.sf(F, q, n - q), rel=1e-10)
    # Large n -> chi2_q.
    assert R.hotelling_pred_p(9.0, 1e9, 3) == pytest.approx(stats.chi2.sf(9.0, 3), rel=1e-6)
    # The prediction p is always more conservative than chi2 at finite n.
    assert R.hotelling_pred_p(30.0, 200, 20) > stats.chi2.sf(30.0, 20)
    for args in [(5.0, 4, 3), (5.0, 3, 3), (5.0, 100, 0), (NAN, 100, 3), (5.0, NAN, 3),
                 (5.0, math.inf, 3)]:
        assert math.isnan(R.hotelling_pred_p(*args)), args
    assert R.hotelling_pred_p(0.0, 100, 3) == 1.0
    assert R.hotelling_pred_p(-1e-12, 100, 3) == 1.0
    assert R.hotelling_pred_p(1e6, 100, 3) == pytest.approx(
        stats.f.sf(1e6 * 100 * 97 / (3 * 99 * 101), 3, 97), rel=1e-8)   # ~1e-194, exact
    assert R.hotelling_pred_p(1e20, 100, 3) == 1e-300
    assert R.hotelling_pred_p(math.inf, 100, 3) == 1e-300
    ps = [R.hotelling_pred_p(t, 100, 5) for t in (1.0, 5.0, 10.0, 20.0, 50.0)]
    assert all(a > b for a, b in zip(ps, ps[1:]))


def test_wilson_hilferty():
    for q in (5, 20, 52):
        for t2 in (0.5 * q, q, 1.5 * q, 2.0 * q):
            exact = stats.norm.isf(stats.chi2.sf(t2, q))
            assert R.wilson_hilferty(t2, q) == pytest.approx(exact, abs=0.06), (q, t2)
        exact = stats.norm.isf(stats.chi2.sf(3.0 * q, q))     # deep tail: within 2 %
        assert R.wilson_hilferty(3.0 * q, q) == pytest.approx(exact, rel=0.02)
    q = 7
    assert R.wilson_hilferty(q * (1 - 2 / (9 * q)) ** 3, q) == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(R.wilson_hilferty(NAN, 3)) and math.isnan(R.wilson_hilferty(1.0, 0))
    assert R.wilson_hilferty(-1e-15, 4) == R.wilson_hilferty(0.0, 4)
    assert R.wilson_hilferty(math.inf, 4) == math.inf


# ------------------------------------------------------------------ SPE
def test_spe():
    rng = np.random.default_rng(6)
    U, _, _ = R.pca_k(np.diag([4.0, 3.0, 2.0, 0.1, 0.1]))
    z = rng.standard_normal(5)
    assert R.spe(z, U) == pytest.approx(z @ z - np.sum((U.T @ z) ** 2), rel=1e-12)
    assert R.spe(U @ np.array([1.0, -2.0, 0.5]), U) == pytest.approx(0.0, abs=1e-24)
    assert R.spe(np.array([0, 0, 0, 3.0, 4.0]), U) == pytest.approx(25.0)
    assert math.isnan(R.spe(np.array([1.0, NAN, 0, 0, 0]), U))


def test_spe_box_params_and_p():
    rng = np.random.default_rng(7)
    g0, h0 = 0.3, 4.0
    x = g0 * rng.chisquare(h0, 20000)
    g, h = R.spe_box_params(np.r_[x, NAN, np.inf])
    assert g == pytest.approx(g0, rel=0.05) and h == pytest.approx(h0, rel=0.05)
    assert all(math.isnan(v) for v in R.spe_box_params(np.arange(9.0)))
    assert all(math.isnan(v) for v in R.spe_box_params(np.full(50, 2.0)))
    assert all(math.isnan(v) for v in R.spe_box_params(np.zeros(50)))
    assert R.spe_p(1.2, g, h) == pytest.approx(stats.chi2.sf(1.2 / g, h), rel=1e-10)
    assert R.spe_p(0.0, g, h) == 1.0 and R.spe_p(-1e-18, g, h) == 1.0
    assert R.spe_p(1e9, g, h) == 1e-300
    for args in [(NAN, g, h), (1.0, NAN, h), (1.0, g, NAN), (1.0, 0.0, h), (1.0, g, -1.0),
                 (1.0, math.inf, h)]:
        assert math.isnan(R.spe_p(*args)), args
    # Calibrated on its own training distribution.
    ps = np.array([R.spe_p(v, g, h) for v in x[:2000]])
    assert stats.kstest(ps, "uniform").statistic < 0.05


# ------------------------------------------------------------------ RBC
def test_rbc_formula_and_reconstruction_identity():
    rng = np.random.default_rng(8)
    A = rng.standard_normal((5, 5))
    Sigma = A @ A.T + 0.5 * np.eye(5)
    Sinv = np.linalg.inv(Sigma)
    z = rng.standard_normal(5) * 2
    rbc, pv = R.rbc_contributions(z, Sinv)
    u = Sinv @ z
    np.testing.assert_allclose(rbc, u ** 2 / np.diag(Sinv), rtol=1e-12)
    np.testing.assert_allclose(pv, stats.chi2.sf(rbc, 1), rtol=1e-10)
    # RBC_f is the drop in T2 when feature f is reconstructed optimally.
    t2 = z @ Sinv @ z
    for f in range(5):
        e = np.zeros(5)
        e[f] = 1.0
        c = (e @ Sinv @ z) / Sinv[f, f]
        zr = z - c * e
        assert t2 - zr @ Sinv @ zr == pytest.approx(rbc[f], rel=1e-9)


def test_rbc_null_is_chi2_1_with_and_without_missing():
    rng = np.random.default_rng(9)
    Sigma = _pair_sigma(4, 0.8) + 0.3 * np.eye(4)
    Sinv = np.linalg.inv(Sigma)
    Z = _draw(3000, Sigma, rng)
    p_full = np.array([R.rbc_contributions(z, Sinv)[1] for z in Z])
    for f in range(4):
        assert stats.kstest(p_full[:, f], "uniform").statistic < 0.03
    Zm = Z.copy()
    Zm[:, 1] = NAN
    out = np.array([R.rbc_contributions(z, Sinv)[1] for z in Zm])
    assert np.all(np.isnan(out[:, 1]))
    for f in (0, 2, 3):
        assert stats.kstest(out[:, f], "uniform").statistic < 0.03
    # Observed terms = RBC on the observed sub-model with inv(Sigma_oo).
    o = [0, 2, 3]
    z = Zm[0]
    rbc, _ = R.rbc_contributions(z, Sinv)
    ref, _ = R.rbc_contributions(z[o], np.linalg.inv(Sigma[np.ix_(o, o)]))
    np.testing.assert_allclose(rbc[o], ref, rtol=1e-10)
    # Numerators are Sinv z_c of the conditionally imputed vector.
    zc = R.conditional_impute(z, Sigma, np.isfinite(z))
    Poo = np.linalg.inv(Sigma[np.ix_(o, o)])
    np.testing.assert_allclose((Sinv @ zc)[o], Poo @ z[o], atol=1e-10)


def test_rbc_edges():
    rbc, pv = R.rbc_contributions(np.full(3, NAN), np.eye(3))
    assert np.all(np.isnan(rbc)) and np.all(np.isnan(pv))
    rbc, pv = R.rbc_contributions(np.array([50.0, 0.0]), np.eye(2))
    assert pv[0] == 1e-300 and pv[1] == 1.0
    with pytest.raises(ValueError):
        R.rbc_contributions(np.zeros(3), np.eye(2))


# ------------------------------------------------------------------ imputation
def test_conditional_impute_properties():
    rng = np.random.default_rng(10)
    A = rng.standard_normal((6, 6))
    Sigma = A @ A.T + np.eye(6)
    z = rng.standard_normal(6)
    obs = np.array([True, False, True, True, False, True])
    z_in = z.copy()
    z_in[~obs] = NAN
    zc = R.conditional_impute(z_in, Sigma, obs)
    o, m = np.flatnonzero(obs), np.flatnonzero(~obs)
    np.testing.assert_array_equal(zc[o], z[o])
    ref = Sigma[np.ix_(m, o)] @ np.linalg.solve(Sigma[np.ix_(o, o)], z[o])
    np.testing.assert_allclose(zc[m], ref, rtol=1e-10)
    # Minimum-Mahalanobis completion: T2 unchanged and zero gradient on m.
    Sinv = np.linalg.inv(Sigma)
    assert zc @ Sinv @ zc == pytest.approx(z[o] @ np.linalg.solve(Sigma[np.ix_(o, o)], z[o]),
                                           rel=1e-10)
    np.testing.assert_allclose((Sinv @ zc)[m], 0.0, atol=1e-10)
    assert np.isnan(z_in[1])                             # input untouched
    # Finite values at unobserved positions are replaced too.
    np.testing.assert_allclose(R.conditional_impute(z, Sigma, obs), zc, rtol=1e-12)


def test_conditional_impute_edges():
    Sigma = _pair_sigma(3)
    z = np.array([1.0, 2.0, 3.0])
    out = R.conditional_impute(z, Sigma, np.ones(3, bool))
    np.testing.assert_array_equal(out, z)
    assert out is not z
    np.testing.assert_array_equal(R.conditional_impute(z, Sigma, np.zeros(3, bool)), 0.0)
    # Marked observed but NaN -> treated as missing.
    out = R.conditional_impute(np.array([2.0, NAN, 0.0]), Sigma, np.ones(3, bool))
    assert out[1] == pytest.approx(RHO * 2.0)
    with pytest.raises(ValueError):
        R.conditional_impute(z, Sigma, np.ones(2, bool))


# ------------------------------------------------------------------ CholCache
def test_cholcache_t2_matches_direct_solve():
    rng = np.random.default_rng(11)
    A = rng.standard_normal((7, 7))
    Sigma = A @ A.T + np.eye(7)
    cc = R.CholCache(Sigma)
    z = rng.standard_normal(7)
    t2, q = cc.t2(z)
    assert q == 7 and t2 == pytest.approx(z @ np.linalg.solve(Sigma, z), rel=1e-10)
    z[[2, 5]] = [NAN, np.inf]
    t2, q = cc.t2(z)
    o = [0, 1, 3, 4, 6]
    assert q == 5
    assert t2 == pytest.approx(z[o] @ np.linalg.solve(Sigma[np.ix_(o, o)], z[o]), rel=1e-10)
    idx, L = cc.factor(np.isfinite(z))
    np.testing.assert_array_equal(idx, o)
    np.testing.assert_allclose(L @ L.T, Sigma[np.ix_(o, o)], atol=1e-10)
    assert np.all(np.triu(L, 1) == 0)
    t2, q = cc.t2(np.full(7, NAN))
    assert math.isnan(t2) and q == 0
    with pytest.raises(ValueError):
        cc.t2(np.zeros(3))
    with pytest.raises(ValueError):
        cc.factor(np.ones(3, bool))


def test_cholcache_lru_and_invalidation():
    Sigma = np.eye(4) + 0.1
    cc = R.CholCache(Sigma, maxsize=2)
    m1, m2, m3 = (np.array(v, bool) for v in ([1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 1]))
    f1 = cc.factor(m1)
    cc.factor(m2)
    assert cc.factor(m1) is f1                           # hit, and m1 is now most recent
    cc.factor(m3)                                        # evicts m2 (least recent)
    assert list(cc._cache) == [m1.tobytes(), m3.tobytes()]
    assert cc.factor(np.array([1, 1, 0, 0])) is f1       # int mask keys like bool
    # Cached arrays are shared, hence read-only.
    with pytest.raises(ValueError):
        f1[1][0, 0] = 99.0
    # The cache owns a copy of Sigma; set_sigma invalidates.
    Sigma[0, 0] = 50.0
    assert cc.t2(np.array([1.0, NAN, NAN, NAN]))[0] == pytest.approx(1.0 / 1.1)
    cc.set_sigma(np.eye(4) * 4.0)
    assert len(cc._cache) == 0
    assert cc.t2(np.array([2.0, NAN, NAN, NAN])) == (pytest.approx(1.0), 1)
    with pytest.raises(ValueError):
        cc.set_sigma(np.ones(3))
    nc = R.CholCache(np.eye(3), maxsize=0)
    nc.t2(np.ones(3))
    assert len(nc._cache) == 0


def test_cholcache_indefinite_submatrix_falls_back_to_floor():
    # Rounding-level indefinite Sigma (a rank-1 matrix): the floor keeps T2 finite.
    v = np.array([1.0, 1.0, 1.0])
    S = np.outer(v, v)
    S[0, 1] = S[1, 0] = 1.0 + 1e-9
    t2, q = R.CholCache(S).t2(np.array([1.0, -1.0, 0.5]))
    assert q == 3 and math.isfinite(t2) and t2 > 0


# ------------------------------------------------------------------ c_step_oas
def test_c_step_oas_clean_fit_is_consistent():
    rng = np.random.default_rng(12)
    Sigma = _pair_sigma(2)
    X = _draw(4000, Sigma, rng)
    mu, S = R.c_step_oas(X)
    assert np.all(np.abs(mu) < 0.06)
    assert _kl(S, Sigma) < 0.01
    assert S[1, 0] / S[0, 0] == pytest.approx(RHO, abs=0.02)
    assert np.allclose(S, S.T) and np.linalg.eigvalsh(S).min() > 0


@pytest.mark.parametrize("kind", ["cluster", "scatter", "few_dims_far", "pair_break"])
def test_c_step_oas_resists_20pct_outliers(kind):
    rng = np.random.default_rng(13)
    p, n = (5, 336)
    Sigma = _pair_sigma(p)
    X = _draw(n, Sigma, rng)
    k = int(0.2 * n)
    if kind == "cluster":                  # tight cluster, collapses onto the clip
        X[:k] = 6.0 + 0.3 * rng.standard_normal((k, p))
    elif kind == "scatter":
        X[:k] = 5.0 * rng.standard_normal((k, p))
    elif kind == "few_dims_far":           # a sustained shift in 3 features
        X[:k, 2:5] = 8.0 + rng.standard_normal((k, 3))
    else:                                  # correlation break, each |z| = 3
        X[:k, 0] = 3.0 + 0.3 * rng.standard_normal(k)
        X[:k, 1] = -3.0 + 0.3 * rng.standard_normal(k)
    mu, S = R.c_step_oas(X)
    mu_c, S_c = R.c_step_oas(X[k:])        # the same fit on the clean rows only
    _, S_o, _ = R.oas(R.winsorize(X))
    assert np.linalg.norm(mu) < 0.3, mu
    assert _kl(S, Sigma) < 0.25 and _kl(S, S_c) < 0.1
    assert _kl(S_o, Sigma) > 0.5                         # the test has teeth
    # The break is still visible after the robust fit.
    z = np.zeros(p)
    z[:2] = [2.0, -2.0]
    assert R.CholCache(S).t2(z - mu)[0] > 30.0


def test_c_step_oas_h_parameter():
    rng = np.random.default_rng(23)
    Sigma = _pair_sigma(3, 0.7)
    X = _draw(400, Sigma, rng)
    for h in (0.5, 0.75, 0.9, 1.0):
        mu, S = R.c_step_oas(X, h=h)
        assert _kl(S, Sigma) < 0.03, h
        assert np.all(np.abs(mu) < 0.15), h


def test_c_step_oas_weights():
    rng = np.random.default_rng(14)
    X = _draw(300, _pair_sigma(3), rng)
    Xc = X.copy()
    Xc[:60] = 6.0
    w = np.ones(300)
    w[:60] = 0.0
    mu_w, S_w = R.c_step_oas(Xc, w)
    mu_c, S_c = R.c_step_oas(X[60:])
    np.testing.assert_allclose(mu_w, mu_c, atol=1e-12)
    np.testing.assert_allclose(S_w, S_c, atol=1e-12)
    mu1, S1 = R.c_step_oas(X, np.full(300, 0.7))
    mu0, S0 = R.c_step_oas(X)
    np.testing.assert_allclose(mu1, mu0, atol=1e-12)
    np.testing.assert_allclose(S1, S0, atol=1e-12)
    # Trust weights move the fit towards the trusted rows.
    w = np.where(np.arange(300) < 150, 1.0, 0.2)
    mu_t, _ = R.c_step_oas(X + np.where(np.arange(300) < 150, 0.0, 1.0)[:, None], w)
    assert np.all(mu_t < 0.5)


def test_c_step_oas_degenerate_inputs():
    rng = np.random.default_rng(15)
    mu, S = R.c_step_oas(np.array([[1.0, 2.0]]))
    np.testing.assert_array_equal(mu, [1.0, 2.0])
    np.testing.assert_allclose(S, np.eye(2))
    mu, S = R.c_step_oas(np.full((5, 3), NAN))
    np.testing.assert_array_equal(mu, 0.0)
    # n < p, identical rows, a constant column, NaN rows: finite and PD, no warnings.
    cases = [rng.standard_normal((20, 52)), np.zeros((50, 4)),
             np.c_[rng.standard_normal((80, 3)), np.ones(80)],
             np.r_[rng.standard_normal((80, 3)), np.full((5, 3), NAN)],
             np.r_[np.zeros((60, 3)), rng.standard_normal((20, 3))]]
    with np.errstate(all="raise"):
        for X in cases:
            mu, S = R.c_step_oas(X)
            assert np.all(np.isfinite(mu)) and np.all(np.isfinite(S))
            assert np.linalg.eigvalsh(S).min() > 0
            R.CholCache(S).t2(np.ones(S.shape[0]))


def test_c_step_oas_row_order_invariant_and_deterministic():
    rng = np.random.default_rng(16)
    X = _draw(200, _pair_sigma(4), rng)
    X[:30] += 5.0
    mu, S = R.c_step_oas(X)
    perm = rng.permutation(200)
    mu_p, S_p = R.c_step_oas(X[perm])
    np.testing.assert_allclose(mu, mu_p, atol=1e-12)
    np.testing.assert_allclose(S, S_p, atol=1e-12)
    mu2, S2 = R.c_step_oas(X)
    np.testing.assert_array_equal(S, S2)


def test_loo_d2_matches_brute_force():
    # Review regression: the closed form must equal refitting without row i
    # (rho and mu_t held at the full-subset values, as documented).
    rng = np.random.default_rng(30)
    m, p = 40, 6
    X = _draw(m, _pair_sigma(p), rng)
    w = rng.uniform(0.5, 2.0, m)
    for ws, tol in ((None, 1e-12), (np.full(m, 0.7), 1e-12), (w, 0.01)):
        mu, S, rho = R.oas(X, ws)
        mut = np.trace(S) / p
        got = R._loo_d2(X, ws, mu, S, rho)
        ref = np.empty(m)
        for i in range(m):
            k = np.arange(m) != i
            wk = np.full(m - 1, 1.0) if ws is None else ws[k]
            wk = wk / wk.sum()
            mui = wk @ X[k]
            Xc = X[k] - mui
            Si = (1 - rho) * (Xc * wk[:, None]).T @ Xc + rho * mut * np.eye(p)
            d = X[i] - mui
            ref[i] = d @ np.linalg.solve(Si, d)
        np.testing.assert_allclose(got, ref, rtol=tol)
        # Out-of-sample d2 exceed the in-sample ones row by row.
        assert np.all(got > R._mahal2(X - mu, S))


def test_c_step_oas_calibrated_at_b06_size():
    """Review regression: at B06's real size (336 x 52, factor-correlated
    zi) the h-subset rows' in-sample d2 were mixed with out-of-sample d2 of
    the rest; ~21 % of clean rows were cut and Sigma came out too tight:
    2.0-3.1 % false alarms at the 1 % level (0.35 % at 1e-3) over 8 seeds.
    With leave-one-out d2: 0.7-0.9 % (<= 0.13 % at 1e-3)."""
    rng = np.random.default_rng(0)
    n, p, fits, m = 336, 52, 15, 400
    A = rng.standard_normal((p, 5))
    C = A @ A.T + np.diag(rng.uniform(0.3, 1.0, p))
    d = np.sqrt(np.diag(C))
    L = np.linalg.cholesky(C / np.outer(d, d))
    ps = []
    for _ in range(fits):
        mu, S = R.c_step_oas(rng.standard_normal((n, p)) @ L.T)
        Z = rng.standard_normal((m, p)) @ L.T - mu
        t2 = np.einsum("ij,ij->i", Z, np.linalg.solve(S, Z.T).T)
        ps += [R.hotelling_pred_p(v, n, p) for v in t2]
    ps = np.array(ps)
    assert 0.003 < np.mean(ps < 0.01) < 0.015
    assert np.mean(ps < 1e-3) < 0.0025


def test_wmedian_half_within_rounding():
    # Review regression: a cumulative weight 1 ulp BELOW one half made the
    # weighted median jump to the next order statistic (3.0, not 2.5).
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert R._wmedian(x, np.array([1.0, 1.0, 1.0, 1.0 + 1e-15])) == pytest.approx(2.5)
    assert R._wmedian(x, np.array([1.0, 1.0 + 1e-15, 1.0, 1.0])) == pytest.approx(2.5)
    assert R._wmedian(x, np.array([0.1] * 4)) == pytest.approx(2.5)
    assert R._wmedian(x, np.array([1.0, 1.0, 1.0, 3.0])) == pytest.approx(3.5)
    assert R._wmedian(x, np.array([1.0, 1.0, 1.0, 4.0])) == 4.0


def test_spe_without_residual_subspace_gives_no_evidence():
    # Review regression: near-isotropic Sigma at p < 10 gives k = p, and the
    # explicit residual was ~1e-31 round-off; the Box fit on that noise
    # produced p < 0.01 on ~1 % of null points (and p ~ 1e-10 at times).
    rng = np.random.default_rng(31)
    for p in (2, 3, 5):
        X = rng.standard_normal((336, p))
        mu, S = R.c_step_oas(X)
        U, _, k = R.pca_k(S)
        assert k == p
        spe_tr = np.array([R.spe(x - mu, U) for x in X])
        assert np.all(spe_tr == 0.0)
        g, h = R.spe_box_params(spe_tr)
        assert math.isnan(g) and math.isnan(h)
        z = rng.standard_normal(p) * 3.0
        assert R.spe(z, U) == 0.0 and math.isnan(R.spe_p(R.spe(z, U), g, h))
    assert math.isnan(R.spe(np.array([1.0, NAN]), np.eye(2)))
    assert math.isnan(R.spe(np.array([1.0, np.inf]), np.eye(2)))


# ------------------------------------------------------------------ B06 scenarios
@pytest.fixture(scope="module")
def pair_model():
    rng = np.random.default_rng(17)
    X = _draw(336, _pair_sigma(2), rng)
    return _fit_model(X), 336


def test_b06a_correlation_break_vs_move_along_correlation(pair_model):
    model, n = pair_model
    assert model[3] == 1                                  # k: one PC explains > 90 %
    # (+2, -2): each |z| <= 2 but it breaks the correlation.
    p_t2, p_spe, _ = _score(model, [2.0, -2.0], n)
    assert p_spe < 1e-4 and p_t2 < 1e-4
    # (+2, +2) moves along the correlation: SPE not significant. Its T2 is
    # 8 / 1.95 = 4.1 on the true Sigma (chi2_2 p = 0.129), so the spec's
    # 'T2 p < 0.05' does not hold at this size (see the report); the fitted
    # T2 is within sampling error of it, and (+3, +3) is significant.
    p_t2, p_spe, t2 = _score(model, [2.0, 2.0], n)
    assert p_spe > 0.05
    assert p_t2 > 0.01 and t2 == pytest.approx(8 / 1.95, rel=0.4)
    p_t2, p_spe, _ = _score(model, [3.0, 3.0], n)
    assert p_t2 < 0.05 and p_spe > 0.05


def test_b06a_rbc_ranks_the_broken_pair_first():
    rng = np.random.default_rng(18)
    X = _draw(336, _pair_sigma(5), rng)
    mu, S = R.c_step_oas(X)
    z = np.zeros(5)
    z[:2] = [2.0, -2.0]
    rbc, pv = R.rbc_contributions(z - mu, np.linalg.inv(S))
    assert set(np.argsort(-rbc)[:2]) == {0, 1}
    assert np.all(pv[:2] < 1e-4) and np.all(pv[2:] > 0.01)
    # A single-feature fault is attributed to that feature, not its partner.
    z = np.zeros(5)
    z[3] = 4.0
    rbc, _ = R.rbc_contributions(z - mu, np.linalg.inv(S))
    assert int(np.argmax(rbc)) == 3


def test_b06b_missing_dimension(pair_model):
    model, n = pair_model
    mu, S = model[0], model[1]
    z = np.array([2.0, NAN]) - mu
    t2, q = R.CholCache(S).t2(z)
    assert q == 1 and math.isfinite(t2) and t2 == pytest.approx(z[0] ** 2 / S[0, 0])
    assert 0.0 < R.hotelling_pred_p(t2, n, q) < 1.0
    zc = R.conditional_impute(z, S, np.isfinite(z))
    assert zc[1] == pytest.approx(RHO * z[0], abs=0.15)
    # With a larger sample the fitted slope converges on 0.95.
    rng = np.random.default_rng(19)
    mu2, S2 = R.c_step_oas(_draw(4000, _pair_sigma(2), rng))
    zc = R.conditional_impute(np.array([2.0, NAN]), S2, np.array([True, False]))
    assert zc[1] == pytest.approx(RHO * 2.0, abs=0.05)
    # SPE of the completed vector: imputing along the correlation is not a break.
    U, _, _ = R.pca_k(S)
    zc = R.conditional_impute(z, S, np.isfinite(z))
    assert R.spe_p(R.spe(zc, U), model[4], model[5]) > 0.05


def test_b06c_prediction_p_is_uniform_under_the_null():
    """Exact case: a fresh sample mean / unbiased covariance per draw, n = 200.

    T2 is affine invariant, so Sigma = I; for normal rows the sample mean
    (N(0, I/n)) and (n - 1) S (Wishart(n - 1, I)) are independent, which lets
    the 2000 training sets be drawn directly instead of as 8e6 normals.
    """
    rng = np.random.default_rng(20)
    n, draws, q = 200, 2000, 20
    mu = rng.standard_normal((draws, q)) / math.sqrt(n)
    S = stats.wishart(df=n - 1, scale=np.eye(q)).rvs(draws, random_state=rng) / (n - 1)
    d = rng.standard_normal((draws, q)) - mu
    t2 = np.einsum("di,di->d", d, np.linalg.solve(S, d[..., None])[..., 0])
    p_pred = np.array([R.hotelling_pred_p(v, n, q) for v in t2])
    assert stats.kstest(p_pred, "uniform").statistic < 0.05
    assert 0.005 < np.mean(p_pred < 0.01) < 0.015
    # The scaling matters: chi2_q on the same T2 is anti-conservative.
    p_chi2 = stats.chi2.sf(t2, q)
    assert stats.kstest(p_chi2, "uniform").statistic > 0.05
    assert np.mean(p_chi2 < 0.01) > 0.015


def test_b06c_prediction_p_with_the_robust_fit():
    """Engine path: c_step_oas on n = 200 fresh rows per draw, rho = 0.95 pair."""
    rng = np.random.default_rng(21)
    n, draws = 200, 2000
    L = np.linalg.cholesky(_pair_sigma(2))
    ps = np.empty(draws)
    for i in range(draws):
        X = rng.standard_normal((n, 2)) @ L.T
        mu, S = R.c_step_oas(X)
        t2, q = R.CholCache(S).t2(rng.standard_normal(2) @ L.T - mu)
        ps[i] = R.hotelling_pred_p(t2, n, q)
    assert stats.kstest(ps, "uniform").statistic < 0.05
    assert 0.004 < np.mean(ps < 0.01) < 0.025


# ------------------------------------------------------------------ cost
def test_cost_is_cpu_cheap():
    rng = np.random.default_rng(22)
    X = rng.standard_normal((336, 52))
    best = min(_timed(lambda: R.c_step_oas(X)) for _ in range(3))
    assert best < 0.1                                    # ~2.6 ms measured, one thread
    S = R.c_step_oas(X)[1]
    cc = R.CholCache(S)
    z = rng.standard_normal(52)
    cc.t2(z)
    best = min(_timed(lambda: cc.t2(z)) for _ in range(20))
    assert best < 1e-3                                   # ~7 us warm


def _timed(f) -> float:
    t = time.perf_counter()
    f()
    return time.perf_counter() - t
