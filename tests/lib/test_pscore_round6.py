"""pscore round 6 (scoring cost): eb_concentration evaluates its alpha grid
in one array pass; it must return what the per-alpha loop returned, with the
same log-likelihood bits at every grid point."""
from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln

from app.engines.behavior.lib import pscore as SC


def _loop_L(counts, N, prior, alphas=SC.EB_ALPHAS):
    c = np.asarray(counts, dtype=np.float64)
    n = c / c.sum() * float(N)
    f = np.asarray(prior, dtype=np.float64)
    f = np.maximum(f / max(f.sum(), 1e-300), 1e-300)
    nz = n > 0
    nn, ff = n[nz], f[nz]
    return [float(gammaln(a) - gammaln(N + a) + np.sum(gammaln(nn + a * ff) - gammaln(a * ff)))
            for a in alphas]


def _loop(counts, N, prior, alphas=SC.EB_ALPHAS):
    c = np.asarray(counts, dtype=np.float64)
    if c.sum() <= 0 or N <= 0:
        return float(alphas[-1])
    best, arg = -math.inf, float(alphas[-1])
    for a, L in zip(alphas, _loop_L(counts, N, prior, alphas)):
        if L > best:
            best, arg = L, float(a)
    return arg


def test_eb_concentration_equals_the_per_alpha_loop():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        k = int(rng.integers(1, 97))
        c = np.zeros(96)
        c[rng.choice(96, k, replace=False)] = rng.random(k) * rng.choice([1.0, 100.0, 1e4])
        pr = rng.random(96) ** rng.choice([1, 5, 20])
        N = float(rng.choice([0.5, 3.0, 40.0, 900.0, 1e5])) * float(rng.random())
        assert SC.eb_concentration(c, N, pr) == _loop(c, N, pr)
        if N > 0:
            A = np.asarray(SC.EB_ALPHAS)
            n = c / c.sum() * N
            f = np.maximum(pr / pr.sum(), 1e-300)
            nn, ff = n[n > 0], f[n > 0]
            af = A[:, None] * ff[None, :]
            L = gammaln(A) - gammaln(N + A) + np.sum(gammaln(nn[None, :] + af) - gammaln(af), axis=1)
            assert [float(x) for x in L] == _loop_L(c, N, pr)
    assert SC.eb_concentration(np.zeros(96), 10.0, np.ones(96)) == SC.EB_ALPHAS[-1]
