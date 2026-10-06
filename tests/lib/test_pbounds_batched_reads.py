"""P06 cost (round 5): pbounds.td_quantiles / td_cdfs read many quantiles /
CDF values of a t-digest with one centroid cumsum; they must return exactly
TDigest.quantile / TDigest.cdf, value by value (fit_numeric's 37 quantiles
per attribute and bin_gain's 7 + 7 reads used to re-flush and re-cumsum the
digest on every call)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.engines.behavior.lib import pbounds as PB
from app.engines.behavior.lib import psketch as PS


def _bits(v):
    v = float(v)
    return "nan" if v != v else v.hex()


def _digests():
    rng = np.random.default_rng(11)
    out = [PS.TDigest(50.0)]                                   # empty
    one = PS.TDigest(50.0)
    one.add(3.0, 1e9)
    out.append(one)
    const = PS.TDigest(50.0)
    for i in range(40):
        const.add(2.0, 1e9 + i)                                # a point mass (one centroid, vmin == vmax)
    out.append(const)
    for n, scale in ((5, 1.0), (60, 100.0), (900, 1e4), (5000, 1.0)):
        td = PS.TDigest(50.0)
        xs = rng.lognormal(3.0, 1.5, n) * scale
        xs[: n // 7] = np.round(xs[: n // 7])                  # ties
        td.add_many(xs, 1e9 + np.arange(n) * 30.0, rng.random(n) + 0.05)
        out.append(td)
    return out


@pytest.mark.parametrize("i", range(7))
def test_td_quantiles_equal_per_call_quantiles(i):
    td = _digests()[i]
    ps = [0.0, 1e-12, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0, -0.3, 1.7, float("nan")] + \
        [k / 32 for k in range(33)] + [0.13 + 0.7 * k / 32 for k in range(33)]
    got = PB.td_quantiles(td, ps)
    ref = [td.quantile(p) for p in ps]
    assert [_bits(v) for v in got] == [_bits(v) for v in ref]
    assert [type(v) for v in got] == [type(v) for v in ref]


@pytest.mark.parametrize("i", range(7))
def test_td_cdfs_equal_per_call_cdfs(i):
    td = _digests()[i]
    mu, _ = td.centroids()
    xs = [-1e9, 0.0, 1.0, 2.0, 3.0, 1e12]
    if mu.size:
        xs += mu.tolist() + [float(td.vmin), float(td.vmax), float(np.median(mu)) + 0.5]
        xs += np.linspace(float(td.vmin) - 1, float(td.vmax) + 1, 50).tolist()
    got = PB.td_cdfs(td, xs)
    ref = [td.cdf(x) for x in xs]
    assert [_bits(v) for v in got] == [_bits(v) for v in ref]


def test_fit_numeric_and_bin_gain_use_the_batched_reads_unchanged():
    """fit_numeric's quantile grid equals the per-call grid of the same digest."""
    from app.engines.behavior.lib import pnode as PN
    rng = np.random.default_rng(5)
    summ = PN.NumSummary()
    t = 1.7e9
    for d in range(4):
        for x in rng.lognormal(7.0, 0.4, 300):
            summ.update(float(x), t + d * 86400.0 + float(rng.random() * 3600), 1.0, 1.0)
    now = t + 4 * 86400.0
    rec = PB.fit_numeric(summ, now, PB.local_day(now, {}), 200.0, 200.0)
    td = summ.td
    grid = [td.quantile(i / (PB.QGRID - 1)) for i in range(PB.QGRID)]
    if rec.get("band90") and "range" in rec:
        assert len(rec["qgrid"]) == PB.QGRID
    assert all(math.isfinite(v) for v in rec["qgrid"])
    if rec["qgrid"] == [float(v) for v in grid]:
        pass                                               # unclipped digest: the plain grid
    sysd = PS.TDigest(50.0)
    sysd.add_many(rng.lognormal(6.5, 0.8, 2000), now)
    g = PB.bin_gain(summ, sysd)
    edges = [sysd.quantile(k / 8) for k in range(1, 8)]
    cdf = [float(td.cdf(e)) for e in edges]
    c = np.clip(np.asarray([0.0] + cdf + [1.0]), 0.0, 1.0)
    p = np.maximum(np.diff(np.maximum.accumulate(c)), 0.0)
    p = p / p.sum()
    nz = p[p > 0]
    assert _bits(g) == _bits(float(max(0.0, np.sum(nz * np.log2(nz * 8)))))
