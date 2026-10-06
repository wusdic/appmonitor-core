"""Tests for engines/behavior/lib/calib.py: sorted calibration rings, the
PWM-GPD tail, p_from_ring, the small-sample blend, health and link seeding.

Statistical checks use fixed seeds (deterministic); tolerances follow the
docs/lib3/engines.md B24 unit tests (a)-(g). The tail fit goes through
evt.gpd_pwm_fit; while that module is still a stub (it is implemented in a
parallel wave) the `pwm` fixture substitutes a reference oracle written from
the evt docstring, so these tests exercise the real fitter as soon as it
lands and never depend on the order in which the waves finish.
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import time

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from app.engines.behavior.lib import calib as K  # noqa: E402
from app.engines.behavior.lib import combine as C  # noqa: E402
from app.engines.behavior.lib import evt as E  # noqa: E402

NAN = float("nan")
M = K.RING_M


# ------------------------------------------------------------------ oracles
def _ref_pwm_fit(exceedances):
    """evt.gpd_pwm_fit exactly as its docstring specifies (test oracle)."""
    y = np.asarray(exceedances, dtype=np.float64).ravel()
    y = np.sort(y[np.isfinite(y)])
    n = y.size
    a0 = float(y.mean()) if n else 0.0
    if n < 3 or a0 <= 0.0:
        return 0.0, max(a0, 1e-12)
    p = (np.arange(1, n + 1) - 0.35) / n
    a1 = float(np.mean((1.0 - p) * y))
    d = a0 - 2.0 * a1
    if d <= 0.0:
        return 0.0, max(a0, 1e-12)
    xi = 2.0 - a0 / d
    sigma = 2.0 * a0 * a1 / d
    if not -0.5 <= xi <= 0.5:
        xi = min(0.5, max(-0.5, xi))
        sigma = a0 * (1.0 - xi)
    return float(xi), float(sigma)


def _evt_ready() -> bool:
    try:
        E.gpd_pwm_fit(np.linspace(0.1, 3.0, 20))
        E.gpd_sf(np.array([1.0]), 0.1, 1.0)
        return True
    except NotImplementedError:
        return False


EVT_READY = _evt_ready()


@pytest.fixture
def pwm(monkeypatch):
    """Real evt.gpd_pwm_fit when implemented, else the docstring oracle."""
    if not EVT_READY:
        monkeypatch.setattr(K.evt, "gpd_pwm_fit", _ref_pwm_fit)
    return K.evt.gpd_pwm_fit


def ks_ref(ps) -> float:
    return float(stats.kstest(np.asarray(ps, dtype=np.float64), "uniform").statistic)


def r32(x: float) -> float:
    return float(np.float32(x))


def make_ring(scores, ts=None, cap: int = M) -> K.Ring:
    scores = np.asarray(scores, dtype=np.float64)
    ts = np.arange(scores.size, dtype=np.float64) if ts is None else np.asarray(ts, dtype=np.float64)
    return K.Ring(cap=cap, scores=scores, ts=ts)


def pairs(ring: K.Ring):
    return sorted(zip(ring.scores.tolist(), ring.ts.tolist()))


class RefRing:
    """Brute-force model of the ring contract: evict min (ts, score)."""

    def __init__(self, cap):
        self.cap = cap
        self.items = []

    def add(self, s, t):
        s = r32(s) if math.isfinite(s) and abs(s) < 3.4e38 else NAN
        if not (math.isfinite(s) and math.isfinite(t)):
            return
        self.items.append((s, float(t)))
        if len(self.items) > self.cap:
            self.items.remove(min(self.items, key=lambda it: (it[1], it[0])))

    def remove_after(self, t):
        before = len(self.items)
        self.items = [it for it in self.items if it[1] <= t]
        return before - len(self.items)

    def pairs(self):
        return sorted(self.items)


def assert_invariant(ring: K.Ring):
    assert ring.scores.dtype == np.float64 and ring.ts.dtype == np.float64
    assert ring.scores.shape == ring.ts.shape
    assert len(ring) <= ring.cap
    assert np.all(np.diff(ring.scores) >= 0)
    assert np.all(np.isfinite(ring.scores)) and np.all(np.isfinite(ring.ts))
    assert np.array_equal(ring.scores, ring.scores.astype(np.float32).astype(np.float64))


# ===================================================================== Ring
def test_add_keeps_scores_sorted_and_ts_aligned():
    rng = np.random.default_rng(1)
    ring = K.Ring()
    xs = rng.normal(size=200)
    for t, x in enumerate(xs):
        ring.add(x, float(t))
        assert_invariant(ring)
    assert len(ring) == 200
    # each (score, ts) pair survives intact
    assert pairs(ring) == sorted((r32(x), float(t)) for t, x in enumerate(xs))


def test_add_rounds_to_float32_and_p_value_ties_consistently():
    ring = K.Ring()
    for t in range(9):
        ring.add(0.1, float(t))                     # 0.1 is not a float32
    assert ring.scores[0] == r32(0.1) != 0.1
    # the float64 score 0.1 ties with all 9 stored entries after rounding
    assert ring.p_value(0.1, 0.5, rand_atom=True) == pytest.approx((0 + 0.5 * 10) / 10)
    assert K.p_from_ring(ring, 0.1, 0.5, rand_atom=True) == ring.p_value(0.1, 0.5, rand_atom=True)
    # ... which is the ring's lowest level: the issued p is the block's upper p
    assert ring.p_value(0.1, 0.5) == K.p_from_ring(ring, 0.1, 0.5) == 1.0


def test_add_ignores_nan_inf_out_of_range_and_nan_ts():
    ring = K.Ring()
    ring.add(NAN, 1.0)
    ring.add(math.inf, 1.0)
    ring.add(-math.inf, 1.0)
    ring.add(1e39, 1.0)                              # beyond float32 -> inf
    ring.add(1.0, NAN)
    ring.add(1.0, math.inf)
    ring.add(None, 1.0)
    assert len(ring) == 0
    ring.add(np.float32(2.5), np.float64(3.0))       # numpy scalars are fine
    assert ring.scores.tolist() == [2.5] and ring.ts.tolist() == [3.0]


def test_capacity_evicts_oldest_ts_not_extreme_score():
    ring = K.Ring()
    for t in range(M):
        ring.add(1000.0 if t == 0 else float(t), float(t))   # oldest is the max
    assert len(ring) == M and ring.scores[-1] == 1000.0
    ring.add(-5.0, float(M))
    assert len(ring) == M
    assert 1000.0 not in ring.scores and ring.scores[0] == -5.0
    assert ring.ts.min() == 1.0


def test_capacity_long_stream_keeps_last_m_ticks():
    rng = np.random.default_rng(2)
    ring = K.Ring()
    xs = rng.exponential(size=3 * M + 17)
    for t, x in enumerate(xs):
        ring.add(x, 100.0 + t)
    assert len(ring) == M
    kept = sorted(ring.ts.tolist())
    assert kept == [100.0 + t for t in range(len(xs) - M, len(xs))]
    assert pairs(ring) == sorted((r32(x), 100.0 + t) for t, x in enumerate(xs) if t >= len(xs) - M)


def test_add_entry_older_than_full_ring_is_evicted_at_once():
    ring = make_ring(np.arange(8.0), ts=np.arange(10.0, 18.0), cap=8)
    s0, t0 = ring.scores, ring.ts
    ring.add(3.5, 5.0)                               # older than everything
    assert ring.scores is s0 and ring.ts is t0       # untouched
    ring.add(3.5, 10.0)                              # ties oldest ts, higher score: kept
    assert 3.5 in ring.scores and 0.0 not in ring.scores and len(ring) == 8


@pytest.mark.parametrize("cap", [1, 2, 5, 16])
def test_add_and_remove_after_match_brute_force_model(cap):
    rng = np.random.default_rng(100 + cap)
    ring, ref = K.Ring(cap=cap), RefRing(cap)
    for step in range(600):
        op = rng.random()
        if op < 0.9:
            # coarse grids force ties in score and in ts, and out-of-order ts
            s = float(rng.integers(0, 6)) + (0.1 if rng.random() < 0.3 else 0.0)
            t = float(rng.integers(0, 40))
            if rng.random() < 0.02:
                s = NAN
            ring.add(s, t)
            ref.add(s, t)
        else:
            t = float(rng.integers(0, 40))
            assert ring.remove_after(t) == ref.remove_after(t)
        assert_invariant(ring)
        assert pairs(ring) == ref.pairs(), f"step {step}"


def test_arrays_are_copy_on_write_snapshots():
    ring = make_ring([1.0, 2.0, 3.0])
    snap_s, snap_t = ring.scores, ring.ts
    shallow = copy.copy(ring)
    ring.add(2.5, 10.0)
    ring.remove_after(5.0)
    assert snap_s.tolist() == [1.0, 2.0, 3.0] and snap_t.tolist() == [0.0, 1.0, 2.0]
    assert shallow.scores.tolist() == [1.0, 2.0, 3.0]
    full = make_ring(np.arange(4.0), cap=4)
    snap = full.scores
    full.add(9.0, 99.0)                              # full-ring evict path
    assert snap.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_cap_lowered_after_construction_evicts_down():
    ring = make_ring(np.arange(8.0), ts=np.arange(8.0))
    ring.cap = 3
    ring.add(0.5, 100.0)
    assert len(ring) == 3
    assert sorted(ring.ts.tolist()) == [6.0, 7.0, 100.0]
    assert_invariant(ring)


def test_constructor_normalises_and_validates():
    r = K.Ring(scores=np.array([3.0, NAN, 1.0, 2.0, 0.1]), ts=np.array([0.0, 1.0, 2.0, NAN, 4.0]))
    assert r.scores.tolist() == [r32(0.1), 1.0, 3.0]
    assert r.ts.tolist() == [4.0, 2.0, 0.0]
    r = K.Ring(cap=2, scores=np.array([5.0, 1.0, 3.0]), ts=np.array([1.0, 3.0, 2.0]))
    assert r.scores.tolist() == [1.0, 3.0] and r.ts.tolist() == [3.0, 2.0]
    with pytest.raises(ValueError):
        K.Ring(scores=np.array([1.0, 2.0]), ts=np.array([1.0]))
    with pytest.raises(ValueError):
        K.Ring(cap=0)
    assert len(K.Ring()) == 0 and K.Ring().cap == M


def test_remove_after_deletes_later_entries_and_drops_gpd():
    # B24 unit test (f): entries after rollback_to are deleted.
    ring = make_ring(np.arange(10.0), ts=np.arange(100.0, 110.0))
    ring.gpd = K.GPDTail(u=1.0, xi=0.1, sigma=1.0, rate=0.1, n=10)
    assert ring.remove_after(200.0) == 0 and ring.gpd is not None
    assert ring.remove_after(NAN) == 0 and len(ring) == 10
    assert ring.remove_after(104.0) == 5
    assert ring.ts.max() == 104.0 and len(ring) == 5 and ring.gpd is None
    assert ring.remove_after(0.0) == 5 and len(ring) == 0
    assert K.Ring().remove_after(1.0) == 0


def test_quantile_matches_numpy_linear():
    rng = np.random.default_rng(3)
    xs = rng.gamma(2.0, size=137)
    ring = make_ring(xs)
    for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0):
        assert ring.quantile(q) == pytest.approx(np.quantile(ring.scores, q), rel=1e-15)
    assert math.isnan(K.Ring().quantile(0.5))
    assert math.isnan(ring.quantile(NAN))
    with pytest.raises(ValueError):
        ring.quantile(1.5)


def test_reset_empties_ring_and_tail():
    ring = make_ring([1.0, 2.0])
    ring.gpd = K.GPDTail(u=1.0, xi=0.0, sigma=1.0, rate=0.5, n=2)
    ring.reset()
    assert len(ring) == 0 and ring.ts.size == 0 and ring.gpd is None
    ring.add(1.0, 1.0)
    assert len(ring) == 1


def test_to_dict_from_dict_round_trip_is_strict_json():
    rng = np.random.default_rng(4)
    ring = make_ring(rng.normal(size=300), ts=1.7e9 + 900.0 * np.arange(300))
    ring.gpd = K.GPDTail(u=1.2, xi=0.15, sigma=0.7, rate=0.1015625, n=256)   # fitted_ts NaN
    d = ring.to_dict()
    assert set(d) == {"scores", "ts", "gpd"}
    assert all(type(x) is float for x in d["scores"])
    back = K.Ring.from_dict(json.loads(json.dumps(d, allow_nan=False)))
    assert np.array_equal(back.scores, ring.scores) and np.array_equal(back.ts, ring.ts)
    assert back.cap == M
    g = back.gpd
    assert (g.u, g.xi, g.sigma, g.rate, g.n) == (1.2, 0.15, 0.7, 0.1015625, 256)
    assert math.isnan(g.fitted_ts)
    ring.gpd = None
    small = K.Ring(cap=8, scores=np.arange(5.0), ts=np.arange(5.0))
    small.gpd = K.GPDTail(1.0, 0.0, 1.0, 0.2, 5, fitted_ts=123.0)
    b2 = K.Ring.from_dict(json.loads(json.dumps(small.to_dict())))
    assert b2.cap == 8 and b2.gpd.fitted_ts == 123.0
    assert K.Ring.from_dict(ring.to_dict()).gpd is None


def test_from_dict_repairs_foreign_layout():
    d = {"scores": [3.0, 1.0, float("nan"), 2.0], "ts": [1.0, 2.0, 3.0, 0.5], "cap": 2}
    r = K.Ring.from_dict(d)
    assert r.scores.tolist() == [1.0, 3.0] and r.ts.tolist() == [2.0, 1.0]
    assert len(K.Ring.from_dict({})) == 0


# ================================================================ p-values
def test_p_value_is_randomized_conformal_on_the_ring():
    rng = np.random.default_rng(5)
    ring = make_ring(rng.normal(size=100))
    for s in (-5.0, -0.3, 0.0, 0.7, 5.0, float(ring.scores[40]), float(ring.scores[0])):
        for u in (0.01, 0.5, 0.99):
            want = C.randomized_conformal_p(ring.scores, r32(s), u)
            assert ring.p_value(s, u, rand_atom=True) == want
            assert 0.0 < want < 1.0
            # at or below the lowest level: 1 (lib/calib "Lower atom")
            assert ring.p_value(s, u) == (1.0 if r32(s) <= ring.scores[0] else want)
    assert K.Ring().p_value(1.0, 0.3, rand_atom=True) == 0.3
    assert K.Ring().p_value(1.0, 0.3) == 1.0
    assert math.isnan(ring.p_value(NAN, 0.5))
    assert math.isnan(ring.p_value(1.0, NAN))


def test_nan_in_gives_nan_out():
    # B24 unit test (g): degraded input -> NaN p, never 1.
    ring = make_ring(np.arange(50.0))
    ring.gpd = K.GPDTail(u=45.0, xi=0.1, sigma=2.0, rate=0.1, n=50)
    assert math.isnan(K.p_from_ring(ring, NAN, 0.5))
    assert math.isnan(K.p_from_ring(K.Ring(), NAN, 0.5))
    assert math.isnan(K.p_from_ring(ring, None, 0.5))
    assert math.isnan(K.p_from_ring(ring, 10.0, NAN))       # conformal branch


def _stream_ps(xs, warm: int = M, randomized: bool = True, tail: bool = True,
               rand_atom: bool = True):
    """B24 pipeline on one stream: score each tick against the current ring,
    then admit it; refit the tail every GPD_REFIT_TICKS. rand_atom: the fully
    randomised p (exactly uniform; the calibration monitors' p), else the
    issued p (1 at the ring's lowest level)."""
    ring = make_ring(xs[:warm])
    if tail:
        ring.gpd = K.fit_tail(ring)
    ps = []
    for k, x in enumerate(xs[warm:]):
        u = C.seeded_uniform("sys", "ent", "det", float(k)) if randomized else 1.0
        ps.append(K.p_from_ring(ring, x, u, rand_atom=rand_atom))
        ring.add(x, float(warm + k))
        if tail and k % K.GPD_REFIT_TICKS == K.GPD_REFIT_TICKS - 1:
            ring.gpd = K.fit_tail(ring, now_ts=float(warm + k))
    return np.asarray(ps)


def test_null_exp_scores_are_uniform(pwm):
    # B24 unit test (a): 2000 null Exp(1) scores -> KS D < 0.03.
    ds = []
    for seed in range(4):
        xs = np.random.default_rng(10 + seed).exponential(size=M + 2000)
        ps = _stream_ps(xs)
        assert np.all((ps > 0) & (ps < 1))
        ds.append(K.ks_uniform(ps))
    assert max(ds) < 0.03, ds


def test_sparse_detector_randomized_vs_deterministic(pwm):
    # B24 unit test (b): 90% zeros. Randomised p is uniform; the classical
    # deterministic conformal p (u = 1) piles up at 1.
    rng = np.random.default_rng(20)
    xs = np.where(rng.random(M + 2000) < 0.9, 0.0, rng.exponential(size=M + 2000))
    d_rand = K.ks_uniform(_stream_ps(xs))
    d_det = K.ks_uniform(_stream_ps(xs, randomized=False))
    assert d_rand < 0.03
    assert d_det > 0.5
    # the issued p: the zeros (the lowest level) at exactly 1, the rest the
    # randomised p - valid (P(p <= a) <= a), no coin flips at the atom
    ps = _stream_ps(xs, rand_atom=False)
    zero = xs[M:] == 0.0
    assert np.all(ps[zero] == 1.0)
    for a in (0.01, 0.05, 0.1, 0.5):
        assert np.mean(ps <= a) <= a + 3.0 * math.sqrt(a * (1 - a) / ps.size)


def test_p_from_ring_without_tail_is_conformal():
    ring = make_ring(np.random.default_rng(6).normal(size=256))
    for s in (-1.0, 0.0, 2.0, 50.0):
        assert K.p_from_ring(ring, s, 0.4) == ring.p_value(s, 0.4)
    # far beyond the ring without a tail: bounded below by u / (M + 1)
    assert K.p_from_ring(ring, 1e6, 0.4) == pytest.approx(0.4 / 257)


def test_p_from_ring_tail_branch_formula_and_threshold():
    ring = make_ring(np.linspace(0.0, 10.0, 256))
    tail = K.GPDTail(u=9.0, xi=0.2, sigma=0.5, rate=0.1, n=256)
    ring.gpd = tail
    # at or below u: conformal, even though a tail exists
    assert K.p_from_ring(ring, 9.0, 0.5) == ring.p_value(9.0, 0.5)
    assert K.p_from_ring(ring, 3.0, 0.5) == ring.p_value(3.0, 0.5)
    # above u, inside the ring range: pure POT formula, u-independent
    s = 9.5
    want = 0.1 * stats.genpareto.sf(r32(s) - 9.0, 0.2, scale=0.5)
    assert K.p_from_ring(ring, s, 0.5) == pytest.approx(want, rel=1e-12)
    assert K.p_from_ring(ring, s, 0.01) == K.p_from_ring(ring, s, 0.99)
    assert K.p_from_ring(ring, s, NAN) == pytest.approx(want, rel=1e-12)
    # far beyond: tiny and below 1/(M+1)
    far = K.p_from_ring(ring, 40.0, 0.5)
    assert far == pytest.approx(0.1 * stats.genpareto.sf(31.0, 0.2, scale=0.5), rel=1e-12)
    assert far < 1.0 / 257


def test_p_from_ring_explicit_tail_overrides_ring_gpd_and_invalid_is_ignored():
    ring = make_ring(np.linspace(0.0, 10.0, 256))
    ring.gpd = K.GPDTail(u=9.0, xi=0.0, sigma=1.0, rate=0.1, n=256)
    other = K.GPDTail(u=9.0, xi=0.0, sigma=0.5, rate=0.1, n=256)
    assert K.p_from_ring(ring, 9.8, 0.5) == pytest.approx(0.1 * math.exp(-0.8), rel=1e-6)
    assert K.p_from_ring(ring, 9.8, 0.5, other) == pytest.approx(0.1 * math.exp(-1.6), rel=1e-6)
    for bad in (K.GPDTail(9.0, 0.1, 0.0, 0.1, 256), K.GPDTail(9.0, NAN, 1.0, 0.1, 256),
                K.GPDTail(NAN, 0.1, 1.0, 0.1, 256), K.GPDTail(9.0, 0.1, 1.0, 0.0, 256)):
        assert not bad.valid()
        assert K.p_from_ring(ring, 9.8, 0.5, bad) == ring.p_value(9.8, 0.5)


def test_p_from_ring_floors_at_1e_300_and_handles_infinite_scores():
    ring = make_ring(np.linspace(0.0, 10.0, 256))
    bounded = K.GPDTail(u=9.0, xi=-0.5, sigma=1.0, rate=0.1, n=256)   # end point 11
    assert K.p_from_ring(ring, 11.5, 0.5, bounded) == 1e-300
    assert K.p_from_ring(ring, 1e30, 0.5, K.GPDTail(9.0, 0.3, 1.0, 0.1, 256)) >= 1e-300
    assert K.p_from_ring(ring, math.inf, 0.5, K.GPDTail(9.0, 0.3, 1.0, 0.1, 256)) == 1e-300
    assert K.p_from_ring(ring, -math.inf, 0.5, rand_atom=True) == pytest.approx(256.5 / 257)
    assert K.p_from_ring(ring, -math.inf, 0.5) == 1.0


def test_p_from_ring_caps_tail_at_conformal_bound_beyond_ring_max():
    # A bounded null (JSD-like): a score beyond every ring entry must get
    # p <= 1/(n+1) even when the fitted tail is heavier than that.
    ring = make_ring(np.linspace(0.0, 1.0, 256))
    heavy = K.GPDTail(u=0.9, xi=0.5, sigma=1.0, rate=0.1, n=256)
    assert K.p_from_ring(ring, 1.05, 0.5, heavy) == pytest.approx(1.0 / 257)
    # inside the ring range the formula is untouched
    inside = K.p_from_ring(ring, 0.95, 0.5, heavy)
    assert inside == pytest.approx(0.1 * stats.genpareto.sf(r32(0.95) - 0.9, 0.5, scale=1.0), rel=1e-12)
    # an empty ring with a (pooled) tail is not capped
    assert K.p_from_ring(K.Ring(), 1.05, 0.5, heavy) == pytest.approx(
        0.1 * stats.genpareto.sf(r32(1.05) - 0.9, 0.5, scale=1.0), rel=1e-12)


@pytest.mark.parametrize("xi", [-0.45, -0.2, -1e-10, 0.0, 1e-10, 0.1, 0.5])
def test_scalar_gpd_sf_matches_scipy(xi):
    ys = np.array([-1.0, 0.0, 1e-9, 0.3, 1.0, 1.9, 2.5, 10.0, 1e3])
    # |xi| < 1e-9 uses the exponential limit (as evt.gpd_sf): relative error
    # ~ xi z^2 / 2, i.e. < 1e-4 even 1000 scales out.
    rel = 1e-9 if abs(xi) >= 1e-9 else 1e-4
    for y in ys:
        got = K._gpd_sf(float(y), xi, 0.9)
        want = 1.0 if y <= 0 else float(stats.genpareto.sf(y, xi, scale=0.9))
        assert got == pytest.approx(want, rel=rel, abs=1e-300), (y, xi)
    assert math.isnan(K._gpd_sf(NAN, 0.1, 1.0))


@pytest.mark.skipif(not EVT_READY, reason="evt.gpd_sf not implemented yet")
def test_scalar_gpd_sf_matches_evt():
    ys = np.array([0.0, 0.1, 1.0, 3.0, 30.0, 300.0])
    for xi in (-0.5, -0.1, 0.0, 0.2, 0.5):
        want = np.asarray(E.gpd_sf(ys, xi, 1.3), dtype=np.float64)
        got = np.array([K._gpd_sf(float(y), xi, 1.3) for y in ys])
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-300)


# ================================================================ fit_tail
def test_fit_tail_threshold_rate_and_exceedances(pwm, monkeypatch):
    rng = np.random.default_rng(7)
    ring = make_ring(rng.exponential(size=256))
    seen = {}

    def spy(y):
        seen["y"] = np.array(y, dtype=np.float64)
        return pwm(y)

    monkeypatch.setattr(K.evt, "gpd_pwm_fit", spy)
    tail = K.fit_tail(ring, now_ts=1234.0)
    u = float(np.quantile(ring.scores, 0.9))
    assert tail.u == u
    assert tail.n == 256 and tail.fitted_ts == 1234.0
    n_u = int(np.sum(ring.scores > u))
    assert tail.rate == n_u / 256 and n_u == 26
    np.testing.assert_array_equal(np.sort(seen["y"]), np.sort(ring.scores[ring.scores > u] - u))
    assert np.all(seen["y"] > 0)
    assert ring.gpd is None                          # pure: caller assigns
    assert 0.0 <= tail.xi <= 0.5 and tail.sigma > 0


def test_fit_tail_needs_min_exceedances(pwm):
    # q_0.90 with linear interpolation: n = 100 -> 10 exceedances, n = 91 -> 9.
    assert K.fit_tail(make_ring(np.arange(100.0))) is not None
    assert K.fit_tail(make_ring(np.arange(99.0))) is not None
    assert K.fit_tail(make_ring(np.arange(91.0))) is None
    assert K.fit_tail(make_ring(np.arange(5.0))) is None
    assert K.fit_tail(K.Ring()) is None
    # ties at the top: nothing strictly above u
    assert K.fit_tail(make_ring(np.r_[np.arange(200.0), np.full(56, 500.0)])) is None
    assert K.fit_tail(make_ring(np.ones(256))) is None


def test_fit_tail_floors_negative_xi_at_exponential_keeping_mean_excess(pwm):
    # Uniform scores are bounded: the raw PWM fit has xi < 0.
    ring = make_ring(np.linspace(0.0, 1.0, 256))
    raw = K.fit_tail(ring, xi_min=-0.5)
    assert raw.xi < 0
    floored = K.fit_tail(ring)
    y = ring.scores[ring.scores > floored.u] - floored.u
    assert floored.xi == 0.0
    assert floored.sigma == pytest.approx(float(np.mean(y)), rel=1e-12)
    assert raw.sigma / (1 - raw.xi) == pytest.approx(float(np.mean(y)), rel=1e-9)  # PWM keeps it
    mid = K.fit_tail(ring, xi_min=-0.1)
    assert mid.xi == -0.1 and mid.sigma == pytest.approx(float(np.mean(y)) * 1.1, rel=1e-12)
    # a heavy tail is not touched by the floor
    q = stats.genpareto.ppf((np.arange(256) + 0.5) / 256, 0.3)
    heavy = make_ring(q)
    assert K.fit_tail(heavy).xi == K.fit_tail(heavy, xi_min=-0.5).xi > 0.1


def test_no_1e300_on_null_exp_scores_beyond_bounded_fit(pwm):
    # Regression for the plug-in xi < 0 failure: an Exp(1) null ring whose raw
    # fit is bounded would give p = 1e-300 to a null score past the end point.
    for seed in range(200):
        xs = np.random.default_rng(1000 + seed).exponential(size=256)
        ring = make_ring(xs)
        raw = K.fit_tail(ring, xi_min=-0.5)
        if raw is not None and raw.xi < -0.15:
            break
    else:
        pytest.skip("no bounded raw fit found")
    s = raw.u - raw.sigma / raw.xi + 0.5             # beyond the raw end point
    assert K.p_from_ring(ring, s, 0.5, raw) == 1e-300
    p = K.p_from_ring(ring, s, 0.5, K.fit_tail(ring))
    true = math.exp(-s)
    assert p > 0.1 * true and p <= 1.0 / 257


def _gpd_setup(xi=0.2, sigma=1.0, mult=10.0):
    u_star = float(stats.genpareto.ppf(0.9, xi, scale=sigma))
    sigma_u = sigma + xi * u_star
    s = u_star + mult * sigma_u
    return s, float(stats.genpareto.sf(s, xi, scale=sigma))


def test_gpd_tail_far_score_within_2x_on_ideal_ring(pwm):
    # B24 unit test (c) on the quantile-exact ring (no sampling noise):
    # a score 10 sigma_u into a xi = 0.2 tail.
    s, true = _gpd_setup()
    ring = make_ring(stats.genpareto.ppf((np.arange(M) + 0.5) / M, 0.2))
    ring.gpd = K.fit_tail(ring)
    assert ring.gpd.xi == pytest.approx(0.2, abs=0.05)
    p = K.p_from_ring(ring, s, 0.5)
    assert p < 1.0 / (M + 1)
    assert 0.5 < p / true < 2.0, (p, true)


def test_gpd_tail_far_score_monte_carlo(pwm):
    # B24 unit test (c) on sampled rings: the median ratio is within 2x of the
    # true tail and far scores essentially always get p < 1/(M+1). (Per ring,
    # 26 exceedances give xi sd ~0.2, so single rings scatter widely.)
    s, true = _gpd_setup()
    ratios = []
    for seed in range(200):
        xs = stats.genpareto.rvs(0.2, size=M, random_state=np.random.default_rng(seed))
        ring = make_ring(xs)
        ring.gpd = K.fit_tail(ring)
        ratios.append(K.p_from_ring(ring, s, 0.5) / true)
    ratios = np.asarray(ratios)
    assert 0.5 < np.median(ratios) < 2.0
    assert np.mean(ratios * true < 1.0 / (M + 1)) >= 0.97
    assert ratios.min() > 1e-6                       # no 1e-300 collapse


def test_tail_reaches_below_conformal_resolution(pwm):
    xs = np.random.default_rng(8).exponential(size=M)
    ring = make_ring(xs)
    ring.gpd = K.fit_tail(ring)
    p = K.p_from_ring(ring, 30.0, 0.5)
    assert 1e-300 <= p < 1e-6
    # monotone in s beyond u
    grid = np.linspace(ring.gpd.u + 1e-3, 40.0, 50)
    pg = [K.p_from_ring(ring, x, 0.5) for x in grid]
    assert all(a >= b for a, b in zip(pg, pg[1:]))


def test_null_tail_rate_is_not_grossly_anti_conservative(pwm):
    # Realised rate at the e_day 0.03 level for 900 s ticks (p <= 3.1e-4) on
    # an Exp(1) null, streaming ring with refits. The spec-exact plug-in
    # (xi_min = -0.5) measured ~4.7x here; the floor brings it near 1.5x.
    xs = np.random.default_rng(30).exponential(size=M + 40000)
    ps = _stream_ps(xs)
    alpha = 0.03 * 900 / 86400
    rate = float(np.mean(ps <= alpha)) / alpha
    assert rate < 3.0, rate
    assert ps.min() > 1e-9


# ============================================================ small sample
def test_blend_small_sample_weight_and_formula():
    def sig(x):
        return 1 / (1 + math.exp(-x))

    def lg(p):
        return math.log(p / (1 - p))

    for n in (0, 1, 10, 63, 64, 200):
        w = n / (n + 64)
        want = sig(w * lg(0.01) + (1 - w) * lg(0.2))
        assert K.blend_small_sample(0.01, 0.2, n) == pytest.approx(want, rel=1e-12)
    assert K.blend_small_sample(0.01, 0.2, 0) == pytest.approx(0.2, rel=1e-12)
    assert K.blend_small_sample(0.01, 0.2, 64) == pytest.approx(sig(0.5 * lg(0.01) + 0.5 * lg(0.2)))
    assert K.blend_small_sample(0.01, 0.2, 10, n0=10) == K.blend_small_sample(0.01, 0.2, 64)
    assert K.blend_small_sample(0.01, 0.2, 5, n0=0) == pytest.approx(0.01, rel=1e-12)
    assert K.blend_small_sample(0.01, 0.2, -3) == pytest.approx(0.2, rel=1e-12)
    assert K.blend_small_sample(0.01, 0.2, NAN) == pytest.approx(0.2, rel=1e-12)
    assert K.blend_small_sample(0.01, 0.2, math.inf) == pytest.approx(0.01, rel=1e-12)


def test_blend_small_sample_nan_and_extremes():
    assert K.blend_small_sample(0.03, NAN, 5) == 0.03
    assert K.blend_small_sample(NAN, 0.4, 5) == 0.4
    assert math.isnan(K.blend_small_sample(NAN, NAN, 5))
    p = K.blend_small_sample(1e-300, 1e-300, 7)
    assert p == pytest.approx(1e-300, rel=1e-9)
    assert 0 < K.blend_small_sample(0.0, 1.0, 32) < 1
    assert K.blend_small_sample(1.0, 1.0, 32) <= 1.0


def test_new_stratum_first_64_ticks_blend_with_pm():
    # B24 unit test (e): after a 900 s -> 60 s switch the new ring is thin;
    # its p sits between the conformal p and pm, pulled toward pm by exactly
    # 1 - w = 64/(n+64) of the logit distance, which shrinks as n grows.
    rng = np.random.default_rng(9)
    ring = K.Ring()
    pm = 0.5

    def lg(p):
        return math.log(p) - math.log1p(-p)

    prev = math.inf
    for n in range(1, 65):
        ring.add(rng.exponential(), float(n))
        p_conf = ring.p_value(20.0, 0.5)                 # far score: tiny conformal p
        p = K.blend_small_sample(p_conf, pm, len(ring))
        assert p_conf < p < pm
        pull = (lg(p) - lg(p_conf)) / (lg(pm) - lg(p_conf))
        assert pull == pytest.approx(64 / (n + 64), rel=1e-9)
        assert pull < prev
        prev = pull


# ================================================================== health
def test_ks_uniform_matches_scipy_and_drops_nonfinite():
    rng = np.random.default_rng(11)
    for n in (1, 5, 100, 1000):
        ps = rng.random(n)
        assert K.ks_uniform(ps) == pytest.approx(ks_ref(ps), abs=1e-12)
    ps = rng.random(50)
    assert K.ks_uniform(np.r_[ps, NAN, math.inf]) == pytest.approx(ks_ref(ps), abs=1e-12)
    assert math.isnan(K.ks_uniform(np.array([])))
    assert math.isnan(K.ks_uniform([NAN]))
    assert K.ks_uniform([1.0] * 10) == pytest.approx(1.0)
    lst = list(rng.random(20))                       # plain lists are accepted
    assert K.ks_uniform(lst) == pytest.approx(ks_ref(lst), abs=1e-12)


@pytest.mark.parametrize("ks,rr,want", [
    (0.01, 1.0, 1.0), (0.05, 1.0, 1.0), (0.051, 1.0, 0.5),
    (0.01, 0.5, 1.0), (0.01, 2.0, 1.0), (0.01, 0.49, 0.5), (0.01, 2.01, 0.5),
    (0.01, 0.0, 0.5), (0.01, math.inf, 0.5), (0.2, 5.0, 0.5),
    (NAN, 1.0, 1.0), (0.01, NAN, 1.0), (NAN, NAN, 1.0), (None, None, 1.0),
    (NAN, 5.0, 0.5), (0.3, NAN, 0.5),
])
def test_health_weight(ks, rr, want):
    assert K.health_weight(ks, rr) == want


# ================================================================ seeding
def test_seed_ring_adds_most_recent_half_of_other():
    own = make_ring(np.arange(10.0), ts=np.arange(1000.0, 1010.0))
    other = make_ring(100.0 + np.arange(300.0), ts=np.arange(300.0))
    before_own, before_other = pairs(own), pairs(other)
    seeded = K.seed_ring(own, other)
    assert len(seeded) == 10 + 128 and seeded.cap == M
    assert sorted(seeded.ts.tolist())[:128] == list(np.arange(172.0, 300.0))
    assert set(own.ts.tolist()) <= set(seeded.ts.tolist())
    assert seeded.gpd is None
    assert_invariant(seeded)
    assert pairs(own) == before_own and pairs(other) == before_other   # inputs untouched


def test_seed_ring_applies_oldest_first_eviction():
    own = make_ring(np.arange(200.0), ts=np.arange(200.0))
    other = make_ring(-np.arange(200.0) - 1, ts=1000.0 + np.arange(200.0))
    seeded = K.seed_ring(own, other)
    assert len(seeded) == M
    # 128 seeded (newest) entries + the 128 newest of own
    assert sorted(seeded.ts.tolist()) == list(np.arange(72.0, 200.0)) + list(1072.0 + np.arange(128.0))
    # equivalent to adding the same entries one by one
    step = copy.copy(own)
    pick = np.argsort(other.ts)[-128:]
    for s, t in zip(other.scores[pick], other.ts[pick]):
        step.add(s, t)
    assert pairs(step) == pairs(seeded)


def test_seed_ring_frac_and_edge_cases():
    own = make_ring(np.arange(4.0), ts=np.arange(4.0), cap=8)
    own.gpd = K.GPDTail(1.0, 0.0, 1.0, 0.25, 4)
    other = make_ring(10.0 + np.arange(20.0), ts=100.0 + np.arange(20.0))
    assert len(K.seed_ring(own, other, frac=0.5)) == 8        # 4 + floor(8 * .5) = 8
    s0 = K.seed_ring(own, other, frac=0.0)
    assert pairs(s0) == pairs(own) and s0.gpd is own.gpd and s0 is not own
    assert pairs(K.seed_ring(own, K.Ring(), 0.5)) == pairs(own)
    full = K.seed_ring(own, other, frac=5.0)                  # clipped to 1
    assert sorted(full.ts.tolist()) == list(112.0 + np.arange(8.0))
    assert len(K.seed_ring(own, other, frac=0.3)) == 6        # floor(2.4) = 2
    with pytest.raises(ValueError):
        K.seed_ring(own, other, frac=NAN)
    few = make_ring([50.0], ts=[7.0])
    assert len(K.seed_ring(own, few)) == 5


# ==================================================================== keys
def test_stratum_keys():
    assert K.stratum_key("wd_day", 900) == "wd_day|900"
    assert K.stratum_key("nwd_night", 60.0) == "nwd_night|60"
    assert K.stratum_key("wd_day", np.int64(3600)) == "wd_day|3600"
    # B24 unit test (d): night and day never share a ring
    assert K.stratum_key("wd_night", 900) != K.stratum_key("wd_day", 900)
    assert K.stratum_key("wd_day", 60) != K.stratum_key("wd_day", 900)
    for bad in ((("wd|day", 900)), ("wd@day", 900), ("", 900), ("wd_day", 90.5), ("wd_day", 0)):
        with pytest.raises(ValueError):
            K.stratum_key(*bad)
    assert K.identity_stratum_key("wd_day", 2) == "wd_day|r2"
    assert K.identity_stratum_key("nwd_day", 0) == "nwd_day|r0"
    for k in (-1, 3, 1.5):
        with pytest.raises(ValueError):
            K.identity_stratum_key("wd_day", k)


def test_ring_key_round_trip():
    for det, st in (("volume_nb", "wd_day|900"), ("identity", "wd_night|r1"),
                    ("meta_inst", "nwd_day|60"), ("meta_all", "nwd_night|3600")):
        key = K.ring_key(det, st)
        assert key == f"{det}@{st}"
        assert K.split_ring_key(key) == (det, st)
    for bad in ("nosep", "@wd_day|900", "det@", "a@b@c"):
        with pytest.raises(ValueError):
            K.split_ring_key(bad)
    with pytest.raises(ValueError):
        K.ring_key("de@t", "wd_day|900")
    with pytest.raises(ValueError):
        K.ring_key("det", "")


# ==================================================================== perf
def test_hot_path_is_cheap():
    # B24 perf: 40 keys x 31 detectors per tick. Generous bounds (shared CI).
    rng = np.random.default_rng(12)
    ring = make_ring(rng.exponential(size=M))
    ring.gpd = K.GPDTail(u=float(ring.quantile(0.9)), xi=0.05, sigma=1.0, rate=0.1, n=M)
    xs = rng.exponential(size=1240).tolist()
    t0 = time.perf_counter()
    for i, x in enumerate(xs):
        K.p_from_ring(ring, x, 0.5)
    t_p = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i, x in enumerate(xs):
        ring.add(x, 1e4 + i)
    t_add = time.perf_counter() - t0
    assert t_p < 0.05, t_p          # ~5 ms measured
    assert t_add < 0.08, t_add      # ~8 ms measured


# ======================================================= review regressions
def test_gpdtail_sf_nan_or_invalid_gives_nan_not_floor():
    # Regression: max(P_FLOOR, nan) is 1e-300, so a NaN score through the
    # public GPDTail.sf came out as the most extreme possible p; sigma = 0
    # raised ZeroDivisionError and sigma < 0 / rate > 1 returned p > 1.
    t = K.GPDTail(u=1.0, xi=0.2, sigma=1.0, rate=0.1, n=256)
    assert math.isnan(t.sf(NAN))
    assert math.isnan(t.sf(None))
    assert t.sf(0.5) == 0.1 and t.sf(1e300) == 1e-300
    for bad in (K.GPDTail(1.0, 0.2, 0.0, 0.1, 256), K.GPDTail(1.0, 0.2, -1.0, 0.9, 256),
                K.GPDTail(1.0, 0.2, 1.0, 5.0, 256), K.GPDTail(1.0, NAN, 1.0, 0.1, 256),
                K.GPDTail(NAN, 0.2, 1.0, 0.1, 256)):
        assert math.isnan(bad.sf(2.0)), bad
    # the scalar sf mirrors evt.gpd_sf on invalid parameters
    for xi, sigma in ((0.2, 0.0), (0.2, -1.0), (NAN, 1.0), (0.1, math.inf)):
        assert math.isnan(K._gpd_sf(1.0, xi, sigma))
        assert math.isnan(float(E.gpd_sf(np.array([1.0]), xi, sigma)[0]))
        assert K._gpd_sf(0.0, xi, sigma) == 1.0


def test_stratum_keys_reject_non_finite_and_non_numeric_with_value_error():
    # Regression: inf raised OverflowError and None TypeError; the contract is
    # ValueError for any malformed part.
    for cc in (math.inf, -math.inf, NAN, None, "900", True):
        with pytest.raises(ValueError):
            K.stratum_key("wd_day", cc)
    for k in (math.inf, NAN, None, "1", True):
        with pytest.raises(ValueError):
            K.identity_stratum_key("wd_day", k)
    assert K.stratum_key("wd_day", np.float32(900.0)) == "wd_day|900"
    assert K.identity_stratum_key("wd_day", np.int8(1)) == "wd_day|r1"


def test_conformal_fast_path_is_bit_identical_to_combine():
    # p_from_ring / Ring.p_value inline the conformal count for speed; it must
    # equal combine.randomized_conformal_p exactly, ties included.
    rng = np.random.default_rng(99)
    ring = make_ring(np.round(rng.exponential(size=M), 1))       # many ties
    probes = np.r_[ring.scores[::7], rng.exponential(size=50), -1.0, 0.0, 1e6, np.inf, -np.inf]
    for s in probes.tolist():
        for u in (0.0, 1e-9, 0.37, 1.0):
            x = r32(s) if abs(s) < 3e38 else s
            want = C.randomized_conformal_p(ring.scores, x, u)
            assert K.p_from_ring(ring, s, u, rand_atom=True) == want
            assert ring.p_value(s, u, rand_atom=True) == want
            # issued: the same except at or below the lowest level
            assert ring.p_value(s, u) == (1.0 if x <= ring.scores[0] else want)
    assert math.isnan(ring.p_value(1.0, NAN)) and math.isnan(ring.p_value(NAN, 0.5))
    with pytest.raises(ValueError):
        ring.p_value(1.0, 1.5)
    assert K.Ring().p_value(1.0, 0.3, rand_atom=True) == 0.3
    # a hand-assigned array that breaks the invariant still goes through combine
    raw = K.Ring()
    raw.scores = np.array([1.0, 2.0, np.nan])
    assert raw.p_value(1.5, 0.5) == C.randomized_conformal_p(raw.scores, 1.5, 0.5)


def test_conformal_branch_meets_b24_per_tick_budget():
    # B24 Perf: 40 keys x 31 detectors ~ 2 ms per tick on the bisect path
    # (median of 5 runs; generous 2x bound for shared CI).
    rng = np.random.default_rng(13)
    ring = make_ring(rng.exponential(size=M))
    xs = (rng.exponential(size=1240) * 0.5).tolist()
    runs = []
    for _ in range(5):
        t0 = time.perf_counter()
        for x in xs:
            K.p_from_ring(ring, x, 0.5)
        runs.append(time.perf_counter() - t0)
    assert sorted(runs)[2] < 0.004, runs


# ------------------------------------------------- spec v2.1 grain strata
def test_grain_and_meta_stratum_keys():
    from app.engines.behavior.lib import calib as C
    from app.engines.behavior.lib import m_calib as MC
    assert C.grain_stratum_key("wd_day", "h") == "wd_day|g:h"
    assert C.grain_stratum_key("wd_day", "q", prov=1) == "wd_day|g:q|p:1"
    assert C.grain_stratum_key("wd_night", "h", tercile=2) == "wd_night|r2|g:h"
    assert C.meta_stratum_key("nwd_day", "h") == "nwd_day|t:h"
    assert C.meta_stratum_key("nwd_day", "t", 60) == "nwd_day|t:t|60"
    with pytest.raises(ValueError):
        C.grain_stratum_key("wd_day", "x")
    # m_calib: grain strata for H / Q stream detectors, v2 strata otherwise
    assert MC.stratum_for("marg_int", "wd_day", 900, grain="h") == "wd_day|g:h"
    assert MC.stratum_for("marg_int_q", "wd_day", 900, grain="q", prov=1) == "wd_day|g:q|p:1"
    assert MC.stratum_for("identity", "wd_day", 900, 1, grain="h") == "wd_day|r1|g:h"
    assert MC.stratum_for("novelty", "wd_day", 900, grain="h") == "wd_day|900"
    assert MC.stratum_for("marg_int", "wd_day", 900) == "wd_day|900"
    code = (5, 1, 2, 0b0101)
    dp, terc, dt, g = MC.decode_pending(code)
    assert (dp, terc) == ("wd_night", 1) and g["dp_h"] == "wd_night" and g["dp_q"] == "nwd_day"
    assert g["prov"]["marg_int_q"] == 1 and g["prov"]["marg_shape_q"] == 0
