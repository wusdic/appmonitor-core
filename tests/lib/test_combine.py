"""Tests for engines/behavior/lib/combine.py: wHMP, randomised conformal p,
e_day severity, seeded uniforms and the logit blend.

The statistical checks (KS uniformity, HMP rate under rho = 0.64) use fixed
seeds so they are deterministic; tolerances follow docs/lib3/engines.md B24 /
B25 unit tests.
"""
from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.special import ndtr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from app.engines.behavior.lib import combine as C  # noqa: E402

NAN = float("nan")
BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend"))


def ks_uniform(ps) -> float:
    """One-sample KS D against U(0, 1)."""
    p = np.sort(np.asarray(ps, dtype=np.float64))
    n = p.size
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - p), np.max(p - (i - 1) / n)))


# ======================================================================= whmp
def test_whmp_masking_regression_example():
    # B25 unit test (a): one p ~ 1 must not mask a small p (ACAT gives ~0.5).
    got = C.whmp([1e-3, 0.999, 0.5])
    assert got == pytest.approx(3.0 / (1e3 + 1 / 0.999 + 2.0), rel=1e-12)
    assert 2.9e-3 < got < 3.1e-3


def test_whmp_insensitive_to_p_one():
    assert C.whmp([1e-6, 1.0]) == pytest.approx(2e-6, rel=1e-5)
    assert C.whmp([1e-6] + [1.0] * 9) == pytest.approx(1e-5, rel=1e-4)


def test_whmp_single_value_is_identity():
    for p in (1e-12, 0.03, 0.5, 1.0):
        assert C.whmp([p]) == pytest.approx(p, rel=1e-15)


def test_whmp_nan_none_and_inf_are_dropped_with_their_weights():
    assert C.whmp([1e-3, NAN]) == pytest.approx(1e-3)
    assert C.whmp([1e-3, NAN, 0.5]) == C.whmp([1e-3, 0.5])
    # the dropped entry's weight is irrelevant, even NaN / huge
    assert C.whmp([1e-3, NAN, 0.5], [1.0, NAN, 2.0]) == C.whmp([1e-3, 0.5], [1.0, 2.0])
    assert C.whmp([1e-3, NAN, 0.5], [1.0, 1e9, 2.0]) == C.whmp([1e-3, 0.5], [1.0, 2.0])
    assert C.whmp([None, 0.2]) == pytest.approx(0.2)
    assert C.whmp([math.inf, -math.inf, 0.2]) == pytest.approx(0.2)


def test_whmp_all_nan_empty_or_zero_weight_is_nan():
    assert math.isnan(C.whmp([NAN, NAN]))
    assert math.isnan(C.whmp([]))
    assert math.isnan(C.whmp(np.array([NAN] * 3)))
    assert math.isnan(C.whmp(np.full(500, NAN)))          # vectorised path too
    assert math.isnan(C.whmp([0.1, 0.2], [0.0, 0.0]))
    assert math.isnan(C.whmp([NAN, 0.2], [1.0, 0.0]))


def test_whmp_zero_weight_entries_are_ignored():
    assert C.whmp([1e-9, 0.5], [0.0, 1.0]) == pytest.approx(0.5)
    assert C.whmp([1e-9, 0.5, 0.25], [0.0, 1.0, 1.0]) == C.whmp([0.5, 0.25])


def test_whmp_weights_scale_invariant_and_default_equal():
    ps = [1e-4, 0.3, 0.9, 0.02]
    ws = [0.5, 2.0, 1.0, 1.5]
    base = C.whmp(ps, ws)
    for k in (1e-6, 3.0, 1e6):
        assert C.whmp(ps, [k * w for w in ws]) == pytest.approx(base, rel=1e-12)
    assert C.whmp(ps, [7.0] * 4) == pytest.approx(C.whmp(ps), rel=1e-14)
    # explicit weighted formula
    want = sum(ws) / sum(w / p for p, w in zip(ps, ws))
    assert base == pytest.approx(want, rel=1e-13)


def test_whmp_bounded_by_min_and_max_p():
    rng = np.random.default_rng(1)
    for _ in range(200):
        k = int(rng.integers(1, 12))
        ps = 10.0 ** rng.uniform(-12, 0, size=k)
        ws = rng.uniform(0.01, 5.0, size=k)
        h = C.whmp(ps.tolist(), ws.tolist())
        assert ps.min() * (1 - 1e-12) <= h <= ps.max() * (1 + 1e-12)
        assert 0.0 < h <= 1.0


def test_whmp_clips_p_and_result():
    assert C.whmp([0.0]) == C.P_FLOOR
    assert C.whmp([-0.5, 0.5]) == pytest.approx(2 * C.P_FLOOR, rel=1e-9)
    assert C.whmp([2.0, 5.0]) == 1.0
    assert C.whmp([1.5, 0.5]) == pytest.approx(2 / 3)       # 1.5 clipped to 1
    got = C.whmp([0.0, 0.5])
    assert math.isfinite(got) and got > 0.0


def test_whmp_huge_weights_do_not_overflow():
    assert C.whmp([1e-300] * 3, [1e300] * 3) == pytest.approx(1e-300, rel=1e-9)
    assert C.whmp([1e-300, 1.0], [1e308, 1e308]) == pytest.approx(2e-300, rel=1e-9)
    big = np.full(200, 1e-300)
    assert C.whmp(big, np.full(200, 1e300)) == pytest.approx(1e-300, rel=1e-9)


@pytest.mark.parametrize("bad", [-1.0, NAN, math.inf])
def test_whmp_rejects_invalid_weights(bad):
    with pytest.raises(ValueError):
        C.whmp([0.1, 0.2], [1.0, bad])
    with pytest.raises(ValueError):
        C.whmp(np.full(100, 0.3), np.r_[np.ones(99), bad])


def test_whmp_rejects_length_mismatch():
    with pytest.raises(ValueError):
        C.whmp([0.1, 0.2], [1.0])
    with pytest.raises(ValueError):
        C.whmp(np.full(100, 0.3), np.ones(99))


def test_whmp_accepts_arrays_tuples_generators_and_numpy_scalars():
    ps = [1e-3, 0.999, 0.5]
    ref = C.whmp(ps)
    assert C.whmp(np.array(ps)) == ref
    assert C.whmp(tuple(ps)) == ref
    assert C.whmp(p for p in ps) == ref
    assert C.whmp([np.float64(p) for p in ps], np.ones(3)) == ref
    assert isinstance(C.whmp(np.array(ps)), float)
    assert C.whmp(np.array([[1e-3, 0.999], [0.5, np.nan]])) == ref       # raveled
    assert C.whmp(np.array([1e-3, None, 0.5, 0.999], dtype=object)) == pytest.approx(ref, rel=1e-14)


def test_whmp_vectorised_path_matches_scalar_path():
    rng = np.random.default_rng(2)
    n = 1000
    ps = 10.0 ** rng.uniform(-15, 0, size=n)
    ps[rng.random(n) < 0.1] = np.nan
    ps[:3] = [0.0, 1.0, 1.7]
    ws = rng.uniform(0.0, 3.0, size=n)
    ws[rng.random(n) < 0.1] = 0.0
    vec = C.whmp(ps, ws)                        # ndarray > _VEC_MIN -> numpy
    sca = C.whmp(ps.tolist(), ws.tolist())      # list -> scalar loop
    assert vec == pytest.approx(sca, rel=1e-12)
    assert C.whmp(ps) == pytest.approx(C.whmp(ps.tolist()), rel=1e-12)


def test_whmp_rate_under_rho064_dependence():
    # B25 unit test (b), pre-meta-calibration: five equicorrelated null
    # p-values (rho = 0.64); the wHMP exceedance rate at 1e-3 stays within
    # [0.8, 1.3]x nominal (Fisher would be ~32x here).
    rng = np.random.default_rng(3)
    n, k, rho = 200_000, 5, 0.64
    z = math.sqrt(rho) * rng.standard_normal((n, 1)) + math.sqrt(1 - rho) * rng.standard_normal((n, k))
    p = ndtr(-z)                                 # one-sided upper tail, U(0,1) marginals
    rows = p.tolist()
    hits = sum(1 for r in rows if C.whmp(r) <= 1e-3)
    ratio = hits / (n * 1e-3)
    assert 0.8 <= ratio <= 1.3, ratio


# ================================================== randomized_conformal_p
def test_conformal_formula_examples():
    ring = np.array([1.0, 2.0, 2.0, 3.0])
    assert C.randomized_conformal_p(ring, 2.0, 0.5) == pytest.approx((1 + 0.5 * 3) / 5)
    assert C.randomized_conformal_p(ring, 2.5, 0.25) == pytest.approx((1 + 0.25) / 5)
    assert C.randomized_conformal_p(ring, 9.0, 0.3) == pytest.approx(0.3 / 5)      # above all
    assert C.randomized_conformal_p(ring, -9.0, 0.3) == pytest.approx((4 + 0.3) / 5)  # below all
    assert C.randomized_conformal_p(ring, 1.0, 0.0) == pytest.approx(3 / 5)
    assert C.randomized_conformal_p(ring, 1.0, 1.0) == pytest.approx(5 / 5)


def test_conformal_matches_brute_force_with_ties():
    rng = np.random.default_rng(4)
    for _ in range(300):
        m = int(rng.integers(0, 40))
        ring = np.sort(rng.integers(0, 6, size=m).astype(np.float64))
        s = float(rng.integers(-1, 7))
        u = float(rng.random())
        want = (np.sum(ring > s) + u * (np.sum(ring == s) + 1)) / (m + 1)
        assert C.randomized_conformal_p(ring, s, u) == pytest.approx(want, rel=1e-14, abs=0)


def test_conformal_open_unit_interval_and_monotone_in_s():
    ring = np.sort(np.random.default_rng(5).exponential(size=256))
    u = C.seeded_uniform("s", "e", "d", 1.0)
    grid = np.linspace(-1, 12, 400)
    ps = [C.randomized_conformal_p(ring, s, u) for s in grid]
    assert all(0.0 < p < 1.0 for p in ps)
    assert all(a >= b for a, b in zip(ps, ps[1:]))
    assert ps[-1] == pytest.approx(u / 257)
    assert ps[0] == pytest.approx((256 + u) / 257)


def test_conformal_nan_empty_and_bad_u():
    ring = np.array([0.0, 1.0])
    assert math.isnan(C.randomized_conformal_p(ring, NAN, 0.5))
    assert math.isnan(C.randomized_conformal_p(ring, None, 0.5))
    assert math.isnan(C.randomized_conformal_p(ring, 0.5, NAN))
    assert C.randomized_conformal_p(np.empty(0), 3.0, 0.37) == 0.37
    assert C.randomized_conformal_p([], 3.0, 0.37) == 0.37
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            C.randomized_conformal_p(ring, 0.5, bad)


def test_conformal_ignores_trailing_nan_and_accepts_lists():
    clean = np.array([0.0, 0.0, 1.0, 2.0])
    dirty = np.sort(np.r_[clean, np.nan, np.nan])      # NaN sorts last
    for s in (-1.0, 0.0, 0.5, 2.0, 5.0):
        want = C.randomized_conformal_p(clean, s, 0.4)
        assert C.randomized_conformal_p(dirty, s, 0.4) == want
        assert C.randomized_conformal_p(clean.tolist(), s, 0.4) == want
    assert C.randomized_conformal_p(np.array([np.nan]), 1.0, 0.4) == 0.4


def test_conformal_float32_ring_counts_ties_at_ring_precision():
    ring32 = np.array([1.1, 1.1, 2.0], dtype=np.float32)
    # s = 1.1 (float64) equals the float32-rounded entries at ring precision
    assert C.randomized_conformal_p(ring32, 1.1, 0.5) == pytest.approx((1 + 0.5 * 3) / 4)
    # the documented caller convention: float32-rounded values in a float64 ring
    ring64 = ring32.astype(np.float64)
    s = float(np.float32(1.1))
    assert C.randomized_conformal_p(ring64, s, 0.5) == pytest.approx((1 + 0.5 * 3) / 4)


def test_conformal_float32_ring_huge_score_rounds_to_inf_without_warning():
    # Regression: np.float32(1e300) raised an overflow RuntimeWarning (an
    # exception under -W error). |s| beyond float32 range must round to +-inf
    # silently, exactly as a float32 cast would.
    import warnings

    ring32 = np.array([0.0, 1.0, 2.0, np.inf], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # s -> +inf ties with the inf entry: (0 + u*(1 + 1)) / 5
        assert C.randomized_conformal_p(ring32, 1e300, 0.5) == pytest.approx(1.0 / 5)
        assert C.randomized_conformal_p(ring32, 3.4028235677973366e38, 0.5) == pytest.approx(1.0 / 5)
        # s -> -inf lies below every entry: (4 + u) / 5
        assert C.randomized_conformal_p(ring32, -1e300, 0.5) == pytest.approx(4.5 / 5)
        # largest finite-rounding value stays finite: above 2.0, below inf
        s = float(np.nextafter(3.4028235677973366e38, 0.0))
        assert C.randomized_conformal_p(ring32, s, 0.5) == pytest.approx((1 + 0.5) / 5)


def _null_rings(rng, n_draws, m, zero_frac):
    x = rng.exponential(size=(n_draws, m + 1))
    if zero_frac:
        x[rng.random((n_draws, m + 1)) < zero_frac] = 0.0
    x = x.astype(np.float32).astype(np.float64)         # stored-precision convention
    rings = np.sort(x[:, :m], axis=1)
    return rings, x[:, m]


@pytest.mark.parametrize("m", [256, 16])
def test_conformal_uniform_under_null_continuous(m):
    # B24 unit test (a): exchangeable Exp(1) scores -> KS D < 0.03 over 2000 draws.
    # (A fixed 2000-draw sample of an exactly uniform p exceeds 0.03 ~6% of
    # the time; the seed is fixed, and the 20000-draw test below is the
    # seed-robust check.)
    rng = np.random.default_rng(0)
    rings, s = _null_rings(rng, 2000, m, 0.0)
    ps = [C.randomized_conformal_p(rings[i], s[i], C.seeded_uniform("sys", f"e{m}", "d", float(i)))
          for i in range(2000)]
    assert ks_uniform(ps) < 0.03


@pytest.mark.parametrize("m", [256, 16])
def test_conformal_uniform_with_90pct_ties_at_zero(m):
    # B24 unit test (b): sparse detector with 90% zeros. Randomised p is
    # uniform; the deterministic control (u = 1) has a point mass at p = 1.
    rng = np.random.default_rng(0)
    rings, s = _null_rings(rng, 2000, m, 0.9)
    assert np.mean(s == 0.0) > 0.85
    ps = [C.randomized_conformal_p(rings[i], s[i], C.seeded_uniform("sys", f"e{m}", "d", float(i)))
          for i in range(2000)]
    det = [C.randomized_conformal_p(rings[i], s[i], 1.0) for i in range(2000)]
    assert ks_uniform(ps) < 0.03
    assert ks_uniform(det) > 0.5
    assert all(0.0 < p < 1.0 for p in ps)

def test_conformal_exactly_uniform_large_sample():
    # Randomised conformal p is exactly U(0, 1) for exchangeable scores at any
    # |C| and any tie structure: over 20000 draws D stays far below 0.03
    # (P(D > 0.015) ~ 3e-4 under exact uniformity).
    rng = np.random.default_rng(99)
    rings, s = _null_rings(rng, 20000, 64, 0.9)
    ps = [C.randomized_conformal_p(rings[i], s[i], C.seeded_uniform("big", float(i)))
          for i in range(20000)]
    assert ks_uniform(ps) < 0.015
    # tail calibration: P(p <= 0.01) ~ 0.01 (sd ~ 7e-4)
    assert abs(float(np.mean(np.asarray(ps) <= 0.01)) - 0.01) < 0.003


# ==================================================================== e_day
def test_e_day_formula_and_cadences():
    assert C.e_day(1e-3, 900.0) == pytest.approx(1e-3 * 86400 / 900)
    assert C.e_day(1e-3, 900.0) == pytest.approx(0.096)
    assert C.e_day(1e-4, 60.0) == pytest.approx(0.144)
    assert C.e_day(1.0, 86400.0) == 1.0
    assert C.e_day(0.0, 60.0) == 0.0
    assert C.e_day(np.float64(1e-3), 300) == pytest.approx(0.288)
    assert C.SECONDS_PER_DAY == 86400.0


def test_e_day_nan_policy_and_bad_dt():
    assert math.isnan(C.e_day(NAN, 900.0))
    assert math.isnan(C.e_day(None, 900.0))
    assert math.isnan(C.e_day(0.1, NAN))
    assert math.isnan(C.p_from_e_day(NAN, 900.0))
    assert math.isnan(C.p_from_e_day(0.1, NAN))
    for bad in (0.0, -60.0, math.inf):
        with pytest.raises(ValueError):
            C.e_day(0.1, bad)
        with pytest.raises(ValueError):
            C.p_from_e_day(0.1, bad)


@pytest.mark.parametrize("dt", [60.0, 300.0, 900.0, 3600.0])
def test_p_from_e_day_inverts_e_day(dt):
    for p in (1e-12, 1e-6, 3e-4, 0.01, 0.5):
        if C.e_day(p, dt) <= 86400 / dt:
            assert C.p_from_e_day(C.e_day(p, dt), dt) == pytest.approx(p, rel=1e-14)
    for e in (3e-6, 3e-4, 3e-3, 0.03):
        assert C.e_day(C.p_from_e_day(e, dt), dt) == pytest.approx(e, rel=1e-14)
    # cadence-free budget: expected null ticks per day with e_day <= e is e
    ticks_per_day = 86400 / dt
    assert C.p_from_e_day(0.03, dt) * ticks_per_day == pytest.approx(0.03)


def test_p_from_e_day_clips_to_unit_interval():
    assert C.p_from_e_day(1e6, 900.0) == 1.0
    assert C.p_from_e_day(math.inf, 900.0) == 1.0
    assert C.p_from_e_day(96.0, 900.0) == 1.0          # once per tick
    assert C.p_from_e_day(0.0, 900.0) == 0.0
    assert C.p_from_e_day(-1.0, 900.0) == 0.0
    assert C.p_from_e_day(0.03, 900.0) == pytest.approx(3.125e-4)


# ============================================================ seeded_uniform
def test_seeded_uniform_follows_documented_formula():
    keys = ("web", "10.0.0.1", "silence", 1700000900.0)
    msg = "|".join(["'web'", "'10.0.0.1'", "'silence'", "1700000900.0"]).encode()
    h = int.from_bytes(hashlib.blake2b(msg, digest_size=8).digest(), "big")
    assert C.seeded_uniform(*keys) == (h + 0.5) / 2 ** 64


def test_seeded_uniform_golden_values_are_stable():
    # Pinned literals: any change here breaks bit-identical replay (B29).
    assert C.seeded_uniform("sys", "ent", "det", 1700000000.0) == 0.18058294844424
    assert C.seeded_uniform("web", "10.0.0.1", "silence", 1700000900.0) == 0.7429914677027796
    assert C.seeded_uniform() == 0.8931675160897388


def test_seeded_uniform_deterministic_and_open_interval():
    us = [C.seeded_uniform("s", "e", "d", 1.7e9 + 60.0 * i) for i in range(5000)]
    assert us == [C.seeded_uniform("s", "e", "d", 1.7e9 + 60.0 * i) for i in range(5000)]
    assert all(0.0 < u < 1.0 for u in us)
    assert len(set(us)) == 5000
    assert ks_uniform(us[:2000]) < 0.03
    assert 0.48 < float(np.mean(us)) < 0.52


def test_seeded_uniform_key_types():
    u = C.seeded_uniform
    # 1, 1.0, '1', True are four different keys
    assert len({u(1), u(1.0), u("1"), u(True)}) == 4
    # numpy scalars normalise to their Python counterparts
    assert u(np.float64(1.5)) == u(1.5)
    assert u(np.float32(0.5)) == u(0.5)                 # exact in float32
    assert u(np.int64(7)) == u(7)
    assert u(np.str_("x")) == u("x")
    assert u(np.bool_(True)) == u(True)
    assert u(("a", np.float64(2.0))) == u(("a", 2.0)) == u(["a", 2.0])
    assert u(None) == u(None) and u(None) != u("None")


def test_seeded_uniform_order_and_separator_safety():
    u = C.seeded_uniform
    assert u("a", "b") != u("b", "a")
    assert u("a|b") != u("a", "b")
    assert u("a", "b|c") != u("a|b", "c")
    assert u("e", "d", 0.0) != u("e", "d", -0.0)


def test_seeded_uniform_never_hits_bounds():
    assert C._u_from_int(0) > 0.0
    assert C._u_from_int(2 ** 64 - 1) < 1.0
    assert C._u_from_int(2 ** 64 - 2 ** 10) < 1.0


def test_seeded_uniform_stable_across_processes():
    code = ("import sys; sys.path.insert(0, %r); "
            "from app.engines.behavior.lib.combine import seeded_uniform as u; "
            "print(repr(u('sys', 'ent', 'det', 1700000000.0)), repr(u('x', 3, 2.5)))") % BACKEND
    here = f"{C.seeded_uniform('sys', 'ent', 'det', 1700000000.0)!r} {C.seeded_uniform('x', 3, 2.5)!r}"
    outs = set()
    for seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                                text=True, check=True).stdout.strip())
    assert outs == {here}


# ============================================================ e_day_severity
@pytest.mark.parametrize("e,want", [
    (0.0, "critical"), (1e-9, "critical"), (3e-6, "critical"),
    (3.0001e-6, "high"), (1e-5, "high"), (3e-4, "high"),
    (3.0001e-4, "medium"), (1e-3, "medium"), (3e-3, "medium"),
    (3.0001e-3, "low"), (0.01, "low"), (0.03, "low"),
    (0.0301, None), (1.0, None), (96.0, None), (math.inf, None),
])
def test_e_day_severity_ladder(e, want):
    assert C.e_day_severity(e) == want


def test_e_day_severity_alpha_mult_scales_thresholds():
    assert C.e_day_severity(0.05, 2.0) == "low"
    assert C.e_day_severity(0.05) is None
    assert C.e_day_severity(0.02, 0.5) is None
    assert C.e_day_severity(5e-6, 2.0) == "critical"
    assert C.e_day_severity(5e-6, 1.0) == "high"
    assert C.e_day_severity(2e-6, 0.25) == "high"


def test_e_day_severity_nan_and_bad_alpha():
    assert C.e_day_severity(NAN) is None
    assert C.e_day_severity(None) is None
    for bad in (0.0, -1.0, NAN, math.inf):
        with pytest.raises(ValueError):
            C.e_day_severity(1e-3, bad)


def test_severity_ladder_constant_is_ordered():
    names = [n for n, _ in C.SEVERITY_E_DAY]
    thr = [t for _, t in C.SEVERITY_E_DAY]
    assert names == ["critical", "high", "medium", "low"]
    assert thr == sorted(thr)


# =============================================================== logit_blend
def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def _lg(p):
    return math.log(p / (1 - p))


def test_logit_blend_endpoints_and_fixed_point():
    for p1, p2 in ((1e-8, 0.4), (0.3, 0.9), (1e-250, 0.5)):
        assert C.logit_blend(p1, p2, 1.0) == pytest.approx(p1, rel=1e-12)
        assert C.logit_blend(p1, p2, 0.0) == pytest.approx(p2, rel=1e-12)
    for p in (1e-200, 1e-6, 0.37, 0.999):
        assert C.logit_blend(p, p, 0.3) == pytest.approx(p, rel=1e-12)


def test_logit_blend_formula_symmetry_and_monotone():
    p1, p2 = 1e-3, 0.2
    for w in (0.1, 0.5, 0.64):
        assert C.logit_blend(p1, p2, w) == pytest.approx(_sig(w * _lg(p1) + (1 - w) * _lg(p2)), rel=1e-12)
        assert C.logit_blend(p1, p2, w) == pytest.approx(C.logit_blend(p2, p1, 1 - w), rel=1e-12)
    ws = np.linspace(0, 1, 50)
    vals = [C.logit_blend(p1, p2, w) for w in ws]
    assert all(a > b for a, b in zip(vals, vals[1:]))        # more weight on the smaller p
    assert all(p1 <= v <= p2 for v in vals)
    # half-way in logit space: 1e-8 vs 0.5 -> ~1e-4
    assert C.logit_blend(1e-8, 0.5, 0.5) == pytest.approx(1e-4, rel=1e-3)


def test_logit_blend_small_sample_weight_usage():
    # calib.blend_small_sample weight n / (n + 64): n = 64 is the logit midpoint
    n = 64
    got = C.logit_blend(0.01, 0.2, n / (n + 64))
    assert got == pytest.approx(_sig(0.5 * _lg(0.01) + 0.5 * _lg(0.2)), rel=1e-12)


def test_logit_blend_nan_passthrough():
    assert C.logit_blend(NAN, 0.3, 0.9) == 0.3
    assert C.logit_blend(0.3, NAN, 0.1) == 0.3
    assert C.logit_blend(None, 0.25, 0.5) == 0.25
    assert math.isnan(C.logit_blend(NAN, NAN, 0.5))
    # passthrough is the value as given, even when NaN weight
    assert C.logit_blend(NAN, 0.3, NAN) == 0.3


def test_logit_blend_extremes_are_clipped_and_finite():
    lo = C.logit_blend(0.0, 0.0, 0.5)
    assert lo == pytest.approx(1e-300, rel=1e-10) and lo > 0.0
    hi = C.logit_blend(1.0, 1.0, 0.5)
    assert 0.999_999_999_999_999 < hi < 1.0
    mid = C.logit_blend(0.0, 1.0, 0.5)
    assert 0.0 < mid < 1.0 and math.isfinite(mid)
    assert C.logit_blend(-0.5, 0.5, 1.0) == pytest.approx(1e-300, rel=1e-10)


def test_logit_blend_weight_clipping_and_nan_weight():
    assert C.logit_blend(1e-4, 0.5, 2.0) == C.logit_blend(1e-4, 0.5, 1.0)
    assert C.logit_blend(1e-4, 0.5, -1.0) == C.logit_blend(1e-4, 0.5, 0.0)
    with pytest.raises(ValueError):
        C.logit_blend(1e-4, 0.5, NAN)
