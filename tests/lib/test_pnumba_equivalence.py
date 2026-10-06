"""Optional numba kernels == their numpy reference, bit for bit (lib/pnumba).

The pure-numpy path is the reference: every kernel is run against it on the
same random inputs and every array of the state is compared exactly (floats
by their bytes, so signed zeros and NaN payloads count)."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from app.engines.behavior.lib import pevalue as PE
from app.engines.behavior.lib import pnumba as PNB

needs_numba = pytest.mark.skipif(not PNB.HAVE_NUMBA, reason="numba not installed (optional accelerator)")


def _state_bytes(ss: PE.SplitStats) -> dict:
    out = {}
    for k, v in vars(ss).items():
        if isinstance(v, np.ndarray):
            out[k] = (v.dtype.str, v.shape, v.tobytes())
        elif isinstance(v, (int, float, bool, type(None))):
            out[k] = repr(v)
    return out


def _run(ss: PE.SplitStats, events, numba_on: bool) -> list:
    old = PNB.set_enabled(numba_on)
    try:
        return [ss.update(v, b, p, w, d).tobytes() for v, b, p, w, d in events]
    finally:
        PNB.set_enabled(old)


def _events(rng: np.random.Generator, T: int, n: int, kv_vals: int, day0: int = 20000) -> list:
    out = []
    p = rng.dirichlet(np.ones(PE.K_B), size=T)
    for k in range(n):
        if rng.random() < 0.1:
            p = rng.dirichlet(np.ones(PE.K_B) * rng.choice([0.1, 1.0, 10.0]), size=T)
        vals = [int(rng.integers(0, kv_vals)) for _ in range(PE.C_MAX)]
        bins = [int(rng.integers(-1, PE.K_B)) if rng.random() < 0.9 else -1 for _ in range(T)]
        w = float(rng.choice([1.0, 0.5, 0.25, rng.random()]))
        out.append((vals, bins, p.copy(), w, day0 + k // max(1, n // 4)))
    return out


@needs_numba
@pytest.mark.parametrize("seed", range(12))
def test_split_stats_update_numba_equals_numpy(seed):
    rng = np.random.default_rng(seed)
    T = int(rng.integers(2, 12))
    ss = PE.SplitStats(T, k_b=PE.K_B, k_v=PE.K_V, C=PE.C_MAX)
    n_cand = int(rng.integers(1, PE.C_MAX + 1))              # masked (A < C) and full (A == C) pairs
    for i in range(n_cand):
        tm = rng.random(T) < 0.8
        ss.set_candidate(i, (f"a{i}", 0), float(rng.integers(2, 64)), tm.tolist())
    ref = copy.deepcopy(ss)
    ev = _events(rng, T, 400, int(rng.integers(2, 20)))
    d_nb = _run(ss, ev, True)
    d_np = _run(ref, ev, False)
    assert d_nb == d_np
    a, b = _state_bytes(ss), _state_bytes(ref)
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == b[k], k
    # the e-process the split rule reads is the same number
    for i in ss.active():
        assert ss.log2_e(i) == ref.log2_e(i)


@needs_numba
def test_split_stats_candidates_change_mid_stream():
    rng = np.random.default_rng(99)
    T = 6
    ss = PE.SplitStats(T, k_b=PE.K_B, k_v=PE.K_V, C=PE.C_MAX)
    for i in range(4):
        ss.set_candidate(i, (f"a{i}", 0), 8.0)
    ref = copy.deepcopy(ss)
    for part in range(5):
        ev = _events(rng, T, 120, 12, day0=20000 + 3 * part)
        assert _run(ss, ev, True) == _run(ref, ev, False)
        i = int(rng.integers(0, PE.C_MAX))
        for x in (ss, ref):
            if part % 2:
                x.drop_candidate(i)
            else:
                x.set_candidate(i, ("z", part), 4.0)
    a, b = _state_bytes(ss), _state_bytes(ref)
    for k in a:
        assert a[k] == b[k], k


@needs_numba
def test_pairwise_sum_is_numpys():
    rng = np.random.default_rng(3)
    for n in list(range(0, 40)) + [127, 128, 129, 255, 300, 1000]:
        for _ in range(20):
            x = rng.normal(size=n) * 10.0 ** rng.integers(-6, 7, size=n)
            assert PNB.pairwise_sum(x, 0, n) == x.sum()
    x = np.array([-0.0])
    assert np.signbit(PNB.pairwise_sum(x, 0, 1)) == np.signbit(x.sum())


@needs_numba
def test_min_max_follow_numpy_signed_zero_and_nan_rules():
    vals = [0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan]
    for a in vals:
        for b in vals:
            m = np.minimum(np.array([a]), np.array([b]))[0]
            M = np.maximum(np.array([a]), np.array([b]))[0]
            for got, want in ((PNB.np_min(a, b), m), (PNB.np_max(a, b), M)):
                assert np.array([got]).tobytes() == np.array([want]).tobytes(), (a, b)
