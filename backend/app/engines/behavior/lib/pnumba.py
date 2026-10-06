"""Optional numba kernels of the pattern tree's hot loops (docs/lib3/progressive.md
§16.13, cost round 5).

numba is an OPTIONAL accelerator: when it is not installed (or
APPMON_NO_NUMBA=1, or `set_enabled(False)`), the callers run their pure-numpy
path. Every kernel here is written to give BIT-IDENTICAL results to that
path, which is the reference:

  * only IEEE add / sub / mul / div / compare in the kernels; transcendental
    functions (log2, exp2) stay in numpy, whose SIMD implementations differ
    from libm in the last bit on ~0.2 % of inputs, so the kernels gather their
    arguments and scatter their results around the numpy call;
  * numpy's reductions are reproduced exactly: `pairwise_sum` is numpy's
    pairwise summation (8 accumulators, blocks of 128), np.minimum /
    np.maximum keep numpy's NaN and signed-zero rule, float32 accumulators
    add in float32;
  * no fastmath (no reassociation, no FMA contraction).

tests/lib/test_pnumba_equivalence.py compares every kernel with its numpy
reference on random inputs, bit for bit.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np

try:                                                    # pragma: no cover - environment dependent
    if os.environ.get("APPMON_NO_NUMBA"):
        raise ImportError("disabled by APPMON_NO_NUMBA")
    import numba as _nb
    HAVE_NUMBA = True
except Exception:                                       # pragma: no cover
    _nb = None
    HAVE_NUMBA = False

_ENABLED = HAVE_NUMBA


def enabled() -> bool:
    """True when the numba kernels are used (installed and not switched off)."""
    return _ENABLED


def set_enabled(on: bool) -> bool:
    """Switch the kernels on / off (tests compare both paths); returns the old state."""
    global _ENABLED
    old = _ENABLED
    _ENABLED = bool(on) and HAVE_NUMBA
    return old


def njit(fn: Callable) -> Callable:
    """numba.njit(cache=True) when numba is available, else the function itself
    (plain Python: correct, slow; only used when the kernels are enabled)."""
    if not HAVE_NUMBA:
        return fn
    return _nb.njit(cache=True, nogil=True)(fn)


# ------------------------------------------------------------------ numpy rules
@njit
def _pw_block(a: Any, start: int, n: int) -> float:
    """numpy's pairwise_sum for n <= 128 (no further split)."""
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[start + i]
        return res
    r0 = a[start]
    r1 = a[start + 1]
    r2 = a[start + 2]
    r3 = a[start + 3]
    r4 = a[start + 4]
    r5 = a[start + 5]
    r6 = a[start + 6]
    r7 = a[start + 7]
    i = 8
    stop = n - (n % 8)
    while i < stop:
        r0 += a[start + i]
        r1 += a[start + i + 1]
        r2 += a[start + i + 2]
        r3 += a[start + i + 3]
        r4 += a[start + i + 4]
        r5 += a[start + i + 5]
        r6 += a[start + i + 6]
        r7 += a[start + i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[start + i]
        i += 1
    return res


@njit
def pairwise_sum(a: Any, start: int, n: int) -> float:
    """numpy's pairwise summation of a[start:start + n] (float64), as
    np.add.reduce computes it for a contiguous run (DOUBLE_pairwise_sum:
    blocks of <= 128 with 8 accumulators, larger runs split at n/2 rounded
    down to a multiple of 8). Iterative (numba's cache does not take
    recursive functions): the split tree is walked with an explicit stack
    of (start, n, partial) frames, combining left + right as the recursion
    does."""
    if n <= 128:
        return _pw_block(a, start, n)
    # post-order evaluation of the split tree
    st_s = np.empty(64, dtype=np.int64)
    st_n = np.empty(64, dtype=np.int64)
    st_state = np.zeros(64, dtype=np.int64)          # 0: new, 1: left done
    st_left = np.zeros(64)
    top = 0
    st_s[0] = start
    st_n[0] = n
    st_state[0] = 0
    result = 0.0
    while top >= 0:
        s0 = st_s[top]
        n0 = st_n[top]
        if n0 <= 128:
            v = _pw_block(a, s0, n0)
            top -= 1
            # deliver v to the parent
            while True:
                if top < 0:
                    result = v
                    break
                if st_state[top] == 0:
                    st_left[top] = v
                    st_state[top] = 1
                    n2 = st_n[top] // 2
                    n2 -= n2 % 8
                    top += 1
                    st_s[top] = st_s[top - 1] + n2
                    st_n[top] = st_n[top - 1] - n2
                    st_state[top] = 0
                    break
                v = st_left[top] + v
                top -= 1
            continue
        n2 = n0 // 2
        n2 -= n2 % 8
        top += 1
        st_s[top] = s0
        st_n[top] = n2
        st_state[top] = 0
    return result


@njit
def np_min(a: float, b: float) -> float:
    """np.minimum(a, b): a when a < b or a is NaN, else b (signed zeros: b)."""
    return a if (a < b or a != a) else b


@njit
def np_max(a: float, b: float) -> float:
    """np.maximum(a, b): a when a > b or a is NaN, else b."""
    return a if (a > b or a != a) else b


# ------------------------------------------------- pevalue.SplitStats.update
@njit
def ss_gather(cnt: Any, den: Any, p_leaf: Any, a: Any, jj: Any, tp: Any, bt: Any, alpha: float,
              kv: int, T: int, kb: int, arg: Any, pl: Any) -> None:
    """pl[k] = max(p_leaf[tp[k], bt[k]], 1e-12); arg[r, k] = (cnt[cell] +
    alpha pl[k]) / (den[row] + alpha) for candidate a[r] in slot jj[r]
    (the argument of the slot predictive's log2, before this event)."""
    Tp = tp.shape[0]
    for k in range(Tp):
        x = p_leaf[tp[k], bt[k]]
        pl[k] = x if (x > 1e-12 or x != x) else 1e-12
    for r in range(a.shape[0]):
        base = (a[r] * (kv + 1) + jj[r]) * T
        for k in range(Tp):
            row = base + tp[k]
            cell = row * kb + bt[k]
            arg[r, k] = (cnt[cell] + alpha * pl[k]) / (den[row] + alpha)


@njit
def ss_scatter(cnt: Any, den: Any, bm: Any, L1: Any, Gt: Any, Gp: Any, S: Any, tmask: Any,
               a: Any, jj: Any, tp: Any, bt: Any, lc: Any, ll: Any, w: float, kv: int, T: int,
               kb: int, C: int, dt: Any, d: Any) -> None:
    """Everything SplitStats.update does with the event's code lengths
    (lc = -log2 slot predictive, ll = -log2 leaf predictive, from numpy)."""
    A = a.shape[0]
    Tp = tp.shape[0]
    w32 = np.float32(w)
    for r in range(A):
        ai = a[r]
        base = (ai * (kv + 1) + jj[r]) * T
        for k in range(Tp):
            t = tp[k]
            row = base + t
            cell = row * kb + bt[k]
            bm[cell] = np.float32(bm[cell] + w32)
            m = tmask[ai, t]
            contrib = lc[r, k] if m else 0.0
            L1[ai * T + t] += w * contrib
            x = w * ((ll[k] if m else 0.0) - contrib)
            dt[r, k] = x
            Gt[ai * T + t] += x
            S[row] += x
    if A > 1:
        for r in range(A):
            for r2 in range(A):
                base = (a[r] * C + a[r2]) * T
                for k in range(Tp):
                    Gp[base + tp[k]] += dt[r, k]
    for r in range(A):
        d[r] = pairwise_sum(dt[r], 0, Tp)
    for r in range(A):
        base = (a[r] * (kv + 1) + jj[r]) * T
        for k in range(Tp):
            row = base + tp[k]
            cnt[row * kb + bt[k]] += w
            den[row] += w


@njit
def ss_tail(G: Any, n: Any, rows: Any, slot_ev: Any, slot_pri: Any, f_pri: float, a: Any, jj: Any,
            d: Any, w: float, C: int, d_out: Any, W: Any, D1: Any, Qaa: Any, Qab: Any, Rmin: Any,
            Rmax: Any) -> None:
    """Per-candidate totals, slot priorities and the pairwise rule-(S) sums."""
    A = a.shape[0]
    for r in range(A):
        G[a[r]] += d[r]
    for r in range(A):
        n[a[r]] += w
    for r in range(A):
        rows[a[r]] += 1.0
    for r in range(A):
        slot_ev[a[r], jj[r]] += w
    for i in range(slot_pri.shape[0]):
        for j in range(slot_pri.shape[1]):
            slot_pri[i, j] = slot_pri[i, j] * f_pri
    for r in range(A):
        slot_pri[a[r], jj[r]] += w
    for r in range(A):
        d_out[a[r]] = d[r]
    inf = np.inf
    if A == C:
        for i in range(C):
            di = d_out[i]
            for j in range(C):
                dj = d_out[j]
                W[i, j] += w
                D1[i, j] += di
                Qaa[i, j] += di * di / w
                Qab[i, j] += di * dj / w
                x = (di - dj) / w
                Rmin[i, j] = np_min(Rmin[i, j], x)
                Rmax[i, j] = np_max(Rmax[i, j], x)
        return
    am = np.zeros(C)
    for r in range(A):
        am[a[r]] = 1.0
    for i in range(C):
        di = d_out[i]
        for j in range(C):
            dj = d_out[j]
            both = am[i] * am[j]
            W[i, j] += w * both
            D1[i, j] += di * am[j]
            Qaa[i, j] += (di * di / w) * am[j]
            Qab[i, j] += di * dj / w
            x = (di - dj) / w
            pair = both > 0
            Rmin[i, j] = np_min(Rmin[i, j], x if pair else inf)
            Rmax[i, j] = np_max(Rmax[i, j], x if pair else -inf)
