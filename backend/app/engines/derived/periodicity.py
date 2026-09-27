"""Periodicity engine (D0) — regular/automated timing on a wall-clock grid.

Human interaction is bursty and irregular; automation (schedulers, polling
clients, C2 beacons, health checks) is periodic. v1 autocorrelated the last 12
*emitted* count points, so idle ticks collapsed: a job firing once an hour at
15-min ticks looked like a constant series. v2 (engines.md D0) runs on the
zero-filled 6-h count grid (derived/fresh.grid), where the gaps are zeros and
the hour is a clean lag-4 peak.

* periodicity_score: the largest autocorrelation at lags 2..n/2 over the
  count targets (FFT, O(n log n) even at 60-s ticks: 360 points).
* beacon_lag: that peak's lag in *seconds* (0 when there is no peak).
* timing_regularity: 1 − CV of the dominant count's rate over *active* ticks
  only, so idle ticks do not make a steady poller look irregular.

A cadence change inside the span (900 s -> 60 s) makes the grid irregular;
the counts are then re-binned onto a regular grid at the coarsest spacing so
the autocorrelation lag keeps a single meaning. Regularity uses rates
(count / tick length), which do not depend on the cadence.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from ...core.engine import Context, Engine
from .aggregation import (ACTIVE_HORIZON_S, SPAN_S, active_mask, emit_window, ensure_retention,
                          grid_many, nan_cv, recently_active, tick_durations, window_dims)

MIN_TICKS = 8            # below this an autocorrelation peak is noise
MIN_LAG = 2


def autocorr_rows(x: np.ndarray, min_lag: int = MIN_LAG) -> Tuple[np.ndarray, np.ndarray]:
    """Per row of x[k, n]: (lag, r) of the strongest autocorrelation peak for
    lags in [min_lag, n // 2] (biased estimator r_j = Σ c_t c_{t+j} / Σ c_t²,
    so long lags with few pairs are damped). Rows that are constant, or too
    short, get (0, 0.0). All rows share one FFT."""
    k, n = x.shape
    lags, rs = np.zeros(k, dtype=np.int64), np.zeros(k)
    hi = n // 2
    if k == 0 or n < MIN_TICKS or hi < min_lag:
        return lags, rs
    c = x - x.mean(axis=1, keepdims=True)
    denom = np.einsum("ij,ij->i", c, c)
    ok = denom > 1e-12 * np.maximum(1.0, np.einsum("ij,ij->i", x, x))
    if not ok.any():
        return lags, rs
    m = 1 << int(math.ceil(math.log2(2 * n)))
    f = np.fft.rfft(c[ok], m, axis=1)
    acf = np.fft.irfft(f * np.conj(f), m, axis=1)[:, min_lag: hi + 1] / denom[ok, None]
    j = np.argmax(acf, axis=1)
    r = acf[np.arange(acf.shape[0]), j]
    pos = r > 0.0
    lags[ok] = np.where(pos, j + min_lag, 0)
    rs[ok] = np.where(pos, np.minimum(1.0, r), 0.0)
    return lags, rs


def autocorr_fft(x: np.ndarray, min_lag: int = MIN_LAG) -> Tuple[int, float]:
    """Single-series autocorr_rows."""
    lags, rs = autocorr_rows(np.asarray(x, dtype=np.float64)[None, :], min_lag)
    return int(lags[0]), float(rs[0])


def regular_counts(ts: np.ndarray, vals: np.ndarray, dur: np.ndarray, now: float,
                   span_s: float) -> Tuple[np.ndarray, float]:
    """Count rows vals[k, n] on a regular grid, and its step (s). A uniform
    grid is returned as is; otherwise counts are summed into bins of the
    coarsest spacing (capped so the span still holds MIN_TICKS bins),
    aligned at `now`."""
    if ts.size == 0:
        return vals, 0.0
    lo, hi = float(dur.min()), float(dur.max())
    if hi <= lo * 1.01:
        return vals, float(np.median(dur))
    step = min(hi, span_s / MIN_TICKS)
    idx = np.floor((now - ts) / step + 1e-9).astype(np.int64)
    nb = int(idx.max()) + 1
    out = np.zeros((vals.shape[0], nb))
    for b in range(vals.shape[0]):
        out[b] = np.bincount(idx, weights=np.nan_to_num(vals[b], nan=0.0), minlength=nb)
    return out[:, ::-1], step


class PeriodicityEngine(Engine):
    name = "derived.periodicity"
    layer = "derived"
    consumes = ["http.requests", "l4.flows", "dns.queries", "act.events"]
    produces = ["derived.periodicity_score", "derived.beacon_lag", "derived.timing_regularity"]
    description = ("Autocorrelation periodicity (lag in seconds) and 1-CV regularity over the "
                   "zero-filled 6-h count grid (dims span_s, n_active).")
    interval = 1

    # priority order: the first target with activity is the "dominant" one
    # for regularity; act.events covers TLS-only / L4-only automation
    TARGETS = ["http.requests", "l4.flows", "dns.queries", "act.events"]

    def __init__(self, span_s: float = SPAN_S, targets: Optional[List[str]] = None, **p):
        super().__init__(**p)
        self.span_s = float(span_s)
        self.targets = list(targets or self.TARGETS)

    def run(self, ctx: Context, observations=None) -> int:
        store, now = ctx.store, ctx.now
        ensure_retention(store)
        n = 0
        for system in store.systems():
            for entity in store.entities(system):
                if not recently_active(store, system, entity, now, min(self.span_s, ACTIVE_HORIZON_S)):
                    continue
                n += self._entity(ctx, system, entity)
        return n

    def _entity(self, ctx: Context, system: str, entity: str) -> int:
        store, now, dt, span = ctx.store, ctx.now, float(ctx.window_s), self.span_s
        ts, clock, names, mat = grid_many(store, system, entity, self.targets,
                                          {t: "counter" for t in self.targets}, now, span, dt)
        if ts.size == 0:
            return 0
        mat = np.nan_to_num(mat, nan=0.0)     # an unmeasured count adds no events
        busy = np.any(mat > 0, axis=1)
        if not busy.any():
            return 0          # no counted activity in the span: undefined, not "regular"
        names = [t for t, b in zip(names, busy) if b]
        mat = mat[busy]
        dur = tick_durations(ts, dt)
        mask = active_mask(clock, mat)
        dims = window_dims(span, int(mask.sum()))

        counts, step = regular_counts(ts, mat, dur, now, span)
        n = 0
        # Fewer than MIN_TICKS bins (e.g. a 6-h span at 3600-s ticks holds
        # 6): no autocorrelation can be estimated, so the score is undefined
        # and is not emitted (B01 reads a stale window feature as NaN). A 0
        # here would claim "not periodic" and teach the baselines of every
        # hourly-cadence bucket a point mass at 0, which a later 900-s tick
        # (a real estimate, typically 0.1-0.3) then hits at z >> 10.
        if counts.shape[1] >= MIN_TICKS:
            lags, scores = autocorr_rows(counts)
            b = int(np.argmax(scores))           # first (priority) target on ties
            best_score = float(scores[b])
            best_lag = float(lags[b]) * step if best_score > 0.0 else 0.0
            src = names[b]
            emit_window(ctx, system, entity, "derived.periodicity_score", best_score, dims, [src])
            emit_window(ctx, system, entity, "derived.beacon_lag", best_lag, dims, [src])
            n = 2
        # regularity: the dominant (first busy) count's rate on active ticks
        if int(mask.sum()) >= 2:
            rate = mat[0, mask] / dur[mask]
            regularity = max(0.0, 1.0 - min(nan_cv(rate), 1.0))     # constant rate => 1.0
            emit_window(ctx, system, entity, "derived.timing_regularity", regularity, dims,
                        [names[0]])
            n += 1
        return n
