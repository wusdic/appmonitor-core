"""Trend engine (D0) — level, direction and a split-half change test.

EWMA (smoothed level), slope (direction/strength) and a change-point score
(recent half vs prior half) for key metrics. They feed the UI's "rising /
falling" read and lib-4 rules.

v1 used the last 40 *emitted* points and a robust z against the prior half's
median/MAD. On sparse activity both break: idle time vanishes, and a series
that is active one tick in four has median 0 and MAD 0, so any activity at all
scored the saturated ±6. v2 (engines.md D0) works on a 12-h wall-clock grid
(derived/fresh.grid: counters zero-filled, gauges NaN) and tests the halves
with a Welch z of the mean difference whose variance is floored — at the
pooled mean for counters (Poisson), at 1 % of the level for gauges — so a
stationary sparse series reads ≈ 0 and a real step reads large.

Cadence can change mid-span: counter values are rescaled to the current tick
length, the EWMA uses a wall-clock half-life (not a per-tick alpha) and the
slope is fitted against time and reported per current tick, so a switch from
900-s to 60-s ticks is not itself a trend.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from ...core.engine import Context, Engine
from .aggregation import (ACTIVE_HORIZON_S, HOUR, TREND_SPAN_S, active_mask, emit_window,
                          ensure_retention, grid_many, metric_kind, recently_active, tick_durations,
                          window_dims)

EWMA_HALF_LIFE_S = 1800.0     # ≈ v1's alpha = 0.3 at 15-min ticks
MIN_POINTS = 6
MIN_HALF = 3
Z_CLIP = 10.0


def ewma_rows(ts: np.ndarray, x: np.ndarray, fin: np.ndarray,
              half_life_s: float = EWMA_HALF_LIFE_S) -> np.ndarray:
    """Per-row EWMA with a wall-clock half-life over the finite points.

    The recursion acc += a_i (x_i − acc), a_i = 1 − 2^(−(t_i − t_prev)/H),
    unrolls to Σ_i w_i x_i with w_i = a_i · 2^(−(t_last − t_i)/H) and a = 1
    at a row's first finite point; t_prev / t_last are that row's previous /
    last finite times. NaN for a row with no finite point."""
    k, n = x.shape
    idx = np.where(fin, np.arange(n)[None, :], -1)
    last = np.maximum.accumulate(idx, axis=1)              # last finite index up to i
    prev = np.concatenate([np.full((k, 1), -1), last[:, :-1]], axis=1)
    t_prev = ts[np.maximum(prev, 0)]
    a = np.where(prev < 0, 1.0, 1.0 - np.exp2(-(ts[None, :] - t_prev) / half_life_s))
    t_last = ts[np.maximum(last[:, -1], 0)]
    w = np.where(fin, a * np.exp2(-(t_last[:, None] - ts[None, :]) / half_life_s), 0.0)
    out = (w * np.where(fin, x, 0.0)).sum(axis=1)
    return np.where(last[:, -1] >= 0, out, math.nan)


def slope_rows(ts: np.ndarray, x: np.ndarray, fin: np.ndarray) -> np.ndarray:
    """Per-row least-squares slope of x against time (units per second) over
    the finite points; 0 with fewer than 2 points or no time spread."""
    cnt = fin.sum(axis=1)
    t = np.where(fin, (ts - ts[-1])[None, :], 0.0)
    y = np.where(fin, x, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        tm = t.sum(axis=1) / cnt
        ym = y.sum(axis=1) / cnt
        tc = np.where(fin, t - tm[:, None], 0.0)
        den = (tc * tc).sum(axis=1)
        num = (tc * np.where(fin, y - ym[:, None], 0.0)).sum(axis=1)
        out = num / den
    return np.where((cnt >= 2) & (den > 1e-9), out, 0.0)


def split_half_z(ts: np.ndarray, x: np.ndarray, fin: np.ndarray, split_ts: float,
                 is_counter: np.ndarray) -> np.ndarray:
    """Per-row Welch z of mean(recent) − mean(prior), halves split at
    split_ts, with a floored variance (counters: the pooled mean, Poisson;
    gauges: (1 % of the level)²), clipped to ±Z_CLIP. NaN when either half
    has fewer than MIN_HALF finite points."""
    first = fin & (ts <= split_ts)[None, :]
    second = fin & (ts > split_ts)[None, :]
    x0 = np.where(fin, x, 0.0)

    def moments(m):
        c = m.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.where(m, x0, 0.0).sum(axis=1) / c
            var = (np.where(m, x0 - mu[:, None], 0.0) ** 2).sum(axis=1) / c
        return c, mu, var

    na, ma, va = moments(first)
    nb, mb, vb = moments(second)
    ok = (na >= MIN_HALF) & (nb >= MIN_HALF)
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = np.abs((na * ma + nb * mb) / (na + nb))
        floor = np.where(is_counter, np.maximum(pooled, 1e-6), (0.01 * pooled) ** 2 + 1e-12)
        se = np.sqrt(np.maximum(va, floor) / na + np.maximum(vb, floor) / nb)
        z = np.clip((mb - ma) / se, -Z_CLIP, Z_CLIP)
    return np.where(ok, z, math.nan)


class TrendEngine(Engine):
    name = "derived.trend"
    layer = "derived"
    consumes = ["l4.*", "http.*", "dns.*", "probe.*", "act.events"]
    produces = ["derived.<metric>.{ewma,slope,changepoint}"]
    description = ("EWMA level, slope and split-half change z over a 12-h wall-clock grid "
                   "(dims span_s, n_active).")
    interval = 1

    DEFAULT_TARGETS = [
        "l4.bytes_up", "l4.distinct_peers", "http.requests",
        "http.latency_ms_avg", "dns.queries", "probe.rtt_ms",
    ]

    def __init__(self, targets: Optional[List[str]] = None, span_s: float = TREND_SPAN_S,
                 kinds: Optional[Dict[str, str]] = None, **p):
        super().__init__(**p)
        self.targets = list(targets or self.DEFAULT_TARGETS)
        self.span_s = float(span_s)
        self.kinds = {t: (kinds or {}).get(t, metric_kind(t)) for t in self.targets}

    def run(self, ctx: Context, observations=None) -> int:
        store, now = ctx.store, ctx.now
        # the targets must survive the whole 12-h span (raw scalars default to 6 h)
        ensure_retention(store, self.targets, self.span_s + HOUR)
        n = 0
        for system in store.systems():
            for entity in store.entities(system):
                if not recently_active(store, system, entity, now, ACTIVE_HORIZON_S):
                    continue
                n += self._entity(ctx, system, entity)
        return n

    def _entity(self, ctx: Context, system: str, entity: str) -> int:
        store, now, dt, span = ctx.store, ctx.now, float(ctx.window_s), self.span_s
        ts, clock, names, mat = grid_many(store, system, entity, self.targets, self.kinds,
                                          now, span, dt)
        if ts.size == 0 or not names:
            return 0
        dur = tick_durations(ts, dt)
        is_c = np.array([self.kinds[t] == "counter" for t in names])
        dims = window_dims(span, int(active_mask(clock, mat[is_c]).sum()))
        # counters per current tick, so a cadence switch is not a trend
        x = np.where(is_c[:, None], mat * (dt / dur)[None, :], mat)
        fin = np.isfinite(x)
        stats = (("ewma", ewma_rows(ts, x, fin)),
                 ("slope", slope_rows(ts, x, fin) * dt),          # per current tick
                 ("changepoint", split_half_z(ts, x, fin, now - span / 2.0, is_c)))
        enough = fin.sum(axis=1) >= MIN_POINTS
        n = 0
        for i, target in enumerate(names):
            if not enough[i]:
                continue
            for stat, vals in stats:
                v = float(vals[i])
                if math.isfinite(v):
                    emit_window(ctx, system, entity, f"derived.{target}.{stat}", v, dims,
                                [target])
                    n += 1
        return n
