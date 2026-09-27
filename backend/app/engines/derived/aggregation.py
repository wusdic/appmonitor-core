"""Aggregation engine (D0) — window statistics on a zero-filled wall-clock grid.

v1 took the last 30 *emitted* points of each raw metric. Raw engines only emit
when an entity is active, so idle time vanished: a host active one tick in four
had the same "mean" as one active on every tick, and a host that went silent
kept its last 30 busy points forever. v2 (engines.md D0) summarises the last
6 h of wall-clock ticks instead (derived/fresh.grid): a tick with no raw point
is a true 0 for counters (the entity was silent — act.events == 0) and NaN for
gauges (nothing was measured), and statistics skip NaN. An entity with no
observation in the span gets nothing: its window is undefined, not "zero".

Cadence may change mid-run (900 s -> 60 s), so a counter's per-tick values are
first put on a common footing: each is rescaled to the *current* tick length
(v * Δt_now / Δt_tick). `sum` stays the true total over the span; mean / p95 /
max / cv are in "per current tick" units and equal the plain statistics when
the cadence is uniform.

This module also holds the small helpers the three D0 engines share
(retention, kinds, tick durations, activity mask).
"""
from __future__ import annotations

import math
import weakref
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import DerivedMetric, MetricKind
from . import fresh

HOUR = 3600.0
SPAN_S = 6 * HOUR               # aggregation / periodicity window
TREND_SPAN_S = 12 * HOUR        # trend split-half window
ACTIVE_HORIZON_S = 6 * HOUR     # no window metrics for an entity silent longer than this

# Retention (contract B + reviewer correction): the D0/D2 grid inputs are
# kept for 24 h so the 12 h trend and the 24 h duty cycle see their whole span.
D0_INPUTS = ("http.requests", "l4.flows", "dns.queries", "act.events")
D0_INPUT_MAX_AGE_S = 24 * HOUR

# Raw counters: a missing tick is a true 0. Everything else (averages, rates,
# probe gauges) is NaN when missing.
COUNTER_METRICS = frozenset({
    "act.events", "l3.bytes_total", "l4.flows", "l4.bytes_up", "l4.bytes_down",
    "l4.distinct_peers", "l4.distinct_dports", "l4.syn_count", "l4.pkts_total",
    "http.requests", "http.status_3xx", "http.status_4xx", "http.status_5xx",
    "http.get_count", "http.write_count", "tls.handshakes", "dns.queries",
    "dns.txt_count", "dns.nxdomain",
})

_RETAINED: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def ensure_retention(store, extra: Iterable[str] = (), extra_age_s: float = 0.0) -> None:
    """Idempotently extend raw retention for the D0 grid inputs (and `extra`
    names to at least `extra_age_s`). set_retention clears the store's rule
    cache, so it is called once per (store, name), not every tick."""
    done = _RETAINED.get(store)
    if done is None:
        done = _RETAINED[store] = {}
    for name in D0_INPUTS:
        if done.get(name, 0.0) < D0_INPUT_MAX_AGE_S:
            store.set_retention(name, max_age_s=D0_INPUT_MAX_AGE_S)
            done[name] = D0_INPUT_MAX_AGE_S
    for name in extra:
        if done.get(name, 0.0) < extra_age_s:
            store.set_retention(name, max_age_s=extra_age_s)
            done[name] = extra_age_s


def metric_kind(name: str) -> str:
    """Grid kind of a raw metric: 'counter' (zero-filled) or 'gauge' (NaN)."""
    return "counter" if name in COUNTER_METRICS else "gauge"


def recently_active(store, system: str, entity: str, now: float,
                    horizon_s: float = ACTIVE_HORIZON_S) -> bool:
    """The entity had an observation in (now - horizon_s, now]. last_seen is
    touched only by real observations (zero-fill uses touch=False)."""
    ls = store.last_seen(system, entity)
    return ls is not None and now - horizon_s < ls <= now


def tick_durations(ts: np.ndarray, dt_now: float) -> np.ndarray:
    """Length of each grid tick: the spacing to the previous tick (the first
    tick borrows its successor's spacing; a lone tick uses Δt_now)."""
    n = ts.size
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.array([float(dt_now)])
    d = np.empty(n)
    d[1:] = np.diff(ts)
    d[0] = d[1]
    return np.maximum(d, 1.0)


def active_mask(clock: np.ndarray, counters: Optional[np.ndarray] = None) -> np.ndarray:
    """Ticks with activity: act.events > 0, or any counter row > 0 (the
    latter keeps the mask right on pipelines without R2's act.events clock)."""
    m = np.nan_to_num(clock, nan=0.0) > 0
    if counters is not None and counters.size:
        m |= np.any(np.nan_to_num(counters, nan=0.0) > 0, axis=0)
    return m


def grid_many(store, system: str, entity: str, names: Sequence[str], kinds: Dict[str, str],
              now: float, span_s: float, dt_s: float, skip_stale: bool = True
              ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """fresh.grid for the clock and several metrics at once.

    Same semantics as fresh.grid (clock = fresh.grid_times; a value counts
    only on an exact tick timestamp; counters 0-filled, gauges NaN), but the
    clock is read once and the fill is a searchsorted instead of a Python
    loop — at 60-s ticks a 12-h grid has 720 points per metric.
    Returns (ts[n], clock[n], kept names, values[k, n]); with skip_stale, a
    metric with no raw point in the span is left out (never used there).
    """
    ts = fresh.grid_times(store, system, entity, now, span_s, dt_s)
    cap = 2 * int(ts.size) + 16
    lo = ts[0] - 1e-6 if ts.size else now
    kept: List[str] = []
    rows: List[np.ndarray] = []
    clock = None
    for name in (fresh.CLOCK, *names):
        is_clock = name == fresh.CLOCK and clock is None
        tail = store.raw_tail(system, entity, name, cap)
        if not is_clock and skip_stale and (not tail or tail[-1].ts <= now - span_s):
            continue
        kind = "counter" if name == fresh.CLOCK else kinds.get(name, metric_kind(name))
        vals = np.full(ts.shape, 0.0 if kind in fresh.COUNTER_KINDS else math.nan)
        pts = [(m.ts, m.value) for m in tail
               if lo <= m.ts <= now and isinstance(m.value, (int, float))]
        if pts and ts.size:
            pt = np.fromiter((p[0] for p in pts), dtype=np.float64, count=len(pts))
            pv = np.fromiter((p[1] for p in pts), dtype=np.float64, count=len(pts))
            if pt.size > 1 and np.any(np.diff(pt) < 0):
                o = np.argsort(pt, kind="stable")     # keeps write order on ties
                pt, pv = pt[o], pv[o]
            # last point written at each grid timestamp
            j = np.searchsorted(pt, ts, side="right") - 1
            ok = j >= 0
            ok[ok] = pt[j[ok]] == ts[ok]
            vals[ok] = pv[j[ok]]
        if is_clock:
            clock = vals
            continue
        kept.append(name)
        rows.append(vals)
    mat = np.vstack(rows) if rows else np.zeros((0, ts.size))
    return ts, clock, kept, mat


def window_dims(span_s: float, n_act: int) -> Dict[str, float]:
    return {"span_s": float(span_s), "n_active": int(n_act)}


def emit_window(ctx: Context, system: str, entity: str, name: str, value: float,
                dims: Dict[str, float], inputs: List[str]) -> None:
    ctx.store.add_derived(DerivedMetric(
        name=name, value=float(value), ts=ctx.now, system=system, entity=entity,
        window_s=int(ctx.window_s), kind=MetricKind.GAUGE, inputs=list(inputs),
        dims=dict(dims)))


def nan_cv(x: np.ndarray) -> float:
    """std/mean over finite values; 0 when the mean is ~0 (v1 convention)."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return math.nan
    m = float(x.mean())
    if abs(m) < 1e-9:
        return 0.0
    return float(x.std() / m)


def row_percentile(x: np.ndarray, fin: np.ndarray, q: float) -> np.ndarray:
    """Per-row percentile over the finite entries (numpy's 'linear' rule);
    NaN for a row with none. One sort for all rows."""
    k, n = x.shape
    cnt = fin.sum(axis=1)
    xs = np.sort(np.where(fin, x, np.inf), axis=1)
    pos = (q / 100.0) * np.maximum(cnt - 1, 0)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, np.maximum(cnt - 1, 0))
    r = np.arange(k)
    a, b = xs[r, lo], xs[r, hi]
    frac = pos - lo
    with np.errstate(invalid="ignore"):          # inf - inf on all-NaN rows
        out = np.where(frac > 0, a + frac * (b - a), a)
    return np.where(cnt > 0, out, math.nan)


def window_stats(vals: np.ndarray, dur: np.ndarray, is_counter: np.ndarray,
                 dt_now: float) -> Dict[str, np.ndarray]:
    """sum/mean/p95/max/cv per row of a [k, n] grid matrix, NaN skipped.

    Counter rows are rescaled to the current tick (x = v·Δt_now/Δt_tick):
    sum is the true total, mean the time-weighted rate per current tick, and
    p95 / max / cv are taken over x. Gauge rows use the values as they are.
    A row with no finite value gets NaN everywhere."""
    fin = np.isfinite(vals)
    cnt = fin.sum(axis=1)
    v0 = np.where(fin, vals, 0.0)
    x = np.where(is_counter[:, None], vals * (float(dt_now) / dur)[None, :], vals)
    x0 = np.where(fin, x, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        total = v0.sum(axis=1)
        wdur = (fin * dur[None, :]).sum(axis=1)
        mean = np.where(is_counter, total / wdur * dt_now, total / cnt)
        xm = x0.sum(axis=1) / cnt
        xsd = np.sqrt((np.where(fin, x - xm[:, None], 0.0) ** 2).sum(axis=1) / cnt)
        cv = np.where(np.abs(xm) < 1e-9, 0.0, xsd / xm)
    mx = np.where(fin, x, -np.inf).max(axis=1) if vals.shape[1] else np.full(len(cnt), -np.inf)
    none = cnt == 0
    out = {"sum": total, "mean": mean, "p95": row_percentile(x, fin, 95.0),
           "max": mx, "cv": cv}
    for key in out:
        out[key] = np.where(none, math.nan, out[key])
    return out


class AggregationEngine(Engine):
    name = "derived.aggregation"
    layer = "derived"
    consumes = ["l3.*", "l4.*", "http.*", "tls.*", "dns.*", "probe.*", "act.events"]
    produces = ["derived.<metric>.{sum,mean,p95,max,cv}"]
    description = ("6-h wall-clock window sum/mean/p95/max/cv over configured raw metrics "
                   "(zero-filled counters, NaN gauges; dims span_s, n_active).")
    interval = 1

    DEFAULT_TARGETS = [
        "l3.bytes_total", "l4.flows", "l4.bytes_up", "l4.bytes_down",
        "l4.distinct_peers", "l4.retransmit_rate", "l4.rtt_ms_avg",
        "http.requests", "http.latency_ms_avg", "http.resp_bytes_avg",
        "tls.handshakes", "dns.queries", "probe.rtt_ms",
    ]
    STATS = ("sum", "mean", "p95", "max", "cv")

    def __init__(self, targets: Optional[List[str]] = None, span_s: float = SPAN_S,
                 kinds: Optional[Dict[str, str]] = None, **p):
        super().__init__(**p)
        self.targets = list(targets or self.DEFAULT_TARGETS)
        self.span_s = float(span_s)
        self.kinds = {t: (kinds or {}).get(t, metric_kind(t)) for t in self.targets}

    def run(self, ctx: Context, observations=None) -> int:
        store, now, dt = ctx.store, ctx.now, float(ctx.window_s)
        ensure_retention(store)
        n = 0
        for system in store.systems():
            for entity in store.entities(system):
                if not recently_active(store, system, entity, now, min(self.span_s, ACTIVE_HORIZON_S)):
                    continue
                n += self._entity(ctx, system, entity, dt)
        return n

    def _entity(self, ctx: Context, system: str, entity: str, dt: float) -> int:
        span = self.span_s
        # a channel the entity never used in the span stays unwritten
        ts, clock, names, mat = grid_many(ctx.store, system, entity, self.targets, self.kinds,
                                          ctx.now, span, dt)
        if ts.size == 0 or not names:
            return 0
        dur = tick_durations(ts, dt)
        is_c = np.array([self.kinds[t] == "counter" for t in names])
        dims = window_dims(span, int(active_mask(clock, mat[is_c]).sum()))
        stats = window_stats(mat, dur, is_c, dt)
        n = 0
        for i, target in enumerate(names):
            if not math.isfinite(stats["sum"][i]):
                continue                     # an all-NaN gauge: nothing measured
            for stat in self.STATS:
                emit_window(ctx, system, entity, f"derived.{target}.{stat}", stats[stat][i],
                            dims, [target])
                n += 1
        return n
