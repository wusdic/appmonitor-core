"""Freshness helpers for the derived layer (engines D0, D1, D2).

v1 derived engines read `latest_raw` whatever its age and re-stamped it with
ts=now, so an entity that went silent kept "emitting" its last ratio forever
and idle time vanished from window statistics. These pure helpers encode the
two rules that fix it (architecture §0.3, "absence is data"):

* instant metrics (ratios, entropies, graph) are computed only from inputs
  written THIS tick — `fresh_raw` returns None otherwise;
* window metrics (aggregation, periodicity, trend, duty) are computed on a
  wall-clock grid where a tick with no raw point is a true 0 for counters
  (the entity was silent) and NaN for gauges (nothing was measured).

The grid clock is the entity's `act.events` series, which R2 writes for every
known entity every tick (zero-filled when silent). Where it is absent (older
pipelines, unit tests) the grid falls back to regular steps of `dt_s`.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

CLOCK = "act.events"
COUNTER_KINDS = ("counter", "count", "bytes")


def fresh_raw(store, system: str, entity: str, name: str, now: float) -> Optional[float]:
    """Numeric value of raw `name` written exactly at `now`, else None."""
    v = store.latest_fresh(system, entity, name, now)
    if v is None or not isinstance(v, (int, float)) or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def fresh_raw_obj(store, system: str, entity: str, name: str, now: float):
    """Any-typed (e.g. dict set) raw value written exactly at `now`, else None."""
    return store.latest_fresh(system, entity, name, now)


def all_fresh(store, system: str, entity: str, names, now: float) -> Optional[List[float]]:
    """Values of every name if ALL are fresh, else None (emit nothing)."""
    out = []
    for n in names:
        v = fresh_raw(store, system, entity, n, now)
        if v is None:
            return None
        out.append(v)
    return out


def grid_times(store, system: str, entity: str, now: float, span_s: float,
               dt_s: float) -> np.ndarray:
    """Tick timestamps in (now - span_s, now] for this entity."""
    lo = now - span_s
    tail = store.raw_tail(system, entity, CLOCK, _cap(span_s, dt_s))
    ts = np.array([m.ts for m in tail if lo < m.ts <= now], dtype=np.float64)
    if ts.size:
        return ts
    k = max(1, int(round(span_s / max(dt_s, 1.0))))
    return now - dt_s * np.arange(k - 1, -1, -1, dtype=np.float64)


def grid(store, system: str, entity: str, name: str, now: float, span_s: float,
         kind: str, dt_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """(ts, values) on the wall-clock grid over (now - span_s, now].

    kind in COUNTER_KINDS: a tick with no point is 0 (silence is zero
    activity). Any other kind: missing is NaN.
    """
    ts = grid_times(store, system, entity, now, span_s, dt_s)
    fill = 0.0 if kind in COUNTER_KINDS else math.nan
    vals = np.full(ts.shape, fill, dtype=np.float64)
    if ts.size == 0:
        return ts, vals
    tail = store.raw_tail(system, entity, name, _cap(span_s, dt_s))
    pts = {m.ts: m.value for m in tail
           if ts[0] - 1e-6 <= m.ts <= now and isinstance(m.value, (int, float))}
    for i, t in enumerate(ts):
        v = pts.get(float(t))
        if v is not None:
            vals[i] = float(v)
    return ts, vals


def n_active(store, system: str, entity: str, now: float, span_s: float, dt_s: float) -> int:
    """Number of grid ticks in the span on which the entity had activity."""
    _ts, ev = grid(store, system, entity, CLOCK, now, span_s, "counter", dt_s)
    return int(np.sum(ev > 0))


def _cap(span_s: float, dt_s: float) -> int:
    # enough points for the span at the smallest cadence we support (60 s),
    # bounded so a 24 h window at 60 s stays cheap
    return int(min(4096, max(16, span_s / max(min(dt_s, 60.0), 1.0) + 4)))
