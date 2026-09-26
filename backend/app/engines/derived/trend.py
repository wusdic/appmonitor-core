"""Trend engine — direction & smoothing over time for key metrics.

EWMA (smoothed level), slope (trend direction/strength) and a simple
change-point score (recent mean vs. prior mean, in robust-z units). These feed
the drift detector and give the UI a "rising/falling" read per metric.
"""
from __future__ import annotations

from typing import List, Optional

from ...core.engine import Context, Engine
from ...models.schema import DerivedMetric, MetricKind
from .util import ewma, robust_stats, robust_z, series_values, slope


class TrendEngine(Engine):
    name = "derived.trend"
    layer = "derived"
    consumes = ["l4.*", "http.*", "dns.*", "probe.*"]
    produces = ["derived.<metric>.{ewma,slope,changepoint}"]
    description = "EWMA level, slope and change-point score over configured metrics."

    DEFAULT_TARGETS = [
        "l4.bytes_up", "l4.distinct_peers", "http.requests",
        "http.latency_ms_avg", "dns.queries", "probe.rtt_ms",
    ]

    def __init__(self, targets: Optional[List[str]] = None, window_points: int = 40, **p):
        super().__init__(**p)
        self.targets = targets or self.DEFAULT_TARGETS
        self.window_points = window_points

    def run(self, ctx: Context, observations=None) -> int:
        n = 0
        for system in ctx.store.systems():
            for entity in ctx.store.entities(system):
                for target in self.targets:
                    vals = series_values(ctx.store.raw_series(system, entity, target))[-self.window_points:]
                    if len(vals) < 6:
                        continue
                    split = max(3, len(vals) // 2)
                    prior, recent = vals[:split], vals[split:]
                    med, mad = robust_stats(prior)
                    cp = robust_z(sum(recent) / len(recent), med, mad)
                    for stat, value in [("ewma", ewma(vals)), ("slope", slope(vals)),
                                        ("changepoint", cp)]:
                        ctx.store.add_derived(DerivedMetric(
                            name=f"derived.{target}.{stat}", value=value, ts=ctx.now,
                            system=system, entity=entity, window_s=ctx.window_s,
                            kind=MetricKind.GAUGE, inputs=[target]))
                        n += 1
        return n
