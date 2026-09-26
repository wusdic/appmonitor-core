"""Aggregation engine — windowed statistics over any numeric raw metric.

Turns per-tick raw samples into windowed summaries (sum/mean/p95/max/cv) so
downstream behaviour engines work on stable features rather than noisy single
ticks. Data-driven: give it a list of raw metric names and the stats you want.
"""
from __future__ import annotations

from typing import List, Optional

from ...core.engine import Context, Engine
from ...models.schema import DerivedMetric, MetricKind
from .util import coefficient_of_variation, percentile, series_values


class AggregationEngine(Engine):
    name = "derived.aggregation"
    layer = "derived"
    consumes = ["l3.*", "l4.*", "http.*", "tls.*", "dns.*", "probe.*"]
    produces = ["derived.<metric>.{sum,mean,p95,max,cv}"]
    description = "Windowed sum/mean/p95/max/coefficient-of-variation over configured raw metrics."

    DEFAULT_TARGETS = [
        "l3.bytes_total", "l4.flows", "l4.bytes_up", "l4.bytes_down",
        "l4.distinct_peers", "l4.retransmit_rate", "l4.rtt_ms_avg",
        "http.requests", "http.latency_ms_avg", "http.resp_bytes_avg",
        "tls.handshakes", "dns.queries", "probe.rtt_ms",
    ]

    def __init__(self, targets: Optional[List[str]] = None, window_points: int = 30, **p):
        super().__init__(**p)
        self.targets = targets or self.DEFAULT_TARGETS
        self.window_points = window_points

    def run(self, ctx: Context, observations=None) -> int:
        n = 0
        for system in ctx.store.systems():
            for entity in ctx.store.entities(system):
                for target in self.targets:
                    series = ctx.store.raw_series(system, entity, target)
                    vals = series_values(series)[-self.window_points:]
                    if not vals:
                        continue
                    stats = {
                        "sum": sum(vals),
                        "mean": sum(vals) / len(vals),
                        "p95": percentile(vals, 95),
                        "max": max(vals),
                        "cv": coefficient_of_variation(vals),
                    }
                    for stat, value in stats.items():
                        ctx.store.add_derived(DerivedMetric(
                            name=f"derived.{target}.{stat}", value=value, ts=ctx.now,
                            system=system, entity=entity, window_s=ctx.window_s,
                            kind=MetricKind.GAUGE, inputs=[target]))
                        n += 1
        return n
