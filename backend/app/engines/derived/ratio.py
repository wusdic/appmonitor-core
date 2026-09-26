"""Ratio engine — composed ratios that only make sense across two raw metrics.

Each ratio is declared as (name, numerator, denominator, guard). Adding a new
composed indicator is a one-line data change, not new code.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from ...core.engine import Context, Engine
from ...models.schema import DerivedMetric, MetricKind


class RatioEngine(Engine):
    name = "derived.ratio"
    layer = "derived"
    consumes = ["http.*", "l4.*", "dns.*"]
    produces = [
        "derived.http_error_rate", "derived.http_5xx_rate",
        "derived.http_success_rate", "derived.upload_dominance",
        "derived.bytes_per_flow", "derived.req_per_peer", "derived.dns_fail_rate",
    ]
    description = "Cross-metric ratios: error rate, upload dominance, bytes-per-flow, etc."

    # (out_name, numerator, denominator)
    RULES: List[Tuple[str, str, str]] = [
        ("derived.http_error_rate", "http.status_4xx", "http.requests"),
        ("derived.http_5xx_rate", "http.status_5xx", "http.requests"),
        ("derived.http_success_rate", "http.status_2xx", "http.requests"),
        ("derived.upload_dominance", "l4.bytes_up", "l4.bytes_down"),
        ("derived.bytes_per_flow", "l3.bytes_total", "l4.flows"),
        ("derived.req_per_peer", "http.requests", "l4.distinct_peers"),
        ("derived.dns_fail_rate", "dns.nxdomain_ratio", ""),  # passthrough copy
    ]

    def _latest(self, ctx: Context, system: str, entity: str, name: str):
        m = ctx.store.latest_raw(system, entity, name)
        return float(m.value) if m and isinstance(m.value, (int, float)) else None

    def run(self, ctx: Context, observations=None) -> int:
        n = 0
        for system in ctx.store.systems():
            for entity in ctx.store.entities(system):
                for out_name, num, den in self.RULES:
                    nv = self._latest(ctx, system, entity, num)
                    if nv is None:
                        continue
                    if den:
                        dv = self._latest(ctx, system, entity, den)
                        if dv is None:
                            continue
                        value = nv / dv if dv > 1e-9 else 0.0
                        inputs = [num, den]
                    else:
                        value = nv
                        inputs = [num]
                    ctx.store.add_derived(DerivedMetric(
                        name=out_name, value=value, ts=ctx.now, system=system,
                        entity=entity, window_s=ctx.window_s, kind=MetricKind.RATE,
                        inputs=inputs))
                    n += 1
        return n
