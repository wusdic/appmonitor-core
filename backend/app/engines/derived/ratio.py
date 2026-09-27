"""Ratio engine — composed ratios that only make sense across two raw metrics.

Each ratio is declared as (name, numerator, denominator). Adding a new
composed indicator is a one-line data change, not new code.

v2 (D1) fixes two ways v1 manufactured evidence out of nothing:

* Freshness. v1 read `latest_raw` whatever its age and re-stamped it with
  ts=now, so an entity that went silent kept "emitting" its last error rate
  forever. A ratio is now written only when numerator AND denominator were
  written this tick (`fresh.all_fresh`); otherwise nothing is written and the
  absence reaches library 3 as absence.
* Zero denominators. v1 wrote 0.0 for x/0, i.e. "no errors" when there were
  no requests. A ratio of nothing is undefined, so it is simply not written.

Every ratio also writes `derived.<name>.n`, its denominator (exposure), so
downstream Beta-Binomial predictives know how much evidence a value carries:
2 errors in 4 requests and 500 in 1000 are not the same observation.
"""
from __future__ import annotations

import math
from typing import List, Tuple

from ...core.engine import Context, Engine
from ...models.schema import DerivedMetric, MetricKind
from .fresh import all_fresh

# (out_name, numerator, denominator, exposure)
#   exposure == denominator for a true num/den ratio. A passthrough (the raw
#   metric is already a fraction) has no denominator of its own: its exposure
#   is the count the fraction was taken over.
RULES: List[Tuple[str, str, str, str]] = [
    ("derived.http_error_rate", "http.status_4xx", "http.requests", "http.requests"),
    ("derived.http_5xx_rate", "http.status_5xx", "http.requests", "http.requests"),
    ("derived.http_success_rate", "http.status_2xx", "http.requests", "http.requests"),
    ("derived.upload_dominance", "l4.bytes_up", "l4.bytes_down", "l4.bytes_down"),
    ("derived.bytes_per_flow", "l3.bytes_total", "l4.flows", "l4.flows"),
    ("derived.req_per_peer", "http.requests", "l4.distinct_peers", "l4.distinct_peers"),
    ("derived.dns_fail_rate", "dns.nxdomain_ratio", "", "dns.queries"),  # passthrough
]

_DEN_EPS = 1e-9


class RatioEngine(Engine):
    name = "derived.ratio"
    layer = "derived"
    consumes = ["http.*", "l4.*", "l3.*", "dns.*"]
    produces = [n for rule in RULES for n in (rule[0], rule[0] + ".n")]
    description = ("Cross-metric ratios (error rate, upload dominance, bytes-per-flow, ...) "
                   "from fresh inputs only, each with its exposure .n.")

    RULES = RULES

    def run(self, ctx: Context, observations=None) -> int:
        store, now = ctx.store, ctx.now
        ws = int(ctx.window_s)
        n = 0
        for system in store.systems():
            # raw engines touch last_seen only when they wrote real data this
            # tick, so an entity with last_seen < now has nothing fresh to read
            for entity in store.entities_active(system, now):
                for out_name, num, den, expo in self.RULES:
                    names = [num, den] if den else [num, expo]
                    vals = all_fresh(store, system, entity, names, now)
                    if vals is None:
                        continue
                    if den:
                        nv, dv = vals
                        if not dv > _DEN_EPS:           # x/0 is undefined: no emission
                            continue
                        value, n_expo = nv / dv, dv
                    else:
                        value, n_expo = vals
                        if not n_expo > 0:              # fraction of zero items
                            continue
                    if not math.isfinite(value):
                        continue
                    store.add_derived(DerivedMetric(
                        name=out_name, value=float(value), ts=now, system=system,
                        entity=entity, window_s=ws, kind=MetricKind.RATE, inputs=names))
                    store.add_derived(DerivedMetric(
                        name=out_name + ".n", value=float(n_expo), ts=now, system=system,
                        entity=entity, window_s=ws, kind=MetricKind.COUNTER,
                        inputs=[expo]))
                    n += 2
        return n
