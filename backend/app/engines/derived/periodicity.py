"""Periodicity engine — detects regular/automated timing (beaconing).

Human interaction is bursty and irregular; automation (schedulers, polling
clients, C2 beacons, health checks) is periodic. This engine scores how
periodic an entity's request-count series is via autocorrelation, plus a
regularity score from the coefficient of variation of the series.
"""
from __future__ import annotations

from ...core.engine import Context, Engine
from ...models.schema import DerivedMetric, MetricKind
from .util import autocorr_peak, coefficient_of_variation, series_values


class PeriodicityEngine(Engine):
    name = "derived.periodicity"
    layer = "derived"
    consumes = ["http.requests", "l4.flows", "dns.queries"]
    produces = ["derived.periodicity_score", "derived.beacon_lag", "derived.timing_regularity"]
    description = "Autocorrelation-based periodicity/beaconing score over event-count series."

    TARGETS = ["http.requests", "l4.flows", "dns.queries"]

    def __init__(self, window_points: int = 12, **p):
        super().__init__(**p)
        self.window_points = window_points

    def run(self, ctx: Context, observations=None) -> int:
        n = 0
        for system in ctx.store.systems():
            for entity in ctx.store.entities(system):
                # Use a RECENT window so the score reflects the entity's current
                # rhythm, not rhythm averaged over its whole (possibly changed)
                # history — a host that just started beaconing must read as
                # regular now even if it was bursty before.
                # periodicity_score (autocorr) and timing_regularity (1-CV) are
                # decoupled: a *constant* cadence has zero autocorr but is
                # maximally regular, so choosing one target by autocorr must not
                # decide the regularity of the other.
                best_score, best_lag = 0.0, 0
                primary_src, primary_vals = "", []
                for target in self.TARGETS:
                    vals = series_values(ctx.store.raw_series(system, entity, target))[-self.window_points:]
                    if len(vals) < 8:
                        continue
                    lag, score = autocorr_peak(vals)
                    if score > best_score:
                        best_score, best_lag = score, lag
                    if not primary_src and sum(vals) > 0:
                        primary_src, primary_vals = target, vals   # dominant activity
                if not primary_src:
                    continue
                cv = coefficient_of_variation(primary_vals)
                regularity = max(0.0, 1.0 - min(cv, 1.0))   # constant => 1.0
                for name, value, kind in [
                    ("derived.periodicity_score", best_score, MetricKind.GAUGE),
                    ("derived.beacon_lag", float(best_lag), MetricKind.GAUGE),
                    ("derived.timing_regularity", regularity, MetricKind.GAUGE),
                ]:
                    ctx.store.add_derived(DerivedMetric(
                        name=name, value=value, ts=ctx.now, system=system,
                        entity=entity, window_s=ctx.window_s, kind=kind,
                        inputs=[primary_src]))
                    n += 1
        return n
