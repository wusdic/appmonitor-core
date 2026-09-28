"""Pipeline orchestrator.

Owns the store + engine registry and runs one processing *tick*:
    observations --> raw engines --> derived engines
                 --> behaviour engines --> signature engines

Raw engines receive the tick's observations; every other layer reads what the
previous layer wrote to the store. The orchestrator knows nothing about any
specific engine — it just runs the registry in layer order — so adding an
engine anywhere is a registration change, not an orchestration change.

v2: `run_tick(..., dt)` sets ctx.window_s to the tick's real Δt (warm-up runs
at 900 s, live at the generator's window), the pipeline's config (contract I)
is handed to every engine, and a compact `ops.engine_health` record is written
at each system's `__system__` pseudo-entity so consumers can tell "my
producer failed this tick" from "nothing happened".
"""
from __future__ import annotations

import gc
import time
from typing import Any, Dict, List, Optional, Tuple

from ..core.engine import Context, Engine, Registry, default_config
from ..core.store import MetricStore
from ..models.schema import SYSTEM_ENTITY, DerivedMetric, MetricKind, Observation

# Cyclic-GC thresholds for the pipeline process (perf, docs/lib3/integration.md
# §9). The store and the lib-3 models are a large, mostly long-lived heap
# (~0.9 M tracked objects on pack A) that grows steadily, so at the
# interpreter defaults (700, 10, 10) a full collection runs every few ticks
# and scans all of it: ~8 % of pack A's wall time, in 0.3-1.3 s pauses that
# set the per-tick p95. Almost all garbage here is freed by reference
# counting; the cyclic collector only changes WHEN unreachable cycles are
# reclaimed, never a computed value.
GC_THRESHOLDS: Tuple[int, int, int] = (10000, 20, 50)
_GC_DEFAULTS = (700, 10, 10)


def configure_gc(thresholds: Tuple[int, int, int] = GC_THRESHOLDS) -> bool:
    """Raise the cyclic-GC thresholds unless the process already chose its
    own (anything but the interpreter defaults). Returns True if it set them."""
    if tuple(gc.get_threshold()) != _GC_DEFAULTS:
        return False
    gc.set_threshold(*thresholds)
    return True


class Pipeline:
    def __init__(self, store: MetricStore, registry: Registry, window_s: int = 60,
                 config: Optional[Dict[str, Any]] = None):
        configure_gc()
        self.store = store
        self.registry = registry
        self.window_s = window_s
        # merged with the contract-I defaults once, then shared by every tick
        self.config: Dict[str, Any] = default_config(config)
        self.tick_count = 0
        self.last_tick_stats: Dict[str, int] = {}

    def run_tick(self, observations: List[Observation], now: Optional[float] = None,
                 training: bool = False, dt: Optional[float] = None) -> Dict[str, int]:
        now = now if now is not None else time.time()
        window_s = self.window_s if dt is None else dt
        ctx = Context(store=self.store, now=now, window_s=window_s, training=training,
                      config=self.config)
        for obs in observations:
            self.store.add_observation(obs)
        stats: Dict[str, int] = {}
        for engine in self.registry.ordered():
            obs_arg = observations if engine.layer == "raw" else None
            stats[engine.name] = engine.safe_run(ctx, obs_arg)
        self._write_engine_health(ctx)
        self.tick_count += 1
        self.last_tick_stats = stats
        return stats

    def _write_engine_health(self, ctx: Context) -> None:
        """One small dict per system per tick: which engines failed at this
        tick (with the error line). The full per-engine records live in
        store.health(); this series is the time-indexed view."""
        engines = self.registry.ordered()
        errors = {e.name: e.last_error[:200] for e in engines
                  if e.last_error_ts is not None and e.last_error_ts == ctx.now}
        value = {"n_engines": len(engines), "n_errors": len(errors), "errors": errors}
        for system in self.store.systems():
            self.store.add_derived(DerivedMetric(
                name="ops.engine_health", value=value, ts=ctx.now, system=system,
                entity=SYSTEM_ENTITY, window_s=ctx.window_s, kind=MetricKind.CATEGORICAL))

    def engine_info(self) -> List[dict]:
        return [e.info() for e in self.registry.ordered()]
