"""Pipeline orchestrator.

Owns the store + engine registry and runs one processing *tick*:
    observations --> raw engines --> derived engines
                 --> behaviour engines --> signature engines

Raw engines receive the tick's observations; every other layer reads what the
previous layer wrote to the store. The orchestrator knows nothing about any
specific engine — it just runs the registry in layer order — so adding an
engine anywhere is a registration change, not an orchestration change.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from ..core.engine import Context, Engine, Registry
from ..core.store import MetricStore
from ..models.schema import Observation


class Pipeline:
    def __init__(self, store: MetricStore, registry: Registry, window_s: int = 60):
        self.store = store
        self.registry = registry
        self.window_s = window_s
        self.tick_count = 0
        self.last_tick_stats: Dict[str, int] = {}

    def run_tick(self, observations: List[Observation], now: Optional[float] = None,
                 training: bool = False) -> Dict[str, int]:
        now = now if now is not None else time.time()
        ctx = Context(store=self.store, now=now, window_s=self.window_s, training=training)
        for obs in observations:
            self.store.add_observation(obs)
        stats: Dict[str, int] = {}
        for engine in self.registry.ordered():
            obs_arg = observations if engine.layer == "raw" else None
            stats[engine.name] = engine.safe_run(ctx, obs_arg)
        self.tick_count += 1
        self.last_tick_stats = stats
        return stats

    def engine_info(self) -> List[dict]:
        return [e.info() for e in self.registry.ordered()]
