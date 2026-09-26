"""Engine framework.

Every capability in the platform is an *engine*: a small, focused unit that
declares what it consumes and produces and exposes one `run` method. Engines
are registered in a `Registry` and wired by a pipeline. They share only the
`MetricStore` and a `Context`, never each other, so any engine can be added,
removed, or replaced independently — the low-coupling requirement.

An engine belongs to exactly one library layer:
    raw       -> 原始指标库
    derived   -> 次生指标库
    behavior  -> 行为库
    signature -> 行为特征库
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .store import MetricStore


@dataclass
class Context:
    """Per-run context handed to every engine. Carries the store, the current
    logical time, and free-form config. Nothing engine-specific lives here so
    the signature stays stable as engines come and go."""

    store: MetricStore
    now: float = field(default_factory=time.time)
    window_s: int = 60
    # During warm-up the pipeline runs in training mode: engines still build
    # baselines / fingerprints / sequence models, but detective engines do NOT
    # emit findings (those early, loosely-based alerts would be noise).
    training: bool = False
    config: Dict[str, object] = field(default_factory=dict)


class Engine:
    """Base class. Subclasses set the class attributes and implement `run`."""

    name: str = "engine"
    layer: str = "raw"                     # raw | derived | behavior | signature
    consumes: List[str] = []               # metric-name prefixes it reads
    produces: List[str] = []               # metric-name prefixes it writes
    description: str = ""

    # Run cadence: an engine with interval=k does real work only every k-th
    # invocation (heavy ML engines stride so a tick stays cheap). Light engines
    # keep interval=1. Cadence is per-engine state, so it changes nothing about
    # how engines couple — the orchestrator still just calls them each tick.
    interval: int = 1

    def __init__(self, **params: object) -> None:
        self.params = params
        self.enabled = True
        self.interval = int(params.get("interval", self.__class__.interval))
        self._calls = 0
        self.last_run: float = 0.0
        self.last_count: int = 0
        self.last_error: str = ""

    # Batch mode: called with the observations of one tick (raw engines) or
    # simply the store (derived/behavior/signature engines).
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:  # noqa: D401
        """Do the engine's work; return count of things produced."""
        raise NotImplementedError

    def safe_run(self, ctx: Context, observations: Optional[List] = None) -> int:
        if not self.enabled:
            return 0
        self._calls += 1
        if self.interval > 1 and (self._calls - 1) % self.interval != 0:
            return 0
        try:
            n = self.run(ctx, observations)
            self.last_run = ctx.now
            self.last_count = n
            self.last_error = ""
            return n
        except Exception as exc:  # engines must not take the pipeline down
            self.last_error = f"{type(exc).__name__}: {exc}"
            return 0

    def info(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "layer": self.layer,
            "consumes": self.consumes,
            "produces": self.produces,
            "description": self.description,
            "enabled": self.enabled,
            "last_run": self.last_run,
            "last_count": self.last_count,
            "last_error": self.last_error,
        }


class Registry:
    """Holds engine instances grouped by layer and preserves declaration
    order within a layer (raw -> derived -> behavior -> signature)."""

    LAYER_ORDER = ["raw", "derived", "behavior", "signature"]

    def __init__(self) -> None:
        self._engines: List[Engine] = []

    def register(self, engine: Engine) -> Engine:
        self._engines.append(engine)
        return engine

    def add(self, *engines: Engine) -> None:
        for e in engines:
            self.register(e)

    def by_layer(self, layer: str) -> List[Engine]:
        return [e for e in self._engines if e.layer == layer]

    def ordered(self) -> List[Engine]:
        out: List[Engine] = []
        for layer in self.LAYER_ORDER:
            out.extend(self.by_layer(layer))
        return out

    def all(self) -> List[Engine]:
        return list(self._engines)

    def get(self, name: str) -> Optional[Engine]:
        return next((e for e in self._engines if e.name == name), None)


# Small helpers reused across engines --------------------------------------- #
def emit_counter(bucket: Dict[str, float], key: str, amount: float = 1.0) -> None:
    bucket[key] = bucket.get(key, 0.0) + amount
