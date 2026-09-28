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

v2 (lib-3, contract I/M): time is wall-clock, not ticks. `ctx.window_s` is the
actual Δt of this tick, engines may declare a wall-clock `period_s` next to
their tick `interval`, and `safe_run` turns failures into observable health
records (and into exceptions under `config['strict']`, which tests and eval
use so that a broken engine can never pass silently).
"""
from __future__ import annotations

import copy
import math
import time
import traceback
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .store import MetricStore

# Contract I defaults. Merged under whatever the caller passes, so an engine
# can always read e.g. ctx.config['tz'] without a fallback of its own.
DEFAULT_CONFIG: Dict[str, Any] = {
    "tz": "Asia/Shanghai",
    "calendar": {"holidays": [], "makeup_workdays": []},
    "ip_classes": [],                      # [{name, cidrs, systems, criticality}]
    "dhcp_scopes": [],
    "sensitive_patterns": [],
    "org_domains": [],
    "alert_budget": {"entity_per_hour": 3, "system_per_day": 20},
    "strict": False,
    "daypart_day_hours": [8, 20],
    "D_min_s": 600,
    # B13 absolute floors per quantity {Q: natural units per horizon}; keys
    # override the engine's built-in defaults one by one (integration R12.2)
    "budget_abs_floor": {},
    # spec v2.1 (docs/lib3/cadence.md D11): 'tick' reproduces v2 exactly
    # (G_h := dt, no Q grain); 'canonical' (the default since M8) is the
    # cadence-invariant grain mode. Engine unit tests default to 'tick'
    # (tests/engines|lib|core/conftest.py), whose maths is cadence-agnostic
    "grain_mode": "canonical",
}


def default_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A fresh (deep-copied) default config with `overrides` applied on top."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if overrides:
        cfg.update(overrides)
    return cfg


@dataclass
class Context:
    """Per-run context handed to every engine. Carries the store, the current
    logical time, and free-form config. Nothing engine-specific lives here so
    the signature stays stable as engines come and go."""

    store: MetricStore
    now: float = field(default_factory=time.time)
    # The actual Δt of this tick in seconds (not a nominal constant): every
    # per-minute rate, e_day and time-based threshold is derived from it.
    window_s: float = 60
    # During warm-up the pipeline runs in training mode: engines still build
    # baselines / fingerprints / sequence models, but detective engines do NOT
    # emit findings (those early, loosely-based alerts would be noise).
    training: bool = False
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # contract I defaults under the caller's keys; the caller's dict is
        # not mutated (the orchestrator shares one config across ticks)
        cfg = self.config or {}
        if any(k not in cfg for k in DEFAULT_CONFIG):
            self.config = default_config(cfg)

    @property
    def dt(self) -> float:
        """Alias of window_s (the tick's Δt in seconds)."""
        return self.window_s

    @property
    def strict(self) -> bool:
        return bool(self.config.get("strict", False))


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
    # Wall-clock cadence (seconds of ctx.now). An engine runs when EITHER its
    # tick interval OR its period has elapsed since its last attempt, so a
    # refit stride means the same thing at 60 s and at 3600 s ticks.
    period_s: Optional[float] = None

    def __init__(self, **params: object) -> None:
        self.params = params
        self.enabled = True
        self.interval = int(params.get("interval", self.__class__.interval))
        ps = params.get("period_s", self.__class__.period_s)
        self.period_s: Optional[float] = float(ps) if ps else None
        self._calls = 0
        self._ticks_since_attempt = math.inf       # first call always runs
        self._last_attempt_ts: Optional[float] = None
        self._entity_bucket: Dict[str, int] = {}
        self.last_run: float = 0.0
        self.last_count: int = 0
        self.last_error: str = ""
        self.error_count: int = 0
        self.last_error_ts: Optional[float] = None
        self.last_traceback: str = ""
        self.last_duration_ms: float = 0.0
        self.runs: int = 0

    # Batch mode: called with the observations of one tick (raw engines) or
    # simply the store (derived/behavior/signature engines).
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:  # noqa: D401
        """Do the engine's work; return count of things produced."""
        raise NotImplementedError

    # ------------------------------------------------------------ scheduling
    def is_due(self, now: float) -> bool:
        """True if the tick interval or the wall-clock period has elapsed
        since the last attempt (pure; `safe_run` does the bookkeeping)."""
        if self._ticks_since_attempt + 1 >= self.interval:
            return True
        if self.period_s and self._last_attempt_ts is not None:
            # a clock that went backwards (new run / replay) also counts
            return now - self._last_attempt_ts >= self.period_s or now < self._last_attempt_ts
        return False

    def entity_due(self, key: Any, now: float, period_s: Optional[float] = None) -> bool:
        """Per-entity refit scheduler: True at most once per `period_s`
        (default self.period_s) for `key`, with a deterministic per-key phase
        crc32(key)/2^32·period so that refits of many entities spread over the
        period instead of all landing on the same tick. The first call for a
        key is always due. Calling it commits: the caller is expected to do
        the refit when it returns True."""
        period = period_s if period_s is not None else self.period_s
        if not period or period <= 0:
            return True
        k = key if isinstance(key, str) else "|".join(map(str, key)) if isinstance(key, tuple) else str(key)
        phase = (zlib.crc32(k.encode("utf-8")) / 4294967296.0) * period
        bucket = math.floor((now - phase) / period)
        if self._entity_bucket.get(k) == bucket:
            return False
        self._entity_bucket[k] = bucket
        return True

    # ------------------------------------------------------------- execution
    def safe_run(self, ctx: Context, observations: Optional[List] = None) -> int:
        if not self.enabled:
            return 0
        self._calls += 1
        if not self.is_due(ctx.now):
            self._ticks_since_attempt += 1
            return 0
        self._ticks_since_attempt = 0
        self._last_attempt_ts = ctx.now
        t0 = time.perf_counter()
        try:
            n = self.run(ctx, observations)
            self.last_duration_ms = (time.perf_counter() - t0) * 1000.0
            self.runs += 1
            self.last_run = ctx.now
            self.last_count = n
            self.last_error = ""
            self._report_health(ctx, ok=True)
            return n
        except Exception as exc:  # engines must not take the pipeline down
            self.last_duration_ms = (time.perf_counter() - t0) * 1000.0
            self.error_count += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_error_ts = ctx.now
            self.last_traceback = _traceback_head(exc)
            self.last_count = 0
            self._report_health(ctx, ok=False)
            if ctx.config.get("strict"):
                raise
            return 0

    def health_record(self, ok: bool = True) -> Dict[str, Any]:
        return {
            "engine": self.name,
            "layer": self.layer,
            "ok": ok,
            "runs": self.runs,
            "last_run": self.last_run,
            "last_count": self.last_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "last_error_ts": self.last_error_ts,
            "traceback": self.last_traceback,
            "duration_ms": round(self.last_duration_ms, 3),
            "interval": self.interval,
            "period_s": self.period_s,
        }

    def _report_health(self, ctx: Context, ok: bool) -> None:
        put = getattr(ctx.store, "put_health", None)
        if put is not None:
            rec = self.health_record(ok)
            rec["ts"] = ctx.now
            put(self.name, rec)

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
            "error_count": self.error_count,
            "last_error_ts": self.last_error_ts,
            "interval": self.interval,
            "period_s": self.period_s,
            "duration_ms": round(self.last_duration_ms, 3),
        }


def _traceback_head(exc: BaseException, max_frames: int = 6, max_chars: int = 2000) -> str:
    """The exception line plus the innermost frames: enough to locate the bug
    in a health record without keeping the whole stack alive."""
    tb = traceback.extract_tb(exc.__traceback__)[-max_frames:]
    lines = [f"{type(exc).__name__}: {exc}"]
    lines += [f"  {f.filename}:{f.lineno} in {f.name}: {f.line or ''}" for f in reversed(tb)]
    return "\n".join(lines)[:max_chars]


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
