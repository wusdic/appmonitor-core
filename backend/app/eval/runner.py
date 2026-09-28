"""Evaluation runner: one (pack, seed) through the full pipeline (docs/lib3/eval.md).

Why a dedicated runner instead of `build.Runtime`: the gates need things the
runtime never keeps. Events are pruned after 30 d, behavior.p after 1 d and
incidents are updated in place, so a post-hoc look at the store cannot say
*when* an incident first reached HIGH, what the calibrated p-values of a clean
control tick were three days ago, or what the baseline anchor looked like just
before an attack started. The runner therefore drives the pipeline tick by tick
(warm-up phases with training=True, the scenario phase with training=False,
always `now = gen.vt` and the phase's real Δt) and records, per tick:

  * per-layer and per-engine CPU time (engines are wrapped, the orchestrator is
    untouched), engine exceptions (strict mode) and stale lib-3 series;
  * a compact per-entity series (behavior.p, e_day, p_family, alarm,
    acc_alarm, risk, active) for calibration / FAR / risk gates;
  * the severity history of every incident that changed this tick;
  * version changes of model.class / model.link / model.control and
    model.baseline snapshots around each malicious window (poisoning gate).

What comes back is a `RunResult` of plain data (dicts, lists, numpy arrays):
it pickles across process boundaries (scripts/evaluate.py runs packs in a
ProcessPoolExecutor) and metrics.py can score it without a live store.

The harness is allowed to *schedule* its snapshots from gen.truth (it has to
know when an attack starts to photograph the anchor before it), but it never
feeds truth into the pipeline: only the metrics read it, as eval.md requires.

Integration seams that other developers own are isolated in one small function
each, so a signature change there is a one-line fix here:
`default_registry_factory` (build.build_registry), `_make_generator`
(generator.TrafficGenerator v2), `_gen_step` (step(dt, live)),
`_resolve_pack` (packs.get_pack) and `_phase_plan` (Pack.phases).
"""
from __future__ import annotations

import copy
import inspect
import math
import time
import traceback
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..core.engine import Registry, default_config
from ..core.store import MetricStore
from ..models.schema import (ORG, SYSTEM_ENTITY, BehaviorEvent, Incident, Label, Severity,
                             is_pseudo_entity)
from ..pipeline.orchestrator import Pipeline

try:  # the detector / family order is part of the store contract (J)
    from ..engines.behavior.lib.detectors import DETECTORS, FAMILIES
except Exception:  # pragma: no cover - lib is present in every checkout
    DETECTORS, FAMILIES = [], []

N_DET = len(DETECTORS)
N_FAM = len(FAMILIES)
LAYERS = ("raw", "derived", "behavior", "signature")

# Series the collector reads each scenario tick (contract B names).
P_VEC = "behavior.p"
P_FAMILY = "behavior.p_family"
E_DAY = "behavior.e_day"
ALARM = "behavior.alarm"
ACC_ALARM = "behavior.acc_alarm"
RISK = "behavior.risk"
ACTIVE = "feature.active"
FEATURE_NAT = "feature.nat"

# Model names whose version history the gates need.
M_CLASS = "model.class"
M_LINK = "model.link"
M_CONTROL = "model.control"
M_BASELINE = "model.baseline"
M_FEEDBACK = "model.feedback"

# Profile extra keys kept in the snapshot (contract G).
PROFILE_EXTRA_KEYS = ("identity", "continuity", "portrait", "regime", "peer_group", "risk",
                      "attribution", "feedback", "calibration", "class_monitor", "maturity")


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    """Everything metrics.py needs from one (pack, seed) run, as plain data.

    Every field has a default so tests can hand-build a partial result.
    Times are epoch seconds of the generator's virtual clock.
    """

    pack: str = ""
    seed: int = 0
    strict: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    phases: List[Dict[str, Any]] = field(default_factory=list)
    # scenario phase: (t_first_tick - dt, t_last_tick), its Δt and tick stamps
    scenario_window: Tuple[float, float] = (0.0, 0.0)
    scenario_dt: float = 900.0
    tick_ts: np.ndarray = field(default_factory=lambda: np.empty(0))
    truth: List[Dict[str, Any]] = field(default_factory=list)
    personas: Dict[str, Dict[str, Any]] = field(default_factory=dict)   # 'sys|ip' -> meta
    fixtures: Dict[str, Any] = field(default_factory=dict)
    entity_first_seen: Dict[str, float] = field(default_factory=dict)  # 'sys|ip' -> ts
    incidents: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    # 'sys|entity' -> {ts, p[n,D], e_day, p_family[n,F], alarm_path, alarm_sev,
    #                  acc_alarm[n,D], risk, active}
    series: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    models: Dict[str, Any] = field(default_factory=dict)
    holdout: Dict[str, Any] = field(default_factory=dict)             # portrait coverage
    timings: Dict[str, Any] = field(default_factory=dict)
    exceptions: List[Dict[str, Any]] = field(default_factory=list)
    stale: List[Dict[str, Any]] = field(default_factory=list)
    memory: Dict[str, Any] = field(default_factory=dict)
    health: Dict[str, Any] = field(default_factory=dict)
    disabled_engines: List[str] = field(default_factory=list)
    labels_added: int = 0
    aborted: Optional[str] = None
    wall_s: float = 0.0
    store: Optional[MetricStore] = None          # only when keep_store=True

    def __getstate__(self) -> Dict[str, Any]:
        # the live store (locks, engine refs) never crosses a process boundary
        st = dict(self.__dict__)
        st["store"] = None
        return st

    def __setstate__(self, st: Dict[str, Any]) -> None:
        self.__dict__.update(st)


@dataclass
class PhasePlan:
    index: int
    n_ticks: int
    dt: float
    aggregated: bool
    training: bool


class RunAborted(RuntimeError):
    """Raised inside the tick loop to stop a run (engine error in 'abort' mode)."""


# --------------------------------------------------------------------------- #
# Integration seams (one function each)
# --------------------------------------------------------------------------- #
def _call_with_supported(fn: Callable, **kwargs: Any) -> Any:
    """Call fn with the subset of kwargs its signature accepts (all of them
    when it takes **kwargs). Keeps factories free to ignore what they do not
    need (pack, config, seed)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)
    params = sig.parameters.values()
    if any(p.kind == p.VAR_KEYWORD for p in params):
        return fn(**kwargs)
    names = {p.name for p in params}
    return fn(**{k: v for k, v in kwargs.items() if k in names})


def default_registry_factory(pack: Any = None, config: Optional[Dict[str, Any]] = None,
                             seed: int = 0) -> Registry:
    """The production engine registry. The only place the runner touches
    build.py; its call signature is expected to change when the lib-3 engines
    are integrated, so adapt it here and nowhere else."""
    from ..pipeline import build
    fn = build.build_registry
    try:
        return _call_with_supported(fn, pack=pack, config=config, seed=seed)
    except TypeError:
        sig_store, composite_rules = build.load_signatures()
        return fn(sig_store, composite_rules)


def _resolve_pack(pack: Any) -> Any:
    if isinstance(pack, str):
        from . import packs  # written concurrently; imported lazily on purpose
        return packs.get_pack(pack)
    return pack


def _make_generator(pack: Any, seed: int, generator_factory: Optional[Callable]) -> Any:
    if generator_factory is not None:
        return _call_with_supported(generator_factory, pack=pack, seed=seed)
    from ..pipeline.generator import TrafficGenerator
    try:
        return TrafficGenerator(seed=seed, pack=pack)
    except TypeError:  # v1 generator (no packs): still usable for a smoke run
        return TrafficGenerator(seed=seed)


def _gen_step(gen: Any, dt: float, live: bool, aggregated: bool) -> List[Any]:
    """One generator tick. `aggregated` is passed only if step() accepts it
    (the v2 generator may derive it from dt >= 900 s itself)."""
    step = gen.step
    try:
        names = set(inspect.signature(step).parameters)
    except (TypeError, ValueError):
        names = set()
    if "aggregated" in names:
        return step(dt=dt, live=live, aggregated=aggregated)
    return step(dt=dt, live=live)


def _phase_fields(ph: Any) -> Tuple[int, float, bool, Optional[bool]]:
    if isinstance(ph, dict):
        return (int(ph["n_ticks"]), float(ph["dt"]), bool(ph.get("aggregated", False)),
                ph.get("training"))
    if hasattr(ph, "n_ticks"):
        return (int(ph.n_ticks), float(ph.dt), bool(getattr(ph, "aggregated", False)),
                getattr(ph, "training", None))
    seq = tuple(ph)
    training = seq[3] if len(seq) > 3 else None
    if isinstance(training, str):
        training = training.lower() in ("warmup", "warm-up", "train", "training")
    return int(seq[0]), float(seq[1]), bool(seq[2]) if len(seq) > 2 else False, training


def _phase_plan(pack: Any) -> List[PhasePlan]:
    """Pack.phases = [(n_ticks, dt, aggregated)]: every phase but the last is
    warm-up (training=True), the last is the scenario phase, unless a phase
    carries an explicit 4th field / `training` attribute or the pack declares
    `n_warmup_phases`."""
    phases = list(getattr(pack, "phases", None) or [])
    if not phases:
        raise ValueError(f"pack {getattr(pack, 'name', pack)!r} has no phases")
    n_warm = getattr(pack, "n_warmup_phases", None)
    out: List[PhasePlan] = []
    for i, ph in enumerate(phases):
        n, dt, agg, training = _phase_fields(ph)
        if training is None:
            training = (i < int(n_warm)) if n_warm is not None else (i < len(phases) - 1)
        out.append(PhasePlan(index=i, n_ticks=n, dt=dt, aggregated=agg, training=bool(training)))
    if all(p.training for p in out):
        out[-1].training = False
    return out


def _pack_config(pack: Any, strict: bool) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    extra = getattr(pack, "config", None)
    if isinstance(extra, dict):
        cfg.update(copy.deepcopy(extra))
    tz = getattr(pack, "tz", None)
    if tz:
        cfg["tz"] = tz
    cal = getattr(pack, "calendar", None)
    if cal is not None:
        cfg["calendar"] = _jsonable(cal)
    cfg["strict"] = bool(strict)
    # contract I defaults made explicit (as Context would fill them each tick),
    # so the run records e.g. the grain mode it ran in (metrics gate 7 splits)
    return default_config(cfg)


# --------------------------------------------------------------------------- #
# Plain-data conversion
# --------------------------------------------------------------------------- #
def _jsonable(obj: Any, depth: int = 0, max_depth: int = 8) -> Any:
    """Deep copy into JSON-able plain data (numpy -> lists, enums -> values,
    dataclasses/objects -> dicts of their public fields)."""
    if depth > max_depth:
        return repr(obj)[:200]
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Severity):
        return obj.value
    if hasattr(obj, "value") and obj.__class__.__module__ == "enum":  # pragma: no cover
        return obj.value
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, depth + 1, max_depth) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        items = sorted(obj, key=repr) if isinstance(obj, (set, frozenset)) else obj
        return [_jsonable(v, depth + 1, max_depth) for v in items]
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _jsonable(getattr(obj, k), depth + 1, max_depth)
                for k in obj.__dataclass_fields__}
    if hasattr(obj, "_asdict"):
        return _jsonable(obj._asdict(), depth + 1, max_depth)
    if hasattr(obj, "__dict__"):
        return {k: _jsonable(v, depth + 1, max_depth) for k, v in vars(obj).items()
                if not k.startswith("_")}
    return repr(obj)[:200]


def _sev(v: Any) -> str:
    if isinstance(v, Severity):
        return v.value
    return str(v or "info").lower()


def incident_to_dict(inc: Incident) -> Dict[str, Any]:
    d = _jsonable(inc)
    d["severity"] = _sev(inc.severity)
    return d


def event_to_dict(ev: BehaviorEvent) -> Dict[str, Any]:
    d = {
        "id": ev.id, "system": ev.system, "entity": ev.entity, "ts": float(ev.ts),
        "kind": ev.kind, "score": _jsonable(ev.score), "severity": _sev(ev.severity),
        "extra": _jsonable(ev.extra), "status": ev.status, "p_value": _jsonable(ev.p_value),
        "e_day": _jsonable(ev.e_day), "axes": list(ev.axes or []),
        "p_by_detector": _jsonable(ev.p_by_detector or {}), "dedupe_key": ev.dedupe_key,
        "incident_id": ev.incident_id, "model_version": _jsonable(ev.model_version),
        "window": _jsonable(ev.window), "description": ev.description,
    }
    return d


# --------------------------------------------------------------------------- #
# Store readers (tolerant of scalar / dict / vector encodings)
# --------------------------------------------------------------------------- #
def _fresh_value(store: MetricStore, s: str, e: str, name: str, now: float) -> Any:
    """Value of `name` written at exactly `now`: a derived/raw point, else a
    vector row (a 1-dim vector collapses to its float)."""
    v = store.latest_fresh(s, e, name, now)
    if v is not None:
        return v
    row = store.vec_at(s, e, name, now)
    if row is None:
        return None
    return float(row[0]) if row.size == 1 else row


def _as_float(v: Any, key: str = "score") -> float:
    if v is None:
        return math.nan
    if isinstance(v, dict):
        for k in (key, "value", "p", "score"):
            if k in v:
                return _as_float(v[k])
        return math.nan
    if isinstance(v, np.ndarray):
        return float(v.reshape(-1)[0]) if v.size else math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


# --------------------------------------------------------------------------- #
# Recorder: timing, exceptions, stale series
# --------------------------------------------------------------------------- #
class _Recorder:
    def __init__(self, registry: Registry, on_error: str) -> None:
        self.on_error = on_error
        self.engines = registry.ordered()
        self.names = [e.name for e in self.engines]
        self.layer_of = {e.name: e.layer for e in self.engines}
        self.idx = {n: i for i, n in enumerate(self.names)}
        self._tick_engine = np.zeros(len(self.names))
        self.engine_rows: List[np.ndarray] = []
        self.tick_rows: List[Tuple[float, float, int, int, float, float, float, float, float,
                                   float, float]] = []
        self.exceptions: List[Dict[str, Any]] = []
        self._instrument()

    def _instrument(self) -> None:
        for eng in self.engines:
            orig = eng.safe_run

            def wrapped(ctx: Any, observations: Any = None, _orig=orig, _eng=eng) -> int:
                t0 = time.perf_counter()
                try:
                    return _orig(ctx, observations)
                except Exception as exc:  # strict mode re-raises out of safe_run
                    self.exceptions.append({
                        "engine": _eng.name, "layer": _eng.layer, "ts": float(ctx.now),
                        "error": f"{type(exc).__name__}: {exc}"[:500],
                        "traceback": "".join(traceback.format_exception(
                            type(exc), exc, exc.__traceback__)[-6:])[-2000:]})
                    if self.on_error == "raise":
                        raise
                    if self.on_error == "abort":
                        raise RunAborted(f"engine {_eng.name} raised: {exc}") from exc
                    return 0
                finally:
                    self._tick_engine[self.idx[_eng.name]] += time.perf_counter() - t0

            eng.safe_run = wrapped  # instance attribute shadows the method

    def begin_tick(self) -> None:
        self._tick_engine[:] = 0.0

    def end_tick(self, now: float, dt: float, phase: int, training: bool, gen_s: float,
                 pipe_s: float, collect_s: float) -> None:
        per = self._tick_engine * 1000.0
        layer_ms = [float(sum(per[i] for i, n in enumerate(self.names)
                              if self.layer_of[n] == layer)) for layer in LAYERS]
        self.tick_rows.append((now, dt, phase, int(training), gen_s * 1000.0, pipe_s * 1000.0,
                               *layer_ms, collect_s * 1000.0))
        self.engine_rows.append(per.astype(np.float32).copy())

    def timings(self) -> Dict[str, Any]:
        cols = ["ts", "dt", "phase", "training", "gen_ms", "pipeline_ms",
                "raw_ms", "derived_ms", "behavior_ms", "signature_ms", "collect_ms"]
        arr = np.asarray(self.tick_rows, dtype=np.float64).reshape(-1, len(cols))
        eng = (np.vstack(self.engine_rows) if self.engine_rows
               else np.zeros((0, len(self.names)), dtype=np.float32))
        return {"columns": cols, "ticks": arr, "engine_names": list(self.names),
                "engine_layers": [self.layer_of[n] for n in self.names], "engine_ms": eng}


class _StaleChecker:
    """A lib-3 series (feature.* / behavior.* vector ring or stored derived
    series) is stale when it has not been written for more than 2 of its own
    periods while its entity is observed this tick and still being processed
    (some other lib-3 series of that entity was written this tick). Its period is the median of
    its last 3 write gaps, floored at the tick Δt, so a 6-hourly refit output
    is not flagged between refits."""

    PREFIXES = ("feature.", "behavior.")
    # written only on some ticks BY CONTRACT, so a gap is not staleness:
    # behavior.alarm only on alarm ticks (B25), behavior.regime on non-normal
    # ticks, transitions and an hourly heartbeat (B28), behavior.degraded only
    # on ticks where a detector ran degraded (contract M, lib/emit causes)
    SPARSE = frozenset({"behavior.alarm", "behavior.regime", "behavior.degraded"})

    def __init__(self) -> None:
        self.found: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._name_cache: Dict[Tuple[str, str], Tuple[Any, List[Tuple[str, str]]]] = {}

    def _names(self, store: MetricStore, s: str, e: str) -> List[Tuple[str, str]]:
        """The entity's lib-3 series, re-scanned only when its name sets grew
        (store.names_signature; the scan is O(derived x vec names))."""
        sig = store.names_signature(s, e)
        hit = self._name_cache.get((s, e))
        if hit is not None and hit[0] == sig:
            return hit[1]
        out = self._scan_names(store, s, e)
        self._name_cache[(s, e)] = (sig, out)
        return out

    def _scan_names(self, store: MetricStore, s: str, e: str) -> List[Tuple[str, str]]:
        vecs = store.vec_names(s, e)
        out = [("vec", v) for v in vecs if v.startswith(self.PREFIXES)]
        vec_cols = {v: set(store.vec_columns(v) or ()) for v in vecs}
        for n in store.derived_names(s, e):
            if not n.startswith(self.PREFIXES) or n in self.SPARSE:
                continue
            if any(n.startswith(v + ".") for v in vecs):
                continue                          # virtual column view of a vector
            if "feature.vec" in vec_cols and n.startswith("feature.") \
                    and n[len("feature."):] in vec_cols["feature.vec"]:
                continue
            out.append(("derived", n))
        return out

    @staticmethod
    def _due_since(store: MetricStore, s: str, e: str, last: float, now: float,
                   period: float) -> bool:
        """A series written once per `period` (an H / Q grain series, spec
        v2.1) is overdue only once the entity has been active for two periods
        since its last write: it is written on the grain's decision ticks
        after an active window, so an entity that wakes up between two
        decision ticks has not missed one yet."""
        ts, A = store.vec_since(s, e, "feature.active", last + 1e-6)
        if not len(ts):
            return True
        a = np.asarray(A, dtype=np.float64).reshape(len(ts), -1)[:, 0]
        on = np.flatnonzero(a >= 0.5)
        if not on.size:
            return False
        return now - float(ts[on[0]]) > 2.0 * period + 1e-6

    def check(self, store: MetricStore, now: float, dt: float) -> None:
        try:
            from ..engines.behavior.lib import m_class
        except Exception:  # pragma: no cover - lib is present in every checkout
            m_class = None
        for s in store.systems():
            # a class key that no longer exists (dissolved role, empty pool)
            # legitimately stops being written
            live = set(m_class.all_class_keys(store, s)) if m_class is not None else None
            keys = store.entities(s) + [p for p in store.pseudo_entities(s)
                                        if p.startswith("class:") and (live is None or p in live)]
            for e in keys:
                names = self._names(store, s, e)
                if not names:
                    continue
                if not e.startswith("class:") and store.last_seen(s, e) != now:
                    continue                      # idle entity: event-driven series may pause
                lw = {n: store.last_write_ts(s, e, n) for _, n in names}
                if not any(v == now for v in lw.values()):
                    continue                      # entity not processed this tick
                for kind, n in names:
                    last = lw[n]
                    if last is None or now - last <= 2.0 * dt + 1e-6:
                        continue
                    if kind == "vec":
                        ts, _ = store.vec_tail(s, e, n, 8)
                    else:
                        ts = np.array([m.ts for m in store.derived_tail(s, e, n, 8)])
                    if len(ts) < 2:
                        continue
                    # recent gaps only: a warm-up cadence must not mask a live stall
                    period = max(float(np.median(np.diff(ts)[-3:])), dt)
                    if period > 1.5 * dt and not self._due_since(store, s, e, last, now, period):
                        continue          # a grain series (spec v2.1) of a re-activated entity
                    if now - last > 2.0 * period + 1e-6:
                        key = (s, e, n)
                        rec = self.found.get(key)
                        if rec is None:
                            self.found[key] = {"system": s, "entity": e, "name": n,
                                               "first_ts": now, "last_write": last,
                                               "period_s": period, "n_ticks": 1}
                        else:
                            rec["n_ticks"] += 1


# --------------------------------------------------------------------------- #
# Collector: per-tick series, incidents, events, model versions
# --------------------------------------------------------------------------- #
class _SeriesBuf:
    __slots__ = ("ts", "p", "e_day", "p_family", "alarm_path", "alarm_sev", "acc_alarm",
                 "risk", "active")

    def __init__(self) -> None:
        self.ts: List[float] = []
        self.p: List[np.ndarray] = []
        self.e_day: List[float] = []
        self.p_family: List[np.ndarray] = []
        self.alarm_path: List[str] = []
        self.alarm_sev: List[str] = []
        self.acc_alarm: List[np.ndarray] = []
        self.risk: List[float] = []
        self.active: List[float] = []

    def to_arrays(self) -> Dict[str, Any]:
        return {
            "ts": np.asarray(self.ts, dtype=np.float64),
            "p": (np.vstack(self.p).astype(np.float32) if self.p
                  else np.zeros((0, N_DET), dtype=np.float32)),
            "e_day": np.asarray(self.e_day, dtype=np.float64),
            "p_family": (np.vstack(self.p_family).astype(np.float32) if self.p_family
                         else np.zeros((0, N_FAM), dtype=np.float32)),
            "alarm_path": list(self.alarm_path),
            "alarm_sev": list(self.alarm_sev),
            "acc_alarm": (np.vstack(self.acc_alarm).astype(np.int8) if self.acc_alarm
                          else np.zeros((0, N_DET), dtype=np.int8)),
            "risk": np.asarray(self.risk, dtype=np.float64),
            "active": np.asarray(self.active, dtype=np.float64),
        }


class _Collector:
    def __init__(self, store: MetricStore) -> None:
        self.store = store
        self.series: Dict[str, _SeriesBuf] = {}
        self.events: Dict[str, BehaviorEvent] = {}
        self.inc_hist: Dict[str, List[Dict[str, Any]]] = {}
        self.inc_last: Dict[str, Tuple[Any, ...]] = {}
        self.open_ids: set = set()
        self.incidents: Dict[str, Incident] = {}
        self.versions: Dict[str, Any] = {}
        self.class_history: List[Dict[str, Any]] = []
        self.link_history: Dict[str, List[Dict[str, Any]]] = {}
        self.control_history: Dict[str, List[Dict[str, Any]]] = {}
        self.baseline_snaps: Dict[str, Dict[str, Any]] = {}
        self.first_seen: Dict[str, float] = {}

    # -- per tick --------------------------------------------------------------
    def on_tick(self, now: float, dt: float, record_series: bool = True) -> None:
        st = self.store
        systems = st.systems()
        for s in systems:
            reals = st.entities(s)
            classes = [p for p in st.pseudo_entities(s) if p.startswith("class:")]
            for e in reals:
                k = f"{s}|{e}"
                if k not in self.first_seen:
                    fs = st.first_seen(s, e)
                    if fs is not None:
                        self.first_seen[k] = float(fs)
            if record_series:
                for e in reals + classes:
                    self._series_row(s, e, now)
            for e in reals + classes:
                self._control(s, e, now)
            self._link(s, now)
        self._class(now)
        self._events(now, dt)
        self._incidents(now)

    def _series_row(self, s: str, e: str, now: float) -> None:
        st = self.store
        buf = self.series.get(f"{s}|{e}")
        if buf is None:
            buf = self.series[f"{s}|{e}"] = _SeriesBuf()
        row = st.vec_at(s, e, P_VEC, now)
        p = (np.asarray(row, dtype=np.float32) if row is not None and row.size == N_DET
             else np.full(N_DET, np.nan, dtype=np.float32))
        fam = np.full(N_FAM, np.nan, dtype=np.float32)
        pf = _fresh_value(st, s, e, P_FAMILY, now)
        if isinstance(pf, dict):
            for i, f in enumerate(FAMILIES):
                if f in pf:
                    fam[i] = _as_float(pf[f], "p")
        elif isinstance(pf, np.ndarray) and pf.size == N_FAM:
            fam[:] = pf
        alarm = _fresh_value(st, s, e, ALARM, now)
        path, sev = "", ""
        if isinstance(alarm, dict) and alarm:
            path = str(alarm.get("path") or "")
            sev = _sev(alarm.get("severity")) if alarm.get("severity") else ""
        acc = np.zeros(N_DET, dtype=np.int8)
        aa = _fresh_value(st, s, e, ACC_ALARM, now)
        if isinstance(aa, dict):
            for d, v in aa.items():
                i = DETECTORS.index(d) if d in DETECTORS else -1
                if i >= 0 and v:
                    acc[i] = 1
        buf.ts.append(now)
        buf.p.append(p)
        buf.e_day.append(_as_float(_fresh_value(st, s, e, E_DAY, now), "e_day"))
        buf.p_family.append(fam)
        buf.alarm_path.append(path)
        buf.alarm_sev.append(sev)
        buf.acc_alarm.append(acc)
        buf.risk.append(_as_float(_fresh_value(st, s, e, RISK, now), "score"))
        buf.active.append(_as_float(_fresh_value(st, s, e, ACTIVE, now)))

    def _events(self, now: float, dt: float) -> None:
        for ev in self.store.events(since=now - 2.0 * dt, limit=10 ** 9):
            if ev.id and ev.id not in self.events:
                self.events[ev.id] = ev

    def _incidents(self, now: float) -> None:
        st = self.store
        changed = {i.id: i for i in st.incidents(since=now)}
        for iid in list(self.open_ids):
            if iid not in changed:
                inc = st.get_incident(iid)
                if inc is not None:
                    changed[iid] = inc
        for iid, inc in changed.items():
            self.incidents[iid] = inc
            sig = (_sev(inc.severity), inc.status, tuple(sorted(inc.axes or ())),
                   tuple(sorted(inc.kinds or ())), round(float(inc.risk or 0.0), 1),
                   len(inc.evidence or ()))
            if self.inc_last.get(iid) != sig:
                self.inc_last[iid] = sig
                self.inc_hist.setdefault(iid, []).append({
                    "ts": now, "severity": sig[0], "status": inc.status,
                    "axes": list(sig[2]), "kinds": list(sig[3]), "risk": float(inc.risk or 0.0),
                    "n_evidence": sig[5]})
            if inc.status == "closed":
                self.open_ids.discard(iid)
            else:
                self.open_ids.add(iid)

    def _version_changed(self, key: str, v: Any) -> bool:
        if v is None:
            return False
        if self.versions.get(key, object()) == v:
            return False
        self.versions[key] = v
        return True

    def _class(self, now: float) -> None:
        v = self.store.model_version(ORG, ORG, M_CLASS)
        if self._version_changed("class", v):
            m = self.store.get_model(ORG, ORG, M_CLASS) or {}
            assign = {k: {"role": _jsonable(a.get("role")), "sub": _jsonable(a.get("sub")),
                          "prob": _as_float(a.get("prob", 1.0)),
                          "pool": _jsonable(a.get("pool")),
                          "static": _jsonable(a.get("static") or [])}
                      for k, a in (m.get("assign") or {}).items() if isinstance(a, dict)}
            self.class_history.append({"ts": now, "version": _jsonable(v), "assign": assign})

    def _link(self, s: str, now: float) -> None:
        v = self.store.model_version(s, SYSTEM_ENTITY, M_LINK)
        if self._version_changed(f"link|{s}", v):
            m = self.store.get_model(s, SYSTEM_ENTITY, M_LINK) or {}
            self.link_history.setdefault(s, []).append(
                {"ts": now, "version": _jsonable(v), "model": _jsonable(m, max_depth=5)})

    def _control(self, s: str, e: str, now: float) -> None:
        v = self.store.model_version(s, e, M_CONTROL)
        if self._version_changed(f"control|{s}|{e}", v):
            m = self.store.get_model(s, e, M_CONTROL) or {}
            rec = {"ts": now, "store_version": _jsonable(v)}
            if isinstance(m, dict):
                for k in ("version", "branch", "rebase_from", "rollback_to", "release",
                          "frozen", "allow_drift"):
                    if k in m:
                        rec[k] = _jsonable(m[k])
            self.control_history.setdefault(f"{s}|{e}", []).append(rec)

    # -- baseline snapshots around malicious windows ---------------------------
    def snapshot_baselines(self, truth: Sequence[Dict[str, Any]], now: float, dt: float,
                           final: bool = False) -> None:
        for row in truth:
            if str(row.get("label")) != "malicious":
                continue
            sid = str(row.get("scenario_id"))
            t0, t1 = _f(row.get("t_start")), _f(row.get("t_end"))
            if math.isnan(t0):
                continue
            rec = self.baseline_snaps.setdefault(sid, {})
            post_at = (t1 if not math.isnan(t1) else t0) + max(8.0 * dt, 7200.0)
            if "pre" not in rec and (now + dt >= t0 or final):
                rec["pre"] = self._baselines(row, now)
                rec["pre"]["late"] = bool(now >= t0)
            if "post" not in rec and (now >= post_at or final):
                rec["post"] = self._baselines(row, now)
                rec["post"]["truncated"] = bool(now < post_at)

    def _baselines(self, row: Dict[str, Any], now: float) -> Dict[str, Any]:
        out: Dict[str, Any] = {"ts": now, "models": {}}
        for k in _row_keys(row):
            s, _, e = k.partition("|")
            m = self.store.get_model(s, e, M_BASELINE)
            if m is not None:
                out["models"][k] = {"version": _jsonable(self.store.model_version(s, e, M_BASELINE)),
                                    "model": _baseline_snapshot(m)}
        return out

    # -- final -----------------------------------------------------------------
    def finish(self) -> None:
        for ev in self.store.events(limit=10 ** 9):
            if ev.id and ev.id not in self.events:
                self.events[ev.id] = ev
        for inc in self.store.incidents():
            self.incidents[inc.id] = inc


def _baseline_snapshot(m: Any) -> Any:
    """What the poisoning gate reads of a model.baseline (integration note
    R7.5): the anchors' plain-data summaries (Anchor._asdict: per-bucket mean,
    sd15, W; what m_baseline.anchor_summary accepts), the golden stats and the
    version fields; gate journals and held rows are left out, which keeps a
    snapshot ~10x smaller."""
    if not isinstance(m, dict):
        return _jsonable(m, max_depth=6)
    keep = ("fmt", "tier", "version", "branch", "n_eff", "allow_drift", "golden", "grain_mode")
    out = {k: _jsonable(m[k], max_depth=6) for k in keep if k in m}
    for k in ("current", "reference"):
        if k in m:
            out[k] = _jsonable(m[k], max_depth=4)
    # spec v2.1 (cadence.md §9.7): per grain. The top-level anchors are the H
    # anchors in canonical mode (the tick anchors in tick mode); the native Q
    # anchor rides along under grains.q
    if m.get("grain_mode") == "canonical":
        g: Dict[str, Any] = {"h": {k: out[k] for k in ("current", "reference") if k in out}}
        q = m.get("q")
        if isinstance(q, dict) and "current" in q:
            g["q"] = {"current": _jsonable(q["current"], max_depth=4),
                      "n_eff": _jsonable(q.get("n_eff"))}
        out["grains"] = g
    return out


def _f(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _row_keys(row: Dict[str, Any]) -> List[str]:
    """'sys|entity' keys of a truth row (entities may already be qualified)."""
    s = str(row.get("system") or "")
    out = []
    for e in row.get("entities") or []:
        e = str(e)
        out.append(e if "|" in e else f"{s}|{e}")
    return out


# --------------------------------------------------------------------------- #
# Simulated analyst (feedback gate 12)
# --------------------------------------------------------------------------- #
class SimulatedAnalyst:
    """Labels `per_day` incidents per day (newest unlabelled first) with a
    verdict derived from truth and flipped with probability `noise`; fp
    verdicts use `fp_scope` (default 'pattern'), the others 'this'. This is
    the eval harness playing the analyst; the label goes through the public
    store API exactly as the UI would put it."""

    def __init__(self, seed: int = 0, noise: float = 0.05, per_day: float = 5.0,
                 fp_scope: str = "pattern") -> None:
        self.rng = np.random.default_rng(zlib.crc32(f"analyst|{seed}".encode()))
        self.fp_scope = str(fp_scope)
        self.noise = float(noise)
        self.per_day = float(per_day)
        self.credit = 0.0
        self.labelled: set = set()
        self.n = 0

    def _verdict(self, inc: Incident, truth: Sequence[Dict[str, Any]]) -> str:
        keys = {f"{inc.system}|{inc.entity}"} | {f"{inc.system}|{x}" for x in inc.entities}
        for row in truth:
            t0, t1 = _f(row.get("t_start")), _f(row.get("t_end"))
            if keys & set(_row_keys(row)) and t0 - 3600 <= inc.opened <= t1 + 86400:
                return "tp" if row.get("label") == "malicious" else "expected_change"
        return "fp"

    def __call__(self, store: MetricStore, now: float, dt: float,
                 truth: Sequence[Dict[str, Any]]) -> int:
        self.credit += self.per_day * dt / 86400.0
        added = 0
        while self.credit >= 1.0:
            cands = [i for i in store.incidents(since=now - 86400.0)
                     if i.id not in self.labelled and not is_pseudo_entity(i.entity)]
            if not cands:
                break
            inc = cands[0]
            verdict = self._verdict(inc, truth)
            if self.rng.random() < self.noise:
                verdict = "fp" if verdict != "fp" else "tp"
            # an analyst dismisses a benign recurrence as a PATTERN (docs
            # lib3 B23: only fp / benign_known labels with a widened scope
            # become suppression policies; gate 12 measures that loop). With
            # scope 'this' on every label the simulated analyst never
            # exercised suppression, and gate 12 could not pass by design.
            scope = self.fp_scope if verdict in ("fp", "benign_known") else "this"
            store.add_label(Label(system=inc.system, entity=inc.entity, target_type="incident",
                                  target_id=inc.id, verdict=verdict, scope=scope,
                                  analyst="sim", ts=now))
            self.labelled.add(inc.id)
            self.credit -= 1.0
            self.n += 1
            added += 1
        return added


# --------------------------------------------------------------------------- #
# Persona metadata (ground truth for classes / portrait gates)
# --------------------------------------------------------------------------- #
def _persona_meta(gen: Any) -> Dict[str, Dict[str, Any]]:
    """{'sys|ip': {archetype, params...}} from whatever the generator exposes:
    `personas` (list or dict) or `systems` {system: [Persona]}."""
    out: Dict[str, Dict[str, Any]] = {}

    def add(system: str, p: Any) -> None:
        ent = getattr(p, "entity", None) if not isinstance(p, dict) else p.get("entity")
        if ent is None:
            return
        sysname = (getattr(p, "system", None) if not isinstance(p, dict) else p.get("system")) \
            or system
        arch = getattr(p, "archetype", None) if not isinstance(p, dict) else p.get("archetype")
        params = getattr(p, "params", None) if not isinstance(p, dict) else p.get("params")
        meta = {"archetype": str(arch or ""), "system": str(sysname), "entity": str(ent)}
        if params is not None:
            meta["params"] = _jsonable(params, max_depth=4)
        out[f"{sysname}|{ent}"] = meta

    personas = getattr(gen, "personas", None)
    if isinstance(personas, dict):
        for k, v in personas.items():
            if isinstance(v, (list, tuple)):
                for p in v:
                    add(str(k), p)
            else:
                add(str(k).partition("|")[0], v)
    elif isinstance(personas, (list, tuple)):
        for p in personas:
            add("", p)
    systems = getattr(gen, "systems", None)
    if isinstance(systems, dict):
        for s, plist in systems.items():
            for p in plist or []:
                if f"{s}|{getattr(p, 'entity', '')}" not in out:
                    add(str(s), p)
    return out


def _truth_of(gen: Any) -> List[Dict[str, Any]]:
    return [_jsonable(r) for r in (getattr(gen, "truth", None) or [])]


# --------------------------------------------------------------------------- #
# run_pack
# --------------------------------------------------------------------------- #
def run_pack(pack_name: Any, seed: int, registry_factory: Optional[Callable] = None,
             strict: bool = True, time_budget_s: Optional[float] = None, *,
             generator_factory: Optional[Callable] = None,
             disable_engines: Iterable[str] = (),
             analyst: Optional[Callable] = None,
             on_error: str = "record",
             record_series: bool = True,
             stale_every: int = 1,
             keep_store: bool = False) -> RunResult:
    """Run one pack for one seed and return a RunResult.

    pack_name        a pack name for packs.get_pack, or a Pack-like object
    registry_factory callable(pack=?, config=?, seed=?) -> Registry
                     (default: build.build_registry via default_registry_factory)
    strict           ctx.config['strict']: engines re-raise. The runner records
                     each exception; on_error='record' (default) keeps the tick
                     going so the whole report is still produced, 'abort' stops
                     the run, 'raise' propagates.
    time_budget_s    stop (aborted='time_budget') once the wall time exceeds it
    disable_engines  engine names switched off (ablation / fault injection)
    analyst          callable(store, now, dt, truth) run after each scenario
                     tick (e.g. SimulatedAnalyst for gate 12)
    """
    t_wall0 = time.perf_counter()
    pack = _resolve_pack(pack_name)
    name = str(getattr(pack, "name", pack_name))
    plan = _phase_plan(pack)
    config = _pack_config(pack, strict)
    store = MetricStore()
    factory = registry_factory or default_registry_factory
    registry = _call_with_supported(factory, pack=pack, config=config, seed=seed)
    disabled = []
    wanted = {str(n) for n in disable_engines}
    for eng in registry.ordered():
        if eng.name in wanted or type(eng).__name__ in wanted:
            eng.enabled = False
            disabled.append(eng.name)
    pipeline = Pipeline(store, registry, window_s=int(plan[0].dt), config=config)
    gen = _make_generator(pack, seed, generator_factory)
    rec = _Recorder(registry, on_error)
    stale = _StaleChecker()
    col = _Collector(store)

    result = RunResult(pack=name, seed=int(seed), strict=bool(strict), config=_jsonable(config),
                       phases=[vars(p).copy() for p in plan], disabled_engines=disabled)
    scen_ticks: List[float] = []
    scen_dt = plan[-1].dt
    holdout_ts: Optional[float] = None
    holdout_portraits: Dict[str, Any] = {}
    memory: Dict[str, Any] = {}
    labels = 0
    tick_i = 0
    try:
        for ph in plan:
            if not ph.training:
                scen_dt = ph.dt
            n_hold = ph.n_ticks - min(ph.n_ticks // 2, int(round(86400.0 / ph.dt)))
            for i in range(ph.n_ticks):
                rec.begin_tick()
                t0 = time.perf_counter()
                obs = _gen_step(gen, ph.dt, live=not ph.training, aggregated=ph.aggregated)
                t1 = time.perf_counter()
                now = float(gen.vt)
                pipeline.run_tick(obs, now=now, training=ph.training, dt=ph.dt)
                t2 = time.perf_counter()
                if not ph.training:
                    scen_ticks.append(now)
                    col.on_tick(now, ph.dt, record_series=record_series)
                    truth_now = _truth_of(gen)
                    col.snapshot_baselines(truth_now, now, ph.dt)
                    if analyst is not None:
                        labels += int(_call_with_supported(
                            analyst, store=store, now=now, dt=ph.dt, truth=truth_now) or 0)
                    if i == n_hold:
                        holdout_ts = now
                        holdout_portraits = _portraits(store)
                if stale_every and tick_i % max(1, int(stale_every)) == 0:
                    stale.check(store, now, ph.dt)
                t3 = time.perf_counter()
                rec.end_tick(now, ph.dt, ph.index, ph.training, t1 - t0, t2 - t1, t3 - t2)
                tick_i += 1
                if time_budget_s is not None and time.perf_counter() - t_wall0 > time_budget_s:
                    raise RunAborted("time_budget")
            memory[f"phase{ph.index}"] = _jsonable(store.memory_report())
    except RunAborted as exc:
        result.aborted = "time_budget" if str(exc) == "time_budget" else f"engine_error: {exc}"
    except Exception as exc:  # on_error='raise' or a harness/generator failure
        result.aborted = f"exception: {type(exc).__name__}: {exc}"
        if on_error == "raise":
            raise
    truth = _truth_of(gen)
    if scen_ticks:
        col.snapshot_baselines(truth, scen_ticks[-1], scen_dt, final=True)
    col.finish()

    memory["final"] = _jsonable(store.memory_report())
    result.truth = truth
    result.personas = _persona_meta(gen)
    result.fixtures = _jsonable(getattr(pack, "fixtures", None) or {})
    result.scenario_dt = float(scen_dt)
    result.tick_ts = np.asarray(scen_ticks, dtype=np.float64)
    if scen_ticks:
        result.scenario_window = (scen_ticks[0] - scen_dt, scen_ticks[-1])
    result.entity_first_seen = dict(col.first_seen)
    result.events = sorted((event_to_dict(e) for e in col.events.values()),
                           key=lambda d: (d["ts"], d["id"]))
    incs = []
    for iid, inc in col.incidents.items():
        d = incident_to_dict(inc)
        d["history"] = col.inc_hist.get(iid, [])
        incs.append(d)
    result.incidents = sorted(incs, key=lambda d: (d.get("opened", 0.0), d.get("id", "")))
    result.series = {k: b.to_arrays() for k, b in col.series.items()}
    result.profiles = _profiles(store)
    result.models = {
        "class": col.class_history[-1] if col.class_history else {},
        "class_history": col.class_history,
        "link": {s: h[-1]["model"] for s, h in col.link_history.items() if h},
        "link_history": col.link_history,
        "control_history": col.control_history,
        "baseline_snaps": col.baseline_snaps,
        "feedback": _jsonable(store.get_model(ORG, ORG, M_FEEDBACK), max_depth=5),
    }
    result.holdout = _holdout(store, holdout_ts, holdout_portraits)
    result.timings = rec.timings()
    result.exceptions = rec.exceptions
    result.stale = list(stale.found.values())
    result.memory = memory
    result.health = _jsonable(store.health(), max_depth=4)
    result.labels_added = labels
    result.wall_s = time.perf_counter() - t_wall0
    if keep_store:
        result.store = store
    return result


def _profiles(store: MetricStore) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for p in store.all_profiles():
        extra = p.extra or {}
        out[f"{p.system}|{p.entity}"] = {
            "system": p.system, "entity": p.entity, "archetype": p.archetype,
            "archetype_confidence": _jsonable(p.archetype_confidence),
            "separability": _jsonable(p.separability), "stable": bool(p.stable),
            "sample_count": int(p.sample_count or 0), "updated": _jsonable(p.updated),
            "extra": {k: _jsonable(extra[k], max_depth=6) for k in PROFILE_EXTRA_KEYS
                      if k in extra},
        }
    return out


def _portraits(store: MetricStore) -> Dict[str, Any]:
    out = {}
    for p in store.all_profiles():
        por = (p.extra or {}).get("portrait")
        if por is not None:
            out[f"{p.system}|{p.entity}"] = _jsonable(por, max_depth=6)
    return out


def _holdout(store: MetricStore, ts: Optional[float], portraits: Dict[str, Any]) -> Dict[str, Any]:
    """Portrait snapshot at `ts` plus the feature.nat rows after it: the
    held-out ticks for the portrait p5-p95 coverage check (gate 11)."""
    if ts is None:
        return {}
    cols = store.vec_columns(FEATURE_NAT) or store.vec_columns("feature.vec")
    if cols is None:
        try:
            from ..engines.behavior.lib.features import FEATURE_NAMES_V2
            cols = list(FEATURE_NAMES_V2)
        except Exception:  # pragma: no cover
            cols = []
    nat: Dict[str, Any] = {}
    grains: Dict[str, Dict[str, Any]] = {"h": {}, "q": {}}
    for s in store.systems():
        for e in store.entities(s):
            t, m = store.vec_since(s, e, FEATURE_NAT, ts + 1e-6)
            if len(t):
                nat[f"{s}|{e}"] = {"ts": t, "values": m}
            # spec v2.1 (cadence.md §9.7): held-out grain rows (active only)
            for g in ("h", "q"):
                tg, mg = store.vec_since(s, e, f"feature.nat.{g}", ts + 1e-6)
                if not len(tg):
                    continue
                ta, ma = store.vec_since(s, e, f"feature.meta.{g}", ts + 1e-6)
                act = {float(a): float(r[0]) > 0.5 for a, r in zip(ta, ma)}
                keep = [i for i, x in enumerate(tg) if act.get(float(x), False)]
                if keep:
                    grains[g][f"{s}|{e}"] = {"ts": np.asarray(tg)[keep],
                                             "values": np.asarray(mg)[keep]}
    out = {"ts": ts, "portraits": portraits, "feature_names": list(cols), "nat": nat}
    if grains["h"] or grains["q"]:
        out["grains"] = grains
    return out
