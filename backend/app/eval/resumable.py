"""Resumable pack runs (round 3): runner.run_pack with checkpoints.

The container that runs the evaluations restarts every few hours, and a
pack-O-real run (35 days) or a 7-day PG4 point at 20 000 IPs takes longer than
one 40-minute slot. `run_pack_resumable` is runner.run_pack's tick loop - the
same calls in the same order, so a resumed run produces the same RunResult as
an uninterrupted one (tests/eval/test_resumable.py) - with the whole loop
state (store, registry, generator, recorder, collector, snapshotter,
counters) pickled to a checkpoint file at the end of a local day once
`segment_s` of wall time has passed since the previous checkpoint. Called
again with the same file, it resumes from the last checkpoint; with
`stop_after_s` it returns None after a checkpoint once that much wall time was
spent in this process (a driver loop then restarts it in a fresh process).

Pickling: cloudpickle (engines hold lambdas and local functions), the
engines' identity-compared sentinels (pevent.ABSENT ...) as references to the
module constants, thread locks re-created on load, weak references and weak dictionaries (engines'
per-store state keyed by the store) pickled with their referents, which the
checkpoint holds strongly; the recorder's per-engine timing wrappers are
removed before pickling and re-installed on load.

Per tick it also records what pscale.run_point taps: the generator's org
events and P00's (batch rows, learned rows), so a PG4 point can be measured
over several processes (pscale.run_point_resumable).
"""
from __future__ import annotations

import copyreg
import os
import pickle
import threading
import time
import weakref
from typing import Any, Callable, Dict, Optional

import numpy as np

from . import runner as R


def _reducers() -> Dict[type, Callable[[Any], Any]]:
    tbl: Dict[type, Callable[[Any], Any]] = {
        type(threading.RLock()): lambda _l: (threading.RLock, ()),
        type(threading.Lock()): lambda _l: (threading.Lock, ()),
        weakref.ReferenceType: lambda r: (weakref.ref, (r(),)) if r() is not None else (type(None), ()),
        weakref.WeakKeyDictionary: lambda d: (weakref.WeakKeyDictionary, (list(d.items()),)),
        weakref.WeakValueDictionary: lambda d: (weakref.WeakValueDictionary, (list(d.items()),)),
        weakref.WeakSet: lambda d: (weakref.WeakSet, (list(d),)),
    }
    return tbl


# Module-level sentinels the engines compare by IDENTITY ('v is EV.ABSENT');
# pickled by value they would come back as equal but distinct objects and every
# such test would fail after a resume (found: P07's key-set presence counted
# stored '⊥' values as a key). They are pickled as references to the module
# constant instead.
_SENTINELS = (("app.engines.behavior.lib.pevent", "ABSENT"),
              ("app.engines.behavior.lib.pselect", "MISSING"),
              ("app.engines.behavior.lib.psketch", "_MISSING"),
              ("app.engines.behavior.likelihood", "_UNSET"),
              ("app.engines.raw.client_stack", "_MISS"))


def _sentinels() -> Dict[int, str]:
    import importlib
    out: Dict[int, str] = {}
    for mod, name in _SENTINELS:
        try:
            out[id(getattr(importlib.import_module(mod), name))] = f"{mod}:{name}"
        except (ImportError, AttributeError):            # pragma: no cover
            continue
    return out


def _sentinel(tag: str) -> Any:
    import importlib
    mod, name = tag.split(":")
    return getattr(importlib.import_module(mod), name)


def ckpt_pickler(f: Any) -> pickle.Pickler:
    """A pickler for the live run state (see the module docstring)."""
    tbl = _reducers()
    sent = _sentinels()
    try:
        import cloudpickle
        base: Any = cloudpickle.CloudPickler
    except ImportError:                                  # pragma: no cover
        base = pickle.Pickler

    class _P(base):                                      # type: ignore[misc, valid-type]
        def persistent_id(self, obj: Any) -> Optional[str]:
            return sent.get(id(obj)) if id(obj) in sent else None

        def reducer_override(self, obj: Any) -> Any:
            red = tbl.get(type(obj))
            if red is not None:
                return red(obj)
            sup = getattr(super(), "reducer_override", None)
            return sup(obj) if sup is not None else NotImplemented
    return _P(f, protocol=pickle.HIGHEST_PROTOCOL)


class _Unpickler(pickle.Unpickler):
    def persistent_load(self, pid: Any) -> Any:
        return _sentinel(str(pid))


def ckpt_load(f: Any) -> Any:
    return _Unpickler(f).load()


def _org_events(gen: Any) -> float:
    org = getattr(gen, "org", None)
    return float(org.stats["events"]) if org is not None else 0.0


def run_pack_resumable(pack: Any, seed: int, ckpt: str, *, segment_s: float = 1800.0,
                       stop_after_s: Optional[float] = None, record_series: bool = False,
                       keep_store: bool = False, registry_factory: Optional[Callable] = None,
                       on_progress: Optional[Callable[[str], None]] = None) -> Optional[R.RunResult]:
    """runner.run_pack (strict, on_error='record', stale_every=1, no analyst,
    no time budget), resumable from `ckpt`. Returns the RunResult, or None
    when it stopped at a checkpoint (stop_after_s). result.tap holds the
    per-tick org events and P00 (rows, learned) for PG4; result.segments the
    number of checkpoints written."""
    t_proc0 = time.perf_counter()
    if os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            st: Dict[str, Any] = ckpt_load(f)
        rec = R._Recorder.__new__(R._Recorder)
        rec.__dict__.update(st.pop("rec_state"))
        rec.engines = st["registry"].ordered()
        rec._instrument()
        st["rec"] = rec
    else:
        p = R._resolve_pack(pack)
        plan = R._phase_plan(p)
        config = R._pack_config(p, True)
        store = R.MetricStore()
        registry = R._call_with_supported(registry_factory or R.default_registry_factory,
                                          pack=p, config=config, seed=seed)
        pipeline = R.Pipeline(store, registry, window_s=int(plan[0].dt), config=config)
        gen = R._make_generator(p, seed, None)
        st = {"pack": p, "plan": plan, "config": config, "store": store, "registry": registry,
              "pipeline": pipeline, "gen": gen, "rec": R._Recorder(registry, "record"),
              "stale": R._StaleChecker(), "col": R._Collector(store), "snapper": R._DaySnapper(p),
              "result": R.RunResult(pack=str(getattr(p, "name", pack)), seed=int(seed), strict=True,
                                    config=R._jsonable(config), phases=[vars(x).copy() for x in plan],
                                    disabled_engines=[]),
              "scen_ticks": [], "scen_dt": plan[-1].dt, "holdout_ts": None, "holdout_portraits": {},
              "memory": {}, "tick_i": 0, "pos": (0, 0), "wall_s": 0.0, "segments": 0,
              "tap_events": [], "tap_batch": []}
    plan, store, pipeline, gen = st["plan"], st["store"], st["pipeline"], st["gen"]
    rec, stale, col, snapper = st["rec"], st["stale"], st["col"], st["snapper"]
    p00 = next((e for e in st["registry"].ordered() if e.name == "raw.event"), None)
    p04 = next((e for e in st["registry"].ordered() if e.name == "behavior.pattern_tree"), None)
    wall0 = float(st["wall_s"])
    t_seg = time.perf_counter()

    def save() -> None:
        engines = st["registry"].ordered()
        wrapped = {id(e): e.__dict__.pop("safe_run") for e in engines if "safe_run" in e.__dict__}
        body = {k: v for k, v in st.items() if k != "rec"}
        body["rec_state"] = {k: v for k, v in rec.__dict__.items() if k != "engines"}
        body["wall_s"] = wall0 + time.perf_counter() - t_proc0
        body["segments"] = st["segments"] + 1
        tmp = ckpt + ".tmp"
        try:
            with open(tmp, "wb") as f:
                ckpt_pickler(f).dump(body)
            os.replace(tmp, ckpt)
        finally:
            for e in engines:
                if id(e) in wrapped:
                    e.safe_run = wrapped[id(e)]
        st["segments"] += 1

    pi0, i0 = st["pos"]
    for pi in range(pi0, len(plan)):
        ph = plan[pi]
        if not ph.training:
            st["scen_dt"] = ph.dt
        n_hold = ph.n_ticks - min(ph.n_ticks // 2, int(round(86400.0 / ph.dt)))
        for i in range(i0 if pi == pi0 else 0, ph.n_ticks):
            rec.begin_tick()
            e0 = _org_events(gen)
            t0 = time.perf_counter()
            obs = R._gen_step(gen, ph.dt, live=not ph.training, aggregated=ph.aggregated)
            t1 = time.perf_counter()
            now = float(gen.vt)
            pipeline.run_tick(obs, now=now, training=ph.training, dt=ph.dt)
            t2 = time.perf_counter()
            st["tap_events"].append(_org_events(gen) - e0)
            ls = (getattr(p00, "last_stats", None) or {}) if p00 is not None else {}
            l4 = (getattr(p04, "last_stats", None) or {}) if p04 is not None else {}
            st["tap_batch"].append((float(ls.get("events", 0) or 0), float(ls.get("learned", 0) or 0),
                                    float(l4.get("learned", 0) or 0) if p04 is not None else float("nan")))
            if not ph.training:
                st["scen_ticks"].append(now)
                col.on_tick(now, ph.dt, record_series=record_series)
                col.snapshot_baselines(R._truth_of(gen), now, ph.dt)
                if i == n_hold:
                    st["holdout_ts"] = now
                    st["holdout_portraits"] = R._portraits(store)
            stale.check(store, now, ph.dt)                 # run_pack's stale_every = 1
            snapper.after_tick(store, now)
            t3 = time.perf_counter()
            rec.end_tick(now, ph.dt, ph.index, ph.training, t1 - t0, t2 - t1, t3 - t2)
            st["tick_i"] += 1
            last = (i + 1 == ph.n_ticks and pi + 1 == len(plan))
            day_end = snapper.clock is not None and snapper.day(now) != snapper.day(now - 1e-6)
            if not last and day_end and time.perf_counter() - t_seg >= segment_s:
                st["pos"] = (pi, i + 1) if i + 1 < ph.n_ticks else (pi + 1, 0)
                save()
                t_seg = time.perf_counter()
                if on_progress is not None:
                    on_progress(f"checkpoint {st['segments']} at tick {st['tick_i']} "
                                f"(day {snapper.day(now - 1e-6)}), {wall0 + t_seg - t_proc0:.0f} s")
                if stop_after_s is not None and time.perf_counter() - t_proc0 >= stop_after_s:
                    return None
        st["memory"][f"phase{ph.index}"] = R._jsonable(store.memory_report())
    # ---- the tail of run_pack, same order
    result, scen_ticks, scen_dt = st["result"], st["scen_ticks"], st["scen_dt"]
    truth = R._truth_of(gen)
    if scen_ticks:
        col.snapshot_baselines(truth, scen_ticks[-1], scen_dt, final=True)
        snapper.after_tick(store, scen_ticks[-1], final=True)
    col.finish()
    memory = st["memory"]
    memory["final"] = R._jsonable(store.memory_report())
    p = st["pack"]
    result.truth = truth
    result.personas = R._persona_meta(gen)
    result.fixtures = R._jsonable(getattr(p, "fixtures", None) or {})
    result.scenario_dt = float(scen_dt)
    result.tick_ts = np.asarray(scen_ticks, dtype=np.float64)
    if scen_ticks:
        result.scenario_window = (scen_ticks[0] - scen_dt, scen_ticks[-1])
    result.entity_first_seen = dict(col.first_seen)
    result.events = sorted((R.event_to_dict(e) for e in col.events.values()),
                           key=lambda d: (d["ts"], d["id"]))
    incs = []
    for iid, inc in col.incidents.items():
        d = R.incident_to_dict(inc)
        d["history"] = col.inc_hist.get(iid, [])
        incs.append(d)
    result.incidents = sorted(incs, key=lambda d: (d.get("opened", 0.0), d.get("id", "")))
    result.series = {k: b.to_arrays() for k, b in col.series.items()}
    result.profiles = R._profiles(store)
    result.models = {
        "class": col.class_history[-1] if col.class_history else {},
        "class_history": col.class_history,
        "link": {s: h[-1]["model"] for s, h in col.link_history.items() if h},
        "link_history": col.link_history,
        "control_history": col.control_history,
        "baseline_snaps": col.baseline_snaps,
        "feedback": R._jsonable(store.get_model(R.ORG, R.ORG, R.M_FEEDBACK), max_depth=5),
    }
    result.holdout = R._holdout(store, st["holdout_ts"], st["holdout_portraits"])
    result.timings = rec.timings()
    result.exceptions = rec.exceptions
    result.stale = list(stale.found.values())
    result.memory = memory
    result.health = R._jsonable(store.health(), max_depth=4)
    result.labels_added = 0
    result.ptruth = R._jsonable(getattr(gen, "ptruth", None) or {}, max_depth=12)
    result.psnaps = snapper.snaps
    org = getattr(gen, "org", None)
    result.gen_stats = dict(getattr(org, "stats", None) or {})
    result.wall_s = wall0 + time.perf_counter() - t_proc0
    result.segments = int(st["segments"])
    result.tap = {"events": list(st["tap_events"]), "batch": list(st["tap_batch"])}
    if keep_store:
        result.store = store
    return result
