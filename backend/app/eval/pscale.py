"""PG4 scaling experiment harness (docs/lib3/progressive.md §12 PG4).

The requirement forbids cost that grows with the number of users, metrics or
servers. PG4 measures it: run the pipeline on packs that differ in one size
only (O-scale-{500,5k,20k}-{0,60,300} for IPs and attributes, O-servers-{20,
100,300} for systems), measure the P-core's steady-state memory and CPU per
event, and fit log-log slopes.

  run_point(pack, seed, registry_factory=None) -> one measurement dict
  pg4_summary(points)                          -> the PG4 gate record
  loglog_slope (from pmetrics)

Memory is the deep size of the P-core models in the store (model.ptree, the
fitted models, registry, selection, who groups, families, budget, views;
numpy buffers by nbytes, containers recursively, shared objects once), per
store key (system / family tree / org). CPU is the wall time of the P engines
(runner timings, by engine name) divided by the number of events the
generator emitted; the per-tick p95 of µs per event is reported for P03
(scoring) and P04 (learning). Engine and model names are parameters, so the
harness is validated today with toy engines (tests/eval/test_pscale.py: a
bounded learner gives slope ~0, a per-IP learner slope ~1) and measures the
real P engines unchanged once they exist.
"""
from __future__ import annotations

import sys
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.store import MetricStore
from ..models.schema import ORG, SYSTEM_ENTITY
from .metrics import check, gate
from .pmetrics import loglog_slope
from .runner import P_ORG, P_SYS_FULL, _fam_keys, run_pack

P_ENGINES = ("raw.event", "derived.event_context", "behavior.attr_registry",
             "behavior.conformity", "behavior.pattern_tree", "behavior.attr_select",
             "behavior.content_bounds", "behavior.payload_grammar", "behavior.binding",
             "behavior.time_window", "behavior.workflow", "behavior.who_groups",
             "behavior.system_profile", "behavior.facets", "behavior.views",
             "behavior.resource_governor")
P_MODELS = tuple(P_SYS_FULL) + tuple(P_ORG) + ("model.pwant",)
SCORING_ENGINE = "behavior.conformity"
LEARNING_ENGINE = "behavior.pattern_tree"
BASE_ATTRS = 40                      # attributes of pack O without synthetic ones (§12 PG4 axis)
MB = 1024.0 * 1024.0


def deep_sizeof(obj: Any, seen: Optional[set] = None, depth: int = 0) -> int:
    """Bytes of an object graph: numpy arrays by nbytes, containers and
    objects recursively (each object once)."""
    if seen is None:
        seen = set()
    i = id(obj)
    if i in seen or depth > 64:
        return 0
    seen.add(i)
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes) + 112
    n = sys.getsizeof(obj)
    if isinstance(obj, (str, bytes, int, float, bool, type(None))):
        return n
    if isinstance(obj, dict):
        for k, v in obj.items():
            n += deep_sizeof(k, seen, depth + 1) + deep_sizeof(v, seen, depth + 1)
        return n
    if isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            n += deep_sizeof(v, seen, depth + 1)
        return n
    slots = getattr(type(obj), "__slots__", None)
    if slots:
        for s in slots:
            if hasattr(obj, s):
                n += deep_sizeof(getattr(obj, s), seen, depth + 1)
    if hasattr(obj, "__dict__"):
        n += deep_sizeof(vars(obj), seen, depth + 1)
    return n


def pcore_memory(store: MetricStore, model_names: Sequence[str] = P_MODELS) -> Dict[str, Any]:
    """Deep size of the P-core models per store key."""
    seen: set = set()
    by_key: Dict[str, int] = {}
    org = 0
    for n in model_names:
        m = store.get_model(ORG, ORG, n)
        if m is not None:
            org += deep_sizeof(m, seen)
    keys = set(store.systems()) | set(_fam_keys(store.get_model(ORG, ORG, "model.sysfam")))
    for s in sorted(keys):
        b = 0
        for n in model_names:
            m = store.get_model(s, SYSTEM_ENTITY, n)
            if m is not None:
                b += deep_sizeof(m, seen)
        if b:
            by_key[s] = b
    return {"total": org + sum(by_key.values()), "org": org, "by_key": by_key}


def pcore_cpu(timings: Mapping[str, Any], events_per_tick: Sequence[float],
              engines: Sequence[str] = P_ENGINES,
              batch_per_tick: Optional[Sequence[Tuple[float, float]]] = None) -> Dict[str, Any]:
    """CPU of the P engines. us_per_event: all P engines over the generator's
    events. The per-event p95s follow §12 PG4's units: scoring (P03) per
    SCORED event (the rows of P00's batches: P03 scores every row) and
    learning (P04) per LEARNED event (P00's learning sample, §6.2.2); with no
    batch tap (batch_per_tick None) both fall back to the generator's events,
    which understates the learning cost by the sampling ratio."""
    names = list(timings.get("engine_names") or [])
    ms = np.asarray(timings.get("engine_ms"), dtype=float)
    if ms.ndim != 2 or not names:
        return {"ms_total": 0.0, "us_per_event": None, "by_engine": {}}
    idx = [i for i, n in enumerate(names) if n in set(engines)]
    ev = np.asarray(events_per_tick, dtype=float)
    k = min(len(ev), ms.shape[0])
    ms, ev = ms[:k], ev[:k]
    tot_ev = float(ev.sum())
    by = {names[i]: float(ms[:, i].sum()) for i in idx}
    total = float(sum(by.values()))
    scored, learned = ev, ev
    if batch_per_tick is not None and len(batch_per_tick) >= k:
        bt = np.asarray(batch_per_tick, dtype=float)[:k]
        scored, learned = bt[:, 0], bt[:, 1]

    def p95(name: str, per: np.ndarray) -> Optional[float]:
        if name not in names:
            return None
        col = ms[:, names.index(name)]
        m = per > 0
        return float(np.quantile(col[m] * 1000.0 / per[m], 0.95)) if m.any() else None
    out = {"ms_total": total, "events": tot_ev,
           "us_per_event": total * 1000.0 / tot_ev if tot_ev else None,
           "scoring_p95_us": p95(SCORING_ENGINE, scored), "learning_p95_us": p95(LEARNING_ENGINE, learned),
           "by_engine": by}
    if batch_per_tick is not None and len(batch_per_tick) >= k:
        out.update({"scored_events": float(scored.sum()), "learned_events": float(learned.sum()),
                    "scoring_us_per_event": (by.get(SCORING_ENGINE, 0.0) * 1000.0 / float(scored.sum())
                                             if scored.sum() else None),
                    "learning_us_per_learned_event": (by.get(LEARNING_ENGINE, 0.0) * 1000.0
                                                      / float(learned.sum()) if learned.sum() else None),
                    "units": "scoring per scored (batch) event, learning per learned event"})
    else:
        out["units"] = "per generator event (no batch tap)"
    return out


class _BatchTap:
    """Records P00's (batch rows, learned rows) at every tick by wrapping the
    raw.event engine's run (its last_stats), so PG4's per-event costs use the
    units of §12: scored events for P03, learned events for P04."""

    def __init__(self) -> None:
        self.rows: List[Tuple[float, float]] = []

    def factory(self, base: Optional[Callable] = None) -> Callable:
        from .runner import _call_with_supported, default_registry_factory

        def make(pack: Any = None, config: Any = None, seed: int = 0) -> Any:
            reg = _call_with_supported(base or default_registry_factory, pack=pack, config=config,
                                       seed=seed)
            for eng in reg.ordered():
                if eng.name == "raw.event":
                    run0 = eng.run

                    def run(ctx: Any, observations: Any = None, _run0=run0, _eng=eng) -> Any:
                        out = _run0(ctx, observations)
                        st = getattr(_eng, "last_stats", None) or {}
                        self.rows.append((float(st.get("events", 0) or 0), float(st.get("learned", 0) or 0)))
                        return out
                    eng.run = run
            return reg
        return make


class _CountingGen:
    """Generator wrapper that records the org events emitted per tick."""

    def __init__(self, gen: Any) -> None:
        self.gen = gen
        self.per_tick: List[float] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.gen, name)

    def step(self, dt: float, live: bool = False, aggregated: Optional[bool] = None) -> List[Any]:
        org = getattr(self.gen, "org", None)
        before = org.stats["events"] if org is not None else 0
        obs = self.gen.step(dt, live=live, aggregated=aggregated)
        after = org.stats["events"] if org is not None else 0
        self.per_tick.append(float(after - before) if org is not None else
                             float(sum(int((o.extra or {}).get("count", 1)) for o in obs)))
        return obs


def run_point(pack: Any, seed: int = 0, registry_factory: Optional[Callable] = None,
              engines: Sequence[str] = P_ENGINES, model_names: Sequence[str] = P_MODELS,
              **run_kw: Any) -> Dict[str, Any]:
    """One scaling measurement: run the pack, then measure P-core memory (end
    state) and CPU per event."""
    from ..pipeline.generator import TrafficGenerator
    from .packs import get_pack
    p = get_pack(pack) if isinstance(pack, str) else pack
    holder: Dict[str, _CountingGen] = {}

    def gen_factory(pack: Any, seed: int) -> Any:
        g = _CountingGen(TrafficGenerator(seed=seed, pack=pack))
        holder["g"] = g
        return g

    tap = _BatchTap()
    res = run_pack(p, seed, registry_factory=tap.factory(registry_factory), generator_factory=gen_factory,
                   keep_store=True, record_series=False, **run_kw)
    mem = pcore_memory(res.store, model_names)
    cpu = pcore_cpu(res.timings, holder["g"].per_tick if "g" in holder else [], engines,
                    batch_per_tick=tap.rows if tap.rows else None)
    org = getattr(p, "org", None)
    n_meta = sum(1 for a in (getattr(org, "attr_schedule", None) or [])
                 if a.name.startswith("f"))
    idle = set((getattr(org, "config", None) or {}).get("idle_systems") or [])
    out = {"pack": getattr(p, "name", str(pack)), "seed": int(seed),
           "n_ips": int(getattr(org, "portal_n", 0) or 0),
           "n_attrs": BASE_ATTRS + n_meta,
           "n_systems": len(getattr(org, "systems", None) or []),
           "mem_bytes": mem["total"], "mem_by_key": mem["by_key"],
           "idle_max_bytes": max([mem["by_key"].get(s, 0) for s in idle] or [0]) if idle else None,
           "cpu": cpu, "events": res.gen_stats.get("events"), "aborted": res.aborted,
           "exceptions": len(res.exceptions)}
    res.store = None
    return out


def _slope(points: Sequence[Mapping[str, Any]], x: str, y: Callable[[Mapping[str, Any]], Any]
           ) -> Optional[float]:
    pts = [(float(p[x]), y(p)) for p in points if y(p)]
    if len({a for a, _ in pts}) < 2:
        return None
    return loglog_slope([a for a, _ in pts], [float(b) for _, b in pts])


def pg4_summary(points: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """PG4 from scaling points (median over seeds per pack first)."""
    by_pack: Dict[str, List[Mapping[str, Any]]] = {}
    for p in points:
        by_pack.setdefault(str(p["pack"]), []).append(p)
    med: List[Dict[str, Any]] = []
    for name, ps in by_pack.items():
        q = dict(ps[0])
        q["mem_bytes"] = float(np.median([x["mem_bytes"] for x in ps]))
        ups = [x["cpu"].get("us_per_event") for x in ps if x["cpu"].get("us_per_event")]
        q["us_per_event"] = float(np.median(ups)) if ups else None
        med.append(q)
    scale = [p for p in med if str(p["pack"]).startswith("O-scale")]
    servers = [p for p in med if str(p["pack"]).startswith("O-servers")]
    base_attrs = min((p["n_attrs"] for p in scale), default=None)
    base_ips = min((p["n_ips"] for p in scale), default=None)
    ip_axis = [p for p in scale if p["n_attrs"] == base_attrs]
    at_axis = [p for p in scale if p["n_ips"] == base_ips]
    m_ip = _slope(ip_axis, "n_ips", lambda p: p["mem_bytes"])
    c_ip = _slope(ip_axis, "n_ips", lambda p: p.get("us_per_event"))
    m_at = _slope(at_axis, "n_attrs", lambda p: p["mem_bytes"])
    c_at = _slope(at_axis, "n_attrs", lambda p: p.get("us_per_event"))
    m_sv = _slope(servers, "n_systems", lambda p: p["mem_bytes"])
    idle = [p.get("idle_max_bytes") for p in servers if p.get("idle_max_bytes") is not None]
    per_tree = [max(p["mem_by_key"].values()) for p in med if p.get("mem_by_key")]
    sc = [p["cpu"].get("scoring_p95_us") for p in med if p["cpu"].get("scoring_p95_us")]
    le = [p["cpu"].get("learning_p95_us") for p in med if p["cpu"].get("learning_p95_us")]
    le_ = lambda v, t: None if v is None else v <= t + 1e-12
    checks = [
        check("memory slope vs IPs (attrs fixed)", m_ip, "<= 0.15 (0.35 with P11 ip mode)", le_(m_ip, 0.15)),
        check("CPU/event slope vs IPs", c_ip, "<= 0.1", le_(c_ip, 0.1)),
        check("memory slope vs attributes (IPs fixed)", m_at, "<= 0.2", le_(m_at, 0.2)),
        check("CPU/event slope vs attributes", c_at, "<= 0.2", le_(c_at, 0.2)),
        check("memory slope vs systems (12 families)", m_sv, "<= 0.3", le_(m_sv, 0.3)),
        check("idle system memory after >= 1 day (MB)", max(idle) / MB if idle else None, "<= 1",
              le_(max(idle) / MB if idle else None, 1.0)),
        check("max P-core memory per tree (MB)", max(per_tree) / MB if per_tree else None, "<= 40",
              le_(max(per_tree) / MB if per_tree else None, 40.0)),
        check("scoring p95 us/event", max(sc) if sc else None, "<= 100", le_(max(sc) if sc else None, 100.0)),
        check("learning p95 us/event", max(le) if le else None, "<= 250", le_(max(le) if le else None, 250.0)),
    ]
    return gate("PG4 resources sublinear in IPs, metrics and servers", m_ip, "slopes per §12", checks,
                points=[{k: p.get(k) for k in ("pack", "n_ips", "n_attrs", "n_systems", "mem_bytes",
                                               "us_per_event")} for p in med])


def run_grid(packs: Iterable[str], seeds: Iterable[int] = (0,), **kw: Any) -> List[Dict[str, Any]]:
    return [run_point(p, s, **kw) for p in packs for s in seeds]
