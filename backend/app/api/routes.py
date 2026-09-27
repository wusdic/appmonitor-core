"""REST API — read models over the running pipeline.

Every endpoint reads the shared store the Runtime exposes; none reach into an
engine. The API is a thin projection layer, which is why swapping engines never
breaks it.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import yaml
from fastapi import APIRouter, HTTPException, Query

from ..engines.behavior.lib.detectors import DETECTORS
from ..engines.behavior.lib.features import FEATURE_NAMES_V2
from ..pipeline.build import Runtime
from .serialize import to_jsonable

router = APIRouter(prefix="/api")

# The Runtime is created and attached by main.py
RUNTIME: Optional[Runtime] = None


def rt() -> Runtime:
    if RUNTIME is None:
        raise HTTPException(503, "runtime not started")
    return RUNTIME


def _catalog_path() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "catalog.yaml")


@router.get("/health")
def health():
    r = rt()
    return {"status": "ok", "warmed": r.warmed, "live_ticks": r.live_ticks,
            "tick_count": r.pipeline.tick_count}


@router.get("/overview")
def overview():
    r = rt()
    systems = r.store.systems()
    events = r.store.events(limit=500)
    matches = r.store.matches(limit=800)
    profiles = r.store.all_profiles()
    sev_counts: dict = {}
    for e in events:
        sev_counts[e.severity.value] = sev_counts.get(e.severity.value, 0) + 1
    cat_counts: dict = {}
    for m in matches:
        cat_counts[m.category] = cat_counts.get(m.category, 0) + 1
    return {
        "systems": [{"id": s, "entities": len(r.store.entities(s))} for s in systems],
        "entity_count": sum(len(r.store.entities(s)) for s in systems),
        "profile_count": len(profiles),
        "event_count": len(events),
        "match_count": len(matches),
        "severity_breakdown": sev_counts,
        "category_breakdown": cat_counts,
        "tick_count": r.pipeline.tick_count,
        "live_ticks": r.live_ticks,
    }


@router.get("/engines")
def engines():
    r = rt()
    infos = r.pipeline.engine_info()
    by_layer: dict = {}
    for info in infos:
        by_layer.setdefault(info["layer"], []).append(info)
    return {"layers": by_layer, "count": len(infos),
            "last_tick_stats": r.pipeline.last_tick_stats}


@router.get("/catalog")
def catalog():
    path = _catalog_path()
    if not os.path.exists(path):
        return {"raw": [], "derived": []}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"raw": [], "derived": []}


@router.get("/signatures")
def signatures():
    r = rt()
    return {"primitives": r.sig_store.as_dicts(), "composite": r.composite_rules}


@router.get("/systems")
def systems():
    r = rt()
    return {"systems": r.store.systems()}


# --------------------------------------------------------------------------- #
# lib-3 v2 read helpers (minimal projection until the full API v2 lands)
# --------------------------------------------------------------------------- #
DRIFT_DETECTORS = ("cusum", "mcusum", "bocpd", "creep")


def _latest_vec(store, system: str, entity: str, name: str):
    """(ts, row) of the newest row of a vec ring, or (None, None)."""
    ts, M = store.vec_tail(system, entity, name, 1)
    if not len(ts):
        return None, None
    return float(ts[-1]), np.asarray(M[-1], dtype=np.float64)


def _latest_scalar(store, system: str, entity: str, name: str) -> float:
    _, row = _latest_vec(store, system, entity, name)
    if row is not None and row.size:
        return float(row[0])
    d = store.latest_derived(system, entity, name)
    try:
        return float(d.value) if d is not None else math.nan
    except (TypeError, ValueError):
        return math.nan


def _evidence_score(p: float) -> float:
    """-log10 of a p-value / expected-count mapped to [0, 1] (1e-6 -> 1)."""
    if not (p == p) or p <= 0.0:
        return 0.0 if not (p == p) else 1.0
    return float(min(1.0, max(0.0, -math.log10(min(p, 1.0)) / 6.0)))


def _anomaly_score(store, system: str, entity: str) -> float:
    """From behavior.e_day (expected equally extreme null ticks per entity-day,
    architecture section 4), else q_all (meta-calibrated p of the fusion)."""
    e_day = _latest_scalar(store, system, entity, "behavior.e_day")
    if e_day == e_day:
        return _evidence_score(e_day)
    return _evidence_score(_latest_scalar(store, system, entity, "behavior.q_all"))


def _drift_score(store, system: str, entity: str) -> float:
    """The strongest change-family accumulator (B14 cusum / mcusum / bocpd /
    creep): calibrated behavior.p where scored, else behavior.pm."""
    for name in ("behavior.p", "behavior.pm"):
        _, row = _latest_vec(store, system, entity, name)
        if row is None or row.size != len(DETECTORS):
            continue
        ps = [row[DETECTORS.index(d)] for d in DRIFT_DETECTORS]
        ps = [p for p in ps if p == p]
        if ps:
            return _evidence_score(min(ps))
    return 0.0


@router.get("/systems/{system}/entities")
def system_entities(system: str):
    r = rt()
    out = []
    for entity in r.store.entities(system):
        prof = r.store.profile(system, entity)
        recent_match = r.store.matches(system=system, entity=entity, limit=1)
        out.append({
            "entity": entity,
            "archetype": prof.archetype if prof else "",
            "archetype_confidence": prof.archetype_confidence if prof else 0.0,
            "separability": prof.separability if prof else 0.0,
            "stable": prof.stable if prof else False,
            "sample_count": prof.sample_count if prof else 0,
            "anomaly_score": round(_anomaly_score(r.store, system, entity), 4),
            "drift_score": round(_drift_score(r.store, system, entity), 4),
            "risk": _finite_or_none(_latest_scalar(r.store, system, entity, "behavior.risk")),
            "current_activity": recent_match[0].label if recent_match else "",
            "current_category": recent_match[0].category if recent_match else "",
        })
    out.sort(key=lambda x: max(x["anomaly_score"], x["drift_score"]), reverse=True)
    return {"system": system, "entities": out}


def _finite_or_none(x: float):
    return round(x, 4) if x == x and math.isfinite(x) else None


def _num(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return x if math.isfinite(x) else 0.0


@router.get("/systems/{system}/entities/{entity}")
def entity_detail(system: str, entity: str):
    r = rt()
    prof = r.store.profile(system, entity)
    if not prof:
        raise HTTPException(404, "no profile yet")
    names = list(prof.feature_names or FEATURE_NAMES_V2)
    _, z = _latest_vec(r.store, system, entity, "behavior.z")
    features = []
    for i, name in enumerate(names):
        cur = _num(prof.fingerprint[i]) if i < len(prof.fingerprint) else 0.0
        med = _num(prof.baseline_median[i]) if i < len(prof.baseline_median) else 0.0
        mad = _num(prof.baseline_mad[i]) if i < len(prof.baseline_mad) else 0.0
        zi = _num(z[i]) if z is not None and i < z.size else 0.0
        features.append({"name": name, "current": round(cur, 3),
                         "baseline": round(med, 3), "spread": round(mad, 3),
                         "z": round(zi, 2), "stable": round(med, 3)})
    return {
        "system": system, "entity": entity,
        "archetype": prof.archetype, "archetype_confidence": prof.archetype_confidence,
        "separability": prof.separability, "stable": prof.stable,
        "sample_count": prof.sample_count, "updated": prof.updated,
        "features": features,
        "seasonal": prof.seasonal,
        "events": to_jsonable(r.store.events(system=system, entity=entity, limit=30)),
        "matches": to_jsonable(r.store.matches(system=system, entity=entity, limit=30)),
    }


@router.get("/systems/{system}/entities/{entity}/metrics")
def entity_metric_names(system: str, entity: str):
    r = rt()
    return {"raw": r.store.raw_names(system, entity),
            "derived": r.store.derived_names(system, entity)}


@router.get("/systems/{system}/entities/{entity}/series")
def entity_series(system: str, entity: str, name: str = Query(...), limit: int = 200):
    r = rt()
    raw = r.store.raw_series(system, entity, name)
    if raw:
        pts = [{"ts": m.ts, "value": m.value if isinstance(m.value, (int, float)) else None}
               for m in raw[-limit:]]
        return {"name": name, "kind": "raw", "points": pts}
    der = r.store.derived_series(system, entity, name)
    pts = [{"ts": m.ts, "value": m.value} for m in der[-limit:]]
    return {"name": name, "kind": "derived", "points": pts}


@router.get("/events")
def events(system: Optional[str] = None, limit: int = 100):
    r = rt()
    return {"events": to_jsonable(r.store.events(system=system, limit=limit))}


@router.get("/matches")
def matches(system: Optional[str] = None, limit: int = 100):
    r = rt()
    return {"matches": to_jsonable(r.store.matches(system=system, limit=limit))}


@router.post("/step")
def step():
    r = rt()
    return {"stats": r.step_once(), "live_ticks": r.live_ticks}
