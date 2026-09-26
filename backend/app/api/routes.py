"""REST API — read models over the running pipeline.

Every endpoint reads the shared store the Runtime exposes; none reach into an
engine. The API is a thin projection layer, which is why swapping engines never
breaks it.
"""
from __future__ import annotations

import os
from typing import List, Optional

import yaml
from fastapi import APIRouter, HTTPException, Query

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


@router.get("/systems/{system}/entities")
def system_entities(system: str, recent_s: float = 45.0):
    r = rt()
    import time as _t
    cutoff = _t.time() - recent_s          # only reflect the current state
    out = []
    for entity in r.store.entities(system):
        prof = r.store.profile(system, entity)
        ev = [e for e in r.store.events(system=system, entity=entity, limit=40)
              if e.ts >= cutoff]
        latest_anom = next((e for e in ev if e.kind == "anomaly"), None)
        latest_drift = next((e for e in ev if e.kind == "drift"), None)
        recent_match = r.store.matches(system=system, entity=entity, limit=1)
        out.append({
            "entity": entity,
            "archetype": prof.archetype if prof else "",
            "archetype_confidence": prof.archetype_confidence if prof else 0.0,
            "separability": prof.separability if prof else 0.0,
            "stable": prof.stable if prof else False,
            "sample_count": prof.sample_count if prof else 0,
            "anomaly_score": latest_anom.score if latest_anom else 0.0,
            "drift_score": latest_drift.score if latest_drift else 0.0,
            "current_activity": recent_match[0].label if recent_match else "",
            "current_category": recent_match[0].category if recent_match else "",
        })
    out.sort(key=lambda x: max(x["anomaly_score"], x["drift_score"]), reverse=True)
    return {"system": system, "entities": out}


@router.get("/systems/{system}/entities/{entity}")
def entity_detail(system: str, entity: str):
    r = rt()
    prof = r.store.profile(system, entity)
    if not prof:
        raise HTTPException(404, "no profile yet")
    features = []
    stable_fp = prof.extra.get("stable_fingerprint", [])
    for i, name in enumerate(prof.feature_names):
        cur = prof.fingerprint[i] if i < len(prof.fingerprint) else 0.0
        med = prof.baseline_median[i] if i < len(prof.baseline_median) else 0.0
        mad = prof.baseline_mad[i] if i < len(prof.baseline_mad) else 0.0
        z = (cur - med) / mad if mad > 1e-6 else 0.0
        features.append({"name": name, "current": round(cur, 3),
                         "baseline": round(med, 3), "spread": round(mad, 3),
                         "z": round(z, 2),
                         "stable": round(stable_fp[i], 3) if i < len(stable_fp) else 0.0})
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
