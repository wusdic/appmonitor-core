"""REST API — read models over the running pipeline.

Every endpoint reads the shared store the Runtime exposes; none reach into an
engine. The API is a thin projection layer, which is why swapping engines never
breaks it.
"""
from __future__ import annotations

import os
from typing import Optional

import yaml
from fastapi import APIRouter, HTTPException, Query

from ..pipeline.build import Runtime
from . import views as V
from .serialize import to_jsonable

router = APIRouter(prefix="/api")

# The Runtime is created and attached by main.py
RUNTIME: Optional[Runtime] = None

# "current activity" = a lib-4 match within this many seconds of STORE time
# (the newest tick), or within one tick when the cadence is slower
CURRENT_ACTIVITY_S = 45.0


def rt() -> Runtime:
    if RUNTIME is None:
        raise HTTPException(503, "runtime not started")
    return RUNTIME


def _catalog_path() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "catalog.yaml")


@router.get("/health")
def health():
    r = rt()
    now = V.store_now(r)
    tz = V.runtime_tz(r)
    return {"status": "ok", "warmed": r.warmed, "live_ticks": r.live_ticks,
            "tick_count": r.pipeline.tick_count, "tz": tz, "now": now,
            "now_local": V.iso(now, tz), "window_s": V.window_s(r)}


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
# Entities (legacy endpoints, migrated to lib-3 v2: docs/lib3/api_ui.md)
# --------------------------------------------------------------------------- #
@router.get("/systems/{system}/entities")
def system_entities(system: str):
    """Entities ranked by behavior.risk (B26). The legacy fields stay for one
    release but are computed the new way: anomaly_score from the latest
    q_all's e_day, drift_score from the change-family scores, archetype =
    class_path, separability = clip(1 - 2 EER_hard, 0, 1) (B15)."""
    r = rt()
    st = r.store
    tz = V.runtime_tz(r)
    with V.read_lock(r):
        now = V.store_now(r)
        dt = V.window_s(r)
        cutoff = now - max(CURRENT_ACTIVITY_S, dt)      # store time, not time.time()
        out = []
        for entity in st.entities(system):
            prof = st.profile(system, entity)
            extra = dict(prof.extra or {}) if prof else {}
            pg = V.peer_group(st, system, entity, extra)
            risk = V.risk_info(st, system, entity, extra)
            recent = st.matches(system=system, entity=entity, since=cutoff, limit=1)
            rg = V.regime_info(st, system, entity)
            out.append({
                "entity": entity,
                # legacy fields (kept for one release, computed the v2 way)
                "archetype": pg.get("class_path") or (prof.archetype if prof else ""),
                "archetype_confidence": V.rnd(pg.get("prob"), 4)
                if pg.get("prob") is not None else (prof.archetype_confidence if prof else 0.0),
                "separability": V.rnd(prof.separability, 4) if prof else None,
                "stable": bool(prof.stable) if prof else False,
                "sample_count": prof.sample_count if prof else 0,
                "anomaly_score": round(V.anomaly_score(st, system, entity, dt), 4),
                "drift_score": round(V.drift_score(st, system, entity), 4),
                "current_activity": recent[0].label if recent else "",
                "current_category": recent[0].category if recent else "",
                # v2
                "risk": risk["score"], "tier": risk["tier"], "trend": risk["trend"],
                "class_path": pg.get("class_path"), "role": pg.get("role"),
                "role_name": pg.get("role_name"), "role_prob": V.rnd(pg.get("prob"), 4),
                "open_incidents": V.open_incident_count(st, system, entity),
                "regime": rg.get("state"),
                "last_seen": st.last_seen(system, entity),
                "last_seen_local": V.iso(st.last_seen(system, entity), tz),
            })
    out.sort(key=lambda x: (x["risk"] is not None, x["risk"] or 0.0,
                            x["anomaly_score"], x["drift_score"]), reverse=True)
    return to_jsonable({"system": system, "tz": tz, "now": now, "now_local": V.iso(now, tz),
                        "entities": out})


@router.get("/systems/{system}/entities/{entity}")
def entity_detail(system: str, entity: str):
    """Profile, portrait summary, risk, identity, continuity and regime. The
    features carry the current value, the predictive p5/p50/p95 in natural
    units and the engine's own z / zr (B04), not a recomputed unfloored z."""
    r = rt()
    st = r.store
    tz = V.runtime_tz(r)
    with V.read_lock(r):
        prof = st.profile(system, entity)
        if not prof:
            raise HTTPException(404, "no profile yet")
        now = V.store_now(r)
        dt = V.window_s(r)
        extra = dict(prof.extra or {})
        pg = V.peer_group(st, system, entity, extra)
        por = extra.get("portrait") if isinstance(extra.get("portrait"), dict) else {}
        body = {
            "system": system, "entity": entity,
            "kind": "class" if entity.startswith("class:") else
                    ("system" if entity.startswith("__") else "entity"),
            "tz": tz, "now": now, "now_local": V.iso(now, tz),
            # legacy
            "archetype": pg.get("class_path") or prof.archetype,
            "archetype_confidence": V.rnd(pg.get("prob"), 4)
            if pg.get("prob") is not None else prof.archetype_confidence,
            "separability": V.rnd(prof.separability, 4), "stable": bool(prof.stable),
            "sample_count": prof.sample_count, "updated": prof.updated,
            "updated_local": V.iso(prof.updated, tz),
            "features": V.feature_rows(st, system, entity, extra, prof, dt),
            "seasonal": prof.seasonal,
            "events": [V.event_view(e, tz) for e in
                       st.events(system=system, entity=entity, limit=30)],
            "matches": to_jsonable(st.matches(system=system, entity=entity, limit=30)),
            # v2
            "class_path": pg.get("class_path"), "peer_group": pg,
            "risk": V.risk_info(st, system, entity, extra),
            "anomaly_score": round(V.anomaly_score(st, system, entity, dt), 4),
            "drift_score": round(V.drift_score(st, system, entity), 4),
            "portrait": {"text_zh": por.get("text_zh"), "text_en": por.get("text_en"),
                         "version": por.get("version"), "updated": por.get("updated"),
                         "diff": por.get("diff") or []},
            "identity": extra.get("identity") or {},
            "attribution": extra.get("attribution") or {},
            "continuity": extra.get("continuity") or {},
            "regime": V.regime_info(st, system, entity),
            "maturity": extra.get("maturity") or {},
            "calibration": extra.get("calibration") or {},
            "feedback": extra.get("feedback") or {},
            "open_incidents": V.open_incident_count(st, system, entity),
            "first_seen": st.first_seen(system, entity), "last_seen": st.last_seen(system, entity),
        }
    return to_jsonable(body)


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
        return to_jsonable({"name": name, "kind": "raw", "points": pts})
    der = r.store.derived_tail(system, entity, name, limit)
    pts = [{"ts": m.ts, "value": m.value} for m in der]
    return to_jsonable({"name": name, "kind": "derived", "points": pts})


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
