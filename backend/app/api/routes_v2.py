"""REST API v2 (docs/lib3/api_ui.md): entity / class / incident / feedback /
operations endpoints over the lib-3 behaviour library.

Read endpoints are projections of the store (public MetricStore API plus the
lib/m_*.py accessors, see views.py); the only writes are analyst feedback
Labels (contract E), which B23 feedback folds on its next tick. All times are
epoch seconds plus an ISO string in ctx.config.tz ('*_local').

Defensive by design: engines refit on their own cadence and B29 explain may
not be registered, so a missing model or series yields null / [] / {} fields,
never a 500. An unknown system, entity, class, incident or event is a 404.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..engines.behavior.lib import m_class, m_feedback, m_governor, m_identity, m_vocab
from ..engines.behavior.lib.classkeys import SYSTEM_KEY, class_id, class_kind, is_class
from ..engines.behavior.lib.detectors import DETECTORS, FAMILIES
from ..models.schema import LABEL_SCOPES, LABEL_VERDICTS, Label
from . import routes
from . import views as V
from .serialize import to_jsonable

router = APIRouter(prefix="/api")

HOUR = 3600.0
DAY = 86400.0
DEFAULT_SCORES_SPAN_S = 24 * HOUR
DEFAULT_TIMELINE_SPAN_S = 30 * DAY
COHERENT_KINDS = ("system_shift", "coherent_shift", "class_shift", "class_adoption_risky")
WILSON_Z = 1.96


def rt():
    return routes.rt()


def _ok(body: Dict[str, Any]) -> Dict[str, Any]:
    return to_jsonable(body)


def _require_system(st, system: str) -> None:
    if system not in st.systems():
        raise HTTPException(404, f"unknown system {system!r}")


def _require_key(st, system: str, entity: str) -> None:
    """A real entity or a pseudo-entity (class / system key) of the system."""
    _require_system(st, system)
    if entity in st.entities(system) or entity in st.pseudo_entities(system):
        return
    if st.profile(system, entity) is not None:
        return
    raise HTTPException(404, f"unknown entity {system}/{entity}")


def _wilson(p: Optional[float], n: Optional[float]) -> Optional[List[float]]:
    """95 % Wilson score interval of a proportion p over n trials."""
    p, n = V.fnum(p), V.fnum(n)
    if p is None or n is None or n <= 0:
        return None
    p = min(1.0, max(0.0, p))
    z2 = WILSON_Z * WILSON_Z
    den = 1.0 + z2 / n
    c = (p + z2 / (2 * n)) / den
    h = WILSON_Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


# ======================================================================
# Entity
# ======================================================================
def _portrait_payload(pobj: Any) -> Dict[str, Any]:
    if not isinstance(pobj, dict):
        return {}
    return {"json": pobj.get("json") or {}, "text_zh": pobj.get("text_zh"),
            "text_en": pobj.get("text_en"), "diff": pobj.get("diff") or [],
            "version": pobj.get("version"), "updated": pobj.get("updated", pobj.get("ts"))}


def _current_workload(st, system: str, entity: str, js: Dict[str, Any], dt: float,
                      tz: str) -> Dict[str, Any]:
    """Current per-15-min value of each workload feature of the portrait
    (feature.nat is the raw per-tick count; bands are per 15 min). spec v2.1
    (cadence.md §9.5): when B01 writes grain rows, per_15min / per_hour are
    the live rolling Q / H values (feature.live.<g>), exact for every feature
    class, and `grains` carries both; otherwise v2's tick scaling."""
    from ..engines.behavior.lib.features import FEATURE_INDEX
    wl = js.get("workload") if isinstance(js.get("workload"), dict) else {}
    ts, nat = V.latest_vec(st, system, entity, "feature.nat")
    live = {g: V.latest_vec(st, system, entity, f"feature.live.{g}")[1] for g in ("h", "q")}
    out: Dict[str, Any] = {}
    for name in wl:
        i = FEATURE_INDEX.get(name)
        v = V.fnum(nat[i]) if nat is not None and i is not None and i < nat.size else None
        gv = {g: (V.fnum(lv[i]) if lv is not None and i is not None and i < lv.size else None)
              for g, lv in live.items()}
        if live["q"] is not None or live["h"] is not None:
            per15 = gv["q"]
        else:
            per15 = v * 900.0 / dt if v is not None and dt > 0 else None
        item = {"per_15min": V.rnd(per15, 3), "per_tick": V.rnd(v, 3)}
        if live["h"] is not None:
            item["per_hour"] = V.rnd(gv["h"], 3)
        if any(lv is not None for lv in live.values()):
            item["grains"] = {g: V.rnd(x, 3) for g, x in gv.items() if live[g] is not None}
        out[name] = item
    return {"ts": ts, "ts_local": V.iso(ts, tz), "window_s": dt, "features": out}


@router.get("/systems/{system}/entities/{entity}/portrait")
def entity_portrait(system: str, entity: str, version: Optional[int] = None):
    """The B30 portrait (json + zh/en narrative + diff) of an IP or a class,
    a given version from the kept history (last 12), the version list, and
    the 7x24 rhythm heatmap / current workload for the entity page."""
    r = rt()
    st = r.store
    tz = V.runtime_tz(r)
    with V.read_lock(r):
        _require_key(st, system, entity)
        now = V.store_now(r)
        extra = V.profile_extra(st, system, entity)
        cur = extra.get("portrait") if isinstance(extra.get("portrait"), dict) else {}
        versions = st.profile_versions(system, entity)
        vlist = [{"version": pv.version, "ts": pv.ts, "ts_local": V.iso(pv.ts, tz)}
                 for pv in versions]
        cur_v = cur.get("version")
        if version is not None and version != cur_v:
            hit = next((pv for pv in versions if pv.version == version), None)
            if hit is None:
                raise HTTPException(404, f"portrait version {version} not kept "
                                         f"(have {[v['version'] for v in vlist]})")
            por = _portrait_payload(hit.obj)
            por["updated"] = hit.ts
        else:
            por = _portrait_payload(cur)
        js = por.get("json") or {}
        cfg = V.runtime_config(r)
        body = {
            "system": system, "entity": entity, "tz": tz, "now": now,
            "version": por.get("version"), "current_version": cur_v,
            "versions": vlist, "portrait": por,
            "updated_local": V.iso(por.get("updated"), tz),
            "rhythm": V.rhythm_grid(st, system, entity, now, tz, cfg.get("calendar")),
            "current": _current_workload(st, system, entity, js, V.window_s(r), tz),
            "risk": V.risk_info(st, system, entity, extra),
        }
    return _ok(body)


@router.get("/systems/{system}/entities/{entity}/portrait/diff")
def entity_portrait_diff(system: str, entity: str,
                         from_: Optional[int] = Query(None, alias="from"),
                         to: Optional[int] = None):
    """Diff between two kept portrait versions (defaults: the previous and
    the current). Uses B30's own signature / diff functions, so the diff has
    the same semantics as the one the engine cuts versions on."""
    r = rt()
    st = r.store
    tz = V.runtime_tz(r)
    with V.read_lock(r):
        _require_key(st, system, entity)
        versions = st.profile_versions(system, entity)
        extra = V.profile_extra(st, system, entity)
        cur = extra.get("portrait") if isinstance(extra.get("portrait"), dict) else {}
        pool: Dict[Any, Dict[str, Any]] = {}
        for pv in versions:
            if isinstance(pv.obj, dict):
                pool.setdefault(pv.version, dict(pv.obj, ts=pv.ts))
        if cur.get("version") is not None:
            pool[cur.get("version")] = dict(cur, ts=cur.get("updated"))
        keys = sorted(k for k in pool if isinstance(k, (int, float)))
        if to is None:
            to = keys[-1] if keys else None
        if from_ is None:
            older = [k for k in keys if to is not None and k < to]
            from_ = older[-1] if older else to
        if from_ not in pool or to not in pool:
            raise HTTPException(404, f"portrait versions {from_!r} / {to!r} not kept "
                                     f"(have {keys})")
        a, b = pool[from_], pool[to]
        diff: List[Dict[str, Any]] = []
        method = "b30.diff_signatures"
        try:
            diff = V.portrait_diff(a, b)
        except Exception:
            method = "stored"
            diff = list(b.get("diff") or []) if from_ != to else []
        body = {"system": system, "entity": entity, "tz": tz, "from": from_, "to": to,
                "from_ts": a.get("ts"), "to_ts": b.get("ts"),
                "from_local": V.iso(a.get("ts"), tz), "to_local": V.iso(b.get("ts"), tz),
                "diff": diff, "method": method,
                "text_from": {"zh": a.get("text_zh"), "en": a.get("text_en")},
                "text_to": {"zh": b.get("text_zh"), "en": b.get("text_en")},
                "available": keys}
    return _ok(body)


def _timeline_item(it: Dict[str, Any], tz: str) -> Dict[str, Any]:
    typ = it.get("type")
    obj = it.get("item")
    ts = it.get("ts")
    out: Dict[str, Any] = {"ts": ts, "ts_local": V.iso(ts, tz), "type": typ}
    if typ == "event":
        out.update(id=obj.id, kind=obj.kind, severity=V.sev_name(obj.severity),
                   status=obj.status, description=obj.description,
                   incident_id=obj.incident_id or None, e_day=V.fnum(obj.e_day),
                   state=(obj.extra or {}).get("state") if isinstance(obj.extra, dict) else None)
    elif typ == "match":
        out.update(kind=obj.category, signature_id=obj.signature_id, label=obj.label,
                   severity=V.sev_name(obj.severity), confidence=V.rnd(obj.confidence, 3))
    elif typ == "incident":
        out.update(id=obj.id, kind="incident", severity=V.sev_name(obj.severity),
                   status=obj.status, opened=obj.opened, last_seen=obj.last_seen,
                   axes=list(obj.axes or []))
    elif typ == "profile_version":
        o = obj.obj if isinstance(getattr(obj, "obj", None), dict) else {}
        out.update(kind=o.get("kind") or "profile_version", version=obj.version,
                   diff=[d.get("field") for d in (o.get("diff") or []) if isinstance(d, dict)])
    elif typ == "risk":
        v = obj.value
        out.update(kind="risk", value=V.rnd(v.get("score") if isinstance(v, dict) else v, 2),
                   tier=V.tier_of(V.fnum(v.get("score") if isinstance(v, dict) else v)))
    return out


@router.get("/systems/{system}/entities/{entity}/timeline")
def entity_timeline(system: str, entity: str, since: Optional[float] = None, limit: int = 300):
    """Indexed store.timeline: events, matches, incidents, profile versions
    and risk band changes, newest first, plus regime / rollback markers."""
    r = rt()
    st = r.store
    tz = V.runtime_tz(r)
    with V.read_lock(r):
        _require_key(st, system, entity)
        now = V.store_now(r)
        since = now - DEFAULT_TIMELINE_SPAN_S if since is None else float(since)
        limit = max(1, min(int(limit), 2000))
        items = [_timeline_item(it, tz) for it in st.timeline(system, entity, since=since,
                                                              limit=limit)]
        regime = []
        try:
            for h in m_governor.descriptor(st, system, entity).get("history") or []:
                ts = V.fnum(h.get("ts"))
                if ts is not None and ts >= since:
                    regime.append(dict(h, ts_local=V.iso(ts, tz)))
        except Exception:
            pass
        body = {"system": system, "entity": entity, "tz": tz, "now": now, "since": since,
                "items": items, "regime_markers": regime}
    return _ok(body)


@router.get("/systems/{system}/entities/{entity}/scores")
def entity_scores(system: str, entity: str, since: Optional[float] = None,
                  max_points: int = 600):
    """Per-detector calibrated p (behavior.p, B24), p per family, q_inst,
    q_all, e_day, evidence (B25), trust / trust_prov (B28), risk (B26) and the
    regime series. Detectors never scored in the window are left out."""
    r = rt()
    st = r.store
    tz = V.runtime_tz(r)
    with V.read_lock(r):
        _require_key(st, system, entity)
        now = V.store_now(r)
        since = now - DEFAULT_SCORES_SPAN_S if since is None else float(since)
        mp = max(10, min(int(max_points), 5000))
        ts, M = st.vec_since(system, entity, "behavior.p", since)
        p: Dict[str, Any] = {"ts": [], "values": {}}
        if len(ts):
            idx = V._thin(len(ts), mp)
            M = np.asarray(M[idx], dtype=np.float64)
            p["ts"] = [float(t) for t in ts[idx]]
            for j, d in enumerate(DETECTORS):
                if j < M.shape[1] and np.isfinite(M[:, j]).any():
                    p["values"][d] = [V.fnum(v) for v in M[:, j]]
        pf_rows = V.dict_series(st, system, entity, "behavior.p_family", since, mp)
        fams = [f for f in FAMILIES if any(f in v for _, v in pf_rows)]
        p_family = {"ts": [t for t, _ in pf_rows],
                    "values": {f: [V.fnum(v.get(f)) for _, v in pf_rows] for f in fams}}
        reg_rows = V.dict_series(st, system, entity, "behavior.regime", since, mp)
        regime = {"ts": [t for t, _ in reg_rows],
                  "state": [v.get("state") for _, v in reg_rows],
                  "type": [v.get("type") for _, v in reg_rows],
                  "p_legit": [V.fnum(v.get("p_legit")) for _, v in reg_rows],
                  "version": [v.get("version") for _, v in reg_rows]}
        body = {"system": system, "entity": entity, "tz": tz, "now": now, "since": since,
                "detectors": list(DETECTORS), "families": list(FAMILIES),
                "p": p, "p_family": p_family}
        for name, key in (("behavior.q_inst", "q_inst"), ("behavior.q_all", "q_all"),
                          ("behavior.e_day", "e_day"), ("behavior.evidence", "evidence"),
                          ("behavior.trust", "trust"), ("behavior.trust_prov", "trust_prov"),
                          ("behavior.risk", "risk")):
            body[key] = V.scalar_series(st, system, entity, name, since, mp)
        body["regime"] = regime
    return _ok(body)


@router.get("/systems/{system}/entities/{entity}/identity")
def entity_identity(system: str, entity: str, since: Optional[float] = None,
                    max_points: int = 600):
    """B15 identifiability (confusion row, traits, recall and EER with 95 %
    Wilson CIs over the held-out windows, T99) and the B16 attribution
    posterior series (behavior.id)."""
    r = rt()
    st = r.store
    tz = V.runtime_tz(r)
    with V.read_lock(r):
        _require_key(st, system, entity)
        now = V.store_now(r)
        since = now - DEFAULT_SCORES_SPAN_S if since is None else float(since)
        model = m_identity.get(st, system)
        stats = m_identity.stats(model, entity) if model else {}
        n = V.fnum(stats.get("n_windows"))
        extra = V.profile_extra(st, system, entity)
        conf = m_identity.confusion(model, entity) if model else {}
        row = sorted(({"entity": k, "share": V.rnd(v, 4)} for k, v in conf.items()),
                     key=lambda x: -(x["share"] or 0.0))
        # EER is estimated from the 3 nearest impostors' held-out windows; the
        # CI treats it as a proportion over the entity's windows (approximate)
        eer = V.fnum(stats.get("eer_hard"))
        rows = V.dict_series(st, system, entity, "behavior.id", since, max(10, int(max_points)))
        post = {"ts": [t for t, _ in rows],
                "posterior_self": [V.fnum(v.get("posterior_self")) for _, v in rows],
                "p_unknown": [V.fnum(v.get("p_unknown")) for _, v in rows],
                "best_other": [(v.get("best_other") or {}).get("entity")
                               if isinstance(v.get("best_other"), dict) else None for _, v in rows],
                "best_other_posterior": [V.fnum((v.get("best_other") or {}).get("posterior"))
                                         if isinstance(v.get("best_other"), dict) else None
                                         for _, v in rows],
                "cusum_other": [V.fnum(v.get("cusum_other")) for _, v in rows],
                "cusum_new": [V.fnum(v.get("cusum_new")) for _, v in rows]}
        body = {
            "system": system, "entity": entity, "tz": tz, "now": now,
            "fitted": bool(model and m_identity.is_fitted(model)),
            "model_version": m_identity.version(model) if model else None,
            "fitted_ts": (model or {}).get("fitted_ts"),
            "enrolled": bool(stats),
            "recall1": V.rnd(stats.get("recall1")), "recallK": V.rnd(stats.get("recallK")),
            "recall1_ci": _wilson(stats.get("recall1"), n),
            "eer_hard": V.rnd(eer), "eer_ci": _wilson(eer, n),
            "separability": V.rnd(stats.get("separability")),
            "t99": V.rnd(stats.get("t99"), 3), "n_windows": n,
            "confusion_row": row,
            "confusable_with": m_identity.confusable_with(model, entity) if model else [],
            "nearest": m_identity.nearest(model, entity) if model else [],
            "anonymity_set": m_identity.anonymity_set(model, entity) if model else [entity],
            "traits": m_identity.distinctive(model, entity) if model else {},
            "modality_share": m_identity.modality_share(model, entity) if model else {},
            "attribution": extra.get("attribution") or {},
            "continuity": extra.get("continuity") or {},
            "posterior": post,
        }
        # identifiability CI as the portrait shows it (B30, Wilson on 1 - 2 EER)
        por = extra.get("portrait") if isinstance(extra.get("portrait"), dict) else {}
        pid = ((por.get("json") or {}).get("identity") or {}) if por else {}
        body["identifiability"] = V.rnd(pid.get("identifiability", stats.get("separability")))
        body["identifiability_ci"] = pid.get("identifiability_ci")
    return _ok(body)


# ======================================================================
# Classes
# ======================================================================
@router.get("/systems/{system}/classes")
def system_classes(system: str):
    """Role, sub, static and pool classes of a system with members, risk and
    open class incidents, plus the super -> role -> sub tree."""
    r = rt()
    st = r.store
    tz = V.runtime_tz(r)
    with V.read_lock(r):
        _require_system(st, system)
        now = V.store_now(r)
        classes = V.class_catalog(st, system)
        for c in classes:
            key = c["key"]
            if key.startswith("class:"):
                risk = V.risk_info(st, system, key)
                c["risk"], c["tier"] = risk["score"], risk["tier"]
                c["open_incidents"] = len([i for i in st.incidents(system=system, entity=key,
                                                                   status=V.LIVE_STATUSES)
                                           if i.entity == key])
                c["has_portrait"] = bool(V.profile_extra(st, system, key).get("portrait"))
            else:
                c["risk"], c["tier"], c["open_incidents"], c["has_portrait"] = None, None, 0, False
            mr = [V.latest_scalar(st, system, ip, "behavior.risk") for ip in c["members"]]
            mr = [x for x in mr if x is not None]
            c["member_risk_max"] = V.rnd(max(mr), 2) if mr else None
        sys_risk = V.risk_info(st, system, SYSTEM_KEY)
        body = {"system": system, "tz": tz, "now": now, "classes": classes,
                "tree": V.class_tree(classes), "system_risk": sys_risk,
                "model_version": m_class.version(st)}
    return _ok(body)


@router.get("/systems/{system}/classes/{cid:path}")
def class_detail(system: str, cid: str):
    """Class portrait, B18 class_monitor bands, active-fraction heatmap,
    adoption history, identifiability, lineage and version."""
    r = rt()
    st = r.store
    tz = V.runtime_tz(r)
    with V.read_lock(r):
        _require_system(st, system)
        now = V.store_now(r)
        key = V.resolve_class_key(cid)
        catalog = {c["key"]: c for c in V.class_catalog(st, system)}
        cat = catalog.get(key)
        if cat is None and key not in st.pseudo_entities(system):
            raise HTTPException(404, f"unknown class {system}/{cid}")
        cat = dict(cat or {"key": key, "kind": class_kind(key), "id": class_id(key),
                           "name": class_id(key), "members": [], "n_members": 0,
                           "member_probs": {}, "parent": None})
        m = m_class.get(st) or {}
        extra = V.profile_extra(st, system, key) if key.startswith("class:") else {}
        por = _portrait_payload(extra.get("portrait"))
        js = por.get("json") or {}
        cm = extra.get("class_monitor") if isinstance(extra.get("class_monitor"), dict) else {}
        # lineage / version from model.class
        lineage, version, desc = [], None, {}
        if cat["kind"] == "role":
            role = (m.get("roles") or {}).get(cat["id"]) or {}
            lineage, version = list(role.get("lineage") or []), role.get("version")
            desc = {k: role.get(k) for k in ("super", "A", "d90", "medoid", "born", "name_parts")}
        elif cat["kind"] == "sub":
            sub = (m.get("subs") or {}).get(cat["id"]) or {}
            lineage, version = list(sub.get("lineage") or []), sub.get("version")
            desc = {k: sub.get(k) for k in ("role", "medoid", "d90")}
        # members with membership bars, risk and outlier flags
        outliers = {o.get("ip") if isinstance(o, dict) else o
                    for o in (js.get("outlier_members") or [])}
        assign = m.get("assign") or {}
        members = []
        for ip in cat["members"]:
            a = assign.get(f"{system}|{ip}") or {}
            risk = V.risk_info(st, system, ip)
            members.append({"ip": ip, "prob": V.rnd(a.get("prob")), "sub": a.get("sub"),
                            "class_path": a.get("class_path"), "risk": risk["score"],
                            "tier": risk["tier"], "outlier": ip in outliers,
                            "open_incidents": V.open_incident_count(st, system, ip)})
        members.sort(key=lambda x: -(x["prob"] or 0.0))
        # adoption history (B08 class vocab adoption records) with risk flags
        adoption = []
        if key.startswith("class:"):
            mask = V.mask_template
            try:
                for rec in m_vocab.adoption_records(st, system, key)[:50]:
                    flags = rec.get("flags") or {}
                    val = rec.get("value")
                    adoption.append({
                        "dim": rec.get("dim"), "value": mask(val) if rec.get("dim") == "tmpl" else val,
                        "first_ts": rec.get("first_ts"), "last_ts": rec.get("last_ts"),
                        "first_local": V.iso(rec.get("first_ts"), tz),
                        "adopted": bool(rec.get("adopted")),
                        "n_members": len(rec.get("members") or {}),
                        "members": sorted((rec.get("members") or {}).keys()),
                        "flags": flags, "risky": any(bool(v) for v in flags.values())})
            except Exception:
                adoption = []
            if not adoption and isinstance(js.get("adoption"), dict):
                for h in js["adoption"].get("history") or []:
                    if isinstance(h, dict):
                        flags = h.get("flags") or {}
                        adoption.append(dict(h, first_local=V.iso(h.get("first_ts"), tz),
                                             risky=any(bool(v) for v in flags.values())))
        incs = [V.incident_summary(i, tz) for i in st.incidents(system=system, entity=key)
                if i.entity == key][:50] if key.startswith("class:") else []
        idm = m_identity.get(st, system)
        ident = js.get("identity") if isinstance(js.get("identity"), dict) else {}
        if not ident and idm and key.startswith("class:"):
            ident = m_identity.class_stats(idm, key)
        body = {
            "system": system, "key": key, "kind": cat["kind"], "id": cat["id"],
            "name": cat.get("name"), "parent": cat.get("parent"), "tz": tz, "now": now,
            "version": version, "class_model_version": m_class.version(st),
            "lineage": lineage, "descriptor": desc,
            "members": members, "n_members": len(members),
            "portrait": por, "portrait_updated_local": V.iso(por.get("updated"), tz),
            "bands": js.get("bands") or (cm.get("aggregate") if cm else None),
            "workload": js.get("workload"),
            "active_frac_heatmap": js.get("active_frac_heatmap")
            or ({"by_bin48": cm.get("active_frac_by_bin")} if cm.get("active_frac_by_bin") else None),
            "class_monitor": {k: cm.get(k) for k in ("last", "n_eff", "version", "updated")} if cm else {},
            "adoption": adoption,
            "adoption_rate": (js.get("adoption") or {}).get("rate") if isinstance(js.get("adoption"), dict) else None,
            "distinctive_tokens": js.get("distinctive_tokens") or [],
            "common_tokens": js.get("common_tokens") or [],
            "outlier_members": js.get("outlier_members") or [],
            "identifiability": ident,
            "risk": V.risk_info(st, system, key) if key.startswith("class:") else {},
            "incidents": incs,
            "rhythm": V.rhythm_grid(st, system, key, now, tz,
                                    V.runtime_config(r).get("calendar"))
            if key.startswith("class:") else None,
            "common": {g: v for g in ("volume", "transport", "app_error", "probe")
                       for _, v in [V.latest_dict(st, system, key, f"behavior.common.{g}")] if v},
        }
    return _ok(body)


# ======================================================================
# Incidents
# ======================================================================
@router.get("/incidents")
def incidents(system: Optional[str] = None, entity: Optional[str] = None,
              status: Optional[str] = None, severity: Optional[str] = None,
              kind: Optional[str] = None, limit: int = 200):
    """Incident queue, newest activity first. status / severity accept a
    comma list; kind is 'entity' or 'class'."""
    r = rt()
    st = r.store
    tz = V.runtime_tz(r)
    if kind is not None and kind not in ("entity", "class"):
        raise HTTPException(422, "kind must be 'entity' or 'class'")
    with V.read_lock(r):
        now = V.store_now(r)
        sts = [x for x in (status or "").split(",") if x] or None
        sevs = {x.lower() for x in (severity or "").split(",") if x}
        rows = []
        for inc in st.incidents(system=system, entity=entity, status=sts):
            if sevs and V.sev_name(inc.severity) not in sevs:
                continue
            is_cls = is_class(inc.entity or "")
            if kind == "class" and not is_cls or kind == "entity" and is_cls:
                continue
            rows.append(V.incident_summary(inc, tz))
        total = len(rows)
        counts: Dict[str, Dict[str, int]] = {"status": {}, "severity": {}}
        for x in rows:
            counts["status"][x["status"]] = counts["status"].get(x["status"], 0) + 1
            counts["severity"][x["severity"]] = counts["severity"].get(x["severity"], 0) + 1
        rows = rows[:max(1, min(int(limit), 2000))]
        body = {"tz": tz, "now": now, "count": total, "counts": counts, "incidents": rows}
    return _ok(body)


@router.get("/incidents/{incident_id}")
def incident_detail(incident_id: str):
    r = rt()
    st = r.store
    tz = V.runtime_tz(r)
    with V.read_lock(r):
        inc = st.get_incident(incident_id)
        if inc is None:
            raise HTTPException(404, f"unknown incident {incident_id!r}")
        now = V.store_now(r)
        fam, axis = V.evidence_bars(inc)
        ex = inc.explanation if isinstance(inc.explanation, dict) else {}
        campaign = None
        if inc.campaign_id:
            peers = [i for i in st.incidents(system=inc.system) if i.campaign_id == inc.campaign_id]
            campaign = {"id": inc.campaign_id,
                        "incidents": [V.incident_summary(i, tz) for i in peers]}
        children = [V.incident_summary(i, tz) for i in st.incidents(system=inc.system)
                    if i.parent_id == inc.id]
        parent = st.get_incident(inc.parent_id) if inc.parent_id else None
        evidence = []
        for ev in inc.evidence or []:
            if isinstance(ev, dict):
                evidence.append(dict(ev, ts_local=V.iso(ev.get("ts"), tz)))
        labels = [V.label_view(lb, tz) for lb in st.labels(system=inc.system)
                  if lb.target_type == "incident" and lb.target_id == inc.id]
        ev_ids = [ev.get("event_id") for ev in inc.evidence or [] if isinstance(ev, dict)
                  and ev.get("event_id")]
        events = []
        for eid in ev_ids[-50:]:
            e = st.get_event(eid)
            if e is not None:
                events.append(V.event_view(e, tz))
        body = V.incident_summary(inc, tz)
        body.update({
            "tz": tz, "now": now,
            "evidence": evidence,
            "evidence_by_family": fam, "evidence_by_axis": axis,
            "explanation": ex, "explanation_available": bool(ex),
            "attributions": ex.get("attributions") or [],
            "counterfactual": V.counterfactual(inc),
            "campaign": campaign,
            "parent": V.incident_summary(parent, tz) if parent is not None else None,
            "children": children, "labels": labels, "events": events,
            "entity_risk": V.risk_info(st, inc.system, inc.entity),
            "feedback_options": {"verdicts": list(LABEL_VERDICTS), "scopes": list(LABEL_SCOPES)},
        })
    return _ok(body)


# ======================================================================
# Feedback (writes Labels; B23 folds them on its next tick)
# ======================================================================
class FeedbackBody(BaseModel):
    verdict: str
    scope: str = "this"
    ttl_s: Optional[float] = Field(None, ge=0)
    note: str = ""
    analyst: str = ""


def _validate(fb: FeedbackBody) -> None:
    if fb.verdict not in LABEL_VERDICTS:
        raise HTTPException(422, f"verdict must be one of {list(LABEL_VERDICTS)}")
    if fb.scope not in LABEL_SCOPES:
        raise HTTPException(422, f"scope must be one of {list(LABEL_SCOPES)}")


def _add_label(r, lb: Label) -> Dict[str, Any]:
    tz = V.runtime_tz(r)
    with V.read_lock(r):
        lid = r.store.add_label(lb)
    return {"ok": True, "label_id": lid, "label": V.label_view(lb, tz)}


@router.post("/incidents/{incident_id}/feedback")
def incident_feedback(incident_id: str, fb: FeedbackBody = Body(...)):
    """Analyst verdict on an incident -> Label(target_type='incident') with
    t0 / t1 = the incident's span. The label's ts is store time, so B23's
    fold windows and B27's 'closed by label' use the pipeline clock."""
    r = rt()
    _validate(fb)
    inc = r.store.get_incident(incident_id)
    if inc is None:
        raise HTTPException(404, f"unknown incident {incident_id!r}")
    lb = Label(system=inc.system, entity=inc.entity, target_type="incident",
               target_id=inc.id, verdict=fb.verdict, scope=fb.scope,
               t0=float(inc.opened) if inc.opened else None,
               t1=float(inc.last_seen) if inc.last_seen else None,
               ttl_s=fb.ttl_s, analyst=fb.analyst, note=fb.note, ts=V.store_now(r))
    return _ok(_add_label(r, lb))


@router.post("/events/{event_id}/feedback")
def event_feedback(event_id: str, fb: FeedbackBody = Body(...)):
    r = rt()
    _validate(fb)
    ev = r.store.get_event(event_id)
    if ev is None:
        raise HTTPException(404, f"unknown event {event_id!r}")
    win = ev.window if isinstance(ev.window, (list, tuple)) and len(ev.window) == 2 else None
    lb = Label(system=ev.system, entity=ev.entity, target_type="event", target_id=ev.id,
               verdict=fb.verdict, scope=fb.scope,
               t0=float(win[0]) if win else float(ev.ts), t1=float(win[1]) if win else float(ev.ts),
               ttl_s=fb.ttl_s, analyst=fb.analyst, note=fb.note, ts=V.store_now(r))
    return _ok(_add_label(r, lb))


@router.get("/label-queue")
def label_queue(system: Optional[str] = None, limit: int = 50):
    """B23's active-learning queue (held > 14 d, highest risk, most
    uncertain) with incident summaries, B28's drifting-too-long keys and the
    most recent labels."""
    r = rt()
    st = r.store
    tz = V.runtime_tz(r)
    with V.read_lock(r):
        now = V.store_now(r)
        queue = []
        try:
            q = m_feedback.label_queue(st, system)
        except Exception:
            q = []
        for it in q:
            it = dict(it)
            inc = st.get_incident(str(it.get("incident_id"))) if it.get("incident_id") else None
            it["incident"] = V.incident_summary(inc, tz) if inc is not None else None
            queue.append(it)
        try:
            gov = m_governor.label_queue(st, system)
        except Exception:
            gov = []
        for g in gov:
            g["since_local"] = V.iso(g.get("since"), tz)
        labels = [V.label_view(lb, tz) for lb in st.labels(system=system)[:max(1, int(limit))]]
        body = {"tz": tz, "now": now, "queue": queue, "governor": gov, "labels": labels,
                "verdicts": list(LABEL_VERDICTS), "scopes": list(LABEL_SCOPES)}
    return _ok(body)


# ======================================================================
# System view
# ======================================================================
@router.get("/systems/{system}/summary")
def system_summary(system: str, since: Optional[float] = None):
    """System view: common-mode state per group (system and class tiers),
    coherent shifts, the campaign graph, system risk and a health digest."""
    r = rt()
    st = r.store
    tz = V.runtime_tz(r)
    with V.read_lock(r):
        _require_system(st, system)
        now = V.store_now(r)
        since = now - DAY if since is None else float(since)
        common: Dict[str, Any] = {}
        keys = [SYSTEM_KEY] + [k for k in st.pseudo_entities(system) if is_class(k)]
        for key in keys:
            row = {}
            for g in ("volume", "transport", "app_error", "probe"):
                t, v = V.latest_dict(st, system, key, f"behavior.common.{g}")
                if v:
                    row[g] = dict(v, ts=t)
            if row:
                common[key] = row
        # common-mode history (system tier): L and the up/down fractions
        hist: Dict[str, Any] = {}
        for g in ("volume", "transport", "app_error", "probe"):
            rows = V.dict_series(st, system, SYSTEM_KEY, f"behavior.common.{g}", since, 400)
            if rows:
                hist[g] = {"ts": [t for t, _ in rows], "L": [V.fnum(v.get("L")) for _, v in rows],
                           "frac_up": [V.fnum(v.get("frac_up")) for _, v in rows],
                           "frac_down": [V.fnum(v.get("frac_down")) for _, v in rows],
                           "dir": [v.get("dir") for _, v in rows]}
        shifts = [V.event_view(e, tz) for e in st.events(system=system, since=since,
                                                          kinds=COHERENT_KINDS, limit=100)]
        # campaign graph: nodes = incidents (live or campaign members), edges
        # = same campaign (star to the oldest) and common-mode parent -> child
        incs = st.incidents(system=system, since=since)
        nodes, edges, campaigns = [], [], {}
        for i in incs:
            if i.campaign_id or i.parent_id or i.status in V.LIVE_STATUSES:
                nodes.append(V.incident_summary(i, tz))
            if i.campaign_id:
                campaigns.setdefault(i.campaign_id, []).append(i)
            if i.parent_id:
                edges.append({"from": i.parent_id, "to": i.id, "type": "common_mode"})
        camp_out = []
        for cid, members in campaigns.items():
            members.sort(key=lambda x: x.opened)
            root = members[0]
            for mbr in members[1:]:
                edges.append({"from": root.id, "to": mbr.id, "type": "campaign"})
            camp_out.append({"id": cid, "incidents": [x.id for x in members],
                             "entities": sorted({x.entity for x in members}),
                             "opened": root.opened, "opened_local": V.iso(root.opened, tz),
                             "axes": sorted({a for x in members for a in (x.axes or [])})})
        _, eh = V.latest_dict(st, system, SYSTEM_KEY, "ops.engine_health")
        body = {"system": system, "tz": tz, "now": now, "since": since,
                "risk": V.risk_info(st, system, SYSTEM_KEY),
                "risk_series": V.scalar_series(st, system, SYSTEM_KEY, "behavior.risk", since, 400),
                "common": common, "common_history": hist, "coherent_shifts": shifts,
                "campaign_graph": {"nodes": nodes, "edges": edges}, "campaigns": camp_out,
                "engine_health": eh,
                "entities": len(st.entities(system)),
                "open_incidents": len(st.incidents(system=system, status=V.LIVE_STATUSES))}
    return _ok(body)


# ======================================================================
# Operations
# ======================================================================
@router.get("/detectors/health")
def detectors_health():
    """Per engine: last run / error, error count and the staleness of each
    series it declares (now - last_write_ts). Per detector and system: KS D,
    realised rate against the e_day budget, weight_mult (B24 calib_health)
    and the share of the keys running the detector (entities and class keys
    that scored it or marked it degraded at their last scored tick) whose run
    was degraded, with the causes by kind (lib/emit: stale, producer_error,
    unscorable, insufficient_support, fallback, provisional)."""
    r = rt()
    st = r.store
    tz = V.runtime_tz(r)
    with V.read_lock(r):
        now = V.store_now(r)
        health = st.health()
        infos = {}
        try:
            infos = {i["name"]: i for i in r.pipeline.engine_info()}
        except Exception:
            pass
        keys = V.all_keys(st)
        engines = []
        for name in sorted(set(health) | set(infos), key=lambda n: (
                ["raw", "derived", "behavior", "signature"].index(
                    (infos.get(n) or health.get(n) or {}).get("layer", "behavior"))
                if (infos.get(n) or health.get(n) or {}).get("layer") in
                ("raw", "derived", "behavior", "signature") else 9, n)):
            h = dict(health.get(name) or {})
            info = infos.get(name) or {}
            series = []
            for sname in V.produced_series(info.get("produces") or []):
                lw = V.last_write_any(st, sname, keys)
                series.append({"name": sname, "last_write_ts": lw,
                               "staleness_s": V.rnd(now - lw, 1) if lw is not None else None})
            last_run = h.get("last_run", info.get("last_run"))
            engines.append({
                "name": name, "layer": h.get("layer", info.get("layer")),
                "ok": h.get("ok", not bool(info.get("last_error"))),
                "last_run": last_run, "last_run_local": V.iso(last_run, tz),
                "last_error": h.get("last_error", info.get("last_error")) or None,
                "error_count": int(h.get("error_count", info.get("error_count", 0)) or 0),
                "last_error_ts": h.get("last_error_ts", info.get("last_error_ts")),
                "last_error_local": V.iso(h.get("last_error_ts", info.get("last_error_ts")), tz),
                "traceback": h.get("traceback") or None,
                "duration_ms": h.get("duration_ms", info.get("duration_ms")),
                "runs": h.get("runs"), "series": series,
                "max_staleness_s": max((x["staleness_s"] for x in series
                                        if x["staleness_s"] is not None), default=None),
            })
        fam = V.detector_family_map()
        detectors: Dict[str, List[Dict[str, Any]]] = {}
        for s in st.systems():
            _, ch = V.latest_dict(st, s, SYSTEM_KEY, "behavior.calib_health")
            deg = V.degraded_counts(st, s)
            rows = []
            for d in DETECTORS:
                c = ch.get(d) if isinstance(ch.get(d), dict) else {}
                dc = deg.get(d) or {}
                n_run = int(dc.get("n", 0))
                n_deg = int(dc.get("degraded", 0))
                rows.append({"detector": d, "family": fam.get(d),
                             "ks": V.rnd(c.get("ks")), "rate_ratio": V.rnd(c.get("rate_ratio")),
                             "weight_mult": V.rnd(c.get("weight_mult")),
                             "n": c.get("n"), "expected": V.rnd(c.get("expected")),
                             "n_keys": n_run,
                             "degraded_share": V.rnd(n_deg / n_run) if n_run else None,
                             "degraded_causes": dict(dc.get("causes") or {}),
                             "monitored": bool(c)})
            detectors[s] = rows
        body = {"tz": tz, "now": now, "now_local": V.iso(now, tz),
                "engines": engines, "detectors": detectors,
                "n_errors": sum(1 for e in engines if e["last_error"]),
                "error_total": sum(e["error_count"] for e in engines)}
    return _ok(body)


def _eval_report_paths() -> List[str]:
    root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    env = os.environ.get("APPMON_EVAL_REPORT")
    paths = [env] if env else []
    paths += [os.path.join(root, "reports", "eval_report.json"),
              os.path.join(root, "eval_out", "eval_report.json")]
    return [os.path.normpath(p) for p in paths]


@router.get("/eval/report")
def eval_report():
    """reports/eval_report.json (scripts/evaluate.py output) if present
    (APPMON_EVAL_REPORT overrides the path; eval_out/ is the script's
    default), else a 404 JSON body."""
    tried = _eval_report_paths()
    for p in tried:
        if p and os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError) as exc:
                return JSONResponse(status_code=500, content={
                    "error": "eval report unreadable", "path": p, "detail": str(exc)})
            if isinstance(data, dict):
                data.setdefault("_source", p)
            return to_jsonable(data)
    return JSONResponse(status_code=404, content={
        "error": "no eval report", "detail": "run scripts/evaluate.py --out reports",
        "searched": tried})


@router.get("/store/memory")
def store_memory():
    r = rt()
    rep = r.store.memory_report()
    tz = V.runtime_tz(r)
    now = V.store_now(r)
    return _ok({"tz": tz, "now": now, "report": rep, **{k: v for k, v in rep.items()}})
