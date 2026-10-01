"""REST API v3: the progressive profile core ("画像模式", docs/lib3/progressive.md
§9.4, requirement S11/S14/S15/S18-S20).

Read-only projections of the P engines' models (progressive_views.py) plus
one write, the operator's name for a learned group (persisted into the
runtime config `who_group_names`, which P11 matches on its next daily run).

  GET  /api/v3/status                                  core on?, registry, systems, groups
  GET  /api/v3/systems                                 per system: nodes by state, statements, who mode
  GET  /api/v3/systems/{s}/view                        system view: actions -> who -> when -> content -> workflow
  GET  /api/v3/systems/{s}/precision                   precision over time (live + measured)
  GET  /api/v3/systems/{s}/lattice                     pattern tree (bounded BFS)
  GET  /api/v3/systems/{s}/entities/{ip}/view          IP view: inherited group pattern, exceptions, bindings
  GET  /api/v3/systems/{s}/entities/{ip}/facets        IP facet tree
  GET  /api/v3/systems/{s}/facets                      system facet tree (P13)
  GET  /api/v3/systems/{s}/strategy                    P12 characterisation, chosen engines
  GET  /api/v3/systems/{s}/attributes                  P02 registry + P05 roles
  GET  /api/v3/groups                                  P11 groups
  GET  /api/v3/groups/{g}/view                         group view: systems -> actions + negatives
  GET  /api/v3/groups/{g}/facets                       group facet tree
  POST /api/v3/groups/{g}/name                         operator name for a group
  GET  /api/v3/patterns/{pid}                          one lattice node: constraints, lineage, drift, violations
  GET  /api/v3/violations                              typed pattern violations
  GET  /api/v3/budget                                  P15 budget, ladder, usage, series

Conventions as v2: strict JSON (NaN -> null), times as epoch seconds plus
'*_local' ISO strings in config tz, unknown system / group / pattern -> 404,
the progressive core not running -> 200 with empty payloads (status says why).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field

from . import progressive_views as PV
from . import routes
from . import views as V
from .serialize import to_jsonable

router = APIRouter(prefix="/api/v3")


def rt():
    return routes.rt()


def _ok(body: Dict[str, Any]) -> Dict[str, Any]:
    return to_jsonable(body)


def _ctx(r: Any):
    return r.store, V.runtime_config(r), V.store_now(r), V.runtime_tz(r)


def _require_system(st: Any, s: str) -> None:
    if s not in PV.known_systems(st):
        raise HTTPException(404, f"unknown system {s!r}")


def _registry_mode(r: Any) -> str:
    m = getattr(r, "registry_mode", None)
    if m:
        return str(m)
    from ..pipeline.build import registry_mode
    try:
        return registry_mode(None, V.runtime_config(r))
    except Exception:
        return "full"


# ===================================================================== status
@router.get("/status")
def status():
    """Is the core running, with which engine set, and what it has learned."""
    r = rt()
    st, cfg, now, tz = _ctx(r)
    with V.read_lock(r):
        mode = _registry_mode(r)
        names = [i.get("name") for i in r.pipeline.engine_info()] if getattr(r, "pipeline", None) else []
        from ..engines.behavior.resource_governor import P_ENGINES
        p_eng = [n for n in names if n in P_ENGINES]
        systems = PV.systems_index(st, cfg, now)
        groups = PV.groups_index(st, members_max=0)
        enabled = bool(((cfg.get("progressive") or {}).get("enabled")) or p_eng)
        present = PV.core_present(st)
    hint_zh = hint_en = None
    if not p_eng:
        hint_zh = ("渐进画像核心未运行：以 APPMON_PROGRESSIVE=decision 启动（组织数据包 O，"
                   "详见 docs/lib3/progressive.md §9.4）")
        hint_en = ("the progressive core is not running: start with APPMON_PROGRESSIVE=decision "
                   "(organisation pack O, docs/lib3/progressive.md §9.4)")
    elif not present:
        hint_zh, hint_en = "渐进画像核心预热中，尚无模式树", "the progressive core is warming up; no pattern tree yet"
    return _ok({"enabled": enabled, "running": bool(p_eng), "present": present, "registry_mode": mode,
                "pack": getattr(r, "pack_name", None), "engines": p_eng, "now": now,
                "now_local": V.iso(now, tz), "tz": tz,
                "n_systems": len(systems), "n_groups": groups["n_groups"],
                "n_statements": sum(x["n_statements"] for x in systems),
                "n_confident": sum(x["n_confident"] for x in systems),
                "systems": systems, "hint_zh": hint_zh, "hint_en": hint_en})


@router.get("/systems")
def systems():
    r = rt()
    st, cfg, now, tz = _ctx(r)
    with V.read_lock(r):
        rows = PV.systems_index(st, cfg, now)
    return _ok({"now": now, "now_local": V.iso(now, tz), "systems": rows})


# ================================================================ system view
@router.get("/systems/{system}/view")
def system_view(system: str, fresh: bool = False, flat: bool = False):
    """The system's portrait: actions (routes) -> statements, each with who
    (IPs / groups / prefixes / any), when (windows), content constraints,
    bindings and workflow edges, confidence, support and version. `fresh`
    renders now instead of returning P14's materialised view; `flat` adds
    the flat statement list (the same statements, not grouped by action)."""
    r = rt()
    st, cfg, now, tz = _ctx(r)
    with V.read_lock(r):
        _require_system(st, system)
        out = PV.system_view(st, system, cfg, now, tz, fresh=fresh)
    if not flat:
        out.pop("statements", None)
    out["now"] = now
    return _ok(out)


@router.get("/systems/{system}/precision")
def system_precision(system: str, days: int = Query(60, ge=1, le=400)):
    """Precision over time: per local day confirmed patterns, specificity,
    splits, drift, violations (live, from the store) and the curve measured
    against the generator's truth (evaluation runs), when one exists."""
    r = rt()
    st, cfg, now, tz = _ctx(r)
    with V.read_lock(r):
        _require_system(st, system)
        out = PV.precision_curve(st, system, cfg, now, tz, days=days)
    return _ok(out)


@router.get("/systems/{system}/lattice")
def system_lattice(system: str, kind: int = 0, root: Optional[int] = None,
                   depth: int = Query(4, ge=0, le=16), limit: int = Query(PV.LATTICE_MAX, ge=1, le=5000)):
    """The pattern tree (lattice) of the system, breadth-first from `root`."""
    r = rt()
    st, cfg, now, tz = _ctx(r)
    with V.read_lock(r):
        _require_system(st, system)
        try:
            out = PV.lattice(st, system, cfg, now, kind=kind, root=root, depth=depth, limit=limit)
        except KeyError:
            raise HTTPException(404, f"unknown node {root} in {system} kind {kind}")
    if out is None:
        out = {"system": system, "tree_key": PV.tree_key(st, system), "kind": kind, "kinds": [],
               "nodes": [], "n_nodes": 0, "root": None, "truncated": False}
    return _ok(out)


@router.get("/systems/{system}/entities/{ip}/view")
def ip_view(system: str, ip: str, limit: int = Query(50, ge=1, le=500)):
    """An IP (a user is an IP or an IP class, never a person): its group's
    pattern inherited, its exceptions, its bindings, the system statements
    that name it, and its violations."""
    r = rt()
    st, cfg, now, tz = _ctx(r)
    with V.read_lock(r):
        _require_system(st, system)
        if not PV.valid_ip(ip):
            raise HTTPException(404, f"not an IP address: {ip!r}")
        out = PV.ip_view(st, system, ip, cfg, now, tz, violations_limit=limit)
        if not out["known"]:
            raise HTTPException(404, f"unknown IP {ip} in {system}")
    return _ok(out)


@router.get("/systems/{system}/entities/{ip}/facets")
def ip_facets(system: str, ip: str, fresh: bool = False):
    r = rt()
    st, cfg, now, tz = _ctx(r)
    with V.read_lock(r):
        _require_system(st, system)
        if not PV.valid_ip(ip):
            raise HTTPException(404, f"not an IP address: {ip!r}")
        out = PV.facets(st, "ip", now, system=system, ip=ip, fresh=fresh)
    return _ok(out)


@router.get("/systems/{system}/facets")
def system_facets(system: str, fresh: bool = False):
    """The facet composition of the system's portrait (P13): functional,
    temporal, spatial, content, sequential, relational, technical, volume,
    identity, risk - each with sub-facets, only where applicable."""
    r = rt()
    st, cfg, now, tz = _ctx(r)
    with V.read_lock(r):
        _require_system(st, system)
        out = PV.facets(st, "system", now, system=system, fresh=fresh)
    return _ok(out)


@router.get("/systems/{system}/strategy")
def system_strategy(system: str):
    """P12: the system's measured characteristics, the chosen strategy per
    dimension (who granularity, content / binding / workflow engines on or
    off, tier) with reasons and history, and its system family."""
    r = rt()
    st, cfg, now, tz = _ctx(r)
    with V.read_lock(r):
        _require_system(st, system)
        out = PV.strategy(st, system, tz)
    if out is None:
        out = {"system": system, "tree_key": PV.tree_key(st, system), "chosen": {}, "engines": [],
               "characteristics": {}, "history": [], "hints": [], "who": {"mode": None}}
    return _ok(out)


@router.get("/systems/{system}/attributes")
def system_attributes(system: str, limit: int = Query(400, ge=1, le=5000)):
    """P02's attribute registry (types, coverage, cardinality) with P05's
    roles: which metrics are used, kept, dropped - learned, never listed."""
    r = rt()
    st, cfg, now, tz = _ctx(r)
    with V.read_lock(r):
        _require_system(st, system)
        out = PV.attributes(st, system, limit=limit)
    if out is None:
        out = {"system": system, "tree_key": PV.tree_key(st, system), "n": 0, "attributes": [],
               "by_role": {}, "by_type": {}}
    return _ok(out)


# ===================================================================== groups
@router.get("/groups")
def groups(members: int = Query(24, ge=0, le=1000)):
    r = rt()
    st, cfg, now, tz = _ctx(r)
    with V.read_lock(r):
        out = PV.groups_index(st, members_max=members)
    return _ok(out)


def _require_group(st: Any, g: str) -> None:
    from ..engines.behavior.lib import m_ptree as MP
    if g not in (MP.who_groups(st).get("groups") or {}):
        raise HTTPException(404, f"unknown group {g!r}")


@router.get("/groups/{group}/view")
def group_view(group: str, fresh: bool = False):
    """The group (user) view: group -> systems -> actions it performs, plus
    the negative statements ("从未在 X 中执行写操作")."""
    r = rt()
    st, cfg, now, tz = _ctx(r)
    with V.read_lock(r):
        _require_group(st, group)
        out = PV.group_view(st, group, cfg, now, tz, fresh=fresh)
    if out is None:
        raise HTTPException(404, f"unknown group {group!r}")
    return _ok(out)


@router.get("/groups/{group}/facets")
def group_facets(group: str, fresh: bool = False):
    r = rt()
    st, cfg, now, tz = _ctx(r)
    with V.read_lock(r):
        _require_group(st, group)
        out = PV.facets(st, "group", now, gid=group, fresh=fresh)
    return _ok(out)


class GroupName(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)


@router.post("/groups/{group}/name")
def group_name(group: str, body: GroupName = Body(...)):
    """Name a learned group (e.g. '综合部'). The name is persisted into the
    runtime config `who_group_names` as {name, ips: members}; P11 matches it
    to the group on its next daily run (Hungarian on Jaccard >= 0.5), so the
    name follows the group when its membership drifts."""
    from ..engines.behavior.lib import m_ptree as MP
    r = rt()
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "empty name")
    with V.read_lock(r):
        st = r.store
        gr = (MP.who_groups(st).get("groups") or {}).get(group)
        if gr is None:
            raise HTTPException(404, f"unknown group {group!r}")
        members = sorted(str(x) for x in gr.get("members") or [] if PV.valid_ip(str(x)))
        cidrs = sorted(str(x) for x in gr.get("members") or [] if "/" in str(x))
        entry: Dict[str, Any] = {"name": name, "ips": members}
        if cidrs:
            entry["cidrs"] = cidrs
        cfgs = [c for c in (getattr(r, "config", None), getattr(getattr(r, "pipeline", None), "config", None))
                if isinstance(c, dict)]
        for cfg in {id(c): c for c in cfgs}.values():
            cur = [x for x in (cfg.get("who_group_names") or [])
                   if isinstance(x, dict) and x.get("name") != name
                   and sorted(x.get("ips") or []) != members]
            cur.append(entry)
            cfg["who_group_names"] = cur
    return _ok({"ok": True, "group": group, "name": name, "pending": True, "entry": entry,
                "note_zh": "名称已写入 who_group_names，P11 在下一次每日聚类时按成员相似度匹配到该组",
                "note_en": "stored in who_group_names; P11 matches it to the group on its next daily run"})


# =================================================================== patterns
@router.get("/patterns/{pid:path}")
def pattern(pid: str):
    """One node of the pattern lattice: its context, parent / children,
    IP exceptions, fitted constraints (bounds, grammar, bindings, windows,
    workflow), the rendered statement, its lifecycle (lineage, confirmed /
    drift / absent / revived / retired events) and the violations it judged."""
    r = rt()
    st, cfg, now, tz = _ctx(r)
    with V.read_lock(r):
        out = PV.pattern_detail(st, pid, cfg, now, tz)
    if out is None:
        raise HTTPException(404, f"unknown pattern {pid!r}")
    return _ok(out)


# ================================================================= violations
@router.get("/violations")
def violations(system: Optional[str] = None, entity: Optional[str] = None,
               type: Optional[str] = None, severity: Optional[str] = None,      # noqa: A002
               since: Optional[float] = None, limit: int = Query(200, ge=1, le=2000)):
    """Pattern violations (P03) with typed reasons: who / when / content /
    seq / novel, flags, observed vs expected, p and per-day p."""
    r = rt()
    st, cfg, now, tz = _ctx(r)
    if type is not None and type not in PV.VTYPES:
        raise HTTPException(422, f"type must be one of {sorted(PV.VTYPES)}")
    if severity is not None and severity not in ("info", "low", "medium", "high", "critical"):
        raise HTTPException(422, "severity must be info|low|medium|high|critical")
    with V.read_lock(r):
        if system is not None:
            _require_system(st, system)
        out = PV.violations(st, now, tz, system=system, entity=entity, vtype=type, severity=severity,
                            since=since, limit=limit)
    out["now"] = now
    return _ok(out)


# ===================================================================== budget
@router.get("/budget")
def budget():
    """P15: per-tree tier and caps, degradation ladder, usage, active and
    earned set sizes, the ops.budget series and the P engines' timings."""
    r = rt()
    st, cfg, now, tz = _ctx(r)
    with V.read_lock(r):
        out = PV.budget(st, now, tz)
    out["now"] = now
    return _ok(out)
