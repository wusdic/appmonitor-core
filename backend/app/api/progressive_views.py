"""Read projections of the progressive profile core for API v3 (routes_v3.py,
docs/lib3/progressive.md §9.4).

Everything here reads the store (the P engines' published models and events)
and returns plain dicts; nothing writes. On-read rendering reuses P14's public
functions (views.system_view / group_view / ip_view) and P13's composers
(facets.compose_*), exactly as the engines use them, so the API never
re-implements a statement rule. The work of every projection is bounded by
the tree's node budget (n_max), the view's statement cap (S_MAX) or an
explicit `limit`; nothing iterates over the IP population.

Projections
  systems_index      per system: tree key, node counts by state, statements,
                     actions, who mode, tier
  system_view        actions -> statements (who / when / content / bindings /
                     workflow blocks) with confidence, support, version
  precision_curve    "longer is more precise", per local day: confirmed
                     patterns, mean specificity (depth), splits, drift,
                     violations, plus the confidence distribution now and,
                     when an evaluation run exists, the curve MEASURED against
                     the generator's truth (reports/progressive/runs)
  group_*            P11 groups, group view (group -> systems -> actions +
                     negative statements)
  ip_view            inherited group pattern + exceptions + bindings +
                     the system statements that name the IP + its violations
  lattice / pattern  the pattern tree (bounded BFS) and one node's detail:
                     path, children, exceptions, fitted constraints, lineage,
                     lifecycle events (drift history), violations
  violations         pattern_violation events with typed, bilingual reasons
  facets             P13 facet tree (stored, else composed on read)
  strategy           P12 characterisation, chosen arms and the engines they
                     switch on / off, family
  attributes         P02 registry with P05 roles
  budget             P15 caps, ladder, usage, ops.budget series
"""
from __future__ import annotations

import glob
import ipaddress
import json
import math
import os
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..models.schema import ORG, SYSTEM_ENTITY
from . import views as V
from .serialize import to_jsonable

DAY = 86400.0
GROUP_PREFIX = "class:grp:"
STATES = ("candidate", "confirmed", "stable", "evolving", "stale", "dormant", "retired")
CONFIDENT = ("confirmed", "stable", "evolving", "stale")
LIFECYCLE_KINDS = ("pattern_confirmed", "pattern_retired", "pattern_replaced", "pattern_drift",
                   "pattern_absent", "pattern_revived")
VIOLATION = "pattern_violation"
LATTICE_MAX = 600
LINEAGE_MAX = 200
SERIES_MAX = 192

# violation types (P03 TYPES) and flags (P03 / lib pbounds, pgrammar, pfd)
VTYPES: Dict[str, Tuple[str, str, str, str]] = {
    "who": ("来源越界", "source outside the pattern",
            "该模式的来源集合已封闭，此来源不在其中", "the pattern's source set is closed and this source is not in it"),
    "when": ("时间窗外", "outside the time windows",
             "动作发生在该模式学习到的时间窗之外", "the action happened outside the pattern's learned windows"),
    "content": ("内容约束违例", "content constraint violated",
                "提交内容超出大小区间、语法、取值集合或绑定关系", "the payload broke a size band, grammar, value set or binding"),
    "seq": ("流程违例", "workflow violated",
            "缺少该动作必需的前置步骤", "a required predecessor of the action is missing"),
    "novel": ("新动作", "new action",
              "该系统从未学习到的动作", "an action the system has never been seen to perform"),
}
FLAGS: Dict[str, Tuple[str, str]] = {
    "outsider_group": ("来源属于其他行为群组", "source belongs to another behavioural group"),
    "unknown_ip": ("从未见过的来源 IP", "never-seen source IP"),
    "system_new": ("该来源首次访问此系统", "first access of this source to the system"),
    "outside_windows": ("不在学习到的时间窗内", "outside the learned windows"),
    "above_range": ("超过学习到的上界", "above the learned range"),
    "below_range": ("低于学习到的下界", "below the learned range"),
    "grammar": ("不符合学习到的取值语法", "does not match the learned grammar"),
    "injection_shape": ("疑似注入形态", "injection-like shape"),
    "length": ("长度越界", "length out of range"),
    "new_value": ("封闭取值集合外的新值", "new value outside the closed set"),
    "new_key": ("出现新的表单/参数键", "new form / query key"),
    "missing_key": ("缺少必含的键", "required key missing"),
    "invariant": ("恒定属性发生变化", "an invariant attribute changed"),
    "cross_binding": ("使用了其他来源绑定的取值", "value bound to another source"),
    "concurrent_use": ("同一绑定值被多个来源同时使用", "bound value used concurrently by several sources"),
    "readdress_candidate": ("疑似换址（原来源已沉默）", "re-addressing candidate (old source went silent)"),
    "unbound_value": ("未绑定的取值", "unbound value"),
    "outside_set": ("取值不在该来源的集合中", "value outside the source's set"),
    "foreign_source": ("外来来源使用绑定字段", "foreign source on a bound field"),
    "missing_predecessor": ("缺少前置步骤", "missing predecessor"),
    "new_action": ("新动作", "new action"),
    "other_branch": ("落入未确认的分支", "fell into an unconfirmed branch"),
    "intensity": ("单位时间次数异常", "unusual hourly intensity"),
    "content": ("内容", "content"), "who": ("来源", "who"), "when": ("时间", "when"),
}
# P12 strategy dimensions -> the engines they switch (progressive.md §6.18.2)
STRATEGY_ENGINES: Dict[str, Tuple[str, str, str]] = {
    "P06": ("behavior.content_bounds", "数值区间与硬界", "numeric bands and hard bounds"),
    "P07": ("behavior.payload_grammar", "载荷语法与键集合", "payload grammar and key sets"),
    "P08": ("behavior.binding", "绑定关系（函数依赖）", "bindings (functional dependencies)"),
    "P09": ("behavior.time_window", "时间窗", "time windows"),
    "P10": ("behavior.workflow", "业务流程", "workflows"),
    "B09": ("behavior.client_identity", "客户端身份", "client identity"),
    "B10": ("behavior.sequence", "动作序列", "action sequences"),
    "B11": ("behavior.timing", "时间间隔", "inter-event timing"),
    "B12": ("behavior.beacon", "信标/周期", "beacons / periodicity"),
    "B13": ("behavior.budget", "流量预算", "volume budget"),
    "B15": ("behavior.identity_model", "身份模型", "identity model"),
}
WHO_MODES = {"ip": ("按 IP", "per IP"), "grp": ("按行为群组", "per behavioural group"),
             "prefix": ("按网段", "per prefix"), "reg": ("按区域", "per region"),
             "none": ("IP 不作为特征", "IP is not a feature"), None: ("未定", "undecided")}
STATE_TEXT = {"candidate": ("候选", "candidate"), "confirmed": ("已确认", "confirmed"),
              "stable": ("稳定", "stable"), "evolving": ("演化中", "evolving"),
              "stale": ("陈旧", "stale"), "dormant": ("休眠", "dormant"), "retired": ("已退役", "retired")}


# ======================================================================= base
def _mp():
    from ..engines.behavior.lib import m_ptree as MP
    return MP


def tree_key(st: Any, s: str) -> str:
    return _mp().tree_key(st, s)


def known_systems(st: Any) -> List[str]:
    """Systems with data plus tree keys with a pattern tree (families)."""
    MP = _mp()
    out = set(st.systems())
    try:
        out |= set(st.batch_systems("evt.batch"))
    except Exception:
        pass
    fam = MP.get_org_model(st, MP.SYSFAM)
    if isinstance(fam, Mapping):
        out |= {str(k) for k in (fam.get("member") or {}).values()}
    return sorted(out)


def ptree(st: Any, s: str) -> Any:
    MP = _mp()
    return MP.get_ptree(st, tree_key(st, s))


def core_present(st: Any) -> bool:
    MP = _mp()
    return any(MP.get_ptree(st, s) is not None for s in known_systems(st))


def local_date(ts: Optional[float], tz_off: float) -> Optional[str]:
    if ts is None or not math.isfinite(float(ts)):
        return None
    return time.strftime("%Y-%m-%d", time.gmtime(float(ts) + tz_off))


def tz_off(config: Mapping[str, Any], now: float) -> float:
    from ..engines.behavior.lib import pwindows as PW
    return PW.tz_offset(config, now)


def valid_ip(x: str) -> bool:
    try:
        ipaddress.ip_address(str(x))
        return True
    except ValueError:
        return False


def _f(x: Any, nd: int = 4) -> Optional[float]:
    return V.rnd(x, nd)


def _quant(xs: Sequence[float], q: float) -> Optional[float]:
    v = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not v:
        return None
    i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return round(v[i], 4)


# ================================================================== statements
def statement(stmt: Mapping[str, Any], tz: str) -> Dict[str, Any]:
    """One P14 statement for the UI: the rendered zh / en sentence plus its
    evidence split into the who / when / content / bindings / workflow blocks."""
    ev = stmt.get("evidence") or {}
    who = ev.get("who") or {}
    part = who.get("part_of")
    return {
        "id": stmt.get("id"), "pattern_id": stmt.get("pattern_id"), "view": stmt.get("view"),
        "subject": stmt.get("subject"), "text_zh": stmt.get("text_zh"), "text_en": stmt.get("text_en"),
        "support": _f(stmt.get("support"), 2), "confidence": _f(stmt.get("confidence"), 4),
        "state": stmt.get("state"), "version": stmt.get("version"), "cver": stmt.get("cver"),
        "first_seen": stmt.get("first_seen"), "last_seen": stmt.get("last_seen"),
        "first_seen_local": V.iso(stmt.get("first_seen"), tz),
        "last_seen_local": V.iso(stmt.get("last_seen"), tz),
        "mass": _f(stmt.get("mass"), 3), "facets": list(stmt.get("facets") or []),
        "act_node": stmt.get("act_node"),
        "route": ev.get("route"), "system": ev.get("system") or ev.get("target_system"),
        "kind": ev.get("kind"), "node": ev.get("node"), "depth": ev.get("depth"),
        "is_exc": bool(ev.get("is_exc")), "negative": bool(ev.get("negative")),
        "part_of": part, "group": who.get("group"),
        "context": ev.get("context") or [],
        "who": who or None, "when": ev.get("when"), "content": ev.get("content") or {},
        "bindings": ev.get("bindings") or {}, "workflow": ev.get("workflow") or [],
        "routes": ev.get("routes"),
    }


def _route_text(route: Any) -> str:
    from ..engines.behavior.lib import prender as PR
    try:
        return PR.route_text(route)
    except Exception:
        return str(route)


def _is_write(route: Any) -> bool:
    from ..engines.behavior.lib import prender as PR
    try:
        return bool(PR.is_write(route))
    except Exception:
        return False


def stored_or_rendered_system_view(st: Any, key: str, config: Mapping[str, Any], now: float,
                                   fresh: bool = False) -> Tuple[Optional[Dict[str, Any]], str]:
    """(view model, 'model' | 'render'): P14's materialised view, or the view
    rendered on read (no view yet, or fresh=True; P14 renders on its 2-h
    cadence and P15's ladder step 6 moves rendering to read time)."""
    MP = _mp()
    v = MP.get_model(st, key, MP.PVIEWS)
    if isinstance(v, Mapping) and v.get("statements") is not None and not fresh:
        return dict(v), "model"
    if MP.get_ptree(st, key) is None:
        return (dict(v), "model") if isinstance(v, Mapping) else (None, "none")
    from ..engines.behavior import views as VW
    r = VW.system_view(st, key, config, now)
    if r is None:
        return (dict(v), "model") if isinstance(v, Mapping) else (None, "none")
    if isinstance(v, Mapping):
        r.setdefault("version", v.get("version"))
    return r, "render"


def system_view(st: Any, s: str, config: Mapping[str, Any], now: float, tz: str,
                fresh: bool = False) -> Dict[str, Any]:
    """System view: actions (route, write?, mass share) -> statements."""
    key = tree_key(st, s)
    v, source = stored_or_rendered_system_view(st, key, config, now, fresh)
    v = v or {}
    stmts = [statement(x, tz) for x in v.get("statements") or []]
    by_id = {x["id"]: x for x in stmts}
    acts = []
    tot = sum(float(a.get("mass") or 0.0) for a in v.get("actions") or []) or 0.0
    for a in v.get("actions") or []:
        ss = [by_id[i] for i in a.get("statements") or [] if i in by_id]
        acts.append({"act_node": a.get("act_node"), "route": a.get("route"),
                     "route_text": _route_text(a.get("route")), "write": _is_write(a.get("route")),
                     "mass": _f(a.get("mass"), 3),
                     "share": _f(float(a.get("mass") or 0.0) / tot, 4) if tot > 0 else None,
                     "n_statements": len(ss), "statements": ss,
                     "confidence": _f(max((x["confidence"] or 0.0) for x in ss), 4) if ss else None})
    confs = [x["confidence"] for x in stmts if x["confidence"] is not None]
    return {"system": s, "tree_key": key, "source": source, "version": v.get("version"),
            "updated": v.get("updated"), "updated_local": V.iso(v.get("updated"), tz),
            "header": v.get("header") or {}, "who_mode": (v.get("header") or {}).get("who_mode"),
            "n_statements": len(stmts), "n_actions": len(acts), "actions": acts,
            "statements": stmts,
            "confidence": {"median": _quant(confs, 0.5), "p10": _quant(confs, 0.1),
                           "p90": _quant(confs, 0.9), "n": len(confs)}}


# ================================================================== systems
def node_counts(tree: Any) -> Dict[str, int]:
    out = {k: 0 for k in STATES}
    for nd in tree.nodes.values():
        out[nd.state] = out.get(nd.state, 0) + 1
    out["retired"] = out.get("retired", 0) + len(getattr(tree, "retired", {}) or {})
    out["dormant"] = out.get("dormant", 0) + len(getattr(tree, "dormant", {}) or {})
    return out


def systems_index(st: Any, config: Mapping[str, Any], now: float) -> List[Dict[str, Any]]:
    MP = _mp()
    wg = MP.who_groups(st)
    modes = wg.get("mode") or {}
    bud = MP.get_org_model(st, MP.BUDGET) or {}
    trees = (bud.get("trees") or {}) if isinstance(bud, Mapping) else {}
    out = []
    for s in known_systems(st):
        key = tree_key(st, s)
        m = MP.get_ptree(st, key)
        counts: Dict[str, Dict[str, int]] = {}
        if m is not None:
            for k, tr in sorted(m.kinds.items()):
                counts[str(k)] = node_counts(tr)
        v = MP.get_model(st, key, MP.PVIEWS)
        sp = MP.get_model(st, key, MP.SYSPROF)
        mode = (modes.get(s) or modes.get(key) or {}).get("mode") if isinstance(modes, Mapping) else None
        conf_n = sum(c.get(x, 0) for c in counts.values() for x in CONFIDENT)
        out.append({
            "system": s, "tree_key": key, "family": key if key != s else None,
            "has_tree": m is not None, "nodes": counts,
            "n_nodes": sum(sum(v_ for k_, v_ in c.items() if k_ not in ("retired", "dormant"))
                           for c in counts.values()),
            "n_confident": conf_n,
            "n_statements": len((v or {}).get("statements") or []) if isinstance(v, Mapping) else 0,
            "n_actions": len((v or {}).get("actions") or []) if isinstance(v, Mapping) else 0,
            "address": ((v or {}).get("header") or {}).get("address") if isinstance(v, Mapping) else None,
            "who_mode": mode or ((sp or {}).get("chosen") or {}).get("who") if isinstance(sp, Mapping) else mode,
            "tier": (trees.get(key) or {}).get("tier"),
            "view_version": (v or {}).get("version") if isinstance(v, Mapping) else None,
            "view_updated": (v or {}).get("updated") if isinstance(v, Mapping) else None,
            "n_entities": len(st.entities(s)) if s in st.systems() else 0,
        })
    return out


# ================================================================ precision
def _events(st: Any, system: Optional[str], kinds: Iterable[str], since: Optional[float] = None,
            entity: Optional[str] = None, limit: int = 5000) -> List[Any]:
    try:
        return st.events(system=system, entity=entity, since=since, kinds=list(kinds), limit=limit)
    except Exception:
        return []


def precision_curve(st: Any, s: str, config: Mapping[str, Any], now: float, tz: str,
                    days: int = 60) -> Dict[str, Any]:
    """Per local day of the system: confirmed (cumulative pattern_confirmed
    minus retired / replaced), mean specificity (depth of the confirmed nodes),
    splits (lineage), drift / absent events, violations. The confidence
    distribution is the current view's. `measured` is the evaluation run's
    curve against the generator's truth, when one exists for this system."""
    MP = _mp()
    key = tree_key(st, s)
    off = tz_off(config, now)
    since = now - days * DAY
    m = MP.get_ptree(st, key)
    rows: Dict[str, Dict[str, Any]] = {}

    def row(ts: float) -> Dict[str, Any]:
        d = local_date(ts, off)
        r = rows.get(d)
        if r is None:
            r = rows[d] = {"date": d, "confirmed_new": 0, "retired": 0, "splits": 0, "drift": 0,
                           "absent": 0, "violations": 0, "violations_medium": 0, "depths": []}
        return r

    depth_of: Dict[Tuple[int, int], int] = {}
    if m is not None:
        for k, tr in m.kinds.items():
            for nid, nd in tr.nodes.items():
                depth_of[(int(k), int(nid))] = int(nd.depth)
            for ent in list(tr.lineage):
                try:
                    t, op = float(ent[0]), str(ent[1])
                except Exception:
                    continue
                if t >= since and op in ("split", "exc_add"):
                    row(t)["splits"] += 1
    for e in _events(st, s, LIFECYCLE_KINDS, since, SYSTEM_ENTITY, limit=20000):
        ex = e.extra or {}
        if ex.get("tree_key") not in (None, key):
            continue
        r = row(e.ts)
        if e.kind == "pattern_confirmed":
            r["confirmed_new"] += 1
            d = depth_of.get((int(ex.get("event_kind") or 0), int(ex.get("node") or -1)))
            if d is not None:
                r["depths"].append(d)
        elif e.kind in ("pattern_retired", "pattern_replaced"):
            r["retired"] += 1
        elif e.kind == "pattern_drift":
            r["drift"] += 1
        elif e.kind == "pattern_absent":
            r["absent"] += 1
    for e in _events(st, s, (VIOLATION,), since, None, limit=50000):
        r = row(e.ts)
        r["violations"] += 1
        if V.sev_name(e.severity) in ("medium", "high", "critical"):
            r["violations_medium"] += 1
    out_rows = []
    cum = 0
    depths_all: List[int] = []
    for d in sorted(rows):
        r = rows[d]
        cum += r["confirmed_new"] - r["retired"]
        depths_all += r.pop("depths")
        r["confirmed"] = max(0, cum)
        r["mean_depth"] = round(sum(depths_all) / len(depths_all), 3) if depths_all else None
        out_rows.append(r)
    view = system_view(st, s, config, now, tz)
    by_state: Dict[str, int] = {}
    if m is not None:
        for tr in m.kinds.values():
            for nd in tr.nodes.values():
                by_state[nd.state] = by_state.get(nd.state, 0) + 1
    conf_depths = [int(nd.depth) for tr in (m.kinds.values() if m is not None else [])
                   for nd in tr.nodes.values() if nd.state in CONFIDENT]
    return {"system": s, "tree_key": key, "days": out_rows,
            "now": {"confidence": view["confidence"], "n_statements": view["n_statements"],
                    "nodes": by_state,
                    "mean_depth_confident": round(sum(conf_depths) / len(conf_depths), 3) if conf_depths else None,
                    "support_median": _quant([x["support"] for x in view["statements"]], 0.5)},
            "measured": measured_curve(s),
            "note_zh": "live 曲线来自运行中的模型（无真值）；measured 曲线是评估运行对生成器真值的实测（docs/lib3/progressive.md §16）",
            "note_en": "the live curve is read from the running models (no truth); the measured curve "
                       "is an evaluation run scored against the generator's truth (progressive.md §16)"}


_MEASURED_CACHE: Dict[str, Any] = {}


def runs_dir() -> str:
    return os.environ.get("APPMON_PROGRESSIVE_RUNS",
                          os.path.join(os.path.dirname(__file__), "..", "..", "..", "reports",
                                       "progressive", "runs"))


def _load_runs() -> List[Dict[str, Any]]:
    paths = sorted(glob.glob(os.path.join(runs_dir(), "*.json")))
    sig = tuple((p, os.path.getmtime(p)) for p in paths)
    if _MEASURED_CACHE.get("sig") == sig:
        return _MEASURED_CACHE["runs"]
    runs = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        sc = d.get("score") or {}
        pg1 = sc.get("pg1") or {}
        days = []
        for k in sorted(pg1, key=lambda x: int(x) if str(x).isdigit() else 0):
            r = pg1[k] or {}
            per_sys: Dict[str, List[int]] = {}
            for pk, pp in (r.get("per_pattern") or {}).items():
                parts = str(pk).split(".")
                if len(parts) >= 2:
                    c = per_sys.setdefault(parts[1], [0, 0])
                    c[1] += 1
                    c[0] += 1 if (pp or {}).get("recovered") else 0
            days.append({"day": r.get("day", int(k) if str(k).isdigit() else k),
                         "recall": r.get("recall"), "precision": r.get("precision"),
                         "ece": r.get("ece"), "mean_depth": r.get("mean_depth"),
                         "n_confirmed": r.get("n_confirmed"), "n_truth": r.get("n_truth"),
                         "components": r.get("components") or {},
                         "recall_by_system": {sy: (round(a / b, 4) if b else None)
                                              for sy, (a, b) in per_sys.items()}})
        runs.append({"file": os.path.basename(p), "pack": sc.get("pack"), "seed": d.get("seed", sc.get("seed")),
                     "registry": d.get("registry"), "days": days})
    _MEASURED_CACHE.update(sig=sig, runs=runs)
    return runs


def measured_curve(system: Optional[str] = None) -> Dict[str, Any]:
    runs = _load_runs()
    out = []
    for r in runs:
        days = []
        for d in r["days"]:
            x = {k: d[k] for k in ("day", "recall", "precision", "ece", "mean_depth", "n_confirmed",
                                   "n_truth")}
            x["system_recall"] = (d["recall_by_system"] or {}).get(system) if system else None
            days.append(x)
        if system and not any(x["system_recall"] is not None for x in days):
            continue
        out.append({"file": r["file"], "pack": r["pack"], "seed": r["seed"], "registry": r["registry"],
                    "days": days})
    return {"available": bool(out), "runs": out}


# =================================================================== groups
def groups_index(st: Any, members_max: int = 24) -> Dict[str, Any]:
    MP = _mp()
    wg = MP.who_groups(st)
    rows = []
    for g, gr in sorted((wg.get("groups") or {}).items(), key=lambda kv: -int(kv[1].get("n") or
                                                                              len(kv[1].get("members") or []))):
        mem = [str(x) for x in gr.get("members") or []]
        rows.append({"id": g, "name": gr.get("name") or g, "name_source": gr.get("name_source"),
                     "auto_name": gr.get("auto_name"), "n": int(gr.get("n") or len(mem)),
                     "members": mem[:members_max], "members_truncated": len(mem) > members_max,
                     "covers": list(gr.get("covers") or [])[:16],
                     "labels": [{"label": x.get("label"), "lift": _f(x.get("lift"), 3),
                                 "support": _f(x.get("support"), 3)}
                                for x in (gr.get("labels") or []) if isinstance(x, Mapping)][:5],
                     "systems": {k: _f(v, 4) for k, v in (gr.get("systems") or {}).items()},
                     "first_seen": gr.get("first_seen"), "changed": gr.get("changed"),
                     "cohesion": _f(gr.get("cohesion"), 4),
                     "provisional": list(gr.get("provisional") or [])[:16],
                     "n_sub": len(gr.get("sub") or [])})
    modes = {k: (v or {}).get("mode") for k, v in (wg.get("mode") or {}).items()} \
        if isinstance(wg.get("mode"), Mapping) else {}
    return {"version": wg.get("version"), "updated": wg.get("updated"), "last_run": wg.get("last_run"),
            "n_groups": len(rows), "n_grouped_ips": len(wg.get("ip2g") or {}),
            "shared": sorted(str(x) for x in (wg.get("shared") or []))[:64],
            "modes": modes, "stats": to_jsonable(wg.get("stats") or {}), "groups": rows}


def group_view(st: Any, g: str, config: Mapping[str, Any], now: float, tz: str,
               fresh: bool = False) -> Optional[Dict[str, Any]]:
    """Group -> systems -> actions (positive statements restricted to the
    members) + negative statements ("从未在 X 中执行写操作")."""
    MP = _mp()
    wg = MP.who_groups(st)
    gr = (wg.get("groups") or {}).get(g)
    if gr is None:
        return None
    v = st.get_model(ORG, f"{GROUP_PREFIX}{g}", MP.PVIEWS)
    source = "model"
    if not isinstance(v, Mapping) or fresh:
        from ..engines.behavior import views as VW
        r = VW.group_view(st, g, config, now)
        if r is not None:
            v, source = r, "render"
    v = dict(v or {})
    stmts = [statement(x, tz) for x in v.get("statements") or []]
    pos = [x for x in stmts if not x["negative"]]
    neg = [x for x in stmts if x["negative"]]
    systems: Dict[str, Dict[str, Any]] = {}
    for x in pos:
        sy = x.get("system") or "?"
        d = systems.setdefault(sy, {"system": sy, "actions": {}, "share": None})
        a = d["actions"].setdefault(x.get("route") or "?", {"route": x.get("route"),
                                                           "route_text": _route_text(x.get("route")),
                                                           "write": _is_write(x.get("route")),
                                                           "statements": []})
        a["statements"].append(x)
    shares = (v.get("header") or {}).get("systems") or gr.get("systems") or {}
    for sy, sh in shares.items():
        systems.setdefault(sy, {"system": sy, "actions": {}, "share": None})["share"] = _f(sh, 4)
    sys_rows = sorted(({"system": d["system"], "share": d["share"],
                        "actions": list(d["actions"].values())} for d in systems.values()),
                      key=lambda d: -(d["share"] or 0.0))
    return {"group": g, "name": gr.get("name") or g, "name_source": gr.get("name_source"),
            "source": source, "version": v.get("version"), "updated": v.get("updated"),
            "header": v.get("header") or {"members": gr.get("members") or []},
            "members": [str(x) for x in gr.get("members") or []],
            "covers": list(gr.get("covers") or []), "systems": sys_rows, "negative": neg,
            "n_statements": len(stmts)}


def ip_view(st: Any, s: str, ip: str, config: Mapping[str, Any], now: float, tz: str,
            violations_limit: int = 50) -> Dict[str, Any]:
    """The IP's inherited group pattern, exceptions, bindings, the system
    statements that name it, and its violations."""
    MP = _mp()
    from ..engines.behavior import views as VW
    key = tree_key(st, s)
    raw = VW.ip_view(st, s, ip, config, now)
    wg = MP.who_groups(st)
    g = raw.get("group")
    gr = (wg.get("groups") or {}).get(g) if g else None
    gv = raw.get("group_view") or {}
    inherited = [statement(x, tz) for x in gv.get("statements") or []]
    sysv = system_view(st, s, config, now, tz)
    naming = []
    for x in sysv["statements"]:
        w = x.get("who") or {}
        if ip in (w.get("members") or []) or ip in (w.get("items") or []):
            naming.append(x)
        elif any(ip in ((b or {}).get("table") or {}) for b in (x.get("bindings") or {}).values()):
            naming.append(x)
    pref = None
    try:
        pref = str(ipaddress.ip_network(f"{ip}/24", strict=False))
    except ValueError:
        pass
    vio = violations(st, now, tz, system=s, entity=ip, limit=violations_limit)
    return {"system": s, "tree_key": key, "ip": ip, "prefix24": pref,
            "group": ({"id": g, "name": (gr or {}).get("name") or g,
                       "n": len((gr or {}).get("members") or []),
                       "covers": list((gr or {}).get("covers") or [])[:8]} if g else None),
            "inherited": inherited,
            "inherited_here": [x for x in inherited if x.get("system") in (s, key)],
            "exceptions": [statement(x, tz) for x in raw.get("exceptions") or []],
            "bindings": _dedupe_bindings(raw.get("bindings") or []),
            "statements": naming, "violations": vio["violations"], "violation_counts": vio["counts"],
            "known": bool(g) or ip in st.entities(s) or bool(naming) or bool(vio["violations"])}


def _dedupe_bindings(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """One row per (action, pair, value): the same binding is fitted at a
    node and at its descendants; keep the strongest (highest LB)."""
    best: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for b in rows:
        k = (str(b.get("route")), str(b.get("pair")), str(b.get("value") if b.get("value") is not None
                                                          else b.get("set")))
        if k not in best or float(b.get("LB") or 0.0) > float(best[k].get("LB") or 0.0):
            best[k] = dict(b)
    return to_jsonable(sorted(best.values(), key=lambda b: (str(b.get("route")), str(b.get("pair")))))


# ================================================================== lattice
def _hier(st: Any, key: str, config: Mapping[str, Any]) -> Any:
    try:
        return _mp().hierarchies(st, key, config)
    except Exception:
        return None


def _ctx_text(ctx: Sequence[Any], hier: Any) -> str:
    from ..engines.behavior.pattern_tree import ctx_text
    try:
        return ctx_text(ctx, hier)
    except Exception:
        return " ∧ ".join(f"{a}" for a, *_ in ctx) or "*"


def _routes(st: Any, key: str, config: Mapping[str, Any], now: float, kind: int, tree: Any
            ) -> Dict[int, Tuple[Optional[str], Optional[int]]]:
    """{nid: (action route, action node)} from P14's route index (on read)."""
    from ..engines.behavior import views as VW
    try:
        c = VW._Ctx(st, key, config, now)
        return {nd.id: (route, act) for nd, route, act in VW._walk(tree, now, c.route_dist(kind))}
    except Exception:
        return {}


def node_brief(tree: Any, nd: Any, t: float, hier: Any, routes: Mapping[int, Any],
               key: str) -> Dict[str, Any]:
    from ..engines.behavior.lib import pnode as PN
    sp = nd.split
    split = None
    if sp is not None:
        split = {"attr": sp.attr, "level": int(sp.level), "n_groups": len(sp.groups),
                 "level_name": (hier.levels(sp.attr)[int(sp.level)] if hier is not None and
                                int(sp.level) < len(hier.levels(sp.attr)) else str(sp.level))}
    last = _ctx_text(nd.ctx[-1:], hier) if nd.ctx else "*"
    route, act = routes.get(nd.id, (None, None))
    try:
        distinct = int(round(nd.who.distinct(t)))
    except Exception:
        distinct = None
    return {"id": int(nd.id), "pattern_id": PN.pattern_id(key, tree.kind, nd.id, nd.version, nd.cver),
            "parent": nd.parent, "depth": int(nd.depth), "state": nd.state, "is_exc": bool(nd.is_exc),
            "label": last, "context": _ctx_text(nd.ctx, hier), "split": split,
            "children": [int(c) for c in tree.children(nd.id) if c in tree.nodes],
            "n_exc": len(nd.exc or {}), "exc_ips": sorted(nd.exc or {})[:8],
            "mass": _f(nd.mass_at(t), 3), "n_c": _f(nd.n_c(t), 2), "distinct_ips": distinct,
            "first_seen": nd.first_seen, "last_seen": nd.last_seen, "created": nd.created,
            "days": int(nd.days_total or 0), "version": int(nd.version), "cver": int(nd.cver),
            "route": route, "route_text": _route_text(route) if route else None, "act_node": act,
            "targets": sorted(nd.targets)[:24], "invariants": {k: str(v[1]) for k, v in (nd.inv or {}).items()}}


def lattice(st: Any, s: str, config: Mapping[str, Any], now: float, kind: int = 0,
            root: Optional[int] = None, depth: int = 4, limit: int = LATTICE_MAX) -> Optional[Dict[str, Any]]:
    """Bounded BFS of the pattern tree from `root` (default the tree root)."""
    MP = _mp()
    key = tree_key(st, s)
    m = MP.get_ptree(st, key)
    if m is None:
        return None
    tree = m.kinds.get(int(kind))
    if tree is None:
        return {"system": s, "tree_key": key, "kind": int(kind), "kinds": sorted(int(k) for k in m.kinds),
                "nodes": [], "n_nodes": 0, "root": None, "truncated": False}
    r0 = tree.root if root is None else int(root)
    if r0 not in tree.nodes:
        raise KeyError(r0)
    hier = _hier(st, key, config)
    routes = _routes(st, key, config, now, int(kind), tree)
    out, q, trunc = [], [(r0, 0)], False
    base = tree.nodes[r0].depth
    while q:
        nid, d = q.pop(0)
        nd = tree.nodes.get(nid)
        if nd is None:
            continue
        if len(out) >= limit:
            trunc = True
            break
        b = node_brief(tree, nd, now, hier, routes, key)
        b["expanded"] = d < depth
        out.append(b)
        if d < depth:
            q.extend((c, d + 1) for c in b["children"])
    return {"system": s, "tree_key": key, "kind": int(kind), "kinds": sorted(int(k) for k in m.kinds),
            "root": r0, "root_depth": int(base), "depth": int(depth), "n_nodes": len(tree.nodes),
            "nodes": out, "truncated": trunc, "budget": dict(getattr(tree, "budget", {}) or {}),
            "counts": node_counts(tree), "n_retired": len(getattr(tree, "retired", {}) or {}),
            "n_exceptions": tree.exceptions_count()}


def parse_pid(pid: str) -> Tuple[str, int, int, Optional[int], Optional[int]]:
    """'p:<key>:<kind>:<nid>[@v.c][|...]' -> (key, kind, nid, version, cver)."""
    body = str(pid).split("|", 1)[0]
    if body.startswith("p:"):
        body = body[2:]
    ver = cver = None
    if "@" in body:
        body, vc = body.rsplit("@", 1)
        a, _, b = vc.partition(".")
        ver = int(a) if a.isdigit() else None
        cver = int(b) if b.isdigit() else None
    key, kind, nid = body.rsplit(":", 2)
    return key, int(kind), int(nid), ver, cver


def _fitted(model: Any, kind: int, nid: int) -> Any:
    from ..engines.behavior.lib import pbounds as PB
    try:
        return PB.lookup(model, kind, nid) if isinstance(model, Mapping) else None
    except Exception:
        return None


def _flow_for(pflow: Any, route: Optional[str]) -> List[Dict[str, Any]]:
    if not isinstance(pflow, Mapping) or not route:
        return []
    from ..engines.behavior.lib import pdfg as DF
    try:
        sc = DF.lookup_scope(pflow, DF.STAR)
        r0 = DF.split_key(route)[0]
        return [e for e in (sc.get("edges") or [])
                if r0 in (DF.split_key(str(e.get("from")))[0], DF.split_key(str(e.get("to")))[0])][:24]
    except Exception:
        return []


def _lineage_entry(ent: Any, tz: str) -> Optional[Dict[str, Any]]:
    try:
        t, op, nid, parents, children, detail = ent
    except (TypeError, ValueError):
        return None
    return {"ts": float(t), "ts_local": V.iso(float(t), tz), "op": str(op), "node": int(nid),
            "parents": [int(x) for x in parents or ()], "children": [int(x) for x in children or ()],
            "detail": to_jsonable(detail) if not isinstance(detail, (str, int, float, type(None)))
            else detail}


def pattern_detail(st: Any, pid: str, config: Mapping[str, Any], now: float, tz: str
                   ) -> Optional[Dict[str, Any]]:
    """One lattice node: brief, path, children, exceptions, fitted constraints,
    statement, lineage, lifecycle (drift history), violations. None = unknown."""
    MP = _mp()
    try:
        key, kind, nid, ver, cver = parse_pid(pid)
    except (ValueError, TypeError):
        return None
    m = MP.get_ptree(st, key)
    if m is None or int(kind) not in m.kinds:
        return None
    tree = m.kinds[int(kind)]
    lineage = [x for x in (_lineage_entry(e, tz) for e in list(tree.lineage))
               if x is not None and (x["node"] == nid or nid in x["parents"] or nid in x["children"])]
    lineage = lineage[-LINEAGE_MAX:]
    life = []
    for e in _events(st, None, LIFECYCLE_KINDS, None, SYSTEM_ENTITY, limit=20000):
        ex = e.extra or {}
        try:
            same = (ex.get("tree_key") == key and int(ex.get("node")) == nid
                    and int(ex.get("event_kind", 0) or 0) == int(kind))
        except (TypeError, ValueError):
            same = False
        if same:
            life.append({"ts": e.ts, "ts_local": V.iso(e.ts, tz), "kind": e.kind,
                         "severity": V.sev_name(e.severity), "description": e.description,
                         "state": ex.get("state"), "pattern_id": ex.get("pattern_id"),
                         "detail": to_jsonable({k: v for k, v in ex.items()
                                                if k not in ("pattern_id", "node", "tree_key", "event_kind",
                                                             "context", "state")})})
    vio = [v for v in violations(st, now, tz, system=None, limit=500)["violations"]
           if v.get("tree_key") == key and v.get("node") == nid]
    nd = tree.nodes.get(nid)
    if nd is None:
        rec = (getattr(tree, "retired", {}) or {}).get(nid) or (getattr(tree, "dormant", {}) or {}).get(nid)
        if rec is None:
            return None
        return {"pattern_id": pid, "tree_key": key, "kind": int(kind), "node": nid, "alive": False,
                "state": "retired" if nid in (getattr(tree, "retired", {}) or {}) else "dormant",
                "record": to_jsonable(rec), "lineage": lineage, "lifecycle": life[:100],
                "violations": vio[:50]}
    hier = _hier(st, key, config)
    routes = _routes(st, key, config, now, int(kind), tree)
    brief = node_brief(tree, nd, now, hier, routes, key)
    path = [{"id": int(p), "label": _ctx_text(tree.nodes[p].ctx[-1:], hier) if tree.nodes[p].ctx else "*",
             "state": tree.nodes[p].state} for p in tree.path_to(nid)]
    children = [node_brief(tree, tree.nodes[c], now, hier, routes, key)
                for c in tree.children(nid) if c in tree.nodes]
    parent = (node_brief(tree, tree.nodes[nd.parent], now, hier, routes, key)
              if nd.parent is not None and nd.parent in tree.nodes else None)
    exc = [{"ip": ip, "node": int(x), "state": tree.nodes[x].state if x in tree.nodes else None}
           for ip, x in sorted((nd.exc or {}).items())]
    from ..engines.behavior.lib import prender as PR
    wg = MP.who_groups(st)
    try:
        who_ev, who_zh, who_en, who_c = PR.who_block(nd.who, now, nd.n_days(), wg.get("ip2g") or {},
                                                    wg.get("groups") or {}, (), None)
    except Exception:
        who_ev, who_zh, who_en, who_c = {}, "", "", None
    try:
        hist = {"workday": [round(float(x), 4) for x in nd.when.density(0, now)],
                "nonworkday": [round(float(x), 4) for x in nd.when.density(1, now)]}
    except Exception:
        hist = None
    constraints = {
        "bounds": to_jsonable(_fitted(MP.get_model(st, key, MP.PBOUNDS), kind, nid)),
        "grammar": to_jsonable(_fitted(MP.get_model(st, key, MP.PGRAMMAR), kind, nid)),
        "bindings": to_jsonable(_fitted(MP.get_model(st, key, MP.PBIND), kind, nid)),
        "windows": to_jsonable(_fitted(MP.get_model(st, key, MP.PWIN), kind, nid)),
        "workflow": to_jsonable(_flow_for(MP.get_model(st, key, MP.PFLOW), brief.get("route"))),
    }
    stmt = None
    systems = [s for s in known_systems(st) if tree_key(st, s) == key] or [key]
    sv, _src = stored_or_rendered_system_view(st, key, config, now)
    for x in (sv or {}).get("statements") or []:
        ev = x.get("evidence") or {}
        if ev.get("node") == nid and int(ev.get("kind", 0) or 0) == int(kind) and \
                not (ev.get("who") or {}).get("part_of"):
            stmt = statement(x, tz)
            break
    parts = [statement(x, tz) for x in (sv or {}).get("statements") or []
             if (x.get("evidence") or {}).get("node") == nid and (x.get("evidence") or {}).get("who", {}).get("part_of")]
    return {"pattern_id": brief["pattern_id"], "requested": pid,
            "stale_id": ver is not None and (ver, cver) != (int(nd.version), int(nd.cver)),
            "tree_key": key, "systems": systems, "kind": int(kind), "node": nid, "alive": True,
            "brief": brief, "state_text": STATE_TEXT.get(nd.state),
            "path": path, "parent": parent, "children": children, "exceptions": exc,
            "who": {"evidence": who_ev, "text_zh": who_zh, "text_en": who_en, "confidence": who_c},
            "when_hist96": hist, "constraints": constraints, "statement": stmt, "parts": parts,
            "lineage": lineage, "lifecycle": life[:100], "violations": vio[:50],
            "seg_start": nd.seg_start, "ref_snapshot": nd.ref is not None}


# =============================================================== violations
def violation(e: Any, tz: str) -> Dict[str, Any]:
    ex = e.extra or {}
    typ = ex.get("type")
    vt = VTYPES.get(typ, (typ, typ, "", ""))
    flags = [str(f) for f in ex.get("flags") or []]
    return {"id": e.id, "ts": e.ts, "ts_local": V.iso(e.ts, tz), "event_ts": ex.get("event_ts"),
            "system": e.system, "entity": e.entity, "type": typ, "type_zh": vt[0], "type_en": vt[1],
            "reason_zh": ex.get("statement_zh") or e.description, "reason_en": ex.get("statement_en"),
            "explain_zh": vt[2], "explain_en": vt[3],
            "flags": [{"flag": f, "zh": FLAGS.get(f, (f, f))[0], "en": FLAGS.get(f, (f, f))[1]} for f in flags],
            "severity": V.sev_name(e.severity), "status": e.status, "axes": list(e.axes or []),
            "p": _f(ex.get("p"), 8), "p_day": _f(ex.get("p_day"), 8), "U": _f(ex.get("U"), 6),
            "n_c": _f(ex.get("n_c"), 2), "sensitivity": _f(ex.get("sensitivity"), 4),
            "observed": to_jsonable(ex.get("observed")), "expected": to_jsonable(ex.get("expected")),
            "pattern_id": ex.get("pattern_id"), "route": ex.get("route"),
            "route_text": _route_text(ex.get("route")) if ex.get("route") else None,
            "node": ex.get("node"), "tree_key": ex.get("tree_key"), "incident_id": e.incident_id or None,
            "high_candidate": bool(ex.get("high_candidate"))}


def violations(st: Any, now: float, tz: str, system: Optional[str] = None, entity: Optional[str] = None,
               vtype: Optional[str] = None, severity: Optional[str] = None,
               since: Optional[float] = None, limit: int = 200) -> Dict[str, Any]:
    evs = _events(st, system, (VIOLATION,), since, entity, limit=max(limit * 5, 1000))
    rows = []
    counts: Dict[str, Dict[str, int]] = {"type": {}, "severity": {}, "flag": {}, "system": {}}
    sev_min = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    for e in evs:
        v = violation(e, tz)
        if vtype and v["type"] != vtype:
            continue
        if severity and sev_min.get(v["severity"], 0) < sev_min.get(severity, 0):
            continue
        for k, val in (("type", v["type"]), ("severity", v["severity"]), ("system", v["system"])):
            counts[k][str(val)] = counts[k].get(str(val), 0) + 1
        for f in v["flags"]:
            counts["flag"][f["flag"]] = counts["flag"].get(f["flag"], 0) + 1
        if len(rows) < limit:
            rows.append(v)
    return {"violations": rows, "counts": counts, "n": sum(counts["type"].values()),
            "types": {k: {"zh": a, "en": b, "explain_zh": c, "explain_en": d}
                      for k, (a, b, c, d) in VTYPES.items()}}


# =================================================================== facets
def _stored_facets(st: Any, s: str, e: str) -> Optional[Dict[str, Any]]:
    p = st.profile(s, e)
    f = (getattr(p, "extra", None) or {}).get("facets") if p is not None else None
    return dict(f) if isinstance(f, Mapping) and f.get("facets") is not None else None


def facet_registry(st: Any) -> List[Dict[str, Any]]:
    from ..engines.behavior.lib import pfacets as PFc
    try:
        reg = PFc.registry(st)
    except Exception:
        return []
    return [{"id": fid, "parent": d.get("parent"), "name_zh": d.get("name_zh"), "name_en": d.get("name_en"),
             "producer": d.get("producer"), "sources": list(d.get("sources") or []),
             "subjects": list(d.get("subjects") or []), "applicable": d.get("applicable"),
             "order": d.get("order")} for fid, d in sorted(reg.items(), key=lambda kv: kv[1].get("order", 999))]


def facets(st: Any, kind: str, now: float, system: Optional[str] = None, gid: Optional[str] = None,
           ip: Optional[str] = None, fresh: bool = False) -> Dict[str, Any]:
    """P13 facet tree of a subject: stored (P13's cadence) else composed now."""
    from ..engines.behavior import facets as F
    stored = None
    if not fresh:
        if kind == "system":
            stored = _stored_facets(st, tree_key(st, system), SYSTEM_ENTITY)
        elif kind == "group":
            stored = _stored_facets(st, ORG, f"{GROUP_PREFIX}{gid}")
        else:
            stored = _stored_facets(st, system, ip)
    if stored is not None:
        tree, source = stored, "model"
    else:
        if kind == "system":
            tree = F.compose_system(st, tree_key(st, system), now)
        elif kind == "group":
            tree = F.compose_group(st, gid, now)
        else:
            tree = F.compose_ip(st, system, ip, now)
        source = "render"
    roots = tree.get("facets") or []

    def count(ns: Sequence[Mapping[str, Any]]) -> int:
        return sum(len(n.get("items") or []) + count(n.get("children") or []) for n in ns)
    return {"subject": tree.get("subject"), "as_of": tree.get("as_of"), "hash": tree.get("hash"),
            "version": tree.get("version"), "source": source, "facets": to_jsonable(roots),
            "n_facets": len(roots), "n_items": count(roots), "registry": facet_registry(st)}


# ================================================================ strategy
def strategy(st: Any, s: str, tz: str) -> Optional[Dict[str, Any]]:
    MP = _mp()
    key = tree_key(st, s)
    sp = MP.get_model(st, key, MP.SYSPROF)
    if not isinstance(sp, Mapping):
        sp = MP.get_model(st, s, MP.SYSPROF)
    if not isinstance(sp, Mapping):
        return None
    chosen = dict(sp.get("chosen") or {})
    reasons = dict(sp.get("reasons") or {})
    engines = []
    for dim, (eng, zh, en) in STRATEGY_ENGINES.items():
        if dim not in chosen:
            continue
        val = str(chosen[dim])
        engines.append({"dim": dim, "engine": eng, "name_zh": zh, "name_en": en, "value": val,
                        "on": val not in ("off", "0", "False", "false"), "reason": reasons.get(dim)})
    who = chosen.get("who")
    fam = MP.get_org_model(st, MP.SYSFAM) or {}
    fid = (fam.get("member") or {}).get(s) if isinstance(fam, Mapping) else None
    arms = sp.get("arms") or {}
    arms_c = {}
    if isinstance(arms, Mapping):
        for dim, a in arms.items():
            arms_c[dim] = to_jsonable(a) if len(json.dumps(to_jsonable(a), default=str)) < 4000 else \
                {"truncated": True}
    hist = list(sp.get("history") or [])[-50:]
    return {"system": s, "tree_key": key, "version": sp.get("version"), "t": sp.get("t"),
            "t_local": V.iso(sp.get("t"), tz), "day": sp.get("day"),
            "members": list(sp.get("members") or []),
            "family": {"id": fid, "members": list(((fam.get("families") or {}).get(str(fid)[4:] if fid else None)
                                                   or (fam.get("families") or {}).get(fid) or []))
                       if isinstance(fam, Mapping) else []} if fid else None,
            "characteristics": to_jsonable(sp.get("characteristics") or {}),
            "measurements": to_jsonable(sp.get("measurements") or {}),
            "chosen": chosen, "reasons": reasons, "probe": list(sp.get("probe") or []),
            "hints": to_jsonable(sp.get("hints") or []), "history": to_jsonable(hist),
            "who": {"mode": who, "zh": WHO_MODES.get(who, (who, who))[0], "en": WHO_MODES.get(who, (who, who))[1]},
            "engines": engines, "arms": arms_c, "lambda": _f(sp.get("lambda"), 6)}


# ============================================================== attributes
def attributes(st: Any, s: str, limit: int = 400) -> Optional[Dict[str, Any]]:
    MP = _mp()
    key = tree_key(st, s)
    reg = MP.get_registry(st, key)
    sel = MP.get_model(st, key, MP.ATTRSEL)
    if reg is None and not isinstance(sel, Mapping):
        return None
    roles = dict((sel or {}).get("roles") or {}) if isinstance(sel, Mapping) else {}
    rows = []
    recs = getattr(reg, "records", {}) if reg is not None else {}
    t = None
    for name, r in sorted(recs.items()):
        try:
            cov = float(reg.coverage(name))
        except Exception:
            cov = None
        try:
            card = float(r.card_estimate())
        except Exception:
            card = None
        top = []
        try:
            tot = float(r.top.total(t)) or 0.0
            for k, c, *_ in r.top.items(t, None, 5):
                top.append([str(k)[:64], _f(float(c) / tot, 4) if tot > 0 else None])
        except Exception:
            top = []
        rows.append({"name": name, "ns": r.ns, "type": r.type, "role": roles.get(name, r.role_sys),
                     "state": r.state, "coverage": _f(cov, 4), "card": _f(card, 1),
                     "entropy": _f(r.entropy, 3), "stability": _f(r.stability, 3),
                     "approx_share": _f(r.approx_share, 3), "locked": bool(r.locked), "parse_as": r.parse_as,
                     "policy": r.policy, "cost_us": _f(r.cost_us, 2), "first_seen": r.first_seen,
                     "last_seen": r.last_seen, "version": r.version, "gone_at": r.gone_at,
                     "kinds": sorted(int(k) for k in r.kinds), "top": top})
    by_role: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    for x in rows:
        by_role[str(x["role"])] = by_role.get(str(x["role"]), 0) + 1
        by_type[str(x["type"])] = by_type.get(str(x["type"]), 0) + 1
    rows.sort(key=lambda x: (-(x["coverage"] or 0.0), x["name"]))
    return {"system": s, "tree_key": key, "n": len(rows), "a_max": getattr(reg, "a_max", None),
            "registry_version": getattr(reg, "version", None),
            "selection_version": (sel or {}).get("version") if isinstance(sel, Mapping) else None,
            "who_mode": (sel or {}).get("who_mode") if isinstance(sel, Mapping) else None,
            "redundant": to_jsonable((sel or {}).get("redundant") or {}) if isinstance(sel, Mapping) else {},
            "by_role": by_role, "by_type": by_type, "attributes": rows[:limit],
            "truncated": len(rows) > limit}


# ================================================================== budget
def budget(st: Any, now: float, tz: str, series_s: float = 8 * DAY) -> Dict[str, Any]:
    MP = _mp()
    b = MP.get_org_model(st, MP.BUDGET)
    b = b if isinstance(b, Mapping) else {}
    systems = {}
    for s, rec in (b.get("systems") or {}).items():
        systems[s] = {"n_active": rec.get("n_active", len(rec.get("active") or [])),
                      "n_earned": len(rec.get("earned") or []), "e_max": rec.get("e_max"),
                      "n_earned_candidates": rec.get("n_earned_candidates"), "ts": rec.get("ts")}
    series: Dict[str, List[Dict[str, Any]]] = {}
    for s in known_systems(st):
        try:
            pts = st.derived_series(s, SYSTEM_ENTITY, MP.OPS_BUDGET)
        except Exception:
            pts = []
        rows = [dict(p.value, ts=p.ts) for p in pts if p.ts >= now - series_s and isinstance(p.value, Mapping)]
        if len(rows) > SERIES_MAX:
            step = len(rows) / float(SERIES_MAX)
            rows = [rows[int(i * step)] for i in range(SERIES_MAX - 1)] + [rows[-1]]
        if rows:
            series[s] = to_jsonable(rows)
    health = {}
    try:
        health = st.health()
    except Exception:
        pass
    from ..engines.behavior.resource_governor import P_ENGINES
    eng = []
    for name, h in sorted(health.items()):
        if name not in P_ENGINES:
            continue
        eng.append({"engine": name, "duration_ms": _f(h.get("duration_ms"), 3), "runs": h.get("runs"),
                    "ok": h.get("ok"), "error_count": h.get("error_count"), "last_count": h.get("last_count")})
    tree_bytes = {}
    for s in known_systems(st):
        key = tree_key(st, s)
        m = MP.get_ptree(st, key)
        if m is not None and key not in tree_bytes:
            try:
                tree_bytes[key] = int(m.nbytes())
            except Exception:
                tree_bytes[key] = None
    return {"version": b.get("version"), "t": b.get("t"), "t_local": V.iso(b.get("t"), tz),
            "trees": to_jsonable(b.get("trees") or {}), "ladder": to_jsonable(b.get("ladder") or {}),
            "usage": to_jsonable(b.get("usage") or {}), "budget": to_jsonable(b.get("budget") or {}),
            "evicted": to_jsonable(b.get("evicted") or {}), "systems": systems, "series": series,
            "ptree_bytes": tree_bytes, "engines": eng, "present": bool(b)}
