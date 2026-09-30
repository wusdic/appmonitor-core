"""Facet registry and portrait composition of the progressive core
(docs/lib3/progressive.md §6.17.1, requirement S18/S19; card P13).

STATUS: implemented (W-P6, facet registry). Store accessors and pure
composition; no engine imports.

Requirement S19: a portrait is read like a person's — appearance, biology,
social attributes, each with sub-aspects. A FACET is a named aspect of a
subject (system, behavioural group, earned IP) produced by one or more
engines. Facets are DECLARED at runtime (no code change in P13 when an engine
adds one): any engine writes a declaration into model.facets@('__org__',
'__org__') under its own key,

    declare(store, 'behavior.binding', {
        'id': 'content.bindings', 'parent': 'content', 'name_zh': '绑定关系',
        'name_en': 'Bindings', 'subjects': ['system', 'group', 'ip'],
        'producer': 'behavior.binding', 'sources': ['model.pbind'],
        'applicable': 'payload_visible', 'render': 'pbind_v1', 'order': 43})

and P13 composes, per subject, the facet TREE of the non-empty facets whose
`applicable` predicate holds on the subject's system (P12 characteristics and
chosen strategy). The default tree (DEFAULT_FACETS) is the §6.17.1 table:

    1 功能 functional   2 时间节律 temporal   3 空间/网络 spatial   4 内容 content
    5 序列/流程 sequential   6 关系/社会 relational   7 技术 technical
    8 量/预算 volume   9 身份/连续性 identity   10 风险/健康 risk

Items of a facet come from (a) the statements of P14's views tagged with the
facet (statement['facets']), and (b) the facet's renderer over its source
models (RENDERERS[render]); a declaration with an unknown renderer gets the
generic one (the source model's version, update time and size, plus any
'items' list the producer publishes), so a new facet is visible at once.
Each facet payload is {id, name_zh, name_en, items: [{text_zh, text_en,
confidence, support, source}], confidence, as_of, version, children: [...]}.

Bounded: at most ITEMS_MAX items per facet, a fixed number of facets, and
per subject O(#facets + statements of the subject's view).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from ....models.schema import ORG, SYSTEM_ENTITY

FACETS = "model.facets"
ITEMS_MAX = 16
SUBJECTS = ("system", "group", "ip")
REQUIRED = ("id", "name_zh", "name_en")


def _d(id_, parent, zh, en, subjects, producer, sources, applicable, render, order):
    return {"id": id_, "parent": parent, "name_zh": zh, "name_en": en, "subjects": list(subjects),
            "producer": producer, "sources": list(sources), "applicable": applicable,
            "render": render, "order": order}


ALL = ("system", "group", "ip")
SG = ("system", "group")
DEFAULT_FACETS: List[Dict[str, Any]] = [
    _d("functional", None, "功能（外观）", "Functional (appearance)", ALL, "p13", (), "always", None, 10),
    _d("functional.actions", "functional", "动作与路由", "Actions and routes", ALL, "behavior.pattern_tree",
       ("model.ptree",), "always", "actions_v1", 11),
    _d("functional.variants", "functional", "动作变体", "Action variants", SG, "behavior.pattern_tree",
       ("model.ptree",), "always", None, 12),
    _d("functional.channels", "functional", "通道构成", "Channel mix", ("system",), "behavior.system_profile",
       ("model.sysprof",), "always", "channels_v1", 13),
    _d("temporal", None, "时间节律", "Temporal rhythm", ALL, "p13", (), "always", None, 20),
    _d("temporal.windows", "temporal", "活动时间窗", "Activity windows", ALL, "behavior.time_window",
       ("model.pwin",), "always", "windows_v1", 21),
    _d("temporal.rhythm", "temporal", "周节律", "Weekly rhythm", ("ip", "group"), "behavior.rhythm",
       ("model.rhythm",), "always", None, 22),
    _d("temporal.automation", "temporal", "自动化周期性", "Automation periodicity", ("system", "ip"),
       "derived.periodicity", ("model.sysprof",), "automation", "automation_v1", 23),
    _d("spatial", None, "空间/网络", "Spatial / network", ALL, "p13", (), "always", None, 30),
    _d("spatial.who", "spatial", "来源IP/网段/区域", "Who: IPs, prefixes, regions", SG,
       "behavior.pattern_tree", ("model.ptree",), "who_visible", "who_v1", 31),
    _d("spatial.destinations", "spatial", "目的地", "Destinations", ("system", "ip"), "behavior.novelty",
       ("model.sysprof",), "always", "destinations_v1", 32),
    _d("spatial.footprint", "spatial", "跨系统足迹", "Cross-system footprint", ("group", "ip"),
       "behavior.who_groups", ("model.who_groups",), "always", None, 33),
    _d("content", None, "内容", "Content", ALL, "p13", (), "payload_or_numeric", None, 40),
    _d("content.bounds", "content", "大小区间与硬界", "Size bands and bounds", ALL, "behavior.content_bounds",
       ("model.pbounds",), "always", "fitted_v1", 41),
    _d("content.grammar", "content", "载荷语法与键集合", "Payload grammar and key sets", ALL,
       "behavior.payload_grammar", ("model.pgrammar",), "payload_visible", "fitted_v1", 42),
    _d("content.bindings", "content", "绑定关系", "Bindings", ALL, "behavior.binding", ("model.pbind",),
       "payload_visible", "fitted_v1", 43),
    _d("content.invariants", "content", "不变式", "Invariants", ("system",), "behavior.attr_select",
       ("model.attrsel",), "always", "invariants_v1", 44),
    _d("sequential", None, "序列/流程", "Sequence / workflow", ALL, "p13", (), "sessions", None, 50),
    _d("sequential.workflows", "sequential", "业务流程与前置步骤", "Workflows and required predecessors", ALL,
       "behavior.workflow", ("model.pflow",), "sessions", "workflows_v1", 51),
    _d("relational", None, "关系/社会", "Relational / social", ALL, "p13", (), "always", None, 60),
    _d("relational.group", "relational", "行为群组与成员", "Behavioural group and members", ALL,
       "behavior.who_groups", ("model.who_groups",), "always", "groups_v1", 61),
    _d("relational.links", "relational", "关联/别名/共用IP", "Links, aliases, shared IPs", ("ip",),
       "behavior.entity_link", ("model.link",), "always", None, 62),
    _d("technical", None, "技术", "Technical", ALL, "p13", (), "always", None, 70),
    _d("technical.stacks", "technical", "客户端栈", "Client stacks", ("system",), "raw.client_stack",
       ("model.attr",), "always", "top_values_v1:client.stack", 71),
    _d("technical.tls", "technical", "TLS 特征", "TLS posture", ("system",), "behavior.attr_registry",
       ("model.attr",), "tls_seen", "top_values_v1:tls.ja3", 72),
    _d("volume", None, "量/预算", "Volume / budget", ALL, "p13", (), "always", None, 80),
    _d("volume.traffic", "volume", "流量规模", "Traffic volume", ("system",), "behavior.system_profile",
       ("model.sysprof",), "always", "volume_v1", 81),
    _d("identity", None, "身份/连续性", "Identity / continuity", ALL, "p13", (), "always", None, 90),
    _d("identity.bindings", "identity", "绑定稳定性", "Binding stability", ("system", "ip"),
       "behavior.binding", ("model.pbind",), "payload_visible", None, 91),
    _d("risk", None, "风险/健康", "Risk / health", ALL, "p13", (), "always", None, 100),
    _d("risk.incidents", "risk", "事件与违规", "Incidents and violations", ALL, "behavior.incident",
       ("incidents",), "always", "incidents_v1", 101),
    _d("risk.strategy", "risk", "画像策略与提示", "Profiling strategy and hints", ("system",),
       "behavior.system_profile", ("model.sysprof",), "always", "strategy_v1", 102),
]


# ============================================================ registry
def validate(decl: Mapping[str, Any]) -> Dict[str, Any]:
    missing = [k for k in REQUIRED if not decl.get(k)]
    if missing:
        raise ValueError(f"facet declaration misses {missing}")
    d = {"parent": None, "subjects": list(ALL), "producer": "", "sources": [], "applicable": "always",
         "render": None, "order": 999}
    d.update({k: copy.deepcopy(v) for k, v in decl.items()})
    d["subjects"] = [s for s in d["subjects"] if s in SUBJECTS]
    return d


def declare(store: Any, owner: str, decl: Mapping[str, Any], ts: Optional[float] = None) -> None:
    """Add or replace a facet declaration under the owner's key of
    model.facets@('__org__', '__org__') (each engine writes only its own key)."""
    d = validate(decl)
    m = store.get_model(ORG, ORG, FACETS)
    m = dict(m) if isinstance(m, Mapping) else {"version": 0, "decls": {}}
    decls = dict(m.get("decls") or {})
    own = dict(decls.get(owner) or {})
    if own.get(d["id"]) == d:
        return
    own[d["id"]] = d
    decls[owner] = own
    m["decls"] = decls
    m["version"] = int(m.get("version", 0)) + 1
    store.put_model(ORG, ORG, FACETS, m, version=m["version"], ts=ts)


def registry(store: Any) -> Dict[str, Dict[str, Any]]:
    """Default facets overlaid by every declared facet (a declaration with a
    default id replaces the default)."""
    out = {d["id"]: dict(d) for d in DEFAULT_FACETS}
    m = store.get_model(ORG, ORG, FACETS)
    if isinstance(m, Mapping):
        for owner, decls in sorted((m.get("decls") or {}).items()):
            for fid, d in (decls or {}).items():
                if isinstance(d, Mapping) and d.get("id"):
                    out[str(d["id"])] = dict(d, owner=owner)
    return out


# ========================================================== applicability
def _chars(sysprof: Any) -> Dict[str, Any]:
    return dict((sysprof or {}).get("characteristics") or {}) if isinstance(sysprof, Mapping) else {}


def _chosen(sysprof: Any) -> Dict[str, Any]:
    return dict((sysprof or {}).get("chosen") or {}) if isinstance(sysprof, Mapping) else {}


PREDICATES: Dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], bool]] = {
    "always": lambda ch, cz: True,
    "payload_visible": lambda ch, cz: not ch or float((ch.get("payload_vis") or {}).get("body", 0.0) or 0.0) > 0.0
    or float((ch.get("payload_vis") or {}).get("body_day", 0.0) or 0.0) >= 1.0,
    "payload_or_numeric": lambda ch, cz: not ch or bool(ch.get("numeric_targets"))
    or float((ch.get("payload_vis") or {}).get("body", 0.0) or 0.0) > 0.0,
    "sessions": lambda ch, cz: not ch or float(ch.get("sess_ident", 0.0) or 0.0) >= 0.3
    or cz.get("P10") == "on",
    "automation": lambda ch, cz: bool(ch) and (float(ch.get("automation", 0.0) or 0.0) > 0.0
                                               or int(ch.get("periodic_ips", 0) or 0) > 0),
    "who_visible": lambda ch, cz: cz.get("who", "ip") != "none" or not ch,
    "tls_seen": lambda ch, cz: not ch or float((ch.get("payload_vis") or {}).get("tls_opaque", 0.0) or 0.0) > 0.0,
}


def applicable(pred: Any, sysprof: Any) -> bool:
    """Evaluate a predicate name ('a&b', 'a|b' allowed) on a system's P12
    profile; an unknown predicate holds (a new facet is shown, not hidden)."""
    ch, cz = _chars(sysprof), _chosen(sysprof)
    expr = str(pred or "always")
    for alt in expr.split("|"):
        ok = True
        for term in alt.split("&"):
            term = term.strip()
            neg = term.startswith("!")
            fn = PREDICATES.get(term.lstrip("!"))
            v = True if fn is None else bool(fn(ch, cz))
            ok = ok and (not v if neg else v)
        if ok:
            return True
    return False


# ============================================================= renderers
def _item(zh: str, en: str, conf: Optional[float] = None, support: Optional[float] = None,
          source: str = "", **kw: Any) -> Dict[str, Any]:
    it = {"text_zh": zh, "text_en": en, "source": source}
    if conf is not None and math.isfinite(float(conf)):
        it["confidence"] = round(float(conf), 3)
    if support is not None and math.isfinite(float(support)):
        it["support"] = round(float(support), 1)
    it.update(kw)
    return it


def _generic(ctx: "Subject", d: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for src in d.get("sources") or ():
        m = ctx.model(src)
        if m is None:
            continue
        items = m.get("items") if isinstance(m, Mapping) else None
        if isinstance(items, list):
            for x in items[:ITEMS_MAX]:
                if isinstance(x, Mapping) and (x.get("text_zh") or x.get("text_en")):
                    out.append(dict(x, source=src))
            continue
        ver = m.get("version") if isinstance(m, Mapping) else getattr(m, "version", None)
        upd = m.get("updated") if isinstance(m, Mapping) else None
        n = len(m) if isinstance(m, (Mapping, list)) else None
        out.append(_item(f"{src}：版本 {ver}" + (f"，{n} 项" if n is not None else ""),
                         f"{src}: version {ver}" + (f", {n} entries" if n is not None else ""),
                         source=src, version=ver, updated=upd))
    return out


def _actions(ctx: "Subject", d: Mapping[str, Any]) -> List[Dict[str, Any]]:
    ptm = ctx.model("model.ptree")
    if ptm is None or 0 not in getattr(ptm, "kinds", {}):
        return []
    tr = ptm.kinds[0]
    t = ctx.now
    rows = []
    for nid, nd in tr.nodes.items():
        route = None
        for a, l, vals, neg in nd.ctx:
            if a in ("http.route", "http.path") and not neg and len(vals) == 1:
                route = str(next(iter(vals)))
        if route is None or nd.is_exc:
            continue
        rows.append((nd.mass_at(t), route, nd.state, nd.n_c(t)))
    rows.sort(key=lambda x: -x[0])
    tot = sum(r[0] for r in rows) or 1.0
    return [_item(f"{r}（占 {100 * m / tot:.0f}%，{st}）", f"{r} ({100 * m / tot:.0f} %, {st})",
                  support=n, source="model.ptree") for m, r, st, n in rows[:ITEMS_MAX]]


def _windows(ctx: "Subject", d: Mapping[str, Any]) -> List[Dict[str, Any]]:
    pw = ctx.model("model.pwin")
    if not isinstance(pw, Mapping):
        return []
    root = pw.get("root") or {}
    out = []
    for dt, ws in (root.get("by_daytype") or {}).items():
        for w in ws[:4]:
            s, e = int(w[0]), int(w[1])
            zh = "工作日" if dt == "wd" else "非工作日"
            out.append(_item(f"{zh} {s // 60:02d}:{s % 60:02d}–{e // 60:02d}:{e % 60:02d}",
                             f"{'workdays' if dt == 'wd' else 'non-workdays'} {s // 60:02d}:{s % 60:02d}-"
                             f"{e // 60:02d}:{e % 60:02d}", source="model.pwin"))
    return out[:ITEMS_MAX]


def _who(ctx: "Subject", d: Mapping[str, Any]) -> List[Dict[str, Any]]:
    ptm = ctx.model("model.ptree")
    if ptm is None or 0 not in getattr(ptm, "kinds", {}):
        return []
    tr = ptm.kinds[0]
    root = tr.nodes.get(tr.root)
    if root is None:
        return []
    t = ctx.now
    lv = root.who.levels
    out = []
    for l, zh, en in ((3, "群组", "groups"), (4, "区域", "regions"), (1, "网段 /24", "/24 prefixes")):
        items = [(k, c) for k, c, _, _ in lv[l].items(t)[:4] if not str(k).endswith("∅") and str(k) != "*"]
        tot = lv[l].total(t) or 1.0
        if items:
            txt = "、".join(f"{k}（{100 * c / tot:.0f}%）" for k, c in items)
            out.append(_item(f"主要{zh}：{txt}", f"main {en}: " + ", ".join(
                f"{k} ({100 * c / tot:.0f} %)" for k, c in items), source="model.ptree"))
    n = root.who.hll.count(t)
    out.append(_item(f"7 天内约 {n:.0f} 个来源 IP", f"about {n:.0f} source IPs in 7 days", source="model.ptree"))
    return out


def _fitted(ctx: "Subject", d: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for src in d.get("sources") or ():
        m = ctx.model(src)
        if not isinstance(m, Mapping):
            continue
        nodes = m.get("nodes") or {}
        n = sum(len(v) for v in nodes.values()) if isinstance(nodes, Mapping) else 0
        g = (m.get("gain") or {}).get("bits_per_event")
        texts = []
        for kind, ents in (nodes.items() if isinstance(nodes, Mapping) else ()):
            for nid, ent in (ents or {}).items():
                if isinstance(ent, Mapping) and ent.get("text_zh"):
                    texts.append((ent.get("text_zh"), ent.get("text_en", ""), ent.get("confidence")))
        for zh, en, c in texts[:ITEMS_MAX - 1]:
            out.append(_item(zh, en, c, source=src))
        out.append(_item(f"{n} 个节点已拟合" + (f"，预测增益 {g:.2f} 比特/事件" if g is not None else ""),
                         f"{n} nodes fitted" + (f", prequential gain {g:.2f} bits/event" if g is not None else ""),
                         source=src, gain=g))
    return out


def _workflows(ctx: "Subject", d: Mapping[str, Any]) -> List[Dict[str, Any]]:
    m = ctx.model("model.pflow")
    if not isinstance(m, Mapping):
        return []
    out = []
    for g, sc in (m.get("scopes") or {}).items():
        for wf in (sc.get("workflows") or [])[:4]:
            if isinstance(wf, Mapping) and wf.get("text_zh"):
                out.append(_item(wf["text_zh"], wf.get("text_en", ""), wf.get("confidence"), source="model.pflow"))
        for r in (sc.get("requires") or [])[:4]:
            if isinstance(r, Mapping) and r.get("to") and r.get("from"):
                out.append(_item(f"{r['to']} 需先经过 {r['from']}", f"{r['to']} requires {r['from']} first",
                                 r.get("confidence"), source="model.pflow"))
    return out[:ITEMS_MAX]


def _groups(ctx: "Subject", d: Mapping[str, Any]) -> List[Dict[str, Any]]:
    wg = ctx.org_model("model.who_groups")
    if not isinstance(wg, Mapping):
        return []
    groups = wg.get("groups") or {}
    out = []
    if ctx.kind == "group":
        g = groups.get(ctx.gid) or {}
        mem = list(g.get("members") or [])
        name = g.get("name") or ctx.gid
        out.append(_item(f"{name}：{len(mem)} 个成员" + (f"（{'、'.join(map(str, mem[:5]))}）" if len(mem) <= 5 else ""),
                         f"{name}: {len(mem)} members", source="model.who_groups"))
    elif ctx.kind == "ip":
        g = (wg.get("ip2g") or {}).get(ctx.entity)
        if g is not None:
            name = (groups.get(g) or {}).get("name") or g
            out.append(_item(f"所属群组：{name}", f"group: {name}", source="model.who_groups"))
    else:
        out.append(_item(f"组织内 {len(groups)} 个行为群组", f"{len(groups)} behavioural groups in the organisation",
                         source="model.who_groups"))
    return out


def _channels(ctx: "Subject", d: Mapping[str, Any]) -> List[Dict[str, Any]]:
    pv = _chars(ctx.sysprof).get("payload_vis") or {}
    if not pv:
        return []
    return [_item(f"HTTP {100 * float(pv.get('http', 0)):.0f}%，含请求体/参数 {100 * float(pv.get('body', 0)):.0f}%，"
                  f"不透明 TLS {100 * float(pv.get('tls_opaque', 0)):.0f}%",
                  f"HTTP {100 * float(pv.get('http', 0)):.0f} %, with body/query {100 * float(pv.get('body', 0)):.0f} %, "
                  f"opaque TLS {100 * float(pv.get('tls_opaque', 0)):.0f} %", source="model.sysprof")]


def _automation(ctx: "Subject", d: Mapping[str, Any]) -> List[Dict[str, Any]]:
    ch = _chars(ctx.sysprof)
    a = float(ch.get("automation", 0.0) or 0.0)
    return [_item(f"约 {100 * a:.0f}% 的活跃来源呈周期性（自动化客户端）",
                  f"about {100 * a:.0f} % of active sources are periodic (automation)", a,
                  source="model.sysprof")] if a > 0 or ch.get("periodic_ips") else []


def _destinations(ctx: "Subject", d: Mapping[str, Any]) -> List[Dict[str, Any]]:
    fam = _chars(ctx.sysprof).get("family")
    mem = (ctx.sysprof or {}).get("members") if isinstance(ctx.sysprof, Mapping) else None
    if fam and mem:
        return [_item(f"系统族 {fam}：{len(mem)} 台服务器（{'、'.join(mem[:6])}）",
                      f"system family {fam}: {len(mem)} servers", source="model.sysprof")]
    return []


def _invariants(ctx: "Subject", d: Mapping[str, Any]) -> List[Dict[str, Any]]:
    sel = ctx.model("model.attrsel")
    if not isinstance(sel, Mapping):
        return []
    inv = [a for a, r in (sel.get("roles") or {}).items() if r == "invariant"]
    return [_item(f"恒定属性：{'、'.join(sorted(inv)[:8])}", f"invariant attributes: {', '.join(sorted(inv)[:8])}",
                  source="model.attrsel")] if inv else []


def _top_values(ctx: "Subject", d: Mapping[str, Any], attr: str) -> List[Dict[str, Any]]:
    reg = ctx.model("model.attr")
    rec = reg.get(attr) if reg is not None and hasattr(reg, "get") else None
    if rec is None:
        return []
    items = rec.top.items(ctx.now)[:5]
    tot = rec.top.total(ctx.now) or 1.0
    if not items:
        return []
    return [_item("、".join(f"{k}（{100 * c / tot:.0f}%）" for k, c, _, _ in items),
                  ", ".join(f"{k} ({100 * c / tot:.0f} %)" for k, c, _, _ in items), source="model.attr")]


def _volume(ctx: "Subject", d: Mapping[str, Any]) -> List[Dict[str, Any]]:
    v = _chars(ctx.sysprof).get("volume") or {}
    if not v:
        return []
    return [_item(f"约 {float(v.get('events_day', 0)):.0f} 事件/天，学习 {100 * float(v.get('learned_share', 0)):.0f}%",
                  f"about {float(v.get('events_day', 0)):.0f} events/day, {100 * float(v.get('learned_share', 0)):.0f} % learned",
                  source="model.sysprof")]


def _incidents(ctx: "Subject", d: Mapping[str, Any]) -> List[Dict[str, Any]]:
    fn = getattr(ctx.store, "incidents", None)
    if fn is None or ctx.system is None:
        return []
    ent = ctx.entity if ctx.kind == "ip" else None
    incs = fn(system=ctx.system, entity=ent, status="open")
    if not incs:
        return []
    return [_item(f"{len(incs)} 个未结事件（最高 {max(str(getattr(i.severity, 'value', i.severity)) for i in incs)}）",
                  f"{len(incs)} open incidents", source="incidents")]


def _strategy(ctx: "Subject", d: Mapping[str, Any]) -> List[Dict[str, Any]]:
    sp = ctx.sysprof
    if not isinstance(sp, Mapping):
        return []
    cz = sp.get("chosen") or {}
    out = [_item(f"谁的粒度：{cz.get('who', '?')}；内容语法 {cz.get('P07', '?')}，绑定 {cz.get('P08', '?')}，"
                 f"流程 {cz.get('P10', '?')}", f"who granularity {cz.get('who', '?')}; grammar {cz.get('P07', '?')}, "
                 f"bindings {cz.get('P08', '?')}, workflows {cz.get('P10', '?')}", source="model.sysprof")]
    for h in sp.get("hints") or ():
        out.append(_item(h.get("text_zh", ""), h.get("text_en", ""), source="model.sysprof", hint=h.get("kind")))
    return out


RENDERERS: Dict[str, Callable[..., List[Dict[str, Any]]]] = {
    "actions_v1": _actions, "windows_v1": _windows, "who_v1": _who, "fitted_v1": _fitted,
    "workflows_v1": _workflows, "groups_v1": _groups, "channels_v1": _channels,
    "automation_v1": _automation, "destinations_v1": _destinations, "invariants_v1": _invariants,
    "volume_v1": _volume, "incidents_v1": _incidents, "strategy_v1": _strategy,
}


# ============================================================ composition
class Subject:
    """A composition subject: kind 'system' (tree key), 'group' (gid) or 'ip'."""

    def __init__(self, store: Any, kind: str, now: float, system: Optional[str] = None,
                 entity: Optional[str] = None, gid: Optional[str] = None,
                 sysprof: Any = None) -> None:
        self.store, self.kind, self.now = store, kind, float(now)
        self.system, self.entity, self.gid = system, entity, gid
        self.sysprof = sysprof
        self._m: Dict[str, Any] = {}

    def model(self, name: str) -> Any:
        if name not in self._m:
            self._m[name] = self.store.get_model(self.system, SYSTEM_ENTITY, name) if self.system else None
        return self._m[name]

    def org_model(self, name: str) -> Any:
        return self.store.get_model(ORG, ORG, name)


def compose(subject: Subject, reg: Mapping[str, Mapping[str, Any]],
            statements: Sequence[Mapping[str, Any]] = ()) -> Dict[str, Any]:
    """The facet tree of one subject: {'subject', 'as_of', 'facets': [...], 'hash'}."""
    flat: Dict[str, Dict[str, Any]] = {}
    for fid, d in reg.items():
        if subject.kind not in (d.get("subjects") or ALL):
            continue
        if not applicable(d.get("applicable"), subject.sysprof):
            continue
        items: List[Dict[str, Any]] = []
        rk = d.get("render")
        if rk:
            name, _, arg = str(rk).partition(":")
            fn = RENDERERS.get(name)
            try:
                if fn is None:
                    items = _generic(subject, d)
                elif arg:
                    items = fn(subject, d, arg)
                else:
                    items = fn(subject, d)
            except Exception as exc:                       # a broken source must not hide the portrait
                items = [_item("（来源读取失败）", f"(source unreadable: {type(exc).__name__})",
                               source=",".join(d.get("sources") or ()))]
        elif d.get("sources") and d.get("parent"):
            items = _generic(subject, d)
        flat[fid] = {"id": fid, "parent": d.get("parent"), "name_zh": d["name_zh"], "name_en": d["name_en"],
                     "order": d.get("order", 999), "items": items[:ITEMS_MAX], "children": []}
    for st in statements:
        tags = st.get("facets") or []
        for tag in tags:
            node = flat.get(tag)
            if node is None and "." in tag:
                node = flat.get(tag.split(".", 1)[0])
            if node is None or len(node["items"]) >= ITEMS_MAX:
                continue
            node["items"].append(_item(st.get("text_zh", ""), st.get("text_en", ""), st.get("confidence"),
                                       st.get("support"), source="model.pviews", statement=st.get("id")))
    # tree: attach children, keep non-empty facets (a parent is kept when a child is)
    for fid, node in flat.items():
        p = node["parent"]
        if p and p in flat:
            flat[p]["children"].append(node)
    def prune(n: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        kids = [c for c in (prune(k) for k in sorted(n["children"], key=lambda x: x["order"])) if c]
        if not n["items"] and not kids:
            return None
        confs = [it["confidence"] for it in n["items"] if "confidence" in it]
        return {"id": n["id"], "name_zh": n["name_zh"], "name_en": n["name_en"], "items": n["items"],
                "confidence": round(min(confs), 3) if confs else None, "children": kids}
    roots = [prune(n) for n in sorted(flat.values(), key=lambda x: x["order"])
             if not n["parent"] or n["parent"] not in flat]
    roots = [r for r in roots if r]
    body = json.dumps(roots, sort_keys=True, default=str, ensure_ascii=False)
    h = hashlib.blake2b(body.encode("utf-8"), digest_size=8).hexdigest()

    def stamp(ns: List[Dict[str, Any]]) -> None:
        for n in ns:
            n["as_of"] = subject.now
            stamp(n["children"])
    stamp(roots)
    return {"subject": {"kind": subject.kind, "system": subject.system, "entity": subject.entity,
                        "group": subject.gid}, "as_of": subject.now, "facets": roots, "hash": h}


def facet_ids(tree: Mapping[str, Any]) -> List[str]:
    out: List[str] = []

    def walk(ns: Iterable[Mapping[str, Any]]) -> None:
        for n in ns:
            out.append(n["id"])
            walk(n.get("children") or ())
    walk(tree.get("facets") or ())
    return out
