"""P14 ViewsEngine (`behavior.views`) — one pattern store, two projections
(docs/lib3/progressive.md §6.17.2-6.17.3, card P14). Library 3 (behaviour).

Requirement S11/S14/S15 ("站在用户行为视角：综合部访问哪几个业务都干什么 …
站在业务系统视角，OA服务器的某类人会在哪个时间段访问我什么页面干什么事"): the
same learned patterns are projected into
  * the SYSTEM view  model.pviews@(tree key, '__system__'):
      system -> action nodes (nodes whose context fixes a route, sorted by mass)
      -> per confident node at or below an action node (who / time / content
      variants, IP exceptions) one statement: who, windows, content constraints,
      bindings, workflow edges through the route;
  * the GROUP view   model.pviews@('__org__', 'class:grp:<g>'):
      group -> members / prefixes -> systems holding >= 5 % of the group's mass
      -> action nodes where the group holds >= 20 % of the node's mass, restricted
      to its members -> NEGATIVE statements: for every other system whose
      who-closed write nodes the group never reached, "综合部在财务系统中从未执行写
      操作（21 天、0 次）" — the fact that makes "综合部去财务系统审批" a who
      violation (P03), readable before it ever happens;
  * the IP view (on read, `ip_view`): the IP's group view plus its exception
      statements and the bindings whose source is the IP.
Each statement is rendered in zh and en (lib/prender) and carries the
machine-readable `evidence` block of the statement contract (eval/pmetrics
header) plus support (n_c), confidence, first / last seen, version, state.
Statements become more precise with observation time because every block is
read from the confidence channel (closedness U, coverage lower bounds, binding
LB, window coverage) and follow behaviour because the models they read do.

Reads   model.ptree, model.pbounds, model.pgrammar, model.pbind, model.pwin,
        model.pflow (P04, P06-P10), model.who_groups (P11: names, members,
        covers, who mode), config ip_classes / dhcp_scopes (configured region
        names), progressive.attr_names (display vocabulary).
Writes  model.pviews (system and group keys), profile versions (a new version
        when the rendered statements change).
Cadence 2 h per key (entity_due) plus `system_view` / `group_view` / `ip_view`
        for rendering on read. Work per system view is O(nodes of the tree);
        group views visit only the systems holding >= 5 % of the group's mass
        and a per-run cache of who-closed write nodes, so the cost never grows
        with the number of IPs. At most S_MAX statements per view.
Inert unless config['progressive']['enabled'].
"""
from __future__ import annotations

import hashlib
import math
import time
from typing import Callable, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ...core.engine import Context, Engine
from ...models.schema import ORG, SYSTEM_ENTITY
from .lib import m_ptree as MP
from .lib import pbounds as PB
from .lib import pdfg as DF
from .lib import pevent as EV
from .lib import pnode as PN
from .lib import psketch as PS
from .lib import prender as PR
from .lib import pwindows as PW

def _route_key(get: Callable[[str], Any]) -> Optional[str]:
    """lib/pdfg.route_key with absent attributes read as None (pdfg tests
    `isinstance(v, str)`, and ABSENT is the string '⊥')."""
    return DF.route_key(lambda a: (lambda v: None if v is EV.ABSENT else v)(get(a)))


PERIOD_S = 2 * 3600.0
GROUP_PREFIX = "class:grp:"
RENDERED = frozenset({"confirmed", "stable", "evolving", "stale"})
ROUTE_ATTRS = ("http.route", "http.path")
S_MAX = 512                      # statements per view
GROUP_SYS_SHARE = 0.05
GROUP_NODE_SHARE = 0.2
SKIP_PREFIX = ("ctx.", "ev.", "net.src", "net.peer_src", "net.dst", "sess.", "http.route",
               "http.path", "http.host")


def _route_of_node(nd: Any, t: float) -> Optional[str]:
    """The action route a node stands for: its context's route constraint (one
    value), else a route-like invariant (non-HTTP systems: TLS SNI, DNS name,
    destination), else a route target holding >= 99 % of the node's mass."""
    for a, l, vals, neg in nd.ctx:
        if a in ROUTE_ATTRS and not neg and l == 0 and len(vals) == 1:
            return str(next(iter(vals)))
    for a, l, vals, neg in nd.ctx:
        if a in ROUTE_ATTRS and not neg and l == 0 and 1 < len(vals) <= 3:
            tg = nd.targets.get(a)
            if isinstance(tg, PN.CatSummary):
                top = tg.top(t, 1)
                if top and str(top[0][0]) in {str(v) for v in vals}:
                    return str(top[0][0])
            return sorted(str(v) for v in vals)[0]
    for a in ROUTE_ATTRS:
        iv = nd.inv.get(a)
        if iv is not None and iv[0] == 0:
            return str(iv[1])
    tg = nd.targets.get("http.route")
    if isinstance(tg, PN.CatSummary):
        v = tg.invariant(t, 0.99, 20.0)
        if v is not None:
            return str(v)
    for a, pre in (("tls.sni", "TLS "), ("dns.qname", "DNS ")):
        iv = nd.inv.get(a)
        if iv is not None:
            e = DF.etld1(str(iv[1]))
            if e:
                return pre + e
        tg = nd.targets.get(a)
        if isinstance(tg, PN.CatSummary):
            v = tg.invariant(t, 0.99, 20.0)
            if v is not None and DF.etld1(str(v)):
                return pre + DF.etld1(str(v))
    return None


def _has_route_ctx(nd: Any) -> bool:
    return any(a in ROUTE_ATTRS and not neg for a, l, vals, neg in nd.ctx)


def _address(root: Any, t: float) -> str:
    iv = root.inv.get("net.dst")
    if iv is not None:
        return str(iv[1])
    tg = root.targets.get("net.dst")
    if isinstance(tg, PN.CatSummary):
        top = tg.top(t, 3)
        tot = tg.ss.total(t)
        if top and tot > 0:
            return "、".join(str(k) for k, c, g, e in top if g / tot >= 0.05)
    return ""


def _config_regions(config: Mapping[str, Any]) -> Set[str]:
    out = set()
    for k in ("ip_classes", "dhcp_scopes"):
        for it in (config or {}).get(k) or []:
            if isinstance(it, Mapping) and it.get("name"):
                out.add(str(it["name"]))
    return out


def _labels(config: Mapping[str, Any]) -> List[Tuple[str, str, str]]:
    out = []
    for it in (EV.pconfig(config).get("attr_names") or []):
        try:
            g, zh, en = it
            out.append((str(g), str(zh), str(en)))
        except (TypeError, ValueError):
            continue
    return out


STATE = "model.pviews_state"
ROUTE_K = 4                      # routes tracked per node in the route index
ROUTE_SHARE = 0.9                # a node stands for one action when one route holds >= 90 % of its mass
INDEX_ROWS = 4096                # learned rows indexed per tree and tick at most


class RouteIndex:
    """Which actions (route keys, lib/pdfg.route_key) end at which pattern
    node: per (kind, node id) a Space-Saving(4) of route keys with H_m-decayed
    MASS and EVIDENCE (one unit per burst of a source on a route, PPC-9), fed
    with the learned rows of each tick at the leaf P03 assigned them (pat.assign
    `leaf`), or routed here when P03 did not run. A node stands for one action
    when one route holds >= 90 % of its subtree's mass AND of its evidence: the
    tree may split on any attribute at any level (a path prefix, a /24, a size
    bin), so the route is read from the events, not guessed from the context,
    and a 60-s monitor that dominates a mixed node's mass (not its evidence)
    does not make the node "its" action. Bounded by the node budget x 4."""

    def __init__(self) -> None:
        self.ss: Dict[Tuple[int, int], Any] = {}
        self.last: Dict[str, float] = {}
        self.burst = PS.BurstEvidence(65536)

    def add(self, kind: int, nid: int, route: str, t: float, mass: float, src: Any = None) -> None:
        k = (int(kind), int(nid))
        s = self.ss.get(k)
        if s is None:
            s = self.ss[k] = PS.DecayedSpaceSaving(ROUTE_K, (PS.H_M,), (PS.H_M,), 0)
        ev = self.burst.unit((src, route), t) if src is not None else 1.0
        s.add(route, t, mass, ev)

    def prune(self, alive: Set[Tuple[int, int]]) -> None:
        for k in [k for k in self.ss if k not in alive]:
            del self.ss[k]

    def subtree(self, tree: Any, kind: int, t: float) -> Dict[int, Dict[str, Tuple[float, float]]]:
        """{nid: {route: (mass, evidence)}} summed over each node's subtree."""
        own: Dict[int, Dict[str, List[float]]] = {}
        for (k, nid), s in self.ss.items():
            if k != kind or nid not in tree.nodes:
                continue
            d = own.setdefault(nid, {})
            for key, c, g, e in s.items(t, 0):
                x = d.setdefault(str(key), [0.0, 0.0])
                x[0] += float(c)
                x[1] += float(e)
        out: Dict[int, Dict[str, Tuple[float, float]]] = {}

        def rec(nid: int) -> Dict[str, List[float]]:
            d = {r: list(v) for r, v in own.get(nid, {}).items()}
            for c in tree.children(nid):
                if c in tree.nodes:
                    for r, (m, e) in rec(c).items():
                        x = d.setdefault(r, [0.0, 0.0])
                        x[0] += m
                        x[1] += e
            out[nid] = {r: (v[0], v[1]) for r, v in d.items()}
            return d
        rec(tree.root)
        return out

    def nbytes(self) -> int:
        return int(sum(s.nbytes() for s in self.ss.values()) + 64 * len(self.ss) + 256)


def _dominant(d: Mapping[str, Any]) -> Optional[str]:
    """The route holding >= ROUTE_SHARE of both the mass and the evidence."""
    if not d:
        return None
    vals = {r: (v if isinstance(v, (tuple, list)) else (float(v), float(v))) for r, v in d.items()}
    tm = sum(v[0] for v in vals.values())
    te = sum(v[1] for v in vals.values())
    if tm <= 0 or te <= 0:
        return None
    r, (m, e) = max(vals.items(), key=lambda kv: (kv[1][1], kv[1][0], kv[0]))
    return r if m / tm >= ROUTE_SHARE and e / te >= ROUTE_SHARE else None


class _Ctx:
    """Everything one tree's rendering reads (once per view)."""

    def __init__(self, store: Any, key: str, config: Mapping[str, Any], now: float) -> None:
        self.store = store
        self.key = key
        self.now = now
        self.config = config
        self.ptm = MP.get_ptree(store, key)
        self.pb = MP.get_model(store, key, MP.PBOUNDS)
        self.pg = MP.get_model(store, key, MP.PGRAMMAR)
        self.pbind = MP.get_model(store, key, MP.PBIND)
        self.pwin = MP.get_model(store, key, MP.PWIN)
        self.pflow = MP.get_model(store, key, MP.PFLOW)
        self.wg = MP.who_groups(store)
        self.ip2g = self.wg.get("ip2g") or {}
        self.groups = self.wg.get("groups") or {}
        self.regions = _config_regions(config)
        self.labels = _labels(config)
        self.tz = PW.tz_offset(config, now)
        modes = self.wg.get("mode") or {}
        m = modes.get(key) or next((v for s, v in modes.items() if MP.tree_key(store, s) == key), None)
        self.mode = (m or {}).get("mode")
        self.edges = self._edges()
        ix = store.get_model(key, SYSTEM_ENTITY, STATE)
        self.index: Optional[RouteIndex] = ix if isinstance(ix, RouteIndex) else None
        self._rd: Dict[int, Dict[int, Dict[str, float]]] = {}

    def route_dist(self, kind: int) -> Dict[int, Dict[str, float]]:
        d = self._rd.get(kind)
        if d is None:
            tree = self.ptm.kinds.get(kind) if self.ptm is not None else None
            d = self._rd[kind] = (self.index.subtree(tree, kind, self.now)
                                  if (self.index is not None and tree is not None) else {})
        return d

    def _edges(self) -> Dict[str, List[Dict[str, Any]]]:
        """Kept workflow edges by route (from and to), scope '*'."""
        out: Dict[str, List[Dict[str, Any]]] = {}
        if not isinstance(self.pflow, Mapping):
            return out
        sc = DF.lookup_scope(self.pflow, DF.STAR)
        for e in sc.get("edges") or []:
            if float(e.get("dep", 0.0)) < DF.DEP_MIN:
                continue
            for r in {DF.split_key(str(e.get("from")))[0], DF.split_key(str(e.get("to")))[0]}:
                out.setdefault(r, []).append(e)
        return out


# ================================================================ statements
def node_statement(c: _Ctx, kind: int, nd: Any, route: str, view: str = "system",
                   subject: Optional[str] = None, restrict: Optional[Set[str]] = None
                   ) -> Optional[Dict[str, Any]]:
    """One statement for a confident node (restrict: member IPs of a group view)."""
    t = c.now
    pid = PN.pattern_id(c.key, kind, nd.id, nd.version, nd.cver)
    who_ev, who_zh, who_en, who_c = PR.who_block(nd.who, t, nd.n_days(), c.ip2g, c.groups,
                                                 c.regions, c.mode)
    if restrict is not None:
        items = [x for x in who_ev.get("items") or [] if str(x) in restrict]
        if who_ev.get("level") == "ip":
            if not items:
                return None
            who_ev = dict(who_ev, items=items, members=items)
            who_zh, who_en = PR.join_zh(items), PR.join_en(items)
    wentry = PW.lookup(c.pwin, kind, nd.id)
    when_ev, when_zh, when_en, when_c = PR.when_block(wentry)
    skip = [a for a in list(((PB.lookup(c.pb, kind, nd.id) or {}).get("attrs") or {}))
            + list(((PB.lookup(c.pg, kind, nd.id) or {}).get("attrs") or {}))
            if a.startswith(SKIP_PREFIX)]
    content, czh, cen, c_c = PR.content_block(PB.lookup(c.pb, kind, nd.id), PB.lookup(c.pg, kind, nd.id),
                                              c.labels, skip)
    bent = PB.lookup(c.pbind, kind, nd.id) if isinstance(c.pbind, Mapping) else None
    binds, bzh, ben, b_c = PR.binding_block(bent, c.labels)
    if restrict is not None:
        for a, b in binds.items():
            if "table" in b:
                b["table"] = {x: y for x, y in b["table"].items() if x in restrict}
                b["LB"] = {x: v for x, v in (b.get("LB") or {}).items() if x in restrict}
    edges = c.edges.get(DF.split_key(route)[0], []) if view == "system" or restrict is not None else []
    flow, fzh, fen, f_c = PR.workflow_block(edges)
    confs = [who_c, when_c, c_c, b_c, f_c]
    conf = float(min(x for x in confs if x is not None and math.isfinite(x))) if confs else 1.0
    if nd.state == "stale":
        conf *= PR.STALE_FACTOR
    sys_label = c.key
    addr = ""
    root = c.ptm.kinds[kind].nodes.get(c.ptm.kinds[kind].root) if c.ptm is not None else None
    if root is not None:
        addr = _address(root, t)
    first = PR.local_date(nd.first_seen, c.tz)
    last = PR.local_date(nd.last_seen, c.tz)
    zh, en = PR.sentence(sys_label, addr, when_zh, when_en, who_zh, who_en, PR.route_text(route),
                         czh, cen, bzh, ben, fzh, fen, conf, first, last, nd.version, nd.cver, nd.state)
    facets = ["functional", "spatial"]
    if when_ev:
        facets.append("temporal")
    if content:
        facets.append("content")
    if binds:
        facets.append("content.bindings")
    if flow:
        facets.append("sequential")
    ev: Dict[str, Any] = {
        "route": route, "system": c.key, "kind": kind, "node": nd.id,
        "context": [[a, int(l), sorted(str(v) for v in vals), bool(neg)] for a, l, vals, neg in nd.ctx],
        "depth": nd.depth, "is_exc": bool(nd.is_exc), "who": who_ev, "content": content,
        "bindings": binds, "workflow": flow}
    if when_ev:
        ev["when"] = when_ev
    return {"id": pid + (f"|{subject}" if subject else ""), "pattern_id": pid, "view": view,
            "subject": subject or c.key, "text_zh": zh, "text_en": en,
            "support": float(nd.n_c(t)), "confidence": conf,
            "first_seen": nd.first_seen, "last_seen": nd.last_seen, "version": nd.version,
            "cver": nd.cver, "state": nd.state, "mass": float(nd.mass_at(t)),
            "facets": facets, "evidence": ev}


def _walk(tree: Any, t: float, rd: Optional[Mapping[int, Mapping[str, float]]] = None):
    """(node, action route or None, action node id) for every node, depth first.
    A node's route is the dominant route of its subtree in the route index
    (>= 90 % of its mass), else what its context / invariants fix (lib-level
    fallback when the index has no data for it). The action node is the
    shallowest node on the path standing for that route."""
    stack = [(tree.root, None, None)]
    while stack:
        nid, proute, pact = stack.pop()
        nd = tree.nodes.get(nid)
        if nd is None:
            continue
        d = rd.get(nid) if rd else None
        if d:
            r = _dominant(d)
        else:
            r = _route_of_node(nd, t) if (_has_route_ctx(nd) or proute is None) else proute
        route, act = (r, pact if r is not None and r == proute else nid) if r is not None else (None, None)
        yield nd, route, act
        for c in reversed(tree.children(nid)):
            stack.append((c, route, act))


def system_view(store: Any, key: str, config: Mapping[str, Any], now: float,
                c: Optional[_Ctx] = None) -> Optional[Dict[str, Any]]:
    """The system view of a tree key (rendered now)."""
    c = c or _Ctx(store, key, config, now)
    if c.ptm is None:
        return None
    t0 = time.perf_counter()
    stmts: List[Dict[str, Any]] = []
    actions: Dict[int, Dict[str, Any]] = {}
    for kind, tree in sorted(c.ptm.kinds.items()):
        if kind != EV.KIND_TXN:
            continue
        for nd, route, act in _walk(tree, now, c.route_dist(kind)):
            if route is None or nd.state not in RENDERED:
                continue
            st = node_statement(c, kind, nd, route)
            if st is None:
                continue
            st["act_node"] = act
            stmts.append(st)
            a = actions.setdefault(act, {"act_node": act, "route": route, "mass": 0.0, "statements": []})
            a["statements"].append(st["id"])
            if nd.id == act:
                a["mass"] = st["mass"]
    stmts.sort(key=lambda s: (-actions.get(s.get("act_node"), {}).get("mass", 0.0), -s["mass"], s["id"]))
    stmts = stmts[:S_MAX]
    root = c.ptm.kinds.get(EV.KIND_TXN)
    addr = _address(root.nodes[root.root], now) if root is not None else ""
    hint_zh = hint_en = ""
    if c.mode == "none":
        hint_zh = "IP 对行为没有区分度（或来源地址被代理转换且未配置可信代理），IP 不作为特征"
        hint_en = "IP carries no behavioural information (or sources are NATed without trusted proxies); IP is not a feature"
    head_zh = f"【{key}" + (f" · {addr}" if addr else "") + f"】{len(actions)} 个动作、{len(stmts)} 条已确认模式"
    head_en = f"[{key}" + (f" · {addr}" if addr else "") + f"] {len(actions)} actions, {len(stmts)} confirmed patterns"
    return {"fmt": 1, "view": "system", "subject": key, "updated": now,
            "header": {"text_zh": head_zh + ("；" + hint_zh if hint_zh else ""),
                       "text_en": head_en + ("; " + hint_en if hint_en else ""),
                       "address": addr, "who_mode": c.mode, "hint": hint_zh or None},
            "statements": stmts,
            "actions": sorted(actions.values(), key=lambda a: -a["mass"]),
            "ms": round((time.perf_counter() - t0) * 1000.0, 2)}


# ================================================================== groups
def _closed_write_nodes(c: _Ctx) -> List[Tuple[Any, str]]:
    """Confident write action nodes whose who summary is closed (the nodes a
    group's negative statements are about)."""
    out = []
    if c.ptm is None:
        return out
    tree = c.ptm.kinds.get(EV.KIND_TXN)
    if tree is None:
        return out
    for nd, route, act in _walk(tree, c.now, c.route_dist(EV.KIND_TXN)):
        if route is None or nd.state not in ("confirmed", "stable") or not PR.is_write(route):
            continue
        if nd.who.closed_level(c.now, nd.n_days()) is None:
            continue
        out.append((nd, route))
    return out


def _group_mass(nd: Any, g: str, members: Set[str], t: float) -> float:
    """Share of the node's mass held by group g (grp level, else member IPs)."""
    lv3 = nd.who.levels[3]
    tot = lv3.total(t)
    key = f"grp:{g}"
    if tot > 0 and key in lv3:
        return float(lv3.count(key, t) / tot)
    lv0 = nd.who.levels[0]
    tot0 = lv0.total(t)
    if tot0 <= 0:
        return 0.0
    return float(sum(lv0.count(ip, t) for ip in members if ip in lv0) / tot0)


def group_view(store: Any, g: str, config: Mapping[str, Any], now: float,
               cache: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """The group view of P11 group g (rendered now)."""
    wg = MP.who_groups(store)
    gr = (wg.get("groups") or {}).get(g)
    if not gr:
        return None
    cache = cache if cache is not None else {}
    members = set(str(m) for m in gr.get("members") or [])
    name = str(gr.get("name") or g)
    subject = f"{GROUP_PREFIX}{g}"
    keys = sorted({MP.tree_key(store, s) for s in store.batch_systems(EV.EVT_BATCH)} |
                  {MP.tree_key(store, s) for s in store.systems()})
    stmts: List[Dict[str, Any]] = []
    used: List[str] = []
    for key in keys:
        c = cache.get(key)
        if c is None:
            c = cache[key] = _Ctx(store, key, config, now)
        if c.ptm is None or EV.KIND_TXN not in c.ptm.kinds:
            continue
        tree = c.ptm.kinds[EV.KIND_TXN]
        root = tree.nodes[tree.root]
        share_sys = float((gr.get("systems") or {}).get(key, 0.0))
        gm_root = _group_mass(root, g, members, now)
        if share_sys >= GROUP_SYS_SHARE:
            used.append(key)
            for nd, route, act in _walk(tree, c.now, c.route_dist(EV.KIND_TXN)):
                if route is None or nd.state not in RENDERED:
                    continue
                if _group_mass(nd, g, members, now) < GROUP_NODE_SHARE:
                    continue
                st = node_statement(c, EV.KIND_TXN, nd, route, view="group", subject=subject,
                                    restrict=members)
                if st is not None:
                    st["act_node"] = act
                    stmts.append(st)
        # negative statements: who-closed write nodes of this system the group never reached
        cw = cache.get(("cw", key))
        if cw is None:
            cw = cache[("cw", key)] = _closed_write_nodes(c)
        never = [(nd, route) for nd, route in cw if _group_mass(nd, g, members, now) <= 0.0]
        if never and (gm_root < GROUP_SYS_SHARE or not any(
                _group_mass(nd, g, members, now) > 0 for nd, _ in cw)):
            n_days = int(root.days_total or 0)
            zh, en = PR.negative_sentence(name, key, n_days)
            routes = sorted({PR.route_text(r) for _, r in never})
            zh += "（封闭的写操作：" + PR.join_zh(routes[:6]) + "）"
            en += " (closed write actions: " + PR.join_en(routes[:6]) + ")"
            stmts.append({"id": f"neg:{g}:{key}", "pattern_id": f"neg:{g}:{key}", "view": "group",
                          "subject": subject, "text_zh": zh, "text_en": en,
                          "support": float(sum(nd.n_c(now) for nd, _ in never)),
                          "confidence": float(min(1.0 - nd.who.levels[nd.who.closed_level(now, nd.n_days())].unseen(now)
                                                  for nd, _ in never)),
                          "state": "confirmed", "version": 1, "cver": 0,
                          "facets": ["relational", "risk"],
                          "evidence": {"negative": True, "target_system": key, "system": key,
                                       "routes": routes, "group": g,
                                       "who": {"level": "grp", "items": [f"grp:{g}"],
                                               "members": sorted(members)}}})
    zh = f"{name}（{len(members)} 个 IP）使用 {PR.join_zh(used) or '（尚无系统）'}"
    en = f"{name} ({len(members)} IPs) uses {PR.join_en(used) or '(no system yet)'}"
    return {"fmt": 1, "view": "group", "subject": subject, "group": g, "name": name,
            "name_source": gr.get("name_source"), "updated": now,
            "header": {"text_zh": zh, "text_en": en, "members": sorted(members),
                       "covers": gr.get("covers") or [], "labels": gr.get("labels") or [],
                       "systems": gr.get("systems") or {}},
            "statements": stmts[:S_MAX]}


def ip_view(store: Any, s: str, ip: str, config: Mapping[str, Any], now: float) -> Dict[str, Any]:
    """The IP view on read: its group view, its exception statements and the
    bindings whose source is the IP (§6.17.3)."""
    wg = MP.who_groups(store)
    g = (wg.get("ip2g") or {}).get(ip)
    out: Dict[str, Any] = {"view": "ip", "subject": ip, "system": s, "group": g, "updated": now,
                           "group_view": group_view(store, g, config, now) if g else None,
                           "exceptions": [], "bindings": []}
    key = MP.tree_key(store, s)
    c = _Ctx(store, key, config, now)
    if c.ptm is None:
        return out
    for kind, tree in c.ptm.kinds.items():
        for nd, route, act in _walk(tree, c.now, c.route_dist(kind)):
            if route is None:
                continue
            xid = nd.exc.get(ip) if nd.exc else None
            if xid is not None and xid in tree.nodes and tree.nodes[xid].state in RENDERED:
                st = node_statement(c, kind, tree.nodes[xid], route)
                if st is not None:
                    out["exceptions"].append(st)
            bent = PB.lookup(c.pbind, kind, nd.id) if isinstance(c.pbind, Mapping) else None
            for pk, rec in ((bent or {}).get("pairs") or {}).items():
                ent = (rec.get("table") or {}).get(ip)
                if ent and (ent.get("bound") or ent.get("set")):
                    out["bindings"].append({"route": route, "pair": pk, "value": ent.get("top"),
                                            "set": ent.get("set"), "LB": ent.get("LB"),
                                            "n": ent.get("n")})
    return out


# ================================================================== engine
class ViewsEngine(Engine):
    name = "behavior.views"
    layer = "behavior"
    consumes = [MP.PTREE, MP.PBOUNDS, MP.PGRAMMAR, MP.PBIND, MP.PWIN, MP.PFLOW, MP.WHO_GROUPS]
    produces = [MP.PVIEWS, "profile_version.pviews"]
    description = "P14: system and group views, zh/en statements with evidence"
    interval = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.view_period_s = float(params.get("view_period_s", PERIOD_S))
        self.last_stats: Dict[str, Any] = {}

    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        if not EV.enabled(ctx.config):
            return 0
        store = ctx.store
        now = float(ctx.now)
        keys = sorted({MP.tree_key(store, s) for s in store.batch_systems(EV.EVT_BATCH)} |
                      {MP.tree_key(store, s) for s in store.systems()})
        self._index(store, ctx.config, now)
        cache: Dict[Any, Any] = {}
        n = 0
        stats: Dict[str, Any] = {}
        for key in keys:
            if MP.get_ptree(store, key) is None or not self.entity_due(("p14", key), now, self.view_period_s):
                continue
            c = cache[key] = _Ctx(store, key, ctx.config, now)
            v = system_view(store, key, ctx.config, now, c)
            if v is None:
                continue
            self._put(store, key, SYSTEM_ENTITY, v, now)
            n += 1
            stats[key] = {"statements": len(v["statements"]), "ms": v["ms"]}
        wg = MP.who_groups(store)
        for g, gr in sorted((wg.get("groups") or {}).items()):
            if not gr.get("materialise", True):
                continue
            if not self.entity_due(("p14g", g), now, self.view_period_s):
                continue
            gv = group_view(store, g, ctx.config, now, cache)
            if gv is None:
                continue
            self._put(store, ORG, f"{GROUP_PREFIX}{g}", gv, now)
            n += 1
        self.last_stats = stats
        return n

    @staticmethod
    def _index(store: Any, config: Mapping[str, Any], now: float) -> int:
        """Route index update from the learned rows of the new txn batches."""
        n = 0
        hier_c: Dict[str, Any] = {}
        for s in sorted(store.batch_systems(EV.EVT_BATCH)):
            key = MP.tree_key(store, s)
            ptm = MP.get_ptree(store, key)
            tree = ptm.kinds.get(EV.KIND_TXN) if ptm is not None else None
            ix = store.get_model(key, SYSTEM_ENTITY, STATE)
            if not isinstance(ix, RouteIndex):
                ix = RouteIndex()
                store.put_model(key, SYSTEM_ENTITY, STATE, ix, version=1, ts=now)
            for ts_b, b in store.batches_since(s, EV.EVT_BATCH, ix.last.get(s, -math.inf)):
                ix.last[s] = ts_b
                if tree is None or getattr(b, "kind", EV.KIND_TXN) != EV.KIND_TXN or b.n == 0:
                    continue
                asg = store.batch_at(s, EV.PAT_ASSIGN, ts_b)
                leaf = asg.dense("leaf", None) if asg is not None and asg.n == b.n and asg.has("leaf") \
                    else None
                cb = store.batch_at(s, EV.EVT_CTX, ts_b)
                if cb is not None and cb.n != b.n:
                    cb = None
                mass = b.mass()
                for i in b.learned_rows()[:INDEX_ROWS].tolist():
                    def get(a: str, i: int = i) -> Any:
                        v = b.get(a, i)
                        if v is EV.ABSENT and cb is not None:
                            v = cb.get(a, i)
                        if v is EV.ABSENT and a == "net.src":
                            return b.ip_of(i)
                        return v
                    r = _route_key(get)
                    if r is None:
                        continue
                    nid = leaf[i] if leaf is not None else None
                    if nid is None or nid is EV.ABSENT or not (nid == nid) or int(nid) not in tree.nodes:
                        h = hier_c.get(key)
                        if h is None:
                            h = hier_c[key] = MP.hierarchies(store, key, config)
                        nid = tree.route(get, h)[-1]
                    ix.add(EV.KIND_TXN, int(nid), r, float(b.ts[i]), float(mass[i]), b.ip_of(i))
                    n += 1
            if tree is not None and len(ix.ss) > 2 * len(tree.nodes) + 64:
                ix.prune({(EV.KIND_TXN, nid) for nid in tree.nodes})
        return n

    @staticmethod
    def _put(store: Any, s: str, e: str, v: Dict[str, Any], now: float) -> None:
        old = store.get_model(s, e, MP.PVIEWS)
        sig = hashlib.blake2b("\n".join(sorted(f"{st['id']}\t{st['text_zh']}" for st in v.get("statements") or []))
                              .encode("utf-8"), digest_size=12).hexdigest()
        ver = int((old or {}).get("version") or 0) if isinstance(old, Mapping) else 0
        if not isinstance(old, Mapping) or old.get("_sig") != sig:
            ver += 1
            v["version"] = ver
            v["_sig"] = sig
            store.put_model(s, e, MP.PVIEWS, v, version=ver, ts=now)
            store.put_profile_version(s, e, ver, {"view": v.get("view"), "subject": v.get("subject"),
                                                  "statements": [(st["id"], st["text_zh"])
                                                                 for st in v.get("statements") or []]},
                                      ts=now)
        else:
            v["version"] = ver
            v["_sig"] = sig
            store.put_model(s, e, MP.PVIEWS, v, version=ver, ts=now)
