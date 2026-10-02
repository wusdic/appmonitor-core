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
  * the DEPARTMENT view model.pviews@('__org__', 'class:grp:dept:<name>'): a
      configured department (who_group_names) that P11 learned as several
      groups (its roles, rec 'dept'), composed from their group views: "综合部
      访问 oa：登录、文档、审批[192.168.1.21]、提交报告[192.168.1.23、10.168.7.121]",
      and a negative statement where every role never wrote (dept_view);
  * the IP view (on read, `ip_view`): the IP's group view plus its exception
      statements and the bindings whose source is the IP.
Round 3 (checked against the requirement's wording on pack O): a system-view
statement names the configured department of its addresses ('财务部
（192.168.2.10）') and the action after its page ('POST /fin/approval/{num}/
approve（审批）'); a binding is stated when it discriminates its sources (3
finance users) even below BIND_MIN_CARD values; a group view states the
group's PART of a node (its members, its windows); "never" agrees with the
group's own P11 action mix and signatures, and what a group / department never
does inside a system it uses is stated too ('销售部 在 oa 中从未执行：审批（…）',
evidence scope 'actions'; the whole-system statement has scope 'system').
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
from .lib import pfd as FD
from .lib import pevent as EV
from .lib import pnode as PN
from .lib import psketch as PS
from .lib import prender as PR
from .lib import pwindows as PW
from .lib.phier import GRP_NONE
from . import conformity as CF

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
        if a in ROUTE_ATTRS and not neg and l == 0 and len(vals) == 1 \
                and next(iter(vals)) is not EV.ABSENT and next(iter(vals)) != EV.ABSENT:
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
            e = DF.host_key(str(iv[1]))
            if e:
                return pre + e
        tg = nd.targets.get(a)
        if isinstance(tg, PN.CatSummary):
            v = tg.invariant(t, 0.99, 20.0)
            if v is not None and DF.host_key(str(v)):
                return pre + DF.host_key(str(v))
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
        self.reg = MP.get_registry(store, key)
        self.wg = MP.who_groups(store)
        self.ip2g = self.wg.get("ip2g") or {}
        self.groups = self.wg.get("groups") or {}
        ws = store.get_model(ORG, ORG, CF.WG_STATE)
        self.sigs = getattr(ws, "sigs", None)          # P11's per-address signatures (group parts)
        self.grp_gain = _grp_gain(MP.get_model(store, key, MP.SYSPROF))
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
                   subject: Optional[str] = None, restrict: Optional[Set[str]] = None,
                   part: Optional[Tuple[str, List[str], str, float]] = None
                   ) -> Optional[Dict[str, Any]]:
    """One statement for a confident node (restrict: member IPs of a group view).

    `part` = (group id, member IPs seen at the node, group name, share): the
    statement about ONE learned group's part of the node (the system view's
    "某类人" decomposition, see group_parts): who = that group's members at
    the node, not a closure claim (the node's other sources are the other
    parts), context + (net.src, grp level, {grp:g}), bindings restricted to
    the members; every other constraint is the node's."""
    t = c.now
    pid = PN.pattern_id(c.key, kind, nd.id, nd.version, nd.cver)
    who_ev, who_zh, who_en, who_c = PR.who_block(nd.who, t, nd.n_days(), c.ip2g, c.groups,
                                                 c.regions, c.mode)
    if part is not None:
        g, mem, gname, share = part[:4]
        restrict = set(mem)
        lst_zh, lst_en = PR.join_zh(mem), PR.join_en(mem)
        if len(mem) > PR.MEMBERS_LISTED:
            lst_zh, lst_en = f"{len(mem)} 个 IP", f"{len(mem)} IPs"
        who_zh, who_en = f"{gname}（{lst_zh}）", f"{gname} ({lst_en})"
        gids = list(part[4]) if len(part) > 4 else [g]
        who_ev = {"level": "grp", "items": [f"grp:{x}" for x in gids], "members": sorted(mem), "group": g,
                  "name": gname, "share": round(float(share), 4), "closed": False,
                  "U": who_ev.get("U"), "confidence": who_c, "part_of": pid}
    elif restrict is not None:
        items = [x for x in who_ev.get("items") or [] if str(x) in restrict]
        if who_ev.get("level") == "ip":
            if not items:
                return None
            who_ev = dict(who_ev, items=items, members=items)
            who_zh, who_en = PR.join_zh(items), PR.join_en(items)
    elif who_ev.get("level") == "ip":
        # (round 3) which KIND of people: the configured department all the
        # stated addresses belong to (through their learned groups' `dept`, or
        # the department's configured addresses), e.g. '财务部（192.168.2.10）'
        # for the approver's role group - the requirement's "某类人"
        ips = [str(x) for x in who_ev.get("members") or who_ev.get("items") or []]
        dn = _dept_of(c, ips)
        lz, le = PR.join_zh(who_ev.get("items") or ips), PR.join_en(who_ev.get("items") or ips)
        if dn or _auto_group_of(c, ips):
            # a department's name; an AUTO-named learned group of these
            # addresses ('G16·finance GET /health+oa GET /health') says nothing
            # the addresses do not: plain addresses then
            who_zh, who_en = (f"{dn}（{lz}）", f"{dn} ({le})") if dn else (lz, le)
            if not who_ev.get("closed"):
                who_zh = f"目前观测到 {who_zh}（来源集合尚未封闭）"
                who_en = f"so far {who_en} (source set not yet closed)"
            if dn:
                who_ev = dict(who_ev, dept=dn)
    elif who_ev.get("level") in ("prefix", "reg"):
        # (round 3) a pool group's prefixes: '研发（10.50.0.0/24、…，约 97 个 IP）'
        pn = _pool_group_of(c, [str(x) for x in who_ev.get("items") or []])
        if pn:
            items = [str(x) for x in who_ev.get("items") or []]
            nd_ = who_ev.get("distinct")
            tail_zh = f"，约 {nd_} 个 IP" if nd_ else ""
            tail_en = f", ~{nd_} IPs" if nd_ else ""
            who_zh = f"{pn}（{PR.join_zh(items)}{tail_zh}）"
            who_en = f"{pn} ({PR.join_en(items)}{tail_en})"
            who_ev = dict(who_ev, group_name=pn)
    wentry = PW.lookup(c.pwin, kind, nd.id)
    own_when = False
    if part is not None:
        # the group's own arrival windows when the node's minute reservoir holds
        # enough of its arrivals, else the node's windows
        pw = PW.part_when(nd.when, part[1], c.tz)
        if pw is not None:
            by = pw.pop("by_daytype", None)
            wentry = {"status": "fitted", "when": pw, "by_daytype": by}
            own_when = True
    when_ev, when_zh, when_en, when_c = PR.when_block(wentry)
    skip = [a for a in list(((PB.lookup(c.pb, kind, nd.id) or {}).get("attrs") or {}))
            + list(((PB.lookup(c.pg, kind, nd.id) or {}).get("attrs") or {}))
            if a.startswith(SKIP_PREFIX)]
    g_ent = PB.lookup(c.pg, kind, nd.id)
    if restrict is not None:
        g_ent = _restrict_closed(g_ent, PB.lookup(c.pbind, kind, nd.id) if isinstance(c.pbind, Mapping)
                                 else None, restrict)
    content, czh, cen, c_c = PR.content_block(PB.lookup(c.pb, kind, nd.id), g_ent, c.labels, skip)
    bent = PB.lookup(c.pbind, kind, nd.id) if isinstance(c.pbind, Mapping) else None
    if bent and c.reg is not None:
        # a "binding" of an attribute with a handful of values system-wide (body
        # format, content type) is a constant of the action, already stated as
        # content; only identifier-like payloads (usernames, accounts) bind
        keep = {}
        for pk, rec in (bent.get("pairs") or {}).items():
            # the same rule decides which pairs P04's held-out test checks
            if FD.binding_stated(rec, FD.payload_card(c.reg, rec)):
                keep[pk] = rec
        bent = dict(bent, pairs=keep)
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
    # the calibrated confidence when P04 has checked the node's constraints on
    # held-out data (pnode.HoldRecord: the probability that every constraint
    # holds on new events); the min-over-parts formula above states the
    # weakest part's NOMINAL coverage and was under-confident on pack O
    # (PG2 ECE 0.36-0.43)
    ph = getattr(nd, "p_hold", None)
    ph = ph(t) if callable(ph) else None
    # A group's part of the node (or a group view's restriction of it) states the
    # node's own constraints for a subset of its sources, so the node's held-out
    # hold rate is its confidence too - unless the part states its own windows
    # (pack O: parts stated the min-of-parts 0.09 where their held-out hold was
    # 1.0, ptree open issue 2)
    if ph is not None and math.isfinite(float(ph)) and not own_when:
        conf = float(ph)
    elif ph is not None and math.isfinite(float(ph)):
        # a part with its own windows states the node's other constraints (whose
        # held-out hold rate is p_hold) plus its own windows (their stated
        # coverage): both must hold, so min(p_hold, own coverage). Evaluator
        # round 3: the min of the parts' NOMINAL coverages stated 0.85-0.97 for
        # mail parts whose node held ~0.5 of its tests, and 0.10 (a workflow
        # edge's dependency strength, no coverage at all) for /docs parts that
        # held 1.0 - the two ends of PG2's reliability diagram
        own = when_c if (when_c is not None and math.isfinite(float(when_c))) else 1.0
        conf = float(min(float(ph), float(own)))
    sys_label = c.key
    addr = ""
    root = c.ptm.kinds[kind].nodes.get(c.ptm.kinds[kind].root) if c.ptm is not None else None
    if root is not None:
        addr = _address(root, t)
    first = PR.local_date(nd.first_seen, c.tz)
    last = PR.local_date(nd.last_seen, c.tz)
    # (round 3) doing what: the action's display name after its page
    # ('POST /fin/approval/{num}/approve（审批）')
    rt = PR.route_text(route)
    aw = action_word(route, c.config)
    rt_zh, rt_en = (f"{rt}（{aw[0]}）", f"{rt} ({aw[1]})") if aw else (rt, rt)
    zh, _ = PR.sentence(sys_label, addr, when_zh, when_en, who_zh, who_en, rt_zh,
                        czh, cen, bzh, ben, fzh, fen, conf, first, last, nd.version, nd.cver, nd.state)
    _, en = PR.sentence(sys_label, addr, when_zh, when_en, who_zh, who_en, rt_en,
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
    ctx_ev = [[a, int(l), sorted(str(v) for v in vals), bool(neg)] for a, l, vals, neg in nd.ctx]
    if part is not None:
        ctx_ev.append(["net.src", 3, [f"grp:{x}" for x in (part[4] if len(part) > 4 else [part[0]])], False])
        pid = f"{pid}|grp:{part[0]}"
    ev: Dict[str, Any] = {
        "route": route, "system": c.key, "kind": kind, "node": nd.id,
        "context": ctx_ev,
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


BIND_MIN_CARD = FD.BIND_MIN_CARD  # distinct payload values system-wide for a binding to be stated
PART_SHARE = 0.05                # a group's part of a node is stated when it holds >= 5 % of its mass
PART_MAX = 8                     # parts per node (largest first)


def _impurity(d: Optional[Mapping[str, Any]], route: str) -> float:
    """Mass share of a node's subtree NOT on its rendered route (route index;
    0 when the index has no data, i.e. the route is fixed by context)."""
    if not d:
        return 0.0
    vals = {r: (float(v[0]) if isinstance(v, (tuple, list)) else float(v)) for r, v in d.items()}
    tm = sum(vals.values())
    return 0.0 if tm <= 0 else max(0.0, 1.0 - vals.get(route, 0.0) / tm)


def _configured_ips(config: Mapping[str, Any]) -> Dict[str, List[str]]:
    """Configured department names -> their listed addresses (who_group_names
    [{name, ips}]; CIDR-defined names have no enumerable members)."""
    out: Dict[str, List[str]] = {}
    for it in (config or {}).get("who_group_names") or []:
        if isinstance(it, Mapping) and it.get("name") and it.get("ips"):
            out.setdefault(str(it["name"]), []).extend(str(x) for x in it["ips"])
    return out


def _dept_of(c: Any, ips: Sequence[str]) -> Optional[str]:
    """The one configured department every address belongs to (its learned
    group's `dept`, else the department's configured addresses), else None."""
    if not ips:
        return None
    cfg_ips = getattr(c, "_cfg_ips", None)
    if cfg_ips is None:
        cfg_ips = {}
        for name, lst in _configured_ips(getattr(c, "config", None) or {}).items():
            for ip in lst:
                cfg_ips.setdefault(ip, name)
        try:
            c._cfg_ips = cfg_ips
        except AttributeError:                      # pragma: no cover
            pass
    out = set()
    for ip in ips:
        ip = ip[7:] if ip.startswith("shared:") else ip
        g = (getattr(c, "ip2g", None) or {}).get(ip)
        d = ((getattr(c, "groups", None) or {}).get(g) or {}).get("dept") if g is not None else None
        d = d or cfg_ips.get(ip)
        if not d:
            return None
        out.add(str(d))
    return next(iter(out)) if len(out) == 1 else None


def _auto_group_of(c: Any, ips: Sequence[str]) -> bool:
    """All addresses belong to ONE learned group whose name is automatic."""
    gs = {(getattr(c, "ip2g", None) or {}).get(ip[7:] if ip.startswith("shared:") else ip) for ip in ips}
    if len(gs) != 1 or None in gs:
        return False
    gr = (getattr(c, "groups", None) or {}).get(next(iter(gs))) or {}
    return gr.get("name_source") == "auto"


def _pool_group_of(c: Any, prefixes: Sequence[str]) -> Optional[str]:
    """The configured name of the pool group (P11 rec 'pool') whose pool
    prefix holds every stated prefix, else None."""
    import ipaddress
    nets = []
    for p in prefixes:
        try:
            nets.append(ipaddress.ip_network(p, strict=False))
        except ValueError:
            return None
    if not nets:
        return None
    for g, gr in (getattr(c, "groups", None) or {}).items():
        pool = gr.get("pool")
        if not pool or gr.get("name_source") != "config":
            continue
        try:
            pn = ipaddress.ip_network(pool, strict=False)
        except ValueError:
            continue
        if all(n.version == pn.version and n.subnet_of(pn) for n in nets):
            return str(gr.get("name") or g)
    return None


def _restrict_closed(g_ent: Optional[Mapping[str, Any]], bent: Optional[Mapping[str, Any]],
                     members: Set[str]) -> Optional[Mapping[str, Any]]:
    """A group's part (or group view) of a node states the node's closed value
    sets for ITS members: when P08 binds every member to one value of a closed
    set (net.src -> attr), the part's closed set is its members' bound values.
    Evaluator round 3: the 销售部 part of the 财务部+销售部 login node stated
    the node's 23 user names (销售部's 20 and 财务部's 3), not its own 20."""
    if not g_ent or not members or not isinstance(bent, Mapping):
        return g_ent
    attrs = g_ent.get("attrs") or {}
    new = None
    for pk, rec in (bent.get("pairs") or {}).items():
        if not isinstance(rec, Mapping) or rec.get("dir") == "rev":
            continue
        X, Y = rec.get("x"), rec.get("y")
        if not X or not Y:
            X, _, Y = str(pk).partition("->")
        if X != "net.src" or Y not in attrs:
            continue
        ga = attrs[Y]
        if not isinstance(ga, Mapping) or ga.get("closed") is None:
            continue
        tab = rec.get("table") or {}
        vals = set()
        for m in members:
            ent = tab.get(m)
            if not isinstance(ent, Mapping) or not ent.get("bound") or ent.get("top") is None:
                vals = None
                break
            vals.add(str(ent["top"]))
        closed = {str(v) for v in ga["closed"]}
        if not vals or not vals <= closed or vals == closed:
            continue
        if new is None:
            new = dict(attrs)
        new[Y] = dict(ga, closed=sorted(vals), part_of_closed=len(closed))
    return g_ent if new is None else dict(g_ent, attrs=new)


def _ctx_admits(nd: Any, ip: str, ip2g: Mapping[str, Any]) -> bool:
    """Whether address `ip` satisfies every net.src constraint of the node's
    context (prefixes, addresses, learned groups); False when one cannot be
    decided (a region)."""
    import ipaddress
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    for attr, _l, vals, neg in getattr(nd, "ctx", ()):
        if attr != "net.src":
            continue
        hit = False
        for v in vals:
            v = str(v)
            if v.startswith("grp:"):
                hit = str(ip2g.get(ip)) == v[4:]
            elif "/" in v:
                try:
                    hit = a in ipaddress.ip_network(v, strict=False)
                except ValueError:
                    return False
            elif v.startswith("reg:"):
                return False
            else:
                hit = v == ip
            if hit:
                break
        if hit == bool(neg):
            return False
    return True


def _discriminates(rec: Mapping[str, Any]) -> bool:
    """pfd.binding_discriminates (shared with P04's held-out test). A payload
    with a handful of values system-wide that every source shares binds every
    source to the same value - a constant of the action, already stated as
    content (the reason for BIND_MIN_CARD, which alone dropped the finance
    username binding: 3 values system-wide)."""
    return FD.binding_discriminates(rec)


GRP_GAIN_N = 200.0               # behaviour evidence units before P12's group gain is believed


def _grp_gain(sp: Any) -> Optional[float]:
    """P12's held-out behaviour gain (bits/event) of conditioning on the P11
    group (who level 3, system_profile measurements 'who_pred'); None while
    unmeasured."""
    try:
        meas = (sp or {}).get("measurements") or {}
        wp = meas.get("who_pred")
        if wp is None or len(wp) < 4 or float(meas.get("who_pred_n") or 0.0) < GRP_GAIN_N:
            return None
        g = float(wp[3])
        return g if math.isfinite(g) else None
    except Exception:                               # pragma: no cover
        return None


def group_parts(c: _Ctx, nd: Any, t: float, impure: float = 0.0, route: Optional[str] = None
                ) -> List[Tuple[Any, ...]]:
    """The learned groups (P11) that make up a node's sources: [(g, members
    seen at the node, name, share of the node's mass[, group ids])], largest
    first, when the node's sources span >= 2 groups (a node of one group is
    already stated about that group).

    Why: the requirement's system view is "OA 服务器的某类人会在哪个时间段访问
    我什么页面干什么事". Where several departments do the same thing (GET /docs
    by 综合部, 财务部 and 销售部 with the same sizes and hours) the lattice
    correctly keeps ONE node - no target differs, so rule (V) never splits it -
    and its who is the union. The per-group statements are the node's
    constraints restricted to each learned group (a refinement that holds
    wherever the node's constraints hold); where a group does behave
    differently the tree splits and the group gets its own node instead.
    Bounded: <= PART_MAX parts per node, read from the node's grp-level who
    summary (never from the population). `impure` = the node's mass share on
    other routes (a route-dominant but mixed node): a group's share must exceed
    PART_SHARE + impure, else the whole part could be traffic of those other
    routes (pack O: '销售部（19 个 IP）访问 GET /fin/approval/list').

    Members: a group member is part of the node when the node's IP-level
    summary shows it OR - at a node without an address context, given the
    action `route` - its own P11 signature holds the action as a recurring use
    (conformity.signature_share >= SIG_STANDING, P03's colleague rule): the
    IP-level summary keeps WHO_K = 8 heavy hitters, and at the 25-source
    GET /docs node of pack O the 综合部 part listed 192.168.1.23 alone and the
    财务部 part 192.168.2.11 alone (PG1 who). Bounded: <= MEMBERS_CHECKED
    members per group.

    Departments: groups P11 names as roles of ONE configured department (rec
    'dept', e.g. its approver and its report writers) are stated as one part
    of that department - the "某类人" an operator configured - with the roles'
    group ids; groups without a configured department stay one part each
    (pack O seed 0: GET /docs had parts {192.168.1.23} and {192.168.1.21}
    for 综合部's two roles, neither of them the department)."""
    if not c.ip2g or c.mode == "none":
        return []
    gg = getattr(c, "grp_gain", None)
    if gg is not None and gg <= 0.0:
        # the system's learned groups carry no behavioural information (P12's
        # held-out gain of conditioning on the group <= 0): a "某类人" part would
        # name a group that does not predict what its members do. Pack O's
        # public portal (gain -2.2 bits/event): parts of one returning visitor
        # each ('G263（10.60.103.206）访问 POST /login'), against PG3's portal
        # login who in {prefix, reg, any}
        return []
    lv3 = nd.who.levels[3] if len(nd.who.levels) > 3 else None
    if lv3 is None:
        return []
    tot = lv3.total(t)
    if tot <= 0:
        return []
    parts = []
    for key, cnt, g_, e_ in lv3.items(t):
        k = str(key)
        if not k.startswith("grp:") or k == GRP_NONE:
            continue
        parts.append((k[4:], float(cnt) / tot))
    lv0 = nd.who.levels[0]
    seen = {str(ip) for ip, *_ in lv0.items(t)}
    sigs = getattr(c, "sigs", None)
    use_sig = route is not None and sigs is not None
    # at a node with an address context, a signature member counts only when
    # its address satisfies that context (evaluator round 3: the 销售部 part
    # of the 财务部+销售部 login node, context 192.168.2.0/24 + 192.168.3.0/24,
    # listed the 5 of 20 members among the node's heavy hitters)
    sig_ctx = any(a == "net.src" for a, l, vals, neg in getattr(nd, "ctx", ()))
    # merge the roles of a configured department
    units: Dict[str, Dict[str, Any]] = {}
    for g, share in parts:
        gr = c.groups.get(g) or {}
        dept = gr.get("dept")
        uk = f"dept:{dept}" if dept else g
        u = units.setdefault(uk, {"gids": [], "share": 0.0,
                                  "name": str(dept) if dept else str(gr.get("name") or g)})
        u["gids"].append(g)
        u["share"] += share
    if len(units) < 2:
        return []
    out = []
    for uk, u in sorted(units.items(), key=lambda kv: -kv[1]["share"]):
        if u["share"] < PART_SHARE + impure:
            continue
        mem: List[str] = []
        for g in u["gids"]:
            allm = [str(m) for m in (c.groups.get(g) or {}).get("members") or [] if "/" not in str(m)]
            for i, m in enumerate(allm):
                # the members SEEN at the node; none seen -> no part (the group key can
                # outlive its membership: an address that left the group, or a heavy-
                # hitter cut; listing the whole group then claimed 19 sales IPs read
                # finance's approval list on pack O)
                if m in seen or (use_sig and i < CF.MEMBERS_CHECKED
                                 and (not sig_ctx or _ctx_admits(nd, m, c.ip2g))
                                 and CF.signature_share(sigs, c.key, m, t, route) >= CF.SIG_STANDING):
                    mem.append(m)
        if uk.startswith("dept:"):
            # the department's configured addresses that P11 has not put in any
            # group (pack O: the finance approver 192.168.2.10 had no group, so
            # 财务部's part of GET /docs listed .11 and .12 only)
            for m in _configured_ips(getattr(c, "config", None) or {}).get(uk[5:], ()):
                if m in mem or m in c.ip2g:
                    continue
                if m in seen or (use_sig and (not sig_ctx or _ctx_admits(nd, m, c.ip2g))
                                 and CF.signature_share(sigs, c.key, m, t, route) >= CF.SIG_STANDING):
                    mem.append(m)
        if not mem:
            continue
        name = u["name"] if len(u["gids"]) > 1 or not uk.startswith("dept:") else \
            str((c.groups.get(u["gids"][0]) or {}).get("name") or u["gids"][0])
        if len(u["gids"]) > 1:
            out.append((uk, sorted(set(mem), key=PR._ip_sort), name, u["share"], tuple(u["gids"])))
        else:
            out.append((u["gids"][0], sorted(set(mem), key=PR._ip_sort), name, u["share"]))
        if len(out) >= PART_MAX:
            break
    return out if len(out) >= 2 else []


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
        rd = c.route_dist(kind)
        for nd, route, act in _walk(tree, now, rd):
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
            # the "某类人" parts of a node shared by several learned groups
            if nd.split is None or nd.id == act:
                for part in group_parts(c, nd, now, _impurity(rd.get(nd.id) if rd else None, route), route):
                    ps = node_statement(c, kind, nd, route, part=part)
                    if ps is None:
                        continue
                    ps["act_node"] = act
                    ps["mass"] = float(st["mass"]) * part[3]
                    stmts.append(ps)
                    a["statements"].append(ps["id"])
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


def _suspects(nd: Any, t: float) -> List[str]:
    """Sources P04 keeps out of a node's who summary as foreign (P03 damped
    them there; pnode.WhoSummary.suspects), [] for summaries without them."""
    f = getattr(nd.who, "suspects", None)
    try:
        return list(f(t)) if callable(f) else []
    except Exception:                               # pragma: no cover
        return []


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


# Display vocabulary for action names (a display aid, never a model input): the
# first rule whose pattern matches a route's literal path words names the action.
# Config progressive.action_names [[regex, zh, en], ...] is tried first (an
# operator's / API catalogue's names); unmatched routes are shown as the route.
ACTION_NAMES: Tuple[Tuple[str, str, str], ...] = (
    (r"(^|/)(login|signin|logon|auth)(/|$)", "登录", "log in"),
    (r"(^|/)(logout|signout)(/|$)", "退出", "log out"),
    (r"approv|/flow/", "审批", "approve"),
    (r"(^|/)report(s)?(/|$)", "报告", "report"),
    (r"(^|/)comment", "评论", "comments"),
    (r"(^|/)(docs?|documents?|files?)(/|$)", "文档", "documents"),
    (r"(^|/)(mail|inbox)(/|$)|^TLS mail", "邮件", "mail"),
    (r"(^|/)voucher", "凭证", "vouchers"),
    (r"(^|/)(customer|crm)(/|$)", "客户", "customers"),
    (r"(^|/)news(/|$)", "新闻", "news"),
    (r"(^|/)(export|backup)(/|$)", "导出", "export"),
    (r"(^|/)health(/|$)", "健康检查", "health check"),
    (r"(^|/)(home|index|portal)(/|$)", "首页", "home page"),
    (r"(^|/)ledger", "账簿", "ledger"),
    (r"^TLS (git|svn|code|gitlab)\.", "代码库", "code repository"),
)


def action_name(route: str, config: Optional[Mapping[str, Any]] = None) -> Tuple[str, str]:
    """(zh, en) display name of an action: '<name>（<METHOD path>）', or the
    route text when no vocabulary entry matches."""
    import re
    rt = PR.route_text(route)
    m, path = PR.route_parts(route)
    rules: List[Tuple[str, str, str]] = []
    for it in (EV.pconfig(config or {}).get("action_names") or []):
        try:
            p, zh, en = it
            rules.append((str(p), str(zh), str(en)))
        except (TypeError, ValueError):
            continue
    rules.extend(ACTION_NAMES)
    for p, zh, en in rules:
        try:
            if re.search(p, path, re.I) or re.search(p, rt, re.I):
                if m == "GET" and zh in ("审批", "报告", "凭证", "评论", "账簿"):
                    zh, en = f"查看{zh}", f"view {en}"
                elif m in PR.WRITE_METHODS and zh in ("报告",):
                    zh, en = "提交报告", "submit reports"
                return f"{zh}（{rt}）", f"{en} ({rt})"
        except re.error:
            continue
    return rt, rt


def action_word(route: str, config: Optional[Mapping[str, Any]] = None) -> Optional[Tuple[str, str]]:
    """(zh, en) bare display name of an action ('审批', 'approve'), None when
    no vocabulary entry matches (the route text then speaks for itself)."""
    zh, en = action_name(route, config)
    rt = PR.route_text(route)
    if zh == rt:
        return None
    suf_zh, suf_en = f"（{rt}）", f" ({rt})"
    return (zh[:-len(suf_zh)] if zh.endswith(suf_zh) else zh, en[:-len(suf_en)] if en.endswith(suf_en) else en)


def activity_statement(g: str, name: str, key: str, acts: Sequence[Mapping[str, Any]], share_sys: float,
                       members: Set[str], subject: str,
                       config: Optional[Mapping[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """The user view's "which systems a group uses and what it does there"
    (综合部 访问 oa：登录、审批、提交报告): P11's per-system action mix of the
    group (its learned signatures), each action with the members that do it
    when not all of them do."""
    if not acts:
        return None
    pz, pe, rows = [], [], []
    for a in acts:
        zh, en = action_name(str(a.get("action")), config)
        mem = [str(m) for m in a.get("members") or []]
        if mem and len(mem) <= PR.MEMBERS_LISTED:
            zh += f"[{PR.join_zh(mem)}]"
            en += f" [{PR.join_en(mem)}]"
        pz.append(zh)
        pe.append(en)
        rows.append({"action": a.get("action"), "share": a.get("share"), "members": mem,
                     "support": a.get("support")})
    zh = f"{name} 访问 {key}（占其活动 {PR.pct(share_sys)}）：{PR.join_zh(pz)}"
    en = f"{name} uses {key} ({PR.pct(share_sys)} of its activity): {PR.join_en(pe)}"
    return {"id": f"act:{g}:{key}", "pattern_id": f"act:{g}:{key}", "view": "group",
            "subject": subject, "text_zh": zh, "text_en": en, "support": float(len(members)),
            "confidence": None, "state": "confirmed", "version": 1, "cver": 0,
            "facets": ["functional", "relational"],
            "evidence": {"activity": True, "system": key, "group": g, "actions": rows,
                         "who": {"level": "grp", "items": [f"grp:{g}"], "members": sorted(members)}}}


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
        if share_sys >= GROUP_SYS_SHARE:
            used.append(key)
            act_st = activity_statement(g, name, key, (gr.get("actions") or {}).get(key) or [],
                                        share_sys, members, subject, config)
            if act_st is not None:
                # how alike the members behave (P11's within-group similarity)
                act_st["confidence"] = float(gr.get("cohesion") or 0.0)
                stmts.append(act_st)
            for nd, route, act in _walk(tree, c.now, c.route_dist(EV.KIND_TXN)):
                if route is None or nd.state not in RENDERED:
                    continue
                gm = _group_mass(nd, g, members, now)
                if gm < GROUP_NODE_SHARE:
                    continue
                # (round 3) the group's part of the node: its members there, its
                # own windows when the node's reservoir holds them - not the
                # node's whole population ('来自 10.50.0.0/16、192.168.0.0/16
                # （约 400 个 IP）' in 销售部's view of the mail node)
                st = node_statement(c, EV.KIND_TXN, nd, route, view="group", subject=subject,
                                    restrict=members)
                if st is not None and (st["evidence"].get("who") or {}).get("level") != "ip":
                    mem = _members_at(c, nd, members, route, now)
                    if mem:
                        st = node_statement(c, EV.KIND_TXN, nd, route, view="group", subject=subject,
                                            part=(g, mem, name, gm)) or st
                if st is not None:
                    st["act_node"] = act
                    stmts.append(st)
        # negative statements: who-closed write nodes of this system the group never reached
        cw = cache.get(("cw", key))
        if cw is None:
            cw = cache[("cw", key)] = _closed_write_nodes(c)
        # (round 3) "never" must agree with what the group's own signatures say
        # it does: P11's action mix of the group and its members' recurring
        # uses. The node summaries carry group LABELS as of learning time, so a
        # group P11 re-formed under a new id had no mass anywhere yet - pack O:
        # '研发·oa POST /login 访问 oa：登录（POST /login）' next to '… 在 oa 中
        # 从未执行写操作（…登录（POST /login））'
        did = {str(a.get("action")) for a in (gr.get("actions") or {}).get(key) or []}

        def reached(nd: Any, route: str) -> bool:
            return _group_mass(nd, g, members, now) > 0.0 or route in did or \
                _sig_uses(c, members, route, now)
        never = [(nd, route) for nd, route in cw if not reached(nd, route)]
        # "never wrote in this system" must hold on EVERY write node, not only the
        # who-closed ones: a group whose own write node is still a candidate (or
        # not closed) did write there (the sentence was false on pack O's OA)
        wr = cache.get(("wr", key))
        if wr is None and never:
            wr = cache[("wr", key)] = [(nd, route) for nd, route, _ in _walk(tree, c.now,
                                                                              c.route_dist(EV.KIND_TXN))
                                       if route is not None and PR.is_write(route)]
        wrote = bool(never) and (any(PR.is_write(a) for a in did) or any(reached(nd, r) for nd, r in wr))
        if never and wrote:
            # (round 3) "never does what" inside a system the group uses: the
            # closed write actions of the system it never performed ('销售部 在
            # oa 中从未执行：审批（POST /approval/{num}/approve）')
            stmts.append(_partial_negative(g, name, key, never, int(root.days_total or 0), members,
                                           subject, config, now))
        if never and not wrote:
            n_days = int(root.days_total or 0)
            zh, en = PR.negative_sentence(name, key, n_days)
            routes = sorted({PR.route_text(r) for _, r in never})
            names = sorted({action_name(r, config) for _, r in never})
            zh += "（封闭的写操作：" + PR.join_zh([x[0] for x in names][:6]) + "）"
            en += " (closed write actions: " + PR.join_en([x[1] for x in names][:6]) + ")"
            # members whose attempts there were judged foreign (P03 damped them, P04
            # kept them out of the pattern's who): stated, not hidden - "never" is
            # about the group's learned behaviour, the attempts are findings
            sus = sorted({str(ip) for nd, _ in never for ip in _suspects(nd, now) if str(ip) in members},
                         key=PR._ip_sort)
            if sus:
                zh += f"；{PR.join_zh(sus)} 的尝试被判定为越权（未学习）"
                en += f"; attempts by {PR.join_en(sus)} were judged foreign (not learned)"
            stmts.append({"id": f"neg:{g}:{key}", "pattern_id": f"neg:{g}:{key}", "view": "group",
                          "subject": subject, "text_zh": zh, "text_en": en,
                          "support": float(sum(nd.n_c(now) for nd, _ in never)),
                          "confidence": float(min(1.0 - nd.who.levels[nd.who.closed_level(now, nd.n_days())].unseen(now)
                                                  for nd, _ in never)),
                          "state": "confirmed", "version": 1, "cver": 0,
                          "facets": ["relational", "risk"],
                          "evidence": {"negative": True, "scope": "system", "target_system": key,
                                       "system": key, "routes": routes, "route_keys": sorted({r for _, r in never}),
                                       "group": g, "foreign_attempts": sus,
                                       "n_days": n_days,
                                       "closed_zh": [x[0] for x in names][:6],
                                       "closed_en": [x[1] for x in names][:6],
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


def _sig_uses(c: Any, members: Set[str], route: str, t: float) -> bool:
    """Some member's own P11 signature holds the action as a recurring use
    (P03's colleague rule, <= MEMBERS_CHECKED members read)."""
    sigs = getattr(c, "sigs", None)
    if sigs is None:
        return False
    for i, m in enumerate(sorted(members)):
        if i >= CF.MEMBERS_CHECKED:
            break
        if "/" not in m and CF.signature_share(sigs, c.key, m, t, route) >= CF.SIG_STANDING:
            return True
    return False


def _members_at(c: Any, nd: Any, members: Set[str], route: Optional[str], t: float) -> List[str]:
    """The group's members at a node: seen in its IP-level summary, or - at a
    node without an address context - holding the action in their own
    signatures (group_parts' rule)."""
    seen = {str(ip) for ip, *_ in nd.who.levels[0].items(t)}
    sigs = getattr(c, "sigs", None)
    use_sig = route is not None and sigs is not None
    # at a node with an address context, a signature member counts only when
    # its address satisfies that context (evaluator round 3: the 销售部 part
    # of the 财务部+销售部 login node, context 192.168.2.0/24 + 192.168.3.0/24,
    # listed the 5 of 20 members among the node's heavy hitters)
    sig_ctx = any(a == "net.src" for a, l, vals, neg in getattr(nd, "ctx", ()))
    out = []
    for i, m in enumerate(sorted(members, key=PR._ip_sort)):
        if "/" in m:
            continue
        if m in seen or (use_sig and i < CF.MEMBERS_CHECKED
                         and CF.signature_share(sigs, c.key, m, t, route) >= CF.SIG_STANDING):
            out.append(m)
    return out


def _partial_negative(g: str, name: str, key: str, never: Sequence[Tuple[Any, str]], n_days: int,
                      members: Set[str], subject: str, config: Mapping[str, Any], now: float
                      ) -> Dict[str, Any]:
    """'<group> 在 <system> 中从未执行：<closed write actions>（n 天、0 次）' for a
    system the group does write in (round 3)."""
    names = sorted({action_name(r, config) for _, r in never})
    zh = f"{name} 在 {key} 中从未执行：" + PR.join_zh([x[0] for x in names][:6]) + f"（{n_days} 天、0 次）"
    en = f"{name} has never performed on {key}: " + PR.join_en([x[1] for x in names][:6]) + \
        f" ({n_days} days, 0 times)"
    conf = float(min(1.0 - nd.who.levels[nd.who.closed_level(now, nd.n_days())].unseen(now) for nd, _ in never))
    return {"id": f"neg:{g}:{key}:actions", "pattern_id": f"neg:{g}:{key}:actions", "view": "group",
            "subject": subject, "text_zh": zh, "text_en": en,
            "support": float(sum(nd.n_c(now) for nd, _ in never)), "confidence": conf,
            "state": "confirmed", "version": 1, "cver": 0, "facets": ["relational", "risk"],
            "evidence": {"negative": True, "scope": "actions", "target_system": key, "system": key,
                         "routes": sorted({PR.route_text(r) for _, r in never}),
                         "route_keys": sorted({r for _, r in never}), "group": g, "n_days": n_days,
                         "closed_zh": [x[0] for x in names][:6], "closed_en": [x[1] for x in names][:6],
                         "who": {"level": "grp", "items": [f"grp:{g}"], "members": sorted(members)}}}


def dept_view(name: str, views: Sequence[Mapping[str, Any]], groups: Mapping[str, Mapping[str, Any]],
              config: Mapping[str, Any], now: float) -> Optional[Dict[str, Any]]:
    """The user view of a configured department whose members P11 learned as
    several groups (its roles): "综合部 访问 oa：登录、文档、审批[192.168.1.21]、
    提交报告[192.168.1.23、10.168.7.121]；在 finance 中从未执行写操作".

    Composed from the rendered views of its groups (no model is read again):
    per system the union of the groups' action mixes - an action's members are
    the members of the groups that do it (or the members listed by P11), its
    share the groups' shares weighted by their sizes; a negative statement only
    for a system where EVERY group's negative statement holds; the groups'
    node statements as they are. Pack O: 综合部's approver and its report
    writers are two learned groups, so neither group view alone stated what
    the requirement asks for (the department's systems and actions)."""
    gids = [str(v.get("group")) for v in views]
    if len(gids) < 2:
        return None
    members: Set[str] = set()
    for g in gids:
        members |= {str(m) for m in (groups.get(g) or {}).get("members") or [] if "/" not in str(m)}
    if not members:
        return None
    subject = f"{GROUP_PREFIX}dept:{name}"
    n_tot = float(sum(len((groups.get(g) or {}).get("members") or []) for g in gids)) or 1.0
    acts: Dict[str, Dict[str, Dict[str, Any]]] = {}
    sys_share: Dict[str, float] = {}
    for g in gids:
        gr = groups.get(g) or {}
        gm = [str(m) for m in gr.get("members") or [] if "/" not in str(m)]
        wg = len(gm) / n_tot
        for key, sh in (gr.get("systems") or {}).items():
            sys_share[key] = sys_share.get(key, 0.0) + wg * float(sh)
        for key, lst in (gr.get("actions") or {}).items():
            for a in lst:
                r = acts.setdefault(key, {}).setdefault(str(a.get("action")),
                                                        {"action": a.get("action"), "share": 0.0, "members": set()})
                r["share"] += wg * float(a.get("share") or 0.0)
                r["members"] |= set(str(m) for m in (a.get("members") or gm) if "/" not in str(m))
    stmts: List[Dict[str, Any]] = []
    used = []
    for key in sorted(acts):
        if sys_share.get(key, 0.0) < GROUP_SYS_SHARE:
            continue
        used.append(key)
        rows = []
        for r in sorted(acts[key].values(), key=lambda x: (-x["share"], str(x["action"]))):
            mem = sorted(r["members"], key=PR._ip_sort)
            rows.append({"action": r["action"], "share": round(r["share"], 4),
                         "members": mem if set(mem) != members else [],
                         "support": round(len(mem) / len(members), 3)})
        st = activity_statement(f"dept:{name}", name, key, rows[:2 * 8], sys_share[key], members,
                                subject, config)
        if st is not None:
            stmts.append(st)
    negs: Dict[str, List[Mapping[str, Any]]] = {}
    never_by: Dict[str, List[Optional[Set[str]]]] = {}
    for vi, v in enumerate(views):
        for st in v.get("statements") or []:
            ev = st.get("evidence") or {}
            if not ev.get("negative"):
                continue
            key = str(ev.get("target_system"))
            if ev.get("scope", "system") == "system":
                negs.setdefault(key, []).append(st)
            lst_ = never_by.setdefault(key, [None] * len(views))
            lst_[vi] = (lst_[vi] or set()) | set(ev.get("route_keys") or [])
    # (round 3) "never does what" of the department inside a system it uses:
    # the closed write actions NONE of its roles performed
    for key, sets in sorted(never_by.items()):
        if len(negs.get(key) or []) >= len(views) or any(x is None for x in sets):
            continue
        inter = set.intersection(*sets)
        if inter:
            nz = sorted({action_name(r, config) for r in inter})
            n_days = max(int((st.get("evidence") or {}).get("n_days") or 0)
                         for v in views for st in v.get("statements") or []
                         if (st.get("evidence") or {}).get("negative")
                         and str((st.get("evidence") or {}).get("target_system")) == key)
            stmts.append({"id": f"neg:dept:{name}:{key}:actions", "pattern_id": f"neg:dept:{name}:{key}:actions",
                          "view": "group", "subject": subject,
                          "text_zh": f"{name} 在 {key} 中从未执行：" + PR.join_zh([x[0] for x in nz][:6])
                          + f"（{n_days} 天、0 次）",
                          "text_en": f"{name} has never performed on {key}: " + PR.join_en([x[1] for x in nz][:6])
                          + f" ({n_days} days, 0 times)",
                          "support": 0.0, "confidence": None, "state": "confirmed", "version": 1, "cver": 0,
                          "facets": ["relational", "risk"],
                          "evidence": {"negative": True, "scope": "actions", "target_system": key, "system": key,
                                       "routes": sorted({PR.route_text(r) for r in inter}),
                                       "route_keys": sorted(inter), "group": f"dept:{name}", "groups": gids,
                                       "who": {"level": "grp", "items": [f"grp:{g}" for g in gids],
                                               "members": sorted(members, key=PR._ip_sort)}}})
    for key, lst in sorted(negs.items()):
        if len(lst) < len(views):
            continue                       # some role of the department did write there
        evs = [st.get("evidence") or {} for st in lst]
        n_days = max(int(e.get("n_days") or 0) for e in evs)
        zh, en = PR.negative_sentence(name, key, n_days)
        routes = sorted({r for e in evs for r in e.get("routes") or []})
        cz = sorted({x for e in evs for x in e.get("closed_zh") or []})
        ce = sorted({x for e in evs for x in e.get("closed_en") or []})
        if cz:
            zh += "（封闭的写操作：" + PR.join_zh(cz[:6]) + "）"
            en += " (closed write actions: " + PR.join_en(ce[:6]) + ")"
        sus = sorted({ip for e in evs for ip in e.get("foreign_attempts") or []}, key=PR._ip_sort)
        if sus:
            zh += f"；{PR.join_zh(sus)} 的尝试被判定为越权（未学习）"
            en += f"; attempts by {PR.join_en(sus)} were judged foreign (not learned)"
        stmts.append({"id": f"neg:dept:{name}:{key}", "pattern_id": f"neg:dept:{name}:{key}", "view": "group",
                      "subject": subject, "text_zh": zh, "text_en": en,
                      "support": float(sum(float(st.get("support") or 0.0) for st in lst)),
                      "confidence": float(min(float(st.get("confidence") or 0.0) for st in lst)),
                      "state": "confirmed", "version": 1, "cver": 0, "facets": ["relational", "risk"],
                      "evidence": {"negative": True, "scope": "system", "target_system": key, "system": key,
                                   "routes": routes, "route_keys": sorted({r for e in evs for r in e.get("route_keys") or []}),
                                   "group": f"dept:{name}", "groups": gids, "foreign_attempts": sus,
                                   "who": {"level": "grp", "items": [f"grp:{g}" for g in gids],
                                           "members": sorted(members, key=PR._ip_sort)}}})
    seen = {st["id"] for st in stmts}
    for v in views:
        for st in v.get("statements") or []:
            ev = st.get("evidence") or {}
            if ev.get("activity") or ev.get("negative") or st.get("id") in seen:
                continue
            seen.add(st["id"])
            stmts.append(st)
    zh = f"{name}（{len(members)} 个 IP，{len(gids)} 个行为群组）使用 {PR.join_zh(used) or '（尚无系统）'}"
    en = f"{name} ({len(members)} IPs, {len(gids)} behavioural groups) uses {PR.join_en(used) or '(no system yet)'}"
    return {"fmt": 1, "view": "group", "subject": subject, "group": f"dept:{name}", "groups": gids,
            "name": name, "name_source": "config", "updated": now,
            "header": {"text_zh": zh, "text_en": en, "members": sorted(members, key=PR._ip_sort),
                       "covers": sorted({c for g in gids for c in (groups.get(g) or {}).get("covers") or []}),
                       "systems": {k: round(v, 4) for k, v in sorted(sys_share.items())}},
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
        groups = wg.get("groups") or {}
        by_dept: Dict[str, List[Dict[str, Any]]] = {}
        for g, gr in sorted(groups.items()):
            if not gr.get("materialise", True):
                continue
            if not self.entity_due(("p14g", g), now, self.view_period_s):
                continue
            gv = group_view(store, g, ctx.config, now, cache)
            if gv is None:
                continue
            self._put(store, ORG, f"{GROUP_PREFIX}{g}", gv, now)
            n += 1
            if gr.get("dept"):
                by_dept.setdefault(str(gr["dept"]), [])
        # a configured department learned as several groups (its roles): one
        # department view composed from its groups' current views
        for name in sorted(by_dept):
            gvs = []
            for g, gr in sorted(groups.items()):
                if str(gr.get("dept") or "") != name:
                    continue
                v = store.get_model(ORG, f"{GROUP_PREFIX}{g}", MP.PVIEWS)
                if isinstance(v, Mapping) and v.get("group") == g:
                    gvs.append(v)
            dv = dept_view(name, gvs, groups, ctx.config, now)
            if dv is None and isinstance(store.get_model(ORG, f"{GROUP_PREFIX}dept:{name}", MP.PVIEWS), Mapping):
                # the department is one learned group again: retire its composed view
                dv = {"fmt": 1, "view": "group", "subject": f"{GROUP_PREFIX}dept:{name}",
                      "group": f"dept:{name}", "name": name, "retired": True, "updated": now,
                      "statements": []}
            if dv is not None:
                self._put(store, ORG, f"{GROUP_PREFIX}dept:{name}", dv, now)
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
