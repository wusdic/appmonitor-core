"""Test scaffolding for the content fitters P06-P08 (W-P4): a stand-in for
P04 that routes learned rows through a fixed pattern tree and updates node
summaries the way docs/lib3/progressive.md §6.5.1 prescribes (core, targets,
and the pair sketches P08 requests in model.pwant['pairs']), with burst
evidence units (§6.5.4). It plays the counting role only; every fitted
constraint the tests inspect comes from the real engines.

Also: `ga_login_events` draws the requirement's example (综合部 GA logging into
OA, one login per member IP per workday) from the org generator's own truth
program (pmetrics.holdout_events on pack O's truth row), and `statement`
turns the fitted models of one node into a statement in the pmetrics
contract so the eval scorer can judge it.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import psketch as PS
from app.engines.behavior.lib import timebins as TB
from app.engines.behavior.lib.phier import Shaped
from app.models.schema import ORG

LOG_HINT = ("bytes", ".len", "size")


def _kind_of(v: Any) -> str:
    if isinstance(v, (frozenset, set)):
        return "set"
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return "num"
    return "text"


class OracleLearner:
    """Fixed tree: root split on `route_attr` (one child per route), each
    route child split on net.src at the grp level (one child per group) when
    `groups` is given. Groups are published as model.who_groups (ip2g)."""

    def __init__(self, store: Any, key: str, t0: float, routes: Sequence[str],
                 groups: Optional[Mapping[str, Sequence[str]]] = None,
                 route_attr: str = "http.route", config: Optional[Mapping[str, Any]] = None,
                 target_filter=None, m_t: int = 8) -> None:
        self.store, self.key, self.config = store, key, dict(config or {})
        self.ptm = MP.ensure_ptree(store, key, t0)
        self.tree = self.ptm.tree(EV.KIND_TXN, t0)
        self.reg = MP.ensure_registry(store, key, config)
        self.burst = PS.BurstEvidence(65536)
        self.m_t = int(m_t)
        self.target_filter = target_filter or (lambda a: a.split(".", 1)[0] in ("body", "q", "net")
                                               and a not in ("net.src", "net.dst", "net.peer_src"))
        if groups:
            ip2g = {ip: g for g, ips in groups.items() for ip in ips}
            store.put_model(ORG, ORG, MP.WHO_GROUPS, {"ip2g": ip2g, "groups": {
                g: {"members": list(ips)} for g, ips in groups.items()}})
        if routes:
            sp = self.tree.split(self.tree.root, route_attr, 0, [[r] for r in routes], t0)
            if groups:
                for c in sp.children:
                    self.tree.split(c, "net.src", 3, [[f"grp:{g}"] for g in groups], t0)
        self.hier = MP.hierarchies(store, key, self.config, self.reg)
        self.tz = self.config.get("tz") or TB.DEFAULT_TZ
        self.route_attr = route_attr
        self.groups = dict(groups or {})
        self.dynamic = False
        self._dmemo: Dict[int, int] = {}

    def ensure_route(self, route: Any, t: float) -> None:
        """Add a child for a route value first seen after the tree was built
        (dynamic mode, used by the pack-level evaluation)."""
        tree = self.tree
        root = tree.nodes[tree.root]
        if root.split is None:
            sp = tree.split(tree.root, self.route_attr, 0, [[route]], t)
            kids = sp.children
        else:
            sp = root.split
            if route in sp.index:
                return
            nid = tree._new(tree.root, 1, root.ctx + ((self.route_attr, 0, frozenset({route}), False),), t)
            sp.groups.append(frozenset({route}))
            sp.children.append(nid)
            sp.reindex()
            kids = [nid]
        if self.groups:
            for c in kids:
                tree.split(c, "net.src", 3, [[f"grp:{g}"] for g in self.groups], t)

    def learn(self, b: Any, ctx_batch: Any = None, trust: Optional[Mapping[str, float]] = None,
              damp: Any = None) -> int:
        """P04's learning of batch b: mass w/pi x trust x damp (damp: P03's
        per-row damping, row-aligned; None = 1)."""
        store, tree = self.store, self.tree
        self.reg.observe_batch(b)
        self.reg.update_types(b.t1)
        want = MP.get_model(store, self.key, MP.PWANT) or {}
        pairs_by = ((want.get("pairs") or {}).get("by_kind") or {}).get(EV.KIND_TXN, {})
        mass = b.mass()
        n = 0
        if self.dynamic and self.route_attr in b.cols:
            for v in set(b.cols[self.route_attr].vals.tolist()):
                self.ensure_route(v, b.t1)
        for i in b.learned_rows():
            i = int(i)
            ip = b.ip_of(i)
            tr = 1.0 if trust is None else float(trust.get(ip, 1.0))
            if tr <= 0:
                continue

            def get(a: str, i: int = i) -> Any:
                if a in b.cols:
                    return b.get(a, i)
                if ctx_batch is not None and a in ctx_batch.cols:
                    return ctx_batch.get(a, i)
                return ip if a == "net.src" else EV.ABSENT
            path = tree.route(get, self.hier)
            leaf = path[-1]
            ts = float(b.ts[i])
            ev = self.burst.unit((ip, leaf), ts, tr)
            m = float(mass[i]) * tr * (1.0 if damp is None else float(damp[i]))
            mk = int(ts // 3600)
            day = self._dmemo.get(mk)
            if day is None:
                day = self._dmemo[mk] = TB.local_datetime(mk * 3600.0, self.tz).date().toordinal()
            keys = [self.hier.gen("net.src", l, ip) for l in range(5)]
            for nid in path:
                nd = tree.nodes[nid]
                nd.update_core(ts, m, ev, keys, ip, day=day)
                for a in b.cols:
                    if not self.target_filter(a):
                        continue
                    v = b.get(a, i)
                    if v is EV.ABSENT:
                        continue
                    k = _kind_of(v)
                    pol = "shape" if isinstance(v, Shaped) else "clear"
                    if a not in nd.targets:
                        if len(nd.targets) >= self.m_t:
                            continue
                        nd.target(a, k, pol, log=k == "num" and any(h in a for h in LOG_HINT))
                    nd.update_target(a, v, ts, m, ev, kind=k, day=day)
                for (X, Y) in pairs_by.get(nid, ()):
                    xa, xl = (X.rsplit("@", 1)[0], int(X.rsplit("@", 1)[1])) if "@" in X else (X, 0)
                    xv, yv = get(xa), get(Y)
                    if xv is EV.ABSENT or yv is EV.ABSENT:
                        continue
                    xg = self.hier.gen(xa, xl, xv)
                    if xg is None:
                        continue
                    nd.pair(X, Y).update(xg, yv, ts, m, ev)
            n += 1
        self.ptm.version += 1
        return n

    def node_for(self, route: str, group: Optional[str] = None) -> Optional[int]:
        tree = self.tree
        root = tree.nodes[tree.root]
        if root.split is None:
            return tree.root
        c = root.split.child_for(route)
        if group is None:
            return c
        nd = tree.nodes[c]
        if nd.split is None:
            return c
        return nd.split.child_for(f"grp:{group}")


# ---------------------------------------------------------------- generator
def org_truth(seed: int = 0):
    from app.eval import pmetrics as PMX
    from app.pipeline.orggen import OrgGenerator, build_org
    gen = OrgGenerator(build_org("O"), seed=seed, pack_name="O")
    pt = gen.ptruth()
    return gen, PMX.PTruth(pt)


def ga_login_row(pt: Any) -> Dict[str, Any]:
    for r in pt.rows:
        if r.get("activity") == "GA.oa.login" and r.get("method") == "POST" \
                and int(r["valid_from_day"]) <= 1:
            return r
    raise LookupError("GA login truth row")


def ga_login_events(row: Mapping[str, Any], gen: Any, days: Iterable[int], seed: int = 0
                    ) -> List[Tuple[int, Dict[str, Any]]]:
    """One login per member IP per workday of the given days, drawn from the
    generator's truth program (the requirement's example)."""
    from app.eval import pmetrics as PMX
    r = np.random.default_rng(seed)
    out = []
    members = list((row.get("gen") or {}).get("members") or [])
    for d in days:
        if not gen.clock.day_kind(gen.date_of(d))[0]:
            continue
        for ip in members:
            ev = PMX.holdout_events([row], r, 1, PMX.Who({"items": [ip]}, {}))
            if ev:
                e = ev[0]
                e["ts"] = gen.day_start(d) + 60.0 * float(e["minute"])
                out.append((d, e))
    return out


def batch_of(system: str, events: Sequence[Mapping[str, Any]], route: str, t0: float, t1: float,
             shape_attrs: Iterable[str] = ("body.kv.password", "body.kv.csrf")) -> Any:
    from app.engines.behavior.lib.phier import shape
    bb = EV.BatchBuilder(system)
    sh = set(shape_attrs)
    for e in events:
        attrs = {"http.route": route}
        for k, v in e.items():
            if k.startswith("body.") or k.startswith("net.bytes"):
                if k == "body.keys":
                    v = frozenset(v)
                elif k in sh:
                    v = Shaped(shape(str(v)))
                attrs[k] = v
        bb.add(float(e["ts"]), e["ip"], attrs, 1.0)
    bb.meta["policy"] = {a: "shape" for a in sh}
    return bb.build(t0, t1)


# ---------------------------------------------------------------- statement
def statement(store: Any, key: str, nid: int, route: str, kind: int = EV.KIND_TXN,
              rounded: bool = False) -> Dict[str, Any]:
    """The fitted constraints of one node in the pmetrics statement contract."""
    content: Dict[str, Any] = {}
    confs: List[float] = []
    for name in (MP.PBOUNDS, MP.PGRAMMAR):
        m = MP.get_model(store, key, name) or {}
        ent = ((m.get("nodes") or {}).get(kind) or {}).get(nid) or {}
        for a, rec in (ent.get("attrs") or {}).items():
            c = content.setdefault(a, {})
            c.update({k: rec[k] for k in ("band90", "range", "n_rng", "cover", "approx", "grammar",
                                          "c_g", "U_s", "required", "closed", "U", "coverage")
                      if k in rec})
            if "cover_hi" in rec:                   # per-statement (95 %) exceedance bound
                c["cover"] = rec["cover_hi"]
            if rounded and (rec.get("disp90") or {}).get("grid"):   # the rendered 1-2-5 band
                c["band90"] = [rec["disp90"]["lo"], rec["disp90"]["hi"]]
            if "confidence" in rec:
                confs.append(float(rec["confidence"]))
    bindings: Dict[str, Any] = {}
    pb = MP.get_model(store, key, MP.PBIND) or {}
    ent = ((pb.get("nodes") or {}).get(kind) or {}).get(nid) or {}
    for pk, rec in (ent.get("pairs") or {}).items():
        if rec.get("x") != "net.src" or rec.get("dir") != "fwd" or not rec["fd"]["holds"]:
            continue
        tab = {x: e["top"] for x, e in rec["table"].items() if e.get("bound")}
        tab.update({x: e["set"] for x, e in rec["table"].items() if e.get("set")})
        bindings[rec["y"]] = {"x": "net.src", "table": tab,
                              "LB": {x: e["LB"] for x, e in rec["table"].items() if e.get("bound")}}
        confs.append(float(rec["confidence"]))
    return {"pattern_id": f"p:{key}:{kind}:{nid}", "state": "confirmed",
            "confidence": float(min(confs)) if confs else math.nan,
            "evidence": {"route": route, "system": key, "content": content, "bindings": bindings}}
