"""Convergence of the temporal engines on the generator's own example
(pack O's OA system; docs/lib3/progressive.md §6.13, §6.14, §6.22; requirement
S5 "早上9点到9点21分 … 登录页面", S10 "后续 … 访问业务审批页面 … 下午5点提交报告",
S3 "用的时间越长越精准").

Traffic: orggen pack O, OA only, through R2, R3, P00 and P01 (the real raw /
derived engines), then
  P09 on an ORACLE pattern tree (root -> route -> department; counted like P04
      counts when summaries, tests/engines/temporal_sim.py) — so the measure
      is P09's, not the lattice's;
  P10 (real engine) on the evt.batch batches.
Optionally (`tree='real'`) P02 / P05 / P04 build the tree and P09 runs on it.

Measures at each checkpoint day, against the truth the generator publishes
(nothing here is read by an engine):
  windows   for every truth pattern of OA with a workday window: IoU and length
            precision |learned ∩ truth| / |learned| of the learned workday windows
            of its (route, department) node vs the truth windows;
  workflow  kept edges (dep >= 0.8): strict precision (edges listed in a truth
            row's workflow), program precision (both actions belong to one truth
            activity of one department: a step of the persona program), and
            recall of the truth workflow edges (dep >= 0.8, delay band overlapping
            the truth band, the pmetrics rule)."""
from __future__ import annotations

import ipaddress
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "engines"))

from app.core.engine import Context  # noqa: E402
from app.core.store import MetricStore  # noqa: E402
from app.engines.behavior.lib import m_ptree as MP  # noqa: E402
from app.engines.behavior.lib import pevent as EV  # noqa: E402
from app.engines.behavior.lib import pwindows as PW  # noqa: E402
from app.engines.behavior.time_window import TimeWindowEngine  # noqa: E402
from app.engines.behavior.workflow import WorkflowEngine  # noqa: E402
from app.engines.derived.event_context import EventContextEngine  # noqa: E402
from app.engines.raw.action_token import ActionTokenEngine  # noqa: E402
from app.engines.raw.client_stack import ClientStackEngine  # noqa: E402
from app.engines.raw.event_builder import EventBuilderEngine  # noqa: E402
from app.eval.pmetrics import norm_route, window_iou  # noqa: E402
from app.pipeline.orggen import OrgGenerator, build_org  # noqa: E402

SYSTEM = "oa"


def _rkey(route: Any) -> Optional[str]:
    if not isinstance(route, str):
        return None
    m, r = norm_route(route)
    return f"{m} {r}"


def _group_map(spec: Any):
    fixed: Dict[str, str] = {}
    nets: List[Tuple[Any, str]] = []
    for d in spec.departments:
        for ip in d.ips:
            fixed[ip] = d.code
        for _, (_, new) in (d.readdress or {}).items():
            fixed[new] = d.code
        if d.pool:
            nets.append((ipaddress.ip_network(d.pool[0], strict=False), d.code))

    def group_of(ip: str) -> str:
        g = fixed.get(ip)
        if g is not None:
            return g
        try:
            a = ipaddress.ip_address(ip)
        except ValueError:
            return "?"
        for n, c in nets:
            if a in n:
                return c
        return "?"
    return group_of


def _truth(gen: OrgGenerator) -> List[Dict[str, Any]]:
    rows = gen._ptruth_rows if hasattr(gen, "_ptruth_rows") else gen.pattern_truth
    return [r for r in rows if r.get("system") == SYSTEM and r.get("kind") == "txn"
            and r.get("method") and r.get("route")]


def _union_len(iv):
    from app.eval.pmetrics import _union_len as U
    return U([(float(a), float(b)) for a, b in iv])


def _inter_len(a, b):
    out = []
    for x0, x1 in a:
        for y0, y1 in b:
            lo, hi = max(x0, y0), min(x1, y1)
            if hi > lo:
                out.append((lo, hi))
    return _union_len(out)


class _Oracle:
    """root -> route (normalised) -> department, counted like P04 counts."""

    def __init__(self, store: Any, t0: float, groups_by_route: Dict[str, List[str]]) -> None:
        from app.engines.behavior.lib import phier as PH
        from app.engines.behavior.lib import pnode as PN
        from app.engines.behavior.lib import ptree as PT
        self.PN = PN
        self.store = store
        self.m = PT.PTreeModel(SYSTEM)
        self.tree = self.m.tree(EV.KIND_TXN, t0)
        self.hier = PH.Hierarchies(registry={})
        routes = sorted(groups_by_route)
        sp = self.tree.split(self.tree.root, "rk", 0, [[rk] for rk in routes], t0)
        self.route_node = dict(zip(routes, sp.children))
        self.node: Dict[Tuple[str, str], int] = {}
        for rk in routes:
            gs = groups_by_route[rk]
            sp2 = self.tree.split(self.route_node[rk], "org.grp", 0, [[g] for g in gs], t0)
            for g, c in zip(gs, sp2.children):
                self.node[(rk, g)] = c
        store.put_model(SYSTEM, "__system__", MP.PTREE, self.m, version=1, ts=t0)

    def learn(self, ts: float, ip: str, rk: str, grp: str, daytype: int, minute: float, day: int) -> None:
        get = {"rk": rk, "org.grp": grp}.get
        path = self.tree.route(lambda a: get(a, EV.ABSENT), self.hier)
        keys = [self.hier.gen("net.src", l, ip) for l in range(self.PN.WHO_LEVELS)]
        for nid in path:
            self.tree.nodes[nid].update_core(ts, 1.0, 1.0, keys, ip, daytype, minute, day)

    def apply_wants(self) -> None:
        want = MP.get_model(self.store, SYSTEM, MP.PWANT) or {}
        mr = ((want.get("p09") or {}).get("minute_reservoir") or {})
        for nid in mr.get(EV.KIND_TXN, []) or []:
            nd = self.tree.nodes.get(int(nid))
            if nd is not None and nd.when.res is None:
                nd.when.want_minutes(True, seed=int(nid))


def measure_windows(store: Any, oracle: _Oracle, truth: List[Dict[str, Any]], day: int) -> Dict[str, Any]:
    pw = MP.get_model(store, SYSTEM, MP.PWIN)
    ious, precs, found = [], [], 0
    per = {}
    for r in truth:
        if not (int(r["valid_from_day"]) <= day < int(r["valid_to_day"])):
            continue
        tw = (r.get("windows") or {}).get("workday") or []
        if not tw or tw == [[0, 1440]]:
            continue
        rk = f"{norm_route(r['method'], r['route'])[0]} {norm_route(r['method'], r['route'])[1]}"
        nid = oracle.node.get((rk, r["group"]))
        ent = PW.lookup(pw, EV.KIND_TXN, nid) if nid is not None else None
        lw = ((ent or {}).get("when") or {}).get("workday") or []
        if not lw:
            ious.append(0.0)
            per[r["tid"]] = None
            continue
        found += 1
        iou = window_iou(tw, lw) or 0.0
        inter = _inter_len([tuple(x) for x in tw], [tuple(x) for x in lw])
        ious.append(iou)
        precs.append(inter / max(_union_len([tuple(x) for x in lw]), 1e-9))
        per[r["tid"]] = (round(iou, 3), lw)
    return {"n": len(ious), "found": found, "iou": float(np.mean(ious)) if ious else None,
            "precision": float(np.mean(precs)) if precs else None,
            "iou_ge_0.7": float(np.mean([x >= 0.7 for x in ious])) if ious else None, "per": per}


def measure_windows_real(store: Any, samples: Dict[str, Dict[str, Any]], truth: List[Dict[str, Any]],
                         day: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Real P04 tree: route the latest real event of each truth pattern and
    take the deepest node on its path that has fitted workday windows."""
    ptm = MP.get_ptree(store, SYSTEM)
    tree = ptm.kinds.get(EV.KIND_TXN) if ptm is not None else None
    pw = MP.get_model(store, SYSTEM, MP.PWIN)
    hier = MP.hierarchies(store, SYSTEM, cfg)
    ious, precs, depth = [], [], []
    per = {}
    for r in truth:
        if not (int(r["valid_from_day"]) <= day < int(r["valid_to_day"])):
            continue
        tw = (r.get("windows") or {}).get("workday") or []
        ev = samples.get(r["tid"])
        if not tw or tw == [[0, 1440]] or ev is None or tree is None:
            continue
        path = tree.route(lambda a, ev=ev: ev.get(a, EV.ABSENT), hier)
        lw, dd = [], None
        for nid in reversed(path):
            ent = PW.lookup(pw, EV.KIND_TXN, nid)
            w = ((ent or {}).get("when") or {}).get("workday") or []
            if w:
                lw, dd = w, tree.nodes[nid].depth
                break
        iou = (window_iou(tw, lw) or 0.0) if lw else 0.0
        ious.append(iou)
        if lw:
            precs.append(_inter_len([tuple(x) for x in tw], [tuple(x) for x in lw])
                         / max(_union_len([tuple(x) for x in lw]), 1e-9))
            depth.append(dd)
        per[r["tid"]] = (round(iou, 3), lw, dd)
    return {"n": len(ious), "iou": float(np.mean(ious)) if ious else None,
            "precision": float(np.mean(precs)) if precs else None,
            "depth": float(np.mean(depth)) if depth else None,
            "nodes": len(tree) if tree is not None else 0, "per": per}


def measure_workflow(store: Any, truth: List[Dict[str, Any]], day: int) -> Dict[str, Any]:
    by_tid = {r["tid"]: r for r in truth}
    valid = [r for r in truth if int(r["valid_from_day"]) <= day < int(r["valid_to_day"])]
    truth_edges: Set[Tuple[str, str]] = set()
    bands: Dict[Tuple[str, str], List[float]] = {}
    for r in valid:
        for frm, _, band in r.get("workflow") or []:
            fr = by_tid.get(frm)
            if fr is None:
                continue
            e = (_rkey(f"{fr['method']} {fr['route']}"), _rkey(f"{r['method']} {r['route']}"))
            truth_edges.add(e)
            bands[e] = list(band)
    program: Set[Tuple[str, str]] = set()
    acts: Dict[Tuple[str, str], Set[str]] = {}
    for r in valid:
        acts.setdefault((r["group"], r["activity"]), set()).add(_rkey(f"{r['method']} {r['route']}"))
    for rs in acts.values():
        for a in rs:
            for b in rs:
                if a != b:
                    program.add((a, b))
    pf = MP.get_model(store, SYSTEM, MP.PFLOW)
    kept = []
    for e in ((pf or {}).get("scopes") or {}).get("*", {}).get("edges", []):
        if e["dep"] >= 0.8:
            kept.append(((_rkey(e["from"]), _rkey(e["to"])), e))
    strict = [k in truth_edges for k, _ in kept]
    prog = [k in truth_edges or k in program for k, _ in kept]
    rec = []
    for te in truth_edges:
        hit = [e for k, e in kept if k == te and e.get("band")
               and e["band"][0] <= bands[te][1] and e["band"][1] >= bands[te][0]]
        rec.append(bool(hit))
    return {"kept": len(kept), "truth_edges": len(truth_edges),
            "strict_precision": float(np.mean(strict)) if strict else None,
            "program_precision": float(np.mean(prog)) if prog else None,
            "recall": float(np.mean(rec)) if rec else None,
            "correct": int(sum(prog)),
            "edges": sorted(f"{k[0]} -> {k[1]}" for k, _ in kept)}


def run(days: Sequence[int] = (2, 5, 10, 15), dt: float = 3600.0, seed: int = 0, tree: str = "oracle",
        timings: Optional[Dict[str, float]] = None, session_mode: Optional[str] = None
        ) -> Dict[int, Dict[str, Any]]:
    spec = build_org("O")
    gen = OrgGenerator(spec, seed=seed, pack_name="O")
    truth = _truth(gen)
    group_of = _group_map(spec)
    cfg = {"grain_mode": "canonical", "strict": True, "tz": spec.tz, "calendar": dict(spec.calendar),
           "progressive": {"enabled": True}}
    st = MetricStore()
    raw = [ActionTokenEngine(), ClientStackEngine(), EventBuilderEngine(), EventContextEngine()]
    lattice: List[Any] = []
    if tree == "real":
        from app.engines.behavior.attr_registry import AttributeRegistryEngine
        from app.engines.behavior.attr_select import AttributeSelectionEngine
        from app.engines.behavior.pattern_tree import PatternTreeEngine
        lattice = [AttributeRegistryEngine(), AttributeSelectionEngine(), PatternTreeEngine()]
    p09 = TimeWindowEngine()
    p10 = WorkflowEngine(session_mode=session_mode) if session_mode else WorkflowEngine()
    t0 = gen.day_start(1)
    off = PW.tz_offset(cfg, t0)
    groups_by_route: Dict[str, List[str]] = {}
    for r in truth:
        rk = _rkey(f"{r['method']} {r['route']}")
        if r["group"] not in groups_by_route.setdefault(rk, []):
            groups_by_route[rk].append(r["group"])
    oracle = _Oracle(st, t0, groups_by_route) if tree == "oracle" else None
    per_day = int(round(86400.0 / dt))
    tim = timings if timings is not None else {}
    samples: Dict[str, Dict[str, Any]] = {}
    out: Dict[int, Dict[str, Any]] = {}
    for k in range(max(days) * per_day):
        a, b = t0 + k * dt, t0 + (k + 1) * dt
        obs = [o for o in gen.step(a, b, aggregated=dt >= 900) if o.system == SYSTEM]
        c = Context(store=st, now=b, window_s=dt, config=cfg)
        for e in raw:
            x = time.perf_counter()
            e.safe_run(c, obs if e.layer == "raw" else None)
            tim[e.name] = tim.get(e.name, 0.0) + time.perf_counter() - x
        if oracle is not None:
            bt = st.batch_at(SYSTEM, EV.EVT_BATCH, b)
            cb = st.batch_at(SYSTEM, EV.EVT_CTX, b)
            if bt is not None and cb is not None and cb.n == bt.n:
                for i in range(bt.n):
                    rk = _rkey(bt.get("http.route", i))
                    if rk is None or rk not in groups_by_route:
                        continue
                    ip = bt.ip_of(i)
                    dtp = 0 if cb.get("ctx.daytype", i) == "workday" else 1
                    ts = float(bt.ts[i])
                    oracle.learn(ts, ip, rk, group_of(ip), dtp, float(cb.get("ctx.tod_min", i)),
                                 int((ts + off) // 86400.0))
            oracle.apply_wants()
        if tree == "real":
            bt = st.batch_at(SYSTEM, EV.EVT_BATCH, b)
            cb = st.batch_at(SYSTEM, EV.EVT_CTX, b)
            if bt is not None:
                day_now = gen.day_of(b - 1.0)
                for i in range(bt.n):
                    rk = _rkey(bt.get("http.route", i))
                    ip = bt.ip_of(i)
                    for r in truth:
                        if rk == _rkey(f"{r['method']} {r['route']}") and group_of(ip) == r["group"] \
                                and int(r["valid_from_day"]) <= day_now < int(r["valid_to_day"]):
                            ev = bt.row(i)
                            if cb is not None and cb.n == bt.n:
                                for nm, v in cb.row(i).items():
                                    ev.setdefault(nm, v)
                            ev.setdefault("net.src", ip)
                            samples[r["tid"]] = ev
        for e in lattice + [p09, p10]:
            x = time.perf_counter()
            e.safe_run(c, None)
            tim[e.name] = tim.get(e.name, 0.0) + time.perf_counter() - x
        d = (k + 1) // per_day
        if (k + 1) % per_day == 0 and d in days:
            res: Dict[str, Any] = {"workflow": measure_workflow(st, truth, d)}
            if oracle is not None:
                res["windows"] = measure_windows(st, oracle, truth, d)
            else:
                res["windows"] = measure_windows_real(st, samples, truth, d, cfg)
            out[d] = res
    return out


if __name__ == "__main__":                                  # pragma: no cover
    import json
    tt: Dict[str, float] = {}
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = sys.argv[2] if len(sys.argv) > 2 else "oracle"
    res = run(days=(2, 5, 10, 15, 21) if mode == "oracle" else (3, 7, 14), seed=seed, timings=tt, tree=mode)
    for d, r in sorted(res.items()):
        w, f = r.get("windows", {}), r["workflow"]
        print(d, "windows", {k: w.get(k) for k in ("n", "found", "iou", "precision", "iou_ge_0.7",
                                                     "depth", "nodes")},
              "workflow", {k: f[k] for k in ("kept", "truth_edges", "strict_precision",
                                                "program_precision", "recall", "correct")})
    print(json.dumps({k: round(v, 1) for k, v in tt.items()}))
