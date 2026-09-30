"""Convergence of the progressive lattice on the generator's example
(docs/lib3/progressive.md §6.22; requirement S2/S3 "用的时间越长越精准").

The OA system of pack O (orggen) runs through R2, R3, P00, P01, P02, P05 and
P04. For every truth pattern of the OA system whose who-set is a list of IPs
(综合部 login 09:00-09:21, the approver's approvals, the reporters' reports,
财务部's OA login, ...), the harness keeps the latest real event of that
pattern and, at each checkpoint day, routes it through the learned tree:

  covering node   = the deepest CONFIDENT node (confirmed / stable / evolving /
                    stale) on the event's path, else the root: the pattern P03
                    scores the event against (§6.16.1); candidates are not yet
                    learned patterns
  who precision   = |heavy set (95 % of mass, /32) of the covering node  ∩  truth IPs|
                    / |heavy set|          (1.0 = the pattern is isolated to its IPs)
  route specific  = the covering node's context fixes the pattern's route / path to
                    a named group of <= 3 routes (1) or not (0)
  specificity     = depth of the covering node
Every truth pattern that has a sample is scored at every checkpoint.

Nothing here is read by an engine; the truth is the persona program itself."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from app.core.engine import Context
from app.core.store import MetricStore
from app.engines.behavior.attr_registry import AttributeRegistryEngine
from app.engines.behavior.attr_select import AttributeSelectionEngine
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.pattern_tree import PatternTreeEngine
from app.engines.derived.event_context import EventContextEngine
from app.engines.raw.action_token import ActionTokenEngine
from app.engines.raw.client_stack import ClientStackEngine
from app.engines.raw.event_builder import EventBuilderEngine
from app.pipeline.orggen import OrgGenerator, build_org

SYSTEM = "oa"


def _truth_rows(gen: OrgGenerator) -> List[Dict[str, Any]]:
    rows = gen._ptruth_rows if hasattr(gen, "_ptruth_rows") else gen.pattern_truth
    out = []
    for r in rows:
        who = r.get("who") or {}
        if r.get("system") != SYSTEM or who.get("level") != "ip" or r.get("kind") != "txn":
            continue
        if not r.get("method") or not r.get("route"):
            continue
        out.append(r)
    return out


def _matches(row: Dict[str, Any], route: Any, ip: str, day: int) -> bool:
    if not isinstance(route, str):
        return False
    lo, hi = int(row.get("valid_from_day", 0)), int(row.get("valid_to_day", 10 ** 6))
    if not lo <= day < hi:
        return False
    return (route.startswith(row["method"] + " ") and route.endswith(" " + row["route"])
            and ip in set(row["who"]["value"]))


def measure(tree: Any, hier: Any, samples: Dict[str, Dict[str, Any]], truth: Dict[str, Dict[str, Any]],
            t: float) -> Dict[str, float]:
    prec, pur, depth = [], [], []
    for tid, ev in samples.items():
        row = truth[tid]
        path = tree.route(lambda a, ev=ev: ev.get(a, EV.ABSENT), hier)
        conf = [n for n in path if tree.nodes[n].state in PN.CONFIDENT_STATES]
        leaf = tree.nodes[conf[-1] if conf else path[0]]
        heavy, _ = leaf.who.heavy_set(0, t)
        if not heavy:
            prec.append(0.0)
            pur.append(0.0)
            depth.append(leaf.depth)
            continue
        ips = set(row["who"]["value"])
        prec.append(len([h for h in heavy if h in ips]) / len(heavy))
        # the pattern's action is identified: the leaf's context fixes its route
        # (or path) to a named group that contains it
        hit = 0.0
        for a, l, vals, neg in leaf.ctx:
            if a in ("http.route", "http.path") and not neg:
                v = hier.gen(a, l, ev.get(a, EV.ABSENT))
                if v in vals and len(vals) <= 3:
                    hit = 1.0
        pur.append(hit)
        depth.append(leaf.depth)
    return {"who_precision": float(np.mean(prec)) if prec else 0.0,
            "route_specific": float(np.mean(pur)) if pur else 0.0,
            "depth": float(np.mean(depth)) if depth else 0.0, "patterns": len(prec)}


def run(days: Sequence[int] = (1, 3, 6, 10), dt: float = 3600.0, seed: int = 0,
        timings: Optional[Dict[str, float]] = None, keep: Optional[Dict[str, Any]] = None,
        p05_period_s: float = 3600.0) -> Dict[int, Dict[str, float]]:
    spec = build_org("O")
    gen = OrgGenerator(spec, seed=seed, pack_name="O")
    truth = {r["tid"]: r for r in _truth_rows(gen)}
    cfg = {"grain_mode": "canonical", "strict": True, "tz": spec.tz, "calendar": dict(spec.calendar),
           "progressive": {"enabled": True}}
    st = MetricStore()
    engines = [ActionTokenEngine(), ClientStackEngine(), EventBuilderEngine(), EventContextEngine(),
               AttributeRegistryEngine(), AttributeSelectionEngine(eval_period_s=p05_period_s),
               PatternTreeEngine()]
    t0 = gen.day_start(1)
    per_day = int(round(86400.0 / dt))
    samples: Dict[str, Dict[str, Any]] = {}
    out: Dict[int, Dict[str, float]] = {}
    tim = timings if timings is not None else {}
    for k in range(max(days) * per_day):
        a, b = t0 + k * dt, t0 + (k + 1) * dt
        obs = [o for o in gen.step(a, b, aggregated=dt >= 900) if o.system == SYSTEM]
        c = Context(store=st, now=b, window_s=dt, config=cfg)
        for e in engines:
            x = time.perf_counter()
            e.safe_run(c, obs if e.layer == "raw" else None)
            tim[e.name] = tim.get(e.name, 0.0) + time.perf_counter() - x
        bt = st.batch_at(SYSTEM, EV.EVT_BATCH, b)
        cb = st.batch_at(SYSTEM, EV.EVT_CTX, b)
        if bt is not None:
            day = gen.day_of(b - 1.0)
            for i in range(bt.n):
                route, ip = bt.get("http.route", i), bt.ip_of(i)
                for tid, row in truth.items():
                    if _matches(row, route, ip, day):
                        ev = bt.row(i)
                        if cb is not None and cb.n == bt.n:
                            for nm, v in cb.row(i).items():
                                ev.setdefault(nm, v)
                        ev.setdefault("net.src", ip)
                        samples[tid] = ev
        if (k + 1) % per_day == 0 and (k + 1) // per_day in days:
            d = (k + 1) // per_day
            m = MP.get_ptree(st, SYSTEM)
            tree = m.kinds.get(EV.KIND_TXN)
            hier = MP.hierarchies(st, SYSTEM, cfg)
            out[d] = dict(measure(tree, hier, samples, truth, b), nodes=len(tree))
    if keep is not None:
        keep.update(store=st, samples=samples, truth=truth, cfg=cfg, now=b)
    return out


if __name__ == "__main__":                                  # pragma: no cover
    tt: Dict[str, float] = {}
    res = run(timings=tt)
    for d, r in sorted(res.items()):
        print(d, r)
    print({k: round(v, 1) for k, v in tt.items()})
