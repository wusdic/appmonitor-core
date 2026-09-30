"""Content fitters P06-P08 on the real progressive lattice (integration of W-P4
with P00-P05; docs/lib3/progressive.md §6.10-§6.12, §6.22).

The OA system of pack O (orggen) runs through R2, R3, P00, P01, P02, P05, P04
and then P06 content_bounds, P07 payload_grammar, P08 binding, all through the
store only (P08 asks P04 for pair sketches through model.pwant, P04 keeps them,
P08 fits them). For each truth pattern of the requirement's example (综合部's OA
login, and 财务部's OA login as a second department), the harness keeps the
latest real event of that pattern and, at each checkpoint day, routes it
through the learned tree and reads the fitted constraints on its path (the
deepest node that has one):

  band_ok      body.len band90 within +-20 % of the truth band (1-2 KB)
  range_ok     body.len hard range within +-25 % of the truth range (0.5-3 KB), or
               no hard range published yet (n_rng < 30)
  grammar_ok   the username grammar accepts every truth username (jack, rose, mike)
  closed       the closed username set, when published, equals the truth set
  bindings     truth IPs whose fitted binding (FD holds) is the truth username
  cover        the range's per-statement exceedance bound (falls with time)
  lb           the smallest binding lower bound LB_x (rises with time)

Nothing here is read by an engine; the truth is the persona program itself."""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Sequence

from app.core.engine import Context
from app.core.store import MetricStore
from app.engines.behavior.attr_registry import AttributeRegistryEngine
from app.engines.behavior.attr_select import AttributeSelectionEngine
from app.engines.behavior.binding import BindingEngine
from app.engines.behavior.content_bounds import ContentBoundsEngine
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.pattern_tree import PatternTreeEngine
from app.engines.behavior.payload_grammar import PayloadGrammarEngine
from app.engines.derived.event_context import EventContextEngine
from app.engines.raw.action_token import ActionTokenEngine
from app.engines.raw.client_stack import ClientStackEngine
from app.engines.raw.event_builder import EventBuilderEngine
from app.pipeline.orggen import OrgGenerator, build_org

SYSTEM = "oa"
ACTIVITIES = ("GA.oa.login", "FIN.oa.login")
Y = "body.kv.username"


def _truth_rows(gen: OrgGenerator) -> Dict[str, Dict[str, Any]]:
    rows = gen._ptruth_rows if hasattr(gen, "_ptruth_rows") else gen.pattern_truth
    out = {}
    for r in rows:
        if r.get("system") == SYSTEM and r.get("activity") in ACTIVITIES and r.get("method") == "POST" \
                and (r.get("content") or {}).get("body.len"):
            out[r["tid"]] = r
    return out


def _matches(row: Dict[str, Any], route: Any, ip: str, day: int) -> bool:
    if not isinstance(route, str):
        return False
    lo, hi = int(row.get("valid_from_day", 0)), int(row.get("valid_to_day", 10 ** 6))
    if not lo <= day < hi:
        return False
    return (route.startswith(row["method"] + " ") and route.endswith(" " + row["route"])
            and ip in set(row["who"]["value"]))


def _rel(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * abs(b)


def _node_rec(model: Any, path: Sequence[int], attr: str) -> Optional[Dict[str, Any]]:
    for nid in reversed(list(path)):
        rec = MP_lookup(model, nid, attr)
        if rec is not None:
            return rec
    return None


def MP_lookup(model: Any, nid: int, attr: str) -> Optional[Dict[str, Any]]:
    ent = (((model or {}).get("nodes") or {}).get(EV.KIND_TXN) or {}).get(int(nid))
    return ((ent or {}).get("attrs") or {}).get(attr)


def _binding(pb: Any, path: Sequence[int]) -> Optional[Dict[str, Any]]:
    for nid in reversed(list(path)):
        ent = (((pb or {}).get("nodes") or {}).get(EV.KIND_TXN) or {}).get(int(nid)) or {}
        for rec in (ent.get("pairs") or {}).values():
            if rec.get("x") == "net.src" and rec.get("y") == Y and rec.get("dir") == "fwd" \
                    and rec["fd"]["holds"]:
                return rec
    return None


def measure(st: MetricStore, tree: Any, hier: Any, samples: Dict[str, Dict[str, Any]],
            truth: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    pbd = MP.get_model(st, SYSTEM, MP.PBOUNDS)
    pgr = MP.get_model(st, SYSTEM, MP.PGRAMMAR)
    pbi = MP.get_model(st, SYSTEM, MP.PBIND)
    out: Dict[str, Dict[str, Any]] = {}
    for tid, ev in samples.items():
        row = truth[tid]
        path = tree.route(lambda a, ev=ev: ev.get(a, EV.ABSENT), hier)
        tc = row["content"]
        res: Dict[str, Any] = {"depth": tree.nodes[path[-1]].depth}
        b = _node_rec(pbd, path, "body.len")
        if b is not None:
            res["band_ok"] = _rel(b["band90"][0], tc["body.len"]["band90"][0], 0.2) and \
                _rel(b["band90"][1], tc["body.len"]["band90"][1], 0.2)
            rg = b.get("range")
            res["range_ok"] = (rg is None or not b.get("hard")) or (
                _rel(rg[0], tc["body.len"]["range"][0], 0.25) and _rel(rg[1], tc["body.len"]["range"][1], 0.25))
            res["cover"] = b.get("cover_hi")
            res["band_cov"] = b.get("coverage")
        g = _node_rec(pgr, path, Y)
        truth_vals = [str(v) for v in (tc.get(Y) or {}).get("closed_values") or []]
        if g is not None and g.get("grammar"):
            rx = re.compile(g["grammar"])
            res["grammar"] = g["grammar"]
            res["grammar_ok"] = all(rx.fullmatch(v) for v in truth_vals)
            res["closed"] = g.get("closed")
            res["closed_ok"] = g.get("closed") is None or sorted(g["closed"]) == sorted(truth_vals)
            res["U_s"] = g.get("U_s")
        rec = _binding(pbi, path)
        want = {ip: str(v) for ip, v in ((row.get("bindings") or {}).get(Y) or {}).items()}
        res["bindings"] = 0
        res["bind_total"] = len(want)
        if rec is not None:
            tab = rec["table"]
            res["bindings"] = sum(1 for ip, v in want.items()
                                  if (tab.get(ip) or {}).get("bound") and str(tab[ip]["top"]) == v)
            lbs = [tab[ip]["LB"] for ip in want if (tab.get(ip) or {}).get("bound")]
            res["lb"] = min(lbs) if lbs else None
        out[tid] = res
    return out


def run(days: Sequence[int] = (3, 7, 11), dt: float = 3600.0, seed: int = 0,
        timings: Optional[Dict[str, float]] = None) -> Dict[int, Dict[str, Dict[str, Any]]]:
    spec = build_org("O")
    gen = OrgGenerator(spec, seed=seed, pack_name="O")
    truth = _truth_rows(gen)
    cfg = {"grain_mode": "canonical", "strict": True, "tz": spec.tz, "calendar": dict(spec.calendar),
           "progressive": {"enabled": True}}
    st = MetricStore()
    engines = [ActionTokenEngine(), ClientStackEngine(), EventBuilderEngine(), EventContextEngine(),
               AttributeRegistryEngine(), AttributeSelectionEngine(), PatternTreeEngine(),
               ContentBoundsEngine(), PayloadGrammarEngine(), BindingEngine()]
    t0 = gen.day_start(1)
    per_day = int(round(86400.0 / dt))
    samples: Dict[str, Dict[str, Any]] = {}
    out: Dict[int, Dict[str, Dict[str, Any]]] = {}
    tim = timings if timings is not None else {}
    b = t0
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
            out[d] = measure(st, tree, hier, {t: s for t, s in samples.items()
                                               if int(truth[t]["valid_from_day"]) <= d
                                               < int(truth[t]["valid_to_day"])}, truth)
    return out


if __name__ == "__main__":                                  # pragma: no cover
    import json
    import sys
    tt: Dict[str, float] = {}
    res = run(days=tuple(int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "3,7,11").split(",")),
              seed=int(sys.argv[2]) if len(sys.argv) > 2 else 0, timings=tt)
    for d, r in sorted(res.items()):
        print(d, json.dumps(r, default=str, ensure_ascii=False))
    print({k: round(v, 1) for k, v in tt.items()})
