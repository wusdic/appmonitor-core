"""Groups, views and conformity on the generator's organisation (pack O, all
systems; docs/lib3/progressive.md §6.15-§6.17, gates PG1 / PG3 / PG6 / PG10).

Traffic: orggen pack O through the real engine chain R2, R3, P00, P01, P02,
P05, P03 (conformity), P04, P06-P10, P11 (who groups), P14 (views) — the
progressive core without the B-library, so incidents are not produced here
(PG6's incident half needs W-P9's B24-B27 hooks). At the end of the snapshot
days the runner's progressive_snapshot is taken; the scorer is eval/pmetrics,
which reads only those snapshots, the pattern_violation events and the truth
the generator published:

  pg1     recall / components / precision of the rendered statements (P14's
          statement contract) at each snapshot day
  pg3     specificity: GA login who, finance approval single IP, portal level,
          P11 ARI against the static departments, DEV pool grouped
  pg10    the OA example statement (day 11 / day 21), finance single IP, the
          GA group view's negative statement about finance writes
  pg6     for each anomaly A1-A10: a pattern_violation of the expected type on
          the scenario entity within one tick of its first event; FAR:
          pattern_violations >= LOW per clean entity-day
  cost    ms per tick of P03 / P11 / P14, P03 us per event

Usage: python tests/eval/views_conformity_pack.py <seed> [days] [dt]
(writes JSON to stdout)."""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "backend"))

from app.core.engine import Context  # noqa: E402
from app.core.store import MetricStore  # noqa: E402
from app.engines.behavior.attr_registry import AttributeRegistryEngine  # noqa: E402
from app.engines.behavior.attr_select import AttributeSelectionEngine  # noqa: E402
from app.engines.behavior.binding import BindingEngine  # noqa: E402
from app.engines.behavior.conformity import ConformityEngine  # noqa: E402
from app.engines.behavior.content_bounds import ContentBoundsEngine  # noqa: E402
from app.engines.behavior.pattern_tree import PatternTreeEngine  # noqa: E402
from app.engines.behavior.payload_grammar import PayloadGrammarEngine  # noqa: E402
from app.engines.behavior.time_window import TimeWindowEngine  # noqa: E402
from app.engines.behavior.views import ViewsEngine  # noqa: E402
from app.engines.behavior.who_groups import WhoGroupsEngine  # noqa: E402
from app.engines.behavior.workflow import WorkflowEngine  # noqa: E402
from app.engines.derived.event_context import EventContextEngine  # noqa: E402
from app.engines.raw.action_token import ActionTokenEngine  # noqa: E402
from app.engines.raw.client_stack import ClientStackEngine  # noqa: E402
from app.engines.raw.event_builder import EventBuilderEngine  # noqa: E402
from app.eval import pmetrics as PM  # noqa: E402
from app.eval.runner import progressive_snapshot  # noqa: E402
from app.pipeline.orggen import OrgGenerator, build_org  # noqa: E402


class _Run:
    """The subset of eval.runner.RunResult that pmetrics reads."""

    def __init__(self) -> None:
        self.truth: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.incidents: List[Dict[str, Any]] = []
        self.psnaps: Dict[int, Dict[str, Any]] = {}
        self.series: Dict[str, Any] = {}
        self.scenario_dt = 3600.0


def _ev(e: Any) -> Dict[str, Any]:
    return {"kind": e.kind, "system": e.system, "entity": e.entity, "ts": e.ts,
            "severity": getattr(e.severity, "value", str(e.severity)), "extra": dict(e.extra or {}),
            "axes": list(e.axes or [])}


def run(seed: int = 0, days: int = 21, dt: float = 3600.0,
        snap_days: Sequence[int] = (3, 7, 11, 14, 21), save: Optional[str] = None) -> Dict[str, Any]:
    spec = build_org("O")
    gen = OrgGenerator(spec, seed=seed, pack_name="O")
    cfg = {"grain_mode": "canonical", "strict": True, "tz": spec.tz, "calendar": dict(spec.calendar)}
    cfg.update(spec.config)
    st = MetricStore()
    p03, p11, p14 = ConformityEngine(), WhoGroupsEngine(), ViewsEngine()
    engines = [ActionTokenEngine(), ClientStackEngine(), EventBuilderEngine(), EventContextEngine(),
               AttributeRegistryEngine(), AttributeSelectionEngine(), p03, PatternTreeEngine(),
               ContentBoundsEngine(), PayloadGrammarEngine(), BindingEngine(), TimeWindowEngine(),
               WorkflowEngine(), p11, p14]
    t0 = gen.day_start(1)
    per_day = int(round(86400.0 / dt))
    run_ = _Run()
    run_.scenario_dt = dt
    tim: Dict[str, float] = {}
    n_ev = 0
    wall = time.time()
    snap_days = [d for d in snap_days if d <= days]
    for k in range(days * per_day):
        a, b = t0 + k * dt, t0 + (k + 1) * dt
        obs = gen.step(a, b, aggregated=dt >= 900)
        c = Context(store=st, now=b, window_s=dt, config=cfg)
        for e in engines:
            x = time.perf_counter()
            e.safe_run(c, obs if e.layer == "raw" else None)
            tim[e.name] = tim.get(e.name, 0.0) + time.perf_counter() - x
        n_ev += int((p03.last_stats or {}).get("events") or 0)
        if (k + 1) % per_day == 0:
            d = (k + 1) // per_day
            if d in snap_days:
                snap = progressive_snapshot(st, full=True)
                snap.update({"day": d, "ts": b})
                run_.psnaps[d] = snap
            print(f"day {d} {time.time() - wall:.0f}s", file=sys.stderr, flush=True)
    run_.truth = gen.truth_rows()
    run_.events = [_ev(e) for e in st.events(limit=10 ** 7)]
    saved = {"run": run_, "ptruth": gen.ptruth(), "cfg": cfg, "seed": seed, "days": days, "dt": dt,
             "tim": tim, "n_ev": n_ev}
    if save:
        import pickle
        with open(save, "wb") as f:
            pickle.dump(saved, f)
    return score(saved)


def _safe_holdout(orig):
    """pmetrics.holdout_check compares numeric bands with held-out values that
    the truth program draws as strings for form fields (body.kv.id = '004217');
    such statements are re-checked without the numeric band of those fields
    (reported to W-P8)."""
    import copy as _copy

    def wrapped(s, rows, pt, r, n=400):
        try:
            return orig(s, rows, pt, r, n)
        except TypeError:
            s2 = _copy.copy(s)
            s2.content = {a: {k: v for k, v in c.items() if k not in ("band90", "range")}
                          if a.startswith(("body.kv.", "q.kv.")) else c for a, c in s.content.items()}
            return orig(s2, rows, pt, r, n)
    return wrapped


def score(saved: Mapping[str, Any]) -> Dict[str, Any]:
    run_ = saved["run"]
    cfg, seed, days, dt, tim, n_ev = (saved[k] for k in ("cfg", "seed", "days", "dt", "tim", "n_ev"))
    if not getattr(PM.holdout_check, "_wrapped", False):
        PM.holdout_check = _safe_holdout(PM.holdout_check)
        PM.holdout_check._wrapped = True
    pt = PM.PTruth(saved["ptruth"])
    ipc = PM.ip_classes_of(cfg)
    out: Dict[str, Any] = {"seed": seed, "days": days, "dt": dt}
    out["pg1"] = {d: {k: v for k, v in PM.pg1_snapshot(run_.psnaps[d], pt, ipc, d, seed).items()
                      if k in ("recall", "components", "precision", "n_stmt", "n_confirmed", "n_truth",
                               "mean_depth", "ece")} for d in sorted(run_.psnaps)}
    out["pg3"] = {d: PM.pg3_specificity(run_.psnaps[d], pt, ipc) for d in sorted(run_.psnaps) if d >= 7}
    out["pg10"] = PM.pg10_views(run_, pt, ipc, seed)
    sbd = {d: PM.statements(s, ipc) for d, s in run_.psnaps.items()}
    out["pg6"] = PM.pg6_anomalies(run_, pt, sbd)
    out["pg6_far"] = PM.pg6_far(run_, pt)
    pv = [e for e in run_.events if e["kind"] == "pattern_violation"]
    by: Dict[str, int] = {}
    for e in pv:
        k = f"{e['system']}|{e['extra'].get('type')}|{e['severity']}"
        by[k] = by.get(k, 0) + 1
    out["violations"] = dict(sorted(by.items()))
    # clean-entity findings per week (precision of the findings over observation time)
    far = PM.pg6_far(run_, pt)
    dirty = set()
    for r in run_.truth:
        if r.get("label") == "malicious" or str(r.get("scenario_id", "")).startswith("R"):
            dirty |= set(map(str, r.get("entities") or []))
    weeks: Dict[str, Dict[str, int]] = {}
    for e in pv:
        if e["entity"] in dirty or e["severity"] not in ("low", "medium", "high", "critical"):
            continue
        wk = f"week{min(3, (pt.day_of_ts(float(e['ts'])) - 1) // 7 + 1)}"
        weeks.setdefault(wk, {"low+": 0})["low+"] += 1
    ed_week: Dict[str, int] = {}
    for sname, per_date in (pt.who_log or {}).items():
        for iso, per in per_date.items():
            wk = f"week{min(3, (pt.day_index(iso) - 1) // 7 + 1)}"
            ed_week[wk] = ed_week.get(wk, 0) + sum(1 for ip in per if ip not in dirty)
    out["far_by_week"] = {w: round(weeks.get(w, {}).get("low+", 0) / max(ed_week.get(w, 1), 1), 4)
                          for w in sorted(ed_week)}
    out["cost"] = {"s_per_engine": {k: round(v, 1) for k, v in tim.items()},
                   "p03_us_per_event": round(tim.get("behavior.conformity", 0.0) * 1e6 / max(n_ev, 1), 1),
                   "events_scored": n_ev}
    return out


if __name__ == "__main__":                                  # pragma: no cover
    if sys.argv[1] == "--score":
        import pickle
        with open(sys.argv[2], "rb") as f:
            print(json.dumps(score(pickle.load(f)), default=str, ensure_ascii=False, indent=1))
        sys.exit(0)
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 21
    dt = float(sys.argv[3]) if len(sys.argv) > 3 else 3600.0
    save = sys.argv[4] if len(sys.argv) > 4 else None
    print(json.dumps(run(seed, days, dt, save=save), default=str, ensure_ascii=False, indent=1))
