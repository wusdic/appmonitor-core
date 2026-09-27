"""Run the lib-3 evaluation: packs x seeds in parallel processes -> eval_report.{json,html}.

Why processes: one pack-seed is ~2-4 min of single-threaded numpy/Python on
one core (eval.md gate 14 caps it at 6 min), and pack-seeds share nothing, so
a ProcessPoolExecutor gives near-linear speed-up and isolates a crashing run.
Each worker runs the pack AND scores it (metrics.score_run) so only the
compact JSON-able score crosses the process boundary, never the store.

Examples:
  .venv/bin/python scripts/evaluate.py --packs A,B,C,D,E --seeds 0,1,2,3,4 --workers 4 --out eval_out
  .venv/bin/python scripts/evaluate.py --packs A --seeds 0 --ablate likelihood,class_monitor
  .venv/bin/python scripts/evaluate.py --packs A --seeds 0,1 --feedback
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

# single-threaded BLAS / OpenMP before numpy loads (integration R15.1 / R19.3)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from app.eval.metrics import score_run  # noqa: E402
from app.eval.report import _clean, write_report  # noqa: E402
from app.eval.runner import SimulatedAnalyst, run_pack  # noqa: E402

DEFAULT_PACKS = ["A", "B", "C", "D", "E"]


def _job(pack: str, seed: int, opts: Dict[str, Any]) -> Dict[str, Any]:
    """Worker: run + score one (pack, seed[, ablation / feedback])."""
    t0 = time.perf_counter()
    try:
        analyst = SimulatedAnalyst(seed=seed) if opts.get("feedback") else None
        res = run_pack(pack, seed, strict=True, time_budget_s=opts.get("time_budget_s"),
                       disable_engines=opts.get("disable", ()), analyst=analyst)
        sc = score_run(res)
    except Exception as exc:  # a broken run must not take the others down
        return {"pack": pack, "seed": seed, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:],
                "disabled_engines": list(opts.get("disable", ()))}
    sc["variant"] = opts.get("variant", "full")
    sc["job_s"] = time.perf_counter() - t0
    return _clean(sc)


def _pack_names(arg: Optional[str]) -> List[str]:
    if arg:
        return [p.strip() for p in arg.split(",") if p.strip()]
    try:
        from app.eval import packs
        names = getattr(packs, "PACKS", None)
        if isinstance(names, dict) and names:
            return sorted(n for n in names if str(n).lower() != "smoke")
    except Exception:
        pass
    return list(DEFAULT_PACKS)


def _seeds(arg: str) -> List[int]:
    arg = arg.strip()
    if ":" in arg:
        a, b = arg.split(":", 1)
        return list(range(int(a), int(b)))
    return [int(x) for x in arg.split(",") if x.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--packs", default=None, help="comma list (default: A,B,C,D,E)")
    ap.add_argument("--seeds", default="0,1,2,3,4", help="comma list or a:b range")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="eval_out", help="output directory")
    ap.add_argument("--time-budget", type=float, default=None,
                    help="per pack-seed wall budget in s (aborts the run)")
    ap.add_argument("--ablate", default="", help="comma list of engine names to disable one "
                                                 "at a time (gate 13)")
    ap.add_argument("--feedback", action="store_true",
                    help="also run each pack-seed with the simulated analyst (gate 12)")
    ap.add_argument("--smoke", action="store_true",
                    help="also run the 'smoke' pack (gate 14 smoke budget)")
    args = ap.parse_args(argv)

    packs, seeds = _pack_names(args.packs), _seeds(args.seeds)
    jobs: List[Tuple[str, int, Dict[str, Any]]] = [(p, s, {"variant": "full",
                                                          "time_budget_s": args.time_budget})
                                                   for p in packs for s in seeds]
    for eng in [e.strip() for e in args.ablate.split(",") if e.strip()]:
        jobs += [(p, s, {"variant": f"ablate:{eng}", "disable": [eng],
                         "time_budget_s": args.time_budget}) for p in packs for s in seeds]
    if args.feedback:
        jobs += [(p, s, {"variant": "feedback", "feedback": True,
                         "time_budget_s": args.time_budget}) for p in packs for s in seeds]
    if args.smoke:
        jobs.append(("smoke", 0, {"variant": "smoke"}))

    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)
    results: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(_job, p, s, o): (p, s, o["variant"]) for p, s, o in jobs}
        for fut in as_completed(futs):
            p, s, v = futs[fut]
            sc = fut.result()
            results.append(sc)
            tag = v.replace(":", "_")
            with open(os.path.join(args.out, "runs", f"{p}_{s}_{tag}.json"), "w",
                      encoding="utf-8") as f:
                json.dump(sc, f, indent=1, ensure_ascii=False)
            status = "ERROR " + sc["error"] if "error" in sc else \
                f"{sc.get('job_s', 0):.1f}s aborted={sc.get('aborted')}"
            print(f"[{time.perf_counter() - t0:7.1f}s] {p}/{s} {v}: {status}", flush=True)

    ok = [r for r in results if "error" not in r]
    full = [r for r in ok if r.get("variant") == "full"]
    ablation: Dict[str, List[Dict[str, Any]]] = {}
    for r in ok:
        if str(r.get("variant", "")).startswith("ablate:"):
            ablation.setdefault(r["variant"].split(":", 1)[1], []).append(r)
    feedback = [r for r in ok if r.get("variant") == "feedback"]
    smoke = next((r for r in ok if r.get("variant") == "smoke"), None)
    rep = write_report(full, args.out, feedback=feedback or None, ablation=ablation or None,
                       smoke=smoke, meta={"packs": packs, "seeds": seeds,
                                          "errors": [{k: r[k] for k in ("pack", "seed", "error")}
                                                     for r in results if "error" in r],
                                          "wall_s": time.perf_counter() - t0})
    s = rep["summary"]
    print(f"gates: {s['gates_pass']} pass, {s['gates_fail']} fail, {s['gates_na']} n/a -> "
          f"{os.path.join(args.out, 'eval_report.html')}")
    return 0 if s["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
