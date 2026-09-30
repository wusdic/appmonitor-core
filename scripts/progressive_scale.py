"""PG4 scaling experiment of the progressive core (docs/lib3/progressive.md §12 PG4).

Each point runs one pack in the 'progressive_only' registry (R2, R3, P00-P15)
and measures P-core memory (deep size of the P models in the store) and CPU
per event (eval/pscale.run_point). Axes:

  ips      pack O with portal_n in --ips (default 500,5000,20000), no synthetic attributes
  attrs    pack O with portal_n 500 and --attrs synthetic attributes (default 0,60,300)
  servers  pack O-servers with --servers systems (default 20,100,300) in 12 families

--days shortens every pack (the full grid at 7 days is several CPU-hours);
the report states the length used. Points are written as <out>/scale_<name>.json
and summarised by pscale.pg4_summary into <out>/pg4.json.

  .venv/bin/python scripts/progressive_scale.py --days 3 --workers 2 --out reports/progressive/scale
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))


def _pack(kind: str, n: int, days: int) -> Any:
    from app.eval import packs as P
    from app.pipeline.orggen import build_org, build_servers_org
    if kind == "ips":
        name = f"O-scale-{n}-0"
        return P._org_pack(name, build_org(name, portal_n=n, n_meta=0, n_days=days),
                           f"scale ips {n}", registry_mode="progressive_only", full_days=[days])
    if kind == "attrs":
        name = f"O-scale-500-{n}"
        return P._org_pack(name, build_org(name, portal_n=500, n_meta=n, n_days=days),
                           f"scale attrs {n}", registry_mode="progressive_only", full_days=[days])
    name = f"O-servers-{n}"
    return P._org_pack(name, build_servers_org(n_days=days, n_systems=n), f"servers {n}",
                       registry_mode="progressive_only", full_days=[days])


def _job(kind: str, n: int, days: int, seed: int) -> Dict[str, Any]:
    from app.eval.pscale import run_point
    from app.eval.report import _clean
    t0 = time.perf_counter()
    try:
        pt = run_point(_pack(kind, n, days), seed)
    except Exception as exc:
        return {"axis": kind, "n": n, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-3000:]}
    pt.update({"axis": kind, "n": n, "days": days, "job_s": time.perf_counter() - t0})
    return _clean(pt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ips", default="500,5000,20000")
    ap.add_argument("--attrs", default="0,60,300")
    ap.add_argument("--servers", default="20,100,300")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="reports/progressive/scale")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs: List[Tuple[str, int]] = []
    for kind, arg in (("ips", args.ips), ("attrs", args.attrs), ("servers", args.servers)):
        for x in (int(v) for v in arg.split(",") if v.strip()):
            if kind == "attrs" and x == 0 and ("ips", 500) in jobs:
                continue                                  # same pack as ips=500
            jobs.append((kind, x))
    pts: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(_job, k, n, args.days, args.seed): (k, n) for k, n in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            k, n = futs[fut]
            pts.append(r)
            with open(os.path.join(args.out, f"scale_{k}_{n}.json"), "w", encoding="utf-8") as f:
                json.dump(r, f, indent=1, ensure_ascii=False)
            print(f"[{time.perf_counter() - t0:7.1f}s] {k}={n}: {r.get('error') or '%.0fs' % r['job_s']}",
                  flush=True)
    from app.eval.pscale import pg4_summary
    ok = [p for p in pts if "error" not in p]
    summ = pg4_summary(ok) if ok else {}
    with open(os.path.join(args.out, "pg4.json"), "w", encoding="utf-8") as f:
        json.dump({"days": args.days, "seed": args.seed, "summary": summ, "points": ok,
                   "errors": [p for p in pts if "error" in p]}, f, indent=1, ensure_ascii=False, default=str)
    print(json.dumps(summ, default=str)[:3000])


if __name__ == "__main__":
    main()
