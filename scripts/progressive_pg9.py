"""PG9 bounded mode: paired `full` / `bounded` runs (docs/lib3/progressive.md
§12 PG9, round-7 lead decision; docs/lib3/eval.md).

  pairs   lib-3 packs (default A,E): each pack's full and bounded run of the
          same seed, started together (two processes), scored with
          metrics.score_run; PG9 asks no accuracy loss (gates 1, 3, 8) and no
          lib-3 CPU increase.
  large   pack O with portal_n transient public IPs (PG4's IP points, default
          5000,20000) on the default lib-3 registry: full and bounded run of the
          same seed together; PG9 asks lib-3 CPU p95 bounded <= 0.5 x full.
          --days shortens the pack; --hours stops both runs after that many
          simulated hours (the full run at 20 000 IPs does not fit in memory for
          a week), and the summary compares the ticks both runs completed.

Each run writes <out>/<kind>_<name>_<mode>.json; --summary assembles
<out>/pg9.json with pmetrics.pg9_bounded, which progressive_report.py reads
(--pg9).

  .venv/bin/python scripts/progressive_pg9.py pairs --packs A,E --out reports/progressive/pg9
  .venv/bin/python scripts/progressive_pg9.py large --ips 5000 --days 3 --out reports/progressive/pg9
  .venv/bin/python scripts/progressive_pg9.py summary --out reports/progressive/pg9
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

MODES = ("full", "bounded")


def _set_mode(pack: Any, mode: str) -> Any:
    pack.config.setdefault("lib3", {})["resource_mode"] = mode
    return pack


def _write(path: str, obj: Dict[str, Any]) -> None:
    from app.eval.report import _clean
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_clean(obj), f, indent=1, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def _pair_job(pack_name: str, seed: int, mode: str, out: str) -> str:
    from app.eval.metrics import score_run
    from app.eval.packs import get_pack
    from app.eval.pmetrics import pg9_run_summary
    from app.eval.pscale import lib3_cost, lib3_tick_cpu
    from app.eval.runner import run_pack
    t0 = time.perf_counter()
    try:
        res = run_pack(_set_mode(get_pack(pack_name), mode), seed, strict=True)
        sc = score_run(res)
        cost = lib3_cost(res.timings)
        ticks = lib3_tick_cpu(res.timings)
        _write(out, {"kind": "pair", "pack": pack_name, "seed": seed, "mode": mode,
                     "summary": pg9_run_summary(sc, cost), "cost": cost,
                     "tick_cpu_ms": [] if ticks is None else [round(float(x), 2) for x in ticks],
                     "far": sc.get("far"), "scenarios": sc.get("scenarios"),
                     "identification": sc.get("identification"), "cpu_mem": sc.get("cpu_mem"),
                     "robustness": sc.get("robustness"), "exceptions": len(res.exceptions),
                     "job_s": time.perf_counter() - t0,
                     "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0})
    except Exception as exc:                                    # recorded, not raised
        _write(out, {"kind": "pair", "pack": pack_name, "seed": seed, "mode": mode,
                     "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-3000:]})
    return out


def scale_pack(n_ips: int, days: int, registry_mode: str = "full") -> Any:
    """Pack O with portal_n transient public IPs (each active on Poisson(3) of
    the days, orggen 'public' department) on the lib-3 registry."""
    from app.eval import packs as P
    from app.pipeline.orggen import build_org
    name = f"O-scale-{n_ips}-0"
    return P._org_pack(name, build_org(name, portal_n=n_ips, n_meta=0, n_days=days),
                       f"PG9 large population: portal_n={n_ips}, {days} d",
                       registry_mode=registry_mode, full_days=[days])


RSS_LIMIT_MB = float(os.environ.get("PG9_RSS_LIMIT_MB", "8500"))


def _rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


class MemoryBudgetExceeded(RuntimeError):
    pass


class _GuardGen:
    """Generator wrapper that stops the run (before the next tick) once the
    process holds more than RSS_LIMIT_MB: run_pack then ends with
    aborted='exception: MemoryBudgetExceeded ...' and the timings of the ticks
    it completed, which the summary compares with the other mode's same ticks.
    (Full-mode lib-3 at 5 000 transient IPs held 6.8 GB within a day; two such
    runs on the 15 GB evaluation machine were killed by the kernel with no
    result.)"""

    def __init__(self, gen: Any, limit_mb: float) -> None:
        self.gen = gen
        self.limit_mb = float(limit_mb)
        self.peak_mb = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.gen, name)

    def step(self, dt: float, live: bool = False, aggregated: Optional[bool] = None) -> List[Any]:
        rss = _rss_mb()
        self.peak_mb = max(self.peak_mb, rss)
        if rss > self.limit_mb:                       # (checked between ticks: keep a margin)
            raise MemoryBudgetExceeded(f"RSS {rss:.0f} MB > {self.limit_mb:.0f} MB")
        return self.gen.step(dt, live=live, aggregated=aggregated)


def _large_job(n_ips: int, days: int, hours: Optional[float], seed: int, mode: str, out: str) -> str:
    from app.eval.pscale import lib3_cost, lib3_tick_cpu
    from app.eval.runner import run_pack
    from app.pipeline.generator import TrafficGenerator
    t0 = time.perf_counter()
    try:
        pack = _set_mode(scale_pack(n_ips, days), mode)
        if hours:
            pack.end_epoch = min(pack.end_epoch, pack.start_epoch + float(hours) * 3600.0)
            n = int(round(float(hours) * 3600.0 / 900.0))
            pack.phases = [(n, 900.0, *tuple(pack.phases[0])[2:])]
        holder: Dict[str, Any] = {}

        def gen_factory(pack: Any, seed: int) -> Any:
            g = holder["g"] = _GuardGen(TrafficGenerator(seed=seed, pack=pack), RSS_LIMIT_MB)
            return g

        res = run_pack(pack, seed, strict=True, record_series=False, generator_factory=gen_factory)
        ticks = lib3_tick_cpu(res.timings)
        tl = res.timings.get("ticks")
        rank = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        n_inc: Dict[str, int] = {}
        for inc in res.incidents or []:
            sevs = [h.get("severity") for h in inc.get("history") or []] + [inc.get("severity")]
            sev = max((str(x) for x in sevs if x), key=lambda x: rank.get(x, 0), default="info")
            n_inc[sev] = n_inc.get(sev, 0) + 1
        _write(out, {"kind": "large", "n_ips": n_ips, "days": days, "hours": hours, "seed": seed,
                     "known_ips_end": len({str(k).split("|", 1)[-1] for k in (res.entity_first_seen or {})}),
                     "mode": mode, "cost": lib3_cost(res.timings),
                     "tick_cpu_ms": [] if ticks is None else [round(float(x), 2) for x in ticks],
                     "tick_ts": [] if tl is None else [float(r[0]) for r in tl],
                     "incidents_by_severity": n_inc, "exceptions": len(res.exceptions),
                     "exception_heads": [str(e)[:300] for e in res.exceptions[:5]],
                     "aborted": res.aborted, "job_s": time.perf_counter() - t0,
                     "rss_limit_mb": RSS_LIMIT_MB,
                     "memory_final": (res.memory or {}).get("final"),
                     "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0})
    except Exception as exc:
        _write(out, {"kind": "large", "n_ips": n_ips, "mode": mode,
                     "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-3000:]})
    return out


def _run_pairs(jobs: List[tuple], workers: int) -> None:
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        for fut in [ex.submit(*j) for j in jobs]:
            print("done", fut.result(), flush=True)


def summary(out: str) -> Dict[str, Any]:
    """pg9.json: the gate (pmetrics.pg9_bounded) and its inputs. Large points
    compare the ticks both runs completed (the shorter run's prefix)."""
    import numpy as np
    from app.eval.pmetrics import pg9_bounded
    runs: Dict[str, Dict[str, Any]] = {}
    for fn in sorted(os.listdir(out)):
        if not fn.endswith(".json") or fn == "pg9.json":
            continue
        with open(os.path.join(out, fn), encoding="utf-8") as f:
            r = json.load(f)
        if "error" in r or r.get("mode") not in MODES:
            continue
        key = (f"{r['pack']} seed {r['seed']}" if r["kind"] == "pair" else
               f"O-scale {r['n_ips']} IPs ({r.get('days')}-day pack" +
               (f", first {r['hours']:g} h)" if r.get("hours") else ")"))
        runs.setdefault(r["kind"], {}).setdefault(key, {})[r["mode"]] = r
    pairs = {k: {m: v[m]["summary"] for m in MODES} for k, v in (runs.get("pair") or {}).items()
             if all(m in v for m in MODES)}
    large: Dict[str, Dict[str, Any]] = {}
    for k, v in (runs.get("large") or {}).items():
        if not all(m in v for m in MODES):
            continue
        a = np.asarray(v["full"]["tick_cpu_ms"], dtype=float)
        b = np.asarray(v["bounded"]["tick_cpu_ms"], dtype=float)
        n = min(a.size, b.size)
        if n == 0:
            continue
        ts = np.asarray(v["full"].get("tick_ts") or [], dtype=float)
        span_h = (float(ts[n - 1] - ts[0]) / 3600.0 + 0.25) if ts.size >= n > 1 else n * 0.25
        large[k] = {m: {"p95_ms": float(np.quantile(x[:n], 0.95)), "mean_ms": float(x[:n].mean()),
                        "n_ticks": n, "span_h": span_h, "basis": "thread CPU, common ticks"}
                    for m, x in (("full", a), ("bounded", b))}
        # the last 24 h both completed: the cost once the known population has grown
        last = slice(max(0, n - 96), n)
        for m, x in (("full", a), ("bounded", b)):
            large[k][m]["p95_ms_last_day"] = float(np.quantile(x[last], 0.95))
    g = pg9_bounded(pairs, large)
    rep = {"gate": g, "pairs": pairs, "large": large,
           "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    _write(os.path.join(out, "pg9.json"), rep)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("pairs", "large", "summary"))
    ap.add_argument("--packs", default="A,E")
    ap.add_argument("--ips", default="5000,20000")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="reports/progressive/pg9")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--modes", default="full,bounded",
                    help="large: the modes to run (one at a time when memory is short: the "
                         "summary compares the ticks both runs completed)")
    ap.add_argument("--sequential", action="store_true",
                    help="run the jobs one after another (a full-mode 5 000-IP run holds GBs)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.what == "summary":
        rep = summary(args.out)
        g = rep["gate"]
        print(f"PG9: pass={g.get('pass')} value={g.get('value')}")
        for c in g["details"]["checks"]:
            print(f"  {c['name']}: {c['value']} ({c['target']}) -> {c['pass']}")
        return
    from app.pipeline.build import signature_status
    st = signature_status()        # round 8: the lib-3 registry needs the lib-4 library
    if not st["ok"]:
        raise SystemExit(f"lib-4 signature library missing ({', '.join(st['missing'])} in {st['dir']}): "
                         "copy or link data/ into the code snapshot, or set APPMON_SIGNATURE_DIR")
    jobs = []
    if args.what == "pairs":
        for p in [x.strip() for x in args.packs.split(",") if x.strip()]:
            for m in MODES:
                fn = os.path.join(args.out, f"pair_{p}_{args.seed}_{m}.json")
                if not (args.skip_existing and os.path.exists(fn)):
                    jobs.append((_pair_job, p, args.seed, m, fn))
    else:
        modes = [m for m in args.modes.split(",") if m in MODES]
        for n in [int(x) for x in args.ips.split(",") if x.strip()]:
            for m in modes:
                tag = f"{n}_{args.days}d" + (f"_{args.hours:g}h" if args.hours else "")
                fn = os.path.join(args.out, f"large_{tag}_{args.seed}_{m}.json")
                if not (args.skip_existing and os.path.exists(fn)):
                    jobs.append((_large_job, n, args.days, args.hours, args.seed, m, fn))
    # a pair's two runs start together (same machine load); pairs one after another
    step = 1 if args.sequential else 2
    for i in range(0, len(jobs), step):
        _run_pairs(jobs[i:i + step], args.workers)


if __name__ == "__main__":
    from app.eval.pscale import pin_hash_seed
    pin_hash_seed()
    main()
