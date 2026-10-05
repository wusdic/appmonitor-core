"""Calibrate lib/pcost's engine price list (docs/lib3/progressive.md §16.12).

Runs pack O (portal 500 IPs, registry progressive_decision) for --days days,
records every engine run (measured duration, the run's own count = health
last_count, the organisation's events of its tick), and fits per engine, by
non-negative least squares on HOURLY sums from day 2 on,

    ms_h = a x runs_h + b x units_h + d x events_h

Prints the COEF table for backend/app/engines/behavior/lib/pcost.py and, per
engine, the hourly R^2 and the modelled / measured ratio per day. Run it once
per reference machine (the round-4 table: 5 days, one core):

  .venv/bin/python scripts/calibrate_costs.py --days 5 --log /tmp/costlog.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))


def record(days: int, path: str, ips: int = 500) -> None:
    from app.core import engine as CE
    from app.engines.behavior.lib import pevent as EV
    from app.eval.packs import pack_o_scale
    from app.eval.pscale import run_point
    f = open(path, "w")
    orig = CE.Engine.safe_run

    def safe_run(self, ctx, observations=None):
        r0 = self.runs
        n = orig(self, ctx, observations)
        if self.runs != r0:
            rec = {"e": self.name, "t": float(ctx.now), "ms": self.last_duration_ms, "c": self.last_count}
            if self.name == "raw.event":
                st = ctx.store
                rec["ev"] = sum(int(b.n) for b in (st.batch_at(s, EV.EVT_BATCH, ctx.now)
                                                    for s in st.batch_systems(EV.EVT_BATCH)) if b is not None)
            f.write(json.dumps(rec) + "\n")
        return n
    CE.Engine.safe_run = safe_run
    try:
        p = pack_o_scale(ips, 0, name="O-scale-cost", n_days=days)
        p.registry_mode = "progressive_decision"
        run_point(p, 0)
    finally:
        CE.Engine.safe_run = orig
        f.close()


def fit(path: str) -> dict:
    import numpy as np
    from scipy.optimize import nnls
    recs = [json.loads(x) for x in open(path)]
    ev_at = {r["t"]: r["ev"] for r in recs if r["e"] == "raw.event"}
    t0 = min(ev_at)
    by = collections.defaultdict(lambda: collections.defaultdict(lambda: np.zeros(4)))
    for r in recs:
        if r["t"] in ev_at:
            by[r["e"]][int((r["t"] - t0) // 3600)] += np.array([r["ms"], 1.0, float(r["c"] or 0), ev_at[r["t"]]])
    out = {}
    for e, hs in sorted(by.items()):
        H = sorted(hs)
        A = np.array([hs[h] for h in H])
        use = np.array([h >= 24 for h in H])
        y, X = A[:, 0], A[:, 1:]
        coef, _ = nnls(X[use], y[use])
        pred = X @ coef
        r2 = 1 - ((y[use] - pred[use]) ** 2).sum() / max(((y[use] - y[use].mean()) ** 2).sum(), 1e-12)
        days = np.array([h // 24 for h in H])
        ratio = [round(float(pred[days == d].sum() / max(y[days == d].sum(), 1e-9)), 2) for d in sorted(set(days))]
        out[e] = tuple(float(f"{x:.4g}") for x in coef)
        print(f"# {e:32s} R2(hourly)={r2:5.2f} model/measured by day={ratio}")
    print("COEF: Dict[str, Tuple[float, float, float]] = {")
    for e, c in out.items():
        print(f'    "{e}": {c},')
    print("}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--log", default="costlog.jsonl")
    ap.add_argument("--fit-only", action="store_true")
    a = ap.parse_args()
    if not a.fit_only:
        record(a.days, a.log)
    fit(a.log)


if __name__ == "__main__":
    main()
