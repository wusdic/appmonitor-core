"""P00/P01/P02/P05 thread CPU per input record on days 3-4 of a 4-day O-scale-500-<n_meta>
run (progressive_only), plus P02's work counters (calls per batch by role, rows per call).
usage: p02prof.py CODE N_META OUT_PREFIX [PROF=1]"""
import os, sys, time, json, cProfile, pstats, io, collections
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
CODE = sys.argv[1]
sys.path.insert(0, os.path.join(CODE, "backend"))
from app.eval import packs as P
from app.eval.runner import run_pack, default_registry_factory, _call_with_supported
from app.pipeline.orggen import build_org
import app.pipeline.orggen as OG
n_meta = int(sys.argv[2]); out = sys.argv[3]; days = 4
PROF = (sys.argv[4] if len(sys.argv) > 4 else "1") == "1"
ENG = ("raw.event", "derived.event_context", "behavior.attr_registry", "behavior.attr_select",
       "behavior.pattern_tree")
name = f"O-scale-500-{n_meta}"
pack = P._org_pack(name, build_org(name, portal_n=500, n_meta=n_meta, n_days=days),
                   f"attrs {n_meta}", registry_mode="progressive_only", full_days=[days])
T_ATTR = OG.OrgGenerator(pack.org, seed=0).day_start(3)
profs = {e: cProfile.Profile() for e in ENG}
cpu = {e: 0.0 for e in ENG}
rows = {"obs": 0}
cnt = collections.Counter()
ON = {"on": False}


def factory(pack=None, config=None, seed=0):
    reg = _call_with_supported(default_registry_factory, pack=pack, config=config, seed=seed)
    for eng in reg.ordered():
        if eng.name == "behavior.attr_registry":
            ob0 = eng._observe

            def _observe(reg_, b, cb, ts_b, roles, kind, s, _o=ob0):
                if ON["on"]:
                    cnt["batches"] += 1
                    names = set(b.cols) | (set(cb.cols) if cb is not None and cb.n == b.n else set())
                    cnt["cols"] += len(names)
                    for nm in names:
                        cnt["role:" + str(roles.get(nm, "-"))] += 1
                    r0 = reg_.observe

                    def obs(nm, vals, *a, _r=r0, **k):
                        cnt["calls"] += 1
                        cnt["rows"] += len(vals)
                        return _r(nm, vals, *a, **k)
                    reg_.observe = obs
                    try:
                        return _o(reg_, b, cb, ts_b, roles, kind, s)
                    finally:
                        del reg_.observe
                return _o(reg_, b, cb, ts_b, roles, kind, s)
            eng._observe = _observe
        if eng.name in ENG:
            r0 = eng.run

            def run(ctx, observations=None, _r=r0, _n=eng.name):
                on = ctx.now - ctx.window_s >= T_ATTR
                ON["on"] = on
                if on and _n == "raw.event":
                    rows["obs"] += sum(int(((o.extra or {}).get("count", 1)) or 1) for o in observations or ())
                t = time.thread_time()
                if on and PROF:
                    profs[_n].enable()
                try:
                    return _r(ctx, observations)
                finally:
                    if on and PROF:
                        profs[_n].disable()
                    if on:
                        cpu[_n] += time.thread_time() - t
            eng.run = run
    return reg


t0 = time.time()
res = run_pack(pack, 0, registry_factory=factory, record_series=False)
ev = rows["obs"]
summ = {"n_meta": n_meta, "days": days, "events": ev, "cpu_s": cpu, "wall_s": time.time() - t0,
        "us_per_event": {e: cpu[e] * 1e6 / max(ev, 1) for e in ENG},
        "p02": dict(cnt), "calls_per_batch": cnt["calls"] / max(1, cnt["batches"]),
        "rows_per_call": cnt["rows"] / max(1, cnt["calls"]), "exceptions": len(res.exceptions)}
json.dump(summ, open(out + ".json", "w"), indent=1)
if PROF:
    for e in ENG:
        s = io.StringIO()
        pstats.Stats(profs[e], stream=s).sort_stats("tottime").print_stats(30)
        open(out + f".{e}.txt", "w").write(s.getvalue())
print(json.dumps(summ))
