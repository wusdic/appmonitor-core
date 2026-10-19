"""Re-score a pickled RunResult with the working tree's pmetrics.
usage: rescore.py RES_PKL OUT_JSON [retro|prosp]
retro = round-7 held-out traffic (HOLDOUT_PROSPECTIVE False), PG2 sign test on;
prosp = round 8 (both lead decisions)."""
import json, os, pickle, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, "/home/user/appmonitor-core/backend")
from app.eval import pmetrics as M
from app.eval.report import _clean

res_p, out = sys.argv[1], sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "prosp"
M.HOLDOUT_PROSPECTIVE = (mode == "prosp")
t0 = time.time()
with open(res_p, "rb") as f:
    res = pickle.load(f)
sc = M.score_prun(res)
sc["_mode"] = mode
sc["_score_s"] = time.time() - t0
with open(out + ".tmp", "w") as f:
    json.dump(_clean(sc), f, default=str, ensure_ascii=False)
os.replace(out + ".tmp", out)
p2 = sc["pg2"]
print(json.dumps({"mode": mode, "ce": p2["calibration"], "conf_ok": p2["conf_nondecreasing"],
                  "conf_ok_strict": p2.get("conf_nondecreasing_strict"), "conf_steps": p2["conf_steps"],
                  "s": round(time.time() - t0, 1)}, default=str))
