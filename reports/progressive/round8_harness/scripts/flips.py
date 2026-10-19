"""Claims whose held-out verdict differs retro vs prospective on given days. usage: flips.py PKL DAY..."""
import sys, pickle, json
sys.path.insert(0, "/home/user/appmonitor-core/backend")
from app.eval import pmetrics as M
res = pickle.load(open(sys.argv[1], "rb"))
pt = M.PTruth(getattr(res, "ptruth", None) or {})
cfg = getattr(res, "config", None) or {}
ipc = M.ip_classes_of(cfg)
snaps = {int(k): v for k, v in (res.psnaps or {}).items()}
seed = int(res.seed)
for day in [int(x) for x in sys.argv[2:]]:
    stmts = M.statements(snaps[day], ipc)
    valid = pt.valid_at_day(day)
    seen = set()
    for s in stmts:
        if not s.confirmed or s.negative or not M.judgeable_context(s):
            continue
        claim = M._claim_key(s)
        if claim in seen:
            continue
        seen.add(claim)
        out = {}
        for mode in (False, True):
            M.HOLDOUT_PROSPECTIVE = mode
            out[mode] = M.holdout_check(s, valid, pt, M._stream(seed, day, 2, claim), 300, day)
        if bool(out[False]["ok"]) != bool(out[True]["ok"]):
            print(day, s.system, s.pattern_id, round(s.confidence, 3), "retro", out[False]["ok"], "prosp", out[True]["ok"],
                  json.dumps({k: v for k, v in out[True].items() if k != "ok"}, default=str)[:600])
