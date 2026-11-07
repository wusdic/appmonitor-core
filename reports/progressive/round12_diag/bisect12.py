"""bisect12.py: key metrics of every round-12 bisection / decision point -> round12_diag/bisect12.json"""
import json, glob, os, sys
sys.argv = ["cmp12.py", "x"]
ns = {}
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cmp12.py")).read()
exec(src.split("new = sys.argv[1]")[0], ns)
summ = ns["summ"]
E = os.path.dirname(os.path.abspath(__file__))
out = {}
for fn in sorted(glob.glob(os.path.join(E, "out", "*", "runs", "*.json"))):
    pt = fn.split(os.sep)[-3]
    if pt == "main" or pt.startswith(("bis_", "b6", "det_")):
        s = summ(fn)
        out[f"{pt}/{os.path.basename(fn)}"] = {k: v for k, v in s.items() if not k.startswith("_") and k not in ("sign", "comp14", "wall", "rss")}
json.dump({"points": {"main": "code12 (final)", "b6": "code12 with B6 pattern_tree (EV12-2)", "det_h1": "code12 under PYTHONHASHSEED=1",
                      "bis_LV": "code11 + learn (pattern_tree, system_profile EV12-1) + views", "bis_VB": "code11 + views + bounds",
                      "bis_V": "code11 + views only", "bis_L": "code11 + learn only (pattern_tree + EV12-1)"},
           "runs": out}, open("/home/user/appmonitor-core/reports/progressive/round12_diag/bisect12.json", "w"), indent=1, ensure_ascii=False, default=str)
print(len(out))
