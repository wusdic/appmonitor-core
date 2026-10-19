"""Compare scores: cmp.py LABEL=path.json ... (a drv7 run JSON {score:...} or a rescore JSON)."""
import json, sys


def load(p):
    d = json.load(open(p))
    return d.get("score", d)


def row(lbl, s):
    pg1 = s["pg1"]
    days = sorted(int(k) for k in pg1)
    last = days[-1]
    d14 = 14 if 14 in days else last
    p2 = s["pg2"]
    cal = p2["calibration"]
    co = p2.get("calibration_observable") or {}
    ck = p2.get("calibration_by_kind") or {}
    f = lambda x: "-" if x is None else f"{x:.3f}"
    steps = ";".join(f"{a}-{b}:{m:+.4f}" + (f"({x[4]}/{x[5]},p={x[6]:.3f})" if len(x) > 6 else "")
                     for x in p2["conf_steps"] for a, b, n, m in [x[:4]])
    print(f"{lbl:14s} rec@14 {f(pg1[str(d14)]['recall'])} @L {f(pg1[str(last)]['recall'])} "
          f"prec@14 {f(pg1[str(d14)]['precision'])} all {f(pg1[str(d14)]['precision_all'])} "
          f"prec@L {f(pg1[str(last)]['precision'])} all {f(pg1[str(last)]['precision_all'])} | "
          f"CE {f(cal['ce'])} conf {f(cal['conf'])} hold {f(cal['hold'])} n {cal['n']} "
          f"ECEp {f(cal['ece'])} | node {f((ck.get('node') or {}).get('ce'))} "
          f"part {f((ck.get('part') or {}).get('ce'))} obs {f(co.get('ce'))} | "
          f"confOK {p2['conf_nondecreasing']} strict {p2.get('conf_nondecreasing_strict')} | {steps}")


for a in sys.argv[1:]:
    lbl, p = a.split("=", 1)
    row(lbl, load(p))
