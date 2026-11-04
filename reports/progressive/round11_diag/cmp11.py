"""cmp11.py <new.json> [extra.json ...]: key metrics of a run against round 10, 9 and 8 (reports/progressive/round{10,9,8}/runs/<same name>) and any extra runs (bisects)."""
import json, os, sys
R8 = "/home/user/appmonitor-core/reports/progressive/round8/runs"
R10 = "/home/user/appmonitor-core/reports/progressive/round10/runs"
R9 = "/home/user/appmonitor-core/reports/progressive/round9/runs"
def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict): return None
        d = d.get(str(k))
    return d
def f(x, n=3):
    return "-" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))
def summ(fn):
    r = json.load(open(fn)); sc = r["score"]; pg1 = sc["pg1"]
    o = {}
    for d in (7, 10, 14, 18, 21, 28, 35):
        if str(d) in pg1:
            o[f"rec{d}"] = g(pg1, d, "recall")
    for d in (7, 14, 21, 35):
        if str(d) in pg1:
            o[f"prec{d}"] = g(pg1, d, "precision"); o[f"precall{d}"] = g(pg1, d, "precision_all")
    cal = g(sc, "pg2", "calibration") or {}
    o["CE"] = cal.get("ce"); o["stated"] = cal.get("conf"); o["held"] = cal.get("hold"); o["ece_pool"] = cal.get("ece")
    o["ECE14"] = g(pg1, 14, "ece")
    far = g(sc, "pg6", "far") or {}
    o["FAR_L"] = far.get("far_low"); o["FAR_M"] = far.get("far_medium"); o["PV_L"] = far.get("pv_far_low")
    an = g(sc, "pg6", "anomalies") or {}
    o["anom"] = f"{sum(1 for v in an.values() if isinstance(v, dict) and v.get('detected'))}/{len(an)}"
    o["missed_anom"] = sorted(k for k, v in an.items() if isinstance(v, dict) and not v.get("detected"))
    ck = (r.get("example") or {}).get("checklist") or []
    o["ck"] = f"{sum(1 for c in ck if c.get('pass'))}/{len(ck)}"
    o["ck_fail"] = [c["clause"][:30] for c in ck if not c.get("pass")]
    pg5 = sc.get("pg5") or {}
    o["D"] = "".join("P" if (pg5.get(f"D{i}") or {}).get("pass") else ("F" if (pg5.get(f"D{i}") or {}).get("pass") is False else "-") for i in range(1, 6))
    o["sign"] = [(a, b, l, gn, round(p, 5)) for a, b, n, m, l, gn, p in (g(sc, "pg2", "conf_steps") or [])]
    o["conf_nd"] = g(sc, "pg2", "conf_nondecreasing")
    o["bind14"] = sc.get("bindings_ga_fin_day14")
    o["fsplits"] = g(sc, "false_splits", "n")
    o["comp14"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in (g(pg1, 14, "components") or {}).items()} if isinstance(g(pg1, 14, "components"), dict) else g(pg1, 14, "components")
    o["exc"] = sc.get("exceptions"); o["wall"] = r.get("wall_s"); o["rss"] = r.get("peak_rss_mb")
    days = sorted(int(k) for k in pg1 if g(pg1, k, "per_pattern"))
    o["_miss"] = {d: sorted(k for k, v in (g(pg1, d, "per_pattern") or {}).items() if not v.get("recovered")) for d in days}
    return o
new = sys.argv[1]; extra = sys.argv[2:]
bn = os.path.basename(new)
cols = [("r11", summ(new))]
for x in extra:
    cols.append((os.path.basename(os.path.dirname(os.path.dirname(x))) or x, summ(x)))
for tag, R in (("r10", R10), ("r9", R9), ("r8", R8)):
    p = os.path.join(R, bn)
    cols.append((tag, summ(p) if os.path.exists(p) else {}))
print(f"== {new}   (" + " | ".join(t for t, _ in cols) + ")")
def fm(v): return f(v, 4) if isinstance(v, float) else v
a = cols[0][1]; r10 = dict(cols)["r10"]
for k in a:
    if k.startswith("_"): continue
    vs = [c.get(k) for _, c in cols]
    flag = "" if vs[0] == r10.get(k) else "  *"
    print(f"  {k:10s} " + " | ".join(str(fm(v)) for v in vs) + flag)
for tag, b in cols[1:]:
    if not b: continue
    for d in sorted(a["_miss"]):
        ma, mb = set(a["_miss"][d]), set((b.get("_miss") or {}).get(d, []))
        if ma != mb:
            print(f"  day {d} vs {tag}: recovered {sorted(mb - ma)}  lost {sorted(ma - mb)}")
