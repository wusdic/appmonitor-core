"""bisect11.py OUT.json: key metrics of every round-11 bisection point and final run (scores only)."""
import json, os, sys, glob
T = "/tmp/claude-0/-home-user-appmonitor-core/e496f878-8276-5793-a91f-1f6d0128e252/scratchpad/prog11"
pts = {"final_code11": f"{T}/ev11/out/main/runs", "r10": "/home/user/appmonitor-core/reports/progressive/round10/runs",
       "fin5_bounds_only": None, "t5_tree_only": f"{T}/tree/out/t5/runs", "t7_tree_only_with_T11-5_6": f"{T}/tree/out/t7/runs",
       "bis_c11t7_merged_with_T11-5_6": f"{T}/ev11/out/bis_c11t7/runs", "bis_t5_tree_only": f"{T}/ev11/out/bis_t5/runs"}
def files(k, d):
    if k == "fin5_bounds_only":
        return sorted(glob.glob(f"{T}/bounds/out/fin5_*/runs/*.json"))
    return sorted(glob.glob(f"{d}/*.json"))
def summ(fn):
    r = json.load(open(fn)); sc = r["score"]; pg1 = sc["pg1"]
    o = {d: {"recall": pg1[d].get("recall"), "precision": pg1[d].get("precision")} for d in ("7", "8", "14", "18", "21", "35") if d in pg1}
    c = sc["pg2"]["calibration"]; far = sc["pg6"]["far"]; an = sc["pg6"]["anomalies"]
    ck = (r.get("example") or {}).get("checklist") or []
    o.update(CE=c.get("ce"), stated=c.get("conf"), held=c.get("hold"), far_low=far.get("far_low"), far_medium=far.get("far_medium"),
             anomalies=f"{sum(1 for v in an.values() if v.get('detected'))}/{len(an)}",
             D="".join("P" if (sc["pg5"].get(f"D{i}") or {}).get("pass") else "F" for i in range(1, 6)),
             checklist=f"{sum(1 for x in ck if x.get('pass'))}/{len(ck)}", bindings_day14=sc.get("bindings_ga_fin_day14"),
             pg11_pass=sorted(k for k, v in (sc.get("pg11") or {}).items() if v.get("pass")))
    return o
out = {}
for k, d in pts.items():
    for fn in files(k, d):
        out.setdefault(k, {})[os.path.basename(fn)[:-5]] = summ(fn)
json.dump(out, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
print({k: sorted(v) for k, v in out.items()})
