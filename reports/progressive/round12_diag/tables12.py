"""tables11.py [runsdir]: pack O / O-real / variants tables, round 11 against rounds 10, 9 and 8."""
import json, os, sys
import numpy as np
E = sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-0/-home-user-appmonitor-core/e496f878-8276-5793-a91f-1f6d0128e252/scratchpad/prog12/ev12/out/main/runs"
R11 = "/home/user/appmonitor-core/reports/progressive/round11/runs"
R10 = "/home/user/appmonitor-core/reports/progressive/round10/runs"
R9 = "/home/user/appmonitor-core/reports/progressive/round9/runs"
R8 = "/home/user/appmonitor-core/reports/progressive/round8/runs"
RS = (("r12", E), ("r11", R11), ("r10", R10), ("r9", R9), ("r8", R8))
def L(d, f):
    p = f"{d}/{f}"
    return json.load(open(p)) if os.path.exists(p) else None
def g(sc, d, k): return (sc["pg1"].get(str(d)) or {}).get(k)
def f3(x): return "–" if x is None else f"{x:.3f}"
def f4(x): return "–" if x is None else f"{x:.4f}"
def D(sc): return "".join("P" if (sc["pg5"].get(f"D{i}") or {}).get("pass") else "F" for i in range(1, 6))
def ck(r): c = r["example"]["checklist"]; return f"{sum(1 for x in c if x.get('pass'))}/{len(c)}"
def anom(sc): a = sc["pg6"]["anomalies"]; return f"{sum(1 for v in a.values() if v.get('detected'))}/{len(a)}"
def sign(sc):
    a, b, n, m, l, gn, p = sc["pg2"]["conf_steps"][-1]; return f"{l} / {gn}, p = {p:.2g}"
def med(v): return float(np.median(v))
seeds = [s for s in range(5) if L(E, f"O_{s}.json")]
print("### pack O per seed; seeds", seeds)
print("| Seed | recall @7 / @10 / @14 / @18 / @21 | precision @7 / @14 / @21 | ECE @14 | pooled CE (stated / held) | FAR >= LOW / >= MEDIUM | D1–D5 | checklist | anomalies | sign test d10→21 | bindings @14 |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for s in seeds:
    for tag, d in RS:
        r = L(d, f"O_{s}.json")
        if r is None: continue
        sc = r["score"]; c = sc["pg2"]["calibration"]; far = sc["pg6"]["far"]
        rec = " / ".join(f"{g(sc, x, 'recall'):.3f}" for x in (7, 10, 14, 18, 21))
        print(f"| {s} {tag} | {rec} | {f3(g(sc,7,'precision'))} / {f3(g(sc,14,'precision'))} / {f3(g(sc,21,'precision'))} | {f3(g(sc,14,'ece'))} | {f3(c['ce'])} ({c['conf']:.3f} / {c['hold']:.3f}) | {f4(far['far_low'])} / {f4(far['far_medium'])} | {D(sc)} | {ck(r)} | {anom(sc)} | {sign(sc)} | {sc.get('bindings_ga_fin_day14')} |")
keys = [("PG1 recall @14", lambda sc: g(sc, 14, "recall")), ("PG1 recall @21", lambda sc: g(sc, 21, "recall")),
        ("PG1 precision @14", lambda sc: g(sc, 14, "precision")), ("PG1 precision @14, all claims", lambda sc: g(sc, 14, "precision_all")),
        ("PG1 precision @21", lambda sc: g(sc, 21, "precision")), ("PG1 precision @7", lambda sc: g(sc, 7, "precision")),
        ("PG2 CE", lambda sc: sc["pg2"]["calibration"]["ce"]), ("… stated", lambda sc: sc["pg2"]["calibration"]["conf"]),
        ("… held", lambda sc: sc["pg2"]["calibration"]["hold"]), ("… pooled plug-in ECE", lambda sc: sc["pg2"]["calibration"]["ece"]),
        ("PG6 FAR >= LOW", lambda sc: sc["pg6"]["far"]["far_low"]), ("PG6 FAR >= MEDIUM", lambda sc: sc["pg6"]["far"]["far_medium"]),
        ("PG6 pattern_violation >= LOW", lambda sc: sc["pg6"]["far"]["pv_far_low"])]
for c in ("who", "when", "content", "bindings", "workflow"):
    keys.append((f"PG1 {c} @14", lambda sc, c=c: (g(sc, 14, "components") or {}).get(c)))
print(); print("| Check (median [min–max] over seeds %s) | round 12 | round 11 | round 10 | round 9 | round 8 |" % seeds); print("|---|---|---|---|---|---|")
for name, fn in keys:
    out = []
    for tag, d in RS:
        v = [fn(L(d, f"O_{s}.json")["score"]) for s in seeds]
        fmt = f4 if "FAR" in name or "pattern_v" in name else f3
        out.append(f"{fmt(med(v))} [{fmt(min(v))}–{fmt(max(v))}]")
    print(f"| {name} | " + " | ".join(out) + " |")
for tag, d in RS:
    rs = [L(d, f"O_{s}.json") for s in seeds]
    print(tag, "mean rec14 s0-2 %.3f" % np.mean([g(r["score"], 14, "recall") for r in rs[:3]]),
          "median prec21 s0-2 %.3f" % med([g(r["score"], 21, "precision") for r in rs[:3]]),
          "D", [[D(r["score"])[i] for r in rs].count("P") for i in range(5)],
          "anom", sum(int(anom(r["score"]).split("/")[0]) for r in rs), "ck", sum(int(ck(r).split("/")[0]) for r in rs),
          "sign pass", sum(1 for r in rs if r["score"]["pg2"]["conf_nondecreasing"]),
          "bind 6/6", sum(1 for r in rs if r["score"].get("bindings_ga_fin_day14") == [6, 6]),
          "days_to_80", [r["score"]["pg2"].get("days_to_80") for r in rs])
print(); print("| | " + " | ".join(f"day {x}" for x in (3, 5, 7, 10, 14, 18, 21)) + " |"); print("|---" * 8 + "|")
for nm, k in (("recall", "recall"), ("precision (observable)", "precision"), ("plug-in ECE", "ece"), ("confirmed statements", "n_confirmed"), ("mean depth", "mean_depth")):
    for tag, d in RS:
        rs = [L(d, f"O_{s}.json")["score"] for s in seeds]
        vals = []
        for x in (3, 5, 7, 10, 14, 18, 21):
            v = [g(sc, x, k) for sc in rs]; v = [y for y in v if y is not None]
            vals.append(f"{med(v):.2f}" if v else "–")
        print(f"| {nm}, {tag} | " + " | ".join(vals) + " |")
# pooled claims
print()
for tag, d in RS:
    for day in (14, 21):
        n = f = 0; cs = []
        for s in seeds:
            sc = L(d, f"O_{s}.json")["score"]; p = sc["pg1"].get(str(day)) or {}
            k = p.get("n_precision") or 0; n += k; f += round(k * (1 - (p.get("precision") or 0)))
            sc_ = p.get("stmt_conf") or {}
        print(tag, day, "pooled observable claims", n, "failed", f, "precision %.3f" % (1 - f / max(n, 1)))
# O-real
print("\n### O-real")
print("| Seed | recall @7 / @14 / @21 / @28 / @35 | precision @14 / @21 / @35 | pooled CE (stated / held) | anomalies | FAR >= LOW / >= MEDIUM | D1–D5 | bindings @14 | checklist (day 35) | sign test d10→21 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for s in range(3):
    for tag, d in RS:
        r = L(d, f"O-real_{s}.json")
        if r is None: continue
        sc = r["score"]; c = sc["pg2"]["calibration"]; far = sc["pg6"]["far"]; ckk = r["example"]["checklist"]; cs = sc["pg2"]["conf_steps"][-1]
        print(f"| {s} {tag} | " + " / ".join(f"{g(sc,x,'recall'):.3f}" for x in (7,14,21,28,35)) + " | " + " / ".join(f"{g(sc,x,'precision'):.3f}" for x in (14,21,35)) +
              f" | {c['ce']:.3f} ({c['conf']:.2f} / {c['hold']:.2f}) | {anom(sc)} | {far['far_low']:.4f} / {far['far_medium']:.4f} | {D(sc)} | {sc.get('bindings_ga_fin_day14')[0]}/6 | {sum(1 for x in ckk if x.get('pass'))}/10 | {cs[4]} / {cs[5]}, p = {cs[6]:.2g} |")
print("\n### variants")
for v in ("O-red_0.json", "O60_0.json"):
    print("|", v, "| r12 | r11 | r10 | r9 | r8 |"); print("|---|---|---|---|---|---|")
    rr = [L(d, v) for _, d in RS]
    if rr[0] is None: continue
    rows = [("recall @7/@14/@21", lambda r: " / ".join(f3(g(r["score"], x, "recall")) for x in (7, 14, 21))),
            ("precision @7/@14/@21", lambda r: " / ".join(f3(g(r["score"], x, "precision")) for x in (7, 14, 21))),
            ("precision all @14", lambda r: f3(g(r["score"], 14, "precision_all"))),
            ("recall/precision @8", lambda r: f"{f3(g(r['score'], 8, 'recall'))} / {f3(g(r['score'], 8, 'precision'))}"),
            ("components @14", lambda r: " / ".join(f3((g(r["score"], 14, "components") or {}).get(c)) for c in ("who", "when", "content", "bindings"))),
            ("anomalies", lambda r: anom(r["score"]) + " " + str(sorted(k for k, x in r["score"]["pg6"]["anomalies"].items() if not x.get("detected")))),
            ("FAR L / M", lambda r: f"{f4(r['score']['pg6']['far']['far_low'])} / {f4(r['score']['pg6']['far']['far_medium'])}"),
            ("D1–D5", lambda r: D(r["score"])), ("CE (stated / held)", lambda r: f"{f3(r['score']['pg2']['calibration']['ce'])} ({r['score']['pg2']['calibration']['conf']:.3f} / {r['score']['pg2']['calibration']['hold']:.3f})"),
            ("sign", lambda r: sign(r["score"])), ("checklist", ck)]
    for nm, fn in rows:
        print(f"| {nm} | " + " | ".join(str(fn(r)) if r else "–" for r in rr) + " |")
