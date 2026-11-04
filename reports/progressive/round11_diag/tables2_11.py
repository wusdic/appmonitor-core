import json
E = "/tmp/claude-0/-home-user-appmonitor-core/e496f878-8276-5793-a91f-1f6d0128e252/scratchpad/prog11/ev11/out/main/runs"
R8 = "/home/user/appmonitor-core/reports/progressive/round10/runs"
L = lambda d, f: json.load(open(f"{d}/{f}"))
g = lambda sc, d, k: (sc["pg1"].get(str(d)) or {}).get(k)
D = lambda sc: "".join("P" if (sc["pg5"].get(f"D{i}") or {}).get("pass") else "F" for i in range(1, 6))
print("| Seed | recall @7 / @14 / @21 / @28 / @35 | precision @14 / @21 / @35 | pooled CE (stated / held) | anomalies | FAR >= LOW / >= MEDIUM | D1–D5 | bindings @14 | checklist (day 35) | sign test d10→21 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for s in range(3):
    for tag, d in (("after", E), ("before", R8)):
        r = L(d, f"O-real_{s}.json"); sc = r["score"]; c = sc["pg2"]["calibration"]; far = sc["pg6"]["far"]
        an = sc["pg6"]["anomalies"]; ck = r["example"]["checklist"]; cs = sc["pg2"]["conf_steps"][-1]
        print(f"| {s} {tag} | " + " / ".join(f"{g(sc,x,'recall'):.2f}" for x in (7,14,21,28,35)) +
              f" | " + " / ".join(f"{g(sc,x,'precision'):.2f}" for x in (14,21,35)) +
              f" | {c['ce']:.3f} ({c['conf']:.2f} / {c['hold']:.2f}) | {sum(1 for v in an.values() if v.get('detected'))}/{len(an)} | {far['far_low']:.4f} / {far['far_medium']:.4f} | {D(sc)} | {sc.get('bindings_ga_fin_day14')[0]}/6 | {sum(1 for x in ck if x.get('pass'))}/10 | {cs[4]}/{cs[5]}, p = {cs[6]:.2g} |")
def flags(fn):
    a = L(*fn)["anomalies"]
    return {k: (", ".join(v.get("types") or []) if v.get("detected") else "**missed**") for k, v in a.items()}
print()
for pack, n in (("O", 5), ("O-real", 3)):
    fl = [flags((E, f"{pack}_{s}.json")) for s in range(n)]
    print(f"| Anomaly ({pack}) | " + " | ".join(f"s{s}" for s in range(n)) + " |")
    print("|---" * (n + 1) + "|")
    for k in sorted(fl[0], key=lambda x: int(x[1:])):
        print(f"| {k} | " + " | ".join(f[k] for f in fl) + " |")
    ck = [L(E, f"{pack}_{s}.json")["example"]["checklist"] for s in range(n)]
    print(f"| Clause ({pack}) | " + " | ".join(f"s{s}" for s in range(n)) + " |")
    print("|---" * (n + 1) + "|")
    for i, c in enumerate(ck[0]):
        print(f"| {c['clause']} | " + " | ".join(("✓ " if x[i].get('pass') else "✗ ") + str(x[i].get('measured')) for x in ck) + " |")
    print()
for v in ("O-red_0.json", "O60_0.json"):
    for d in (E, R8):
        r = L(d, v); a = r["anomalies"]
        print(v, d[-12:], {k: (", ".join(x.get("types") or []) if x.get("detected") else "MISSED") for k, x in a.items()})
