"""Aggregate round-7 re-scoring: published (round-7 scorer) / retro (round-8 PG2 rule only) /
prosp (both lead decisions). Writes OUT.json and prints a markdown table."""
import json, os, sys, statistics as S
H = os.path.dirname(os.path.abspath(__file__))
PUB = "/home/user/appmonitor-core/reports/progressive/runs"
out_p = sys.argv[1]
runs = [("O", s) for s in range(5)] + [("O-real", s) for s in range(3)]


def load(p):
    d = json.load(open(p))
    return d.get("score", d)


def row(sc):
    pg1 = sc["pg1"]; days = sorted(int(k) for k in pg1); last = days[-1]
    d14 = 14 if 14 in days else last
    p2 = sc["pg2"]; cal = p2["calibration"]
    steps = p2["conf_steps"]
    return {"recall14": pg1[str(d14)]["recall"], "recallL": pg1[str(last)]["recall"],
            "prec14": pg1[str(d14)]["precision"], "prec14_all": pg1[str(d14)]["precision_all"],
            "precL": pg1[str(last)]["precision"], "ce": cal["ce"], "conf": cal["conf"], "hold": cal["hold"],
            "n_calib": cal["n"], "conf_ok": p2["conf_nondecreasing"],
            "conf_ok_strict": p2.get("conf_nondecreasing_strict", p2["conf_nondecreasing"]),
            "steps": steps, "last_day": last}


res = {}
for pk, s in runs:
    k = f"{pk}_{s}"
    ent = {}
    pp = os.path.join(PUB, f"{k}.json")
    if os.path.exists(pp):
        ent["published"] = row(load(pp))
    for m in ("retro", "prosp"):
        p = os.path.join(H, "rs", f"{k}.{m}.json")
        if os.path.exists(p):
            ent[m] = row(load(p))
    rp = os.path.join(H, "runs7", f"{k}.json")
    if os.path.exists(rp) and "published" in ent:
        r = row(load(rp))
        ent["reproduced"] = all(r[x] == ent["published"][x] for x in ("recall14", "prec14", "ce", "steps"))
    res[k] = ent
# claims whose verdict changed retro -> prosp (days >= 7; calib lists are aligned per claim)
for pk, s in runs:
    k = f"{pk}_{s}"
    pa, pb = (os.path.join(H, "rs", f"{k}.{m}.json") for m in ("retro", "prosp"))
    if not (os.path.exists(pa) and os.path.exists(pb)):
        continue
    A, B = load(pa)["pg1"], load(pb)["pg1"]
    dn = up = n = 0
    for d in A:
        if int(d) < 7:
            continue
        ca, cb = A[d].get("calib") or [], B[d].get("calib") or []
        n += len(ca)
        dn += sum(1 for x, y in zip(ca, cb) if x[1] > y[1])
        up += sum(1 for x, y in zip(ca, cb) if x[1] < y[1])
    res[k]["flips"] = {"n": n, "held_to_not": dn, "not_to_held": up,
                       "pairs": [x[2] for x in res[k]["prosp"]["steps"]]}
json.dump(res, open(out_p, "w"), indent=1, default=str)
f = lambda x: "–" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
print("| run | recall@14 | precision@14 (all) pub → prosp | CE pub → prosp | stated / held pub → prosp | PG2 conf: strict → sign test | steps (median; lost/gained, p) |")
print("|---|---|---|---|---|---|---|")
for k, e in res.items():
    if "prosp" not in e:
        continue
    a, b = e.get("published") or e["retro"], e["prosp"]
    st = "; ".join(f"{x[0]}→{x[1]} {x[3]:+.4f} ({x[4]}/{x[5]}, p={x[6]:.2f})" for x in b["steps"])
    print(f"| {k} | {f(a['recall14'])} | {f(a['prec14'])} ({f(a['prec14_all'])}) → {f(b['prec14'])} ({f(b['prec14_all'])}) "
          f"| {f(a['ce'])} → {f(b['ce'])} | {f(a['conf'])} / {f(a['hold'])} → {f(b['hold'])} "
          f"| {'✓' if a['conf_ok_strict'] else '✗'} → {'✓' if b['conf_ok'] else '✗'} | {st} |")
for pk in ("O", "O-real"):
    ks = [k for k in res if k.startswith(pk + "_") and "prosp" in res[k]]
    if not ks:
        continue
    for m in ("published", "retro", "prosp"):
        v = [res[k][m] for k in ks if m in res[k]]
        if not v:
            continue
        ok = sum(1 for x in v if (x["conf_ok"] if m != "published" else x["conf_ok_strict"]))
        print(f"{pk} {m}: n={len(v)} CE median {S.median(x['ce'] for x in v):.3f} mean {S.mean(x['ce'] for x in v):.3f} "
              f"prec14 median {S.median(x['prec14'] for x in v):.3f} all {S.median(x['prec14_all'] for x in v):.3f} "
              f"hold {S.median(x['hold'] for x in v):.3f} conf {S.median(x['conf'] for x in v):.3f} PG2 conf ok {ok}/{len(v)}")
