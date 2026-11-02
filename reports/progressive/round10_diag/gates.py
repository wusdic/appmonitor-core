"""gates.py new.json old.json [older.json]: gate pass + every check whose value or pass changed."""
import json, sys
reps = [json.load(open(f)) for f in sys.argv[1:]]
def fmt(v):
    if isinstance(v, float): return f"{v:.4g}"
    return str(v)[:50]
for g in sorted(reps[0]["gates"], key=lambda x: int(x[2:])):
    G = [r["gates"].get(g) or {} for r in reps]
    print(f"== {g}: " + " | ".join(str(x.get("pass")) for x in G))
    cks = [{c["name"]: c for c in ((x.get("details") or {}).get("checks") or [])} for x in G]
    for name in cks[0]:
        vals = [(c.get(name) or {}).get("value") for c in cks]; ps = [(c.get(name) or {}).get("pass") for c in cks]
        if any(fmt(v) != fmt(vals[0]) for v in vals[1:]) or any(p != ps[0] for p in ps[1:]):
            print(f"   {name}: " + " | ".join(f"{fmt(v)} ({p})" for v, p in zip(vals, ps)))
