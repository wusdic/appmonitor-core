"""farinc.py <codedir> <res.pkl> [out.json]: the incidents counted in PG6 FAR (clean entity-days), with
severity, time, and the evidence findings; plus D4 drift incidents."""
import json, os, pickle, sys, datetime
code, fn = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(code, "backend"))
from app.eval import pmetrics as PM
res = pickle.load(open(fn, "rb"))
pt = PM.PTruth(getattr(res, "ptruth", None) or {})
dirty = set()
for r in getattr(res, "truth", None) or []:
    if r.get("label") == "malicious" or str(r.get("scenario_id", "")).startswith("R"):
        dirty |= set(map(str, r.get("entities") or []))
        for x in (r.get("params") or {}).values():
            if isinstance(x, str) and PM._ip(x) is not None:
                dirty.add(x)
clean = set()
for s, per_date in (pt.who_log or {}).items():
    for iso, per in per_date.items():
        for ip in per:
            if ip not in dirty:
                clean.add((s, ip))
out = []
for inc in res.incidents or []:
    k = (inc.get("system"), inc.get("entity"))
    if k not in clean:
        continue
    mx = max([PM.sev_rank(h.get("severity")) for h in inc.get("history") or []] + [PM.sev_rank(inc.get("severity"))])
    if mx < PM.SEV_RANK["low"]:
        continue
    ev = []
    for e in (inc.get("evidence") or [])[:6]:
        ev.append(str({x: e.get(x) for x in ("kind", "detector", "family", "p", "description", "extra")} if isinstance(e, dict) else e)[:400])
    ts = inc.get("first_ts") or inc.get("ts") or inc.get("opened")
    out.append({"system": k[0], "entity": k[1], "max_sev": mx, "id": inc.get("id"),
                "ts": str(datetime.datetime.utcfromtimestamp(float(ts)) + datetime.timedelta(hours=8)) if isinstance(ts, (int, float)) else str(ts),
                "keys": sorted(inc.keys()), "summary": str(inc.get("summary") or inc.get("title") or "")[:300], "evidence": ev})
out.sort(key=lambda x: (-x["max_sev"], x["system"], str(x["entity"])))
for o in out:
    print(o["max_sev"], o["system"], o["entity"], o["ts"], o["summary"][:200])
    for e in o["evidence"][:3]:
        print("    ", e[:300])
if len(sys.argv) > 3:
    json.dump(out, open(sys.argv[3], "w"), ensure_ascii=False, indent=1, default=str)
