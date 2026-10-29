import os, pickle, sys, json
code, fn = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(code, "backend"))
res = pickle.load(open(fn, "rb")); ps = res.psnaps
for day in sorted(ps):
    d = ps[day]["systems"].get("portal") or {}
    if "model.ptree" not in d: continue
    pt = d["model.ptree"]["kinds"]; pb = (d.get("model.pbounds") or {}).get("nodes") or {}
    out = []
    for k, kv in sorted(pt.items()):
        for nid, nd in sorted(kv["nodes"].items(), key=lambda x: int(x[0])):
            r = [c for c in nd.get("ctx") or [] if c[0] == "http.route"]
            if not r or not any("GET www.corp.example /news/{num}" == v for v in r[-1][2]): continue
            tg = nd.get("targets") or []
            rec = ((pb.get(k) or {}).get(str(nid)) or {})
            rec = rec.get("attrs", rec) if isinstance(rec, dict) else {}
            bu = rec.get("net.bytes_up") if isinstance(rec, dict) else None
            out.append(f"{k}:{nid}[{nd.get('state')[:4]}] bu_t={'net.bytes_up' in tg} nz={sum(1 for t in tg if t.startswith('meta.z'))}/{len(tg)} " +
                       (json.dumps({x: bu.get(x) for x in ('band98', 'range', 'n_rng', 'excluded_days')}, default=str) if isinstance(bu, dict) else "-"))
    print(day, " | ".join(out))
