import json, os, sys
sys.path.insert(0, "/home/user/appmonitor-core/backend")
from app.eval.pscale import pg4_summary
D12 = "/tmp/claude-0/-home-user-appmonitor-core/e496f878-8276-5793-a91f-1f6d0128e252/scratchpad/prog12/ev12/out/scale"
D11 = "/home/user/appmonitor-core/reports/progressive/scale_r11"
D10 = "/home/user/appmonitor-core/reports/progressive/scale_r10"
D9 = "/home/user/appmonitor-core/reports/progressive/scale_r9"
D8 = "/home/user/appmonitor-core/reports/progressive/scale_r8"
pts = [("ips", 500, "500 IPs, 40 attributes"), ("ips", 20000, "20000 IPs, 40 attributes"), ("attrs", 300, "500 IPs, 340 attributes"),
       ("servers", 20, "20 servers"), ("servers", 100, "100 servers"), ("servers", 300, "300 servers")]
def ld(d, k, n):
    f = os.path.join(d, f"scale_{k}_{n}.json")
    return json.load(open(f)) if os.path.exists(f) else None
print("| Point (7 days) | memory MB r10 → r11 → r12 | P-core µs/event (thread CPU) r10 → r11 → r12 | scoring p95 µs r10 → r11 → r12 | learning p95 µs r10 → r11 → r12 | wall s r10 → r11 → r12 |")
print("|---|---|---|---|---|---|")
def cpu(p, key):
    return p["cpu"].get(key)
for k, n, lab in pts:
    a, b, cc = ld(D10, k, n), ld(D11, k, n), ld(D12, k, n)
    def f(p, fn):
        if p is None: return "–"
        try: return fn(p)
        except Exception as e: return "?"
    cols = [lambda p: f"{p['mem_bytes']/1048576:.1f}", lambda p: f"{p['cpu']['us_per_event_cpu']:.0f}",
            lambda p: f"{p['cpu']['scoring_p95_us_cpu']:.0f}", lambda p: f"{p['cpu']['learning_p95_us_cpu']:.0f}",
            lambda p: f"{p['wall_s']:.0f}"]
    print(f"| {lab} | " + " | ".join(f"{f(a, c)} → {f(b, c)} → {f(cc, c)}" for c in cols) + " |")
for tag, d in (("r9", D9), ("r10", D10), ("r11", D11), ("r12", D12)):
    ps = [p for p in (ld(d, k, n) for k, n, _ in pts) if p]
    s = pg4_summary(ps)
    print(tag, json.dumps({kk: v for kk, v in s.items() if not isinstance(v, (list, dict))}, default=str))
    for c in (s.get("details") or {}).get("checks", []):
        print("  ", c.get("name"), c.get("value"), c.get("pass"))
