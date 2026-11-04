import json, numpy as np
R9="/tmp/claude-0/-home-user-appmonitor-core/e496f878-8276-5793-a91f-1f6d0128e252/scratchpad/prog11/ev11/out/main/runs"; R8="/home/user/appmonitor-core/reports/progressive/round10/runs"
def load(d): return [json.load(open(f"{d}/O_{s}.json")) for s in range(5)]
a9,a8=load(R9),load(R8)
P=[("P00","raw.event"),("P01","derived.event_context"),("P02","behavior.attr_registry"),("P03","behavior.conformity"),("P04","behavior.pattern_tree"),("P05","behavior.attr_select"),("P06","behavior.content_bounds"),("P07","behavior.payload_grammar"),("P08","behavior.binding"),("P09","behavior.time_window"),("P10","behavior.workflow"),("P11","behavior.who_groups"),("P12","behavior.system_profile"),("P13","behavior.facets"),("P14","behavior.views"),("P15","behavior.resource_governor")]
REF=["behavior.calibration","behavior.fusion","behavior.risk","behavior.incident"]
med=lambda a,k: float(np.median([r["engine_s"].get(k,0) for r in a]))
ref9=float(np.median([sum(r["engine_s"].get(k,0) for k in REF) for r in a9])); ref8=float(np.median([sum(r["engine_s"].get(k,0) for k in REF) for r in a8]))
L=ref9/ref8
print(f"load factor (B24-B27 r11/r10) {L:.3f}")
print("| Engine | round 10 s | round 11 s | ratio | ratio / load factor |\n|---|---|---|---|---|")
t8=t9=0
for p,k in P:
    x8,x9=med(a8,k),med(a9,k); t8+=x8; t9+=x9
    print(f"| {p} `{k}` | {x8:.0f} | {x9:.0f} | {x9/x8:.2f} | {x9/x8/L:.2f} |")
c8=float(np.median([sum(r['engine_s'].get(k,0) for _,k in P) for r in a8])); c9=float(np.median([sum(r['engine_s'].get(k,0) for _,k in P) for r in a9]))
print(f"| P-core total | {c8:.0f} | {c9:.0f} | {c9/c8:.2f} | {c9/c8/L:.2f} |")
print(f"| B24–B27 (unchanged since round 9; load reference) | {ref8:.0f} | {ref9:.0f} | {L:.2f} | 1.00 |")
w8=np.median([r['wall_s'] for r in a8]); w9=np.median([r['wall_s'] for r in a9]); m8=np.median([r['peak_rss_mb'] for r in a8]); m9=np.median([r['peak_rss_mb'] for r in a9])
print(f"| wall s / peak RSS MB | {w8:.0f} / {m8:.0f} | {w9:.0f} / {m9:.0f} | | |")
