"""ex9.py <run.json> <day> : the user's example in both views (P14 text verbatim) + checklist."""
import json, sys
r = json.load(open(sys.argv[1])); day = sys.argv[2]
ex = r["example"]; d = ex["days"].get(day) or {}
print("## checklist")
for c in ex.get("checklist") or []:
    print(f"- {'✓' if c.get('pass') else '✗'} {c['clause']} | {c.get('measured')}")
WANT = ("/login", "/approve", "/report/generate", "/flow/list", "/approval/list")
print("## system view oa")
for s in d.get("oa_statements") or []:
    print(f"- [{s['pattern_id']}] {s['state']} {s['confidence']:.2f} who={s['who'].get('members')}")
    if any(w in s["route"] for w in WANT):
        print("  > " + s.get("text_zh", ""))
print("## system view finance")
for s in d.get("finance_statements") or []:
    print(f"- [{s['pattern_id']}] {s['state']} {s['confidence']:.2f} {s['route']} who={s['who'].get('members')}")
    if "/approve" in s["route"]:
        print("  > " + s.get("text_zh", ""))
print("## dept views")
for v in d.get("dept_views") or []:
    print(f"- {v['view']} groups={v.get('groups')} :: {v['header_zh']} ({len(v['statements'])} statements)")
    for st in v["statements"][:8]:
        print("  - " + ("(neg) " if st.get("negative") else "") + st["text_zh"][:400])
print("## group views")
for v in d.get("group_views") or []:
    print(f"- {v['group']} :: {v['header_zh']} ({len(v['statements'])} statements)")
    for st in v["statements"]:
        if any(w in st["text_zh"] for w in ("/login", "approve")):
            print("  - " + st["text_zh"][:500])
print("ga_groups", d.get("ga_groups"))
