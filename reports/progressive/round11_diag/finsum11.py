import json
E="/tmp/claude-0/-home-user-appmonitor-core/e496f878-8276-5793-a91f-1f6d0128e252/scratchpad/prog11/ev11/out/main/runs"
RS=[("r8","reports/progressive/round8/runs"),("r9","reports/progressive/round9/runs"),("r10","reports/progressive/round10/runs"),("r11",E)]
rows=["FIN.finance.bookkeeping#0@1","FIN.finance.bookkeeping#1@1","FIN.finance.bookkeeping#2@1","FIN.finance.login#0@1","FIN.finance.login#1@1"]
for row in rows+["FIN.oa.documents@1","FIN.oa.login@1","GA.oa.documents","GA.oa.approvals#2@1"]:
    cells=[]
    for s in range(3):
        c=[]
        for tag,d in RS:
            pg1=json.load(open(f"{d}/O-real_{s}.json"))["score"]["pg1"]
            days=[int(k) for k in pg1 if (pg1[k].get("per_pattern") or {})]
            if row in rows or row=="GA.oa.approvals#2@1":
                rec=[dd for dd in sorted(days) if (pg1[str(dd)]["per_pattern"].get(row) or {}).get("recovered")]
                c.append(f"d{rec[0]}, {len(rec)}{' (d35 ✓)' if 35 in rec else ''}" if rec else "never")
            else:
                n=sum(1 for dd in days for k,v in pg1[str(dd)]["per_pattern"].items() if k.startswith(row.split("@")[0]) and (("@1" in k) == ("@1" in row)) and v.get("recovered"))
                c.append(str(n))
        cells.append(" / ".join(c))
    print(f"| {row} | " + " | ".join(cells) + " |")
