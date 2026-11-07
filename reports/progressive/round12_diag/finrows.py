"""finrows.py run.json...: days 11-35 on which FIN.finance.* rows (post-move @1 and bookkeeping) are recovered."""
import json, sys
for fn in sys.argv[1:]:
    pg1 = json.load(open(fn))["score"]["pg1"]
    ks = sorted({k for d in pg1.values() if isinstance(d, dict) for k in (d.get("per_pattern") or {}) if k.startswith("FIN.finance.login") or "@1" in k and k.startswith("FIN")})
    print(fn)
    for k in ks:
        days = [int(d) for d, v in pg1.items() if d.isdigit() and k in (v.get("per_pattern") or {}) and v["per_pattern"][k]["recovered"]]
        valid = [int(d) for d, v in pg1.items() if d.isdigit() and k in (v.get("per_pattern") or {})]
        print(f"  {k}: found {min(days) if days else '-'}..{max(days) if days else '-'} on {len(days)} of {len(valid)} valid days; missing {sorted(set(valid)-set(days))[:12]}")
