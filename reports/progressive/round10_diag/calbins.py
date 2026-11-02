"""calbins.py run.json [run2.json ...]: per-bin breakdown of the pooled debiased CE (days >= 7, pg1 'calib' rows [conf, held, ...])."""
import json, sys, numpy as np
for fn in sys.argv[1:]:
    r = json.load(open(fn)); pg1 = r["score"]["pg1"]
    rows = [x for d, p in pg1.items() if d.isdigit() and int(d) >= 7 for x in (p.get("calib") or [])]
    c = np.array([x[0] for x in rows], float); o = np.array([x[1] for x in rows], float)
    ok = np.isfinite(c) & np.isfinite(o); c, o = c[ok], o[ok]
    idx = np.minimum((c * 10).astype(int), 9); tot = 0
    print(f"== {fn[-60:]}  n={c.size} conf={c.mean():.3f} hold={o.mean():.3f}")
    for b in range(10):
        m = idx == b; nb = int(m.sum())
        if not nb: continue
        ob, cb = o[m].mean(), c[m].mean(); var = ob*(1-ob)/(nb-1) if nb >= 2 else cb*(1-cb)
        t = nb / c.size * ((ob - cb) ** 2 - var); tot += t
        print(f"  bin {b}: n={nb:4d} c={cb:.3f} o={ob:.3f} contrib={t:+.5f}")
    print(f"  CE={np.sqrt(max(0, tot)):.4f}")
