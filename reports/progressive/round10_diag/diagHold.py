"""diagHold.py (round 10: + P04 HoldRecord (T, F, TF, c) of every ptree node of <sys> whose p_hold < 0.3; full statements with confidence < 0.3, stops after day_hi; + ptree node counts of every system in NC, statements only for <sys>) <codedir> <pack> <seed> <out.json> <sys,...> <day_lo> <day_hi>: run (no scoring) and dump, at
every local 23:45 tick of days [lo, hi], the P14 statements of the systems (pattern_id, route, state, conf, who),
the ptree nodes (id, route-ish split, who level / items, state) and model.who_groups 'readdressed'."""
import json, os, sys, datetime
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
code, pack_name, seed, out, SYS, DLO, DHI = sys.argv[1:8]
SYS = SYS.split(","); DLO, DHI = int(DLO), int(DHI)
NC = ["portal", "oa", "finance", "crm", "mail"]
sys.path.insert(0, os.path.join(code, "backend"))
from app.eval.packs import get_pack
from app.eval.runner import run_pack
from app.engines.behavior import views as VW
from app.engines.behavior.lib import m_ptree as MP
SNAP = {}; T0 = {}
_run = VW.ViewsEngine.run
def _r(self, ctx, observations=None):
    n = _run(self, ctx, observations)
    now = float(ctx.now)
    T0.setdefault("t0", now)
    loc = datetime.datetime.utcfromtimestamp(now) + datetime.timedelta(hours=8)
    d = (loc.date() - (datetime.datetime.utcfromtimestamp(T0["t0"]) + datetime.timedelta(hours=8)).date()).days + 1
    if DLO <= d <= DHI and loc.hour == 23 and loc.minute == 45:
      try:
        day = {}
        st = ctx.store
        keys = list(SYS)
        sf = MP.get_org_model(st, MP.SYSFAM) or {}
        for s in SYS:
            k = (sf.get("member") or {}).get(s) if isinstance(sf, dict) else None
            if k and k not in keys:
                keys.append(k)
        for k in keys:
            k = str(k)
            pv = MP.get_model(st, k, MP.PVIEWS) or {}
            day[k] = {"statements": [s_ for s_ in (pv.get("statements") or []) if float(s_.get("confidence") or 0) < 0.3],
                      "attrsel_targets": (MP.get_model(st, k, "model.attrsel") or {}).get("targets_sys"),
                      "pbind_stated": (MP.get_model(st, k, "model.pbind") or {}).get("bound_stated")}
        from app.eval.pmetrics import ptree_node_count
        for k in SYS:
            tr = MP.get_model(st, k, MP.PTREE)
            hd = {}
            for nid, nd in sorted(((f"{kk}:{ni}", nn) for kk, tt in sorted((getattr(tr, "kinds", None) or {}).items()) for ni, nn in tt.nodes.items()), key=lambda x: x[0]):
                try:
                    ph = nd.p_hold(now)
                except Exception:
                    ph = None
                try:
                    hw = VW.hold_without_when(nd)
                except Exception:
                    hw = None
                if (ph is None or ph >= 0.3) and (hw is None or hw >= 0.3):
                    continue
                hr = nd.hold
                hd[str(nid)] = {"p_hold": ph, "hold_wo_when": hw, "facet_tests": {a: list(b) for a, b in (hr.facet_tests() or {}).items()} if hr else None, "state": getattr(nd, "state", None), "route": str(getattr(nd, "split", None))[:200],
                                "T": list(hr.T) if hr else None, "F": dict(hr.F) if hr else None,
                                "TF": {a: list(b) for a, b in (hr.TF or {}).items()} if hr else None,
                                "c": {a: [round(x, 3) for x in b] for a, b in (hr.c or {}).items()} if hr else None,
                                "seg": hr.seg if hr else None,
                                "meta": {a: b for a, b in (nd.meta or {}).items() if "hold" in str(a)}}
            day.setdefault("hold", {})[k] = hd
        nc = {}
        for k in sorted(set(NC) | set(keys)):
            try:
                _tr = MP.get_model(st, k, MP.PTREE)
                nc[k] = None if _tr is None else sum(len(_t.nodes) for _t in (_tr.kinds or {}).values())
            except Exception as exc:
                nc[k] = repr(exc)
        day["node_count"] = nc
        wg = MP.get_org_model(st, MP.WHO_GROUPS) or {}
        day["who_groups"] = {"readdressed": wg.get("readdressed"),
                             "groups": {g: v.get("members") if isinstance(v, dict) else v for g, v in sorted((wg.get("groups") or {}).items())}}
        SNAP[f"d{d}"] = day
      except Exception as exc:
        SNAP[f"d{d}"] = {"error": repr(exc)}
      json.dump({"snap": SNAP, "partial": True}, open(out + ".part", "w"), default=str, ensure_ascii=False)
      if d >= DHI:
          json.dump({"snap": SNAP}, open(out + ".tmp", "w"), default=str, ensure_ascii=False)
          os.replace(out + ".tmp", out); print("done", len(SNAP), flush=True); os._exit(0)
    return n
VW.ViewsEngine.run = _r
pack = get_pack(pack_name)
pack.registry_mode = "progressive_decision"
res = run_pack(pack, int(seed), strict=True, record_series=False)
json.dump({"snap": SNAP}, open(out + ".tmp", "w"), default=str, ensure_ascii=False)
os.replace(out + ".tmp", out)
print("done", len(SNAP))
