"""Variant: prospective held-out window skipping truth-calendar holidays (extended to 7 non-holiday days).
usage: rescore_hol.py RES_PKL OUT_JSON"""
import json, os, pickle, sys, datetime as _dt
sys.path.insert(0, "/home/user/appmonitor-core/backend")
from app.eval import pmetrics as M
from app.eval.report import _clean
orig = M.PTruth.traffic_after


def traffic_after(self, day):
    cache = self.__dict__.setdefault("_traffic_after_h", {})
    if day in cache:
        return cache[day]
    hol = set()
    for iso in (self.raw.get("days") or {}).get("holidays") or ():
        hol.add(self.day_index(iso))
    back = None
    out = {}
    for row in self.valid_at_day(day):
        tid = row["tid"]
        vf, vt = int(row["valid_from_day"]), int(row["valid_to_day"])
        per_day = {self.day_index(iso): per for iso, per in (self.opp.get(tid) or {}).items()}
        per_dt = {}
        x, used = int(day), 0
        while used < self.PROSPECTIVE_DAYS and x < int(day) + 60:
            x += 1
            if x >= vt and vt <= self.n_days:
                break
            y = self.proxy_day(x)
            if y in hol:
                continue
            used += 1
            if y < max(1, vf) or y - 1 >= len(self.workday):
                continue
            per = per_day.get(y)
            if not per:
                continue
            dt = "workday" if self.workday[y - 1] else "nonworkday"
            acc = per_dt.setdefault(dt, {})
            for ip in sorted(per):
                acc[ip] = acc.get(ip, 0.0) + float(per[ip])
        if not any(sum(v.values()) > 0 for v in per_dt.values()):
            if back is None:
                back = self.traffic(day)
            per_dt = back.get(tid) or {}
        out[tid] = per_dt
    cache[day] = out
    return out


M.PTruth.traffic_after = traffic_after
res = pickle.load(open(sys.argv[1], "rb"))
pt = M.PTruth(res.ptruth)
print("holidays", sorted(pt.day_index(i) for i in pt.raw["days"]["holidays"]))
sc = M.score_prun(res)
json.dump(_clean(sc), open(sys.argv[2], "w"), default=str, ensure_ascii=False)
print(json.dumps(sc["pg2"]["calibration"], default=str))
