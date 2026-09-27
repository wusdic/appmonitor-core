"""B30 PortraitEngine (docs/lib3/engines.md '## B30'): spec unit test plus edges.

The world: synthetic owner models built in their real layouts (m_rhythm
counts, m_baseline anchors folded with make_row / commit_many, m_vocab /
m_client entity dicts, a model.class assignment) for one IP whose rhythm is
active 09-18 on workdays, whose top template is /orders at 40 % and whose
client stack is chrome/126; plus a static CIDR class of 3 IPs.

Spec test mapping:
  window '09:00-18:00' +- 30 min, top template /orders -> test_spec_window_template_stack
  window 14-23 -> diff reports 活跃时段                  -> test_spec_window_change_diff
  static CIDR class of 3 IPs: members and class bands  -> test_spec_static_class_portrait
"""
from __future__ import annotations

import json
import time
from typing import Dict, Iterable, Optional

import numpy as np

from helpers import DT, T0, make_store, make_tctx, run_engine

from app.engines.behavior import portrait as P
from app.engines.behavior.lib import features as F
from app.engines.behavior.lib import m_baseline as MB
from app.engines.behavior.lib import m_client as MC
from app.engines.behavior.lib import m_rhythm as MR
from app.engines.behavior.lib import m_vocab as MV
from app.engines.behavior.lib.classkeys import ORG
from app.engines.behavior.portrait import PortraitEngine
from app.models.schema import BehaviorEvent, EntityProfile, Severity

S = "erp"
IP = "10.20.1.12"
DAYS = 8
NOW = T0 + DAYS * 86400.0
TWO_H = 7200.0
CHROME = "0123456789abcdef0123456789abcdef|chrome/126|win|128|w16"
TEMPLATES = {"GET erp.corp /orders": 40.0, "GET erp.corp /home": 25.0,
             "POST erp.corp /orders/new": 15.0, "GET erp.corp /reports/{num}": 12.0,
             "GET erp.corp /static/app.js": 8.0}


# ------------------------------------------------------------------ builders
def rhythm_model(h0: int, h1: int, now: float = NOW) -> Dict:
    """Workday slots of hours [h0, h1) always active, everything else silent."""
    m = MR.new_model("entity")
    st = m["state"]
    for c in range(MR.N48):
        b48 = c // 4
        wd = b48 < 24
        st["N48"][c] = 30.0 if wd else 12.0
        hr = b48 % 24
        on = wd and (h0 <= hr < h1 if h0 < h1 else (hr >= h0 or hr < h1))
        st["A48"][c] = st["N48"][c] if on else 0.0
    st.update(t_ref=now, t_first=now - DAYS * 86400.0, t_last=now,
              n_slots=int(st["N48"].sum()))
    return m


def nat_row(level: float, rng: np.random.Generator, dt: float = DT) -> np.ndarray:
    nat = np.full(F.FEATURE_DIM, np.nan)
    base = {"http_requests": 10.0, "flows": 20.0, "bytes_up": 5e4, "bytes_down": 2e5}
    for n, v in base.items():
        nat[F.FEATURE_INDEX[n]] = float(rng.poisson(v * level * dt / 900.0))
    return nat


def baseline_model(hours: Iterable[int], level: float = 1.0, seed: int = 1,
                   days: int = DAYS) -> Dict:
    hours = set(hours)
    rng = np.random.default_rng(seed)
    m = MB.new_model()
    for k in range(int(days * 86400 / DT)):
        t = T0 + k * DT
        tc = make_tctx(t)
        if tc["day_type"] == "workday" and int(tc["hour_local"]) in hours:
            MB.commit_many([m["current"]], [MB.make_row(t, nat_row(level, rng), DT, tc)],
                           [1.0], [MB.CAP_CURRENT], [0.0])
    m.update(fmt=MB.FMT, tier="entity", version=0, branch=0)
    return m


def vocab_model(tmpl: Dict[str, float], now: float = NOW) -> Dict:
    dims = {"tmpl": tmpl, "sni": {"corp.example": 30.0}, "dport": {"443": 50.0, "80": 5.0}}
    return {"fmt": 1, "kind": "entity", "version": 0, "ts": now, "class_key": None,
            "state": {"H": MV.HALF_LIFE_S, "t_ref": now, "g": 0.0,
                      "dims": {d: {v: [c, max(2, int(c)), T0, now] for v, c in vals.items()}
                               for d, vals in dims.items()},
                      "N": {d: float(sum(v.values())) for d, v in dims.items()},
                      "N1": {d: 0.0 for d in dims}, "clock": now, "first": T0,
                      "days": {}, "jsd": {}, "n_rows": 100}}


def client_model(tok: str = CHROME, now: float = NOW) -> Dict:
    return {"fmt": 1, "kind": "entity", "version": 0, "ts": now, "class_key": None,
            "state": {"H": MC.HALF_LIFE_S, "clock": now, "c": {tok: [50.0, 200, T0, now, 0]},
                      "N": 50.0, "gaps": {}, "n_rows": 200}}


_BASE_CACHE: Dict = {}


def cached_baseline(h0: int, h1: int) -> Dict:
    """Baseline anchors are the slow part to build; tests only read them."""
    key = (h0, h1)
    if key not in _BASE_CACHE:
        _BASE_CACHE[key] = baseline_model(range(h0, h1))
    return _BASE_CACHE[key]


def world(ip: str = IP, h0: int = 9, h1: int = 18, store=None, baseline: bool = True):
    store = store or make_store()
    store.put_model(S, ip, MR.MODEL, rhythm_model(h0, h1))
    if baseline:
        store.put_model(S, ip, MB.MODEL, cached_baseline(h0, h1))
    store.put_model(S, ip, MV.MODEL, vocab_model(TEMPLATES))
    store.put_model(S, ip, MC.MODEL, client_model())
    store.register_entity(S, ip)
    return store


def class_model(store, assign: Dict[str, Dict], roles: Optional[Dict] = None) -> None:
    store.put_model(ORG[0], ORG[1], "model.class",
                    {"assign": {f"{S}|{ip}": a for ip, a in assign.items()},
                     "roles": roles or {}, "subs": {}, "statics": {}, "pools": {}, "version": 1})


def portrait(store, key: str = IP) -> Dict:
    prof = store.profile(S, key)
    assert prof is not None and "portrait" in prof.extra
    return prof.extra["portrait"]


# ----------------------------------------------------------------- spec tests
def test_spec_window_template_stack():
    store = world()
    n = run_engine(PortraitEngine(), store, NOW)
    assert n == 1
    por = portrait(store)
    js = por["json"]
    wd = js["rhythm"]["workday"]
    assert abs(wd["start_h"] - 9.0) <= 0.5 and abs(wd["start_h"] + wd["len_h"] - 18.0) <= 0.5
    assert js["active_hours"] == "09:00–18:00"
    top = js["top_templates"][0]
    assert top["template"].endswith("/orders") and abs(top["share"] - 0.4) < 1e-6
    dom = js["client"]["dominant"][0]
    assert dom["ua"] == "chrome/126" and dom["os"] == "win" and dom["share"] == 1.0
    # workload in natural units: ~10 requests per 15 min on an active workday tick
    wl = js["workload"]["http_requests"]["workday"]
    assert 3 <= wl["p10"] <= wl["p50"] <= wl["p90"] <= 20 and 8 <= wl["p50"] <= 12
    # never active on non-workdays: no band (the hyperprior is not a measurement)
    assert js["workload"]["http_requests"]["nonworkday"] is None
    bu = js["workload"]["bytes_up"]["workday"]                 # t family on log1p(rate)
    assert 3.5e4 <= bu["p50"] <= 6.5e4 and bu["p10"] < bu["p50"] < bu["p90"]
    assert por["version"] == 1 and por["diff"] == []
    assert "09:00–18:00" in por["text_zh"] and "/orders" in por["text_zh"]
    assert "chrome/126" in por["text_en"] and "active on workdays 09:00–18:00" in por["text_en"]
    json.dumps(por["json"], allow_nan=False)          # JSON-safe
    pv = store.profile_versions(S, IP)
    assert len(pv) == 1 and pv[0].version == 1 and pv[0].obj["kind"] == "portrait"


def test_spec_window_change_diff():
    store = world()
    eng = PortraitEngine()
    run_engine(eng, store, NOW)
    store.put_model(S, IP, MR.MODEL, rhythm_model(14, 23))
    run_engine(eng, store, NOW + TWO_H)
    por = portrait(store)
    assert por["version"] == 2
    labels = [d["label_zh"] for d in por["diff"]]
    assert "活跃时段" in labels
    d = next(x for x in por["diff"] if x["label_zh"] == "活跃时段")
    assert d["before"] == "09:00–18:00" and d["after"] == "14:00–23:00" and d["shift_h"] >= 4
    assert "活跃时段" in por["text_zh"] and "active window" in por["text_en"]
    assert por["json"]["active_hours"] == "14:00–23:00"
    assert [v.version for v in store.profile_versions(S, IP)] == [2, 1]


def test_spec_static_class_portrait():
    store = make_store()
    ips = ["10.30.0.1", "10.30.0.2", "10.30.0.3"]
    for i, ip in enumerate(ips):
        world(ip, store=store, baseline=False)
    class_model(store, {ip: {"role": None, "prob": 1.0, "static": ["office"], "pool": None,
                             "super": "human", "class_path": "human/none"} for ip in ips})
    ck = "class:static:office"
    cm = {"kind": "static", "members": ips, "n_members": 3, "bucket": 10,
          "aggregate": {"quantiles": [0.05, 0.5, 0.95], "exposure_s": 900.0,
                        "features": {"http_requests": [12.0, 30.0, 55.0],
                                     "flows": [30.0, 60.0, 95.0]}},
          "active_frac_by_bin": [1.0 if 9 <= h < 18 else 0.0 for h in range(24)] + [0.0] * 24,
          "adoption": {"rate": 0.1, "n_records": 2, "recent": []}, "n_eff": 120.0}
    store.put_profile(EntityProfile(system=S, entity=ck, updated=NOW,
                                    extra={"class_monitor": cm,
                                           "risk": {"score": 0.1, "tier": "low"}}))
    store.put_model(S, ck, "model.classagg", {"version": 3, "current": cached_baseline(9, 18)[
        "current"]})
    n = run_engine(PortraitEngine(), store, NOW)
    assert n == 4                                             # 3 IPs + the class
    js = portrait(store, ck)["json"]
    assert js["kind"] == "class" and js["class_kind"] == "static" and js["name"] == "office"
    assert [m["ip"] for m in js["members"]] == ips and all(m["prob"] == 1.0 for m in js["members"])
    assert js["n_members"] == 3
    b = js["bands"]["features"]["http_requests"]
    assert (b["p5"], b["p50"], b["p95"]) == (12.0, 30.0, 55.0)
    assert js["active_frac_heatmap"]["workday"][10] == 1.0
    assert js["cohesion"] is not None and js["cohesion"] > 0.99     # identical members
    assert js["outlier_members"] == []
    assert {c["template"] for c in js["common_tokens"]} >= {"GET erp.corp /orders"}
    assert js["workload"]["http_requests"]["workday"]["p50"] > 0
    assert js["rhythm"]["workday"]["window"] == "09:00–18:00"         # from the heatmap
    assert js["risk"]["tier"] == "low"
    txt = portrait(store, ck)["text_zh"]
    assert txt.startswith("office（静态网段，3个成员") and "风险低" in txt


# ------------------------------------------------------------------- edges
def test_empty_store_and_silent_entity():
    store = make_store()
    assert run_engine(PortraitEngine(), store, NOW) == 0
    store.register_entity(S, "10.9.9.9")                      # no model, never seen
    assert run_engine(PortraitEngine(), store, NOW) == 1
    por = portrait(store, "10.9.9.9")
    js = por["json"]
    assert js["status"] == "no_model" and js["workload"] == {} and js["rhythm"] == {}
    assert js["active_hours"] is None and js["top_templates"] == []
    assert por["text_zh"].startswith("10.9.9.9：尚无行为模型")
    json.dumps(js, allow_nan=False)


def test_nan_inputs_are_none_never_defaults():
    store = world()
    ident = {"fmt": 1, "version": 2, "stats": {IP: {"separability": float("nan"),
                                                    "recall1": 0.9, "n_windows": 40,
                                                    "confusable_with": ["10.20.1.11"]}}}
    store.put_model(S, "__system__", "model.identity", ident)
    rm = MR.new_model("entity")                              # a rhythm with no data at all
    store.put_model(S, IP, MR.MODEL, rm)
    store.put_profile(EntityProfile(system=S, entity=IP, extra={
        "risk": {"score": float("nan"), "tier": None, "degraded": True}}))
    run_engine(PortraitEngine(), store, NOW)
    js = portrait(store)["json"]
    idn = js["identity"]
    assert idn["identifiability"] is None and idn["identifiability_ci"] is None
    assert idn["recall1"] == 0.9 and idn["confusable_with"] == ["10.20.1.11"]
    assert js["rhythm"].get("workday") is None and js["active_hours"] is None
    assert js["risk"]["score"] is None and js["risk"]["degraded"] is True
    json.dumps(js, allow_nan=False)


def test_identifiability_wilson_ci_and_confusable_text():
    store = world()
    class_model(store, {IP: {"role": "r1", "prob": 0.92, "static": [], "pool": None,
                             "super": "human", "class_path": "human/r1"},
                        **{f"10.20.1.{i}": {"role": "r1", "prob": 0.8, "static": [],
                                            "pool": None, "super": "human"} for i in (13, 14, 15)}},
                roles={"r1": {"name": "erp-prod", "members": []}})
    ident = {"fmt": 1, "version": 2, "stats": {IP: {"separability": 0.94, "recall1": 0.95,
                                                    "n_windows": 50,
                                                    "confusable_with": ["10.20.1.11"]}},
             "anonymity_sets": [[IP, "10.20.1.11"]]}
    store.put_model(S, "__system__", "model.identity", ident)
    run_engine(PortraitEngine(), store, NOW)
    por = portrait(store)
    idn = por["json"]["identity"]
    lo, hi = idn["identifiability_ci"]
    assert lo < 0.94 < hi and 0.8 < lo and hi <= 1.0
    assert idn["anonymity_set"] == [IP, "10.20.1.11"] and idn["class_size"] == 4
    assert por["text_zh"].startswith(
        "10.20.1.12：erp-prod 人工交互用户(同类4个,置信0.92)，工作日09:00–18:00活跃")
    assert "可辨识度0.94(易混淆:10.20.1.11)" in por["text_zh"]


def test_training_mode_emits_no_events_but_learns():
    store = world()
    run_engine(PortraitEngine(), store, NOW, training=True)
    assert store.events() == []
    assert portrait(store)["version"] == 1


def test_cadence_switch_900_to_60():
    store = world()
    eng = PortraitEngine()
    run_engine(eng, store, NOW, dt=900.0)
    b900 = portrait(store)["json"]["workload"]["http_requests"]
    run_engine(eng, store, NOW + TWO_H, dt=60.0)
    por = portrait(store)
    b60 = por["json"]["workload"]["http_requests"]
    assert b900["exposure_s"] == 900.0 and b60["exposure_s"] == 60.0
    assert b60["p95"] < b900["p95"]                          # per-tick band shrinks with dt
    assert b60["workday"] == b900["workday"]                 # 15-min bands are cadence-free
    assert por["version"] == 1 and por["diff"] == []         # no spurious version
    # the per-tick band counts inactive ticks (absence is data): p5 is 0
    assert b900["p5"] == 0.0 and b900["includes_inactive"] is True


def test_refresh_is_per_key_and_scheduled():
    store = world()
    eng = PortraitEngine()
    assert run_engine(eng, store, NOW) == 1
    assert run_engine(eng, store, NOW + 900.0) == 0          # refreshed < 2 h ago
    # safe_run honours interval 8 / period 2 h
    eng2 = PortraitEngine()
    runs = [run_engine(eng2, store, NOW + k * 900.0, scheduled=True) for k in range(9)]
    assert runs[0] == 1 and sum(runs[1:8]) == 0 and eng2.runs == 2


def test_template_mix_change_and_workload_shift_are_versioned():
    store = world()
    eng = PortraitEngine()
    run_engine(eng, store, NOW)
    # a tiny change: no new version
    t2 = dict(TEMPLATES, **{"GET erp.corp /orders": 41.0})
    store.put_model(S, IP, MV.MODEL, vocab_model(t2))
    run_engine(eng, store, NOW + TWO_H)
    assert portrait(store)["version"] == 1
    # privilege misuse at unchanged volume: the diff lists the new paths
    t3 = dict(TEMPLATES, **{"GET erp.corp /admin/users/{num}": 150.0,
                            "GET erp.corp /admin/export": 60.0})
    store.put_model(S, IP, MV.MODEL, vocab_model(t3))
    run_engine(eng, store, NOW + 2 * TWO_H)
    por = portrait(store)
    assert por["version"] == 2
    d = next(x for x in por["diff"] if x["field"] == "top_templates")
    assert d["jsd"] > 0.1 and "GET erp.corp /admin/users/{num}" in d["added"]
    assert "常用模板" in por["text_zh"]
    # workload x3: quantile shift > 25 %
    store.put_model(S, IP, MB.MODEL, baseline_model(range(9, 18), level=3.0, seed=2))
    run_engine(eng, store, NOW + 3 * TWO_H)
    por = portrait(store)
    assert por["version"] == 3
    q = [x for x in por["diff"] if x["field"] == "workload"]
    assert any(x["feature"] == "http_requests" and x["day_type"] == "workday" for x in q)


def test_no_path_or_query_values_in_portrait():
    store = world()
    raw = {"GET erp.corp /orders/view/987654321?id=4242&token=s3cr3tvalue": 50.0,
           "GET erp.corp /u/john.doe@example.com/profile": 30.0,
           "GET erp.corp /orders": 20.0}
    store.put_model(S, IP, MV.MODEL, vocab_model(raw))
    run_engine(PortraitEngine(), store, NOW)
    por = portrait(store)
    blob = json.dumps(por, ensure_ascii=False)
    for secret in ("987654321", "4242", "s3cr3tvalue", "john.doe"):
        assert secret not in blob
    tops = [t["template"] for t in por["json"]["top_templates"]]
    assert "GET erp.corp /orders/view/{num}?id&token" in tops
    assert "GET erp.corp /u/{email}/profile" in tops


def test_timeline_last_five_and_risk_reasons():
    store = world()
    for k in range(7):
        store.add_event(BehaviorEvent(system=S, entity=IP, ts=NOW - 3600.0 * (k + 1),
                                      kind="rare_access", score=0.5, severity=Severity.LOW,
                                      description="GET erp.corp /orders/view/99887766"))
    store.put_profile(EntityProfile(system=S, entity=IP, extra={"risk": {
        "score": 0.62, "tier": "medium", "trend": 0.1, "stages": ["behavior"],
        "top_reasons": [{"key": "det:novelty", "source": "det", "name": "novelty",
                         "L": 1.2, "share": 0.7, "detail": "/orders/view/99887766"}]}}))
    run_engine(PortraitEngine(), store, NOW)
    js = portrait(store)["json"]
    assert len(js["timeline"]) == 5 and js["timeline"][0]["kind"] == "rare_access"
    assert js["timeline"][0]["severity"] == "low"
    assert js["risk"]["tier"] == "medium" and js["risk"]["top_reasons"][0]["name"] == "novelty"
    blob = json.dumps(portrait(store), ensure_ascii=False)
    assert "99887766" not in blob                              # descriptions never copied
    assert "风险中(0.62)" in portrait(store)["text_zh"]


def test_role_class_outlier_and_distinctive_tokens():
    store = make_store()
    ips = [f"10.40.0.{i}" for i in range(1, 6)]
    for ip in ips[:4]:
        world(ip, store=store, baseline=False)
    world(ips[4], store=store, baseline=False)
    store.put_model(S, ips[4], MV.MODEL, vocab_model({"GET files.corp /upload": 90.0,
                                                      "GET erp.corp /orders": 10.0}))
    class_model(store, {ip: {"role": "r7", "prob": 0.9, "static": [], "pool": None,
                             "super": "human"} for ip in ips},
                roles={"r7": {"name": "erp-users", "members": []}})
    sysv = {"fmt": 1, "kind": "system", "version": 1, "built": NOW, "H": MV.HALF_LIFE_S,
            "dims": {"tmpl": {"GET erp.corp /orders": [300.0, 5, T0, NOW, 5, 100],
                              "GET erp.corp /home": [100.0, 4, T0, NOW, 4, 100],
                              "GET wiki.corp /page": [5000.0, 50, T0, NOW, 50, 100]}},
            "N": {"tmpl": 5400.0}, "N1": {"tmpl": 0.0}, "n_ent": 60.0, "members": 60}
    store.put_model(S, "__system__", MV.MODEL, sysv)
    run_engine(PortraitEngine(), store, NOW)
    js = portrait(store, "class:r7")["json"]
    assert js["name"] == "erp-users" and js["n_members"] == 5
    assert all(abs(m["prob"] - 0.9) < 1e-9 for m in js["members"])
    assert js["outlier_members"] == [ips[4]]
    assert js["distinctive_tokens"] and \
        js["distinctive_tokens"][0]["template"] in {"GET erp.corp /orders", "GET erp.corp /home",
                                                    "POST erp.corp /orders/new",
                                                    "GET erp.corp /reports/{num}",
                                                    "GET erp.corp /static/app.js",
                                                    "GET files.corp /upload"}
    assert "GET wiki.corp /page" not in {d["template"] for d in js["distinctive_tokens"]}


def test_mixture_quantiles_match_single_nb():
    from app.engines.behavior.lib import bayes
    q = P._mixture_quantiles(np.array([10.0]), np.array([5.0]), np.array([1.0]), 0.0,
                             (0.1, 0.5, 0.9))
    exact = [float(bayes.nb_ppf(x, 10.0, 5.0)) for x in (0.1, 0.5, 0.9)]
    assert all(abs(a - b) <= max(1.0, 0.1 * b) for a, b in zip(q, exact))
    # a point mass of 60 % at zero moves the median to 0
    q0 = P._mixture_quantiles(np.array([10.0]), np.array([5.0]), np.array([0.4]), 0.6,
                              (0.5, 0.95))
    assert q0[0] == 0.0 and q0[1] > 5
    assert P._mixture_quantiles(np.array([10.0]), np.array([5.0]), np.array([0.0]), 0.0,
                                (0.5,)) is None


def test_static_consumes_are_absolute_and_contract_names():
    eng = PortraitEngine()
    assert eng.interval == 8 and eng.period_s == 7200.0 and eng.layer == "behavior"
    for bad in ("behavior.z", "behavior.zr", "behavior.zi"):
        assert bad not in eng.consumes
    assert "profile.extra.portrait" in eng.produces


def test_perf_generous_bound():
    store = make_store()
    n = 60
    base = cached_baseline(9, 18)
    for i in range(n):
        ip = f"10.50.{i // 250}.{i % 250}"
        store.put_model(S, ip, MR.MODEL, rhythm_model(9, 18))
        store.put_model(S, ip, MB.MODEL, base)
        store.put_model(S, ip, MV.MODEL, vocab_model(TEMPLATES))
        store.put_model(S, ip, MC.MODEL, client_model())
        store.register_entity(S, ip)
    t = time.perf_counter()
    assert run_engine(PortraitEngine(), store, NOW) == n
    per_ms = (time.perf_counter() - t) * 1000.0 / n
    assert per_ms < 25.0, per_ms
