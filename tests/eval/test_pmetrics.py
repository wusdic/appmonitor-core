"""Progressive-core gate metrics (eval/pmetrics.py): the scorer gives a perfect
learner (the truth restated as statements) full marks, and each component
fails on the matching degradation."""
from __future__ import annotations

import copy
import os
import re
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from app.eval import packs as P
from app.eval import pmetrics as M
from app.pipeline import orggen as G
from app.pipeline.generator import TrafficGenerator

DAYS = (3, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21)


@pytest.fixture(scope="module")
def org_run():
    pack = P.get_pack("O")
    g = TrafficGenerator(seed=0, pack=pack)
    for _ in range(pack.n_ticks):
        g.step(900.0)
    pt = M.PTruth(g.ptruth)
    return pack, g, pt


def _fake_run(pack, g, snaps, events=(), incidents=()):
    return SimpleNamespace(pack="O", seed=0, truth=copy.deepcopy(g.truth), ptruth=g.ptruth,
                           psnaps=snaps, config=pack.config, events=list(events),
                           incidents=list(incidents), series={}, scenario_dt=900.0,
                           gen_stats={}, aborted=None)


def _stmt(snap, system, route, pred=lambda s: True):
    for s in snap["systems"][system]["model.pviews"]["statements"]:
        if s["evidence"]["route"] == route and pred(s):
            return s
    raise KeyError(route)


GA = {"10.168.7.121", "192.168.1.21", "192.168.1.23"}
is_ga = lambda s: set(s["evidence"]["who"].get("items") or []) == GA


def test_oracle_scores_perfect(org_run):
    pack, g, pt = org_run
    ipc = M.ip_classes_of(pack.config)
    r = M.pg1_snapshot(M.truth_as_statements(pt, 14), pt, ipc, 14, 0, precision_n=150)
    assert r["n_truth"] >= 30 and r["recall"] == 1.0
    assert all(v == 1.0 for v in r["components"].values())
    assert r["precision"] == 1.0 and r["ece"] <= 0.05 + 1e-9


def test_opportunity_rule(org_run):
    pack, g, pt = org_run
    # GA login after D2 opens on day 13: 5 workdays x 3 logins = 15 < 20 by day 21
    assert not M.eligible(pt.by_tid["GA.oa.login#0@2"], pt, 21)
    assert M.eligible(pt.by_tid["GA.oa.login#0"], pt, 11)
    assert not M.eligible(pt.by_tid["GA.oa.login#0"], pt, 3)       # 9 logins over 3 dates
    assert not M.eligible(pt.by_tid["SALES.oa.weekly#0"], pt, 21)  # 2 Fridays (day 5 is a holiday)


@pytest.mark.parametrize("damage,component", [
    (lambda s: s["evidence"]["when"].update(workday=[[570, 591]]), "when"),
    (lambda s: s["evidence"]["who"]["items"].extend(["192.168.1.99", "192.168.1.98"]), "who"),
    (lambda s: s["evidence"]["content"]["body.kv.username"].update(grammar="[a-z]{3,8}"), "content"),
    (lambda s: s["evidence"]["content"]["body.len"].update(band90=[1500, 2048]), "content"),
    (lambda s: s["evidence"]["content"]["body.keys"].update(required=["username"]), "content"),
    (lambda s: s["evidence"]["bindings"]["body.kv.username"]["table"].update(
        {"192.168.1.21": "rose", "192.168.1.23": "jack"}), "bindings"),
])
def test_component_failures(org_run, damage, component):
    pack, g, pt = org_run
    ipc = M.ip_classes_of(pack.config)
    snap = M.truth_as_statements(pt, 11)
    damage(_stmt(snap, "oa", "POST /login", is_ga))
    r = M.pg1_snapshot(snap, pt, ipc, 11, 0, precision_n=100)
    rec = r["per_pattern"]["GA.oa.login#0"]
    assert not rec["recovered"] and rec["components"][component] is False
    assert r["recall"] < 1.0


def test_workflow_and_precision_failures(org_run):
    pack, g, pt = org_run
    ipc = M.ip_classes_of(pack.config)
    snap = M.truth_as_statements(pt, 13)
    for s in snap["systems"]["oa"]["model.pviews"]["statements"]:
        s["evidence"]["workflow"] = []
    # a statement learned from nothing in the truth, and one with a wrong window
    bogus = copy.deepcopy(_stmt(snap, "oa", "POST /login", is_ga))
    bogus["evidence"]["route"] = "GET /admin/export"
    wrong = _stmt(snap, "finance", "POST /fin/approval/{id}/approve")
    wrong["evidence"]["when"]["workday"] = [[1200, 1260]]
    snap["systems"]["oa"]["model.pviews"]["statements"].append(bogus)
    r = M.pg1_snapshot(snap, pt, ipc, 13, 0, precision_n=100)
    assert r["per_pattern"]["GA.oa.approvals#1"]["components"]["workflow"] is False
    assert r["precision"] < 1.0
    n = r["n_confirmed"]
    assert r["precision"] == pytest.approx((n - 2) / n)


def test_candidates_do_not_count(org_run):
    pack, g, pt = org_run
    snap = M.truth_as_statements(pt, 14)
    for rec in snap["systems"].values():
        for s in rec["model.pviews"]["statements"]:
            s["state"] = "candidate"
    r = M.pg1_snapshot(snap, pt, M.ip_classes_of(pack.config), 14, 0, precision_n=20)
    assert r["recall"] == 0.0 and r["precision"] is None


def test_pg2_convergence_logic(org_run):
    pack, g, pt = org_run
    pg1 = {d: {"recall": x, "recall_by_period": {"daily": x, "weekly": w}, "mean_depth": dp,
               "ece": 0.03, "stmt_conf": {"GA.oa.documents#0": c}, "who_U": {"a": u}}
           for d, x, w, dp, c, u in [(3, 0.2, None, 1.0, 0.5, 0.2), (7, 0.85, 0.1, 1.5, 0.7, 0.1),
                                     (10, 0.9, 0.3, 2.0, 0.8, 0.05), (14, 0.7, 0.5, 2.0, 0.9, 0.04),
                                     (21, 0.92, 0.9, 2.2, 0.95, 0.03)]}
    out = M.pg2_convergence(pg1, pt)
    assert out["days_to_80"] == {"daily": 7, "weekly": 21}
    assert out["recall_monotone"] is True               # the dip is inside the drift days
    pg1[21]["recall"] = 0.6
    assert M.pg2_convergence(pg1, pt)["recall_monotone"] is False
    assert out["depth_nondecreasing"] and out["conf_nondecreasing"] and out["U_nonincreasing"]


def test_pg3_and_groups(org_run):
    pack, g, pt = org_run
    ipc = M.ip_classes_of(pack.config)
    snap = M.truth_as_statements(pt, 21)
    ip2g = {}
    for code, rec in pt.groups.items():
        for m in rec.get("members") or []:
            ip2g[m["ip"]] = code
    snap["org"]["model.who_groups"] = {"ip2g": ip2g}
    out = M.pg3_specificity(snap, pt, ipc)
    assert out["ga_login_who"] and out["ga_bindings"] == 3 and out["fin_approval_who"]
    assert out["portal_login_level"] and out["portal_bindings"] == 0
    assert out["ari"] == pytest.approx(1.0) and out["dev_pool_grouped"] and out["dev_who_ok"]
    # a portal statement closed on single IPs, and DEV patterns on single pool IPs, fail
    pl = _stmt(snap, "portal", "POST /login")
    pl["evidence"]["who"] = {"level": "ip", "items": ["10.60.1.2"], "closed": True, "U": 0.01}
    code = _stmt(snap, "code", "TLS git.corp.local")
    code["evidence"]["who"] = {"level": "ip", "items": ["10.50.1.7"], "closed": True, "U": 0.01}
    out = M.pg3_specificity(snap, pt, ipc)
    assert out["portal_login_level"] is False and out["dev_who_ok"] is False
    shuffled = dict(zip(ip2g, np.random.default_rng(0).permutation(list(ip2g.values()))))
    snap["org"]["model.who_groups"] = {"ip2g": shuffled}
    assert M.pg3_specificity(snap, pt, ipc)["ari"] < 0.5


def test_pg5_drift_with_oracle_snapshots(org_run):
    pack, g, pt = org_run
    snaps = {d: M.truth_as_statements(pt, d) for d in DAYS}
    run = _fake_run(pack, g, snaps)
    out = M.pg5_drift(run, pt, M._stmts_by_day(snaps, M.ip_classes_of(pack.config)))
    assert out["D1"]["pass"] and out["D1"]["confirm_workdays"] == 1
    assert out["D2"]["pass"] and out["D2"]["switch_day"] == 13 and out["D2"]["needed_day"] >= 16
    assert out["D3"]["pass"] and out["D4"]["pass"] and out["D5"]["pass"]
    assert out["non_adoption_binding"]["pass"]
    # a learner that keeps mike and adopts rose for .21 fails D2 and non-adoption
    for d in DAYS:
        for s in snaps[d]["systems"]["oa"]["model.pviews"]["statements"]:
            t = s["evidence"]["bindings"].get("body.kv.username", {}).get("table", {})
            if "10.168.7.121" in t:
                t["10.168.7.121"] = "mike"
            if "192.168.1.21" in t and d >= 19:
                t["192.168.1.21"] = "rose"
    out = M.pg5_drift(run, pt, M._stmts_by_day(snaps, M.ip_classes_of(pack.config)))
    assert out["D2"]["pass"] is False and out["non_adoption_binding"]["pass"] is False


def test_pg6_detection_and_far(org_run):
    pack, g, pt = org_run
    tr = {r["scenario_id"]: r for r in g.truth}
    a1, a3 = tr["A1"], tr["A3"]
    evs = [{"kind": "pattern_violation", "system": "finance", "entity": "192.168.1.23",
            "ts": a1["t_first"] + 900.0, "severity": "medium",
            "extra": {"type": "who", "flags": ["outsider_group", "system_new"]}},
           {"kind": "pattern_violation", "system": "oa", "entity": "192.168.3.25",
            "ts": a3["t_first"] + 3 * 900.0, "severity": "medium",
            "extra": {"type": "content", "flags": ["injection_shape"]}},       # too late
           {"kind": "pattern_violation", "system": "oa", "entity": "192.168.3.21",
            "ts": a3["t_first"], "severity": "low", "extra": {"type": "when"}}]  # clean entity
    incs = [{"system": "finance", "entity": "192.168.1.23", "opened": a1["t_first"] + 900,
             "severity": "high", "history": [{"ts": a1["t_first"] + 900, "severity": "high"}],
             "explanation": {"top_features": ["conf_who"]}},
            {"system": "oa", "entity": "192.168.3.22", "opened": a1["t_first"], "severity": "low",
             "history": [{"ts": a1["t_first"], "severity": "low"}]}]
    run = _fake_run(pack, g, {}, evs, incs)
    an = M.pg6_anomalies(run, pt, {})
    assert an["A1"]["detected"] and an["A1"]["top_reason_ok"]
    assert not an["A3"]["violation"] and not an["A3"]["detected"]
    assert set(an) == {f"A{i}" for i in range(1, 11)}
    far = M.pg6_far(run, pt)
    assert far["inc_low"] == 1 and far["inc_medium"] == 0 and far["pv_low"] == 1
    assert far["entity_days"] > 1000 and far["far_low"] < 0.01


def test_who_code_lengths_pick_the_right_level(org_run):
    pack, g, pt = org_run
    regions = {f"reg:{c}": M._net(c) for c, _ in G.PORTAL_REGIONS}
    groups = {m["ip"]: code for code, rec in pt.groups.items() for m in rec.get("members") or []}
    gsize = {code: max(1, len({m["ip"] for m in rec.get("members") or []}))
             for code, rec in pt.groups.items()}
    oa = M.who_code_lengths(pt.who_log["oa"], groups, gsize, regions)
    code = M.who_code_lengths(pt.who_log["code"], groups, gsize, regions)
    fin = M.who_code_lengths(pt.who_log["finance"], groups, gsize, regions)
    assert min(oa, key=oa.get) == "grp" and min(code, key=code.get) == "grp"
    assert min(fin, key=fin.get) in ("ip", "grp")
    # a one-shot population spread over regions: a region (or no IP) beats /32
    r = np.random.default_rng(0)
    log = {f"2025-09-{d:02d}": {f"10.60.{int(r.integers(0, 256))}.{int(r.integers(1, 255))}": 1
                                for _ in range(300)} for d in range(1, 22)}
    one = M.who_code_lengths(log, {}, {}, regions)
    assert one["reg"] < one["ip"] and one["none"] < one["ip"]


def test_regex_tools():
    r = np.random.default_rng(0)
    assert M.regex_contained(r"[a-z]{4}(\.[a-z])?", r"[a-z.]{1,10}", r) is True
    assert M.regex_contained(r"[a-z]{3,12}", r"[a-z.]{1,10}", r) is False
    assert M.regex_contained(r"(jack|rose|mike\.w)", r"[a-z.]{1,10}", r) is True
    xs = M.sample_regex(r"[0-9a-f]{32}", r, 50)
    assert all(re.fullmatch(r"[0-9a-f]{32}", x) for x in xs)
    assert M.norm_route("POST oa.corp.local /approval/{num}/approve") == ("POST", "/approval/{}/approve")
    assert M.norm_route("GET", "/docs/123") == ("GET", "/docs/{}")
    assert M.window_iou([[540, 561]], [[541, 561]]) == pytest.approx(20 / 21)
    assert M.window_iou([], []) is None
    assert M.ari(list("aabbcc"), list("xxyyzz")) == pytest.approx(1.0)
    assert M.loglog_slope([1, 10, 100], [5, 50, 500]) == pytest.approx(1.0)
    assert M.ece([0.9, 0.9], [1.0, 1.0]) == pytest.approx(0.1)


def test_pg7_schema(org_run):
    pack, g, pt0 = org_run
    pt = M.PTruth(dict(g.ptruth, attr_truth=[
        {"name": "meta.f001", "type": "categorical", "cls": "informative", "systems": ["oa"],
         "appears": pt0.day_start[2] + 100.0},
        {"name": "meta.f002", "type": "numeric", "cls": "noise", "systems": ["oa"],
         "appears": pt0.day_start[2] + 100.0},
        {"name": "meta.f003", "type": "categorical", "cls": "constant", "systems": ["oa"],
         "appears": pt0.day_start[2] + 100.0}]))
    reg = {"meta.f001": {"type": "categorical", "first_seen": pt0.day_start[2] + 900.0, "role_sys": "split"},
           "meta.f002": {"type": "numeric", "first_seen": pt0.day_start[2] + 900.0, "role_sys": "dropped"},
           "meta.f003": {"type": "ordinal", "first_seen": pt0.day_start[2] + 2000.0,
                         "role_sys": "target"}}
    snaps = {d: {"systems": {"oa": {"model.attr": {"attrs": reg}}}} for d in (3, 4, 5)}
    run = _fake_run(pack, g, snaps)
    out = M.pg7_schema(run, pt)
    assert out["registered_first_tick"] == pytest.approx(2 / 3)
    assert out["type_correct"] == pytest.approx(2 / 3)
    assert out["informative_kept"] == 1.0 and out["noise_dropped"] == 1.0
    assert out["constants_invariant"] == 0.0 and out["role_within_24h"] == 1.0


def test_pg8_arms(org_run):
    pack, g, pt = org_run
    good = {s: {"model.sysprof": {"chosen": {k: v[0] for k, v in arms.items()}}}
            for s, arms in pt.strategy.items()}
    bad = copy.deepcopy(good)
    bad["oa"]["model.sysprof"]["chosen"]["who"] = "none"
    snaps = {d: {"systems": good if d < 12 else bad} for d in (7, 10, 14, 21)}
    out = M.pg8_adaptation(_fake_run(pack, g, snaps), pt)
    assert out["arms_ok_share"] == pytest.approx(5 / 6)
    assert out["max_switches"] == 1
    assert out["systems"]["oa"]["who_within_5pct"] is False
    assert out["systems"]["code"]["who_within_5pct"] is True       # grp is the best level


def test_pg10_views(org_run):
    pack, g, pt = org_run
    snaps = {d: M.truth_as_statements(pt, d) for d in (11, 21)}
    for d in (11, 21):
        s = _stmt(snaps[d], "oa", "POST /login", is_ga)
        s["evidence"]["content"]["body.kv.username"]["grammar"] = r"[a-z]{4}(\.[a-z])?"
    snaps[21]["group_views"] = {"class:grp:G1": {"statements": [
        {"pattern_id": "neg", "text_zh": "综合部在财务系统中从未执行写操作", "view": "group",
         "state": "stable", "evidence": {"negative": True, "target_system": "finance",
                                         "who": {"level": "grp", "members": sorted(GA)}}}]}}
    out = M.pg10_views(_fake_run(pack, g, snaps), pt, M.ip_classes_of(pack.config))
    assert out == {"day11": True, "day21": True, "finance_single_ip": True, "ga_negative_finance": True}
    s = _stmt(snaps[21], "oa", "POST /login", is_ga)
    s["evidence"]["content"]["body.kv.username"]["grammar"] = r"[a-z.]{1,12}"
    assert M.pg10_views(_fake_run(pack, g, snaps), pt, M.ip_classes_of(pack.config))["day21"] is False


def test_score_prun_and_gates(org_run):
    pack, g, pt = org_run
    snaps = {d: M.truth_as_statements(pt, d, confidence=0.97) for d in DAYS}
    sc = M.score_prun(_fake_run(pack, g, snaps), precision_n=40)
    assert sc["pg1_day14"]["recall"] == 1.0 and sc["core_active"]
    assert sc["bindings_ga_fin_day14"] == [6, 6]
    gates = M.compute_pgates([sc, dict(sc, seed=1)])
    assert set(gates) == {f"PG{i}" for i in range(1, 12)}
    g1 = gates["PG1"]["details"]["checks"]
    assert next(c for c in g1 if c["name"] == "recall@14")["pass"] is True
    assert gates["PG5"]["details"]["checks"][0]["pass"] is True
    empty = M.score_prun(_fake_run(pack, g, {}))
    ge = M.compute_pgates([empty])
    assert ge["PG1"]["pass"] is None and ge["PG6"]["value"] is None      # no P engine: n/a


def test_pg9_compare():
    on = {"G1": {"pass": True, "value": 0.96}, "G3": {"pass": False, "value": 0.3}}
    off = {"G1": {"pass": True, "value": 0.97}, "G3": {"pass": True, "value": 0.1}}
    g = M.pg9_compare(on, off)
    assert g["pass"] is False and g["value"] == 1


def test_no_engine_mentions_synthetic_attributes():
    """PG7: no engine or lib module names the generator's synthetic attributes."""
    root = os.path.join(os.path.dirname(__file__), "..", "..", "backend", "app", "engines")
    pat = re.compile(r"meta\.f\d{3}|meta\.z\d{2}|x-client-ver|waf\.score")
    hits = []
    for d, _, files in os.walk(root):
        for f in files:
            if f.endswith(".py"):
                with open(os.path.join(d, f), encoding="utf-8") as fh:
                    if pat.search(fh.read()):
                        hits.append(f)
    assert not hits
