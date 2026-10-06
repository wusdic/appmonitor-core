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


def test_d4_counts_only_incidents_that_involve_the_grown_population(org_run):
    """Round 5 (lead decision, progressive.md §16.13): PG5 D4 judges the
    adaptation to the PUB population's growth on portal. An incident of a
    portal source outside the grown pattern (the AUTO health monitor's
    alarm-only conformity incident, round 4 seeds 3/4) is a false alarm
    for PG6's FAR, not a D4 failure; one of a PUB source, or with a P03
    finding on a PUB route, still fails D4. Fails on the round-4 scorer,
    which counted every incident of any portal source."""
    pack, g, pt = org_run
    snaps = {d: M.truth_as_statements(pt, d) for d in DAYS}
    d4 = next(r for r in g.truth if r["scenario_id"] == "D4")
    ts = float(d4["t_start"]) + 3 * 86400.0
    pub_ip = next(ip for per in pt.who_log["portal"].values() for ip in per
                  if ip.startswith(("10.60.", "10.61.", "172.16.")))

    def inc(i, ent, evidence):
        return {"id": f"inc{i}", "system": "portal", "entity": ent, "entities": [ent],
                "severity": "low", "opened": ts, "evidence": evidence,
                "history": [{"ts": ts, "severity": "low"}]}
    alarm = [{"ts": ts, "source": "alarm", "path": "single_tick", "severity": "low",
              "families": ["conformity"], "p_by_detector": {"conf_seq": 2.4e-4}}]
    sbd = M._stmts_by_day(snaps, M.ip_classes_of(pack.config))
    mon = _fake_run(pack, g, snaps, incidents=[inc(1, "192.168.9.9", alarm)])
    out = M.pg5_drift(mon, pt, sbd)["D4"]
    assert out["pass"] and out["incidents_low"] == 0 and out["incidents_low_other_sources"] == 1
    pub = _fake_run(pack, g, snaps, incidents=[inc(2, pub_ip, alarm)])
    assert M.pg5_drift(pub, pt, sbd)["D4"]["pass"] is False
    pv_ev = {"id": "ev9", "system": "portal", "entity": "192.168.9.9", "kind": "pattern_violation",
             "extra": {"route": "POST portal.corp.local /comment"}}
    pv = [{"ts": ts, "source": "event", "kind": "pattern_violation", "event_id": "ev9"}]
    on_route = _fake_run(pack, g, snaps, events=[pv_ev], incidents=[inc(3, "192.168.9.9", pv)])
    assert M.pg5_drift(on_route, pt, sbd)["D4"]["pass"] is False


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


def _oracle_ctx(pack, pt):
    groups = {m["ip"]: code for code, rec in pt.groups.items()
              if rec.get("kind") in ("static", "service", "shared", "automation", "nat", "pool")
              for m in rec.get("members") or []}
    gsize = {code: (len({m["ip"] for m in rec.get("members") or []}) if rec.get("kind") != "pool"
                    else M._net(rec["cidr"]).num_addresses) for code, rec in pt.groups.items()}
    regions = {f"reg:{n}:{c}": M._net(c) for n, cs in M.ip_classes_of(pack.config).items() for c in cs}
    return groups, gsize, regions


def test_who_arm_oracle_agrees_with_the_strategy_truth(org_run):
    """PG8 truth semantics (progressive.md §16.2 A1): the who arms listed in
    the generator's strategy truth are exactly the arms the offline utility
    (held-out behaviour gain given who + address tie-break, exact groups)
    puts within 5 % + 0.01 bits/event of the best, and the best is listed."""
    pack, g, pt = org_run
    groups, gsize, regions = _oracle_ctx(pack, pt)
    wd = M._workdays(pt)
    for s, arms in pt.strategy.items():
        w = M.who_arm_utilities(pt.act_log[s], groups, gsize, regions, wd)
        U = w["U"]
        best = max(U, key=U.get)
        within = {a for a in U if U[a] >= U[best] - 0.05 * abs(U[best]) - 0.01}
        assert best in arms["who"], (s, U)
        assert set(arms["who"]) <= within, (s, U, arms["who"])
    # OA: the departments, not their addresses, predict behaviour (研发's pool
    # users re-address daily); finance: per IP (approver vs bookkeepers)
    oa = M.who_arm_utilities(pt.act_log["oa"], groups, gsize, regions, wd)
    assert oa["gain"]["grp"] > oa["gain"]["ip"] + 0.5
    fin = M.who_arm_utilities(pt.act_log["finance"], groups, gsize, regions, wd)
    assert fin["gain"]["ip"] > fin["gain"]["grp"]


def test_who_arm_oracle_on_synthetic_logs():
    """Three departments with their own actions in their own /24s, members
    re-leased daily: /24 (= department) predicts behaviour, per IP must
    relearn; returning public visitors doing the same things: no level
    predicts and per-IP conditioning loses."""
    r = np.random.default_rng(1)
    log, pub = {}, {}
    for d in range(1, 15):
        iso = f"2025-03-{d:02d}"
        per, pp = {}, {}
        for k in (1, 2, 3):
            for ip in r.choice(200, 10, replace=False):
                for j in range(3):
                    key = f"10.0.{k}.{int(ip)}\tPOST /d{k}/a{j}\t{9 + k}"
                    per[key] = per.get(key, 0) + int(r.integers(1, 3))
        for _ in range(200):
            k = int(r.integers(0, 300))                  # 300 returning visitors in 4 /16s
            ip = f"203.{k % 4}.{k // 4}.{k % 7 + 1}"
            key = f"{ip}\tGET /p{int(r.integers(0, 4))}\t{int(r.integers(8, 20))}"
            pp[key] = pp.get(key, 0) + 1
        log[iso], pub[iso] = per, pp
    w = M.who_arm_utilities(log, {}, {}, {})
    assert w["gain"]["/24"] > 1.0 and w["gain"]["/24"] > w["gain"]["ip"] + 0.3
    assert max(w["U"], key=w["U"].get) == "prefix"
    wp = M.who_arm_utilities(pub, {}, {}, {})
    assert wp["gain"]["ip"] < 0 and max(wp["U"], key=wp["U"].get) != "ip"


def test_pg8_switches_ignore_probe_days_and_resource_dims(org_run):
    """A P08 probe day (P12 runs an 'off' fitter for one day in 14 to measure
    it) and P15's tier changes are not strategy switches (pack O seed 0:
    finance counted 4 'switches', 3 of them its P08 probe and back)."""
    pack, g, pt = org_run
    base = {s: {k: v[0] for k, v in arms.items()} for s, arms in pt.strategy.items()}
    snaps = {}
    for d in (7, 10, 14, 15, 16, 21):
        ch = {s: dict(c, tier="S" if d != 14 else "XS") for s, c in base.items()}
        probe = []
        if d == 15:
            ch["finance"]["P08"] = "off" if base["finance"]["P08"] == "on" else "on"
            probe = ["P08"]
        snaps[d] = {"systems": {s: {"model.sysprof": {"chosen": c, "probe": probe if s == "finance" else []}}
                                for s, c in ch.items()}}
    out = M.pg8_adaptation(_fake_run(pack, g, snaps), pt)
    assert out["max_switches"] == 0, {s: v["switches_after_7"] for s, v in out["systems"].items()}


def test_grp_truth_accepts_a_prefix_who_with_the_same_partition_of_sources():
    """A grp truth (销售部 opens the CRM) is recovered by a prefix statement
    whose prefixes hold every member and no other org source (the same
    partition, PG8's settled semantics, §16.9 A1) - not by one that also
    holds another department's address or misses a member."""
    from app.eval import pmetrics as PMx
    sales = [f"192.168.3.{20 + i}" for i in range(20)]
    truth = {"level": "grp", "value": "SALES", "members": sales}
    others = {"192.168.2.10", "192.168.1.21", "10.50.0.7"}
    exact = PMx.Who({"level": "prefix", "items": ["192.168.3.0/24"]}, {})
    wide = PMx.Who({"level": "prefix", "items": ["192.168.0.0/16"]}, {})
    part = PMx.Who({"level": "prefix", "items": ["192.168.3.0/28"]}, {})
    assert PMx.who_compatible(truth, exact, others)
    assert not PMx.who_compatible(truth, exact)                 # without the org's sources: unchanged
    assert not PMx.who_compatible(truth, wide, others)          # holds 财务部 and 综合部 too
    assert not PMx.who_compatible(truth, part, others)          # misses members .32-.39
    pt = PMx.PTruth({"group_truth": {"SALES": {"members": [{"ip": ip} for ip in sales]},
                                     "FIN": {"members": [{"ip": "192.168.2.10"}]}}})
    assert pt.others_of(truth) == {"192.168.2.10"}
    assert pt.others_of({"level": "ip", "value": sales}) is None


def test_pg1_content_compares_the_fitted_band_not_its_display_rounding():
    """PG1's band check reads the fitted band (band90_raw) and observed range
    (range_raw): the display grid may round '332-794 B' to '200-800 B' while
    keeping the band's coverage, which failed the +-20 % endpoint check on a
    correctly learned band (pack O portal login)."""
    from app.eval import pmetrics as PMx
    row = {"content": {"body.len": {"band90": [332.8, 793.6], "range": [307.2, 819.2], "core": True}}}
    raw = {"evidence": {"route": "POST portal /login", "content": {"body.len": {
        "band90": [200.0, 800.0], "band90_raw": [340.0, 780.0],
        "range": [300.0, 900.0], "range_raw": [310.0, 830.0]}}}}
    s = PMx.LStmt(raw, "portal", {})
    assert PMx.content_matches(row, s, np.random.default_rng(0))[0]
    off = {"evidence": {"route": "POST portal /login", "content": {"body.len": {
        "band90": [200.0, 800.0], "band90_raw": [200.0, 780.0], "range": [300.0, 900.0]}}}}
    assert not PMx.content_matches(row, PMx.LStmt(off, "portal", {}), np.random.default_rng(0))[0]
    old = {"evidence": {"route": "POST portal /login", "content": {"body.len": {
        "band90": [340.0, 780.0], "range": [310.0, 830.0]}}}}           # no raw keys: display values
    assert PMx.content_matches(row, PMx.LStmt(old, "portal", {}), np.random.default_rng(0))[0]


def test_holdout_events_follow_the_traffic_mix():
    """Round 3: held-out events of a statement follow the traffic of its
    context (rows, day types and member sources in proportion to the events
    the program emitted), not one equal share per truth row: pack O's mail
    nodes (four departments, 销售部 + 研发 ~ 90 % of the records) were checked
    against 25 % per department, so windows fitted on the real mix failed."""
    base = {"system": "mail", "method": "TLS", "route": "m", "windows": {"workday": [[540, 560]]},
            "who": {"level": "ip", "value": ["10.0.0.1"]}, "gen": {"members": {"10.0.0.1": 1.0}}}
    a = dict(base, tid="A#0")
    b = dict(base, tid="B#0", windows={"workday": [[800, 820]]},
             gen={"members": {"10.0.0.2": 0.5, "10.0.0.3": 0.5}})
    traffic = {"A#0": {"workday": {"10.0.0.1": 900}},
               "B#0": {"workday": {"10.0.0.2": 95, "10.0.0.3": 5}}}
    r = np.random.default_rng(0)
    evs = M.holdout_events([a, b], r, 4000, None, traffic)
    share_a = np.mean([e["tid"] == "A#0" for e in evs])
    assert abs(share_a - 0.9) < 0.02, share_a
    eb = [e for e in evs if e["tid"] == "B#0"]
    assert np.mean([e["ip"] == "10.0.0.2" for e in eb]) > 0.9
    # without traffic: round-2 equal shares (kept bit-identical for re-scoring)
    evs0 = M.holdout_events([a, b], np.random.default_rng(0), 4000)
    assert abs(np.mean([e["tid"] == "A#0" for e in evs0]) - 0.5) < 0.03
    # PTruth.traffic reads the opportunities per day type
    # PTruth.traffic reads the opportunities per day type, over the row's
    # lineage (a segment opened on a weekend has its activity's rate)
    pt = M.PTruth({"pattern_truth": [dict(a, valid_from_day=1, valid_to_day=6, lineage="A#0"),
                                     dict(a, tid="A#0@1", valid_from_day=6, valid_to_day=9, lineage="A#0")],
                   "opportunities": {"A#0": {"2025-09-01": {"10.0.0.1": 3}, "2025-09-05": {"10.0.0.1": 1}},
                                     "A#0@1": {"2025-09-06": {"10.0.0.1": 2},
                                               "2025-09-08": {"10.0.0.1": 7}}},
                   "days": {"start": "2025-09-01", "n_days": 8, "day_start": list(range(9)),
                            "workday": [True] * 5 + [False, False, True]}})
    assert pt.traffic(7) == {"A#0@1": {"workday": {"10.0.0.1": 4.0}, "nonworkday": {"10.0.0.1": 2.0}}}


def test_pg1_range_is_judged_on_what_the_emitted_data_could_show(org_run):
    """Evaluator round 3: a content row's hard range is the generator's
    support; a tail the generator never drew cannot be learned. Pack O seed 0
    emits 45 综合部 logins, none of the 5 % below 1 KB, so '100 % in
    0.5-3 KB' was unrecoverable on any engine (example clause, PG10 day 21).
    PG1 now compares the learned range with the support cut to the emitted
    extremes; a learned range that misses an emitted extreme still fails."""
    pack, g, pt = org_run
    row = next(r for r in pt.rows if r["tid"] == "GA.oa.login#0")
    lo, hi = pt.data_range(row, "body.len", 21)
    assert 1024.0 < lo and hi < 3072.0                          # no tail drawn on seed 0
    obs = pt.observable(row, 21)
    tc = obs["content"]["body.len"]
    assert tc["range"] == [max(512.0, lo), min(3072.0, hi)] and tc["range_support"] == [512.0, 3072.0]
    assert row["content"]["body.len"]["range"] == [512.0, 3072.0]          # the truth itself unchanged

    def stmt(rg):
        return M.LStmt({"evidence": {"route": "POST oa /login", "content": {"body.len": {
            "band90": [1024.0, 2048.0], "band90_raw": [1030.0, 2040.0], "range": rg, "range_raw": rg}}}},
            "oa", {})
    r = np.random.default_rng(0)
    learned = stmt([lo, hi])
    def ok(rw, st):
        return M.content_matches(rw, st, r)[1]["body.len"]
    assert ok(obs, learned)
    assert not ok(row, learned)                                  # the support check could never pass
    assert not ok(obs, stmt([1.5 * lo, hi]))                     # misses emitted small logins
    assert ok(obs, stmt([512.0, 3072.0]))                        # the support itself still matches
    # runs scored before the truth carried extremes are judged as before
    assert M.PTruth({k: v for k, v in g.ptruth.items() if k != "extremes"}).observable(row, 21) is row


def test_pg1_window_is_compared_at_the_coverage_it_states(org_run):
    """Evaluator round 3: the truth's windows are the central 99 % of a step's
    arrival law. A statement stating 89 % coverage is compared with the law's
    central 89 % (pack O portal login, normal law over 07:00-23:00: a correct
    89 % window [09:32, 20:04] had IoU 0.66 against the 99 % window and failed
    PG1 'when' on every seed). A misplaced window still fails."""
    pack, g, pt = org_run
    row = next(r for r in pt.rows if r["tid"] == "PUB.portal.visit#1")

    def stmt(wins, cov):
        return M.LStmt({"evidence": {"route": "POST portal /login",
                                     "when": {"workday": wins, "nonworkday": wins, "coverage": cov}}},
                       "portal", {})
    learned = [[572.0, 1204.0]]                                  # pack O seed 0, day 21
    assert M.window_iou(row["windows"]["workday"], learned) < 0.7
    w89 = M.law_windows(row, "workday", 0.89)
    assert len(w89) == 1 and M.window_iou(w89, learned) >= 0.7
    assert M.when_compatible(row, stmt(learned, 0.89))
    assert not M.when_compatible(row, stmt(learned, 0.99))       # claims 99 %: judged against 99 %
    shifted = [[x + 150 for x in learned[0]]]
    assert not M.when_compatible(row, stmt(shifted, 0.89))
    # a narrow uniform window (综合部 login) is unaffected
    login = next(r for r in pt.rows if r["tid"] == "GA.oa.login#0")
    assert M.when_compatible(login, stmt(login["windows"]["workday"], 0.9))


def test_login_bindings_ask_for_what_could_be_learned_by_the_day(org_run):
    """Evaluator round 3: PG1's 'GA + FIN bindings 6/6 at day 14' asked for
    D2's renamed user (10.168.7.121 -> mike.w from day 13) one day after the
    change, which no learner can bind yet (PG5 D2 gives the rename its own
    latency): 5/6 on every seed. The check now reads the newest segment with
    PG1's evidence (>= 20 opportunities on >= 3 dates)."""
    pack, g, pt = org_run
    ipc = M.ip_classes_of(pack.config)
    before_d2 = M.truth_as_statements(pt, 11)          # the bindings as they were (… -> mike)
    assert M.login_bindings(before_d2, pt, ipc, 14) == (6, 6)
    late = M.truth_as_statements(pt, 21)               # mike.w, learned long after D2
    assert M.login_bindings(late, pt, ipc, 21) == (6, 6)
    assert M.login_bindings(before_d2, pt, ipc, 21)[0] < 6   # by day 21 the rename must be learned


def test_calibration_error_is_debiased_for_small_statement_sets():
    """§16.12: one snapshot judges ~60 statements; the plug-in ECE of a
    perfectly calibrated set is ~0.1 there, so the gate reads the debiased
    calibration error, which is ~0 for a calibrated set and finds a real bias."""
    r = np.random.default_rng(5)
    ces, eces = [], []
    for k in range(40):
        c = r.uniform(0.3, 0.95, 60)
        o = (r.random(60) < c).astype(float)
        cal = M.calibration(c, o)
        ces.append(cal["ce"])
        eces.append(cal["ece"])
        assert cal["ece_null"] == pytest.approx(cal["ece_null"], abs=1e-12) and cal["ece_null"] > 0.05
    assert np.median(eces) > 0.08                 # the plug-in's floor at n = 60
    assert np.median(ces) < 0.05
    c = r.uniform(0.3, 0.7, 600)                  # under-confident by 0.2 (round 3: 0.48 vs 0.73)
    o = (r.random(600) < c + 0.2).astype(float)
    cal = M.calibration(c, o)
    assert 0.15 < cal["ce"] < 0.26 and cal["hold"] - cal["conf"] > 0.15
    assert M.calibration([], [])["ce"] is None


def test_pg2_confidence_trend_is_paired_on_a_fixed_cohort(org_run):
    """A newly recovered pattern enters at a low confidence: the median over
    whatever is recovered falls although every old pattern gained (round 3,
    seed 0, days 7 -> 10). PG2 compares the same patterns across the spec's
    snapshot days, outside the drift days, as recall."""
    pack, g, pt = org_run
    un = sorted(r["tid"] for r in pt.rows if int(r["valid_from_day"]) == 1
                and int(r["valid_to_day"]) > pt.n_days)[:4]
    assert len(un) == 4

    def day(confs, us=None):
        return {"recall": 0.8, "recall_by_period": {}, "mean_depth": 1.0, "ece": 0.1,
                "stmt_conf": dict(zip(un, confs)), "who_U": us or {}}
    pg1 = {7: day([0.6, 0.7]), 10: day([0.65, 0.75, 0.2, 0.25]), 13: day([0.1, 0.1, 0.1, 0.1]),
           14: day([0.66, 0.76, 0.3, 0.3]), 21: day([0.7, 0.8, 0.35, 0.4])}
    out = M.pg2_convergence(pg1, pt)
    assert out["median_conf"][1][1] < out["median_conf"][0][1]      # the unpaired median dipped (7 -> 10)
    assert out["conf_nondecreasing"] is True
    assert [s[:3] for s in out["conf_steps"]] == [[7, 10, 2], [10, 21, 4]]   # 13 and 14: drift days
    pg1[21] = day([0.7, 0.8, 0.1, 0.1])                              # the old patterns lose confidence
    assert M.pg2_convergence(pg1, pt)["conf_nondecreasing"] is False
    # the unseen-IP mass, paired the same way
    pg1 = {7: day([0.5], {"a": 0.1}), 10: day([0.5], {"a": 0.05, "b": 0.3}), 21: day([0.5], {"a": 0.04, "b": 0.2})}
    assert M.pg2_convergence(pg1, pt)["U_nonincreasing"] is True


def test_pg2_calibration_pools_the_claims_of_days_7_on(org_run):
    pack, g, pt = org_run
    r = np.random.default_rng(2)
    pg1 = {}
    for d in (5, 7, 10, 14, 21):
        c = r.uniform(0.4, 0.95, 60)
        o = (r.random(60) < (c if d >= 7 else 0.0 * c)).astype(int)
        pg1[d] = {"recall": 0.8, "recall_by_period": {}, "ece": M.ece(c, o),
                  "calib": [[float(x), int(y), 0] for x, y in zip(c, o)]}
    out = M.pg2_convergence(pg1, pt)
    assert out["calibration"]["n"] == 240                            # day 5 not pooled
    assert out["ece"] == out["calibration"]["ce"] < 0.05
    assert out["ece_last_day"] == pg1[21]["ece"] > 0.05


def test_pg10_window_is_judged_at_the_coverage_it_states(org_run):
    """§16.12: PG10 asked both edges within 2 min of the 99 % window (08:30-
    08:51); a window stated at 74 % coverage of that uniform law is ~15.5 min
    long and could never pass (round 3, seed 4: 08:30-08:45 at 74 %). It is now
    compared with the law's intervals holding the stated coverage."""
    pack, g, pt = org_run
    row = next(x for x in pt.valid_at_day(21) if x["activity"] == "GA.oa.login" and x["step"] == 0)
    tw = row["windows"]["workday"]
    a, b = tw[0]
    assert b - a >= 15
    assert M.window_edges_ok(row, "workday", [[a, b]], 0.99)
    assert M.window_edges_ok(row, "workday", [[a + 1, b]], 0.74)          # under-claims: the 99 % edges
    full = M.law_mass(row, "workday", [[a, b]])
    assert full > 0.95
    k = 0.74
    w = [[a, a + k * (b - a)]]                                             # a 74 % interval, left-anchored
    assert M.window_edges_ok(row, "workday", w, k)
    assert not M.window_edges_ok(row, "workday", w, 0.99)                  # claims 99 %: edge 5 min short
    assert not M.window_edges_ok(row, "workday", [[a, a + 0.4 * (b - a)]], k)   # too short for 74 %
    assert not M.window_edges_ok(row, "workday", [[a, b], [b + 15, b + 25]], k)  # a window outside


def test_pg2_trends_read_the_spec_snapshot_days(org_run):
    """Runs snapshot every day since round 3; §12's trend checks are defined
    on the snapshot days {3, 5, 7, 10, 14, 21}. A day-to-day dip between them
    (seed 1: mean depth 1.0 on day 3, 0.9 on day 4) is reported, not judged."""
    pack, g, pt = org_run
    rec = {d: min(0.9, 0.1 * d) for d in range(1, 22)}
    rec[8] = rec[7] - 0.2
    dep = {d: 1.0 + 0.05 * d for d in range(1, 22)}
    dep[4] = 0.9
    pg1 = {d: {"recall": rec[d], "recall_by_period": {}, "mean_depth": dep[d]} for d in range(1, 22)}
    out = M.pg2_convergence(pg1, pt)
    assert out["trend_days"] == [3, 5, 7, 10, 14, 21]
    assert out["recall_monotone"] is True and out["recall_violations_daily"] == [[7, 8]]
    assert out["depth_nondecreasing"] is True
    pg1[10]["recall"] = rec[7] - 0.2                                     # a dip ON a snapshot day
    assert M.pg2_convergence(pg1, pt)["recall_monotone"] is False


def test_r13_ratio_reads_the_single_tree_before_it_joined():
    """§16.12: pack O's CRM may join R13's branch family (same application);
    the memory ratio then compares with its own tree's last snapshot."""
    t = lambda n: {"nodes": ["x" * 10] * n}                                      # noqa: E731
    snaps = {5: {"systems": {"crm": {"model.ptree": t(10)}, "fam:1": {}}},
             7: {"systems": {"crm": {"model.ptree": None}, "fam:1": {"model.ptree": t(15)}}}}
    r = M.family_size_ratio(snaps, "fam:1", "crm")
    assert r is not None and 1.0 < r < 2.0
    assert M.family_size_ratio({7: {"systems": {"fam:1": {"model.ptree": t(5)}}}}, "fam:1", "crm") is None


def test_statement_order_does_not_change_the_evaluators_draws(org_run, monkeypatch):
    """Evaluator round 4 (views owner's open issue): the precision draws came
    from one stream per snapshot, so a pure reordering or folding of statements
    moved precision by up to 0.03. Each claim (and each truth pattern's
    recovery) now draws from its own stream."""
    pack, g, pt = org_run
    ipc = M.ip_classes_of(pack.config)
    seen = {}

    def spy_holdout(s, valid, pt_, r, n, day):
        seen.setdefault(M._claim_key(s), []).append(float(r.random()))
        return {"ok": True}

    real_recover = M.recover
    rec_draws = {}

    def spy_recover(row, stmts, edges, pt_, r):
        rec_draws.setdefault(row["tid"], []).append(float(r.random()))
        return real_recover(row, stmts, edges, pt_, r)

    monkeypatch.setattr(M, "holdout_check", spy_holdout)
    monkeypatch.setattr(M, "recover", spy_recover)
    snap = M.truth_as_statements(pt, 14)
    M.pg1_snapshot(snap, pt, ipc, 14, 0, precision_n=20)
    rev = copy.deepcopy(snap)
    for rec in rev["systems"].values():
        rec["model.pviews"]["statements"].reverse()
    M.pg1_snapshot(rev, pt, ipc, 14, 0, precision_n=20)
    assert len(seen) >= 10
    assert all(len(v) == 2 and v[0] == v[1] for v in seen.values())
    assert all(len(v) == 2 and v[0] == v[1] for v in rec_draws.values())


def test_a_period_without_an_eligible_pattern_is_not_measured():
    """Evaluator round 4: pack O's only weekly pattern (SALES' Friday report)
    has two Fridays in 21 days, so no weekly pattern is ever eligible (>= 3
    dates); "weekly patterns: 80 % recall by day 21" read None as 'never' and
    failed PG2 on every seed. With no eligible pattern of a period the check is
    not measured; with one that never reaches 80 % it still fails."""
    def sc(weekly):
        pg1 = {d: {"recall_by_period": {"daily": 0.9, "weekly": weekly}} for d in (7, 14, 21)}
        return {"pack": "O", "n_snapshots": 3, "pg1": pg1,
                "pg2": {"days_to_80": {"daily": 7, "weekly": None}}}
    assert M._ttr_ok(sc(None), "weekly", 21) is None
    assert M._ttr_ok(sc(None), "daily", 7) is True
    assert M._ttr_ok(sc(0.5), "weekly", 21) is False


def test_engine_family_statements_are_judged_for_their_member_systems():
    """Evaluator round 5 (EV5-7): P12's family keys are counters ('fam:1'),
    the truth names a family after a system ('oa': [oa, oa-r2]). A statement
    of the family tree matched no truth row of oa (O-real seed 0: oa's recall
    0 from day 11 once its frozen pre-join view was retired, EV5-6)."""
    st = {"id": "p:fam:1:0:3@1.0", "pattern_id": "p:fam:1:0:3@1.0", "state": "confirmed",
          "confidence": 0.9, "text_zh": "【fam:1】… GET /docs", "view": "system",
          "evidence": {"system": "fam:1", "method": "GET", "route": "/docs"}}
    snap = {"systems": {"fam:1": {"model.pviews": {"view": "system", "statements": [st]}}},
            "org": {"model.sysfam": {"member": {"oa": "fam:1", "oa-r2": "fam:1"},
                                     "families": {"fam:1": ["oa", "oa-r2"]}}}}
    stmts = M.statements(snap, {})
    assert len(stmts) == 1 and stmts[0].members == {"oa", "oa-r2"}
    fam = {"oa": {"oa", "oa-r2"}, "oa-r2": {"oa", "oa-r2"}}
    pt = SimpleNamespace(family_of=lambda x: fam.get(x, {x}))
    s = stmts[0]
    assert M._sys_match(s.system, {"system": "oa"}, pt, s.members)
    assert M._sys_match(s.system, {"system": "oa-r2"}, pt, s.members)
    assert not M._sys_match(s.system, {"system": "crm"}, pt, s.members)
    assert M._find(stmts, "oa", "GET", "/docs") == [s]       # the example / PG3 / PG10 lookups
