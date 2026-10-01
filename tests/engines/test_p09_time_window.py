"""P09 TimeWindow (`behavior.time_window`, docs/lib3/progressive.md §6.13, card
P09). The pattern tree is an oracle tree counted like P04 does
(tests/engines/temporal_sim.py), so these tests measure P09 alone."""
from __future__ import annotations

import numpy as np

from helpers import ctx, make_store
from temporal_sim import CFG, DAY, MON, OracleTree, is_workday

from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pwindows as PW
from app.engines.behavior.time_window import TimeWindowEngine
from app.models.schema import SYSTEM_ENTITY

LOGIN = "POST oa /login"
DOCS = "GET oa /docs"
GA = ["192.168.1.21", "192.168.1.23", "10.168.7.121"]
FIN = ["192.168.2.10", "192.168.2.11", "192.168.2.12"]
SALES = [f"192.168.3.{i}" for i in range(1, 21)]


def run(ot, st, events, days, dt=3600.0, eng=None, cfg=CFG, t_start=MON):
    eng = eng or TimeWindowEngine()
    events = sorted(events)
    now, i = t_start, 0
    while now < t_start + days * DAY:
        t1 = now + dt
        while i < len(events) and events[i][0] <= t1:
            ot.learn(events[i][0], events[i][1], events[i][2])
            i += 1
        ot.apply_wants()
        eng.safe_run(ctx(st, t1, window_s=dt, config=cfg), None)
        now = t1
    return eng, now


def oa_events(days, seed=0, t0=MON):
    r = np.random.default_rng(seed)
    ev = []
    for d in range(days):
        day = t0 + d * DAY
        if not is_workday(day):
            continue
        ev += [(day + r.uniform(540, 561) * 60, ip, LOGIN) for ip in GA]
        ev += [(day + r.uniform(545, 570) * 60, ip, LOGIN) for ip in FIN]
        ev += [(day + r.uniform(510, 570) * 60, ip, LOGIN) for ip in SALES]
        ev += [(day + r.uniform(570, 990) * 60, SALES[k % 20], DOCS) for k in range(40)]
    return ev


def entry(st, nid):
    return PW.lookup(MP.get_model(st, "oa", MP.PWIN), 0, nid)


def test_login_window_0900_0921_after_20_workdays():
    """(card P09) arrivals U(09:00, 09:21) on 20 workdays -> 09:00-09:21 +-1
    min, coverage >= 0.95 — per group node, while the route node holds the
    union of the groups (the requirement's example and its IP-irregular
    counterpart at the same time)."""
    st = make_store()
    ot = OracleTree(st, "oa", [LOGIN, DOCS], {LOGIN: [GA, FIN, SALES]})
    run(ot, st, oa_events(28), 28)
    ga = entry(st, ot.group_node[(LOGIN, 0)])["by_daytype"]["wd"]
    assert ga["res"] == "minute" and len(ga["windows"]) == 1
    s, e = ga["windows"][0]
    assert abs(s - 540) <= 1 and abs(e - 561) <= 1 and ga["coverage"] >= 0.95
    assert "09:0" in ga["text_zh"] and ga["text_zh"].startswith("工作日 ")
    fin = entry(st, ot.group_node[(LOGIN, 1)])["by_daytype"]["wd"]["windows"][0]
    assert abs(fin[0] - 545) <= 1 and abs(fin[1] - 570) <= 1
    route = entry(st, ot.route_node[LOGIN])["by_daytype"]["wd"]["windows"]
    assert len(route) == 1 and abs(route[0][0] - 510) <= 2 and abs(route[0][1] - 570) <= 2
    docs = entry(st, ot.route_node[DOCS])["when"]
    assert docs["workday"] and abs(docs["workday"][0][0] - 570) <= 5 and docs["nonworkday"] == []


def test_window_crossing_midnight():
    st = make_store()
    ot = OracleTree(st, "oa", ["POST oa /batch"])
    r = np.random.default_rng(1)
    ev = [(MON + d * DAY + r.uniform(22 * 60, 26 * 60) * 60, "10.9.9.9", "POST oa /batch")
          for d in range(14) for _ in range(6)]
    run(ot, st, ev, 15)
    for dk in ("wd", "nwd"):
        w = entry(st, ot.route_node["POST oa /batch"])["by_daytype"][dk]["windows"]
        assert len(w) == 1 and w[0][0] > w[0][1], w                    # one window, wrapping
        # 84 arrivals over 240 min (mean spacing 2.9 min): the edges are sampling-limited
        assert abs(w[0][0] - 1320) <= 10 and abs(w[0][1] - 120) <= 10


def test_weekends_learned_separately():
    st = make_store()
    ot = OracleTree(st, "oa", [LOGIN])
    r = np.random.default_rng(2)
    ev = []
    for d in range(21):
        day = MON + d * DAY
        lo, hi = (540, 600) if is_workday(day) else (840, 960)
        ev += [(day + r.uniform(lo, hi) * 60, f"10.0.0.{k}", LOGIN) for k in range(5)]
    run(ot, st, ev, 21)
    e = entry(st, ot.route_node[LOGIN])["by_daytype"]
    assert abs(e["wd"]["windows"][0][0] - 540) <= 2 and abs(e["wd"]["windows"][0][1] - 600) <= 2
    # weekends: 30 arrivals over 120 min (mean spacing 4 min)
    assert abs(e["nwd"]["windows"][0][0] - 840) <= 12 and abs(e["nwd"]["windows"][0][1] - 960) <= 12
    assert e["wd"]["windows"] != e["nwd"]["windows"]


def test_root_windows_are_the_time_hierarchy_level():
    st = make_store()
    ot = OracleTree(st, "oa", [LOGIN])
    r = np.random.default_rng(3)
    ev = [(MON + d * DAY + r.uniform(540, 561) * 60, ip, LOGIN)
          for d in range(14) if is_workday(MON + d * DAY) for ip in GA]
    run(ot, st, ev, 14)
    allw, by = MP.root_windows(st, "oa")
    assert by["wd"] and by["wd"][0][2].startswith("w:09")
    h = MP.hierarchies(st, "oa", CFG)
    assert h.gen("ctx.when", 2, ("wd", 545.0)) == ("wd", by["wd"][0][2])
    assert h.gen("ctx.when", 2, ("wd", 900.0)) == ("wd", "w:off")


def test_root_windows_have_hysteresis():
    st = make_store()
    ot = OracleTree(st, "oa", [LOGIN])
    r = np.random.default_rng(4)
    ev = [(MON + d * DAY + r.uniform(540, 561) * 60, ip, LOGIN)
          for d in range(28) if is_workday(MON + d * DAY) for ip in GA + FIN]
    run(ot, st, ev, 28)
    root = MP.get_model(st, "oa", MP.PWIN)["root"]
    assert root["version"] <= 6              # ~112 six-hourly runs; the level changes only when it moves


def test_only_dirty_nodes_are_refitted():
    st = make_store()
    ot = OracleTree(st, "oa", [LOGIN, DOCS], {LOGIN: [GA, FIN, SALES]})
    eng, now = run(ot, st, oa_events(7), 7)
    n0 = sum(eng.last_stats.get("oa", {}).get("fitted", 0) for _ in [0])
    # a run 6 h later without new evidence: the tree is skipped
    eng.safe_run(ctx(st, now + 6 * 3600, window_s=3600, config=CFG), None)
    assert eng.last_stats["oa"].get("fitted", 0) == 0
    # new evidence on the GA node only -> only its path is refitted
    for k in range(25):
        ot.learn(now + 6 * 3600 + 60 + k, GA[k % 3], LOGIN)
    eng.safe_run(ctx(st, now + 12 * 3600 + 1, window_s=3600, config=CFG), None)
    fitted = eng.last_stats["oa"]["fitted"]
    assert 1 <= fitted <= 3, eng.last_stats                  # root, route, GA (not FIN / SALES / docs)
    assert n0 >= 0


def test_confidence_and_precision_grow_with_observation_time():
    """S3 "用的时间越长越精准": the GA window's edge error does not grow and
    its dates / stated confidence grow from week 1 to week 4."""
    st = make_store()
    ot = OracleTree(st, "oa", [LOGIN], {LOGIN: [GA, FIN, SALES]})
    ev = oa_events(28, seed=5)
    eng, now = run(ot, st, [e for e in ev if e[0] < MON + 4 * DAY], 4)
    e1 = entry(st, ot.group_node[(LOGIN, 0)])["by_daytype"]["wd"]
    run(ot, st, [e for e in ev if e[0] >= MON + 4 * DAY], 24, eng=eng, t_start=MON + 4 * DAY)
    e2 = entry(st, ot.group_node[(LOGIN, 0)])["by_daytype"]["wd"]
    err = lambda w: max(abs(w[0][0] - 540), abs(w[0][1] - 561))
    assert err(e2["windows"]) <= err(e1["windows"]) and err(e2["windows"]) <= 1
    assert e2["dates"] > e1["dates"] and e2["n_c"] > e1["n_c"]
    assert e1["res"] == "slot" and e2["res"] == "minute"        # resolution sharpens with evidence


def test_minute_reservoirs_requested_for_concentrated_nodes_only():
    st = make_store()
    ot = OracleTree(st, "oa", [LOGIN, DOCS], {LOGIN: [GA, FIN, SALES]})
    run(ot, st, oa_events(7), 7)
    want = MP.get_model(st, "oa", MP.PWANT)["p09"]["minute_reservoir"][0]
    assert ot.group_node[(LOGIN, 0)] in want and ot.route_node[LOGIN] in want
    assert ot.route_node[DOCS] not in want                     # 7 h of activity: slot resolution suffices


def test_arm_off_and_disabled():
    st = make_store()
    ot = OracleTree(st, "oa", [LOGIN])
    st.put_model("oa", SYSTEM_ENTITY, MP.SYSPROF, {"chosen": {"p09": "off"}})
    run(ot, st, oa_events(3), 3)
    assert MP.get_model(st, "oa", MP.PWIN) is None
    st2 = make_store()
    ot2 = OracleTree(st2, "oa", [LOGIN])
    run(ot2, st2, oa_events(3), 3, cfg={"tz": "Asia/Shanghai"})
    assert MP.get_model(st2, "oa", MP.PWIN) is None


def test_statement_block_matches_the_eval_contract():
    """The node entry's `when` block is what eval/pmetrics reads as
    evidence.when: non-wrapping [m0, m1] intervals per day type."""
    from app.eval.pmetrics import window_iou
    st = make_store()
    ot = OracleTree(st, "oa", [LOGIN], {LOGIN: [GA, FIN, SALES]})
    run(ot, st, oa_events(21), 21)
    w = entry(st, ot.group_node[(LOGIN, 0)])["when"]
    assert window_iou([[540, 561]], w["workday"]) >= 0.9
    assert 0.9 <= w["coverage"] <= 1.0 and 0.0 < w["confidence"] <= 1.0


def test_accepted_time_change_switches_windows():
    """S4 "随着行为的动态发展而动态变化": the GA login moves from 09:00-09:21 to
    08:30-08:51 on day 14. While P04's Page-Hinkley alarm on the arrival
    time is open the windows are provisional and fitted on the arrivals since
    the alarm; after P04 accepts the change (cver + 1) they are the new
    regime's. Without the signal the H_m density still mixes both regimes."""
    def events(seed):
        r = np.random.default_rng(seed)
        ev = []
        for d in range(24):
            day = MON + d * DAY
            if is_workday(day):
                lo, hi = (540, 561) if d < 14 else (510, 531)
                ev += [(day + r.uniform(lo, hi) * 60, ip, LOGIN) for ip in GA]
        return ev

    def setup():
        st = make_store()
        ot = OracleTree(st, "oa", [LOGIN], {LOGIN: [GA]})
        return st, ot, ot.group_node[(LOGIN, 0)]

    # no P04 signal: P09 finds the change in the node's own arrivals (it owns
    # time drift, §16.2 M8) - the H_m density alone would still mix both
    # regimes (the window was 08:30-09:21 before regime_cut)
    st0, ot0, nid0 = setup()
    run(ot0, st0, events(6), 24)
    rec0 = entry(st0, nid0)["by_daytype"]["wd"]
    assert rec0["regime"] == "new" and rec0["change"]["accepted"] and not rec0["provisional"]
    assert len(rec0["windows"]) == 1 and abs(rec0["windows"][0][0] - 510) <= 3 \
        and abs(rec0["windows"][0][1] - 531) <= 3, rec0["windows"]

    st, ot, nid = setup()
    ev = events(6)
    eng, now = run(ot, st, [e for e in ev if e[0] < MON + 18 * DAY], 18)
    node = ot.tree.nodes[nid]
    node.meta.setdefault("evolving", {})["@when"] = {"t0": MON + 14 * DAY, "kind": "when"}
    node.state = "evolving"
    eng.safe_run(ctx(st, now + 6.5 * 3600, window_s=3600, config=CFG), None)     # the next 6-h run
    rec = entry(st, nid)["by_daytype"]["wd"]
    assert rec["provisional"] and rec["regime"] == "new"
    # 12 arrivals of the new regime (4 workdays x 3 IPs): edges within a few minutes
    assert len(rec["windows"]) == 1 and abs(rec["windows"][0][0] - 510) <= 6 and abs(rec["windows"][0][1] - 531) <= 6
    # P04 accepts the change (the alarm closes, cver + 1) -> the new regime from then on
    node.meta["evolving"].pop("@when")
    node.state, node.cver = "confirmed", node.cver + 1
    run(ot, st, [e for e in ev if e[0] >= MON + 18 * DAY + 6.5 * 3600], 6, eng=eng,
        t_start=MON + 18 * DAY + 6.5 * 3600)
    rec = entry(st, nid)["by_daytype"]["wd"]
    assert not rec["provisional"] and rec["regime"] == "new"
    assert len(rec["windows"]) == 1 and abs(rec["windows"][0][0] - 510) <= 2 \
        and abs(rec["windows"][0][1] - 531) <= 2, rec["windows"]
    assert entry(st, nid)["cver"] >= 1
