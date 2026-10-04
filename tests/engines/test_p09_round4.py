"""P09 round 4: windows describe the node's CURRENT arrivals of its members,
with an honest coverage (lib/pwindows.fit_regime / _accepted /
cv_coverage, behavior.time_window._points).

Measured on pack O (round 3, F4 runs):
  * seed 3, day 21: the 综合部 login node stated '08:33-08:51、09:09-09:14'
    (IoU 0.69): two pre-D1 dates of the three members at 09:09-09:13 (too few
    for regime_cut) joined A2's 09:10 logins of 192.168.1.21;
  * seed 1, day 14: the 60-s health monitors stated a 23-minute window
    (coverage 38 %) fitted on the minutes since a P04 alarm;
  * seeds 0-1, day 14: 'when' failed 13 / 11 PG1 precision constraints;
    stated coverages 0.87-0.94 held 0.71-0.84 on fresh arrivals;
  * seeds 0-1: A9's daily 10:30 page (192.168.3.33, never a member) cut the
    approver's 10:00-11:30 window in two;
  * seed 0, day 14: the 综合部 part of the shared login node stated the window
    before D1 (held-out coverage 0.13 against a stated 0.80)."""
from __future__ import annotations

import numpy as np

from app.engines.behavior import time_window as TW
from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.lib import pwindows as PW

DAY = 86400.0
OFF = 8 * 3600.0
D0 = 20341                                     # a local date index (2025-09-10, a Wednesday)
GA = ["192.168.1.21", "192.168.1.23", "10.168.7.121"]


def _ts(date, minute):
    return date * DAY - OFF + minute * 60.0


def _hist(pts):
    h = np.zeros(PW.SLOTS)
    for p in pts:
        h[int(p[0] // PW.SLOT_MIN) % PW.SLOTS] += p[2] if len(p) > 2 else 1.0
    return h


def _seed3_points():
    """The 综合部 login arrivals of pack O seed 3 up to day 21 (node 21 backed
    off to its ancestor; minutes local)."""
    rows = [(0, 549, GA[0]), (0, 552, GA[2]), (0, 559, GA[1]), (1, 550, GA[0]), (1, 552, GA[2]),
            (1, 553, GA[1])]                                       # before D1 (09:00-09:21)
    for k, d in enumerate((2, 5, 6, 7, 8, 9)):                     # D1: 08:30-08:51
        rows += [(d, 515 + k, GA[0]), (d, 518 + k, GA[1]), (d, 523 + (k % 3), GA[2])]
    rows += [(d, 550, GA[0]) for d in (7, 8, 9)]                   # A2: 09:10, three workdays
    rows += [(8, 185, GA[2])]                                      # A4: 03:05
    return [(float(m), _ts(D0 + d, m), 1.0, s) for d, m, s in rows]


def test_a_stale_regime_and_one_sources_late_logins_form_no_window():
    pts = _seed3_points()
    rec = PW.fit_regime(_hist(pts), float(len(pts)), pts, tz_offset_s=OFF)
    assert len(rec["windows"]) == 1, rec["windows"]
    s, e = rec["windows"][0]
    assert 510 <= s <= 518 and 524 <= e <= 532
    assert rec["coverage"] < 0.85                  # A2 / A4 / the old regime are outside


def test_a_stale_block_does_not_join_the_current_window():
    """Pack O seed 0 (round 4 run 1, day 19): the pre-D1 logins 09:03-09:20 (two
    dates, three members) and A2's 09:10 formed a block 13 minutes after the
    D1 block 08:32-08:50; adjacent active blocks join across < 15 minutes, so
    the node stated '08:32-09:21' (checklist IoU 0.37). A block must persist
    (_persistence) before it joins a window."""
    rows = [(0, 543, GA[1]), (0, 548, GA[2]), (0, 560, GA[0]), (1, 551, GA[1]), (1, 558, GA[0]),
            (1, 559, GA[2]), (2, 513, GA[0]), (2, 529, GA[2]), (2, 529, GA[1]), (5, 512, GA[2]),
            (5, 515, GA[0]), (5, 530, GA[1]), (6, 513, GA[2]), (6, 518, GA[0]), (6, 523, GA[1]),
            (7, 516, GA[0]), (7, 519, GA[2]), (7, 525, GA[1]), (7, 550, GA[0]), (8, 185, GA[2]),
            (8, 518, GA[0]), (8, 526, GA[1]), (8, 528, GA[2]), (8, 550, GA[0]), (9, 512, GA[1]),
            (9, 512, GA[0]), (9, 522, GA[2]), (9, 550, GA[0])]
    pts = [(float(m), _ts(D0 + d, m), 1.0, s) for d, m, s in rows]
    rec = PW.fit_regime(_hist(pts), float(len(pts)), pts, tz_offset_s=OFF)
    assert len(rec["windows"]) == 1 and 510 <= rec["windows"][0][0] <= 514 \
        and 529 <= rec["windows"][0][1] <= 533, rec["windows"]


def test_a_shared_window_that_persists_is_kept():
    """Two departments' login windows at one node, both used every day."""
    pts = []
    for d in range(10):
        pts += [(510.0 + d, _ts(D0 + d, 510 + d), 1.0, GA[k]) for k in range(3)]
        pts += [(600.0 + d, _ts(D0 + d, 600 + d), 1.0, f"192.168.2.1{k}") for k in range(3)]
    rec = PW.fit_regime(_hist(pts), float(len(pts)), pts, tz_offset_s=OFF)
    assert len(rec["windows"]) == 2, rec["windows"]


def test_a_sparse_broad_law_keeps_its_segments():
    """Two bookkeepers' ledger pages, ~9 a day each over 09:20-17:30, 14 dates:
    a sparse segment of a broad law is never 'stale' (finance ledger, pack O;
    a recent-support-only rule cut 14:15-16:10 out of it on day 10)."""
    r = np.random.default_rng(4)
    pts = []
    for d in range(14):
        for src in ("192.168.2.11", "192.168.2.12"):
            for m in r.uniform(560, 1050, 9):
                pts.append((float(m), _ts(D0 + d, float(m)), 1.0, src))
    rec = PW.fit_regime(_hist(pts), float(len(pts)), pts, tz_offset_s=OFF)
    span = sum(b - a for a, b in rec["windows"])
    assert span >= 0.85 * (1050 - 560) and rec["coverage_in"] >= 0.9, rec["windows"]


def test_a_chance_cluster_of_a_flat_day_is_no_window():
    """Pack O seed 1, days 13-14: a 60-s health monitor's reservoir (166
    workday minutes spread over the day) held 8 minutes within 07:16-07:26;
    that block alone was 'active' (3x the background) and the statement said
    '工作日 07:16-07:26 (覆盖 5 %)'. Windows holding a minority of the arrivals
    describe no time-of-day law: all day."""
    r = np.random.default_rng(0)
    pts = [(float(m), _ts(D0 + int(i % 9), float(m)), 1.0, "192.168.9.9") for i, m in enumerate(r.uniform(0, 1440, 156))]
    pts += [(436.0 + k, _ts(D0 + k % 9, 436.0 + k), 1.0, "192.168.9.9") for k in range(10)]
    rec = PW.fit_regime(_hist(pts), float(len(pts)), pts, tz_offset_s=OFF)
    assert rec["windows"] == [[0, 1440]] and rec["all_day"], rec["windows"]


def test_alarm_minutes_spanning_one_day_do_not_replace_the_windows():
    """A 60-s monitor (arrivals all day on 8 dates) with a P04 alarm opened
    20 minutes ago: the arrivals since the alarm do not describe a day."""
    pts = [(float(m), _ts(D0 + d, m), 1.0, "192.168.9.9") for d in range(8) for m in range(0, 1440, 3)]
    t_alarm = _ts(D0 + 7, 1440 - 40)
    rec = PW.fit_regime(_hist(pts), float(len(pts)), pts, tz_offset_s=OFF, since0=t_alarm, provisional0=True)
    assert rec["windows"] == [[0, 1440]] and rec["coverage"] >= 0.99
    assert rec["regime"] == "mixed"


def test_alarm_branch_of_the_engine_fit():
    """The same through TimeWindowEngine.fit_node (a P04 alarm on '@when')."""
    node = _node(range(8), lambda d: [(m, "192.168.9.9") for m in range(0, 1440, 3)])
    t_alarm = _ts(D0 + 7, 1440 - 40)
    now = _ts(D0 + 7, 1439)
    e = TW.TimeWindowEngine().fit_node(node, now, OFF, None, {"when_t0": t_alarm}, None)
    assert e["by_daytype"]["wd"]["windows"] == [[0, 1440]]


def _coverage_of(windows, law_minutes):
    return float(np.mean([PW.in_windows(m, windows) for m in law_minutes]))


def test_stated_coverage_is_not_above_what_fresh_arrivals_show():
    """Sparse arrivals of a broad law (N(15:00, 4 h) clipped to 07:00-23:00,
    the portal's): averaged over 30 samples, the stated coverage exceeds the
    law's mass inside the windows by < 0.04 (in-sample: ~0.08)."""
    r = np.random.default_rng(0)
    law = np.clip(r.normal(900, 240, 20000), 420, 1380)
    gaps_new, gaps_in = [], []
    for rep in range(30):
        pts = []
        for d in range(6):
            for m in np.clip(r.normal(900, 240, 8), 420, 1380):
                pts.append((float(m), _ts(D0 + d, float(m)), 1.0, f"10.60.{rep}.{d}"))
        rec = PW.fit_regime(_hist(pts), float(len(pts)), pts, tz_offset_s=OFF)
        true = _coverage_of(rec["windows"], law)
        gaps_new.append(rec["coverage"] - true)
        gaps_in.append(rec.get("coverage_in", rec["coverage"]) - true)
    assert np.mean(gaps_new) < 0.04, (np.mean(gaps_new), np.mean(gaps_in))
    assert np.mean(gaps_new) < np.mean(gaps_in)


def _node(dates, rows_of, sus=(), members=()):
    nd = PN.Node(1, None, 0, 0, (), _ts(D0, 0))
    nd.when.want_minutes(True, R=4096, seed=0)
    for d in dates:
        for m, src in rows_of(d):
            ts = _ts(D0 + d, m)
            nd.when.update(0, float(m), ts, 1.0, 1.0, src=src)
    for ip in members:
        nd.who.update([ip, None, None, None, None], ip, _ts(D0, 600), 1.0, 1.0)
    for ip in sus:
        nd.who.mark_suspect(ip, _ts(D0 + max(dates), 700))
    return nd


def test_an_outsiders_arrivals_do_not_shape_the_windows():
    """The approver at 10:00-11:30 and 15:00-16:00; an outsider held out as
    suspect by the node's who (A9) adds one 10:30 page a day."""
    r = np.random.default_rng(3)

    def rows(d):
        out = [(float(m), "192.168.2.10") for m in r.uniform(600, 690, 3)]
        out += [(float(m), "192.168.2.10") for m in r.uniform(900, 960, 2)]
        return out + [(630.0, "192.168.3.33")]
    nd = _node(range(10), rows, sus=["192.168.3.33"], members=["192.168.2.10"])
    now = _ts(D0 + 9, 1200)
    pts = TW._points(nd, 0, t=now)
    assert all(p[3] != "192.168.3.33" for p in pts)
    assert len(TW._points(nd, 0)) == len(pts) + 10      # without t: every arrival
    # a member that turned suspect keeps its arrivals
    nd2 = _node(range(10), rows, sus=["192.168.2.10"], members=["192.168.2.10"])
    assert any(p[3] == "192.168.2.10" for p in TW._points(nd2, 0, t=now))


def test_a_groups_part_states_its_current_window():
    """The 综合部 part of a login node shared with another department: the
    part's window follows D1 (09:00-09:21 -> 08:30-08:51 on date 8 of 14)."""
    def rows(d):
        ga = [(float(540 + 3 * k + (d % 4)) if d < 8 else float(512 + 3 * k + (d % 4)), GA[k]) for k in range(3)]
        return ga + [(float(560 + 4 * k + (d % 3)), f"192.168.3.2{k}") for k in range(4)]
    nd = _node(range(14), rows)
    pw = PW.part_when(nd.when, GA, OFF)
    wd = pw["workday"]
    assert len(wd) == 1 and abs(wd[0][0] - 512) <= 2 and abs(wd[0][1] - 522) <= 2, wd


def test_back_off_keeps_a_member_the_ancestor_holds_suspect():
    """Pack O seed 2 (round 4 run 4, day 19): the 综合部 login node (three
    members, created after two pre-D1 dates of its own) backs off to a wider
    login node that never tracked 192.168.1.21 at its IP level and held it out
    as suspect after A2's 09:10 logins. Filtering the node's members by the
    ANCESTOR's suspect test read two of three sources, the back-off fell back
    to the node's own points (2 dates before D1: no regime cut) and the node
    stated the stale 08:31-09:20 (checklist IoU 0.40). Suspicion is the
    member node's business: the ancestor's arrivals of all three members are
    read and the window follows D1."""
    class _T:
        nodes = {}
    net = ("net.src", 0, frozenset(GA), False)
    anc = PN.Node(1, None, 0, 0, (), _ts(D0, 0))
    anc.when.want_minutes(True, R=4096, seed=0)
    kid = PN.Node(2, 1, 1, 0, (net,), _ts(D0 + 2, 0))
    kid.when.want_minutes(True, R=4096, seed=0)
    _T.nodes = {1: anc, 2: kid}
    dates = [0, 1, 2, 3, 6, 7, 8, 9, 10, 13, 14, 15]          # D1 from date 6 (index 4)
    for i, d in enumerate(dates):
        for k, ip in enumerate(GA):
            m = 542.0 + 6 * k + (d % 3) if i < 4 else 512.0 + 6 * k + (d % 3)
            ts = _ts(D0 + d, m)
            anc.when.update(0, m, ts, 1.0, 1.0, src=ip)
            if d >= 2:
                kid.when.update(0, m, ts, 1.0, 1.0, src=ip)
        for j in range(4):                                     # the other department
            m = 520.0 + 9 * j + (d % 4)
            anc.when.update(0, m, _ts(D0 + d, m), 1.0, 1.0, src=f"192.168.3.2{j}")
            anc.who.update([f"192.168.3.2{j}", None, None, None, None], f"192.168.3.2{j}",
                           _ts(D0 + d, m), 1.0, 1.0)
        if i >= 9:                                             # A2: 09:10 from .21
            for nd in (anc, kid):
                nd.when.update(0, 550.0, _ts(D0 + d, 550), 1.0, 1.0, src=GA[0])
    for ip in GA:
        kid.who.update([ip, None, None, None, None], ip, _ts(D0 + 2, 600), 1.0, 1.0)
    now = _ts(D0 + 15, 1200)
    anc.who.mark_suspect(GA[0], now - 3600.0)
    kid.who.mark_suspect(GA[0], now - 3600.0)
    assert GA[0] not in anc.who.levels[0] and GA[0] in kid.who.levels[0]
    pts, a = TW._backoff(_T, kid, 0, TW._points(kid, 0, t=now), now)
    assert a == 1 and {p[3] for p in pts} == set(GA)
    e = TW.TimeWindowEngine().fit_node(kid, now, OFF, None, {}, _T)
    wd = e["when"]["workday"]
    assert len(wd) == 1 and 510 <= wd[0][0] <= 514 and 524 <= wd[0][1] <= 532, wd


def test_violating_arrivals_do_not_shape_the_windows():
    """Pack O seed 1 (round 4 run 4, day 21): A10's ~150 comment posts from 48
    unknown addresses within 09:00-11:40 of the last (non-work) day - content
    violations P03 judged at p <= 1e-3, learned by P04 at full weight - passed
    regime_cut as a coordinated change of the nonworkday law, and portal POST
    /comment stated 08:55-11:40 against the truth 07:05-23:12. Rows in P06's
    violation ledger (content_bounds.point_ledger) leave the arrivals."""
    r = np.random.default_rng(11)
    nd = PN.Node(1, None, 0, 0, (), _ts(D0, 0))
    nd.when.want_minutes(True, R=4096, seed=0)
    for d in (3, 4, 10, 11, 17):                                # five non-work dates
        for m in r.uniform(425, 1390, 30):
            nd.when.update(1, float(m), _ts(D0 + d, m), 1.0, 1.0, src=f"10.60.{d}.{int(m) % 200}")
    burst = set()
    for i, m in enumerate(np.linspace(540, 700, 150)):
        ts, src = _ts(D0 + 18, m), f"203.0.113.{i % 48}"
        nd.when.update(1, float(m), ts, 1.0, 1.0, src=src)
        burst.add((round(ts, 3), src))
    now = _ts(D0 + 18, 1400)
    raw = TW.TimeWindowEngine().fit_node(nd, now, OFF, None, {}, None)["when"]["nonworkday"]
    clean = TW.TimeWindowEngine().fit_node(nd, now, OFF, None, {}, None, burst)["when"]["nonworkday"]
    assert sum(e - s for s, e in clean) >= 600, clean           # the day-long law
    assert sum(e - s for s, e in raw) < 300, raw                  # the burst alone


def test_p06_ledger_records_violating_arrivals_for_p09():
    from app.core.engine import Context
    from app.core.store import MetricStore
    from app.engines.behavior import content_bounds as CB
    from app.engines.behavior.lib import pevent as EV
    from pcontent_oracle import OracleLearner
    cfg = {"progressive": {"enabled": True}, "grain_mode": "tick", "strict": True}
    t0 = 1_788_220_800.0
    store = MetricStore()
    orc = OracleLearner(store, "oa", t0, ["POST /login"], config=cfg)
    nid = orc.node_for("POST /login")
    eng = CB.ContentBoundsEngine()
    for d in range(3):
        T = t0 + d * DAY + 12 * 3600
        bb = EV.BatchBuilder("oa")
        for i in range(6):
            bb.add(T - 3600 + i * 7.0, f"192.168.1.{20 + i}", {"http.route": "POST /login",
                                                              "body.len": 1500.0 + i}, 1.0)
        b = bb.build(T - 43200, T)
        store.add_batch("oa", EV.EVT_BATCH, T, b)
        rr = np.arange(b.n, dtype=np.int32)
        vt = np.zeros(b.n)
        if d == 2:
            vt[-1] = 4.0
        cols = {"leaf": EV.Col(rr, np.full(b.n, float(nid))), "vtype": EV.Col(rr, vt)}
        store.add_batch("oa", EV.PAT_ASSIGN, T, b.aligned(cols, {"kind": 0, "tree_key": "oa"}))
        orc.learn(b)
        eng.safe_run(Context(store=store, now=T + 3600, window_s=60, config=dict(cfg)))
    T = t0 + 2 * DAY + 12 * 3600
    assert CB.point_ledger(store, "oa", 0) == {(round(T - 3600 + 5 * 7.0, 3), "192.168.1.25")}


def test_a_groups_part_leaves_violating_arrivals_out():
    def rows(d):
        return [(float(512 + 3 * k + (d % 4)), GA[k]) for k in range(3)]
    nd = _node(range(10), rows)
    burst = set()
    for d in (7, 8, 9):
        for j in range(6):
            m = 700.0 + j
            ts = _ts(D0 + d, m)
            nd.when.update(0, m, ts, 1.0, 1.0, src=GA[0])
            burst.add((round(ts, 3), GA[0]))
    raw = PW.part_when(nd.when, GA, OFF)["workday"]
    clean = PW.part_when(nd.when, GA, OFF, drop=burst)["workday"]
    assert len(clean) == 1 and abs(clean[0][0] - 512) <= 2 and clean[0][1] <= 524, clean
    assert raw != clean
