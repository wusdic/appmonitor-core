"""P09 learning hygiene (docs/lib3/progressive.md §6.13, §6.9.2-§6.9.3):
damped arrivals do not shape a window, a young who-split node reads its
sources' arrivals from its ancestor's reservoir, and stationary arrivals
show no regime change.

Pack O (oa, seed 0, day 21), before: the 综合部 login node created by a late
split (day ~18) had no minute reservoir of its own and was fitted in slot mode
on its H_m histogram: 08:30-09:15, which kept A2's 09:10 logins and gave IoU
0.47 against the truth 08:30-08:51 (the window after drift D1)."""
from __future__ import annotations

import numpy as np

from helpers import ctx, make_store
from temporal_sim import CFG, DAY, MON, OracleTree, is_workday

from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pwindows as PW
from app.engines.behavior.time_window import TimeWindowEngine
from test_p09_time_window import FIN, GA, LOGIN, SALES, entry, run


def test_damped_arrivals_do_not_widen_the_window():
    st = make_store()
    ot = OracleTree(st, "oa", [LOGIN], {LOGIN: [GA]})
    r = np.random.default_rng(7)
    ev = []
    for d in range(21):
        day = MON + d * DAY
        if is_workday(day):
            ev += [(day + r.uniform(510, 531) * 60, ip, 1.0) for ip in GA]
            if d >= 14:      # a borrowed credential at 09:10, learned with outlier damping (0.1)
                ev += [(day + 550 * 60 + k, GA[0], 0.1) for k in range(2)]
    ev.sort()
    eng = TimeWindowEngine()
    now, i = MON, 0
    while now < MON + 21 * DAY:
        t1 = now + 3600.0
        while i < len(ev) and ev[i][0] <= t1:
            ot.learn(ev[i][0], ev[i][1], LOGIN, mass=ev[i][2], evidence=ev[i][2])
            i += 1
        ot.apply_wants()
        eng.safe_run(ctx(st, t1, window_s=3600.0, config=CFG), None)
        now = t1
    nid = ot.group_node[(LOGIN, 0)]
    w = entry(st, nid)["by_daytype"]["wd"]
    assert len(w["windows"]) == 1 and abs(w["windows"][0][0] - 510) <= 2 \
        and abs(w["windows"][0][1] - 531) <= 2, w["windows"]
    # the same arrivals at full weight are a (second) window: the weights decide
    pts = _pts(ot, nid)
    assert any(abs(m - 550) < 1 for m, _ in pts)
    assert len(PW.fit_daytype(ot.tree.nodes[nid].when.hist[0], 60.0, pts)["windows"]) == 2


def _pts(ot, nid):
    return [(float(it[1]), float(t)) for it, _w, t in ot.tree.nodes[nid].when.res.items() if it[0] == 0]


def test_young_split_child_reads_its_sources_from_the_ancestor_reservoir():
    st = make_store()
    ot = OracleTree(st, "oa", [LOGIN])
    r = np.random.default_rng(8)
    ev = []
    for d in range(21):
        day = MON + d * DAY
        if is_workday(day):
            ev += [(day + r.uniform(510, 531) * 60, ip, LOGIN) for ip in GA]
            ev += [(day + r.uniform(545, 570) * 60, ip, LOGIN) for ip in FIN]
            ev += [(day + r.uniform(500, 600) * 60, ip, LOGIN) for ip in SALES]
    eng, now = run(ot, st, ev, 20)
    # P04 splits the route node by source on day 20 and seeds the child's who summary (M5)
    route = ot.route_node[LOGIN]
    sp = ot.tree.split(route, "net.src", 0, [list(GA)], now)
    child = ot.tree.nodes[sp.children[0]]
    for ip in GA:
        child.who.update([ot.hier.gen("net.src", l, ip) for l in range(5)], ip, now, 10.0, 10.0)
    for k in range(3):                                         # its first own arrivals
        ot.learn(now + 3600 + k, GA[k], LOGIN)
    eng.safe_run(ctx(st, now + 7 * 3600, window_s=3600, config=CFG), None)
    rec = entry(st, child.id)["by_daytype"]["wd"]
    assert rec["res"] == "minute" and rec["backoff"] == route
    assert len(rec["windows"]) == 1 and abs(rec["windows"][0][0] - 510) <= 3 \
        and abs(rec["windows"][0][1] - 531) <= 3, rec["windows"]


def test_no_regime_change_in_stationary_arrivals():
    hits = 0
    for seed in range(20):
        r = np.random.default_rng(100 + seed)
        pts = []
        for d in range(21):
            day = MON + d * DAY
            if is_workday(day):
                for k in range(6):
                    t = day + r.uniform(500, 600) * 60
                    pts.append(((t - MON) / 60.0 % 1440, t, 1.0, f"ip{k}"))
        h = np.zeros(96)
        for m, *_ in pts:
            h[int(m // 15)] += 1
        if PW.regime_cut(pts, h, tz_offset_s=8 * 3600) is not None:
            hits += 1
    assert hits <= 1


def test_department_schedule_change_accepted_after_three_workdays():
    """§6.9.2: a coordinated time change (3 IPs) is accepted after 3 workdays
    (pack O D1: the 综合部 login moved 09:00-09:21 -> 08:30-08:51 on day 12;
    the change needed 4 workdays when 10 arrivals were required after it)."""
    st = make_store()
    ot = OracleTree(st, "oa", [LOGIN], {LOGIN: [GA]})
    r = np.random.default_rng(11)
    ev = []
    for d in range(17):
        day = MON + d * DAY
        if is_workday(day):
            lo, hi = (540, 561) if d < 14 else (510, 531)
            ev += [(day + r.uniform(lo, hi) * 60, ip, LOGIN) for ip in GA]
    run(ot, st, ev, 17)                                   # through Wed, day 16 (3 new workdays)
    rec = entry(st, ot.group_node[(LOGIN, 0)])["by_daytype"]["wd"]
    assert rec["regime"] == "new" and rec["change"]["accepted"], rec.get("change")
    assert len(rec["windows"]) == 1 and abs(rec["windows"][0][0] - 510) <= 6 \
        and abs(rec["windows"][0][1] - 531) <= 6, rec["windows"]
