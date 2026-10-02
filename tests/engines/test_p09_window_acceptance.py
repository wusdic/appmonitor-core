"""P09: a time window only one source supports at a node of several sources is
accepted only after REGIME_SINGLE_DATES dates (lib/pwindows._accepted, the
§6.9.2 single-source rule regime_cut already applies to a change).

Pack O, A2 (192.168.1.21 logging in as rose at 09:10, days 17-21): P03
flagged the logins and B28 held the source, but the held rows were released
and learned at full weight, so the 综合部 login node stated
'工作日 08:32–08:51、09:10–09:11' (seed 0 and others, round 2): an anomaly
formed a window segment of the profile."""
from __future__ import annotations

import numpy as np

from helpers import ctx, make_store
from temporal_sim import CFG, DAY, MON, OracleTree, is_workday

from app.engines.behavior.lib import pwindows as PW
from app.engines.behavior.time_window import TimeWindowEngine
from test_p09_time_window import GA, LOGIN, entry


def _run(extra_days, src=None, sources=GA, days=21):
    st = make_store()
    ot = OracleTree(st, "oa", [LOGIN], {LOGIN: [sources]})
    r = np.random.default_rng(7)
    ev = []
    wd = 0
    for d in range(days):
        day = MON + d * DAY
        if not is_workday(day):
            continue
        ev += [(day + r.uniform(510, 531) * 60, ip, 1.0) for ip in sources]
        if wd in extra_days:            # an extra arrival at 09:10, learned at full weight
            ev.append((day + 550 * 60 + r.uniform(0, 30), src or sources[0], 1.0))
        wd += 1
    ev.sort()
    eng = TimeWindowEngine()
    now, i = MON, 0
    while now < MON + days * DAY:
        t1 = now + 3600.0
        while i < len(ev) and ev[i][0] <= t1:
            ot.learn(ev[i][0], ev[i][1], LOGIN, mass=ev[i][2], evidence=ev[i][2])
            i += 1
        ot.apply_wants()
        eng.safe_run(ctx(st, t1, window_s=3600.0, config=CFG), None)
        now = t1
    nid = ot.group_node[(LOGIN, 0)]
    return entry(st, nid)["by_daytype"]["wd"]


def test_one_sources_three_dates_at_another_time_form_no_window():
    # A2: three workdays among the eight the node's arrivals cover (after D1's
    # regime change the window was fitted on ~20 recent arrivals: 3 of them A2's)
    w = _run(extra_days={5, 6, 7}, days=10)
    assert len(w["windows"]) == 1 and abs(w["windows"][0][0] - 510) <= 2 \
        and abs(w["windows"][0][1] - 531) <= 2, w["windows"]
    assert w["coverage"] < 1.0                           # the arrivals are outside, not ignored


def test_a_persisting_single_source_habit_is_accepted():
    w = _run(extra_days=set(range(3, 10)), days=14)     # seven workdays at 09:10
    assert len(w["windows"]) == 2 and any(abs(s - 550) <= 2 for s, _ in w["windows"]), w["windows"]


def test_two_sources_at_another_time_form_a_window():
    st_days = {5, 6, 7}
    w1 = _run(extra_days=st_days, src=GA[0], days=10)
    # the same three dates by two sources (a coordinated change): a window
    pts = [(510.0 + k, MON + d * DAY, 1.0, ip) for d in range(10) for k, ip in enumerate(GA)]
    pts += [(550.0, MON + d * DAY + 1, 1.0, ip) for d in (7, 8, 9) for ip in GA[:2]]
    h = np.zeros(PW.SLOTS)
    for m, *_ in pts:
        h[int(m // PW.SLOT_MIN)] += 1
    rec = PW.fit_daytype(h, float(len(pts)), pts)
    assert len(rec["windows"]) == 2, rec["windows"]
    assert len(w1["windows"]) == 1


def test_a_single_source_node_keeps_its_second_window():
    pts = [(510.0 + (d % 5), MON + d * DAY, 1.0, "192.168.2.10") for d in range(10)]
    pts += [(900.0 + (d % 3), MON + d * DAY + 7 * 3600, 1.0, "192.168.2.10") for d in (7, 8, 9)]
    h = np.zeros(PW.SLOTS)
    for m, *_ in pts:
        h[int(m // PW.SLOT_MIN)] += 1
    rec = PW.fit_daytype(h, float(len(pts)), pts)
    assert len(rec["windows"]) == 2, rec["windows"]
