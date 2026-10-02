"""P10: route renames are adopted (lib/pdfg.detect_renames / adopt_renames).

Pack O, D3 (day 14): the approval pages of 192.168.1.21 moved from
/approval/... to /flow/.... The first /flow/list row was a new action for
P03 (p_novel 1.4e-4), the finding opened an incident and B28 quarantined the
source, so P10 - which counts trusted rows only - never saw the new routes, its
dictionary never held them and P03 kept flagging them every day: the profile
never adopted the rename (PG5 D3 0/5 seeds, the approval statements stale on
day 21). P10's route ledger now sees every event, and a new route that
replaced an established one of the same sources (disjoint time, one literal
path segment changed, same workflow position) for RENAME_DATES dates takes the
old action's id."""
from __future__ import annotations

import numpy as np

from helpers import make_store
from temporal_sim import DAY, MON, is_workday
from test_p10_workflow import APPROVE, DOC, DOCS, HOME, ITEM, LIST, LOGIN, R, edges, model, run

from app.engines.behavior.lib import pdfg as DF

FLIST, FITEM, FAPPROVE = R("GET", "/flow/list"), R("GET", "/flow/{num}"), R("POST", "/flow/{num}/approve")
ADMIN = R("GET", "/admin/list")
GA1 = "192.168.1.21"


def world(days, switch_day, seed=0, probe=False):
    r = np.random.default_rng(seed)
    ev = []
    for d in range(days):
        day = MON + d * DAY
        if not is_workday(day):
            continue
        renamed = d >= switch_day
        lst, item, appr = (FLIST, FITEM, FAPPROVE) if renamed else (LIST, ITEM, APPROVE)
        t = day + r.uniform(540, 561) * 60
        ev.append((t, GA1, LOGIN, {}))
        t += r.uniform(5, 30)
        ev.append((t, GA1, HOME, {}))
        for route in (lst, item, appr, item, appr):
            t += r.uniform(60, 360)
            ev.append((t, GA1, route, {}))
        if probe and d >= switch_day:
            ev.append((t + 600, "192.168.3.9", ADMIN, {}))      # a probe of a look-alike page
        for k in range(10):
            t2 = day + r.uniform(570, 990) * 60
            ev.append((t2, f"192.168.3.{k}", DOCS, {}))
            ev.append((t2 + r.uniform(5, 60), f"192.168.3.{k}", DOC, {}))
    return ev


def test_rename_shape_one_literal_segment():
    assert DF.rename_shape(LIST, FLIST) == (0, "approval", "flow")
    assert DF.rename_shape(APPROVE, FAPPROVE) == (0, "approval", "flow")
    assert DF.rename_shape(APPROVE, R("POST", "/flow/{num}/sign")) is None   # two segments differ
    assert DF.rename_shape(LIST, R("POST", "/flow/list")) is None
    assert DF.rename_shape(ITEM, R("GET", "/approval/list")) is None   # a placeholder moved


def test_quarantined_source_rename_is_adopted_after_two_dates():
    st = make_store()
    switch = 14                                        # a Monday (MON + 14 days)

    def quarantine(s, t):
        # P03's new-action finding on the first renamed page holds the source (B28)
        # until its incident closes, after the findings stopped (the adoption)
        q = MON + switch * DAY + 9.3 * 3600 <= t < MON + (switch + 2) * DAY + 12 * 3600
        s.add_vec("oa", GA1, "behavior.quarantine", t, [1.0 if q else 0.0])
    run(st, world(25, switch), 25, hooks=[quarantine])
    m = model(st)
    acts = m.state.acts
    assert acts.id_of(FLIST) == acts.id_of(LIST) is not None
    assert acts.id_of(FITEM) == acts.id_of(ITEM) and acts.id_of(FAPPROVE) == acts.id_of(APPROVE)
    assert m["renamed"][FLIST]["from"] == LIST and m["renamed"][FLIST]["sources"] == [GA1]
    # the mined workflow reads the new names
    assert (FLIST, FITEM) in edges(st)
    # P03's view: the renamed page is a known action with its old transitions
    sc = DF.seq_scores(m, DF.STAR, HOME, FLIST, 0, MON + 25 * DAY)
    assert sc["b"] == acts.id_of(LIST) and not sc["p_trans"] < 0.1
    succ = DF.successors(m.state, DF.STAR, acts.id_of(HOME), MON + 25 * DAY)
    assert set(succ) == {acts.id_of(FLIST)}                  # home -> (old and new) list: one action


def test_a_lookalike_page_while_the_old_one_is_used_is_not_a_rename():
    st = make_store()
    # the old pages keep being used (no switch) while another source probes /admin/list
    ev = world(18, 99) + [(MON + d * DAY + 15 * 3600, GA1, ADMIN, {}) for d in range(14, 18)]
    run(st, ev, 18)
    m = model(st)
    assert ADMIN not in m["renamed"] and m.state.acts.id_of(ADMIN) != m.state.acts.id_of(LIST)
    # one date only: not yet confirmed
    st2 = make_store()
    run(st2, world(16, 15), 16)                        # day 15 = Tuesday: one renamed date
    assert FLIST not in model(st2)["renamed"]


def test_successor_candidate_for_p03_before_confirmation():
    st = make_store()
    run(st, world(15, 14), 15)                         # day 14: the first renamed date
    m = model(st)
    today = DF.EPOCH_ORD + int((MON + 14 * DAY + 8 * 3600) // DAY)
    assert DF.successor_candidate(m, FLIST, GA1, today + 1) == LIST
    assert DF.successor_candidate(m, FLIST, "192.168.3.1", today + 1) is None   # not its route
    # a look-alike page while the old one is still used today is no candidate
    st2 = make_store()
    run(st2, world(15, 99), 15)
    assert DF.successor_candidate(model(st2), ADMIN, GA1, today) is None
