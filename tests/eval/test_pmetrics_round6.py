"""Evaluator round 6: PG1 precision on claims whose truth behaviour has been
observable for >= 1 workday (lead decision), and per-system judging of a
statement of a system that has truth rows of its own (OI-1)."""
from __future__ import annotations

import datetime as dt

import numpy as np

from app.eval import pmetrics as M

START = dt.date(2025, 9, 1)            # a Monday


def _iso(day: int) -> str:
    return (START + dt.timedelta(days=day - 1)).isoformat()


def _row(tid, system, ips, vf=1, vt=99, route="/crm/customer", wd=True, lineage=None):
    return {"tid": tid, "lineage": lineage or tid, "system": system, "systems": [system],
            "method": "GET", "route": route, "valid_from_day": vf, "valid_to_day": vt,
            "who": {"level": "grp", "members": list(ips)},
            "windows": {"workday": [[540.0, 1080.0]] if wd else [],
                        "nonworkday": [] if wd else [[600.0, 900.0]]},
            "content": {}, "gen": {"members": list(ips)}}


def _truth(rows, n_days=21, opp=None, families=None):
    workday = [(START + dt.timedelta(days=i)).weekday() < 5 for i in range(n_days)]
    return M.PTruth({"pattern_truth": rows,
                     "days": {"start": START.isoformat(), "n_days": n_days, "workday": workday,
                              "day_start": [float(i * 86400) for i in range(n_days + 1)]},
                     "opportunities": opp or {},
                     "system_truth": {"families": families or {}}})


def _stmt(system, ips, route="/crm/customer", conf=0.9):
    raw = {"id": f"p:{system}:1", "pattern_id": f"p:{system}:1", "state": "confirmed",
           "confidence": conf, "view": "system",
           "evidence": {"system": system, "method": "GET", "route": route,
                        "who": {"level": "ip", "items": list(ips), "closed": True, "U": 0.05}}}
    return M.LStmt(raw, system, {})


def _opp(ips, days):
    return {_iso(d): {ip: 3 for ip in ips} for d in days}


def test_a_branch_statement_is_judged_on_its_own_systems_rows():
    """O-real: crm-01..12 form the truth family crm-br, each branch with its
    own users. A statement of crm-01's own tree was judged on held-out events
    of every branch, so its who held ~1/12 (stated 0.83, held 0.57)."""
    b1 = [f"192.168.111.{i}" for i in range(1, 6)]
    b2 = [f"192.168.111.{i}" for i in range(6, 11)]
    rows = [_row("t1", "crm-01", b1), _row("t2", "crm-02", b2)]
    opp = {"t1": _opp(b1, range(1, 15)), "t2": _opp(b2, range(1, 15))}
    pt = _truth(rows, opp=opp, families={"crm-br": ["crm-01", "crm-02"]})
    s = _stmt("crm-01", b1)
    assert M._sys_match("crm-01", rows[0], pt)
    assert not M._sys_match("crm-01", rows[1], pt)
    h = M.holdout_check(s, pt.valid_at_day(14), pt, np.random.default_rng(0), 200, 14)
    assert h["supported"] and h["ok"], h["constraints"]
    assert h["tids"] == ["t1"]
    # a statement named after the truth family (no rows of its own) still
    # reads every member's rows
    assert M._sys_match("crm-br", rows[1], pt)


def test_precision_judges_a_claim_only_once_its_behaviour_was_observable_for_a_workday():
    """Lead decision (round 6): a segment opened by a drift on Saturday (day
    6) has no workday of its own by Sunday's snapshot; a claim judged on it
    describes behaviour no learner could have observed and is not part of the
    headline precision. It counts again after the next workday's events."""
    old = ["10.0.0.1", "10.0.0.2"]
    new = ["10.0.0.7", "10.0.0.8"]
    rows = [_row("a0", "oa", old, vf=1, vt=6, lineage="L"), _row("a1", "oa", new, vf=6, lineage="L")]
    opp = {"a0": _opp(old, range(1, 6)), "a1": _opp(new, range(6, 15))}
    pt = _truth(rows, opp=opp)
    s = _stmt("oa", old)                        # the pre-drift who
    for day, judged in ((7, False), (8, True)):    # day 7 Sunday, day 8 Monday
        h = M.holdout_check(s, pt.valid_at_day(day), pt, np.random.default_rng(1), 200, day)
        assert h["supported"] and not h["ok"]
        assert M.claim_observable(s, h, pt, day) is judged, day
    # an absence: the route of a segment that ended is observable as absent
    # only after a workday without it
    gone = [_row("g0", "oa", old, vf=1, vt=6, route="/approval/list")]
    pt2 = _truth(gone, opp={"g0": _opp(old, range(1, 6))})
    s2 = _stmt("oa", old, route="/approval/list")
    for day, judged in ((7, False), (8, True)):
        h = M.holdout_check(s2, pt2.valid_at_day(day), pt2, np.random.default_rng(2), 50, day)
        assert not h["supported"]
        assert M.claim_observable(s2, h, pt2, day) is judged, day
    # a weekend-only segment is observable on a day of its own day type
    wk = [_row("w0", "oa", old, vf=1, wd=False, route="/batch")]
    pt3 = _truth(wk, opp={"w0": _opp(old, [6])})
    assert pt3.row_observable(wk[0], 6) and not pt3.row_observable(wk[0], 5)


def test_pg1_reports_observable_precision_as_headline_and_all_claims_beside_it():
    old = ["10.0.0.1", "10.0.0.2"]
    new = ["10.0.0.7", "10.0.0.8"]
    other = ["10.0.1.1", "10.0.1.2"]
    rows = [_row("a0", "oa", old, vf=1, vt=6, lineage="L"), _row("a1", "oa", new, vf=6, lineage="L"),
            _row("b0", "oa", other, route="/docs")]
    opp = {"a0": _opp(old, range(1, 6)), "a1": _opp(new, range(6, 15)), "b0": _opp(other, range(1, 15))}
    pt = _truth(rows, opp=opp)
    st = [{"id": "p:oa:1", "pattern_id": "p:oa:1", "state": "confirmed", "confidence": 0.9,
           "view": "system", "evidence": {"system": "oa", "method": "GET", "route": "/crm/customer",
                                          "who": {"level": "ip", "items": old, "closed": True, "U": 0.05}}},
          {"id": "p:oa:2", "pattern_id": "p:oa:2", "state": "confirmed", "confidence": 0.9,
           "view": "system", "evidence": {"system": "oa", "method": "GET", "route": "/docs",
                                          "who": {"level": "ip", "items": other, "closed": True, "U": 0.05}}}]
    snap = {"systems": {"oa": {"model.pviews": {"view": "system", "statements": st}}}}
    r = M.pg1_snapshot(snap, pt, {}, 7, seed=0, precision_n=100)
    assert r["precision"] == 1.0 and r["n_precision"] == 1
    assert r["precision_all"] == 0.5 and r["n_precision_all"] == 2 and r["n_unobservable"] == 1
    assert sorted(c[3] for c in r["calib"]) == [0, 1]
