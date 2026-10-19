"""Evaluator round 8: the held-out binding check accepts a renamed value's
predecessor exactly as PG1's login-binding check does (EV8-1), and P10's
held edges count as stated workflow edges (EV8-2)."""
from __future__ import annotations

import datetime as dt

import numpy as np

from app.eval import pmetrics as M

START = dt.date(2025, 9, 1)            # a Monday
X = "10.0.0.1"
ATTR = "body.kv.username"


def _iso(day: int) -> str:
    return (START + dt.timedelta(days=day - 1)).isoformat()


def _seg(tid, vf, vt, user):
    return {"tid": tid, "lineage": "L", "activity": "GA.oa.login", "step": 0,
            "system": "oa", "systems": ["oa"], "method": "POST", "route": "/login",
            "valid_from_day": vf, "valid_to_day": vt,
            "who": {"level": "ip", "members": [X]},
            "windows": {"workday": [[540.0, 600.0]], "nonworkday": []},
            "content": {}, "bindings": {ATTR: {X: user}},
            "gen": {"members": [X], "usernames": {X: user},
                    "fields": {"username": {"kind": "bound"}}}}


def _truth(new_days):
    """mike on days 1-12, renamed mike.w from day 13 (D2); X logs in under
    the new name on `new_days` only."""
    rows = [_seg("S0", 1, 13, "mike"), _seg("S1", 13, 22, "mike.w")]
    opp = {"S0": {_iso(d): {X: 3} for d in range(1, 13)},
           "S1": {_iso(d): {X: 3} for d in new_days}}
    workday = [(START + dt.timedelta(days=i)).weekday() < 5 for i in range(21)]
    return M.PTruth({"pattern_truth": rows,
                     "days": {"start": START.isoformat(), "n_days": 21, "workday": workday,
                              "day_start": [float(i * 86400) for i in range(22)]},
                     "opportunities": opp, "system_truth": {"families": {}}})


def _stmt(user):
    raw = {"id": "p:oa:1", "pattern_id": "p:oa:1", "state": "confirmed", "confidence": 0.95,
           "view": "system",
           "evidence": {"system": "oa", "method": "POST", "route": "/login",
                        "who": {"level": "ip", "items": [X], "closed": True, "U": 0.0},
                        "bindings": {ATTR: {"table": {X: user}, "LB": {X: 0.9}}}}}
    return M.LStmt(raw, "oa", {})


def _bind_hold(pt, s, day):
    h = M.holdout_check(s, pt.valid_at_day(day), pt, np.random.default_rng(0), 100, day)
    return {c[0]: c[2] for c in h["constraints"]}[f"{ATTR}.bind.{X}"]


def test_held_out_binding_accepts_the_predecessor_until_the_new_value_is_learnable():
    """Pack O seed 1 / O-real seed 1, day 14: '10.168.7.121 -> mike' (D2 renamed
    the user on day 13, and .121 had not logged in since) held 0.0 on held-out
    events of the post-rename segment, while PG1's login-binding check accepts
    'mike' until .121 has used 'mike.w' REBIND_N times on REBIND_DATES dates."""
    pt = _truth(new_days=[15, 16, 17, 18, 19])               # first login under mike.w on day 15
    assert M.accepted_values(pt, pt.by_tid["S1"], ATTR, X, 14) == {"mike", "mike.w"}
    assert _bind_hold(pt, _stmt("mike"), 14) == 1.0           # not learnable yet: accepted
    assert _bind_hold(pt, _stmt("mike.w"), 14) == 1.0
    # once the new value is learnable, the old one fails
    assert M.accepted_values(pt, pt.by_tid["S1"], ATTR, X, 19) == {"mike.w"}
    assert _bind_hold(pt, _stmt("mike"), 19) == 0.0
    # a value never bound to X fails at any time
    assert _bind_hold(pt, _stmt("jack"), 14) == 0.0


def test_login_bindings_and_held_out_check_share_one_acceptance_rule():
    pt = _truth(new_days=[15, 16, 17, 18, 19])
    snap = {"systems": {}}
    # login_bindings reads the newest valid segment; accepted_values is its rule
    for day in (14, 16, 19):
        acc = M.accepted_values(pt, pt.by_tid["S1"], ATTR, X, day)
        assert ("mike" in acc) == (day < 16)                  # 3 events/date: 6 on 2 dates by day 16
    assert M.login_bindings(snap, pt, {}, 14) == (0, 1)


def _edge_stmt(dep, held=None, dep_upper=None):
    e = {"from": "GET /a", "to": "GET /b", "dep": dep, "band": [0.0, 60.0]}
    if held is not None:
        e["held"] = held
        e["dep_upper"] = dep_upper
    raw = {"id": "p:oa:2", "pattern_id": "p:oa:2", "state": "confirmed", "confidence": 0.9,
           "view": "system",
           "evidence": {"system": "oa", "method": "GET", "route": "/b",
                        "who": {"level": "ip", "items": [X], "closed": True, "U": 0.0},
                        "workflow": [e]}}
    return M.LStmt(raw, "oa", {})


def test_an_edge_p10_holds_under_hysteresis_counts_at_its_upper_bound():
    """Round 8 (P10 hysteresis): an accepted edge is retired only when the 95 %
    upper bound of its dependency falls below 0.8; while P10 holds it, it is
    stated with 'held' and 'dep_upper'. The scorer's dep >= 0.8 read the point
    estimate and discarded edges P10 still states (pack O seed 1
    GA.oa.approvals#1 on days 8-21)."""
    plain = M._edges([_edge_stmt(0.76)])
    held = M._edges([_edge_stmt(0.76, held=True, dep_upper=0.88)])
    assert plain[0][3] == 0.76
    assert held[0][3] == 0.88
    # a held edge whose upper bound is also low stays below the threshold
    assert M._edges([_edge_stmt(0.5, held=True, dep_upper=0.7)])[0][3] == 0.7
