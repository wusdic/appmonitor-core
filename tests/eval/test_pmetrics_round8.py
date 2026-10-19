"""(round 8, harness owner) PG2 paired confidence judged against its own
sampling noise, and prospective held-out events."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.eval import pmetrics as M


def _pt(n_pat: int, n_days: int = 21) -> M.PTruth:
    rows = [{"tid": f"P{i:02d}", "valid_from_day": 1, "valid_to_day": n_days + 1}
            for i in range(n_pat)]
    return M.PTruth({"pattern_truth": rows, "days": {"n_days": n_days}})


def _day(conf):
    return {"recall": 0.9, "recall_by_period": {}, "mean_depth": 1.0, "ece": 0.05,
            "stmt_conf": {f"P{i:02d}": c for i, c in enumerate(conf)}, "who_U": {}}


def test_sign_test_exact_values():
    assert M.sign_test_less([-0.1] * 5) == (5, 0, pytest.approx(1 / 32))
    assert M.sign_test_less([-0.1, 0.0, 0.1]) == (1, 1, pytest.approx(0.75))
    assert M.sign_test_less([0.0, 1e-12]) == (0, 0, 1.0)
    n_neg, n_pos, p = M.sign_test_less([-1] * 10 + [1] * 2)
    assert (n_neg, n_pos) == (10, 2) and p == pytest.approx(79 / 4096)


def test_saturated_confidence_noise_does_not_fail_pg2():
    """Round 7, pack O seed 1 days 5 -> 7: 12 unchanged patterns at a stated
    0.95-0.98, median paired change -0.0037, about as many gaining as losing.
    The strict median >= 0 failed it; the sign test does not."""
    r = np.random.default_rng(7)
    a = 0.96 + 0.01 * r.standard_normal(12)
    d = np.array([-0.0040, -0.0037, -0.0037, -0.0050, -0.0021, -0.0060,
                  0.0030, 0.0045, 0.0012, 0.0080, 0.0026, 0.0009])
    pg1 = {5: _day(a), 7: _day(a + d), 10: _day(a + d + 0.01)}
    out = M.pg2_convergence(pg1, _pt(12))
    step = out["conf_steps"][0]
    assert step[:3] == [5, 7, 12] and step[3] < 0 and step[4:6] == [6, 6]
    assert out["conf_nondecreasing_strict"] is False        # the round-7 rule failed it
    assert out["conf_nondecreasing"] is True


def test_a_consistent_loss_still_fails_pg2():
    """Most unchanged patterns losing confidence between snapshot days is a
    decrease beyond noise, however small each change."""
    a = np.full(12, 0.97)
    d = np.array([-0.004] * 10 + [0.002] * 2)
    pg1 = {5: _day(a), 7: _day(a + d)}
    out = M.pg2_convergence(pg1, _pt(12))
    assert out["conf_steps"][0][6] < M.PG2_ALPHA
    assert out["conf_nondecreasing"] is False
    # and steps scored before round 8 (no p) keep the strict rule
    assert M.paired_decrease([5, 7, 12, -0.003]) is True
    assert M.paired_decrease([5, 7, 12, 0.0]) is False


# --------------------------------------------------------------------------- #
# prospective held-out events (round 8, lead decision)
# --------------------------------------------------------------------------- #
def _pt_traffic(n_days: int = 21):
    """One segment (days 1-21) whose member A stops emitting after day 9 (a
    re-addressing: its user now emits from B), and a segment that ends on day
    12 (a drift) followed by its successor."""
    import datetime as dt
    start = dt.date(2025, 9, 1)                       # a Monday
    iso = [(start + dt.timedelta(days=i)).isoformat() for i in range(n_days)]
    workday = [(start + dt.timedelta(days=i)).weekday() < 5 for i in range(n_days)]
    opp = {"S": {}, "D0": {}, "D1": {}}
    for d in range(1, n_days + 1):
        per = {"10.0.0.2": 5}
        if d <= 9:
            per["10.0.0.1"] = 5
        opp["S"][iso[d - 1]] = per
        (opp["D0"] if d < 12 else opp["D1"])[iso[d - 1]] = {"10.0.0.3": 3 if workday[d - 1] else 1}
    win = {"workday": [[540.0, 600.0]], "nonworkday": [[600.0, 660.0]]}
    rows = [{"tid": "S", "system": "oa", "valid_from_day": 1, "valid_to_day": n_days + 1,
             "windows": win, "gen": {"members": ["10.0.0.1", "10.0.0.2"]}},
            {"tid": "D0", "lineage": "D", "system": "oa", "valid_from_day": 1, "valid_to_day": 12,
             "windows": win, "gen": {"members": ["10.0.0.3"]}},
            {"tid": "D1", "lineage": "D", "system": "oa", "valid_from_day": 12,
             "valid_to_day": n_days + 1, "windows": win, "gen": {"members": ["10.0.0.3"]}}]
    return M.PTruth({"pattern_truth": rows, "opportunities": opp,
                     "days": {"n_days": n_days, "start": iso[0], "workday": workday,
                              "day_start": [float(i) for i in range(n_days + 1)]}})


def test_held_out_traffic_is_drawn_from_after_the_snapshot():
    """Round 7 drew a claim's held-out events from the 14 days up to the
    snapshot, so a source that had stopped (re-addressed) still weighed in;
    a user of the profile only meets the traffic that follows it."""
    pt = _pt_traffic()
    back = pt.traffic(10)["S"]
    fwd = pt.traffic_after(10)["S"]
    assert back["workday"].get("10.0.0.1", 0) > 0              # the retrospective mix holds A
    assert set(fwd["workday"]) == {"10.0.0.2"} and set(fwd["nonworkday"]) == {"10.0.0.2"}
    assert sum(fwd["workday"].values()) == 5 * 5               # days 11-17: 5 workdays
    r = np.random.default_rng(0)
    rows = [pt.by_tid["S"]]
    evs = M.holdout_events(rows, r, 200, None, pt.traffic_after(10))
    assert evs and all(e["ip"] == "10.0.0.2" for e in evs)
    evs_b = M.holdout_events(rows, np.random.default_rng(0), 200, None, pt.traffic(10))
    assert any(e["ip"] == "10.0.0.1" for e in evs_b)


def test_prospective_window_stops_at_the_segment_end_and_continues_past_the_run():
    pt = _pt_traffic()
    # D0 ends on day 12 (a drift): from day 10 only day 11 is D0's own future
    f = pt.traffic_after(10)["D0"]
    assert sum(f["workday"].values()) == 3 and "nonworkday" not in f   # day 11 = a Thursday
    # on the last day the window is the run's last week, same weekdays
    assert [pt.proxy_day(x) for x in (21, 22, 28, 29)] == [21, 15, 21, 15]
    last = pt.traffic_after(21)["S"]
    assert sum(last["workday"].values()) == 5 * 5 and sum(last["nonworkday"].values()) == 2 * 5
    # a segment valid at the snapshot but silent after it falls back to its past mix
    pt2 = _pt_traffic()
    pt2.opp["S"] = {k: v for k, v in pt2.opp["S"].items() if pt2.day_index(k) <= 10}
    assert pt2.traffic_after(10)["S"] == pt2.traffic(10)["S"]
