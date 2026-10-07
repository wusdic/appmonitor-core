"""P14 views, round 6 (time / views owner): a drifting statement's window
term is the probability that the windows HOLD at their stated coverage given
the young dates' arrivals, not that posterior's mean. Fails on the round-5
code."""
from __future__ import annotations

from app.engines.behavior import views as VW


def test_drifting_windows_are_stated_at_their_hold_probability():
    """Pack O D1, day 14 (seed 0): the established login window (09:00-09:21,
    coverage 0.9) and the first new workday's 3 logins all outside: round 5
    stated it at the young arrivals' posterior coverage (0 + 2 x 0.9) / (3 + 2)
    = 0.36 and it held 0; the probability that the window holds is ~0.007."""
    drift = {"workday": {"date": 20345, "p": 0.003, "k": 0, "n": 3, "sources": 3, "coverage": 0.36}}
    got = VW.drift_conf(drift, {"coverage": 0.9}, 0.36)
    assert got < 0.01
    # a young date that agrees with the windows leaves them near-certain
    ok = VW.drift_conf({"workday": dict(drift["workday"], k=3, coverage=0.92)}, {"coverage": 0.9}, 0.92)
    assert ok > 0.5
