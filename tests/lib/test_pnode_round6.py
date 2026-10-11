"""lib/pnode round 6: a time-of-day claim is not judged on a time slice."""
from __future__ import annotations

from app.engines.behavior.lib import pnode as PN

DAY = 86400.0
T0 = 1_780_000_000.0 - (1_780_000_000.0 % DAY)


def _feed(hr, t0, n_units, step, when_hit, who_hit=True):
    """n_units sources, one event each, `step` seconds apart from t0."""
    for i in range(n_units):
        t = t0 + i * step
        cl = ("src", i, int(t // DAY))
        hr.add("who", who_hit(i) if callable(who_hit) else who_hit, 0.9, t, cluster=cl)
        hr.add("when", when_hit(i) if callable(when_hit) else when_hit, 0.9, t, cluster=cl)


def test_a_when_claim_is_not_failed_by_a_one_hour_batch():
    """A busy public node fills a test's 50 units in minutes: a batch of
    late-evening arrivals (the 10 % tail outside a 90 % window) is a time
    slice, not a day's arrivals. Pack O seeds 0-1: 51 of 53 failed portal
    tests failed on 'when' that held on the evaluator's day-wide draws."""
    hr = PN.HoldRecord()
    # one test of 60 units in the evening tail (all outside the window)
    _feed(hr, T0 + 22 * 3600, 60, 30.0, when_hit=False)
    # the rest of the day: 9 batches of 60 units across the window, all inside
    for b in range(9):
        _feed(hr, T0 + DAY + 8 * 3600 + b * 3600, 60, 30.0, when_hit=True)
    _feed(hr, T0 + 2 * DAY + 9 * 3600, 60, 30.0, when_hit=True)
    ps, n = hr.tests()
    assert n >= 9
    assert ps == n, (ps, n, hr.failures())          # no test failed on the evening slice


def test_a_wrong_when_claim_still_fails_once_a_day_is_seen():
    hr = PN.HoldRecord()
    for d in range(3):
        for b in range(12):
            # half of every day's arrivals fall outside the stated window
            _feed(hr, T0 + d * DAY + 6 * 3600 + b * 3600, 60, 30.0, when_hit=lambda i: i % 2 == 0)
    ps, n = hr.tests()
    assert n >= 10
    assert hr.failures().get("when", 0.0) > 5.0, hr.failures()
    # judged on the trailing day in every test after the first day: a wrong
    # window keeps failing tests (not once a day)
    assert ps < 0.6 * n, (ps, n)
