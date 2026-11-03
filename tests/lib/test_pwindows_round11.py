"""lib/pwindows round 11: a day type seen on two dates is stated at its
date cross-validated coverage, not at the in-sample coverage its windows
were chosen on (progressive.md §16.18.11 item 7: O60 seed 0, day 7, portal's
non-workday windows fitted on the first weekend stated 0.82 / 0.89 where the
arrival law puts 0.74 / 0.77)."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.lib import pwindows as PW

DAY = 86400.0
T0 = 1_780_000_000.0 - (1_780_000_000.0 % DAY)
MU, SD = 900.0, 200.0                 # a broad daytime law (a portal's visitors)


def _pts(seed: int, ndates: int, per_date: int):
    rng = np.random.default_rng(seed)
    pts = []
    for d in range(ndates):
        for i in range(per_date):
            m = float(np.clip(rng.normal(MU, SD), 0.0, 1439.0))
            pts.append((m, T0 + d * 7 * DAY + m * 60.0, 1.0, f"10.0.{seed % 250}.{i}"))
    h = np.zeros(PW.SLOTS)
    for m, *_ in pts:
        h[int(m // PW.SLOT_MIN)] += 1.0
    return h, pts


def _law(wins) -> float:
    return float(sum(norm.cdf(e, MU, SD) - norm.cdf(s, MU, SD) for s, e in wins))


def test_two_dates_are_cross_validated():
    h, pts = _pts(3, 2, 45)
    cvi = {}
    fc = PW.cv_coverage(h, pts, 0.0, info=cvi)
    assert cvi["dates"] == 2 and fc is not None          # round 10: None (3 dates needed)
    rec = PW.fit_regime(h, float(len(pts)), pts)
    assert rec.get("cv") is not None and rec["coverage"] <= rec["coverage_in"]
    one = [p for p in pts if p[1] < T0 + DAY]
    assert PW.cv_coverage(h, one, 0.0) is None           # one date: nothing to hold out


def test_two_date_windows_hold_at_their_stated_coverage():
    held, stated, law = [], [], []
    for seed in range(40):
        h, pts = _pts(seed, 2, 45)
        rec = PW.fit_regime(h, float(len(pts)), pts)
        if rec is None or rec.get("all_day"):
            continue
        c = float(rec["coverage"])
        th = _law(rec["windows"])
        stated.append(c)
        law.append(th)
        held.append(th >= c - PN.hold_eps(c))
    assert len(held) >= 35
    # round 10 (in-sample coverage): held in 0.68 of the fits, stated 0.92 for a law share of 0.90
    assert np.mean(held) >= 0.78, np.mean(held)
    assert np.mean(stated) <= np.mean(law) + 0.01


# ------------------------------------------------- the fit date is censored
def _broad(seed: int, ndates: int, per_date: int, now_min: float):
    """A broad daytime law over `ndates` complete dates, then the fit date
    observed up to `now_min` (local minute) - P09 fits at 15:15 / 21:15."""
    rng = np.random.default_rng(seed)
    pts = []
    for d in range(ndates + 1):
        for i in range(per_date):
            m = float(np.clip(rng.normal(MU, SD), 0.0, 1439.0))
            if d == ndates and m >= now_min:
                continue                                  # not yet arrived
            pts.append((m, T0 + d * DAY + m * 60.0, 1.0, f"10.1.{d}.{i}"))
    h = np.zeros(PW.SLOTS)
    for m, *_ in pts:
        h[int(m // PW.SLOT_MIN)] += 1.0
    now = T0 + ndates * DAY + now_min * 60.0
    return h, pts, now


def test_a_partial_fit_date_is_not_a_change_of_the_law():
    """O60 seed 0, day 8 (15:09): the day's 23 portal comments, all before
    15:09, against full earlier days were a 'change' (regime_cut p 1e-4); the
    node became evolving and PUB.portal.visit#3 was lost."""
    cuts_old = cuts_new = 0
    for seed in range(20):
        h, pts, now = _broad(seed, 5, 40, 15 * 60.0)
        cuts_old += PW.regime_cut(pts, h, 0.0, min_after=1) is not None
        cuts_new += PW.regime_cut(pts, h, 0.0, min_after=1, now=now) is not None
        rec = PW.fit_regime(h, float(len(pts)), pts, now=now)
        assert rec is not None and not rec.get("drift"), seed
    assert cuts_old >= 15 and cuts_new <= 1, (cuts_old, cuts_new)


def test_a_real_change_on_the_fit_date_is_still_seen():
    # a login window over by the fit time: the date is its whole day (not
    # censored), and a move to 08:30 on it is found as before
    rng = np.random.default_rng(5)
    pts = []
    for d in range(6):
        for i in range(6):
            m = (540.0 if d < 5 else 510.0) + 20.0 * rng.random()
            pts.append((m, T0 + d * DAY + m * 60.0, 1.0, f"192.168.1.{20 + i}"))
    h = np.zeros(PW.SLOTS)
    for m, *_ in pts:
        h[int(m // PW.SLOT_MIN)] += 1.0
    now = T0 + 5 * DAY + 15 * 3600.0
    assert PW._censored(pts, 0.0, now) is None
    assert PW.regime_cut(pts, h, 0.0, min_after=1, now=now) is not None


def test_windows_are_fitted_on_complete_dates():
    """O60 seed 0, day 7 (21:15): non-workday windows fitted on two dates,
    the second only up to 21:15, ended at 20:18 and stated 0.82 where the law
    puts 0.74; the fit date's arrivals wait for the next day's fit."""
    h, pts, now = _broad(11, 1, 200, 20 * 60.0 + 15.0)
    cen = PW._censored(pts, 0.0, now)
    assert cen is not None and abs(cen[1] - 1215.0) < 1e-6
    kept = PW._complete(pts, 0.0, now)
    assert kept and all(p[1] < T0 + DAY for p in kept)
    assert PW._complete(pts, 0.0, None) == pts
