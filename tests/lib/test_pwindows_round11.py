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
