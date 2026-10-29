"""Round 10, P09: windows fitted from few arrivals are STATED at a coverage
they hold (docs/lib3/progressive.md §16.17.11 item 1, OI8-f / OI8-h).

The point coverage of fitted windows (the predictive rank bound, lowered to
the date cross-validated posterior mean) is the share the next arrival falls
inside ON AVERAGE: a statement made at it fails its held-out check whenever
the true share is below it by more than the check's tolerance - in 8-18 % of
fits of a normal arrival law over 9-60 sessions (round 9 pack O seed 1: the
finance ledger windows stated 0.966-0.971, held 0.93-0.94). The stated
coverage is now the largest c with P(theta >= c - eps(c)) >= 0.95 under the
fit's evidence (pwindows.coverage_bound, _state_lb); the point value stays
for the fit's own decisions ('coverage_pt')."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.lib import pwindows as PW

DAY = 86400.0
T0 = 1.7e9


def _fit(seed: int, days: int, per: int, mu: float = 600.0, sd: float = 40.0):
    r = np.random.default_rng(seed)
    pts = []
    for d in range(days):
        for j, m in enumerate(r.normal(mu, sd, per)):
            pts.append((float(m), T0 + d * DAY + float(m) * 60.0, 1.0, f"10.0.0.{j}"))
    hist = np.zeros(96)
    for p in pts:
        hist[int(p[0] // 15) % 96] += 1.0
    rec = PW.fit_regime(hist, float(len(pts)), pts, 0.0)
    true = sum(norm.cdf(b, mu, sd) - norm.cdf(a, mu, sd) for a, b in rec["windows"]) \
        if rec is not None and not rec.get("all_day") else None
    return rec, true


def test_stated_window_coverage_holds_at_small_n():
    """Normal arrival law, 7 workdays x 4 sources (28 sessions): the share
    of fits whose stated coverage fails the contract's check (true share <
    c - eps) is <= 5 % with the stated bound; at the point value it was 18 %."""
    fails_stated = fails_point = n = 0
    for rep in range(80):
        rec, true = _fit(rep * 31 + 7, 7, 4)
        if true is None:
            continue
        n += 1
        c, c_pt = rec["coverage"], rec["coverage_pt"]
        assert c <= c_pt + 1e-12
        fails_stated += true < c - PN.hold_eps(c)
        fails_point += true < c_pt - PN.hold_eps(c_pt)
    assert n >= 60
    assert fails_stated / n <= 0.05, (fails_stated, n)
    assert fails_point / n >= 0.1, (fails_point, n)     # the defect the bound fixes


def test_the_bound_approaches_the_point_value_with_evidence():
    small, _ = _fit(3, 3, 3)
    large, _ = _fit(3, 30, 6)
    gap = lambda r: r["coverage_pt"] - r["coverage"]       # noqa: E731
    assert gap(small) > gap(large) >= 0.0
    assert gap(large) < 0.03


def test_window_hold_reads_the_bound_back():
    """P14's part confidence (window_hold) is the probability that the windows
    hold at their STATED coverage: >= 0.95 by construction (up to the
    in-sample / cross-validated posterior it is computed from)."""
    for seed in (1, 2, 5):
        rec, _ = _fit(seed, 7, 4)
        h = PW.window_hold(rec)
        assert h is not None and h >= 0.9, (seed, h, rec.get("coverage"), rec.get("coverage_pt"))


def test_decisions_keep_the_point_coverage():
    """The fit's confidence (coverage x day stability), the drift test and a
    young regime's previous law read the point coverage, not the bound."""
    rec, _ = _fit(11, 7, 4)
    assert math.isclose(PW.confidence(rec), rec["coverage_pt"] * float(rec.get("stability") or 1.0))
    rd = dict(rec)
    young = [(1000.0, T0 + 9 * DAY + 1000.0 * 60.0, 1.0, "10.0.0.1"),
             (1001.0, T0 + 9 * DAY + 1001.0 * 60.0, 1.0, "10.0.0.2")]
    a = PW.recent_drift(rd, young, 0.0, alpha=1.0)
    b = PW.recent_drift(dict(rd, coverage=rd["coverage_pt"]), young, 0.0, alpha=1.0)
    assert a == b
    # an all-day record states 1.0 as before
    flat = {"windows": [[0, 1440]], "all_day": True, "coverage": 1.0}
    assert PW.coverage_bound(flat) is None
