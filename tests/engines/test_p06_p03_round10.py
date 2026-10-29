"""Round 10, P06 / P03: the range-floor loop (docs/lib3/progressive.md
§16.17.11 item 4, §16.17.12 item 6).

O-red seed 0 (round 9): the portal's GET /news/{num} route split into young
children; on one child P06's clean_range dropped both sides of its confidence
segment's only day, because the day's extremes were values of rows P03 had
judged violations (on any attribute) and every reservoir item EQUAL to an
excluded value was removed with it - so the record had range None, n_rng 0.
P03's range floor was then off, the next net.bytes_up = 460 against the band
the node itself stated got the bounded GPD tail's p = 1e-9, was judged a
violation, excluded again: six MEDIUM false alarms and D4.

  * clean_range: an excluded row removes its OWN reservoir item (by its
    time; one item of its value when the time is unknown), not every row of
    its value (pbounds._clean_pool; P06's ledger keeps the row's time);
  * P03: a record without an observed range floors values at / one
    resolution step beyond the band98 it states (conformity._floor_ref), so a
    value at the band's own end can become a clean observation again."""
from __future__ import annotations

import math

import numpy as np

from app.engines.behavior import conformity as CF
from app.engines.behavior.lib import pbounds as PB
from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.lib import pscore as SC

T0 = 1_788_220_800.0
DAY = 86400.0
DAY0 = 739500


def _day_of(ts: float) -> int:
    return DAY0 + int((ts - T0) // DAY)


def _young_segment(log: bool = True):
    """Three days of 470-520 B requests, then a new confidence segment whose
    only day holds two 460-B requests (below the running Q(0.10): both in the
    lower exceedance reservoir) and 480-500 B otherwise."""
    r = np.random.default_rng(4)
    num = PN.NumSummary(log=log)
    for k in range(3):
        for j in range(200):
            num.update(float(r.integers(470, 521)), T0 + k * DAY + j * 60, 1.0, 1.0, day=DAY0 + k)
    num.reset_confidence(T0 + 3 * DAY, day=DAY0 + 3)
    rows = [460.0, 460.0] + [float(r.integers(480, 501)) for _ in range(28)]
    ts = [T0 + 3 * DAY + 3600 + j * 60 for j in range(len(rows))]
    for v, t in zip(rows, ts):
        num.update(v, t, 1.0, 1.0, day=DAY0 + 3)
    return num, rows, ts


def test_an_excluded_row_removes_its_own_reservoir_item_not_its_clean_twin():
    num, rows, ts = _young_segment()
    lows = sorted(round(math.exp(y)) for y, _w, t in num.lo_res.items() if _day_of(t) == DAY0 + 3)
    assert lows == [460, 460]                          # the fixture: both 460s are tail items
    # the SECOND 460-B row was judged a violation (on another attribute): the
    # first one is a clean observation of the same value, and stays the minimum
    excl = {DAY0 + 3: [(num.y(460.0), ts[1])]}
    lo, hi, n, dropped = PB.clean_range(num, DAY0 + 3, excl, day_of=_day_of)
    assert abs(math.exp(lo) - 460.0) < 1e-6, (lo, hi)
    assert dropped == 0 and n == 29.0                  # the excluded row leaves the count
    # without its time (a ledger saved before round 10) one item of its value goes
    lo2, _, _, _ = PB.clean_range(num, DAY0 + 3, {DAY0 + 3: [num.y(460.0)]}, day_of=_day_of)
    assert abs(math.exp(lo2) - 460.0) < 1e-6
    # both 460-B rows excluded: the side leaves the range (no clean value of the day)
    excl2 = {DAY0 + 3: [(num.y(460.0), ts[0]), (num.y(460.0), ts[1])]}
    lo3, hi3, n3, _ = PB.clean_range(num, DAY0 + 3, excl2, day_of=_day_of)
    assert not math.isfinite(lo3) and n3 == 0.0


def test_the_young_segment_keeps_its_range_and_states_it():
    num, rows, ts = _young_segment()
    excl = {DAY0 + 3: [(num.y(460.0), ts[1])]}
    rec = PB.fit_numeric(num, ts[-1] + 60, DAY0 + 3, n_c=30.0, n_eff=30.0, unit="B",
                         excl=excl, day_of=_day_of)
    assert rec["range"] is not None and abs(rec["range"][0] - 460.0) < 1e-6, rec.get("range")


def _no_range_record():
    """A fitted record whose young segment lost its range (range None) while
    its band98 starts at 461 B and a bounded GPD lower tail ends just below it:
    the round-9 portal record's shape."""
    lg = math.log
    q = [lg(461.0)] * 4 + list(np.linspace(lg(462.0), lg(516.0), 29))
    return {"kind": "num", "log": True, "unit": "B", "band98": [461.0, 516.0], "band90": [470.0, 516.0],
            "qgrid": [float(x) for x in q], "n_c": 300.0, "n_rng": 0.0, "range": None,
            "tail_lo": [461.0, -0.9, 0.45, 40], "tail_hi": None}


def test_a_value_at_the_stated_bands_end_is_not_at_the_probability_floor():
    rec = _no_range_record()
    n = rec["n_c"]
    fast = CF._NumFast(rec)
    for x, floor in ((461.0, SC.conformal_rank_p(1.0, n)),     # the band's own lower end
                     (460.0, SC.conformal_rank_p(0.0, n))):    # one byte beyond it (integer data)
        p_fast, _ = CF._check("num", fast, x, None)
        p_ref, _ = CF._check("num", rec, x, None)
        assert p_fast == p_ref
        assert p_fast >= floor > 1e-3, (x, p_fast)         # not a violation for P06's ledger (p <= 1e-3)
    # the raw tail still says 'impossible' (what the floor corrects) ...
    p_raw, _ = fast.p(460.0)
    assert p_raw < 1e-6
    # ... and a value far below the band keeps its tail p
    p_far, _ = CF._check("num", fast, 300.0, None)
    assert p_far < 1e-6


def test_the_observed_range_still_takes_precedence():
    rec = dict(_no_range_record(), range=[455.0, 516.0], n_rng=50.0)
    p, _ = CF._check("num", CF._NumFast(rec), 455.0, None)
    assert p == max(CF._NumFast(rec).p(455.0)[0], SC.conformal_rank_p(1.0, 50.0))
    assert CF._floor_ref(rec) == ([455.0, 516.0], 50.0)
    assert CF._floor_ref(dict(rec, range=None)) == ([461.0, 516.0], 300.0)
    assert CF._floor_ref(dict(rec, range=None, n_c=float("nan"))) == (None, None)
