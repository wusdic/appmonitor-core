"""B14 round 8: BOCPD's bounded-influence clip is taken around the current
run's level and scale, not at +-3 absolute.

PG9 pack A seed 0: B06's WH score of oa-portal 10.30.2.23 sits well below 0
(hourly sums mean -2.7 in full mode, -4.4 in bounded mode, whose B06 model is
the one rebuilt at promotion); pinned at the -3 clip the channel had almost no
variance and one ordinary WH hour of +2.5 read cp 0.994 in bounded mode - a
bocpd alarm and a MEDIUM incident full mode does not raise (gate 3 8 vs 7).
"""
from __future__ import annotations

import math

import numpy as np

from app.engines.behavior import changepoint as CP


def _run_hours(x_int, x_wh):
    """Feed hourly inputs through B14's hour bookkeeping (one tick per hour);
    returns the cp series."""
    eng = CP.ChangepointEngine()
    run = {"hour": {"idx": None}, "bocpd": CP.bocpd_new(), "cp_prob": math.nan,
           "boc": {"on": False, "onset": math.nan, "axes": []}}
    t0 = 1_741_000_000.0 - (1_741_000_000.0 % 3600.0)
    cps = []
    for i, (a, b) in enumerate(zip(x_int, x_wh)):
        eng._roll_hour(run, t0 + (i + 1) * 3600.0, a, b)
        cps.append(run["cp_prob"])
    return np.asarray(cps, dtype=float), run


def _offset_series(n=160, seed=3):
    rng = np.random.default_rng(seed)
    x_int = rng.standard_normal(n)
    x_wh = -4.5 + 2.0 * rng.standard_normal(n)        # B06 WH: offset, wider than 1
    return x_int, x_wh


def test_one_ordinary_hour_of_an_offset_channel_is_not_a_changepoint():
    x_int, x_wh = _offset_series()
    x_wh[-1] = 2.5                                     # +3.5 sd of the channel's own spread
    cps, _ = _run_hours(x_int, x_wh)
    assert cps[-1] < CP.BOC_ALARM, cps[-1]
    assert np.nanmax(cps[24:]) < CP.BOC_ALARM          # no alarm anywhere on the null


def test_a_sustained_shift_of_an_offset_channel_stands_out_of_its_null():
    """Sensitivity is kept: with the absolute clip the offset channel's null
    hours reached cp 0.9998 and a +4 sd shift 0.06; around the run's level
    the shift's cp is several times the null's maximum."""
    x_int, x_wh = _offset_series(n=172)
    null, _ = _run_hours(x_int, x_wh)
    x_wh[160:] = 4.0 + 0.5 * np.random.default_rng(9).standard_normal(12)   # ~ +4 sd for 12 h
    cps, _ = _run_hours(x_int, x_wh)
    assert np.nanmax(cps[160:]) > 3.0 * np.nanmax(null[24:160])


def test_fresh_run_clip_is_the_old_absolute_clip():
    st = CP.bocpd_new()
    x = np.array([7.0, -9.0])
    assert np.allclose(CP.bocpd_clip(st, x), [3.0, -3.0])
    y = CP.bocpd_clip(st, np.array([math.nan, 1.5]))
    assert math.isnan(y[0]) and y[1] == 1.5


def test_centred_unit_input_matches_absolute_clip_closely():
    rng = np.random.default_rng(1)
    x_int, x_wh = rng.standard_normal(200), rng.standard_normal(200)
    cps, _ = _run_hours(x_int, x_wh)
    assert np.nanmax(cps[24:]) < CP.BOC_ALARM
