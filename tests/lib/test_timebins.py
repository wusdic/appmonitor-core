"""Tests for engines/behavior/lib/timebins.py: calendar (holidays, 调休
make-up workdays), cadence class, the local clock (tz, DST, 15-min slots and
their UTC bounds), dayparts, the feature.tctx dict and its float32 encoding,
and the circular-hour / von Mises helpers used by B03 and B07.

The DST tests use real zoneinfo transitions (Europe/Berlin 2026, Lord Howe's
30-minute DST, Kathmandu's 15-minute jump and Monrovia's 1972 jump of
44 min 30 s, which is not slot aligned) and check slot_bounds against a
brute-force scan of slot_of over every second around the transition.
"""
from __future__ import annotations

import datetime as dt
import math
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from app.engines.behavior.lib import timebins as TB  # noqa: E402
from helpers import make_tctx  # noqa: E402

UTC = dt.timezone.utc
NAN = float("nan")


def utc(*a) -> float:
    return dt.datetime(*a, tzinfo=UTC).timestamp()


# 2026-10-01T02:00Z = 10:00 Thursday in Shanghai (engines.md B01 (c))
TS_B01 = utc(2026, 10, 1, 2, 0)
# Europe/Berlin 2026: CET -> CEST at 01:00Z on 29 Mar, back at 01:00Z on 25 Oct
BERLIN_SPRING = utc(2026, 3, 29, 1, 0)
BERLIN_FALL = utc(2026, 10, 25, 1, 0)


# ------------------------------------------------------------------ contract
def test_constants_frozen():
    assert TB.DEFAULT_TZ == "Asia/Shanghai"
    assert TB.DEFAULT_DAY_HOURS == (8, 20)
    assert TB.SLOT_S == 900
    assert TB.CADENCE_CLASSES == (60, 300, 900, 3600)
    assert TB.DAYPARTS == ("wd_day", "wd_night", "nwd_day", "nwd_night")
    assert TB.DAY_TYPES == ("workday", "nonworkday")
    assert TB.TCTX_FIELDS == ("hour_local", "dow", "day_type", "bin48", "bin168",
                              "slot", "daypart", "cc")


def test_b01_time_context_target():
    cal = TB.parse_calendar({"holidays": ["2026-10-01"]})
    c = TB.tctx(TS_B01, "Asia/Shanghai", cal, dt=900.0)
    assert set(c) == set(TB.TCTX_FIELDS)
    assert c["hour_local"] == 10.0 and isinstance(c["hour_local"], float)
    assert c["day_type"] == "nonworkday" and c["daypart"] == "nwd_day"
    assert c["dow"] == 3                                  # Thursday
    assert c["bin48"] == 10 + 24 and c["bin168"] == 3 * 24 + 10
    assert c["slot"] == (int(TS_B01) + 8 * 3600) // 900 == 1989832
    assert c["cc"] == 900
    for f in ("dow", "bin48", "bin168", "slot", "cc"):
        assert type(c[f]) is int
    # same tick without the holiday: an ordinary workday morning
    c2 = TB.tctx(TS_B01, dt=900.0)
    assert c2["day_type"] == "workday" and c2["daypart"] == "wd_day" and c2["bin48"] == 10


def test_b01_via_config():
    cfg = {"tz": "Asia/Shanghai", "calendar": {"holidays": ["2026-10-01"]},
           "daypart_day_hours": [8, 20]}
    c = TB.tctx_from_config(TS_B01, cfg, 900.0)
    assert (c["hour_local"], c["day_type"], c["daypart"]) == (10.0, "nonworkday", "nwd_day")


# ------------------------------------------------------------------ calendar
def test_day_type_weekdays_weekends():
    mon = dt.date(2026, 9, 28)
    assert mon.weekday() == 0
    for k in range(7):
        d = mon + dt.timedelta(days=k)
        assert TB.day_type(d) == ("workday" if k < 5 else "nonworkday")
        assert TB.day_type(d, TB.Calendar()) == TB.day_type(d)


def test_day_type_holiday_and_makeup_saturday():
    # 2026 National Day: 1-7 Oct off, Saturday 10 Oct is a 调休 make-up workday
    cal = TB.parse_calendar({
        "holidays": [f"2026-10-0{d}" for d in range(1, 8)],
        "makeup_workdays": ["2026-10-10"],
    })
    assert dt.date(2026, 10, 10).weekday() == 5
    assert TB.day_type(dt.date(2026, 10, 1), cal) == "nonworkday"      # Thursday holiday
    assert TB.day_type(dt.date(2026, 10, 5), cal) == "nonworkday"      # Monday holiday
    assert TB.day_type(dt.date(2026, 10, 10), cal) == "workday"        # make-up Saturday
    assert TB.day_type(dt.date(2026, 10, 11), cal) == "nonworkday"     # ordinary Sunday
    assert TB.day_type(dt.date(2026, 10, 8), cal) == "workday"         # ordinary Thursday
    # a datetime is reduced to its date
    assert TB.day_type(dt.datetime(2026, 10, 10, 23, 59), cal) == "workday"

    sat_10 = utc(2026, 10, 10, 1, 30)                                  # 09:30 local
    c = TB.tctx(sat_10, calendar=cal, dt=60.0)
    assert c["dow"] == 5 and c["day_type"] == "workday" and c["daypart"] == "wd_day"
    assert c["bin48"] == 9 and c["bin168"] == 5 * 24 + 9 and c["cc"] == 60
    c_plain = TB.tctx(sat_10, dt=60.0)
    assert c_plain["day_type"] == "nonworkday" and c_plain["bin48"] == 33


def test_makeup_wins_over_holiday():
    d = dt.date(2026, 10, 10)
    cal = TB.Calendar(holidays=frozenset({d}), makeup_workdays=frozenset({d}))
    assert TB.day_type(d, cal) == "workday"


def test_calendar_day_follows_local_date_not_utc():
    # 2026-09-30T16:30Z is already 1 Oct 00:30 in Shanghai
    cal = TB.parse_calendar({"holidays": ["2026-10-01"]})
    c = TB.tctx(utc(2026, 9, 30, 16, 30), calendar=cal)
    assert c["day_type"] == "nonworkday" and c["daypart"] == "nwd_night"
    assert c["bin48"] == 24 and c["dow"] == 3
    c = TB.tctx(utc(2026, 9, 30, 15, 59), calendar=cal)                # 23:59 on 30 Sep
    assert c["day_type"] == "workday" and c["bin48"] == 23 and c["dow"] == 2


def test_parse_calendar_inputs():
    assert TB.parse_calendar(None) == TB.Calendar()
    assert TB.parse_calendar({}) == TB.Calendar()
    assert TB.parse_calendar({"holidays": None, "makeup_workdays": []}) == TB.Calendar()
    cal = TB.parse_calendar({
        "holidays": ["2026-10-01", dt.date(2026, 10, 2), dt.datetime(2026, 10, 3, 12, 0),
                     " 2026-10-04 ", "2026-10-05T00:00:00"],
        "makeup_workdays": "2026-10-10",                                 # lone string
    })
    assert cal.holidays == frozenset(dt.date(2026, 10, d) for d in range(1, 6))
    assert cal.makeup_workdays == frozenset({dt.date(2026, 10, 10)})
    assert isinstance(cal.holidays, frozenset)
    # a {date: name} mapping iterates over its dates
    cal2 = TB.parse_calendar({"holidays": {"2026-10-01": "National Day"}})
    assert cal2.holidays == frozenset({dt.date(2026, 10, 1)})
    # a Calendar passes straight through
    assert TB.parse_calendar(cal) is cal


@pytest.mark.parametrize("bad", [
    {"holidays": ["2026-13-01"]},
    {"holidays": ["not a date"]},
    {"makeup_workdays": [20261001]},
    ["2026-10-01"],
])
def test_parse_calendar_rejects_bad_config(bad):
    with pytest.raises(ValueError):
        TB.parse_calendar(bad)


def test_parse_calendar_cache_follows_contents():
    cfg = {"holidays": ["2026-10-01"], "makeup_workdays": []}
    a = TB.parse_calendar(cfg)
    assert TB.parse_calendar(cfg) is a                                  # cached
    assert TB.parse_calendar({"holidays": ["2026-10-01"]}) is a         # same contents
    cfg["holidays"].append("2026-10-02")                                # in-place edit
    b = TB.parse_calendar(cfg)
    assert b is not a and dt.date(2026, 10, 2) in b.holidays
    assert dt.date(2026, 10, 2) not in a.holidays                       # frozen, unchanged
    for k in range(3 * TB._CAL_CACHE_MAX):                              # bounded cache
        TB.parse_calendar({"holidays": [dt.date(2020, 1, 1) + dt.timedelta(days=k)]})
    assert len(TB._cal_cache) <= TB._CAL_CACHE_MAX


# ------------------------------------------------------------------ cadence
def test_cadence_class_exact_and_neighbours():
    for c in TB.CADENCE_CLASSES:
        assert TB.cadence_class(c) == c
        assert TB.cadence_class(float(c) * 1.01) == c
        assert TB.cadence_class(float(c) * 0.99) == c
    assert TB.cadence_class(1.0) == 60 and TB.cadence_class(1e-9) == 60
    assert TB.cadence_class(86400.0) == 3600
    assert TB.cadence_class(120) == 60       # linear distance would say 60 too
    assert TB.cadence_class(150) == 300      # log: 150 > sqrt(60*300)=134.2 -> 300
    assert TB.cadence_class(600) == 900      # log: 600 > sqrt(300*900)=519.6 -> 900
    assert TB.cadence_class(1700) == 900
    assert TB.cadence_class(np.float64(3000.0)) == 3600
    assert TB.cadence_class(np.int64(300)) == 300


def test_cadence_class_ties_go_to_larger():
    # 1800^2 == 900 * 3600: exact log-space tie
    assert TB.cadence_class(1800) == 3600
    assert TB.cadence_class(math.nextafter(1800.0, 0.0)) == 900
    for lo, hi in ((60, 300), (300, 900)):
        g = math.sqrt(lo * hi)
        assert TB.cadence_class(math.nextafter(g, 0.0) * (1 - 1e-12)) == lo
        assert TB.cadence_class(g * (1 + 1e-12)) == hi


def test_cadence_class_matches_log_argmin():
    rng = np.random.default_rng(3)
    for d in np.exp(rng.uniform(math.log(5.0), math.log(2e5), 2000)):
        errs = [abs(math.log(d / c)) for c in TB.CADENCE_CLASSES]
        best = min(range(4), key=lambda i: (errs[i], -TB.CADENCE_CLASSES[i]))
        assert TB.cadence_class(float(d)) == TB.CADENCE_CLASSES[best]


def test_cadence_class_edge_inputs():
    assert TB.cadence_class(0.0) == 60 and TB.cadence_class(-5.0) == 60
    assert TB.cadence_class(math.inf) == 3600
    with pytest.raises(ValueError):
        TB.cadence_class(NAN)


# ------------------------------------------------------------------ local clock
def test_local_datetime():
    d = TB.local_datetime(TS_B01)
    assert d.tzinfo is not None and d.utcoffset() == dt.timedelta(hours=8)
    assert (d.year, d.month, d.day, d.hour, d.minute) == (2026, 10, 1, 10, 0)
    assert TB.local_datetime(TS_B01, "UTC").hour == 2
    b = TB.local_datetime(utc(2026, 7, 1, 12), "Europe/Berlin")
    assert b.hour == 14 and b.utcoffset() == dt.timedelta(hours=2)
    assert TB.local_datetime(utc(2026, 1, 1, 12), "Europe/Berlin").hour == 13
    with pytest.raises(ValueError):
        TB.local_datetime(TS_B01, "Mars/Olympus_Mons")
    with pytest.raises(ValueError):
        TB.local_datetime(NAN)


def test_slot_of_shanghai():
    base = TB.slot_of(TS_B01)
    assert base == 1989832 and base % 96 == 40            # 10:00 is slot 40 of the day
    assert TB.slot_of(TS_B01 + 899.999) == base
    assert TB.slot_of(TS_B01 + 900) == base + 1
    assert TB.slot_of(TS_B01 - 0.001) == base - 1
    assert TB.slot_of(TS_B01, "UTC") == int(TS_B01) // 900
    # a local midnight starts a slot divisible by 96
    assert TB.slot_of(utc(2026, 9, 30, 16, 0)) % 96 == 0


def test_berlin_dst_local_hours():
    tz = "Europe/Berlin"
    # winter (+1), summer (+2), same UTC time of day
    assert TB.tctx(utc(2026, 1, 15, 7, 0), tz)["hour_local"] == 8.0
    assert TB.tctx(utc(2026, 7, 15, 7, 0), tz)["hour_local"] == 9.0
    # spring forward: 01:59:59 CET is followed by 03:00:00 CEST
    before = TB.tctx(BERLIN_SPRING - 1, tz)
    after = TB.tctx(BERLIN_SPRING, tz)
    assert before["hour_local"] == pytest.approx(2.0 - 1 / 3600)
    assert after["hour_local"] == 3.0
    assert before["bin168"] == 6 * 24 + 1 and after["bin168"] == 6 * 24 + 3   # Sunday
    assert after["slot"] - before["slot"] == 5            # slots 02:00-02:45 skipped
    # fall back: 02:59:59 CEST is followed by 02:00:00 CET (the hour repeats)
    before = TB.tctx(BERLIN_FALL - 1, tz)
    after = TB.tctx(BERLIN_FALL, tz)
    assert before["hour_local"] == pytest.approx(3.0 - 1 / 3600)
    assert after["hour_local"] == 2.0
    assert after["slot"] == before["slot"] - 3            # back to the 02:00 slot
    assert TB.tctx(BERLIN_FALL + 3600, tz)["hour_local"] == 3.0
    assert TB.tctx(BERLIN_FALL + 3600, tz)["slot"] == before["slot"] + 1
    # Sunday 25 Oct: nonworkday; the day has 25 hours of ticks but 24 bins
    assert after["day_type"] == "nonworkday" and after["daypart"] == "nwd_night"


@pytest.mark.parametrize("tz", ["Asia/Shanghai", "Europe/Berlin", "America/New_York",
                                "Asia/Kolkata", "Australia/Lord_Howe", "UTC"])
def test_tctx_matches_aware_datetime(tz):
    """Every field agrees with a direct computation from the aware datetime
    over a year of pseudo-random integer ticks (DST days included)."""
    rng = np.random.default_rng(7)
    t0 = utc(2026, 1, 1)
    ts_all = np.concatenate([
        t0 + rng.integers(0, 366 * 86400, 1500),
        BERLIN_SPRING + np.arange(-7200, 7200, 450),
        BERLIN_FALL + np.arange(-7200, 7200, 450),
    ])
    cal = TB.parse_calendar({"holidays": ["2026-10-01", "2026-05-01"],
                             "makeup_workdays": ["2026-10-10"]})
    for ts in ts_all.tolist():
        ts = float(ts)
        c = TB.tctx(ts, tz, cal, dt=300.0)
        d = TB.local_datetime(ts, tz)
        assert c["hour_local"] == pytest.approx(d.hour + d.minute / 60 + d.second / 3600,
                                                abs=1e-12)
        assert c["dow"] == d.weekday()
        assert c["day_type"] == TB.day_type(d.date(), cal)
        nwd = c["day_type"] == "nonworkday"
        assert c["bin48"] == d.hour + 24 * nwd
        assert c["bin168"] == d.weekday() * 24 + d.hour
        local_s = (d.replace(tzinfo=None) - dt.datetime(1970, 1, 1)).total_seconds()
        assert c["slot"] == int(local_s // 900) == TB.slot_of(ts, tz)
        assert c["daypart"] == ("nwd_" if nwd else "wd_") + ("day" if 8 <= d.hour < 20 else "night")
        assert c["cc"] == 300
        assert 0 <= c["bin48"] < 48 and 0 <= c["bin168"] < 168


def test_tctx_fractional_ts_is_self_consistent():
    """hour_local, bins and slot come from one local clock, so a ts a hair
    below a boundary is on the same side of it in every field."""
    for ts in (TS_B01 - 1e-6, TS_B01 - 0.4, TS_B01 + 1e-6, TS_B01 + 899.9999):
        c = TB.tctx(ts)
        h = math.floor(c["hour_local"])
        assert c["bin168"] % 24 == h == c["bin48"] % 24
        assert (c["slot"] % 96) // 4 == h
        assert c["slot"] == TB.slot_of(ts)
    assert TB.tctx(TS_B01 - 1e-6)["bin48"] == 9
    assert TB.tctx(TS_B01 + 90.0)["hour_local"] == pytest.approx(10.025)


def test_tctx_rejects_nonfinite_ts():
    for bad in (NAN, math.inf):
        with pytest.raises(ValueError):
            TB.tctx(bad)
        with pytest.raises(ValueError):
            TB.slot_of(bad)


# ------------------------------------------------------------------ slot bounds
def test_slot_bounds_regular():
    s = TB.slot_of(TS_B01)
    assert TB.slot_bounds(s) == (TS_B01, TS_B01 + 900)
    assert TB.slot_bounds(s + 1)[0] == TB.slot_bounds(s)[1]
    assert TB.slot_bounds(s, "UTC") == (s * 900.0, s * 900.0 + 900)
    rng = np.random.default_rng(11)
    for ts in (utc(2026, 1, 1) + rng.integers(0, 366 * 86400, 200)).tolist():
        for tz in ("Asia/Shanghai", "Europe/Berlin", "Asia/Kathmandu"):
            a, b = TB.slot_bounds(TB.slot_of(ts, tz), tz)
            assert a <= ts < b and b - a == 900
            assert TB.slot_of(a, tz) == TB.slot_of(b - 1, tz) == TB.slot_of(ts, tz)


def _check_bounds_by_scan(tz: str, t_lo: int, t_hi: int, step: int = 1) -> int:
    """slot_bounds(s) must equal the first maximal run of sampled seconds with
    slot_of == s (the pre-transition pass of a repeated slot); slots that are
    skipped (spring forward) must be empty intervals at the next instant."""
    runs: dict = {}                       # slot -> [start, end) of its first pass
    closed: set = set()
    prev = None
    for t in range(t_lo, t_hi, step):
        s = TB.slot_of(t, tz)
        if s != prev:
            if prev is not None:
                closed.add(prev)
                for gap in range(prev + 1, s):
                    assert TB.slot_bounds(gap, tz) == (float(t), float(t)), (tz, gap)
            runs.setdefault(s, [t, t + step])
        elif s not in closed:
            runs[s][1] = t + step
        prev = s
    cut = {TB.slot_of(t_lo, tz), TB.slot_of(t_hi - step, tz)}   # runs cut by the window
    checked = 0
    for s, (a, b) in runs.items():
        if s not in cut:
            assert TB.slot_bounds(s, tz) == (float(a), float(b)), (tz, s)
            checked += 1
    return checked


@pytest.mark.parametrize("tz,t_center,step", [
    ("Europe/Berlin", BERLIN_SPRING, 60),
    ("Europe/Berlin", BERLIN_FALL, 60),
    ("Australia/Lord_Howe", utc(2026, 4, 4, 15, 0), 60),   # 30-min fall back
    ("Australia/Lord_Howe", utc(2026, 10, 3, 15, 30), 60),  # 30-min spring forward
    ("Asia/Kathmandu", utc(1985, 12, 31, 18, 30), 60),     # +5:30 -> +5:45
    ("Africa/Monrovia", utc(1972, 1, 7, 0, 44, 30), 30),   # -0:44:30 -> 0, not slot aligned
])
def test_slot_bounds_across_transitions(tz, t_center, step):
    t_center = int(t_center)
    n = _check_bounds_by_scan(tz, t_center - 3 * 3600, t_center + 3 * 3600, step)
    assert n >= 15


def test_slot_bounds_berlin_details():
    tz = "Europe/Berlin"
    s = TB.slot_of(BERLIN_SPRING - 1, tz)                  # 01:45 CET
    assert TB.slot_bounds(s, tz) == (BERLIN_SPRING - 900, BERLIN_SPRING)
    for k in range(1, 5):                                  # 02:00 .. 02:45 do not exist
        assert TB.slot_bounds(s + k, tz) == (BERLIN_SPRING, BERLIN_SPRING)
    assert TB.slot_bounds(s + 5, tz) == (BERLIN_SPRING, BERLIN_SPRING + 900)
    # fall back: 02:45 CEST first pass, then 03:00 CET one hour later
    s = TB.slot_of(BERLIN_FALL - 1, tz)
    assert TB.slot_bounds(s, tz) == (BERLIN_FALL - 900, BERLIN_FALL)
    assert TB.slot_of(BERLIN_FALL + 2700, tz) == s        # second pass, same slot
    assert TB.slot_bounds(s - 3, tz) == (BERLIN_FALL - 3600, BERLIN_FALL - 2700)
    assert TB.slot_bounds(s + 1, tz) == (BERLIN_FALL + 3600, BERLIN_FALL + 4500)


def test_slot_bounds_partial_slot_monrovia():
    # 1972-01-07 00:44:30Z: local clock jumps from 23:59:59.x (-0:44:30) to
    # 00:44:30 GMT, so only the last 30 s of the 00:30-00:45 slot exist.
    tz = "Africa/Monrovia"
    T = utc(1972, 1, 7, 0, 44, 30)
    s0 = TB.slot_of(T, tz)
    assert TB.slot_bounds(s0, tz) == (T, T + 30)
    assert TB.slot_bounds(s0 - 1, tz) == (T, T)
    assert TB.slot_bounds(s0 - 2, tz) == (T, T)
    assert TB.slot_bounds(s0 + 1, tz) == (T + 30, T + 930)


# ------------------------------------------------------------------ daypart
def test_daypart_boundaries():
    assert TB.daypart("workday", 8.0) == "wd_day"
    assert TB.daypart("workday", 7.9999) == "wd_night"
    assert TB.daypart("workday", 19.9999) == "wd_day"
    assert TB.daypart("workday", 20.0) == "wd_night"
    assert TB.daypart("nonworkday", 12.0) == "nwd_day"
    assert TB.daypart("nonworkday", 0.0) == "nwd_night"
    assert TB.daypart("nonworkday", 23.99) == "nwd_night"
    assert TB.daypart("workday", 8.5, (9, 18)) == "wd_night"
    assert TB.daypart("workday", 17.99, (9, 18)) == "wd_day"
    # a wrapping window (night-shift site)
    assert TB.daypart("workday", 23.0, (22, 6)) == "wd_day"
    assert TB.daypart("workday", 3.0, (22, 6)) == "wd_day"
    assert TB.daypart("workday", 12.0, (22, 6)) == "wd_night"
    assert set(TB.daypart(t, h) for t in TB.DAY_TYPES for h in (3.0, 12.0)) == set(TB.DAYPARTS)


def test_daypart_rejects_bad_input():
    with pytest.raises(ValueError):
        TB.daypart("holiday", 10.0)
    with pytest.raises(ValueError):
        TB.daypart("workday", NAN)


# ------------------------------------------------------------------ config
def test_tctx_from_config_defaults_and_overrides():
    assert TB.tctx_from_config(TS_B01, {}, 60.0) == TB.tctx(TS_B01, dt=60.0)
    assert TB.tctx_from_config(TS_B01, None, 60.0) == TB.tctx(TS_B01, dt=60.0)
    assert TB.tctx_from_config(TS_B01, {"tz": None, "calendar": None}, 60.0) \
        == TB.tctx(TS_B01, dt=60.0)
    c = TB.tctx_from_config(TS_B01, {"tz": "Europe/Berlin"}, 3600.0)
    assert c["hour_local"] == 4.0 and c["daypart"] == "wd_night" and c["cc"] == 3600
    # day hours as list, tuple and "9-18" string
    for dh in ([11, 20], (11, 20), "11-20", "11–20"):
        c = TB.tctx_from_config(TS_B01, {"daypart_day_hours": dh}, 900.0)
        assert c["daypart"] == "wd_night"                 # 10:00 is before 11
    c = TB.tctx_from_config(TS_B01, {"daypart_day_hours": [10, 20]}, 900.0)
    assert c["daypart"] == "wd_day"
    for bad in ("8", [8], [8, 30], "a-b"):
        with pytest.raises(ValueError):
            TB.tctx_from_config(TS_B01, {"daypart_day_hours": bad}, 900.0)
    with pytest.raises(ValueError):
        TB.tctx_from_config(TS_B01, {"tz": "Nowhere/Land"}, 900.0)


def test_tctx_accepts_tzinfo_instance():
    from zoneinfo import ZoneInfo
    assert TB.tctx(TS_B01, ZoneInfo("Asia/Shanghai")) == TB.tctx(TS_B01)
    assert TB.slot_bounds(TB.slot_of(TS_B01), ZoneInfo("Asia/Shanghai")) == (TS_B01, TS_B01 + 900)


def test_agrees_with_test_helper_on_shared_fields():
    """tests/helpers.make_tctx delegates to timebins.tctx: every contract
    field agrees exactly, plus the helper's daypart_id."""
    rng = np.random.default_rng(5)
    hol, mk = ["2026-10-01", "2026-10-02"], ["2026-10-10"]
    cal = TB.parse_calendar({"holidays": hol, "makeup_workdays": mk})
    for ts in (utc(2026, 9, 25) + rng.integers(0, 30 * 86400, 300)).tolist():
        for tz in ("Asia/Shanghai", "Europe/Berlin"):
            a = TB.tctx(float(ts), tz, cal, dt=900.0)
            b = make_tctx(float(ts), tz=tz, dt=900.0, holidays=hol, makeup_workdays=mk)
            for f in TB.TCTX_FIELDS:
                assert a[f] == b[f], (f, ts, tz)
            assert TB.DAYPARTS[b["daypart_id"]] == b["daypart"]


# ------------------------------------------------------------------ encoding
def test_encode_decode_roundtrip():
    cal = TB.parse_calendar({"holidays": ["2026-10-01"]})
    rng = np.random.default_rng(2)
    for ts in (utc(2026, 1, 1) + rng.integers(0, 366 * 86400, 200) + 0.5).tolist():
        for dt_s in (60.0, 900.0, 3600.0):
            c = TB.tctx(ts, "Europe/Berlin", cal, dt=dt_s)
            v = TB.encode_tctx(c)
            assert v.dtype == np.float32 and v.shape == (8,)
            d = TB.decode_tctx(v)
            assert set(d) == set(TB.TCTX_FIELDS)
            for f in TB.TCTX_FIELDS:
                if f == "hour_local":
                    assert d[f] == pytest.approx(c[f], abs=2e-6)
                else:
                    assert d[f] == c[f] and type(d[f]) is type(c[f])
    c = TB.tctx(TS_B01, calendar=cal, dt=900.0)
    v = TB.encode_tctx(c)
    assert v.tolist() == [10.0, 3.0, 1.0, 34.0, 82.0, 1989832.0, 2.0, 900.0]
    # the slot of a far-future tick is still exact in float32
    far = TB.tctx(utc(2400, 1, 1, 0, 15))
    assert far["slot"] < 2 ** 24
    assert TB.decode_tctx(TB.encode_tctx(far))["slot"] == far["slot"]


def test_encode_decode_edge_cases():
    # indices are accepted as well as names (re-encoding a decoded row)
    c = TB.tctx(TS_B01, dt=900.0)
    alt = dict(c, day_type=0, daypart=0.0)
    assert np.array_equal(TB.encode_tctx(alt), TB.encode_tctx(c))
    # missing fields encode as NaN and decode as None (hour_local stays NaN)
    v = TB.encode_tctx({"bin48": 3})
    assert np.isnan(v[[0, 1, 2, 4, 5, 6, 7]]).all() and v[3] == 3.0
    d = TB.decode_tctx(v)
    assert math.isnan(d["hour_local"]) and d["bin48"] == 3
    assert d["dow"] is None and d["day_type"] is None and d["daypart"] is None
    # decode accepts float64 and 2-D single rows
    assert TB.decode_tctx(TB.encode_tctx(c).astype(np.float64).reshape(1, 8)) == \
        TB.decode_tctx(TB.encode_tctx(c))
    with pytest.raises(ValueError):
        TB.encode_tctx(dict(c, daypart="evening"))
    with pytest.raises(ValueError):
        TB.encode_tctx(dict(c, day_type=5))
    with pytest.raises(ValueError):
        TB.decode_tctx(np.zeros(7, dtype=np.float32))


# ------------------------------------------------------------------ circular hours
def test_hour_distance():
    assert TB.hour_distance(1.0, 23.0) == 2.0
    assert TB.hour_distance(23.0, 1.0) == -2.0
    assert TB.hour_distance(10.5, 10.5) == 0.0
    assert TB.hour_distance(12.0, 0.0) == 12.0
    assert TB.hour_distance(0.0, 12.0) == 12.0            # (-12, 12]: never -12
    assert TB.hour_distance(36.0, 1.0) == 11.0            # inputs outside [0, 24) wrap
    assert TB.hour_distance(-1e-17, 12.0) == 12.0
    assert TB.hour_distance(0.0, 1e-17) == pytest.approx(0.0, abs=1e-12)
    rng = np.random.default_rng(1)
    for h1, h2 in rng.uniform(-48, 48, (500, 2)).tolist():
        d = TB.hour_distance(h1, h2)
        assert -12.0 < d <= 12.0
        assert math.isclose((h2 + d - h1) % 24.0, 0.0, abs_tol=1e-9) or \
            math.isclose((h2 + d - h1) % 24.0, 24.0, abs_tol=1e-9)
        if abs(d) < 11.999:
            assert TB.hour_distance(h2, h1) == pytest.approx(-d, abs=1e-9)
    assert math.isnan(TB.hour_distance(NAN, 1.0))


def test_von_mises_weights_formula_and_support():
    def ref(dh, k=4.0):
        return math.exp(k * (math.cos(2 * math.pi * dh / 24) - 1))

    w = TB.von_mises_weights(10.5)                        # at a bin centre: 5 bins
    assert sorted(w) == [8, 9, 10, 11, 12]
    assert w[10] == 1.0
    for b, dh in ((8, -2), (9, -1), (11, 1), (12, 2)):
        assert w[b] == pytest.approx(ref(dh), rel=1e-12)
    assert w[8] == w[12] and w[9] == w[11]
    assert w[12] == pytest.approx(0.5851, abs=1e-4)
    w = TB.von_mises_weights(10.0)                        # on a boundary: 4 bins
    assert sorted(w) == [8, 9, 10, 11]
    assert w[9] == pytest.approx(ref(0.5)) and w[10] == pytest.approx(ref(0.5))
    w = TB.von_mises_weights(10.3)
    assert sorted(w) == [8, 9, 10, 11]                    # 12.5 is 2.2 h away
    assert w[10] == pytest.approx(ref(0.2)) and w[8] == pytest.approx(ref(-1.8))


def test_von_mises_weights_wrap_and_params():
    w = TB.von_mises_weights(0.2)
    assert sorted(w) == [0, 1, 22, 23]
    assert w[23] == pytest.approx(math.exp(4 * (math.cos(2 * math.pi * 0.7 / 24) - 1)))
    assert sorted(TB.von_mises_weights(23.5)) == [0, 1, 21, 22, 23]
    assert TB.von_mises_weights(24.5) == TB.von_mises_weights(0.5)
    assert TB.von_mises_weights(-0.5) == TB.von_mises_weights(23.5)
    assert set(TB.von_mises_weights(5.0, kappa=0.0).values()) == {1.0}
    assert TB.von_mises_weights(5.5, max_dh=0) == {5: 1.0}
    assert TB.von_mises_weights(5.5, max_dh=-1) == {}
    full = TB.von_mises_weights(5.5, max_dh=12)
    assert sorted(full) == list(range(24))
    assert min(full, key=full.get) == 17                  # opposite side of the clock
    assert full[17] == pytest.approx(math.exp(-8.0))
    big = TB.von_mises_weights(5.5, kappa=50.0)
    assert all(0.0 < x <= 1.0 for x in big.values())
    with pytest.raises(ValueError):
        TB.von_mises_weights(NAN)


# ------------------------------------------------------------------ cost
def test_tctx_is_cheap():
    cfg = {"tz": "Europe/Berlin",
           "calendar": {"holidays": [f"2026-{m:02d}-01" for m in range(1, 13)],
                        "makeup_workdays": ["2026-10-10"]}}
    n = 5000
    t0 = time.perf_counter()
    for k in range(n):
        TB.tctx_from_config(TS_B01 + 60.0 * k, cfg, 60.0)
    per_call = (time.perf_counter() - t0) / n
    assert per_call < 100e-6                             # ~5-10 us on a laptop


# ------------------------------------------------------------------ regressions (review)
def test_calendar_normalises_string_dates():
    # a Calendar built directly from ISO strings used to never match a date,
    # so the holiday was silently a workday
    cal = TB.Calendar(holidays=frozenset({"2026-10-01"}), makeup_workdays={"2026-10-10"})
    assert cal.holidays == frozenset({dt.date(2026, 10, 1)})
    assert cal.makeup_workdays == frozenset({dt.date(2026, 10, 10)})
    assert TB.day_type(dt.date(2026, 10, 1), cal) == "nonworkday"
    assert TB.day_type(dt.date(2026, 10, 10), cal) == "workday"
    assert TB.tctx(TS_B01, calendar=cal)["daypart"] == "nwd_day"
    assert cal == TB.parse_calendar({"holidays": ["2026-10-01"],
                                     "makeup_workdays": ["2026-10-10"]})
    with pytest.raises(ValueError):
        TB.Calendar(holidays=frozenset({"2026-13-01"}))


def test_tctx_accepts_raw_calendar_mapping():
    # engines may pass ctx.config['calendar'] straight through (was AttributeError)
    raw = {"holidays": ["2026-10-01"], "makeup_workdays": ["2026-10-10"]}
    c = TB.tctx(TS_B01, "Asia/Shanghai", raw, dt=900.0)
    assert (c["hour_local"], c["day_type"], c["daypart"]) == (10.0, "nonworkday", "nwd_day")
    assert TB.day_type(dt.date(2026, 10, 10), raw) == "workday"


def test_tctx_consistent_just_below_local_midnight_near_epoch():
    # L = -1e-300: divmod's remainder rounds up to 86400; the day used to roll
    # over to 1 Jan (bin168 72, Thursday) while slot said -1 (31 Dec 23:45).
    c = TB.tctx(-1e-300, "UTC")
    assert c["slot"] == -1 == TB.slot_of(-1e-300, "UTC")
    assert c["dow"] == 2 and c["bin168"] == 2 * 24 + 23 and c["bin48"] == 23
    assert 23.0 <= c["hour_local"] < 24.0 and c["daypart"] == "wd_night"


def test_encode_hour_local_stays_inside_its_hour():
    # 09:59:59.9999 local: float32(9.99999997) == 10.0 used to disagree with bin48 = 9
    ts = utc(2026, 10, 1, 1, 59, 59, 999900)
    c = TB.tctx(ts)
    assert c["bin48"] == 9 and c["hour_local"] < 10.0
    d = TB.decode_tctx(TB.encode_tctx(c))
    assert math.floor(d["hour_local"]) == 9 == d["bin48"]
    assert d["hour_local"] == pytest.approx(c["hour_local"], abs=2e-6)
    # 23:59:59.9999 must not encode as 24.0
    c = TB.tctx(utc(2026, 10, 1, 15, 59, 59, 999900))
    assert TB.decode_tctx(TB.encode_tctx(c))["hour_local"] < 24.0


def test_decode_rejects_bad_indices_and_maps_inf_to_none():
    v = TB.encode_tctx(TB.tctx(TS_B01, dt=900.0))
    for pos, bad in ((2, -1.0), (6, -1.0), (6, 4.0), (2, 0.5)):
        w = v.copy()
        w[pos] = bad          # -1 used to wrap silently to 'nonworkday' / 'nwd_night'
        with pytest.raises(ValueError):
            TB.decode_tctx(w)
    w = v.copy()
    w[5] = np.inf             # used to raise OverflowError
    assert TB.decode_tctx(w)["slot"] is None
