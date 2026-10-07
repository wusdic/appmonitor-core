"""Round 6 (time / views owner): small-sample window estimation in lib/pwindows.

* exact two-sample KS law for small samples (regime_cut's power on a young
  regime held in a shallow reservoir);
* UMVU edge extension of minute-mode windows and the matching predictive
  coverage (D1: the window of a new regime is right within 3 workdays);
* the rank bound on the slot-mode coverage;
* the flat-day rule on the out-of-sample coverage.
Each test fails on the round-5 code (checked against a HEAD snapshot)."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from app.engines.behavior.lib import pwindows as PW

OFF = 8 * 3600.0
DAY0 = 20333                     # a local date index (2025-09-01)


def _ts(day: int, minute: float) -> float:
    return (DAY0 + day) * 86400.0 - OFF + minute * 60.0


def _hist(pts):
    h = np.zeros(PW.SLOTS)
    for p in pts:
        h[int(p[0] // PW.SLOT_MIN) % PW.SLOTS] += p[2]
    return h


def _iou(w, t):
    inter = sum(max(0, min(b, t[1]) - max(a, t[0])) for a, b in w)
    return inter / ((t[1] - t[0]) + sum(b - a for a, b in w) - inter)


# ---------------------------------------------------------------- exact KS
def test_exact_two_sample_ks_tail_matches_enumeration():
    """_ks2_exact_sf is the exact null law of D: checked against a full
    enumeration of the C(m + n, m) label assignments of small samples."""
    for m, n in ((3, 4), (4, 4), (5, 3)):
        Ds = []
        for comb in itertools.combinations(range(m + n), m):
            a = np.asarray(comb, dtype=float)
            b = np.asarray([i for i in range(m + n) if i not in comb], dtype=float)
            D, _ = PW._wks(a, np.ones(m), b, np.ones(n))
            Ds.append(round(D * m * n))
        Ds = np.asarray(Ds)
        for k in sorted(set(Ds.tolist())):
            assert PW._ks2_exact_sf(int(k), m, n) == pytest.approx(float(np.mean(Ds >= k)), abs=1e-12)


def test_regime_cut_finds_a_separated_young_regime_from_few_arrivals():
    """Pack O seed 3, day 15: 综合部's login part held 9 arrivals of the old
    law (3 workdays, 09:07-09:19) and 6 of the new one (2 workdays,
    08:38-08:50), completely separated (D = 1). Kolmogorov's limit read
    p = 0.0015 > REGIME_ALPHA (no cut: the part stated the union of both
    laws); the exact law gives 2 / C(15, 6) = 0.0004."""
    old = [547, 549, 551, 549, 552, 559, 550, 552, 553]
    new = [518, 520, 520, 519, 529, 530]
    pts = []
    for i, m in enumerate(old):
        pts.append((float(m), _ts(i // 3, m), 1.0, f"s{i % 3}"))
    for i, m in enumerate(new):
        pts.append((float(m), _ts(3 + i // 3, m), 1.0, f"s{i % 3}"))
    cut = PW.regime_cut(pts, _hist(pts), tz_offset_s=OFF, min_after=1)
    assert cut is not None
    assert cut["since"] == pytest.approx(_ts(3, 0.0))
    assert cut["p"] <= PW.REGIME_ALPHA


# ------------------------------------------------------- edge extension
def test_young_window_extended_by_the_mean_spacing():
    """Pack O seed 4, day 16: the 9 arrivals of 综合部's new login law
    (08:30-08:51) spanned 08:31-08:44; the hull stated 08:31-08:44
    (IoU 0.62 < 0.7, D1 confirmed 5 workdays after the change). Extended by
    R / (k - 1) on each side (k = 9 sessions) the window reads
    08:30-08:46 (IoU >= 0.7); the stated coverage keeps the hull's rank
    bound 1 - 2 / (n + 1), a valid lower bound for the wider window."""
    mins = [511, 515, 523, 517, 517, 522, 513, 516, 517]
    pts = [(float(m), _ts(i // 3, m), 1.0, f"s{i % 3}") for i, m in enumerate(mins)]
    rec = PW.fit_daytype(_hist(pts), float(len(pts)), pts, tz_offset_s=OFF, min_points=PW.REGIME_POINTS)
    assert rec["res"] == "minute" and len(rec["windows"]) == 1
    s, e = rec["windows"][0]
    assert s <= 510 and e >= 525
    assert _iou(rec["windows"], (510, 531)) >= 0.7
    assert rec["coverage"] == pytest.approx(1.0 - 2.0 / (len(pts) + 1.0))


def test_a_session_is_one_draw_of_the_arrival_law():
    """Pack O seed 0, finance approvals (10:00-11:30, one approver): ~30 page
    views of ~10 sessions spanned 10:17-11:28. The views of one session share
    its start, so the extension counts sessions, not views: R / 9, not
    R / 29."""
    pts = []
    starts = [672, 612, 696, 636, 720, 648, 684, 624, 708, 660]
    for d, st in enumerate(starts):
        for j in range(3):
            m = st + 1.0 * j
            pts.append((float(m), _ts(d, m), 1.0, "192.168.2.10"))
    rec = PW.fit_daytype(_hist(pts), float(len(pts)), pts, tz_offset_s=OFF)
    assert rec["res"] == "minute" and len(rec["windows"]) == 1
    s, e = rec["windows"][0]
    R = 722.0 - 612.0
    assert s <= 612 - math.floor(R / 9.0) and e >= 723 + math.floor(R / 9.0)      # R / 9 (sessions)
    assert s < 612 - math.ceil(R / 29.0) - 1                                       # not R / 29 (views)


def test_edge_extension_keeps_a_conservative_coverage():
    """The extended window's stated coverage (the hull's rank bound) never
    exceeds the probability a next arrival of the same uniform law falls
    inside (Monte Carlo), and the extension leaves less outside than the hull."""
    r = np.random.default_rng(7)
    for n in (6, 9, 20):
        hits, claims, hull = [], [], []
        for i in range(400):
            m = np.sort(r.uniform(600.0, 640.0, n))
            pts = [(float(x), _ts(j, x), 1.0, "a") for j, x in enumerate(m)]
            rec = PW.fit_daytype(_hist(pts), float(n), pts, tz_offset_s=OFF, min_points=5)
            if rec["res"] != "minute" or rec["all_day"] or len(rec["windows"]) != 1:
                continue
            s, e = rec["windows"][0]
            claims.append(rec["coverage"])
            hits.append(max(0.0, float(min(e, 640.0) - max(s, 600.0))) / 40.0)   # P(next inside)
            hull.append((math.floor(m[-1]) + 1.0 - math.floor(m[0])) / 40.0)
        assert len(claims) > 300
        assert float(np.mean(claims)) <= float(np.mean(hits)) + 0.01
        assert float(np.mean(hits)) > float(np.mean(hull))


def test_well_sampled_window_is_not_widened():
    """A normal window (many arrivals) keeps its hull: the extension is
    below half a minute and the edges round back."""
    r = np.random.default_rng(3)
    m = np.floor(r.uniform(540.0, 561.0, 120))
    m[0], m[1] = 540.0, 560.0
    pts = [(float(x), _ts(i // 6, x), 1.0, f"s{i % 6}") for i, x in enumerate(m)]
    rec = PW.fit_daytype(_hist(pts), float(len(pts)), pts, tz_offset_s=OFF)
    assert rec["windows"] == [[540, 561]]


# --------------------------------------------------------- slot coverage
def test_slot_mode_coverage_is_predictive():
    """O-real seed 0, crm-02 days 7-14: a broad law in slot mode stated its
    in-sample slot share 1.00 ('09:00-18:00', ~100 arrivals) and held
    0.96-0.98: the next arrival can fall beyond the sample's extremes."""
    h = np.zeros(PW.SLOTS)
    h[36:72] = 1.0
    rec = PW.fit_daytype(h, 100.0, None, tz_offset_s=OFF)
    assert rec["res"] == "slot" and rec["windows"] == [[540, 1080]]
    assert rec["coverage"] == pytest.approx(1.0 - 2.0 / 101.0)


# ------------------------------------------------- flat out of sample
def test_windows_that_do_not_hold_out_of_sample_are_all_day():
    """Every date clusters somewhere else (a broad law sampled sparsely):
    in-sample the clusters are windows holding every arrival, out of sample
    none of them holds the other dates' arrivals. Round 5 stated them at the
    lowered coverage (portal's non-workday POST /login: '20:01-20:59' at
    0.20, held 0.01-0.03); they describe no time-of-day law."""
    centres = [480, 1180, 620, 1040, 760, 900]
    pts = []
    for d, c in enumerate(centres):
        for j in range(4):
            pts.append((float(c + j), _ts(d, c + j), 1.0, f"s{j}"))
    rec = PW.fit_regime(_hist(pts), float(len(pts)), pts, tz_offset_s=OFF)
    assert rec["all_day"] and rec["windows"] == [[0, PW.DAY_MIN]]
    assert rec["coverage"] == 1.0
