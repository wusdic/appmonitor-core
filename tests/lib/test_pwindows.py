"""lib/pwindows: activity windows on the circular day (docs/lib3/progressive.md
§6.13). Every accuracy statement below is checked over many seeds."""
from __future__ import annotations

import collections
import math
import time

import numpy as np
import pytest

from app.engines.behavior.lib import pwindows as W


def hist(minutes):
    h = np.zeros(96)
    np.add.at(h, (np.asarray(minutes) // 15).astype(int) % 96, 1)
    return h


def pts(minutes, spacing=86400.0 / 3):
    return list(zip(minutes, np.arange(len(minutes)) * spacing))


def edge_err(w, lo, hi):
    return max(abs(w[0] - lo), abs(w[1] - hi))


# ------------------------------------------------------------------ helpers
def test_ncp_prior_matches_scargle_eq21():
    n = 60
    assert W.ncp_prior(n, 0.05) == pytest.approx(4 - math.log(73.53 * 0.05 * n ** -0.478))


def test_bayesian_blocks_finds_a_step():
    c = np.array([0.0] * 10 + [10.0] * 5 + [0.0] * 10)
    blocks = W.bayesian_blocks(c, np.ones(25), W.ncp_prior(50))
    assert (10, 14) in blocks and len(blocks) == 3


def test_unwrap_cut_is_in_the_quiet_night():
    m = np.random.default_rng(0).uniform(480, 1200, 300)
    cut = W.unwrap_cut(hist(m))
    assert cut < 480 or cut >= 1200


def test_intervals_labels_and_membership():
    assert W.as_intervals([[1320, 120]]) == [[0, 120], [1320, 1440]]
    assert W.in_windows(30, [[1320, 120]]) and W.in_windows(1400, [[1320, 120]])
    assert not W.in_windows(600, [[1320, 120]])
    assert W.label(540, 561) == "w:0900-0921"
    assert not W.moved([[540, 561]], [[545, 570]]) and W.moved([[540, 561]], [[540, 600]])
    assert W.moved([[540, 561]], [[540, 561], [1020, 1030]])
    assert W.concentrated(hist(np.full(10, 545.0))) and not W.concentrated(hist(np.arange(0, 1440, 10)))


# ------------------------------------------------------------------ fitting
def test_login_window_0900_0921_minute_resolution():
    """60 arrivals U(09:00, 09:21) (20 workdays x 3 IPs): 09:00-09:21 within
    +-1 minute in >= 95 % of 200 seeds, coverage >= 0.95 always."""
    errs, covs = [], []
    for seed in range(200):
        m = np.random.default_rng(seed).uniform(540, 561, 60)
        f = W.fit_daytype(hist(m), 60, pts(m))
        assert f["res"] == "minute" and len(f["windows"]) == 1
        errs.append(edge_err(f["windows"][0], 540, 561))
        covs.append(f["coverage"])
    assert np.mean(np.asarray(errs) <= 1) >= 0.95, collections.Counter(errs)
    assert min(covs) >= 0.95


def test_slot_mode_before_the_minute_reservoir():
    m = np.random.default_rng(1).uniform(540, 561, 6)
    f = W.fit_daytype(hist(m), 6, None)
    assert f["res"] == "slot" and f["windows"] == [[540, 570]]     # slot edges contain the truth


def test_precision_rises_with_observations():
    """Longer observation -> tighter windows: the mean edge error falls
    monotonically from 6 to 15 to 60 arrivals (100 seeds each)."""
    means = []
    for n in (6, 15, 60):
        e = []
        for seed in range(100):
            m = np.random.default_rng(seed).uniform(540, 561, n)
            f = W.fit_daytype(hist(m), n, pts(m) if n >= W.MIN_POINTS else None)
            e.append(edge_err(f["windows"][0], 540, 561))
        means.append(float(np.mean(e)))
    assert means[0] > means[1] > means[2], means
    assert means[2] <= 0.5


def test_window_crossing_midnight_is_one_window():
    for seed in range(20):
        m = np.random.default_rng(seed).uniform(22 * 60, 26 * 60, 200) % 1440
        for p in (pts(m, 3600.0), None):
            f = W.fit_daytype(hist(m), 200, p)
            assert len(f["windows"]) == 1
            s, e = f["windows"][0]
            tol = 15 if p is None else 6                 # slot edges / 200 points over 240 min
            assert s > e and abs(s - 1320) <= tol and abs(e - 120) <= tol, f["windows"]


def test_broad_activity_is_found_where_the_literal_mean_rule_fails():
    """08:00-20:00 has only 2x the mean rate (< kappa = 3): r_bg = n / 1440
    alone would publish nothing; the off-hours background finds it."""
    assert (1440 / 720) < W.KAPPA
    for seed in range(100):
        m = np.random.default_rng(seed).uniform(480, 1200, 300)
        f = W.fit_daytype(hist(m), 300, None)
        assert len(f["windows"]) == 1 and edge_err(f["windows"][0], 480, 1200) <= 15, f["windows"]


def test_flat_day_is_all_day():
    m = np.random.default_rng(0).uniform(0, 1440, 400)
    f = W.fit_daytype(hist(m), 400, None)
    assert f["all_day"] and f["windows"] == [[0, 1440]] and f["coverage"] == 1.0


def test_two_windows_with_sparse_noise():
    """09:00-09:21 and 17:00-17:14 plus 5 random arrivals: both windows within
    +-3 minutes in >= 95 % of 300 seeds; noise never forms a window."""
    ok = 0
    for seed in range(300):
        r = np.random.default_rng(seed)
        m = np.concatenate([r.uniform(540, 561, 40), r.uniform(1020, 1034, 30), r.uniform(0, 1440, 5)])
        f = W.fit_daytype(hist(m), 75, pts(m, 3600.0))
        w = f["windows"]
        ok += len(w) == 2 and edge_err(w[0], 540, 561) <= 3 and edge_err(w[1], 1020, 1034) <= 3
    assert ok / 300 >= 0.95


def test_day_stability_and_dates_from_the_reservoir():
    r = np.random.default_rng(3)
    m = r.uniform(540, 561, 60)
    ts = np.repeat(np.arange(20) * 86400.0, 3)[:60] + 3600
    m[-1] = 900.0                                        # one date with an arrival outside
    f = W.fit_daytype(hist(m), 60, list(zip(m, ts)))
    assert f["dates"] == 20
    assert f["stability"] == pytest.approx(19 / 20)
    assert W.confidence(f) == pytest.approx(f["coverage"] * 19 / 20)


def test_rendering():
    f = W.fit_daytype(hist(np.full(20, 545.0)), 20, None)
    f.update(windows=[[540, 561]], coverage=0.97, dates=21)
    assert W.render_zh("wd", f) == "工作日 09:00–09:21（覆盖 97 %，21 个工作日）"
    assert W.render_en("wd", f) == "workdays 09:00–09:21 (coverage 97 %, 21 dates)"


def test_fit_cost_is_bounded():
    m = np.random.default_rng(0).uniform(480, 1200, 256)
    t0 = time.perf_counter()
    for _ in range(10):
        W.fit_daytype(hist(m), 300, None)
    slot = (time.perf_counter() - t0) / 10
    t0 = time.perf_counter()
    for _ in range(10):
        W.fit_daytype(hist(m), 300, pts(m, 3600.0))
    minute = (time.perf_counter() - t0) / 10
    assert slot < 0.02 and minute < 0.06, (slot, minute)
