"""lib/pbounds (P06 maths, docs/lib3/progressive.md §6.10, card P06 tests)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.engines.behavior.lib import pbounds as PB
from app.engines.behavior.lib import pnode as PN

KB = 1024.0
T0 = 1_780_000_000.0
DAY = 86400.0


def _mixture(r, n):
    """90 % U(1, 2) KB + 5 % U(0.5, 1) KB + 5 % U(2, 3) KB (the requirement's login bodies)."""
    u = r.random(n)
    lo = np.where(u < 0.9, 1 * KB, np.where(u < 0.95, 0.5 * KB, 2 * KB))
    return lo + r.random(n) * np.where(u < 0.9, 1 * KB, np.where(u < 0.95, 0.5 * KB, 1 * KB))


def _num(values, days=10, log=True, t0=T0):
    s = PN.NumSummary(log=log)
    per = int(math.ceil(len(values) / days))
    for i, v in enumerate(values):
        d = i // per
        s.update(float(v), t0 + d * DAY + (i % per) * 60.0, 1.0, 1.0, day=700000 + d)
    return s, t0 + days * DAY, 700000 + days - 1


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_band_1_2_kb_and_range_with_rank_bound(seed):
    r = np.random.default_rng(seed)
    vals = _mixture(r, 200)
    s, t, day = _num(vals)
    rec = PB.fit_numeric(s, t, day, n_c=200, n_eff=200, unit="B")
    d = rec["disp90"]
    assert abs(rec["band90"][0] / KB - 1) < 0.25 and abs(rec["band90"][1] / (2 * KB) - 1) < 0.25
    tol = 2 * math.sqrt(0.09 / 200)            # two binomial standard errors at n = 200
    assert 0.88 - tol <= d["coverage"] <= 0.95 + tol
    emp = np.mean((vals >= d["lo"]) & (vals <= d["hi"]))
    assert 0.86 <= emp <= 0.96
    assert rec["n_rng"] == pytest.approx(200.0)
    assert rec["cover"] == pytest.approx(2.0 / 201.0)
    assert rec["hard"] is True
    assert rec["range"][0] == pytest.approx(vals.min(), rel=1e-9)
    assert rec["range"][1] == pytest.approx(vals.max(), rel=1e-9)
    assert rec["disp_range"]["text"] == "0.5–3 KB"


def test_band_renders_1_2_kb_on_most_samples():
    """Display rounding of the requirement's example over 60 independent samples
    of n = 200 (measured: 84 of 100 render exactly "1–2 KB"; the others render a
    wider band whose own coverage stays within sampling error of 90 %)."""
    hits = 0
    for seed in range(100, 160):
        s, t, day = _num(_mixture(np.random.default_rng(seed), 200))
        rec = PB.fit_numeric(s, t, day, n_c=200, n_eff=200, unit="B")
        hits += rec["disp90"]["text"] == "1–2 KB"
    assert hits / 60 >= 0.75


def test_range_uses_ring_observations_not_decayed_evidence():
    r = np.random.default_rng(3)
    s, t, day = _num(_mixture(r, 60), days=6)
    rec = PB.fit_numeric(s, t + 20 * DAY, day + 20, n_c=5.0, n_eff=2.0, unit="B")
    # the H_m evidence has decayed, the rank bound still counts the 60 ring observations
    assert rec["n_rng"] == pytest.approx(60.0)
    assert rec["cover"] == pytest.approx(2 / 61)


def test_gpd_tail_scores_12kb_below_1e4():
    r = np.random.default_rng(4)
    vals = _mixture(r, 2000)
    s, t, day = _num(vals, days=20)
    rec = PB.fit_numeric(s, t, day, n_c=2000, unit="B")
    assert rec["tail_hi"] is not None and rec["tail_hi"][3] >= 30
    p, flags = PB.p_value(rec, 12 * KB, num=s, t=t)
    assert p <= 1e-4 and "above_range" in flags
    p2, _ = PB.p_value(rec, 12 * KB)          # from the fitted record alone (reference snapshot)
    assert p2 <= 1e-4
    p_in, fl = PB.p_value(rec, 1.5 * KB, num=s, t=t)
    assert p_in > 0.3 and not fl


def test_approx_share_withholds_hard_bound():
    s, t, day = _num(_mixture(np.random.default_rng(5), 100))
    rec = PB.fit_numeric(s, t, day, n_c=100, approx_share=0.5, unit="B")
    assert rec["hard"] is False and "range" not in rec and "cover" not in rec
    assert "observed" in rec and "band90" in rec


def test_small_n_not_hard_and_cover_tightens_with_time():
    r = np.random.default_rng(6)
    vals = _mixture(r, 400)
    covers = []
    for n in (10, 30, 100, 400):
        s, t, day = _num(vals[:n], days=max(1, n // 10))
        rec = PB.fit_numeric(s, t, day, n_c=n, unit="B")
        covers.append(rec["cover"])
        assert rec["hard"] is (n >= 30)
    assert covers == sorted(covers, reverse=True)


def test_range_restarts_after_confidence_reset():
    s = PN.NumSummary(log=True)
    for d in range(5):
        for k in range(10):
            s.update(5000.0 + k, T0 + d * DAY + k, 1, 1, day=700000 + d)
    s.reset_confidence(T0 + 5 * DAY, day=700005)
    for d in range(5, 8):
        for k in range(10):
            s.update(1000.0 + k, T0 + d * DAY + k, 1, 1, day=700000 + d)
    rec = PB.fit_numeric(s, T0 + 8 * DAY, 700007, n_c=30)
    assert rec["range"][1] < 1100 and rec["n_rng"] == pytest.approx(30)


def test_pin_widens_never_narrows():
    s, t, day = _num(_mixture(np.random.default_rng(7), 100))
    base = PB.fit_numeric(s, t, day, unit="B")
    wide = PB.fit_numeric(s, t, day, unit="B", pin={"lo": 0.0, "hi": 10 * KB})
    narrow = PB.fit_numeric(s, t, day, unit="B", pin={"lo": 900.0, "hi": 1000.0})
    assert wide["range"] == [0.0, 10 * KB]
    assert narrow["range"] == base["range"]


def test_digest_fit_and_gain():
    td = PN.NumSummary(log=False)
    r = np.random.default_rng(8)
    for i, v in enumerate(r.poisson(3, 500)):
        td.update(float(v), T0 + i, 1, 1, day=700000)
    rec = PB.fit_digest(td.td, T0 + 500)
    assert rec["p99"] >= rec["band90"][1] and rec["hard"] is False
    # gain: a node concentrated in one system bin saves ~3 bits against 8 equal bins
    sysd = PN.NumSummary(log=False)
    for i, v in enumerate(r.uniform(0, 8000, 4000)):
        sysd.update(float(v), T0 + i, 1, 1)
    node = PN.NumSummary(log=False)
    for i, v in enumerate(r.uniform(100, 900, 200)):
        node.update(float(v), T0 + i, 1, 1)
    assert PB.bin_gain(node, sysd.td) > 2.5
    assert PB.bin_gain(sysd, sysd.td) < 0.1


def test_dirty_rule():
    nd = PN.Node(1, None, 0, 0, (), T0)
    nd.update_core(T0, 1.0, 1.0, ["1.1.1.1"], "1.1.1.1")
    mark = PB.fit_mark(nd, T0)
    assert not PB.is_dirty(mark, nd, T0 + 60)
    nd.update_core(T0 + 100, 1.0, 1.0, ["1.1.1.1"], "1.1.1.1")
    assert PB.is_dirty(mark, nd, T0 + 120)                    # +100 % evidence
    mark = PB.fit_mark(nd, T0 + 120)
    nd.state = "confirmed"
    assert PB.is_dirty(mark, nd, T0 + 130)                    # state change


def test_positive_lower_edge_of_a_wide_band_is_not_displayed_as_zero():
    """A heavy-tailed size (mail uploads 1.1 KB - 230 KB, git pushes up to
    2 MB) has a 90 % band whose width is set by its upper edge: rounded on a
    grid that coarse, the lower edge read '0' ('0-200 KB'), a band that
    states no lower bound (pack O: every mail / code statement failed PG1's
    band endpoints check for the lower edge only). The lower edge is rounded
    on its own grid under the same coverage check, and an observed range
    keeps a positive minimum."""
    import numpy as np
    from app.engines.behavior.lib import pbounds as PB
    xs = np.sort(np.exp(np.random.default_rng(0).uniform(np.log(800), np.log(300000), 5000)))
    cdf = lambda v: float(np.searchsorted(xs, v) / len(xs))
    lo, hi = (float(x) for x in np.quantile(xs, [0.05, 0.95]))
    d = PB.round_band(lo, hi, cdf, "B", n=1000)
    assert d["lo"] > 0 and abs(d["lo"] - lo) <= 0.2 * lo, d
    assert 0.85 <= d["coverage"] <= 0.95 and abs(d["hi"] - hi) <= 0.2 * hi
    assert not d["text"].startswith("0–")
    r = PB.round_range(800.0, 300000.0, "B")
    assert 0 < r["lo"] <= 800.0 and r["hi"] >= 300000.0 and r["lo"] >= 0.5 * 800.0
    assert PB.round_range(530.0, 3050.0, "B")["text"] == "0.5–3 KB"          # narrow ranges unchanged
