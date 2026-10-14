"""Round 7 (time / workflow / content / views owner): lib/pwindows.

* the flat-day rule needs dated evidence and a test against the windowed
  alternative (O60 seed 0: 财务部's mail part on three workdays was stated
  '工作日 全天 (覆盖 100 %)').
Each test marked "fails on round 6" was checked against a HEAD (52e813f)
snapshot."""
from __future__ import annotations

import numpy as np

from app.engines.behavior.lib import pwindows as PW

OFF = 8 * 3600.0
DAY0 = 20333                     # a local date index (2025-09-01)
MAIL = [(555, 573), (810, 843)]  # 财务部's mail law in pack O (09:15-09:33, 13:30-14:03)


def _ts(day: int, minute: float) -> float:
    return (DAY0 + day) * 86400.0 - OFF + minute * 60.0


def _hist(pts):
    h = np.zeros(PW.SLOTS)
    for p in pts:
        h[int(p[0] // PW.SLOT_MIN) % PW.SLOTS] += p[2]
    return h


def _mail(seed: int, dates: int = 3):
    """Three addresses, one session a date each (1-4 views a few seconds to
    a minute apart) in one of the two windows."""
    r = np.random.default_rng(seed)
    pts = []
    for d in range(dates):
        for s in range(3):
            a, b = MAIL[r.integers(0, 2)]
            m0 = r.uniform(a, b)
            for v in range(r.integers(1, 5)):
                m = min(b - 0.01, m0 + v * r.uniform(0, 1.5))
                pts.append((m, _ts(d, m), 1.0, f"192.168.2.1{s}"))
    return pts


def test_three_dates_of_a_two_window_law_are_never_declared_flat():
    """Fails on round 6 (12 of 100 fits all-day): with three dates the
    leave-dates-out coverage is the share of one date's arrivals in the
    windows of the other two, below one half whenever the dates used
    different parts of the windows - not evidence of a flat day."""
    flat = 0
    for sd in range(100):
        pts = _mail(sd)
        rec = PW.fit_regime(_hist(pts), float(len(pts)), pts, tz_offset_s=OFF)
        if rec["all_day"]:
            flat += 1
            continue
        # the windows are the law's, at an honest (lowered) coverage
        assert all(any(a - 30 <= s and e <= b + 30 for a, b in MAIL) for s, e in rec["windows"])
        assert rec["coverage"] < 1.0
    assert flat == 0


def test_flat_needs_dates_and_a_windowed_law_no_better_than_a_flat_day():
    """cv_coverage's log score against a flat day: positive for a
    concentrated law (even when its narrow windows hold few fresh arrivals),
    negative for clusters that move from date to date; and _flat_supported
    asks for FLAT_DATES dates as well."""
    conc = _mail(3, dates=6)
    info: dict = {}
    assert PW.cv_coverage(_hist(conc), conc, OFF, info=info) is not None
    assert info["dates"] == 6 and info["llr"] > 0.5
    assert not PW._flat_supported(info)
    centres = [480, 1180, 620, 1040, 760, 900]
    scat = [(float(c + j), _ts(d, c + j), 1.0, f"s{j}") for d, c in enumerate(centres) for j in range(4)]
    info2: dict = {}
    PW.cv_coverage(_hist(scat), scat, OFF, info=info2)
    assert info2["dates"] == 6 and info2["llr"] < 0.0
    assert PW._flat_supported(info2)
    assert not PW._flat_supported(dict(info2, dates=PW.FLAT_DATES - 1))


# ------------------------------------------------- cost (round 7): exact
def _bb_rows(counts, widths, ncp):
    """The round-6 per-row Bayesian Blocks DP (the reference)."""
    c = np.asarray(counts, dtype=np.float64)
    w = np.asarray(widths, dtype=np.float64)
    M = c.size
    cumN = np.concatenate(([0.0], np.cumsum(c)))
    cumT = np.concatenate(([0.0], np.cumsum(w)))
    best = np.zeros(M)
    last = np.zeros(M, dtype=np.int64)
    for r in range(M):
        N = cumN[r + 1] - cumN[: r + 1]
        T = cumT[r + 1] - cumT[: r + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            fit = np.where(N > 0, N * (np.log(np.maximum(N, 1e-300)) - np.log(T)), 0.0)
        A = fit - ncp
        A[1:] += best[:r]
        j = int(np.argmax(A))
        last[r] = j
        best[r] = A[j]
    out = []
    r = M - 1
    while r >= 0:
        j = int(last[r])
        out.append((j, r))
        r = j - 1
    return out[::-1]


def test_bayesian_blocks_matrix_form_equals_the_row_form():
    rng = np.random.default_rng(7)
    for _ in range(80):
        occ = np.sort(rng.choice(1440, size=int(rng.integers(5, 300)), replace=False))
        mins = np.repeat(occ, rng.integers(1, 4, size=occ.size)).astype(float) + rng.random()
        wts = rng.choice([1.0, 0.5, 0.25, 1.0 / 3.0], size=mins.size)
        cts, wd, _ = PW._cells_minutes(mins, int(rng.integers(0, 1440)), wts)
        ncp = PW.ncp_prior(float(wts.sum()))
        assert PW.bayesian_blocks(cts, wd, ncp) == _bb_rows(cts, wd, ncp)
    keep = PW.BB_MATRIX_MAX
    try:                                      # the large-M (row) path is the reference itself
        PW.BB_MATRIX_MAX = 0
        assert PW.bayesian_blocks(cts, wd, ncp) == _bb_rows(cts, wd, ncp)
    finally:
        PW.BB_MATRIX_MAX = keep


def _sessions_ref(pts, off):
    keys = [None] * len(pts)
    grp = {}
    for i, p in enumerate(pts):
        if len(p) > 3 and p[3]:
            grp.setdefault((int((float(p[1]) + off) // 86400.0), str(p[3])), []).append(i)
        else:
            keys[i] = ("pt", i)
    for k, idx in grp.items():
        idx.sort(key=lambda j: float(pts[j][1]))
        sess, last = 0, None
        for j in idx:
            t = float(pts[j][1])
            if last is not None and t - last > PW.SESSION_GAP_MIN * 60.0:
                sess += 1
            keys[j] = (k[0], k[1], sess)
            last = t
    return keys


def test_vectorised_sessions_equal_the_per_group_loop():
    rng = np.random.default_rng(3)
    for _ in range(100):
        pts = []
        for i in range(int(rng.integers(0, 200))):
            t = _ts(int(rng.integers(0, 6)), float(rng.uniform(0, 1440)))
            src = str(rng.choice(["a", "b", "c", ""]))
            pts.append((0.0, t, 1.0, src) if rng.random() < 0.95 else (0.0, t, 1.0))
        assert PW._sessions(pts, OFF) == _sessions_ref(pts, OFF)


def test_lazy_established_fit_equals_the_eager_one():
    """fit_regime step 4b cross-validates the established law only when its
    in-sample windows already show a drift on the young dates: same records
    as cross-validating every time (a drift and a stable law)."""
    rng = np.random.default_rng(11)
    for moved in (False, True):
        pts = []
        for d in range(10):
            lo = 510.0 if (moved and d >= 8) else 540.0
            for s in range(3):
                m = float(rng.uniform(lo, lo + 21.0))
                pts.append((m, _ts(d, m), 1.0, f"10.0.0.{s}"))
        lazy = PW.fit_regime(_hist(pts), float(len(pts)), pts, tz_offset_s=OFF)
        est = [p for p in pts if PW._local_date(p[1], OFF) < PW._young_dates(pts, OFF)]
        eager = PW.fit_regime(_hist(pts), float(len(pts)), est, tz_offset_s=OFF, young=False)
        if moved:
            assert lazy.get("drift") and lazy["windows"] == eager["windows"]
            assert lazy["coverage"] == eager["coverage"] and lazy.get("cv") == eager.get("cv")
        else:
            assert not lazy.get("drift")
