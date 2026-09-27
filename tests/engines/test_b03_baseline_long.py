"""B03 BaselineEngine over long horizons (3600-s ticks): the golden anchor
from low-risk weekly reference snapshots, hour-of-week location cells after
4 weeks of commits, and the per-feature half-life selection. The B01 / B28
stand-in is the one of test_b03_baseline."""
from __future__ import annotations

import numpy as np
import pytest

from helpers import T0, make_store, run_engine

from app.engines.behavior.baseline import BaselineEngine
from app.engines.behavior.lib import features as F
from app.engines.behavior.lib import m_baseline as MB

from test_b03_baseline import BYTES, REQ, S, E, feed, model, nat_row, tctx

H = 3600.0


# ------------------------------------------------------- long horizons
def test_golden_anchor_and_hour_of_week_cells():
    """30 days at 3600 s: Mondays run at 3x. Low-risk weeks give weekly
    reference snapshots (golden = their median); a high-risk entity gets
    none. After 4 weeks the current anchor refines Monday's location with its
    bin168 cell."""
    rng = np.random.default_rng(24)
    st, eng = make_store(), BaselineEngine()
    risky = "10.0.0.9"

    def rate(i, t):
        return 30.0 if tctx(t, H)["dow"] == 0 else 10.0

    t = T0
    for i in range(30 * 24):
        t = T0 + i * H
        for e, risk in ((E, 10.0), (risky, 50.0)):
            feed(st, t, nat_row(rng, rate(i, t), dt=H), e=e, dt=H)
            st.add_vec(S, e, "behavior.risk", t, np.array([risk], dtype=np.float32))
        run_engine(eng, st, t, dt=H)
    m = model(st)
    assert MB.has_golden(m) and 2 <= len(m["golden"]["snaps"]) <= 4
    assert not MB.has_golden(model(st, risky))
    assert np.array_equal(MB.golden_offset(st, S, E), np.zeros(F.FEATURE_DIM))
    assert MB.golden_offset(st, S, risky) is None
    # the reference predictive is the golden anchor (live reference stats unused)
    tc = tctx(t, H)
    p_ref = MB.predictive(st, S, E, tc, anchor="reference", model=m)
    alt = dict(m)
    alt["reference"] = MB.new_anchor(week=False, select=False)
    p_alt = MB.predictive(st, S, E, tc, anchor="reference", model=alt)
    assert np.allclose(p_ref.mean, p_alt.mean)
    # eval poisoning gate accessor: live model and its plain-data snapshot agree
    mu_g, sd_g = MB.anchor_summary(m, "golden", "http_requests")
    mu_c, sd_c = MB.anchor_summary(m, "current", "http_requests")
    snap = {"current": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in m["current"]._asdict().items()}}
    assert MB.anchor_summary(snap, "current", "http_requests") == pytest.approx((mu_c, sd_c))
    assert 9.0 < mu_g < 16.0 and 9.0 < mu_c < 16.0 and sd_g > 0.0
    assert np.isnan(MB.anchor_summary(model(st, risky), "golden", REQ)[0])
    # hour-of-week cells: a Monday 10:00 predictive sits near Monday's level
    cur = m["current"]
    assert cur.week_mode() and MB.maturity(m)["mode"] == "bin168"
    mon = next(T0 + k * H for k in range(30 * 24 - 7 * 24, 30 * 24)
               if tctx(T0 + k * H, H)["dow"] == 0 and int(tctx(T0 + k * H, H)["hour_local"]) == 10)
    tcm = tctx(mon, H)
    pr = MB.predictive(st, S, E, tcm, model=m)
    b48 = MB.bucket_means(cur)[0][tcm["bin48"], REQ]
    assert pr.mode == "bin168" and pr.mu[REQ] > b48 + 3.0 and pr.mu[REQ] > 20.0
    tcw = tctx(mon + 2 * 86400.0, H)                           # Wednesday stays near 10
    assert MB.predictive(st, S, E, tcw, model=m).mu[REQ] < 14.0


def test_half_life_selection_prefers_short_memory_for_a_shifting_feature():
    """bytes_up jumps 8x every 2 days (a shifting level), http_requests is
    stationary: the pinball loss picks 7 d for the former and keeps a longer
    half-life for the latter."""
    rng = np.random.default_rng(25)
    st, eng = make_store(), BaselineEngine()
    t = T0
    for i in range(16 * 24):
        t = T0 + i * H
        x = nat_row(rng, 10.0, dt=H, bytes_=1e5 * (8.0 if (i // 48) % 2 else 1.0))
        feed(st, t, x, dt=H)
        run_engine(eng, st, t, dt=H)
    hl = MB.hl_days(model(st))
    assert set(hl) <= set(MB.HL_CAND_DAYS)
    assert hl[BYTES] == 7.0
    assert hl[REQ] >= 14.0
