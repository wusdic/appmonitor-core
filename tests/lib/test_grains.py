"""lib/grains (spec v2.1, docs/lib3/cadence.md §2, §6, §7)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.engines.behavior.lib import combine
from app.engines.behavior.lib import detectors as det
from app.engines.behavior.lib import grains as GR

H0 = 1_741_536_000.0          # 2025-03-09 16:00 UTC (an epoch multiple of 3600)
CAN = "canonical"
CFG = {"grain_mode": CAN, "tz": "Asia/Shanghai"}


def _decisions(dt, g, n, t0=H0, mode=CAN):
    return [t0 + k * dt for k in range(1, n + 1) if GR.decision(t0 + k * dt, dt, g, mode)]


@pytest.mark.parametrize("dt, every_h, every_q", [(60, 60, 15), (300, 12, 3), (900, 4, 1),
                                                  (3600, 1, None)])
def test_decision_ticks_at_every_cadence(dt, every_h, every_q):
    n = int(4 * 86400 / dt)
    hs = _decisions(dt, "h", n)
    assert len(hs) == n // every_h
    assert np.allclose(np.diff(hs), 3600.0)
    if every_q is None:
        assert not GR.observable("q", dt, CAN) and _decisions(dt, "q", n) == []
    else:
        qs = _decisions(dt, "q", n)
        assert len(qs) == n // every_q and np.allclose(np.diff(qs), 900.0)
    # H decision ticks are Q decision ticks too (nested boundaries)
    if every_q is not None:
        assert set(hs) <= set(_decisions(dt, "q", n))


def test_misaligned_ticks_and_cadence_switch():
    off = 433.0                                  # 12:07:13-style phase
    dt = 900.0
    ticks = [H0 + off + k * dt for k in range(1, 400)]
    hs = [t for t in ticks if GR.decision(t, dt, "h", CAN)]
    gaps = np.diff(hs)
    assert np.all(np.abs(gaps - 3600.0) <= dt)   # G +- dt apart
    for t in hs:                                 # the interval holds an epoch multiple of G
        assert math.floor(t / 3600.0) > math.floor((t - dt) / 3600.0)
    # 3600 -> 900 switch at an aligned t0: next H decision is t0 + 3600
    t0 = H0 + 10 * 3600.0
    after = [t0 + k * 900.0 for k in range(1, 9)]
    hd = [t for t in after if GR.decision(t, 900.0, "h", CAN)]
    assert hd[0] == t0 + 3600.0


def test_tick_mode_degenerates_to_v2():
    for dt in (60.0, 900.0, 3600.0):
        assert GR.decision(H0 + 7.0, dt, "h", "tick") and not GR.decision(H0, dt, "q", "tick")
        assert GR.grain_s("h", dt, "tick") == dt and not GR.observable("q", dt, "tick")
        assert GR.series("feature.nat", "h", "tick") == "feature.nat"
        assert GR.period_s("cusum", dt, "tick") == dt
        assert GR.tick_type(H0 + 5.0, dt, "tick") == "h"
        for q in (1e-3, 0.5, 2.5e-7):
            assert GR.e_day_tick(q, H0, dt, "tick") == combine.e_day(q, dt)
        assert GR.evidence_arl_ticks("t", dt, "tick") == pytest.approx(33 * 86400 / dt)
        assert GR.scored_mask(H0, dt, "h", {"grain_mode": "tick"}).all()
    assert GR.mode_of({}) == "tick" and GR.mode_of({"grain_mode": CAN}) == CAN
    assert GR.series("feature.nat", "q", CAN) == "feature.nat.q"
    assert GR.series("behavior.z", "h", CAN) == "behavior.z"
    assert GR.series("behavior.z", "q", CAN) == "behavior.z.q"


@pytest.mark.parametrize("dt", [60.0, 300.0, 900.0, 3600.0])
def test_tick_types_sum_and_budget(dt):
    tot = sum(GR.n_per_day(t, dt, CAN) for t in GR.TAUS)
    assert tot == pytest.approx(86400.0 / dt)
    # beta renormalised over the types present
    bs = [GR.beta(t, dt, CAN) for t in GR.TAUS]
    assert sum(bs) == pytest.approx(1.0)
    # expected null single-tick alarms: sum_tau n_tau (0.03 beta_tau / n_tau) = 0.03
    exp = sum(GR.n_per_day(t, dt, CAN) * min(1.0, 0.03 / GR.e_day_mult(t, dt, CAN))
              for t in GR.TAUS if GR.n_per_day(t, dt, CAN) > 0)
    assert exp == pytest.approx(0.03)
    # the count of each tick type over a day matches n_per_day
    n = int(86400 / dt)
    types = [GR.tick_type(H0 + k * dt, dt, CAN) for k in range(1, n + 1)]
    for t in GR.TAUS:
        assert types.count(t) == pytest.approx(GR.n_per_day(t, dt, CAN))


def test_e_day_table_of_the_design():
    # cadence.md §7.3 table
    assert GR.e_day_mult("h", 3600.0, CAN) == pytest.approx(24.0)
    assert GR.e_day_mult("h", 900.0, CAN) == pytest.approx(36.0)
    assert GR.e_day_mult("q", 900.0, CAN) == pytest.approx(216.0)
    assert GR.e_day_mult("h", 60.0, CAN) == pytest.approx(48.0)
    assert GR.e_day_mult("q", 60.0, CAN) == pytest.approx(288.0)
    assert GR.e_day_mult("t", 60.0, CAN) == pytest.approx(5376.0)
    assert GR.e_day_mult("t", 300.0, CAN) == pytest.approx(768.0)
    # identical to v2 at 3600 s
    assert GR.e_day_tick(1e-3, H0 + 3600, 3600.0, CAN) == pytest.approx(combine.e_day(1e-3, 3600.0))


def test_periods_and_evidence_arls():
    assert GR.period_s("cusum", 900.0, CAN) == 3600.0
    assert GR.period_s("marg_int_q", 60.0, CAN) == 900.0
    assert GR.period_s("novelty", 60.0, CAN) == 60.0
    assert GR.period_s("cusum", 3600.0, CAN) == 3600.0
    # S_h: 66 d in hours at every cadence
    for dt in (60.0, 900.0, 3600.0):
        assert GR.evidence_arl_ticks("h", dt, CAN) == pytest.approx(66 * 24)
    assert GR.evidence_arl_ticks("t", 900.0, CAN) == pytest.approx(66 * 96)
    assert GR.e_inst({"t": 1e-4, "h": 1e-3}, 900.0, CAN) == pytest.approx(min(1e-4 * 96, 1e-3 * 24))
    assert GR.e_inst({"t": float("nan")}, 900.0, CAN) != GR.e_inst({"t": float("nan")}, 900.0, CAN)
    assert det.acc_level(1e-3, "cusum", 900.0, period_s=3600.0) == pytest.approx(
        math.log(1e3) / math.log(det.arl_days("cusum") * 24))


def test_row_tctx_is_the_window_midpoint():
    t = H0                                       # 2025-03-10 00:00 local (UTC+8)
    tc_h = GR.row_tctx(t, "h", 900.0, CFG)
    tc_q = GR.row_tctx(t, "q", 900.0, CFG)
    assert tc_h["hour_local"] == pytest.approx(23.5)
    assert tc_q["hour_local"] == pytest.approx(23.875)
    tick = GR.row_tctx(t, "h", 900.0, {"grain_mode": "tick", "tz": "Asia/Shanghai"})
    assert tick["hour_local"] == pytest.approx(0.0)


def test_span_decisions_are_local_time_aligned():
    cfg = dict(CFG)
    # local midnight 2025-03-10 00:00 +08:00 = H0
    mid = H0
    assert GR.span_decision(mid, 900.0, "duty_cycle", cfg)
    assert not GR.span_decision(mid + 900.0, 900.0, "duty_cycle", cfg)
    assert GR.span_decision(mid + 6 * 3600.0, 3600.0, "periodicity", cfg)
    assert not GR.span_decision(mid + 5 * 3600.0, 3600.0, "periodicity", cfg)
    # 60-s ticks: exactly one per 6 h
    n = sum(GR.span_decision(mid + k * 60.0, 60.0, "periodicity", cfg) for k in range(1, 1441))
    assert n == 4
    m = GR.scored_mask(mid, 900.0, "h", cfg)
    from app.engines.behavior.lib import features as F
    assert m[F.FEATURE_INDEX["duty_cycle"]] and m[F.FEATURE_INDEX["periodicity"]]
    mq = GR.scored_mask(mid, 900.0, "q", cfg)
    assert not mq[F.FEATURE_INDEX["duty_cycle"]] and mq[F.FEATURE_INDEX["flows"]]
    # DST (Europe/Berlin, 2025-03-30): local midnight of the 23-h day still found
    ber = {"grain_mode": CAN, "tz": "Europe/Berlin"}
    t_mid = 1743289200.0                         # 2025-03-30 00:00 CET
    assert GR.span_decision(t_mid, 900.0, "duty_cycle", ber)
    t_next = 1743372000.0                        # 2025-03-31 00:00 CEST
    assert GR.span_decision(t_next, 900.0, "duty_cycle", ber)
    assert not GR.span_decision(t_next - 900.0, 900.0, "duty_cycle", ber)


def test_transfer_moments():
    v = GR.v_from_omega(np.array([0.0, 4.0, 8.0, 20.0]))
    assert np.allclose(v, [1.0, 4.0, 7.0, 7.0])
    mu, r = GR.transfer_nb(2.0, 8.0, 4.0)
    assert mu == 2.0 and float(r) == pytest.approx(2.0)
    p, c = GR.transfer_bb(0.3, 99.0, 4.0)
    assert p == 0.3 and float(c) == pytest.approx(24.0)
    _, c2 = GR.transfer_bb(0.3, 10.0, 7.0)
    assert float(c2) == GR.C_T_MIN
    loc, sc, df = GR.transfer_t(1.0, 0.5, 10.0, 4.0, -0.1)
    assert float(loc) == pytest.approx(0.9) and float(sc) == pytest.approx(1.0) and df == 10.0
    d = GR.jensen_delta(0.5, 10.0, 4.0)
    assert float(d) == pytest.approx(-3.0 * 0.25 * 10.0 / 8.0 / 2.0)
    # EB: no data -> parent; lots of data -> own ratio
    assert float(GR.omega_eb(0.0, 0.0, 0.0, 4.0)) == 4.0
    assert float(GR.omega_eb(1e6, 1e6, 1e6, 4.0)) == pytest.approx(1.0, rel=1e-3)
    assert float(GR.delta_eb(0.0, 0.0, 0.2)) == pytest.approx(0.2)
    assert float(GR.delta_eb(-1e5, 1e6, 0.2)) == pytest.approx(-0.1, rel=1e-3)


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9, 1.0])
def test_omega_recovers_quarter_overdispersion(rho):
    """cadence.md §6.3: a gamma/lognormal-mixed Poisson with intra-hour
    correlation rho; the paired-hour estimator recovers the Q-grain
    overdispersion 1/r_Q = v / r_H."""
    rng = np.random.default_rng(7)
    n, m, e_q = 40000, GR.M, 15.0
    lam0, sd = 2.0, 0.6
    zh = rng.standard_normal(n)[:, None]
    zj = rng.standard_normal((n, m))
    lam = lam0 * np.exp(sd * (math.sqrt(rho) * zh + math.sqrt(1.0 - rho) * zj) - sd * sd / 2.0)
    x = rng.poisson(lam * e_q)
    lam_h = lam.mean(axis=1)
    mu = float(lam_h.mean())
    r_h = mu * mu / float(lam_h.var())                  # H excess: Var(mean lambda) = mu^2 / r_H
    a, b, _ = GR.paired_stats("nb", x.T, np.full((m, n), e_q), x.sum(axis=1), m * e_q,
                              {"mean": mu, "r": r_h})
    omega = float(np.sum(a) / np.sum(b))
    v_hat = float(GR.v_from_omega(omega)) if omega <= GR.OMEGA_MAX else 1 + 0.75 * omega
    v_true = float(lam.var()) / float(lam_h.var())      # (1/r_Q) / (1/r_H)
    assert v_hat == pytest.approx(v_true, rel=0.03)


def test_paired_stats_t_and_bb():
    rng = np.random.default_rng(3)
    y = rng.normal(0.0, 1.0, size=(4, 2000))
    a, b, d = GR.paired_stats("t", y, np.ones_like(y), y.mean(axis=0), 1.0, {"var": 0.25})
    assert float(np.mean(a)) == pytest.approx(1.0, rel=0.05) and np.allclose(b, 0.25)
    assert np.allclose(d, 0.0)
    k = rng.binomial(40, 0.3, size=(4, 2000))
    a, b, _ = GR.paired_stats("bb", k, np.full(k.shape, 40.0), k.sum(axis=0), 160.0,
                              {"mean": 0.3, "c": 50.0})
    assert abs(float(np.mean(a))) < 2e-3            # no within-hour excess: a ~ 0
