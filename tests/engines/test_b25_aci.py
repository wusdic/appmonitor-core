"""Round 4: adaptive conformal inference (ACI) on B25's single-tick decision.

The meta rings make q_all uniform only while the entity's null is
exchangeable with its ring (no drift, no warm-up / live shift, no cadence
switch, no dependence the ring cannot see). The ACI layer tracks the realised
exceedance rate per (key, tick type) and per (system, tick type) and shifts
e_day by 10^theta so the rate follows the budget (0.03 per entity-day at the
decision level), adapting only on committed, period-trusted, live rows and
rate-limited per key, tick type, level and hour, within [-1, 3] decades.
Multi-level: e_day <= 3 and 0.3 learn 10-100x faster than the decision
level and are extrapolated to it under a power-law distortion model.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from helpers import put_model

from app.engines.behavior import fusion as F
from app.engines.behavior.lib import grains as GR
from app.engines.behavior.lib import m_calib
from app.engines.behavior.lib.classkeys import SYSTEM_KEY

from test_b25_fusion import E, S, Rig, p_at

MULT = {t: GR.e_day_mult(t, 900.0, GR.CANONICAL) for t in ("h", "q")}


def _sim(a: float, days: int = 20, n_ent: int = 35, seed: int = 0, aci: bool = True):
    """35 entities at 900 s (canonical tick types h / q), q_all = U^a (a > 1:
    anti-conservative, P(q <= x) = x^(1/a)), 4-tick commit delay; realised
    rate of e_day <= 0.03 after a 2-day burn-in, x nominal, per tick type."""
    rng = np.random.default_rng(seed)
    sys_st = {"th": {}}
    keys = [{} for _ in range(n_ent)]
    hits = {"h": 0, "q": 0}
    n = {"h": 0, "q": 0}
    pend = []
    for tick in range(days * 96):
        ts = tick * 900.0
        tau = "h" if tick % 4 == 0 else "q"
        ast = F.SINGLE_E_DAY / MULT[tau]
        for i in range(n_ent):
            e_raw = rng.random() ** a * MULT[tau]
            th = F.aci_shift(sys_st["th"], keys[i].get("th"), tau, ast) if aci else 0.0
            if tick >= 2 * 96:
                n[tau] += 1
                hits[tau] += e_raw * 10.0 ** th <= F.SINGLE_E_DAY
            pend.append((tick + 4, i, e_raw, tau, ast, ts))
        while pend and pend[0][0] <= tick:
            _, i, e_raw, tau, ast, t0 = pend.pop(0)
            if aci:
                F.aci_update(sys_st, keys[i], tau, e_raw, ast, t0)
    return {t: hits[t] / (n[t] * F.SINGLE_E_DAY / MULT[t]) for t in hits}, sys_st


def test_aci_step_moves_by_eta_and_is_clipped():
    assert F.aci_step(0.0, True, 0.01, 0.1, -1.0, 3.0) == pytest.approx(0.099)
    assert F.aci_step(0.0, False, 0.01, 0.1, -1.0, 3.0) == pytest.approx(-0.001)
    assert F.aci_step(2.99, True, 0.0, 0.1, -1.0, 3.0) == 3.0
    assert F.aci_step(-0.99, False, 0.5, 0.1, -1.0, 3.0) == -1.0


def test_multilevel_aci_keeps_a_calibrated_null_in_band():
    ratio, st = _sim(1.0)
    for tau, r in ratio.items():
        assert 0.5 <= r <= 2.0, (tau, r)
    assert all(abs(v[-1]) < 0.5 for v in st["th"].values())


def test_multilevel_aci_restores_the_budget_of_an_anti_conservative_null():
    """q = U^1.5: without ACI 10.7x (h) / 21x (q) nominal; with it in band."""
    raw, _ = _sim(1.5, days=6, aci=False)
    assert raw["h"] > 5.0 and raw["q"] > 10.0
    ratio, st = _sim(1.5)
    for tau, r in ratio.items():
        assert 0.5 <= r <= 2.0, (tau, r)
    assert st["th"]["q"][-1] > 1.0


def test_shift_extrapolates_shallow_levels_by_depth():
    ast = 1e-4
    th = {"q": [0.5, 0.0, 0.0]}                 # only e_day <= 3 has moved
    d_dec, d_3 = -math.log10(ast), -math.log10(ast * 3.0 / 0.03)
    assert F.aci_shift(th, None, "q", ast) == pytest.approx(0.5 * d_dec / d_3)
    assert F.aci_shift(th, None, "q") == 0.0    # without alpha*: the decision level only
    assert F.aci_shift({"q": [0.0, 0.0, 9.0]}, None, "q", ast) == F.ACI_TH_MAX
    assert F.aci_shift({"q": 0.7}, {"q": [0.0, 0.0, 0.2]}, "q", ast) == pytest.approx(0.9)


def test_error_bursts_are_rate_limited():
    """An attack released by mistake (a run of alarm-level rows within an
    hour) moves each level by one step at most."""
    sys_st, key = {"th": {}}, {}
    for k in range(40):
        F.aci_update(sys_st, key, "q", 1e-6, 1e-4, 1000.0 + 60.0 * k)
    lv = sys_st["th"]["q"]
    for i, v in enumerate(lv):
        assert v <= F.ACI_ETA_LEVEL[i] * F.ACI_ETA_SYS + 1e-12
    # the next hour counts again
    F.aci_update(sys_st, key, "q", 1e-6, 1e-4, 1000.0 + 3600.0)
    assert sys_st["th"]["q"][-1] > lv[-1] if isinstance(lv, list) else True


# ------------------------------------------------------------ engine level
NULL = {"marg_int": 0.4, "novelty": 0.6}


def _sys_aci(rig):
    m = rig.store.get_model(S, SYSTEM_KEY, m_calib.MODEL)
    return (m or {}).get(F.ACI) or {}


def test_engine_observes_live_ticks_of_trusted_periods_only():
    """ACI observes a live tick at scoring time when the governor's PREVIOUS
    tick exists (finite trust) and did not quarantine the key: the state
    before the tick, never the quarantine its own alarm opens."""
    rig = Rig()
    for _ in range(12):
        rig.step({E: dict(NULL)}, training=True)
    assert _sys_aci(rig).get("n", 0) == 0                     # warm-up ticks never adapt
    for _ in range(12):
        rig.step({E: dict(NULL)}, trust=float("nan"))
    assert _sys_aci(rig).get("n", 0) == 1                     # only the first (after a warm-up tick)
    rig2 = Rig()
    for _ in range(12):
        rig2.step({E: dict(NULL)}, quarantine=1.0)
    assert _sys_aci(rig2).get("n", 0) == 0                    # an open incident / regime
    rig3 = Rig()
    for _ in range(12):
        rig3.step({E: dict(NULL)})
    assert _sys_aci(rig3)["n"] == 11                          # every tick after the first
    # an alarm tick is observed as an error although it opens its incident
    rig3.step({E: {"marg_int": p_at(1e-4), "novelty": 0.6}}, quarantine=1.0)
    assert _sys_aci(rig3)["n_err"] == 1


def test_engine_alarm_carries_the_shift_and_raw_e_day():
    rig = Rig()
    for _ in range(8):
        rig.step({E: dict(NULL)})
    rig.store.get_model(S, SYSTEM_KEY, m_calib.MODEL)[F.ACI]["th"]["tick|900"] = [0.0, 0.0, 1.0]
    rig.step({E: {"marg_int": p_at(1e-3), "novelty": 0.6}})
    a = rig.alarm()
    assert a is not None and a["aci"] == pytest.approx(1.0, abs=1e-2)
    assert a["e_day"] == pytest.approx(a["e_day_raw"] * 10.0 ** a["aci"], rel=1e-6)
    # a shift of 3 decades takes the same evidence off the single-tick path
    # (the evidence CUSUM, which ACI does not touch, still alarms)
    rig.store.get_model(S, SYSTEM_KEY, m_calib.MODEL)[F.ACI]["th"]["tick|900"] = [0.0, 0.0, 3.0]
    rig.step({E: {"marg_int": p_at(1e-3), "novelty": 0.6}})
    a = rig.alarm()
    assert a is None or F.PATH_SINGLE not in a["paths"]


def test_a_sub_alarm_attack_does_not_raise_its_own_keys_shift():
    """Pack A T7 (one rare-resource access per tick for a day, below the
    alarm level): a key-level shallow ACI level learnt it and extrapolated it
    to the decision depth. The key tier tracks the decision level only (its
    errors are alarms, which quarantine the key); the shallow levels are
    pooled over the system, where 1 of 35 keys moves them little."""
    sys_st, keys = {"th": {}}, [{} for _ in range(35)]
    rng = np.random.default_rng(4)
    ast = F.SINGLE_E_DAY / MULT["q"]
    th0 = None
    for tick in range(96):                                    # one day at 900 s
        ts = tick * 900.0
        for i in range(35):
            if i == 0:
                e_raw = 1.0                                  # e_day 1: under 3, over 0.3
            else:
                e_raw = rng.random() * MULT["q"]
            F.aci_update(sys_st, keys[i], "q", e_raw, ast, ts)
        if tick == 0:
            th0 = F.aci_shift(sys_st["th"], keys[0].get("th"), "q", ast)
    k_th = keys[0]["th"]["q"]
    assert k_th[0] == 0.0 and k_th[1] == 0.0 and k_th[-1] <= 0.0
    # the attacked key's decision-level shift is not raised by the attack
    assert F.aci_shift(sys_st["th"], keys[0].get("th"), "q", ast) < 0.35
