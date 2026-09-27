"""B14 spec unit test (a): wall-clock false-alarm rate of the CUSUM bank.

N(0,1) residuals at dt = 900 s for 60 simulated days over 50 seeds: the
family (48-chart bank) alarm rate is <= 0.03 per entity-day (design 0.02).
The same test at dt = 60 s gives a rate within [0.5, 2]x of that, because the
thresholds come from ARLs in days (seq.h_gauss(k, 2400 d * 86400 / dt)).

At 900 s the bank runs through m_cp.bank_tick, the exact step the engine
calls (latch, onset and reset included), batch-vectorised over the 50 seeds.
At 60 s the same 3000 entity-days are 4.3M ticks, too slow for bank_tick's
per-tick latch bookkeeping, so the 48 charts run as a lean restart-at-alarm
CUSUM (S = max(0, S + side*psi - k), psi = clip(z, -3, 3), alarm at the
engine's m_cp.bank_h(60) thresholds, family alarms counted once per tick).
The 900 s lean run is checked against bank_tick so both paths agree.
"""
from __future__ import annotations

import numpy as np

from helpers import T0

from app.engines.behavior.lib import m_cp

SEEDS, DAYS = 50, 60


def bank_rate(dt: float, seed: int) -> float:
    n = int(round(DAYS * 86400 / dt))
    B = m_cp.new_bank((SEEDS,))
    h, phi = m_cp.bank_h(dt), np.zeros(m_cp.N_KEY)
    rng = np.random.default_rng(seed)
    alarms = 0
    X = rng.standard_normal((n, SEEDS, m_cp.N_KEY))
    for i in range(n):
        B, out = m_cp.bank_tick(B, X[i], T0 + i * dt, dt, phi, h)
        alarms += int(out["rise"].sum())
    return alarms / (SEEDS * DAYS)


def lean_rate(dt: float, seed: int, block: int = 2880) -> float:
    n = int(round(DAYS * 86400 / dt))
    h = m_cp.bank_h(dt).astype(np.float32)
    k = m_cp.CHART_K.astype(np.float32)
    side = m_cp.CHART_SIDE.astype(np.float32)
    rng = np.random.default_rng(seed)
    S = np.zeros((SEEDS, m_cp.N_CHARTS), dtype=np.float32)
    alarms = 0
    for c in range(0, n, block):
        psi = np.clip(rng.standard_normal((min(block, n - c), SEEDS, m_cp.N_KEY),
                                          dtype=np.float32), -m_cp.PSI_CLIP, m_cp.PSI_CLIP)
        for y in psi[:, :, m_cp.CHART_FEAT] * side - k:
            S += y
            np.maximum(S, 0.0, out=S)
            over = S >= h
            if over.any():
                alarms += int(over.any(axis=1).sum())
                S[over] = 0.0
    return alarms / (SEEDS * DAYS)


def test_a_null_rate_wall_clock():
    r900 = bank_rate(900.0, seed=1)
    assert r900 <= 0.03
    assert r900 >= 0.005                     # thresholds are not absurdly conservative either
    r900_lean = lean_rate(900.0, seed=2)
    assert 0.5 <= r900_lean / r900 <= 2.0    # the lean simulator matches the engine's bank
    r60 = lean_rate(60.0, seed=3)
    assert 0.5 <= r60 / r900 <= 2.0
    assert r60 <= 0.03
