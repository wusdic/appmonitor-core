"""Round 4: the evidence CUSUM threshold is derived from the entity's own
calibrated per-stream null (fusion.solve_h, used by the hourly audit), and
its ARL0 is verified by simulation.

The nominal h = (ln ARL - 3.07) / 0.94 realises the 33-day ARL only when
q_inst is exactly uniform and independent. A q stream whose -ln q is 1.5x
heavier (a meta ring 1.5 decades too light at 1e-3) alarms ~20x as often;
the audit's old step rule (+5 % per audit, at most +20 %) could not correct
that, and its null stream was selected by trust >= 0.5, i.e. by q_inst
itself. The solved h keeps the realised rate within [0.5, 2]x of target.
"""
from __future__ import annotations

import numpy as np

from app.engines.behavior import fusion as F

DT = 900.0
PER_DAY = 86400.0 / DT
TARGET = 1.0 / F.EVIDENCE_ARL_DAYS


def _realised(q: np.ndarray, h: float) -> float:
    """Onsets per day of the live (no reset) CUSUM, x target."""
    _, on = F.evidence_path(q, h)
    return on / (q.size / PER_DAY) / TARGET


def test_nominal_h_realises_the_arl_on_a_uniform_null():
    q = np.random.default_rng(1).random(int(3000 * PER_DAY))
    assert 0.5 <= _realised(q, F.evidence_h(DT)) <= 2.0


def _solved(kind_pow: float):
    hb = F.evidence_h(DT)
    long = np.random.default_rng(1).random(int(3000 * PER_DAY)) ** kind_pow
    out = []
    for sd in range(8):
        hist = np.random.default_rng(100 + sd).random(int(7 * PER_DAY)) ** kind_pow
        h, _ = F.solve_h(hist, hb, DT, np.random.default_rng(sd), TARGET)
        assert hb <= h <= F.H_MULT_MAX * hb + 1e-9
        out.append((h, _realised(long, h)))
    return hb, long, out


def test_solved_h_never_lowers_the_nominal_and_keeps_a_uniform_null():
    hb, long, out = _solved(1.0)
    assert sum(h == hb for h, _ in out) >= 6
    assert 0.5 <= float(np.median([r for _, r in out])) <= 2.0


def test_solved_h_restores_the_arl_of_an_anti_conservative_null():
    hb, long, out = _solved(1.5)
    assert _realised(long, hb) > 5.0                      # nominal h: ~20x
    assert 0.5 <= float(np.median([r for _, r in out])) <= 2.0


def test_reset_alarms_counts_restarted_runs():
    x = np.array([3.0, 3.0, -10.0, 6.0, 0.0, 6.0])
    # S: 3, 6* (restart), 0, 6* (restart), 0, 6*
    assert F.reset_alarms(x, 5.0, 100) == 3
    assert F.reset_alarms(x, 5.0, 1) == 2               # stops after max_count + 1
    assert F.reset_alarms(np.full(10, -1.0), 1.0, 100) == 0


def test_cusum_input_is_calibrated_on_the_keys_own_live_null():
    """Round 4 (FusionEngine._qcal): a key whose q_inst is anti-conservative
    (q = U^2: -ln q twice as heavy, the evidence CUSUM alarms ~90x as often)
    has its CUSUM input corrected to q^(1/v) once the live share of q <= 0.05
    is significantly above 0.05; a uniform q stream is left alone."""
    rng = np.random.default_rng(5)
    n = int(400 * PER_DAY)
    for power, band in ((2.0, True), (1.0, False)):
        q = rng.random(n) ** power
        st = {}
        adj = np.array([F.FusionEngine._qcal(st, "t", float(x), True, k * DT)
                        for k, x in enumerate(q)])
        half = n // 2
        raw_r = _realised(q[half:], F.evidence_h(DT))
        adj_r = _realised(adj[half:], F.evidence_h(DT))
        if band:
            assert raw_r > 5.0
            assert 0.5 <= adj_r <= 2.0, adj_r
        else:
            assert np.array_equal(adj, q)
    # an unobserved tick (quarantined period, warm-up) never moves the state
    st = {}
    F.FusionEngine._qcal(st, "t", 1e-9, False, 0.0)
    assert not st.get(F.QCAL)
