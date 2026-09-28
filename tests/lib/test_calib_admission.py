"""Round 4: null-ring admission independent of the row's own score, and the
contamination-bounded tail (lib/gating.period_weight / release_weight,
lib/calib.robust_tail; B24 and B25 learners run with null_ring=True).

A calibration ring estimates the null distribution of a score. If a row's
admission depends on its own score (behavior.trust carries the governor's
evidence factor clip(log10(e_inst / 0.1), 0, 1) and [no alarm], both
functions of the row's fused p), the ring's upper tail - the part the GPD is
fitted to - is thinned, the issued p become anti-conservative, which thins
more rows: a feedback loop. These tests simulate an exact null and measure
the realised exceedance rate at alpha.
"""
from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engines"))

from app.engines.behavior.lib import calib, gating, m_calib  # noqa: E402

ALPHAS = (1e-3, 3e-4)


def _stream(seed: int, n_warm: int, n_live: int, contam: float, fit, admit_rule: str,
            dt: float = 900.0):
    """One M = 256 ring, commit delay 4, refit every 16 admissions (B24 / B25
    mechanics without the engines). Null: Exp(1) in log10 units (-log10 U).
    Contamination: an attack-level score (beyond the null's 1e-4 quantile)
    admitted like any row of a trusted period. Returns realised / nominal at
    ALPHAS on the clean live rows."""
    rng = np.random.default_rng(seed)
    r = calib.Ring()
    for i in range(n_warm):
        r.add(-math.log10(rng.random()), float(i))
    r.gpd = fit(r)
    cnt = 0
    hits = np.zeros(len(ALPHAS))
    n_clean = 0
    pend = []
    for t in range(n_warm, n_warm + n_live):
        foreign = rng.random() < contam
        x = -math.log10(rng.random())
        if foreign:
            x = 4.0 + 3.0 * rng.exponential(1.0)
        p = m_calib.p_value(r, x, rng.random())
        if not foreign:
            n_clean += 1
            hits += np.array([p <= a for a in ALPHAS])
        if admit_rule == "row_trust":            # B28's trust of the row itself
            e_inst = p * 86400.0 / dt
            w = min(1.0, max(0.0, math.log10(max(e_inst, 1e-300) / 0.1)))
            if e_inst <= 0.03:
                w = 0.0                          # [no alarm at t]
        else:
            w = 1.0                              # period trust: not quarantined
        pend.append((t, x, w))
        if len(pend) > 4:
            tt, xx, ww = pend.pop(0)
            if rng.random() < ww:
                r.add(xx, float(tt))
                cnt += 1
                if cnt >= calib.GPD_REFIT_TICKS:
                    r.gpd = fit(r)
                    cnt = 0
    return hits / (n_clean * np.asarray(ALPHAS))


def _mean_ratio(**kw):
    return np.mean([_stream(seed=s, **kw) for s in range(3)], axis=0)


def test_row_trust_admission_is_anti_conservative_period_admission_is_not():
    """Exact null, no contamination: admitting rows with their own trust
    gives > 2x nominal at 1e-3 / 3e-4 (measured 4.5 / 6.8x, 11x at 1e-4);
    period-only admission with the robust tail stays in [0.5, 2]."""
    bad = _mean_ratio(n_warm=256, n_live=20000, contam=0.0, fit=calib.robust_tail,
                      admit_rule="row_trust")
    good = _mean_ratio(n_warm=256, n_live=20000, contam=0.0, fit=calib.robust_tail,
                       admit_rule="period")
    assert (bad > 2.0).all(), bad
    assert ((good >= 0.5) & (good <= 2.0)).all(), good


def test_one_percent_contamination_stays_in_band_with_the_trimmed_tail():
    """H0 with 1 % attack-level rows admitted (a released attack, a warm-up
    extreme): the contamination-bounded tail keeps the realised rate in
    [0.5, 2]x; the untrimmed fits bloat the tail (conservative < 0.5x)."""
    good = _mean_ratio(n_warm=256, n_live=20000, contam=0.01, fit=calib.robust_tail,
                       admit_rule="period")
    raw = _mean_ratio(n_warm=256, n_live=20000, contam=0.01, fit=calib.fit_tail,
                      admit_rule="period")
    assert ((good >= 0.5) & (good <= 2.0)).all(), good
    assert (raw < 0.5).all(), raw


def test_trim_count_respects_the_contamination_bound():
    rng = np.random.default_rng(3)
    y = np.sort(rng.exponential(0.43, 26))
    assert calib.trim_count(y, calib.TRIM_ALPHA, 6) == 0 or y[-1] > 0.43 * 6
    y2 = np.sort(np.concatenate([y[:18], [20.0] * 8]))
    assert calib.trim_count(y2, calib.TRIM_ALPHA, 6) == 6           # capped
    assert calib.trim_count(y2, calib.TRIM_ALPHA, 0) == 0
    # a clean exponential tail loses an entry in ~3.5 % of the fits (the
    # rank-based scale is noisy; the calibration cost of it is inside the
    # realised rates measured above)
    touched = sum(calib.trim_count(np.sort(rng.exponential(0.43, 26)), calib.TRIM_ALPHA, 6) > 0
                  for _ in range(2000))
    assert touched / 2000 < 0.05


def test_robust_tail_on_a_clean_ring_is_the_predictive_exponential():
    rng = np.random.default_rng(5)
    r = calib.Ring(scores=-np.log10(rng.random(256)), ts=np.arange(256.0))
    t = calib.robust_tail(r)
    assert t is not None and t.xi >= 1.0 / 26 - 1e-12
    assert 0.08 <= t.rate <= 0.11
    assert calib.robust_tail(calib.Ring(scores=np.arange(5.0), ts=np.arange(5.0))) is None


# ------------------------------------------------------------ engine level
def test_b24_live_null_is_calibrated_although_trust_follows_the_rows_own_p():
    """B24 end to end: a live null stream whose governor trust is computed
    from each row's own issued p (as B28's evidence factor and [no alarm]
    are). With period admission (null_ring=True) the realised rate of
    p <= 1e-3 is in [0.5, 2]x; with the row's trust (null_ring=False, the
    rule before round 4) the ring thins its own tail and it is > 2x."""
    from helpers import run_engine, set_trust
    from test_b24_calibration import E, S, Rig
    from app.engines.behavior.lib import emit

    def run(null_ring: bool) -> float:
        rig = Rig(daypart="wd_day")
        rng = np.random.default_rng(21)
        for _ in range(300):
            rig.step({E: {"marg_int": float(rng.exponential())}}, training=True)
        for lr in rig.eng._learners.values():
            lr.null_ring = null_ring
        hits = n = 0
        for _ in range(12000):
            ts = rig.t
            emit.write_scores(rig.store, S, E, ts, {"marg_int": float(rng.exponential())})
            rig.write_tctx(E, ts, rig.dt)
            run_engine(rig.eng, rig.store, ts, dt=rig.dt)
            p = rig.p(E, "marg_int", ts)
            e_inst = p * 86400.0 / rig.dt              # B28: e_inst, trust_prov, [no alarm]
            tr = 0.0 if e_inst <= 0.03 else min(1.0, max(0.0, math.log10(e_inst / 0.1)))
            set_trust(rig.store, S, E, [ts], tr, quarantine=0.0)
            rig.t = ts + rig.dt
            n += 1
            hits += p <= 1e-3
        return hits / (n * 1e-3)

    good = run(True)
    bad = run(False)
    print("realised / nominal at 1e-3: period", good, "row trust", bad)
    assert 0.5 <= good <= 2.0, good
    assert bad > 2.0, bad
