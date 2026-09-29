"""Round 4 (evaluator): bounded charts and the end-of-shift test of B14's
latch (lib/m_cp._latch_tick), and the bounded B25 evidence CUSUM.

Symptom: after a 24-h attack the cusum / mcusum latches held for days (pack
A seed 0: T9b 58 h, T12 36 h, T16 > 28 h after the attack) because the clean
clock only advanced on rows where NO alarmed chart rose and S was unbounded;
B25's evidence CUSUM kept S = 415 / 2146 (pack B T14 / T15) and drained at
-2 per tick for days."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.engines.behavior import fusion as FU
from app.engines.behavior.lib import m_cp, seq

DT = 3600.0


def _run(delta, dur, post, B=200, seed=0, persist=False, end_test=True, monkeypatch=None):
    if not end_test:
        orig = m_cp._latch_tick
        monkeypatch.setattr(m_cp, "_latch_tick",
                            lambda L, S, So, h, z, now, dt, k=None, mu0=0.0, zero=0.0:
                            orig(L, S, So, h, z, now, dt))
    rng = np.random.default_rng(seed)
    st, mc = m_cp.new_bank((B,)), m_cp.new_mc((B,))
    h, hmc = m_cp.bank_h(DT), m_cp.mcusum_h(DT)
    phi = np.zeros(m_cp.N_KEY)
    rel = {"bank": np.full(B, np.nan), "mc": np.full(B, np.nan)}
    seen = {"bank": np.zeros(B, bool), "mc": np.zeros(B, bool)}
    top = 0.0
    t = 0.0
    for i in range(100 + dur + post):
        x = rng.standard_normal((B, m_cp.N_KEY))
        if 100 <= i < 100 + dur or (persist and i >= 100):
            x[:, :3] += delta
        t += DT
        st, _ = m_cp.bank_tick(st, x, t, DT, phi, h)
        mc, _ = m_cp.mc_tick(mc, np.clip(x, -3, 3), t, DT, hmc)
        top = max(top, float(np.max(st["S"] / h)),
                  float(np.max(np.sqrt(np.sum(mc["S"] ** 2, axis=-1)) / hmc)))
        for name, on in (("bank", st["latch"]["on"]), ("mc", mc["latch"]["on"])):
            seen[name] |= on
            if i >= 100 + dur:
                m = np.isnan(rel[name]) & ~on & seen[name]
                rel[name][m] = i - (100 + dur) + 1
    return rel, seen, top


def test_statistics_are_bounded_at_four_h():
    _, _, top = _run(3.0, 48, 0, B=20)
    assert top <= m_cp.S_CAP_MULT * (1 + 1e-5)


def test_a_loud_shift_releases_its_latch_within_hours_of_its_end(monkeypatch):
    rel, seen, _ = _run(2.0, 24, 150)
    assert seen["bank"].all() and seen["mc"].all()
    assert np.nanmedian(rel["bank"]) <= 12 and np.nanmedian(rel["mc"]) <= 5
    # the old rule alone (2 x span of clean rows): ~50 / 23 rows
    rel0, _, _ = _run(2.0, 24, 150, end_test=False, monkeypatch=monkeypatch)
    assert np.nanmedian(rel0["bank"]) >= 3 * np.nanmedian(rel["bank"])
    assert np.nanmedian(rel0["mc"]) >= 3 * np.nanmedian(rel["mc"])


@pytest.mark.parametrize("delta", [1.0, 2.0])
def test_a_persisting_shift_keeps_its_latch(delta):
    rel, seen, _ = _run(delta, 0, 300, persist=True)
    assert seen["bank"].mean() > 0.95
    # false releases (each re-alarms within h / (delta - k) rows) stay rare
    assert np.isfinite(rel["bank"]).mean() <= 0.12
    assert np.isfinite(rel["mc"]).mean() <= 0.06


def test_a_capped_chart_does_not_go_clean():
    """At the cap S stops rising; without counting the cap as growth, a
    persisting loud shift was released by the clean clock every 2 spans."""
    rel, _, _ = _run(3.0, 0, 200, B=50, persist=True)
    assert np.isfinite(rel["bank"]).mean() <= 0.1


def test_replay_reproduces_the_capped_bank_exactly():
    rng = np.random.default_rng(3)
    st = m_cp.new_bank()
    h = m_cp.bank_h(DT)
    phi = np.zeros(m_cp.N_KEY)
    params = {"h": h, "h_mc": m_cp.mcusum_h(DT), "W": np.eye(m_cp.MC_D),
              "Sigma": np.eye(m_cp.MC_D), "dt": DT}
    rstate = {"S": np.zeros(m_cp.N_CHARTS), "mc": np.zeros(m_cp.MC_D),
              "prev": np.full(m_cp.N_KEY, np.nan), "phi": phi}
    step = m_cp.replay_step(params)
    for i in range(80):
        x = rng.standard_normal(m_cp.N_KEY) + (3.0 if 10 <= i < 60 else 0.0)
        S_before = st["S"].copy()
        st, out = m_cp.bank_tick(st, x, (i + 1) * DT, DT, phi, h)
        inp = {"x": x, "phi": phi, "adjacent": True}
        if out["reset"]:
            inp["mode"] = ("zero_S",)
        rstate, _ = step(rstate, inp)
        assert np.allclose(rstate["S"], st["S"]), i
        del S_before


def test_evidence_cusum_is_bounded_and_drains_after_the_evidence():
    h = 8.0
    cap = FU.evidence_cap(h)
    assert cap == seq.EVIDENCE_CAP_MULT * h
    S = 0.0
    for _ in range(50):                       # a long attack: q = 1e-6 per tick
        S, _, al = FU.evidence_update(S, 1e-6, h)
    assert al and S == cap
    n = 0
    while S >= h:                             # null ticks: -ln q = 1 on average
        S, _, _ = FU.evidence_update(S, math.exp(-1.0), h)
        n += 1
    assert n <= math.ceil((cap - h) / 2.0) + 1
    # the unbounded CUSUM would have kept 50 x (13.8 - 3) = 540 and drained for 266 ticks
    Su = 0.0
    for _ in range(50):
        Su = seq.evidence_cusum_step(Su, 1e-6)
    assert Su > 500


def test_evidence_path_matches_the_bounded_recursion():
    rng = np.random.default_rng(0)
    q = rng.uniform(size=3000)
    q[1000:1040] = 1e-8                       # an attack excursion
    h = 9.0
    S, onsets = FU.evidence_path(q, h)
    s, ref, prev, n_on = 0.0, [], 0.0, 0
    for v in q:
        s2, _, al = FU.evidence_update(s, v, h)
        n_on += int(al and s < h)
        s = s2
        ref.append(s)
    assert np.allclose(S, ref) and onsets == n_on


# ------------------------------------------- BOCPD alarm level (round 4)
def _boc_rate(gen, thr, clip, days=60, n=4, seed=0):
    from app.engines.behavior import changepoint as C
    rng = np.random.default_rng(seed)
    onsets = 0
    for _ in range(n):
        st, on = C.bocpd_new(), False
        for _h in range(days * 24):
            x = gen(rng)
            if clip:
                x = np.clip(x, -seq.PSI_CLIP, seq.PSI_CLIP)
            st = C.bocpd_step(st, x)
            cp = C.bocpd_prob(st) if C.bocpd_steps(st) > C.BOC_WINDOW_H else math.nan
            a = cp >= thr
            onsets += int(a and not on)
            on = a
    return onsets / (days * n)


def test_bocpd_alarm_level_is_the_ville_bound_of_its_budget():
    from app.engines.behavior import changepoint as C
    from app.engines.behavior.lib.detectors import DETECTOR_INFO
    budget = DETECTOR_INFO["bocpd"]["budget_per_day"]
    assert C.bocpd_pm(C.BOC_ALARM) == pytest.approx(budget / 24.0, rel=1e-6)
    t3 = lambda r: r.standard_t(3, 2) / math.sqrt(3.0)     # noqa: E731  heavy-tailed hourly input
    old = _boc_rate(t3, 0.8, clip=False)
    new = _boc_rate(t3, C.BOC_ALARM, clip=True)
    assert old > 20 * budget                                 # the former level and raw input
    assert new <= 2 * budget


def test_bocpd_evidence_on_a_step_still_reaches_fusion():
    """At the Ville level bocpd's own alarm is reserved for decisive
    evidence (a step is rarely that: P(r <= 3 h) falls again as the new run
    grows); its model p 1/BF still carries the step to B24 / B25."""
    from app.engines.behavior import changepoint as C
    rng = np.random.default_rng(1)
    mins = {0.0: [], 3.0: []}
    for step in mins:
        for _ in range(8):
            st = C.bocpd_new()
            mn = 1.0
            for h in range(24 * 10 + 24):
                x = rng.standard_normal(2) + (step if h >= 240 else 0.0)
                st = C.bocpd_step(st, np.clip(x, -seq.PSI_CLIP, seq.PSI_CLIP))
                if h >= 240:
                    mn = min(mn, C.bocpd_pm(C.bocpd_prob(st)))
            mins[step].append(mn)
    assert np.median(mins[3.0]) <= 1e-2 < np.median(mins[0.0])
