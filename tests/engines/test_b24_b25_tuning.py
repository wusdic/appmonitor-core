"""W7 tuning fixes to B24 / B25 (integration §8 "Tuning / design candidates").

B24: the small-sample prior is pm with its two atoms randomised over their
null mass in the entity's own pm history (m_calib.pm_prior), so a pm at the
float floor on every null tick (a sparse nightly host scored against its
peers) or an accumulator's p_eq = 1 atom no longer puts the issued p at the
floor or back in a point mass at 1.

B25: warm-up rows are not gated by their own evidence; the meta tail is
fitted robustly. Round 4 superseded the W7 row-evidence cap on released
rows: B24 and B25 admit by the PERIOD's trust only (lib/gating
.period_weight / release_weight) and handle contamination with the
trimmed tail fit (calib.robust_tail).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from helpers import put_model

from app.engines.behavior import fusion as F
from app.engines.behavior.lib import calib, m_calib
from app.engines.behavior.lib import m_governor as MG

from test_b24_calibration import E, S, Rig as CalRig, ks
from test_b25_fusion import Rig as FusRig


# ----------------------------------------------------------------- B24
def test_pm_at_the_floor_on_every_null_tick_no_longer_floors_p():
    """A night-only entity (every tick in one daypart, so no other-daypart
    prior) whose pm is 0 (1e-300 stored as float32) on every null tick: while
    its stratum ring holds < 64 entries the prior is the floor atom
    randomised over its mass in the entity's pm history (u x 1 here), i.e.
    uniform, not 1e-300."""
    rng = np.random.default_rng(3)
    rig = CalRig(daypart="wd_night")
    ps = []
    for k in range(60):
        ts = rig.step({E: {"marg_int": float(rng.exponential())}}, pm={E: {"marg_int": 0.0}})
        n = m_calib.ring_size(rig.model(), "marg_int", calib.stratum_key("wd_night", 900))
        n_pm = m_calib.ring_size(rig.model(), "marg_int", m_calib.pm_stratum("marg_int", 900))
        assert n < m_calib.SMALL_N
        if n_pm >= m_calib.PM_RING_MIN:
            ps.append(rig.p(E, "marg_int", ts))
    ps = np.asarray(ps)
    assert ps.size >= 30
    # uniform p: min of ~35 null p below 1e-4 has probability ~0.4 %
    assert ps.min() > 1e-4 and ks(ps) < 0.3


def test_accumulator_p_eq_atom_is_randomised_not_a_point_mass_at_one():
    """creep's stationary p_eq = 1 whenever its statistic is 0: the blend with
    pm = 1 gave p ~ 1 on every zero tick of a small ring (creep KS D = 1.0)."""
    rng = np.random.default_rng(8)
    rig = CalRig(daypart="wd_day")
    ps = []
    for k in range(60):
        zero = rng.random() < 0.85
        x = 0.0 if zero else float(rng.exponential(2.0))
        pm = 1.0 if zero else float(np.float32(math.exp(-x / 2.0) * 0.15))
        ts = rig.step({E: {"creep": x}}, pm={E: {"creep": pm}})
        n_pm = m_calib.ring_size(rig.model(), "creep", m_calib.pm_stratum("creep", 900))
        if zero and n_pm >= m_calib.PM_RING_MIN:
            ps.append(rig.p(E, "creep", ts))
    ps = np.asarray(ps)
    assert ps.size >= 25
    assert np.mean(ps > 0.99) < 0.2 and 0.25 < float(np.mean(ps)) < 0.75


def test_pm_prior_rules():
    u = 0.3
    assert m_calib.pm_prior(None, 0.2, u) == 0.2                  # the body: pm as is
    assert math.isnan(m_calib.pm_prior(None, float("nan"), u))
    # the atom pm = 1 with no history: mid-p over the Laplace pi = 1/2
    assert m_calib.pm_prior(None, 1.0, u) == 1.0 - 0.5 + u * 0.5
    # an unseen floor keeps its evidence
    assert m_calib.pm_prior(None, 0.0, u) == 0.0
    r = calib.Ring()
    for i in range(10):
        r.add(m_calib.pm_score(1.0), float(i))                   # 10 atoms, below min_n
    pi1 = 11.0 / 12.0
    assert m_calib.pm_prior(r, 1.0, u) == 1.0 - pi1 + u * pi1
    for i in range(10, 40):
        r.add(m_calib.pm_score(0.0), float(i))                   # 30 floors (stored 0)
    assert m_calib.pm_score(1e-300) == m_calib.pm_score(0.0) == m_calib.PM_ATOM_FLOOR
    # >= min_n: the floor atom holds 30 / 40 of the history -> u * 0.75
    assert m_calib.pm_prior(r, 1e-40, u) == pytest.approx(u * 30 / 40)
    assert m_calib.pm_prior(r, 1.0, u) == pytest.approx(1.0 - 10 / 40 + u * 10 / 40)
    # round 4: the body is calibrated on the pm history - here 30 of 40 past
    # pm were at the float floor, so a body pm of 0.004 is no evidence at all
    v = (30 * m_calib.PM_ATOM_FLOOR * math.log(10.0) + m_calib.PM_POW_KAPPA) / (
        40 + m_calib.PM_POW_KAPPA)
    assert m_calib.pm_prior(r, 0.004, u) == pytest.approx(0.004 ** (1.0 / v))


def test_pm_prior_calibrates_a_routinely_extreme_pm_and_keeps_a_valid_one():
    """Round 4: a detector whose pm sits at 1e-9 .. 1e-13 on null ticks (B06's
    young covariance, mini pack) no longer passes that pm on as the
    small-sample prior; a calibrated pm (uniform history) keeps its
    resolution. With >= 100 entries pm is calibrated on the ring itself."""
    rng = np.random.default_rng(2)
    bad, good = calib.Ring(), calib.Ring()
    for i in range(40):
        bad.add(m_calib.pm_score(10.0 ** -rng.uniform(9, 13)), float(i))
        good.add(m_calib.pm_score(float(rng.random())), float(i))
    assert m_calib.pm_prior(bad, 1e-13, 0.5) > 0.1               # routine: no evidence
    assert m_calib.pm_prior(good, 1e-6, 0.5) < 1e-5              # resolution kept
    big = calib.Ring()
    for i in range(300):
        big.add(m_calib.pm_score(10.0 ** -rng.uniform(9, 13)), float(i))
    big.gpd = calib.robust_tail(big)
    assert m_calib.pm_prior(big, 1e-11, 0.5) > 0.2                # mid-history
    assert m_calib.pm_prior(big, 1e-40, 0.5) < 1e-4               # far beyond it


def test_released_rows_enter_the_ring_and_the_trimmed_tail_ignores_them():
    """Round 4: a release is the governor's verdict that the period was
    normal (RETURNED, or an incident closed while normal), so its held rows
    are admitted whatever their own score (lib/gating.release_weight); an
    attack released by mistake is handled by the contamination-bounded tail
    fit (calib.robust_tail), not by thinning released rows by their own
    evidence. Six alarm-level rows in a 256 ring leave the tail where the
    clean ring's is, and a repeat of the same score still gets p < 1e-6."""
    rng = np.random.default_rng(2)
    rig = CalRig(daypart="wd_day")
    for _ in range(254):
        rig.step({E: {"marg_int": float(rng.exponential())}})
    held = []
    for _ in range(6):
        ts = rig.step({E: {"marg_int": 25.0}}, quarantine=1.0)
        held.append(ts)
    # as B28 does: the release and quarantine 0 are written on the same tick,
    # the learners apply them on the next one
    rig.step({E: {"marg_int": float(rng.exponential())}})
    put_model(rig.store, S, E, "model.control", {"version": 0, "release": [held[0], rig.t - 900]})
    for _ in range(4):                                        # D: the rows not yet held
        rig.step({E: {"marg_int": float(rng.exponential())}})
    r = m_calib.ring(rig.model(), "marg_int", calib.stratum_key("wd_day", 900))
    assert set(held) <= set(r.ts.tolist())                    # admitted
    tail = calib.robust_tail(r)
    keep = r.scores < 25.0
    clean = calib.Ring(scores=r.scores[keep], ts=r.ts[keep])
    ref = calib.robust_tail(clean)
    raw = calib.fit_tail(r)                                   # untrimmed, as before round 4
    # the six foreign rows are trimmed: the tail is the clean ring's (up to
    # the predictive floor 1/n_u of its 6 fewer exceedances)
    assert 0.5 < tail.sf(8.0) / ref.sf(8.0) < 2.0 and tail.xi < 0.1
    assert raw.sf(8.0) > 10 * tail.sf(8.0)
    ts = rig.step({E: {"marg_int": 25.0}})
    assert rig.p(E, "marg_int", ts) < 1e-6


# ----------------------------------------------------------------- B25
NULL = {"marg_int": 0.4, "novelty": 0.6}


def _meta(rig):
    return rig.store.get_model(S, E, m_calib.MODEL)["meta"]


def test_warmup_rows_are_not_gated_by_their_own_evidence():
    """Warm-up rows (trust 1) enter the meta rings whatever their p_all: the
    governor writes no row-evidence weight in training, since gating the
    null on its own evidence truncates its tail (integration §8.2)."""
    rig = FusRig()
    for k in range(12):
        rig.step({E: {"marg_int": 1e-30 if k % 2 else 0.4, "novelty": 0.6}}, training=True)
    st = _meta(rig)["state"]
    assert st["n_admit"] == 8                     # every committed row
    rs = F.meta_rings(_meta(rig))
    assert max(float(r.scores.max()) for r in rs.values()) > 20.0


def test_release_admits_meta_rows_by_the_period_verdict():
    """Round 4: released rows enter the meta rings (the period was judged
    normal); their own p_all does not gate them (lib/gating.release_weight)."""
    rig = FusRig()
    held = []
    for k in range(8):
        held.append(rig.step({E: {"marg_int": 1e-28, "novelty": 0.6}}, quarantine=1.0))
    assert _meta(rig)["state"]["n_admit"] == 0
    rig.step({E: dict(NULL)})             # B28: release + quarantine 0 on the same tick
    put_model(rig.store, S, E, "model.control", {"version": 0, "release": [held[0], rig.t - 900]})
    rig.step({E: dict(NULL)})
    assert _meta(rig)["state"]["n_admit"] == 6     # the 5 held rows + 1 committed now
    for _ in range(3):                # the last rows of the released period: admitted too
        rig.step({E: dict(NULL)})
    assert _meta(rig)["state"]["n_admit"] == 9
    got = set()
    for r in F.meta_rings(_meta(rig)).values():
        got |= set(r.ts.tolist())
    assert set(held) <= got
    rs = F.meta_rings(_meta(rig))
    assert max(float(r.scores.max()) for r in rs.values()) > 20.0


def test_missing_evidence_weight_leaves_the_gate_alone():
    assert MG.evidence_weight(FusRig().store, S, E, 0.0) == 1.0


def test_winsorised_meta_tail_ignores_a_few_contaminating_extremes():
    """-log10 of valid p has an exponential tail: 250 null scores plus 6
    warm-up extremes (21 .. 37 decades) fitted xi = 0.5 without the robust
    step (q of a score 6 decades out ~ 3e-2: saturated); with it the tail is
    near the clean ring's again."""
    rng = np.random.default_rng(12)
    clean = calib.Ring()
    for i, x in enumerate(-np.log10(rng.random(250))):
        clean.add(float(x), float(i))
    dirty = calib.Ring(scores=clean.scores.copy(), ts=clean.ts.copy())
    for j, x in enumerate((21.0, 24.0, 28.0, 31.0, 34.0, 37.0)):
        dirty.add(x, 1000.0 + j)
    raw = calib.fit_tail(dirty, xi_min=0.0)
    rob = F.meta_tail(dirty)
    ref = F.meta_tail(clean)
    assert raw.xi > 0.3 and rob.xi < 0.15
    assert raw.sf(6.0) > 1e-2 and rob.sf(6.0) < 1e-3 and ref.sf(6.0) < 1e-3
    # a clean exponential tail is left alone in ~1 % of fits (n_u = 26)
    rng = np.random.default_rng(0)
    touched = sum(not np.array_equal(calib.winsorise_exceedances(y, F.META_WINSOR_ALPHA), y)
                  for y in (np.sort(rng.exponential(0.43, 26)) for _ in range(2000)))
    assert touched / 2000 < 0.02


# ------------------------------------------------ round 4: live power correction
def test_pcal_v_needs_a_significant_excess():
    assert m_calib.pcal_v(None) == 1.0
    assert m_calib.pcal_v([10.0, 20.0, 0.0]) == 1.0                  # n < 30
    assert m_calib.pcal_v([25.0, 500.0, 0.0]) == 1.0                 # exactly nominal
    assert m_calib.pcal_v([40.0, 500.0, 0.0]) == 1.0                 # 8 %: not significant
    v = m_calib.pcal_v([135.0, 500.0, 0.0])                          # 27 %: significant
    assert v == pytest.approx(math.log(0.05) / math.log(0.27))       # the point estimate
    assert 2.0 < v < 3.0
    assert m_calib.pcal_apply(1e-6, v) == pytest.approx(1e-6 ** (1.0 / v))
    assert m_calib.pcal_apply(0.5, 1.0) == 0.5
    st = m_calib.pcal_observe(None, True, 0.0)
    st = m_calib.pcal_observe(st, False, m_calib.PCAL_HL_S)          # one half-life later
    assert st == pytest.approx([0.5, 1.5, m_calib.PCAL_HL_S])


def test_live_power_correction_follows_a_warmup_to_live_shift():
    """Warm-up null Exp(1), live null Exp(2.5) (a detector whose live scores
    are heavier than its warm-up ring: the go-live step of spe / t2 / identity
    in pack A): the issued live p are anti-conservative until the system's
    live share of p <= 0.05 is significant, then p^(1/v) brings the realised
    rate at p <= 0.01 back to <= 2x; a live stream that matches the warm-up
    keeps v = 1."""
    from app.engines.behavior.lib.classkeys import SYSTEM_KEY
    ents = [f"10.0.0.{k}" for k in range(1, 11)]
    for scale, shifted in ((2.5, True), (1.0, False)):
        rng = np.random.default_rng(7)
        # ten entities, hourly ticks: each ring (256 rows) still holds the
        # warm-up rows for the whole live stretch, as an H stratum does for
        # weeks after go-live; the correction is pooled over the system
        rig = CalRig(entities=ents, daypart="wd_day", dt=3600.0)
        for _ in range(300):
            rig.step({e: {"marg_int": float(rng.exponential())} for e in ents}, training=True)
        ps = []
        for _ in range(60):
            ts = rig.step({e: {"marg_int": float(rng.exponential(scale))} for e in ents})
            ps.append([rig.p(e, "marg_int", ts) for e in ents])
        pc = rig.store.get_model(S, SYSTEM_KEY, m_calib.MODEL)[m_calib.PCAL]
        v = m_calib.pcal_v(pc[m_calib.pcal_key("marg_int", 3600)])
        ps = np.asarray(ps)
        if shifted:
            assert v > 1.5
            assert np.mean(ps[:3] <= 0.01) > 0.05                    # before: anti-conservative
            assert np.mean(ps[30:] <= 0.01) <= 0.02                   # after: in band
        else:
            assert v == 1.0
            assert np.mean(ps <= 0.01) <= 0.02
