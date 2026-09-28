"""W7 tuning fixes to B24 / B25 (integration §8 "Tuning / design candidates").

B24: the small-sample prior is pm with its two atoms randomised over their
null mass in the entity's own pm history (m_calib.pm_prior), so a pm at the
float floor on every null tick (a sparse nightly host scored against its
peers) or an accumulator's p_eq = 1 atom no longer puts the issued p at the
floor or back in a point mass at 1.

B25: meta rings admit a row with min(gate weight, the governor's
row-evidence weight) (m_governor.evidence_weight, live rows only), so a
release (trust_prov, no accumulator factor) cannot admit a row whose
accumulators were at alarm level; warm-up rows are not gated by their own
evidence; the meta tail is fitted to winsorised exceedances.
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
    assert m_calib.pm_prior(r, 0.004, u) == 0.004                 # body untouched


def test_released_rows_do_not_enter_detector_rings_at_alarm_level():
    """A release commits held rows with trust_prov; the governor's evidence
    weight (accumulator at alarm level -> 0) caps it on live ticks."""
    rng = np.random.default_rng(1)
    rig = CalRig(daypart="wd_day")
    for _ in range(20):
        rig.step({E: {"marg_int": float(rng.exponential())}})
    held = []
    for _ in range(6):
        ts = rig.step({E: {"marg_int": 25.0}}, quarantine=1.0)
        rig.store.add_vec(S, E, MG.TRUST_EVIDENCE, ts, [0.0], window_s=900)
        held.append(ts)
    put_model(rig.store, S, E, "model.control", {"version": 0, "release": [held[0], held[-1]]})
    rig.step({E: {"marg_int": 1.0}})
    r = m_calib.ring(rig.model(), "marg_int", calib.stratum_key("wd_day", 900))
    assert r.scores.max() < 25.0
    assert not set(r.ts.tolist()) & set(held)


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


def test_release_caps_meta_admission_by_the_evidence_weight():
    rig = FusRig()
    held = []
    for k in range(8):
        t = rig.step({E: {"marg_int": 1e-28, "novelty": 0.6}}, quarantine=1.0)
        rig.store.add_vec(S, E, MG.TRUST_EVIDENCE, t, [0.0], window_s=900)
        held.append(t)
    assert _meta(rig)["state"]["n_admit"] == 0
    put_model(rig.store, S, E, "model.control", {"version": 0, "release": [held[0], held[-1]]})
    rig.step({E: dict(NULL)})
    assert _meta(rig)["state"]["n_admit"] == 0 and not F.meta_rings(_meta(rig))


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
