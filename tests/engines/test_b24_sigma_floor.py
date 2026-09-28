"""B24 wiring of the p-score tail floor (round 4, evaluator; lib/calib
P_SCORE_SIGMA, m_calib.tail_sigma_min).

A ring of compressed warm-up p-scores (score = -log10 pm with pm = U^0.4, a
young model's conservative predictive) and then a live score far beyond the
ring: with the row's pm equal to 10^-score the issued p follows pm's own
decay beyond the tail threshold (>= rate 10^-(x - u)); a score without a pm
(bocpd, seq) keeps the plain GPD extrapolation. B29's replay (p_replay with
the row's pm) reproduces the issued p.
"""
from __future__ import annotations

import numpy as np

from test_b24_calibration import E, Rig, S

from app.engines.behavior.lib import calib, gating, m_calib


def _warm(rig: Rig, with_pm: bool) -> None:
    rng = np.random.default_rng(21)
    xs = -0.4 * np.log10(rng.random(calib.RING_M + gating.commit_delay_ticks(900.0) + 40))
    for x in xs:
        x = float(x)
        rig.step({E: {"marg_int": x}}, pm={E: {"marg_int": 10.0 ** -x}} if with_pm else None)


def test_live_p_score_beyond_a_compressed_ring_follows_its_pm():
    x = 6.0
    rig = Rig(daypart="wd_day")
    _warm(rig, True)
    ts = rig.step({E: {"marg_int": x}}, pm={E: {"marg_int": 10.0 ** -x}})
    p = rig.p(E, "marg_int", ts)
    ring = m_calib.ring(rig.model(), "marg_int", calib.stratum_key("wd_day", 900))
    t = ring.gpd
    assert t is not None and x > ring.scores[-1]
    floor = t.rate * 10.0 ** -(x - t.u)
    assert p >= 0.999 * min(floor, 1.0 / (len(ring) + 1)), (p, floor)
    assert p > 1e3 * t.sf(x)            # the plain fit extrapolates far steeper
    # B29 replay with the row's pm reproduces the issued p
    u = m_calib.uniform(S, E, "marg_int", ts)
    assert m_calib.p_replay(rig.model(), "marg_int", "wd_day", 900, x, u,
                            pm=10.0 ** -x) == p


def test_a_score_without_pm_keeps_the_plain_extrapolation():
    x = 6.0
    rig = Rig(daypart="wd_day")
    _warm(rig, False)
    ts = rig.step({E: {"marg_int": x}})
    p = rig.p(E, "marg_int", ts)
    ring = m_calib.ring(rig.model(), "marg_int", calib.stratum_key("wd_day", 900))
    assert ring.gpd is not None
    assert p == m_calib.issued(ring.gpd.sf(x))
