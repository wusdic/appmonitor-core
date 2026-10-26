"""pwindows round 9: window_hold counts the date cross-validation once (OI8-j).

_apply_cv states the posterior mean of the cross-validated share,
c = (h + FWD_PRIOR c_in) / (m + FWD_PRIOR), when that is below the in-sample
coverage c_in. window_hold centred its prior on that already-lowered c and
added the cross-validated share as evidence again, so a part whose few dates
cross-validated badly was stated 'P(holds) = 0.0' at a coverage that its own
posterior puts at the mean. Pack O seed 0, day 7: the 销售部 mail part
p:mail:0:4@1.0|grp:G10 was stated 0.0 and held. Fails on round 8."""
from __future__ import annotations

import numpy as np
from scipy.stats import beta

from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.lib import pwindows as PW


def _lowered(c_in, h, m, w_points):
    c = min(c_in, (h + PW.FWD_PRIOR * c_in) / (m + PW.FWD_PRIOR))
    return {"res": "minute", "windows": [[570, 590], [800, 830]], "coverage": c, "coverage_in": c_in,
            "cv": [h, m], "w_points": w_points, "all_day": False}


def test_a_cross_validated_coverage_is_not_counted_twice():
    rec = _lowered(0.9, 0.0, 6.0, 6.0)
    c = rec["coverage"]
    h = PW.window_hold(rec)
    # the posterior of the share inside the stated windows, the prior centred on
    # the in-sample coverage the cross-validation was weighed against
    post = beta(0.0 * 6.0 + PW.FWD_PRIOR * 0.9, 6.0 + PW.FWD_PRIOR * 0.1)
    assert abs(post.mean() - c) < 1e-9
    assert abs(h - post.sf(c - PN.hold_eps(c))) < 1e-9
    assert 0.4 < h < 0.9, h
    # more cross-validated dates against the windows: more certain
    assert PW.window_hold(_lowered(0.9, 3.0, 30.0, 30.0)) > 0.4


def test_unchanged_when_the_cross_validation_did_not_lower_the_coverage():
    rec = _lowered(0.8, 45.0, 50.0, 50.0)
    assert rec["coverage"] == 0.8
    s_ = 45.0 / 50.0
    a, b = s_ * 50.0 + PW.FWD_PRIOR * 0.8, (1 - s_) * 50.0 + PW.FWD_PRIOR * 0.2
    want = float(beta(a, b).sf(0.8 - PN.hold_eps(0.8)))
    assert np.isclose(PW.window_hold(rec), want)
    no_in = {k: v for k, v in rec.items() if k != "coverage_in"}
    assert np.isclose(PW.window_hold(no_in), want)
