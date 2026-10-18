"""lib/pbounds round 8: P06's band and tail coverage.

(1) Atoms. A transport size that takes a handful of exact values (pack O's
portal GET /news net.bytes_up: 460, 461, 470, 505, 516 B) had its band end
interpolated BETWEEN two atoms by the t-digest: [461, 516] stated a 0.97
coverage and held 0.79-0.86 on every day of pack O seed 3 (prequential probe):
the 460-byte requests were all outside. The band end is now the atom.
Fails on the round-7 code."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.engines.behavior.lib import pbounds as PB
from app.engines.behavior.lib import pnode as PN

T0 = 1_780_000_000.0
DAY = 86400.0
ATOMS = np.array([460.0, 461.0, 470.0, 471.0, 504.0, 505.0, 515.0, 516.0])
P_ATOMS = np.array([0.07, 0.55, 0.08, 0.1, 0.06, 0.08, 0.03, 0.03])


def _num(values, days=10, log=True):
    s = PN.NumSummary(log=log)
    per = int(math.ceil(len(values) / days))
    for i, v in enumerate(values):
        d = i // per
        s.update(float(v), T0 + d * DAY + (i % per) * 60.0, 1.0, 1.0, day=700000 + d)
    return s, T0 + days * DAY, 700000 + days - 1


# the portal's measured mix (pack O seed 3, days 1-3): the top atom holds 29 %
P_NEWS = np.array([0.015, 0.106, 0.021, 0.188, 0.036, 0.318, 0.024, 0.292])


@pytest.mark.parametrize("probs", [P_ATOMS, P_NEWS])
def test_a_band_end_between_two_atoms_takes_the_atom(probs):
    P_ATOMS = probs / probs.sum()
    r = np.random.default_rng(0)
    vals = r.choice(ATOMS, size=3000, p=P_ATOMS)
    s, t, day = _num(vals)
    rec = PB.fit_numeric(s, t, day, n_c=3000, n_eff=3000, unit="B")
    lo, hi = rec["band90"]
    fresh = r.choice(ATOMS, size=20000, p=P_ATOMS)
    held = float(np.mean((fresh >= lo - 1e-6) & (fresh <= hi + 1e-6)))
    assert min(abs(lo - a) for a in ATOMS) < 1e-6 and min(abs(hi - a) for a in ATOMS) < 1e-6
    assert held >= rec["coverage"] - 0.02, (lo, hi, held, rec["coverage"])
    assert held >= 0.9


def test_continuous_data_is_unchanged_by_the_atom_rule():
    r = np.random.default_rng(1)
    vals = r.uniform(10000.0, 80000.0, 3000)
    s, t, day = _num(vals)
    qv = PB.td_quantiles(s.td, [0.05, 0.95])
    obs = PB._observed(s, False, float(np.log(vals.min())))
    assert PB.snap_atom(qv[0], obs, down=True) == qv[0]
    obs = PB._observed(s, True, float(np.log(vals.max())))
    assert PB.snap_atom(qv[1], obs, down=False) == qv[1]
    assert PB.snap_atom(5.0, {4.0: 3, 4.5: 1}, down=True) == 5.0      # nearest below is a singleton
    assert PB.snap_atom(5.0, {4.0: 1, 4.5: 3}, down=True) == 4.5
    assert PB.snap_atom(4.5, {4.5: 3}, down=True) == 4.5


def test_band_and_range_ends_at_integer_atoms_are_exact():
    """exp(log(461)) = 461.00000000000006: the stated band [461.00000000000006,
    516] excluded every 461-byte row (10.6 % of the portal's news requests) and
    the range [460.0000000000001, 516] every 460-byte one."""
    assert math.exp(math.log(461.0)) != 461.0          # the float fact the fix is about
    assert PB._inv(math.log(461.0), True) == 461.0
    assert PB._inv(math.log(460.0), True) == 460.0
    assert PB._inv(math.log(461.3), True) == math.exp(math.log(461.3))
    r = np.random.default_rng(3)
    vals = r.choice(ATOMS, size=3000, p=P_NEWS / P_NEWS.sum())
    s, t, day = _num(vals)
    rec = PB.fit_numeric(s, t, day, n_c=3000, n_eff=3000, unit="B")
    assert rec["range"] == [460.0, 516.0]
    lo, hi = rec["band90"]
    held = float(np.mean((vals >= lo) & (vals <= hi)))      # strict comparisons, as a reader's
    assert held >= rec["coverage"] - 0.02 and lo in ATOMS and hi in ATOMS
