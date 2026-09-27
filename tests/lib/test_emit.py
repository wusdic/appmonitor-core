"""Same-tick multi-writer convention for detector outputs."""
import math

import numpy as np
import pytest

from helpers import T0, make_store

from app.engines.behavior.lib import emit
from app.engines.behavior.lib.detectors import N_DETECTORS

S, E = "sys", "10.0.0.1"


def test_two_engines_merge_into_one_row():
    st = make_store()
    emit.write_scores(st, S, E, T0, {"marg_int": 3.0, "marg_shape": 1.0},
                      axes={"marg_int": ["volume"]})
    emit.write_scores(st, S, E, T0, {"t2": 5.0}, axes={"t2": ["volume", "app"]},
                      acc_alarm={"cusum": 0})
    ts, M = st.vec_tail(S, E, emit.SCORE, 10)
    assert len(ts) == 1 and M.shape == (1, N_DETECTORS)
    assert emit.read_row(st, S, E, emit.SCORE, T0) == {"marg_int": 3.0, "marg_shape": 1.0, "t2": 5.0}
    assert emit.read_dict(st, S, E, emit.AXES, T0) == {"marg_int": ["volume"], "t2": ["volume", "app"]}
    assert emit.read_dict(st, S, E, emit.ACC_ALARM, T0) == {"cusum": 0}


def test_next_tick_starts_fresh_nan_row():
    st = make_store()
    emit.write_scores(st, S, E, T0, {"marg_int": 3.0})
    emit.write_scores(st, S, E, T0 + 900, {"t2": 1.0})
    assert emit.read_row(st, S, E, emit.SCORE, T0 + 900) == {"t2": 1.0}
    assert math.isnan(emit.read_array(st, S, E, emit.SCORE, T0 + 900)[0])
    assert emit.read_dict(st, S, E, emit.AXES, T0 + 900) == {}


def test_virtual_views_and_none_as_nan():
    st = make_store()
    emit.write_scores(st, S, E, T0, {"novelty": None, "beacon": 2.5})
    assert st.latest_derived(S, E, "behavior.score.beacon").value == pytest.approx(2.5)
    assert emit.read_row(st, S, E, emit.SCORE, T0) == {"beacon": 2.5}
    emit.write_pvalues(st, S, E, T0, {"beacon": 1e-4})
    assert st.latest_derived(S, E, "behavior.p.beacon").value == pytest.approx(1e-4, rel=1e-3)


def test_unknown_detector_rejected():
    with pytest.raises(KeyError):
        emit.write_scores(make_store(), S, E, T0, {"nope": 1.0})
