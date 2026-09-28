"""B01 grain rows (spec v2.1, docs/lib3/cadence.md §2-§5)."""
from __future__ import annotations

import numpy as np

from helpers import make_store, run_engine

from app.engines.behavior.feature_vector import FeatureVectorEngine
from app.models.schema import AcquisitionMethod, RawMetric

S, E = "sys", "10.0.0.1"
H0 = 1_741_536_000.0
CAN = {"grain_mode": "canonical", "tz": "Asia/Shanghai"}


def _tick(st, now, flows):
    st.add_raw(RawMetric(name="l4.flows", value=float(flows), ts=now, system=S, entity=E,
                         method=AcquisitionMethod.PASSIVE_FLOW))


def _drive(dt, n, config, flows=3.0):
    st, eng = make_store(), FeatureVectorEngine()
    for k in range(1, n + 1):
        now = H0 + k * dt
        _tick(st, now, flows)
        run_engine(eng, st, now, dt=dt, config=config)
    return st


def test_decision_rows_only_on_decision_ticks_and_live_every_tick():
    st = _drive(900.0, 12, CAN)
    ts, _ = st.vec_since(S, E, "feature.nat.h", 0.0)
    assert ts.tolist() == [H0 + 3600.0, H0 + 7200.0, H0 + 10800.0]
    tq, _ = st.vec_since(S, E, "feature.nat.q", 0.0)
    assert len(tq) == 12
    tl, L = st.vec_since(S, E, "feature.live.h", 0.0)
    assert len(tl) == 12
    tm, Mm = st.vec_since(S, E, "feature.meta.h", 0.0)
    assert Mm[0].tolist() == [1.0, 3600.0, 4.0, 4.0, 3600.0]
    _, N = st.vec_since(S, E, "feature.nat.h", 0.0)
    assert float(N[0][2]) == 12.0                       # 4 ticks x 3 flows
    tp, P = st.vec_since(S, E, "feature.part", 0.0)
    assert len(tp) >= 8 and P.shape[1] == 47           # retained 2 h
    ts_h, _ = st.vec_since(S, E, "feature.sketch.h", 0.0)
    assert len(ts_h) == 3
    assert st.latest_derived(S, E, "feature.expo.h").value["flows"] == 12.0


def test_no_q_grain_at_3600_and_every_tick_is_an_h_decision():
    st = _drive(3600.0, 5, CAN)
    assert len(st.vec_since(S, E, "feature.nat.h", 0.0)[0]) == 5
    assert len(st.vec_since(S, E, "feature.nat.q", 0.0)[0]) == 0
    assert len(st.vec_since(S, E, "feature.live.q", 0.0)[0]) == 0


def test_tick_mode_writes_nothing_new():
    st = _drive(900.0, 8, {"grain_mode": "tick"})
    for name in ("feature.part", "feature.live.h", "feature.nat.h", "feature.meta.h",
                 "feature.nat.q", "feature.sketch.h"):
        assert len(st.vec_since(S, E, name, 0.0)[0]) == 0, name
    assert len(st.vec_since(S, E, "feature.nat", 0.0)[0]) == 8


def test_partial_window_after_a_gap_counts_true_zeros():
    """Silent ticks are zeros; coverage is the pipeline's wall clock."""
    st, eng = make_store(), FeatureVectorEngine()
    for k in range(1, 9):
        now = H0 + k * 900.0
        if k in (1, 2):
            _tick(st, now, 5.0)
        run_engine(eng, st, now, dt=900.0, config=CAN)
    _, N = st.vec_since(S, E, "feature.nat.h", 0.0)
    _, Mm = st.vec_since(S, E, "feature.meta.h", 0.0)
    assert float(N[0][2]) == 10.0 and float(N[1][2]) == 0.0
    assert Mm[1].tolist()[:2] == [0.0, 3600.0]          # inactive hour, fully covered
    assert np.isnan(N[1][14])                           # a ratio of nothing is undefined
