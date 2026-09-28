"""Round 4 (gate 14): count-and-age retention and lossless compaction of
per-tick dict series (core/store.py DICT_POINT_CAPS, _compact, memory_report).

At 60 s an 8-d age rule kept 11520 dicts per series and entity (~163 MB per
entity extrapolated to 8 d); every reader of these series looks back a
bounded number of points (the audit in store.py). Each test fails on the
pre-round-4 store.
"""
from __future__ import annotations

import sys

import numpy as np

from helpers import T0

from app.core import store as S
from app.core.store import MetricStore
from app.engines.behavior.lib import emit
from app.models.schema import DerivedMetric, MetricKind

E = ("erp-prod", "10.20.1.11")


def _add(st, name, ts, value, dt=60, inputs=None):
    st.add_derived(DerivedMetric(name=name, value=value, ts=ts, system=E[0], entity=E[1],
                                 window_s=dt, kind=MetricKind.CATEGORICAL, inputs=inputs))


def _series(st, name):
    return st.derived_series(E[0], E[1], name)


def test_per_tick_dict_series_are_point_capped_at_60s():
    st = MetricStore()
    n = S.DICT_POINT_CAP + 500
    for i in range(n):
        _add(st, "behavior.rhythm", T0 + 60.0 * i, {"S_hi": float(i), "slot": i})
    rows = _series(st, "behavior.rhythm")
    assert len(rows) == S.DICT_POINT_CAP                  # age alone (8 d) keeps all
    assert rows[-1].value["slot"] == n - 1 and rows[0].value["slot"] == n - S.DICT_POINT_CAP
    for i in range(n):
        _add(st, "feature.tctx", T0 + 60.0 * i, {"hour_local": float(i), "slot": i})
    assert len(_series(st, "feature.tctx")) == S.DICT_POINT_CAPS["feature.tctx"] >= 4 * 96
    # behavior.degraded had no rule at all (20000 points)
    for i in range(n):
        emit.write_degraded(st, E[0], E[1], T0 + 60.0 * i, {"spe": f"stale:x{i}"}, 60)
    assert len(_series(st, emit.DEGRADED)) == S.DICT_POINT_CAPS["behavior.degraded"]


def test_cap_is_neutral_at_900s_and_p_family_keeps_a_day_at_60s():
    """>= 8 d at 900 s: the age rule binds first, as before round 4."""
    assert S.DICT_POINT_CAP * 900.0 >= 8 * 86400.0
    st = MetricStore()
    for i in range(12 * 96):                                # 12 d at 900 s
        _add(st, "behavior.acc_alarm", T0 + 900.0 * i, {"cusum": i % 2}, dt=900)
    assert len(_series(st, "behavior.acc_alarm")) == 8 * 96 + 1
    st2 = MetricStore()
    for i in range(3000):                                   # 50 h at 60 s
        _add(st2, "behavior.p_family", T0 + 60.0 * i, {"shape": 1.0 / (i + 2)})
    assert len(_series(st2, "behavior.p_family")) == 1441   # its 1-d age rule only
    # vector rings are untouched by the dict caps
    for i in range(S.DICT_POINT_CAP + 10):
        st2.add_vec(E[0], E[1], "behavior.regime", T0 + 60.0 * i, np.zeros(1, np.float32))
    assert st2.vec_tail(E[0], E[1], "behavior.regime", 10 ** 6)[0].size == S.DICT_POINT_CAP + 10


def test_equal_consecutive_rows_share_one_object_but_never_the_newest():
    st = MetricStore()
    for i in range(5):
        _add(st, "behavior.regime", T0 + 60.0 * i, {"state": "normal", "version": 1},
             inputs=["behavior.p"])
    rows = _series(st, "behavior.regime")
    assert rows[0].value is rows[1].value is rows[2].value is rows[3].value
    assert rows[4].value is not rows[3].value              # the newest is its own
    assert rows[0].inputs is rows[3].inputs
    # upsert into the newest row does not leak into the shared older rows
    st.upsert_dict(E[0], E[1], "behavior.regime", T0 + 240.0, {"state": "suspect"})
    rows = _series(st, "behavior.regime")
    assert [r.value["state"] for r in rows] == ["normal"] * 4 + ["suspect"]
    # a new row after it: the suspect row is compared, not shared with normal
    _add(st, "behavior.regime", T0 + 300.0, {"state": "normal", "version": 1})
    rows = _series(st, "behavior.regime")
    assert rows[4].value["state"] == "suspect" and rows[4].value is not rows[3].value


def test_compaction_is_exact():
    """Only structurally identical values share: types, key order, NaN."""
    same, diff = S._same, S._same
    assert same({"a": 1, "b": [1, 2.0]}, {"a": 1, "b": [1, 2.0]})
    assert not diff({"a": 1}, {"a": 1.0})                  # int vs float
    assert not diff({"a": True}, {"a": 1})
    assert not diff({"a": 1, "b": 2}, {"b": 2, "a": 1})    # key order
    assert not diff({"a": float("nan")}, {"a": float("nan")})
    assert not diff({"a": np.array([1.0, 2.0])}, {"a": np.array([1.0, 2.0])})
    st = MetricStore()
    _add(st, "behavior.axes", T0, {"x": 1})
    _add(st, "behavior.axes", T0 + 60, {"x": 1.0})
    _add(st, "behavior.axes", T0 + 120, {"x": 1.0})
    rows = _series(st, "behavior.axes")
    assert type(rows[1].value["x"]) is float and rows[1].value is not rows[0].value


def test_memory_report_counts_a_shared_value_once():
    st = MetricStore()
    health = {f"d{i}": {"ks_d": 0.01, "rate_ratio": 1.0, "weight_mult": 1.0} for i in range(30)}
    for i in range(1000):                                   # B24: one object reused hourly
        st.add_derived(DerivedMetric(name="behavior.calib_health", value=health, ts=T0 + 60 * i,
                                     system="erp-prod", entity="__system__", window_s=60))
    rep = st.memory_report()
    one = sys.getsizeof(health) + sum(sys.getsizeof(k) + sys.getsizeof(v)
                                      for k, v in health.items())
    assert rep["derived_bytes"] < 1000 * 200 + 2 * one     # rows + one value, not 1000 values
    assert rep["derived_bytes"] > 1000 * 50
