"""Store fast paths (perf, docs/lib3/integration.md §9) against the plain
search they short-cut: vec_at on the newest row / beyond it, the unwrapped
ring search, drop_before when nothing expired, names_signature, and the
8-day retention of the lib-3 series that had no rule."""
import numpy as np

from app.core import store as ST
from app.core.store import MetricStore, _VecRing
from app.models.schema import DerivedMetric, MetricKind

S, E = "sys", "10.0.0.1"


def _ref_search(ring: _VecRing, t: float, side: str) -> int:
    ts, _ = ring.take(0, ring.n)
    return int(np.searchsorted(ts, t, side=side))


def test_ring_search_and_vec_at_match_a_plain_search_on_wrapped_rings():
    rng = np.random.default_rng(0)
    for cap in (1, 2, 5, 16):
        st = MetricStore()
        st.set_retention("x.", max_points=cap)
        ts_all = np.cumsum(rng.integers(1, 4, 60)).astype(float)
        for i, t in enumerate(ts_all):
            st.add_vec(S, E, "x.v", t, [float(i), -float(i)])
            ring = st._vec[ST._k(S, E, "x.v")]
            kept, rows = ring.take(0, ring.n)
            for q in np.concatenate((kept, kept + 0.5, kept - 0.5, [-1.0, 1e12, np.nan])):
                for side in ("left", "right"):
                    assert ring.search(q, side) == _ref_search(ring, q, side)
                got = st.vec_at(S, E, "x.v", float(q))
                hit = np.flatnonzero(kept == q)
                if hit.size:
                    np.testing.assert_array_equal(got, rows[hit[-1]])
                else:
                    assert got is None


def test_drop_before_fast_path_keeps_the_same_rows():
    st = MetricStore()
    st.set_retention("y.", max_age_s=10.0)
    written = []
    for t in np.arange(0.0, 100.0, 3.0):
        st.add_vec(S, E, "y.v", float(t), [t])
        written.append(float(t))
        ts, _ = st.vec_tail(S, E, "y.v", 1000)
        assert ts.tolist() == [x for x in written if x >= t - 10.0]


def test_names_signature_moves_when_a_name_is_added():
    st = MetricStore()
    s0 = st.names_signature(S, E)
    st.add_vec(S, E, "behavior.z", 1.0, [1.0])
    s1 = st.names_signature(S, E)
    st.add_vec(S, E, "behavior.z", 2.0, [1.0])
    assert st.names_signature(S, E) == s1 != s0
    st.add_derived(DerivedMetric(name="behavior.rhythm", value={"a": 1}, ts=2.0, system=S,
                                 entity=E, window_s=0, kind=MetricKind.CATEGORICAL))
    assert st.names_signature(S, E) != s1


def test_unbounded_lib3_series_keep_eight_days():
    st = MetricStore()
    for name in ("behavior.acc_alarm", "behavior.rhythm", "behavior.timing", "behavior.id",
                 "behavior.class", "behavior.common.flag", "behavior.cp.prob",
                 "behavior.seq.class_llr"):
        assert st._rule("derived", name)[1] == 8 * 86400.0, name
    assert st._rule("derived", "behavior.common.q.volume")[1] == 6 * 3600.0
    assert st._rule("vec", "behavior.class.agg")[1] == 9 * 86400.0
