"""The stale checker's cached name scan (perf) equals a fresh scan as the
entity gains new series."""
from app.core.store import MetricStore
from app.eval.runner import _StaleChecker
from app.models.schema import DerivedMetric, MetricKind

S, E = "sys", "10.0.0.1"


def _der(st, name, ts):
    st.add_derived(DerivedMetric(name=name, value={"x": 1}, ts=ts, system=S, entity=E,
                                 window_s=0, kind=MetricKind.CATEGORICAL))


def test_cached_names_follow_new_series():
    st = MetricStore()
    st.register_vector_names("feature.vec", ["bytes_up", "flows"], "feature.")
    chk = _StaleChecker()
    st.add_vec(S, E, "feature.vec", 1.0, [1.0, 2.0])
    _der(st, "behavior.rhythm", 1.0)
    for step, name in enumerate(("behavior.timing", "behavior.alarm", "other.x",
                                 "behavior.id", "feature.expo")):
        assert chk._names(st, S, E) == chk._scan_names(st, S, E)
        assert chk._names(st, S, E) is chk._names(st, S, E)       # cached while unchanged
        _der(st, name, 2.0 + step)
    st.add_vec(S, E, "behavior.z", 9.0, [0.0])
    assert chk._names(st, S, E) == chk._scan_names(st, S, E)
    assert ("vec", "behavior.z") in chk._names(st, S, E)
