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


def _raw(st, ts):
    from app.models.schema import RawMetric
    st.add_raw(RawMetric(name="l4.flows", value=1.0, ts=ts, system=S, entity=E))


def test_a_per_window_series_is_owed_only_by_a_later_window():
    """B21's clock behavior.xsys is written at the first ACTIVE tick of each
    epoch-aligned H window. An entity that appears one tick before a window
    edge writes at t, t+1, then hourly: the gap-median period (900 s) flagged
    it at t+4 (pack A seed 0); sporadic entities were flagged the same way
    (pack E seed 0). Owed only by activity in a later window, and then at
    that tick; a real stall of an ordinary series is still found."""
    DT = 900.0
    st = MetricStore()
    chk = _StaleChecker()
    t0 = 1000 * 3600.0 - DT                   # one tick before a window edge
    writes = {t0, t0 + DT} | {t0 + k * DT for k in range(5, 24, 4)}
    for k in range(16):
        ts = t0 + k * DT
        _raw(st, ts)
        st.add_vec(S, E, "feature.active", ts, [1.0])
        if ts in writes:
            st.add_vec(S, E, "behavior.xsys", ts, [1.0])
        chk.check(st, ts, DT)
    assert not chk.found
    # an established series that stops is stale
    for k in range(16, 24):
        ts = t0 + k * DT
        _raw(st, ts)
        st.add_vec(S, E, "feature.active", ts, [1.0])
        if ts in writes:
            st.add_vec(S, E, "behavior.xsys", ts, [1.0])
        if k < 20:
            st.add_vec(S, E, "behavior.tick_series", ts, [1.0])
        chk.check(st, ts, DT)
    assert [n for (_, _, n) in chk.found] == ["behavior.tick_series"]


def test_a_stalled_per_window_series_is_stale():
    DT = 900.0
    st = MetricStore()
    chk = _StaleChecker()
    t0 = 2000 * 3600.0
    for k in range(20):
        ts = t0 + k * DT
        _raw(st, ts)
        st.add_vec(S, E, "feature.active", ts, [1.0])
        st.add_vec(S, E, "behavior.other", ts, [1.0])
        if k in (1, 5):                      # then B21 stops scoring the pair
            st.add_vec(S, E, "behavior.xsys", ts, [1.0])
        chk.check(st, ts, DT)
    assert [n for (_, _, n) in chk.found] == ["behavior.xsys"]
    # sporadic activity inside the last write's window owes nothing
    st2, chk2 = MetricStore(), _StaleChecker()
    for k in range(12):
        ts = t0 + k * DT
        _raw(st2, ts)
        act = 1.0 if k in (1, 3) else 0.0
        st2.add_vec(S, E, "feature.active", ts, [act])
        st2.add_vec(S, E, "behavior.other", ts, [1.0])
        if k in (0, 1):
            st2.add_vec(S, E, "behavior.xsys", ts, [1.0])
        chk2.check(st2, ts, DT)
    assert not chk2.found
