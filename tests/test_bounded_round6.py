"""lib-3 bounded mode, round 6 (docs/lib3/progressive.md §10, §16.13.12):
skipped / unearned IPs are represented correctly - a series a B-engine has
started is not left stale by the bounded prefilter."""
from __future__ import annotations

import sys
import os

import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "engines"))

from helpers import DT, T0, make_store, run_engine  # noqa: E402

from app.engines.behavior.lib import pactive as PA  # noqa: E402
from app.engines.behavior.lib import m_timing as TM  # noqa: E402
from app.engines.behavior.timing import SERIES, TimingEngine  # noqa: E402
from app.models.schema import ORG, DerivedMetric, MetricKind  # noqa: E402

S, E = "erp", "10.0.0.7"
BOUNDED = {"lib3": {"resource_mode": "bounded"}}


def _publish_sets(st, now, active, earned=()):
    """P15's record (model.budget) as the bounded engines read it."""
    st.put_model(ORG, ORG, PA.BUDGET, {"systems": {S: {"ts": now, "active": sorted(active),
                                                       "earned": sorted(earned)}}}, ts=now)


def _periodicity(st, now, v):
    st.add_derived(DerivedMetric(name="derived.periodicity_score", value=float(v), ts=now,
                                 system=S, entity=E, window_s=int(DT), kind=MetricKind.GAUGE))


def test_prefilter_continues_an_ip_the_engine_has_started_on():
    st = make_store()
    now = T0
    _publish_sets(st, now, [E])
    _periodicity(st, now, 0.3)
    assert not PA.periodic_candidate(st, S, E, BOUNDED, TM.MODEL)      # never started
    st.put_model(S, E, TM.MODEL, {"fmt": TM.FMT}, ts=now)
    assert PA.periodic_candidate(st, S, E, BOUNDED, TM.MODEL)          # started: kept
    assert not PA.periodic_candidate(st, S, E, BOUNDED)                # entry test alone
    assert PA.periodic_candidate(st, S, E, {}, TM.MODEL)               # full mode


def test_b11_series_not_stale_when_the_periodicity_score_dips_below_the_entry_threshold():
    """A machine client whose periodicity score hovers around 0.5 (round 5,
    bounded pack A: api-gateway's callers; 88 stale B11 series): B11 must keep
    writing behavior.timing on every active tick once it has started."""
    from test_b11_timing import Feed, scraper_times

    st, eng = make_store(), TimingEngine()
    feed = Feed(st, S)
    rng = np.random.default_rng(3)
    times = scraper_times(rng, T0 - DT, T0 + 40 * DT, gap=20.0, jitter=0.5)
    written = []
    for i in range(24):
        now = T0 + i * DT
        _publish_sets(st, now, [E])
        _periodicity(st, now, 0.62 if i < 8 else 0.41)
        feed.tick(E, now, times[(times > now - DT) & (times <= now)], DT)
        run_engine(eng, st, now, training=False, config=BOUNDED)
        written.append(any(m.ts == now for m in st.derived_tail(S, E, SERIES, 2)))
    assert all(written[:8])
    assert all(written[8:]), [i for i, w in enumerate(written) if not w]


def test_b11_never_starts_an_unearned_ip_below_the_threshold():
    from test_b11_timing import Feed, scraper_times

    st, eng = make_store(), TimingEngine()
    feed = Feed(st, S)
    rng = np.random.default_rng(4)
    times = scraper_times(rng, T0 - DT, T0 + 10 * DT, gap=20.0, jitter=0.5)
    for i in range(6):
        now = T0 + i * DT
        _publish_sets(st, now, [E])
        _periodicity(st, now, 0.2)
        feed.tick(E, now, times[(times > now - DT) & (times <= now)], DT)
        run_engine(eng, st, now, training=False, config=BOUNDED)
    assert st.get_model(S, E, TM.MODEL) is None
    assert not st.derived_tail(S, E, SERIES, 1)


# ---------------------------------------------------------------- B18 (round 6)
# bounded pack A seed 0 (round 5): class:r3 of erp-prod - every member idle
# beyond the linger window, so B01 wrote no row for any of them - had
# class_int / class_shape / class_rhythm degraded 'stale:feature.nat' (64
# pipeline_degraded events) and its behavior.class.agg[.h] series went stale.

def _b01_idle_rows(canon_ticks=4):
    """B01's rows for a registered entity without any data, canonical mode."""
    from app.engines.behavior.feature_vector import FeatureVectorEngine
    from app.engines.behavior.lib import grains as GR
    st, eng = make_store(), FeatureVectorEngine()
    st.register_entity(S, "10.0.9.1")
    t0 = math.floor(T0 / 3600.0) * 3600.0
    cfg = {"grain_mode": GR.CANONICAL}
    now = t0
    for k in range(1, canon_ticks + 1):
        now = t0 + k * DT
        run_engine(eng, st, now, config=cfg)
    return st, now


def test_implicit_idle_rows_equal_what_b01_writes_for_an_idle_entity():
    from app.engines.behavior import class_monitor as CM
    st, now = _b01_idle_rows()
    nat = st.vec_at(S, "10.0.9.1", "feature.nat", now)
    assert np.array_equal(np.asarray(nat, np.float32), CM.idle_row(DT).astype(np.float32),
                          equal_nan=True)
    nh = st.vec_at(S, "10.0.9.1", "feature.nat.h", now)
    mh = st.vec_at(S, "10.0.9.1", "feature.meta.h", now)
    assert nh is not None and float(mh[0]) == 0.0
    assert np.array_equal(np.asarray(nh, np.float32),
                          CM.idle_row_h(float(mh[1])).astype(np.float32), equal_nan=True)


def _b18_world(bounded_tick: bool):
    from test_b18_class_monitor import CK as _CK, put_classes, warm, member_tick
    import test_b18_class_monitor as TB18
    from app.engines.behavior import class_monitor as CM
    from app.engines.behavior.class_monitor import ClassMonitorEngine
    st, eng = make_store(), ClassMonitorEngine()
    ips = [f"10.0.5.{i}" for i in range(1, 6)]
    TB18.put_classes(st, {"r1": ips})
    warm(st, eng, ips)
    now = TB18.T + 96 * TB18.DT
    other = "10.0.7.7"                                   # the system's one active IP
    member_tick(st, other, now, TB18.nat_row(np.random.default_rng(5)))
    if bounded_tick:
        st.put_model(ORG, ORG, PA.BUDGET, {"systems": {TB18.S: {
            "ts": now, "active": [other], "earned": []}}}, ts=now)
        run_engine(eng, st, now, config=BOUNDED)
    else:
        for ip in ips:                                   # full mode: B01's idle rows
            member_tick(st, ip, now, CM.idle_row(TB18.DT), active=False)
        run_engine(eng, st, now)
    return st, now, _CK, TB18


def test_b18_members_outside_the_bounded_set_are_idle_not_stale():
    from app.engines.behavior.lib import emit
    from app.engines.behavior.class_monitor import AGG
    st_b, now, ck, TB18 = _b18_world(True)
    assert emit.read_dict(st_b, TB18.S, ck, emit.DEGRADED, now) == {}
    assert st_b.vec_at(TB18.S, ck, AGG, now) is not None             # the learner clock ticks
    row_b = emit.read_row(st_b, TB18.S, ck, emit.PM, now)
    assert "class_rhythm" in row_b and "class_int" not in row_b         # silent, not degraded
    # identical to full mode, where B01 wrote the members' idle rows
    st_f, _, _, _ = _b18_world(False)
    assert np.array_equal(st_b.vec_at(TB18.S, ck, AGG, now), st_f.vec_at(TB18.S, ck, AGG, now),
                          equal_nan=True)
    assert row_b == emit.read_row(st_f, TB18.S, ck, emit.PM, now)


def test_series_owed_ends_the_earned_only_series_of_a_demoted_ip():
    st = make_store()
    _publish_sets(st, T0, [E, "10.0.0.8"], earned=["10.0.0.8"])
    for n in ("behavior.wh", "behavior.rhythm", "behavior.cp.prob", "behavior.acc_alarm"):
        assert not PA.series_owed(st, S, E, n, BOUNDED)              # demoted / unearned
        assert PA.series_owed(st, S, "10.0.0.8", n, BOUNDED)         # earned
        assert PA.series_owed(st, S, E, n, {})                       # full mode
        assert PA.series_owed(st, S, "class:r1", n, BOUNDED)
    assert PA.series_owed(st, S, E, "behavior.timing", BOUNDED)      # not earned-only
