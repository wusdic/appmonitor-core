"""MetricStore v2 (contract B/D): vector rings, virtual views, freshness,
pseudo-entities, indexed events/matches, labels, incidents, models,
checkpoints, health, retention, timeline, snapshot."""
import math
import time

import numpy as np
import pytest

from helpers import (DT, T0, add_obs_tick, add_raw_series, add_vec_rows, make_store,
                     make_tctx, put_model, set_trust)

from app.core.store import CHECKPOINT_MAX, MetricStore
from app.models.schema import (BehaviorEvent, DerivedMetric, EntityProfile, Incident, Label,
                               RawMetric, Severity, SignatureMatch)

S, E = "sys", "10.0.0.1"
H = 3600.0


# ------------------------------------------------------------------ vectors
def test_vec_append_tail_since_at():
    st = make_store()
    rows = np.arange(40, dtype=np.float32).reshape(10, 4)
    ts = add_vec_rows(st, S, E, "feature.vec", rows)
    t, M = st.vec_tail(S, E, "feature.vec", 3)
    assert M.dtype == np.float32 and M.shape == (3, 4)
    assert list(t) == ts[-3:]
    np.testing.assert_array_equal(M, rows[-3:])
    t, M = st.vec_tail(S, E, "feature.vec", 100)          # more than stored
    assert M.shape == (10, 4)
    t, M = st.vec_since(S, E, "feature.vec", ts[7])       # inclusive
    assert list(t) == ts[7:]
    t, M = st.vec_range(S, E, "feature.vec", ts[2], ts[4])
    assert list(t) == ts[2:5]
    np.testing.assert_array_equal(st.vec_at(S, E, "feature.vec", ts[5]), rows[5])
    assert st.vec_at(S, E, "feature.vec", ts[5] + 1) is None
    assert st.vec_latest(S, E, "feature.vec")[0] == ts[-1]
    # copies, not views into the ring
    M[:] = -1
    assert st.vec_tail(S, E, "feature.vec", 1)[1][0, 0] == rows[-1, 0]
    # missing series -> empty arrays with the known dim
    t, M = st.vec_tail(S, "other", "feature.vec", 5)
    assert t.shape == (0,) and M.shape == (0, 4)
    assert E in st.entities(S)
    assert st.vec_names(S, E) == ["feature.vec"] and st.vec_dim("feature.vec") == 4


def test_vec_dim_mismatch_and_ordering():
    st = make_store()
    st.add_vec(S, E, "v", 10.0, [1, 2, 3])
    with pytest.raises(ValueError):
        st.add_vec(S, E, "v", 11.0, [1, 2])
    with pytest.raises(ValueError):
        st.add_vec(S, E, "v", 9.0, [1, 2, 3])               # out of order
    st.add_vec(S, E, "v", 10.0, [7, 8, 9])                  # same ts: overwrite
    t, M = st.vec_tail(S, E, "v", 5)
    assert list(t) == [10.0] and list(M[0]) == [7, 8, 9]
    st.add_vec(S, E, "v", 12.0, [np.nan, 1, 2])             # NaN is legal
    assert math.isnan(st.vec_tail(S, E, "v", 1)[1][0, 0])


def test_vec_ring_wraparound_and_growth():
    st = MetricStore(max_points=100)
    st.set_retention("ring.", max_points=50)
    n = 173                                                 # grows 64 -> 50 cap, wraps
    for i in range(n):
        st.add_vec(S, E, "ring.x", float(i), [i, -i])
    t, M = st.vec_tail(S, E, "ring.x", 1000)
    assert len(t) == 50
    assert list(t) == [float(i) for i in range(n - 50, n)]
    np.testing.assert_array_equal(M[:, 0], np.arange(n - 50, n, dtype=np.float32))
    # since / at / range straddling the physical wrap point
    t2, M2 = st.vec_since(S, E, "ring.x", 150.0)
    assert list(t2) == [float(i) for i in range(150, n)]
    assert st.vec_at(S, E, "ring.x", 130.0)[1] == -130
    assert st.vec_at(S, E, "ring.x", 100.0) is None         # evicted
    t3, _ = st.vec_range(S, E, "ring.x", 125.5, 140.0)
    assert list(t3) == [float(i) for i in range(126, 141)]


def test_vec_age_retention():
    st = make_store()
    st.set_retention("short.", max_age_s=3 * DT)
    add_vec_rows(st, S, E, "short.v", np.zeros((10, 2)))
    t, _ = st.vec_tail(S, E, "short.v", 100)
    assert len(t) == 4 and t[-1] - t[0] == 3 * DT


def test_virtual_views_are_columns_not_copies():
    st = make_store()
    names = ["bytes_up", "bytes_down", "flows"]
    st.register_vector_names("feature.vec", names, "feature.")
    rows = np.array([[1, 2, 3], [4, np.nan, 6]], dtype=np.float32)
    ts = add_vec_rows(st, S, E, "feature.vec", rows)
    assert {"feature.bytes_up", "feature.bytes_down", "feature.flows"} <= set(st.derived_names(S, E))
    ser = st.derived_series(S, E, "feature.flows")
    assert [m.value for m in ser] == [3.0, 6.0] and [m.ts for m in ser] == ts
    assert isinstance(ser[0], DerivedMetric) and ser[0].inputs == ["feature.vec"]
    assert st.latest_derived(S, E, "feature.bytes_up").value == 4.0
    assert math.isnan(st.latest_derived(S, E, "feature.bytes_down").value)
    assert [m.value for m in st.derived_tail(S, E, "feature.bytes_up", 1)] == [4.0]
    assert st.latest_fresh(S, E, "feature.flows", ts[-1]) == 6.0
    assert st.latest_fresh(S, E, "feature.flows", ts[-1] + DT) is None
    assert st.last_write_ts(S, E, "feature.flows") == ts[-1]
    # not stored: no DerivedMetric objects exist for the views
    assert st.memory_report()["derived_points"] == 0
    # NaN views are skipped by snapshot
    snap = st.snapshot(S, E)
    assert snap["feature.bytes_up"] == 4.0 and "feature.bytes_down" not in snap
    # entity without the vector has no virtual names
    assert st.derived_names(S, "other") == []
    assert st.derived_series(S, "other", "feature.flows") == []


def test_virtual_detector_views_and_dict_series():
    st = make_store()
    st.register_vector_names("behavior.p", ["marg_int", "t2"], "behavior.p.")
    st.add_vec(S, E, "behavior.p", T0, [0.5, np.nan])
    assert st.latest_derived(S, E, "behavior.p.marg_int").value == 0.5
    # dict-valued series still go through add_derived
    st.add_derived(DerivedMetric("behavior.alarm", {"path": "single", "severity": "low"},
                                 T0, S, E, 900))
    assert st.latest_fresh(S, E, "behavior.alarm", T0)["path"] == "single"


# ------------------------------------------------------------------ tails / freshness
def test_raw_tail_and_freshness():
    st = make_store()
    ts = add_raw_series(st, S, E, "http.requests", [1, 2, 3, 4, 5])
    assert [m.value for m in st.raw_tail(S, E, "http.requests", 2)] == [4, 5]
    assert [m.value for m in st.raw_tail(S, E, "http.requests", 50)] == [1, 2, 3, 4, 5]
    assert st.raw_tail(S, E, "nope", 3) == []
    assert st.latest_fresh(S, E, "http.requests", ts[-1]) == 5
    assert st.latest_fresh(S, E, "http.requests", ts[-1] + DT) is None
    assert st.latest_raw_at(S, E, "http.requests", ts[2]).value == 3
    assert st.latest_raw_at(S, E, "http.requests", ts[2] + 1) is None
    from helpers import add_derived_series
    dts = add_derived_series(st, S, E, "derived.error_rate", [0.1, 0.2, 0.3])
    assert [m.value for m in st.derived_tail(S, E, "derived.error_rate", 2)] == [0.2, 0.3]
    assert st.latest_fresh(S, E, "derived.error_rate", dts[-1]) == 0.3


def test_snapshot_fresh_only():
    st = make_store()
    add_obs_tick(st, S, E, T0, {"http.requests": 10, "l4.flows": 3})
    add_obs_tick(st, S, E, T0 + DT, {"http.requests": 12})
    assert st.snapshot(S, E) == {"http.requests": 12.0, "l4.flows": 3.0}
    assert st.snapshot(S, E, now=T0 + DT) == {"http.requests": 12.0}


# ------------------------------------------------------------------ entities
def test_first_last_seen_and_touch():
    st = make_store()
    add_obs_tick(st, S, E, T0, {"l4.flows": 1})
    add_obs_tick(st, S, E, T0 + DT, {"l4.flows": 1})
    add_obs_tick(st, S, E, T0 + 2 * DT, {"act.events": 0}, touch=False)   # zero fill
    assert st.first_seen(S, E) == T0
    assert st.last_seen(S, E) == T0 + DT
    assert st.last_write_ts(S, E, "act.events") == T0 + 2 * DT
    add_obs_tick(st, S, "10.0.0.2", T0 + 2 * DT, {"l4.flows": 1})
    assert st.entities_active(S, T0 + 2 * DT) == ["10.0.0.2"]
    assert st.entities_active(S, T0) == [E, "10.0.0.2"]
    assert st.first_seen(S, "ghost") is None


def test_pseudo_entity_guard_and_registry():
    st = make_store()
    for bad in ("__system__", "class:r1", "__org__"):
        with pytest.raises(ValueError):
            st.add_raw(RawMetric("l4.flows", 1, T0, S, bad))
    add_obs_tick(st, S, E, T0, {"l4.flows": 1})
    st.add_vec(S, "class:r1", "behavior.class", T0, [1.0])
    st.add_derived(DerivedMetric("behavior.risk", 0.2, T0, S, "__system__", 900))
    put_model(st, "__org__", "__org__", "model.class", {"roles": {}})
    st.put_profile(EntityProfile(system=S, entity="class:static:dmz"))
    assert st.entities(S) == [E]
    assert st.pseudo_entities(S) == ["__system__", "class:r1", "class:static:dmz"]
    assert st.entities(S, include_pseudo=True) == [E, "__system__", "class:r1", "class:static:dmz"]
    assert st.pseudo_entities("__org__") == ["__org__"]
    assert "__org__" not in st.systems()


# ------------------------------------------------------------------ events / matches
def _ev(ts, kind="first_seen", ent=E, sys_=S):
    return BehaviorEvent(system=sys_, entity=ent, ts=ts, kind=kind, score=0.5)


def test_events_indexed_queries():
    st = make_store()
    ids = []
    for i in range(20):
        ids.append(st.add_event(_ev(T0 + i * DT, kind="beacon" if i % 2 else "first_seen",
                                    ent=E if i % 4 else "10.0.0.9")))
    st.add_event(_ev(T0 + 5 * DT, sys_="other"))
    assert len(set(ids)) == 20 and all(ids)
    evs = st.events(system=S, limit=200)
    assert len(evs) == 20 and evs[0].ts > evs[-1].ts                  # newest first
    evs = st.events(system=S, entity=E, since=T0 + 10 * DT)
    assert [e.ts for e in evs] == [T0 + i * DT for i in range(19, 9, -1) if i % 4]
    evs = st.events(system=S, since=T0 + 10 * DT, kinds=["beacon"])
    assert all(e.kind == "beacon" for e in evs) and len(evs) == 5
    assert len(st.events(limit=3)) == 3
    assert len(st.events()) == 21
    assert [e.system for e in st.events(entity=E, since=T0 + 5 * DT)].count("other") == 1
    assert st.events(system="nosuch") == []
    # late (out of order) insert lands in ts order
    st.add_event(_ev(T0 + 0.5 * DT, kind="late"))
    assert st.events(system=S, entity=E, since=T0, kinds=["late"])[0].ts == T0 + 0.5 * DT
    got = st.events(system=S, entity=E, since=T0)
    assert [e.ts for e in got] == sorted((e.ts for e in got), reverse=True)


def test_event_get_update():
    st = make_store()
    eid = st.add_event(_ev(T0))
    assert st.get_event(eid).status == "open"
    st.update_event(eid, status="acked", incident_id="inc1")
    assert st.get_event(eid).status == "acked" and st.get_event(eid).incident_id == "inc1"
    with pytest.raises(ValueError):
        st.update_event(eid, ts=T0 + 1)
    with pytest.raises(ValueError):
        st.update_event(eid, status="bogus")
    with pytest.raises(KeyError):
        st.update_event("nope", status="closed")
    e2 = _ev(T0)
    e2.id = eid
    with pytest.raises(ValueError):
        st.add_event(e2)


def test_event_and_match_retention():
    st = MetricStore(max_points=50)
    for i in range(80):
        st.add_event(_ev(T0 + i))
    assert len(st.events(limit=1000)) == 50
    assert st.events(system=S, entity=E, limit=1000)[-1].ts == T0 + 30
    old_id = st.events(limit=1000)[-1].id
    st.add_event(_ev(T0 + 31 * 86400.0))                  # 30-day age cut
    assert len(st.events(system=S, limit=1000)) == 1
    assert st.get_event(old_id) is None


def test_matches_indexed():
    st = make_store()
    for i in range(10):
        st.add_match(SignatureMatch(S, E, T0 + i * DT, "sig", "lbl",
                                    "scan" if i < 5 else "exfil", 0.9))
    ms = st.matches(system=S, entity=E, since=T0 + 3 * DT, categories=["scan"])
    assert [m.ts for m in ms] == [T0 + 4 * DT, T0 + 3 * DT]
    assert len(st.matches(limit=4)) == 4
    assert st.matches(system=S, entity="x") == []


def test_indexed_query_is_sublinear():
    st = make_store()
    for i in range(20000):
        st.add_event(_ev(T0 + i))
    t0 = time.perf_counter()
    for _ in range(200):
        st.events(system=S, entity=E, since=T0 + 19990)
    assert (time.perf_counter() - t0) / 200 < 1e-3


# ------------------------------------------------------------------ labels / incidents / profiles
def test_labels():
    st = make_store()
    a = st.add_label(Label(system=S, entity=E, target_type="event", target_id="ev1",
                           verdict="fp", ts=T0))
    st.add_label(Label(system=S, entity="class:r1", target_type="class", verdict="tp",
                       scope="class", ts=T0 + 10))
    assert a and len(st.labels()) == 2
    assert [lb.id for lb in st.labels(system=S, entity=E)] == [a]
    assert len(st.labels(since=T0 + 5)) == 1
    with pytest.raises(ValueError):
        Label(verdict="maybe")


def test_incidents():
    st = make_store()
    inc = Incident(system=S, entity=E, entities=[E, "10.0.0.5"], opened=T0, last_seen=T0,
                   severity=Severity.MEDIUM)
    iid = st.put_incident(inc)
    assert iid and st.get_incident(iid) is inc
    inc.status, inc.last_seen, inc.close_reason = "closed", T0 + DT, "returned"
    assert st.put_incident(inc) == iid                      # update by id
    st.put_incident(Incident(system=S, entity="class:r1", opened=T0 + 2 * DT,
                             last_seen=T0 + 2 * DT))
    assert len(st.incidents(system=S)) == 2
    assert st.incidents(system=S)[0].entity == "class:r1"   # newest first
    assert [i.id for i in st.incidents(entity="10.0.0.5")] == [iid]
    assert st.incidents(status="open")[0].entity == "class:r1"
    assert len(st.incidents(status=["open", "closed"], since=T0 + DT)) == 2
    assert st.incidents(since=T0 + 1.5 * DT)[0].entity == "class:r1"
    # 90-day retention
    st.put_incident(Incident(system=S, entity=E, opened=T0 + 91 * 86400.0,
                             last_seen=T0 + 91 * 86400.0))
    assert st.get_incident(iid) is None
    with pytest.raises(ValueError):
        Incident(close_reason="because")


def test_profile_versions():
    st = make_store()
    for v in range(15):
        st.put_profile_version(S, E, v, {"portrait": v}, ts=T0 + v)
    pv = st.profile_versions(S, E)
    assert len(pv) == 12 and pv[0].version == 14 and pv[-1].version == 3
    assert st.profile_versions(S, E, n=2)[1].obj == {"portrait": 13}
    st.put_profile_version(S, "e2", 1, EntityProfile(system=S, entity="e2", updated=T0))
    assert st.profile_versions(S, "e2")[0].ts == T0


# ------------------------------------------------------------------ models / checkpoints
def test_models_and_versions():
    st = make_store()
    assert st.get_model(S, E, "model.baseline", default="none") == "none"
    assert st.model_version(S, E, "model.baseline") is None
    m = {"n_eff": 3}
    st.put_model(S, E, "model.baseline", m)
    assert st.get_model(S, E, "model.baseline") is m and st.model_version(S, E, "model.baseline") == 1
    st.put_model(S, E, "model.baseline", m)
    assert st.model_version(S, E, "model.baseline") == 2
    st.put_model(S, E, "model.baseline", m, version=7)
    assert st.model_version(S, E, "model.baseline") == 7
    st.put_model(S, "__system__", "model.link", {"version": 4, "links": []})
    assert st.model_version(S, "__system__", "model.link") == 4
    assert st.model_names(S, E) == ["model.baseline"]
    st.put_model(S, E, "model.cp", {}, ts=T0)
    assert st.last_write_ts(S, E, "model.cp") == T0


def test_checkpoint_basic():
    st = make_store()
    assert st.get_checkpoint(S, E, "baseline.current") is None
    st.put_checkpoint(S, E, "baseline.current", T0, "a")
    st.put_checkpoint(S, E, "baseline.current", T0 + H, "b")
    st.put_checkpoint(S, E, "baseline.current", T0 + H, "b2")          # same ts: replace
    assert st.get_checkpoint(S, E, "baseline.current") == (T0 + H, "b2")
    assert st.get_checkpoint(S, E, "baseline.current", T0 + H - 1) == (T0, "a")
    assert st.get_checkpoint(S, E, "baseline.current", T0 - 1) is None
    assert st.get_checkpoint(S, E, "other") is None


@pytest.mark.parametrize("step", [H, 900.0, 60.0 * 7])
def test_checkpoint_geometric_retention(step):
    st = make_store()
    n = int(400 * H / step)
    worst = 0.0
    for i in range(n):
        t = T0 + i * step
        st.put_checkpoint(S, E, "lrn", t, i)
        times = st.checkpoint_times(S, E, "lrn")
        assert len(times) <= CHECKPOINT_MAX
        assert times[-1] == t                                   # newest always kept
        if i * step > 200 * H:
            # a restorable state exists for any rollback up to 7 days back ...
            for age_h in (1, 2, 3, 5, 10, 24, 50, 100, 168):
                tgt = t - age_h * H
                got = st.get_checkpoint(S, E, "lrn", tgt)
                assert got is not None and got[0] <= tgt
                worst = max(worst, (tgt - got[0]) / (age_h * H))
            # ... and nothing is kept beyond the one floor checkpoint past 168 h
            assert sum(1 for x in times if t - x > 168 * H) <= 1
    # spacing is geometric: the state found is within a few multiples of the age
    assert worst < 3.0
    ages = sorted((times[-1] - x) / H for x in times)
    assert ages[1] <= 2.0 and ages[-1] >= 168.0


# ------------------------------------------------------------------ health / retention / memory
def test_health_and_last_write():
    st = make_store()
    st.put_health("behavior.x", {"ok": False, "error_count": 2})
    h = st.health()
    assert h["behavior.x"]["error_count"] == 2
    h["behavior.x"]["error_count"] = 99                        # returned copy
    assert st.health()["behavior.x"]["error_count"] == 2
    assert st.last_write_ts(S, E, "nothing") is None
    add_raw_series(st, S, E, "l4.flows", [1, 2], t0=T0)
    assert st.last_write_ts(S, E, "l4.flows") == T0 + DT


def test_default_retention_table():
    st = make_store()
    n = 40                                                     # 10 h at 900 s
    add_raw_series(st, S, E, "http.status_4xx", list(range(n)))
    add_raw_series(st, S, E, "http.requests", list(range(n)))   # D0 input: 24 h
    add_raw_series(st, S, E, "tls.sni_set", [{"a.com": 1}] * n)
    from helpers import add_derived_series
    add_derived_series(st, S, E, "derived.error_rate", [0.1] * n)
    add_derived_series(st, S, E, "feature.legacy", [1.0] * n)  # no rule: max_points only
    set_trust(st, S, E, [T0 + i * DT for i in range(n)], 1.0)
    span = lambda ser: ser[-1].ts - ser[0].ts
    assert span(st.raw_series(S, E, "http.status_4xx")) <= 6 * H
    assert len(st.raw_series(S, E, "http.status_4xx")) == 25
    assert len(st.raw_series(S, E, "http.requests")) == n
    assert span(st.raw_series(S, E, "tls.sni_set")) <= 1 * H
    assert span(st.derived_series(S, E, "derived.error_rate")) <= 2 * H
    assert len(st.derived_series(S, E, "feature.legacy")) == n
    # governance scalars are 1-element vec rings kept 8 d
    assert len(st.vec_since(S, E, "behavior.trust", T0 - 1)[0]) == n
    assert len(st.vec_since(S, E, "behavior.trust_prov", T0 - 1)[0]) == n
    assert st.vec_at(S, E, "behavior.quarantine", T0)[0] == 0.0


def test_set_retention_overrides_and_max_points():
    st = make_store()
    add_raw_series(st, S, E, "l4.syn_count", list(range(10)))
    st.set_retention("l4.", max_points=3)
    st.add_raw(RawMetric("l4.syn_count", 99, T0 + 10 * DT, S, E))
    assert [m.value for m in st.raw_series(S, E, "l4.syn_count")] == [8, 9, 99]
    st.set_retention("behavior.foo", max_age_s=DT)
    from helpers import add_derived_series
    add_derived_series(st, S, E, "behavior.foo", [1, 2, 3, 4])
    assert [m.value for m in st.derived_series(S, E, "behavior.foo")] == [3, 4]


def test_memory_report():
    st = make_store()
    add_vec_rows(st, S, E, "feature.vec", np.zeros((50, 52)))       # within 1 d
    add_raw_series(st, S, E, "l4.flows", [1] * 10)
    st.add_event(_ev(T0))
    st.put_checkpoint(S, E, "l", T0, np.zeros(100, dtype=np.float16))
    rep = st.memory_report()
    assert rep["vec_rows"] == 50 and rep["vec_bytes"] >= 50 * 52 * 4
    assert rep["raw_points"] == 10 and rep["events"] == 1 and rep["checkpoints"] == 1
    assert rep["checkpoint_bytes"] == 200 and rep["entities"] == 1
    assert rep["bytes_per_entity"] == rep["approx_bytes"] > 0


# ------------------------------------------------------------------ timeline
def test_timeline_merges_sources():
    st = make_store()
    st.add_event(_ev(T0 + 1 * DT))
    st.add_match(SignatureMatch(S, E, T0 + 2 * DT, "sig", "lbl", "scan", 0.9))
    st.put_incident(Incident(system=S, entity=E, opened=T0 + 3 * DT, last_seen=T0 + 4 * DT))
    st.put_profile_version(S, E, 1, {"p": 1}, ts=T0 + 5 * DT)
    for i, r in enumerate([0.1, 0.12, 0.13, 0.5, 0.52]):          # 2 material changes
        st.add_derived(DerivedMetric("behavior.risk", r, T0 + i * DT, S, E, 900))
    tl = st.timeline(S, E)
    assert [x["type"] for x in tl] == ["profile_version", "incident", "risk", "match",
                                       "event", "risk"]
    assert [x["ts"] for x in tl] == sorted((x["ts"] for x in tl), reverse=True)
    tl2 = st.timeline(S, E, since=T0 + 2 * DT)
    assert {x["type"] for x in tl2} == {"profile_version", "incident", "risk", "match"}
    assert len(st.timeline(S, E, limit=2)) == 2


def test_make_tctx_helper():
    # 2026-10-01T02:00Z is 10:00 in Shanghai on a Thursday (engines.md B01 (c))
    ts = 1790820000.0
    c = make_tctx(ts, holidays=["2026-10-01"])
    assert c["hour_local"] == 10 and c["day_type"] == "nonworkday" and c["daypart"] == "nwd_day"
    c2 = make_tctx(ts)
    assert c2["daypart"] == "wd_day" and c2["dow"] == 3 and c2["bin168"] == 3 * 24 + 10
    # timebins encodings: slot counts 15-min slots since the local epoch,
    # bin48 = hour + 24 * nonworkday
    assert c2["slot"] == 1989832 and c2["bin48"] == 10 and c2["cc"] == 900
    assert c["bin48"] == 34 and c["daypart_id"] == 2


def test_checkpoint_retention_covers_every_rollback_target():
    """Hourly checkpoints for 10 days: for every target T up to 168 h back
    there must be a kept checkpoint in [T - 24 h, T] (gating's replay needs
    journal rows no older than 192 h), and the count stays bounded."""
    from app.core.store import CHECKPOINT_MAX, MetricStore
    store = MetricStore()
    H = 3600.0
    t0 = 1_700_000_000.0
    for h in range(240):
        now = t0 + h * H
        store.put_checkpoint("s", "e", "baseline", now, {"h": h})
        times = store.checkpoint_times("s", "e", "baseline")
        assert len(times) <= CHECKPOINT_MAX
        if h < 170:
            continue
        for back in range(0, 169):
            target = now - back * H
            got = store.get_checkpoint("s", "e", "baseline", at_or_before=target)
            assert got is not None
            assert target - got[0] <= 24 * H, (h, back, target - got[0])
