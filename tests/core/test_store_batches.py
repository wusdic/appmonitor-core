"""MetricStore batch series (progressive core §5.6): add / at / since,
replacement, age retention relative to the newest batch, compaction to the
learned rows, the ops.tick clock, and the Templater's read-only apply_path."""
from __future__ import annotations

import numpy as np

from app.core.store import MetricStore
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib.template import Templater


def _batch(n=4, learn=None):
    b = EV.BatchBuilder("oa")
    for i in range(n):
        b.add(float(i), f"10.0.0.{i}", {"http.route": f"GET oa /r{i}", "net.bytes_up": 10.0 * i})
    bt = b.build(0.0, 60.0)
    if learn is not None:
        bt.learn[:] = learn
    return bt


def test_add_at_since_replace():
    st = MetricStore()
    st.add_batch("oa", EV.EVT_BATCH, 60.0, "a")
    st.add_batch("oa", EV.EVT_BATCH, 120.0, "b")
    st.add_batch("oa", EV.EVT_BATCH, 90.0, "mid")                 # late: kept sorted
    assert st.batch_at("oa", EV.EVT_BATCH, 120.0) == "b"
    assert st.batch_at("oa", EV.EVT_BATCH, 100.0) is None
    assert [t for t, _ in st.batches_since("oa", EV.EVT_BATCH, 60.0)] == [90.0, 120.0]
    st.add_batch("oa", EV.EVT_BATCH, 120.0, "b2")
    assert st.batch_at("oa", EV.EVT_BATCH, 120.0) == "b2"
    assert st.batch_systems(EV.EVT_BATCH) == ["oa"]
    assert st.entities("oa") == [] and st.first_seen("oa", "10.0.0.1") is None   # no entity side effects


def test_age_retention_relative_to_newest_and_raise_only():
    st = MetricStore()
    for k in range(20):
        st.add_batch("oa", EV.EVT_BATCH, 900.0 * k, k)
    ts = st.batch_times("oa", EV.EVT_BATCH)
    assert ts[0] >= ts[-1] - 4500.0 and ts[-1] == 900.0 * 19
    st.ensure_retention("evt.", max_age_s=9000.0)
    for k in range(20, 40):
        st.add_batch("oa", EV.EVT_BATCH, 900.0 * k, k)
    ts = st.batch_times("oa", EV.EVT_BATCH)
    assert ts[-1] - ts[0] <= 9000.0 and ts[-1] - ts[0] > 4500.0


def test_compact_batch_keeps_learned_rows_and_ids():
    st = MetricStore()
    bt = _batch(4, learn=[True, False, True, False])
    st.add_batch("oa", EV.EVT_BATCH, 60.0, bt)
    before = st.memory_report()["batch_bytes"]
    assert st.compact_batch("oa", EV.EVT_BATCH, 60.0, keep_cols=["http.route"])
    c = st.batch_at("oa", EV.EVT_BATCH, 60.0)
    assert c.n == 2 and list(c.rid) == [0, 2] and c.names() == ["http.route"]
    assert st.memory_report()["batch_bytes"] < before
    assert not st.compact_batch("oa", EV.EVT_BATCH, 999.0)
    assert st.compact_batch("oa", EV.EVT_BATCH, 60.0, fn=lambda b: "x")
    assert st.batch_at("oa", EV.EVT_BATCH, 60.0) == "x"
    assert st.drop_batches("oa", EV.EVT_BATCH) == 1


def test_tick_clock():
    st = MetricStore()
    for k in range(200):
        st.put_tick("oa", 900.0 * k)
    t = st.tick_times("oa")
    assert t[-1] == 900.0 * 199 and t[-1] - t[0] <= 25 * 3600
    assert st.tick_times("oa", since=900.0 * 197) == [900.0 * 198, 900.0 * 199]
    assert st.tick_times("none") == []


def test_templater_apply_path_is_read_only():
    tp = Templater()
    for i in range(20):
        tp.template_path("oa.local", "POST", f"/approval/{1000 + i}/approve")
        tp.template_path("oa.local", "GET", "/approval/list")
    snap = tp.to_dict()
    r = tp.apply_path("oa.local", "POST", "/approval/99999/approve")
    assert r == "POST oa.local /approval/{num}/approve"
    assert tp.apply_path("oa.local", "GET", "/approval/list") == "GET oa.local /approval/list"
    assert tp.apply_path("oa.local", "GET", "/never/seen") == "GET oa.local /{var}/{var}"
    assert tp.to_dict() == snap                                  # no count / node / vocab change
