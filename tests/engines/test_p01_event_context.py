"""P01 EventContext (derived.event_context): time / calendar / session
context aligned with evt.batch, and metric-window events (card P01)."""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pytest

from helpers import ctx, make_store, obs

from app.engines.behavior.lib import pevent as EV
from app.engines.derived.event_context import EventContextEngine
from app.engines.raw.event_builder import EventBuilderEngine
from app.models.schema import DerivedMetric, RawMetric

# 2026-09-28 (Monday) 00:00 Asia/Shanghai
MON = _dt.datetime(2026, 9, 28, tzinfo=_dt.timezone(_dt.timedelta(hours=8))).timestamp()
ON = {"enabled": True}


def _cfg(**kw):
    p = dict(ON)
    p.update(kw.pop("progressive", {}))
    c = {"progressive": p}
    c.update(kw)
    return c


def _tick(store, observations, now, dt, cfg, p01=None):
    c = ctx(store, now, window_s=dt, config=cfg)
    EventBuilderEngine().safe_run(c, observations)
    p01 = p01 or EventContextEngine()
    p01.safe_run(c, None)
    return p01


def _ev(ip, ts, path="/a", sess=None):
    l7 = {"headers": {}}
    if sess:
        l7["sess"] = sess
    return obs("oa", ip, ts, peer="10.9.9.9", dst_port=80, http_method="GET", http_host="oa",
               http_path=path, http_status=200, extra={"l7": l7})


def _sessions(times, dt, ip="192.168.1.21", sess=None):
    st = make_store()
    cfg = _cfg()
    p01 = None
    out = {}
    t_end = max(times) + dt
    now = (min(times) // dt) * dt
    while now <= t_end + dt:
        batch_obs = [_ev(ip, t, sess=sess) for t in times if now - dt < t <= now]
        p01 = _tick(st, batch_obs, now, dt, cfg, p01)
        cb = st.batch_at("oa", EV.EVT_CTX, now)
        eb = st.batch_at("oa", EV.EVT_BATCH, now)
        if cb is not None:
            for i in range(cb.n):
                out[float(eb.ts[i])] = (cb.get("ctx.sid", i), cb.get("ctx.sess_pos", i),
                                        cb.get("ctx.think_s", i))
        now += dt
    return out


def test_time_context_and_calendar_classes():
    st = make_store()
    cfg = _cfg(calendar={"holidays": ["2026-10-01"], "makeup_workdays": ["2026-10-10"]})
    cases = {MON + 9 * 3600 + 5 * 60: ("workday", "workday"),          # Mon 09:05
             MON + 5 * 86400 + 3600: ("nonworkday", "weekend"),        # Sat
             MON + 3 * 86400 + 3600: ("nonworkday", "holiday"),        # Thu 2026-10-01
             MON + 12 * 86400 + 3600: ("workday", "makeup")}           # Sat 2026-10-10
    first = None
    for ts, (dtype, dclass) in sorted(cases.items()):     # retention is relative to the newest
        now = ts + 30
        _tick(st, [_ev("1.1.1.1", ts)], now, 60.0, cfg)
        cb = st.batch_at("oa", EV.EVT_CTX, now)
        assert cb.get("ctx.daytype", 0) == dtype and cb.get("ctx.dayclass", 0) == dclass
        first = first or cb
    cb = first
    assert cb.get("ctx.tod_min", 0) == 545.0 and cb.get("ctx.dow", 0) == 0
    assert cb.get("ctx.when", 0) == ("wd", 545) and cb.get("ctx.dom", 0) == 28
    assert cb.get("ctx.mend", 0) == 1                                  # Sep 28-30: last 3 workdays


def test_sessions_across_tick_boundaries_and_cadence_invariant_ids():
    """Gaps across tick boundaries keep one session; a 60-s and a 900-s run
    of the same events give identical session ids."""
    times = [MON + 9 * 3600 + k * 50.0 for k in range(12)] + \
            [MON + 14 * 3600 + k * 30.0 for k in range(5)]
    a = _sessions(times, 60.0)
    b = _sessions(times, 900.0)
    assert a == b
    sids = [a[t][0] for t in sorted(a)]
    assert len(set(sids[:12])) == 1 and len(set(sids[12:])) == 1 and sids[0] != sids[12]
    assert [a[t][1] for t in sorted(a)][:3] == [0, 1, 2]
    assert a[sorted(a)[1]][2] == pytest.approx(50.0)


def test_two_users_behind_one_ip_by_session_key():
    times = [MON + 9 * 3600 + k * 20.0 for k in range(6)]
    st = make_store()
    cfg = _cfg()
    now = MON + 9 * 3600 + 200
    o = [_ev("10.0.0.1", t, sess="A" if k % 2 else "B") for k, t in enumerate(times)]
    _tick(st, o, now, 300.0, cfg)
    cb = st.batch_at("oa", EV.EVT_CTX, now)
    eb = st.batch_at("oa", EV.EVT_BATCH, now)
    by = {}
    for i in range(cb.n):
        by.setdefault(eb.get("sess.key", i), set()).add(cb.get("ctx.sid", i))
    assert len(by) == 2 and all(len(v) == 1 for v in by.values())
    assert set.union(*by.values()).__len__() == 2


def test_window_events_new_metric_enters_and_w_max_priority():
    st = make_store()
    cfg = _cfg(progressive={"defaults": {"w_max": 50}})
    day3 = MON + 2 * 86400 + 10 * 3600
    ips = [f"10.0.{i // 250}.{i % 250}" for i in range(400)]
    o = [_ev(ip, day3 - 10) for ip in ips]
    for k, ip in enumerate(ips):
        st.add_raw(RawMetric(name="http.requests", value=float(k % 7), ts=day3, system="oa",
                             entity=ip))
    st.add_derived(DerivedMetric(name="derived.brand_new_metric", value=2.5, ts=day3, system="oa",
                                 entity=ips[0], window_s=60))
    _tick(st, o, day3, 60.0, cfg)
    wb = st.batch_at("oa", EV.EVT_WIN, day3)
    assert wb is not None and wb.kind == EV.KIND_WIN and wb.n == 50
    assert wb.meta["n_active"] == 400
    # every IP was new in the grain -> equal priority; HT mass estimates the population
    # priority sampling: unbiased (tests/lib), one draw of 50 of 400 has sd ~ 13 %
    assert wb.mass().sum() == pytest.approx(400, rel=0.35)
    mean_ht = float(np.sum(wb.dense("m.http.requests", 0.0) * wb.mass()) / wb.mass().sum())
    assert mean_ht == pytest.approx(np.mean([k % 7 for k in range(400)]), rel=0.2)
    rows0 = [i for i in range(wb.n) if wb.ip_of(i) == ips[0]]
    if rows0:
        assert wb.get("derived.brand_new_metric", rows0[0]) is EV.ABSENT
        assert wb.get("m.derived.brand_new_metric", rows0[0]) == 2.5
    assert all(wb.get("ev.ch", i) == "win" for i in range(wb.n))


def test_normal_day_flag_skips_holidays():
    st = make_store()
    cfg = _cfg(calendar={"holidays": ["2026-10-01"]})
    p01 = EventContextEngine()
    for d in range(6):                                          # Mon .. Sat
        now = MON + d * 86400 + 10 * 3600
        p01 = _tick(st, [_ev("1.1.1.1", now - 5)], now, 60.0, cfg, p01)
    m = st.get_model("oa", "__system__", "model.pcal")
    hol = _dt.date(2026, 10, 1).toordinal()
    assert m["normal"][hol] is False
    assert m["normal"][_dt.date(2026, 9, 29).toordinal()] is True
