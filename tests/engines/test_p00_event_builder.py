"""P00 EventBuilder (raw.event): observations -> open-attribute event batches
(docs/lib3/progressive.md card P00, tests (a)-(j) at engine level)."""
from __future__ import annotations

import numpy as np
import pytest

from helpers import T0, ctx, make_store, obs

from app.engines.behavior.lib import pevent as EV
from app.engines.raw.action_token import ActionTokenEngine
from app.engines.raw.event_builder import EventBuilderEngine
from app.models.schema import AcquisitionMethod

ON = {"progressive": {"enabled": True}}


def _run(store, observations, now=T0 + 60, config=None, dt=60.0, with_r2=True):
    cfg = dict(ON)
    cfg.update(config or {})
    c = ctx(store, now, window_s=dt, config=cfg)
    if with_r2:
        ActionTokenEngine().safe_run(c, observations)
    eng = EventBuilderEngine()
    n = eng.safe_run(c, observations)
    return eng, n


def _login(ip, user, ts, **kw):
    body = f"username={user}&password=secret1&captcha=1234"
    extra = {"l7": {"body": body, "body_len": len(body),
                    "headers": {"content-type": "application/x-www-form-urlencoded",
                                "user-agent": "Mozilla/5.0"}}}
    extra.update(kw.pop("extra", {}))
    return obs("oa", ip, ts, peer="192.168.100.100", dst_port=8080, http_method="POST",
               http_host="oa.local", http_path="/login", http_status=302, bytes_up=1500,
               bytes_down=400, user_agent="Mozilla/5.0", extra=extra, **kw)


def test_disabled_by_default_writes_nothing():
    st = make_store()
    c = ctx(st, T0 + 60)
    assert EventBuilderEngine().safe_run(c, [_login("192.168.1.21", "jack", T0)]) == 0
    assert st.batch_systems(EV.EVT_BATCH) == []


def test_open_attributes_route_body_policy():
    st = make_store()
    o = [_login("192.168.1.21", "jack", T0 + i) for i in range(5)]
    eng, n = _run(st, o)
    b = st.batch_at("oa", EV.EVT_BATCH, T0 + 60)
    assert n == 5 and b.n == 5 and b.kind == EV.KIND_TXN
    assert b.get("net.src", 0) == "192.168.1.21"
    assert b.get("net.dst", 0) == "192.168.100.100:8080"
    assert b.get("http.route", 0) == "POST oa.local /login"
    assert b.get("ev.ch", 0) == "http" and b.get("http.sclass", 0) == "3xx"
    assert b.get("body.kv.username", 0) == "jack"
    assert b.get("body.kv.password", 0) == "L6 D1"                  # secret: shape only
    assert b.get("body.keys", 0) == frozenset({"username", "password", "captcha"})
    assert b.get("hdr.content-type", 0) == "application/x-www-form-urlencoded"
    assert b.get("client.stack", 0) is not EV.ABSENT
    assert b.meta["policy"]["body.kv.password"] == "shape"
    assert b.learn.all() and (b.pi == 1).all()                      # below E_learn


def test_aggregated_record_ev_sample_and_approx():
    """(c) count 40 with 8 ev_sample rows -> 8 events of weight 5 with their
    own sizes; without ev_sample one event of weight 40 flagged approx."""
    st = make_store()
    rows = [{"o": float(i), "up": 1000 + 100 * i, "down": 300, "st": 200} for i in range(8)]
    agg = obs("oa", "192.168.1.23", T0, peer="192.168.100.100", dst_port=8080,
              http_method="GET", http_host="oa.local", http_path="/doc/list", http_status=200,
              bytes_up=1400, bytes_down=300,
              extra={"count": 40, "bytes_up_total": 56000, "bytes_down_total": 12000,
                     "ev_sample": rows})
    plain = obs("oa", "192.168.1.24", T0, peer="192.168.100.100", dst_port=8080,
                http_method="GET", http_host="oa.local", http_path="/doc/list", http_status=200,
                bytes_up=1400, bytes_down=300,
                extra={"count": 40, "bytes_up_total": 56000, "bytes_down_total": 12000})
    _run(st, [agg, plain])
    b = st.batch_at("oa", EV.EVT_BATCH, T0 + 60)
    ips = np.asarray([b.ip_of(i) for i in range(b.n)])
    r23 = np.flatnonzero(ips == "192.168.1.23")
    r24 = np.flatnonzero(ips == "192.168.1.24")
    assert len(r23) == 8 and np.allclose(b.w[r23], 5.0)
    assert sorted(b.get("net.bytes_up", int(i)) for i in r23) == [1000.0 + 100 * k for k in range(8)]
    assert all((b.flags[r23] & EV.FLAG_APPROX) == 0)
    assert len(r24) == 1 and b.w[r24[0]] == 40.0 and b.flags[r24[0]] & EV.FLAG_APPROX
    assert b.get("net.bytes_up", int(r24[0])) == pytest.approx(1400.0)
    assert b.ts[r23].tolist() == [T0 + k for k in range(8)]


def test_new_meta_key_becomes_attribute_and_pseudo_active_skipped():
    """(d) a new extra['meta'] key appears with no code change; (f) pseudo
    entities and active probes are skipped."""
    st = make_store()
    o = [_login("192.168.1.21", "jack", T0, extra={"meta": {"waf": {"score": 3}}}),
         obs("oa", "__system__", T0, http_method="GET", http_path="/"),
         obs("oa", "10.0.0.9", T0, method=AcquisitionMethod.ACTIVE_PROBE, rtt_ms=3.0)]
    eng, n = _run(st, o, with_r2=False)
    b = st.batch_at("oa", EV.EVT_BATCH, T0 + 60)
    assert n == 1 and b.get("meta.waf.score", 0) == 3.0
    assert eng.last_stats["skipped"] == 2
    assert b.get("http.route", 0) is EV.ABSENT                        # no R2 templater yet


def test_trusted_proxy_resolution_through_engine():
    """(g)"""
    st = make_store()
    o = _login("192.168.0.5", "jack", T0)
    o.extra["l7"]["headers"]["x-forwarded-for"] = "10.1.2.3, 192.168.0.9"
    _run(st, [o], config={"progressive": {"enabled": True, "trusted_proxies": ["192.168.0.0/24"]}})
    b = st.batch_at("oa", EV.EVT_BATCH, T0 + 60)
    assert b.get("net.src", 0) == "10.1.2.3" and b.get("net.peer_src", 0) == "192.168.0.5"
    assert b.ips == ["10.1.2.3"]


def test_sample_rate_scales_mass_not_sampling():
    """(i) extra.sample_rate = 10 multiplies mass by 10, pi / learn unchanged."""
    st = make_store()
    o = [_login("192.168.1.21", "jack", T0 + i, extra={"sample_rate": 10}) for i in range(3)]
    _run(st, o)
    b = st.batch_at("oa", EV.EVT_BATCH, T0 + 60)
    assert np.allclose(b.w, 10.0) and (b.pi == 1).all() and b.learn.all()
    assert b.mass().sum() == pytest.approx(30.0)


def test_learning_sample_bounded_and_rare_strata_full():
    """(e) at engine level: E_learn = e_rate x dt; the rare route is learned in full."""
    st = make_store()
    o = [obs("portal", f"10.0.{i % 200}.{i % 250}", T0 + i * 0.01, peer="10.9.9.9", dst_port=443,
             http_method="GET", http_host="portal", http_path="/home", http_status=200)
         for i in range(3000)]
    o += [obs("portal", "10.0.0.7", T0 + 5, peer="10.9.9.9", dst_port=443, http_method="POST",
              http_host="portal", http_path="/admin/export", http_status=200)]
    _run(st, o, config={"progressive": {"enabled": True, "defaults": {"e_rate": 5.0}}})
    b = st.batch_at("portal", EV.EVT_BATCH, T0 + 60)
    assert b.n == 3001
    assert b.learn.sum() <= 5.0 * 60 * 1.3
    rare = [i for i in range(b.n) if b.get("http.method", i) == "POST"]
    assert b.learn[rare].all() and b.pi[rare][0] == 1.0
    assert b.mass().sum() == pytest.approx(3001, rel=0.08)


def test_r_tick_cap_keeps_a_row_per_record_and_mass():
    """(j) a tick above R_tick rows keeps >= 1 row per record, mass-preserving."""
    st = make_store()
    recs = []
    for k in range(30):
        rows = [{"o": float(i), "up": 100 + i, "down": 10} for i in range(64)]
        recs.append(obs("s", f"10.0.0.{k}", T0, peer="10.9.9.9", dst_port=80, http_method="GET",
                        http_host="h", http_path=f"/p{k}", extra={"count": 640, "ev_sample": rows}))
    _run(st, recs, config={"progressive": {"enabled": True, "defaults": {"r_tick": 300}}},
         with_r2=False)
    b = st.batch_at("s", EV.EVT_BATCH, T0 + 60)
    assert b.n <= 300 + 30
    per = {}
    for i in range(b.n):
        per[b.ip_of(i)] = per.get(b.ip_of(i), 0.0) + float(b.w[i])
    assert len(per) == 30 and all(v == pytest.approx(640.0) for v in per.values())


def test_ev_sample_row_meta_becomes_attributes():
    """Aggregated records carry per-event adapter / WAF fields on their
    ev_sample rows (§5.1.2 'meta'): each row's meta.* is an attribute of that
    row's event (before the fix they were dropped, so a WAF score or a new
    adapter field was invisible in aggregated mode)."""
    st = make_store()
    rows = [{"o": float(i), "up": 1000, "down": 300, "st": 200,
             **({"meta": {"waf.score": i % 3, "f007": "x"}} if i % 2 == 0 else {})} for i in range(6)]
    agg = obs("oa", "192.168.1.23", T0, peer="192.168.100.100", dst_port=8080,
              http_method="GET", http_host="oa.local", http_path="/doc/list", http_status=200,
              bytes_up=1000, bytes_down=300,
              extra={"count": 6, "bytes_up_total": 6000, "bytes_down_total": 1800, "ev_sample": rows})
    _run(st, [agg])
    b = st.batch_at("oa", EV.EVT_BATCH, T0 + 60)
    assert "meta.waf.score" in b.names() and "meta.f007" in b.names()
    vals = [b.get("meta.waf.score", i) for i in range(b.n)]
    assert sorted(v for v in vals if v is not EV.ABSENT) == [0, 1, 2]
    assert sum(v is EV.ABSENT for v in vals) == 3
