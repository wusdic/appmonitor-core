"""P00 round 6 (scoring-cost owner): the client behind a trusted proxy is
resolved per ev_sample row of an aggregated record (O-real R1)."""
from __future__ import annotations

from helpers import T0, ctx, make_store, obs

from app.engines.behavior.lib import pevent as EV
from app.engines.raw.event_builder import EventBuilderEngine

CFG = {"progressive": {"enabled": True, "trusted_proxies": ["192.168.100.99/32"]}}


def test_aggregated_proxy_record_resolves_the_client_of_each_ev_sample_row():
    """An aggregated (900-s) record from a trusted proxy carries the forwarding
    headers on its ev_sample rows' l7 views, not on the record: before round 6
    every row's net.src was the proxy's address (O-real: all proxied oa events
    from day 8 came from 192.168.100.99, the A2-A6 actor was the proxy and oa
    lost its per-group patterns)."""
    st = make_store()
    clients = ["192.168.1.21", "192.168.3.7", "192.168.1.21", "10.50.0.4"]
    rows = [{"o": float(i), "up": 900, "down": 300, "st": 200,
             "l7": {"headers": {"x-forwarded-for": c, "user-agent": "Mozilla/5.0"}}}
            for i, c in enumerate(clients)]
    agg = obs("oa", "192.168.100.99", T0, peer="192.168.100.100", dst_port=8080,
              http_method="GET", http_host="oa.local", http_path="/home", http_status=200,
              bytes_up=900, bytes_down=300,
              extra={"count": 12, "bytes_up_total": 10800, "bytes_down_total": 3600,
                     "ev_sample": rows})
    EventBuilderEngine().safe_run(ctx(st, T0 + 60, window_s=60.0, config=CFG), [agg])
    b = st.batch_at("oa", EV.EVT_BATCH, T0 + 60)
    assert b.n == 4
    order = sorted(range(b.n), key=lambda i: float(b.ts[i]))
    assert [b.ip_of(i) for i in order] == clients
    assert [b.get("net.src", i) for i in order] == clients
    assert all(b.get("net.peer_src", i) == "192.168.100.99" for i in order)


def test_an_untrusted_source_keeps_its_own_address_on_every_row():
    st = make_store()
    rows = [{"o": 0.0, "l7": {"headers": {"x-forwarded-for": "10.1.2.3"}}}]
    agg = obs("oa", "192.168.5.5", T0, http_method="GET", http_host="oa.local", http_path="/home",
              http_status=200, extra={"count": 3, "ev_sample": rows})
    EventBuilderEngine().safe_run(ctx(st, T0 + 60, window_s=60.0, config=CFG), [agg])
    b = st.batch_at("oa", EV.EVT_BATCH, T0 + 60)
    assert b.get("net.src", 0) == "192.168.5.5" and b.get("net.peer_src", 0) is EV.ABSENT
