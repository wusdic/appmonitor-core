"""spec v2.1 raw / derived grain inputs (docs/lib3/cadence.md §3.2-§3.3):
SetSketch series of R1 / R2, act.slot_events, D1 dns_dga_named_n, D2 think
parts and the slot-grid duty cycle. All of them are canonical-mode only."""
from __future__ import annotations

import math

import pytest

from helpers import make_store, obs, run_engine

from app.engines.behavior.lib import sketch as SK
from app.engines.derived.session import SessionEngine
from app.engines.raw.action_token import ActionTokenEngine
from app.engines.raw.l4flow import L4FlowEngine
from app.engines.raw.tls import TLSEngine

S, E = "erp", "10.1.1.1"
H0 = 1_741_536_000.0
CAN = {"grain_mode": "canonical"}


def _raw(st, name, ts):
    m = st.latest_raw_at(S, E, name, ts)
    return None if m is None else m.value


def test_r1_set_sketches_count_the_full_sets():
    st = make_store()
    now = H0 + 900.0
    os_ = [obs(S, E, H0 + i, peer=f"10.9.0.{i % 300}", dst_port=1000 + i % 40, bytes_up=10,
               bytes_down=20, app_proto="tls", tls_sni=f"h{i % 7}.x", tls_version="TLS1.3",
               ja3=f"j{i % 3}") for i in range(600)]
    run_engine(L4FlowEngine(), st, now, dt=900.0, observations=os_, config=CAN)
    run_engine(TLSEngine(), st, now, dt=900.0, observations=os_, config=CAN)
    assert SK.set_count(_raw(st, "l4.peer_ids", now)) == pytest.approx(
        _raw(st, "l4.distinct_peers", now), rel=0.13)
    assert SK.set_count(_raw(st, "l4.dport_ids", now)) == _raw(st, "l4.distinct_dports", now) == 40
    assert SK.set_count(_raw(st, "tls.ja3_ids", now)) == 3
    st2 = make_store()
    run_engine(L4FlowEngine(), st2, now, dt=900.0, observations=os_)      # tick mode
    assert _raw(st2, "l4.peer_ids", now) is None


def test_r2_slot_events_sum_to_act_events_and_split_aggregates():
    st = make_store()
    now = H0 + 3600.0
    os_ = [obs(S, E, H0 + 60.0 * i, app_proto="http", http_method="GET", http_host="a.corp",
               http_path=f"/p/{i % 5}", http_status=200, peer="10.9.0.1", dst_port=80)
           for i in range(40)]
    # one aggregated record of weight 8 whose samples fall in two slots
    os_.append(obs(S, E, H0 + 1000.0, app_proto="http", http_method="GET", http_host="a.corp",
                   http_path="/agg", http_status=200, peer="10.9.0.1", dst_port=80,
                   extra={"count": 8, "ts_sample": [0.0, 10.0, 1000.0, 1010.0]}))
    run_engine(ActionTokenEngine(), st, now, dt=3600.0, observations=os_, config=CAN)
    se = _raw(st, "act.slot_events", now)
    assert sum(se.values()) == pytest.approx(_raw(st, "act.events", now))
    assert set(se) <= {H0, H0 + 900.0, H0 + 1800.0, H0 + 2700.0}
    # the aggregated record: 4 of its 8 events in slot 1 (1000, 1010), 4 in slot 2
    plain = {k: 0.0 for k in se}
    for i in range(40):
        k = math.floor((H0 + 60.0 * i) / 900.0) * 900.0
        plain[k] += 1.0
    assert se[H0 + 900.0] - plain[H0 + 900.0] == pytest.approx(4.0)
    assert se[H0 + 1800.0] - plain[H0 + 1800.0] == pytest.approx(4.0)
    assert SK.set_count(_raw(st, "act.template_ids", now)) == _raw(st, "act.distinct_templates", now)


def test_d2_slot_duty_is_the_same_at_60_900_and_3600():
    """The same events give the same canonical duty cycle whatever the tick."""
    events = [H0 + 120.0, H0 + 130.0, H0 + 2000.0, H0 + 7300.0, H0 + 7400.0]
    out = {}
    for dt in (60.0, 900.0, 3600.0):
        st = make_store()
        r2, d2 = ActionTokenEngine(), SessionEngine()
        n = int(3 * 3600.0 / dt)
        for k in range(1, n + 1):
            now = H0 + k * dt
            os_ = [obs(S, E, t, app_proto="http", http_method="GET", http_host="a.corp",
                       http_path="/x", http_status=200, peer="10.9.0.1", dst_port=80)
                   for t in events if now - dt <= t < now]
            run_engine(r2, st, now, dt=dt, observations=os_, config=CAN)
            run_engine(d2, st, now, dt=dt, config=CAN)
        m = st.latest_derived(S, E, "derived.activity_duty_cycle")
        out[dt] = float(m.value)
    assert out[60.0] == pytest.approx(out[900.0]) == pytest.approx(out[3600.0])
    # 3 active slots (0: 02:00-02:10, 2: 33:20, 8: 2:01:40-2:03:20) of 12 covered
    assert out[900.0] == pytest.approx(3.0 / 12.0)
