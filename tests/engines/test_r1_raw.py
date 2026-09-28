"""R1 raw engines (l2l3 / l4flow / http / tls / dns / active_probe), lib-3 upgrade.

Spec: docs/lib3/engines.md '## R1'. Covers the four spec assertions (a)-(d)
plus weighting, full sets with '__other__', eTLD+1 sets, freshness /
presence, NaN inputs, training, cadence switches, FEATURE_SPEC source
coverage and a generous perf bound.
"""
import math
import random
import time

import pytest

from helpers import T0, DT, make_store, run_engine

from app.engines.behavior.lib import features as F
from app.engines.raw.active_probe import ActiveProbeEngine
from app.engines.raw.dns import DNSEngine
from app.engines.raw.http import HTTPEngine
from app.engines.raw.l2l3 import L2L3Engine
from app.engines.raw.l4flow import L4FlowEngine, peer_key
from app.engines.raw.tls import TLSEngine
from app.models.schema import AcquisitionMethod, Observation, Reachability

S, E = "erp", "10.20.1.5"
ENGINES = (L2L3Engine, L4FlowEngine, HTTPEngine, TLSEngine, DNSEngine, ActiveProbeEngine)


# ------------------------------------------------------------------ builders
def http_obs(path="/home", method="GET", status=200, *, entity=E, up=400, down=4000,
             dur=50.0, ts=T0, **kw):
    return Observation(ts=ts, system=S, entity=entity, peer="10.0.1.10", l3_proto="ip",
                       l4_proto="tcp", dst_port=443, bytes_up=up, bytes_down=down, pkts_up=2,
                       pkts_down=4, rtt_ms=10.0, win_size=64240, duration_ms=dur,
                       app_proto="http", http_method=method, http_host="erp.corp.local",
                       http_path=path, http_status=status, user_agent="Mozilla/5.0",
                       content_type="text/html", **kw)


def dns_obs(qname="erp.corp.local", qtype="A", rcode="NOERROR", *, entity=E, ts=T0, **kw):
    return Observation(ts=ts, system=S, entity=entity, peer="10.0.0.53", l4_proto="udp",
                       dst_port=53, app_proto="dns", dns_qname=qname, dns_qtype=qtype,
                       dns_rcode=rcode, bytes_up=80, bytes_down=120, pkts_up=1, pkts_down=1,
                       **kw)


def tls_obs(sni="api.example.com", version="TLS1.3", *, entity=E, dur=30.0, ts=T0, **kw):
    return Observation(ts=ts, system=S, entity=entity, peer="203.0.113.7", l4_proto="tcp",
                       dst_port=443, app_proto="tls", tls_version=version, tls_sni=sni,
                       tls_cipher="TLS_AES_128_GCM_SHA256", ja3="771,4865-4866,0-11-10",
                       ja3s="771,4865", bytes_up=900, bytes_down=5000, pkts_up=4, pkts_down=6,
                       duration_ms=dur, **kw)


def flow_obs(dport=443, peer="10.0.1.10", *, entity=E, ts=T0, **kw):
    kw.setdefault("l4_proto", "tcp")
    return Observation(ts=ts, system=S, entity=entity, peer=peer, dst_port=dport,
                       bytes_up=kw.pop("bytes_up", 100), bytes_down=kw.pop("bytes_down", 200),
                       pkts_up=kw.pop("pkts_up", 1), pkts_down=kw.pop("pkts_down", 1), **kw)


def probe_obs(reach=Reachability.REACHABLE, *, entity=E, ts=T0, rtt=5.0, **kw):
    return Observation(ts=ts, system=S, entity=entity, peer=entity,
                       method=AcquisitionMethod.ACTIVE_PROBE, reachability=reach, rtt_ms=rtt,
                       hop_count=4, open_ports=(443, 80) if reach == Reachability.REACHABLE else (),
                       **kw)


def run_all(store, obs, now=T0, dt=DT, training=False, scheduled=False, engines=None):
    engines = engines or [cls() for cls in ENGINES]
    n = sum(run_engine(e, store, now, training=training, dt=dt, observations=obs,
                       scheduled=scheduled) for e in engines)
    return n, engines


def val(store, name, entity=E, now=T0):
    """Value of raw `name` written exactly at `now` (None if absent/stale)."""
    m = store.latest_raw(S, entity, name)
    return m.value if m is not None and m.ts == now else None


def rich_tick(entity=E, ts=T0):
    return ([http_obs("/a", entity=entity, ts=ts), http_obs("/b", "POST", 302, entity=entity, ts=ts),
             http_obs("/c", "PUT", 404, entity=entity, ts=ts), http_obs("/d", status=503, entity=entity, ts=ts),
             dns_obs(entity=entity, ts=ts), dns_obs("x.tun.example.org", "TXT", "NXDOMAIN", entity=entity, ts=ts),
             tls_obs(entity=entity, ts=ts), tls_obs("old.example.com", "TLS1.0", entity=entity, ts=ts),
             flow_obs(22, tcp_flags="SYN", retransmits=1, rtt_ms=3.0, duration_ms=20.0, ttl=64,
                      entity=entity, ts=ts),
             probe_obs(entity=entity, ts=ts)])


# ------------------------------------------------------------- spec (a)-(d)
def test_a_full_path_set_keeps_singleton_and_sums_to_total():
    st = make_store()
    obs = [http_obs("/p0")]                                   # the path used once
    obs += [http_obs(f"/p{1 + i % 39}") for i in range(99)]   # 39 more paths, 99 requests
    run_engine(HTTPEngine(), st, T0, observations=obs)
    top = val(st, "http.top_paths")
    assert "/p0" in top and top["/p0"] == 1
    assert len([k for k in top if k != "__other__"]) == 40
    assert sum(top.values()) == 100
    assert val(st, "http.requests") == 100.0
    assert val(st, "http.distinct_paths") == 40.0


def test_a_truncated_set_carries_remainder_in_other():
    st = make_store()
    obs = [http_obs(f"/p{i}") for i in range(100)] + [http_obs("/hot") for _ in range(50)]
    run_engine(HTTPEngine(), st, T0, observations=obs)
    top = val(st, "http.top_paths")
    assert len(top) == 65 and top["/hot"] == 50
    assert top["__other__"] == 150 - sum(v for k, v in top.items() if k != "__other__")
    assert sum(top.values()) == 150
    assert val(st, "http.distinct_paths") == 101.0           # exact, not truncated


def test_b_aggregated_record_counts_its_weight():
    st = make_store()
    o = http_obs("/orders/view/{id}", up=10, down=10, dur=40.0,
                 extra={"count": 50, "bytes_up_total": 25000, "bytes_down_total": 500000,
                        "ts_sample": [1.0, 2.0]})
    run_engine(HTTPEngine(), st, T0, observations=[o, http_obs("/other", dur=100.0)])
    assert val(st, "http.requests") == 51.0
    assert val(st, "http.top_paths")["/orders/view/{id}"] == 50
    assert val(st, "http.get_count") == 51.0
    assert val(st, "http.status_2xx") == 51.0
    assert val(st, "http.methods") == {"GET": 51}
    assert val(st, "http.req_bytes_avg") == pytest.approx((25000 + 400) / 51)
    assert val(st, "http.resp_bytes_avg") == pytest.approx((500000 + 4000) / 51)
    assert val(st, "http.latency_ms_avg") == pytest.approx((50 * 40 + 100) / 51)


def test_c_pseudo_entity_is_dropped_and_counted():
    st = make_store()
    bad = [o for ent in ("__system__", "class:web", "class:static:dmz") for o in rich_tick(ent)]
    before = (st.systems(), st.entities(S), st.entities(S, include_pseudo=True))
    n, engines = run_all(st, bad, scheduled=True)
    assert n == 0
    assert (st.systems(), st.entities(S), st.entities(S, include_pseudo=True)) == before
    for ent in ("__system__", "class:web"):
        assert st.raw_names(S, ent) == []
    health = st.health()
    for e in engines:
        assert health[e.name]["ok"] and health[e.name]["dropped_pseudo"] > 0
    assert health["raw.http"]["dropped_pseudo"] == 3 * 4
    assert health["raw.active_probe"]["dropped_pseudo"] == 3


def test_c_pseudo_system_and_empty_keys_are_dropped():
    st = make_store()
    obs = [http_obs(), Observation(ts=T0, system="__org__", entity="10.0.0.9", http_method="GET",
                                   app_proto="http", http_path="/x"),
           Observation(ts=T0, system=S, entity="", http_method="GET", app_proto="http",
                       http_path="/x")]
    eng = HTTPEngine()
    run_engine(eng, st, T0, observations=obs)
    assert st.systems() == [S] and st.entities(S) == [E]
    assert eng.dropped_pseudo == 1 and eng.dropped_invalid == 1
    assert eng.health_record()["dropped_invalid"] == 1


def test_d_dport_set_contains_every_port():
    st = make_store()
    ports = [22, 53, 80, 443, 3306, 8080, 5432] + list(range(10000, 10030))
    obs = [flow_obs(p) for p in ports] + [flow_obs(443), flow_obs(443)]
    run_engine(L4FlowEngine(), st, T0, observations=obs)
    ds = val(st, "l4.dport_set")
    assert {str(p) for p in ports} <= set(ds)
    assert "__other__" not in ds and ds["443"] == 3
    assert sum(ds.values()) == len(obs) and val(st, "l4.distinct_dports") == len(ports)


# ------------------------------------------------------------------ l4flow
def test_l4_scan_truncates_to_64_plus_other_summing_to_flows():
    st = make_store()
    obs = [flow_obs(1000 + i, peer=f"10.30.{i % 5}.{i}", tcp_flags="SYN") for i in range(200)]
    run_engine(L4FlowEngine(), st, T0, observations=obs)
    ds, ps = val(st, "l4.dport_set"), val(st, "l4.peer_set")
    assert len(ds) == 65 and sum(ds.values()) == 200
    assert ps == {f"10.30.{k}.0/24": 40 for k in range(5)}
    assert val(st, "l4.syn_count") == 200.0 and val(st, "l4.flows") == 200.0
    assert val(st, "l4.distinct_peers") == 200.0


def test_l4_weighted_averages_totals_and_new_metrics():
    st = make_store()
    obs = [flow_obs(443, rtt_ms=10.0, duration_ms=100.0, win_size=1000, retransmits=1,
                    pkts_up=3, pkts_down=5),
           flow_obs(443, rtt_ms=20.0, duration_ms=300.0, win_size=3000, retransmits=0,
                    pkts_up=1, pkts_down=1, tcp_flags="SYN",
                    extra={"count": 3, "bytes_up_total": 1500, "bytes_down_total": 9000})]
    run_engine(L4FlowEngine(), st, T0, observations=obs)
    run_engine(L2L3Engine(), st, T0, observations=obs)
    assert val(st, "l4.flows") == 4.0
    assert val(st, "l4.syn_count") == 3.0
    assert val(st, "l4.pkts_total") == 8 + 3 * 2
    assert val(st, "l4.retransmit_rate") == pytest.approx(1 / 14)
    assert val(st, "l4.rtt_ms_avg") == pytest.approx((10 + 3 * 20) / 4)
    assert val(st, "l4.win_size_avg") == pytest.approx((1000 + 3 * 3000) / 4)
    assert val(st, "l4.flow_duration_ms_avg") == pytest.approx((100 + 3 * 300) / 4)
    assert val(st, "l4.bytes_up") == 100 + 1500 and val(st, "l4.bytes_down") == 200 + 9000
    assert val(st, "l3.bytes_total") == val(st, "l4.bytes_up") + val(st, "l4.bytes_down")
    assert val(st, "l3.pkts_total") == val(st, "l4.pkts_total")
    assert val(st, "l4.dport_set") == {"443": 4}
    assert val(st, "l3.proto.tcp_ratio") == 1.0


def test_l4_aggregate_retransmit_total_is_not_multiplied_by_the_weight():
    st = make_store()
    obs = [flow_obs(443, retransmits=1, pkts_up=3, pkts_down=5),
           flow_obs(443, retransmits=0, pkts_up=1, pkts_down=1,
                    extra={"count": 50, "bytes_up_total": 5000, "bytes_down_total": 5000,
                           "retransmits_total": 4})]
    run_engine(L4FlowEngine(), st, T0, observations=obs)
    assert val(st, "l4.pkts_total") == 8 + 50 * 2
    assert val(st, "l4.retransmit_rate") == pytest.approx((1 + 4) / 108)


def test_peer_key_buckets():
    assert peer_key("10.1.2.3") == "10.1.2.0/24"
    assert peer_key("10.1.2.3:8443") == "10.1.2.0/24"
    assert peer_key("2001:db8::1") == "2001:db8::/64"
    assert peer_key("API.Corp.Local.") == "api.corp.local"
    assert peer_key("") == ""


def test_active_records_are_not_entity_traffic():
    st = make_store()
    run_all(st, [probe_obs(), flow_obs(443)])
    assert val(st, "l4.flows") == 1.0 and val(st, "l3.pkts_total") == 2.0
    assert val(st, "l4.distinct_peers") == 1.0


# -------------------------------------------------------------- http / dns / tls
def test_http_counts_ratios_and_status_classes():
    st = make_store()
    obs = [http_obs("/a"), http_obs("/b", "POST", 302), http_obs("/c", "put", 404),
           http_obs("/d", "DELETE", 503), http_obs("/e", "HEAD", 0)]
    run_engine(HTTPEngine(), st, T0, observations=obs)
    got = {k: val(st, f"http.{k}") for k in ("requests", "status_2xx", "status_3xx", "status_4xx",
                                             "status_5xx", "get_count", "write_count",
                                             "get_ratio", "post_ratio", "write_ratio")}
    assert got == {"requests": 5.0, "status_2xx": 1.0, "status_3xx": 1.0, "status_4xx": 1.0,
                   "status_5xx": 1.0, "get_count": 1.0, "write_count": 3.0,
                   "get_ratio": 0.2, "post_ratio": 0.2, "write_ratio": 0.6}
    assert val(st, "http.methods") == {"GET": 1, "POST": 1, "PUT": 1, "DELETE": 1, "HEAD": 1}


def test_dns_counts_sets_and_etld1():
    st = make_store()
    obs = ([dns_obs("A1b2C3.tun.Example.org.", "txt", "NXDOMAIN"),
            dns_obs("zz99.tun.example.org", "TXT", "SERVFAIL")]
           + [dns_obs("mail.corp.com.cn") for _ in range(3)] + [dns_obs("x.bbc.co.uk")])
    run_engine(DNSEngine(), st, T0, observations=obs)
    assert val(st, "dns.queries") == 6.0
    assert val(st, "dns.txt_count") == 2.0 and val(st, "dns.txt_ratio") == pytest.approx(2 / 6)
    assert val(st, "dns.nxdomain_ratio") == pytest.approx(2 / 6)
    qs = val(st, "dns.qname_set")
    assert qs["a1b2c3.tun.example.org"] == 1 and qs["mail.corp.com.cn"] == 3
    assert val(st, "dns.qname_etld1_set") == {"example.org": 2, "corp.com.cn": 3, "bbc.co.uk": 1}
    assert val(st, "dns.qtype_set") == {"TXT": 2, "A": 4}
    lens = [22, 20, 16, 16, 16, 11]
    assert val(st, "dns.qname_len_avg") == pytest.approx(sum(lens) / 6)
    assert val(st, "dns.avg_qname_len") == val(st, "dns.qname_len_avg")
    assert val(st, "dns.distinct_qnames") == 4.0


def test_dns_many_names_sum_to_total():
    st = make_store()
    obs = [dns_obs(f"n{i}.dga.example.net") for i in range(300)]
    run_engine(DNSEngine(), st, T0, observations=obs)
    qs = val(st, "dns.qname_set")
    assert len(qs) == 65 and sum(qs.values()) == 300
    assert val(st, "dns.qname_etld1_set") == {"example.net": 300}
    assert val(st, "dns.distinct_qnames") == 300.0


def test_tls_sets_weak_ratio_and_handshake_time():
    st = make_store()
    obs = [tls_obs("a.cdn.example.com", dur=20.0), tls_obs("b.cdn.example.com", dur=40.0),
           tls_obs("x.example.co.uk", "TLSv1.0", dur=0.0, extra={"handshake_ms": 90.0}),
           tls_obs("y.example.co.uk", "SSLv3", dur=float("nan")),
           # HTTP over TLS: duration is request latency, not a handshake sample
           http_obs("/", tls_version="TLS1.3", tls_sni="erp.corp.local", dur=900.0)]
    run_engine(TLSEngine(), st, T0, observations=obs)
    assert val(st, "tls.handshakes") == 5.0
    assert val(st, "tls.weak_version_ratio") == pytest.approx(2 / 5)
    assert val(st, "tls.handshake_ms_avg") == pytest.approx((20 + 40 + 90) / 3)
    assert val(st, "tls.sni_etld1_set") == {"example.com": 2, "example.co.uk": 2,
                                            "corp.local": 1}
    assert len(val(st, "tls.sni_set")) == 5 and val(st, "tls.distinct_sni") == 5.0
    assert val(st, "tls.ja3_set") == {"771,4865-4866,0-11-10": 4}


def test_tls_handshake_average_absent_without_samples():
    st = make_store()
    run_engine(TLSEngine(), st, T0, observations=[tls_obs(dur=0.0)])
    assert val(st, "tls.handshakes") == 1.0 and val(st, "tls.handshake_ms_avg") is None


# --------------------------------------------------------------- active probe
def test_probe_counts_state_and_does_not_touch_presence():
    st = make_store()
    obs = [probe_obs(ts=T0 - 50), probe_obs(Reachability.TIMEOUT, ts=T0 - 10, rtt=0.0),
           probe_obs(ts=T0 - 30, extra={"count": 2})]
    run_engine(ActiveProbeEngine(), st, T0, observations=obs)
    assert val(st, "probe.probes") == 4.0
    assert val(st, "probe.loss_ratio") == pytest.approx(1 / 4)
    assert val(st, "probe.reachable") == pytest.approx(3 / 4)
    assert val(st, "probe.state") == "timeout"                   # latest by obs.ts
    assert val(st, "probe.rtt_ms") == pytest.approx(5.0)
    assert val(st, "probe.open_ports") == 2.0
    # a probe is our action, not the entity's: presence is not refreshed
    assert st.entities(S) == [E] and st.last_seen(S, E) is None


# ------------------------------------------------------------ edge cases
def test_empty_input_writes_nothing():
    st = make_store()
    for obs in (None, []):
        n, engines = run_all(st, obs)
        assert n == 0
    assert st.systems() == [] and st.entities(S) == []


def test_silent_entity_gets_no_fresh_metrics_and_keeps_last_seen():
    st = make_store()
    run_all(st, rich_tick(E) + rich_tick("10.20.1.6"))
    assert st.last_seen(S, E) == T0
    t1 = T0 + DT
    run_all(st, rich_tick("10.20.1.6", ts=t1), now=t1)
    assert st.last_seen(S, E) == T0 and st.last_seen(S, "10.20.1.6") == t1
    for name in st.raw_names(S, E):
        assert st.latest_raw(S, E, name).ts == T0            # nothing re-stamped
        assert st.latest_fresh(S, E, name, t1) is None


def test_every_metric_is_stamped_now_and_declared_in_produces():
    st = make_store()
    obs = rich_tick() + [Observation(ts=T0 - 400, system=S, entity=E, l3_proto="ipv6",
                                     l4_proto="udp", dst_port=123, peer="2001:db8::5")]
    _, engines = run_all(st, obs)
    produced = {p for e in engines for p in e.produces}
    names = st.raw_names(S, E)
    assert set(names) <= produced
    for name in names:
        assert st.latest_raw(S, E, name).ts == T0
    assert st.last_seen(S, E) == T0 and st.first_seen(S, E) == T0


def test_nan_and_garbage_inputs_never_produce_nan():
    st = make_store()
    nan, inf = float("nan"), float("inf")
    obs = [http_obs(dur=nan, up=nan, down=-5), http_obs(status=nan, dur=inf),
           flow_obs(dport=nan, rtt_ms=nan, duration_ms=nan, win_size=nan, retransmits=nan,
                    pkts_up=nan, ttl=nan, bytes_up=nan),
           flow_obs(443, extra={"count": nan, "bytes_up_total": nan, "bytes_down_total": -1}),
           flow_obs(443, extra={"count": "junk"}), flow_obs(443, extra={"count": 0}),
           flow_obs(443, extra={"count": -3}), flow_obs(443, extra={"count": "4"}),
           dns_obs(extra={"count": 2.9}), tls_obs(dur=nan, extra={"handshake_ms": "x"}),
           probe_obs(rtt=nan, extra={"count": None})]
    run_all(st, obs)
    for name in st.raw_names(S, E):
        v = st.latest_raw(S, E, name).value
        vals = list(v.values()) if isinstance(v, dict) else [v]
        for x in vals:
            if not isinstance(x, str):
                assert math.isfinite(x), (name, v)
    # flows: 2 http + nan-field flow 1 + count nan -> 1 + 'junk' -> 1 + (0, -3 -> nothing)
    # + '4' -> 4 + dns count 2.9 -> 2 + tls 1 (the probe is not traffic)
    assert val(st, "l4.flows") == 2 + 1 + 1 + 1 + 4 + 2 + 1
    assert val(st, "dns.queries") == 2.0
    assert val(st, "http.latency_ms_avg") is None               # no valid sample
    assert val(st, "http.requests") == 2.0 and val(st, "http.status_2xx") == 1.0
    assert val(st, "tls.handshake_ms_avg") is None
    assert val(st, "probe.probes") == 1.0 and val(st, "probe.rtt_ms") is None


def test_zero_count_only_entity_emits_nothing():
    st = make_store()
    run_all(st, [http_obs(extra={"count": 0}), flow_obs(extra={"count": 0})])
    assert st.raw_names(S, E) == []


def test_training_mode_emits_same_metrics_and_no_events():
    a, b = make_store(), make_store()
    run_all(a, rich_tick())
    run_all(b, rich_tick(), training=True)
    assert b.events() == [] and b.matches() == []
    assert sorted(a.raw_names(S, E)) == sorted(b.raw_names(S, E))
    for name in a.raw_names(S, E):
        assert a.latest_raw(S, E, name).value == b.latest_raw(S, E, name).value


def test_cadence_switch_900_to_60():
    st = make_store()
    engines = [cls() for cls in ENGINES]
    ticks = [(T0, 900.0), (T0 + 60, 60.0), (T0 + 120, 60.0)]
    for now, dt in ticks:
        run_all(st, rich_tick(ts=now - dt / 2), now=now, dt=dt, engines=engines, scheduled=True)
        assert val(st, "http.requests", now=now) == 4.0           # a count, not a rate
        assert val(st, "l4.flows", now=now) == 9.0
        assert st.last_seen(S, E) == now
    assert [m.ts for m in st.raw_series(S, E, "http.requests")] == [t for t, _ in ticks]


def test_sets_are_independent_of_record_order():
    obs = ([http_obs(f"/p{i}") for i in range(80)] + [http_obs("/p3")] * 3
           + [flow_obs(2000 + i) for i in range(90)])
    shuffled = list(obs)
    random.Random(7).shuffle(shuffled)
    a, b = make_store(), make_store()
    run_all(a, obs)
    run_all(b, shuffled)
    for name in ("http.top_paths", "l4.dport_set"):
        assert val(a, name) == val(b, name)
        assert len(val(a, name)) == 65


def test_every_raw_feature_source_is_emitted():
    """Every lib-1 metric FEATURE_SPEC v2 reads is written by some R1 engine;
    act.* (R2) and derived.* (D0-D2) are the other producers."""
    st = make_store()
    run_all(st, rich_tick())
    raw = {m for m in F.all_source_metrics() if not m.startswith(("act.", "derived."))}
    missing = {m for m in raw if val(st, m) is None}
    assert not missing
    assert raw <= {p for cls in ENGINES for p in cls.produces}


def test_registry_constructs_without_arguments():
    for cls in ENGINES:
        e = cls()
        assert e.layer == "raw" and e.interval == 1 and e.period_s is None
        assert e.consumes == ["<observations>"]


def test_perf_2k_observations():
    rng = random.Random(3)
    obs = []
    for i in range(2000):
        ent = f"10.20.{i % 4}.{i % 40}"
        k = rng.random()
        if k < 0.6:
            obs.append(http_obs(f"/orders/view/{rng.randint(1, 300)}", entity=ent,
                                tls_version="TLS1.3", tls_sni="erp.corp.local"))
        elif k < 0.85:
            obs.append(dns_obs(f"h{rng.randint(1, 60)}.example.com", entity=ent))
        elif k < 0.98:
            obs.append(flow_obs(rng.randint(1, 3000), peer=f"10.9.{rng.randint(0, 9)}.{rng.randint(1, 200)}",
                                entity=ent, tcp_flags="SYN"))
        else:
            obs.append(probe_obs(entity=ent))
    st = make_store()
    engines = [cls() for cls in ENGINES]
    times = []
    for r in range(5):
        t0 = time.perf_counter()
        run_all(st, obs, now=T0 + r * 60, dt=60, engines=engines)
        times.append(time.perf_counter() - t0)
    # measured ~14 ms (v1 was ~14 ms with fewer metrics); generous for CI noise
    assert sorted(times)[2] < 0.15
