"""D1 RatioEngine / EntropyEngine / GraphEngine (engines/derived/{ratio,entropy,graph}.py).

Spec unit tests (docs/lib3/engines.md, D1):
  (a) raw metrics at t0 only, engine run at t0 and t0+900: derived.http_error_rate
      exists at t0 and not at t0+900;
  (b) a zero denominator writes nothing;
  (c) graph: a peer seen at t0 and again at t0 + 31 d counts as new.
Plus: .n / _n exposure companions named as FEATURE_SPEC_V2 n_source expects,
'__other__' handling, full 64-entry sets, empty store, silent entity, NaN
inputs, training mode (no events), cadence 900 -> 60, bounded peer history,
replay clock going backwards, and perf.
"""
from __future__ import annotations

import math
import time

from helpers import DT, T0, add_obs_tick, make_store, run_engine

from app.engines.behavior.lib.features import FEATURE_NSRC, FEATURE_SOURCE
from app.engines.derived.entropy import EntropyEngine
from app.engines.derived.graph import GraphEngine
from app.engines.derived.ratio import RatioEngine

S, E = "sys", "10.0.0.1"
DAY = 86400.0

HTTP = {"http.requests": 100.0, "http.status_2xx": 80.0, "http.status_4xx": 15.0,
        "http.status_5xx": 5.0}
L4 = {"l4.flows": 10.0, "l4.bytes_up": 4000.0, "l4.bytes_down": 16000.0,
      "l4.distinct_peers": 4.0, "l3.bytes_total": 25000.0}
DNS = {"dns.queries": 20.0, "dns.nxdomain_ratio": 0.25}


def _d(st, name, ts, e=E):
    m = st.latest_derived(S, e, name)
    return m.value if (m is not None and m.ts == ts) else None


def _all_derived(st, e=E):
    return {n: st.latest_derived(S, e, n) for n in st.derived_names(S, e)}


# ------------------------------------------------------------------ spec (a)
def test_a_error_rate_only_when_fresh():
    st = make_store()
    add_obs_tick(st, S, E, T0, {**HTTP, **L4, **DNS})
    eng = RatioEngine()
    run_engine(eng, st, T0)
    assert _d(st, "derived.http_error_rate", T0) == 0.15
    run_engine(eng, st, T0 + DT)
    assert _d(st, "derived.http_error_rate", T0 + DT) is None
    # nothing at all was re-stamped at t0+900
    assert all(m.ts == T0 for m in _all_derived(st).values())


def test_a_entropy_and_graph_only_when_fresh():
    st = make_store()
    sets = {"tls.sni_set": {"a.com": 5, "b.com": 5}, "http.top_paths": {"/x": 3, "/y": 7},
            "dns.qname_set": {"a.com": 4, "b.com": 6}, "tls.ja3_set": {"j1": 10},
            "l4.distinct_peers": 2.0}
    add_obs_tick(st, S, E, T0, sets)
    en, gr = EntropyEngine(), GraphEngine()
    run_engine(en, st, T0)
    run_engine(gr, st, T0)
    assert _d(st, "derived.sni_entropy", T0) == 1.0
    assert _d(st, "derived.fanout", T0) == 2.0
    assert _d(st, "derived.new_peer_count", T0) == 2.0
    run_engine(en, st, T0 + DT)
    run_engine(gr, st, T0 + DT)
    assert all(m.ts == T0 for m in _all_derived(st).values())


# ------------------------------------------------------------------ spec (b)
def test_b_zero_denominator_writes_nothing():
    st = make_store()
    add_obs_tick(st, S, E, T0, {"http.requests": 0.0, "http.status_4xx": 0.0,
                                "http.status_5xx": 0.0, "http.status_2xx": 0.0,
                                "l4.bytes_up": 100.0, "l4.bytes_down": 0.0,
                                "dns.queries": 0.0, "dns.nxdomain_ratio": 0.0})
    run_engine(RatioEngine(), st, T0)
    for name in ("derived.http_error_rate", "derived.http_5xx_rate",
                 "derived.http_success_rate", "derived.upload_dominance",
                 "derived.dns_fail_rate"):
        assert st.latest_derived(S, E, name) is None, name
        assert st.latest_derived(S, E, name + ".n") is None, name


def test_ratio_values_and_exposure():
    st = make_store()
    add_obs_tick(st, S, E, T0, {**HTTP, **L4, **DNS})
    n = run_engine(RatioEngine(), st, T0)
    assert n == 14
    exp = {"derived.http_error_rate": (0.15, 100), "derived.http_5xx_rate": (0.05, 100),
           "derived.http_success_rate": (0.8, 100), "derived.upload_dominance": (0.25, 16000),
           "derived.bytes_per_flow": (2500, 10), "derived.req_per_peer": (25, 4),
           "derived.dns_fail_rate": (0.25, 20)}
    for name, (v, nn) in exp.items():
        assert math.isclose(_d(st, name, T0), v), name
        assert _d(st, name + ".n", T0) == nn, name
    # the exposure companions B01 reads exist under exactly these names
    assert FEATURE_NSRC["dns_fail_rate"] == "derived.dns_fail_rate.n"
    assert FEATURE_SOURCE["dns_fail_rate"] == "derived.dns_fail_rate"


def test_ratio_requires_all_inputs_fresh():
    st = make_store()
    add_obs_tick(st, S, E, T0, {"l4.flows": 10.0})
    add_obs_tick(st, S, E, T0 + DT, {"l3.bytes_total": 500.0})   # flows stale
    run_engine(RatioEngine(), st, T0 + DT)
    assert st.latest_derived(S, E, "derived.bytes_per_flow") is None


# ------------------------------------------------------------------ spec (c)
def test_c_peer_seen_again_after_31_days_is_new():
    st = make_store()
    g = GraphEngine()
    add_obs_tick(st, S, E, T0, {"tls.sni_set": {"www.peer.com": 3}})
    run_engine(g, st, T0)
    assert _d(st, "derived.new_peer_count", T0) == 1.0
    t1 = T0 + 1 * DAY
    add_obs_tick(st, S, E, t1, {"tls.sni_set": {"www.peer.com": 3}})
    run_engine(g, st, t1)
    assert _d(st, "derived.new_peer_count", t1) == 0.0
    assert _d(st, "derived.peer_novelty", t1) == 0.0
    t2 = t1 + 31 * DAY
    add_obs_tick(st, S, E, t2, {"tls.sni_set": {"www.peer.com": 3}})
    run_engine(g, st, t2)
    assert _d(st, "derived.new_peer_count", t2) == 1.0
    assert _d(st, "derived.peer_novelty", t2) == 1.0
    # and it is known again right after
    t3 = t2 + DT
    add_obs_tick(st, S, E, t3, {"tls.sni_set": {"www.peer.com": 3}})
    run_engine(g, st, t3)
    assert _d(st, "derived.new_peer_count", t3) == 0.0


def test_graph_history_expires_and_stays_bounded():
    st = make_store()
    g = GraphEngine(max_peers=100)
    add_obs_tick(st, S, E, T0, {"l4.peer_set": {f"10.{i}.0.0/24": 1 for i in range(64)}})
    run_engine(g, st, T0)
    assert len(g.seen_peers(S, E)) == 64
    t = T0 + 31 * DAY
    add_obs_tick(st, S, E, t, {"l4.peer_set": {"192.168.1.0/24": 1}})
    run_engine(g, st, t)
    assert set(g.seen_peers(S, E)) == {"192.168.1.0/24"}      # old ones swept
    for k in range(5):
        tk = t + (k + 1) * DT
        add_obs_tick(st, S, E, tk, {"l4.peer_set": {f"172.{k}.{i}.0/24": 1 for i in range(64)}})
        run_engine(g, st, tk)
        assert len(g.seen_peers(S, E)) <= 100
    # idle entities lose their whole history after the TTL
    t_far = t + 40 * DAY
    add_obs_tick(st, S, "10.9.9.9", t_far, {"l4.distinct_peers": 1.0})
    run_engine(g, st, t_far)
    assert g.seen_peers(S, E) == {}


def test_graph_etld1_other_and_concentration():
    st = make_store()
    g = GraphEngine()
    add_obs_tick(st, S, E, T0, {
        "tls.sni_set": {"a.cdn.example.com": 6, "b.cdn.example.com": 2, "__other__": 2},
        "dns.qname_set": {"x1.example.com": 1, "__other__": 5},
        "l4.distinct_peers": 3.0})
    run_engine(g, st, T0)
    # no *_etld1_set written: full names are mapped to eTLD+1; __other__ is no peer
    assert set(g.seen_peers(S, E)) == {"example.com"}
    assert _d(st, "derived.new_peer_count", T0) == 1.0
    # top named SNI over the true total (incl. __other__)
    assert math.isclose(_d(st, "derived.dest_concentration", T0), 0.6)
    # the etld1 sets are preferred when fresh
    add_obs_tick(st, S, E, T0 + DT, {"tls.sni_etld1_set": {"other.org": 3},
                                     "tls.sni_set": {"www.other.org": 3}})
    run_engine(g, st, T0 + DT)
    assert "other.org" in g.seen_peers(S, E)
    assert _d(st, "derived.fanout", T0 + DT) is None          # distinct_peers stale


# ------------------------------------------------------------------- entropy
def test_entropy_full_sets_other_and_n():
    st = make_store()
    paths = {f"/p{i}": 1 for i in range(64)}
    paths["__other__"] = 936          # a tunnel: thousands of one-off values
    qn = {"k3j9x2q7.evil.com": 3, "aaaa.good.com": 1}
    add_obs_tick(st, S, E, T0, {"http.top_paths": paths, "dns.qname_set": qn,
                                "tls.ja3_set": {"j1": 4, "j2": 1, "__other__": 1},
                                "tls.sni_set": {"only.com": 50}})
    run_engine(EntropyEngine(), st, T0)
    # uniform over the 64 named entries -> max entropy, not "concentrated"
    assert math.isclose(_d(st, "derived.path_entropy", T0), 1.0)
    assert _d(st, "derived.path_entropy_n", T0) == 1000.0
    assert _d(st, "derived.sni_entropy", T0) == 0.0
    assert _d(st, "derived.sni_entropy_n", T0) == 50.0
    assert _d(st, "derived.ja3_diversity", T0) == 3.0
    assert _d(st, "derived.ja3_diversity_n", T0) == 6.0
    assert _d(st, "derived.dns_name_entropy_n", T0) == 4.0
    assert _d(st, "derived.dns_dga_score_n", T0) == 4.0
    exp_dga = (3 * 3.0 + 1 * 0.0) / 4          # 'k3j9x2q7' has 8 distinct chars
    assert math.isclose(_d(st, "derived.dns_dga_score", T0), exp_dga)
    # companions are named as FEATURE_SPEC_V2 expects
    for feat in ("path_entropy", "dns_name_entropy", "sni_entropy"):
        assert FEATURE_NSRC[feat] == FEATURE_SOURCE[feat] + "_n"
        assert st.latest_derived(S, E, FEATURE_NSRC[feat]) is not None


def test_entropy_ignores_only_other_and_bad_counts():
    st = make_store()
    add_obs_tick(st, S, E, T0, {"http.top_paths": {"__other__": 9},
                                "tls.sni_set": {"a.com": math.nan, "b.com": -1, "c.com": 0},
                                "dns.qname_set": {}})
    run_engine(EntropyEngine(), st, T0)
    assert st.derived_names(S, E) == []


# ---------------------------------------------------------------- edge cases
ENGINES = (RatioEngine, EntropyEngine, GraphEngine)


def test_empty_store():
    st = make_store()
    for cls in ENGINES:
        assert run_engine(cls(), st, T0) == 0


def test_silent_entity_gets_nothing():
    st = make_store()
    add_obs_tick(st, S, E, T0, {**HTTP, **L4, **DNS, "tls.sni_set": {"a.com": 5}})
    # at t1 only the zero-filled act.events clock is written (touch=False)
    add_obs_tick(st, S, E, T0 + DT, {"act.events": 0.0}, touch=False)
    add_obs_tick(st, S, "10.0.0.2", T0 + DT, {**HTTP})
    for cls in ENGINES:
        run_engine(cls(), st, T0 + DT)
    assert st.derived_names(S, E) == []
    assert _d(st, "derived.http_error_rate", T0 + DT, e="10.0.0.2") == 0.15


def test_nan_inputs_write_nothing():
    st = make_store()
    add_obs_tick(st, S, E, T0, {"http.requests": math.nan, "http.status_4xx": 3.0,
                                "l4.bytes_up": math.inf, "l4.bytes_down": 10.0,
                                "l4.distinct_peers": math.nan,
                                "dns.queries": 5.0, "dns.nxdomain_ratio": math.nan})
    for cls in ENGINES:
        run_engine(cls(), st, T0)
    names = st.derived_names(S, E)
    assert not any("http_" in n or "dns_fail" in n or "fanout" in n for n in names)
    for n in names:
        assert math.isfinite(st.latest_derived(S, E, n).value)


def test_training_mode_emits_no_events_but_learns():
    st = make_store()
    g = GraphEngine()
    add_obs_tick(st, S, E, T0, {**HTTP, **L4, "tls.sni_set": {"a.com": 5, "b.com": 1}})
    for cls in (RatioEngine, EntropyEngine):
        run_engine(cls(), st, T0, training=True)
    run_engine(g, st, T0, training=True)
    assert st.events() == []
    assert _d(st, "derived.http_error_rate", T0) == 0.15
    assert set(g.seen_peers(S, E)) == {"a.com", "b.com"}


def test_cadence_900_to_60():
    st = make_store()
    engs = [cls() for cls in ENGINES]
    t = T0
    for k, dt in enumerate([DT, DT, 60.0, 60.0, 60.0]):
        t += dt
        scale = dt / DT
        add_obs_tick(st, S, E, t, {"http.requests": 100.0 * scale,
                                   "http.status_4xx": 15.0 * scale,
                                   "tls.sni_set": {"a.com": 10 * scale, "b.com": 10 * scale},
                                   "l4.peer_set": {f"10.0.{k}.0/24": 1}})
        for eng in engs:
            run_engine(eng, st, t, dt=dt)
        m = st.latest_derived(S, E, "derived.http_error_rate")
        assert m.ts == t and m.window_s == int(dt)
        assert math.isclose(m.value, 0.15)                    # cadence invariant
        assert _d(st, "derived.sni_entropy", t) == 1.0
        # a.com + b.com are new once, then one new /24 per tick
        assert _d(st, "derived.new_peer_count", t) == (3.0 if k == 0 else 1.0)


def test_replay_clock_backwards_keeps_peers_known():
    st = make_store()
    g = GraphEngine()
    add_obs_tick(st, S, E, T0 + DT, {"tls.sni_set": {"a.com": 1}})
    run_engine(g, st, T0 + DT)
    add_obs_tick(st, S, E, T0, {"tls.sni_set": {"a.com": 1}})
    run_engine(g, st, T0)
    assert g.seen_peers(S, E)["a.com"] == T0 + DT
    assert _d(st, "derived.new_peer_count", T0) == 0.0


def test_scheduled_run_and_registry_construction():
    st = make_store()
    add_obs_tick(st, S, E, T0, {**HTTP})
    for cls in ENGINES:
        eng = cls()
        assert eng.layer == "derived" and eng.interval == 1
        run_engine(eng, st, T0, scheduled=True)
    assert _d(st, "derived.http_error_rate.n", T0) == 100.0


def test_perf():
    st = make_store()
    n_ent = 200
    engs = [cls() for cls in ENGINES]
    ents = [f"10.1.{i // 250}.{i % 250}" for i in range(n_ent)]

    def tick(t):
        for i, e in enumerate(ents):
            add_obs_tick(st, S, e, t, {
                **HTTP, **L4, **DNS,
                "http.top_paths": {f"/p{j}": j + 1 for j in range(64)},
                "tls.sni_set": {f"h{j}.s{i}.com": j + 1 for j in range(64)},
                "dns.qname_set": {f"q{j}x{i}.net": 1 for j in range(64)},
                "tls.ja3_set": {"j1": 3},
                "l4.peer_set": {f"10.{j}.{i % 200}.0/24": 1 for j in range(64)}})

    tick(T0)
    for eng in engs:
        run_engine(eng, st, T0)
    tick(T0 + DT)
    t0 = time.perf_counter()
    for eng in engs:
        run_engine(eng, st, T0 + DT)
    per_entity_ms = (time.perf_counter() - t0) * 1000.0 / n_ent
    # spec: 2 ms per tick; generous bound per active entity for CI noise
    assert per_entity_ms < 2.0, per_entity_ms
