"""features grain additions (spec v2.1, docs/lib3/cadence.md §3-§4)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.engines.behavior.lib import features as F

VALUES = {
    "l4.bytes_up": 12000.0, "l4.bytes_down": 480000.0, "l4.flows": 30.0, "http.requests": 40.0,
    "dns.queries": 12.0, "tls.handshakes": 9.0, "act.events": 61.0, "l3.bytes_total": 520000.0,
    "derived.new_peer_count": 2.0, "http.write_count": 6.0, "http.get_count": 30.0,
    "http.status_4xx": 3.0, "http.status_5xx": 1.0, "http.status_3xx": 2.0,
    "http.latency_ms_avg": 85.0, "http.resp_bytes_avg": 9000.0, "http.req_bytes_avg": 400.0,
    "act.new_template_ratio": 0.1, "derived.dns_dga_score": 2.4,
    "derived.dns_dga_named_n": 12.0, "derived.dns_fail_rate": 0.25,
    "derived.dns_fail_rate.n": 12.0, "dns.txt_count": 1.0, "dns.qname_len_avg": 18.0,
    "tls.weak_version_ratio": 1.0 / 9.0, "tls.handshake_ms_avg": 33.0, "l4.pkts_total": 800.0,
    "l4.retransmit_rate": 0.01, "l4.rtt_ms_avg": 4.2, "l4.syn_count": 25.0,
    "l4.flow_duration_ms_avg": 1500.0, "probe.probes": 2.0, "probe.reachable": 1.0,
    "probe.loss_ratio": 0.0, "derived.sni_entropy": 0.4, "derived.sni_entropy_n": 9.0,
    "derived.path_entropy": 0.7, "derived.path_entropy_n": 40.0,
    "derived.dns_name_entropy": 0.9, "derived.dns_name_entropy_n": 12.0,
    "derived.dest_concentration": 0.5, "derived.periodicity_score": 0.3,
    "derived.timing_regularity": 0.6, "derived.req_per_session": 12.0,
    "derived.activity_duty_cycle": 0.25, "l4.distinct_peers": 7.0, "l4.distinct_dports": 3.0,
    "act.distinct_templates": 11.0, "derived.ja3_diversity": 2.0,
    "derived.think_log_sum": 3.0 * math.log(5.0), "derived.think_gaps": 3.0,
    "derived.think_time_s_avg": 5.0, "feature.active": 1.0,
}
SNI = {"a.example.com": 4.0, "b.example.com": 3.0, "c.example.org": 2.0}
PATHS = {f"/p{i}": 4.0 for i in range(10)}
QN = {"x.corp": 6.0, "y.corp": 4.0, "zz.net": 2.0}


def _get(vals):
    return lambda m: vals.get(m)


def _one_tick(vals, dt):
    S = F.compute_parts(_get(vals), dt)
    maps = {"tls.sni_set": SNI, "http.top_paths": PATHS, "dns.qname_set": QN}
    sets = {"distinct_peers": vals["l4.distinct_peers"], "distinct_dports": vals["l4.distinct_dports"],
            "distinct_templates": vals["act.distinct_templates"],
            "ja3_diversity": vals["derived.ja3_diversity"]}
    span = {"periodicity": vals["derived.periodicity_score"],
            "timing_regularity": vals["derived.timing_regularity"],
            "req_per_session": vals["derived.req_per_session"],
            "duty_cycle": vals["derived.activity_duty_cycle"]}
    return S, F.grain_values(S, dt, dt, sets=sets, maps=maps, span=span)


def test_tables():
    assert F.PART_DIM == 47 and len(F.PART_NAMES) == 47
    classes = [F.GRAIN_CLASS[n] for n in F.FEATURE_NAMES_V2]
    assert classes.count("add") == 40 and classes.count("set") == 4
    assert classes.count("map") == 4 and classes.count("span") == 4
    tr = [F.TRANSFER[n] for n in F.FEATURE_NAMES_V2]
    assert sum(t not in (None, "none") for t in tr) == 40
    assert tr.count("none") == 8 and tr.count(None) == 4
    assert F.TRANSFER["flows"] == "rate" and F.TRANSFER["http_4xx_rate"] == "ratio"
    assert F.TRANSFER["bytes_up"] == "jensen" and F.TRANSFER["updown_log"] == "mean"
    assert all(F.GRAIN_CLASS[n] in ("add", "set") for n in F.KEY_FEATURES)


@pytest.mark.parametrize("dt", [60.0, 900.0, 3600.0])
def test_one_tick_grain_equals_compute_features(dt):
    vals = dict(VALUES)
    vals["derived.sni_entropy"] = F.map_value("sni_entropy", {"tls.sni_set": SNI})
    vals["derived.path_entropy"] = F.map_value("path_entropy", {"http.top_paths": PATHS})
    vals["derived.dns_name_entropy"] = F.map_value("dns_name_entropy", {"dns.qname_set": QN})
    vals["derived.dest_concentration"] = F.map_value("dest_concentration", {"tls.sni_set": SNI})
    vec0, nat0 = F.compute_features(_get(vals), dt)
    S, (vec, nat) = _one_tick(vals, dt)
    skip = {F.FEATURE_INDEX["think_time"]}          # geometric mean vs median by design
    for i in range(F.FEATURE_DIM):
        if i in skip:
            continue
        a, b = vec0[i], vec[i]
        assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, rel=1e-9, abs=1e-12), \
            F.FEATURE_NAMES_V2[i]
        a, b = nat0[i], nat[i]
        assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, rel=1e-9, abs=1e-12), \
            F.FEATURE_NAMES_V2[i]
    assert nat[F.FEATURE_INDEX["think_time"]] == pytest.approx(5.0)


def test_parts_are_additive_over_ticks():
    """Two ticks summed == one tick holding their union (avg features are
    the n-weighted mean of the per-tick means)."""
    a = dict(VALUES)
    b = dict(VALUES, **{"l4.flows": 10.0, "l4.rtt_ms_avg": 8.0, "http.requests": 10.0,
                        "http.latency_ms_avg": 20.0, "http.write_count": 1.0})
    Sa = F.compute_parts(_get(a), 1800.0)
    Sb = F.compute_parts(_get(b), 1800.0)
    vec, nat = F.grain_values(Sa + Sb, 3600.0, 3600.0)
    I = F.FEATURE_INDEX
    assert nat[I["flows"]] == 40.0
    assert nat[I["rtt"]] == pytest.approx((4.2 * 30 + 8.0 * 10) / 40.0)
    assert nat[I["http_latency"]] == pytest.approx((85.0 * 40 + 20.0 * 10) / 50.0)
    assert nat[I["http_write_ratio"]] == pytest.approx(7.0 / 50.0)
    assert vec[I["flows"]] == pytest.approx(math.log1p(40.0 * 60.0 / 3600.0))
    # set / map / span features without inputs are undefined, never 0
    assert np.isnan(nat[I["distinct_peers"]]) and np.isnan(nat[I["sni_entropy"]])
    assert np.isnan(nat[I["duty_cycle"]])


def test_stale_sources_give_zero_parts_and_nan_ratios():
    S = F.compute_parts(lambda m: None, 900.0, active=0.0)
    assert S[0] == 900.0 and not S[1:].any()
    vec, nat = F.grain_values(S, 900.0, 900.0, sets={"distinct_peers": 0.0})
    I = F.FEATURE_INDEX
    assert nat[I["flows"]] == 0.0 and vec[I["flows"]] == 0.0
    assert np.isnan(nat[I["http_4xx_rate"]]) and np.isnan(nat[I["rtt"]])
    assert nat[I["distinct_peers"]] == 0.0
    assert np.isnan(vec[F.CLR_IDX]).all()


def test_coverage_gate_on_set_and_map_features():
    S = F.compute_parts(_get(VALUES), 900.0)
    sets = {"distinct_peers": 5.0}
    maps = {"tls.sni_set": SNI}
    _, nat = F.grain_values(S, 3000.0, 3600.0, sets=sets, maps=maps)
    assert np.isnan(nat[F.FEATURE_INDEX["distinct_peers"]])
    assert np.isnan(nat[F.FEATURE_INDEX["sni_entropy"]])
    _, nat = F.grain_values(S, 3500.0, 3600.0, sets=sets, maps=maps)
    assert nat[F.FEATURE_INDEX["distinct_peers"]] == 5.0
    assert nat[F.FEATURE_INDEX["sni_entropy"]] == pytest.approx(
        F.map_value("sni_entropy", maps))


def test_grain_exposure():
    S = F.compute_parts(_get(VALUES), 900.0)
    ex = F.grain_exposure(S + S)
    assert ex == {"http": 80.0, "dns": 24.0, "tls": 18.0, "flows": 60.0, "probe": 4.0}
