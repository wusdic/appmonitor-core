"""The behavioural feature contract.

This ordered list is the single shared vocabulary that ties the behaviour
engines together. FeatureVectorEngine builds a vector in this order;
Baseline/Anomaly/Fingerprint/Drift/Clustering all index into the same
positions. Changing behaviour features is a data edit here, not code changes
across five engines — that is what keeps them decoupled.

Each entry: (feature_name, source_metric, log_scale)
  source_metric may be a raw or derived metric name; the store snapshot merges
  both. log_scale=True applies log1p to compress heavy-tailed volume metrics.
"""
from __future__ import annotations

from typing import List, Tuple

FEATURE_SPEC: List[Tuple[str, str, bool]] = [
    # volume & directionality
    ("bytes_up", "l4.bytes_up", True),
    ("bytes_down", "l4.bytes_down", True),
    ("updown_ratio", "l4.updown_ratio", False),
    ("flows", "l4.flows", True),
    ("bytes_per_flow", "derived.bytes_per_flow", True),
    # breadth / graph
    ("distinct_peers", "l4.distinct_peers", True),
    ("distinct_dports", "l4.distinct_dports", True),
    ("fanout", "derived.fanout", True),
    ("peer_novelty", "derived.peer_novelty", False),
    ("dest_concentration", "derived.dest_concentration", False),
    # application workload
    ("http_requests", "http.requests", True),
    ("http_write_ratio", "http.write_ratio", False),
    ("http_error_rate", "derived.http_error_rate", False),
    ("http_5xx_rate", "derived.http_5xx_rate", False),
    ("http_latency", "http.latency_ms_avg", True),
    ("resp_bytes_avg", "http.resp_bytes_avg", True),
    ("distinct_paths", "http.distinct_paths", True),
    ("path_entropy", "derived.path_entropy", False),
    # name resolution
    ("dns_queries", "dns.queries", True),
    ("dns_name_entropy", "derived.dns_name_entropy", False),
    ("dns_dga_score", "derived.dns_dga_score", False),
    ("dns_fail_rate", "derived.dns_fail_rate", False),
    # encrypted-session posture
    ("tls_handshakes", "tls.handshakes", True),
    ("sni_entropy", "derived.sni_entropy", False),
    ("tls_weak_ratio", "tls.weak_version_ratio", False),
    # timing & rhythm
    ("periodicity", "derived.periodicity_score", False),
    ("timing_regularity", "derived.timing_regularity", False),
    ("think_time", "derived.think_time_s_avg", True),
    ("req_per_session", "derived.req_per_session", True),
    ("duty_cycle", "derived.activity_duty_cycle", False),
    # transport health
    ("retransmit_rate", "l4.retransmit_rate", False),
    ("rtt", "l4.rtt_ms_avg", True),
    # reachability (active)
    ("probe_reachable", "probe.reachable", False),
    ("probe_loss", "probe.loss_ratio", False),
]

FEATURE_NAMES: List[str] = [f[0] for f in FEATURE_SPEC]
FEATURE_DIM: int = len(FEATURE_SPEC)
