"""Core data schema for the AppMonitor profiling platform.

All engines communicate through these normalized structures. Nothing in an
engine references another engine directly — they only read/write objects of
these types keyed by *metric name*, which is what keeps the engine set
low-coupled and individually replaceable.

Layers of data:
    Observation   -> normalized passive-decode or active-probe event
    RawMetric     -> a single measured value with acquisition provenance
    DerivedMetric -> value computed from raw metrics over a window
    EntityProfile -> per (system, entity) behavioural baseline / fingerprint
    BehaviorEvent -> a detected deviation / behavioural finding
    SignatureMatch-> a preset metric-combination matched to a semantic meaning
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class AcquisitionMethod(str, Enum):
    """How a raw metric was obtained. Every raw metric carries one so an
    operator can tell passive-observed truth from actively-probed inference."""

    PASSIVE_SPAN = "passive_span"          # mirror/SPAN/TAP port, full packet
    PASSIVE_FLOW = "passive_flow"          # NetFlow/IPFIX/sFlow flow records
    PASSIVE_LOG = "passive_log"            # syslog / access log shipped off-host
    ACTIVE_PROBE = "active_probe"          # ICMP / TCP-connect / banner / HTTP
    ACTIVE_DNS = "active_dns"              # our own resolver lookups
    ACTIVE_TLS = "active_tls"              # our own TLS handshake / cert fetch
    DERIVED = "derived"                    # produced by a derived engine
    INFERRED = "inferred"                  # model output (behaviour/signature)


class Reachability(str, Enum):
    """First-class outcome of an active probe so cross-network unreachability
    is data, not a missing value."""

    REACHABLE = "reachable"
    UNREACHABLE = "unreachable"            # ICMP/TCP refused or no route
    TIMEOUT = "timeout"                    # crossed a boundary, no answer
    FILTERED = "filtered"                  # silently dropped (firewall)
    DEGRADED = "degraded"                  # partial / high-loss path


class MetricKind(str, Enum):
    GAUGE = "gauge"
    COUNTER = "counter"
    RATE = "rate"
    DISTRIBUTION = "distribution"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# --------------------------------------------------------------------------- #
# Observations (input to the raw-metric engines)
# --------------------------------------------------------------------------- #
@dataclass
class Observation:
    """A single normalized event. Produced by capture adapters (SPAN decode,
    flow collector, active prober). Engines never touch raw packets — an
    adapter decodes once into this shape, keeping decode concerns out of the
    metric logic."""

    ts: float                              # epoch seconds
    system: str                            # business system id this concerns
    entity: str                            # ip or ip-class token (the "who")
    peer: str = ""                         # the other endpoint (server ip/host)
    method: AcquisitionMethod = AcquisitionMethod.PASSIVE_SPAN
    # protocol stack fields (any subset present depending on decode depth)
    l3_proto: str = ""                     # ip / ipv6
    l4_proto: str = ""                     # tcp / udp / icmp
    src_port: int = 0
    dst_port: int = 0
    bytes_up: int = 0                      # entity -> peer
    bytes_down: int = 0                    # peer -> entity
    pkts_up: int = 0
    pkts_down: int = 0
    ttl: int = 0
    tcp_flags: str = ""                    # e.g. "SYN,ACK"
    win_size: int = 0
    retransmits: int = 0
    rtt_ms: float = 0.0                    # SYN/SYN-ACK or probe RTT
    duration_ms: float = 0.0
    # application layer (optional, present after L7 decode)
    app_proto: str = ""                    # http / dns / tls / smtp ...
    http_method: str = ""
    http_host: str = ""
    http_path: str = ""
    http_status: int = 0
    user_agent: str = ""
    content_type: str = ""
    tls_version: str = ""
    tls_cipher: str = ""
    tls_sni: str = ""
    ja3: str = ""                          # client TLS fingerprint
    ja3s: str = ""                         # server TLS fingerprint
    dns_qname: str = ""
    dns_qtype: str = ""
    dns_rcode: str = ""
    # active-probe outcome
    reachability: Optional[Reachability] = None
    banner: str = ""
    open_ports: Tuple[int, ...] = ()
    hop_count: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@dataclass
class RawMetric:
    """One measured value emitted by a raw-metric engine."""

    name: str                              # e.g. "l4.flow.bytes_up"
    value: Any                             # float | int | str | dict
    ts: float
    system: str
    entity: str
    kind: MetricKind = MetricKind.GAUGE
    method: AcquisitionMethod = AcquisitionMethod.PASSIVE_SPAN
    dims: Dict[str, str] = field(default_factory=dict)  # peer, port, host...
    unit: str = ""

    @property
    def key(self) -> str:
        return f"{self.system}|{self.entity}|{self.name}"


@dataclass
class DerivedMetric:
    """A value computed from one or more raw metrics over a window."""

    name: str                              # e.g. "derived.error_rate"
    value: Any
    ts: float
    system: str
    entity: str
    window_s: int
    kind: MetricKind = MetricKind.GAUGE
    inputs: List[str] = field(default_factory=list)   # provenance
    dims: Dict[str, str] = field(default_factory=dict)
    unit: str = ""

    @property
    def key(self) -> str:
        return f"{self.system}|{self.entity}|{self.name}"


# --------------------------------------------------------------------------- #
# Behaviour layer
# --------------------------------------------------------------------------- #
@dataclass
class EntityProfile:
    """The dynamic per-(system, entity) behavioural picture. This is the
    '行为库' content — never a static list, always generated. Holds the
    baseline statistics, the behavioural fingerprint vector, the archetype
    (user-class) label, and a separability score describing how distinguishable
    this entity is from others."""

    system: str
    entity: str
    updated: float = field(default_factory=time.time)
    feature_names: List[str] = field(default_factory=list)
    fingerprint: List[float] = field(default_factory=list)      # current vector
    baseline_median: List[float] = field(default_factory=list)
    baseline_mad: List[float] = field(default_factory=list)     # robust spread
    seasonal: Dict[str, List[float]] = field(default_factory=dict)  # tod/dow
    archetype: str = ""                    # cluster label / user-class
    archetype_confidence: float = 0.0
    separability: float = 0.0              # how uniquely identifiable (0..1)
    sample_count: int = 0
    stable: bool = False                   # enough samples for baseline
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BehaviorEvent:
    """A behavioural finding: an anomaly, a drift from own fingerprint, or an
    unusual action sequence."""

    system: str
    entity: str
    ts: float
    kind: str                              # anomaly | drift | sequence | class
    score: float                           # 0..1 normalized
    severity: Severity = Severity.INFO
    contributors: List[Tuple[str, float]] = field(default_factory=list)  # feat,z
    description: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Signature layer (行为特征库)
# --------------------------------------------------------------------------- #
@dataclass
class SignatureMatch:
    """A preset metric-combination signature that fired, mapping observed
    metrics to a human-meaningful activity."""

    system: str
    entity: str
    ts: float
    signature_id: str
    label: str                             # what the entity appears to be doing
    category: str                          # browse | api | transfer | admin ...
    confidence: float                      # 0..1
    matched_terms: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    severity: Severity = Severity.INFO
