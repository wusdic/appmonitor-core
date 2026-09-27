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
    Label         -> an analyst verdict on an event / incident / entity / class
    Incident      -> the notified unit: merged findings on one entity or class

Schema v2 (lib-3): the high-volume objects are `slots=True` dataclasses and
`dims` / `inputs` default to None instead of a fresh dict/list, because the
store keeps tens of thousands of them per entity and the per-instance
`__dict__` plus two empty containers used to dominate memory. Readers must
treat `dims is None` / `inputs is None` as empty.
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
# Pseudo-entities (contract B)
# --------------------------------------------------------------------------- #
# Series that describe a whole system, the organisation or a class live under
# these entity keys. They are never observed on the wire, so raw ingestion
# rejects them and `store.entities()` hides them.
SYSTEM_ENTITY = "__system__"
ORG = "__org__"                           # used as both system and entity
CLASS_PREFIX = "class:"                   # class:<rid>, class:static:<n>, class:pool:<cidr>


def is_pseudo_entity(entity: str) -> bool:
    """True for '__*' (system/org aggregates) and 'class:*' keys."""
    return entity.startswith("__") or entity.startswith(CLASS_PREFIX)


# Closed vocabularies (contract E). Kept as tuples so they are cheap to check.
EVENT_STATUSES = ("open", "suppressed", "acked", "closed")
INCIDENT_STATUSES = ("open", "acked", "suppressed", "closed")
INCIDENT_CLOSE_REASONS = ("returned", "accepted", "labelled", "timeout")
LABEL_TARGET_TYPES = ("event", "incident", "entity", "class")
LABEL_VERDICTS = ("tp", "fp", "expected_change", "benign_known", "unsure")
LABEL_SCOPES = ("this", "pattern", "entity", "class", "system")


# --------------------------------------------------------------------------- #
# Observations (input to the raw-metric engines)
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
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
@dataclass(slots=True)
class RawMetric:
    """One measured value emitted by a raw-metric engine."""

    name: str                              # e.g. "l4.flow.bytes_up"
    value: Any                             # float | int | str | dict
    ts: float
    system: str
    entity: str
    kind: MetricKind = MetricKind.GAUGE
    method: AcquisitionMethod = AcquisitionMethod.PASSIVE_SPAN
    dims: Optional[Dict[str, Any]] = None  # peer, port, host... (None == {})
    unit: str = ""

    @property
    def key(self) -> str:
        return f"{self.system}|{self.entity}|{self.name}"


@dataclass(slots=True)
class DerivedMetric:
    """A value computed from one or more raw metrics over a window."""

    name: str                              # e.g. "derived.error_rate"
    value: Any
    ts: float
    system: str
    entity: str
    window_s: int
    kind: MetricKind = MetricKind.GAUGE
    inputs: Optional[List[str]] = None     # provenance (None == [])
    dims: Optional[Dict[str, Any]] = None  # e.g. window {span_s, n_active}
    unit: str = ""

    @property
    def key(self) -> str:
        return f"{self.system}|{self.entity}|{self.name}"


# --------------------------------------------------------------------------- #
# Behaviour layer
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
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
    # lib-3: clip(1 - 2*EER_hard, 0, 1) from the cross-validated identity model
    separability: float = 0.0              # how uniquely identifiable (0..1)
    sample_count: int = 0
    # lib-3: n_eff >= 96 and calibration healthy
    stable: bool = False                   # enough samples for baseline
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BehaviorEvent:
    """A behavioural finding: an anomaly, a drift from own fingerprint, an
    unusual action sequence, or (lib-3) any discrete kind from contract F.

    The v2 fields all default, so v1 constructors keep working. `id` is
    assigned by `store.add_event` when left empty; `status` is the only field
    expected to change after insertion (via `store.update_event`)."""

    system: str
    entity: str
    ts: float
    kind: str                              # contract F kinds (legacy: anomaly|drift|sequence)
    score: float                           # 0..1 normalized
    severity: Severity = Severity.INFO
    contributors: List[Tuple[str, float]] = field(default_factory=list)  # feat,z
    description: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    # ---- v2 (contract E)
    id: str = ""
    status: str = "open"                   # open | suppressed | acked | closed
    p_value: Optional[float] = None        # fused / detector p at emission
    e_day: Optional[float] = None          # p * 86400 / dt (expected null ticks per day)
    axes: List[str] = field(default_factory=list)
    p_by_detector: Dict[str, float] = field(default_factory=dict)
    dedupe_key: str = ""
    incident_id: str = ""
    model_version: Optional[int] = None
    window: Optional[Tuple[float, float]] = None   # (t0, t1) the finding covers

    def __post_init__(self) -> None:
        if self.status not in EVENT_STATUSES:
            raise ValueError(f"BehaviorEvent.status {self.status!r} not in {EVENT_STATUSES}")


# --------------------------------------------------------------------------- #
# Signature layer (行为特征库)
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
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


# --------------------------------------------------------------------------- #
# Decision layer (lib-3 v2): analyst labels and incidents
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Label:
    """An analyst verdict. Labels are never pruned: feedback learning (B23)
    and eval replay both need the full history. `scope` widens a verdict from
    this one target to a pattern / entity / class / system; `ttl_s` (None =
    forever) bounds how long a widened verdict applies."""

    id: str = ""
    system: str = ""
    entity: str = ""                       # may be class:<id>
    target_type: str = "event"             # event | incident | entity | class
    target_id: str = ""
    verdict: str = "unsure"                # tp | fp | expected_change | benign_known | unsure
    scope: str = "this"                    # this | pattern | entity | class | system
    t0: Optional[float] = None
    t1: Optional[float] = None
    ttl_s: Optional[float] = None
    analyst: str = ""
    note: str = ""
    ts: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.target_type not in LABEL_TARGET_TYPES:
            raise ValueError(f"Label.target_type {self.target_type!r} not in {LABEL_TARGET_TYPES}")
        if self.verdict not in LABEL_VERDICTS:
            raise ValueError(f"Label.verdict {self.verdict!r} not in {LABEL_VERDICTS}")
        if self.scope not in LABEL_SCOPES:
            raise ValueError(f"Label.scope {self.scope!r} not in {LABEL_SCOPES}")


@dataclass(slots=True)
class Incident:
    """The only thing that notifies (B27). One incident merges the events,
    alarms and matches of an entity (or a class, entity = 'class:<id>') over
    time; it is updated in place via `store.put_incident` (keyed by id)."""

    id: str = ""
    system: str = ""
    entity: str = ""                       # an IP or class:<id>
    entities: List[str] = field(default_factory=list)   # members / aliases involved
    kinds: List[str] = field(default_factory=list)      # event kinds merged in
    axes: List[str] = field(default_factory=list)
    status: str = "open"                   # open | acked | suppressed | closed
    opened: float = 0.0
    last_seen: float = 0.0
    severity: Severity = Severity.LOW
    e_day_min: Optional[float] = None      # most extreme e_day seen
    risk: float = 0.0
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    explanation: Dict[str, Any] = field(default_factory=dict)
    narrative: str = ""
    campaign_id: str = ""
    parent_id: str = ""
    close_reason: Optional[str] = None     # returned | accepted | labelled | timeout

    def __post_init__(self) -> None:
        if self.status not in INCIDENT_STATUSES:
            raise ValueError(f"Incident.status {self.status!r} not in {INCIDENT_STATUSES}")
        if self.close_reason is not None and self.close_reason not in INCIDENT_CLOSE_REASONS:
            raise ValueError(f"Incident.close_reason {self.close_reason!r} "
                             f"not in {INCIDENT_CLOSE_REASONS}")
