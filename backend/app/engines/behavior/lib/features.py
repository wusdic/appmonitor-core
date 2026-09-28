"""FEATURE_SPEC v2: the 52-feature per-tick representation (contract A).

This ordered table is the single vocabulary every lib-3 engine indexes into.
FeatureVectorEngine (B01) builds `feature.vec` / `feature.nat` in exactly this
order; baseline, likelihood, multivariate, changepoint, identity ... all use
FEATURE_INDEX / GROUPS instead of hard-coded positions, so changing a feature
is a data edit here, not a code change across engines.

Why per-kind transforms: the vector must be cadence independent and presence
aware. Counts become per-minute rates in log1p space (so 15x the count at
dt=900 equals the count at dt=60), stale counts are a true 0 (absence is
data), while ratios / averages / bounded descriptors of *nothing* are NaN
(undefined, never 0). Ratios carry their exposure n so downstream predictives
(Beta-Binomial) stay exact; bounded descriptors need n >= 5 items to mean
anything.

Each FEATURE_SPEC_V2 entry is ``(name, source, kind, n_source, group)``:

source
    * ``"metric.name"``            -- read the (fresh) metric value directly;
    * ``(num, den)``               -- the formula num / den (2-tuple);
    * ``("logratio1p", a, b)``     -- the formula log((a + 1) / (b + 1)).
    For kind ``ratio`` with a (num, den) source the ratio numerator k = num
    and exposure n = den. For kind ``ratio`` with a plain metric source the
    metric is a fraction and k = fraction * n.
n_source
    ``None`` (no exposure gate beyond freshness), a metric name, or a tuple of
    metric names whose values are summed (e.g. tls.handshakes + dns.queries).
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

Source = Union[str, Tuple[str, str], Tuple[str, str, str]]
NSource = Union[None, str, Tuple[str, ...]]
FeatureEntry = Tuple[str, Source, str, NSource, str]

KINDS: Tuple[str, ...] = ("count", "bytes", "ratio", "avg", "bounded",
                          "gauge", "window", "clr", "ctx")
LOGRATIO = "logratio1p"

# Per-feature transform overrides for vec (default is chosen by kind).
#   updown_log is already a log ratio (can be negative) -> identity, not log(v)
#   req_per_session is a heavy-tailed window count      -> log1p
VEC_TX: Dict[str, str] = {"updown_log": "identity", "req_per_session": "log1p"}

AVG_FLOOR = 1e-3        # log(v) floor for avg features: log(0) would be -inf
BOUNDED_EPS = 1e-3      # clip for bounded logit
BOUNDED_MIN_N = 5       # bounded descriptors need >= 5 items
CLR_PSEUDO = 0.5        # CLR pseudo-count, applied to per-minute rates

_HTTPR = "http.requests"
_DNSQ = "dns.queries"
_TLSH = "tls.handshakes"
_FLOWS = "l4.flows"

FEATURE_SPEC_V2: List[FeatureEntry] = [
    # ---------------------------------------------------------------- volume
    ("bytes_up", "l4.bytes_up", "bytes", None, "volume"),
    ("bytes_down", "l4.bytes_down", "bytes", None, "volume"),
    ("flows", _FLOWS, "count", None, "volume"),
    ("http_requests", _HTTPR, "count", None, "volume"),
    ("dns_queries", _DNSQ, "count", None, "volume"),
    ("tls_handshakes", _TLSH, "count", None, "volume"),
    ("intensity", "act.events", "count", None, "volume"),
    ("bytes_per_flow", ("l3.bytes_total", _FLOWS), "avg", _FLOWS, "volume"),
    ("updown_log", (LOGRATIO, "l4.bytes_up", "l4.bytes_down"), "avg", _FLOWS, "volume"),
    # --------------------------------------------------------------- breadth
    ("distinct_peers", "l4.distinct_peers", "count", None, "breadth"),
    ("distinct_dports", "l4.distinct_dports", "count", None, "breadth"),
    ("distinct_templates", "act.distinct_templates", "count", None, "breadth"),
    ("new_peer_count", "derived.new_peer_count", "count", None, "breadth"),
    ("dest_concentration", "derived.dest_concentration", "bounded", (_TLSH, _DNSQ), "breadth"),
    # ------------------------------------------------------------------- app
    ("http_write_ratio", ("http.write_count", _HTTPR), "ratio", _HTTPR, "app"),
    ("http_get_ratio", ("http.get_count", _HTTPR), "ratio", _HTTPR, "app"),
    ("http_4xx_rate", ("http.status_4xx", _HTTPR), "ratio", _HTTPR, "app"),
    ("http_5xx_rate", ("http.status_5xx", _HTTPR), "ratio", _HTTPR, "app"),
    ("http_3xx_rate", ("http.status_3xx", _HTTPR), "ratio", _HTTPR, "app"),
    ("http_latency", "http.latency_ms_avg", "avg", _HTTPR, "app"),
    ("resp_bytes_avg", "http.resp_bytes_avg", "avg", _HTTPR, "app"),
    ("req_bytes_avg", "http.req_bytes_avg", "avg", _HTTPR, "app"),
    ("path_entropy", "derived.path_entropy", "bounded", "derived.path_entropy_n", "app"),
    ("new_template_ratio", "act.new_template_ratio", "ratio", _HTTPR, "app"),
    # ------------------------------------------------------------------- dns
    ("dns_name_entropy", "derived.dns_name_entropy", "bounded", "derived.dns_name_entropy_n", "dns"),
    ("dns_dga_score", "derived.dns_dga_score", "avg", _DNSQ, "dns"),
    ("dns_fail_rate", "derived.dns_fail_rate", "ratio", "derived.dns_fail_rate.n", "dns"),
    ("dns_txt_ratio", ("dns.txt_count", _DNSQ), "ratio", _DNSQ, "dns"),
    ("dns_qname_len", "dns.qname_len_avg", "avg", _DNSQ, "dns"),
    # ------------------------------------------------------------------- tls
    ("sni_entropy", "derived.sni_entropy", "bounded", "derived.sni_entropy_n", "tls"),
    ("tls_weak_ratio", "tls.weak_version_ratio", "ratio", _TLSH, "tls"),
    ("ja3_diversity", "derived.ja3_diversity", "count", None, "tls"),
    ("tls_handshake_ms", "tls.handshake_ms_avg", "avg", _TLSH, "tls"),
    # ---------------------------------------------------------------- timing
    ("periodicity", "derived.periodicity_score", "window", None, "timing"),
    ("timing_regularity", "derived.timing_regularity", "window", None, "timing"),
    ("think_time", "derived.think_time_s_avg", "avg", None, "timing"),
    ("req_per_session", "derived.req_per_session", "window", None, "timing"),
    ("duty_cycle", "derived.activity_duty_cycle", "window", None, "timing"),
    # ------------------------------------------------------------- transport
    ("retransmit_rate", "l4.retransmit_rate", "ratio", "l4.pkts_total", "transport"),
    ("rtt", "l4.rtt_ms_avg", "avg", _FLOWS, "transport"),
    ("syn_ratio", ("l4.syn_count", _FLOWS), "ratio", _FLOWS, "transport"),
    ("flow_duration", "l4.flow_duration_ms_avg", "avg", _FLOWS, "transport"),
    # ----------------------------------------------------------------- probe
    ("probe_reachable", "probe.reachable", "gauge", None, "probe"),
    ("probe_loss", "probe.loss_ratio", "ratio", "probe.probes", "probe"),
    # ------------------------------------------------ comp (CLR composition)
    ("comp_get", "http.get_count", "clr", None, "comp"),
    ("comp_write", "http.write_count", "clr", None, "comp"),
    ("comp_4xx", "http.status_4xx", "clr", None, "comp"),
    ("comp_5xx", "http.status_5xx", "clr", None, "comp"),
    ("comp_dns", _DNSQ, "clr", None, "comp"),
    ("comp_tls", _TLSH, "clr", None, "comp"),
    ("comp_flows", _FLOWS, "clr", None, "comp"),
    ("comp_syn", "l4.syn_count", "clr", None, "comp"),
]

FEATURE_DIM: int = len(FEATURE_SPEC_V2)
FEATURE_NAMES_V2: List[str] = [f[0] for f in FEATURE_SPEC_V2]
FEATURE_SOURCE: Dict[str, Source] = {f[0]: f[1] for f in FEATURE_SPEC_V2}
FEATURE_KIND: Dict[str, str] = {f[0]: f[2] for f in FEATURE_SPEC_V2}
FEATURE_NSRC: Dict[str, NSource] = {f[0]: f[3] for f in FEATURE_SPEC_V2}
FEATURE_GROUP: Dict[str, str] = {f[0]: f[4] for f in FEATURE_SPEC_V2}
FEATURE_INDEX: Dict[str, int] = {n: i for i, n in enumerate(FEATURE_NAMES_V2)}

GROUP_ORDER: Tuple[str, ...] = ("volume", "breadth", "app", "dns", "tls", "timing",
                                "transport", "probe", "comp")
GROUPS: Dict[str, List[int]] = {g: [i for i, f in enumerate(FEATURE_SPEC_V2) if f[4] == g]
                                for g in GROUP_ORDER}

KEY_FEATURES: List[str] = [
    "bytes_up", "bytes_down", "flows", "http_requests", "dns_queries", "tls_handshakes",
    "distinct_peers", "distinct_templates", "http_write_ratio", "http_4xx_rate",
    "updown_log", "bytes_per_flow",
]
KEY_FEATURE_IDX: List[int] = [FEATURE_INDEX[n] for n in KEY_FEATURES]

CLR_FEATURES: List[str] = [n for n in FEATURE_NAMES_V2 if FEATURE_KIND[n] == "clr"]
CLR_IDX: List[int] = [FEATURE_INDEX[n] for n in CLR_FEATURES]

EXPOSURE_CHANNELS: List[str] = ["http", "dns", "tls", "flows", "probe"]
EXPOSURE_SOURCE: Dict[str, str] = {
    "http": _HTTPR, "dns": _DNSQ, "tls": _TLSH, "flows": _FLOWS, "probe": "probe.probes",
}

# Kinds whose stale value is a true 0 (absence is data) vs NaN (undefined).
ZERO_WHEN_STALE = frozenset({"count", "bytes"})


# ------------------------------------------------------------------ sources
def source_kind(src: Source) -> str:
    """'metric' | 'div' | 'logratio' for a FEATURE_SPEC source."""
    if isinstance(src, str):
        return "metric"
    if len(src) == 3 and src[0] == LOGRATIO:
        return "logratio"
    if len(src) == 2:
        return "div"
    raise ValueError(f"bad feature source {src!r}")


def source_metrics(src: Union[Source, NSource]) -> List[str]:
    """Metric names a source (or n_source) reads, in order."""
    if src is None:
        return []
    if isinstance(src, str):
        return [src]
    if len(src) == 3 and src[0] == LOGRATIO:
        return [src[1], src[2]]
    return list(src)


def all_source_metrics() -> List[str]:
    """Every metric name FEATURE_SPEC_V2 reads (sources and n_sources), sorted."""
    out = set()
    for _, src, _, nsrc, _ in FEATURE_SPEC_V2:
        out.update(source_metrics(src))
        out.update(source_metrics(nsrc))
    return sorted(out)


Getter = Callable[[str], Optional[float]]


def _num(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _n_value(nsrc: NSource, get: Getter) -> Optional[float]:
    if nsrc is None:
        return None
    if isinstance(nsrc, str):
        return _num(get(nsrc))
    vals = [_num(get(m)) for m in nsrc]
    vals = [v for v in vals if v is not None]
    return float(sum(vals)) if vals else None


def feature_inputs(idx: int, get: Getter) -> Tuple[Optional[float], Optional[float]]:
    """Resolve feature `idx` to the (v, n) pair `transform` expects.

    `get(metric)` must return the metric value only if it is fresh (written at
    ctx.now), else None. For ratio features v is the numerator count k.
    """
    name, src, kind, nsrc, _ = FEATURE_SPEC_V2[idx]
    n = _n_value(nsrc, get)
    if nsrc is not None and n is None:
        n = 0.0     # a declared exposure that is stale gates the feature to NaN
    sk = source_kind(src)
    if sk == "metric":
        v = _num(get(src))  # type: ignore[arg-type]
        if kind == "ratio" and v is not None:
            v = None if n is None else v * n     # fraction -> k
        return v, n
    if sk == "logratio":
        a, b = _num(get(src[1])), _num(get(src[2]))
        if a is None or b is None:
            return None, n
        return math.log((max(a, 0.0) + 1.0) / (max(b, 0.0) + 1.0)), n
    num, den = _num(get(src[0])), _num(get(src[1]))
    if kind == "ratio":
        return num, n
    if num is None or den is None or den <= 0:
        return None, n
    return num / den, n


# --------------------------------------------------------------- transforms
def _logit(p: float) -> float:
    return math.log(p) - math.log1p(-p)


def transform(kind: str, v: Optional[float], n: Optional[float] = None, dt_s: float = 60.0,
              *, tx: Optional[str] = None) -> Tuple[float, float]:
    """Per-kind transform -> (vec_value, nat_value). NaN means undefined.

    count/bytes : vec = log1p(v*60/dt), nat = v (raw count this tick); stale -> (0, 0)
    ratio       : v = k, n = exposure; vec = logit((k+0.5)/(n+1)), nat = k/n;
                  stale or n <= 0 -> NaN; k is clipped to [0, n]
    avg         : vec = log(max(v, AVG_FLOOR)), nat = v; stale, v < 0 or
                  (n given and n < 1) -> NaN. n None = no exposure gate.
    bounded     : vec = logit(clip(v, 1e-3, 1-1e-3)), nat = v; needs n >= 5
    gauge/window: vec = nat = v when fresh, else NaN
    ctx         : (NaN, v): context only, never scored
    `tx` overrides the vec transform: 'identity' | 'log1p' | 'log'.
    clr is multivariate: use `clr(counts, dt_s)`.
    """
    nan = float("nan")
    v = _num(v)
    n = _num(n)
    if kind in ("count", "bytes"):
        if v is None:
            return 0.0, 0.0
        v = max(v, 0.0)
        return math.log1p(v * 60.0 / float(dt_s)), v
    if kind == "clr":
        raise ValueError("clr features are built jointly with features.clr(counts, dt_s)")
    if v is None:
        return nan, nan
    if kind == "ratio":
        if n is None or n <= 0:
            return nan, nan
        k = min(max(v, 0.0), n)
        return _logit((k + 0.5) / (n + 1.0)), k / n
    if kind == "avg":
        if n is not None and n < 1:
            return nan, nan
        if tx == "identity":
            return v, v
        if tx == "log1p":
            return (math.log1p(v), v) if v > -1.0 else (nan, nan)
        if v < 0:
            return nan, nan
        return math.log(max(v, AVG_FLOOR)), v
    if kind == "bounded":
        if n is None or n < BOUNDED_MIN_N:
            return nan, nan
        p = min(max(v, BOUNDED_EPS), 1.0 - BOUNDED_EPS)
        return _logit(p), v
    if kind in ("gauge", "window"):
        if tx == "log1p":
            return (math.log1p(v), v) if v > -1.0 else (nan, nan)
        if tx == "log":
            return (math.log(max(v, AVG_FLOOR)), v) if v >= 0 else (nan, nan)
        return v, v
    if kind == "ctx":
        return nan, v
    raise ValueError(f"unknown feature kind {kind!r}")


def transform_feature(idx: int, v: Optional[float], n: Optional[float], dt_s: float) -> Tuple[float, float]:
    """`transform` with the kind and override looked up for feature `idx`."""
    name = FEATURE_NAMES_V2[idx]
    return transform(FEATURE_KIND[name], v, n, dt_s, tx=VEC_TX.get(name))


def clr(counts: Sequence[Optional[float]], dt_s: float = 60.0) -> Tuple[np.ndarray, np.ndarray]:
    """Centred log-ratio of per-minute rates: y = log(x*60/dt + 0.5); vec = y - mean(y).

    Rates (not raw counts) make the composition cadence independent. Stale
    (None/NaN) counts are 0. If the total is 0 the composition is undefined
    and both outputs are all-NaN. nat = x / sum(x) (shares).
    """
    x = np.array([0.0 if _num(c) is None else max(float(c), 0.0) for c in counts], dtype=float)
    tot = x.sum()
    if tot <= 0:
        full = np.full(x.shape, np.nan)
        return full, full.copy()
    y = np.log(x * 60.0 / float(dt_s) + CLR_PSEUDO)
    return y - y.mean(), x / tot


def compute_features(get: Getter, dt_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Build (vec[52], nat[52]) float64 from a freshness-aware getter.

    `get(metric)` returns the value written at ctx.now or None when stale.
    Pure function of the getter; B01 wraps it around store.latest_fresh.
    """
    vec = np.full(FEATURE_DIM, np.nan)
    nat = np.full(FEATURE_DIM, np.nan)
    for i, (name, src, kind, _nsrc, _g) in enumerate(FEATURE_SPEC_V2):
        if kind == "clr":
            continue
        v, n = feature_inputs(i, get)
        vec[i], nat[i] = transform(kind, v, n, dt_s, tx=VEC_TX.get(name))
    counts = [_num(get(FEATURE_SOURCE[n])) for n in CLR_FEATURES]  # type: ignore[arg-type]
    cv, cn = clr(counts, dt_s)
    vec[CLR_IDX] = cv
    nat[CLR_IDX] = cn
    return vec, nat


def exposure(get: Getter) -> Dict[str, float]:
    """feature.expo {channel: n} from fresh exposure sources (stale -> 0)."""
    return {ch: (_num(get(m)) or 0.0) for ch, m in EXPOSURE_SOURCE.items()}


def group_of(idx: int) -> str:
    return FEATURE_SPEC_V2[idx][4]


def features_in(groups: Union[str, Sequence[str]]) -> List[int]:
    """Indices of all features in the given group(s), in spec order."""
    gs = [groups] if isinstance(groups, str) else list(groups)
    return [i for g in gs for i in GROUPS[g]]


def as_mapping(vec: Sequence[float]) -> Mapping[str, float]:
    """{name: value} view of a 52-vector (for profiles / explanations)."""
    return {n: float(vec[i]) for i, n in enumerate(FEATURE_NAMES_V2)}


# ============================================================================
# spec v2.1: grain additions (docs/lib3/cadence.md §3-§4)
# ============================================================================
# Additive per-tick parts. Each entry is (name, source, weight_source, rule):
#   rule 'sum'  : part = fresh value of `source` (stale -> 0)
#   rule 'prod' : part = source * weight_source when BOTH are fresh, else 0
#                 (an average's / a fraction's sum; its weight is a separate part)
#   rule 'when' : part = weight_source when `source` is fresh, else 0
#                 (the weight of an average that exists this tick)
#   rule 'dt' / 'active' : the tick length / feature.active
PART_SPEC: List[Tuple[str, Optional[str], Optional[str], str]] = [
    ("p_dt", None, None, "dt"),
    ("p_active", None, None, "active"),
    ("p_up", "l4.bytes_up", None, "sum"),
    ("p_down", "l4.bytes_down", None, "sum"),
    ("p_flows", _FLOWS, None, "sum"),
    ("p_http", _HTTPR, None, "sum"),
    ("p_dns", _DNSQ, None, "sum"),
    ("p_tls", _TLSH, None, "sum"),
    ("p_events", "act.events", None, "sum"),
    ("p_l3b", "l3.bytes_total", None, "sum"),
    ("p_newpeer", "derived.new_peer_count", None, "sum"),
    ("p_write", "http.write_count", None, "sum"),
    ("p_get", "http.get_count", None, "sum"),
    ("p_4xx", "http.status_4xx", None, "sum"),
    ("p_5xx", "http.status_5xx", None, "sum"),
    ("p_3xx", "http.status_3xx", None, "sum"),
    ("p_lat_s", "http.latency_ms_avg", _HTTPR, "prod"),
    ("p_lat_n", "http.latency_ms_avg", _HTTPR, "when"),
    ("p_resp_s", "http.resp_bytes_avg", _HTTPR, "prod"),
    ("p_req_s", "http.req_bytes_avg", _HTTPR, "prod"),
    ("p_ntr_k", "act.new_template_ratio", _HTTPR, "prod"),
    ("p_ntr_n", "act.new_template_ratio", _HTTPR, "when"),
    ("p_dga_s", "derived.dns_dga_score", "derived.dns_dga_named_n", "prod"),
    ("p_dga_n", "derived.dns_dga_score", "derived.dns_dga_named_n", "when"),
    ("p_nx_k", "derived.dns_fail_rate", "derived.dns_fail_rate.n", "prod"),
    ("p_nx_n", "derived.dns_fail_rate", "derived.dns_fail_rate.n", "when"),
    ("p_txt", "dns.txt_count", None, "sum"),
    ("p_qlen_s", "dns.qname_len_avg", _DNSQ, "prod"),
    ("p_qlen_n", "dns.qname_len_avg", _DNSQ, "when"),
    ("p_weak_k", "tls.weak_version_ratio", _TLSH, "prod"),
    ("p_hsms_s", "tls.handshake_ms_avg", _TLSH, "prod"),
    ("p_hsms_n", "tls.handshake_ms_avg", _TLSH, "when"),
    ("p_think_ls", "derived.think_log_sum", None, "sum"),
    ("p_think_n", "derived.think_gaps", None, "sum"),
    ("p_pkts", "l4.pkts_total", None, "sum"),
    ("p_retx_k", "l4.retransmit_rate", "l4.pkts_total", "prod"),
    ("p_rtt_s", "l4.rtt_ms_avg", _FLOWS, "prod"),
    ("p_rtt_n", "l4.rtt_ms_avg", _FLOWS, "when"),
    ("p_syn", "l4.syn_count", None, "sum"),
    ("p_dur_s", "l4.flow_duration_ms_avg", _FLOWS, "prod"),
    ("p_dur_n", "l4.flow_duration_ms_avg", _FLOWS, "when"),
    ("p_probes", "probe.probes", None, "sum"),
    ("p_reach_s", "probe.reachable", "probe.probes", "prod"),
    ("p_loss_k", "probe.loss_ratio", "probe.probes", "prod"),
    ("p_sni_n", "derived.sni_entropy_n", None, "sum"),
    ("p_path_n", "derived.path_entropy_n", None, "sum"),
    ("p_dnsent_n", "derived.dns_name_entropy_n", None, "sum"),
]
PART_NAMES: List[str] = [p[0] for p in PART_SPEC]
PART_DIM: int = len(PART_SPEC)
PART_INDEX: Dict[str, int] = {n: i for i, n in enumerate(PART_NAMES)}
if PART_DIM != 47:                                        # contract (cadence.md §3.1)
    raise ImportError(f"features.PART_SPEC must have 47 parts, has {PART_DIM}")

# how the grain value of each feature is built (cadence.md §4)
SET_SOURCES: Dict[str, str] = {
    "distinct_peers": "l4.peer_ids", "distinct_dports": "l4.dport_ids",
    "distinct_templates": "act.template_ids", "ja3_diversity": "tls.ja3_ids",
}
MAP_SOURCES: Dict[str, Tuple[str, ...]] = {
    "dest_concentration": ("tls.sni_set", "dns.qname_set"),
    "path_entropy": ("http.top_paths",),
    "dns_name_entropy": ("dns.qname_set",),
    "sni_entropy": ("tls.sni_set",),
}
MAP_RAW_SETS: Tuple[str, ...] = ("tls.sni_set", "http.top_paths", "dns.qname_set")
SPAN_FEATURES: Dict[str, float] = {"periodicity": 21600.0, "timing_regularity": 21600.0,
                                   "req_per_session": 86400.0, "duty_cycle": 86400.0}
GRAIN_CLASS: Dict[str, str] = {
    n: ("set" if n in SET_SOURCES else "map" if n in MAP_SOURCES
        else "span" if n in SPAN_FEATURES else "add") for n in FEATURE_NAMES_V2}
_T_JENSEN = {"bytes_up", "bytes_down", "bytes_per_flow", "http_latency", "resp_bytes_avg",
             "req_bytes_avg", "dns_dga_score", "dns_qname_len", "tls_handshake_ms", "rtt",
             "flow_duration"}
_T_MEAN = {"updown_log", "think_time", "retransmit_rate", "probe_reachable", "probe_loss"}


def _transfer_of(name: str) -> Optional[str]:
    gc = GRAIN_CLASS[name]
    if gc == "span":
        return None
    if gc in ("set", "map"):
        return "none"
    kind = FEATURE_KIND[name]
    if name in _T_JENSEN:
        return "jensen"
    if name in _T_MEAN or kind == "clr":
        return "mean"
    if kind == "count":
        return "rate"
    if kind == "ratio":
        return "ratio"
    return "mean"


TRANSFER: Dict[str, Optional[str]] = {n: _transfer_of(n) for n in FEATURE_NAMES_V2}
SPAN_IDX: List[int] = [FEATURE_INDEX[n] for n in SPAN_FEATURES]
SET_IDX: List[int] = [FEATURE_INDEX[n] for n in SET_SOURCES]
MAP_IDX: List[int] = [FEATURE_INDEX[n] for n in MAP_SOURCES]
NONE_TRANSFER_IDX: List[int] = [FEATURE_INDEX[n] for n in FEATURE_NAMES_V2
                                if TRANSFER[n] == "none"]
_P = PART_INDEX


def compute_parts(get: Getter, dt_s: float, active: Optional[float] = None) -> np.ndarray:
    """float64[47] additive parts of one tick (cadence.md §3.1). `get` is the
    freshness-aware getter of compute_features (None when stale). A stale
    source is a part of 0; an average's sum and weight are both 0 when the
    average is absent. `active` defaults to get('feature.active')."""
    out = np.zeros(PART_DIM)
    for i, (name, src, wsrc, rule) in enumerate(PART_SPEC):
        if rule == "dt":
            out[i] = float(dt_s)
        elif rule == "active":
            a = _num(active) if active is not None else _num(get("feature.active"))
            out[i] = 1.0 if (a is not None and a > 0.5) else 0.0
        elif rule == "sum":
            v = _num(get(src))
            if v is None:
                continue
            # counts are >= 0; a sum of log gaps may be negative (sub-second gaps)
            out[i] = v if (v > 0.0 or name == "p_think_ls") else 0.0
        else:
            v = _num(get(src))
            w = _num(get(wsrc))
            if v is None or w is None or not w > 0.0:
                continue
            if rule == "prod":
                out[i] = v * w
            else:                                           # 'when'
                out[i] = w
    return out


def _avg(S: np.ndarray, s: str, n: str) -> Optional[float]:
    w = S[_P[n]]
    return float(S[_P[s]] / w) if w > 0.0 else None


def _entropy_of(named: Mapping[str, float]) -> float:
    vals = [c for c in named.values() if c > 0]
    if len(vals) <= 1:
        return 0.0 if vals else math.nan
    tot = float(sum(vals))
    h = -sum((c / tot) * math.log2(c / tot) for c in vals)
    return h / math.log2(len(vals))


def _split_map(m: Optional[Mapping[str, Any]]) -> Optional[Tuple[Dict[str, float], float]]:
    """(named entries, total incl. '__other__') of a merged count map (D1's split_set)."""
    if not isinstance(m, Mapping) or not m:
        return None
    named: Dict[str, float] = {}
    total = 0.0
    for k, v in m.items():
        c = _num(v)
        if c is None or not c > 0.0:
            continue
        total += c
        if k != "__other__":
            named[str(k)] = c
    return (named, total) if total > 0.0 else None


def map_value(feature: str, maps: Mapping[str, Any]) -> Optional[float]:
    """D1's value of a map feature on merged count maps (None when undefined)."""
    if feature == "dest_concentration":
        for src in MAP_SOURCES[feature]:
            sp = _split_map(maps.get(src))
            if sp is None:
                continue
            named, total = sp
            top = max(named.values()) if named else 0.0
            if top > 0.0:
                return top / total
        return None
    sp = _split_map(maps.get(MAP_SOURCES[feature][0]))
    if sp is None:
        return None
    named, _total = sp
    if not named:
        return None
    v = _entropy_of(named)
    return v if math.isfinite(v) else None


def grain_values(S: Sequence[float], cov_s: float, G: float,
                 sets: Optional[Mapping[str, Optional[float]]] = None,
                 maps: Optional[Mapping[str, Any]] = None,
                 span: Optional[Mapping[str, Optional[float]]] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """(vec[52], nat[52]) of a grain window from the part sums S[47] and its
    coverage cov_s (cadence.md §4): additive features from S with dt := cov,
    set features from the SetSketch union sizes `sets` {feature: count},
    map features from the merged raw maps `maps` {raw set name: {key: count}}
    (D1's rules), span features from `span` {feature: value}. Set and map
    features are NaN when their input is None or cov < COVER_MIN G. At
    cov = dt with one tick and untruncated sets it equals compute_features
    except think_time (geometric mean of the gaps instead of their median)
    and ja3_diversity (exact distinct JA3 instead of named + 1)."""
    S = np.asarray(S, dtype=np.float64).reshape(-1)
    if S.size != PART_DIM:
        raise ValueError(f"grain_values: S must have {PART_DIM} parts, got {S.size}")
    cov = float(cov_s)
    if not (cov > 0.0 and math.isfinite(cov)):
        raise ValueError(f"grain_values: bad cov_s {cov_s!r}")
    covered = cov >= 0.95 * float(G) * (1.0 - 1e-9)
    sets = sets or {}
    maps = maps or {}
    span = span or {}
    vals: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    def s(name: str) -> float:
        return float(S[_P[name]])

    def ratio(k: str, n: str) -> Tuple[Optional[float], Optional[float]]:
        return s(k), s(n)

    vals["bytes_up"] = (s("p_up"), None)
    vals["bytes_down"] = (s("p_down"), None)
    vals["flows"] = (s("p_flows"), None)
    vals["http_requests"] = (s("p_http"), None)
    vals["dns_queries"] = (s("p_dns"), None)
    vals["tls_handshakes"] = (s("p_tls"), None)
    vals["intensity"] = (s("p_events"), None)
    fl = s("p_flows")
    vals["bytes_per_flow"] = ((s("p_l3b") / fl) if fl > 0.0 else None, fl)
    vals["updown_log"] = ((math.log((max(s("p_up"), 0.0) + 1.0) / (max(s("p_down"), 0.0) + 1.0))
                           if fl > 0.0 else None), fl)
    vals["new_peer_count"] = (s("p_newpeer"), None)
    ht = s("p_http")
    vals["http_write_ratio"] = ratio("p_write", "p_http")
    vals["http_get_ratio"] = ratio("p_get", "p_http")
    vals["http_4xx_rate"] = ratio("p_4xx", "p_http")
    vals["http_5xx_rate"] = ratio("p_5xx", "p_http")
    vals["http_3xx_rate"] = ratio("p_3xx", "p_http")
    vals["http_latency"] = (_avg(S, "p_lat_s", "p_lat_n"), s("p_lat_n"))
    vals["resp_bytes_avg"] = (((s("p_resp_s") / ht) if ht > 0.0 else None), ht)
    vals["req_bytes_avg"] = (((s("p_req_s") / ht) if ht > 0.0 else None), ht)
    vals["new_template_ratio"] = ratio("p_ntr_k", "p_ntr_n")
    vals["dns_dga_score"] = (_avg(S, "p_dga_s", "p_dga_n") if s("p_dns") >= 1.0 else None,
                             s("p_dns"))
    vals["dns_fail_rate"] = ratio("p_nx_k", "p_nx_n")
    vals["dns_txt_ratio"] = ratio("p_txt", "p_dns")
    vals["dns_qname_len"] = (_avg(S, "p_qlen_s", "p_qlen_n"), s("p_qlen_n"))
    vals["tls_weak_ratio"] = ratio("p_weak_k", "p_tls")
    vals["tls_handshake_ms"] = (_avg(S, "p_hsms_s", "p_hsms_n"), s("p_hsms_n"))
    vals["retransmit_rate"] = ratio("p_retx_k", "p_pkts")
    vals["rtt"] = (_avg(S, "p_rtt_s", "p_rtt_n"), s("p_rtt_n"))
    vals["syn_ratio"] = ratio("p_syn", "p_flows")
    vals["flow_duration"] = (_avg(S, "p_dur_s", "p_dur_n"), s("p_dur_n"))
    pr = s("p_probes")
    vals["probe_reachable"] = ((s("p_reach_s") / pr) if pr > 0.0 else None, None)
    vals["probe_loss"] = ratio("p_loss_k", "p_probes")

    vec = np.full(FEATURE_DIM, np.nan)
    nat = np.full(FEATURE_DIM, np.nan)
    for i, name in enumerate(FEATURE_NAMES_V2):
        kind = FEATURE_KIND[name]
        gc = GRAIN_CLASS[name]
        if kind == "clr":
            continue
        if gc == "set":
            c = sets.get(name)
            v = _num(c) if covered else None
            if v is None:
                continue                                   # undefined, not 0
            vec[i], nat[i] = transform(kind, v, None, cov, tx=VEC_TX.get(name))
            continue
        if gc == "map":
            if not covered:
                continue
            if name == "dest_concentration":
                n = s("p_tls") + s("p_dns")
            elif name == "path_entropy":
                n = s("p_path_n")
            elif name == "sni_entropy":
                n = s("p_sni_n")
            else:
                n = s("p_dnsent_n")
            v = map_value(name, maps)
            vec[i], nat[i] = transform(kind, v, n, cov, tx=VEC_TX.get(name))
            continue
        if gc == "span":
            v = _num(span.get(name))
            vec[i], nat[i] = transform(kind, v, None, cov, tx=VEC_TX.get(name))
            continue
        if name == "think_time":
            n = s("p_think_n")
            if n >= 2.0:
                mlog = s("p_think_ls") / n
                v = math.exp(mlog)
                vec[i], nat[i] = max(mlog, math.log(AVG_FLOOR)), v
            continue
        v, n = vals[name]
        vec[i], nat[i] = transform(kind, v, n, cov, tx=VEC_TX.get(name))
    counts = [s("p_get"), s("p_write"), s("p_4xx"), s("p_5xx"), s("p_dns"), s("p_tls"),
              s("p_flows"), s("p_syn")]
    cv, cn = clr(counts, cov)
    vec[CLR_IDX] = cv
    nat[CLR_IDX] = cn
    return vec, nat


def grain_exposure(S: Sequence[float]) -> Dict[str, float]:
    """feature.expo.<g> {channel: sum n} from the part sums."""
    S = np.asarray(S, dtype=np.float64).reshape(-1)
    return {"http": float(S[_P["p_http"]]), "dns": float(S[_P["p_dns"]]),
            "tls": float(S[_P["p_tls"]]), "flows": float(S[_P["p_flows"]]),
            "probe": float(S[_P["p_probes"]])}
