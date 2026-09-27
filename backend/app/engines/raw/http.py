"""HTTP raw-metric engine — application layer (cleartext or via proxy logs).

Acquisition: passive SPAN decode of cleartext HTTP, or off-host access-log
shipping (reverse proxy / WAF / LB logs). Emits request-mix, status-class
counts, latency and payload sizes plus categorical evidence (methods, top
paths, content types, user-agents) used later by the signature engines.

v2 (lib-3, R1). Why each rule exists:

* Weighted records. A record may stand for w = extra['count'] requests
  (aggregated access logs, the generator's aggregated mode). Counters add w,
  payload bytes come from extra['bytes_up_total'/'bytes_down_total'] when
  present (else w * bytes), averages are w-weighted.
* Full sets. The categorical sets keep the top 64 values plus '__other__'
  (the v1 top 8 lost the rare paths that novelty and budgets look for), so
  the values always sum to the true request count.
* Counts next to ratios. http.get_count / http.write_count are the numerators
  FEATURE_SPEC v2 needs to rebuild exact Beta-Binomial exposure; the v1
  *_ratio metrics are kept unchanged for the lib-4 signatures.
* Absence is data. Metrics exist only for entities with requests this tick
  (ts = ctx.now); a latency average over no timed request is not emitted
  rather than faked as 0.
* Pseudo-entities ('__*', 'class:*') never carry raw data: such records are
  dropped and counted in this engine's health record (dropped_pseudo).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import (AcquisitionMethod, MetricKind, Observation, RawMetric,
                              is_pseudo_entity)

TOP_K = 64
OTHER = "__other__"
_ACTIVE = frozenset({AcquisitionMethod.ACTIVE_PROBE, AcquisitionMethod.ACTIVE_DNS,
                     AcquisitionMethod.ACTIVE_TLS})
_NUM = (int, float, np.integer, np.floating)
_INF = math.inf


# ------------------------------------------------------------------ helpers
# The per-record loop checks numeric fields inline as `0 < x < _INF` (False
# for NaN, inf, negatives and 0 = "not measured"): a helper call per field
# costs ~10x the comparison and this loop sees every record of the tick.
def _weight(ex: Dict[str, Any]) -> int:
    """Records a record stands for: int(extra['count']), default 1. An
    unparsable or non-finite count is one record and <= 0 is none (the same
    rule as R2, so act.events and these counters agree on the same input)."""
    c = ex.get("count")
    if c is None:
        return 1
    if c.__class__ is int:
        return c if c > 0 else 0
    try:
        v = float(c)
    except (TypeError, ValueError, OverflowError):
        return 1
    if not math.isfinite(v):
        return 1
    return int(v) if v >= 1.0 else 0


def _total(x: Any) -> Optional[float]:
    """A pre-aggregated total if usable (finite, >= 0), else None."""
    if isinstance(x, _NUM) and not isinstance(x, bool) and 0 <= x < _INF:
        return float(x)
    return None


def _bytes(o: Observation, ex: Dict[str, Any], w: int) -> Tuple[float, float]:
    """(up, down) bytes of a weighted record: extra totals first, else w * bytes."""
    up = _total(ex.get("bytes_up_total"))
    down = _total(ex.get("bytes_down_total"))
    if up is None:
        x = o.bytes_up
        up = w * x if 0 < x < _INF else 0.0
    if down is None:
        x = o.bytes_down
        down = w * x if 0 < x < _INF else 0.0
    return up, down


def _full_set(d: Dict[str, int], k: int = TOP_K) -> Dict[str, int]:
    """Top-k values plus '__other__' (the remainder), so values sum to the
    true total. Ties rank by key, so the kept set does not depend on the
    order the records arrived in."""
    if len(d) <= k:
        return dict(d)
    items = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))
    out = dict(items[:k])
    rest = sum(n for _, n in items[k:])
    if rest > 0:
        out[OTHER] = out.get(OTHER, 0) + rest
    return out


class _Acc:
    __slots__ = ("req", "sc", "lat_s", "lat_w", "up", "down", "get", "post", "write",
                 "methods", "paths", "uas", "ctypes")

    def __init__(self) -> None:
        self.req = 0
        self.sc = [0, 0, 0, 0, 0, 0]          # index = status class 1..5
        self.lat_s = 0.0
        self.lat_w = 0
        self.up = self.down = 0.0
        self.get = self.post = self.write = 0
        self.methods: Dict[str, int] = {}
        self.paths: Dict[str, int] = {}
        self.uas: Dict[str, int] = {}
        self.ctypes: Dict[str, int] = {}


class HTTPEngine(Engine):
    name = "raw.http"
    layer = "raw"
    consumes = ["<observations>"]
    produces = [
        "http.requests", "http.status_2xx", "http.status_3xx",
        "http.status_4xx", "http.status_5xx", "http.latency_ms_avg",
        "http.req_bytes_avg", "http.resp_bytes_avg", "http.distinct_paths",
        "http.get_count", "http.write_count",
        "http.get_ratio", "http.post_ratio", "http.write_ratio",
        "http.methods", "http.top_paths", "http.user_agents", "http.content_types",
    ]
    description = ("HTTP request mix, status classes, latency, payload sizes and full "
                   "categorical sets from weighted records.")

    WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        self.dropped_pseudo = 0             # last run: records of pseudo-entities
        self.dropped_invalid = 0            # last run: records with an empty system / entity

    def health_record(self, ok: bool = True) -> Dict[str, Any]:
        rec = super().health_record(ok)
        rec["dropped_pseudo"] = self.dropped_pseudo
        rec["dropped_invalid"] = self.dropped_invalid
        return rec

    def run(self, ctx: Context, observations: Optional[List[Observation]] = None) -> int:
        self.dropped_pseudo = self.dropped_invalid = 0
        writes = self.WRITE_METHODS
        acc: Dict[Tuple[str, str], Optional[_Acc]] = {}
        for o in observations or ():
            # an HTTP request: a decoded method, or an 'http' record with a path / host
            # (the same selection as R2, so http.requests and act.events agree)
            if not (o.http_method or (o.app_proto == "http" and (o.http_path or o.http_host))) \
                    or o.method in _ACTIVE:
                continue
            key = (o.system, o.entity)
            a = acc.get(key, False)
            if a is False:
                if not o.system or not o.entity:
                    self.dropped_invalid += 1
                    continue
                a = None if (is_pseudo_entity(o.entity) or is_pseudo_entity(o.system)) else _Acc()
                acc[key] = a
            if a is None:
                self.dropped_pseudo += 1
                continue
            ex = o.extra
            if ex:
                w = _weight(ex)
                if w <= 0:
                    continue
                up, down = _bytes(o, ex, w)
            else:
                w = 1
                up, down = o.bytes_up, o.bytes_down
                if not 0 < up < _INF:
                    up = 0.0
                if not 0 < down < _INF:
                    down = 0.0
            a.req += w
            a.up += up
            a.down += down
            x = o.http_status
            if 100 <= x < 600:                    # 0 = no response, NaN = unknown
                a.sc[x // 100 if x.__class__ is int else int(x) // 100] += w
            x = o.duration_ms
            if 0 < x < _INF:
                a.lat_s += w * x
                a.lat_w += w
            m = o.http_method
            if m:
                m = m.upper()
                a.methods[m] = a.methods.get(m, 0) + w
                if m == "GET":
                    a.get += w
                elif m in writes:
                    a.write += w
                    if m == "POST":
                        a.post += w
            p = o.http_path
            if p:
                a.paths[p] = a.paths.get(p, 0) + w
            ua = o.user_agent
            if ua:
                a.uas[ua] = a.uas.get(ua, 0) + w
            ct = o.content_type
            if ct:
                a.ctypes[ct] = a.ctypes.get(ct, 0) + w

        add = ctx.store.add_raw
        now = ctx.now
        method = AcquisitionMethod.PASSIVE_SPAN
        C, G, R, K = MetricKind.COUNTER, MetricKind.GAUGE, MetricKind.RATE, MetricKind.CATEGORICAL
        n = 0
        for (system, entity), a in acc.items():
            if a is None or a.req <= 0:
                continue
            req = float(a.req)
            sc = a.sc
            out: List[Tuple[str, Any, MetricKind, str]] = [
                ("http.requests", req, C, "req"),
                ("http.status_2xx", float(sc[2]), C, "req"),
                ("http.status_3xx", float(sc[3]), C, "req"),
                ("http.status_4xx", float(sc[4]), C, "req"),
                ("http.status_5xx", float(sc[5]), C, "req"),
                ("http.req_bytes_avg", a.up / req, G, "bytes"),
                ("http.resp_bytes_avg", a.down / req, G, "bytes"),
                ("http.distinct_paths", float(len(a.paths)), G, "paths"),
                ("http.get_count", float(a.get), C, "req"),
                ("http.write_count", float(a.write), C, "req"),
                ("http.get_ratio", a.get / req, R, "ratio"),
                ("http.post_ratio", a.post / req, R, "ratio"),
                ("http.write_ratio", a.write / req, R, "ratio"),
            ]
            if a.lat_w:
                out.append(("http.latency_ms_avg", a.lat_s / a.lat_w, G, "ms"))
            for mname, dist in (("http.methods", a.methods), ("http.top_paths", a.paths),
                                ("http.user_agents", a.uas), ("http.content_types", a.ctypes)):
                if dist:
                    out.append((mname, _full_set(dist), K, ""))
            for mname, value, kind, unit in out:
                add(RawMetric(name=mname, value=value, ts=now, system=system, entity=entity,
                              kind=kind, method=method, unit=unit))
            n += len(out)
        return n
