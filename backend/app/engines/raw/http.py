"""HTTP raw-metric engine — application layer (cleartext or via proxy logs).

Acquisition: passive SPAN decode of cleartext HTTP, or off-host access-log
shipping (reverse proxy / WAF / LB logs). Emits request-mix, status-class
counts, latency and payload sizes plus categorical evidence (methods, top
paths, content types, user-agents) used later by the signature engines.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ...core.engine import Context, Engine
from ...models.schema import AcquisitionMethod, MetricKind, Observation, RawMetric


class HTTPEngine(Engine):
    name = "raw.http"
    layer = "raw"
    consumes = ["<observations>"]
    produces = [
        "http.requests", "http.status_2xx", "http.status_3xx",
        "http.status_4xx", "http.status_5xx", "http.latency_ms_avg",
        "http.req_bytes_avg", "http.resp_bytes_avg", "http.distinct_paths",
        "http.get_ratio", "http.post_ratio", "http.write_ratio",
        "http.methods", "http.top_paths", "http.user_agents", "http.content_types",
    ]
    description = "HTTP request mix, status classes, latency, payload sizes and categorical evidence."

    WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

    def run(self, ctx: Context, observations: Optional[List[Observation]] = None) -> int:
        observations = observations or []
        agg: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        methods: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        paths: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        uas: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        ctypes: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for o in observations:
            if o.app_proto != "http" or not o.http_method:
                continue
            key = (o.system, o.entity)
            a = agg[key]
            a["req"] += 1
            sc = o.http_status
            if 200 <= sc < 300:
                a["2xx"] += 1
            elif 300 <= sc < 400:
                a["3xx"] += 1
            elif 400 <= sc < 500:
                a["4xx"] += 1
            elif sc >= 500:
                a["5xx"] += 1
            if o.duration_ms:
                a["lat_sum"] += o.duration_ms
                a["lat_n"] += 1
            a["req_bytes"] += o.bytes_up
            a["resp_bytes"] += o.bytes_down
            m = o.http_method.upper()
            methods[key][m] += 1
            if m == "GET":
                a["get"] += 1
            if m == "POST":
                a["post"] += 1
            if m in self.WRITE_METHODS:
                a["write"] += 1
            if o.http_path:
                paths[key][o.http_path] += 1
            if o.user_agent:
                uas[key][o.user_agent] += 1
            if o.content_type:
                ctypes[key][o.content_type] += 1

        n = 0
        for key, a in agg.items():
            system, entity = key
            req = max(a["req"], 1.0)
            numeric = [
                ("http.requests", a["req"], MetricKind.COUNTER, "req"),
                ("http.status_2xx", a.get("2xx", 0), MetricKind.COUNTER, "req"),
                ("http.status_3xx", a.get("3xx", 0), MetricKind.COUNTER, "req"),
                ("http.status_4xx", a.get("4xx", 0), MetricKind.COUNTER, "req"),
                ("http.status_5xx", a.get("5xx", 0), MetricKind.COUNTER, "req"),
                ("http.latency_ms_avg",
                 a["lat_sum"] / a["lat_n"] if a.get("lat_n") else 0.0, MetricKind.GAUGE, "ms"),
                ("http.req_bytes_avg", a["req_bytes"] / req, MetricKind.GAUGE, "bytes"),
                ("http.resp_bytes_avg", a["resp_bytes"] / req, MetricKind.GAUGE, "bytes"),
                ("http.distinct_paths", float(len(paths[key])), MetricKind.GAUGE, "paths"),
                ("http.get_ratio", a.get("get", 0) / req, MetricKind.RATE, "ratio"),
                ("http.post_ratio", a.get("post", 0) / req, MetricKind.RATE, "ratio"),
                ("http.write_ratio", a.get("write", 0) / req, MetricKind.RATE, "ratio"),
            ]
            for mname, value, kind, unit in numeric:
                ctx.store.add_raw(RawMetric(
                    name=mname, value=value, ts=ctx.now, system=system, entity=entity,
                    kind=kind, method=AcquisitionMethod.PASSIVE_SPAN, unit=unit))
                n += 1
            # categorical distributions (top-k) as CATEGORICAL metrics
            for mname, dist in [("http.methods", methods[key]), ("http.top_paths", paths[key]),
                                ("http.user_agents", uas[key]), ("http.content_types", ctypes[key])]:
                top = dict(sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:8])
                if top:
                    ctx.store.add_raw(RawMetric(
                        name=mname, value=top, ts=ctx.now, system=system, entity=entity,
                        kind=MetricKind.CATEGORICAL, method=AcquisitionMethod.PASSIVE_SPAN))
                    n += 1
        return n
