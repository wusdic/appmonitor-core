"""L2/L3 raw-metric engine — link & network layer.

Acquisition: passive SPAN/TAP decode (or flow records). No host agent.
Produces per-entity, per-tick network-layer facts: packet/byte volumes,
protocol mix, TTL (an OS / hop-distance hint), and IPv4/IPv6 split.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ...core.engine import Context, Engine
from ...models.schema import AcquisitionMethod, MetricKind, Observation, RawMetric


class L2L3Engine(Engine):
    name = "raw.l2l3"
    layer = "raw"
    consumes = ["<observations>"]
    produces = [
        "l3.pkts_total", "l3.bytes_total", "l3.proto.tcp_ratio",
        "l3.proto.udp_ratio", "l3.proto.icmp_ratio", "l3.ttl_min",
        "l3.ipv6_ratio",
    ]
    description = "Link/network-layer volumes, protocol mix and TTL from passive decode."

    def run(self, ctx: Context, observations: Optional[List[Observation]] = None) -> int:
        observations = observations or []
        agg: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        ttl_min: Dict[Tuple[str, str], int] = {}
        for o in observations:
            key = (o.system, o.entity)
            a = agg[key]
            a["pkts"] += o.pkts_up + o.pkts_down
            a["bytes"] += o.bytes_up + o.bytes_down
            proto = (o.l4_proto or "other").lower()
            a[f"proto_{proto}"] += 1
            a["flows"] += 1
            if o.l3_proto == "ipv6":
                a["ipv6"] += 1
            if o.ttl:
                k = key
                ttl_min[k] = o.ttl if k not in ttl_min else min(ttl_min[k], o.ttl)

        n = 0
        for (system, entity), a in agg.items():
            flows = max(a["flows"], 1.0)
            emit = [
                ("l3.pkts_total", a["pkts"], MetricKind.COUNTER, "packets"),
                ("l3.bytes_total", a["bytes"], MetricKind.COUNTER, "bytes"),
                ("l3.proto.tcp_ratio", a.get("proto_tcp", 0) / flows, MetricKind.RATE, "ratio"),
                ("l3.proto.udp_ratio", a.get("proto_udp", 0) / flows, MetricKind.RATE, "ratio"),
                ("l3.proto.icmp_ratio", a.get("proto_icmp", 0) / flows, MetricKind.RATE, "ratio"),
                ("l3.ipv6_ratio", a.get("ipv6", 0) / flows, MetricKind.RATE, "ratio"),
            ]
            if (system, entity) in ttl_min:
                emit.append(("l3.ttl_min", float(ttl_min[(system, entity)]),
                             MetricKind.GAUGE, "hops"))
            for name, value, kind, unit in emit:
                ctx.store.add_raw(RawMetric(
                    name=name, value=value, ts=ctx.now, system=system, entity=entity,
                    kind=kind, method=AcquisitionMethod.PASSIVE_SPAN, unit=unit))
                n += 1
        return n
