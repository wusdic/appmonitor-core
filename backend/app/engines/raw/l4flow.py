"""L4 flow raw-metric engine — transport layer.

Acquisition: passive SPAN decode or NetFlow/IPFIX. Emits per-entity flow
volumes and directionality, plus TCP health signals (retransmits, RTT,
window) that need no host agent to observe.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ...core.engine import Context, Engine
from ...models.schema import AcquisitionMethod, MetricKind, Observation, RawMetric


class L4FlowEngine(Engine):
    name = "raw.l4flow"
    layer = "raw"
    consumes = ["<observations>"]
    produces = [
        "l4.flows", "l4.bytes_up", "l4.bytes_down", "l4.updown_ratio",
        "l4.distinct_peers", "l4.distinct_dports", "l4.syn_count",
        "l4.retransmit_rate", "l4.rtt_ms_avg", "l4.win_size_avg",
        "l4.flow_duration_ms_avg",
    ]
    description = "TCP/UDP flow volume, directionality, fan-out and TCP health from passive decode."

    def run(self, ctx: Context, observations: Optional[List[Observation]] = None) -> int:
        observations = observations or []
        agg: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        peers: Dict[Tuple[str, str], set] = defaultdict(set)
        dports: Dict[Tuple[str, str], set] = defaultdict(set)
        for o in observations:
            key = (o.system, o.entity)
            a = agg[key]
            a["flows"] += 1
            a["bytes_up"] += o.bytes_up
            a["bytes_down"] += o.bytes_down
            a["pkts"] += o.pkts_up + o.pkts_down
            a["retransmits"] += o.retransmits
            if "SYN" in o.tcp_flags and "ACK" not in o.tcp_flags:
                a["syn"] += 1
            if o.rtt_ms:
                a["rtt_sum"] += o.rtt_ms
                a["rtt_n"] += 1
            if o.win_size:
                a["win_sum"] += o.win_size
                a["win_n"] += 1
            if o.duration_ms:
                a["dur_sum"] += o.duration_ms
                a["dur_n"] += 1
            if o.peer:
                peers[key].add(o.peer)
            if o.dst_port:
                dports[key].add(o.dst_port)

        n = 0
        for key, a in agg.items():
            system, entity = key
            up, down = a["bytes_up"], a["bytes_down"]
            updown = up / down if down > 0 else (up if up > 0 else 0.0)
            metrics = [
                ("l4.flows", a["flows"], MetricKind.COUNTER, "flows"),
                ("l4.bytes_up", up, MetricKind.COUNTER, "bytes"),
                ("l4.bytes_down", down, MetricKind.COUNTER, "bytes"),
                ("l4.updown_ratio", updown, MetricKind.GAUGE, "ratio"),
                ("l4.distinct_peers", float(len(peers[key])), MetricKind.GAUGE, "peers"),
                ("l4.distinct_dports", float(len(dports[key])), MetricKind.GAUGE, "ports"),
                ("l4.syn_count", a.get("syn", 0), MetricKind.COUNTER, "packets"),
                ("l4.retransmit_rate",
                 a["retransmits"] / max(a["pkts"], 1.0), MetricKind.RATE, "ratio"),
                ("l4.rtt_ms_avg",
                 a["rtt_sum"] / a["rtt_n"] if a.get("rtt_n") else 0.0, MetricKind.GAUGE, "ms"),
                ("l4.win_size_avg",
                 a["win_sum"] / a["win_n"] if a.get("win_n") else 0.0, MetricKind.GAUGE, "bytes"),
                ("l4.flow_duration_ms_avg",
                 a["dur_sum"] / a["dur_n"] if a.get("dur_n") else 0.0, MetricKind.GAUGE, "ms"),
            ]
            for mname, value, kind, unit in metrics:
                ctx.store.add_raw(RawMetric(
                    name=mname, value=value, ts=ctx.now, system=system, entity=entity,
                    kind=kind, method=AcquisitionMethod.PASSIVE_FLOW, unit=unit))
                n += 1
        return n
