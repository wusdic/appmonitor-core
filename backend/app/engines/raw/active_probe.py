"""Active-probe raw-metric engine.

Acquisition: probes *we* originate — ICMP echo, TCP connect, banner grab,
HTTP HEAD, TLS cert fetch, traceroute. It never installs anything on the
target; it measures the target from the network.

Cross-network reality is handled as a first-class outcome: when a target sits
behind a boundary we cannot cross, the probe result is UNREACHABLE / TIMEOUT /
FILTERED rather than a missing sample. Downstream engines treat those states
as information (e.g. "server X became unreachable from segment Y") instead of
gaps, and reachability is emitted as its own boolean/categorical metric so a
profile built partly on active data degrades gracefully when a path closes.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ...core.engine import Context, Engine
from ...models.schema import (
    AcquisitionMethod,
    MetricKind,
    Observation,
    RawMetric,
    Reachability,
)


class ActiveProbeEngine(Engine):
    name = "raw.active_probe"
    layer = "raw"
    consumes = ["<observations>"]
    produces = [
        "probe.reachable", "probe.rtt_ms", "probe.open_ports",
        "probe.hop_count", "probe.state", "probe.loss_ratio",
    ]
    description = ("Active reachability/RTT/open-port/banner probes with "
                   "cross-network unreachable & timeout as first-class states.")

    _STATE_SCORE = {
        Reachability.REACHABLE: 1.0,
        Reachability.DEGRADED: 0.5,
        Reachability.FILTERED: 0.2,
        Reachability.UNREACHABLE: 0.0,
        Reachability.TIMEOUT: 0.0,
    }

    def run(self, ctx: Context, observations: Optional[List[Observation]] = None) -> int:
        observations = observations or []
        agg: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        state: Dict[Tuple[str, str], Reachability] = {}
        ports: Dict[Tuple[str, str], set] = defaultdict(set)

        for o in observations:
            if o.method != AcquisitionMethod.ACTIVE_PROBE or o.reachability is None:
                continue
            key = (o.system, o.entity)
            a = agg[key]
            a["probes"] += 1
            a["reach_score"] += self._STATE_SCORE.get(o.reachability, 0.0)
            if o.reachability in (Reachability.UNREACHABLE, Reachability.TIMEOUT,
                                  Reachability.FILTERED):
                a["lost"] += 1
            if o.reachability == Reachability.REACHABLE and o.rtt_ms:
                a["rtt_sum"] += o.rtt_ms
                a["rtt_n"] += 1
            if o.hop_count:
                a["hop"] = max(a.get("hop", 0), o.hop_count)
            for p in o.open_ports:
                ports[key].add(p)
            # keep the worst-but-latest state as the representative
            state[key] = o.reachability

        n = 0
        for key, a in agg.items():
            system, entity = key
            probes = max(a["probes"], 1.0)
            reachable = a["reach_score"] / probes
            st = state.get(key, Reachability.REACHABLE)
            emit = [
                ("probe.reachable", reachable, MetricKind.GAUGE,
                 AcquisitionMethod.ACTIVE_PROBE, "score", {}),
                ("probe.loss_ratio", a.get("lost", 0) / probes, MetricKind.RATE,
                 AcquisitionMethod.ACTIVE_PROBE, "ratio", {}),
                ("probe.open_ports", float(len(ports[key])), MetricKind.GAUGE,
                 AcquisitionMethod.ACTIVE_PROBE, "ports", {}),
                ("probe.state", st.value, MetricKind.CATEGORICAL,
                 AcquisitionMethod.ACTIVE_PROBE, "", {}),
            ]
            if a.get("rtt_n"):
                emit.append(("probe.rtt_ms", a["rtt_sum"] / a["rtt_n"], MetricKind.GAUGE,
                             AcquisitionMethod.ACTIVE_PROBE, "ms", {}))
            if a.get("hop"):
                emit.append(("probe.hop_count", a["hop"], MetricKind.GAUGE,
                             AcquisitionMethod.ACTIVE_PROBE, "hops", {}))
            for mname, value, kind, method, unit, dims in emit:
                ctx.store.add_raw(RawMetric(
                    name=mname, value=value, ts=ctx.now, system=system, entity=entity,
                    kind=kind, method=method, unit=unit, dims=dims))
                n += 1
        return n
