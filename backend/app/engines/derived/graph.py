"""Graph engine — communication-graph shape per entity.

Builds a lightweight per-tick edge view (entity -> peers) from HTTP/TLS/DNS
categorical sets and flow peer counts, then derives fan-out, peer novelty
(new peers vs. the entity's history) and destination concentration. Peer
novelty is a strong signal of a change in what a system/user talks to.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Set

from ...core.engine import Context, Engine
from ...models.schema import DerivedMetric, MetricKind


class GraphEngine(Engine):
    name = "derived.graph"
    layer = "derived"
    consumes = ["l4.distinct_peers", "tls.sni_set", "dns.qname_set"]
    produces = [
        "derived.fanout", "derived.peer_novelty",
        "derived.dest_concentration", "derived.new_peer_count",
    ]
    description = "Fan-out, peer novelty vs. history and destination concentration."

    def __init__(self, **p):
        super().__init__(**p)
        self._seen: Dict[str, Set[str]] = defaultdict(set)  # (sys|entity)->peers

    def _peers(self, ctx, system, entity) -> Set[str]:
        peers: Set[str] = set()
        for name in ("tls.sni_set", "dns.qname_set"):
            m = ctx.store.latest_raw(system, entity, name)
            if m and isinstance(m.value, dict):
                peers.update(m.value.keys())
        return peers

    def run(self, ctx: Context, observations=None) -> int:
        n = 0
        for system in ctx.store.systems():
            for entity in ctx.store.entities(system):
                key = f"{system}|{entity}"
                peers = self._peers(ctx, system, entity)
                fanout_m = ctx.store.latest_raw(system, entity, "l4.distinct_peers")
                fanout = float(fanout_m.value) if fanout_m and isinstance(fanout_m.value, (int, float)) else float(len(peers))
                seen = self._seen[key]
                new_peers = peers - seen
                novelty = len(new_peers) / len(peers) if peers else 0.0
                seen.update(peers)
                # destination concentration: share of the single top peer
                concentration = 0.0
                m = ctx.store.latest_raw(system, entity, "tls.sni_set")
                if not (m and isinstance(m.value, dict)):
                    m = ctx.store.latest_raw(system, entity, "dns.qname_set")
                if m and isinstance(m.value, dict) and m.value:
                    total = sum(m.value.values()) or 1
                    concentration = max(m.value.values()) / total
                for name, value, kind in [
                    ("derived.fanout", fanout, MetricKind.GAUGE),
                    ("derived.peer_novelty", novelty, MetricKind.RATE),
                    ("derived.new_peer_count", float(len(new_peers)), MetricKind.GAUGE),
                    ("derived.dest_concentration", concentration, MetricKind.GAUGE),
                ]:
                    ctx.store.add_derived(DerivedMetric(
                        name=name, value=value, ts=ctx.now, system=system, entity=entity,
                        window_s=ctx.window_s, kind=kind, inputs=["l4.distinct_peers"]))
                    n += 1
        return n
