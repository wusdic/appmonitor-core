"""Entropy engine — dispersion of categorical activity.

High destination/port/domain entropy separates scanners and tunnels from
focused business use. Character-level entropy of queried domain names flags
algorithmically-generated (DGA) or data-in-DNS names. Reads the CATEGORICAL
raw metrics (…_set / …_ratio distributions).
"""
from __future__ import annotations

from typing import Dict

from ...core.engine import Context, Engine
from ...models.schema import DerivedMetric, MetricKind
from .util import char_entropy, normalized_entropy


class EntropyEngine(Engine):
    name = "derived.entropy"
    layer = "derived"
    consumes = ["tls.sni_set", "dns.qname_set", "http.top_paths", "tls.ja3_set"]
    produces = [
        "derived.sni_entropy", "derived.dns_name_entropy",
        "derived.path_entropy", "derived.dns_dga_score", "derived.ja3_diversity",
    ]
    description = "Shannon entropy of destination/domain/path sets + character-entropy DGA score."

    def _cat(self, ctx, system, entity, name) -> Dict[str, float]:
        m = ctx.store.latest_raw(system, entity, name)
        if m and isinstance(m.value, dict):
            return {k: float(v) for k, v in m.value.items()}
        return {}

    def run(self, ctx: Context, observations=None) -> int:
        n = 0
        for system in ctx.store.systems():
            for entity in ctx.store.entities(system):
                sni = self._cat(ctx, system, entity, "tls.sni_set")
                qnames = self._cat(ctx, system, entity, "dns.qname_set")
                paths = self._cat(ctx, system, entity, "http.top_paths")
                ja3 = self._cat(ctx, system, entity, "tls.ja3_set")

                emit = []
                if sni:
                    emit.append(("derived.sni_entropy", normalized_entropy(sni.values()), ["tls.sni_set"]))
                if paths:
                    emit.append(("derived.path_entropy", normalized_entropy(paths.values()), ["http.top_paths"]))
                if ja3:
                    emit.append(("derived.ja3_diversity", float(len(ja3)), ["tls.ja3_set"]))
                if qnames:
                    emit.append(("derived.dns_name_entropy", normalized_entropy(qnames.values()), ["dns.qname_set"]))
                    # DGA: mean per-name character entropy, weighted by count
                    total = sum(qnames.values()) or 1.0
                    dga = sum(char_entropy(name.split(".")[0]) * cnt
                              for name, cnt in qnames.items()) / total
                    emit.append(("derived.dns_dga_score", dga, ["dns.qname_set"]))

                for name, value, inputs in emit:
                    ctx.store.add_derived(DerivedMetric(
                        name=name, value=value, ts=ctx.now, system=system, entity=entity,
                        window_s=ctx.window_s, kind=MetricKind.GAUGE, inputs=inputs))
                    n += 1
        return n
