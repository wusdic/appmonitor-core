"""DNS raw-metric engine.

Acquisition: passive SPAN decode of port-53 traffic, or off-host resolver
query logs. Emits query volume, type mix, failure (NXDOMAIN/SERVFAIL) rate,
and the queried-name set — inputs for the entropy/DGA-detection derived
engine and for signature matching (tunneling, beaconing to odd domains).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ...core.engine import Context, Engine
from ...models.schema import AcquisitionMethod, MetricKind, Observation, RawMetric


class DNSEngine(Engine):
    name = "raw.dns"
    layer = "raw"
    consumes = ["<observations>"]
    produces = [
        "dns.queries", "dns.distinct_qnames", "dns.nxdomain_ratio",
        "dns.txt_ratio", "dns.avg_qname_len", "dns.qname_set", "dns.qtype_set",
    ]
    description = "DNS query volume, type mix, failure rate and queried-name set from passive decode."

    def run(self, ctx: Context, observations: Optional[List[Observation]] = None) -> int:
        observations = observations or []
        agg: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        qnames: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        qtypes: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for o in observations:
            if o.app_proto != "dns" and not o.dns_qname:
                continue
            key = (o.system, o.entity)
            a = agg[key]
            a["q"] += 1
            if o.dns_rcode in ("NXDOMAIN", "SERVFAIL"):
                a["fail"] += 1
            if o.dns_qtype == "TXT":
                a["txt"] += 1
            if o.dns_qname:
                a["qname_len_sum"] += len(o.dns_qname)
                qnames[key][o.dns_qname] += 1
            if o.dns_qtype:
                qtypes[key][o.dns_qtype] += 1

        n = 0
        for key, a in agg.items():
            system, entity = key
            q = max(a["q"], 1.0)
            numeric = [
                ("dns.queries", a["q"], MetricKind.COUNTER, "queries"),
                ("dns.distinct_qnames", float(len(qnames[key])), MetricKind.GAUGE, "names"),
                ("dns.nxdomain_ratio", a.get("fail", 0) / q, MetricKind.RATE, "ratio"),
                ("dns.txt_ratio", a.get("txt", 0) / q, MetricKind.RATE, "ratio"),
                ("dns.avg_qname_len", a["qname_len_sum"] / q, MetricKind.GAUGE, "chars"),
            ]
            for mname, value, kind, unit in numeric:
                ctx.store.add_raw(RawMetric(
                    name=mname, value=value, ts=ctx.now, system=system, entity=entity,
                    kind=kind, method=AcquisitionMethod.PASSIVE_SPAN, unit=unit))
                n += 1
            for mname, dist in [("dns.qname_set", qnames[key]), ("dns.qtype_set", qtypes[key])]:
                top = dict(sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:12])
                if top:
                    ctx.store.add_raw(RawMetric(
                        name=mname, value=top, ts=ctx.now, system=system, entity=entity,
                        kind=MetricKind.CATEGORICAL, method=AcquisitionMethod.PASSIVE_SPAN))
                    n += 1
        return n
