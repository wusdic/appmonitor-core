"""TLS raw-metric engine — encrypted-traffic metadata.

Acquisition: passive SPAN decode of the TLS handshake (which is cleartext up
to ChangeCipherSpec). No decryption, no host agent. Emits handshake counts,
version/cipher posture, SNI diversity, and JA3/JA3S client/server fingerprints
— the levers that let us profile encrypted sessions without touching payload.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ...core.engine import Context, Engine
from ...models.schema import AcquisitionMethod, MetricKind, Observation, RawMetric


_WEAK_VERSIONS = {"SSL3.0", "TLS1.0", "TLS1.1"}


class TLSEngine(Engine):
    name = "raw.tls"
    layer = "raw"
    consumes = ["<observations>"]
    produces = [
        "tls.handshakes", "tls.distinct_sni", "tls.weak_version_ratio",
        "tls.handshake_ms_avg", "tls.ja3_set", "tls.ja3s_set",
        "tls.sni_set", "tls.cipher_set",
    ]
    description = "TLS handshake metadata: version posture, SNI diversity, JA3/JA3S fingerprints."

    def run(self, ctx: Context, observations: Optional[List[Observation]] = None) -> int:
        observations = observations or []
        agg: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        sni: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        ja3: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        ja3s: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        ciphers: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for o in observations:
            if o.app_proto != "tls" and not o.tls_version:
                continue
            key = (o.system, o.entity)
            a = agg[key]
            a["hs"] += 1
            if o.tls_version in _WEAK_VERSIONS:
                a["weak"] += 1
            if o.duration_ms:
                a["hs_ms_sum"] += o.duration_ms
                a["hs_ms_n"] += 1
            if o.tls_sni:
                sni[key][o.tls_sni] += 1
            if o.ja3:
                ja3[key][o.ja3] += 1
            if o.ja3s:
                ja3s[key][o.ja3s] += 1
            if o.tls_cipher:
                ciphers[key][o.tls_cipher] += 1

        n = 0
        for key, a in agg.items():
            system, entity = key
            hs = max(a["hs"], 1.0)
            numeric = [
                ("tls.handshakes", a["hs"], MetricKind.COUNTER, "handshakes"),
                ("tls.distinct_sni", float(len(sni[key])), MetricKind.GAUGE, "hosts"),
                ("tls.weak_version_ratio", a.get("weak", 0) / hs, MetricKind.RATE, "ratio"),
                ("tls.handshake_ms_avg",
                 a["hs_ms_sum"] / a["hs_ms_n"] if a.get("hs_ms_n") else 0.0,
                 MetricKind.GAUGE, "ms"),
            ]
            for mname, value, kind, unit in numeric:
                ctx.store.add_raw(RawMetric(
                    name=mname, value=value, ts=ctx.now, system=system, entity=entity,
                    kind=kind, method=AcquisitionMethod.PASSIVE_SPAN, unit=unit))
                n += 1
            for mname, dist in [("tls.sni_set", sni[key]), ("tls.ja3_set", ja3[key]),
                                ("tls.ja3s_set", ja3s[key]), ("tls.cipher_set", ciphers[key])]:
                top = dict(sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:8])
                if top:
                    ctx.store.add_raw(RawMetric(
                        name=mname, value=top, ts=ctx.now, system=system, entity=entity,
                        kind=MetricKind.CATEGORICAL, method=AcquisitionMethod.PASSIVE_SPAN))
                    n += 1
        return n
