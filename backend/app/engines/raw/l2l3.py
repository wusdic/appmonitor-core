"""L2/L3 raw-metric engine — link & network layer.

Acquisition: passive SPAN/TAP decode (or flow records). No host agent.
Produces per-entity, per-tick network-layer facts: packet/byte volumes,
protocol mix, TTL (an OS / hop-distance hint), and IPv4/IPv6 split.

v2 (lib-3, R1). Why each rule exists:

* Weighted records: a record may stand for w = extra['count'] flows
  (IPFIX-style aggregates, the generator's aggregated mode). Packet and flow
  counts add w, bytes come from extra['bytes_up_total'/'bytes_down_total']
  when present (else w * bytes), and the protocol / IPv6 shares are over
  weighted flows, so l3.bytes_total agrees with l4.bytes_up + l4.bytes_down.
* Absence is data: metrics only for entities with records this tick, stamped
  ts = ctx.now; l3.ttl_min only when some record carried a valid TTL.
* Our own active probes (method ACTIVE_*) are not the entity's traffic.
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


class _Acc:
    __slots__ = ("flows", "pkts", "bytes", "tcp", "udp", "icmp", "ipv6", "ttl_min")

    def __init__(self) -> None:
        self.flows = self.tcp = self.udp = self.icmp = self.ipv6 = 0
        self.pkts = self.bytes = 0.0
        self.ttl_min = _INF


class L2L3Engine(Engine):
    name = "raw.l2l3"
    layer = "raw"
    consumes = ["<observations>"]
    produces = [
        "l3.pkts_total", "l3.bytes_total", "l3.proto.tcp_ratio",
        "l3.proto.udp_ratio", "l3.proto.icmp_ratio", "l3.ttl_min",
        "l3.ipv6_ratio",
    ]
    description = ("Link/network-layer volumes, protocol mix and TTL from weighted "
                   "passive records.")

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
        acc: Dict[Tuple[str, str], Optional[_Acc]] = {}
        for o in observations or ():
            if o.method in _ACTIVE:
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
            a.flows += w
            a.bytes += up + down
            x = o.pkts_up + o.pkts_down
            if 0 < x < _INF:
                a.pkts += w * x
            p = o.l4_proto
            if p:
                if p != "tcp" and p != "udp":
                    p = p.lower()
                if p == "tcp":
                    a.tcp += w
                elif p == "udp":
                    a.udp += w
                elif p == "icmp" or p == "icmpv6":
                    a.icmp += w
            l3 = o.l3_proto
            if l3 and (l3 == "ipv6" or l3.lower() == "ipv6"):
                a.ipv6 += w
            t = o.ttl
            if 0 < t <= 255 and t < a.ttl_min:        # 0 = not decoded
                a.ttl_min = t

        add = ctx.store.add_raw
        now = ctx.now
        method = AcquisitionMethod.PASSIVE_SPAN
        C, G, R = MetricKind.COUNTER, MetricKind.GAUGE, MetricKind.RATE
        n = 0
        for (system, entity), a in acc.items():
            if a is None or a.flows <= 0:
                continue
            flows = float(a.flows)
            out: List[Tuple[str, float, MetricKind, str]] = [
                ("l3.pkts_total", a.pkts, C, "packets"),
                ("l3.bytes_total", a.bytes, C, "bytes"),
                ("l3.proto.tcp_ratio", a.tcp / flows, R, "ratio"),
                ("l3.proto.udp_ratio", a.udp / flows, R, "ratio"),
                ("l3.proto.icmp_ratio", a.icmp / flows, R, "ratio"),
                ("l3.ipv6_ratio", a.ipv6 / flows, R, "ratio"),
            ]
            if a.ttl_min < _INF:
                out.append(("l3.ttl_min", float(a.ttl_min), G, "hops"))
            for mname, value, kind, unit in out:
                add(RawMetric(name=mname, value=value, ts=now, system=system, entity=entity,
                              kind=kind, method=method, unit=unit))
            n += len(out)
        return n
