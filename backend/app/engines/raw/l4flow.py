"""L4 flow raw-metric engine — transport layer.

Acquisition: passive SPAN decode or NetFlow/IPFIX. Emits per-entity flow
volumes and directionality, plus TCP health signals (retransmits, RTT,
window) that need no host agent to observe.

v2 (lib-3, R1). Why each rule exists:

* Weighted records. A record may stand for w = extra['count'] flows (IPFIX
  aggregates, the generator's aggregated mode at dt >= 900 s). Every counter
  adds w, bytes come from extra['bytes_up_total'/'bytes_down_total'] when
  present (else w * bytes), averages are w-weighted. Without this, 15-minute
  aggregated ticks would look ~50x quieter than 60 s ticks.
* Full sets. l4.dport_set / l4.peer_set keep the top 64 values plus
  '__other__', so the values always sum to the true flow count (novelty,
  the feature sketch and breadth budgets need the tail, not a top 8).
  Peers are keyed by /24 (IPv6: /64) or by the normalised service host.
* Absence is data. Metrics are emitted only for entities that had records
  this tick, stamped ts = ctx.now (add_raw then maintains last_seen). An
  average of nothing (no RTT / window / duration sample) is not emitted:
  0.0 would be a fake measurement, while absence reads as NaN downstream.
* Our own active probes (method ACTIVE_*) are not the entity's traffic and
  are left to ActiveProbeEngine.
* Pseudo-entities ('__*', 'class:*') never carry raw data: such records are
  dropped and counted in this engine's health record (dropped_pseudo).
"""
from __future__ import annotations

import ipaddress
import math
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import (AcquisitionMethod, MetricKind, Observation, RawMetric,
                              is_pseudo_entity)
from ..behavior.lib.names import normalize_host

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


@lru_cache(maxsize=1 << 14)
def peer_key(peer: str) -> str:
    """Peer bucket for l4.peer_set: IPv4 /24, IPv6 /64, else the normalised host."""
    h = normalize_host(peer)
    if not h:
        return ""
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:                        # not an IP literal: a service host
        return h
    plen = 24 if ip.version == 4 else 64
    return str(ipaddress.ip_network(f"{h}/{plen}", strict=False))


class _Acc:
    __slots__ = ("flows", "up", "down", "pkts", "retx", "syn", "rtt_s", "rtt_w",
                 "win_s", "win_w", "dur_s", "dur_w", "peers", "dports")

    def __init__(self) -> None:
        self.flows = 0
        self.up = self.down = self.pkts = self.retx = 0.0
        self.syn = 0
        self.rtt_s = self.win_s = self.dur_s = 0.0
        self.rtt_w = self.win_w = self.dur_w = 0
        self.peers: Dict[str, int] = {}
        self.dports: Dict[int, int] = {}


class L4FlowEngine(Engine):
    name = "raw.l4flow"
    layer = "raw"
    consumes = ["<observations>"]
    produces = [
        "l4.flows", "l4.bytes_up", "l4.bytes_down", "l4.updown_ratio",
        "l4.distinct_peers", "l4.distinct_dports", "l4.syn_count", "l4.pkts_total",
        "l4.retransmit_rate", "l4.rtt_ms_avg", "l4.win_size_avg",
        "l4.flow_duration_ms_avg", "l4.dport_set", "l4.peer_set",
    ]
    description = ("TCP/UDP flow volume, directionality, fan-out (full port / peer sets) and "
                   "TCP health from weighted passive flow records.")

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
            a.up += up
            a.down += down
            x = o.pkts_up + o.pkts_down
            if 0 < x < _INF:
                a.pkts += w * x
            x = o.retransmits
            if 0 < x < _INF:
                a.retx += w * x
            fl = o.tcp_flags
            if fl:
                fl = fl.upper()
                if "SYN" in fl and "ACK" not in fl:
                    a.syn += w
            x = o.rtt_ms
            if 0 < x < _INF:
                a.rtt_s += w * x
                a.rtt_w += w
            x = o.win_size
            if 0 < x < _INF:
                a.win_s += w * x
                a.win_w += w
            x = o.duration_ms
            if 0 < x < _INF:
                a.dur_s += w * x
                a.dur_w += w
            p = o.peer
            if p:
                a.peers[p] = a.peers.get(p, 0) + w
            x = o.dst_port
            if 0 < x < 65536:
                if x.__class__ is not int:
                    x = int(x)
                a.dports[x] = a.dports.get(x, 0) + w

        add = ctx.store.add_raw
        now = ctx.now
        method = AcquisitionMethod.PASSIVE_FLOW
        C, G, R, K = MetricKind.COUNTER, MetricKind.GAUGE, MetricKind.RATE, MetricKind.CATEGORICAL
        n = 0
        for (system, entity), a in acc.items():
            if a is None or a.flows <= 0:
                continue
            up, down = a.up, a.down
            updown = up / down if down > 0 else (up if up > 0 else 0.0)
            out: List[Tuple[str, Any, MetricKind, str]] = [
                ("l4.flows", float(a.flows), C, "flows"),
                ("l4.bytes_up", up, C, "bytes"),
                ("l4.bytes_down", down, C, "bytes"),
                ("l4.updown_ratio", updown, G, "ratio"),
                ("l4.distinct_peers", float(len(a.peers)), G, "peers"),
                ("l4.distinct_dports", float(len(a.dports)), G, "ports"),
                ("l4.syn_count", float(a.syn), C, "flows"),
                ("l4.pkts_total", a.pkts, C, "packets"),
            ]
            if a.pkts > 0:
                out.append(("l4.retransmit_rate", a.retx / a.pkts, R, "ratio"))
            if a.rtt_w:
                out.append(("l4.rtt_ms_avg", a.rtt_s / a.rtt_w, G, "ms"))
            if a.win_w:
                out.append(("l4.win_size_avg", a.win_s / a.win_w, G, "bytes"))
            if a.dur_w:
                out.append(("l4.flow_duration_ms_avg", a.dur_s / a.dur_w, G, "ms"))
            if a.dports:
                out.append(("l4.dport_set",
                            _full_set({str(p): c for p, c in a.dports.items()}), K, ""))
            if a.peers:
                ps: Dict[str, int] = {}
                for p, c in a.peers.items():
                    pk = peer_key(p)
                    if pk:
                        ps[pk] = ps.get(pk, 0) + c
                if ps:
                    out.append(("l4.peer_set", _full_set(ps), K, ""))
            for mname, value, kind, unit in out:
                add(RawMetric(name=mname, value=value, ts=now, system=system, entity=entity,
                              kind=kind, method=method, unit=unit))
            n += len(out)
        return n
