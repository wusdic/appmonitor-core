"""Graph engine — communication-graph shape per entity.

Builds a per-tick peer view of each entity from its fresh TLS/DNS name sets
(at eTLD+1, so CDN shards and random sub-labels count once) and its flow
peer buckets (l4.peer_set: /24, /64 or service host), then derives fan-out,
new peers vs. the entity's recent history and destination concentration.

v2 (D1):

* Freshness. Every output is written only when its inputs were written
  this tick; a silent entity gets nothing (absence is data), where v1
  re-emitted fan-out and a 0.0 novelty forever.
* Forgetting. v1's seen-set never decayed, so a peer an attacker once used
  was "known" forever, and memory grew without bound. The history is now
  {peer: last_seen_ts}; a peer not seen for 30 days is new again, and such
  entries are swept so the dict stays bounded by what was seen in 30 days
  (plus a hard per-entity cap).
* derived.fanout and derived.peer_novelty are kept for library-4 rules
  only; FEATURE_SPEC v2 uses l4.distinct_peers and B08 novelty instead.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

from ...core.engine import Context, Engine
from ...models.schema import DerivedMetric, MetricKind
from ..behavior.lib.names import etld1 as _etld1
from .fresh import fresh_raw, fresh_raw_obj

OTHER = "__other__"
DAY = 86400.0
PEER_TTL_S = 30 * DAY          # a peer unseen this long counts as new again
SWEEP_EVERY_S = DAY            # expiry sweep cadence per entity (wall clock)
MAX_PEERS = 20000              # hard cap per entity; oldest half dropped beyond it

# (set already at eTLD+1, full-name fallback mapped through etld1)
NAME_SOURCES: Tuple[Tuple[str, str], ...] = (
    ("tls.sni_etld1_set", "tls.sni_set"),
    ("dns.qname_etld1_set", "dns.qname_set"),
)
ADDR_SOURCE = "l4.peer_set"

# host names repeat tick after tick; the PSL walk is the hot spot of this engine
etld1 = lru_cache(maxsize=1 << 16)(_etld1)


def _keys(value) -> List[str]:
    """Named keys of a raw set with a positive count ('__other__' excluded)."""
    if not isinstance(value, dict):
        return []
    return [str(k) for k, v in value.items()
            if k != OTHER and isinstance(v, (int, float)) and v > 0]


def concentration(value) -> Optional[float]:
    """Share of the single top named value in the set's true total."""
    if not isinstance(value, dict):
        return None
    total, top = 0.0, 0.0
    for k, v in value.items():
        if not isinstance(v, (int, float)) or not (v > 0 and math.isfinite(v)):
            continue
        total += v
        if k != OTHER and v > top:
            top = float(v)
    if total <= 0 or top <= 0:
        return None
    return top / total


class GraphEngine(Engine):
    name = "derived.graph"
    layer = "derived"
    consumes = ["l4.distinct_peers", "l4.peer_set", "tls.sni_set", "tls.sni_etld1_set",
                "dns.qname_set", "dns.qname_etld1_set"]
    produces = [
        "derived.fanout", "derived.peer_novelty",
        "derived.dest_concentration", "derived.new_peer_count",
    ]
    description = ("Fan-out, new peers vs. a 30-day decayed history and destination "
                   "concentration, from fresh inputs only.")

    def __init__(self, **p):
        super().__init__(**p)
        self.ttl_s = float(p.get("peer_ttl_s", PEER_TTL_S))
        self.max_peers = int(p.get("max_peers", MAX_PEERS))
        self._seen: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._swept: Dict[Tuple[str, str], float] = {}
        self._gswept: Optional[float] = None

    # ------------------------------------------------------------ peers
    def _peers(self, store, system: str, entity: str, now: float) -> Tuple[set, List[str]]:
        peers: set = set()
        inputs: List[str] = []
        for reg_name, full_name in NAME_SOURCES:
            v = fresh_raw_obj(store, system, entity, reg_name, now)
            if isinstance(v, dict):
                peers.update(_keys(v))
                inputs.append(reg_name)
                continue
            v = fresh_raw_obj(store, system, entity, full_name, now)
            if isinstance(v, dict):
                peers.update(etld1(k) for k in _keys(v))
                inputs.append(full_name)
        v = fresh_raw_obj(store, system, entity, ADDR_SOURCE, now)
        if isinstance(v, dict):
            peers.update(_keys(v))
            inputs.append(ADDR_SOURCE)
        peers.discard("")
        return peers, inputs

    def _learn(self, key: Tuple[str, str], peers: Iterable[str], now: float) -> int:
        """Count peers new within the TTL, then record them as seen at now."""
        seen = self._seen.get(key)
        if seen is None:
            seen = self._seen[key] = {}
        ttl = self.ttl_s
        new = 0
        for p in peers:
            last = seen.get(p)
            # a clock that went backwards (replay) leaves now - last < 0: seen
            if last is None or now - last > ttl:
                new += 1
            if last is None or now > last:
                seen[p] = now
        self._sweep(key, seen, now)
        return new

    def _sweep(self, key: Tuple[str, str], seen: Dict[str, float], now: float) -> None:
        last = self._swept.get(key)
        over = len(seen) > self.max_peers
        if not over and last is not None and 0 <= now - last < SWEEP_EVERY_S:
            return
        self._swept[key] = now
        cutoff = now - self.ttl_s
        for p in [p for p, t in seen.items() if t < cutoff]:
            del seen[p]
        if len(seen) > self.max_peers:
            keep = sorted(seen.items(), key=lambda kv: kv[1])[len(seen) - self.max_peers // 2:]
            seen.clear()
            seen.update(keep)

    def _sweep_idle(self, now: float) -> None:
        """Drop the whole history of entities silent for longer than the TTL
        (their per-entity sweep only runs when they are active)."""
        if self._gswept is not None and 0 <= now - self._gswept < SWEEP_EVERY_S:
            return
        self._gswept = now
        cutoff = now - self.ttl_s
        for key in [k for k, t in self._swept.items() if t < cutoff]:
            seen = self._seen.get(key)
            if not seen or max(seen.values()) < cutoff:
                self._seen.pop(key, None)
                self._swept.pop(key, None)

    def seen_peers(self, system: str, entity: str) -> Dict[str, float]:
        """Copy of the decayed history (tests / explain)."""
        return dict(self._seen.get((system, entity), {}))

    # -------------------------------------------------------------- run
    def run(self, ctx: Context, observations=None) -> int:
        store, now = ctx.store, ctx.now
        ws = int(ctx.window_s)
        n = 0
        self._sweep_idle(now)
        for system in store.systems():
            for entity in store.entities_active(system, now):
                out: List[Tuple[str, float, MetricKind, List[str]]] = []
                fanout = fresh_raw(store, system, entity, "l4.distinct_peers", now)
                if fanout is not None:
                    out.append(("derived.fanout", fanout, MetricKind.GAUGE,
                                ["l4.distinct_peers"]))
                peers, inputs = self._peers(store, system, entity, now)
                if inputs:
                    new = self._learn((system, entity), peers, now)
                    out.append(("derived.new_peer_count", float(new), MetricKind.GAUGE, inputs))
                    if peers:
                        out.append(("derived.peer_novelty", new / len(peers),
                                    MetricKind.RATE, inputs))
                # destination concentration on full names, TLS first (v1 order)
                for src in ("tls.sni_set", "dns.qname_set"):
                    c = concentration(fresh_raw_obj(store, system, entity, src, now))
                    if c is not None:
                        out.append(("derived.dest_concentration", c, MetricKind.GAUGE, [src]))
                        break
                for name, value, kind, inp in out:
                    store.add_derived(DerivedMetric(
                        name=name, value=float(value), ts=now, system=system, entity=entity,
                        window_s=ws, kind=kind, inputs=inp))
                    n += 1
        return n
