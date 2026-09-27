"""Entropy engine — dispersion of categorical activity.

High destination/path/domain entropy separates scanners and tunnels from
focused business use. Character-level entropy of queried domain names flags
algorithmically-generated (DGA) or data-in-DNS names. Reads the CATEGORICAL
raw sets ({value: count}, top 64 plus '__other__', R1).

v2 (D1):

* Freshness. A value is written only when its set was written this tick;
  v1 re-emitted the last set of a silent entity forever.
* Full sets. The 64 named entries are the distribution. '__other__' is not a
  category (it is the remainder of many), so it is left out of the entropy:
  counting it as one value would make a tunnel with thousands of random
  names look *concentrated*. Its mass still counts towards exposure.
* Exposure. Each value also writes `derived.<x>_n`, the number of items the
  set describes (sum of its counts, '__other__' included, which R1 makes the
  true request / query / handshake count). B01 gates bounded descriptors on
  n >= 5: the entropy of three requests means nothing.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from ...core.engine import Context, Engine
from ...models.schema import DerivedMetric, MetricKind
from .fresh import fresh_raw_obj
from .util import char_entropy, normalized_entropy

OTHER = "__other__"

# out name -> source set
SOURCES: Dict[str, str] = {
    "derived.sni_entropy": "tls.sni_set",
    "derived.path_entropy": "http.top_paths",
    "derived.ja3_diversity": "tls.ja3_set",
    "derived.dns_name_entropy": "dns.qname_set",
    "derived.dns_dga_score": "dns.qname_set",
}


def split_set(value) -> Optional[Tuple[Dict[str, float], float]]:
    """(named entries with finite positive counts, total mass incl. __other__)
    for a raw categorical set, or None if it is not a usable set."""
    if not isinstance(value, dict) or not value:
        return None
    named: Dict[str, float] = {}
    total = 0.0
    for k, v in value.items():
        if not isinstance(v, (int, float)):
            continue
        c = float(v)
        if not (c > 0 and math.isfinite(c)):
            continue
        total += c
        if k != OTHER:
            named[str(k)] = c
    if total <= 0:
        return None
    return named, total


@lru_cache(maxsize=1 << 16)
def _label_entropy(qname: str) -> float:
    """Character entropy (bits) of the first label; qnames repeat across ticks."""
    return char_entropy(qname.split(".", 1)[0].lower())


def dga_score(qnames: Dict[str, float]) -> float:
    """Count-weighted mean character entropy (bits) of the first label."""
    tot = sum(qnames.values())
    if tot <= 0:
        return math.nan
    return sum(_label_entropy(q) * c for q, c in qnames.items()) / tot


class EntropyEngine(Engine):
    name = "derived.entropy"
    layer = "derived"
    consumes = ["tls.sni_set", "dns.qname_set", "http.top_paths", "tls.ja3_set"]
    produces = [n for name in SOURCES for n in (name, name + "_n")]
    description = ("Shannon entropy of destination/domain/path sets + character-entropy "
                   "DGA score, from fresh sets only, each with its item count _n.")

    def run(self, ctx: Context, observations=None) -> int:
        store, now = ctx.store, ctx.now
        ws = int(ctx.window_s)
        n = 0
        for system in store.systems():
            for entity in store.entities_active(system, now):
                sets: Dict[str, Optional[Tuple[Dict[str, float], float]]] = {}
                emit: List[Tuple[str, float, float, str]] = []
                for out_name, src in SOURCES.items():
                    if src not in sets:
                        sets[src] = split_set(fresh_raw_obj(store, system, entity, src, now))
                    got = sets[src]
                    if got is None:
                        continue
                    named, total = got
                    if out_name == "derived.ja3_diversity":
                        # named fingerprints, +1 when a remainder exists (lower bound)
                        value = float(len(named) + (1 if total > sum(named.values()) else 0))
                    elif out_name == "derived.dns_dga_score":
                        value = dga_score(named)
                    else:
                        value = normalized_entropy(named.values()) if named else math.nan
                    if not math.isfinite(value):
                        continue
                    emit.append((out_name, value, total, src))
                for out_name, value, total, src in emit:
                    store.add_derived(DerivedMetric(
                        name=out_name, value=float(value), ts=now, system=system,
                        entity=entity, window_s=ws, kind=MetricKind.GAUGE, inputs=[src]))
                    store.add_derived(DerivedMetric(
                        name=out_name + "_n", value=float(total), ts=now, system=system,
                        entity=entity, window_s=ws, kind=MetricKind.COUNTER, inputs=[src]))
                    n += 2
        return n
