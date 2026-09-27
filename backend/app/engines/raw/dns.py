"""DNS raw-metric engine.

Acquisition: passive SPAN decode of port-53 traffic, or off-host resolver
query logs. Emits query volume, type mix, failure (NXDOMAIN/SERVFAIL) rate,
and the queried-name set — inputs for the entropy/DGA-detection derived
engine and for signature matching (tunneling, beaconing to odd domains).

v2 (lib-3, R1). Why each rule exists:

* Weighted records: a record may stand for w = extra['count'] queries
  (aggregated resolver logs, the generator's aggregated mode); counters add
  w and the qname-length average is w-weighted.
* Full sets: qname / qtype sets keep the top 64 values plus '__other__', so
  the values sum to the true query count (the v1 top 12 hid exactly the
  long tail a DGA or tunnel produces). Qnames are normalised (DNS is case
  insensitive: lower case, no trailing dot) and additionally aggregated at
  eTLD+1 (dns.qname_etld1_set): a tunnel's thousand random labels under one
  domain are one destination for novelty and the feature sketch.
* Counts next to ratios: dns.txt_count and dns.qname_len_avg are what
  FEATURE_SPEC v2 reads; the v1 dns.txt_ratio / dns.avg_qname_len are kept
  (same values, lib-4 signatures use them).
* Absence is data: metrics only for entities with queries this tick,
  ts = ctx.now; a qname-length average over no named query is not emitted.
  Pseudo-entities ('__*', 'class:*') are dropped and counted in this
  engine's health record (dropped_pseudo).
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from ...core.engine import Context, Engine
from ...models.schema import (AcquisitionMethod, MetricKind, Observation, RawMetric,
                              is_pseudo_entity)
from ..behavior.lib.names import etld1

TOP_K = 64
OTHER = "__other__"
_ACTIVE = frozenset({AcquisitionMethod.ACTIVE_PROBE, AcquisitionMethod.ACTIVE_DNS,
                     AcquisitionMethod.ACTIVE_TLS})
_FAIL_RCODES = frozenset({"NXDOMAIN", "SERVFAIL", "3", "2"})   # names and numeric codes
_TXT = frozenset({"TXT", "16"})


# ------------------------------------------------------------------ helpers
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
def _etld1(host: str) -> str:
    return etld1(host)


class _Acc:
    __slots__ = ("q", "fail", "txt", "len_s", "len_w", "qnames", "qtypes")

    def __init__(self) -> None:
        self.q = self.fail = self.txt = 0
        self.len_s = 0.0
        self.len_w = 0
        self.qnames: Dict[str, int] = {}
        self.qtypes: Dict[str, int] = {}


class DNSEngine(Engine):
    name = "raw.dns"
    layer = "raw"
    consumes = ["<observations>"]
    produces = [
        "dns.queries", "dns.distinct_qnames", "dns.nxdomain_ratio",
        "dns.txt_ratio", "dns.txt_count", "dns.avg_qname_len", "dns.qname_len_avg",
        "dns.qname_set", "dns.qname_etld1_set", "dns.qtype_set",
    ]
    description = ("DNS query volume, type mix, failure rate and full queried-name sets "
                   "(also at eTLD+1) from weighted passive decode.")

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
            if (o.app_proto != "dns" and not o.dns_qname) or o.method in _ACTIVE:
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
            else:
                w = 1
            a.q += w
            rc = o.dns_rcode
            if rc and rc.upper() in _FAIL_RCODES:
                a.fail += w
            qt = o.dns_qtype
            if qt:
                qt = qt.upper()
                a.qtypes[qt] = a.qtypes.get(qt, 0) + w
                if qt in _TXT:
                    a.txt += w
            qn = o.dns_qname
            if qn:
                qn = qn.strip().lower().rstrip(".")
                if qn:
                    a.len_s += w * len(qn)
                    a.len_w += w
                    a.qnames[qn] = a.qnames.get(qn, 0) + w

        add = ctx.store.add_raw
        now = ctx.now
        method = AcquisitionMethod.PASSIVE_SPAN
        C, G, R, K = MetricKind.COUNTER, MetricKind.GAUGE, MetricKind.RATE, MetricKind.CATEGORICAL
        n = 0
        for (system, entity), a in acc.items():
            if a is None or a.q <= 0:
                continue
            q = float(a.q)
            out: List[Tuple[str, Any, MetricKind, str]] = [
                ("dns.queries", q, C, "queries"),
                ("dns.distinct_qnames", float(len(a.qnames)), G, "names"),
                ("dns.nxdomain_ratio", a.fail / q, R, "ratio"),
                ("dns.txt_ratio", a.txt / q, R, "ratio"),
                ("dns.txt_count", float(a.txt), C, "queries"),
            ]
            if a.len_w:
                avg = a.len_s / a.len_w
                out.append(("dns.qname_len_avg", avg, G, "chars"))
                out.append(("dns.avg_qname_len", avg, G, "chars"))
            if a.qnames:
                reg: Dict[str, int] = {}
                for qn, c in a.qnames.items():
                    e = _etld1(qn)
                    if e:
                        reg[e] = reg.get(e, 0) + c
                out.append(("dns.qname_set", _full_set(a.qnames), K, ""))
                if reg:
                    out.append(("dns.qname_etld1_set", _full_set(reg), K, ""))
            if a.qtypes:
                out.append(("dns.qtype_set", _full_set(a.qtypes), K, ""))
            for mname, value, kind, unit in out:
                add(RawMetric(name=mname, value=value, ts=now, system=system, entity=entity,
                              kind=kind, method=method, unit=unit))
            n += len(out)
        return n
