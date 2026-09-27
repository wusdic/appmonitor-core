"""TLS raw-metric engine — encrypted-traffic metadata.

Acquisition: passive SPAN decode of the TLS handshake (which is cleartext up
to ChangeCipherSpec). No decryption, no host agent. Emits handshake counts,
version/cipher posture, SNI diversity, and JA3/JA3S client/server fingerprints
— the levers that let us profile encrypted sessions without touching payload.

v2 (lib-3, R1). Why each rule exists:

* Weighted records: a record may stand for w = extra['count'] handshakes;
  counters add w and the handshake-time average is w-weighted.
* Full sets: SNI / JA3 / JA3S / cipher sets keep the top 64 values plus
  '__other__' so the values sum to the true handshake count. SNIs are
  normalised (lower case, no trailing dot) and additionally aggregated at
  eTLD+1 (tls.sni_etld1_set), so CDN shards of one organisation count as one
  destination for novelty and the feature sketch.
* Handshake time: extra['handshake_ms'] when the decoder provides it,
  otherwise duration_ms of a record that is TLS without L7 decode. For an
  HTTP-over-TLS record duration_ms is the request latency, not the
  handshake, so it is not used; a handshake average over no sample is not
  emitted (absence reads as NaN downstream, never a fake 0).
* Absence is data: metrics only for entities with handshakes this tick,
  ts = ctx.now. Pseudo-entities ('__*', 'class:*') are dropped and counted
  in this engine's health record (dropped_pseudo).
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import (AcquisitionMethod, MetricKind, Observation, RawMetric,
                              is_pseudo_entity)
from ..behavior.lib.names import etld1

TOP_K = 64
OTHER = "__other__"
_ACTIVE = frozenset({AcquisitionMethod.ACTIVE_PROBE, AcquisitionMethod.ACTIVE_DNS,
                     AcquisitionMethod.ACTIVE_TLS})
_NUM = (int, float, np.integer, np.floating)
_INF = math.inf

# Weak protocol versions, after normalisation ('TLSv1.0' -> 'TLS1.0', 'SSLv3' -> 'SSL3').
_WEAK_VERSIONS = frozenset({"SSL2", "SSL2.0", "SSL3", "SSL3.0", "TLS1", "TLS1.0", "TLS1.1"})
_TLS_PROTOS = frozenset({"tls", "https", "ssl"})      # as R2
_NO_L7 = frozenset({"", "tls", "https", "ssl"})


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


def _extra_ms(x: Any) -> float:
    """A decoder-supplied duration from extra (untyped): finite and > 0, else 0.0."""
    if isinstance(x, _NUM) and not isinstance(x, bool) and 0 < x < _INF:
        return float(x)
    return 0.0


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


@lru_cache(maxsize=256)
def _weak(version: str) -> bool:
    v = version.upper().replace(" ", "").replace("_", "").replace("V", "")
    return v in _WEAK_VERSIONS


def _by_etld1(names: Dict[str, int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for h, c in names.items():
        e = _etld1(h)
        if e:
            out[e] = out.get(e, 0) + c
    return out


class _Acc:
    __slots__ = ("hs", "weak", "ms_s", "ms_w", "sni", "ja3", "ja3s", "ciphers")

    def __init__(self) -> None:
        self.hs = self.weak = 0
        self.ms_s = 0.0
        self.ms_w = 0
        self.sni: Dict[str, int] = {}
        self.ja3: Dict[str, int] = {}
        self.ja3s: Dict[str, int] = {}
        self.ciphers: Dict[str, int] = {}


class TLSEngine(Engine):
    name = "raw.tls"
    layer = "raw"
    consumes = ["<observations>"]
    produces = [
        "tls.handshakes", "tls.distinct_sni", "tls.weak_version_ratio",
        "tls.handshake_ms_avg", "tls.ja3_set", "tls.ja3s_set",
        "tls.sni_set", "tls.sni_etld1_set", "tls.cipher_set",
    ]
    description = ("TLS handshake metadata: version posture, SNI diversity (full and eTLD+1 "
                   "sets), JA3/JA3S fingerprints, handshake time.")

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
            ap = o.app_proto
            if not (o.tls_version or o.tls_sni or ap in _TLS_PROTOS) or o.method in _ACTIVE:
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
            x = 0.0
            if ex:
                w = _weight(ex)
                if w <= 0:
                    continue
                x = _extra_ms(ex.get("handshake_ms"))
            else:
                w = 1
            a.hs += w
            v = o.tls_version
            if v and _weak(v):
                a.weak += w
            if not x and not o.http_method and ap in _NO_L7:
                x = o.duration_ms                 # TLS without L7: the record is the session
            if 0 < x < _INF:
                a.ms_s += w * x
                a.ms_w += w
            s = o.tls_sni
            if s:
                s = s.strip().lower().rstrip(".")
                if s:
                    a.sni[s] = a.sni.get(s, 0) + w
            j = o.ja3
            if j:
                a.ja3[j] = a.ja3.get(j, 0) + w
            j = o.ja3s
            if j:
                a.ja3s[j] = a.ja3s.get(j, 0) + w
            c = o.tls_cipher
            if c:
                a.ciphers[c] = a.ciphers.get(c, 0) + w

        add = ctx.store.add_raw
        now = ctx.now
        method = AcquisitionMethod.PASSIVE_SPAN
        C, G, R, K = MetricKind.COUNTER, MetricKind.GAUGE, MetricKind.RATE, MetricKind.CATEGORICAL
        n = 0
        for (system, entity), a in acc.items():
            if a is None or a.hs <= 0:
                continue
            hs = float(a.hs)
            out: List[Tuple[str, Any, MetricKind, str]] = [
                ("tls.handshakes", hs, C, "handshakes"),
                ("tls.distinct_sni", float(len(a.sni)), G, "hosts"),
                ("tls.weak_version_ratio", a.weak / hs, R, "ratio"),
            ]
            if a.ms_w:
                out.append(("tls.handshake_ms_avg", a.ms_s / a.ms_w, G, "ms"))
            if a.sni:
                out.append(("tls.sni_set", _full_set(a.sni), K, ""))
                out.append(("tls.sni_etld1_set", _full_set(_by_etld1(a.sni)), K, ""))
            for mname, dist in (("tls.ja3_set", a.ja3), ("tls.ja3s_set", a.ja3s),
                                ("tls.cipher_set", a.ciphers)):
                if dist:
                    out.append((mname, _full_set(dist), K, ""))
            for mname, value, kind, unit in out:
                add(RawMetric(name=mname, value=value, ts=now, system=system, entity=entity,
                              kind=kind, method=method, unit=unit))
            n += len(out)
        return n
