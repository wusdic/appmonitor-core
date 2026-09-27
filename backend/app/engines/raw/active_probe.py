"""Active-probe raw-metric engine.

Acquisition: probes *we* originate — ICMP echo, TCP connect, banner grab,
HTTP HEAD, TLS cert fetch, traceroute. It never installs anything on the
target; it measures the target from the network.

Cross-network reality is handled as a first-class outcome: when a target sits
behind a boundary we cannot cross, the probe result is UNREACHABLE / TIMEOUT /
FILTERED rather than a missing sample. Downstream engines treat those states
as information (e.g. "server X became unreachable from segment Y") instead of
gaps, and reachability is emitted as its own boolean/categorical metric so a
profile built partly on active data degrades gracefully when a path closes.

v2 (lib-3, R1 conventions):

* probe.probes is the exposure n of probe.loss_ratio (FEATURE_SPEC v2
  probe_loss is a ratio with n = probe.probes). A record may stand for
  w = extra['count'] probes with the same outcome; every count and share is
  w-weighted.
* Presence: a probe is *our* action, not the entity's behaviour. Metrics are
  stamped ts = ctx.now but written with add_raw(touch=False), so probing a
  host every tick does not refresh its last_seen: an unreachable, silent
  server must still look silent to feature.active, rhythm and silence
  detection ("absence is data").
* probe.state is the outcome of the latest probe of the tick (by obs.ts).
* Pseudo-entities ('__*', 'class:*') never carry raw data: such records are
  dropped and counted in this engine's health record (dropped_pseudo).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import (
    AcquisitionMethod,
    MetricKind,
    Observation,
    RawMetric,
    Reachability,
    is_pseudo_entity,
)

_NUM = (int, float, np.integer, np.floating)
_INF = math.inf
_LOST = frozenset({Reachability.UNREACHABLE, Reachability.TIMEOUT, Reachability.FILTERED})


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


class _Acc:
    __slots__ = ("probes", "reach", "lost", "rtt_s", "rtt_w", "hop", "ports",
                 "state", "state_ts")

    def __init__(self) -> None:
        self.probes = self.lost = 0
        self.reach = self.rtt_s = 0.0
        self.rtt_w = 0
        self.hop = 0.0
        self.ports: set = set()
        self.state: Optional[Reachability] = None
        self.state_ts = -_INF


class ActiveProbeEngine(Engine):
    name = "raw.active_probe"
    layer = "raw"
    consumes = ["<observations>"]
    produces = [
        "probe.probes", "probe.reachable", "probe.rtt_ms", "probe.open_ports",
        "probe.hop_count", "probe.state", "probe.loss_ratio",
    ]
    description = ("Active reachability/RTT/open-port/banner probes with "
                   "cross-network unreachable & timeout as first-class states.")

    _STATE_SCORE = {
        Reachability.REACHABLE: 1.0,
        Reachability.DEGRADED: 0.5,
        Reachability.FILTERED: 0.2,
        Reachability.UNREACHABLE: 0.0,
        Reachability.TIMEOUT: 0.0,
    }

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
        score = self._STATE_SCORE
        acc: Dict[Tuple[str, str], Optional[_Acc]] = {}
        for o in observations or ():
            r = o.reachability
            if o.method != AcquisitionMethod.ACTIVE_PROBE or r is None:
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
            a.probes += w
            a.reach += w * score.get(r, 0.0)
            if r in _LOST:
                a.lost += w
            if r == Reachability.REACHABLE:
                x = o.rtt_ms
                if 0 < x < _INF:
                    a.rtt_s += w * x
                    a.rtt_w += w
            x = o.hop_count
            if a.hop < x < _INF:
                a.hop = float(x)
            if o.open_ports:
                a.ports.update(o.open_ports)
            t = o.ts if isinstance(o.ts, _NUM) and math.isfinite(o.ts) else -_INF
            if t >= a.state_ts:
                a.state, a.state_ts = r, t

        add = ctx.store.add_raw
        now = ctx.now
        method = AcquisitionMethod.ACTIVE_PROBE
        G, R, K = MetricKind.GAUGE, MetricKind.RATE, MetricKind.CATEGORICAL
        n = 0
        for (system, entity), a in acc.items():
            if a is None or a.probes <= 0:
                continue
            probes = float(a.probes)
            st = a.state if a.state is not None else Reachability.REACHABLE
            out: List[Tuple[str, Any, MetricKind, str]] = [
                ("probe.probes", probes, MetricKind.COUNTER, "probes"),
                ("probe.reachable", a.reach / probes, G, "score"),
                ("probe.loss_ratio", a.lost / probes, R, "ratio"),
                ("probe.open_ports", float(len(a.ports)), G, "ports"),
                ("probe.state", st.value, K, ""),
            ]
            if a.rtt_w:
                out.append(("probe.rtt_ms", a.rtt_s / a.rtt_w, G, "ms"))
            if a.hop:
                out.append(("probe.hop_count", a.hop, G, "hops"))
            for mname, value, kind, unit in out:
                # touch=False: our probe is not the entity's activity (docstring)
                add(RawMetric(name=mname, value=value, ts=now, system=system, entity=entity,
                              kind=kind, method=method, unit=unit), touch=False)
            n += len(out)
        return n
