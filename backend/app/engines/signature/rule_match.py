"""RuleMatchEngine — evaluate the signature library against each entity.

For every entity it takes the flat metric snapshot (raw+derived numerics plus
a few categorical helpers) and evaluates every in-scope signature. Conditions
can be hard (0/1) or *soft* (fuzzy membership 0..1 via a tolerance band), so a
signature that is "almost" satisfied yields a proportionate confidence instead
of silently failing. Confidence = weighted mean of AND-memberships × best
OR-membership × weight, gated by the NONE conditions.

Emits a SignatureMatch per firing signature — the answer to "what is this
entity doing right now?"

Counter clauses are read per 15-min grain (evaluator round 3). A raw counter
(`http.requests`, `l4.bytes_up`, ...) is a per-TICK total, so a threshold on
it meant something different at every cadence: `c2_beacon`'s `http.requests
<= 10` never matched a 30-s health checker at 900 s (30 per tick) and matched
it on every 60-s tick (2 per tick); at 60 s `maintenance`, `search`, `health`
and `browse` appeared on control entities that never matched them in the
900-s warm-up, which B08 then scored as first_seen / JSD drift of its lib-4
category dimension. The value of an additive counter is therefore its total
over the trailing 900 s (the Q grain, cadence.md §2; a tick straddling the
window start pro rata), and a tick longer than 900 s is scaled to 900 s, so
steady traffic gives the same membership at 60, 900 and 3600 s and the
thresholds keep the meaning they had at the packs' 900-s cadence (canonical
grain mode; tick mode keeps v2's per-tick reading as the golden reference). Distinct
counts (distinct peers / ports / paths / qnames) are not additive and stay
per tick (at a fine cadence they can only shrink, i.e. never create a match).
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from ...core.engine import Context, Engine
from ..behavior.lib import grains as GR
from ...models.schema import Severity, SignatureMatch
from .store import Signature, SignatureStore


# additive raw counters: read as totals per 15-min grain (see module docstring)
GRAIN_S = 900.0
ADDITIVE_COUNTERS = frozenset({
    "act.events", "l3.bytes_total", "l4.flows", "l4.bytes_up", "l4.bytes_down",
    "l4.syn_count", "l4.pkts_total", "http.requests", "http.status_2xx", "http.status_3xx",
    "http.status_4xx", "http.status_5xx", "http.get_count", "http.write_count",
    "tls.handshakes", "dns.queries", "dns.txt_count", "dns.nxdomain",
})


class RuleMatchEngine(Engine):
    name = "signature.rule_match"
    layer = "signature"
    consumes = ["<snapshot>"]
    produces = ["match.*"]
    description = "Evaluates the preset signature library (fuzzy) against each entity snapshot."

    def __init__(self, store: SignatureStore, min_confidence: float = 0.55, **p):
        super().__init__(**p)
        self.sig_store = store
        self.min_confidence = min_confidence
        # (tick end, Δt) of the recent ticks, to pro-rate a past tick that
        # straddles the start of the 900-s window (all entities share ticks)
        self._ticks: Deque[Tuple[float, float]] = deque()

    def _log_tick(self, now: float, dt: float) -> None:
        tk = self._ticks
        while tk and tk[-1][0] >= now:          # a replayed / restarted clock
            tk.pop()
        tk.append((now, dt))
        horizon = now - 2.0 * GRAIN_S - max(dt, GRAIN_S)
        while tk and tk[0][0] < horizon:
            tk.popleft()

    def _tick_dt(self, ts: float, default: float) -> float:
        for t, d in reversed(self._ticks):
            if abs(t - ts) < 1e-6:
                return d
            if t < ts:
                break
        return default

    def per_grain(self, store: Any, system: str, entity: str, name: str, value: float,
                  now: float, dt: float) -> float:
        """Total of the additive counter `name` per 900 s ending at now."""
        if dt >= GRAIN_S:
            return value * GRAIN_S / dt
        lo, total = now - GRAIN_S, 0.0
        for m in store.raw_tail(system, entity, name, int(GRAIN_S // max(dt, 1.0)) + 2):
            ts = float(m.ts)
            if not (lo - GRAIN_S < ts <= now) or not isinstance(m.value, (int, float)):
                continue
            v = float(m.value)
            if not math.isfinite(v):
                continue
            if ts == now:
                total += v
                continue
            d = self._tick_dt(ts, dt)
            ov = ts - max(ts - d, lo)
            if ov > 0.0 and d > 0.0:
                total += v * min(1.0, ov / d)
        # an entity observed for less than one grain: a rate over the observed span
        fs = store.first_seen(system, entity)
        if fs is not None:
            covered = now - float(fs) + dt
            if 0.0 < covered < GRAIN_S:
                total *= GRAIN_S / covered
        return total

    def run(self, ctx: Context, observations=None) -> int:
        n = 0
        now, dt = float(ctx.now), float(ctx.window_s)
        canon = GR.canonical(ctx.config)
        self._log_tick(now, dt)
        for system in ctx.store.systems():
            sigs = self.sig_store.for_system(system)
            if not sigs:
                continue
            names = self._metric_names(sigs)
            # spec v2.1 canonical grain mode only: tick mode keeps v2's per-tick
            # reading (the golden reference, cadence.md M8)
            counters = [m for m in names if m in ADDITIVE_COUNTERS] if canon else []
            for entity in ctx.store.entities(system):
                # fresh-only (written at this tick): a match answers "what is
                # the entity doing right now". Without `now` the latest value
                # of any age was used, so a host that uploaded once kept
                # matching bulk_upload (HIGH) on every idle tick afterwards
                # (eval: the nightly backup hosts matched on ~150 of 152 ticks,
                # which zeroed their warm-up trust and made B26 / B28 treat
                # them as attackers from the first live tick)
                snap = ctx.store.snapshot(system, entity, now=ctx.now, names=names)
                if not snap:
                    continue
                for name in counters:
                    v = snap.get(name)
                    if isinstance(v, (int, float)):
                        snap[name] = self.per_grain(ctx.store, system, entity, name,
                                                    float(v), now, dt)
                for sig in sigs:
                    conf, terms = self._evaluate(sig, snap)
                    if conf >= self.min_confidence:
                        ctx.store.add_match(SignatureMatch(
                            system=system, entity=entity, ts=ctx.now,
                            signature_id=sig.id, label=sig.label, category=sig.category,
                            confidence=round(conf, 3), matched_terms=terms,
                            severity=Severity(sig.severity),
                            evidence={t: round(snap.get(t.split(" ")[0], 0.0), 3) for t in terms}))
                        n += 1
        return n

    @staticmethod
    def _metric_names(sigs) -> Tuple[str, ...]:
        """Every metric the system's signatures reference: the snapshot is
        restricted to them (same values, a fraction of the cost once lib-3
        writes hundreds of series and vector views per entity)."""
        return tuple(sorted({str(c.get("metric", "")) for sig in sigs
                             for c in list(sig.all) + list(sig.any) + list(sig.none)
                             if c.get("metric")}))

    # --------------------------------------------------------------- evaluation
    def _evaluate(self, sig: Signature, snap: Dict[str, float]) -> Tuple[float, List[str]]:
        # NONE gate
        for cond in sig.none:
            mem, _ = self._member(cond, snap)
            if mem >= 0.5:
                return 0.0, []
        terms: List[str] = []
        and_mems: List[float] = []
        for cond in sig.all:
            mem, term = self._member(cond, snap)
            and_mems.append(mem)
            if mem >= 0.5 and term:
                terms.append(term)
        # a hard AND failure (any zero) collapses confidence
        if and_mems and min(and_mems) <= 1e-6:
            return 0.0, []
        and_score = sum(and_mems) / len(and_mems) if and_mems else 1.0

        or_score = 1.0
        if sig.any:
            best, best_term = 0.0, ""
            for cond in sig.any:
                mem, term = self._member(cond, snap)
                if mem > best:
                    best, best_term = mem, term
            or_score = best
            if best >= 0.5 and best_term:
                terms.append(best_term)
            if best <= 1e-6:
                return 0.0, []

        conf = and_score * or_score * min(sig.weight, 1.5)
        return min(conf, 1.0), terms

    def _member(self, cond: Dict[str, Any], snap: Dict[str, float]) -> Tuple[float, str]:
        metric = cond.get("metric", "")
        op = cond.get("op", "gt")
        value = cond.get("value")
        soft = cond.get("soft")
        present = metric in snap
        x = snap.get(metric, 0.0)
        term = f"{metric} {op} {value}"

        if op == "present":
            return (1.0 if present else 0.0), (metric if present else "")
        if not present:
            return 0.0, ""

        def hard(cond_ok: bool) -> Tuple[float, str]:
            return (1.0, term) if cond_ok else (0.0, "")

        if op in ("gt", "ge"):
            thr = float(value)
            if soft:
                return self._soft_ge(x, thr, float(soft)), term
            return hard(x > thr if op == "gt" else x >= thr)
        if op in ("lt", "le"):
            thr = float(value)
            if soft:
                return self._soft_le(x, thr, float(soft)), term
            return hard(x < thr if op == "lt" else x <= thr)
        if op == "eq":
            return hard(abs(x - float(value)) < 1e-9)
        if op == "ne":
            return hard(abs(x - float(value)) >= 1e-9)
        if op == "between":
            lo, hi = float(value[0]), float(value[1])
            return hard(lo <= x <= hi)
        if op == "in":
            return hard(x in set(value))
        if op == "not_in":
            return hard(x not in set(value))
        return 0.0, ""

    @staticmethod
    def _soft_ge(x: float, thr: float, band: float) -> float:
        if x >= thr:
            return 1.0
        if x <= thr - band:
            return 0.0
        return (x - (thr - band)) / band

    @staticmethod
    def _soft_le(x: float, thr: float, band: float) -> float:
        if x <= thr:
            return 1.0
        if x >= thr + band:
            return 0.0
        return ((thr + band) - x) / band
