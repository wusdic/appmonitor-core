"""SequenceEngine — models the *order* of activities, not just their volume.

Behaviour has grammar: for one business system a normal user might go
login → browse → search → view → logout. This engine treats the stream of
signature *categories* an entity triggers (produced by the signature layer on
prior ticks) as a first-order Markov chain, learns the transition matrix per
(system, archetype) online, and scores the surprisal of each new transition.
An out-of-grammar jump (e.g. login → bulk-export with no browse in between)
scores high even when every underlying metric looks individually normal.

It reads only `store.matches` (previous ticks) — no coupling to the signature
engines beyond the shared match records.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Tuple

from ...core.engine import Context, Engine
from ...models.schema import BehaviorEvent, Severity


class SequenceEngine(Engine):
    name = "behavior.sequence"
    layer = "behavior"
    consumes = ["match.*"]
    produces = ["event.sequence"]
    description = "First-order Markov surprisal over the entity's activity-category sequence."

    def __init__(self, surprisal_threshold: float = 4.0, min_transitions: int = 30, **p):
        super().__init__(**p)
        self.threshold = surprisal_threshold
        self.min_transitions = min_transitions
        # transition counts keyed by (system, archetype): {(a,b): count}
        self._trans: Dict[str, Dict[Tuple[str, str], float]] = defaultdict(lambda: defaultdict(float))
        self._totals: Dict[str, float] = defaultdict(float)
        self._last: Dict[str, str] = {}          # per entity last category
        self._seen_ts: Dict[str, float] = {}     # last processed match ts per entity

    def run(self, ctx: Context, observations=None) -> int:
        n = 0
        for system in ctx.store.systems():
            for entity in ctx.store.entities(system):
                raw_matches = ctx.store.matches(system=system, entity=entity, limit=120)
                if not raw_matches:
                    continue
                # Collapse each tick to its single dominant activity (highest
                # confidence, composites preferred) so the "sequence" reflects
                # what the entity was mainly doing, not every co-firing label.
                by_ts: Dict[float, tuple] = {}
                for m in raw_matches:
                    weight = m.confidence + (0.5 if m.signature_id.startswith("composite:") else 0.0)
                    cur = by_ts.get(m.ts)
                    if cur is None or weight > cur[0]:
                        by_ts[m.ts] = (weight, m)
                matches = [v[1] for _, v in sorted(by_ts.items())]
                ekey = f"{system}|{entity}"
                prof = ctx.store.profile(system, entity)
                model_key = f"{system}|{prof.archetype if prof and prof.archetype else 'all'}"
                last_seen = self._seen_ts.get(ekey, 0.0)
                for m in matches:
                    if m.ts <= last_seen:
                        continue
                    cat = m.category or "other"
                    prev = self._last.get(ekey)
                    if prev is not None:
                        surp = self._surprisal(model_key, prev, cat)
                        self._learn(model_key, prev, cat)   # learn even in warm-up
                        if not ctx.training and surp >= self.threshold \
                                and self._totals[model_key] >= self.min_transitions:
                            ctx.store.add_event(BehaviorEvent(
                                system=system, entity=entity, ts=m.ts, kind="sequence",
                                score=round(min(1.0, surp / 8.0), 3),
                                severity=Severity.MEDIUM if surp < 5 else Severity.HIGH,
                                contributors=[(f"{prev}->{cat}", round(surp, 2))],
                                description=f"Unusual activity transition {prev} → {cat} "
                                            f"for this {prof.archetype if prof else 'entity'} "
                                            f"(surprisal {surp:.1f} bits)"))
                            n += 1
                    self._last[ekey] = cat
                    self._seen_ts[ekey] = m.ts
        return n

    def _surprisal(self, model_key: str, a: str, b: str) -> float:
        trans = self._trans[model_key]
        out_total = sum(v for (x, _), v in trans.items() if x == a)
        if out_total <= 0:
            return 0.0
        p = (trans.get((a, b), 0.0) + 0.5) / (out_total + 0.5 * 8)   # Laplace
        return -math.log2(max(p, 1e-6))

    def _learn(self, model_key: str, a: str, b: str) -> None:
        self._trans[model_key][(a, b)] += 1.0
        self._totals[model_key] += 1.0
