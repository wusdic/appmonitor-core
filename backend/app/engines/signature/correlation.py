"""CorrelationEngine — compose primitive signatures into higher-level activities.

A single signature says "this looks like an auth" or "this looks like a bulk
transfer". The meaning often lives in the *combination over time*: an auth
followed by a bulk transfer to a new destination within a few minutes is a
data-staging pattern; a recon signature co-occurring with a tunnel signature is
something else again. This engine reads the recent SignatureMatch stream per
entity and applies composite rules:

  sequence: [a, b, c]  -> categories must appear in this order within window_s
  cooccur:  [a, b]     -> categories must all appear within window_s (any order)

It emits a higher-level SignatureMatch (category/severity from the rule) whose
confidence is the product of the constituent matches' confidences. Composite
rules are data (data/signatures/composite.yaml), same as primitives.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ...core.engine import Context, Engine
from ...models.schema import Severity, SignatureMatch


class CorrelationEngine(Engine):
    name = "signature.correlation"
    layer = "signature"
    consumes = ["match.*"]
    produces = ["match.composite.*"]
    description = "Temporal composition of primitive signatures into higher-level activity patterns."

    def __init__(self, rules: List[Dict[str, Any]], min_confidence: float = 0.5, **p):
        super().__init__(**p)
        self.rules = rules or []
        self.min_confidence = min_confidence
        self._emitted: set = set()   # (system,entity,rule_id,bucket) dedupe

    def run(self, ctx: Context, observations=None) -> int:
        n = 0
        for system in ctx.store.systems():
            for entity in ctx.store.entities(system):
                matches = ctx.store.matches(system=system, entity=entity, limit=100)
                # exclude composite matches to avoid feedback loops
                matches = [m for m in matches if not m.signature_id.startswith("composite:")]
                if not matches:
                    continue
                for rule in self.rules:
                    if rule.get("scope") and system not in rule["scope"]:
                        continue
                    window = float(rule.get("window_s", 300))
                    recent = [m for m in matches if ctx.now - m.ts <= window]
                    if not recent:
                        continue
                    hit, conf, terms = self._match_rule(rule, recent)
                    if hit and conf >= self.min_confidence:
                        bucket = int(ctx.now // max(window, 1))
                        dk = (system, entity, rule["id"], bucket)
                        if dk in self._emitted:
                            continue
                        self._emitted.add(dk)
                        ctx.store.add_match(SignatureMatch(
                            system=system, entity=entity, ts=ctx.now,
                            signature_id=f"composite:{rule['id']}",
                            label=rule.get("label", rule["id"]),
                            category=rule.get("category", "composite"),
                            confidence=round(conf, 3), matched_terms=terms,
                            severity=Severity(rule.get("severity", "medium")),
                            evidence={"pattern": rule.get("sequence") or rule.get("cooccur")}))
                        n += 1
        return n

    def _match_rule(self, rule, recent: List[SignatureMatch]):
        recent = sorted(recent, key=lambda m: m.ts)
        if "sequence" in rule:
            seq = rule["sequence"]
            # Use each category's first-occurrence time and best confidence, and
            # require those times to be non-decreasing in the specified order.
            # This tolerates several matches sharing a tick (same ts) while still
            # enforcing multi-tick ordering (a -> b -> c across the window).
            first_ts: dict = {}
            best_conf: dict = {}
            for m in recent:
                if m.category in seq:
                    if m.category not in first_ts:
                        first_ts[m.category] = m.ts
                    best_conf[m.category] = max(best_conf.get(m.category, 0.0), m.confidence)
            if not all(c in first_ts for c in seq):
                return False, 0.0, []
            times = [first_ts[c] for c in seq]
            if all(times[i] <= times[i + 1] for i in range(len(times) - 1)):
                conf = 1.0
                terms = []
                for c in seq:
                    conf *= best_conf[c]
                    terms.append(c)
                return True, conf, terms
            return False, 0.0, []
        if "cooccur" in rule:
            need = set(rule["cooccur"])
            by_cat: Dict[str, float] = {}
            terms = []
            for m in recent:
                if m.category in need and m.confidence > by_cat.get(m.category, 0):
                    by_cat[m.category] = m.confidence
            if need.issubset(by_cat.keys()):
                conf = 1.0
                for c in need:
                    conf *= by_cat[c]
                    terms.append(c)
                return True, conf, terms
            return False, 0.0, []
        return False, 0.0, []
