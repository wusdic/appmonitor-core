"""RuleMatchEngine — evaluate the signature library against each entity.

For every entity it takes the flat metric snapshot (raw+derived numerics plus
a few categorical helpers) and evaluates every in-scope signature. Conditions
can be hard (0/1) or *soft* (fuzzy membership 0..1 via a tolerance band), so a
signature that is "almost" satisfied yields a proportionate confidence instead
of silently failing. Confidence = weighted mean of AND-memberships × best
OR-membership × weight, gated by the NONE conditions.

Emits a SignatureMatch per firing signature — the answer to "what is this
entity doing right now?"
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ...core.engine import Context, Engine
from ...models.schema import Severity, SignatureMatch
from .store import Signature, SignatureStore


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

    def run(self, ctx: Context, observations=None) -> int:
        n = 0
        for system in ctx.store.systems():
            sigs = self.sig_store.for_system(system)
            if not sigs:
                continue
            for entity in ctx.store.entities(system):
                snap = ctx.store.snapshot(system, entity)
                if not snap:
                    continue
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
