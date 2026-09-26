"""DriftEngine — "this entity is not behaving like itself".

Where AnomalyEngine asks "is this tick far from the baseline centre?", Drift
asks the identity question: has the *shape* of behaviour moved away from the
entity's established fingerprint, and has it stayed moved? It compares a
short recent window's median against the long-term stable fingerprint using
cosine distance on MAD-scaled vectors, and requires the divergence to persist
across several ticks before firing, so a single spike doesn't count as drift.

This is what catches "the account/host is doing something out of character"
even when every individual metric is within its own global range.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import BehaviorEvent, Severity
from .feature_vector import read_feature_matrix
from .features import FEATURE_NAMES


class DriftEngine(Engine):
    name = "behavior.drift"
    layer = "behavior"
    consumes = ["feature.*", "profile.stable_fingerprint"]
    produces = ["event.drift"]
    description = "Persistent divergence of recent behaviour from the entity's stable fingerprint."

    def __init__(self, recent: int = 5, distance_threshold: float = 0.35,
                 persist_ticks: int = 3, **p):
        super().__init__(**p)
        self.recent = recent
        self.threshold = distance_threshold
        self.persist = persist_ticks
        self._streak: Dict[str, int] = defaultdict(int)

    def run(self, ctx: Context, observations=None) -> int:
        if ctx.training:                     # warm-up: no alerts
            return 0
        n = 0
        for system in ctx.store.systems():
            for entity in ctx.store.entities(system):
                prof = ctx.store.profile(system, entity)
                if not prof or not prof.stable:
                    continue
                stable = prof.extra.get("stable_fingerprint")
                if not stable:
                    continue
                rows, _ = read_feature_matrix(ctx.store, system, entity)
                if len(rows) < self.recent + 4:
                    continue
                recent = np.median(np.asarray(rows[-self.recent:], dtype=float), axis=0)
                stable_v = np.asarray(stable, dtype=float)
                mad = np.asarray(prof.baseline_mad, dtype=float) if prof.baseline_mad else np.ones_like(stable_v)
                dist, contributors = self._scaled_cosine(recent, stable_v, mad)

                key = f"{system}|{entity}"
                if dist >= self.threshold:
                    self._streak[key] += 1
                else:
                    self._streak[key] = 0

                if self._streak[key] == self.persist:  # fire once when it locks in
                    ctx.store.add_event(BehaviorEvent(
                        system=system, entity=entity, ts=ctx.now, kind="drift",
                        score=round(min(1.0, dist), 3), severity=self._severity(dist),
                        contributors=contributors[:6],
                        description="Behaviour has shifted away from this entity's "
                                    "established fingerprint (" + ", ".join(
                                        f"{nm}{'↑' if d > 0 else '↓'}" for nm, d in contributors[:3]) + ")",
                        extra={"cosine_distance": round(dist, 3),
                               "persisted_ticks": self._streak[key]}))
                    n += 1
        return n

    @staticmethod
    def _scaled_cosine(a: np.ndarray, b: np.ndarray, mad: np.ndarray) -> Tuple[float, List[Tuple[str, float]]]:
        floor = np.maximum(0.05 * np.abs(b), 0.05)
        w = 1.0 / np.maximum(mad, floor)
        aw, bw = a * w, b * w
        na, nb = np.linalg.norm(aw), np.linalg.norm(bw)
        if na < 1e-9 or nb < 1e-9:
            return 0.0, []
        dist = 1.0 - float(np.dot(aw, bw) / (na * nb))
        delta = (a - b) * w
        contributors = sorted(
            ((FEATURE_NAMES[i], float(delta[i])) for i in range(len(delta))),
            key=lambda kv: abs(kv[1]), reverse=True)
        return max(0.0, min(1.0, dist)), contributors

    @staticmethod
    def _severity(dist: float) -> Severity:
        if dist >= 0.7:
            return Severity.HIGH
        if dist >= 0.5:
            return Severity.MEDIUM
        return Severity.LOW
