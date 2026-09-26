"""FingerprintEngine — the stable behavioural identity of an entity.

The current fingerprint (this tick's vector) is noisy; this engine distils a
*stable* fingerprint = robust centre (median) of the entity's recent history,
which is what actually identifies "how this IP/IP-class normally behaves".

It also computes a **separability** score: how distinguishable this entity is
from every other entity in the same business system, measured as the cosine
distance to its nearest neighbour in fingerprint space (normalised 0..1). High
separability => the profile alone can pick this entity out of the crowd; low
separability => it looks like many others and needs more features to tell
apart. This directly answers the requirement that a profile be precise enough
to distinguish one (or one class of) user from others.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ...core.engine import Context, Engine
from .feature_vector import read_feature_matrix
from .util_norm import zscore_normalise


class FingerprintEngine(Engine):
    name = "behavior.fingerprint"
    layer = "behavior"
    consumes = ["feature.*"]
    produces = ["profile.stable_fingerprint", "profile.separability"]
    description = "Stable (median) behavioural fingerprint + nearest-neighbour separability score."

    def __init__(self, min_samples: int = 12, **p):
        super().__init__(**p)
        self.min_samples = min_samples

    def run(self, ctx: Context, observations=None) -> int:
        # pass 1: compute each entity's stable fingerprint
        stable: dict = {}
        for system in ctx.store.systems():
            for entity in ctx.store.entities(system):
                rows, _ = read_feature_matrix(ctx.store, system, entity)
                if len(rows) < self.min_samples:
                    continue
                arr = np.asarray(rows, dtype=float)
                fp = np.median(arr, axis=0)
                stable[(system, entity)] = fp
                prof = ctx.store.profile(system, entity)
                if prof:
                    prof.extra["stable_fingerprint"] = fp.tolist()
                    ctx.store.put_profile(prof)

        # pass 2: separability = distance to nearest neighbour within the system,
        # computed on z-normalised fingerprints so no single big-scale feature
        # dominates the geometry.
        n = 0
        by_system: dict = {}
        for (system, entity), fp in stable.items():
            by_system.setdefault(system, []).append((entity, fp))
        for system, items in by_system.items():
            if len(items) < 2:
                for entity, _ in items:
                    prof = ctx.store.profile(system, entity)
                    if prof:
                        prof.separability = 1.0
                        ctx.store.put_profile(prof)
                        n += 1
                continue
            mat = np.asarray([fp for _, fp in items], dtype=float)
            norm = zscore_normalise(mat)
            for i, (entity, _) in enumerate(items):
                sep = self._nearest_distance(norm, i)
                prof = ctx.store.profile(system, entity)
                if prof:
                    prof.separability = round(sep, 4)
                    ctx.store.put_profile(prof)
                    n += 1
        return n

    @staticmethod
    def _nearest_distance(norm: np.ndarray, i: int) -> float:
        me = norm[i]
        best = 2.0
        for j in range(norm.shape[0]):
            if j == i:
                continue
            a, b = me, norm[j]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-9 or nb < 1e-9:
                d = 1.0
            else:
                d = 1.0 - float(np.dot(a, b) / (na * nb))
            best = min(best, d)
        return max(0.0, min(1.0, best))
