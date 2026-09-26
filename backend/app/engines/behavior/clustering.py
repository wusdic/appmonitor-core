"""ClusteringEngine — generalise entities into behavioural archetypes.

Per business system, clusters the stable fingerprints of all entities into
archetypes (user-classes): e.g. "interactive browser", "API/batch client",
"scanner/crawler", "bulk-transfer". This is the *generalisation* half of the
behaviour library — one class profile that describes many similar IPs, so a
new IP can be typed instantly and judged against its class as well as itself.

Uses KMeans with a silhouette sweep to pick k, on z-normalised fingerprints.
Falls back gracefully when there are too few entities to cluster.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ...core.engine import Context, Engine
from .features import FEATURE_NAMES
from .util_norm import zscore_normalise


class ClusteringEngine(Engine):
    name = "behavior.clustering"
    layer = "behavior"
    consumes = ["profile.stable_fingerprint"]
    produces = ["profile.archetype"]
    description = "KMeans archetype discovery per system with silhouette-selected k and named classes."

    def __init__(self, k_min: int = 2, k_max: int = 6, min_entities: int = 4, **p):
        super().__init__(**p)
        self.k_min, self.k_max, self.min_entities = k_min, k_max, min_entities

    def run(self, ctx: Context, observations=None) -> int:
        n = 0
        for system in ctx.store.systems():
            entities: List[str] = []
            vecs: List[List[float]] = []
            for entity in ctx.store.entities(system):
                prof = ctx.store.profile(system, entity)
                fp = prof.extra.get("stable_fingerprint") if prof else None
                if fp:
                    entities.append(entity)
                    vecs.append(fp)
            if len(vecs) < self.min_entities:
                continue
            mat = zscore_normalise(np.asarray(vecs, dtype=float))
            labels, centers = self._cluster(mat)
            names = self._name_clusters(centers)
            for entity, label, dist in zip(entities, labels[0], labels[1]):
                prof = ctx.store.profile(system, entity)
                if not prof:
                    continue
                prof.archetype = names[label]
                prof.archetype_confidence = round(float(max(0.0, 1.0 - dist)), 3)
                ctx.store.put_profile(prof)
                n += 1
        return n

    def _cluster(self, mat: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        best_k, best_score, best_model = self.k_min, -1.0, None
        upper = min(self.k_max, mat.shape[0] - 1)
        for k in range(self.k_min, max(self.k_min, upper) + 1):
            if k >= mat.shape[0]:
                break
            model = KMeans(n_clusters=k, n_init=6, random_state=11).fit(mat)
            if len(set(model.labels_)) < 2:
                continue
            try:
                score = silhouette_score(mat, model.labels_)
            except Exception:
                score = -1.0
            if score > best_score:
                best_k, best_score, best_model = k, score, model
        if best_model is None:
            from sklearn.cluster import KMeans as KM
            best_model = KM(n_clusters=self.k_min, n_init=6, random_state=11).fit(mat)
        labels = best_model.labels_
        centers = best_model.cluster_centers_
        # normalised distance of each point to its centre (0..1-ish)
        dists = np.linalg.norm(mat - centers[labels], axis=1)
        dmax = float(dists.max()) if dists.size else 1.0
        dnorm = dists / dmax if dmax > 1e-9 else dists
        return (labels, dnorm), centers

    def _name_clusters(self, centers: np.ndarray) -> Dict[int, str]:
        """Heuristically label each centroid by its dominant features so the
        archetype is human-readable, not just 'cluster 3'."""
        idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
        names: Dict[int, str] = {}
        for c, center in enumerate(centers):
            def g(name):
                return float(center[idx[name]]) if name in idx else 0.0
            tags = []
            if g("periodicity") > 0.6 or g("timing_regularity") > 0.6:
                tags.append("automated")
            if g("http_write_ratio") > 0.5 or g("distinct_paths") > 0.5:
                tags.append("interactive")
            if g("bytes_down") > 0.8 and g("bytes_up") < 0.3:
                tags.append("bulk-consumer")
            if g("bytes_up") > 0.8 and g("bytes_down") < 0.3:
                tags.append("bulk-producer")
            if g("distinct_peers") > 0.8 or g("fanout") > 0.8:
                tags.append("broad-fanout")
            if g("dns_dga_score") > 0.5 or g("sni_entropy") > 0.7:
                tags.append("high-dispersion")
            if not tags:
                tags.append(f"class-{c}")
            names[c] = "/".join(tags[:3])
        # de-duplicate identical names
        seen: Dict[str, int] = {}
        for c in list(names):
            base = names[c]
            if base in seen.values():
                names[c] = f"{base}#{c}"
            seen[c] = names[c]
        return names
