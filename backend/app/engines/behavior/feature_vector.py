"""FeatureVectorEngine — assemble the current per-entity feature vector.

Reads the store snapshot (latest raw+derived numerics) for each entity, maps
it onto FEATURE_SPEC, applies log compression where declared, then:
  * writes the vector into the EntityProfile as the current fingerprint, and
  * emits every component as a `feature.<name>` derived series.

That second step is deliberate: the feature *history* lives in the shared
store, so Baseline / Fingerprint / Anomaly / Drift read it through the normal
store API and never reach into this engine's memory. This engine is the only
one that knows how to *build* a vector; the rest only consume `feature.*`.
"""
from __future__ import annotations

import math
from typing import Dict, List

from ...core.engine import Context, Engine
from ...models.schema import DerivedMetric, EntityProfile, MetricKind
from .features import FEATURE_NAMES, FEATURE_SPEC


def read_feature_matrix(store, system: str, entity: str, window: int = 240):
    """Return (matrix rows=time, cols=features, timestamps). Shared helper so
    every behaviour engine reconstructs history the same way from the store."""
    cols = []
    ts_ref: List[float] = []
    for name in FEATURE_NAMES:
        series = store.derived_series(system, entity, f"feature.{name}")[-window:]
        cols.append([float(m.value) for m in series])
        if len(series) > len(ts_ref):
            ts_ref = [m.ts for m in series]
    if not cols or not cols[0]:
        return [], []
    length = min(len(c) for c in cols)
    if length == 0:
        return [], ts_ref
    rows = [[cols[f][i] for f in range(len(cols))] for i in range(length)]
    return rows, ts_ref[-length:]


class FeatureVectorEngine(Engine):
    name = "behavior.feature_vector"
    layer = "behavior"
    consumes = ["l4.*", "http.*", "tls.*", "dns.*", "derived.*", "probe.*"]
    produces = ["feature.*", "profile.fingerprint"]
    description = "Assembles the per-entity behavioural feature vector from raw+derived metrics."

    def build_vector(self, snap: Dict[str, float]) -> List[float]:
        vec: List[float] = []
        for _name, source, log_scale in FEATURE_SPEC:
            v = snap.get(source, 0.0)
            if log_scale:
                v = math.log1p(max(v, 0.0))
            vec.append(float(v))
        return vec

    def run(self, ctx: Context, observations=None) -> int:
        n = 0
        for system in ctx.store.systems():
            for entity in ctx.store.entities(system):
                snap = ctx.store.snapshot(system, entity)
                if not snap:
                    continue
                vec = self.build_vector(snap)
                for fname, value in zip(FEATURE_NAMES, vec):
                    ctx.store.add_derived(DerivedMetric(
                        name=f"feature.{fname}", value=value, ts=ctx.now,
                        system=system, entity=entity, window_s=ctx.window_s,
                        kind=MetricKind.GAUGE, inputs=["snapshot"]))
                rows, _ = read_feature_matrix(ctx.store, system, entity)
                prof = ctx.store.profile(system, entity) or EntityProfile(system=system, entity=entity)
                prof.feature_names = FEATURE_NAMES
                prof.fingerprint = vec
                prof.sample_count = len(rows)
                prof.stable = prof.sample_count >= 12
                prof.updated = ctx.now
                ctx.store.put_profile(prof)
                n += 1
        return n
