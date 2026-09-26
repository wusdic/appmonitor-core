"""BaselineEngine — per-entity robust baseline & seasonal profile.

For each entity it computes, over the feature history in the store:
  * per-feature median and MAD (robust centre & spread), and
  * time-of-day seasonal medians (24 buckets) for the most human-driven
    features, so "unusual for 3am" differs from "unusual for 10am".

Robust statistics (median/MAD) are used rather than mean/std so a few spikes
don't inflate the "normal" band and mask later anomalies.
"""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import EntityProfile
from .feature_vector import read_feature_matrix
from .features import FEATURE_NAMES


SEASONAL_FEATURES = ["http_requests", "bytes_up", "flows", "dns_queries", "duty_cycle"]


class BaselineEngine(Engine):
    name = "behavior.baseline"
    layer = "behavior"
    consumes = ["feature.*"]
    produces = ["profile.baseline"]
    description = "Robust median/MAD baseline plus 24-bucket time-of-day seasonal medians."

    def __init__(self, min_samples: int = 12, ref_window: int = 130, guard: int = 6, **p):
        super().__init__(**p)
        self.min_samples = min_samples
        self.ref_window = ref_window     # size of the reference distribution
        self.guard = guard               # exclude the most-recent `guard` points

    def run(self, ctx: Context, observations=None) -> int:
        n = 0
        for system in ctx.store.systems():
            for entity in ctx.store.entities(system):
                rows, ts = read_feature_matrix(ctx.store, system, entity)
                if len(rows) < self.min_samples:
                    continue
                # Lagged reference window: measure "normal" from established
                # history, excluding the most recent `guard` ticks, so a fresh
                # behavioural change is scored against the pre-change normal for
                # a detection-latency period instead of being instantly absorbed.
                if len(rows) > self.min_samples + self.guard:
                    ref = rows[max(0, len(rows) - self.ref_window - self.guard):
                               len(rows) - self.guard]
                    ref_ts = ts[max(0, len(ts) - self.ref_window - self.guard):
                                len(ts) - self.guard] if ts else ts
                else:
                    ref, ref_ts = rows, ts
                arr = np.asarray(ref, dtype=float)
                ts = ref_ts
                med = np.median(arr, axis=0)
                mad = np.median(np.abs(arr - med), axis=0) * 1.4826
                prof = ctx.store.profile(system, entity) or EntityProfile(system=system, entity=entity)
                prof.baseline_median = med.tolist()
                prof.baseline_mad = mad.tolist()
                prof.seasonal = self._seasonal(arr, ts)
                prof.stable = True
                prof.updated = ctx.now
                ctx.store.put_profile(prof)
                n += 1
        return n

    def _seasonal(self, arr: np.ndarray, ts: List[float]) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        if len(ts) != arr.shape[0]:
            return out
        hours = [time.gmtime(t).tm_hour for t in ts]
        idx = {name: FEATURE_NAMES.index(name) for name in SEASONAL_FEATURES if name in FEATURE_NAMES}
        for name, col in idx.items():
            buckets = [[] for _ in range(24)]
            for h, v in zip(hours, arr[:, col]):
                buckets[h].append(float(v))
            out[name] = [float(np.median(b)) if b else 0.0 for b in buckets]
        return out
