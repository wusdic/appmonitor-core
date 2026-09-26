"""AnomalyEngine — multi-method deviation scoring against the entity's baseline.

Fuses three independent detectors so no single method's blind spot dominates:
  1. Robust-z distance of the current vector from the per-feature baseline
     (median/MAD) — interpretable, gives per-feature contributors.
  2. IsolationForest trained on the entity's own history — catches odd feature
     *combinations* a per-feature test misses.
  3. Seasonal residual for time-of-day features — "high for this hour".

Scores are normalized to 0..1 and combined with a soft-OR (noisy-max) so a
strong signal from any one detector surfaces, while agreement raises severity.
Emits a BehaviorEvent with ranked contributors when the fused score clears a
threshold.
"""
from __future__ import annotations

import math
import time
from typing import List, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import BehaviorEvent, Severity
from .baseline import SEASONAL_FEATURES
from .feature_vector import read_feature_matrix
from .features import FEATURE_NAMES


def _sigmoid(x: float, k: float = 1.0) -> float:
    return 1.0 / (1.0 + math.exp(-k * x))


class AnomalyEngine(Engine):
    name = "behavior.anomaly"
    layer = "behavior"
    consumes = ["feature.*", "profile.baseline"]
    produces = ["event.anomaly"]
    description = "Fused robust-z + IsolationForest + seasonal-residual anomaly score."

    def __init__(self, threshold: float = 0.6, strong: float = 0.8, min_samples: int = 16, **p):
        super().__init__(**p)
        self.threshold = threshold
        self.strong = strong                 # fire immediately at/above this
        self.min_samples = min_samples
        self._prev: dict = {}                # last score per entity (persistence)

    def run(self, ctx: Context, observations=None) -> int:
        if ctx.training:                     # warm-up: baselines only, no alerts
            return 0
        n = 0
        for system in ctx.store.systems():
            for entity in ctx.store.entities(system):
                prof = ctx.store.profile(system, entity)
                if not prof or not prof.stable or not prof.baseline_median:
                    continue
                cur = np.asarray(prof.fingerprint, dtype=float)
                med = np.asarray(prof.baseline_median, dtype=float)
                mad = np.asarray(prof.baseline_mad, dtype=float)
                if cur.shape != med.shape:
                    continue

                z_score, contributors = self._robust_z(cur, med, mad)
                iso_score = self._isoforest(ctx, system, entity, cur)
                seas_score = self._seasonal(prof, cur)

                # The robust per-feature z is the PRIMARY, interpretable signal.
                # IsolationForest (odd feature combinations) and the seasonal
                # residual are corroborating BOOSTERS: each has its own
                # false-positive mode (forest contamination, sparse hour buckets),
                # so on their own they may raise at most a sub-threshold flag —
                # only when the primary signal is present do they push a finding
                # to high/critical. This keeps normal entities quiet.
                primary = z_score
                combo = max(iso_score, seas_score)
                fused = primary + (1.0 - primary) * combo * (0.4 + 0.6 * primary)
                fused = min(1.0, fused)
                agree = sum(1 for p in (z_score, iso_score, seas_score) if p > 0.5)

                # Persistence guard: a strong anomaly fires at once; a weaker one
                # must be confirmed by the previous tick, so a single-tick burst
                # from an ordinarily bursty entity doesn't raise a critical alert.
                ekey = f"{system}|{entity}"
                prev = self._prev.get(ekey, 0.0)
                self._prev[ekey] = fused
                fire = fused >= self.strong or (fused >= self.threshold and prev >= self.threshold)
                if fire:
                    ctx.store.add_event(BehaviorEvent(
                        system=system, entity=entity, ts=ctx.now, kind="anomaly",
                        score=round(fused, 3), severity=self._severity(fused),
                        contributors=contributors[:6],
                        description=self._describe(contributors[:3]),
                        extra={"z": round(z_score, 3), "iso": round(iso_score, 3),
                               "seasonal": round(seas_score, 3)}))
                    n += 1
        return n

    def _robust_z(self, cur, med, mad) -> Tuple[float, List[Tuple[str, float]]]:
        # Floor the spread generously so a structurally-zero feature (entropy,
        # DGA score, loss ratio — normally 0 with ~0 spread) can't turn a small
        # absolute change into a huge z. Then clip z to a sane band.
        floor = np.maximum(0.2 * np.abs(med), 0.2)
        safe_mad = np.maximum(mad, floor)
        z = np.clip((cur - med) / safe_mad, -12.0, 12.0)
        contributors = sorted(
            ((FEATURE_NAMES[i], float(z[i])) for i in range(len(z))),
            key=lambda kv: abs(kv[1]), reverse=True)
        # Aggregate by the *accumulated excess* beyond 2.5σ across features, so a
        # single noisy feature can't score high on its own — a real anomaly
        # deviates on several dimensions at once (exfil, scan, brute-force all
        # move many features), while ordinary jitter on one feature does not.
        excess = np.clip(np.abs(z) - 2.5, 0.0, None)
        strong = float(excess.sum())
        score = 1.0 - np.exp(-strong / 6.0)
        return float(min(1.0, max(0.0, score))), contributors

    def _isoforest(self, ctx, system, entity, cur) -> float:
        rows, _ = read_feature_matrix(ctx.store, system, entity)
        if len(rows) < self.min_samples:
            return 0.0
        try:
            from sklearn.ensemble import IsolationForest
            X = np.asarray(rows, dtype=float)
            clf = IsolationForest(n_estimators=50, contamination="auto", random_state=7)
            clf.fit(X)
            # score_samples: higher = more normal. A *typical* point sits in the
            # middle of the training distribution, so we must not treat "middle"
            # as anomalous. Only the tail below the learned outlier cutoff scores
            # above zero; everything more normal than the cutoff scores 0.
            raw = float(clf.score_samples(cur.reshape(1, -1))[0])
            train = clf.score_samples(X)
            cutoff = float(np.percentile(train, 8))   # ~contamination tail
            lo = float(train.min())
            if raw >= cutoff:
                return 0.0
            span = cutoff - lo
            if span < 1e-9:
                return 0.5
            return float(min(1.0, (cutoff - raw) / span))
        except Exception:
            return 0.0

    def _seasonal(self, prof, cur) -> float:
        if not prof.seasonal:
            return 0.0
        hour = time.gmtime(prof.updated).tm_hour
        worst = 0.0
        for name, buckets in prof.seasonal.items():
            if name not in FEATURE_NAMES or len(buckets) != 24:
                continue
            expected = buckets[hour]
            if abs(expected) <= 1e-6:          # empty/unsupported hour bucket
                continue                       # no seasonal evidence, don't guess
            idx = FEATURE_NAMES.index(name)
            val = float(cur[idx])
            resid = abs(val - expected) / abs(expected)   # relative deviation
            worst = max(worst, _sigmoid(resid - 2.5, k=1.0))
        return worst

    @staticmethod
    def _severity(score: float) -> Severity:
        if score >= 0.9:
            return Severity.CRITICAL
        if score >= 0.8:
            return Severity.HIGH
        if score >= 0.7:
            return Severity.MEDIUM
        return Severity.LOW

    @staticmethod
    def _describe(top: List[Tuple[str, float]]) -> str:
        if not top:
            return "Behaviour deviates from baseline."
        parts = [f"{name} {'↑' if z > 0 else '↓'}{abs(z):.1f}σ" for name, z in top]
        return "Deviation: " + ", ".join(parts)
