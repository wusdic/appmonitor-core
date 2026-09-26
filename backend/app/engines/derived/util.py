"""Shared numeric helpers for derived + behaviour engines.

Pure functions only, so any engine can import them without creating a coupling
to another engine. NumPy where it helps; graceful on empty input."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def series_values(series) -> List[float]:
    return [float(m.value) for m in series if isinstance(m.value, (int, float))]


def shannon_entropy(counts: Iterable[float]) -> float:
    """Shannon entropy in bits of a distribution given as counts."""
    vals = [c for c in counts if c > 0]
    total = sum(vals)
    if total <= 0:
        return 0.0
    return float(-sum((c / total) * math.log2(c / total) for c in vals))


def normalized_entropy(counts: Iterable[float]) -> float:
    """Entropy scaled to 0..1 against the max for that number of categories."""
    vals = [c for c in counts if c > 0]
    if len(vals) <= 1:
        return 0.0
    return shannon_entropy(vals) / math.log2(len(vals))


def char_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq: Dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    return shannon_entropy(freq.values())


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), p))


def robust_stats(values: Sequence[float]) -> Tuple[float, float]:
    """Median and MAD (median absolute deviation, scaled to ~sigma)."""
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) * 1.4826
    return med, mad


def robust_z(value: float, med: float, mad: float) -> float:
    if mad <= 1e-9:
        return 0.0 if abs(value - med) < 1e-9 else 6.0 * (1 if value > med else -1)
    return (value - med) / mad


def ewma(values: Sequence[float], alpha: float = 0.3) -> float:
    if not values:
        return 0.0
    acc = float(values[0])
    for v in values[1:]:
        acc = alpha * float(v) + (1 - alpha) * acc
    return acc


def slope(values: Sequence[float]) -> float:
    """Least-squares slope per step; describes trend direction/strength."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    y = np.asarray(values, dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = float(((x - xm) ** 2).sum())
    if denom <= 1e-9:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)


def coefficient_of_variation(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    m = float(arr.mean())
    if abs(m) < 1e-9:
        return 0.0
    return float(arr.std() / m)


def autocorr_peak(values: Sequence[float], min_lag: int = 2, max_lag: int = 64) -> Tuple[int, float]:
    """Return (best_lag, strength 0..1) of the strongest autocorrelation peak.
    Used to detect periodic / beaconing behaviour in an event-count series."""
    n = len(values)
    if n < min_lag * 2 + 1:
        return 0, 0.0
    arr = np.asarray(values, dtype=float)
    arr = arr - arr.mean()
    denom = float((arr * arr).sum())
    if denom <= 1e-9:
        return 0, 0.0
    best_lag, best = 0, 0.0
    upper = min(max_lag, n - 1)
    for lag in range(min_lag, upper + 1):
        corr = float((arr[:-lag] * arr[lag:]).sum()) / denom
        if corr > best:
            best, best_lag = corr, lag
    return best_lag, max(0.0, min(1.0, best))


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 1.0
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return float(1.0 - np.dot(va, vb) / (na * nb))
