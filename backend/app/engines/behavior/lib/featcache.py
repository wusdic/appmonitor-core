"""Per-tick read cache over MetricStore vector series (optional helper).

STATUS: minimal contract stub.

Why: ~20 behaviour engines read the same feature.vec / feature.nat /
feature.tctx / behavior.z rows of every entity on every tick. A tiny cache
keyed by (system, entity, name) and invalidated when `now` changes turns
those into one store lookup each, without any engine holding state across
ticks (the store stays the only source of truth). Values are returned
read-only (numpy arrays with writeable=False) so one engine cannot corrupt
another's view.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class FeatCache:
    """Cache bound to one tick. Reads go through store.vec_at(s, e, name, now)
    (fresh-only: a row written at an earlier tick returns None)."""
    store: Any
    now: float
    _vec: Dict[Tuple[str, str, str], Optional[np.ndarray]] = field(default_factory=dict)

    def reset(self, now: float) -> None:
        """Drop everything if `now` differs from the bound tick."""
        raise NotImplementedError("featcache.FeatCache.reset")

    def vec(self, s: str, e: str, name: str) -> Optional[np.ndarray]:
        """Read-only float array written at `now`, or None."""
        raise NotImplementedError("featcache.FeatCache.vec")

    def feature(self, s: str, e: str, feature: str, which: str = "vec") -> float:
        """One named feature from feature.vec / feature.nat at `now` (NaN if absent)."""
        raise NotImplementedError("featcache.FeatCache.feature")
