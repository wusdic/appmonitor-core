"""In-memory time-series store shared by every engine.

Deliberately small and dependency-free. In a real deployment this interface is
the seam where you drop in ClickHouse / VictoriaMetrics / TimescaleDB: engines
only ever call `add_*`, `series`, `latest`, `snapshot`. They never assume a
storage backend, which is what lets the store be swapped without touching a
single engine.
"""
from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional

from ..models.schema import (
    BehaviorEvent,
    DerivedMetric,
    EntityProfile,
    Observation,
    RawMetric,
    SignatureMatch,
)


class MetricStore:
    """Thread-safe ring-buffer store. Bounded so a long-running demo cannot
    grow without limit."""

    def __init__(self, max_points: int = 20000) -> None:
        self._lock = threading.RLock()
        self._max = max_points
        # keyed series of (ts, value)
        self._raw: Dict[str, Deque[RawMetric]] = defaultdict(lambda: deque(maxlen=max_points))
        self._derived: Dict[str, Deque[DerivedMetric]] = defaultdict(lambda: deque(maxlen=max_points))
        self._events: Deque[BehaviorEvent] = deque(maxlen=max_points)
        self._matches: Deque[SignatureMatch] = deque(maxlen=max_points)
        self._profiles: Dict[str, EntityProfile] = {}
        self._observations: Deque[Observation] = deque(maxlen=max_points)
        self._systems: set[str] = set()
        self._entities: Dict[str, set[str]] = defaultdict(set)

    # ------------------------------------------------------------------ ingest
    def add_observation(self, obs: Observation) -> None:
        with self._lock:
            self._observations.append(obs)
            self._systems.add(obs.system)
            self._entities[obs.system].add(obs.entity)

    def add_raw(self, m: RawMetric) -> None:
        with self._lock:
            self._raw[m.key].append(m)
            self._systems.add(m.system)
            self._entities[m.system].add(m.entity)

    def add_derived(self, m: DerivedMetric) -> None:
        with self._lock:
            self._derived[m.key].append(m)

    def add_event(self, e: BehaviorEvent) -> None:
        with self._lock:
            self._events.append(e)

    def add_match(self, m: SignatureMatch) -> None:
        with self._lock:
            self._matches.append(m)

    def put_profile(self, p: EntityProfile) -> None:
        with self._lock:
            self._profiles[f"{p.system}|{p.entity}"] = p

    # ------------------------------------------------------------------- query
    def systems(self) -> List[str]:
        with self._lock:
            return sorted(self._systems)

    def entities(self, system: str) -> List[str]:
        with self._lock:
            return sorted(self._entities.get(system, set()))

    def raw_series(self, system: str, entity: str, name: str) -> List[RawMetric]:
        with self._lock:
            return list(self._raw.get(f"{system}|{entity}|{name}", ()))

    def derived_series(self, system: str, entity: str, name: str) -> List[DerivedMetric]:
        with self._lock:
            return list(self._derived.get(f"{system}|{entity}|{name}", ()))

    def raw_names(self, system: str, entity: str) -> List[str]:
        prefix = f"{system}|{entity}|"
        with self._lock:
            return sorted(k[len(prefix):] for k in self._raw if k.startswith(prefix))

    def derived_names(self, system: str, entity: str) -> List[str]:
        prefix = f"{system}|{entity}|"
        with self._lock:
            return sorted(k[len(prefix):] for k in self._derived if k.startswith(prefix))

    def latest_raw(self, system: str, entity: str, name: str) -> Optional[RawMetric]:
        s = self.raw_series(system, entity, name)
        return s[-1] if s else None

    def latest_derived(self, system: str, entity: str, name: str) -> Optional[DerivedMetric]:
        s = self.derived_series(system, entity, name)
        return s[-1] if s else None

    def snapshot(self, system: str, entity: str) -> Dict[str, float]:
        """Latest numeric value of every raw+derived metric for an entity.
        This flat name->value map is what the signature engines match against,
        so they need no knowledge of how the values were produced."""
        snap: Dict[str, float] = {}
        with self._lock:
            for name in self.raw_names(system, entity):
                m = self.latest_raw(system, entity, name)
                if m is not None and isinstance(m.value, (int, float)):
                    snap[name] = float(m.value)
            for name in self.derived_names(system, entity):
                m = self.latest_derived(system, entity, name)
                if m is not None and isinstance(m.value, (int, float)):
                    snap[name] = float(m.value)
        return snap

    def profile(self, system: str, entity: str) -> Optional[EntityProfile]:
        with self._lock:
            return self._profiles.get(f"{system}|{entity}")

    def all_profiles(self, system: Optional[str] = None) -> List[EntityProfile]:
        with self._lock:
            ps = list(self._profiles.values())
        return [p for p in ps if system is None or p.system == system]

    def events(self, system: Optional[str] = None, entity: Optional[str] = None,
               limit: int = 200) -> List[BehaviorEvent]:
        with self._lock:
            out = [e for e in self._events
                   if (system is None or e.system == system)
                   and (entity is None or e.entity == entity)]
        return out[-limit:][::-1]

    def matches(self, system: Optional[str] = None, entity: Optional[str] = None,
                limit: int = 200) -> List[SignatureMatch]:
        with self._lock:
            out = [m for m in self._matches
                   if (system is None or m.system == system)
                   and (entity is None or m.entity == entity)]
        return out[-limit:][::-1]

    def recent_observations(self, limit: int = 500) -> List[Observation]:
        with self._lock:
            return list(self._observations)[-limit:]
