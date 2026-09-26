"""Shared test helpers: build synthetic MetricStores so any engine can be unit
tested in isolation (engines only talk to the store, so this is all they need).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.core.engine import Context  # noqa: E402
from app.core.store import MetricStore  # noqa: E402
from app.models.schema import (  # noqa: E402
    AcquisitionMethod,
    DerivedMetric,
    EntityProfile,
    MetricKind,
    Observation,
    RawMetric,
)

T0 = 1_700_000_000.0          # fixed epoch so tests are deterministic
DT = 900.0                    # 15-minute ticks by default


def make_store() -> MetricStore:
    return MetricStore()


def ctx(store: MetricStore, now: float, training: bool = False, window_s: int = 60) -> Context:
    return Context(store=store, now=now, window_s=window_s, training=training)


def add_raw_series(store: MetricStore, system: str, entity: str, name: str,
                   values: Sequence, t0: float = T0, dt: float = DT,
                   kind: MetricKind = MetricKind.GAUGE) -> List[float]:
    """Append a raw series; returns the timestamps used."""
    ts = []
    for i, v in enumerate(values):
        t = t0 + i * dt
        store.add_raw(RawMetric(name=name, value=v, ts=t, system=system, entity=entity,
                                kind=kind, method=AcquisitionMethod.PASSIVE_SPAN))
        ts.append(t)
    return ts


def add_derived_series(store: MetricStore, system: str, entity: str, name: str,
                       values: Sequence, t0: float = T0, dt: float = DT) -> List[float]:
    ts = []
    for i, v in enumerate(values):
        t = t0 + i * dt
        store.add_derived(DerivedMetric(name=name, value=v, ts=t, system=system,
                                        entity=entity, window_s=int(dt)))
        ts.append(t)
    return ts


def add_feature_rows(store: MetricStore, system: str, entity: str,
                     rows: Iterable[Sequence[float]], names: Optional[List[str]] = None,
                     t0: float = T0, dt: float = DT) -> List[float]:
    """Write a feature matrix as `feature.<name>` series (the lib-3 contract)
    and register the entity so store.entities() sees it."""
    from app.engines.behavior.features import FEATURE_NAMES
    names = names or FEATURE_NAMES
    rows = [list(r) for r in rows]
    ts = []
    for i, row in enumerate(rows):
        t = t0 + i * dt
        for name, v in zip(names, row):
            store.add_derived(DerivedMetric(name=f"feature.{name}", value=float(v), ts=t,
                                            system=system, entity=entity, window_s=int(dt)))
        ts.append(t)
    # make the entity visible to engines iterating store.entities(system)
    store.add_raw(RawMetric(name="l4.flows", value=1.0, ts=t0, system=system, entity=entity))
    return ts


def put_profile(store: MetricStore, system: str, entity: str, **fields) -> EntityProfile:
    p = store.profile(system, entity) or EntityProfile(system=system, entity=entity)
    for k, v in fields.items():
        setattr(p, k, v)
    store.put_profile(p)
    return p


def obs(system: str, entity: str, ts: float, **fields) -> Observation:
    return Observation(ts=ts, system=system, entity=entity, **fields)
