"""Shared test helpers: build synthetic MetricStores so any engine can be unit
tested in isolation (engines only talk to the store, so this is all they need).
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.core.engine import Context, Engine  # noqa: E402
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


def ctx(store: MetricStore, now: float, training: bool = False, window_s: float = 60,
        config: Optional[Dict[str, Any]] = None) -> Context:
    return Context(store=store, now=now, window_s=window_s, training=training,
                   config=dict(config or {}))


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


# --------------------------------------------------------------------------- #
# lib-3 v2 helpers (store v2 / engine core)
# --------------------------------------------------------------------------- #
def add_vec_rows(store: MetricStore, system: str, entity: str, name: str,
                 rows: Iterable[Sequence[float]], t0: float = T0, dt: float = DT,
                 register: bool = True) -> List[float]:
    """Append rows to a float32 vector ring at t0, t0+dt, ...; returns the
    timestamps. Real entities are registered (no raw data, first/last_seen
    untouched) so engines iterating store.entities() see them."""
    ts = []
    for i, row in enumerate(rows):
        t = t0 + i * dt
        store.add_vec(system, entity, name, t, np.asarray(row, dtype=np.float32),
                      window_s=int(dt))
        ts.append(t)
    if register:
        store.register_entity(system, entity)
    return ts


def add_obs_tick(store: MetricStore, system: str, entity: str, ts: float,
                 metrics: Dict[str, Any], touch: bool = True) -> None:
    """Write one tick of raw metrics at ts, as a raw engine would (ts = now,
    touching first_seen/last_seen)."""
    for name, v in metrics.items():
        store.add_raw(RawMetric(name=name, value=v, ts=ts, system=system, entity=entity,
                                method=AcquisitionMethod.PASSIVE_SPAN), touch=touch)


def put_model(store: MetricStore, system: str, entity: str, name: str, obj: Any,
              version: Any = None) -> Any:
    store.put_model(system, entity, name, obj, version=version)
    return obj


DAYPARTS = ("wd_day", "wd_night", "nwd_day", "nwd_night")
CADENCE_CLASSES = (60, 300, 900, 3600)


def make_tctx(ts: float, tz: str = "Asia/Shanghai", dt: float = DT,
              holidays: Sequence[str] = (), makeup_workdays: Sequence[str] = (),
              day_hours: Sequence[int] = (8, 20)) -> Dict[str, Any]:
    """Reference time context for tests (contract B feature.tctx). Delegates
    to lib.timebins.tctx so tests and engines can never disagree on the
    encodings: hour_local (float), dow (Mon=0), day_type (workday|nonworkday,
    holiday and 调休 aware; dates as 'YYYY-MM-DD'), bin48 = hour +
    24*nonworkday, bin168 = hour of week, slot = 15-min slots since the local
    epoch, daypart and the log-nearest cadence class cc. daypart_id indexes
    DAYPARTS for numeric rings."""
    from app.engines.behavior.lib import timebins as TB
    cal = TB.parse_calendar({"holidays": list(holidays),
                             "makeup_workdays": list(makeup_workdays)})
    out = dict(TB.tctx(float(ts), tz, cal, tuple(day_hours), float(dt)))
    out["daypart_id"] = DAYPARTS.index(out["daypart"])
    return out


def set_trust(store: MetricStore, system: str, entity: str, ts_list: Iterable[float],
              value: float, prov: Optional[float] = None,
              quarantine: Optional[float] = None) -> None:
    """Write behavior.trust (and trust_prov, quarantine) at each ts, as the
    governor (B28) would: 1-element float32 vec rings (helpers_api.md §0,
    read by lib.gating via vec_at / vec_since). prov defaults to value;
    quarantine to 0."""
    prov = value if prov is None else prov
    q = 0.0 if quarantine is None else quarantine
    for t in ts_list:
        for name, v in (("behavior.trust", value), ("behavior.trust_prov", prov),
                        ("behavior.quarantine", q)):
            store.add_vec(system, entity, name, t, np.asarray([float(v)], dtype=np.float32),
                          window_s=int(DT))


def run_engine(engine: Engine, store: MetricStore, now: float, training: bool = False,
               dt: float = DT, config: Optional[Dict[str, Any]] = None,
               observations: Optional[List[Observation]] = None,
               scheduled: bool = False) -> int:
    """Run one engine for one tick with ctx.config['strict'] = True, so any
    exception fails the test. By default it bypasses interval/period
    scheduling (calls engine.run); scheduled=True goes through safe_run."""
    cfg = {"strict": True}
    cfg.update(config or {})
    c = Context(store=store, now=now, window_s=dt, training=training, config=cfg)
    if scheduled:
        return engine.safe_run(c, observations)
    return engine.run(c, observations)
