"""FeatureVectorEngine (B01) — the per-tick representation every lib-3 engine reads.

For each real entity and tick it writes, at ts = ctx.now:

  feature.vec[52]    FEATURE_SPEC v2 transformed row (lib/features.compute_features)
  feature.nat[52]    the same features in natural units (rates, shares, ms ...)
  feature.sketch[80] signed-hash Hellinger sketch of five categorical namespaces
  feature.active[1]  1 iff the entity had an observation in (now - dt, now]
  feature.expo       {http, dns, tls, flows, probe: n} exposure (dict series)
  feature.tctx       {hour_local, dow, day_type, bin48, bin168, slot, daypart, cc}

and refreshes profile.fingerprint / stable / updated. Nothing here learns and
nothing reads history: the row is a pure function of what was written THIS
tick, so replay and cadence changes cannot make it drift.

Why these rules (architecture §0.3, contract A):
  * Freshness. Every source is read with store.latest_fresh (value only if it
    was written at ctx.now). v1 read the latest value whatever its age, so a
    silent entity kept "emitting" its last ratio forever. Now a stale count
    is a true 0 (absence is data) and a stale ratio / average / bounded value
    is NaN (a ratio of nothing is undefined, not 0).
  * Cadence independence. Counts are per-minute rates (log1p(v*60/dt)) with
    dt = ctx.window_s, the real Δt of this tick, so a 900 -> 60 s switch does
    not shift any count, avg, bounded, window or clr column. Ratio columns
    are exposure dependent by design (the +0.5/+1 smoothing); B04 handles
    exposure through the Beta-Binomial predictive and feature.expo.
  * Activity. active = (store.last_seen == now). Raw engines stamp every
    point at ctx.now and touch last_seen, while zero-fills (R2 act.events)
    and our own probes do not; the v1 rule "an event in the last dt/2" missed
    entities whose events all fell in the first half of a tick.
  * Storage. feature.<name> scalars are virtual column views of feature.vec
    (store.register_vector_names), not 52 DerivedMetric objects per tick:
    that duplication was the v1 memory blow-up. Single-owner scalars and
    vectors are float32 rings, dict outputs are derived points (helpers_api
    §0.1). The ring dtype is float32, so profile.fingerprint is taken from
    the float32-rounded row: the profile and the ring always agree.
  * Time context is computed once per tick from ctx.config (tz, calendar with
    holidays / 调休 make-up workdays, daypart_day_hours) through
    lib/timebins, the single place that defines "Monday 09:00 local".

The sketch reads fresh raw sets only (a stale namespace is an all-zero block):

  act.tokens            (R2)  -> namespace act.tokens
  client.stack_set      (R3)  -> namespace client.stack_set  ({tok: {n, ...}})
  tls.sni_etld1_set     (R1)  -> namespace sni_etld1
  dns.qname_etld1_set   (R1)  -> namespace dns_etld1
  l4.dport_set          (R1)  -> namespace l4.dport_set

Profile: fingerprint = vec (NaN -> None); stable = model.baseline n_eff >= 96
and calibration healthy (profile.extra.calibration.healthy from B24; while
B24 has not described the entity yet there is no evidence of miscalibration
and only n_eff decides); updated = now. The v1 12-sample flag is gone.

B01 never emits events, so ctx.training changes nothing here.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import DerivedMetric, EntityProfile, MetricKind
from .lib import features as F
from .lib import sketch as SK
from .lib import timebins as TB

VEC = "feature.vec"
NAT = "feature.nat"
SKETCH = "feature.sketch"
ACTIVE = "feature.active"
EXPO = "feature.expo"
TCTX = "feature.tctx"
VIRTUAL_PREFIX = "feature."
BASELINE = "model.baseline"

STABLE_N_EFF = 96.0

# sketch namespace -> raw series carrying its {token: count} set (R1-R3 names)
SKETCH_SOURCES: Tuple[Tuple[str, str], ...] = (
    ("act.tokens", "act.tokens"),
    ("client.stack_set", "client.stack_set"),
    ("sni_etld1", "tls.sni_etld1_set"),
    ("dns_etld1", "dns.qname_etld1_set"),
    ("l4.dport_set", "l4.dport_set"),
)
if tuple(ns for ns, _ in SKETCH_SOURCES) != SK.SKETCH_NAMESPACES:   # block order is the contract
    raise ImportError("feature_vector.SKETCH_SOURCES out of sync with sketch.SKETCH_NAMESPACES")

FEATURE_NAMES: List[str] = list(F.FEATURE_NAMES_V2)


def _n_eff(model: Any) -> float:
    """model.baseline n_eff (contract C) as a total: a finite scalar, or the
    finite sum of a per-bucket sequence / mapping; 0 when absent. Same reading
    as B24's maturity gate, so 'stable' and 'mature' never disagree."""
    if model is None:
        return 0.0
    v = model.get("n_eff") if isinstance(model, Mapping) else getattr(model, "n_eff", None)
    return _total(v)


def _total(v: Any) -> float:
    if v is None or isinstance(v, bool):
        return 0.0
    if isinstance(v, Mapping):
        return sum(_total(x) for x in v.values())
    try:
        a = np.asarray(v, dtype=np.float64)
    except (TypeError, ValueError):
        return 0.0
    if a.ndim == 0:
        x = float(a)
        return x if math.isfinite(x) else 0.0
    return float(np.sum(a[np.isfinite(a)]))


def _calibration_healthy(prof: EntityProfile) -> bool:
    info = prof.extra.get("calibration") if isinstance(prof.extra, dict) else None
    if isinstance(info, Mapping) and "healthy" in info:
        return bool(info["healthy"])
    return True


def read_feature_matrix(store, system: str, entity: str, window: int = 240):
    """LEGACY shim for the v1 engines (baseline/fingerprint/anomaly/drift v1,
    deleted at integration): (rows, ts) from the feature.vec ring, NaN -> 0
    because those engines assume dense rows. lib-3 engines read the ring."""
    ts, M = store.vec_tail(system, entity, VEC, window)
    if not len(ts):
        return [], []
    rows = np.nan_to_num(M.astype(np.float64), nan=0.0).tolist()
    return rows, [float(t) for t in ts]


class FeatureVectorEngine(Engine):
    name = "behavior.feature_vector"
    layer = "behavior"
    consumes = ["l3.*", "l4.*", "http.*", "tls.*", "dns.*", "probe.*", "act.*",
                "client.stack_set", "derived.*", "model.baseline", "profile.extra.calibration"]
    produces = [VEC, NAT, EXPO, ACTIVE, TCTX, SKETCH, "feature.*",
                "profile.fingerprint", "profile.stable", "profile.updated"]
    description = ("Presence-aware, exposure-carrying, cadence-independent per-tick feature "
                   "vector (FEATURE_SPEC v2), categorical sketch and local time context.")
    interval = 1

    def run(self, ctx: Context, observations=None) -> int:
        store = ctx.store
        now = float(ctx.now)
        dt = float(ctx.window_s)
        if not (math.isfinite(dt) and dt > 0):
            raise ValueError(f"FeatureVectorEngine: bad ctx.window_s {ctx.window_s!r}")
        # idempotent: feature.<name> becomes a virtual column of feature.vec
        store.register_vector_names(VEC, FEATURE_NAMES, VIRTUAL_PREFIX)
        tctx = TB.tctx_from_config(now, ctx.config, dt)   # same for every entity
        n = 0
        for s in store.systems():
            for e in store.entities(s):
                self._entity(store, s, e, now, dt, tctx)
                n += 1
        return n

    # ------------------------------------------------------------ one entity
    def _entity(self, store, s: str, e: str, now: float, dt: float,
                tctx: Dict[str, Any]) -> None:
        cache: Dict[str, Any] = {}

        def get(metric: str) -> Optional[Any]:
            # several features share a source (http.requests is read ~12x)
            try:
                return cache[metric]
            except KeyError:
                v = cache[metric] = store.latest_fresh(s, e, metric, now)
                return v

        vec, nat = F.compute_features(get, dt)
        expo = F.exposure(get)
        ns_counts = {}
        for ns, metric in SKETCH_SOURCES:
            v = store.latest_fresh(s, e, metric, now)
            if isinstance(v, Mapping) and v:
                ns_counts[ns] = v
        sk = SK.sketch_vector(ns_counts)
        active = 1.0 if store.last_seen(s, e) == now else 0.0

        w = int(round(dt))
        vec32 = vec.astype(np.float32)
        store.add_vec(s, e, VEC, now, vec32, window_s=w)
        store.add_vec(s, e, NAT, now, nat.astype(np.float32), window_s=w)
        store.add_vec(s, e, SKETCH, now, sk, window_s=w)
        store.add_vec(s, e, ACTIVE, now, np.array([active], dtype=np.float32), window_s=w)
        store.add_derived(DerivedMetric(
            name=EXPO, value=expo, ts=now, system=s, entity=e, window_s=w,
            kind=MetricKind.CATEGORICAL, inputs=list(F.EXPOSURE_SOURCE.values())))
        store.add_derived(DerivedMetric(
            name=TCTX, value=dict(tctx), ts=now, system=s, entity=e, window_s=w,
            kind=MetricKind.CATEGORICAL, inputs=["config.tz", "config.calendar"]))

        prof = store.profile(s, e)
        if prof is None:
            prof = EntityProfile(system=s, entity=e, updated=now)
        if prof.feature_names != FEATURE_NAMES:
            prof.feature_names = list(FEATURE_NAMES)
        prof.fingerprint = [None if x != x else x for x in vec32.tolist()]
        prof.stable = (_n_eff(store.get_model(s, e, BASELINE)) >= STABLE_N_EFF
                       and _calibration_healthy(prof))
        prof.updated = now
        store.put_profile(prof)
