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

spec v2.1 grains (docs/lib3/cadence.md §2-§5; canonical grain mode only, tick
mode writes nothing new). Every tick B01 also writes the 47 additive parts
of the tick (feature.part, lib/features.compute_parts) and the ROLLING nat
rows of each observable grain over (now - G, now] (feature.live.h, .q: the
additive features every tick, set / map / span columns only on decision
ticks). On a grain's DECISION tick (lib/grains.decision: the tick's
interval holds an epoch multiple of G, the same for every entity) it writes
the complete grain row that every learner and scorer consumes:

  feature.nat.<g>[52]   grain values in natural units (features.grain_values:
                        additive features from the part sums with dt := cov,
                        distinct counts from the union of the per-tick
                        SetSketches, entropies / concentration from the
                        merged count maps, span features at their span
                        decision ticks only)
  feature.vec.<g>[52]   the FEATURE_SPEC transform with dt := cov
  feature.meta.<g>[5]   [active, cov_s, n_ticks, n_active, G]
  feature.expo.<g>      {http, dns, tls, flows, probe: sum n}
  feature.sketch.h[80]  the categorical sketch of the merged H-window maps

The presence of feature.nat.<g> at ts = now is the decision flag. A row is
only as wide as the ticks it covers (cov = sum of their dt).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import DerivedMetric, EntityProfile, MetricKind
from .lib import features as F
from .lib import grains as GR
from .lib import pactive as PA
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

# spec v2.1 grain series (cadence.md §5.2)
PART = "feature.part"
LIVE = "feature.live"
META = "feature.meta"
RAW_SET_KEEP_S = 75 * 60.0      # map / sketch raw sets must cover G_h + max dt

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
        grain = None
        if GR.canonical(ctx.config):
            self._ensure_grain_retention(store)
            log = self.__dict__.setdefault("_tick_log", [])
            if not log or now > log[-1][0]:
                log.append((now, dt))
            while log and log[0][0] <= now - 2 * GR.GRAIN_S["h"]:
                log.pop(0)
            cov_log = {}
            for g in GR.GRAINS:
                lo = now - GR.GRAIN_S[g] + 1e-3
                sel = [d for t, d in log if t >= lo]
                cov_log[g] = (float(sum(sel)), len(sel))
            grain = {"cov_log": cov_log,
                "due": GR.due(now, dt, GR.CANONICAL),
                "obs": {g: GR.observable(g, dt, GR.CANONICAL) for g in GR.GRAINS},
                "span": {f: GR.span_decision(now, dt, f, ctx.config) for f in GR.SPAN_S},
            }
        n = 0
        for s in store.systems():
            # bounded mode (§10.3): rows for the active and earned IPs only; an
            # unearned idle IP's row is implicit (feature.active = 0)
            for e in PA.entities(store, s, now, ctx.config):
                self._entity(store, s, e, now, dt, tctx, grain)
                n += 1
        return n

    def _ensure_grain_retention(self, store) -> None:
        """Raise-only: the raw sets merged over an H window (maps, sketch
        namespaces) must outlive G_h + dt."""
        if getattr(self, "_ret_store", None) is store:
            return
        for name in set(F.MAP_RAW_SETS) | {m for _, m in SKETCH_SOURCES} | {
                "act.stream", "act.tokens", "tls.sni_etld1_set", "dns.qname_etld1_set",
                "l4.dport_set"}:
            store.ensure_retention(name, max_age_s=RAW_SET_KEEP_S)
        self._ret_store = store

    # ------------------------------------------------------------ one entity
    def _entity(self, store, s: str, e: str, now: float, dt: float,
                tctx: Dict[str, Any], grain: Optional[Dict[str, Any]] = None) -> None:
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

        if grain is not None:
            self._grains(store, s, e, now, dt, get, active, grain)

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


    # ------------------------------------------------------ spec v2.1 grains
    def _grains(self, store, s: str, e: str, now: float, dt: float, get, active: float,
                grain: Dict[str, Any]) -> None:
        """feature.part every tick; feature.live.<g> every tick where g is
        observable; the decision rows of each due grain (module docstring)."""
        w = int(round(dt))
        parts = F.compute_parts(get, dt, active=active)
        store.add_vec(s, e, PART, now, parts.astype(np.float32), window_s=w)
        span_now: Optional[Dict[str, Optional[float]]] = None
        for g in GR.GRAINS:
            if not grain["obs"][g]:
                continue
            G = GR.GRAIN_S[g]
            lo = now - G + 1e-3
            ts, P = store.vec_since(s, e, PART, lo)
            if not len(ts):
                continue
            P = np.asarray(P, dtype=np.float64)
            S = P.sum(axis=0)
            # coverage is the wall clock the pipeline covered (its own tick log):
            # before an entity's first appearance its parts are true zeros
            cl, nl = grain["cov_log"][g]
            cov = max(float(S[0]), cl)
            S[0] = cov
            n_ticks = max(int(len(ts)), nl)
            n_act = int(np.count_nonzero(P[:, 1] > 0.5))
            due = grain["due"][g]
            if due:
                sets = {f: SK.set_count(SK.set_union(_raw_window(store, s, e, src, lo, now)))
                        for f, src in F.SET_SOURCES.items()}
                maps = {src: _merge_maps(_raw_window(store, s, e, src, lo, now))
                        for src in F.MAP_RAW_SETS}
                span: Dict[str, Optional[float]] = {}
                if g == "h":
                    if span_now is None:
                        span_now = {name: (store.latest_fresh(s, e, F.FEATURE_SOURCE[name], now)
                                           if grain["span"].get(name) else None)
                                    for name in F.SPAN_FEATURES}
                    span = span_now
                vec, nat = F.grain_values(S, cov, G, sets=sets, maps=maps, span=span)
            else:
                vec, nat = F.grain_values(S, cov, G)
            nat32 = nat.astype(np.float32)
            store.add_vec(s, e, f"{LIVE}.{g}", now, nat32, window_s=w)
            if not due:
                continue
            meta = np.array([1.0 if n_act > 0 else 0.0, cov, n_ticks, n_act, G],
                            dtype=np.float32)
            store.add_vec(s, e, f"{NAT}.{g}", now, nat32, window_s=w)
            store.add_vec(s, e, f"{VEC}.{g}", now, vec.astype(np.float32), window_s=w)
            store.add_vec(s, e, f"{META}.{g}", now, meta, window_s=w)
            store.add_derived(DerivedMetric(
                name=f"{EXPO}.{g}", value=F.grain_exposure(S), ts=now, system=s, entity=e,
                window_s=w, kind=MetricKind.CATEGORICAL, inputs=[PART]))
            if g == "h":
                ns_counts = {}
                for ns, metric in SKETCH_SOURCES:
                    mm = _merge_maps(_raw_window(store, s, e, metric, lo, now))
                    if mm:
                        ns_counts[ns] = mm
                store.add_vec(s, e, f"{SKETCH}.h", now, SK.sketch_vector(ns_counts), window_s=w)


def _raw_window(store, s: str, e: str, name: str, lo: float, hi: float) -> List[Any]:
    """Values of raw `name` with lo <= ts <= hi (oldest first)."""
    n = 8
    while True:
        tail = store.raw_tail(s, e, name, n)
        if not tail or len(tail) < n or tail[0].ts < lo or n >= 1 << 16:
            break
        n *= 4
    return [m.value for m in tail if lo <= m.ts <= hi]


def _merge_maps(values: List[Any]) -> Dict[str, float]:
    """Sum count maps key by key ('__other__' summed like any key; a value
    that is itself a mapping contributes its 'n', as sketch_block reads it)."""
    out: Dict[str, float] = {}
    for v in values:
        if not isinstance(v, Mapping):
            continue
        for k, c in v.items():
            if isinstance(c, Mapping):
                c = c.get("n")
            try:
                x = float(c)
            except (TypeError, ValueError):
                continue
            if x > 0.0 and math.isfinite(x):
                out[k] = out.get(k, 0.0) + x
    return out
