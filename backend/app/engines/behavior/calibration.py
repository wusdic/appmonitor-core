"""B24 CalibrationEngine: the single owner of detector p-values (behavior.p).

Why a separate calibration stage: the 31 detectors emit raw scores on
incomparable scales (-log10 p, CUSUM statistics, JSD, bits, counts), several
with a point mass at their minimum (silence 0, CUSUM 0, novelty 0), and their
null distributions shift with the time of day and with the tick cadence.
Fusion (B25) can only combine and budget evidence if every score is first
turned into a p-value that is uniform under the entity's own null, valid in
finite samples, meaningful far into the tail and never 1 by construction.

How (docs/lib3/engines.md B24):
  * Mondrian rings. Per (entity or class key, detector, stratum) a sorted
    ring of the last M = 256 null scores, each with its tick ts (lib/calib
    Ring). stratum = daypart x cadence class; identity uses (daypart, regime
    tercile) (lib/m_calib.regime_tercile). Night scores are therefore never
    compared with the day ring, and a 60-s score never with 900-s history.
  * Randomised conformal p = (#{c > s} + U (#{c == s} + 1)) / (|C| + 1),
    U = combine.seeded_uniform(s, e, d, ts) so replay (B29) is bit-exact.
    Ties at a sparse detector's zero spread uniformly instead of piling at 1.
  * GPD tail. Above u = q_0.90(C), with >= 10 exceedances, a PWM-GPD fit
    gives p far below 1/(M+1). The fit is refreshed once a ring has taken 16
    additions since its last fit, lazily: at the first later tick whose score
    reaches the tail (x >= the ring's 0.9 order statistic, or above the old
    fit's u). Every tail actually used is therefore at most 16 additions old,
    as with an eager refit every 16 ticks, while a null ring (90 % body
    scores) refits ~40 % less often; a fit costs ~80 us, the largest per-tick
    item. Counters start at a per-ring phase so refits spread over ticks. Two
    guards from lib/calib (see its docstring for the measurements): xi is
    floored at 0 (a spuriously bounded fit would give p = 1e-300 to the next
    null tick past its end point), and the tail p is capped at 1/(n+1)
    beyond the ring maximum.
  * Small samples (|C| < 64, e.g. a new stratum after a cadence switch): the
    conformal p is logit-blended with weight n/(n+64) against the first
    usable prior of behavior.pm[d] (exposure-exact model p; accumulators write
    their stationary-tail p_eq there), the class-pooled ring (the union of the
    role members' rings for the same key, >= 64 entries) and, for identity
    only, the entity's own settled-regime ring of the same daypart.
  * Admission (contract H). Rings learn through lib/gating: a scored tick is
    committed D = max(4 ticks, 600 s) later with its trust weight, held while
    the entity is quarantined, released / rebased / frozen by model.control.
    A ring cannot take a fractional weight, so a committed row is admitted
    with probability trust (seeded thinning, deterministic under replay):
    dropping every partially trusted tick would cut the null's upper tail
    out of the ring and make p anti-conservative. Admission also needs entity
    maturity n_eff >= 48 (model.baseline n_eff, or the engine's own count of
    trusted commits in 15-minute equivalents, whichever is larger): scores
    of an immature model do not describe the entity's null.
  * Rollback deletes ring entries with ts > rollback_to (the ring carries
    ts). Checkpoint + replay, the generic GatedLearner route, cannot work
    here: behavior.score is retained for 1 d, so replaying committed rows
    from a checkpoint would silently drop them, and checkpointing ~100 rings
    per entity hourly would cost megabytes per entity. Journal rows after
    the onset move to held exactly as in the generic learner.
  * A model.control version change resets the rings (the new regime starts
    its own null); rebase replays the held rows after the reset.
  * Health (per detector per system, behavior.calib_health@(s, __system__)):
    KS D of the randomised p on trusted committed ticks (last 2048, needs
    >= 1024) and the realised rate of e_day <= 0.03 against its expectation
    on all committed ticks (decayed, half-life 14 d, needs >= 10 expected).
    The rate uses all non-quarantined commits, not only trusted ones: an
    exceedance lowers its own tick's trust, so on trusted ticks the realised
    rate would be ~0 and every detector would look broken. KS D > 0.05 or a
    rate ratio outside [0.5, 2] gives weight_mult = 0.5 (lib/calib).
  * NaN score (degraded / unscored) gives NaN p, never 1. behavior.p is a
    float32 ring, so an issued p is floored at the smallest normal float32
    (1.2e-38) instead of calib's 1e-300: a stored 0 would read as
    "impossible" downstream. B24 emits no events, so training mode only
    changes the trust default of gating.

Store: reads behavior.score, behavior.pm, behavior.p (own, at commit time,
for health), behavior.trust / trust_prov / quarantine (via gating),
feature.tctx, behavior.regime, model.control, model.link, model.class,
model.baseline (n_eff only); writes behavior.p (emit.write_pvalues),
model.calib@(s, e | class:<id>) and @(s, __system__) (health state),
behavior.calib_health@(s, __system__) (dict series, one point per tick,
the value object changes only when re-evaluated, hourly) and
profile.extra.calibration. The layout of model.calib is documented in
lib/m_calib.py, the accessor module consumers use.
"""
from __future__ import annotations

import math
import weakref
import zlib
from bisect import bisect_left, bisect_right
from collections import deque
from collections.abc import Mapping      # not typing.Mapping: isinstance is on the hot path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import DerivedMetric, EntityProfile, MetricKind
from .lib import calib, combine, emit, gating, m_calib, m_class, timebins
from .lib.classkeys import CLASS_PREFIX, SYSTEM_KEY
from .lib.detectors import DETECTOR_INDEX, DETECTORS, N_DETECTORS

MODEL = m_calib.MODEL
RINGS = m_calib.RINGS
SCORE, PM, P = emit.SCORE, emit.PM, emit.P
TCTX = "feature.tctx"
REGIME = "behavior.regime"
CALIB_HEALTH = "behavior.calib_health"
BASELINE = "model.baseline"
LEARNER = "calibration"

SMALL_N = calib.SMALL_N                 # 64
GPD_REFIT_TICKS = calib.GPD_REFIT_TICKS  # 16
TAIL_MIN_N = int(math.ceil(calib.MIN_EXCEED / (1.0 - calib.TAIL_Q)))   # 100: a fit is possible
MATURITY_N_EFF = 48.0                   # admission needs entity n_eff >= 48
MATURITY_UNIT_S = 900.0                 # own count in 15-minute tick equivalents
SCORE_RETENTION_S = 86400.0             # behavior.score is kept 1 d (contract B)
HEALTH_E_DAY = 0.03                     # realised-rate check at e_day <= 0.03
HEALTH_KS_CAP = 2048                    # trusted p kept per detector per system
HEALTH_KS_MIN = 1024                    # KS D > 0.05 is ~1 % null at n = 1024
HEALTH_RATE_HL_S = 14 * 86400.0         # decay of exceedance / expected counts
HEALTH_RATE_MIN_EXPECTED = 10.0         # Poisson(10) stays in [0.5, 2]x ~97 %
HEALTH_EVAL_S = 3600.0                  # re-evaluate health hourly
HEALTH_RETENTION_S = 8 * 86400.0
PROFILE_EVERY_S = 3600.0                # profile.extra.calibration refresh
ADMIT_SALT = "b24.admit"
P_ISSUED_FLOOR = m_calib.P_ISSUED_FLOOR  # behavior.p is float32: no stored 0 (m_calib)

_ID_IDX = DETECTOR_INDEX["identity"]
_DP_INDEX = {d: i for i, d in enumerate(timebins.DAYPARTS)}
_NAN = math.nan


class _Row(NamedTuple):
    """One committed tick as fetched for the learner."""
    s: str
    e: str
    ts: float
    scores: np.ndarray          # behavior.score row (float32[31])
    pvals: Optional[np.ndarray]  # behavior.p issued at ts (health)
    daypart: str
    tercile: int
    dt: float


# ----------------------------------------------------------------- pending
def _encode(daypart: str, tercile: int, dt: float) -> int:
    """Stratum issued at a tick as one small int: daypart, tercile and dt (ms)."""
    return _DP_INDEX[daypart] + 4 * int(tercile) + 12 * int(round(dt * 1000.0))


def _decode(code: int) -> Tuple[str, int, float]:
    c = int(code)
    return timebins.DAYPARTS[c % 4], (c // 4) % 3, (c // 12) / 1000.0


# ----------------------------------------------------------------- helpers
def _as_count(v: Any) -> float:
    """n_eff as a total: finite scalar, or the finite sum of a sequence / mapping."""
    if v is None or isinstance(v, bool):
        return 0.0
    if isinstance(v, Mapping):
        return sum(_as_count(x) for x in v.values())
    try:
        a = np.asarray(v, dtype=np.float64)
    except (TypeError, ValueError):
        return 0.0
    if a.ndim == 0:
        x = float(a)
        return x if math.isfinite(x) else 0.0
    return float(np.sum(a[np.isfinite(a)]))


def baseline_n_eff(store, s: str, e: str) -> float:
    """model.baseline n_eff (contract C) as a total; 0 when absent."""
    b = store.get_model(s, e, BASELINE)
    if b is None:
        return 0.0
    v = b.get("n_eff") if isinstance(b, Mapping) else getattr(b, "n_eff", None)
    return _as_count(v)


def _latest_value(store, s: str, e: str, name: str) -> Any:
    m = store.latest_derived(s, e, name)
    return None if m is None else m.value


def _tctx_at(store, s: str, e: str, ts: float, search: int = 1) -> Optional[Mapping]:
    """feature.tctx written at exactly ts (dict series, or its vec encoding)."""
    for m in reversed(store.derived_tail(s, e, TCTX, search)):
        if m.ts == ts and isinstance(m.value, Mapping):
            return m.value
        if m.ts < ts:
            break
    row = store.vec_at(s, e, TCTX, ts)
    if row is not None and len(row) == len(timebins.TCTX_FIELDS):
        return timebins.decode_tctx(row)
    return None


def _phase(e: str, key: str) -> int:
    """Initial refit counter of a new ring: spreads GPD refits over ticks."""
    return zlib.crc32(f"{e}|{key}".encode("utf-8")) % GPD_REFIT_TICKS


def _refit(r: calib.Ring, ts: float) -> None:
    r.gpd = calib.fit_tail(r, now_ts=ts)


def _tail_due(r: calib.Ring, count: int, x: float) -> bool:
    """A refit is due before scoring x: the fit is stale (>= 16 additions) and x
    may be judged by the tail. x >= scores[floor(0.9 (n - 1))] is a superset of
    x > q_0.90 (np.quantile interpolates above that order statistic); x above
    the old fit's u is where p_from_ring would use the stale tail."""
    n = r.scores.size
    if count < GPD_REFIT_TICKS or n < TAIL_MIN_N:
        return False
    g = r.gpd
    return x >= r.scores[int(calib.TAIL_Q * (n - 1))] or (g is not None and x > g.u)


def new_model() -> Dict[str, Any]:
    """An empty B24 part of model.calib (layout in lib/m_calib.py)."""
    return {"layout": m_calib.LAYOUT, "version": 0, RINGS: {}, "refit": {},
            "gate": gating.GateState(), "pending": {}, "n_own": 0.0, "n_admit": 0,
            "resets": 0, "dt": _NAN, "profile_ts": -math.inf}


def _ensure_layout(model: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Add the B24 keys to a model.calib dict in place (B25 may have created it
    with its meta rings only); stored ring dicts become live Rings."""
    if model is None:
        return new_model()
    if not isinstance(model, dict):
        raise TypeError(f"model.calib must be a dict, got {type(model).__name__}")
    if isinstance(model.get("gate"), gating.GateState) and "pending" in model:
        return model                        # steady state: already ours and live
    for k, v in new_model().items():
        model.setdefault(k, v)
    rs = model[RINGS]
    if any(not isinstance(r, calib.Ring) for r in rs.values()):
        model[RINGS] = m_calib.rings(model)
    if not isinstance(model["gate"], gating.GateState):
        model["gate"] = gating.GateState.from_dict(model["gate"])
    # JSON round trips turn the float ts keys into strings and non-finite
    # numbers into None; keep ts order and restore the defaults
    pend = model["pending"] or {}
    model["pending"] = {t: int(c) for t, c in sorted((float(k), c) for k, c in pend.items())}
    for k, v in new_model().items():
        if isinstance(v, (int, float)) and not isinstance(model[k], (int, float)):
            model[k] = v
    model["refit"] = {k: int(c) for k, c in (model["refit"] or {}).items() if c is not None}
    return model


def _reset_rings(model: Dict[str, Any]) -> None:
    """Version change: the new regime starts an empty null (new dicts, so a
    snapshot holding the old ones is untouched)."""
    model[RINGS] = {}
    model["refit"] = {}
    model["resets"] = int(model.get("resets", 0)) + 1


def remove_after(model: Dict[str, Any], tau: float) -> int:
    """Delete ring entries with ts > tau (rollback). A ring that lost entries
    also lost its fit (calib.Ring.remove_after); it is refitted before its
    tail is next used."""
    removed = 0
    for key, r in model[RINGS].items():
        k = r.remove_after(tau)             # drops the fit when anything is removed
        if k:
            removed += k
            model["refit"][key] = GPD_REFIT_TICKS     # refit before the tail is used
    return removed


# ----------------------------------------------------------------- health
def new_health() -> Dict[str, Any]:
    return {"t": _NAN, "exc": np.zeros(N_DETECTORS), "exp": np.zeros(N_DETECTORS),
            "ks": [deque(maxlen=HEALTH_KS_CAP) for _ in range(N_DETECTORS)],
            "last_eval": None, "out": {}}


def _ensure_health(hs: Dict[str, Any]) -> Dict[str, Any]:
    """Live health state from a JSON-restored one (lists -> arrays / deques)."""
    if isinstance(hs.get("exc"), np.ndarray) and isinstance(hs.get("exp"), np.ndarray):
        return hs
    fresh = new_health()
    for k in ("exc", "exp"):
        a = np.asarray([np.nan if v is None else v for v in (hs.get(k) or [])], dtype=np.float64)
        if a.shape == (N_DETECTORS,):
            fresh[k] = np.where(np.isfinite(a), a, 0.0)
    for i, buf in enumerate((hs.get("ks") or [])[:N_DETECTORS]):
        fresh["ks"][i].extend(float(v) for v in buf if v is not None)
    t = hs.get("t")
    fresh["t"] = float(t) if isinstance(t, (int, float)) else _NAN
    le = hs.get("last_eval")
    fresh["last_eval"] = float(le) if isinstance(le, (int, float)) else None
    fresh["out"] = dict(hs.get("out") or {})
    hs.clear()
    hs.update(fresh)
    return hs


def health_add(hs: Dict[str, Any], ts: float, dt: float, pvals: np.ndarray,
               trusted: bool) -> None:
    """Fold one committed tick's p-values into the system health state."""
    p = np.asarray(pvals, dtype=np.float64).reshape(-1)
    fin = np.isfinite(p)
    if not fin.any():
        return
    idx = np.flatnonzero(fin)
    t0 = hs["t"]
    if t0 != t0 or ts > t0:
        if t0 == t0:
            f = 2.0 ** (-(ts - t0) / HEALTH_RATE_HL_S)
            hs["exc"] *= f
            hs["exp"] *= f
        hs["t"] = ts
        w = 1.0
    else:                                   # a released (older) row arrives decayed
        w = 2.0 ** (-(t0 - ts) / HEALTH_RATE_HL_S)
    thr = HEALTH_E_DAY * dt / 86400.0      # p <= thr  <=>  e_day <= 0.03
    pv = p[idx]
    hs["exc"][idx] += w * (pv <= thr)
    hs["exp"][idx] += w * thr
    if trusted:
        ks = hs["ks"]
        for i, v in zip(idx.tolist(), pv.tolist()):
            ks[i].append(v)


def evaluate_health(ks_bufs, exc: np.ndarray, expected: np.ndarray) -> Dict[str, Dict[str, float]]:
    """{detector: {ks, rate_ratio, weight_mult, n, expected}} for every detector
    with data. A check without enough data is NaN and neutral (calib.health_weight)."""
    out: Dict[str, Dict[str, float]] = {}
    for i, d in enumerate(DETECTORS):
        buf = ks_bufs[i]
        n = len(buf)
        ex = float(expected[i])
        if n == 0 and ex <= 0.0:
            continue
        ks_d = calib.ks_uniform(np.fromiter(buf, dtype=np.float64, count=n)) \
            if n >= HEALTH_KS_MIN else _NAN
        rr = float(exc[i]) / ex if ex >= HEALTH_RATE_MIN_EXPECTED else _NAN
        out[d] = {"ks": ks_d, "rate_ratio": rr, "weight_mult": calib.health_weight(ks_d, rr),
                  "n": n, "expected": ex}
    return out


# ----------------------------------------------------------------- learner
class _RingLearner(gating.GatedLearner):
    """GatedLearner whose rollback deletes ring entries after the onset
    instead of restoring a checkpoint (see the module docstring; checkpoints
    are disabled with ckpt_every_s = inf). Release, rebase, frozen, version
    and link seeding are the generic ones."""

    def _rollback(self, store, s, e, state, g, tau, now, dt_s):  # noqa: D401
        j = g.journal
        k = bisect_right(j, tau, key=lambda r: r.ts)
        moved = j[k:]
        removed = remove_after(state, tau)
        g.journal = j[:k]
        if moved and not g.frozen:
            g.held = sorted(g.held + moved, key=lambda r: r.ts)
        g.applied["_last_rollback"] = {"tau": tau, "wall": now, "moved": len(moved),
                                       "removed": removed, "mode": "delete_after",
                                       "complete": True}
        return state


class _PoolCache:
    """Per-tick, per-system cache of class keys and members for pooling."""

    def __init__(self, store, s: str) -> None:
        self.store, self.s = store, s
        self._ck: Dict[str, Optional[str]] = {}
        self._members: Dict[str, List[str]] = {}

    def class_key(self, e: str) -> Optional[str]:
        if e.startswith(CLASS_PREFIX) or e.startswith("__"):
            return None
        if e not in self._ck:
            self._ck[e] = m_class.class_key(self.store, self.s, e)
        return self._ck[e]

    def member_rings(self, ck: str, e: str, key: str):
        mem = self._members.get(ck)
        if mem is None:
            mem = self._members[ck] = m_class.class_members(self.store, self.s, ck)
        for m in mem:
            if m == e:
                continue
            model = self.store.get_model(self.s, m, MODEL)
            if isinstance(model, Mapping):
                r = (model.get(RINGS) or {}).get(key)
                if r is not None:
                    yield r


# ================================================================== engine
class CalibrationEngine(Engine):
    name = "behavior.calibration"
    layer = "behavior"
    consumes = ["behavior.score", "behavior.pm", "behavior.trust", "behavior.trust_prov",
                "behavior.quarantine", "feature.tctx", "behavior.regime", "model.control",
                "model.link", "model.class", "model.baseline"]
    produces = ["behavior.p", "model.calib", "behavior.calib_health",
                "profile.extra.calibration"]
    description = ("Randomised, Mondrian-stratified conformal p-values with a GPD tail per "
                   "(entity, detector, daypart x cadence); trust-gated, reversible rings; "
                   "calibration health weights.")
    interval = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._learners: Dict[float, _RingLearner] = {}
        self._sink: List[Tuple[float, float, np.ndarray, bool]] = []
        self._n_base = 0.0
        self._reset_flag = False
        self._config: Dict[str, Any] = {}
        self._ret_store: Optional[weakref.ref] = None
        self._keys: Dict[Tuple[str, str], List[str]] = {}

    # ------------------------------------------------------------- plumbing
    def _learner(self, d_min_s: Any) -> _RingLearner:
        d = float(d_min_s) if d_min_s is not None else gating.D_MIN_S
        lr = self._learners.get(d)
        if lr is None:
            lr = self._learners[d] = _RingLearner(
                name=LEARNER, init=new_model, update=self._update, fetch=self._fetch,
                dump=m_calib.to_json, load=self._load, merge=self._merge,
                on_rebase=self._on_rebase, d_min_s=d, ckpt_every_s=math.inf, clock=SCORE)
        return lr

    def _ensure_retention(self, store) -> None:
        """behavior.calib_health: at least 8 d (contract B; store default). Raise-only."""
        if self._ret_store is None or self._ret_store() is not store:
            store.ensure_retention(CALIB_HEALTH, None, HEALTH_RETENTION_S)
            self._ret_store = weakref.ref(store)

    def _ring_key(self, d: str, stratum: str) -> str:
        return self._keys_for(stratum, stratum)[DETECTOR_INDEX[d]]

    def _keys_for(self, st: str, st_id: str) -> List[str]:
        """Ring key of every detector for (stratum, identity stratum), cached."""
        k = (st, st_id)
        keys = self._keys.get(k)
        if keys is None:
            keys = self._keys[k] = [calib.ring_key(d, st_id if i == _ID_IDX else st)
                                    for i, d in enumerate(DETECTORS)]
        return keys

    def _daypart(self, store, s: str, e: str, ts: float, dt: float, search: int = 1) -> Tuple[str, Optional[float]]:
        """(daypart, B01 cadence class or None) at ts: feature.tctx, else config."""
        t = _tctx_at(store, s, e, ts, search)
        if t is not None and t.get("daypart") in _DP_INDEX:
            cc = t.get("cc")
            return t["daypart"], (float(cc) if cc is not None else None)
        return timebins.tctx_from_config(ts, self._config, dt)["daypart"], None

    # ------------------------------------------------------ learner callbacks
    def _fetch(self, store, s: str, e: str, ts: float) -> Optional[_Row]:
        row = store.vec_at(s, e, SCORE, ts)
        if row is None:
            return None                     # pruned (score retention) or never scored
        model = store.get_model(s, e, MODEL)
        code = model.get("pending", {}).get(ts) if isinstance(model, Mapping) else None
        if code is not None:
            dp, terc, dt = _decode(code)
        else:                               # restart / lost bookkeeping: rebuild from tctx
            dt0 = model.get("dt") if isinstance(model, Mapping) else None
            dt0 = float(dt0) if isinstance(dt0, (int, float)) and dt0 > 0 else MATURITY_UNIT_S
            dp, cc = self._daypart(store, s, e, ts, dt0, search=64)
            dt, terc = (cc if cc else dt0), 0
        return _Row(s, e, ts, row, store.vec_at(s, e, P, ts), dp, terc, dt)

    def _update(self, model: Dict[str, Any], row: _Row, w: float) -> Dict[str, Any]:
        w = float(w)
        w = 0.0 if not w > 0.0 else (1.0 if w > 1.0 else w)     # NaN -> 0
        mature = max(self._n_base, float(model["n_own"])) >= MATURITY_N_EFF
        model["n_own"] = float(model["n_own"]) + w * row.dt / MATURITY_UNIT_S
        if w >= 1.0:
            admit = True
        elif w > 0.0:
            admit = combine.seeded_uniform(row.s, row.e, ADMIT_SALT, float(row.ts)) < w
        else:
            admit = False
        if admit and mature:
            cc = timebins.cadence_class(row.dt)
            keys = self._keys_for(calib.stratum_key(row.daypart, cc),
                                  calib.identity_stratum_key(row.daypart, row.tercile))
            rings, refit = model[RINGS], model["refit"]
            ts = row.ts
            for i, x in enumerate(row.scores.tolist()):
                if not math.isfinite(x):
                    continue
                key = keys[i]
                r = rings.get(key)
                if r is None:
                    r = rings[key] = calib.Ring()
                r.add(x, ts)
                c = refit.get(key)
                refit[key] = (_phase(row.e, key) if c is None else c) + 1   # fit is lazy (_score)
            model["n_admit"] = int(model["n_admit"]) + 1
        if row.pvals is not None:
            self._sink.append((row.ts, row.dt, row.pvals, admit))
        return model

    def _on_rebase(self, model: Dict[str, Any], tau: float) -> Dict[str, Any]:
        _reset_rings(model)
        self._reset_flag = True
        return model

    def _merge(self, own: Dict[str, Any], other: Dict[str, Any], w: float) -> Dict[str, Any]:
        """Link seeding: own ring += the other's most recent M*w entries per key."""
        for key, ro in m_calib.rings(other).items():
            nr = calib.seed_ring(own[RINGS].get(key) or calib.Ring(), ro, frac=w)
            own[RINGS][key] = nr
            if nr.gpd is None:
                own["refit"][key] = GPD_REFIT_TICKS       # refit before the tail is used
        return own

    def _load(self, blob: Any) -> Dict[str, Any]:
        return _ensure_layout(dict(blob) if isinstance(blob, Mapping) else None)

    # ------------------------------------------------------------------ run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        now = float(ctx.now)
        dt = float(ctx.window_s)
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError(f"calibration: ctx.window_s={ctx.window_s!r} is not a positive cadence")
        cc = timebins.cadence_class(dt)
        self._config = ctx.config or {}
        self._ensure_retention(store)
        learner = self._learner(self._config.get("D_min_s", gating.D_MIN_S))
        n_out = 0
        for s in store.systems():
            keys = store.entities(s) + [k for k in store.pseudo_entities(s)
                                        if k.startswith(CLASS_PREFIX)]
            sysm = store.get_model(s, SYSTEM_KEY, MODEL)
            hs = sysm.get(m_calib.HEALTH) if isinstance(sysm, dict) else None
            if hs is not None:
                hs = _ensure_health(hs)
            health_out = hs["out"] if hs is not None else {}
            pool = _PoolCache(store, s)
            for e in keys:
                self._sink = []
                n_out += self._run_key(ctx, store, s, e, now, dt, cc, learner, pool, health_out)
                if self._sink:
                    if hs is None:
                        sysm = sysm if isinstance(sysm, dict) else {"layout": m_calib.LAYOUT}
                        hs = sysm[m_calib.HEALTH] = new_health()
                        store.put_model(s, SYSTEM_KEY, MODEL, sysm, ts=now)
                    for ts, dt_r, pv, trusted in self._sink:
                        health_add(hs, ts, dt_r, pv, trusted)
            self._sink = []
            if hs is not None:
                self._write_health(store, s, hs, now, dt)
        return n_out

    def _write_health(self, store, s: str, hs: Dict[str, Any], now: float, dt: float) -> None:
        le = hs["last_eval"]
        if le is None or now - le >= HEALTH_EVAL_S or now < le:
            hs["out"] = evaluate_health(hs["ks"], hs["exc"], hs["exp"])
            hs["last_eval"] = now
        last = store.latest_derived(s, SYSTEM_KEY, CALIB_HEALTH)
        if last is not None and last.ts == now:
            last.value = hs["out"]
        else:
            store.add_derived(DerivedMetric(name=CALIB_HEALTH, value=hs["out"], ts=now,
                                            system=s, entity=SYSTEM_KEY, window_s=int(dt),
                                            kind=MetricKind.CATEGORICAL))

    def _run_key(self, ctx: Context, store, s: str, e: str, now: float, dt: float, cc: int,
                 learner: _RingLearner, pool: _PoolCache, health_out: Mapping) -> int:
        model = store.get_model(s, e, MODEL)
        row = store.vec_at(s, e, SCORE, now)
        if model is None and row is None:
            return 0                        # never scored: nothing to learn or calibrate
        model = _ensure_layout(model)
        gate = model["gate"]
        # the gate's own version (as B25 does): applied['version'] is not
        # recorded while model.control's version equals the default 0, so a
        # later bare 0 -> 2 change would read as seen=None and skip the reset
        seen = gate.version
        self._n_base = baseline_n_eff(store, s, e)
        self._reset_flag = False
        model, gate = learner.seed_from_link(
            store, s, e, model, gate, lambda src: self._other(store, s, src))
        model, gate = learner.step(store, s, e, model, gate, now, dt, training=ctx.training)
        if gate.version != seen and not self._reset_flag and model[RINGS]:
            _reset_rings(model)             # version change without rebase_from
                                            # (nothing to reset on a first-seen version)
        # rows older than the score retention can never be fetched again, so a
        # longer journal (gating keeps 8 d) would only be copied every tick
        j = gate.journal
        if j and j[0].ts < now - SCORE_RETENTION_S:
            gate.journal = j[bisect_left(j, now - SCORE_RETENTION_S, key=lambda r: r.ts):]
        model["gate"] = gate
        model["version"] = int(gate.version)
        n_out = 0
        if row is not None:
            n_out = self._score(store, s, e, now, dt, cc, model, row, pool, health_out)
        pend = model["pending"]
        cutoff = now - SCORE_RETENTION_S
        while pend:
            t0 = next(iter(pend))
            if t0 >= cutoff:
                break
            del pend[t0]
        model["dt"] = dt
        store.put_model(s, e, MODEL, model, version=model["version"], ts=now)
        return n_out

    def _other(self, store, s: str, src: str) -> Optional[Dict[str, Any]]:
        m = store.get_model(s, src, MODEL)
        return m if isinstance(m, Mapping) and m.get(RINGS) else None

    # -------------------------------------------------------------- scoring
    def _score(self, store, s: str, e: str, now: float, dt: float, cc: int,
               model: Dict[str, Any], row: np.ndarray, pool: _PoolCache,
               health_out: Mapping) -> int:
        dp, _ = self._daypart(store, s, e, now, dt)
        terc = m_calib.regime_tercile(_latest_value(store, s, e, REGIME))
        st = calib.stratum_key(dp, cc)
        st_id = calib.identity_stratum_key(dp, terc)
        keys = self._keys_for(st, st_id)
        rings, refit = model[RINGS], model["refit"]
        pm = store.vec_at(s, e, PM, now)
        out: Dict[str, float] = {}
        sizes: Dict[str, int] = {}
        for i, x in enumerate(row.tolist()):
            if x != x:
                continue                    # unscored / degraded: p stays NaN
            d = DETECTORS[i]
            key = keys[i]
            r = rings.get(key)
            n = 0 if r is None else r.scores.size
            u = m_calib.uniform(s, e, d, now)
            if n >= SMALL_N:
                c = refit.get(key)
                if c is None:               # restored / seeded ring without a counter
                    c = refit[key] = _phase(e, key)
                if _tail_due(r, c, x):
                    _refit(r, now)
                    refit[key] = 0
                p = calib.p_from_ring(r, x, u)          # = m_calib.p_value without a prior
            else:
                prior = self._prior(i, d, x, u, pm, pool, e, key, rings, dp, terc, cc)
                p = m_calib.p_value(r, x, u, prior)
            out[d] = p if p >= P_ISSUED_FLOOR else P_ISSUED_FLOOR   # float32 ring
            sizes[d] = n
        if out:
            emit.write_pvalues(store, s, e, now, out, window_s=int(dt))
        model["pending"][now] = _encode(dp, terc, dt)
        pts = float(model["profile_ts"])
        if now - pts >= PROFILE_EVERY_S or now < pts:
            self._profile(store, s, e, now, model, st, st_id, sizes, health_out)
        return len(out)

    def _prior(self, i: int, d: str, x: float, u: float, pm: Optional[np.ndarray],
               pool: _PoolCache, e: str, key: str, rings: Mapping, dp: str, terc: int,
               cc: int = 0) -> float:
        """Small-sample prior: the entity's own rings of the other dayparts at
        this cadence (pooled, >= 64 entries), else pm[d], else the
        class-pooled ring, else (identity) the settled-regime ring; NaN when
        none is usable (conformal p alone).

        The own-daypart pool comes first (integration): the first workday
        after a weekend warm-up, or the first night, is a new stratum for
        every detector, and several pm are only approximately calibrated
        (timing's overdispersed G test, novelty's bits, the accumulators'
        stationary tails); the entity's own null of the same score at the same
        cadence is a better prior than any of them. A cadence switch still
        falls through to pm (no ring of the new cadence exists)."""
        if i != _ID_IDX and cc:
            own = [r for r in (rings.get(self._ring_key(d, calib.stratum_key(p, cc)))
                               for p in timebins.DAYPARTS if p != dp) if r is not None]
            if own:
                p, _ = m_calib.pooled_p(own, x, u)
                if p == p:
                    return p
        if pm is not None:
            v = float(pm[i])
            if 0.0 <= v <= 1.0:
                return v
        ck = pool.class_key(e)
        if ck is not None:
            p, _ = m_calib.pooled_p(pool.member_rings(ck, e, key), x, u)
            if p == p:
                return p
        if i == _ID_IDX and terc != 0:
            r0 = rings.get(self._ring_key(d, calib.identity_stratum_key(dp, 0)))
            if r0 is not None and len(r0) >= SMALL_N:
                return m_calib.p_value(r0, x, u)
        return _NAN

    def _profile(self, store, s: str, e: str, now: float, model: Dict[str, Any], st: str,
                 st_id: str, sizes: Mapping[str, int], health_out: Mapping) -> None:
        maturity = max(self._n_base, float(model["n_own"]))
        wm = {d: m_calib.weight_mult(health_out, d) for d in sizes}
        n_tails = sum(r.gpd is not None for r in model[RINGS].values())
        info = {
            "version": int(model["version"]),
            "stratum": st,
            "identity_stratum": st_id,
            "n_rings": len(model[RINGS]),
            "n_tails": n_tails,
            "n_admit": int(model["n_admit"]),
            "maturity": maturity,
            "mature": maturity >= MATURITY_N_EFF,
            "n_current": dict(sizes),
            "small_sample": sorted(d for d, n in sizes.items() if n < SMALL_N),
            "weight_mult": {d: w for d, w in wm.items() if w < 1.0},
            "healthy": all(w >= 1.0 for w in wm.values()),
            "resets": int(model["resets"]),
            "updated": now,
        }
        prof = store.profile(s, e)
        if prof is None:
            prof = EntityProfile(system=s, entity=e, updated=now)
        prof.extra["calibration"] = info
        store.put_profile(prof)
        model["profile_ts"] = now
