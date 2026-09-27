"""ChangepointEngine (B14) — persistent, intermittent and gradual shifts.

Replaces the cosine DriftEngine. Why a changepoint engine at all: the
instantaneous detectors (B04/B06) ask "is this tick unusual?", so an attacker
who moves slowly or intermittently never produces an unusual tick, and a
baseline that keeps learning would absorb him. B14 therefore
  * measures against the REFERENCE anchor (behavior.zr): the current anchor
    may creep 0.1 sigma15/day and could cancel a slow shift, the reference
    (or golden) anchor barely moves, so a poisoned current anchor cannot
    hide the shift;
  * accumulates evidence with four accumulators, each with its own
    wall-clock false-alarm budget (thresholds from ARLs in days, so the rate
    per entity-day does not change when the cadence switches 900 -> 60 s):
      cusum   48 AR(1)-prewhitened one-sided CUSUMs (12 key features x 2
              sides x k in {0.25, 1.0}), ARL 2400 d each (0.02/entity-day);
      mcusum  Crosier MCUSUM on the whitened 12-vector, ARL 100 d;
      bocpd   hourly Bayesian online changepoint detection on the intensity
              residual and on B06's Wilson-Hilferty score (Normal-Gamma,
              hazard 1/168 h); cp.prob = P(run length <= 3 h);
      creep   daily Mann-Kendall / Sen slope over 14 days of
              (entity - class median) group means, against the golden anchor;
  * estimates the onset tau-hat (the last tick the alarmed statistic was at
    its zero level) so the governor can roll learners back to before it, and
    reports the cumulative excess in natural units (what was added, not just
    how many sigmas).

The CUSUM / MCUSUM maths lives in lib/m_cp.py (pure, batch-vectorised) so
B29 replays exactly what ran here; the state vector is stored every tick in
behavior.cusum_state. What B14 LEARNS (AR(1) phi, the residual correlation
used when model.density has none, the vec-per-zr scale that converts sigmas
to log units, the committed residual buffer the audit bootstraps) goes
through lib/gating: late (D), trust-weighted, checkpointed and reversible
under model.control. Detection statistics are not learned state; they reset
when the governor releases (RETURNED) or rebases (ACCEPTED) an episode, and
by the 2 (t - tau-hat) clean-time rule.
"""
from __future__ import annotations

import copy
import datetime as _dt
import importlib
import math
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
from scipy.special import gammaln

from ...core.engine import Context, Engine
from ...models.schema import BehaviorEvent, EntityProfile, Severity
from .lib import combine, emit, m_class, m_cp, robustcov, seq, timebins
from .lib.features import (FEATURE_DIM, FEATURE_GROUP, FEATURE_INDEX, FEATURE_KIND,
                           FEATURE_NAMES_V2, GROUP_ORDER, KEY_FEATURES, VEC_TX)
from .lib.gating import GatedLearner, GateState

ZR = m_cp.ZR
WH = "behavior.wh"
VEC = "feature.vec"
NAT = "feature.nat"
ACTIVE = "feature.active"
DENSITY = "model.density"
BASELINE = "model.baseline"
DETS = ("cusum", "mcusum", "bocpd", "creep")

# ---- learner --------------------------------------------------------------
BUF_CAP = 480                  # committed key-residual rows kept (audit, phi, correlation)
BUF_MIN_W = 0.5                # rows committed with lower trust do not enter the buffer
REFIT_EVERY = 32               # commits between phi / correlation refits
PHI_MIN_PAIRS = 48             # adjacent same-cadence pairs needed to refit phi
SCALE_HL_S = 7 * 86400.0       # half-life of the vec-per-zr scale statistics
SCALE_MIN_N = 8.0
CORR_MIN_ROWS = 24

# ---- chart inputs: the entity's own support at the bucket ------------------
SUPPORT_MIN_EXPO_MIN = 60.0    # weighted exposure minutes of own committed rows
SUPPORT_MIN_ROWS = 2.0         # decayed rows of the feature itself

# ---- BOCPD ----------------------------------------------------------------
HAZARD = 1.0 / 168.0           # per hour
R_MAX = 336                    # hours; longer runs are merged at R_MAX
PRUNE = 1e-4
BOC_A0, BOC_K0 = 1.0, 1.0      # Normal-Gamma prior on standardised hourly sums
BOC_WINDOW_H = 3               # cp.prob = P(r <= 3 h)
# Alarm level for cp.prob: measured null rate 0.001-0.004 per entity-day for
# hourly N(0,1) inputs with per-tick AR(1) phi in [0, 0.6] (8 x 200 d each),
# within the 0.005/day change-path share of one detector.
BOC_ALARM = 0.8
BOC_MAX_GAP_H = 48             # silent hours stepped with hazard only (beyond: no-op)
_LOGPI = math.log(math.pi)

# ---- creep ----------------------------------------------------------------
CREEP_DAYS = 14
CREEP_MIN_DAYS = 10
CREEP_MIN_MEMBERS = 3          # a class median needs >= 3 members (entity included)
CREEP_P = 0.01
CREEP_SLOPE = 0.05             # log-units per day
DAY_MIN_TICKS = 4
DAILY_KEEP = 28
N_GROUPS = len(GROUP_ORDER)
_GMAT = np.array([[1.0 if FEATURE_GROUP[n] == g else 0.0 for n in FEATURE_NAMES_V2]
                  for g in GROUP_ORDER])

# ---- natural-unit excess ---------------------------------------------------
_KEY_KIND = [("identity" if VEC_TX.get(n) == "identity" else FEATURE_KIND[n])
             for n in KEY_FEATURES]
_EXTENSIVE = np.array([k in ("count", "bytes") for k in _KEY_KIND])
_K025 = np.flatnonzero(m_cp.CHART_K == m_cp.KS[0])       # the k = 0.25 charts (both sides)

try:        # golden anchor offsets, when B03's accessor module exists
    _m_baseline = importlib.import_module(f"{__package__}.lib.m_baseline")
except ModuleNotFoundError:
    _m_baseline = None
try:        # B06's density accessor, when it exists
    _m_density = importlib.import_module(f"{__package__}.lib.m_density")
except ModuleNotFoundError:
    _m_density = None


# ============================================================== learner
def _learn_init() -> Dict[str, Any]:
    return {
        "buf_ts": np.empty(0), "buf_dt": np.empty(0), "buf_x": np.empty((0, m_cp.N_KEY)),
        "phi": np.zeros(m_cp.N_KEY), "phi_dt": math.nan,
        "corr": None, "corr_rev": 0,
        "sc_v": np.zeros(FEATURE_DIM), "sc_z": np.zeros(FEATURE_DIM),
        "sc_n": np.zeros(FEATURE_DIM), "sc_t": math.nan,
        "n": 0,
    }


def _gaps(ts: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """gap[i]: committed row i does not directly follow row i-1 on the clock."""
    g = np.ones(ts.size, dtype=bool)
    if ts.size > 1:
        g[1:] = ~(np.abs(np.diff(ts) - dt[1:]) <= 1e-6)
    return g


def _refit(st: Dict[str, Any]) -> None:
    """phi (seq.ar1_phi on adjacent same-cadence pairs) and the key-residual
    correlation (robustcov.oas on the prewhitened buffer). Deterministic in
    the buffer, so restore + replay reproduces it."""
    ts, dts, X = st["buf_ts"], st["buf_dt"], st["buf_x"]
    if ts.size < 2 or not math.isfinite(dts[-1]):
        return
    cc = timebins.cadence_class(float(dts[-1]))
    same = np.array([math.isfinite(d) and timebins.cadence_class(float(d)) == cc for d in dts])
    gap = _gaps(ts, dts) | ~same | ~np.roll(same, 1)
    n_pairs = int((~gap[1:] & same[1:]).sum()) if ts.size > 1 else 0
    if n_pairs >= PHI_MIN_PAIRS:
        Xs = np.where(same[:, None], X, np.nan)
        seqx = np.insert(Xs, np.flatnonzero(gap[1:]) + 1, np.nan, axis=0)
        st["phi"] = np.array([seq.ar1_phi(seqx[:, f]) for f in range(m_cp.N_KEY)])
        st["phi_dt"] = float(dts[-1])
    prev = np.vstack([np.full((1, m_cp.N_KEY), np.nan), X[:-1]])
    prev[gap] = np.nan
    psi = m_cp.prewhiten(X, prev, st["phi"])
    cols = np.flatnonzero(np.isfinite(psi).mean(axis=0) >= 0.5)
    if cols.size >= 2:
        P = psi[:, cols]
        P = P[np.isfinite(P).all(axis=1)]
        if P.shape[0] >= CORR_MIN_ROWS:
            _, Sig, _ = robustcov.oas(P)
            C = np.eye(m_cp.N_KEY)
            C[np.ix_(cols, cols)] = Sig
            st["corr"] = C
            st["corr_rev"] = int(st["corr_rev"]) + 1


def _learn_update(st: Dict[str, Any], row: Tuple, w: float) -> Dict[str, Any]:
    """Commit one row (in place; checkpoints are deep copies). Rows may arrive
    older than the newest one (release / rebase): the buffer is kept in ts
    order and the scale statistics decay the ROW weight, never the state
    backwards, so any commit order gives the same sums."""
    w = float(w)
    if not w > 0.0:
        return st
    ts, dt_row, zr, zr_prev, vec, vec_prev = row
    st["n"] = int(st["n"]) + 1
    if w >= BUF_MIN_W:
        x = np.asarray(zr, dtype=np.float64)[m_cp.KEY_IDX]
        i = int(np.searchsorted(st["buf_ts"], ts, side="right"))
        st["buf_ts"] = np.insert(st["buf_ts"], i, ts)[-BUF_CAP:]
        st["buf_dt"] = np.insert(st["buf_dt"], i, dt_row)[-BUF_CAP:]
        st["buf_x"] = np.insert(st["buf_x"], i, x, axis=0)[-BUF_CAP:]
    if zr_prev is not None and vec is not None and vec_prev is not None:
        dz = np.asarray(zr, dtype=np.float64) - np.asarray(zr_prev, dtype=np.float64)
        dv = np.asarray(vec, dtype=np.float64) - np.asarray(vec_prev, dtype=np.float64)
        ok = np.isfinite(dz) & np.isfinite(dv)
        t_ref = st["sc_t"]
        if not math.isfinite(t_ref):
            t_ref = st["sc_t"] = float(ts)
        if ts >= t_ref:
            f = 2.0 ** (-(ts - t_ref) / SCALE_HL_S)
            st["sc_v"], st["sc_z"], st["sc_n"] = st["sc_v"] * f, st["sc_z"] * f, st["sc_n"] * f
            st["sc_t"] = float(ts)
            wr = w
        else:
            wr = w * 2.0 ** (-(t_ref - ts) / SCALE_HL_S)
        st["sc_v"] = st["sc_v"] + wr * np.where(ok, dv * dv, 0.0)
        st["sc_z"] = st["sc_z"] + wr * np.where(ok, dz * dz, 0.0)
        st["sc_n"] = st["sc_n"] + wr * ok
    if st["n"] % REFIT_EVERY == 0:
        _refit(st)
    return st


def _learn_merge(own: Dict[str, Any], other: Dict[str, Any], w: float) -> Dict[str, Any]:
    """Link seeding B := B_own + w A: scale statistics add; a learner without
    its own phi / correlation fit adopts the linked entity's."""
    own["sc_v"] = own["sc_v"] + w * np.asarray(other["sc_v"])
    own["sc_z"] = own["sc_z"] + w * np.asarray(other["sc_z"])
    own["sc_n"] = own["sc_n"] + w * np.asarray(other["sc_n"])
    if not math.isfinite(own["phi_dt"]) and math.isfinite(other.get("phi_dt", math.nan)):
        own["phi"] = np.array(other["phi"], dtype=np.float64)
        own["phi_dt"] = float(other["phi_dt"])
    if own.get("corr") is None and other.get("corr") is not None:
        own["corr"] = np.array(other["corr"], dtype=np.float64)
        own["corr_rev"] = int(own["corr_rev"]) + 1
    return own


def _fetch(store: Any, s: str, e: str, ts: float) -> Optional[Tuple]:
    """Committed row at ts: zr, the previous clock tick's zr (for the AR(1)
    pair) and feature.vec at both ticks (for the vec-per-zr scale)."""
    zr = store.vec_at(s, e, ZR, ts)
    if zr is None:
        return None
    tc, _ = store.vec_range(s, e, ACTIVE, ts - 5400.0, ts)
    k = int(np.searchsorted(tc, ts - 1e-6, side="left"))
    prev_ts = float(tc[k - 1]) if k > 0 else math.nan
    if not math.isfinite(prev_ts):
        return (float(ts), math.nan, zr, None, store.vec_at(s, e, VEC, ts), None)
    return (float(ts), float(ts - prev_ts), zr, store.vec_at(s, e, ZR, prev_ts),
            store.vec_at(s, e, VEC, ts), store.vec_at(s, e, VEC, prev_ts))


def _copy_state(st: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(st)


def phi_at(learn: Mapping[str, Any], dt: float) -> np.ndarray:
    """phi at cadence dt: an AR(1) sampled at dt0 has phi(dt) = phi(dt0)^(dt/dt0)
    (continuous-time OU), so a cadence switch keeps whitening until pairs at
    the new cadence refit it. float32-rounded (it is stored for replay)."""
    phi = np.asarray(learn["phi"], dtype=np.float64)
    dt0 = float(learn.get("phi_dt", math.nan))
    if not math.isfinite(dt0) or dt0 <= 0.0 or dt == dt0:
        out = phi
    else:
        out = np.where(phi > 0.0, np.power(np.clip(phi, 1e-12, 1.0), dt / dt0), 0.0)
    return m_cp._f32(np.clip(out, seq.AR1_PHI_CLIP[0], seq.AR1_PHI_CLIP[1]))


def scale_of(learn: Mapping[str, Any]) -> np.ndarray:
    """vec units per zr unit per feature, sqrt(sum dvec^2 / sum dzr^2) over
    adjacent committed pairs (first differences cancel the slow seasonal
    level); NaN until SCALE_MIN_N pairs."""
    n = np.asarray(learn["sc_n"])
    z = np.asarray(learn["sc_z"])
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.sqrt(np.asarray(learn["sc_v"]) / z)
    return np.where((n >= SCALE_MIN_N) & (z > 0.0), s, np.nan)


# ============================================================== BOCPD
def bocpd_new(d: int = 2) -> Dict[str, np.ndarray]:
    return {"r": np.zeros(1), "p": np.ones(1), "mu": np.zeros((d, 1)),
            "ka": np.full((d, 1), BOC_K0), "al": np.full((d, 1), BOC_A0),
            "be": np.full((d, 1), 1.0)}


def bocpd_step(st: Mapping[str, np.ndarray], x: np.ndarray) -> Dict[str, np.ndarray]:
    """One hourly step of Adams-MacKay BOCPD with independent Normal-Gamma
    dims sharing the run length (NaN dims: likelihood 1, no update). A new
    run's prior variance is the current MAP run's (empirical Bayes), so an
    autocorrelated but stable null does not look like a change of scale.
    Runs longer than R_MAX merge at R_MAX (dropping them would move all mass
    to short runs: a spurious change every two weeks)."""
    x = np.asarray(x, dtype=np.float64)
    obs = np.isfinite(x)
    r0, p0 = st["r"], st["p"]
    mu, ka, al, be = st["mu"], st["ka"], st["al"], st["be"]
    j = int(np.argmax(p0))
    b0 = np.clip(be[:, j] / al[:, j], 0.25, 16.0) * BOC_A0
    if obs.any():
        o = np.flatnonzero(obs)
        xo = x[o][:, None]
        m, k_, a, b = mu[o], ka[o], al[o], be[o]
        sc2 = b * (k_ + 1.0) / (a * k_)
        d2 = (xo - m) ** 2
        lt = (gammaln(a + 0.5) - gammaln(a) - 0.5 * np.log(2.0 * a * sc2) - 0.5 * _LOGPI
              - (a + 0.5) * np.log1p(d2 / (2.0 * a * sc2)))
        ll = lt.sum(axis=0)
        wgt = p0 * np.exp(ll - ll.max())
        mu, ka, al, be = mu.copy(), ka.copy(), al.copy(), be.copy()
        mu[o] = (k_ * m + xo) / (k_ + 1.0)
        ka[o] = k_ + 1.0
        al[o] = a + 0.5
        be[o] = b + k_ * d2 / (2.0 * (k_ + 1.0))
    else:
        wgt = p0
    p = np.concatenate(([wgt.sum() * HAZARD], wgt * (1.0 - HAZARD)))
    r = np.concatenate(([0.0], np.minimum(r0 + 1.0, R_MAX)))
    nd = x.size
    mu = np.concatenate((np.zeros((nd, 1)), mu), axis=1)
    ka = np.concatenate((np.full((nd, 1), BOC_K0), ka), axis=1)
    al = np.concatenate((np.full((nd, 1), BOC_A0), al), axis=1)
    be = np.concatenate((b0[:, None], be), axis=1)
    if r.size > 2 and r[-2] == R_MAX:                        # two capped runs: merge
        keep_last = p[-1] >= p[-2]
        p[-1 if keep_last else -2] += p[-2 if keep_last else -1]
        p[-2 if keep_last else -1] = 0.0
    p = p / p.sum()
    keep = p >= PRUNE
    keep[0] = True
    if not keep.all():
        p = p[keep] / p[keep].sum()
        r, mu, ka, al, be = r[keep], mu[:, keep], ka[:, keep], al[:, keep], be[:, keep]
    return {"r": r, "p": p, "mu": mu, "ka": ka, "al": al, "be": be}


def bocpd_prob(st: Mapping[str, np.ndarray], window_h: int = BOC_WINDOW_H) -> float:
    """P(run length <= window_h hours)."""
    return float(np.sum(st["p"][st["r"] <= window_h]))


# ============================================================== engine
class ChangepointEngine(Engine):
    name = "behavior.changepoint"
    layer = "behavior"
    consumes = ["behavior.zr", "behavior.wh", "feature.active", "feature.vec", "feature.nat",
                "model.density", "model.baseline", "model.class", "behavior.trust",
                "behavior.quarantine", "model.control"]
    produces = ["model.cp", "behavior.score", "behavior.pm", "behavior.acc_alarm",
                "behavior.axes", "behavior.degraded", "behavior.cp.prob", "behavior.cp.onset",
                "behavior.cusum_state", "profile.extra.regime.delta_by_feature",
                "event.baseline_creep"]
    description = ("Reference-anchored CUSUM bank, MCUSUM, hourly BOCPD and daily creep test "
                   "with wall-clock thresholds, onset estimation and replayable state.")
    interval = 1
    period_s = None

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.learner: GatedLearner = GatedLearner(
            name="cp", init=_learn_init, update=_learn_update, fetch=_fetch,
            dump=_copy_state, load=_copy_state, merge=_learn_merge)
        self._audit_next = -math.inf
        self._wcache: Dict[Tuple[str, str], Tuple[Any, np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------ run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        now, dt = float(ctx.now), float(ctx.window_s)
        self.learner.d_min_s = float(ctx.config.get("D_min_s", 600) or 600)
        tz = ctx.config.get("tz") or timebins.DEFAULT_TZ
        day_lo = timebins.local_datetime(now - 1e-6, tz).date().toordinal()
        day_hi = timebins.local_datetime(now + 1e-6, tz).date().toordinal()
        n = 0
        audit_pool: List[Tuple[float, str, str]] = []
        for s in store.systems():
            closed: List[Tuple[str, Dict[str, Any], int]] = []
            for e in store.entities(s):
                res = self._entity(ctx, s, e, day_lo, day_hi)
                if res is None:
                    continue
                model, scored, closed_day = res
                n += int(scored)
                if closed_day is not None:
                    closed.append((e, model, closed_day))
                if model["learn"]["buf_ts"].size >= 4 * m_cp.AUDIT_BLOCK:
                    audit_pool.append((float(model["run"].get("audit_ts", -math.inf)), s, e))
            med_cache: Dict[Tuple[str, int], Optional[np.ndarray]] = {}
            for e, model, day in closed:           # every member has closed its day now
                self._creep(ctx, s, e, model, day, med_cache)
        if audit_pool and now >= self._audit_next:
            self._audit(ctx, min(audit_pool))
            self._audit_next = now + 3600.0
        return n

    # ------------------------------------------------------------ per entity
    def _new_model(self) -> Dict[str, Any]:
        return {"version": 0, "ts": math.nan, "gate": GateState().to_dict(),
                "learn": _learn_init(), "run": self._new_run()}

    @staticmethod
    def _new_run() -> Dict[str, Any]:
        return {
            "last_ts": -math.inf, "dt": math.nan,
            "bank": m_cp.new_bank(), "mc": m_cp.new_mc(),
            "hmult": 1.0, "audit_ts": -math.inf, "audit": {},
            "hour": {"idx": None, "s_int": 0.0, "n_int": 0, "s_wh": 0.0, "n_wh": 0},
            "bocpd": bocpd_new(), "cp_prob": math.nan, "boc": {"on": False, "onset": math.nan},
            "day": {"idx": None, "sum": np.zeros(N_GROUPS), "n": np.zeros(N_GROUPS)},
            "daily": {"idx": [], "val": []},
            "creep": {"groups": {}, "p": math.nan, "pm": math.nan, "on": False, "axes": []},
            "excess": {k: np.zeros((2, m_cp.N_KEY)) for k in ("n", "z", "obs", "ref", "nn")},
            "whiten": {}, "alarm": {d: 0 for d in DETS}, "episode": {}, "delta_written": False,
        }

    def _entity(self, ctx: Context, s: str, e: str, day_lo: int, day_hi: int
                ) -> Optional[Tuple[Dict[str, Any], bool, Optional[int]]]:
        store = ctx.store
        now, dt = float(ctx.now), float(ctx.window_s)
        zr_row = store.vec_at(s, e, ZR, now)
        model = store.get_model(s, e, m_cp.MODEL)
        if model is None:
            if zr_row is None:
                return None                        # never scored by B04: nothing to do yet
            model = self._new_model()
        # ---- learning: gated, delayed, reversible (contract H)
        gate0 = GateState.from_dict(model["gate"])
        learn, gate = self.learner.step(store, s, e, model["learn"], gate0, now, dt,
                                        training=ctx.training)
        learn, gate = self.learner.seed_from_link(
            store, s, e, learn, gate, lambda a: self._other_learn(store, s, a))
        run = model["run"]
        self._apply_control(run, gate0, gate)
        if run.get("training") and not ctx.training:
            # End of warm-up: the charts ran against a model that was being
            # learnt from those very rows (cold backoff, hyperprior buckets);
            # their accumulated level is not evidence about live behaviour, so
            # the live episode starts from S = 0 like after a release.
            self._reset_charts(run)
        run["training"] = bool(ctx.training)
        model["learn"], model["gate"], model["version"] = learn, gate.to_dict(), int(gate.version)
        model["ts"] = now
        # ---- hour / day bookkeeping happens with or without data
        closed_day = self._roll_day(run, day_lo)
        if zr_row is None:
            self._roll_hour(run, now, math.nan, math.nan)
            self._idle_outputs(ctx, s, e, run)
            if day_hi != day_lo and closed_day is None:
                closed_day = self._roll_day(run, day_hi)
            store.put_model(s, e, m_cp.MODEL, model, ts=now)
            return model, False, closed_day
        # ---- detection
        x52 = np.asarray(zr_row, dtype=np.float64)
        xk = self._supported(store, s, e, ctx, x52[m_cp.KEY_IDX])
        adjacent = now - float(run["last_ts"]) <= 1.5 * dt + 1e-6
        phi = phi_at(learn, dt)
        h = m_cp.bank_h(dt) * float(run["hmult"])
        bank, bout = m_cp.bank_tick(run["bank"], xk, now, dt, phi, h, adjacent)
        psi = bout["psi"]
        Sig, W = self._whitener(store, s, e, learn, run)
        w = m_cp.mc_whiten(psi, W, Sig)
        h_mc = m_cp.mcusum_h(dt)
        mc = run["mc"]
        if np.isfinite(w).any():
            mc, _ = m_cp.mc_tick(mc, w, now, dt, h_mc)
        run["last_ts"], run["dt"] = now, dt
        # hourly BOCPD inputs: intensity residual and B06's WH score
        wh = store.vec_at(s, e, WH, now)
        x_wh = float(wh[0]) if wh is not None and len(wh) else math.nan
        self._roll_hour(run, now, float(x52[FEATURE_INDEX["intensity"]]), x_wh)
        # daily group means for the creep test
        fin = np.isfinite(x52)
        cnt = _GMAT @ fin
        gm = np.where(cnt > 0, (_GMAT @ np.where(fin, x52, 0.0)) / np.maximum(cnt, 1.0), np.nan)
        day = run["day"]
        ok = np.isfinite(gm)
        day["sum"] = day["sum"] + np.where(ok, gm, 0.0)
        day["n"] = day["n"] + ok
        if day_hi != day_lo and closed_day is None:       # the tick ends exactly at midnight
            closed_day = self._roll_day(run, day_hi)
        self._track_excess(store, s, e, run, bank, xk, learn, now, dt)
        self._outputs(ctx, s, e, run, learn, bank, mc, h, h_mc, phi)
        store.put_model(s, e, m_cp.MODEL, model, ts=now)
        return model, True, closed_day

    @staticmethod
    def _other_learn(store: Any, s: str, other: str) -> Optional[Dict[str, Any]]:
        m = store.get_model(s, other, m_cp.MODEL)
        return copy.deepcopy(m["learn"]) if m else None

    def _apply_control(self, run: Dict[str, Any], g0: GateState, g1: GateState) -> None:
        """A release (RETURNED) ends the episode: statistics restart. A rebase
        (ACCEPTED) also restarts BOCPD and drops the creep history, which
        belongs to the old regime."""
        released = g1.applied.get("release") != g0.applied.get("release")
        rebased = g1.applied.get("rebase_from") != g0.applied.get("rebase_from")
        if not (released or rebased):
            return
        ChangepointEngine._reset_charts(run)
        if rebased:
            run["bocpd"] = bocpd_new()
            run["boc"] = {"on": False, "onset": math.nan}
            run["daily"] = {"idx": [], "val": []}
            run["creep"] = {"groups": {}, "p": math.nan, "pm": math.nan, "on": False, "axes": []}

    def _supported(self, store: Any, s: str, e: str, ctx: Context,
                   xk: np.ndarray) -> np.ndarray:
        """Key residuals of features the entity's OWN baseline identifies at
        this bucket; the others are NaN (a chart input of 0, no reset).

        zr is exactly N(0, 1) only under a predictive fitted to the entity. In
        a bucket it has no committed rows of (first workday after a weekend
        warm-up, a new entity, a new hour of the day) the predictive is the
        class / system / hyperprior backoff, whose residuals carry a
        systematic offset (e.g. Beta(0.5, 0.5) for a 4xx rate gives z ~ -2 on
        every quiet tick) that a CUSUM with k = 0.25 turns into an alarm
        within hours. Such inputs say nothing about a change of THIS entity."""
        if _m_baseline is None:
            return xk
        model = store.get_model(s, e, BASELINE)
        if not isinstance(model, Mapping) or "current" not in model:
            return xk                      # no B03 in this pipeline: support unknown
        tc = timebins.tctx_from_config(float(ctx.now), ctx.config, float(ctx.window_s))
        W, expo = _m_baseline.own_support(model, tc)
        if expo < SUPPORT_MIN_EXPO_MIN:
            return np.full_like(xk, np.nan)
        return np.where(W[m_cp.KEY_IDX] >= SUPPORT_MIN_ROWS, xk, np.nan)

    @staticmethod
    def _reset_charts(run: Dict[str, Any]) -> None:
        """CUSUM / MCUSUM statistics and the natural-unit excess restart at 0."""
        run["bank"] = m_cp.new_bank()
        run["mc"] = m_cp.new_mc()
        for k in run["excess"]:
            run["excess"][k] = np.zeros((2, m_cp.N_KEY))

    # ------------------------------------------------------------ whitening
    def _whitener(self, store: Any, s: str, e: str, learn: Mapping[str, Any],
                  run: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Key-feature correlation for the MCUSUM: model.density (B06) when it
        carries a full Sigma, else B14's own OAS fit on committed prewhitened
        residuals, else identity. Cached per source version."""
        dens_v = store.model_version(s, e, DENSITY)
        key = ("density", dens_v, learn.get("corr_rev"))
        hit = self._wcache.get((s, e))
        if hit is not None and hit[0] == key:
            return hit[1], hit[2]
        Sig = self._density_sigma(store, s, e) if dens_v is not None else None
        src = "density"
        if Sig is None:
            Sig, src = learn.get("corr"), "own" if learn.get("corr") is not None else "identity"
        C, W = m_cp.whitener(Sig)
        self._wcache[(s, e)] = (key, C, W)
        run["whiten"] = {"Sigma": C, "W": W, "src": src}
        return C, W

    @staticmethod
    def _density_sigma(store: Any, s: str, e: str) -> Optional[np.ndarray]:
        """Full 52x52 (or key 12x12) Sigma from model.density, if it has one."""
        if _m_density is not None and callable(getattr(_m_density, "sigma", None)):
            S = _m_density.sigma(store, s, e)
        else:
            d = store.get_model(s, e, DENSITY)
            S = None
            if isinstance(d, Mapping):
                for k in ("Sigma", "sigma", "cov"):
                    if d.get(k) is not None:
                        S = d[k]
                        break
                cc = d.get("chol_cache")
                if S is None and cc is not None:
                    S = cc.get("Sigma") if isinstance(cc, Mapping) else getattr(cc, "Sigma", None)
                if S is None and d.get("U_k") is not None and d.get("lam") is not None:
                    S = _factor_sigma(d["U_k"], d["lam"])
        if S is None:
            return None
        A = np.asarray(S, dtype=np.float64)
        if A.shape == (FEATURE_DIM, FEATURE_DIM):
            return A[np.ix_(m_cp.KEY_IDX, m_cp.KEY_IDX)]
        if A.shape == (m_cp.N_KEY, m_cp.N_KEY):
            return A
        return None

    # ------------------------------------------------------------ hour / day
    def _roll_hour(self, run: Dict[str, Any], now: float, x_int: float, x_wh: float) -> None:
        """Accumulate this tick into its hour (the hour containing the tick's
        end); a finished hour is one BOCPD step on the standardised sums
        sum/sqrt(n) (N(0,1) under an iid null at any cadence)."""
        hidx = int(math.floor((now - 1e-6) / 3600.0))
        hr = run["hour"]
        if hr["idx"] is None:
            hr.update(idx=hidx, s_int=0.0, n_int=0, s_wh=0.0, n_wh=0)
        elif hidx > hr["idx"]:
            self._close_hour(run)
            gap = min(hidx - hr["idx"] - 1, BOC_MAX_GAP_H)
            for _ in range(max(gap, 0)):
                run["bocpd"] = bocpd_step(run["bocpd"], np.full(2, np.nan))
            hr.update(idx=hidx, s_int=0.0, n_int=0, s_wh=0.0, n_wh=0)
        if math.isfinite(x_int):
            hr["s_int"] += x_int
            hr["n_int"] += 1
        if math.isfinite(x_wh):
            hr["s_wh"] += x_wh
            hr["n_wh"] += 1
        if now - (hidx + 1) * 3600.0 >= -1e-6:              # tick ends on the hour
            self._close_hour(run)
            hr.update(idx=hidx + 1, s_int=0.0, n_int=0, s_wh=0.0, n_wh=0)

    @staticmethod
    def _close_hour(run: Dict[str, Any]) -> None:
        hr = run["hour"]
        x = np.array([hr["s_int"] / math.sqrt(hr["n_int"]) if hr["n_int"] else math.nan,
                      hr["s_wh"] / math.sqrt(hr["n_wh"]) if hr["n_wh"] else math.nan])
        st = bocpd_step(run["bocpd"], x)
        run["bocpd"] = st
        cp = bocpd_prob(st)
        run["cp_prob"] = cp
        boc = run["boc"]
        if cp >= BOC_ALARM:
            j = int(np.argmax(st["p"]))
            r = float(st["r"][j])
            onset = (hr["idx"] + 1 - max(r, 0.0)) * 3600.0       # start of the MAP run
            if not boc["on"]:
                boc["onset"] = onset
            boc["on"] = True
            mu = st["mu"][:, j]
            boc["axes"] = [a for a, m in zip(("volume", "shape"), mu) if abs(m) >= 1.0] \
                or ["volume", "shape"]
        else:
            boc["on"] = False
            boc["onset"] = math.nan

    @staticmethod
    def _roll_day(run: Dict[str, Any], didx: int) -> Optional[int]:
        """Close the stored day when `didx` is a later day. Returns the closed day."""
        day = run["day"]
        if day["idx"] is None:
            day["idx"] = didx
            return None
        if didx <= day["idx"]:
            return None
        closed = int(day["idx"])
        val = np.where(day["n"] >= DAY_MIN_TICKS, day["sum"] / np.maximum(day["n"], 1.0), np.nan)
        dl = run["daily"]
        dl["idx"] = (list(dl["idx"]) + [closed])[-DAILY_KEEP:]
        dl["val"] = (list(dl["val"]) + [val])[-DAILY_KEEP:]
        day.update(idx=didx, sum=np.zeros(N_GROUPS), n=np.zeros(N_GROUPS))
        return closed

    # ------------------------------------------------------------ excess
    def _track_excess(self, store: Any, s: str, e: str, run: Dict[str, Any],
                      bank: Mapping[str, Any], xk: np.ndarray, learn: Mapping[str, Any],
                      now: float, dt: float) -> None:
        """Running sums since each k = 0.25 chart last sat at 0, per side and
        key feature: zr, and the natural value next to its reference value
        inv(vec - scale * zr), so an episode's excess is reported in bytes /
        requests (extensive kinds, summed) or ratio / average units (means)."""
        ex = run["excess"]
        live = np.asarray(bank["S"])[_K025].reshape(2, m_cp.N_KEY) > 0.0
        fin = np.isfinite(xk)
        for k in ex:
            ex[k] = np.where(live, ex[k], 0.0)
        ex["n"] = ex["n"] + (live & fin)
        ex["z"] = ex["z"] + np.where(live & fin, xk, 0.0)
        if not live.any():
            return
        vec = store.vec_at(s, e, VEC, now)
        nat = store.vec_at(s, e, NAT, now)
        if vec is None or nat is None:
            return
        v = np.asarray(vec, dtype=np.float64)[m_cp.KEY_IDX]
        o = np.asarray(nat, dtype=np.float64)[m_cp.KEY_IDX]
        sc = scale_of(learn)[m_cp.KEY_IDX]
        ref = _natural_ref(v - sc * xk, dt)
        good = live & np.isfinite(ref) & np.isfinite(o) & fin
        ex["obs"] = ex["obs"] + np.where(good, o, 0.0)
        ex["ref"] = ex["ref"] + np.where(good, ref, 0.0)
        ex["nn"] = ex["nn"] + good

    # ------------------------------------------------------------ outputs
    def _outputs(self, ctx: Context, s: str, e: str, run: Dict[str, Any], learn: Mapping,
                 bank: Mapping[str, Any], mc: Mapping[str, Any], h: np.ndarray, h_mc: float,
                 phi: np.ndarray) -> None:
        store, now, dt = ctx.store, float(ctx.now), float(ctx.window_s)
        S = np.asarray(bank["S"])
        p_bank = float(np.min(m_cp.peq(S)))
        stat = float(mc["stat"])
        p_mc = m_cp.mcusum_p(stat)
        cp = float(run["cp_prob"])
        cr = run["creep"]
        scores = {"cusum": -math.log10(p_bank), "mcusum": -math.log10(p_mc),
                  "bocpd": -math.log10(max(1.0 - cp, seq.P_FLOOR)) if math.isfinite(cp) else None,
                  "creep": -math.log10(cr["p"]) if math.isfinite(cr["p"]) else None}
        pm = {"cusum": p_bank, "mcusum": p_mc,
              "creep": cr["pm"] if math.isfinite(cr["pm"]) else None}
        on = {"cusum": bool(bank["latch"]["on"]), "mcusum": bool(mc["latch"]["on"]),
              "bocpd": bool(run["boc"]["on"]), "creep": bool(cr["on"])}
        axes: Dict[str, List[str]] = {}
        if on["cusum"]:
            f = np.unique(m_cp.CHART_FEAT[np.asarray(bank["latch"]["alarmed"])])
            axes["cusum"] = sorted({m_cp.KEY_GROUPS[i] for i in f})
        if on["mcusum"]:
            axes["mcusum"] = self._mc_axes(run, mc)
        if on["bocpd"]:
            axes["bocpd"] = list(run["boc"].get("axes") or ["volume", "shape"])
        if on["creep"]:
            axes["creep"] = list(cr["axes"])
        emit.write_scores(store, s, e, now, scores, pm=pm, axes=axes or None,
                          acc_alarm={d: int(v) for d, v in on.items()}, window_s=int(dt))
        onsets = [float(bank["latch"]["onset"])] if on["cusum"] else []
        if on["mcusum"]:
            onsets.append(float(mc["latch"]["onset"]))
        if on["bocpd"]:
            onsets.append(float(run["boc"]["onset"]))
        onsets = [o for o in onsets if math.isfinite(o)]
        onset = min(onsets) if onsets else math.nan
        store.add_vec(s, e, m_cp.CP_PROB, now, [cp], window_s=int(dt))
        store.add_vec(s, e, m_cp.CP_ONSET, now, [onset], window_s=int(dt))
        store.add_vec(s, e, m_cp.CUSUM_STATE, now, m_cp.state_row(bank, mc, phi), window_s=int(dt))
        run["alarm"] = {d: int(v) for d, v in on.items()}
        run["episode"] = ({"onset": onset, "axes": sorted({a for v in axes.values() for a in v})}
                          if any(on.values()) else {})
        self._write_delta(store, s, e, run, bank, mc, on)

    def _idle_outputs(self, ctx: Context, s: str, e: str, run: Dict[str, Any]) -> None:
        """No zr at now. An ACTIVE tick without zr means B04 is stale or failed:
        NaN + behavior.degraded (contract M). An inactive tick is not scored;
        a latched episode keeps reporting its alarm and onset (the evidence
        is still there, silence is B07's business)."""
        store, now, dt = ctx.store, float(ctx.now), float(ctx.window_s)
        act = store.vec_at(s, e, ACTIVE, now)
        if act is not None and len(act) and float(act[0]) > 0.5:
            emit.write_scores(store, s, e, now, {d: None for d in DETS},
                              degraded={d: "stale:behavior.zr" for d in DETS}, window_s=int(dt))
            return
        alarm = run.get("alarm") or {}
        if any(alarm.values()):
            emit.write_scores(store, s, e, now, {}, acc_alarm=dict(alarm), window_s=int(dt))
            ep = run.get("episode") or {}
            store.add_vec(s, e, m_cp.CP_ONSET, now, [float(ep.get("onset", math.nan))],
                          window_s=int(dt))

    @staticmethod
    def _mc_dir(run: Mapping[str, Any], mc: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """(direction, carriers) of the MCUSUM state in psi units, L S with
        L = W^-1; carriers are the key features with |d| >= half the largest."""
        W = (run.get("whiten") or {}).get("W")
        S = np.asarray(mc["S"], dtype=np.float64)
        d = np.linalg.solve(W, S) if W is not None else S
        a = np.abs(d)
        if not a.max() > 0.0:
            return d, np.zeros(0, dtype=np.intp)
        return d, np.flatnonzero(a >= 0.5 * a.max())

    @classmethod
    def _mc_axes(cls, run: Mapping[str, Any], mc: Mapping[str, Any]) -> List[str]:
        """Groups of the features carrying the MCUSUM state."""
        _, idx = cls._mc_dir(run, mc)
        return sorted({m_cp.KEY_GROUPS[i] for i in idx})

    def _write_delta(self, store: Any, s: str, e: str, run: Dict[str, Any],
                     bank: Mapping[str, Any], mc: Mapping[str, Any], on: Mapping[str, bool]
                     ) -> None:
        """profile.extra.regime.delta_by_feature (only that key; B28 owns the
        rest of regime). Written while an episode is latched, cleared once."""
        if not (on["cusum"] or on["mcusum"]):
            if run.get("delta_written"):
                self._put_delta(store, s, e, {})
                run["delta_written"] = False
            return
        ex = run["excess"]
        sides: Dict[Tuple[int, int], None] = {}
        if on["cusum"]:
            for c in np.flatnonzero(np.asarray(bank["latch"]["alarmed"])):
                sides[(0 if m_cp.CHART_SIDE[c] > 0 else 1, int(m_cp.CHART_FEAT[c]))] = None
        if on["mcusum"]:                     # the features carrying the MCUSUM direction
            d, idx = self._mc_dir(run, mc)
            for f in idx:
                sides[(0 if d[f] > 0 else 1, int(f))] = None
        out: Dict[str, Any] = {}
        for sd, f in sides:
            n = float(ex["n"][sd, f])
            nn = float(ex["nn"][sd, f])
            d: Dict[str, Any] = {"dir": "+" if sd == 0 else "-", "ticks": int(n),
                                 "z_mean": float(ex["z"][sd, f] / n) if n else None,
                                 "unit": _KEY_KIND[f]}
            if nn > 0:
                obs, ref = float(ex["obs"][sd, f]), float(ex["ref"][sd, f])
                if _EXTENSIVE[f]:
                    d.update(excess=obs - ref, observed=obs, reference=ref)
                else:
                    d.update(excess=(obs - ref) / nn, observed=obs / nn, reference=ref / nn)
            name = KEY_FEATURES[f]
            if name in out and abs(out[name].get("z_mean") or 0.0) >= abs(d["z_mean"] or 0.0):
                continue
            out[name] = d
        self._put_delta(store, s, e, out)
        run["delta_written"] = True

    @staticmethod
    def _put_delta(store: Any, s: str, e: str, delta: Dict[str, Any]) -> None:
        p = store.profile(s, e)
        if p is None:
            p = EntityProfile(system=s, entity=e)
        reg = p.extra.get("regime")
        if not isinstance(reg, dict):
            reg = p.extra["regime"] = {}
        reg["delta_by_feature"] = delta
        store.put_profile(p)

    # ------------------------------------------------------------ creep
    def _creep(self, ctx: Context, s: str, e: str, model: Dict[str, Any], day: int,
               cache: Dict[Tuple[str, int], Optional[np.ndarray]]) -> None:
        """Daily: Mann-Kendall over the last 14 days of (entity - class median)
        group means of zr, relative to the golden anchor; Sen slope converted
        to log-units with the learned vec-per-zr scale."""
        store, now = ctx.store, float(ctx.now)
        run = model["run"]
        days = np.arange(day - CREEP_DAYS + 1, day + 1)
        own = self._daily_matrix(run, days)
        if int(np.isfinite(own).any(axis=1).sum()) < CREEP_MIN_DAYS:
            return
        med = self._class_median(store, s, e, days, cache)
        if med is None:                                # no peer tier: against golden alone
            med = np.zeros_like(own)
        X = own - med + self._golden_offset(store, s, e)[None, :]
        sc = scale_of(model["learn"])
        cnt_g = _GMAT @ np.isfinite(sc)
        scale_g = np.where(cnt_g > 0, (_GMAT @ np.where(np.isfinite(sc), sc, 0.0))
                           / np.maximum(cnt_g, 1.0), 1.0)
        gate = GateState.from_dict(model["gate"])
        thr = CREEP_SLOPE + abs(float(gate.allow_drift))
        cr = run["creep"]
        prev_on = {g for g, v in (cr.get("groups") or {}).items() if v.get("alarm")}
        groups: Dict[str, Dict[str, Any]] = {}
        p_eff: List[float] = []
        for gi, g in enumerate(GROUP_ORDER):
            v = X[:, gi]
            if int(np.isfinite(v).sum()) < CREEP_MIN_DAYS:
                continue
            _, p = seq.mann_kendall(v)
            slope = seq.sen_slope(v, days.astype(np.float64))
            slope_log = slope * float(scale_g[gi])
            alarm = bool(p < CREEP_P and abs(slope_log) > thr)
            groups[g] = {"p": float(p), "slope_z": float(slope), "slope_log": float(slope_log),
                         "alarm": alarm}
            p_eff.append(p if abs(slope_log) > thr else 1.0)
        if not groups:
            return
        p_min = min(p_eff)
        new_on = [g for g, v in groups.items() if v["alarm"] and g not in prev_on]
        cr.update(groups=groups, p=float(p_min), pm=float(min(1.0, len(p_eff) * p_min)),
                  on=any(v["alarm"] for v in groups.values()),
                  axes=sorted(g for g, v in groups.items() if v["alarm"]), day=int(day))
        if new_on and not ctx.training:
            tz = ctx.config.get("tz") or timebins.DEFAULT_TZ
            self._emit_creep(store, s, e, model, groups, new_on, now, int(days[0]), tz)

    @staticmethod
    def _daily_matrix(run: Mapping[str, Any], days: np.ndarray) -> np.ndarray:
        X = np.full((days.size, N_GROUPS), np.nan)
        dl = run.get("daily") or {}
        pos = {int(d): i for i, d in enumerate(days)}
        for d, v in zip(dl.get("idx", []), dl.get("val", [])):
            i = pos.get(int(d))
            if i is not None:
                X[i] = v
        return X

    def _class_median(self, store: Any, s: str, e: str, days: np.ndarray,
                      cache: Dict[Tuple[str, int], Optional[np.ndarray]]) -> Optional[np.ndarray]:
        """[days, groups] median of the entity's class INCLUDING the entity
        (a median of the other two members of a 3-member class is their mean,
        which carries half of a creeping member's ramp); cells need >= 3
        members with a value. Contract L: a class with < 3 members backs off
        to the system tier. Cached per (tier, day) for the tick, so a system
        of N entities costs O(N) per day, not O(N^2). None: no usable tier."""
        ck = m_class.class_key(store, s, e)
        tiers = ([(ck, lambda: m_class.class_members(store, s, ck))] if ck is not None else [])
        tiers.append(("__system__", lambda: store.entities(s)))
        for key, members in tiers:
            ck_ = (key, int(days[-1]))
            if ck_ not in cache:
                mats = []
                for p in members():
                    m = store.get_model(s, p, m_cp.MODEL)
                    if m:
                        mats.append(self._daily_matrix(m["run"], days))
                med = None
                if len(mats) >= CREEP_MIN_MEMBERS:
                    M = np.stack(mats)
                    cnt = np.isfinite(M).sum(axis=0)
                    with warnings.catch_warnings():    # all-NaN cells: masked below
                        warnings.simplefilter("ignore", RuntimeWarning)
                        med = np.where(cnt >= CREEP_MIN_MEMBERS, np.nanmedian(M, axis=0), np.nan)
                cache[ck_] = med
            if cache[ck_] is not None:
                return cache[ck_]
        return None

    @staticmethod
    def _golden_offset(store: Any, s: str, e: str) -> np.ndarray:
        """Per-group (reference - golden) mean offset in reference sigmas, so
        the daily values are measured against the golden anchor. zr is already
        relative to golden once B03 swaps it in; before that the offset comes
        from m_baseline.golden_offset (or model.baseline['golden']['offset_z'])
        when available, else 0."""
        off = None
        fn = getattr(_m_baseline, "golden_offset", None) if _m_baseline is not None else None
        if callable(fn):
            off = fn(store, s, e)
        else:
            mb = store.get_model(s, e, BASELINE)
            g = mb.get("golden") if isinstance(mb, Mapping) else None
            if isinstance(g, Mapping) and g.get("offset_z") is not None:
                off = g["offset_z"]
        if off is None:
            return np.zeros(N_GROUPS)
        o = np.asarray(off, dtype=np.float64).reshape(-1)
        if o.size != FEATURE_DIM:
            return np.zeros(N_GROUPS)
        fin = np.isfinite(o)
        cnt = _GMAT @ fin
        return np.where(cnt > 0, (_GMAT @ np.where(fin, o, 0.0)) / np.maximum(cnt, 1.0), 0.0)

    @staticmethod
    def _emit_creep(store: Any, s: str, e: str, model: Mapping[str, Any],
                    groups: Mapping[str, Mapping[str, Any]], new_on: Sequence[str],
                    now: float, day0: int, tz: str) -> None:
        p = min(groups[g]["p"] for g in new_on)
        ed = combine.e_day(p, 86400.0)                    # one test per entity-day
        sev = combine.e_day_severity(ed) or "low"
        worst = max(new_on, key=lambda g: abs(groups[g]["slope_log"]))
        slope = groups[worst]["slope_log"]
        store.add_event(BehaviorEvent(
            system=s, entity=e, ts=now, kind="baseline_creep",
            score=float(min(1.0, -math.log10(max(p, seq.P_FLOOR)) / 10.0)),
            severity=Severity(sev),
            contributors=[(g, float(groups[g]["slope_log"])) for g in new_on],
            description=(f"Gradual creep against the golden baseline: {', '.join(new_on)} "
                         f"{'rising' if slope > 0 else 'falling'} "
                         f"{abs(slope):.3f} log-units/day over {CREEP_DAYS} days "
                         f"relative to the class (Mann-Kendall p = {p:.2g})"),
            extra={"groups": {g: dict(v) for g, v in groups.items()},
                   "days": CREEP_DAYS, "golden": _m_baseline is not None},
            p_value=float(p), e_day=float(ed), axes=list(new_on),
            p_by_detector={"creep": float(p)},
            dedupe_key=f"baseline_creep|{s}|{e}|{'+'.join(sorted(new_on))}",
            model_version=int(model.get("version", 0)),
            window=(float(local_day_start(day0, tz)), float(now))))

    # ------------------------------------------------------------ audit
    def _audit(self, ctx: Context, pick: Tuple[float, str, str]) -> None:
        """Round-robin (one entity per hour): bootstrap the committed residuals
        and measure the bank's realised null alarm rate; > 2x target -> h x 1.1
        (at most 1.5x). Deviation: a rate below half the target relaxes h by
        1/1.1 (never below 1x), so audit noise cannot only ratchet h up."""
        store, now, dt = ctx.store, float(ctx.now), float(ctx.window_s)
        _, s, e = pick
        model = store.get_model(s, e, m_cp.MODEL)
        learn, run = model["learn"], model["run"]
        rng = np.random.default_rng(int(combine.seeded_uniform("b14-audit", s, e, now) * 2 ** 53))
        res = m_cp.audit_rate(learn["buf_x"], _gaps(learn["buf_ts"], learn["buf_dt"]),
                              phi_at(learn, dt), dt, float(run["hmult"]), rng)
        rate = res.get("rate", math.nan)
        hm = float(run["hmult"])
        if math.isfinite(rate):
            if rate > 2.0 * m_cp.FAMILY_RATE:
                hm = min(m_cp.HMULT_MAX, hm * 1.1)
            elif rate < 0.5 * m_cp.FAMILY_RATE:
                hm = max(1.0, hm / 1.1)
        run["hmult"] = hm
        run["audit_ts"] = now
        run["audit"] = {"ts": now, "rate": float(rate), "h_mult": hm,
                        "n": float(res.get("n", 0.0))}
        store.put_model(s, e, m_cp.MODEL, model, ts=now)


def local_day_start(day_ordinal: int, tz: str) -> float:
    """UTC epoch of local midnight (tz) on a proleptic day ordinal: the creep
    days are local calendar days, so the event window starts there."""
    d = _dt.date.fromordinal(int(day_ordinal))
    return _dt.datetime(d.year, d.month, d.day, tzinfo=ZoneInfo(tz)).timestamp()


def _natural_ref(v: np.ndarray, dt: float) -> np.ndarray:
    """Invert the key features' vec transform at vec value v (reference level):
    count/bytes log1p(rate/min) -> count per tick; ratio logit -> fraction;
    avg log -> value; identity (updown_log) -> itself."""
    out = np.full(v.shape, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        for i, kind in enumerate(_KEY_KIND):
            x = v[i]
            if not math.isfinite(x):
                continue
            if kind in ("count", "bytes"):
                out[i] = math.expm1(min(x, 700.0)) * dt / 60.0
            elif kind == "ratio":
                out[i] = 1.0 / (1.0 + math.exp(-min(max(x, -700.0), 700.0)))
            elif kind == "avg":
                out[i] = math.exp(min(x, 700.0))
            else:
                out[i] = x
    return out


def _factor_sigma(U_k: Any, lam: Any) -> Optional[np.ndarray]:
    """Key-feature covariance from model.density's contract fields (PCA
    loadings U_k [52, k], eigenvalues lam [k]) completed as a factor model:
    U diag(lam) U^T plus the residual variance 1 - diag on the diagonal (the
    residuals are standardised, so each feature's total variance is ~1).
    Invalid shapes -> None (the caller falls back to its own correlation)."""
    U = np.asarray(U_k, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64).reshape(-1)
    if U.ndim != 2 or U.shape[0] != FEATURE_DIM or U.shape[1] != lam.size or lam.size == 0:
        return None
    Uk = U[m_cp.KEY_IDX]
    S = (Uk * np.clip(lam, 0.0, None)) @ Uk.T
    d = np.diag(S).copy()
    S[np.diag_indices_from(S)] = d + np.maximum(1.0 - d, 0.05)
    return S if np.isfinite(S).all() else None
