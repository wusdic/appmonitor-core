"""B25 FusionEngine: calibrated detector evidence -> per-entity decisions on a
wall-clock false-alarm budget (docs/lib3/engines.md B25, architecture section 4).

Why this stage exists: B24 turns every detector score into a p-value that is
uniform under the entity's own null, but 31 dependent p-values per tick are
not a decision. Fusion combines them without being masked by p ~ 1, removes
what is left of the combination's miscalibration per entity, spends a fixed
number of null alarms per entity-day whatever the cadence, and grades the
result by which behavioural axes moved, not by which detector fired.

How:
  1. Families. p_family = wHMP(p_d) over the family's non-NaN p (lib/combine,
     within-family weights = calibration-health weight_mult). The harmonic
     mean is dominated by the smallest p and insensitive to p ~ 1:
     HMP([1e-3, 0.999, 0.5]) = 3e-3 where ACAT gives 0.5. A family whose
     members are listed in behavior.degraded and have no valid p is written
     as NaN in behavior.p_family ("degraded", never p = 1).
     p_inst = wHMP over the instantaneous members of the families that have
     any; p_all = wHMP over every family. Family weight = feedback family_w
     (lib/m_feedback) x the mean weight_mult of its valid members.
  2. Meta-calibration. Per entity, rings of s = -log10 p_inst and -log10 p_all
     stratified by daypart x cadence class (Mondrian, as B24) live in
     model.calib['meta'] under 'meta_inst@<stratum>' / 'meta_all@<stratum>'
     (layout: lib/m_calib.py; B24 owns every other key). q is B24's own
     p rule, m_calib.p_value: randomised conformal p (U seeded on (s, e,
     'meta_inst' | 'meta_all', ts)), GPD tail above q_0.90, and below 64
     entries the logit blend with the raw HMP p as prior, so B29 recomputes
     q from a ring snapshot with m_calib.p_from_snapshot(..., pm=p_raw).
     e_day = q_all * 86400 / dt.
     Tail shape: the tail is fitted with xi floored at 1/n_u (n_u = number
     of exceedances) instead of calib's plain floor at 0. -log10 of a
     (nearly) valid p has an exponential tail, and the Bayesian predictive of
     an exponential tail whose scale is estimated from n_u exceedances is a
     GPD with xi = 1/n_u (Lomax); the plug-in exponential is anti-conservative
     by Jensen. Measured on the unit-test null (5 p-values, rho = 0.64,
     12 seeds x 2e5 ticks, one M = 256 ring, 4-tick commit delay): realised
     rate at q <= 1e-3 is 1.19x nominal with the xi >= 0 floor and 1.01x
     (0.90 - 1.11) with the predictive floor; 0.97x at 1e-4. Raw HMP: 1.09x.
  3. Evidence CUSUM over instantaneous evidence only: S = max(0, S - ln q_inst - 3)
     (lib/seq), alarm while S >= h = (ln ARL - 3.07)/0.94 with ARL = 33 d in
     ticks (h = 5.31 at 900 s, 8.19 at 60 s). S is not reset on alarm (it
     keeps growing under a persistent attack, MEDIUM at S >= 2h, and decays
     at the null drift of -2 per tick afterwards); alarm episodes are counted
     at onsets (S crosses h), which realise the 33-day ARL (measured
     0.029-0.033 onsets per entity-day at 60/900/3600 s). A tick with no
     instantaneous evidence (q_inst NaN) leaves S unchanged and cannot alarm.
     Audit: once per hour per system one entity (round robin) is checked by
     a moving-block bootstrap (1 h blocks, 200 resampled days) of its
     trusted q_inst over the last 7 d (24 h of 900-s ticks holds too few
     extremes to estimate a 1/33-per-day rate); a realised rate above 2x the
     target raises that entity's h by 5 % (at most +20 % in total), a rate
     at or below target relaxes it by 1 %.
  4. Paths: single tick when e_day(q_all) <= 0.03 alpha_mult (feedback
     alpha_mult per system); evidence alarm; accumulator alarm (any
     behavior.acc_alarm) with the e_day of its own p.
  5. Severity from e_day (combine.e_day_severity, thresholds x alpha_mult).
     HIGH needs >= 2 axes with family e_day <= 0.03 within 4 ticks, or 2
     consecutive ticks at e_day <= 3e-3 alpha_mult, or a discrete finding >=
     HIGH (store.events since t - dt); CRITICAL needs >= 3 axes or a lib-4
     match >= HIGH (store.matches since t - dt, one-tick lag). An uncorroborated
     candidate falls to the highest level it is corroborated for (>= MEDIUM).
     An evidence-only alarm is LOW (MEDIUM if S >= 2h).
  6. Reading rules on the axes of the significant families (family e_day <=
     0.03; if none, the families with e_day <= 1, else the smallest):
     self/peer 2x2 (entities: self abnormal with peer p > 0.1 -> down one,
     both abnormal -> up one within the corroboration limit); the common-mode
     flag (behavior.common.flag) lowers an alarm whose axes are all flagged
     volume / transport / app-error groups by one level, never more; caps:
     volume-only MEDIUM (entity), temporal-only LOW when a schedule_shift event
     explains it (last 24 h), class keys LOW for an intensity-only or an
     app-error / transport-only (system-wide) change. Lowering never drops
     an alarm below LOW (B27 parents common-mode member alarms).
  7. BH at q = 0.05 across a system's keys per tick removes LOW single-tick
     alarms whose q_all fails the step-up threshold.
  8. behavior.alarm {path, severity, axes, families, ...} is written only on
     ticks with an alarm, and never in training mode (warm-up alarms would
     be noise; everything else is still computed and learned). No alarm point
     at ts = now means "no alarm"; behavior.q_all has a row whenever fusion
     ran for the key (NaN when nothing was scored).

Meta rings learn through lib/gating exactly as B24's rings (contract H):
row t is committed D = max(4 ticks, D_min_s) later, admitted with
probability trust(t) (seeded thinning: a ring cannot take a fractional
weight, and dropping partially trusted ticks would cut the null's tail),
held while quarantined, released / rebased / frozen by model.control. A
rollback deletes ring entries after the onset (the rings carry ts) and moves
the journal rows after it to held; a version change resets the rings. Row
values are kept 1 d (the same horizon B24 has for behavior.score), so rows
held longer than that cannot be released.

pipeline_degraded (contract M): when more than 30 % of an entity's families
in play are degraded, one system event (entity '__system__', extra.entity)
is emitted per entity per hour.

Store: reads behavior.p, behavior.axes, behavior.acc_alarm, behavior.degraded,
behavior.common.flag, behavior.calib_health@(s, __system__), feature.tctx,
behavior.trust / trust_prov / quarantine and model.control / model.link (via
gating), model.feedback (m_feedback), store.events, store.matches; writes
behavior.p_family (dict), behavior.q_inst / q_all / e_day / evidence
(1-element float32 vec rings), behavior.alarm (dict, alarm ticks only),
model.calib['meta'] (meta rings and B25 bookkeeping), pipeline_degraded events.
"""
from __future__ import annotations

import math
import weakref
import zlib
from bisect import bisect_left, bisect_right
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import BehaviorEvent, DerivedMetric, MetricKind, Severity
from .lib import calib, combine, emit, gating, m_calib, m_feedback, seq, timebins
from .lib.classkeys import CLASS_PREFIX, SYSTEM_KEY
from .lib.detectors import (ACC_DETECTORS, DETECTOR_INDEX, DETECTOR_INFO, DETECTORS, FAMILIES,
                            FAMILY_IDX, INSTANT_FAMILIES, N_DETECTORS, family_members)

MODEL = m_calib.MODEL
META = m_calib.META
P = emit.P
Q_INST = "behavior.q_inst"
Q_ALL = "behavior.q_all"
E_DAY = "behavior.e_day"
EVIDENCE = "behavior.evidence"
P_FAMILY = "behavior.p_family"
ALARM = "behavior.alarm"
COMMON_FLAG = "behavior.common.flag"
CALIB_HEALTH = "behavior.calib_health"
TCTX = "feature.tctx"
B24_ENGINE = "behavior.calibration"
META_INST, META_ALL = "meta_inst", "meta_all"
LEARNER = "fusion.meta"
ADMIT_SALT = "b25.admit"
STATE = "state"                          # B25 bookkeeping inside model.calib['meta']

PENDING_RETENTION_S = 86400.0            # meta row values kept for commit / release
E_DAY_RETENTION_S = 8 * 86400.0          # behavior.e_day, as q_all (contract B)

# decision constants (engines.md B25)
EVIDENCE_ARL_DAYS = 33.0
SINGLE_E_DAY = 0.03                      # single-tick path, x alpha_mult
AXIS_E_DAY = 0.03                        # a family "significant" for corroboration / rules
CONSEC_E_DAY = 3e-3                      # 2 consecutive ticks at <= this (x alpha_mult)
CONTRIB_E_DAY = 1.0                      # rule axes when no family is significant
CORR_TICKS = 4
BH_Q = 0.05
PEER_NORMAL_P = 0.1
SCHEDULE_SHIFT_LOOKBACK_S = 86400.0
DEGRADED_FRAC = 0.3
DEGRADED_EVENT_EVERY_S = 3600.0

# evidence audit
H_MULT_MAX = 1.2
H_MULT_STEP_UP = 0.05
H_MULT_STEP_DOWN = 0.01
AUDIT_PERIOD_S = 3600.0
AUDIT_WINDOW_S = 7 * 86400.0
AUDIT_MIN_S = 2 * 86400.0
AUDIT_BOOT_DAYS = 200.0
AUDIT_BLOCK_S = 3600.0
AUDIT_TRUST_MIN = 0.5

PATH_SINGLE = "single_tick"
PATH_EVIDENCE = "evidence_cusum"
PATH_ACC = "accumulator"
_PATH_ORDER = {PATH_SINGLE: 0, PATH_ACC: 1, PATH_EVIDENCE: 2}      # tie-break (lower wins)

LEVELS = ("low", "medium", "high", "critical")
_RANK = {None: 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
_BY_RANK = {v: k for k, v in _RANK.items()}
_HIGH_SEV = frozenset({Severity.HIGH, Severity.CRITICAL, "high", "critical"})

# contract F discrete findings (the corroborating kinds; not incident / regime / ops)
DISCRETE_KINDS = frozenset({
    "first_seen", "rare_access", "class_adopted", "client_change", "client_impersonation",
    "identity_mismatch", "unknown_identity", "low_identifiability", "entity_resolution",
    "possible_impersonation", "shared_ip", "identity_moved", "link_retracted",
    "new_entity_matched", "new_entity_unmatched", "class_transition", "class_split",
    "class_merge", "peer_outlier", "system_shift", "coherent_shift", "class_shift",
    "class_adoption_risky", "schedule_shift", "beacon", "budget_exceeded", "baseline_creep",
})

# axes the common-mode flag may discount, and the class-level system-wide set
VOLUME, TRANSPORT, APP_ERROR = "volume", "transport", "app_error"
TEMPORAL, PEER = "temporal", "peer"
COMMON_MODE_AXES = frozenset({VOLUME, TRANSPORT, APP_ERROR})
_AXIS_ALIASES = {"app": APP_ERROR, "app-error": APP_ERROR, "apperror": APP_ERROR,
                 "app_errors": APP_ERROR}

_NAN = math.nan
_FAM_ALL: List[Tuple[str, List[int]]] = [(f, list(FAMILY_IDX[f])) for f in FAMILIES]
_FAM_INST: List[Tuple[str, List[int]]] = [
    (f, [DETECTOR_INDEX[d] for d in family_members(f, "inst")]) for f in INSTANT_FAMILIES]
_FAMILY_OF: List[str] = [str(DETECTOR_INFO[d]["family"]) for d in DETECTORS]
_FID: List[int] = [FAMILIES.index(f) for f in _FAMILY_OF]
_INST: List[bool] = [DETECTOR_INFO[d]["kind"] == "inst" for d in DETECTORS]
_ORDER = range(N_DETECTORS)
_DEFAULT_AXES: List[Tuple[str, ...]] = [tuple(DETECTOR_INFO[d]["axes"]) for d in DETECTORS]
_ACC_SET = frozenset(ACC_DETECTORS)
_ONES = [1.0] * N_DETECTORS


def _f(x: Any) -> float:
    if x is None:
        return _NAN
    try:
        return float(x)
    except (TypeError, ValueError):
        return _NAN


def canonical_axis(axis: Any) -> str:
    """Lower-case axis name with the app-error spellings unified ('app' is the
    feature group whose change is an app-error change for the reading rules)."""
    a = str(axis).strip().lower()
    return _AXIS_ALIASES.get(a, a)


# ======================================================================= fusion
class Fused(NamedTuple):
    p_family: Dict[str, float]      # families with >= 1 valid p
    p_inst: float                   # wHMP over instantaneous members' family p
    p_all: float                    # wHMP over every family
    w_family: Dict[str, float]      # across-family weight used for p_all


def fuse(p_row: Sequence[float], family_w: Optional[Mapping[str, float]] = None,
         wmult: Optional[Sequence[float]] = None) -> Fused:
    """Step 1: family, instantaneous and overall weighted HMP of one p row
    (aligned to DETECTORS). NaN / out-of-range p are unscored and dropped;
    a family (or the whole row) with no valid p gives no entry (NaN).

    Within a family the weight of detector d is wmult[d] (calibration health);
    across families it is family_w[f] x mean(wmult over the valid members).
    The within-family sums are combine.whmp's formula written out (sum w /
    sum w/p, p clipped to [1e-300, 1]; weights are health multipliers in
    (0, 1], so no rescaling is needed): one pass over the 31 columns instead
    of 12 whmp calls per entity per tick. The across-family steps call
    combine.whmp.
    """
    vals = p_row.tolist() if isinstance(p_row, np.ndarray) else list(p_row)
    wm = _ONES if wmult is None else wmult
    nf = len(FAMILIES)
    sw = [0.0] * nf
    swp = [0.0] * nf
    iw = [0.0] * nf
    iwp = [0.0] * nf
    cnt = [0] * nf
    icnt = [0] * nf
    pfl = combine.P_FLOOR
    for i in _ORDER:
        v = vals[i]
        if 0.0 <= v <= 1.0:                     # False for NaN
            f = _FID[i]
            w = wm[i]
            x = w / (v if v > pfl else pfl)
            sw[f] += w
            swp[f] += x
            cnt[f] += 1
            if _INST[i]:
                iw[f] += w
                iwp[f] += x
                icnt[f] += 1
    fw = family_w or {}
    pf: Dict[str, float] = {}
    wf: Dict[str, float] = {}
    ip: List[float] = []
    iws: List[float] = []
    for f in range(nf):
        c = cnt[f]
        if not c:
            continue
        name = FAMILIES[f]
        g = float(fw.get(name, 1.0))
        p = sw[f] / swp[f]
        pf[name] = 1.0 if p > 1.0 else p
        wf[name] = g * sw[f] / c
        if icnt[f]:
            p = iw[f] / iwp[f]
            ip.append(1.0 if p > 1.0 else p)
            iws.append(g * iw[f] / icnt[f])
    p_all = combine.whmp(list(pf.values()), list(wf.values())) if pf else _NAN
    p_inst = combine.whmp(ip, iws) if ip else _NAN
    return Fused(pf, p_inst, p_all, wf)


# ============================================================ meta-calibration
def meta_score(p: float) -> float:
    """Meta ring score s = -log10 p (p clipped to [1e-300, 1]); NaN -> NaN."""
    p = _f(p)
    if not p == p:
        return _NAN
    p = 1.0 if p > 1.0 else (combine.P_FLOOR if p < combine.P_FLOOR else p)
    return -math.log10(p)


def meta_tail(ring: calib.Ring, ts: float = _NAN) -> Optional[calib.GPDTail]:
    """GPD tail of a meta ring with the predictive xi floor 1/n_u (module docstring)."""
    n = len(ring)
    if n < calib.MIN_EXCEED:
        return None
    n_u = max(calib.MIN_EXCEED, n - 1 - int(math.floor(calib.TAIL_Q * (n - 1))))
    return calib.fit_tail(ring, now_ts=ts, xi_min=1.0 / n_u)


def meta_add(ring: calib.Ring, score: float, ts: float, count: int) -> int:
    """Add one admitted score and refit the tail every GPD_REFIT_TICKS
    additions; returns the new refit counter."""
    ring.add(score, ts)
    count += 1
    if count >= calib.GPD_REFIT_TICKS:
        ring.gpd = meta_tail(ring, ts)
        count = 0
    return count


def meta_q(ring: Optional[calib.Ring], score: float, u: float, p_raw: float) -> float:
    """q = B24's p rule on the meta ring (m_calib.p_value): conformal + tail,
    blended with the raw HMP p below 64 entries. NaN score -> NaN."""
    return m_calib.p_value(ring, score, u, p_raw)


def meta_phase(entity: str, key: str) -> int:
    """Initial refit counter of a new ring (spreads GPD refits over ticks, as B24)."""
    return zlib.crc32(f"{entity}|{key}".encode("utf-8")) % calib.GPD_REFIT_TICKS


# ============================================================== evidence CUSUM
def evidence_h(dt_s: float, h_mult: float = 1.0) -> float:
    """h = (ln ARL - 3.07) / 0.94 with ARL = 33 d in ticks, times the audit multiplier."""
    return seq.h_evidence(seq.arl_ticks(EVIDENCE_ARL_DAYS, dt_s)) * float(h_mult)


def evidence_update(S: float, q_inst: float, h: float) -> Tuple[float, bool, bool]:
    """One tick: (S', updated, alarm). NaN q_inst leaves S unchanged and cannot
    alarm; a NaN stored S restarts at 0 (seq.evidence_cusum_step)."""
    S = _f(S)
    q = _f(q_inst)
    if not q == q:
        return (S if S == S else 0.0), False, False
    S2 = seq.evidence_cusum_step(S, q)
    return S2, True, S2 >= h


def evidence_path(q: np.ndarray, h: float, S0: float = 0.0,
                  chunk: int = 65536) -> Tuple[np.ndarray, int]:
    """Vectorised evidence_update over a q_inst stream (Lindley recursion
    S_t = X_t - min(-S_{t0}, min_{j<=t} X_j) per chunk, X = cumsum(-ln q - 3)).
    NaN q is a non-update (increment 0, no alarm). Returns (S[t], number of
    alarm onsets = updated ticks with S >= h whose previous S < h)."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    upd = np.isfinite(q)
    x = np.zeros(q.size)
    qq = np.clip(q[upd], combine.P_FLOOR, 1.0)
    x[upd] = -np.log(qq) - seq.EVIDENCE_DRIFT
    out = np.empty(q.size)
    s_last = float(S0) if math.isfinite(S0) else 0.0
    for a in range(0, q.size, chunk):
        X = np.cumsum(x[a:a + chunk])
        m = np.minimum(np.minimum.accumulate(X), -s_last)
        S = X - m
        out[a:a + chunk] = S
        s_last = float(S[-1])
    prev = np.concatenate(([float(S0) if math.isfinite(S0) else 0.0], out[:-1]))
    onsets = int(np.count_nonzero(upd & (out >= h) & (prev < h)))
    return out, onsets


def audit_rate(q: np.ndarray, h: float, dt_s: float, rng: np.random.Generator,
               days: float = AUDIT_BOOT_DAYS, block_s: float = AUDIT_BLOCK_S) -> float:
    """Evidence-alarm onsets per day of a moving-block bootstrap of `q`
    (blocks of max(4, block_s/dt) ticks, `days` of resampled stream).
    NaN when there are fewer than 4 blocks of finite q."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    q = q[np.isfinite(q)]
    b = max(4, int(round(block_s / dt_s)))
    if q.size < 4 * b:
        return _NAN
    length = int(math.ceil(days * 86400.0 / dt_s))
    nb = int(math.ceil(length / b))
    starts = rng.integers(0, q.size - b + 1, size=nb)
    idx = (starts[:, None] + np.arange(b)[None, :]).reshape(-1)[:length]
    _, onsets = evidence_path(q[idx], h)
    return onsets / float(days)


# ======================================================================== BH
def bh_threshold(pvals: Iterable[float], q: float = BH_Q) -> float:
    """Benjamini-Hochberg step-up threshold over the finite p: the largest
    p_(k) with p_(k) <= k q / m; -1 when nothing is rejected (so `p <= thr`
    is the rejection test)."""
    p = np.asarray([v for v in pvals if v == v], dtype=np.float64)
    m = p.size
    if m == 0:
        return -1.0
    p.sort()
    ok = np.flatnonzero(p <= q * np.arange(1, m + 1) / m)
    return float(p[ok[-1]]) if ok.size else -1.0


# ================================================================ severity
def lower(sev: Optional[str], floor: str = "low") -> Optional[str]:
    """One level down, never below `floor` (an alarm stays an alarm)."""
    if sev is None:
        return None
    return _BY_RANK[max(_RANK[floor], _RANK[sev] - 1)]


def raise_(sev: Optional[str], ceiling: str) -> Optional[str]:
    """One level up, never above `ceiling` (the corroboration limit)."""
    if sev is None:
        return None
    return _BY_RANK[max(_RANK[sev], min(_RANK[ceiling], _RANK[sev] + 1))]


def cap(sev: Optional[str], ceiling: str) -> Optional[str]:
    if sev is None:
        return None
    return _BY_RANK[min(_RANK[sev], _RANK[ceiling])]


# ================================================================ meta state
def new_meta() -> Dict[str, Any]:
    """An empty B25 part of model.calib: meta rings live at the top level of
    this dict (m_calib.ring finds them); bookkeeping under 'state'."""
    return {STATE: {"gate": gating.GateState(), "pending": {}, "refit": {}, "S": 0.0,
                    "h_mult": 1.0, "hist": [], "n_admit": 0, "resets": 0,
                    "degraded_ts": -math.inf, "audit": None}}


def meta_rings(meta: Mapping[str, Any]) -> Dict[str, calib.Ring]:
    return {k: v for k, v in meta.items() if isinstance(v, calib.Ring)}


def _ensure_meta(model: Dict[str, Any]) -> Dict[str, Any]:
    """model['meta'] with live Rings and a live GateState (created if absent)."""
    meta = model.get(META)
    if not isinstance(meta, dict):
        meta = model[META] = new_meta()
        return meta
    st = meta.get(STATE)
    if isinstance(st, dict) and isinstance(st.get("gate"), gating.GateState):
        return meta                                  # steady state
    fresh = new_meta()[STATE]
    if not isinstance(st, dict):
        st = meta[STATE] = fresh
    for k, v in fresh.items():
        st.setdefault(k, v)
    st["gate"] = gating.GateState.from_dict(st.get("gate"))
    for k, v in list(meta.items()):
        if k != STATE and not isinstance(v, calib.Ring):
            r = m_calib.as_ring(v)
            if r is None:
                del meta[k]
            else:
                meta[k] = r
    return meta


def _reset_rings(meta: Dict[str, Any]) -> None:
    for k in list(meta_rings(meta)):
        del meta[k]
    st = meta[STATE]
    st["refit"] = {}
    st["resets"] = int(st.get("resets", 0)) + 1


def _remove_after(meta: Dict[str, Any], tau: float) -> int:
    removed = 0
    refit = meta[STATE]["refit"]
    for key, r in meta_rings(meta).items():
        k = r.remove_after(tau)
        if k:
            removed += k
            r.gpd = meta_tail(r, tau)
            refit[key] = 0
    return removed


class _MRow(NamedTuple):
    s: str
    e: str
    ts: float
    s_inst: float
    s_all: float
    stratum: str


class _MetaLearner(gating.GatedLearner):
    """GatedLearner whose rollback deletes meta ring entries after the onset
    (rings carry ts; B24 does the same) instead of checkpoint + replay."""

    def _rollback(self, store, s, e, state, g, tau, now, dt_s):  # noqa: D401
        j = g.journal
        k = bisect_right(j, tau, key=lambda r: r.ts)
        moved = j[k:]
        removed = _remove_after(state, tau)
        g.journal = j[:k]
        if moved and not g.frozen:
            g.held = sorted(g.held + moved, key=lambda r: r.ts)
        g.applied["_last_rollback"] = {"tau": tau, "wall": now, "moved": len(moved),
                                       "removed": removed, "mode": "delete_after",
                                       "complete": True}
        return state


# ================================================================ per tick
class _SysCtx(NamedTuple):
    s: str
    fw: Dict[str, float]
    alpha: float
    wm: List[float]
    h_base: float
    daypart: str
    b24_failed: bool


class _Rec:
    """Phase-1 result of one key, finalised after the system-wide BH."""
    __slots__ = ("e", "is_class", "st", "q_all", "e_day", "S", "S_prev", "h", "updated",
                 "ev_alarm", "fused", "sig", "fam_axes", "acc", "e_acc", "row")

    def __init__(self, **kw: Any) -> None:
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


class FusionEngine(Engine):
    name = "behavior.fusion"
    layer = "behavior"
    consumes = ["behavior.p", "behavior.axes", "behavior.acc_alarm", "behavior.degraded",
                "behavior.common.flag", "behavior.calib_health", "feature.tctx",
                "behavior.trust", "behavior.trust_prov", "behavior.quarantine",
                "model.feedback", "model.control", "model.link", "store.events",
                "store.matches"]
    produces = ["behavior.p_family", "behavior.q_inst", "behavior.q_all", "behavior.e_day",
                "behavior.evidence", "behavior.alarm", "model.calib.meta",
                "event.pipeline_degraded"]
    description = ("Weighted-HMP family fusion, per-entity meta-calibration (Mondrian rings, "
                   "GPD tail), wall-clock evidence CUSUM, accumulator alarms, corroborated "
                   "severities and axis reading rules.")
    interval = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._learners: Dict[float, _MetaLearner] = {}
        self._strata: Dict[Tuple[str, int], str] = {}
        self._keys: Dict[Tuple[str, str], str] = {}
        self._reset_flag = False
        self._ret_store: Optional[weakref.ref] = None

    # ------------------------------------------------------------- plumbing
    def _learner(self, d_min_s: Any) -> _MetaLearner:
        d = float(d_min_s) if d_min_s is not None else gating.D_MIN_S
        lr = self._learners.get(d)
        if lr is None:
            lr = self._learners[d] = _MetaLearner(
                name=LEARNER, init=new_meta, update=self._update, fetch=self._fetch,
                dump=m_calib.to_json, load=lambda blob: blob, merge=self._merge,
                on_rebase=self._on_rebase, d_min_s=d, ckpt_every_s=math.inf, clock=Q_ALL)
        return lr

    def _ensure_retention(self, store) -> None:
        """behavior.e_day is missing from contract B's table: keep it 8 d like q_all."""
        if self._ret_store is None or self._ret_store() is not store:
            store.set_retention(E_DAY, None, E_DAY_RETENTION_S)
            self._ret_store = weakref.ref(store)

    def _stratum(self, daypart: str, cc: int) -> str:
        k = (daypart, cc)
        st = self._strata.get(k)
        if st is None:
            st = self._strata[k] = calib.stratum_key(daypart, cc)
        return st

    def _ring_key(self, kind: str, stratum: str) -> str:
        k = (kind, stratum)
        key = self._keys.get(k)
        if key is None:
            key = self._keys[k] = calib.ring_key(kind, stratum)
        return key

    # ------------------------------------------------------ learner callbacks
    def _fetch(self, store, s: str, e: str, ts: float) -> Optional[_MRow]:
        model = store.get_model(s, e, MODEL)
        meta = model.get(META) if isinstance(model, Mapping) else None
        if not isinstance(meta, Mapping):
            return None
        v = meta[STATE]["pending"].get(ts)
        if v is None:
            return None                              # older than the value horizon
        return _MRow(s, e, ts, v[0], v[1], v[2])

    def _update(self, meta: Dict[str, Any], row: _MRow, w: float) -> Dict[str, Any]:
        w = float(w)
        w = 0.0 if not w > 0.0 else (1.0 if w > 1.0 else w)     # NaN -> 0
        if w >= 1.0:
            admit = True
        elif w > 0.0:
            admit = combine.seeded_uniform(row.s, row.e, ADMIT_SALT, float(row.ts)) < w
        else:
            admit = False
        if not admit:
            return meta
        st = meta[STATE]
        refit = st["refit"]
        for kind, x in ((META_INST, row.s_inst), (META_ALL, row.s_all)):
            if not x == x:
                continue
            key = self._ring_key(kind, row.stratum)
            r = meta.get(key)
            if r is None:
                r = meta[key] = calib.Ring()
                refit[key] = meta_phase(row.e, key)
            refit[key] = meta_add(r, x, row.ts, refit.get(key, 0))
        st["n_admit"] = int(st.get("n_admit", 0)) + 1
        return meta

    def _on_rebase(self, meta: Dict[str, Any], tau: float) -> Dict[str, Any]:
        _reset_rings(meta)
        self._reset_flag = True
        return meta

    def _merge(self, own: Dict[str, Any], other: Dict[str, Any], w: float) -> Dict[str, Any]:
        """Link seeding: own meta ring += the other's most recent M*w entries per key."""
        refit = own[STATE]["refit"]
        for key, ro in meta_rings(other).items():
            nr = calib.seed_ring(own.get(key) or calib.Ring(), ro, frac=w)
            if len(nr):
                nr.gpd = meta_tail(nr, float(nr.ts.max()))
            own[key] = nr
            refit[key] = 0
        return own

    def _other_meta(self, store, s: str, src: str) -> Optional[Dict[str, Any]]:
        m = store.get_model(s, src, MODEL)
        meta = m.get(META) if isinstance(m, Mapping) else None
        return meta if isinstance(meta, dict) and meta_rings(meta) else None

    # ------------------------------------------------------------------ run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        now = float(ctx.now)
        dt = float(ctx.window_s)
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError(f"fusion: ctx.window_s={ctx.window_s!r} is not a positive cadence")
        config = ctx.config or {}
        self._ensure_retention(store)
        learner = self._learner(config.get("D_min_s", gating.D_MIN_S))
        cc = timebins.cadence_class(dt)
        daypart = timebins.tctx_from_config(now, config, dt)["daypart"]
        b24_failed = store.engine_failed(B24_ENGINE, now)
        fw = m_feedback.family_weights(store)
        h_base = evidence_h(dt)
        n_out = 0
        for s in store.systems():
            keys = store.entities(s) + [k for k in store.pseudo_entities(s)
                                        if k.startswith(CLASS_PREFIX)]
            if not keys:
                continue
            ch = store.latest_derived(s, SYSTEM_KEY, CALIB_HEALTH)
            chv = ch.value if ch is not None and isinstance(ch.value, Mapping) else None
            wm = [m_calib.weight_mult(chv, d) for d in DETECTORS] if chv else _ONES
            sc = _SysCtx(s, fw, m_feedback.alpha_mult(store, s), wm, h_base, daypart, b24_failed)
            recs: List[_Rec] = []
            for e in keys:
                rec = self._phase1(ctx, store, sc, e, now, dt, cc, learner)
                if rec is not None:
                    recs.append(rec)
            if recs:
                bh = bh_threshold(r.q_all for r in recs)
                for rec in recs:
                    n_out += self._finalise(ctx, store, sc, rec, now, dt, bh)
            self._audit(store, s, keys, now, dt, h_base)
        return n_out

    # -------------------------------------------------------------- phase 1
    def _phase1(self, ctx: Context, store, sc: _SysCtx, e: str, now: float, dt: float, cc: int,
                learner: _MetaLearner) -> Optional[_Rec]:
        s = sc.s
        row = None if sc.b24_failed else store.vec_at(s, e, P, now)
        model = store.get_model(s, e, MODEL)
        has_meta = isinstance(model, Mapping) and isinstance(model.get(META), Mapping)
        if row is None and not has_meta:
            return None                              # never fused, nothing scored
        if model is None:
            model = {}
            store.put_model(s, e, MODEL, model, version=0, ts=now)   # before fetch reads it
        elif not isinstance(model, dict):
            raise TypeError(f"model.calib must be a dict, got {type(model).__name__}")
        meta = _ensure_meta(model)
        st = meta[STATE]
        # --- learn: commit row t - D into the meta rings (contract H)
        gate = st["gate"]
        # gate.version, not applied['version']: gating's fast path leaves
        # applied['version'] unset while the control version equals the gate's
        seen = gate.version
        self._reset_flag = False
        meta, gate = learner.seed_from_link(store, s, e, meta, gate,
                                            lambda src: self._other_meta(store, s, src))
        meta, gate = learner.step(store, s, e, meta, gate, now, dt, training=ctx.training)
        if gate.version != seen and not self._reset_flag:
            _reset_rings(meta)                       # version change without rebase_from
        j = gate.journal
        if j and j[0].ts < now - PENDING_RETENTION_S:
            gate.journal = j[bisect_left(j, now - PENDING_RETENTION_S, key=lambda r: r.ts):]
        st["gate"] = gate
        pend = st["pending"]
        cutoff = now - PENDING_RETENTION_S
        while pend:
            t0 = next(iter(pend))
            if t0 >= cutoff:
                break
            del pend[t0]
        store.put_model(s, e, MODEL, model, version=m_calib.version(model), ts=now)

        S_prev = _f(st.get("S"))
        S_prev = S_prev if S_prev == S_prev else 0.0
        if st.get("warm") and not ctx.training:
            S_prev = 0.0        # warm-up evidence (cold rings, models being learnt) ends here
        st["warm"] = bool(ctx.training)
        h = sc.h_base * float(st.get("h_mult", 1.0))
        win = int(dt)
        if row is None:                              # fused before, unscored now
            nan1 = [_NAN]
            store.add_vec(s, e, Q_INST, now, nan1, window_s=win)
            store.add_vec(s, e, Q_ALL, now, nan1, window_s=win)
            store.add_vec(s, e, E_DAY, now, nan1, window_s=win)
            store.add_vec(s, e, EVIDENCE, now, [S_prev], window_s=win)
            return None

        # --- 1) families, p_inst, p_all
        fz = fuse(row, sc.fw, sc.wm)
        degraded = self._degraded_families(store, s, e, now, fz)
        # --- 2) meta-calibration
        dp = self._daypart(store, s, e, now, sc.daypart)
        stratum = self._stratum(dp, cc)
        s_inst = meta_score(fz.p_inst)
        s_all = meta_score(fz.p_all)
        q_inst = meta_q(meta.get(self._ring_key(META_INST, stratum)), s_inst,
                        m_calib.uniform(s, e, META_INST, now), fz.p_inst)
        q_all = meta_q(meta.get(self._ring_key(META_ALL, stratum)), s_all,
                       m_calib.uniform(s, e, META_ALL, now), fz.p_all)
        if s_inst == s_inst or s_all == s_all:
            pend[now] = (s_inst, s_all, stratum)
        e_day = combine.e_day(q_all, dt)
        # --- 3) evidence CUSUM
        S, updated, ev_alarm = evidence_update(S_prev, q_inst, h)
        st["S"] = S
        # --- writes
        store.add_vec(s, e, Q_INST, now, [m_calib.issued(q_inst)], window_s=win)
        store.add_vec(s, e, Q_ALL, now, [m_calib.issued(q_all)], window_s=win)
        store.add_vec(s, e, E_DAY, now, [e_day], window_s=win)
        store.add_vec(s, e, EVIDENCE, now, [S], window_s=win)
        pf_out: Dict[str, float] = dict(fz.p_family)
        for f in degraded:
            pf_out[f] = _NAN
        if pf_out:
            store.add_derived(DerivedMetric(name=P_FAMILY, value=pf_out, ts=now, system=s,
                                            entity=e, window_s=win, kind=MetricKind.CATEGORICAL))
        if degraded:
            self._maybe_degraded_event(store, s, e, now, st, fz, degraded)
        # --- significant families and accumulator alarms (axes read lazily)
        thr = AXIS_E_DAY * dt / 86400.0
        sig = [f for f, p in fz.p_family.items() if p <= thr]
        acc: List[Tuple[str, float]] = []
        aa = emit.read_dict(store, s, e, emit.ACC_ALARM, now)
        if aa:
            for d, v in aa.items():
                if d in _ACC_SET and _f(v) >= 0.5:
                    pv = _f(row[DETECTOR_INDEX[d]])
                    acc.append((d, pv if 0.0 <= pv <= 1.0 else _NAN))
        e_acc = _NAN
        if acc:
            fin = [p for _, p in acc if p == p]
            if fin:
                e_acc = combine.e_day(min(fin), dt)
        return _Rec(e=e, is_class=e.startswith(CLASS_PREFIX), st=st, q_all=q_all, e_day=e_day,
                    S=S, S_prev=S_prev, h=h, updated=updated, ev_alarm=ev_alarm, fused=fz,
                    sig=sig, fam_axes=None, acc=acc, e_acc=e_acc, row=row)

    def _daypart(self, store, s: str, e: str, now: float, fallback: str) -> str:
        """daypart from B01's feature.tctx at now (dict or vec form), else config."""
        m = store.latest_derived(s, e, TCTX)
        if m is not None and m.ts == now and isinstance(m.value, Mapping):
            dp = m.value.get("daypart")
            if dp in timebins.DAYPARTS:
                return str(dp)
        row = store.vec_at(s, e, TCTX, now)
        if row is not None and len(row) == len(timebins.TCTX_FIELDS):
            dp = timebins.decode_tctx(row).get("daypart")
            if dp in timebins.DAYPARTS:
                return str(dp)
        return fallback

    def _degraded_families(self, store, s: str, e: str, now: float, fz: Fused) -> List[str]:
        """Families with a member in behavior.degraded at now and no valid p."""
        deg = emit.read_dict(store, s, e, emit.DEGRADED, now)
        if not deg:
            return []
        out = set()
        for d in deg:
            i = DETECTOR_INDEX.get(d)
            if i is not None and _FAMILY_OF[i] not in fz.p_family:
                out.add(_FAMILY_OF[i])
        return sorted(out)

    def _maybe_degraded_event(self, store, s: str, e: str, now: float, st: Dict[str, Any],
                              fz: Fused, degraded: List[str]) -> None:
        """contract M: > 30 % of the families in play degraded -> pipeline_degraded,
        at most once per entity per hour."""
        in_play = len(fz.p_family) + len(degraded)
        frac = len(degraded) / in_play
        last = _f(st.get("degraded_ts"))
        if frac <= DEGRADED_FRAC or (last == last and 0.0 <= now - last < DEGRADED_EVENT_EVERY_S):
            return
        st["degraded_ts"] = now
        store.add_event(BehaviorEvent(
            system=s, entity=SYSTEM_KEY, ts=now, kind="pipeline_degraded", score=frac,
            severity=Severity.INFO,
            description=f"{len(degraded)}/{in_play} detector families degraded for {e}",
            extra={"entity": e, "families": list(degraded), "fraction": frac,
                   "engine": self.name},
            dedupe_key=f"pipeline_degraded|{s}|{e}"))

    # -------------------------------------------------------------- phase 2
    def _family_axes(self, store, s: str, rec: _Rec, now: float, dt: float,
                     fams: Iterable[str]) -> Dict[str, Set[str]]:
        """Axes of each family: the axes (behavior.axes, else registry defaults)
        of its driving members = members at e_day <= 0.03, else the smallest p."""
        axd = emit.read_dict(store, s, rec.e, emit.AXES, now)
        vals = rec.row.tolist() if isinstance(rec.row, np.ndarray) else list(rec.row)
        thr = AXIS_E_DAY * dt / 86400.0
        out: Dict[str, Set[str]] = {}
        for f in fams:
            members = [(vals[i], i) for i in FAMILY_IDX[f] if 0.0 <= vals[i] <= 1.0]
            if not members:
                continue
            drivers = [i for p, i in members if p <= thr] or [min(members)[1]]
            ax: Set[str] = set()
            for i in drivers:
                ax |= self._axes_of(axd, i)
            out[f] = ax
        return out

    @staticmethod
    def _axes_of(axd: Mapping[str, Any], i: int) -> Set[str]:
        v = axd.get(DETECTORS[i]) if axd else None
        if isinstance(v, str):
            v = [v]
        ax = {canonical_axis(a) for a in (v or ()) if a}
        return ax or {canonical_axis(a) for a in _DEFAULT_AXES[i]}

    def _finalise(self, ctx: Context, store, sc: _SysCtx, rec: _Rec, now: float, dt: float,
                  bh: float) -> int:
        s, e, st = sc.s, rec.e, rec.st
        alpha = sc.alpha
        fz: Fused = rec.fused
        # axes of the significant families and of the alarmed accumulators
        sig_fams = set(rec.sig)
        sig_axes: Set[str] = set()
        axd = None
        if sig_fams or rec.acc:
            fa = self._family_axes(store, s, rec, now, dt, sig_fams)
            for f in sig_fams:
                sig_axes |= fa.get(f, set())
            if rec.acc:
                axd = emit.read_dict(store, s, e, emit.AXES, now)
                for d, _ in rec.acc:
                    sig_fams.add(_FAMILY_OF[DETECTOR_INDEX[d]])
                    sig_axes |= self._axes_of(axd, DETECTOR_INDEX[d])
        e_path = rec.e_day
        if rec.e_acc == rec.e_acc and not rec.e_acc >= e_path:
            e_path = rec.e_acc                       # min, NaN-aware

        # --- 4) paths and their severity candidates
        paths: Dict[str, Optional[str]] = {}
        rules: List[str] = []
        if rec.e_day <= SINGLE_E_DAY * alpha:
            sev = combine.e_day_severity(rec.e_day, alpha)
            if sev == "low" and not rec.q_all <= bh:
                rules.append("bh")                   # 7) BH on the LOW single-tick path
            else:
                paths[PATH_SINGLE] = sev
        if rec.ev_alarm:
            paths[PATH_EVIDENCE] = "medium" if rec.S >= 2.0 * rec.h else "low"
        if rec.acc:
            paths[PATH_ACC] = combine.e_day_severity(rec.e_acc, alpha) or "low"

        # corroboration history (every scored tick, alarm or not)
        hist = [h for h in (st.get("hist") or []) if now - CORR_TICKS * dt < _f(h[0]) < now]
        prev = hist[-1] if hist and _f(hist[-1][0]) >= now - 1.5 * dt else None
        st["hist"] = (hist + [[now, e_path, sorted(sig_axes)]])[-CORR_TICKS:]
        if not paths:
            return 0

        # --- 5) corroboration limit (store lookups only when a level above
        # MEDIUM is at stake: a HIGH+ candidate or a possible self/peer raise)
        axes4 = set(sig_axes)
        for h in hist[-(CORR_TICKS - 1):]:
            axes4.update(h[2])
        n_axes = len(axes4)
        consec = (prev is not None and _f(prev[1]) <= CONSEC_E_DAY * alpha
                  and e_path <= CONSEC_E_DAY * alpha)
        want = max(_RANK[v] for v in paths.values())
        peer_p = fz.p_family.get(PEER, _NAN)
        self_abn = any(f != PEER for f in sig_fams)
        both_abn = not rec.is_class and self_abn and PEER in sig_fams and peer_p == peer_p
        allowed = "medium"
        if want >= _RANK["medium"] + (0 if both_abn else 1):
            if n_axes >= 2 or consec or self._discrete_high(store, s, e, now, dt):
                allowed = "high"
            if n_axes >= 3 or self._lib4_high(store, s, e, now, dt):
                allowed = "critical"
        capped = {p: cap(v, allowed) for p, v in paths.items()}
        path = min(capped, key=lambda p: (-_RANK[capped[p]], _PATH_ORDER[p]))
        sev = capped[path]
        if want > _RANK[allowed]:
            rules.append(f"corroboration_{allowed}")

        # --- 6) reading rules on the axes
        rule_axes = sig_axes
        rule_fams = sorted(sig_fams)
        if not rule_axes:
            contrib = [f for f, p in fz.p_family.items() if combine.e_day(p, dt) <= CONTRIB_E_DAY]
            if not contrib and fz.p_family:
                contrib = [min(fz.p_family, key=fz.p_family.get)]
            fa = self._family_axes(store, s, rec, now, dt, contrib)
            rule_axes = set().union(*fa.values()) if fa else set()
            rule_fams = sorted(contrib)
        if not rec.is_class and self_abn and peer_p == peer_p:      # self / peer 2x2
            if both_abn:
                up = raise_(sev, allowed)
                if up != sev:
                    sev = up
                    rules.append("self_peer_up")
            elif peer_p > PEER_NORMAL_P:
                sev = lower(sev)
                rules.append("self_peer_down")
        if rule_axes and rule_axes <= COMMON_MODE_AXES:
            flags = emit.read_dict(store, s, e, COMMON_FLAG, now)
            flagged = {canonical_axis(g) for g, v in (flags or {}).items() if _f(v) >= 0.5}
            if rule_axes <= flagged:
                sev = lower(sev)
                rules.append("common_mode")
        if rule_axes:
            if rec.is_class:
                core = rule_axes - {PEER}
                if core and core <= {VOLUME}:
                    sev = cap(sev, "low")
                    rules.append("class_intensity_only")
                elif core and core <= COMMON_MODE_AXES:
                    sev = cap(sev, "low")
                    rules.append("class_system_wide")
            elif rule_axes <= {VOLUME}:
                if _RANK[sev] > _RANK["medium"]:
                    rules.append("volume_only")
                sev = cap(sev, "medium")
            if rule_axes <= {TEMPORAL} and store.events(
                    s, e, since=now - SCHEDULE_SHIFT_LOOKBACK_S, kinds=("schedule_shift",),
                    limit=1):
                sev = cap(sev, "low")
                rules.append("schedule_shift")

        # --- 8) behavior.alarm (never during training: no alerts in warm-up)
        if ctx.training:
            return 0
        alarm = {
            "path": path,
            "paths": sorted(paths, key=_PATH_ORDER.get),
            "severity": sev,
            "axes": sorted(rule_axes),
            "families": rule_fams,
            "e_day": rec.e_day,
            "e_day_path": e_path,
            "q_all": rec.q_all,
            "evidence": rec.S,
            "h": rec.h,
            "onset": bool(rec.ev_alarm and rec.S_prev < rec.h),
            "acc": sorted(d for d, _ in rec.acc),
            "n_axes": n_axes,
            "rules": rules,
        }
        store.add_derived(DerivedMetric(name=ALARM, value=alarm, ts=now, system=s, entity=e,
                                        window_s=int(dt), kind=MetricKind.CATEGORICAL))
        return 1

    @staticmethod
    def _discrete_high(store, s: str, e: str, now: float, dt: float) -> bool:
        for ev in store.events(s, e, since=now - dt, kinds=DISCRETE_KINDS):
            if ev.severity in _HIGH_SEV:
                return True
        return False

    @staticmethod
    def _lib4_high(store, s: str, e: str, now: float, dt: float) -> bool:
        for m in store.matches(s, e, since=now - dt):
            if m.severity in _HIGH_SEV:
                return True
        return False

    # ---------------------------------------------------------------- audit
    def _audit(self, store, s: str, keys: Sequence[str], now: float, dt: float,
               h_base: float) -> None:
        """Hourly block-bootstrap audit of one key's evidence-alarm rate (round robin)."""
        if not self.entity_due(("fusion.audit", s), now, AUDIT_PERIOD_S):
            return
        cand = []
        for e in keys:
            m = store.get_model(s, e, MODEL)
            if isinstance(m, Mapping) and isinstance(m.get(META), Mapping):
                cand.append(e)
        if not cand:
            return
        e = cand[int(now // AUDIT_PERIOD_S) % len(cand)]
        st = store.get_model(s, e, MODEL)[META].get(STATE)
        if not isinstance(st, dict):
            return
        ts, M = store.vec_since(s, e, Q_INST, now - AUDIT_WINDOW_S)
        if ts.size * dt < AUDIT_MIN_S:
            return
        q = M[:, 0].astype(np.float64)
        tt, TM = store.vec_since(s, e, gating.TRUST, now - AUDIT_WINDOW_S)
        if tt.size:                                  # trusted ticks only (null stream)
            idx = np.searchsorted(tt, ts)
            ok = idx < tt.size
            ok[ok] = tt[idx[ok]] == ts[ok]
            tv = np.full(ts.size, _NAN)
            tv[ok] = TM[idx[ok], 0]
            q = q[tv >= AUDIT_TRUST_MIN]
        if q.size * dt < AUDIT_MIN_S:
            return
        h_mult = float(st.get("h_mult", 1.0))
        rng = np.random.default_rng(zlib.crc32(f"{s}|{e}|{int(now)}".encode("utf-8")))
        rate = audit_rate(q, h_base * h_mult, dt, rng)
        if not rate == rate:
            return
        target = 1.0 / EVIDENCE_ARL_DAYS
        if rate > 2.0 * target:
            h_mult = min(H_MULT_MAX, h_mult + H_MULT_STEP_UP)
        elif rate <= target:
            h_mult = max(1.0, h_mult - H_MULT_STEP_DOWN)
        st["h_mult"] = h_mult
        st["audit"] = {"ts": now, "rate": rate, "h_mult": h_mult, "n": int(q.size)}
