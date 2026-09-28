"""B28 GovernorEngine: poisoning-resistant model governance for entities and
classes (docs/lib3/engines.md B28, architecture section 3, contract H).

Why: every learner (B03, B06-B14, B18, B24, B25, ...) learns late, from
trusted rows, and reversibly (lib/gating) - but somebody has to decide what
"trusted" means, when learning must stop, when held rows may be learned after
all and when already-learned rows must be forgotten. An attacker who is
learned becomes invisible; a legitimate change that is never learned alarms
forever. This engine is that decision, and the only writer of
behavior.trust / trust_prov / quarantine and model.control.

Per key (every real entity and every class key 'class:*' of a system), every
tick:

Trust (architecture section 3):
  e_inst     = q_inst * 86400 / dt (B25's meta-calibrated evidence; q_all
               when q_inst is unscored)
  trust_prov = clip(log10(e_inst / 0.1), 0, 1) x [no alarm at t]
               x [no discrete finding >= MEDIUM at t]
  trust      = trust_prov x [no open incident] x [regime normal/returned/accepted]
               x [every accumulator < h/2]
  quarantine = open incident (keyed by this entity) OR regime in
               {suspect, drifting, rejected}
  ctx.training => trust = trust_prov = 1 unless a lib-4 match >= HIGH exists
  (one-tick lag; a HIGH match inside its learnt per-(entity, rule) habit
  envelope, lib/m_habit, does not count: sanctioned recurring automation such
  as a nightly backup would otherwise never leave a trusted night hour for
  B13), and the regime machine does not run (a warm-up is by
  definition the reference period; a SUSPECT carried out of it would
  deadlock the first live tick).
  trust_evidence = on live ticks only, the live trust's gates on the row
               itself: [no alarm] x [no finding >= MEDIUM] x [every
               accumulator < h/2]; NaN on a degraded tick. B24 and B25 cap
               their ring admission with it (m_governor.evidence_weight),
               which binds only for a release (trust_prov has no
               accumulator factor). It carries neither the regime state nor
               the q_inst evidence factor, and is not written in training:
               gating warm-up rows on their own evidence truncates the null
               the rings estimate (see _trust_evidence).
  Unscored is not suspicious: when neither q_inst nor q_all is scored (a
  silent entity, a class key with no instantaneous detector, cold start) the
  evidence factor is 1 and the other factors still gate. Otherwise a quiet
  entity could never learn its own silence (absence is data) and a new one
  could never mature. A tick on which B25 failed (contract M) writes trust =
  trust_prov = NaN (lib/gating then uses weight 0 live) and never counts
  as quiet.
  Accumulator level L_d (B27's scale): ln(1/p_d) / ln(ARL_ticks_d) from the
  calibrated p of each accumulator (>= 1 while acc_alarm is set). The p form
  is used for cusum / mcusum too: m_cp.level is the max of 48 raw charts,
  which sits above h/4 on most null ticks, so a return test on it would
  never pass.

Regime machine (live only):
  normal -> suspect on an alarm (behavior.alarm at t), cp.prob > 0.7, a
      baseline_creep event, or an accumulator at >= h/2. DEVIATION (guard):
      the h/2 accumulator trigger additionally needs the rarest accumulator
      reading to be rarer than 0.1 per entity-day after a Bonferroni factor
      over the scored accumulators (e_day(p_d) * n_acc <= 0.1), or its own
      acc_alarm. Literal h/2 fires on ~9 % of null ticks at 900 s (and more
      at 60 s); every such SUSPECT turns DRIFTING at LOW after 1 h and
      blocks B03's reference anchor for +-24 h. The trust factor keeps the
      literal h/2.
      Onset tau-hat = B14's episode onset (exact, model.cp) or the
      behavior.cp.onset row at t, else t - dt; later, earlier cp onsets
      refine it. If tau-hat is before the commit frontier t - D dt, model.control
      .rollback_to = tau-hat - dt (at most 7 d back, at most once per hour:
      deferred otherwise) and a regime(rollback) event. Quarantine is 1 on that
      tick already, as gating requires.
  suspect -> drifting after >= 1 h and >= 4 ticks (event at LOW).
  suspect / drifting -> returned when every accumulator < h/4 and no alarm
      for max(8 ticks, 2 h): release = [min(tau-hat, q_floor), t] (q_floor =
      the commit frontier when quarantine began: rows after it were held),
      live incidents of the key closed (close_reason returned; B27 notifies),
      quarantine 0 from this tick.
  -> accepted / rejected by the legit log-odds (lib/m_governor.decide);
      a REJECT (freeze) also needs the episode to be corroborated
      (m_governor.reject_corroborated: its own alarm / accumulator evidence
      on >= 4 ticks spanning >= 1 h, two independent malicious sources, or a
      tp label) - one lib-4 HIGH match meeting a one-tick rhythm alarm on the
      first live tick froze entities (integration §8):
      type from the evidence axes (c2 > exfil > identity > new_entity >
      categorical > ramp > shape > rhythm > intensity), prior by type, ln LR
      terms (peer concordance, lib-4, system-tier novelty / external upload,
      identity self-posterior, identity mismatch / client concurrency,
      dispersion ratio, time in regime). Evaluated every tick; the drifting
      event is refreshed every 24 ticks and a key DRIFTING for 14 d is put
      on the label queue (model.governor.label_queue + a regime event).
      ACCEPT: model.control {version + 1, rebase_from = tau-hat, allow_drift =
      the slope for a ramp}, a profile version, regime(accepted) at INFO,
      live incidents closed (accepted). REJECT: model.control.frozen = True
      (held rows discarded), still quarantined, no version bump; once the key
      is quiet again (the returned conditions) it is unfrozen and back to
      normal - no permanent lockout.
  Labels (B23's model.feedback accept / freeze records via m_feedback):
      expected_change -> accept (rebase_from = the record's t0 or tau-hat);
      tp -> reject and freeze (rollback to t0 - dt when it is before the
      frontier). A label freeze lasts while m_feedback.is_frozen.
  Classes: the same machine at 'class:<id>' keys; peer concordance there is
      the fraction of members whose own episode moves the same way within
      +-1 h, and a class change accepts at class level once that fraction
      has held >= 50 % for 24 h (accepted_class_change on the class
      control). Concordant members of a class that accepted within 24 h
      accept with it (class-level fast acceptance).
  Quarantine that ends without a regime episode (an incident closed by B27
      while normal) releases [q_floor, t], so held rows are never
      stranded.

Evidence series. Stationarity (time evidence) and the dispersion ratio are
measured on a per-tick level: the mean over the regime's feature groups of
the group-mean behavior.zr (B04, deseasonalised residuals against the
reference anchor), or feature.vec (log units) when B04 is not running. The
pre-change level has an exponentially weighted (7 d) mean / pairwise
covariance per group kept on trusted normal ticks; within an episode the
level is kept in wall-clock bins (max(900 s, dt)). Mann-Kendall runs on at
most 48 re-binned points (cheap, and hourly-scale bins dampen serial
correlation); time evidence is the largest j in {3, 2, 1} such that the last
j T_type/3 lie inside the episode, after the last new axis, and pass MK p >
0.1 (residuals around the Sen trend for an accepted-slope ramp). The ramp
slope in log-units/day comes from B14's baseline_creep event, else
model.baseline 'slope_log' / 'slope_diag' (requested from B03), else the
Sen slope of the level (log units when the level is feature.vec).

Known limitation: at RETURNED / ACCEPTED the last D rows before the verdict
are not held (quarantine turns 0 before they reach the frontier), so the
learners commit them with their recorded trust (0 while suspect).

Store: reads behavior.alarm, behavior.q_inst, behavior.q_all, behavior.p,
behavior.acc_alarm, behavior.cp.prob, model.cp (m_cp.onset), behavior.zr,
feature.vec, behavior.id, behavior.common.flag, store.incidents,
store.events(since, kinds), store.matches(since) (one-tick lag),
model.feedback (m_feedback), model.class (m_class), model.baseline (slope
diagnostics, safe defaults), health of behavior.fusion; writes
behavior.trust / trust_prov / quarantine / trust_evidence (1-element float32
vec rings),
behavior.regime (dict series), model.control@(s, e|class:<id>),
model.governor, profile.extra.regime (merged: B14 owns delta_by_feature),
profile versions (put_profile_version on ACCEPT), incident closes
(returned / accepted) and BehaviorEvent kind='regime'.
"""
from __future__ import annotations

import math
import weakref
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import (BehaviorEvent, DerivedMetric, EntityProfile, Incident, MetricKind,
                              Severity)
from .lib import combine, emit, gating, m_class, m_cp, m_feedback, m_link, seq
from .lib import grains as GR
from .lib import m_governor as MG
from .lib import m_habit as HB
from .lib.classkeys import CLASS_PREFIX, is_class
from .lib.detectors import ACC_DETECTORS, DETECTOR_INDEX, DETECTOR_INFO, arl_days
from .lib.features import FEATURE_DIM, GROUP_ORDER, GROUPS

HOUR = 3600.0
DAY = 86400.0
_NAN = math.nan

# ------------------------------------------------------------------ series
ALARM = "behavior.alarm"
Q_INST = "behavior.q_inst"
Q_ALL = "behavior.q_all"
Q_INST_H = "behavior.q_inst.h"          # spec v2.1: H-stream instantaneous evidence
HOLD_H_S = 3600.0 + 1e-3                # H-stream values hold until the next H tick
P = emit.P
ACC_ALARM = emit.ACC_ALARM
CP_PROB = m_cp.CP_PROB
ZR = "behavior.zr"
VEC = "feature.vec"
ID = "behavior.id"
COMMON_FLAG = "behavior.common.flag"
BASELINE = "model.baseline"
FUSION_ENGINE = "behavior.fusion"
KIND = "regime"

# ------------------------------------------------------------------ trust
E_INST_REF = 0.1                 # trust_prov = clip(log10(e_inst / 0.1), 0, 1)
ACC_TRUST_LEVEL = 0.5            # trust factor: every accumulator < h/2
ACC_SUSPECT_LEVEL = 0.5          # SUSPECT trigger: some accumulator >= h/2 ...
ACC_SUSPECT_E_DAY = 0.1          # ... rarer than 0.1 / entity-day after Bonferroni (guard)
ACC_RETURN_LEVEL = 0.25          # RETURNED: every accumulator < h/4
CP_PROB_SUSPECT = 0.7
FINDING_SEV = frozenset({Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL,
                         "medium", "high", "critical"})
HIGH_SEV = frozenset({Severity.HIGH, Severity.CRITICAL, "high", "critical"})

# ------------------------------------------------------------------ machine
DRIFT_MIN_S, DRIFT_MIN_TICKS = HOUR, 4
QUIET_TICKS, QUIET_S = 8, 2 * HOUR
UPDATE_TICKS = 24
LABEL_QUEUE_S = 14 * DAY
ROLLBACK_MIN_INTERVAL_S = gating.ROLLBACK_MIN_INTERVAL_S
ROLLBACK_MAX_DEPTH_S = gating.ROLLBACK_MAX_DEPTH_S
PEER_WINDOW_S = HOUR             # concordant onsets within +-1 h
PEER_FRAC = 0.5
PEER_RECENT_S = DAY              # a member's closed episode still counts for 24 h
CLASS_ACCEPT_S = DAY             # class-level accept after 24 h of concordance
NEW_ENTITY_AGE_S = 3 * DAY       # first seen within 3 d before the onset -> new_entity
                                 # (and >= 1 d after the system's first entity)
ID_SELF_MIN = 0.9
ID_FRESH_S = HOUR

# ------------------------------------------------------------------ evidence series
BIN_MIN_S = 900.0
MK_POINTS = 48
MK_P = 0.1
MK_MIN_POINTS = 4
DISPERSION_MAX = 1.5             # variance ratio post / pre
DISP_MIN_N = 8
BASE_HL_S = 7 * DAY
BASE_MIN_W = 8.0
MAX_BINS_S = 15 * DAY
HISTORY_CAP = 32
EPISODES_CAP = 16
REGIME_HEARTBEAT_S = HOUR
EVENT_SCAN = 20000

LIVE = ("open", "acked", "suppressed")

# contract F discrete findings (the kinds that gate trust_prov and feed the evidence)
DISCRETE_KINDS = frozenset({
    "first_seen", "rare_access", "class_adopted", "client_change", "client_impersonation",
    "identity_mismatch", "unknown_identity", "low_identifiability", "entity_resolution",
    "possible_impersonation", "shared_ip", "identity_moved", "link_retracted",
    "new_entity_matched", "new_entity_unmatched", "class_transition", "class_split",
    "class_merge", "peer_outlier", "system_shift", "coherent_shift", "class_shift",
    "class_adoption_risky", "schedule_shift", "beacon", "budget_exceeded", "baseline_creep",
})
NOVELTY_KINDS = frozenset({"first_seen", "rare_access", "class_adopted"})
MISMATCH_KINDS = frozenset({"identity_mismatch", "possible_impersonation", "client_impersonation"})
NEW_ENTITY_KINDS = frozenset({"new_entity_matched", "new_entity_unmatched"})

# ------------------------------------------------------------------ axes -> type
C2_AXES = frozenset({"c2"})
EXFIL_AXES = frozenset({"exfil", "exfiltration"})
IDENTITY_AXES = frozenset({"identity", "credential"})
CATEGORICAL_AXES = frozenset({"categorical", "privilege", "discovery", "collection", "lateral",
                              "breadth", "sequence"})
SHAPE_AXES = frozenset({"shape", "comp", "app", "app_error", "dns", "tls", "transport", "probe"})
RHYTHM_AXES = frozenset({"temporal", "timing", "off_hours"})
_AXIS_ALIASES = {"app-error": "app_error", "apperror": "app_error"}
# axis -> feature groups whose level the episode tracks
AXIS_GROUPS: Dict[str, Tuple[str, ...]] = {
    "volume": ("volume",), "breadth": ("breadth",), "app": ("app",), "app_error": ("app",),
    "dns": ("dns",), "tls": ("tls",), "timing": ("timing",), "transport": ("transport",),
    "probe": ("probe",), "comp": ("comp",), "shape": ("comp",), "exfil": ("volume",),
}
DEFAULT_GROUPS = ("volume",)

_ACC_IDX = np.asarray([DETECTOR_INDEX[d] for d in ACC_DETECTORS], dtype=np.intp)
_ACC_SET = frozenset(ACC_DETECTORS)
_GIDX = {g: i for i, g in enumerate(GROUP_ORDER)}
_NG = len(GROUP_ORDER)
_GMAT = np.zeros((_NG, FEATURE_DIM), dtype=np.float64)
for _g, _idx in GROUPS.items():
    _GMAT[_GIDX[_g], list(_idx)] = 1.0
_GMAT.setflags(write=False)

_SEV_OF = {MG.SUSPECT: Severity.INFO, MG.DRIFTING: Severity.LOW, MG.RETURNED: Severity.INFO,
           MG.ACCEPTED: Severity.INFO, MG.REJECTED: Severity.MEDIUM, MG.ROLLBACK: Severity.INFO}


def _f(x: Any) -> float:
    if x is None:
        return _NAN
    try:
        return float(x)
    except (TypeError, ValueError):
        return _NAN


def _enc(x: Any) -> Optional[float]:
    """JSON-safe float: non-finite -> None."""
    v = _f(x)
    return v if math.isfinite(v) else None


def canonical_axis(a: Any) -> str:
    s = str(a).strip().lower()
    return _AXIS_ALIASES.get(s, s)


def evidence_factor(q_inst: float, q_all: float, dt: float,
                    e_inst: Optional[float] = None) -> float:
    """clip(log10(e_inst / 0.1), 0, 1) with e_inst = q * 86400 / dt (q_inst,
    else q_all); 1.0 when nothing was scored (see the module docstring).
    spec v2.1: `e_inst` given (canonical mode: grains.e_inst over the
    streams, min_s q_s N_s) replaces the per-tick one."""
    if e_inst is not None and e_inst == e_inst:
        e = max(float(e_inst), 0.0)
        if e <= 0.0:
            return 0.0
        v = math.log10(e / E_INST_REF)
        return 0.0 if v <= 0.0 else (1.0 if v >= 1.0 else v)
    q = q_inst if q_inst == q_inst else q_all
    if not q == q:
        return 1.0
    e = combine.e_day(max(float(q), 0.0), dt)
    if e <= 0.0:
        return 0.0
    v = math.log10(e / E_INST_REF)
    return 0.0 if v <= 0.0 else (1.0 if v >= 1.0 else v)


def acc_level(p: float, ln_arl: float) -> float:
    """B27's accumulator level ~ S/h from the calibrated p: ln(1/p) / ln(ARL_ticks)."""
    if not (0.0 <= p <= 1.0) or not ln_arl > 0.0:
        return _NAN
    return -math.log(max(p, 1e-300)) / ln_arl


def group_levels(row: Optional[np.ndarray]) -> np.ndarray:
    """Group means (GROUP_ORDER) of a 52-vector over its finite entries; NaN
    for a group with none."""
    out = np.full(_NG, np.nan)
    if row is None:
        return out
    x = np.asarray(row, dtype=np.float64).reshape(-1)
    if x.size != FEATURE_DIM:
        return out
    fin = np.isfinite(x)
    cnt = _GMAT @ fin
    sm = _GMAT @ np.where(fin, x, 0.0)
    ok = cnt > 0
    out[ok] = sm[ok] / cnt[ok]
    return out


# ============================================================ evidence series maths
def new_base(src: Optional[str]) -> Dict[str, Any]:
    """Pre-change level moments: M[0] pair weights, M[1] sums of x_i over the
    pairs (i, j) both observed, M[2] sums of x_i x_j (EW, half-life 7 d)."""
    return {"src": src, "M": np.zeros((3, _NG, _NG)), "ts": None}


def base_due(base: Mapping[str, Any], now: float) -> bool:
    """At most one base sample per BIN_MIN_S of wall clock: an unbiased
    subsample of per-tick levels, so the cost does not grow at 60 s ticks."""
    last = _f(base.get("ts"))
    return not (last == last and 0.0 <= now - last < BIN_MIN_S - 1e-6)


def base_update(base: Dict[str, Any], lv: np.ndarray, now: float) -> None:
    """EW (half-life 7 d wall clock) pairwise moments of the group levels."""
    fin = np.isfinite(lv)
    if not fin.any():
        return
    last = _f(base.get("ts"))
    g = 2.0 ** (-(now - last) / BASE_HL_S) if last == last and now > last else 1.0
    v = np.empty((2, _NG))
    v[0] = fin
    v[1] = np.where(fin, lv, 0.0)
    M = base["M"]
    M *= g
    M[0] += v[0][:, None] * v[0]
    M[1:] += v[1][None, :, None] * v[:, None, :]      # [x_i f_j, x_i x_j]
    base["ts"] = now


def base_moments(base: Optional[Mapping[str, Any]], groups: Sequence[str]
                 ) -> Tuple[float, float]:
    """(mean, variance) of the equal-weight mean of `groups`' levels under the
    pre-change base; NaN when the base is too thin."""
    if not base:
        return _NAN, _NAN
    M = np.asarray(base["M"])
    W = M[0]
    idx = [_GIDX[g] for g in groups if g in _GIDX and W[_GIDX[g], _GIDX[g]] >= BASE_MIN_W]
    if not idx:
        return _NAN, _NAN
    SX, SXX = M[1], M[2]
    ii = np.asarray(idx)
    Wsub = W[np.ix_(ii, ii)]
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = SX[np.ix_(ii, ii)] / Wsub                 # mean of x_i over pairs (i, j)
        C = SXX[np.ix_(ii, ii)] / Wsub - mx * mx.T
    C = np.where(Wsub >= BASE_MIN_W, C, 0.0)
    m = np.diag(mx)
    w = np.full(len(idx), 1.0 / len(idx))
    var = float(w @ C @ w)
    return float(w @ m), (var if var > 1e-12 else _NAN)


def _rebin(t: np.ndarray, n: np.ndarray, s: np.ndarray, k: int = MK_POINTS
           ) -> Tuple[np.ndarray, np.ndarray]:
    """(t, mean) of at most k equal-count groups of consecutive bins (n-weighted)."""
    if t.size <= k:
        return t, s / n
    starts = (np.arange(k) * t.size) // k               # np.array_split boundaries
    nn = np.add.reduceat(n, starts)
    return np.add.reduceat(t * n, starts) / nn, np.add.reduceat(s, starts) / nn


def bins_arrays(bins: Sequence[Sequence[float]]) -> Tuple[np.ndarray, ...]:
    if not bins:
        e = np.empty(0)
        return e, e, e, e
    B = np.asarray(bins, dtype=np.float64)
    ok = B[:, 1] > 0
    B = B[ok]
    return B[:, 0], B[:, 1], B[:, 2], B[:, 3]


def level_slope_per_day(bins: Sequence[Sequence[float]]) -> float:
    """Sen slope (level units per day) over the episode bins (re-binned)."""
    t, n, s, _ = bins_arrays(bins)
    if t.size < 2:
        return _NAN
    tt, mm = _rebin(t, n, s)
    return seq.sen_slope(mm, tt / DAY)


def stationary_steps(bins: Sequence[Sequence[float]], now: float, step_s: float,
                     start: float, tol: float, slope_per_day: float = 0.0) -> int:
    """Largest j in {3, 2, 1} such that [now - j step, now] lies after `start`
    (within tol) and the re-binned residuals (level - slope * t) over it pass
    Mann-Kendall at p > MK_P with >= MK_MIN_POINTS points; 0 otherwise."""
    t, n, s, _ = bins_arrays(bins)
    if t.size < MK_MIN_POINTS or not step_s > 0.0:
        return 0
    sl = slope_per_day if slope_per_day == slope_per_day else 0.0
    for j in (MG.TIME_STEPS_MAX, 2, 1):
        lo = now - j * step_s
        if lo < start - tol:
            continue
        sel = t >= lo
        if int(sel.sum()) < MK_MIN_POINTS:
            continue
        tt, mm = _rebin(t[sel], n[sel], s[sel])
        if tt.size < MK_MIN_POINTS:
            continue
        r = mm - sl * (tt / DAY)
        _, p = seq.mann_kendall(r)
        if p > MK_P:
            return j
    return 0


def post_variance(bins: Sequence[Sequence[float]], slope_per_day: float = 0.0) -> Tuple[float, int]:
    """Per-tick variance of the level around the episode mean (or trend), and N."""
    t, n, s, q = bins_arrays(bins)
    N = int(n.sum()) if n.size else 0
    if N < 2:
        return _NAN, N
    m = s / n
    sl = slope_per_day if slope_per_day == slope_per_day else 0.0
    tr = sl * (t / DAY)
    a = float(((m - tr) * n).sum() / N)
    fit = a + tr
    within = float(np.maximum(q - s * s / n, 0.0).sum())
    between = float((n * (m - fit) ** 2).sum())
    return (within + between) / (N - 1), N


def regime_type(axes: Set[str], flags: Mapping[str, Any], young: bool, ramp: bool) -> str:
    """Evidence axes -> regime type, the most dangerous reading first."""
    if axes & C2_AXES or flags.get("beacon"):
        return MG.C2
    if axes & EXFIL_AXES or flags.get("exfil_budget"):
        return MG.EXFIL
    if axes & IDENTITY_AXES or flags.get("id_mismatch"):
        return MG.IDENTITY
    if young:
        return MG.NEW_ENTITY
    if axes & CATEGORICAL_AXES:
        return MG.CATEGORICAL
    if ramp:
        return MG.RAMP
    if axes & SHAPE_AXES:
        return MG.SHAPE
    if axes & RHYTHM_AXES or flags.get("schedule_shift"):
        return MG.RHYTHM
    return MG.INTENSITY


def episode_groups(axes: Set[str]) -> List[str]:
    out: List[str] = []
    for a in sorted(axes):
        for g in AXIS_GROUPS.get(a, ()):
            if g not in out:
                out.append(g)
    return out or list(DEFAULT_GROUPS)


def _habitual_high(store: Any, s: str, e: str, m: Any) -> bool:
    """A HIGH (never CRITICAL) lib-4 match inside its habit envelope."""
    sev = str(getattr(m.severity, "value", m.severity)).lower()
    return sev == HB.SEVERITY and HB.habituated(store, s, e, m.signature_id, float(m.ts))


# ============================================================== per-tick inputs
class _Sys:
    """Per-system reads shared by every key of one tick (one indexed query each)."""
    __slots__ = ("s", "events", "matches", "incidents", "members", "class_of", "fb", "fb_ver",
                 "first", "retracted")

    def __init__(self, store: Any, s: str, now: float, lo: float) -> None:
        self.s = s
        self.events: Dict[str, List[BehaviorEvent]] = {}
        for ev in store.events(system=s, since=lo, kinds=DISCRETE_KINDS, limit=EVENT_SCAN):
            if lo < ev.ts <= now and ev.status != "suppressed":
                self.events.setdefault(ev.entity, []).append(ev)
        self.matches: Dict[str, List[Any]] = {}
        for m in store.matches(system=s, since=lo, limit=EVENT_SCAN):
            if lo <= m.ts < now:                   # one-tick lag: lib-4 runs after us
                self.matches.setdefault(m.entity, []).append(m)
        self.incidents: Dict[str, List[Incident]] = {}
        for inc in store.incidents(system=s, status=LIVE):
            self.incidents.setdefault(inc.entity, []).append(inc)
        self.members: Dict[str, List[str]] = {}
        self.class_of: Dict[str, Optional[str]] = {}
        self.fb = m_feedback.get(store)
        self.fb_ver = self.fb.get("version") if self.fb else None
        self.first: Optional[float] = None       # the system's earliest first_seen (lazy)
        # retracted continuity links by their seeded end (B17, integration R21.0)
        self.retracted: Dict[str, List[Dict[str, Any]]] = {}
        for lk in m_link.retractions(m_link.get(store, s), since=now - ROLLBACK_MAX_DEPTH_S):
            self.retracted.setdefault(str(lk["to"]), []).append(lk)


class _Obs:
    """Everything the machine needs about one key at one tick."""
    __slots__ = ("alarm", "alarm_axes", "q_inst", "q_all", "acc_max", "acc_alarmed",
                 "acc_suspect", "cp_prob", "cp_onset", "cp_episode", "findings_med",
                 "lib4_high", "events", "creep_slope", "creep_axes", "lv", "src",
                 "lv_loaded", "incidents", "degraded", "e_inst")


# ====================================================================== engine
class GovernorEngine(Engine):
    _canon = False
    name = "behavior.governor"
    layer = "behavior"
    consumes = ["behavior.alarm", "behavior.q_inst", "behavior.q_all", "behavior.p",
                "behavior.acc_alarm", "behavior.cp.prob", "behavior.cp.onset", "model.cp",
                "behavior.zr", "feature.vec", "behavior.id", "behavior.common.flag",
                "store.incidents", "event.*", "match.*", "model.feedback", "model.class",
                "model.baseline"]
    produces = ["behavior.trust", "behavior.trust_prov", "behavior.quarantine",
                "behavior.trust_evidence", "behavior.regime", "model.control", "model.governor",
                "profile.extra.regime", "profile.versions", "event.regime", "incident.close"]
    description = ("Trust / provisional trust / quarantine rings, the regime state machine "
                   "(suspect, drifting, returned, accepted, rejected) with legitimate-change "
                   "log-odds, retroactive rollback, release, rebase and freeze through "
                   "model.control, class-level acceptance and a cold-start path that cannot "
                   "deadlock.")
    interval = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._prev: "weakref.WeakKeyDictionary[Any, float]" = weakref.WeakKeyDictionary()
        self._lnarl: Dict[float, np.ndarray] = {}

    # ------------------------------------------------------------------ run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        now = float(ctx.now)
        dt = float(ctx.window_s)
        self._canon = GR.canonical(ctx.config)
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError(f"governor: ctx.window_s={ctx.window_s!r} is not a positive cadence")
        cfg = ctx.config or {}
        d_min = cfg.get("D_min_s", gating.D_MIN_S)
        frontier = gating.commit_frontier(now, dt, d_min)
        prev = self._prev.get(store)
        lo = prev if prev is not None and prev < now else now - dt
        degraded = bool(store.engine_failed(FUSION_ENGINE, now))
        n = 0
        for s in store.systems():
            ents = list(store.entities(s))
            classes = sorted({k for k in store.pseudo_entities(s) if k.startswith(CLASS_PREFIX)}
                             | set(m_class.all_class_keys(store, s)))
            if not ents and not classes:
                continue
            sc = _Sys(store, s, now, lo)
            for e in ents + classes:            # members first: classes read their verdicts
                self._key(ctx, sc, e, now, dt, frontier, degraded)
                n += 1
        self._prev[store] = now
        return n

    # ------------------------------------------------------------ one key
    def _key(self, ctx: Context, sc: _Sys, e: str, now: float, dt: float, frontier: float,
             degraded: bool) -> None:
        store, s = ctx.store, sc.s
        model = store.get_model(s, e, MG.MODEL)
        first = not isinstance(model, dict)
        if first:
            model = new_model(now)
        prev_state = model["regime"]
        ob = self._observe(store, sc, e, now, dt, degraded)
        tev = None if ctx.training else self._trust_evidence(ob)
        if ctx.training:
            self._training(model, now)
            trust = prov = 0.0 if ob.lib4_high else 1.0
        else:
            self._labels(store, sc, e, model, ob, now, dt, frontier)
            self._machine(store, sc, e, model, ob, now, dt, frontier)
            prov, trust = self._trust(model, ob, dt, sc, e)
        if sc.retracted.get(e):
            self._link_retractions(store, s, e, model, sc.retracted[e], now)
        q = self._quarantine(model, sc, e)
        if not ctx.training:
            self._orphan_release(store, s, e, model, q, now, frontier)
        model["last_q"] = int(q)
        if model["regime"] in MG.TRUSTED_STATES and trust == trust and trust >= 0.5 \
                and base_due(model["base"], now):
            lv = self._level(store, s, e, model, ob, now)
            if lv is not None:
                base_update(model["base"], lv, now)
        win = int(dt)
        store.add_vec(s, e, MG.TRUST, now, [trust], window_s=win)
        store.add_vec(s, e, MG.TRUST_PROV, now, [prov], window_s=win)
        store.add_vec(s, e, MG.QUARANTINE, now, [1.0 if q else 0.0], window_s=win)
        if tev is not None:
            store.add_vec(s, e, MG.TRUST_EVIDENCE, now, [tev], window_s=win)
        changed = model["regime"] != prev_state or first
        self._write_regime(store, s, e, model, now, dt, changed)
        self._write_profile(store, s, e, model, now, changed)
        model["n"] = int(model.get("n", 0)) + 1
        store.put_model(s, e, MG.MODEL, model, version=int(model.get("version", 0)), ts=now)

    # ------------------------------------------------------------ inputs
    def _ln_arl(self, dt: float) -> np.ndarray:
        """ln ARL_ticks of each accumulator (ACC_DETECTORS order) at cadence dt.
        spec v2.1 (canonical mode): the ARL in each detector's own periods."""
        key = (dt, self._canon)
        hit = self._lnarl.get(key)
        if hit is None:
            md = GR.CANONICAL if self._canon else GR.TICK
            hit = self._lnarl[key] = np.array([
                math.log(arl_days(d) * DAY / GR.period_s(d, dt, md)) for d in ACC_DETECTORS])
        return hit

    def _periods(self, dt: float) -> np.ndarray:
        md = GR.CANONICAL if self._canon else GR.TICK
        return np.array([GR.period_s(d, dt, md) for d in ACC_DETECTORS])

    def _observe(self, store: Any, sc: _Sys, e: str, now: float, dt: float,
                 degraded: bool) -> _Obs:
        s = sc.s
        ob = _Obs()
        ob.degraded = degraded
        a = store.latest_derived(s, e, ALARM)
        ob.alarm = a.value if (a is not None and a.ts == now and isinstance(a.value, Mapping)
                               and not degraded) else None
        ob.alarm_axes = {canonical_axis(x) for x in (ob.alarm or {}).get("axes") or ()}
        ob.q_inst = _vec1(store, s, e, Q_INST, now)
        ob.q_all = _vec1(store, s, e, Q_ALL, now)
        ob.e_inst = None
        # accumulators
        aa = _dict_at(store, s, e, ACC_ALARM, now)
        prow = store.vec_at(s, e, P, now)
        if self._canon:
            # spec v2.1: H-stream values (accumulators, q_inst.h) hold between
            # H ticks; e_inst = min over the streams (grains.e_inst)
            qh = _latest1(store, s, e, Q_INST_H, now - HOLD_H_S)
            ob.e_inst = GR.e_inst({"t": ob.q_inst, "h": qh}, dt, GR.CANONICAL)
            prow = _latest_rows(store, s, e, P, now - HOLD_H_S)
            if not aa:
                tail = store.derived_tail(s, e, ACC_ALARM, 1)
                if tail and tail[-1].ts >= now - HOLD_H_S and isinstance(tail[-1].value, Mapping):
                    aa = tail[-1].value
        alarmed = {d for d, v in aa.items() if d in _ACC_SET and _f(v) >= 0.5} if aa else set()
        ob.acc_alarmed = alarmed
        ob.acc_max = 1.0 if alarmed else _NAN
        ob.acc_suspect = bool(alarmed)
        if prow is not None:
            p = np.asarray(prow, dtype=np.float64)[_ACC_IDX]
            ok = (p >= 0.0) & (p <= 1.0)                     # False for NaN
            if ok.any():
                # vectorised lib/detectors.acc_level (the scale B27 shares)
                L = -np.log(np.maximum(p[ok], 1e-300)) / self._ln_arl(dt)[ok]
                mx = float(L.max())
                if not ob.acc_max >= mx:
                    ob.acc_max = mx
                if not ob.acc_suspect:
                    # guard: h/2 AND rarer than ACC_SUSPECT_E_DAY per day after Bonferroni
                    lim = ACC_SUSPECT_E_DAY * self._periods(dt)[ok] / DAY / int(ok.sum())
                    ob.acc_suspect = bool(np.any((L >= ACC_SUSPECT_LEVEL) & (p[ok] <= lim)))
        # changepoint
        ob.cp_prob = _vec1(store, s, e, CP_PROB, now)
        cpm = m_cp.get(store, s, e)
        run = cpm.get("run") if isinstance(cpm, Mapping) else None
        ob.cp_episode = bool(isinstance(run, Mapping) and run.get("episode"))
        if ob.cp_episode:
            ob.cp_onset = m_cp.onset(store, s, e)            # exact (float64, model.cp)
        elif ob.cp_prob == ob.cp_prob or cpm is not None:
            ob.cp_onset = m_cp.onset(store, s, e, at=now)    # the ring row at t, if any
        else:
            ob.cp_onset = _NAN                               # B14 not running here
        # discrete findings, lib-4
        evs = sc.events.get(e, [])
        ob.events = evs
        ob.findings_med = any(ev.severity in FINDING_SEV for ev in evs
                              if ev.kind in DISCRETE_KINDS)
        # a HIGH match inside its learnt per-(entity, rule) envelope (lib/m_habit,
        # B26 runs before us) is sanctioned recurring activity, not a HIGH
        # finding: it neither zeroes the warm-up trust nor counts as a
        # malicious source (lead decision, round 4)
        ob.lib4_high = any(m.severity in HIGH_SEV and not _habitual_high(store, s, e, m)
                           for m in sc.matches.get(e, []))
        ob.creep_slope, ob.creep_axes = _NAN, set()
        for ev in evs:
            if ev.kind == "baseline_creep":
                ob.creep_axes |= {canonical_axis(x) for x in (ev.axes or ())}
                sl = _creep_slope(ev)
                if sl == sl and not abs(sl) <= abs(ob.creep_slope):
                    ob.creep_slope = sl
        ob.lv, ob.src, ob.lv_loaded = None, None, False   # read lazily (_level)
        ob.incidents = sc.incidents.get(e, [])
        return ob

    def _level(self, store: Any, s: str, e: str, model: Dict[str, Any], ob: _Obs,
               now: float) -> Optional[np.ndarray]:
        """The group levels at now (group means of zr, else feature.vec), read
        once per tick and only when needed (a base update is due or an episode
        is open); the level source is kept consistent (_sync_base)."""
        if not ob.lv_loaded:
            ob.lv_loaded = True
            if self._canon:
                # spec v2.1: zr of the H rows, held between H ticks; the
                # H-grain vec as the fallback source
                row = _latest_row(store, s, e, ZR, now - HOLD_H_S)
                if row is not None:
                    ob.lv, ob.src = group_levels(row), "zr"
                else:
                    row = _latest_row(store, s, e, "feature.vec.h", now - HOLD_H_S)
                    if row is not None:
                        ob.lv, ob.src = group_levels(row), "vec"
                self._sync_base(model, ob)
                return ob.lv
            row = store.vec_at(s, e, ZR, now)
            if row is not None:
                ob.lv, ob.src = group_levels(row), "zr"
            else:
                row = store.vec_at(s, e, VEC, now)
                if row is not None:
                    ob.lv, ob.src = group_levels(row), "vec"
            self._sync_base(model, ob)
        return ob.lv

    # ------------------------------------------------------------ training
    def _training(self, model: Dict[str, Any], now: float) -> None:
        """Warm-up: no regime decisions, no events; a leftover episode is
        dropped silently and the key starts live in NORMAL."""
        model["train_end"] = now
        if model["regime"] != MG.NORMAL:
            model["episode"] = None
            _set_state(model, MG.NORMAL, now, reason="training")

    def _sync_base(self, model: Dict[str, Any], ob: _Obs) -> None:
        """Keep the level source consistent: zr (B04) wins once it exists; a
        missing row of the chosen source is a NaN level, never a switch back."""
        base = model["base"]
        if ob.src is None:
            return
        if base.get("src") is None or (ob.src == "zr" and base["src"] != "zr"):
            model["base"] = new_base(ob.src)
        elif ob.src != base["src"]:
            ob.lv = None

    # ------------------------------------------------------------ labels
    def _labels(self, store: Any, sc: _Sys, e: str, model: Dict[str, Any], ob: _Obs,
                now: float, dt: float, frontier: float) -> None:
        """Apply new expected_change / tp records from B23 (once each)."""
        fb = sc.fb
        if not fb or (not fb.get("accept") and not fb.get("freeze")):
            return
        if model.get("fb_version") == sc.fb_ver and sc.fb_ver is not None:
            return
        model["fb_version"] = sc.fb_ver
        mark = model.get("fb_mark") or [-math.inf, -1]
        recs = []
        for kind, fn in (("accept", m_feedback.accepts), ("freeze", m_feedback.freezes)):
            for r in fn(store, sc.s, e):
                key = (_f(r.get("ts")), int(r.get("seq", 0) or 0))
                if key > (float(mark[0]), int(mark[1])):
                    recs.append((key, kind, r))
        if not recs:
            return
        recs.sort(key=lambda x: x[0])
        for key, kind, r in recs:
            model["fb_mark"] = [key[0], key[1]]
            t0 = _f(r.get("t0"))
            if kind == "freeze":
                self._label_freeze(store, sc, e, model, ob, now, dt, frontier, t0, r)
            elif model["regime"] in (MG.SUSPECT, MG.DRIFTING, MG.REJECTED):
                ep = model["episode"]
                tau = t0 if t0 == t0 else _f(ep.get("onset"))
                self._accept(store, sc, e, model, now, tau, reason="label", extra={
                    "label_id": r.get("label_id"), "scope": r.get("scope")})

    def _label_freeze(self, store: Any, sc: _Sys, e: str, model: Dict[str, Any], ob: _Obs,
                      now: float, dt: float, frontier: float, t0: float,
                      rec: Mapping[str, Any]) -> None:
        if model["regime"] not in (MG.SUSPECT, MG.DRIFTING, MG.REJECTED):
            if model["regime"] != MG.NORMAL:
                _set_state(model, MG.NORMAL, now, reason="settled")
            onset = t0 if _valid_onset(t0, now) else now - dt
            self._open_episode(store, sc, e, model, ob, now, dt, onset, reason="label")
            model["episode"]["flags"]["malicious_label"] = True
            self._announce(store, sc.s, e, model, now, dt, frontier)
        ep = model["episode"]
        ep["flags"]["malicious_label"] = True
        ep["label_frozen"] = True
        if _valid_onset(t0, now) and t0 < _f(ep.get("onset")):
            ep["onset"] = max(t0, now - ROLLBACK_MAX_DEPTH_S)
            model["onset"] = ep["onset"]
            self._want_rollback(store, sc.s, e, model, now, dt, frontier)
        if model["regime"] != MG.REJECTED:
            self._reject(store, sc.s, e, model, now, reason="label",
                         extra={"label_id": rec.get("label_id")})

    # ------------------------------------------------------------ the machine
    def _machine(self, store: Any, sc: _Sys, e: str, model: Dict[str, Any], ob: _Obs,
                 now: float, dt: float, frontier: float) -> None:
        s = sc.s
        st = model["regime"]
        if st in (MG.RETURNED, MG.ACCEPTED):
            if _f(model.get("since")) >= now:
                return                          # decided this tick (a label)
            _set_state(model, MG.NORMAL, now, reason="settled")
            st = MG.NORMAL
        elif st != MG.NORMAL and not isinstance(model.get("episode"), Mapping):
            _set_state(model, MG.NORMAL, now, reason="settled")     # defensive
            st = MG.NORMAL
        opened = False
        if st == MG.NORMAL:
            trig = self._trigger(ob)
            if trig is None:
                return
            onset = ob.cp_onset if _valid_onset(ob.cp_onset, now) else now - dt
            self._open_episode(store, sc, e, model, ob, now, dt, onset, reason=trig)
            st = model["regime"]
            opened = True
        ep = model["episode"]
        self._accumulate(store, sc, e, model, ep, ob, now, dt, frontier)
        if opened:
            self._announce(store, s, e, model, now, dt, frontier)
        if st == MG.REJECTED:
            if self._quiet(ep, ob, now, dt) and not self._label_frozen(store, s, e, ep):
                self._unfreeze(store, s, e, model, now)
            return
        if self._quiet(ep, ob, now, dt):
            self._return(store, s, e, model, now)
            return
        if st == MG.SUSPECT and now - _f(ep["start"]) >= DRIFT_MIN_S \
                and int(ep["ticks"]) >= DRIFT_MIN_TICKS:
            _set_state(model, MG.DRIFTING, now, reason="persisting")
            ep["drift_tick"] = int(ep["ticks"])
            self._event(store, s, e, model, now, MG.DRIFTING)
        self._evaluate(store, sc, e, model, ep, ob, now, dt)

    def _trigger(self, ob: _Obs) -> Optional[str]:
        if ob.alarm is not None:
            return "alarm"
        if ob.cp_prob > CP_PROB_SUSPECT:
            return "cp_prob"
        if any(ev.kind == "baseline_creep" for ev in ob.events):
            return "baseline_creep"
        if ob.acc_suspect:
            return "accumulator"
        return None

    def _open_episode(self, store: Any, sc: _Sys, e: str, model: Dict[str, Any], ob: _Obs,
                      now: float, dt: float, onset: float, reason: str) -> None:
        """normal -> suspect: the episode's working state (announced separately,
        once its type is known)."""
        s = sc.s
        self._level(store, s, e, model, ob, now)
        onset = max(float(onset), now - ROLLBACK_MAX_DEPTH_S)
        axes = set(ob.alarm_axes) | set(ob.creep_axes)
        groups = episode_groups(axes)
        mean_pre, var_pre = base_moments(model["base"], groups)
        young = self._young(store, sc, e, onset) or any(
            ev.kind in NEW_ENTITY_KINDS for ev in ob.events)
        model["branch"] = int(model.get("branch", 0)) + 1
        model["episode"] = {
            "start": now, "onset": onset, "reason": reason, "axes": sorted(axes),
            "groups": groups, "src": model["base"].get("src"),
            "bw": max(BIN_MIN_S, dt), "bins": [], "mean_pre": _enc(mean_pre),
            "var_pre": _enc(var_pre), "flags": {}, "last_alarm": now,
            "last_new_axis": now, "ticks": 0, "young": young, "ramp": False,
            "creep_slope": None, "direction": 0, "concord_since": None,
            "rollback_to": None, "pending_rollback": None, "drift_tick": None,
            "last_update_tick": 0, "label_frozen": False, "degraded_ts": None,
        }
        model["onset"] = onset
        model["label_queue"] = None
        model["type"] = regime_type(axes, {}, young, bool(ob.creep_axes))
        _set_state(model, MG.SUSPECT, now, reason=reason)

    def _young(self, store: Any, sc: _Sys, e: str, onset: float) -> bool:
        """First seen within NEW_ENTITY_AGE_S before the onset, and at least a
        day after the system's first entity (in a fresh deployment every
        entity is 'new'; only a newcomer to a watched system is)."""
        if is_class(e):
            return False
        fs = store.first_seen(sc.s, e)
        if fs is None or onset - fs > NEW_ENTITY_AGE_S:
            return False
        if sc.first is None:
            seen = [store.first_seen(sc.s, x) for x in store.entities(sc.s)]
            sc.first = min((x for x in seen if x is not None), default=fs)
        return fs - sc.first >= DAY

    def _announce(self, store: Any, s: str, e: str, model: Dict[str, Any], now: float,
                  dt: float, frontier: float) -> None:
        """regime(suspect) with the type read from this tick's evidence, then
        the rollback decision (quarantine is already 1 on this tick)."""
        ep = model["episode"]
        model["type"] = regime_type(set(ep["axes"]), ep["flags"], bool(ep.get("young")),
                                    bool(ep.get("ramp")))
        self._event(store, s, e, model, now, MG.SUSPECT, extra={"trigger": ep.get("reason")})
        self._want_rollback(store, s, e, model, now, dt, frontier)

    # --------------------------------------------------- evidence accumulation
    def _accumulate(self, store: Any, sc: _Sys, e: str, model: Dict[str, Any],
                    ep: Dict[str, Any], ob: _Obs, now: float, dt: float,
                    frontier: float) -> None:
        s = sc.s
        ep["ticks"] = int(ep["ticks"]) + 1
        if ob.alarm is not None:
            ep["last_alarm"] = now
        if ob.alarm is not None or ob.acc_alarmed:
            # the anomaly's own persistence (REJECT corroboration, m_governor)
            ep["ev_ticks"] = int(ep.get("ev_ticks") or 0) + 1
            if ep.get("ev_first") is None:
                ep["ev_first"] = now
            ep["ev_last"] = now
        if ob.degraded:
            ep["degraded_ts"] = now
        # axes: a new one restarts the clean duration
        new_axes = set(ob.alarm_axes) | set(ob.creep_axes)
        for d in ob.acc_alarmed:
            new_axes |= _acc_axes(store, s, e, now, d)
        known = set(ep["axes"])
        if new_axes - known:
            if ep["ticks"] > 1:
                ep["last_new_axis"] = now
            ep["axes"] = sorted(known | new_axes)
        # sticky flags
        fl = ep["flags"]
        if ob.lib4_high:
            fl["lib4_high"] = True
        if "beacon" in ob.acc_alarmed or any(ev.kind == "beacon" for ev in ob.events):
            fl["beacon"] = True
        if "budget_exfil" in ob.acc_alarmed:
            fl["exfil_budget"] = True
        if ob.creep_axes or "creep" in ob.acc_alarmed or any(
                ev.kind == "baseline_creep" for ev in ob.events):
            ep["ramp"] = True
        if ob.creep_slope == ob.creep_slope:
            ep["creep_slope"] = ob.creep_slope
        idm = store.latest_derived(s, e, ID)
        if idm is not None and now - idm.ts <= max(ID_FRESH_S, dt) \
                and isinstance(idm.value, Mapping):
            v = _f(idm.value.get("posterior_self"))
            if v == v:
                ep["id_self"] = v
        for ev in ob.events:
            k = ev.kind
            ex = ev.extra or {}
            if k in MISMATCH_KINDS:
                fl["id_mismatch"] = True
            elif k == "client_change" and (ex.get("concurrent") or ex.get("concurrency")):
                fl["client_concurrency"] = True
            elif k == "schedule_shift":
                fl["schedule_shift"] = True
            elif k in NEW_ENTITY_KINDS:
                ep["young"] = True
            if k in NOVELTY_KINDS:
                flags = ex.get("flags") if isinstance(ex.get("flags"), Mapping) else ex
                tier = str(ex.get("tier") or "")
                ext_up = bool(flags.get("external")) and bool(flags.get("upload_dominant"))
                if tier == "system" and _idf_high(ex) or ext_up:
                    fl["sys_novelty"] = True
                if tier == "system" and (flags.get("sensitive") or flags.get("admin")):
                    fl["sys_sensitive"] = True
        # onset refinement from B14 (an alarm-opened episode, cp confirming later)
        if ob.cp_episode and _valid_onset(ob.cp_onset, now) \
                and ob.cp_onset < _f(ep["onset"]) - dt:
            ep["onset"] = max(float(ob.cp_onset), now - ROLLBACK_MAX_DEPTH_S)
            model["onset"] = ep["onset"]
            self._want_rollback(store, s, e, model, now, dt, frontier)
        elif ep.get("pending_rollback") is not None:
            self._want_rollback(store, s, e, model, now, dt, frontier)
        # level bins
        lv = self._level(store, s, e, model, ob, now)
        if lv is not None and ep.get("src") == model["base"].get("src"):
            idx = [_GIDX[g] for g in ep["groups"] if g in _GIDX]
            vals = lv[idx]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                x = float(vals.mean())
                bw = float(ep["bw"])
                t0 = math.floor(now / bw) * bw
                bins = ep["bins"]
                if bins and bins[-1][0] == t0:
                    b = bins[-1]
                    b[1] += 1.0
                    b[2] += x
                    b[3] += x * x
                else:
                    bins.append([t0, 1.0, x, x * x])
                    cut = now - MAX_BINS_S
                    while bins and bins[0][0] < cut:
                        bins.pop(0)
        mp = _f(ep.get("mean_pre"))
        t, n, sm, _ = bins_arrays(ep["bins"])
        if mp == mp and n.size:
            diff = float(sm.sum() / n.sum()) - mp
            ep["direction"] = 1 if diff > 0 else (-1 if diff < 0 else 0)

    # ------------------------------------------------------------ verdicts
    def _terms(self, store: Any, sc: _Sys, e: str, model: Dict[str, Any], ep: Dict[str, Any],
               ob: _Obs, now: float, dt: float) -> Tuple[str, Dict[str, float], bool, bool,
                                                        float]:
        """(type, ln LR terms, ramp_ok, ramp_blocked, detrend slope/day)."""
        fl = ep["flags"]
        axes = set(ep["axes"])
        typ = regime_type(axes, fl, bool(ep.get("young")), bool(ep.get("ramp")))
        terms: Dict[str, float] = {}
        # peer concordance (entity: its class; class key: its members)
        frac = self._concordance(store, sc, e, model, ep, now)
        ep["concord"] = _enc(frac)
        common = False
        if not is_class(e) and frac < PEER_FRAC:
            cf = _dict_at(store, sc.s, e, COMMON_FLAG, now)
            flagged = {canonical_axis(g) for g, v in cf.items() if _f(v) >= 0.5}
            common = bool(flagged & set(ep["groups"]))
        if frac >= PEER_FRAC or common:
            terms["peer"] = MG.LR_PEER
            if ep.get("concord_since") is None:
                ep["concord_since"] = now
        else:
            ep["concord_since"] = None
        if fl.get("lib4_high"):
            terms["lib4"] = MG.LR_LIB4
        if fl.get("sys_novelty"):
            terms["sys_novelty"] = MG.LR_SYS_NOVELTY
        if _f(ep.get("id_self")) >= ID_SELF_MIN:
            terms["id_self"] = MG.LR_ID_SELF
        if fl.get("id_mismatch") or fl.get("client_concurrency"):
            terms["id_mismatch"] = MG.LR_ID_MISMATCH
        # ramp slope (log units / day) and the detrend slope (level units / day)
        ramp_ok = ramp_blocked = False
        detrend = 0.0
        if typ == MG.RAMP:
            lslope = level_slope_per_day(ep["bins"])
            slope = _f(ep.get("creep_slope"))
            if slope != slope:
                slope = _baseline_slope(store, sc.s, e)
            if slope != slope and ep.get("src") == "vec":
                slope = lslope
            ep["slope"] = _enc(slope)
            if slope == slope and abs(slope) > MG.RAMP_MAX_SLOPE:
                ramp_blocked = True
            elif lslope == lslope:
                detrend = lslope
                t, n, sm, _ = bins_arrays(ep["bins"])
                if t.size >= MK_MIN_POINTS:
                    tt, mm = _rebin(t, n, sm)
                    _, p = seq.mann_kendall(mm - detrend * (tt / DAY))
                    ramp_ok = bool(p > MK_P and slope == slope)
        # dispersion ratio
        vp = _f(ep.get("var_pre"))
        vpost, N = post_variance(ep["bins"], detrend)
        if vp == vp and vp > 0.0 and N >= DISP_MIN_N and vpost == vpost:
            ratio = vpost / vp
            ep["dispersion"] = _enc(ratio)
            if ratio <= DISPERSION_MAX:
                terms["dispersion"] = MG.LR_DISPERSION
        # time in regime
        step = MG.t_type(typ) / 3.0
        start = max(_f(ep["start"]), _f(ep["last_new_axis"]))
        steps = stationary_steps(ep["bins"], now, step, start, max(float(ep["bw"]), dt),
                                 detrend)
        te = MG.time_evidence(typ, steps)
        if te > 0.0:
            terms["time"] = te
        return typ, terms, ramp_ok, ramp_blocked, detrend

    def _evaluate(self, store: Any, sc: _Sys, e: str, model: Dict[str, Any],
                  ep: Dict[str, Any], ob: _Obs, now: float, dt: float) -> None:
        s = sc.s
        typ, terms, ramp_ok, ramp_blocked, _ = self._terms(store, sc, e, model, ep, ob, now, dt)
        x = MG.logodds(typ, terms, ramp_ok)
        p = MG.p_from_logodds(x)
        model["type"] = typ
        model["logodds"] = x
        model["p_legit"] = p
        model["evidence"] = {"prior": MG.prior(typ, ramp_ok), **terms}
        fl = ep["flags"]
        malicious = bool(fl.get("lib4_high") or fl.get("id_mismatch") or fl.get("beacon")
                         or fl.get("exfil_budget") or fl.get("sys_sensitive")
                         or fl.get("malicious_label"))
        negative = MG.terms_negative(terms)
        onset = _f(ep["onset"])
        in_regime = now - _f(ep["start"])            # time in regime (since SUSPECT)
        span = _f(ep.get("ev_last")) - _f(ep.get("ev_first"))
        corroborated = MG.reject_corroborated(fl, int(ep.get("ev_ticks") or 0), span)
        ep["corroborated"] = bool(corroborated)
        verdict = MG.decide(typ, x, in_regime, malicious, negative, ramp_blocked,
                            corroborated=corroborated)
        if verdict is None and is_class(e) and not malicious and not ramp_blocked:
            cs = _f(ep.get("concord_since"))
            if cs == cs and now - cs >= CLASS_ACCEPT_S:
                verdict = "class"
        if verdict is None and not is_class(e) and not malicious and not ramp_blocked:
            if self._class_accepted(store, sc, e, ep, now):
                verdict = "class_member"
        if verdict in ("accept", "class", "class_member"):
            extra: Dict[str, Any] = {}
            acc_cls = None
            if verdict == "class":
                acc_cls = {"ts": now, "onset": onset, "direction": int(ep.get("direction", 0)),
                           "members": self._members(store, sc, e)}
                model["class_accept"] = {"ts": now, "onset": onset,
                                         "direction": int(ep.get("direction", 0))}
            elif verdict == "class_member" or "peer" in terms:
                acc_cls = sc.class_of.get(e)
            extra["verdict"] = verdict
            self._accept(store, sc, e, model, now, onset, reason=verdict, extra=extra,
                         ramp_slope=_f(ep.get("slope")) if typ == MG.RAMP else _NAN,
                         accepted_class_change=acc_cls)
            return
        if verdict == "reject":
            self._reject(store, s, e, model, now, reason="logodds")
            return
        if model["regime"] == MG.DRIFTING:
            k = int(ep["ticks"]) - int(ep.get("drift_tick") or 0)
            if k > 0 and k % UPDATE_TICKS == 0:
                self._event(store, s, e, model, now, MG.DRIFTING,
                            extra={"update": k // UPDATE_TICKS})
            if in_regime >= LABEL_QUEUE_S and not model.get("label_queue"):
                model["label_queue"] = {"ts": now, "since": _f(ep["start"])}
                self._event(store, s, e, model, now, MG.DRIFTING, extra={"label_queue": True})

    # ------------------------------------------------------------ peers
    def _members(self, store: Any, sc: _Sys, ck: str) -> List[str]:
        mem = sc.members.get(ck)
        if mem is None:
            mem = sc.members[ck] = m_class.class_members(store, sc.s, ck)
        return mem

    def _class_key(self, store: Any, sc: _Sys, e: str) -> Optional[str]:
        if e not in sc.class_of:
            sc.class_of[e] = m_class.class_key(store, sc.s, e)
        return sc.class_of[e]

    def _concordance(self, store: Any, sc: _Sys, e: str, model: Dict[str, Any],
                     ep: Dict[str, Any], now: float) -> float:
        """Fraction of the class moving the same way within +-1 h of this onset."""
        if is_class(e):
            members = self._members(store, sc, e)
            me = None
        else:
            ck = self._class_key(store, sc, e)
            if ck is None:
                return 0.0
            members = self._members(store, sc, ck)
            me = e
        if not members:
            return 0.0
        onset = _f(ep["onset"])
        d0 = int(ep.get("direction", 0))
        votes: Dict[int, int] = {}
        for m in members:
            if m == me:
                continue
            rec = _member_move(store, sc.s, m, now)
            if rec is None:
                continue
            mo, md = rec
            if abs(mo - onset) <= PEER_WINDOW_S and md != 0:
                votes[md] = votes.get(md, 0) + 1
        if me is not None:
            if d0 == 0:
                return 0.0
            return (1 + votes.get(d0, 0)) / len(members)
        if not votes:
            return 0.0
        d, k = max(votes.items(), key=lambda kv: kv[1])
        if d0 == 0:
            ep["direction"] = d
        elif d != d0:
            k = votes.get(d0, 0)
        return k / len(members)

    def _class_accepted(self, store: Any, sc: _Sys, e: str, ep: Dict[str, Any],
                        now: float) -> bool:
        ck = self._class_key(store, sc, e)
        if ck is None:
            return False
        cm = MG.get(store, sc.s, ck)
        ca = (cm or {}).get("class_accept")
        if not isinstance(ca, Mapping) or now - _f(ca.get("ts")) > PEER_RECENT_S:
            return False
        return (abs(_f(ca.get("onset")) - _f(ep["onset"])) <= PEER_WINDOW_S
                and int(ca.get("direction", 0)) == int(ep.get("direction", 0)) != 0)

    # ------------------------------------------------------------ transitions
    def _quiet(self, ep: Dict[str, Any], ob: _Obs, now: float, dt: float) -> bool:
        """Every accumulator < h/4 and no alarm for max(8 ticks, 2 h); a tick
        on which fusion failed is never quiet."""
        if ob.degraded or ob.acc_max >= ACC_RETURN_LEVEL:
            return False
        last = _f(ep["last_alarm"])
        deg = _f(ep.get("degraded_ts"))
        if deg == deg and deg > last:
            last = deg
        return now - last >= max(QUIET_TICKS * dt, QUIET_S)

    def _return(self, store: Any, s: str, e: str, model: Dict[str, Any], now: float) -> None:
        ep = model["episode"]
        t0 = _f(ep["onset"])
        qf = _f(model.get("q_floor"))
        if qf == qf:
            t0 = min(t0, qf)
        self._control(store, s, e, model, now, release=[t0, now])
        model["q_floor"] = None
        self._close_incidents(store, s, e, now, "returned")
        _set_state(model, MG.RETURNED, now, reason="quiet")
        self._close_episode(model, now, MG.RETURNED)
        self._event(store, s, e, model, now, MG.RETURNED, extra={"release": [t0, now]})

    def _accept(self, store: Any, sc: _Sys, e: str, model: Dict[str, Any], now: float,
                tau: float, reason: str, extra: Optional[Dict[str, Any]] = None,
                ramp_slope: float = _NAN, accepted_class_change: Any = None) -> None:
        s = sc.s
        prev_v = max(MG.version(store, s, e), int(model.get("version", 0)))
        v = prev_v + 1
        tau = tau if tau == tau else _f(model["episode"]["onset"])
        # the accepted regime's permitted reference drift: its slope for a ramp,
        # none for a step (a new regime supersedes an earlier ramp's allowance)
        kw: Dict[str, Any] = {"version": v, "rebase_from": tau, "frozen": False,
                              "allow_drift": float(ramp_slope) if ramp_slope == ramp_slope
                              else 0.0}
        if accepted_class_change is not None:
            kw["accepted_class_change"] = accepted_class_change
        self._control(store, s, e, model, now, **kw)
        model["version"] = v
        model["q_floor"] = None
        self._close_incidents(store, s, e, now, "accepted")
        _set_state(model, MG.ACCEPTED, now, reason=reason)
        self._close_episode(model, now, MG.ACCEPTED)
        store.put_profile_version(s, e, v, self._version_snapshot(store, s, e, model, prev_v,
                                                                   tau, now), ts=now)
        self._event(store, s, e, model, now, MG.ACCEPTED,
                    extra={"rebase_from": tau, "version": v, "reason": reason, **(extra or {})})

    def _reject(self, store: Any, s: str, e: str, model: Dict[str, Any], now: float,
                reason: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self._control(store, s, e, model, now, frozen=True)
        _set_state(model, MG.REJECTED, now, reason=reason)
        self._event(store, s, e, model, now, MG.REJECTED, extra={"reason": reason,
                                                                  **(extra or {})})

    def _unfreeze(self, store: Any, s: str, e: str, model: Dict[str, Any], now: float) -> None:
        self._control(store, s, e, model, now, frozen=False)
        model["q_floor"] = None
        _set_state(model, MG.RETURNED, now, reason="unfrozen")
        self._close_episode(model, now, MG.REJECTED)
        self._event(store, s, e, model, now, MG.RETURNED, extra={"from": MG.REJECTED,
                                                                  "frozen": False})

    def _label_frozen(self, store: Any, s: str, e: str, ep: Mapping[str, Any]) -> bool:
        return bool(ep.get("label_frozen")) and m_feedback.is_frozen(store, s, e)

    def _want_rollback(self, store: Any, s: str, e: str, model: Dict[str, Any], now: float,
                       dt: float, frontier: float) -> None:
        """rollback_to = tau-hat - dt when tau-hat is before the commit frontier;
        at most 7 d back, at most once per hour (deferred, not dropped)."""
        ep = model["episode"]
        tau = _f(ep["onset"])
        if not tau < frontier:
            ep["pending_rollback"] = None
            return
        target = max(tau - dt, now - ROLLBACK_MAX_DEPTH_S + dt)
        # Warm-up rows are trusted by definition (ctx.training => trust = 1,
        # section 3): a rollback never reaches into them. Without this floor an
        # onset estimated at the start of the data (a chart that accumulated
        # while the model was still being learnt) erases the whole model.
        floor = _f(model.get("train_end"))
        if floor == floor and target < floor:
            target = floor
            if not target < frontier:
                ep["pending_rollback"] = None
                return
        done = _f(ep.get("rollback_to"))
        if done == done and done <= target:
            ep["pending_rollback"] = None
            return
        last = _f(model.get("last_rollback_ts"))
        if last == last and now - last < ROLLBACK_MIN_INTERVAL_S:
            ep["pending_rollback"] = target
            return
        self._control(store, s, e, model, now, rollback_to=target)
        ep["rollback_to"] = target
        ep["pending_rollback"] = None
        model["last_rollback_ts"] = now
        model["last_rollback_to"] = target
        model["rollbacks"] = int(model.get("rollbacks", 0)) + 1
        self._event(store, s, e, model, now, MG.ROLLBACK, extra={"rollback_to": target,
                                                                  "onset": tau})

    def _link_retractions(self, store: Any, s: str, e: str, model: Dict[str, Any],
                          links: Sequence[Mapping], now: float) -> None:
        """B17 retracted a continuity link A -> e: undo the seed. model.control
        {rollback_to: t_link, release: [t_link, now]} makes every learner
        restore its state before the seed (lib/gating barriers put the seed at
        t_link) and recommit e's own rows at once with their trust_prov
        (integration note R21.0; B17 test (c) shows equality with an unseeded
        twin). No quarantine is needed: nothing waits for a verdict. Done once
        per link; deferred (not dropped) while a rollback of this tick or the
        last hour is pending, because gating applies a release immediately but
        rate-limits the rollback, which would leave the rows after t_link held."""
        done = model.setdefault("retract_done", [])
        for lk in links:
            tau = _f(lk.get("rollback_to"))
            if tau != tau:
                tau = _f(lk.get("ts", lk.get("t_link")))
            key = f"{lk.get('from')}>{lk.get('to')}@{_enc(tau)}"
            if key in done or tau != tau:
                continue
            if now - tau > ROLLBACK_MAX_DEPTH_S:
                done.append(key)
                continue
            last = _f(model.get("last_rollback_ts"))
            ctl = store.get_model(s, e, MG.CONTROL)
            if (last == last and now - last < ROLLBACK_MIN_INTERVAL_S) or \
                    (isinstance(ctl, Mapping) and _f(ctl.get("ts")) == now):
                continue                                  # retried on a later tick
            self._control(store, s, e, model, now, rollback_to=tau, release=[tau, now])
            model["last_rollback_ts"] = now
            model["last_rollback_to"] = tau
            model["rollbacks"] = int(model.get("rollbacks", 0)) + 1
            done.append(key)
            del done[:-64]
            self._event(store, s, e, model, now, MG.ROLLBACK,
                        extra={"rollback_to": tau, "release": [tau, now],
                               "reason": "link_retracted", "from": lk.get("from")})
            return                                        # one directive per tick

    def _close_episode(self, model: Dict[str, Any], now: float, final: str) -> None:
        ep = model.get("episode")
        if isinstance(ep, Mapping):
            eps = model["episodes"]
            eps.append({"onset": _enc(ep.get("onset")), "start": _enc(ep.get("start")),
                        "end": now, "state": final, "type": model.get("type"),
                        "direction": int(ep.get("direction", 0))})
            del eps[:-EPISODES_CAP]
        model["episode"] = None
        model["label_queue"] = None

    # ------------------------------------------------------------ trust
    def _trust(self, model: Dict[str, Any], ob: _Obs, dt: float, sc: _Sys,
               e: str) -> Tuple[float, float]:
        if ob.degraded:
            return _NAN, _NAN
        prov = evidence_factor(ob.q_inst, ob.q_all, dt, getattr(ob, "e_inst", None))
        if ob.alarm is not None or ob.findings_med:
            prov = 0.0
        trust = prov
        live = [i for i in sc.incidents.get(e, []) if i.status in LIVE]
        if live or model["regime"] not in MG.TRUSTED_STATES or ob.acc_max >= ACC_TRUST_LEVEL:
            trust = 0.0
        return prov, trust

    def _trust_evidence(self, ob: _Obs) -> float:
        """behavior.trust_evidence (m_governor.evidence_weight), live ticks
        only: the live trust's gates on the row itself - [no alarm] x [no
        discrete finding >= MEDIUM] x [every accumulator < h/2] - without
        the regime / incident state and without the q_inst evidence factor.
        A normal live commit's trust already contains these; they bind only
        where a learner commits with trust_prov, i.e. a release, which would
        otherwise admit a row whose accumulators were at alarm level (B24,
        B25). Not written in training: gating warm-up rows on their own
        evidence truncates the null the rings estimate (measured, W7 tuning:
        pack A single-tick exceedance 5x -> 18-62x nominal)."""
        if ob.degraded:
            return _NAN
        if ob.alarm is not None or ob.findings_med or ob.acc_max >= ACC_TRUST_LEVEL:
            return 0.0
        return 1.0

    def _quarantine(self, model: Dict[str, Any], sc: _Sys, e: str) -> bool:
        live = any(i.status in LIVE for i in sc.incidents.get(e, []))
        return live or model["regime"] in MG.QUARANTINE_STATES

    def _orphan_release(self, store: Any, s: str, e: str, model: Dict[str, Any], q: bool,
                        now: float, frontier: float) -> None:
        """Quarantine bookkeeping. q_floor = the commit frontier on the first
        quarantined tick: learners committed every row up to it, and every
        row after it that they processed while quarantined is held, so a
        release from min(tau-hat, q_floor) leaves nothing stranded. A
        quarantine that ends without an episode (an incident closed while
        normal) releases [q_floor, t] here."""
        if q:
            if model.get("q_floor") is None:
                model["q_floor"] = frontier
            return
        qf = _f(model.get("q_floor"))
        if qf == qf and int(model.get("last_q", 0)) == 1:
            self._control(store, s, e, model, now, release=[qf, now])
        model["q_floor"] = None

    # ------------------------------------------------------------ writes
    def _control(self, store: Any, s: str, e: str, model: Dict[str, Any], now: float,
                 **kw: Any) -> None:
        old = store.get_model(s, e, MG.CONTROL)
        ctl = dict(old) if isinstance(old, Mapping) else {
            "version": int(model.get("version", 0)), "branch": 0, "rebase_from": None,
            "rollback_to": None, "release": None, "frozen": False, "allow_drift": 0.0,
            "accepted_class_change": None}
        ctl.update(kw)
        ctl["branch"] = int(model.get("branch", 0))
        ctl["ts"] = now
        store.put_model(s, e, MG.CONTROL, ctl, version=int(ctl.get("version") or 0), ts=now)

    def _close_incidents(self, store: Any, s: str, e: str, now: float, reason: str) -> None:
        for inc in store.incidents(system=s, entity=e, status=LIVE):
            if inc.entity != e:
                continue
            inc.status = "closed"
            inc.close_reason = reason
            inc.evidence.append({"ts": now, "source": "b28", "state": "closed",
                                 "close_reason": reason})
            store.put_incident(inc)

    def _event(self, store: Any, s: str, e: str, model: Dict[str, Any], now: float, state: str,
               extra: Optional[Dict[str, Any]] = None) -> str:
        ep = model.get("episode") if isinstance(model.get("episode"), Mapping) else {}
        onset = _f(model.get("onset"))
        p = _f(model.get("p_legit"))
        ex = {"state": state, "type": model.get("type"), "onset": _enc(onset),
              "p_legit": _enc(p), "logodds": _enc(model.get("logodds")),
              "evidence": dict(model.get("evidence") or {}),
              "version": int(model.get("version", 0)), "branch": int(model.get("branch", 0)),
              "axes": list(ep.get("axes") or [])}
        ex.update(extra or {})
        typ = model.get("type") or "unknown"
        desc = f"regime {state} ({typ})"
        if p == p:
            desc += f", P(legitimate) = {p:.2f}"
        return store.add_event(BehaviorEvent(
            system=s, entity=e, ts=now, kind=KIND, score=float(p) if p == p else 0.0,
            severity=_SEV_OF.get(state, Severity.INFO), description=desc, extra=ex,
            axes=list(ep.get("axes") or []),
            dedupe_key=f"regime|{s}|{e}|{int(model.get('branch', 0))}|{state}",
            model_version=int(model.get("version", 0)),
            window=(float(onset), float(now)) if onset == onset else None))

    def _write_regime(self, store: Any, s: str, e: str, model: Dict[str, Any], now: float,
                      dt: float, changed: bool) -> None:
        st = model["regime"]
        last = _f(model.get("regime_ts"))
        if not (changed or st != MG.NORMAL or last != last or now - last >= REGIME_HEARTBEAT_S
                or now < last):
            return
        model["regime_ts"] = now
        val = {"state": st, "type": model.get("type") if st != MG.NORMAL else None,
               "onset": _enc(model.get("onset")) if st != MG.NORMAL else None,
               "p_legit": _enc(model.get("p_legit")) if st != MG.NORMAL else None,
               "logodds": _enc(model.get("logodds")) if st != MG.NORMAL else None,
               "version": int(model.get("version", 0)), "branch": int(model.get("branch", 0)),
               "since": _enc(model.get("since"))}
        store.add_derived(DerivedMetric(name=MG.REGIME, value=val, ts=now, system=s, entity=e,
                                        window_s=int(dt), kind=MetricKind.CATEGORICAL))

    def _write_profile(self, store: Any, s: str, e: str, model: Dict[str, Any], now: float,
                       changed: bool) -> None:
        """profile.extra.regime on transitions, on the first tick and every 24
        ticks of an open episode (merged: B14 owns delta_by_feature)."""
        n = int(model.get("n", 0))
        if not (changed or (model["regime"] != MG.NORMAL and n % UPDATE_TICKS == 0)):
            return
        prof = store.profile(s, e)
        if prof is None:
            prof = EntityProfile(system=s, entity=e, updated=now)
        reg = prof.extra.get("regime")
        if not isinstance(reg, dict):
            reg = prof.extra["regime"] = {}
        normal = model["regime"] == MG.NORMAL
        reg.update({
            "version": int(model.get("version", 0)), "branch": int(model.get("branch", 0)),
            "state": model["regime"], "onset": None if normal else _enc(model.get("onset")),
            "type": None if normal else model.get("type"),
            "p_legit": None if normal else _enc(model.get("p_legit")),
            "history": [dict(h) for h in model["history"][-10:]],
            "rollbacks": {"count": int(model.get("rollbacks", 0)),
                          "last_ts": _enc(model.get("last_rollback_ts")),
                          "last_to": _enc(model.get("last_rollback_to"))},
            "label_queue": bool(model.get("label_queue")),
        })
        store.put_profile(prof)

    def _version_snapshot(self, store: Any, s: str, e: str, model: Dict[str, Any],
                          prev_v: int, tau: float, now: float) -> Dict[str, Any]:
        prof = store.profile(s, e)
        snap: Dict[str, Any] = {}
        if prof is not None:
            snap = {"fingerprint": list(prof.fingerprint),
                    "baseline_median": list(prof.baseline_median),
                    "baseline_mad": list(prof.baseline_mad),
                    "archetype": prof.archetype,
                    "extra": {k: prof.extra[k] for k in ("maturity", "model_state", "regime")
                              if k in prof.extra}}
        return {"version": int(model.get("version", 0)), "prev_version": prev_v, "ts": now,
                "rebase_from": _enc(tau), "type": model.get("type"),
                "p_legit": _enc(model.get("p_legit")),
                "evidence": dict(model.get("evidence") or {}), "profile": snap}


# ================================================================== helpers
def new_model(now: float) -> Dict[str, Any]:
    return {"fmt": 1, "regime": MG.NORMAL, "since": now, "onset": None, "type": None,
            "logodds": None, "p_legit": None, "evidence": {}, "history": [], "episodes": [],
            "episode": None, "version": 0, "branch": 0, "rollbacks": 0,
            "last_rollback_ts": None, "last_rollback_to": None, "label_queue": None,
            "class_accept": None, "base": new_base(None), "q_floor": None, "last_q": 0,
            "fb_version": None, "fb_mark": None, "regime_ts": None, "n": 0,
            "train_end": None}


def _set_state(model: Dict[str, Any], state: str, now: float, reason: str) -> None:
    model["regime"] = state
    model["since"] = now
    if state == MG.NORMAL:
        model["onset"] = None
        model["type"] = None
        model["logodds"] = None
        model["p_legit"] = None
        model["evidence"] = {}
    h = model["history"]
    h.append({"state": state, "ts": now, "onset": _enc(model.get("onset")),
              "type": model.get("type"), "p_legit": _enc(model.get("p_legit")),
              "reason": reason})
    del h[:-HISTORY_CAP]


def _latest1(store: Any, s: str, e: str, name: str, since: float) -> float:
    """spec v2.1: the newest value of a 1-element ring written since `since`."""
    ts, M = store.vec_since(s, e, name, since)
    if not len(ts):
        return _NAN
    v = float(M[-1, 0])
    return v if math.isfinite(v) else _NAN


def _latest_row(store: Any, s: str, e: str, name: str, since: float) -> Optional[np.ndarray]:
    ts, M = store.vec_since(s, e, name, since)
    return np.asarray(M[-1], dtype=np.float64) if len(ts) else None


def _latest_rows(store: Any, s: str, e: str, name: str, since: float) -> Optional[np.ndarray]:
    """Per column the newest finite value since `since` (NaN if none): an
    H-stream p holds until the next H tick while T-stream p are per tick."""
    ts, M = store.vec_since(s, e, name, since)
    if not len(ts):
        return None
    M = np.asarray(M, dtype=np.float64)
    out = np.full(M.shape[1], np.nan)
    for j in range(M.shape[0]):
        row = M[j]
        ok = np.isfinite(row)
        out[ok] = row[ok]
    return out


def _dict_at(store: Any, s: str, e: str, name: str, now: float) -> Mapping[str, Any]:
    """A dict series' point written at exactly now (the governor runs at now,
    so the newest point is the only candidate), else {}."""
    m = store.latest_derived(s, e, name)
    if m is not None and m.ts == now and isinstance(m.value, Mapping):
        return m.value
    return {}


def _vec1(store: Any, s: str, e: str, name: str, now: float) -> float:
    row = store.vec_at(s, e, name, now)
    if row is None or not len(row):
        return _NAN
    return float(row[0])


def _valid_onset(onset: float, now: float) -> bool:
    return onset == onset and now - ROLLBACK_MAX_DEPTH_S * 2 <= onset <= now


def _creep_slope(ev: BehaviorEvent) -> float:
    """Largest |slope_log| (log-units / day) of a baseline_creep event's groups."""
    best = _NAN
    groups = (ev.extra or {}).get("groups")
    items: List[float] = []
    if isinstance(groups, Mapping):
        on = {canonical_axis(a) for a in (ev.axes or ())}
        for g, v in groups.items():
            if isinstance(v, Mapping) and (not on or canonical_axis(g) in on):
                items.append(_f(v.get("slope_log")))
    for g, v in ev.contributors or ():
        items.append(_f(v))
    for v in items:
        if v == v and not abs(v) <= abs(best):
            best = v
    return best


def _baseline_slope(store: Any, s: str, e: str) -> float:
    """B03 slope diagnostics (log-units / day), safe when absent: model.baseline
    'slope_log' or 'slope_diag'.{'slope_log'} (not yet in contract C)."""
    mb = store.get_model(s, e, BASELINE)
    if not isinstance(mb, Mapping):
        return _NAN
    v = _f(mb.get("slope_log"))
    if v != v and isinstance(mb.get("slope_diag"), Mapping):
        v = _f(mb["slope_diag"].get("slope_log"))
    return v


def _idf_high(ex: Mapping[str, Any]) -> bool:
    """System-tier novelty with IDF > ln(N/2): a value new to the system has
    df = 0, so an unknown IDF counts as high."""
    idf = _f(ex.get("idf_system"))
    n = _f(ex.get("n_system"))
    if idf != idf or n != n or n < 2.0:
        return True
    return idf > math.log(n / 2.0)


def _acc_axes(store: Any, s: str, e: str, now: float, d: str) -> Set[str]:
    axd = _dict_at(store, s, e, emit.AXES, now)
    v = axd.get(d)
    if isinstance(v, str):
        v = [v]
    if v:
        return {canonical_axis(a) for a in v if a}
    return {canonical_axis(a) for a in DETECTOR_INFO[d]["axes"]}


def _member_move(store: Any, s: str, m: str, now: float) -> Optional[Tuple[float, int]]:
    """(onset, direction) of a member's open episode or of one closed within
    24 h; None when the member is not moving."""
    mm = MG.get(store, s, m)
    if not mm:
        return None
    ep = mm.get("episode")
    if isinstance(ep, Mapping) and mm.get("regime") in MG.OPEN_STATES:   # not a rejected one
        return _f(ep.get("onset")), int(ep.get("direction", 0))
    for c in reversed(mm.get("episodes") or []):
        if now - _f(c.get("end")) <= PEER_RECENT_S and c.get("state") == MG.ACCEPTED:
            return _f(c.get("onset")), int(c.get("direction", 0))
        break
    return None


