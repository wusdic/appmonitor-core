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
     Robust fit (round 4): calib.robust_tail - at most 2 % of the ring (6 of
     256 entries) beyond the 99 % outlier bound of n_u exponential excesses
     (scale from ranks) is trimmed from the fit AND from the counts, rate =
     (n_u - k) / (n - k). The earlier winsorisation (at 99.9 %, no bound on
     the count, outliers kept in n) left 1 % attack-level rows at 0.3x; -log10 of a
     valid p has an exponential tail, so a heavier fitted tail only comes
     from contaminating extremes (warm-up p_all of 1e-21 .. 1e-37 gave xi ~
     0.3 - 0.5 and saturated q_all near 5e-4, integration §8).
  3. Evidence CUSUM over instantaneous evidence only: S = max(0, S - ln q_inst - 3)
     (lib/seq), alarm while S >= h = (ln ARL - 3.07)/0.94 with ARL = 33 d in
     ticks (h = 5.31 at 900 s, 8.19 at 60 s). S is not reset on alarm (it
     keeps growing under a persistent attack, MEDIUM at S >= 2h, and decays
     at the null drift of -2 per tick afterwards); alarm episodes are counted
     at onsets (S crosses h), which realise the 33-day ARL (measured
     0.029-0.033 onsets per entity-day at 60/900/3600 s). A tick with no
     instantaneous evidence (q_inst NaN) leaves S unchanged and cannot alarm.
     Audit (round 4): h is derived from the entity's own calibrated
     per-stream null, not the nominal formula (which assumes iid uniform q).
     Once per hour per system AUDIT_KEYS keys (round robin) get solve_h: a
     moving-block bootstrap (1-h blocks, 1000 resampled days) of the q_inst
     history of their trusted PERIODS over the last 7 d (gating.period_
     trusted: governor tick present, not quarantined at the tick before -
     the old selection by trust >= 0.5 kept only q_inst >= ~3e-3 at 900 s,
     i.e. the audit selected its null stream by the statistic it audits),
     raw and with its tail above the 0.9 quantile redrawn from the fitted
     exponential tail; h* = the smallest h whose restarted-CUSUM alarm rate
     (ARL0's definition) is <= the target, max over the two paths, in
     [h_nominal, 2 h_nominal]. h_mult moves towards h* / h_nominal by at most
     +0.25 / -0.05 per audit. Simulated (tests/engines/test_b25_evidence_arl
     .py): uniform q 0.97x target (h* = nominal); -ln q 1.5x heavier 19.6x at
     the nominal h, median 1.3x at h*.
  4. Paths: single tick when e_day(q_all) <= 0.03 alpha_mult (feedback
     alpha_mult per system); evidence alarm; accumulator alarm (any
     behavior.acc_alarm) with the e_day of its own p.
     Adaptive conformal layer (round 4): the e_day of the single-tick path
     (and of the severity grading, behavior.e_day) is e_day_raw 10^theta.
     theta = system + key shift per ACI stratum (tick type h / q / t|cc; tick
     mode tick|cc), tracked at three levels e_day <= 3 / 0.3 / 0.03 by ACI
     (Gibbs & Candes 2021, quantile tracking on the log threshold: theta +=
     eta (err - alpha_l), alpha_l = the level's null rate per tick): the
     decision level directly, the shallow ones 10 - 100x faster and
     extrapolated to the decision depth by the power-law model theta_l
     proportional to -log10 alpha_l (aci_shift: max of the direct and the
     extrapolated shift). Guarantee: with a fixed step the long-run error
     frequency of any sequence is within (range + eta) / (eta T) of alpha_l.
     It must not hide attacks: the key tier tracks the decision level only
     (its errors are alarms, each opening an incident; shallow key levels
     learnt a persistent sub-alarm attack - pack A T7 - and extrapolated it),
     the shallow levels are pooled over the system (one key ~1/n_keys of the
     evidence); a tick is observed only live and when its
     PERIOD is trusted - the governor's previous tick exists and did not
     quarantine the key (the state before the tick: its own alarm opens its
     incident, so ticks inside an open incident or a suspect / drifting /
     rejected regime are never observed); per key, tick type and level at
     most one error counts per hour (decision level) or per 86400 / (4 l)
     s (shallow level l: ~1/4 of its null error rate per key-day), so a
     persistent sub-alarm run of one key among 35 moved the pooled shift by
     0.08 decades in a day, inside the null jitter (0.17); theta_sys in
     [-1, 3], theta_key in [-1, 2], total <= 3 decades. Simulated 35 entities at 900 s (tests/engines/
     test_b25_aci.py): q = U^1.5 (10.7x / 21x nominal on h / q ticks) ->
     0.8 / 1.3x; a calibrated null stays in [0.5, 2]. behavior.alarm carries
     e_day_raw and aci (decades).
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

Meta rings learn through lib/gating exactly as B24's rings (contract H,
round 4): row t is committed D = max(4 ticks, D_min_s) later with the trust
of its PERIOD (null_ring=True: gating.period_weight / release_weight), held
while quarantined, released / rebased / frozen by model.control. Not with
behavior.trust, whose evidence factor is q_inst itself, i.e. the rings'
own output: that was the selection feedback measured in integration §8.2
(a q_inst-based factor took pack A to 62x) and in round 4 (exact null: 3.7 -
6.3x at 1e-3 .. 1e-4). The W7 row-evidence cap on releases is removed for
the same reason; a released attack is trimmed by the robust tail. Below
900 s the per-tick-type strata ('t', and meta_inst_t) admit one row per
900-s slot (a rule on ts) so they span 64 h at every cadence, and while
they are young (< 64) their prior is the raw p made conservative by a
learned power, p^(1/v) (_t_prior; P(p_raw <= 0.05) = 0.05^(1/v), v0 = 2):
after a 900 -> 60 s switch they start empty. A
rollback deletes ring entries after the onset (the rings carry ts) and moves
the journal rows after it to held; a version change resets the rings. Row
values are kept 1 d (the same horizon B24 has for behavior.score), so rows
held longer than that cannot be released.

pipeline_degraded (contract M): when more than 30 % of an entity's families
in play are degraded, one system event (entity '__system__', extra.entity)
is emitted per entity per hour.

spec v2.1 streams (docs/lib3/cadence.md §7.3-§7.5; canonical grain mode).
Every detector belongs to a stream: h (scored on H rows at H decision
ticks), q (Q rows at Q decision ticks) or t (every tick). Each tick has a
type tau (h on an H decision tick, else q on a Q decision tick, else t);
p_all is the wHMP of whatever was scored (NaN = not scored, never
degraded), meta-calibrated in the stratum (daypart, tau[, cc for t]), and
    e_day = q_all n_tau / beta_tau          (lib/grains.e_day_tick)
so the single-tick budget stays 0.03 per entity-day at every cadence and
equals q 86400 / dt at 3600 s. There is one evidence CUSUM per stream: S_t
(behavior.evidence) over the T-stream instantaneous detectors every tick and
S_h (behavior.evidence.h) over the H-stream instantaneous detectors except
identity (whose windows overlap) on H ticks, each with half of the evidence
budget (ARL 66 d in its own periods). Q evidence enters no CUSUM (it is
nested in the hour's H evidence). Accumulator and family e_days count the
detector's own periods; the corroboration window is max(4 dt, 1 h) and
"consecutive" means consecutive decision ticks of the same type. A Q
detector with a provisional score (behavior.prov < 0.5) enters its family
at half weight and an alarm that needs it is capped at MEDIUM.

Store: reads behavior.p, behavior.axes, behavior.acc_alarm, behavior.degraded,
behavior.common.flag, behavior.calib_health@(s, __system__), feature.tctx,
behavior.trust / trust_prov / quarantine and model.control / model.link (via
gating: period trust), model.feedback (m_feedback), store.events, store.matches; writes
behavior.p_family (dict), behavior.q_inst / q_all / e_day / evidence
(1-element float32 vec rings), behavior.alarm (dict, alarm ticks only),
model.calib['meta'] (meta rings and B25 bookkeeping, incl. the key's ACI state),
model.calib@(s, __system__)['aci'] (the system ACI state; B24 keeps 'health' in the
same dict), pipeline_degraded events.
"""
from __future__ import annotations

import copy
import math
import weakref
import zlib
from bisect import bisect_left, bisect_right
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import BehaviorEvent, DerivedMetric, MetricKind, Severity
from .lib import calib, combine, emit, gating, m_calib, m_class, m_feedback, seq, timebins
from .lib import pactive as PA
from .lib import grains as GR
from .lib.classkeys import CLASS_PREFIX, SYSTEM_KEY
from .lib.detectors import (ACC_DETECTORS, DETECTOR_INDEX, DETECTOR_INFO, DETECTORS, FAMILIES,
                            FAMILY_IDX, INSTANT_FAMILIES, N_DETECTORS, Q_DETECTORS,
                            family_members)

MODEL = m_calib.MODEL
META = m_calib.META
P = emit.P
Q_INST = "behavior.q_inst"
Q_ALL = "behavior.q_all"
E_DAY = "behavior.e_day"
EVIDENCE = "behavior.evidence"
Q_INST_H = "behavior.q_inst.h"            # spec v2.1: H-stream instantaneous evidence
EVIDENCE_H = "behavior.evidence.h"
PROV = "behavior.prov"
META_INST_H, META_INST_T = "meta_inst_h", "meta_inst_t"
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

# evidence audit: h from the entity's calibrated per-stream null (round 4)
H_MULT_MAX = 2.0          # solved h <= 2 x nominal (a 25x excess needs ~1.6x at 900 s)
H_MULT_STEP_UP = 0.25     # per audit, towards the solved h
H_MULT_STEP_DOWN = 0.05
AUDIT_PERIOD_S = 3600.0
AUDIT_KEYS = 4            # keys audited per system and hour (round robin)
AUDIT_WINDOW_S = 7 * 86400.0
AUDIT_MIN_S = 2 * 86400.0
AUDIT_BOOT_DAYS = 1000.0
AUDIT_BLOCK_S = 3600.0
TAIL_SMOOTH_Q = 0.9        # solve_h: redraw resampled values above this quantile
AUDIT_TRUST_MIN = 0.5     # unused since round 4 (period trust); kept for old imports

# evidence CUSUM input calibration per key and stream (round 4; _qcal)
QCAL = "qcal"

# per-tick-type ('t') meta strata below 900 s (round 4)
THIN_SLOT_S = 900.0        # one admission per 900-s slot
XFER = "xfer"              # meta[STATE]: {'<kind>@<t stratum>': [k, n]} raw-p transfer counts
_T_TAG = "|t:t|"           # calib.meta_stratum_key of tick type t

# adaptive conformal inference on the single-tick decision (round 4; module
# docstring "Adaptive conformal layer")
ACI = "aci"
ACI_ETA_SYS = 0.05        # decades per counted error, system pool (per tick type)
ACI_ETA_KEY = 0.1         # decades per counted error, per key and tick type
ACI_TH_MIN = -1.0         # never more than 10x more permissive than the rings
ACI_TH_MAX_SYS = 3.0
ACI_TH_MAX_KEY = 2.0
ACI_TH_MAX = 3.0          # total shift <= 3 decades
ACI_ERR_GAP_S = 3600.0    # at most one counted error per key, tick type, level and hour
ACI_LEVELS = (3.0, 0.3, SINGLE_E_DAY)   # e_day levels tracked (multi-level ACI); last = decision
ACI_ETA_LEVEL = (0.3, 0.6, 1.0)         # step x per level: the shallow levels see 10 - 100x the
                                        # errors, and their jitter is amplified by the extrapolation

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
    "first_access_system",       # B21 cross_system (P2, round 4)
    "pattern_violation",         # P03 conformity (progressive.md §9.2)
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
_STREAM: List[str] = [str(DETECTOR_INFO[d]["stream"]) for d in DETECTORS]
# spec v2.1 evidence streams: instantaneous detectors of the T stream, and of
# the H stream except identity (overlapping windows: single-tick only)
_INST_T_IDX = [i for i, d in enumerate(DETECTORS) if _INST[i] and _STREAM[i] == "t"]
_INST_H_IDX = [i for i, d in enumerate(DETECTORS)
               if _INST[i] and _STREAM[i] == "h" and not DETECTOR_INFO[d]["overlap"]]
_Q_IDX = [DETECTOR_INDEX[d] for d in Q_DETECTORS]


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


# meta tail: winsorise exceedances beyond the (1 - alpha) bound of the max of
# n_u exponential excesses (calib.winsorise_exceedances). With the rank-based
# scale estimate the realised touch rate on a clean exponential tail is 1.0 %
# at n_u = 26 (a full M = 256 ring) and 3 % at n_u = 10 (2e4 simulated fits).
META_WINSOR_ALPHA = 0.001


def meta_tail(ring: calib.Ring, ts: float = _NAN) -> Optional[calib.GPDTail]:
    """GPD tail of a meta ring: calib.robust_tail (the predictive xi floor
    1/n_u fitted to the exceedances left after trimming at most 2 % of the
    ring as contamination; module docstring, "Tail shape")."""
    return calib.robust_tail(ring, now_ts=ts)


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
    S2 = seq.evidence_cusum_step(S, q, evidence_cap(h))
    return S2, True, S2 >= h


def evidence_cap(h: float) -> float:
    """Bound of an evidence CUSUM with threshold h (seq.EVIDENCE_CAP_MULT x h,
    round 4): after the evidence stops the statistic is below h within
    ~h / 2 ticks at the null drift -2, instead of S / 2 for an unbounded S;
    the MEDIUM level (S >= 2 h) is still reached."""
    return seq.EVIDENCE_CAP_MULT * h if h == h and h > 0 else math.inf


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
    cap = evidence_cap(h)
    for a in range(0, q.size, chunk):
        X = np.cumsum(x[a:a + chunk])
        m = np.minimum(np.minimum.accumulate(X), -s_last)
        S = X - m
        if math.isfinite(cap) and S.size and float(S.max()) > cap:
            # the bounded recursion has no closed form: fall back to the loop
            # (only chunks whose free path exceeds the cap, i.e. after an alarm)
            s = s_last
            xs = x[a:a + chunk]
            for j in range(xs.size):
                s = min(cap, max(0.0, s + xs[j]))
                S[j] = s
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


# ======================================================= adaptive conformal
def aci_key(tau: Optional[str], cc: int) -> str:
    """ACI stratum of a tick: its type ('h', 'q', 't|<cc>'; tick mode 'tick|<cc>')."""
    if tau is None:
        return f"tick|{int(cc)}"
    return f"t|{int(cc)}" if tau == "t" else str(tau)


def aci_step(theta: float, err: bool, alpha_star: float, eta: float, lo: float,
             hi: float) -> float:
    """One ACI update in decades of e_day (quantile tracking on the log
    threshold, Gibbs & Candes 2021 in score space): theta + eta (err - alpha*),
    clipped to [lo, hi]. With a fixed eta the long-run error frequency of
    any sequence satisfies |mean err - alpha*| <= (hi - lo + eta) / (eta T)."""
    th = float(theta) + float(eta) * ((1.0 if err else 0.0) - float(alpha_star))
    return lo if th < lo else hi if th > hi else th


def _levels(v: Any) -> List[float]:
    """Per-level thetas of one ACI key (a list aligned to ACI_LEVELS; a bare
    number from an older state is the decision level)."""
    n = len(ACI_LEVELS)
    if isinstance(v, (list, tuple)) and len(v) == n:
        out = [_f(x) for x in v]
        return [x if x == x else 0.0 for x in out]
    x = _f(v)
    return [0.0] * (n - 1) + [x if x == x else 0.0]


def aci_shift(sys_th: Optional[Mapping[str, Any]], key_th: Optional[Mapping[str, Any]],
              k: str, alpha_star: float = _NAN) -> float:
    """Total shift in decades applied to a tick's e_day (multi-level ACI).

    theta_l = system + key theta at each level l of ACI_LEVELS (e_day <= 3,
    0.3, 0.03). The decision level tracks its own rate directly but learns
    slowly (~alpha* per row); the shallow levels learn 10 - 100x faster. A
    power-law distortion of q (P(q <= x) = x^(1/a)) needs a shift
    proportional to the level's depth in decades, theta_l = (a - 1) d_l with
    d_l = -log10 alpha_l, so the shallow levels are extrapolated to the
    decision depth through the origin: theta = max(theta_0.03, max_l theta_l
    d_0.03 / d_l). Without alpha* (NaN) only the decision level is used.
    Clipped to [ACI_TH_MIN, ACI_TH_MAX]."""
    ts_ = _levels((sys_th or {}).get(k))
    tk_ = _levels((key_th or {}).get(k))
    th = [a + b for a, b in zip(ts_, tk_)]
    t = th[-1]
    a = _f(alpha_star)
    if a == a and 0.0 < a < 1.0:
        d_dec = -math.log10(a)
        for lvl, v in zip(ACI_LEVELS[:-1], th[:-1]):
            al = a * lvl / SINGLE_E_DAY
            if 0.0 < al < 1.0 and v > 0.0:
                t = max(t, v * d_dec / -math.log10(al))
    return ACI_TH_MIN if t < ACI_TH_MIN else ACI_TH_MAX if t > ACI_TH_MAX else t


def aci_update(sys_st: Dict[str, Any], key_st: Dict[str, Any], k: str, e_raw: float,
               alpha_star: float, ts: float) -> bool:
    """Fold one committed, period-trusted live row into every level of both
    ACI tiers. At level l (e_day threshold ACI_LEVELS[l], null rate alpha_l =
    alpha* l / 0.03) err_l = the row's raw e_day under that level's CURRENT
    total shift would have crossed it. Counted errors are rate-limited per
    key, tick type and level (ACI_ERR_GAP_S): an error within the gap is not
    counted and does not move that level either way. Returns the decision
    level's err as counted."""
    e = _f(e_raw)
    a = _f(alpha_star)
    if not (e == e and a == a and 0.0 < a < 1.0):
        return False
    sth = sys_st.setdefault("th", {})
    kth = key_st.setdefault("th", {})
    last = key_st.setdefault("last_err", {})
    s_l, k_l = _levels(sth.get(k)), _levels(kth.get(k))
    t_l = last.get(k)
    t_l = list(t_l) if isinstance(t_l, (list, tuple)) and len(t_l) == len(ACI_LEVELS) \
        else [None] * len(ACI_LEVELS)
    err_dec = False
    for i, lvl in enumerate(ACI_LEVELS):
        al = a * lvl / SINGLE_E_DAY
        if not 0.0 < al < 1.0:
            continue
        err = e * 10.0 ** (s_l[i] + k_l[i]) <= lvl
        if err:
            # per key: at most one error per level and gap; the gap of a
            # shallow level is ~ 1 / (4 x its null error rate per key-day,
            # >= lvl beta) so that one key's persistent sub-alarm run adds
            # little to the pooled count (the decision level: 1 h)
            gap = ACI_ERR_GAP_S if i == len(ACI_LEVELS) - 1 else \
                min(86400.0, max(ACI_ERR_GAP_S, 86400.0 / (4.0 * lvl)))
            tl = _f(t_l[i])
            if tl == tl and 0.0 <= ts - tl < gap:
                continue
            t_l[i] = float(ts)
        f = ACI_ETA_LEVEL[i]
        s_l[i] = aci_step(s_l[i], err, al, f * ACI_ETA_SYS, ACI_TH_MIN, ACI_TH_MAX_SYS)
        if i == len(ACI_LEVELS) - 1:
            # the key tier tracks the decision level only: its errors are
            # alarms, each of which opens an incident (the key is then not
            # observed); a shallow key level would learn a persistent
            # SUB-alarm attack of its own key and extrapolate it to the
            # decision depth (pack A T7). Shallow levels are pooled over the
            # system, where one key is ~1/n_keys of the evidence.
            k_l[i] = aci_step(k_l[i], err, al, f * ACI_ETA_KEY, ACI_TH_MIN, ACI_TH_MAX_KEY)
        if i == len(ACI_LEVELS) - 1:
            err_dec = err
    sth[k], kth[k], last[k] = s_l, k_l, t_l
    sys_st["n"] = int(sys_st.get("n", 0)) + 1
    sys_st["n_err"] = int(sys_st.get("n_err", 0)) + int(err_dec)
    return err_dec


def reset_alarms(x: np.ndarray, h: float, max_count: int, chunk: int = 4096) -> int:
    """Alarms of the CUSUM S = max(0, S + x_t) RESTARTED at 0 after each
    alarm (the run lengths whose mean is the ARL), counted up to max_count + 1
    (the caller only needs "more than max_count"). Vectorised per chunk:
    S_t = C_t - min(C_base, min C_s) with C the cumulative sum since the last
    restart."""
    C = np.cumsum(np.asarray(x, dtype=np.float64))
    n = C.size
    start, base, count = 0, 0.0, 0
    while start < n and count <= max_count:
        end = min(n, start + chunk)
        seg = C[start:end]
        m = np.minimum(np.minimum.accumulate(seg), base)
        hit = np.flatnonzero(seg - m >= h)
        if hit.size:
            tau = start + int(hit[0])
            count += 1
            base = float(C[tau])
            start = tau + 1
        else:
            base = float(m[-1])
            start = end
    return count


def _solve_h_path(x: np.ndarray, h_base: float, target: float, days: float,
                  mult_max: float, iters: int) -> Tuple[float, float]:
    """Bisection of reset_alarms on one bootstrap increment path x."""
    cap = int(math.floor(target * days))            # rate <= target  <=>  count <= cap
    n0 = reset_alarms(x, h_base, 10 * cap + 10)
    rate0 = n0 / float(days)
    if n0 <= cap:
        return float(h_base), rate0
    lo, hi = float(h_base), float(h_base) * float(mult_max)
    if reset_alarms(x, hi, cap) > cap:
        return hi, rate0
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        if reset_alarms(x, mid, cap) > cap:
            lo = mid
        else:
            hi = mid
    return hi, rate0


def solve_h(q: np.ndarray, h_base: float, dt_s: float, rng: np.random.Generator,
            target: float, days: float = AUDIT_BOOT_DAYS, block_s: float = AUDIT_BLOCK_S,
            mult_max: float = H_MULT_MAX, iters: int = 6) -> Tuple[float, float]:
    """The evidence threshold from the entity's own per-stream null (round 4).

    A moving-block bootstrap (blocks of max(4, block_s / dt) ticks, `days` of
    resampled stream) of the period-trusted q history gives increments
    x = -ln q - 3; the alarm rate of the CUSUM restarted after each alarm
    (ARL0's definition; reset_alarms) is a decreasing function of h, so the
    smallest h in [h_base, mult_max h_base] whose rate <= target is found by
    bisection (mult_max h_base when even that exceeds the target). Two paths
    from the same resampled positions:
      * raw: the history's own values - keeps the dependence of consecutive
        extremes exactly, but can only recycle the few extremes 7 days hold;
      * tail-smoothed: every resampled value above the history's 0.9
        quantile of -ln q redrawn from the exponential tail fitted to the
        history's exceedances (Lomax predictive: scale ~ InvGamma(n_u, n_u
        mean excess)) - extrapolates the marginal tail, loses the joint
        values of a cluster (not its positions).
    h* = max of the two. Measured (tests/engines/test_b25_evidence_arl.py,
    8 x 7-day histories, realised onset rate of the live no-reset CUSUM on
    3000 fresh days, x target): iid uniform 0.97 (h* = h_base in 7 of 8);
    q = U^1.25 / U^1.5 (-ln q 1.25 / 1.5x heavier) 5.7 / 19.6 at h_base,
    median 1.3 / 1.3 at h*. Strong serial dependence (AR(1) 0.7 of the
    probit, runs of 4 equal q) is only partly corrected (median 2.8 / 6.8x
    from 6.5 / 10.8x): 7 days hold few independent clusters. Returns (h*, rate at
    h_base per day, from the raw path); (NaN, NaN) with fewer than 4 blocks
    of finite q."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    q = q[np.isfinite(q)]
    b = max(4, int(round(block_s / dt_s)))
    if q.size < 4 * b:
        return _NAN, _NAN
    length = int(math.ceil(days * 86400.0 / dt_s))
    nb = int(math.ceil(length / b))
    starts = rng.integers(0, q.size - b + 1, size=nb)
    idx = (starts[:, None] + np.arange(b)[None, :]).reshape(-1)[:length]
    y = -np.log(np.clip(q, combine.P_FLOOR, 1.0))
    yb = y[idx]
    h_raw, rate0 = _solve_h_path(yb - seq.EVIDENCE_DRIFT, h_base, target, days, mult_max,
                                 iters)
    h_star = h_raw
    u = float(np.quantile(y, TAIL_SMOOTH_Q))
    exc = y[y > u] - u
    if exc.size >= 10 and h_raw < h_base * mult_max:
        n_u = exc.size
        sig = float(np.mean(exc))
        ys = yb.copy()
        tail = ys > u
        k = int(np.count_nonzero(tail))
        lam = rng.gamma(n_u, 1.0 / (n_u * sig), size=k)
        ys[tail] = u + rng.exponential(1.0, size=k) / lam
        h_sm, _ = _solve_h_path(ys - seq.EVIDENCE_DRIFT, h_base, target, days, mult_max, iters)
        h_star = max(h_raw, h_sm)
    return h_star, rate0


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
def _stream_p(row: Sequence[float], idx: Sequence[int], fw: Mapping[str, float],
              wm: Sequence[float]) -> float:
    """spec v2.1: the instantaneous p of one stream: fuse() restricted to the
    stream's instantaneous detectors (the others NaN)."""
    r = [_NAN] * N_DETECTORS
    any_ = False
    for i in idx:
        v = row[i]
        if 0.0 <= v <= 1.0:
            r[i] = v
            any_ = True
    if not any_:
        return _NAN
    return fuse(r, fw, wm).p_inst


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


POOL_REF = "pool_ref"     # meta[STATE]: bounded mode, the class pool's meta-ring holder (in memory)
POOL_META = "meta_pool"   # model.calib@(s, '__pool__...')[POOL_META] = {'rings': {...}, 'refit': {...}}


def _mr(meta: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(holder of the meta rings, its refit counters): the entity's own meta
    dict, or - bounded mode, unearned IP - its class pool's (progressive.md
    §10.3: 'unearned: class meta rings')."""
    pr = meta[STATE].get(POOL_REF)
    if isinstance(pr, dict):
        return pr["rings"], pr["refit"]
    return meta, meta[STATE]["refit"]


class _MRow(NamedTuple):
    s: str
    e: str
    ts: float
    s_inst: float
    s_all: float
    stratum: str
    grain: Optional[Tuple] = None     # spec v2.1: (s_inst_h, st_all, st_inst_t, st_inst_h)
    aci: Optional[Tuple] = None       # round 4: (e_day raw, ACI key, alpha*, training, dt)


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
    grain: Optional[Dict[str, Any]] = None    # spec v2.1 tick context (canonical mode)
    aci: Optional[Dict[str, Any]] = None      # round 4: the system's ACI state
    qcal: Optional[Dict[str, Any]] = None     # round 4: the system's CUSUM-input calibration


class _Rec:
    """Phase-1 result of one key, finalised after the system-wide BH."""
    __slots__ = ("e", "is_class", "st", "q_all", "e_day", "S", "S_prev", "h", "updated",
                 "ev_alarm", "fused", "sig", "fam_axes", "acc", "e_acc", "row", "gx",
                 "e_raw", "aci")

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
        self._aci_sys: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------- plumbing
    def _learner(self, d_min_s: Any) -> _MetaLearner:
        d = float(d_min_s) if d_min_s is not None else gating.D_MIN_S
        lr = self._learners.get(d)
        if lr is None:
            lr = self._learners[d] = _MetaLearner(
                name=LEARNER, init=new_meta, update=self._update, fetch=self._fetch,
                dump=m_calib.to_json, load=lambda blob: blob, merge=self._merge,
                on_rebase=self._on_rebase, d_min_s=d, ckpt_every_s=math.inf, clock=Q_ALL,
                null_ring=True)
        return lr

    def _ensure_retention(self, store) -> None:
        """behavior.e_day: at least 8 d like q_all (contract B; store default). Raise-only."""
        if self._ret_store is None or self._ret_store() is not store:
            store.ensure_retention(E_DAY, None, E_DAY_RETENTION_S)
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
        aci = None
        if isinstance(v[-1], (tuple, list)):         # round 4: ACI fields last
            aci, v = tuple(v[-1]), v[:-1]
        if len(v) > 3:                               # spec v2.1 row
            return _MRow(s, e, ts, v[0], v[1], v[2], (v[3], v[4], v[5], v[6]), aci)
        return _MRow(s, e, ts, v[0], v[1], v[2], None, aci)

    def _update(self, meta: Dict[str, Any], row: _MRow, w: float) -> Dict[str, Any]:
        w = float(w)
        w = 0.0 if not w > 0.0 else (1.0 if w > 1.0 else w)     # NaN -> 0
        # w is the PERIOD's trust (lib/gating.period_weight / release_weight,
        # null_ring=True): module docstring "Meta rings learn ..."
        if w >= 1.0:
            admit = True
        elif w > 0.0:
            admit = combine.seeded_uniform(row.s, row.e, ADMIT_SALT, float(row.ts)) < w
        else:
            admit = False
        if not admit:
            return meta
        st = meta[STATE]
        rings, refit = _mr(meta)
        if row.grain is not None:
            s_h, st_all, st_t, st_h = row.grain
            items = ((META_INST_T, row.s_inst, st_t), (META_ALL, row.s_all, st_all),
                     (META_INST_H, s_h, st_h))
        else:
            items = ((META_INST, row.s_inst, row.stratum), (META_ALL, row.s_all, row.stratum))
        dt = _f(row.aci[4]) if row.aci is not None and len(row.aci) > 4 else _NAN
        slot = int(round(row.ts)) % int(THIN_SLOT_S) if dt == dt else 0
        for kind, x, stratum in items:
            if not x == x or stratum is None:
                continue
            key = self._ring_key(kind, stratum)
            r = rings.get(key)
            if row.grain is not None and _T_TAG in stratum and dt == dt:
                # round 4: per-tick-type strata ('t', only at dt < 900): learn
                # the raw-p transfer while young (meta_xfer_v) and admit one
                # row per 900-s slot (a rule on ts), so the ring spans 64 h at
                # every cadence (lib/calib, calibration.py THIN_SLOT_S)
                if r is None or len(r) < m_calib.XFER_LEARN_N:
                    xs = st.setdefault(XFER, {})
                    kk = xs.get(key)
                    kk = [0.0, 0.0] if not isinstance(kk, list) or len(kk) != 2 else kk
                    kk[0] = float(kk[0]) + (1.0 if 10.0 ** (-x) <= m_calib.XFER_X else 0.0)
                    kk[1] = float(kk[1]) + 1.0
                    xs[key] = kk
                if dt < THIN_SLOT_S:
                    lo = int(round(dt)) if kind == META_ALL else 0
                    if not lo <= slot < lo + int(round(dt)):
                        continue
            if r is None:
                r = rings[key] = calib.Ring()
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
        if isinstance(own[STATE].get(POOL_REF), dict):
            return own                       # bounded mode: a pooled IP has no meta rings of its own
        refit = own[STATE]["refit"]
        for key, ro in meta_rings(other).items():
            nr = calib.seed_ring(own.get(key) or calib.Ring(), ro, frac=w)
            if len(nr):
                nr.gpd = meta_tail(nr, float(nr.ts.max()))
            own[key] = nr
            refit[key] = 0
        return own

    def _bounded_meta(self, store, s: str, e: str, meta: Dict[str, Any], now: float,
                      config: Optional[Mapping[str, Any]]) -> None:
        """lib3.resource_mode = bounded with lib3.pool_unearned (opt-in, off by
        default: pactive.pooled; progressive.md §10.3): an UNEARNED IP's
        meta rings are its class pool's (the pseudo entity pactive.pool_of,
        shared with B24's pooled rings); its CUSUM, ACI and gate bookkeeping
        stay its own (small) and are released by P15 after 7 idle days. A
        promoted IP starts its own meta rings from a copy of the pool; a
        rollback of a pooled IP leaves the pool's rings (its quarantined
        periods were never admitted: period trust). Full mode: no effect."""
        st = meta[STATE]
        if PA.pooled(store, s, e, config):
            ck = m_class.class_key(store, s, e)
            pk = PA.pool_of(store, s, e, ck)
            pm = store.get_model(s, pk, MODEL)
            if not isinstance(pm, dict):
                pm = {"layout": m_calib.LAYOUT, "rings": {}, "refit": {}, "pool": True}
                store.put_model(s, pk, MODEL, pm, ts=now)
            holder = pm.get(POOL_META)
            if not isinstance(holder, dict):
                holder = pm[POOL_META] = {"rings": {}, "refit": {}}
            if st.get(POOL_REF) is not holder:
                for k in list(meta_rings(meta)):     # demotion: own meta rings are dropped
                    del meta[k]
                st[POOL_REF] = holder
        elif isinstance(st.get(POOL_REF), dict):
            pr = st.pop(POOL_REF)
            for k, r in pr["rings"].items():           # promotion: warm start from the pool
                meta[k] = copy.copy(r)
            st["refit"] = dict(pr["refit"])

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
        gctx = None
        if GR.canonical(config):
            tau = GR.tick_type(now, dt, GR.CANONICAL)
            dp_h = GR.row_tctx(now, "h", dt, config)["daypart"]
            dp_q = (GR.row_tctx(now, "q", dt, config)["daypart"]
                    if GR.observable("q", dt, GR.CANONICAL) else dp_h)
            gctx = {
                "tau": tau, "cc": cc, "dp_t": daypart, "dp_h": dp_h, "dp_q": dp_q,
                "h_due": GR.decision(now, dt, "h", GR.CANONICAL),
                "mult": GR.e_day_mult(tau, dt, GR.CANONICAL),
                "h_t": seq.h_evidence(GR.evidence_arl_ticks("t", dt, GR.CANONICAL)),
                "h_h": seq.h_evidence(GR.evidence_arl_ticks("h", dt, GR.CANONICAL)),
                "per": [GR.period_s(d, dt, GR.CANONICAL) for d in DETECTORS],
                "corr_s": max(CORR_TICKS * dt, GR.GRAIN_S["h"]),
            }
            h_base = gctx["h_t"]
        n_out = 0
        for s in store.systems():
            keys = PA.entities(store, s, now, ctx.config) + [k for k in store.pseudo_entities(s)
                                                             if k.startswith(CLASS_PREFIX)]
            if not keys:
                continue
            ch = store.latest_derived(s, SYSTEM_KEY, CALIB_HEALTH)
            chv = ch.value if ch is not None and isinstance(ch.value, Mapping) else None
            wm = [m_calib.weight_mult(chv, d) for d in DETECTORS] if chv else _ONES
            aci_st = self._system_aci(store, s, now)
            sc = _SysCtx(s, fw, m_feedback.alpha_mult(store, s), wm, h_base, daypart, b24_failed,
                         gctx, aci_st, aci_st.setdefault(QCAL, {}))
            self._aci_sys = sc.aci
            recs: List[_Rec] = []
            for e in keys:
                rec = self._phase1(ctx, store, sc, e, now, dt, cc, learner)
                if rec is not None:
                    recs.append(rec)
            if recs:
                bh = bh_threshold(r.q_all for r in recs)
                for rec in recs:
                    n_out += self._finalise(ctx, store, sc, rec, now, dt, bh)
            self._audit(store, s, keys, now, dt, h_base, gctx)
        self._aci_sys = None
        return n_out

    @staticmethod
    def _system_aci(store, s: str, now: float) -> Dict[str, Any]:
        """The system-level ACI state in model.calib@(s, __system__)['aci']
        (B24 keeps its health state in the same dict; each touches its own key)."""
        m = store.get_model(s, SYSTEM_KEY, MODEL)
        if not isinstance(m, dict):
            m = {"layout": m_calib.LAYOUT}
            store.put_model(s, SYSTEM_KEY, MODEL, m, ts=now)
        a = m.get(ACI)
        if not isinstance(a, dict):
            a = m[ACI] = {"th": {}, "n": 0, "n_err": 0}
        return a

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
        self._bounded_meta(store, s, e, meta, now, ctx.config)
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
        if sc.grain is not None:
            return self._phase1_grain(ctx, store, sc, e, now, dt, st, meta, pend, row,
                                      S_prev, h, win)
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
        mrings = _mr(meta)[0]
        q_inst = meta_q(mrings.get(self._ring_key(META_INST, stratum)), s_inst,
                        m_calib.uniform(s, e, META_INST, now), fz.p_inst)
        q_all = meta_q(mrings.get(self._ring_key(META_ALL, stratum)), s_all,
                       m_calib.uniform(s, e, META_ALL, now), fz.p_all)
        e_raw = combine.e_day(q_all, dt)
        akey = aci_key(None, cc)
        theta = aci_shift((sc.aci or {}).get("th"), (st.get(ACI) or {}).get("th"), akey,
                          SINGLE_E_DAY * dt / 86400.0)
        e_day = e_raw * 10.0 ** theta if e_raw == e_raw else _NAN
        self._aci_observe(ctx, store, sc, e, st, now, dt, akey, e_raw,
                          SINGLE_E_DAY * dt / 86400.0)
        if s_inst == s_inst or s_all == s_all:
            pend[now] = (s_inst, s_all, stratum,
                         (e_raw, akey, SINGLE_E_DAY * dt / 86400.0, bool(ctx.training), dt))
        # --- 3) evidence CUSUM
        obs = self._qcal_observable(ctx, store, s, e, now, dt)
        S, updated, ev_alarm = evidence_update(
            S_prev, self._qcal(sc.qcal, f"inst|{cc}", q_inst, obs, now), h)
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
                    sig=sig, fam_axes=None, acc=acc, e_acc=e_acc, row=row, e_raw=e_raw,
                    aci=theta)

    # ------------------------------------------------------ spec v2.1 streams
    def _phase1_grain(self, ctx: Context, store, sc: _SysCtx, e: str, now: float, dt: float,
                      st: Dict[str, Any], meta: Dict[str, Any], pend: Dict[float, Any],
                      row: Optional[np.ndarray], S_prev: float, h: float,
                      win: int) -> Optional[_Rec]:
        """Canonical mode: tick types, per-stream evidence CUSUMs (module docstring)."""
        s, gx = sc.s, sc.grain
        tau = gx["tau"]
        Sh_prev = _f(st.get("S_h"))
        Sh_prev = Sh_prev if Sh_prev == Sh_prev else 0.0
        if st.get("warm_h") and not ctx.training:
            # warm-up evidence ends at the first live tick; stored at once, since
            # S_h is otherwise written on H ticks only and the first live tick
            # need not be one (the warm-up S_h would then resume on the next H tick)
            Sh_prev = 0.0
            st["S_h"] = 0.0
        st["warm_h"] = bool(ctx.training)
        h_h = gx["h_h"] * float(st.get("h_mult_h", 1.0))
        if row is None:                              # fused before, unscored now
            nan1 = [_NAN]
            store.add_vec(s, e, Q_INST, now, nan1, window_s=win)
            store.add_vec(s, e, Q_ALL, now, nan1, window_s=win)
            store.add_vec(s, e, E_DAY, now, nan1, window_s=win)
            store.add_vec(s, e, EVIDENCE, now, [S_prev], window_s=win)
            if gx["h_due"]:
                store.add_vec(s, e, Q_INST_H, now, nan1, window_s=win)
                store.add_vec(s, e, EVIDENCE_H, now, [Sh_prev], window_s=win)
            return None
        rowl = row.tolist() if isinstance(row, np.ndarray) else list(row)
        # provisional Q scores: half weight in their family (cadence.md §7.5)
        prov = emit.read_dict(store, s, e, PROV, now) if tau != "t" else {}
        prov_q = [DETECTOR_INDEX[d] for d in Q_DETECTORS
                  if 0.0 <= _f(rowl[DETECTOR_INDEX[d]]) <= 1.0 and _f(prov.get(d)) < 0.5]
        wm = list(sc.wm)
        for i in prov_q:
            wm[i] = wm[i] * 0.5
        fz = fuse(row, sc.fw, wm)
        degraded = self._degraded_families(store, s, e, now, fz)
        dp_t = self._daypart(store, s, e, now, sc.daypart)
        dp_tau = gx["dp_h"] if tau == "h" else gx["dp_q"] if tau == "q" else dp_t
        st_all = calib.meta_stratum_key(dp_tau, tau, gx["cc"] if tau == "t" else None)
        st_t = calib.meta_stratum_key(dp_t, "t", gx["cc"])
        st_h = calib.meta_stratum_key(gx["dp_h"], "h")
        # stream-wise instantaneous evidence
        p_t = _stream_p(rowl, _INST_T_IDX, sc.fw, wm)
        p_h = _stream_p(rowl, _INST_H_IDX, sc.fw, wm) if gx["h_due"] else _NAN
        s_t, s_h, s_all = meta_score(p_t), meta_score(p_h), meta_score(fz.p_all)
        mrings = _mr(meta)[0]
        r_t = mrings.get(self._ring_key(META_INST_T, st_t))
        q_t = meta_q(r_t, s_t, m_calib.uniform(s, e, META_INST, now),
                     self._t_prior(st, META_INST_T, st_t, r_t, p_t))
        q_h = meta_q(mrings.get(self._ring_key(META_INST_H, st_h)), s_h,
                     m_calib.uniform(s, e, META_INST_H, now), p_h)
        r_all = mrings.get(self._ring_key(META_ALL, st_all))
        q_all = meta_q(r_all, s_all, m_calib.uniform(s, e, META_ALL, now),
                       self._t_prior(st, META_ALL, st_all, r_all, fz.p_all))
        e_raw = q_all * gx["mult"] if q_all == q_all else _NAN
        akey = aci_key(tau, gx["cc"])
        theta = aci_shift((sc.aci or {}).get("th"), (st.get(ACI) or {}).get("th"), akey,
                          SINGLE_E_DAY / gx["mult"])
        e_day = e_raw * 10.0 ** theta if e_raw == e_raw else _NAN
        self._aci_observe(ctx, store, sc, e, st, now, dt, akey, e_raw, SINGLE_E_DAY / gx["mult"])
        if s_t == s_t or s_all == s_all or s_h == s_h:
            pend[now] = (s_t, s_all, st_all, s_h, st_all, st_t, st_h,
                         (e_raw, akey, SINGLE_E_DAY / gx["mult"], bool(ctx.training), dt))
        # evidence CUSUMs: S_t every tick, S_h on H ticks
        obs = self._qcal_observable(ctx, store, s, e, now, dt)
        S, upd_t, al_t = evidence_update(S_prev, self._qcal(sc.qcal, f"t|{gx['cc']}", q_t, obs, now),
                                         h)
        st["S"] = S
        Sh, upd_h, al_h = Sh_prev, False, False
        if gx["h_due"]:
            Sh, upd_h, al_h = evidence_update(Sh_prev, self._qcal(sc.qcal, "h", q_h, obs, now), h_h)
            st["S_h"] = Sh
        store.add_vec(s, e, Q_INST, now, [m_calib.issued(q_t)], window_s=win)
        store.add_vec(s, e, Q_ALL, now, [m_calib.issued(q_all)], window_s=win)
        store.add_vec(s, e, E_DAY, now, [e_day], window_s=win)
        store.add_vec(s, e, EVIDENCE, now, [S], window_s=win)
        if gx["h_due"]:
            store.add_vec(s, e, Q_INST_H, now, [m_calib.issued(q_h)], window_s=win)
            store.add_vec(s, e, EVIDENCE_H, now, [Sh], window_s=win)
        pf_out: Dict[str, float] = dict(fz.p_family)
        for f in degraded:
            pf_out[f] = _NAN
        if pf_out:
            store.add_derived(DerivedMetric(name=P_FAMILY, value=pf_out, ts=now, system=s,
                                            entity=e, window_s=win, kind=MetricKind.CATEGORICAL))
        if degraded:
            self._maybe_degraded_event(store, s, e, now, st, fz, degraded)
        # significant families on their own periods' e_day scale
        per = gx["per"]
        sig: List[str] = []
        for f, p in fz.p_family.items():
            mem = [per[i] for i in FAMILY_IDX[f] if 0.0 <= _f(rowl[i]) <= 1.0]
            if mem and p * 86400.0 / min(mem) <= AXIS_E_DAY:
                sig.append(f)
        acc: List[Tuple[str, float]] = []
        aa = emit.read_dict(store, s, e, emit.ACC_ALARM, now)
        e_acc = _NAN
        if aa:
            for d, v in aa.items():
                if d in _ACC_SET and _f(v) >= 0.5:
                    pv = _f(rowl[DETECTOR_INDEX[d]])
                    acc.append((d, pv if 0.0 <= pv <= 1.0 else _NAN))
            fin = [p * 86400.0 / per[DETECTOR_INDEX[d]] for d, p in acc if p == p]
            if fin:
                e_acc = min(fin)
        # which stream's evidence alarm (S_h is judged on H ticks only)
        ev_alarm = al_t or al_h
        S_rep, S_prev_rep, h_rep, stream = S, S_prev, h, "t"
        if al_h and (not al_t or Sh / h_h > S / h):
            S_rep, S_prev_rep, h_rep, stream = Sh, Sh_prev, h_h, "h"
        # an alarm that needs provisional Q evidence is capped at MEDIUM
        prov_cap = False
        if prov_q and tau == "q":
            r2 = list(rowl)
            for i in prov_q:
                r2[i] = _NAN
            fz2 = fuse(r2, sc.fw, sc.wm)
            prov_cap = not (fz2.p_all == fz2.p_all and fz2.p_all * gx["mult"] <= SINGLE_E_DAY)
        return _Rec(e=e, is_class=e.startswith(CLASS_PREFIX), st=st, q_all=q_all, e_day=e_day,
                    S=S_rep, S_prev=S_prev_rep, h=h_rep, updated=upd_t or upd_h,
                    ev_alarm=ev_alarm, fused=fz, sig=sig, fam_axes=None, acc=acc, e_acc=e_acc,
                    row=row, gx={"tau": tau, "stream": stream, "prov_cap": prov_cap,
                                 "per": per, "corr_s": gx["corr_s"]}, e_raw=e_raw, aci=theta)

    @staticmethod
    def _qcal_observable(ctx: Context, store, s: str, e: str, now: float, dt: float) -> bool:
        """A live tick of a trusted period (the governor's previous tick
        present and not quarantining the key), as for ACI."""
        if ctx.training:
            return False
        tr = store.vec_at(s, e, gating.TRUST, now - dt)
        if tr is None or not len(tr) or not math.isfinite(float(tr[0])):
            return False
        return not gating.is_quarantined(store, s, e, now, dt)

    @staticmethod
    def _qcal(qc: Optional[Dict[str, Any]], stream: str, q: float, observe: bool,
              now: float) -> float:
        """Round 4: the evidence CUSUM input on the SYSTEM's calibrated live
        null. q_inst is meta-calibrated per key by a ring that lags a live
        shift (pack A seed 0: one control key's q_inst.h at <= 1e-3 on 3 % of
        its live H rows, 31x). Per system, stream and cadence the live share
        of q <= 0.05 on ticks of trusted periods (decayed, half-life 3 d)
        gives the power v of P(q <= x) = x^(1/v) (m_calib.pcal_v: v = 1 until
        the excess is significant, never below 1), and the CUSUM takes
        q^(1/v). Pooled over the system's keys on purpose: a key-level
        version learnt a persistent SUB-alarm attack of its own key within
        hours (pack A T7, one rare-resource access per tick for a day: the
        key's share of q <= 0.05 rose, v followed, the CUSUM never alarmed),
        where one key of a system moves the pooled share by ~1/n_keys; ticks
        inside an open incident / suspect regime are not observed at all.
        The audit's solved h covers each key's own excess and dependence."""
        if not q == q or qc is None:
            return q
        stats = qc.get(stream)
        out = m_calib.pcal_apply(q, m_calib.pcal_v(stats))
        if observe:
            qc[stream] = m_calib.pcal_observe(stats, q <= m_calib.PCAL_X, now)
        return out

    @staticmethod
    def _aci_observe(ctx: Context, store, sc: _SysCtx, e: str, st: Dict[str, Any], now: float,
                     dt: float, akey: str, e_raw: float, alpha_star: float) -> None:
        """Round 4: one ACI observation, at scoring time, of a live tick whose
        PERIOD is trusted: the governor ran at t - dt (finite behavior.trust)
        and the key was not quarantined then (gating.is_quarantined: the state
        BEFORE this tick). The tick's own alarm opens its incident and
        quarantines the key from t on, so a rule on quarantine at t (or on the
        row's trust) would never see an error - the selection bias of the
        rings again. Ticks inside an open incident or a suspect / drifting /
        rejected regime (an attack under way) are not observed; together
        with the per-hour rate limit an attack contributes at most one error
        per level and hour before its incident opens."""
        if ctx.training or sc.aci is None or not e_raw == e_raw:
            return
        tr = store.vec_at(sc.s, e, gating.TRUST, now - dt)
        if tr is None or not len(tr) or not math.isfinite(float(tr[0])):
            return
        if gating.is_quarantined(store, sc.s, e, now, dt):
            return
        aci_update(sc.aci, st.setdefault(ACI, {}), akey, e_raw, alpha_star, now)

    def _t_prior(self, st: Mapping[str, Any], kind: str, stratum: str,
                 ring: Optional[calib.Ring], p_raw: float) -> float:
        """Small-sample prior of a meta ring: the raw wHMP p, and for a
        per-tick-type ('t') stratum while it is young (< SMALL_N) the raw p
        made conservative by the learned power, p^(1/v) (m_calib.xfer_v over
        the stratum's own committed rows: P(p_raw <= 0.05) = 0.05^(1/v);
        prior v = 2). After a 900 -> 60 s switch these strata start empty and
        the raw HMP of freshly transferred B24 p (calibration.py) is their only
        evidence; round 4, integration §10.7 item 2."""
        if _T_TAG not in stratum or not p_raw == p_raw:
            return p_raw
        if ring is not None and len(ring) >= calib.SMALL_N:
            return p_raw
        v = m_calib.xfer_v((st.get(XFER) or {}).get(self._ring_key(kind, stratum)))
        return min(1.0, max(0.0, p_raw)) ** (1.0 / v)

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
        per = rec.gx["per"] if rec.gx is not None else None
        thr = AXIS_E_DAY * dt / 86400.0
        out: Dict[str, Set[str]] = {}
        for f in fams:
            members = [(vals[i], i) for i in FAMILY_IDX[f] if 0.0 <= vals[i] <= 1.0]
            if not members:
                continue
            if per is not None:                  # spec v2.1: each member's own period
                drivers = [i for p, i in members if p * 86400.0 / per[i] <= AXIS_E_DAY] \
                    or [min(members)[1]]
            else:
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
        gx = rec.gx
        if gx is None:
            hist = [h for h in (st.get("hist") or []) if now - CORR_TICKS * dt < _f(h[0]) < now]
            prev = hist[-1] if hist and _f(hist[-1][0]) >= now - 1.5 * dt else None
            st["hist"] = (hist + [[now, e_path, sorted(sig_axes)]])[-CORR_TICKS:]
        else:
            # spec v2.1: wall-clock window max(4 dt, 1 h); "consecutive" = the
            # previous decision tick of the same type
            win_s = gx["corr_s"]
            hist = [h for h in (st.get("hist") or []) if now - win_s < _f(h[0]) < now]
            tau = gx["tau"]
            per_tau = GR.GRAIN_S.get(tau, dt) if tau != "t" else dt
            same = [h for h in hist if (h[3] if len(h) > 3 else "t") == tau]
            prev = same[-1] if same and _f(same[-1][0]) >= now - 1.5 * max(per_tau, dt) else None
            keep = max(CORR_TICKS, int(math.ceil(win_s / dt)) + 1)
            st["hist"] = (hist + [[now, e_path, sorted(sig_axes), tau]])[-keep:]
        if not paths:
            return 0

        # --- 5) corroboration limit (store lookups only when a level above
        # MEDIUM is at stake: a HIGH+ candidate or a possible self/peer raise)
        axes4 = set(sig_axes)
        for h in (hist[-(CORR_TICKS - 1):] if gx is None else hist):
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
        if gx is not None and gx.get("prov_cap") and PATH_SINGLE in paths \
                and _RANK[allowed] > _RANK["medium"]:
            allowed = "medium"                 # provisional Q evidence: never HIGH
            rules.append("provisional_q")
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
            if gx is not None and gx["tau"] != "h":
                flags = dict(flags or {})
                for g, v in (emit.read_dict(store, s, e, COMMON_FLAG + ".q", now) or {}).items():
                    flags[g] = max(_f(flags.get(g, 0.0)) if flags.get(g) is not None else 0.0,
                                   _f(v))
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
            "e_day_raw": rec.e_raw,          # before the ACI shift (round 4)
            "aci": rec.aci,                  # decades: e_day = e_day_raw 10^aci
        }
        if gx is not None:
            alarm["tau"] = gx["tau"]
            if PATH_EVIDENCE in paths:
                alarm["stream"] = gx["stream"]
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
               h_base: float, gctx: Optional[Mapping[str, Any]] = None) -> None:
        """Hourly block-bootstrap audit of one key's evidence-alarm rate (round
        robin). spec v2.1 canonical mode: per stream, the T stream on its
        q_inst ticks against its own ARL (EVIDENCE_ARL_DAYS / share), and the
        H stream on behavior.q_inst.h (one row per hour) with h_h / h_mult_h."""
        if not self.entity_due(("fusion.audit", s), now, AUDIT_PERIOD_S):
            return
        cand = []
        for e in keys:
            m = store.get_model(s, e, MODEL)
            if isinstance(m, Mapping) and isinstance(m.get(META), Mapping):
                cand.append(e)
        if not cand:
            return
        k0 = int(now // AUDIT_PERIOD_S) * AUDIT_KEYS
        for e in dict.fromkeys(cand[(k0 + i) % len(cand)] for i in range(AUDIT_KEYS)):
            st = store.get_model(s, e, MODEL)[META].get(STATE)
            if not isinstance(st, dict):
                continue
            if gctx is None:
                self._audit_stream(store, s, e, st, now, Q_INST, dt, h_base, "h_mult", "audit",
                                   1.0 / EVIDENCE_ARL_DAYS)
                continue
            self._audit_stream(store, s, e, st, now, Q_INST, dt, gctx["h_t"], "h_mult",
                               "audit", 1.0 / GR.evidence_arl_days("t", GR.CANONICAL))
            per_h = GR.stream_period_s("h", dt, GR.CANONICAL)
            self._audit_stream(store, s, e, st, now, Q_INST_H, per_h, gctx["h_h"], "h_mult_h",
                               "audit_h", 1.0 / GR.evidence_arl_days("h", GR.CANONICAL))

    def _audit_stream(self, store, s: str, e: str, st: Dict[str, Any], now: float, name: str,
                      per_s: float, h_base: float, mult_key: str, rec_key: str,
                      target: float) -> None:
        ts, M = store.vec_since(s, e, name, now - AUDIT_WINDOW_S)
        if ts.size * per_s < AUDIT_MIN_S:
            return
        q = M[:, 0].astype(np.float64)
        # the null stream: ticks of trusted PERIODS (gating.period_trusted: a
        # governor tick, not quarantined), never selected by trust >= 0.5,
        # whose evidence factor is q_inst itself (the audit would only see
        # ticks with q_inst >= ~3e-3 at 900 s and never find an excess)
        q = q[gating.period_trusted(store, s, e, ts)]
        if q.size * per_s < AUDIT_MIN_S:
            return
        h_mult = float(st.get(mult_key, 1.0))
        rng = np.random.default_rng(zlib.crc32(f"{s}|{e}|{int(now)}".encode("utf-8")))
        h_star, rate = solve_h(q, h_base, per_s, rng, target)
        if not h_star == h_star:
            return
        goal = min(H_MULT_MAX, max(1.0, h_star / h_base))
        if goal > h_mult:
            h_mult = min(goal, h_mult + H_MULT_STEP_UP)
        else:
            h_mult = max(goal, h_mult - H_MULT_STEP_DOWN)
        st[mult_key] = h_mult
        st[rec_key] = {"ts": now, "rate": rate, "h_star": h_star, "h_mult": h_mult,
                       "n": int(q.size)}
