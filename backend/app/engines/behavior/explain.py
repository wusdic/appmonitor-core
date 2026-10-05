"""B29 ExplainEngine: a faithful, testable explanation for every incident that
opens or escalates (docs/lib3/engines.md B29).

Why a separate engine: B27 decides WHETHER a human is told; the analyst then
needs to know WHAT is unusual in units they can check (bytes, requests,
percentages against the normal range for this hour and day type), what is
NEW, and -- the part that makes an explanation testable -- which minimal set
of deviations, had it not happened, would have kept the incident from
opening. That counterfactual is only honest if it is recomputed through the
same statistics that raised the incident, so this engine never approximates
a detector it can recompute: stateless detectors are re-scored through their
owners' pure helpers against the stored models, stateful ones are REPLAYED
from their stored inputs (lib/replay), and the result goes through B24's p
rule on the model.calib snapshot, B25's weighted HMP, meta-calibration,
evidence CUSUM and decision paths. What it cannot recompute is held at the
factual value and listed in the scope, so validity is never overstated.

When: only on incidents that opened, reopened or escalated this tick (a new
incident, a severity rise or a new axis since the last explanation), never
in training (no incidents open there). Runs after B28 and before B30.

What (incident.explanation; the narrative also in incident.narrative):
 1) Numeric attribution at the incident's trigger tick t* (now, or the
    newest scored tick of the incident):
    * natural units: the observed value per 15 min next to the predictive
      p5 / p50 / p95 of this hour and day type from profile.extra.model_state
      (B04; recomputed with m_baseline.quantiles when model_state describes
      another bucket), e.g. 'bytes_up 38.2 MB/15 min; usual Tue 14:00
      0.4-1.1 MB (35x)';
    * RBC chi2_1 p-values (m_density.contributions_model on behavior.zi, the
      row B06 scored) and Garthwaite-Koch shares c_i = w_i^2 / D^2 with
      w = Sigma^(-1/2) (zi - mu) (symmetric inverse root, D^2 = T^2), on the
      entity's model.density; without a fitted density model, Sigma = I
      (shares z_i^2 / sum z^2, p = behavior.pf);
    * BH at q = 0.05 over the p-values (flag 'bh').
    Class keys: the class aggregate row (behavior.class.agg) against the
    class's own conjugate anchors (model.classagg, m_baseline.midp) and the
    class_monitor bands.
 2) Categorical: new tokens with their tier and first-seen time (the
    first_seen / rare_access findings of the episode, B08), vanished habitual
    values (share >= 2 % of a dimension, known >= 3 d, not seen for >= 6 h
    and not since the episode began; model.vocab), and the client-stack diff
    (client.stack_set of the episode against model.client: stacks new to the
    entity, dominant stacks missing).
 3) Sequence: the 3 least likely transitions of the trigger tick's act.stream
    under the entity's PPM with B10's backoff (m_seq.loglik).
 4) Counterfactual (faithful replay) of the OPENING decision: "had these
    deviations not happened, would the incident have opened?" -- evaluated
    at the tick the incident opened or reopened (B27's evidence mark; the
    spec's "stop when the incident's opening condition no longer holds"),
    so an escalation re-explains the incident but keeps asking the question
    about its opening, from retained data (behavior.p / score / pm 1 d,
    behavior.alarm 8 d, the 6-h zr / cusum_state rings). Candidates, most
    important first: the top attributed numeric features of the opening
    tick (|z| >= 1), then the new tokens of that tick, then the unusual
    off-hours activity. lib/replay.minimal_set adds them
    until the recomputed decision flips, then drops every member that is not
    needed (1-minimal). Neutralising a feature resets its natural value to
    the bucket median of its current predictive (so z, zr become the
    median's residual, ~0); a token is removed; off-hours slots become
    silent. Recomputed per tick of the replay window:
      marg_int / marg_shape / peer  B04's exact mid-p (m_baseline.midp) of the
                    changed features, the stored pf of the others, B04's
                    two-stage harmonic-mean channels over model.groups
                    (m_density.groups(split_by_feature_group=True));
      t2 / spe      m_density.score_model on zi shifted by the change of z;
      cusum / mcusum  REPLAY of B14's bank and Crosier MCUSUM from the
                    behavior.cusum_state ring (m_cp.replay_rows /
                    replay_step: the exact bank inputs after B14's own-
                    support mask, per-tick phi, adjacency, restarts and
                    latch resets found by m_cp.mark_resets) over the
                    accumulator window from the episode onset tau-hat
                    (m_cp.onset; <= 96 ticks, <= the 6-h ring); the latch is
                    the replayed crossing of h. The MCUSUM's whitened input
                    of every tick is recovered exactly from consecutive
                    recorded states (m_cp.mc_input: B14's whitener follows
                    model.density refits and is not stored per tick); a
                    counterfactual moves it by the change of the whitened
                    residual under the current whitener;
      offhours      m_rhythm.offhours_replay over the rhythm ledger;
      novelty       0 when every value new at the tick is removed, else held;
    then p = m_calib.p_replay (B24's full prior order) on the model.calib
    snapshot with B24's stratum of that tick (m_calib.issued_stratum) and
    seeded U; B25's fuse (weighted HMP with feedback family weights and
    calibration-health weight_mult), q from the meta rings
    (m_calib.p_value with the raw HMP p as the small-sample prior), e_day,
    the evidence CUSUM replayed over its current excursion (behavior.evidence
    from its last zero), the accumulator alarms, BH across the system's keys
    for a LOW single-tick alarm, and B27's opening conditions (discrete
    findings >= MEDIUM not removed; a risk opening is held). Held
    detectors keep their stored p. Reported: counterfactual_set (features by
    name, 'token:<dim>=<value>', 'offhours'), counterfactual_valid (the
    factual recompute reproduces a trigger AND the neutralised one has
    none), counterfactual_scope {recomputed, replayed, held, window,
    ticks, fidelity}; fidelity compares the factual recompute with the
    stored p / q / decision (max |log10 p| error, CUSUM replay error, resets).
 5) Peer context: behavior.common.<group> at the class key and __system__
    (B05), the entity's common-mode flags, the class members' current e_day
    (how many are also unusual) and the peer family p.
 6) Nearest labelled pattern: cosine between this incident's deviation
    vector (behavior.z at t*, 52 dims) and those of labelled incidents
    (explanation.deviation, else their evidence feature z).
 7) Deterministic zh / en templates: a headline and 3 bullets, always with
    the natural-unit range of the top feature.

Coupling: reads only the store and the accessor modules (m_baseline,
m_density, m_cp, m_calib, m_rhythm, m_vocab, m_client, m_seq, m_class,
m_feedback); imports no engine. Writes incident.explanation /
incident.narrative through store.put_incident. Learns nothing, so there is
nothing to gate or roll back.
"""
from __future__ import annotations

import dataclasses
import math
import time
import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import special as sp

from ...core.engine import Context, Engine
from .lib import bayes, calib, combine, emit, m_calib, m_class, m_client, m_cp, m_density
from .lib import grains as GR
from .lib import m_baseline as MB
from .lib import m_feedback, m_rhythm, m_seq, m_vocab, seq, timebins
from .lib import replay as RP
from .lib.classkeys import CLASS_PREFIX, SYSTEM_KEY, is_class
from .lib.detectors import (ACC_DETECTORS, DETECTOR_INDEX, DETECTOR_INFO, DETECTORS, FAMILIES,
                            N_DETECTORS, Q_DETECTORS)
from .lib.features import (FEATURE_DIM, FEATURE_GROUP, FEATURE_INDEX, FEATURE_KIND,
                           FEATURE_NAMES_V2, GROUP_ORDER, GROUPS, KEY_FEATURE_IDX)

NF = FEATURE_DIM
_NAN = math.nan
DAY = 86400.0
HOUR = 3600.0

# ---- store names (contract B)
NAT = "feature.nat"
TCTX = "feature.tctx"
Z = "behavior.z"
ZR = "behavior.zr"
ZI = "behavior.zi"
PF = "behavior.pf"
P = emit.P
SCORE = emit.SCORE
PM = emit.PM
Q_INST = "behavior.q_inst"
Q_ALL = "behavior.q_all"
E_DAY = "behavior.e_day"
EVIDENCE = "behavior.evidence"
ALARM = "behavior.alarm"
P_FAMILY = "behavior.p_family"
RISK = "behavior.risk"
CALIB_HEALTH = "behavior.calib_health"
COMMON_FLAG = "behavior.common.flag"
COMMON_GROUPS = ("volume", "transport", "app_error", "probe")
RHYTHM_SERIES = "behavior.rhythm"
CLASS_AGG = "behavior.class.agg"
CLASSAGG_MODEL = "model.classagg"
# spec v2.1 grain series (docs/lib3/cadence.md §9.4)
CLASS_AGG_H = "behavior.class.agg.h"
Q_INST_H = "behavior.q_inst.h"
EVIDENCE_H = "behavior.evidence.h"
PROV = "behavior.prov"
DENSITY_Q = m_density.MODEL + ".q"
META_INST_T, META_INST_H, META_ALL = "meta_inst_t", "meta_inst_h", "meta_all"
PROV_MIN = 0.5
STACK_SET = "client.stack_set"

# ---- B25 / B27 constants the decision recompute mirrors (engines.md B25, B27)
EVIDENCE_ARL_DAYS = 33.0
SINGLE_E_DAY = 0.03
BH_Q = 0.05
OPEN_FINDING_RANK = 2
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
NOVELTY_KINDS = ("first_seen", "rare_access")
XSYS_DIM = "xsys"               # B21 first_access_system token dimension (cross_system.XSYS_DIM)
# round 4: the detector a discrete finding comes from (neutralising the
# detector removes its findings of the decision tick as well)
FINDING_DETECTORS: Dict[str, Tuple[str, ...]] = {
    "first_seen": ("novelty",), "rare_access": ("novelty",),
    "client_change": ("client",), "client_impersonation": ("client",),
    "identity_mismatch": ("identity",), "unknown_identity": ("identity",),
    "beacon": ("beacon",), "budget_exceeded": ("budget_vol", "budget_exfil", "budget_breadth"),
    "baseline_creep": ("creep",), "schedule_shift": ("offhours",),
    "first_access_system": ("cross_system",),
    "pattern_violation": ("conf_who", "conf_when", "conf_content", "conf_seq", "conf_novel"),
}
# the numeric feature that carries a discrete finding (round 4, opening
# evidence): a decisive finding (>= MEDIUM, it opens an incident alone) ranks
# its feature first; first_seen / rare_access by the dimension of the value
FINDING_FEATURES: Dict[str, Tuple[str, ...]] = {
    "client_change": ("ja3_diversity",), "client_impersonation": ("ja3_diversity",),
    "beacon": ("periodicity",), "budget_exceeded": ("bytes_up",),
    "novel:tmpl": ("new_template_ratio", "distinct_templates"),
    "novel:sni": ("new_peer_count",), "novel:dns": ("new_peer_count",),
    "novel:peer": ("new_peer_count", "distinct_peers"), "novel:dport": ("distinct_dports",),
}
# B27's risk opening (incident.py): a family at e_day <= 0.1 within 24 h
RISK_FAM_E_DAY = 0.1
RISK_LOOKBACK_S = DAY
NEUTRAL_P = 0.5                 # a neutralised detector reads as the median of its null
CF_DET_P = 0.05                 # a detector drives the decision below this p
CF_DETECTORS = 10               # detector candidates tried by the counterfactual
OPEN_EV_BASE_Q = 0.9            # the entity's routine level of -log10 pf (opening evidence)
_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}

# ---- explanation parameters
TOP_ATTR = 8                    # attributions reported
CF_CANDIDATES = 10              # numeric candidates tried by the counterfactual
CF_Z_MIN = 1.0                  # a numeric candidate needs |z| >= 1 (else nothing to reset)
MAX_WINDOW_TICKS = 96           # replay window cap (engines.md B29 perf)
BH_ALPHA = 0.05
STATE_QS = (0.05, 0.5, 0.95)
BAND_DT_S = 900.0               # natural units per 15 min (model_state exposure)
VANISH_SHARE = 0.02
VANISH_KNOWN_S = 3 * DAY
VANISH_QUIET_S = 6 * HOUR
STACK_NEW_SHARE = 0.01
STACK_DOMINANT_SHARE = 0.10
FIDELITY_TOL_LOG10 = 0.05       # |log10 p_recomputed - log10 p_stored| counted as exact
CLASS_REF_MIN_NEFF = 24.0       # B18 uses its reference anchor from this n_eff on
SEED_TAG = "B04"

_B04_DETS = ("marg_int", "marg_shape", "peer")
_B06_DETS = ("t2", "spe")
_B04Q_DETS = ("marg_int_q", "marg_shape_q")
_B06Q_DETS = ("t2_q", "spe_q")
_STREAM = [str(DETECTOR_INFO[d].get("stream", "t")) for d in DETECTORS]
_INST_T_IDX = [i for i, d in enumerate(DETECTORS)
               if DETECTOR_INFO[d]["kind"] == "inst" and _STREAM[i] == "t"]
_INST_H_IDX = [i for i, d in enumerate(DETECTORS)
               if DETECTOR_INFO[d]["kind"] == "inst" and _STREAM[i] == "h"
               and not DETECTOR_INFO[d].get("overlap")]
_Q_IDX = [DETECTOR_INDEX[d] for d in Q_DETECTORS]
_B14_DETS = ("cusum", "mcusum")
_ACC_SET = frozenset(ACC_DETECTORS)
_FAMILY_OF = [str(DETECTOR_INFO[d]["family"]) for d in DETECTORS]
_FID = [FAMILIES.index(f) for f in _FAMILY_OF]
_INST = [DETECTOR_INFO[d]["kind"] == "inst" for d in DETECTORS]
_KEY_SET = frozenset(KEY_FEATURE_IDX)
_VOL_IDX = np.asarray(GROUPS["volume"], dtype=np.intp)

DOW_EN = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
DOW_ZH = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")

# natural-unit display per feature (engines.md B29: 'bytes_up 38.2 MB/15 min')
_BYTES = frozenset({"bytes_up", "bytes_down"})
_BYTES_AVG = frozenset({"bytes_per_flow", "resp_bytes_avg", "req_bytes_avg"})
_MS = frozenset({"http_latency", "tls_handshake_ms", "rtt", "flow_duration"})
_ZH_NAME = {
    "bytes_up": "上行流量", "bytes_down": "下行流量", "flows": "连接数",
    "http_requests": "HTTP请求数", "dns_queries": "DNS查询数", "tls_handshakes": "TLS握手数",
    "intensity": "操作次数", "bytes_per_flow": "每连接字节", "updown_log": "上下行比(对数)",
    "distinct_peers": "对端数", "distinct_dports": "目的端口数",
    "distinct_templates": "访问模板数", "new_peer_count": "新对端数",
    "http_write_ratio": "写请求占比", "http_4xx_rate": "4xx占比", "http_5xx_rate": "5xx占比",
    "http_latency": "HTTP时延", "dns_fail_rate": "DNS失败率", "dns_txt_ratio": "TXT查询占比",
}


def _f(x: Any) -> float:
    if x is None:
        return _NAN
    try:
        return float(x)
    except (TypeError, ValueError):
        return _NAN


def _fin(x: Any) -> Optional[float]:
    v = _f(x)
    return v if math.isfinite(v) else None


def _r(x: Any, nd: int = 4) -> Optional[float]:
    v = _f(x)
    if not math.isfinite(v):
        return None
    return float(f"{v:.{nd}g}")


def _sev_name(x: Any) -> str:
    v = getattr(x, "value", x)
    return str(v or "low").lower()


def _sev_rank(x: Any) -> int:
    return _RANK.get(_sev_name(x), 0)


def _neglog10(p: float) -> float:
    return -math.log10(max(p, bayes.P_FLOOR)) if p == p else _NAN


# =================================================================== units
def _fmt_num(v: float) -> str:
    """Three significant digits, no exponent below 1000 (38.2, 0.4, 25, 1234)."""
    if abs(v) >= 1000:
        return f"{v:.0f}"
    if v == 0:
        return "0"
    return f"{v:.3g}"


def _fmt_bytes(v: float) -> Tuple[str, str]:
    """(number, unit) with one unit for the value (B, KB, MB, GB)."""
    a = abs(v)
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if a >= scale:
            return _fmt_num(v / scale), unit
    return _fmt_num(v), "B"


def _bytes_scale(vals: Sequence[float]) -> Tuple[str, float]:
    m = max((abs(v) for v in vals if v == v), default=0.0)
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if m >= scale:
            return unit, scale
    return "B", 1.0


def feature_unit(name: str) -> Tuple[str, float, bool]:
    """(unit label, display scale, per_15min) of a feature in natural units."""
    kind = FEATURE_KIND.get(name, "")
    if name in _BYTES:
        return "B", 1.0, True
    if kind in ("count", "bytes"):
        return "", 1.0, True
    if kind == "ratio":
        return "%", 100.0, False
    if name in _BYTES_AVG:
        return "B", 1.0, False
    if name in _MS:
        return "ms", 1.0, False
    if name == "think_time":
        return "s", 1.0, False
    return "", 1.0, False


def fmt_range(name: str, obs: float, lo: float, mid: float, hi: float,
              band_s: float = BAND_DT_S) -> Dict[str, Any]:
    """Display strings of an observed natural value and its usual band
    (band_s: the exposure of count / bytes values; spec v2.1 H rows 3600)."""
    unit, scale, per15 = feature_unit(name)
    suffix = (f"/{int(band_s // 60)} min" if band_s < 3600.0 else
              ("/h" if band_s == 3600.0 else f"/{band_s / 3600.0:g} h")) if per15 else ""
    if unit == "B":
        u, sc = _bytes_scale([obs, lo, hi, mid])
        f = lambda v: _fmt_num(v / sc)                       # noqa: E731
        unit_s = u
    else:
        f = lambda v: _fmt_num(v * scale)                    # noqa: E731
        unit_s = unit
    sp_ = "" if unit_s in ("%", "") else " "
    obs_s = f"{f(obs)}{sp_}{unit_s}{suffix}" if obs == obs else "n/a"
    rng = f"{f(lo)}–{f(hi)}{sp_}{unit_s}" if lo == lo and hi == hi else ""
    ratio = _NAN
    if obs == obs and mid == mid:
        if abs(mid) > 1e-12 and FEATURE_KIND.get(name) != "clr" and name != "updown_log":
            ratio = obs / mid
    return {"observed_text": obs_s, "range": rng.strip(), "ratio": ratio, "unit": unit_s + suffix}


# ============================================================ B04 channels
class _Layout:
    """B04's dependence-group layout (likelihood.GroupLayout semantics):
    groups of model.groups split by FEATURE_SPEC group, missing features as
    singletons, sorted by their first index."""

    def __init__(self, groups: Sequence[Sequence[int]]) -> None:
        gs = [sorted(int(i) for i in g) for g in groups if len(g)]
        seen = {i for g in gs for i in g}
        gs = sorted(gs + [[i] for i in range(NF) if i not in seen], key=lambda g: g[0])
        self.M = np.zeros((NF, len(gs)))
        fg = []
        for j, g in enumerate(gs):
            self.M[g, j] = 1.0
            fg.append(FEATURE_GROUP[FEATURE_NAMES_V2[g[0]]])
        self.fg_names = [n for n in GROUP_ORDER if n in fg]
        self.FG = np.zeros((len(gs), len(self.fg_names)))
        for j, n in enumerate(fg):
            self.FG[j, self.fg_names.index(n)] = 1.0
        self.vol = np.array([n == "volume" for n in fg], dtype=bool)


def _hmp_cols(p: np.ndarray, M: np.ndarray) -> np.ndarray:
    ok = np.isfinite(p)
    inv = np.where(ok, 1.0 / np.clip(np.where(ok, p, 1.0), bayes.P_FLOOR, 1.0), 0.0)
    n = ok.astype(np.float64) @ M
    s = inv @ M
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n > 0.0, n / s, np.nan)
    return np.minimum(out, 1.0)


def _hmp(p: np.ndarray) -> float:
    ok = np.isfinite(p)
    if not ok.any():
        return _NAN
    return min(1.0, float(ok.sum() / np.sum(1.0 / np.clip(p[ok], bayes.P_FLOOR, 1.0))))


def channels(p: np.ndarray, lay: _Layout) -> Tuple[float, float, float]:
    """(p_int, p_shape, p_all) of B04's two-stage harmonic mean."""
    pg = _hmp_cols(np.asarray(p, dtype=np.float64), lay.M)
    return _hmp(pg[lay.vol]), _hmp(pg[~lay.vol]), _hmp(pg)


# ================================================================ B25 fuse
def fuse(p_row: Sequence[float], family_w: Optional[Mapping[str, float]],
         wmult: Optional[Sequence[float]]) -> Tuple[Dict[str, float], float, float]:
    """B25 step 1 (fusion.fuse semantics): (p_family, p_inst, p_all)."""
    vals = [float(v) for v in p_row]
    wm = wmult if wmult is not None else [1.0] * N_DETECTORS
    nf = len(FAMILIES)
    sw, swp, iw, iwp = [0.0] * nf, [0.0] * nf, [0.0] * nf, [0.0] * nf
    cnt, icnt = [0] * nf, [0] * nf
    pfl = combine.P_FLOOR
    for i in range(N_DETECTORS):
        v = vals[i]
        if 0.0 <= v <= 1.0:
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
        if not cnt[f]:
            continue
        name = FAMILIES[f]
        g = float(fw.get(name, 1.0))
        p = sw[f] / swp[f]
        pf[name] = 1.0 if p > 1.0 else p
        wf[name] = g * sw[f] / cnt[f]
        if icnt[f]:
            p = iw[f] / iwp[f]
            ip.append(1.0 if p > 1.0 else p)
            iws.append(g * iw[f] / icnt[f])
    p_all = combine.whmp(list(pf.values()), list(wf.values())) if pf else _NAN
    p_inst = combine.whmp(ip, iws) if ip else _NAN
    return pf, p_inst, p_all


def bh_threshold(pvals: Iterable[float], q: float = BH_Q) -> float:
    p = np.asarray([v for v in pvals if v == v], dtype=np.float64)
    m = p.size
    if m == 0:
        return -1.0
    p.sort()
    ok = np.flatnonzero(p <= q * np.arange(1, m + 1) / m)
    return float(p[ok[-1]]) if ok.size else -1.0


def _meta_score(p: float) -> float:
    p = _f(p)
    if p != p:
        return _NAN
    p = 1.0 if p > 1.0 else (combine.P_FLOOR if p < combine.P_FLOOR else p)
    return -math.log10(p)


# ============================================================== store reads
def _vec(store, s: str, e: str, name: str, ts: float) -> Optional[np.ndarray]:
    row = store.vec_at(s, e, name, ts)
    return None if row is None else np.asarray(row, dtype=np.float64)


def _scalar(store, s: str, e: str, name: str, ts: float) -> float:
    row = store.vec_at(s, e, name, ts)
    return float(row[0]) if row is not None and len(row) else _NAN


def _dict_at(store, s: str, e: str, name: str, ts: float, depth: int = 8
             ) -> Optional[Dict[str, Any]]:
    """Dict series value written at exactly ts (searching the newest `depth`
    points), else None."""
    m = store.latest_derived(s, e, name)
    if m is not None and m.ts == ts and isinstance(m.value, Mapping):
        return dict(m.value)
    for m in reversed(store.derived_tail(s, e, name, max(8, int(depth)))):
        if m.ts == ts and isinstance(m.value, Mapping):
            return dict(m.value)
        if m.ts < ts:
            break
    return None


def _tctx_at(store, s: str, e: str, ts: float, config: Mapping[str, Any], dt: float
             ) -> Dict[str, Any]:
    """feature.tctx written at ts (dict series or vec encoding), else config."""
    for m in reversed(store.derived_tail(s, e, TCTX, 256)):
        if m.ts == ts and isinstance(m.value, Mapping) and m.value.get("bin48") is not None:
            return dict(m.value)
        if m.ts < ts:
            break
    row = store.vec_at(s, e, TCTX, ts)
    if row is not None and len(row) == len(timebins.TCTX_FIELDS):
        return dict(timebins.decode_tctx(row))
    return dict(timebins.tctx_from_config(ts, config, dt))


def _day_label(tc: Mapping[str, Any]) -> Tuple[str, str]:
    dow = int(_f(tc.get("dow")) if _f(tc.get("dow")) == _f(tc.get("dow")) else 0) % 7
    h = _f(tc.get("hour_local"))
    hh = int(math.floor(h)) if h == h else 0
    return f"{DOW_EN[dow]} {hh:02d}:00", f"{DOW_ZH[dow]} {hh:02d}:00"


# ======================================================== counterfactual
class _Tick:
    """Factual inputs of one tick of the replay window, read once."""

    __slots__ = ("ts", "dt", "nat", "z", "zi", "pf", "p", "pm", "score", "tc", "strat",
                 "preds", "has_tier", "fact", "dt_tick", "grain", "qv")

    def __init__(self) -> None:
        self.preds: Optional[Dict[str, Any]] = None
        self.dt_tick = _NAN
        self.grain: Optional[Dict[str, Any]] = None   # spec v2.1: {dp_h, dp_q, prov}
        self.qv: Optional["_QView"] = None


class _QView:
    """spec v2.1: the Q-grain row of a Q decision tick (B04 / B06 Q pass inputs)."""

    __slots__ = ("nat", "dt", "tc", "z", "zi", "pf", "preds", "med", "obs", "fact")

    def __init__(self) -> None:
        self.preds: Optional[Dict[str, Any]] = None
        self.med: Optional[np.ndarray] = None
        self.obs: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self.fact: Dict[str, Any] = {}


class _Neutral:
    """A candidate set split by kind."""

    def __init__(self, cands: Sequence[str]) -> None:
        self.features = sorted(FEATURE_INDEX[c] for c in cands if c in FEATURE_INDEX)
        self.tokens = {c[len("token:"):] for c in cands if c.startswith("token:")}
        self.offhours = "offhours" in cands
        self.detectors = {c[len("detector:"):] for c in cands if c.startswith("detector:")}


class Counterfactual:
    """Recompute of the incident decision at `now` with a candidate set
    neutralised (module docstring, step 4). One instance per incident; the
    factual reads and the stateless models are cached across candidate sets."""

    def __init__(self, store, s: str, e: str, now: float, dt: float,
                 config: Mapping[str, Any], inc: Any, max_ticks: int = MAX_WINDOW_TICKS,
                 depth: int = 8) -> None:
        # `now` is the decision tick: the tick the incident (re)opened
        self.store, self.s, self.e, self.now, self.dt = store, s, e, float(now), float(dt)
        self.depth = int(depth)
        self.config = config
        self.inc = inc
        self.max_ticks = int(max_ticks)
        self.is_class = is_class(e)
        self.calib = store.get_model(s, e, m_calib.MODEL)
        meta = self.calib.get(m_calib.META) if isinstance(self.calib, Mapping) else None
        self.meta = meta if isinstance(meta, Mapping) else {}
        mst = self.meta.get("state") if isinstance(self.meta, Mapping) else None
        self.h_mult = _f((mst or {}).get("h_mult", 1.0)) if isinstance(mst, Mapping) else 1.0
        if not self.h_mult == self.h_mult:
            self.h_mult = 1.0
        self.fw = m_feedback.family_weights(store)
        self.alpha = m_feedback.alpha_mult(store, s)
        self.density = None if self.is_class else m_density.get(store, s, e)
        if self.density is not None and not m_density.is_fitted(self.density):
            self.density = None
        # spec v2.1 canonical grain mode (docs/lib3/cadence.md §9.4)
        self.canon = GR.canonical(config)
        self.cp_dt = max(self.dt, GR.GRAIN_S["h"]) if self.canon else self.dt
        mst0 = mst if isinstance(mst, Mapping) else {}
        hmh = _f(mst0.get("h_mult_h", 1.0))
        self.h_mult_h = hmh if hmh == hmh else 1.0
        bp = mst0.get("pending")
        self.b25_pend = bp if isinstance(bp, Mapping) else {}
        self.density_q = None
        if self.canon and not self.is_class:
            dq = store.get_model(s, e, DENSITY_Q)
            self.density_q = dq if isinstance(dq, Mapping) and m_density.is_fitted(dq) else None
        self.layout = _Layout(m_density.groups(store, s, split_by_feature_group=True))
        self.ticks: Dict[float, _Tick] = {}
        self._pred_cache: Dict[Tuple[int, int, str], Tuple[Dict[str, Any], bool]] = {}
        self._fc: Optional[Dict[str, Any]] = None
        self._med: Dict[float, np.ndarray] = {}
        self.cand_idx: Optional[Set[int]] = None
        self._obs: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._pk: Dict[float, np.ndarray] = {}
        self._ev_on_v: Optional[bool] = None
        self._wm: Optional[List[float]] = None
        self._zr_med: Dict[float, Dict[int, float]] = {}
        self.key_cands: List[int] = []
        self.recomputed: Set[str] = set()
        self.replayed: Set[str] = set()
        self.held: Set[str] = set()
        self.neutralised: Set[str] = set()
        self._rh: Optional[List[Tuple[float, float, Set[str]]]] = None
        self.fid: Dict[str, Any] = {"p_log10_err": 0.0, "n_p": 0}
        self._cls_rings: Optional[List[Any]] = None
        self._cache: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        self.classagg = store.get_model(s, e, CLASSAGG_MODEL) if self.is_class else None
        # ---- stored decision at now
        self.alarm = _dict_at(store, s, e, ALARM, self.now, self.depth)
        self.acc = _dict_at(store, s, e, emit.ACC_ALARM, self.now, self.depth) or {}
        self.findings = [ev for ev in store.events(s, e, since=self.now, kinds=DISCRETE_KINDS)
                         if float(ev.ts) == self.now and ev.status != "suppressed"
                         and _sev_rank(ev.severity) >= OPEN_FINDING_RANK]
        self.risk_open = any(isinstance(x, Mapping) and x.get("source") == "risk"
                             and _f(x.get("ts")) == self.now for x in (inc.evidence or ()))
        # ---- replay windows
        self._cusum = self._prepare_cusum() if not self.is_class else None
        self._off = self._prepare_offhours() if not self.is_class else None
        self.ev_window = self._evidence_window()

    # ------------------------------------------------------------ ticks
    def tick(self, ts: float) -> Optional[_Tick]:
        t = self.ticks.get(ts)
        if t is not None:
            return t
        st, s, e = self.store, self.s, self.e
        t = _Tick()
        t.ts = ts
        t.p = _vec(st, s, e, P, ts)
        t.score = _vec(st, s, e, SCORE, ts)
        t.fact = {}
        pm = _vec(st, s, e, PM, ts)
        t.pm = pm if pm is not None else np.full(N_DETECTORS, np.nan)
        strat = m_calib.issued_stratum(self.calib, ts)
        t.tc = _tctx_at(st, s, e, ts, self.config, self.dt)
        if strat is None:
            strat = (str(t.tc.get("daypart", "wd_day")),
                     m_calib.regime_tercile(_last_regime(st, s, e)), self.dt)
        t.strat = strat
        t.dt = float(strat[2]) if strat[2] > 0 else self.dt
        if self.is_class:
            t.nat = _vec(st, s, e, CLASS_AGG, ts)
            t.z = t.zi = t.pf = None
        else:
            t.nat = _vec(st, s, e, NAT, ts)
            t.z = _vec(st, s, e, Z, ts)
            t.zi = _vec(st, s, e, ZI, ts)
            t.pf = _vec(st, s, e, PF, ts)
        t.has_tier = False
        t.dt_tick = t.dt
        if self.canon:
            self._grain_views(t)
        self.ticks[ts] = t
        return t

    def _grain_views(self, t: _Tick) -> None:
        """spec v2.1: the numeric recompute of tick t uses the grain rows B04 /
        B06 / B18 scored there: the H row on H decision ticks (feature.nat.h
        masked like B04, exposure = its coverage, the window-midpoint tctx),
        the Q row on Q decision ticks (t.qv); between decision ticks the grain
        detectors were not scored (their p is NaN) and nothing is recomputed."""
        st, s, e, ts, dtk = self.store, self.s, self.e, t.ts, t.dt_tick
        cfg = self.config
        dp_t = str(t.tc.get("daypart", "wd_day"))
        g = m_calib.issued_grain(self.calib, ts)
        if g is None:
            dp_h = GR.row_tctx(ts, "h", dtk, cfg)["daypart"]
            dp_q = (GR.row_tctx(ts, "q", dtk, cfg)["daypart"]
                    if GR.observable("q", dtk, GR.CANONICAL) else dp_h)
            pv = emit.read_dict(st, s, e, PROV, ts)
            prov = {d: int(_f(pv[d]) < PROV_MIN) for d in Q_DETECTORS
                    if d in pv and _f(pv[d]) == _f(pv[d])}
            g = {"dp_h": dp_h, "dp_q": dp_q, "prov": prov}
        g = dict(g)
        g["dp_t"] = dp_t
        t.grain = g
        h_due = GR.decision(ts, dtk, "h", GR.CANONICAL)
        if self.is_class:
            t.nat = None
            if h_due:
                agg = _vec(st, s, e, CLASS_AGG_H, ts)
                cov = _NAN
                for m in m_class.class_members(st, s, e):
                    mh = st.vec_at(s, m, "feature.meta.h", ts)
                    if mh is not None and st.vec_at(s, m, "feature.nat.h", ts) is not None:
                        cov = float(mh[1]) if not cov >= float(mh[1]) else cov
                if agg is not None and cov > 0.0:
                    t.nat, t.dt = agg, cov
                    t.tc = GR.row_tctx(ts, "h", dtk, cfg)
            return
        t.nat = None
        if h_due:
            meta = _vec(st, s, e, "feature.meta.h", ts)
            nat = _vec(st, s, e, "feature.nat.h", ts)
            if meta is not None and float(meta[0]) > 0.5 and nat is not None:
                x = nat.copy()
                x[~GR.scored_mask(ts, dtk, "h", cfg)] = np.nan
                t.nat, t.dt = x, float(meta[1])
                t.tc = GR.row_tctx(ts, "h", dtk, cfg)
        if t.nat is None:
            t.z = t.zi = t.pf = None
        if GR.decision(ts, dtk, "q", GR.CANONICAL) and GR.observable("q", dtk, GR.CANONICAL):
            meta = _vec(st, s, e, "feature.meta.q", ts)
            nat = _vec(st, s, e, "feature.nat.q", ts)
            if meta is not None and float(meta[0]) > 0.5 and nat is not None:
                v = _QView()
                v.tc = GR.row_tctx(ts, "q", dtk, cfg)
                v.preds = MB.predictive_q(st, s, e, v.tc)
                x = nat.copy()
                x[~GR.scored_mask(ts, dtk, "q", cfg)] = np.nan
                cur = v.preds["current"]
                if cur.scored is not None:
                    x = np.where(cur.scored, x, np.nan)
                v.nat, v.dt = x, float(meta[1])
                v.z = _vec(st, s, e, Z + ".q", ts)
                v.zi = _vec(st, s, e, ZI + ".q", ts)
                v.pf = _vec(st, s, e, PF + ".q", ts)
                t.qv = v

    def _preds(self, t: _Tick) -> Dict[str, Any]:
        if t.preds is None:
            pk = (int(_f(t.tc.get("bin48"))), int(_f(t.tc.get("bin168"))),
                  str(t.tc.get("day_type")))
            hit = self._pred_cache.get(pk)
            if hit is not None:
                t.preds, t.has_tier = hit
                return t.preds
            if self.is_class:
                m = self.classagg
                pc = MB.anchor_predictive(m["current"], t.tc)
                ref = m.get("reference")
                pr = None
                if ref is not None and not ref.empty and MB.n_eff(ref) >= CLASS_REF_MIN_NEFF:
                    pr = MB.anchor_predictive(ref, t.tc, anchor="reference")
                t.preds = {"current": pc, "reference": pr, "class": None}
            else:
                t.preds = MB.predictive_set(self.store, self.s, self.e, t.tc)
                t.has_tier = MB.tier_model(self.store, self.s,
                                           MB.parent_key(self.store, self.s, self.e)) is not None
            self._pred_cache[pk] = (t.preds, t.has_tier)
        return t.preds

    def zr_median(self, ts: float) -> Dict[int, float]:
        """{key feature: zr of its bucket median} at tick ts for every numeric
        candidate that is a CUSUM key feature (one midp pass per tick: a
        feature's PIT depends on its own value only, so resetting all
        candidates together gives each one's residual)."""
        hit = self._zr_med.get(ts)
        if hit is not None:
            return hit
        out: Dict[int, float] = {}
        feats = self.key_cands
        t = self.tick(ts) if feats else None
        if t is not None and t.nat is not None:
            preds = self._preds(t)
            pr = preds["reference"]
            if pr is not None:
                med = self._median(t)
                nat_cf = t.nat.copy()
                for f in feats:
                    if t.nat[f] == t.nat[f] and med[f] == med[f]:
                        nat_cf[f] = med[f]
                u_r, _ = MB.midp(pr, nat_cf, t.dt)
                with np.errstate(all="ignore"):
                    zrc = sp.ndtri(np.clip(u_r, bayes.PHI_CLIP, 1.0 - bayes.PHI_CLIP))
                for f in feats:
                    if zrc[f] == zrc[f]:
                        out[int(f)] = float(zrc[f])
        self._zr_med[ts] = out
        return out

    # ------------------------------------------------------ CUSUM replay
    def _prepare_cusum(self) -> Optional[Dict[str, Any]]:
        st, s, e = self.store, self.s, self.e
        if st.vec_latest(s, e, m_cp.CUSUM_STATE) is None:
            return None
        # spec v2.1: B14 steps on H decision ticks (period max(dt, 3600)), so
        # the replay covers <= 96 H states (the 4-d cusum_state ring)
        cdt = self.cp_dt
        params = m_cp.replay_params(st, s, e, dt=cdt)
        tau = m_cp.onset(st, s, e)
        opened = float(self.inc.opened)
        start = tau - cdt if tau == tau else opened - 4.0 * cdt
        start = min(start, opened - cdt)
        start = max(start, self.now - self.max_ticks * cdt)
        s0 = m_cp.replay_state(st, s, e, start)
        if s0 is None:                        # the ring does not reach back: its oldest row
            t, _ = st.vec_range(s, e, m_cp.CUSUM_STATE, -math.inf, self.now)
            if not len(t):
                return None
            s0 = m_cp.replay_state(st, s, e, float(t[0]))
            if s0 is None:
                return None
        ts0, state0 = s0
        rows = m_cp.replay_rows(st, s, e, ts0, self.now, dt=cdt, dt_of=self._dt_of)
        if not rows:
            return None                       # (an idle tick keeps the last row's state)
        rows, info = m_cp.mark_resets(params, state0, rows)
        complete = not (tau == tau and ts0 > tau - self.dt)
        return {"params": params, "state0": m_cp.replay_state0(state0), "rows": rows,
                "t0": ts0, "info": info, "complete": complete, "tau": tau}

    def _dt_of(self, ts: float) -> Optional[float]:
        """Cadence of tick ts as B24 recorded it (None when not recorded)."""
        strat = m_calib.issued_stratum(self.calib, ts)
        v = float(strat[2]) if strat is not None and strat[2] > 0 else None
        if v is not None and self.canon:
            v = max(v, GR.GRAIN_S["h"])
        return v

    def _cusum_run(self, zr_cf: Mapping[float, Dict[int, float]]) -> Dict[str, Any]:
        """Replay the bank with the neutralised key residuals; returns the
        scores at now and the latches (a crossing of h in the window)."""
        cz = self._cusum
        step = m_cp.replay_step(cz["params"])
        mc_max = [0.0]

        def perturb(ts: float, inp: Mapping[str, Any]) -> Mapping[str, Any]:
            vals = zr_cf.get(ts)
            if not vals or "resync" in (inp.get("mode") or ()):
                return inp
            return dict(inp, x=m_cp.neutralize_key(inp["x"], list(vals), vals))

        def step2(state, inp):
            new, sc = step(state, inp)
            mc_max[0] = max(mc_max[0], float(new.get("mc_ratio", 0.0)))
            return new, sc

        res = RP.replay(step2, cz["state0"], cz["rows"], perturb=perturb)
        fs = res.final_state
        p_bank = float(np.min(m_cp.peq(fs["S"])))
        p_mc = m_cp.mcusum_p(float(np.sqrt(np.sum(fs["mc"] * fs["mc"]))))
        return {"score": {"cusum": _neglog10(p_bank), "mcusum": _neglog10(p_mc)},
                "pm": {"cusum": p_bank, "mcusum": p_mc},
                "latch": {"cusum": res.max_score >= 1.0, "mcusum": mc_max[0] >= 1.0},
                "max": res.max_score}

    # ---------------------------------------------------- off-hours replay
    def _prepare_offhours(self) -> Optional[Dict[str, Any]]:
        if not int(_f(self.acc.get("offhours", 0)) or 0):
            return None
        model = self.store.get_model(self.s, self.e, "model.rhythm")
        if not isinstance(model, Mapping) or not model.get("ledger"):
            return None
        rh = _dict_at(self.store, self.s, self.e, RHYTHM_SERIES, self.now, self.depth) or {}
        h_off = _f(rh.get("h_off"))
        since = float(self.inc.opened) - 7 * DAY
        fact = m_rhythm.offhours_replay(model, (), since=since, until=self.now)
        w_rec = _f(rh.get("W_off"))
        return {"model": model, "since": since, "h": h_off, "fact": fact,
                "err": abs(fact["W_final"] - w_rec) if w_rec == w_rec else _NAN,
                "slots": [j for j in fact["active_unusual"]
                          if timebins.slot_bounds(j, self.config.get("tz", timebins.DEFAULT_TZ))[1]
                          > float(self.inc.opened) - 4 * HOUR]}

    # ---------------------------------------------------- evidence window
    def _evidence_window(self, name: str = EVIDENCE) -> List[float]:
        """Ticks of the evidence CUSUM's current excursion (after its last
        stored zero, <= max_ticks), ending at now; [now] when S is 0.
        spec v2.1: name = behavior.evidence.h for the H stream (rows on H
        ticks, so the look-back is max_ticks H periods)."""
        per = self.cp_dt if name == EVIDENCE_H else self.dt
        ts, M = self.store.vec_range(self.s, self.e, name,
                                     self.now - (self.max_ticks + 1) * per, self.now)
        if not len(ts) or float(ts[-1]) != self.now:
            return [self.now]
        ts, M = ts[-(self.max_ticks + 1):], M[-(self.max_ticks + 1):]
        out: List[float] = []
        for i in range(len(ts) - 1, -1, -1):
            out.append(float(ts[i]))
            if i > 0 and not float(M[i - 1, 0]) > 0.0:
                break
        out.reverse()
        return out

    def _s_before(self, ts: float, name: str = EVIDENCE) -> float:
        t, M = self.store.vec_range(self.s, self.e, name, -math.inf, ts - 1e-6)
        if not len(t):
            return 0.0
        v = float(M[-1, 0])
        return v if v == v else 0.0

    # ------------------------------------------------------------ p rule
    def _class_rings(self, d: str, t: _Tick) -> Optional[List[Any]]:
        if self.is_class:
            return None
        if self._cls_rings is None:
            ck = m_class.class_key(self.store, self.s, self.e)
            mem = m_class.class_members(self.store, self.s, ck) if ck else []
            self._cls_rings = [x for x in (self.store.get_model(self.s, m, m_calib.MODEL)
                                           for m in mem if m != self.e)
                               if isinstance(x, Mapping)]
        dp, terc, dtc = t.strat
        cc = timebins.cadence_class(dtc)
        gr, dp, prov = self._grain_of(d, t, dp)
        key = m_calib.ring_key_for(d, dp, cc, terc, grain=gr, prov=prov)
        return [r for r in ((m.get(m_calib.RINGS) or {}).get(key) for m in self._cls_rings)
                if r is not None]

    def _grain_of(self, d: str, t: _Tick, dp: str) -> Tuple[Optional[str], str, int]:
        """(grain, daypart, prov) of detector d's B24 stratum at tick t: spec
        v2.1 H / Q stream detectors use their grain row's daypart (and Q the
        provisional flag B24 recorded); T-stream detectors and tick mode keep
        the tick's daypart and cadence class."""
        if not self.canon or t.grain is None:
            return None, dp, 0
        stv = _STREAM[DETECTOR_INDEX[d]]
        if stv == "h":
            return "h", str(t.grain.get("dp_h", dp)), 0
        if stv == "q":
            return "q", str(t.grain.get("dp_q", dp)), int((t.grain.get("prov") or {}).get(d, 0))
        return None, dp, 0

    def p_of(self, d: str, t: _Tick, score: float, pm: float) -> float:
        dp, terc, dtc = t.strat
        cc = timebins.cadence_class(dtc)
        gr, dpg, prov = self._grain_of(d, t, dp)
        return m_calib.p_replay(self.calib, d, dpg, cc, score,
                                m_calib.uniform(self.s, self.e, d, t.ts), tercile=terc,
                                pm=pm, class_rings=self._class_rings(d, t),
                                grain=gr, prov=prov)

    def _check_p(self, d: str, t: _Tick, score: float, pm: float) -> None:
        """Fidelity: the factual score's recomputed p against the stored p."""
        i = DETECTOR_INDEX[d]
        ps = float(t.p[i])
        if not (ps == ps and score == score):
            return
        pr = self.p_of(d, t, score, pm)
        if pr == pr and pr > 0 and ps > 0:
            err = abs(math.log10(pr) - math.log10(ps))
            self.fid["p_log10_err"] = max(self.fid["p_log10_err"], err)
            self.fid["n_p"] += 1
            if err > FIDELITY_TOL_LOG10:
                self.fid.setdefault("inexact", {})[d] = _r(err, 3)

    # ---------------------------------------------------- stateless recompute
    def _numeric(self, t: _Tick, feats: Sequence[int], check: bool
                 ) -> Tuple[Dict[str, float], Dict[str, float], Dict[int, float]]:
        """(scores, pm, zr_cf by feature) of the B04 / B06 (entity) or B18
        class_int (class) detectors with `feats` reset to the bucket median."""
        scores: Dict[str, float] = {}
        pms: Dict[str, float] = {}
        zr_cf: Dict[int, float] = {}
        if check and t.nat is not None:
            self._check_factual(t)
        if t.nat is None or not feats:
            return scores, pms, zr_cf
        preds = self._preds(t)
        pc, pr, pk = preds["current"], preds["reference"], preds.get("class")
        nat = t.nat
        med = self._median(t)
        nat_cf = nat.copy()
        for f in feats:
            if nat[f] == nat[f] and med[f] == med[f]:
                nat_cf[f] = med[f]
        n0, d0, w0 = self._observed(t)
        n1, d1, w1 = MB.observe(nat_cf, t.dt)
        changed = np.flatnonzero((n0 != n1) | (d0 != d1) | (w0 != w1))
        if not changed.size:
            return scores, pms, zr_cf
        # the predictive p of the changed features only: an unchanged ratio
        # feature is masked (its scalar Beta-binomial tail is the costly
        # part of midp); counts and t features are vectorised anyway
        nat_ev = nat_cf.copy()
        keep = np.zeros(NF, dtype=bool)
        keep[changed] = True
        nat_ev[MB.RAT[~keep[MB.RAT]]] = np.nan
        u_c, p_c = MB.midp(pc, nat_ev, t.dt)
        if pr is not None:
            u_r, p_r = MB.midp(pr, nat_ev, t.dt)
            pfc = np.asarray(bayes.combine_anchors(p_c, p_r), dtype=np.float64)
        else:
            u_r, p_r = np.full(NF, np.nan), np.full(NF, np.nan)
            pfc = p_c
        if self.is_class:
            p_fact = self._class_pf(t)
            p_new = p_fact.copy()
            p_new[changed] = pfc[changed]
            pv = p_new[_VOL_IDX]
            p_int = combine.whmp(pv[np.isfinite(pv)].tolist()) if np.isfinite(pv).any() else _NAN
            pv0 = p_fact[_VOL_IDX]
            p0 = combine.whmp(pv0[np.isfinite(pv0)].tolist()) if np.isfinite(pv0).any() \
                else _NAN
            scores["class_int"], pms["class_int"] = self._anchor(t, "class_int", _neglog10(p0),
                                                                 _neglog10(p_int))
            self.recomputed.add("class_int")
            if any(FEATURE_GROUP[FEATURE_NAMES_V2[i]] != "volume" for i in changed):
                self.held.add("class_shape")
            return scores, pms, zr_cf
        # ---- entity: B04 channels
        if t.pf is not None:
            pf_cf = t.pf.copy()
            pf_cf[changed] = pfc[changed]
            p_int, p_shape, _ = channels(pf_cf, self.layout)
            scores["marg_int"], pms["marg_int"] = _neglog10(p_int), p_int
            scores["marg_shape"], pms["marg_shape"] = _neglog10(p_shape), p_shape
            self.recomputed.update(("marg_int", "marg_shape"))
        if t.has_tier and pk is not None and math.isfinite(float(t.p[DETECTOR_INDEX["peer"]])):
            pk0 = self._p_class_tier(t)
            _, pk_ch = MB.midp(pk, nat_ev, t.dt)
            pk_cf = pk0.copy()
            pk_cf[changed] = pk_ch[changed]
            _, _, p_peer = channels(pk_cf, self.layout)
            scores["peer"], pms["peer"] = _neglog10(p_peer), p_peer
            self.recomputed.add("peer")
        # ---- z / zr / zi of the changed features (mid-distribution PIT)
        with np.errstate(all="ignore"):
            zc = sp.ndtri(np.clip(u_c, bayes.PHI_CLIP, 1.0 - bayes.PHI_CLIP))
            zrc = sp.ndtri(np.clip(u_r, bayes.PHI_CLIP, 1.0 - bayes.PHI_CLIP))
        for f in changed.tolist():
            if f in _KEY_SET and zrc[f] == zrc[f]:
                zr_cf[int(f)] = float(zrc[f])
        if self.density is not None and t.zi is not None and t.z is not None:
            zi_cf = t.zi.copy()
            for f in changed.tolist():
                if zi_cf[f] == zi_cf[f] and t.z[f] == t.z[f] and zc[f] == zc[f]:
                    zi_cf[f] = zi_cf[f] + (zc[f] - t.z[f])
            sc = m_density.score_model(self.density, zi_cf)
            if sc.q > 0:
                s0 = t.fact.get("b06")
                if s0 is None:
                    s0 = t.fact["b06"] = m_density.score_model(self.density, t.zi)
                scores["t2"], pms["t2"] = self._anchor(t, "t2", _neglog10(s0.p_t2),
                                                       _neglog10(sc.p_t2))
                scores["spe"], pms["spe"] = self._anchor(t, "spe", _neglog10(s0.p_spe),
                                                         _neglog10(sc.p_spe))
                self.recomputed.update(_B06_DETS)
        return scores, pms, zr_cf

    def _numeric_q(self, t: _Tick, feats: Sequence[int], check: bool
                   ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """spec v2.1: (scores, pm) of the Q-grain detectors (B04 marg_*_q,
        B06 t2_q / spe_q) of tick t's Q row with `feats` reset to the Q
        predictive's bucket median (cadence.md §9.4: Q attribution when a _q
        detector drove the decision)."""
        v = t.qv
        scores: Dict[str, float] = {}
        pms: Dict[str, float] = {}
        if v is None:
            return scores, pms
        if check:
            self._check_factual_q(t)
        if not feats:
            return scores, pms
        pc, pr = v.preds["current"], v.preds["reference"]
        if v.med is None:
            v.med = MB.quantiles(pc, [0.5], dt_s=v.dt, nat=v.nat)[0]
        if v.obs is None:
            v.obs = MB.observe(v.nat, v.dt)
        nat_cf = v.nat.copy()
        for f in feats:
            if v.nat[f] == v.nat[f] and v.med[f] == v.med[f]:
                nat_cf[f] = v.med[f]
        n0, d0, w0 = v.obs
        n1, d1, w1 = MB.observe(nat_cf, v.dt)
        changed = np.flatnonzero((n0 != n1) | (d0 != d1) | (w0 != w1))
        if not changed.size:
            return scores, pms
        u_c, p_c = MB.midp(pc, nat_cf, v.dt)
        if pr is not None:
            _, p_r = MB.midp(pr, nat_cf, v.dt)
            pfc = np.asarray(bayes.combine_anchors(p_c, p_r), dtype=np.float64)
        else:
            pfc = p_c
        if v.pf is not None:
            pf_cf = v.pf.copy()
            pf_cf[changed] = pfc[changed]
            a, b, _ = channels(pf_cf, self.layout)
            scores["marg_int_q"], pms["marg_int_q"] = _neglog10(a), a
            scores["marg_shape_q"], pms["marg_shape_q"] = _neglog10(b), b
            self.recomputed.update(_B04Q_DETS)
        if self.density_q is not None and v.zi is not None and v.z is not None:
            with np.errstate(all="ignore"):
                zc = sp.ndtri(np.clip(u_c, bayes.PHI_CLIP, 1.0 - bayes.PHI_CLIP))
            zi_cf = v.zi.copy()
            for f in changed.tolist():
                if zi_cf[f] == zi_cf[f] and v.z[f] == v.z[f] and zc[f] == zc[f]:
                    zi_cf[f] = zi_cf[f] + (zc[f] - v.z[f])
            sc = m_density.score_model(self.density_q, zi_cf)
            if sc.q > 0:
                s0 = v.fact.get("b06")
                if s0 is None:
                    s0 = v.fact["b06"] = m_density.score_model(self.density_q, v.zi)
                scores["t2_q"], pms["t2_q"] = self._anchor(t, "t2_q", _neglog10(s0.p_t2),
                                                           _neglog10(sc.p_t2))
                scores["spe_q"], pms["spe_q"] = self._anchor(t, "spe_q", _neglog10(s0.p_spe),
                                                             _neglog10(sc.p_spe))
                self.recomputed.update(_B06Q_DETS)
        return scores, pms

    def _check_factual_q(self, t: _Tick) -> None:
        v = t.qv
        if v is None:
            return
        if v.pf is not None:
            a, b, _ = channels(v.pf, self.layout)
            self._check_p("marg_int_q", t, _neglog10(a), a)
            self._check_p("marg_shape_q", t, _neglog10(b), b)
        if self.density_q is not None and v.zi is not None:
            s0 = m_density.score_model(self.density_q, v.zi)
            if s0.q > 0:
                self._check_p("t2_q", t, _neglog10(s0.p_t2), s0.p_t2)
                self._check_p("spe_q", t, _neglog10(s0.p_spe), s0.p_spe)

    def _anchor(self, t: _Tick, d: str, s_new_fact: float, s_new_cf: float
                ) -> Tuple[float, float]:
        """(score, pm) of detector d in the counterfactual, anchored on the
        stored score: B06 / B18 refit their models right after scoring on the
        same tick, so the model this engine reads may not be the one that
        scored. When the recomputed factual score equals the stored one the
        recompute is exact; otherwise the stored score is moved by the
        change the neutralisation makes under the current model (flagged
        'anchored' in the fidelity record)."""
        stored = float(t.score[DETECTOR_INDEX[d]]) if t.score is not None else _NAN
        if not (stored == stored and s_new_fact == s_new_fact):
            return s_new_cf, 10.0 ** (-s_new_cf) if s_new_cf == s_new_cf else _NAN
        if abs(s_new_fact - stored) <= 1e-6 * (1.0 + abs(stored)):
            return s_new_cf, 10.0 ** (-s_new_cf)
        self.fid.setdefault("anchored", {})[d] = _r(s_new_fact - stored, 3)
        sc = max(0.0, stored + (s_new_cf - s_new_fact))
        return sc, 10.0 ** (-sc)

    def _check_factual(self, t: _Tick) -> None:
        """Fidelity: recompute the factual scores of the stateless detectors
        this engine re-scores (same code path, nothing neutralised) and
        compare their p with the stored behavior.p."""
        if self.is_class:
            p_fact = self._class_pf(t)
            pv0 = p_fact[_VOL_IDX]
            if np.isfinite(pv0).any():
                p0 = combine.whmp(pv0[np.isfinite(pv0)].tolist())
                self._check_p("class_int", t, _neglog10(p0), p0)
            return
        if t.pf is not None:
            a, b, _ = channels(t.pf, self.layout)
            self._check_p("marg_int", t, _neglog10(a), a)
            self._check_p("marg_shape", t, _neglog10(b), b)
        self._preds(t)
        if t.has_tier and math.isfinite(float(t.p[DETECTOR_INDEX["peer"]])):
            _, _, p0 = channels(self._p_class_tier(t), self.layout)
            self._check_p("peer", t, _neglog10(p0), p0)
        if self.density is not None and t.zi is not None:
            s0 = m_density.score_model(self.density, t.zi)
            if s0.q > 0:
                self._check_p("t2", t, _neglog10(s0.p_t2), s0.p_t2)
                self._check_p("spe", t, _neglog10(s0.p_spe), s0.p_spe)

    def _median(self, t: _Tick) -> np.ndarray:
        """Bucket medians (natural units, this tick's exposure and ratio n)
        of the current predictive at tick t, cached per tick."""
        m = self._med.get(t.ts)
        if m is None:
            pc = self._preds(t)["current"]
            if self.cand_idx is not None:
                # only the candidates' medians are used: the Beta-binomial
                # quantile search of the other ratio features is skipped
                c = np.array(pc.c, dtype=np.float64, copy=True)
                off = np.ones(NF, dtype=bool)
                off[list(self.cand_idx)] = False
                c[off] = np.nan
                pc = dataclasses.replace(pc, c=c)
            m = self._med[t.ts] = MB.quantiles(pc, [0.5], dt_s=t.dt, nat=t.nat)[0]
        return m

    def _observed(self, t: _Tick) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        o = self._obs.get(t.ts)
        if o is None:
            o = self._obs[t.ts] = MB.observe(t.nat, t.dt)
        return o

    def _p_class_tier(self, t: _Tick) -> np.ndarray:
        """Factual class-tier mid-p of every feature at tick t (B04's peer
        input; not stored, so recomputed once per tick)."""
        v = self._pk.get(t.ts)
        if v is None:
            _, v = MB.midp(self._preds(t)["class"], t.nat, t.dt)
            self._pk[t.ts] = v
        return v

    def _class_pf(self, t: _Tick) -> np.ndarray:
        preds = self._preds(t)
        _, p_c = MB.midp(preds["current"], t.nat, t.dt)
        if preds["reference"] is not None:
            _, p_r = MB.midp(preds["reference"], t.nat, t.dt)
            return np.asarray(bayes.combine_anchors(p_c, p_r), dtype=np.float64)
        return p_c

    # ----------------------------------------------------------- per tick
    def _p_row(self, t: _Tick, nz: _Neutral, check: bool) -> np.ndarray:
        """Counterfactual p row of tick t: the stored p with the recomputed
        stateless detectors replaced."""
        p = t.p.copy()
        scores, pms, _ = self._numeric(t, nz.features, check)
        if t.qv is not None:
            sq, pq = self._numeric_q(t, nz.features, check)
            scores.update(sq)
            pms.update(pq)
        for d, sc in scores.items():
            p[DETECTOR_INDEX[d]] = self.p_of(d, t, sc, pms.get(d, _NAN))
        self._neutralise(p, nz)
        return p

    def _neutralise(self, p: np.ndarray, nz: _Neutral) -> None:
        """round 4: a neutralised detector reads at least NEUTRAL_P (the median
        of its null) wherever it was scored; NaN stays NaN."""
        for d in nz.detectors:
            i = DETECTOR_INDEX.get(d)
            if i is not None and math.isfinite(float(p[i])):
                p[i] = max(float(p[i]), NEUTRAL_P)     # only ever raises p (monotone)
                self.neutralised.add(d)
        if nz.tokens and m_feedback.token_str(XSYS_DIM, self.s) in nz.tokens:
            # B21 scores one (system, IP) pair: removing the new system of a
            # first access ('token:xsys=<system>', its finding's token)
            # removes the pair's novelty, i.e. cross_system reads its null
            # median at this key (round 4, evaluator)
            i = DETECTOR_INDEX["cross_system"]
            if math.isfinite(float(p[i])):
                p[i] = max(float(p[i]), NEUTRAL_P)
                self.recomputed.add("cross_system")

    def _finding_removed(self, ev: Any, nz: _Neutral) -> bool:
        """A discrete finding does not hold when every new token it carries is
        removed, or when the detector that emitted it is neutralised."""
        toks = set(finding_tokens(ev))
        if toks and toks <= nz.tokens:
            return True
        return bool(set(FINDING_DETECTORS.get(str(ev.kind), ())) & nz.detectors)

    def _risk_hits(self) -> List[Tuple[float, float, Set[str]]]:
        """B27's risk opening needs a family at e_day <= 0.1 newer than the
        key's last incident activity within 24 h: the (ts, window_s, families)
        of those hits before the decision tick, from the stored p_family."""
        if self._rh is not None:
            return self._rh
        last = -math.inf
        for x in (self.inc.evidence or ()):
            if isinstance(x, Mapping) and _f(x.get("ts")) < self.now:
                last = max(last, _f(x.get("ts")))
        for other in self.store.incidents(system=self.s, entity=self.e):
            if other.id != self.inc.id and other.entity == self.e \
                    and float(other.opened) < self.now:
                last = max(last, min(float(other.last_seen), self.now - 1.0))
        lo = max(self.now - RISK_LOOKBACK_S, last)
        out: List[Tuple[float, float, Set[str]]] = []
        n = int(min(2000, math.ceil(RISK_LOOKBACK_S / max(self.dt, 1.0)) + 4))
        for pt in self.store.derived_tail(self.s, self.e, P_FAMILY, n):
            if not (lo < pt.ts <= self.now) or not isinstance(pt.value, Mapping):
                continue
            pdt = float(pt.window_s) if pt.window_s and pt.window_s > 0 else self.dt
            fams = {str(f) for f, v in pt.value.items()
                    if _f(v) == _f(v) and _f(v) * DAY / pdt <= RISK_FAM_E_DAY}
            if fams:
                out.append((float(pt.ts), pdt, fams))
        self._rh = out
        return out

    def _risk_armed(self, nz: _Neutral, wm: Sequence[float]) -> bool:
        """Does B27's risk opening still hold with nz neutralised? The risk
        level itself is held (removing evidence only lowers it, so holding it
        never overstates validity); the family hits that arm the trigger are
        recomputed through the p row and fuse. A hit whose factual recompute
        does not reproduce it is held."""
        if not self.risk_open:
            return False
        hits = self._risk_hits()
        if not hits:
            return True
        for ts, pdt, fams in hits:
            t = self.tick(ts)
            if t is None or t.p is None:
                return True
            fact = self._fam_hits(t, _Neutral(()), wm, pdt)
            if not (fact & fams):
                return True                     # not reproduced: held
            if self._fam_hits(t, nz, wm, pdt):
                return True
        return False

    def _fam_hits(self, t: _Tick, nz: _Neutral, wm: Sequence[float], pdt: float) -> Set[str]:
        p = self._p_row(t, nz, False)
        pf = self._q_canon(t, p, wm)[0] if self.canon else self._q(t, p, wm)[0]
        return {f for f, v in pf.items() if v == v and v * DAY / pdt <= RISK_FAM_E_DAY}

    def _novelty(self, t: _Tick, nz: _Neutral, p: np.ndarray) -> None:
        """novelty -> 0 when every value first seen at the tick is removed."""
        i = DETECTOR_INDEX["novelty"]
        if not nz.tokens or not math.isfinite(float(p[i])):
            return
        evs = self.store.events(self.s, self.e, since=t.ts, kinds=NOVELTY_KINDS)
        toks = set()
        for ev in evs:
            if float(ev.ts) == t.ts:
                toks.update(m_feedback.event_new_tokens(ev))
        prof = self.store.profile(self.s, self.e)
        last = (((prof.extra if prof is not None else {}) or {}).get("categorical") or {}) \
            .get("last") or {}
        k_new = int(_f(last.get("k_new")) if _f(last.get("k_new")) == _f(last.get("k_new"))
                    else 0)
        if toks and toks <= nz.tokens and k_new <= len(toks) and _f(last.get("ts")) == t.ts:
            p[i] = self.p_of("novelty", t, 0.0, _NAN)
            self.recomputed.add("novelty")
        else:
            self.held.add("novelty")

    # ------------------------------------------------------------ evaluate
    def evaluate(self, cands: Sequence[str], check: bool = False,
                 full: bool = False) -> Dict[str, Any]:
        """Recompute the decision at now with `cands` neutralised.

        The trigger is an OR of B25's paths and B27's openers, so the paths
        that need only the current tick (single tick, accumulators, findings,
        a held risk opening) are decided first; the evidence CUSUM, which
        needs the whole excursion re-scored, is replayed only when nothing
        else triggers (or `full`). The evidence path is replayed only when
        it holds factually: neutralising deviations can only raise p (every
        step from a residual to q is monotone), so it cannot switch it on."""
        key = tuple(sorted(cands))
        hit = self._cache.get(key)
        if hit is not None and (hit["complete"] or not full):
            return hit
        nz = _Neutral(cands)
        now_t = self.tick(self.now)
        if now_t is not None and now_t.p is None:
            now_t = None
        out: Dict[str, Any] = {"paths": [], "findings": [], "q_all": _NAN, "q_inst": _NAN,
                               "e_day": _NAN, "S": _NAN, "acc": [], "complete": True}
        acc = {d: int(_f(v) >= 0.5) for d, v in self.acc.items() if d in _ACC_SET}
        p_now: Optional[np.ndarray] = None
        if now_t is not None:
            p_now = self._p_row(now_t, nz, check)
        # --- CUSUM bank / MCUSUM: replay over the accumulator window
        if self._cusum is not None and p_now is not None:
            kf = [f for f in nz.features if f in _KEY_SET]
            zr_by_tick: Dict[float, Dict[int, float]] = {}
            if kf:
                for ts, _inp in self._cusum["rows"]:
                    zm = self.zr_median(ts)
                    vals = {f: zm[f] for f in kf if f in zm}
                    if vals:
                        zr_by_tick[ts] = vals
            cr = self._cusum_run(zr_by_tick) if zr_by_tick else self._fact_cusum
            for d in _B14_DETS:
                i = DETECTOR_INDEX[d]
                if math.isfinite(float(now_t.p[i])):
                    p_now[i] = self.p_of(d, now_t, cr["score"][d], cr["pm"][d])
                    if check:
                        fr = self._fact_cusum
                        self._check_p(d, now_t, fr["score"][d], fr["pm"][d])
                if not self.canon or d in self.acc:
                    # spec v2.1: B14 writes its accumulator flags on H ticks only
                    acc[d] = int(cr["latch"][d])
            self.replayed.update(_B14_DETS)
            out["cusum_max"] = cr["max"]
        # --- off-hours W
        if self._off is not None:
            if nz.offhours:
                r = m_rhythm.offhours_replay(self._off["model"], self._off["slots"],
                                             since=self._off["since"], until=self.now)
                i = DETECTOR_INDEX["offhours"]
                if p_now is not None and math.isfinite(float(now_t.p[i])):
                    p_now[i] = self.p_of("offhours", now_t, r["W_final"],
                                         2.0 ** (-r["W_final"]))
                if self._off["h"] == self._off["h"]:
                    acc["offhours"] = int(r["W_final"] >= self._off["h"])
            self.replayed.add("offhours")
        # --- novelty at now
        if p_now is not None:
            self._novelty(now_t, nz, p_now)
            self._neutralise(p_now, nz)          # after the B14 / off-hours replays
        for d in nz.detectors:
            if d in acc:
                acc[d] = 0
        # --- fuse and meta-calibrate now; paths that need only now
        wm = self._wmult()
        h = self._h_t()
        paths: List[str] = []
        if p_now is not None:
            if self.canon:
                pf, q_inst, q_all, q_h, e_day = self._q_canon(now_t, p_now, wm)
                out.update(q_all=q_all, q_inst=q_inst, q_inst_h=q_h, p_family=pf, p_row=p_now)
            else:
                pf, q_inst, q_all = self._q(now_t, p_now, wm)
                out.update(q_all=q_all, q_inst=q_inst, p_family=pf, p_row=p_now)
                e_day = combine.e_day(q_all, self.dt) if q_all == q_all else _NAN
            if q_all == q_all and e_day == e_day:
                out["e_day"] = e_day
                if e_day <= SINGLE_E_DAY * self.alpha:
                    sev = combine.e_day_severity(e_day, self.alpha)
                    if sev != "low" or q_all <= self._bh(q_all):
                        paths.append("single_tick")
        on = [d for d, v in acc.items() if v]
        if on:
            paths.append("accumulator")
        out["acc"] = sorted(on)
        fnd = [ev for ev in self.findings if not self._finding_removed(ev, nz)]
        out["findings"] = sorted({ev.kind for ev in fnd})
        risk = self._risk_armed(nz, wm) if self.risk_open else False
        out["risk"] = bool(risk)
        out["h"] = h
        quick = bool(paths or fnd or risk)
        # --- spec v2.1: the H-stream evidence CUSUM (S_h) over its excursion
        if self.canon and p_now is not None and self._ev_on_h and (full or not quick):
            hh = self._h_h()
            win_h = self._evidence_window(EVIDENCE_H)
            S = self._s_before(win_h[0], EVIDENCE_H)
            for ts in win_h:
                if ts == self.now:
                    q_h = out.get("q_inst_h", _NAN)
                else:
                    t = self.tick(ts)
                    if t is None or t.p is None:
                        continue
                    p = self._p_row(t, nz, False)
                    q_h = self._q_canon(t, p, wm)[3]
                if q_h == q_h:
                    S = seq.evidence_cusum_step(S, q_h, seq.EVIDENCE_CAP_MULT * hh)
            out["S_h"] = S
            if S >= hh:
                paths.append("evidence_cusum")
        elif self.canon and p_now is not None and self._ev_on_h:
            out["complete"] = False
        # --- evidence CUSUM over its current excursion
        if "evidence_cusum" in paths:
            pass
        elif p_now is not None and self._ev_on and (full or not quick):
            S = self._s_before(self.ev_window[0])
            n_win = len(self.ev_window)
            for j, ts in enumerate(self.ev_window):
                # exact early exit: one tick lowers S by at most the drift 3
                # (-ln q >= 0), so S - 3 (ticks left) >= h keeps the alarm
                if not full and S - seq.EVIDENCE_DRIFT * (n_win - j) >= h:
                    break
                if ts == self.now:
                    q_inst = out["q_inst"]
                else:
                    t = self.tick(ts)
                    if t is None or t.p is None:
                        continue
                    p = self._p_row(t, nz, False)
                    q_inst = (self._q_canon(t, p, wm)[1] if self.canon
                              else self._q(t, p, wm)[1])
                if q_inst == q_inst:
                    S = seq.evidence_cusum_step(S, q_inst, seq.EVIDENCE_CAP_MULT * h)
            out["S"] = S
            if S >= h:
                paths.append("evidence_cusum")
        elif p_now is not None and self._ev_on:
            out["complete"] = False                 # decided without the evidence replay
            out["S"] = _NAN
        elif p_now is not None:
            out["S"] = _scalar(self.store, self.s, self.e, EVIDENCE, self.now)
        out["paths"] = paths
        out["trigger"] = bool(paths or fnd or risk)
        self._cache[key] = out
        return out

    @property
    def _ev_on(self) -> bool:
        """The stored evidence statistic is at or above h at now (B25 wrote it)."""
        if self._ev_on_v is None:
            S = _scalar(self.store, self.s, self.e, EVIDENCE, self.now)
            self._ev_on_v = bool(S == S and S >= self._h_t())
        return self._ev_on_v

    @property
    def _ev_on_h(self) -> bool:
        """spec v2.1: the stored H-stream statistic S_h is at or above h_h at
        now (B25 judges S_h on H decision ticks only)."""
        if not self.canon or not GR.decision(self.now, self.dt, "h", GR.CANONICAL):
            return False
        S = _scalar(self.store, self.s, self.e, EVIDENCE_H, self.now)
        return bool(S == S and S >= self._h_h())

    def _h_t(self) -> float:
        """Threshold of the (T-stream) evidence CUSUM B25 used at now."""
        if self.canon:
            return seq.h_evidence(GR.evidence_arl_ticks("t", self.dt, GR.CANONICAL)) \
                * self.h_mult
        return seq.h_evidence(seq.arl_ticks(EVIDENCE_ARL_DAYS, self.dt)) * self.h_mult

    def _h_h(self) -> float:
        return seq.h_evidence(GR.evidence_arl_ticks("h", self.dt, GR.CANONICAL)) * self.h_mult_h

    def _q_canon(self, t: _Tick, p: np.ndarray, wm: Sequence[float]
                 ) -> Tuple[Dict[str, float], float, float, float, float]:
        """spec v2.1 (B25 canonical phase 1): (p_family, q_t, q_all, q_h,
        e_day) of one p row: provisional Q detectors at half weight, the meta
        strata B25 recorded in its pending entry (else rebuilt from the tick
        type and the grain dayparts), e_day = q_all x n_tau / beta_tau."""
        ts, dtk = t.ts, t.dt_tick
        tau = GR.tick_type(ts, dtk, GR.CANONICAL)
        row = [float(x) for x in p]
        wm2 = list(wm)
        if tau != "t":
            pv = emit.read_dict(self.store, self.s, self.e, PROV, ts)
            for i in _Q_IDX:
                d = DETECTORS[i]
                if 0.0 <= row[i] <= 1.0 and _f(pv.get(d)) < PROV_MIN:
                    wm2[i] = wm2[i] * 0.5
        pf, p_inst, p_all = fuse(row, self.fw, wm2)
        g = t.grain or {}
        pend = self.b25_pend.get(ts)
        cc = timebins.cadence_class(dtk)
        if isinstance(pend, (list, tuple)) and len(pend) >= 7:
            st_all, st_t, st_h = str(pend[2]), str(pend[5]), str(pend[6])
        else:
            dp_t = str(g.get("dp_t", t.strat[0]))
            dp_h = str(g.get("dp_h", dp_t))
            dp_q = str(g.get("dp_q", dp_h))
            dp_tau = dp_h if tau == "h" else dp_q if tau == "q" else dp_t
            st_all = calib.meta_stratum_key(dp_tau, tau, cc if tau == "t" else None)
            st_t = calib.meta_stratum_key(dp_t, "t", cc)
            st_h = calib.meta_stratum_key(dp_h, "h")
        if "|t:" in st_all:
            tau = st_all.rsplit("|t:", 1)[1].split("|", 1)[0] or tau
        p_t = self._stream_p(row, _INST_T_IDX, wm2)
        p_h = (self._stream_p(row, _INST_H_IDX, wm2)
               if GR.decision(ts, dtk, "h", GR.CANONICAL) else _NAN)
        q_t = m_calib.p_value(m_calib.ring(self.meta, META_INST_T, st_t), _meta_score(p_t),
                              m_calib.uniform(self.s, self.e, "meta_inst", ts), p_t)
        q_h = m_calib.p_value(m_calib.ring(self.meta, META_INST_H, st_h), _meta_score(p_h),
                              m_calib.uniform(self.s, self.e, META_INST_H, ts), p_h)
        q_all = m_calib.p_value(m_calib.ring(self.meta, META_ALL, st_all), _meta_score(p_all),
                                m_calib.uniform(self.s, self.e, "meta_all", ts), p_all)
        mult = GR.e_day_mult(tau, dtk, GR.CANONICAL)
        e_day = q_all * mult if q_all == q_all else _NAN
        return pf, q_t, q_all, q_h, e_day

    def _stream_p(self, row: Sequence[float], idx: Sequence[int], wm: Sequence[float]
                  ) -> float:
        """B25's instantaneous p of one stream: fuse() on the stream's
        instantaneous detectors only."""
        r = [_NAN] * N_DETECTORS
        any_ = False
        for i in idx:
            v = row[i]
            if 0.0 <= v <= 1.0:
                r[i] = v
                any_ = True
        if not any_:
            return _NAN
        return fuse(r, self.fw, wm)[1]

    def _q(self, t: _Tick, p: np.ndarray, wm: Sequence[float]
           ) -> Tuple[Dict[str, float], float, float]:
        """(p_family, q_inst, q_all) of one p row: B25's fuse and meta rings."""
        pf, p_inst, p_all = fuse(p, self.fw, wm)
        stratum = calib.stratum_key(t.strat[0], timebins.cadence_class(t.dt))
        q_inst = m_calib.p_value(m_calib.ring(self.meta, "meta_inst", stratum),
                                 _meta_score(p_inst),
                                 m_calib.uniform(self.s, self.e, "meta_inst", t.ts), p_inst)
        q_all = m_calib.p_value(m_calib.ring(self.meta, "meta_all", stratum),
                                _meta_score(p_all),
                                m_calib.uniform(self.s, self.e, "meta_all", t.ts), p_all)
        return pf, q_inst, q_all

    @property
    def _fact_cusum(self) -> Dict[str, Any]:
        """The CUSUM replay without neutralisation (cached)."""
        if self._fc is None:
            self._fc = self._cusum_run({})
        return self._fc

    def _wmult(self) -> List[float]:
        if self._wm is None:
            m = self.store.latest_derived(self.s, SYSTEM_KEY, CALIB_HEALTH)
            chv = m.value if m is not None and isinstance(m.value, Mapping) else None
            self._wm = ([m_calib.weight_mult(chv, d) for d in DETECTORS] if chv
                        else [1.0] * N_DETECTORS)
        return self._wm

    def _bh(self, q_mine: float) -> float:
        keys = self.store.entities(self.s) + [k for k in self.store.pseudo_entities(self.s)
                                              if k.startswith(CLASS_PREFIX)]
        qs = [q_mine]
        for k in keys:
            if k == self.e:
                continue
            v = _scalar(self.store, self.s, k, Q_ALL, self.now)
            if v == v:
                qs.append(v)
        return bh_threshold(qs)

    # ------------------------------------------------------------ search
    def driving_detectors(self) -> List[str]:
        """round 4: the detectors that drove the decision, most extreme first:
        p <= CF_DET_P at the decision tick, an accumulator flag on, the
        emitter of a finding >= MEDIUM, and the detectors below CF_DET_P at
        the family hits that arm a risk opening (min p over those ticks). They
        are counterfactual candidates after the features and tokens: had the
        detector read as the median of its null wherever it was scored in the
        recomputed window (NEUTRAL_P), would the decision still hold?"""
        best: Dict[str, float] = {}

        def take(row: Optional[np.ndarray]) -> None:
            if row is None:
                return
            for i, v in enumerate(np.asarray(row, dtype=np.float64)):
                if v == v and v <= CF_DET_P:
                    d = DETECTORS[i]
                    best[d] = min(best.get(d, 1.0), float(v))

        now_t = self.tick(self.now)
        take(now_t.p if now_t is not None else None)
        for d, v in self.acc.items():
            if d in DETECTOR_INDEX and _f(v) >= 0.5:
                best[d] = min(best.get(d, 1.0), 0.0)
        for ev in self.findings:
            for d in FINDING_DETECTORS.get(str(ev.kind), ()):
                best[d] = min(best.get(d, 1.0), 0.0)
        if self.risk_open:
            for ts, _pdt, _f2 in self._risk_hits():
                t = self.tick(ts)
                take(t.p if t is not None else None)
        if self._ev_on or self._ev_on_h:
            for ts in self.ev_window[-self.max_ticks:]:
                t = self.tick(ts)
                take(t.p if t is not None else None)
        return [d for d, _ in sorted(best.items(), key=lambda kv: (kv[1], kv[0]))][:CF_DETECTORS]

    def explain(self, numeric: Sequence[str], tokens: Sequence[str]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        stored_paths = sorted((self.alarm or {}).get("paths") or
                              ([self.alarm.get("path")] if self.alarm else []))
        cands: List[str] = list(numeric[:CF_CANDIDATES])
        self.key_cands = [FEATURE_INDEX[c] for c in cands
                          if c in FEATURE_INDEX and FEATURE_INDEX[c] in _KEY_SET]
        self.cand_idx = {FEATURE_INDEX[c] for c in cands if c in FEATURE_INDEX}
        cands += [f"token:{t}" for t in tokens]
        if self._off is not None and self._off["slots"]:
            cands.append("offhours")
        cands += [f"detector:{d}" for d in self.driving_detectors()]
        fact = self.evaluate((), check=True, full=True)
        # every accumulator the factual recompute latches drives the decision
        # too (a replayed B14 latch is not in the stored flags between H ticks)
        cands += [f"detector:{d}" for d in fact.get("acc") or ()
                  if f"detector:{d}" not in cands]
        subset: List[str] = []
        n_eval = 1
        if fact["trigger"] and cands:
            subset, n_eval = RP.minimal_set(cands, lambda c: not self.evaluate(c)["trigger"])
        cf = self.evaluate(tuple(subset), full=True) if subset else fact
        blocking: List[str] = []
        if not subset and fact["trigger"]:
            allc = self.evaluate(tuple(cands), full=True) if cands else fact
            pr = allc.get("p_row")
            if pr is not None:
                thr = SINGLE_E_DAY * self.dt / DAY
                blocking = [DETECTORS[i] for i in np.argsort(pr) if math.isfinite(pr[i])
                            and pr[i] <= thr and DETECTORS[i] not in self.recomputed
                            and DETECTORS[i] not in self.replayed]
            blocking += [f"acc:{d}" for d in allc.get("acc") or ()
                         if d not in self.replayed]
            blocking += [f"residual:{d}" for d in allc.get("acc") or () if d in self.replayed]
            if pr is not None:
                thr = SINGLE_E_DAY * self.dt / DAY
                blocking += [f"residual:{DETECTORS[i]}" for i in np.argsort(pr)
                             if math.isfinite(pr[i]) and pr[i] <= thr
                             and (DETECTORS[i] in self.recomputed)]
            blocking += [f"path:{x}" for x in allc.get("paths") or ()]
            blocking += [f"finding:{k}" for k in allc.get("findings") or ()]
            if allc.get("risk"):
                blocking.append("risk")
        valid = bool(fact["trigger"] and subset and not cf["trigger"])
        cz = self._cusum
        fid = {
            "decision_match": sorted(fact["paths"]) == sorted(stored_paths)
            if self.alarm is not None else not fact["paths"],
            "stored_paths": stored_paths, "recomputed_paths": list(fact["paths"]),
            "p_log10_err": _r(self.fid["p_log10_err"], 3), "n_p_checked": self.fid["n_p"],
            "inexact": dict(self.fid.get("inexact") or {}),
            "anchored": dict(self.fid.get("anchored") or {}),
            "q_all_stored": _r(_scalar(self.store, self.s, self.e, Q_ALL, self.now)),
            "q_all_recomputed": _r(fact["q_all"]),
        }
        if cz is not None:
            fid["cusum"] = {"max_err": _r(cz["info"]["max_err"], 3),
                            "n_reset": cz["info"]["n_reset"],
                            "n_resync": cz["info"]["n_resync"],
                            "window_complete": bool(cz["complete"])}
        if self._off is not None:
            fid["offhours_W_err"] = _r(self._off["err"], 3)
        if valid:
            reason = "decision flips"
        elif not fact["trigger"]:
            reason = "factual recompute has no trigger"
        elif not subset:
            reason = "no candidate set flips the decision"
        else:
            reason = "neutralised decision still triggers"
        tn = self.ticks.get(self.now)
        held = sorted(self.held | {DETECTORS[i] for i in range(N_DETECTORS)
                                   if tn is not None and tn.p is not None
                                   and math.isfinite(float(tn.p[i]))}
                      - self.recomputed - self.replayed)
        window = [self.ev_window[0] if (self.ev_window and self._ev_on) else self.now,
                  self.now]
        if cz is not None:
            window[0] = min(window[0], float(cz["t0"]))
        scope = {"recomputed": sorted(self.recomputed), "replayed": sorted(self.replayed),
                 "held": held, "neutralised": sorted(d for d in self.neutralised
                                                     if f"detector:{d}" in subset),
                 "window": window,
                 "ticks": {"evidence": len(self.ev_window) if self._ev_on else 0,
                           "cusum": len(cz["rows"]) if cz is not None else 0},
                 "held_triggers": (["risk_level"] if self.risk_open else []) + sorted(
                     {ev.kind for ev in self.findings
                      if not m_feedback.event_new_tokens(ev)}),
                 "grains": sorted({_STREAM[DETECTOR_INDEX[d]] for d in
                                   self.recomputed | self.replayed
                                   if _STREAM[DETECTOR_INDEX[d]] in ("h", "q")}
                                  | ({"h"} if self.canon and self.replayed & set(_B14_DETS)
                                     else set())) if self.canon else [],
                 "fidelity": fid, "evaluations": n_eval, "blocking": blocking,
                 "ms": round(1000.0 * (time.perf_counter() - t0), 2)}
        return {"set": subset, "valid": valid, "reason": reason, "scope": scope,
                "factual": _jsonable_decision(fact), "neutralised": _jsonable_decision(cf),
                "candidates": cands}


def finding_tokens(ev: Any) -> List[str]:
    """New categorical values a discrete finding rests on: its new tokens
    (m_feedback.event_new_tokens), and for B09's client events the new stack
    ('stack=<id>'): removing that stack removes the finding."""
    out = list(m_feedback.event_new_tokens(ev))
    ex = getattr(ev, "extra", None) or {}
    if not out and str(getattr(ev, "kind", "")) in ("client_change", "client_impersonation") \
            and isinstance(ex, Mapping) and ex.get("stack"):
        out.append(m_feedback.token_str("stack", ex["stack"]))
    return out


def _jsonable_decision(d: Mapping[str, Any]) -> Dict[str, Any]:
    return {"trigger": bool(d.get("trigger")), "paths": list(d.get("paths") or []),
            "acc": list(d.get("acc") or []), "findings": list(d.get("findings") or []),
            "q_all": _r(d.get("q_all")), "q_inst": _r(d.get("q_inst")),
            "e_day": _r(d.get("e_day")), "evidence": _r(d.get("S")), "h": _r(d.get("h")),
            "cusum_max": _r(d.get("cusum_max"), 4)}


def _last_regime(store, s: str, e: str) -> Any:
    m = store.latest_derived(s, e, "behavior.regime")
    return None if m is None else m.value


# ================================================================== engine
class ExplainEngine(Engine):
    name = "behavior.explain"
    layer = "behavior"
    consumes = ["store.incidents", "profile.extra.model_state", "profile.extra.categorical",
                "profile.extra.client_stacks", "profile.extra.sequence",
                "profile.extra.class_monitor", NAT, TCTX, Z, ZR, ZI, PF, P, SCORE, PM,
                m_cp.CUSUM_STATE, Q_INST, Q_ALL, EVIDENCE, ALARM, P_FAMILY, RISK,
                "behavior.common", COMMON_FLAG, CLASS_AGG, "act.stream", STACK_SET,
                "model.baseline", "model.density", "model.groups", "model.cp", "model.calib",
                "model.rhythm", "model.vocab", "model.client", "model.seq", "model.class",
                "model.classagg", "model.feedback", "store.events", "store.labels"]
    produces = ["incident.explanation", "incident.narrative"]
    description = ("Faithful incident explanations: natural-unit attribution against the "
                   "bucket's normal range, RBC / Garthwaite-Koch shares with BH, new and "
                   "vanished tokens, client-stack diff, unlikely transitions, a minimal "
                   "counterfactual recomputed by deterministic replay (CUSUM bank, MCUSUM, "
                   "off-hours W) through calibration, fusion and the decision rule, peer "
                   "context, nearest labelled pattern, zh/en narrative.")
    interval = 1

    def __init__(self, max_window_ticks: int = MAX_WINDOW_TICKS, **params: Any) -> None:
        super().__init__(**params)
        self.max_window_ticks = int(max_window_ticks)
        self.last_ms: List[float] = []

    # ------------------------------------------------------------------ run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        if ctx.training:
            return 0
        store = ctx.store
        now = float(ctx.now)
        dt = float(ctx.window_s)
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError(f"explain: ctx.window_s={ctx.window_s!r} is not a positive cadence")
        n = 0
        self.last_ms = []
        # since: the incidents active now plus those opened within the last
        # hour (the H opening-row refresh of _due)
        for inc in store.incidents(since=now - GR.GRAIN_S["h"] - dt):
            if inc.status == "closed" or not self._due(inc, now, dt):
                continue
            t0 = time.perf_counter()
            expl = self.explain(store, inc, now, dt, ctx.config or {})
            inc.explanation = expl
            inc.narrative = expl.get("narrative_zh") or expl.get("narrative_en") or ""
            store.put_incident(inc)
            self.last_ms.append(1000.0 * (time.perf_counter() - t0))
            n += 1
        return n

    @staticmethod
    def _due(inc: Any, now: float, dt: float = 900.0) -> bool:
        """Opened, reopened or escalated this tick (a severity rise or a new
        axis since the last explanation), or (round 4) the first H decision
        tick after the opening when the explanation has no H opening row yet
        (the trailing hour that contains the opening is scored there)."""
        ex = inc.explanation if isinstance(inc.explanation, Mapping) else {}
        if not ex or not ex.get("ts"):
            return True
        oe = ex.get("opening_evidence")
        if isinstance(oe, Mapping) and "h" not in (oe.get("grains") or {}) \
                and float(inc.opened) < now <= float(inc.opened) + GR.GRAIN_S["h"] + 1e-6 \
                and GR.decision(now, dt, "h", GR.CANONICAL):
            return True
        if float(inc.last_seen) < now and float(inc.opened) < now:
            return False
        for x in reversed(inc.evidence or ()):
            if not isinstance(x, Mapping) or _f(x.get("ts")) < now:
                break
            if x.get("source") in ("b27", "feedback") and x.get("state") in (
                    "open", "reopen", "promoted", "escaped"):
                return True
        if _sev_rank(inc.severity) > _RANK.get(str(ex.get("severity", "low")), 0):
            return True
        return bool(set(inc.axes or ()) - set(ex.get("axes") or ()))

    # --------------------------------------------------------------- explain
    def explain(self, store, inc: Any, now: float, dt: float,
                config: Mapping[str, Any]) -> Dict[str, Any]:
        s, k = inc.system, inc.entity
        canon = GR.canonical(config)
        grain = self._driving_grain(store, s, k, inc, now) if canon else None
        t_star = self._trigger_tick(store, s, k, inc, now, grain)
        tc = _tctx_at(store, s, k, t_star, config, dt)
        if grain is not None:
            tc = GR.row_tctx(t_star, grain, dt, config)       # the grain row's midpoint
        day_en, day_zh = _day_label(tc)
        prev = inc.explanation if isinstance(inc.explanation, Mapping) else {}
        opening = None
        if is_class(k):
            attrs, deviation = self._class_attributions(store, s, k, t_star, dt, tc,
                                                        grain=grain)
        else:
            opening = self._opening_evidence(store, s, k, inc, now, prev) if canon else None
            got = self._opening_attributions(store, s, k, dt, config, opening, prev)
            if got is not None:
                attrs, deviation, tc_o = got
                day_en, day_zh = _day_label(tc_o)
            else:
                attrs, deviation = self._attributions(store, s, k, t_star, dt, tc, grain=grain)
        # (evaluator round 4) an incident opened by P03 pattern violations:
        # its first reason is the violated constraint of the learned pattern
        pv = None if is_class(k) else self._violation_attr(store, inc, now)
        if pv is not None:
            attrs = [pv] + [a for a in attrs if a.get("feature") != pv["feature"]]
        new_tokens = self._new_tokens(store, inc, now, dt)
        vanished = [] if is_class(k) else self._vanished(store, s, k, inc, now)
        stacks = {} if is_class(k) else self._stack_diff(store, s, k, inc, now, dt)
        transitions = [] if is_class(k) else self._transitions(store, s, k, t_star)
        # ---- counterfactual of the opening decision (engines.md B29 step 4:
        # "stop when the incident's opening condition no longer holds")
        t_open = self._open_tick(inc, now)
        prev_cf = prev.get("counterfactual") if isinstance(prev, Mapping) else None
        if isinstance(prev_cf, Mapping) and prev_cf.get("decision_ts") == t_open:
            # an escalation: the opening decision's counterfactual was computed
            # at the opening tick, with that tick's model snapshots and while
            # its inputs were retained; it is the answer, carried unchanged
            cf = dict(prev_cf)
            cf["carried_from"] = prev.get("ts")
            return self._finish(store, inc, now, dt, s, k, t_star, tc, day_en, day_zh, attrs,
                                deviation, new_tokens, vanished, stacks, transitions, cf,
                                opening)
        dt_open = dt
        strat = m_calib.issued_stratum(store.get_model(s, k, m_calib.MODEL), t_open)
        if strat is not None and strat[2] > 0:
            dt_open = float(strat[2])
        if canon:
            # spec v2.1: candidates from the grain rows scored at (or last
            # before) the opening tick, both grains, by |z|
            cand: Dict[str, float] = {}
            for g in ("h", "q"):
                tg = self._trigger_tick(store, s, k, inc, t_open, g)
                if is_class(k):
                    ao, _ = self._class_attributions(store, s, k, tg, dt_open, tc,
                                                     bands=False, grain=g)
                else:
                    ao, _ = self._attributions(store, s, k, tg, dt_open, tc, bands=False,
                                               grain=g)
                for a in ao:
                    if a.get("z") is not None and abs(float(a["z"])) >= CF_Z_MIN:
                        cand[a["feature"]] = max(cand.get(a["feature"], 0.0), abs(float(a["z"])))
            numeric = sorted(cand, key=lambda n: -cand[n])
        else:
            if t_open == t_star:
                attrs_open = attrs
            elif is_class(k):
                attrs_open, _ = self._class_attributions(store, s, k, t_open, dt_open, tc,
                                                         bands=False)
            else:
                attrs_open, _ = self._attributions(store, s, k, t_open, dt_open, tc, bands=False)
            numeric = [a["feature"] for a in attrs_open
                       if a.get("z") is not None and abs(float(a["z"])) >= CF_Z_MIN]
        toks_open = []
        for ev in store.events(s, k, since=t_open, kinds=NOVELTY_KINDS, limit=1000):
            if float(ev.ts) == t_open:
                toks_open.extend(m_feedback.event_new_tokens(ev))
        for ev in store.events(s, k, since=t_open, kinds=sorted(DISCRETE_KINDS), limit=1000):
            if float(ev.ts) == t_open and ev.kind not in NOVELTY_KINDS:
                toks_open.extend(finding_tokens(ev))
        depth = int((now - t_open) / max(dt, 1.0)) + 8
        try:
            cfo = Counterfactual(store, s, k, t_open, dt_open, config, inc,
                                 self.max_window_ticks, depth=depth)
            cf = cfo.explain(numeric, list(dict.fromkeys(toks_open)))
            cf["decision_ts"] = t_open
        except Exception as exc:                         # never lose the rest of the explanation
            if (config or {}).get("strict"):
                raise
            cf = {"set": [], "valid": False, "reason": f"error: {type(exc).__name__}: {exc}",
                  "scope": {"recomputed": [], "replayed": [], "held": []},
                  "decision_ts": t_open}
        return self._finish(store, inc, now, dt, s, k, t_star, tc, day_en, day_zh, attrs,
                            deviation, new_tokens, vanished, stacks, transitions, cf, opening)

    def _finish(self, store, inc: Any, now: float, dt: float, s: str, k: str, t_star: float,
                tc: Mapping[str, Any], day_en: str, day_zh: str,
                attrs: List[Dict[str, Any]], deviation: List[float],
                new_tokens: List[Dict[str, Any]], vanished: List[Dict[str, Any]],
                stacks: Dict[str, Any], transitions: List[Dict[str, Any]],
                cf: Dict[str, Any], opening: Optional[Dict[str, Any]] = None
                ) -> Dict[str, Any]:
        """Assemble the explanation dict and its narrative."""
        peer = self._peer_context(store, s, k, now, dt)
        nearest = self._nearest(store, s, inc, deviation)
        top = next((a for a in attrs if a.get("range")), None)
        if top is None:
            top = self._fallback_range(store, s, k, t_star, dt, tc)
            if top is not None:
                attrs = attrs + [top]
        natural_range = None
        if top is not None:
            natural_range = {"feature": top["feature"], "observed": top.get("observed_text"),
                             "usual": top["range"], "when": day_en}
        expl: Dict[str, Any] = {
            "ts": now, "version": 1, "t_star": t_star, "severity": _sev_name(inc.severity),
            "axes": sorted(inc.axes or ()), "when": {"en": day_en, "zh": day_zh},
            "attributions": attrs, "natural_range": natural_range,
            "new_tokens": new_tokens, "vanished": vanished, "client_stacks": stacks,
            "transitions": transitions,
            "counterfactual_set": list(cf.get("set") or []),
            "counterfactual_valid": bool(cf.get("valid")),
            "counterfactual_scope": cf.get("scope") or {},
            "counterfactual": cf,
            "peer_context": peer, "nearest_pattern": nearest,
            "risk": _r(_scalar(store, s, k, RISK, now), 4),     # 1-element vec ring (R5.3)
            "deviation": deviation,
            "p_by_detector": self._pbd(store, s, k, t_star, dt),
        }
        if opening:
            expl["opening_evidence"] = opening
        zh, en, parts = self._narrative(inc, expl, day_en, day_zh)
        expl.update(narrative_zh=zh, narrative_en=en, headline_zh=parts["h_zh"],
                    headline_en=parts["h_en"], bullets_zh=parts["b_zh"], bullets_en=parts["b_en"])
        return expl

    # ------------------------------------------------------------ helpers
    @classmethod
    def _violation_attr(cls, store, inc: Any, now: float) -> Optional[Dict[str, Any]]:
        """The violated constraint of the P03 pattern violation that opened (or
        last reopened) the incident, as its first attribution: feature
        'conf_<type>' (the P03 detector: who / when / content / seq / novel),
        the constraint (`constraint`: the attribute, the window, the
        predecessor), observed vs expected (`observed_text`, `range`) and the
        finding's statement. Why (evaluator round 4): under the progressive
        decision chain the numeric attributions of such incidents were empty
        and the explanation fell back to the entity's request count ('操作次数
        n/a，常态 0–58') - PG6's 'B29 top reason = violated constraint' was 0
        on every seed of rounds 2-3 although P03's finding says exactly what
        was violated ('body.kv.username 与已学到的绑定不符')."""
        t_open = cls._open_tick(inc, now)
        best = None
        for x in inc.evidence or ():
            if not isinstance(x, Mapping) or x.get("source") != "event" \
                    or x.get("kind") != "pattern_violation" or _f(x.get("ts")) > now:
                continue
            key = (abs(_f(x.get("ts")) - t_open), -_sev_rank(x.get("severity")))
            if best is None or key < best[0]:
                best = (key, x)
        if best is None:
            return None
        try:
            ev = store.get_event(str(best[1].get("event_id")))
        except Exception:
            ev = None
        ex = (ev.extra if ev is not None and isinstance(ev.extra, Mapping) else None) or {}
        typ = str(ex.get("type") or "")
        if not typ:
            return None
        obs, exp = ex.get("observed"), ex.get("expected")
        return {"feature": f"conf_{typ}", "group": "pattern", "share": 1.0, "z": None,
                "p": _r(_f(ex.get("p")), 6), "rbc": None, "bh": True,
                "pf": _r(_f(ex.get("p_day")), 6),
                "constraint": str(obs) if typ == "content" else typ,
                "observed": obs, "observed_text": str(obs) if obs is not None else "n/a",
                "range": str(exp) if exp not in (None, "") else None,
                "flags": list(ex.get("flags") or []), "pattern_id": ex.get("pattern_id"),
                "route": ex.get("route"), "statement_zh": ex.get("statement_zh"),
                "statement_en": ex.get("statement_en"), "unit": ""}

    @staticmethod
    def _open_tick(inc: Any, now: float) -> float:
        """The tick the incident last opened or reopened (B27's evidence mark),
        else its opened time."""
        for x in reversed(inc.evidence or ()):
            if isinstance(x, Mapping) and x.get("source") == "b27" \
                    and x.get("state") in ("open", "reopen") and _f(x.get("ts")) <= now:
                return float(x["ts"])
        return min(float(inc.opened), now)

    @staticmethod
    def _opening_evidence(store, s: str, k: str, inc: Any, now: float,
                          prev: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """round 4: calibrated per-feature evidence of the decision rows that
        cover the incident's FIRST opening, per grain: the first H / Q row at or
        after inc.opened (the trailing grain that contains the opening),
        e_f = -log10 pf_f minus the entity's pre-incident q90 of -log10 pf_f
        (retained rows before opened - 1 h, >= 8 of them; a feature that is
        routinely extreme for this entity is discounted by its own level).

        Why: hit@3 was 0.1-0.25 because the attributions ranked Garthwaite-Koch
        shares of the whitened residual at the LATEST explained tick (often an
        escalation days after the opening) instead of the evidence that opened
        the incident; on pack A seed 0 the perturbed feature was in the top 3
        of this ranking for 10 of 14 threat incidents against 2 of 14 before.
        behavior.pf is kept 6 h, so a grain found once is carried in the
        explanation (prev['opening_evidence']) and later explanations reuse it."""
        t1 = float(inc.opened)
        carried = prev.get("opening_evidence") if isinstance(prev, Mapping) else None
        out: Dict[str, Any] = {"t_opened": t1, "grains": {}}
        if isinstance(carried, Mapping) and _f(carried.get("t_opened")) == t1:
            out["grains"].update({g: dict(v) for g, v in (carried.get("grains") or {}).items()})
        for g, suf in (("h", ""), ("q", ".q")):
            if g in out["grains"]:
                continue
            t, M = store.vec_range(s, k, PF + suf, t1 - DAY, now)
            if not len(t):
                continue
            t = np.asarray(t, dtype=np.float64)
            M = np.asarray(M, dtype=np.float64).reshape(len(t), -1)
            idx = np.flatnonzero((t >= t1 - 1e-6) & (t <= t1 + GR.GRAIN_S[g] + 1e-6))
            if not idx.size:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                L = -np.log10(np.clip(M, 1e-300, 1.0))
            base = L[t < t1 - HOUR]
            b = np.zeros(NF)
            if base.shape[0] >= 8:
                with warnings.catch_warnings():         # an all-NaN feature column
                    warnings.simplefilter("ignore", RuntimeWarning)
                    b = np.nanquantile(base, OPEN_EV_BASE_Q, axis=0)
                b = np.where(np.isfinite(b), b, 0.0)
            ev = L[idx[0]] - b
            out["grains"][g] = {"ts": float(t[idx[0]]),
                                "ev": [_r(x, 3) if math.isfinite(x) else None for x in ev]}
        if "findings" not in out:
            fnd: Dict[str, str] = dict((carried or {}).get("findings") or {}) \
                if isinstance(carried, Mapping) and _f(carried.get("t_opened")) == t1 else {}
            for ev in store.events(s, k, since=t1, kinds=sorted(DISCRETE_KINDS), limit=1000):
                if float(ev.ts) != t1 or ev.status == "suppressed" \
                        or _sev_rank(ev.severity) < OPEN_FINDING_RANK:
                    continue
                key = str(ev.kind)
                if key in NOVELTY_KINDS:
                    ex = ev.extra if isinstance(ev.extra, Mapping) else {}
                    key = f"novel:{ex.get('dim')}"
                for f in FINDING_FEATURES.get(key, ()):
                    fnd.setdefault(f, str(ev.kind))
            out["findings"] = fnd
        return out if (out["grains"] or out.get("findings")) else None

    def _opening_attributions(self, store, s: str, k: str, dt: float,
                              config: Mapping[str, Any], opening: Optional[Mapping[str, Any]],
                              prev: Mapping[str, Any]
                              ) -> Optional[Tuple[List[Dict[str, Any]], List[float],
                                                  Mapping[str, Any]]]:
        """Attributions of the opening rows ranked by their evidence (max over
        grains) on the grain row holding the strongest evidence, recomputed
        while that row is retained, else carried from the previous explanation
        of the same opening."""
        if not opening:
            return None
        gr = opening["grains"]
        if not gr:
            return None
        vecs = {g: np.asarray([_f(x) for x in v["ev"]], dtype=np.float64) for g, v in gr.items()}
        rank = np.full(NF, -np.inf)
        for v in vecs.values():
            rank = np.fmax(rank, np.where(np.isfinite(v), v, -np.inf))
        # a decisive finding outranks every numeric deviation of the opening
        top = float(np.max(rank)) if np.isfinite(rank).any() else 0.0
        for f in (opening.get("findings") or {}):
            if f in FEATURE_INDEX:
                i = FEATURE_INDEX[f]
                rank[i] = max(rank[i], max(top, 0.0) + 1.0)
        rank = np.where(np.isfinite(rank), rank, np.nan)
        g_best = max(vecs, key=lambda g: (np.nanmax(np.where(np.isfinite(vecs[g]), vecs[g],
                                                             -np.inf)), g))
        ts = float(gr[g_best]["ts"])
        tc = GR.row_tctx(ts, g_best, dt, config)
        attrs, dev = self._attributions(store, s, k, ts, dt, tc, grain=g_best, rank=rank)
        if attrs:
            for a in attrs:
                a["evidence"] = _r(rank[FEATURE_INDEX[a["feature"]]], 3)
            return attrs, dev, tc
        pa = prev.get("attributions") if isinstance(prev, Mapping) else None
        po = prev.get("opening_evidence") if isinstance(prev, Mapping) else None
        if pa and isinstance(po, Mapping) and _f(po.get("t_opened")) == _f(opening["t_opened"]):
            return list(pa), list(prev.get("deviation") or [0.0] * NF), tc
        return None

    @staticmethod
    def _driving_grain(store, s: str, k: str, inc: Any, now: float) -> str:
        """spec v2.1 (cadence.md §9.4): 'q' when a Q detector holds the
        smallest p of the grain detectors over the incident (its p rows up to
        now), else 'h'."""
        lo = min(float(inc.opened), now) - 1.0
        t, M = store.vec_range(s, k, P, lo, now)
        best = {"h": math.inf, "q": math.inf}
        for row in M[-96:]:
            for i, st in enumerate(_STREAM):
                v = float(row[i])
                if st in best and v == v and v < best[st]:
                    best[st] = v
        return "q" if best["q"] < best["h"] else "h"

    @staticmethod
    def _trigger_tick(store, s: str, k: str, inc: Any, now: float,
                      grain: Optional[str] = None) -> float:
        if grain is None:
            name = CLASS_AGG if is_class(k) else Z
        elif is_class(k):
            name = CLASS_AGG_H
        else:
            name = Z + (".q" if grain == "q" else "")
        if store.vec_at(s, k, name, now) is not None:
            return now
        lo = min(float(inc.opened), now) - 1.0
        t, _ = store.vec_range(s, k, name, lo, now)
        return float(t[-1]) if len(t) else now

    def _fallback_range(self, store, s: str, k: str, ts: float, dt: float,
                        tc: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """A natural-unit range when no attribution carries one (no scored
        row at the trigger tick, e.g. an incident opened by a discrete
        finding on an idle entity): the activity feature 'intensity' (else the
        first key feature with a band) against this bucket's band."""
        try:
            if is_class(k):
                model = store.get_model(s, k, CLASSAGG_MODEL)
                if not isinstance(model, Mapping) or model.get("current") is None:
                    return None
                bands = MB.quantiles(MB.anchor_predictive(model["current"], tc), STATE_QS,
                                     dt_s=BAND_DT_S)
                nat = _vec(store, s, k, CLASS_AGG, ts)
            else:
                bands = self._bands(store, s, k, tc)
                nat = _vec(store, s, k, NAT, ts)
        except Exception:
            return None
        if bands is None:
            return None
        for name in ["intensity"] + [FEATURE_NAMES_V2[i] for i in KEY_FEATURE_IDX]:
            i = FEATURE_INDEX[name]
            if math.isfinite(bands[0, i]) and math.isfinite(bands[2, i]):
                nan = np.full(NF, np.nan)
                item = self._attr_item(i, nan, nan, nan, np.zeros(NF, dtype=bool), None, None,
                                       nat, bands, dt)
                item["share"] = None
                item["fallback"] = True
                if item.get("range"):
                    return item
        return None

    @staticmethod
    def _pbd(store, s: str, k: str, ts: float, dt: float) -> Dict[str, float]:
        row = store.vec_at(s, k, P, ts)
        if row is None:
            return {}
        p = np.asarray(row, dtype=np.float64)
        ok = [i for i in np.argsort(p) if math.isfinite(p[i])][:6]
        return {DETECTORS[i]: _r(p[i], 4) for i in ok}

    def _bands(self, store, s: str, e: str, tc: Mapping[str, Any],
               grain: Optional[str] = None) -> Optional[np.ndarray]:
        """[3, 52] p5 / p50 / p95 per 15 min for tc's bucket: model_state when it
        describes this bucket, else m_baseline.quantiles on the predictive.
        spec v2.1: grain 'h' / 'q' gives the grain's band (per hour / per
        15 min) from model_state['grains'][grain], else its predictive."""
        prof = store.profile(s, e)
        ms = ((prof.extra if prof is not None else {}) or {}).get("model_state") or {}
        if grain is not None:
            ms = ((ms.get("grains") or {}).get(grain) or {}) if isinstance(ms, Mapping) else {}
            if grain == "q":
                pred = MB.predictive_q(store, s, e, tc)["current"]
            else:
                pred = MB.predictive_set(store, s, e, tc)["current"]
            if ms and int(_f(ms.get("bucket"))) == int(pred.bucket) and ms.get("features"):
                out = np.full((3, NF), np.nan)
                for i, n in enumerate(FEATURE_NAMES_V2):
                    v = ms["features"].get(n)
                    if v is not None and len(v) == 3:
                        out[:, i] = [_f(x) for x in v]
                return out
            return MB.quantiles(pred, STATE_QS, dt_s=GR.GRAIN_S[grain])
        pred = MB.predictive_set(store, s, e, tc)["current"]
        if ms and int(_f(ms.get("bucket"))) == int(pred.bucket) and ms.get("features"):
            out = np.full((3, NF), np.nan)
            for i, n in enumerate(FEATURE_NAMES_V2):
                v = ms["features"].get(n)
                if v is not None and len(v) == 3:
                    out[:, i] = [_f(x) for x in v]
            return out
        return MB.quantiles(pred, STATE_QS, dt_s=BAND_DT_S)

    def _attributions(self, store, s: str, e: str, ts: float, dt: float,
                      tc: Mapping[str, Any], bands: bool = True, grain: Optional[str] = None,
                      rank: Optional[np.ndarray] = None
                      ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Numeric attribution of the row scored at ts. spec v2.1: grain
        'h' / 'q' reads that grain's rows (feature.nat.<g>, behavior.z[.q],
        zi, pf), its density and its band; values are per grain exposure."""
        suf = ".q" if grain == "q" else ""
        z = _vec(store, s, e, Z + suf, ts)
        zi = _vec(store, s, e, ZI + suf, ts)
        pf = _vec(store, s, e, PF + suf, ts)
        band_s = BAND_DT_S
        if grain is not None:
            nat = _vec(store, s, e, f"feature.nat.{grain}", ts)
            dt = band_s = GR.GRAIN_S[grain]
        else:
            nat = _vec(store, s, e, NAT, ts)
        if z is None:
            return [], [0.0] * NF
        if grain == "q":
            dens = store.get_model(s, e, DENSITY_Q)
            dens = dens if isinstance(dens, Mapping) and m_density.is_fitted(dens) else None
        else:
            dens = m_density.get(store, s, e)
        zz = zi if zi is not None and np.isfinite(zi).any() else z
        share, p_rbc, rbc = gk_shares(dens, zz)
        if share is None:
            fin = np.isfinite(z)
            tot = float(np.sum(np.where(fin, z * z, 0.0)))
            share = np.where(fin, z * z / tot, np.nan) if tot > 0 else np.full(NF, np.nan)
            p_rbc = pf if pf is not None else np.full(NF, np.nan)
            rbc = np.full(NF, np.nan)
        bh = bh_flags(p_rbc, BH_ALPHA)
        bands = self._bands(store, s, e, tc, grain) if bands else None
        if rank is not None:
            # round 4: ranked by the calibrated per-feature evidence of the
            # incident's decision rows (evidence_rank); share breaks ties
            r_ = np.nan_to_num(np.asarray(rank, dtype=np.float64), nan=-np.inf)
            order = [i for i in sorted(range(NF), key=lambda i: (-r_[i], -np.nan_to_num(
                share[i], nan=-1.0), i)) if math.isfinite(r_[i])]
        else:
            order = [i for i in np.argsort(-np.nan_to_num(share, nan=-1.0), kind="stable")
                     if math.isfinite(share[i])]
        out: List[Dict[str, Any]] = []
        for i in order[:TOP_ATTR]:
            it = self._attr_item(i, share, p_rbc, rbc, bh, z, pf, nat, bands, dt, band_s)
            if grain is not None:
                it["grain"] = grain
            out.append(it)
        dev = [round(float(v), 3) if math.isfinite(v) else 0.0 for v in z]
        return out, dev

    @staticmethod
    def _attr_item(i: int, share, p_rbc, rbc, bh, z, pf, nat, bands, dt,
                   band_s: float = BAND_DT_S) -> Dict[str, Any]:
        name = FEATURE_NAMES_V2[i]
        _, _, per15 = feature_unit(name)
        obs = _NAN if nat is None else float(nat[i])
        if per15 and obs == obs:
            obs = obs * band_s / dt
        lo, mid, hi = (float(bands[0, i]), float(bands[1, i]), float(bands[2, i])) \
            if bands is not None else (_NAN, _NAN, _NAN)
        rng = fmt_range(name, obs, lo, mid, hi, band_s)
        item = {"feature": name, "group": FEATURE_GROUP[name], "share": _r(share[i]),
                "z": _r(z[i], 3) if z is not None else None,
                "p": _r(p_rbc[i]) if p_rbc is not None else None,
                "rbc": _r(rbc[i]) if rbc is not None else None, "bh": bool(bh[i]),
                "pf": _r(pf[i]) if pf is not None else None,
                "observed": _r(obs), "usual": [_r(lo), _r(mid), _r(hi)],
                "observed_text": rng["observed_text"], "unit": rng["unit"]}
        if rng["range"]:
            item["range"] = rng["range"]
        if rng["ratio"] == rng["ratio"]:
            item["ratio"] = _r(rng["ratio"], 3)
        return item

    def _class_attributions(self, store, s: str, k: str, ts: float, dt: float,
                            tc: Mapping[str, Any], bands: bool = True,
                            grain: Optional[str] = None
                            ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """spec v2.1: in canonical mode the class row is B18's H row
        (behavior.class.agg.h, per hour; there is no Q class row)."""
        band_s = BAND_DT_S
        if grain is not None:
            agg = _vec(store, s, k, CLASS_AGG_H, ts)
            dt = band_s = GR.GRAIN_S["h"]
        else:
            agg = _vec(store, s, k, CLASS_AGG, ts)
        model = store.get_model(s, k, CLASSAGG_MODEL)
        if agg is None or not isinstance(model, Mapping) or model.get("current") is None:
            return [], [0.0] * NF
        pc = MB.anchor_predictive(model["current"], tc)
        u, p = MB.midp(pc, agg, dt)
        with np.errstate(all="ignore"):
            z = sp.ndtri(np.clip(u, bayes.PHI_CLIP, 1.0 - bayes.PHI_CLIP))
        z = np.where(np.isfinite(p), z, np.nan)
        fin = np.isfinite(z)
        tot = float(np.sum(np.where(fin, z * z, 0.0)))
        share = np.where(fin, z * z / tot, np.nan) if tot > 0 else np.full(NF, np.nan)
        bh = bh_flags(p, BH_ALPHA)
        want_bands = bands
        prof = store.profile(s, k)
        cm = ((prof.extra if prof is not None else {}) or {}).get("class_monitor") or {}
        bands = None
        aggq = (cm.get("aggregate") or {}).get("features") or {}
        if _f((cm.get("aggregate") or {}).get("exposure_s", BAND_DT_S)) != band_s:
            aggq = {}
        if aggq and int(_f(cm.get("bucket"))) == int(tc.get("bin48", -1)):
            bands = np.full((3, NF), np.nan)
            for n, v in aggq.items():
                if n in FEATURE_INDEX and v is not None and len(v) == 3:
                    bands[:, FEATURE_INDEX[n]] = [_f(x) for x in v]
        if want_bands:
            q = MB.quantiles(pc, STATE_QS, dt_s=band_s, nat=agg)
            bands = q if bands is None else np.where(np.isfinite(bands), bands, q)
        order = [i for i in np.argsort(-np.nan_to_num(share, nan=-1.0), kind="stable")
                 if math.isfinite(share[i])]
        out = [self._attr_item(i, share, p, np.full(NF, np.nan), bh, z, p, agg, bands, dt,
                               band_s)
               for i in order[:TOP_ATTR]]
        dev = [round(float(v), 3) if math.isfinite(v) else 0.0 for v in z]
        return out, dev

    @staticmethod
    def _new_tokens(store, inc: Any, now: float, dt: float) -> List[Dict[str, Any]]:
        """New values of the episode: B08 first_seen / rare_access findings
        (tier, first-seen ts, bits, flags), newest last."""
        s, k = inc.system, inc.entity
        seen: Dict[str, Dict[str, Any]] = {}
        ids = {x.get("event_id") for x in inc.evidence or () if isinstance(x, Mapping)}
        evs = [ev for ev in store.events(s, k, since=float(inc.opened) - 4 * dt,
                                         kinds=NOVELTY_KINDS + ("class_adopted",), limit=500)]
        for eid in ids:
            if eid:
                ev = store.get_event(eid)
                if ev is not None and ev.kind in NOVELTY_KINDS:
                    evs.append(ev)
        for ev in sorted(evs, key=lambda x: float(x.ts)):
            ex = ev.extra or {}
            toks = m_feedback.event_new_tokens(ev)
            for tok in toks:
                if tok in seen:
                    continue
                seen[tok] = {"token": tok, "dim": ex.get("dim"), "value": ex.get("value"),
                             "tier": ex.get("tier"), "first_seen": float(ev.ts),
                             "kind": ev.kind, "bits": _r(ex.get("bits"), 3),
                             "adopted": bool(ex.get("adopted", False)),
                             "flags": {f: bool(v) for f, v in (ex.get("flags") or {}).items()},
                             "event_id": ev.id}
        return list(seen.values())[-20:]

    @staticmethod
    def _vanished(store, s: str, e: str, inc: Any, now: float) -> List[Dict[str, Any]]:
        """Habitual values (share >= 2 % of their dimension, known >= 3 d) the
        entity's committed history has not seen for >= 6 h while the same
        dimension kept being seen (so an idle or quarantined entity, whose
        commits are held, does not make everything look vanished)."""
        m = m_vocab.get(store, s, e)
        if m is None:
            return []
        out = []
        for dim in m_vocab.DIMS:
            tot = m_vocab.total(m, dim, now)
            if not tot > 0.0:
                continue
            cnt = m_vocab.counts(m, dim, now)
            ents = {v: (m_vocab.entry(m, dim, v, now) or {}) for v in cnt}
            last_dim = max((_f(x.get("last_ts")) for x in ents.values()
                            if _f(x.get("last_ts")) == _f(x.get("last_ts"))), default=_NAN)
            if last_dim != last_dim:
                continue
            for v, c in cnt.items():
                share = c / tot
                if share < VANISH_SHARE:
                    continue
                first, last = _f(ents[v].get("first_ts")), _f(ents[v].get("last_ts"))
                if first == first and now - first >= VANISH_KNOWN_S and last == last \
                        and last_dim - last >= VANISH_QUIET_S:
                    out.append({"token": m_vocab.value_key(dim, v), "dim": dim, "value": v,
                                "share": _r(share, 3), "last_seen": last,
                                "idle_h": _r((last_dim - last) / HOUR, 3)})
        out.sort(key=lambda d: -float(d["share"] or 0.0))
        return out[:8]

    @staticmethod
    def _stack_diff(store, s: str, e: str, inc: Any, now: float, dt: float) -> Dict[str, Any]:
        model = m_client.get(store, s, e)
        if model is None or m_client.kind(model) != "entity":
            return {}
        cur: Dict[str, float] = {}
        for m in reversed(store.raw_tail(s, e, STACK_SET, 64)):
            if m.ts < float(inc.opened) - dt:
                break
            for tok, c in m_client.stack_counts(m.value).items():
                cur[tok] = cur.get(tok, 0.0) + float(c)
        if not cur:
            return {}
        sh = m_client.shares(model, now)
        new = [{"token": t, "n": _r(c, 3), "share_before": _r(sh.get(t, 0.0), 3)}
               for t, c in sorted(cur.items(), key=lambda kv: -kv[1])
               if sh.get(t, 0.0) < STACK_NEW_SHARE]
        missing = [{"token": t, "share_before": _r(v, 3)} for t, v in
                   sorted(sh.items(), key=lambda kv: -kv[1])
                   if v >= STACK_DOMINANT_SHARE and t not in cur]
        return {"new": new[:5], "missing": missing[:5], "n_current": len(cur)}

    @staticmethod
    def _transitions(store, s: str, e: str, ts: float) -> List[Dict[str, Any]]:
        try:
            _t, syms, _f_, _a = m_seq.stream_symbols(store, s, e, ts)
        except Exception:
            return []
        if len(syms) < 2:
            return []
        model = m_seq.get(store, s, e)
        bits = m_seq.loglik(model, syms, m_seq.backoff(store, s, e), m_seq.vocab_size(store, s))
        items = [(float(bits[i]), syms[i - 1], syms[i]) for i in range(1, len(syms))
                 if math.isfinite(float(bits[i]))]
        items.sort(key=lambda x: -x[0])
        out, seen = [], set()
        for b, a, c in items:
            if (a, c) in seen:
                continue
            seen.add((a, c))
            out.append({"from": a, "to": c, "bits": _r(b, 3), "p": _r(2.0 ** (-b), 3)})
            if len(out) == 3:
                break
        return out

    @staticmethod
    def _peer_context(store, s: str, k: str, now: float, dt: float) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        ck = k if is_class(k) else m_class.class_key(store, s, k)
        out["class_key"] = ck
        common: Dict[str, Any] = {}
        for key in ([ck] if ck else []) + [SYSTEM_KEY]:
            g_out = {}
            for g in COMMON_GROUPS:
                v = _dict_at(store, s, key, f"behavior.common.{g}", now)
                if v:
                    g_out[g] = {x: _r(v.get(x), 3) for x in ("L", "frac_up", "frac_down",
                                                            "dir", "run", "n_scored")}
            if g_out:
                common[key] = g_out
        out["common_mode"] = common
        if not is_class(k):
            flags = emit.read_dict(store, s, k, COMMON_FLAG, now)
            out["common_flag"] = {g: int(_f(v) >= 0.5) for g, v in (flags or {}).items()}
            pfam = _dict_at(store, s, k, P_FAMILY, now) or {}
            out["peer_p"] = _r(pfam.get("peer"))
        members = m_class.class_members(store, s, ck) if ck else []
        eds = []
        for m in members:
            if m == k:
                continue
            q = _scalar(store, s, m, Q_ALL, now)
            if q == q:
                eds.append(combine.e_day(q, dt))
        out["members"] = len(members)
        out["members_scored"] = len(eds)
        out["members_unusual"] = int(sum(1 for x in eds if x <= SINGLE_E_DAY))
        out["members_median_e_day"] = _r(float(np.median(eds))) if eds else None
        return out

    @staticmethod
    def _nearest(store, s: str, inc: Any, dev: Sequence[float]) -> Optional[Dict[str, Any]]:
        v = np.asarray(dev, dtype=np.float64)
        nv = float(np.linalg.norm(v))
        if not nv > 0:
            return None
        best = None
        for lb in store.labels(system=s):
            if lb.target_type != "incident" or lb.target_id == inc.id or lb.verdict == "unsure":
                continue
            other = store.get_incident(lb.target_id)
            if other is None:
                continue
            w = _deviation_of(other)
            if w is None:
                continue
            nw = float(np.linalg.norm(w))
            if not nw > 0:
                continue
            cos = float(v @ w / (nv * nw))
            if best is None or cos > best["cosine"]:
                best = {"incident_id": other.id, "entity": other.entity, "verdict": lb.verdict,
                        "scope": lb.scope, "cosine": round(cos, 4), "label_id": lb.id}
        return best

    # ------------------------------------------------------------ narrative
    @staticmethod
    def _narrative(inc: Any, ex: Mapping[str, Any], day_en: str, day_zh: str
                   ) -> Tuple[str, str, Dict[str, Any]]:
        sev = _sev_name(inc.severity)
        who = inc.entity
        attrs = ex.get("attributions") or []
        top = next((a for a in attrs if a.get("range")), attrs[0] if attrs else None)
        axes = ", ".join(inc.axes or ()) or "-"
        e_min = _f(inc.e_day_min)
        what = "class" if is_class(who) else "entity"
        rare_en = (f"as rare as once in {_fmt_days(1.0 / e_min)} for this {what}"
                   if e_min == e_min and e_min > 0 else f"unusual for this {what}")
        rare_zh = f"约{_fmt_days_zh(1.0 / e_min)}一遇" if e_min == e_min and e_min > 0 else "异常"
        sev_zh = {"low": "低", "medium": "中", "high": "高", "critical": "严重"}.get(sev, sev)
        if top is not None and attrs and attrs[0].get("statement_zh"):
            # a violated pattern constraint (P03): its own statement
            top = attrs[0]
            f_en = str(top.get("statement_en") or top["statement_zh"])
            f_zh = str(top["statement_zh"])
        elif top is not None:
            name = top["feature"]
            rng = top.get("range") or "n/a"
            ratio = top.get("ratio")
            rtxt = f" ({ratio:g}×)" if isinstance(ratio, (int, float)) else ""
            f_en = f"{name} {top.get('observed_text')}; usual {day_en} {rng}{rtxt}"
            f_zh = (f"{_ZH_NAME.get(name, name)} {top.get('observed_text')}，"
                    f"{day_zh} 常态 {rng}{rtxt}")
        else:
            f_en = f"no numeric deviation at the trigger tick (usual {day_en} range n/a)"
            f_zh = f"触发时刻无数值偏离({day_zh})"
        h_en = f"{sev.upper()} incident on {who}: {f_en} — {rare_en}."
        h_zh = f"{who} {sev_zh}级事件：{f_zh}，{rare_zh}。"
        b_en: List[str] = []
        b_zh: List[str] = []
        others = [a for a in attrs if a.get("range") and a is not top][:2]
        if others:
            b_en.append("Also: " + "; ".join(
                f"{a['feature']} {a.get('observed_text')} (usual {a['range']})" for a in others))
            b_zh.append("其他偏离：" + "；".join(
                f"{_ZH_NAME.get(a['feature'], a['feature'])} {a.get('observed_text')}"
                f"(常态 {a['range']})" for a in others))
        else:
            b_en.append(f"Axes: {axes}.")
            b_zh.append(f"偏离维度：{axes}。")
        new = ex.get("new_tokens") or []
        van = ex.get("vanished") or []
        if new:
            t = new[-1]
            b_en.append(f"New: {t['token']} (first seen at {t.get('tier') or 'entity'} tier)"
                        + (f" and {len(new) - 1} more" if len(new) > 1 else "") + ".")
            b_zh.append(f"新出现：{t['token']}（{_tier_zh(t.get('tier'))}首次）"
                        + (f"等{len(new)}项" if len(new) > 1 else "") + "。")
        elif van:
            b_en.append(f"Vanished habit: {van[0]['token']} (share {van[0]['share']}).")
            b_zh.append(f"习惯性访问消失：{van[0]['token']}（占比{van[0]['share']}）。")
        else:
            peer = ex.get("peer_context") or {}
            n_u, n_s = peer.get("members_unusual"), peer.get("members_scored")
            if n_s:
                b_en.append(f"Peers: {n_u} of {n_s} class members are also unusual now.")
                b_zh.append(f"同类对比：{n_s}个同类成员中{n_u}个同时异常。")
            else:
                b_en.append("No new categorical values in this episode.")
                b_zh.append("本次无新增类别值。")
        cf_set = ex.get("counterfactual_set") or []
        if ex.get("counterfactual_valid"):
            b_en.append("Without " + ", ".join(cf_set) + " the incident would not have opened "
                        "(recomputed by replay).")
            b_zh.append("若无 " + "、".join(cf_set) + "，重放重算表明不会触发该事件。")
        else:
            reason = (ex.get("counterfactual") or {}).get("reason", "")
            b_en.append(f"Counterfactual: no minimal set found ({reason}).")
            b_zh.append("反事实：未找到可消除该事件的最小特征集。")
        zh = h_zh + "\n" + "\n".join("· " + b for b in b_zh[:3])
        en = h_en + "\n" + "\n".join("- " + b for b in b_en[:3])
        return zh, en, {"h_zh": h_zh, "h_en": h_en, "b_zh": b_zh[:3], "b_en": b_en[:3]}


# ============================================================ pure maths
def gk_shares(model: Optional[Mapping[str, Any]], z: np.ndarray
              ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Garthwaite-Koch shares c_i = w_i^2 / D^2, w = Sigma_oo^(-1/2) (z - mu)_o
    (symmetric inverse root on the observed modelled dims, D^2 = T^2) and
    the RBC chi2_1 p-values of m_density.contributions_model. (None, p, rbc)
    without a fitted model."""
    rbc = np.full(NF, np.nan)
    p = np.full(NF, np.nan)
    if model is None or not m_density.is_fitted(model):
        return None, p, rbc
    rbc, p = m_density.contributions_model(model, z)
    mu = np.asarray(model["mu"], dtype=np.float64)
    Sig = np.asarray(model["Sigma"], dtype=np.float64)
    cols = np.asarray(model.get("cols", np.arange(NF)), dtype=np.intp)
    zc = np.asarray(z, dtype=np.float64) - mu
    obs = cols[np.isfinite(zc[cols])]
    share = np.full(NF, np.nan)
    if not obs.size:
        return share, p, rbc
    S = Sig[np.ix_(obs, obs)]
    lam, U = np.linalg.eigh(0.5 * (S + S.T))
    lam = np.maximum(lam, 1e-9 * max(float(lam.max()), 1e-12))
    w = U @ ((U.T @ zc[obs]) / np.sqrt(lam))
    d2 = float(w @ w)
    if d2 > 0:
        share[obs] = w * w / d2
    return share, p, rbc


def bh_flags(p: Optional[np.ndarray], q: float) -> np.ndarray:
    """Benjamini-Hochberg rejections at level q over the finite p."""
    out = np.zeros(NF, dtype=bool)
    if p is None:
        return out
    p = np.asarray(p, dtype=np.float64)
    ok = np.flatnonzero(np.isfinite(p))
    if not ok.size:
        return out
    o = ok[np.argsort(p[ok], kind="stable")]
    m = o.size
    below = np.flatnonzero(p[o] <= q * np.arange(1, m + 1) / m)
    if below.size:
        out[o[:below[-1] + 1]] = True
    return out


def _deviation_of(inc: Any) -> Optional[np.ndarray]:
    ex = inc.explanation if isinstance(inc.explanation, Mapping) else {}
    dev = ex.get("deviation")
    if isinstance(dev, (list, tuple)) and len(dev) == NF:
        return np.asarray([_f(x) if _f(x) == _f(x) else 0.0 for x in dev])
    v = np.zeros(NF)
    hit = False
    for x in inc.evidence or ():
        if isinstance(x, Mapping) and isinstance(x.get("features"), Mapping):
            for n, z in x["features"].items():
                i = FEATURE_INDEX.get(str(n).split(".")[-1])
                if i is not None and _f(z) == _f(z) and abs(_f(z)) > abs(v[i]):
                    v[i] = _f(z)
                    hit = True
    return v if hit else None


def _fmt_days(d: float) -> str:
    if d >= 1000 * 365:
        return "more than 1000 years"
    if d >= 365:
        return f"{d / 365:.1f} years"
    if d >= 1:
        return f"{d:.0f} days"
    return f"{d * 24:.1f} hours"


def _fmt_days_zh(d: float) -> str:
    if d >= 1000 * 365:
        return "千年以上"
    if d >= 365:
        return f"{d / 365:.1f}年"
    if d >= 1:
        return f"{d:.0f}天"
    return f"{d * 24:.1f}小时"


def _tier_zh(t: Any) -> str:
    return {"system": "全系统", "class": "同类", "entity": "本IP"}.get(str(t), "本IP")
