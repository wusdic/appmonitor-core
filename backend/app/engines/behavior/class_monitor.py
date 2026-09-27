"""ClassMonitorEngine (B18) — every class is an entity: aggregate profile,
calibrated class-level detectors, class events and a class portrait.

Why: a compromised subnet, a hijacked DHCP pool or a role-wide behaviour
change moves many members a little and together. Each member's own view then
discounts it twice: B05 removes the shared movement as common mode, and B08
discounts a value the class adopts together. Neither discount may hide the
class-level risk, so the class itself is scored, as a pseudo-entity
'class:<id>' with its own baseline, detectors, trust and portrait.

Classes per system (model.class via lib/m_class): role classes with >= 2
members ('class:<rid>'), static CIDR classes with >= 3 members
('class:static:<name>') and pools ('class:pool:<cidr>'). Members are those
with membership probability >= 0.5 (static / pool membership is exact).

Per class and tick:
  1. Aggregate row (behavior.class.agg[52], natural units, the feature.nat
     layout) from the members' feature.nat / expo / active and act.tokens:
     counts and bytes summed (exposure dt is shared, so a count stays
     "per tick" and the NB exposure is still dt/60 minutes); ratios pooled
     sum k / sum n; averages weighted by their exposure count (bytes_per_flow
     by flows, latency by requests, ...), else by member; composition shares
     from the pooled counts. behavior.class {active_frac = m_act/m, m, m_act,
     expo (summed), coherence, n_tokens, new_ext} is the dict companion.
  2. model.classagg: the same conjugate seasonal maths as B03
     (lib/m_baseline: exposure-exact NB / Beta-binomial / Student-t, bin48 +
     bin168, hyperpriors) with a CURRENT anchor (rate-capped, committed at
     t - D) and a REFERENCE anchor (24-h delay, trust == 1 rows only, no class
     incident or abnormal class regime within +-24 h, allow_drift). Both learn
     through lib/gating with the class trust B28 writes at the class key and
     honour model.control@(s, class:<id>): rollback_to / release /
     rebase_from / frozen / allow_drift. The learner clock is the
     behavior.class.agg ring. The current learner also carries the class's
     token profile, its JSD calibration ring, the rhythm (active-member)
     statistics and the adoption-rate statistics, so all of them roll back
     together.
  3. Detectors at the class key (lib/emit), scored against the model as of
     its last commit:
     class_int        volume group; per feature p_f = min(1, 2 min(p_cur,
                      p_ref)) (the reference only once it holds data), HMP
                      over the group. Instantaneous, axes [volume].
     class_shape      ratio features (per FEATURE_SPEC group), the comp
                      group and the JSD of the pooled token distribution
                      against the class token profile (conformal p from the
                      class's committed JSD ring, >= 32 entries); HMP of the
                      parts. Axes from the parts with p < 0.01 (app_error,
                      transport or shape). Instantaneous.
     class_rhythm     number of active members per 15-min slot against a
                      Beta-binomial per bin48 (overdispersion by moments,
                      pooled-bin prior), seeded randomised PIT -> z, two
                      one-sided CUSUMs (k = 0.5), threshold from the class
                      path's wall-clock budget (seq.h_for on the slot clock).
                      Accumulator, axes [temporal].
     class_novel      over values new to the class adopted by m_v >= 2
                      members within 24 h (adoption ledgers B08 writes to
                      model.vocab@class; a static / pool class reads the
                      ledgers of its members' role classes):
                      score = sum_v w_v (-log10 P(M >= m_v)), M - 1 ~ Beta-
                      binomial(n - 1) at the class's historical adoption
                      rate (learned from matured records), w_v = 3 external,
                      x3 upload-dominant, x3 sensitive, x2 new eTLD+1
                      org-wide, 0.1 for an internal read-only template, 1
                      otherwise; pm = 10^-score. Accumulator (the 24-h window
                      holds), axes exfil / c2 / categorical.
     class_coherence  per FEATURE_SPEC group and direction, the number of
                      members whose group p (Bonferroni over the group's
                      behavior.pf, direction from behavior.z) is < 0.01,
                      against Binomial(n, 0.01); Bonferroni over 2 x groups.
                      Instantaneous, axes [peer, <group axis>].
  4. Events (never in training): coherent_shift (INFO) at the onset of an
     intensity-only coherent shift (class_int p <= 1e-3, class_shape not
     significant, >= 50 % of the scored members deviating the same way on
     volume, or B05's class common-mode direction); class_shift (LOW) at the
     onset of an app-error / transport class_shape change (extra.system_wide
     from B05's __system__ common mode); class_adoption_risky when
     class_novel alarms on an external, upload-dominant or sensitive value
     (once per value per 24 h). B25 caps intensity-only and system-wide
     class alarms at LOW by their axes.
  5. profile(class).extra.class_monitor, hourly: aggregate p5/p50/p95 (15-min
     exposure, current anchor, this bucket), active fraction by bin48, the
     adoption rate and recent adoptions, the last scores.

NaN means unscored / degraded, never p = 1: an inactive class gets no
class_int / class_shape; members without B04 rows give no class_coherence;
B01 failing (or no feature.nat row for any member at now) writes NaN plus
behavior.degraded for the B01-fed detectors. Absence is data: the rhythm
chart counts silent members, and a silent class is scored for class-wide
silence. ctx.training learns and scores but emits no events. ctx.window_s is
the real dt of the tick (the cadence may switch 900 -> 60 s); the rhythm
chart runs on the wall-clock slot clock.

Store:
  reads   feature.nat, feature.active (vec rings), feature.expo (dict),
          act.tokens (raw, fresh), behavior.z / behavior.pf (members),
          behavior.common.<g> (class key and __system__), behavior.trust /
          trust_prov / quarantine and model.control at the class key (via
          lib/gating), behavior.regime (class key), store.incidents,
          model.class (m_class), model.vocab@class (adoption ledgers and
          class vocabulary, m_vocab), ctx.config (tz, calendar, D_min_s)
  writes  behavior.class.agg[52] (float32 ring, the learner clock),
          behavior.class (dict), behavior.score / pm / axes / acc_alarm /
          degraded [class_int, class_shape, class_rhythm, class_novel,
          class_coherence] at class:<id> (lib/emit), model.classagg@(s,
          class:<id>), checkpoints 'classagg.current' / 'classagg.reference',
          profile(class).extra.class_monitor, events coherent_shift,
          class_shift, class_adoption_risky
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import special as sp

from ...core.engine import Context, Engine
from ...models.schema import BehaviorEvent, DerivedMetric, EntityProfile, MetricKind, Severity
from .lib import bayes, calib, combine, emit, seq
from .lib import features as F
from .lib import gating as G
from .lib import m_baseline as MB
from .lib import m_class
from .lib import m_vocab as V
from .lib import timebins as TB
from .lib.classkeys import SYSTEM_KEY, class_kind
from .lib.detectors import DETECTOR_INFO, arl_days

MODEL = "model.classagg"
FMT = 1
AGG = "behavior.class.agg"
CLASS = "behavior.class"
NAT = "feature.nat"
ACTIVE = "feature.active"
EXPO = "feature.expo"
TOKENS = "act.tokens"
Z = "behavior.z"
PF = "behavior.pf"
COMMON_PREFIX = "behavior.common."
REGIME = "behavior.regime"
LEARNER_CUR = "classagg.current"
LEARNER_REF = "classagg.reference"
B01_ENGINE = "behavior.feature_vector"
B04_ENGINE = "behavior.likelihood"
DETECTORS = ("class_int", "class_shape", "class_rhythm", "class_novel", "class_coherence")

DAY = 86400.0
HOUR = 3600.0
NF = F.FEATURE_DIM
MIN_MEMBERS = {"role": 2, "static": 3, "pool": 1}

EVENT_P = 1e-3                    # class_int / class_shape "significant" for events
COHERENT_FRAC = 0.5
REF_MIN_NEFF = 24.0               # the reference anchor is used once it holds data
# tokens / JSD
TOK_HL_S = 14 * DAY
TOK_CAP = 512                     # profile values kept (the dropped mass stays in N)
TOK_ROW_CAP = 128                 # values per learning row
JSD_MIN_N = 20.0                  # accesses in the tick
JSD_MIN_PROFILE = 50.0            # decayed accesses in the profile
JSD_MIN_RING = 32
JSD_RING = calib.RING_M
# rhythm
RHYTHM_HL_S = 28 * DAY
RHYTHM_K = 0.5
RHYTHM_PRIOR_K = 4.0              # pooled-bin prior strength (rows)
RHYTHM_MIN_W = 8.0                # pooled committed slots before scoring
Z_CLIP = 4.0
# adoption
ADOPT_WINDOW_S = DAY
ADOPT_KNOWN_SLACK_S = HOUR        # class vocabulary first_ts this much older: not new
TALLY_HORIZON_S = 2 * DAY
ADOPT_HL_S = 28 * DAY
ADOPT_A0, ADOPT_B0 = 0.2, 3.8     # prior co-adoption rate 0.05 with 4 pseudo-trials
W_EXTERNAL, W_UPLOAD, W_SENSITIVE, W_NEW_ORG, W_INTERNAL_READ = 3.0, 3.0, 3.0, 2.0, 0.1
READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
RISKY_KEEP_S = DAY
HIST_KEEP = 10
# coherence
COH_P = 0.01
COH_RATE = 0.01
# bookkeeping
META_KEEP_S = G.JOURNAL_MAX_AGE_S + DAY      # the reference commits 24 h late
TOK_KEEP_S = 26 * HOUR
PROFILE_PERIOD_S = HOUR
REF_PERIOD_S = HOUR
ABNORMAL_REGIMES = frozenset({"suspect", "drifting", "rejected", "rollback"})
P_FLOOR = 1e-300
_NAN = math.nan
_TCTX_CACHE_MAX = 4096

# ------------------------------------------------------------ feature sets
_I = F.FEATURE_INDEX
_KIND = [F.FEATURE_KIND[n] for n in F.FEATURE_NAMES_V2]
_GROUP = [F.FEATURE_GROUP[n] for n in F.FEATURE_NAMES_V2]
SUM_IDX = np.array([i for i, k in enumerate(_KIND) if k in ("count", "bytes")], dtype=np.intp)
RAT = MB.RAT                                   # Beta-binomial ratios (n = a count feature)
RAT_N = MB.RATIO_N_IDX
CLR_IDX = np.asarray(F.CLR_IDX, dtype=np.intp)
_COUNT_OF_METRIC = {F.FEATURE_SOURCE[n]: _I[n] for n in F.FEATURE_NAMES_V2
                    if F.FEATURE_KIND[n] == "count" and isinstance(F.FEATURE_SOURCE[n], str)}
_N_ALIAS = {"derived.dns_fail_rate.n": "dns.queries"}


def _n_counts(nsrc: Any) -> Optional[List[int]]:
    """Count features whose sum is a feature's exposure n (None: not in feature.nat)."""
    if nsrc is None:
        return None
    out = []
    for m in ([nsrc] if isinstance(nsrc, str) else list(nsrc)):
        i = _COUNT_OF_METRIC.get(_N_ALIAS.get(m, m))
        if i is None:
            return None
        out.append(i)
    return out


# weighted-average features: everything not summed, not a BB ratio, not comp
AVG_IDX = np.array([i for i in range(NF) if i not in set(SUM_IDX) | set(RAT) | set(CLR_IDX)
                    and _KIND[i] != "ctx"], dtype=np.intp)
_WMAT = np.zeros((AVG_IDX.size, NF))          # exposure weight = sum of these counts
_HAS_W = np.zeros(AVG_IDX.size, dtype=bool)
for _j, _f in enumerate(AVG_IDX):
    _nc = _n_counts(F.FEATURE_NSRC[F.FEATURE_NAMES_V2[_f]])
    if _nc:
        _WMAT[_j, _nc] = 1.0
        _HAS_W[_j] = True
# composition counts rebuilt from the pooled row (features.CLR_FEATURES order)
_CLR_SRC = [(_I["http_get_ratio"], _I["http_requests"]), (_I["http_write_ratio"], _I["http_requests"]),
            (_I["http_4xx_rate"], _I["http_requests"]), (_I["http_5xx_rate"], _I["http_requests"]),
            (_I["dns_queries"], -1), (_I["tls_handshakes"], -1), (_I["flows"], -1),
            (_I["syn_ratio"], _I["flows"])]

VOL_IDX = np.asarray(F.GROUPS["volume"], dtype=np.intp)
T_RATIO_IDX = np.array([i for i in range(NF) if _KIND[i] == "ratio" and i not in set(RAT)],
                       dtype=np.intp)
SHAPE_RATIO_IDX = np.concatenate((RAT, T_RATIO_IDX))
# features the predictive is evaluated on (others are masked to NaN: cheaper)
_SCORED = np.zeros(NF, dtype=bool)
_SCORED[VOL_IDX] = True
_SCORED[SHAPE_RATIO_IDX] = True
_SCORED[CLR_IDX] = True
APP_ERROR_F = frozenset({_I["http_4xx_rate"], _I["http_5xx_rate"], _I["http_latency"]})
TRANSPORT_F = frozenset(F.GROUPS["transport"]) | frozenset(F.GROUPS["probe"])
GROUP_NAMES: Tuple[str, ...] = tuple(F.GROUP_ORDER)
GROUP_IDX = [np.asarray(F.GROUPS[g], dtype=np.intp) for g in GROUP_NAMES]
_GROUP_AXIS = {"volume": "volume", "breadth": "breadth", "app": "shape", "dns": "shape",
               "tls": "shape", "timing": "temporal", "transport": "transport",
               "probe": "transport", "comp": "shape"}


_G_START = np.array([int(ix[0]) for ix in GROUP_IDX], dtype=np.intp)
if not all(np.array_equal(ix, np.arange(ix[0], ix[0] + ix.size)) for ix in GROUP_IDX) \
        or int(_G_START[0]) != 0 or sum(ix.size for ix in GROUP_IDX) != NF:
    raise ImportError("class_monitor: FEATURE_SPEC groups must be contiguous blocks in GROUP_ORDER")
_G_OF = np.repeat(np.arange(len(GROUP_IDX)), [ix.size for ix in GROUP_IDX])
_APP_ERR_COL = np.zeros(NF, dtype=bool)
_APP_ERR_COL[sorted(APP_ERROR_F)] = True


def feature_axis(i: int) -> str:
    """Reading axis of one feature (fusion's class caps key on these)."""
    if i in APP_ERROR_F:
        return "app_error"
    if i in TRANSPORT_F:
        return "transport"
    return _GROUP_AXIS[_GROUP[i]]


# ================================================================ aggregate
def aggregate(N: np.ndarray) -> np.ndarray:
    """Class row [52] (feature.nat units) from member rows N[k, 52]: counts
    and bytes summed, ratios pooled sum k / sum n, averages weighted by their
    exposure count (per member otherwise), comp shares from pooled counts."""
    N = np.asarray(N, dtype=np.float64).reshape(-1, NF)
    out = np.full(NF, np.nan)
    if not N.shape[0]:
        return out
    fin = np.isfinite(N)
    Z0 = np.where(fin, N, 0.0)
    out[SUM_IDX] = np.maximum(Z0[:, SUM_IDX], 0.0).sum(axis=0)
    n = np.maximum(Z0[:, RAT_N], 0.0)
    use = fin[:, RAT] & (n > 0.0)
    sk = np.where(use, np.clip(Z0[:, RAT], 0.0, 1.0) * n, 0.0).sum(axis=0)
    sn = np.where(use, n, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[RAT] = np.where(sn > 0.0, sk / sn, np.nan)
        w = np.where(_HAS_W[None, :], np.maximum(Z0, 0.0) @ _WMAT.T, 1.0)
        w = np.where(fin[:, AVG_IDX], w, 0.0)
        sw = w.sum(axis=0)
        out[AVG_IDX] = np.where(sw > 0.0, (w * Z0[:, AVG_IDX]).sum(axis=0) / sw, np.nan)
        cnt = np.array([out[a] * (out[b] if b >= 0 else 1.0) for a, b in _CLR_SRC])
        cnt = np.where(np.isfinite(cnt) & (cnt > 0.0), cnt, 0.0)
        tot = cnt.sum()
        out[CLR_IDX] = cnt / tot if tot > 0.0 else np.nan
    return out


# ============================================================ decayed sums
def _dec_new(width: Any) -> Dict[str, Any]:
    return {"T0": _NAN, "v": np.zeros(width)}


def _dec_fold(d: Dict[str, Any], ts: float, hl: float, idx: Any, contrib: np.ndarray,
              w: float) -> None:
    """Add w * contrib at row idx with the row's own decay (any row order):
    stored = sum w 2^((t - T0)/hl); the true value at t is stored 2^-((t - T0)/hl)."""
    if d["T0"] != d["T0"]:
        d["T0"] = float(ts)
    x = (ts - d["T0"]) / hl
    if x > 40.0:                                  # re-anchor before 2^x overflows
        d["v"] = d["v"] * 2.0 ** (-x)
        d["T0"], x = float(ts), 0.0
    d["v"][idx] += float(w) * 2.0 ** x * contrib


def _dec_true(d: Dict[str, Any], now: float, hl: float) -> np.ndarray:
    if d["T0"] != d["T0"]:
        return np.zeros_like(d["v"])
    return d["v"] * 2.0 ** (-(now - d["T0"]) / hl)


# ================================================================ learner
class _Row(NamedTuple):
    """One committed tick: the anchor row (None when the class was inactive)
    and its learning extras (meta tuple, see _meta_row)."""
    ts: float
    base: Optional[MB.Row]
    meta: Tuple


class _Cur(NamedTuple):
    """Current learner state: the anchor and the auxiliary class statistics."""
    anc: MB.Anchor
    aux: Dict[str, Any]


def _new_aux() -> Dict[str, Any]:
    return {"tok": {"T0": _NAN, "c": {}, "N": 0.0}, "jsd": calib.Ring(JSD_RING),
            "rh": _dec_new((49, 4)), "ad": _dec_new(4)}


def _init_cur() -> _Cur:
    return _Cur(MB.new_anchor(week=True, select=True), _new_aux())


def _init_ref() -> MB.Anchor:
    return MB.new_anchor(week=False, select=False, hl_days=MB.HL_REF_DAYS)


def _fold_tokens(tok: Dict[str, Any], counts: Mapping[str, float], ts: float, w: float) -> None:
    if tok["T0"] != tok["T0"]:
        tok["T0"] = float(ts)
    x = (ts - tok["T0"]) / TOK_HL_S
    c = tok["c"]
    if x > 40.0:
        f = 2.0 ** (-x)
        for k in c:
            c[k] *= f
        tok["N"] *= f
        tok["T0"], x = float(ts), 0.0
    g = float(w) * 2.0 ** x
    tot = 0.0
    for k, n in counts.items():
        c[k] = c.get(k, 0.0) + n * g
        tot += n
    tok["N"] += tot * g
    if len(c) > TOK_CAP:
        for k, _ in sorted(c.items(), key=lambda kv: kv[1])[:len(c) - TOK_CAP]:
            del c[k]


def _update_cur(state: _Cur, row: _Row, w: float) -> _Cur:
    """GatedLearner update: queue the anchor row (folded once per tick for all
    classes, m_baseline.flush_many) and fold the auxiliary statistics."""
    w = float(w)
    if not (w > 0.0 and math.isfinite(w)):
        return state
    if row.base is not None:
        MB.queue(state.anc, row.base, w, cap=MB.CAP_CURRENT)
    _dt, _mact, jsd, slot, tallies, tok = row.meta
    aux = state.aux
    if tok:
        _fold_tokens(aux["tok"], tok, row.ts, w)
    if jsd == jsd and jsd is not None and combine.seeded_uniform("classagg.jsd", row.ts) < w:
        aux["jsd"].add(float(jsd), row.ts)        # seeded thinning: a ring takes no weights
    if slot is not None:
        b, a, m = slot
        if m > 0:
            c = np.array([1.0, a, m, a * a / m])
            _dec_fold(aux["rh"], row.ts, RHYTHM_HL_S, int(b), c, w)
            _dec_fold(aux["rh"], row.ts, RHYTHM_HL_S, 48, c, w)
    for s_, n_ in tallies:
        if n_ > 0:
            _dec_fold(aux["ad"], row.ts, ADOPT_HL_S, slice(None),
                      np.array([1.0, s_, n_, s_ * s_ / n_]), w)
    return state


def _update_ref(state: MB.Anchor, row: MB.Row, w: float) -> MB.Anchor:
    """Reference admission: trust == 1 and eligible (decided at fetch)."""
    if not row.elig or not (float(w) >= 1.0 - 1e-6):
        return state
    return MB.queue(state, row, 1.0, cap=MB.CAP_REFERENCE, drift=row.drift)


def _dump_cur(st: _Cur) -> Dict[str, Any]:
    aux = st.aux
    tok = aux["tok"]
    return {"anc": MB.dump(st.anc),
            "aux": {"tok": {"T0": tok["T0"], "c": dict(tok["c"]), "N": tok["N"]},
                    "jsd": aux["jsd"].to_dict(),
                    "rh": {"T0": aux["rh"]["T0"], "v": aux["rh"]["v"].copy()},
                    "ad": {"T0": aux["ad"]["T0"], "v": aux["ad"]["v"].copy()}}}


def _load_cur(blob: Mapping[str, Any]) -> _Cur:
    a = blob["aux"]
    tok = a["tok"]
    aux = {"tok": {"T0": float(tok["T0"]), "c": dict(tok["c"]), "N": float(tok["N"])},
           "jsd": calib.Ring.from_dict(a["jsd"]),
           "rh": {"T0": float(a["rh"]["T0"]), "v": np.array(a["rh"]["v"], dtype=np.float64)},
           "ad": {"T0": float(a["ad"]["T0"]), "v": np.array(a["ad"]["v"], dtype=np.float64)}}
    return _Cur(MB.load(blob["anc"]), aux)


def new_model(kind: Optional[str]) -> Dict[str, Any]:
    """Empty model.classagg@(s, class:<id>) (hyperprior predictives)."""
    cur = _init_cur()
    return {
        "fmt": FMT, "kind": "classagg", "class_kind": kind, "version": 0, "branch": 0,
        "current": cur.anc, "aux": cur.aux, "reference": _init_ref(),
        "gate": G.GateState(), "gate_ref": G.GateState(),
        "meta": {}, "ref_elig": {}, "ctl_sig": None,
        "run": {"slot": None, "slot_bin": None, "slot_end": None, "slot_act": [],
                "slot_m": 0, "S_hi": 0.0, "S_lo": 0.0, "rh_scored": False, "tallied": {},
                "coh_on": False, "shift_on": False, "risky": {}, "hist": [], "last": {},
                "prune_ts": -math.inf},
        "members": [], "n_members": 0, "n_eff": 0.0, "allow_drift": 0.0, "held": [],
        "ts": _NAN,
    }


def _valid(model: Any) -> bool:
    return (isinstance(model, dict) and model.get("fmt") == FMT
            and model.get("kind") == "classagg" and isinstance(model.get("current"), MB.Anchor))


def _control_sig(control: Any) -> Optional[Tuple]:
    if control is None:
        return None
    d = G.control_directives(control)
    return (d["rollback_to"], d["release"], d["rebase_from"], d["frozen"], d["version"])


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return _NAN


def _hmp(ps: Sequence[float]) -> float:
    return combine.whmp(list(ps)) if len(ps) else _NAN


# ============================================================= per system
class _Sys:
    """Per-tick, per-system reads shared by its classes."""

    def __init__(self, store: Any, s: str, now: float) -> None:
        self.store, self.s, self.now = store, s, now
        self.common_sys: Dict[str, Dict[str, Any]] = {}
        self._ledgers: Optional[List[Tuple[Dict[str, Any], Dict[str, Any]]]] = None
        self.has_vocab = False          # some role class has a model.vocab (B08 runs)
        self.b01_failed = store.engine_failed(B01_ENGINE, now)
        self.b04_failed = store.engine_failed(B04_ENGINE, now)
        for g in ("app_error", "transport"):
            v = emit.read_dict(store, s, SYSTEM_KEY, COMMON_PREFIX + g, now)
            if v:
                self.common_sys[g] = v

    def ledgers(self) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """(record, class vocabulary model) for every adoption record of the
        system's role-class ledgers touched within the tally horizon."""
        if self._ledgers is None:
            out = []
            lo = self.now - TALLY_HORIZON_S
            keys = {k for k in self.store.pseudo_entities(self.s) if class_kind(k) == "role"}
            keys.update(k for k in m_class.all_class_keys(self.store, self.s)
                        if class_kind(k) == "role")
            for ck in sorted(keys):
                vm = V.get(self.store, self.s, ck)
                if not vm:
                    continue
                self.has_vocab = True
                for rec in (vm.get("adoption") or {}).values():
                    if _f(rec.get("last_ts")) >= lo and rec.get("members"):
                        out.append((rec, vm))
            self._ledgers = out
        return self._ledgers


# ================================================================== engine
class ClassMonitorEngine(Engine):
    name = "behavior.class_monitor"
    layer = "behavior"
    consumes = [NAT, ACTIVE, EXPO, "feature.tctx", TOKENS, Z, PF, COMMON_PREFIX + "*",
                "behavior.trust", "behavior.trust_prov", "behavior.quarantine", REGIME,
                "model.class", "model.vocab", "model.control", "incidents"]
    produces = [AGG, CLASS, MODEL, "checkpoint:" + LEARNER_CUR, "checkpoint:" + LEARNER_REF,
                "behavior.score", "behavior.pm", "behavior.axes", "behavior.acc_alarm",
                "behavior.degraded", "profile.extra.class_monitor", "event.coherent_shift",
                "event.class_shift", "event.class_adoption_risky"]
    description = ("Class-as-entity monitor: member aggregate with its own two-anchor gated "
                   "conjugate baseline, class_int / class_shape / class_rhythm / class_novel / "
                   "class_coherence at class:<id>, class events and class profiles.")
    interval = 1

    def __init__(self, profile_period_s: float = PROFILE_PERIOD_S,
                 ref_period_s: float = REF_PERIOD_S, **params: Any) -> None:
        super().__init__(**params)
        self.profile_period_s = float(profile_period_s)
        self.ref_period_s = float(ref_period_s)
        self._cfg: Dict[str, Any] = {}
        self._now = 0.0
        self._tctx_cache: Dict[Tuple[float, float], Dict[str, Any]] = {}
        self._ret_stores: Set[int] = set()
        self._class_cache: Dict[str, Tuple[Tuple, Dict[str, List[str]]]] = {}
        self._ref_pred: Dict[Tuple[str, str], Tuple[Tuple, MB.Pred]] = {}
        self._meta: Dict[float, Tuple] = {}
        self._ref_ctx: Optional[Tuple[Dict[str, Any], List[Tuple[float, float]], float]] = None
        self._cur = G.GatedLearner(
            name=LEARNER_CUR, init=_init_cur, update=_update_cur, fetch=self._fetch_cur,
            dump=_dump_cur, load=_load_cur, on_rebase=self._on_rebase_cur,
            ckpt_every_s=G.CKPT_EVERY_S, clock=AGG)
        self._ref = G.GatedLearner(
            name=LEARNER_REF, init=_init_ref, update=_update_ref, fetch=self._fetch_ref,
            dump=MB.dump, load=MB.load, on_rebase=MB.on_rebase_reference, d_min_s=DAY,
            ckpt_every_s=G.CKPT_EVERY_REF_S, clock=AGG)

    # ------------------------------------------------------------------- run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        now, dt = float(ctx.now), float(ctx.window_s)
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError(f"ClassMonitorEngine: bad ctx.window_s {ctx.window_s!r}")
        self._cfg, self._now = ctx.config, now
        self._cur.d_min_s = float(ctx.config.get("D_min_s") or G.D_MIN_S)
        if id(store) not in self._ret_stores:           # the learner clock must cover replay
            store.set_retention(AGG, None, META_KEEP_S)
            self._ret_stores.add(id(store))
        if len(self._tctx_cache) > _TCTX_CACHE_MAX:
            self._tctx_cache.clear()
        tc_now = self._tctx(now, dt)
        tc_slot = self._tctx(now - 0.5 * min(dt, float(TB.SLOT_S)), dt)
        done: List[Tuple[str, str, Dict[str, Any]]] = []
        for s in store.systems():
            classes = self._classes(store, s)
            if not classes:
                continue
            sc = _Sys(store, s, now)
            for ck, mem in classes.items():
                model = self._class(ctx, sc, ck, mem, now, dt, tc_now, tc_slot)
                if model is not None:
                    done.append((s, ck, model))
        # fold every queued anchor row, vectorised across classes
        MB.flush_many([m[k] for _, _, m in done for k in ("current", "reference")])
        for s, ck, model in done:
            gate = model["gate"]
            model["version"], model["branch"] = int(gate.version), int(gate.branch)
            model["allow_drift"] = float(gate.allow_drift)
            model["held"] = gate.held
            model["n_eff"] = MB.n_eff(model["current"])
            model["ts"] = now
            store.put_model(s, ck, MODEL, model, version=int(gate.version), ts=now)
            if self.entity_due((s, ck, "profile"), now, self.profile_period_s):
                self._profile(store, s, ck, model, tc_now, now)
        return len(done)

    def _classes(self, store: Any, s: str) -> Dict[str, List[str]]:
        """{class key: members} of the classes monitored in s (cached per
        model.class object / version)."""
        mc = m_class.get(store)
        sig = (id(mc), m_class.version(store), len(mc.get("assign", {}) or {}))
        hit = self._class_cache.get(s)
        if hit is not None and hit[0] == sig:
            return hit[1]
        out: Dict[str, List[str]] = {}
        for ck in m_class.all_class_keys(store, s, min_members=MIN_MEMBERS["role"]):
            mem = m_class.class_members(store, s, ck)
            if len(mem) >= MIN_MEMBERS.get(class_kind(ck) or "role", 2):
                out[ck] = mem
        self._class_cache[s] = (sig, out)
        return out

    def _tctx(self, ts: float, dt: float) -> Dict[str, Any]:
        key = (float(ts), float(TB.cadence_class(dt)))
        tc = self._tctx_cache.get(key)
        if tc is None:
            tc = self._tctx_cache[key] = TB.tctx_from_config(ts, self._cfg, dt)
        return tc

    # -------------------------------------------------------------- a class
    def _class(self, ctx: Context, sc: _Sys, ck: str, members: List[str], now: float,
               dt: float, tc_now: Dict[str, Any], tc_slot: Dict[str, Any]
               ) -> Optional[Dict[str, Any]]:
        store, s = ctx.store, sc.s
        rows, present, active_members = [], [], []
        seen_any = False
        for m in members:
            nat = store.vec_at(s, m, NAT, now)
            if nat is None:
                seen_any = seen_any or store.last_write_ts(s, m, NAT) is not None
                continue
            present.append(m)
            rows.append(nat)
            a = store.vec_at(s, m, ACTIVE, now)
            if a is not None and float(a[0]) > 0.5:
                active_members.append(m)
        model = store.get_model(s, ck, MODEL)
        if not present and not seen_any and not _valid(model):
            return None                          # nothing ever observed for this class
        if not _valid(model):
            model = new_model(class_kind(ck))
        model["members"], model["n_members"] = list(members), len(members)
        run = model["run"]
        scores: Dict[str, float] = {}
        pm: Dict[str, float] = {}
        axes: Dict[str, List[str]] = {}
        acc: Dict[str, int] = {}
        degraded: Dict[str, str] = {}
        info: Dict[str, Any] = {"m": len(present), "m_act": len(active_members)}
        b01_bad = sc.b01_failed or not present
        cause = f"producer_error:{B01_ENGINE}" if sc.b01_failed else "stale:feature.nat"

        # ---- 1) aggregate row (every tick with member rows: the learner clock)
        agg = None
        jsd = _NAN
        tokens: Dict[str, float] = {}
        tok_row: Optional[Dict[str, float]] = None
        if present and not sc.b01_failed:
            agg = aggregate(np.asarray(rows, dtype=np.float64))
            store.add_vec(s, ck, AGG, now, agg.astype(np.float32), window_s=int(round(dt)))
            expo: Dict[str, float] = {}
            for m in active_members:
                ex = store.latest_fresh(s, m, EXPO, now)
                if isinstance(ex, Mapping):
                    for k, v in ex.items():
                        x = _f(v)
                        if x == x:
                            expo[k] = expo.get(k, 0.0) + x
                tk = store.latest_fresh(s, m, TOKENS, now)
                if isinstance(tk, Mapping):
                    for k, v in tk.items():
                        x = _f(v)
                        if k != "__other__" and x > 0.0:
                            tokens[str(k)] = tokens.get(str(k), 0.0) + x
            info.update(active_frac=len(active_members) / len(present), expo=expo,
                        n_tokens=float(sum(tokens.values())))
            if tokens:
                if len(tokens) > TOK_ROW_CAP:
                    tok_row = dict(sorted(tokens.items(), key=lambda kv: -kv[1])[:TOK_ROW_CAP])
                else:
                    tok_row = tokens

        # ---- 3a) class_int / class_shape (active class only)
        if b01_bad:
            for d in ("class_int", "class_shape", "class_rhythm"):
                scores[d] = _NAN
                degraded[d] = cause
        elif active_members:
            p_int, ax_int, u_dir, p_shape, ax_shape, jsd = self._int_shape(
                s, ck, model, agg, tokens, now, dt, tc_now)
            scores["class_int"], pm["class_int"] = _score(p_int), p_int
            scores["class_shape"], pm["class_shape"] = _score(p_shape), p_shape
            if p_int == p_int:
                axes["class_int"] = ax_int
            if p_shape == p_shape:
                axes["class_shape"] = ax_shape
            info["p_int"], info["p_shape"], info["dir_int"] = p_int, p_shape, u_dir
            info["axes_shape"] = ax_shape

        # ---- 3b) class_rhythm (slot clock; silent members count)
        slot_obs = None
        if not b01_bad:
            slot_obs = self._rhythm(s, ck, model, present, active_members, now, dt, tc_slot,
                                    scores, pm, axes, acc)

        # ---- 3c) class_coherence (members' B04 rows)
        coh = None if sc.b04_failed else self._coherence(store, s, active_members, now)
        if coh is None:
            # active members but no B04 row for any of them: degraded, not "coherent"
            if sc.b04_failed or (len(active_members) >= 2 and not any(
                    store.vec_at(s, m, Z, now) is not None for m in active_members)):
                scores["class_coherence"] = _NAN
                degraded["class_coherence"] = (f"producer_error:{B04_ENGINE}"
                                               if sc.b04_failed else "stale:behavior.z")
        else:
            p_coh, ax_coh, frac = coh
            scores["class_coherence"], pm["class_coherence"] = _score(p_coh), p_coh
            axes["class_coherence"] = ax_coh
            info["coherence"] = frac
            info["p_coh"] = p_coh

        # ---- 3d) class_novel (adoption ledgers)
        tallies, novel = self._novel(sc, ck, model, set(members), now, dt, info)
        if novel is not None:
            p_nov, ax_nov, alarm, top = novel
            scores["class_novel"], pm["class_novel"] = _score(p_nov), p_nov
            axes["class_novel"] = ax_nov
            acc["class_novel"] = int(alarm)
            info["p_novel"] = p_nov
        else:
            top = []

        emit.write_scores(store, s, ck, now, scores, pm=pm or None, axes=axes or None,
                          acc_alarm=acc or None, degraded=degraded or None,
                          window_s=int(round(dt)))
        if agg is not None:
            val = {k: v for k, v in info.items() if k in ("m", "m_act", "active_frac", "expo",
                                                          "coherence", "n_tokens", "new_ext")}
            val["dt"] = dt
            store.add_derived(DerivedMetric(name=CLASS, value=val, ts=now, system=s, entity=ck,
                                            window_s=int(round(dt)), kind=MetricKind.CATEGORICAL))
        run["last"] = {"ts": now, **{d: pm.get(d, _NAN) for d in DETECTORS},
                       "active_frac": info.get("active_frac", _NAN)}

        # ---- 4) events
        if not ctx.training:
            self._events(sc, ck, model, info, top, acc.get("class_novel", 0), now, dt)
        else:
            run["coh_on"] = run["shift_on"] = False

        # ---- 2) learning (after scoring: the scores used the pre-commit model)
        if agg is not None or slot_obs is not None or tallies:
            model["meta"][now] = (dt, len(active_members) if agg is not None else 0,
                                  jsd if jsd == jsd else None, slot_obs, tuple(tallies),
                                  tok_row)
        self._learn(ctx, s, ck, model, now, dt)
        return model

    # ------------------------------------------------------ class_int / shape
    def _int_shape(self, s: str, ck: str, model: Dict[str, Any], agg: np.ndarray,
                   tokens: Mapping[str, float], now: float, dt: float,
                   tc: Mapping[str, Any]) -> Tuple[float, List[str], int, float, List[str], float]:
        nat = np.where(_SCORED, agg, np.nan)
        pc = MB.anchor_predictive(model["current"], tc)
        u_c, p_c = MB.midp(pc, nat, dt)
        ref = model["reference"]
        if not ref.empty and MB.n_eff(ref) >= REF_MIN_NEFF:
            key = (id(ref), ref.n_commit, ref.T, int(tc["bin48"]))
            hit = self._ref_pred.get((s, ck))
            if hit is not None and hit[0] == key:
                pr = hit[1]
            else:
                pr = MB.anchor_predictive(ref, tc, anchor="reference")
                self._ref_pred[(s, ck)] = (key, pr)
            _, p_r = MB.midp(pr, nat, dt)
            p = np.asarray(bayes.combine_anchors(p_c, p_r), dtype=np.float64)
        else:
            p = p_c
        # class_int: HMP over the volume group
        pv = p[VOL_IDX]
        ok = np.isfinite(pv)
        p_int = _hmp(pv[ok].tolist())
        u_dir = 0
        if ok.any():
            j = VOL_IDX[ok][int(np.argmin(pv[ok]))]
            u_dir = 1 if u_c[j] > 0.5 else -1
        # class_shape: ratios per FEATURE_SPEC group, the comp group, JSD
        parts: List[Tuple[float, str]] = []
        by_group: Dict[str, List[int]] = {}
        for i in SHAPE_RATIO_IDX:
            if np.isfinite(p[i]):
                by_group.setdefault(_GROUP[i], []).append(int(i))
        for g, idx in by_group.items():
            pg = p[idx]
            j = idx[int(np.argmin(pg))]
            parts.append((_hmp(pg.tolist()), feature_axis(j)))
        pcl = p[CLR_IDX]
        if np.isfinite(pcl).any():
            parts.append((_hmp(pcl[np.isfinite(pcl)].tolist()), "shape"))
        jsd, p_jsd = self._jsd(s, ck, model["aux"], tokens, now)
        if p_jsd == p_jsd:
            parts.append((p_jsd, "shape"))
        p_shape = _hmp([q for q, _ in parts])
        sig = sorted({a for q, a in parts if q < 0.01})
        if not sig and parts:
            sig = [min(parts)[1]]
        return p_int, ["volume"], u_dir, p_shape, sig or ["shape"], jsd

    @staticmethod
    def _jsd(s: str, ck: str, aux: Dict[str, Any], tokens: Mapping[str, float],
             now: float) -> Tuple[float, float]:
        """(JSD in bits of the pooled token distribution vs the class profile,
        conformal p from the committed JSD ring); NaN when either side is thin."""
        tok = aux["tok"]
        n_p = float(sum(tokens.values())) if tokens else 0.0
        c = tok["c"]
        qs = float(sum(c.values()))
        if n_p < JSD_MIN_N or not c or tok["T0"] != tok["T0"]:
            return _NAN, _NAN
        if tok["N"] * 2.0 ** (-(now - tok["T0"]) / TOK_HL_S) < JSD_MIN_PROFILE or qs <= 0.0:
            return _NAN, _NAN
        acc = 0.0
        q_in = 0.0
        for k, n in tokens.items():
            pp = n / n_p
            q = c.get(k, 0.0) / qs
            m = 0.5 * (pp + q)
            acc += pp * math.log2(pp / m)
            if q > 0.0:
                acc += q * math.log2(q / m)
                q_in += q
        jsd = max(0.0, 0.5 * acc + 0.5 * max(0.0, 1.0 - q_in))
        ring = aux["jsd"]
        if len(ring) < JSD_MIN_RING:
            return jsd, _NAN
        u = combine.seeded_uniform(s, ck, "class_shape.jsd", now)
        return jsd, float(calib.p_from_ring(ring, jsd, u))

    # ------------------------------------------------------------ class_rhythm
    def _rhythm(self, s: str, ck: str, model: Dict[str, Any], present: List[str],
                active: List[str], now: float, dt: float, tc_slot: Mapping[str, Any],
                scores: Dict[str, float], pm: Dict[str, float], axes: Dict[str, List[str]],
                acc: Dict[str, int]) -> Optional[Tuple[int, int, int]]:
        """Active members per 15-min slot -> Beta-binomial PIT -> two CUSUMs.
        Returns the finalised slot observation (bin48, a, m) to learn, if any."""
        run = model["run"]
        slot = int(tc_slot["slot"])
        obs = None
        if run["slot"] is not None and run["slot"] != slot:
            obs = self._close_slot(s, ck, model, now)            # a gap or an unaligned tick
        if run["slot"] != slot:
            tz = self._cfg.get("tz") or TB.DEFAULT_TZ
            run.update(slot=slot, slot_bin=int(tc_slot["bin48"]),
                       slot_end=float(TB.slot_bounds(slot, tz)[1]), slot_act=[], slot_m=0)
        acts = set(run["slot_act"])
        acts.update(active)
        run["slot_act"] = sorted(acts)
        run["slot_m"] = max(int(run["slot_m"]), len(present))
        if now >= float(run["slot_end"]) - 1e-6:                 # this tick completes the slot
            o = self._close_slot(s, ck, model, now)
            obs = o if o is not None else obs
            run["slot"] = None
        if run["rh_scored"]:
            S = max(float(run["S_hi"]), float(run["S_lo"]))
            h = seq.h_for("gauss", 2.0 * arl_days("class_rhythm"), max(dt, float(TB.SLOT_S)),
                          k=RHYTHM_K)
            scores["class_rhythm"] = S
            pm["class_rhythm"] = float(np.asarray(seq.cusum_stationary_p(S, RHYTHM_K, 2)))
            axes["class_rhythm"] = ["temporal"]
            acc["class_rhythm"] = int(S >= h)
        return obs

    def _close_slot(self, s: str, ck: str, model: Dict[str, Any],
                    now: float) -> Optional[Tuple[int, int, int]]:
        run = model["run"]
        b, a, m = int(run["slot_bin"]), len(run["slot_act"]), int(run["slot_m"])
        if m <= 0:
            return None
        rh = _dec_true(model["aux"]["rh"], now, RHYTHM_HL_S)
        Wp, skp, snp = rh[48, 0], rh[48, 1], rh[48, 2]
        if Wp >= RHYTHM_MIN_W and snp > 0.0:
            p_pool = min(max(skp / snp, 1e-3), 1.0 - 1e-3)
            a0, b0 = RHYTHM_PRIOR_K * p_pool, RHYTHM_PRIOR_K * (1.0 - p_pool)
            W, sk, sn, skk = rh[b]
            p_hat, phi = bayes.ratio_posterior(a0, b0, W, sk, sn, skk, 0.0)
            c = min(sn + RHYTHM_PRIOR_K, float(phi))
            aa, bb = float(p_hat) * c, (1.0 - float(p_hat)) * c
            # randomised PIT from the scalar CDF path (the pmf is the CDF step)
            lo = float(bayes.bb_cdf(a - 1, m, aa, bb)) if a > 0 else 0.0
            eq = max(0.0, float(bayes.bb_cdf(a, m, aa, bb)) - lo)
            v = combine.seeded_uniform(s, ck, "class_rhythm", run["slot"])
            u = min(max(lo + v * eq, 0.0), 1.0)
            z = min(max(float(bayes.phi_inv(u)), -Z_CLIP), Z_CLIP)
            run["S_hi"] = max(0.0, float(run["S_hi"]) + z - RHYTHM_K)
            run["S_lo"] = max(0.0, float(run["S_lo"]) - z - RHYTHM_K)
            run["rh_scored"] = True
        return (b, a, m)

    # --------------------------------------------------------- class_coherence
    @staticmethod
    def _coherence(store: Any, s: str, active: List[str], now: float
                   ) -> Optional[Tuple[float, List[str], Dict[str, Dict[str, float]]]]:
        """Binomial tail of same-direction member deviations per group."""
        P, Zs = [], []
        for m in active:
            z = store.vec_at(s, m, Z, now)
            if z is None:
                continue
            z = np.asarray(z, dtype=np.float64)
            pf = store.vec_at(s, m, PF, now)
            if pf is None:
                with np.errstate(invalid="ignore"):
                    p = 2.0 * sp.ndtr(-np.abs(z))
            else:
                p = np.asarray(pf, dtype=np.float64)
            P.append(p)
            Zs.append(z)
        n = len(P)
        if n < 2:
            return None
        P = np.asarray(P)
        Zs = np.asarray(Zs)
        # FEATURE_SPEC groups are contiguous column blocks: one reduceat per quantity
        ok = np.isfinite(P) & np.isfinite(Zs)
        Pm = np.where(ok, P, np.inf)
        k = np.add.reduceat(ok, _G_START, axis=1)                      # [n, G]
        gmin = np.minimum.reduceat(Pm, _G_START, axis=1)
        at = ok & (Pm == gmin[:, _G_OF])                              # the group's argmin(s)
        dirn = np.sign(np.add.reduceat(np.where(at, np.sign(Zs), 0.0), _G_START, axis=1))
        app = np.add.reduceat(at & _APP_ERR_COL, _G_START, axis=1) > 0
        flag = (k > 0) & (gmin * np.maximum(k, 1) < COH_P)
        n_g = (k > 0).sum(axis=0)
        ups = (flag & (dirn > 0)).sum(axis=0)
        dns = (flag & (dirn < 0)).sum(axis=0)
        best = (2.0, "", 0)
        frac: Dict[str, Dict[str, float]] = {}
        for gi in np.flatnonzero(n_g):
            g, ng = GROUP_NAMES[gi], int(n_g[gi])
            up, dn = int(ups[gi]), int(dns[gi])
            frac[g] = {"up": up / ng, "down": dn / ng, "n": ng}
            for x, d in ((up, 1), (dn, -1)):
                if x <= 0:
                    continue
                pt = float(sp.bdtrc(x - 1, ng, COH_RATE))
                if pt < best[0]:
                    is_app = g == "app" and bool((flag[:, gi] & (dirn[:, gi] == d)
                                                  & app[:, gi]).any())
                    best = (pt, "app_error" if is_app else _GROUP_AXIS[g], d)
        if not frac:
            return None
        p = min(1.0, max(best[0], P_FLOOR) * 2 * len(frac)) if best[1] else 1.0
        ax = ["peer", best[1]] if best[1] and p < COH_P else ["peer"]
        return p, ax, frac

    # ------------------------------------------------------------- class_novel
    def _novel(self, sc: _Sys, ck: str, model: Dict[str, Any], memset: Set[str],
               now: float, dt: float, info: Dict[str, Any]
               ) -> Tuple[List[Tuple[float, float]], Optional[Tuple[float, List[str], bool, List]]]:
        """(adoption tallies matured this tick, (p, axes, alarm, top records) or None)."""
        led = sc.ledgers()
        if not sc.has_vocab:
            return [], None                      # B08 never described a class: unscored
        run = model["run"]
        n = len(memset)
        # merge the ledgers' records restricted to this class's members
        merged: Dict[str, Dict[str, Any]] = {}
        for rec, vm in led:
            mem = {ip: _f(t) for ip, t in (rec.get("members") or {}).items() if ip in memset}
            if not mem:
                continue
            key = V.value_key(rec.get("dim"), rec.get("value"))
            t_all = min(_f(t) for t in rec["members"].values())
            ent = ((vm.get("dims") or {}).get(rec.get("dim")) or {}).get(rec.get("value"))
            known = ent is not None and _f(ent[2]) < t_all - ADOPT_KNOWN_SLACK_S
            cur = merged.get(key)
            if cur is None:
                merged[key] = {"dim": rec.get("dim"), "value": rec.get("value"), "members": mem,
                               "flags": dict(rec.get("flags") or {}), "known": known}
            else:
                for ip, t in mem.items():
                    cur["members"][ip] = min(t, cur["members"].get(ip, math.inf))
                for f, v in (rec.get("flags") or {}).items():
                    cur["flags"][f] = bool(cur["flags"].get(f)) or bool(v)
                cur["known"] = cur["known"] and known
        # historical adoption rate (committed statistics)
        ad = _dec_true(model["aux"]["ad"], now, ADOPT_HL_S)
        p_hat, phi = bayes.ratio_posterior(ADOPT_A0, ADOPT_B0, ad[0], ad[1], ad[2], ad[3], 0.0)
        c = min(ad[2] + ADOPT_A0 + ADOPT_B0, float(phi))
        a_bb, b_bb = float(p_hat) * c, (1.0 - float(p_hat)) * c
        score = 0.0
        tallies: List[Tuple[float, float]] = []
        contrib: List[Dict[str, Any]] = []
        new_ext = 0
        tallied = run["tallied"]
        lo = now - ADOPT_WINDOW_S
        for key, r in merged.items():
            if r["known"]:
                continue
            ts = sorted(r["members"].values())
            fl = r["flags"]
            if fl.get("external") and any(now - dt < t <= now for t in ts):
                new_ext += 1
            t1 = ts[0]
            if key not in tallied and now - TALLY_HORIZON_S < t1 <= lo and n > 1:
                s_ = sum(1 for t in ts if t <= t1 + ADOPT_WINDOW_S) - 1
                tallies.append((float(min(s_, n - 1)), float(n - 1)))
                tallied[key] = t1
            m_v = sum(1 for t in ts if t >= lo)
            if m_v < 2 or n < 2:
                continue
            P = float(bayes.bb_sf(min(m_v, n) - 2, n - 1, a_bb, b_bb))
            P = min(1.0, max(P, P_FLOOR))
            w, ax = _adoption_weight(r)
            x = w * -math.log10(P)
            score += x
            contrib.append({"dim": r["dim"], "value": r["value"], "m": m_v, "n": n, "P": P,
                            "w": w, "score": x, "axes": ax, "flags": dict(fl),
                            "first_ts": t1, "key": key})
        for key in [k for k, t in tallied.items() if _f(t) < now - TALLY_HORIZON_S - DAY]:
            del tallied[key]
        info["new_ext"] = new_ext
        p = min(1.0, max(10.0 ** (-score), P_FLOOR)) if score < 300.0 else P_FLOOR
        contrib.sort(key=lambda r: -r["score"])
        if contrib:
            axes = sorted({a for r in contrib if r["score"] >= 1.0 for a in r["axes"]}) \
                or list(contrib[0]["axes"])
            hist = [h for h in run["hist"] if h["key"] not in {r["key"] for r in contrib[:3]}]
            run["hist"] = ([{k: r[k] for k in ("key", "dim", "value", "m", "n", "P", "w", "score")}
                            | {"ts": now} for r in contrib[:3]] + hist)[:HIST_KEEP]
        else:
            axes = ["categorical"]
        info["adopt_rate"] = float(p_hat)
        budget = float(DETECTOR_INFO["class_novel"]["budget_per_day"])
        alarm = combine.e_day(p, dt) <= budget
        return tallies, (p, axes, alarm, contrib)

    # ------------------------------------------------------------------ events
    @staticmethod
    def _events(sc: _Sys, ck: str, model: Dict[str, Any], info: Dict[str, Any],
                top: List[Dict[str, Any]], novel_alarm: int, now: float, dt: float) -> None:
        store, s = sc.store, sc.s
        run = model["run"]
        ver = int(model["gate"].version)
        # coherent intensity-only shift (INFO)
        p_int, p_shape = _f(info.get("p_int")), _f(info.get("p_shape"))
        d = int(info.get("dir_int") or 0)
        coh = (info.get("coherence") or {}).get("volume") or {}
        frac = _f(coh.get("up" if d > 0 else "down"))
        common = emit.read_dict(store, s, ck, COMMON_PREFIX + "volume", now)
        coherent = (frac >= COHERENT_FRAC) or (d != 0 and _idir(common.get("dir")) == d)
        on = (p_int <= EVENT_P) and not (p_shape <= EVENT_P) and coherent
        if on and not run["coh_on"]:
            store.add_event(BehaviorEvent(
                system=s, entity=ck, ts=now, kind="coherent_shift", score=_unit(p_int),
                severity=Severity.INFO,
                description=(f"Coherent intensity-only shift of {ck}: aggregate volume "
                             f"{'up' if d > 0 else 'down'} (p={p_int:.2g}), mix unchanged"),
                extra={"class": ck, "direction": d, "p_int": p_int, "p_shape": p_shape,
                       "frac_members": frac, "members": model["n_members"]},
                p_value=p_int, e_day=combine.e_day(p_int, dt), axes=["volume"],
                p_by_detector={"class_int": p_int}, dedupe_key=f"coherent_shift|{s}|{ck}|{now:.0f}",
                model_version=ver, window=(now - dt, now)))
        run["coh_on"] = bool(on)
        # app-error / transport class shift (LOW); system-wide flag from B05
        ax = set(info.get("axes_shape") or ())
        on2 = (p_shape <= EVENT_P) and bool(ax) and ax <= {"app_error", "transport"}
        if on2 and not run["shift_on"]:
            sysw = any(_idir((sc.common_sys.get(g) or {}).get("dir")) != 0 for g in ax)
            store.add_event(BehaviorEvent(
                system=s, entity=ck, ts=now, kind="class_shift", score=_unit(p_shape),
                severity=Severity.LOW,
                description=(f"{', '.join(sorted(ax))} shift across {ck} (p={p_shape:.2g})"
                             + ("; system-wide" if sysw else "")),
                extra={"class": ck, "axes": sorted(ax), "p_shape": p_shape, "system_wide": sysw},
                p_value=p_shape, e_day=combine.e_day(p_shape, dt), axes=sorted(ax),
                p_by_detector={"class_shape": p_shape},
                dedupe_key=f"class_shift|{s}|{ck}|{now:.0f}", model_version=ver,
                window=(now - dt, now)))
        run["shift_on"] = bool(on2)
        # risky adoption
        risky = run["risky"]
        for key in [k for k, t in risky.items() if _f(t) < now - RISKY_KEEP_S]:
            del risky[key]
        if not novel_alarm:
            return
        p_nov = _f(info.get("p_novel"))
        e = combine.e_day(p_nov, dt)
        sev = combine.e_day_severity(e) or "low"
        sev = "high" if sev == "critical" else sev          # uncorroborated: B25 / B27 grade
        for r in top:
            fl = r["flags"]
            if not (fl.get("external") or fl.get("upload") or fl.get("sensitive")):
                continue
            if r["key"] in risky or r["score"] < 1.0:
                continue
            risky[r["key"]] = now
            store.add_event(BehaviorEvent(
                system=s, entity=ck, ts=now, kind="class_adoption_risky", score=_unit(p_nov),
                severity=Severity(sev),
                description=(f"{r['m']} of {r['n']} members of {ck} adopted {r['dim']} "
                             f"{r['value']} within 24 h (P={r['P']:.2g}, weight {r['w']:g})"),
                extra={"class": ck, "dim": r["dim"], "value": r["value"], "m": r["m"],
                       "n": r["n"], "P": r["P"], "weight": r["w"], "flags": dict(fl),
                       "external": bool(fl.get("external")),
                       "upload_dominant": bool(fl.get("upload")),
                       "sensitive": bool(fl.get("sensitive")),
                       "new_external_domain": bool(fl.get("new_eTLD1_org"))},
                p_value=p_nov, e_day=e, axes=list(r["axes"]),
                p_by_detector={"class_novel": p_nov},
                dedupe_key=f"class_adoption_risky|{s}|{ck}|{r['key']}", model_version=ver,
                window=(r["first_ts"], now)))

    # ---------------------------------------------------------------- learning
    def _learn(self, ctx: Context, s: str, ck: str, model: Dict[str, Any], now: float,
               dt: float) -> None:
        store = ctx.store
        training = bool(ctx.training)
        self._meta = model["meta"]
        try:
            st, gate = self._cur.step(store, s, ck, _Cur(model["current"], model["aux"]),
                                      model["gate"], now, dt, training)
            model["current"], model["aux"], model["gate"] = st.anc, st.aux, gate
            control = store.get_model(s, ck, G.CONTROL_MODEL)
            sig = _control_sig(control)
            if self.entity_due((s, ck, "reference"), now, self.ref_period_s) \
                    or sig != model["ctl_sig"]:
                model["ctl_sig"] = sig
                ivs = self._ref_marks(store, s, ck, now)
                self._ref_ctx = (model, ivs, float(gate.allow_drift))
                try:
                    ref, gref = self._ref.step(store, s, ck, model["reference"],
                                               model["gate_ref"], now, dt, training)
                finally:
                    self._ref_ctx = None
                model["reference"], model["gate_ref"] = ref, gref
        finally:
            self._meta = {}
        self._prune(model, now)

    def _fetch_cur(self, store: Any, s: str, e: str, ts: float) -> Optional[_Row]:
        meta = self._meta.get(ts)
        if meta is None:
            return None
        base = None
        if meta[1] >= 1:
            nat = store.vec_at(s, e, AGG, ts)
            if nat is not None:
                base = MB.make_row(ts, nat, meta[0], self._tctx(ts, meta[0]))
        return _Row(ts, base, meta)

    def _fetch_ref(self, store: Any, s: str, e: str, ts: float) -> Optional[MB.Row]:
        """Active rows with no class incident / abnormal regime within +-24 h
        (decided once per row ts, so a replay makes the same decision)."""
        meta = self._meta.get(ts)
        if meta is None or meta[1] < 1:
            return None
        drift = 0.0
        if self._ref_ctx is not None:
            model, ivs, drift = self._ref_ctx
            el = model["ref_elig"]
            ok = el.get(ts)
            if ok is None:
                ok = el[ts] = not any(a <= ts + DAY and b >= ts - DAY for a, b in ivs)
            if not ok:
                return None
        nat = store.vec_at(s, e, AGG, ts)
        if nat is None:
            return None
        return MB.make_row(ts, nat, meta[0], self._tctx(ts, meta[0]), drift=drift)

    def _on_rebase_cur(self, state: _Cur, tau: float) -> _Cur:
        """ACCEPTED: the new regime's rows are committed without the rate cap."""
        MB.on_rebase_current(state.anc, tau, until=self._now + DAY)
        return state

    @staticmethod
    def _ref_marks(store: Any, s: str, ck: str, now: float) -> List[Tuple[float, float]]:
        """Class incidents and abnormal class regimes (intervals) that keep
        rows out of the reference anchor."""
        ivs: List[Tuple[float, float]] = []
        for inc in store.incidents(system=s, entity=ck, since=now - META_KEEP_S - DAY):
            b = now if inc.status == "open" else float(inc.last_seen)
            ivs.append((float(inc.opened), max(b, float(inc.opened))))
        since = None
        for p in store.derived_tail(s, ck, REGIME, 256):
            st = p.value.get("state") if isinstance(p.value, Mapping) else None
            abn = st is not None and str(st).lower() in ABNORMAL_REGIMES
            if abn and since is None:
                since = float(p.ts)
            elif not abn and since is not None:
                ivs.append((since, float(p.ts)))
                since = None
        if since is not None:
            ivs.append((since, now))
        return ivs

    @staticmethod
    def _prune(model: Dict[str, Any], now: float) -> None:
        """Hourly: learning extras older than any replay (tokens after 26 h),
        reference decisions, stale run bookkeeping."""
        run = model["run"]
        if now - float(run["prune_ts"]) < HOUR and now >= float(run["prune_ts"]):
            return
        run["prune_ts"] = now
        meta = model["meta"]
        for ts in [t for t in meta if t < now - META_KEEP_S]:
            del meta[ts]
        for ts, row in list(meta.items()):
            if row[5] is not None and ts < now - TOK_KEEP_S:
                meta[ts] = row[:5] + (None,)
        el = model["ref_elig"]
        for ts in [t for t in el if t < now - META_KEEP_S]:
            del el[ts]

    # ----------------------------------------------------------------- profile
    def _profile(self, store: Any, s: str, ck: str, model: Dict[str, Any],
                 tc: Mapping[str, Any], now: float) -> None:
        pred = MB.anchor_predictive(model["current"], tc)
        names = [F.FEATURE_NAMES_V2[i] for i in VOL_IDX]
        # volume features only: the ratio ppf searches would cost ~5 ms
        pred = dataclasses.replace(pred, c=np.full(NF, np.nan))
        Q = MB.quantiles(pred, [0.05, 0.5, 0.95], dt_s=900.0)
        agg = {n: [_fin(Q[0, i]), _fin(Q[1, i]), _fin(Q[2, i])] for n, i in zip(names, VOL_IDX)}
        rh = _dec_true(model["aux"]["rh"], now, RHYTHM_HL_S)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(rh[:48, 2] > 0.0, rh[:48, 1] / rh[:48, 2], np.nan)
        ad = _dec_true(model["aux"]["ad"], now, ADOPT_HL_S)
        run = model["run"]
        extra = {
            "kind": model.get("class_kind"), "members": list(model["members"]),
            "n_members": int(model["n_members"]), "bucket": int(tc["bin48"]),
            "aggregate": {"quantiles": [0.05, 0.5, 0.95], "exposure_s": 900.0,
                          "features": agg},
            "active_frac_by_bin": [_fin(x) for x in frac.tolist()],
            "adoption": {"rate": _fin((ad[1] + ADOPT_A0) / (ad[2] + ADOPT_A0 + ADOPT_B0)),
                         "n_records": _fin(ad[0]), "recent": [dict(h) for h in run["hist"]]},
            "last": dict(run["last"]), "n_eff": _fin(MB.n_eff(model["current"])),
            "version": int(model["version"]), "updated": now,
        }
        prof = store.profile(s, ck) or EntityProfile(system=s, entity=ck, updated=now)
        if not isinstance(prof.extra, dict):
            prof.extra = {}
        prof.extra["class_monitor"] = extra
        prof.updated = now
        store.put_profile(prof)


# ================================================================ helpers
def _adoption_weight(r: Mapping[str, Any]) -> Tuple[float, List[str]]:
    """w_v and axes of one adopted value (engines.md B18 step 3)."""
    fl = r.get("flags") or {}
    ext, up = bool(fl.get("external")), bool(fl.get("upload"))
    sens, new_org = bool(fl.get("sensitive")), bool(fl.get("new_eTLD1_org"))
    w = W_EXTERNAL if ext else 1.0
    if up:
        w *= W_UPLOAD
    if sens:
        w *= W_SENSITIVE
    if new_org:
        w *= W_NEW_ORG
    if not (ext or up or sens or new_org) and r.get("dim") == "tmpl":
        method = str(r.get("value") or "").split(" ", 1)[0].upper()
        if method in READ_METHODS:
            w = W_INTERNAL_READ
    if ext and up:
        ax = ["exfil"]
    elif ext or new_org:
        ax = ["c2"]
    else:
        ax = ["categorical"]
    return w, ax


def _score(p: float) -> float:
    return -math.log10(max(p, P_FLOOR)) if p == p else _NAN


def _unit(p: float) -> float:
    """Event score in [0, 1] from a p-value (-log10 p / 10, capped)."""
    return min(1.0, _score(p) / 10.0) if p == p else 0.0


def _idir(x: Any) -> int:
    v = _f(x)
    return 0 if v != v else (1 if v > 0 else -1 if v < 0 else 0)


def _fin(x: Any) -> Optional[float]:
    v = _f(x)
    return v if math.isfinite(v) else None
