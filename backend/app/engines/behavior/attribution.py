"""AttributionEngine (B16): who is behind this IP's traffic right now?

Why: an IP is an address, not a person. When a stolen session or credential
is replayed from another desk, when a known host is swapped for another
behind the same address, or when a role account is taken over, the IP's own
detectors (all normalised to the IP itself) may see nothing unusual in
volume or shape. The question has to be asked the other way round: under
whose model is this window of behaviour most likely, and is that the IP
itself, somebody else we know, or nobody we know (open set)?

Identity is ABSOLUTE (architecture §0 rule 7, decisions.md). Every modality
scores the window under each candidate's OWN models on absolute data:
feature.vec, feature.sketch, behavior.timing, act.tokens, act.stream,
client.stack_set and the activity clock. It never reads the self-normalised
per-feature z series, which remove exactly the between-entity information
attribution needs (a static test asserts this). The window representation,
the metric and the modality LLRs come from lib/m_identity (owner B15), so
the numbers B15 calibrated on and the numbers scored here are the same.

Per system and tick (docs/lib3/engines.md '## B16'):
  1. Window: the entity's last K = 4 active ticks (feature.active), sliding
     by one active tick. Idle ticks abstain: nothing is scored and the
     CUSUMs keep their value. Tick rows (m_identity.tick_row) and the raw
     modality evidence of a tick (m_identity.tick_modal_data; raw sets live
     1 h) are cached when the tick is scored, so a window spanning hours of
     idle time is still complete.
  2. z = m_identity.transform(window_vector(rows)) in B15's LDA space
     (within-entity covariance = I). window_vector here is bit-identical to
     m_identity.window_vector (tested) without numpy's masked-array
     nanmedian / apply_along_axis nanpercentile, ~10x cheaper for 4 rows.
  3. Candidates: self, the top 5 other enrolled entities by Gaussian
     score, enrolled linked aliases (B17), the own role class and the best
     other role class.
  4. Modality LLRs against the system background (the "somebody" / unknown
     reference), each under the candidate's own models: gauss, vocab (x
     min(n_tok, 20)/20), rhythm presence and client stacks every tick; PPM
     and gap terms at window completion (every K active ticks, cached per
     candidate; a candidate that joins in between is scored on the cached
     window evidence). Each is calibrated a_m * llr + b_m and capped at +-4
     nats (m_identity.calibrate), so one modality (a browser upgrade, a new
     client library) can never flip identity on its own. No evidence (NaN)
     contributes 0, the background, never a guess.
  5. Posterior over the candidates plus an explicit unknown hypothesis
     (log-evidence 0 = the background; prior 0.05; self 0.5; the other
     candidates share 0.45). Linked aliases count as self.
     score.identity = -log10 p_self, p_self the self-typicality tail of the
     window's LDA distance calibrated on B15's held-out genuine windows of
     the same model version (m_identity.typicality_p; round 4, see
     score_p); pm.identity = p_self, a valid p-value as B24's small-sample
     prior. B24 calibrates it with (daypart, regime) strata. The axis
     identity follows pi_self < 0.5 (instantaneous).
  6. Other-identity CUSUM: lambda = clip(max_{j != e} L_j - L_e, -4, 8),
     S = max(0, S + lambda - 0.5), h = seq.h_for('llr', 100 d) counted in the
     entity's windows per day (its active ticks over the last day). At
     S >= h: identity_mismatch, MEDIUM, HIGH when j* is active at the same
     time (impersonation), INFO unless j* won >= 3 of the last 4 windows,
     pi_j* >= 0.9, the CV confusion(e, j*) < 0.1 and j* is not in e's
     anonymity set. The own role class never counts as "another identity".
  7. Unknown CUSUM: +1 when max_j p_j < 0.01 (chi2 typicality of z under
     each individual candidate, its squared distance calibrated by the
     candidate's held-out genuine T99 from B15: t99_scale), else -0.5;
     unknown_identity at 2: HIGH when the class typicality p < 0.01 ('unlike anyone'; the system background
     stands in when the IP has no class), MEDIUM for 'same class, different
     individual', INFO for a young (not enrolled) IP.
  8. An entity with continuity.shared_ip (or entity_kind 'ip-class', B17)
     gets every event downgraded to INFO. The common-mode flag is never read:
     it must not suppress identity.

Cost: the cheap modalities are summed from per-tick, per-candidate LLRs
cached when a tick is first scored (vocab, client and rhythm are additive
over a window's counts / cells), with the owners' backoff chains resolved
once per candidate and tick; ~15-20 us per candidate, ~0.75 ms per entity
and tick in all (tests/engines/test_b16_attribution_edges.py measures it).

Training mode scores and runs the CUSUMs but emits no events (a chart that
reaches its threshold is reset silently, so warm-up never leaves it primed).
B16 owns no model; its chart state lives in the engine and is mirrored in
behavior.id, so a restarted engine resumes it (alert holds are rebuilt from
the event index). A failed B01 tick writes NaN + degraded; an unfitted
model.identity means nothing is enrolled yet and B16 abstains.

spec v2.1 windows (docs/lib3/cadence.md §8; canonical grain mode): a window
is the last 4 ACTIVE H ROWS (m_identity.grain_row: feature.vec.h,
feature.sketch.h, behavior.timing and the clock of the row's midpoint), i.e.
4 h of wall-clock data at every cadence, the windows B15 enrols. B16 runs on
H decision ticks only; the modality evidence of a row is the H window's
merged evidence (m_identity.grain_modal_data). The instantaneous score is
issued on every active H row (overlapping windows: single-tick path only);
the two CUSUMs step only on NON-overlapping windows (every 4th active H
row), with h counted in those windows per day. A full window is 4 covered
H rows (cov >= 0.95 h) within a 48-h lookback.

Store: reads feature.active, feature.vec, feature.sketch, feature.tctx,
act.tokens, act.stream, act.stream_frac, tls/dns/l4 sets, client.stack_set,
behavior.timing, model.identity, model.vocab, model.rhythm, model.seq,
model.client, model.timing, model.class, model.link,
profile.extra.continuity. Writes behavior.score[identity] / pm / axes /
degraded (lib/emit), behavior.id (dict series), profile.extra.attribution;
events identity_mismatch, unknown_identity.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.special import chdtrc, chdtri

from ...core.engine import Context, Engine
from ...models.schema import BehaviorEvent, DerivedMetric, EntityProfile, MetricKind, Severity
from .lib import emit
from .lib import pactive as PA
from .lib import grains as GR
from .lib import m_class
from .lib import m_client as MC
from .lib import m_identity as MI
from .lib import m_rhythm as MR
from .lib import m_seq as MS
from .lib import m_vocab as MV
from .lib import seq as SQ
from .lib.classkeys import SYSTEM_KEY, is_class, role_key

DETECTOR = "identity"
AXES = ["identity"]
B01_ENGINE = "behavior.feature_vector"
LINK_MODEL = "model.link"
ID_SERIES = "behavior.id"
ACTIVE = "feature.active"

K = MI.K_WIN                           # active ticks per window (4)
MODALITIES = MI.MODALITIES
CAP = MI.LLR_CAP                       # nats per modality per window
VOCAB_N = MI.VOCAB_N                    # vocab LLR x min(n_tok, 20)/20 (lib/m_identity)
N_TOP = 5                              # other entities by LDA score
UNKNOWN_PRIOR = 0.05
SELF_PRIOR = 0.5
LAMBDA_LO, LAMBDA_HI, CUSUM_K = -4.0, 8.0, 0.5
ARL_DAYS = 100.0
U_UP, U_DOWN, U_H = 1.0, 0.5, 2.0
P_TYPICAL = 0.01
POST_MIN, WINS_MIN, CONF_MAX = 0.9, 3, 0.1
LOOKBACK_S = 86400.0                   # activity history for windows / day (feature.vec: 1 d)
LOOKBACK_H_S = 2 * 86400.0             # spec v2.1: H rows (feature.vec.h is kept 2 d)
META_H = "feature.meta.h"
SEQ_MAX, GAP_MAX = 128, 256            # window evidence bounds (PPM cost)
ALERT_HOLD_S = 6 * 3600.0
PROFILE_EVERY_S = 3600.0
AXES_POST_MAX = 0.5                    # pi_self below this -> the identity axis contributes
_NAN = math.nan


# ============================================================ small helpers
def _f(x: Any) -> float:
    if isinstance(x, bool) or x is None:
        return _NAN
    try:
        return float(x)
    except (TypeError, ValueError, OverflowError):
        return _NAN


def _int(x: Any) -> int:
    v = _f(x)
    return int(v) if math.isfinite(v) else 0


def _json(v: Any, nd: int = 4) -> Any:
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return round(v, nd) if math.isfinite(v) else None
    return v


def chi2_p(q: float, k: int) -> float:
    """Upper chi2_k tail of a squared Mahalanobis distance (NaN in, NaN out)."""
    if not (q == q) or k < 1:
        return _NAN
    return float(max(chdtrc(k, max(q, 0.0)), 1e-300))


def t99_scale(model: Mapping[str, Any], cand: str, r: int) -> float:
    """Factor that calibrates an entity's squared LDA distance before its chi2_r
    tail: chi2_r's 99 % point over the entity's own held-out genuine 99 %
    point T99 (B15's blocked CV), so p < P_TYPICAL <=> d2 > T99.

    Why: the within-entity covariance is I only for the pooled (WCCN) average;
    held-out genuine windows of one entity sit much farther from its mean
    than chi2_r says (eval pack A: T99 = 200-1100 for workstations against a
    chi2_14 99 % point of 29), so the raw chi2 tail called most of a
    person's own windows 'unlike every known individual' while their self
    posterior was > 0.98 (unknown_identity MEDIUM, 'same class, different
    individual', on 13 of 19 enrolled humans in a clean week). 1 when B15
    has no T99 for the candidate."""
    t = MI.t99(model, cand)
    if not (t == t and t > 0.0) or r < 1:
        return 1.0
    return float(chdtri(r, P_TYPICAL)) / t


def score_p(model: Mapping[str, Any], z: np.ndarray, selfish: Sequence[str],
            pi_self: float) -> float:
    """p behind score.identity (round 4): the largest held-out-calibrated
    self-typicality over the entity and its linked aliases, i.e. how unusual
    this window is for the identity that owns the address, on the scale of
    that identity's own held-out windows under the SAME model version.

    Why not -log10 pi_self (v2): the posterior's null moved with every daily
    refit (modality calibration) and every role relabel (the own class is a
    candidate), so B24's rings, filled under earlier versions, did not match
    the live null (pack A API clients: live 0.005-0.009 against a ring max
    of 0.003-0.005, identity p < 1e-3 on 25x the nominal share of clean
    ticks). The posterior still drives the axis and the two CUSUMs. A model
    without typicality fits (older layout) keeps pi_self; an entity that is
    not enrolled has no identity to be typical of: NaN."""
    ps = [MI.self_typicality(model, z, j) for j in selfish]
    ps = [p for p in ps if p == p]
    if ps:
        return float(max(ps))
    if isinstance(model, Mapping) and model.get("typ"):
        return _NAN
    return pi_self


def typicality(model: Mapping[str, Any], z: np.ndarray, cand: str) -> float:
    """p of the window under a candidate's Gaussian in LDA space: an entity
    (covariance I, the distance calibrated by its held-out T99: t99_scale),
    a class key (diagonal class_var) or SYSTEM_KEY (the full background
    Gaussian). NaN when the candidate is not in the model."""
    r = int(z.size)
    m = MI.entity_mean(model, cand)
    if m is not None and m.size == r:
        return chi2_p(float(np.sum((z - m) ** 2)) * t99_scale(model, cand, r), r)
    m = MI.class_mean(model, cand)
    if m is not None and m.size == r:
        v = np.maximum(np.asarray((model.get("class_var") or {}).get(cand, np.ones(r)),
                                  dtype=np.float64), 1e-6)
        return chi2_p(float(np.sum((z - m) ** 2 / v)), r)
    bg = model.get("bg")
    if cand == SYSTEM_KEY and isinstance(bg, Mapping) and bg.get("prec") is not None:
        d = z - np.asarray(bg["mu"], dtype=np.float64)
        return chi2_p(float(d @ np.asarray(bg["prec"], dtype=np.float64) @ d), r)
    return _NAN


def evidence(calib: Mapping[str, Tuple[float, float]], llr: Mapping[str, float]) -> float:
    """L_j: sum over modalities of clip(a_m llr + b_m, -4, 4) (m_identity.calibrate
    with (a_m, b_m) = m_identity.llr_calib resolved once per tick); a NaN
    modality (no evidence) contributes 0, the background."""
    tot = 0.0
    for m in MODALITIES:
        x = llr.get(m, _NAN)
        if x == x and not math.isinf(x):
            a, b = calib[m]
            tot += min(CAP, max(-CAP, a * x + b))
    return tot


# ============================================================ fast window vector
def window_vector(rows: Any) -> np.ndarray:
    """B15's representation: lib/m_identity.window_vector (integration R20.0:
    B15 moved to the one-sort quantile path, bit-identical to B16's former
    local copy and as fast, so B15 / B16 / B17 share one implementation)."""
    return MI.window_vector(rows)


def _is_active(store: Any, s: str, e: str, ts: float, name: str = ACTIVE) -> bool:
    a = store.vec_at(s, e, name, ts)
    return a is not None and float(a[0]) >= 0.5


# ============================================================ per-entity state
def _new_state() -> Dict[str, Any]:
    return {"S": 0.0, "U": 0.0, "winners": [], "n_act": 0, "rows": {}, "md": {}, "cheap": {},
            "slow": None, "hold": {}, "ts": None, "prev": None, "profile_ts": -math.inf}


def _snap(st: Mapping[str, Any]) -> Dict[str, Any]:
    return {"S": st["S"], "U": st["U"], "winners": list(st["winners"]), "n_act": st["n_act"],
            "rows": dict(st["rows"]), "md": dict(st["md"]),
            "cheap": {t: dict(v) for t, v in st["cheap"].items()}, "slow": st["slow"],
            "hold": dict(st["hold"]), "profile_ts": st["profile_ts"]}


def _restore_state(store: Any, s: str, e: str, now: float) -> Dict[str, Any]:
    """Resume the charts from the last behavior.id point before `now` (an
    engine restart must not reset a half-way CUSUM) and the alert holds from
    the event index."""
    st = _new_state()
    for m in reversed(store.derived_tail(s, e, ID_SERIES, 4)):
        if m.ts < now and isinstance(m.value, Mapping):
            v = m.value
            S, U = _f(v.get("cusum_other")), _f(v.get("cusum_new"))
            st["S"] = S if S == S and S >= 0.0 else 0.0
            st["U"] = U if U == U and U >= 0.0 else 0.0
            st["winners"] = [str(x) for x in (v.get("winners") or [])][-K:]
            st["n_act"] = _int(v.get("n_act"))
            break
    for ev in store.events(system=s, entity=e, since=now - ALERT_HOLD_S,
                           kinds=["identity_mismatch", "unknown_identity"], limit=50):
        key = ev.extra.get("hold_key") if isinstance(ev.extra, Mapping) else None
        if key:
            st["hold"][str(key)] = max(st["hold"].get(str(key), -math.inf), float(ev.ts))
    return st


def continuity(store: Any, s: str, e: str) -> Tuple[Set[str], bool]:
    """(aliases of e from profile.extra.continuity, shared-IP flag)."""
    aliases: Set[str] = set()
    shared = False
    prof = store.profile(s, e)
    c = prof.extra.get("continuity") if prof is not None and isinstance(prof.extra, Mapping) \
        else None
    if isinstance(c, Mapping):
        for a in c.get("aliases") or ():
            aliases.add(str(a))
        for k in ("linked_from", "linked_to"):
            v = c.get(k)
            if isinstance(v, str) and v:
                aliases.add(v)
            elif isinstance(v, (list, tuple)):
                aliases.update(str(x) for x in v if x)
        shared = bool(c.get("shared_ip")) or c.get("entity_kind") == "ip-class"
    aliases.discard(e)
    return aliases, shared


def link_aliases(store: Any, s: str) -> Dict[str, Set[str]]:
    """Unretracted links of model.link@(s, __system__) as an alias map (read
    defensively: B17 owns the layout; entries are dicts with from/to or a/b,
    or [a, b, ...] pairs)."""
    out: Dict[str, Set[str]] = {}
    m = store.get_model(s, SYSTEM_KEY, LINK_MODEL)
    links = m.get("links") if isinstance(m, Mapping) else None
    if isinstance(links, Mapping):
        links = list(links.values())
    for ln in links or ():
        a = b = None
        if isinstance(ln, Mapping):
            if ln.get("retracted") or ln.get("status") == "retracted":
                continue
            a = ln.get("from", ln.get("a", ln.get("src")))
            b = ln.get("to", ln.get("b", ln.get("dst")))
        elif isinstance(ln, (list, tuple)) and len(ln) >= 2:
            a, b = ln[0], ln[1]
        if isinstance(a, str) and isinstance(b, str) and a != b:
            out.setdefault(a, set()).add(b)
            out.setdefault(b, set()).add(a)
    return out


def _vocab_ll(bo: MV.Backoff, vocab: Mapping[str, Mapping[str, float]]) -> float:
    ll = 0.0
    for dim, vals in vocab.items():
        for n, p in zip(vals.values(), bo.probs(dim, vals.keys())):
            ll += n * math.log(max(p, 1e-300))
    return ll


def _client_ll(bo: MC.Backoff, stacks: Mapping[str, float]) -> float:
    return float(sum(n * math.log(max(bo.p(t), MC.P_FLOOR)) for t, n in stacks.items()))


def tick_background(sc: "_Sys", md: Any) -> Tuple[float, float, float]:
    """Background log-likelihoods (vocab, client, rhythm) of one tick's
    evidence: shared by every candidate, so computed once per tick."""
    v = _vocab_ll(sc.bg_vocab(), md.vocab) if md.vocab else _NAN
    c = _client_ll(sc.bg_client(), md.stacks) if md.stacks else _NAN
    r = _NAN
    if md.cells:
        pb = [sc.bg.rhythm_p(*x) for x in md.cells]
        if all(math.isfinite(p) for p in pb):
            r = sum(math.log(min(1.0 - 1e-6, max(1e-6, p))) for p in pb)
    return v, c, r


def tick_cheap_llrs(sc: "_Sys", cand: str, md: Any, bg: Tuple[float, float, float]
                    ) -> Tuple[float, float, float]:
    """(vocab, client, rhythm) LLRs of ONE tick's evidence under `cand`
    against the background `bg` (tick_background): m_identity.modality_logliks'
    maths with the chains resolved once per tick. All three are sums over the
    tick's counts / cells, so a window's LLR is the sum of its ticks' LLRs."""
    v = c = r = _NAN
    if md.vocab:
        bo = sc.vocab_bo(cand)
        if bo is not None:
            v = _vocab_ll(bo, md.vocab) - bg[0]
    if md.stacks:
        bo = sc.client_bo(cand)
        if bo is not None:
            c = _client_ll(bo, md.stacks) - bg[1]
    if md.cells and bg[2] == bg[2]:
        model = sc.bg._model(MR.MODEL, cand)
        if isinstance(model, Mapping) and "state" in model:
            ll = MR.loglik(model, [1.0] * len(md.cells), list(md.cells))
            if math.isfinite(ll):
                r = ll - bg[2]
    return v, c, r


class _Sys:
    """One system at one tick: the identity model, the modality background
    and small lookups shared by every entity (candidates overlap)."""

    def __init__(self, store: Any, s: str, now: float, model: Mapping[str, Any],
                 active_name: str = ACTIVE) -> None:
        self.store, self.s, self.now, self.model = store, s, now, model
        self.active_name = active_name
        self.bg = MI.Background(store, s, now)
        self.smap = MS.SymbolMap.from_store(store, s)
        self.enrolled = set((model.get("means") or {}).keys())
        self.calib = {m: MI.llr_calib(model, m) for m in MODALITIES}
        self.classes = [c for c in (model.get("class_means") or {})
                        if is_class(c) and not c.startswith(("class:static:", "class:pool:"))]
        self.links = link_aliases(store, s)
        self._role: Dict[str, Optional[str]] = {}
        self._active: Dict[str, bool] = {}
        self._ck: Dict[str, Optional[str]] = {}
        self._n_members: Dict[str, int] = {}
        self._vocab: Dict[str, Optional[MV.Backoff]] = {}
        self._client: Dict[str, Optional[MC.Backoff]] = {}
        self._bg_vocab: Optional[MV.Backoff] = None
        self._bg_client: Optional[MC.Backoff] = None

    # cheap-modality tiers, resolved once per candidate and tick (the same
    # chains m_identity.modality_logliks builds on every call)
    def class_key(self, e: str) -> Optional[str]:
        """m_class.class_key with the role sizes resolved once per tick."""
        if e not in self._ck:
            rid = m_class.role_id(self.store, self.s, e)
            ck = None
            if rid is not None:
                n = self._n_members.get(rid)
                if n is None:
                    n = self._n_members[rid] = len(m_class.members(self.store, self.s, rid))
                ck = role_key(rid) if n >= m_class.MIN_MEMBERS else None
            self._ck[e] = ck
        return self._ck[e]

    def vocab_bo(self, cand: str) -> Optional[MV.Backoff]:
        """The m_vocab.backoff_models chain of `cand` (class keys cached)."""
        if cand not in self._vocab:
            sys_ = self.bg.vocab_sys
            if is_class(cand):
                ent, cls = None, MV.get(self.store, self.s, cand)
            else:
                ck = self.class_key(cand)
                ent = MV.get(self.store, self.s, cand)
                cls = MV.get(self.store, self.s, ck) if ck else None
            self._vocab[cand] = MV.Backoff(ent, cls, sys_, now=self.now) \
                if (ent is not None or cls is not None or sys_ is not None) else None
        return self._vocab[cand]

    def client_bo(self, cand: str) -> Optional[MC.Backoff]:
        """The m_client.backoff_models chain of `cand` (class keys cached)."""
        if cand not in self._client:
            sys_ = self.bg.client_sys
            if is_class(cand):
                ent, cls = None, MC.class_tier(sys_, cand)
            else:
                ent = MC.get(self.store, self.s, cand)
                cls = MC.class_tier(sys_, self.class_key(cand))
            ok = ent is not None or cls is not None or sys_ is not None
            ent = ent if MC.kind(ent) == "entity" else None
            self._client[cand] = MC.Backoff(ent, cls, sys_, now=self.now) if ok else None
        return self._client[cand]

    def bg_vocab(self) -> MV.Backoff:
        if self._bg_vocab is None:
            self._bg_vocab = MV.Backoff(None, None, self.bg.vocab_sys, now=self.now)
        return self._bg_vocab

    def bg_client(self) -> MC.Backoff:
        if self._bg_client is None:
            self._bg_client = MC.Backoff(None, None, self.bg.client_sys, now=self.now)
        return self._bg_client

    def role_key(self, e: str) -> Optional[str]:
        if e not in self._role:
            rid = m_class.role_id(self.store, self.s, e)
            self._role[e] = role_key(rid) if rid is not None else None
        return self._role[e]

    def active(self, e: str) -> bool:
        if e not in self._active:
            self._active[e] = _is_active(self.store, self.s, e, self.now, self.active_name)
        return self._active[e]


# ============================================================ the engine
class AttributionEngine(Engine):
    name = "behavior.attribution"
    layer = "behavior"
    consumes = [ACTIVE, "feature.vec", "feature.sketch", "feature.tctx", "act.tokens",
                "act.stream", "act.stream_frac", "tls.sni_etld1_set", "dns.qname_etld1_set",
                "l4.dport_set", "client.stack_set", "behavior.timing", MI.MODEL,
                "model.vocab", "model.rhythm", "model.seq", "model.client", "model.timing",
                "model.class", LINK_MODEL, "profile.extra.continuity"]
    produces = ["behavior.score", "behavior.pm", "behavior.axes", "behavior.degraded",
                ID_SERIES, "profile.extra.attribution",
                "event.identity_mismatch", "event.unknown_identity"]
    description = ("Open-set identity attribution per IP: a 4-active-tick absolute window "
                   "scored under self, the nearest known IPs and role classes with calibrated, "
                   "capped modality LLRs (LDA Gaussian, vocab, rhythm, PPM, client, gaps); "
                   "posterior with an unknown hypothesis; other-identity and unknown CUSUMs.")
    interval = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._state: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._canon = False
        self._config: Dict[str, Any] = {}

    # ------------------------------------------------------------------ run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        now, dt = float(ctx.now), float(ctx.window_s)
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError(f"AttributionEngine: bad ctx.window_s {ctx.window_s!r}")
        b01_failed = store.engine_failed(B01_ENGINE, now)
        self._canon = GR.canonical(ctx.config)
        self._config = ctx.config
        if self._canon and not GR.decision(now, dt, "h", GR.CANONICAL):
            return 0                           # spec v2.1: identity is scored on H rows
        return sum(self._system(ctx, s, now, dt, b01_failed) for s in store.systems())

    def _system(self, ctx: Context, s: str, now: float, dt: float, b01_failed: bool) -> int:
        store = ctx.store
        model = MI.get(store, s)
        if not MI.is_fitted(model):
            return 0                                  # nothing enrolled yet: abstain
        act_name = META_H if self._canon else ACTIVE
        sc = _Sys(store, s, now, model, act_name)
        n = 0
        for e in PA.entities(store, s, now, ctx.config):
            key = (s, e)
            if b01_failed:
                if key in self._state or store.vec_at(s, e, act_name, now - dt) is not None:
                    emit.write_scores(store, s, e, now, {DETECTOR: _NAN},
                                      degraded={DETECTOR: f"producer_error:{B01_ENGINE}"},
                                      window_s=int(dt))
                    n += 1
                continue
            if not sc.active(e):
                continue                              # no row / idle: abstain
            st = self._state.get(key)
            if st is None:
                st = self._state[key] = _restore_state(store, s, e, now)
            if st["ts"] == now and st["prev"] is not None:
                st.update(_snap(st["prev"]))          # re-run of the same tick
            else:
                st["prev"], st["ts"] = _snap(st), now
            n += self._entity(ctx, sc, e, st, now, dt)
        return n

    # ------------------------------------------------------------- per entity
    def _entity(self, ctx: Context, sc: _Sys, e: str, st: Dict[str, Any], now: float,
                dt: float) -> int:
        store, s, model = ctx.store, sc.s, sc.model
        win = int(dt)

        # ---- window: last K active ticks; windows per day for the threshold
        canon = self._canon
        full_ok = True
        if canon:
            # spec v2.1: the last K active H rows (48-h lookback); the chart
            # steps once per NON-overlapping window, so h counts those per day
            ts_a, A = store.vec_since(s, e, META_H, now - LOOKBACK_H_S)
            ts_a = np.asarray(ts_a, dtype=np.float64)
            Am = np.asarray(A, dtype=np.float64).reshape(len(ts_a), -1) if ts_a.size \
                else np.zeros((0, 5))
            act = Am[:, 0] if ts_a.size else np.zeros(0)
            on = ts_a[act >= 0.5]
            cov_on = Am[act >= 0.5, 1] if ts_a.size else np.zeros(0)
            wts = [float(t) for t in on[-K:]]
            full_ok = bool(cov_on[-K:].size == K and np.all(
                cov_on[-K:] >= GR.COVER_MIN * GR.GRAIN_S["h"]))
            per = max(dt, GR.GRAIN_S["h"])
            span = max(now - float(ts_a[0]) + per, per) if ts_a.size else per
            rpd = min(max(float(on.size) * 86400.0 / min(span, LOOKBACK_H_S), 1.0),
                      86400.0 / per)
            wpd = max(rpd / K, 1.0 / K)
            h = SQ.h_for("llr", ARL_DAYS, 86400.0 / wpd)
        else:
            ts_a, A = store.vec_since(s, e, ACTIVE, now - LOOKBACK_S)
            ts_a = np.asarray(ts_a, dtype=np.float64)
            act = np.asarray(A, dtype=np.float64).reshape(-1) if ts_a.size else np.zeros(0)
            on = ts_a[act >= 0.5]
            wts = [float(t) for t in on[-K:]]
            span = max(now - float(ts_a[0]) + dt, dt) if ts_a.size else dt
            wpd = min(max(float(on.size) * 86400.0 / min(span, LOOKBACK_S), 1.0), 86400.0 / dt)
            h = SQ.h_for("llr", ARL_DAYS, 86400.0 / wpd)

        # ---- per-tick caches: tick rows and raw modality evidence
        row = (MI.grain_row(store, s, e, now, config=self._config, dt=dt) if canon
               else MI.tick_row(store, s, e, now))
        if row is None:
            emit.write_scores(store, s, e, now, {DETECTOR: _NAN},
                              degraded={DETECTOR: "stale:feature.vec"}, window_s=win)
            return 1
        keep = set(wts)
        st["rows"] = {t: r for t, r in st["rows"].items() if t in keep}
        st["md"] = {t: m for t, m in st["md"].items() if t in keep}
        st["cheap"] = {t: m for t, m in st["cheap"].items() if t in keep and t != now}
        st["rows"][now] = row
        st["md"][now] = (MI.grain_modal_data(store, s, e, now, smap=sc.smap, config=self._config,
                                             dt=dt) if canon
                         else MI.tick_modal_data(store, s, e, now, smap=sc.smap))
        rows = []
        for t in wts:
            r = st["rows"].get(t)
            if r is None:                             # restart: rebuild from the rings
                r = (MI.grain_row(store, s, e, t, config=self._config, dt=dt) if canon
                     else MI.tick_row(store, s, e, t))
                if r is not None:
                    st["rows"][t] = r
            if r is not None:
                rows.append(r)
        st["n_act"] = int(st["n_act"]) + 1
        # spec v2.1: the CUSUMs step only on non-overlapping windows
        step = (not canon) or int(st["n_act"]) % K == 0

        # ---- identity space
        ck = sc.role_key(e)
        z = MI.transform(model, window_vector(np.vstack(rows)), class_key=ck)
        if not z.size or not np.all(np.isfinite(z)):
            emit.write_scores(store, s, e, now, {DETECTOR: _NAN},
                              degraded={DETECTOR: "unscorable:model.identity"}, window_s=win)
            return 1

        # ---- candidates
        aliases, shared = continuity(store, s, e)
        aliases |= sc.links.get(e, set())
        selfish = {e} | aliases
        l_all = MI.scores(model, z)                       # {entity: l_e(z)}, vectorised
        others = [j for j, _ in sorted(((j, v) for j, v in l_all.items() if j != e),
                                       key=lambda t: (-t[1], t[0]))[:N_TOP]]
        cands: List[str] = [e] + others + sorted(a for a in aliases
                                                  if a in sc.enrolled and a not in others)
        own_cls = ck if ck in sc.classes else None
        rest = [(MI.gauss_loglik(model, z, c), c) for c in sc.classes if c != own_cls]
        rest = [(v, c) for v, c in rest if v == v]
        cands += ([own_cls] if own_cls else []) + ([max(rest)[1]] if rest else [])

        # ---- modality LLRs
        l_bg = MI.bg_loglik(model, z)
        llr: Dict[str, Dict[str, float]] = {
            j: {"gauss": (l_all[j] if j in l_all else MI.gauss_loglik(model, z, j)) - l_bg}
            for j in cands}
        self._cheap(sc, st, wts, cands, llr)
        self._slow(sc, st, wts, cands, llr)
        L = {j: evidence(sc.calib, llr[j]) for j in cands}

        # ---- posterior with the unknown hypothesis
        n_oth = max(1, len(cands) - 1)
        lp = {j: L[j] + math.log(SELF_PRIOR if j == e else
                                  (1.0 - SELF_PRIOR - UNKNOWN_PRIOR) / n_oth) for j in cands}
        lp_unk = math.log(UNKNOWN_PRIOR)
        mx = max(max(lp.values()), lp_unk)
        zsum = sum(math.exp(v - mx) for v in lp.values()) + math.exp(lp_unk - mx)
        post = {j: math.exp(v - mx) / zsum for j, v in lp.items()}
        p_unknown = math.exp(lp_unk - mx) / zsum
        pi_self = sum(post[j] for j in cands if j in selfish)

        # ---- other-identity CUSUM and the window winners
        rivals = [j for j in cands if j not in selfish and j != own_cls]
        j_star = max(rivals, key=lambda j: (L[j], j)) if rivals else None
        lam = _NAN
        if j_star is not None:
            lam = min(LAMBDA_HI, max(LAMBDA_LO, L[j_star] - L[e]))
            if step:
                st["S"] = max(0.0, float(st["S"]) + lam - CUSUM_K)
        top = max((j for j in cands if j != own_cls), key=lambda j: (L[j], j))
        st["winners"] = (list(st["winners"]) + [e if top in selfish else top])[-K:]

        # ---- unknown CUSUM: typicality under each individual candidate
        ln2pi = z.size * math.log(2.0 * math.pi)            # d2 = -2 l_e(z) - r ln 2 pi
        ps = [chi2_p((-2.0 * l_all[j] - ln2pi) * t99_scale(model, j, z.size), z.size)
              for j in cands if j in l_all]
        p_max = max(ps) if ps else _NAN
        # the chi2 typicality is calibrated on full K-row windows (B15 fits on
        # them); a partial window (a new or long-silent entity's first active
        # ticks: IQR 0, a noisier median) only lets the chart decay
        if p_max == p_max and step:
            up = p_max < P_TYPICAL and len(rows) >= K and full_ok
            st["U"] = max(0.0, float(st["U"]) + (U_UP if up else -U_DOWN))
        class_p = typicality(model, z, own_cls if own_cls else SYSTEM_KEY)

        # ---- events
        info = {"pi_self": pi_self, "post": post, "L": L, "llr": llr, "j_star": j_star,
                "lam": lam, "h": h, "p_max": p_max, "class_p": class_p, "own_cls": own_cls,
                "p_unknown": p_unknown, "shared": shared, "wts": wts, "wpd": wpd}
        n_ev = 0
        if j_star is not None and st["S"] >= h:
            n_ev += self._mismatch(ctx, sc, e, st, now, dt, info)
            st["S"] = 0.0
        if st["U"] >= U_H:
            n_ev += self._unknown(ctx, sc, e, st, now, dt, info)
            st["U"] = 0.0
        st["hold"] = {k: t for k, t in st["hold"].items() if now - t < ALERT_HOLD_S}

        # ---- outputs: the score is the self-typicality tail (held-out
        # calibrated by B15 for the model version scoring it; an alias linked
        # by B17 counts as self), NaN when the IP is not enrolled
        p_self = score_p(model, z, [j for j in cands if j in selfish], pi_self)
        if len(rows) < K or not full_ok:
            # the held-out calibration is of FULL K-row windows (B15 enrols
            # nothing else): a partial window (the first rows after a weekend /
            # holiday gap longer than the lookback, a new or long-silent IP;
            # IQR 0, noisier medians) has no calibrated p (pack B seed 0: 13 of
            # the 15 human identity p < 1e-3 on clean ticks were the 09:00 /
            # 10:00 windows after the weekend and the holiday)
            p_self = _NAN
        info["p_self"] = p_self
        emit.write_scores(store, s, e, now,
                          {DETECTOR: -math.log10(max(p_self, 1e-300)) if p_self == p_self
                           else _NAN},
                          pm={DETECTOR: p_self},
                          axes={DETECTOR: AXES} if pi_self < AXES_POST_MAX else None,
                          window_s=win)
        best: Dict[str, Any] = {}
        if j_star is not None:
            best = {"entity": j_star, "posterior": _json(post[j_star]), "L": _json(L[j_star]),
                    "kind": "class" if is_class(j_star) else "entity"}
        store.add_derived(DerivedMetric(
            name=ID_SERIES, ts=now, system=s, entity=e, window_s=win,
            kind=MetricKind.CATEGORICAL, inputs=["feature.vec", "feature.sketch", MI.MODEL],
            value={"posterior_self": _json(pi_self), "p_self": _json(p_self), "best_other": best,
                   "p_unknown": _json(p_unknown), "cusum_other": _json(float(st["S"])),
                   "cusum_new": _json(float(st["U"])), "h": _json(h), "lambda": _json(lam),
                   "p_max": _json(p_max), "class_p": _json(class_p),
                   "winners": list(st["winners"]), "n_act": int(st["n_act"]),
                   "n_window": len(rows)}))
        if n_ev or now - float(st["profile_ts"]) >= PROFILE_EVERY_S or now < st["profile_ts"]:
            st["profile_ts"] = now
            self._profile(store, s, e, now, info, st, sc)
        return 1

    # ------------------------------------------------------------- modalities
    def _cheap(self, sc: _Sys, st: Dict[str, Any], wts: Sequence[float], cands: List[str],
               llr: Dict[str, Dict[str, float]]) -> None:
        """vocab, rhythm presence and client stacks over the window, every
        tick: per-tick LLRs are cached per candidate when the tick is first
        seen (a candidate joining later is scored on the cached tick
        evidence) and summed over the window."""
        cache = st["cheap"]
        n_tok = 0.0
        for t in wts:
            md = st["md"].get(t)
            if md is not None:
                n_tok += sum(float(c) for vals in md.vocab.values() for c in vals.values())
        f_voc = min(n_tok, VOCAB_N) / VOCAB_N    # few tokens carry little vocab evidence
        for j in cands:
            tot = [0.0, 0.0, 0.0]
            seen = [False, False, False]
            for t in wts:
                md = st["md"].get(t)
                if md is None:
                    continue
                row = cache.get(t)
                if row is None:
                    row = cache[t] = {"": tick_background(sc, md)}
                x = row.get(j)
                if x is None:
                    x = row[j] = tick_cheap_llrs(sc, j, md, row[""])
                for i in range(3):
                    if x[i] == x[i]:
                        tot[i] += x[i]
                        seen[i] = True
            llr[j]["vocab"] = tot[0] * f_voc if seen[0] else _NAN
            llr[j]["client"] = tot[1] if seen[1] else _NAN
            llr[j]["rhythm"] = tot[2] if seen[2] else _NAN

    def _slow(self, sc: _Sys, st: Dict[str, Any], wts: Sequence[float], cands: List[str],
              llr: Dict[str, Dict[str, float]]) -> None:
        """PPM and gap terms: refreshed when a window completes (every K
        active ticks) and on the first scored tick; a candidate that joins
        between completions is scored on the cached window evidence."""
        slow = st["slow"]
        if slow is None or int(st["n_act"]) % K == 0:
            toks: List[str] = []
            gaps: List[np.ndarray] = []
            for t in wts:
                m = st["md"].get(t)
                if m is not None:
                    toks.extend(m.tokens)
                    gaps.append(m.gaps)
            g = np.concatenate(gaps) if gaps else np.zeros(0)
            slow = st["slow"] = {"data": MI.ModalData(tokens=toks[-SEQ_MAX:], gaps=g[-GAP_MAX:]),
                                 "llr": {}}
        for j in cands:
            hit = slow["llr"].get(j)
            if hit is None:
                out = MI.modality_logliks(sc.store, sc.s, j, slow["data"], sc.now, sc.bg)
                hit = slow["llr"][j] = (out["seq"], out["timing"])
            llr[j]["seq"], llr[j]["timing"] = hit

    # ---------------------------------------------------------------- events
    def _mismatch(self, ctx: Context, sc: _Sys, e: str, st: Dict[str, Any], now: float,
                  dt: float, info: Mapping[str, Any]) -> int:
        store, s, model = ctx.store, sc.s, sc.model
        j = info["j_star"]
        post = float(info["post"][j])
        wins = sum(1 for w in st["winners"] if w == j)
        klass = is_class(j)
        conf = 0.0 if klass else _f(MI.confusion(model, e, j))
        if not klass and j in MI.anonymity_set(model, e):
            conf = 1.0
        concurrent = (not klass) and sc.active(j)
        reasons = []
        if wins < WINS_MIN:
            reasons.append(f"wins {wins}/{K}")
        if post < POST_MIN:
            reasons.append(f"posterior {post:.2f}")
        if not conf < CONF_MAX:
            reasons.append(f"confusion {conf:.2f}")
        if info["shared"]:
            reasons.append("shared_ip")
        sev = Severity.INFO if reasons else (Severity.HIGH if concurrent else Severity.MEDIUM)
        hold_key = f"mismatch|{j}|{sev.value}"
        if ctx.training or now - st["hold"].get(hold_key, -math.inf) < ALERT_HOLD_S:
            return 0
        st["hold"][hold_key] = now
        kind = "class" if klass else "entity"
        what = m_class.role_name(store, j[len("class:"):]) if klass else j
        desc = (f"window of {e} is attributed to {kind} {what} (posterior {post:.2f}, "
                f"{wins}/{K} windows)" + (", which is active at the same time" if concurrent
                                          else ""))
        store.add_event(BehaviorEvent(
            system=s, entity=e, ts=now, kind="identity_mismatch", score=post, severity=sev,
            description=desc, axes=list(AXES),
            extra={"looks_like": j, "looks_like_kind": kind, "posterior": _json(post),
                   "posterior_self": _json(info["pi_self"]),
                   "p_unknown": _json(info["p_unknown"]), "cusum": _json(float(st["S"])),
                   "h": _json(info["h"]), "wins": wins, "confusion": _json(conf),
                   "concurrent": bool(concurrent), "downgraded": reasons,
                   "shared_ip": bool(info["shared"]),
                   "llr": {m: _json(v) for m, v in info["llr"][j].items()},
                   "llr_self": {m: _json(v) for m, v in info["llr"][e].items()},
                   "hold_key": hold_key},
            dedupe_key=f"identity_mismatch|{s}|{e}|{j}",
            model_version=MI.version(model), window=(float(info["wts"][0]) - dt, now)))
        return 1

    def _unknown(self, ctx: Context, sc: _Sys, e: str, st: Dict[str, Any], now: float,
                 dt: float, info: Mapping[str, Any]) -> int:
        store, s, model = ctx.store, sc.s, sc.model
        young = e not in sc.enrolled
        cp = info["class_p"]
        if young:
            sev, why = Severity.INFO, "young"
        elif cp == cp and cp < P_TYPICAL:
            sev, why = Severity.HIGH, "unlike_anyone"
        else:
            sev, why = Severity.MEDIUM, "same_class_different_individual"
        if info["shared"]:
            sev = Severity.INFO
        hold_key = f"unknown|{sev.value}"
        if ctx.training or now - st["hold"].get(hold_key, -math.inf) < ALERT_HOLD_S:
            return 0
        st["hold"][hold_key] = now
        p_max = info["p_max"]
        desc = {"young": f"{e} is not enrolled yet and resembles no known identity",
                "unlike_anyone": f"window of {e} is unlike any known individual or class",
                "same_class_different_individual":
                    f"window of {e} fits its class but no known individual"}[why]
        store.add_event(BehaviorEvent(
            system=s, entity=e, ts=now, kind="unknown_identity",
            score=1.0 - (p_max if p_max == p_max else 0.0), severity=sev, description=desc,
            axes=list(AXES),
            extra={"reason": why, "p_max": _json(p_max), "class_p": _json(cp),
                   "class_key": info["own_cls"], "p_unknown": _json(info["p_unknown"]),
                   "posterior_self": _json(info["pi_self"]), "cusum": _json(float(st["U"])),
                   "young": bool(young), "shared_ip": bool(info["shared"]),
                   "hold_key": hold_key},
            dedupe_key=f"unknown_identity|{s}|{e}",
            model_version=MI.version(model), window=(float(info["wts"][0]) - dt, now)))
        return 1

    # ---------------------------------------------------------------- profile
    def _profile(self, store: Any, s: str, e: str, now: float, info: Mapping[str, Any],
                 st: Mapping[str, Any], sc: _Sys) -> None:
        post, L, llr = info["post"], info["L"], info["llr"]
        top = sorted(post, key=lambda j: (-post[j], j))[:4]
        j_star = info["j_star"]
        p = store.profile(s, e) or EntityProfile(system=s, entity=e)
        p.extra["attribution"] = {
            "posterior_self": _json(info["pi_self"]), "p_unknown": _json(info["p_unknown"]),
            "looks_like": j_star,
            "looks_like_posterior": _json(post[j_star]) if j_star else None,
            "candidates": [{"id": j, "kind": "class" if is_class(j) else "entity",
                            "posterior": _json(post[j]), "L": _json(L[j]),
                            "llr": {m: _json(v) for m, v in llr[j].items()}} for j in top],
            "cusum_other": _json(float(st["S"])), "cusum_new": _json(float(st["U"])),
            "h": _json(info["h"]), "windows_per_day": _json(info["wpd"]),
            "p_max": _json(info["p_max"]), "class_p": _json(info["class_p"]),
            "enrolled": e in sc.enrolled, "shared_ip": bool(info["shared"]),
            "model_version": MI.version(sc.model), "updated": now,
        }
        store.put_profile(p)
