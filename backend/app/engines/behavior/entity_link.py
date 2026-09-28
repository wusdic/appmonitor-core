"""EntityLinkEngine (B17): keep identity continuous across re-addressing.

Why: every detector keys its memory by IP. When DHCP or a VPN pool hands a
person a new address, the new IP starts cold (every novelty fires, no
baseline, no identity) and the old one goes "silent"; an IP-hopping actor
exploits exactly this to stay below every per-IP budget; a NAT puts several
people behind one address and makes its "persona" a blur. B17 decides, with
calibrated evidence, which addresses are the same actor, so that learners can
seed the new IP from the old one (contract H: B := B_own + 0.5 A), budgets
and incidents can follow the actor, and a takeover dressed up as a swap (the
old persona on a new device) is flagged instead of linked.

Identity here is ABSOLUTE (architecture principle 7): a new entity's windows
are scored under each candidate's OWN models on absolute data (feature.vec,
feature.sketch, behavior.timing, act.tokens / act.stream, client.stack_set)
through lib/m_identity, with B15's calibration and B16's per-modality cap.
The linking path never reads the self-normalised z / zr / zi series (they
make every IP look like "itself"); behavior.zi is read ONLY by the shared-IP
mixture test, which asks whether one IP's own residuals are bimodal (tested).

Per system and tick (docs/lib3/engines.md '## B17'):
  1. Retraction: an active link A -> B whose A is active again while B was
     active within RETRACT_OVERLAP_S is retracted (link_retracted). The record
     stays with rollback_to = t_link, so B28 writes model.control.rollback_to
     and every learner drops the seed (lib/gating keeps seeds behind a
     barrier at t_link); links older than ROLLBACK_MAX_DEPTH_S are no longer
     retracted (the address has simply been re-used).
  2. Trigger: a real entity B with store.first_seen within 24 h and fewer
     than 24 active ticks is evaluated on each of its first EVAL_TICKS = 8
     active ticks. Candidates A: same system (and the same DHCP scope, or the
     same B02 pool, when B has one), first_seen(A) < first_seen(B),
     last_seen(A) <= first_seen(B) + dt (no activity of A after B appeared),
     gap <= MAX_GAP_S, no active outgoing link; at most N_CAND by a cheap
     prior (topology + time + capped Gaussian LLR).
  3. Fellegi-Sunter log-odds LO(B, A), windows = B's active ticks cut into
     K = 4 (the last one partial):
       behaviour = sum_w [L_A(w) - logsumexp_{j != A, incl. new} L_j(w)],
         L_j = sum over {gauss, rhythm, seq, timing} of B15's calibrated LLR
         capped at +-4 nats (m_identity), j over the other candidates, the 5
         nearest enrolled entities and "new" (the background, L = 0);
       vocab = sum_w calibrated, capped MNB LLR of B's values under A's
         vocab chain vs the system tier (x min(n_tok, 20)/20);
       device = m_link.device_llr (share-weighted ln LR of B's stacks under
         "same device as A" vs "a system member");
       topology +2 (same /24 or DHCP scope) else -2; time -gap/lease.
     A candidate not enrolled in model.identity gets its Gaussian centre from
     its own recent absolute windows (feature.vec is kept 1 d).
  4. Resolve with a Hungarian one-to-one assignment over the pending new
     entities and their candidates. Link when LO >= 5, the margin over the
     runner-up (B's row and A's column) >= 2, B has >= MIN_TICKS active
     ticks, the behavioural evidence (behaviour + vocab) is positive, so
     device and topology priors alone can never link, and A is still silent
     (last_seen(A) <= first_seen(B) + dt, not active now). conf = sigma(LO - 5).
     A candidate's own continuity chain is never its rival (an IP-hopping
     actor's previous address looks like its predecessor, which is the point).
     On link: model.link version + 1 (learners seed on the version increase),
     continuity for both ends, entity_resolution on B, identity_moved on A.
     If behaviour matches (LO without the device term >= 5, margin 2) but the
     device LR <= 0.1: possible_impersonation instead of a link.
  5. Shared IP, every min(32 ticks, 8 h) per active entity with >= 64 rows
     (Engine.entity_due: a per-entity phase spreads the fits over the period):
     Gaussian mixture k = 1 vs 2 (diagonal, on PCA(<= 4) of the centred last
     96 active zi rows; EM here, deterministic and ~1-3 ms). shared_ip when
     BIC(2) < BIC(1) - 10 on 3 consecutive runs, both components (weight >=
     0.15) co-occur in the same local hours (histogram overlap >= 0.3), and
     two disjoint stacks (different ja3n AND UA) overlap in
     client.stack_events (within 5 min, B09's concurrency rule) on >= 2 ticks
     of the lookback. Then entity_kind = 'ip-class' (shared_ip is INFO, and
     B16 downgrades identity events of an ip-class). Three negative runs
     clear the flag.
  6. Actors: active links chained within 24 h (m_link.build_actors).

Training mode links, retracts and flags as usual (it is learning) but emits
no events. Absence is data: an entity without a feature.active row at now is
simply not active; NaN evidence contributes 0 (never a guess).

State: model.link@(s, '__system__') holds links, actors, the shared-IP
streaks and the trigger bookkeeping (restart-safe). The engine keeps only
caches: per-tick absolute rows and raw modality evidence of new entities
(raw sets live 1 h) and the zi ring for the mixture test (zi lives 6 h; a
restart rebuilds it from what the store still has).

Store: reads store.first_seen / last_seen, feature.active, feature.vec,
feature.sketch, feature.tctx, behavior.timing, act.tokens, act.stream,
tls/dns/l4 sets, client.stack_set, client.stack_events, behavior.zi (shared
IP only), model.identity, model.vocab, model.client, model.rhythm, model.seq,
model.timing, model.class, ctx.config.dhcp_scopes. Writes
model.link@(s, '__system__'), profile.extra.continuity; events
entity_resolution, possible_impersonation, shared_ip, identity_moved,
link_retracted.
"""
from __future__ import annotations

import copy
import math
from collections import deque
from collections.abc import Mapping
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ...core.engine import Context, Engine
from ...models.schema import BehaviorEvent, EntityProfile, Severity
from .lib import m_class
from .lib import m_client as MC
from .lib import grains as GR
from .lib import m_identity as MI
from .lib import m_link as ML
from .lib import m_seq as MS
from .lib.classkeys import SYSTEM_KEY
from .lib.gating import ROLLBACK_MAX_DEPTH_S
from .lib.stack import parse_stack_token, stack_id

ACTIVE = "feature.active"
TCTX = "feature.tctx"
ZI = "behavior.zi"                     # shared-IP mixture test only (see docstring)

# trigger
NEW_S = 86400.0                        # first_seen within 24 h
NEW_MAX_ACTIVE = 24                    # fewer than 24 active ticks
EVAL_TICKS = 8                         # evaluated on B's first 8 active ticks
MIN_TICKS = 2                          # active ticks of B before a link (persistence)
MAX_GAP_S = 7 * 86400.0                # A silent longer than this is no candidate
N_CAND = 8                             # candidates A per new entity
N_RIVALS = 5                           # nearest enrolled entities as alternatives
K = MI.K_WIN
CAP = MI.LLR_CAP
VOCAB_N = MI.VOCAB_N                    # vocab LLR x min(n_tok, 20)/20 (lib/m_identity)
BEHAV_MODS = ("gauss", "rhythm", "seq", "timing")
CENTRE_TICKS = 16                      # ad hoc Gaussian centre: A's last 16 active ticks
LOOKBACK_S = 86400.0                   # feature.vec retention
# retraction
RETRACT_OVERLAP_S = 3600.0             # "B is active": seen within the last hour
PENDING_KEEP_S = NEW_S
# shared IP
SH_ROWS = 96
SH_MIN_ROWS = 64
SH_EVERY_TICKS = 32
SH_EVERY_S = 8 * 3600.0
SH_BIC_DELTA = 10.0
SH_RUNS = 3
SH_CLEAR_RUNS = 3
SH_MIN_WEIGHT = 0.15
SH_HOUR_OVERLAP = 0.3
SH_PCA = 4
SH_CLIP = 8.0
SH_OVERLAP_TICKS = 2
SH_OVERLAP_KEEP = 16
CONCURRENT_S = 300.0                   # B09 / R3 stack concurrency rule
_NAN = math.nan
_LN_IMP = math.log(ML.IMPERSONATION_LR)


# ============================================================ small helpers
def _f(x: Any) -> float:
    if isinstance(x, bool) or x is None:
        return _NAN
    try:
        return float(x)
    except (TypeError, ValueError, OverflowError):
        return _NAN


def _j(v: Any, nd: int = 4) -> Any:
    v = _f(v)
    return round(v, nd) if math.isfinite(v) else None


def _nz(x: float) -> float:
    return x if x == x and not math.isinf(x) else 0.0


def _lse(xs: Sequence[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


# spec v2.1 (docs/lib3/cadence.md §8): in canonical grain mode the rows are the
# H rows (clock feature.meta.h, m_identity.grain_row / grain_modal_data) and
# B17 runs on H decision ticks; set per run by the engine.
_MODE: Dict[str, Any] = {"canon": False, "cfg": {}, "dt": 900.0}


def _clock() -> str:
    return "feature.meta.h" if _MODE["canon"] else ACTIVE


def _row(store: Any, s: str, e: str, t: float) -> Optional[np.ndarray]:
    if _MODE["canon"]:
        return MI.grain_row(store, s, e, t, config=_MODE["cfg"], dt=_MODE["dt"])
    return MI.tick_row(store, s, e, t)


def _modal(store: Any, s: str, e: str, t: float, smap: Any) -> Any:
    if _MODE["canon"]:
        return MI.grain_modal_data(store, s, e, t, smap=smap, config=_MODE["cfg"],
                                   dt=_MODE["dt"])
    return MI.tick_modal_data(store, s, e, t, smap=smap)


def _active(store: Any, s: str, e: str, ts: float) -> bool:
    a = store.vec_at(s, e, _clock(), ts)
    return a is not None and float(a[0]) >= 0.5


def _active_ts(store: Any, s: str, e: str, since: float) -> List[float]:
    ts, A = store.vec_since(s, e, _clock(), since)
    if not len(ts):
        return []
    A = np.asarray(A, dtype=np.float64)
    a = A[:, 0] if A.ndim == 2 else A.reshape(-1)
    return [float(t) for t, v in zip(ts, a) if v >= 0.5]


def _hour(store: Any, s: str, e: str, now: float) -> float:
    m = store.latest_derived(s, e, TCTX)
    if m is None or m.ts != now or not isinstance(m.value, Mapping):
        return _NAN
    return _f(m.value.get("hour_local"))


# ============================================================ shared-IP maths
def _gauss_ll_diag(Y: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Per-row log-density under a diagonal Gaussian."""
    return -0.5 * (np.sum(np.log(2.0 * math.pi * var)) + np.sum((Y - mu) ** 2 / var, axis=1))


def gmm2_diag(Y: np.ndarray, init: np.ndarray, iters: int = 100, tol: float = 1e-5,
              reg: float = 1e-4) -> Tuple[float, np.ndarray, np.ndarray]:
    """EM for a 2-component diagonal Gaussian mixture from a hard initial
    split (bool[n]). Returns (log-likelihood, weights[2], labels[n]).
    Deterministic, ~1 ms at 96 x 4 (sklearn's GaussianMixture costs ~25 ms
    per fit at this size, mostly per-call overhead)."""
    n = Y.shape[0]
    R = np.stack([~init, init], axis=1).astype(np.float64)
    ll_prev = -math.inf
    ll = -math.inf
    w = np.full(2, 0.5)
    for _ in range(iters):
        Nk = R.sum(axis=0) + 1e-12
        w = Nk / n
        mu = (R.T @ Y) / Nk[:, None]
        var = np.maximum((R.T @ (Y * Y)) / Nk[:, None] - mu * mu, 0.0) + reg
        lp = (np.log(w) - 0.5 * np.sum(np.log(2.0 * math.pi * var), axis=1)
              - 0.5 * np.sum((Y[:, None, :] - mu[None]) ** 2 / var[None], axis=2))
        m = lp.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(lp - m).sum(axis=1))
        ll = float(lse.sum())
        R = np.exp(lp - lse[:, None])
        if ll - ll_prev <= tol * abs(ll):
            break
        ll_prev = ll
    return ll, w, np.argmax(R, axis=1)


def mixture_test(Z: np.ndarray, hours: np.ndarray) -> Dict[str, float]:
    """One shared-IP run on the zi rows Z[n, 52] (NaN allowed) with local
    hours[n]: BIC(1) - BIC(2) of diagonal Gaussian mixtures on PCA(<= 4) of
    the centred usable columns (NaN -> 0 = expected, clipped at +-8), the
    smaller component weight, and the local hour-histogram overlap of the two
    components. NaN delta when the rows cannot support a fit. Few PCs keep
    the BIC penalty small (measured: two personas 3 sigma apart in 6 features
    give delta >= 60 at 96 rows, unimodal rows <= -10). k = 2 is fitted by EM
    from the median split of PC1 (the direction a persona split dominates),
    so the test is deterministic and replayable."""
    out = {"delta": _NAN, "w_min": _NAN, "hour_overlap": _NAN, "d": 0}
    X = np.asarray(Z, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < SH_MIN_ROWS:
        return out
    fin = np.isfinite(X)
    keep = fin.mean(axis=0) >= 0.8
    X = np.clip(np.where(fin, X, 0.0), -SH_CLIP, SH_CLIP)[:, keep]   # z scale: 0 = expected
    if X.shape[1] == 0:
        return out
    X = X[:, X.std(axis=0) > 1e-6]
    if X.shape[1] == 0:
        return out
    X = X - X.mean(axis=0)            # zi is already in z units: no rescaling, so a
    n = X.shape[0]                    # persona split dominates PC1
    d = int(min(SH_PCA, X.shape[1], n - 1))
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    Y = U[:, :d] * S[:d]
    var1 = Y.var(axis=0) + 1e-4
    ll1 = float(_gauss_ll_diag(Y, Y.mean(axis=0), var1).sum())
    best = None
    for c in range(d):                # PC1's median split; the next PC if it is degenerate
        split = Y[:, c] > np.median(Y[:, c])
        if split.any() and not split.all():
            best = gmm2_diag(Y, split)
            break
    if best is None:
        return out
    ll2, w, lab = best
    bic1 = -2.0 * ll1 + (2 * d) * math.log(n)
    bic2 = -2.0 * ll2 + (4 * d + 1) * math.log(n)
    out.update(d=d, delta=float(bic1 - bic2), w_min=float(np.min(w)))
    h = np.asarray(hours, dtype=np.float64)
    ok = np.isfinite(h)
    if ok.sum() >= 2:
        hb = np.floor(np.mod(h[ok], 24.0)).astype(int)
        hist = []
        for k in (0, 1):
            cnt = np.bincount(hb[lab[ok] == k], minlength=24).astype(np.float64)
            hist.append(cnt / cnt.sum() if cnt.sum() > 0 else cnt)
        out["hour_overlap"] = float(np.minimum(hist[0], hist[1]).sum())
    return out


def disjoint(tok_a: str, tok_b: str) -> bool:
    """Two stacks of different devices: neither the TLS library (ja3n; '-' =
    no fingerprint, which ties nothing) nor the UA (family/major) is shared."""
    a, b = parse_stack_token(tok_a), parse_stack_token(tok_b)
    ja, jb = a.get("ja3n"), b.get("ja3n")
    ua = f"{a.get('ua_family')}/{a.get('ua_major')}"
    ub = f"{b.get('ua_family')}/{b.get('ua_major')}"
    ja_ok = ja != jb or ja == "-"          # an unknown fingerprint ties nothing together
    return ja_ok and ua != ub


def stacks_overlap(events: Any, tokens: Sequence[str]) -> bool:
    """True when two disjoint stacks have episodes within CONCURRENT_S of each
    other in client.stack_events ([[stack_id, first_ts, last_ts, n]]); the
    ids join to tokens through lib/stack.stack_id (R3 contract)."""
    if not isinstance(events, (list, tuple)) or len(events) < 2:
        return False
    tok = {stack_id(t): t for t in tokens if isinstance(t, str) and t != MC.OTHER}
    eps: List[Tuple[str, float, float]] = []
    for r in events:
        if not isinstance(r, (list, tuple)) or len(r) < 3:
            continue
        t = tok.get(int(_nz(_f(r[0]))))
        f0, l0 = _f(r[1]), _f(r[2])
        if t is not None and math.isfinite(f0) and math.isfinite(l0):
            eps.append((t, f0, l0))
    for i in range(len(eps)):
        for k in range(i + 1, len(eps)):
            (ta, fa, la), (tb, fb, lb) = eps[i], eps[k]
            if ta != tb and max(fa, fb) <= min(la, lb) + CONCURRENT_S and disjoint(ta, tb):
                return True
    return False


# ============================================================ per-tick system view
class _Sys:
    """One system at one tick: identity model, modality background, lookups."""

    def __init__(self, store: Any, s: str, now: float, dt: float, cfg: Mapping) -> None:
        self.store, self.s, self.now, self.dt = store, s, now, dt
        self.idm = MI.get(store, s)
        self.fitted = MI.is_fitted(self.idm)
        self.calib = {m: MI.llr_calib(self.idm, m) for m in MI.MODALITIES}
        self.scopes = ML.parse_scopes(cfg.get("dhcp_scopes"))
        self._bg: Optional[MI.Background] = None
        self._smap: Optional[MS.SymbolMap] = None
        self._centre: Dict[str, Optional[np.ndarray]] = {}
        self._fs: Dict[str, Optional[float]] = {}
        self._ls: Dict[str, Optional[float]] = {}
        self._sys_shares: Optional[Dict[str, float]] = None
        self.model: Mapping[str, Any] = {}
        self._aliases: Dict[str, Set[str]] = {}

    def aliases(self, e: str) -> Set[str]:
        """e's continuity chain under the model being built this tick."""
        if e not in self._aliases:
            self._aliases[e] = set(ML.aliases(self.model, e))
        return self._aliases[e]

    @property
    def bg(self) -> MI.Background:
        if self._bg is None:
            self._bg = MI.Background(self.store, self.s, self.now)
        return self._bg

    @property
    def smap(self) -> MS.SymbolMap:
        if self._smap is None:
            self._smap = MS.SymbolMap.from_store(self.store, self.s)
        return self._smap

    def fs(self, e: str) -> Optional[float]:
        if e not in self._fs:
            self._fs[e] = self.store.first_seen(self.s, e)
        return self._fs[e]

    def ls(self, e: str) -> Optional[float]:
        if e not in self._ls:
            self._ls[e] = self.store.last_seen(self.s, e)
        return self._ls[e]

    def cal(self, m: str, llr: float) -> float:
        """B15 calibration + B16 cap; NaN = no evidence."""
        if not (llr == llr) or math.isinf(llr):
            return _NAN
        a, b = self.calib[m]
        return min(CAP, max(-CAP, a * llr + b))

    def centre(self, e: str) -> Optional[np.ndarray]:
        """e's Gaussian centre in LDA space: B15's mean when enrolled, else the
        mean of e's own recent absolute windows (last CENTRE_TICKS active
        ticks of the last day, cut into K)."""
        if e in self._centre:
            return self._centre[e]
        c = MI.entity_mean(self.idm, e) if self.fitted else None
        if c is None and self.fitted:
            ts = _active_ts(self.store, self.s, e, self.now - LOOKBACK_S)[-CENTRE_TICKS:]
            rows = [r for r in (_row(self.store, self.s, e, t) for t in ts)
                    if r is not None]
            if len(rows) >= 2:
                ck = m_class.class_key(self.store, self.s, e)
                W = [MI.window_vector(np.vstack(rows[i:i + K])) for i in range(0, len(rows), K)]
                Z = MI.transform_many(self.idm, np.vstack(W), [ck] * len(W))
                if Z.size and np.all(np.isfinite(Z)):
                    c = Z.mean(axis=0)
        self._centre[e] = c
        return c

    def sys_shares(self) -> Dict[str, float]:
        if self._sys_shares is None:
            m = MC.get(self.store, self.s, SYSTEM_KEY)
            c = MC.counts(m) if m is not None else {}
            tot = sum(c.values())
            self._sys_shares = {t: v / tot for t, v in c.items()} if tot > 0 else {}
        return self._sys_shares


# ============================================================ the engine
class EntityLinkEngine(Engine):
    name = "behavior.entity_link"
    layer = "behavior"
    consumes = ["store.first_seen", "store.last_seen", ACTIVE, "feature.vec", "feature.sketch",
                TCTX, "behavior.timing", "act.tokens", "act.stream", "tls.sni_etld1_set",
                "dns.qname_etld1_set", "l4.dport_set", "client.stack_set",
                "client.stack_events", ZI, MI.MODEL, "model.vocab", "model.client",
                "model.rhythm", "model.seq", "model.timing", "model.class",
                "config.dhcp_scopes"]
    produces = [ML.MODEL, "profile.extra.continuity", "event.entity_resolution",
                "event.possible_impersonation", "event.shared_ip", "event.identity_moved",
                "event.link_retracted"]
    description = ("Entity resolution across re-addressing: Fellegi-Sunter log-odds of a new "
                   "IP against silent predecessors (calibrated absolute behaviour, vocab, "
                   "device, topology and time priors), Hungarian one-to-one links with "
                   "retraction, impersonation guard, shared-IP mixtures and actor chains.")
    interval = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        # (s, B) -> {'rows': {ts: tick row}, 'md': {ts: ModalData}, 'win': {wkey: {cand: tuple}}}
        self._new: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # (s, e) -> deque[(ts, zi row, local hour)]
        self._zi: Dict[Tuple[str, str], Deque[Tuple[float, np.ndarray, float]]] = {}
        self._last: Dict[Tuple[str, str], float] = {}   # zi collection high-water mark

    # ------------------------------------------------------------------ run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        now, dt = float(ctx.now), float(ctx.window_s)
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError(f"EntityLinkEngine: bad ctx.window_s {ctx.window_s!r}")
        canon = GR.canonical(ctx.config)
        _MODE.update(canon=canon, cfg=ctx.config, dt=dt)
        if canon and not GR.decision(now, dt, "h", GR.CANONICAL):
            return 0                              # spec v2.1: H rows only
        if canon:
            dt = max(dt, GR.GRAIN_S["h"])         # the rows' period
        return sum(self._system(ctx, s, now, dt) for s in store.systems())

    def _system(self, ctx: Context, s: str, now: float, dt: float) -> int:
        store = ctx.store
        old = ML.get(store, s)
        # copy-on-write: links / pending are small and copied; shared-IP records
        # (one per entity) are copied only when written (_own)
        model = {**ML.empty(), **(old or {})}
        model["links"] = [dict(lk) for lk in model["links"] or () if isinstance(lk, Mapping)]
        model["pending"] = copy.deepcopy(dict(model["pending"] or {}))
        model["shared"] = dict(model["shared"] or {})
        ents = store.entities(s)
        if not ents:
            return 0
        sc = _Sys(store, s, now, dt, ctx.config)
        sc.model = model
        active = {e: _active(store, s, e, now) for e in ents}
        out: List[BehaviorEvent] = []
        touched: Set[str] = set()
        before = (ML.version(model), len(model["links"]))

        dirty = self._retract(sc, model, active, out, touched)
        dirty |= self._triggers(sc, model, active, out, touched)
        dirty |= self._shared(sc, model, active, out, touched)

        links_changed = (ML.version(model), len(model["links"])) != before
        if links_changed:
            model["actors"] = ML.build_actors(model["links"])
        if dirty or links_changed or old is None:
            model["updated"] = now
            store.put_model(s, SYSTEM_KEY, ML.MODEL, model, version=ML.version(model), ts=now)
        n = self._continuity(store, s, model, touched, now)
        if not ctx.training:
            for ev in out:
                ev.model_version = ML.version(model)
                store.add_event(ev)
        return n + len(out) + int(bool(dirty or links_changed))

    # ------------------------------------------------------------- retraction
    def _retract(self, sc: _Sys, model: Dict[str, Any], active: Mapping[str, bool],
                 out: List[BehaviorEvent], touched: Set[str]) -> bool:
        now, changed = sc.now, False
        for lk in model["links"]:
            if ML.is_retracted(lk):
                continue
            a, b, t_link = str(lk["from"]), str(lk["to"]), _f(lk.get("ts"))
            if not active.get(a) or not (now - t_link <= ROLLBACK_MAX_DEPTH_S):
                continue
            lb = sc.ls(b)
            if lb is None or lb < now - RETRACT_OVERLAP_S:
                continue                               # B not active: A simply came back
            lk.update(status="retracted", retracted=True, retracted_ts=now, rollback_to=t_link,
                      reason="from_reactivated")
            model["version"] = ML.version(model) + 1
            touched.update((a, b))
            changed = True
            out.append(BehaviorEvent(                  # on B: its seed is what gets undone
                system=sc.s, entity=b, ts=now, kind="link_retracted", score=1.0,
                severity=Severity.LOW,
                description=(f"link {a} -> {b} retracted: {a} is active again while {b} "
                             f"is active; learners roll back to the link time"),
                axes=["identity"],
                extra={"from": a, "to": b, "link_ts": t_link, "rollback_to": t_link,
                       "link_id": lk.get("id"), "conf": lk.get("conf")},
                dedupe_key=f"link_retracted|{sc.s}|{lk.get('id')}",
                window=(t_link, now)))
        return changed

    # ---------------------------------------------------------------- triggers
    def _triggers(self, sc: _Sys, model: Dict[str, Any], active: Mapping[str, bool],
                  out: List[BehaviorEvent], touched: Set[str]) -> bool:
        store, s, now, dt = sc.store, sc.s, sc.now, sc.dt
        pend: Dict[str, Any] = model["pending"]
        dirty = False
        act_links = [lk for lk in model["links"] if not ML.is_retracted(lk)]
        into = {str(lk["to"]) for lk in act_links}
        out_of = {str(lk["from"]) for lk in act_links}
        evaluated: List[str] = []
        for b in sorted(e for e, on in active.items() if on):
            fs = sc.fs(b)
            if fs is None or now - fs > NEW_S or b in into:
                continue
            p = pend.get(b)
            if p is not None and p.get("done"):
                continue
            wts = _active_ts(store, s, b, fs - 0.5 * dt)
            if now not in wts:
                wts.append(now)
            n_act = len(wts)
            if n_act > EVAL_TICKS or n_act >= NEW_MAX_ACTIVE:
                if p is not None:
                    p["done"] = True
                    dirty = True
                continue
            cands = self._candidates(sc, b, fs, active, out_of)
            if not cands and p is None:
                continue                              # nobody silent to be: nothing to keep
            p = pend.setdefault(b, {"fs": fs, "rows": {}, "imp": [], "done": False})
            p.update(fs=fs, n=n_act, ts=now, done=n_act >= EVAL_TICKS)
            p["rows"] = self._evaluate(sc, b, wts, cands) if cands else {}
            evaluated.append(b)
            dirty = True
        if evaluated:
            self._impersonation(sc, model, evaluated, out, touched)
            self._resolve(sc, model, evaluated, active, out, touched)
        # bookkeeping: expire old / linked entries and the caches of finished entities
        into = {str(lk["to"]) for lk in model["links"] if not ML.is_retracted(lk)}
        for b in list(pend):
            if not (now - _f(pend[b].get("fs")) <= PENDING_KEEP_S) or b in into:
                del pend[b]
                dirty = True
        for key in [k for k in self._new if k[0] == s and (k[1] not in pend
                                                           or pend[k[1]].get("done"))]:
            del self._new[key]
        return dirty

    def _candidates(self, sc: _Sys, b: str, fs: float, active: Mapping[str, bool],
                    out_of: Set[str]) -> List[str]:
        """Silent predecessors of b (see module doc), best N_CAND by a cheap prior."""
        store, s = sc.store, sc.s
        sb = ML.scope_of(b, sc.scopes)
        pool_b = (m_class.assignment(store, s, b) or {}).get("pool")
        z_b = None
        pre: List[Tuple[float, str]] = []
        for a in active:
            if a == b or a in out_of:
                continue
            fa, la = sc.fs(a), sc.ls(a)
            if fa is None or la is None or fa >= fs or la > fs + sc.dt:
                continue
            gap = max(0.0, fs - la)
            if gap > MAX_GAP_S:
                continue
            if sb is not None and ML.scope_of(a, sc.scopes) != sb:
                continue
            if sb is None and pool_b and (m_class.assignment(store, s, a) or {}).get("pool") \
                    != pool_b:
                continue
            prior = ML.topology_prior(a, b, sc.scopes) + ML.time_prior(
                gap, ML.lease_for(a, b, sc.scopes))
            m = MI.entity_mean(sc.idm, a) if sc.fitted else None
            if m is not None:
                if z_b is None:
                    z_b = self._z_now(sc, b)
                if z_b is not None and z_b.size == m.size:
                    prior += _nz(sc.cal("gauss", MI.gauss_llr(sc.idm, z_b, a)))
            pre.append((prior, a))
        pre.sort(key=lambda t: (-t[0], t[1]))
        return [a for _, a in pre[:N_CAND]]

    def _z_now(self, sc: _Sys, b: str) -> Optional[np.ndarray]:
        row = _row(sc.store, sc.s, b, sc.now)
        if row is None:
            return None
        z = MI.transform(sc.idm, MI.window_vector(row.reshape(1, -1)))
        return z if z.size and np.all(np.isfinite(z)) else None

    # --------------------------------------------------------------- evidence
    def _evaluate(self, sc: _Sys, b: str, wts: List[float], cands: List[str]
                  ) -> Dict[str, Dict[str, Any]]:
        """LO(B, A) and its terms for every candidate A (module doc, step 3)."""
        store, s, now = sc.store, sc.s, sc.now
        cache = self._new.setdefault((s, b), {"rows": {}, "md": {}, "win": {}})
        keep = set(wts)
        for k in ("rows", "md"):
            cache[k] = {t: v for t, v in cache[k].items() if t in keep}
        for t in wts:
            if t not in cache["rows"]:
                r = _row(store, s, b, t)
                if r is not None:
                    cache["rows"][t] = r
            if t not in cache["md"] and now - t <= 3600.0:   # raw sets are kept 1 h
                cache["md"][t] = _modal(store, s, b, t, sc.smap)
        windows = [tuple(wts[i:i + K]) for i in range(0, len(wts), K)]
        cache["win"] = {w: v for w, v in cache["win"].items() if w in set(windows)}
        beh = {a: [] for a in cands}
        voc = {a: [] for a in cands}
        stacks: Dict[str, float] = {}
        for t in wts:
            md = cache["md"].get(t)
            if md is not None:
                for tok, n in md.stacks.items():
                    stacks[tok] = stacks.get(tok, 0.0) + n
        for w in windows:
            wc = cache["win"].setdefault(w, {})
            miss = [a for a in cands if a not in wc]
            if miss:
                self._score_window(sc, b, w, cache, cands, wc)
            for a in cands:
                bw, vw = wc[a]
                if bw == bw:
                    beh[a].append(bw)
                if vw == vw:
                    voc[a].append(vw)
        shares_sys = sc.sys_shares()
        rows: Dict[str, Dict[str, Any]] = {}
        for a in cands:
            la = sc.ls(a)
            gap = max(0.0, _f(sc.fs(b)) - _f(la))
            topo = ML.topology_prior(a, b, sc.scopes)
            tprior = ML.time_prior(gap, ML.lease_for(a, b, sc.scopes))
            b_tot = float(sum(beh[a])) if beh[a] else _NAN
            v_tot = float(sum(voc[a])) if voc[a] else _NAN
            ma = MC.get(store, s, a)
            dev = ML.device_llr(MC.shares(ma, now), shares_sys, stacks) \
                if MC.kind(ma) == "entity" and shares_sys else _NAN
            lo_nd = _nz(b_tot) + _nz(v_tot) + topo + _nz(tprior)
            rows[a] = {"lo": lo_nd + _nz(dev), "lo_nd": lo_nd, "behaviour": b_tot,
                       "vocab": v_tot, "device": dev, "topology": topo, "time": tprior,
                       "gap_s": gap,
                       "has": bool(b_tot == b_tot or v_tot == v_tot),
                       "pos": bool(_nz(b_tot) + _nz(v_tot) > 0.0), "n": len(wts)}
        return rows

    def _score_window(self, sc: _Sys, b: str, w: Tuple[float, ...], cache: Dict[str, Any],
                      cands: List[str], wc: Dict[str, Tuple[float, float]]) -> None:
        """Per-window (behaviour, vocab) of every candidate: L_j over the
        candidates, the nearest enrolled rivals and 'new' (L = 0)."""
        store, s, now = sc.store, sc.s, sc.now
        rows = [cache["rows"][t] for t in w if t in cache["rows"]]
        data = MI.ModalData()
        for t in w:
            md = cache["md"].get(t)
            if md is not None:
                data.extend(md)
        z = None
        if sc.fitted and rows:
            z = MI.transform(sc.idm, MI.window_vector(np.vstack(rows)))
            if not (z.size and np.all(np.isfinite(z))):
                z = None
        rivals: List[str] = []
        if z is not None:
            rivals = [j for j, _ in MI.top_k(sc.idm, z, N_RIVALS, exclude=set(cands) | {b})]
        l_bg = MI.bg_loglik(sc.idm, z) if z is not None else _NAN
        L: Dict[str, float] = {}
        has: Dict[str, bool] = {}
        vocab: Dict[str, float] = {}
        for j in list(cands) + rivals:
            llr = MI.modality_logliks(store, s, j, data, now, sc.bg)
            g = _NAN
            if z is not None:
                c = sc.centre(j)
                if c is not None and c.size == z.size and l_bg == l_bg:
                    g = -0.5 * (float(np.sum((z - c) ** 2)) + z.size * math.log(2 * math.pi)) - l_bg
            llr["gauss"] = g
            parts = [sc.cal(m, llr.get(m, _NAN)) for m in BEHAV_MODS]
            has[j] = any(x == x for x in parts)
            L[j] = sum(_nz(x) for x in parts)
            # modality_logliks' vocab LLR already carries min(n_tok, 20)/20
            vocab[j] = sc.cal("vocab", llr.get("vocab", _NAN))
        for a in cands:
            if not has[a]:
                wc[a] = (_NAN, vocab[a])
                continue
            same = sc.aliases(a)                    # a's own chain is not a rival
            alt = [L[j] for j in L if j != a and j not in same] + [0.0]
            wc[a] = (L[a] - _lse(alt), vocab[a])

    # ------------------------------------------------------------- decisions
    def _impersonation(self, sc: _Sys, model: Dict[str, Any], evaluated: List[str],
                       out: List[BehaviorEvent], touched: Set[str]) -> None:
        """Behaviour matches A (LO without the device term >= 5, margin 2) but
        the device LR <= 0.1: possible_impersonation, never a link."""
        for b in evaluated:
            p = model["pending"][b]
            rows = p["rows"]
            if not rows or int(p.get("n", 0)) < MIN_TICKS:
                continue
            order = sorted(rows, key=lambda a: (-rows[a]["lo_nd"], a))
            a = order[0]
            r = rows[a]
            ru = rows[order[1]]["lo_nd"] if len(order) > 1 else -math.inf
            dev = _f(r["device"])
            if not (r["lo_nd"] >= ML.LINK_LO and r["lo_nd"] - ru >= ML.LINK_MARGIN
                    and r["pos"] and dev == dev and dev <= _LN_IMP):
                continue
            if a in p["imp"]:
                continue
            p["imp"].append(a)
            touched.add(b)
            out.append(BehaviorEvent(
                system=sc.s, entity=b, ts=sc.now, kind="possible_impersonation",
                score=float(ML.conf(r["lo_nd"])), severity=Severity.MEDIUM,
                description=(f"new {b} behaves like silent {a} (log-odds {r['lo_nd']:.1f}) but "
                             f"from a different device (LR {math.exp(dev):.2f}); not linked"),
                axes=["identity"],
                extra={"looks_like": a, "lo_behaviour": _j(r["lo_nd"]),
                       "device_lr": _j(math.exp(dev)), "terms": _terms(r),
                       "n_ticks": int(p.get("n", 0))},
                dedupe_key=f"possible_impersonation|{sc.s}|{b}|{a}",
                window=(float(p["fs"]), sc.now)))

    def _resolve(self, sc: _Sys, model: Dict[str, Any], evaluated: List[str],
                 active: Mapping[str, bool], out: List[BehaviorEvent], touched: Set[str]
                 ) -> None:
        """Hungarian one-to-one over the pending new entities; links need LO >=
        5 and margin >= 2 over the runner-up of B's row and of A's column."""
        pend = model["pending"]
        bs = sorted(b for b, p in pend.items() if p.get("rows") and "linked" not in p)
        if not bs:
            return
        cols = sorted({a for b in bs for a in pend[b]["rows"]})
        NEG = -1e9
        M = np.full((len(bs), len(cols)), NEG)
        for i, b in enumerate(bs):
            p = pend[b]
            for a, r in p["rows"].items():
                if a in p["imp"] or not r["has"]:
                    continue
                M[i, cols.index(a)] = r["lo"]
        ri, ci = linear_sum_assignment(M, maximize=True)
        for i, j in zip(ri, ci):
            b, a = bs[i], cols[j]
            lo = float(M[i, j])
            if b not in evaluated or lo < ML.LINK_LO:
                continue
            row = np.delete(M[i], j)
            col = np.delete(M[:, j], i)
            alt = [x for x in np.concatenate([row, col]) if x > NEG / 2]
            margin = lo - max(alt) if alt else math.inf
            p, r = pend[b], pend[b]["rows"][a]
            dev = _f(r["device"])
            if margin < ML.LINK_MARGIN or int(p.get("n", 0)) < MIN_TICKS or not r["pos"] \
                    or (dev == dev and dev <= _LN_IMP):
                continue
            la = sc.ls(a)
            if la is None or la > _f(p["fs"]) + sc.dt or active.get(a):
                continue                              # A's continued silence confirms
            self._link(sc, model, a, b, lo, margin, r, out, touched)
            p["linked"] = a

    def _link(self, sc: _Sys, model: Dict[str, Any], a: str, b: str, lo: float, margin: float,
              r: Mapping[str, Any], out: List[BehaviorEvent], touched: Set[str]) -> None:
        now = sc.now
        cf = ML.conf(lo)
        dev = _f(r["device"])
        lk = {"id": f"{a}>{b}@{int(now)}", "from": a, "to": b, "ts": now, "lo": _j(lo),
              "conf": _j(cf), "margin": _j(margin) if math.isfinite(margin) else None,
              "terms": _terms(r), "device_lr": _j(math.exp(dev)) if dev == dev else None,
              "gap_s": _j(r["gap_s"], 1), "n_ticks": int(r["n"]), "status": "active",
              "retracted": False, "retracted_ts": None, "rollback_to": None, "reason": None}
        model["links"].append(lk)
        model["version"] = ML.version(model) + 1
        touched.update((a, b))
        cid = ML.continuity_id(model, b)
        out.append(BehaviorEvent(
            system=sc.s, entity=b, ts=now, kind="entity_resolution", score=float(cf),
            severity=Severity.INFO,
            description=(f"{b} is the continuation of {a} (log-odds {lo:.1f}, confidence "
                         f"{cf:.2f}, silent gap {r['gap_s'] / 60.0:.0f} min)"),
            axes=["identity"],
            extra={"linked_from": a, "conf": _j(cf), "lo": _j(lo),
                   "margin": lk["margin"], "terms": lk["terms"], "continuity_id": cid,
                   "link_id": lk["id"], "n_ticks": int(r["n"])},
            dedupe_key=f"entity_resolution|{sc.s}|{lk['id']}",
            window=(_f(sc.fs(b)), now)))
        out.append(BehaviorEvent(
            system=sc.s, entity=a, ts=now, kind="identity_moved", score=float(cf),
            severity=Severity.INFO, description=f"identity of {a} moved to {b}",
            axes=["identity"],
            extra={"linked_to": b, "conf": _j(cf), "lo": _j(lo), "continuity_id": cid,
                   "link_id": lk["id"]},
            dedupe_key=f"identity_moved|{sc.s}|{lk['id']}",
            window=(_f(sc.ls(a)), now)))

    # --------------------------------------------------------------- shared IP
    def _zi_ring(self, sc: _Sys, e: str) -> Deque[Tuple[float, np.ndarray, float]]:
        key = (sc.s, e)
        ring = self._zi.get(key)
        if ring is None:                          # (re)start: what the store still has
            ring = self._zi[key] = deque(maxlen=SH_ROWS)
            ts, Z = sc.store.vec_since(sc.s, e, ZI, sc.now - 8 * 3600.0)
            hours = {m.ts: _f(m.value.get("hour_local"))
                     for m in sc.store.derived_tail(sc.s, e, TCTX, 4 * SH_ROWS)
                     if isinstance(m.value, Mapping)}
            for t, z in zip(ts, Z):
                t = float(t)
                if t < sc.now and _active(sc.store, sc.s, e, t):
                    ring.append((t, np.asarray(z, dtype=np.float32), hours.get(t, _NAN)))
            self._last[key] = ring[-1][0] if ring else -math.inf
        return ring

    def _shared(self, sc: _Sys, model: Dict[str, Any], active: Mapping[str, bool],
                out: List[BehaviorEvent], touched: Set[str]) -> bool:
        store, s, now, dt = sc.store, sc.s, sc.now, sc.dt
        shared: Dict[str, Any] = model["shared"]
        dirty = False
        every = min(SH_EVERY_TICKS * dt, SH_EVERY_S)
        for e, on in active.items():
            if not on:
                continue
            ring = self._zi_ring(sc, e)
            key = (s, e)
            if now > self._last.get(key, -math.inf):
                z = store.vec_at(s, e, ZI, now)
                if z is not None:
                    ring.append((now, np.asarray(z, dtype=np.float32), _hour(store, s, e, now)))
                self._last[key] = now
            ev = store.latest_raw_at(s, e, "client.stack_events", now)
            ss = store.latest_raw_at(s, e, "client.stack_set", now)
            toks = list(ss.value.keys()) if ss is not None and isinstance(ss.value, Mapping) \
                else []
            if ev is not None and stacks_overlap(ev.value, toks):
                ov = list((shared.get(e) or {}).get("overlap") or [])
                if not ov or ov[-1] != now:
                    _own(shared, e)["overlap"] = (ov + [now])[-SH_OVERLAP_KEEP:]
                    dirty = True
            if len(ring) < SH_MIN_ROWS or not self.entity_due(("shared", s, e), now, every):
                continue                              # per-entity phase spreads the fits
            rec = _own(shared, e)
            dirty = True
            ts = np.array([t for t, _, _ in ring])
            res = mixture_test(np.vstack([z for _, z, _ in ring]),
                               np.array([h for _, _, h in ring]))
            positive = bool(res["delta"] > SH_BIC_DELTA and res["w_min"] >= SH_MIN_WEIGHT
                            and res["hour_overlap"] >= SH_HOUR_OVERLAP)
            rec["last_run"] = now
            rec["streak"] = int(rec.get("streak", 0)) + 1 if positive else 0
            rec["bic_delta"] = (list(rec.get("bic_delta") or []) + [_j(res["delta"], 2)])[-SH_RUNS:]
            rec["hour_overlap"] = _j(res["hour_overlap"])
            rec["w_min"] = _j(res["w_min"])
            n_ov = sum(1 for t in rec["overlap"] if t >= float(ts[0]) - dt)
            if rec.get("flag"):
                rec["neg"] = 0 if positive else int(rec.get("neg", 0)) + 1
                if rec["neg"] >= SH_CLEAR_RUNS:
                    rec.update(flag=False, since=None, neg=0)
                    touched.add(e)
            elif rec["streak"] >= SH_RUNS and n_ov >= SH_OVERLAP_TICKS:
                rec.update(flag=True, since=now, neg=0)
                touched.add(e)
                out.append(BehaviorEvent(
                    system=s, entity=e, ts=now, kind="shared_ip", score=1.0,
                    severity=Severity.INFO,
                    description=(f"{e} carries two co-occurring personas on disjoint client "
                                 f"stacks (NAT / shared host); treated as an IP class"),
                    axes=["identity"],
                    extra={"bic_delta": rec["bic_delta"], "hour_overlap": rec["hour_overlap"],
                           "w_min": rec["w_min"], "overlap_ticks": n_ov, "runs": rec["streak"],
                           "entity_kind": "ip-class"},
                    dedupe_key=f"shared_ip|{s}|{e}", window=(float(ts[0]), now)))
        for key in [k for k in self._zi if k[0] == s and k[1] not in active]:
            del self._zi[key]
        return dirty

    # --------------------------------------------------------------- profiles
    def _continuity(self, store: Any, s: str, model: Mapping[str, Any], touched: Set[str],
                    now: float) -> int:
        """profile.extra.continuity for every entity whose links or shared flag
        changed, plus their chains (aliases move with every link)."""
        ents: Set[str] = set()
        for e in touched:
            ents.add(e)
            ents.update(ML.aliases(model, e))
            for lk in model["links"]:
                if e in (lk["from"], lk["to"]):
                    ents.update((str(lk["from"]), str(lk["to"])))
        n = 0
        for e in sorted(ents):
            c = ML.continuity(model, e)
            p = store.profile(s, e) or EntityProfile(system=s, entity=e)
            if p.extra.get("continuity") == c:
                continue
            p.extra["continuity"] = c
            p.updated = now
            store.put_profile(p)
            n += 1
        return n


def _own(shared: Dict[str, Any], e: str) -> Dict[str, Any]:
    """Copy-on-write shared-IP record (the published model is never mutated)."""
    rec = shared[e] = {**_new_shared(), **(shared.get(e) or {})}
    return rec


def _new_shared() -> Dict[str, Any]:
    return {"last_run": None, "streak": 0, "bic_delta": [], "hour_overlap": None,
            "w_min": None, "overlap": [], "flag": False, "since": None, "neg": 0}


def _terms(r: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: _j(r.get(k)) for k in ("behaviour", "vocab", "device", "topology", "time")}
