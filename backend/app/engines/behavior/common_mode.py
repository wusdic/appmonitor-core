"""CommonModeEngine (B05) — remove benign common-mode movement from residuals.

Why: month-end batches, a WAN brown-out or an upstream 5xx storm move every
member of a class (or the whole system) at once. Per-entity residuals then
light up everywhere, and each entity's detectors report the same shared
cause N times. This engine subtracts the *shared* component from each
entity's normal scores, but only where that is safe:

  * Eligible groups only: volume, transport, app-error (http_5xx_rate,
    http_latency) and probe. Breadth, categorical, dns, tls, identity and
    composition columns are never touched (zi = z there), so a campaign
    cannot hide new destinations, new clients or a changed mix by moving in
    step with its peers.
  * Leave-one-out: the common factor for entity e is
        L_g^(-e) = median over OTHER active, non-quarantined, non-excluded
                   members m of mean_{f in g} z_{m,f}
    within e's role class (>= 3 others), else the system (>= 3 others), else
    0. An entity that alone shifts keeps its full residual (the median of
    the others does not move), and a median needs a majority to be moved.
  * Loading beta_{e,g} = (sum w L z + 4) / (sum w L^2 + 4): a ridge shrink
    toward 1 (with no data beta = 1, i.e. plain subtraction), fitted on
    COMMITTED rows only (lib/gating: delayed by D, trust-weighted, held while
    quarantined, reversible via model.control) with a 7-d half-life, and
    refreshed every 16 ticks per entity (or at once after a control action).
    The gate is stepped on those refit ticks (commits batch up to 16 rows;
    see _learn for why that is never less conservative).
  * zi_f = z_f - beta_g L_g for f in an eligible group g; zi_f = z_f else.
  * Exclusions (flag 0 and zi = z): system-tier novelty (B08 first_seen /
    rare_access event with extra.tier in {system, org}), a lib-4 match >=
    MEDIUM, client or identity evidence with p < 0.01, or budget_exfil
    evidence (acc_alarm or p < 0.01), at t-1 (those engines run after B05)
    or this tick. Excluded entities are also left out of everyone else's
    pool, so compromised members cannot define "normal" for the others.
  * behavior.common.flag[g] = 1 when |z_g| > 2 and |zi_g| < 1 (fusion lowers
    an alarm whose axes are all flagged by one level).
  * Coherence: behavior.common.<g> at '__system__' and at every class key
    (median, direction fractions, run length) feeds B18; one system_shift
    (INFO) is emitted when >= 50 % of the system's scored entities show
    |z_g| > 2 in the same direction for >= 2 consecutive ticks.

p for an exclusion detector d at a tick is behavior.p[d] (B24) if finite,
else behavior.pm[d], else 10^-score[d] (these scores are -log10 tails).

State. B05 owns no contract-C model, so the learner state (4 decayed sums
per group), its GateState, the current loadings and the per-tick learning
rows (L_g, z_g for the committed-row fetch; kept 8 d = the gating journal
horizon, so rollback replays and releases are exact) live in the engine
instance. Checkpoints go through store.put_checkpoint as usual. A restarted
engine starts again from beta = 1, which is the safe prior.

Absence is data: entities with no behavior.z row at ctx.now get nothing.
An active entity (feature.active = 1) without a z row means B04 was
degraded: it gets an all-NaN zi row (unscored, never "normal").
ctx.training learns as usual but emits no events.

Store: reads behavior.z, feature.active, behavior.quarantine (via gating),
behavior.score / pm / p / acc_alarm (t-1 and now), model.class (m_class),
store.events(first_seen, rare_access), store.matches, model.control and
behavior.trust / trust_prov (via gating); writes behavior.zi (52-dim float32
ring), behavior.common.flag (dict, real entities), behavior.common.<g> (dict,
'__system__' and class keys), checkpoints 'common_mode', event system_shift.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import BehaviorEvent, DerivedMetric, MetricKind, Severity
from .lib import emit
from .lib import gating as G
from .lib import m_class
from .lib.classkeys import SYSTEM_KEY, class_kind
from .lib.detectors import DETECTOR_INDEX
from .lib.features import FEATURE_DIM, FEATURE_INDEX, GROUPS

Z = "behavior.z"
ZI = "behavior.zi"
ACTIVE = "feature.active"
FLAG = "behavior.common.flag"
COMMON_PREFIX = "behavior.common."
LEARNER = "common_mode"
EVENT_KIND = "system_shift"

# eligible groups (contract: never breadth, categorical, dns, tls, identity, comp)
ELIGIBLE: Dict[str, List[int]] = {
    "volume": list(GROUPS["volume"]),
    "transport": list(GROUPS["transport"]),
    "app_error": [FEATURE_INDEX["http_5xx_rate"], FEATURE_INDEX["http_latency"]],
    "probe": list(GROUPS["probe"]),
}
GROUP_NAMES: Tuple[str, ...] = tuple(ELIGIBLE)
NG = len(GROUP_NAMES)
COMMON_SERIES: Dict[str, str] = {g: COMMON_PREFIX + g for g in GROUP_NAMES}

MIN_OTHERS = 3                  # LOO median needs >= 3 other members
RIDGE = 4.0                     # beta = (sum L z + 4) / (sum L^2 + 4)
REFIT_TICKS = 16
HALF_LIFE_S = 7 * 86400.0       # decay of the loading sums (rows are committed late / out of order)
BETA_CLIP = (0.0, 2.0)          # an anti-correlated or runaway loading is never applied
FLAG_Z = 2.0                    # flag: |z_g| > 2 and |zi_g| < 1
FLAG_ZI = 1.0
EXCL_P = 0.01                   # client / identity / budget_exfil evidence
EXCL_DETECTORS = ("client", "identity", "budget_exfil")
NOVELTY_KINDS = ("first_seen", "rare_access")
SYSTEM_TIERS = frozenset({"system", "org", "fs_system"})
MATCH_MIN_RANK = 2              # >= MEDIUM
COHERENT_Z = 2.0
COHERENT_FRAC = 0.5
COHERENT_TICKS = 2
COHERENT_MIN_N = 3              # "50 % of the system" needs a system
BUF_MAX_AGE_S = G.JOURNAL_MAX_AGE_S

_SEV_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
_NAN = math.nan

# 52 x NG 0/1 membership: group sums / counts in one matmul
_GMAT = np.zeros((FEATURE_DIM, NG), dtype=np.float64)
for _j, _g in enumerate(GROUP_NAMES):
    _GMAT[ELIGIBLE[_g], _j] = 1.0
_COL_GROUP = np.full(FEATURE_DIM, -1, dtype=np.int64)          # feature -> group slot or -1
for _j, _g in enumerate(GROUP_NAMES):
    _COL_GROUP[ELIGIBLE[_g]] = _j
_ELIG_COLS = np.flatnonzero(_COL_GROUP >= 0)
_ELIG_SLOT = _COL_GROUP[_ELIG_COLS]
_EXCL_COLS = [DETECTOR_INDEX[d] for d in EXCL_DETECTORS]
_EXFIL_COL = DETECTOR_INDEX["budget_exfil"]


# ================================================================ learner
# The state is 3 x NG plain floats: pure-Python tuples beat 4-element numpy
# arrays by ~5x per committed row, and dump / load are trivially JSON-safe.
def _init_state() -> Dict[str, Any]:
    z = (0.0,) * NG
    return {"t": _NAN, "lz": z, "ll": z, "n": z}


def _update(state: Dict[str, Any], row: "_Row", w: float) -> Dict[str, Any]:
    """Fold one committed row (L_g, z_g) with weight w. Decay is applied to
    the ROW when it is older than the state (release commits old rows after
    newer ones), so restore + replay is bit-identical (gating contract)."""
    ts, L, zb = row
    t0 = state["t"]
    lz, ll, n = state["lz"], state["ll"], state["n"]
    if not math.isfinite(t0) or ts > t0:
        if math.isfinite(t0):
            d = 2.0 ** (-(ts - t0) / HALF_LIFE_S)
            lz, ll, n = (tuple(x * d for x in lz), tuple(x * d for x in ll),
                         tuple(x * d for x in n))
        t0, wr = ts, float(w)
    else:
        wr = float(w) * 2.0 ** (-(t0 - ts) / HALF_LIFE_S)
    if wr > 0.0:
        lz2, ll2, n2 = list(lz), list(ll), list(n)
        for j in range(NG):
            lj, zj = L[j], zb[j]
            if math.isfinite(lj) and math.isfinite(zj):
                lz2[j] += wr * lj * zj
                ll2[j] += wr * lj * lj
                n2[j] += wr
        lz, ll, n = tuple(lz2), tuple(ll2), tuple(n2)
    return {"t": t0, "lz": lz, "ll": ll, "n": n}


def _dump(state: Dict[str, Any]) -> Dict[str, Any]:
    return {"t": state["t"], "lz": list(state["lz"]), "ll": list(state["ll"]),
            "n": list(state["n"])}


def _load(blob: Any) -> Dict[str, Any]:
    if not blob:
        return _init_state()
    return {"t": float(blob.get("t", _NAN)), "lz": tuple(float(x) for x in blob["lz"]),
            "ll": tuple(float(x) for x in blob["ll"]), "n": tuple(float(x) for x in blob["n"])}


def loadings(state: Dict[str, Any], now: float) -> np.ndarray:
    """beta_g = (sum w L z + 4) / (sum w L^2 + 4) with the sums decayed to now."""
    t0 = state["t"]
    d = 2.0 ** (-(now - t0) / HALF_LIFE_S) if math.isfinite(t0) and now > t0 else 1.0
    beta = (d * np.asarray(state["lz"]) + RIDGE) / (d * np.asarray(state["ll"]) + RIDGE)
    return np.clip(beta, *BETA_CLIP)


class _Row(NamedTuple):
    ts: float
    L: Tuple[float, ...]     # [NG] factor used at ts (NaN: no tier)
    zb: Tuple[float, ...]    # [NG] entity's own group means at ts


class _Buf:
    """Per-entity learning rows (ts-sorted), kept BUF_MAX_AGE_S for exact
    commit / release / rollback replay. Rows are 2*NG float32."""

    __slots__ = ("ts", "rows", "i0", "n")

    def __init__(self, cap: int = 64) -> None:
        self.ts = np.empty(cap, dtype=np.float64)
        self.rows = np.empty((cap, 2 * NG), dtype=np.float32)
        self.i0 = 0
        self.n = 0

    def append(self, ts: float, L: np.ndarray, zb: np.ndarray) -> None:
        if self.n > self.i0 and ts <= self.ts[self.n - 1]:
            # same tick rewritten, or the clock went back (new run / replay):
            # rows at or after ts are superseded
            self.n = max(self.i0, int(np.searchsorted(self.ts[self.i0:self.n], ts)) + self.i0)
        if self.n == self.ts.shape[0]:
            live = self.n - self.i0
            cap = max(64, 2 * live)
            ts_new = np.empty(cap, dtype=np.float64)
            rows_new = np.empty((cap, 2 * NG), dtype=np.float32)
            ts_new[:live] = self.ts[self.i0:self.n]
            rows_new[:live] = self.rows[self.i0:self.n]
            self.ts, self.rows, self.i0, self.n = ts_new, rows_new, 0, live
        self.ts[self.n] = ts
        self.rows[self.n, :NG] = L
        self.rows[self.n, NG:] = zb
        self.n += 1

    def get(self, ts: float) -> Optional[_Row]:
        i = int(np.searchsorted(self.ts[self.i0:self.n], ts)) + self.i0
        if i < self.n and self.ts[i] == ts:
            r = self.rows[i].tolist()
            return _Row(float(ts), tuple(r[:NG]), tuple(r[NG:]))
        return None

    def prune(self, before: float) -> None:
        self.i0 = int(np.searchsorted(self.ts[self.i0:self.n], before)) + self.i0

    def newest(self) -> float:
        return float(self.ts[self.n - 1]) if self.n > self.i0 else -math.inf


class _Ent:
    __slots__ = ("state", "gate", "beta", "buf", "csig")

    def __init__(self) -> None:
        self.state = _init_state()
        self.gate = G.GateState()
        self.beta = np.ones(NG)
        self.buf = _Buf()
        self.csig: Optional[Tuple] = None


# =========================================================== pure helpers
def group_means(Zm: np.ndarray) -> np.ndarray:
    """[n, 52] z -> [n, NG] nanmean over each eligible group (NaN if none)."""
    fin = np.isfinite(Zm)
    s = np.where(fin, Zm, 0.0) @ _GMAT
    c = fin.astype(np.float64) @ _GMAT
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1.0), np.nan)


def loo_median(v: np.ndarray, pool: np.ndarray, min_others: int = MIN_OTHERS) -> np.ndarray:
    """Leave-one-out median for every row of v ([n] or [n, k], columns
    independent).

    pool: bool mask (same shape) of the finite values that may define the
    median. A pool position gets the median of its column's pool without
    itself; any other position gets the median of the whole pool; NaN where
    fewer than `min_others` values remain. O(n log n) per column: removing
    the element of rank r from the sorted pool shifts indices >= r by one."""
    one = v.ndim == 1
    V = v[:, None] if one else v
    P = pool[:, None] if one else pool
    n, k = V.shape
    X = np.where(P, V, np.inf)                       # non-pool sorts last
    order = np.argsort(X, axis=0, kind="stable")
    Xs = np.take_along_axis(X, order, axis=0)
    m = P.sum(axis=0)
    cols = np.arange(k)
    lo, hi = np.clip((m - 1) // 2, 0, n - 1), np.clip(m // 2, 0, n - 1)
    full = np.where(m >= min_others, 0.5 * (Xs[lo, cols] + Xs[hi, cols]), np.nan)
    out = np.where(P, np.nan, full[None, :])
    kk = m - 1
    if n >= 2 and (kk >= min_others).any():
        r = np.empty_like(order)
        np.put_along_axis(r, order, np.broadcast_to(np.arange(n)[:, None], (n, k)), axis=0)
        lo, hi = np.clip((kk - 1) // 2, 0, n - 2), np.clip(kk // 2, 0, n - 2)
        a = np.where(lo[None, :] < r, Xs[lo, cols][None, :], Xs[lo + 1, cols][None, :])
        b = np.where(hi[None, :] < r, Xs[hi, cols][None, :], Xs[hi + 1, cols][None, :])
        out = np.where(P & (kk >= min_others)[None, :], 0.5 * (a + b), out)
    return out[:, 0] if one else out


_CTRL_KEYS = ("version", "branch", "rebase_from", "rollback_to", "release", "frozen",
              "allow_drift")


def _control_sig(control: Optional[Dict[str, Any]]) -> Optional[Tuple]:
    """Cheap change detector for model.control (B28 may re-put or mutate it)."""
    if not control:
        return None
    return tuple(repr(control.get(k)) for k in _CTRL_KEYS)


def _sev_rank(sev: Any) -> int:
    v = getattr(sev, "value", sev)
    return _SEV_RANK.get(str(v).lower(), 0)


def _tier_is_system(extra: Any) -> bool:
    if not isinstance(extra, dict):
        return False
    t = extra.get("tier")
    if isinstance(t, str) and t.lower() in SYSTEM_TIERS:
        return True
    tiers = extra.get("tiers")
    if isinstance(tiers, (list, tuple, set)):
        return any(isinstance(x, str) and x.lower() in SYSTEM_TIERS for x in tiers)
    return bool(extra.get("fs_system"))


# ================================================================ engine
class CommonModeEngine(Engine):
    name = "behavior.common_mode"
    layer = "behavior"
    consumes = [Z, ACTIVE, "behavior.quarantine", "behavior.trust", "behavior.trust_prov",
                "model.class", "model.control", "behavior.score", "behavior.pm", "behavior.p",
                "behavior.acc_alarm", "event.first_seen", "event.rare_access", "match.*"]
    produces = [ZI, FLAG, COMMON_PREFIX + "*", "event." + EVENT_KIND]
    description = ("Leave-one-out class/system common factor on volume, transport, app-error "
                   "and probe groups; ridge-shrunk gated loadings; zi = z - beta*L with "
                   "anti-laundering exclusions; coherent system_shift.")
    interval = 1

    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        self.refit_ticks = int(params.get("refit_ticks", REFIT_TICKS))
        self._ents: Dict[Tuple[str, str], _Ent] = {}
        self._learners: Dict[float, G.GatedLearner] = {}
        self._cur: Optional[_Buf] = None
        self._class_cache: Dict[str, Tuple[Tuple, Dict[str, List[str]]]] = {}

    # ------------------------------------------------------------ learner
    def _learner(self, d_min_s: float) -> G.GatedLearner:
        lrn = self._learners.get(d_min_s)
        if lrn is None:
            lrn = self._learners[d_min_s] = G.GatedLearner(
                name=LEARNER, init=_init_state, update=_update, fetch=self._fetch,
                dump=_dump, load=_load, d_min_s=d_min_s, ckpt_every_s=G.CKPT_EVERY_S,
                clock=ZI)
        return lrn

    def _fetch(self, store, s: str, e: str, ts: float) -> Optional[_Row]:
        buf = self._cur
        return buf.get(ts) if buf is not None else None

    def beta(self, system: str, entity: str) -> np.ndarray:
        """Current loadings (diagnostics / tests): {group order GROUP_NAMES}."""
        st = self._ents.get((system, entity))
        return st.beta.copy() if st is not None else np.ones(NG)

    # ---------------------------------------------------------------- run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        now, dt = float(ctx.now), float(ctx.window_s)
        d_min = ctx.config.get("D_min_s", G.D_MIN_S)
        lrn = self._learner(float(d_min) if d_min else G.D_MIN_S)
        n = 0
        for s in store.systems():
            n += self._system(ctx, lrn, s, now, dt)
        return n

    def _system(self, ctx: Context, lrn: G.GatedLearner, s: str, now: float, dt: float) -> int:
        store = ctx.store
        ents: List[str] = []
        rows: List[np.ndarray] = []
        for e in store.entities(s):
            z = store.vec_at(s, e, Z, now)
            if z is not None:
                ents.append(e)
                rows.append(z)
            else:
                a = store.vec_at(s, e, ACTIVE, now)
                if a is not None and a[0] >= 0.5:
                    # active but unscored by B04: degraded, never "normal"
                    store.add_vec(s, e, ZI, now, np.full(FEATURE_DIM, np.nan, np.float32),
                                  window_s=int(dt))
                self._learn(ctx, lrn, s, e, now, dt)
        if not ents:
            # nobody scored this tick: the aggregates are undefined, written
            # as such so the series keep their cadence
            empty = np.zeros((0, NG))
            none = np.zeros(0, dtype=np.int64)
            no = np.zeros(0, dtype=bool)
            self._aggregates(ctx, s, SYSTEM_KEY, none, empty, no, now, dt,
                             len(store.entities(s)), coherence=True)
            for ck, mem in self._classes(store, s).items():
                self._aggregates(ctx, s, ck, none, empty, no, now, dt, len(mem),
                                 coherence=False)
            return 0
        n = len(ents)
        Zm = np.asarray(rows, dtype=np.float64)                  # [n, 52]
        zb = group_means(Zm)                                     # [n, NG]
        # a z row at now means active (B04 scores only feature.active == 1 ticks)
        excluded = np.array([self._excluded(store, s, e, now, dt) for e in ents], dtype=bool)
        quar = np.array([G.is_quarantined(store, s, e, now, dt) for e in ents], dtype=bool)
        eligible = ~(excluded | quar)

        # class map (role classes only for the LOO tier; every class key for outputs)
        pos = {e: i for i, e in enumerate(ents)}
        class_idx: Dict[str, np.ndarray] = {}
        class_size: Dict[str, int] = {}
        role_of = np.full(n, -1, dtype=np.int64)
        role_keys: List[str] = []
        for ck, mem in self._classes(store, s).items():
            idx = np.array(sorted(pos[m] for m in mem if m in pos), dtype=np.int64)
            # every class key gets its aggregates each tick (undefined when
            # no member is scored: see _aggregates), not only the classes
            # with an active member
            class_size[ck] = len(mem)
            class_idx[ck] = idx
            if idx.size:
                if class_kind(ck) == "role":
                    role_of[idx] = len(role_keys)
                    role_keys.append(ck)

        # leave-one-out factor per group: class tier, else system tier, else 0
        pool = eligible[:, None] & np.isfinite(zb)
        L = np.full((n, NG), np.nan)
        for r in range(len(role_keys)):
            idx = np.flatnonzero(role_of == r)
            L[idx] = loo_median(zb[idx], pool[idx])
        miss = ~np.isfinite(L)
        if miss.any():
            L[miss] = loo_median(zb, pool)[miss]
        L_used = np.where(np.isfinite(L), L, 0.0)

        # score (vectorised with each entity's loadings as of its last refit)
        sts: List[_Ent] = []
        for e in ents:
            st = self._ents.get((s, e))
            if st is None:
                st = self._ents[(s, e)] = _Ent()
            sts.append(st)
        B = np.array([st.beta for st in sts])                   # [n, NG]
        shift = np.where(excluded[:, None], 0.0, B * L_used)   # excluded: zi = z
        Zim = Zm.copy()
        Zim[:, _ELIG_COLS] -= shift[:, _ELIG_SLOT]
        with np.errstate(invalid="ignore"):
            F = (np.abs(zb) > FLAG_Z) & (np.abs(zb - shift) < FLAG_ZI) & ~excluded[:, None]
        Zi32 = Zim.astype(np.float32)
        learn_row = ~excluded & np.isfinite(L).any(axis=1)
        for i, e in enumerate(ents):
            store.add_vec(s, e, ZI, now, Zi32[i], window_s=int(dt))
            store.add_derived(DerivedMetric(
                name=FLAG, value=dict(zip(GROUP_NAMES, F[i].astype(int).tolist())), ts=now,
                system=s, entity=e, window_s=int(dt), kind=MetricKind.CATEGORICAL))
            if learn_row[i]:
                sts[i].buf.append(now, L[i], zb[i])
            self._learn(ctx, lrn, s, e, now, dt)

        # system / class aggregates and coherence
        n_sys = len(store.entities(s))
        self._aggregates(ctx, s, SYSTEM_KEY, np.arange(n), zb, eligible, now, dt, n_sys,
                         coherence=True)
        for ck, idx in class_idx.items():
            self._aggregates(ctx, s, ck, idx, zb, eligible, now, dt, class_size[ck],
                             coherence=False)
        return n

    def _classes(self, store, s: str) -> Dict[str, List[str]]:
        """{class key: members in s} via m_class, cached per model.class
        object / version (the membership changes at B02's refits only)."""
        mc = m_class.get(store)
        sig = (id(mc), m_class.version(store), len(mc.get("assign", {}) or {}))
        hit = self._class_cache.get(s)
        if hit is not None and hit[0] == sig:
            return hit[1]
        out = {ck: m_class.class_members(store, s, ck) for ck in m_class.all_class_keys(store, s)}
        self._class_cache[s] = (sig, out)
        return out

    # ------------------------------------------------------------ learning
    def _learn(self, ctx: Context, lrn: G.GatedLearner, s: str, e: str, now: float,
               dt: float) -> None:
        """Commit due rows and refit the loadings. The gate is stepped only on
        the entity's refit tick (every refit_ticks, per-entity phase) or when
        model.control changes: loadings are only read at refits, so per-tick
        commits would buy nothing. Batching is never less conservative than
        per-tick commits: a quarantined tick has trust 0 anyway, and a
        quarantine at step time holds the whole batch (release commits it)."""
        st = self._ents.get((s, e))
        if st is None:
            return
        store = ctx.store
        csig = _control_sig(store.get_model(s, e, G.CONTROL_MODEL))
        due = self.entity_due((s, e, LEARNER), now, period_s=self.refit_ticks * dt)
        if not due and csig == st.csig:
            return
        st.csig = csig
        self._cur = st.buf
        try:
            st.state, st.gate = lrn.step(store, s, e, st.state, st.gate, now, dt,
                                         training=ctx.training)
        finally:
            self._cur = None
        st.buf.prune(now - BUF_MAX_AGE_S)          # what commit / release / replay may fetch
        st.beta = loadings(st.state, now)

    # ---------------------------------------------------------- exclusions
    def _excluded(self, store, s: str, e: str, now: float, dt: float) -> bool:
        since = now - dt
        for m in store.matches(s, e, since=since, limit=64):
            if _sev_rank(m.severity) >= MATCH_MIN_RANK:
                return True
        for ev in store.events(s, e, since=since, kinds=NOVELTY_KINDS, limit=64):
            if _tier_is_system(ev.extra):
                return True
        ts_s, S = store.vec_tail(s, e, emit.SCORE, 2)
        for k in range(ts_s.shape[0]):
            t = float(ts_s[k])
            if t > now or not (t == now or now - t <= 2.0 * dt + 1e-6):
                continue
            row = S[k]
            if not np.isfinite(row[_EXCL_COLS]).any():
                continue
            p_row = emit.read_array(store, s, e, emit.P, t)
            pm_row = emit.read_array(store, s, e, emit.PM, t)
            for c in _EXCL_COLS:
                sc = float(row[c])
                if not math.isfinite(sc):
                    continue
                p = float(p_row[c])
                if not math.isfinite(p):
                    p = float(pm_row[c])
                if not math.isfinite(p):
                    p = 10.0 ** (-max(sc, 0.0))
                if p < EXCL_P:
                    return True
            if math.isfinite(float(row[_EXFIL_COL])):
                acc = emit.read_dict(store, s, e, emit.ACC_ALARM, t)
                if int(acc.get("budget_exfil", 0) or 0) == 1:
                    return True
        return False

    # ------------------------------------------------------- aggregates
    def _aggregates(self, ctx: Context, s: str, key: str, idx: np.ndarray, zb: np.ndarray,
                    eligible: np.ndarray, now: float, dt: float, n_members: int,
                    coherence: bool) -> None:
        """behavior.common.<g> at a pseudo-entity: non-LOO median over the
        eligible scored members, direction fractions over all scored members,
        and (system only) the coherent-direction run length."""
        store = ctx.store
        runs: Dict[str, Tuple[int, int, int]] = {}         # g -> (dir, run, prev_run)
        vals: Dict[str, Dict[str, Any]] = {}
        V = zb[idx]                                        # [k, NG]
        fin = np.isfinite(V)
        nfs = fin.sum(axis=0)
        ups = (V > COHERENT_Z).sum(axis=0)                 # NaN compares False
        downs = (V < -COHERENT_Z).sum(axis=0)
        P = np.sort(np.where(fin & eligible[idx][:, None], V, np.inf), axis=0)
        nps = (fin & eligible[idx][:, None]).sum(axis=0)
        for j, g in enumerate(GROUP_NAMES):
            nf = int(nfs[j])
            npool = int(nps[j])
            # no member scored on this group this tick: the aggregate is
            # undefined, written as such (L NaN, dir 0) rather than skipped,
            # so the series keeps its cadence (contract: a produced series
            # with an undefined value writes NaN, it does not go silent)
            up = float(ups[j]) / nf if nf else _NAN
            down = float(downs[j]) / nf if nf else _NAN
            med = _NAN
            if npool:
                h = npool // 2
                med = float(P[h, j]) if npool % 2 else 0.5 * float(P[h - 1, j] + P[h, j])
            dirn = 0
            if nf >= COHERENT_MIN_N:
                if up >= COHERENT_FRAC:
                    dirn = 1
                elif down >= COHERENT_FRAC:
                    dirn = -1
            val: Dict[str, Any] = {
                "L": med, "n": npool, "n_scored": nf, "n_members": int(n_members),
                "frac_up": up, "frac_down": down, "dir": dirn, "dt": dt,
            }
            if coherence:
                prev = _prev_point(store, s, key, COMMON_SERIES[g], now, dt)
                prun = int(prev.get("run", 0)) if prev and int(prev.get("dir", 0)) == dirn else 0
                pr_any = int(prev.get("run", 0)) if prev else 0
                run = prun + 1 if dirn != 0 else 0
                val["run"] = run
                runs[g] = (dirn, run, pr_any)
            vals[g] = val
            store.add_derived(DerivedMetric(name=COMMON_SERIES[g], value=val, ts=now, system=s,
                                            entity=key, window_s=int(dt),
                                            kind=MetricKind.CATEGORICAL))
        if coherence and runs and not ctx.training:
            self._maybe_emit(store, s, now, dt, runs, vals)

    @staticmethod
    def _maybe_emit(store, s: str, now: float, dt: float,
                    runs: Dict[str, Tuple[int, int, int]], vals: Dict[str, Dict[str, Any]]) -> None:
        """One system_shift per episode: a group reaches COHERENT_TICKS while
        no group was already coherent at the previous tick."""
        hot = {g: r for g, r in runs.items() if r[0] != 0 and r[1] >= COHERENT_TICKS}
        if not hot:
            return
        if any(r[2] >= COHERENT_TICKS for r in runs.values()):
            return                                   # episode already announced
        groups = sorted(hot)
        frac = max(vals[g]["frac_up"] if hot[g][0] > 0 else vals[g]["frac_down"] for g in groups)
        desc = ", ".join(f"{g} {'up' if hot[g][0] > 0 else 'down'} "
                         f"({(vals[g]['frac_up'] if hot[g][0] > 0 else vals[g]['frac_down']):.0%} "
                         f"of {vals[g]['n_scored']})" for g in groups)
        run = max(hot[g][1] for g in groups)
        store.add_event(BehaviorEvent(
            system=s, entity=SYSTEM_KEY, ts=now, kind=EVENT_KIND, score=float(frac),
            severity=Severity.INFO, description=f"System-wide common-mode shift: {desc}",
            extra={"groups": {g: {"dir": hot[g][0], "run": hot[g][1], "L": vals[g]["L"],
                                  "frac_up": vals[g]["frac_up"],
                                  "frac_down": vals[g]["frac_down"],
                                  "n_scored": vals[g]["n_scored"]} for g in groups}},
            axes=[g for g in groups],
            dedupe_key=f"{EVENT_KIND}|{s}|{now:.0f}",
            window=(now - (run - 1) * dt, now)))


def _prev_point(store, s: str, key: str, name: str, now: float, dt: float) -> Optional[Dict]:
    """The dict written at the previous tick (ts < now, within 1.5 ticks of
    either cadence), else None."""
    for m in reversed(store.derived_tail(s, key, name, 2)):
        if m.ts >= now or not isinstance(m.value, dict):
            continue
        pdt = float(m.value.get("dt", dt) or dt)
        return m.value if now - m.ts <= 1.5 * max(dt, pdt) + 1e-6 else None
    return None
