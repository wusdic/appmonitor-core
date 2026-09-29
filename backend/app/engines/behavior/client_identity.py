"""ClientIdentityEngine (B09): a different client stack on a known IP.

Why: an IP says where traffic comes from, not what sends it. When a stolen
credential is replayed from a script on a user's machine, when a tool is
pointed at a service from a known host, or when a second device hides behind
the same NAT address, the IP's volumes and timing may look normal while the
client stack (TLS library, UA, OS, TCP stack; lib/stack tokens from R3)
changes. The hard part is not seeing a new stack but telling an intruder
from the weekly browser upgrade and the org-wide rollout, so the score
combines four discriminators instead of alarming on novelty alone.

Per entity and tick (inputs: R3 client.stack_set / client.stack_events /
client.os_ua_ttl_pairs, all timestamped by the real event times):
  1. Model (lib/m_client): a hierarchical Dirichlet over stack tokens,
     p(s|e) = (n_e + 5 p(s|class)) / (N_e + 5), backing off to the system,
     counts decayed with a 14-d half-life. Counts are slot-equivalents (a
     tick adds share * dt / 900), so N_e is how long the client was seen,
     identical at 60 / 900 / 3600-s ticks and not inflated by request volume.
  2. Surprise S = sum_s share_s * min(20, -log2 p(s|e)) with shares over the
     requests of the last 10 ticks: the bits per request of the recent client
     mix. A new stack ramps in with its share, so a one-tick blip and the
     first tick of a replacement stay small while a sustained takeover grows.
  3. Discriminators for each stack s1 new to the entity (p(s1|e) < 0.05):
       C  concurrency: the dominant stack s0 (p > 0.5) and s1 have stack
          episodes (client.stack_events intervals, R3 cuts them at 5-min
          gaps) that overlap or lie within 5 min of each other, including
          s0 episodes of the previous tick, and s0 is still active after
          s1's first event: two clients at once. A clean handover (s0 stops,
          s1 starts seconds later: a browser restart or upgrade) is left to
          the replacement test instead; counting it would flag every upgrade.
       I  inconsistency: the UA-declared OS contradicts the TTL class
          (stack.os_ttl_consistent on the token, or on a matching
          client.os_ua_ttl_pairs entry), or p(ja3n | UA family/major) < 0.01
          in the system co-occurrence table (>= 16 slot-equivalents of that
          UA): a copied UA on another TLS library.
       R  rollout: the share of the other class members, and of the other
          entities of the system, that first used s1 within 7 d (the larger
          of the two; acquisitions are recorded live in the system model, and
          only once the entity's chain has evidence, so start-up is not a
          rollout).
     risk = sigmoid(0.6 S + 3 C + 2 I - 4 R - 3), the largest over the new
     stacks (without one, C = I = R = 0). score.client = -log10(1 - risk +
     1e-6); behavior.pm.client = 1 - risk + 1e-6. Instantaneous, axis identity.
  4. Replacement: s0 absent for longer than its P99 absence gap, measured on
     the entity's ACTIVE clock (a silent host has not replaced anything;
     floor max(30 min, 2 dt)). Events:
       client_impersonation (HIGH) when C or I holds and risk > 0.8
         (once per (entity, stack) per 24 h);
       client_change (INFO) on a replacement s0 -> s1 when R >= 0.3 or the
         UA family is the same with a higher major version.
  5. Learning is trust-gated, delayed, checkpointed and reversible through
     lib/gating.GatedLearner (model.control rollback_to / release /
     rebase_from / frozen honoured; link seeding merges a linked entity's
     counts at weight 0.5). Tick rows are kept 8 d in the model because R3's
     raw series live 1 h. The system tier (with the class tiers and the
     ja3n | UA co-occurrence table) is rebuilt hourly from the committed
     entity models, so it inherits their gating and rollbacks.
Training mode learns (trust 1) and scores but emits no events. A tick with
no fingerprinted traffic is not scored (absence of a stack is data, not
evidence about the client); a tick where R3 failed writes NaN + degraded.
The pure read side (predictive, loglik for identity candidates, descriptors,
layouts) lives in lib/m_client.py.
"""
from __future__ import annotations

import bisect
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ...core.engine import Context, Engine
from ...models.schema import BehaviorEvent, EntityProfile, Severity
from .lib import combine, emit
from .lib import gating as G
from .lib import grains as GR
from .lib import m_class
from .lib import m_client as MC
from .lib.classkeys import SYSTEM_KEY, role_key
from .lib.stack import os_ttl_consistent, stack_id

DETECTOR = "client"
AXES = ["identity"]
LEARNER = "client"
R3_ENGINE = "raw.client_stack"
STACK_SET = "client.stack_set"
STACK_EVENTS = "client.stack_events"
PAIRS = "client.os_ua_ttl_pairs"

RECENT_TICKS = 10                  # share window for S
RECENT_GRAIN_S = 900.0             # canonical: the window is 10 x max(dt, 900 s) (round 4)
CONC_GAP_S = 300.0                 # concurrency: episodes within 5 min
REPLACE_MIN_S = 1800.0             # replacement floor on the active clock
COOC_P_MIN = 0.01                  # p(ja3n | UA) below this -> inconsistent
COOC_MIN_N = 16.0                  # slot-equivalents of the UA in the table
MIN_EVIDENCE = 16.0                # N_e + N_c + N_s below this -> unscored
RISK_EVENT = 0.8
ROLLOUT_CHANGE = 0.3
W_S, W_C, W_I, W_R, W_0 = 0.6, 3.0, 2.0, 4.0, 3.0
RISK_EPS = 1e-6
SYS_REBUILD_S = 3600.0
PROFILE_EVERY_S = 3600.0
ALERT_HOLD_S = 86400.0
ROW_KEEP_S = G.JOURNAL_MAX_AGE_S + 3600.0   # rows outlive the journal and held rows
EPS_KEEP = 8                       # recent episodes kept per stack
AXES_PM_MAX = 0.05
_NAN = math.nan
_TS_EPS = 1e-6


# ================================================================ model state
def new_state() -> Dict[str, Any]:
    return {"H": MC.HALF_LIFE_S, "clock": None, "c": {}, "N": 0.0, "gaps": {}, "n_rows": 0}


def _copy_state(st: Mapping[str, Any]) -> Dict[str, Any]:
    """Structural copy (<= ENTITY_CAP small lists): update mutates in place."""
    return {"H": st["H"], "clock": st["clock"], "N": st["N"], "n_rows": st["n_rows"],
            "c": {t: list(x) for t, x in st["c"].items()},
            "gaps": {t: list(g) for t, g in st["gaps"].items()}}


def _update(st: Dict[str, Any], row: Tuple[float, tuple, tuple, tuple], w: float
            ) -> Dict[str, Any]:
    """GatedLearner update: fold one tick row (ts, tokens, slot weights,
    absence gaps) with trust weight w. The state is decayed forward to the
    newest row; an older row (release / replay) is added decayed instead, so
    any commit order gives the same counts."""
    ts, toks, ws, gaps = row
    if not (w > 0.0):
        return st
    H = float(st["H"])
    clock = st["clock"]
    if clock is None or ts > clock:
        if clock is not None:
            f = 2.0 ** (-(ts - clock) / H)
            for x in st["c"].values():
                x[0] *= f
            st["N"] *= f
        st["clock"] = clock = ts
        add = w
    else:
        add = w * 2.0 ** (-(clock - ts) / H)
    c = st["c"]
    for t, x in zip(toks, ws):
        v = x * add
        e = c.get(t)
        if e is None:
            c[t] = [v, 1, ts, ts, 0]
        else:
            e[0] += v
            e[1] += 1
            e[2] = min(e[2], ts)
            e[3] = max(e[3], ts)
        st["N"] += v
    for t, g in gaps:
        e = c.get(t)
        if e is not None:
            e[4] += 1
            gl = st["gaps"].setdefault(t, [])
            gl.append(float(g))
            del gl[:-MC.GAP_KEEP]
    if len(c) > MC.ENTITY_CAP:                   # evict the smallest (mass stays in N)
        for t, _ in sorted(c.items(), key=lambda kv: (kv[1][0], kv[0]))[:len(c) - MC.ENTITY_CAP]:
            del c[t]
            st["gaps"].pop(t, None)
    st["n_rows"] += 1
    return st


def _merge(own: Dict[str, Any], other: Mapping[str, Any], w: float) -> Dict[str, Any]:
    """Link seeding: own + w * other (other decayed to own's clock)."""
    st = _copy_state(own)
    oc = other.get("clock")
    if oc is None:
        return st
    if st["clock"] is None:
        st["clock"] = oc
    f = w * 2.0 ** (-max(0.0, st["clock"] - oc) / float(st["H"]))
    for t, x in other["c"].items():
        e = st["c"].get(t)
        if e is None:
            st["c"][t] = [x[0] * f, int(x[1]), x[2], x[3], int(x[4])]
        else:
            e[0] += x[0] * f
            e[2], e[3] = min(e[2], x[2]), max(e[3], x[3])
    st["N"] += float(other["N"]) * f
    return st


def _dump(st: Dict[str, Any]) -> Dict[str, Any]:
    return st                         # GatedLearner deep-copies blobs on put and load


def _load(blob: Any) -> Dict[str, Any]:
    return blob


class _Rows:
    """Tick rows in ascending ts, kept ROW_KEEP_S: what the gated learner folds
    at commit (t - D), release / rebase (held rows) and rollback replay. A row
    is (tokens, slot weights, gaps) as tuples, ~150 B."""

    __slots__ = ("ts", "d")

    def __init__(self) -> None:
        self.ts: List[float] = []
        self.d: List[tuple] = []

    def __len__(self) -> int:
        return len(self.ts)

    def put(self, ts: float, data: tuple) -> None:
        i = bisect.bisect_left(self.ts, ts - _TS_EPS)
        if i < len(self.ts) and abs(self.ts[i] - ts) <= _TS_EPS:
            self.d[i] = data                     # a re-run of the same tick
        else:
            self.ts.insert(i, ts)
            self.d.insert(i, data)

    def find(self, ts: float) -> Optional[tuple]:
        i = bisect.bisect_left(self.ts, ts - _TS_EPS)
        if i < len(self.ts) and abs(self.ts[i] - ts) <= _TS_EPS:
            return self.d[i]
        return None

    def any_in(self, lo: float, hi: float) -> bool:
        """A row with lo < ts <= hi exists."""
        i = bisect.bisect_right(self.ts, lo)
        return i < len(self.ts) and self.ts[i] <= hi + _TS_EPS

    def prune(self, cutoff: float) -> None:
        i = bisect.bisect_left(self.ts, cutoff)
        if i:
            del self.ts[:i]
            del self.d[:i]


def new_entity_model() -> Dict[str, Any]:
    return {"fmt": MC.FMT, "kind": "entity", "version": 0, "ts": None, "class_key": None,
            "state": new_state(), "gate": G.GateState(), "rows": _Rows(),
            "live": {"recent": [], "eps": {}, "miss": {}, "alerted": {}, "changed": {},
                     "last": {}, "snap_ts": None, "snap": None}}


def new_system_model() -> Dict[str, Any]:
    return {"fmt": MC.FMT, "kind": "system", "version": 0, "built": None,
            "H": MC.HALF_LIFE_S, "c": {}, "N": 0.0, "n_ent": 0, "classes": {},
            "cooc": {}, "ua_N": {}, "acq": {}, "known": {}}


# ================================================================ tick inputs
class _Tick:
    """One entity's fingerprinted traffic this tick."""
    __slots__ = ("counts", "eps", "pairs")

    def __init__(self, counts: Dict[str, float], eps: Dict[str, List[Tuple[float, float]]],
                 pairs: Dict[str, float]) -> None:
        self.counts, self.eps, self.pairs = counts, eps, pairs


def _f(x: Any) -> float:
    if isinstance(x, bool):
        return _NAN
    try:
        return float(x)
    except (TypeError, ValueError, OverflowError):
        return _NAN


def _read_tick(store: Any, s: str, e: str, now: float) -> Optional[_Tick]:
    """This tick's stacks {token: n}, their episodes {token: [(first, last)]}
    and the OS/UA/TTL pairs; None when the entity sent no fingerprinted
    traffic (or only unusable entries: NaN / <= 0 counts, '__other__')."""
    ss = store.latest_fresh(s, e, STACK_SET, now)
    counts = MC.stack_counts(ss)
    if not counts:
        return None
    by_id = {stack_id(t): t for t in counts}
    eps: Dict[str, List[Tuple[float, float]]] = {}
    rows = store.latest_fresh(s, e, STACK_EVENTS, now)
    if isinstance(rows, (list, tuple)):
        for r in rows:
            if not isinstance(r, (list, tuple)) or len(r) < 3:
                continue
            t = by_id.get(r[0])
            f0, l0 = _f(r[1]), _f(r[2])
            if t is None or not (math.isfinite(f0) and math.isfinite(l0)):
                continue
            eps.setdefault(t, []).append((min(f0, l0), max(f0, l0)))
    for t in counts:
        if t not in eps:                     # no episodes: the tick envelope of the stack
            v = ss.get(t) if isinstance(ss, Mapping) else None
            if isinstance(v, Mapping):
                f0, l0 = _f(v.get("first_ts")), _f(v.get("last_ts"))
                if math.isfinite(f0) and math.isfinite(l0):
                    eps[t] = [(min(f0, l0), max(f0, l0))]
    pairs: Dict[str, float] = {}
    pv = store.latest_fresh(s, e, PAIRS, now)
    if isinstance(pv, Mapping):
        for k, n in pv.items():
            n = _f(n)
            if isinstance(k, str) and k != MC.OTHER and n > 0.0 and math.isfinite(n):
                pairs[k] = n
    return _Tick(counts, eps, pairs)


class _Classes:
    """Per-tick cache of class keys and members (m_class.members scans every
    assignment, so it is resolved once per role instead of once per entity)."""

    def __init__(self, store: Any, s: str) -> None:
        self.store, self.s = store, s
        self._members: Dict[str, List[str]] = {}
        self._keys: Dict[str, Optional[str]] = {}

    def members(self, rid: str) -> List[str]:
        m = self._members.get(rid)
        if m is None:
            m = self._members[rid] = m_class.members(self.store, self.s, rid)
        return m

    def key(self, e: str) -> Optional[str]:
        if e not in self._keys:
            rid = m_class.role_id(self.store, self.s, e)
            ck = None
            if rid is not None and len(self.members(rid)) >= m_class.MIN_MEMBERS:
                ck = role_key(rid)
            self._keys[e] = ck
        return self._keys[e]

    def members_of(self, ck: Optional[str]) -> Optional[List[str]]:
        if not ck:
            return None
        return self.members(ck[len("class:"):])


class _Ent:
    __slots__ = ("e", "model", "tick", "degraded", "ck", "bk", "p")

    def __init__(self, e: str, model: Dict[str, Any], tick: Optional[_Tick],
                 degraded: bool = False) -> None:
        self.e, self.model, self.tick, self.degraded = e, model, tick, degraded
        self.ck: Optional[str] = None
        self.bk: Optional[MC.Backoff] = None
        self.p: Dict[str, float] = {}


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _json(v: Any, nd: int = 4) -> Any:
    if isinstance(v, float):
        return round(v, nd) if math.isfinite(v) else None
    return v


# ================================================================ the engine
class ClientIdentityEngine(Engine):
    name = "behavior.client_identity"
    layer = "behavior"
    consumes = [STACK_SET, STACK_EVENTS, PAIRS, "model.class", "feature.active",
                "behavior.trust", "behavior.trust_prov", "behavior.quarantine",
                "model.control", "model.link"]
    produces = ["model.client", "behavior.score", "behavior.pm", "behavior.axes",
                "behavior.degraded", "profile.extra.client_stacks",
                "event.client_impersonation", "event.client_change"]
    description = ("Client-stack identity per IP: hierarchical Dirichlet over R3 stack "
                   "tokens; share-weighted surprise, concurrency of the dominant and a new "
                   "stack, UA/OS/TTL and ja3n|UA inconsistency, rollout share -> risk; "
                   "impersonation and benign client-change events; trust-gated learning.")
    interval = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._canon = False
        self._rows_cur: Optional[_Rows] = None
        self._learner = G.GatedLearner(
            name=LEARNER, init=new_state, update=_update, fetch=self._fetch,
            dump=_dump, load=_load, merge=_merge)

    # ------------------------------------------------------------------ run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        now, dt = float(ctx.now), float(ctx.window_s)
        self._learner.d_min_s = float(ctx.config.get("D_min_s") or G.D_MIN_S)
        self._canon = GR.canonical(ctx.config)
        r3_failed = store.engine_failed(R3_ENGINE, now)
        return sum(self._system(ctx, s, now, dt, r3_failed) for s in store.systems())

    def _system(self, ctx: Context, s: str, now: float, dt: float, r3_failed: bool) -> int:
        store = ctx.store
        sysm = MC.get(store, s, SYSTEM_KEY)
        had_sys = sysm is not None and sysm.get("fmt") == MC.FMT and sysm.get("kind") == "system"
        if not had_sys:
            sysm = new_system_model()
        cls = _Classes(store, s)
        items: List[_Ent] = []
        for e in store.entities(s):
            model = store.get_model(s, e, MC.MODEL)
            if not (isinstance(model, dict) and model.get("fmt") == MC.FMT
                    and model.get("kind") == "entity"):
                model = None
            if r3_failed:
                if model is not None:
                    items.append(_Ent(e, model, None, degraded=True))
                continue
            tick = _read_tick(store, s, e, now)
            if tick is None and model is None:
                continue
            it = _Ent(e, model if model is not None else new_entity_model(), tick)
            it.ck = cls.key(e)
            items.append(it)
        if not items and not had_sys:
            return 0

        # pass 1: predictive per entity and live acquisitions (so a rollout in
        # this very tick already counts for every member)
        acq, known = sysm["acq"], sysm["known"]
        lo = now - MC.ROLLOUT_WINDOW_S
        for it in items:
            if it.tick is None:
                continue
            it.bk = MC.Backoff(it.model, MC.class_tier(sysm, it.ck), sysm, now)
            known[it.e] = now
            # without evidence at any tier every stack looks new: start-up is
            # not a rollout, so sightings count as acquisitions only once scored
            mature = it.bk.evidence() >= MIN_EVIDENCE
            for t in it.tick.counts:
                p = it.p[t] = it.bk.p(t)
                if mature and p < MC.NEW_P:
                    a = acq.setdefault(t, {})
                    if a.get(it.e, -math.inf) < lo:
                        a[it.e] = now

        n = 0
        for it in items:
            n += self._entity(ctx, s, it, sysm, cls, now, dt)

        if self.entity_due((s, SYSTEM_KEY, "rebuild"), now, SYS_REBUILD_S):
            _rebuild(store, s, sysm, now, cls)
        store.put_model(s, SYSTEM_KEY, MC.MODEL, sysm, version=sysm["version"], ts=now)
        return n

    # ------------------------------------------------------------- per entity
    def _entity(self, ctx: Context, s: str, it: _Ent, sysm: Dict[str, Any], cls: _Classes,
                now: float, dt: float) -> int:
        store, model, e = ctx.store, it.model, it.e
        win = int(dt)
        out: Optional[Dict[str, Any]] = None
        if it.degraded:
            emit.write_scores(store, s, e, now, {DETECTOR: _NAN},
                              degraded={DETECTOR: f"producer_error:{R3_ENGINE}"}, window_s=win)
        elif it.tick is not None:
            out = self._score(ctx, s, it, sysm, cls, now, dt)
        self._learn(ctx, s, e, model, now, dt)
        model["rows"].prune(now - ROW_KEEP_S)
        model["ts"], model["class_key"] = now, it.ck
        store.put_model(s, e, MC.MODEL, model, version=model["version"], ts=now)
        if out is None:
            return 1 if it.degraded else 0
        if out["scored"]:
            pm = out["pm"]
            emit.write_scores(
                store, s, e, now, {DETECTOR: out["score"]}, pm={DETECTOR: pm},
                axes={DETECTOR: AXES} if (pm <= AXES_PM_MAX or out["impersonation"]) else None,
                window_s=win)
        if self.entity_due((s, e, "profile"), now, PROFILE_EVERY_S) or out["events"]:
            _write_profile(store, s, e, model, out, now)
        return 1

    def _score(self, ctx: Context, s: str, it: _Ent, sysm: Dict[str, Any], cls: _Classes,
               now: float, dt: float) -> Dict[str, Any]:
        """Live bookkeeping (recent shares, episodes, active-clock absences,
        the learner row) and the tick's risk and events."""
        e, model, tick, bk = it.e, it.model, it.tick, it.bk
        assert tick is not None and bk is not None
        live, st = model["live"], model["state"]
        _snapshot(live, now)
        counts = tick.counts

        # recent request shares: the last 10 ticks (tick mode); canonical
        # mode (round 4, evaluator): the last 10 x max(dt, 900 s) of wall
        # clock, i.e. 2.5 h at 60 s as at 900 s. Ten 60-s ticks are ten
        # minutes, in which a second browser used briefly held most of the
        # requests: S and the risk sigmoid jumped and a human's ordinary
        # second client was reported as client_impersonation (HIGH) at 60 s
        # only (pack E seeds 0 / 1: oa-portal 10.30.2.29, pm 0.19).
        if self._canon:
            win = RECENT_TICKS * max(dt, RECENT_GRAIN_S)
            rec = [r for r in live["recent"] if now - win + _TS_EPS < r[0] < now]
        else:
            rec = [r for r in live["recent"] if now - RECENT_TICKS * dt + _TS_EPS < r[0] < now]
            rec = rec[-(RECENT_TICKS - 1):]
        rec = rec + [[now, dict(counts)]]
        live["recent"] = rec
        agg: Dict[str, float] = {}
        for _, c in rec:
            for t, v in c.items():
                agg[t] = agg.get(t, 0.0) + v
        tot = sum(agg.values())
        bits = {t: bk.bits(t) for t in agg}
        S = sum(v / tot * bits[t] for t, v in agg.items()) if tot > 0.0 else _NAN

        # dominant stack under the model (committed stacks + this tick's)
        p_of = dict(it.p)
        for t in st["c"]:
            if t not in p_of:
                p_of[t] = bk.p(t)
        s0 = max(p_of, key=lambda t: (p_of[t], t)) if p_of else None
        if s0 is not None and not p_of[s0] > 0.5:
            s0 = None

        # active-clock absences -> gap samples for the learner row
        miss = live["miss"]
        row_gaps: List[Tuple[str, float]] = []
        for t in set(st["c"]) | set(miss):
            if t in counts:
                g = miss.pop(t, 0.0)
                if g > 0.0:
                    row_gaps.append((t, g))
            elif t in st["c"]:
                miss[t] = miss.get(t, 0.0) + dt
            else:
                miss.pop(t, None)
        tot_now = sum(counts.values())
        toks = tuple(sorted(counts))
        model["rows"].put(now, (toks, tuple(counts[t] / tot_now * dt / MC.SLOT_S for t in toks),
                                tuple(sorted(row_gaps))))

        # discriminators for each stack new to the entity
        members = cls.members_of(it.ck)
        prev_eps = live["eps"]
        best: Optional[Dict[str, Any]] = None
        for s1 in toks:
            if s1 == s0 or not it.p[s1] < MC.NEW_P:
                continue
            C = int(s0 is not None and _concurrent(
                tick.eps.get(s1, ()), list(prev_eps.get(s1, ())),
                list(tick.eps.get(s0, ())) + list(prev_eps.get(s0, ()))))
            i_os, i_ja3n, p_j = _inconsistency(s1, tick.pairs, sysm)
            I = int(i_os or i_ja3n)
            R = _rollout(sysm, s1, e, now, members)
            z = (W_S * S if S == S else 0.0) + W_C * C + W_I * I - W_R * R - W_0
            risk = _sigmoid(z)
            if best is None or risk > best["risk"]:
                best = {"s1": s1, "C": C, "I": I, "I_os": bool(i_os), "I_ja3n": bool(i_ja3n),
                        "p_ja3n_ua": p_j, "R": R, "risk": risk}
        if best is None:
            best = {"s1": None, "C": 0, "I": 0, "I_os": False, "I_ja3n": False,
                    "p_ja3n_ua": _NAN, "R": 0.0,
                    "risk": _sigmoid(W_S * S - W_0) if S == S else _NAN}

        # episodes carried to the next tick (concurrency across the boundary)
        keep_lo = now - dt - CONC_GAP_S
        eps: Dict[str, List[Tuple[float, float]]] = {}
        for t, lst in list(prev_eps.items()) + list(tick.eps.items()):
            kept = [x for x in lst if x[1] >= keep_lo]
            if kept:
                eps.setdefault(t, []).extend(kept)
        live["eps"] = {t: sorted(v)[-EPS_KEEP:] for t, v in eps.items()}

        scored = bk.evidence() >= MIN_EVIDENCE and best["risk"] == best["risk"]
        risk = best["risk"]
        pm = min(1.0, 1.0 - risk + RISK_EPS) if scored else _NAN
        score = -math.log10(1.0 - risk + RISK_EPS) if scored else _NAN
        out: Dict[str, Any] = {"scored": scored, "score": score, "pm": pm, "S": S,
                               "s0": s0, "impersonation": False, "events": 0, **best}
        live["last"] = {"ts": now, "S": _json(S), "C": best["C"], "I": best["I"],
                        "R": _json(best["R"]), "risk": _json(risk), "s0": s0,
                        "s1": best["s1"], "scored": bool(scored)}

        training = bool(ctx.training)
        # client_impersonation
        if (scored and best["s1"] is not None and (best["C"] or best["I"])
                and risk > RISK_EVENT):
            out["impersonation"] = True
            last = live["alerted"].get(best["s1"], -math.inf)
            if not training and now - last >= ALERT_HOLD_S:
                live["alerted"][best["s1"]] = now
                _event_impersonation(ctx.store, s, e, now, dt, out, model)
                out["events"] += 1
        # replacement -> client_change
        if s0 is not None and s0 not in counts:
            out["events"] += self._replacement(ctx, s, e, model, sysm, members, s0, agg,
                                               counts, now, dt, training)
        live["alerted"] = {t: ts for t, ts in live["alerted"].items()
                           if now - ts < ALERT_HOLD_S}
        return out

    def _replacement(self, ctx: Context, s: str, e: str, model: Dict[str, Any],
                     sysm: Dict[str, Any], members: Optional[List[str]], s0: str,
                     agg: Dict[str, float], counts: Dict[str, float], now: float, dt: float,
                     training: bool) -> int:
        live = model["live"]
        p99 = MC.p99_gap(model, s0)
        thr = max(p99 if p99 == p99 else 0.0, REPLACE_MIN_S, 2.0 * dt)
        gap = live["miss"].get(s0, 0.0)
        if not gap > thr:
            return 0
        cands = [t for t in agg if t != s0 and t in counts]
        if not cands:
            return 0
        s1 = max(cands, key=lambda t: (agg[t], t))
        prev = live["changed"].get(s0)
        if prev is not None and prev[0] == s1 and prev[2]:
            return 0                                # already reported
        R = _rollout(sysm, s1, e, now, members)
        p0, p1 = MC.parse(s0), MC.parse(s1)
        upgrade = p0.family == p1.family and p1.major > p0.major >= 0
        emitted = (R >= ROLLOUT_CHANGE or upgrade) and not training
        live["changed"][s0] = [s1, prev[1] if prev is not None and prev[0] == s1 else now,
                               emitted]
        if not emitted:
            return 0
        ctx.store.add_event(BehaviorEvent(
            system=s, entity=e, ts=now, kind="client_change", score=0.0,
            severity=Severity.INFO,
            description=(f"client stack replaced: {p0.ua}/{p0.os} -> {p1.ua}/{p1.os}"
                         + (" (version upgrade)" if upgrade else f" (rollout {R:.0%})")),
            extra={"from": s0, "to": s1, "R": _json(R), "upgrade": bool(upgrade),
                   "absent_active_s": _json(gap), "p99_gap_s": _json(p99),
                   "threshold_s": _json(thr), "ua_from": p0.ua, "ua_to": p1.ua},
            axes=list(AXES), dedupe_key=f"client_change|{s}|{e}|{s0}|{s1}",
            model_version=int(model["gate"].version), window=(now - gap, now)))
        return 1

    # ---------------------------------------------------------------- learning
    def _learn(self, ctx: Context, s: str, e: str, model: Dict[str, Any], now: float,
               dt: float) -> None:
        """Gated commit of due rows (+ control, + link seeding). Skipped when no
        row is due and there is no model.control: the learner would only
        advance its cursor over rowless ticks, which the next step does anyway."""
        store, rows, gate = ctx.store, model["rows"], model["gate"]
        st = model["state"]
        frontier = G.commit_frontier(now, dt, self._learner.d_min_s)
        self._rows_cur = rows
        try:
            if rows.any_in(gate.last_ts, frontier) or \
                    store.get_model(s, e, G.CONTROL_MODEL) is not None:
                st, gate = self._learner.step(store, s, e, _copy_state(st), gate, now, dt,
                                              training=bool(ctx.training))
            st, gate = self._learner.seed_from_link(store, s, e, st, gate,
                                                    lambda a: _other_state(store, s, a))
        finally:
            self._rows_cur = None
        model["state"], model["gate"], model["version"] = st, gate, int(gate.version)

    def _fetch(self, store: Any, s: str, e: str, ts: float) -> Optional[tuple]:
        rows = self._rows_cur
        if rows is None:
            return None
        d = rows.find(float(ts))
        return None if d is None else (float(ts),) + d


# ============================================================== discriminators
def _concurrent(eps1: Any, eps1_prev: Any, eps0: Any) -> bool:
    """Some episode of s1 this tick and some episode of s0 (this or the
    previous tick) overlap or are <= 5 min apart, and the two interleave: s0
    is still active after s1's first event. A clean handover (s0 ends, s1
    starts seconds later: a browser restart or upgrade) is a replacement,
    not two clients at once."""
    if not eps1 or not eps0:
        return False
    close = False
    for f1, l1 in eps1:
        for f0, l0 in eps0:
            if max(f0, f1) - min(l0, l1) <= CONC_GAP_S:
                close = True
                break
        if close:
            break
    if not close:
        return False
    first1 = min(f for f, _ in list(eps1) + list(eps1_prev))
    return max(l for _, l in eps0) > first1


def _inconsistency(token: str, pairs: Mapping[str, float], sysm: Mapping[str, Any]
                   ) -> Tuple[bool, bool, float]:
    """(UA-OS vs TTL contradiction, p(ja3n | UA) < 0.01, that p)."""
    p = MC.parse(token)
    i_os = os_ttl_consistent(p.os, p.ttl) is False
    if not i_os:
        # R3's pairs carry the declared OS explicitly ('?' when none)
        for k in pairs:
            d, _, rest = k.partition("|")
            fam, _, ttl = rest.partition("|")
            if fam == p.family and ttl == p.ttl and d != "?" \
                    and os_ttl_consistent(d, ttl) is False:
                i_os = True
                break
    pj = _NAN
    i_j = False
    if p.ja3n != "-" and p.family not in ("none", "other"):
        pj, n_ua = MC.p_ja3n_given_ua(sysm, p.ja3n, p.ua)
        i_j = n_ua >= COOC_MIN_N and pj == pj and pj < COOC_P_MIN
    return i_os, i_j, pj


def _rollout(sysm: Mapping[str, Any], token: str, e: str, now: float,
             members: Optional[List[str]]) -> float:
    r_cls, r_sys = MC.rollout_share(sysm, token, e, now, members)
    vals = [r for r in (r_cls, r_sys) if r == r]
    return max(vals) if vals else 0.0


# ================================================================ helpers
def _snapshot(live: Dict[str, Any], now: float) -> None:
    """Make a re-run of the same tick exact: the first run at `now` saves the
    cross-tick state; a second run restores it (no double-counted absence)."""
    if live.get("snap_ts") == now and live.get("snap") is not None:
        snap = live["snap"]
        live["miss"] = dict(snap["miss"])
        live["eps"] = {t: list(v) for t, v in snap["eps"].items()}
        live["recent"] = list(snap["recent"])
    else:
        live["snap_ts"] = now
        live["snap"] = {"miss": dict(live["miss"]),
                        "eps": {t: list(v) for t, v in live["eps"].items()},
                        "recent": list(live["recent"])}


def _other_state(store: Any, s: str, entity: str) -> Optional[Dict[str, Any]]:
    m = MC.get(store, s, entity)
    if MC.kind(m) != "entity" or not isinstance(m.get("state"), dict):
        return None
    return _copy_state(m["state"])


def _rebuild(store: Any, s: str, sysm: Dict[str, Any], now: float, cls: _Classes) -> None:
    """System tier, class tiers and the ja3n | UA co-occurrence table from the
    members' committed entity models, decayed to `now`; prune the live
    acquisition / known tables to the rollout window."""
    c: Dict[str, float] = {}
    classes: Dict[str, Dict[str, Any]] = {}
    cooc: Dict[str, Dict[str, float]] = {}
    ua_n: Dict[str, float] = {}
    N = 0.0
    n_ent = 0
    for e in store.entities(s):
        m = MC.get(store, s, e)
        if MC.kind(m) != "entity":
            continue
        st = m["state"]
        f = MC.factor(m, now)
        if not st["c"] or not st["N"] * f > 0.0:
            continue
        n_ent += 1
        ck = cls.key(e)
        tier = None
        if ck:
            tier = classes.setdefault(ck, {"c": {}, "N": 0.0, "members": 0})
            tier["members"] += 1
            tier["N"] += st["N"] * f
        N += st["N"] * f
        for t, x in st["c"].items():
            v = x[0] * f
            c[t] = c.get(t, 0.0) + v
            if tier is not None:
                tier["c"][t] = tier["c"].get(t, 0.0) + v
            p = MC.parse(t)
            if p.ja3n != "-" and p.family not in ("none", "other"):
                row = cooc.setdefault(p.ua, {})
                row[p.ja3n] = row.get(p.ja3n, 0.0) + v
                ua_n[p.ua] = ua_n.get(p.ua, 0.0) + v
    sysm.update(c=_cap(c), N=N, n_ent=n_ent, cooc=cooc, ua_N=ua_n, built=now,
                version=int(sysm.get("version", 0)) + 1,
                classes={k: {**v, "c": _cap(v["c"])} for k, v in classes.items()})
    lo = now - MC.ROLLOUT_WINDOW_S
    acq = {}
    for t, a in sysm["acq"].items():
        a = {x: ts for x, ts in a.items() if ts >= lo}
        if a:
            acq[t] = a
    sysm["acq"] = acq
    sysm["known"] = {x: ts for x, ts in sysm["known"].items() if ts >= lo}


def _cap(c: Dict[str, float]) -> Dict[str, float]:
    if len(c) <= MC.TIER_CAP:
        return c
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:MC.TIER_CAP])


def _event_impersonation(store: Any, s: str, e: str, now: float, dt: float,
                         out: Dict[str, Any], model: Dict[str, Any]) -> None:
    s1, s0 = out["s1"], out["s0"]
    p1 = MC.parse(s1)
    why = []
    if out["C"]:
        why.append("concurrent with the dominant stack")
    if out["I_os"]:
        why.append(f"UA claims {p1.os} but TTL class {p1.ttl}")
    if out["I_ja3n"]:
        why.append(f"TLS fingerprint unseen with {p1.ua}")
    pm = out["pm"]
    store.add_event(BehaviorEvent(
        system=s, entity=e, ts=now, kind="client_impersonation", score=float(out["risk"]),
        severity=Severity.HIGH,
        description=f"new client stack {p1.ua}/{p1.os}/ttl{p1.ttl}: " + "; ".join(why),
        extra={"stack": s1, "dominant": s0, "S": _json(out["S"]), "C": out["C"],
               "I": out["I"], "I_os": out["I_os"], "I_ja3n": out["I_ja3n"],
               "p_ja3n_given_ua": _json(out["p_ja3n_ua"]), "R": _json(out["R"]),
               "risk": _json(out["risk"], 6), "ua": p1.ua, "os": p1.os, "ttl": p1.ttl},
        p_value=pm, e_day=float(combine.e_day(pm, dt)), axes=list(AXES),
        p_by_detector={DETECTOR: pm}, dedupe_key=f"client_impersonation|{s}|{e}|{s1}",
        model_version=int(model["gate"].version), window=(now - dt, now)))


def _write_profile(store: Any, s: str, e: str, model: Dict[str, Any], out: Mapping[str, Any],
                   now: float) -> None:
    """profile.extra.client_stacks: dominant stacks in plain terms, the mix
    and the last tick's discriminators (JSON-friendly: NaN -> None)."""
    d = MC.descriptors(model, k=5, now=now)
    p = store.profile(s, e) or EntityProfile(system=s, entity=e)
    p.extra["client_stacks"] = {
        "dominant": [{k: _json(v) for k, v in x.items()} for x in d["dominant"]],
        "n_stacks": d["n_stacks"], "entropy_bits": _json(d["entropy_bits"]),
        "ua_mix": {k: _json(v) for k, v in d["ua_mix"].items()},
        "os_mix": {k: _json(v) for k, v in d["os_mix"].items()},
        "maturity": _json(d["maturity"]), "recent": dict(model["live"].get("last") or {}),
        "version": int(model.get("version", 0)), "updated": now,
    }
    store.put_profile(p)
