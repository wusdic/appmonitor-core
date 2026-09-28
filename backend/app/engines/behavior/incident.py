"""B27 IncidentEngine: the only notifier (docs/lib3/engines.md B27).

Why a separate engine: every detector, fusion path and discrete producer
speaks per tick, and an analyst cannot act on per-tick output. This engine
turns alarms and findings into a few entity- and class-centric incidents
with a lifecycle, and it is the ONLY place that decides whether a human is
told. Keeping that decision in one place is what lets the eval bound
notifications per episode, and lets feedback suppress a known-benign
pattern without touching calibration or risk.

Opening (never during ctx.training; everything else still runs):
  * an alarm (behavior.alarm at ts = now, any path, entity or class key);
  * a discrete finding >= MEDIUM (store.events of the contract F discrete
    kinds, status not suppressed);
  * risk >= 30 (Medium) on the last two risk rows AND a family at
    e_day <= 0.1 within 24 h that no incident of the key has already
    covered (the family hit must be newer than the key's last incident
    activity, otherwise the decaying risk of a closed attack would reopen
    it on every tick, which is exactly what "close never on risk" forbids),
    and the risk itself must be >= 30 beyond what the key's last incident
    already covered (its risk then, decayed with the slowest B26 half-life):
    a weak null family hit on top of old risk is not a new episode.
  Lower findings, lib-4 matches (one-tick lag, store.matches(since = the
  previous run)) and the key's risk only join an existing incident.
Join: the live incident of the same entity whatever its gap (one incident
  per episode; an open incident IS the episode), or the live incident of a
  continuity alias / actor (model.link@(s, '__system__')) if its gap is
  <= max(4 ticks, 1 h). Otherwise a closed incident of the key closed
  within 24 h is reopened with its id (not one closed by a label: a
  labelled case is final for B23), else a new one opens.
Escalate on a severity increase or a new axis.
Close (never on risk):
  (a) the governor regime becomes RETURNED (a transition seen in
      behavior.regime or a regime event after the incident (re)opened);
  (b) the regime becomes ACCEPTED -> close_reason accepted (B28 may also
      close it itself; an externally closed incident still gets its close
      notification here);
  (c) a label on the incident, one of its events, or its entity / class
      (verdict other than 'unsure') -> labelled;
  (d) quiet -> timeout: no alarm or finding for max(8 ticks, 2 h), every
      accumulator < h/4 and e_day(q_inst) >= 1 on the last 4 q_inst rows.
      Accumulator level L ~ ln(1/p) / ln(ARL_ticks) from the calibrated p
      of every accumulator, cusum / mcusum included (lib/detectors.acc_level,
      shared with B28: the exponential tail P(S >= x) ~ e^(-theta x) of a
      CUSUM with theta h ~ ln ARL), and L >= 1 while acc_alarm is set.
      NaN (unscored) q_inst or p neither confirm nor block; (d) is skipped
      on a tick where fusion failed (contract M).
  An incident open for more than 14 d is handed to the label queue once
  (an 'update' notification plus an evidence mark; B23 queues open
  incidents older than 14 d as 'held'); it is never closed by age.
Common mode: when >= 50 % of a class's members (>= 3 members, >= 2
  alarming) alarm in the same tick with axes only in {volume, transport,
  app_error}, those members' incidents become children of the class-key
  incident (kinds += coherent_shift; created LOW if the class key has no
  incident of its own): status 'suppressed' + parent_id (the Incident
  status vocabulary has no 'suppressed_common'; the evidence entry carries
  state = 'suppressed_common'). Children never notify. A child whose own
  evidence gains any other axis is promoted back to a notifying root. A
  member whose incident already had other axes is not parented. Closing a
  parent closes its children with the same reason. Children are linked by
  parent_id only, NOT listed in the parent's `entities`, so B26 keeps
  their evidence at the suppressed weight.
Campaign: live root incidents of a system opened within 1 h of each other
  whose pattern tokens (axes, top-5 feature signs, new tokens; the same
  tokeniser as feedback, lib/m_feedback.build_tokens) have Jaccard >= 0.5
  AND share at least one feature or new-token (two incidents alike only in
  a bare axis such as 'volume' are not a campaign) are grouped by
  union-find under campaign_id (the oldest id already present in the
  group, else 'cmp-' + its smallest incident id).
Feedback: lib/m_feedback.suppression_match on every opened or changed
  incident; a match stores it as status 'suppressed' (still counted by
  B26 at 0.25), and a later change that no longer matches (new axis,
  rarer e_day) escapes: status open and a notification.
Notifications: BehaviorEvent(kind='incident', extra.state in {open,
  escalate, update, close}, e_day, axes, p_by_detector, incident_id).
  open / escalate spend a token from BOTH buckets (alert_budget: 3 per
  entity per hour, 20 per system per day; continuous refill); overflow is
  queued per system (one entry per incident, coalesced) and released
  highest-risk first as tokens refill. update / close are informational,
  free, and sent only for incidents that have announced themselves.
Event updates: merged findings get incident_id (and status suppressed /
  closed following the incident).

State: runtime bookkeeping (live index, token buckets, queue, dedupe sets)
lives in the engine instance per store (WeakKeyDictionary); a new engine
on an existing store rebuilds the live index from store.incidents, so a
restart loses only token-bucket levels (they restart full). Nothing is
learned from traffic, so there is nothing to gate or roll back.

Store: reads behavior.alarm, behavior.p_family, behavior.e_day,
behavior.q_inst, behavior.risk, behavior.p, behavior.acc_alarm,
behavior.regime, behavior.z (pattern features),
store.events / matches / labels / incidents, model.feedback (m_feedback),
model.link, model.class (m_class); writes store incidents (entity and
class), BehaviorEvent kind='incident', event incident_id / status updates.
"""
from __future__ import annotations

import math
import weakref
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import BehaviorEvent, Incident, Severity
from .lib import emit, m_class, m_feedback
from .lib import grains as GR
from .lib import stages as STG
from .lib.classkeys import CLASS_PREFIX, SYSTEM_KEY, class_kind, is_class
from .lib.detectors import (ACC_DETECTORS, DETECTOR_INDEX, DETECTORS, FAMILY_DEFAULT_AXES,
                            acc_level)

ALARM = "behavior.alarm"
P_FAMILY = "behavior.p_family"
E_DAY = "behavior.e_day"
Q_INST = "behavior.q_inst"
Q_INST_H = "behavior.q_inst.h"          # spec v2.1: H-stream instantaneous evidence
RISK = "behavior.risk"
REGIME = "behavior.regime"
P = emit.P
ACC_ALARM = emit.ACC_ALARM
LINK_MODEL = "model.link"
FUSION_ENGINE = "behavior.fusion"
KIND = "incident"

HOUR = 3600.0
DAY = 86400.0

# lifecycle (engines.md B27)
JOIN_TICKS, JOIN_S = 4, HOUR               # alias / actor join gap max(4 ticks, 1 h)
QUIET_TICKS, QUIET_S = 8, 2 * HOUR         # (d) no alarm / finding for max(8 ticks, 2 h)
Q_TICKS = 4                                # (d) e_day(q_inst) >= 1 on the last 4 rows
Q_E_DAY_MIN = 1.0
ACC_QUIET_LEVEL = 0.25                     # (d) every accumulator < h/4
REOPEN_S = DAY
HELD_S = 14 * DAY
# opening
OPEN_FINDING_RANK = 2                      # discrete finding >= MEDIUM opens
RISK_MEDIUM, RISK_HIGH = 30.0, 60.0
RISK_FAM_E_DAY = 0.1
RISK_LOOKBACK_S = DAY
# The slowest B26 half-life (novelty / identity, engines.md B26): an upper
# bound of what the evidence an incident already covered still adds to risk.
RISK_OLD_HL_S = 72 * HOUR
# common mode / campaign
COMMON_AXES = frozenset({"volume", "transport", "app_error"})
COMMON_FRAC = 0.5
COMMON_MIN_MEMBERS = 3
COMMON_MIN_ALARMS = 2
CAMPAIGN_S = HOUR
CAMPAIGN_JACCARD = 0.5
# evidence
PBD_E_DAY = 1.0                            # detectors at <= once-a-day rarity are reported
PBD_MAX = 8
EVIDENCE_EVERY_S = HOUR                    # repeat alarm entries at most hourly unless new
MAX_EVIDENCE, EVIDENCE_HEAD = 512, 64
QUEUE_MAX = 200
LIVE = ("open", "acked", "suppressed")
RESYNC_S = HOUR

LEVELS = ("info", "low", "medium", "high", "critical")
_RANK = {s: i for i, s in enumerate(LEVELS)}
_SCORE = {"info": 0.0, "low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
_AXIS_ALIASES = {"app": "app_error", "app-error": "app_error", "apperror": "app_error",
                 "app_errors": "app_error"}

# contract F discrete findings (not incident / regime / ops kinds)
DISCRETE_KINDS = frozenset({
    "first_seen", "rare_access", "class_adopted", "client_change", "client_impersonation",
    "identity_mismatch", "unknown_identity", "low_identifiability", "entity_resolution",
    "possible_impersonation", "shared_ip", "identity_moved", "link_retracted",
    "new_entity_matched", "new_entity_unmatched", "class_transition", "class_split",
    "class_merge", "peer_outlier", "system_shift", "coherent_shift", "class_shift",
    "class_adoption_risky", "schedule_shift", "beacon", "budget_exceeded", "baseline_creep",
})
_REGIME_CLOSE = {"returned": "returned", "accepted": "accepted"}
_ACC_IDX = [(d, DETECTOR_INDEX[d]) for d in ACC_DETECTORS]
_CANON = [False]                # spec v2.1: set per run (the grain mode of the tick)
_NAN = math.nan


# ====================================================================== helpers
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


def sev_name(x: Any) -> str:
    """'info' | 'low' | ... from a Severity, a string or None ('info')."""
    v = getattr(x, "value", x)
    s = str(v).lower() if v is not None else "info"
    return s if s in _RANK else "info"


def sev_rank(x: Any) -> int:
    return _RANK[sev_name(x)]


def canonical_axis(axis: Any) -> str:
    """Lower-case axis with the app-error spellings unified (as fusion)."""
    a = str(axis).strip().lower()
    return _AXIS_ALIASES.get(a, a)


def canonical_axes(axes: Any) -> Set[str]:
    return {canonical_axis(a) for a in (axes or ()) if a}


def acc_level_from_p(p: float, detector: str, dt_s: float,
                     period_s: Optional[float] = None) -> float:
    """Approximate S/h of an accumulator from its calibrated p (the shared
    lib/detectors.acc_level scale, also used by B28). spec v2.1: period_s
    counts the ARL in the detector's grain periods (grains.period_s)."""
    return acc_level(p, detector, dt_s, period_s)


class TokenBucket:
    """Continuous-refill token bucket: `cap` tokens per `period_s`."""
    __slots__ = ("cap", "period", "tokens", "ts")

    def __init__(self, cap: float, period_s: float, now: float) -> None:
        self.cap = float(cap)
        self.period = float(period_s)
        self.tokens = float(cap)
        self.ts = float(now)

    def level(self, now: float, cap: Optional[float] = None) -> float:
        if cap is not None and float(cap) != self.cap:
            self.cap = float(cap)
        el = max(0.0, now - self.ts)
        if self.cap > 0 and self.period > 0:
            self.tokens = min(self.cap, self.tokens + el * self.cap / self.period)
        self.ts = max(self.ts, now)
        return self.tokens

    def take(self) -> None:
        self.tokens -= 1.0


def campaign_groups(items: List[Tuple[str, float, frozenset]],
                    window_s: float = CAMPAIGN_S,
                    jaccard_min: float = CAMPAIGN_JACCARD) -> List[List[str]]:
    """Union-find over (id, opened, tokens): link two items opened within
    `window_s` whose token Jaccard >= jaccard_min and whose shared tokens
    include a feature ('f:') or new-token ('n:') token. Returns the groups
    of >= 2 ids (each sorted), in order of their smallest id."""
    n = len(items)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = sorted(range(n), key=lambda i: items[i][1])
    for a_pos, i in enumerate(order):
        ti = items[i][2]
        if not ti:
            continue
        for j in order[a_pos + 1:]:
            if items[j][1] - items[i][1] > window_s:
                break
            tj = items[j][2]
            if not tj:
                continue
            inter = ti & tj
            if not inter or not any(t[:2] in ("f:", "n:") for t in inter):
                continue
            if len(inter) / len(ti | tj) >= jaccard_min:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    groups: Dict[int, List[str]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(items[i][0])
    out = [sorted(g) for g in groups.values() if len(g) >= 2]
    out.sort(key=lambda g: g[0])
    return out


def _clean(v: Any) -> Any:
    """JSON-safe floats (NaN / inf -> None) inside evidence entries."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, (np.floating, np.integer)):
        return _clean(v.item())
    if isinstance(v, Mapping):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_clean(x) for x in v]
    return v


def _sig(x: float, digits: int = 4) -> float:
    return float(f"{x:.{digits}g}")


# ================================================================ runtime state
class _Live:
    """Per live incident bookkeeping (reconstructible from the Incident)."""
    __slots__ = ("id", "s", "key", "last_hit", "since", "announced", "supp", "pbd", "axes",
                 "feats", "new", "tokens", "tok_dirty", "ev_ts", "held_sent", "regime_ts",
                 "event_ids", "lib4")

    def __init__(self, inc: Incident, now: float, announced: bool = False) -> None:
        self.id = inc.id
        self.s = inc.system
        self.key = inc.entity
        self.last_hit = float(inc.last_seen or inc.opened or now)
        self.since = float(inc.opened or now)
        self.announced = announced
        self.supp: Optional[str] = ("common" if inc.parent_id else
                                    "policy" if inc.status == "suppressed" else None)
        self.pbd: Dict[str, float] = {}
        self.axes: Set[str] = canonical_axes(inc.axes)
        self.feats: Dict[str, float] = {}
        self.new: Set[str] = set()
        self.tokens: frozenset = frozenset()
        self.tok_dirty = True
        self.ev_ts = -math.inf               # last alarm evidence entry
        self.held_sent = False
        self.regime_ts = -math.inf
        self.event_ids: List[str] = []
        self.lib4: Dict[str, float] = {}     # signature -> last lib-4 evidence entry ts
        for ent in inc.evidence or []:
            if not isinstance(ent, Mapping):
                continue
            for d, p in (ent.get("p_by_detector") or {}).items():
                pv = _fin(p)
                if pv is not None and (d not in self.pbd or pv < self.pbd[d]):
                    self.pbd[d] = pv
            for n, z in (ent.get("features") or {}).items():
                zv = _fin(z)
                if zv is not None and (n not in self.feats or abs(zv) > abs(self.feats[n])):
                    self.feats[n] = zv
            self.new.update(str(t) for t in ent.get("new_tokens") or ())
            if ent.get("source") == "alarm":
                self.ev_ts = max(self.ev_ts, _f(ent.get("ts")))
            if ent.get("source") == "lib4" and ent.get("signature_id"):
                sid = str(ent["signature_id"])
                self.lib4[sid] = max(self.lib4.get(sid, -math.inf), _f(ent.get("ts")))
            if ent.get("event_id"):
                self.event_ids.append(str(ent["event_id"]))
            st = ent.get("state")
            if st == "label_queue":
                self.held_sent = True
            if st in ("open", "reopen"):
                self.since = max(self.since, _f(ent.get("ts")))

    def refresh_tokens(self) -> frozenset:
        if self.tok_dirty:
            self.tokens, _ = m_feedback.build_tokens((), sorted(self.axes), self.feats,
                                                     sorted(self.new))
            self.tok_dirty = False
        return self.tokens


class _RiskMem:
    __slots__ = ("until", "hit_ts", "fams", "e_min")

    def __init__(self, until: float) -> None:
        self.until = until
        self.hit_ts = -math.inf
        self.fams: List[str] = []
        self.e_min = _NAN


class _StoreState:
    def __init__(self) -> None:
        self.last_run: Optional[float] = None
        self.synced: Optional[float] = None
        self.live: Dict[str, _Live] = {}
        self.by_key: Dict[Tuple[str, str], str] = {}
        self.seen_ev: Dict[str, float] = {}
        self.seen_m: Dict[Tuple[str, str, float, str], float] = {}
        self.n_labels = 0
        self.risk_mem: Dict[Tuple[str, str], _RiskMem] = {}
        self.key_last: Dict[Tuple[str, str], float] = {}
        self.b_ent: Dict[Tuple[str, str], TokenBucket] = {}
        self.b_sys: Dict[str, TokenBucket] = {}
        self.pending: Dict[str, Dict[str, List[Any]]] = {}     # s -> {inc id: [state, t_q]}
        # (s, e, signature) -> [first_ts, n_ticks, last_ts]: lib/stages habit rule
        self.habits: Dict[Tuple[str, str, str], List[float]] = {}


class _Trig:
    __slots__ = ("alarm", "findings", "risk", "matches", "habitual")

    def __init__(self) -> None:
        self.alarm: Optional[Dict[str, Any]] = None
        self.findings: List[BehaviorEvent] = []
        self.risk: Optional[Dict[str, Any]] = None
        self.matches: List[Any] = []
        self.habitual: List[bool] = []           # aligned with matches


# ====================================================================== engine
class IncidentEngine(Engine):
    _canon = False
    name = "behavior.incident"
    layer = "behavior"
    consumes = ["behavior.alarm", "behavior.p_family", "behavior.e_day", "behavior.q_inst",
                "behavior.risk", "behavior.acc_alarm", "behavior.p", "behavior.regime",
                "behavior.z", "store.events", "store.matches", "store.labels",
                "model.feedback", "model.link", "model.class", "model.cp"]
    produces = ["store.incidents", "event.incident"]
    description = ("Entity and class incidents: open on alarms / findings >= MEDIUM / "
                   "risk with fresh family evidence, join, escalate, close on regime, "
                   "labels or quiet (never on risk); common-mode parenting, campaigns, "
                   "feedback suppression and token-bucket notifications.")
    interval = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._states: "weakref.WeakKeyDictionary[Any, _StoreState]" = \
            weakref.WeakKeyDictionary()

    # ------------------------------------------------------------------ run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        now = float(ctx.now)
        dt = float(ctx.window_s)
        self._canon = GR.canonical(ctx.config)
        _CANON[0] = self._canon
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError(f"incident: ctx.window_s={ctx.window_s!r} is not a positive cadence")
        training = bool(ctx.training)
        st = self._states.get(store)
        if st is None or (st.last_run is not None and now < st.last_run):
            st = self._states[store] = _StoreState()        # new store / clock went back
        lo = st.last_run if st.last_run is not None else now - dt
        budget = (ctx.config or {}).get("alert_budget") or {}
        cap_e = float(budget.get("entity_per_hour", 3))
        cap_s = float(budget.get("system_per_day", 20))
        fusion_failed = store.engine_failed(FUSION_ENGINE, now)

        out: List[Tuple[Incident, str, Dict[str, Any]]] = []     # free notifications
        tok: Dict[str, List[Tuple[str, str]]] = {}              # s -> [(inc id, state)]
        self._sync(store, st, now, training, out)
        touched: Set[str] = set()
        self._labels(store, st, now, out, touched)
        n = 0
        for s in store.systems():
            n += self._system(store, st, s, now, dt, lo, training, fusion_failed, out, tok,
                              touched)
        for inc_id in touched:
            inc = store.get_incident(inc_id)
            if inc is not None:
                store.put_incident(inc)
        if not training:
            for inc, state, extra in out:
                self._emit(store, inc, state, now, dt, extra)
                n += 1
            for s in set(tok) | set(st.pending):
                n += self._flush(store, st, s, tok.get(s, []), now, dt, cap_e, cap_s)
        self._prune(st, lo)
        st.last_run = now
        return n

    # ----------------------------------------------------------- live index
    def _sync(self, store, st: _StoreState, now: float, training: bool,
              out: List[Tuple[Incident, str, Dict[str, Any]]]) -> None:
        """Drop incidents closed or pruned elsewhere (B28 closes on ACCEPT,
        the API acks); hourly, pick up live incidents this instance does not
        know (a restart on an existing store)."""
        for iid in list(st.live):
            inc = store.get_incident(iid)
            lv = st.live[iid]
            if inc is None:
                self._forget(st, iid)
            elif inc.status == "closed":
                if lv.announced and not training:
                    out.append((inc, "close", {"close_reason": inc.close_reason,
                                               "closed_by": "external"}))
                self._forget(st, iid)
        if st.synced is None or now - st.synced >= RESYNC_S:
            st.synced = now
            for inc in store.incidents(status=LIVE):
                if inc.id not in st.live:
                    self._track(st, inc, now, announced=not (inc.status == "suppressed"))

    def _track(self, st: _StoreState, inc: Incident, now: float, announced: bool) -> _Live:
        lv = st.live[inc.id] = _Live(inc, now, announced)
        st.by_key[(inc.system, inc.entity)] = inc.id
        for e in inc.entities or ():
            st.by_key.setdefault((inc.system, e), inc.id)
        return lv

    def _forget(self, st: _StoreState, iid: str) -> None:
        lv = st.live.pop(iid, None)
        if lv is None:
            return
        for k in [k for k, v in st.by_key.items() if v == iid and k[0] == lv.s]:
            del st.by_key[k]
        pend = st.pending.get(lv.s)
        if pend:
            pend.pop(iid, None)

    def _live_for(self, store, st: _StoreState, s: str, key: str) -> Optional[Incident]:
        iid = st.by_key.get((s, key))
        if iid is None:
            return None
        inc = store.get_incident(iid)
        if inc is None or inc.status not in LIVE:
            self._forget(st, iid)
            return None
        return inc

    # --------------------------------------------------------------- labels
    def _labels(self, store, st: _StoreState, now: float,
                out: List[Tuple[Incident, str, Dict[str, Any]]], touched: Set[str]) -> None:
        """(c) close on a label. Labels are never pruned and are appended, so
        the new ones are the first (len - seen) of the newest-first list."""
        allv = store.labels()
        new = allv[:max(0, len(allv) - st.n_labels)]
        st.n_labels = len(allv)
        for lb in reversed(new):
            if lb.verdict == "unsure":
                continue
            targets: List[Incident] = []
            if lb.target_type == "incident" and lb.target_id:
                inc = store.get_incident(lb.target_id)
                if inc is not None:
                    targets.append(inc)
            elif lb.target_type == "event" and lb.target_id:
                ev = store.get_event(lb.target_id)
                if ev is not None and ev.incident_id:
                    inc = store.get_incident(ev.incident_id)
                    if inc is not None:
                        targets.append(inc)
            elif lb.entity:
                targets.extend(i for i in store.incidents(system=lb.system or None,
                                                          entity=lb.entity, status=LIVE)
                               if i.entity == lb.entity)
            for inc in targets:
                if inc.status in LIVE:
                    self._close(store, st, inc, "labelled", now, out, touched,
                                extra={"label_id": lb.id, "verdict": lb.verdict})

    # --------------------------------------------------------------- system
    def _system(self, store, st: _StoreState, s: str, now: float, dt: float, lo: float,
                training: bool, fusion_failed: bool,
                out: List[Tuple[Incident, str, Dict[str, Any]]],
                tok: Dict[str, List[Tuple[str, str]]], touched: Set[str]) -> int:
        keys = store.entities(s) + [k for k in store.pseudo_entities(s)
                                    if k.startswith(CLASS_PREFIX)]
        trig: Dict[str, _Trig] = {}

        def T(k: str) -> _Trig:
            t = trig.get(k)
            if t is None:
                t = trig[k] = _Trig()
            return t

        # --- 1) this tick's inputs
        for k in keys:
            m = store.latest_derived(s, k, ALARM)
            if m is not None and m.ts == now and isinstance(m.value, Mapping):
                T(k).alarm = dict(m.value)
        for ev in reversed(store.events(s, since=lo, kinds=DISCRETE_KINDS, limit=5000)):
            if ev.id in st.seen_ev or ev.ts > now:
                continue
            st.seen_ev[ev.id] = ev.ts
            if ev.entity.startswith("__") or ev.incident_id:
                continue
            T(ev.entity).findings.append(ev)
        for mt in reversed(store.matches(s, since=lo, limit=5000)):
            mk = (mt.entity, mt.signature_id, float(mt.ts), s)
            if mk in st.seen_m or mt.ts > now:
                continue
            st.seen_m[mk] = mt.ts
            tr = T(mt.entity)
            tr.matches.append(mt)
            # habits are learnt from every tick, warm-up included (as in B26)
            tr.habitual.append(STG.habit_step(st.habits, (s, mt.entity, str(mt.signature_id)),
                                              float(mt.ts))
                               and LEVELS[sev_rank(mt.severity)] in STG.HABIT_SEVERITIES)
        regime_ev: Dict[str, List[BehaviorEvent]] = {}
        for ev in store.events(s, since=lo, kinds=("regime",), limit=1000):
            if ev.id in st.seen_ev:
                continue
            st.seen_ev[ev.id] = ev.ts
            regime_ev.setdefault(ev.entity, []).append(ev)
        if not training:
            for k in keys:
                t = trig.get(k)
                if (t is not None and (t.alarm is not None or any(
                        sev_rank(e.severity) >= OPEN_FINDING_RANK for e in t.findings))):
                    continue
                if self._live_for(store, st, s, k) is not None:
                    continue
                r = self._risk_trigger(store, st, s, k, now, dt)
                if r is not None:
                    T(k).risk = r

        # --- 2) common mode: member -> class key
        parent_of = self._common_mode(store, s, trig) if not training else {}

        # --- 3) apply triggers: class keys first (parents), then entities
        changed: Set[str] = set()
        ordered = sorted(trig, key=lambda k: (not is_class(k), k))
        parents: Dict[str, Incident] = {}
        for ck in sorted(set(parent_of.values())):
            p = self._parent(store, st, s, ck, trig, parent_of, now, dt, training, tok, out,
                             touched, changed)
            if p is not None:
                parents[ck] = p
        for k in ordered:
            if k in parents:
                continue
            ck = parent_of.get(k)
            self._apply(store, st, s, k, trig[k], now, dt, training, tok, out, touched,
                        changed, parent=parents.get(ck) if ck else None)

        # --- 4) lifecycle of the live incidents of this system
        for iid in [i for i, lv in st.live.items() if lv.s == s]:
            lv = st.live.get(iid)
            inc = store.get_incident(iid)
            if lv is None or inc is None or inc.status not in LIVE:
                continue
            reason = self._regime_reason(store, s, inc, lv, regime_ev.get(inc.entity, ()))
            if reason is None and not fusion_failed and self._quiet(store, s, inc, lv, now, dt):
                reason = "timeout"
            if reason is not None:
                self._close(store, st, inc, reason, now, out, touched)
                continue
            if now - float(inc.opened) > HELD_S and not lv.held_sent:
                lv.held_sent = True
                self._evidence(inc, {"ts": now, "source": "b27", "state": "label_queue",
                                     "age_d": (now - float(inc.opened)) / DAY})
                touched.add(inc.id)
                if lv.announced:
                    out.append((inc, "update", {"reason": "label_queue"}))

        # --- 5) campaigns
        if changed:
            self._campaigns(store, st, s, now, out, touched)
        return 0

    # ---------------------------------------------------------------- risk
    def _risk_trigger(self, store, st: _StoreState, s: str, k: str, now: float,
                      dt: float) -> Optional[Dict[str, Any]]:
        row = store.vec_at(s, k, RISK, now)
        if row is None or not float(row[0]) >= RISK_MEDIUM:
            return None
        ts, M = store.vec_tail(s, k, RISK, 2)
        if len(ts) < 2 or not float(M[0, 0]) >= RISK_MEDIUM:
            return None
        mem = st.risk_mem.get((s, k))
        if mem is None:
            mem = st.risk_mem[(s, k)] = _RiskMem(now - RISK_LOOKBACK_S)
        if now > mem.until:                      # fold p_family points in (until, now]
            n = int(min(2000, math.ceil((now - mem.until) / dt) + 2))
            for pt in store.derived_tail(s, k, P_FAMILY, n):
                if pt.ts <= mem.until or pt.ts > now or not isinstance(pt.value, Mapping):
                    continue
                pdt = float(pt.window_s) if pt.window_s and pt.window_s > 0 else dt
                hits = []
                e_min = _NAN
                for fam, p in pt.value.items():
                    pv = _f(p)
                    if pv == pv:
                        e = pv * DAY / pdt
                        if e <= RISK_FAM_E_DAY:
                            hits.append(str(fam))
                            e_min = e if not e_min <= e else e_min
                if hits:
                    mem.hit_ts, mem.fams, mem.e_min = pt.ts, sorted(hits), e_min
            mem.until = now
        if mem.hit_ts < now - RISK_LOOKBACK_S:
            return None
        last = st.key_last.get((s, k))
        if last is None:
            incs = [i for i in store.incidents(system=s, entity=k) if i.entity == k]
            last = st.key_last[(s, k)] = max((float(i.last_seen) for i in incs), default=-math.inf)
        if mem.hit_ts <= last:
            return None                          # that evidence already had its incident
        r = float(row[0])
        # (evaluator round 3) the RISK must be new too, not only the family hit:
        # after a quiet close the key's risk decays over days (B26 half-lives
        # 12-72 h) while a family at e_day <= 0.1 is an ordinary null event
        # (~0.1 per family and entity-day, ~1 a day over the families), so the
        # old risk + any later weak hit reopened the incident within hours
        # (pack A: 58 reopenings of 23 control incidents, median 6 h after
        # the close; pack B: the sanctioned health checker 10.40.9.9 reopened
        # ~40 times on risk alone). Only the risk the key's last incident did
        # not already cover counts: the old part is bounded above by the risk
        # at that time decayed with the slowest half-life.
        if math.isfinite(last) and self._uncovered_risk(store, s, k, r, last, now) < RISK_MEDIUM:
            return None
        axes: Set[str] = set()
        for fam in mem.fams:
            axes.update(FAMILY_DEFAULT_AXES.get(fam, ()))
        return {"risk": r, "severity": "medium" if r >= RISK_HIGH else "low",
                "families": list(mem.fams), "axes": sorted(axes), "e_day": mem.e_min,
                "hit_ts": mem.hit_ts}

    @staticmethod
    def _uncovered_risk(store, s: str, k: str, r_now: float, last: float, now: float) -> float:
        """Risk not explained by the risk the key had at its last incident
        activity: B26's risk is 100 (1 - exp(-x)) with x additive in the
        evidence, so the old part at now is at most x_last 2^(-(now-last)/H)
        with H the slowest half-life."""
        ts, M = store.vec_since(s, k, RISK, last - RISK_LOOKBACK_S)
        r_old = _NAN
        for t, rw in zip(ts, M):
            if float(t) <= last + 1e-6 and float(rw[0]) == float(rw[0]):
                r_old = float(rw[0])
        if not r_old > 0.0:
            return r_now

        def x_of(r: float) -> float:
            return -math.log(max(1e-12, 1.0 - min(max(r, 0.0), 99.999) / 100.0))
        x_old = x_of(r_old) * 2.0 ** (-max(0.0, now - last) / RISK_OLD_HL_S)
        return 100.0 * (1.0 - math.exp(-max(0.0, x_of(r_now) - x_old)))

    # --------------------------------------------------------- common mode
    @staticmethod
    def _common_only(alarm: Optional[Mapping[str, Any]]) -> bool:
        if not alarm:
            return False
        ax = canonical_axes(alarm.get("axes"))
        return bool(ax) and ax <= COMMON_AXES

    def _common_mode(self, store, s: str, trig: Mapping[str, _Trig]) -> Dict[str, str]:
        """{member: class key} for members whose common-only alarm is part of
        a >= 50 % same-tick class alarm (role classes preferred)."""
        cand = {k for k, t in trig.items()
                if not is_class(k) and self._common_only(t.alarm)}
        if len(cand) < COMMON_MIN_ALARMS:
            return {}
        cks = set(m_class.all_class_keys(store, s))
        cks.update(k for k in store.pseudo_entities(s) if k.startswith(CLASS_PREFIX))
        out: Dict[str, str] = {}
        for ck in sorted(cks, key=lambda c: (class_kind(c) != "role", c)):
            members = m_class.class_members(store, s, ck)
            if len(members) < COMMON_MIN_MEMBERS:
                continue
            hit = [m for m in members if m in cand]
            if len(hit) >= max(COMMON_MIN_ALARMS, math.ceil(COMMON_FRAC * len(members))):
                for m in hit:
                    out.setdefault(m, ck)
        return out

    def _parent(self, store, st: _StoreState, s: str, ck: str, trig: Mapping[str, _Trig],
                parent_of: Mapping[str, str], now: float, dt: float, training: bool,
                tok: Dict[str, List[Tuple[str, str]]],
                out: List[Tuple[Incident, str, Dict[str, Any]]], touched: Set[str],
                changed: Set[str]) -> Optional[Incident]:
        """The class-key incident parenting this tick's common-mode members
        (the class key's own triggers are folded in the same call); joined,
        reopened or created at LOW (a coherent common-mode class change is
        capped at LOW, architecture section 5)."""
        members = sorted(m for m, c in parent_of.items() if c == ck)
        axes: Set[str] = set()
        for m in members:
            axes |= canonical_axes((trig[m].alarm or {}).get("axes"))
        force = {"kind": "coherent_shift", "severity": "low", "members": members,
                 "axes": sorted(axes), "e_day": self._e_day_at(store, s, ck, now)}
        return self._apply(store, st, s, ck, trig.get(ck) or _Trig(), now, dt, training, tok,
                           out, touched, changed, force=force)

    # --------------------------------------------------------------- apply
    def _target(self, store, st: _StoreState, s: str, k: str, now: float, dt: float,
                reopen: bool) -> Tuple[Optional[Incident], str]:
        """(incident, how) with how in {'live', 'alias', 'reopen', ''}."""
        inc = self._live_for(store, st, s, k)
        if inc is not None:
            return inc, "live"
        if not is_class(k):
            gap = max(JOIN_TICKS * dt, JOIN_S)
            for a in self._aliases(store, s, k):
                ai = self._live_for(store, st, s, a)
                if ai is not None:
                    lv = st.live.get(ai.id)
                    last = lv.last_hit if lv is not None else float(ai.last_seen)
                    if now - last <= gap:
                        return ai, "alias"
        if not reopen:
            return None, ""
        best = None
        for i in store.incidents(system=s, entity=k, status="closed", since=now - REOPEN_S):
            if i.entity != k or i.close_reason == "labelled":
                continue
            if best is None or float(i.last_seen) > float(best.last_seen):
                best = i
        if best is not None:
            return best, "reopen"
        return None, ""

    @staticmethod
    def _aliases(store, s: str, k: str) -> List[str]:
        """Continuity aliases and actor-chain members of k (model.link)."""
        link = store.get_model(s, SYSTEM_KEY, LINK_MODEL)
        if not isinstance(link, Mapping):
            return []
        out: Set[str] = set()
        links = link.get("links")
        for lk in (links.values() if isinstance(links, Mapping) else links or ()):
            if not isinstance(lk, Mapping):
                continue
            if lk.get("retracted") or lk.get("status") == "retracted" \
                    or lk.get("state") == "retracted" or lk.get("active") is False:
                continue
            a, b = lk.get("from"), lk.get("to")
            if a == k and b:
                out.add(str(b))
            elif b == k and a:
                out.add(str(a))
        actors = link.get("actors")
        for act in (actors.values() if isinstance(actors, Mapping) else actors or ()):
            mem = act.get("members") if isinstance(act, Mapping) else act
            if isinstance(mem, (list, tuple, set)) and k in mem:
                out.update(str(x) for x in mem)
        out.discard(k)
        return sorted(out)

    def _apply(self, store, st: _StoreState, s: str, k: str, t: _Trig, now: float, dt: float,
               training: bool, tok: Dict[str, List[Tuple[str, str]]],
               out: List[Tuple[Incident, str, Dict[str, Any]]], touched: Set[str],
               changed: Set[str], parent: Optional[Incident] = None,
               force: Optional[Dict[str, Any]] = None) -> Optional[Incident]:
        openers = (t.alarm is not None or t.risk is not None or force is not None
                   or any(sev_rank(e.severity) >= OPEN_FINDING_RANK and e.status != "suppressed"
                          for e in t.findings))
        inc, how = self._target(store, st, s, k, now, dt, reopen=openers and not training)
        state: Optional[str] = None
        if inc is None:
            if training or not openers:
                return None
            inc = Incident(system=s, entity=k, entities=[] if is_class(k) else [k],
                           status="open", opened=now, last_seen=now, severity=Severity.LOW)
            store.put_incident(inc)                      # assigns the id
            lv = self._track(st, inc, now, announced=False)
            self._evidence(inc, {"ts": now, "source": "b27", "state": "open"})
            state = "open"
        elif how == "reopen":
            # a fresh episode: the old parent link does not carry over (this
            # tick's common-mode test re-parents it if it still applies)
            inc.status, inc.close_reason, inc.parent_id = "open", None, ""
            lv = self._track(st, inc, now, announced=False)
            lv.supp = None
            lv.since = now
            lv.held_sent = False
            self._evidence(inc, {"ts": now, "source": "b27", "state": "reopen"})
            state = "open"
        else:
            lv = st.live.get(inc.id) or self._track(st, inc, now, announced=True)
            if how == "alias" and k not in inc.entities:
                inc.entities.append(k)
                st.by_key[(s, k)] = inc.id
                self._evidence(inc, {"ts": now, "source": "b27", "state": "alias",
                                     "entity": k})
        touched.add(inc.id)
        prev_rank = sev_rank(inc.severity) if state is None else 0
        prev_axes = set(lv.axes) if state is None else set()

        hit = self._merge(store, st, s, k, inc, lv, t, now, dt, force)
        r = store.vec_at(s, inc.entity, RISK, now)
        if r is not None and float(r[0]) == float(r[0]):
            inc.risk = float(r[0])
        if hit:
            lv.last_hit = now
            inc.last_seen = now
            st.key_last[(s, inc.entity)] = now
        changed.add(inc.id)

        # common mode: a member whose evidence is all volume / transport /
        # app-error becomes (or stays) a silent child of the class incident
        if parent is not None and parent.id != inc.id and lv.axes <= COMMON_AXES:
            if inc.parent_id != parent.id or lv.supp != "common":
                was = inc.parent_id
                inc.parent_id = parent.id
                if inc.status == "open":
                    inc.status = "suppressed"
                lv.supp = "common"
                self._evidence(inc, {"ts": now, "source": "b27", "state": "suppressed_common",
                                     "parent_id": parent.id})
                self._drop_pending(st, s, inc.id)
                self._mark_events(store, inc, lv, "suppressed")
                if lv.announced and not was:
                    out.append((inc, "update", {"parent_id": parent.id,
                                                "reason": "common_mode"}))
            plv = st.live.get(parent.id)
            if plv is not None and hit:
                plv.last_hit = now
                parent.last_seen = now
                touched.add(parent.id)
            return inc
        if lv.supp == "common":
            if lv.axes <= COMMON_AXES:
                return inc                              # a child stays silent
            inc.parent_id = ""                          # other axes: a root again
            if inc.status == "suppressed":
                inc.status = "open"
            lv.supp = None
            self._evidence(inc, {"ts": now, "source": "b27", "state": "promoted"})
            self._mark_events(store, inc, lv, "open")
            self._notify(tok, s, inc, "escalate" if lv.announced else "open")
            return inc

        # feedback suppression (and its escape)
        grew = state is not None or sev_rank(inc.severity) > prev_rank or lv.axes - prev_axes
        if grew or lv.supp == "policy":
            pol = m_feedback.suppression_match(store, inc, now) if inc.status != "acked" else None
            if pol is not None and inc.status == "open":
                inc.status = "suppressed"
                lv.supp = "policy"
                self._evidence(inc, {"ts": now, "source": "feedback", "state": "suppressed",
                                     "policy": pol.get("id"),
                                     "similarity": _f(pol.get("similarity"))})
                self._drop_pending(st, s, inc.id)
                self._mark_events(store, inc, lv, "suppressed")
                return inc
            if pol is None and lv.supp == "policy":
                inc.status = "open"
                lv.supp = None
                self._evidence(inc, {"ts": now, "source": "feedback", "state": "escaped"})
                self._mark_events(store, inc, lv, "open")
                self._notify(tok, s, inc, "escalate" if lv.announced else "open")
                return inc
            if lv.supp == "policy":
                return inc
        if state is not None:
            self._notify(tok, s, inc, state, reopened=(how == "reopen"))
        elif grew:
            self._notify(tok, s, inc, "escalate")
        return inc

    def _merge(self, store, st: _StoreState, s: str, k: str, inc: Incident, lv: _Live,
               t: _Trig, now: float, dt: float, force: Optional[Dict[str, Any]]) -> bool:
        """Fold this tick's evidence into the incident; True if it contained
        an alarm or a finding >= LOW (the quiet clock restarts)."""
        hit = False
        kinds = set(inc.kinds or ())
        sev = sev_rank(inc.severity)
        e_min = _f(inc.e_day_min)

        def e_upd(e: float) -> None:
            nonlocal e_min
            if e == e and not e_min <= e:
                e_min = e

        if force is not None:
            kinds.add(force["kind"])
            sev = max(sev, _RANK[force["severity"]])
            e_upd(_f(force.get("e_day")))
            ax = canonical_axes(force.get("axes"))
            if ax - lv.axes:
                lv.tok_dirty = True
            lv.axes |= ax
            self._evidence(inc, {"ts": now, "source": "common_mode", "state": "parent",
                                 "members": list(force.get("members") or ()),
                                 "axes": sorted(ax)})
            hit = True
        a = t.alarm
        if a is not None:
            hit = True
            kinds.add("alarm")
            sev = max(sev, max(1, sev_rank(a.get("severity"))))
            ax = canonical_axes(a.get("axes"))
            if not ax:
                for fam in a.get("families") or ():
                    ax.update(FAMILY_DEFAULT_AXES.get(str(fam), ()))
            e = _f(a.get("e_day_path"))
            if not e == e:
                e = _f(a.get("e_day"))
            if not e == e:
                e = self._e_day_at(store, s, k, now)
            pbd = self._pbd(store, s, k, now, dt, a.get("acc") or ())
            new_det = bool(set(pbd) - set(lv.pbd))
            new_ax = bool(ax - lv.axes)
            better = e == e and (not e_min == e_min or e < e_min / 10.0)
            first = lv.ev_ts == -math.inf
            if first or new_det or new_ax or better or sev > sev_rank(inc.severity) \
                    or now - lv.ev_ts >= EVIDENCE_EVERY_S:
                feats = dict(m_feedback.top_features(
                    m_feedback.z_features(store, s, k, now)))
                self._evidence(inc, {"ts": now, "source": "alarm", "path": a.get("path"),
                                     "severity": sev_name(a.get("severity")),
                                     "axes": sorted(ax), "families": list(a.get("families") or ()),
                                     "e_day": e, "p_by_detector": pbd, "features": feats,
                                     "acc": list(a.get("acc") or ())})
                lv.ev_ts = now
                if feats:
                    for n_, z in feats.items():
                        if n_ not in lv.feats or abs(z) > abs(lv.feats[n_]):
                            lv.feats[n_] = z
                    lv.tok_dirty = True
            lv.axes |= ax
            for d, p in pbd.items():
                if d not in lv.pbd or p < lv.pbd[d]:
                    lv.pbd[d] = p
            e_upd(e)
            if new_ax:
                lv.tok_dirty = True
        for ev in t.findings:
            r = sev_rank(ev.severity)
            if r >= 1:
                hit = True
            kinds.add(ev.kind)
            sev = max(sev, r)
            ax = canonical_axes(ev.axes)
            if not ax:
                fam = m_feedback.EVENT_FAMILY.get(ev.kind)
                ax = set(FAMILY_DEFAULT_AXES.get(fam, ())) if fam else set()
            e = _f(ev.e_day)
            if not e == e and r >= 1:
                e = m_feedback.SEVERITY_E_DAY.get(LEVELS[r], _NAN)
            e_upd(e)
            pbd = {str(d): _sig(float(p)) for d, p in (ev.p_by_detector or {}).items()
                   if _fin(p) is not None}
            new = m_feedback.event_new_tokens(ev)
            self._evidence(inc, {"ts": float(ev.ts), "source": "event", "kind": ev.kind,
                                 "event_id": ev.id, "severity": LEVELS[r], "axes": sorted(ax),
                                 "e_day": e, "p_by_detector": pbd, "new_tokens": new})
            if ax - lv.axes or new:
                lv.tok_dirty = True
            lv.axes |= ax
            lv.new.update(new)
            for d, p in pbd.items():
                if d not in lv.pbd or p < lv.pbd[d]:
                    lv.pbd[d] = p
            lv.event_ids.append(ev.id)
            upd: Dict[str, Any] = {"incident_id": inc.id}
            if inc.status == "suppressed" and ev.status == "open":
                upd["status"] = "suppressed"
            store.update_event(ev.id, **upd)
        if t.risk is not None:
            rk = t.risk
            hit = True
            kinds.add("risk")
            sev = max(sev, _RANK[rk["severity"]])
            e_upd(_f(rk.get("e_day")))
            ax = set(rk.get("axes") or ())
            if ax - lv.axes:
                lv.tok_dirty = True
            lv.axes |= ax
            self._evidence(inc, {"ts": now, "source": "risk", "risk": rk["risk"],
                                 "families": rk["families"], "axes": sorted(ax),
                                 "e_day": rk.get("e_day"), "family_hit_ts": rk["hit_ts"]})
        for mt, habitual in zip(t.matches, t.habitual or [False] * len(t.matches)):
            r = sev_rank(mt.severity)
            # a HABITUAL match (lib/stages: the entity's routine activity, e.g.
            # an integration host's 'high_error_backend' medium match every
            # tick) joins as evidence but does not restart the quiet clock: it
            # kept FP incidents open for days, and a threat that started
            # during that time only escalated them instead of opening its own
            if r >= OPEN_FINDING_RANK and not habitual:
                hit = True
            # a signature that matches every tick (routine health_check /
            # api_client info matches of a poller) used to add an entry per
            # tick and evicted the alarm evidence from the capped list (512
            # entries: 477 lib-4 info entries on a health checker's week-long
            # incident). An entry per signature: >= MEDIUM at most hourly,
            # below MEDIUM once per (re)opening.
            sid = str(mt.signature_id)
            last = lv.lib4.get(sid, -math.inf)
            if last >= lv.since and (r < OPEN_FINDING_RANK or now - last < EVIDENCE_EVERY_S):
                continue
            lv.lib4[sid] = now
            self._evidence(inc, {"ts": float(mt.ts), "source": "lib4",
                                 "signature_id": mt.signature_id, "category": mt.category,
                                 "severity": LEVELS[r], "confidence": _f(mt.confidence)})
        inc.kinds = sorted(kinds)
        inc.axes = sorted(lv.axes)
        inc.severity = Severity(LEVELS[max(1, sev)])
        inc.e_day_min = e_min if e_min == e_min else None
        return hit

    # ------------------------------------------------------------- readers
    @staticmethod
    def _e_day_at(store, s: str, k: str, now: float) -> float:
        row = store.vec_at(s, k, E_DAY, now)
        return float(row[0]) if row is not None else _NAN

    @staticmethod
    def _pbd(store, s: str, k: str, now: float, dt: float,
             acc: Iterable[str]) -> Dict[str, float]:
        """{detector: p} of this tick's calibrated p at <= once-a-day rarity
        (at most PBD_MAX, smallest first) plus the alarmed accumulators;
        the single smallest p when nothing is that rare."""
        row = store.vec_at(s, k, P, now)
        if row is None:
            return {}
        p = np.asarray(row, dtype=np.float64)
        ok = np.flatnonzero(np.isfinite(p))
        if not ok.size:
            return {}
        if _CANON[0]:                             # spec v2.1: each detector's own period
            thr_i = {i: PBD_E_DAY * GR.period_s(DETECTORS[i], dt, GR.CANONICAL) / DAY
                     for i in ok.tolist()}
        else:
            thr_i = {i: PBD_E_DAY * dt / DAY for i in ok.tolist()}
        sel = [i for i in ok[np.argsort(p[ok], kind="stable")] if p[i] <= thr_i[int(i)]][:PBD_MAX]
        if not sel:
            sel = [int(ok[np.argmin(p[ok])])]
        out = {DETECTORS[i]: _sig(float(p[i])) for i in sel}
        for d in acc:
            i = DETECTOR_INDEX.get(str(d))
            if i is not None and math.isfinite(p[i]):
                out[DETECTORS[i]] = _sig(float(p[i]))
        return out

    def _regime_reason(self, store, s: str, inc: Incident, lv: _Live,
                       events: Iterable[BehaviorEvent]) -> Optional[str]:
        """(a)/(b): a transition INTO returned / accepted after the incident
        (re)opened, from behavior.regime points or regime events."""
        reason = None
        prev: Optional[str] = None
        for pt in store.derived_tail(s, inc.entity, REGIME, 8):
            v = pt.value
            state = str((v.get("state", v.get("regime")) if isinstance(v, Mapping) else v)
                        or "").lower()
            if pt.ts > lv.regime_ts:
                if (state in _REGIME_CLOSE and prev != state and pt.ts >= lv.since
                        and (prev is not None or pt.ts > lv.since)):
                    reason = _REGIME_CLOSE[state]
                lv.regime_ts = pt.ts
            prev = state
        for ev in events:
            state = str((ev.extra or {}).get("state", "")).lower()
            if state in _REGIME_CLOSE and ev.ts >= lv.since:
                reason = _REGIME_CLOSE[state]
        return reason

    def _quiet(self, store, s: str, inc: Incident, lv: _Live, now: float, dt: float) -> bool:
        """(d): quiet for max(8 ticks, 2 h), accumulators < h/4, and
        e_day(q_inst) >= 1 on the last 4 rows (NaN rows are neutral)."""
        win = max(QUIET_TICKS * dt, QUIET_S)
        if now - lv.last_hit < win:
            return False
        k = inc.entity
        ts, M = store.vec_tail(s, k, Q_INST, Q_TICKS + 1)
        if len(ts):
            q = M[:, 0].astype(np.float64)
            for i in range(max(0, len(ts) - Q_TICKS), len(ts)):
                if ts[i] <= now - win:
                    # a row older than the quiet window is not current
                    # evidence: an entity that went idle (a weekend, a
                    # holiday) kept its last alarming rows as its "latest"
                    # q_inst for days, and its incident could never close
                    continue
                qi = q[i]
                if not math.isfinite(qi):
                    continue
                rdt = float(ts[i] - ts[i - 1]) if i > 0 and ts[i] > ts[i - 1] else dt
                if qi * DAY / rdt < Q_E_DAY_MIN:
                    return False
        acc = emit.read_dict(store, s, k, ACC_ALARM, now)
        if any(_f(v) >= 0.5 for v in acc.values()):
            return False
        if self._canon:
            return self._quiet_h(store, s, k, now, dt)
        # every accumulator, cusum / mcusum included, on the calibrated-p scale
        # (integration R13.2 / R14.4: m_cp.level, the raw max S/h over 48
        # charts, is >= h/4 on ~88 % of null ticks, so '< h/4' on it would
        # almost never let a quiet close happen)
        row = store.vec_at(s, k, P, now)
        if row is not None:
            for d, i in _ACC_IDX:
                if acc_level_from_p(float(row[i]), d, dt) >= ACC_QUIET_LEVEL:
                    return False
        return True

    @staticmethod
    def _quiet_h(store, s: str, k: str, now: float, dt: float) -> bool:
        """spec v2.1 (cadence.md §9.3): the latest H-stream evidence (<= 1 h
        old) must be quiet too, and every accumulator's LATEST p (H ones are
        written hourly) below the quiet level on its own period's ARL."""
        lo = now - GR.GRAIN_S["h"] - 1e-3
        ts, M = store.vec_since(s, k, Q_INST_H, lo)
        if len(ts):
            qh = float(M[-1, 0])
            if math.isfinite(qh) and qh * GR.n_per_day("h", dt, GR.CANONICAL) < Q_E_DAY_MIN:
                return False
        tsp, MP = store.vec_since(s, k, P, lo)
        if len(tsp):
            MP = np.asarray(MP, dtype=np.float64)
            for d, i in _ACC_IDX:
                col = MP[:, i]
                fin = np.flatnonzero(np.isfinite(col))
                if not fin.size:
                    continue
                per = GR.period_s(d, dt, GR.CANONICAL)
                if acc_level_from_p(float(col[fin[-1]]), d, dt, per) >= ACC_QUIET_LEVEL:
                    return False
        tail = store.derived_tail(s, k, ACC_ALARM, 1)
        if tail and tail[-1].ts >= lo and isinstance(tail[-1].value, dict) \
                and any(_f(v) >= 0.5 for v in tail[-1].value.values()):
            return False                         # the latest accumulator latch still holds
        return True

    # ------------------------------------------------------------- actions
    def _close(self, store, st: _StoreState, inc: Incident, reason: str, now: float,
               out: List[Tuple[Incident, str, Dict[str, Any]]], touched: Set[str],
               extra: Optional[Dict[str, Any]] = None) -> None:
        lv = st.live.get(inc.id) or _Live(inc, now, announced=inc.status != "suppressed")
        inc.status = "closed"
        inc.close_reason = reason
        inc.last_seen = max(float(inc.last_seen), now)
        self._evidence(inc, {"ts": now, "source": "b27", "state": "close", "reason": reason,
                             **(extra or {})})
        touched.add(inc.id)
        self._mark_events(store, inc, lv, "closed")
        st.key_last[(inc.system, inc.entity)] = inc.last_seen
        if lv.announced:
            out.append((inc, "close", {"close_reason": reason, **(extra or {})}))
        self._forget(st, inc.id)
        for cid, clv in list(st.live.items()):
            if clv.s != inc.system:
                continue
            child = store.get_incident(cid)
            if child is not None and child.parent_id == inc.id and child.status in LIVE:
                self._close(store, st, child, reason, now, out, touched,
                            extra={"parent_id": inc.id})

    @staticmethod
    def _mark_events(store, inc: Incident, lv: _Live, status: str) -> None:
        for eid in lv.event_ids:
            ev = store.get_event(eid)
            if ev is None or ev.status == status or ev.status == "acked":
                continue
            if status == "open" and ev.status != "suppressed":
                continue
            store.update_event(eid, status=status)

    @staticmethod
    def _evidence(inc: Incident, entry: Dict[str, Any]) -> None:
        ev = inc.evidence
        ev.append(_clean(entry))
        if len(ev) > MAX_EVIDENCE:
            inc.evidence = ev[:EVIDENCE_HEAD] + ev[len(ev) - (MAX_EVIDENCE - EVIDENCE_HEAD):]

    @staticmethod
    def _drop_pending(st: _StoreState, s: str, iid: str) -> None:
        pend = st.pending.get(s)
        if pend:
            pend.pop(iid, None)

    @staticmethod
    def _notify(tok: Dict[str, List[Tuple[str, str]]], s: str, inc: Incident, state: str,
                reopened: bool = False) -> None:
        tok.setdefault(s, []).append((inc.id, "reopen" if reopened else state))

    # ----------------------------------------------------------- campaigns
    def _campaigns(self, store, st: _StoreState, s: str, now: float,
                   out: List[Tuple[Incident, str, Dict[str, Any]]], touched: Set[str]) -> None:
        items: List[Tuple[str, float, frozenset]] = []
        incs: Dict[str, Incident] = {}
        for iid, lv in st.live.items():
            if lv.s != s or lv.supp == "common":
                continue
            inc = store.get_incident(iid)
            if inc is None or inc.status not in LIVE or inc.parent_id:
                continue
            incs[iid] = inc
            items.append((iid, float(inc.opened), lv.refresh_tokens()))
        if len(items) < 2:
            return
        for grp in campaign_groups(items):
            have = sorted({incs[i].campaign_id for i in grp if incs[i].campaign_id})
            cid = have[0] if have else f"cmp-{grp[0]}"
            for iid in grp:
                inc = incs[iid]
                if inc.campaign_id != cid:
                    inc.campaign_id = cid
                    touched.add(iid)
                    self._evidence(inc, {"ts": now, "source": "b27", "state": "campaign",
                                         "campaign_id": cid, "members": grp})
                    if st.live[iid].announced:
                        out.append((inc, "update", {"campaign_id": cid, "reason": "campaign"}))

    # ------------------------------------------------------- notifications
    def _flush(self, store, st: _StoreState, s: str, new: List[Tuple[str, str]], now: float,
               dt: float, cap_e: float, cap_s: float) -> int:
        """Token buckets (both must hold a token); overflow stays queued and
        is released highest-risk first as the buckets refill."""
        pend = st.pending.setdefault(s, {})
        for iid, state in new:
            cur = pend.get(iid)
            if cur is None:
                pend[iid] = [state, now]
            elif cur[0] not in ("open", "reopen"):
                cur[0] = state                       # an undelivered open stays an open
        if not pend:
            return 0
        bs = st.b_sys.get(s)
        if bs is None:
            bs = st.b_sys[s] = TokenBucket(cap_s, DAY, now)
        cand = []
        for iid, (state, tq) in list(pend.items()):
            inc = store.get_incident(iid)
            lv = st.live.get(iid)
            if inc is None or lv is None or inc.status not in ("open", "acked") or inc.parent_id:
                del pend[iid]
                continue
            r = store.vec_at(s, inc.entity, RISK, now)
            risk = float(r[0]) if r is not None and float(r[0]) == float(r[0]) else _f(inc.risk)
            risk = risk if risk == risk else 0.0
            cand.append((-risk, -sev_rank(inc.severity), tq, iid, inc, lv, state))
        cand.sort(key=lambda c: c[:4])
        n = 0
        for _, _, tq, iid, inc, lv, state in cand:
            if bs.level(now, cap_s) < 1.0:
                break
            be = st.b_ent.get((s, inc.entity))
            if be is None:
                be = st.b_ent[(s, inc.entity)] = TokenBucket(cap_e, HOUR, now)
            if be.level(now, cap_e) < 1.0:
                continue
            bs.take()
            be.take()
            del pend[iid]
            extra: Dict[str, Any] = {}
            if state == "reopen":
                state, extra["reopened"] = "open", True
            if tq < now:
                extra["queued_at"] = tq
            if state == "open":
                lv.announced = True
            elif not lv.announced:
                state = "open"
                lv.announced = True
            self._emit(store, inc, state, now, dt, extra)
            n += 1
        if len(pend) > QUEUE_MAX:
            keep = sorted(pend.items(), key=lambda kv: -_f(
                (store.get_incident(kv[0]) or Incident()).risk))[:QUEUE_MAX]
            st.pending[s] = dict(keep)
        return n

    def _emit(self, store, inc: Incident, state: str, now: float, dt: float,
              extra: Dict[str, Any]) -> None:
        lv_pbd: Dict[str, float] = {}
        for ent in inc.evidence or ():
            for d, p in (ent.get("p_by_detector") or {}).items() if isinstance(ent, Mapping) \
                    else ():
                pv = _fin(p)
                if pv is not None and (d not in lv_pbd or pv < lv_pbd[d]):
                    lv_pbd[d] = pv
        sev = sev_name(inc.severity)
        r = store.vec_at(inc.system, inc.entity, RISK, now)
        risk = _fin(r[0]) if r is not None else None       # unscored risk is None, not 0
        x = {"state": state, "incident_id": inc.id, "entities": list(inc.entities),
             "kinds": list(inc.kinds), "severity": sev, "campaign_id": inc.campaign_id,
             "parent_id": inc.parent_id, "risk": risk, "status": inc.status}
        x.update(_clean(extra))
        e = _fin(inc.e_day_min)
        ev = BehaviorEvent(
            system=inc.system, entity=inc.entity, ts=now, kind=KIND, score=_SCORE[sev],
            severity=Severity(sev),
            description=(f"incident {inc.id} {state}: {sev} on {inc.entity} "
                         f"axes={','.join(inc.axes) or '-'}"),
            extra=x, status="closed" if state == "close" else "open",
            p_value=min(lv_pbd.values()) if lv_pbd else None, e_day=e,
            axes=list(inc.axes), p_by_detector=lv_pbd,
            dedupe_key=f"incident|{inc.id}|{state}|{now:.0f}", incident_id=inc.id,
            window=(float(inc.opened), now))
        store.add_event(ev)

    # ---------------------------------------------------------------- prune
    @staticmethod
    def _prune(st: _StoreState, lo: float) -> None:
        if st.seen_ev and len(st.seen_ev) > 256:
            st.seen_ev = {k: v for k, v in st.seen_ev.items() if v >= lo}
        if st.seen_m and len(st.seen_m) > 256:
            st.seen_m = {k: v for k, v in st.seen_m.items() if v >= lo}
