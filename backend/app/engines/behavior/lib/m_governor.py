"""model.governor / model.control accessors and the legitimate-change arithmetic
(owner: B28 GovernorEngine; contract B, C, F, G, H).

Why a module: every learner obeys model.control through lib/gating, B03 needs
"was the entity in a non-NORMAL regime near ts" for its reference anchor, B24
stratifies identity rings by regime, B27 closes incidents on RETURNED /
ACCEPTED, B29 / B30 / the UI explain why a change was (not) accepted. They
all read the governor's outputs through these functions, so the layout of
model.governor can evolve in one place. The log-odds arithmetic lives here
as pure functions so B29 and the tests reproduce the engine's verdict
exactly. No consumer ever mutates model.governor or model.control.

States (lower-case strings everywhere: model, behavior.regime, events):
    normal -> suspect -> drifting -> {returned, accepted, rejected}
    returned / accepted are one-tick states (the next tick is normal);
    rejected lasts until the entity is quiet again (no permanent lockout).
Types: intensity, shape, categorical, rhythm, ramp, identity, new_entity,
plus c2 and exfil (engines.md B28 prior "c2/exfil -4"; contract F lists
only the first seven, see the engine docstring).

Legit log-odds (engines.md B28):
    x = prior(type) + sum of ln LR terms
    prior: intensity 0, rhythm 0, ramp +0.5 (only |Sen slope| <= 0.05
           log-units/day with stationary residuals around the trend),
           shape -1, categorical -1.5, new_entity -1, identity -3, c2/exfil -4
    terms: peer +2.08, lib4 -2.3, sys_novelty -1.6, id_self +0.69,
           id_mismatch -2.3, dispersion +0.69,
           time +ln 2 per T_type/3 of stationary clean duration, cap ln 8
           (T_type 1 d intensity / ramp / new_entity, 3 d rhythm, 7 d shape /
           categorical; identity, c2 and exfil never accrue time evidence)
    P = 1 / (1 + exp(-x))
decide(): ACCEPT when P >= 0.9, duration >= T_type and no malicious-type
evidence (a ramp steeper than 0.05 log-units/day never); new_entity also
accepts after 3 d with neither malicious evidence nor a negative term (the
cold-start path cannot deadlock). REJECT when P <= 0.2 and there is evidence
beyond the prior (a malicious-type flag or a negative term): the type prior
alone only holds the entity in DRIFTING, it never discards data.

Layout of model.governor@(s, e | class:<id>) (a dict, stored by reference):
    {
      "fmt": 1,
      "regime": state, "since": ts the state was entered,
      "onset": tau-hat | None, "type": type | None,
      "logodds": float | None, "p_legit": float | None,
      "evidence": {term: ln LR},             # the terms behind logodds
      "history": [{state, ts, onset, type, p_legit, reason}],   # last 32
      "episodes": [{onset, start, end, state, type, direction}], # closed, last 16
      "episode": {...} | None,               # the open episode (engine working state)
      "version": int, "branch": int,         # champion version / challenger branch
      "rollbacks": int, "last_rollback_ts": ts | None, "last_rollback_to": ts | None,
      "label_queue": {ts, since} | None,     # DRIFTING > 14 d, waiting for a label
      "class_accept": {ts, onset, direction} | None,   # class keys only
      ...                                    # private engine bookkeeping
    }
model.control@(s, e | class:<id>): {version, branch, rebase_from, rollback_to,
release [t0, t1], frozen, allow_drift, accepted_class_change} (contract C, H).
Series: behavior.trust / trust_prov / quarantine are 1-element float32 vec
rings (every tick, every key); behavior.regime is a dict derived series
{state, type, onset, p_legit, logodds, version, branch, since} written on
every non-normal tick, on every transition and at least hourly while normal,
so the latest point at or before ts is the state at ts.

Accessors (all read-only; absent model -> neutral defaults):
    get(store, s, e) -> dict | None
    regime(store, s, e) -> str                         # 'normal' when unknown
    state(store, s, e) -> {regime, since, onset, type, p_legit, logodds, version, branch}
    control(store, s, e) -> dict                       # gating.control_directives form
    version(store, s, e) -> int
    is_quarantined(store, s, e, at=None) -> bool       # latest defined value <= at
    trust(store, s, e, at=None) -> float               # NaN when none / degraded
    trust_prov(store, s, e, at=None) -> float
    regime_at(store, s, e, ts) -> str                  # from the transition history
    episodes(store, s, e, since=None) -> [{onset, start, end, state, type}]  # open one: end None
    in_regime_window(store, s, e, ts, window_s=86400) -> bool   # B03 reference admission
    label_queue(store, system=None) -> [{system, entity, since, type, p_legit, ts}]
    descriptor(store, s, e) -> dict                    # portrait / UI summary
Pure arithmetic:
    t_type(type) -> seconds; prior(type, ramp_ok=False) -> float
    time_evidence(type, steps) -> float; logodds(type, terms, ramp_ok=False) -> float
    p_from_logodds(x) -> float
    decide(type, x, duration_s, malicious, negative, ramp_blocked=False)
        -> 'accept' | 'reject' | None
    terms_negative(terms) -> bool                      # any ln LR term (not the prior) < 0
duration_s is the time in regime (since SUSPECT). A ramp's allow_drift is in
log-units per day (B03's unit); a non-ramp ACCEPT writes allow_drift = 0.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

import numpy as np

from . import gating

MODEL = "model.governor"
CONTROL = "model.control"
REGIME = "behavior.regime"
TRUST = gating.TRUST
TRUST_PROV = gating.TRUST_PROV
QUARANTINE = gating.QUARANTINE

HOUR = 3600.0
DAY = 86400.0

# ------------------------------------------------------------------ states
NORMAL, SUSPECT, DRIFTING = "normal", "suspect", "drifting"
RETURNED, ACCEPTED, REJECTED = "returned", "accepted", "rejected"
ROLLBACK = "rollback"                      # event state only (contract F)
STATES = (NORMAL, SUSPECT, DRIFTING, RETURNED, ACCEPTED, REJECTED)
OPEN_STATES = frozenset({SUSPECT, DRIFTING})          # an episode under evaluation
QUARANTINE_STATES = frozenset({SUSPECT, DRIFTING, REJECTED})
TRUSTED_STATES = frozenset({NORMAL, RETURNED, ACCEPTED})

# ------------------------------------------------------------------- types
INTENSITY, SHAPE, CATEGORICAL, RHYTHM, RAMP = "intensity", "shape", "categorical", "rhythm", "ramp"
IDENTITY, NEW_ENTITY, C2, EXFIL = "identity", "new_entity", "c2", "exfil"
TYPES = (INTENSITY, SHAPE, CATEGORICAL, RHYTHM, RAMP, IDENTITY, NEW_ENTITY, C2, EXFIL)

PRIOR: Dict[str, float] = {INTENSITY: 0.0, RHYTHM: 0.0, RAMP: 0.5, SHAPE: -1.0,
                           CATEGORICAL: -1.5, NEW_ENTITY: -1.0, IDENTITY: -3.0,
                           C2: -4.0, EXFIL: -4.0}
T_TYPE_S: Dict[str, float] = {INTENSITY: DAY, RAMP: DAY, NEW_ENTITY: DAY, RHYTHM: 3 * DAY,
                              SHAPE: 7 * DAY, CATEGORICAL: 7 * DAY, IDENTITY: 7 * DAY,
                              C2: 7 * DAY, EXFIL: 7 * DAY}
NO_TIME_EVIDENCE = frozenset({IDENTITY, C2, EXFIL})

# ln LR terms (engines.md B28, spec values)
LR_PEER = 2.08            # >= 50 % of the class moving the same way within +-1 h
LR_LIB4 = -2.3            # lib-4 match >= HIGH
LR_SYS_NOVELTY = -1.6     # system-tier novelty (IDF > ln(N/2)) or an external upload destination
LR_ID_SELF = 0.69         # identity self-posterior >= 0.9
LR_ID_MISMATCH = -2.3     # identity mismatch / impersonation or client concurrency
LR_DISPERSION = 0.69      # post-change dispersion ratio <= 1.5
TIME_STEP = math.log(2.0)  # per T_type / 3 of stationary clean duration
TIME_STEPS_MAX = 3         # cap ln 8

P_ACCEPT = 0.9
P_REJECT = 0.2
RAMP_MAX_SLOPE = 0.05      # log-units per day
NEW_ENTITY_ACCEPT_S = 3 * DAY


# ============================================================ pure arithmetic
def t_type(typ: Optional[str]) -> float:
    """T_type in seconds (unknown type: the intensity day)."""
    return T_TYPE_S.get(str(typ), DAY)


def prior(typ: Optional[str], ramp_ok: bool = False) -> float:
    """Prior log-odds of a legitimate change of this type. The ramp's +0.5
    needs a gentle, stationary ramp (ramp_ok); otherwise a ramp counts as a
    plain intensity change (0)."""
    if typ == RAMP:
        return PRIOR[RAMP] if ramp_ok else PRIOR[INTENSITY]
    return PRIOR.get(str(typ), PRIOR[INTENSITY])


def time_evidence(typ: Optional[str], steps: int) -> float:
    """+ln 2 per completed T_type/3 of stationary clean duration, capped at ln 8;
    identity, c2 and exfil never accrue time evidence."""
    if typ in NO_TIME_EVIDENCE:
        return 0.0
    k = max(0, min(TIME_STEPS_MAX, int(steps)))
    return k * TIME_STEP


def logodds(typ: Optional[str], terms: Mapping[str, float], ramp_ok: bool = False) -> float:
    """prior(type) + sum of the finite ln LR terms (a 'prior' key is ignored)."""
    x = prior(typ, ramp_ok)
    for k, v in terms.items():
        if k == "prior":
            continue
        v = _f(v)
        if v == v:
            x += v
    return x


def p_from_logodds(x: float) -> float:
    """Logistic P(legitimate); NaN in -> NaN out, overflow-safe."""
    x = _f(x)
    if x != x:
        return math.nan
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def decide(typ: Optional[str], x: float, duration_s: float, malicious: bool, negative: bool,
           ramp_blocked: bool = False) -> Optional[str]:
    """The verdict of one evaluation: 'accept', 'reject' or None (hold).

    accept: P >= 0.9 and duration >= T_type and no malicious-type evidence and
            not a ramp steeper than RAMP_MAX_SLOPE; a new_entity also accepts
            after NEW_ENTITY_ACCEPT_S with no malicious evidence and no
            negative term.
    reject: P <= 0.2 with malicious-type evidence or a negative term (the
            prior alone never rejects: it holds as DRIFTING for a label).
    """
    p = p_from_logodds(x)
    dur = _f(duration_s)
    if p != p or dur != dur:
        return None
    if not malicious and not ramp_blocked:
        if p >= P_ACCEPT and dur >= t_type(typ):
            return "accept"
        if typ == NEW_ENTITY and not negative and dur >= NEW_ENTITY_ACCEPT_S:
            return "accept"
    if p <= P_REJECT and (malicious or negative):
        return "reject"
    return None


# ================================================================ accessors
def get(store: Any, s: str, e: str) -> Optional[Dict[str, Any]]:
    m = store.get_model(s, e, MODEL)
    return m if isinstance(m, Mapping) else None


def regime(store: Any, s: str, e: str) -> str:
    m = get(store, s, e)
    st = str((m or {}).get("regime") or NORMAL).lower()
    return st if st in STATES else NORMAL


def state(store: Any, s: str, e: str) -> Dict[str, Any]:
    """Summary of the current regime (neutral when the governor never ran)."""
    m = get(store, s, e) or {}
    return {
        "regime": regime(store, s, e),
        "since": m.get("since"),
        "onset": m.get("onset"),
        "type": m.get("type"),
        "p_legit": m.get("p_legit"),
        "logodds": m.get("logodds"),
        "version": int(m.get("version") or 0),
        "branch": int(m.get("branch") or 0),
    }


def control(store: Any, s: str, e: str) -> Dict[str, Any]:
    """model.control normalised by gating.control_directives (None per absent key)."""
    return gating.control_directives(store.get_model(s, e, CONTROL))


def version(store: Any, s: str, e: str) -> int:
    v = control(store, s, e).get("version")
    return int(v) if v is not None else 0


def _ring_value(store: Any, s: str, e: str, name: str, at: Optional[float]) -> float:
    """Latest non-NaN value of a 1-element ring at or before `at` (newest if None)."""
    if at is None:
        hit = store.vec_latest(s, e, name)
        if hit is None:
            return math.nan
        v = float(np.asarray(hit[1]).reshape(-1)[0])
        if v == v:
            return v
        at = float(hit[0])
    row = store.vec_at(s, e, name, float(at))
    if row is not None:
        v = float(np.asarray(row).reshape(-1)[0])
        if v == v:
            return v
    t, M = store.vec_since(s, e, name, float(at) - gating.HELD_MAX_AGE_S)
    if not len(t):
        return math.nan
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(M, dtype=np.float64).reshape(len(t), -1)[:, 0]
    ok = np.flatnonzero((t <= float(at)) & ~np.isnan(v))
    return float(v[ok[-1]]) if ok.size else math.nan


def is_quarantined(store: Any, s: str, e: str, at: Optional[float] = None) -> bool:
    """behavior.quarantine at or before `at` (latest defined value; none -> False)."""
    v = _ring_value(store, s, e, QUARANTINE, at)
    return bool(v == v and v > 0.5)


def trust(store: Any, s: str, e: str, at: Optional[float] = None) -> float:
    """behavior.trust at exactly `at` (latest row if None); NaN when absent or degraded."""
    return _exact(store, s, e, TRUST, at)


def trust_prov(store: Any, s: str, e: str, at: Optional[float] = None) -> float:
    return _exact(store, s, e, TRUST_PROV, at)


def _exact(store: Any, s: str, e: str, name: str, at: Optional[float]) -> float:
    if at is None:
        hit = store.vec_latest(s, e, name)
        return math.nan if hit is None else float(np.asarray(hit[1]).reshape(-1)[0])
    row = store.vec_at(s, e, name, float(at))
    return math.nan if row is None else float(np.asarray(row).reshape(-1)[0])


def regime_at(store: Any, s: str, e: str, ts: float) -> str:
    """The regime in force at ts: the state of the last transition at or
    before ts (the model keeps the last 32 transitions, including the return
    to normal the tick after returned / accepted; older ts read 'normal')."""
    m = get(store, s, e)
    if not m:
        return NORMAL
    cur = NORMAL
    for h in m.get("history") or []:
        t = _f(h.get("ts"))
        if t == t and t <= float(ts):
            cur = str(h.get("state") or NORMAL)
        elif t == t:
            break
    return cur


def episodes(store: Any, s: str, e: str, since: Optional[float] = None) -> List[Dict[str, Any]]:
    """Closed episodes plus the open one (end None), oldest first, whose end
    (or now, when open) is >= since."""
    m = get(store, s, e)
    if not m:
        return []
    out = [dict(ep) for ep in (m.get("episodes") or [])]
    cur = m.get("episode")
    if isinstance(cur, Mapping):
        out.append({"onset": cur.get("onset"), "start": cur.get("start"), "end": None,
                     "state": m.get("regime"), "type": m.get("type")})
    if since is not None:
        out = [ep for ep in out if ep.get("end") is None or _f(ep.get("end")) >= since]
    return out


def in_regime_window(store: Any, s: str, e: str, ts: float, window_s: float = DAY) -> bool:
    """True when a non-normal episode (from its onset to its end) lies within
    +-window_s of ts: B03's reference anchor admits a row only if not."""
    t = float(ts)
    for ep in episodes(store, s, e, since=t - window_s):
        a = _f(ep.get("onset"))
        if a != a:
            a = _f(ep.get("start"))
        b = _f(ep.get("end"))
        if b != b:
            b = math.inf
        if a == a and a - window_s <= t <= b + window_s:
            return True
    return False


def label_queue(store: Any, system: Optional[str] = None) -> List[Dict[str, Any]]:
    """Keys held DRIFTING for more than 14 d that need an analyst label."""
    out: List[Dict[str, Any]] = []
    systems = [system] if system is not None else store.systems()
    for s in systems:
        keys = list(store.entities(s)) + [k for k in store.pseudo_entities(s)
                                          if k.startswith("class:")]
        for e in keys:
            m = get(store, s, e)
            lq = (m or {}).get("label_queue")
            if isinstance(lq, Mapping) and (m or {}).get("regime") in OPEN_STATES:
                out.append({"system": s, "entity": e, "since": lq.get("since"),
                            "ts": lq.get("ts"), "type": m.get("type"),
                            "p_legit": m.get("p_legit")})
    out.sort(key=lambda r: (_f(r.get("since")), r["system"], r["entity"]))
    return out


def descriptor(store: Any, s: str, e: str) -> Dict[str, Any]:
    """Portrait / UI summary (JSON-safe): the state, the legit evidence and
    the recent history."""
    m = get(store, s, e)
    if not m:
        return {"state": NORMAL, "version": 0, "branch": 0}
    return {
        "state": regime(store, s, e),
        "since": m.get("since"),
        "onset": m.get("onset"),
        "type": m.get("type"),
        "p_legit": m.get("p_legit"),
        "logodds": m.get("logodds"),
        "evidence": dict(m.get("evidence") or {}),
        "version": int(m.get("version") or 0),
        "branch": int(m.get("branch") or 0),
        "rollbacks": int(m.get("rollbacks") or 0),
        "history": [dict(h) for h in (m.get("history") or [])[-10:]],
        "label_queue": bool(m.get("label_queue")),
    }


def _f(x: Any) -> float:
    if x is None:
        return math.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def terms_negative(terms: Mapping[str, float]) -> bool:
    """True when any ln LR term (not the prior) is negative."""
    return any(_f(v) < 0.0 for k, v in terms.items() if k != "prior")
