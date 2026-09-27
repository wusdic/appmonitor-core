"""Read accessors and shared maths for model.client (owner: B09 ClientIdentityEngine;
contract C).

Why a module: the client stack mix of an IP (which TLS library / UA / OS /
TCP stack combinations it sends, and how much of its traffic each carries) is
evidence well beyond B09's impersonation check. B15-B18 score a window of
stacks under each identity candidate, B02 describes roles by their client mix,
B30 narrates the dominant stacks. They must use the SAME hierarchical smoothing
B09 scores with, so the maths lives here once, as pure functions over the model
dicts. Consumers never mutate a model.

Tokens are lib/stack tokens 'ja3n|ua_family/major|os|ttl_class|win_class'
(R3 client.stack_set keys).

Count unit: SLOT-EQUIVALENTS. A tick adds share_s * dt / 900 to each stack s
(share = the stack's fraction of the entity's fingerprinted requests in that
tick) times the governor's trust. N_e is therefore "15-min slots of observed
client traffic", the same at 60, 900 or 3600-s ticks, and it does not grow with
request volume: requests inside a session are not independent draws, so request
counts would make a busy host's entity tier absurdly confident and the class
backoff meaningless.

Layouts (stored by reference with put_model; objects are live):
  entity  model.client@(s, ip):
    {'fmt': 1, 'kind': 'entity', 'version': int (gate version), 'ts': float,
     'class_key': 'class:<rid>' | None,
     'state': {                                   # the gated learner state
         'H': half-life s (14 d), 'clock': newest committed row ts (counts are
         true AT the clock; at `now` multiply by 2^(-(now - clock)/H)),
         'c': {token: [c, n_ticks, first_ts, last_ts, n_gaps]},
         'N': sum of c (evicted stacks' mass stays in N),
         'gaps': {token: [active-time absence gaps in s, most recent GAP_KEEP]},
         'n_rows': int},
     'gate' / 'rows' / 'live': B09 private (GateState, row buffer, live state;
         live['last'] = {'ts', 'S', 'C', 'I', 'R', 'risk', 's0', 's1'})}
    n_ticks = committed ticks the stack was present in (undecayed); n_gaps =
    how many of its returns followed an absence (the gaps list keeps the
    newest GAP_KEEP of them). ENTITY_CAP stacks, smallest evicted.
  system  model.client@(s, '__system__'):
    {'fmt': 1, 'kind': 'system', 'version': int, 'built': ts, 'H': s,
     'c': {token: c}, 'N': float, 'n_ent': members with a model at the build,
     'classes': {'class:<rid>': {'c': {token: c}, 'N': float, 'members': int}},
     'cooc': {'ua_family/major': {ja3n: c}}, 'ua_N': {'ua_family/major': float},
     'acq': {token: {entity: ts}},    live: first sighting of a stack NEW to the
                                      entity (p_e < NEW_P), kept ROLLOUT_WINDOW_S
     'known': {entity: ts}}           live: last tick with client traffic
    Tiers (c, N, classes, cooc) are rebuilt hourly by B09 from the member
    entity models (so they inherit the members' trust gating and rollbacks):
    counts decayed to 'built'. The class tiers live inside the system model
    because contract C gives model.client no class key.

Hierarchical Dirichlet backoff (BACKOFF = priors.DIRICHLET_BACKOFF = 5,
U = 1 / UNIVERSE):
    p_s(t) = (c_s + U) / (N_s + 1)
    p_c(t) = (c_c + 5 p_s) / (N_c + 5)      (class tier skipped when absent)
    p_e(t) = (c_e + 5 p_c) / (N_e + 5)      (entity tier skipped when absent)
    Each tier is normalised over the universe of UNIVERSE tokens.
    surprisal = min(SURPRISE_CAP_BITS, -log2 p_e): below ~1e-6 the chain only
    says "never seen at any tier", and how many more bits that costs depends on
    the tiers' sizes, not on the client.

Accessor signatures (pure reads; missing data gives the documented default):
    MODEL, HALF_LIFE_S, BACKOFF, UNIVERSE, ENTITY_CAP, TIER_CAP, SURPRISE_CAP_BITS,
    NEW_P, ROLLOUT_WINDOW_S, SLOT_S
    get(store, s, key) -> dict | None              key: ip | '__system__'
    kind(model) -> 'entity' | 'system' | None
    parse(token) -> Parsed(ja3n, ua, family, major, os, ttl, win)   (memoised)
    factor(model, now=None) -> float               stored -> true count multiplier
    counts(model, now=None) -> {token: c}          entity: decayed to now; system: at 'built'
    total(model, now=None) -> float
    class_tier(sys_model, class_key) -> dict | None {'c', 'N', 'members'} (a tier view)
    Backoff(ent, cls, sys, now=None)   .tiers(t) -> (p_e, p_c, p_s); .p(t); .bits(t)
                                       .evidence() -> N_e + N_c + N_s (slot-equivalents)
    backoff_models(store, s, e) -> (ent, cls_tier, sys)   class via m_class.class_key
    prob(store, s, e, token, now=None) -> p_e
    surprisal(store, s, e, token, now=None) -> bits (capped)
    loglik(model, stack_counts, sys_model=None, cls=None, now=None) -> nats
        multinomial log-likelihood sum_t n_t ln p_e(t) of {token: n} (or a raw
        client.stack_set {token: {'n': ...}}) under an ENTITY model with backoff;
        model None scores under (cls, sys). Non-finite / <= 0 counts and
        '__other__' are skipped; nothing observed gives 0.0.
    loglik_store(store, s, e, stack_counts, now=None) -> nats (models resolved)
    stack_counts(stack_set) -> {token: n}         normalises client.stack_set
    shares(model, now=None) -> {token: share of the entity's committed mass}
    dominant(model, k=3, now=None) -> [(token, share)]  largest committed shares
    p99_gap(model, token) -> active-time absence P99 in s (0.0 when absences are
        rarer than 1 %, NaN with no history)
    p_ja3n_given_ua(sys_model, ja3n, ua) -> (p, N_ua)   system co-occurrence
        table; (NaN, N) when N_ua is 0 or the inputs are unusable
    rollout_share(sys_model, token, entity, now, members=None) -> (R_class, R_sys)
        share of the other class members (members given) / other known entities
        that first used `token` within ROLLOUT_WINDOW_S; NaN when too few
    recent(model) -> live['last'] (the last scored tick's S, C, I, R, risk) or {}
    descriptors(model, k=5, now=None) -> dict      portrait / profile descriptor (B30):
        {'dominant': [{token, share, ja3n, ua, os, ttl, win, first_ts, last_ts}],
         'n_stacks', 'entropy_bits', 'ua_mix', 'os_mix', 'mass', 'maturity'}
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

from . import m_class
from .classkeys import SYSTEM_KEY, is_class
from .priors import DIRICHLET_BACKOFF
from .stack import parse_stack_token

MODEL = "model.client"
FMT = 1
HALF_LIFE_S = 14 * 86400.0
BACKOFF = float(DIRICHLET_BACKOFF)
UNIVERSE = 4096                     # base measure U = 1/4096 of the system tier
ENTITY_CAP = 32                     # stacks per entity model
TIER_CAP = 4096                     # stacks per system / class tier
SURPRISE_CAP_BITS = 20.0
NEW_P = 0.05                        # p_e below this: the stack is new to the entity
ROLLOUT_WINDOW_S = 7 * 86400.0
ROLLOUT_MIN_OTHERS = 2              # fewer other members / entities -> R is NaN
SLOT_S = 900.0                      # count unit: one 15-min slot of full share
GAP_KEEP = 16
OTHER = "__other__"
P_FLOOR = 1e-300


class Parsed(NamedTuple):
    ja3n: str           # md5 hex, 'ja4:<..>' or '-'
    ua: str             # 'family/major' (the client.ua_set key)
    family: str
    major: int          # -1 when not a decimal
    os: str
    ttl: str
    win: str


@lru_cache(maxsize=8192)
def parse(token: str) -> Parsed:
    p = parse_stack_token(token)
    try:
        major = int(p["ua_major"])
    except (TypeError, ValueError):
        major = -1
    return Parsed(p["ja3n"], f"{p['ua_family']}/{p['ua_major']}", p["ua_family"], major,
                  p["os"], p["ttl_class"], p["win_class"])


# ------------------------------------------------------------------ basics
def get(store, system: str, key: str) -> Optional[Dict[str, Any]]:
    m = store.get_model(system, key, MODEL, default=None)
    return m if isinstance(m, dict) else None


def kind(model: Optional[Mapping]) -> Optional[str]:
    return model.get("kind") if isinstance(model, dict) else None


def _table(model: Optional[Mapping]) -> Optional[Mapping]:
    """The count-holding dict: model['state'] for entities, the model itself
    for the system tier and for class-tier views ({'c', 'N'})."""
    if not isinstance(model, dict):
        return None
    if model.get("kind") == "entity":
        st = model.get("state")
        return st if isinstance(st, dict) else None
    return model if isinstance(model.get("c"), dict) else None


def _decay(dt: float, H: float) -> float:
    return 2.0 ** (-dt / H) if dt > 0.0 else 1.0


def factor(model: Optional[Mapping], now: Optional[float] = None) -> float:
    """Multiplier from stored counts to true counts at `now` (entity: from its
    clock; system: from 'built'; a class-tier view carries its parent's 'built')."""
    st = _table(model)
    if st is None or now is None:
        return 1.0
    ref = st.get("clock") if model.get("kind") == "entity" else model.get("built")
    try:
        ref = float(ref)
        now = float(now)
    except (TypeError, ValueError):
        return 1.0
    if not (math.isfinite(ref) and math.isfinite(now)):
        return 1.0
    return _decay(now - ref, float(st.get("H", model.get("H", HALF_LIFE_S)) or HALF_LIFE_S))


def _count(x: Any) -> float:
    """Stored count of an entry: entity entries are lists, tier entries floats."""
    return float(x[0]) if isinstance(x, list) else float(x)


def counts(model: Optional[Mapping], now: Optional[float] = None) -> Dict[str, float]:
    st = _table(model)
    if st is None:
        return {}
    f = factor(model, now)
    return {t: _count(x) * f for t, x in st["c"].items()}


def total(model: Optional[Mapping], now: Optional[float] = None) -> float:
    st = _table(model)
    if st is None:
        return 0.0
    return float(st.get("N", 0.0)) * factor(model, now)


def class_tier(sys_model: Optional[Mapping], class_key: Optional[str]) -> Optional[Dict]:
    """A class tier of the system model as a stand-alone tier view
    ({'kind': 'class', 'c', 'N', 'members', 'built', 'H'}), or None."""
    if not class_key or not isinstance(sys_model, dict):
        return None
    t = (sys_model.get("classes") or {}).get(class_key)
    if not isinstance(t, dict) or not isinstance(t.get("c"), dict):
        return None
    return {"kind": "class", "c": t["c"], "N": float(t.get("N", 0.0)),
            "members": int(t.get("members", 0)), "built": sys_model.get("built"),
            "H": sys_model.get("H", HALF_LIFE_S)}


# ---------------------------------------------------------------- backoff
class Backoff:
    """Hierarchical Dirichlet predictive entity -> class -> system at `now`.
    Tables and totals are resolved once per instance, so scoring the handful
    of stacks of a tick costs a few dict lookups each."""

    __slots__ = ("_e", "_c", "_s", "k", "U")

    def __init__(self, ent: Optional[Mapping], cls: Optional[Mapping], sys: Optional[Mapping],
                 now: Optional[float] = None, k: float = BACKOFF) -> None:
        self._e = self._resolve(ent, now)
        self._c = self._resolve(cls, now)
        self._s = self._resolve(sys, now)
        self.k = float(k)
        self.U = 1.0 / UNIVERSE

    @staticmethod
    def _resolve(m: Optional[Mapping], now: Optional[float]) -> Optional[tuple]:
        st = _table(m)
        if st is None:
            return None
        f = factor(m, now)
        return st["c"], f, max(0.0, float(st.get("N", 0.0)) * f)

    def evidence(self) -> float:
        return sum(t[2] for t in (self._e, self._c, self._s) if t is not None)

    def tiers(self, token: str) -> Tuple[float, float, float]:
        k, s, c, e = self.k, self._s, self._c, self._e
        if s is None:
            p_s = self.U
        else:
            x = s[0].get(token)
            p_s = ((_count(x) * s[1] if x is not None else 0.0) + self.U) / (s[2] + 1.0)
        if c is None:
            p_c = p_s
        else:
            x = c[0].get(token)
            p_c = ((_count(x) * c[1] if x is not None else 0.0) + k * p_s) / (c[2] + k)
        if e is None:
            p_e = p_c
        else:
            x = e[0].get(token)
            p_e = ((_count(x) * e[1] if x is not None else 0.0) + k * p_c) / (e[2] + k)
        return p_e, p_c, p_s

    def p(self, token: str) -> float:
        return self.tiers(token)[0]

    def bits(self, token: str, cap: float = SURPRISE_CAP_BITS) -> float:
        return min(cap, -math.log2(max(self.tiers(token)[0], P_FLOOR)))


def backoff_models(store, system: str, entity: str
                   ) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """(entity model, class tier view, system model) for scoring under `entity`.
    A class key scores under (class, system), '__system__' under the system."""
    sys = get(store, system, SYSTEM_KEY)
    if entity == SYSTEM_KEY:
        return None, None, sys
    if is_class(entity):
        return None, class_tier(sys, entity), sys
    ck = m_class.class_key(store, system, entity)
    return get(store, system, entity), class_tier(sys, ck), sys


def prob(store, system: str, entity: str, token: str, now: Optional[float] = None) -> float:
    return Backoff(*backoff_models(store, system, entity), now=now).p(token)


def surprisal(store, system: str, entity: str, token: str,
              now: Optional[float] = None) -> float:
    return Backoff(*backoff_models(store, system, entity), now=now).bits(token)


def stack_counts(stack_set: Any) -> Dict[str, float]:
    """{token: n} from client.stack_set ({token: {'n': ...}}) or a plain
    {token: n}; '__other__', non-finite and <= 0 counts are dropped."""
    out: Dict[str, float] = {}
    if not isinstance(stack_set, Mapping):
        return out
    for t, v in stack_set.items():
        if t == OTHER or not isinstance(t, str):
            continue
        n = v.get("n") if isinstance(v, Mapping) else v
        if isinstance(n, bool):
            continue
        try:
            n = float(n)
        except (TypeError, ValueError):
            continue
        if n > 0.0 and math.isfinite(n):
            out[t] = n
    return out


def loglik(model: Optional[Mapping], stack_counts_: Any, sys_model: Optional[Mapping] = None,
           cls: Optional[Mapping] = None, now: Optional[float] = None) -> float:
    """sum_t n_t ln p_e(t) (nats) under an entity model with backoff to the
    class tier view and the system model (see module doc)."""
    obs = stack_counts(stack_counts_)
    if not obs:
        return 0.0
    ent = model if kind(model) == "entity" else None
    b = Backoff(ent, cls, sys_model, now=now)
    return float(sum(n * math.log(max(b.p(t), P_FLOOR)) for t, n in obs.items()))


def loglik_store(store, system: str, entity: str, stack_counts_: Any,
                 now: Optional[float] = None) -> float:
    ent, cls, sys = backoff_models(store, system, entity)
    return loglik(ent, stack_counts_, sys, cls, now)


# ------------------------------------------------------------ descriptors
def shares(model: Optional[Mapping], now: Optional[float] = None) -> Dict[str, float]:
    """Share of the model's retained committed mass per stack (sums to 1)."""
    c = counts(model, now)
    tot = sum(c.values())
    return {t: v / tot for t, v in c.items()} if tot > 0.0 else {}


def dominant(model: Optional[Mapping], k: int = 3,
             now: Optional[float] = None) -> List[Tuple[str, float]]:
    sh = shares(model, now)
    return sorted(sh.items(), key=lambda kv: (-kv[1], kv[0]))[:max(0, int(k))]


def p99_gap(model: Optional[Mapping], token: str) -> float:
    """Active-time absence P99 of a stack: its returns follow a zero gap
    (present in consecutive active ticks) or a recorded absence; the P99 over
    all returns is 0 when absences are rarer than 1 %, else the matching
    upper quantile of the kept absences. NaN without history."""
    st = _table(model)
    if st is None or kind(model) != "entity":
        return math.nan
    x = st["c"].get(token)
    if x is None or x[1] <= 0:
        return math.nan
    n_ticks, n_gaps = float(x[1]), float(x[4])
    gaps = sorted((st.get("gaps") or {}).get(token) or ())
    if n_gaps <= 0.01 * n_ticks or not gaps:
        return 0.0
    q = 1.0 - 0.01 * n_ticks / n_gaps              # quantile inside the absences
    i = min(len(gaps) - 1, max(0, int(math.ceil(q * len(gaps))) - 1))
    return float(gaps[i])


def p_ja3n_given_ua(sys_model: Optional[Mapping], ja3n: str, ua: str) -> Tuple[float, float]:
    if not isinstance(sys_model, dict) or not ja3n or not ua:
        return math.nan, 0.0
    n_ua = float((sys_model.get("ua_N") or {}).get(ua, 0.0))
    if not n_ua > 0.0:
        return math.nan, 0.0
    c = float(((sys_model.get("cooc") or {}).get(ua) or {}).get(ja3n, 0.0))
    return min(1.0, c / n_ua), n_ua


def rollout_share(sys_model: Optional[Mapping], token: str, entity: str, now: float,
                  members: Optional[List[str]] = None) -> Tuple[float, float]:
    """(R_class, R_sys): share of the OTHER class members (`members`, e.g.
    m_class.class_members) and of the other entities with client traffic
    within ROLLOUT_WINDOW_S that first used `token` within ROLLOUT_WINDOW_S.
    NaN for a side with fewer than ROLLOUT_MIN_OTHERS others."""
    if not isinstance(sys_model, dict):
        return math.nan, math.nan
    lo = float(now) - ROLLOUT_WINDOW_S
    acq = (sys_model.get("acq") or {}).get(token) or {}
    known = sys_model.get("known") or {}
    adopters = {x for x, ts in acq.items() if x != entity and ts >= lo}
    r_cls = math.nan
    if members is not None:
        others = [m for m in members if m != entity and known.get(m, -math.inf) >= lo]
        if len(others) >= ROLLOUT_MIN_OTHERS:
            r_cls = sum(1 for m in others if m in adopters) / len(others)
    n_known = sum(1 for x, ts in known.items() if x != entity and ts >= lo)
    r_sys = (sum(1 for x in adopters if known.get(x, -math.inf) >= lo) / n_known
             if n_known >= ROLLOUT_MIN_OTHERS else math.nan)
    return r_cls, r_sys


def recent(model: Optional[Mapping]) -> Dict[str, Any]:
    if kind(model) != "entity":
        return {}
    return dict((model.get("live") or {}).get("last") or {})


def descriptors(model: Optional[Mapping], k: int = 5,
                now: Optional[float] = None) -> Dict[str, Any]:
    st = _table(model)
    if st is None or kind(model) != "entity":
        return {"dominant": [], "n_stacks": 0, "entropy_bits": math.nan, "ua_mix": {},
                "os_mix": {}, "mass": 0.0, "maturity": 0.0}
    sh = shares(model, now)
    dom = []
    for t, v in sorted(sh.items(), key=lambda kv: (-kv[1], kv[0]))[:k]:
        p, x = parse(t), st["c"][t]
        dom.append({"token": t, "share": v, "ja3n": p.ja3n, "ua": p.ua, "os": p.os,
                    "ttl": p.ttl, "win": p.win, "first_ts": float(x[2]),
                    "last_ts": float(x[3])})
    ua_mix: Dict[str, float] = {}
    os_mix: Dict[str, float] = {}
    for t, v in sh.items():
        p = parse(t)
        ua_mix[p.ua] = ua_mix.get(p.ua, 0.0) + v
        os_mix[p.os] = os_mix.get(p.os, 0.0) + v
    ent = -sum(v * math.log2(v) for v in sh.values() if v > 0.0) if sh else math.nan
    mass = total(model, now)
    return {"dominant": dom, "n_stacks": len(sh), "entropy_bits": ent, "ua_mix": ua_mix,
            "os_mix": os_mix, "mass": mass, "maturity": mass / (mass + BACKOFF)}
