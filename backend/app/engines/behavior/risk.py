"""B26 RiskEngine: one explainable, decaying 0-100 risk per entity, per class
and per system (docs/lib3/engines.md B26).

Why a separate score next to fusion's alarms: an alarm is a per-tick decision
on a false-alarm budget, so an attack that stays below every alarm on every
tick (a JA3 variant, one rare path, a +0.7 sigma shift, one off-hours slot)
never alarms. Risk integrates that weak, heterogeneous evidence over hours to
days, and it is what the UI ranks entities by.

Evidence per tick (all in the same "surprise" unit so they add up):
  * continuous: b_f = W_f * max(0, log10(1 / e_day(p_family))), the excess
    surprise over what the null produces once per day. Under the null
    E[b_f] per DAY is W_f / ln 10 whatever the cadence (the per-tick excess
    of an Exp(1) variable over log(86400/dt) has mean dt/86400), so the same
    L_ref serves 60 s and 900 s ticks. Common-mode-flagged volume
    (behavior.common.flag['volume'], B05) counts x 0.3.
  * episode saturation: the n-th consecutive contributing tick of the same
    (key, family) counts x 0.5^((n-1)/4), so a sustained episode adds at most
    1/(1 - 0.5^0.25) = 6.3 x b. A tick with a valid, unremarkable p ends the
    streak unless an incident of the key is still open (the incident is the
    episode); a NaN (unscored) family neither contributes nor ends it.
  * discrete findings (store.events): fixed weights by kind (first_seen by
    tier), the n-th repeat of one key within 24 h counts x 0.5^(n-1), so any
    number of repeats adds < 2 x a single one.
  * lib-4 matches with a one-tick lag (the signature layer runs after the
    behaviour layer): info 0 / low 5 / medium 15 / high 30 / critical 50,
    x confidence, damped per signature id like events. A HABITUAL match
    (integration): a signature of severity <= medium that the entity has
    matched on >= HABIT_MIN_TICKS ticks, the first of them >= HABIT_S (24 h)
    ago, counts x 0. lib-4 severities grade activities (a routine login,
    form write, admin page or a poor-TCP link is 'low', a noisy backend
    'medium'), so without this every busy entity carried L ~ 30-50 of
    routine matches (risk 40-60, and the auth / admin / transfer categories
    added kill-chain stages) on every live tick. A NEW routine activity
    still counts for its first day; high / critical always count. The habit
    memory is learnt from every tick, warm-up included, and is not cleared
    with the evidence on the first live tick.
  * everything that belongs to a suppressed incident (or a suppressed event)
    counts x 0.25 (m_feedback.SUPPRESSED_RISK_WEIGHT): feedback may silence
    notifications, never evidence.
Decay is wall-clock, per component: L <- L * 2^(-elapsed/H) with H = 12 h
(behaviour), 24 h (change, sequence), 72 h (novelty, identity), 48 h (c2,
exfil, and anything from a CRITICAL alarm or finding).
Risk = 100 * (1 - exp(-M * c_s * sum_k(pi_k * L_k) / L_ref)) with
  M = min(2.2, 1 + 0.3 * (#kill-chain stages in 24 h - 1)) (lib/stages; a
      family earns its stage only at e_day <= 0.03 so the null keeps M ~ 1),
  c_s in [0.5, 2] the criticality (ctx.config criticality / ip_classes),
  pi_k the feedback multiplier of the component's family (m_feedback),
  L_ref = 60 FIXED: tuned offline on clean replays, never from live data, so
      an attacker cannot raise the bar by being noisy (null mean L ~ 7 gives
      risk ~ 11; the null p99.99 of L ~ 17-19 gives <= 27).
Tiers: low < 30 <= medium < 60 <= high < 85 <= critical; a tier is left
downwards only 10 points below its entry threshold (hysteresis).
Class risk = max(the class key's own risk from its own detectors and
findings, mean of the top-3 member risks). System risk = max over the
system's entities and classes.

State (decayed components, streaks, repeat history, stage times, tier) lives
in the engine instance per store (a WeakKeyDictionary: a new store, i.e. a
new run, starts clean; a clock that goes backwards resets the key). Nothing
is learned from live data, so there is nothing to gate or roll back.
Everything written at ctx.now; ctx.window_s is the real dt of the tick
(e_day), the elapsed wall time since the key's last update drives decay.

Store: reads behavior.p_family, behavior.axes, behavior.alarm,
behavior.common.flag (dicts at now), store.events / store.matches /
store.incidents, model.feedback (m_feedback), model.class (m_class),
store health of behavior.fusion (contract M: a fusion failure at this tick
gives NaN risk); writes behavior.risk (1-element float32 vec ring) at each
entity, class:<id> and '__system__', and profile.extra.risk
{score, tier, trend, top_reasons, stages, ...}.
"""
from __future__ import annotations

import ipaddress
import math
import weakref
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from ...core.engine import Context, Engine
from ...models.schema import EntityProfile
from .lib import emit, m_class, m_feedback
from .lib.classkeys import CLASS_PREFIX, STATIC_PREFIX, SYSTEM_KEY
from .lib.detectors import DETECTOR_INFO, FAMILIES, FAMILY_DEFAULT_AXES
from .lib.stages import FLAG_NAMES, stage_for_axis, stage_for_category, stage_for_event

RISK = "behavior.risk"
P_FAMILY = "behavior.p_family"
ALARM = "behavior.alarm"
COMMON_FLAG = "behavior.common.flag"
FUSION_ENGINE = "behavior.fusion"

HOUR = 3600.0
DAY = 86400.0
L_REF = 60.0                    # fixed (decisions.md): never set from live data
SAT_TICKS = 4.0                 # episode saturation 0.5^((n-1)/4)
M_STEP, M_MAX = 0.3, 2.2        # stage multiplier
STAGE_WINDOW_S = DAY
REPEAT_WINDOW_S = DAY
STAGE_E_DAY = 0.03              # a family earns its stage at the LOW level
COMMON_VOLUME_MULT = 0.3
SUPPRESSED_MULT = m_feedback.SUPPRESSED_RISK_WEIGHT
ADOPTED_MULT = 0.1              # B08 adoption discount (first_seen marked adopted)
CRIT_RANGE = (0.5, 2.0)
TIER_NAMES = ("low", "medium", "high", "critical")
TIER_THRESHOLDS = (30.0, 60.0, 85.0)     # entry into medium / high / critical
TIER_HYSTERESIS = 10.0
TREND_WINDOW_S = HOUR
TOP_REASONS = 5
L_DROP = 1e-4                   # components below this are forgotten
E_DAY_FLOOR = 1e-300

# Continuous evidence weights W_f. xsys (P2, lateral) is not in the spec's
# table; it gets the weight of the other breadth-like families.
FAMILY_W: Dict[str, float] = {
    "intensity": 1.0, "shape": 1.5, "categorical": 1.5, "identity": 2.0, "c2": 2.0,
    "exfil": 2.0, "breadth": 1.5, "change": 1.5, "temporal": 1.0, "sequence": 1.5,
    "peer": 1.0, "xsys": 1.5,
}

HALF_LIFE_S: Dict[str, float] = {
    "behaviour": 12 * HOUR, "change": 24 * HOUR, "sequence": 24 * HOUR,
    "novelty": 72 * HOUR, "identity": 72 * HOUR,
    "c2": 48 * HOUR, "exfil": 48 * HOUR, "critical": 48 * HOUR,
}
FAMILY_DECAY: Dict[str, str] = {
    "intensity": "behaviour", "shape": "behaviour", "peer": "behaviour",
    "temporal": "behaviour", "breadth": "behaviour", "xsys": "behaviour",
    "change": "change", "sequence": "sequence",
    "categorical": "novelty", "identity": "identity",
    "c2": "c2", "exfil": "exfil",
}

# Discrete findings (contract F kinds). first_seen is weighted by tier.
FIRST_SEEN_W: Dict[str, float] = {"system": 15.0, "class": 8.0, "entity": 3.0}
EVENT_W: Dict[str, float] = {
    "rare_access": 15.0, "client_impersonation": 20.0, "identity_mismatch": 20.0,
    "possible_impersonation": 20.0, "new_entity_unmatched": 20.0, "beacon": 20.0,
    "budget_exceeded": 20.0, "class_adoption_risky": 20.0,
}
EVENT_KINDS: Tuple[str, ...] = tuple(sorted({"first_seen", *EVENT_W}))
EVENT_DECAY: Dict[str, str] = {
    "first_seen": "novelty", "rare_access": "novelty", "class_adoption_risky": "novelty",
    "client_impersonation": "identity", "identity_mismatch": "identity",
    "possible_impersonation": "identity", "new_entity_unmatched": "identity",
    "beacon": "c2", "budget_exceeded": "exfil",
}
EVENT_AXES: Dict[str, List[str]] = {
    "first_seen": ["categorical"], "rare_access": ["categorical"],
    "class_adoption_risky": ["categorical"], "budget_exceeded": ["exfil"],
    "new_entity_unmatched": ["peer"],
}
_TIER_ALIASES = {"system": "system", "org": "system", "class": "class", "role": "class",
                 "static": "class", "pool": "class", "sub": "class", "entity": "entity",
                 "ip": "entity"}

SIG_W: Dict[str, float] = {"info": 0.0, "low": 5.0, "medium": 15.0, "high": 30.0,
                           "critical": 50.0}
# habitual lib-4 activity (see module doc): discounted at these severities
HABIT_SEVERITIES = frozenset({"info", "low", "medium"})
HABIT_S = 86400.0              # first match of the (entity, signature) at least this old
HABIT_MIN_TICKS = 4            # ... and matched on at least this many ticks
HABIT_MULT = 0.0
HABIT_FORGET_S = 30 * 86400.0  # a habit not seen for 30 d is forgotten
_STAGE_DECAY = {"c2": "c2", "exfiltration": "exfil", "identity": "identity"}


def _sev(x: Any) -> str:
    v = getattr(x, "value", x)
    return str(v).lower() if v is not None else "info"


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


# ============================================================ evidence (pure)
def excess_surprise(p: float, dt_s: float) -> float:
    """log10(1/e_day(p)) clipped at 0; NaN for NaN / missing p."""
    if p is None or not p == p:
        return math.nan
    e = max(float(p), E_DAY_FLOOR) * DAY / dt_s
    return -math.log10(e) if e < 1.0 else 0.0


def saturation(n: int) -> float:
    """Weight of the n-th (1-based) consecutive contributing tick."""
    return 0.5 ** ((n - 1) / SAT_TICKS)


def stage_multiplier(n_stages: int) -> float:
    return min(M_MAX, 1.0 + M_STEP * (max(1, n_stages) - 1))


def risk_from_load(load: float, m: float = 1.0, crit: float = 1.0) -> float:
    """100 (1 - exp(-M c_s load / L_ref)); `load` = sum pi_k L_k."""
    if not load == load:
        return math.nan
    return 100.0 * (1.0 - math.exp(-m * crit * max(0.0, load) / L_REF))


def tier_of(risk: float, prev: Optional[str] = None) -> str:
    """Tier with -10 hysteresis: a tier is kept while risk >= its entry
    threshold - 10, and entered upwards at the threshold itself."""
    raw = sum(1 for t in TIER_THRESHOLDS if risk >= t)
    cur = TIER_NAMES.index(prev) if prev in TIER_NAMES else 0
    if raw >= cur:
        return TIER_NAMES[raw]
    while cur > raw and risk < TIER_THRESHOLDS[cur - 1] - TIER_HYSTERESIS:
        cur -= 1
    return TIER_NAMES[cur]


def event_weight(ev: Any) -> float:
    """Base weight of a discrete finding (before repeat damping / suppression)."""
    kind = ev.kind
    extra = ev.extra or {}
    if kind == "first_seen":
        w = FIRST_SEEN_W[_event_tier(extra)]
    else:
        w = EVENT_W.get(kind, 0.0)
    if w <= 0.0:
        return 0.0
    disc = extra.get("discount")
    if extra.get("adopted"):
        w *= ADOPTED_MULT
    elif _finite(disc) and 0.0 < float(disc) <= 1.0:
        w *= float(disc)
    return w


def _event_tier(extra: Mapping[str, Any]) -> str:
    return _TIER_ALIASES.get(str(extra.get("tier", "entity")).lower(), "entity")


def event_key(ev: Any) -> str:
    """Repeat-damping key: the producer's dedupe_key, else kind + new token."""
    if ev.dedupe_key:
        return f"{ev.kind}|{ev.dedupe_key}"
    x = ev.extra or {}
    tok = x.get("token")
    if tok is None and ("dim" in x or "value" in x):
        tok = f"{x.get('dim', '*')}={x.get('value', '')}"
    if tok is None:
        tok = x.get("key", x.get("signature_id", ""))
    return f"{ev.kind}|{tok}"


def event_stages(ev: Any) -> Set[str]:
    """Kill-chain stages of a finding: the kind itself (identity, beacon), else
    its axes refined by the context flags it carries. A system- or class-tier
    first_seen / rare access is low-prevalence by definition; rare_access is
    emitted for sensitive values."""
    st = stage_for_event(ev.kind)
    if st:
        return {st}
    extra = ev.extra or {}
    flags: Dict[str, bool] = {}
    fl = extra.get("flags")
    if isinstance(fl, Mapping):
        flags.update({k: bool(v) for k, v in fl.items() if k in FLAG_NAMES})
    flags.update({k: bool(extra[k]) for k in FLAG_NAMES if k in extra})
    if ev.kind in ("first_seen", "rare_access") and _event_tier(extra) != "entity":
        flags.setdefault("low_prevalence", True)
    if ev.kind == "rare_access":
        flags.setdefault("sensitive", True)
    axes = list(ev.axes or ()) or EVENT_AXES.get(ev.kind, [])
    out = set()
    for a in axes:
        s = stage_for_axis(a, **flags)
        if s:
            out.add(s)
    return out


def match_decay(sev: str, stage: Optional[str]) -> str:
    if sev == "critical":
        return "critical"
    return _STAGE_DECAY.get(stage or "", "behaviour")


# ================================================================ state (pure)
class RiskState:
    """Decaying evidence of one key (entity or class). Components are
    '<source>:<name>' with their own half-life; see the module docstring."""

    __slots__ = ("last_ts", "L", "H", "fam", "detail", "comp_ts", "streak", "repeats",
                 "stage_ts", "tier", "hist", "risk")

    def __init__(self) -> None:
        self.last_ts: Optional[float] = None
        self.L: Dict[str, float] = {}
        self.H: Dict[str, float] = {}
        self.fam: Dict[str, Optional[str]] = {}      # component -> feedback family
        self.detail: Dict[str, str] = {}
        self.comp_ts: Dict[str, float] = {}          # component -> last contribution
        self.streak: Dict[str, int] = {}
        self.repeats: Dict[str, List[float]] = {}
        self.stage_ts: Dict[str, float] = {}
        self.tier: str = "low"
        self.hist: Deque[Tuple[float, float]] = deque()
        self.risk: float = 0.0

    # ------------------------------------------------------------- decay
    def decay_to(self, now: float) -> None:
        """Wall-clock decay of every component up to `now` (idempotent)."""
        if self.last_ts is not None and now > self.last_ts and self.L:
            el = now - self.last_ts
            fac: Dict[float, float] = {}
            drop = []
            for k, v in self.L.items():
                h = self.H[k]
                f = fac.get(h)
                if f is None:
                    f = fac[h] = 2.0 ** (-el / h)
                v *= f
                if v < L_DROP:
                    drop.append(k)
                else:
                    self.L[k] = v
            for k in drop:
                for d in (self.L, self.H, self.fam, self.detail, self.comp_ts):
                    d.pop(k, None)
        if self.last_ts is None or now > self.last_ts:
            self.last_ts = now

    # ---------------------------------------------------------- evidence
    def add(self, comp: str, amount: float, decay: str, now: float, stages: Iterable[str] = (),
            family: Optional[str] = None, detail: str = "") -> None:
        if amount > 0.0:
            h = HALF_LIFE_S[decay]
            # a slower (e.g. critical) contribution keeps the component slow
            self.H[comp] = max(h, self.H.get(comp, 0.0)) if comp in self.L else h
            self.L[comp] = self.L.get(comp, 0.0) + amount
            self.fam[comp] = family
            self.comp_ts[comp] = now
            if detail:
                self.detail[comp] = detail
        for s in stages:
            self.stage_ts[s] = now

    def add_families(self, now: float, dt_s: float, p_family: Mapping[str, float], *,
                     episode: bool = False, volume_common: bool = False, mult: float = 1.0,
                     critical: bool = False,
                     stages_of: Optional[Any] = None) -> Dict[str, float]:
        """Continuous evidence of one tick; returns {family: added L}.
        `stages_of(family) -> set` gives the stages a family earns at the LOW
        level (default: its default axes)."""
        out: Dict[str, float] = {}
        for f, p in p_family.items():
            if f not in FAMILY_W:
                continue
            x = excess_surprise(p, dt_s)
            if not x == x:
                continue                                   # unscored: no evidence either way
            if x <= 0.0:
                if not episode:
                    self.streak.pop(f, None)
                continue
            b = FAMILY_W[f] * x
            if f == "intensity" and volume_common:
                b *= COMMON_VOLUME_MULT
            n = self.streak.get(f, 0) + 1
            self.streak[f] = n
            b *= saturation(n) * mult
            st: Iterable[str] = ()
            if x >= _STAGE_X:
                st = stages_of(f) if stages_of is not None else default_family_stages(f)
            decay = "critical" if critical and HALF_LIFE_S["critical"] > \
                HALF_LIFE_S[FAMILY_DECAY[f]] else FAMILY_DECAY[f]
            self.add(f"family:{f}", b, decay, now, st, family=f,
                     detail=f"e_day={max(float(p), E_DAY_FLOOR) * DAY / dt_s:.2g}")
            out[f] = b
        return out

    def add_repeated(self, comp: str, key: str, weight: float, decay: str, now: float,
                     ts: float, stages: Iterable[str] = (), family: Optional[str] = None,
                     detail: str = "") -> float:
        """A discrete finding: the n-th occurrence of `key` in 24 h counts
        x 0.5^(n-1). Returns the L added."""
        hist = self.repeats.get(key)
        if hist is None:
            hist = self.repeats[key] = []
        cut = now - REPEAT_WINDOW_S
        if hist and hist[0] < cut:
            hist[:] = [t for t in hist if t >= cut]
        amount = weight * 0.5 ** len(hist)
        hist.append(ts)
        self.add(comp, amount, decay, now, stages, family=family, detail=detail)
        return amount

    def prune(self, now: float) -> None:
        cut = now - REPEAT_WINDOW_S
        for k in [k for k, h in self.repeats.items() if not h or h[-1] < cut]:
            del self.repeats[k]
        cut = now - STAGE_WINDOW_S
        for s in [s for s, t in self.stage_ts.items() if t < cut]:
            del self.stage_ts[s]

    # ----------------------------------------------------------- readout
    def stages(self, now: float) -> List[str]:
        cut = now - STAGE_WINDOW_S
        return sorted(s for s, t in self.stage_ts.items() if t >= cut)

    def total(self) -> float:
        return sum(self.L.values())

    def load(self, pi: Mapping[str, float]) -> float:
        """sum_k pi_k L_k (pi of the component's family; 1 when it has none)."""
        return sum(v * pi.get(self.fam.get(k) or "", 1.0) for k, v in self.L.items())

    def score(self, now: float, pi: Mapping[str, float], crit: float) -> float:
        return risk_from_load(self.load(pi), stage_multiplier(len(self.stages(now))), crit)

    def finalize(self, now: float, risk: float) -> Tuple[str, float]:
        """Update tier (hysteresis) and the trend (change over the last hour)."""
        if not risk == risk:
            return self.tier, math.nan
        self.tier = tier_of(risk, self.tier)
        h = self.hist
        h.append((now, risk))
        while len(h) > 1 and h[1][0] <= now - TREND_WINDOW_S:
            h.popleft()
        self.risk = risk
        return self.tier, risk - h[0][1]


_STAGE_X = -math.log10(STAGE_E_DAY)


def default_family_stages(family: str) -> Set[str]:
    out = set()
    for a in FAMILY_DEFAULT_AXES.get(family, ()):
        s = stage_for_axis(a)
        if s:
            out.add(s)
    return out


# ====================================================================== engine
class RiskEngine(Engine):
    name = "behavior.risk"
    layer = "behavior"
    consumes = ["behavior.p_family", "behavior.alarm", "behavior.axes", "behavior.common.flag",
                "event.*", "match.*", "store.incidents", "model.feedback", "model.class"]
    produces = ["behavior.risk", "profile.extra.risk"]
    description = ("Decaying 0-100 risk per entity, class and system: cadence-invariant "
                   "excess surprise, episode saturation, damped discrete findings and lib-4 "
                   "matches, kill-chain stage multiplier, fixed L_ref = 60, tiers with "
                   "hysteresis.")
    interval = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._states: "weakref.WeakKeyDictionary[Any, Dict[Tuple[str, str], RiskState]]" = \
            weakref.WeakKeyDictionary()
        self._crit_cache: Tuple[Any, List[Tuple[Optional[Set[str]], List[Any], float, str]]] = \
            (None, [])
        self._warm: Set[int] = set()        # stores whose last tick was a training tick
        # (s, e, signature_id) -> [first_ts, n_ticks, last_ts] of lib-4 matches (habits)
        self._habits: "weakref.WeakKeyDictionary[Any, Dict[Tuple[str, str, str], List[float]]]" \
            = weakref.WeakKeyDictionary()

    # ------------------------------------------------------------------ run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        now = float(ctx.now)
        dt = float(ctx.window_s)
        states = self._states.get(store)
        if states is None:
            states = self._states[store] = {}
        if ctx.training:
            self._warm.add(id(store))
        elif id(store) in self._warm:
            # First live tick after warm-up: the warm-up evidence was scored
            # against models that were still being learnt from those very rows
            # (cold backoff, uncalibrated rings). It is not evidence about the
            # entity, so live risk starts from 0 (the risk series of the
            # warm-up stays in the store).
            self._warm.discard(id(store))
            states.clear()
        fb = m_feedback.get(store)
        pi = {f: m_feedback.risk_mult(fb, f) for f in FAMILIES}
        degraded = store.engine_failed(FUSION_ENGINE, now)
        n = 0
        for s in store.systems():
            inc = self._incident_map(store, s)
            ent_risk: Dict[str, float] = {}
            for e in store.entities(s):
                st = self._state(states, s, e, now)
                r = self._score_key(ctx, store, s, e, st, now, dt, pi, inc, degraded)
                ent_risk[e] = r
                self._write(store, s, e, st, now, dt, r, pi)
                n += 1
            cls_risk: Dict[str, float] = {}
            keys = {p for p in store.pseudo_entities(s) if p.startswith(CLASS_PREFIX)}
            keys.update(m_class.all_class_keys(store, s))
            for c in sorted(keys):
                st = self._state(states, s, c, now)
                own = self._score_key(ctx, store, s, c, st, now, dt, pi, inc, degraded)
                members = m_class.class_members(store, s, c)
                top = sorted(((ent_risk[m], m) for m in members
                              if m in ent_risk and ent_risk[m] == ent_risk[m]), reverse=True)[:3]
                mview = sum(r for r, _ in top) / len(top) if top else math.nan
                r = _nanmax((own, mview))
                cls_risk[c] = r
                self._write(store, s, c, st, now, dt, r, pi, extra={
                    "own": _rnd(own), "members": _rnd(mview),
                    "top_members": [{"entity": m, "risk": _rnd(v)} for v, m in top]})
                n += 1
            if ent_risk or cls_risk:
                self._write_system(store, s, now, dt, ent_risk, cls_risk)
                n += 1
        return n

    # -------------------------------------------------------------- state
    @staticmethod
    def _state(states: Dict[Tuple[str, str], RiskState], s: str, e: str,
               now: float) -> RiskState:
        st = states.get((s, e))
        if st is None or (st.last_ts is not None and now < st.last_ts):
            st = states[(s, e)] = RiskState()            # new key / clock went back
        return st

    @staticmethod
    def _incident_map(store, s: str) -> Dict[str, Set[str]]:
        """{key: statuses} of the system's live incidents (entity and members)."""
        out: Dict[str, Set[str]] = {}
        for inc in store.incidents(system=s, status=("open", "acked", "suppressed")):
            for k in {inc.entity, *(inc.entities or ())}:
                out.setdefault(k, set()).add(inc.status)
        return out

    # ------------------------------------------------------------ scoring
    def _score_key(self, ctx: Context, store, s: str, e: str, st: RiskState, now: float,
                   dt: float, pi: Mapping[str, float], inc: Mapping[str, Set[str]],
                   degraded: bool) -> float:
        if degraded:
            return math.nan          # contract M; state untouched, next tick catches up
        if st.last_ts is not None and now == st.last_ts:
            return st.risk           # already folded this tick (re-run): idempotent
        prev = st.last_ts
        win0 = now - dt if prev is None else prev
        st.decay_to(now)
        statuses = inc.get(e, ())
        episode = bool(statuses)
        suppressed = "suppressed" in statuses and not ({"open", "acked"} & set(statuses))
        mult = SUPPRESSED_MULT if suppressed else 1.0

        # --- continuous evidence
        pf = emit.read_dict(store, s, e, P_FAMILY, now)
        if pf:
            lazy: Dict[str, Any] = {}

            def stages_of(f: str) -> Set[str]:
                if "axes" not in lazy:
                    lazy["axes"] = _family_axes(emit.read_dict(store, s, e, emit.AXES, now))
                axes = lazy["axes"].get(f) or FAMILY_DEFAULT_AXES.get(f, ())
                return {x for x in (stage_for_axis(a) for a in axes) if x}

            # side inputs are read only when they can matter (p < dt/day: the
            # family contributes; p <= 0.03 dt/day: it earns a stage)
            p_min = min((p for p in pf.values() if p == p), default=math.nan)
            vol = False
            ip = pf.get("intensity")
            if ip is not None and ip * DAY < dt:
                vol = bool(emit.read_dict(store, s, e, COMMON_FLAG, now).get("volume"))
            crit = False
            if p_min * DAY <= STAGE_E_DAY * dt:
                alarm = emit.read_dict(store, s, e, ALARM, now)
                crit = _sev(alarm.get("severity")) == "critical"
            st.add_families(now, dt, pf, episode=episode, volume_common=vol, mult=mult,
                            critical=crit, stages_of=stages_of)
        # (no p_family: silent / unscored key -> only decay; streaks are kept)

        # --- discrete findings, (prev, now]
        evs = store.events(s, e, since=win0, kinds=EVENT_KINDS, limit=1000)
        for ev in reversed(evs):     # oldest first, so repeats damp in time order
            if prev is not None and ev.ts <= prev:
                continue
            if ev.ts > now:
                continue
            w = event_weight(ev)
            if w <= 0.0:
                continue
            m = SUPPRESSED_MULT if (suppressed or ev.status == "suppressed") else 1.0
            sev = _sev(ev.severity)
            decay = "critical" if sev == "critical" else EVENT_DECAY.get(ev.kind, "behaviour")
            tier = f"@{_event_tier(ev.extra or {})}" if ev.kind == "first_seen" else ""
            st.add_repeated(f"event:{ev.kind}{tier}", event_key(ev), w * m, decay, now, ev.ts,
                            event_stages(ev), family=m_feedback.EVENT_FAMILY.get(ev.kind),
                            detail=_event_detail(ev))

        # --- lib-4 matches, one tick late: [prev, now)
        habits = self._habits.get(store)
        if habits is None:
            habits = self._habits[store] = {}
        for mt in reversed(store.matches(s, e, since=win0, limit=1000)):
            if mt.ts >= now:
                continue
            sev = _sev(mt.severity)
            habitual = self._habit(habits, s, e, str(mt.signature_id), float(mt.ts))
            w = SIG_W.get(sev, 0.0) * min(1.0, max(0.0, float(mt.confidence or 0.0)))
            if habitual and sev in HABIT_SEVERITIES:
                w *= HABIT_MULT
            if w <= 0.0:
                continue
            stage = stage_for_category(mt.category)
            st.add_repeated(f"lib4:{mt.signature_id}", f"sig|{mt.signature_id}", w * mult,
                            match_decay(sev, stage), now, mt.ts, (stage,) if stage else (),
                            detail=f"{mt.label or mt.signature_id} ({sev}, "
                                   f"conf {float(mt.confidence or 0.0):.2f})")
        st.prune(now)
        return st.score(now, pi, self._criticality(ctx.config, s, e))

    @staticmethod
    def _habit(habits: Dict[Tuple[str, str, str], List[float]], s: str, e: str, sig: str,
               ts: float) -> bool:
        """Record one lib-4 match of (s, e, sig) at ts (once per tick) and say
        whether the activity was already habitual before it."""
        k = (s, e, sig)
        h = habits.get(k)
        if h is None or ts - h[2] > HABIT_FORGET_S:
            habits[k] = [ts, 1.0, ts]
            return False
        habitual = h[1] >= HABIT_MIN_TICKS and ts - h[0] >= HABIT_S
        if ts > h[2]:
            h[1] += 1.0
            h[2] = ts
        return habitual

    # ------------------------------------------------------------ writing
    def _write(self, store, s: str, e: str, st: RiskState, now: float, dt: float, r: float,
               pi: Mapping[str, float], extra: Optional[Dict[str, Any]] = None) -> None:
        store.add_vec(s, e, RISK, now, [r], window_s=int(dt))
        prof = store.profile(s, e)
        if prof is None:
            prof = EntityProfile(system=s, entity=e, updated=now)
        if not r == r:
            old = prof.extra.get("risk")
            prof.extra["risk"] = dict(old or {}, degraded=True, updated=now)
            store.put_profile(prof)
            return
        tier, trend = st.finalize(now, r)
        load = st.load(pi)
        stages = st.stages(now)
        out = {
            "score": _rnd(r), "tier": tier, "trend": _rnd(trend),
            "top_reasons": _top_reasons(st, pi, load),
            "stages": stages, "L": _rnd(st.total()), "M": stage_multiplier(len(stages)),
            "updated": now, "degraded": False,
        }
        if extra:
            out.update(extra)
        prof.extra["risk"] = out
        store.put_profile(prof)

    def _write_system(self, store, s: str, now: float, dt: float, ent: Mapping[str, float],
                      cls: Mapping[str, float]) -> None:
        allv = [(v, k) for k, v in (*ent.items(), *cls.items()) if v == v]
        r = max(allv)[0] if allv else math.nan
        store.add_vec(s, SYSTEM_KEY, RISK, now, [r], window_s=int(dt))
        prof = store.profile(s, SYSTEM_KEY) or EntityProfile(system=s, entity=SYSTEM_KEY,
                                                             updated=now)
        allv.sort(reverse=True)
        prof.extra["risk"] = {
            "score": _rnd(r), "tier": tier_of(r) if r == r else None,
            "top": [{"key": k, "risk": _rnd(v)} for v, k in allv[:TOP_REASONS]],
            "n_medium_plus": sum(1 for v, _ in allv if v >= TIER_THRESHOLDS[0]),
            "updated": now,
        }
        store.put_profile(prof)

    # -------------------------------------------------------- criticality
    def _criticality(self, cfg: Mapping[str, Any], s: str, e: str) -> float:
        """c_s in [0.5, 2]: ctx.config['criticality'] (a number, or a mapping
        keyed 'system|entity', entity, system or 'default'), else the max
        criticality of the ip_classes whose CIDRs contain the entity (a
        static class key takes its own class's), else 1."""
        c = cfg.get("criticality")
        val: Optional[float] = None
        if isinstance(c, Mapping):
            for k in (f"{s}|{e}", e, s, "default"):
                if k in c and _finite(c[k]):
                    val = float(c[k])
                    break
        elif _finite(c):
            val = float(c)
        if val is None:
            val = self._ip_class_crit(cfg.get("ip_classes") or [], s, e)
        if val is None:
            return 1.0
        return min(CRIT_RANGE[1], max(CRIT_RANGE[0], val))

    def _ip_class_crit(self, classes: List[Any], s: str, e: str) -> Optional[float]:
        if not classes:
            return None
        if self._crit_cache[0] is not classes:
            parsed = []
            for c in classes:
                if not isinstance(c, Mapping) or not _finite(c.get("criticality")):
                    continue
                systems = set(c.get("systems") or ()) or None
                nets = []
                for cidr in c.get("cidrs") or ():
                    try:
                        nets.append(ipaddress.ip_network(str(cidr), strict=False))
                    except ValueError:
                        continue
                parsed.append((systems, nets, float(c["criticality"]), str(c.get("name", ""))))
            self._crit_cache = (classes, parsed)
        parsed = self._crit_cache[1]
        best: Optional[float] = None
        if e.startswith(STATIC_PREFIX):
            name = e[len(STATIC_PREFIX):]
            for systems, _nets, crit, cname in parsed:
                if cname == name and (systems is None or s in systems):
                    best = crit if best is None else max(best, crit)
            return best
        try:
            ip = ipaddress.ip_address(e)
        except ValueError:
            return None
        for systems, nets, crit, _ in parsed:
            if systems is not None and s not in systems:
                continue
            if any(ip.version == n.version and ip in n for n in nets):
                best = crit if best is None else max(best, crit)
        return best


# ===================================================================== helpers
def _family_axes(axes_by_det: Mapping[str, Any]) -> Dict[str, List[str]]:
    """{family: axes} from behavior.axes {detector: [axis]} (refined axes such
    as exfil / privilege / credential that the detectors wrote this tick)."""
    out: Dict[str, List[str]] = {}
    for d, ax in axes_by_det.items():
        info = DETECTOR_INFO.get(d)
        if info is None or not ax:
            continue
        lst = out.setdefault(str(info["family"]), [])
        for a in ax:
            if a not in lst:
                lst.append(str(a))
    return out


def _event_detail(ev: Any) -> str:
    x = ev.extra or {}
    parts = [ev.kind]
    if ev.kind == "first_seen":
        parts.append(f"tier={_event_tier(x)}")
    tok = x.get("token")
    if tok is None and ("dim" in x or "value" in x):
        tok = f"{x.get('dim', '*')}={x.get('value', '')}"
    if tok is not None:
        parts.append(str(tok))
    return " ".join(parts)


def _top_reasons(st: RiskState, pi: Mapping[str, float], load: float) -> List[Dict[str, Any]]:
    items = sorted(((v * pi.get(st.fam.get(k) or "", 1.0), k) for k, v in st.L.items()),
                   reverse=True)[:TOP_REASONS]
    out = []
    for w, k in items:
        src, _, name = k.partition(":")
        out.append({"key": k, "source": src, "name": name, "L": _rnd(st.L[k]),
                    "share": _rnd(w / load) if load > 0 else None,
                    "half_life_h": st.H[k] / HOUR, "last_ts": st.comp_ts.get(k),
                    "detail": st.detail.get(k, "")})
    return out


def _nanmax(vals: Iterable[float]) -> float:
    v = [x for x in vals if x == x]
    return max(v) if v else math.nan


def _rnd(x: Any, nd: int = 3) -> Optional[float]:
    return round(float(x), nd) if x is not None and x == x else None
