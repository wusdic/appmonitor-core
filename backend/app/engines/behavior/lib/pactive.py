"""Active / earned sets and bounded mode for the B-library
(docs/lib3/progressive.md §10.1-§10.3, §6.19; W-P6).

STATUS: implemented (W-P6). Store accessors only (model.budget is written by
P15 behavior/resource_governor.py; model.earned by B04 in bounded mode).

`config['lib3']['resource_mode']` in {'full', 'bounded'} (default 'full'). In
full mode every accessor returns exactly what the engines used before
(store.entities(s)), so full-mode results are unchanged by construction. In
bounded mode a B-engine iterates

    entities(store, s, now, config)   = A_t(s) ∪ E_t(s)
        A_t(s)  active set: IPs with observations in the linger window
                (default 24 h, covering the accumulators' short horizons),
                IPs with an open incident, quarantined IPs (P15, §6.19);
        E_t(s)  earned set: per-IP models earned by their prequential gain
                over the class model (B04 shadow records, §10.2), selected by P15;
    earned_entities(store, s, config) = E_t(s) (per-entity models: B03, B06,
                B07, B10, B14, B15 fit only these; every other IP is covered
                by its class / group models and by the pattern tree).

A missing or stale model.budget (P15 not registered, first tick) falls back to
the full list, so bounded mode can never silently drop an entity it has no
information about. Cost: O(|A_t| + |E_t|) per call (sets are published as
sorted lists; `contains` uses a per-tick frozenset cache).
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from ....models.schema import ORG

BUDGET = "model.budget"
EARNED = "model.earned"
MODES = ("full", "bounded")
_CACHE: Dict[Tuple[int, str, float, str], FrozenSet[str]] = {}


def mode(config: Optional[Mapping[str, Any]]) -> str:
    m = ((config or {}).get("lib3") or {}).get("resource_mode", "full")
    return m if m in MODES else "full"


def bounded(config: Optional[Mapping[str, Any]]) -> bool:
    return mode(config) == "bounded"


def _budget(store: Any) -> Mapping[str, Any]:
    b = store.get_model(ORG, ORG, BUDGET)
    return b if isinstance(b, Mapping) else {}


def system_sets(store: Any, s: str) -> Optional[Mapping[str, Any]]:
    """P15's record of system s ({'active', 'earned', 'ts', ...}) or None."""
    rec = (_budget(store).get("systems") or {}).get(s)
    return rec if isinstance(rec, Mapping) else None


STALE_S = 2 * 3600.0            # sets older than this are not used (P15 not running)
LINGER_S = 24 * 3600.0


def _fresh(rec: Optional[Mapping[str, Any]], now: Optional[float]) -> bool:
    if rec is None:
        return False
    if now is None:
        return True
    ts = rec.get("ts")
    return ts is not None and float(ts) >= float(now) - STALE_S


def tick_entities(store: Any, s: str, since: float, now: float) -> FrozenSet[str]:
    """Entities of system s observed after `since` (the observation deque walked
    back from its newest end: O(observations since), cached per tick). Engines
    that run before P15 in a tick (raw, derived, B01 when P15 sits later) see
    the previous tick's sets plus these."""
    key = (id(store), s, float(now), "tick")
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    out = set()
    lim = 1024
    while True:
        obs = store.recent_observations(lim)
        for o in obs:
            if o.ts > since and o.system == s:
                out.add(o.entity)
        if len(obs) < lim or not obs or obs[0].ts <= since:
            break
        if lim >= 1 << 20:
            out |= set(store.entities_active(s, since + 1e-6))
            break
        lim *= 4
    if len(_CACHE) > 4096:
        _CACHE.clear()
    hit = _CACHE[key] = frozenset(e for e in out if e and not e.startswith(("class:", "__")))
    return hit


def entities(store: Any, s: str, now: Optional[float], config: Optional[Mapping[str, Any]],
             include_earned: bool = True) -> List[str]:
    """Entities a B-engine processes this tick (module docstring). Full mode:
    store.entities(s), unchanged."""
    if not bounded(config):
        return store.entities(s)
    rec = system_sets(store, s)
    if not _fresh(rec, now):
        return store.entities(s)
    out = set(rec.get("active") or ())
    if include_earned:
        out |= set(rec.get("earned") or ())
    ts = float(rec.get("ts") or 0.0)
    if now is not None and ts < float(now) - 1e-6:
        out |= tick_entities(store, s, ts, float(now))
    return sorted(out)


def earned_entities(store: Any, s: str, config: Optional[Mapping[str, Any]],
                    now: Optional[float] = None) -> Optional[FrozenSet[str]]:
    """E_t(s) in bounded mode (None in full mode = every entity is 'earned')."""
    if not bounded(config):
        return None
    rec = system_sets(store, s)
    if rec is None:
        return None
    key = (id(store), s, float(rec.get("ts") or 0.0), "earned")
    hit = _CACHE.get(key)
    if hit is None:
        if len(_CACHE) > 4096:
            _CACHE.clear()
        hit = _CACHE[key] = frozenset(rec.get("earned") or ())
    return hit


def is_earned(store: Any, s: str, e: str, config: Optional[Mapping[str, Any]]) -> bool:
    """True in full mode; in bounded mode membership of E_t(s) (True when P15
    has not published yet: no information -> no restriction)."""
    ee = earned_entities(store, s, config)
    return True if ee is None else e in ee


def applicability(store: Any, s: str, engine: str, config: Optional[Mapping[str, Any]]) -> str:
    """P12's B-engine applicability ('on' | 'class' | 'off') for engine ids
    'B09', 'B10', ... in bounded mode; always 'on' in full mode."""
    if not bounded(config):
        return "on"
    from ....models.schema import SYSTEM_ENTITY
    sp = store.get_model(s, SYSTEM_ENTITY, "model.sysprof")
    ch = (sp or {}).get("chosen") if isinstance(sp, Mapping) else None
    if isinstance(ch, Mapping) and engine in ch:
        v = str(ch[engine])
        return v if v in ("on", "class", "off") else "on"
    return "on"


PERIODIC_MIN = 0.5              # B11 / B12 prefilter: derived.periodicity_score (§10.3)


def periodic_candidate(store: Any, s: str, e: str, config: Optional[Mapping[str, Any]]) -> bool:
    """B11 timing / B12 beacon prefilter in bounded mode: earned IPs and IPs whose
    periodicity score (D0) is >= 0.5; always True in full mode."""
    if not bounded(config):
        return True
    ee = earned_entities(store, s, config)
    if ee is None or e in ee:                 # no sets yet (no restriction) or earned
        return True
    v = store.latest_derived(s, e, "derived.periodicity_score")
    val = getattr(v, "value", None)
    return isinstance(val, (int, float)) and float(val) >= PERIODIC_MIN


def linger_s(config: Optional[Mapping[str, Any]]) -> float:
    """The active-set linger window (config lib3.linger_s, default 24 h)."""
    try:
        return float(((config or {}).get("lib3") or {}).get("linger_s") or LINGER_S)
    except (TypeError, ValueError):
        return LINGER_S


def e_max(store: Any, s: str, default: int = 32) -> int:
    rec = system_sets(store, s) or {}
    try:
        return int(rec.get("e_max", default))
    except (TypeError, ValueError):
        return default
