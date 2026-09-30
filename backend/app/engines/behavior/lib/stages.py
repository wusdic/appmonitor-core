"""Kill-chain stage map (contract K).

Risk (B26) multiplies evidence by the number of distinct stages seen in 24 h,
and incidents / portraits narrate by stage. Stages are derived from evidence
*axes* (not detector names), from discrete event kinds and from lib-4
signature categories, so any engine that labels its axes correctly is mapped
without touching this module. Context flags refine the categorical, breadth
and sequence axes (e.g. a categorical novelty to an external, upload-dominant
destination is exfiltration, a sensitive one is privilege).
"""
from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Set

STAGES: List[str] = ["behavior", "off_hours", "discovery", "credential", "privilege",
                     "collection", "c2", "exfiltration", "identity", "lateral"]

# Axes that are plain behavioural deviation. Feature-group names are also
# valid axes (B04/B14 write the groups whose p < 0.01).
_BEHAVIOR_AXES = frozenset({"volume", "shape", "peer", "change", "app", "dns", "tls",
                            "timing", "transport", "probe", "comp",
                            "content"})   # P03 content / binding violations (progressive.md §9.2)
_DIRECT_AXES = {
    "temporal": "off_hours",
    "exfil": "exfiltration",
    "privilege": "privilege",
    "credential": "credential",
    "identity": "identity",
    "c2": "c2",
    "xsys": "lateral",
    "lateral": "lateral",
    "discovery": "discovery",
    "collection": "collection",
}

FLAG_NAMES = ("sensitive", "admin", "external", "upload_dominant", "new_external_domain",
              "low_prevalence", "internal", "object_ids", "auth", "login_4xx")

_EVENT_STAGE = {
    "identity_mismatch": "identity",
    "unknown_identity": "identity",
    "client_impersonation": "identity",
    "possible_impersonation": "identity",
    "beacon": "c2",
}

_CATEGORY_STAGE = {
    "recon": "discovery", "scan": "discovery",
    "auth": "credential", "bruteforce": "credential",
    "tunnel": "c2", "beacon": "c2", "c2": "c2",
    "transfer": "exfiltration", "exfil": "exfiltration",
    "admin": "privilege",
}


def stage_for_axis(axis: str, *, sensitive: bool = False, admin: bool = False,
                   external: bool = False, upload_dominant: bool = False,
                   new_external_domain: bool = False, low_prevalence: bool = False,
                   internal: bool = False, object_ids: bool = False, auth: bool = False,
                   login_4xx: bool = False) -> Optional[str]:
    """Stage of one evidence axis, refined by context flags. None = unknown axis.

    categorical: sensitive|admin -> privilege; external & upload_dominant ->
    exfiltration; new_external_domain|low_prevalence -> c2; else behavior.
    breadth: object_ids -> collection; otherwise (internal peers/ports) ->
    discovery. sequence: auth token family or 4xx on login -> credential,
    else behavior. `internal` is accepted for symmetry with the contract text.
    """
    if axis in _BEHAVIOR_AXES:
        return "behavior"
    if axis in _DIRECT_AXES:
        return _DIRECT_AXES[axis]
    if axis == "categorical":
        if sensitive or admin:
            return "privilege"
        if external and upload_dominant:
            return "exfiltration"
        if new_external_domain or low_prevalence:
            return "c2"
        return "behavior"
    if axis == "breadth":
        return "collection" if object_ids else "discovery"
    if axis == "sequence":
        return "credential" if (auth or login_4xx) else "behavior"
    return None


def stage_for_event(kind: str) -> Optional[str]:
    """Stage implied by a discrete event kind itself (identity events, beacon)."""
    return _EVENT_STAGE.get(kind)


def stage_for_category(category: str) -> Optional[str]:
    """Stage of a lib-4 signature category (recon/scan, auth/bruteforce, ...)."""
    return _CATEGORY_STAGE.get((category or "").lower())


def stages_for(axes: Iterable[str], flags: Optional[Mapping[str, bool]] = None) -> Set[str]:
    """Distinct known stages over a set of axes sharing the same context flags."""
    kw = {k: bool(v) for k, v in (flags or {}).items() if k in FLAG_NAMES}
    out: Set[str] = set()
    for a in axes:
        st = stage_for_axis(a, **kw)
        if st is not None:
            out.add(st)
    return out


# ------------------------------------------------------------ lib-4 habits
# A routine lib-4 activity of an entity (integration notes §4): a signature of
# severity <= medium that the (entity, signature) has matched on at least
# HABIT_MIN_TICKS ticks, the first of them at least HABIT_S ago. B26 gives it
# no risk weight and B27 does not let it restart an incident's quiet clock;
# each engine keeps its own memory (engines share no state) with this rule.
HABIT_SEVERITIES = frozenset({"info", "low", "medium"})
HABIT_S = 86400.0              # first match of the (entity, signature) at least this old
HABIT_MIN_TICKS = 4            # ... and matched on at least this many ticks
HABIT_FORGET_S = 30 * 86400.0  # a habit not seen for 30 d is forgotten


def habit_step(habits: dict, key: tuple, ts: float) -> bool:
    """Record one lib-4 match of `key` (e.g. (system, entity, signature)) at
    ts (once per tick) and say whether the activity was already habitual
    before it. `habits` maps key -> [first_ts, n_ticks, last_ts]."""
    h = habits.get(key)
    if h is None or ts - h[2] > HABIT_FORGET_S:
        habits[key] = [ts, 1.0, ts]
        return False
    habitual = h[1] >= HABIT_MIN_TICKS and ts - h[0] >= HABIT_S
    if ts > h[2]:
        h[1] += 1.0
        h[2] = ts
    return habitual
