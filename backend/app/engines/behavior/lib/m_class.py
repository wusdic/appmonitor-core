"""Read accessors for model.class (owner: B02 peer_group; contract C, L).

Layout of model.class@('__org__', '__org__'):
    {
      "assign": {"<system>|<ip>": {"role": rid, "sub": sid, "prob": float,
                                   "static": [name, ...], "pool": cidr|None,
                                   "super": "human"|"machine"}},
      "roles":  {rid: {"name": str, "members": ["<system>|<ip>", ...],
                       "medoid": "<system>|<ip>", "lineage": [...], "version": int}},
      "subs": {...}, "statics": {name: {...}}, "pools": {cidr: {...}},
      "version": int,
    }

Every consumer that backs off from an entity to its class (B03 hierarchy,
B07 class rhythm prior, B08 class vocab, B09 class stacks, B10 class PPM,
B13, B18, ...) goes through these functions, so the layout can evolve in one
place. Role ids are global; class *state* is always keyed per system as
(system, 'class:<rid>') (contract L). A role with fewer than MIN_MEMBERS
members in a system is not used as a backoff tier there (back off to the
system tier instead, never to org).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .classkeys import ORG, pool_key, role_key, static_key

MODEL = "model.class"
MIN_MEMBERS = 3        # contract L: < 3 members in a system -> system tier
MIN_PROB = 0.5         # membership probability to count as a member


def get(store) -> Dict:
    return store.get_model(ORG[0], ORG[1], MODEL, default=None) or {}


def assignment(store, system: str, entity: str) -> Optional[Dict]:
    return get(store).get("assign", {}).get(f"{system}|{entity}")


def role_id(store, system: str, entity: str) -> Optional[str]:
    a = assignment(store, system, entity)
    if not a or a.get("role") in (None, "", "unique"):
        return None
    return str(a["role"])


def members(store, system: str, rid: str, min_prob: float = MIN_PROB) -> List[str]:
    """Real entities of role `rid` in `system` with prob >= min_prob."""
    out = []
    for key, a in get(store).get("assign", {}).items():
        s, _, ip = key.partition("|")
        if s == system and str(a.get("role")) == str(rid) and float(a.get("prob", 1.0)) >= min_prob:
            out.append(ip)
    return sorted(out)


def class_key(store, system: str, entity: str, min_members: int = MIN_MEMBERS) -> Optional[str]:
    """'class:<rid>' to back off to for this entity in this system, or None
    when unassigned / unique / the role is too small in this system."""
    rid = role_id(store, system, entity)
    if rid is None:
        return None
    if len(members(store, system, rid)) < min_members:
        return None
    return role_key(rid)


def all_class_keys(store, system: str, min_members: int = 2) -> List[str]:
    """Every class pseudo-entity key alive in `system`: role classes with at
    least `min_members` members, static CIDR classes and pools."""
    m = get(store)
    keys = set()
    roles: Dict[str, int] = {}
    for key, a in m.get("assign", {}).items():
        s, _, _ip = key.partition("|")
        if s != system:
            continue
        r = a.get("role")
        if r not in (None, "", "unique") and float(a.get("prob", 1.0)) >= MIN_PROB:
            roles[str(r)] = roles.get(str(r), 0) + 1
        for name in a.get("static", []) or []:
            keys.add(static_key(name))
        if a.get("pool"):
            keys.add(pool_key(a["pool"]))
    keys.update(role_key(r) for r, n in roles.items() if n >= min_members)
    return sorted(keys)


def class_members(store, system: str, key: str) -> List[str]:
    """Members (real IPs in `system`) of any class key ('class:<rid>',
    'class:static:<name>', 'class:pool:<cidr>')."""
    m = get(store)
    out = []
    for k, a in m.get("assign", {}).items():
        s, _, ip = k.partition("|")
        if s != system:
            continue
        if key.startswith("class:static:"):
            if key[len("class:static:"):] in (a.get("static") or []):
                out.append(ip)
        elif key.startswith("class:pool:"):
            if a.get("pool") == key[len("class:pool:"):]:
                out.append(ip)
        elif key.startswith("class:"):
            if str(a.get("role")) == key[len("class:"):] and float(a.get("prob", 1.0)) >= MIN_PROB:
                out.append(ip)
    return sorted(out)


def role_name(store, rid: Optional[str]) -> str:
    if rid is None:
        return ""
    return str(get(store).get("roles", {}).get(str(rid), {}).get("name", rid))


def version(store) -> int:
    return int(get(store).get("version", 0))
