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
B02 also keeps (readers ignore what they do not need): assign.class_path
('super/role[/sub]'), assign.A (per-IP automation index), assign.provisional /
pend / D (cold typing and transition hysteresis); roles.super / A / d90 / desc /
cluster / subs / absent / name_parts; and the private '_state' key (B02 only).
The model is copy-on-write: every assignment change is put as a new dict with
version + 1, which is what the member index below keys on.

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


class _Index:
    """Per-(model object, version) member index, so class_key / members /
    class_members / all_class_keys are O(1)-ish instead of a scan of every
    assignment per call (integration R16.2: class_key runs inside
    m_baseline.parent_key / predictive_set once per entity per tick, which
    made the scan O(N^2) per tick). model.class is copy-on-write (B02 puts a
    new dict with version + 1 on every assignment change), so the model
    object's identity and version key the cache."""
    __slots__ = ("key", "model", "roles", "statics", "pools", "role_n")

    def __init__(self, m: Dict) -> None:
        assign = m.get("assign", {}) or {}
        self.model = m                      # keeps id(m) from being reused
        self.key = (id(m), m.get("version"), id(assign), len(assign))
        roles: Dict[tuple, List[str]] = {}
        statics: Dict[tuple, List[str]] = {}
        pools: Dict[tuple, List[str]] = {}
        for key, a in assign.items():
            sys_, _, ip = key.partition("|")
            r = a.get("role")
            if r not in (None, "") and float(a.get("prob", 1.0)) >= MIN_PROB:
                roles.setdefault((sys_, str(r)), []).append(ip)
            for name in a.get("static", []) or []:
                statics.setdefault((sys_, str(name)), []).append(ip)
            if a.get("pool"):
                pools.setdefault((sys_, str(a["pool"])), []).append(ip)
        for d in (roles, statics, pools):
            for v in d.values():
                v.sort()
        self.roles, self.statics, self.pools = roles, statics, pools
        role_n: Dict[str, Dict[str, int]] = {}
        for (sys_, r), ips in roles.items():
            if r != "unique":
                role_n.setdefault(sys_, {})[r] = len(ips)
        self.role_n = role_n


_CACHE: List[_Index] = []


def _index(m: Dict) -> _Index:
    assign = m.get("assign", {}) or {}
    key = (id(m), m.get("version"), id(assign), len(assign))
    if _CACHE and _CACHE[0].key == key and _CACHE[0].model is m:
        return _CACHE[0]
    ix = _Index(m)
    _CACHE[:] = [ix]
    return ix


def members(store, system: str, rid: str, min_prob: float = MIN_PROB) -> List[str]:
    """Real entities of role `rid` in `system` with prob >= min_prob."""
    m = get(store)
    if min_prob == MIN_PROB:
        return list(_index(m).roles.get((system, str(rid)), ()))
    out = []
    for key, a in m.get("assign", {}).items():
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
    if len(_index(get(store)).roles.get((system, rid), ())) < min_members:
        return None
    return role_key(rid)


def all_class_keys(store, system: str, min_members: int = 2) -> List[str]:
    """Every class pseudo-entity key alive in `system`: role classes with at
    least `min_members` members, static CIDR classes and pools."""
    ix = _index(get(store))
    keys = {static_key(name) for (s, name) in ix.statics if s == system}
    keys.update(pool_key(c) for (s, c) in ix.pools if s == system)
    keys.update(role_key(r) for r, n in ix.role_n.get(system, {}).items() if n >= min_members)
    return sorted(keys)


def class_members(store, system: str, key: str) -> List[str]:
    """Members (real IPs in `system`) of any class key ('class:<rid>',
    'class:static:<name>', 'class:pool:<cidr>')."""
    ix = _index(get(store))
    if key.startswith("class:static:"):
        return list(ix.statics.get((system, key[len("class:static:"):]), ()))
    if key.startswith("class:pool:"):
        return list(ix.pools.get((system, key[len("class:pool:"):]), ()))
    if key.startswith("class:"):
        return list(ix.roles.get((system, key[len("class:"):]), ()))
    return []


def role_name(store, rid: Optional[str]) -> str:
    if rid is None:
        return ""
    return str(get(store).get("roles", {}).get(str(rid), {}).get("name", rid))


def version(store) -> int:
    return int(get(store).get("version", 0))
