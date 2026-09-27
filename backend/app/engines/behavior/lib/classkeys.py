"""Pseudo-entity keys (contract B and L).

Classes, the system tier and the org tier are stored as pseudo-entities next
to real IPs, so every per-entity mechanism (models, detectors, risk,
incidents, portraits) works for them unchanged. Keys are built only here so
that no engine hand-formats 'class:...' strings, and role classes are keyed
by their stable *id*, never by a display name (renaming a role must keep its
models).

A class key always lives under the real system it describes:
(system, 'class:<rid>'). The org tier is the pair ORG = ('__org__', '__org__').
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

SYSTEM_KEY = "__system__"
ORG_SYSTEM = "__org__"
ORG_ENTITY = "__org__"
ORG: Tuple[str, str] = (ORG_SYSTEM, ORG_ENTITY)
CLASS_PREFIX = "class:"
STATIC_PREFIX = "class:static:"
POOL_PREFIX = "class:pool:"


def role_key(rid: Union[str, int]) -> str:
    """Role (dynamic) class key: 'class:<rid>'."""
    rid = str(rid)
    if not rid or rid.startswith(("static:", "pool:")):
        raise ValueError(f"invalid role id {rid!r}")
    return CLASS_PREFIX + rid


def static_key(name: str) -> str:
    """Static CIDR class key from ctx.config.ip_classes: 'class:static:<name>'."""
    if not name:
        raise ValueError("empty static class name")
    return STATIC_PREFIX + str(name)


def pool_key(cidr: str) -> str:
    """Synthesised pool class key: 'class:pool:<cidr>'."""
    if not cidr:
        raise ValueError("empty pool cidr")
    return POOL_PREFIX + str(cidr)


def is_pseudo(entity: str) -> bool:
    """True for '__system__', '__org__', any '__*' key and any 'class:*' key."""
    return entity.startswith("__") or entity.startswith(CLASS_PREFIX)


def is_class(entity: str) -> bool:
    return entity.startswith(CLASS_PREFIX)


def class_kind(entity: str) -> Optional[str]:
    """'static' | 'pool' | 'role' for a class key, None for anything else."""
    if entity.startswith(STATIC_PREFIX):
        return "static"
    if entity.startswith(POOL_PREFIX):
        return "pool"
    if entity.startswith(CLASS_PREFIX):
        return "role"
    return None


def class_id(entity: str) -> Optional[str]:
    """The id part of a class key (rid, static name or cidr); None otherwise."""
    kind = class_kind(entity)
    if kind == "static":
        return entity[len(STATIC_PREFIX):]
    if kind == "pool":
        return entity[len(POOL_PREFIX):]
    if kind == "role":
        return entity[len(CLASS_PREFIX):]
    return None


def is_org(system: str, entity: str) -> bool:
    return (system, entity) == ORG


def assign_key(system: str, entity: str) -> str:
    """Key of model.class.assign: 'system|ip'."""
    return f"{system}|{entity}"
