"""Store accessors for the progressive core's models and batches
(docs/lib3/progressive.md §5.6, §6.20).

STATUS: implemented (W-P0). Engines couple only through the store; these
helpers fix the store names, the tree-key resolution (system families) and
the construction of generalisation hierarchies from the current models, so
every P engine reads the same thing the same way.

Tree key: a system's patterns live under key(s) = model.sysfam['member'][s]
('fam:<id>') when the system belongs to a family, else s itself. Per-tree
models are stored at (key, '__system__'); org-wide ones at ('__org__', '__org__').
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ....models.schema import ORG, SYSTEM_ENTITY
from . import pevent as EV
from .phier import Hierarchies, Regions
from .pregistry import A_MAX, AttrRegistry
from .ptree import PTreeModel, TIERS

# model names (§5.6)
PTREE = "model.ptree"
ATTR = "model.attr"
ATTRSEL = "model.attrsel"
PWANT = "model.pwant"
PBOUNDS = "model.pbounds"
PGRAMMAR = "model.pgrammar"
PBIND = "model.pbind"
PWIN = "model.pwin"
PFLOW = "model.pflow"
WHO_GROUPS = "model.who_groups"
SYSPROF = "model.sysprof"
SYSFAM = "model.sysfam"
FACETS = "model.facets"
PVIEWS = "model.pviews"
BUDGET = "model.budget"
OPS_BUDGET = "ops.budget"
OPS_TICK = "ops.tick"
CHECKPOINT = "ptree"


def tree_key(store: Any, system: str) -> str:
    """The pattern-tree key of a system: its family key or itself."""
    fam = store.get_model(ORG, ORG, SYSFAM)
    if isinstance(fam, Mapping):
        k = (fam.get("member") or {}).get(system)
        if k:
            return str(k)
    return system


# ----------------------------------------------------------------- models
def get_ptree(store: Any, key: str) -> Optional[PTreeModel]:
    return store.get_model(key, SYSTEM_ENTITY, PTREE)


def ensure_ptree(store: Any, key: str, t: float) -> PTreeModel:
    m = get_ptree(store, key)
    if m is None:
        m = PTreeModel(key)
        store.put_model(key, SYSTEM_ENTITY, PTREE, m, version=1, ts=t)
    return m


def put_ptree(store: Any, key: str, m: PTreeModel, t: float) -> None:
    m.version += 1
    store.put_model(key, SYSTEM_ENTITY, PTREE, m, version=m.version, ts=t)


def get_registry(store: Any, key: str) -> Optional[AttrRegistry]:
    return store.get_model(key, SYSTEM_ENTITY, ATTR)


def ensure_registry(store: Any, key: str, config: Optional[Mapping[str, Any]] = None,
                    a_max: int = A_MAX) -> AttrRegistry:
    reg = get_registry(store, key)
    if reg is None:
        pc = EV.pconfig(config)
        hints = tuple((pc.get("type_hints") or {}).get("code") or ())
        reg = AttrRegistry(key, a_max=a_max, code_hints=hints or AttrRegistry("x").code_hints,
                           day_offset_s=_tz_offset(config))
        store.put_model(key, SYSTEM_ENTITY, ATTR, reg, version=1)
    return reg


def _tz_offset(config: Optional[Mapping[str, Any]]) -> float:
    try:
        from zoneinfo import ZoneInfo
        import datetime as _dt
        tz = ZoneInfo((config or {}).get("tz") or "Asia/Shanghai")
        off = _dt.datetime(2026, 1, 15, tzinfo=tz).utcoffset()
        return float(off.total_seconds()) if off is not None else 0.0
    except Exception:
        return 8 * 3600.0


def get_model(store: Any, key: str, name: str, default: Any = None) -> Any:
    return store.get_model(key, SYSTEM_ENTITY, name, default)


def get_org_model(store: Any, name: str, default: Any = None) -> Any:
    return store.get_model(ORG, ORG, name, default)


def who_groups(store: Any) -> Dict[str, Any]:
    wg = get_org_model(store, WHO_GROUPS)
    return wg if isinstance(wg, dict) else {}


def root_windows(store: Any, key: str) -> Tuple[List[Tuple[int, int, str]],
                                                   Dict[str, List[Tuple[int, int, str]]]]:
    """System-root windows of P09 (model.pwin['root']) for the time level l2:
    ([(start, end, label)], {'wd': [...], 'nwd': [...]})."""
    pw = get_model(store, key, PWIN)
    if not isinstance(pw, Mapping):
        return [], {}
    root = pw.get("root") or {}
    allw = [tuple(w) for w in root.get("all", [])]
    by = {k: [tuple(w) for w in v] for k, v in (root.get("by_daytype") or {}).items()}
    return allw, by


def hierarchies(store: Any, key: str, config: Optional[Mapping[str, Any]] = None,
                registry: Optional[AttrRegistry] = None) -> Hierarchies:
    """The Hierarchies of a tree from the current models: registry types and
    hierarchy models, P11 ip2g, regions (config ip_classes / dhcp_scopes,
    else P11 prefix covers), P09 root windows, shared IPs (who_groups['shared']
    and the registry's snat flags), family members (net.dst levels)."""
    reg = registry if registry is not None else get_registry(store, key)
    wg = who_groups(store)
    regions = Regions.from_config(config)
    if not len(regions) and wg.get("covers"):
        regions = Regions([(str(g), list(c)) for g, c in (wg.get("covers") or {}).items()])
    allw, by = root_windows(store, key)
    fam = get_org_model(store, SYSFAM) or {}
    members = {}
    if isinstance(fam, Mapping):
        members = dict((fam.get("dst_members") or {}).get(key, {}))
    cfg = config or {}
    return Hierarchies(registry=reg if reg is not None else {}, ip2g=wg.get("ip2g") or {},
                       n_groups=len(wg.get("groups") or {}), regions=regions, windows=allw,
                       windows_dt=by, shared=set(wg.get("shared") or ()),
                       dst_members=members, dst_sites=dict(cfg.get("dst_sites") or {}),
                       day_hours=tuple(cfg.get("daypart_day_hours") or (8, 20)))


def budget_for(store: Any, key: str) -> Dict[str, Any]:
    """P15's caps for a tree (model.budget['trees'][key]); {} before P15 ran."""
    b = get_org_model(store, BUDGET)
    if isinstance(b, Mapping):
        return dict((b.get("trees") or {}).get(key, {}))
    return {}


def tier_n_max(tier: str) -> int:
    return TIERS.get(tier, TIERS["M"])


# ---------------------------------------------------------------- batches
def batches_since(store: Any, system: str, name: str, since: float,
                  until: Optional[float] = None) -> List[Tuple[float, Any]]:
    """[(ts, batch)] with since < ts <= until (store batch series)."""
    out = store.batches_since(system, name, since)
    if until is not None:
        out = [(ts, b) for ts, b in out if ts <= until]
    return out


def learnable_batches(store: Any, system: str, name: str, last_learned: Optional[float],
                      now: float, delay: float) -> List[Tuple[float, Any]]:
    """Batches of tick t' with last_learned < t' <= now - D (§6.9.3)."""
    since = -float("inf") if last_learned is None else float(last_learned)
    return batches_since(store, system, name, since, now - delay)
