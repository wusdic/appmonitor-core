"""P13 Facets (`behavior.facets`) — dynamic facet registry and multi-facet
portrait composition (docs/lib3/progressive.md §6.17.1, card P13). Library 3.

Requirement S18/S19 ("不是一个引擎一个视角 … 多个算法从不同方面叠加起来 … 就像看一
个人有外观维度、生物学特征、社会属性"): the portrait of a system, a behavioural
group or an (earned) IP is a TREE of facets — functional, temporal, spatial,
content, sequential, relational, technical, volume, identity, risk — each
with sub-facets, each filled by the engines that produce it. Facets are
declared at runtime in model.facets (lib/pfacets.declare): an engine that
adds an aspect adds a declaration, and the next composition shows it; no code
change here. A facet is shown only where it applies (P12's characteristics:
no content facets on an opaque TLS system, no workflow facet without
identifiable sessions) and only when it has items.

Reads   model.facets (declarations), each declaration's sources, model.sysprof
        (applicability), model.pviews (P14 statements tagged with facets),
        model.who_groups (groups), model.budget (earned IPs, ladder step 6).
Writes  profile.extra['facets'] of (s, '__system__') for every system, of
        ('__org__', 'class:grp:<g>') for the G_MAX largest groups, and of (s, ip)
        for EARNED IPs only (every other IP is composed on read with
        `compose_ip`, as B30 does in bounded mode: O(earned), not O(#IPs));
        a profile version when a subject's facet tree changes; the default
        declarations under the owner 'behavior.facets' in model.facets.
Cadence 2 h per subject (entity_due, crc32 phase); under ladder step 6 of
        P15 only on read.
Inert unless config['progressive']['enabled'].
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from ...core.engine import Context, Engine
from ...models.schema import ORG, SYSTEM_ENTITY, EntityProfile
from .lib import m_ptree as MP
from .lib import pevent as EV
from .lib import pfacets as PFc

PERIOD_S = 2 * 3600.0
G_MAX = 256
GROUP_PREFIX = "class:grp:"
OWNER = "behavior.facets"


def _statements(view: Any) -> List[Mapping[str, Any]]:
    """Statements of a P14 view model (flat list or per-action lists)."""
    if not isinstance(view, Mapping):
        return []
    out = list(view.get("statements") or [])
    acts = view.get("actions") or []
    for a in (acts.values() if isinstance(acts, Mapping) else acts):
        if isinstance(a, Mapping):
            out += list(a.get("statements") or [])
    seen, uniq = set(), []
    for st in out:
        if not isinstance(st, Mapping):
            continue
        k = st.get("id") or id(st)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(st)
    return uniq


def compose_system(store: Any, key: str, now: float) -> Dict[str, Any]:
    sp = MP.get_model(store, key, MP.SYSPROF)
    subj = PFc.Subject(store, "system", now, system=key, sysprof=sp)
    return PFc.compose(subj, PFc.registry(store), _statements(MP.get_model(store, key, MP.PVIEWS)))


def compose_group(store: Any, gid: str, now: float) -> Dict[str, Any]:
    subj = PFc.Subject(store, "group", now, gid=gid)
    view = store.get_model(ORG, f"{GROUP_PREFIX}{gid}", MP.PVIEWS)
    return PFc.compose(subj, PFc.registry(store), _statements(view))


def compose_ip(store: Any, s: str, ip: str, now: float) -> Dict[str, Any]:
    """An IP's facet tree (on read for unearned IPs): its system's applicability,
    its group's statements and its own statements (exceptions, bindings)."""
    key = MP.tree_key(store, s)
    sp = MP.get_model(store, key, MP.SYSPROF)
    subj = PFc.Subject(store, "ip", now, system=key, entity=ip, sysprof=sp)
    wg = MP.who_groups(store)
    g = (wg.get("ip2g") or {}).get(ip)
    sts = []
    if g is not None:
        sts += _statements(store.get_model(ORG, f"{GROUP_PREFIX}{g}", MP.PVIEWS))
    own = [st for st in _statements(MP.get_model(store, key, MP.PVIEWS))
           if ip in str((st.get("evidence") or {}).get("who") or "")]
    return PFc.compose(subj, PFc.registry(store), sts + own)


class FacetsEngine(Engine):
    name = "behavior.facets"
    layer = "behavior"
    consumes = [PFc.FACETS, MP.SYSPROF, MP.PVIEWS, MP.WHO_GROUPS, MP.BUDGET]
    produces = ["profile.extra.facets", PFc.FACETS]
    description = "P13: dynamic facet registry and multi-facet portraits (system, group, earned IP)"
    interval = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.period = float(params.get("facet_period_s", PERIOD_S))
        self.last_stats: Dict[str, Any] = {}

    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        if not EV.enabled(ctx.config):
            return 0
        store = ctx.store
        now = float(ctx.now)
        self._declare_defaults(store, now)
        bud = MP.get_org_model(store, MP.BUDGET) or {}
        if isinstance(bud, Mapping) and (bud.get("ladder") or {}).get("refresh_on_read"):
            self.last_stats = {"skipped": "refresh_on_read"}
            return 0
        n = 0
        systems = sorted(set(store.batch_systems(EV.EVT_BATCH)))
        keys: Dict[str, List[str]] = {}
        for s in systems:
            keys.setdefault(MP.tree_key(store, s), []).append(s)
        for key, members in keys.items():
            if MP.get_model(store, key, MP.SYSPROF) is None and MP.get_ptree(store, key) is None:
                continue
            if not self.entity_due(("p13", key), now, self.period):
                continue
            tree = compose_system(store, key, now)
            for s in members:
                n += self._write(store, s, SYSTEM_ENTITY, tree, now)
        wg = MP.who_groups(store)
        groups = wg.get("groups") or {}
        sizes = sorted(groups, key=lambda g: -len((groups.get(g) or {}).get("members") or ()))[:G_MAX]
        for g in sizes:
            if not self.entity_due(("p13g", g), now, self.period):
                continue
            n += self._write(store, ORG, f"{GROUP_PREFIX}{g}", compose_group(store, g, now), now)
        for s in systems:
            rec = ((bud.get("systems") or {}).get(s) or {}) if isinstance(bud, Mapping) else {}
            for ip in rec.get("earned") or ():
                if not self.entity_due(("p13ip", s, ip), now, self.period):
                    continue
                n += self._write(store, s, ip, compose_ip(store, s, ip, now), now)
        self.last_stats = {"composed": n}
        return n

    @staticmethod
    def _declare_defaults(store: Any, now: float) -> None:
        m = store.get_model(ORG, ORG, PFc.FACETS)
        if isinstance(m, Mapping) and OWNER in (m.get("decls") or {}):
            return
        m = dict(m) if isinstance(m, Mapping) else {"version": 0, "decls": {}}
        decls = dict(m.get("decls") or {})
        decls[OWNER] = {d["id"]: dict(d) for d in PFc.DEFAULT_FACETS}
        m["decls"] = decls
        m["version"] = int(m.get("version", 0)) + 1
        store.put_model(ORG, ORG, PFc.FACETS, m, version=m["version"], ts=now)

    @staticmethod
    def _write(store: Any, s: str, e: str, tree: Mapping[str, Any], now: float) -> int:
        prof = store.profile(s, e)
        if prof is None:
            prof = EntityProfile(system=s, entity=e, updated=now)
        old = (prof.extra or {}).get("facets") if isinstance(prof.extra, dict) else None
        prof.extra = dict(prof.extra or {})
        prof.extra["facets"] = dict(tree)
        store.put_profile(prof)
        if not isinstance(old, Mapping) or old.get("hash") != tree.get("hash"):
            ver = int((old or {}).get("version", 0)) + 1 if isinstance(old, Mapping) else 1
            prof.extra["facets"]["version"] = ver
            store.put_profile_version(s, e, f"facets.{ver}", {"kind": "facets", "ts": now,
                                                               "facets": tree.get("facets")}, ts=now)
        else:
            prof.extra["facets"]["version"] = old.get("version", 1)
        return 1
