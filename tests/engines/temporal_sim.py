"""Test harness for the temporal engines (P09 time windows, P10 workflows).

P09 reads only the when-summaries that P04 keeps per pattern node; to test it
in isolation this harness builds an ORACLE pattern tree (root -> one child per
route -> optionally one child per IP group) with lib/ptree and counts events
into it exactly as P04 does for the when summary (Node.update_core on every
node of the event's path, minute reservoirs created on P09's request, as
P04's maintenance step does). P10 reads evt.batch batches, which `batch()`
builds with pevent.BatchBuilder."""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import phier as PH
from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.lib import ptree as PT
from app.models.schema import SYSTEM_ENTITY

TZ = _dt.timezone(_dt.timedelta(hours=8))
MON = _dt.datetime(2026, 9, 28, tzinfo=TZ).timestamp()      # a Monday, 00:00 local (+08:00)
DAY = 86400.0
CFG = {"progressive": {"enabled": True}, "tz": "Asia/Shanghai"}


def local(ts: float) -> _dt.datetime:
    return _dt.datetime.fromtimestamp(ts, TZ)


def is_workday(ts: float) -> bool:
    return local(ts).weekday() < 5


def minute_of(ts: float) -> float:
    d = local(ts)
    return d.hour * 60 + d.minute + d.second / 60.0


class OracleTree:
    """root -> route children -> (optional) group children of IP lists."""

    def __init__(self, store: Any, key: str, routes: Sequence[str],
                 groups: Optional[Mapping[str, Sequence[Sequence[str]]]] = None, t0: float = MON) -> None:
        self.store, self.key = store, key
        self.m = PT.PTreeModel(key)
        self.tree = self.m.tree(EV.KIND_TXN, t0)
        self.hier = PH.Hierarchies(registry={})
        sp = self.tree.split(self.tree.root, "http.route", 0, [[r] for r in routes], t0)
        self.route_node = {r: c for r, c in zip(routes, sp.children)}
        self.group_node: Dict[Tuple[str, int], int] = {}
        for r, gs in (groups or {}).items():
            n = self.route_node[r]
            sp2 = self.tree.split(n, "net.src", 0, [list(g) for g in gs], t0)
            for j, c in enumerate(sp2.children):
                self.group_node[(r, j)] = c
        store.put_model(key, SYSTEM_ENTITY, MP.PTREE, self.m, version=1, ts=t0)

    def learn(self, ts: float, ip: str, route: str, mass: float = 1.0, evidence: float = 1.0) -> List[int]:
        get = {"http.route": route, "net.src": ip}.get
        path = self.tree.route(lambda a: get(a, EV.ABSENT), self.hier)
        self.store.register_entity(self.key, ip)              # the system is known, as in a pipeline
        keys = [self.hier.gen("net.src", l, ip) for l in range(PN.WHO_LEVELS)]
        d = local(ts)
        daytype = 0 if d.weekday() < 5 else 1
        day = d.date().toordinal()
        for nid in path:
            self.tree.nodes[nid].update_core(ts, mass, evidence, keys, ip, daytype, minute_of(ts), day)
        return path

    def apply_wants(self) -> None:
        """P04's maintenance step: create the minute reservoirs P09 asked for."""
        want = MP.get_model(self.store, self.key, MP.PWANT) or {}
        mr = ((want.get("p09") or {}).get("minute_reservoir") or {})
        for nid in mr.get(EV.KIND_TXN, mr.get(str(EV.KIND_TXN), [])) or []:
            nd = self.tree.nodes.get(int(nid))
            if nd is not None and nd.when.res is None:
                nd.when.want_minutes(True, seed=int(nid))


def batch(system: str, rows: Iterable[Tuple[float, str, Mapping[str, Any]]], t0: float, t1: float,
          learn: Optional[Sequence[bool]] = None) -> EV.EventBatch:
    """An evt.batch from (ts, ip, attrs) rows (http.route etc. in attrs)."""
    b = EV.BatchBuilder(system, EV.KIND_TXN)
    rows = sorted(rows, key=lambda r: r[0])
    for ts, ip, attrs in rows:
        a = {"net.src": ip}
        a.update(attrs)
        b.add(ts, ip, a, float(a.pop("__w", 1.0)))
    bt = b.build(t0, t1)
    if learn is not None:
        bt.learn[:] = np.asarray(learn, dtype=bool)
        bt.pi[:] = np.where(bt.learn, 1.0, 0.0).astype(np.float32) + 1e-6
    return bt
