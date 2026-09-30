"""Small simulation harness for the progressive-core learners (P02, P05, P04):
synthetic behaviour events written straight into the store's batch series
(evt.batch with the ctx.* context columns P01 would add), one tick at a time.
Used by tests/engines/test_p04_pattern_tree.py and test_p05_attr_select.py."""
from __future__ import annotations

import datetime as _dt
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from helpers import ctx, make_store

from app.engines.behavior.attr_registry import AttributeRegistryEngine
from app.engines.behavior.attr_select import AttributeSelectionEngine
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.pattern_tree import PatternTreeEngine
from app.models.schema import SYSTEM_ENTITY

TZ = _dt.timezone(_dt.timedelta(hours=8))
MON = _dt.datetime(2026, 9, 28, tzinfo=TZ).timestamp()      # a Monday, 00:00 local
DAY = 86400.0
CFG = {"progressive": {"enabled": True}, "tz": "Asia/Shanghai"}

Event = Tuple[float, str, str, Dict[str, Any]]                 # (ts, system, ip, attrs)


def local(ts: float) -> _dt.datetime:
    return _dt.datetime.fromtimestamp(ts, TZ)


def is_workday(ts: float) -> bool:
    return local(ts).weekday() < 5


def ctx_attrs(ts: float) -> Dict[str, Any]:
    d = local(ts)
    minute = d.hour * 60 + d.minute + d.second / 60.0
    wd = d.weekday() < 5
    return {"ctx.tod_min": float(round(minute, 3)), "ctx.daytype": "workday" if wd else "nonworkday",
            "ctx.when": ("wd" if wd else "nwd", int(minute))}


class Sim:
    """Runs P02 (+ optionally P05) and P04 on synthetic events; `sel` pins
    model.attrsel for isolated P04 tests."""

    def __init__(self, dt: float = 900.0, sel: Optional[Mapping[str, Any]] = None,
                 p05: bool = False, config: Optional[Dict[str, Any]] = None, t0: float = MON) -> None:
        self.st = make_store()
        self.dt = float(dt)
        self.now = float(t0)
        self.sel = dict(sel) if sel is not None else None
        self.cfg = dict(CFG if config is None else config)
        self.p02 = AttributeRegistryEngine()
        self.p05 = AttributeSelectionEngine() if p05 else None
        self.p04 = PatternTreeEngine()
        self.pending: List[Event] = []
        self.hooks: List[Callable[["Sim"], None]] = []

    # --------------------------------------------------------------- events
    def add(self, events: Iterable[Event]) -> None:
        self.pending.extend(events)

    def tick(self) -> None:
        t1 = self.now + self.dt
        cur = [e for e in self.pending if self.now < e[0] <= t1]
        self.pending = [e for e in self.pending if e[0] > t1]
        by_sys: Dict[str, List[Event]] = {}
        for e in cur:
            by_sys.setdefault(e[1], []).append(e)
        for s, evs in by_sys.items():
            b = EV.BatchBuilder(s, EV.KIND_TXN)
            for ts, _, ip, attrs in sorted(evs, key=lambda x: x[0]):
                a = {"net.src": ip, "ev.ch": "http"}
                a.update(ctx_attrs(ts))
                a.update(attrs)
                w = float(a.pop("__w", 1.0))
                a.pop("__pi", None)
                b.add(ts, ip, a, w)
            batch = b.build(self.now, t1)
            pis = [float(e[3].get("__pi", 1.0)) for e in sorted(evs, key=lambda x: x[0])]
            batch.pi[:] = np.asarray(pis, dtype=np.float32)
            self.st.add_batch(s, EV.EVT_BATCH, t1, batch)
            if self.sel is not None:
                self.st.put_model(MP.tree_key(self.st, s), SYSTEM_ENTITY, MP.ATTRSEL, dict(self.sel), ts=t1)
        self.now = t1
        c = ctx(self.st, t1, window_s=self.dt, config=self.cfg)
        for h in self.hooks:
            h(self)
        self.p02.safe_run(c, None)
        if self.p05 is not None:
            self.p05.safe_run(c, None)
        self.p04.safe_run(c, None)

    def run_until(self, t_end: float) -> None:
        while self.now < t_end:
            self.tick()

    # ---------------------------------------------------------------- reads
    def tree(self, s: str = "oa", kind: int = EV.KIND_TXN):
        m = MP.get_ptree(self.st, MP.tree_key(self.st, s))
        return None if m is None else m.kinds.get(kind)

    def events(self, kind: str) -> List[Any]:
        return [e for e in self.st.events() if e.kind == kind]

    def splits(self, s: str = "oa", kind: int = EV.KIND_TXN) -> List[Tuple]:
        tr = self.tree(s, kind)
        return [] if tr is None else [x for x in tr.lineage if x[1] == "split"]


def daily(fn: Callable[[int, float, np.random.Generator], List[Event]], days: int,
          seed: int = 0, t0: float = MON) -> List[Event]:
    """Events of `days` days: fn(day index, day start ts, rng) -> events."""
    rng = np.random.default_rng(seed)
    out: List[Event] = []
    for d in range(days):
        out.extend(fn(d, t0 + d * DAY, rng))
    return out
