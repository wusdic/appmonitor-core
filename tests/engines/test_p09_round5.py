"""P09 time windows, round 5 (diagnosed on pack O, round-4 code):
regression tests that fail on the round-4 code."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.engines.behavior import time_window as TW

T0 = 1_757_000_000.0
DAY = 86400.0


class _Res:
    def __init__(self, items):
        self._items = items

    def items(self):
        return list(self._items)

    def __len__(self):
        return len(self._items)


def _node(nid, parent, ctx, items, level0):
    who = SimpleNamespace(levels=[{ip: 1.0 for ip in level0}], suspects=lambda t: [])
    return SimpleNamespace(id=nid, parent=parent, ctx=ctx, when=SimpleNamespace(res=_Res(items)), who=who)


def test_a_group_split_child_reads_every_member_of_its_group_from_the_ancestor():
    """Pack O seed 0, day 21: the 销售部 part of OA's GET /docs (context
    grp G10) backed off to its route node's reservoir through the 8 addresses
    its who summary tracked at level 0, not the group's 20: the late
    arrivals of the others were missing and its window ended at 16:59."""
    rng = np.random.default_rng(0)
    sales = [f"192.168.3.{i}" for i in range(20, 40)]
    items = []
    for d in range(10):
        for ip in sales:
            m = float(rng.uniform(570, 1050))
            items.append(((0, m, ip), 1.0, T0 + d * DAY + m * 60.0))
    route_ctx = [("http.route", 0, frozenset({"GET oa /docs"}), False)]
    anc = _node(5, None, route_ctx, items, sales[:8])
    child = _node(25, 5, route_ctx + [("net.src", 3, frozenset({"grp:G10"}), False)], [], sales[:8])
    tree = SimpleNamespace(nodes={5: anc, 25: child})
    ip2g = {ip: "G10" for ip in sales}
    pts, a = TW._backoff(tree, child, 0, [], T0 + 11 * DAY, None, ip2g)
    assert a == 5
    assert {p[3] for p in pts} == set(sales)
    assert len(pts) == len(items)
    # an address of another group is not the child's
    ip2g["192.168.3.39"] = "G7"
    pts, _ = TW._backoff(tree, child, 0, [], T0 + 11 * DAY, None, ip2g)
    assert "192.168.3.39" not in {p[3] for p in pts}
