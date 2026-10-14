"""P09 time windows, round 7 (time / workflow / content / views owner):
regression tests that fail on the round-6 code (52e813f)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.engines.behavior import time_window as TW
from app.engines.behavior.lib import pwindows as PW

T0 = 1_757_030_400.0 - 8 * 3600.0          # a local midnight (UTC+8)
DAY = 86400.0
OFF = 8 * 3600.0
GA = ["10.168.7.121", "192.168.1.21", "192.168.1.23"]


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


def test_backoff_reads_the_ancestor_with_the_longest_dated_history():
    """Pack O seed 3, day 16: the 综合部 login node (created after D1) backed
    off to its parent, whose reservoir had started on day 10 and held GA's
    logins of five dates; regime_cut needs three dates before a change, so
    the node stated the union of both laws (08:35-09:23, IoU 0.30 with the
    new 08:30-08:51). The grandparent held the same sources since day 0."""
    rng = np.random.default_rng(5)
    old, new = [], []
    for d in range(16):
        if d % 7 in (5, 6):
            continue                                  # weekends
        lo = 540.0 if d < 11 else 510.0
        for ip in GA:
            m = float(np.floor(rng.uniform(lo, lo + 21.0)))
            it = ((0, m, ip), 1.0, T0 + d * DAY + m * 60.0)
            old.append(it)
            if d >= 9:
                new.append(it)                         # the parent's young reservoir
    route = [("http.route", 0, frozenset({"POST oa /login"}), False)]
    root = _node(3, None, route, old, GA)
    par = _node(18, 3, route + [("net.src", 1, frozenset({"10.50.0.0/24"}), True)], new, GA)
    kid = _node(21, 18, par.ctx + [("net.src", 3, frozenset({"grp:G10"}), False)], [], GA)
    tree = SimpleNamespace(nodes={3: root, 18: par, 21: kid})
    ip2g = {ip: "G10" for ip in GA}
    pts, anc = TW._backoff(tree, kid, 0, [], T0 + 16 * DAY, None, ip2g)
    assert anc == 3 and len(pts) == len(old)
    h = np.zeros(PW.SLOTS)
    for p in pts:
        h[int(p[0] // 15)] += p[2]
    rec = PW.fit_regime(h, float(len(pts)), pts, tz_offset_s=OFF)
    assert rec.get("regime") == "new"
    assert all(505 <= s and e <= 540 for s, e in rec["windows"])
