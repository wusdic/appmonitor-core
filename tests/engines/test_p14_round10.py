"""P14 views, round 10 (tree / views owner)."""
from __future__ import annotations

from types import SimpleNamespace as NS

from test_p14_round9 import _Lvl

from app.engines.behavior import views as VW

DAY = 86400.0


class _Sigs:
    def __init__(self, last):
        self.last = last

    def get(self, m):
        x = self.last.get(m)
        return None if x is None else NS(last=x, days=10)


def test_a_member_silent_for_weeks_is_not_a_part_of_the_node():
    """O-real R3 (round 9, day 35): 财务部's clerk left 192.168.2.11 for .51 on
    day 9, but the node's IP-level summary still tracked .11's decayed mass,
    so the G18 and 财务部 parts of the finance login / ledger nodes named .11
    (PG11 R3 old_still_live). A member whose own signature has been silent
    for two weeks is not a current source of the node."""
    now = 35 * DAY
    groups = {"G18": {"name": "G18·finance GET /fin/ledger",
                      "members": ["192.168.2.11", "192.168.2.12", "192.168.2.51"]},
              "G59": {"name": "财务部·finance GET /fin/approval/list", "members": ["192.168.2.10"]}}
    ip2g = {m: g for g, gr in groups.items() for m in gr["members"]}
    sigs = _Sigs({"192.168.2.10": now - 3600.0, "192.168.2.11": 9 * DAY,
                  "192.168.2.12": now - 3600.0, "192.168.2.51": now - 7200.0})
    c = NS(ip2g=ip2g, mode="grp", groups=groups, sigs=sigs, key="finance", config={})
    ips = [("192.168.2.12", 6.0), ("192.168.2.51", 5.0), ("192.168.2.10", 4.0), ("192.168.2.11", 0.3)]
    nd = NS(ctx=[], who=NS(levels=[_Lvl(ips), _Lvl([]), _Lvl([]), _Lvl([("grp:G18", 11.3), ("grp:G59", 4.0)])]))
    parts = VW.group_parts(c, nd, now, 0.0, "POST fin /fin/login")
    by = {p[0]: p for p in parts}
    assert by["G18"][1] == ["192.168.2.12", "192.168.2.51"]
    assert all("192.168.2.11" not in p[1] for p in parts)
    # still a part member while its silence is short (a holiday week)
    sigs.last["192.168.2.11"] = now - 8 * DAY
    parts = VW.group_parts(c, nd, now, 0.0, "POST fin /fin/login")
    assert "192.168.2.11" in {p[0]: p for p in parts}["G18"][1]
