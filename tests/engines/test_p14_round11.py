"""P14 views, round 11 (tree / views owner)."""
from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np

from test_p14_round9 import _Lvl

from app.engines.behavior import views as VW

DAY = 86400.0
ROUTE = "POST fin /fin/login"


class _Items:
    def __init__(self, keys):
        self.keys = list(keys)

    def key_of(self, i):
        return self.keys[i]


class _Sigs:
    """P11 signatures: {ip: (last row ts, {item: weight})}."""

    def __init__(self, sig):
        self.sig = sig
        self.items = _Items(sorted({k for _, it in sig.values() for k in it}))

    def get(self, ip):
        x = self.sig.get(ip)
        return None if x is None else NS(last=x[0], days=20)

    def weights(self, ip, t):
        it = self.sig[ip][1]
        ids = np.array([self.items.keys.index(k) for k in sorted(it)], dtype=np.int64)
        return ids, np.array([it[k] for k in sorted(it)], dtype=float)


def test_a_configured_department_states_its_unlisted_colleague_at_the_node():
    """O-real R3 (round 10, seed 0): 财务部's clerk moved from the listed
    192.168.2.11 to the unlisted .51 on day 9. P11 put .51 into G18 with its
    colleague .12 and G18's part of the finance login named it, but the
    configured 财务部 part listed only the configured addresses (.10 / .12
    once .11 fell silent), so FIN.finance.login#0@1 (who .10 / .12 / .51) was
    never found. The department's part now states the colleague its members'
    learned groups hold when its own signature recurs on the action; an
    address of another configured department, or one without the action, is
    not added."""
    now = 35 * DAY
    groups = {"G18": {"name": "G18·finance GET /fin/ledger",
                      "members": ["192.168.1.21", "192.168.2.12", "192.168.2.51", "192.168.2.77"]},
              "G59": {"name": "财务部·finance GET /fin/approval/list", "members": ["192.168.2.10"]},
              "G82": {"name": "G82", "members": ["192.168.2.11"]}}
    ip2g = {m: g for g, gr in groups.items() for m in gr["members"]}
    login, ledger = f"finance|{ROUTE}", "finance|GET fin /fin/ledger"
    sigs = _Sigs({"192.168.2.10": (now - 3600.0, {login: 1.0, ledger: 3.0}),
                  "192.168.2.11": (9 * DAY, {login: 1.0, ledger: 3.0}),
                  "192.168.2.12": (now - 3600.0, {login: 1.0, ledger: 3.0}),
                  "192.168.2.51": (now - 7200.0, {login: 1.0, ledger: 3.0}),
                  "192.168.1.21": (now - 3600.0, {login: 1.0, ledger: 3.0}),
                  "192.168.2.77": (now - 3600.0, {ledger: 5.0})})
    cfg = {"who_group_names": [{"name": "财务部", "ips": ["192.168.2.10", "192.168.2.11", "192.168.2.12"]},
                               {"name": "综合部", "ips": ["192.168.1.21"]}]}
    c = NS(ip2g=ip2g, mode="grp", groups=groups, sigs=sigs, key="finance", config=cfg)
    # .51 is a suspect at the node (P04 keeps its rows out of the IP level)
    ips = [("192.168.2.12", 6.0), ("192.168.2.10", 5.0), ("192.168.2.11", 0.3)]
    nd = NS(ctx=[], who=NS(levels=[_Lvl(ips), _Lvl([]), _Lvl([]), _Lvl([("grp:G18", 6.0), ("grp:G59", 5.0)])]))
    parts = VW.group_parts(c, nd, now, 0.0, ROUTE)
    by = {p[0]: p for p in parts}
    assert by["cfg:财务部"][1] == ["192.168.2.10", "192.168.2.12", "192.168.2.51"]
    # never another configured department's address, nor one without the action
    assert all("192.168.1.21" not in by["cfg:财务部"][1] and "192.168.2.77" not in p[1] for p in parts)
