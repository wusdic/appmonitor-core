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
    # (round 12) P11 saw the clerk move (readdressed .11 -> .51): the colleague
    # is 财务部's, not 综合部's, although G18 holds one address of each
    c = NS(ip2g=ip2g, mode="grp", groups=groups, sigs=sigs, key="finance", config=cfg,
           readdr={"192.168.2.11": "192.168.2.51"})
    # .51 is a suspect at the node (P04 keeps its rows out of the IP level)
    ips = [("192.168.2.12", 6.0), ("192.168.2.10", 5.0), ("192.168.2.11", 0.3)]
    nd = NS(ctx=[], who=NS(levels=[_Lvl(ips), _Lvl([]), _Lvl([]), _Lvl([("grp:G18", 6.0), ("grp:G59", 5.0)])]))
    parts = VW.group_parts(c, nd, now, 0.0, ROUTE)
    by = {p[0]: p for p in parts}
    assert by["cfg:财务部"][1] == ["192.168.2.10", "192.168.2.12", "192.168.2.51"]
    assert all("192.168.2.51" not in p[1] for k, p in by.items() if k == "cfg:综合部")
    # never another configured department's address, nor one without the action
    assert all("192.168.1.21" not in by["cfg:财务部"][1] and "192.168.2.77" not in p[1] for p in parts)


def test_a_group_part_keeps_its_sources_earlier_extremes():
    """Item 4 (round 10, pack O seed 1): the 综合部 login node (a /24 split's
    child, 'all within 643-2743 B' from the extremes its sources' records
    carried into it) was merged back into its parent on day 18 (split saving
    < 0); 综合部 was then a group part of the parent, its range refitted on the
    group's held-out rows of the last days only (P04 'gnum': 1038-2743 B), and
    the 643 B login of day 2 fell out of the statement (truth 0.5-3 KB). The
    part's range now takes the group's sources' extremes the node keeps per
    source (pnode.SourceExtremes), within the node's confidence segment."""
    import numpy as np
    from helpers import make_store
    from app.engines.behavior.lib import pbounds as PB
    from app.engines.behavior.lib import pnode as PN
    cfg = {"progressive": {"enabled": True}, "tz": "Asia/Shanghai"}
    t = 1.7e9
    day = PB.local_day(t, cfg)
    rng = np.random.default_rng(1)
    g = PN.NumSummary(log=True)
    allv = PN.NumSummary(log=True)
    allv.seed_extreme(600.0, day - 19, day, extend=True)       # the node's own segment from day 2
    for d in range(4, 0, -1):
        for i in range(8):
            v = float(rng.uniform(1038.0, 2743.0))
            ts = t - d * 86400.0 + i * 60.0
            g.update(v, ts, 1.0, 1.0, day=day - d)
            allv.update(v, ts, 1.0, 1.0, day=day - d)
    s24 = PN.SourceExtremes()
    s24.note("192.168.1.0/24", "body.len", 643.0, day - 19, t - 19 * 86400.0, rep="192.168.1.21")
    s24.note("192.168.1.0/24", "body.len", 2100.0, day - 18, t - 18 * 86400.0, rep="192.168.1.21")
    s24.note("192.168.3.0/24", "body.len", 300.0, day - 10, t - 10 * 86400.0, rep="192.168.3.20")
    # a member's 12 KB row inside the group's own record (a violation P06 cleaned from the node)
    sx = PN.SourceExtremes()
    sx.note("192.168.1.23", "body.len", 1200.0, day - 2, t - 2 * 86400.0)
    sx.note("192.168.1.23", "body.len", 12288.0, day - 2, t - 2 * 86400.0 + 60.0)
    nd = NS(meta={"gnum": {"grp:G9": {"body.len": g}}, "sext24": s24, "sext": sx}, targets={"body.len": allv},
            mass_at=lambda t_: 400.0, n_c=lambda t_: 400.0, n_m=lambda t_: 400.0)
    c = NS(ip2g={"192.168.1.21": "G9", "192.168.1.23": "G9", "192.168.3.20": "G2"}, reg=None,
           config=cfg, store=make_store(), key="oa")
    ent = {"attrs": {"body.len": {"kind": "num", "unit": "B", "range": [600.0, 2743.0]}}}
    rec = VW.part_content(c, nd, ent, ["G9"], t)["attrs"]["body.len"]
    assert rec.get("part") and rec["range"][0] <= 643.0 + 1e-6, rec.get("range")
    assert rec["range"][0] > 300.0                                # never another group's source
    assert rec["range"][1] < 3000.0                               # nor a row inside its own record
    # an extreme before the node's own segment of the attribute is not taken
    allv.seg_day = day - 5
    rec = VW.part_content(c, nd, ent, ["G9"], t)["attrs"]["body.len"]
    assert rec["range"][0] >= 1038.0 - 1e-6, rec.get("range")
