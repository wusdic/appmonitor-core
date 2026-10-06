"""P14 views, round 5 (diagnosed on pack O, round-4 code): regression tests that
fail on the round-4 code."""
from __future__ import annotations

from app.engines.behavior import views as VW
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import ptree as PT

from test_p14_views import T0

HEALTH = "GET oa.corp /health"
LOGIN = "POST oa.corp /login"


def _route_split_tree():
    m = PT.PTreeModel("oa")
    tr = m.tree(EV.KIND_TXN, T0, create=True)
    sp = tr.split(tr.root, "http.route", 0, [[HEALTH], [LOGIN]], T0)
    return tr, sp.children


def test_a_node_split_on_the_route_stands_for_no_single_action():
    """Pack O round 4: the OA / finance roots (split on http.route) were stated
    as 'GET /health' because the 60-s monitor held >= 90 % of their mass and
    evidence; their group parts (销售部 'GET /health') had no support."""
    tr, (h, lg) = _route_split_tree()
    rd = {tr.root: {HEALTH: (95.0, 95.0), LOGIN: (5.0, 5.0)},
          h: {HEALTH: (95.0, 95.0)}, lg: {LOGIN: (5.0, 5.0)}}
    walked = {nd.id: (route, act) for nd, route, act in VW._walk(tr, T0, rd)}
    assert walked[tr.root] == (None, None)
    assert walked[h] == (HEALTH, h)
    assert walked[lg] == (LOGIN, lg)


class _Lvl:
    def __init__(self, items):
        self._items = items                       # [(key, count)]

    def total(self, t):
        return float(sum(c for _, c in self._items))

    def items(self, t, *a):
        return [(k, c, c, 0.0) for k, c in self._items]


def test_a_configured_departments_busy_member_is_in_its_part():
    """Pack O seed 0, day 17: 192.168.2.12 posted OA comments as often as its
    colleagues (3.8 decayed events; .10 5.0, .11 6.5) but they were 1.7 % of
    a signature dominated by its finance work (< SIG_STANDING 2 %), and
    财务部's part of POST /docs/{num}/comment listed .10 and .11 only (PG1 who
    of FIN.oa.documents#2 failed on every day)."""
    from types import SimpleNamespace
    from app.engines.behavior import conformity as CF
    from app.engines.behavior.lib import pminhash as MH
    route = "POST oa /docs/{num}/comment"
    sigs = MH.SigStore()
    fin = ["192.168.2.10", "192.168.2.11", "192.168.2.12"]
    sales = ["192.168.3.20", "192.168.3.21"]
    for d in range(10):
        t = T0 + d * 86400.0 + 36000.0
        for ip, comments, other in ((fin[0], 1.0, 10.0), (fin[1], 1.0, 10.0), (fin[2], 0.8, 60.0),
                                    (sales[0], 1.0, 10.0), (sales[1], 1.0, 10.0)):
            sigs.add(ip, f"oa|{route}", t, comments, 1.0, d)
            sigs.add(ip, "oa|GET oa /docs", t, 5.0, 1.0, d)
            sigs.add(ip, "finance|GET fin /fin/ledger" if ip.startswith("192.168.2.") else "crm|GET crm /x",
                     t, other, 1.0, d)
    t = T0 + 10 * 86400.0
    assert CF.signature_share(sigs, "oa", fin[2], t, route) < CF.SIG_STANDING
    assert CF.signature_share(sigs, "oa", fin[2], t, route, within_key=True) >= CF.SIG_STANDING
    groups = {"G8": {"dept": "财务部", "members": [fin[0]], "name": "财务部·x"},
              "G34": {"dept": "财务部", "members": fin[1:], "name": "财务部"},
              "G10": {"dept": "销售部", "members": sales, "name": "销售部"}}
    ip2g = {fin[0]: "G8", fin[1]: "G34", fin[2]: "G34", sales[0]: "G10", sales[1]: "G10"}
    c = SimpleNamespace(ip2g=ip2g, mode="ip", groups=groups, sigs=sigs, key="oa",
                        config={"who_group_names": [{"name": "财务部", "ips": fin}]})
    nd = SimpleNamespace(ctx=[], who=SimpleNamespace(levels=[
        _Lvl([(fin[0], 3.0), (fin[1], 3.0), (sales[0], 3.0)]), _Lvl([]), _Lvl([]),
        _Lvl([("grp:G8", 3.0), ("grp:G34", 4.0), ("grp:G10", 6.0)])]))
    parts = VW.group_parts(c, nd, t, 0.0, route)
    fin_part = next(p for p in parts if p[0] == "dept:财务部")
    assert fin_part[1] == fin
