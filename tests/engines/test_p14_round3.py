"""P14 round 3 (groups_views owner): the views against the requirement's wording.

用户视角 "综合部访问哪几个业务都干什么 …（从不干什么）", 业务系统视角 "OA 服务器的某类人
会在哪个时间段访问我什么页面干什么事". Pack O, round 2 (seed 0, day 21): a re-formed
DEV group's view said '在 oa 中从未执行写操作（…登录（POST /login））' next to its own
'访问 oa：登录（POST /login）'; no view said what a department never does inside
a system it uses; the finance approval statement named 192.168.2.10 without its
department; the finance username binding (3 users) was never stated
(BIND_MIN_CARD = 8 distinct values system-wide)."""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List

import pytest

from helpers import make_store

from app.engines.behavior import views as VW
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import ptree as PT
from app.models.schema import ORG, SYSTEM_ENTITY

from test_p14_views import CFG, DAY, GA, T0, TZ, USERS, _keys

LOGIN = "POST oa.corp /login"
APPROVE = "POST oa.corp /approval/{num}/approve"
FIN_APPROVE = "POST fin.corp /fin/approval/{num}/approve"
FIN_LOGIN = "POST fin.corp /fin/login"
APPROVER = "192.168.1.50"
FIN = {"192.168.2.10": "lucy", "192.168.2.11": "tom", "192.168.2.12": "kate"}


def _tree(key: str, routes: Dict[str, List[tuple]], days: int = 21) -> PT.PTreeModel:
    """routes: {route: [(ip, group)]}; one learned event per source and workday."""
    m = PT.PTreeModel(key)
    tr = m.tree(EV.KIND_TXN, T0, create=True)
    sp = tr.split(tr.root, "http.route", 0, [[r] for r in routes], T0)
    root = tr.nodes[tr.root]
    t = T0
    for (route, srcs), cid in zip(routes.items(), sp.children):
        node = tr.nodes[cid]
        for d in range(days):
            day0 = T0 + d * DAY
            if _dt.datetime.fromtimestamp(day0, TZ).weekday() >= 5:
                continue
            for k, (ip, g) in enumerate(srcs):
                t = day0 + (9 * 60 + 5 * k + 1) * 60.0
                dayn = int((t + 8 * 3600) // DAY)
                for nd in (root, node):
                    nd.update_core(t, 1.0, 1.0, _keys(ip, g), ip, 0, (t + 8 * 3600) % DAY / 60.0, dayn)
        node.state = "confirmed"
    root.state = "confirmed"
    m.t_last = t
    return m


def _route_key(r: str) -> str:
    return r


@pytest.fixture()
def org():
    st = make_store()
    oa = _tree("oa", {LOGIN: [(ip, "G1") for ip in GA], APPROVE: [(APPROVER, "G3")]})
    st.put_model("oa", SYSTEM_ENTITY, MP.PTREE, oa, version=1)
    fin = _tree("finance", {FIN_APPROVE: [("192.168.2.10", "G7")],
                            FIN_LOGIN: [(ip, "G6" if ip != "192.168.2.10" else "G7") for ip in FIN]})
    st.put_model("finance", SYSTEM_ENTITY, MP.PTREE, fin, version=1)
    groups = {
        "G1": {"id": "G1", "name": "综合部", "dept": "综合部", "members": list(GA), "covers": [],
               "systems": {"oa": 1.0}, "actions": {"oa": [{"action": LOGIN, "share": 1.0, "members": []}]}},
        "G3": {"id": "G3", "name": "G3·oa POST /approval", "members": [APPROVER], "covers": [],
               "systems": {"oa": 1.0}, "actions": {"oa": [{"action": APPROVE, "share": 1.0, "members": []}]}},
        # P11 re-formed under a new id: its members' rows at the login node carry
        # no grp:G5 label (the node summaries keep the label of learning time)
        "G5": {"id": "G5", "name": "研发", "members": ["10.50.0.7", "10.50.0.9"], "covers": [],
               "systems": {"oa": 1.0}, "actions": {"oa": [{"action": LOGIN, "share": 1.0, "members": []}]}},
        "G6": {"id": "G6", "name": "财务部", "dept": "财务部", "members": ["192.168.2.11", "192.168.2.12"],
               "covers": [], "systems": {"finance": 1.0},
               "actions": {"finance": [{"action": FIN_LOGIN, "share": 1.0, "members": []}]}},
        "G7": {"id": "G7", "name": "财务部·finance POST /fin/approval/{num}/approve", "dept": "财务部",
               "members": ["192.168.2.10"], "covers": [], "systems": {"finance": 1.0},
               "actions": {"finance": [{"action": FIN_APPROVE, "share": 0.8, "members": []},
                                       {"action": FIN_LOGIN, "share": 0.2, "members": []}]}}}
    ip2g = {m: g for g, gr in groups.items() for m in gr["members"]}
    st.put_model(ORG, ORG, MP.WHO_GROUPS, {"groups": groups, "ip2g": ip2g,
                                           "mode": {"oa": {"mode": "ip"}, "finance": {"mode": "ip"}}})
    for s in ("oa", "finance"):
        st.add_batch(s, EV.EVT_BATCH, T0 + 21 * DAY, EV.BatchBuilder(s).build(T0, T0 + 21 * DAY))
    st.now = max(oa.t_last, fin.t_last) + 3600.0
    return st


def _negs(v: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in v["statements"] if s["evidence"].get("negative")]


def test_never_statement_agrees_with_what_the_group_itself_does(org):
    v = VW.group_view(org, "G5", CFG, org.now)
    act = [s for s in v["statements"] if s["evidence"].get("activity")]
    assert act and "登录（POST /login）" in act[0]["text_zh"]
    # no "never wrote in oa" next to its own logins ...
    assert not [s for s in _negs(v) if s["evidence"]["target_system"] == "oa"
                and s["evidence"].get("scope", "system") == "system"]
    # ... but what it never does there is stated
    part = [s for s in _negs(v) if s["evidence"].get("scope") == "actions"]
    assert part and part[0]["text_zh"].startswith("研发 在 oa 中从未执行：审批（POST /approval/{num}/approve）")


def test_department_view_says_what_none_of_its_roles_does_in_a_used_system(org):
    eng = VW.ViewsEngine()
    g1 = VW.group_view(org, "G1", CFG, org.now)
    part = [s for s in _negs(g1) if s["evidence"].get("scope") == "actions"]
    assert part and "审批（POST /approval/{num}/approve）" in part[0]["text_zh"]
    # 财务部 = its voucher clerks + its approver (a one-address role): composed
    gv6 = VW.group_view(org, "G6", CFG, org.now)
    gv7 = VW.group_view(org, "G7", CFG, org.now)
    groups = MP.who_groups(org)["groups"]
    dv = VW.dept_view("财务部", [gv6, gv7], groups, CFG, org.now)
    act = [s for s in dv["statements"] if s["evidence"].get("activity")]
    assert act and "审批（POST /fin/approval/{num}/approve）[192.168.2.10]" in act[0]["text_zh"]
    assert eng is not None


def test_system_view_names_the_kind_of_people_and_what_they_do(org):
    v = VW.system_view(org, "finance", CFG, org.now)
    ap = [s for s in v["statements"] if s["evidence"]["route"] == FIN_APPROVE and "|grp:" not in s["id"]]
    assert ap and "财务部（192.168.2.10）访问 POST /fin/approval/{num}/approve（审批）" in ap[0]["text_zh"]
    assert ap[0]["evidence"]["who"]["level"] == "ip" and ap[0]["evidence"]["who"]["dept"] == "财务部"
    assert "opens POST /fin/approval/{num}/approve (approve)" in ap[0]["text_en"]


class _Reg:
    class _R:
        def __init__(self, n: int) -> None:
            self.n = n

        def card_estimate(self) -> float:
            return float(self.n)

    def __init__(self, cards: Dict[str, int]) -> None:
        self.cards = cards
        self.records: Dict[str, Any] = {}

    def get(self, name: str) -> Any:
        return self._R(self.cards[name]) if name in self.cards else None


def test_binding_of_a_small_department_is_stated(org):
    """Three finance users bound to their three addresses (3 values system-wide)
    are a binding; a body format every source shares is not."""
    tr = MP.get_ptree(org, "finance").kinds[0]
    nid = tr.nodes[tr.root].split.child_for(FIN_LOGIN)
    table = {ip: {"n": 40.0, "top": u, "k": 40.0, "bound": True, "LB": 0.9} for ip, u in FIN.items()}
    const = {ip: {"n": 40.0, "top": "form", "k": 40.0, "bound": True, "LB": 0.9} for ip in FIN}
    org.put_model("finance", SYSTEM_ENTITY, MP.PBIND, {"fmt": 1, "nodes": {0: {nid: {"status": "fitted", "pairs": {
        "net.src->body.kv.username": {"x": "net.src", "y": "body.kv.username", "dir": "fwd",
                                      "fd": {"g3": 0.0, "holds": True, "n": 120.0}, "table": table},
        "net.src->body.fmt": {"x": "net.src", "y": "body.fmt", "dir": "fwd",
                              "fd": {"g3": 0.0, "holds": True, "n": 120.0}, "table": const}}}}}})
    org.put_model("finance", SYSTEM_ENTITY, MP.ATTR, _Reg({"body.kv.username": 3, "body.fmt": 1}))
    v = VW.system_view(org, "finance", CFG, org.now)
    st = [s for s in v["statements"] if s["evidence"]["route"] == FIN_LOGIN and "|grp:" not in s["id"]][0]
    assert "body.kv.username" in st["evidence"]["bindings"]
    assert "body.fmt" not in st["evidence"]["bindings"]
    assert "192.168.2.10 → username=lucy" in st["text_zh"]


def test_group_view_states_the_groups_part_of_a_shared_node(org):
    """A node used by many sources renders its population ('来自 192.168.0.0/16
    （约 12 个 IP）'); the group's view of it states the group (its members)."""
    sales = [f"192.168.3.{i}" for i in range(20, 28)]
    # 综合部 reads mail several times a day (heavy hitters of the node's IP level)
    mail = _tree("mail", {"TLS mail.corp": [(ip, "G1") for ip in GA * 3] + [(ip, "G9") for ip in sales]})
    org.put_model("mail", SYSTEM_ENTITY, MP.PTREE, mail, version=1)
    org.add_batch("mail", EV.EVT_BATCH, T0 + 21 * DAY, EV.BatchBuilder("mail").build(T0, T0 + 21 * DAY))
    wg = dict(MP.who_groups(org))
    groups = dict(wg["groups"])
    groups["G1"] = dict(groups["G1"], systems={"oa": 0.7, "mail": 0.3})
    groups["G9"] = {"id": "G9", "name": "销售部", "members": sales, "covers": [], "systems": {"mail": 1.0}}
    org.put_model(ORG, ORG, MP.WHO_GROUPS, dict(wg, groups=groups,
                                                ip2g=dict(wg["ip2g"], **{ip: "G9" for ip in sales})))
    sv = VW.system_view(org, "mail", CFG, org.now)
    node_st = [s for s in sv["statements"] if "|grp:" not in s["id"]][0]
    assert node_st["evidence"]["who"]["level"] != "ip"
    v = VW.group_view(org, "G1", CFG, org.now)
    st = [s for s in v["statements"] if s["evidence"].get("route") == "TLS mail.corp"]
    assert st and st[0]["evidence"]["who"]["level"] == "grp"
    assert set(st[0]["evidence"]["who"]["members"]) == set(GA)
    assert "综合部（" in st[0]["text_zh"]


def _set_groups(org, extra: Dict[str, Any]) -> None:
    wg = dict(MP.who_groups(org))
    groups = dict(wg["groups"], **extra)
    ip2g = dict(wg["ip2g"])
    for g, gr in extra.items():
        ip2g.update({m: g for m in gr["members"]})
    org.put_model(ORG, ORG, MP.WHO_GROUPS, dict(wg, groups=groups, ip2g=ip2g))


def test_kind_of_people_for_a_pool_and_for_an_automatic_group(org):
    dev = [f"10.50.{k % 2}.{10 + k}" for k in range(12)]
    leases = [f"10.50.{k % 2}.{100 + k}" for k in range(5)]        # fresh leases, not grouped yet
    code = _tree("code", {"TLS git.corp.local": [(ip, "G5") for ip in dev] + [(ip, "∅") for ip in leases],
                          "GET code /health": [("192.168.9.9", "G16")]})
    org.put_model("code", SYSTEM_ENTITY, MP.PTREE, code, version=1)
    org.add_batch("code", EV.EVT_BATCH, T0 + 21 * DAY, EV.BatchBuilder("code").build(T0, T0 + 21 * DAY))
    _set_groups(org, {
        "G5": {"id": "G5", "name": "研发", "name_source": "config", "members": dev, "covers": ["10.50.0.0/22"],
               "pool": "10.50.0.0/22", "systems": {"code": 1.0}},
        "G16": {"id": "G16", "name": "G16·code GET /health", "name_source": "auto", "members": ["192.168.9.9"],
                "covers": [], "systems": {"code": 1.0}}})
    v = VW.system_view(org, "code", CFG, org.now)
    git = [s for s in v["statements"] if s["evidence"]["route"] == "TLS git.corp.local"][0]
    assert git["evidence"]["who"]["level"] in ("prefix", "reg")
    assert "研发（10.50." in git["text_zh"] and "代码库" in git["text_zh"]
    hc = [s for s in v["statements"] if s["evidence"]["route"] == "GET code /health"][0]
    assert "G16·" not in hc["text_zh"] and "192.168.9.9访问 GET /health（健康检查）" in hc["text_zh"]
