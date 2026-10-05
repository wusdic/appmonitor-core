"""P14 round 4 (conformity / views owner): readable profiles in both views.

Pack O round 3-4 (seed 0 day 14, tree-round-4 code): the system view stated the
same group's part of an action twice - at the all-department GET /home and
POST /login nodes and again at their office-subnet children (6 duplicate
statements on seed 0, 38 of 90 statements on seed 1); the OA view opened with a
one-address health monitor; the sentences stated a confidence but no support,
group activity statements neither; 综合部's department view listed 11 actions
and held a statement for 4 of them; a listed who covering 90-95 % of the
sources stated U ~ 0.001 (a 99.9 % claim); a group's part stated its node's
p_hold although P04 now tests each group's own held-out events."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from helpers import make_store

from app.engines.behavior import views as VW
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import prender as PR

from test_p14_views import DAY, T0, store  # noqa: F401  (fixture)


def _st(sid: str, act: int, node: int, route: str, who: Dict[str, Any], depth: int = 1,
        mass: float = 10.0, conf: float = 0.5) -> Dict[str, Any]:
    return {"id": sid, "act_node": act, "mass": mass, "confidence": conf, "support": mass,
            "text_zh": sid, "text_en": sid,
            "evidence": {"route": route, "node": node, "depth": depth, "who": who}}


def _tree(edges: Dict[int, tuple]) -> Any:
    """edges: {child: (parent, split attribute of the parent)}."""
    nodes: Dict[int, Any] = {0: SimpleNamespace(parent=None, split=None)}
    for c, (p, attr) in sorted(edges.items()):
        nodes.setdefault(c, SimpleNamespace(parent=p, split=None))
        nodes[c].parent = p
        nodes.setdefault(p, SimpleNamespace(parent=None, split=None))
        nodes[p].split = SimpleNamespace(attr=attr)
    return SimpleNamespace(nodes=nodes)


SALES = {"level": "grp", "name": "销售部", "members": [f"192.168.3.{i}" for i in range(20, 40)]}
GA = {"level": "grp", "name": "综合部", "members": ["192.168.1.21", "192.168.1.23", "10.168.7.121"]}


def test_a_groups_part_at_a_node_and_at_its_source_split_child_is_stated_once():
    """The office-subnet child of the all-department login node (split on
    net.src) refines the same people's behaviour: the 销售部 / 综合部 parts of
    the parent are duplicates of the child's and are dropped; the parent's own
    statement (a different who: all departments + 研发) stays."""
    tree = _tree({4: (1, "net.src")})
    allw = {"level": "prefix", "items": ["192.168.0.0/16", "10.50.0.0/16"]}
    sts = [_st("p:4", 1, 1, "GET /home", allw),
           _st("p:4|grp:G11", 1, 1, "GET /home", SALES),
           _st("p:4|grp:dept:综合部", 1, 1, "GET /home", GA),
           _st("p:16", 1, 4, "GET /home", {"level": "prefix", "items": ["192.168.3.0/24", "192.168.1.0/24"]}, 2),
           _st("p:16|grp:G11", 1, 4, "GET /home", SALES, 2),
           _st("p:16|grp:dept:综合部", 1, 4, "GET /home", GA, 2)]
    prim, folded = VW.fold_duplicates(sts, tree)
    assert {s["id"] for s in prim} == {"p:4", "p:16", "p:16|grp:G11", "p:16|grp:dept:综合部"}
    # the parent's parts stay published as folded alternatives of the child's
    assert {(s["id"], s["folded_into"]) for s in folded} == {
        ("p:4|grp:G11", "p:16|grp:G11"), ("p:4|grp:dept:综合部", "p:16|grp:dept:综合部")}


def test_a_named_subset_of_a_department_is_not_its_duplicate():
    """'综合部（192.168.1.21、192.168.1.23）' at a source-split child is two of the
    department's three people: the parent's 综合部 part (all three) stays
    primary (offline rescoring seed 1: name-only identity dropped it and lost
    GA.oa.documents)."""
    tree = _tree({27: (5, "net.src")})
    sub = dict(GA, members=["192.168.1.21", "192.168.1.23"])
    sts = [_st("p:5|grp:dept:综合部", 5, 5, "GET /docs", GA), _st("p:27|grp:dept:综合部", 5, 27, "GET /docs", sub, 2)]
    prim, folded = VW.fold_duplicates(sts, tree)
    assert len(prim) == 2 and not folded


def test_variants_below_a_content_split_are_not_duplicates():
    """Children of a content / time split are variants of the behaviour (the
    siblings hold the rest): the parent's statement and the children's stay
    even with the same who (offline rescoring seed 1: dropping them cost 2.6-4.8
    recall points)."""
    tree = _tree({5: (1, "net.bytes_down"), 6: (1, "net.bytes_down")})
    who = {"level": "prefix", "items": ["10.60.0.0/16"]}
    sts = [_st("p:1", 1, 1, "GET /news/{num}", who),
           _st("p:5", 1, 5, "GET /news/{num}", who, 2),
           _st("p:6", 1, 6, "GET /news/{num}", who, 2)]
    assert len(VW.dedup_statements(sts, tree)) == 3
    # ... but below a source split of the content split's child, the same who is a duplicate
    tree = _tree({5: (1, "net.bytes_down"), 7: (5, "net.src")})
    sts = [_st("p:5", 1, 5, "GET /news/{num}", who, 2), _st("p:7", 1, 7, "GET /news/{num}", who, 3)]
    assert [s["id"] for s in VW.dedup_statements(sts, tree)] == ["p:7"]


def test_statements_are_ordered_by_how_many_sources_perform_the_action():
    """A one-address health monitor with 10 000 events no longer heads the
    view; within an action the action node comes first and each node is
    followed by its groups' parts."""
    sts = [_st("health", 1, 1, "GET /health", {"level": "ip", "items": ["192.168.9.9"], "distinct": 1},
               mass=10000.0),
           _st("docs", 5, 5, "GET /docs", {"level": "prefix", "items": ["192.168.0.0/16"], "distinct": 25},
               mass=266.0),
           _st("docs|grp:G11", 5, 5, "GET /docs", SALES, mass=158.0),
           _st("docs-child", 5, 9, "GET /docs", {"level": "ip", "items": ["192.168.1.21"], "distinct": 1},
               depth=2, mass=200.0),
           _st("docs|grp:dept:综合部", 5, 5, "GET /docs", GA, mass=50.0)]
    actions = {1: {"mass": 10000.0}, 5: {"mass": 266.0}}
    ids = [s["id"] for s in VW.order_statements(sts, actions)]
    assert ids == ["docs", "docs|grp:G11", "docs|grp:dept:综合部", "docs-child", "health"]


def test_every_statement_ends_with_confidence_and_support():
    zh, en = PR.evidence_tail(0.712, 379.4, "events")
    assert zh == "置信 0.71 · 依据 379 次" and en == "Confidence 0.71, support 379 events"
    st = VW.activity_statement("G1", "综合部", "oa", [{"action": "POST oa.corp /login", "share": 1.0}],
                               0.8, {"192.168.1.21", "192.168.1.23"}, "class:grp:G1", {}, conf=0.66)
    assert st["confidence"] == 0.66
    assert st["text_zh"].endswith("置信 0.66 · 依据 2 个成员")


def _who(weights: Dict[str, float], days: int = 30) -> Any:
    from app.engines.behavior.lib import pnode as PN
    w = PN.WhoSummary()
    t = T0
    for d in range(days):
        for ip, k in weights.items():
            for j in range(int(k)):
                t = T0 + d * DAY + 60.0 * j
                w.update([ip, None, None, None, None], ip, t, 1.0, 1.0)
    return w, t


def test_a_listed_who_states_the_mass_its_list_leaves_out():
    """Three addresses holding 96 % of a closed node's sources, two more 2 %
    each: the statement lists the three and must state U >= 0.04 (the chance
    that the next source is not listed), not the level's unseen mass (~0.001),
    which the evaluator reads as a 99.9 % claim (round 4: GET /docs listed
    three /24s, left 10.168.7.0/24 out and held 0.92-0.96)."""
    w, t = _who({"192.168.1.21": 16, "192.168.1.23": 16, "192.168.1.24": 16,
                 "192.168.1.30": 1, "192.168.1.31": 1})
    ev, zh, en, c = PR.who_block(w, t + 60.0, 30, {}, {})
    assert ev["level"] == "ip" and len(ev["items"]) == 3 and ev["closed"]
    assert w.levels[0].unseen(t + 60.0) < 0.02
    assert ev["U"] == pytest.approx(0.04, abs=0.01)
    assert c == pytest.approx(1.0 - ev["U"])


def test_a_groups_part_states_its_own_held_out_hold_rate(monkeypatch):
    """P04 keeps a hold record per learned group (pnode meta 'hold_g'): a
    group's part states the group's own tests pooled with the node's hold
    probability as prior (views.part_hold); a group without tests states the
    node's p_hold."""
    from app.engines.behavior.lib import ptree as PT
    from app.engines.behavior.lib import pnode as PN
    from app.models.schema import ORG, SYSTEM_ENTITY
    ga = ["192.168.1.21", "192.168.1.23", "10.168.7.121"]
    fin = ["192.168.2.10", "192.168.2.11"]
    m = PT.PTreeModel("oa")
    tr = m.tree(EV.KIND_TXN, T0, create=True)
    sp = tr.split(tr.root, "http.route", 0, [["GET oa.corp /docs"]], T0)
    node, root = tr.nodes[sp.children[0]], tr.nodes[tr.root]
    t = T0
    for d in range(21):
        day0 = T0 + d * 86400.0
        for k, (ip, g) in enumerate([(x, "G1") for x in ga] + [(x, "G2") for x in fin]):
            t = day0 + (10 * 60 + 7 * k) * 60.0
            p = ip.split(".")
            keys = [ip, ".".join(p[:3]) + ".0/24", ".".join(p[:2]) + ".0.0/16", f"grp:{g}", "reg:∅"]
            for nd in (root, node):
                nd.update_core(t, 1.0, 1.0, keys, ip, 0, (t + 8 * 3600) % 86400 / 60.0,
                               int((t + 8 * 3600) // 86400))
    for nd in (root, node):
        nd.state = "confirmed"
    orig = PN.Node.p_hold
    monkeypatch.setattr(PN.Node, "p_hold", lambda self, _t: 0.45 if self is node else orig(self, _t))
    hr = PN.HoldRecord()
    for i in range(650):                       # >= 6 held-out tests of 100 checks
        hr.add("who", True, 0.95, t - 60.0 * (650 - i))
    node.meta["hold_g"] = {"grp:G1": hr}
    node.meta["hold_prior"] = (2.0, 2.0)
    ps, n = hr.tests()
    assert n > 4
    want_g1 = (ps + 4.0 * 0.45) / (n + 4.0)
    m.t_last = t
    st = make_store()
    st.put_model("oa", SYSTEM_ENTITY, MP.PTREE, m, version=1)
    st.put_model(ORG, ORG, MP.WHO_GROUPS, {
        "groups": {"G1": {"id": "G1", "name": "综合部", "members": ga},
                   "G2": {"id": "G2", "name": "财务部", "members": fin}},
        "ip2g": dict({ip: "G1" for ip in ga}, **{ip: "G2" for ip in fin}),
        "mode": {"oa": {"mode": "ip"}}})
    v = VW.system_view(st, "oa", {"progressive": {"enabled": True}, "tz": "Asia/Shanghai"}, t + 3600.0)
    parts = {s["evidence"]["who"]["name"]: s for s in v["statements"]
             if (s["evidence"].get("who") or {}).get("part_of")}
    # the group's own tests pooled with the node's rate (prior strength a + b = 4)
    assert parts["综合部"]["confidence"] == pytest.approx(want_g1, abs=1e-6)
    assert parts["综合部"]["confidence"] > 0.7         # its own passing tests lift it above the node
    # a group without tests of its own states the node's rate
    assert parts["财务部"]["confidence"] == pytest.approx(0.45)
    # support: the part's share of the node's events, stated in the text
    assert "依据" in parts["综合部"]["text_zh"]
    assert parts["综合部"]["support"] < float(node.n_c(t + 3600.0))


def test_department_view_states_every_action_the_system_views_state_about_it():
    """综合部's roles hold < 20 % of the shared GET /docs node each, so no
    role view has a statement for it; the system view's 综合部 part of that
    node completes the department view (one statement per action: the most
    specific)."""
    groups = {
        "G22": {"id": "G22", "name": "综合部·审批", "dept": "综合部", "members": ["192.168.1.21"],
                "systems": {"oa": 1.0}, "cohesion": 1.0,
                "actions": {"oa": [{"action": "GET oa.corp /docs", "share": 0.5, "members": []}]}},
        "G10": {"id": "G10", "name": "综合部", "dept": "综合部", "members": ["10.168.7.121", "192.168.1.23"],
                "systems": {"oa": 1.0}, "cohesion": 0.5,
                "actions": {"oa": [{"action": "GET oa.corp /docs", "share": 0.5, "members": []}]}}}
    views = [{"group": "G22", "statements": []}, {"group": "G10", "statements": []}]
    ga = sorted(GA["members"])
    part = lambda sid, depth, conf: {  # noqa: E731
        "id": sid, "text_zh": sid, "text_en": sid, "mass": 50.0, "confidence": conf,
        "evidence": {"system": "oa", "route": "GET oa.corp /docs", "depth": depth, "node": depth,
                     "who": {"level": "grp", "name": "综合部", "members": ga}}}
    other = {"id": "p:sales", "text_zh": "x", "text_en": "x", "mass": 90.0, "confidence": 0.9,
             "evidence": {"system": "oa", "route": "GET oa.corp /docs", "depth": 1,
                          "who": {"level": "grp", "name": "销售部", "members": SALES["members"]}}}
    sv = {"oa": {"statements": [part("p:5|grp:dept:综合部", 1, 0.7), part("p:9|grp:dept:综合部", 2, 0.6), other]}}
    dv = VW.dept_view("综合部", views, groups, {}, T0, sv)
    docs = [s for s in dv["statements"] if (s.get("evidence") or {}).get("route") == "GET oa.corp /docs"]
    assert [s["id"] for s in docs] == ["p:9|grp:dept:综合部|class:grp:dept:综合部"]
    assert docs[0]["view"] == "group" and docs[0]["subject"] == "class:grp:dept:综合部"
    act = [s for s in dv["statements"] if (s.get("evidence") or {}).get("activity")][0]
    # the activity statement states the roles' size-weighted cohesion and its support
    assert act["confidence"] == pytest.approx(1.0 / 3 + 0.5 * 2 / 3, abs=1e-3)
    assert "依据 3 个成员" in act["text_zh"]
    # without the system views (round-3 behaviour) the action has no statement
    dv0 = VW.dept_view("综合部", views, groups, {}, T0)
    assert not [s for s in dv0["statements"] if (s.get("evidence") or {}).get("route") == "GET oa.corp /docs"]


def test_a_confirmed_rename_is_stated_under_the_new_page():
    """D3 (/approval/ -> /flow/ for the approver): once P10 confirms the
    rename (model.pflow 'renamed'), the system view states the old action's
    pattern under the new page, marked as renamed, and no longer under the old
    one; when the lattice has a confident node of its own for the new page,
    that node is stated instead."""
    from test_p14_round3 import _tree as tree3
    from app.models.schema import ORG, SYSTEM_ENTITY
    old, new = "GET oa.corp /approval/list", "GET oa.corp /flow/list"
    st = make_store()
    m = tree3("oa", {old: [("192.168.1.21", "G3")], "POST oa.corp /login": [("192.168.1.23", "G1")]})
    st.put_model("oa", SYSTEM_ENTITY, MP.PTREE, m, version=1)
    st.put_model(ORG, ORG, MP.WHO_GROUPS, {"groups": {}, "ip2g": {}, "mode": {"oa": {"mode": "ip"}}})
    cfg = {"progressive": {"enabled": True}, "tz": "Asia/Shanghai"}
    v = VW.system_view(st, "oa", cfg, m.t_last + 3600.0)
    assert any(s["evidence"]["route"] == old for s in v["statements"])
    st.put_model("oa", SYSTEM_ENTITY, MP.PFLOW, {"renamed": {new: {"from": old, "day": 0, "sources": ["192.168.1.21"]}}})
    v = VW.system_view(st, "oa", cfg, m.t_last + 3600.0)
    routes = [s["evidence"]["route"] for s in v["statements"]]
    assert old not in routes and new in routes
    s = next(s for s in v["statements"] if s["evidence"]["route"] == new)
    assert s["evidence"]["adopted_from"] == old and "原 GET /approval/list" in s["text_zh"]
    assert s["state"] in ("confirmed", "stable")
    assert all(new in c[2] for c in s["evidence"]["context"] if c[0] == "http.route")
    # the old page's node already stale: its own (unconfirmed) statement stays
    # as history pointing at the successor, the adopted one is the current
    tr = m.kinds[EV.KIND_TXN]
    for nd in tr.nodes.values():
        if nd.depth == 1 and any("/approval/list" in str(c[2]) for c in nd.ctx):
            nd.state = "stale"
    v = VW.system_view(st, "oa", cfg, m.t_last + 3600.0)
    olds = [s for s in v["statements"] if s["evidence"]["route"] == old]
    assert len(olds) == 1 and olds[0]["state"] == "stale" and olds[0]["evidence"]["renamed_to"] == new
    assert "页面已更名为 GET /flow/list" in olds[0]["text_zh"]
    assert any(s["evidence"]["route"] == new and s["state"] == "confirmed" for s in v["statements"])
    # the new page has its own confident node: it is stated, the adoption ends
    assert VW.renamed_routes({"renamed": {new: {"from": old}}}, {new}) == {}


def test_department_view_of_a_pool_states_its_prefix_level_statements():
    """研发 (a DHCP pool) is stated in the system views by its prefixes
    ('研发（10.50.0.0/24、…，约 174 个 IP）', who level prefix, group_name 研发):
    its department view holds those statements too (pack O seed 1: 0 of 4)."""
    groups = {"G5": {"id": "G5", "name": "研发", "dept": "研发", "members": ["10.50.0.7"], "systems": {"oa": 1.0},
                     "actions": {"oa": [{"action": "POST oa.corp /login", "share": 1.0}]}},
              "G6": {"id": "G6", "name": "研发·x", "dept": "研发", "members": ["10.50.1.9"], "systems": {"oa": 1.0},
                     "actions": {"oa": [{"action": "POST oa.corp /login", "share": 1.0}]}}}
    views = [{"group": "G5", "statements": []}, {"group": "G6", "statements": []}]
    st = {"id": "p:13", "text_zh": "x", "text_en": "x", "mass": 90.0, "confidence": 0.7,
          "evidence": {"system": "oa", "route": "POST oa.corp /login", "depth": 1,
                       "who": {"level": "prefix", "items": ["10.50.0.0/24", "10.50.1.0/24"], "group_name": "研发"}}}
    dv = VW.dept_view("研发", views, groups, {}, T0, {"oa": {"statements": [st]}})
    assert any(s["id"] == "p:13|class:grp:dept:研发" for s in dv["statements"])


def test_an_ancestor_mixing_routes_is_folded_into_its_route_child():
    """The finance root (90 % health checks, other routes 10 %) is rendered as
    'GET /health' with the other routes' upload sizes; its http.route child
    holds the health checks alone. Both state the same action for the same
    monitor: the child is the primary statement, the root's is folded (pack O
    seed 0 day 14: finance /health twice, mail root parts next to the TLS
    child's)."""
    tree = _tree({1: (0, "http.route"), 4: (1, "net.src")})
    mon = {"level": "ip", "items": ["192.168.9.9"]}
    sts = [_st("p:0", 0, 0, "GET /health", mon, depth=0), _st("p:1", 0, 1, "GET /health", mon)]
    prim, folded = VW.fold_duplicates(sts, tree)
    assert [s["id"] for s in prim] == ["p:1"]
    assert [(s["id"], s["folded_into"]) for s in folded] == [("p:0", "p:1")]
    # route then source splits: the group's part at the root folds into the deepest one
    sts = [_st("p:0|grp:G11", 0, 0, "TLS mail", SALES, depth=0), _st("p:4|grp:G11", 0, 4, "TLS mail", SALES, 2)]
    prim, folded = VW.fold_duplicates(sts, tree)
    assert [s["id"] for s in prim] == ["p:4|grp:G11"]


class _H:
    """A hierarchy stub: ctx.think_s numeric with 7 log-edges (8 bins)."""

    def kind(self, a):
        return "num" if a in ("ctx.think_s", "net.pkts_down") else ("text" if a.startswith("hdr.") else "cat")

    def _num_edges(self, a):
        import math
        import numpy as np
        if a == "ctx.think_s":
            return np.log(np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])), True
        return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), False


def test_variant_conditions_are_rendered_readably():
    h = _H()
    # level 3 = 2 bins of 4: both values = every bin -> the request has a predecessor
    assert VW._cond_phrase(h, "ctx.think_s", 3, ["0", "1"], False)[0] == "会话内后续请求"
    assert VW._cond_phrase(h, "ctx.think_s", 3, [EV.ABSENT], False)[0] == "会话首个请求"
    assert VW._cond_phrase(h, "net.pkts_down", 3, [0, 1], False)[0] == "有下行包数"
    assert VW._cond_phrase(h, "net.pkts_down", 1, [EV.ABSENT], False)[0] == "无下行包数"
    # level 1 bins 0-1 -> below the second edge; level 2 bin 1 (bins 2-3) -> a range
    assert VW._cond_phrase(h, "ctx.think_s", 1, [0, 1], False)[0] == "距上一请求间隔 < 1 s"
    assert VW._cond_phrase(h, "ctx.think_s", 2, [1], False)[0] == "距上一请求间隔 1 s–4 s"
    assert VW._cond_phrase(h, "net.pkts_down", 1, [5, 6, 7], False)[0] == "下行包数 ≥ 5"
    assert VW._cond_phrase(h, "hdr.user-agent", 1, ["U1 L6 / D1", "U1 L6 / D2"], False)[0] == \
        "请求头 user-agent 为 2 种特定形态"
    assert VW._cond_phrase(h, "http.method", 0, ["GET"], True)[0] == "非（方法 为 GET）"
    # 'not absent' is 'present'; a numeric value without a trailing '.0'
    assert VW._cond_phrase(h, "ctx.think_s", 3, [EV.ABSENT], True)[0] == "会话内后续请求"
    assert VW._cond_phrase(h, "hdr.user-agent.len", 0, [111.0], False)[0] == "请求头 user-agent.len 为 111"
    # route and source constraints are the action and the WHO, not conditions
    zh, en, ev = VW.variant_condition(h, [("http.route", 0, ["GET oa /x"], False),
                                          ("net.src", 1, ["10.50.0.0/24"], False),
                                          ("ctx.think_s", 3, [EV.ABSENT], False)])
    assert zh == "会话首个请求" and [e["attr"] for e in ev] == ["ctx.think_s"]


def test_variants_of_an_action_state_their_condition(monkeypatch):
    """Two children of GET /fin/ledger/{num} split on ctx.think_s (first
    request of a session vs later ones) rendered the same 'who opens route'
    sentence with different numbers: each now names its condition, so the
    view shows variants, not duplicates."""
    from app.engines.behavior.lib import ptree as PT
    from app.models.schema import ORG, SYSTEM_ENTITY
    fin = ["192.168.2.11", "192.168.2.12"]
    m = PT.PTreeModel("finance")
    tr = m.tree(EV.KIND_TXN, T0, create=True)
    sp = tr.split(tr.root, "http.route", 0, [["GET fin /fin/ledger/{num}"]], T0)
    act = sp.children[0]
    sp2 = tr.split(act, "ctx.think_s", 3, [[EV.ABSENT], ["0", "1"]], T0)
    nodes = [tr.nodes[tr.root], tr.nodes[act]] + [tr.nodes[c] for c in sp2.children]
    t = T0
    for d in range(21):
        for k, ip in enumerate(fin):
            t = T0 + d * 86400.0 + (10 * 60 + 7 * k) * 60.0
            p = ip.split(".")
            keys = [ip, ".".join(p[:3]) + ".0/24", ".".join(p[:2]) + ".0.0/16", "grp:G2", "reg:∅"]
            for nd in nodes:
                nd.update_core(t, 1.0, 1.0, keys, ip, 0, (t + 8 * 3600) % 86400 / 60.0,
                               int((t + 8 * 3600) // 86400))
    for nd in nodes:
        nd.state = "confirmed"
    m.t_last = t
    st = make_store()
    st.put_model("finance", SYSTEM_ENTITY, MP.PTREE, m, version=1)
    st.put_model(ORG, ORG, MP.WHO_GROUPS, {"groups": {}, "ip2g": {}, "mode": {"finance": {"mode": "ip"}}})
    v = VW.system_view(st, "finance", {"progressive": {"enabled": True}, "tz": "Asia/Shanghai"}, t + 3600.0)
    by = {s["evidence"]["node"]: s for s in v["statements"] + v.get("folded", [])}
    c0, c1 = sp2.children
    assert "GET /fin/ledger/{num}（查看账簿）（条件：会话首个请求）" in by[c0]["text_zh"]
    # (no registry here: the bins cannot be decoded, but no bin index is shown)
    assert "（条件：距上一请求间隔 在特定区间）" in by[c1]["text_zh"]
    assert by[c1]["evidence"]["condition"][0]["attr"] == "ctx.think_s"
    assert "condition" not in by[act]["evidence"]       # the action node itself is no variant
    assert "(when: first request of a session)" in by[c0]["text_en"]


def test_a_pool_stated_by_its_configured_region_is_named_after_its_department():
    """P12's 'reg' arm states 研发's DHCP pool as 'reg:研发 DHCP': resolved to the
    scope's CIDR it lies inside the pool group's pool, so the statement is
    labelled 研发 (group_name) and the department view can hold it."""
    c = SimpleNamespace(config={"dhcp_scopes": [{"name": "研发 DHCP", "cidr": "10.50.0.0/22"}]},
                        groups={"G5": {"id": "G5", "name": "研发", "name_source": "config",
                                       "pool": "10.50.0.0/22"}})
    assert VW.pool_label(c, ["reg:研发 DHCP"]) == ("研发", ["研发 DHCP"])
    assert VW.pool_label(c, ["10.50.1.0/24"]) == ("研发", ["10.50.1.0/24"])
    assert VW.pool_label(c, ["reg:public-0"])[0] is None
    assert VW.pool_label(c, ["192.168.3.0/24"])[0] is None


def _drift_tree(monkeypatch, drift: bool, fin=()):
    from app.engines.behavior.lib import ptree as PT
    from app.engines.behavior.lib import pnode as PN
    from app.models.schema import ORG, SYSTEM_ENTITY
    ga = ["192.168.1.21", "192.168.1.23", "10.168.7.121"]
    m = PT.PTreeModel("oa")
    tr = m.tree(EV.KIND_TXN, T0, create=True)
    sp = tr.split(tr.root, "http.route", 0, [["GET oa.corp /home"]], T0)
    node, root = tr.nodes[sp.children[0]], tr.nodes[tr.root]
    t = T0
    for d in range(14):
        for k, (ip, g) in enumerate([(x, "G1") for x in ga] + [(x, "G2") for x in fin]):
            t = T0 + d * 86400.0 + (9 * 60 + 7 * k) * 60.0
            p = ip.split(".")
            keys = [ip, ".".join(p[:3]) + ".0/24", ".".join(p[:2]) + ".0.0/16", f"grp:{g}", "reg:∅"]
            for nd in (root, node):
                nd.update_core(t, 1.0, 1.0, keys, ip, 0, (t + 8 * 3600) % 86400 / 60.0,
                               int((t + 8 * 3600) // 86400))
    for nd in (root, node):
        nd.state = "confirmed"
    orig = PN.Node.p_hold
    monkeypatch.setattr(PN.Node, "p_hold", lambda self, _t: 0.8 if self is node else orig(self, _t))
    m.t_last = t
    st = make_store()
    st.put_model("oa", SYSTEM_ENTITY, MP.PTREE, m, version=1)
    st.put_model(ORG, ORG, MP.WHO_GROUPS, {
        "groups": {"G1": {"id": "G1", "name": "综合部", "members": ga},
                   "G2": {"id": "G2", "name": "财务部", "members": list(fin)}},
        "ip2g": dict({ip: "G1" for ip in ga}, **{ip: "G2" for ip in fin}), "mode": {"oa": {"mode": "ip"}}})
    rec = {"windows": [[540, 562]], "coverage": 0.9, "confidence": 0.3 if drift else 0.9,
           "text_zh": "工作日 09:00–09:22", "text_en": "workdays 09:00-09:22"}
    when = {"workday": [[540, 562]], "nonworkday": [], "coverage": 0.9,
            "confidence": 0.3 if drift else 0.9}
    if drift:
        rec["drift"] = when["drift"] = {"workday": {"p": 0.001, "dates_after": 1}}
    st.put_model("oa", SYSTEM_ENTITY, MP.PWIN, {"version": 1, "nodes": {EV.KIND_TXN: {node.id: {
        "status": "fitted", "when": when, "by_daytype": {"wd": rec}}}}}, version=1)
    v = VW.system_view(st, "oa", {"progressive": {"enabled": True}, "tz": "Asia/Shanghai"}, t + 3600.0)
    return [s for s in v["statements"] if s["evidence"]["route"] == "GET oa.corp /home"
            and s["evidence"]["node"] == node.id]


def test_a_node_whose_windows_drift_is_stated_as_evolving(monkeypatch):
    """Spec §6.9.2: a content change makes the node evolving and P09 fits
    provisional constraints. P09 marks the windows 'drift' (a time-of-day
    change younger than its acceptance period); the node statement used to
    stay 'confirmed' at P04's hold rate (pack O seed 0 day 14: 综合部 home /
    login stated confirmed 0.61-0.77 while holding 0.08-0.12)."""
    sts = _drift_tree(monkeypatch, drift=True)
    assert sts and all(s["state"] == "evolving" for s in sts)
    assert all(s["confidence"] <= 0.3 + 1e-9 for s in sts)
    assert all(s["evidence"]["when"].get("drift") for s in sts)
    assert any("正在变化" in s["text_zh"] for s in sts)
    # without the mark: confirmed at the node's held-out hold rate
    sts = _drift_tree(monkeypatch, drift=False)
    assert sts and all(s["state"] == "confirmed" for s in sts)
    assert any(s["confidence"] == pytest.approx(0.8) for s in sts)


def test_a_groups_part_leaves_violating_rows_out_of_its_windows(monkeypatch):
    """P09 fits a node's windows without the rows P03 judged violations (P06's
    ledger, content_bounds.point_ledger); a group's part (pwindows.part_when)
    was fitted with them (content owner's open issue): the views now pass the
    tree's ledger as `drop`."""
    from app.engines.behavior import content_bounds as CB
    from app.engines.behavior.lib import pwindows as PW
    led = {(1234.5, "192.168.1.21")}
    seen = []
    monkeypatch.setattr(CB, "point_ledger", lambda store, key, kind: set(led))

    def spy(when, members, tz=0.0, min_points=PW.MIN_POINTS, drop=None):
        seen.append(drop)
        return None
    monkeypatch.setattr(PW, "part_when", spy)
    _drift_tree(monkeypatch, drift=False, fin=("192.168.2.10", "192.168.2.11", "192.168.2.12"))
    assert seen and all(d == led for d in seen)


def test_a_groups_part_states_its_own_numeric_band(monkeypatch):
    """Evaluator round 4: a group's part of a shared node states the GROUP's
    numeric band (P04 keeps the stated numeric constraints' values per tracked
    group, pnode meta 'gnum'; views.part_content refits them). Before, the part
    stated the node's band: pack O, the 综合部 part of OA's login node (truth
    1-2 KB) stated the node's 0.5-1.5 KB on every seed (PG10 day 11)."""
    import numpy as np
    from app.engines.behavior.lib import ptree as PT
    from app.engines.behavior.lib import pnode as PN
    from app.engines.behavior.lib import pbounds as PB
    from app.engines.behavior.pattern_tree import PatternTreeEngine
    from app.models.schema import ORG, SYSTEM_ENTITY
    rng = np.random.default_rng(3)
    ga = ["192.168.1.21", "192.168.1.23", "10.168.7.121"]
    sales = [f"192.168.3.{i}" for i in range(20, 24)]
    m = PT.PTreeModel("oa")
    tr = m.tree(EV.KIND_TXN, T0, create=True)
    sp = tr.split(tr.root, "http.route", 0, [["POST oa.corp /login"]], T0)
    node, root = tr.nodes[sp.children[0]], tr.nodes[tr.root]
    node.state = "confirmed"
    node.ref = {"t": T0, "hold": {"body.len": ("num", 512.0, 1536.0, 0.9)}}
    allv = PN.NumSummary(log=True)
    t = T0
    for d in range(14):
        day = 739000 + d
        for k, (ip, g, lo, hi) in enumerate([(x, "G1", 1024, 2048) for x in ga]
                                            + [(x, "G2", 500, 1400) for x in sales]):
            t = T0 + d * 86400.0 + (9 * 60 + k) * 60.0
            v = float(rng.uniform(lo, hi))
            p = ip.split(".")
            keys = [ip, ".".join(p[:3]) + ".0/24", ".".join(p[:2]) + ".0.0/16", f"grp:{g}", "reg:∅"]
            for nd in (root, node):
                nd.update_core(t, 1.0, 1.0, keys, ip, 0, (t + 8 * 3600) % 86400 / 60.0, day)
            allv.update(v, t, 1.0, 1.0, day=day)
            PatternTreeEngine._hold_check(None, node, lambda a, v=v: v if a == "body.len" else EV.ABSENT,
                                          keys, 0, 540.0, t, 1.0, 1.0, day)
    root.state = "confirmed"
    assert set((node.meta.get("gnum") or {})) == {"grp:G1", "grp:G2"}
    m.t_last = t
    now = t + 3600.0
    rec = PB.fit_numeric(allv, now, PB.local_day(now, {"tz": "Asia/Shanghai"}), 200.0, 200.0, 0.0, "B")
    st = make_store()
    st.put_model("oa", SYSTEM_ENTITY, MP.PTREE, m, version=1)
    st.put_model("oa", SYSTEM_ENTITY, MP.PBOUNDS, {"version": 1, "nodes": {EV.KIND_TXN: {node.id: {
        "status": "fitted", "attrs": {"body.len": rec}}}}}, version=1)
    st.put_model(ORG, ORG, MP.WHO_GROUPS, {
        "groups": {"G1": {"id": "G1", "name": "综合部", "members": ga},
                   "G2": {"id": "G2", "name": "销售部", "members": sales}},
        "ip2g": dict({ip: "G1" for ip in ga}, **{ip: "G2" for ip in sales}),
        "mode": {"oa": {"mode": "ip"}}})
    v = VW.system_view(st, "oa", {"progressive": {"enabled": True}, "tz": "Asia/Shanghai"}, now)
    parts = {s["evidence"]["who"]["name"]: s for s in v["statements"]
             if (s["evidence"].get("who") or {}).get("part_of")}
    b_ga = parts["综合部"]["evidence"]["content"]["body.len"]["band90"]
    b_sa = parts["销售部"]["evidence"]["content"]["body.len"]["band90"]
    assert 950 <= b_ga[0] <= 1300 and 1900 <= b_ga[1] <= 2150, b_ga      # not the node's
    assert b_sa[1] <= 1450, b_sa


def _login_grammar(closed=None):
    rec = {"kind": "text", "grammar": "[a-z]{3,8}", "mode": "shape", "charset": ["L"],
           "alnum": "a-z", "len": [3, 8], "c_g": 1.0, "U_s": 0.005, "n": 380.0}
    if closed is not None:
        rec.update(closed=list(closed), U=0.004)
    return {"attrs": {"body.kv.username": rec}}


def _bound(tab):
    return {"pairs": {"net.src->body.kv.username": {
        "x": "net.src", "y": "body.kv.username",
        "table": {ip: {"top": v, "bound": True, "LB": 0.999} for ip, v in tab.items()}}}}


def test_a_part_whose_members_are_bound_states_their_values_as_a_closed_set():
    """Evaluator round 4: OA's all-department login node holds an OPEN user-name
    set (a DHCP pool's names keep arriving) but each of 财务部's three addresses
    is bound to one name; the 财务部 part stated the node's '[a-z]{3,8}' and no
    closed set, so FIN.oa.login was missed on every seed of pack O. The part's
    values are its members' bound values (closed by the bindings, U = 1 - the
    smallest lower bound) and its grammar is theirs."""
    fin = {"192.168.2.10": "kate", "192.168.2.11": "lucy", "192.168.2.12": "tom"}
    other = {"192.168.3.20": "amy", "192.168.3.32": "michelle"}
    out = VW._restrict_closed(_login_grammar(), _bound({**fin, **other}), set(fin))
    rec = out["attrs"]["body.kv.username"]
    assert rec["closed"] == ["kate", "lucy", "tom"]
    assert rec["grammar"] == "[a-z]{3,4}" and rec["len"] == [3, 4]
    assert rec["U"] == pytest.approx(0.001)
    # rendered as a closed set with the part's grammar
    content, zh, _en, _c = PR.content_block(None, out)
    assert content["body.kv.username"]["closed"] == ["kate", "lucy", "tom"]
    assert any("[a-z]{3,4}" in z for z in zh)


def test_a_part_with_an_unbound_member_keeps_the_nodes_value_constraints():
    tab = {"192.168.2.10": "kate", "192.168.2.11": "lucy"}
    bent = _bound(tab)
    bent["pairs"]["net.src->body.kv.username"]["table"]["192.168.2.12"] = {"top": "tom", "bound": False}
    g = _login_grammar()
    assert VW._restrict_closed(g, bent, {"192.168.2.10", "192.168.2.11", "192.168.2.12"}) is g


def test_a_closed_node_sets_restriction_also_states_the_parts_grammar():
    names = ["amy", "brandon", "kate", "lucy", "michelle", "tom"]
    fin = {"192.168.2.10": "kate", "192.168.2.11": "lucy", "192.168.2.12": "tom"}
    out = VW._restrict_closed(_login_grammar(names), _bound(fin), set(fin))
    rec = out["attrs"]["body.kv.username"]
    assert rec["closed"] == ["kate", "lucy", "tom"] and rec["part_of_closed"] == 6
    assert rec["grammar"] == "[a-z]{3,4}"


def test_after_a_confirmed_rename_the_user_view_lists_the_new_page(store):
    """EV-10. Pack O D3 (/approval/ -> /flow/ for the approver on day 14): P11's
    action mix of the group decays slowly, so on day 21 the department view of
    every seed listed '查看审批（GET /approval/list）[192.168.1.21]' while the
    system view stated those pages '近期未出现（页面已更名为 …）'. Under P10's
    confirmed renames an old page counts as its new page, merged with the new
    page's own row."""
    from app.models.schema import ORG, SYSTEM_ENTITY
    old, new = "POST oa.corp /approval/{num}/approve", "POST oa.corp /flow/{num}/approve"
    wg = dict(store.get_model(ORG, ORG, MP.WHO_GROUPS))
    gr = dict(wg["groups"]["G1"])
    gr["actions"] = {"oa": [
        {"action": "POST oa.corp /login", "share": 0.5, "support": 1.0, "members": []},
        {"action": old, "share": 0.3, "support": 0.333, "members": ["192.168.1.21"]},
        {"action": new, "share": 0.2, "support": 0.333, "members": ["192.168.1.21"]}]}
    wg["groups"] = dict(wg["groups"], G1=gr)
    store.put_model(ORG, ORG, MP.WHO_GROUPS, wg)
    store.put_model("oa", SYSTEM_ENTITY, MP.PFLOW,
                    {"renamed": {new: {"from": old, "day": 0, "sources": ["192.168.1.21"]}}})
    gv = VW.group_view(store, "G1", {"progressive": {"enabled": True}, "tz": "Asia/Shanghai"}, store.now)
    act = next(s for s in gv["statements"] if s["evidence"].get("activity"))
    assert "/approval/" not in act["text_zh"]
    assert act["text_zh"].count("审批（POST /flow/{num}/approve）[192.168.1.21]") == 1
    row = next(r for r in act["evidence"]["actions"] if r["action"] == new)
    assert row["share"] == pytest.approx(0.5)
    # the department view composes its roles' mixes under the same renames
    groups = {
        "G22": {"id": "G22", "name": "综合部·审批", "dept": "综合部", "members": ["192.168.1.21"],
                "systems": {"oa": 1.0}, "cohesion": 1.0,
                "actions": {"oa": [{"action": old, "share": 0.6, "members": []},
                                   {"action": new, "share": 0.1, "members": []}]}},
        "G10": {"id": "G10", "name": "综合部", "dept": "综合部", "members": ["10.168.7.121", "192.168.1.23"],
                "systems": {"oa": 1.0}, "cohesion": 0.5,
                "actions": {"oa": [{"action": "POST oa.corp /login", "share": 1.0, "members": []}]}}}
    views = [{"group": "G22", "statements": []}, {"group": "G10", "statements": []}]
    dv = VW.dept_view("综合部", views, groups, {}, T0, renames={"oa": VW.rename_map(
        {"renamed": {new: {"from": old}}})})
    act = next(s for s in dv["statements"] if (s.get("evidence") or {}).get("activity"))
    assert "/approval/" not in act["text_zh"]
    assert [r["action"] for r in act["evidence"]["actions"]].count(new) == 1


def test_a_groups_part_states_only_its_own_members_bindings(store):
    """EV-11. A part's binding text was rendered from the node's whole P08
    table and only the evidence was restricted afterwards: pack O, the part
    of 192.168.1.21 at OA's login node stated '10.168.7.121 → username=mike.w、
    … username=rose 只来自 …' (the other members' bindings and reverse pairs)."""
    from app.models.schema import SYSTEM_ENTITY
    from test_p14_views import CFG
    tree = MP.get_ptree(store, "oa").kinds[EV.KIND_TXN]
    nid = tree.nodes[tree.root].split.children[0]
    bind = store.get_model("oa", SYSTEM_ENTITY, MP.PBIND)
    ent = bind["nodes"][0][nid]
    ent["pairs"]["body.kv.username->client.stack"] = {
        "x": "body.kv.username", "y": "client.stack", "dir": "rev",
        "fd": {"g3": 0.0, "holds": True, "n": 180.0},
        "table": {u: {"n": 60.0, "top": "-|chrome/126|win|128|w16", "bound": True, "LB": 0.95}
                  for u in ("jack", "mike", "rose")}}
    store.put_model("oa", SYSTEM_ENTITY, MP.PBIND, bind)
    c = VW._Ctx(store, "oa", CFG, store.now)
    nd = tree.nodes[nid]
    full = VW.node_statement(c, EV.KIND_TXN, nd, "POST oa.corp /login")
    assert "192.168.1.23 → username=rose" in full["text_zh"] and "username=rose 只来自" in full["text_zh"]
    st = VW.node_statement(c, EV.KIND_TXN, nd, "POST oa.corp /login",
                           part=("G1", ["192.168.1.21"], "综合部", 0.33))
    zh = st["text_zh"]
    assert "192.168.1.21 → username=jack" in zh and "username=jack 只来自" in zh
    assert "rose" not in zh and "mike" not in zh
