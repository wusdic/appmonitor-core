"""P14 ViewsEngine (behavior.views; docs/lib3/progressive.md §6.17, card P14):
golden rendering of the requirement's OA example from a fixed model fixture
(zh and en), the negative group statement, number rounding, the statement
contract read by eval/pmetrics, and the route index."""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List

import numpy as np
import pytest

from helpers import ctx, make_store

from app.engines.behavior import views as VW
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import prender as PR
from app.engines.behavior.lib import ptree as PT
from app.eval.pmetrics import LStmt, bindings_match, statements, group_statements
from app.models.schema import ORG, SYSTEM_ENTITY

TZ = _dt.timezone(_dt.timedelta(hours=8))
T0 = _dt.datetime(2026, 9, 1, tzinfo=TZ).timestamp()        # Tuesday 00:00 local
DAY = 86400.0
GA = ["192.168.1.21", "192.168.1.23", "10.168.7.121"]
USERS = {"192.168.1.21": "jack", "192.168.1.23": "rose", "10.168.7.121": "mike"}
LOGIN = "POST oa.corp /login"
HOME = "GET oa.corp /home"
CFG = {"progressive": {"enabled": True}, "tz": "Asia/Shanghai"}


def _keys(ip: str, g: str) -> List[Any]:
    p = ip.split(".")
    return [ip, ".".join(p[:3]) + ".0/24", ".".join(p[:2]) + ".0.0/16", f"grp:{g}", "reg:∅"]


def _build_tree(key: str, route: str, ips: List[str], g: str, days: int, dst: str) -> PT.PTreeModel:
    m = PT.PTreeModel(key)
    tr = m.tree(EV.KIND_TXN, T0, create=True)
    sp = tr.split(tr.root, "http.route", 0, [[route]], T0)
    node = tr.nodes[sp.children[0]]
    root = tr.nodes[tr.root]
    t = T0
    for d in range(days):
        day0 = T0 + d * DAY
        if _dt.datetime.fromtimestamp(day0, TZ).weekday() >= 5:
            continue
        for k, ip in enumerate(ips):
            t = day0 + (9 * 60 + 5 * k + 1) * 60.0
            dayn = int((t + 8 * 3600) // DAY)
            for nd in (root, node):
                nd.update_core(t, 1.0, 1.0, _keys(ip, g), ip, 0, (t + 8 * 3600) % DAY / 60.0, dayn)
    for nd in (root, node):
        nd.state = "confirmed"
    root.inv["net.dst"] = (0, dst)
    m.t_last = t
    return m


def _fitted(nid: int) -> Dict[str, Any]:
    bounds = {"fmt": 1, "nodes": {0: {nid: {"status": "fitted", "attrs": {"body.len": {
        "kind": "num", "unit": "B", "band90": [1031.0, 2041.0], "band98": [700.0, 2800.0],
        "coverage": 0.9, "coverage_emp": 0.9,
        "disp90": {"lo": 1024.0, "hi": 2048.0, "coverage": 0.9, "text": "1–2 KB"},
        "range": [530.0, 3050.0], "disp_range": {"lo": 512.0, "hi": 3072.0, "text": "0.5–3 KB"},
        "n_rng": 183.0, "cover": 2.0 / 184.0, "hard": True, "n_c": 60.0, "confidence": 0.97,
        "approx": 0.0}}}}}}
    grammar = {"fmt": 1, "nodes": {0: {nid: {"status": "fitted", "attrs": {
        "body.keys": {"kind": "set", "n": 60.0, "required": ["captcha", "password", "username"],
                      "optional": [], "presence": {"captcha": 1.0, "password": 1.0, "username": 1.0},
                      "p_new_key": 0.008, "p_missing": {}, "confidence": 0.99},
        "body.kv.username": {"kind": "text", "n": 60.0, "grammar": "[a-z]{4}", "c_g": 1.0,
                             "U_s": 0.008, "len": [4, 4], "closed": ["jack", "mike", "rose"],
                             "U": 0.008, "confidence": 0.99}}}}}}
    table = {ip: {"n": 60.0, "top": u, "k": 60.0, "bound": True, "LB": 0.95, "p_viol": 0.01}
             for ip, u in USERS.items()}
    bind = {"fmt": 1, "nodes": {0: {nid: {"status": "fitted", "pairs": {
        "net.src->body.kv.username": {
            "x": "net.src", "y": "body.kv.username", "dir": "fwd",
            "fd": {"g3": 0.0, "holds": True, "one_to_one": True, "n": 180.0},
            "table": table, "bound_values": {u: [ip] for ip, u in USERS.items()}}}}}}}
    win = {"fmt": 1, "nodes": {0: {nid: {
        "status": "fitted",
        "by_daytype": {"wd": {"windows": [[540, 561]], "coverage": 0.97, "dates": 21,
                              "text_zh": "工作日 09:00–09:21（覆盖 97 %，21 个工作日）",
                              "text_en": "workdays 09:00–09:21 (coverage 97 %, 21 dates)"}},
        "when": {"workday": [[540, 561]], "nonworkday": [], "coverage": 0.97, "confidence": 0.97}}}}}
    flow = {"scopes": {"*": {"edges": [{"from": LOGIN, "to": HOME, "dep": 0.95, "band": [1.0, 5.0],
                                        "confidence": 0.93}]}}}
    return {MP.PBOUNDS: bounds, MP.PGRAMMAR: grammar, MP.PBIND: bind, MP.PWIN: win, MP.PFLOW: flow}


@pytest.fixture()
def store():
    st = make_store()
    oa = _build_tree("oa", LOGIN, GA, "G1", 21, "192.168.100.100:8080")
    st.put_model("oa", SYSTEM_ENTITY, MP.PTREE, oa, version=1)
    nid = oa.kinds[0].nodes[oa.kinds[0].root].split.children[0]
    for name, obj in _fitted(nid).items():
        st.put_model("oa", SYSTEM_ENTITY, name, obj)
    fin = _build_tree("finance", "POST fin.corp /fin/approval/{num}/approve", ["192.168.2.10"], "G2", 21,
                      "192.168.100.110:8443")
    st.put_model("finance", SYSTEM_ENTITY, MP.PTREE, fin, version=1)
    st.put_model(ORG, ORG, MP.WHO_GROUPS, {
        "groups": {"G1": {"id": "G1", "name": "综合部", "members": list(GA), "covers": [],
                          "systems": {"oa": 1.0}},
                   "G2": {"id": "G2", "name": "财务部", "members": ["192.168.2.10"], "covers": [],
                          "systems": {"finance": 1.0}}},
        "ip2g": dict({ip: "G1" for ip in GA}, **{"192.168.2.10": "G2"}),
        "mode": {"oa": {"mode": "ip"}, "finance": {"mode": "ip"}}})
    for s in ("oa", "finance"):
        st.add_batch(s, EV.EVT_BATCH, T0 + 21 * DAY, EV.BatchBuilder(s).build(T0, T0 + 21 * DAY))
    st.now = oa.t_last + 3600.0
    return st


GOLDEN_ZH = ("【oa · 192.168.100.100:8080】工作日 09:00–09:21（覆盖 97 %，21 个工作日），"
             "综合部（10.168.7.121、192.168.1.21、192.168.1.23）访问 POST /login："
             "提交数据量 90 % 在 1–2 KB，全部在 0.5–3 KB（n = 183，下次越界概率 ≤ 1.1 %）；"
             "表单键必含 captcha=、password=、username=；username= 取值 `[a-z]{4}`，"
             "username= 取值集合封闭 {jack, mike, rose}；"
             "绑定：10.168.7.121 → username=mike、192.168.1.21 → username=jack、"
             "192.168.1.23 → username=rose（g3 = 0.00，各 ≥ 60 次）。"
             "流程：POST /login → GET /home（间隔 1 秒–5 秒）。"
             "置信 0.93 · 首次 2026-09-01 · 最近 2026-09-21 · v1.0")
GOLDEN_EN = ("[oa · 192.168.100.100:8080] On workdays 09:00–09:21 (coverage 97 %, 21 dates), "
             "综合部 (10.168.7.121, 192.168.1.21, 192.168.1.23) opens POST /login: "
             "90 % of submitted size within 1–2 KB, all within 0.5–3 KB (n = 183, P(next outside) ≤ 1.1 %); "
             "form keys always carry captcha=, password=, username=; username= matching `[a-z]{4}`, "
             "username= in the closed set {jack, mike, rose}; "
             "bound values 10.168.7.121 → username=mike, 192.168.1.21 → username=jack, "
             "192.168.1.23 → username=rose (g3 = 0.00, each ≥ 60 times). "
             "Workflow: POST /login → GET /home. "
             "Confidence 0.93, first seen 2026-09-01, last seen 2026-09-21, v1.0.")


def _login_stmt(v):
    return next(s for s in v["statements"] if s["evidence"]["route"] == LOGIN)


def test_golden_rendering_oa_example(store):
    v = VW.system_view(store, "oa", CFG, store.now)
    st = _login_stmt(v)
    assert st["text_zh"] == GOLDEN_ZH
    assert st["text_en"] == GOLDEN_EN
    assert st["state"] == "confirmed"
    assert 0.9 < st["confidence"] <= 1.0
    assert v["header"]["address"] == "192.168.100.100:8080"


def test_statement_contract_is_what_pmetrics_reads(store):
    v = VW.system_view(store, "oa", CFG, store.now)
    ls = LStmt(_login_stmt(v), "oa", {})
    assert (ls.method, ls.route) == ("POST", "/login")
    assert ls.confirmed and ls.who.ipset() == set(GA) and ls.who.closed
    assert ls.who.U < 0.05
    assert ls.when["workday"] == [(540.0, 561.0)]
    c = ls.content["body.len"]
    assert c["band90"] == [1024.0, 2048.0] and c["range"] == [512.0, 3072.0]
    assert ls.content["body.keys"]["required"] == ["captcha", "password", "username"]
    assert ls.content["body.kv.username"]["grammar"] == "[a-z]{4}"
    assert ls.binding_table("body.kv.username") == {ip: {u} for ip, u in USERS.items()}
    row = {"bindings": {"body.kv.username": dict(USERS)}}
    assert bindings_match(row, ls)[0]
    assert ls.workflow and ls.workflow[0]["from"] == LOGIN
    # the scorer collects the statement from a snapshot of the model
    snap = {"systems": {"oa": {"model.pviews": v}}}
    assert any(s.pattern_id == ls.pattern_id for s in statements(snap, {}))


def test_negative_group_statement(store):
    gv = VW.group_view(store, "G1", CFG, store.now)
    neg = [s for s in gv["statements"] if s["evidence"].get("negative")]
    assert len(neg) == 1
    s = neg[0]
    assert s["evidence"]["target_system"] == "finance"
    assert s["text_zh"].startswith("综合部 在 finance 中从未执行写操作（")
    assert "0 次" in s["text_zh"] and "POST /fin/approval/{num}/approve" in s["text_zh"]
    # positive statements restricted to the members, no negative one about its own system
    pos = [x for x in gv["statements"] if not x["evidence"].get("negative")]
    assert pos and all(set(x["evidence"]["who"]["items"]) <= set(GA) for x in pos)
    # finance's own group has no negative statement about finance
    fv = VW.group_view(store, "G2", CFG, store.now)
    assert not any(x["evidence"].get("negative") and x["evidence"]["target_system"] == "finance"
                   for x in fv["statements"])
    ls = group_statements({"group_views": {"class:grp:G1": gv}}, {})["class:grp:G1"]
    assert any(x.negative and x.target_system == "finance" for x in ls)


def test_engine_materialises_views_and_versions(store):
    eng = VW.ViewsEngine()
    eng.safe_run(ctx(store, store.now, window_s=3600.0, config=CFG), None)
    v = store.get_model("oa", SYSTEM_ENTITY, MP.PVIEWS)
    gv = store.get_model(ORG, "class:grp:G1", MP.PVIEWS)
    assert v["version"] == 1 and gv is not None
    assert store.profile_versions("oa", SYSTEM_ENTITY)
    # not re-rendered before the 2-h period; re-rendered unchanged -> same version
    eng.safe_run(ctx(store, store.now + 3 * 3600.0, window_s=3600.0, config=CFG), None)
    assert store.get_model("oa", SYSTEM_ENTITY, MP.PVIEWS)["version"] == 1
    assert "class:grp:G1" in store.pseudo_entities(ORG)


def test_who_rendering_levels():
    from app.engines.behavior.lib import pnode as PN
    t = T0
    # an open population: any IP
    w = PN.WhoSummary()
    rng = np.random.default_rng(0)
    for i in range(400):
        ip = f"10.{60 + i % 3}.{rng.integers(0, 255)}.{rng.integers(1, 255)}"
        w.update([ip, None, None, None, None], ip, t + i, 1.0, 1.0)
    ev, zh, en, _ = PR.who_block(w, t + 400, 10, {}, {})
    assert ev["level"] == "any" and zh.startswith("任意 IP")
    # prefixes: the same population keyed at /16
    w2 = PN.WhoSummary()
    for i in range(400):
        ip = f"10.{60 + i % 3}.{rng.integers(0, 255)}.{rng.integers(1, 255)}"
        w2.update([ip, None, f"10.{60 + i % 3}.0.0/16", None, None], ip, t + i, 1.0, 1.0)
    ev, zh, en, _ = PR.who_block(w2, t + 400, 10, {}, {})
    assert ev["level"] == "prefix" and set(ev["items"]) == {"10.60.0.0/16", "10.61.0.0/16", "10.62.0.0/16"}
    # a single-IP closed node
    w3 = PN.WhoSummary()
    for d in range(30):
        w3.update(["192.168.2.10", None, None, None, None], "192.168.2.10", t + d * DAY, 1.0, 1.0)
    ev, zh, en, c = PR.who_block(w3, t + 30 * DAY, 20, {}, {})
    assert ev["level"] == "ip" and ev["items"] == ["192.168.2.10"] and ev["closed"]
    assert c == pytest.approx(1 - ev["U"])


def test_number_rounding_helpers():
    assert PR.pct(0.011) == "1 %"
    assert PR.pct(2 / 184, up=True) == "1.1 %"          # a bound is rounded up
    assert PR.pct(0.1234, up=True) == "13 %"
    assert PR.pct(0.97) == "97 %"
    assert PR.pct(0.004) == "0.4 %"
    assert PR.route_text("POST oa.corp.local /approval/{num}/approve") == "POST /approval/{num}/approve"
    assert PR.route_text("TLS mail.corp") == "TLS mail.corp"
    assert PR.attr_label("body.kv.username") == "username="
    assert PR.attr_label("meta.waf.score", "en") == "waf.score"
    assert PR.attr_label("x.unknown") == "x.unknown"


def test_route_index_names_the_action_of_any_split(store):
    """A node whose context does not name a route (a /24 split above the route
    split, a size bin) stands for the route that holds >= 90 % of its events."""
    ix = VW.RouteIndex()
    tr = MP.get_ptree(store, "oa").kinds[0]
    leaf = tr.nodes[tr.root].split.children[0]
    other = tr.nodes[tr.root].split.other
    for i in range(50):
        ix.add(0, leaf, LOGIN, T0 + i, 1.0)
    for i in range(10):
        ix.add(0, other, HOME, T0 + i, 1.0)
    d = ix.subtree(tr, 0, T0 + 60)
    assert VW._dominant(d[leaf]) == LOGIN
    assert VW._dominant(d[tr.root]) is None               # 50 / 60 < 90 %
    assert VW._dominant(d[other]) == HOME


def test_statements_become_more_certain_with_observation_time():
    """S3: the same pattern observed longer is rendered with a closed who set
    and a smaller unseen-source mass (the confidence channel), never looser."""
    us, closed = [], []
    for days in (4, 8, 16, 32):
        st = make_store()
        m = _build_tree("oa", LOGIN, GA, "G1", days, "192.168.100.100:8080")
        st.put_model("oa", SYSTEM_ENTITY, MP.PTREE, m, version=1)
        st.put_model(ORG, ORG, MP.WHO_GROUPS, {"groups": {"G1": {"id": "G1", "name": "综合部", "members": GA}},
                                               "ip2g": {ip: "G1" for ip in GA}})
        v = VW.system_view(st, "oa", CFG, m.t_last + 3600.0)
        s = _login_stmt(v)
        us.append(s["evidence"]["who"]["U"])
        closed.append(s["evidence"]["who"]["closed"])
    assert closed == [False, True, True, True]           # >= 5 active days before a closed who set
    assert all(us[i + 1] < us[i] for i in range(len(us) - 1))


def test_view_cost_bounded_in_ips_and_attributes():
    """PPC-3 / S1: a view reads the tree (<= N_max nodes) and the fitted models,
    never a per-IP structure: rendering a node learned from 20 000 source IPs
    costs what one learned from 3 costs, and fitted records of 300 attributes
    registered elsewhere do not enter a node's statement unless fitted there."""
    import time as _t
    t_small = t_big = 0.0
    lens = []
    for n_ips in (3, 20000):
        st = make_store()
        ips = GA if n_ips == 3 else [f"10.{60 + i // 60000}.{(i // 250) % 256}.{i % 250 + 1}"
                                     for i in range(n_ips)]
        m = PT.PTreeModel("oa")
        tr = m.tree(EV.KIND_TXN, T0, create=True)
        sp = tr.split(tr.root, "http.route", 0, [[LOGIN]], T0)
        nd = tr.nodes[sp.children[0]]
        for k, ip in enumerate(ips):
            t = T0 + (k % 21) * DAY + 9 * 3600
            nd.update_core(t, 1.0, 1.0, _keys(ip, "G1"), ip, 0, 540.0, int((t + 8 * 3600) // DAY))
        nd.state = "confirmed"
        st.put_model("oa", SYSTEM_ENTITY, MP.PTREE, m, version=1)
        reg_noise = {"fmt": 1, "nodes": {0: {tr.root: {"status": "fitted", "attrs": {
            f"meta.f{j:03d}": {"kind": "num", "band90": [0.0, 1.0], "n_c": 50.0, "confidence": 0.9}
            for j in range(300)}}}}}
        st.put_model("oa", SYSTEM_ENTITY, MP.PBOUNDS, reg_noise)
        x = _t.perf_counter()
        v = VW.system_view(st, "oa", CFG, T0 + 22 * DAY)
        dt_ = _t.perf_counter() - x
        s = _login_stmt(v)
        lens.append(len(s["text_zh"]))
        assert not any(a.startswith("meta.f") for a in s["evidence"]["content"])
        assert len(s["evidence"]["who"]["items"]) <= PR.IP_LIST_MAX
        if n_ips == 3:
            t_small = dt_
        else:
            t_big = dt_
    assert t_big < 5 * t_small + 0.02
    assert lens[1] < 3 * lens[0]
