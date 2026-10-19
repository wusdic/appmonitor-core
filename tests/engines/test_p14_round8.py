"""P14 round 8: per-facet statement confidence. A moving time window lowers
only the 'when' facet: the statement WITH the window is 'evolving' at the
window's confidence, and the node's other clauses (who, content, bindings,
workflow) are stated on their own ('~when' facet statement) at their own
confidence (pack O seed 2, days 12-15: D1 moved 综合部's login window, the
login statements were 'evolving' at 0.016-0.019 and their intact bindings were
stated nowhere: PG1 'GA + FIN bindings 6/6 at day 14' 3/6)."""
from __future__ import annotations

import pytest

from helpers import make_store

from app.engines.behavior import views as VW
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.eval import pmetrics as PM

from test_p14_views import T0

GA = {"192.168.1.21": "jack", "192.168.1.23": "rose", "10.168.7.121": "mike.w"}


def _tree(monkeypatch, drift: bool, content: bool = False, tested: bool = False):
    from app.engines.behavior.lib import ptree as PT
    from app.engines.behavior.lib import pnode as PN
    from app.models.schema import ORG, SYSTEM_ENTITY
    ga = list(GA)
    m = PT.PTreeModel("oa")
    tr = m.tree(EV.KIND_TXN, T0, create=True)
    sp = tr.split(tr.root, "http.route", 0, [["POST oa.corp /login"]], T0)
    node, root = tr.nodes[sp.children[0]], tr.nodes[tr.root]
    t = T0
    for d in range(14):
        for k, ip in enumerate(ga):
            t = T0 + d * 86400.0 + (9 * 60 + 7 * k) * 60.0
            p = ip.split(".")
            keys = [ip, ".".join(p[:3]) + ".0/24", ".".join(p[:2]) + ".0.0/16", "grp:G1", "reg:∅"]
            for nd in (root, node):
                nd.update_core(t, 1.0, 1.0, keys, ip, 0, (t + 8 * 3600) % 86400 / 60.0,
                               int((t + 8 * 3600) // 86400))
    for nd in (root, node):
        nd.state = "confirmed"
    orig = PN.Node.p_hold
    monkeypatch.setattr(PN.Node, "p_hold", lambda self, _t: 0.8 if self is node else orig(self, _t))
    if tested:
        # P04 has judged the node on held-out events: 9 of 10 passed, all of
        # them without the time windows
        hr = PN.HoldRecord()
        hr.T = [9.0, 10.0]
        hr.TF = {"~when": [10.0, 10.0]}
        node.meta["hold"] = hr
        node.meta["hold_prior"] = (0.95, 0.05)
    m.t_last = t
    st = make_store()
    st.put_model("oa", SYSTEM_ENTITY, MP.PTREE, m, version=1)
    st.put_model(ORG, ORG, MP.WHO_GROUPS, {
        "groups": {"G1": {"id": "G1", "name": "综合部", "members": ga}},
        "ip2g": {ip: "G1" for ip in ga}, "mode": {"oa": {"mode": "ip"}}})
    rec = {"windows": [[540, 562]], "coverage": 0.9, "confidence": 0.02 if drift else 0.9,
           "text_zh": "工作日 09:00–09:22", "text_en": "workdays 09:00-09:22"}
    when = {"workday": [[540, 562]], "nonworkday": [], "coverage": 0.9,
            "confidence": 0.02 if drift else 0.9}
    if drift:
        rec["drift"] = when["drift"] = {"workday": {"p": 0.001, "dates_after": 1}}
    st.put_model("oa", SYSTEM_ENTITY, MP.PWIN, {"version": 1, "nodes": {EV.KIND_TXN: {node.id: {
        "status": "fitted", "when": when, "by_daytype": {"wd": rec}}}}}, version=1)
    table = {ip: {"n": 40.0, "top": u, "k": 40.0, "bound": True, "LB": 0.99} for ip, u in GA.items()}
    st.put_model("oa", SYSTEM_ENTITY, MP.PBIND, {"fmt": 1, "nodes": {EV.KIND_TXN: {node.id: {
        "status": "fitted", "pairs": {"net.src->body.kv.username": {
            "x": "net.src", "y": "body.kv.username", "dir": "fwd",
            "fd": {"g3": 0.0, "holds": True, "n": 120.0}, "table": table}}}}}})
    if content:
        # a young node's content: a length band from 7 rows (nominal coverage < 0.8)
        from app.engines.behavior.lib import pbounds as PB
        num = PN.NumSummary(log=True)
        for i, x in enumerate((900.0, 1010.0, 1150.0, 1240.0, 1400.0, 1610.0, 1790.0)):
            num.update(x, T0 + i * 86400.0, 1.0, 1.0, day=739000 + i)
        rec = PB.fit_numeric(num, t + 3600.0, PB.local_day(t + 3600.0, {"tz": "Asia/Shanghai"}),
                             7.0, 7.0, 0.0, "B")
        st.put_model("oa", SYSTEM_ENTITY, MP.PBOUNDS, {"version": 1, "nodes": {EV.KIND_TXN: {node.id: {
            "status": "fitted", "attrs": {"body.len": rec}}}}}, version=1)
    v = VW.system_view(st, "oa", {"progressive": {"enabled": True}, "tz": "Asia/Shanghai"}, t + 3600.0)
    return [s for s in v["statements"] if s["evidence"]["route"] == "POST oa.corp /login"
            and s["evidence"]["node"] == node.id and "|grp:" not in s["id"]]


def test_a_drifting_window_lowers_only_the_when_facet(monkeypatch):
    sts = _tree(monkeypatch, drift=True)
    evo = [s for s in sts if s["evidence"].get("facet") != "~when"]
    fac = [s for s in sts if s["evidence"].get("facet") == "~when"]
    assert len(evo) == 1 and len(fac) == 1, "the bindings of a drifting statement are stated nowhere"
    e, f = evo[0], fac[0]
    # the statement with the window: evolving at the window's confidence
    assert e["state"] == "evolving" and e["confidence"] <= 0.02 + 1e-9
    assert e["evidence"]["facet_conf"]["when"] <= 0.02 + 1e-9
    assert e["evidence"]["facet_conf"]["bindings"] == pytest.approx(0.8)
    assert "时间置信" in e["text_zh"]
    # its other clauses: their own statement, confirmed at their own confidence
    assert f["state"] == "confirmed" and f["confidence"] == pytest.approx(0.8)
    assert "when" not in f["evidence"] and f["evidence"]["time_clause"]["of"] == e["id"]
    assert f["evidence"]["bindings"]["body.kv.username"]["table"] == GA
    assert f["evidence"]["who"] == e["evidence"]["who"]
    assert "temporal" not in f["facets"] and "content.bindings" in f["facets"]
    assert f["id"] == e["id"] + "|~when" and "10.168.7.121 → username=mike.w" in f["text_zh"]
    # the evaluator reads it as a confirmed statement carrying the bindings
    ls = PM.LStmt(f, "oa", {})
    assert ls.confirmed and ls.binding_table("body.kv.username")["192.168.1.21"] == {"jack"}


def test_no_facet_statement_without_drift(monkeypatch):
    sts = _tree(monkeypatch, drift=False)
    assert len(sts) == 1 and sts[0]["state"] == "confirmed"
    assert sts[0]["evidence"]["facet_conf"]["who"] == pytest.approx(0.8)
    assert "时间置信" not in sts[0]["text_zh"]


def test_rest_conf_reads_the_without_when_facet_record():
    class HR:
        def facet_tests(self):
            return {"~when": (18.0, 20.0)}

    class Nd:
        hold = HR()
        meta = {"hold_prior": (1.0, 1.0)}

        def p_hold(self, t):
            return 0.3

    assert VW.rest_conf(Nd(), 0.0, 0.5) == pytest.approx(19.0 / 22.0)
    Nd.hold = None
    assert VW.rest_conf(Nd(), 0.0, 0.5) == pytest.approx(0.3)


def test_an_edge_p10_holds_under_hysteresis_is_rendered():
    """(round 8) P10 keeps an accepted edge whose point dependency dipped
    under the cut while its upper evidence bound did not ('held'); P14 must
    state it (with the flag) rather than re-filter on the point value (pack O
    seed 1, GA.oa.approvals list -> item held at dep 0.754 on day 9 was in
    P10's kept set and in no statement)."""
    from app.engines.behavior.lib import prender as PR
    ctx = VW._Ctx.__new__(VW._Ctx)
    lst, item, appr = "GET oa.corp /approval/list", "GET oa.corp /approval/{num}", \
        "POST oa.corp /approval/{num}/approve"
    ctx.pflow = {"scopes": {"*": {"edges": [
        {"from": lst, "to": item, "dep": 0.7544, "held": True, "dep_upper": 0.8823,
         "band": [45.5, 273.7], "confidence": 0.75},
        {"from": item, "to": appr, "dep": 0.8057, "band": [43.8, 381.3], "confidence": 0.8},
        {"from": appr, "to": lst, "dep": 0.654, "band": [10.0, 60.0], "confidence": 0.6}]}}}
    edges = ctx._edges()
    assert [e["to"] for e in edges["GET oa.corp /approval/{num}"]] == [item, appr]
    assert all(e["to"] != lst for es in edges.values() for e in es)   # never kept: not stated
    ev, zh, en, f_c = PR.workflow_block(edges["GET oa.corp /approval/{num}"])
    held = [e for e in ev if e["to"] == item]
    assert held and held[0]["held"] and held[0]["dep_upper"] == 0.8823 and held[0]["dep"] == 0.7544
    assert "/approval/list" in zh


def test_an_untested_node_states_only_its_identity_facets_apart(monkeypatch):
    """(round 8) A node P04 has not judged on held-out events yet has no record
    of its clauses (its p_hold is its kind's prior mean): its '~when' facet
    statement carries who / bindings / workflow at their own confidences and
    leaves the content - nominal coverages from a handful of rows - to the
    statement with the window. O-real seed 2, day 14: the family's young login
    node (0 tests, n = 7) stated its content apart at the prior mean 0.95; the
    password grammar held 0.5 of held-out events (4 of 16 failing statements)."""
    sts = _tree(monkeypatch, drift=True, content=True, tested=False)
    evo = [s for s in sts if s["evidence"].get("facet") != "~when"]
    fac = [s for s in sts if s["evidence"].get("facet") == "~when"]
    assert len(evo) == 1 and len(fac) == 1
    e, f = evo[0], fac[0]
    c_c = e["evidence"]["content"]["body.len"]["confidence"]
    assert c_c < 0.8                                         # the young band's own confidence
    # the evolving statement keeps the content; its content facet is bounded by it
    assert e["evidence"]["content"] and e["evidence"]["facet_conf"]["content"] <= c_c + 1e-4
    assert e["evidence"]["facet_conf"]["bindings"] == pytest.approx(0.8)
    # the facet statement: identity facets only, confirmed, at their confidence
    assert f["state"] == "confirmed" and f["confidence"] == pytest.approx(0.8)
    assert not f["evidence"]["content"] and f["evidence"].get("content_untested")
    assert "content" not in f["facets"] and "content.bindings" in f["facets"]
    assert f["evidence"]["bindings"]["body.kv.username"]["table"] == GA
    ls = PM.LStmt(f, "oa", {})
    assert ls.confirmed and ls.binding_table("body.kv.username")["192.168.1.21"] == {"jack"}


def test_a_tested_node_states_its_content_apart_at_its_record(monkeypatch):
    sts = _tree(monkeypatch, drift=True, content=True, tested=True)
    fac = [s for s in sts if s["evidence"].get("facet") == "~when"]
    assert len(fac) == 1
    f = fac[0]
    assert f["evidence"]["content"] and "content" in f["facets"]
    assert not f["evidence"].get("content_untested")
    # P04's record of the node without its windows: (10 + 0.95) / (10 + 1)
    assert f["confidence"] == pytest.approx(10.95 / 11.0)


def test_held_out_tested():
    from app.engines.behavior.lib import pnode as PN

    class Nd:
        hold = None
    assert not VW.held_out_tested(Nd())
    Nd.hold = PN.HoldRecord()
    assert not VW.held_out_tested(Nd())
    Nd.hold.T = [1.0, 1.0]
    assert VW.held_out_tested(Nd())
