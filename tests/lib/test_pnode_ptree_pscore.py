"""lib/pnode.py, lib/ptree.py, lib/pscore.py, lib/m_ptree.py: the pattern /
node data model, tree operations with lineage, routing with generalisation
hierarchies, bounded memory, pure scoring functions and store accessors."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.core.store import MetricStore
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import phier as H
from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.lib import pscore as SC
from app.engines.behavior.lib import psketch as PS
from app.engines.behavior.lib import ptree as PT

DAY = PS.DAY
T0 = 1_700_000_000.0


def _who_keys(h, ip):
    return [h.gen("net.src", l, ip) for l in range(5)]


def test_who_summary_closes_on_three_daily_sources_and_opens_on_churn():
    h = H.Hierarchies({"net.src": {"type": "ip"}})
    who = PN.WhoSummary()
    ips = ["192.168.1.21", "192.168.1.23", "10.168.7.121"]
    for d in range(30):
        for ip in ips:
            who.update(_who_keys(h, ip), ip, T0 + d * DAY, 1.0, 1.0)
    t = T0 + 30 * DAY
    assert who.closed_level(t, n_days=30) == 0
    heavy, cov = who.heavy_set(0, t)
    assert sorted(heavy) == sorted(ips) and cov > 0.99
    assert who.closed_level(t, n_days=4) is None                 # < 5 normal days
    p, l, U = SC.who_p(who, _who_keys(h, "192.168.1.23"), t, 30)
    assert p == 1.0 and l == 0 and U < 0.01
    p, l, U = SC.who_p(who, _who_keys(h, "192.168.2.10"), t, 30)
    assert p == pytest.approx(U) and p < 0.01
    portal = PN.WhoSummary()
    for i in range(2000):
        ip = f"10.{i % 7}.{(i * 13) % 250}.{(i * 29) % 250}"
        portal.update(_who_keys(h, ip), ip, T0 + i * 60, 1.0, 1.0)
    assert portal.closed_level(T0 + 2000 * 60, n_days=30, levels=[0, 1]) is None
    p, _, _ = SC.who_p(portal, _who_keys(h, "8.8.8.8"), T0 + 2000 * 60, 30, levels=[0, 1])
    assert math.isnan(p)                                          # open population: NaN
    assert 1500 < portal.distinct(T0 + 2000 * 60) < 2600          # HLL p=6 (~13 %)


def test_who_code_lengths_prefer_prefix_under_churn():
    """§6.18.2: under churn the /32 level keeps paying escapes + 32 bits."""
    h = H.Hierarchies({"net.src": {"type": "ip"}})
    who = PN.WhoSummary()
    space = [0.0, 8.0, 16.0, 0.0, 0.0]
    esc = [32.0, 24.0, 16.0, 1.0, 1.0]
    for i in range(3000):
        ip = f"10.1.{i % 4}.{(i * 7919) % 250}"                  # DHCP-like pool of 4 /24s
        keys = _who_keys(h, ip)
        code = who.code_lengths(keys, T0 + i * 30, space, esc)
        who.update(keys, ip, T0 + i * 30, 1.0, 1.0, code)
    assert who.code[1] < who.code[0]


def test_when_summary_and_when_p():
    rng = np.random.default_rng(0)
    w = PN.WhenSummary()
    for d in range(30):
        for _ in range(3):
            w.update(0, 540 + rng.uniform(0, 21), T0 + d * DAY, 1.0, 1.0)
    t = T0 + 30 * DAY
    assert w.evidence(0, t) >= 60                                 # confidence channel
    assert SC.when_p(w, 0, 545, t) > 0.2
    # 03:05 vs 09:00-09:21 from >= 60 units: 0.5 * 94 * (0.1/96) / (N + 0.1) ~ 7.5e-4
    assert SC.when_p(w, 0, 185, t) <= 1e-3
    assert math.isnan(SC.when_p(PN.WhenSummary(), 0, 185, t))


def test_num_summary_ring_range_and_reset():
    num = PN.NumSummary(log=False)
    for d in range(10):
        for x in (1000.0 + d, 2000.0 - d):
            num.update(x, T0 + d * DAY, 1.0, 1.0, day=100 + d)
    mn, mx, n = num.observed_range(109)
    assert (mn, mx, n) == (1000.0, 2000.0, 20.0)
    num.reset_confidence(T0 + 10 * DAY, day=110)
    num.update(1500.0, T0 + 10 * DAY, 1.0, 1.0, day=110)
    assert num.observed_range(110) == (1500.0, 1500.0, 1.0)
    w, mu, var = num.moments(T0 + 10 * DAY, PS.CH_M)
    assert mu == pytest.approx(1500.0, abs=10)


def test_cat_p_and_invariant():
    c = PN.CatSummary()
    for i in range(100):
        c.update("GET" if i % 10 else "POST", T0, 1.0, 1.0)
    assert SC.cat_p(c, "GET", T0) > SC.cat_p(c, "POST", T0) > SC.cat_p(c, "DELETE", T0)
    assert SC.cat_p(c, "DELETE", T0) < 0.02
    inv = PN.CatSummary()
    for i in range(40):
        inv.update("oa:8080", T0, 1.0, 1.0)
    assert inv.invariant(T0) == "oa:8080" and c.invariant(T0) is None


def test_text_set_pair_summaries():
    t = PN.TextSummary()
    for v in ("jack", "rose", "mike") * 10:
        t.update(v, T0, 1.0, 1.0)
    assert t.shapes.items(T0)[0][0] == "L4"
    assert len(t.values) == 3
    ts = PN.TextSummary(policy="shape")
    ts.update("L8 D2", T0, 1.0, 1.0)
    assert ts.values is None and ts.shapes.items(T0)[0][0] == "L8 D2"
    s = PN.SetSummary()
    for _ in range(50):
        s.update(frozenset({"username", "password"}), T0, 1.0, 1.0)
    s.update(frozenset({"username", "password", "captcha"}), T0, 1.0, 1.0)
    pres = s.presence(T0)
    assert pres["username"] == pytest.approx(1.0) and pres["captcha"] < 0.05
    p = PN.PairSketch()
    for ip, u in (("1.1.1.21", "jack"), ("1.1.1.23", "rose")) * 5:
        p.update(ip, u, T0, 1.0, 1.0)
    tab = p.table(T0)
    assert tab["1.1.1.21"][0][0] == "jack" and tab["1.1.1.21"][0][1] == pytest.approx(5.0)
    big = PN.PairSketch()
    for i in range(1000):
        big.update(f"ip{i}", "u", T0, 1.0, 1.0)
    assert len(big.y) <= PN.PAIR_KX                               # y tables follow x's cap


def _route_event(route, ip, tod):
    d = {"http.route": route, "net.src": ip, "ctx.tod_min": tod}
    return lambda a: d.get(a, EV.ABSENT)


def test_tree_split_route_other_branch_exceptions_and_lineage():
    h = H.Hierarchies({"net.src": {"type": "ip"}})
    tr = PT.Tree(EV.KIND_TXN, T0)
    sp = tr.split(tr.root, "http.route", 0, [{"POST oa /login"}, {"GET oa /approval/list"}], T0)
    login = sp.children[0]
    assert tr.route(_route_event("POST oa /login", "1.1.1.1", 540), h) == [tr.root, login]
    assert tr.route(_route_event("PUT oa /new", "1.1.1.1", 540), h)[-1] == sp.other
    assert tr.route(lambda a: EV.ABSENT, h)[-1] == sp.other           # absence routes too
    sp2 = tr.split(login, "net.src", 1, [{"192.168.1.0/24"}], T0 + 1)
    assert tr.route(_route_event("POST oa /login", "192.168.1.21", 540), h)[-1] == sp2.children[0]
    assert tr.node(sp2.children[0]).ctx == (
        ("http.route", 0, frozenset({"POST oa /login"}), False),
        ("net.src", 1, frozenset({"192.168.1.0/24"}), False))
    assert tr.node(sp2.other).ctx[-1][3] is True                       # negated `other`
    exc = tr.add_exception(sp2.children[0], "192.168.1.21", T0 + 2)
    assert tr.node(exc).is_exc and tr.node(sp2.children[0]).exc == {"192.168.1.21": exc}
    ops = [x[1] for x in tr.lineage]
    assert ops == ["create", "split", "split", "exc_add"]
    ids_before = set(tr.nodes)
    retired = tr.collapse(login, T0 + 3)
    assert set(retired) == {sp2.children[0], sp2.other, exc}
    assert login in tr.nodes and tr.node(login).split is None
    new = tr.split(login, "ctx.tod_min", 1, [{36}], T0 + 4)
    assert not (set(new.children) | {new.other}) & ids_before          # ids never reused
    assert len(tr.retired) == 3
    # gone split attribute: route by the heaviest child
    tr.node(new.other).mass.add(T0 + 5, 10.0)
    assert tr.route(_route_event("POST oa /login", "x", 36), h, gone={"ctx.tod_min"},
                    mass_t=T0 + 5)[-1] == new.other


def test_sibling_merge_and_retire_leaf():
    h = H.Hierarchies({})
    tr = PT.Tree(EV.KIND_TXN, T0)
    sp = tr.split(tr.root, "http.method", 0, [{"GET"}, {"POST"}, {"PUT"}], T0)
    g, p, u = sp.children
    for nid, n in ((g, 5), (p, 3)):
        for _ in range(n):
            tr.node(nid).update_core(T0, 1.0, 1.0, [None] * 5, "ip")
    tr.merge_siblings(tr.root, g, p, T0 + 1)
    assert tr.node(g).mass_at(T0) == pytest.approx(8.0)
    assert tr.route(lambda a: "POST" if a == "http.method" else EV.ABSENT, h)[-1] == g
    tr.retire_leaf(u, T0 + 2)
    assert tr.route(lambda a: "PUT" if a == "http.method" else EV.ABSENT, h)[-1] == sp.other
    with pytest.raises(ValueError):
        tr.retire_leaf(sp.other, T0 + 3)


def test_node_memory_bounded_under_many_ips_and_values():
    """(i)-style: per-node memory does not grow with the number of IPs or values."""
    h = H.Hierarchies({"net.src": {"type": "ip"}})
    nd = PN.Node(0, None, 0, 0, (), T0)
    rng = np.random.default_rng(0)
    for i in range(200):
        ip = f"10.{i % 256}.{(i * 7) % 256}.{(i * 13) % 256}"
        nd.update_core(T0 + i, 1.0, 1.0, _who_keys(h, ip), ip, 0, 540.0, 1000)
        nd.update_target("http.status", int(rng.choice([200, 302])), T0 + i, 1.0, 1.0)
        nd.update_target("net.bytes_up", float(rng.lognormal(7, 1)), T0 + i, 1.0, 1.0, "num")
        nd.update_target("body.kv.username", f"user{i}", T0 + i, 1.0, 1.0, "text")
    nb0 = nd.nbytes()
    for i in range(200, 10200):
        ip = f"10.{i % 256}.{(i * 7) % 256}.{(i * 13) % 256}"
        nd.update_core(T0 + i, 1.0, 1.0, _who_keys(h, ip), ip, 0, 540.0, 1000)
        nd.update_target("http.status", int(rng.choice([200, 302])), T0 + i, 1.0, 1.0)
        nd.update_target("net.bytes_up", float(rng.lognormal(7, 1)), T0 + i, 1.0, 1.0, "num")
        nd.update_target("body.kv.username", f"user{i}", T0 + i, 1.0, 1.0, "text")
    assert nd.nbytes() <= nb0 * 1.25 + 4096
    assert nd.nbytes() < 64_000


def test_confidence_channel_grows_and_resets():
    """(j)-style: 3 units per workday; n_c keeps growing past the H_m plateau,
    and an accepted change restarts it from the H_m state."""
    nd = PN.Node(0, None, 0, 0, (), T0)
    for d in range(60):
        for _ in range(3):
            nd.update_core(T0 + d * DAY, 1.0, 1.0, [None] * 5, "ip")
    t = T0 + 60 * DAY
    assert nd.n_c(t) > 2.5 * nd.n_m(t)
    nd.reset_confidence(t)
    assert nd.n_c(t) == pytest.approx(nd.n_m(t))


def test_days_bitmap():
    nd = PN.Node(0, None, 0, 0, (), T0)
    for d in (100, 101, 103, 103, 170):
        nd.touch_day(d)
    assert nd.days_total == 4 and nd.n_days() == 1                  # 170 shifted out 100-103
    nd2 = PN.Node(1, None, 0, 0, (), T0)
    for d in (10, 12, 11):
        nd2.touch_day(d)
    assert nd2.n_days() == 3 and nd2.days_total == 3


def test_pattern_id_roundtrip():
    pid = PN.pattern_id("fam:3", 0, 17, 4, 2)
    assert pid == "p:fam:3:0:17@4.2"
    assert PN.parse_pattern_id(pid) == ("fam:3", 0, 17, 4, 2)


def test_score_combinators():
    assert SC.dual_anchor(0.2, float("nan")) == 0.2
    assert SC.dual_anchor(0.2, 0.01) == pytest.approx(0.02)
    assert SC.content_p([0.1, float("nan"), 0.01]) == pytest.approx(0.02)
    assert SC.event_p([0.5, 1e-4]) == pytest.approx(5e-4)
    assert math.isnan(SC.event_p([float("nan")]))
    assert SC.day_p(1e-3, 500) == pytest.approx(1 - 0.999 ** 500)
    assert SC.vtype_mask({"who": 1e-4, "content": 0.5, "seq": 1e-3}) == 0b1001
    assert SC.invariant_p(0, 99) == pytest.approx(0.005)


def test_m_ptree_accessors_and_family_key():
    st = MetricStore()
    assert MP.tree_key(st, "oa1") == "oa1"
    st.put_model("__org__", "__org__", MP.SYSFAM, {"member": {"oa1": "fam:1", "oa2": "fam:1"},
                                                  "dst_members": {"fam:1": {"10.0.0.1:80": "oa1"}}})
    assert MP.tree_key(st, "oa2") == "fam:1"
    m = MP.ensure_ptree(st, "fam:1", T0)
    tr = m.tree(EV.KIND_TXN, T0)
    assert MP.get_ptree(st, "fam:1") is m and tr.kind == 0
    reg = MP.ensure_registry(st, "fam:1", {"progressive": {"type_hints": {"code": ["status"]}}})
    assert reg.code_hints == ("status",) and MP.get_registry(st, "fam:1") is reg
    st.put_model("__org__", "__org__", MP.WHO_GROUPS, {"ip2g": {"1.1.1.1": "GA"},
                                                      "groups": {"GA": {}}})
    st.put_model("fam:1", "__system__", MP.PWIN, {"root": {"all": [(540, 561, "w:0900-0921")]}})
    h = MP.hierarchies(st, "fam:1", {"ip_classes": [{"name": "HQ", "cidrs": ["1.1.0.0/16"]}]})
    assert h.gen("net.src", 3, "1.1.1.1") == "grp:GA"
    assert h.gen("net.src", 4, "1.1.1.1") == "reg:HQ"
    assert h.gen("ctx.tod_min", 2, 550) == "w:0900-0921"
    assert h.gen("net.dst", 1, "10.0.0.1:80") == "sys:oa1"
