"""API v3 — the progressive profile core ("画像模式", docs/lib3/progressive.md §9.4).

Two fixtures:
  * `live`: the Runtime main.py starts with APPMON_PROGRESSIVE=decision
    (api/progressive_runtime.py: pack O, P-core + B24-B29), warmed up a few
    simulated days at 900 s and then run for live 60-s ticks. Every endpoint
    must answer 200 with its documented keys and strict JSON on real models,
    unknown keys must 404, and the lattice / pattern / view projections must
    be consistent with each other.
  * `golden`: the requirement's OA example as fitted models (the P14 golden
    fixture: 综合部's three IPs log in to OA 09:00-09:21 with 1-2 KB forms and
    username bindings jack / mike / rose; the finance approver), materialised
    by the real P14 / P13 engines, plus P03 / P04 events. The projections must
    carry the who / when / content / binding / workflow blocks, the negative
    group statement, the IP's binding, the drift history and typed violations.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import sys
import threading
import types
from typing import Any, Dict, List

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from asgi_client import make_client  # noqa: E402

from app.api import progressive_runtime as PRT  # noqa: E402
from app.api import progressive_views as PV  # noqa: E402
from app.api import routes  # noqa: E402
from app.core.engine import Context, default_config  # noqa: E402
from app.core.store import MetricStore  # noqa: E402
from app.engines.behavior import facets as P13  # noqa: E402
from app.engines.behavior import views as P14  # noqa: E402
from app.engines.behavior.lib import m_ptree as MP  # noqa: E402
from app.engines.behavior.lib import pevent as EV  # noqa: E402
from app.engines.behavior.lib import pnode as PN  # noqa: E402
from app.engines.behavior.lib import ptree as PT  # noqa: E402
from app.main import app  # noqa: E402
from app.models.schema import ORG, SYSTEM_ENTITY, BehaviorEvent, Severity  # noqa: E402

DAYS = float(os.environ.get("APPMON_TEST_PROG_DAYS", "4"))
LIVE = 2


def _no_nan(obj: Any) -> None:
    if isinstance(obj, float):
        assert math.isfinite(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _no_nan(v)
    elif isinstance(obj, list):
        for v in obj:
            _no_nan(v)


def _has(d: Dict[str, Any], keys: List[str]) -> None:
    missing = [k for k in keys if k not in d]
    assert not missing, f"missing keys {missing} in {sorted(d)}"


def _get(client, path, status=200):
    resp = client.get(path)
    assert resp.status_code == status, (path, resp.status_code, resp.text[:500])
    d = resp.json()
    if status == 200:
        _no_nan(d)
    return d


# ====================================================================== live
@pytest.fixture(scope="module")
def live():
    r = PRT.make_progressive_runtime("progressive_decision", "O", days=DAYS, seed=0, live_period_s=0.0)
    r.warmup()
    for _ in range(LIVE):
        r.step_once()
    prev = routes.RUNTIME
    routes.RUNTIME = r
    yield r
    routes.RUNTIME = prev


@pytest.fixture(scope="module")
def lclient(live):
    return make_client(app)


def test_runtime_factory_and_env():
    assert PRT.resolve_mode(None) is None and PRT.resolve_mode("") is None and PRT.resolve_mode("0") is None
    assert PRT.resolve_mode("decision") == "progressive_decision"
    assert PRT.resolve_mode("only") == "progressive_only"
    assert PRT.resolve_mode("full") == "full+progressive"
    with pytest.raises(ValueError):
        PRT.resolve_mode("everything")
    assert PRT.runtime_from_env({}) is None
    with pytest.raises(ValueError):
        PRT.make_progressive_runtime(pack_name="A")          # not an organisation pack


def test_live_status_and_systems(lclient, live):
    d = _get(lclient, "/api/v3/status")
    _has(d, ["enabled", "running", "present", "registry_mode", "pack", "engines", "now", "now_local",
             "tz", "n_systems", "n_groups", "n_statements", "n_confident", "systems"])
    assert d["running"] and d["present"] and d["registry_mode"] == "progressive_decision"
    assert "behavior.pattern_tree" in d["engines"] and "behavior.views" in d["engines"]
    assert d["now"] == pytest.approx(live.gen.vt)
    names = {s["system"] for s in d["systems"]}
    assert {"oa", "finance", "portal"} <= names
    s = _get(lclient, "/api/v3/systems")
    for row in s["systems"]:
        _has(row, ["system", "tree_key", "has_tree", "nodes", "n_nodes", "n_confident", "n_statements",
                   "n_actions", "who_mode", "tier"])
        assert row["has_tree"] and row["n_nodes"] >= 1


def test_live_system_view_and_precision(lclient, live):
    for s in ("oa", "finance"):
        v = _get(lclient, f"/api/v3/systems/{s}/view")
        _has(v, ["system", "tree_key", "source", "version", "header", "who_mode", "n_statements",
                 "n_actions", "actions", "confidence"])
        assert "statements" not in v                        # grouped by action unless flat
        flat = _get(lclient, f"/api/v3/systems/{s}/view?flat=true&fresh=true")
        assert flat["source"] == "render"
        assert flat["n_statements"] == len(flat["statements"])
        for a in v["actions"]:
            _has(a, ["route", "route_text", "write", "share", "statements"])
            for st in a["statements"]:
                _has(st, ["id", "pattern_id", "text_zh", "text_en", "support", "confidence", "state",
                          "version", "who", "when", "content", "bindings", "workflow", "node"])
                assert st["state"] in PV.CONFIDENT
        p = _get(lclient, f"/api/v3/systems/{s}/precision")
        _has(p, ["system", "days", "now", "measured", "note_zh", "note_en"])
        assert p["days"], "no day of the live curve"
        assert sum(d["splits"] for d in p["days"]) >= 1    # the route-first partition at least
        conf = [d["confirmed"] for d in p["days"]]
        assert conf[-1] >= 1 and all(c >= 0 for c in conf)


def test_live_lattice_and_patterns_are_consistent(lclient, live):
    lat = _get(lclient, "/api/v3/systems/oa/lattice?depth=8")
    _has(lat, ["system", "tree_key", "kind", "kinds", "root", "n_nodes", "nodes", "truncated", "budget",
               "counts"])
    ids = {n["id"] for n in lat["nodes"]}
    assert lat["root"] in ids and len(ids) == len(lat["nodes"])
    if not lat["truncated"]:
        assert all(c in ids for n in lat["nodes"] for c in n["children"])
    tree = MP.get_ptree(live.store, "oa").kinds[0]
    assert lat["n_nodes"] == len(tree.nodes)
    routed = [n for n in lat["nodes"] if n["route"]]
    assert routed, "no node carries an action route"
    for n in lat["nodes"][:12]:
        d = _get(lclient, f"/api/v3/patterns/{n['pattern_id']}")
        _has(d, ["pattern_id", "tree_key", "kind", "node", "alive", "brief", "path", "parent", "children",
                 "exceptions", "who", "when_hist96", "constraints", "statement", "lineage", "lifecycle",
                 "violations"])
        assert d["node"] == n["id"] and d["path"][-1]["id"] == n["id"] and d["path"][0]["id"] == lat["root"]
        assert sorted(c["id"] for c in d["children"]) == sorted(n["children"])
        assert len(d["when_hist96"]["workday"]) == 96
    # a confirmed node's lifecycle carries its pattern_confirmed event
    conf = [n for n in lat["nodes"] if n["state"] in PV.CONFIDENT]
    assert conf
    d = _get(lclient, f"/api/v3/patterns/{conf[0]['pattern_id']}")
    assert any(e["kind"] == "pattern_confirmed" for e in d["lifecycle"])
    # subtree from a child, another kind, unknown node / pattern
    sub = _get(lclient, f"/api/v3/systems/oa/lattice?root={routed[0]['id']}&depth=1")
    assert sub["root"] == routed[0]["id"]
    _get(lclient, "/api/v3/systems/oa/lattice?kind=1")
    _get(lclient, "/api/v3/systems/oa/lattice?root=999999", 404)
    _get(lclient, "/api/v3/patterns/p:oa:0:999999@1.0", 404)
    _get(lclient, "/api/v3/patterns/garbage", 404)


def test_live_groups_ip_facets_strategy_attributes_budget(lclient, live):
    g = _get(lclient, "/api/v3/groups")
    _has(g, ["n_groups", "n_grouped_ips", "groups", "modes"])
    assert g["n_groups"] >= 1
    gid = g["groups"][0]["id"]
    gv = _get(lclient, f"/api/v3/groups/{gid}/view")
    _has(gv, ["group", "name", "members", "systems", "negative", "header"])
    gf = _get(lclient, f"/api/v3/groups/{gid}/facets")
    _has(gf, ["subject", "facets", "n_items", "registry", "source"])
    ip2g = MP.who_groups(live.store).get("ip2g") or {}
    ip = next(x for x in sorted(ip2g) if PV.valid_ip(x) and x in live.store.entities("oa")) \
        if any(PV.valid_ip(x) and x in live.store.entities("oa") for x in ip2g) else live.store.entities("oa")[0]
    iv = _get(lclient, f"/api/v3/systems/oa/entities/{ip}/view")
    _has(iv, ["system", "ip", "prefix24", "group", "inherited", "inherited_here", "exceptions", "bindings",
              "statements", "violations", "violation_counts", "known"])
    assert iv["known"]
    _get(lclient, f"/api/v3/systems/oa/entities/{ip}/facets")
    f = _get(lclient, "/api/v3/systems/oa/facets")
    assert f["n_facets"] >= 1 and {x["id"] for x in f["facets"]} & {"functional", "spatial", "content"}
    assert f["registry"] and all("producer" in d for d in f["registry"])
    s = _get(lclient, "/api/v3/systems/oa/strategy")
    _has(s, ["system", "tree_key", "characteristics", "chosen", "reasons", "engines", "who", "history",
             "hints"])
    assert s["chosen"] and s["engines"]
    assert {e["dim"] for e in s["engines"]} >= {"P06", "P07", "P08", "P09", "P10"}
    a = _get(lclient, "/api/v3/systems/oa/attributes")
    _has(a, ["system", "n", "attributes", "by_role", "by_type"])
    assert a["n"] >= 10 and {"http.route", "net.src"} & {x["name"] for x in a["attributes"]}
    b = _get(lclient, "/api/v3/budget")
    _has(b, ["trees", "ladder", "usage", "systems", "series", "engines", "ptree_bytes", "present"])
    assert b["present"] and "oa" in b["trees"] and b["ptree_bytes"].get("oa", 0) > 0
    assert any(e["engine"] == "behavior.pattern_tree" for e in b["engines"])
    v = _get(lclient, "/api/v3/violations?limit=5")
    _has(v, ["violations", "counts", "n", "types"])
    assert set(v["types"]) == set(PV.VTYPES)


def test_live_unknown_keys_404_and_bad_filters_422(lclient):
    _get(lclient, "/api/v3/systems/nope/view", 404)
    _get(lclient, "/api/v3/systems/nope/precision", 404)
    _get(lclient, "/api/v3/systems/nope/lattice", 404)
    _get(lclient, "/api/v3/systems/nope/strategy", 404)
    _get(lclient, "/api/v3/groups/nope/view", 404)
    _get(lclient, "/api/v3/groups/nope/facets", 404)
    _get(lclient, "/api/v3/systems/oa/entities/not-an-ip/view", 404)
    _get(lclient, "/api/v3/systems/oa/entities/203.0.113.250/view", 404)
    _get(lclient, "/api/v3/violations?type=bogus", 422)
    _get(lclient, "/api/v3/violations?severity=bogus", 422)
    _get(lclient, "/api/v3/violations?system=nope", 404)


def test_live_group_name_is_persisted_into_config(lclient, live):
    gid = sorted((MP.who_groups(live.store).get("groups") or {}))[0]
    resp = lclient.post(f"/api/v3/groups/{gid}/name", json={"name": "综合部"})
    assert resp.status_code == 200, resp.text
    d = resp.json()
    assert d["ok"] and d["pending"] and d["name"] == "综合部"
    for cfg in (live.config, live.pipeline.config):
        assert any(x.get("name") == "综合部" for x in cfg.get("who_group_names") or [])
    # renaming the same group replaces the entry instead of piling up
    lclient.post(f"/api/v3/groups/{gid}/name", json={"name": "综合部 OA"})
    names = [x["name"] for x in live.pipeline.config["who_group_names"]]
    assert "综合部 OA" in names and "综合部" not in names
    assert lclient.post("/api/v3/groups/nope/name", json={"name": "x"}).status_code == 404
    assert lclient.post(f"/api/v3/groups/{gid}/name", json={"name": ""}).status_code == 422


def test_v1_v2_routes_still_served(lclient):
    assert lclient.get("/api/health").status_code == 200
    assert lclient.get("/api/systems").status_code == 200
    assert lclient.get("/api/incidents").status_code == 200
    assert lclient.get("/app/js/progressive.js").status_code == 200


# ==================================================================== golden
TZ = _dt.timezone(_dt.timedelta(hours=8))
T0 = _dt.datetime(2026, 9, 1, tzinfo=TZ).timestamp()        # Tuesday 00:00 local
DAY = 86400.0
GA = ["192.168.1.21", "192.168.1.23", "10.168.7.121"]
USERS = {"192.168.1.21": "jack", "192.168.1.23": "rose", "10.168.7.121": "mike"}
LOGIN = "POST oa.corp /login"
HOME = "GET oa.corp /home"
APPROVE = "POST fin.corp /fin/approval/{num}/approve"
CFG = default_config({"progressive": {"enabled": True}, "tz": "Asia/Shanghai"})


def _keys(ip: str, g: str) -> List[Any]:
    p = ip.split(".")
    return [ip, ".".join(p[:3]) + ".0/24", ".".join(p[:2]) + ".0.0/16", f"grp:{g}", "reg:∅"]


def _tree(key: str, route: str, ips: List[str], g: str, days: int, dst: str) -> PT.PTreeModel:
    """The P14 golden fixture: a root split by route, one action node fed
    with one event per IP and workday."""
    m = PT.PTreeModel(key)
    tr = m.tree(EV.KIND_TXN, T0, create=True)
    sp = tr.split(tr.root, "http.route", 0, [[route]], T0)
    node, root = tr.nodes[sp.children[0]], tr.nodes[tr.root]
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


def _ctx(st: MetricStore, now: float) -> Context:
    return Context(store=st, now=now, window_s=3600.0, training=False, config=CFG)


@pytest.fixture(scope="module")
def golden():
    st = MetricStore()
    oa = _tree("oa", LOGIN, GA, "G1", 21, "192.168.100.100:8080")
    st.put_model("oa", SYSTEM_ENTITY, MP.PTREE, oa, version=1)
    nid = oa.kinds[0].nodes[oa.kinds[0].root].split.children[0]
    for name, obj in _fitted(nid).items():
        st.put_model("oa", SYSTEM_ENTITY, name, obj)
    fin = _tree("finance", APPROVE, ["192.168.2.10"], "G2", 21, "192.168.100.110:8443")
    st.put_model("finance", SYSTEM_ENTITY, MP.PTREE, fin, version=1)
    st.put_model(ORG, ORG, MP.WHO_GROUPS, {
        "version": 3,
        "groups": {"G1": {"id": "G1", "name": "综合部", "name_source": "config", "members": list(GA),
                          "covers": [], "systems": {"oa": 1.0}, "n": 3},
                   "G2": {"id": "G2", "name": "财务部", "name_source": "config", "members": ["192.168.2.10"],
                          "covers": [], "systems": {"finance": 1.0}, "n": 1}},
        "ip2g": dict({ip: "G1" for ip in GA}, **{"192.168.2.10": "G2"}),
        "mode": {"oa": {"mode": "ip"}, "finance": {"mode": "ip"}}})
    end = T0 + 21 * DAY
    for s in ("oa", "finance"):
        st.add_batch(s, EV.EVT_BATCH, end, EV.BatchBuilder(s).build(T0, end))
    now = oa.t_last + 3600.0
    # P14 materialises the views, P13 the facet trees (the real engines)
    P14.ViewsEngine().safe_run(_ctx(st, now), None)
    P13.FacetsEngine().safe_run(_ctx(st, now), None)
    pid = PN.pattern_id("oa", 0, nid, 1, 0)
    # P04 lifecycle events and one P03 violation of each kind of interest
    for kind, ts, desc, extra in (
            ("pattern_confirmed", T0 + 3 * DAY, "模式已确认：POST /login", {"n_c": 30.0, "days": 3}),
            ("pattern_drift", T0 + 15 * DAY, "模式 POST /login 的 body.len 发生了已确认的合法变化",
             {"attr": "body.len", "dir": "up", "days": 3, "ips": GA})):
        st.add_event(BehaviorEvent(system="oa", entity=SYSTEM_ENTITY, ts=ts, kind=kind, score=0.0,
                                   severity=Severity.INFO, description=desc,
                                   extra=dict(extra, pattern_id=pid, node=nid, tree_key="oa", event_kind=0,
                                              context="http.route=" + LOGIN, state="confirmed")))
    st.add_event(BehaviorEvent(
        system="oa", entity="192.168.1.21", ts=now - 600.0, kind="pattern_violation", score=0.6,
        severity=Severity.MEDIUM, description="192.168.1.21（综合部）提交 username=rose",
        axes=["credential"], p_value=1e-6,
        extra={"pattern_id": pid, "statement_zh": "192.168.1.21（综合部）在 oa 执行 POST /login：username=rose 绑定于 192.168.1.23",
               "statement_en": "192.168.1.21 submitted username=rose, bound to 192.168.1.23",
               "type": "content", "flags": ["cross_binding"], "observed": "rose", "expected": "jack",
               "p": 1e-6, "p_day": 1e-5, "U": None, "n_c": 60.0, "route": LOGIN, "node": nid,
               "tree_key": "oa", "event_ts": now - 600.0}))
    st.add_event(BehaviorEvent(
        system="finance", entity="192.168.1.21", ts=now - 300.0, kind="pattern_violation", score=0.9,
        severity=Severity.HIGH, description="综合部 IP 在财务系统审批",
        extra={"pattern_id": PN.pattern_id("finance", 0, 1, 1, 0), "type": "who",
               "statement_zh": "192.168.1.21（综合部） 在 finance 执行 POST /fin/approval/{num}/approve：该模式的来源封闭于 192.168.2.10",
               "statement_en": "192.168.1.21 performed the approval; sources closed on 192.168.2.10",
               "flags": ["outsider_group", "system_new"], "observed": "192.168.1.21",
               "expected": ["192.168.2.10"], "p": 1e-4, "p_day": 1e-4, "U": 0.01, "n_c": 20.0,
               "route": APPROVE, "node": 1, "tree_key": "finance", "event_ts": now - 300.0}))
    r = types.SimpleNamespace(
        store=st, config=CFG, gen=types.SimpleNamespace(vt=now), _lock=threading.Lock(), warmed=True,
        live_ticks=0, window_s=60, registry_mode="progressive_decision", pack_name="golden",
        pipeline=types.SimpleNamespace(config=CFG, tick_count=0, window_s=60, last_tick_stats={},
                                       engine_info=lambda: [{"name": "behavior.pattern_tree"},
                                                            {"name": "behavior.views"}]))
    r.nid = nid
    r.pid = pid
    return r


@pytest.fixture()
def gclient(golden):
    prev = routes.RUNTIME
    routes.RUNTIME = golden
    yield make_client(app)
    routes.RUNTIME = prev


def _login(view):
    return next(st for a in view["actions"] for st in a["statements"] if st["route"] == LOGIN)


def test_golden_system_view_blocks(gclient, golden):
    v = _get(gclient, "/api/v3/systems/oa/view")
    assert v["source"] == "model" and v["version"] == 1 and v["who_mode"] == "ip"
    assert v["header"]["address"] == "192.168.100.100:8080"
    a = v["actions"][0]
    assert a["route"] == LOGIN and a["write"] and a["share"] == pytest.approx(1.0)
    st = _login(v)
    assert st["text_zh"].startswith("【oa · 192.168.100.100:8080】工作日 09:00–09:21")
    assert "综合部（10.168.7.121、192.168.1.21、192.168.1.23）访问 POST /login" in st["text_zh"]
    assert st["text_en"] and st["state"] == "confirmed" and 0.9 < st["confidence"] <= 1.0
    assert st["who"]["level"] == "ip" and st["who"]["closed"] and sorted(st["who"]["members"]) == sorted(GA)
    assert st["when"]["workday"] == [[540, 561]]
    assert st["content"]["body.len"]["band90"] == [1024.0, 2048.0]
    assert st["content"]["body.len"]["range"] == [512.0, 3072.0]
    assert st["content"]["body.kv.username"]["grammar"] == "[a-z]{4}"
    assert st["bindings"]["body.kv.username"]["table"] == USERS
    assert st["workflow"][0]["from"] == LOGIN and st["workflow"][0]["to"] == HOME
    assert st["node"] == golden.nid and st["pattern_id"] == golden.pid
    assert {"functional", "temporal", "content", "content.bindings", "sequential"} <= set(st["facets"])
    fresh = _get(gclient, "/api/v3/systems/oa/view?fresh=true&flat=true")
    assert fresh["source"] == "render" and _login(fresh)["text_zh"] == st["text_zh"]
    assert [x["id"] for x in fresh["statements"]] == [x["id"] for a in fresh["actions"] for x in a["statements"]]


def test_golden_group_view_negative_statement_and_systems(gclient):
    gl = _get(gclient, "/api/v3/groups")
    assert {g["id"]: g["name"] for g in gl["groups"]} == {"G1": "综合部", "G2": "财务部"}
    gv = _get(gclient, "/api/v3/groups/G1/view")
    assert gv["name"] == "综合部" and sorted(gv["members"]) == sorted(GA)
    assert [s["system"] for s in gv["systems"]] == ["oa"]
    acts = gv["systems"][0]["actions"]
    assert acts[0]["route"] == LOGIN and acts[0]["statements"]
    assert all(set(x["who"]["items"]) <= set(GA) for a in acts for x in a["statements"])
    neg = gv["negative"]
    assert len(neg) == 1 and neg[0]["negative"] and neg[0]["system"] == "finance"
    assert neg[0]["text_zh"].startswith("综合部 在 finance 中从未执行写操作")
    assert neg[0]["routes"] == ["POST /fin/approval/{num}/approve"]
    # finance's own group: no negative statement about finance
    fv = _get(gclient, "/api/v3/groups/G2/view")
    assert not any(x["system"] == "finance" for x in fv["negative"])
    f = _get(gclient, "/api/v3/groups/G1/facets")
    assert f["subject"]["group"] == "G1" and f["n_items"] >= 1


def test_golden_ip_view_inherits_group_and_binding(gclient):
    iv = _get(gclient, "/api/v3/systems/oa/entities/192.168.1.21/view")
    assert iv["group"]["id"] == "G1" and iv["group"]["name"] == "综合部"
    assert iv["prefix24"] == "192.168.1.0/24"
    assert [b["value"] for b in iv["bindings"]] == ["jack"]
    assert iv["bindings"][0]["pair"] == "net.src->body.kv.username" and iv["bindings"][0]["LB"] == 0.95
    assert any(s["route"] == LOGIN for s in iv["statements"])               # the system statement names it
    assert any(s["route"] == LOGIN for s in iv["inherited_here"])           # the group's own statement
    assert any(s["negative"] for s in iv["inherited"])                      # incl. "never writes in finance"
    assert iv["exceptions"] == []
    assert [v["type"] for v in iv["violations"]] == ["content"]            # only oa's violation here
    # the finance approver: no group statement in oa, but known through its group
    fin = _get(gclient, "/api/v3/systems/finance/entities/192.168.2.10/view")
    assert fin["group"]["id"] == "G2" and fin["bindings"] == []
    f = _get(gclient, "/api/v3/systems/oa/entities/192.168.1.21/facets")
    assert f["subject"]["entity"] == "192.168.1.21"


def test_golden_pattern_detail_lifecycle_and_violations(gclient, golden):
    d = _get(gclient, f"/api/v3/patterns/{golden.pid}")
    assert d["alive"] and not d["stale_id"] and d["systems"] == ["oa"]
    assert [p["id"] for p in d["path"]] == [0, golden.nid]
    assert d["brief"]["route"] == LOGIN or d["brief"]["context"].startswith("http.route=")
    assert d["parent"]["id"] == 0 and d["children"] == []
    assert d["statement"]["pattern_id"] == golden.pid
    assert d["constraints"]["bounds"]["attrs"]["body.len"]["band90"] == [1031.0, 2041.0]
    assert d["constraints"]["bindings"]["pairs"]["net.src->body.kv.username"]["table"]["192.168.1.21"]["top"] == "jack"
    assert d["constraints"]["windows"]["when"]["workday"] == [[540, 561]]
    assert d["constraints"]["workflow"][0]["to"] == HOME
    kinds = [e["kind"] for e in d["lifecycle"]]
    assert "pattern_confirmed" in kinds and "pattern_drift" in kinds        # drift history
    drift = next(e for e in d["lifecycle"] if e["kind"] == "pattern_drift")
    assert drift["detail"]["attr"] == "body.len"
    assert [v["type"] for v in d["violations"]] == ["content"]
    assert any(x["op"] == "split" for x in d["lineage"])
    assert d["who"]["evidence"]["level"] == "ip"
    assert sum(d["when_hist96"]["workday"]) > 0 and len(d["when_hist96"]["nonworkday"]) == 96
    # an older version of the id still resolves (flagged), the group-part suffix is ignored
    old = _get(gclient, f"/api/v3/patterns/p:oa:0:{golden.nid}@0.0|grp:G1")
    assert old["stale_id"] and old["node"] == golden.nid
    root = _get(gclient, "/api/v3/patterns/p:oa:0:0")
    assert [c["id"] for c in root["children"]] and root["parent"] is None


def test_golden_violations_typed_reasons_and_filters(gclient):
    v = _get(gclient, "/api/v3/violations")
    assert v["n"] == 2 and v["counts"]["type"] == {"content": 1, "who": 1}
    who = next(x for x in v["violations"] if x["type"] == "who")
    assert who["type_zh"] == "来源越界" and who["severity"] == "high"
    assert {f["flag"] for f in who["flags"]} == {"outsider_group", "system_new"}
    assert next(f for f in who["flags"] if f["flag"] == "outsider_group")["zh"] == "来源属于其他行为群组"
    assert who["reason_zh"].startswith("192.168.1.21（综合部） 在 finance") and who["expected"] == ["192.168.2.10"]
    assert who["route_text"] == "POST /fin/approval/{num}/approve"
    c = _get(gclient, "/api/v3/violations?type=content")
    assert [x["type"] for x in c["violations"]] == ["content"] and c["violations"][0]["flags"][0]["flag"] == "cross_binding"
    hi = _get(gclient, "/api/v3/violations?severity=high")
    assert [x["system"] for x in hi["violations"]] == ["finance"]
    fs = _get(gclient, "/api/v3/violations?system=oa&entity=192.168.1.21")
    assert fs["n"] == 1
    assert _get(gclient, f"/api/v3/violations?since={2e9}")["n"] == 0


def test_golden_precision_curve_lattice_facets_and_empty_models(gclient, golden):
    p = _get(gclient, "/api/v3/systems/oa/precision?days=400")
    by = {d["date"]: d for d in p["days"]}
    d_conf = PV.local_date(T0 + 3 * DAY, 8 * 3600.0)
    assert by[d_conf]["confirmed_new"] == 1 and by[d_conf]["mean_depth"] == 1.0
    assert by[PV.local_date(T0 + 15 * DAY, 8 * 3600.0)]["drift"] == 1
    assert p["days"][-1]["confirmed"] == 1 and sum(d["violations"] for d in p["days"]) == 1
    assert p["now"]["confidence"]["n"] == 1
    lat = _get(gclient, "/api/v3/systems/oa/lattice")
    assert lat["n_nodes"] == len(lat["nodes"]) == 3
    login = next(n for n in lat["nodes"] if n["id"] == golden.nid)
    assert login["route"] == LOGIN and login["distinct_ips"] == 3 and login["state"] == "confirmed"
    assert next(n for n in lat["nodes"] if n["id"] == 0)["split"]["attr"] == "http.route"
    f = _get(gclient, "/api/v3/systems/oa/facets")
    assert f["source"] == "model" and f["n_items"] >= 1
    assert _get(gclient, "/api/v3/systems/oa/facets?fresh=true")["source"] == "render"
    # models the golden store does not have: empty payloads, never a 500
    s = _get(gclient, "/api/v3/systems/oa/strategy")
    assert s["chosen"] == {} and s["engines"] == []
    a = _get(gclient, "/api/v3/systems/oa/attributes")
    assert a["n"] == 0
    b = _get(gclient, "/api/v3/budget")
    assert b["present"] is False and b["ptree_bytes"]["oa"] > 0
    st = _get(gclient, "/api/v3/status")
    assert st["n_statements"] >= 2 and st["n_groups"] == 2


def test_measured_curve_reads_evaluation_runs(gclient, tmp_path, monkeypatch):
    run = {"seed": 3, "registry": "progressive_decision", "score": {"pack": "O", "seed": 3, "pg1": {
        "7": {"day": 7, "recall": 0.5, "precision": 0.6, "ece": 0.2, "mean_depth": 1.5, "n_confirmed": 4,
              "n_truth": 4, "per_pattern": {"GA.oa.login#0": {"recovered": True},
                                            "GA.oa.approvals#0": {"recovered": False},
                                            "FIN.finance.approve#0": {"recovered": True}}},
        "14": {"day": 14, "recall": 0.75, "precision": 0.7, "ece": 0.1, "mean_depth": 2.0,
               "n_confirmed": 6, "n_truth": 4, "per_pattern": {"GA.oa.login#0": {"recovered": True},
                                                               "GA.oa.approvals#0": {"recovered": True}}}}}}
    (tmp_path / "O_3.json").write_text(json.dumps(run))
    monkeypatch.setenv("APPMON_PROGRESSIVE_RUNS", str(tmp_path))
    m = _get(gclient, "/api/v3/systems/oa/precision")["measured"]
    assert m["available"] and m["runs"][0]["seed"] == 3
    assert [(d["day"], d["system_recall"], d["precision"]) for d in m["runs"][0]["days"]] == \
        [(7, 0.5, 0.6), (14, 1.0, 0.7)]
    assert _get(gclient, "/api/v3/systems/finance/precision")["measured"]["runs"][0]["days"][0]["system_recall"] == 1.0
    monkeypatch.setenv("APPMON_PROGRESSIVE_RUNS", str(tmp_path / "none"))
    assert _get(gclient, "/api/v3/systems/oa/precision")["measured"] == {"available": False, "runs": []}


def test_status_without_the_progressive_core(gclient):
    r = types.SimpleNamespace(store=MetricStore(), config=default_config({}), gen=types.SimpleNamespace(vt=T0),
                              _lock=threading.Lock(),
                              pipeline=types.SimpleNamespace(config={}, engine_info=lambda: [{"name": "behavior.risk"}]))
    prev = routes.RUNTIME
    routes.RUNTIME = r
    try:
        d = _get(gclient, "/api/v3/status")
        assert not d["running"] and not d["present"] and d["registry_mode"] == "full"
        assert "APPMON_PROGRESSIVE" in d["hint_zh"] and d["systems"] == []
        assert _get(gclient, "/api/v3/groups")["groups"] == []
        assert _get(gclient, "/api/v3/violations")["violations"] == []
        assert _get(gclient, "/api/v3/budget")["present"] is False
    finally:
        routes.RUNTIME = prev
