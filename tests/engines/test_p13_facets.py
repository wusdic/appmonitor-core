"""P13 Facets (progressive.md §6.17.1; card P13) and lib/pfacets."""
from __future__ import annotations

import time

import pytest

from helpers import ctx, make_store

from app.engines.behavior.facets import FacetsEngine, compose_ip
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import pfacets as PFc
from app.models.schema import ORG, SYSTEM_ENTITY

T0 = 1_790_000_000.0
CFG = {"progressive": {"enabled": True}, "tz": "Asia/Shanghai"}


def batch(st, s, t1):
    b = EV.BatchBuilder(s, EV.KIND_TXN)
    b.add(t1 - 1.0, "10.0.0.1", {"net.src": "10.0.0.1", "ev.ch": "http"}, 1.0)
    st.add_batch(s, EV.EVT_BATCH, t1, b.build(t1 - 60.0, t1))


def sysprof(body=0.3, tls=0.0, sess=0.9, automation=0.0):
    return {"characteristics": {"payload_vis": {"http": 1.0 - tls, "body": body, "tls_opaque": tls},
                                "sess_ident": sess, "automation": automation, "numeric_targets": True,
                                "volume": {"events_day": 1000, "learned_share": 1.0}},
            "chosen": {"who": "ip", "P07": "on" if body else "off", "P08": "on" if body else "off",
                       "P10": "on" if sess >= 0.6 else "off"}, "hints": []}


def view(tags):
    return {"statements": [{"id": f"st{i}", "text_zh": f"陈述{i}", "text_en": f"statement {i}",
                            "confidence": 0.9, "support": 30.0, "facets": t} for i, t in enumerate(tags)]}


def tree_of(st, s, e=SYSTEM_ENTITY):
    return st.profile(s, e).extra["facets"]


def test_facet_tree_from_statements_and_applicability():
    st = make_store()
    for s, sp in (("oa", sysprof()), ("mail", sysprof(body=0.0, tls=1.0, sess=0.1))):
        batch(st, s, T0)
        st.put_model(s, SYSTEM_ENTITY, MP.SYSPROF, sp)
        st.put_model(s, SYSTEM_ENTITY, MP.PVIEWS, view([["functional", "spatial", "content.bindings"],
                                                        ["temporal", "sequential"]]))
    FacetsEngine().safe_run(ctx(st, T0, config=CFG), None)
    oa = PFc.facet_ids(tree_of(st, "oa"))
    mail = PFc.facet_ids(tree_of(st, "mail"))
    assert {"functional", "spatial", "content", "content.bindings", "temporal", "sequential",
            "risk", "risk.strategy"} <= set(oa)
    # opaque TLS mail: no binding facet, no sequential facet (sessions not identifiable)
    assert "content.bindings" not in mail and "sequential" not in mail
    assert "functional" in mail and "temporal" in mail
    top = [n["id"] for n in tree_of(st, "oa")["facets"]]
    assert top == sorted(top, key=lambda x: [d["order"] for d in PFc.DEFAULT_FACETS if d["id"] == x][0])
    # the registry model exists at org level with the defaults (e.g. the automation facet)
    reg = st.get_model(ORG, ORG, PFc.FACETS)
    assert "temporal.automation" in reg["decls"]["behavior.facets"]


def test_runtime_declared_facet_appears_in_the_next_composition():
    st = make_store()
    batch(st, "oa", T0)
    st.put_model("oa", SYSTEM_ENTITY, MP.SYSPROF, sysprof())
    eng = FacetsEngine(facet_period_s=3600.0)
    eng.safe_run(ctx(st, T0, config=CFG), None)
    assert "social.colleagues" not in PFc.facet_ids(tree_of(st, "oa"))
    # a (test) engine declares a new facet with its own source model; no code change in P13
    PFc.declare(st, "behavior.test_engine", {
        "id": "social.colleagues", "parent": "relational", "name_zh": "常交往人群", "name_en": "Usual peers",
        "subjects": ["system"], "producer": "behavior.test_engine", "sources": ["model.colleagues"],
        "applicable": "always", "render": "colleagues_v9", "order": 65})
    st.put_model("oa", SYSTEM_ENTITY, "model.colleagues",
                 {"version": 3, "items": [{"text_zh": "常与财务部共访", "text_en": "co-accesses with finance"}]})
    batch(st, "oa", T0 + 3600.0)
    eng.safe_run(ctx(st, T0 + 3600.0, config=CFG), None)
    tree = tree_of(st, "oa")
    assert "social.colleagues" in PFc.facet_ids(tree)
    rel = next(n for n in tree["facets"] if n["id"] == "relational")
    col = next(c for c in rel["children"] if c["id"] == "social.colleagues")
    assert col["items"][0]["text_zh"] == "常与财务部共访"
    # a new version was recorded because the tree changed; an unchanged tree adds none
    n_ver = len(st.profile_versions("oa", SYSTEM_ENTITY))
    batch(st, "oa", T0 + 7200.0)
    eng.safe_run(ctx(st, T0 + 7200.0, config=CFG), None)
    assert len(st.profile_versions("oa", SYSTEM_ENTITY)) == n_ver
    with pytest.raises(ValueError):
        PFc.declare(st, "x", {"id": "bad"})


def test_ip_portraits_only_for_earned_ips_and_on_read_otherwise():
    st = make_store()
    batch(st, "oa", T0)
    st.put_model("oa", SYSTEM_ENTITY, MP.SYSPROF, sysprof())
    ips = [f"10.1.{i // 256}.{i % 256}" for i in range(5000)]
    st.put_model(ORG, ORG, MP.WHO_GROUPS, {"groups": {"g1": {"name": "综合部", "members": ips[:3]}},
                                           "ip2g": {ip: "g1" for ip in ips[:3]}})
    st.put_model(ORG, ORG, MP.BUDGET, {"systems": {"oa": {"earned": ips[:2], "active": ips[:10]}}})
    st.put_model(ORG, "class:grp:g1", MP.PVIEWS, view([["relational"], ["functional"]]))
    t0 = time.perf_counter()
    FacetsEngine().safe_run(ctx(st, T0, config=CFG), None)
    el = time.perf_counter() - t0
    assert st.profile("oa", ips[0]) is not None and st.profile("oa", ips[1]) is not None
    assert st.profile("oa", ips[2]) is None                    # unearned: composed on read
    on_read = compose_ip(st, "oa", ips[2], T0)
    assert "relational" in PFc.facet_ids(on_read)
    g = st.profile(ORG, "class:grp:g1").extra["facets"]
    assert "relational.group" in PFc.facet_ids(g)
    assert el < 1.0                                             # O(earned + groups), not O(5000 IPs)


def test_step6_refresh_on_read_skips_composition():
    st = make_store()
    batch(st, "oa", T0)
    st.put_model("oa", SYSTEM_ENTITY, MP.SYSPROF, sysprof())
    st.put_model(ORG, ORG, MP.BUDGET, {"ladder": {"refresh_on_read": True}})
    assert FacetsEngine().safe_run(ctx(st, T0, config=CFG), None) == 0
    assert st.profile("oa", SYSTEM_ENTITY) is None


def test_applicability_predicates():
    assert PFc.applicable("payload_visible", sysprof(body=0.2))
    assert not PFc.applicable("payload_visible", sysprof(body=0.0, tls=1.0))
    assert PFc.applicable("automation", sysprof(automation=0.2))
    assert not PFc.applicable("automation", sysprof())
    assert PFc.applicable("unknown_predicate", sysprof())
    assert PFc.applicable("always&!automation", sysprof())
    assert PFc.applicable("payload_visible", None)             # no profile yet: show
