"""Evaluator round 5 (EV5-6): a system that joined a family (P12) is rendered
under the family's tree key; the system view it had under its own key must not
stay in the store frozen at the join. O-real seed 0: oa joined fam:1 on day
11, its own model.pviews kept 42 statements unchanged and 'confirmed' to day
35 (D3's renamed approval pages never went stale). Fails on the round-5 code
before the fix (no _retire_members; the frozen view stays)."""
from __future__ import annotations

from helpers import make_store

from app.engines.behavior import views as VW
from app.engines.behavior.lib import m_ptree as MP
from app.models.schema import ORG, SYSTEM_ENTITY

T = 1_758_000_000.0


def _frozen_view(key: str) -> dict:
    st = {"id": f"p:{key}:0:9@1.0", "pattern_id": f"p:{key}:0:9@1.0", "state": "confirmed",
          "text_zh": "【oa】… POST /approval/{num}/approve（审批）", "text_en": "…",
          "evidence": {"system": key, "route": "POST /approval/{num}/approve"}}
    return {"fmt": 1, "view": "system", "subject": key, "updated": T, "statements": [st]}


def test_a_family_members_own_system_view_is_retired():
    store = make_store()
    eng = VW.ViewsEngine()
    for s in ("oa", "oa-2", "crm"):
        store.put_model(s, SYSTEM_ENTITY, MP.PVIEWS, _frozen_view(s), version=1, ts=T)
    store.put_model(ORG, ORG, MP.SYSFAM, {"version": 1, "member": {"oa": "fam:1", "oa-2": "fam:1"}},
                    version=1, ts=T)
    assert eng._retire_members(store, T + 3600) == 2
    for s in ("oa", "oa-2"):
        v = store.get_model(s, SYSTEM_ENTITY, MP.PVIEWS)
        assert v["retired"] is True and v["family"] == "fam:1" and v["statements"] == []
    # a system outside any family keeps its view; a second pass writes nothing
    assert store.get_model("crm", SYSTEM_ENTITY, MP.PVIEWS)["statements"]
    assert eng._retire_members(store, T + 7200) == 0


def test_the_engine_run_retires_the_frozen_view():
    from helpers import ctx
    store = make_store()
    store.put_model("oa", SYSTEM_ENTITY, MP.PVIEWS, _frozen_view("oa"), version=1, ts=T)
    store.put_model(ORG, ORG, MP.SYSFAM, {"version": 1, "member": {"oa": "fam:1"}}, version=1, ts=T)
    VW.ViewsEngine().run(ctx(store, T + 3600, config={"progressive": {"enabled": True}}))
    v = store.get_model("oa", SYSTEM_ENTITY, MP.PVIEWS)
    assert v.get("retired") is True and v["statements"] == []
