"""P03 round 6: the same-batch session overlay marks a content variant's
bare route like P10's session pass (temporal owner OI-11)."""
from __future__ import annotations

from helpers import ctx

import test_p03_conformity as T3

from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.models.schema import SYSTEM_ENTITY

FORM, GEN = "GET oa /report/form", "POST oa /report/generate"


def _fixture():
    from app.engines.behavior.workflow import WorkflowEngine
    fx = T3.Fx()
    for d in T3.workdays(20):
        for ip in T3.GA[1:]:
            fx.learn("oa", GEN, ip, d + 17 * 3600 + 120)
    fx.tree("oa", [GEN, FORM])
    p10 = WorkflowEngine(mine_period_s=6 * 3600.0)
    days = T3.workdays(21)
    for d in days[:20]:
        for h in range(24):
            t1 = d + (h + 1) * 3600.0
            if h == 17:
                b = EV.BatchBuilder("oa", EV.KIND_TXN)
                for ip in T3.GA[1:]:
                    b.add(d + 17 * 3600 + 10, ip, {"http.route": FORM, "net.src": ip})
                    b.add(d + 17 * 3600 + 120, ip, {"http.route": GEN, "net.src": ip})
                fx.st.add_batch("oa", EV.EVT_BATCH, t1, b.build(t1 - 3600, t1))
            fx.st.ensure_retention("evt.", max_age_s=6 * 3600.0)
            p10.safe_run(ctx(fx.st, t1, window_s=3600.0, config=T3.CFG), None)
    flow = fx.st.get_model("oa", SYSTEM_ENTITY, MP.PFLOW)
    assert any(r["to"] == GEN and r["from"] == FORM for r in flow["scopes"]["*"]["requires"])
    return fx, days


def test_a_young_variant_of_the_predecessor_in_the_same_batch_counts():
    """P04 splits the form page's node on a content attribute after P10 mined
    'generate requires form': the form's events now carry the variant key
    'GET oa /report/form#vN', which P10's dictionary does not hold yet. A user
    opening the form and generating the report in the same tick was scored
    'without the required predecessor' (the overlay set only the variant's
    Bloom bits); P10 itself marks the bare route too."""
    fx, days = _fixture()
    tr = fx.tree("oa", [GEN, FORM])
    form_nd = fx.node("oa", FORM)
    tr.split(form_nd.id, "body.kv.tpl", 0, [["monthly"]], T3.T0)
    fx.st.put_model("oa", SYSTEM_ENTITY, MP.PTREE, MP.get_ptree(fx.st, "oa"), version=2)
    t = days[20] + 17 * 3600 + 120
    b, asg = fx.score("oa", [(t - 100, "192.168.1.23", {"http.route": FORM, "body.kv.tpl": "monthly"}),
                             (t, "192.168.1.23", {"http.route": GEN, "http.method": "POST"})])
    acts = [asg.get("act", i) for i in range(b.n)]
    assert any(a.startswith(FORM + "#v") for a in acts)          # the predecessor is a variant
    v = [e for e in fx.violations("192.168.1.23") if e.extra["type"] == "seq"]
    assert not v, [e.description for e in v]
    # without the form page the requirement still holds
    b2, _ = fx.score("oa", [(t + 7200, "192.168.1.21", {"http.route": GEN, "http.method": "POST"})])
    assert [e for e in fx.violations("192.168.1.21") if e.extra["type"] == "seq"]
