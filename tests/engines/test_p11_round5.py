"""P11 who groups, round 5 (diagnosed on pack O, round-4 code): regression tests
that fail on the round-4 code."""
from __future__ import annotations

import numpy as np

from helpers import ctx, make_store

from app.engines.behavior import who_groups as WG
from app.engines.behavior.lib import m_ptree as MP
from app.models.schema import SYSTEM_ENTITY

from test_p11_who_groups import DAY, T0, dept_ips

OLD = {"oa|GET oa /approval/list": 15, "oa|POST oa /approval/{num}/approve": 15}
NEW = {"oa|GET oa /flow/list": 15, "oa|POST oa /flow/{num}/approve": 15}


def _signatures(st, days, appr_prog, rng):
    base = {"appr": {"oa|POST oa /login": 5, "oa|GET oa /docs": 10},
            "rep": {"oa|POST oa /login": 5, "oa|GET oa /docs": 10, "oa|GET oa /report/form": 12,
                    "oa|POST oa /report/generate": 12, "mail|TLS mail": 8},
            "fin": {"finance|POST fin /fin/login": 5, "finance|GET fin /fin/ledger": 20,
                    "finance|POST fin /fin/voucher": 10, "mail|TLS mail": 8}}
    who = {"192.168.1.21": "appr", "192.168.1.23": "rep", "10.168.7.121": "rep",
           "192.168.2.10": "fin", "192.168.2.11": "fin", "192.168.2.12": "fin"}
    for d in days:
        t = T0 + d * DAY + 36000
        for src, p in who.items():
            prog = dict(base[p], **(appr_prog if p == "appr" else {}))
            for it, m in prog.items():
                st.sigs.add(src, it, t, m * (1 + 0.1 * rng.standard_normal()), 3.0, d)


def test_a_groups_name_follows_a_confirmed_rename_of_its_pages():
    """Pack O round 4: after D3 (/approval/ -> /flow/ on day 14, confirmed by
    P10) the approver's group kept the display name '综合部·oa GET
    /approval/list' to day 21 (sticky labels over slowly decaying signatures)."""
    rng = np.random.default_rng(0)
    st, eng, store = WG.WGState(), WG.WhoGroupsEngine(), make_store()
    cfg = {"progressive": {"enabled": True},
           "who_group_names": [{"name": "综合部", "ips": dept_ips()["GA"]},
                               {"name": "财务部", "ips": dept_ips()["FIN"]}]}
    _signatures(st, range(10), OLD, rng)
    model = WG.empty_model()
    now = T0 + 10 * DAY
    eng._cluster(ctx(store, now, config=cfg), st, model, now)
    g = model["ip2g"]["192.168.1.21"]
    assert "/approval/" in model["groups"][g]["name"]
    # the pages are renamed; P10 confirmed it; three days of the new pages
    store.put_model("oa", SYSTEM_ENTITY, MP.PFLOW, {"renamed": {
        "GET oa /flow/list": {"from": "GET oa /approval/list", "day": 10},
        "POST oa /flow/{num}/approve": {"from": "POST oa /approval/{num}/approve", "day": 10}}})
    _signatures(st, range(10, 13), NEW, rng)
    now = T0 + 13 * DAY
    eng._cluster(ctx(store, now, config=cfg), st, model, now)
    gr = model["groups"][model["ip2g"]["192.168.1.21"]]
    assert "/approval/" not in gr["name"] and "/flow/" in gr["name"]
    assert not any("/approval/" in x["item"] for x in gr["labels"])
