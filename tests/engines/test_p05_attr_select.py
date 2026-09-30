"""P05 AttributeSelection (behavior.attr_select): roles, kept sets, per-node
targets, 'IP is not a feature' (docs/lib3/progressive.md §6.4, card P05)."""
from __future__ import annotations

import numpy as np

from ptree_sim import DAY, MON, Sim, daily, is_workday

from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.models.schema import SYSTEM_ENTITY

GA = {"192.168.1.21": "jack", "192.168.1.23": "rose", "10.168.7.121": "mike"}


def _org(d, t0, rng):
    """An OA-like system: a health monitor every 5 min, page views of 12 IPs,
    and the 3 GA logins with key=value bodies."""
    ev = []
    for k in range(288):
        ev.append((t0 + k * 300 + 7, "oa", "192.168.9.9",
                   {"http.route": "GET oa /health", "net.bytes_down": float(rng.uniform(180, 220)),
                    "hdr.user-agent": "python-requests", "noise.a": f"n{rng.integers(0, 4)}"}))
    if not is_workday(t0):
        return ev
    for i in range(12):
        for _ in range(6):
            ev.append((t0 + rng.uniform(9, 17) * 3600, "oa", f"192.168.3.{20 + i}",
                       {"http.route": f"GET oa /docs/{rng.integers(0, 3)}",
                        "net.bytes_down": float(rng.uniform(5000, 20000)), "hdr.user-agent": "Mozilla",
                        "noise.a": f"n{rng.integers(0, 4)}", "const.a": "c"}))
    for ip, u in GA.items():
        ev.append((t0 + (9 * 60 + rng.uniform(0, 21)) * 60, "oa", ip,
                   {"http.route": "POST oa /login", "net.bytes_up": float(rng.uniform(1024, 2048)),
                    "body.kv.username": u, "body.keys": frozenset({"username", "password"}),
                    "hdr.user-agent": "Mozilla", "noise.a": f"n{rng.integers(0, 4)}", "const.a": "c"}))
    return ev


def test_roles_targets_and_split_candidates_are_learned():
    sim = Sim(p05=True)
    sim.add(daily(_org, 3, seed=1))
    sim.run_until(MON + 3 * DAY)
    sel = sim.st.get_model("oa", SYSTEM_ENTITY, MP.ATTRSEL)
    assert sel and sel["runs"] >= 24                                    # hourly, not every tick
    roles = sel["roles"]
    assert roles["noise.a"] in ("dropped", "shape")
    # const.a is constant where present and present exactly on the browser traffic:
    # its presence may split (absence is a value), its value is never a target
    assert roles["const.a"] != "target"
    assert "const.a" not in sel["targets_sys"][0]
    splits = {a for a, _ in sel["split_cands"][0]}
    assert "http.route" in splits
    tg = sel["targets_sys"][0]
    assert "noise.a" not in tg and "net.src" not in tg
    assert any(e.kind == "attribute_role" for e in sim.st.events())


def test_ip_not_a_feature_on_a_random_population():
    def portal(d, t0, rng):
        ev = []
        for _ in range(600):
            ip = f"10.{rng.integers(0, 250)}.{rng.integers(0, 250)}.{rng.integers(1, 250)}"
            r = int(rng.integers(0, 3))
            ev.append((t0 + rng.uniform(0, DAY), "portal", ip,
                       {"http.route": f"GET p /r{r}", "net.bytes_down": float(rng.uniform(1000, 2000) * (r + 1)),
                        "hdr.user-agent": ["a", "b", "c"][int(rng.integers(0, 3))]}))
        return ev
    sim = Sim(p05=True)
    sim.add(daily(portal, 2, seed=2))
    sim.run_until(MON + 2 * DAY)
    sel = sim.st.get_model("portal", SYSTEM_ENTITY, MP.ATTRSEL)
    assert sel["who_mode"] == "none", sel.get("ip_info")
    assert "net.src" not in {a for a, _ in sel["split_cands"][0]}


def test_node_targets_follow_the_node():
    """The login node's targets include its body attributes although logins are
    a tiny share of the system's mass (the stratified probe routed through the tree)."""
    sim = Sim(p05=True)
    sim.add(daily(_org, 8, seed=3))
    sim.run_until(MON + 8 * DAY)
    sel = sim.st.get_model("oa", SYSTEM_ENTITY, MP.ATTRSEL)
    tr = sim.tree()
    assert {"net.bytes_up", "body.kv.username"} & set(sel["targets_sys"][0]), sel["targets_sys"][0]
    ev = {"http.route": "POST oa /login", "net.src": "192.168.1.21", "hdr.user-agent": "Mozilla",
          "net.bytes_up": 1500.0, "body.kv.username": "jack", "ctx.tod_min": 545.0, "ctx.daytype": "workday"}
    hier = MP.hierarchies(sim.st, "oa", sim.cfg)
    leaf = tr.route(lambda a: ev.get(a, EV.ABSENT), hier)[-1]
    tg = sim.p04._targets_of(None, tr, tr.nodes[leaf], 0, sel)
    assert {"net.bytes_up", "body.kv.username"} & set(tg), (tr.nodes[leaf].ctx, tg)
