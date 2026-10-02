"""P03 round 3 (groups_views owner): hierarchical back-off of the when density.

Pack O, A4 (a 03:05 OA login of a 综合部 address): the HDR p of an empty slot
under a node's own density floors at ~alpha_t / (N + alpha_t), so a young
department node (N ~ 40) could not declare the slot rare while its route node
(N ~ 900) could; detection depended on which node covered the event."""
from __future__ import annotations

import datetime as _dt

import numpy as np

from app.engines.behavior import conformity as CF
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.lib import pscore as SC
from app.models.schema import SYSTEM_ENTITY, Severity

from test_p03_conformity import CFG, GA, LOGIN, TZ, Fx, local_minute, workdays

OTHERS = [f"192.168.3.{i}" for i in range(20, 32)]


def _learn_path(fx: Fx, key: str, ip: str, ts: float) -> None:
    """One learned login on every node of its path (P04 counts an event on
    every node from the root to its leaf)."""
    tr = fx.tree(key, [LOGIN])
    hier = MP.hierarchies(fx.st, key, CFG)
    keys = [hier.gen("net.src", l, ip) for l in range(PN.WHO_LEVELS)]
    day = int((ts + 8 * 3600) // 86400.0)
    wd = 0 if _dt.datetime.fromtimestamp(ts, TZ).weekday() < 5 else 1
    attrs = {"http.route": LOGIN, "net.src": ip}
    for nid in tr.route(lambda a: attrs.get(a), hier, (), ts):
        tr.nodes[nid].update_core(ts, 1.0, 1.0, keys, ip, wd, local_minute(ts), day)
    for nd in tr.nodes.values():
        if nd.state == "candidate" and nd.n_c(ts) >= 1:
            nd.state = "confirmed"


def _young_department_node():
    """The login route node learned 30 workdays of 15 sources' logins
    (08:30-10:00); its 综合部 child (a who split) is young: 15 workdays of
    3 logins a day at 08:30-08:51 - in the uniform-prior regime its empty
    slots have HDR p ~ 1e-3."""
    fx = Fx()
    rng = np.random.default_rng(7)
    days = workdays(48)
    for d in days[:30]:
        for ip in GA + OTHERS:
            _learn_path(fx, "oa", ip, d + (8 * 60 + 30 + 90 * rng.random()) * 60.0)
    tr = fx.tree("oa", [LOGIN])
    route_nd = fx.node("oa", LOGIN)
    tr.split(route_nd.id, "net.src", 0, [GA], days[30])
    child = tr.nodes[route_nd.split.children[0]]
    for d in days[30:45]:
        for ip in GA:
            _learn_path(fx, "oa", ip, d + (8 * 60 + 30 + 21 * rng.random()) * 60.0)
        for ip in OTHERS:
            _learn_path(fx, "oa", ip, d + (9 * 60 + 60 * rng.random()) * 60.0)
    for nd in tr.nodes.values():
        nd.state = "stable"
    fx.put("oa", MP.PWIN, child.id, attrs={})
    m = fx.st.get_model("oa", SYSTEM_ENTITY, MP.PWIN)
    m["nodes"][0][child.id]["when"] = {"workday": [[510, 531]], "nonworkday": []}
    return fx, tr, route_nd, child, days


def test_young_node_declares_an_empty_slot_rare_when_its_parents_agree():
    fx, tr, route_nd, child, days = _young_department_node()
    rng = np.random.default_rng(3)
    # the node's ordinary events are scored first (they also calibrate its p-values)
    for d in days[45:47]:
        fx.score("oa", [(d + (8 * 60 + 30 + 21 * rng.random()) * 60.0, ip, {"http.route": LOGIN}) for ip in GA])
    assert not [e for e in fx.violations() if e.extra["type"] == "when"]
    t = days[47] + 3 * 3600 + 5 * 60                              # 03:05 local
    N = child.when.evidence(0, t)
    assert 20 <= N <= 80 and route_nd.when.evidence(0, t) >= 300
    # the child's own density alone cannot go below ~alpha_t / (N + alpha_t)
    own = CF._hdr_table(SC.when_density(child.when.hist[0], N))
    slot = int(local_minute(t) // 15)
    assert own[slot] > 5e-4
    b, asg = fx.score("oa", [(t, "10.168.7.121", {"http.route": LOGIN})])
    assert int(asg.get("conf", 0)) == child.id                    # judged at the young node
    # ... the route node (N ~ 340) agrees: the young node states the route's resolution
    assert asg.get("p_when", 0) <= 3e-4 and asg.get("p_when", 0) < own[slot] / 3
    w = [e for e in fx.violations("10.168.7.121") if e.extra["type"] == "when"]
    assert w and w[0].severity == Severity.MEDIUM


def test_slot_used_by_the_parents_population_is_not_rare_at_the_young_node():
    """The prior works both ways: 09:30 is empty at the young 综合部 node but
    the route's other sources log in then, so the slot keeps a share of the
    parent's mass (the uniform floor called it as rare as 03:05)."""
    fx, tr, route_nd, child, days = _young_department_node()
    tc = CF._TreeCtx(fx.eng, fx.st, "oa", "oa", CFG, days[46])
    cur, _ = tc.when_table(0, child, 0, tr)
    night = int(3 * 60 // 15)
    busy = int((9 * 60 + 30) // 15)
    assert child.when.hist[0][busy] == 0
    assert cur[busy] > 10 * cur[night]
    assert cur[busy] > 1e-3


def test_eb_concentration_tracks_how_much_the_child_follows_its_parent():
    prior = np.zeros(96)
    prior[34:40] = 1.0
    prior = (prior + 1e-3) / (prior + 1e-3).sum()
    same = prior * 500
    narrow = np.zeros(96)
    narrow[34] = 40.0
    a_same = SC.eb_concentration(same, 500.0, prior)
    a_narrow = SC.eb_concentration(narrow, 40.0, prior)
    assert a_same > 100 * a_narrow
    f = SC.when_density_prior(narrow, 40.0, prior)
    assert abs(f.sum() - 1.0) < 1e-9 and f[0] < f[36] < f[34]


def test_slot_its_parent_has_seen_keeps_the_young_nodes_own_floor():
    """A slot the parent's population used now and then (a late login) but the
    young node never saw is not declared rarer than the node's own sample can
    say (pack O day 5: portal logins at 22:45 became ~1e-5 under the
    parent-prior density alone, +10 incidents >= LOW)."""
    fx, tr, route_nd, child, days = _young_department_node()
    for d in days[38:45]:
        for ip in OTHERS[:3]:
            _learn_path(fx, "oa", ip, d + (22 * 60 + 47) * 60.0)        # late logins of the route's others
    tc = CF._TreeCtx(fx.eng, fx.st, "oa", "oa", CFG, days[46])
    cur, _ = tc.when_table(0, child, 0, tr)
    h, N = tc._when_counts(child, 0)
    own = CF._hdr_table(SC.when_density(h, N))
    late = int((22 * 60 + 45) // 15)
    hier = CF._hdr_table(tc.when_pred(0, tr, child, 0))
    pt = tc.when_ptab(0, tr, route_nd, 0)
    assert pt[late] >= own[late]                           # the route does not find 22:45 rare
    assert hier[late] < own[late]                          # the prior alone would call it rarer
    assert cur[late] >= own[late]                          # ... the node keeps its own floor


def test_value_bound_to_another_source_up_the_path_is_a_cross_binding():
    """Pack O seed 2: the 综合部 login node (a who-split child) held .21 -> jack
    but not rose (bound to .23 at the route node); A2 - rose's credential from
    .21 - was 'unbound_value' (p 0.71) and went unreported."""
    import pytest
    fx, tr, route_nd, child, days = _young_department_node()
    users = dict(zip(GA, ["jack", "rose", "mike"]))
    pair = "net.src->body.kv.username"

    def rec(src_users):
        table = {ip: {"n": 15.0, "top": u, "k": 15.0, "bound": True, "LB": 0.95, "p_viol": 0.02}
                 for ip, u in src_users.items()}
        return {pair: {"x": "net.src", "y": "body.kv.username", "dir": "fwd", "fd": {"g3": 0.0, "holds": True},
                       "table": table, "bound_values": {u: [ip] for ip, u in src_users.items()}}}
    fx.put("oa", MP.PBIND, route_nd.id, pairs=rec(users))
    # the child's younger table: .21 bound, .23's value not (yet) in it
    m = fx.st.get_model("oa", SYSTEM_ENTITY, MP.PBIND)
    m["nodes"][0][child.id] = {"status": "fitted", "pairs": rec({"192.168.1.21": "jack"})}
    fx.st.put_model("oa", SYSTEM_ENTITY, MP.PBIND, m)
    t = days[46] + 8 * 3600 + 40 * 60
    _, asg = fx.score("oa", [(t, "192.168.1.21", {"http.route": LOGIN, "body.kv.username": "rose"})])
    assert int(asg.get("conf", 0)) == child.id
    v = [e for e in fx.violations("192.168.1.21") if e.extra["type"] == "content"]
    assert v and "cross_binding" in v[0].extra["flags"] and v[0].severity == Severity.MEDIUM
    assert asg.get("damp", 0) == pytest.approx(0.1)


def test_lease_of_a_pool_group_is_neither_outsider_nor_system_new():
    """研发's 24-h leases (pack O seed 2, days 11-12): P11 absorbed the day's
    leases into the merged pool group, whose id was its largest predecessor's
    (a code slice) - the OA nodes knew the OA slice's label only - and a lease
    has no recurring colleague signature, so leases logging in to OA were
    'outsider_group, system_new' MEDIUM findings."""
    fx = Fx()
    days = workdays(16)
    pool = [f"10.50.{k % 4}.{k + 2}" for k in range(150)]
    hier_ip2g = {}
    from app.models.schema import ORG
    wg = fx.st.get_model(ORG, ORG, MP.WHO_GROUPS)
    slice_ips = pool[:140]
    fx.st.put_model(ORG, ORG, MP.WHO_GROUPS, dict(wg, ip2g=dict(wg["ip2g"], **{ip: "G20" for ip in slice_ips})))
    for i, d in enumerate(days[:14]):
        for k, ip in enumerate(pool[i * 10:(i + 1) * 10]):
            fx.learn("oa", LOGIN, ip, d + (9 * 60 + 30 + 3 * k) * 60.0)
    nd = fx.node("oa", LOGIN)
    tr = fx.tree("oa", [LOGIN])
    root = tr.nodes[tr.root]
    t = days[15]
    new = pool[140]
    groups = {"G5": {"id": "G5", "name": "研发", "members": slice_ips + [new], "pool": "10.50.0.0/22",
                     "lineage": ["G20"]}}
    ip2g = {ip: "G5" for ip in slice_ips + [new]}
    lv3 = root.who.levels[CF.GRP_LEVEL]
    assert "grp:G20" in lv3 and "grp:G5" not in lv3
    # the system knows the group under its predecessor's label
    assert CF._label_known(lv3, "G5", groups)
    assert not CF._label_known(lv3, "G5", {"G5": dict(groups["G5"], lineage=[])})
    # a pool group's standing at the node is its leases' group-level evidence
    assert not CF.group_outsider(nd, "G5", new, t, ip2g, groups, lambda m: False)
    # (a static group with no recurring colleague at the node is an outsider)
    static = {"G5": dict(groups["G5"], pool=None)}
    assert CF.group_outsider(nd, "G5", new, t, ip2g, static, lambda m: False)
