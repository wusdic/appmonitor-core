"""P03 round 10: a route node's time-of-day law starts with itself
(conformity._TreeCtx.time_parent; docs/lib3/progressive.md §16.17.11 item 4,
§16.17.12 item 7).

O-red seed 0 (round 9) missed A4 (10.77.2.14's 03:05 OA login, a source that
logs in at 08:xx): no damping was involved - the login's when p was 2.0e-3
(> 1e-3, no finding). The hierarchical back-off of the slot density took the
ROOT's law as the login route node's prior, and the root of a system mixes
every action: AUTO's /health monitor polls around the clock, a third of the
root's mass at night, so the route node (339 evidence units, none at night)
read 03:00 at 4.1e-3 against its own 1.3e-4 and its young 'other sources'
child inherited 1.0e-3. A route split's children are different actions, not
samples of the mixture above them."""
from __future__ import annotations

import datetime as _dt

import numpy as np

from app.engines.behavior import conformity as CF
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.lib import pscore as SC
from app.models.schema import SYSTEM_ENTITY, Severity

from test_p03_conformity import CFG, GA, LOGIN, TZ, Fx, local_minute, workdays

HEALTH = "GET oa /health"
OTHERS = [f"192.168.3.{i}" for i in range(20, 32)]


def _learn_path(fx: Fx, ip: str, ts: float, route: str) -> None:
    tr = fx.tree("oa", [LOGIN, HEALTH])
    hier = MP.hierarchies(fx.st, "oa", CFG)
    keys = [hier.gen("net.src", l, ip) for l in range(PN.WHO_LEVELS)]
    day = int((ts + 8 * 3600) // 86400.0)
    wd = 0 if _dt.datetime.fromtimestamp(ts, TZ).weekday() < 5 else 1
    attrs = {"http.route": route, "net.src": ip}
    for nid in tr.route(lambda a: attrs.get(a), hier, (), ts):
        tr.nodes[nid].update_core(ts, 1.0, 1.0, keys, ip, wd, local_minute(ts), day)


def _system_with_a_monitor():
    """A login route learned over 30 workdays (15 sources, 08:30-10:00, ~340
    units at H_m) beside a /health monitor polling every 10 minutes around
    the clock at the root; the login route then splits off its 综合部
    sources (young child, 15 workdays of 08:30-08:51)."""
    fx = Fx()
    tr = fx.tree("oa", [LOGIN, HEALTH])
    rng = np.random.default_rng(7)
    days = workdays(48)
    for d in days[:45]:
        for k in range(0, 1440, 10):
            _learn_path(fx, "192.168.9.9", d + k * 60.0, HEALTH)
    for d in days[:30]:
        for ip in GA + OTHERS:
            _learn_path(fx, ip, d + (8 * 60 + 30 + 90 * rng.random()) * 60.0, LOGIN)
    route_nd = fx.node("oa", LOGIN)
    tr.split(route_nd.id, "net.src", 0, [GA], days[30])
    child = tr.nodes[route_nd.split.children[0]]
    for d in days[30:45]:
        for ip in GA:
            _learn_path(fx, ip, d + (8 * 60 + 30 + 21 * rng.random()) * 60.0, LOGIN)
        for ip in OTHERS:
            _learn_path(fx, ip, d + (9 * 60 + 60 * rng.random()) * 60.0, LOGIN)
    for nd in tr.nodes.values():
        nd.state = "stable"
    fx.put("oa", MP.PWIN, child.id, attrs={})
    m = fx.st.get_model("oa", SYSTEM_ENTITY, MP.PWIN)
    m["nodes"][0][child.id]["when"] = {"workday": [[510, 531]], "nonworkday": []}
    return fx, tr, route_nd, child, days


def test_a_route_nodes_time_law_does_not_back_off_to_the_systems_mixture():
    fx, tr, route_nd, child, days = _system_with_a_monitor()
    t = days[47] + 3 * 3600 + 5 * 60                              # 03:05 local
    # the 03:05 login of a 综合部 address is a MEDIUM 'when' finding (a write at night)
    b, asg = fx.score("oa", [(t, "10.168.7.121", {"http.route": LOGIN})])
    assert asg.get("p_when", 0) <= 1e-3
    w = [e for e in fx.violations("10.168.7.121") if e.extra["type"] == "when"]
    assert w and w[0].severity == Severity.MEDIUM
    tc = CF._TreeCtx(fx.eng, fx.st, "oa", "oa", CFG, t)
    assert tc.time_parent(tr, route_nd) is None                   # the route node starts the hierarchy
    assert tc.time_parent(tr, child) is route_nd                  # a source split keeps its parent
    slot = int(local_minute(t) // 15)
    root = tr.nodes[tr.root]
    assert root.when.hist[0][slot] > 0                            # the monitor uses 03:00
    h, N = tc._when_counts(route_nd, 0)
    own = CF._hdr_table(SC.when_density(h, N))
    cur, _ = tc.when_table(0, route_nd, 0, tr)
    assert cur[slot] <= own[slot] * (1 + 1e-9)                    # the route's own resolution


def test_a_slot_the_routes_own_population_uses_is_not_rare():
    """09:30: empty at the young 综合部 node, used by the route's other sources."""
    fx, tr, route_nd, child, days = _system_with_a_monitor()
    tc = CF._TreeCtx(fx.eng, fx.st, "oa", "oa", CFG, days[46])
    cur, _ = tc.when_table(0, child, 0, tr)
    busy = int((9 * 60 + 30) // 15)
    night = int(3 * 60 // 15)
    assert child.when.hist[0][busy] == 0
    assert cur[busy] > 1e-3 and cur[busy] > 10 * cur[night]
