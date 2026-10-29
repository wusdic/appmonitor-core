"""P04 / pnode / prender, round 10 (tree owner)."""
from __future__ import annotations

from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.lib import prender as PR


def _keys(ip: str):
    return [ip, ip.rsplit(".", 1)[0] + ".0/24", "192.168.0.0/16", None, None]


def test_a_long_judged_foreign_member_clears_soon_after_p03_accepts_it():
    """The suspect test is capped (CUSUM): weeks of rows P03 damped as foreign
    do not have to be repaid row for row once P03 accepts the source (its
    group gained standing). O-real R3: the readdressed clerk 192.168.2.51 at
    the finance ledger node."""
    w = PN.WhoSummary()
    t = 1_790_035_200.0
    for i in range(30):                                  # colleagues of its /24 use the node
        for ip in ("192.168.2.10", "192.168.2.12"):
            w.update(_keys(ip), ip, t + i * 60.0, 1.0, 1.0)
    new = "192.168.2.51"
    for i in range(300):                                 # three weeks of damped rows
        w.mark_suspect(new, t + 3600.0 + i * 600.0)
    assert w.is_suspect(new, t + 3600.0 + 300 * 600.0)
    t1 = t + 3600.0 + 301 * 600.0
    cleared = None
    for j in range(40):                                  # now accepted, supported by its /24
        if w.observe_suspect(new, t1 + j * 60.0, w.support(_keys(new), new, t1 + j * 60.0)):
            cleared = j + 1
            break
    assert cleared is not None and cleared <= 8


def test_an_unsupported_suspect_still_never_clears():
    """A source without colleagues at the node (a slow poisoner, A9) stays
    out however long it persists: the cap does not clear anything by itself."""
    w = PN.WhoSummary()
    t = 1_790_035_200.0
    w.update(_keys("192.168.2.10"), "192.168.2.10", t, 1.0, 1.0)
    for i in range(50):
        w.mark_suspect("10.9.9.9", t + i * 600.0)
    for j in range(200):
        assert not w.observe_suspect("10.9.9.9", t + 40000.0 + j * 60.0, False)
    assert w.is_suspect("10.9.9.9", t + 40000.0 + 200 * 60.0)


def test_a_group_who_clause_names_the_nodes_sources_outside_the_group():
    """A node whose observed sources include an address outside its group's
    P11 membership must not state the group's members alone (O-real R3: the
    clerk's new address 192.168.2.51 at 财务部's nodes, outside the group P11
    had not regrouped yet)."""
    w = PN.WhoSummary()
    t = 1_790_035_200.0

    def k(ip, g):
        return [ip, ip.rsplit(".", 1)[0] + ".0/24", "192.168.0.0/16", g, None]
    for d in range(8):
        for j in range(20):
            ts = t + d * 86400 + j * 60
            for ip in ("192.168.2.10", "192.168.2.12"):
                w.update(k(ip, "grp:G18"), ip, ts, 1.0, 1.0)
            if j % 6 == 0 and d >= 3:
                w.update(k("192.168.2.51", "grp:∅"), "192.168.2.51", ts, 1.0, 1.0)
    for i in range(40):                      # one-off visitors: the address level stays open
        w.update(k(f"192.168.2.{100 + i}", "grp:∅"), f"192.168.2.{100 + i}", t + 7 * 86400 + i, 0.01, 1.0)
    now = t + 8 * 86400
    groups = {"G18": {"members": ["192.168.2.10", "192.168.2.12"], "name": "财务部"}}
    ev, zh, en, _c = PR.who_block(w, now, 8, {"192.168.2.10": "G18", "192.168.2.12": "G18"}, groups)
    assert ev["level"] == "grp"
    assert "192.168.2.51" in ev["members"] and "192.168.2.51" in zh and "192.168.2.51" in en
    assert set(ev["members"]) >= {"192.168.2.10", "192.168.2.12"}


# --------------------------------------------------------------- item 2
OLD_R = "POST oa.corp.local /approval/{num}/approve"
NEW_R = "POST oa.corp.local /flow/{num}/approve"


def _approvals(d, t0, rng):
    from ptree_sim import is_workday
    out = []
    if not is_workday(t0 + 12 * 3600):
        return out
    route = OLD_R if d < 10 else NEW_R
    for k in range(4):
        ts = t0 + (9.5 + 0.25 * k + 0.05 * rng.random()) * 3600.0
        out.append((ts, "oa", "192.168.1.21", {"http.route": route, "http.method": "POST",
                                                "body.len": float(rng.integers(900, 1400)),
                                                "net.bytes_down": float(rng.integers(300, 600))}))
    for k in range(6):                     # other traffic of the system
        ts = t0 + (9.0 + k) * 3600.0 + 60.0 * rng.random()
        out.append((ts, "oa", f"192.168.1.{30 + k}", {"http.route": "GET oa.corp.local /docs",
                                                       "http.method": "GET",
                                                       "net.bytes_down": float(rng.integers(5000, 9000))}))
    return out


def test_a_renamed_route_hands_its_pattern_over_to_the_new_routes_node():
    """Item 2 (the conformity owner's round-9 finding): a route renamed
    mid-run (P10 confirmed the rename: model.pflow 'renamed'). Once the new
    route's node is confident and the old route's node stale, the old node's
    when / target summaries move into the new node (its statement keeps the
    pattern's history: windows and bands from all its days, not its first
    few) and the old node retires; before, the old node stayed stale and its
    history was dropped with it 30 days later."""
    from ptree_sim import DAY, MON, Sim, daily
    from app.engines.behavior.lib import m_ptree as MP
    from app.models.schema import SYSTEM_ENTITY

    sim = Sim(p05=True)
    sim.add(daily(_approvals, 24))

    def pflow(s):
        if s.now >= MON + 12 * DAY:
            s.st.put_model("oa", SYSTEM_ENTITY, MP.PFLOW,
                           {"renamed": {NEW_R: {"from": OLD_R, "day": 12, "sources": ["192.168.1.21"]}}}, ts=s.now)
    sim.hooks.append(pflow)
    old_when = {}

    def watch(s):
        tr = s.tree()
        if tr is None:
            return
        for nd in tr.nodes.values():
            rc = [c for c in nd.ctx if c[0] == "http.route"]
            if rc and set(rc[-1][2]) == {OLD_R} and nd.split is None:
                old_when["mass"] = nd.when.mass(0, s.now)
                old_when["n"] = nd.n_obs()
    sim.hooks.append(watch)
    sim.run_until(MON + 24 * DAY)
    tr = sim.tree()
    news = [nd for nd in tr.nodes.values() if [c for c in nd.ctx if c[0] == "http.route"]
            and set([c for c in nd.ctx if c[0] == "http.route"][-1][2]) == {NEW_R} and nd.split is None]
    olds = [nd for nd in tr.nodes.values() if [c for c in nd.ctx if c[0] == "http.route"]
            and set([c for c in nd.ctx if c[0] == "http.route"][-1][2]) == {OLD_R}]
    assert len(news) == 1, [(n.id, n.ctx, n.state) for n in tr.nodes.values()]
    nn = news[0]
    assert nn.state in ("confirmed", "stable")
    assert (nn.meta.get("handover") or {}).get("route") == OLD_R
    assert not olds                                        # the old node retired into it
    assert old_when.get("n", 0) > 0
    # its history: the old route's observations are the new node's too
    assert nn.n_obs() >= old_when["n"]
    assert "body.len" in nn.targets
    ev = [e for e in sim.events("pattern_retired") if (e.extra or {}).get("renamed_to") == NEW_R]
    assert len(ev) == 1
