"""P04, round 12 (learning owner): a new node's birth summaries."""
from __future__ import annotations

from ptree_sim import DAY, MON, Sim, daily


def _evidence(sm, t):
    for nm in ("ss", "values"):
        x = getattr(sm, nm, None)
        if x is not None and hasattr(x, "total"):
            return float(x.total(t))
    return float(sm.moments(t)[0])


def test_a_new_node_learns_its_birth_summaries_until_p05_ranks_it():
    """Pack O seed 3 (rounds 10-11): the OA approve node p:oa:0:10 was born on
    day 2 with 24 replayed summaries, the approval opinion among them. P05
    refits node lists once a day and needs n_min probe rows of a node, so for
    its first day the node had no list of its own and learned under its
    nearest ancestor's (the root's: the login form's fields, no opinion); the
    rare opinion 退回 lost that day's evidence and P07 could not show the set
    closed until day 11 (GA.oa.approvals#2 missing on days 6-10). A node P05
    has not ranked yet keeps learning the summaries it was born with, for at
    most BORN_DAYS; once P05 lists it, its own list rules."""
    from app.engines.behavior import pattern_tree as PT_
    sel = {"targets_sys": {0: ["net.bytes_up", "body.kv.username"]},
           "split_cands": {0: [("net.src", 1), ("net.src", 0)]}, "roles": {}}
    sim = Sim(sel=sel)
    ga = {"192.168.1.21": "jack", "192.168.1.23": "rose", "10.168.7.121": "mike"}
    pop = [f"192.168.3.{i}" for i in range(20, 32)]
    login = {"http.route": "POST oa /login", "http.method": "POST"}

    def day(d, t0, rng):
        ev = [(t0 + (9 * 60 + rng.uniform(0, 21)) * 60, "oa", ip,
               dict(login, **{"net.bytes_up": float(rng.uniform(1024, 2048)), "body.kv.username": u}))
              for ip, u in ga.items()]
        ev += [(t0 + rng.uniform(13 * 60, 17 * 60) * 60, "oa", ip,
                dict(login, **{"net.bytes_up": float(rng.uniform(600, 1400)), "body.kv.username": f"s{i:02d}x"}))
               for i, ip in enumerate(pop)]
        return ev
    sim.add(daily(day, 20, seed=1))
    kids = None
    while sim.now < MON + 20 * DAY and kids is None:
        sim.run_until(sim.now + sim.dt)
        sp = sim.splits()
        if sp:
            kids = [int(k) for k in sp[0][4]]
    assert kids, "no split in 20 days"
    tr = sim.tree()
    kids = [k for k in kids if k in tr.nodes and "body.kv.username" in tr.nodes[k].targets]
    assert kids, "no child inherited the username summary"
    # the system list moves on and P05 has no list for the new nodes yet: they
    # inherit the system list without the username
    sim.sel = dict(sel, targets_sys={0: ["net.bytes_up"]})
    t0 = sim.now
    e0 = {k: _evidence(tr.nodes[k].targets["body.kv.username"], t0) for k in kids}
    sim.run_until(t0 + 1.0 * DAY)
    tr = sim.tree()
    grown = [k for k in kids if k in tr.nodes and "body.kv.username" in tr.nodes[k].targets
             and _evidence(tr.nodes[k].targets["body.kv.username"], sim.now) > e0[k] + 0.5]
    assert grown, "the new nodes stopped learning the username they were born with"
    # once P05 ranks a node (its own list without the username) the birth list ends
    sim.sel = dict(sel, targets_sys={0: ["net.bytes_up"]},
                   node_overrides={0: {k: ["net.bytes_up"] for k in kids}})
    sim.run_until(sim.now + 1.5 * DAY)
    tr = sim.tree()
    for k in kids:
        if k in tr.nodes:
            assert tr.nodes[k].meta.get("born_tl") is None
    assert PT_.BORN_DAYS >= 1


def test_birth_summaries_p05_leaves_out_end_with_the_birth_list():
    """Pack O seed 0 (round 12, born_tl): mail's 研发 node p:mail:0:3, born on
    day 4 from its parent's replayed rows, learned their net.pkts_down /
    net.pkts_up as birth summaries; P05 listed the node that evening without
    them, yet the summaries stayed to TG_STALE_DAYS (day 7): P06 stated their
    band, fitted to the parent's mixture, the node's first held-out test failed
    on it and the statement said 0.33 (code11 0.67) on days 6-7 while it held.
    A birth summary that neither the node's own P05 list nor its ancestors'
    list holds ends with the birth list."""
    sel = {"targets_sys": {0: ["net.bytes_up", "body.kv.username"]},
           "split_cands": {0: [("net.src", 1), ("net.src", 0)]}, "roles": {}}
    sim = Sim(sel=sel)
    ga = {"192.168.1.21": "jack", "192.168.1.23": "rose", "10.168.7.121": "mike"}
    pop = [f"192.168.3.{i}" for i in range(20, 32)]
    login = {"http.route": "POST oa /login", "http.method": "POST"}

    def day(d, t0, rng):
        ev = [(t0 + (9 * 60 + rng.uniform(0, 21)) * 60, "oa", ip,
               dict(login, **{"net.bytes_up": float(rng.uniform(1024, 2048)), "body.kv.username": u}))
              for ip, u in ga.items()]
        ev += [(t0 + rng.uniform(13 * 60, 17 * 60) * 60, "oa", ip,
                dict(login, **{"net.bytes_up": float(rng.uniform(600, 1400)), "body.kv.username": f"s{i:02d}x"}))
               for i, ip in enumerate(pop)]
        return ev
    sim.add(daily(day, 20, seed=1))
    kids = None
    while sim.now < MON + 20 * DAY and kids is None:
        sim.run_until(sim.now + sim.dt)
        sp = sim.splits()
        if sp:
            kids = [int(k) for k in sp[0][4]]
    assert kids, "no split in 20 days"
    tr = sim.tree()
    kids = [k for k in kids if k in tr.nodes and "body.kv.username" in tr.nodes[k].targets]
    assert kids, "no child inherited the username summary"
    sim.sel = dict(sel, targets_sys={0: ["net.bytes_up"]})
    sim.run_until(sim.now + 0.5 * DAY)
    tr = sim.tree()
    assert any(k in tr.nodes and tr.nodes[k].meta.get("born_tl") for k in kids)
    # P05 ranks the new nodes without the username: the birth summary goes
    # with the birth list, not TG_STALE_DAYS later
    sim.sel = dict(sel, targets_sys={0: ["net.bytes_up"]},
                   node_overrides={0: {k: ["net.bytes_up"] for k in kids}})
    sim.run_until(sim.now + 1.5 * DAY)
    tr = sim.tree()
    alive = [k for k in kids if k in tr.nodes]
    assert alive
    for k in alive:
        assert tr.nodes[k].meta.get("born_tl") is None
        assert "body.kv.username" not in tr.nodes[k].targets, k


def test_the_held_out_test_does_not_check_a_fit_of_a_dropped_summary():
    """Pack O seed 0 (round 12): mail's 研发 node dropped its birth summary
    net.pkts_down at midnight of day 5; the reference built at 04:00 still
    took P06's band of it from the fit of the evening before (P06 refits
    hourly, and only once the tree learns again), so the whole of day 5 was
    checked against a band no statement made that day, the node's first
    held-out test failed on it and the statement said 0.33 while it held. The
    held-out constraints take the fitters' entries of the node's current
    summaries only (and P06's per-source rate, which has no summary)."""
    from app.engines.behavior import pattern_tree as PT_
    from app.engines.behavior.lib import m_ptree as MP_
    sel = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)
    pop = [f"192.168.3.{i}" for i in range(20, 32)]
    login = {"http.route": "POST oa /login", "http.method": "POST"}

    def day(d, t0, rng):
        return [(t0 + rng.uniform(9 * 60, 17 * 60) * 60, "oa", ip,
                 dict(login, **{"net.bytes_up": float(rng.uniform(600, 1400))}))
                for ip in pop for _ in range(2)]
    sim.add(daily(day, 8, seed=2))
    sim.run_until(MON + 6 * DAY)
    tr = sim.tree()
    nd = next(n for n in tr.nodes.values() if "net.bytes_up" in n.targets)
    t = sim.now
    targets = {a: PT_._compact(sm, t) for a, sm in nd.targets.items()}
    assert "net.pkts_down" not in targets
    band = {"band90": [1.0, 9.0], "coverage": 0.9}
    rate = "rate.ip_h"                  # content_bounds.RATE_ATTR
    fit = {MP_.PBOUNDS: {"attrs": {"net.bytes_up": dict(band), "net.pkts_down": dict(band),
                                   rate: dict(band)}},
           MP_.PGRAMMAR: {"attrs": {"body.kv.gone": {"kind": "text", "closed": ["a", "b"], "U": 0.01}}}}
    cons = PT_._hold_constraints(nd, t, targets, fit)
    assert "net.bytes_up" in cons and rate in cons
    assert "net.pkts_down" not in cons and "body.kv.gone" not in cons
