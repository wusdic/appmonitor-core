"""P04 round 7 (tree owner)."""
from __future__ import annotations

from ptree_sim import DAY, MON

from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.lib import ptree as PT


def _route(i):
    return (("http.route", 0, frozenset({f"GET oa.corp.local /r{i}"}), False),)


def test_an_unstated_root_does_not_set_the_prior_of_statements():
    """(round 7) The empirical-Bayes prior of a statement's held-out
    confidence is the hold rate of statements like it. A tree's root over
    several actions is never stated (P14 states actions), and on pack O its
    transport bands - a mixture of every route - failed nearly every test
    (seed 1: oa's root 0 of 6.8 tests on day 10, portal's 0 of 9.8). Two or
    three such records in the pool made the beta-binomial fit U-shaped with
    mean 0.65-0.85, the confidence every untested statement then stated
    while they held 0.88-0.95 on the evaluator's held-out events. Fails on
    round-6 code (the roots' records join the pool: prior mean < 0.85)."""
    from app.engines.behavior.lib import ptree as PT
    from app.engines.behavior.pattern_tree import PatternTreeEngine
    eng = PatternTreeEngine()
    m = PT.PTreeModel("oa")
    tr = m.tree(0, MON)
    cons = {"who": (0, frozenset({"1.1.1.1"}), 0.95)}
    for i in range(10):                            # statements of actions: they hold
        nd = PN.Node(100 + i, tr.root, 1, 0, _route(i), MON)
        nd.state = "confirmed"
        nd.ref = {"hold": dict(cons)}
        nd.meta["hold"] = PN.HoldRecord()
        nd.meta["hold"].T = [6.0, 6.0] if i % 4 else [5.0, 6.0]
        tr.nodes[nd.id] = nd
    for j in range(3):                             # roots of other trees: never stated, always failing
        root = PN.Node(200 + j, None, 0, 0, (), MON)
        # a real root: the route partition (split on http.route) under one SNI
        # (pack O portal / oa); P14 states no action for it
        root.split = PT.Split("http.route", 0, [frozenset({"GET a /x"})], [300 + j], 400 + j)
        root.inv["tls.sni"] = (0, "www.corp.example")
        root.state = "stable"
        root.ref = {"hold": dict(cons)}
        root.meta["hold"] = PN.HoldRecord()
        root.meta["hold"].T = [0.0, 12.0]
        tr.nodes[root.id] = root
    fresh = PN.Node(99, tr.root, 1, 0, _route(99), MON)     # stated, no test yet
    fresh.state = "confirmed"
    fresh.ref = {"hold": dict(cons)}
    tr.nodes[fresh.id] = fresh
    eng._hold_priors(m, 1, MON)
    eng._hold_priors(m, 2, MON + DAY)
    p = fresh.p_hold(MON + DAY)
    assert p is not None and p >= 0.9, (p, fresh.meta.get("hold_prior"))


def _gen_sim(conf_of):
    """GET /a every hour (the root's mixture) and a daily POST /gen whose
    every row P03 damps (0.1) against the node `conf_of(s)` returns."""
    import numpy as np
    from ptree_sim import Sim, daily
    from app.engines.behavior.lib import pevent as EV
    sel = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)

    def damp(s):
        b = s.st.batch_at("oa", EV.EVT_BATCH, s.now)
        if b is None:
            return
        rows = []
        for i in range(b.n):
            if b.get("http.route", i) == "POST x /gen":
                rows.append({"damp": 0.1, "conf": float(conf_of(s))})
            else:
                rows.append({"damp": 1.0})
        s.st.add_batch("oa", EV.PAT_ASSIGN, s.now, b.aligned(EV.cols_from_rows(b.n, rows)))
    sim.hooks.append(damp)

    def day(d, t0, rng):
        ev = [(t0 + 8 * 3600 + h * 3600 + rng.uniform(0, 600), "oa", "10.0.0.%d" % (h % 5),
               {"http.route": "GET x /a", "net.bytes_up": 500.0}) for h in range(10)]
        ev += [(t0 + 17 * 3600 + k * 300, "oa", "10.0.0.9", {"http.route": "POST x /gen", "net.bytes_up": 50000.0})
               for k in range(2)]
        return ev
    sim.add(daily(day, 8, seed=2))
    return sim


def _gen_node(tr):
    for nd in tr.nodes.values():
        if any(a == "http.route" and set(vals) == {"POST x /gen"} and not neg for a, _l, vals, neg in nd.ctx):
            return nd
    return None


def test_a_new_action_damped_against_the_mixture_learns_undamped_at_its_own_node():
    """(round 7) P03 scores a row on its most specific CONFIRMED node; for a
    new action whose own node is a candidate that is the root - the mixture
    of every action - and the action's sizes are outliers there (pack O GA
    POST /report/generate: 20-60 KB bodies against the root's 0.2-1.5 KB).
    Damped at 0.1, the new node collected its 20 observations days late
    (eligible on day 12, confirmed on day 15-16 on 4 of 5 seeds). The
    action's own node learns such rows undamped; the root keeps the damping.
    Fails on round-6 code (n_obs ~ 0.1 per row)."""
    from ptree_sim import MON
    sim = _gen_sim(lambda s: s.tree().root if s.tree() is not None else 0)
    sim.run_until(MON + 8 * DAY)
    tr = sim.tree()
    gen = _gen_node(tr)
    assert gen is not None
    assert gen.n_obs() >= 10.0, gen.n_obs()        # 16 rows, at full weight once the node exists


def test_a_row_damped_against_its_own_actions_node_stays_damped():
    """Guard: a row damped against a node that states its action (an
    outlier of its own pattern) is learned damped everywhere, as before."""
    from ptree_sim import MON
    sim = _gen_sim(lambda s: (_gen_node(s.tree()).id if s.tree() is not None and _gen_node(s.tree()) is not None
                              else -1))
    sim.run_until(MON + 8 * DAY)
    gen = _gen_node(sim.tree())
    assert gen is not None
    assert gen.n_obs() <= 4.0, gen.n_obs()


def test_a_confirmed_node_without_a_reference_states_the_pooled_base_rate():
    """(round 7) A node confirmed at a day's end has no reference statement
    (and so no held-out record or kind prior) until SNAPSHOT_HOUR; its
    p_hold was None that day and P14 stated the min-over-parts fallback
    (pack O seed 0, day 17: new GET /docs nodes stated 0.08, held 1.0). It
    states the pooled prior's mean instead."""
    from app.engines.behavior.pattern_tree import PatternTreeEngine
    eng = PatternTreeEngine()
    m = PT.PTreeModel("oa")
    tr = m.tree(0, MON)
    cons = {"who": (0, frozenset({"1.1.1.1"}), 0.95)}
    for i in range(10):
        nd = PN.Node(100 + i, tr.root, 1, 0, _route(i), MON)
        nd.state = "confirmed"
        nd.ref = {"hold": dict(cons)}
        nd.meta["hold"] = PN.HoldRecord()
        nd.meta["hold"].T = [6.0, 6.0]
        tr.nodes[nd.id] = nd
    new = PN.Node(99, tr.root, 1, 0, _route(99), MON)
    new.state = "confirmed"                         # no ref, no hold record yet
    tr.nodes[new.id] = new
    eng._hold_priors(m, 1, MON)
    eng._hold_priors(m, 2, MON + DAY)
    p = new.p_hold(MON + DAY)
    assert p is not None and p >= 0.85, p
