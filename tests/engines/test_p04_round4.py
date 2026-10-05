"""P04 round 4 (docs/lib3/progressive.md §16.12): rows learned before a node
learned reach its split children; calibrated, monotone statement confidence;
tree-side precision."""
from __future__ import annotations

import math

import numpy as np
import pytest

from ptree_sim import DAY, MON, Sim, daily, is_workday

from app.engines.behavior.lib import pnode as PN

GA = ["192.168.1.21", "192.168.1.23", "192.168.1.25"]
POP = [f"192.168.3.{i}" for i in range(20, 32)]
LOGIN = {"http.route": "POST oa /login", "http.method": "POST"}


def _login(ts, ip, size):
    return (ts, "oa", ip, dict(LOGIN, **{"net.bytes_up": float(size)}))


def _ga_range(small_day: int):
    sel = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: [("net.src", 1)]}, "roles": {}}
    sim = Sim(sel=sel)

    def day(d, t0, rng):
        if not is_workday(t0):
            return []
        ev = []
        for i, ip in enumerate(GA):
            size = 520.0 if (d == small_day and i == 0) else rng.uniform(1024, 2048)
            ev.append(_login(t0 + (9 * 60 + rng.uniform(0, 21)) * 60, ip, size))
        for ip in POP:
            ev.append(_login(t0 + rng.uniform(13 * 60, 17 * 60) * 60, ip, rng.uniform(600, 1400)))
        return ev
    sim.add(daily(day, 14, seed=31))
    sim.run_until(MON + 14 * DAY)
    assert sim.splits(), "no split"
    tr = sim.tree()
    ga = [nd for nd in tr.nodes.values() if nd.parent is not None and nd.split is None
          and set(nd.who.heavy_set(0, sim.now)[0]) == set(GA)]
    assert ga, "no 综合部 node"
    num = ga[0].targets["net.bytes_up"]
    ring_days = sorted(int(x) for x in num.ring[:, 0] if np.isfinite(x))
    lo, hi, _ = num.observed_range(ring_days[-1])
    if num.log:
        lo, hi = math.exp(lo), math.exp(hi)
    return lo, hi


@pytest.mark.parametrize("small_day", [0, 1])
def test_rows_before_the_node_learned_reach_the_split_childs_range(small_day):
    """综合部's 520 B login comes on the route's first day (its rows wait for
    the route node and are replayed into it at its birth) or on the node's
    first day (before it has LEARN_MIN units and starts learning). Round 3's
    per-(candidate, value) extremes and row reservoir exist only while the
    node learns, so the 综合部 child created by the later /24 split stated a
    range starting above 1 KB (pack O seed 1: the 643 B login of day 2, the
    '100 % in 0.5-3 KB' clause). The leaf keeps per-source extremes over its
    whole life (pnode.SourceExtremes) and a source split's child starts from
    its sources' extremes."""
    lo, hi = _ga_range(small_day)
    assert lo <= 520.0 + 1e-6 and hi <= 2048.0 + 1e-6, (lo, hi)


def test_source_extremes_bounded_and_mergeable():
    sx = PN.SourceExtremes(K=4)
    for i in range(10):
        sx.note(f"10.0.0.{i}", "x", float(i), 5, 100.0 + i)
    assert len(sx) == 4 and set(sx.d) == {f"10.0.0.{i}" for i in range(6, 10)}
    sx.note("10.0.0.9", "x", -1.0, 6, 200.0)
    u = sx.union(lambda ip: ip.endswith(".9"))
    assert u["x"][0] == -1.0 and u["x"][1] == 6 and u["x"][2] == 9.0
    # an extreme older than the ring is replaced by the next value
    sx.note("10.0.0.9", "x", 3.0, 6 + PN.RING_DAYS, 300.0)
    assert sx.union(lambda ip: ip.endswith(".9"))["x"][:2] == [3.0, 6 + PN.RING_DAYS]
    other = PN.SourceExtremes(K=4)
    other.note("10.0.0.1", "x", 42.0, 7, 400.0)
    sx.merge(other)
    assert len(sx) == 4 and "10.0.0.1" in sx.d


# ---------------------------------------------------- calibrated confidence
def _week_tests(hr, t, n_tests, n_per_test, k_cons, q, rng, nom=0.9):
    """n_tests weekly held-out tests of a statement of k_cons constraints whose
    true coverage is q (nominal nom); one event every few hours."""
    for _ in range(n_tests):
        for e in range(n_per_test):
            ts = t + e * 3600.0 * 6
            for c in range(k_cons):
                hr.add(f"c{c}", bool(rng.random() < q), nom, ts)
        t += PN.HOLD_BATCH_S + 60.0
    hr.add("c0", True, nom, t)                     # closes the last test
    return t


def test_small_batch_test_uses_the_batch_tolerance():
    """A rare pattern (20 checked events a week) whose 12 constraints hold at
    their nominal 90 % coverage: a held-out test passes when each constraint's
    coverage on the batch is within the batch's own 3-sigma error. Round 3
    judged a 20-event batch by the error of a 300-event check (0.85 for a
    90 % band), so a correct band failed one weekly test in four and the
    statement ~all of them (pack O: statements stated 0.47-0.55 while holding
    0.68-0.81, their constraints at or above nominal on every kind). A wrong
    statement (coverage 0.6) still fails its tests."""
    rng = np.random.default_rng(4)
    good, bad = PN.HoldRecord(), PN.HoldRecord()
    _week_tests(good, MON, 8, 20, 12, 0.9, rng)
    t = _week_tests(bad, MON, 8, 20, 12, 0.6, rng)
    assert good.p_hold(t) >= 0.8, good.tests()
    assert bad.p_hold(t) <= 0.2, bad.tests()
    assert PN.batch_tol(0.9, 300) == pytest.approx(PN.hold_eps(0.9))


def test_confidence_does_not_fall_between_tests_or_when_a_test_passes():
    """The record forgets per test, not per day: between two weekly tests the
    stated confidence is constant, and a passed test never lowers it. With
    round 3's 14-day time decay a weekly-tested stable statement lost ~30 %
    of its record between tests and fell back toward its prior (PG2's
    'median confidence non-decreasing' failed on such day-to-day dips)."""
    rng = np.random.default_rng(5)
    hr = PN.HoldRecord()
    pr = (2.0, 2.0)
    t = MON
    trace = []
    for wk in range(10):
        t = _week_tests(hr, t, 1, 15, 4, 1.0, rng)
        trace.append(hr.p_hold(t, pr))
        trace.append(hr.p_hold(t + 5 * DAY, pr))     # days without a test
    assert all(b >= a - 1e-12 for a, b in zip(trace, trace[1:])), trace
    assert trace[-1] > 0.8


def test_statement_prior_is_kept_for_its_confidence_segment():
    """Once a statement has its own held-out tests it keeps the empirical-Bayes
    prior it had for the rest of its confidence segment: a later refit of the
    pooled prior (other statements failing their tests around a drift day)
    does not move it. A statement without a test states its kind's current
    base rate, and a structural restart of its record (HoldRecord.drop()) takes
    the kind's current prior."""
    from app.engines.behavior.lib import ptree as PT
    from app.engines.behavior.pattern_tree import PatternTreeEngine
    eng = PatternTreeEngine()
    m = PT.PTreeModel("oa")
    tr = m.tree(0, MON)
    nodes = []
    for i in range(12):
        nd = PN.Node(100 + i, tr.root, 1, 0, (), MON)
        nd.state = "confirmed"
        nd.ref = {"hold": {"who": (0, frozenset({"1.1.1.1"}), 0.95)}}
        hr = nd.meta["hold"] = PN.HoldRecord()
        hr.T = [9.0, 10.0]
        tr.nodes[nd.id] = nd
        nodes.append(nd)
    eng._hold_priors(m, 1, MON)                    # pool day 1
    eng._hold_priors(m, 2, MON + DAY)              # prior fitted from day 1, assigned
    p0 = nodes[0].meta["hold_prior"]
    assert p0 is not None and p0[0] / sum(p0) > 0.7
    for nd in nodes[1:]:                           # the others start failing (a drift)
        nd.meta["hold"].T = [1.0, 10.0]
    fresh = PN.Node(99, tr.root, 1, 0, (), MON)    # stated, not tested yet
    fresh.state = "confirmed"
    fresh.ref = {"hold": {"who": (0, frozenset({"1.1.1.1"}), 0.95)}}
    tr.nodes[fresh.id] = fresh
    eng._hold_priors(m, 3, MON + 2 * DAY)
    eng._hold_priors(m, 4, MON + 3 * DAY)          # refit on the failing pool
    assert nodes[0].meta["hold_prior"] == p0       # untouched statement keeps its prior
    pf = fresh.meta["hold_prior"]
    assert pf[0] / sum(pf) < p0[0] / sum(p0)       # untested: the kind's current base rate
    nodes[0].meta["hold"].drop()                   # a new confidence segment
    eng._hold_priors(m, 5, MON + 4 * DAY)
    p1 = nodes[0].meta["hold_prior"]
    assert p1 != p0 and p1[0] / sum(p1) < p0[0] / sum(p0)


def test_who_nominal_is_the_share_of_the_listed_sources():
    """The who constraint's nominal is the share the listed heavy set holds
    (WhoSummary.stated_unseen), not 1 - U: a node whose heavy set covers 95 %
    with a 5 % tail of light colleagues states ~0.95, so a batch where the tail
    holds its usual 5 % passes (pack O: who held 0.95-0.96 against 0.997)."""
    from app.engines.behavior.pattern_tree import _hold_constraints
    nd = PN.Node(1, 0, 1, 0, (), MON)
    t = MON
    heavy = [f"192.168.1.{i}" for i in range(1, 7)]
    light = [f"192.168.3.{i}" for i in range(1, 9)]
    for d in range(10):
        for k in range(200):
            t = MON + d * DAY + k * 60.0
            ip = light[k % 8] if k % 20 == 0 else heavy[k % 6]
            nd.update_core(t, 1.0, 1.0, [ip, ip.rsplit(".", 1)[0] + ".0/24", "192.168.0.0/16", "grp:∅", "reg:∅"],
                           ip, 0, 600.0, 739900 + d)
    lvl = nd.who.closed_level(t, nd.n_days())
    assert lvl is not None
    cons = _hold_constraints(nd, t, {})
    level, items, nom = cons["who"]
    share = nd.who.heavy_set(level, t)[1]
    assert nom <= share + 1e-9 and nom < 0.99, (level, len(items), nom, share)
    assert nd.who.stated_unseen(level, t) >= 1.0 - share - 1e-9


def test_part_confidence_is_its_groups_own_held_out_record():
    """A shared node's part for a learned group states the held-out tests of
    the group's own events (Node.p_hold_group): one group's events always
    hold, the other's fail; the node's p_hold mixes both."""
    from app.engines.behavior.pattern_tree import PatternTreeEngine
    nd = PN.Node(1, 0, 1, 0, (), MON)
    nd.state = "confirmed"
    nd.ref = {"t": MON, "hold": {"when": ("win", {0: ((480.0, 600.0),)}, 0.9)}}
    t = MON
    for k in range(12000):                     # 42 days, 40 sources (a test is 100 source-days)
        t += 300.0
        g = "grp:G1" if k % 2 else "grp:G2"
        minute = 500.0 if g == "grp:G1" else (500.0 if k % 4 == 0 else 700.0)
        keys = [f"10.0.{k % 40}.1", f"10.0.{k % 40}.0/24", "10.0.0.0/16", g, "reg:∅"]
        PatternTreeEngine._hold_check(None, nd, lambda a: None, keys, 0, minute, t, 1.0)
    assert nd.p_hold_group(["G1"]) > 0.9
    assert nd.p_hold_group(["grp:G2"]) < 0.1
    assert nd.p_hold_group(["G9"]) is None
    assert nd.p_hold(t) < 0.1


def test_pool_leases_do_not_evict_a_recurring_sources_extremes():
    """A DHCP pool's daily leases (80 new addresses a day, logging in after
    综合部) pass through the login node: per-address records alone (64 kept)
    would lose 综合部's early 520 B login to the leases. The /24 records keep
    it for a prefix split (and the address records evict the least-seen key,
    a one-off lease, before a recurring address)."""
    sel = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: [("net.src", 1)]}, "roles": {}}
    sim = Sim(sel=sel)

    def day(d, t0, rng):
        if not is_workday(t0):
            return []
        ev = []
        for i, ip in enumerate(GA):
            size = 520.0 if (d == 1 and i == 0) else rng.uniform(1024, 2048)
            ev.append(_login(t0 + (9 * 60 + rng.uniform(0, 21)) * 60, ip, size))
        for k in range(80):
            ip = f"10.50.{int(rng.integers(0, 4))}.{int(rng.integers(1, 255))}"
            ev.append(_login(t0 + rng.uniform(10 * 60, 17 * 60) * 60, ip, rng.uniform(600, 1400)))
        return ev
    sim.add(daily(day, 14, seed=7))
    sim.run_until(MON + 14 * DAY)
    tr = sim.tree()
    ga = [nd for nd in tr.nodes.values() if nd.parent is not None and nd.split is None
          and set(nd.who.heavy_set(0, sim.now)[0]) == set(GA)]
    assert ga, [(nd.id, nd.ctx) for nd in tr.nodes.values()]
    num = ga[0].targets["net.bytes_up"]
    ring_days = sorted(int(x) for x in num.ring[:, 0] if np.isfinite(x))
    lo, hi, _ = num.observed_range(ring_days[-1])
    if num.log:
        lo, hi = math.exp(lo), math.exp(hi)
    assert lo <= 520.0 + 1e-6, (lo, hi)
    sx = PN.SourceExtremes(K=3)
    for i in range(5):
        sx.note("10.0.0.1", "x", 1.0, 1, float(i))          # recurring
    for i in range(20):
        sx.note(f"10.50.0.{i}", "x", 2.0, 1, 100.0 + i)     # one-off leases
    assert "10.0.0.1" in sx.d


def test_extremes_of_an_attribute_before_it_is_a_target():
    """body.len became a target of the OA login node only after 192.168.1.21's
    643 B login of day 2 (pack O seed 1): the node never learned that value,
    and no record of it reached the 综合部 node. A leaf records the extremes
    of every numeric attribute P05 keeps (its roles), target or not yet."""
    sel0 = {"targets_sys": {0: ["http.method"]}, "split_cands": {0: [("net.src", 1)]},
            "roles": {"net.bytes_up": "split"}}
    sel1 = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: [("net.src", 1)]}, "roles": {}}
    sim = Sim(sel=sel0)

    def switch(sm):
        if sm.now >= MON + 4 * DAY:
            sm.sel = sel1
    sim.hooks.append(switch)

    def day(d, t0, rng):
        if not is_workday(t0):
            return []
        ev = []
        for i, ip in enumerate(GA):
            size = 520.0 if (d == 1 and i == 0) else rng.uniform(1024, 2048)
            ev.append(_login(t0 + (9 * 60 + rng.uniform(0, 21)) * 60, ip, size))
        for ip in POP:
            ev.append(_login(t0 + rng.uniform(13 * 60, 17 * 60) * 60, ip, rng.uniform(600, 1400)))
        return ev
    sim.add(daily(day, 16, seed=31))
    sim.run_until(MON + 16 * DAY)
    assert sim.splits(), "no split"
    tr = sim.tree()
    ga = [nd for nd in tr.nodes.values() if nd.parent is not None and nd.split is None
          and set(nd.who.heavy_set(0, sim.now)[0]) == set(GA)]
    assert ga
    num = ga[0].targets["net.bytes_up"]
    ring_days = sorted(int(x) for x in num.ring[:, 0] if np.isfinite(x))
    lo, hi, _ = num.observed_range(ring_days[-1])
    if num.log:
        lo, hi = math.exp(lo), math.exp(hi)
    assert lo <= 520.0 + 1e-6, (lo, hi)


def test_hold_prior_is_weak_when_the_records_cannot_tell_the_dispersion():
    """Records of one test each (the first days) say nothing about how much
    statements differ: the beta-binomial likelihood is flat in the prior's
    strength, and the grid's strongest prior (s = 32) won on floating-point
    ties - pack O seed 0 kept a Beta(11.2, 20.8) fitted on day 4-5, and its
    health-check statements stated 0.53 after 12 passed tests of 12. A weak
    hyperprior on the strength makes such a fit weak; with real evidence of
    homogeneity (test_empirical_bayes_prior_calibrates_few_test_statements)
    the prior stays informative."""
    rng = np.random.default_rng(1)
    for n in (8, 14, 20, 40):
        recs = [(float(rng.random() < 0.4), 1.0) for _ in range(n)]
        pr = PN.fit_hold_prior(recs)
        assert pr is not None and sum(pr) <= 2.0, (n, pr)
    p = (12.0 + 0.5 * 2) / (12.0 + 2.0)
    assert p > 0.9


def _clustered_hold(cover: float, seed: int = 5) -> float:
    """20 sources a day, one session of 10 page views each at the SAME arrival
    minute (a session's events share its time); the source's minute is inside
    the stated 0.9 window with probability `cover`. Returns p_hold after 60 days."""
    from app.engines.behavior.pattern_tree import PatternTreeEngine
    rng = np.random.default_rng(seed)
    nd = PN.Node(1, 0, 1, 0, (), MON)
    nd.state = "confirmed"
    nd.ref = {"t": MON, "hold": {"when": ("win", {0: ((480.0, 600.0),)}, 0.9)}}
    t = MON
    for d in range(60):
        for i in range(20):
            ip = f"10.0.{i}.1"
            minute = rng.uniform(480, 600) if rng.random() < cover else rng.uniform(700, 900)
            t0 = MON + d * DAY + i * 600.0
            for j in range(10):
                keys = [ip, f"10.0.{i}.0/24", "10.0.0.0/16", "grp:∅", "reg:∅"]
                PatternTreeEngine._hold_check(None, nd, lambda a: None, keys, 0, minute, t0 + j * 30.0, 1.0)
    return nd.p_hold(t0 + DAY)


def test_held_out_tests_count_a_sources_day_as_one_unit():
    """Evaluator round 4 (pnode.cluster_tol): the checks of one source's day are
    one unit of evidence - a session's page views share its arrival minute. A
    window that holds 0.9 of the SOURCES' arrivals failed ~1 test in 4 when the
    100 views of 10 sessions were judged as 100 independent checks (p_hold
    0.73; pack O seed 0, days 7-21: node statements stated 0.69 and held 0.91
    on the evaluator's independent held-out events). A test is now 50
    source-days judged at their cluster-robust error (2 sigma); a window
    holding 0.75 or 0.6 still fails its tests."""
    assert _clustered_hold(0.9) > 0.85              # 0.73 with event-level batches
    assert _clustered_hold(0.75) < 0.4              # a quarter of the sources outside: still fails
    assert _clustered_hold(0.6) < 0.2
