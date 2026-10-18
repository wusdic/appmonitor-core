"""P04 round 8: confirmation as soon as the rule is met (docs/lib3/progressive.md
§16.15.11 item 1, GA.oa.report)."""
from __future__ import annotations

from ptree_sim import DAY, MON, Sim, daily

from app.engines.behavior import pattern_tree as P4


def _org(d, t0, rng):
    """A busy action (page views every 10 min) and a daily report made twice
    at 17:00, every day."""
    ev = []
    for k in range(144):
        ev.append((t0 + k * 600 + 11, "oa", f"192.168.3.{20 + k % 6}",
                   {"http.route": "GET oa /docs", "net.bytes_down": float(rng.uniform(5000, 9000))}))
    for j in range(2):
        ev.append((t0 + 17 * 3600 + 300 * j + 7, "oa", f"192.168.1.{21 + 2 * j}",
                   {"http.route": "POST oa /report/generate", "net.bytes_up": float(rng.uniform(2e4, 6e4))}))
    return ev


def _report(sim):
    tr = sim.tree()
    for nd in tr.nodes.values():
        if any("/report/generate" in str(v) for a, l, vals, n in nd.ctx for v in vals if not n):
            return nd
    return None


def test_a_pattern_is_confirmed_the_day_its_evidence_meets_the_rule():
    """The report's 20th observation comes at 17:05 of day 10: it is confirmed
    (and referenced, so its held-out record starts) that evening, not at the
    first tick after midnight with its reference at 04:00 of day 11 (round 7:
    a pattern eligible on Friday was stated on Monday)."""
    sim = Sim()
    sim.add(daily(_org, 11, seed=4))
    sim.run_until(MON + 9 * DAY + 17 * 3600)
    nd = _report(sim)
    assert nd is not None and nd.state == "candidate" and nd.n_obs() < P4.N_CONF, (nd and nd.state, nd and nd.n_obs())
    sim.run_until(MON + 9 * DAY + 20 * 3600)
    nd = _report(sim)
    assert nd.n_obs() >= P4.N_CONF
    assert nd.state == "confirmed", (nd.state, nd.n_obs(), nd.n_days())
    assert isinstance(nd.ref, dict) and nd.ref.get("t", 0) <= MON + 9 * DAY + 20 * 3600
    assert MON + 9 * DAY + 17 * 3600 <= nd.meta["confirmed_at"] < MON + 10 * DAY


def test_the_rule_still_needs_its_dates():
    """Twenty observations on fewer than CONF_DATES dates do not confirm."""
    def burst(d, t0, rng):
        ev = _org(d, t0, rng)
        if d == 0:
            ev += [(t0 + 10 * 3600 + 60 * j, "oa", "192.168.1.22",
                    {"http.route": "POST oa /upload", "net.bytes_up": 900.0}) for j in range(30)]
        if d == 1:
            ev += [(t0 + 10 * 3600 + 60 * j, "oa", "192.168.1.22",
                    {"http.route": "POST oa /upload", "net.bytes_up": 900.0}) for j in range(3)]
        return ev
    sim = Sim()
    sim.add(daily(burst, 3, seed=5))
    sim.run_until(MON + 2 * DAY + 12 * 3600)
    tr = sim.tree()
    up = [nd for nd in tr.nodes.values()
          if any("/upload" in str(v) for a, l, vals, n in nd.ctx for v in vals if not n)]
    assert all(nd.state == "candidate" for nd in up), [(nd.state, nd.n_obs(), nd.n_days()) for nd in up]


def test_a_waiting_route_row_counts_as_one_observation_of_its_route():
    """Each member opens the report form at 17:00 and generates the report a
    minute later. While the generate route waits for its node (the route
    partition needs rows on two dates), its rows wait at the root, where the
    form row of the same source a minute earlier made them half a burst unit.
    Replayed into the generate node they count one observation each (round 7:
    half, so the node trailed its events and confirmed days late)."""
    def org(d, t0, rng):
        ev = []
        for k in range(144):
            ev.append((t0 + k * 600 + 11, "oa", f"192.168.3.{20 + k % 6}",
                       {"http.route": "GET oa /docs", "net.bytes_down": float(rng.uniform(5000, 9000))}))
        for j, ip in enumerate(("192.168.1.23", "10.168.7.121")):
            ev.append((t0 + 17 * 3600 + 15 * j + 3, "oa", ip, {"http.route": "GET oa /report/form"}))
            ev.append((t0 + 17 * 3600 + 15 * j + 63, "oa", ip,
                       {"http.route": "POST oa /report/generate", "net.bytes_up": float(rng.uniform(2e4, 6e4))}))
        return ev
    sim = Sim()
    sim.add(daily(org, 3, seed=6))
    sim.run_until(MON + 2 * DAY + 20 * 3600)
    tr = sim.tree()
    gen = [nd for nd in tr.nodes.values()
           if any("/report/generate" in str(v) for a, l, vals, n in nd.ctx for v in vals if not n)]
    assert len(gen) == 1
    assert gen[0].n_obs() == 6.0, gen[0].n_obs()


def test_an_untested_statement_states_the_first_test_pass_rate_of_its_kind():
    """An untested statement's confidence is the rate at which statements
    pass their FIRST held-out test, not the beta-binomial mean of whole
    records: those are dominated by the long records of statements that kept
    holding (a failing one is re-stated and starts again with a short
    record). Here half of the statements failed their first test; the
    survivors passed 20 of 20. Round 7 stated ~0.9 for the next new
    statement (O-real seed 0: untested statements stated 0.95, held 0.80)."""
    from app.engines.behavior.lib import pnode as PN
    from app.engines.behavior.lib import ptree as PT
    from app.engines.behavior.pattern_tree import PatternTreeEngine

    def route(i):
        return (("http.route", 0, frozenset({f"GET oa.corp.local /r{i}"}), False),)
    eng = PatternTreeEngine()
    m = PT.PTreeModel("oa")
    tr = m.tree(0, MON)
    cons = {"who": (0, frozenset({"1.1.1.1"}), 0.95)}
    for i in range(16):
        nd = PN.Node(100 + i, tr.root, 1, 0, route(i), MON)
        nd.state = "confirmed"
        nd.ref = {"hold": dict(cons)}
        hr = nd.meta["hold"] = PN.HoldRecord()
        if i % 2:
            hr.T, hr.ti, hr.fl = [20.0, 20.0], 20, []             # held from its first test on
        else:
            hr.T, hr.ti, hr.fl = [0.0, 1.0], 1, [[0, ["when"]]]   # failed its first (and only) test
        tr.nodes[nd.id] = nd
    fresh = PN.Node(99, tr.root, 1, 0, route(99), MON)
    fresh.state = "confirmed"
    fresh.ref = {"hold": dict(cons)}
    tr.nodes[fresh.id] = fresh
    eng._hold_priors(m, 1, MON)
    eng._hold_priors(m, 2, MON + DAY)
    p = fresh.p_hold(MON + DAY)
    pooled = eng._hp["prior"]["*"]
    assert pooled[0] / (pooled[0] + pooled[1]) > 0.6          # the whole-record fit
    assert p is not None and 0.4 <= p <= 0.6, (p, fresh.meta.get("hold_prior"))
    # a tested statement keeps the whole-record prior under its own tests
    t = tr.nodes[101]
    assert t.meta["hold_prior"] == tuple(round(float(x), 4) for x in pooled) or \
        t.meta["hold_prior"] == tuple(round(float(x), 4) for x in eng._hp["prior"][PN.hold_kind(1)])


def test_the_non_workday_windows_are_a_constraint_of_their_own():
    """A statement with right workday windows and a wrong weekend window: its
    weekend checks are a minority of the node's and, pooled with the workday
    ones under one 'when' key, never failed a test (O-real seed 0: portal
    GET / stated 0.98 and held 0.72 against nominal 0.89 on non-workdays).
    Judged per day type, as a reader judges them, its tests fail."""
    from app.engines.behavior.lib import pnode as PN
    from app.engines.behavior.pattern_tree import PatternTreeEngine
    nd = PN.Node(5, 0, 1, 0, (("http.route", 0, frozenset({"GET p /"}), False),), MON)
    nd.state = "stable"
    nd.ref = {"t": MON, "sn": 1, "hold": {"when": ("win", {0: ((540.0, 1020.0),), 1: ((600.0, 660.0),)}, 0.9)}}
    for d in range(7):
        wd = d < 5
        for i in range(40 if wd else 10):
            ts = MON + d * DAY + 9 * 3600 + i * 600
            minute = 600.0 if wd else (630.0 if i % 2 == 0 else 900.0)   # half the weekend outside
            ip = f"10.0.0.{i % 20}"
            PatternTreeEngine._hold_check(None, nd, lambda a: None, (ip,), 0 if wd else 1, minute, ts, 1.0,
                                          1.0, d)
    hr = nd.meta["hold"]
    hr._close()                                          # the last (weekend) test
    ps, n = hr.tests()
    assert n >= 2 and ps < n, (ps, n, hr.failures(), hr.fl)


def test_a_decisively_failing_test_is_decided_before_its_batch_fills():
    """A quiet node's batch fills in a week. When two of its four members are
    readdressed, its who fails on half of the checks: the open test counts
    as failed once that is beyond a 2.5-sigma cluster-robust margin on >= 10
    units (provisionally, until the batch closes and is judged as usual), not
    only after the week (round 7: the statement kept its confidence
    meanwhile; O-real finance /fin/ledger stated 0.97 with its open batch's
    who at 0.61)."""
    from app.engines.behavior.lib import pnode as PN
    hr = PN.HoldRecord()
    for d in range(6):
        for i, ip in enumerate(("10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4")):
            for k in range(4):
                ts = MON + d * DAY + 9 * 3600 + i * 600 + k * 60
                hit = not (d >= 1 and ip in ("10.0.0.3", "10.0.0.4"))     # readdressed from day 1 on
                hr.add("who", hit, 0.99, ts, 1.0, (ip, d))
    assert hr.ti == 0                                    # the batch is still open (a week)
    assert hr.tests() == (0.0, 1.0), hr.tests()
    assert hr.first_test() is False
    assert abs(hr.p_hold(MON + 6 * DAY, (9.0, 1.0)) - 9.0 / 11.0) < 1e-9      # prior 0.9, one failed test
    # the week closes the batch: one failed test, judged by the usual rule
    hr.add("who", True, 0.99, MON + 8 * DAY, 1.0, ("10.0.0.1", 8))
    assert hr.ti == 1 and hr.tests() == (0.0, 1.0) and hr.fl[0][1] == ["who"]


def test_a_holding_statement_is_not_failed_early():
    """The early decision needs a decisive margin: a who that holds at its
    nominal share (one check in fifty from an unlisted source) passes."""
    from app.engines.behavior.lib import pnode as PN
    hr = PN.HoldRecord()
    j = 0
    for d in range(14):
        for i in range(10):
            for k in range(5):
                j += 1
                hr.add("who", j % 50 != 0, 0.97, MON + d * DAY + 9 * 3600 + i * 600 + k * 60, 1.0,
                       (f"10.0.0.{i}", d))
    ps, n = hr.tests()
    assert hr.ti >= 2 and ps == n and not hr.fl, (ps, n, hr.fl)


def _shift_run(attr):
    """A monitor-like action, 40 rows each workday; from day 14 its `attr`
    doubles. Returns the node's (state, evolving keys) per day end."""
    from ptree_sim import is_workday

    def day(d, t0, rng):
        if not is_workday(t0):
            return []
        f = 2.0 if d >= 14 else 1.0
        return [(t0 + 9 * 3600 + k * 400.0, "crm", f"192.168.111.{1 + k % 3}",
                 {"http.route": "GET crm /crm/customer/{num}", attr: float(rng.uniform(900, 1100)) * f})
                for k in range(40)]
    sel = {"targets_sys": {0: [attr]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)
    sim.add(daily(day, 21, seed=9))
    sim.run_until(MON + 10 * DAY)
    out = []
    while sim.now < MON + 21 * DAY:
        sim.tick()
        nd = sim.tree("crm").nodes[sim.top("crm")]
        out.append((round((sim.now - MON) / DAY, 2), nd.state, sorted((nd.meta.get("evolving") or {}).keys())))
    return out


def test_a_shift_of_a_transport_measure_does_not_withdraw_the_pattern():
    """A daily-mean alarm on a transport measure (hold facet 'net': the
    connection's duration) marks that attribute as under change; the node -
    who comes, when, to do what - keeps its state (round 7: `evolving`, the
    statement withdrawn: pack O seed 0's monitor on portal /health, days
    16-21). A shift of the request's own content still makes it evolving."""
    net = _shift_run("net.dur_ms")
    assert any("net.dur_ms" in ev for _, _, ev in net), net            # the alarm is raised ...
    assert all(st in ("confirmed", "stable") for _, st, _ in net), net  # ... the pattern stays stated
    body = _shift_run("body.len")
    assert any(st == "evolving" for _, st, _ in body), body
