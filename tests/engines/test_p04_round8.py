"""P04 round 8 (docs/lib3/progressive.md §16.15.11): evidence units of waiting
route rows, untested-statement priors, per-day-type window tests, transport
alarms, facet records after a re-statement, compact per-key records.
(Confirmation within the tick the rule is met was measured and withdrawn: see
PatternTree._confirm.)"""
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


def test_a_restated_constraint_takes_its_failures_out_of_the_facet_tests_too():
    """A test failed only by the who of a statement that is then re-stated
    (M45 drop) is a pass of what remains stated - for the whole statement
    (round 7) and for each facet record: the '~when' record (the statement
    without its windows, what a part stating its own windows composes) kept
    the failure, and the finance login / home parts of O-real seed 0 stated
    (0 + a) / (1 + a + b) = 0.30 on days 17-35 and held 0.92-0.96."""
    from app.engines.behavior.lib import pnode as PN
    for lite in (False, True):
        hr = PN.HoldRecord(lite=lite)
        for d in range(7):
            for i in range(8):
                ts = MON + d * DAY + 9 * 3600 + i * 600
                hr.add("who", i < 4, 0.99, ts, 1.0, (f"10.0.0.{i}", d))      # half the checks outside
                hr.add("body.len", True, 0.9, ts, 1.0, (f"10.0.0.{i}", d))
        hr._close()
        ps, n = hr.tests()
        assert ps == 0.0 and n >= 1.0, (ps, n)
        ft = hr.facet_tests()
        assert ft["~when"] == (0.0, n), ft
        hr.drop(["who"])
        assert abs(hr.tests()[0] - n) < 1e-9, hr.tests()
        ft = hr.facet_tests()
        assert abs(ft["~when"][0] - n) < 1e-9, (lite, ft)
        if not lite:
            for fac in ("who", "~net", "content"):
                assert abs(ft[fac][0] - ft[fac][1]) < 1e-9, (fac, ft)


def test_observed_extremes_are_compact_records():
    """Per-source observed extremes (P04 sext / sext24, and vext per candidate
    value) are one record per (key, attribute): a flat float array of
    [lo, day, hi, day], not a list of boxed numbers (~190 bytes; with PG4's
    340 attributes these records were 4.5 of the trees' 35 MB). Same values."""
    from app.eval.pscale import deep_sizeof
    from app.engines.behavior.lib import pnode as PN
    sx = PN.SourceExtremes(K=64)
    attrs = [f"meta.f{i:03d}" for i in range(20)]
    for k in range(50):
        for d in range(3):
            sx.note_many(f"10.0.0.{k}", [(a, 100.0 * k + i + d) for i, a in enumerate(attrs)], 20000 + d,
                         1.7e9 + d * 86400.0)
    per = (deep_sizeof(sx) - deep_sizeof(PN.SourceExtremes(K=64))) / (50 * 20)
    assert per <= 100.0, per
    u = sx.union(lambda rep: rep in ("10.0.0.3", "10.0.0.7"))
    lo, dlo, hi, dhi = u["meta.f002"]
    assert (lo, int(dlo), hi, int(dhi)) == (302.0, 20000, 704.0, 20002)
    m = sx.subset(lambda rep: rep == "10.0.0.3").merge(sx.subset(lambda rep: rep == "10.0.0.7"))
    assert list(m.union(lambda rep: True)["meta.f002"]) == [302.0, 20000.0, 704.0, 20002.0]


def test_held_out_records_are_compact_and_keep_their_cluster_positions():
    """A held-out record's per-constraint counts are float arrays, and a
    constraint's per-cluster checks keep no ordinal map while the constraint
    has seen the batch's clusters in order (most constraints are checked on
    every event): ~25 % less memory per tracked group record, the same
    (n, hits) per cluster in the same order as the mapped form."""
    import random
    from app.eval.pscale import deep_sizeof
    from app.engines.behavior.lib import pnode as PN
    hr = PN.HoldRecord(lite=True)
    keys = [f"meta.f{i:03d}" for i in range(30)]
    for d in range(3):
        for u in range(15):
            for k in keys:
                hr.add(k, True, 0.9, 1.7e9 + d * 86400 + u * 600, 1.0, (f"10.0.0.{u}", d))
    assert deep_sizeof(hr) / len(hr.b) <= 1800.0, deep_sizeof(hr) / len(hr.b)

    def ref_add(st, g, w, hit):                       # the mapped form of round 7
        pos, n, h = st
        while len(pos) <= g:
            pos.append(-1)
        if pos[g] < 0:
            pos[g] = len(n)
            n.append(0.0)
            h.append(0.0)
        n[pos[g]] += w
        if hit:
            h[pos[g]] += w
    rng = random.Random(5)
    for trial in range(200):
        kc, st = PN._KC(), ([], [], [])
        g_max = 0
        for _ in range(rng.randint(1, 60)):
            g = rng.choice([g_max, g_max, max(0, g_max - rng.randint(0, 3)), g_max + rng.randint(1, 3)])
            g_max = max(g_max, g + 1)
            w, hit = rng.choice([1.0, 0.5, 0.25]), rng.random() < 0.8
            kc.add(g, w, hit)
            ref_add(st, g, w, hit)
        assert list(kc.values()) == list(zip(st[1], st[2])), trial
