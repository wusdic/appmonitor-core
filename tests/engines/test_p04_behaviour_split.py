"""P04 split criterion: a split is paid for by the BEHAVIOUR it explains (action
fields, content, time), not by the context (who acts, the client stack), and
damped foreign sources never become part of a pattern's who (docs/lib3/
progressive.md §6.5.3, §6.9.2-§6.9.4; deviations M26-M32 of §16)."""
from __future__ import annotations

import numpy as np

from ptree_sim import DAY, MON, Sim, daily, is_workday

from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.lib import pselect as SEL

GA = {"192.168.1.21": "jack", "192.168.1.23": "rose", "10.168.7.121": "mike"}
POP = [f"192.168.3.{i}" for i in range(20, 32)]
LOGIN = {"http.route": "POST oa /login", "http.method": "POST"}


def _ev(ts, ip, size, **kw):
    return (ts, "oa", ip, dict(LOGIN, **{"net.bytes_up": float(size)}, **kw))


def _first_split(sim):
    sp = sim.splits()
    return sp[0][5] if sp else None


def _route_node(sim, route):
    tr = sim.tree()
    root = tr.nodes[tr.root] if tr is not None else None
    if root is None or root.split is None:
        return None
    for g, c in zip(root.split.groups, root.split.children):
        if any(route in str(v) for v in g):
            return c
    return None


# ------------------------------------------------------------- M26: @who
def test_source_identity_alone_never_pays_for_a_split():
    """A client attribute that is a function of the source (each IP one TCP
    window value, 4 values over 16 IPs) and explains nothing the sources DO
    (same sizes, same minutes for everyone): it predicts WHO comes, which is
    context. Before M26 the who target `@who` was coded for every non-who
    candidate, so the window attribute 'explained' the sources and split the
    login node within days."""
    ips = [f"192.168.5.{i}" for i in range(10, 26)]
    win = {ip: f"w{i % 4}" for i, ip in enumerate(ips)}
    sel = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: [("net.win", 0)]}, "roles": {}}
    sim = Sim(sel=sel)

    def day(d, t0, rng):
        if not is_workday(t0):
            return []
        return [_ev(t0 + rng.uniform(9 * 60, 17 * 60) * 60, ip, rng.uniform(600, 1400), **{"net.win": win[ip]})
                for ip in ips for _ in range(3)]
    sim.add(daily(day, 14, seed=3))
    sim.run_until(MON + 14 * DAY)
    assert not sim.splits(), [x[5] for x in sim.splits()]
    ss = sim.tree().nodes[sim.top()].split_stats
    assert ss is not None and ss.checks >= 5


# ----------------------------------------------------- M27: who first
def test_department_client_stack_yields_to_the_who_level():
    """The department uses its own client stack (a source property, P05's
    who_proxies): splitting on the stack or on the department's /24s separates
    the same events. The who level is chosen - the pattern is "综合部's IPs log
    in at 09:00-09:21", not "clients with window class lo" - and the stack
    never becomes the split (before M26/M27 the stack split first: it also
    'explained' the sources, and with 2 values it pays less prequential regret
    than the who level with 5)."""
    sel = {"targets_sys": {0: ["net.bytes_up"]},
           "split_cands": {0: [("net.win", 0), ("net.src", 1)]}, "roles": {},
           "who_proxies": ["net.win"]}
    sim = Sim(sel=sel)

    def day(d, t0, rng):
        if not is_workday(t0):
            return []
        ev = []
        for ip in GA:
            ev.append(_ev(t0 + (9 * 60 + rng.uniform(0, 21)) * 60, ip, rng.uniform(1024, 2048), **{"net.win": "lo"}))
        for ip in POP:
            ev.append(_ev(t0 + rng.uniform(13 * 60, 17 * 60) * 60, ip, rng.uniform(600, 1400), **{"net.win": "hi"}))
        return ev
    sim.add(daily(day, 15, seed=4))
    sim.run_until(MON + 15 * DAY)
    det = _first_split(sim)
    assert det is not None, "no split"
    assert det["attr"] == "net.src", det
    assert all(x[5].get("attr") != "net.win" for x in sim.splits())


def test_who_proxies_are_source_properties_not_bound_action_fields():
    """P05's source properties: a client stack shared by many sources and
    carried by every event is one; a login form's user name - a function of
    the source too, but present on one action only - is not, and neither is
    the day type of sources that only come on workdays."""
    from ptree_sim import Sim as _Sim
    sim = _Sim(p05=True)
    rng = np.random.default_rng(5)
    stack = {ip: ("chrome" if i % 2 else "edge") for i, ip in enumerate(POP)}
    # four shared accounts, three sources each: a function of the source shared by
    # several sources, like the stack - but a field of one action only
    ev = []
    for d in range(3):
        t0 = MON + d * DAY
        for i, ip in enumerate(POP):
            for _ in range(6):
                ev.append((t0 + rng.uniform(9, 17) * 3600, "oa", ip,
                           {"http.route": "GET oa /docs", "client.stack": stack[ip],
                            "net.dur_ms": float(rng.uniform(10, 90))}))
            ev.append((t0 + rng.uniform(9, 10) * 3600, "oa", ip,
                       dict(LOGIN, **{"client.stack": stack[ip], "body.kv.username": ("ann", "bert", "chris", "dieter")[i % 4],
                                      "net.dur_ms": float(rng.uniform(10, 90))})))
    sim.add(ev)
    sim.run_until(MON + 3 * DAY)
    pr = sim.p05.probes[("oa", EV.KIND_TXN)]
    from app.engines.behavior.lib import m_ptree as MP
    hier = MP.hierarchies(sim.st, "oa", sim.cfg)
    out = SEL.who_proxies(pr, sim.now, hier, ["client.stack", "body.kv.username", "ctx.daytype", "net.dur_ms"])
    assert "client.stack" in out
    assert "body.kv.username" not in out and "ctx.daytype" not in out and "net.dur_ms" not in out


# --------------------------------------------- M28: @when bin width
def test_login_minute_pays_for_the_department_split():
    """Departments that differ ONLY in their login minute inside one 4-hour
    block (综合部 09:00-09:21, the others 10:00-11:30; same sizes). The route
    node now chooses its @when width from its own arrivals (LEARN_MIN), so the
    minute pays for a who split within a week; before M28 it started learning
    from its first rows, the 4-hour fallback width (08:00-12:00, one bin) won
    and stayed for the 14-day episode, and no split was ever paid for."""
    sel = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: [("net.src", 1)]}, "roles": {}}
    sim = Sim(sel=sel)

    def day(d, t0, rng):
        # a health monitor around the clock (every 15 min, every day): the
        # attributes are typed (P02) before the login route appears, as on a
        # running system, and the tree's root mixes every hour of the day
        ev = [(t0 + k * 900.0 + 7.0, "oa", "192.168.9.9",
               {"http.route": "GET oa /health", "net.bytes_up": float(rng.uniform(300, 500))})
              for k in range(96)]
        if d < 1 or not is_workday(t0):
            return ev
        ev += [_ev(t0 + (9 * 60 + rng.uniform(0, 21)) * 60, ip, rng.uniform(600, 1400)) for ip in GA]
        ev += [_ev(t0 + rng.uniform(10 * 60, 11.5 * 60) * 60, ip, rng.uniform(600, 1400)) for ip in POP]
        return ev
    sim.add(daily(day, 11, seed=6))
    sim.run_until(MON + 11 * DAY)
    login = [x for x in sim.splits() if x[2] == _route_node(sim, "POST oa /login")]
    assert login and login[0][5]["attr"] == "net.src", [x[5] for x in sim.splits()]
    tr = sim.tree()
    nid = login[0][2]
    sp = tr.nodes[nid].split
    ga = {sp.child_for(f"{ip.rsplit('.', 1)[0]}.0/24") for ip in GA}
    pop = {sp.child_for(f"{ip.rsplit('.', 1)[0]}.0/24") for ip in POP}
    assert len(ga) == 1 and not (ga & pop), (sp.groups,)


def test_split_children_judge_the_when_width_on_their_parents_arrivals():
    """A split's children start with the evidence the split statistics held
    for them (M5) but with an empty arrival histogram; their first coder takes
    the @when width from the parent's arrivals (same route), not the 4-hour
    fallback judged on one row."""
    sel = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: [("net.src", 1)]}, "roles": {}}
    sim = Sim(sel=sel)

    def day(d, t0, rng):
        if not is_workday(t0):
            return []
        ev = [_ev(t0 + (9 * 60 + rng.uniform(0, 21)) * 60, ip, rng.uniform(1024, 2048)) for ip in GA]
        ev += [_ev(t0 + rng.uniform(8.5 * 60, 9.5 * 60) * 60, ip, rng.uniform(600, 1400)) for ip in POP]
        return ev
    sim.add(daily(day, 12, seed=7))
    sim.run_until(MON + 12 * DAY)
    assert sim.splits(), "no split"
    tr = sim.tree()
    nid = sim.splits()[0][2]
    kids = [tr.nodes[c] for c in tr.nodes[nid].split.all_children() if c in tr.nodes]
    coders = [k.meta.get("C") for k in kids if k.meta.get("C") is not None]
    assert coders, "children never learned"
    assert all(c.wdiv <= 60 for c in coders), [c.wdiv for c in coders]


# ------------------------------------------ M29: damped foreign sources
def _assign_hook(foreign):
    """P03 stand-in: rows of `foreign` are learned damped as a foreign source
    (damp 0.1, p_who 0.03) on their FIRST day only; later rows come undamped
    (P03 may stop flagging a source whose events made it look familiar)."""
    first = {}

    def hook(s):
        b = s.st.batch_at("oa", EV.EVT_BATCH, s.now)
        if b is None:
            return
        rows = []
        for i in range(b.n):
            ip = b.ip_of(i)
            day = int(b.ts[i] // DAY)
            if ip in foreign and first.setdefault(ip, day) == day:
                rows.append({"damp": 0.1, "p_who": 0.03})
            else:
                rows.append({"damp": 1.0, "p_who": 1.0})
        s.st.add_batch("oa", EV.PAT_ASSIGN, s.now, b.aligned(EV.cols_from_rows(b.n, rows)))
    return hook


def test_damped_foreign_source_never_enters_the_who_summary():
    """One approver reads the approval list a few times every workday; from
    day 6 a foreign source reads it once a day (low-and-slow, pack O's A9).
    P03 damps its first rows as foreign; whatever P03 does later, the source
    stays out of the node's who summary (mass, evidence, unseen-source
    estimate, heavy set), so the pattern is still "approver only" on day 20.
    Before M29 damped rows entered the who summary at 0.1 and the later
    undamped ones at full weight: the source reached the heavy set."""
    sel = {"targets_sys": {0: ["net.dur_ms"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)
    appr, a9 = "192.168.2.10", "192.168.3.33"
    sim.hooks.append(_assign_hook({a9}))

    def day(d, t0, rng):
        if not is_workday(t0):
            return []
        ev = [(t0 + rng.uniform(10 * 60, 11.5 * 60) * 60, "oa", appr,
               {"http.route": "GET oa /fin/approval/list", "net.dur_ms": float(rng.uniform(20, 60))})
              for _ in range(2)]
        if d >= 6:
            ev.append((t0 + 10.5 * 3600, "oa", a9,
                       {"http.route": "GET oa /fin/approval/list", "net.dur_ms": float(rng.uniform(20, 60))}))
        return ev
    sim.add(daily(day, 20, seed=8))
    sim.run_until(MON + 20 * DAY)
    nd = sim.tree().nodes[sim.top()]
    at = sim.now
    who0 = nd.who.levels[0]
    assert a9 not in who0, [(k, round(g, 2)) for k, _, g, _ in who0.items(at)]
    heavy, cov = nd.who.heavy_set(0, at)
    assert heavy == [appr] and cov > 0.99
    assert nd.who.is_suspect(a9, at)


def test_suspect_record_is_bounded_and_merges():
    w = PN.WhoSummary()
    for i in range(PN.SUS_MAX + 5):
        w.mark_suspect(f"10.0.0.{i}", 1000.0 + i)
    assert len(w.suspects(1100.0)) == PN.SUS_MAX
    assert not w.is_suspect("10.0.0.0", 1100.0)            # oldest forgotten first
    assert not w.is_suspect(f"10.0.0.{PN.SUS_MAX + 4}", 1100.0 + PN.SUS_KEEP_S + 1)
    v = PN.WhoSummary()
    v.merge(w)
    assert v.is_suspect(f"10.0.0.{PN.SUS_MAX + 4}", 1100.0)


# --------------------------------------- M30: calibrated statement confidence
def test_hold_record_is_calibrated_and_rises_with_evidence():
    """HoldRecord.p_hold is the posterior probability that a constraint's
    coverage on new data is >= nominal - eps. Under its own prior (coverage
    uniform) it is calibrated: among constraints given p in a bin, the share
    that truly holds matches p (ECE <= 0.05 over 4 000 simulated constraints
    with 5-200 held-out checks each). For a constraint that keeps holding it
    rises with the checks; for one that keeps failing it falls to ~0."""
    rng = np.random.default_rng(12)
    ps, ok = [], []
    for _ in range(4000):
        theta = rng.uniform()
        n = int(rng.integers(5, 200))
        hits = rng.random(n) < theta
        hr = PN.HoldRecord()
        for k, h in enumerate(hits):
            hr.add("band", bool(h), 0.9, MON + k * 600.0)
        ps.append(hr.p_hold(MON + n * 600.0))
        ok.append(theta >= 0.9 - PN.hold_eps(0.9))
    ps, ok = np.asarray(ps), np.asarray(ok, dtype=float)
    bins = np.minimum((ps * 10).astype(int), 9)
    ece = sum(abs(ps[bins == b].mean() - ok[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    assert ece <= 0.05, ece
    good, bad = PN.HoldRecord(), PN.HoldRecord()
    trace = []
    for k in range(400):
        t = MON + k * 3600.0
        good.add("who", True, 0.95, t)
        good.add("band", rng.random() < 0.93, 0.9, t)
        bad.add("band", rng.random() < 0.6, 0.9, t)
        if k % 50 == 49:
            trace.append(good.p_hold(t))
    assert all(b >= a - 0.02 for a, b in zip(trace, trace[1:])), trace
    assert trace[-1] > 0.95 and trace[0] < trace[-1]
    assert bad.p_hold(MON + 400 * 3600.0) < 0.01
    good.drop(["band"])
    assert set(good.counts(MON).keys()) == {"who"}


def test_node_confidence_is_prequential_and_grows_on_a_stable_pattern():
    """A stable daily pattern (one approver, 10:00-11:30, sizes 0.8-1.5 KB):
    once confirmed, every learned event is checked against the previous day's
    reference statement; the node's p_hold exists, grows with the days and
    ends high; the snapshot view (to_plain) carries it. Before M30 the node
    had no confidence of its own (the stated confidence was the smallest
    nominal coverage of its parts and fell with time on pack O). Its level is
    what the checks support: the stated 90 % band, a plug-in q05-q95 of a few
    dozen values, covers ~86 % of the next days' values, which is close to
    the 85 % a 90 % claim tolerates, so the confidence stays moderate."""
    sel = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)
    appr = "192.168.2.10"

    def day(d, t0, rng):
        if not is_workday(t0):
            return []
        return [(t0 + rng.uniform(10 * 60, 11.5 * 60) * 60, "oa", appr,
                 {"http.route": "POST oa /fin/approve", "net.bytes_up": float(rng.uniform(800, 1500))})
                for _ in range(4)]
    sim.add(daily(day, 28, seed=13))
    seen = []
    for d in range(28):
        sim.run_until(MON + (d + 1) * DAY)
        nd = sim.tree().nodes[sim.top()]
        p = nd.p_hold(sim.now)
        if p is not None:
            seen.append(p)
    assert len(seen) >= 8, seen
    assert np.median(seen[-5:]) > 10 * np.median(seen[:3]), seen     # rises with the evidence
    plain = sim.tree().nodes[sim.top()].to_plain()
    # what the node STATES is checked: its closed who and its arrival window
    # (no content fitter runs in this simulation, so no content constraint)
    assert plain["p_hold"] is not None and set(plain["hold"]) >= {"who", "when"}


def test_body_size_is_not_paid_for_by_its_own_fields():
    """The length of a body is the sum of its fields' lengths: a split on the
    body-size bin must not be paid for by predicting the length of its padding
    field (pack O: both children of the OA login node split on body.len, paid
    by body.kv.viewstate.len)."""
    assert SEL.same_source("body.len", "body.kv.viewstate.len")
    assert SEL.same_source("body.kv.viewstate.len", "body.len")
    assert SEL.same_source("body.len", "body.keys")
    assert not SEL.same_source("body.kv.username", "body.kv.password")
    assert not SEL.same_source("net.bytes_up", "body.kv.viewstate.len")


# --------------------------------------------- M31: bounded pair sketches
def test_pair_sketches_are_bounded_per_node_and_follow_the_requests():
    """P08 asks for binding pair counts (model.pwant); a node holds at most
    PAIRS_NODE_MAX of them, and the ones P08 no longer requests are dropped at
    the daily pass. Before M31 a node kept a sketch for every pair ever
    requested, so P-core memory grew with the number of attributes (PG4)."""
    from app.engines.behavior import pattern_tree as P4
    from app.engines.behavior.lib import m_ptree as MP
    from app.models.schema import SYSTEM_ENTITY
    sel = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)
    ys = [f"meta.f{i:03d}" for i in range(20)]

    def want(ys_):
        sim.st.put_model("oa", SYSTEM_ENTITY, MP.PWANT,
                         {"p08": {"pairs": [{"x": "net.src", "x_level": 0, "y": y} for y in ys_]}})
    want(ys)

    def day(d, t0, rng):
        return [_ev(t0 + rng.uniform(9, 17) * 3600, ip, rng.uniform(600, 1400),
                    **{y: f"v{(i + j) % 3}" for j, y in enumerate(ys)})
                for i, ip in enumerate(POP) for _ in range(2)]
    sim.add(daily(day, 4, seed=14))
    sim.run_until(MON + 2 * DAY)
    tr = sim.tree()
    assert max(len(nd.pairs) for nd in tr.nodes.values()) == P4.PAIRS_NODE_MAX
    want(ys[10:12])                                  # P08 now asks for two pairs only
    sim.run_until(MON + 4 * DAY)
    held = set().union(*[set(nd.pairs) for nd in tr.nodes.values()])
    assert held <= {("net.src", y) for y in ys[10:12]}, held


# ------------------------------------- M33: a split's children keep learning
def test_department_isolated_by_a_second_who_split():
    """A large pool (研发-like, two /24s, 09:30-11:00) is split off first; the
    `other` child (综合部 09:00-09:21 + 销售部 08:30-09:30) must still be able to
    split again at /24 and isolate 综合部's three addresses. Before M33 the
    child's who candidate was judged constant on the parent's copied top-8
    sources (all 销售部), dropped, and - the check cadence counting only
    while a candidate was tracked - never offered again; and a child
    relearned the per-/24 distributions from zero instead of starting from
    the parent's counts."""
    ga, sales = list(GA), POP
    dev = [f"10.50.{k}.{i}" for k in range(2) for i in range(10, 22)]
    sel = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: [("net.src", 1)]}, "roles": {}}
    sim = Sim(sel=sel)

    def day(d, t0, rng):
        if not is_workday(t0):
            return []
        ev = [_ev(t0 + (9 * 60 + rng.uniform(0, 21)) * 60, ip, rng.uniform(600, 1400)) for ip in ga]
        ev += [_ev(t0 + rng.uniform(8.5 * 60, 9.5 * 60) * 60, ip, rng.uniform(600, 1400)) for ip in sales]
        ev += [_ev(t0 + rng.uniform(9.5 * 60, 11 * 60) * 60, ip, rng.uniform(600, 1400))
               for ip in dev if rng.random() < 0.5]
        return ev
    sim.add(daily(day, 21, seed=21))
    sim.run_until(MON + 21 * DAY)
    from app.engines.behavior.lib import m_ptree as MP
    tr = sim.tree()
    hier = MP.hierarchies(sim.st, "oa", sim.cfg)

    def leaf(ip):
        return tr.route(lambda a: {"http.route": "POST oa /login", "net.src": ip}.get(a, EV.ABSENT), hier)[-1]
    ga_l = {leaf(ip) for ip in ga}
    assert len(ga_l) == 1, ga_l
    assert not ga_l & {leaf(ip) for ip in sales + dev}
    assert len(sim.splits()) >= 2


# --------------------------------------- M34: dropped attributes are registry-only
def test_dropped_attribute_keeps_registry_only_statistics():
    """§6.4: a `dropped` attribute is registry-only (presence, HLL). Its value
    summaries are released when P05 drops it and rebuilt when it gets a role
    again, so the registry no longer grows by a full summary per noise
    attribute (PG4 attribute axis)."""
    from app.engines.behavior.lib import pregistry as PR
    reg = PR.AttrRegistry("oa")
    rng = np.random.default_rng(15)
    names = [f"meta.f{i:03d}" for i in range(40)]
    for k in range(50):
        t = MON + k * 600.0
        for nm in names:
            reg.observe(nm, list(rng.normal(100, 30, 8)), t)
    full = reg.nbytes()
    for nm in names[:30]:
        reg.set_role(nm, "dropped")
    for k in range(50, 60):
        t = MON + k * 600.0
        for nm in names:
            reg.observe(nm, list(rng.normal(100, 30, 8)), t)
    rec = reg.records[names[0]]
    assert rec.num is None and rec.mom is None and rec.top.k <= PR.TOP_K_DROPPED
    assert rec.card.count() > 100                       # presence and distinct values are still kept
    assert reg.nbytes() < 0.6 * full, (reg.nbytes(), full)
    reg.set_role(names[0], "target")
    reg.observe(names[0], [1.0, 2.0, 3.0], MON + 61 * 600.0)
    assert reg.records[names[0]].top.k == PR.TOP_K and reg.records[names[0]].num is not None


# ------------------------------- M35: structural alarms are provisional
def test_structural_alarm_without_a_day_level_change_keeps_the_pattern_stated():
    """An event-level ADWIN alarm on a busy node opens a provisional change: the
    node stays confirmed / stable (a stated pattern) unless a whole normal day
    shows its mean loss above the pre-alarm level, and the alarm is cleared as
    false after T_persist + 2 normal days without that. Before M35 every alarm
    made the node `evolving` at once: pack O's 60-s health monitors were not a
    stated pattern on day 14 and day 21 of every seed."""
    from app.engines.behavior import pattern_tree as P4
    from app.engines.behavior.lib import m_ptree as MP
    sel = {"targets_sys": {0: ["net.dur_ms"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)

    def day(d, t0, rng):
        return [(t0 + k * 600.0 + 5.0, "oa", "192.168.9.9",
                 {"http.route": "GET oa /health", "net.dur_ms": float(rng.lognormal(3.0, 0.3))})
                for k in range(144)]
    sim.add(daily(day, 16, seed=16))
    sim.run_until(MON + 6 * DAY)
    tr = sim.tree()
    nd = tr.nodes[sim.top()]
    assert nd.state in ("confirmed", "stable"), nd.state
    m = MP.get_ptree(sim.st, "oa")
    lc = P4._LC(sim.p04, sim.st, "oa", sim.now, sim.cfg, 8 * 3600.0)
    lc.m, lc.aux = m, sim.p04.aux(m)
    sim.p04._structural_alarm(lc, tr, nd, sim.now)          # a (false) event-level alarm
    assert nd.state in ("confirmed", "stable") and nd.meta.get("drift") is not None
    for d in range(7, 16):
        sim.run_until(MON + d * DAY)
        assert nd.state in ("confirmed", "stable"), (d, nd.state)
    assert nd.meta.get("drift") is None                       # cleared as a false alarm


# ------------------------ M36: P05 split utility on behaviour (time of day)
def test_who_level_gets_the_split_role_when_groups_differ_only_in_time():
    """Opaque traffic (pack O's mail): each department uses its own client
    stack (a source property) and sends sizes nobody predicts; the departments
    differ only in WHEN they come. P05 measures a split candidate's utility on
    the behaviour - the time of day included, source properties excluded - so
    the address gets the split role. Before M36 the utility was measured on the
    system target list alone, which was the TCP-window class: net.src stayed a
    `shape` attribute and no who split was ever tracked."""
    sim = Sim(p05=True)
    rng = np.random.default_rng(17)
    # the client stack (TCP window class) is per source but shared across the
    # departments, so the department is told by its mail hours only
    groups = [([f"192.168.{g}.{i}" for i in range(20, 30)], h0, None) for g, h0 in ((1, 9.0), (2, 13.0), (3, 16.0))]
    ev = []
    for d in range(3):
        t0 = MON + d * DAY
        for ips, h0, win in groups:
            for ip in ips:
                for _ in range(4):
                    ev.append((t0 + (h0 + rng.uniform(0, 0.5)) * 3600, "mail", ip,
                               {"http.route": "TLS mail", "net.win": f"w{int(ip.rsplit('.', 1)[1]) % 2}",
                                "net.bytes_up": float(rng.lognormal(8, 1.0))}))
    sim.add(ev)
    sim.run_until(MON + 3 * DAY)
    from app.engines.behavior.lib import m_ptree as MP
    from app.models.schema import SYSTEM_ENTITY
    sel = sim.st.get_model("mail", SYSTEM_ENTITY, MP.ATTRSEL)
    assert sel["roles"].get("net.src") == "split", sel["roles"]
    # the department level (/24) is the one that explains the behaviour; before
    # M36 only /32 was offered, chosen for predicting each source's own window
    # class (a source property)
    assert ("net.src", 1) in [tuple(x) for x in sel["split_cands"][0]], sel["split_cands"]
