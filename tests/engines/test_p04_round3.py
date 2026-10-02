"""P04 round 3 (docs/lib3/progressive.md §16.11): split-inherited statistics,
evidence-based suspicion, calibrated statement confidence."""
from __future__ import annotations

import math

import numpy as np

from ptree_sim import DAY, MON, Sim, daily, is_workday

from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import pnode as PN

GA = ["192.168.1.21", "192.168.1.23", "192.168.1.25"]
POP = [f"192.168.3.{i}" for i in range(20, 32)]
LOGIN = {"http.route": "POST oa /login", "http.method": "POST"}


def _login(ts, ip, size, **kw):
    return (ts, "oa", ip, dict(LOGIN, **{"net.bytes_up": float(size)}, **kw))


def _split_children(sim):
    tr = sim.tree()
    out = []
    for nd in tr.nodes.values():
        if nd.split is not None and nd.split.attr == "net.src":
            out.append(nd)
    return out


# ------------------------------------------------- split-inherited statistics
def test_split_child_is_born_with_its_own_history():
    """综合部 (three addresses) logs in at 09:00-09:21 with 1-2 KB forms, the
    sales floor at 13:00-17:00 with 0.6-1.4 KB forms; on two days one
    综合部 login is a short 520 B form (days 3-4: the login
    route node exists and learns from day 2). The login node is split on the source
    /24 after some days of learning. The 综合部 child must state the range of
    ITS logins, the early 520 B ones included, over the days before the split
    as well. Before round 3 a split child started with empty numeric summaries
    (only categorical coder targets were seeded): its observed range began at
    the split, so the requirement's '100 % in 0.5-3 KB' could not be stated
    for the 综合部 login node created on day 10-15 (§16.10.5)."""
    sel = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: [("net.src", 1)]}, "roles": {}}
    sim = Sim(sel=sel)

    def day(d, t0, rng):
        if not is_workday(t0):
            return []
        ev = []
        for i, ip in enumerate(GA):
            size = 520.0 if (d in (3, 4) and i == 0) else rng.uniform(1024, 2048)
            ev.append(_login(t0 + (9 * 60 + rng.uniform(0, 21)) * 60, ip, size))
        for ip in POP:
            ev.append(_login(t0 + rng.uniform(13 * 60, 17 * 60) * 60, ip, rng.uniform(600, 1400)))
        return ev
    sim.add(daily(day, 14, seed=31))
    sim.run_until(MON + 14 * DAY)
    sp = sim.splits()
    assert sp, "no split"
    t_split = min(x[0] for x in sp)
    d_split = int((t_split - MON) // DAY)
    assert d_split >= 5, d_split                         # the 520 B logins are pre-split history
    tr = sim.tree()
    ga = None
    for nd in tr.nodes.values():
        if nd.parent is None or nd.split is not None:
            continue
        heavy, _ = nd.who.heavy_set(0, sim.now)
        if set(heavy) == set(GA):
            ga = nd
    assert ga is not None, [(n.id, n.who.heavy_set(0, sim.now)[0]) for n in tr.nodes.values()]
    num = ga.targets.get("net.bytes_up")
    assert isinstance(num, PN.NumSummary)
    ring_days = sorted(int(x) for x in num.ring[:, 0] if np.isfinite(x))
    lo, hi, n_rng = num.observed_range(ring_days[-1])
    if num.log:
        lo, hi = math.exp(lo), math.exp(hi)
    assert lo <= 520.0 + 1e-6, (lo, hi, n_rng)          # the child's own pre-split extreme
    assert hi <= 2048.0 + 1e-6                           # and no sales row
    split_day = ring_days[-1] - (int((sim.now - t_split) // DAY))
    assert ring_days[0] < split_day - 1, (ring_days, split_day)
    # the routed rows are the child's own reservoir (a later split inherits again)
    assert ga.meta.get("rres") is not None and len(ga.meta["rres"]) > 0
    assert all(r[1] in GA for r in ga.meta["rres"].rows())


def test_split_rows_bounded_and_drop_long_text():
    r = PN.SplitRows(R=16, seed=1)
    for i in range(100):
        r.offer((float(i), "1.1.1.1", 0, 0, 0.0, 1.0, 1.0, True, False, {"a": float(i)}, {}), 1.0, float(i))
    assert len(r) == 16
    assert [x[0] for x in r.rows()] == sorted(x[0] for x in r.rows())
    assert not PN.keep_value("x" * (PN.SPLIT_TEXT_MAX + 1)) and PN.keep_value("jack")


# ------------------------------------------------------ evidence-based suspicion
def _damp_hook(rule):
    """P03 stand-in: rule(ip, day index) -> pat.assign row."""
    def hook(s):
        bt = s.st.batch_at("oa", EV.EVT_BATCH, s.now)
        if bt is None:
            return
        rows = [rule(bt.ip_of(i), int((bt.ts[i] - MON) // DAY)) for i in range(bt.n)]
        s.st.add_batch("oa", EV.PAT_ASSIGN, s.now, bt.aligned(EV.cols_from_rows(bt.n, rows)))
    return hook


CLEAN = {"damp": 1.0, "p_who": 1.0, "p_when": 1.0, "p_content": 1.0}
FOREIGN = {"damp": 0.1, "p_who": 0.03, "p_when": 0.5, "p_content": 1.0}


def test_member_damped_as_foreign_once_is_cleared_by_its_colleagues():
    """综合部's three addresses log in every workday; on day 7 P03 damps
    192.168.1.21's login as FOREIGN (the who is why: a light source of the
    node's narrow who). Its colleagues (same /24) keep using the node, so its
    later rows are member evidence and the sequential test clears it: on day
    16 the node's who names all three addresses again. Before round 3 the
    suspect flag was sticky while the source kept sending rows: .21 never
    re-entered the who (§16.10.9 P04 open issue). A foreign address with no
    colleague at the node and damped once (A9-like, see M29's test) stays
    suspect whatever P03 does later."""
    sel = {"targets_sys": {0: ["net.dur_ms"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)
    a, f = GA[0], "192.168.7.33"

    def rule(ip, d):
        if (ip == a and d == 7) or (ip == f and d == 8):
            return dict(FOREIGN)
        return dict(CLEAN)
    sim.hooks.append(_damp_hook(rule))

    def day(d, t0, rng):
        if not is_workday(t0):
            return []
        ev = [(t0 + (9 * 60 + rng.uniform(0, 21)) * 60, "oa", ip,
               {"http.route": "POST oa /login", "net.dur_ms": float(rng.uniform(20, 60))}) for ip in GA]
        if d >= 8:
            ev.append((t0 + 9.2 * 3600, "oa", f, {"http.route": "POST oa /login", "net.dur_ms": 30.0}))
        return ev
    sim.add(daily(day, 9, seed=5))
    sim.run_until(MON + 9 * DAY)
    nd = sim.tree().nodes[sim.top()]
    assert nd.who.is_suspect(a, sim.now)                # damped as foreign on day 7 (Wed)
    sim.add(daily(day, 17, seed=5)[len([e for e in daily(day, 9, seed=5)]):])
    sim.run_until(MON + 17 * DAY)
    nd = sim.tree().nodes[sim.top()]
    at = sim.now
    assert not nd.who.is_suspect(a, at)
    heavy, _ = nd.who.heavy_set(0, at)
    assert a in heavy, heavy
    assert nd.who.is_suspect(f, at) and f not in heavy


def test_suspicion_llr_needs_colleagues_not_repetition():
    w = PN.WhoSummary()
    keys = ["10.0.0.5", "10.0.0.0/24", "10.0.0.0/16", None, None]
    w.mark_suspect("10.0.0.5", 0.0)
    for i in range(50):                                 # repetition alone: still suspect
        assert not w.observe_suspect("10.0.0.5", 10.0 + i, w.support(keys, "10.0.0.5", 10.0 + i))
    assert w.is_suspect("10.0.0.5", 100.0)
    w.update(["10.0.0.7", "10.0.0.0/24", "10.0.0.0/16", None, None], "10.0.0.7", 100.0, 5.0, 5.0)
    assert w.support(keys, "10.0.0.5", 101.0)
    assert not w.observe_suspect("10.0.0.5", 101.0, True)   # 4.64 - 4.64 = 0 > -2
    assert w.observe_suspect("10.0.0.5", 102.0, True)       # cleared
    assert not w.is_suspect("10.0.0.5", 103.0)


# ------------------------------------------------- held-out checks of shaped values
def test_hold_check_matches_shape_only_values_by_instance():
    """Secrets and long values reach P04 as their level-1 shape (value policy,
    phier.Shaped: a password 'a1b2c3d4e5' arrives as 'L1 D1 L1 ...'). The
    held-out check of the statement's grammar must test a concrete instance of
    the shape, as P03 / P07 do. Before round 3 the shape TEXT was matched
    against the grammar: every password / viewstate check failed (pack O, OA
    /login: password#grammar 0 of 289 checks), so every login statement failed
    every held-out test and stated a confidence near 0."""
    from app.engines.behavior.lib.phier import Shaped, shape
    from app.engines.behavior.pattern_tree import PatternTreeEngine

    class _N:
        meta = {}
        ref = {"t": 1.0, "hold": {"body.kv.password#grammar": ("rx", "[A-Za-z0-9]{7,30}", 0.99)}}
    nd = _N()
    v = Shaped(shape("a1b2c3d4e5"))
    assert str(v) != "a1b2c3d4e5"
    for i in range(5):
        PatternTreeEngine._hold_check(None, nd, lambda a: v, [None] * 5, 0, 600.0, 100.0 + i, 1.0)
    n, h, _ = nd.meta["hold"].counts(105.0)["body.kv.password#grammar"]
    assert n > 0 and h == n


# ------------------------------------------- calibrated statement confidence (M47)
def _sim_statements(rng, n_stmt, a_true, b_true, n_tests):
    th = rng.beta(a_true, b_true, n_stmt)
    k = rng.integers(1, n_tests + 1, n_stmt)
    past = np.array([rng.binomial(int(n), p) for n, p in zip(k, th)], dtype=float)
    nxt = (rng.random(n_stmt) < th).astype(float)          # the held-out next test
    return th, k.astype(float), past, nxt


def _ece(conf, obs, bins=10):
    conf, obs = np.asarray(conf), np.asarray(obs)
    idx = np.clip((conf * bins).astype(int), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(conf[m].mean() - obs[m].mean())
    return e


def test_empirical_bayes_prior_calibrates_few_test_statements():
    """Most statements have one to three held-out tests by day 14. With the
    uniform Beta(1, 1) prior they state 1/3 - 2/3 whatever statements of their
    kind do; when 80 % of such statements hold, that is under-confident. The
    pooled (empirical-Bayes) prior learns the kind's hold rate from the other
    statements: on HELD-OUT next tests its reliability error is far smaller."""
    rng = np.random.default_rng(7)
    eb, uni = [], []
    for rep in range(20):
        _, n, past, nxt = _sim_statements(rng, 60, 8.0, 2.0, 3)
        pr = PN.fit_hold_prior(list(zip(past, n)))
        assert pr is not None
        m_fit = pr[0] / (pr[0] + pr[1])
        assert 0.65 <= m_fit <= 0.95, pr
        p_eb = (past + pr[0]) / (n + pr[0] + pr[1])
        p_uni = (past + 1.0) / (n + 2.0)
        eb.append(_ece(p_eb, nxt))
        uni.append(_ece(p_uni, nxt))
    assert np.mean(eb) < 0.6 * np.mean(uni), (np.mean(eb), np.mean(uni))
    assert np.mean(eb) < 0.12


def test_hold_prior_needs_enough_statements_and_is_monotone_in_evidence():
    assert PN.fit_hold_prior([(1.0, 1.0)] * (PN.HOLD_PRIOR_MIN_NODES - 1)) is None
    hr = PN.HoldRecord()
    pr = (3.0, 2.0)
    ps = []
    t = 0.0
    for i in range(12):                                   # a stable statement: every test passes
        for j in range(int(PN.HOLD_BATCH_N) + 1):
            t += 60.0
            hr.add("who", True, 0.95, t)
        ps.append(hr.p_hold(t + 1.0, pr))
    ps = [p for p in ps if p == p]
    assert all(b >= a - 1e-12 for a, b in zip(ps, ps[1:])), ps
    assert ps[-1] > pr[0] / sum(pr)
    assert math.isnan(PN.HoldRecord().p_hold(0.0))
    assert PN.HoldRecord().p_hold(0.0, pr) == 0.6          # no test yet: the kind's mean


def test_node_states_its_kind_prior_before_its_first_test():
    nd = PN.Node(1, 0, 1, 0, (), 0.0)
    assert nd.p_hold(10.0) is None
    nd.meta["hold_prior"] = (4.0, 1.0)
    assert nd.p_hold(10.0) == 0.8


# ------------------------------------------------ P05: structural key sets as targets
def test_form_key_set_is_tracked_at_its_action_node():
    """A CRM-like system: customer pages (no body) and one form action, POST
    /visit, whose key set is always {customer, note, viewstate}. The key set is
    constant wherever present, so it carried no predictive gain (dropped /
    redundant with the method) and was skipped at the node as a constant: P04
    never tracked it at the visit node and P07 never stated the visit form's
    required keys (pack O: SALES.crm#1, PUB.portal.visit#1 content misses).
    A key set fixed by the action is the action's structure: P05 rates it a
    target and the action node tracks it (a SetSummary P07 fits)."""
    keys = frozenset({"customer", "note", "viewstate"})

    def crm(d, t0, rng):
        if not is_workday(t0):
            return []
        ev = []
        for i in range(12):
            ip = f"192.168.3.{20 + i}"
            for _ in range(8):
                ev.append((t0 + rng.uniform(9, 17) * 3600, "crm", ip,
                           {"http.route": "GET crm /customer/{num}", "http.method": "GET",
                            "net.bytes_down": float(rng.uniform(4000, 9000))}))
            for _ in range(2):
                ev.append((t0 + rng.uniform(10, 16) * 3600, "crm", ip,
                           {"http.route": "POST crm /visit", "http.method": "POST", "body.keys": keys,
                            "body.len": float(rng.uniform(1200, 2400)),
                            "net.bytes_down": float(rng.uniform(300, 600))}))
        return ev
    sim = Sim(p05=True)
    sim.add(daily(crm, 5, seed=11))
    sim.run_until(MON + 5 * DAY)
    from app.models.schema import SYSTEM_ENTITY
    from app.engines.behavior.lib import m_ptree as MP
    sel = sim.st.get_model("crm", SYSTEM_ENTITY, MP.ATTRSEL)
    assert sel["roles"].get("body.keys") in ("split", "target"), sel["roles"].get("body.keys")
    tr = sim.tree("crm")
    root = tr.nodes[tr.root]
    visit = None
    for g, c in zip(root.split.groups, root.split.children):
        if any("/visit" in str(v) for v in g):
            visit = tr.nodes[c]
    assert visit is not None
    assert isinstance(visit.targets.get("body.keys"), PN.SetSummary), sorted(visit.targets)


def test_route_node_is_born_with_the_rows_it_waited_for():
    """A route gets its own node once it recurred (>= 3 rows over >= 2 dates);
    the rows it waited for were learned by the non-learning `other` child only,
    so a daily two-person action (the 17:00 report) started its node without
    its first two days and confirmed after day 18 on pack O. The route node is
    born with those rows (their evidence, dates, arrivals and content)."""
    sel = {"targets_sys": {0: ["body.len"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)
    reporters = ["192.168.1.23", "10.168.7.121"]

    def day(d, t0, rng):
        if not is_workday(t0):
            return []
        ev = [(t0 + rng.uniform(9, 17) * 3600, "oa", "192.168.3.20",
               {"http.route": "GET oa /docs", "body.len": 0.0}) for _ in range(5)]
        ev += [(t0 + (17 * 60 + rng.uniform(0, 10)) * 60, "oa", ip,
                {"http.route": "POST oa /report/generate", "body.len": float(rng.uniform(20000, 60000))})
               for ip in reporters]
        return ev
    sim.add(daily(day, 4, seed=3))
    sim.run_until(MON + 4 * DAY)
    tr = sim.tree()
    root = tr.nodes[tr.root]
    rep = None
    for g, c in zip(root.split.groups, root.split.children):
        if any("/report" in str(v) for v in g):
            rep = tr.nodes[c]
    assert rep is not None
    # 4 workdays x 2 reports: every one of them is the node's (created on day 2)
    assert rep.n_days() == 4, rep.n_days()
    assert rep.n_m(sim.now) >= 6.0, rep.n_m(sim.now)          # 8 rows (H_m-decayed); without: ~4
    assert set(k for k, *_ in rep.who.levels[0].items(sim.now)) == set(reporters)


def test_confirmation_counts_observations_not_decayed_mass():
    """N_CONF is a number of observations: a daily action seen 20 times over
    two weeks has 20 observations, but its H_l-decayed evidence sum n_c is
    ~17 (the 17:00 report confirmed on day 18 instead of day 14). The node
    counts its segment's observations undecayed; an accepted structural
    change restarts the count at the segment's H_m evidence."""
    nd = PN.Node(1, 0, 1, 0, (), 0.0)
    for i in range(20):
        nd.update_core(i * 0.7 * DAY, 1.0, 1.0, [None] * 5, "1.1.1.1")
    t = 19 * 0.7 * DAY
    assert nd.n_c(t) < 19.0
    assert abs(nd.n_obs() - 20.0) < 1e-9
    nd.reset_confidence(t)
    assert nd.n_obs() <= nd.n_m(t) + 1e-9


def test_action_attributes_rank_before_source_properties_at_a_node():
    """A public portal: every visitor has its own client (TTL, TCP window,
    stack - source properties, random over the population: high entropy);
    the comment form carries text, a news id and a size in one 1.5-3 KB bin
    (low entropy). The comment node's targets describe the ACTION first: its
    body size is tracked although the client noise has more bits. Before
    round 3 the node's m_t slots went to the client properties and the body
    size was stated from one observation (pack O PUB.portal.visit#3)."""
    rng0 = np.random.default_rng(4)
    ips = [f"10.60.{i // 200}.{i % 200 + 1}" for i in range(300)]
    cli = {ip: (int(rng0.integers(0, 16)) * 8 + 32, int(rng0.integers(0, 16)) * 1024 + 8192,
                f"stack{int(rng0.integers(0, 16))}",
                str(rng0.choice(["ua1", "ua2", "ua3", "ua4", "ua5", "ua6"])),
                str(rng0.choice(["a", "b", "c", "d"])), str(rng0.choice(["x", "y", "z", "w"])))
           for ip in ips}

    def portal(d, t0, rng):
        ev = []
        for _ in range(400):
            ip = ips[int(rng.integers(0, len(ips)))]
            ttl, win, st, ua, h1, h2 = cli[ip]
            base = {"net.ttl": ttl, "net.win": win, "client.stack": st, "hdr.user-agent": ua,
                    "hdr.accept-language": h1, "hdr.dnt": h2}
            u = rng.random()
            if u < 0.75:
                ev.append((t0 + rng.uniform(7, 23) * 3600, "portal", ip,
                           dict(base, **{"http.route": "GET portal /news/{num}", "http.method": "GET",
                                         "net.bytes_down": float(rng.uniform(5000, 40000))})))
            elif u < 0.85:                                # uploads: body sizes over three decades
                ev.append((t0 + rng.uniform(7, 23) * 3600, "portal", ip,
                           dict(base, **{"http.route": "POST portal /upload", "http.method": "POST",
                                         "body.len": float(np.exp(rng.uniform(np.log(300), np.log(300000)))),
                                         "net.bytes_down": float(rng.uniform(200, 900))})))
            else:
                ev.append((t0 + rng.uniform(7, 23) * 3600, "portal", ip,
                           dict(base, **{"http.route": "POST portal /comment", "http.method": "POST",
                                         "body.len": float(rng.uniform(1600, 1700)),
                                         "body.kv.text": "".join(rng.choice(list("abcdefgh"), int(rng.integers(20, 80)))),
                                         "body.kv.news": str(int(rng.integers(1000, 9999))),
                                         "net.dur_ms": float(rng.uniform(10, 200)),
                                         "net.bytes_down": float(rng.uniform(200, 900))})))
        return ev
    sim = Sim(p05=True)
    sim.add(daily(portal, 3, seed=12))
    sim.run_until(MON + 3 * DAY)
    from app.engines.behavior.lib import m_ptree as MP
    from app.engines.behavior.lib import pselect as SEL
    tr = sim.tree("portal")
    root = tr.nodes[tr.root]
    cid = next(c for g, c in zip(root.split.groups, root.split.children) if any("/comment" in str(v) for v in g))
    pr = sim.p05.probes[("portal", EV.KIND_TXN)]
    hier = MP.hierarchies(sim.st, "portal", sim.cfg)
    tsys = ["net.ttl", "net.win", "client.stack", "body.len", "body.kv.text", "net.dur_ms"]
    prox = SEL.who_proxies(pr, sim.now, hier, tsys)
    assert {"net.ttl", "net.win", "client.stack"} <= set(prox), prox
    m_t = 3
    plain = SEL.node_targets_from_probe(tr, pr, sim.now, hier, tsys, m_t=m_t, n_min=12)
    assert "body.len" not in plain[cid], plain[cid]          # entropy alone: client noise wins
    ov = SEL.node_targets_from_probe(tr, pr, sim.now, hier, tsys, m_t=m_t, n_min=12, proxies=prox)[cid]
    assert "body.len" in ov and "body.kv.text" in ov, ov
    assert all(ov.index(p) > ov.index("body.len") for p in prox if p in ov), ov


def test_lazy_offer_keeps_the_same_sample():
    from app.engines.behavior.lib import psketch as PS
    a, b = PS.WeightedReservoir(8, PS.H_M, 3), PS.WeightedReservoir(8, PS.H_M, 3)
    built = []
    for i in range(200):
        a.offer(i, 1.0 + (i % 3), 3600.0 * i)
        b.offer_lazy(lambda i=i: built.append(i) or i, 1.0 + (i % 3), 3600.0 * i)
    assert [x for x, _, _ in a.items()] == [x for x, _, _ in b.items()]
    assert len(built) < 200


def test_hold_check_numeric_band_is_float_noise_tolerant():
    """P06 fits bands on log values: a constant 409-byte request comes back as
    the band [409.00000000000017, 409.00000000000017]. Every held-out check of
    such a statement failed (pack O: every health-check node, 0 of 1 066
    checks), so their confidence was ~0 although they held exactly."""
    from app.engines.behavior.pattern_tree import PatternTreeEngine
    import math as _m
    v = _m.exp(_m.log(409.0))

    class _N:
        meta = {}
        ref = {"t": 1.0, "hold": {"net.bytes_up": ("num", v, v, 0.99),
                                  "net.bytes_up#range": ("range", v, v, 0.995)}}
    nd = _N()
    assert v != 409.0
    for i in range(5):
        PatternTreeEngine._hold_check(None, nd, lambda a: 409.0, [None] * 5, 0, 600.0, 100.0 + i, 1.0)
    cs = nd.meta["hold"].counts(105.0)
    assert all(h == n > 0 for n, h, _ in cs.values()), cs


def test_split_child_range_keeps_its_rare_pre_split_extremes(monkeypatch):
    """The row reservoir is a sample: with few rows kept, a child's rare
    pre-split extreme (综合部's < 1 KB logins, 5 % of its rows) is usually not
    in it, and the child's stated range began above 1 KB (pack O: range_raw
    min 1 053 / 1 038 B on seeds 0 / 1, '100 % in 0.5-3 KB' failed). The leaf
    tracks per (split candidate, value) the observed extremes of its numeric
    targets with their days (VFDT-style sufficient statistics); the child's
    range starts with its own extremes even when the sample missed them."""
    from app.engines.behavior.pattern_tree import PatternTreeEngine
    monkeypatch.setattr(PatternTreeEngine, "_keep_row", lambda *a, **k: None)   # no row sample at all
    sel = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: [("net.src", 1)]}, "roles": {}}
    sim = Sim(sel=sel)

    def day(d, t0, rng):
        if not is_workday(t0):
            return []
        ev = []
        for i, ip in enumerate(GA):
            size = 520.0 if (d in (3, 4) and i == 0) else rng.uniform(1024, 2048)
            ev.append(_login(t0 + (9 * 60 + rng.uniform(0, 21)) * 60, ip, size))
        for ip in POP:
            ev.append(_login(t0 + rng.uniform(13 * 60, 17 * 60) * 60, ip, rng.uniform(600, 1400)))
        return ev
    sim.add(daily(day, 14, seed=31))
    sim.run_until(MON + 14 * DAY)
    assert sim.splits()
    assert int((min(x[0] for x in sim.splits()) - MON) // DAY) >= 5   # the 520 B logins are pre-split
    tr = sim.tree()
    ga = next(nd for nd in tr.nodes.values() if nd.parent is not None and nd.split is None
              and set(nd.who.heavy_set(0, sim.now)[0]) == set(GA))
    num = ga.targets["net.bytes_up"]
    ring_days = sorted(int(x) for x in num.ring[:, 0] if np.isfinite(x))
    lo, hi, _ = num.observed_range(ring_days[-1])
    if num.log:
        lo, hi = math.exp(lo), math.exp(hi)
    assert lo <= 520.0 + 1e-6 and hi <= 2048.0 + 1e-6, (lo, hi)
