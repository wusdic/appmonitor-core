"""P04 PatternTree (behavior.pattern_tree): evidence-driven specialisation and
generalisation, exceptions, budget, lifecycle and drift (docs/lib3/progressive.md
§6.5-§6.9, card P04 tests (a)-(m))."""
from __future__ import annotations

import math
import time
import tracemalloc

import numpy as np
import pytest

from ptree_sim import DAY, MON, Event, Sim, daily, is_workday

from app.engines.behavior import pattern_tree as P4
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.lib import psketch as PS
from app.engines.behavior.lib import ptree as PT
from app.models.schema import SYSTEM_ENTITY

GA = {"192.168.1.21": "jack", "192.168.1.23": "rose", "10.168.7.121": "mike"}
POP = [f"192.168.3.{i}" for i in range(20, 32)]
LOGIN = {"http.route": "POST oa /login", "http.method": "POST"}


def _login(ts, ip, size, user):
    return (ts, "oa", ip, dict(LOGIN, **{"net.bytes_up": float(size), "body.kv.username": user}))


def _org_day(d, t0, rng, ga_start=9 * 60.0, ga_len=21.0):
    """GA logs in 09:00-09:21 with 1-2 KB bodies; a 12-IP population logs in
    13:00-17:00 with 0.6-1.4 KB bodies (distinct time windows, workdays only)."""
    if not is_workday(t0):
        return []
    ev = []
    for ip, u in GA.items():
        m = ga_start + rng.uniform(0, ga_len)
        ev.append(_login(t0 + m * 60, ip, rng.uniform(1024, 2048), u))
    for i, ip in enumerate(POP):
        m = rng.uniform(13 * 60, 17 * 60)
        ev.append(_login(t0 + m * 60, ip, rng.uniform(600, 1400), f"s{i:02d}x"))
    return ev


SEL_WHO = {"targets_sys": {0: ["net.bytes_up"]},
           "split_cands": {0: [("net.src", 1), ("net.src", 0)]}, "roles": {}}


# ------------------------------------------------------------------ (a), (d)
def test_a_who_split_appears_within_ten_workdays_only_when_valid():
    sim = Sim(sel=SEL_WHO)
    sim.add(daily(_org_day, 15, seed=1))
    first = None
    for d in range(15):
        sim.run_until(MON + (d + 1) * DAY)
        sp = sim.splits()
        if sp and first is None:
            first = d
    assert sim.splits(), "no split in 15 days"
    ts, op, nid, _, kids, det = sim.splits()[0]
    # 11 workdays in days 0..14 (Mon..Fri x 2 + Mon); within 10 workdays
    workdays = sum(1 for d in range(first + 1) if is_workday(MON + d * DAY))
    assert workdays <= 10, workdays
    assert det["attr"] == "net.src"
    # rule (V): the anytime-valid e-value crossed tau0 + log2 C_ever
    assert det["log2_e"] >= det["threshold"] >= 10.0
    # (d) value grouping: the department's addresses form one child, the rest is `other`
    tr = sim.tree()
    sp = tr.nodes[nid].split if nid in tr.nodes else None
    assert sp is not None
    ga_groups = [g for g in sp.groups if any(str(v).startswith(("192.168.1", "10.168.7")) for v in g)]
    assert len(ga_groups) == 1
    pop_in_named = [g for g in sp.groups if any(str(v).startswith("192.168.3") for v in g)]
    assert len(pop_in_named) <= 1


# ------------------------------------------------------------------------ (b)
def test_b_independent_attribute_never_splits():
    """A candidate independent of every target, monitored every 32 units for
    ten days on 8 independent trees: no false split (bound 2^-10 H(C_ever))."""
    sel = {"targets_sys": {0: ["net.bytes_up", "http.status"]},
           "split_cands": {0: [("meta.z", 0)]}, "roles": {}}
    sim = Sim(sel=sel)
    systems = [f"s{i:02d}" for i in range(8)]

    def day(d, t0, rng):
        ev = []
        for s in systems:
            for _ in range(30):
                m = rng.uniform(8 * 60, 18 * 60)
                ev.append((t0 + m * 60, s, f"10.0.{rng.integers(0, 4)}.{rng.integers(1, 9)}",
                           {"http.route": "GET x /a", "meta.z": f"z{rng.integers(0, 4)}",
                            "net.bytes_up": float(rng.lognormal(7, 0.5)),
                            "http.status": int(rng.choice([200, 302, 404], p=[0.8, 0.15, 0.05]))}))
        return ev
    sim.add(daily(day, 10, seed=7))
    sim.run_until(MON + 10 * DAY)
    checks = 0
    for s in systems:
        tr = sim.tree(s)
        assert tr is not None and len(tr) == 1, (s, [x for x in tr.lineage][:3])
        ss = tr.nodes[tr.root].split_stats
        assert ss is not None and ss.checks >= 5
        checks += ss.checks
    assert checks >= 48


# ----------------------------------------------------------------------- (b'')
def test_b2_mass_is_not_evidence():
    """An aggregated row of mass 40 and a thinned row with HT weight 1000 each
    contribute at most one evidence unit (PPC-9); mass carries the weights."""
    sel = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)
    t = MON + 10 * 3600.0
    sim.add([(t, "oa", "10.1.1.1", {"http.route": "GET x /a", "net.bytes_up": 100.0, "__w": 40.0}),
             (t + 1000.0, "oa", "10.1.1.2", {"http.route": "GET x /a", "net.bytes_up": 100.0, "__pi": 0.001})])
    sim.run_until(MON + 1 * DAY)
    root = sim.tree().nodes[0]
    at = MON + 1 * DAY
    assert root.n_eff.get(PS.CH_S, at) <= 2.0 + 1e-9
    assert root.n_eff.get(PS.CH_S, at) >= 0.5
    assert root.mass.get(PS.CH_S, at) >= 0.5 * (40.0 + 1000.0)


# ------------------------------------------------------------------------ (e)
def test_e_split_pruned_after_children_converge():
    sel = {"targets_sys": {0: ["net.bytes_up", "http.status"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)

    def day(d, t0, rng):
        return [(t0 + rng.uniform(0, DAY), "oa", f"10.0.0.{rng.integers(1, 30)}",
                 {"http.route": "GET x /a", "meta.z": f"z{rng.integers(0, 2)}",
                  "net.bytes_up": float(rng.lognormal(7, 0.5)), "http.status": 200})
                for _ in range(80)]
    sim.add(daily(day, 9, seed=3))
    sim.run_until(MON + 1 * DAY)
    tr = sim.tree()
    tr.split(tr.root, "meta.z", 0, [["z0"], ["z1"]], sim.now)       # a split that saves nothing
    sim.run_until(MON + 9 * DAY)
    assert tr.nodes[tr.root].split is None
    assert any(op in ("prune", "merge") for _, op, *_ in tr.lineage)


# ------------------------------------------------------------------------ (g)
def test_g_exception_for_differing_ip_not_for_bound_usernames():
    sel = {"targets_sys": {0: ["net.bytes_up", "body.kv.username"]}, "split_cands": {0: []},
           "roles": {}}
    sim = Sim(sel=sel)
    ips = [f"192.168.1.{i}" for i in range(10, 18)]
    odd = ips[0]

    def day(d, t0, rng):
        if not is_workday(t0):
            return []
        ev = []
        for k, ip in enumerate(ips):
            for _ in range(4):
                m = rng.uniform(9 * 60, 17 * 60)
                size = rng.uniform(3000, 4000) if ip == odd else rng.uniform(1024, 2048)
                ev.append(_login(t0 + m * 60, ip, size, f"user{k}"))
        return ev
    sim.add(daily(day, 12, seed=5))
    sim.run_until(MON + 12 * DAY)
    tr = sim.tree()
    root = tr.nodes[tr.root]
    assert root.state in PN.CONFIDENT_STATES
    assert set(root.exc) == {odd}, root.exc
    xn = tr.nodes[root.exc[odd]]
    assert "net.bytes_up" in xn.meta["targets"]
    assert "body.kv.username" not in xn.meta["targets"]


# ------------------------------------------------------------------------ (h)
def test_h_quarantined_events_held_released_and_rejected():
    sel = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)
    q_ip, r_ip = "10.9.9.1", "10.9.9.2"

    def quarantine(s):
        for ip in (q_ip, r_ip):
            s.st.add_vec("oa", ip, "behavior.quarantine", s.now, [1.0])
    sim.hooks.append(quarantine)
    t = MON + 9 * 3600.0
    sim.add([(t + i * 60, "oa", ip, {"http.route": "GET x /a", "net.bytes_up": 500.0})
             for i in range(5) for ip in (q_ip, r_ip, "10.0.0.1")])
    sim.run_until(MON + 14 * 3600.0)
    root = sim.tree().nodes[0]
    at = sim.now
    seen = {k for k, *_ in root.who.levels[0].items(at)}
    assert "10.0.0.1" in seen and q_ip not in seen and r_ip not in seen
    aux = sim.p04.aux(MP.get_ptree(sim.st, "oa"))
    assert len(aux["held"][("oa", q_ip)]) == 5
    sim.hooks.clear()
    sim.st.put_model("oa", q_ip, "model.control", {"version": 1, "release": (t - 1, t + 3600)})
    sim.st.put_model("oa", r_ip, "model.control", {"version": 1, "frozen": True})
    sim.tick()
    seen = {k for k, *_ in root.who.levels[0].items(sim.now)}
    assert q_ip in seen and r_ip not in seen
    assert ("oa", q_ip) not in aux["held"] and ("oa", r_ip) not in aux["held"]


# ------------------------------------------------------------------------ (j)
def test_j_confidence_grows_then_restarts_after_accepted_window_change():
    sel = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)
    start = {"m": 9 * 60.0}

    def day(d, t0, rng):
        if not is_workday(t0):
            return []
        m0 = 9 * 60.0 if d < 33 else 14 * 60.0
        return [_login(t0 + (m0 + rng.uniform(0, 21)) * 60, ip, rng.uniform(1024, 2048), u)
                for ip, u in GA.items()]
    sim.add(daily(day, 42, seed=11))
    sim.run_until(MON + 33 * DAY)
    root = sim.tree().nodes[0]
    at = sim.now
    # longer is more precise: the closed who-set's unseen mass on the confidence channel
    U = root.who.levels[0].unseen(at)
    assert root.n_c(at) >= 50 and U <= 0.01, (root.n_c(at), U)
    conf_before = root.when.evidence(0, at, conf=True)
    assert conf_before > 1.3 * root.when.evidence(0, at, conf=False)
    drift = []
    for d in range(34, 43):
        sim.run_until(MON + d * DAY)
        drift = [e for e in sim.events("pattern_drift") if e.extra.get("attr") == "@when"]
        if drift:
            break
    assert drift, [e.extra for e in sim.events("pattern_drift")]
    assert drift[0].ts > MON + 33 * DAY
    # at acceptance the time-of-day confidence restarted from the H_m state
    at = sim.now
    assert root.when.evidence(0, at, conf=True) < conf_before
    assert root.when.evidence(0, at, conf=True) == pytest.approx(root.when.evidence(0, at, conf=False), rel=0.05)
    assert sim.p04 is not None and root.state in ("confirmed", "stable")


# ------------------------------------------------------------------------ (k)
def test_k_outlier_damping_scales_mass_and_evidence():
    sel = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)

    def damp(s):
        b = s.st.batch_at("oa", EV.EVT_BATCH, s.now)
        if b is not None:
            cols = EV.cols_from_rows(b.n, [{"damp": 0.1}] * b.n)
            s.st.add_batch("oa", EV.PAT_ASSIGN, s.now, b.aligned(cols))
    sim.hooks.append(damp)
    t = MON + 9 * 3600.0
    sim.add([(t + i * 900.0, "oa", f"10.0.0.{i}", {"http.route": "GET x /a", "net.bytes_up": 500.0})
             for i in range(10)])
    sim.run_until(MON + DAY)
    root = sim.tree().nodes[0]
    at = sim.now
    assert root.n_eff.get(PS.CH_L, at) == pytest.approx(1.0, rel=0.05)     # 10 rows x 0.1
    assert root.mass.get(PS.CH_L, at) == pytest.approx(1.0, rel=0.05)


# ------------------------------------------------------------------------ (l)
def test_l_split_on_gone_attribute_collapses_at_next_daily_check():
    sel = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)

    def day(d, t0, rng):
        # the adapter stops sending header x after day 0 (a schema change)
        extra = {"hdr.x": "a"} if d == 0 else {}
        return [(t0 + rng.uniform(0, DAY), "oa", "10.0.0.1",
                 dict({"http.route": "GET x /a", "net.bytes_up": 100.0}, **extra)) for _ in range(20)]
    sim.add(daily(day, 3, seed=1))
    sim.run_until(MON + DAY + 3600)
    tr = sim.tree()
    tr.split(tr.root, "hdr.x", 0, [["a"]], sim.now)
    reg = MP.get_registry(sim.st, "oa")
    reg.records["hdr.x"].state = "gone"
    sim.run_until(MON + 2 * DAY + 3600)
    assert tr.nodes[tr.root].split is None
    assert any(op == "prune" and isinstance(det, dict) and det.get("gone") == "hdr.x"
               for _, op, _, _, _, det in tr.lineage)


# ------------------------------------------------------------------------ (m)
def test_m_month_end_pattern_retired_dormant_then_revived():
    sel = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)
    sim.add([(MON + 10 * 3600.0, "oa", "10.0.0.1", {"http.route": "GET x /a", "cal.me": 0,
                                                    "net.bytes_up": 1.0})])
    sim.run_until(MON + 3600 * 12)
    tr = sim.tree()
    sp = tr.split(tr.root, "cal.me", 0, [[1]], sim.now)
    child = tr.nodes[sp.children[0]]
    # a monthly pattern: seen on three regularly spaced dates, then quiet for 30 d
    child.state = "stale"
    child.meta["stale_at"] = sim.now - 31 * DAY
    child.days_bits = (1 << 0) | (1 << 21) | (1 << 42)
    child.days_total = 3
    child.last_day = P4._local_day(sim.now, 8 * 3600) - 1
    sim.run_until(MON + DAY + 3600)
    assert sp.children == [] and len(tr.dormant) == 1
    assert sim.events("pattern_retired")
    t = sim.now + 3600
    sim.add([(t, "oa", "10.0.0.2", {"http.route": "GET x /a", "cal.me": 1, "net.bytes_up": 1.0})])
    sim.run_until(t + 6 * 3600)
    rev = sim.events("pattern_revived")
    assert rev and not tr.dormant
    new = tr.nodes[sp.children[0]]
    assert new.state == "confirmed" and new.n_m(sim.now) > 0


# ------------------------------------------------------------------------ (f)
def test_f_revision_replaces_prefix_split_by_learned_groups():
    """Two behavioural groups spread over two /24s (3:1 in each): the tree first
    splits on /24 (the only IP level available); once P11 publishes groups, EFDT
    revision replaces the /24 split by the group split (§6.6)."""
    g1 = ["192.168.1.1", "192.168.1.2", "192.168.1.4", "192.168.2.3"]
    g2 = ["192.168.1.3", "192.168.2.1", "192.168.2.2", "192.168.2.4"]
    sel = {"targets_sys": {0: ["net.bytes_up"]},
           "split_cands": {0: [("net.src", 1), ("net.src", 3)]}, "roles": {}}
    sim = Sim(sel=sel)

    def day(d, t0, rng):
        ev = []
        for ips, (h0, h1), (lo, hi) in ((g1, (9, 12), (1024, 2048)), (g2, (13, 17), (3072, 4096))):
            for ip in ips:
                for _ in range(20):
                    ev.append((t0 + rng.uniform(h0, h1) * 3600, "oa", ip,
                               {"http.route": "POST oa /login", "net.bytes_up": float(rng.uniform(lo, hi))}))
        return ev
    sim.add(daily(day, 20, seed=9))
    tr = None
    for d in range(1, 9):
        sim.run_until(MON + d * DAY)
        tr = sim.tree()
        if tr is not None and tr.nodes[tr.root].split is not None:
            break
    root = tr.nodes[tr.root]
    assert root.split is not None and (root.split.attr, root.split.level) == ("net.src", 1)
    groups = {ip: "g1" for ip in g1} | {ip: "g2" for ip in g2}
    sim.st.put_model("__org__", "__org__", MP.WHO_GROUPS,
                     {"ip2g": groups, "groups": {"g1": {"members": g1}, "g2": {"members": g2}}})
    for d in range(d + 1, 21):
        sim.run_until(MON + d * DAY)
        if (root.split.attr, root.split.level) == ("net.src", 3):
            break
    assert (root.split.attr, root.split.level) == ("net.src", 3), [x[1:3] for x in tr.lineage]
    assert any(op == "replace" for _, op, *_ in tr.lineage)
    assert sim.events("pattern_replaced")
    kids = [tr.nodes[c] for c in root.split.children]
    assert sorted(len(g) for g in root.split.groups) in ([1], [1, 1])


# --------------------------------------------------- pairs, budget, snapshots
def test_binding_pairs_requested_by_p08_are_counted():
    sim = Sim(sel=SEL_WHO)
    sim.st.put_model("oa", SYSTEM_ENTITY, MP.PWANT,
                     {"p08": {"pairs": [{"x": "net.src", "x_level": 0, "y": "body.kv.username"}]}})
    sim.add(daily(_org_day, 3, seed=2))
    sim.run_until(MON + 3 * DAY)
    root = sim.tree().nodes[0]
    ps = root.pairs[("net.src", "body.kv.username")]
    tab = ps.table(sim.now)
    for ip, u in GA.items():
        assert tab[ip][0][0] == u


def test_node_budget_is_enforced():
    sel = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: [("http.route", 0)]}, "roles": {}}
    sim = Sim(sel=sel)
    sim.st.put_model("__org__", "__org__", MP.BUDGET, {"trees": {"oa": {"n_max": 6, "l_max": 4}}})

    def day(d, t0, rng):
        return [(t0 + rng.uniform(8, 18) * 3600, "oa", f"10.0.0.{rng.integers(1, 50)}",
                 {"http.route": f"GET oa /r{r}", "net.bytes_up": float(rng.lognormal(5 + r, 0.2))})
                for r in range(12) for _ in range(12)]
    sim.add(daily(day, 6, seed=4))
    sizes = []
    for d in range(1, 7):
        sim.run_until(MON + d * DAY)
        sizes.append(len(sim.tree()))
    assert max(sizes) <= 6, sizes
    assert sim.splits()


def test_reference_snapshot_of_confirmed_nodes():
    sel = {"targets_sys": {0: ["net.bytes_up", "body.kv.username"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)

    def day(d, t0, rng):
        return [_login(t0 + (9 * 60 + rng.uniform(0, 300)) * 60, f"192.168.1.{k}", rng.uniform(1024, 2048), f"u{k}")
                for k in range(10)]
    sim.add(daily(day, 6, seed=6))
    sim.run_until(MON + 6 * DAY)
    root = sim.tree().nodes[0]
    assert root.state in ("confirmed", "stable")
    assert root.ref is not None and "net.bytes_up" in root.ref["targets"]
    assert root.ref["targets"]["net.bytes_up"]["q"][2] > 0


def test_pairs_in_p08_by_kind_shape():
    sim = Sim(sel=SEL_WHO)
    sim.st.put_model("oa", SYSTEM_ENTITY, MP.PWANT, {"pairs": {"fmt": 1, "updated": MON, "by_kind": {
        0: {0: [["net.src", "body.kv.username"], ["net.src@1", "body.kv.username"]]}}}})
    sim.add(daily(_org_day, 2, seed=2))
    sim.run_until(MON + 2 * DAY)
    root = sim.tree().nodes[0]
    assert ("net.src", "body.kv.username") in root.pairs
    tab = root.pairs[("net.src@1", "body.kv.username")].table(sim.now)
    assert "192.168.1.0/24" in tab


# ------------------------------------------------------------------------ (c)
def _dup_run(order):
    sel = {"targets_sys": {0: ["net.bytes_up", "body.kv.username"]},
           "split_cands": {0: [(a, 0) for a in order]}, "roles": {}}
    sim = Sim(sel=sel)

    def day(d, t0, rng):
        out = []
        for ts, s, ip, a in _org_day(d, t0, rng):
            dept = "ga" if ip in GA else "pop"
            out.append((ts, s, ip, dict(a, **{"meta.dept": dept, "meta.dept_copy": dept})))
        return out
    sim.add(daily(day, 12, seed=3))
    sim.run_until(MON + 12 * DAY)
    return sim.splits()


def test_c_tie_between_equal_candidates_is_broken_deterministically():
    """Two copies of one informative attribute: rule (S) sees a tie (identical
    savings, empirical-Bernstein eps = 0 <= tau_tie) and does not block the
    split; the first-tracked candidate wins, identically on every run."""
    a = _dup_run(["meta.dept", "meta.dept_copy"])
    b = _dup_run(["meta.dept", "meta.dept_copy"])
    c = _dup_run(["meta.dept_copy", "meta.dept"])
    assert a, "the tie blocked the split"
    assert a[0][5]["attr"] == "meta.dept" and b[0][5]["attr"] == "meta.dept"
    assert a[0][0] == b[0][0]                      # same time, same decision
    assert c[0][5]["attr"] == "meta.dept_copy" and c[0][0] == a[0][0]


# ------------------------------------------------------------------------ (d)
def test_d_value_grouping_one_group_of_three_ips_plus_other():
    """At /32 the department's three addresses (two different /24s) form ONE
    child; every other IP stays in `other` or in one population group (KT value
    grouping, §6.5.5). The population has 4 IPs here so that every source holds
    a value slot (k_v = 8): with more equally active sources than slots a
    single /32 level cannot hold them (measured: 12 + 3 sources, no split in
    15 days); the tree then reaches the department through /24 (test a) or the
    learned group level (test f)."""
    sel = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: [("net.src", 0)]}, "roles": {}}
    sim = Sim(sel=sel)
    pop = POP[:4]

    def day(d, t0, rng):
        return [e for e in _org_day(d, t0, rng) if e[2] in GA or e[2] in pop]
    sim.add(daily(day, 15, seed=2))
    sim.run_until(MON + 15 * DAY)
    assert sim.splits(), "no split"
    nid = sim.splits()[0][2]
    sp = sim.tree().nodes[nid].split
    assert sp.attr == "net.src" and sp.level == 0
    named = [set(map(str, g)) for g in sp.groups]
    assert set(GA) in named, named
    assert all(not (g & set(GA)) or g == set(GA) for g in named), named
    assert sum(1 for g in named if g & set(pop)) <= 1, named


# --------------------------------------------------------------- (h') delay
def test_h_quarantine_during_learning_delay_holds_the_rows():
    """Delayed learning (§6.9.3): rows of an IP that is quarantined AFTER its
    tick but before the tick is learned (t - D) are held, not learned."""
    sel = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)
    bad = "10.9.9.7"
    t = MON + 9 * 3600.0 + 60.0

    def quarantine_later(s):
        if s.now >= t + 1800.0:                     # flagged half an hour after its events
            s.st.add_vec("oa", bad, "behavior.quarantine", s.now, [1.0])
    sim.hooks.append(quarantine_later)
    sim.add([(t + i * 30, "oa", ip, {"http.route": "GET x /a", "net.bytes_up": 500.0})
             for i in range(4) for ip in (bad, "10.0.0.1")])
    sim.run_until(MON + 13 * 3600.0)
    root = sim.tree().nodes[0]
    seen = {k for k, *_ in root.who.levels[0].items(sim.now)}
    assert "10.0.0.1" in seen and bad not in seen
    aux = sim.p04.aux(MP.get_ptree(sim.st, "oa"))
    assert len(aux["held"][("oa", bad)]) == 4


def test_published_model_carries_no_working_state():
    """Plain-data copies of model.ptree (eval snapshots, API) do not copy P04's
    private working state (held rows, burst runs)."""
    sim = Sim(sel=SEL_WHO)
    sim.add(daily(_org_day, 2, seed=1))
    sim.run_until(MON + 2 * DAY)
    m = MP.get_ptree(sim.st, "oa")
    assert "aux" not in vars(m) and all(not k.startswith("aux") for k in vars(m))
    assert sim.p04.aux(m)["last"]
