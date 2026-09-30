"""Regression tests of the progressive-core integration fixes (2026-09-30,
docs/lib3/progressive.md §16 "Integration results"): each test pins one root
cause measured on pack O.

  * burst runs restart after RUN_MAX (a 60-s monitor was never confirmed);
  * route-first partition of txn roots (routes were peeled one binary group per
    level and split by /24 before the approval / report routes separated);
  * selective MDL gain of a split (irrelevant high-entropy targets cancelled
    the saving of a valid /24 split at the OA login node);
  * the system view's per-group parts of a node shared by several groups
    ("某类人" decomposition)."""
from __future__ import annotations

import datetime as _dt
import math
from typing import Any, List

import numpy as np

from helpers import make_store
from ptree_sim import DAY, MON, Sim, daily, is_workday

from app.engines.behavior import views as VW
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevalue as PE
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import psketch as PS
from app.engines.behavior.lib import ptree as PT
from app.eval.pmetrics import statements
from app.models.schema import ORG, SYSTEM_ENTITY


# ------------------------------------------------------------ burst runs
def test_burst_run_restarts_after_run_max_for_a_continuous_source():
    b = PS.BurstEvidence(64)
    tot = sum(b.unit(("192.168.9.9", 1), MON + 60.0 * i) for i in range(3 * 60))
    h60 = sum(1.0 / (r + 1) for r in range(60))
    # three one-hour runs of 60 rows, not one run of 180 rows
    assert abs(tot - 3 * h60) < 1e-6, tot
    # a gap longer than tau_burst still restarts a run
    b2 = PS.BurstEvidence(64)
    assert b2.unit("k", 0.0) == 1.0 and b2.unit("k", 10.0) == 0.5 and b2.unit("k", 400.0) == 1.0


def test_continuous_monitor_route_is_confirmed():
    """A 60-s health check with a noisy duration target: its route node is a
    confirmed pattern, and stays one (no lasting `evolving` state from the
    structural detector on a stationary source)."""
    sel = {"targets_sys": {0: ["net.bytes_down", "net.dur_ms"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)

    def day(d, t0, rng):
        return [(t0 + 60.0 * i + rng.uniform(0, 1), "oa", "192.168.9.9",
                 {"http.route": "GET oa /health", "net.bytes_down": 200.0,
                  "net.dur_ms": float(rng.lognormal(3.0, 0.6))}) for i in range(1440)]
    sim.add(daily(day, 12, seed=1))
    sim.run_until(MON + 5 * DAY)
    tr = sim.tree()
    nd = tr.nodes[sim.top()]
    assert nd.ctx and nd.ctx[-1][0] == "http.route"
    assert nd.state in ("confirmed", "stable"), (nd.state, nd.n_c(sim.now))
    states = []
    for d in range(6, 13):
        sim.run_until(MON + d * DAY)
        states.append(nd.state)
    assert states[-1] in ("confirmed", "stable"), states
    assert states.count("evolving") <= 5, states


# ------------------------------------------------------- route partition
def test_root_is_partitioned_by_route_and_the_partition_is_kept():
    sel = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: [("net.src", 1)]}, "roles": {}}
    sim = Sim(sel=sel)
    routes = ["POST oa /login", "GET oa /home", "GET oa /approval/list"]

    def day(d, t0, rng):
        if not is_workday(t0):
            return []
        ev = []
        for r, route in enumerate(routes):
            for k in range(6):
                ev.append((t0 + (9 * 60 + 10 * r + k) * 60.0, "oa", f"10.1.{r}.{k + 1}",
                           {"http.route": route, "net.bytes_up": float(300 * (r + 1) + rng.uniform(0, 50))}))
        if d == 0:                                    # a one-off route never gets a node
            ev.append((t0 + 12 * 3600.0, "oa", "10.9.9.9", {"http.route": "GET oa /admin/export",
                                                            "net.bytes_up": 100.0}))
        return ev
    sim.add(daily(day, 10, seed=2))
    sim.run_until(MON + 10 * DAY)
    tr = sim.tree()
    root = tr.nodes[tr.root]
    assert root.split is not None and root.split.attr == "http.route" and root.split.level == 0
    named = {next(iter(g)) for g in root.split.groups}
    assert named == set(routes), named
    # every route node's context fixes exactly one route; the partition is never pruned
    for cid in root.split.children:
        a, l, vals, neg = tr.nodes[cid].ctx[-1]
        assert a == "http.route" and not neg and len(vals) == 1
    assert not any(op in ("prune", "merge", "replace") and nid == tr.root
                   for _, op, nid, *_ in tr.lineage)
    # the route nodes are confirmed patterns after >= 3 dates; the `other` bag of
    # waiting / one-off routes never is (a new route is scored as novel there)
    assert all(tr.nodes[c].state in ("confirmed", "stable") for c in root.split.children)
    assert tr.nodes[root.split.other].state == "candidate"


# ---------------------------------------------------------- selective gain
def test_selective_gain_ignores_targets_the_split_does_not_predict():
    """One target depends on the candidate, eight are high-entropy noise: the
    plain summed saving G carries the noise targets' prequential regret, the
    selective gain counts only the targets the split saves bits on."""
    rng = np.random.default_rng(3)
    T = 9
    ss = PE.SplitStats(T, k_b=9, k_v=8, C=1)
    ss.set_candidate(0, ("net.src", 1), card_hint=16.0)
    lc = np.zeros((T, 9))
    for k in range(120):
        if k % 20 == 0:
            ss.close_block()                          # blocks of 20 units (undated use)
        v = int(rng.integers(0, 4))
        bins = [v * 2] + [int(rng.integers(0, 9)) for _ in range(T - 1)]
        p = (lc + 2.0 / 9) / (lc.sum(axis=1, keepdims=True) + 2.0)
        ss.update([f"10.0.{v}.0/24"], bins, p, 1.0)
        for t, b_ in enumerate(bins):
            lc[t, b_] += 1.0
    ss.close_block()
    sel = ss.selective_gain(0)
    assert ss.log2_e(0) >= PE.TAU0 + 1
    assert sel > float(ss.G[0]) + 20.0, (sel, float(ss.G[0]))
    assert sel > 0


# ------------------------------------------------ system view group parts
TZ = _dt.timezone(_dt.timedelta(hours=8))
T0 = _dt.datetime(2026, 9, 1, tzinfo=TZ).timestamp()


def _keys(ip: str, g: str) -> List[Any]:
    p = ip.split(".")
    return [ip, ".".join(p[:3]) + ".0/24", ".".join(p[:2]) + ".0.0/16", f"grp:{g}", "reg:∅"]


def test_system_view_states_each_groups_part_of_a_shared_node():
    ga = ["192.168.1.21", "192.168.1.23", "10.168.7.121"]
    fin = ["192.168.2.10", "192.168.2.11"]
    m = PT.PTreeModel("oa")
    tr = m.tree(EV.KIND_TXN, T0, create=True)
    sp = tr.split(tr.root, "http.route", 0, [["GET oa.corp /docs"]], T0)
    node, root = tr.nodes[sp.children[0]], tr.nodes[tr.root]
    t = T0
    for d in range(21):
        day0 = T0 + d * DAY
        if _dt.datetime.fromtimestamp(day0, TZ).weekday() >= 5:
            continue
        for k, (ip, g) in enumerate([(x, "G1") for x in ga] + [(x, "G2") for x in fin]):
            t = day0 + (10 * 60 + 7 * k) * 60.0
            dayn = int((t + 8 * 3600) // DAY)
            for nd in (root, node):
                nd.update_core(t, 1.0, 1.0, _keys(ip, g), ip, 0, (t + 8 * 3600) % DAY / 60.0, dayn)
    for nd in (root, node):
        nd.state = "confirmed"
    m.t_last = t
    st = make_store()
    st.put_model("oa", SYSTEM_ENTITY, MP.PTREE, m, version=1)
    st.put_model(ORG, ORG, MP.WHO_GROUPS, {
        "groups": {"G1": {"id": "G1", "name": "综合部", "members": ga},
                   "G2": {"id": "G2", "name": "财务部", "members": fin}},
        "ip2g": dict({ip: "G1" for ip in ga}, **{ip: "G2" for ip in fin}),
        "mode": {"oa": {"mode": "ip"}}})
    v = VW.system_view(st, "oa", {"progressive": {"enabled": True}, "tz": "Asia/Shanghai"}, t + 3600.0)
    parts = [s for s in v["statements"] if (s["evidence"].get("who") or {}).get("part_of")]
    by_g = {s["evidence"]["who"]["group"]: s for s in parts}
    assert set(by_g) == {"G1", "G2"}
    assert sorted(by_g["G1"]["evidence"]["who"]["members"]) == sorted(ga)
    assert sorted(by_g["G2"]["evidence"]["who"]["members"]) == sorted(fin)
    for s in parts:
        assert s["evidence"]["who"]["closed"] is False              # a part, not a closure claim
        assert ["net.src", 3, [f"grp:{s['evidence']['who']['group']}"], False] in s["evidence"]["context"]
        assert s["evidence"]["route"] == "GET oa.corp /docs"
        assert "综合部" in s["text_zh"] or "财务部" in s["text_zh"]
    # the scorer reads each part as a statement about the group's members
    ls = statements({"systems": {"oa": {"model.pviews": v}}}, {})
    got = {frozenset(x.who.ipset()) for x in ls if x.who.level == "grp"}
    assert frozenset(ga) in got and frozenset(fin) in got
    # without learned groups there are no parts (the node statement stands alone)
    st.put_model(ORG, ORG, MP.WHO_GROUPS, {"groups": {}, "ip2g": {}, "mode": {"oa": {"mode": "ip"}}})
    v2 = VW.system_view(st, "oa", {"progressive": {"enabled": True}, "tz": "Asia/Shanghai"}, t + 3600.0)
    assert v2["statements"] and not any((s["evidence"].get("who") or {}).get("part_of")
                                         for s in v2["statements"])


# ------------------------------------------- k-sample e-process (rule V)
def _feed(ss: PE.SplitStats, rng: np.random.Generator, days: int, per_day: int, dist_of_slot,
          slot_p, check_every: int = 32) -> float:
    """Days of events: slot ~ slot_p, target bin ~ dist_of_slot(slot); the leaf
    predictive is the running pooled Dirichlet; returns sup log2 e over checks."""
    T, K = ss.T, ss.kb
    lc = np.zeros((T, K))
    sup = -math.inf
    k = 0
    for d in range(days):
        for _ in range(per_day):
            j = int(rng.choice(len(slot_p), p=slot_p))
            b = int(rng.choice(K, p=dist_of_slot(j)))
            p = (lc + 2.0 / K) / (lc.sum(axis=1, keepdims=True) + 2.0)
            ss.update([f"v{j}"], [b], p, 1.0, day=d)
            lc[0, b] += 1.0
            k += 1
            if k % check_every == 0:
                ss.roll(d)
                sup = max(sup, ss.log2_e(0))
        ss.roll(d + 1)
        sup = max(sup, ss.log2_e(0))
    return sup


def test_k_sample_e_process_keeps_the_false_split_bound_under_the_null():
    """Target independent of the candidate: sup over checks of log2 e reaches
    tau = 5 in <= 2^-5 of the runs (Ville), measured over 300 runs."""
    rng = np.random.default_rng(11)
    base = np.array([0.4, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01])
    hits = 0
    runs = 300
    for _ in range(runs):
        ss = PE.SplitStats(1, k_b=9, k_v=8, C=1)
        ss.set_candidate(0, ("net.src", 1), card_hint=16.0)
        sup = _feed(ss, rng, days=10, per_day=40, dist_of_slot=lambda j: base,
                    slot_p=[0.6, 0.2, 0.1, 0.05, 0.05])
        hits += sup >= 5.0
    assert hits / runs <= 2 ** -5 + 0.02, hits


def test_k_sample_e_process_finds_a_small_department_the_ml_e_value_misses():
    """A department of 5 % of a shared node's events whose target differs (a
    username bucket / login size): the k-sample e-process passes tau0 + 3 bits
    within 10 days, the universal-inference e-value (pooled ML denominator,
    §6.5.5 text) does not."""
    rng = np.random.default_rng(5)
    base = np.full(9, 1.0 / 9)
    dept = np.array([0.45, 0.45, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01])
    ss = PE.SplitStats(1, k_b=9, k_v=8, C=1)
    ss.set_candidate(0, ("net.src", 1), card_hint=16.0)
    sup = _feed(ss, rng, days=10, per_day=40, dist_of_slot=lambda j: dept if j == 2 else base,
                slot_p=[0.7, 0.25, 0.05])
    assert sup >= PE.TAU0 + 3, sup
    assert ss.log2_e_ui(0) < PE.TAU0, ss.log2_e_ui(0)


def test_retarget_keeps_evidence_and_conserves_e_process_wealth():
    """A target-list change keeps the kept targets' statistics, moves the
    dropped target's wealth to the new one (log2 e unchanged at the switch)
    and the result stays valid under the null (random retargets)."""
    rng = np.random.default_rng(2)
    base = np.full(9, 1.0 / 9)
    dept = np.array([0.45, 0.45, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01])
    ss = PE.SplitStats(2, k_b=9, k_v=8, C=1)
    ss.set_candidate(0, ("net.src", 1), card_hint=16.0)
    lc = np.zeros((2, 9))
    for d in range(6):
        for _ in range(40):
            j = int(rng.choice(3, p=[0.6, 0.3, 0.1]))
            b0 = int(rng.choice(9, p=dept if j == 2 else base))
            b1 = int(rng.choice(9, p=base))
            p = (lc + 2.0 / 9) / (lc.sum(axis=1, keepdims=True) + 2.0)
            ss.update([f"v{j}"], [b0, b1], p, 1.0, day=d)
            lc[0, b0] += 1
            lc[1, b1] += 1
    ss.roll(6)
    before = ss.log2_e(0)
    e0 = float(ss.E[0, 0])
    n0 = ss.cnt[0, :, 0].sum()
    ss.retarget([0, None], np.array([[True, True]]))     # target 1 replaced by a new one
    assert ss.log2_e(0) == __import__("pytest").approx(before, abs=1e-9)
    assert float(ss.E[0, 0]) == e0 and ss.cnt[0, :, 0].sum() == n0   # the kept target's evidence
    assert ss.cnt[0, :, 1].sum() == 0 and float(ss.E[0, 1]) == 0.0
    # null validity with retargets every 2 days: sup log2 e >= 5 in <= 2^-5 of runs
    hits, runs = 0, 200
    for r_ in range(runs):
        rr = np.random.default_rng(100 + r_)
        s2 = PE.SplitStats(2, k_b=9, k_v=8, C=1)
        s2.set_candidate(0, ("x", 1), card_hint=16.0)
        lc = np.zeros((2, 9))
        sup = -math.inf
        for d in range(10):
            if d and d % 2 == 0:
                s2.roll(d)
                sup = max(sup, s2.log2_e(0))
                s2.retarget([0, None], np.array([[True, True]]))
                lc[1] = 0.0
            for _ in range(40):
                j = int(rr.choice(3, p=[0.6, 0.3, 0.1]))
                b = [int(rr.choice(9, p=base)), int(rr.choice(9, p=base))]
                p = (lc + 2.0 / 9) / (lc.sum(axis=1, keepdims=True) + 2.0)
                s2.update([f"v{j}"], b, p, 1.0, day=d)
                lc[0, b[0]] += 1
                lc[1, b[1]] += 1
        s2.roll(10)
        sup = max(sup, s2.log2_e(0))
        hits += sup >= 5.0
    assert hits / runs <= 2 ** -5 + 0.02, hits


def test_group_part_states_its_own_arrival_window():
    """Two groups share a route node at different hours (09:00-09:21 and
    15:00-15:30): each part's statement carries its own window, fitted on the
    node's minute-reservoir arrivals of its members."""
    ga = ["192.168.1.21", "192.168.1.23", "10.168.7.121"]
    fin = ["192.168.2.10", "192.168.2.11"]
    m = PT.PTreeModel("oa")
    tr = m.tree(EV.KIND_TXN, T0, create=True)
    sp = tr.split(tr.root, "http.route", 0, [["POST oa.corp /login"]], T0)
    node, root = tr.nodes[sp.children[0]], tr.nodes[tr.root]
    node.when.want_minutes(True, seed=1)
    rng = np.random.default_rng(4)
    t = T0
    for d in range(21):
        day0 = T0 + d * DAY
        if _dt.datetime.fromtimestamp(day0, TZ).weekday() >= 5:
            continue
        for ip, g, (m0, m1) in [(x, "G1", (540, 561)) for x in ga] + [(x, "G2", (900, 930)) for x in fin]:
            mi = float(rng.uniform(m0, m1))
            t = day0 + mi * 60.0
            dayn = int((t + 8 * 3600) // DAY)
            for nd in (root, node):
                nd.update_core(t, 1.0, 1.0, _keys(ip, g), ip, 0, mi, dayn)
    for nd in (root, node):
        nd.state = "confirmed"
    m.t_last = t
    st = make_store()
    st.put_model("oa", SYSTEM_ENTITY, MP.PTREE, m, version=1)
    st.put_model(ORG, ORG, MP.WHO_GROUPS, {
        "groups": {"G1": {"id": "G1", "name": "综合部", "members": ga},
                   "G2": {"id": "G2", "name": "财务部", "members": fin}},
        "ip2g": dict({ip: "G1" for ip in ga}, **{ip: "G2" for ip in fin}),
        "mode": {"oa": {"mode": "ip"}}})
    v = VW.system_view(st, "oa", {"progressive": {"enabled": True}, "tz": "Asia/Shanghai"}, t + 3600.0)
    parts = {s["evidence"]["who"]["group"]: s for s in v["statements"]
             if (s["evidence"].get("who") or {}).get("part_of")}
    w1 = parts["G1"]["evidence"]["when"]["workday"]
    w2 = parts["G2"]["evidence"]["when"]["workday"]
    assert len(w1) == 1 and 535 <= w1[0][0] <= 545 and 556 <= w1[0][1] <= 566, w1
    assert len(w2) == 1 and 895 <= w2[0][0] <= 905 and 925 <= w2[0][1] <= 935, w2
    assert "09:" in parts["G1"]["text_zh"] and "15:" in parts["G2"]["text_zh"]


def test_split_children_carry_their_evidence_and_are_not_pruned_young():
    """A who split's children start with the evidence, dates and (restricted)
    who summary the split statistics held for them, so a department's node is
    a confirmed pattern soon after the split; the split is not judged by the
    prune rule before its children are PRUNE_MIN_AGE old."""
    import test_p04_pattern_tree as T4
    from app.engines.behavior import pattern_tree as P4
    sim = Sim(sel=T4.SEL_WHO)
    sim.add(daily(T4._org_day, 21, seed=1))
    first = None
    for d in range(1, 22):
        sim.run_until(MON + d * DAY)
        if sim.splits():
            first = d
            break
    assert first is not None
    tr = sim.tree()
    nid = sim.splits()[0][2]
    sp = tr.nodes[nid].split
    kids = [tr.nodes[c] for c in sp.all_children()]
    seeded = [k for k in kids if k.n_c(sim.now) > 0]
    assert len(seeded) >= 2
    for k in seeded:
        assert k.days_total >= 2
        ips = {str(ip) for ip, *_ in k.who.levels[0].items(sim.now)}
        kid_of = {sp.child_for(f"{ip.rsplit('.', 1)[0]}.0/24") for ip in ips}
        assert kid_of <= {k.id}, (k.id, ips)           # only the child's own addresses (the
        #                                                 parent's who sketch holds its heavy sources)
    # the department's child starts with the evidence of its logins before the
    # split and is a confirmed pattern once 20 units in all have accrued
    # (3 logins a workday; the route node itself learns from day 2: day 12),
    # not 20 units AFTER the split
    ga_kid = tr.nodes[sp.child_for("192.168.1.0/24")]
    assert ga_kid.n_c(sim.now) >= 5.0, ga_kid.n_c(sim.now)
    sim.run_until(MON + 12 * DAY)
    assert ga_kid.state in ("confirmed", "stable", "evolving"), (ga_kid.state, ga_kid.n_c(sim.now))
    assert tr.nodes[nid].split is not None                  # not pruned while young
    assert P4.PRUNE_MIN_AGE >= 7 * DAY


# ------------------------------------------------------------- bindings (P08)
def test_bindings_are_screened_at_the_ip_level_whatever_the_who_arm():
    """P12 chose `prefix` for OA's who; bindings (192.168.1.21 -> jack) are
    about one address and are screened at the IP level anyway; `none` (a
    public portal) screens none."""
    from app.engines.behavior.binding import BindingEngine
    st = make_store()
    for arm, want in (("prefix", (0, 1)), ("ip", (0,)), ("grp", (0, 3)), ("none", ())):
        st.put_model("oa", SYSTEM_ENTITY, "model.sysprof", {"chosen": {"who": arm}})
        assert BindingEngine._who_levels(st, "oa") == want, arm
    st.put_model("oa", SYSTEM_ENTITY, "model.sysprof", {})
    assert 0 in BindingEngine._who_levels(st, "oa")


def test_small_closed_value_field_of_an_action_is_a_target_not_noise():
    """An approval's opinion (3 values, present on 4 % of the events, no context
    predicts which) is a closed-set target; a 4-value attribute on every event
    that nothing predicts stays noise."""
    from app.engines.behavior.lib import pselect as SEL

    def st(cov, card):
        return {"H": 1.2, "cov": cov, "U_t": 0.0, "U_tc": 0.0, "U_s_max": 0.0, "best_levels": [],
                "S": 0.95, "distinct0": 0.001, "CR0": 0.0, "level": 0, "CR": 0.0, "card": {0: card},
                "H_p": 1.2, "n_p": 60, "CR_p": 0.0, "kind": "cat", "cov_rows": cov}
    out = SEL.assign_roles({"body.kv.opinion": st(0.04, 3), "noise.a": st(1.0, 4)}, {}, {}, {},
                           0.0, lambda a: (0,))
    assert out["roles"]["body.kv.opinion"] == "target"
    assert out["roles"]["noise.a"] == "dropped"


def test_scorer_does_not_judge_statements_of_unreproducible_contexts():
    """A pattern defined by a request-size bin is a sub-population the truth
    program does not label: the scorer excludes it from precision (and counts
    it) instead of checking it against every event of the route."""
    from app.eval.pmetrics import LStmt, judgeable_context
    base = {"state": "confirmed", "evidence": {"route": "GET oa /home", "who": {}, "context": []}}
    s1 = LStmt(dict(base, evidence=dict(base["evidence"], context=[["http.route", 0, ["GET oa /home"], False],
                                                                   ["net.src", 1, ["10.0.0.0/24"], False]])),
               "oa", {})
    s2 = LStmt(dict(base, evidence=dict(base["evidence"], context=[["http.route", 0, ["GET oa /home"], False],
                                                                   ["net.bytes_up", 0, ["496.0"], False]])),
               "oa", {})
    assert judgeable_context(s1) and not judgeable_context(s2)


def test_scorer_accepts_group_members_that_are_prefixes():
    """A P11 group over /24 items renders prefixes as members; the grp-alt
    who check of a prefix truth must treat them as networks (it raised)."""
    from app.eval.pmetrics import Who, who_compatible
    truth = {"level": "prefix", "value": [["10.60.0.0/16", 1.0]], "alt": ["grp"]}
    w_in = Who({"level": "grp", "items": ["grp:7"], "members": ["10.60.10.0/24", "10.60.11.7"]}, {})
    w_out = Who({"level": "grp", "items": ["grp:7"], "members": ["10.60.10.0/24", "192.168.1.0/24"]}, {})
    assert who_compatible(truth, w_in)
    assert not who_compatible(truth, w_out)


def test_group_negative_statement_needs_no_write_at_all_in_the_system():
    """'综合部在 X 中从未执行写操作' was emitted when the group's own write node was
    not yet who-closed (pack O: '综合部 在 oa 中从未执行写操作' next to its logins)."""
    import datetime as _d
    from app.engines.behavior.lib import m_ptree as MP
    from app.engines.behavior.lib import pevent as EVT
    from app.models.schema import ORG as _ORG, SYSTEM_ENTITY as _SYS
    tz = _d.timezone(_d.timedelta(hours=8))
    t0 = _d.datetime(2026, 9, 1, tzinfo=tz).timestamp()
    ga = ["192.168.1.21", "192.168.1.23", "10.168.7.121"]
    fin = "192.168.2.10"
    r_fin, r_ga = "POST fin.corp /fin/approval/{num}/approve", "POST fin.corp /fin/expense"

    def keys(ip, g):
        p = ip.split(".")
        return [ip, ".".join(p[:3]) + ".0/24", ".".join(p[:2]) + ".0.0/16", f"grp:{g}", "reg:∅"]

    def build(ga_state):
        m = PT.PTreeModel("finance")
        tr = m.tree(EVT.KIND_TXN, t0, create=True)
        sp = tr.split(tr.root, "http.route", 0, [[r_fin], [r_ga]], t0)
        root, n_fin, n_ga = tr.nodes[tr.root], tr.nodes[sp.children[0]], tr.nodes[sp.children[1]]
        t = t0
        for d in range(21):
            day0 = t0 + d * DAY
            if _d.datetime.fromtimestamp(day0, tz).weekday() >= 5:
                continue
            for k, (ip, g, nd) in enumerate([(fin, "G2", n_fin)] + [(ip, "G1", n_ga) for ip in ga]):
                t = day0 + (10 * 60 + 5 * k) * 60.0
                dayn = int((t + 8 * 3600) // DAY)
                for x in (root, nd):
                    x.update_core(t, 1.0, 1.0, keys(ip, g), ip, 0, (t + 8 * 3600) % DAY / 60.0, dayn)
        root.state = n_fin.state = "confirmed"
        n_ga.state = ga_state
        m.t_last = t
        st = make_store()
        st.put_model("finance", _SYS, MP.PTREE, m, version=1)
        st.put_model(_ORG, _ORG, MP.WHO_GROUPS, {
            "groups": {"G1": {"id": "G1", "name": "综合部", "members": ga, "covers": [], "systems": {"oa": 1.0}},
                       "G2": {"id": "G2", "name": "财务部", "members": [fin], "covers": [],
                              "systems": {"finance": 1.0}}},
            "ip2g": dict({ip: "G1" for ip in ga}, **{fin: "G2"}), "mode": {"finance": {"mode": "ip"}}})
        st.add_batch("finance", EVT.EVT_BATCH, t0 + 21 * DAY, EVT.BatchBuilder("finance").build(t0, t0 + 21 * DAY))
        st.now = t + 3600.0
        return st

    cfg = {"progressive": {"enabled": True}, "tz": "Asia/Shanghai"}
    for state in ("candidate", "confirmed"):
        st = build(state)
        gv = VW.group_view(st, "G1", cfg, st.now)
        neg = [s for s in gv["statements"] if s["evidence"].get("negative")
               and s["evidence"]["target_system"] == "finance"]
        assert not neg, (state, [s["text_zh"] for s in neg])


def test_group_part_of_a_mixed_node_must_exceed_the_other_routes_share():
    """A route-dominant node can hold up to 10 % of other routes; a group whose
    share could be that traffic is not stated as doing the route."""
    assert VW._impurity(None, "GET a /x") == 0.0
    assert abs(VW._impurity({"GET a /x": (92.0, 90.0), "GET a /y": (8.0, 10.0)}, "GET a /x") - 0.08) < 1e-9

    class _Lv:
        def __init__(self, d):
            self.d = d

        def total(self, t):
            return sum(self.d.values())

        def items(self, t):
            return [(k, v, 0, 0) for k, v in self.d.items()]

    class _Who:
        levels = [_Lv({"1.1.1.1": 90.0, "2.2.2.2": 6.0, "3.3.3.3": 4.0}), None, None,
                  _Lv({"grp:A": 90.0, "grp:B": 6.0, "grp:C": 4.0})]

    class _Nd:
        who = _Who()

    class _C:
        ip2g = {"1.1.1.1": "A", "2.2.2.2": "B", "3.3.3.3": "C"}
        mode = "ip"
        groups = {"A": {"members": ["1.1.1.1"], "name": "a"}, "B": {"members": ["2.2.2.2"], "name": "b"},
                  "C": {"members": ["3.3.3.3"], "name": "c"}}

    assert [p[0] for p in VW.group_parts(_C(), _Nd(), 0.0)] == ["A", "B"]
    assert VW.group_parts(_C(), _Nd(), 0.0, impure=0.08) == []      # B's 6 % may be the other routes


def test_a_foreign_source_does_not_become_an_insider_by_persisting():
    """P03's outsider_group test needs the group's standing from OTHER members:
    the IP's own earlier events at the node do not count (A9 slow poisoning)."""
    from app.engines.behavior.conformity import group_outsider
    m = PT.PTreeModel("finance")
    tr = m.tree(EV.KIND_TXN, T0, create=True)
    nd = tr.nodes[tr.root]
    t = T0
    for d in range(10):
        t = T0 + d * DAY + 36000.0
        dayn = int((t + 8 * 3600) // DAY)
        nd.update_core(t, 1.0, 1.0, _keys("192.168.2.10", "FIN"), "192.168.2.10", 0, 600.0, dayn)
        nd.update_core(t + 60, 1.0, 1.0, _keys("192.168.3.33", "SALES"), "192.168.3.33", 0, 630.0, dayn)
    assert f"grp:SALES" in nd.who.levels[3]                 # present, but only through .33 itself
    assert group_outsider(nd, "SALES", "192.168.3.33", t)
    assert not group_outsider(nd, "FIN", "192.168.2.11", t)   # a colleague of the approver
    for d in range(5):
        tt = t + (d + 1) * DAY
        nd.update_core(tt, 1.0, 1.0, _keys("192.168.3.21", "SALES"), "192.168.3.21", 0, 600.0,
                       int((tt + 8 * 3600) // DAY))
    assert not group_outsider(nd, "SALES", "192.168.3.33", t + 6 * DAY)


def test_group_part_lists_only_members_seen_at_the_node():
    class _Lv:
        def __init__(self, d):
            self.d = d

        def total(self, t):
            return sum(self.d.values())

        def items(self, t):
            return [(k, v, 0, 0) for k, v in self.d.items()]

    class _Nd:
        class who:
            levels = [_Lv({"1.1.1.1": 70.0, "1.1.1.2": 10.0}), None, None,
                      _Lv({"grp:A": 80.0, "grp:B": 20.0})]

    class _C:
        ip2g = {"1.1.1.1": "A", "1.1.1.2": "A"}
        mode = "ip"
        groups = {"A": {"members": ["1.1.1.1", "1.1.1.2", "1.1.1.3"], "name": "a"},
                  "B": {"members": ["2.2.2.%d" % i for i in range(19)], "name": "b"}}   # none seen here

    parts = VW.group_parts(_C(), _Nd(), 0.0)
    assert parts == []          # B's key outlived its members at the node: no '19 IPs' part, and A alone is no split
