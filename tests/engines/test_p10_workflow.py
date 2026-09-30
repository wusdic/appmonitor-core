"""P10 Workflow (`behavior.workflow`, docs/lib3/progressive.md §6.14, card P10)."""
from __future__ import annotations

import numpy as np

from helpers import ctx, make_store
from temporal_sim import CFG, DAY, MON, batch, is_workday

from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pdfg as DF
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import ptree as PT
from app.engines.behavior.workflow import WorkflowEngine
from app.models.schema import ORG, SYSTEM_ENTITY

H = "oa.corp.local"


def R(m, p):
    return f"{m} {H} {p}"


LOGIN, HOME = R("POST", "/login"), R("GET", "/home")
LIST, ITEM, APPROVE = R("GET", "/approval/list"), R("GET", "/approval/{num}"), R("POST", "/approval/{num}/approve")
FORM, GEN = R("GET", "/report/form"), R("POST", "/report/generate")
DOCS, DOC = R("GET", "/docs"), R("GET", "/docs/{num}")


def run(st, events, days, eng=None, dt=900.0, cfg=CFG, t_start=MON, learn=None, hooks=()):
    """events: (ts, ip, route, extra attrs); one evt.batch per tick."""
    eng = eng or WorkflowEngine(mine_period_s=6 * 3600)
    events = sorted(events, key=lambda e: e[0])
    now, i = t_start, 0
    while now < t_start + days * DAY:
        t1 = now + dt
        rows = []
        while i < len(events) and events[i][0] <= t1:
            ts, ip, route, extra = events[i]
            rows.append((ts, ip, dict({"http.route": route}, **extra)))
            i += 1
        if rows:
            rows.sort(key=lambda r: r[0])
            lf = [learn(r) for r in rows] if learn is not None else None
            st.add_batch("oa", EV.EVT_BATCH, t1, batch("oa", rows, now, t1, lf))
        for h in hooks:
            h(st, t1)
        eng.safe_run(ctx(st, t1, window_s=dt, config=cfg), None)
        now = t1
    return eng, now


def model(st):
    return MP.get_model(st, "oa", MP.PFLOW)


def edges(st, g="*"):
    return {(e["from"], e["to"]): e for e in model(st)["scopes"].get(g, {}).get("edges", [])}


def approvals(days, seed=0, ip="192.168.1.21"):
    r = np.random.default_rng(seed)
    ev = []
    for d in range(days):
        day = MON + d * DAY
        if not is_workday(day):
            continue
        t = day + r.uniform(540, 561) * 60
        ev.append((t, ip, LOGIN, {}))
        for route in (LIST, ITEM, APPROVE, ITEM, APPROVE):
            t += r.uniform(60, 360)
            ev.append((t, ip, route, {}))
        for k in range(10):                                   # other people's traffic
            t2 = day + r.uniform(570, 990) * 60
            ev.append((t2, f"192.168.3.{k}", DOCS, {}))
            ev.append((t2 + r.uniform(5, 60), f"192.168.3.{k}", DOC, {}))
    return ev


def test_login_approval_list_approve_mined():
    """(card P10) login -> approval list -> item -> approve (1-6 min apart) is
    mined with dep >= 0.8 and its delay bands; the item <-> approve loop (the
    next item after an approval) is recognised as a length-two loop."""
    st = make_store()
    run(st, approvals(21), 21)
    E = edges(st)
    for a, b in ((LOGIN, LIST), (LIST, ITEM), (ITEM, APPROVE)):
        e = E[(a, b)]
        assert e["dep"] >= 0.8 and e["count"] >= 10, e
        assert e["band"][0] <= 360 and e["band"][1] >= 60, e
    assert E[(ITEM, APPROVE)]["loop2"] is not None
    flows = [w["routes"] for w in model(st)["scopes"]["*"]["workflows"]]
    assert any(w[:4] == [LOGIN, LIST, ITEM, APPROVE] for w in flows), flows
    assert (DOCS, DOC) in E


def reports(days, ips=("192.168.1.23", "10.168.7.121", "192.168.1.30"), seed=1, skip_form=()):
    r = np.random.default_rng(seed)
    ev = []
    for d in range(days):
        for h0 in (600, 1020):                               # two report sessions a day, hours apart
            for ip in ips:
                t = MON + d * DAY + (h0 + r.uniform(0, 10)) * 60
                if (d, h0, ip) not in skip_form:
                    ev.append((t, ip, FORM, {}))
                ev.append((t + r.uniform(60, 240), ip, GEN, {}))
    return ev


def test_required_predecessor_after_15_sessions():
    """(card P10) after ~15 consistent sessions POST /report/generate requires
    GET /report/form (Jeffreys 5 % quantile >= 0.85), not before; a session
    with the submission only then scores p_req <= 0.05."""
    st = make_store()
    eng, now = run(st, [e for e in reports(3) if e[0] < MON + DAY], 1)
    assert not model(st)["scopes"]["*"]["requires"] or \
        all(r["to"] != GEN for r in model(st)["scopes"]["*"]["requires"])   # 6 sessions: not yet
    run(st, [e for e in reports(3) if e[0] >= MON + DAY], 2, eng=eng, t_start=MON + DAY)
    req = [r for r in model(st)["scopes"]["*"]["requires"] if r["to"] == GEN]
    assert req and req[0]["from"] == FORM and req[0]["lb"] >= 0.85 and req[0]["c_b"] >= 15
    m = model(st)
    t = MON + 3 * DAY
    alone = DF.seq_scores(m, "*", None, GEN, 0, t)
    assert alone["p_req"] <= 0.05 and alone["p_seq"] <= 0.1
    ok = DF.seq_scores(m, "*", FORM, GEN, DF.bloom_bits(DF.h64(FORM)), t)
    assert not ok["p_req"] == ok["p_req"]                    # NaN: nothing required is missing


def test_transition_p_value():
    st = make_store()
    run(st, approvals(28) + reports(28), 28)
    m = model(st)
    t = MON + 28 * DAY
    good = DF.seq_scores(m, "*", FORM, GEN, DF.bloom_bits(DF.h64(FORM)), t)
    odd = DF.seq_scores(m, "*", FORM, APPROVE, DF.bloom_bits(DF.h64(FORM)), t)
    assert good["p_trans"] > 0.3 and odd["p_trans"] < 0.05, (good, odd)
    assert m["gain"]["bits_per_event"] > 1.0                  # the DFG predicts the next action


def interleaved(days, key: bool, seed=2):
    """Two users behind one NAT address running different workflows at the
    same time: A B C and X Y Z, interleaved event by event."""
    r = np.random.default_rng(seed)
    ev = []
    for d in range(days):
        for s0 in range(4):
            t = MON + d * DAY + (540 + 60 * s0) * 60
            for j, (u, v) in enumerate(zip((R("GET", "/a"), R("GET", "/b"), R("GET", "/c")),
                                          (R("GET", "/x"), R("GET", "/y"), R("GET", "/z")))):
                ev.append((t + 120 * j, "10.1.1.1", u, {"sess.key": "k1"} if key else {}))
                ev.append((t + 120 * j + r.uniform(20, 60), "10.1.1.1", v, {"sess.key": "k2"} if key else {}))
    return ev


def test_nat_interleaving_and_session_keys():
    """(card P10) interleaved sessions behind one address destroy the true
    dependencies (documented limitation); with the address flagged shared
    (P11 / B17) and per-user sess.key they are separated again."""
    st = make_store()
    run(st, interleaved(7, key=False), 7)
    E = edges(st)
    assert (R("GET", "/a"), R("GET", "/b")) not in E
    st2 = make_store()
    st2.put_model(ORG, ORG, MP.WHO_GROUPS, {"shared": ["10.1.1.1"]})
    run(st2, interleaved(7, key=True), 7)
    E2 = edges(st2)
    for a, b in (("/a", "/b"), ("/b", "/c"), ("/x", "/y"), ("/y", "/z")):
        assert (R("GET", a), R("GET", b)) in E2, sorted(E2)


def test_quarantined_ip_adds_nothing_and_counting_waits_for_D():
    st = make_store()

    def quarantine(s, t):
        s.add_vec("oa", "192.168.1.21", "behavior.quarantine", t, [1.0])

    eng, _ = run(st, approvals(3), 3, hooks=[quarantine])
    m = model(st)
    assert m["stats"]["quarantined"] > 0
    assert all("approval" not in e["to"] for e in m["scopes"]["*"]["edges"])
    ids = {DF.split_key(k)[0] for k in m.state.acts.k2i}
    assert LOGIN not in ids and LIST not in ids                 # the IP's actions were never learned
    # counting is delayed by D = max(4 ticks, 600 s)
    st2 = make_store()
    eng2 = WorkflowEngine()
    b = batch("oa", [(MON + 10, "1.1.1.1", {"http.route": LOGIN}), (MON + 20, "1.1.1.1", {"http.route": HOME})],
              MON, MON + 900)
    st2.add_batch("oa", EV.EVT_BATCH, MON + 900, b)
    for k in range(1, 5):
        eng2.safe_run(ctx(st2, MON + 900 * k, window_s=900, config=CFG), None)
        assert model(st2)["stats"]["counted"] == 0
    eng2.safe_run(ctx(st2, MON + 900 * 5, window_s=900, config=CFG), None)
    assert model(st2)["stats"]["counted"] == 2


def test_unlearned_rows_shape_sessions_but_are_not_counted():
    st = make_store()
    ev = approvals(21)
    run(st, ev, 21, learn=lambda r: r[2]["http.route"] != LIST)   # the list rows are thinned out
    m = model(st)
    assert m["stats"]["counted"] == m["stats"]["rows"] - sum(1 for e in ev if e[2] == LIST)
    E = edges(st)
    assert (LIST, ITEM) in E                     # its successor was learned with its true predecessor
    assert (LOGIN, ITEM) not in E                # the thinned row still separates login from the item


def test_evicted_action_ids_are_never_reused():
    st = make_store()
    eng = WorkflowEngine(mine_period_s=3600)
    m = DF.PFlowModel()
    m.state.acts = DF.ActionDict(k=8)
    st.put_model("oa", SYSTEM_ENTITY, MP.PFLOW, m)
    r = np.random.default_rng(3)
    ev = [(MON + 600 + 30 * i, f"10.0.0.{i % 5}", R("GET", f"/p{int(r.integers(0, 40))}"), {}) for i in range(2000)]
    run(st, ev, 1, eng=eng)
    acts = m.state.acts
    assert acts.retired > 0 and len(acts.k2i) <= 8
    live = set(acts.i2k)
    assert max(live) < acts.next_id and len(live) == len(acts.k2i)
    for k in m.state.edges.keys():
        assert k[1] in live and k[2] in live                   # no statistic of a retired id survives


def test_group_scopes():
    """Per-group workflows: GA does A -> B, FIN does A -> C (scope '*' holds both)."""
    st = make_store()
    ga, fin = ["192.168.1.21", "192.168.1.23"], ["192.168.2.10", "192.168.2.11"]
    st.put_model(ORG, ORG, MP.WHO_GROUPS, {"ip2g": {**{ip: 1 for ip in ga}, **{ip: 2 for ip in fin}}})
    r = np.random.default_rng(4)
    ev = []
    for d in range(10):
        for ip in ga + fin:
            t = MON + d * DAY + r.uniform(540, 600) * 60
            ev.append((t, ip, R("GET", "/a"), {}))
            ev.append((t + 60, ip, R("GET", "/b") if ip in ga else R("GET", "/c"), {}))
    run(st, ev, 10)
    sc = model(st)["scopes"]
    assert set(sc) >= {"*", "1", "2"}
    assert (R("GET", "/a"), R("GET", "/b")) in edges(st, "1") and (R("GET", "/a"), R("GET", "/c")) not in edges(st, "1")
    assert (R("GET", "/a"), R("GET", "/c")) in edges(st, "2")
    assert {(R("GET", "/a"), R("GET", "/b")), (R("GET", "/a"), R("GET", "/c"))} <= set(edges(st, "*"))


def test_action_variants_from_content_splits():
    """An action variant = the node reached through a content split below
    the route (a POST /login answered 200 vs 302 are different actions)."""
    st = make_store()
    ptm = PT.PTreeModel("oa")
    tree = ptm.tree(EV.KIND_TXN, MON)
    sp = tree.split(tree.root, "http.route", 0, [[LOGIN]], MON)
    tree.split(sp.children[0], "http.status", 0, [[302]], MON)
    st.put_model("oa", SYSTEM_ENTITY, MP.PTREE, ptm)
    ev = []
    for d in range(3):
        for k in range(20):
            t = MON + d * DAY + 36000 + 300 * k
            ok = k % 2 == 0
            ev.append((t, f"10.0.0.{k}", LOGIN, {"http.status": 302 if ok else 200}))
            ev.append((t + 5, f"10.0.0.{k}", HOME if ok else LOGIN, {"http.status": 200}))
    run(st, ev, 3)
    keys = set(model(st).state.acts.k2i)
    assert any(k.startswith(LOGIN + "#v") for k in keys) and len([k for k in keys if k.startswith(LOGIN)]) == 2


def test_ctx_session_mode_follows_p01_ids():
    st = make_store()
    eng = WorkflowEngine(session_mode="ctx")
    rows = [(MON + 10, "1.1.1.1", {"http.route": LOGIN}), (MON + 20, "1.1.1.1", {"http.route": HOME}),
            (MON + 30, "1.1.1.1", {"http.route": DOCS})]
    b = batch("oa", rows, MON, MON + 900)
    st.add_batch("oa", EV.EVT_BATCH, MON + 900, b)
    cb = b.aligned(EV.cols_from_rows(3, [{"ctx.sid": "s1"}, {"ctx.sid": "s1"}, {"ctx.sid": "s2"}]))
    st.add_batch("oa", EV.EVT_CTX, MON + 900, cb)
    eng.safe_run(ctx(st, MON + 900, window_s=900, config=CFG), None)
    pend = model(st).state.pending[("oa", MON + 900)]
    starts = [a[6] for a in pend if a[0] == "row"]
    assert starts == [True, False, True]


def test_arm_off_and_disabled():
    st = make_store()
    st.put_model("oa", SYSTEM_ENTITY, MP.SYSPROF, {"chosen": {"p10": "off"}})
    run(st, approvals(2), 2)
    assert model(st) is None
    st2 = make_store()
    run(st2, approvals(2), 2, cfg={"tz": "Asia/Shanghai"})
    assert model(st2) is None
