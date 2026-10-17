"""P03 round 7: family-merged session view, calibration under accepted
regime changes, and the batch scoring paths (conformity / pscore owner)."""
from __future__ import annotations

from helpers import ctx

import test_p03_conformity as T3

from app.engines.behavior import conformity as CF
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.models.schema import ORG, SYSTEM_ENTITY

FORM, GEN = "GET oa /report/form", "POST oa /report/generate"
FAM = "fam:1"


def _family_fixture():
    """oa and oa-r2 are one engine family (P12: one tree key, one P10 model);
    P10 learned 'generate requires form' from sessions on oa."""
    from app.engines.behavior.workflow import WorkflowEngine
    fx = T3.Fx()
    fx.st.put_model(ORG, ORG, MP.SYSFAM, {"member": {"oa": FAM, "oa-r2": FAM},
                                          "families": {FAM: {"members": ["oa", "oa-r2"]}}})
    for d in T3.workdays(20):
        for ip in T3.GA[1:]:
            fx.learn(FAM, GEN, ip, d + 17 * 3600 + 120)
    fx.tree(FAM, [GEN, FORM])
    p10 = WorkflowEngine(mine_period_s=6 * 3600.0)
    days = T3.workdays(21)
    for d in days[:20]:
        for h in range(24):
            t1 = d + (h + 1) * 3600.0
            if h == 17:
                b = EV.BatchBuilder("oa", EV.KIND_TXN)
                for ip in T3.GA[1:]:
                    b.add(d + 17 * 3600 + 10, ip, {"http.route": FORM, "net.src": ip})
                    b.add(d + 17 * 3600 + 120, ip, {"http.route": GEN, "net.src": ip})
                fx.st.add_batch("oa", EV.EVT_BATCH, t1, b.build(t1 - 3600, t1))
            fx.st.ensure_retention("evt.", max_age_s=6 * 3600.0)
            p10.safe_run(ctx(fx.st, t1, window_s=3600.0, config=T3.CFG), None)
    flow = fx.st.get_model(FAM, SYSTEM_ENTITY, MP.PFLOW)
    assert any(r["to"] == GEN and r["from"] == FORM for r in flow["scopes"]["*"]["requires"])
    return fx, days


def _score_tick(fx, per_system, dt=3600.0):
    """One tick holding a batch per system; P03 runs once."""
    t1 = max(e[0] for evs in per_system.values() for e in evs) + 1.0
    for system, events in per_system.items():
        b = EV.BatchBuilder(system, EV.KIND_TXN)
        for ts, ip, attrs in sorted(events, key=lambda e: e[0]):
            a = {"net.src": ip, "ctx.tod_min": T3.local_minute(ts), "ctx.daytype": "workday"}
            a.update(attrs)
            b.add(ts, ip, a)
        fx.st.add_batch(system, EV.EVT_BATCH, t1, b.build(t1 - dt, t1))
    fx.eng.safe_run(ctx(fx.st, t1, window_s=dt, config=T3.CFG), None)
    return t1


def test_a_workflow_spread_over_a_replica_pair_is_one_session():
    """The load balancer serves a user's form page from oa-r2 and the report
    generation from oa within one tick. The members were scored one system
    after the other with a per-batch overlay, so oa's POST read 'without the
    required predecessor' (O-real seed 2: ~20 MEDIUM missing_predecessor
    incidents on SALES users across oa / oa-r2). The family is one session
    view: scored in time order across its members."""
    fx, days = _family_fixture()
    t = days[20] + 17 * 3600 + 120
    _score_tick(fx, {"oa": [(t, "192.168.1.23", {"http.route": GEN, "http.method": "POST"})],
                     "oa-r2": [(t - 100, "192.168.1.23", {"http.route": FORM})]})
    v = [e for e in fx.violations("192.168.1.23") if e.extra["type"] == "seq"]
    assert not v, [e.description for e in v]
    asg = fx.st.batch_at("oa", EV.PAT_ASSIGN, t + 1.0)
    assert asg is not None and asg.get("prev_act", 0) == FORM
    assert fx.st.batch_at("oa-r2", EV.PAT_ASSIGN, t + 1.0) is not None
    # the requirement still holds across the family: generation with no form anywhere
    t2 = t + 7200
    _score_tick(fx, {"oa": [(t2, "192.168.1.21", {"http.route": GEN, "http.method": "POST"})],
                     "oa-r2": [(t2 - 100, "192.168.1.23", {"http.route": FORM})]})
    assert [e for e in fx.violations("192.168.1.21") if e.extra["type"] == "seq"]


def test_a_single_system_scores_as_before():
    """Systems outside a family keep one overlay per batch (scores unchanged)."""
    fx, days = _family_fixture()
    st = fx.st.get_model(ORG, ORG, CF.STATE)
    assert st is None or isinstance(st, CF.ConfState)
    t = days[20] + 17 * 3600 + 120
    _score_tick(fx, {"oa": [(t - 100, "192.168.1.23", {"http.route": FORM}),
                            (t, "192.168.1.23", {"http.route": GEN, "http.method": "POST"})]})
    assert not [e for e in fx.violations("192.168.1.23") if e.extra["type"] == "seq"]


# ------------------------------------------------------------------ OI7-2
PLOGIN = "POST portal /login"


def _sat(n):
    """n Saturdays (local) from T0 on."""
    out, d = [], 0
    while len(out) < n:
        t = T3.T0 + d * T3.DAY
        if T3._dt.datetime.fromtimestamp(t, T3.TZ).weekday() == 5:
            out.append(t)
        d += 1
    return out


def _content_child_fixture():
    """The portal login node's population logs in from 07:00 to 23:00 on
    non-workdays (evening slots included); P04 split it on a CONTENT attribute
    (net.pkts_down) three non-workdays ago, so the child's own non-workday
    sample holds 360 logins, all between 13:00 and 15:00."""
    from app.engines.behavior.lib import pnode as PN
    fx = T3.Fx()
    tr = fx.tree("portal", [PLOGIN])
    login = fx.node("portal", PLOGIN)
    hier = MP.hierarchies(fx.st, "portal", T3.CFG)
    sats = _sat(12)
    k = 0

    def learn(nodes, ts, ip):
        keys = [hier.gen("net.src", l, ip) for l in range(PN.WHO_LEVELS)]
        day = int((ts + 8 * 3600) // T3.DAY)
        for nd in nodes:
            nd.update_core(ts, 1.0, 1.0, keys, ip, 1, T3.local_minute(ts), day)
    for d in sats[:9]:
        for m in range(7 * 60, 23 * 60, 8):                 # 07:00-23:00, all evening slots used
            k += 1
            learn([tr.nodes[tr.root], login], d + m * 60.0, f"10.61.{k % 50}.{k % 200}")
    tr.split(login.id, "net.pkts_down", 0, [[2]], sats[8])
    child = tr.nodes[tr.nodes[login.id].split.child_for(2)]
    for d in sats[9:12]:
        for m in range(13 * 60, 15 * 60):                   # the child's own sample: 13:00-15:00
            k += 1
            ip = f"10.61.{k % 50}.{k % 200}"
            learn([tr.nodes[tr.root], login, child], d + m * 60.0, ip)
    for nd in tr.nodes.values():
        if nd.state == "candidate":
            nd.state = "confirmed"
    fx.st.put_model("portal", SYSTEM_ENTITY, MP.PTREE, MP.get_ptree(fx.st, "portal"), version=2)
    return fx, sats, login, child


def test_a_content_split_child_is_no_rarer_in_time_than_its_parent():
    """OI7-2 (pack O seed 1, D4): a young content-split child's own sparse
    non-workday counts set the 'when' p of a slot its parent's population
    uses (max(p_hier, min(p_own, p_parent)) let p_own decide: 5.8e-4 for a
    PUB visitor's 21:45 Saturday login). A content split is not a time
    sub-population: the child's p is at least its parent's."""
    fx, sats, login, child = _content_child_fixture()
    t = sats[11] + 7 * T3.DAY + 21 * 3600 + 45 * 60       # the next Saturday, 21:45
    a = {"net.src": "10.61.39.151", "http.route": PLOGIN, "http.method": "POST", "net.pkts_down": 2,
         "ctx.tod_min": T3.local_minute(t), "ctx.daytype": "nonworkday"}
    b, asg = fx.score("portal", [(t, "10.61.39.151", a)])
    assert asg.get("conf", 0) == child.id
    p = asg.get("p_when", 0)
    assert p >= 1e-2, p
    # a slot the PARENT never saw stays rare at the child (A4-type departures)
    t2 = sats[11] + 14 * T3.DAY + 3 * 3600 + 5 * 60        # 03:05
    a2 = dict(a, **{"ctx.tod_min": T3.local_minute(t2)})
    _, asg2 = fx.score("portal", [(t2, "10.61.39.152", a2)])
    assert asg2.get("p_when", 0) <= 1e-3, asg2.get("p_when", 0)


# ------------------------------------------------------- intensity (A7, O-real)
def _score_w(fx, system, events, dt=3600.0):
    """Fx.score with each row's event mass (evt.batch `w`: a sampled sensor's
    rows carry the sample rate)."""
    t1 = max(e[0] for e in events) + 1.0
    b = EV.BatchBuilder(system, EV.KIND_TXN)
    for ts, ip, attrs, w in sorted(events, key=lambda e: e[0]):
        a = {"net.src": ip, "ctx.tod_min": T3.local_minute(ts), "ctx.daytype": "workday"}
        a.update(attrs)
        b.add(ts, ip, a, w=w)
    fx.st.add_batch(system, EV.EVT_BATCH, t1, b.build(t1 - dt, t1))
    fx.now = t1
    fx.eng.safe_run(ctx(fx.st, t1, window_s=dt, config=T3.CFG), None)
    return t1


def _sampled_portal():
    import numpy as np
    fx = T3.Fx()
    route = "POST portal /login"
    rng = np.random.default_rng(5)
    for d in T3.workdays(8):
        for i in range(50):
            fx.learn("portal", route, f"10.60.{i}.{int(rng.integers(1, 250))}", d + (10 + rng.random()) * 3600)
    nd = fx.node("portal", route)
    fx.put("portal", MP.PBOUNDS, nd.id, attrs={"rate.ip_h": {
        "kind": "num", "band90": [4.0, 4.0], "band98": [4.0, 7.8], "qgrid": [4.0] * 31 + [4.2, 8.0],
        "range": None, "n_c": 774.0, "mass": 397.0, "hard": False, "log": False}})
    fx.eng = CF.ConformityEngine(k_int=4096)
    t0 = T3.workdays(9)[-1]
    for h in range(36):            # 36 finished hours x 75 one-login visitors, each row = 4 logins' mass
        th = t0 + h * 3600.0
        _score_w(fx, "portal", [(th + i * 40.0, f"10.61.{(h * 75 + i) // 250 % 250}.{(h * 75 + i) % 250}",
                                 {"http.route": route, "http.method": "POST"}, 4.0) for i in range(75)])
    return fx, route, t0


def test_a_sampled_sensors_hour_count_is_the_ips_events_not_its_rows():
    """O-real A7 (R12: the portal sensor keeps 1 event in 4, each row's mass
    is 4): 400 logins in one hour arrive as 100 rows. Counted in rows, the
    burst was 100 against one-row IP-hours - Cantelli p 1.02e-4, a LOW
    finding, no incident on any seed. The hour count is the IP's events
    (the rows' mass), as every IP-hour of the reference is."""
    fx, route, t0 = _sampled_portal()
    t = t0 + 37 * 3600.0
    _score_w(fx, "portal", [(t + i * 20.0, "10.60.7.7", {"http.route": route, "http.method": "POST"}, 4.0)
                            for i in range(100)])
    v = [e for e in fx.violations("10.60.7.7") if e.extra["type"] == "content" and "intensity" in e.extra["flags"]]
    assert v and max(CF.SEV_RANK[e.severity] for e in v) >= CF.SEV_RANK[CF.Severity.MEDIUM], \
        [(e.severity, e.extra["p"]) for e in v]
    assert not [e for e in fx.violations() if e.entity != "10.60.7.7" and "intensity" in e.extra["flags"]]


def test_the_hour_count_enters_the_tick_score_once():
    """The hour count is one cumulative statistic: the tick's conf_content
    score of the bursting IP carries its p (x2 for the combination with the
    events' own content p), not 1 - (1 - p)^n over the tick's n events."""
    fx, route, t0 = _sampled_portal()
    t = t0 + 37 * 3600.0
    t1 = _score_w(fx, "portal", [(t + i * 20.0, "10.60.7.7", {"http.route": route, "http.method": "POST"}, 4.0)
                                 for i in range(100)])
    rec = fx.st.latest_derived("portal", "10.60.7.7", CF.CONF_SERIES)
    assert rec is not None and rec.ts == t1
    v = [e for e in fx.violations("10.60.7.7") if "intensity" in e.extra["flags"]]
    p_int = min(e.extra["p"] for e in v)
    c = rec.value["conf_content"]
    assert c["p"] <= 2.0 * p_int * 1.0001, (c, p_int)


# --------------------------------------------- OI-f (tree owner): mixture content
def test_a_new_actions_content_is_not_judged_against_the_route_mixture():
    """Tree owner OI-f (pack O, GA.oa.report#1): POST /report/generate's node
    is a young candidate under oa's route partition; the event is covered by
    the partition's node, whose numeric records describe the OTHER actions
    (login / home bodies 0.2-1.5 KB). Its 50 KB body was a content violation
    (p ~1e-9) and the row was learned damped (0.1) - the new action's node
    then took ~10x longer to confirm. The mixture does not model a specific
    new action's content; bindings and who still apply."""
    fx = T3.Fx()
    tr = fx.tree("oa", [T3.LOGIN, GEN])
    for d in T3.workdays(12):
        for j, ip in enumerate(T3.GA):
            fx.learn("oa", T3.LOGIN, ip, d + (9 * 60 + 3 * j) * 60.0, targets={"body.len": 900.0 + 50 * j})
    root = tr.nodes[tr.root]
    root.state = "confirmed"
    gen_nd = tr.nodes[root.split.child_for(GEN)]
    gen_nd.state = "candidate"
    fx.put("oa", MP.PBOUNDS, root.id, attrs={"body.len": {
        "kind": "num", "band90": [300.0, 1400.0], "band98": [200.0, 1500.0],
        "qgrid": [200.0, 400.0, 900.0, 1200.0, 1500.0], "range": [200.0, 1500.0], "n_c": 2000.0,
        "hard": True, "cover": 1e-4, "log": False}})
    t = T3.workdays(13)[-1] + 17 * 3600
    _, asg = fx.score("oa", [(t, "192.168.1.23", {"http.route": GEN, "http.method": "POST", "body.len": 50000.0})])
    assert asg.get("conf", 0) == root.id
    import math
    assert math.isnan(asg.get("p_content", 0)), asg.get("p_content", 0)
    assert asg.get("damp", 0) == 1.0
    assert not [e for e in fx.violations("192.168.1.23") if e.extra["type"] == "content"]
    # the partition's records still judge an event of a confident action child's sibling: the
    # login action itself (its own node confirmed) is covered by its node, not the root
    _, asg2 = fx.score("oa", [(t + 60, "192.168.1.23", {"http.route": T3.LOGIN, "http.method": "POST",
                                                        "body.len": 50000.0})])
    assert asg2.get("conf", 0) != root.id


# ------------------------------------- 'when' calibration under drift (O-red A4)
def _regime_fixture(accept: bool):
    """The 综合部 login node: 30 workdays of logins 09:02-09:25 (P09 window
    09:00-09:22), then D1 moves the window to 08:10-08:35 for three workdays,
    scored before they are learned; P09 then (optionally) accepts the new
    regime; three more workdays of new-regime logins; A4 at 03:05."""
    fx = T3.Fx()
    login = "POST oa /login"
    days = T3.workdays(40)
    for d in days[:30]:
        for j, ip in enumerate(T3.GA):
            for k in range(2):
                fx.learn("oa", login, ip, d + (9 * 60 + 2 + 7 * j + 3 * k) * 60.0)
    nd = fx.node("oa", login)

    def wins(lo, hi, since=None):
        rec = {"since": since, "regime": "new", "provisional": False} if since else {"regime": None}
        fx.st.put_model("oa", SYSTEM_ENTITY, MP.PWIN, {"fmt": 1, "version": 1, "nodes": {0: {nd.id: {
            "status": "fitted", "when": {"workday": [[lo, hi]], "nonworkday": []},
            "by_daytype": {"wd": rec}}}}})
    wins(540, 562)
    for i, d in enumerate(days[30:36]):
        if i == 3 and accept:
            wins(490, 515, since=days[30])           # P09 accepts D1's window on its dates
        for j, ip in enumerate(T3.GA):
            for k in range(2):
                ts = d + (8 * 60 + 11 + 7 * j + 4 * k) * 60.0
                fx.score("oa", [(ts, ip, {"http.route": login, "http.method": "POST"})])
                fx.learn("oa", login, ip, ts)
    n_drift = len([e for e in fx.violations() if e.extra["type"] == "when"])
    ts = days[36] + (3 * 60 + 5) * 60.0
    _, asg = fx.score("oa", [(ts, T3.GA[2], {"http.route": login, "http.method": "POST"})])
    return fx, asg, n_drift


def test_an_accepted_regime_change_segments_the_when_calibration():
    """O-red seed 0, A4: the login node's 'when' calibration took D1's
    out-of-window logins (scored against a density that had not learned the
    new window) as its normal and mapped a 03:05 login's p_when 4.7e-4 to
    0.036 - no finding. Once P09 accepts the new regime the calibration
    restarts (scores before the acceptance are not this regime's exceedance
    frequency), and arrivals inside P09's windows are calibrated apart from
    those outside them (a lagging density's in-window scores mask nothing)."""
    fx, asg, n_drift = _regime_fixture(accept=True)
    v = [e for e in fx.violations(T3.GA[2]) if e.extra["type"] == "when" and e.extra["event_ts"] == asg.ts[0]]
    assert asg.get("p_when", 0) <= 1e-3, asg.get("p_when", 0)
    assert v and v[0].severity == CF.Severity.MEDIUM
    # the drift itself and the new regime's logins: at most one finding (its first login)
    assert n_drift <= 1, n_drift
    st = fx.st.get_model(ORG, ORG, CF.STATE)
    assert any(k[-1] == CF.WHEN_OUT for k in st.cal.h._d)


def test_a_nodes_first_extreme_score_is_not_its_routine():
    """The node-routine test (>= E_MAX such scores a day) read the
    calibration AFTER the event's own score was added: a key the event had
    just created held one unit over the 1-hour age floor, ~24 a day, so the
    first extreme 'when' (or content) score of a young node was routine and
    never a finding (O-red: the 综合部 login node was created on D1's day)."""
    fx = T3.Fx()
    import numpy as np
    rng = np.random.default_rng(1)
    days = T3.workdays(50)
    for d in days[:45]:
        for ip in T3.GA:
            fx.learn("oa", T3.LOGIN, ip, d + (9 * 60 + 21 * rng.random()) * 60.0)
    nd = fx.node("oa", T3.LOGIN)
    fx.put("oa", MP.PWIN, nd.id, attrs={})
    fx.st.get_model("oa", SYSTEM_ENTITY, MP.PWIN)["nodes"][0][nd.id]["when"] = {
        "workday": [[540, 561]], "nonworkday": []}
    t = days[48] + 3 * 3600 + 5 * 60                              # 03:05, the node's first scored event
    _, asg = fx.score("oa", [(t, "10.168.7.121", {"http.route": T3.LOGIN, "http.method": "POST"})])
    assert asg.get("p_when", 0) <= 1e-3
    w = [e for e in fx.violations("10.168.7.121") if e.extra["type"] == "when"]
    assert w and w[0].severity == CF.Severity.MEDIUM
