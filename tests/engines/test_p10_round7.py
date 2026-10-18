"""P10, round 7 (evaluator; the conformity owner's OC7-1): the members of a
replica family share one tree key and one session state, so their rows are
taken in one time order. Run member by member, a session that moved between
the replicas within a tick was read out of order (O-real R9: the report form
served by oa-r2 at 17:03, the generate served by oa at 17:06 of the same tick
were seen generate-first), and P10 never held 'generate requires form'
(O-real A5 missed on every seed). Fails on the round-6 / HEAD code."""
from __future__ import annotations

import numpy as np
from temporal_sim import CFG, DAY, MON, batch, is_workday
from test_p10_workflow import DOC, DOCS, FORM, GEN

from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pdfg as DF
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.workflow import WorkflowEngine
from app.models.schema import ORG
from helpers import ctx, make_store

FAM = "fam:1"
MEMBERS = ("oa", "oa-r2")


def run_family(st, events, days, dt=900.0, t_start=MON):
    """events: (ts, ip, route, member); one evt.batch per member and tick."""
    st.put_model(ORG, ORG, MP.SYSFAM, {"member": {m: FAM for m in MEMBERS}}, version=1, ts=t_start)
    eng = WorkflowEngine(mine_period_s=6 * 3600)
    events = sorted(events, key=lambda e: e[0])
    now, i = t_start, 0
    while now < t_start + days * DAY:
        t1 = now + dt
        rows = {m: [] for m in MEMBERS}
        while i < len(events) and events[i][0] <= t1:
            ts, ip, route, m = events[i]
            rows[m].append((ts, ip, {"http.route": route}))
            i += 1
        for m in MEMBERS:
            if rows[m]:
                st.add_batch(m, EV.EVT_BATCH, t1, batch(m, rows[m], now, t1))
        eng.safe_run(ctx(st, t1, window_s=dt, config=CFG), None)
        now = t1
    return eng


def balanced(days, users=6, seed=3):
    """A load balancer sends the report form to oa-r2 and the generate, a few
    minutes later in the same 15-min tick, to oa; document reading spreads
    over both members."""
    r = np.random.default_rng(seed)
    ev = []
    for d in range(days):
        day = MON + d * DAY
        if not is_workday(day):
            continue
        for u in range(users):
            ip = f"192.168.1.{20 + u}"
            t = day + (17 * 60 + r.uniform(0, 3)) * 60          # 17:00-17:03, tick ends 17:15
            ev.append((t, ip, FORM, "oa-r2"))
            ev.append((t + r.uniform(60, 300), ip, GEN, "oa"))
            t = day + r.uniform(560, 900) * 60
            ev.append((t, ip, DOCS, MEMBERS[u % 2]))
            ev.append((t + r.uniform(5, 60), ip, DOC, MEMBERS[(u + 1) % 2]))
    return ev


def test_a_family_session_crossing_replicas_within_a_tick_is_read_in_time_order():
    st = make_store()
    run_family(st, balanced(14), 14)
    model = MP.get_model(st, FAM, MP.PFLOW)
    E = {(e["from"], e["to"]): e for e in model["scopes"].get("*", {}).get("edges", [])}
    e = E.get((FORM, GEN))
    assert e is not None and e["dep"] >= DF.DEP_MIN, sorted(E)
    # member by member (oa before oa-r2) the generate was read first and the
    # form after it, in the same session: an edge generate -> form
    assert (GEN, FORM) not in E
    req = [(q["a"], q["b"]) for q in model["scopes"]["*"].get("requires", [])]
    assert any(a == FORM and b == GEN for a, b in req) or any(
        q.get("from") == FORM and q.get("to") == GEN for q in model["scopes"]["*"].get("requires", [])), req


def test_every_members_batches_are_consumed_once():
    st = make_store()
    ev = balanced(3)
    eng = run_family(st, ev, 3)
    model = MP.get_model(st, FAM, MP.PFLOW)
    st_ = model.state
    assert st_.n_rows == len(ev)
    assert set(st_.last_batch) == set(MEMBERS)
    assert eng is not None
