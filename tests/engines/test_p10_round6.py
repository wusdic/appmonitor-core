"""P10, round 6 (time / workflow owner): an address whose session keys
interleave - several clients behind one SNAT proxy that neither P11 / B17
nor P12 flagged - has its sessions told apart by key (lib/pdfg
FlowState.track_key), so the workflows behind it are learned; one user's or
a health checker's keys never mark an address. Fails on the round-5 code."""
from __future__ import annotations

import numpy as np
from temporal_sim import DAY, MON, is_workday
from test_p10_workflow import DOC, DOCS, FORM, GEN, R, edges, run

from app.engines.behavior.lib import pdfg as DF
from helpers import make_store

PROXY = "192.168.100.99"
HEALTH = R("GET", "/health")


def proxied(days, users=6, seed=5, health_keys=True):
    """O-real R1: the reporters and colleagues reach OA through one proxy
    address, a monitor checks /health through it every 60 s (a fresh cookie
    per request), every user session has its own cookie."""
    r = np.random.default_rng(seed)
    ev = []
    for d in range(days):
        day = MON + d * DAY
        if not is_workday(day):
            continue
        t = day + 8 * 3600
        n = 0
        while t < day + 18 * 3600:
            ev.append((t, PROXY, HEALTH, {"sess.key": f"h{d}-{n}"} if health_keys else {}))
            t += 60.0
            n += 1
        for u in range(users):
            k = f"u{u}-{d}"
            t = day + (1020 + r.uniform(0, 10)) * 60
            ev.append((t, PROXY, FORM, {"sess.key": k + "r"}))
            ev.append((t + r.uniform(60, 240), PROXY, GEN, {"sess.key": k + "r"}))
            for s in range(2):
                t = day + r.uniform(560, 960) * 60
                ev.append((t, PROXY, DOCS, {"sess.key": f"{k}d{s}"}))
                ev.append((t + r.uniform(5, 60), PROXY, DOC, {"sess.key": f"{k}d{s}"}))
    return ev


def test_workflows_behind_an_unflagged_proxy_are_learned():
    st = make_store()
    run(st, proxied(14), 14)
    E = edges(st)
    e = E.get((FORM, GEN))
    assert e is not None and e["dep"] >= DF.DEP_MIN, sorted(E)
    assert (DOCS, DOC) in E
    # the health checks are not part of anyone's workflow
    assert (HEALTH, FORM) not in E


def test_one_users_keys_and_per_request_keys_do_not_mark_an_address():
    fs = DF.FlowState(1024)
    t = MON
    # a monitor with a fresh key per request: keys never recur
    for n in range(200):
        assert not fs.track_key("10.0.0.9", f"h{n}", t + 60.0 * n, 1800.0)
    # one user, two tabs interleaving: a recurring key, but only two keys active
    for n in range(50):
        assert not fs.track_key("10.0.0.7", "tab1" if n % 2 else "tab2", t + 30.0 * n, 1800.0)
    # five clients interleaving behind one address
    marked = [fs.track_key("10.0.0.1", f"c{n % 5}", t + 20.0 * n, 1800.0) for n in range(12)]
    assert not marked[4] and marked[-1]


def test_a_requirement_is_met_by_any_content_variant_of_its_action():
    """(pack O seed 1, day 7) P04 split the predecessor's node on content: its
    new events carry the key 'route#vN', the requirement was mined on the key
    they had before; a session that performed the predecessor in a variant
    has not skipped it."""
    from test_p10_workflow import model, reports
    st = make_store()
    run(st, reports(3), 3)
    m = model(st)
    req = [r for r in m["scopes"]["*"]["requires"] if r["to"] == GEN]
    assert req and req[0]["from"] == FORM
    t = MON + 3 * DAY
    fs = m.state
    for v in (FORM + "#v7", FORM + "#v9"):
        fs.acts.ensure(v, t)
        sc = DF.seq_scores(m, "*", v, GEN, DF.bloom_bits(DF.h64(v)), t)
        assert not sc["missing"] and not sc["p_req"] == sc["p_req"], sc     # NaN: nothing missing
    # a session without any variant of the form page still misses it
    alone = DF.seq_scores(m, "*", None, GEN, 0, t)
    assert alone["missing"] and alone["p_req"] <= 0.05


def test_a_new_variant_of_the_predecessor_still_counts_in_the_session():
    """The session pass marks a variant's route in the session's Bloom set,
    so a variant too young for the action dictionary (counting waits D)
    still satisfies a requirement on the route."""
    from app.engines.behavior.lib import pevent as EV
    from app.engines.behavior.lib import ptree as PT
    from app.engines.behavior.lib import m_ptree as MP
    from app.models.schema import SYSTEM_ENTITY
    from test_p10_workflow import model, reports
    st = make_store()
    eng, now = run(st, reports(3), 3)
    assert any(r["to"] == GEN and r["from"] == FORM for r in model(st)["scopes"]["*"]["requires"])
    # P04 splits the form page's node on its status: new form events are 'FORM#vN'
    ptm = PT.PTreeModel("oa")
    tree = ptm.tree(EV.KIND_TXN, now)
    sp = tree.split(tree.root, "http.route", 0, [[FORM]], now)
    tree.split(sp.children[0], "http.status", 0, [[200]], now)
    st.put_model("oa", SYSTEM_ENTITY, MP.PTREE, ptm)
    ip = "192.168.9.9"
    t0 = now + 600.0
    run(st, [(t0, ip, FORM, {"http.status": 200})], 900.0 / DAY, eng=eng, t_start=now)
    m = model(st)
    e = m.state.sessions.peek((ip, DF.NO_KEY))
    assert e is not None and e[2].startswith(FORM + "#v"), e
    sc = DF.seq_scores(m, "*", e[2], GEN, int(e[4]), t0 + 120.0)
    assert not sc["missing"], sc


def test_a_rename_is_adopted_for_an_address_whose_sessions_are_keyed():
    """The route ledger (rename detection) follows the ADDRESS session also
    once the address is keyed: the renamed list page must take the place the
    old one held (after the home page), not open a keyed session of its own."""
    from test_p10_rename import FAPPROVE, FITEM, FLIST, GA1
    from test_p10_workflow import APPROVE, HOME, ITEM, LIST, LOGIN, model
    r = np.random.default_rng(3)
    ev = []
    switch = 14
    for d in range(25):
        day = MON + d * DAY
        if not is_workday(day):
            continue
        lst, item, appr = (FLIST, FITEM, FAPPROVE) if d >= switch else (LIST, ITEM, APPROVE)
        # from day 10 several clients share the address before the logins: keys interleave
        for k in range(6 if d >= 10 else 0):
            t = day + 8.5 * 3600 + 30.0 * k
            ev.append((t, GA1, DOCS, {"sess.key": f"d{d}-{k}"}))
            ev.append((t + 200.0, GA1, DOC, {"sess.key": f"d{d}-{k}"}))
        t = day + r.uniform(540, 561) * 60
        ev.append((t, GA1, LOGIN, {"sess.key": f"l{d}"}))
        ev.append((t + r.uniform(5, 30), GA1, HOME, {"sess.key": f"l{d}"}))
        t += 60.0
        for route in (lst, item, appr, item, appr):
            t += r.uniform(60, 360)
            ev.append((t, GA1, route, {"sess.key": f"a{d}"}))
    st = make_store()
    run(st, ev, 25)
    m = model(st)
    assert m.state.keyed.get(GA1) is not None
    acts = m.state.acts
    assert acts.id_of(FLIST) == acts.id_of(LIST) is not None, m["renamed"]
