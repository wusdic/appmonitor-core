"""P10: a required predecessor missing at the START of a session is judged
against every session of the scope (docs/lib3/progressive.md §6.14,
lib/pdfg.p_req). Pack O, anomaly A5 (192.168.1.23 generating the 17:00 report
without opening the form page, day 19): with ~20 report sessions the
requires-test alone gives p_req >= 0.024, above P03's emission bound
SEQ_P = 0.02, so the missing predecessor was flagged but never reported."""
from __future__ import annotations

import numpy as np

from helpers import make_store
from test_p10_workflow import DOC, DOCS, FORM, GEN, HOME, LOGIN, model, run
from temporal_sim import DAY, MON, is_workday

from app.engines.behavior.conformity import SEQ_P
from app.engines.behavior.lib import pdfg as DF


def ga_days(days, seed=0):
    r = np.random.default_rng(seed)
    ev = []
    for d in range(days):
        day = MON + d * DAY
        if not is_workday(day):
            continue
        for k in range(8):                                  # morning login sessions
            ip = f"192.168.1.{30 + k}"
            t = day + r.uniform(510, 531) * 60
            ev += [(t, ip, LOGIN, {}), (t + r.uniform(1, 3), ip, HOME, {})]
            t2 = day + r.uniform(600, 900) * 60             # a later reading session
            ev += [(t2, ip, DOCS, {}), (t2 + r.uniform(10, 50), ip, DOC, {})]
        for ip in ("192.168.1.23", "10.168.7.121"):         # the 17:00 report
            t = day + r.uniform(1020, 1030) * 60
            ev += [(t, ip, FORM, {}), (t + r.uniform(60, 240), ip, GEN, {})]
    return ev


def test_report_without_its_form_at_a_session_start_is_reportable():
    st = make_store()
    run(st, ga_days(14), 14)
    m = model(st)
    req = [r for r in m["scopes"]["*"]["requires"] if r["to"] == GEN]
    assert req and req[0]["from"] == FORM
    t = MON + 14 * DAY
    stt = m.state
    b = stt.acts.id_of(GEN)
    c_b = DF._ev(stt.pcnt, ("*", b), t)
    p_kt = 0.5 / (c_b + 1.0)
    assert p_kt > SEQ_P                                   # the requires-test alone cannot report it
    alone = DF.seq_scores(m, "*", None, GEN, 0, t)        # the session opens with the report
    assert alone["missing"] and alone["p_req"] <= SEQ_P, (alone, c_b)
    # inside a running session (something came before) only the requires-test speaks
    mid = DF.seq_scores(m, "*", DOC, GEN, DF.bloom_bits(DF.h64(DOC)), t)
    assert mid["missing"] and abs(mid["p_req"] - p_kt) < 1e-9
    ok = DF.seq_scores(m, "*", FORM, GEN, DF.bloom_bits(DF.h64(FORM)), t)
    assert not ok["missing"]


def test_a_regrouped_department_is_judged_against_the_population_until_its_scope_knows_the_action():
    """P11 re-forms a group under a new id: P10 mines the new group's scope
    from a few days of traffic (frequent actions' edges, no requirement for a
    daily report yet), and the report without its form page must still be
    judged - against '*' - instead of finding no requirement at all (pack O
    seed 1: 综合部 G12 -> G31 on day 17, A5 on day 19 not reported)."""
    from app.models.schema import ORG
    from app.engines.behavior.lib import m_ptree as MP
    st = make_store()
    ips = [f"192.168.1.{30 + k}" for k in range(8)] + ["192.168.1.23", "10.168.7.121"]
    st.put_model(ORG, ORG, MP.WHO_GROUPS, {"ip2g": {ip: "G12" for ip in ips}})
    ev = ga_days(16)
    cut = MON + 14 * DAY

    def regroup(s, t):
        if cut <= t < cut + 900.0:
            s.put_model(ORG, ORG, MP.WHO_GROUPS, {"ip2g": {ip: "G31" for ip in ips}})
    run(st, ev, 16, hooks=(regroup,))
    m = model(st)
    t = MON + 16 * DAY
    assert "G31" in m["scopes"]                                    # mined from two days
    assert not [r for r in m["scopes"]["G31"]["requires"] if r["to"] == GEN]
    alone = DF.seq_scores(m, "G31", None, GEN, 0, t)
    assert alone["missing"] and alone["p_req"] <= SEQ_P, alone
    ok = DF.seq_scores(m, "G31", FORM, GEN, DF.bloom_bits(DF.h64(FORM)), t)
    assert not ok["missing"]
