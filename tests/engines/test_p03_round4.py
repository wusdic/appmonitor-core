"""Round 4 (groups_views owner): P03 predictive tails and route renames.

Each test fails on the round-3 code (HEAD c67c985)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.engines.behavior import conformity as CF
from app.engines.behavior.lib import pbounds as PB
from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.lib import pscore as SC
from app.engines.behavior.lib.evt import gpd_pwm_fit

T0 = 1_700_000_000.0


def _lognormal_record(seed: int, mu: float, sd: float, n: int = 2000):
    rng = np.random.default_rng(seed)
    num = PN.NumSummary(log=True)
    xs = np.exp(rng.normal(mu, sd, n))
    for i, x in enumerate(xs):
        num.update(float(x), T0 + i * 60, 1.0, 1.0, day=int(i // 200))
    rec = PB.fit_numeric(num, T0 + n * 60, 10, n_c=float(n), n_eff=0.75 * n, unit="ms")
    return rec, xs


def test_a_value_just_past_the_observed_maximum_is_not_extreme():
    """Pack O (D4 / D5, legitimate): a portal comment's 222 ms duration against
    an observed maximum of 200 ms scored p 4.4e-7 and a git net.pkts_down inside
    the displayed range p 1e-9 - the PWM GPD of the reservoir's 64 excesses was
    bounded (xi < 0) and ended BEFORE the observed maximum, so the plug-in tail
    put every value past its end point at the floor. The predictive tail keeps
    the parameters' uncertainty and only the ones whose support reaches the
    observed maximum: a new maximum of a lognormal(3.5, 0.6) sample (true
    two-sided p ~1e-4 at 1.02 x max) is not a 1e-9 event, while values far
    beyond stay extreme."""
    rec, xs = _lognormal_record(3, 3.5, 0.6)
    th = rec["tail_hi"]
    assert th and th[1] < 0 and th[0] + th[2] / -th[1] < rec["range"][1]   # fitted end point < max
    fast = CF._NumFast(rec)
    mx = rec["range"][1]
    p_plug = PB.p_value(rec, 1.02 * mx)[0]
    assert p_plug < 1e-8                                            # the round-3 behaviour
    p_new = CF._check("num", fast, 1.02 * mx, None)[0]
    assert p_new >= 1e-5
    # still monotone and still extreme far out (A3's 12 KB login body is ~6x the max)
    p5 = CF._check("num", fast, 5 * mx, None)[0]
    p20 = CF._check("num", fast, 20 * mx, None)[0]
    assert p_new > p5 > p20
    assert p5 <= 1e-6 and p20 <= 1e-8


def test_predictive_tail_is_calibrated_where_the_plug_in_is_not():
    """Exchangeable fresh values beyond Q(0.99): the rate of p <= 1e-4 should be
    about its nominal 0.5e-4 per value (one side of a two-sided p). The plug-in
    GPD of 64 excesses fires several times too often (round 3: every LOW content
    finding of D4 / D5 was such a value); the predictive stays within 2x."""
    rng = np.random.default_rng(11)
    a = 1e-4
    hits = {"plug": 0, "pred": 0}
    tot = 0
    for _ in range(25):
        s = rng.lognormal(3.5, 0.6, 1000)
        y = np.log(s)
        u = float(np.quantile(y, 0.9))
        sel = y[y > u]
        sel = rng.choice(sel, 64, replace=False)
        un = math.exp(u)
        exc = np.exp(sel) - un
        xi, sig = gpd_pwm_fit(exc)
        hi = math.exp(float(np.quantile(y, 0.99)))
        new = rng.lognormal(3.5, 0.6, 40000)
        tot += new.size
        for v in new[new > hi]:
            hits["plug"] += SC.tail_p(v, un, xi, sig, 0.1) <= a
            hits["pred"] += SC.tail_p(v, un, xi, sig, 0.1, k=64, x_max=float(s.max())) <= a
    nominal = tot * a / 2
    assert hits["plug"] > 3 * nominal
    assert hits["pred"] < 2 * nominal


def test_predictive_sf_reduces_to_plug_in_without_k_and_is_bounded():
    assert SC.tail_p(5.0, 1.0, 0.1, 2.0, 0.1) == pytest.approx(
        0.2 * (1 + 0.1 * 4.0 / 2.0) ** (-1 / 0.1))
    # many excesses: the predictive approaches the plug-in
    p_plug = SC.tail_p(5.0, 1.0, 0.1, 2.0, 0.1)
    p_big = SC.tail_p(5.0, 1.0, 0.1, 2.0, 0.1, k=1e7)
    assert p_big == pytest.approx(p_plug, rel=1e-3)
    # just past the end point of a bounded fit (xi -0.3, sigma 3: support ends
    # at 10, the observed maximum excess): the plug-in is 0, the predictive is
    # of the order of the tail's rank probability 1 / (k + 1)
    from app.engines.behavior.lib.evt import gpd_sf
    assert float(gpd_sf(np.asarray([10.5]), -0.3, 3.0)[0]) == 0.0
    sf = SC.gpd_predictive_sf(10.5, -0.3, 3.0, 64, z_max=10.0)
    assert 0.1 / 65 < sf < 2.0 / 65
    assert SC.tail_p(0.5, 1.0, 0.1, 2.0, 0.1, k=64) == 1.0


# ------------------------------------------------------------------ renames
def test_a_renamed_page_of_its_own_source_is_not_a_new_action():
    """Pack O D3 (day 14): the approver's pages moved from /approval/ to /flow/.
    Each renamed page was a MEDIUM `new_action` (p_novel 1.4e-4); the incident
    held 192.168.1.21's rows, so the /flow/ nodes never confirmed and the old
    statements went stale without successors (PG5 D3 0/5). The page an
    established route of THE SAME source became (lib/pdfg.successor_candidate:
    same shape, one literal segment renamed, the old route not used today) is
    a candidate rename - flagged, not novel - until P10 adopts it; the same
    page from another source is still a new action."""
    from test_p03_conformity import APPROVE, _novel_findings, _novel_fixture, workdays
    from app.engines.behavior.lib import m_ptree as MP
    from app.engines.behavior.lib import pdfg as DF
    from app.models.schema import SYSTEM_ENTITY, Severity
    fx = _novel_fixture()
    flow = fx.st.get_model("finance", SYSTEM_ENTITY, MP.PFLOW)
    t = workdays(12)[-1] + 10 * 3600
    today = DF.EPOCH_ORD + int(t // 86400.0)
    for d in range(10, 6, -1):                      # four earlier dates of the approver
        flow.state.see_route(APPROVE, "192.168.2.10", today - d, None)
    new = "POST fin /fin/flow/{num}/approve"
    # another source on the renamed page: a new action of that source
    fx.score("finance", [(t - 600, "192.168.2.11", {"http.route": new, "http.method": "POST"})])
    assert _novel_findings(fx, ["192.168.2.11"])["192.168.2.11"] == [Severity.MEDIUM]
    b, asg = fx.score("finance", [(t, "192.168.2.10", {"http.route": new, "http.method": "POST"})])
    assert _novel_findings(fx, ["192.168.2.10"])["192.168.2.10"] == []
    assert "rename_candidate" in str(asg.get("flags", 0))
    assert "new_action" not in str(asg.get("flags", 0))


# ------------------------------------------------- sequence: route-level re-scoring
def _portal_flow(n: int = 300):
    """P10 state of n portal visits: GET / opens every session, then login and
    a few articles of the content variant #v11 (a P04 content split)."""
    from app.engines.behavior.lib import pdfg as DF
    st = DF.FlowState()
    home, login, news = "GET www /", "POST www /login", "GET www /news/{num}#v11"
    for k in range(n):
        t = T0 + 600.0 * k
        seq = [home, login, news, news, news]
        ids = [st.acts.add(x, t, 1, 1)[0] for x in seq]
        for x in ids:
            st.cnt.add(("*", x), t, 1, 1)
        st.starts.add(("*", ids[0]), t, 1, 1)
        for x, y in zip(ids, ids[1:]):
            st.add_edge("*", x, y, t, 1, 1, DF.delay_bin(60.0))
    st.marg = None
    return st, T0 + 600.0 * n


def test_a_new_content_variant_of_the_same_route_is_not_a_sequence_anomaly():
    """Pack O seed 1 D4 (days 12-18): P04 re-split GET /news/{num} and minted
    variant ids (#v17-#v19) without transition history; a reader's next article
    scored p_trans 6.5e-8 .. 0 and three readers of the growing public
    population became conf_seq incidents. At the route level (the variants of
    one route are one action for the session model) the hop is ordinary."""
    from app.engines.behavior.lib import pdfg as DF
    st, t = _portal_flow()
    v19, _ = st.acts.add("GET www /news/{num}#v19", t, 1, 1)
    st.cnt.add(("*", v19), t, 1, 1)
    st.marg = None
    a = st.acts.id_of("GET www /news/{num}#v11")
    assert DF.p_trans(st, "*", a, v19, t) < 1e-3
    assert CF.seq_coarse_p(st, a, v19, t) > 0.3


def test_a_revisit_merged_into_the_session_is_scored_as_a_session_start():
    """A reader back on the portal 5-28 minutes after the last article (P10's
    30-minute gap merges the visits): news -> GET / scored p_trans 3e-8. GET /
    opens every session, so the event is unsurprising as the start of a visit."""
    from app.engines.behavior.lib import pdfg as DF
    st, t = _portal_flow()
    a = st.acts.id_of("GET www /news/{num}#v11")
    home = st.acts.id_of("GET www /")
    assert DF.p_trans(st, "*", a, home, t) < 1e-3
    assert CF.seq_coarse_p(st, a, home, t) >= 0.49        # the only opener: HDR p 1/2 (tie with itself)


def test_a_jump_to_a_route_that_neither_follows_nor_opens_stays_extreme():
    """The coarser null keeps its power: a page that never follows an article
    and never opens a session (an admin action) is still improbable."""
    from app.engines.behavior.lib import pdfg as DF
    st, t = _portal_flow()
    adm, _ = st.acts.add("POST www /admin/delete", t, 1, 1)
    st.cnt.add(("*", adm), t, 1, 1)
    st.marg = None
    a = st.acts.id_of("GET www /news/{num}#v11")
    assert CF.seq_coarse_p(st, a, adm, t) < 0.01
    assert math.isnan(CF.seq_coarse_p(st, None, adm, t))


def test_p03_rescores_an_improbable_transition_at_the_route_level():
    """Wiring: P03's p_seq of a revisit (docs -> home within P10's session gap)
    is the route-level re-score, not 2 x p_trans."""
    from test_p03_conformity import CFG, Fx, GA, ctx, workdays
    from app.engines.behavior.workflow import WorkflowEngine
    from app.engines.behavior.lib import m_ptree as MP
    from app.engines.behavior.lib import pdfg as DF
    from app.engines.behavior.lib import pevent as EV
    from app.models.schema import SYSTEM_ENTITY
    fx = Fx()
    home, docs = "GET oa /home", "GET oa /docs"
    for d in workdays(20):
        for ip in GA:
            fx.learn("oa", home, ip, d + 9 * 3600)
            fx.learn("oa", docs, ip, d + 9 * 3600 + 60)
    fx.tree("oa", [home, docs])
    p10 = WorkflowEngine(mine_period_s=6 * 3600.0)
    days = workdays(21)
    for d in days[:20]:
        for h in range(24):
            t1 = d + (h + 1) * 3600.0
            if h == 9:
                b = EV.BatchBuilder("oa", EV.KIND_TXN)
                for ip in GA:
                    b.add(d + 9 * 3600 + 10, ip, {"http.route": home, "net.src": ip})
                    for j in range(5):
                        b.add(d + 9 * 3600 + 60 + 30 * j, ip, {"http.route": docs, "net.src": ip})
                fx.st.add_batch("oa", EV.EVT_BATCH, t1, b.build(t1 - 3600, t1))
            fx.st.ensure_retention("evt.", max_age_s=6 * 3600.0)
            p10.safe_run(ctx(fx.st, t1, window_s=3600.0, config=CFG), None)
    flow = fx.st.get_model("oa", SYSTEM_ENTITY, MP.PFLOW)
    st = flow.state
    t = days[20] + 9 * 3600
    a, b = st.acts.id_of(docs), st.acts.id_of(home)
    assert a is not None and b is not None
    assert DF.p_trans(st, "*", a, b, t) < 0.01                 # docs is never followed by home
    _, asg = fx.score("oa", [(t, GA[0], {"http.route": home, "http.method": "GET"}),
                             (t + 60, GA[0], {"http.route": docs, "http.method": "GET"}),
                             (t + 400, GA[0], {"http.route": home, "http.method": "GET"})])
    assert asg.get("p_seq", 2) > 0.1                           # home opens every visit
