"""lib/pfd round 8: a multi-user source (a NAT gateway several users share:
many values, none dominant) is neither a binding nor a counter-example of the
node's dependency. O-real's branch NAT 192.168.30.1 (R2) alone failed the
'POST /login' nodes' dependency (g3 0.15-0.43 > FD_G3 = 0.05), so no binding
there was credential-grade and A2 (192.168.1.21 using rose's credential) was
judged against no binding on seed 1. Fails on the round-7 code."""
from __future__ import annotations

import pytest

from app.engines.behavior.lib import pfd as FD
from app.engines.behavior.lib import pnode as PN

T0 = 1_780_000_000.0
DAY = 86400.0
GA = [("192.168.1.21", "jack"), ("192.168.1.23", "rose"), ("10.168.7.121", "mike")]
SALES = [(f"192.168.3.{20 + i}", f"user{i:02d}") for i in range(12)]
NAT = "192.168.30.1"


def _sketch(nat_users: int, logins: int = 8):
    ps = PN.PairSketch()
    t = T0
    for d in range(logins):
        for x, y in GA + SALES:
            ps.update(x, y, t + d * DAY, 1.0, 1.0)
        for u in range(nat_users):
            ps.update(NAT, f"br{u}", t + d * DAY + 60.0 * u, 1.0, 1.0)
    return ps


def test_a_nat_gateway_does_not_veto_the_dependency():
    r = FD.fit_pair(_sketch(10), T0 + 8 * DAY)
    nat = r["table"][NAT]
    assert nat.get("multi") and not nat.get("bound") and not nat.get("set")
    assert r["fd"]["holds"], r["fd"]
    assert all(r["table"][x]["bound"] for x, _ in GA)
    # the dependency is still credential-grade: rose from .21 is a cross binding
    assert FD.check_forward(r, "192.168.1.21", "rose")[1] == ["cross_binding"]


def test_a_source_with_a_dominant_value_or_few_values_is_still_judged():
    c = {"y": {"a": 9.0, "b": 1.0}}
    assert not FD.multi_source(c, 9.0, 10.0)                        # dominant value
    c = {"y": {f"u{i}": 2.0 for i in range(FD.SET_K)}}
    assert not FD.multi_source(c, 2.0, 2.0 * FD.SET_K)              # few values: a set binding's case
    c = {"y": {f"u{i}": 2.0 for i in range(FD.MULTI_K)}}
    assert FD.multi_source(c, 2.0, 2.0 * FD.MULTI_K)
    c = {"y": {"u0": 2.0, "u1": 2.0}, "U": 0.7}                     # most values untracked
    assert FD.multi_source(c, 2.0, 20.0)
    assert not FD.multi_source(c, 1.0, 2.0, n_bind=5.0)             # too little evidence



SHARED = "192.168.1.40"


def _hist(rows):
    h = FD.ValueHistory()
    for x, y, day in rows:
        h.observe(x, y, T0 + day * DAY + 300.0, 20_000 + day, True, True)
    return h


def test_a_coexisting_newcomer_no_other_source_holds_is_unexplained():
    """(round 8) Which PENDING values x's own binding must count: a shared
    terminal's second user (rose2, first seen 3 days after jack2, and jack2
    keeps logging in) - yes; a rename (mike -> mike.w, mike never seen
    again: `refuted` / 'superseded' handle it) - no; a borrowed credential
    (rose, the established value of 192.168.1.23) - no."""
    h = _hist([(SHARED, "jack2", d) for d in range(6)] + [(SHARED, "rose2", d) for d in (3, 5)]
              + [("10.168.7.121", "mike", d) for d in range(4)]
              + [("10.168.7.121", "mike.w", d) for d in (4, 5)]
              + [("192.168.1.21", "jack", d) for d in range(6)] + [("192.168.1.21", "rose", d) for d in (4, 5)])
    pend = {"rose2": "pending"}
    assert FD.unexplained_values(h.get(SHARED), pend, ["jack", "rose", "mike"]) == ["rose2"]
    assert FD.unexplained_values(h.get("10.168.7.121"), {"mike.w": "pending"}, ["jack", "rose"]) == []
    assert FD.unexplained_values(h.get("192.168.1.21"), {"rose": "pending"}, ["rose", "mike"]) == []
    assert FD.unexplained_values(h.get(SHARED), {}, []) == []


def test_an_unexplained_value_lowers_its_sources_claim_only():
    """(round 8) The shared terminal's claim '-> jack2' counts rose2's
    evidence (O-real seed 0: stated at LB 0.997 at four family login nodes
    from day 21, held 0.0); the node's dependency is judged as before. Fails
    on the round-7 / mine8 code (no unexplained_lb / unexplained_evidence)."""
    ps = PN.PairSketch()
    for d in range(8):
        for x, y in GA:
            ps.update(x, y, T0 + d * DAY, 1.0, 1.0)
        ps.update(SHARED, "jack2", T0 + d * DAY + 120.0, 1.0, 1.0)
        if d >= 3:
            ps.update(SHARED, "rose2", T0 + d * DAY + 600.0, 1.0, 1.0)
    t = T0 + 8 * DAY
    rec = FD.fit_pair(ps, t, exclude={SHARED: {"rose2": "pending"}})
    ent = rec["table"][SHARED]
    assert ent["top"] == "jack2" and ent.get("bound") and ent["LB"] > 0.9     # the over-confident claim
    extra = FD.unexplained_evidence(ps, SHARED, ["rose2"], t)
    assert extra > 3.0
    assert FD.unexplained_lb(ent, extra) < FD.LB_BIND <= ent["LB"]
    assert FD.unexplained_lb(ent, 0.0) == pytest.approx(ent["LB"])
    # pooled entries count the value's days from the history
    hx = {"v": {"jack2": [T0, t, 8, list(range(8))], "rose2": [T0 + 3 * DAY, t, 5, [3, 4, 5, 6, 7]]}}
    assert FD.unexplained_evidence(ps, SHARED, ["rose2"], t, hx, pooled=True) == 5.0
