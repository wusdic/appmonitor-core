"""lib/pfd round 8: a multi-user source (a NAT gateway several users share:
many values, none dominant) is neither a binding nor a counter-example of the
node's dependency. O-real's branch NAT 192.168.30.1 (R2) alone failed the
'POST /login' nodes' dependency (g3 0.15-0.43 > FD_G3 = 0.05), so no binding
there was credential-grade and A2 (192.168.1.21 using rose's credential) was
judged against no binding on seed 1. Fails on the round-7 code."""
from __future__ import annotations

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
