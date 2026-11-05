"""lib/pfd, round 12 (views owner's proposal for the bounds owner's file):
the values a source's sketch evicted stay unseen mass when a pending value is
excluded from its counts.

O-real seed 2 (round 11, day 19): the branch NAT 192.168.30.1 (about ten
users, far more than the four values its pair sketch tracks: evicted
evidence 57 of 69) had a pending newcomer ('mztie') excluded by
ValueHistory.classify; pair_counts recomputed U as (N1 + 0.5) / (n + 1) from
the tracked values only, dropping the sketch's eviction term, so U fell to
~0.05 and the NAT was no longer multi_source but a judged counter-example
(g3 0.23 > FD_G3): the family login nodes 0:3 / 0:14 lost their
net.src -> username dependency and the day-19 statement stated no binding
table (D2's 10.168.7.121 -> mike.w switch reached the statements a day late).
"""
from __future__ import annotations

from app.engines.behavior.lib import pfd as FD
from app.engines.behavior.lib import pnode as PN

T0 = 1_780_000_000.0
DAY = 86400.0
GA = [("192.168.1.21", "jack"), ("192.168.1.23", "rose"), ("10.168.7.121", "mike")]
SALES = [(f"192.168.3.{20 + i}", f"user{i:02d}") for i in range(12)]
NAT = "192.168.30.1"


def _sketch(nat_users: int = 10, days: int = 8):
    ps = PN.PairSketch()
    for d in range(days):
        for x, y in GA + SALES:
            ps.update(x, y, T0 + d * DAY, 1.0, 1.0)
        for u in range(nat_users):
            ps.update(NAT, f"br{u}", T0 + d * DAY + 60.0 * u, 1.0, 1.0)
    return ps


def test_a_pending_value_does_not_close_a_many_valued_source():
    ps, t = _sketch(), T0 + 8 * DAY
    tab = ps.y.get(NAT)
    assert tab.eviction_evidence(t) > 0.5 * tab.total_evidence(t)     # more users than tracked slots
    tracked = [str(k) for k, _, _, _ in tab.items(t, 0)]
    base = FD.fit_pair(ps, t)
    assert base["table"][NAT].get("multi") and base["fd"]["holds"]
    # one tracked value pending (a newcomer not yet confirmed): the source is
    # still a gateway of many users, not a closed source and not a counter-example
    cnt = FD.pair_counts(ps, t, exclude={NAT: {tracked[-1]: "pending"}})[NAT]
    assert cnt["U"] > FD.SET_U, cnt["U"]
    rec = FD.fit_pair(ps, t, exclude={NAT: {tracked[-1]: "pending"}})
    nat = rec["table"][NAT]
    assert nat.get("multi") and not nat.get("bound"), nat
    assert rec["fd"]["holds"], rec["fd"]
    assert all(rec["table"][x]["bound"] for x, _ in GA)


def test_exclusion_on_a_fully_tracked_source_keeps_the_tracked_estimate():
    """No eviction: U after the exclusion is the tracked Good-Turing estimate
    (unchanged behaviour for a source whose values all fit the sketch)."""
    ps = PN.PairSketch()
    for d in range(8):
        ps.update("192.168.1.40", "jack2", T0 + d * DAY, 1.0, 1.0)
        if d % 2:
            ps.update("192.168.1.40", "rose2", T0 + d * DAY + 60, 1.0, 1.0)
    t = T0 + 8 * DAY
    assert ps.y.get("192.168.1.40").eviction_evidence(t) == 0.0
    c = FD.pair_counts(ps, t, exclude={"192.168.1.40": {"rose2": "pending"}})["192.168.1.40"]
    assert abs(c["U"] - 0.5 / (c["n"] + 1.0)) < 1e-9
