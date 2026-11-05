"""lib/pfd, round 12 (bounds owner): a pooled source's set is read from its days.

O-real seed 2 (round 11, days 12-16): the shared terminal 192.168.1.40 (jack2
on 5 days, rose2 on 3) was pooled on its tree-level days (n = 8 days) while
fit_pair's set rule read the node's young sketch (two logins) against that
pooled n: the set covered 2 of 8 'units', so the terminal was neither a set
nor an open set but a judged counter-example (top 5 / 8, g3 0.059 > FD_G3):
the family login node's net.src -> username dependency never held and no
binding of the node was stated (bindings @14 2/6). With `support_days`
(history_days per source) the set rule reads the same days the pooled n
counts.
"""
from __future__ import annotations

from app.engines.behavior.lib import pfd as FD
from app.engines.behavior.lib import pnode as PN

DAY = 86400.0
T0 = 1_788_220_800.0          # a Monday 00:00 UTC
FAM = [("192.168.1.21", "jack"), ("192.168.1.23", "rose"), ("10.168.7.121", "mike"),
       ("192.168.2.10", "lucy"), ("192.168.2.11", "tom"), ("192.168.2.12", "kate")]
TERM = "192.168.1.40"


def _family_login(term_days):
    ps, h = PN.PairSketch(), FD.ValueHistory()
    for d in range(8):
        t = T0 + d * DAY + 9 * 3600
        for i, (x, y) in enumerate(FAM):
            h.observe(x, y, t + i, 700000 + d, True)
            if d >= 6:
                ps.update(x, y, t + i, 1.0, 1.0)
        y40 = term_days(d)
        h.observe(TERM, y40, t + 50, 700000 + d, True)
        if d >= 6:
            ps.update(TERM, y40, t + 50, 1.0, 1.0)
    return ps, h, T0 + 8 * DAY


def _fit(ps, h, t, days=True):
    xs = sorted(map(str, ps.x.keys()))
    sup = {x: FD.history_support(h.get(x)) for x in xs}
    kw = {"support_days": {x: FD.history_days(h.get(x)) for x in xs}} if days else {}
    return FD.fit_pair(ps, t, support=sup, **kw)


def test_history_days_counts_days_per_value_and_honours_exclude():
    _, h, _ = _family_login(lambda d: "jack2" if d in (0, 2, 3, 5, 7) else "rose2")
    assert FD.history_days(h.get(TERM)) == {"jack2": 5.0, "rose2": 3.0}
    assert FD.history_days(h.get(TERM), {"rose2": "pending"}) == {"jack2": 5.0}
    assert FD.history_days(None) == {}
    # the pooled n of history_support is the sum of the same days
    assert FD.history_support(h.get(TERM))[2] == 8.0


def test_a_pooled_shared_terminal_is_an_open_set_not_a_counter_example():
    ps, h, t = _family_login(lambda d: "jack2" if d in (0, 2, 3, 5, 7) else "rose2")
    old = _fit(ps, h, t, days=False)
    assert not old["fd"]["holds"] and old["fd"]["g3"] > FD.FD_G3        # the round-11 veto
    rec = _fit(ps, h, t)
    e = rec["table"][TERM]
    assert e.get("pooled") and not e.get("bound")
    assert e.get("set_open") == ["jack2", "rose2"], e
    assert rec["fd"]["holds"], rec["fd"]
    assert all(rec["table"][x].get("bound") and rec["table"][x]["top"] == y for x, y in FAM)


def test_a_pooled_source_with_one_value_is_still_bound():
    """The days are read only where the pooled source is not bound: a source
    seen with one value on every day keeps its binding."""
    ps, h, t = _family_login(lambda d: "jack2")
    rec = _fit(ps, h, t)
    e = rec["table"][TERM]
    assert e.get("bound") and e["top"] == "jack2" and not e.get("set_open")
    assert rec["fd"]["holds"]
