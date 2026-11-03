"""lib/pfd round 11: a SET entry of a reverse record (y used by 2-4 sources)
is refined per member - the concurrency test reads only its major members
(share >= 1/|set|), and an unknown source is a readdress candidate when the
value's dominant source has been silent since the start of the day, even
while a colleague sharing the value is active (progressive.md §16.18.12
item 8; O-real R3: the finance clerk 192.168.2.11 -> .51 met the set entry
tls.ja3 -> {.10, .11, .12} while .10 was active, 'concurrent_use')."""
from __future__ import annotations

from app.engines.behavior.lib import pfd as FD
from app.engines.behavior.lib import pnode as PN

T0 = 1_780_000_000.0
DAY = 86400.0
JA3 = "ja3:finance-client"
OWN, MATE, MINOR = "192.168.2.11", "192.168.2.10", "192.168.2.12"
NEW = "192.168.2.51"


def _fit(counts):
    """A reverse pair (value -> source): JA3 used by the sources of `counts`."""
    ps = PN.PairSketch()
    i = 0
    for src, n in counts:
        for _ in range(n):
            ps.update(JA3, src, T0 + i * 1800.0, 1.0, 1.0)
            i += 1
    return FD.fit_pair(ps, T0 + i * 1800.0)


def test_set_entry_keeps_each_members_share():
    r = _fit([(OWN, 30), (MATE, 18), (MINOR, 12)])
    e = r["table"][JA3]
    assert not e.get("bound") and e["set"] == sorted([OWN, MATE, MINOR])
    sh = dict(zip(e["set"], e["shares"]))
    assert sh[OWN] > sh[MATE] > sh[MINOR] > 0.0
    assert abs(sum(e["shares"]) - 1.0) < 0.05
    assert FD.set_members(e)[0][0] == OWN


def test_silent_dominant_member_makes_a_readdress_while_a_colleague_is_active():
    r = _fit([(OWN, 30), (MATE, 24), (MINOR, 10)])
    now = T0 + 10 * DAY + 9 * 3600.0
    day_start = now - 9 * 3600.0
    act = {MATE: now - 600.0, MINOR: now - 2 * 3600.0, OWN: now - 2 * DAY}
    info = {}
    p, f = FD.check_reverse(r, NEW, JA3, now, act.get, x_known=False, day_start=day_start, info=info)
    # round 10: ['foreign_source', 'concurrent_use'] (MATE active 10 min ago)
    assert f == ["foreign_source", "readdress_candidate"]
    assert info["src"] == OWN
    # a KNOWN source is never a readdress: the colleague's concurrency stands
    p, f = FD.check_reverse(r, NEW, JA3, now, act.get, x_known=True, day_start=day_start)
    assert f == ["foreign_source", "concurrent_use"]


def test_minor_member_activity_is_not_concurrent_use():
    r = _fit([(OWN, 30), (MATE, 24), (MINOR, 6)])
    e = r["table"][JA3]
    assert dict(zip(e["set"], e["shares"]))[MINOR] < 1.0 / 3.0
    now = T0 + 10 * DAY + 9 * 3600.0
    day_start = now - 9 * 3600.0
    # the dominant member was active this morning (no readdress), the major
    # colleague not within T_conc; only the minor member is active right now
    act = {OWN: now - 3 * 3600.0, MATE: now - 4 * 3600.0, MINOR: now - 300.0}
    p, f = FD.check_reverse(r, NEW, JA3, now, act.get, x_known=False, day_start=day_start)
    assert f == ["foreign_source"]                  # round 10: + 'concurrent_use'
    # a MAJOR member active right now is concurrent use, as before
    act[MATE] = now - 300.0
    p, f = FD.check_reverse(r, NEW, JA3, now, act.get, x_known=False, day_start=day_start)
    assert f == ["foreign_source", "concurrent_use"]


def test_bound_entry_and_old_set_entries_unchanged():
    # a bound reverse entry: concurrency first, as in round 10
    ps = PN.PairSketch()
    for x, y in [("192.168.1.21", "jack"), ("192.168.1.23", "rose"), ("10.168.7.121", "mike")]:
        for _ in range(8):
            ps.update(y, x, T0, 1, 1)
    r = FD.fit_pair(ps, T0)
    now = T0 + 10 * DAY
    info = {}
    p, f = FD.check_reverse(r, "10.168.99.5", "mike", now, {"10.168.7.121": now - 2 * DAY}.get,
                            x_known=False, day_start=now - 9 * 3600, info=info)
    assert f == ["foreign_source", "readdress_candidate"] and info["src"] == "10.168.7.121"
    # a set entry fitted without shares: equal shares, the first by name dominant
    rec = {"table": {JA3: {"set": [MATE, OWN], "U": 0.01, "n": 20.0}}}
    assert [m for m, _ in FD.set_members(rec["table"][JA3])] == [MATE, OWN]
    act = {MATE: now - 2 * DAY, OWN: now - 600.0}
    p, f = FD.check_reverse(rec, NEW, JA3, now, act.get, x_known=False, day_start=now - 9 * 3600)
    # both are dominant (tie); MATE is silent -> a readdress of MATE
    assert f == ["foreign_source", "readdress_candidate"]


def test_silent_major_member_is_a_readdress_while_the_largest_colleague_is_active():
    """O-real seed 2 (R3): the clerk's stack {.10 0.14, .11 0.41, .12 0.46} -
    the colleague .12 holds the largest share and was active at 09:00; .51's
    first finance login (09:03) was 'concurrent_use' (a HIGH incident) under
    the dominant-member rule. The silent MAJOR member .11 makes it a
    readdress of .11; with every major member active it stays concurrent."""
    r = _fit([(MATE, 14), (OWN, 41), (MINOR, 46)])
    e = r["table"][JA3]
    sh = dict(zip(e["set"], e["shares"]))
    assert sh[MINOR] > sh[OWN] > 1.0 / 3.0 > sh[MATE]
    now = T0 + 10 * DAY + 9 * 3600.0 + 225.0
    day_start = now - 9 * 3600.0 - 225.0
    act = {MATE: now - 160.0, MINOR: now - 225.0, OWN: now - DAY - 3600.0}
    info = {}
    p, f = FD.check_reverse(r, NEW, JA3, now, act.get, x_known=False, day_start=day_start, info=info)
    assert f == ["foreign_source", "readdress_candidate"]       # dominant rule: concurrent_use
    assert info["src"] == OWN
    # every major member active today: an additional user of the value
    act[OWN] = now - 3600.0
    p, f = FD.check_reverse(r, NEW, JA3, now, act.get, x_known=False, day_start=day_start)
    assert f == ["foreign_source", "concurrent_use"]
    # a silent MINOR member is no readdress source
    act[OWN] = now - 3600.0; act[MATE] = now - 2 * DAY
    p, f = FD.check_reverse(r, NEW, JA3, now, act.get, x_known=False, day_start=day_start)
    assert f == ["foreign_source", "concurrent_use"]


def test_a_known_source_stays_a_readdress_of_the_source_it_replaced_today():
    """O-real seed 2: .51's first finance login (09:03:45) was a readdress of
    .11; the next event, 2 s later, found .51 KNOWN (its first event made it
    so) and was 'concurrent_use' with the active colleague .12 - a HIGH
    incident before P11 recorded the move. `readdr_of` (conformity's note of
    the day) keeps the known source a candidate of THAT source only."""
    r = _fit([(MATE, 14), (OWN, 41), (MINOR, 46)])
    now = T0 + 10 * DAY + 9 * 3600.0 + 227.0
    day_start = now - 9 * 3600.0 - 227.0
    act = {MATE: now - 160.0, MINOR: now - 227.0, OWN: now - DAY - 3600.0, NEW: now - 2.0}
    p, f = FD.check_reverse(r, NEW, JA3, now, act.get, x_known=True, day_start=day_start)
    assert f == ["foreign_source", "concurrent_use"]              # without the note
    info = {}
    p, f = FD.check_reverse(r, NEW, JA3, now, act.get, x_known=True, day_start=day_start,
                            info=info, readdr_of=OWN)
    assert f == ["foreign_source", "readdress_candidate"] and info["src"] == OWN
    # a note naming a source that is not a (major) source of this value: no readdress here
    p, f = FD.check_reverse(r, NEW, JA3, now, act.get, x_known=True, day_start=day_start,
                            readdr_of=MATE)
    assert f == ["foreign_source", "concurrent_use"]
    # the replaced source active again today: no longer a candidate
    act[OWN] = now - 600.0
    p, f = FD.check_reverse(r, NEW, JA3, now, act.get, x_known=True, day_start=day_start,
                            readdr_of=OWN)
    assert "readdress_candidate" not in f
    # a bound value (the clerk's user name): the same rule
    ps = PN.PairSketch()
    for x, y in [(OWN, "tom"), ("192.168.1.23", "rose"), ("10.168.7.121", "mike")]:
        for _ in range(8):
            ps.update(y, x, T0, 1, 1)
    rb = FD.fit_pair(ps, T0)
    assert rb["table"]["tom"].get("bound")
    act = {OWN: now - DAY}
    p, f = FD.check_reverse(rb, NEW, "tom", now, act.get, x_known=True, day_start=day_start)
    assert f == ["foreign_source"]
    p, f = FD.check_reverse(rb, NEW, "tom", now, act.get, x_known=True, day_start=day_start,
                            readdr_of=OWN)
    assert f == ["foreign_source", "readdress_candidate"]
