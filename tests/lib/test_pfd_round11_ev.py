"""Proposed for the bounds owner (lib/pfd.fit_pair): a shared terminal whose small value set has not
closed yet (2 users over 8 logins: U = 0.5/9 > SET_U) is no counter-example of the node's binding.
O-real seed 2 (round 10, day 12-15): family login node fam:1 0:18, 192.168.1.40 (jack2 x5, rose2 x3)
judged -> g3 0.059 > FD_G3 -> holds False -> no net.src -> username binding stated (bindings @14 0/6)."""
from app.engines.behavior.lib import pfd as FD
from app.engines.behavior.lib import pnode as PN

DAY = 86400.0


def test_an_unclosed_small_set_is_not_a_counter_example():
    ps = PN.PairSketch()
    t0 = 1.75e9
    users = {"192.168.1.21": "jack", "192.168.1.23": "rose", "10.168.7.121": "mike",
             "192.168.2.10": "lucy", "192.168.2.11": "tom", "192.168.2.12": "kate"}
    for d in range(8):
        t = t0 + d * DAY
        for i, (ip, u) in enumerate(sorted(users.items())):
            ps.update(ip, u, t + i * 60.0, 1.0, 1.0)
        ps.update("192.168.1.40", "jack2" if d in (0, 1, 3, 4, 6) else "rose2", t + 900.0, 1.0, 1.0)
    rec = FD.fit_pair(ps, t0 + 8 * DAY)
    assert rec["fd"]["holds"], rec["fd"]
