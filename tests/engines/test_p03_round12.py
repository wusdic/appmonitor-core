"""P03 round 12 (bounds owner): the when HDR tables read a slot density
smoothed over the adjacent 15-minute slots (pscore.when_smooth), with slot
densities within 20 % tied in the HDR order (pscore.when_hdr_table).

Pack O seed 1, round 11 (D4): round 11's correct Bonferroni change left the
24-hour monitor 192.168.9.9 learned less, so the portal root's non-workday
22:45 slot held 51.5 of mass against ~54 in the other monitor-only slots -
the lowest of a near-flat stretch. Under the HDR ordering the slot's p was
1.3e-3 at the root, and the login node's hierarchical density (its own
22:45 slot empty between a used 22:00 and a used 23:00, the root's
predictive as prior) put the PUB login 10.60.123.142 (Saturday 22:47) at
p_when 4.9e-4: a MEDIUM incident and D4's failure. The ordering of a flat
stretch is noise (a 5 % difference of a 24-hour source's learned rows);
smoothing the density over +-15 minutes before the HDR table takes it to
5.7e-2 there, while an empty slot in an empty stretch (A4's 03:05: 1.8e-4 ->
1.5e-4 on the same run) stays rare.
"""
from __future__ import annotations

import numpy as np

from app.engines.behavior import conformity as CF
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.lib import pscore as SC

from test_p03_conformity import CFG, GA, LOGIN, TZ, Fx, local_minute, workdays  # noqa: F401

OTHERS = [f"10.60.{i}.{i + 10}" for i in range(1, 13)]
MONITOR = "192.168.9.9"
HEALTH = "GET oa /health"


def _learn(fx: Fx, routes, ip: str, route: str, ts: float, w: float = 1.0) -> None:
    tr = fx.tree("oa", routes)
    hier = MP.hierarchies(fx.st, "oa", CFG)
    keys = [hier.gen("net.src", l, ip) for l in range(PN.WHO_LEVELS)]
    day = int((ts + 8 * 3600) // 86400.0)
    attrs = {"http.route": route, "net.src": ip}
    for nid in tr.route(lambda a: attrs.get(a), hier, (), ts):
        tr.nodes[nid].update_core(ts, w, w, keys, ip, 0, local_minute(ts), day)
    for nd in tr.nodes.values():
        if nd.state == "candidate" and nd.n_c(ts) >= 1:
            nd.state = "confirmed"


def _evening_portal(monitor_w: float):
    """A login node used 08:30-10:00 and in the evening 20:00-22:14 and
    23:00-23:14 (22:15-22:59 never: the population's evening law has a
    sampling gap there), under a root that also holds a 24-hour machine
    source (one row per 15 minutes on another route) learned at weight
    `monitor_w`, whose 22:45 rows are left unlearned on most days."""
    fx = Fx()
    routes = [LOGIN, HEALTH]
    rng = np.random.default_rng(11)
    days = workdays(30)
    for k, d in enumerate(days[:28]):
        for ip in OTHERS:
            _learn(fx, routes, ip, LOGIN, d + (8 * 60 + 30 + 90 * rng.random()) * 60.0)
        for ip in OTHERS[: 4 + k % 3]:
            m = 20 * 60 + 135 * rng.random() if rng.random() < 0.8 else 23 * 60 + 14 * rng.random()
            _learn(fx, routes, ip, LOGIN, d + m * 60.0)
        for s in range(96):
            if s == 91 and k % 4:                    # 22:45: the machine source's rows unlearned
                continue
            _learn(fx, routes, MONITOR, HEALTH, d + (15 * s + 7) * 60.0, monitor_w)
    tr = fx.tree("oa", routes)
    for nd in tr.nodes.values():
        nd.state = "stable"
    return fx, tr, fx.node("oa", LOGIN), days


def _tables(monitor_w: float):
    fx, tr, nd, days = _evening_portal(monitor_w)
    tc = CF._TreeCtx(fx.eng, fx.st, "oa", "oa", CFG, days[28])
    cur, _ = tc.when_table(0, nd, 0, tr)
    h, N = tc._when_counts(nd, 0)
    return cur, h, N


def test_a_slot_in_the_populations_evening_is_not_rare_whatever_the_machine_sources_weight(monkeypatch):
    late, night = int((22 * 60 + 47) // 15), int((3 * 60 + 5) // 15)
    for w in (0.4, 1.0):
        cur, h, N = _tables(w)
        assert h[late] == 0 and h[late - 1] == 0 and h[late + 1] > 0 and h[late - 3] > 0
        assert N >= 300
        assert cur[late] >= 1e-2, (w, cur[late])
        with monkeypatch.context() as m:                          # round 11: no smoothing
            m.setattr(SC, "when_hdr_table", lambda f: CF._hdr_table(np.asarray(f, dtype=np.float64)))
            raw, _, _ = _tables(w)
        assert raw[late] < 1e-3, (w, raw[late])                   # the D4 false alarm
        # the night stretch holds only the machine source's mass (through the
        # prior): its slots are tied with each other instead of ordered by
        # the noise of the source's rows, and stay well below the evening gap;
        # A4's 03:05 at a young department node stays <= 3e-4
        # (test_p03_round3::test_young_node_declares_an_empty_slot_rare...)
        st = slice(int(60 // 15), int(6 * 60 // 15))
        assert cur[st].max() - cur[st].min() < 0.1 * cur[st].max()
        assert cur[night] < 1e-2 and cur[night] < cur[late] / 3, (w, cur[night], cur[late])


def test_when_smooth_spreads_a_slot_over_its_neighbours_and_keeps_an_empty_stretch_empty():
    f = np.zeros(96)
    f[88] = 3.0
    f[92] = 7.0
    g = SC.when_smooth(f)
    assert abs(g.sum() - f.sum()) < 1e-12                          # mass preserved
    assert g[91] > 0 and g[89] > 0 and g[90] == 0                # one slot each side
    assert g[12] == 0                                             # 03:00 in an empty stretch
    h = np.zeros(96)
    h[95], h[0] = 1.0, 1.0
    assert SC.when_smooth(h)[1] > 0 and SC.when_smooth(h)[94] > 0  # the day is a circle


def test_a_flat_24_hour_stretch_has_no_rare_slot_and_an_empty_one_stays_rare():
    """A 24-hour machine source: every slot holds ~20 evidence units with
    +-10 % noise. Without ties the slot that happened to receive the fewest
    rows was the HDR minimum (pack O seed 1: the monitor 192.168.9.9's own
    rows at p_when 3e-3 - 1e-2, upgraded to a MEDIUM incident by fusion's
    evidence CUSUM on day 6)."""
    rng = np.random.default_rng(5)
    flat = 20.0 * (1.0 + 0.1 * rng.uniform(-1.0, 1.0, 96))
    f = SC.when_density(flat, float(flat.sum()))
    old = CF._hdr_table(f)
    new = SC.when_hdr_table(f)
    assert old.min() < 1e-2
    assert new.min() > 0.05
    # a login node: a day of 08:30-10:00 logins, the night empty
    day = np.zeros(96)
    day[34:40] = 60.0
    g = SC.when_density(day, float(day.sum()))
    assert SC.when_hdr_table(g)[12] < 1e-3                       # 03:00
    assert SC.when_hdr_table(g)[37] > 0.3
    # tie = 0 is the mid-p HDR table of the smoothed density
    assert np.allclose(SC.when_hdr_table(f, tie=0.0), CF._hdr_table(SC.when_smooth(f)))
