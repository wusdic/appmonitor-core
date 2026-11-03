"""P04 / P07 / P08, round 11 (tree owner)."""
from __future__ import annotations

from ptree_sim import DAY, MON, Sim, daily

NEWS = {"http.route": "GET portal /news/{num}", "http.method": "GET"}


def test_a_target_the_node_no_longer_learns_is_not_kept_as_its_present():
    """Item 1a (round 10, pack O seed 2): portal GET /news/{num} learned
    net.win on rows of one client kind (65535) while it was a target, then
    P05's node targets left it out. The frozen summary stayed: P07 stated
    'net.win closed {65535}' (nominal 0.89) for a week while a third of the
    node's rows carried 65535, every held-out test failed on that clause and
    the statement stood at 0.03-0.17. A summary the node has not learned for
    two daily checks is dropped (no fitter, statement or held-out test reads
    it); one still targeted is kept."""
    sel = {"targets_sys": {0: ["net.bytes_down", "net.win"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)
    clients = [(f"10.60.1.{i}", 65535.0 if i % 3 == 0 else 64240.0) for i in range(30)]

    def day(d, t0, rng):
        out = []
        for i, (ip, win) in enumerate(clients):
            if d < 3 and win != 65535.0:
                continue                         # the first days: one client kind only
            for _ in range(2):
                out.append((t0 + rng.uniform(8 * 3600, 20 * 3600), "portal", ip,
                            dict(NEWS, **{"net.bytes_down": float(rng.uniform(3000, 9000)), "net.win": win})))
        return out
    sim.add(daily(day, 10, seed=5))

    def retarget(s):
        if s.now >= MON + 3 * DAY:
            s.sel = {"targets_sys": {0: ["net.bytes_down"]}, "split_cands": {0: []}, "roles": {}}
    sim.hooks.append(retarget)
    sim.run_until(MON + 3 * DAY)
    tr = sim.tree("portal")
    nd = tr.nodes[sim.top("portal")]
    assert "net.win" in nd.targets and "net.bytes_down" in nd.targets
    assert [str(k) for k, *_ in nd.targets["net.win"].ss.items(sim.now)] == ["65535.0"]
    sim.run_until(MON + 10 * DAY)
    nd = tr.nodes[sim.top("portal")]
    assert "net.win" not in nd.targets                      # dropped: no longer learned
    assert "net.bytes_down" in nd.targets                   # still a target: kept
    if isinstance(nd.ref, dict):
        assert "net.win" not in (nd.ref.get("hold") or {})
        assert "net.win" not in (nd.ref.get("targets") or {})


def test_a_target_taken_in_and_out_of_the_node_slots_is_kept():
    """P05 refits a node's target slots every few hours and an attribute near
    the cut moves in and out (pack O seed 0, crm POST /crm/visit: net.pkts_down
    untargeted at three consecutive midnights while learned between them); it
    is learned part of every day, not a frozen sample, and keeps its summary."""
    with_w = {"targets_sys": {0: ["net.bytes_down", "net.win"]}, "split_cands": {0: []}, "roles": {}}
    without = {"targets_sys": {0: ["net.bytes_down"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=with_w)
    clients = [(f"10.60.1.{i}", 65535.0 if i % 3 == 0 else 64240.0) for i in range(30)]

    def day(d, t0, rng):
        return [(t0 + rng.uniform(8 * 3600, 20 * 3600), "portal", ip,
                 dict(NEWS, **{"net.bytes_down": float(rng.uniform(3000, 9000)), "net.win": win}))
                for ip, win in clients for _ in range(2)]
    sim.add(daily(day, 10, seed=6))

    def flicker(s):
        # a target from 12:00 to 18:00 local time only: never at a daily check
        h = ((s.now - MON) % DAY) / 3600.0
        s.sel = with_w if (s.now < MON + 3 * DAY or 12.0 <= h < 18.0) else without
    sim.hooks.append(flicker)
    sim.run_until(MON + 10 * DAY)
    nd = sim.tree("portal").nodes[sim.top("portal")]
    assert "net.win" in nd.targets
    assert {str(k) for k, *_ in nd.targets["net.win"].ss.items(sim.now)} == {"65535.0", "64240.0"}
    # its history: every day's afternoon rows, not one afternoon's (~35 units)
    assert nd.targets["net.win"].ss.total_evidence(sim.now) > 150.0
