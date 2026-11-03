"""P04 / P07 / P08, round 11 (tree owner)."""
from __future__ import annotations

from ptree_sim import DAY, MON, Sim, daily

NEWS = {"http.route": "GET portal /news/{num}", "http.method": "GET"}


def test_a_target_the_node_no_longer_learns_is_not_kept_as_its_present():
    """Item 1a (round 10, pack O seed 2): portal GET /news/{num} learned
    net.win on a few rows of one client (65535) while it was a target, then
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
            if d < 1 and win != 65535.0:
                continue                         # the first day: one client kind only
            for _ in range(2):
                out.append((t0 + rng.uniform(8 * 3600, 20 * 3600), "portal", ip,
                            dict(NEWS, **{"net.bytes_down": float(rng.uniform(3000, 9000)), "net.win": win})))
        return out
    sim.add(daily(day, 9, seed=5))

    def retarget(s):
        if s.now >= MON + 1 * DAY:
            s.sel = {"targets_sys": {0: ["net.bytes_down"]}, "split_cands": {0: []}, "roles": {}}
    sim.hooks.append(retarget)
    sim.run_until(MON + 2 * DAY)
    tr = sim.tree("portal")
    nd = tr.nodes[sim.top("portal")]
    assert "net.win" in nd.targets and "net.bytes_down" in nd.targets
    assert [str(k) for k, *_ in nd.targets["net.win"].ss.items(sim.now)] == ["65535.0"]
    sim.run_until(MON + 9 * DAY)
    nd = tr.nodes[sim.top("portal")]
    assert "net.win" not in nd.targets                      # dropped: no longer learned
    assert "net.bytes_down" in nd.targets                   # still a target: kept
    if isinstance(nd.ref, dict):
        assert "net.win" not in (nd.ref.get("hold") or {})
        assert "net.win" not in (nd.ref.get("targets") or {})
