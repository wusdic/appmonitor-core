"""P04 / P07 / P08, round 11 (tree owner)."""
from __future__ import annotations

import numpy as np

from app.engines.behavior.lib import pnode as PN

import math

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


def test_a_dropped_target_leaves_no_drift_state_behind():
    """A dropped summary takes its content drift detector's daily means, its
    drift episode and its invariant with it (pattern_tree._forget_target).
    Kept, a summary made again for the attribute later was judged against
    the old summary's daily means (the round-6 note on re-made summaries)."""
    both = {"targets_sys": {0: ["net.bytes_up", "net.bytes_down"]}, "split_cands": {0: []}, "roles": {}}
    one = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=both)
    clients = [f"10.60.1.{i}" for i in range(20)]

    def day(d, t0, rng):
        return [(t0 + rng.uniform(8 * 3600, 20 * 3600), "portal", ip,
                 dict(NEWS, **{"net.bytes_up": float(rng.uniform(800, 1200)),
                               "net.bytes_down": float(rng.uniform(3000, 9000))}))
                for ip in clients for _ in range(3)]
    sim.add(daily(day, 11, seed=7))
    sim.hooks.append(lambda s: setattr(s, "sel", both if s.now < MON + 6 * DAY else one))
    sim.run_until(MON + 6 * DAY)
    nd = sim.tree("portal").nodes[sim.top("portal")]
    assert any(str(k).startswith("net.bytes_down|") for k in (nd.meta.get("dph") or {}))
    sim.run_until(MON + 11 * DAY)
    nd = sim.tree("portal").nodes[sim.top("portal")]
    assert "net.bytes_down" not in nd.targets
    assert not any(str(k).startswith("net.bytes_down|") for k in (nd.meta.get("dph") or {}))
    assert "net.bytes_down" not in (nd.meta.get("evolving") or {})
    assert "net.bytes_down" not in (nd.meta.get("tgd") or {})
    assert any(str(k).startswith("net.bytes_up|") for k in (nd.meta.get("dph") or {}))


def test_a_group_part_is_held_out_tested_on_the_bands_it_states():
    """Item 1b (round 11; pack O seed 2, oa GET /docs/{}): P14 states a tracked
    group's part of a shared node with the GROUP's own bands (part_content),
    but the group's held-out record checked the NODE's bands on the group's
    rows. A group in the tail of the node's distribution failed every test of
    a band its part does not state, and the part (views.part_hold: the group's
    tests pooled with the node's rate) was stated 0.5-0.8 while it held every
    evaluator test. The group's record now checks the group's own band (the
    reference's 'hold_g', built from the group's earlier rows only)."""
    from app.engines.behavior.pattern_tree import PatternTreeEngine, _group_bands
    from app.engines.behavior import views as VW
    rng = np.random.default_rng(3)
    nd = PN.Node(1, 0, 1, 0, (), MON)
    nd.state = "confirmed"
    node_band = ("num", 100.0, 1000.0, 0.9)          # the node's band: fitted on its (mostly G1) rows
    nd.ref = {"t": MON, "hold": {"body.len": node_band}}
    t = MON
    for d in range(21):
        day = 739900 + d
        if d > 0:                                   # the daily reference: built from earlier rows only
            nd.ref = {"t": t, "hold": {"body.len": node_band},
                      "hold_g": _group_bands(nd, t, {"body.len": node_band}, day)}
        for k in range(200):
            t = MON + d * DAY + 8 * 3600 + k * 120.0
            g2 = k % 8 == 0                          # G2: 1/8 of the rows, all above the node's band
            g = "grp:G2" if g2 else "grp:G1"
            ip = f"10.0.{2 if g2 else 1}.{k % 5 + 1}"
            keys = [ip, ip.rsplit(".", 1)[0] + ".0/24", "10.0.0.0/16", g, "reg:∅"]
            v = float(rng.uniform(1200.0, 1500.0) if g2 else rng.uniform(150.0, 900.0))
            nd.update_core(t, 1.0, 1.0, keys, ip, 0, 600.0, day)
            PatternTreeEngine._hold_check(None, nd, lambda a, v=v: v if a == "body.len" else None,
                                          keys, 0, 600.0, t, 1.0, 1.0, day)
    gb = nd.ref.get("hold_g") or {}
    assert "grp:G2" in gb and gb["grp:G2"]["body.len"][1] > 1000.0      # G2's own band
    p_node = nd.p_hold(t)
    # the part states G2's own band and G2's rows keep to it: its confidence is
    # the node's, not dragged down by tests of the node's band
    assert VW.part_hold(nd, ["G2"], 0.95) > 0.9, (nd.meta["hold_g"]["grp:G2"].tests(), p_node)
    assert VW.part_hold(nd, ["G1"], 0.95) > 0.9


def test_a_group_part_is_not_held_out_tested_on_the_nodes_who_list():
    """Item 1b (round 11; pack O seed 2, mail TLS / oa login parts): a group's
    part states the GROUP's members as its who; its held-out record checked
    the NODE's who (the node's listed heavy set) on the group's rows, so a
    group of light sources outside that list failed every test on 'who' and
    its part was stated far below a node that held every test. The group's
    record no longer judges the node's who list; the node's record still does."""
    from app.engines.behavior.pattern_tree import PatternTreeEngine
    from app.engines.behavior import views as VW
    nd = PN.Node(1, 0, 1, 0, (), MON)
    nd.state = "confirmed"
    heavy = frozenset(f"10.0.1.{i}" for i in range(1, 6))
    nd.ref = {"t": MON, "hold": {"who": (0, heavy, 0.8), "when": ("win", {0: ((480.0, 720.0),)}, 0.9)}}
    t = MON
    for k in range(12000):                      # 42 days: G2 (light, unlisted sources) is 1/20 of the rows
        t += 300.0
        g2 = k % 20 == 0
        ip = f"10.0.2.{k % 3 + 1}" if g2 else f"10.0.1.{k % 5 + 1}"
        keys = [ip, ip.rsplit(".", 1)[0] + ".0/24", "10.0.0.0/16", "grp:G2" if g2 else "grp:G1", "reg:∅"]
        PatternTreeEngine._hold_check(None, nd, lambda a: None, keys, 0, 600.0, t, 1.0)
    assert nd.p_hold(t) > 0.85                                  # the node's who holds (95 % listed)
    ps, n = nd.meta["hold_g"]["grp:G2"].tests()
    assert n > 3 and ps == n                                    # G2's tests judge its window, and pass
    assert VW.part_hold(nd, ["G2"], nd.p_hold(t)) >= nd.p_hold(t) - 1e-9
    assert nd.meta["hold"].F.get("who", 0.0) == 0.0
