"""P04 round 6 (tree owner): the structural-drift loss of a node active on both
day types is coded per day type (pattern_tree.Coder.pred_daytype), and its
levels restart with the coder."""
from __future__ import annotations

import numpy as np

from ptree_sim import DAY, MON, Sim, daily, is_workday

from app.engines.behavior import pattern_tree as P4


def _monitor(weekend_mix, workday_mix):
    def day(d, t0, rng):
        mix = workday_mix if is_workday(t0) else weekend_mix
        vals, p = zip(*mix)
        out = []
        for k in range(144):
            v = vals[int(rng.choice(len(vals), p=np.asarray(p) / sum(p)))]
            out.append((t0 + k * 600.0 + 5.0, "fin", "192.168.9.9",
                        {"http.route": "GET fin /health", "http.sclass": v,
                         "net.dur_ms": float(rng.lognormal(3.0 if is_workday(t0) else 4.0, 0.3))}))
        return out
    return day


def _states(sim, days):
    out = []
    for d in range(days):
        sim.run_until(MON + (d + 1) * DAY)
        tr = sim.tree("fin")
        if tr is None or len(tr.nodes) < 2:
            out.append(None)
            continue
        nd = tr.nodes[sim.top("fin")]
        out.append((nd.state, bool(nd.meta.get("drift"))))
    return out


def test_a_24x7_monitor_is_not_read_as_changing_on_every_weekend():
    """A health monitor whose weekend responses differ from its workdays'
    (a maintenance mode) is ONE stable pattern. Coded by one pooled predictive,
    its weekend events were judged against counts that drift towards the
    workday mix during every working week, and the first weekend day of a week
    read as a structural change (pack O seed 2, finance /health; AUTO.monitor.
    finance missed at day 14)."""
    sel = {"targets_sys": {0: ["http.sclass", "net.dur_ms"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)
    sim.add(daily(_monitor([("2xx", 0.3), ("5xx", 0.7)], [("2xx", 0.97), ("5xx", 0.03)]), 36, seed=3))
    st = _states(sim, 36)
    late = [s for s in st[14:] if s is not None]
    assert late and all(s[0] in ("confirmed", "stable") for s in late), st
    drifts = [e for e in sim.events("pattern_drift") if (e.extra or {}).get("kind") == "structural"]
    assert not drifts, [(e.ts, e.description) for e in drifts]


def test_an_open_structural_alarm_closes_when_the_coder_is_replaced():
    """An alarm opened under one coder is judged with levels coded by it; once
    P04 replaces the coder (a retarget, a new episode) the days are coded
    differently and the comparison reads the encoding change, not the node's
    (pack O seed 2, finance /health: retargeted on Monday 15, coded 2.95 bits
    against the old workday level 2.05, `evolving` - AUTO.monitor.finance
    missed). The alarm closes and the node returns to its state."""
    sel = {"targets_sys": {0: ["http.sclass", "net.dur_ms"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)
    sim.add(daily(_monitor([("2xx", 0.3), ("5xx", 0.7)], [("2xx", 0.97), ("5xx", 0.03)]), 14, seed=5))
    sim.run_until(MON + 9 * DAY + 10 * 3600)                   # Wednesday of week 2, 10:00
    tr = sim.tree("fin")
    nd = tr.nodes[sim.top("fin")]
    assert nd.state in ("confirmed", "stable"), nd.state
    lc = P4._LC(sim.p04, sim.st, "fin", sim.now, sim.cfg, 8 * 3600.0)
    from app.engines.behavior.lib import m_ptree as MP
    lc.m = MP.get_ptree(sim.st, "fin")
    lc.aux = sim.p04.aux(lc.m)
    sim.p04._structural_alarm(lc, tr, nd, sim.now)
    dr = nd.meta["drift"]
    dr["ref"] = 0.5                                             # far below anything the new coder codes
    prev = nd.state
    nd.state = "evolving"
    nd.meta["C"] = sim.p04._new_coder(lc, nd, list(nd.meta["C"].targets), sim.now, tr)
    sim.run_until(MON + 9 * DAY + 14 * 3600)
    assert "drift" not in nd.meta
    assert nd.state == prev


def test_coder_day_type_counts_start_from_the_pooled_workday_counts():
    c = P4.Coder(["a"], [[("x", 0.5), ("y", 0.5)]])
    for _ in range(10):
        c.add(c.bins(["x"]), 1.0, 0)
    assert c.lcd is None                                    # workday-only: pooled = workday
    p0 = c.pred_daytype(0)
    assert np.allclose(p0, c.pred())
    c.add(c.bins(["y"]), 1.0, 1)                           # first weekend row
    assert c.lcd is not None
    assert c.lcd[0, 0, c.bins(["x"])[0]] == 10.0 and c.lcd[1, 0, c.bins(["y"])[0]] == 1.0
    for _ in range(20):
        c.add(c.bins(["y"]), 1.0, 1)
    pw, pn = c.pred_daytype(0), c.pred_daytype(1)
    bx, by = c.bins(["x"])[0], c.bins(["y"])[0]
    assert pw[0, bx] > 0.6 and pn[0, by] > 0.7             # each day type predicts its own mix


def test_a_holiday_week_without_traffic_does_not_make_patterns_stale():
    """A configured holiday is not a missed workday, whether or not the
    system had traffic on it (P01's calendar record exists only for days with
    events). O-real R8 (holiday days 22-28): every CRM branch node went
    `stale` with a pattern_absent event on day 22 - the days were classed by
    their weekday."""
    import datetime as _dt
    from ptree_sim import CFG, local
    hol = [(local(MON) + _dt.timedelta(days=14 + k)).date().isoformat() for k in range(5)]
    cfg = dict(CFG, calendar={"holidays": hol, "makeup_workdays": []})

    def day(d, t0, rng):
        if not is_workday(t0) or local(t0).date().isoformat() in hol:
            return []
        out = []
        for u, ip in enumerate(("192.168.111.1", "192.168.111.2", "192.168.111.3")):
            t = t0 + 9 * 3600 + rng.uniform(0, 3600)
            for k in range(6):
                out.append((t + k * 300.0, "crm-01", ip, {"http.route": "GET crm /crm/customer/{num}",
                                                          "net.bytes_down": float(rng.integers(2000, 9000))}))
        return out
    sel = {"targets_sys": {0: ["net.bytes_down"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel, config=cfg)
    sim.add(daily(day, 24, seed=8))
    states = []
    for d in range(12, 24):
        sim.run_until(MON + (d + 1) * DAY)
        tr = sim.tree("crm-01")
        states.append((d, tr.nodes[sim.top("crm-01")].state))
    assert all(s in ("confirmed", "stable") for _, s in states), states
    assert not [e for e in sim.events("pattern_absent")
                if MON + 14 * DAY <= e.ts <= MON + 21 * DAY], "pattern_absent during the holiday"


def test_a_target_summary_remade_on_another_scale_is_not_a_drift():
    """P04 keeps <= M_T + 4 target summaries per node and re-makes a dropped
    one with P02's current typing - possibly on the other scale (log /
    linear). The daily-mean Page-Hinkley history of the attribute was on the
    old scale: the first day on the new one read as a jump of thousands of
    standard deviations (O-real seed 0, CRM branches after the holiday:
    net.resp_len 11 930 against a log-scale mu 9.2) and the node stayed
    `evolving` to the end of the run."""
    from app.engines.behavior.lib import pnode as PN

    def day(d, t0, rng):
        if not is_workday(t0):
            return []
        return [(t0 + 9 * 3600 + k * 400.0, "crm", f"192.168.111.{1 + k % 3}",
                 {"http.route": "GET crm /crm/customer/{num}", "net.resp_len": float(rng.uniform(2000, 20000))})
                for k in range(40)]
    sel = {"targets_sys": {0: ["net.resp_len"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)
    sim.add(daily(day, 30, seed=9))
    sim.run_until(MON + 17 * DAY)
    tr = sim.tree("crm")
    nd = tr.nodes[sim.top("crm")]
    old = nd.targets["net.resp_len"]
    assert isinstance(old, PN.NumSummary)
    nd.targets["net.resp_len"] = PN.NumSummary(log=not old.log)      # re-made on the other scale
    states = []
    for d in range(17, 30):
        sim.run_until(MON + (d + 1) * DAY)
        states.append((d, nd.state, sorted((nd.meta.get("evolving") or {}).keys())))
    assert all(s in ("confirmed", "stable") and not ev for _, s, ev in states), states


def test_a_row_records_a_bounded_number_of_attributes():
    """Waiting route rows and per-source extremes record the attributes P05
    keeps; bounded at ROW_ATTRS_MAX (system targets first, then by utility),
    so P04's per-row work does not grow with the number of kept roles (PG4's
    340-attribute point: 70-80 split roles per system)."""
    class _Stub:
        def __init__(self, sel):
            self.rpart_attrs = {}
            self._sel = sel

        def selection(self, kind):
            return self._sel
    roles = {f"meta.f{i:03d}": "split" for i in range(90)}
    roles.update({"body.len": "target", "net.bytes_down": "target"})
    ustat = {a: {"U_t": 1.0 / (i + 1)} for i, a in enumerate(sorted(roles))}
    sel = {"targets_sys": {0: ["net.bytes_down"]}, "roles": roles, "ustat": ustat}
    attrs = P4.PatternTreeEngine._row_attrs(None, _Stub(sel), 0)
    assert len(attrs) <= P4.ROW_ATTRS_MAX
    assert attrs[0] == "net.bytes_down"
    best = sorted([a for a in roles if a != "net.bytes_down"], key=lambda a: -ustat[a]["U_t"])
    assert set(best[:P4.ROW_ATTRS_MAX - 1]) == set(attrs[1:])
    small = {"targets_sys": {0: ["net.bytes_down"]}, "roles": {"body.len": "target", "x.y": "split"}}
    assert P4.PatternTreeEngine._row_attrs(None, _Stub(small), 0) == ["net.bytes_down", "body.len", "x.y"]
