"""P04 round 5, evaluator (docs/lib3/progressive.md §16.13): the structural
drift check compares each day's mean coded loss with the pre-alarm level of
ITS day type (the ADWINs already run per day type)."""
from __future__ import annotations
from ptree_sim import DAY, MON, Sim, daily, is_workday


def test_structural_alarm_judges_each_day_against_its_own_day_types_level():
    """A 24x7 monitor's weekend loss sits above its workday loss. Before
    round 5 the pre-alarm level (meta loss_ew) pooled both day types, so
    after any ADWIN alarm every weekend day read 'higher' and the node went
    `evolving` (pack O seed 2: finance /health 2.9-3.0 bits on weekends
    against a pooled 2.12; AUTO.monitor.finance unrecovered at days 14 and
    21 on seeds 1-2). A weekend day at its own weekend level is not a
    change; a workday 0.8 bit above the workday level still is."""
    from app.engines.behavior import pattern_tree as P4
    from app.engines.behavior.lib import m_ptree as MP
    from app.engines.behavior.lib import psketch as PS
    sel = {"targets_sys": {0: ["net.dur_ms"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)

    def day(d, t0, rng):
        mu = 3.0 if is_workday(t0) else 4.0
        return [(t0 + k * 600.0 + 5.0, "oa", "192.168.9.9",
                 {"http.route": "GET oa /health", "net.dur_ms": float(rng.lognormal(mu, 0.3))})
                for k in range(144)]
    sim.add(daily(day, 12, seed=16))
    sim.run_until(MON + 11 * DAY + 20 * 3600)          # Friday of week 2, 20:00
    tr = sim.tree()
    nd = tr.nodes[sim.top()]
    assert nd.state in ("confirmed", "stable"), nd.state
    assert nd.meta.get("loss_ew") is not None and nd.meta.get("loss_ew_nwd") is not None
    # the node's pre-alarm loss: 2.0 bits on workdays, 3.0 on weekends (a 24x7 monitor)
    t = sim.now
    for key, lvl in (("loss_ew", 2.0), ("loss_ew_nwd", 3.0)):
        dv = nd.meta[key] = PS.DecayedVector([PS.H_M, PS.H_M])
        dv.add(t, [lvl * 100.0, 100.0])
    m = MP.get_ptree(sim.st, "oa")
    lc = P4._LC(sim.p04, sim.st, "oa", t, sim.cfg, 8 * 3600.0)
    lc.m, lc.aux = m, sim.p04.aux(m)
    sim.p04._structural_alarm(lc, tr, nd, t)
    dr = nd.meta["drift"]
    sat = int((t + 8 * 3600.0) // DAY) + 1
    dr["dsum"][sat], dr["dn"][sat] = 3.1 * 144, 144      # Saturday at its weekend level
    dr.setdefault("nwd", {})[sat] = 1
    sim.p04._drift_daily(lc, tr, nd, t + DAY, sat + 1, "oa", lambda a, b: b - a)
    assert nd.state in ("confirmed", "stable"), nd.state
    mon = sat + 2                                        # a workday 0.8 bit above its level
    dr["dsum"][mon], dr["dn"][mon] = 2.8 * 144, 144
    sim.p04._drift_daily(lc, tr, nd, t + 3 * DAY, mon + 1, "oa", lambda a, b: b - a)
    assert nd.state == "evolving"


def test_a_day_type_without_its_own_level_is_not_judged():
    """An encoding change restarts both levels with the ADWINs; an alarm on a
    Friday evening after such a restart carries a workday level only. Its
    weekend days are not judged against the workday level (pack O seed 3,
    day 13: portal's /health alarm with ref_nwd = NaN), nor against a level
    built on a handful of events."""
    from app.engines.behavior import pattern_tree as P4
    from app.engines.behavior.lib import m_ptree as MP
    from app.engines.behavior.lib import psketch as PS
    sel = {"targets_sys": {0: ["net.dur_ms"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)

    def day(d, t0, rng):
        return [(t0 + k * 600.0 + 5.0, "oa", "192.168.9.9",
                 {"http.route": "GET oa /health", "net.dur_ms": float(rng.lognormal(3.0, 0.3))})
                for k in range(144)]
    sim.add(daily(day, 12, seed=17))
    sim.run_until(MON + 11 * DAY + 20 * 3600)
    tr = sim.tree()
    nd = tr.nodes[sim.top()]
    t = sim.now
    nd.meta.pop("loss_ew_nwd", None)
    dv = nd.meta["loss_ew"] = PS.DecayedVector([PS.H_M, PS.H_M])
    dv.add(t, [2.0 * 100.0, 100.0])
    few = nd.meta["loss_ew_nwd"] = PS.DecayedVector([PS.H_M, PS.H_M])
    few.add(t, [1.0 * 5.0, 5.0])                          # 5 weekend events: no level
    m = MP.get_ptree(sim.st, "oa")
    lc = P4._LC(sim.p04, sim.st, "oa", t, sim.cfg, 8 * 3600.0)
    lc.m, lc.aux = m, sim.p04.aux(m)
    sim.p04._structural_alarm(lc, tr, nd, t)
    dr = nd.meta["drift"]
    assert dr["ref"] == 2.0 and dr["ref_nwd"] != dr["ref_nwd"]
    sat = int((t + 8 * 3600.0) // DAY) + 1
    dr["dsum"][sat], dr["dn"][sat] = 3.0 * 144, 144
    dr.setdefault("nwd", {})[sat] = 1
    sim.p04._drift_daily(lc, tr, nd, t + DAY, sat + 1, "oa", lambda a, b: b - a)
    assert nd.state in ("confirmed", "stable"), nd.state


def test_a_transient_structural_change_clears_after_quiet_days():
    """A structural alarm whose loss was higher on fewer than T_persist days
    and then back at its level for T_persist + 2 normal days is a transient:
    the node returns to its stated state. Before round 5 one higher day kept
    the alarm (and the node `evolving`) until expiry, 14 days (pack O seed 2:
    finance /health higher on days 13 and 15, back at its level from day 16,
    still `evolving` on day 21)."""
    from app.engines.behavior import pattern_tree as P4
    from app.engines.behavior.lib import m_ptree as MP
    from app.engines.behavior.lib import psketch as PS
    sel = {"targets_sys": {0: ["net.dur_ms"]}, "split_cands": {0: []}, "roles": {}}
    sim = Sim(sel=sel)

    def day(d, t0, rng):
        return [(t0 + k * 600.0 + 5.0, "oa", "192.168.9.9",
                 {"http.route": "GET oa /health", "net.dur_ms": float(rng.lognormal(3.0, 0.3))})
                for k in range(144)]
    sim.add(daily(day, 12, seed=18))
    sim.run_until(MON + 11 * DAY + 20 * 3600)
    tr = sim.tree()
    nd = tr.nodes[sim.top()]
    t = sim.now
    for key in ("loss_ew", "loss_ew_nwd"):
        dv = nd.meta[key] = PS.DecayedVector([PS.H_M, PS.H_M])
        dv.add(t, [2.0 * 100.0, 100.0])
    m = MP.get_ptree(sim.st, "oa")
    lc = P4._LC(sim.p04, sim.st, "oa", t, sim.cfg, 8 * 3600.0)
    lc.m, lc.aux = m, sim.p04.aux(m)
    sim.p04._structural_alarm(lc, tr, nd, t)
    dr = nd.meta["drift"]
    d0 = int((t + 8 * 3600.0) // DAY) + 1
    means = [2.9, 2.4, 2.8, 2.1, 2.0, 2.0, 2.0, 1.9]       # two higher days, then five quiet ones
    for i, mu in enumerate(means):
        dr["dsum"][d0 + i], dr["dn"][d0 + i] = mu * 144, 144
        sim.p04._drift_daily(lc, tr, nd, t + (i + 1) * DAY, d0 + i + 1, "oa", lambda a, b: b - a)
        if i == 0:
            assert nd.state == "evolving"
        if nd.meta.get("drift") is None:
            break
    assert nd.meta.get("drift") is None and nd.state in ("confirmed", "stable"), (i, nd.state)
    assert i == 7                                        # cleared after the 5th quiet day
