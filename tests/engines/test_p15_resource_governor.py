"""P15 ResourceGovernor (progressive.md §6.19, §6.20, §10.1; card P15)."""
from __future__ import annotations

import time

import numpy as np
import pytest

from helpers import ctx, make_store

from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pactive as PA
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.resource_governor import ResourceGovernorEngine
from app.models.schema import ORG, SYSTEM_ENTITY, BehaviorEvent, Incident, Observation

T0 = 1_790_000_000.0
DAY = 86400.0
CFG = {"progressive": {"enabled": True}, "tz": "Asia/Shanghai"}


def put_batch(st, s, t1, ips, dt=3600.0):
    b = EV.BatchBuilder(s, EV.KIND_TXN)
    for i, ip in enumerate(ips):
        b.add(t1 - dt + 1.0 + i, ip, {"net.src": ip, "ev.ch": "http", "http.route": "GET x /a"}, 1.0)
    batch = b.build(t1 - dt, t1)
    batch.learn[:] = True
    st.add_batch(s, EV.EVT_BATCH, t1, batch)


def budget(st):
    return st.get_model(ORG, ORG, MP.BUDGET)


def test_ladder_one_step_per_three_over_ticks_and_recovery_after_24h():
    st = make_store()
    eng = ResourceGovernorEngine()
    cfg = dict(CFG, progressive={"enabled": True, "budget": {"pcore_cpu_share": 0.01}})
    dt = 60.0
    steps = []
    t = T0
    for k in range(40):                                        # 40 min of heavy P-core work
        t += dt
        st.put_health("behavior.pattern_tree", {"ts": t - dt, "duration_ms": 3000.0})
        eng.safe_run(ctx(st, t, window_s=dt, config=cfg), None)
        steps.append(budget(st)["ladder"]["step"])
    # the 1-h CPU window needs 15 min before it can judge; then one step per 3 ticks
    first = steps.index(1)
    assert first >= 14
    assert steps[first + 1] == 1 and steps[first + 2] == 1 and steps[first + 3] == 2
    assert steps[-1] <= 7
    # step 5 is skipped outside bounded mode
    assert 5 not in steps
    lad = budget(st)["ladder"]
    assert lad["no_explore"] and (lad["skip_body_parsing"] == (lad["step"] >= 7))
    top = steps[-1]
    # idle afterwards: undo one step per 24 h under 70 % of budget
    for k in range(int(3 * DAY / 600)):
        t += 600.0
        eng.safe_run(ctx(st, t, window_s=600.0, config=cfg), None)
    assert budget(st)["ladder"]["step"] in (top - 2, top - 3)
    # caps follow the ladder: e_rate halved while step >= 1
    put_batch(st, "oa", t + 600.0, ["10.0.0.1"])
    eng.safe_run(ctx(st, t + 600.0, window_s=600.0, config=cfg), None)
    caps = budget(st)["trees"]["oa"]
    assert caps["e_rate"] == (5.0 if budget(st)["ladder"]["step"] >= 1 else 10.0)


def test_step7_flags_reach_the_consumers():
    st = make_store()
    eng = ResourceGovernorEngine()
    cfg = dict(CFG, progressive={"enabled": True, "budget": {"pcore_cpu_share": 1e-6}})
    t = T0
    for k in range(80):
        t += 60.0
        put_batch(st, "oa", t, ["10.0.0.1"], dt=60.0)
        st.put_health("raw.event", {"ts": t - 60.0, "duration_ms": 100.0})
        eng.safe_run(ctx(st, t, window_s=60.0, config=cfg), None)
    b = budget(st)
    assert b["ladder"]["step"] == 7
    caps = MP.budget_for(st, "oa")
    assert caps["skip_body_parsing"] is True and caps["score_sample_k"] > 1


def test_active_set_is_active_plus_open_state_not_all_known():
    """10 000 known IPs, 10 active this tick, 1 with an open incident, 1 with a
    B28 regime event: A_t = 12, and P15's cost does not grow with known IPs."""
    def run(n_known):
        st = make_store()
        for i in range(n_known):
            st.register_entity("oa", f"10.1.{i // 256}.{i % 256}")
        eng = ResourceGovernorEngine()
        cfg = dict(CFG, lib3={"resource_mode": "bounded", "linger_s": 1800.0})
        t = T0 + 2 * DAY
        st.put_incident(Incident(id="i1", system="oa", entity="10.1.0.200", opened=t - 100,
                                 last_seen=t - 100, status="open"))
        st.add_event(BehaviorEvent(system="oa", entity="10.1.0.201", ts=t - 3600, kind="regime",
                                   score=0.0))
        act = [f"10.1.0.{i}" for i in range(10)]
        for o in act:
            st.add_observation(Observation(ts=t - 5.0, system="oa", entity=o))
        t0 = time.perf_counter()
        for _ in range(5):
            eng.safe_run(ctx(st, t, window_s=60.0, config=cfg), None)
        el = (time.perf_counter() - t0) / 5
        return st, el
    st, el_small = run(100)
    rec = budget(st)["systems"]["oa"]
    assert rec["n_active"] == 12 and "10.1.0.200" in rec["active"] and "10.1.0.201" in rec["active"]
    assert PA.entities(st, "oa", T0 + 2 * DAY, {"lib3": {"resource_mode": "bounded"}}) == rec["active"]
    # full mode: every known entity (unchanged behaviour)
    assert len(PA.entities(st, "oa", T0 + 2 * DAY, {})) == 100
    st2, el_big = run(10_000)
    assert budget(st2)["systems"]["oa"]["n_active"] == 12
    assert el_big <= 3.0 * el_small + 2e-3, (el_small, el_big)


def test_linger_window_keeps_recent_ips_then_drops_them():
    st = make_store()
    eng = ResourceGovernorEngine()
    cfg = dict(CFG, lib3={"resource_mode": "bounded", "linger_s": 3600.0})
    t = T0
    put_batch(st, "oa", t, ["10.0.0.1", "10.0.0.2"], dt=60.0)
    eng.safe_run(ctx(st, t, window_s=60.0, config=cfg), None)
    assert budget(st)["systems"]["oa"]["active"] == ["10.0.0.1", "10.0.0.2"]
    for k in range(1, 70):
        t += 60.0
        put_batch(st, "oa", t, ["10.0.0.2"], dt=60.0)
        eng.safe_run(ctx(st, t, window_s=60.0, config=cfg), None)
    assert budget(st)["systems"]["oa"]["active"] == ["10.0.0.2"]


def test_idle_trees_get_xs_and_are_checkpointed_after_30_days_then_restored():
    """300 systems, 20 active: the 280 idle trees are tier XS after a day, leave
    memory after 30 idle days and come back on their next event."""
    st = make_store()
    eng = ResourceGovernorEngine()
    systems = [f"s{i:03d}" for i in range(300)]
    t = T0
    for s in systems:
        MP.ensure_ptree(st, s, t).tree(EV.KIND_TXN, t)
        put_batch(st, s, t, ["10.0.0.1"])
    eng.safe_run(ctx(st, t, window_s=3600.0, config=CFG), None)
    active = systems[:20]
    hours = list(range(1, 49)) + list(range(54, 31 * 24 + 7, 6))    # hourly, then 6-hourly ticks
    prev = 0
    for h in hours:
        t = T0 + h * 3600.0
        for s in active:
            put_batch(st, s, t, ["10.0.0.1", "10.0.0.2"], dt=3600.0 * (h - prev))
        eng.safe_run(ctx(st, t, window_s=3600.0 * (h - prev), config=CFG), None)
        prev = h
        if h == 26:
            trees = budget(st)["trees"]
            assert all(trees[s]["tier"] == "XS" and trees[s]["idle"] for s in systems[20:])
            assert all(not trees[s]["idle"] for s in active)
    b = budget(st)
    assert set(b["evicted"]) == set(systems[20:])
    assert all(MP.get_ptree(st, s) is None for s in systems[20:])
    assert all(MP.get_ptree(st, s) is not None for s in active)
    # restore on the next event, before the learners run
    t += 3600.0
    put_batch(st, "s250", t, ["10.9.9.9"])
    eng.safe_run(ctx(st, t, window_s=3600.0, config=CFG), None)
    assert MP.get_ptree(st, "s250") is not None and "s250" not in budget(st)["evicted"]


def test_allocation_water_fills_by_weight_within_budget():
    st = make_store()
    eng = ResourceGovernorEngine()
    cfg = dict(CFG, progressive={"enabled": True, "budget": {"mem_mb_total": 60}})
    t = T0
    for s, n in (("big", 400), ("mid", 40), ("small", 4)):
        MP.ensure_ptree(st, s, t)
        st.put_model(s, SYSTEM_ENTITY, MP.SYSPROF, {"chosen": {"tier": "M"},
                                                    "characteristics": {"criticality": 1.0}})
    for h in range(1, 6):
        t = T0 + h * 3600.0
        for s, n in (("big", 400), ("mid", 40), ("small", 4)):
            put_batch(st, s, t, [f"10.0.{k // 256}.{k % 256}" for k in range(n)])
        eng.safe_run(ctx(st, t, window_s=3600.0, config=cfg), None)
    trees = budget(st)["trees"]
    total = sum({"XS": 1, "S": 5, "M": 30, "L": 60}[c["tier"]] for c in trees.values())
    assert total <= 0.8 * 60
    order = ["XS", "S", "M", "L"]
    assert order.index(trees["big"]["tier"]) >= order.index(trees["mid"]["tier"]) >= \
        order.index(trees["small"]["tier"])
    assert trees["big"]["tier"] == "M"
    # LRU caps are sized to the sources seen, not to a fixed maximum
    assert trees["small"]["s_sess"] <= 65536


def test_earned_set_top_emax_with_demotion_hysteresis():
    st = make_store()
    eng = ResourceGovernorEngine()
    cfg = dict(CFG, lib3={"resource_mode": "bounded"})
    ips = {f"10.0.0.{i}": {"g": 3.0 * 60 * (1 + i / 100), "n": 60} for i in range(50)}
    ips["10.0.0.99"] = {"g": 0.0, "n": 10, "forced": True}
    st.put_model("oa", SYSTEM_ENTITY, PA.EARNED, {"ips": ips})
    st.put_model("oa", SYSTEM_ENTITY, MP.SYSPROF, {"chosen": {"e_max": "32"}})
    t = T0
    put_batch(st, "oa", t, ["10.0.0.1"])
    eng.safe_run(ctx(st, t, window_s=3600.0, config=cfg), None)
    e = budget(st)["systems"]["oa"]["earned"]
    assert len(e) == 32 and "10.0.0.99" in e and "10.0.0.49" in e and "10.0.0.0" not in e
    assert PA.is_earned(st, "oa", "10.0.0.49", cfg) and not PA.is_earned(st, "oa", "10.0.0.0", cfg)
    assert PA.is_earned(st, "oa", "10.0.0.0", {})            # full mode: no restriction
    # 10.0.0.49 drops below tau/2: kept for 2 more daily checks, demoted on the 3rd
    ips["10.0.0.49"] = {"g": 0.5 * 60, "n": 60}
    kept = []
    for d in range(1, 5):
        t = T0 + d * DAY
        put_batch(st, "oa", t, ["10.0.0.1"])
        eng.safe_run(ctx(st, t, window_s=3600.0, config=cfg), None)
        kept.append("10.0.0.49" in budget(st)["systems"]["oa"]["earned"])
    assert kept == [True, True, False, False]


def test_inert_without_progressive_or_bounded():
    st = make_store()
    eng = ResourceGovernorEngine()
    put_batch(st, "oa", T0, ["10.0.0.1"])
    assert eng.safe_run(ctx(st, T0, window_s=60.0, config={}), None) == 0
    assert st.get_model(ORG, ORG, MP.BUDGET) is None


def test_pactive_before_p15_uses_previous_sets_plus_this_ticks_observations():
    """Raw / derived engines run before P15 in a tick: they see the previous
    tick's sets plus the entities observed since, and the full list when no
    fresh sets exist (bounded mode never drops an entity it knows nothing about)."""
    st = make_store()
    cfg = {"lib3": {"resource_mode": "bounded"}}
    for i in range(50):
        st.register_entity("oa", f"10.0.0.{i}")
    assert len(PA.entities(st, "oa", T0, cfg)) == 50                  # no sets yet
    st.put_model(ORG, ORG, MP.BUDGET, {"systems": {"oa": {"ts": T0, "active": ["10.0.0.1"],
                                                           "earned": ["10.0.0.2"]}}})
    st.add_observation(Observation(ts=T0 + 30.0, system="oa", entity="10.0.0.7"))
    assert PA.entities(st, "oa", T0 + 60.0, cfg) == ["10.0.0.1", "10.0.0.2", "10.0.0.7"]
    assert PA.entities(st, "oa", T0 + 60.0, cfg, include_earned=False) == ["10.0.0.1", "10.0.0.7"]
    assert len(PA.entities(st, "oa", T0 + 3 * 3600.0, cfg)) == 50      # stale sets: full list
    assert PA.periodic_candidate(st, "oa", "10.0.0.2", cfg)            # earned
    assert not PA.periodic_candidate(st, "oa", "10.0.0.1", cfg)        # neither earned nor periodic
    assert PA.periodic_candidate(st, "oa", "10.0.0.1", {})             # full mode
