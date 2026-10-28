"""B13 BudgetEngine edge cases: training mode, cadence 900 -> 60 s, rollback
and release through model.control (bins carry their hour, so held rows come
back into the right bins), link seeding, model.link parsing and a perf check
with a generous bound."""
from __future__ import annotations

import math

import numpy as np
import pytest

from helpers import make_store, put_model, run_engine, set_trust, cpu_s

from test_b13_budget import DAY, E, H, S, T_MID, Feed, acc, budget_row, events

from app.engines.behavior import budget as BG
from app.engines.behavior.budget import MODEL, BudgetEngine, hourly_values


def local_day(ts: float) -> int:
    return int((ts + 8 * H) // DAY)            # Asia/Shanghai, no DST


def test_training_mode_learns_but_never_alarms():
    st, eng = make_store(), BudgetEngine()
    feed = Feed(st)
    dt = 3600.0
    rng = np.random.default_rng(1)
    for d in range(10):
        for k in range(24):
            now = T_MID + d * DAY + (k + 1) * dt
            up = 2e6 * math.exp(0.3 * rng.normal()) * (20.0 if d == 9 else 1.0)
            feed.tick(E, now, dt, up=up, active=True)
            run_engine(eng, st, now, training=True, dt=dt)
            assert not any(acc(st, now).values())
    assert not events(st)
    m = st.get_model(S, E, MODEL)
    assert m["fit"]["src"][0] == BG.SRC_OWN_IMMATURE          # learned (>= 7 d, no peers)
    row = budget_row(st, now)
    assert row["bytes_up.day"][0] > row["bytes_up.day"][1]   # would have alarmed live
    assert row["bytes_up.day"][2] < 1e-6


def test_cadence_switch_900_to_60_keeps_wall_clock_budgets():
    st, eng = make_store(), BudgetEngine()
    feed = Feed(st)
    rng = np.random.default_rng(4)
    rate = 1e4                                                # bytes per minute
    now = T_MID
    for i in range(8 * 96):                                   # 8 d at 900 s
        now = T_MID + (i + 1) * 900.0
        feed.tick(E, now, 900.0, up=rate * 15 * math.exp(0.05 * rng.normal()), active=True)
        run_engine(eng, st, now, training=True, dt=900.0)
    t_switch = now
    for i in range(3 * 60):                                   # 3 h at 60 s, live, trusted
        now = t_switch + (i + 1) * 60.0
        feed.tick(E, now, 60.0, up=rate * math.exp(0.05 * rng.normal()), active=True)
        set_trust(st, S, E, [now], 1.0)
        run_engine(eng, st, now, training=False, dt=60.0)
        assert not any(acc(st, now).values()), i
    row = budget_row(st, now)                                 # a full hour at 60 s
    assert row["bytes_up.1h"][0] == pytest.approx(rate * 60, rel=0.03)
    assert row["slots.1h"][0] == 4.0                          # 15-min slots, not ticks
    m = st.get_model(S, E, MODEL)
    assert m["rows"].frac[m["rows"].n - 1] == pytest.approx(60 / 3600)
    # the committed bin of the first 60-s hour is a full-hour rate like the 900-s ones
    a = int((t_switch + 8 * H) // H)                          # local hour after the switch
    sl = a % BG.NB
    st_ = m["state"]
    assert st_["hour"][sl] == a and st_["sc"][sl] == pytest.approx(1.0, abs=1e-6)
    assert st_["sx"][sl, 0] / st_["sw"][sl] == pytest.approx(rate * 60, rel=0.03)
    assert not events(st)


def test_rollback_holds_bins_and_release_restores_them():
    st, eng = make_store(), BudgetEngine()
    feed = Feed(st)
    dt = 3600.0
    now = T_MID
    for i in range(3 * 24):
        now = T_MID + (i + 1) * dt
        feed.tick(E, now, dt, up=1e6, active=True)
        run_engine(eng, st, now, training=True, dt=dt)
    tau = now - 30 * H                                        # older than the frontier
    day0 = local_day(now) - BG.HIST_DAYS
    v0, ok0 = hourly_values(st.get_model(S, E, MODEL)["state"], 0, day0)
    a_tau = int((tau + 8 * H) // H)
    idx = a_tau - day0 * 24                                   # tau's hour in the history
    assert ok0[idx + 2] and v0[idx + 2] == pytest.approx(1e6)
    put_model(st, S, E, "model.control", {"version": 0, "rollback_to": tau})
    now += dt
    feed.tick(E, now, dt, up=1e6, active=True)
    run_engine(eng, st, now, training=True, dt=dt)
    m = st.get_model(S, E, MODEL)
    assert m["gate"].applied.get("rollback_to") == tau
    assert np.all(np.isinf(m["fit"]["ts"]))                   # refit on the corrected history
    v1, ok1 = hourly_values(m["state"], 0, day0)
    assert ok1[idx - 2] and not ok1[idx + 2]                  # after tau: held, not learned
    assert len(m["gate"].held) >= 20
    put_model(st, S, E, "model.control", {"version": 0, "rollback_to": tau,
                                          "release": [tau, now]})
    now += dt
    feed.tick(E, now, dt, up=1e6, active=True)
    run_engine(eng, st, now, training=True, dt=dt)
    m = st.get_model(S, E, MODEL)
    v2, ok2 = hourly_values(m["state"], 0, day0)
    assert ok2[idx + 2] and v2[idx + 2] == pytest.approx(1e6)  # released into its own bin
    assert not m["gate"].held


def test_link_seed_merges_predecessor_history_and_destinations():
    st, eng = make_store(), BudgetEngine()
    feed = Feed(st)
    dt = 3600.0
    a, b = "10.3.0.1", "10.3.0.2"
    now = T_MID
    for i in range(3 * 24):
        now = T_MID + (i + 1) * dt
        feed.tick(a, now, dt, up=2e6, stream=[(now - 10.0, 2e6, 777)], active=True)
        run_engine(eng, st, now, training=True, dt=dt)
    now += dt
    feed.tick(b, now, dt, up=1e3, active=True)
    put_model(st, S, "__system__", "model.link",
              {"version": 1, "links": [{"from": a, "to": b, "ts": now}], "actors": {}})
    run_engine(eng, st, now, training=True, dt=dt)
    mb = st.get_model(S, b, MODEL)
    day0 = local_day(now) - BG.HIST_DAYS
    va, oka = hourly_values(st.get_model(S, a, MODEL)["state"], 0, day0)
    vb, okb = hourly_values(mb["state"], 0, day0)
    assert oka.sum() > 40 and np.array_equal(oka, okb)
    assert np.allclose(va[oka], vb[okb])                     # B := B_own + 0.5 A (rates)
    assert mb["gate"].link_version == 1
    assert 777 in mb["live"]["dest_first"]                   # A's destinations are not new


def test_actor_chain_parsing_variants():
    f = BG._actor_chains
    assert f(None, S) == []
    assert f({"actors": {"x": ["a", "b"]}}, S) == [["a", "b"]]
    assert f({"actors": [{"members": [f"{S}|a", "other|z", {"system": S, "entity": "b"}]}]},
             S) == [["a", "b"]]
    assert f({"actors": [{"chain": ["a"]}, "junk", {"entities": ["a", "a", "c"]}]},
             S) == [["a", "c"]]
    lk = {"links": [{"from": "a", "to": "b"}, {"from": "c", "to": "b", "state": "retracted"},
                    {"from": "d", "to": "b", "active": False}]}
    assert BG._link_sources(lk, "b") == ["a"]


def _synthetic_model(rng, today: int, level: float) -> dict:
    """A mature committed ring (28 past days) without running 28 days."""
    m = BG.new_model()
    st_ = m["state"]
    hours = np.arange((today - BG.HIST_DAYS) * 24, today * 24, dtype=np.int64)
    sl = hours % BG.NB
    ph = hours % 24
    lv = np.where((ph >= 8) & (ph < 19), 1.0, 0.2) * level
    st_["hour"][sl] = hours
    st_["sw"][sl] = 1.0
    st_["sc"][sl] = 1.0
    noise = np.exp(0.3 * rng.normal(size=(hours.size, BG.NQ)))
    st_["sx"][sl] = lv[:, None] * noise * np.array([1, 4, 1e-4, 4e-6, 0, 0, 1e-5, 5e-6, 5e-6])
    return m


def _reference_work(n_ent: int = 20) -> float:
    """A fixed workload of the same kind as B13's per-entity evaluation (many
    small-array numpy reductions - log, median, MAD, mean - and dict updates
    in a Python loop; budget._nanmed / _robust / phase_stats dominate B13's
    profile). Its CPU time is this process's speed on that kind of work at
    the moment it is measured: host speed and contention (shared caches,
    frequency) move it as they move B13's."""
    rng = np.random.default_rng(12345)
    X = rng.lognormal(size=(n_ent, 9, 27))
    acc: dict = {}
    tot = 0.0
    for e in range(n_ent):
        for q in range(9):
            for h in range(3):
                L = np.log(X[e, q, h:h + 24] + 0.01)
                med = np.median(L)
                mad = np.median(np.abs(L - med))
                z = float((L[-1] - med) / max(float(mad), 0.02)) + float(L.mean())
                acc[(e, q, h)] = acc.get((e, q, h), 0.0) + z
                tot += z
    return tot


# B13's per-tick CPU for 40 entities, in units of the reference workload's CPU
# measured interleaved in the same process (round 9). The bounds are the
# round-7 absolute ones (80 ms median, 400 ms max) expressed in reference units
# on the host where both were measured together (reference 19-21 ms, B13
# 43-50 ms, i.e. ~2.3 reference units; the 80 ms bound = 4 units): what the
# test checks is unchanged there, and it no longer depends on the host's speed
# or load (round 8: 77.9-83 ms alone on a slower host, 93-116 ms beside eval
# runs, against B13 unchanged since round 7; progressive.md 16.16.12 item 22).
REF_UNITS_MEDIAN = 4.0
REF_UNITS_MAX = 20.0


def test_perf_forty_entities_generous_bound():
    st, eng = make_store(), BudgetEngine()
    feed = Feed(st)
    rng = np.random.default_rng(9)
    dt = 900.0
    ents = [f"10.9.0.{i}" for i in range(40)]
    start = T_MID + 30 * DAY
    today = local_day(start + dt / 2)
    for e in ents:
        put_model(st, S, e, MODEL, _synthetic_model(rng, today, 2e6))
    times, refs = [], []
    for i in range(48):
        now = start + (i + 1) * dt
        for e in ents:
            up = 5e5 * math.exp(0.3 * rng.normal())
            feed.tick(e, now, dt, up=up, down=4 * up, req=10.0, writes=1.0,
                      stream=[(now - 5.0, up, 1234)], tokens={"GET x /a|2xx": 3.0},
                      objs={"x/a/{num}": {"n": 2, "ids": [str(i), str(i + 1)]}})
        set_trust(st, S, ents[0], [now], 1.0)
        t0 = cpu_s()                       # process CPU, not wall (helpers.cpu_s)
        _reference_work()
        refs.append(cpu_s() - t0)
        t0 = cpu_s()
        run_engine(eng, st, now, training=False, dt=dt)
        times.append(cpu_s() - t0)
    m = st.get_model(S, ents[-1], MODEL)
    assert (m["fit"]["src"] == BG.SRC_OWN).sum() >= 4         # fitted round-robin
    ref_ms = 1000.0 * float(np.median(refs))
    med_ms = 1000.0 * float(np.median(times[24:]))
    max_ms = 1000.0 * max(times)
    print(f"B13 perf: 40 entities, median {med_ms:.1f} ms CPU/tick ({med_ms / ref_ms:.2f} ref), "
          f"first-fit ticks max {max_ms:.1f} ms ({max_ms / ref_ms:.2f} ref); reference {ref_ms:.1f} ms")
    assert ref_ms > 0.0
    assert med_ms < REF_UNITS_MEDIAN * ref_ms, (med_ms, ref_ms)
    assert max_ms < REF_UNITS_MAX * ref_ms, (max_ms, ref_ms)
