"""Deterministic engine costs for P15 (budget, ladder) and P12 (arm costs):
lib/pcost, progressive.md §16.12. Two runs that do the same work must make the
same decisions whatever the machine's load (round 3: PG8 0.67 vs 0.83 on the
same seed, §16.11.8 item 6)."""
from __future__ import annotations

import copy

import numpy as np

from helpers import ctx, make_store
from ptree_sim import DAY, MON, daily

from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pcost as PC
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.resource_governor import ResourceGovernorEngine
from app.engines.behavior.system_profile import SystemProfileEngine, _engine_costs
from app.models.schema import ORG, SYSTEM_ENTITY

T0 = 1_790_000_000.0
HEALTHY = ("raw.event", "behavior.pattern_tree", "behavior.binding", "behavior.payload_grammar",
           "behavior.workflow", "behavior.fusion")


def _batch(st, s, t1, n, dt):
    b = EV.BatchBuilder(s, EV.KIND_TXN)
    for i in range(n):
        ip = f"10.0.{i % 7}.{i % 50}"
        b.add(t1 - dt + 1.0 + i * dt / (n + 1), ip, {"net.src": ip, "ev.ch": "http", "http.route": "GET x /a"}, 1.0)
    batch = b.build(t1 - dt, t1)
    batch.learn[:] = True
    st.add_batch(s, EV.EVT_BATCH, t1, batch)


def _drive(walls, cfg, units=40, ticks=12, dt=900.0, n_ev=(30, 60)):
    """P15 over `ticks` ticks; every engine reports the same counted work
    (`units`) each tick but the wall-clock durations given by `walls(k, eng)`."""
    st = make_store()
    eng = ResourceGovernorEngine()
    t = T0
    for k in range(ticks):
        t += dt
        _batch(st, "oa", t, n_ev[k % 2], dt)
        for e in HEALTHY:
            ts = t if e.startswith("raw.") else t - dt        # behaviour engines ran after P15 last tick
            st.put_health(e, {"ts": ts, "duration_ms": walls(k, e), "last_count": units})
        eng.safe_run(ctx(st, t, window_s=dt, config=cfg), None)
    return st, t


def test_identical_work_costs_the_same_whatever_the_wall_clock():
    cfg = {"progressive": {"enabled": True}, "tz": "Asia/Shanghai"}
    r = np.random.default_rng(0)
    noise = {(k, e): float(r.uniform(1, 400)) for k in range(12) for e in HEALTHY}
    st_a, t = _drive(lambda k, e: 5.0, cfg)
    st_b, _ = _drive(lambda k, e: noise[(k, e)], cfg)
    ua = st_a.get_model(ORG, ORG, MP.BUDGET)["usage"]
    ub = st_b.get_model(ORG, ORG, MP.BUDGET)["usage"]
    assert ua["cost_model"] == "counted"
    # the decisions' inputs: CPU shares (ladder, P12's shadow price) and P12's per-engine costs
    assert ua["pcore_cpu_share"] == ub["pcore_cpu_share"] > 0
    assert ua["lib3_cpu_share"] == ub["lib3_cpu_share"]
    ca, cb = _engine_costs(st_a, t), _engine_costs(st_b, t)
    assert ca == cb and ca["behavior.binding"] > 0
    # the measured time is still reported, and differs
    assert ua["pcore_wall_share"] != ub["pcore_wall_share"]
    # 'wall' mode decides on the measured time (operations' choice): it differs
    wcfg = {"progressive": {"enabled": True, "budget": {"cost_model": "wall"}}, "tz": "Asia/Shanghai"}
    st_c, _ = _drive(lambda k, e: 5.0, wcfg)
    st_d, _ = _drive(lambda k, e: noise[(k, e)], wcfg)
    assert _engine_costs(st_c, t) != _engine_costs(st_d, t)


def test_counted_cost_grows_with_the_work_and_drives_the_ladder():
    cfg = {"progressive": {"enabled": True}, "tz": "Asia/Shanghai"}
    st_lo, t = _drive(lambda k, e: 1.0, cfg, units=10)
    st_hi, _ = _drive(lambda k, e: 1.0, cfg, units=10_000)
    lo, hi = _engine_costs(st_lo, t), _engine_costs(st_hi, t)
    assert all(hi[e] > lo[e] for e in ("behavior.pattern_tree", "raw.event"))
    # more events at the same counted units cost more too (the per-event term)
    st_ev, _ = _drive(lambda k, e: 1.0, cfg, units=10, n_ev=(300, 600))
    assert sum(_engine_costs(st_ev, t).values()) * 10 > sum(lo.values())     # per event: ~flat
    tot = lambda st: st.get_model(ORG, ORG, MP.BUDGET)["usage"]["pcore_cpu_share"]  # noqa: E731
    assert tot(st_ev) > tot(st_lo)
    # heavy counted work over a tight budget engages the ladder; heavy wall time with no work does not
    tight = {"progressive": {"enabled": True, "budget": {"pcore_cpu_share": 1e-4}}, "tz": "Asia/Shanghai"}
    st_w, _ = _drive(lambda k, e: 1.0, tight, units=200_000, ticks=16)
    assert st_w.get_model(ORG, ORG, MP.BUDGET)["ladder"]["step"] >= 1
    st_n, _ = _drive(lambda k, e: 50_000.0, tight, units=0, ticks=16, n_ev=(1, 1))
    assert st_n.get_model(ORG, ORG, MP.BUDGET)["ladder"]["step"] == 0


def test_run_cost_is_the_price_of_the_count():
    a, b, d = PC.coef("behavior.pattern_tree")
    assert PC.run_ms("behavior.pattern_tree", 100, 1000) == a + 100 * b + 1000 * d
    assert PC.run_ms("behavior.pattern_tree", 100, 1000, 2.0) == 2 * (a + 100 * b + 1000 * d)
    assert PC.run_cost_ms("x.unknown", {"last_count": 10, "duration_ms": 99.0}, 0) == \
        PC.DEFAULT_COEF[0] + 10 * PC.DEFAULT_COEF[1]
    assert PC.run_cost_ms("x.unknown", {"last_count": 10, "duration_ms": 99.0}, 0, {"cost_model": "wall"}) == 99.0
    assert PC.run_ms("behavior.binding", None, float("nan")) == PC.coef("behavior.binding")[0]


def _oa_events(days):
    def fn(d, t0, r):
        out = []
        if (d % 7) >= 5:
            return out
        for i, ip in enumerate(("192.168.1.21", "192.168.1.23", "10.168.7.121")):
            t = t0 + 9 * 3600 + r.uniform(0, 1200)
            out.append((t, "oa", ip, {"http.route": "POST oa.corp /oa/login", "http.method": "POST",
                                      "http.host": "oa.corp", "body.kv.username": f"u{i}",
                                      "body.len": float(r.integers(1024, 2048)), "sess.key": f"s{d}{ip}",
                                      "client.stack": f"ua-{ip}", "net.dst": "oa.corp:8080"}))
            for k in range(5):
                t += r.uniform(20, 240)
                out.append((t, "oa", ip, {"http.route": f"GET oa.corp /oa/p{k}", "http.method": "GET",
                                          "http.host": "oa.corp", "sess.key": f"s{d}{ip}",
                                          "client.stack": f"ua-{ip}", "net.dst": "oa.corp:8080"}))
        return out
    return daily(fn, days, seed=3)


def _inject(st, ev, t1, dt):
    evs = sorted((e for e in ev if t1 - dt < e[0] <= t1), key=lambda x: x[0])
    if evs:
        bb = EV.BatchBuilder("oa", EV.KIND_TXN)
        for ts, _, ip, attrs in evs:
            bb.add(ts, ip, dict({"net.src": ip, "ev.ch": "http"}, **attrs), 1.0)
        batch = bb.build(t1 - dt, t1)
        batch.learn[:] = True
        st.add_batch("oa", EV.EVT_BATCH, t1, batch)


def _p12_run(fitter_ms, days=4):
    """P12 alone (no P15): the fitters' own records carry their wall-clock
    'ms' (different between the two runs) and the same counted work."""
    ev = _oa_events(days)
    st = make_store()
    eng = SystemProfileEngine()
    cfg = {"progressive": {"enabled": True}, "tz": "Asia/Shanghai"}
    dt = 3600.0
    t = MON
    for k in range(int(days * DAY / dt)):
        t += dt
        _inject(st, ev, t, dt)
        st.put_model("oa", SYSTEM_ENTITY, MP.PBIND, {"updated": t, "gain": {
            "bits_per_event": 0.0, "ms": fitter_ms(k), "nodes_fitted": 3}})
        eng.safe_run(ctx(st, t, window_s=dt, config=cfg), None)
    return st.get_model("oa", SYSTEM_ENTITY, MP.SYSPROF)


def test_p12_arm_costs_without_p15_do_not_read_the_fitters_stopwatch():
    r = np.random.default_rng(1)
    noise = [float(r.uniform(0.1, 50.0)) for _ in range(200)]
    a = _p12_run(lambda k: 2.0)
    b = _p12_run(lambda k: noise[k])
    ca = a["arms"]["P08"]["on"]["cost"]
    assert ca is not None and ca > 0
    assert ca == b["arms"]["P08"]["on"]["cost"]
    strip = lambda p: {k: copy.deepcopy(p.get(k)) for k in ("chosen", "arms", "reasons")}  # noqa: E731
    assert strip(a) == strip(b)
