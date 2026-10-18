"""lib-3 bounded mode (docs/lib3/progressive.md §10, W-P6): the B-library
processes the active and earned IPs of P15's sets, so its per-tick cost does
not grow with the number of IDLE known IPs; full mode is unchanged.

The whole production registry (build.build_registry) + P15, 5 active IPs and
N idle known IPs that were seen once, 12 h of 1-h ticks; the cost is measured
on the ticks after the idle IPs left the (test: 6-h) linger window."""
from __future__ import annotations

import numpy as np
import pytest

from app.core.store import MetricStore
from app.engines.behavior.lib import pactive as PA
from app.engines.behavior.resource_governor import ResourceGovernorEngine
from app.models.schema import ORG, SYSTEM_ENTITY, Observation
from app.pipeline import build
from app.pipeline.orchestrator import Pipeline
from helpers import cpu_s

T0 = 1_790_000_000.0
ACTIVE = [f"192.168.1.{i}" for i in range(1, 6)]
LINGER = 6 * 3600.0                       # a short linger window keeps the test fast


def obs(ts, ip, r):
    return Observation(ts=ts, system="oa", entity=ip, peer="192.168.100.100", l3_proto="ip",
                       l4_proto="tcp", dst_port=8080, bytes_up=int(r.integers(300, 3000)),
                       bytes_down=int(r.integers(1000, 30000)), pkts_up=4, pkts_down=6,
                       app_proto="http", http_method="GET", http_host="oa",
                       http_path=f"/oa/page{int(r.integers(0, 5))}", http_status=200,
                       user_agent=f"ua-{ip}")


CONFIGS = ((20, "bounded"), (400, "bounded"), (20, "full"))
SNAPSHOTS = {}                     # config -> signature.rule_match snapshots of the measured ticks


def _pipeline(n_idle, mode, dt):
    reg = build.build_registry()
    i = next(k for k, e in enumerate(reg._engines) if e.name == "behavior.feature_vector")
    reg._engines.insert(i, ResourceGovernorEngine())
    st = MetricStore()
    cfg = {"strict": True, "tz": "Asia/Shanghai", "grain_mode": "tick",
           "lib3": {"resource_mode": mode, "linger_s": LINGER}}
    return st, reg, Pipeline(st, reg, window_s=int(dt), config=cfg)


def run_all(configs=CONFIGS, hours=12, dt=3600.0):
    """The configurations tick by tick, interleaved, so that a change of
    machine load hits all of them alike; each tick's cost is the process CPU
    time of run_tick (helpers.cpu_s), not wall time."""
    state = {k: (*_pipeline(*k, dt), np.random.default_rng(0)) for k in configs}
    times = {k: [] for k in configs}
    stats = {}
    for h in range(hours):
        t1 = T0 + (h + 1) * dt
        for k in configs:
            st, reg, pipe, r = state[k]
            n_idle = k[0]
            o = [obs(t1 - dt + r.uniform(0, dt), ip, r) for ip in ACTIVE for _ in range(3)]
            if h == 0:
                o += [obs(t1 - 10.0, f"10.9.{j // 256}.{j % 256}", r) for j in range(n_idle)]
            rm = next(e for e in reg._engines if e.name == "signature.rule_match")
            if h == hours - 5:
                rm.n_snapshots = 0
            t0 = cpu_s()
            stats[k] = pipe.run_tick(o, now=t1, dt=dt)
            el = cpu_s() - t0
            if h >= hours - 5:
                times[k].append(el)
            SNAPSHOTS[k] = rm.n_snapshots
    return {k: (state[k][0], float(np.median(times[k])), stats[k]) for k in configs}


@pytest.fixture(scope="module")
def runs():
    return run_all()


def test_bounded_cost_flat_in_idle_known_ips(runs):
    st_s, t_small, _ = runs[(20, "bounded")]
    st_b, t_big, _ = runs[(400, "bounded")]
    print("bounded tick s (20 idle, 400 idle):", round(t_small, 3), round(t_big, 3))
    assert len(st_b.entities("oa")) == 405                  # the idle IPs are still known
    rec = st_b.get_model(ORG, ORG, "model.budget")["systems"]["oa"]
    assert set(rec["active"]) == set(ACTIVE)                 # ... but not processed
    # 20x the idle known IPs: per-tick CPU within noise (log-log slope <= 0.1 -> x1.35)
    assert t_big <= 1.35 * t_small + 0.05, (t_small, t_big)
    # and by operation count: lib-4 snapshots only the IPs with records of
    # the tick (it took a store snapshot of every known IP before round 7)
    assert SNAPSHOTS[(400, "bounded")] == SNAPSHOTS[(20, "bounded")] == 5 * len(ACTIVE)


def test_full_mode_unchanged_processes_every_known_ip(runs):
    st_f, _, _ = runs[(20, "full")]
    assert PA.entities(st_f, "oa", T0, {}) == st_f.entities("oa")
    # full mode writes feature rows for every known entity, bounded mode only for the active
    idle = "10.9.0.3"
    t_last = T0 + 12 * 3600.0
    assert st_f.vec_at("oa", idle, "feature.vec", t_last) is not None
    st_b, _, _ = runs[(20, "bounded")]
    assert st_b.vec_at("oa", idle, "feature.vec", t_last) is None
    assert st_b.vec_at("oa", ACTIVE[0], "feature.vec", t_last) is not None


def test_bounded_mode_earned_records_and_no_unearned_per_ip_ppm(runs):
    st_b, _, _ = runs[(20, "bounded")]
    rec = st_b.get_model("oa", SYSTEM_ENTITY, PA.EARNED)
    assert rec is not None and set(rec["ips"]) <= set(ACTIVE) | {f"10.9.0.{k}" for k in range(20)}
    assert all(r["n"] > 0 for ip, r in rec["ips"].items() if ip in ACTIVE)
    earned = set(st_b.get_model(ORG, ORG, "model.budget")["systems"]["oa"]["earned"])
    for ip in ACTIVE:
        if ip not in earned:
            assert st_b.get_model("oa", ip, "model.seq") is None      # B10 kept no per-IP PPM
    st_f, _, _ = runs[(20, "full")]
    assert st_f.get_model("oa", SYSTEM_ENTITY, PA.EARNED) is None     # full mode: no records
