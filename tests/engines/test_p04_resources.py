"""Resource bounds of the progressive lattice (docs/lib3/progressive.md §7,
P04 test (i), requirement S1): memory and learning time per event must not
grow with the number of IPs or of attributes. Each test scales one axis by
100x / 30x with the event count fixed and asserts the measured ratios."""
from __future__ import annotations

import time

import numpy as np

from ptree_sim import DAY, MON, Sim

from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import ptree as PT
from app.eval.pscale import deep_sizeof
from app.models.schema import SYSTEM_ENTITY

N_EVENTS = 4000
SEL = {"targets_sys": {0: ["net.bytes_up", "http.status", "body.kv.username"]},
       "split_cands": {0: [("http.route", 0), ("net.src", 1), ("net.src", 0), ("ctx.tod_min", 1)]},
       "roles": {}}


def _events(n_ips, n_attrs, seed=0):
    rng = np.random.default_rng(seed)
    ips = [f"10.{(i >> 16) & 255}.{(i >> 8) & 255}.{i & 255}" for i in range(1, n_ips + 1)]
    out = []
    for k in range(N_EVENTS):
        ts = MON + 8 * 3600 + k * (10 * 3600 / N_EVENTS)
        r = int(rng.integers(0, 6))
        a = {"http.route": f"POST s /r{r}", "net.bytes_up": float(rng.lognormal(6 + r * 0.3, 0.3)),
             "http.status": int(rng.choice([200, 302])), "body.kv.username": f"u{rng.integers(0, 40)}"}
        for j in range(n_attrs):
            a[f"meta.f{j:03d}"] = float(rng.integers(0, 5))
        out.append((ts, "s", ips[int(rng.integers(0, n_ips))], a))
    return out


def _run(n_ips, n_attrs, sel=SEL, p05=False):
    sim = Sim(dt=3600.0, sel=sel, p05=p05)
    sim.add(_events(n_ips, n_attrs))
    t_p04 = [0.0]
    orig = sim.p04.safe_run

    def timed(c, o=None):
        a = time.perf_counter()
        r = orig(c, o)
        t_p04[0] += time.perf_counter() - a
        return r
    sim.p04.safe_run = timed
    sim.run_until(MON + DAY)
    m = MP.get_ptree(sim.st, "s")
    tree = m.kinds[0]
    aux = sim.p04.aux(m)
    learned = N_EVENTS
    return {"tree_bytes": tree.nbytes(), "nodes": len(tree),
            "ptree_deep": deep_sizeof(m) - deep_sizeof(aux["burst"]) - deep_sizeof(aux["seen"]),
            "burst_entries": len(aux["burst"]), "seen_entries": len(aux["seen"]),
            "reg_deep": deep_sizeof(MP.get_registry(sim.st, "s")),
            "sel_deep": deep_sizeof(sim.st.get_model("s", SYSTEM_ENTITY, MP.ATTRSEL)),
            "us_per_event": t_p04[0] / learned * 1e6}


def test_i_memory_and_time_do_not_grow_with_ips():
    small = _run(100, 0)
    large = _run(10_000, 0)
    # the tree (nodes, summaries, split statistics) is bounded by its node budget,
    # never by the population: <= 1.5x at 100x the IPs, and far below tier M
    assert large["tree_bytes"] <= 1.5 * small["tree_bytes"] + 200_000, (small, large)
    assert large["ptree_deep"] <= 1.5 * small["ptree_deep"] + 500_000, (small, large)
    assert large["nodes"] <= PT.TIERS["M"]
    assert large["ptree_deep"] < 40 * 1024 * 1024
    # per-source state: burst runs of the last tick + tau_burst only (not the
    # population), and last activity of the heavy sources of confident nodes
    assert large["burst_entries"] <= 2 * N_EVENTS / 10 + 50, large
    assert large["seen_entries"] <= 8 * large["nodes"], large
    assert large["us_per_event"] <= 2.0 * small["us_per_event"] + 50.0, (small, large)


def test_i_memory_and_time_do_not_grow_with_attributes():
    small = _run(300, 10, sel=None, p05=True)
    large = _run(300, 300, sel=None, p05=True)
    # P04 touches m_t targets and C candidates per event whatever the attribute count
    assert large["tree_bytes"] <= 1.5 * small["tree_bytes"] + 200_000, (small, large)
    assert large["us_per_event"] <= 2.0 * small["us_per_event"] + 50.0, (small, large)
    # registry and selection are bounded by A_max, R_p and A_probe (not per event)
    assert large["reg_deep"] < 16 * 1024 * 1024 and large["sel_deep"] < 16 * 1024 * 1024
