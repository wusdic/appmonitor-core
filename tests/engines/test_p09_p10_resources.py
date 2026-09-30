"""Resource bounds of the temporal engines (requirement S1: "不能是遍历所有的
用户和服务器的行为"; docs/lib3/progressive.md §6.13, §6.14, §6.20, §7).

P10 touches each event once (session pass) and each learned row once more
(counting); its state is capped sketches plus LRUs sized by P15's budget.
P09 reads only fixed-size when summaries of the tree's nodes and refits dirty
nodes only. Both are measured while the number of IPs and of attributes per
event grows by two orders of magnitude."""
from __future__ import annotations

import pickle
import time

import numpy as np

from helpers import ctx, make_store
from temporal_sim import CFG, DAY, MON, OracleTree, batch

from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.time_window import TimeWindowEngine
from app.engines.behavior.workflow import WorkflowEngine
from app.models.schema import ORG

ROUTES = [f"GET oa /r{i}" for i in range(20)]
S_SESS = 4096                     # P15's session cap for the tree in this test


def p10_batches(n_ips: int, n_attrs: int, n_events: int = 6000, seed: int = 0):
    r = np.random.default_rng(seed)
    ips = [f"10.{i // 65536}.{(i // 256) % 256}.{i % 256}" for i in range(n_ips)]
    extra = {f"hdr.x{j}": f"v{j}" for j in range(n_attrs)}
    ts = np.sort(r.uniform(MON, MON + DAY, n_events))
    who = r.integers(0, n_ips, n_events)
    route = r.integers(0, len(ROUTES), n_events)
    out, dt = [], 900.0
    for k in range(int(DAY / dt)):
        a, b = MON + k * dt, MON + (k + 1) * dt
        sel = np.flatnonzero((ts > a) & (ts <= b))
        rows = [(float(ts[i]), ips[who[i]], dict(extra, **{"http.route": ROUTES[route[i]]})) for i in sel]
        out.append((b, batch("oa", rows, a, b) if rows else None))
    return out


def run_p10(n_ips: int, n_attrs: int):
    st = make_store()
    st.put_model(ORG, ORG, MP.BUDGET, {"trees": {"oa": {"s_sess": S_SESS}}})
    batches = p10_batches(n_ips, n_attrs)
    eng = WorkflowEngine()
    t_run = 0.0
    ticks = [(t1, b) for t1, b in batches] + [(MON + DAY + 900.0 * k, None) for k in range(1, 6)]
    for t1, b in ticks:                                     # (the last 5 drain the learning delay D)
        if b is not None:
            st.add_batch("oa", EV.EVT_BATCH, t1, b)
        t0 = time.perf_counter()
        eng.safe_run(ctx(st, t1, window_s=900.0, config=CFG), None)
        t_run += time.perf_counter() - t0
    m = MP.get_model(st, "oa", MP.PFLOW)
    rows = m["stats"]["rows"]
    return {"us_per_event": round(t_run * 1e6 / rows, 1), "sessions": len(m.state.sessions),
            "state_bytes": len(pickle.dumps(m.state, protocol=4)), "nbytes": m.state.nbytes(),
            "rows": rows, "counted": m["stats"]["counted"]}


def test_p10_cost_and_memory_do_not_grow_with_ips_or_attributes():
    small = run_p10(100, 0)
    mid = run_p10(1000, 0)
    big = run_p10(10000, 0)
    wide = run_p10(1000, 200)
    print("P10 scale", {"ips100": small, "ips1000": mid, "ips10000": big, "ips1000_attrs200": wide})
    assert small["counted"] == small["rows"] == big["counted"]
    # sessions are bounded by P15's cap, not by the population
    assert small["sessions"] <= 100 and big["sessions"] <= S_SESS
    # per-event CPU is flat in #IPs and #attributes (the engine reads only the route columns);
    # long sessions (100 IPs, ~60 events each) cost the most: up to E_MAX earlier actions per row
    assert big["us_per_event"] <= 1.5 * small["us_per_event"], (small, big)
    assert wide["us_per_event"] <= 1.5 * mid["us_per_event"] + 5.0, (mid, wide)
    # retained state: bounded by the caps (x10 IPs past the cap -> < x3 state; attributes -> flat)
    assert big["state_bytes"] <= 3 * mid["state_bytes"], (mid, big)
    assert abs(wide["state_bytes"] - mid["state_bytes"]) <= 0.05 * mid["state_bytes"], (mid, wide)
    assert big["nbytes"] <= 8_000_000
    # absolute budget (§7.3 / PG4: learning p95 <= 250 us per learned event; here every
    # row is learned and the session pass is included), with headroom for slow machines
    for r in (small, mid, big, wide):
        assert r["us_per_event"] <= 400.0, r


def p09_tree(n_ips: int, seed: int = 0, n_attrs: int = 0):
    """10 route nodes, 15 000 events over 10 days from n_ips IPs; with n_attrs,
    every leaf also models n_attrs target attributes (as P04 would), which P09
    must not read."""
    st = make_store()
    routes = [f"GET oa /r{i}" for i in range(10)]
    ot = OracleTree(st, "oa", routes)
    r = np.random.default_rng(seed)
    ips = [f"10.{i // 65536}.{(i // 256) % 256}.{i % 256}" for i in range(n_ips)]
    for d in range(10):
        for k in range(1500):
            j = int(r.integers(0, 10))
            ts = MON + d * DAY + (480 + 60 * j + r.uniform(0, 45)) * 60
            path = ot.learn(ts, ips[int(r.integers(0, n_ips))], routes[j])
            if n_attrs and k % 10 == 0:
                leaf = ot.tree.nodes[path[-1]]
                for a in range(n_attrs):
                    leaf.update_target(f"hdr.x{a}", f"v{int(r.integers(0, 4))}", ts, 1.0, 1.0)
    ot.apply_wants()
    return st, ot


def test_p09_cost_is_per_dirty_node_not_per_ip():
    """The fit of a node reads its fixed-size when summary: the same time
    and the same published size whether 100 or 10 000 IPs made its events;
    a run without new evidence fits nothing (periodic cost follows the
    evidence that arrived, §6.20)."""
    res = {}
    for n, n_attrs in ((100, 0), (10000, 0), (1000, 200)):
        st, ot = p09_tree(n, n_attrs=n_attrs)
        eng = TimeWindowEngine()
        t0 = time.perf_counter()
        eng.safe_run(ctx(st, MON + 10 * DAY, window_s=3600.0, config=CFG), None)
        dt_fit = time.perf_counter() - t0
        n_fit = eng.last_stats["oa"]["fitted"]
        size = len(repr(MP.get_model(st, "oa", MP.PWIN)))
        eng.safe_run(ctx(st, MON + 10 * DAY + 7 * 3600, window_s=3600.0, config=CFG), None)
        res[(n, n_attrs)] = (dt_fit / n_fit, size, n_fit, eng.last_stats["oa"].get("fitted", 0))
    print("P09 scale", res)
    a, b, w = res[(100, 0)], res[(10000, 0)], res[(1000, 200)]
    assert a[2] == b[2] == w[2] == 11                          # root + 10 routes (the empty `other` is skipped)
    assert b[0] <= 2.0 * a[0] + 0.002 and w[0] <= 2.0 * a[0] + 0.002
    assert abs(b[1] - a[1]) <= 0.05 * a[1] and abs(w[1] - a[1]) <= 0.05 * a[1]
    assert a[3] == 0 and b[3] == 0 and w[3] == 0               # nothing new -> nothing refitted
    # absolute budget (§7.3: one Bayesian-Blocks DP per dirty node and day type,
    # <= 513 minute cells): well under 100 ms per node
    assert max(a[0], b[0], w[0]) <= 0.1, res
