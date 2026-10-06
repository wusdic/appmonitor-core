"""Round 5: P03 / P11 / P15 state must not depend on Python's per-process
str-hash salt (PYTHONHASHSEED) or on memory addresses.

docs/lib3/progressive.md §16.12.6: P11's cover lists followed set iteration
order; a 4-day pack O-scale run under two hash seeds (registry owner's
harness) also differed in P03's hourly intensity sketches (eviction ties
broken by id(key)), P11's pending moves and P15's per-system records. Each
case runs the same computation in fresh interpreters with different hash
seeds and requires identical output (each fails on the round-4 code)."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
SEEDS = ("0", "1", "2", "3", "77")


def _outputs(code: str) -> set:
    outs = set()
    for seed in SEEDS:
        env = dict(os.environ, PYTHONHASHSEED=seed, OMP_NUM_THREADS="1")
        env["PYTHONPATH"] = BACKEND
        r = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env,
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-2000:]
        outs.add(r.stdout)
    return outs


def test_prefix_covers_do_not_depend_on_hash_seed():
    """P11: a group's covering prefixes were chosen greedily over candidates
    in the order of a set of its members; equal-gain candidates were taken
    in hash order (pack O-scale: 'covers' differed between processes)."""
    code = """
    from app.engines.behavior.lib import plouvain as LV
    mem = ["10.%d.%d.%d" % (a, b, c) for a in range(1, 5) for b in (7, 9) for c in (11, 12, 13)]
    act = LV.ActiveIndex(mem + ["10.%d.200.1" % a for a in range(1, 5)])
    for cov in (0.5, 0.6, 0.75, 0.9):
        print(LV.prefix_covers(mem, act, cover=cov, partial=True, max_prefixes=3))
    p = {("k%d" % i): 0.1 * (i % 7) + 1e-9 * i for i in range(60)}
    q = {("k%d" % i): 0.13 * (i % 5) + 3e-9 * i for i in range(20, 90)}
    print(LV._wj(p, q).hex())
    """
    outs = _outputs(code)
    assert len(outs) == 1, outs


def test_hour_sketch_evicts_ties_in_push_order():
    """P03: HourSS broke ties of the minimum count by id(key) & 0xFFFF, a
    memory address, so two runs of the same seed evicted different pairs."""
    from app.engines.behavior import conformity as CF
    ss = CF.HourSS(k=8)
    keys = [("192.168.1.%d" % i, 0, i) for i in range(8)]
    for k in keys:
        ss.add(k, 1.0)
    evicted = []
    for j in range(8):
        before = set(ss.c)
        ss.add(("10.0.0.%d" % j, 0, 100 + j), 1.0)
        evicted.append(sorted(before - set(ss.c)))
    # every eviction takes the oldest pair among the minimum counts
    assert evicted[0] == [keys[0]] and evicted[1] == [keys[1]] and evicted[2] == [keys[2]]


def test_governor_linger_order_does_not_depend_on_hash_seed():
    """P15: the linger LRU of recently active addresses took a tick's sources
    in set order; its recency order (truncated to the last 4096 addresses for
    the bounded-mode release) differed between processes (pack O-scale, 4 days,
    two hash seeds: model.budget_state 'linger' key order)."""
    code = """
    from collections import OrderedDict
    from types import SimpleNamespace
    from app.engines.behavior import resource_governor as RG
    eng = RG.ResourceGovernorEngine()
    st = SimpleNamespace(linger={}, last_event={}, rate={})
    store = SimpleNamespace(batch_systems=lambda kind: [])
    srcs = {"oa": {"10.0.%d.%d" % (i // 200, i % 200) for i in range(600)}}
    eng._activity(store, st, srcs, 1.75e9)
    print(list(st.linger["oa"].keys())[:50])
    """
    outs = _outputs(code)
    assert len(outs) == 1, outs
