"""B10 republishes a tier unchanged when none of its members' learned states
changed since the key's last build (perf, docs/lib3/integration.md §9). The
result must be exactly what a full rebuild gives: this runs the same traffic
through two engines, one of which always rebuilds, and compares every tier
model and every score."""
from __future__ import annotations

import math

import numpy as np

from helpers import DT, T0, make_store, run_engine
from test_b10_sequence import BASE, S, Sys, clerk_paths, session

from app.engines.behavior.lib import emit
from app.engines.behavior.lib import m_seq
from app.engines.behavior.lib.classkeys import SYSTEM_KEY
from app.engines.behavior.sequence import SequenceEngine

ENTS = ("10.0.0.1", "10.0.0.2", "10.0.0.3")


def _tier_dump(store):
    m = m_seq.get(store, S, SYSTEM_KEY)
    if m is None:
        return None
    return (m["ppm"].to_dict(), m["ppm_cat"].to_dict(), repr(m["gap"]), repr(m["dwell"]),
            m["session_gap"], m["entropy_rate"], m["members"], m["version"], m["ts"])


def _same(a, b) -> bool:
    return a == b or (isinstance(a, float) and isinstance(b, float) and a != a and b != b)


def test_reused_tiers_equal_full_rebuilds():
    stores = [make_store(), make_store()]
    syss = [Sys(st, paths=BASE) for st in stores]
    engs = [SequenceEngine(), SequenceEngine()]
    rng = np.random.default_rng(3)
    reused = 0
    prev_ppm = None
    for i in range(200):
        now = T0 + i * DT
        idle = 60 <= i < 120 or i >= 170                 # nights: no traffic at all
        plan = {e: (None if idle or (e == ENTS[2] and i % 3) else clerk_paths(rng))
                for e in ENTS}
        for st, sy, eng in zip(stores, syss, engs):
            for e in ENTS:
                if plan[e] is not None:
                    sy.write(e, now, session(sy, now - 800, plan[e], np.random.default_rng(i)))
            run_engine(eng, st, now, training=i < 100)
        engs[1]._tier_sig.clear()                        # the reference always rebuilds
        a, b = _tier_dump(stores[0]), _tier_dump(stores[1])
        assert a == b, i
        m = m_seq.get(stores[0], S, SYSTEM_KEY)
        if m is not None:
            if prev_ppm is not None and m["ppm"] is prev_ppm and a[7] > 1:
                reused += 1
            prev_ppm = m["ppm"]
        for e in ENTS:
            ra = emit.read_row(stores[0], S, e, emit.SCORE, now)
            rb = emit.read_row(stores[1], S, e, emit.SCORE, now)
            assert ra.keys() == rb.keys() and all(_same(ra[k], rb[k]) for k in ra), (i, e)
    assert reused > 0                                    # the reuse path was exercised
