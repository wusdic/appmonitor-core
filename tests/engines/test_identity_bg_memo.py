"""m_identity.Background memoises the candidate-independent background side
of each modality LLR (perf, docs/lib3/integration.md §9): scoring many
candidates against one shared Background gives exactly the values of a fresh
Background per call."""
from __future__ import annotations

import numpy as np

from helpers import DT, put_model
from test_b16_attribution import A, B, C, ENTS, S, World

from app.engines.behavior.lib import m_identity as MI
from app.engines.behavior.lib import m_timing as MTI
from app.engines.behavior.lib import ppm as PPM
from app.engines.behavior.lib.classkeys import SYSTEM_KEY


def _same(a: float, b: float) -> bool:
    return (a == b) or (a != a and b != b)


def test_shared_background_equals_a_fresh_one_per_candidate():
    w = World().setup()
    st = w.store
    sys_ppm = PPM.PPMModel(order=0)
    rng = np.random.default_rng(4)
    for i, e in enumerate(ENTS):
        pm = PPM.PPMModel(order=3)
        toks = [f"id:{int(t)}|2xx" for t in rng.integers(10 * i, 10 * i + 6, 120)]
        PPM.update(pm, toks, ts=w.now)
        PPM.update(sys_ppm, toks, ts=w.now)
        put_model(st, S, e, "model.seq", {"kind": "entity", "ppm": pm, "version": 1})
        tm = MTI.new_state()
        tm[MTI.bin_index([2.0 + 10.0 * i])[0]] = 500.0
        tm[MTI.T] = w.now
        put_model(st, S, e, "model.timing", {"fmt": 1, "version": 1, "state": tm})
    put_model(st, S, SYSTEM_KEY, "model.seq", {"kind": "system", "ppm": sys_ppm})
    w.step({B: w.p[A]})
    for e in (A, B):
        md = MI.tick_modal_data(st, S, e, w.now)
        md.tokens = [f"id:{int(t)}|2xx" for t in rng.integers(0, 40, 30)]
        md.gaps = rng.uniform(1.0, 60.0, 20)
        bg = MI.Background(st, S, w.now)
        for cand in (A, B, C, "class:r1", A):                  # A twice: a memo hit
            got = MI.modality_logliks(st, S, cand, md, w.now, bg)
            ref = MI.modality_logliks(st, S, cand, md, w.now, MI.Background(st, S, w.now))
            assert got.keys() == ref.keys()
            assert all(_same(got[k], ref[k]) for k in ref), (e, cand, got, ref)
            assert got["seq"] == got["seq"]              # the seq memo is exercised
        other = MI.ModalData(tokens=md.tokens[:5], gaps=md.gaps[:3])   # another window
        for cand in (A, B):
            g = MI.modality_logliks(st, S, cand, other, w.now, bg)
            r = MI.modality_logliks(st, S, cand, other, w.now, MI.Background(st, S, w.now))
            assert all(_same(g[k], r[k]) for k in r)
