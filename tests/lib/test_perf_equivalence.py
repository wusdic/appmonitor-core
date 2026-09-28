"""Perf rewrites == their reference implementations (docs/lib3/integration.md §9).

Each optimised lib routine is compared with the straightforward version it
replaced (kept here verbatim as the reference) on random inputs: exact for
discrete / structural outputs, bit-identical floats where the arithmetic is
the same sequence of operations.
"""
from __future__ import annotations

import heapq
import math
import random

import numpy as np

from app.engines.behavior.lib import m_seq
from app.engines.behavior.lib import ppm as P


# ---------------------------------------------------------------- references
def ref_top_ngrams(model, n=2, k=10):
    p = m_seq.ppm(model)
    if p is None or not p.counts or n < 1 or n > p.order + 1:
        return []
    f = 2.0 ** (-p.g)
    cand = []
    for ctx, d in p.counts.items():
        if len(ctx) != n - 1 or m_seq.BOS in ctx:
            continue
        for sym, c in d.items():
            cand.append((c * f, ctx + (sym,)))
    top = heapq.nlargest(k, cand, key=lambda x: x[0])
    return [{"ngram": list(g), "count": round(float(c), 3)} for c, g in top]


def _rand_ppm(rng: random.Random, n_sessions: int, vocab: int, t0: float = 1.7e9) -> P.PPMModel:
    m = P.PPMModel()
    syms = [f"t{i}" for i in range(vocab)] + [m_seq.BOS]
    for i in range(n_sessions):
        toks = [rng.choice(syms[:-1]) for _ in range(rng.randint(1, 12))]
        # repeated equal weights create tied counts (the tie order matters)
        w = rng.choice([1.0, 1.0, 0.5, 0.25, rng.random()])
        P.update(m, toks, w=w, ts=t0 + 600.0 * i + rng.choice([0.0, 3600.0 * 24 * rng.random()]),
                 history=(m_seq.BOS,) if rng.random() < 0.5 else ())
    return m


def test_top_ngrams_equals_the_reference():
    rng = random.Random(4)
    for trial in range(60):
        pm = _rand_ppm(rng, rng.randint(0, 80), rng.randint(1, 25))
        model = {"ppm": pm}
        for n in (1, 2, 3, 4, 5):
            for k in (1, 3, 8, 10, 50):
                assert m_seq.top_ngrams(model, n, k) == ref_top_ngrams(model, n, k), (trial, n, k)


def ref_merge(own, other, w=0.5):
    """ppm.merge before the perf rewrite (verbatim)."""
    w = float(w)
    if other is None or other is own or not (w > 0.0 and math.isfinite(w)) or not other.counts:
        return own
    P._ensure_index(other)
    P._ensure_index(own)
    H_o = other.half_life_s if (other.half_life_s > 0 and math.isfinite(other.half_life_s)) else math.inf
    t_o = other.t_ref + (other.g * H_o if math.isfinite(H_o) else 0.0)
    g_ts = P._advance(own, t_o)
    f = w * 2.0 ** (g_ts - other.g)
    if not f > 0.0:
        return own
    order = max(0, int(own.order))
    counts, tot = own.counts, own._tot
    thr = P._DEAD * 2.0 ** own.g
    for ctx, d in other.counts.items():
        if len(ctx) > order:
            continue
        dst = counts.get(ctx)
        add = 0.0
        for s, c in d.items():
            v = c * f
            if dst is not None and s in dst:
                dst[s] += v
            elif v >= thr:
                if dst is None:
                    dst = counts[ctx] = {}
                dst[s] = v
            else:
                continue
            add += v
        if dst is not None:
            tot[ctx] = tot.get(ctx, 0.0) + add
    obs = own.n_obs
    d0 = counts.get((), {})
    for s, n in other.n_obs.items():
        if s in d0:
            obs[s] = obs.get(s, 0) + int(n)
    for s in d0:
        if s not in obs:
            obs[s] = 1
    own._n1 = P._singleton_mass(d0, obs)
    own.n_tokens = tot.get((), 0.0) * 2.0 ** (-own.g)
    fs = w * 2.0 ** min(0.0, g_ts - own.g)
    ost = other.stats
    if len(ost) == 3 and all(math.isfinite(x) for x in ost):
        own.stats = [a + fs * b for a, b in zip(own.stats, ost)]
    return own


def _state(m: P.PPMModel):
    return (m.order, m.t_ref, m.g, m.n_tokens, list(m.stats),
            [(c, list(d.items())) for c, d in m.counts.items()],
            list(m._tot.items()), list(m.n_obs.items()), m._n1)


def test_merge_is_bit_identical_to_the_reference():
    rng = random.Random(8)
    for trial in range(40):
        members = [_rand_ppm(rng, rng.randint(0, 60), rng.randint(1, 30),
                             t0=1.7e9 + rng.random() * 86400 * 60) for _ in range(rng.randint(1, 6))]
        if rng.random() < 0.2:          # an ancient member whose entries fall below 1e-9
            members.append(_rand_ppm(rng, 5, 5, t0=1.7e9 - 86400 * 3000))
        for order, w in ((P.PPM_MAX_ORDER, 1.0), (0, 1.0), (2, 0.37), (P.PPM_MAX_ORDER, 1e-12)):
            a, b = P.PPMModel(order=order), P.PPMModel(order=order)
            for mb in members:
                P.merge(a, mb, w)
                ref_merge(b, mb, w)
            assert _state(a) == _state(b), (trial, order, w)
