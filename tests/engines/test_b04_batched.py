"""B04 batched scoring == the per-row reference path (perf, docs/lib3/integration.md §9).

The engine scores all rows of a tick at once (likelihood.score_features_many
with the array NB kernel, m_baseline.predictive_set_many / predictive_q_many /
quantiles_many, the grouped bb_ppf). These tests hold every batched output
to the per-row reference implementation on random inputs, bit for bit.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.engines.behavior import likelihood as L
from app.engines.behavior.lib import bayes
from app.engines.behavior.lib import m_baseline as MB

NF = L.NF


def rand_pred(rng) -> MB.Pred:
    arr = {k: np.full(NF, np.nan) for k in ("mu", "r", "p", "c", "df", "loc", "scale")}
    arr["mu"][MB.CNT] = np.exp(rng.uniform(-6, 5, MB.CNT.size))
    arr["r"][MB.CNT] = np.exp(rng.uniform(-3, 12, MB.CNT.size))
    arr["r"][MB.CNT[rng.random(MB.CNT.size) < 0.1]] = 1e13          # Poisson switch
    arr["p"][MB.RAT] = rng.uniform(1e-4, 0.9999, MB.RAT.size)
    arr["c"][MB.RAT] = np.exp(rng.uniform(np.log(2.0), np.log(2e3), MB.RAT.size))
    arr["df"][MB.NIG] = rng.uniform(1.0, 60.0, MB.NIG.size)
    arr["loc"][MB.NIG] = rng.normal(0, 3, MB.NIG.size)
    arr["scale"][MB.NIG] = np.exp(rng.uniform(-3, 2, MB.NIG.size))
    mean = np.where(np.isfinite(arr["mu"]), arr["mu"],
                    np.where(np.isfinite(arr["p"]), arr["p"], arr["loc"]))
    return MB.Pred(mean=mean, **arr)


def rand_row(rng, pred: MB.Pred, dt: float) -> np.ndarray:
    x = np.full(NF, np.nan)
    m = pred.mu[MB.CNT] * dt / 60.0
    x[MB.CNT] = rng.poisson(np.minimum(m * np.exp(rng.normal(0, 1.5, m.size)), 1e6))
    x[MB.CNT[rng.random(MB.CNT.size) < 0.05]] = np.nan
    x[MB.RAT] = rng.uniform(0, 1, MB.RAT.size)
    x[MB.RAT[rng.random(MB.RAT.size) < 0.3]] = rng.choice([0.0, 1.0])
    x[MB.NIG] = np.abs(rng.normal(1.0, 2.0, MB.NIG.size)) * 10.0
    x[MB.NIG[rng.random(MB.NIG.size) < 0.1]] = np.nan
    return x


def test_score_features_many_is_bit_identical_to_the_per_row_path():
    rng = np.random.default_rng(7)
    m = 60
    cur = [rand_pred(rng) for _ in range(m)]
    ref = [rand_pred(rng) if rng.random() < 0.8 else None for _ in range(m)]
    cls = [rand_pred(rng) if rng.random() < 0.6 else None for _ in range(m)]
    dts = rng.choice([60.0, 900.0, 3600.0, 2700.0], m)
    X = np.stack([rand_row(rng, cur[i], dts[i]) for i in range(m)])
    keys = [("sys", f"10.0.0.{i}", 1.7e9 + 900.0 * i) + (("q",) if i % 3 == 0 else ())
            for i in range(m)]
    many = L.score_features_many(cur, ref, cls, X, dts, keys)
    for i in range(m):
        one = L.score_features(cur[i], ref[i], cls[i], X[i], float(dts[i]), keys[i])
        for name, a, b in zip(one._fields, one, many):
            assert np.array_equal(a, b[i], equal_nan=True), (i, name, a, b[i])


def test_family_midp_many_nb_is_bit_identical():
    rng = np.random.default_rng(3)
    preds = [rand_pred(rng) for _ in range(40)]
    X = np.stack([rand_row(rng, p, 900.0) for p in preds])
    num, den, wf = MB._observe_many(X, np.full(40, 900.0))
    u, p, eq = L.family_midp_many(L.stack_preds(preds), num, den, wf)
    for i, pr in enumerate(preds):
        u1, p1, _ = L.family_midp(pr, num[i], den[i], wf[i])
        c = MB.CNT
        assert np.array_equal(np.isnan(p1[c]), np.isnan(p[i, c]))
        ok = ~np.isnan(p1[c])
        assert np.array_equal(p1[c][ok], p[i, c][ok]) and np.array_equal(u1[c][ok], u[i, c][ok])


def test_batched_bb_ppf_is_exact():
    rng = np.random.default_rng(1)
    for _ in range(150):
        G = int(rng.integers(1, 14))
        n = np.where(rng.random(G) < 0.1, rng.integers(1025, 3000, G),
                     rng.integers(1, 400, G)).astype(float)
        p = rng.uniform(1e-4, 0.999, G)
        c = np.exp(rng.uniform(np.log(2), np.log(1e4), G))
        a, b = p * c, (1 - p) * c
        q = np.array([0.05, 0.5, 0.95, 0.0, 1.0, 1e-9, 1 - 1e-9])[:, None]
        new = bayes.bb_ppf(q, n[None, :], a[None, :], b[None, :])
        for qi, qq in enumerate(q[:, 0]):
            for g in range(G):
                if qq <= 0.0:
                    ref = 0.0
                elif qq >= 1.0:
                    ref = n[g]
                else:
                    ref = bayes._bb_ppf_group(np.array([qq]), n[g], a[g], b[g])[0]
                assert new[qi, g] == ref


def test_pit_uniforms_many_equals_seeded_uniform():
    rng = np.random.default_rng(2)
    need = rng.random((5, NF)) < 0.4
    keys = [("s", f"e{i}", 1.7e9 + i) for i in range(5)]
    many = L.pit_uniforms_many(keys, need)
    for i in range(5):
        assert np.array_equal(many[i], L.pit_uniforms(keys[i], need[i]))


def test_engine_batched_tick_equals_the_per_entity_path():
    """The engine's batched tick writes the same z / zr / pf rows and scores
    as scoring each entity through _entity on the same store."""
    from helpers import run_engine
    from test_b04_likelihood import ENTS, S, Trained, base_row, feed

    tr = Trained()
    t = tr.tick()
    for e in ENTS:
        feed(tr.store, t, base_row(tr.rng, req=40.0), e=e)
    run_engine(L.LikelihoodEngine(), tr.store, t)
    batched = {e: [tr.store.vec_at(S, e, n, t).copy() for n in (L.Z, L.ZR, L.PF)]
               for e in ENTS}
    ref_eng = L.LikelihoodEngine()
    lay = ref_eng._layout(L.m_density.groups(tr.store, S, split_by_feature_group=True))
    for e in ENTS:
        nat = tr.store.vec_at(S, e, L.NAT, t)
        ref_eng._entity(tr.store, S, e, t, 900.0, nat, L._tctx_at(tr.store, S, e, t), lay)
        for a, name in zip(batched[e], (L.Z, L.ZR, L.PF)):
            b = tr.store.vec_at(S, e, name, t)
            assert np.array_equal(a, b, equal_nan=True), (e, name)


def test_quantiles_many_equals_quantiles():
    rng = np.random.default_rng(9)
    preds = [rand_pred(rng) for _ in range(25)]
    dts = rng.choice([900.0, 3600.0], 25)
    qs = (0.05, 0.5, 0.95)
    many = MB.quantiles_many(preds, qs, dts)
    for i, p in enumerate(preds):
        one = MB.quantiles(p, qs, dt_s=float(dts[i]))
        assert np.array_equal(np.isnan(one), np.isnan(many[i]))
        ok = ~np.isnan(one)
        assert np.array_equal(one[ok], many[i][ok]), i


def _same_pred(a: MB.Pred, b: MB.Pred) -> bool:
    import dataclasses
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray):
            if not np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
                return False
        elif x != y and not (x != x and y != y):
            return False
    return True


def test_batched_predictives_are_bit_identical():
    from app.engines.behavior.lib import grains as GR
    from app.engines.behavior.lib import timebins as TB
    from test_b04_likelihood import CFG, ENTS, S, Trained

    tr = Trained()
    for t in (tr.tick(), tr.tick() + 3 * 3600.0, tr.tick() + 11 * 3600.0):
        tc = TB.tctx_from_config(t, CFG, 900.0)
        items = [(S, e, tc) for e in ENTS] + [(S, "10.9.9.9", tc)]     # + an unknown entity
        for one, many in ((MB.predictive_set, MB.predictive_set_many),
                          (MB.predictive_q, MB.predictive_q_many)):
            got = many(tr.store, items)
            for (s, e, c), d in zip(items, got):
                ref = one(tr.store, s, e, c)
                assert ref.keys() == d.keys()
                for k in ref:
                    assert _same_pred(ref[k], d[k]), (one.__name__, e, k)
