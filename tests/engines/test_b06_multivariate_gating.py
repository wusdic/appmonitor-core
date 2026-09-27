"""B06 MultivariateEngine learner governance (contract H via lib/gating):
quarantine holds rows and release commits them (older than the 6 h zi ring:
through the engine's per-slot archive), rollback_to removes rows committed
after tau and refits at once, frozen stops commits, rebase_from down-weights
the old regime, and link seeding merges the source's rows at half weight."""
from __future__ import annotations

import numpy as np

from helpers import DT, T0, make_store, run_engine, set_trust

from app.engines.behavior import multivariate as MV
from app.engines.behavior.lib import emit, m_density
from app.engines.behavior.lib.features import FEATURE_DIM
from app.engines.behavior.multivariate import DENSITY, ZI, MultivariateEngine

S, E = "erp", "10.0.0.1"


def pair_rows(rng: np.random.Generator, n: int, rho: float = 0.9, p: int = 3) -> np.ndarray:
    C = np.eye(p)
    C[0, 1] = C[1, 0] = rho
    X = np.full((n, FEATURE_DIM), np.nan)
    X[:, :p] = rng.standard_normal((n, p)) @ np.linalg.cholesky(C).T
    return X


def tick(eng, store, t, row, *, training=False, trust=1.0, quarantine=0.0, e=E):
    store.add_vec(S, e, ZI, t, np.asarray(row, dtype=np.float32), window_s=int(DT))
    store.register_entity(S, e)
    if not training:
        set_trust(store, S, e, [t], trust, quarantine=quarantine)
    run_engine(eng, store, t, training=training, dt=DT)


def train(eng, store, X, t0=T0, e=E):
    for i, r in enumerate(X):
        tick(eng, store, t0 + i * DT, r, training=True, e=e)
    return t0 + len(X) * DT


def state(store, e=E):
    return store.get_model(S, e, DENSITY)["_state"]


def break_row(a: float = 2.5) -> np.ndarray:
    x = np.full(FEATURE_DIM, np.nan)
    x[:3] = (a, -a, 0.0)
    return x


def test_quarantine_holds_then_release_commits_from_archive():
    rng = np.random.default_rng(1)
    store = make_store()
    eng = MultivariateEngine()
    t = train(eng, store, pair_rows(rng, 120))
    n0 = len(state(store)["ts"])
    tq = []
    for r in pair_rows(rng, 40):                      # 10 h quarantined
        tick(eng, store, t, r, quarantine=1.0)
        tq.append(t)
        t += DT
    st = state(store)
    assert len(st["ts"]) <= n0 + 1                    # nothing committed while held
    rows = store.get_model(S, E, DENSITY)["_gate"].held
    held = [r.ts for r in rows]
    assert len(held) >= 35
    # rows held with trust_prov > 0 (the last training rows became candidates on
    # live ticks without a trust ring: w = 0, released but never learned)
    old = [r.ts for r in rows if r.w_prov > 0 and store.vec_at(S, E, ZI, r.ts) is None]
    assert old                                        # beyond the 6 h zi ring
    v0 = m_density.get(store, S, E)["version"]
    store.put_model(S, E, "model.control", {"release": [held[0], tq[-1]]})
    tick(eng, store, t, pair_rows(rng, 1)[0])
    st = state(store)
    assert set(old) <= set(st["ts"])                  # released rows fetched from the archive
    # only this tick's candidate is held (quarantine at t - 1 was still 1)
    assert len(store.get_model(S, E, DENSITY)["_gate"].held) <= 1
    assert m_density.get(store, S, E)["version"] > v0  # control -> refit at once


def test_rollback_removes_rows_after_tau_and_refits():
    rng = np.random.default_rng(2)
    store = make_store()
    eng = MultivariateEngine()
    t = train(eng, store, pair_rows(rng, 200))
    p_clean = m_density.get(store, S, E) and m_density.score(store, S, E, break_row()).p_spe
    t_att = t
    for i in range(60):                               # a missed attack: committed with trust 1
        x = pair_rows(rng, 1)[0]
        x[1] = -x[0]
        tick(eng, store, t, x, quarantine=1.0 if i == 59 else 0.0)   # B28: quarantine first
        t += DT
    assert max(state(store)["ts"]) > t_att
    tau = t_att - DT
    store.put_model(S, E, "model.control", {"rollback_to": tau})
    v0 = m_density.get(store, S, E)["version"]
    tick(eng, store, t, pair_rows(rng, 1)[0], quarantine=1.0)
    gate = store.get_model(S, E, DENSITY)["_gate"]
    assert gate.applied["_last_rollback"]["complete"] is True
    st = state(store)
    assert max(st["ts"]) <= tau
    assert all(r.ts > tau for r in gate.held) and len(gate.held) >= 50
    m = m_density.get(store, S, E)
    assert m["version"] > v0 and all(ts <= tau for ts in m["_oos_ts"])
    # the correlation break is a break again (as on the clean model)
    assert m_density.score_model(m, break_row()).p_spe < max(1e-3, 10 * p_clean)


def test_frozen_stops_commits():
    rng = np.random.default_rng(3)
    store = make_store()
    eng = MultivariateEngine()
    t = train(eng, store, pair_rows(rng, 100))
    store.put_model(S, E, "model.control", {"frozen": True})
    n0 = len(state(store)["ts"])
    for r in pair_rows(rng, 12):
        tick(eng, store, t, r)
        t += DT
    assert len(state(store)["ts"]) == n0
    assert store.get_model(S, E, DENSITY)["_gate"].frozen
    # scoring continues on the frozen model
    assert 0.0 < emit.read_row(store, S, E, emit.PM, t - DT)["t2"] <= 1.0


def test_rebase_downweights_old_regime():
    rng = np.random.default_rng(4)
    store = make_store()
    eng = MultivariateEngine()
    t = train(eng, store, pair_rows(rng, 100))
    tau = t
    for r in pair_rows(rng, 12):
        tick(eng, store, t, r, quarantine=1.0)
        t += DT
    store.put_model(S, E, "model.control", {"rebase_from": tau, "version": 1})
    tick(eng, store, t, pair_rows(rng, 1)[0])
    st = state(store)
    w = dict(zip(st["ts"], st["w"]))
    assert all(v == MV.REBASE_OLD_W for ts, v in w.items() if ts < tau)
    assert any(ts >= tau and v == 1.0 for ts, v in w.items())
    assert store.get_model(S, E, DENSITY)["_gate"].version == 1


def test_link_seeding_merges_source_rows_at_half_weight():
    rng = np.random.default_rng(5)
    store = make_store()
    eng = MultivariateEngine()
    a, b = "10.0.0.7", "10.0.0.8"
    t = train(eng, store, pair_rows(rng, 80), e=a)
    for i, r in enumerate(pair_rows(rng, 8)):
        tick(eng, store, t + i * DT, r, training=True, e=b)
    t += 8 * DT
    own = set(state(store, b)["slot"])
    store.put_model(S, "__system__", "model.link",
                    {"version": 1, "links": [{"from": a, "to": b, "ts": t}]})
    tick(eng, store, t, pair_rows(rng, 1)[0], training=True, e=b)
    st = state(store, b)
    src = dict(zip(state(store, a)["slot"], state(store, a)["w"]))
    seeded = [(sl, w) for sl, w in zip(st["slot"], st["w"]) if sl not in own and sl in src]
    assert len(seeded) >= 70 and all(w == 0.5 * src[sl] for sl, w in seeded)
    assert m_density.get(store, S, b)["n"] > 50        # fitted at once on the seeded rows
