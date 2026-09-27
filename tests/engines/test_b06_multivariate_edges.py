"""B06 MultivariateEngine edge cases: empty store, silent / cold entities,
NaN and degraded inputs (contract M), training mode, trust weighting, the
900 -> 60 s cadence switch and a generous per-tick cost bound."""
from __future__ import annotations

import math
import time

import numpy as np

from helpers import DT, T0, make_store, run_engine, set_trust

from app.engines.behavior import multivariate as MV
from app.engines.behavior.lib import emit, m_density
from app.engines.behavior.lib.features import FEATURE_DIM
from app.engines.behavior.multivariate import ACTIVE, DENSITY, WH, ZI, MultivariateEngine

S, E = "erp", "10.0.0.1"


def null_rows(rng: np.random.Generator, n: int, p: int = 4) -> np.ndarray:
    C = np.full((p, p), 0.3) + 0.7 * np.eye(p)
    X = np.full((n, FEATURE_DIM), np.nan)
    X[:, :p] = rng.standard_normal((n, p)) @ np.linalg.cholesky(C).T
    return X


def feed(eng, store, rows, t0=T0, dt=DT, training=True, e=E, trust=None):
    ts = []
    for i, r in enumerate(rows):
        t = t0 + i * dt
        store.add_vec(S, e, ZI, t, np.asarray(r, dtype=np.float32), window_s=int(dt))
        store.register_entity(S, e)
        if trust is not None:
            set_trust(store, S, e, [t], trust)
        run_engine(eng, store, t, training=training, dt=dt)
        ts.append(t)
    return ts


def degraded(store, t, e=E):
    return emit.read_dict(store, S, e, emit.DEGRADED, t)


# ------------------------------------------------------------------ absence
def test_empty_store_runs_and_writes_nothing():
    store = make_store()
    assert run_engine(MultivariateEngine(), store, T0) == 0
    assert store.systems() == []


def test_silent_entity_writes_nothing():
    store = make_store()
    store.register_entity(S, E)
    assert run_engine(MultivariateEngine(), store, T0) == 0
    assert store.get_model(S, E, DENSITY) is None
    assert store.vec_at(S, E, emit.SCORE, T0) is None and store.vec_at(S, E, WH, T0) is None


def test_active_without_zi_is_degraded_not_normal():
    store = make_store()
    store.register_entity(S, E)
    store.add_vec(S, E, ACTIVE, T0, [1.0])
    run_engine(MultivariateEngine(), store, T0)
    assert degraded(store, T0) == {"t2": "stale:behavior.zi", "spe": "stale:behavior.zi"}
    assert emit.read_row(store, S, E, emit.SCORE, T0) == {}          # NaN, never p = 1
    assert math.isnan(float(store.vec_at(S, E, WH, T0)[0]))


def test_all_nan_zi_is_degraded_and_never_learned():
    store = make_store()
    eng = MultivariateEngine()
    ts = feed(eng, store, np.full((60, FEATURE_DIM), np.nan))
    assert degraded(store, ts[-1]) == {"t2": "nan:behavior.zi", "spe": "nan:behavior.zi"}
    assert store.get_model(S, E, DENSITY) is None


def test_cold_entity_is_not_scored():
    store = make_store()
    eng = MultivariateEngine()
    ts = feed(eng, store, null_rows(np.random.default_rng(0), 30))
    assert not store.get_model(S, E, DENSITY)["fitted"]
    assert m_density.get(store, S, E) is None
    assert emit.read_row(store, S, E, emit.SCORE, ts[-1]) == {}
    assert store.vec_at(S, E, WH, ts[-1]) is None


# ------------------------------------------------------------------ NaN / degraded
def test_partial_nan_row_is_scored_on_observed_dims():
    rng = np.random.default_rng(1)
    store = make_store()
    eng = MultivariateEngine()
    ts = feed(eng, store, null_rows(rng, 200))
    t = ts[-1] + DT
    x = np.full(FEATURE_DIM, np.nan)
    x[0], x[1] = 1.0, -0.5
    x[2] = np.inf                                                   # non-finite = missing
    feed(eng, store, [x], t0=t, training=False)
    pm = emit.read_row(store, S, E, emit.PM, t)
    assert 0.0 < pm["t2"] <= 1.0
    s = m_density.score(store, S, E, x)
    assert s.q == 2 and np.isfinite(s.z_completed[:4]).all()


def test_b05_failure_degrades_even_with_a_zi_row():
    rng = np.random.default_rng(2)
    store = make_store()
    eng = MultivariateEngine()
    ts = feed(eng, store, null_rows(rng, 120))
    t = ts[-1] + DT
    store.put_health(MV.B05_ENGINE, {"ok": False, "last_error_ts": t})
    feed(eng, store, null_rows(rng, 1), t0=t, training=False)
    cause = "producer_error:" + MV.B05_ENGINE
    assert degraded(store, t) == {"t2": cause, "spe": cause}
    assert emit.read_row(store, S, E, emit.SCORE, t) == {}


def test_training_and_live_emit_no_events():
    rng = np.random.default_rng(3)
    store = make_store()
    eng = MultivariateEngine()
    X = null_rows(rng, 150)
    X[120:, :4] += 6.0                                              # loud anomaly in training
    ts = feed(eng, store, X)
    feed(eng, store, X[120:], t0=ts[-1] + DT, training=False, trust=1.0)
    assert store.events() == []


# ------------------------------------------------------------------ gating
def test_live_rows_learn_only_with_trust():
    rng = np.random.default_rng(4)
    store = make_store()
    eng = MultivariateEngine()
    ts = feed(eng, store, null_rows(rng, 80), training=False)       # no trust ring: w = 0
    st = store.get_model(S, E, DENSITY)["_state"]
    assert st["ts"] == []
    feed(eng, store, null_rows(rng, 80), t0=ts[-1] + DT, training=False, trust=0.5)
    st = store.get_model(S, E, DENSITY)["_state"]
    assert len(st["ts"]) == 80 - 4 and set(st["w"]) == {0.5}         # D = 4 ticks at 900 s
    m = m_density.get(store, S, E)
    assert m is not None and m["n"] == 76.0                          # equal weights: n_eff = n


def test_update_is_order_independent_and_one_row_per_slot():
    rng = np.random.default_rng(5)
    X = null_rows(rng, 400).astype(np.float32)
    ts = T0 + 60.0 * np.arange(400)                                  # 60 s: 15 rows per slot
    a = MV._init()
    for i in range(400):
        a = MV._update(a, (ts[i], X[i]), 1.0)
    b = MV._init()
    for i in rng.permutation(400):
        b = MV._update(b, (ts[i], X[i]), 1.0)
    assert a["ts"] == b["ts"] and a["slot"] == b["slot"]
    assert len(set(a["slot"])) == len(a["slot"]) == len({int(t // MV.SLOT_S) for t in ts})
    # the kept row of each slot is its newest
    last = {int(t // MV.SLOT_S): t for t in ts}
    assert a["ts"] == [last[s] for s in a["slot"]]
    blob = MV._load(MV._dump(a))
    assert blob["ts"] == a["ts"]
    assert np.allclose(np.asarray(blob["X"]), np.asarray(a["X"]), atol=5e-3, equal_nan=True)


# ------------------------------------------------------------------ cadence
def test_cadence_switch_900_to_60():
    rng = np.random.default_rng(6)
    store = make_store()
    eng = MultivariateEngine()
    ts = feed(eng, store, null_rows(rng, 120))
    v0 = m_density.get(store, S, E)["version"]
    ts2 = feed(eng, store, null_rows(rng, 80), t0=ts[-1] + 60.0, dt=60.0)
    now = ts2[-1]
    model = store.get_model(S, E, DENSITY)
    st = model["_state"]
    assert len(set(st["slot"])) == len(st["slot"]) <= MV.CAP
    assert max(st["ts"]) <= now - 10 * 60.0 + 1e-6                  # D = 10 ticks at 60 s
    # refits follow wall clock (16 ticks = 16 min at 60 s), not every tick
    assert 3 <= model["version"] - v0 <= 8
    assert 0.0 < emit.read_row(store, S, E, emit.PM, now)["t2"] <= 1.0
    assert store.vec_at(S, E, ZI, now) is not None


# ------------------------------------------------------------------ perf
def test_perf_steady_state_tick():
    """20 entities x 52 dims with block and scattered NaN, buffers full: the
    mean tick (scoring + gating + amortised refits every 16 ticks) stays well
    inside a generous bound (measured ~1 ms per entity single-threaded)."""
    rng = np.random.default_rng(7)
    p, n_ent = FEATURE_DIM, 20
    A = rng.standard_normal((p, 4)) * 0.6
    C = A @ A.T + np.eye(p)
    d = np.sqrt(np.diag(C))
    Lc = np.linalg.cholesky(C / np.outer(d, d))

    def rows(n):
        X = rng.standard_normal((n, p)) @ Lc.T
        X[np.ix_(rng.random(n) < 0.3, range(16, 26))] = np.nan
        X[rng.random(X.shape) < 0.03] = np.nan
        return X.astype(np.float32)

    store = make_store()
    eng = MultivariateEngine()
    ents = [f"10.0.1.{i}" for i in range(n_ent)]
    for e in ents:
        store.add_vec(S, e, ZI, T0, rows(1)[0])
        store.register_entity(S, e)
    run_engine(eng, store, T0, training=True)
    for e in ents:                                                  # full buffers
        st, X = MV._init(), rows(MV.CAP)
        for i in range(MV.CAP):
            st = MV._update(st, (T0 - (MV.CAP - i) * DT, X[i]), 1.0)
        store.get_model(S, e, DENSITY)["_state"] = st
    times = []
    for k in range(1, 34):
        t = T0 + k * DT
        for e in ents:
            store.add_vec(S, e, ZI, t, rows(1)[0])
        a = time.perf_counter()
        run_engine(eng, store, t, training=True)
        times.append(time.perf_counter() - a)
    assert all(m_density.get(store, S, e) is not None for e in ents)
    steady = float(np.mean(times[1:]))                              # tick 1 refits everyone
    assert steady < 0.25, f"mean tick {steady * 1e3:.1f} ms for {n_ent} entities"


def test_reset_density_writes_undefined_wh_instead_of_going_silent():
    """Once an entity has been scored, a tick it cannot score (the density was
    reset, e.g. by a rollback) writes behavior.wh = NaN: the series stays on
    its cadence (eval robustness gate: stale wh on the L6 / L8 newcomers)."""
    rng = np.random.default_rng(2)
    store = make_store()
    eng = MultivariateEngine()
    ts = feed(eng, store, null_rows(rng, 200))
    t = ts[-1] + DT
    feed(eng, store, null_rows(rng, 1), t0=t, training=False)
    assert np.isfinite(float(store.vec_at(S, E, WH, t)[0]))
    store.get_model(S, E, DENSITY)["fitted"] = False           # reset
    t2 = t + DT
    feed(eng, store, null_rows(rng, 1), t0=t2, training=False)
    assert math.isnan(float(store.vec_at(S, E, WH, t2)[0]))
