"""B06 MultivariateEngine (docs/lib3/engines.md '## B06'): the spec unit tests
(a)-(c), with the reviewed correction for (a) (T2-significant case at
(+3, +3); (+2, +2) at rho = 0.95 is T2 = 4.1, chi2_2 p = 0.13), plus the
scoring maths checked against lib/robustcov.

Spec test mapping:
  (a) test_a_correlation_break_spe_and_rbc, test_a_magnitude_t2_not_spe
  (b) test_b_missing_dimension_t2_and_imputation
  (c) test_c_null_hotelling_prediction_ks
Rows are written straight into behavior.zi (B05's output ring); training mode
supplies trust = 1 through lib/gating's default.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from helpers import DT, T0, make_store, run_engine

from app.engines.behavior import multivariate as MV
from app.engines.behavior.lib import emit, m_density, robustcov
from app.engines.behavior.lib.features import FEATURE_DIM
from app.engines.behavior.multivariate import WH, ZI, MultivariateEngine

S, E = "erp", "10.0.0.1"


def corr_rows(rng: np.random.Generator, n: int, p: int = 5, rho: float = 0.95) -> np.ndarray:
    """n x 52 zi rows: features 0 and 1 correlated at rho, 2..p-1 independent,
    the rest missing (NaN, i.e. unmodelled)."""
    C = np.eye(p)
    C[0, 1] = C[1, 0] = rho
    Z = rng.standard_normal((n, p)) @ np.linalg.cholesky(C).T
    X = np.full((n, FEATURE_DIM), np.nan)
    X[:, :p] = Z
    return X


def vec(**kw: float) -> np.ndarray:
    """52-dim zi row, NaN except the given f<idx>=value entries."""
    x = np.full(FEATURE_DIM, np.nan)
    for k, v in kw.items():
        x[int(k[1:])] = v
    return x


def feed(eng: MultivariateEngine, store, rows, t0: float = T0, dt: float = DT,
         training: bool = True, e: str = E) -> list:
    ts = []
    for i, r in enumerate(rows):
        t = t0 + i * dt
        store.add_vec(S, e, ZI, t, np.asarray(r, dtype=np.float32), window_s=int(dt))
        store.register_entity(S, e)
        run_engine(eng, store, t, training=training, dt=dt)
        ts.append(t)
    return ts


@pytest.fixture(scope="module")
def trained():
    """One entity trained through the engine on 400 ticks of the rho = 0.95 pair
    plus 3 independent features."""
    rng = np.random.default_rng(7)
    store = make_store()
    eng = MultivariateEngine()
    ts = feed(eng, store, corr_rows(rng, 400))
    return store, eng, ts[-1] + DT


def score_tick(store, eng, t, x):
    store.add_vec(S, E, ZI, t, np.asarray(x, dtype=np.float32), window_s=int(DT))
    run_engine(eng, store, t, training=False, dt=DT)
    return emit.read_row(store, S, E, emit.PM, t), emit.read_row(store, S, E, emit.SCORE, t)


# ------------------------------------------------------------------ (a)
def test_a_correlation_break_spe_and_rbc(trained):
    store, eng, t = trained
    model = m_density.get(store, S, E)
    assert model is not None and model["box_src"] == "oos"       # out-of-sample Box
    assert model["n"] >= 300 and list(model["cols"]) == [0, 1, 2, 3, 4]
    x = vec(f0=2.0, f1=-2.0, f2=0.0, f3=0.0, f4=0.0)
    assert np.nanmax(np.abs(x)) <= 2.0
    pm, sc = score_tick(store, eng, t, x)
    assert pm["spe"] < 1e-4
    assert sc["spe"] == pytest.approx(-math.log10(pm["spe"]), rel=1e-5)
    # RBC ranks features 1 and 2 (idx 0, 1) first
    lc = store.profile(S, E).extra["last_contrib"]
    assert lc["ts"] == t
    assert {d["idx"] for d in lc["top"][:2]} == {0, 1}
    assert all(d["p"] < 1e-4 for d in lc["top"][:2]) and lc["top"][2]["p"] > 0.01
    axes = emit.read_dict(store, S, E, emit.AXES, t)
    assert axes["spe"] == ["volume"]
    rbc, pv = m_density.contributions_model(model, x)
    assert set(np.argsort(-np.nan_to_num(rbc, nan=-1.0))[:2].tolist()) == {0, 1}


def test_a_magnitude_t2_not_spe(trained):
    store, eng, t = trained
    pm, sc = score_tick(store, eng, t + 2 * DT, vec(f0=3.0, f1=3.0))
    assert pm["t2"] < 0.05
    assert pm["spe"] > 0.05                                      # SPE not significant
    wh = store.vec_at(S, E, WH, t + 2 * DT)
    assert wh is not None and float(wh[0]) > 1.5
    # the pure accessor agrees with what the engine wrote
    s = m_density.score(store, S, E, vec(f0=3.0, f1=3.0))
    assert s.q == 2 and s.p_t2 == pytest.approx(pm["t2"], rel=1e-5)


# ------------------------------------------------------------------ (b)
def test_b_missing_dimension_t2_and_imputation():
    rng = np.random.default_rng(11)
    store = make_store()
    eng = MultivariateEngine()
    ts = feed(eng, store, corr_rows(rng, 380, p=2))
    t = ts[-1] + DT
    x = vec(f0=2.0)                                             # (+2, NaN)
    pm, sc = score_tick(store, eng, t, x)
    assert math.isfinite(sc["t2"]) and math.isfinite(pm["t2"])
    # the completion sits on the fitted correlation: no spurious break
    assert pm["spe"] > 0.05
    s = m_density.score(store, S, E, x)
    assert s.q == 1 and math.isfinite(s.t2)
    model = m_density.get(store, S, E)
    z2 = s.z_completed[1] + model["mu"][1]
    assert z2 == pytest.approx(0.95 * 2.0, abs=0.15)
    # same completion as the reference helper, same T2 as the reference cache
    zc = x[:2] - model["mu"][:2]
    ref = robustcov.conditional_impute(zc, model["Sigma"][:2, :2], np.isfinite(zc))
    assert s.z_completed[:2] == pytest.approx(ref, abs=1e-10)
    t2_ref, q_ref = robustcov.CholCache(model["Sigma"][:2, :2]).t2(zc)
    assert (s.t2, s.q) == (pytest.approx(t2_ref, rel=1e-10), q_ref)
    # p from the cross-fitted F null (round 4) once the buffer holds one
    s_t2, d2 = model["null"]["t2"]
    assert s.p_t2 == pytest.approx(m_density.f_sf(t2_ref, s_t2, 1.0, d2))


# ------------------------------------------------------------------ (c)
def test_c_null_hotelling_prediction_ks():
    """2000 null draws, each scored under a fresh n = 200 fit (the Hotelling
    prediction distribution is marginal over the fit): KS D < 0.05."""
    rng = np.random.default_rng(3)
    p = 5
    A = rng.standard_normal((p, p))
    C = A @ A.T / p + np.eye(p)
    d = np.sqrt(np.diag(C))
    Lc = np.linalg.cholesky(C / np.outer(d, d))
    ps = []
    X = np.full((201, FEATURE_DIM), np.nan)
    for _ in range(2000):
        X[:, :p] = rng.standard_normal((201, p)) @ Lc.T
        m = MV.fit_model(X[:200], crossfit=False)
        assert m["n_pred"] == 200.0
        ps.append(m_density.score_model(m, X[200]).p_t2)
    ps = np.asarray(ps)
    assert np.isfinite(ps).all()
    assert stats.kstest(ps, "uniform").statistic < 0.05


# ------------------------------------------------------------ maths parity
def test_rbc_matches_reference_with_missing():
    rng = np.random.default_rng(5)
    m = MV.fit_model(corr_rows(rng, 336, p=6))
    x = vec(f0=1.5, f1=-2.5, f3=0.7, f5=-1.1)                   # f2, f4 missing
    rbc, pv = m_density.contributions_model(m, x)
    cols = m["cols"]
    z = x[cols] - m["mu"][cols]
    r_ref, p_ref = robustcov.rbc_contributions(z, np.linalg.inv(m["Sigma"][np.ix_(cols, cols)]))
    assert rbc[cols] == pytest.approx(r_ref, rel=1e-8, nan_ok=True)
    assert pv[cols] == pytest.approx(p_ref, rel=1e-8, nan_ok=True)
    assert np.isnan(rbc[[2, 4]]).all() and np.isnan(rbc[6:]).all()
    # SPE of the completed vector equals robustcov.spe on the imputed row
    s = m_density.score_model(m, x)
    zc = robustcov.conditional_impute(z, m["Sigma"][np.ix_(cols, cols)], np.isfinite(z))
    assert s.spe == pytest.approx(robustcov.spe(zc, m["U_k"][cols]), rel=1e-9)
    assert s.wh == pytest.approx(robustcov.wilson_hilferty(
        s.t2 * 336 * (336 - 4) / (335 * 337), 4), rel=1e-9)


def test_null_wh_is_standard_normal_and_spe_calibrated():
    """The engine-trained model on the null: WH ~ N(0, 1) and the out-of-sample
    Box SPE p is not anti-conservative at 5 %."""
    rng = np.random.default_rng(21)
    store = make_store()
    eng = MultivariateEngine()
    feed(eng, store, corr_rows(rng, 400, p=6))
    m = m_density.get(store, S, E)
    sc = [m_density.score_model(m, x) for x in corr_rows(rng, 1500, p=6)]
    wh = np.array([s.wh for s in sc])
    assert abs(wh.mean()) < 0.15 and 0.8 < wh.std() < 1.2
    p_spe = np.array([s.p_spe for s in sc])
    assert np.isfinite(p_spe).all()
    assert (p_spe < 0.05).mean() < 0.08


# ------------------------------------------- round 4: cross-fitted F null
def _hetero_rows(rng: np.random.Generator, n: int, L: np.ndarray, day: int = 24) -> np.ndarray:
    """Correlated rows whose scale drifts between 'days' (log-sd 0.35): the
    clean live zi of pack A (mean zi^2 moves by +-25 % between multi-day
    periods), which no single Gaussian describes."""
    p = L.shape[0]
    Z = rng.standard_normal((n, p)) @ L.T
    s = np.repeat(np.exp(0.35 * rng.standard_normal(n // day + 1)), day)[:n]
    X = np.full((n, FEATURE_DIM), np.nan)
    X[:, :p] = Z * s[:, None]
    return X


def test_crossfit_null_calibrates_a_heteroscedastic_null():
    """Held-out rows of a day-to-day scale mixture: the cross-fitted F null
    keeps T2 and SPE near nominal at 1 % and 0.1 %, where the Gaussian
    Hotelling / Box p-values are 20-300x anti-conservative (the round-4
    pack-A failure: T2 44x, SPE 11x nominal at 1e-3)."""
    rng = np.random.default_rng(11)
    p = 24
    A = rng.standard_normal((p, 4))
    C = A @ A.T + 0.3 * np.eye(p)
    d = np.sqrt(np.diag(C))
    L = np.linalg.cholesky(C / np.outer(d, d))
    X = _hetero_rows(rng, 336, L)
    w, ts = np.ones(336), np.arange(336) * 3600.0
    m = MV.fit_model(X, w, ts=ts, crossfit=True)
    nst = MV.cross_null(X, w, ts, None, 0.0, m["theta1"], m["k"])
    null = MV.null_of(m, nst)
    assert null is not None and null["spe"] is not None and null["t2"] is not None
    m_null = dict(m, null=null)
    F = _hetero_rows(rng, 6000, L)
    legacy = [m_density.score_model(m, x) for x in F]
    new = [m_density.score_model(m_null, x) for x in F]
    x2 = lambda v: float(np.mean(np.asarray(v) < 1e-2) / 1e-2)     # noqa: E731
    x3 = lambda v: float(np.mean(np.asarray(v) < 1e-3) / 1e-3)     # noqa: E731
    assert x2([s.p_t2 for s in legacy]) > 10.0                     # the failure mode
    for key in ("p_t2", "p_spe"):
        v = [getattr(s, key) for s in new]
        assert x2(v) < 2.0, key                  # conservative draws are allowed: 14
        assert x3(v) < 3.0, key                  # days of scale are a small sample
    # homoscedastic Gaussian rows: d2 large (no scale mixing) and both
    # p-values near nominal (held-out SPE with the full model's k and theta1)
    Z = rng.standard_normal((6336, p)) @ L.T
    G = np.full((6336, FEATURE_DIM), np.nan)
    G[:, :p] = Z
    mg = MV.fit_model(G[:336], crossfit=False)
    ng = MV.null_of(mg, MV.cross_null(G[:336], w, ts, None, 0.0, mg["theta1"], mg["k"]))
    assert ng["t2"][1] > 30.0
    sg = [m_density.score_model(dict(mg, null=ng), x) for x in G[336:]]
    for key in ("p_t2", "p_spe"):
        v = [getattr(s, key) for s in sg]
        assert 0.5 < x2(v) < 2.0, key


def test_engine_marks_provisional_until_the_null_exists():
    """Scores against a model without a cross-fitted null carry
    behavior.degraded 'provisional:mv_null'; with >= CROSSFIT_MIN buffered
    rows the model has a null and the flag is gone."""
    rng = np.random.default_rng(3)
    store = make_store()
    eng = MultivariateEngine()
    ts = feed(eng, store, corr_rows(rng, 40))
    t = ts[-1] + DT
    score_tick(store, eng, t, vec(f0=0.1, f1=0.1, f2=0.0, f3=0.0, f4=0.0))
    dg = emit.read_dict(store, S, E, emit.DEGRADED, t)
    assert dg.get("t2") == MV.PROV_CAUSE and dg.get("spe", MV.PROV_CAUSE) == MV.PROV_CAUSE
    ts = feed(eng, store, corr_rows(rng, 120), t0=t + DT)
    model = m_density.get(store, S, E)
    assert model["null"] is not None and model["_null_st"]["n_rows"] >= MV.CROSSFIT_MIN
    t = ts[-1] + DT
    score_tick(store, eng, t, vec(f0=0.1, f1=0.1, f2=0.0, f3=0.0, f4=0.0))
    assert not emit.read_dict(store, S, E, emit.DEGRADED, t)
    assert m_density.descriptor(model)["null"]["src"] == "crossfit"
