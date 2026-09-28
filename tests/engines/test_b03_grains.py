"""spec v2.1 B03 / m_baseline grain maths (docs/lib3/cadence.md §6).

- the KAPPA_T pseudo-rows reproduce the transferred predictive T exactly
  (count mean, moment overdispersion r plus the Gamma shape term of kappa
  rows, ratio mean and concentration, t location and scale);
- the H -> Q transfer of a Pred: NB r / v, BB (1 + c) / v - 1 (>= C_T_MIN),
  t scale sqrt(v), means unchanged, 'none' features untouched;
- a canonical B03 run on H and Q decision rows commits decision rows only,
  keeps n_eff in H-row units x M, and the Q predictive's native share
  pi_nat = W / (W + KAPPA_T) grows with the native Q rows ('none' features
  unscored until KAPPA_T native rows).
"""
from __future__ import annotations

import numpy as np
import pytest

from helpers import T0, make_store, run_engine

from app.core.engine import default_config
from app.engines.behavior.baseline import BaselineEngine
from app.engines.behavior.lib import features as F
from app.engines.behavior.lib import grains as GR
from app.engines.behavior.lib import m_baseline as MB
from app.engines.behavior.lib import timebins as TB
from app.models.schema import DerivedMetric, MetricKind

NF = F.FEATURE_DIM
IDX = F.FEATURE_INDEX
CFG = default_config({"strict": True, "grain_mode": "canonical"})


def _pred(seed: int = 0) -> MB.Pred:
    rng = np.random.default_rng(seed)
    mu, r, p, c = (np.full(NF, np.nan) for _ in range(4))
    df, loc, sc = (np.full(NF, np.nan) for _ in range(3))
    mu[MB.CNT] = rng.uniform(0.5, 20.0, MB.CNT.size)
    r[MB.CNT] = rng.uniform(2.0, 50.0, MB.CNT.size)
    p[MB.RAT] = rng.uniform(0.05, 0.6, MB.RAT.size)
    c[MB.RAT] = rng.uniform(10.0, 200.0, MB.RAT.size)
    df[MB.NIG] = 30.0
    loc[MB.NIG] = rng.normal(3.0, 1.0, MB.NIG.size)
    sc[MB.NIG] = rng.uniform(0.2, 1.0, MB.NIG.size)
    mean = np.where(np.isfinite(mu), mu, np.where(np.isfinite(p), p, loc))
    return MB.Pred(mu, r, p, c, df, loc, sc, mean)


def test_pseudo_stats_reproduce_the_transferred_predictive():
    pred = _pred()
    k = GR.KAPPA_T
    par = MB._params(MB.pseudo_stats(pred, k), np.zeros((1, NF)))
    np.testing.assert_allclose(par.mu[0], pred.mu[MB.CNT], rtol=1e-12)
    # moment overdispersion r, plus the Gamma posterior shape a = kappa x 15 mu
    exp_r = 1.0 / (1.0 / pred.r[MB.CNT] + 1.0 / (k * 15.0 * pred.mu[MB.CNT]))
    np.testing.assert_allclose(par.r[0], exp_r, rtol=1e-10)
    np.testing.assert_allclose(par.p[0], pred.p[MB.RAT], rtol=1e-12)
    np.testing.assert_allclose(par.c[0], pred.c[MB.RAT], rtol=1e-10)
    np.testing.assert_allclose(par.loc[0], pred.loc[MB.NIG], atol=1e-12)
    np.testing.assert_allclose(par.scale[0], pred.scale[MB.NIG], rtol=1e-10)


@pytest.mark.parametrize("v", [1.0, 2.0, 4.0])
def test_transfer_pred_moments(v):
    pred = _pred(1)
    vv = np.full(NF, v)
    t = MB.transfer_pred(pred, vv, np.zeros(NF))
    tr = np.array([F.TRANSFER[n] or "-" for n in F.FEATURE_NAMES_V2])
    ok = (tr != "none") & (tr != "-")
    cnt = MB.CNT[ok[MB.CNT]]
    rat = MB.RAT[ok[MB.RAT]]
    nig = MB.NIG[ok[MB.NIG]]
    np.testing.assert_allclose(t.mu[cnt], pred.mu[cnt])
    np.testing.assert_allclose(t.r[cnt], pred.r[cnt] / v, rtol=1e-12)
    np.testing.assert_allclose(t.p[rat], pred.p[rat])
    np.testing.assert_allclose(t.c[rat], np.maximum(GR.C_T_MIN, (1.0 + pred.c[rat]) / v - 1.0),
                               rtol=1e-12)
    np.testing.assert_allclose(t.scale[nig] / pred.scale[nig], np.sqrt(v), rtol=1e-6)
    none = np.flatnonzero(tr == "none")
    for arr_t, arr_p in ((t.r, pred.r), (t.c, pred.c), (t.scale, pred.scale)):
        np.testing.assert_array_equal(np.isnan(arr_t[none]), np.isnan(arr_p[none]))
    assert t.grain == "q"


# ------------------------------------------------------ canonical B03 run
S, E = "sys", "10.0.0.1"


def _rows(rng, g: str) -> np.ndarray:
    G = GR.GRAIN_S[g]
    x = np.full(NF, np.nan)
    for n in ("flows", "http_requests", "dns_queries", "tls_handshakes", "intensity"):
        x[IDX[n]] = float(rng.poisson(0.5 * G / 60.0))
    x[IDX["bytes_up"]] = float(rng.lognormal(10.0, 0.3)) * G / 900.0
    x[IDX["bytes_down"]] = float(rng.lognormal(11.0, 0.3)) * G / 900.0
    x[IDX["http_4xx_rate"]] = float(rng.binomial(int(x[IDX["http_requests"]]), 0.05)) / max(
        x[IDX["http_requests"]], 1.0)
    x[IDX["distinct_peers"]] = float(rng.poisson(4))
    return x


def _feed(store, t: float, dt: float, rng) -> None:
    store.register_entity(S, E)
    store.add_vec(S, E, "feature.active", t, np.array([1.0], np.float32), window_s=int(dt))
    tc = TB.tctx_from_config(t, CFG, dt)
    store.add_derived(DerivedMetric(name="feature.tctx", value=tc, ts=t, system=S, entity=E,
                                    window_s=int(dt), kind=MetricKind.CATEGORICAL))
    for g in GR.GRAINS:
        if GR.decision(t, dt, g, GR.CANONICAL):
            store.add_vec(S, E, f"feature.nat.{g}", t, _rows(rng, g).astype(np.float32),
                          window_s=int(dt))
            store.add_vec(S, E, f"feature.meta.{g}", t,
                          np.array([1.0, GR.GRAIN_S[g]], np.float32), window_s=int(dt))


def test_canonical_learners_commit_decision_rows_and_q_share_grows():
    store = make_store()
    rng = np.random.default_rng(3)
    eng = BaselineEngine()
    dt = 900.0
    t = T0 - (T0 % 3600.0)
    prov_hist = []
    for i in range(4 * 24 * 3):                   # 3 days at 900 s
        t += dt
        _feed(store, t, dt, rng)
        run_engine(eng, store, t, training=True, dt=dt, config=CFG)
        if i % 24 == 23:
            m = store.get_model(S, E, MB.MODEL)
            tc = GR.row_tctx(t, "q", dt, CFG)
            pq = MB.predictive_q(store, S, E, tc, model=m)["current"]
            prov_hist.append(float(pq.prov[IDX["flows"]]))
    m = store.get_model(S, E, MB.MODEL)
    assert m["grain_mode"] == GR.CANONICAL
    # the H learner committed H rows only (one per hour), the Q learner Q rows
    jh = [r.ts for r in m["gate"].journal]
    assert jh and all(float(x) % 3600.0 == 0.0 for x in jh)
    jq = [r.ts for r in m["q"]["gate"].journal]
    assert jq and all(float(x) % 900.0 == 0.0 for x in jq)
    assert len(jq) > len(jh)
    # pi_nat rises as native Q rows accumulate, within [0, 1)
    assert all(0.0 <= x < 1.0 for x in prov_hist)
    assert prov_hist[-1] > prov_hist[0]
