"""Round 4: the identity score is the self-typicality tail calibrated on B15's
HELD-OUT genuine distances of the same model version (lib/m_identity
typicality_p, identity_model.typicality_fits, attribution.score_p).

Why: the v2 score was -log10 of the self posterior; its null moved with every
refit (modality calibration) and every role relabel (the own class is a
candidate), so B24's rings filled under earlier versions did not match the
live null. Pack A: identity p < 1e-3 on ~21x the nominal share of clean
ticks in round 3, API clients (machine personas, tight within-entity
scatter) the main contributor; the pack-A seed-0 KS of the clean identity p
fell from 0.129 to 0.035 with this change.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from helpers import DT, make_store, put_model

from app.core.engine import Context
from app.engines.behavior import attribution as AT
from app.engines.behavior.identity_model import IdentityModelEngine, typicality_fits
from app.engines.behavior.lib import emit
from app.engines.behavior.lib import m_identity as MI

import test_b15_identity_model as T15
import test_b16_attribution as T16


# ------------------------------------------------------------ pure helpers
def test_scaled_chi2_fit_recovers_scale_and_shape():
    rng = np.random.default_rng(3)
    for c, nu in ((0.05, 6.0), (4.0, 20.0)):
        h = c * rng.chisquare(nu, 4000)
        c_hat, nu_hat = MI.fit_scaled_chi2(h)
        assert c_hat == pytest.approx(c, rel=0.25)
        assert nu_hat == pytest.approx(nu, rel=0.3)
    assert all(math.isnan(x) for x in MI.fit_scaled_chi2([1.0, 2.0]))


def test_shrinkage_pulls_small_samples_to_the_role():
    fits = {"m1": (0.05, 5.0, 300), "m2": (0.06, 5.0, 300), "m3": (5.0, 5.0, 3),
            "h1": (4.0, 10.0, 300), "h2": (4.4, 10.0, 300)}
    out = MI.shrink_typicality(fits, {"m1": "r1", "m2": "r1", "m3": "r1", "h1": "r2", "h2": "r2"})
    assert out["m1"][0] == pytest.approx(0.05, rel=0.1)       # well identified: its own
    assert out["m3"][0] < 0.2                                  # 3 windows: toward the role
    assert out["h1"][0] > 3.0


def test_typicality_p_is_floored_by_the_heldout_sample():
    h = sorted(np.linspace(1.0, 10.0, 99).tolist())
    model = {"typ": {"e": [1e-3, 4.0, h, 99]}}                 # a very tight parametric fit
    p = MI.typicality_p(model, "e", 5.0)
    assert p >= 0.5 * 55 / 100                                 # >= #{h >= 5} / (n + 1)
    assert math.isnan(MI.typicality_p(model, "other", 5.0))
    assert math.isnan(MI.typicality_p({}, "e", 5.0))


# ------------------------------------------------------ B15: calibration
class _Tight(T15.Persona):
    """A machine persona: a tenth of the pooled within-entity scatter."""

    def __init__(self, mu, u, scale):
        super().__init__(mu, u)
        self.scale = scale

    def row(self, rng):
        vec = self.mu + rng.normal(0.0, self.scale, 52)
        sk = T15._unit_blocks(self.u + rng.normal(0.0, 0.15 * self.scale, 80))
        tim = {"B": 0.2 + rng.normal(0, 0.05 * self.scale), "M": 0.1,
               "think_mu": self.think + rng.normal(0, 0.2 * self.scale), "think_sigma": 1.0}
        return vec, sk, tim


def _personas():
    rng = np.random.default_rng(5)
    out = {}
    for i in range(6):
        mu = np.zeros(52)
        mu[:5] = 2.0 * i
        scale = 0.1 if i < 3 else 1.0                          # 3 machines, 3 humans
        out[f"10.0.1.{i}"] = _Tight(mu, T15._unit_blocks(rng.normal(0, 1, 80)), scale)
    return out


@pytest.fixture(scope="module")
def fitted():
    store = make_store()
    eng = IdentityModelEngine()
    per = _personas()
    now = T15.feed(store, eng, per, 200)
    eng.refit(Context(store=store, now=now, window_s=DT, training=False,
                      config={"strict": True}))
    return store, per, MI.get(store, T15.S)


def test_fit_publishes_typicality_for_every_entity(fitted):
    _, per, model = fitted
    assert set(model["typ"]) == set(per)
    for e, (c, nu, h, n) in model["typ"].items():
        assert c > 0 and nu > 0 and n >= 9 and len(h) <= MI.TYP_KEEP


def test_fresh_windows_get_valid_pvalues_for_machines_and_humans(fitted):
    """Out-of-sample windows of every persona, tight or wide, get p-values that
    are close to U(0, 1): the calibration comes from the entity's own held-out
    distances, not from the pooled (WCCN) within-entity scatter."""
    store, per, model = fitted
    now = float(model["fitted_ts"])
    rng = np.random.default_rng(99)
    ps = {"machine": [], "human": []}
    for e, p in per.items():
        for w in range(100):
            rows = []
            for k in range(MI.K_WIN):                 # the next K ticks, as live rows
                vec, sk, tim = p.row(rng)
                ts = now + (w * MI.K_WIN + k + 1) * DT
                row = np.concatenate([vec, sk, [tim["B"], tim["M"], tim["think_mu"]],
                                      MI.clock_features(T15.make_tctx(ts))])
                rows.append(row)
            z = MI.transform(model, MI.window_vector(np.vstack(rows)))
            ps["machine" if p.scale < 0.5 else "human"].append(MI.self_typicality(model, z, e))
    for grp, v in ps.items():
        v = np.asarray(v)
        assert np.all(np.isfinite(v))
        # valid (blocked CV is conservative by construction: its models see a
        # third less data) and close to uniform, the far tail not inflated
        for a in (0.01, 0.05, 0.1):
            assert np.mean(v < a) <= 1.5 * a + 0.01, (grp, a)
        assert stats.kstest(v, "uniform").statistic < 0.25, grp


def test_typicality_fits_use_the_heldout_column_of_each_entity():
    rng = np.random.default_rng(1)
    lab = np.repeat(np.arange(3), 40)
    D2 = np.full((120, 3), np.nan)
    D2[np.arange(120), lab] = rng.chisquare(4, 120) * np.array([0.1, 1.0, 10.0])[lab]
    out = typicality_fits(D2, lab, ["a", "b", "c"], {"a": "r", "b": "r", "c": "r"})
    cs = [out[e][0] for e in "abc"]
    assert cs[0] < cs[1] < cs[2]


# ------------------------------------------------------------- B16: score
def _with_typ(w: T16.World) -> dict:
    """model.identity of the B16 world plus B15-style typicality fits from an
    independent sample of each persona's windows (held-out stand-in)."""
    model = w.fit()
    rng = np.random.default_rng(21)
    fits = {}
    samples = {}
    for e in T16.ENTS:
        d2 = []
        for i in range(80):
            rows = [w.tick_row(*[w.draw(w.p[e], rng)[j] for j in (0, 3)],
                               T16.T0 - 86400.0 + i * 3700.0 + k * DT) for k in range(AT.K)]
            z = MI.transform(model, MI.window_vector(np.vstack(rows)))
            d2.append(float(np.sum((z - np.asarray(model["means"][e])) ** 2)))
        c, nu = MI.fit_scaled_chi2(d2)
        fits[e] = (c, nu, len(d2))
        samples[e] = sorted(d2)
    shr = MI.shrink_typicality(fits, {e: T16.ROLE for e in T16.ENTS})
    model["typ"] = {e: [shr[e][0], shr[e][1], samples[e], len(samples[e])] for e in T16.ENTS}
    return model


def _pm(w: T16.World, e: str) -> float:
    row = emit.read_row(w.store, T16.S, e, emit.PM, w.now) or {}
    return float(row.get("identity", math.nan))


def test_score_is_the_self_typicality_and_ignores_the_own_class():
    """The v2 score (pi_self) moved when the own role class candidate changed
    (a relabel or a refit of the class); the typicality does not."""
    w = T16.World()
    w.setup()
    model = _with_typ(w)
    put_model(w.store, T16.S, "__system__", "model.identity", model, version=2)
    w.step({})
    pm_a = {e: _pm(w, e) for e in T16.ENTS}
    moved = dict(model)
    ck = f"class:{T16.ROLE}"
    moved["class_means"] = {ck: (np.asarray(model["class_means"][ck]) * 0.2).tolist()}
    moved["class_var"] = {ck: (np.asarray(model["class_var"][ck]) * 0.05).tolist()}
    put_model(w.store, T16.S, "__system__", "model.identity", moved, version=3)
    w.eng.run(Context(store=w.store, now=w.now, window_s=DT, training=False,
                      config={"strict": True}))                 # re-run of the same tick
    for e in T16.ENTS:
        assert _pm(w, e) == pytest.approx(pm_a[e], rel=1e-9)
        assert 0.0 < pm_a[e] <= 1.0


def test_clean_ticks_give_uniform_identity_pm():
    """pm.identity (B24's small-sample prior) is a valid p-value on clean
    windows; the v2 pm = pi_self sat at ~1 (KS ~ 1 against U(0, 1))."""
    w = T16.World()
    w.setup()
    put_model(w.store, T16.S, "__system__", "model.identity", _with_typ(w), version=2)
    pms = []
    for k in range(160):
        w.step({})
        if k % AT.K == 0:                                      # non-overlapping windows
            pms.extend(_pm(w, e) for e in T16.ENTS)
    pms = np.asarray(pms)
    assert np.all(np.isfinite(pms))
    assert stats.kstest(pms, "uniform").statistic < 0.2
    assert not w.events()


def test_impersonated_window_is_atypical_of_the_owner():
    w = T16.World()
    w.setup()
    put_model(w.store, T16.S, "__system__", "model.identity", _with_typ(w), version=2)
    for _ in range(AT.K):
        w.step({T16.B: w.p[T16.A]})
    assert _pm(w, T16.B) < 1e-3


def test_partial_window_after_a_long_gap_has_no_calibrated_p():
    """The held-out calibration is of full K-row windows; the first rows after
    a gap longer than the lookback (Monday 09:00 after a weekend) score NaN,
    not a spuriously atypical window (pack B seed 0: 13 of 15 human identity
    p < 1e-3 on clean ticks were such windows)."""
    w = T16.World()
    w.setup()
    put_model(w.store, T16.S, "__system__", "model.identity", _with_typ(w), version=2)
    for _ in range(int(AT.LOOKBACK_S // DT) + 4):             # B idle for > 1 day
        w.step({T16.B: None})
    got = []
    for _ in range(AT.K):
        w.step({})
        got.append(_pm(w, T16.B))
    assert all(math.isnan(x) for x in got[:AT.K - 1]), got
    assert math.isfinite(got[-1])
