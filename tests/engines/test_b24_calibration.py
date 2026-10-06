"""B24 CalibrationEngine: docs/lib3/engines.md B24 unit tests (a)-(g).

The engine is driven in isolation. `Rig` plays the surrounding engines of a
tick: detectors write behavior.score / behavior.pm through lib/emit, B01
writes feature.tctx, B24 runs, then B28 writes trust / trust_prov /
quarantine for that tick (so B24 commits row t - D with trust(t - D), as in
the pipeline). Maturity comes from model.baseline n_eff unless a test says
otherwise. Statistical checks use fixed seeds.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import numpy as np
import pytest

from helpers import DT, T0, make_store, make_tctx, put_model, run_engine, set_trust

from app.engines.behavior.calibration import CalibrationEngine
from app.engines.behavior.lib import calib, emit, gating, m_calib
from app.models.schema import DerivedMetric, MetricKind

S, E = "sys", "10.0.0.1"
NAN = float("nan")


class Rig:
    """One simulated pipeline around B24 (see module docstring)."""

    def __init__(self, entities: Iterable[str] = (E,), dt: float = DT, mature: bool = True,
                 daypart: Optional[str] = None, t0: float = T0) -> None:
        self.store = make_store()
        self.eng = CalibrationEngine()
        self.dt = dt
        self.t = t0
        self.daypart = daypart
        self.keys = list(entities)
        for e in self.keys:
            self.store.register_entity(S, e)
            if mature:
                put_model(self.store, S, e, "model.baseline", {"n_eff": 100.0})

    def write_tctx(self, e: str, ts: float, dt: float) -> Dict:
        t = make_tctx(ts, dt=dt)
        t.pop("daypart_id", None)
        if self.daypart is not None:
            t["daypart"] = self.daypart
        self.store.add_derived(DerivedMetric(name="feature.tctx", value=t, ts=ts, system=S,
                                             entity=e, window_s=int(dt),
                                             kind=MetricKind.CATEGORICAL))
        return t

    def step(self, scores: Dict[str, Dict[str, float]], pm: Optional[Dict] = None,
             trust: float = 1.0, quarantine: float = 0.0, dt: Optional[float] = None,
             training: bool = False, govern: bool = True, tctx: bool = True) -> float:
        dt = self.dt if dt is None else dt
        ts = self.t
        for e, sc in scores.items():
            emit.write_scores(self.store, S, e, ts, sc, pm=(pm or {}).get(e))
            if tctx and not e.startswith("class:"):
                self.write_tctx(e, ts, dt)
        run_engine(self.eng, self.store, ts, training=training, dt=dt)
        if govern:
            for e in self.keys:
                set_trust(self.store, S, e, [ts], trust, quarantine=quarantine)
        self.t = ts + dt
        self.dt = dt
        return ts

    def p(self, e: str, d: str, ts: float) -> float:
        return float(emit.read_array(self.store, S, e, emit.P, ts)[emit.DETECTORS.index(d)])

    def model(self, e: str = E) -> Dict:
        return self.store.get_model(S, e, m_calib.MODEL)


def ks(ps) -> float:
    return calib.ks_uniform(np.asarray(ps, dtype=np.float64))


def health_ps(rig: "Rig", d: str) -> np.ndarray:
    """The p B24's health checks saw for detector d (trusted committed ticks):
    the fully randomised p (lib/calib "Lower atom": the issued p is 1 at a
    ring's lowest level, the monitors keep the exactly uniform one)."""
    hs = rig.store.get_model(S, "__system__", m_calib.MODEL)[m_calib.HEALTH]
    return np.asarray(list(hs["ks"][emit.DETECTORS.index(d)]), dtype=np.float64)


def assert_valid(ps, levels=(0.01, 0.05, 0.1, 0.25, 0.5)) -> None:
    """P(p <= a) <= a within 3 binomial sd at every level (valid / conservative)."""
    ps = np.asarray(ps, dtype=np.float64)
    for a in levels:
        assert np.mean(ps <= a) <= a + 3.0 * np.sqrt(a * (1 - a) / ps.size), a


# ---------------------------------------------------------------- (a)
def test_a_null_exp1_is_uniform():
    """(a) 2000 null Exp(1) scores: KS D of the issued p < 0.03."""
    rng = np.random.default_rng(11)
    rig = Rig()
    ps = []
    for x in rng.exponential(1.0, 2000):
        ts = rig.step({E: {"marg_int": float(x)}})
        ps.append(rig.p(E, "marg_int", ts))
    ps = np.asarray(ps)
    assert np.all(np.isfinite(ps)) and np.all((ps > 0) & (ps <= 1))
    assert_valid(ps)
    # issued p: 1 only at / below the ring's lowest level (empty ring included)
    assert np.mean(ps == 1.0) < 0.05
    # the health checks see the randomised p: exactly uniform
    hp = health_ps(rig, "marg_int")
    assert hp.size > 1500 and np.all((hp > 0) & (hp < 1))
    assert ks(hp) < 0.03
    # rings are Mondrian by daypart: 20 days at 900 s visit all four dayparts
    keys = set(m_calib.rings(rig.model()))
    assert {k.split("@")[1].split("|")[0] for k in keys} >= {"wd_day", "wd_night"}
    assert all(len(r) <= calib.RING_M for r in m_calib.rings(rig.model()).values())


# ---------------------------------------------------------------- (b)
def test_b_sparse_detector_randomised_vs_deterministic():
    """(b) 90 % zeros: randomised p KS D < 0.03; the deterministic conformal p
    (u = 1, i.e. (#{c >= s} + 1)/(n + 1)) on the same rings gives D > 0.5."""
    rng = np.random.default_rng(5)
    rig = Rig(daypart="wd_day")
    p_rand, p_det = [], []
    xs = np.where(rng.random(2000) < 0.9, 0.0, rng.exponential(1.0, 2000))
    for x in xs:
        ts = rig.step({E: {"silence": float(x)}})
        p = rig.p(E, "silence", ts)
        p_rand.append(p)
        model = rig.model()
        st = calib.stratum_key("wd_day", 900)
        # B29 replay: the snapshot reproduces the issued p bit for bit
        u = m_calib.uniform(S, E, "silence", ts)
        assert m_calib.issued(m_calib.p_from_snapshot(model, "silence", st, x, u)) == p
        p_det.append(m_calib.p_from_snapshot(model, "silence", st, x, 1.0))
    assert ks(p_det) > 0.5
    # issued: every zero at a ring holding zeros gets 1 (no coin flip), the
    # rest the randomised p; valid at every level
    p_rand = np.asarray(p_rand)
    mature = np.arange(xs.size) >= 100
    assert np.all(p_rand[(xs == 0.0) & mature] == 1.0)
    assert_valid(p_rand)
    # the health checks see the randomised p, uniform as before round 5
    hp = health_ps(rig, "silence")
    assert hp.size > 1500 and ks(hp) < 0.03 and hp.max() < 1.0


# ---------------------------------------------------------------- (c)
def _body_tail_quantile(q: np.ndarray, xi: float = 0.2, sigma: float = 1.0) -> np.ndarray:
    """F^-1 of: 90 % Uniform(0, 1) body, 10 % GPD(xi, sigma) tail above u0 = 1."""
    q = np.asarray(q, dtype=np.float64)
    body = q / 0.9
    tq = (q - 0.9) / 0.1
    tail = 1.0 + sigma / xi * ((1.0 - tq) ** (-xi) - 1.0)
    return np.where(q < 0.9, body, tail)


def test_c_gpd_tail_far_beyond_the_ring():
    """(c) A score 10 sigma into a xi = 0.2 GPD tail: p within 2x of the true
    tail and below 1/(M + 1). The ring holds the exact quantile set of the
    distribution (fed in a seeded order, two full cycles), so the check is on
    the tail mechanics rather than on one noisy sample."""
    rng = np.random.default_rng(3)
    grid = _body_tail_quantile((np.arange(calib.RING_M) + 0.5) / calib.RING_M)
    rig = Rig(daypart="wd_day")
    for _ in range(2):
        for x in rng.permutation(grid):
            rig.step({E: {"cusum": float(x)}})
    for _ in range(gating.commit_delay_ticks(DT) + 16):      # flush + at least one refit
        rig.step({E: {"cusum": float(rng.choice(grid))}})
    s = 1.0 + 10.0 * 1.0
    p_true = 0.1 * (1.0 + 0.2 * 10.0) ** (-1.0 / 0.2)
    ts = rig.step({E: {"cusum": s}})
    p = rig.p(E, "cusum", ts)
    ring = m_calib.ring(rig.model(), "cusum", calib.stratum_key("wd_day", 900))
    assert len(ring) == calib.RING_M and ring.gpd is not None and ring.gpd.xi > 0.05
    assert p < 1.0 / (calib.RING_M + 1)
    assert p_true / 2.0 <= p <= 2.0 * p_true, (p, p_true)


# ---------------------------------------------------------------- (d)
def test_d_night_scores_never_use_the_day_ring():
    """(d) Real local time drives the daypart; night scores sit 50 above day
    scores. Every ring entry's ts lies in its ring's daypart, and no night tick
    is judged against the day ring (which would give p ~ 1/(n+1) or less)."""
    rng = np.random.default_rng(8)
    rig = Rig()
    night_p = []
    for i in range(700):
        ts = rig.t
        dp = make_tctx(ts, dt=DT)["daypart"]
        x = rng.normal(50.0 if dp.endswith("night") else 0.0, 1.0)
        rig.step({E: {"t2": float(x)}})
        if dp.endswith("night") and i > 300:
            night_p.append(rig.p(E, "t2", ts))
    rs = m_calib.rings(rig.model())
    assert len(rs) >= 2
    for key, r in rs.items():
        d, st = calib.split_ring_key(key)
        dp = st.split("|")[0]
        assert all(make_tctx(t, dt=DT)["daypart"] == dp for t in r.ts)
        if dp.endswith("night"):
            assert r.scores.min() > 40.0
        else:
            assert r.scores.max() < 10.0
    night_p = np.asarray(night_p)
    assert night_p.size > 50 and night_p.min() > 1e-5     # the day ring would give ~1e-20
    assert ks(night_p) < 0.15


# ---------------------------------------------------------------- (e)
def test_e_cadence_switch_blends_with_the_transferred_ring_for_64_ticks():
    """(e) After 900 s -> 60 s the new stratum ring starts empty. Round 4: its
    small-sample prior is the cadence transfer - the same daypart's 900-s
    ring's p made conservative by the learned power v (m_calib.xfer_prior),
    p_900^(1/v) - logit-blended with weight n/(n + 64) until the 60-s ring
    holds 64 entries; the issued p is marked provisional:cc_transfer; pm is
    no longer the prior while a source ring exists."""
    rng = np.random.default_rng(21)
    rig = Rig(daypart="wd_day")
    for _ in range(300):
        rig.step({E: {"marg_int": float(rng.exponential())}}, pm={E: {"marg_int": 0.5}})
    st900, st60 = calib.stratum_key("wd_day", 900), calib.stratum_key("wd_day", 60)
    assert m_calib.ring_size(rig.model(), "marg_int", st900) == calib.RING_M
    blended = 0
    for k in range(120):
        x = float(rng.exponential())
        pm = float(np.float32(rng.uniform(0.01, 0.99)))       # as stored in behavior.pm
        ts = rig.step({E: {"marg_int": x}}, pm={E: {"marg_int": pm}}, dt=60.0)
        p = rig.p(E, "marg_int", ts)
        u = m_calib.uniform(S, E, "marg_int", ts)
        # commits precede scoring in a tick: the ring after the step is the one used
        model = rig.model()
        ring = m_calib.ring(model, "marg_int", st60)
        n = 0 if ring is None else len(ring)
        conf = calib.p_from_ring(ring if ring is not None else calib.Ring(), x, u)
        deg = emit.read_dict(rig.store, S, E, emit.DEGRADED, ts)
        if k < 64:
            assert n < 64
            src = m_calib.ring(model, "marg_int", st900)
            v = m_calib.xfer_v(model[m_calib.XFER].get(m_calib.xfer_key("marg_int", 900, 60)))
            assert 1.0 <= v <= m_calib.XFER_V_MAX
            prior = calib.p_from_ring(src, x, u) ** (1.0 / v)
            want = calib.blend_small_sample(conf, prior, n)
            above = 0 if ring is None else int(np.sum(ring.scores > calib._r32(x)))
            if above:                          # own-history floor of the blend
                want = max(want, above / (n + 1.0))
            assert p == m_calib.issued(want)
            assert deg.get("marg_int") == "provisional:cc_transfer"
            blended += 1
        if n >= 64:
            assert p == m_calib.issued(conf)        # native: no prior
            assert "marg_int" not in deg
    assert blended == 64
    assert m_calib.ring_size(rig.model(), "marg_int", st60) >= 64
    assert m_calib.ring_size(rig.model(), "marg_int", st900) == calib.RING_M
    # the 60-s scores share the 900-s null here: v has moved from 2 towards 1
    v = m_calib.xfer_v(rig.model()[m_calib.XFER][m_calib.xfer_key("marg_int", 900, 60)])
    assert v < m_calib.XFER_V0


def test_e2_cadence_transfer_is_conservative_when_the_new_cadence_is_heavier():
    """A 60-s null three times as heavy as the 900-s one (Exp(3) vs Exp(1):
    exactly the power model with v = 3): the learned v rises above the
    prior 2, and the provisional p stay in band (realised rate of p <= 0.01
    on the transfer ticks <= 2x) where the raw 900-s ring is ~ 20x
    anti-conservative."""
    rng = np.random.default_rng(5)
    rig = Rig(daypart="wd_day")
    for _ in range(300):
        rig.step({E: {"marg_int": float(rng.exponential())}})
    ps, raw = [], []
    st900 = calib.stratum_key("wd_day", 900)
    src = m_calib.ring(rig.model(), "marg_int", st900)
    for k in range(60):
        x = float(rng.exponential(3.0))
        ts = rig.step({E: {"marg_int": x}}, dt=60.0)
        ps.append(rig.p(E, "marg_int", ts))
        raw.append(calib.p_from_ring(src, x, 0.5))
    v = m_calib.xfer_v(rig.model()[m_calib.XFER][m_calib.xfer_key("marg_int", 900, 60)])
    assert v > m_calib.XFER_V0
    ps, raw = np.asarray(ps), np.asarray(raw)
    assert np.mean(raw <= 0.01) > 0.1
    assert np.mean(ps <= 0.01) <= 0.02 + 1e-9


# ---------------------------------------------------------------- (f)
def test_f_rollback_deletes_entries_after_onset():
    """(f) model.control.rollback_to deletes every ring entry after the onset;
    the journal rows after it are held and come back on release."""
    rng = np.random.default_rng(2)
    rig = Rig(daypart="wd_day")
    ts_list = [rig.step({E: {"marg_int": float(rng.exponential()), "novelty": 0.0}})
               for _ in range(100)]
    ring = m_calib.ring(rig.model(), "marg_int", calib.stratum_key("wd_day", 900))
    tau = ts_list[60]
    assert ring.ts.max() > tau
    # B28: quarantine at the tick it writes rollback_to (tick t-1 of the next run)
    set_trust(rig.store, S, E, [ts_list[-1]], 1.0, quarantine=1.0)
    put_model(rig.store, S, E, "model.control", {"version": 0, "rollback_to": tau})
    rig.step({E: {"marg_int": 1.0}}, quarantine=1.0)
    m = rig.model()
    for key, r in m_calib.rings(m).items():
        assert len(r) and r.ts.max() <= tau, key
    gate = m["gate"]
    held = [r.ts for r in gate.held]
    assert held and min(held) > tau and all(r.ts <= tau for r in gate.journal)
    assert gate.applied["_last_rollback"]["removed"] > 0
    # quarantined ticks keep holding; release [tau, now] commits them again
    for _ in range(5):
        rig.step({E: {"marg_int": 1.0}}, quarantine=1.0)
    n_held = len(rig.model()["gate"].held)
    now = rig.t
    put_model(rig.store, S, E, "model.control",
              {"version": 0, "rollback_to": tau, "release": [tau, now]})
    rig.step({E: {"marg_int": 1.0}})
    m = rig.model()
    ring = m_calib.ring(m, "marg_int", calib.stratum_key("wd_day", 900))
    assert ring.ts.max() > tau
    assert len(m["gate"].held) < n_held


# ---------------------------------------------------------------- (g)
def test_g_nan_in_gives_nan_out():
    """(g) A NaN (degraded) score gives NaN p, never 1; other columns are unaffected."""
    rng = np.random.default_rng(4)
    rig = Rig(daypart="wd_day")
    for _ in range(80):
        rig.step({E: {"marg_int": float(rng.exponential()), "t2": float(rng.exponential())}})
    ts = rig.step({E: {"marg_int": NAN, "t2": 0.5}}, pm={E: {"marg_int": 0.2}})
    row = emit.read_array(rig.store, S, E, emit.P, ts)
    assert math.isnan(row[emit.DETECTORS.index("marg_int")])
    assert 0.0 < row[emit.DETECTORS.index("t2")] < 1.0
    # untouched detectors stay NaN (unscored), never 1
    assert np.all(np.isnan(np.delete(row, emit.DETECTORS.index("t2"))))
    # a NaN score is never admitted into a ring
    for r in m_calib.rings(rig.model()).values():
        assert np.all(np.isfinite(r.scores))
    assert math.isnan(m_calib.p_value(calib.Ring(), NAN, 0.5, 0.3))
    assert math.isnan(m_calib.p_from_snapshot(
        rig.model(), "t2", calib.stratum_key("wd_day", 900), NAN, 0.5, pm=0.3))


def test_e3_canonical_60s_t_stream_thinned_and_transferred():
    """Round 4, canonical mode, 900 -> 60 s: a T-stream ring admits one row per
    900-s slot (a rule on ts), so after 900 minutes it holds 60 rows, not 256
    minutes of one daypart; until it has native support its p takes the
    transfer from the 900-s ring (provisional:cc_transfer), with v learned
    from every committed native row; a null shared by both cadences stays
    calibrated."""
    from helpers import run_engine, set_trust
    rig = Rig(daypart="wd_day")
    cfg = {"grain_mode": "canonical"}
    rng = np.random.default_rng(1)

    def step(x, dt, training=False):
        ts = rig.t
        emit.write_scores(rig.store, S, E, ts, {"timing": x})
        rig.write_tctx(E, ts, dt)
        run_engine(rig.eng, rig.store, ts, training=training, dt=dt, config=cfg)
        set_trust(rig.store, S, E, [ts], 1.0, quarantine=0.0)
        rig.t = ts + dt
        return ts

    rig.t = (int(rig.t) // 3600 + 1) * 3600.0
    for _ in range(300):
        step(float(rng.exponential()), 900.0, training=True)
    ps, deg = [], []
    for _ in range(900):
        ts = step(float(rng.exponential()), 60.0)
        ps.append(rig.p(E, "timing", ts))
        deg.append(emit.read_dict(rig.store, S, E, emit.DEGRADED, ts).get("timing"))
    m = rig.model()
    n60 = m_calib.ring_size(m, "timing", calib.stratum_key("wd_day", 60))
    assert 55 <= n60 <= 62                                    # ~ 900 / 15 (minus the commit delay)
    k, n = m[m_calib.XFER][m_calib.xfer_key("timing", 900, 60)]
    assert n >= 850                                           # learnt from every committed row
    assert 1.0 <= m_calib.xfer_v([k, n]) < 1.5                 # the cadences share the null
    assert deg[-1] == "provisional:cc_transfer"               # still < 64 native rows
    ps = np.asarray(ps)
    assert np.mean(ps <= 0.01) <= 0.02 and np.mean(ps <= 0.05) <= 0.1
