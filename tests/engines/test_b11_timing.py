"""B11 TimingEngine (docs/lib3/engines.md '## B11'): spec unit tests (a)-(c)
plus edge cases. Engines only talk to the store, so a tiny R2 + B01 stand-in
(`Feed`) writes act.stream / act.stream_frac / act.events exactly as
lib/m_template documents (cap -> contiguous chunk + stream_frac, zero-filled
act.events on silent ticks) and the feature.active clock the gated learner
steps on.

Spec test mapping:
  (a) test_a_human_sessions_burstiness_in_range (+ the pure log-normal note in
      test_a_pure_lognormal_renewal_matches_theory)
  (b) test_b_scraper_against_human_model
  (c) test_c_strict_period_train
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from helpers import DT, T0, make_store, run_engine, set_trust

from app.engines.behavior.lib import emit, evt
from app.engines.behavior.lib import m_template as MT
from app.engines.behavior.lib import m_timing as TM
from app.engines.behavior.timing import (DETECTOR, SERIES, TimingEngine, _compare,
                                         strict_period)
from app.models.schema import RawMetric

S = "erp"
E = "10.0.0.1"
LN8 = math.log(8.0)


# ------------------------------------------------------------------ fixtures
class Feed:
    """R2 + B01 stand-in for one system."""

    def __init__(self, store, system: str = S) -> None:
        self.store, self.s = store, system

    def tick(self, e: str, now: float, times, dt: float = DT, frac=None) -> None:
        t = np.sort(np.asarray(times, dtype=np.float64))
        f = 1.0
        if t.size > MT.STREAM_CAP:               # R2: one contiguous chunk of a long session
            f = MT.STREAM_CAP / t.size
            t = t[:MT.STREAM_CAP]
        if frac is not None:
            f = frac
        n = int(t.size)
        if n:
            arr = np.zeros(n, dtype=MT.STREAM_DTYPE)
            arr["ts"] = t
            arr.flags.writeable = False
            for name, v in (("act.stream", arr), ("act.stream_frac", f)):
                self.store.add_raw(RawMetric(name=name, value=v, ts=now, system=self.s, entity=e))
        ev = n / f if (f == f and f > 0) else float(n)
        self.store.add_raw(RawMetric(name="act.events", value=float(ev), ts=now, system=self.s,
                                     entity=e), touch=n > 0)
        self.store.add_vec(self.s, e, "feature.active", now,
                           np.array([1.0 if n else 0.0], np.float32), window_s=int(dt))


def human_times(rng, t0: float, t1: float, think=(LN8, 1.0), brk=300.0, sess=15.0):
    """Generator persona: sessions of Geometric(1/sess) events with
    LogNormal(ln 8 s, 1) think times, separated by Exp(brk) breaks."""
    out, t = [], t0
    while t < t1:
        for _ in range(int(rng.geometric(1.0 / sess))):
            out.append(t)
            t += float(np.exp(rng.normal(*think)))
        t += float(rng.exponential(brk))
    a = np.asarray(out)
    return a[a < t1]


def lognormal_times(rng, t0: float, t1: float, mu=LN8, sigma=1.0):
    n = int(3 * (t1 - t0) / math.exp(mu + sigma * sigma / 2)) + 64
    t = t0 + np.cumsum(np.exp(rng.normal(mu, sigma, n)))
    return t[t < t1]


def scraper_times(rng, t0: float, t1: float, gap=0.8, jitter=0.02):
    t = np.arange(t0, t1, gap)
    return np.sort(t + rng.normal(0.0, jitter, t.size))


def per_tick(times: np.ndarray, now: float, dt: float) -> np.ndarray:
    return times[(times > now - dt) & (times <= now)]


def run_stream(st, feed, eng, times, t_start, n_ticks, dt=DT, e=E, training=True):
    """Feed `times` tick by tick; returns the last tick ts."""
    now = t_start
    for i in range(n_ticks):
        now = t_start + i * dt
        feed.tick(e, now, per_tick(times, now, dt), dt)
        run_engine(eng, st, now, training=training, dt=dt)
    return now


def trained_human(seed=1, n_ticks=192, **kw):
    """Two days of a human at 900 s in training mode."""
    st, eng = make_store(), TimingEngine()
    feed = Feed(st)
    rng = np.random.default_rng(seed)
    times = human_times(rng, T0 - DT, T0 + (n_ticks + 200) * DT, **kw)
    now = run_stream(st, feed, eng, times, T0, n_ticks)
    return st, feed, eng, times, rng, now


def timing(st, now, e=E):
    for m in reversed(st.derived_series(S, e, SERIES)):
        if m.ts == now:
            return m.value
    return None


def score(st, now, e=E):
    return emit.read_row(st, S, e, emit.SCORE, now).get(DETECTOR, math.nan)


def pm(st, now, e=E):
    return emit.read_row(st, S, e, emit.PM, now).get(DETECTOR, math.nan)


def model(st, e=E):
    return st.get_model(S, e, TM.MODEL)


# ------------------------------------------------------------------ spec (a)
def test_a_human_sessions_burstiness_in_range():
    """Human sessions (think time LogNormal(ln 8 s, sigma = 1), breaks between
    sessions) give B in [0.2, 0.6]; the think-time fit recovers (ln 8, 1)
    when model.seq supplies the session gap."""
    st, eng = make_store(), TimingEngine()
    st.put_model(S, E, "model.seq", {"session_gap": 150.0})
    feed = Feed(st)
    rng = np.random.default_rng(3)
    times = human_times(rng, T0 - DT, T0 + 200 * DT)
    now = run_stream(st, feed, eng, times, T0, 192)
    rec = timing(st, now)
    assert rec is not None and set(rec) == {"B", "M", "think_mu", "think_sigma", "period",
                                            "period_p"}
    assert 0.2 <= rec["B"] <= 0.6
    assert abs(rec["M"]) < 0.3
    d = TM.descriptors(model(st), now)
    assert 0.2 <= d["B"] <= 0.6                               # committed model agrees
    assert abs(d["think_mu"] - LN8) < 0.15
    assert 0.85 <= d["think_sigma"] <= 1.1
    assert 5.0 < d["think_median_s"] < 11.0
    assert d["gap_p10_s"] < d["gap_p50_s"] < d["gap_p90_s"]
    assert math.isnan(rec["period"])                          # a human is not a strict train


def test_a_pure_lognormal_renewal_matches_theory():
    """A pure log-normal renewal (no sessions) with sigma = 1 has CV =
    sqrt(e - 1), B = 0.135: the spec's [0.2, 0.6] needs the session breaks of
    a human (test above). sigma = 1.3 lands inside the range."""
    for sigma, lo, hi in ((1.0, 0.08, 0.2), (1.3, 0.2, 0.6)):
        st, eng = make_store(), TimingEngine()
        feed = Feed(st)
        rng = np.random.default_rng(7)
        times = lognormal_times(rng, T0 - DT, T0 + 100 * DT, sigma=sigma)
        now = run_stream(st, feed, eng, times, T0, 96)
        rec = timing(st, now)
        cv = math.sqrt(math.exp(sigma * sigma) - 1.0)
        assert abs(TM.b_from_moments(1.0, cv * cv) - (cv - 1) / (cv + 1)) < 1e-12
        assert lo <= rec["B"] <= hi, (sigma, rec["B"])
        assert abs(rec["think_sigma"] - sigma) < 0.12


# ------------------------------------------------------------------ spec (b)
def test_b_scraper_against_human_model():
    """A constant 0.8-s scraper: B < -0.8 and timing p < 1e-3 against the
    human model (the model is not polluted: live, no trust -> weight 0)."""
    st, feed, eng, _, rng, now = trained_human()
    human = model(st)
    mass0 = TM.mass(human)
    assert TM.n_eff(human) > 1000
    h_scores = [score(st, now - k * DT) for k in range(20)]
    assert all(s == s for s in h_scores)
    sc = scraper_times(rng, now, now + 30 * DT)
    alarms, pms = [], []
    for i in range(1, 29):
        t = now + i * DT
        feed.tick(E, t, per_tick(sc, t, DT))
        run_engine(eng, st, t)
        pms.append(pm(st, t))
        alarms.append(emit.read_dict(st, S, E, emit.ACC_ALARM, t).get(DETECTOR))
        assert score(st, t) > max(h_scores)
        assert DETECTOR in emit.read_dict(st, S, E, emit.AXES, t)
    assert max(pms) < 1e-3
    assert all(a == 1 for a in alarms)
    rec = timing(st, t)                                      # window is pure scraper now
    assert rec["B"] < -0.8
    assert rec["think_sigma"] < 0.1 and abs(math.exp(rec["think_mu"]) - 0.8) < 0.05
    # live without trust: nothing learned (the mass only decays)
    assert TM.mass(model(st)) <= mass0 + 1e-9
    # the pure accessor agrees: scraper gaps are unlikely under the human model
    g_s, g_h = np.full(200, 0.8), np.diff(human_times(np.random.default_rng(9), 0, 20000))
    ll_s = TM.loglik(model(st), g_s) / g_s.size
    ll_h = TM.loglik(model(st), g_h) / g_h.size
    assert ll_s < ll_h - 1.0
    assert st.events() == []                                 # B11 emits no event kinds


def test_b_human_holdout_no_alarm():
    """The same human, live with trust: the recent window stays close to the
    model (no accumulator alarm, pm mostly well above the alarm level)."""
    st, feed, eng, times, _, now = trained_human()
    t_live = [now + i * DT for i in range(1, 49)]
    set_trust(st, S, E, t_live, 1.0)
    pms, alarms = [], []
    for t in t_live:
        feed.tick(E, t, per_tick(times, t, DT))
        run_engine(eng, st, t)
        pms.append(pm(st, t))
        alarms.append(emit.read_dict(st, S, E, emit.ACC_ALARM, t).get(DETECTOR, 0))
    assert sum(alarms) == 0
    assert np.nanmedian(pms) > 0.05


# ------------------------------------------------------------------ spec (c)
@pytest.mark.parametrize("mode,dt", [("iid", 60.0), ("renewal", 60.0), ("iid", 900.0)])
def test_c_strict_period_train(mode, dt):
    """A 37 s +- 0.3 s train over 2 h: period in 35-40 s, Rayleigh p < 1e-4."""
    st, eng = make_store(), TimingEngine()
    feed = Feed(st)
    rng = np.random.default_rng(11)
    t0 = T0 - dt
    if mode == "iid":
        k = np.arange(0, int(7200 / 37) + 2)
        times = t0 + 37.0 * k + rng.normal(0, 0.3, k.size)
    else:
        times = t0 + np.cumsum(37.0 + rng.normal(0, 0.3, 200))
    n_ticks = int(7200 / dt) + 1
    now = run_stream(st, feed, eng, times, T0, n_ticks, dt=dt)
    rec = timing(st, now)
    assert 35.0 <= rec["period"] <= 40.0
    assert rec["period_p"] < 1e-4
    assert rec["B"] < -0.9                                   # strictly regular
    per, pp = TM.period(model(st), now)
    assert per == rec["period"] and pp == rec["period_p"]
    last2h = times[(times > now - 7200) & (times <= now)]
    assert evt.rayleigh_p(last2h, per) < 1e-4
    assert TM.period(model(st), now + 7 * 3600)[0] != TM.period(model(st), now + 7 * 3600)[0]


def test_c_no_period_for_poisson_and_jittered_beacon():
    """Poisson streams never confirm (FPR <= the confirm level on 30 seeds);
    a renewal-jittered (+-30 %) beacon is left to B12."""
    hits = 0
    for seed in range(30):
        rng = np.random.default_rng(100 + seed)
        t = np.cumsum(rng.exponential(25.0, 400))
        _, p, _ = strict_period(t[t < 7200])
        hits += int(p == p and p <= 1e-3)
    assert hits <= 1
    rng = np.random.default_rng(5)
    t = np.cumsum(300.0 * (1 + 0.3 * rng.standard_normal(24)))
    _, p, _ = strict_period(t)
    assert not p <= 1e-3
    # too few events or too short a span: unscored, never p = 1
    per, p, n = strict_period([1.0, 2.0, 3.0])
    assert math.isnan(per) and math.isnan(p) and n == 3


def test_c_sampled_stream_gives_no_spurious_period():
    """A capped scraper keeps one chunk per tick; the holes are tick-periodic
    and must not read as a strict train."""
    st, eng = make_store(), TimingEngine()
    feed = Feed(st)
    rng = np.random.default_rng(2)
    sc = scraper_times(rng, T0 - DT, T0 + 12 * DT)
    now = run_stream(st, feed, eng, sc, T0, 10)
    rec = timing(st, now)
    assert math.isnan(rec["period"])
    assert rec["B"] < -0.8


# ------------------------------------------------------------------ edges
def test_engine_metadata_and_registry_construction():
    eng = TimingEngine()
    assert eng.name and eng.layer == "behavior" and eng.interval == 1
    for name in ("model.timing", "behavior.score", "behavior.acc_alarm", "behavior.timing",
                 "profile.extra.timing"):
        assert name in eng.produces
    for name in ("act.stream", "act.stream_frac", "model.seq", "behavior.trust",
                 "model.control"):
        assert name in eng.consumes


def test_empty_store():
    st = make_store()
    assert run_engine(TimingEngine(), st, T0) == 0
    assert st.systems() == []


def test_silent_entity_writes_nothing():
    st, eng = make_store(), TimingEngine()
    feed = Feed(st)
    for i in range(8):
        feed.tick(E, T0 + i * DT, [])
        run_engine(eng, st, T0 + i * DT)
    assert model(st) is None
    assert math.isnan(score(st, T0 + 7 * DT))
    assert emit.read_dict(st, S, E, emit.DEGRADED, T0 + 7 * DT) == {}


def test_silence_is_data_gap_spans_silent_ticks():
    """Complete ticks around 3 silent ticks: the gap across the silence is a
    true gap and lands in the histogram; an unknown tick breaks the chain."""
    st, eng = make_store(), TimingEngine()
    feed = Feed(st)
    feed.tick(E, T0, [T0 - 20.0, T0 - 10.0])
    run_engine(eng, st, T0)
    for i in (1, 2, 3):
        feed.tick(E, T0 + i * DT, [])
        run_engine(eng, st, T0 + i * DT)
    t4 = T0 + 4 * DT
    feed.tick(E, t4, [t4 - 5.0])
    run_engine(eng, st, t4)
    rows = model(st)["rows"]
    i = rows.find(t4)
    assert i >= 0
    gap = (t4 - 5.0) - (T0 - 10.0)
    assert rows.counts[i][TM.bin_index([gap])[0]] == 1 and rows.counts[i].sum() == 1
    # events with unknown times (act.events > 0, no stream) break the chain
    t5 = t4 + DT
    st.add_raw(RawMetric(name="act.events", value=3.0, ts=t5, system=S, entity=E))
    st.add_vec(S, E, "feature.active", t5, np.array([1.0], np.float32))
    run_engine(eng, st, t5)
    assert math.isnan(model(st)["live"]["last_ev"])


def test_nan_inputs():
    st, eng = make_store(), TimingEngine()
    st.put_model(S, E, "model.seq", {"session_gap": float("nan")})
    feed = Feed(st)
    # NaN stream_frac reads as complete; NaN rows are dropped and the tick
    # then counts as sampled (a dropped row would merge two gaps)
    feed.tick(E, T0, [T0 - 30.0, T0 - 20.0, T0 - 10.0], frac=float("nan"))
    run_engine(eng, st, T0)
    t1 = T0 + DT
    feed.tick(E, t1, [t1 - 30.0, float("nan"), t1 - 20.0, t1 - 10.0])
    run_engine(eng, st, t1)
    rows = model(st)["rows"]
    assert rows.counts[rows.find(t1)].sum() == 2              # no boundary gap, no NaN gap
    assert rows.stats[rows.find(t1)][0] > 1.0                 # weight 1/frac
    # all-NaN timestamps with events: unknown, chain broken, no crash
    t2 = t1 + DT
    feed.tick(E, t2, [float("nan")] * 3)
    run_engine(eng, st, t2)
    assert math.isnan(model(st)["live"]["last_ev"])
    # immature model: unscored (NaN), never p = 1
    assert math.isnan(score(st, t2)) and math.isnan(pm(st, t2))
    assert TM.loglik(None, [1.0, 2.0]) != TM.loglik(None, [1.0, 2.0])
    assert np.isnan(TM.loglik_per_gap(model(st), [float("nan"), -1.0])).all()


def test_training_mode_raises_no_alarm():
    st, feed, eng, _, rng, now = trained_human(n_ticks=120)
    sc = scraper_times(rng, now, now + 6 * DT)
    for i in range(1, 5):
        t = now + i * DT
        feed.tick(E, t, per_tick(sc, t, DT))
        run_engine(eng, st, t, training=True)
        assert pm(st, t) < 1e-3                               # still scored ...
        assert emit.read_dict(st, S, E, emit.ACC_ALARM, t).get(DETECTOR) == 0   # ... no alarm
    assert st.events() == []


def test_cadence_switch_900_to_60():
    """Wall-clock maths: after 2 d at 900 s the same human at 60 s ticks keeps
    its descriptors, learns on and raises no alarm."""
    st, feed, eng, times, _, now = trained_human()
    b900 = timing(st, now)["B"]
    mass0 = TM.mass(model(st))
    t_live = [now + 60.0 * i for i in range(1, 361)]
    set_trust(st, S, E, t_live, 1.0)
    alarms = 0
    for t in t_live:
        feed.tick(E, t, per_tick(times, t, 60.0), dt=60.0)
        run_engine(eng, st, t, dt=60.0)
        alarms += emit.read_dict(st, S, E, emit.ACC_ALARM, t).get(DETECTOR, 0)
    t = t_live[-1]
    assert alarms == 0
    assert pm(st, t) > 1e-3
    assert abs(timing(st, t)["B"] - b900) < 0.15
    assert TM.mass(model(st)) > mass0 * math.exp(-6 * 3600 / TM.TAU_S) + 50


def test_degraded_r2_failure_and_stale_events():
    st, feed, eng, _, _, now = trained_human(n_ticks=100)
    t = now + DT
    st.put_health("raw.action_token", {"engine": "raw.action_token", "last_error_ts": t})
    run_engine(eng, st, t)
    assert math.isnan(score(st, t))
    assert emit.read_dict(st, S, E, emit.DEGRADED, t) == {
        DETECTOR: "producer_error:raw.action_token"}
    # R2 did not zero-fill a recently seen entity: stale input
    t2 = t + DT
    run_engine(eng, st, t2)
    assert emit.read_dict(st, S, E, emit.DEGRADED, t2) == {DETECTOR: "stale:act.stream"}
    assert math.isnan(score(st, t2))
    # a brand-new entity during an R2 failure is degraded too
    st.add_raw(RawMetric(name="l4.flows", value=1.0, ts=t2, system=S, entity="10.0.0.2"))
    st.put_health("raw.action_token", {"engine": "raw.action_token", "last_error_ts": t2})
    run_engine(eng, st, t2)
    assert emit.read_dict(st, S, "10.0.0.2", emit.DEGRADED, t2) == {
        DETECTOR: "producer_error:raw.action_token"}


def test_quarantine_holds_then_release_commits():
    st, feed, eng, times, _, now = trained_human(n_ticks=120)
    t_live = [now + i * DT for i in range(1, 11)]
    set_trust(st, S, E, t_live, 1.0, quarantine=1.0)
    for t in t_live:
        feed.tick(E, t, per_tick(times, t, DT))
        run_engine(eng, st, t)
    gate = model(st)["gate"]
    assert len(gate.held) >= 5
    mass_q = TM.mass(model(st))
    n_journal = len(gate.journal)
    # B28: quarantine is read at t - 1, so it writes 0 there with the release
    t = t_live[-1] + DT
    set_trust(st, S, E, [t_live[-1], t], 1.0, quarantine=0.0)
    st.put_model(S, E, "model.control", {"release": [T0, t]})
    feed.tick(E, t, per_tick(times, t, DT))
    run_engine(eng, st, t)
    g2 = model(st)["gate"]
    assert len(g2.held) == 0
    assert len(g2.journal) >= n_journal + 5
    assert TM.mass(model(st)) > mass_q + 50


def test_rollback_restores_fit_on_rows_before_tau():
    """rollback_to tau: the committed state equals a fit on the rows <= tau
    (checkpoint + replay through the engine's row buffer)."""
    def run(pollute: bool):
        st, feed, eng, times, rng, now = trained_human(n_ticks=120, seed=4)
        sc = scraper_times(np.random.default_rng(8), now, now + 12 * DT)
        t_live = [now + i * DT for i in range(1, 9)]
        if pollute:
            set_trust(st, S, E, t_live, 1.0)
        for t in t_live:
            feed.tick(E, t, per_tick(sc, t, DT))
            run_engine(eng, st, t)
        return st, feed, eng, now, t_live[-1]

    ref, _, _, tau, _ = run(False)                           # scraper learned at weight 0
    st, feed, eng, tau2, t_end = run(True)
    assert tau == tau2
    polluted = model(st)["state"].copy()
    assert TM.mass({"state": polluted}) > TM.mass(model(ref)) + 100
    t = t_end + DT
    set_trust(st, S, E, [t_end], 1.0, quarantine=1.0)
    st.put_model(S, E, "model.control", {"rollback_to": tau})
    feed.tick(E, t, [])
    run_engine(eng, st, t)
    got, want = model(st)["state"], model(ref)["state"]
    assert np.allclose(got[TM.HIST], want[TM.HIST], rtol=1e-9, atol=1e-9)
    assert np.allclose(got[:TM.T], want[:TM.T], rtol=1e-9, atol=1e-9, equal_nan=True)


def test_rerun_same_tick_is_idempotent():
    st, feed, eng, times, _, now = trained_human(n_ticks=100)
    s1, st1 = score(st, now), model(st)["state"].copy()
    n_rows = len(model(st)["rows"])
    run_engine(eng, st, now, training=True)
    assert len(model(st)["rows"]) == n_rows
    assert score(st, now) == s1
    assert np.array_equal(model(st)["state"], st1, equal_nan=True)


def test_profile_extra_timing():
    st, feed, eng, _, _, now = trained_human(n_ticks=100)
    ex = st.profile(S, E).extra["timing"]
    for k in ("B", "M", "think_mu", "think_sigma", "gap_p50_s", "n_eff", "period", "recent",
              "hist", "version"):
        assert k in ex
    assert len(ex["hist"]) == TM.N_BINS
    assert abs(sum(v for v in ex["hist"] if v is not None) - 1.0) < 1e-3


def test_link_seeding_merges_other_entity():
    st, feed, eng, _, _, now = trained_human(n_ticks=100)
    e2 = "10.0.0.2"
    t = now + DT
    feed.tick(e2, t, [t - 30.0, t - 20.0, t - 9.0])
    run_engine(eng, st, t, training=True)
    m_a = TM.mass(model(st))
    st.put_model(S, "__system__", "model.link",
                 {"version": 1, "links": [{"from": E, "to": e2, "ts": t}]})
    t2 = t + DT
    feed.tick(e2, t2, [t2 - 30.0])
    run_engine(eng, st, t2, training=True)
    assert TM.mass(model(st, e2)) >= 0.5 * m_a * 0.99


# ------------------------------------------------------------------ m_timing
def test_m_timing_accessors():
    st, *_ = trained_human(n_ticks=100)
    m = model(st)
    p = TM.pmf(m)
    assert abs(p.sum() - 1.0) < 1e-12 and (p > 0).all()
    q = TM.quantiles(m, [0.1, 0.5, 0.9])
    assert np.all(np.diff(q) > 0)
    assert math.isnan(TM.quantile(m, 1.5))
    lp = TM.think_logpdf(m, [8.0, 1000.0])
    assert lp[0] > lp[1]
    assert TM.dispersion(m, 0) >= 1.0 and TM.dispersion(None) == TM.PHI_PRIOR
    for f in (TM.burstiness, TM.memory, TM.n_eff, TM.mass):
        v = f(None)
        assert v != v or v == 0.0
    assert TM.is_empty(None) and TM.descriptors({})["B"] != TM.descriptors({})["B"]
    assert TM.bin_index([0.001, 0.01, 86400.0, 1e9, -1.0, float("nan")]).tolist() == \
        [0, 0, TM.N_BINS - 1, TM.N_BINS - 1, -1, -1]
    assert np.allclose(TM.gaps_from_times([3.0, 1.0, 2.0]), [1.0, 1.0])
    assert TM.gaps_from_times([0.0, 100.0, 101.0], frac=0.5).tolist() == [1.0]
    # the engine's fused comparison equals the documented pure functions
    rng = np.random.default_rng(0)
    a, b = rng.random(32), rng.random(32)
    a[3] = 0.0
    jsd, g, df = _compare(a, 50.0, b, 400.0)
    g2, df2 = TM.g_two_sample(a, 50.0, b, 400.0)
    assert abs(jsd - TM.jsd_bits(a, b)) < 1e-12 and abs(g - g2) < 1e-9 and df == df2
    assert TM.jsd_bits(a, a) < 1e-12 and math.isnan(TM.jsd_bits(np.zeros(32), a))


# ------------------------------------------------------------------ perf
def test_perf_40_entities():
    """Spec: <= 2 ms per tick per entity (generous bound on shared CI)."""
    st, eng = make_store(), TimingEngine()
    feed = Feed(st)
    rng = np.random.default_rng(1)
    ents = [f"10.0.1.{i}" for i in range(40)]
    streams = {e: human_times(rng, T0 - DT, T0 + 30 * DT) for e in ents}
    ms = []
    for i in range(24):
        now = T0 + i * DT
        for e in ents:
            feed.tick(e, now, per_tick(streams[e], now, DT))
        t0 = time.perf_counter()
        run_engine(eng, st, now, training=True)
        ms.append((time.perf_counter() - t0) * 1000.0)
    per_entity = float(np.median(ms[4:])) / len(ents)
    assert per_entity < 2.0, per_entity


def test_active_tick_without_a_true_gap_writes_undefined_descriptors():
    """An active tick whose window holds no true gap yet (a lone event, e.g.
    the first event of a new entity or a sampled tick) used to write no
    behavior.timing at all, so the series went stale while the entity was
    active (pack C, round 2). It now writes undefined (NaN) descriptors."""
    st, eng = make_store(), TimingEngine()
    feed = Feed(st)
    feed.tick(E, T0, [T0 - 100.0])
    run_engine(eng, st, T0, training=True, dt=DT)
    d = timing(st, T0)
    assert d is not None and set(d) >= {"B", "M", "think_mu", "think_sigma", "period"}
    assert all(not (v == v) for k, v in d.items() if k in ("B", "M", "think_mu", "think_sigma"))
    feed.tick(E, T0 + DT, [])                           # a silent tick writes nothing
    run_engine(eng, st, T0 + DT, training=True, dt=DT)
    assert timing(st, T0 + DT) is None
