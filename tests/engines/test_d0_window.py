"""D0 window engines (engines/derived/{aggregation,periodicity,trend}.py).

Spec unit test (docs/lib3/engines.md, D0): http.requests on 1 of every 4
ticks for 96 ticks at Δt = 900 with act.events zero-filled ->
  * aggregation mean ≈ 0.25 × the active value;
  * periodicity lag = 3600 s with score > 0.5;
  * an entity silent for more than 6 h gets no window metrics.
Plus: dims {span_s, n_active}, ts = now, retention of the grid inputs, gauges
NaN-skipped, empty store, NaN inputs, training (no events), cadence
900 -> 60, trend split-half behaviour, perf.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from helpers import T0, add_obs_tick, run_engine
from helpers import make_store as _bare_store

from app.engines.derived import fresh
from app.engines.derived.aggregation import SPAN_S, AggregationEngine, ensure_retention, grid_many
from app.engines.derived.periodicity import PeriodicityEngine, autocorr_fft
from app.engines.derived.trend import TrendEngine

S, E = "erp", "10.0.0.1"
DT = 900.0
V = 40.0


def make_store():
    """A store whose retention the D0 engines have already extended, as in the
    pipeline, where they run from the first tick (retention prunes on append,
    so data written before the first run would be cut to the 6-h default)."""
    st = _bare_store()
    ensure_retention(st, TrendEngine.DEFAULT_TARGETS, 13 * 3600.0)
    return st


def _d(st, name, now, e=E, s=S):
    """The derived point written at exactly `now`, else None."""
    for m in reversed(st.derived_tail(s, e, name, 4)):
        if m.ts == now:
            return m
    return None


def _tick(st, e, t, active, extra=None, value=V):
    """One raw tick as R1 + R2 would write it: act.events every tick (touch
    only when active), the counters only when active."""
    add_obs_tick(st, S, e, t, {"act.events": value if active else 0.0}, touch=active)
    if active:
        m = {"http.requests": value}
        m.update(extra or {})
        add_obs_tick(st, S, e, t, m)


def _periodic_store(n=96, e=E, every=4, extra=None):
    st = make_store()
    for i in range(n):
        _tick(st, e, T0 + i * DT, i % every == 0, extra)
    return st, T0 + (n - 1) * DT


def _run_all(st, now, dt=DT, training=False):
    return sum(run_engine(eng, st, now, dt=dt, training=training)
               for eng in (AggregationEngine(), PeriodicityEngine(), TrendEngine()))


def _window_names(st, e=E):
    return [n for n in st.derived_names(S, e)]


# ------------------------------------------------------------------ spec
def test_spec_aggregation_mean_is_quarter_of_active_value():
    st, now = _periodic_store()
    run_engine(AggregationEngine(), st, now, dt=DT)
    m = _d(st, "derived.http.requests.mean", now)
    assert m is not None and m.value == pytest.approx(0.25 * V, rel=0.02)
    assert _d(st, "derived.http.requests.sum", now).value == pytest.approx(6 * V)
    assert _d(st, "derived.http.requests.max", now).value == pytest.approx(V)
    # 24 ticks in 6 h, 6 active; dims carried on every output
    assert m.dims == {"span_s": SPAN_S, "n_active": 6}
    assert m.ts == now and m.inputs == ["http.requests"]


def test_spec_periodicity_lag_3600_score_above_half():
    st, now = _periodic_store()
    run_engine(PeriodicityEngine(), st, now, dt=DT)
    score = _d(st, "derived.periodicity_score", now)
    lag = _d(st, "derived.beacon_lag", now)
    assert score is not None and score.value > 0.5
    assert lag.value == pytest.approx(3600.0)
    assert score.dims == {"span_s": SPAN_S, "n_active": 6}
    # constant active value -> perfectly regular over active ticks
    assert _d(st, "derived.timing_regularity", now).value == pytest.approx(1.0)


def test_spec_silent_entity_over_6h_gets_no_window_metrics():
    st, _ = _periodic_store(n=40)
    # the entity then falls silent: act.events keeps being zero-filled
    t = T0 + 39 * DT
    for i in range(1, 30):                   # 29 * 900 s ≈ 7.25 h of silence
        t = T0 + (39 + i) * DT
        _tick(st, E, t, False)
    # a second, active entity proves the engines did run
    _tick(st, "10.0.0.2", t, True)
    n = _run_all(st, t)
    assert n > 0
    assert not any(_d(st, name, t) for name in _window_names(st))
    assert _d(st, "derived.http.requests.mean", t, e="10.0.0.2") is not None


def test_silent_under_6h_still_windows_idle_time():
    st, _ = _periodic_store(n=40)
    t = T0 + 39 * DT
    for i in range(1, 9):                    # 2 h of silence
        t = T0 + (39 + i) * DT
        _tick(st, E, t, False)
    run_engine(AggregationEngine(), st, t, dt=DT)
    m = _d(st, "derived.http.requests.mean", t)
    # 24 grid ticks, active at i % 4 == 0 among the first 16 -> 4 active
    assert m.dims["n_active"] == 4 and m.value == pytest.approx(4 * V / 24)


# ------------------------------------------------------------------ semantics
def test_gauges_skip_nan_and_counters_zero_fill():
    st, now = _periodic_store(extra={"http.latency_ms_avg": 30.0})
    run_engine(AggregationEngine(), st, now, dt=DT)
    assert _d(st, "derived.http.latency_ms_avg.mean", now).value == pytest.approx(30.0)
    assert _d(st, "derived.http.latency_ms_avg.cv", now).value == pytest.approx(0.0)
    assert _d(st, "derived.http.requests.cv", now).value == pytest.approx(math.sqrt(3.0))
    # channels never used in the span are not written
    assert _d(st, "derived.dns.queries.mean", now) is None
    assert _d(st, "derived.probe.rtt_ms.mean", now) is None


def test_retention_extended_for_grid_inputs():
    st = _bare_store()
    now = T0 + 3 * DT
    for i in range(4):
        _tick(st, E, T0 + i * DT, True, {"l4.bytes_up": 100.0})
    _run_all(st, now)
    # write 20 h of data after the engines set retention: 24 h inputs survive
    for i in range(4, 84):
        _tick(st, E, T0 + i * DT, True, {"l4.bytes_up": 100.0, "l4.rtt_ms_avg": 1.0})
    ts0 = [m.ts for m in st.raw_series(S, E, "http.requests")]
    assert T0 + 83 * DT - ts0[0] > 12 * 3600
    ev = [m.ts for m in st.raw_series(S, E, "act.events")]
    assert T0 + 83 * DT - ev[0] > 12 * 3600
    # trend targets are kept for the 12-h span (+1 h), other raw scalars 6 h
    for i in range(84, 100):
        _tick(st, E, T0 + i * DT, True, {"l4.bytes_up": 100.0, "l4.rtt_ms_avg": 1.0})
    up = [m.ts for m in st.raw_series(S, E, "l4.bytes_up")]
    assert T0 + 99 * DT - up[0] == pytest.approx(13 * 3600)
    rtt = [m.ts for m in st.raw_series(S, E, "l4.rtt_ms_avg")]
    assert T0 + 99 * DT - rtt[0] == pytest.approx(6 * 3600)
    _run_all(st, now)                        # idempotent


def test_empty_store():
    st = make_store()
    assert _run_all(st, T0) == 0


def test_nan_inputs_do_not_poison_stats():
    st = make_store()
    for i in range(24):
        t = T0 + i * DT
        add_obs_tick(st, S, E, t, {"act.events": 5.0, "http.requests": 5.0,
                                   "http.latency_ms_avg": math.nan if i % 3 else 20.0})
    now = T0 + 23 * DT
    _run_all(st, now)
    assert _d(st, "derived.http.latency_ms_avg.mean", now).value == pytest.approx(20.0)
    for name in _window_names(st):
        m = _d(st, name, now)
        if m is not None:
            assert math.isfinite(m.value), name
    # all-NaN gauge -> nothing written
    st2 = make_store()
    for i in range(10):
        add_obs_tick(st2, S, E, T0 + i * DT, {"act.events": 1.0, "l4.rtt_ms_avg": math.nan})
    run_engine(AggregationEngine(), st2, T0 + 9 * DT, dt=DT)
    assert _d(st2, "derived.l4.rtt_ms_avg.mean", T0 + 9 * DT) is None


def test_training_mode_emits_no_events():
    st, now = _periodic_store()
    assert _run_all(st, now, training=True) > 0
    assert st.events() == []


def test_every_output_is_stamped_now_with_window_dims():
    st, now = _periodic_store(extra={"l4.flows": 3.0, "http.latency_ms_avg": 12.0})
    n = _run_all(st, now)
    written = [_d(st, name, now) for name in _window_names(st)]
    written = [m for m in written if m is not None]
    assert len(written) == n > 0
    for m in written:
        assert m.ts == now and set(m.dims) == {"span_s", "n_active"}
        assert m.dims["span_s"] in (SPAN_S, 12 * 3600.0) and m.dims["n_active"] > 0
    # nothing is re-stamped at an older tick
    assert all(len(st.derived_tail(S, E, m.name, 10)) == 1 for m in written)


def test_nan_counter_values_are_skipped():
    st = make_store()
    for i in range(48):
        t = T0 + i * DT
        v = math.nan if i == 45 else (V if i % 4 == 0 else None)
        add_obs_tick(st, S, E, t, {"act.events": 1.0 if v is not None else 0.0},
                     touch=v is not None)
        if v is not None:
            add_obs_tick(st, S, E, t, {"http.requests": v})
    now = T0 + 47 * DT
    _run_all(st, now)
    assert _d(st, "derived.http.requests.sum", now).value == pytest.approx(6 * V)
    assert _d(st, "derived.http.requests.mean", now).value == pytest.approx(6 * V / 23)
    assert _d(st, "derived.beacon_lag", now).value == pytest.approx(3600.0)
    assert math.isfinite(_d(st, "derived.http.requests.changepoint", now).value)


def test_no_act_events_clock_falls_back_to_regular_grid():
    st = make_store()
    for i in range(24):
        if i % 4 == 0:
            add_obs_tick(st, S, E, T0 + i * DT, {"http.requests": V})
    now = T0 + 23 * DT
    add_obs_tick(st, S, E, now, {"l4.flows": 1.0})
    _run_all(st, now)
    m = _d(st, "derived.http.requests.mean", now)
    assert m.value == pytest.approx(0.25 * V) and m.dims["n_active"] == 7


def test_scheduled_run_via_safe_run():
    st, now = _periodic_store()
    for eng in (AggregationEngine(), PeriodicityEngine(), TrendEngine()):
        assert run_engine(eng, st, now, dt=DT, scheduled=True) > 0
        assert st.health()[eng.name]["ok"]


def test_grid_many_matches_fresh_grid():
    st = make_store()
    for i in range(30):
        t = T0 + i * DT
        _tick(st, E, t, i % 3 == 0, {"http.latency_ms_avg": 10.0 + i})
    add_obs_tick(st, S, E, T0 + 29 * DT + 7.0, {"http.requests": 99.0})   # off-grid point
    now = T0 + 29 * DT
    names = ["http.requests", "http.latency_ms_avg", "dns.queries"]
    ts, clock, kept, mat = grid_many(st, S, E, names, {}, now, SPAN_S, DT, skip_stale=False)
    assert kept == names
    g = dict(zip(kept, mat))
    for name in names:
        kind = "gauge" if name.endswith("_avg") else "counter"
        ts_f, v_f = fresh.grid(st, S, E, name, now, SPAN_S, kind, DT)
        assert np.array_equal(ts, ts_f)
        assert np.array_equal(g[name], v_f, equal_nan=True), name
    assert np.array_equal(clock, fresh.grid(st, S, E, "act.events", now, SPAN_S, "counter", DT)[1])
    _ts, _c, kept2, mat2 = grid_many(st, S, E, names, {}, now, SPAN_S, DT)
    assert kept2 == ["http.requests", "http.latency_ms_avg"] and mat2.shape == (2, 24)


# ------------------------------------------------------------------ periodicity
def test_autocorr_fft_matches_direct():
    rng = np.random.default_rng(3)
    x = rng.poisson(3, 50).astype(float)
    x[::5] += 20
    lag, r = autocorr_fft(x)
    c = x - x.mean()
    direct = [float(c[:-k] @ c[k:]) / float(c @ c) for k in range(2, 26)]
    assert lag == 5 and r == pytest.approx(max(direct))
    assert autocorr_fft(np.ones(30)) == (0, 0.0)
    assert autocorr_fft(np.arange(5.0)) == (0, 0.0)


def test_bursty_random_series_scores_low():
    st = make_store()
    rng = np.random.default_rng(7)
    for i in range(48):
        v = float(rng.poisson(2) * (rng.random() < 0.6))
        _tick(st, E, T0 + i * DT, v > 0, value=v)
    now = T0 + 47 * DT
    run_engine(PeriodicityEngine(), st, now, dt=DT)
    assert _d(st, "derived.periodicity_score", now).value < 0.5
    assert _d(st, "derived.timing_regularity", now).value < 1.0


def test_single_active_tick_has_no_regularity():
    st = make_store()
    for i in range(24):
        _tick(st, E, T0 + i * DT, i == 23)
    now = T0 + 23 * DT
    run_engine(PeriodicityEngine(), st, now, dt=DT)
    assert _d(st, "derived.periodicity_score", now) is not None
    assert _d(st, "derived.timing_regularity", now) is None


# ------------------------------------------------------------------ cadence
def _cadence_switch_store(n_fine, rate_per_min=2.0, beacon_s=None):
    """24 h at 900-s ticks then n_fine 60-s ticks. Without beacon_s: a constant
    rate. With beacon_s: one burst of 60 events every beacon_s seconds."""
    st = make_store()
    t = T0
    for i in range(96):
        t = T0 + i * DT
        if beacon_s:
            on = (t - T0) % beacon_s == 0
            _tick(st, E, t, on, value=60.0)
        else:
            _tick(st, E, t, True, value=rate_per_min * 15.0)
    t_switch = t
    for j in range(1, n_fine + 1):
        t = t_switch + 60.0 * j
        if beacon_s:
            on = (t - T0) % beacon_s == 0
            _tick(st, E, t, on, value=60.0)
        else:
            _tick(st, E, t, True, value=rate_per_min)
    return st, t


def test_cadence_switch_aggregation_is_rate_consistent():
    st, now = _cadence_switch_store(30)
    run_engine(AggregationEngine(), st, now, dt=60.0)
    # mean / max / p95 in per-current-tick units: 2 per 60-s tick
    assert _d(st, "derived.http.requests.mean", now).value == pytest.approx(2.0, rel=0.02)
    assert _d(st, "derived.http.requests.max", now).value == pytest.approx(2.0, rel=0.02)
    assert _d(st, "derived.http.requests.cv", now).value == pytest.approx(0.0, abs=1e-9)
    # sum is the true total over the 6-h span
    tot = _d(st, "derived.http.requests.sum", now).value
    assert tot == pytest.approx(2.0 * 360, rel=0.05)


def test_cadence_switch_is_not_a_trend():
    st, now = _cadence_switch_store(30)
    run_engine(TrendEngine(), st, now, dt=60.0)
    assert abs(_d(st, "derived.http.requests.changepoint", now).value) < 1.0
    assert _d(st, "derived.http.requests.ewma", now).value == pytest.approx(2.0, rel=0.02)
    assert abs(_d(st, "derived.http.requests.slope", now).value) < 1e-6
    run_engine(PeriodicityEngine(), st, now, dt=60.0)
    assert _d(st, "derived.timing_regularity", now).value == pytest.approx(1.0)
    assert _d(st, "derived.periodicity_score", now).value < 0.5


def test_cadence_switch_keeps_beacon_lag_in_seconds():
    st, now = _cadence_switch_store(120, beacon_s=3600.0)
    run_engine(PeriodicityEngine(), st, now, dt=60.0)
    assert _d(st, "derived.beacon_lag", now).value == pytest.approx(3600.0)
    assert _d(st, "derived.periodicity_score", now).value > 0.5
    # fully at 60 s after 6 h: same lag in seconds, fine resolution
    t = now
    for j in range(1, 361):
        t = now + 60.0 * j
        _tick(st, E, t, (t - T0) % 3600.0 == 0, value=60.0)
    run_engine(PeriodicityEngine(), st, t, dt=60.0)
    assert _d(st, "derived.beacon_lag", t).value == pytest.approx(3600.0)
    assert _d(st, "derived.periodicity_score", t).value > 0.5


# ------------------------------------------------------------------ trend
def test_trend_stationary_sparse_series_is_not_a_changepoint():
    st, now = _periodic_store()
    run_engine(TrendEngine(), st, now, dt=DT)
    cp = _d(st, "derived.http.requests.changepoint", now)
    assert cp is not None and abs(cp.value) < 1.0
    assert cp.dims == {"span_s": 12 * 3600.0, "n_active": 12}


def test_trend_step_up_is_detected():
    st = make_store()
    for i in range(96):
        _tick(st, E, T0 + i * DT, True, value=10.0 if i < 72 else 40.0)
    now = T0 + 95 * DT
    run_engine(TrendEngine(), st, now, dt=DT)
    assert _d(st, "derived.http.requests.changepoint", now).value > 5.0
    assert _d(st, "derived.http.requests.slope", now).value > 0.0
    assert _d(st, "derived.http.requests.ewma", now).value > 30.0


def test_trend_gauge_nan_skipped_and_short_series_skipped():
    st = make_store()
    for i in range(48):
        t = T0 + i * DT
        add_obs_tick(st, S, E, t, {"act.events": 1.0})
        if i % 2 == 0:
            add_obs_tick(st, S, E, t, {"http.latency_ms_avg": 50.0 if i < 24 else 100.0})
    now = T0 + 47 * DT
    run_engine(TrendEngine(), st, now, dt=DT)
    assert _d(st, "derived.http.latency_ms_avg.changepoint", now).value == pytest.approx(10.0)
    assert _d(st, "derived.http.latency_ms_avg.ewma", now).value == pytest.approx(100.0, rel=0.01)
    st2 = make_store()
    for i in range(4):
        add_obs_tick(st2, S, E, T0 + i * DT, {"act.events": 1.0, "http.latency_ms_avg": 5.0})
    run_engine(TrendEngine(), st2, T0 + 3 * DT, dt=DT)
    assert _d(st2, "derived.http.latency_ms_avg.ewma", T0 + 3 * DT) is None


# ------------------------------------------------------------------ perf
def test_perf_many_entities():
    st = make_store()
    ents = [f"10.1.{i // 250}.{i % 250}" for i in range(100)]
    for i in range(96):
        t = T0 + i * DT
        for k, e in enumerate(ents):
            on = (i + k) % 3 == 0
            add_obs_tick(st, S, e, t, {"act.events": 10.0 if on else 0.0}, touch=on)
            if on:
                add_obs_tick(st, S, e, t, {"http.requests": 10.0, "l4.flows": 4.0,
                                           "l4.bytes_up": 1e4, "http.latency_ms_avg": 20.0})
    now = T0 + 95 * DT
    engines = (AggregationEngine(), PeriodicityEngine(), TrendEngine())
    _run_all(st, now)                        # warm
    t0 = time.perf_counter()
    for eng in engines:
        run_engine(eng, st, now, dt=DT)
    per_entity_ms = (time.perf_counter() - t0) * 1000.0 / len(ents)
    # measured ≈ 1.8 ms per entity for all three at 900-s ticks; generous bound
    assert per_entity_ms < 6.0, per_entity_ms


def test_periodicity_undefined_when_the_span_holds_too_few_ticks():
    """A 6-h span at 3600-s ticks holds 6 grid points (< MIN_TICKS = 8): no
    autocorrelation can be estimated, so no periodicity score / lag is
    written (B01 reads the window feature as NaN), while regularity still
    is. A written 0 taught every hourly-cadence baseline bucket a point mass
    at 0 that the first 900-s estimate hit at z >> 10 (eval pack A)."""
    st = make_store()
    dt = 3600.0
    rng = np.random.default_rng(5)
    for i in range(24):
        _tick(st, E, T0 + i * dt, True, value=float(rng.poisson(40)))
    now = T0 + 23 * dt
    run_engine(PeriodicityEngine(), st, now, dt=dt)
    assert _d(st, "derived.periodicity_score", now) is None
    assert _d(st, "derived.beacon_lag", now) is None
    assert _d(st, "derived.timing_regularity", now) is not None
