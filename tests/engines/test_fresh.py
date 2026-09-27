"""derived/fresh.py: freshness gate and zero-filled wall-clock grid."""
import math

import numpy as np

from helpers import T0, DT, add_obs_tick, make_store

from app.engines.derived import fresh

S, E = "sys", "10.0.0.1"


def test_fresh_raw_only_at_now():
    st = make_store()
    add_obs_tick(st, S, E, T0, {"http.requests": 5.0})
    assert fresh.fresh_raw(st, S, E, "http.requests", T0) == 5.0
    assert fresh.fresh_raw(st, S, E, "http.requests", T0 + DT) is None
    assert fresh.all_fresh(st, S, E, ["http.requests", "l4.flows"], T0) is None


def test_grid_zero_fills_counters_and_nans_gauges():
    st = make_store()
    for i in range(8):
        t = T0 + i * DT
        add_obs_tick(st, S, E, t, {"act.events": 4.0 if i % 4 == 0 else 0.0}, touch=i % 4 == 0)
        if i % 4 == 0:
            add_obs_tick(st, S, E, t, {"http.requests": 4.0, "http.latency_ms_avg": 30.0})
    now = T0 + 7 * DT
    ts, v = fresh.grid(st, S, E, "http.requests", now, 8 * DT, "counter", DT)
    assert len(ts) == 8 and list(v) == [4, 0, 0, 0, 4, 0, 0, 0]
    _, g = fresh.grid(st, S, E, "http.latency_ms_avg", now, 8 * DT, "gauge", DT)
    assert g[0] == 30.0 and math.isnan(g[1])
    assert fresh.n_active(st, S, E, now, 8 * DT, DT) == 2


def test_grid_falls_back_to_regular_steps():
    st = make_store()
    add_obs_tick(st, S, E, T0 + 3 * DT, {"http.requests": 2.0})
    ts, v = fresh.grid(st, S, E, "http.requests", T0 + 3 * DT, 4 * DT, "counter", DT)
    assert np.allclose(ts, T0 + DT * np.arange(4)) and list(v) == [0, 0, 0, 2]
