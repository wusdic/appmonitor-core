"""D2 SessionEngine (engines/derived/session.py).

Spec unit test (docs/lib3/engines.md, D2): act.events active on 1 tick in 4
over 96 ticks, with 8-s gaps in act.stream:
  - duty = 0.25 +- 0.02;
  - think_time ~ 8 s on active ticks and absent on idle ticks;
  - the window metrics are written on every tick.
Plus: session gap from model.seq, completed-session window, stream_frac
reweighting and sampled-gap rule, cross-tick sessions, bootstrap from a
store with history, empty store, silent entity, long-silent entity, NaN
inputs, training mode (no events), cadence 900 -> 60, replay, end-to-end
with R2, and perf.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from helpers import DT, T0, add_obs_tick, make_store, obs, put_model, run_engine

from app.engines.behavior.lib import m_template as MT
from app.engines.derived.session import SessionEngine
from app.models.schema import AcquisitionMethod, RawMetric

S, E = "sys", "10.0.0.1"
DAY = 86400.0
WINDOW = ("derived.activity_duty_cycle", "derived.session_count")
ALL_OUT = WINDOW + ("derived.req_per_session", "derived.think_time_s_avg")


def _rows(ts_list, frac_rows=None):
    a = np.zeros(len(ts_list), dtype=MT.STREAM_DTYPE)
    a["ts"] = np.asarray(ts_list, dtype=np.float64)
    a["token_id"] = 1
    a["outcome"] = 2
    a.flags.writeable = False
    return a


def _active(st, now, ev_ts, frac=1.0, e=E, events=None):
    """One active R2 tick: act.events, act.stream (rows at ev_ts), stream_frac."""
    n = float(events if events is not None else len(ev_ts) / frac)
    add_obs_tick(st, S, e, now, {"act.events": n, "act.stream": _rows(ev_ts),
                                 "act.stream_frac": frac})


def _idle(st, now, e=E):
    """R2's zero fill for a known entity with no observation (touch=False)."""
    st.add_raw(RawMetric(name="act.events", value=0.0, ts=now, system=S, entity=e,
                         method=AcquisitionMethod.PASSIVE_SPAN), touch=False)


def _burst(start, n=10, gap=8.0):
    return [start + k * gap for k in range(n)]


def _at(st, name, ts, e=E):
    m = st.latest_derived(S, e, name)
    return m if (m is not None and m.ts == ts) else None


def _v(st, name, ts, e=E):
    m = _at(st, name, ts, e)
    return None if m is None else m.value


def _spec_run(eng=None, training=False):
    st = make_store()
    eng = eng or SessionEngine()
    written = []
    for i in range(96):
        now = T0 + i * DT
        if i % 4 == 0:
            _active(st, now, _burst(now - DT + 10.0))
        else:
            _idle(st, now)
        run_engine(eng, st, now, training=training, dt=DT)
        written.append({n: _v(st, n, now) for n in ALL_OUT})
    return st, eng, written


# ------------------------------------------------------------------ spec
def test_spec_duty_think_and_window_every_tick():
    st, eng, written = _spec_run()
    last = T0 + 95 * DT
    duty = _v(st, "derived.activity_duty_cycle", last)
    assert duty == pytest.approx(0.25, abs=0.02)
    assert _at(st, "derived.activity_duty_cycle", last).dims == {"span_s": DAY, "n_active": 24}
    for i, w in enumerate(written):
        # window metrics on every tick
        for n in WINDOW:
            assert w[n] is not None, (i, n)
        if i % 4 == 0:
            assert w["derived.think_time_s_avg"] == pytest.approx(8.0, abs=1e-6)
        else:
            assert w["derived.think_time_s_avg"] is None, i
        # req_per_session from the first completed session on (1 h idle > 30 min)
        if i >= 2:
            assert w["derived.req_per_session"] == pytest.approx(10.0)
    # 24 bursts, each its own session (3600 s apart > G = 1800), all completed
    assert _v(st, "derived.session_count", last) == 24.0
    m = _at(st, "derived.think_time_s_avg", T0 + 92 * DT)
    assert m.dims["n"] == 9 and m.dims["session_gap_s"] == 1800.0


def test_duty_ramps_on_the_wall_clock_grid():
    _st, _eng, written = _spec_run()
    # tick 0 is the only tick so far and active; after 4 ticks 1 of 4 is active
    assert written[0]["derived.activity_duty_cycle"] == 1.0
    assert written[3]["derived.activity_duty_cycle"] == pytest.approx(0.25)
    assert written[5]["derived.activity_duty_cycle"] == pytest.approx(2 / 6)


# ---------------------------------------------------------- session gap / window
def test_session_gap_from_model_seq():
    def run(gap):
        st = make_store()
        if gap is not None:
            put_model(st, S, E, "model.seq", {"session_gap": gap})
        eng = SessionEngine()
        now = T0
        # two bursts 600 s apart in one tick, then silence
        _active(st, now, _burst(now - 890.0, 5) + _burst(now - 250.0, 5))
        run_engine(eng, st, now)
        for k in range(1, 4):
            _idle(st, now + k * DT)
            run_engine(eng, st, now + k * DT)
        return st, now + 3 * DT
    st, t = run(300.0)
    assert _v(st, "derived.session_count", t) == 2.0
    assert _v(st, "derived.req_per_session", t) == 5.0
    assert _at(st, "derived.session_count", t).dims["session_gap_s"] == 300.0
    st, t = run(None)                      # default 30 min: one session
    assert _v(st, "derived.session_count", t) == 1.0
    assert _v(st, "derived.req_per_session", t) == 10.0
    st, t = run(float("nan"))              # invalid gap -> default
    assert _v(st, "derived.session_count", t) == 1.0


def test_open_session_not_counted_until_idle_longer_than_gap():
    st = make_store()
    eng = SessionEngine()
    _active(st, T0, _burst(T0 - 100.0, 5))
    run_engine(eng, st, T0)
    assert _v(st, "derived.session_count", T0) == 0.0
    assert _v(st, "derived.req_per_session", T0) is None      # undefined, not 0
    _idle(st, T0 + DT)
    run_engine(eng, st, T0 + DT)          # idle 900+68 s < 1800: still open
    assert _v(st, "derived.session_count", T0 + DT) == 0.0
    _idle(st, T0 + 2 * DT)
    run_engine(eng, st, T0 + 2 * DT)
    assert _v(st, "derived.session_count", T0 + 2 * DT) == 1.0


def test_sessions_leave_the_24h_window():
    st = make_store()
    eng = SessionEngine()
    _active(st, T0, _burst(T0 - 100.0, 4))
    run_engine(eng, st, T0)
    for i in range(1, 98):
        _idle(st, T0 + i * DT)
        run_engine(eng, st, T0 + i * DT)
        if i == 90:
            assert _v(st, "derived.session_count", T0 + i * DT) == 1.0
    t = T0 + 97 * DT
    assert _v(st, "derived.session_count", t) == 0.0
    assert _v(st, "derived.req_per_session", t) is None
    assert _v(st, "derived.activity_duty_cycle", t) == 0.0


def test_session_spans_ticks_and_boundary_gap_is_think_time():
    st = make_store()
    eng = SessionEngine()
    _active(st, T0, [T0 - 30.0, T0 - 20.0])
    run_engine(eng, st, T0)
    # 50 s after the previous tick's last event, then 20-s gaps
    _active(st, T0 + DT, [T0 + 30.0, T0 + 50.0, T0 + 70.0])
    run_engine(eng, st, T0 + DT)
    m = _at(st, "derived.think_time_s_avg", T0 + DT)
    assert m.dims["n"] == 3 and m.value == pytest.approx(20.0)   # median(50, 20, 20)
    for k in (2, 3, 4):
        _idle(st, T0 + k * DT)
        run_engine(eng, st, T0 + k * DT)
    t = T0 + 4 * DT
    assert _v(st, "derived.session_count", t) == 1.0
    assert _v(st, "derived.req_per_session", t) == 5.0


def test_think_time_needs_two_gaps():
    st = make_store()
    eng = SessionEngine()
    _active(st, T0, [T0 - 50.0, T0 - 40.0])      # one gap only
    run_engine(eng, st, T0)
    assert _v(st, "derived.think_time_s_avg", T0) is None
    assert _v(st, "derived.activity_duty_cycle", T0) == 1.0
    # identical timestamps carry no timing: still one usable gap
    st = make_store()
    _active(st, T0, [T0 - 100.0, T0 - 100.0, T0 - 100.0, T0 - 95.0])
    run_engine(SessionEngine(), st, T0)
    assert _v(st, "derived.think_time_s_avg", T0) is None


# ---------------------------------------------------------- stream_frac
def test_stream_frac_reweights_events_not_sessions():
    def run(frac):
        st = make_store()
        eng = SessionEngine()
        _active(st, T0, _burst(T0 - 500.0, 10, 2.0), frac=frac)
        run_engine(eng, st, T0)
        for k in (1, 2, 3):
            _idle(st, T0 + k * DT)
            run_engine(eng, st, T0 + k * DT)
        return st, T0 + 3 * DT
    st, t = run(1.0)
    assert (_v(st, "derived.session_count", t), _v(st, "derived.req_per_session", t)) == (1.0, 10.0)
    st, t = run(0.25)
    assert _v(st, "derived.session_count", t) == 1.0
    assert _v(st, "derived.req_per_session", t) == pytest.approx(40.0)


def test_sampled_tick_uses_only_certain_gaps():
    # 60-s gaps: true gaps when complete, possibly spanning dropped sessions
    # when R2 sampled the tick (only gaps <= 30 s are certain then)
    for frac, expect in ((1.0, 60.0), (0.5, None)):
        st = make_store()
        _active(st, T0, _burst(T0 - 800.0, 6, 60.0), frac=frac)
        run_engine(SessionEngine(), st, T0)
        assert _v(st, "derived.think_time_s_avg", T0) == expect
    st = make_store()
    ts = _burst(T0 - 800.0, 4, 5.0) + _burst(T0 - 400.0, 4, 5.0)
    _active(st, T0, ts, frac=0.5)
    run_engine(SessionEngine(), st, T0)
    m = _at(st, "derived.think_time_s_avg", T0)
    assert m.value == 5.0 and m.dims["n"] == 6


def test_missing_frac_means_complete_rows():
    st = make_store()
    add_obs_tick(st, S, E, T0, {"act.events": 6.0, "act.stream": _rows(_burst(T0 - 800, 6, 60.0))})
    run_engine(SessionEngine(), st, T0)
    assert _v(st, "derived.think_time_s_avg", T0) == 60.0


# ---------------------------------------------------------- edge cases
def test_empty_store():
    st = make_store()
    assert run_engine(SessionEngine(), st, T0) == 0


def test_silent_entity_writes_nothing():
    st = make_store()
    st.register_entity(S, E)
    eng = SessionEngine()
    for i in range(3):
        assert run_engine(eng, st, T0 + i * DT) == 0
    assert st.derived_names(S, E) == []


def test_idle_entity_reports_decaying_window_not_stale_values():
    st = make_store()
    eng = SessionEngine()
    _active(st, T0, _burst(T0 - 800.0))
    run_engine(eng, st, T0)
    duties = []
    for i in range(1, 12):
        _idle(st, T0 + i * DT)
        run_engine(eng, st, T0 + i * DT)
        duties.append(_v(st, "derived.activity_duty_cycle", T0 + i * DT))
        assert _v(st, "derived.think_time_s_avg", T0 + i * DT) is None
    assert duties == sorted(duties, reverse=True)
    assert duties[-1] == pytest.approx(1 / 12)


def test_no_clock_this_tick_writes_no_window():
    """R2 stopped zero-filling (entity silent > 30 d) or did not run: the grid
    has no tick at now, so nothing is re-stamped."""
    st = make_store()
    eng = SessionEngine()
    _active(st, T0, _burst(T0 - 800.0))
    run_engine(eng, st, T0)
    assert run_engine(eng, st, T0 + DT) == 0
    for n in ALL_OUT:
        assert _at(st, n, T0 + DT) is None


def test_nan_inputs():
    st = make_store()
    eng = SessionEngine()
    ts = [T0 - 100.0, math.nan, T0 - 90.0, math.inf, T0 - 80.0]
    add_obs_tick(st, S, E, T0, {"act.events": math.nan, "act.stream": _rows(ts),
                                "act.stream_frac": math.nan})
    run_engine(eng, st, T0)
    assert _v(st, "derived.think_time_s_avg", T0) == 10.0       # non-finite rows dropped
    assert _v(st, "derived.activity_duty_cycle", T0) == 0.0     # NaN events: not active
    add_obs_tick(st, S, E, T0 + DT, {"act.events": 3.0, "act.stream": "garbage",
                                     "act.stream_frac": "x"})
    run_engine(eng, st, T0 + DT)
    assert _v(st, "derived.activity_duty_cycle", T0 + DT) == 0.5
    assert _v(st, "derived.think_time_s_avg", T0 + DT) is None
    for n in ALL_OUT:
        m = _at(st, n, T0 + DT)
        assert m is None or math.isfinite(m.value)


def test_unsorted_stream_rows():
    st = make_store()
    _active(st, T0, [T0 - 10.0, T0 - 30.0, T0 - 20.0, T0 - 40.0])
    run_engine(SessionEngine(), st, T0)
    assert _v(st, "derived.think_time_s_avg", T0) == 10.0


def test_training_mode_emits_no_events_same_outputs():
    st_t, _e, w_t = _spec_run(training=True)
    st_n, _e, w_n = _spec_run(training=False)
    assert st_t.events() == [] and st_n.events() == []
    assert w_t == w_n


def test_cadence_900_to_60_is_time_weighted():
    st = make_store()
    eng = SessionEngine()
    t = T0
    for i in range(48):                     # 12 h at 900 s, active every tick
        t = T0 + i * DT
        _active(st, t, _burst(t - DT + 10.0))
        run_engine(eng, st, t, dt=DT)
    for k in range(1, 61):                  # 1 h at 60 s, idle
        now = t + k * 60.0
        _idle(st, now)
        run_engine(eng, st, now, dt=60.0)
    end = t + 3600.0
    # 12 h active out of 13 h of wall clock (tick-count share would be 48/108)
    assert _v(st, "derived.activity_duty_cycle", end) == pytest.approx(12 / 13, abs=1e-6)
    assert _at(st, "derived.activity_duty_cycle", end).dims["n_active"] == 48
    # the 12-h activity was one session (gaps of ~830 s < 30 min), now complete
    assert _v(st, "derived.session_count", end) == 1.0
    assert _v(st, "derived.req_per_session", end) == 480.0
    # 60-s active ticks keep producing a fresh think time
    now = end + 60.0
    _active(st, now, _burst(now - 50.0, 4, 10.0))
    run_engine(eng, st, now, dt=60.0)
    assert _v(st, "derived.think_time_s_avg", now) == 10.0


def test_bootstrap_from_history_and_retention():
    st = make_store()
    run_engine(SessionEngine(), st, T0 - DT)      # first run sets 24 h retention
    for i in range(96):
        now = T0 + i * DT
        if i % 4 == 0:
            _active(st, now, _burst(now - DT + 10.0))
        else:
            _idle(st, now)
    last = T0 + 95 * DT
    assert st.raw_tail(S, E, "act.events", 1000)[0].ts == T0    # 24 h kept
    eng = SessionEngine()                          # fresh state, full history
    run_engine(eng, st, last)
    assert _v(st, "derived.activity_duty_cycle", last) == pytest.approx(0.25, abs=0.02)
    # act.stream is kept 1 h: sessions seen at bootstrap are the recent ones only
    # (one session per retained stream tick; the store prunes on append)
    kept = len(st.raw_tail(S, E, "act.stream", 100))
    assert 1 <= kept <= 2
    assert _v(st, "derived.session_count", last) == float(kept)
    assert _v(st, "derived.req_per_session", last) == 10.0


def test_replay_clock_backwards_resets():
    _st, eng, _w = _spec_run()
    # the same engine instance replays a fresh store from T0
    written = []
    st = make_store()
    for i in range(8):
        now = T0 + i * DT
        if i % 4 == 0:
            _active(st, now, _burst(now - DT + 10.0))
        else:
            _idle(st, now)
        run_engine(eng, st, now)
        written.append(_v(st, "derived.activity_duty_cycle", now))
    assert written[0] == 1.0 and written[7] == pytest.approx(0.25)


def test_rerun_same_tick_is_idempotent():
    st = make_store()
    eng = SessionEngine()
    _active(st, T0, _burst(T0 - 800.0))
    assert run_engine(eng, st, T0) == 3
    assert run_engine(eng, st, T0) == 0
    assert len(st.derived_tail(S, E, "derived.activity_duty_cycle", 10)) == 1


def test_scheduled_through_safe_run_and_metadata():
    eng = SessionEngine()
    assert eng.layer == "derived" and eng.interval == 1
    assert set(eng.produces) == set(ALL_OUT)
    assert {"act.events", "act.stream", "act.stream_frac", "model.seq"} <= set(eng.consumes)
    st = make_store()
    _active(st, T0, _burst(T0 - 800.0))
    assert run_engine(eng, st, T0, scheduled=True) == 3


def test_end_to_end_with_r2():
    from app.engines.raw.action_token import ActionTokenEngine
    st = make_store()
    r2, d2 = ActionTokenEngine(), SessionEngine()
    e = "10.1.2.3"
    for i in range(8):
        now = T0 + i * DT
        o = []
        if i % 4 == 0:
            for k in range(6):
                o.append(obs("erp", e, now - 600.0 + 8.0 * k, app_proto="http",
                             http_method="GET", http_host="erp.corp",
                             http_path=f"/orders/view/{k}", http_status=200,
                             peer="10.9.0.10", dst_port=80, bytes_up=300, bytes_down=5000))
        run_engine(r2, st, now, observations=o)
        run_engine(d2, st, now)
        m = st.latest_derived("erp", e, "derived.activity_duty_cycle")
        assert m is not None and m.ts == now
        tt = st.latest_derived("erp", e, "derived.think_time_s_avg")
        if i % 4 == 0:
            assert tt.ts == now and tt.value == pytest.approx(8.0)
        else:
            assert tt is None or tt.ts != now
    m = st.latest_derived("erp", e, "derived.activity_duty_cycle")
    assert m.value == pytest.approx(0.25)


# ---------------------------------------------------------- perf
def test_perf_many_entities():
    st = make_store()
    eng = SessionEngine()
    ents = [f"10.0.1.{k}" for k in range(40)]
    for e in ents:                              # all on the grid from the start
        _idle(st, T0 - DT, e)
    run_engine(eng, st, T0 - DT)
    spent = 0.0
    n_ticks = 96
    for i in range(n_ticks):
        now = T0 + i * DT
        for j, e in enumerate(ents):
            if (i + j) % 4 == 0:
                _active(st, now, _burst(now - DT + 10.0, 30, 8.0), e=e)
            else:
                _idle(st, now, e)
        t0 = time.perf_counter()
        run_engine(eng, st, now)
        spent += time.perf_counter() - t0
    per_entity_ms = spent / (n_ticks * len(ents)) * 1e3
    assert per_entity_ms < 1.0, per_entity_ms          # spec: < 1 ms (generous: per entity)
