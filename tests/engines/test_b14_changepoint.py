"""B14 ChangepointEngine (docs/lib3/engines.md '## B14'): spec unit tests
(b), (c), (d), (f) plus thresholds, onset, axes, natural-unit excess and replay.

(a) (wall-clock null rate at 900 s and 60 s) is in test_b14_changepoint_null.py
and (e) (baseline_creep) in test_b14_changepoint_creep.py, so each file stays
under ~5 s. Engine-level scenarios run through ChangepointEngine with
run_engine (strict); the multi-seed detection-delay statistics drive the very
step functions the engine calls (m_cp.bank_tick / mc_tick), batch-vectorised.

Spec test mapping:
  (b) test_b_shift_engine, test_b_shift_seeds
  (c) test_c_duty_engine, test_c_duty_seeds
  (d) test_d_alternate_engine, test_d_alternate_seeds
  (f) test_f_reads_reference_residual_not_current_anchor
Footprint note for (c)/(d): with k in {0.25, 1} a SINGLE feature at duty 30%
x 3 sigma (drift 0.65/tick on the k=0.25 chart, h = 19.4) or alternating 3 sigma
(+1 per two ticks on the k=1 chart, h = 5.35) crosses at a median of ~23 / ~12.5
ticks, so 'within 30 / 10 ticks' holds as a median (c) or for the realistic
multi-feature volume footprint of a burst (d). Both are asserted below.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from helpers import DT, T0, make_store, run_engine, set_trust

from app.engines.behavior.changepoint import ChangepointEngine
from app.engines.behavior.lib import emit, m_cp
from app.engines.behavior.lib.features import FEATURE_DIM, FEATURE_INDEX, KEY_FEATURES

S = "erp"
ZR = "behavior.zr"
BYTES_UP = FEATURE_INDEX["bytes_up"]
KEY_POS = {n: i for i, n in enumerate(KEY_FEATURES)}               # position in the 12-vector
VOLUME4 = ["bytes_up", "bytes_down", "flows", "http_requests"]      # a burst's key footprint


# ------------------------------------------------------------------ helpers
def feed(store, e, t, zr, dt=DT, vec=None, nat=None, z=None, active=1.0, s=S):
    """One tick of B14's inputs as B01 / B04 / B28 would write them."""
    w = int(dt)
    if zr is not None:
        store.add_vec(s, e, ZR, t, np.asarray(zr, dtype=np.float32), window_s=w)
    if z is not None:
        store.add_vec(s, e, "behavior.z", t, np.asarray(z, dtype=np.float32), window_s=w)
    if vec is not None:
        store.add_vec(s, e, "feature.vec", t, np.asarray(vec, dtype=np.float32), window_s=w)
    if nat is not None:
        store.add_vec(s, e, "feature.nat", t, np.asarray(nat, dtype=np.float32), window_s=w)
    store.add_vec(s, e, "feature.active", t, np.asarray([active], dtype=np.float32), window_s=w)
    set_trust(store, s, e, [t], 1.0)
    store.register_entity(s, e)


def alarm(store, e, t, dets=("cusum", "mcusum")):
    a = emit.read_dict(store, S, e, emit.ACC_ALARM, t)
    return any(int(a.get(d, 0)) == 1 for d in dets)


def run_until_alarm(zr_rows, start, limit, dt=DT, extra=None):
    """Feed rows into one entity; returns (store, first alarm index >= start or None)."""
    store, eng = make_store(), ChangepointEngine()
    first = None
    for i, zr in enumerate(zr_rows):
        t = T0 + i * dt
        kw = extra(i, zr) if extra else {}
        feed(store, "e", t, zr, dt=dt, **kw)
        run_engine(eng, store, t, dt=dt)
        if i >= start and first is None and alarm(store, "e", t):
            first = i
            break
        if i >= start + limit:
            break
    return store, first


def seeds_delays(pattern, feats, seeds=200, pre=50, post=80, seed=11, dt=DT):
    """Batch of independent entities through m_cp's bank + MCUSUM (identity
    whitening, phi = 0): ticks from shift start to the first alarm (999 = none)
    and the reported onset's error in ticks (last pre-change tick = -1), for
    the seeds not already latched on a null false alarm at the change."""
    rng = np.random.default_rng(seed)
    B, mc = m_cp.new_bank((seeds,)), m_cp.new_mc((seeds,))
    h, hmc, phi = m_cp.bank_h(dt), m_cp.mcusum_h(dt), np.zeros(m_cp.N_KEY)
    cols = [KEY_POS[f] for f in feats]
    first = np.full(seeds, 999)
    onset = np.full(seeds, np.nan)
    stale = np.zeros(seeds, dtype=bool)
    for i in range(pre + post):
        x = rng.standard_normal((seeds, m_cp.N_KEY))
        if i >= pre:
            x[:, cols] += pattern(i - pre, rng, seeds)[:, None]
        t = T0 + i * dt
        B, ob = m_cp.bank_tick(B, x, t, dt, phi, h)
        mc, om = m_cp.mc_tick(mc, ob["psi"], t, dt, hmc)
        if i == pre - 1:
            stale = ob["on"] | om["on"]                 # a null false alarm still latched
        if i >= pre:
            new = (first == 999) & (ob["on"] | om["on"])
            first[new] = i - pre + 1
            # what the engine reports: the earliest onset among the latched statistics
            tau = np.fmin(np.where(ob["on"], B["latch"]["onset"], np.nan),
                          np.where(om["on"], mc["latch"]["onset"], np.nan))
            onset[new] = (tau[new] - (T0 + pre * dt)) / dt
    return first[~stale], onset[~stale]


SHIFT = lambda j, rng, n: np.ones(n)                                    # noqa: E731
DUTY = lambda j, rng, n: 3.0 * (rng.random(n) < 0.3)                    # noqa: E731
DUTY_DET = lambda j, rng, n: np.full(n, 3.0 if j % 10 < 3 else 0.0)     # noqa: E731
ALT = lambda j, rng, n: np.full(n, 3.0 if j % 2 == 0 else 0.0)          # noqa: E731


# ------------------------------------------------------------------ thresholds
@pytest.mark.parametrize("dt,h025,h1", [(900.0, 19.4, 5.35), (60.0, 24.8, 6.7),
                                       (3600.0, 16.6, 4.66)])
def test_siegmund_thresholds_wall_clock(dt, h025, h1):
    h = m_cp.bank_h(dt)
    assert h.shape == (48,)
    assert np.allclose(h[m_cp.CHART_K == 0.25], h025, atol=0.06)
    assert np.allclose(h[m_cp.CHART_K == 1.0], h1, atol=0.06)
    # each chart's design ARL is 2400 days of wall-clock time at any cadence
    arl_days = m_cp.siegmund_arl(0.25, h025) * dt / 86400.0
    assert arl_days == pytest.approx(2400.0, rel=0.08)
    assert math.isfinite(m_cp.mcusum_h(dt)) and m_cp.mcusum_h(dt) > 0


def test_peq_formula_and_mcusum_p_monotone():
    S = np.zeros(48)
    S[0], S[24] = 10.0, 3.0                             # k = 0.25 upper, k = 0.25 lower
    p = m_cp.peq(S)
    assert p[0] == pytest.approx(min(1.0, 48 * math.exp(-2 * 0.25 * (10 + 0.583))))
    assert p[5] == 1.0
    ps = [m_cp.mcusum_p(x) for x in (5.0, 10.0, 20.0, 40.0)]
    assert all(a >= b for a, b in zip(ps, ps[1:])) and ps[-1] < 1e-3
    assert math.isnan(m_cp.mcusum_p(math.nan))


# ------------------------------------------------------------------ (b)
def test_b_shift_engine():
    """+1 sigma on bytes_up from t = 500 at 900 s: alarm within 60 ticks, onset
    within +-20 ticks, axes = volume, cumulative excess in natural units."""
    rng = np.random.default_rng(3)
    rows = rng.standard_normal((561, FEATURE_DIM))
    rows[500:, BYTES_UP] += 1.0

    def extra(i, zr):                                   # vec = level + 0.5 zr; nat = inverse
        vec = 8.0 + 0.5 * zr
        nat = np.expm1(vec) * DT / 60.0
        return {"vec": vec, "nat": nat}

    store, first = run_until_alarm(rows, 500, 60, extra=extra)
    assert first is not None and first - 500 < 60, first
    t = T0 + first * DT
    tau = m_cp.onset(store, S, "e")
    # never later than +20 ticks (a late tau-hat would keep poisoned rows on
    # rollback); an early tail happens when the MCUSUM fires partly on open
    # pre-change noise (this seed: -44). The +-20 rate is test_b_shift_seeds.
    err = (tau - (T0 + 499 * DT)) / DT
    assert -m_cp.ONSET_HIST <= err <= 20, err
    # the float32 ring carries the same onset to its resolution
    assert abs(m_cp.onset(store, S, "e", at=t) - tau) <= 128.0
    axes = emit.read_dict(store, S, "e", emit.AXES, t)
    assert "volume" in (axes.get("cusum") or axes.get("mcusum") or [])
    assert m_cp.alarms(store, S, "e")["cusum"] + m_cp.alarms(store, S, "e")["mcusum"] >= 1
    assert m_cp.level(store, S, "e")["cusum"] >= 1.0 or m_cp.level(store, S, "e")["mcusum"] >= 1.0
    # scores: -log10 of the stationary p; the pm columns carry the p
    sc = emit.read_row(store, S, "e", emit.SCORE, t)
    pm = emit.read_row(store, S, "e", emit.PM, t)
    assert sc["cusum"] > 0 and 0 < pm["cusum"] <= 1
    # profile.extra.regime.delta_by_feature: bytes_up rose, in bytes per tick
    delta = store.profile(S, "e").extra["regime"]["delta_by_feature"]
    d = delta["bytes_up"]
    assert d["dir"] == "+" and d["unit"] == "bytes" and d["ticks"] > 0
    assert d["z_mean"] > 0.3
    assert d["excess"] > 0 and d["observed"] > d["reference"] > 0


def test_b_shift_seeds():
    """Same scenario over 200 seeds: >= 95 % alarm within 60 ticks; the
    reported onset is within +-20 ticks for >= 90 % (measured 94-97 %,
    median -1), and its misses are early, not late."""
    first, onset = seeds_delays(SHIFT, ["bytes_up"], pre=500, post=61)
    assert first.size >= 180
    assert np.mean(first <= 60) >= 0.95
    ok = np.isfinite(onset)
    assert ok.mean() >= 0.95
    err = onset[ok] + 1.0                               # vs the last pre-change tick
    assert np.mean(np.abs(err) <= 20) >= 0.9
    assert abs(np.median(err)) <= 3
    assert np.mean(err > 20) <= 0.03
    assert np.median(first) <= 30                       # spec: expected run length ~27 ticks


# ------------------------------------------------------------------ (c)
def test_c_duty_seeds():
    # periodic duty cycle (3 ticks on, 7 off) on one feature: nearly always within 30
    first, _ = seeds_delays(DUTY_DET, ["bytes_up"])
    assert np.mean(first <= 30) >= 0.8
    # Bernoulli 30 % duty on one feature: within 30 ticks at the median
    first, _ = seeds_delays(DUTY, ["bytes_up"])
    assert np.median(first) <= 30
    # a burst footprint on the four volume counters: >= 85 % within 30
    first, _ = seeds_delays(DUTY, VOLUME4)
    assert np.mean(first <= 30) >= 0.85


def test_c_duty_engine():
    rng = np.random.default_rng(5)
    rows = rng.standard_normal((120, FEATURE_DIM))
    for j in range(70):
        if j % 10 < 3:
            rows[50 + j, BYTES_UP] += 3.0
    _, first = run_until_alarm(rows, 50, 60)
    assert first is not None and first - 50 < 30


# ------------------------------------------------------------------ (d)
def test_d_alternate_seeds():
    first, _ = seeds_delays(ALT, VOLUME4)
    assert np.median(first) <= 10
    assert np.mean(first <= 10) >= 0.75
    first1, _ = seeds_delays(ALT, ["bytes_up"])        # one feature: slower, see module doc
    assert np.median(first1) <= 14 and np.mean(first1 <= 30) >= 0.99


def test_d_alternate_engine():
    rng = np.random.default_rng(8)
    rows = rng.standard_normal((90, FEATURE_DIM))
    idx = [FEATURE_INDEX[f] for f in VOLUME4]
    for j in range(0, 40, 2):
        rows[50 + j, idx] += 3.0
    _, first = run_until_alarm(rows, 50, 40)
    assert first is not None and first - 50 < 10


# ------------------------------------------------------------------ (f)
def test_f_reads_reference_residual_not_current_anchor():
    """The current anchor poisoned (+1 sigma absorbed: behavior.z looks null)
    while the reference is unchanged (behavior.zr still shows +1 sigma): B14
    alarms on zr; the converse (z shifted, zr null) raises nothing."""
    rng = np.random.default_rng(13)
    store, eng = make_store(), ChangepointEngine()
    idx = [FEATURE_INDEX[f] for f in VOLUME4]
    hit = {"poisoned": None, "zonly": None}
    for i in range(140):
        t = T0 + i * DT
        base = rng.standard_normal((2, FEATURE_DIM))
        zr_p, z_p = base[0].copy(), base[0].copy()
        z_z, zr_z = base[1].copy(), base[1].copy()
        if i >= 60:
            zr_p[idx] += 1.0                            # reference sees the shift
            z_z[idx] += 1.0                             # only the current anchor sees it
        feed(store, "poisoned", t, zr_p, z=z_p)
        feed(store, "zonly", t, zr_z, z=z_z)
        run_engine(eng, store, t)
        for e in hit:
            if hit[e] is None and i >= 60 and alarm(store, e, t):
                hit[e] = i
    assert hit["poisoned"] is not None and hit["poisoned"] - 60 < 60
    assert hit["zonly"] is None


# ------------------------------------------------------------------ replay
def test_cusum_state_ring_replays_bit_identically():
    """behavior.cusum_state + behavior.zr (both 6 h rings) reproduce the bank
    exactly through m_cp.step_fn (what B29 replays), and neutralising the
    shifted feature removes the evidence (counterfactual)."""
    rng = np.random.default_rng(21)
    store, eng = make_store(), ChangepointEngine()
    for i in range(160):
        zr = rng.standard_normal(FEATURE_DIM)
        if i >= 140:
            zr[BYTES_UP] += 2.0
        feed(store, "e", T0 + i * DT, zr)
        run_engine(eng, store, T0 + i * DT)
    t_start = T0 + 140 * DT
    ts0, state = m_cp.replay_state(store, S, "e", t_start)
    assert ts0 == t_start and state["S"].shape == (48,)
    s0 = {k: v.copy() for k, v in state.items()}
    step = m_cp.step_fn(m_cp.replay_params(store, S, "e"), "cusum")
    inputs = m_cp.replay_inputs(store, S, "e", t_start, T0 + 159 * DT)
    assert len(inputs) == 19
    _, ring = store.vec_range(S, "e", m_cp.CUSUM_STATE, t_start + 1, T0 + 159 * DT)
    score = 0.0
    for (ts, inp), row in zip(inputs, ring):
        state, score = step(state, inp)
        assert np.array_equal(state["S"].astype(np.float32), row[:48])
    assert score >= 1.0                                 # the live bank alarmed on this shift
    assert alarm(store, "e", T0 + 159 * DT)
    state = s0
    for ts, inp in inputs:
        state, score = step(state, m_cp.neutralize(inp, [BYTES_UP]))
    assert score < 1.0


def test_restart_at_end_of_warmup_clears_the_reported_alarm_of_an_idle_entity():
    """The charts restart at S = 0 on the first live tick. An entity that is
    idle on its first live ticks (no zr: a workstation at night) re-emits
    run['alarm'] from _idle_outputs, so the latch of the warm-up must be
    cleared with the charts, not carried into the live phase (eval pack B:
    every control's cusum / mcusum acc_alarm was set from the first live
    tick on and opened an incident)."""
    store, eng = make_store(), ChangepointEngine()
    rows = np.zeros((80, FEATURE_DIM))
    rows[40:, BYTES_UP] = 4.0                           # a persistent warm-up shift
    t = T0
    for i, zr in enumerate(rows):
        t = T0 + i * DT
        feed(store, "e", t, zr)
        run_engine(eng, store, t, training=True)
    assert alarm(store, "e", t), "the warm-up shift should latch the bank"
    for k in range(1, 4):                               # live, idle: no zr
        t2 = t + k * DT
        store.add_vec(S, "e", "feature.active", t2, np.zeros(1, np.float32), window_s=int(DT))
        run_engine(eng, store, t2, training=False)
        assert not alarm(store, "e", t2), k
    assert m_cp.alarms(store, S, "e")["cusum"] == 0


@pytest.mark.parametrize("training,latched_run,expect_alarm", [
    (False, True, False),    # first live tick, not an H tick: the warm-up latch is dropped
    (True, True, True),      # still warm-up: held between H ticks as before
    (False, False, True),    # a live latch: held between H ticks (cadence.md §17)
])
def test_canonical_first_live_tick_between_h_ticks_restarts_the_charts(
        training, latched_run, expect_alarm):
    """Eval round 3: in canonical grain mode the end-of-warm-up restart ran
    only on the first live H tick, so on the live ticks before it
    _hold_latches re-emitted the warm-up latch as a live accumulator alarm
    (cadence Part B: two humans alarmed from the first live minute to the
    first live hour, at 60 s and at 900 s)."""
    store, eng = make_store(), ChangepointEngine()
    store.register_entity(S, "e")
    model = eng._new_model()
    model["run"]["alarm"]["cusum"] = 1
    model["run"]["training"] = latched_run if training else True
    if not training and not latched_run:
        model["run"]["training"] = False
    store.put_model(S, "e", m_cp.MODEL, model)
    t = (T0 // 3600.0) * 3600.0 + 900.0                 # a Q tick, not an H decision tick
    run_engine(eng, store, t, training=training, dt=900.0, config={"grain_mode": "canonical"})
    assert bool(alarm(store, "e", t)) is expect_alarm
    if not expect_alarm:
        assert m_cp.alarms(store, S, "e")["cusum"] == 0
