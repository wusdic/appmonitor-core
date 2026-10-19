"""B14 round 8 (conformity / scoring / changepoint owner): a run B14 starts
late - bounded mode's promotion of an IP to the earned set, or a model resumed
after the IP left it - carries the IP's retained pre-promotion statistics.

PG9 pack A seed 0 (§16.15.8): oa-portal 10.30.2.23 was promoted on 03-05
01:00, a week after full mode's model of it started; on 03-12 05:00 the
younger BOCPD run read cp 0.994 (full mode 0.206) and a bocpd accumulator
alarm became a MEDIUM incident full mode does not raise (8 vs 7).
"""
from __future__ import annotations

import math

import numpy as np

from helpers import T0, make_store, run_engine
from test_b14_changepoint import S, feed

from app.engines.behavior import changepoint as CP
from app.engines.behavior.lib import m_cp
from app.engines.behavior.lib.features import FEATURE_DIM

DT = 3600.0
N = 90


def _inputs(seed: int = 5):
    """N hourly ticks: zr rows (None on idle hours), B06 wh scores."""
    rng = np.random.default_rng(seed)
    zr, wh = [], []
    for i in range(N):
        if i % 24 in (2, 3, 4) or i in (40, 41):
            zr.append(None)                         # idle hours (no zr), a 2-hour gap
            wh.append(None)
            continue
        x = rng.standard_normal(FEATURE_DIM)
        if 30 <= i < 50:
            x = x * 0.3                             # a quiet stretch the late run never sees
        zr.append(x)
        wh.append(float(rng.standard_normal()) if i % 5 else None)
    return zr, wh


def _store():
    store = make_store()
    for name in (CP.ZR, CP.WH, "feature.meta.h"):   # B04's / B06's 4-day H-row retention
        store.ensure_retention(name, max_age_s=4 * 86400.0)
    zr, wh = _inputs()
    for i in range(N):
        t = T0 + i * DT
        feed(store, "e", t, zr[i], dt=DT, active=0.0 if zr[i] is None else 1.0)
        if wh[i] is not None:
            store.add_vec(S, "e", CP.WH, t, np.asarray([wh[i]], dtype=np.float32), window_s=int(DT))
    return store


def _run(ticks):
    """B14 on its own store, processing only the given ticks (the store holds
    every tick's inputs, as B03 / B04 / B06 write them for an active IP)."""
    store = _store()
    eng = CP.ChangepointEngine()
    for i in ticks:
        run_engine(eng, store, T0 + i * DT, dt=DT)
    return store.get_model(S, "e", m_cp.MODEL)["run"]


def _same_bocpd(a, b):
    for k in ("r", "p", "mu", "ka", "al", "be", "n"):
        assert np.allclose(a[k], b[k], rtol=0, atol=0), k


def test_a_late_started_run_carries_the_ips_retained_history():
    full = _run(range(N))
    late = _run(range(60, N))                    # promoted at tick 60
    assert CP.bocpd_steps(full["bocpd"]) == CP.bocpd_steps(late["bocpd"]) > 80
    _same_bocpd(full["bocpd"], late["bocpd"])
    assert math.isclose(full["cp_prob"], late["cp_prob"], rel_tol=0, abs_tol=0)
    assert full["daily"]["idx"] == late["daily"]["idx"]
    assert all(np.array_equal(a, b, equal_nan=True) for a, b in zip(full["daily"]["val"], late["daily"]["val"]))
    assert np.array_equal(full["day"]["sum"], late["day"]["sum"])


def test_a_resumed_run_replays_the_ticks_it_skipped():
    full = _run(range(N))
    gap = _run(list(range(0, 35)) + list(range(70, N)))   # left the earned set for 35 h
    _same_bocpd(full["bocpd"], gap["bocpd"])
    assert full["daily"]["idx"] == gap["daily"]["idx"]


def test_a_run_processed_every_tick_is_unchanged():
    """Full mode: the model starts at the first zr row, every later tick is
    processed, so there is nothing to replay (bit-identical by construction)."""
    store = _store()
    eng = CP.ChangepointEngine()
    calls = []
    orig = eng._catch_up

    def spy(*a, **k):
        n = orig(*a, **k)
        calls.append(n)
        return n
    eng._catch_up = spy
    for i in range(N):
        run_engine(eng, store, T0 + i * DT, dt=DT)
    assert calls and max(calls) == 0


# ------------------------------------------------------------ bounded mode
N_LONG = 150                                     # > the 4-day zr retention
PROMOTE = 120


def _bounded_cfg():
    return {"lib3": {"resource_mode": "bounded"}}


def _publish(store, now, earned):
    from app.models.schema import ORG
    store.put_model(ORG, ORG, "model.budget",
                    {"systems": {S: {"active": ["e"], "earned": ["e"] if earned else [], "ts": now}}},
                    ts=now)


def _long_store():
    store = make_store()
    for name in (CP.ZR, CP.WH, "feature.meta.h"):
        store.ensure_retention(name, max_age_s=4 * 86400.0)
    rng = np.random.default_rng(9)
    return store, rng


def _feed_long(store, rng, i):
    t = T0 + i * DT
    if i % 24 in (3, 4, 5):
        feed(store, "e", t, None, dt=DT, active=0.0)
        return
    x = rng.standard_normal(FEATURE_DIM) * (2.0 if i < 30 else 1.0)   # a noisier first stretch
    feed(store, "e", t, x, dt=DT)
    store.add_vec(S, "e", CP.WH, t, np.asarray([float(rng.standard_normal())], dtype=np.float32),
                  window_s=int(DT))


def test_a_promoted_ip_starts_from_its_history_beyond_the_zr_retention():
    """Bounded mode keeps BOCPD's two inputs of every active IP (CP_IN) for
    R_MAX + 1 day; an IP promoted after 120 h of activity (zr retained 96 h)
    gets exactly full mode's run."""
    sf, rf = _long_store()
    sb, rb = _long_store()
    ef, eb = CP.ChangepointEngine(), CP.ChangepointEngine()
    for i in range(N_LONG):
        t = T0 + i * DT
        _feed_long(sf, rf, i)
        _feed_long(sb, rb, i)
        run_engine(ef, sf, t, dt=DT)
        _publish(sb, t, i >= PROMOTE)
        run_engine(eb, sb, t, dt=DT, config=_bounded_cfg())
        if i < PROMOTE:
            assert sb.get_model(S, "e", m_cp.MODEL) is None          # not modelled while unearned
    full = sf.get_model(S, "e", m_cp.MODEL)["run"]
    bnd = sb.get_model(S, "e", m_cp.MODEL)["run"]
    assert CP.bocpd_steps(full["bocpd"]) == CP.bocpd_steps(bnd["bocpd"]) > 140
    _same_bocpd(full["bocpd"], bnd["bocpd"])
    assert math.isclose(full["cp_prob"], bnd["cp_prob"], rel_tol=0, abs_tol=0)
    assert CP.CP_IN not in sf.vec_names(S, "e")                         # full mode: nothing new
