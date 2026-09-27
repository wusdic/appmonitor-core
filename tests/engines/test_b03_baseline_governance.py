"""B03 BaselineEngine: governance (model.control rebase / frozen / allow_drift,
reference admission around incidents and abnormal regimes), long horizons
(golden anchor, hour-of-week cells, half-life selection), checkpoint block
sharing, link seeding, the pure m_baseline API B18 reuses, and a perf check.
The B01 / B28 stand-in is the one of test_b03_baseline."""
from __future__ import annotations

import copy
import time

import numpy as np
import pytest

from helpers import DT, T0, make_store, put_model, run_engine

from app.engines.behavior.baseline import LEARNER_CUR, BaselineEngine
from app.engines.behavior.lib import features as F
from app.engines.behavior.lib import m_baseline as MB
from app.models.schema import DerivedMetric, Incident, MetricKind

from test_b03_baseline import REQ, S, E, _data_rate, feed, model, nat_row, tctx

H = 3600.0


def _run(st, eng, n, dt=DT, t0=T0, rate=lambda i, t: 10.0, e_list=(E,), rng=None, **kw):
    rng = rng if rng is not None else np.random.default_rng(0)
    t = t0
    for i in range(n):
        t = t0 + i * dt
        for e in e_list:
            feed(st, t, nat_row(rng, rate(i, t), dt=dt), e=e, dt=dt, **kw)
        run_engine(eng, st, t, dt=dt)
    return t


# --------------------------------------------------------------- control
def test_rebase_commits_new_regime_uncapped_and_resets_reference():
    rng = np.random.default_rng(20)
    st, eng = make_store(), BaselineEngine()
    n_mat, n_new = 3 * 96, 96
    t = T0
    for i in range(n_mat + n_new):                    # new regime quarantined -> held
        t = T0 + i * DT
        feed(st, t, nat_row(rng, 10.0 if i < n_mat else 60.0), quarantine=1.0 if i >= n_mat else 0.0)
        run_engine(eng, st, t)
    t_new = T0 + n_mat * DT
    m = model(st)
    assert len(MB.held(m)) >= n_new - 2
    before = np.nanmax(_data_rate(m["current"]))
    assert before < 13.0
    put_model(st, S, E, "model.control", {"version": 1, "rebase_from": t_new})
    t += DT
    feed(st, t, nat_row(rng, 60.0))
    run_engine(eng, st, t)
    m = model(st)
    assert m["version"] == 1 and st.model_version(S, E, MB.MODEL) == 1
    # every held row is committed; only the row due this tick is held, since
    # quarantine(t - 1) was still 1 (gating semantics)
    assert len(MB.held(m)) <= 1
    pr = MB.predictive(st, S, E, tctx(t), model=m)
    assert pr.mu[REQ] > 35.0                              # uncapped: the new level is learned
    # the reference restarts one day into the new regime
    for k in range(1, 3 * 96):
        t += DT
        feed(st, t, nat_row(rng, 60.0))
        run_engine(eng, st, t)
    m = model(st)
    assert m["reference"].n_reset == 1
    rates = _data_rate(m["reference"])
    assert np.nanmin(rates) > 40.0                        # only new-regime rows since the reset


def test_frozen_stops_commits_and_drops_held_rows():
    rng = np.random.default_rng(21)
    st, eng = make_store(), BaselineEngine()
    t = _run(st, eng, 20, rng=rng)
    n = model(st)["current"].n_commit
    put_model(st, S, E, "model.control", {"version": 0, "frozen": True})
    t = _run(st, eng, 10, t0=t + DT, rng=rng)
    m = model(st)
    assert m["current"].n_commit == n and not MB.held(m) and m["gate"].frozen


def test_allow_drift_widens_the_reference_band():
    moved = []
    for drift in (0.0, 0.5):
        rng = np.random.default_rng(22)
        st, eng = make_store(), BaselineEngine()
        if drift:
            put_model(st, S, E, "model.control", {"version": 0, "allow_drift": drift})
        t = _run(st, eng, 7 * 24, dt=H, rng=rng)
        ref = model(st)["reference"]
        St = MB.true_stats(ref)
        mature = ((St[:, MB.FULL.W[REQ]] >= 1.25 * MB.CAP_MIN_W)      # clearly past the
                  & (St[:, MB.FULL.EXPO] >= 1.25 * MB.CAP_MIN_E))     # maturity threshold
        before = _data_rate(ref)
        _run(st, eng, 2 * 24, dt=H, t0=t + H, rate=lambda i, t: 30.0, rng=rng)
        after = _data_rate(model(st)["reference"])
        assert mature.sum() >= 12
        moved.append(np.max(np.abs(after - before)[mature]))
        assert model(st)["allow_drift"] == drift
    assert moved[0] < 0.5                    # 0.03 sigma15 per band-day
    assert moved[1] > moved[0] + 2.0         # + an accepted ramp of 0.5 log-units per day


# -------------------------------------------------------- reference admission
def test_reference_skips_rows_near_incidents_and_abnormal_regimes():
    rng = np.random.default_rng(23)
    st, eng = make_store(), BaselineEngine()
    e2 = "10.0.0.2"
    t_inc = T0 + 30 * H
    st.put_incident(Incident(id="i1", system=S, entity=E, entities=[E], status="closed",
                             opened=t_inc, last_seen=t_inc + 2 * H))
    reg_t = T0 + 30 * H
    for i in range(4 * 24):
        t = T0 + i * H
        for e in (E, e2):
            feed(st, t, nat_row(rng, 10.0, dt=H), e=e, dt=H)
        if t in (reg_t, reg_t + 3 * H):                   # e2: suspect for 3 h
            st.add_derived(DerivedMetric(name="behavior.regime", ts=t, system=S, entity=e2,
                                         value={"state": "suspect" if t == reg_t else "normal"},
                                         window_s=int(H), kind=MetricKind.CATEGORICAL))
        run_engine(eng, st, t, dt=H)
    for e, lo, hi in ((E, t_inc, t_inc + 2 * H), (e2, reg_t, reg_t + 3 * H)):
        el = model(st, e)["ref_elig"]
        assert el, e
        for ts, ok in el.items():
            near = (ts >= lo - 86400.0) and (ts <= hi + 86400.0)
            assert ok == (not near), (e, ts)
        assert any(el.values()) and not all(el.values())
    # the same data without incident / regime admits every row
    st2, eng2 = make_store(), BaselineEngine()
    _run(st2, eng2, 4 * 24, dt=H, rng=np.random.default_rng(23))
    assert all(model(st2)["ref_elig"].values())
    assert MB.n_eff(model(st2)["reference"]) > MB.n_eff(model(st)["reference"])


# ------------------------------------------------------- checkpoints / link
def test_checkpoints_share_untouched_blocks_and_round_trip():
    rng = np.random.default_rng(26)
    st, eng = make_store(), BaselineEngine()
    _run(st, eng, 6 * 4, rng=rng)                            # 6 h at 900 s
    times = st.checkpoint_times(S, E, LEARNER_CUR)
    assert len(times) >= 3
    (_, b1), (_, b2) = (st.get_checkpoint(S, E, LEARNER_CUR, times[-2]),
                        st.get_checkpoint(S, E, LEARNER_CUR, times[-1]))
    shared = sum(x is y for x, y in zip(b1["b48"], b2["b48"]))
    assert 30 <= shared < 48                                 # ~1 h of commits touches few buckets
    assert copy.deepcopy(b2) is b2
    a = MB.load(b2)                           # rows queued at checkpoint time travel along
    MB.flush_many([a])
    assert a.T == times[-1] and not a.pending
    cur = MB.load(MB.dump(model(st)["current"]))
    assert np.allclose(MB.true_stats(cur), MB.true_stats(model(st)["current"]), rtol=1e-6)


def test_link_seeding_adds_half_of_the_linked_entity():
    rng = np.random.default_rng(27)
    st, eng = make_store(), BaselineEngine()
    a_ent, b_ent = "10.0.0.5", "10.0.0.6"
    t = _run(st, eng, 60, e_list=(a_ent, b_ent), rng=rng,
             rate=lambda i, t: 10.0)
    A = MB.true_stats(model(st, a_ent)["current"])
    Bm = MB.true_stats(model(st, b_ent)["current"])
    put_model(st, S, "__system__", "model.link",
              {"version": 1, "links": [{"from": a_ent, "to": b_ent, "ts": t}]})
    t += DT
    for e in (a_ent, b_ent):
        feed(st, t, nat_row(rng), e=e)
    run_engine(eng, st, t)
    B2 = MB.true_stats(model(st, b_ent)["current"])
    w = MB.FULL.W[REQ]
    gain = B2[:, w].sum() - Bm[:, w].sum()
    # B := B_own + 0.5 A (plus this tick's own committed row, spread weight <= 5)
    assert 0.5 * A[:, w].sum() - 1e-6 <= gain <= 0.5 * A[:, w].sum() + 5.0
    assert model(st, b_ent)["gate"].link_version == 1


# ------------------------------------------------------- pure API (B18)
def test_pure_api_commit_many_equals_sequential_commits():
    rng = np.random.default_rng(28)
    a1, a2 = MB.new_anchor(), MB.new_anchor()
    s1, s2 = MB.new_anchor(), MB.new_anchor()
    for i in range(150):
        t = T0 + i * DT
        r1 = MB.make_row(t, nat_row(rng, 10.0), DT, tctx(t))
        r2 = MB.make_row(t, nat_row(rng, 30.0), DT, tctx(t))
        MB.commit_many([a1, a2], [r1, r2], [1.0, 0.5], [MB.CAP_CURRENT] * 2, [0.0, 0.0])
        MB.commit(s1, r1, 1.0)
        MB.commit(s2, r2, 0.5)
    for a, b in ((a1, s1), (a2, s2)):
        assert np.allclose(MB.true_stats(a), MB.true_stats(b), rtol=1e-12, atol=1e-12)
        assert np.allclose(MB.true_cells(a), MB.true_cells(b), rtol=1e-12, atol=1e-12)
    tc = tctx(T0 + 149 * DT)
    p1 = MB.anchor_predictive(a1, tc)
    assert p1.mu[REQ] == pytest.approx(10.0, rel=0.1)
    # a parent tier as prior (B18: class aggregate under its system)
    E = MB.true_stats(a2)[tc["bin48"]]
    pb = MB.anchor_predictive(MB.new_anchor(), tc, parent=(E, np.ones(F.FEATURE_DIM),
                                                           np.full(F.FEATURE_DIM, 10.0)))
    assert pb.mu[REQ] == pytest.approx(MB.anchor_predictive(a2, tc).mu[REQ], rel=1e-6)
    pair = MB.new_model()
    assert pair["current"].a168 is not None and pair["reference"].a168 is None
    # queued rows are folded by flush_many and survive a checkpoint blob
    q = MB.new_anchor()
    for i in range(5):
        t = T0 + i * DT
        MB.queue(q, MB.make_row(t, nat_row(rng), DT, tctx(t)), 1.0)
    blob = MB.dump(q)
    q2 = MB.load(blob)
    MB.flush_many([q, q2])
    assert q.n_commit == q2.n_commit == 5 and not q.pending
    assert np.allclose(MB.true_stats(q), MB.true_stats(q2))


def test_eb_kappa_and_tiers():
    rng = np.random.default_rng(29)
    st, eng = make_store(), BaselineEngine()
    ents = [f"10.0.3.{i}" for i in range(5)]
    rates = {e: 5.0 + 5.0 * i for i, e in enumerate(ents)}
    t = T0
    for i in range(100):
        t = T0 + i * DT
        for e in ents:
            feed(st, t, nat_row(rng, rates[e]), e=e)
        run_engine(eng, st, t)
    sysm = MB.tier_model(st, S, "__system__")
    k = sysm["kappa"]
    assert np.all((k >= MB.KAPPA_MIN) & (k <= MB.KAPPA_MAX))
    assert k[REQ] < 10.0                                    # members differ a lot: weak pooling
    assert sorted(sysm["members"]) == sorted(ents)
    orgm = MB.tier_model(st, S, "org")
    assert np.allclose(orgm["stats"], sysm["stats"])
    # members keep their order and stay near their own rate: kappa = 2 rows
    # (30 min of pseudo-exposure) against about a day of own data
    tc = tctx(t)
    mus = [MB.predictive(st, S, e, tc).mu[REQ] for e in ents]
    assert all(a < b for a, b in zip(mus, mus[1:]))
    for e, mu in zip(ents, mus):
        assert mu == pytest.approx(rates[e], rel=0.4)
    cls = MB.predictive(st, S, ents[0], tc, tier="class")   # leave-one-out system tier
    assert cls.mu[REQ] > rates[ents[0]] + 3.0


# -------------------------------------------------------------------- perf
def test_perf_40_entities():
    rng = np.random.default_rng(30)
    st, eng = make_store(), BaselineEngine()
    ents = [f"10.1.0.{i}" for i in range(40)]
    ticks = []
    for i in range(24 + 48):
        t = T0 + i * DT
        for e in ents:
            feed(st, t, nat_row(rng), e=e)
        a = time.perf_counter()
        run_engine(eng, st, t)
        if i >= 24:
            ticks.append(time.perf_counter() - a)
    # spec target <= 7 ms / tick at 40 entities; generous bound for CI noise
    # (hourly ticks also fold the reference batch, checkpoints, tiers, profiles)
    assert float(np.mean(ticks)) < 0.06
    assert all(model(st, e)["current"].n_commit > 0 for e in ents)
