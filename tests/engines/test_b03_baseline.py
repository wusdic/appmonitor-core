"""B03 BaselineEngine (docs/lib3/engines.md '## B03'): spec unit tests (a)-(f)
plus edge cases. A tiny B01 + B28 stand-in (`feed`) writes feature.nat /
feature.active and the governance rings (behavior.trust / trust_prov /
quarantine) exactly as those engines do; tctx comes from the same
timebins function B01 uses.

Spec test mapping:
  (a) test_a_untrusted_attack_rows_leave_current_mean
  (b) test_b_step_moves_current_and_reference_within_their_caps
  (c) test_c_empty_store_cold_start_predictive   (known correction: p > 1e-3)
  (d) test_d_commits_lag_now_by_exactly_D
  (e) test_e_rollback_equals_offline_fit_and_holds_attack_rows
  (f) test_f_new_entity_returns_class_predictive_mean
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from helpers import DT, T0, make_store, make_tctx, put_model, run_engine, set_trust

from app.engines.behavior.baseline import BaselineEngine
from app.engines.behavior.lib import bayes
from app.engines.behavior.lib import features as F
from app.engines.behavior.lib import gating as G
from app.engines.behavior.lib import m_baseline as MB
from app.engines.behavior.lib import timebins as TB
from app.core.engine import default_config

S = "erp"
E = "10.0.0.1"
IDX = F.FEATURE_INDEX
REQ = IDX["http_requests"]
FLOWS = IDX["flows"]
WRITE = IDX["http_write_ratio"]
BYTES = IDX["bytes_up"]
LAT = IDX["http_latency"]
COUNT_IDX = [i for i, n in enumerate(F.FEATURE_NAMES_V2) if F.FEATURE_KIND[n] == "count"]
CFG = default_config({"strict": True})


def nat_row(rng, req_per_min=10.0, dt=DT, write=0.1, bytes_=1e5, lat=100.0):
    """feature.nat as B01 writes it for an HTTP client: counts are raw per
    tick (stale -> 0), ratios k/n, averages in natural units, NaN = absent."""
    x = np.full(F.FEATURE_DIM, np.nan)
    x[COUNT_IDX] = 0.0
    n = float(rng.poisson(req_per_min * dt / 60.0))
    x[REQ] = n
    x[FLOWS] = float(rng.poisson(0.2 * req_per_min * dt / 60.0))
    if n > 0:
        x[WRITE] = rng.binomial(int(n), write) / n
        x[LAT] = lat * rng.lognormal(0.0, 0.2)
    x[BYTES] = bytes_ * rng.lognormal(0.0, 0.2)
    return x


def feed(store, t, nat, e=E, s=S, dt=DT, active=1.0, trust=1.0, quarantine=0.0, gov=True):
    store.add_vec(s, e, "feature.nat", t, np.asarray(nat, dtype=np.float32), window_s=int(dt))
    store.add_vec(s, e, "feature.active", t, np.array([active], dtype=np.float32), window_s=int(dt))
    store.register_entity(s, e)
    if gov:
        set_trust(store, s, e, [t], trust, quarantine=quarantine)


def tctx(t, dt=DT):
    return TB.tctx_from_config(t, CFG, dt)


def model(store, e=E):
    return store.get_model(S, e, MB.MODEL)


def D_ticks(dt):
    return G.commit_delay_ticks(dt, 600.0)


# ------------------------------------------------------------------ (a)
def test_a_untrusted_attack_rows_leave_current_mean():
    rng = np.random.default_rng(1)
    st, eng = make_store(), BaselineEngine()
    D = D_ticks(DT)
    for i in range(240):                                    # then Poisson(300 / min), trust 0
        t = T0 + i * DT
        feed(st, t, nat_row(rng, 10.0 if i < 200 else 300.0), trust=1.0 if i < 200 else 0.0)
        run_engine(eng, st, t)
        if i == 200 + D - 1:                                # the last clean row is committed
            m = model(st)
            assert MB.last_commit_ts(m) == T0 + 199 * DT
            neff_before = MB.n_eff(m)
            mean_before, _ = MB.bucket_means(m["current"])
    m = model(st)
    # every row of the attack window that is due is committed with weight 0
    assert MB.last_commit_ts(m) == t - D * DT
    assert MB.n_eff(m) <= neff_before
    pr = MB.predictive(st, S, E, tctx(t), model=m)
    assert pr.mu[REQ] == pytest.approx(10.0, rel=0.05)
    mean_after, _ = MB.bucket_means(m["current"])
    used = np.isfinite(mean_before[:, REQ]) & (MB.true_stats(m["current"])[:, MB.FULL.W[REQ]] > 8)
    assert used.sum() >= 20
    assert np.allclose(mean_after[used, REQ], 10.0, rtol=0.05)
    assert np.all(np.abs(mean_after[used, REQ] - mean_before[used, REQ]) < 1e-9)


# ------------------------------------------------------------------ (b)
def _band_snapshots(series, ts, hour_offset):
    """Per bin48 bucket H: values at the ticks closest to local H + offset
    (between the bucket's daily clusters of commits, 12 h from its band edge)."""
    local = np.array([tctx(t)["hour_local"] for t in ts])
    out = {}
    for H in range(24):
        target = (H + hour_offset) % 24
        d = np.abs(((local - target + 12) % 24) - 12)
        pick = np.flatnonzero(d < 0.13)
        out[H] = pick
    return out


def _data_rate(anchor):
    """Per-minute request rate of each bin48 bucket from the data alone
    (sum x / sum e), the mean the rate cap bounds: decay scales both sums
    alike, so only commits move it."""
    St = MB.true_stats(anchor)
    if St is None:
        return np.full(48, np.nan)
    with np.errstate(all="ignore"):
        return St[:, MB.FULL.NUM[REQ]] / St[:, MB.FULL.DEN[REQ]]


def test_b_step_moves_current_and_reference_within_their_caps():
    rng = np.random.default_rng(2)
    st, eng = make_store(), BaselineEngine()
    n_mat, n_step = 5 * 96, 3 * 96
    ts, cur_mean, cur_sd, ref_mean, ref_sd = [], [], [], [], []
    for i in range(n_mat + n_step):
        t = T0 + i * DT
        feed(st, t, nat_row(rng, 10.0 if i < n_mat else 100.0))
        run_engine(eng, st, t)
        m = model(st)
        _, b = MB.bucket_means(m["current"])
        _, d = MB.bucket_means(m["reference"])
        ts.append(t)
        cur_mean.append(_data_rate(m["current"]))
        cur_sd.append(b[:, REQ])
        ref_mean.append(_data_rate(m["reference"]))
        ref_sd.append(d[:, REQ])
    cur_mean, cur_sd = np.array(cur_mean), np.array(cur_sd)
    ref_mean, ref_sd = np.array(ref_mean), np.array(ref_sd)
    nwd = np.array([tctx(t)["day_type"] == "nonworkday" for t in ts])
    snaps = _band_snapshots(cur_mean, ts, 12.5)
    checked = 0
    for H, pick in snaps.items():
        pick = [p for p in pick if p >= n_mat - 96]           # from a day before the step
        b = H + (24 if nwd[pick[0]] else 0) if pick else H
        for p0, p1 in zip(pick[:-1], pick[1:]):
            if nwd[p0] != nwd[p1]:
                continue
            bb = H + (24 if nwd[p0] else 0)
            s_max = np.nanmax(cur_sd[p0:p1 + 1, bb])
            assert abs(cur_mean[p1, bb] - cur_mean[p0, bb]) <= 0.1 * s_max * (1 + 1e-3)
            r_max = np.nanmax(ref_sd[p0:p1 + 1, bb])
            assert abs(ref_mean[p1, bb] - ref_mean[p0, bb]) <= 0.03 * r_max * (1 + 1e-3)
            checked += 1
    assert checked >= 24
    # without the cap one step day would move a mature mean ~9x; with it the
    # anchors stay near the old level
    m = model(st)
    mature = MB.true_stats(m["current"])[:, MB.FULL.W[REQ]] > 20
    assert np.nanmax(cur_mean[-1][mature]) < 13.0
    assert np.nanmax(ref_mean[-1][mature]) < 11.5
    assert np.nanmin(cur_mean[-1][mature]) > 9.0


# ------------------------------------------------------------------ (c)
def test_c_empty_store_cold_start_predictive():
    st = make_store()
    tc = make_tctx(T0)
    x = np.full(F.FEATURE_DIM, np.nan)
    x[REQ] = 150.0                                            # 10 req/min over 15 min
    pr = MB.predictive(st, S, E, tc)
    u, p = MB.midp(pr, x, DT)
    # known correction (priors.COLD_START_NOTE): the Gamma(0.5) hyperprior
    # gives two-sided p ~ 3.4e-3; the spec's 0.05 is not reachable with a0 = 0.5
    assert p[REQ] > 1e-3
    assert p[REQ] == pytest.approx(3.4e-3, rel=0.1)
    assert math.isfinite(u[REQ])
    # every family gives a finite predictive; missing values are NaN, never p = 1
    assert np.all(np.isfinite(pr.mean))
    assert np.isnan(p[WRITE]) and np.isnan(p[LAT])
    q = MB.quantiles(pr, [0.05, 0.5, 0.95], DT)
    assert np.all(np.isfinite(q[:, REQ])) and q[0, REQ] <= q[1, REQ] <= q[2, REQ]
    # the engine on a fresh entity publishes the same hyperprior predictive
    eng = BaselineEngine()
    feed(st, T0, x)
    run_engine(eng, st, T0)
    pr2 = MB.predictive(st, S, E, tc)
    assert MB.midp(pr2, x, DT)[1][REQ] > 1e-3
    assert np.allclose(pr2.mean, pr.mean, equal_nan=True)


# ------------------------------------------------------------------ (d)
@pytest.mark.parametrize("dt", [900.0, 60.0, 3600.0])
def test_d_commits_lag_now_by_exactly_D(dt):
    rng = np.random.default_rng(3)
    st, eng = make_store(), BaselineEngine()
    D = D_ticks(dt)
    for i in range(D + 12):
        t = T0 + i * dt
        feed(st, t, nat_row(rng, dt=dt), dt=dt)
        run_engine(eng, st, t, dt=dt)
        m = model(st)
        if i >= D:
            assert MB.last_commit_ts(m) == t - D * dt
            assert m["gate"].last_ts == t - D * dt
        else:
            assert math.isnan(MB.last_commit_ts(m))
    assert m["current"].n_commit == 12


def test_d_cadence_switch_900_to_60_keeps_rate_and_lag():
    rng = np.random.default_rng(4)
    st, eng = make_store(), BaselineEngine()
    t = T0
    for i in range(2 * 96):
        t = T0 + i * DT
        feed(st, t, nat_row(rng, 10.0, dt=DT))
        run_engine(eng, st, t, dt=DT)
    t_switch = t
    for j in range(1, 8 * 60):
        t = t_switch + j * 60.0
        feed(st, t, nat_row(rng, 10.0, dt=60.0), dt=60.0)
        run_engine(eng, st, t, dt=60.0)
        if j > 10:
            assert MB.last_commit_ts(model(st)) == t - 10 * 60.0
    m = model(st)
    pr = MB.predictive(st, S, E, tctx(t), model=m)
    assert pr.mu[REQ] == pytest.approx(10.0, rel=0.08)      # per-minute rate, not per-tick
    # 60-s rows are folded with 1-minute exposure (the gap to the previous tick)
    means, _ = MB.bucket_means(m["current"])
    W = MB.true_stats(m["current"])[:, MB.FULL.W[REQ]]
    assert np.allclose(means[W > 8, REQ], 10.0, rtol=0.1)


# ------------------------------------------------------------------ (e)
def test_e_rollback_equals_offline_fit_and_holds_attack_rows():
    rng = np.random.default_rng(5)
    st, eng = make_store(), BaselineEngine()
    D = D_ticks(DT)
    rows = []
    for i in range(120):
        t = T0 + (i + 1) * DT
        nat = nat_row(rng, 10.0) if i < 100 else nat_row(rng, 60.0, bytes_=2e6, write=0.6)
        rows.append((t, nat.astype(np.float32).astype(np.float64)))
        feed(st, t, nat)
        run_engine(eng, st, t)
    t100, t_last = rows[99][0], rows[-1][0]
    t = t_last
    for k in range(1, D + 1):                                # all 120 rows committed; the
        t = t_last + k * DT                                  # governor quarantines at the
        feed(st, t, nat_row(rng, 10.0), quarantine=1.0 if k == D else 0.0)   # last tick
        run_engine(eng, st, t)
    assert MB.last_commit_ts(model(st)) == t_last
    # governor: rollback to t100 (quarantine(t - 1) = 1 holds everything after)
    t += DT
    feed(st, t, nat_row(rng, 10.0), quarantine=1.0)
    put_model(st, S, E, "model.control", {"version": 0, "rollback_to": t100})
    run_engine(eng, st, t)
    m = model(st)
    # offline fit on rows <= t100 with their recorded weights (trust 1)
    off = MB.new_anchor(week=True, select=True)
    for ts, nat in rows[:100]:
        MB.commit(off, MB.make_row(ts, nat, DT, tctx(ts)), 1.0)
    cur = m["current"]
    assert not cur.pending
    a, b = MB.true_stats(cur), MB.true_stats(off)
    scale = np.abs(b).max(axis=0) + 1e-12
    assert np.all(np.abs(a - b) <= 1e-6 * scale)
    assert np.allclose(MB.true_cells(cur), MB.true_cells(off), rtol=1e-6, atol=1e-9)
    assert cur.T == off.T and cur.n_commit == off.n_commit
    assert np.array_equal(cur.hl, off.hl)
    held = set(MB.held(m))
    assert {ts for ts, _ in rows[100:]} <= held
    assert MB.last_commit_ts(m) == t100
    assert m["gate"].applied["_last_rollback"]["complete"]


# ------------------------------------------------------------------ (f)
def _class_model(members, rid="r1", s=S):
    assign = {f"{s}|{e}": {"role": rid, "sub": f"{rid}.0", "prob": 0.9, "static": [], "pool": None,
                           "super": "machine"} for e in members}
    return {"assign": assign, "roles": {rid: {"name": "erp-clients", "members": list(assign)}},
            "subs": {}, "statics": {}, "pools": {}, "version": 1}


def test_f_new_entity_returns_class_predictive_mean():
    rng = np.random.default_rng(6)
    st, eng = make_store(), BaselineEngine()
    members = [f"10.0.1.{i}" for i in range(4)]
    new = "10.0.1.99"
    put_model(st, "__org__", "__org__", "model.class", _class_model(members + [new]))
    t = T0
    for i in range(150):
        t = T0 + i * DT
        for j, e in enumerate(members):
            feed(st, t, nat_row(rng, 8.0 + 2 * j, write=0.05 + 0.03 * j), e=e)
        run_engine(eng, st, t)
    st.register_entity(S, new)                               # known, never committed
    run_engine(eng, st, t + DT)
    m_new = model(st, new)
    assert m_new is not None and m_new["current"].empty
    tm = MB.tier_model(st, S, "class:r1")
    assert tm is not None and tm["tier"] == "class" and sorted(tm["members"]) == sorted(members + [new])
    tc = tctx(t + DT)
    p_new = MB.predictive(st, S, new, tc)
    p_cls = MB.predictive(st, S, "class:r1", tc)
    assert np.allclose(p_new.mean, p_cls.mean, rtol=1e-9, atol=1e-12)
    assert 8.0 < p_new.mu[REQ] < 14.0
    # the class tier sits in the system tier, which sits in the org tier
    assert MB.tier_model(st, S, "__system__")["tier"] == "system"
    assert MB.tier_model(st, S, "org")["tier"] == "org"
    # a member's predictive is its own data with the class as a leave-one-out prior
    p_m0 = MB.predictive(st, S, members[0], tc)
    assert p_m0.mu[REQ] < p_cls.mu[REQ]


# --------------------------------------------------------------- edges
def test_empty_store_and_silent_entity():
    st, eng = make_store(), BaselineEngine()
    assert run_engine(eng, st, T0) == 0
    rng = np.random.default_rng(7)
    for i in range(20):                                      # absent: active = 0 rows
        t = T0 + i * DT
        feed(st, t, nat_row(rng), active=0.0)
        run_engine(eng, st, t)
    m = model(st)
    assert m["current"].empty and MB.n_eff(m) == 0.0
    assert m["gate"].last_ts == t - D_ticks(DT) * DT            # cursor advanced
    assert not m["gate"].journal
    pr = MB.predictive(st, S, E, tctx(t))
    assert np.all(np.isfinite(pr.mean))


def test_nan_inputs_are_unobserved_not_zero():
    st, eng = make_store(), BaselineEngine()
    rng = np.random.default_rng(8)
    for i in range(12):
        t = T0 + i * DT
        x = np.full(F.FEATURE_DIM, np.nan)                  # degraded B01 row
        if i % 2:
            x = nat_row(rng)
            x[WRITE] = np.nan                               # n = 0 -> ratio unobserved
            x[LAT] = np.nan
        feed(st, t, x)
        run_engine(eng, st, t)
    m = model(st)
    St = MB.true_stats(m["current"])
    assert St[:, MB.FULL.W[WRITE]].sum() == 0.0 and St[:, MB.FULL.W[LAT]].sum() == 0.0
    assert St[:, MB.FULL.W[REQ]].sum() > 0.0
    assert np.all(np.isfinite(St))
    # observe(): NaN never becomes a count of 0 or a p of 1
    num, den, wf = MB.observe(np.full(F.FEATURE_DIM, np.nan), DT)
    assert not wf.any()
    u, p = MB.midp(MB.predictive(st, S, E, tctx(t)), np.full(F.FEATURE_DIM, np.nan), DT)
    assert np.all(np.isnan(p))


def test_training_mode_learns_without_trust_and_emits_nothing():
    rng = np.random.default_rng(9)
    for training in (True, False):
        st, eng = make_store(), BaselineEngine()
        for i in range(30):
            t = T0 + i * DT
            feed(st, t, nat_row(rng), gov=False)            # no governor rings at all
            run_engine(eng, st, t, training=training)
        m = model(st)
        assert len(m["gate"].journal) == 30 - D_ticks(DT)
        assert (MB.n_eff(m) > 0) == training                # missing trust: 1 training, 0 live
        assert st.events() == []


def test_quarantine_holds_and_release_commits():
    rng = np.random.default_rng(10)
    st, eng = make_store(), BaselineEngine()
    D = D_ticks(DT)
    for i in range(40):
        t = T0 + i * DT
        feed(st, t, nat_row(rng), quarantine=1.0 if 20 <= i < 30 else 0.0)
        run_engine(eng, st, t)
    m = model(st)
    held = MB.held(m)
    assert len(held) >= 8
    n_before = m["current"].n_commit
    put_model(st, S, E, "model.control", {"version": 0, "release": [min(held), max(held)]})
    t += DT
    feed(st, t, nat_row(rng))
    run_engine(eng, st, t)
    m = model(st)
    assert not MB.held(m)
    assert m["current"].n_commit == n_before + len(held) + 1


def test_profile_legacy_fields_and_maturity():
    rng = np.random.default_rng(11)
    st, eng = make_store(), BaselineEngine()
    for i in range(120):
        t = T0 + i * DT
        feed(st, t, nat_row(rng))
        run_engine(eng, st, t)
    prof = st.profile(S, E)
    assert len(prof.baseline_median) == F.FEATURE_DIM == len(prof.baseline_mad)
    assert prof.baseline_median[REQ] == pytest.approx(math.log1p(10.0), abs=0.1)
    assert set(prof.seasonal) >= {"http_requests", "bytes_up", "flows", "dns_queries"}
    assert all(len(v) == 24 for v in prof.seasonal.values())
    mat = prof.extra["maturity"]
    assert mat["stage"] in ("cold", "warming", "mature") and mat["n_eff"] > 48
    assert mat["mode"] == "bin48" and mat["golden"] is False
    assert model(st)["n_eff"] == pytest.approx(MB.n_eff(model(st)))
    # the batched profile path equals predictive() + vec_median_sd()
    tc = tctx(t)
    med, sd = MB.profile_many(st, S, [E], [model(st)], tc)
    m1, s1 = MB.vec_median_sd(MB.predictive(st, S, E, tc))
    assert np.allclose(med[0], m1) and np.allclose(sd[0], s1)


def test_accessors_quantiles_loglik_and_set():
    rng = np.random.default_rng(12)
    st, eng = make_store(), BaselineEngine()
    for i in range(150):
        t = T0 + i * DT
        feed(st, t, nat_row(rng))
        run_engine(eng, st, t)
    tc = tctx(t)
    ps = MB.predictive_set(st, S, E, tc)
    assert set(ps) == {"current", "reference", "class"}
    assert np.allclose(ps["current"].mean, MB.predictive(st, S, E, tc).mean)
    assert np.allclose(ps["reference"].mean, MB.predictive(st, S, E, tc, anchor="reference").mean)
    q = MB.quantiles(ps["current"], [0.05, 0.5, 0.95], DT)
    for f in (REQ, WRITE, BYTES, LAT):
        assert q[0, f] <= q[1, f] <= q[2, f]
    assert 100 < q[1, REQ] < 200 and 0.03 < q[1, WRITE] < 0.2 and 5e4 < q[1, BYTES] < 2e5
    x = nat_row(rng)
    ll = MB.loglik(ps["current"], x, DT)
    assert np.isfinite(ll[[REQ, WRITE, BYTES, LAT]]).all()
    u, p = MB.midp(ps["current"], x, DT)
    assert np.all((p[[REQ, BYTES, LAT]] > 1e-4) & (p[[REQ, BYTES, LAT]] <= 1.0))
    x[REQ] = 3.0                                             # far below 150 per 15 min
    assert MB.midp(ps["current"], x, DT)[1][REQ] < 1e-20
    mn = MB.mean_nat(ps["current"], DT)
    assert mn[REQ] == pytest.approx(150.0, rel=0.1) and mn[BYTES] == pytest.approx(1e5, rel=0.3)
    s15 = MB.sd15(ps["current"])
    assert np.all(s15[[REQ, WRITE, BYTES]] > 0)
    d = MB.descriptors(model(st))
    assert len(d["features"]["http_requests"]["workday"]) == 24
    assert MB.golden_offset(st, S, E) is None
    assert set(MB.hl_days(model(st))) <= set(MB.HL_CAND_DAYS)


def test_nb_and_bb_predictives_agree_with_bayes():
    """Predictive parameters feed lib/bayes exactly: the count mid-p equals
    bayes.nb_midp at mean mu * e and size r; the ratio uses BB(p c, (1 - p) c)."""
    rng = np.random.default_rng(13)
    st, eng = make_store(), BaselineEngine()
    for i in range(100):
        t = T0 + i * DT
        feed(st, t, nat_row(rng))
        run_engine(eng, st, t)
    pr = MB.predictive(st, S, E, tctx(t))
    x = nat_row(rng)
    u, p = MB.midp(pr, x, DT)
    uu, pp = bayes.nb_midp(float(x[REQ]), float(pr.mu[REQ] * DT / 60.0), float(pr.r[REQ]))
    assert p[REQ] == pytest.approx(pp) and u[REQ] == pytest.approx(uu)
    k = round(x[WRITE] * x[REQ])
    uu, pp = bayes.bb_midp(float(k), float(x[REQ]), pr.p[WRITE] * pr.c[WRITE],
                           (1 - pr.p[WRITE]) * pr.c[WRITE])
    assert p[WRITE] == pytest.approx(pp)
