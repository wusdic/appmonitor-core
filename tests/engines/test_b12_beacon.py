"""B12 BeaconEngine (engines/behavior/beacon.py).

Spec unit tests (docs/lib3/engines.md, B12), with the helper-review
corrections (sub-second times from act.rare_events; lib/evt renewal LRT, exact
null table and window-corrected Z^2):
  (a) one request every 300 s +- 30 % for 20 events to a rare destination on
      top of 22 normal requests per tick: p < 1e-6 (end to end through R2);
  (b) a health check to svc.corp.local shared by 3 class members: no event;
  (c) 1000 Poisson destination streams: < 0.5 % have p < 1e-3 (engine p and
      the exact kappa = 1 table on its own);
  (d) log-normal human gaps, sigma = 1: < 0.5 % have p < 1e-4.
Plus: the saddlepoint against the exact table, learned class sharing at
P_hat +- 10 %, allowlist, cooldown, empty store, silent entity, NaN inputs,
R2 failure, training mode, cadence 900 -> 60, gating (frozen / quarantine),
buffer bounds and a perf bound.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from helpers import T0, add_obs_tick, make_store, obs, put_model, run_engine, set_trust

from app.engines.behavior import beacon as B
from app.engines.behavior.lib import emit, evt
from app.engines.behavior.lib import m_template as MT
from app.engines.raw.action_token import ActionTokenEngine

S = "erp"
DT = 900.0
DID = 424242
FILLERS = [f"10.0.9.{i}" for i in range(1, 10)]


# ------------------------------------------------------------------ helpers
def _beacon_times(rng, t0, n, period=300.0, jitter=0.3):
    return t0 + np.cumsum(period * (1.0 + rng.uniform(-jitter, jitter, n)))


def _feed(st, e, now, rows_by_dest, events=None):
    """Write one tick of act.rare_events (+ act.events) as R2 would."""
    rare = {int(d): [[float(t), float(u), float(v)] for t, u, v in rows]
            for d, rows in rows_by_dest.items() if len(rows)}
    m = {"act.events": float(events if events is not None
                             else sum(len(r) for r in rows_by_dest.values()))}
    if rare:
        m["act.rare_events"] = rare
    add_obs_tick(st, S, e, now, m)


def _fillers(st, now, ents=FILLERS):
    for f in ents:
        add_obs_tick(st, S, f, now, {"act.events": 5.0})


def _drive(st, eng, e, times, sizes=None, t_start=T0, dt=DT, ticks=None, training=False,
           did=DID, extra=None, fillers=True, config=None):
    """Feed event `times` for entity e tick by tick (events in (now - dt, now])."""
    times = np.asarray(times, dtype=np.float64)
    sizes = np.full(times.size, 1500.0) if sizes is None else np.asarray(sizes, float)
    n_ticks = ticks if ticks is not None else int(math.ceil((times.max() - t_start) / dt)) + 1
    nows = []
    for k in range(1, n_ticks + 1):
        now = t_start + k * dt
        sel = (times > now - dt) & (times <= now)
        rows = [(t, z * 0.3, z * 0.7) for t, z in zip(times[sel], sizes[sel])]
        _feed(st, e, now, {did: rows}, events=max(1, len(rows)))
        if extra:
            extra(st, now)
        if fillers:
            _fillers(st, now)
        run_engine(eng, st, now, training=training, dt=dt, config=config)
        nows.append(now)
    return nows


def _score(st, e, ts):
    return emit.read_row(st, S, e, emit.SCORE, ts).get("beacon", math.nan)


def _beacon_events(st, e=None):
    return [ev for ev in st.events(S, e, limit=1000) if ev.kind == "beacon"]


def _class_model(members, rid="r1", extra_assign=None):
    assign = {f"{S}|{ip}": {"role": rid, "prob": 1.0} for ip in members}
    assign.update(extra_assign or {})
    return {"assign": assign,
            "roles": {rid: {"name": "ops", "members": [f"{S}|{m}" for m in members]}},
            "version": 1}


def _pair(st, e, did=DID):
    m = st.get_model(S, e, B.MODEL)
    return None if m is None else m["pairs"].get(did)


# ------------------------------------------------------------- spec (a)
def _http(e, ts, path="/orders/list", host="erp.corp"):
    return obs(S, e, ts, app_proto="http", http_method="GET", http_host=host, http_path=path,
               http_status=200, peer="10.9.0.10", dst_port=80, bytes_up=300, bytes_down=5000)


def _tls(e, ts, sni, up=420, down=1300):
    return obs(S, e, ts, app_proto="tls", tls_sni=sni, dst_port=443, peer="203.0.113.7",
               bytes_up=up, bytes_down=down)


def test_a_jittered_beacon_through_r2():
    rng = np.random.default_rng(11)
    st, r2, eng = make_store(), ActionTokenEngine(), B.BeaconEngine()
    ents = [f"10.0.0.{i}" for i in range(1, 11)]
    victim = ents[0]
    bt = _beacon_times(rng, T0 + 50.0, 20)                  # 20 events, sub-second times
    did = MT.dest_id_of(MT.dest_name_of("c2.evil-beacon.net"))
    alarms = []
    for k in range(1, 13):
        now = T0 + k * DT
        batch = []
        for e in ents:                                     # 22 normal requests per tick each
            batch += [_http(e, now - DT + 30.0 + 37.3 * i) for i in range(22)]
        sel = bt[(bt > now - DT) & (bt <= now)]
        batch += [_tls(victim, float(t), "c2.evil-beacon.net") for t in sel]
        run_engine(r2, st, now, dt=DT, observations=batch)
        run_engine(eng, st, now, dt=DT)
        acc = emit.read_dict(st, S, victim, emit.ACC_ALARM, now)
        alarms.append(acc.get("beacon"))
    pr = _pair(st, victim, did)
    assert pr is not None and pr["t"].size == 20
    assert np.all(np.diff(pr["t"]) > 0) and not np.all(pr["t"] == np.round(pr["t"]))
    res = pr["res"]
    assert res["n"] == 20
    assert res["p"] < 1e-6
    assert res["p_renewal"] < 1e-6 and res["kappa"] > 5.0
    assert abs(res["period"] - 300.0) < 60.0
    evs = _beacon_events(st, victim)
    assert len(evs) == 1                                  # alerted once (24 h cooldown)
    ev = evs[0]
    assert ev.axes == ["c2"] and ev.extra["dest_id"] == did
    assert ev.extra["value"] == "evil-beacon.net" and ev.extra["dim"] == "dest"
    assert ev.p_value < 1e-6 and ev.dedupe_key == f"beacon|{S}|{victim}|{did}"
    assert 1 in alarms
    # the other entities never saw a rare destination: scored 0, never alarmed
    for e in ents[1:]:
        assert _score(st, e, T0 + 12 * DT) == 0.0
        assert not _beacon_events(st, e)
    prof = st.profile(S, victim).extra["beacons"]
    assert prof["pairs"][0]["dest"] == "evil-beacon.net" and prof["pairs"][0]["p"] < 1e-6


def test_a_twenty_events_power():
    """The p < 1e-6 of (a) holds for (nearly) every jitter draw, not just one."""
    rng = np.random.default_rng(5)
    ps = np.array([B.evaluate_pair(_beacon_times(rng, 0.0, 20), np.full(20, np.nan),
                                   np.zeros(0), 0.5)["p"] for _ in range(300)])
    assert np.median(ps) < 1e-8
    assert np.mean(ps < 1e-6) >= 0.99


# ------------------------------------------------------------- spec (b)
def test_b_class_shared_health_check_no_event():
    st, eng = make_store(), B.BeaconEngine()
    members = ["10.0.1.1", "10.0.1.2", "10.0.1.3"]
    put_model(st, "__org__", "__org__", "model.class", _class_model(members))
    did = MT.dest_id_of("svc.corp.local")
    for k in range(1, 13):
        now = T0 + k * DT
        for j, m in enumerate(members):                    # strict 60-s health check
            ts = np.arange(now - DT + 1.0 + 7.0 * j, now + 1e-9, 60.0)
            _feed(st, m, now, {did: [(t, 200.0, 400.0) for t in ts]})
        _fillers(st, now)
        run_engine(eng, st, now, dt=DT)
        for m in members:
            assert emit.read_dict(st, S, m, emit.ACC_ALARM, now).get("beacon", 0) == 0
    assert not _beacon_events(st)
    for m in members:                                     # class-shared: never buffered
        assert _pair(st, m, did) is None


def test_b_prevalence_gate_without_class():
    """The same strict health check on 3 unrelated entities: scored (rare,
    not class-shared) but never alerted, because 3 entities use it (> 2)."""
    st, eng = make_store(), B.BeaconEngine()
    users = ["10.0.2.1", "10.0.2.2", "10.0.2.3"]
    fill = [f"10.0.9.{i}" for i in range(1, 16)]           # 3 of 18 entities: 17 % <= 20 %
    did = MT.dest_id_of("svc.corp.local")
    for k in range(1, 13):
        now = T0 + k * DT
        for j, m in enumerate(users):
            ts = np.arange(now - DT + 1.0 + 7.0 * j, now + 1e-9, 60.0)
            _feed(st, m, now, {did: [(t, 200.0, 400.0) for t in ts]})
        _fillers(st, now, fill)
        run_engine(eng, st, now, dt=DT)
    assert not _beacon_events(st)
    res = _pair(st, users[0], did)["res"]
    assert res["p"] < 1e-10 and res["prev_entities"] == 3.0
    scores = [_score(st, users[0], T0 + k * DT) for k in range(1, 13)]
    assert max(x for x in scores if x == x) > 10.0            # scored, just not alerted


def test_learned_class_sharing_period_tolerance():
    """A member whose beacon to the destination was LEARNED (committed through
    the gate) whitelists the same destination at P_hat +- 10 % for its class
    mates; a different period is still alerted."""
    rng = np.random.default_rng(3)
    members = ["10.0.3.1", "10.0.3.2", "10.0.3.3", "10.0.3.4"]
    for period, expect_event in ((310.0, False), (600.0, True)):
        st, eng = make_store(), B.BeaconEngine()
        put_model(st, "__org__", "__org__", "model.class", _class_model(members))
        ref = _beacon_times(rng, T0, 40, period=300.0, jitter=0.1)
        _drive(st, eng, members[1], ref, ticks=14, training=True)
        known = st.get_model(S, members[1], B.MODEL)["state"]["known"]
        assert DID in known and abs(math.exp(known[DID][0]) - 300.0) < 15.0
        t1 = T0 + 14 * DT
        bt = _beacon_times(rng, t1, 24, period=period, jitter=0.1)
        _drive(st, eng, members[0], bt, t_start=t1, ticks=8)
        res = _pair(st, members[0])["res"]
        assert res["p"] < 1e-5
        assert res["class_shared"] is (not expect_event)
        assert bool(_beacon_events(st, members[0])) is expect_event


# ------------------------------------------------------------- spec (c)
def test_c_poisson_streams_valid():
    rng = np.random.default_rng(21)
    ps, pt = [], []
    for i in range(1000):
        n = int(rng.integers(12, 129))
        t = np.cumsum(rng.exponential(300.0, n))
        r = B.evaluate_pair(t, rng.lognormal(7.0, 1.0, n), np.zeros(0), 0.5)
        ps.append(r["p"])
        pt.append(r["p_poisson"])
    ps, pt = np.array(ps), np.array(pt)
    assert np.isfinite(ps).all()
    assert np.mean(ps < 1e-3) < 0.005
    assert np.mean(pt < 1e-3) < 0.005                 # the exact kappa = 1 table alone
    assert np.mean(ps < 0.05) < 0.05


def test_c_exact_table_small_n():
    rng = np.random.default_rng(22)
    for n_ev in (12, 20, 40):
        ps = []
        for _ in range(1000):
            k, lr = evt.gamma_renewal_lrt(rng.exponential(1.0, n_ev - 1))
            ps.append(evt.beacon_null_p(lr, n_ev - 1))
        assert np.mean(np.array(ps) < 1e-3) < 0.005


# ------------------------------------------------------------- spec (d)
def test_d_lognormal_human_gaps():
    rng = np.random.default_rng(31)
    ps = []
    for i in range(1000):
        n = int(rng.integers(12, 257))
        t = np.cumsum(rng.lognormal(4.0, 1.0, n))
        ps.append(B.evaluate_pair(t, rng.lognormal(7.0, 1.0, n), np.zeros(0), 0.5)["p"])
    assert np.mean(np.array(ps) < 1e-4) < 0.005


def test_d_kappa1_alone_fails_at_full_buffer():
    """Why the renewal p is taken at kappa0 = 1.5: against kappa = 1 alone,
    256-event log-normal (sigma = 1) buffers reject far too often."""
    rng = np.random.default_rng(32)
    p1, p15 = [], []
    for _ in range(600):
        iv = rng.lognormal(4.0, 1.0, 255)
        p, _, _, p_poi = B.renewal_p(iv)
        p1.append(p_poi)
        p15.append(p)
    assert np.mean(np.array(p1) < 1e-4) > 0.005
    assert np.mean(np.array(p15) < 1e-4) == 0.0


# ------------------------------------------------------------ maths checks
@pytest.mark.parametrize("n", [11, 19, 39, 255])
@pytest.mark.parametrize("log10p", [-3.0, -6.0, -9.0])
def test_saddlepoint_matches_exact_table(n, log10p):
    row = max(i for i, m in enumerate(evt.BEACON_NULL_N) if m <= n)
    nr = evt.BEACON_NULL_N[row]
    lr = evt.BEACON_NULL_LR[row][list(evt.BEACON_NULL_LOG10P).index(log10p)]
    lo, hi = 1e-12, 1.0                                   # s with LR(s) = lr (monotone)
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if evt._kappa_lr_scalar(mid, nr)[1] > lr:
            lo = mid
        else:
            hi = mid
    p = B._gamma_lower_p(math.sqrt(lo * hi), nr, 1.0)
    assert abs(math.log10(p) - log10p) < 0.02


def test_saddlepoint_monte_carlo_kappa0():
    rng = np.random.default_rng(41)
    n = 19
    x = rng.gamma(1.5, 1.0, (100000, n))
    m = x.mean(axis=1, keepdims=True)
    d = x / m - 1.0
    s = np.mean(d - np.log1p(d), axis=1)
    for q in (1e-2, 1e-3):
        c = float(np.quantile(s, q))
        assert abs(math.log10(B._gamma_lower_p(c, n, 1.5) / q)) < 0.1
    assert B._gamma_lower_p(10.0, n, 1.5) == 1.0          # above the null mean: one-sided
    assert B._gamma_lower_p(0.0, n, 1.5) < 1e-30          # equal intervals
    assert math.isnan(B._gamma_lower_p(math.nan, n, 1.5))


def test_strict_period_and_size_rank():
    rng = np.random.default_rng(51)
    t = np.cumsum(np.full(60, 3600.0)) + rng.normal(0.0, 1.0, 60)
    r = B.evaluate_pair(t, np.full(60, 900.0), np.sort(rng.gamma(2.0, 2.0, 200)), 0.5)
    assert r["p_z2"] < 1e-20 and abs(r["period"] - 3600.0) < 5.0
    assert r["p_size"] < 1.0 / 150                         # most constant of 200 pairs
    assert r["p_size"] == pytest.approx(0.5 / 201)         # above all 200: (0 + u) / (N + 1)
    # the size test alone never alarms: it is floored at 1 / (N + 1)
    assert B._rank_p(1e6, np.full(20, 1e6), 0.5) == pytest.approx(0.5 * 21 / 21)
    assert math.isnan(B._rank_p(5.0, np.arange(19.0), 0.5))
    assert math.isnan(B._rank_p(5.0, np.arange(20.0), 0.5, own=3.0))   # own excluded
    assert B._rank_p(100.0, np.arange(20.0), 0.5) == pytest.approx(0.5 / 21)


# ------------------------------------------------------------ engine edges
def test_empty_store():
    st, eng = make_store(), B.BeaconEngine()
    assert run_engine(eng, st, T0, dt=DT) == 0
    add_obs_tick(st, S, "10.0.0.1", T0, {"act.events": 0.0})
    assert run_engine(eng, st, T0 + DT, dt=DT) == 0
    assert st.get_model(S, "10.0.0.1", B.MODEL) is None


def test_active_without_rare_is_zero_and_not_persisted():
    st, eng = make_store(), B.BeaconEngine()
    add_obs_tick(st, S, "10.0.0.1", T0, {"act.events": 12.0})
    run_engine(eng, st, T0, dt=DT)
    assert _score(st, "10.0.0.1", T0) == 0.0
    assert emit.read_dict(st, S, "10.0.0.1", emit.ACC_ALARM, T0) == {"beacon": 0}
    assert st.get_model(S, "10.0.0.1", B.MODEL) is None


def test_silent_entity_keeps_buffer_and_is_unscored():
    rng = np.random.default_rng(61)
    st, eng = make_store(), B.BeaconEngine()
    e = "10.0.0.1"
    _drive(st, eng, e, _beacon_times(rng, T0, 16), ticks=6)
    n0 = _pair(st, e)["t"].size
    now = T0 + 7 * DT
    add_obs_tick(st, S, e, now, {"act.events": 0.0})      # silent tick (R2 zero-fill)
    _fillers(st, now)
    run_engine(eng, st, now, dt=DT)
    assert math.isnan(_score(st, e, now))                  # nothing scored
    assert _pair(st, e)["t"].size == n0                    # buffer kept
    # events age out after 7 d
    later = now + 8 * 86400.0
    add_obs_tick(st, S, e, later, {"act.events": 3.0})
    _fillers(st, later)
    run_engine(eng, st, later, dt=DT)
    assert _pair(st, e) is None
    assert _score(st, e, later) == 0.0


def test_nan_inputs():
    rng = np.random.default_rng(71)
    st, eng = make_store(), B.BeaconEngine()
    e = "10.0.0.1"
    t = _beacon_times(rng, T0, 20)
    nows = _drive(st, eng, e, t, sizes=np.full(20, np.nan))   # sizes all unknown
    pr = _pair(st, e)
    assert pr["t"].size == 20 and math.isnan(pr["res"]["p_size"])
    assert pr["res"]["p"] < 1e-5                           # renewal alone still decides
    # NaN / inf times are dropped, NaN bytes kept as unknown sizes; the next
    # evaluation (EVAL_EVERY_TICKS later) sees 21 finite events
    for k in range(1, B.EVAL_EVERY_TICKS + 1):
        now = nows[-1] + k * DT
        rows = [(math.nan, 1.0, 2.0), (float(t[-1]) + 300.0, math.nan, math.nan),
                (math.inf, 1.0, 1.0)] if k == 1 else []
        _feed(st, e, now, {DID: rows}, events=3)
        _fillers(st, now)
        run_engine(eng, st, now, dt=DT)
        assert math.isfinite(_score(st, e, now))           # active tick: scored, never NaN
    pr = _pair(st, e)
    assert pr["t"].size == 21 and np.isfinite(pr["t"]).all()
    assert pr["last_eval"] == now and pr["res"]["n"] == 21
    assert math.isnan(pr["res"]["p_size"]) and pr["res"]["p"] < 1e-5
    # too few intervals: every test is undefined -> NaN, never p = 1
    r = B.evaluate_pair(np.array([0.0, 1.0]), np.full(2, np.nan), np.zeros(0), 0.5)
    assert math.isnan(r["p"])
    assert math.isnan(B.evaluate_pair(np.array([]), np.array([]), np.zeros(0), 0.5)["p"])


def test_r2_failure_degrades():
    rng = np.random.default_rng(81)
    st, eng = make_store(), B.BeaconEngine()
    e = "10.0.0.1"
    _drive(st, eng, e, _beacon_times(rng, T0, 16), ticks=5)
    now = T0 + 6 * DT
    st.put_health("raw.action_token", {"engine": "raw.action_token", "last_error_ts": now})
    run_engine(eng, st, now, dt=DT)
    assert math.isnan(_score(st, e, now))
    assert emit.read_array(st, S, e, emit.SCORE, now)[B.emit.DETECTOR_INDEX["beacon"]] != 0
    assert emit.read_dict(st, S, e, emit.DEGRADED, now) == {
        "beacon": "producer_error:raw.action_token"}


def test_training_learns_without_alarm():
    rng = np.random.default_rng(91)
    st, eng = make_store(), B.BeaconEngine()
    e = "10.0.0.1"
    nows = _drive(st, eng, e, _beacon_times(rng, T0, 40), ticks=14, training=True)
    assert not _beacon_events(st)
    scores = [_score(st, e, t) for t in nows]
    assert max(s for s in scores if s == s) > 6.0          # still scored
    for t in nows:
        assert emit.read_dict(st, S, e, emit.ACC_ALARM, t).get("beacon", 0) == 0
    m = st.get_model(S, e, B.MODEL)
    assert DID in m["state"]["known"]                      # committed after D
    assert m["gate"].journal
    assert st.profile(S, e).extra["beacons"]["pairs"][0]["established"] is True


def test_live_untrusted_rows_do_not_learn_and_quarantine_holds():
    rng = np.random.default_rng(92)
    st, eng = make_store(), B.BeaconEngine()
    e = "10.0.0.1"
    t = _beacon_times(rng, T0, 40)
    ticks = [T0 + k * DT for k in range(1, 15)]
    set_trust(st, S, e, ticks, 1.0, quarantine=1.0)       # quarantined throughout
    _drive(st, eng, e, t, ticks=14)
    m = st.get_model(S, e, B.MODEL)
    assert not m["state"]["known"]
    assert m["gate"].held                                 # held, not committed
    assert _beacon_events(st, e)                          # detection is not gated


def test_frozen_control_stops_learning():
    rng = np.random.default_rng(93)
    st, eng = make_store(), B.BeaconEngine()
    e = "10.0.0.1"
    put_model(st, S, e, "model.control", {"version": 0, "frozen": True})
    _drive(st, eng, e, _beacon_times(rng, T0, 40), ticks=14, training=True)
    assert not st.get_model(S, e, B.MODEL)["state"]["known"]


def test_allowlisted_destination_is_not_scored_or_alerted():
    rng = np.random.default_rng(94)
    st, eng = make_store(), B.BeaconEngine()
    e = "10.0.0.1"
    put_model(st, "__org__", "__org__", "model.feedback",
              {"allowlist": {f"{S}|{e}": {"dest": {str(DID): None}}}})
    nows = _drive(st, eng, e, _beacon_times(rng, T0, 30), ticks=12)
    assert not _beacon_events(st)
    res = _pair(st, e)["res"]
    assert res["p"] < 1e-5 and res["allowlisted"] is True
    assert all(_score(st, e, t) == 0.0 for t in nows)


def test_cooldown_and_throttle():
    rng = np.random.default_rng(95)
    st, eng = make_store(), B.BeaconEngine()
    e = "10.0.0.1"
    t = _beacon_times(rng, T0, 330)                       # ~27.5 h of beaconing
    evals, alarms = [], []

    def probe(st_, now):                                  # runs before each tick's engine run
        pr = _pair(st_, e)
        evals.append(None if pr is None else pr["last_eval"])
        alarms.append(emit.read_dict(st_, S, e, emit.ACC_ALARM, now - DT).get("beacon", 0))

    nows = _drive(st, eng, e, t, ticks=110, extra=probe)
    # store models are live objects: read the throttle from last_eval snapshots
    evals = evals[1:] + [_pair(st, e)["last_eval"]]
    idx = [i for i, (now, le) in enumerate(zip(nows, evals)) if le == now]
    assert len(idx) >= 4 and min(np.diff(idx)) >= B.EVAL_EVERY_TICKS
    alarms = alarms[1:] + [emit.read_dict(st, S, e, emit.ACC_ALARM, nows[-1]).get("beacon", 0)]
    assert [i for i, a in enumerate(alarms) if a] == idx[1:] or sum(alarms) >= len(idx) - 1
    assert sum(alarms) >= 2                               # acc_alarm on every alarming eval
    evs = sorted(_beacon_events(st, e), key=lambda ev: ev.ts)
    assert len(evs) == 2                                  # one event per pair per 24 h ...
    assert evs[1].ts - evs[0].ts >= B.EVENT_COOLDOWN_S    # ... and again after the cooldown
    assert evs[1].ts - evs[0].ts <= B.EVENT_COOLDOWN_S + B.EVAL_EVERY_TICKS * DT
    assert _pair(st, e)["t"].size == B.BUF_MAX            # 256-event buffer bound


def test_buffer_cap_256():
    rng = np.random.default_rng(96)
    st, eng = make_store(), B.BeaconEngine()
    e = "10.0.0.1"
    t = _beacon_times(rng, T0, 400, period=20.0)
    _drive(st, eng, e, t, ticks=10)
    pr = _pair(st, e)
    assert pr["t"].size == B.BUF_MAX
    assert pr["t"][-1] == pytest.approx(t[-1])


def test_cadence_switch_900_to_60():
    rng = np.random.default_rng(97)
    st, eng = make_store(), B.BeaconEngine()
    e = "10.0.0.1"
    t = _beacon_times(rng, T0, 60, period=120.0)
    _drive(st, eng, e, t[t <= T0 + 5 * DT], ticks=5)
    t_switch = T0 + 5 * DT
    le0 = _pair(st, e)["last_eval"]
    evals = []
    for k in range(1, 25):                                 # 24 ticks of 60 s
        now = t_switch + 60.0 * k
        sel = t[(t > now - 60.0) & (t <= now)]
        _feed(st, e, now, {DID: [(x, 400.0, 900.0) for x in sel]}, events=max(1, sel.size))
        _fillers(st, now)
        run_engine(eng, st, now, dt=60.0)
        if _pair(st, e)["last_eval"] == now:
            evals.append(k)
    assert le0 <= t_switch and evals
    assert min(np.diff(evals)) >= B.EVAL_EVERY_TICKS if len(evals) > 1 else True
    assert len(evals) >= 3                                  # 4-tick throttle at 60 s, not 3600 s
    res = _pair(st, e)["res"]
    assert res["p"] < 1e-6 and abs(res["period"] - 120.0) < 25.0


def test_learner_update_is_order_free_and_mergeable():
    rows = [(T0 + 900.0 * i, ((1, 300.0 + i), (2, 60.0))) for i in range(6)]
    a = B._init_state()
    for r in rows:
        B._update(a, r, 1.0)
    b = B._init_state()
    for r in reversed(rows):
        B._update(b, r, 1.0)
    for d in (1, 2):
        assert a["known"][d][0] == pytest.approx(b["known"][d][0], rel=1e-12)
        assert a["known"][d][1] == pytest.approx(b["known"][d][1], rel=1e-12)
    c = B._merge(B._init_state(), a, 0.5)
    assert c["known"][1][1] == pytest.approx(0.5 * a["known"][1][1])
    assert B._update(B._init_state(), rows[0], 0.0) == B._init_state()
    old = (T0 - 30 * 86400.0, ((3, 100.0),))
    d = B._update(B._init_state(), old, 1.0)
    d = B._update(d, rows[0], 1.0)
    assert 3 not in d["known"]                              # pruned in row time


def test_perf_bound():
    rng = np.random.default_rng(99)
    st, eng = make_store(), B.BeaconEngine()
    ents = [f"10.1.0.{i}" for i in range(60)]
    t_all = {e: [_beacon_times(rng, T0, 256, period=20.0 + i),
                 np.cumsum(rng.exponential(25.0, 256)) + T0] for i, e in enumerate(ents)}
    times = []
    for k in range(1, 7):
        now = T0 + k * DT
        for e in ents:
            rows = {}
            for j, t in enumerate(t_all[e]):
                sel = t[(t > now - DT) & (t <= now)]
                rows[1000 + j] = [(x, 100.0, 300.0) for x in sel]
            _feed(st, e, now, rows)
        _fillers(st, now, [f"10.2.0.{i}" for i in range(400)])
        t0 = time.perf_counter()
        run_engine(eng, st, now, dt=DT)
        times.append(time.perf_counter() - t0)
    # per-tick evaluation is bounded (MAX_EVAL_PER_TICK, Z2_MAX_PER_TICK); the rest is
    # O(entities x pairs) bookkeeping. Generous bound for CI noise.
    assert max(times[1:]) < 0.5
    evaluated = sum(1 for e in ents for pr in st.get_model(S, e, B.MODEL)["pairs"].values()
                    if pr["res"] is not None)
    assert evaluated >= B.MAX_EVAL_PER_TICK
