"""B09 ClientIdentityEngine edge cases: empty store, silent entity, NaN /
junk inputs, training mode, cadence 900 -> 60 s, R3 failure (degraded),
governance (quarantine, rollback, frozen, link seeding), NAT mixtures, a
brand-new entity with a forged stack, the m_client accessors and a perf check.
"""
from __future__ import annotations

import math
import time

import numpy as np

from helpers import DT, T0, make_store, put_model, run_engine, set_trust
from test_b09_client_identity import (CHROME126, E, FORGED, PYTHON, S, Feed, _token,
                                      events, pm_at, score_at, times, warm)

from app.engines.behavior.client_identity import (DETECTOR, ClientIdentityEngine,
                                                  _concurrent)
from app.engines.behavior.lib import emit
from app.engines.behavior.lib import m_client as MC
from app.models.schema import RawMetric

PEERS = {f"10.0.0.{i}": CHROME126 for i in range(2, 6)}


def _model(store, e=E):
    return MC.get(store, S, e)


# ------------------------------------------------------------------ basics
def test_empty_store_and_no_client_data():
    store, eng = make_store(), ClientIdentityEngine()
    assert run_engine(eng, store, T0) == 0
    # an entity with other data but no fingerprinted traffic: nothing written
    store.add_raw(RawMetric(name="l4.flows", value=3.0, ts=T0, system=S, entity=E))
    assert run_engine(eng, store, T0) == 0
    assert _model(store) is None and MC.get(store, S, "__system__") is None


def test_silent_entity_is_unscored_and_not_absent():
    store, eng = make_store(), ClientIdentityEngine()
    feed = Feed(store)
    now = warm(feed, eng, {E: CHROME126, **PEERS}, 40)
    m = _model(store)
    for k in range(1, 9):                      # E silent for 2 h, peers active
        now = T0 + (39 + k) * DT
        feed.tick(now, {e: [(st, times(now))] for e, st in PEERS.items()})
        run_engine(eng, store, now)
        assert math.isnan(pm_at(store, E, now))            # unscored, never p = 1
        assert not math.isnan(pm_at(store, "10.0.0.2", now))
    # silence is not absence of the stack: the active clock did not move
    assert m["live"]["miss"] == {}
    assert m["gate"].last_ts > T0 + 30 * DT                # learning kept stepping
    now += DT                                              # E comes back: quiet score
    feed.tick(now, {E: [(CHROME126, times(now))],
                    **{e: [(st, times(now))] for e, st in PEERS.items()}})
    run_engine(eng, store, now)
    assert pm_at(store, E, now) > 0.5
    assert not events(store, "client_change") and not events(store, "client_impersonation")


def test_nan_and_junk_inputs():
    store, eng = make_store(), ClientIdentityEngine()
    feed = Feed(store)
    now = warm(feed, eng, {E: CHROME126, **PEERS}, 30)
    tok = _token(CHROME126)
    junk_sets = [
        {tok: {"n": math.nan, "first_ts": now, "last_ts": now}, "__other__": {"n": 5}},
        {tok: {"n": -3}, "x": {"n": "abc"}, 7: {"n": 2}},
        "not a dict",
        {tok: {"n": math.inf}},
    ]
    for k, js in enumerate(junk_sets, start=1):
        now = T0 + (29 + k) * DT
        for name, v in (("client.stack_set", js),
                        ("client.stack_events", [[None, math.nan, 1.0, 2], "bad", [1]]),
                        ("client.os_ua_ttl_pairs", {"win|chrome|64": math.nan, 5: 1})):
            store.add_raw(RawMetric(name=name, value=v, ts=now, system=S, entity=E))
        store.add_vec(S, E, "feature.active", now, np.array([1.0], np.float32))
        set_trust(store, S, E, [now], 1.0)
        run_engine(eng, store, now)
        assert math.isnan(pm_at(store, E, now))            # nothing usable: unscored
    # usable counts with junk episodes / pairs: scored from the stack_set envelope
    now += DT
    store.add_raw(RawMetric(name="client.stack_set", system=S, entity=E, ts=now,
                            value={tok: {"n": 12, "first_ts": math.nan, "last_ts": now}}))
    store.add_raw(RawMetric(name="client.stack_events", system=S, entity=E, ts=now,
                            value=[[1, math.nan, math.nan, 3]]))
    run_engine(eng, store, now)
    p = pm_at(store, E, now)
    assert 0.5 < p <= 1.0
    assert np.isfinite(score_at(store, E, now))


def test_training_learns_but_emits_no_events():
    store, eng = make_store(), ClientIdentityEngine()
    feed = Feed(store)
    now = warm(feed, eng, {E: CHROME126, **PEERS}, 60)
    for k in range(1, 8):
        now = T0 + (59 + k) * DT
        traffic = {e: [(st, times(now))] for e, st in PEERS.items()}
        traffic[E] = [(CHROME126, times(now)), (PYTHON, times(now, offset=30.0))]
        feed.tick(now, traffic, trust=False)
        run_engine(eng, store, now, training=True)
    assert not store.events(system=S, limit=1000)
    assert MC.recent(_model(store))["risk"] is not None
    assert np.isfinite(score_at(store, E, now))            # scores are still written
    # it learned python-requests (training trust = 1 without B28 rings)
    assert _token(PYTHON) in _model(store)["state"]["c"]


def test_degraded_when_r3_failed():
    store, eng = make_store(), ClientIdentityEngine()
    feed = Feed(store)
    now = warm(feed, eng, {E: CHROME126, **PEERS}, 10)
    now += DT
    store.put_health("raw.client_stack", {"engine": "raw.client_stack", "ok": False,
                                          "last_error_ts": now})
    n = run_engine(eng, store, now)
    assert n == 1 + len(PEERS)
    assert math.isnan(score_at(store, E, now))
    deg = emit.read_dict(store, S, E, emit.DEGRADED, now)
    assert deg == {DETECTOR: "producer_error:raw.client_stack"}


# ------------------------------------------------------------------ cadence
def test_cadence_900_to_60():
    store, eng = make_store(), ClientIdentityEngine()
    feed = Feed(store)
    now = warm(feed, eng, {E: CHROME126, **PEERS}, 60)
    dt = 60.0
    t1 = now
    for k in range(1, 91):                     # 90 min of 60-s ticks, same clients
        now = t1 + k * dt
        feed.tick(now, {e: [(st, times(now, dt, step=10.0))]
                        for e, st in {E: CHROME126, **PEERS}.items()}, dt=dt)
        run_engine(eng, store, now, dt=dt)
        p = pm_at(store, E, now)
        assert p > 0.5, (k, p)
    m = _model(store)
    rows = m["rows"]
    # a 60-s row carries dt/900 slot-equivalents, a 900-s row 1
    assert math.isclose(sum(rows.find(now)[1]), dt / 900.0, rel_tol=1e-9)
    assert math.isclose(sum(rows.find(t1)[1]), 1.0, rel_tol=1e-9)
    assert not events(store, "client_change") and not events(store, "client_impersonation")
    # impersonation still found within a few 60-s ticks
    fired = None
    for k in range(1, 6):
        now += dt
        traffic = {e: [(st, times(now, dt, step=10.0))] for e, st in PEERS.items()}
        traffic[E] = [(CHROME126, times(now, dt, step=10.0)),
                      (PYTHON, times(now, dt, step=10.0, offset=5.0))]
        feed.tick(now, traffic, dt=dt)
        run_engine(eng, store, now, dt=dt)
        if events(store, "client_impersonation", E):
            fired = k
            break
    assert fired is not None and fired <= 3


def test_concurrency_whatever_the_tick_length():
    # interleaved at 2-min gaps -> concurrent; a clean handover is not
    a = [(0.0, 50.0), (300.0, 400.0)]
    b = [(120.0, 200.0)]
    assert _concurrent(b, [], a)
    assert not _concurrent([(420.0, 600.0)], [], [(0.0, 400.0)])      # s0 stopped first
    assert not _concurrent([(900.0, 950.0)], [], [(0.0, 500.0)])      # > 5 min apart
    assert _concurrent([(700.0, 750.0)], [], [(0.0, 500.0), (760.0, 800.0)])


# ------------------------------------------------------------------ governance
def test_quarantine_holds_rows_and_rollback_forgets_them():
    store, eng = make_store(), ClientIdentityEngine()
    feed = Feed(store)
    now = warm(feed, eng, {E: CHROME126, **PEERS}, 60, training=False)
    t_py = now + DT
    # python-requests for 12 ticks, trusted: it gets committed after D ticks
    for k in range(1, 13):
        now = T0 + (59 + k) * DT
        traffic = {e: [(st, times(now))] for e, st in PEERS.items()}
        traffic[E] = [(CHROME126, times(now)), (PYTHON, times(now, offset=30.0))]
        feed.tick(now, traffic)
        run_engine(eng, store, now)
    py = _token(PYTHON)
    assert py in _model(store)["state"]["c"]
    # the governor quarantines E (gating reads quarantine at t - 1), then
    # rolls it back to before the python traffic
    for step in range(2):
        if step == 1:
            put_model(store, S, E, "model.control", {"version": 1, "rollback_to": t_py - DT})
        now += DT
        feed.tick(now, {E: [(CHROME126, times(now)), (PYTHON, times(now, offset=30.0))],
                        **{e: [(st, times(now))] for e, st in PEERS.items()}}, trust=False)
        set_trust(store, S, E, [now], 0.0, quarantine=1.0)
        for e in PEERS:
            set_trust(store, S, e, [now], 1.0)
        run_engine(eng, store, now)
    m = _model(store)
    assert py not in m["state"]["c"]
    assert m["gate"].held and all(r.ts >= t_py for r in m["gate"].held)
    # while quarantined, nothing more is learned
    for _ in range(6):
        now += DT
        feed.tick(now, {E: [(PYTHON, times(now))],
                        **{e: [(st, times(now))] for e, st in PEERS.items()}}, trust=False)
        set_trust(store, S, E, [now], 0.0, quarantine=1.0)
        run_engine(eng, store, now)
    assert py not in _model(store)["state"]["c"]
    # frozen: held rows are dropped and commits stop
    put_model(store, S, E, "model.control", {"version": 1, "rollback_to": t_py - DT,
                                             "frozen": True})
    now += DT
    feed.tick(now, {E: [(PYTHON, times(now))]})
    run_engine(eng, store, now)
    m = _model(store)
    assert m["gate"].frozen and not m["gate"].held


def test_link_seeding_merges_the_linked_entity():
    store, eng = make_store(), ClientIdentityEngine()
    feed = Feed(store)
    old, new = "10.0.9.1", "10.0.9.2"
    now = warm(feed, eng, {old: PYTHON, new: CHROME126, **PEERS}, 20)
    n_py_before = MC.counts(_model(store, new), now).get(_token(PYTHON), 0.0)
    put_model(store, S, "__system__", "model.link",
              {"version": 1, "links": [{"from": old, "to": new, "ts": now}]})
    now += DT
    feed.tick(now, {e: [(st, times(now))] for e, st in
                    {old: PYTHON, new: CHROME126, **PEERS}.items()})
    run_engine(eng, store, now)
    got = MC.counts(_model(store, new), now).get(_token(PYTHON), 0.0)
    assert n_py_before == 0.0 and got > 1.0
    assert _model(store, new)["gate"].link_version == 1


# ------------------------------------------------------------------ scenarios
def test_nat_mixture_is_quiet():
    store, eng = make_store(), ClientIdentityEngine()
    feed = Feed(store)
    firefox = (PYTHON[0][:-1] + "4", "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 "
               "Firefox/128.0", 64, 29200)
    now = T0
    for i in range(80):                         # two devices behind one IP from day one
        now = T0 + i * DT
        traffic = {e: [(st, times(now))] for e, st in PEERS.items()}
        traffic[E] = [(CHROME126, times(now)), (firefox, times(now, offset=20.0))]
        feed.tick(now, traffic)
        run_engine(eng, store, now, training=i < 60)
    assert not events(store, "client_impersonation")
    assert pm_at(store, E, now) > 0.5
    dom = MC.dominant(_model(store), k=2, now=now)
    assert len(dom) == 2 and abs(dom[0][1] - 0.5) < 0.1


def test_new_entity_with_forged_stack():
    store, eng = make_store(), ClientIdentityEngine()
    feed = Feed(store)
    now = warm(feed, eng, dict(PEERS), 40)
    now += DT
    feed.tick(now, {"10.0.7.7": [(FORGED, times(now))],
                    **{e: [(st, times(now))] for e, st in PEERS.items()}})
    run_engine(eng, store, now)
    ev = events(store, "client_impersonation", "10.0.7.7")
    assert ev and ev[0].extra["I"] == 1 and ev[0].extra["C"] == 0
    assert ev[0].extra["dominant"] is None            # no history: nothing dominant yet


# ------------------------------------------------------------------ accessors
def test_m_client_accessors_and_profile():
    store, eng = make_store(), ClientIdentityEngine()
    feed = Feed(store)
    now = warm(feed, eng, {E: CHROME126, "10.0.9.1": PYTHON, **PEERS}, 30)
    m, sysm = _model(store), MC.get(store, S, "__system__")
    ch, py = _token(CHROME126), _token(PYTHON)
    assert MC.kind(m) == "entity" and MC.kind(sysm) == "system"
    # loglik prefers the entity's own stack mix (identity candidates)
    obs = {ch: 20}
    assert MC.loglik(m, obs, sysm, now=now) > MC.loglik(_model(store, "10.0.9.1"), obs, sysm,
                                                       now=now)
    assert MC.loglik_store(store, S, E, {ch: {"n": 20}}, now=now) == \
        MC.loglik(m, obs, sysm, now=now)
    assert MC.loglik(m, {}, sysm) == 0.0 and MC.loglik(m, {"__other__": 3}, sysm) == 0.0
    p = MC.prob(store, S, E, ch, now)
    assert 0.8 < p < 1.0
    assert MC.prob(store, S, E, py, now) < MC.NEW_P
    assert MC.surprisal(store, S, E, "never|seen/0|?|0|w0", now) == MC.SURPRISE_CAP_BITS
    # the system predictive is normalised over the universe
    b = MC.Backoff(None, None, sysm, now)
    seen = sum(b.p(t) for t in sysm["c"])
    assert math.isclose(seen + (MC.UNIVERSE - len(sysm["c"])) * b.U / (MC.total(sysm, now) + 1),
                        1.0, rel_tol=1e-6)
    assert MC.dominant(m, now=now)[0][0] == ch
    assert MC.p99_gap(m, ch) == 0.0 and math.isnan(MC.p99_gap(m, py))
    pj, n = MC.p_ja3n_given_ua(sysm, MC.parse(ch).ja3n, "chrome/126")
    assert pj == 1.0 and n > 16
    d = MC.descriptors(m, now=now)
    assert d["dominant"][0]["ua"] == "chrome/126" and d["dominant"][0]["os"] == "win"
    assert d["n_stacks"] == 1 and d["maturity"] > 0.8
    prof = store.profile(S, E)
    cs = prof.extra["client_stacks"]
    assert cs["dominant"][0]["token"] == ch and cs["recent"]["scored"] is True
    assert MC.stack_counts({ch: {"n": 3}, "__other__": {"n": 2}, "x": {"n": math.nan}}) == {ch: 3}
    r_cls, r_sys = MC.rollout_share(sysm, ch, E, now)
    assert math.isnan(r_cls) and r_sys == 0.0


def test_perf_per_tick():
    store, eng = make_store(), ClientIdentityEngine()
    feed = Feed(store)
    ents = {f"10.2.{i // 250}.{i % 250}": (CHROME126 if i % 5 else PYTHON) for i in range(60)}
    warm(feed, eng, ents, 12)
    t_run = 0.0
    n_ticks = 8
    for k in range(n_ticks):
        now = T0 + (12 + k) * DT
        feed.tick(now, {e: [(st, times(now))] for e, st in ents.items()})
        t = time.perf_counter()
        run_engine(eng, store, now)
        t_run += time.perf_counter() - t
    per_entity_ms = 1e3 * t_run / (n_ticks * len(ents))
    print(f"B09 per entity-tick: {per_entity_ms:.3f} ms")
    assert per_entity_ms < 2.0, per_entity_ms          # spec: <= 1 ms per tick
