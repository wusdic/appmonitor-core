"""B15 IdentityModelEngine edge cases: governance of the window learner
(trust, quarantine / release, rollback, link seeding), cadence switch, fit
stride, live modality calibration and a perf bound. Spec tests and the shared
synthetic feed live in test_b15_identity_model.py."""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from helpers import DT, T0, make_store, put_model, run_engine, set_trust

from app.core.engine import Context
from app.engines.behavior import identity_model as IM
from app.engines.behavior.identity_model import IdentityModelEngine
from app.engines.behavior.lib import m_identity as MI
from app.engines.behavior.lib import m_rhythm
from app.models.schema import RawMetric
from test_b15_identity_model import (DISTINCT, S, Persona, _unit_blocks, feed, spec_personas,
                                     write_tick)


def test_untrusted_and_quarantined_rows_are_not_learned():
    store = make_store()
    eng = IdentityModelEngine()
    p = spec_personas()
    rng = np.random.default_rng(11)
    e_bad, e_q = DISTINCT[0], DISTINCT[1]
    ts = [T0 + k * DT for k in range(40)]
    set_trust(store, S, e_bad, ts, 0.0)                         # trust 0: skipped
    set_trust(store, S, e_q, ts, 1.0, quarantine=1.0)           # quarantined: held
    for now in ts:
        write_tick(store, e_bad, now, p[e_bad], rng)
        write_tick(store, e_q, now, p[e_q], rng)
        run_engine(eng, store, now, training=False)
    idw = store.get_model(S, "__system__", MI.IDWIN)
    assert len(idw["ents"][e_bad]["vecs"]) == 0
    assert len(idw["ents"][e_q]["vecs"]) == 0
    assert len(idw["ents"][e_q]["gate"].held) >= 30
    # the governor releases the episode: the held rows become windows
    put_model(store, S, e_q, "model.control", {"version": 1, "release": [T0, ts[-1]]})
    set_trust(store, S, e_q, [ts[-1] + DT], 1.0, quarantine=0.0)
    write_tick(store, e_q, ts[-1] + DT, p[e_q], rng)
    run_engine(eng, store, ts[-1] + DT, training=False)
    assert len(idw["ents"][e_q]["vecs"]) >= 7


def test_rollback_removes_windows_after_tau():
    store = make_store()
    eng = IdentityModelEngine()
    p = {DISTINCT[0]: spec_personas()[DISTINCT[0]]}
    now = feed(store, eng, p, 100, training=True)
    rec = store.get_model(S, "__system__", MI.IDWIN)["ents"][DISTINCT[0]]
    n_before = len(rec["vecs"])
    tau = T0 + 50 * DT
    put_model(store, S, DISTINCT[0], "model.control", {"version": 1, "rollback_to": tau})
    set_trust(store, S, DISTINCT[0], [now + DT], 1.0, quarantine=1.0)
    write_tick(store, DISTINCT[0], now + DT, p[DISTINCT[0]], np.random.default_rng(0))
    run_engine(eng, store, now + DT, training=False)
    rec = store.get_model(S, "__system__", MI.IDWIN)["ents"][DISTINCT[0]]
    assert all(w[1] <= tau for w in rec["state"]["wins"])
    assert len(rec["vecs"]) < n_before
    assert set(rec["vecs"]) == {IM.wkey(w[1]) for w in rec["state"]["wins"]}


def test_link_seeding_copies_half_of_the_windows():
    store = make_store()
    eng = IdentityModelEngine()
    per = spec_personas()
    a, b = DISTINCT[0], DISTINCT[1]
    now = feed(store, eng, {a: per[a], b: per[b]}, 44, training=True)
    idw = store.get_model(S, "__system__", MI.IDWIN)
    n_a = len(idw["ents"][a]["state"]["wins"])
    put_model(store, S, "__system__", "model.link",
              {"version": 1, "links": [{"from": a, "to": b, "ts": now}], "actors": []})
    now += DT
    write_tick(store, a, now, per[a], np.random.default_rng(1))
    write_tick(store, b, now, per[b], np.random.default_rng(2))
    run_engine(eng, store, now, training=True)
    wins_b = idw["ents"][b]["state"]["wins"]
    seeded = [w for w in wins_b if w[2] == a]
    assert len(seeded) == math.ceil(n_a / 2)
    names, Xs, _ = eng._gather(idw)
    assert b in names and Xs[names.index(b)].shape[0] == len(wins_b)


def test_cadence_switch_900_to_60():
    store = make_store()
    eng = IdentityModelEngine()
    per = {e: spec_personas()[e] for e in DISTINCT[:2]}
    now = feed(store, eng, per, 40, dt=900.0, training=True)
    now = feed(store, eng, per, 120, t0=now + 60.0, dt=60.0, training=True)
    idw = store.get_model(S, "__system__", MI.IDWIN)
    for e in per:
        wins = idw["ents"][e]["state"]["wins"]
        t1 = [w[1] for w in wins]
        assert t1 == sorted(t1) and len(set(t1)) == len(t1)
        # every committed tick is used once: 40 + 120 ticks minus the 60-s D (10 ticks)
        assert len(wins) == (40 + 120 - 10) // 4
        assert idw["ents"][e]["gate"].last_ts == pytest.approx(now - 10 * 60.0)


def test_fit_stride_and_first_fit():
    store = make_store()
    eng = IdentityModelEngine(fit_ticks=96)
    per = {e: spec_personas()[e] for e in DISTINCT[:2]}
    feed(store, eng, per, 45, training=True)          # 9 windows each after 40 committed ticks
    model = MI.get(store, S)
    assert model is not None and model["version"] == 1          # first fit as soon as possible
    feed(store, eng, per, 30, t0=T0 + 45 * DT, training=True)
    assert MI.get(store, S)["version"] == 1                      # stride not elapsed
    idw = store.get_model(S, "__system__", MI.IDWIN)
    assert idw["sched"]["runs"] == 1


def test_live_modality_calibration_samples_rhythm():
    store = make_store()
    eng = IdentityModelEngine()
    per = {e: spec_personas()[e] for e in DISTINCT[:3]}
    for i, e in enumerate(per):                                   # own rhythm models
        m = m_rhythm.new_model("entity")
        m["state"]["N48"][:] = 50.0
        m["state"]["A48"][:] = 45.0 if i == 0 else 5.0 * i
        put_model(store, S, e, m_rhythm.MODEL, m)
    rng = np.random.default_rng(4)
    for k in range(24):
        now = T0 + k * DT
        for e, p in per.items():
            write_tick(store, e, now, p, rng)
            store.add_raw(RawMetric(name="act.events", value=1.0, ts=now, system=S, entity=e))
        run_engine(eng, store, now, training=True)
    ring = store.get_model(S, "__system__", MI.IDWIN)["cal"]["ring"]["rhythm"]
    assert ring
    lab = np.asarray([r[1] for r in ring])
    assert {0, 1} <= set(lab.tolist())
    assert all(math.isfinite(r[0]) for r in ring)
    # the entity whose model says 'always active' gets the largest genuine LLR
    for m in ("vocab", "seq", "client", "timing"):              # no raw evidence written
        assert store.get_model(S, "__system__", MI.IDWIN)["cal"]["ring"].get(m, []) == []


def test_ring_calibration_is_fitted():
    store = make_store()
    eng = IdentityModelEngine()
    now = feed(store, eng, {e: spec_personas()[e] for e in DISTINCT[:2]}, 44, training=True)
    idw = store.get_model(S, "__system__", MI.IDWIN)
    rng = np.random.default_rng(8)
    idw["cal"]["ring"]["vocab"] = ([[float(x), 1, now] for x in rng.normal(3, 2, 300)]
                                   + [[float(x), 0, now] for x in rng.normal(-3, 2, 300)])
    eng.refit(Context(store=store, now=now, window_s=DT, config={"strict": True}))
    a, b = MI.llr_calib(MI.get(store, S), "vocab")
    assert a == pytest.approx(1.5, rel=0.2) and abs(b) < 0.3     # true LLR slope 6/4


def test_perf_collection_and_fit():
    store = make_store()
    eng = IdentityModelEngine()
    rng = np.random.default_rng(12)
    per = {}
    for i in range(20):
        mu = np.zeros(52)
        mu[i % 10] = 1.5 * (1 + i // 10)
        per[f"10.1.0.{i}"] = Persona(mu, _unit_blocks(rng.normal(0, 1, 80)))
    t = time.perf_counter()
    now = feed(store, eng, per, 60, training=True)
    tick_ms = (time.perf_counter() - t) / 60 * 1000.0
    c = Context(store=store, now=now, window_s=DT, config={"strict": True})
    t = time.perf_counter()
    assert eng.refit(c) == 20
    fit_s = time.perf_counter() - t
    assert tick_ms < 80.0, tick_ms          # includes the synthetic feed writes
    assert fit_s < 2.0, fit_s
