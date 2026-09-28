"""B02 PeerGroupEngine: static classes and pools, cold-start edge cases
(empty store, silent entity, NaN inputs), training mode, cadence 900 -> 60,
the m_class reader contract, versioning and perf.

Uses the World builder of test_b02_peer_group.py."""
from __future__ import annotations

import math
import time

import numpy as np

from helpers import make_store, run_engine

from app.engines.behavior.lib import features as F
from app.engines.behavior.lib import m_class
from app.engines.behavior.lib.classkeys import ORG, SYSTEM_KEY
from app.engines.behavior.peer_group import (MODEL, REFIT_TICKS, PeerGroupEngine, level1,
                                             match_ids)
from test_b02_peer_group import (DT, NOW, ROLES, S, World, events, ip,
                                 new_entity_tick, role_of)

H6 = 6 * 3600.0 + DT


def fit(w: World, eng: PeerGroupEngine, now: float = NOW, **kw) -> dict:
    run_engine(eng, w.store, now, dt=DT, **kw)
    return w.model()


def preset(w: World, roles: dict, next_role: int = 10) -> None:
    """A previous run's model.class: {rid: [ips]} (established members)."""
    m = PeerGroupEngine._new_model(None)
    m["_state"]["next_role"] = next_role
    m["_state"]["last_refit"] = NOW - 7 * 3600.0
    for rid, ips in roles.items():
        keys = [f"{S}|{e}" for e in ips]
        m["roles"][rid] = {"name": rid, "members": keys, "cluster": keys, "super": "human",
                           "lineage": [], "version": 1, "absent": 0, "next_sub": 1}
        for k in keys:
            m["assign"][k] = {"role": rid, "sub": None, "prob": 0.9, "static": [], "pool": None,
                              "super": "human", "class_path": f"human/{rid}"}
    w.store.put_model(ORG[0], ORG[1], MODEL, m, version=1)


# ================================================================ metadata
def test_metadata_and_no_arg_constructor():
    eng = PeerGroupEngine()
    assert eng.name == "behavior.peer_group" and eng.layer == "behavior"
    assert eng.interval == 1                   # cold path every tick; refit has its own stride
    for r in ("model.baseline", "model.vocab", "model.rhythm", "model.client", "model.timing",
              "behavior.quarantine"):
        assert r in eng.consumes
    assert "model.class" in eng.produces


def test_empty_store():
    store = make_store()
    eng = PeerGroupEngine()
    assert run_engine(eng, store, NOW, dt=DT) == 0
    mc = store.get_model(ORG[0], ORG[1], MODEL)
    assert mc["assign"] == {} and mc["roles"] == {}
    assert m_class.all_class_keys(store, S) == []
    assert run_engine(eng, store, NOW + DT, dt=DT) == 0


def test_silent_entity_and_nan_inputs():
    w = World(outlier=False)
    eng = PeerGroupEngine()
    fit(w, eng)
    # silent: registered, no models, never active -> untouched, no cold record
    w.store.register_entity(S, "10.0.5.5")
    # NaN: active ticks whose feature.nat is all NaN and no raw sets
    rng = np.random.default_rng(1)
    for k in range(1, 4):
        ts = NOW + k * DT
        new_entity_tick(w.store, S, "10.0.5.6", ROLES[0], ts, rng,
                        nat=np.full(F.FEATURE_DIM, np.nan), raw=False)
        run_engine(eng, w.store, ts, dt=DT)
    mc = w.model()
    assert f"{S}|10.0.5.5" not in mc["assign"] and f"{S}|10.0.5.5" not in mc["_state"]["cold"]
    evs = events(w.store, "new_entity_matched", "10.0.5.6") + \
        events(w.store, "new_entity_unmatched", "10.0.5.6")
    assert len(evs) == 1
    assert math.isfinite(evs[0].extra["prob"]) and math.isfinite(evs[0].extra["D"])
    assert not events(w.store, "new_entity_matched", "10.0.5.5")


def test_training_mode_learns_but_emits_nothing():
    w = World()
    eng = PeerGroupEngine()
    fit(w, eng, training=True)
    rng = np.random.default_rng(2)
    for k in range(1, 4):
        ts = NOW + k * DT
        new_entity_tick(w.store, S, "10.0.1.98", ROLES[1], ts, rng)
        run_engine(eng, w.store, ts, dt=DT, training=True)
    assert w.store.events(limit=100) == []
    mc = w.model()
    assert role_of(mc, S, "10.0.9.9") == "unique"
    assert role_of(mc, S, "10.0.1.98") == role_of(mc, S, ip(1, 0))
    # training ends: the entity typed during training is not announced late
    ts = NOW + 4 * DT
    new_entity_tick(w.store, S, "10.0.1.98", ROLES[1], ts, rng)
    run_engine(eng, w.store, ts, dt=DT)
    run_engine(eng, w.store, ts + H6, dt=DT)
    assert not events(w.store, "new_entity_matched") and not events(w.store, "peer_outlier")


def test_cadence_900_to_60_same_descriptor_and_tick_stride():
    w = World(outlier=False)
    eng = PeerGroupEngine()
    fit(w, eng)
    rng = np.random.default_rng(4)
    # the same traffic at 900-s ticks and at 60-s ticks (1/15 of the counts)
    t = NOW
    for k in range(3):
        t += DT
        new_entity_tick(w.store, S, "10.0.1.90", ROLES[1], t, rng, dt=DT)
        run_engine(eng, w.store, t, dt=DT)
    for k in range(3):
        t += 60.0
        new_entity_tick(w.store, S, "10.0.1.91", ROLES[1], t, rng, dt=60.0)
        run_engine(eng, w.store, t, dt=60.0)
    ev = {e: events(w.store, "new_entity_matched", e) for e in ("10.0.1.90", "10.0.1.91")}
    assert len(ev["10.0.1.90"]) == 1 and len(ev["10.0.1.91"]) == 1
    assert ev["10.0.1.90"][0].extra["role"] == ev["10.0.1.91"][0].extra["role"]
    st = w.model()["_state"]
    c0 = np.array(st["cold"][f"{S}|10.0.1.90"]["ch"]) / 3
    c1 = np.array(st["cold"][f"{S}|10.0.1.91"]["ch"]) / 3
    # per-minute rates, not per-tick (http and flows: the others round to 0 at 60 s)
    assert np.allclose(c0[[0, 3]], c1[[0, 3]], rtol=0.15)
    # at 60-s ticks the refit follows the 16-tick stride, not 6 h
    v0 = w.store.model_version(ORG[0], ORG[1], MODEL)
    last = st["last_refit"]
    for k in range(REFIT_TICKS):
        t += 60.0
        run_engine(eng, w.store, t, dt=60.0)
    assert w.model()["_state"]["last_refit"] > last
    assert w.store.model_version(ORG[0], ORG[1], MODEL) > v0


def test_static_classes_profiles_and_reader_contract():
    w = World()
    cfg = {"ip_classes": [{"name": "servers", "cidrs": ["10.0.1.0/24"], "systems": [],
                           "criticality": "high"},
                          {"name": "other-sys", "cidrs": ["10.0.0.0/16"], "systems": ["oa"]}]}
    mc = fit(w, PeerGroupEngine(), config=cfg)
    assert mc["assign"][f"{S}|{ip(1, 0)}"]["static"] == ["servers"]
    assert mc["assign"][f"{S}|{ip(0, 0)}"]["static"] == []
    assert sorted(mc["statics"]["servers"]["members"]) == [f"{S}|{ip(1, i)}" for i in range(4)]
    keys = m_class.all_class_keys(w.store, S)
    r_api = role_of(mc, S, ip(1, 0))
    assert "class:static:servers" in keys and f"class:{r_api}" in keys
    assert m_class.class_members(w.store, S, "class:static:servers") == [ip(1, i) for i in range(4)]
    assert m_class.class_key(w.store, S, ip(1, 2)) == f"class:{r_api}"
    assert m_class.class_key(w.store, S, "10.0.9.9") is None      # unique
    assert m_class.role_name(w.store, r_api) == mc["roles"][r_api]["name"]
    # entity profile (contract G) and class profiles
    p = w.store.profile(S, ip(1, 0))
    pg = p.extra["peer_group"]
    assert {"role", "role_name", "sub", "prob", "static_classes", "pool", "class_path"} <= set(pg)
    assert p.archetype == pg["class_path"] and p.archetype_confidence == pg["prob"]
    assert pg["static_classes"] == ["servers"]
    cp = w.store.profile(S, f"class:{r_api}")
    assert cp.extra["peer_group"]["n_members"] == 4 and cp.archetype.endswith(r_api)
    assert w.store.profile(S, "class:static:servers").extra["peer_group"]["criticality"] == "high"
    assert f"class:{r_api}" in w.store.pseudo_entities(S)


def test_pools_of_short_lived_ips():
    w = World(outlier=False)
    eng = PeerGroupEngine()
    fit(w, eng)
    rng = np.random.default_rng(9)
    pool_ips = [f"10.0.8.{i}" for i in range(1, 6)]
    t = NOW
    for k in range(3):
        t += DT
        for e in pool_ips:
            new_entity_tick(w.store, S, e, ROLES[1], t, rng)
        run_engine(eng, w.store, t, dt=DT)
    mc = fit(w, eng, t + H6)
    r_api = role_of(mc, S, ip(1, 0))
    assert mc["pools"]["10.0.8.0/24"]["role"] == r_api
    assert all(mc["assign"][f"{S}|{e}"]["pool"] == "10.0.8.0/24" for e in pool_ips)
    assert m_class.class_members(w.store, S, "class:pool:10.0.8.0/24") == sorted(pool_ips)
    # a linkable member (model.link) disqualifies the pool
    w.store.put_model(S, SYSTEM_KEY, "model.link", {"links": [{"from": "10.0.8.1", "to": "10.0.8.2"}],
                                                   "version": 1})
    mc = fit(w, eng, t + 2 * H6)
    assert mc["pools"] == {} and mc["assign"][f"{S}|10.0.8.1"]["pool"] is None


def test_version_bumps_with_a_new_object_on_change():
    w = World(outlier=False)
    eng = PeerGroupEngine()
    m0 = fit(w, eng)
    v0 = w.store.model_version(ORG[0], ORG[1], MODEL)
    run_engine(eng, w.store, NOW + DT, dt=DT)              # nothing to do: unchanged
    assert w.model() is m0 and w.store.model_version(ORG[0], ORG[1], MODEL) == v0
    rng = np.random.default_rng(11)
    for k in range(2, 5):
        new_entity_tick(w.store, S, "10.0.2.99", ROLES[2], NOW + k * DT, rng)
        run_engine(eng, w.store, NOW + k * DT, dt=DT)
    assert w.model() is not m0 and w.store.model_version(ORG[0], ORG[1], MODEL) == v0 + 1


def test_helpers_level1_fallback_and_matching():
    # > 30 % noise -> average linkage cut at the largest gap
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(0, 0.3, (5, 2)), rng.normal(5, 0.3, (5, 2))]
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    lab, _ = level1(D / D.max())
    assert len(set(lab[:5])) == 1 and len(set(lab[5:])) == 1 and lab[0] != lab[5]
    assert level1(np.zeros((1, 1)))[0].tolist() == [0]
    ids, J, _ = match_ids([{"a", "b", "c"}, {"x", "y"}], {"r1": {"a", "b"}, "r2": {"q"}})
    assert ids == ["r1", None] and J[0, 0] == 2 / 3


def test_perf_40_entities():
    roles = [dict(r, name=f"{r['name']}{k}") for k in range(3) for r in ROLES]
    w = World(n_per_role=5, roles=roles[:8], outlier=False)        # 40 entities
    eng = PeerGroupEngine()
    run_engine(eng, w.store, NOW, dt=DT)                           # warm (imports)
    t0 = time.perf_counter()
    run_engine(eng, w.store, NOW + H6, dt=DT)
    refit_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    for k in range(1, 11):
        run_engine(eng, w.store, NOW + H6 + k * DT, dt=DT)
    tick_ms = (time.perf_counter() - t0) * 100.0
    assert len(w.model()["assign"]) == 40
    assert refit_ms < 400.0, refit_ms          # generous (≈ 15-40 ms measured)
    assert tick_ms < 20.0, tick_ms             # cold path only


def test_clustered_entity_whose_evidence_decayed_is_not_new():
    from app.engines.behavior.lib import m_baseline as MB
    w = World(outlier=False)
    eng = PeerGroupEngine()
    mc = fit(w, eng)
    back = ip(2, 3)
    rid = role_of(mc, S, back)
    m = MB.new_model()
    m.update(fmt=MB.FMT, tier="entity", version=0)
    w.store.put_model(S, back, "model.baseline", m)        # n_eff 0: young again
    rng = np.random.default_rng(12)
    for k in range(1, 5):
        new_entity_tick(w.store, S, back, ROLES[2], NOW + k * DT, rng)
        run_engine(eng, w.store, NOW + k * DT, dt=DT)
    assert not events(w.store, "new_entity_matched", back)
    assert not events(w.store, "new_entity_unmatched", back)
    assert role_of(w.model(), S, back) == rid
    mc = fit(w, eng, NOW + H6)                             # a refit keeps its role
    assert role_of(mc, S, back) == rid and not mc["assign"][f"{S}|{back}"].get("provisional")


def test_canonical_refit_stride_is_wall_clock_at_60s():
    """Round 4 (evaluator): in canonical grain mode the 16-tick stride counts
    ticks of at least one Q grain (16 x max(dt, 900 s) = 4 h), so a 900 -> 60 s
    switch does not make B02 re-cluster every 16 minutes (pack E seed 0: 90
    refits a day, consecutive-refit ARI min 0.68, a class key dissolving and
    re-forming within the hour). Tick mode keeps the raw tick stride (the
    test above)."""
    w = World(outlier=False)
    eng = PeerGroupEngine()
    canon = {"grain_mode": "canonical"}
    fit(w, eng, config=canon)
    last = w.model()["_state"]["last_refit"]
    t = NOW
    for k in range(4 * REFIT_TICKS):                       # 64 minutes at 60 s
        t += 60.0
        run_engine(eng, w.store, t, dt=60.0, config=canon)
    assert w.model()["_state"]["last_refit"] == last
    t = last + REFIT_TICKS * 900.0                         # 4 h after the last refit
    run_engine(eng, w.store, t, dt=60.0, config=canon)
    assert w.model()["_state"]["last_refit"] == t
