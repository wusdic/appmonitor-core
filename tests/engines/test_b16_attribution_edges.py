"""B16 AttributionEngine edge cases: empty / unfitted stores, idle ticks, NaN
rows, training mode, a 900 -> 60 s cadence switch, a failed B01 tick, engine
restart and same-tick re-runs, B17 continuity (shared IP, linked alias),
confusable pairs, young IPs, the slow PPM / gap modalities, exactness of the
fast paths against lib/m_identity, and a perf bound at 40 entities.
The synthetic world is test_b16_attribution.World."""
from __future__ import annotations

import math
import time

import numpy as np

from helpers import DT, T0, make_store, put_model, run_engine
from test_b16_attribution import A, B, C, D, ENTS, S, World

from app.engines.behavior import attribution as AT
from app.engines.behavior.attribution import AttributionEngine
from app.engines.behavior.lib import emit
from app.engines.behavior.lib.detectors import DETECTOR_INDEX
from app.engines.behavior.lib import m_identity as MI
from app.engines.behavior.lib import m_template as MT
from app.engines.behavior.lib import m_timing as MTI
from app.engines.behavior.lib import ppm as PPM
from app.engines.behavior.lib.classkeys import SYSTEM_KEY
from app.models.schema import AcquisitionMethod, EntityProfile, RawMetric


def _score(w: World, e: str, ts: float) -> float:
    return float(emit.read_array(w.store, S, e, emit.SCORE, ts)[DETECTOR_INDEX["identity"]])


# ============================================================== stores
def test_empty_store_and_unfitted_model_abstain():
    st = make_store()
    eng = AttributionEngine()
    assert run_engine(eng, st, T0) == 0
    w = World()
    for _ in range(3):
        assert w.step({}) == 0                   # rows but no model.identity: abstain
    assert math.isnan(_score(w, B, w.now))
    assert w.store.latest_derived(S, B, "behavior.id") is None
    put_model(w.store, S, SYSTEM_KEY, "model.identity", {"fmt": 1, "means": {}})
    assert w.step({}) == 0                       # present but not fitted


def test_idle_ticks_abstain_and_keep_the_charts():
    w = World().setup()
    w.step({B: w.p[A]})
    w.step({B: w.p[A]})
    before = w.idrow(B)
    for _ in range(3):
        w.step({B: None})                        # B silent
        assert math.isnan(_score(w, B, w.now))
        assert w.store.latest_derived(S, B, "behavior.id").ts < w.now
    w.step({B: w.p[A]})
    after = w.idrow(B)
    assert after["n_act"] == before["n_act"] + 1  # idle ticks were not counted
    assert after["n_window"] == AT.K


def test_nan_feature_rows_are_scored_not_crashed():
    w = World().setup()
    nan_vec = np.full(52, np.nan)
    for _ in range(5):
        w.step({B: w.p[B]}, extra={B: {"vec": nan_vec}})
        sc = _score(w, B, w.now)
        assert math.isfinite(sc) and sc >= 0.0
    assert math.isfinite(w.idrow(B)["posterior_self"])


# ============================================================== modes
def test_training_mode_emits_no_events_and_does_not_prime():
    w = World().setup()
    for _ in range(8):
        w.step({B: w.p[A], C: w.alien}, training=True)
    assert not w.events()
    assert w.idrow(B)["cusum_other"] < w.idrow(B)["h"]
    assert w.idrow(C)["cusum_new"] < AT.U_H
    # training ends with a full window of genuine rows; the first live ticks
    # then do not alarm on a primed chart
    for _ in range(AT.K):
        w.step({}, training=True)
    for _ in range(4):
        w.step({})
    assert not w.events()


def test_cadence_switch_900_to_60():
    w = World().setup()
    h900 = w.idrow(B)["h"]
    for _ in range(90):
        w.step({}, dt=60.0)
    assert not w.events()
    row = w.idrow(B)
    assert math.isfinite(row["h"]) and row["h"] > h900   # more windows per day
    assert row["posterior_self"] > 0.5
    assert math.isfinite(_score(w, B, w.now))
    # an impersonation at 60 s is still caught
    for _ in range(8):
        w.step({B: w.p[A]}, dt=60.0)
    ev = w.events(B, kinds=("identity_mismatch",))
    assert ev and ev[0].extra["looks_like"] == A


def test_b01_failure_writes_nan_and_degraded():
    w = World().setup()
    w.now += DT
    for e in ENTS:
        w.feed(e, w.now, w.p[e])
    w.store.put_health("behavior.feature_vector", {"ok": False, "last_error_ts": w.now})
    run_engine(w.eng, w.store, w.now)
    assert math.isnan(_score(w, B, w.now))
    deg = emit.read_dict(w.store, S, B, emit.DEGRADED, w.now)
    assert deg["identity"] == "producer_error:behavior.feature_vector"


def test_restart_resumes_the_charts_and_rerun_is_idempotent():
    w = World().setup()
    w.step({B: w.p[A]})
    w.step({B: w.p[A]})
    w.step({B: w.p[A]})
    s_before = w.idrow(B)["cusum_other"]
    # same-tick re-run: identical chart state
    run_engine(w.eng, w.store, w.now)
    assert w.idrow(B)["cusum_other"] == s_before
    # a fresh engine resumes from behavior.id and still alarms in time
    w.eng = AttributionEngine()
    for _ in range(3):
        w.step({B: w.p[A]})
    ev = w.events(B, kinds=("identity_mismatch",))
    assert ev and ev[0].extra["looks_like"] == A


# ============================================================== continuity
def _continuity(w: World, e: str, **c) -> None:
    p = w.store.profile(S, e) or EntityProfile(system=S, entity=e)
    p.extra["continuity"] = dict(c)
    w.store.put_profile(p)


def test_shared_ip_downgrades_events_to_info():
    w = World().setup()
    _continuity(w, B, shared_ip=True, entity_kind="ip-class")
    for _ in range(6):
        w.step({B: w.p[A]})
    ev = w.events(B, kinds=("identity_mismatch",))
    assert ev and all(e.severity.value == "info" for e in ev)
    assert "shared_ip" in ev[0].extra["downgraded"]


def test_linked_alias_counts_as_self():
    w = World().setup()
    _continuity(w, B, continuity_id="c1", aliases=[A], linked_from=A)
    for _ in range(6):
        w.step({B: w.p[A], A: None})             # B is A re-addressed (DHCP move)
    assert not w.events(B, kinds=("identity_mismatch",))
    assert w.idrow(B)["posterior_self"] > 0.9


def test_confusable_pair_is_info():
    w = World()
    w.setup()
    m = w.store.get_model(S, SYSTEM_KEY, "model.identity")
    m["confusion"][B][A] = 0.3
    for _ in range(6):
        w.step({B: w.p[A]})
    ev = w.events(B, kinds=("identity_mismatch",))
    assert ev and ev[0].severity.value == "info"
    assert any("confusion" in r for r in ev[0].extra["downgraded"])


def test_young_ip_unknown_is_info():
    w = World().setup(enrolled=(A, C, D))        # B is not enrolled yet
    for _ in range(6):
        w.step({B: w.alien})
    ev = w.events(B, kinds=("unknown_identity",))
    assert ev and ev[0].severity.value == "info" and ev[0].extra["reason"] == "young"
    assert not w.events(B, kinds=("identity_mismatch",))


# ============================================================== slow modalities
def _stream(e: str, now: float, tids, step: float) -> RawMetric:
    """An R2 act.stream point: one row per event, `step` s apart, 2xx."""
    n = len(tids)
    rows = np.zeros(n, dtype=MT.STREAM_DTYPE)
    rows["ts"] = now - DT + 1.0 + np.arange(n) * step
    rows["token_id"] = tids
    rows["outcome"] = 2
    return RawMetric(name="act.stream", value=rows, ts=now, system=S, entity=e,
                     method=AcquisitionMethod.PASSIVE_SPAN)


def test_seq_and_timing_modalities_favour_the_owner():
    w = World().setup()
    st = w.store
    seqs = {A: [11, 12, 13, 14], B: [21, 22, 23, 24]}
    gap = {A: 2.0, B: 40.0}
    sys_ppm = PPM.PPMModel(order=0)
    for e in (A, B):
        pm = PPM.PPMModel(order=3)
        toks = [f"id:{t}|2xx" for t in seqs[e]] * 30
        PPM.update(pm, toks, ts=w.now)
        PPM.update(sys_ppm, toks, ts=w.now)
        put_model(st, S, e, "model.seq", {"kind": "entity", "ppm": pm, "version": 1})
        tm = MTI.new_state()
        tm[MTI.bin_index([gap[e]])[0]] = 500.0
        tm[MTI.T] = w.now
        put_model(st, S, e, "model.timing", {"fmt": 1, "version": 1, "state": tm})
    put_model(st, S, SYSTEM_KEY, "model.seq", {"kind": "system", "ppm": sys_ppm})

    def tick(owner_of_b: str) -> None:
        w.now += DT
        for e in ENTS:
            w.feed(e, w.now, w.p[e] if e != B else w.p[owner_of_b])
            src = owner_of_b if e == B else e
            if src in seqs:
                st.add_raw(_stream(e, w.now, seqs[src] * 8, gap[src]))
        run_engine(w.eng, st, w.now)

    for _ in range(4):
        tick(B)
    w.eng._state[(S, B)]["profile_ts"] = -math.inf     # force a profile refresh
    tick(B)
    att = st.profile(S, B).extra["attribution"]
    ll = {c["id"]: c["llr"] for c in att["candidates"]}
    assert ll[B]["seq"] is not None and ll[B]["seq"] > 0.0
    assert ll[B]["timing"] is not None and ll[B]["timing"] > 0.0
    if A in ll and ll[A]["seq"] is not None:
        assert ll[B]["seq"] > ll[A]["seq"]
        assert ll[B]["timing"] > ll[A]["timing"]


# ============================================================== exactness
def test_fast_window_vector_is_bit_identical_to_m_identity():
    rng = np.random.default_rng(3)
    for _ in range(400):
        k = int(rng.integers(1, 6))
        R = rng.normal(0.0, float(rng.choice([1e-3, 1.0, 1e3])), (k, MI.TICK_DIM))
        R[:, :132] = R[:, :132].astype(np.float32)
        R[rng.random(R.shape) < float(rng.choice([0.0, 0.2, 0.7]))] = np.nan
        a, b = MI.window_vector(R), AT.window_vector(R)
        assert np.array_equal(a, b, equal_nan=True)


def test_cached_cheap_llrs_equal_m_identity_modality_logliks():
    w = World().setup()
    w.step({B: w.p[A]})
    sc = AT._Sys(w.store, S, w.now, MI.get(w.store, S))
    md = MI.tick_modal_data(w.store, S, B, w.now)
    bg = AT.tick_background(sc, md)
    for cand in (A, B, C, "class:r1"):
        v, c, _ = AT.tick_cheap_llrs(sc, cand, md, bg)
        ref = MI.modality_logliks(w.store, S, cand, MI.ModalData(vocab=md.vocab,
                                                                 stacks=md.stacks), w.now)
        assert math.isclose(v, ref["vocab"], rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(c, ref["client"], rel_tol=1e-9, abs_tol=1e-9)


# ============================================================== perf
def test_perf_40_entities():
    import test_b16_attribution as T
    ents = [f"10.1.0.{i}" for i in range(40)]
    old = T.ENTS
    try:
        T.ENTS = tuple(ents)
        w = T.World(seed=5)
        rng = np.random.default_rng(1)
        base = rng.normal(0.0, 1.0, 52)
        w.p = {e: T.Persona(rng, base, i, ["GET erp.corp /home/0|2xx"])
               for i, e in enumerate(ents)}
        w.setup(enrolled=ents, warm=5)
        n, spent = 6, 0.0
        for _ in range(n):
            w.now += DT
            for e in ents:
                w.feed(e, w.now, w.p[e])
            t1 = time.perf_counter()                 # count only the engine
            run_engine(w.eng, w.store, w.now)
            spent += time.perf_counter() - t1
        per_tick = spent / n
    finally:
        T.ENTS = old
    # spec: <= 10 ms per tick at 40 entities on the reference box; generous here
    print(f"B16 perf: {per_tick * 1e3:.1f} ms per tick at 40 entities")
    assert per_tick < 0.25, per_tick
