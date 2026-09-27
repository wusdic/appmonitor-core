"""B17 EntityLinkEngine: edge cases (empty store, silent entities, NaN inputs,
training mode, cadence 900 -> 60, the one-to-one margin, retraction scope,
actor chains through the engine, restart), the static absolute-identity
check, the m_link accessors / maths, and a perf check. The world is the one
of test_b17_entity_link."""
from __future__ import annotations

import inspect
import math
import time

import numpy as np
import pytest

from helpers import DT, T0, make_store, run_engine

from app.engines.behavior import entity_link as EL
from app.engines.behavior.entity_link import EntityLinkEngine, mixture_test, stacks_overlap
from app.engines.behavior.lib import m_link as ML
from app.engines.behavior.lib import stack as STK

from test_b17_entity_link import A, B, C, D1, D2, ENTS, S, World, stack_tok


# ------------------------------------------------------------- empty / silent
def test_empty_store_and_no_entities():
    st, eng = make_store(), EntityLinkEngine()
    assert run_engine(eng, st, T0) == 0
    assert run_engine(eng, st, T0 + DT, training=True) == 0
    assert st.get_model(S, "__system__", ML.MODEL) is None
    assert not st.events()


def test_silent_and_established_entities_trigger_nothing():
    w = World().setup(warm=4)
    for _ in range(30):                      # D1 idle throughout, the others as usual
        w.step({D1: None})
    assert not w.events()
    lm = w.link()
    assert ML.version(lm) == 0 and not ML.links(lm) and not lm["pending"]
    # an idle tick of a young entity is not an evaluation (absence is data)
    w.step({B: None})
    assert B not in w.link()["pending"]


def test_bad_window_raises():
    st, eng = make_store(), EntityLinkEngine()
    from app.core.engine import Context
    with pytest.raises(ValueError):
        eng.run(Context(store=st, now=T0, window_s=0.0, config={"strict": True}))


# ------------------------------------------------------------------ NaN
def test_nan_absolute_rows_are_not_evidence():
    w = World().setup()
    nan_vec = np.full(52, np.nan)
    for _ in range(6):                        # B01 wrote an all-NaN row; A's stack, no tokens
        w.step({A: None, B: w.p[A]}, extra={B: {"vec": nan_vec}})
    assert not w.events(kinds=("entity_resolution",))
    rows = w.link()["pending"][B]["rows"]
    r = rows[A]
    assert math.isfinite(r["device"]) and r["device"] > 0      # the device alone matches ...
    assert r["lo"] < ML.LINK_LO or not r["pos"]                 # ... and never links alone
    for rr in rows.values():
        assert all(not isinstance(v, float) or not math.isinf(v) for v in rr.values())


def test_nan_and_short_zi_rows_do_not_flag_shared():
    res = mixture_test(np.full((80, 52), np.nan), np.zeros(80))
    assert math.isnan(res["delta"])
    res = mixture_test(np.zeros((10, 52)), np.zeros(10))
    assert math.isnan(res["delta"])


# ------------------------------------------------------------- training
def test_training_mode_links_but_emits_nothing():
    w = World().setup()
    for _ in range(6):
        w.step({A: None, B: w.p[A], C: w.newp}, training=True)
    assert not w.store.events()
    lm = w.link()
    assert [(lk["from"], lk["to"]) for lk in ML.links(lm)] == [(A, B)]
    assert w.continuity(B)["linked_from"] == A


# ------------------------------------------------------------- cadence
def test_cadence_switch_900_to_60_links_within_4_active_ticks():
    w = World().setup(warm=8)                 # history at 900 s
    ts_b = []
    for _ in range(6):                        # the move happens under 60-s ticks
        w.step({A: None, B: w.p[A], C: w.newp}, dt=60.0)
        ts_b.append(w.now)
    ev = w.events(B, kinds=("entity_resolution",))
    assert len(ev) == 1 and ev[0].ts <= ts_b[3]
    assert ev[0].extra["linked_from"] == A and ev[0].extra["conf"] >= 0.9
    assert ev[0].extra["terms"]["time"] > -0.01          # gap of one 60-s tick in seconds
    assert not w.events(C)


# ------------------------------------------------------ one-to-one margin
def test_two_indistinguishable_predecessors_are_not_linked():
    w = World()
    w.p[D2] = w.p[A]                          # D2 is a clone of A (an anonymity set)
    w.setup()
    for _ in range(8):
        w.step({A: None, D2: None, B: w.p[A]})
    assert not w.events(kinds=("entity_resolution", "identity_moved"))
    rows = w.link()["pending"][B]["rows"]
    assert rows[A]["lo"] >= ML.LINK_LO and rows[D2]["lo"] >= ML.LINK_LO
    assert abs(rows[A]["lo"] - rows[D2]["lo"]) < ML.LINK_MARGIN


def test_evaluation_stops_after_8_active_ticks():
    w = World().setup()
    for _ in range(10):                       # C resembles nobody silent: never linked
        w.step({A: None, C: w.newp})
    p = w.link()["pending"][C]
    assert p["done"] and p["n"] == EL.EVAL_TICKS
    assert not w.events(kinds=("entity_resolution",))


# ------------------------------------------------------------- retraction
def test_return_of_a_after_b_went_quiet_does_not_retract():
    w = World().setup()
    for _ in range(4):
        w.step({A: None, B: w.p[A]})
    assert ML.links(w.link())
    for _ in range(6):                        # B leaves (> 1 h) before A is back
        w.step({A: None}, absent=(B,))
    w.step({})                                # A active again, B long gone
    assert not w.events(kinds=("link_retracted",))
    assert ML.links(w.link())


def test_actor_chain_through_the_engine():
    w = World().setup()
    E2 = "10.0.0.150"
    for _ in range(4):
        w.step({A: None, B: w.p[A]})
    for _ in range(4):                        # the actor hops again within 24 h
        w.step({A: None, B: None, E2: w.p[A]})
    lm = w.link()
    pairs = [(lk["from"], lk["to"]) for lk in ML.links(lm)]
    assert pairs == [(A, B), (B, E2)]
    assert ML.version(lm) == 2
    acts = ML.actors(lm)
    assert len(acts) == 1 and acts[0]["members"] == [A, B, E2]
    c = w.continuity(E2)
    assert c["continuity_id"] == f"cid:{A}" and c["linked_from"] == B
    assert c["aliases"] == sorted([A, B])
    assert w.continuity(B)["linked_to"] == E2 and w.continuity(B)["linked_from"] == A


def test_restart_resumes_from_the_model():
    w = World().setup()
    w.step({A: None, B: w.p[A]})
    w.eng = EntityLinkEngine()                # caches lost; bookkeeping is in model.link
    for _ in range(3):
        w.step({A: None, B: w.p[A]})
    assert len(w.events(B, kinds=("entity_resolution",))) == 1


# ----------------------------------------------------------------- static
def test_linking_path_is_absolute_never_self_normalised_z():
    cons = EntityLinkEngine.consumes
    for bad in ("behavior.z", "behavior.zr", "behavior.pf"):
        assert bad not in cons
    for need in ("feature.vec", "feature.sketch", "behavior.timing", "model.identity"):
        assert need in cons
    src = inspect.getsource(EL)
    for bad in ('"behavior.z"', '"behavior.zr"'):
        assert bad not in src
    # behavior.zi is read by the shared-IP mixture test only
    for fn in (EntityLinkEngine._triggers, EntityLinkEngine._candidates,
               EntityLinkEngine._evaluate, EntityLinkEngine._score_window,
               EntityLinkEngine._resolve, EntityLinkEngine._impersonation,
               EntityLinkEngine._retract, EL._Sys):
        code = inspect.getsource(fn)
        assert "ZI" not in code and "behavior.z" not in code


# ------------------------------------------------------------- m_link maths
def _lk(a, b, ts, **kw):
    d = {"id": f"{a}>{b}@{int(ts)}", "from": a, "to": b, "ts": ts, "status": "active",
         "retracted": False}
    d.update(kw)
    return d


def test_m_link_accessors_actors_and_continuity():
    t = T0
    lks = [_lk("a", "b", t), _lk("b", "c", t + 3600), _lk("c", "d", t + 3 * 86400),
           _lk("x", "y", t + 10, retracted=True, status="retracted", rollback_to=t + 10,
               retracted_ts=t + 99)]
    m = {"fmt": 1, "version": 4, "links": lks, "actors": ML.build_actors(lks),
         "shared": {"y": {"flag": True}}, "pending": {}}
    acts = ML.actors(m)
    assert [a["members"] for a in acts] == [["a", "b", "c"], ["c", "d"]]
    assert ML.actor_of(m, "d")["members"] == ["c", "d"]
    assert ML.actor_members(m, "zz") == ["zz"]
    assert ML.aliases(m, "a") == ["b", "c", "d"] and ML.chain(m, "d") == ["a", "b", "c", "d"]
    assert ML.continuity_id(m, "c") == "cid:a" and ML.continuity_id(m, "x") is None
    assert ML.linked_from(m, "c") == "b" and ML.linked_to(m, "c") == "d"
    assert ML.linked_from(m, "y") is None                     # retracted
    assert ML.seeds_for(m, "b") == [("a", t)] and ML.seeds_for(m, "y") == []
    rt = ML.retractions(m)
    assert len(rt) == 1 and rt[0]["rollback_to"] == t + 10
    assert ML.retractions(m, since=t + 100) == []
    c = ML.continuity(m, "y")
    assert c == {"continuity_id": None, "aliases": [], "linked_from": None, "linked_to": None,
                 "entity_kind": "ip-class", "shared_ip": True}
    d = ML.descriptors(m, "b")
    assert d["link_in"]["from"] == "a" and d["link_out"]["to"] == "c" and d["actor"]
    assert ML.version(None) == 0 and ML.links(None) == [] and ML.actors(None) == []
    # lib/gating reads the same retraction flag
    from app.engines.behavior.lib import gating as G
    assert G._retracted(lks[3]) and not G._retracted(lks[0])


def test_m_link_priors_and_device_llr():
    scopes = ML.parse_scopes(["10.1.0.0/16", {"cidr": "10.2.0.0/16", "lease_s": 3600},
                              "garbage", {"cidr": None}])
    assert len(scopes) == 2
    assert ML.topology_prior("10.0.0.1", "10.0.0.200") == ML.TOPO_SAME
    assert ML.topology_prior("10.0.0.1", "10.0.1.2") == ML.TOPO_OTHER
    assert ML.topology_prior("10.1.0.1", "10.1.9.2", scopes) == ML.TOPO_SAME
    assert ML.topology_prior("host-a", "10.0.0.1") == 0.0
    assert ML.lease_for("10.2.0.1", "10.2.3.4", scopes) == 3600.0
    assert ML.lease_for("10.0.0.1", "10.0.0.2", scopes) == ML.DEFAULT_LEASE_S
    assert ML.time_prior(0.0, 3600) == 0.0 and ML.time_prior(7200, 3600) == pytest.approx(-2.0)
    assert math.isnan(ML.time_prior(math.nan))
    assert ML.conf(5.0) == pytest.approx(0.5) and ML.conf(7.2) > 0.9
    assert ML.conf(-800.0) >= 0.0 and math.isnan(ML.conf(math.nan))
    sa = {"s1": 0.7, "s2": 0.3}
    sys_ = {"s1": 0.1, "s2": 0.05, "s3": 0.5}
    same = ML.device_llr(sa, sys_, {"s1": 7.0, "s2": 3.0})
    assert same > 1.0
    assert ML.device_llr(sa, sys_, {"new": 5.0}) == pytest.approx(math.log(ML.DEVICE_CHANGE_P))
    assert ML.device_llr(sa, sys_, {"s3": 5.0}) == pytest.approx(math.log(ML.DEVICE_CHANGE_P))
    assert math.isnan(ML.device_llr({}, sys_, {"s1": 1.0}))
    assert math.isnan(ML.device_llr(sa, sys_, {}))


def test_stack_overlap_rule():
    t1, t2 = stack_tok(1, 0), stack_tok(2, 0, os_="mac", ttl=64)
    same_dev = t1.replace("|w64", "|w32")                 # same ja3n and UA
    ev = [[STK.stack_id(t1), 0.0, 100.0, 3], [STK.stack_id(t2), 350.0, 500.0, 3]]
    assert stacks_overlap(ev, [t1, t2])                   # 250 s apart <= 5 min
    ev_far = [[STK.stack_id(t1), 0.0, 100.0, 3], [STK.stack_id(t2), 500.0, 600.0, 3]]
    assert not stacks_overlap(ev_far, [t1, t2])
    ev_same = [[STK.stack_id(t1), 0.0, 100.0, 3], [STK.stack_id(same_dev), 50.0, 90.0, 3]]
    assert not stacks_overlap(ev_same, [t1, same_dev])
    assert not stacks_overlap(None, [t1]) and not stacks_overlap([[1, 2]], [t1])


# ------------------------------------------------------------------ perf
def test_perf_steady_state_and_trigger_ticks():
    w = World().setup(warm=2)
    extra_ents = [f"10.0.1.{i}" for i in range(36)]      # 40 entities in all
    for e in extra_ents:
        w.p[e] = w.p[ENTS[len(e) % 4]]
    steady = {e: w.p[e] for e in extra_ents}
    for _ in range(3):
        w.step(steady)
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        w.step(steady)
    feed_and_run = (time.perf_counter() - t0) / n
    # engine only, steady state
    eng_t = []
    for _ in range(10):
        w.now += DT
        for e in list(ENTS) + extra_ents:
            w.feed(e, w.now, w.p[e])
        t1 = time.perf_counter()
        run_engine(w.eng, w.store, w.now)
        eng_t.append(time.perf_counter() - t1)
    assert np.median(eng_t) < 0.010, (np.median(eng_t), feed_and_run)   # spec 1 ms; generous
    # trigger ticks: a move plus a new persona
    t1 = time.perf_counter()
    for _ in range(4):
        w.step({**steady, A: None, B: w.p[A], C: w.newp})
    assert (time.perf_counter() - t1) / 4 < 0.25
    assert w.events(B, kinds=("entity_resolution",))


def test_published_model_is_never_mutated_in_place():
    import copy
    w = World().setup()
    w.step({A: None, B: w.p[A], C: w.newp})
    lm0 = w.link()
    snap = copy.deepcopy(lm0)
    for _ in range(3):                        # link, pending updates, continuity
        w.step({A: None, B: w.p[A], C: w.newp})
    assert w.link() is not lm0 and ML.links(w.link())
    assert lm0["links"] == snap["links"] and lm0["version"] == snap["version"]
    assert lm0["pending"].keys() == snap["pending"].keys()
    for b in snap["pending"]:
        assert lm0["pending"][b]["n"] == snap["pending"][b]["n"]
