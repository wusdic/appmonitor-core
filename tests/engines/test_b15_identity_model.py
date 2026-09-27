"""B15 IdentityModelEngine (docs/lib3/engines.md '## B15'): spec unit test plus
edge cases. The feed writes ABSOLUTE rows only (feature.vec, feature.sketch,
feature.active, feature.tctx, behavior.timing), exactly as B01 / B11 store
them; identity must never look at behavior.z / zr / zi.

Spec test mapping:
  6 synthetic entities (4 distinct: means 1.5 within-sd apart in 5 dims plus
  distinct sketches; 2 identical)
    -> test_spec_distinct_entities_are_identified
    -> test_spec_identical_pair_is_confusable
  static check (consumes excludes behavior.z / behavior.zi, known correction:
  zr too, and the source never reads them)  -> test_static_absolute_inputs_only
"""
from __future__ import annotations

import inspect
import math
import time

import numpy as np
import pytest

from helpers import DT, T0, make_store, make_tctx, put_model, run_engine

from app.core.engine import Context
from app.engines.behavior import identity_model as IM
from app.engines.behavior.identity_model import IdentityModelEngine
from app.engines.behavior.lib import m_identity as MI
from app.models.schema import DerivedMetric, MetricKind

S = "erp"
DISTINCT = ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"]
TWINS = ["10.0.0.8", "10.0.0.9"]


# ------------------------------------------------------------------ fixtures
def _unit_blocks(v: np.ndarray) -> np.ndarray:
    out = v.copy()
    for b in range(5):
        blk = out[16 * b:16 * (b + 1)]
        n = np.linalg.norm(blk)
        if n > 0:
            out[16 * b:16 * (b + 1)] = blk / n
    return out


class Persona:
    """Absolute feature row generator: vec ~ N(mu, 1) per dim, sketch = unit
    blocks of (u + 0.15 noise)."""

    def __init__(self, mu: np.ndarray, u: np.ndarray, think: float = 2.0) -> None:
        self.mu, self.u, self.think = mu, u, think

    def row(self, rng):
        vec = self.mu + rng.normal(0.0, 1.0, 52)
        sk = _unit_blocks(self.u + rng.normal(0.0, 0.15, 80))
        tim = {"B": 0.2 + rng.normal(0, 0.05), "M": 0.1 + rng.normal(0, 0.05),
               "think_mu": self.think + rng.normal(0, 0.2), "think_sigma": 1.0}
        return vec, sk, tim


def spec_personas(seed: int = 7):
    rng = np.random.default_rng(seed)
    out = {}
    for i, e in enumerate(DISTINCT):
        mu = np.zeros(52)
        mu[:5] = 1.5 * i                       # 1.5 within-sd apart in each of 5 dims
        out[e] = Persona(mu, _unit_blocks(rng.normal(0, 1, 80)))
    mu_t = np.zeros(52)
    mu_t[:5] = -1.5
    u_t = _unit_blocks(rng.normal(0, 1, 80))
    for e in TWINS:                             # identical distribution
        out[e] = Persona(mu_t.copy(), u_t.copy())
    return out


def write_tick(store, e: str, now: float, persona: Persona, rng, dt: float = DT,
               active: bool = True, s: str = S) -> None:
    w = int(dt)
    if active:
        vec, sk, tim = persona.row(rng)
    else:
        vec, sk, tim = np.zeros(52), np.zeros(80), None
    store.add_vec(s, e, "feature.vec", now, vec.astype(np.float32), window_s=w)
    store.add_vec(s, e, "feature.sketch", now, sk.astype(np.float32), window_s=w)
    store.add_vec(s, e, "feature.active", now, np.array([1.0 if active else 0.0], np.float32),
                  window_s=w)
    store.add_derived(DerivedMetric(name="feature.tctx", value=make_tctx(now, dt=dt), ts=now,
                                    system=s, entity=e, window_s=w, kind=MetricKind.CATEGORICAL))
    if tim is not None:
        store.add_derived(DerivedMetric(name="behavior.timing", value=tim, ts=now, system=s,
                                        entity=e, window_s=w, kind=MetricKind.CATEGORICAL))
    store.register_entity(s, e)


def feed(store, eng, personas, n_ticks: int, t0: float = T0, dt: float = DT, seed: int = 1,
         training: bool = True, run: bool = True):
    rng = np.random.default_rng(seed)
    now = t0
    for k in range(n_ticks):
        now = t0 + k * dt
        for e, p in personas.items():
            write_tick(store, e, now, p, rng, dt)
        if run:
            run_engine(eng, store, now, training=training, dt=dt)
    return now


@pytest.fixture(scope="module")
def spec_world():
    store = make_store()
    eng = IdentityModelEngine()
    now = feed(store, eng, spec_personas(), 176)
    c = Context(store=store, now=now, window_s=DT, training=False, config={"strict": True})
    t = time.perf_counter()
    n = eng.refit(c)
    fit_s = time.perf_counter() - t
    return store, eng, now, n, fit_s


# ------------------------------------------------------------------ spec
def test_spec_distinct_entities_are_identified(spec_world):
    store, _eng, _now, n, _ = spec_world
    assert n == 6
    model = MI.get(store, S)
    assert MI.is_fitted(model)
    for e in DISTINCT:
        st = MI.stats(model, e)
        assert st["n_windows"] >= 25
        assert st["recallK"] >= 0.95, (e, st)
        assert st["recall1"] >= 0.9, (e, st)
        assert st["eer_hard"] < 0.05, (e, st)
        assert st["separability"] > 0.85
        prof = store.profile(S, e)
        assert prof.separability == pytest.approx(st["separability"])
        assert prof.extra["identity"]["eer_hard"] == st["eer_hard"]
        assert not set(MI.confusable_with(model, e)) & set(DISTINCT)


def test_spec_identical_pair_is_confusable(spec_world):
    store, _eng, _now, _n, _ = spec_world
    model = MI.get(store, S)
    a, b = TWINS
    for x, y in ((a, b), (b, a)):
        st = MI.stats(model, x)
        assert st["eer_hard"] > 0.3, st
        assert st["separability"] < 0.4
        assert y in st["confusable_with"]
        assert y in store.profile(S, x).extra["identity"]["confusable_with"]
        assert MI.confusion(model, x, y) > 0.2
        assert MI.anonymity_set(model, x) == sorted(TWINS)
    assert sorted(TWINS) in MI.anonymity_sets(model)
    # live refit (training=False): INFO low_identifiability for the twins only
    evs = store.events(S, kinds=["low_identifiability"])
    assert sorted({ev.entity for ev in evs}) == sorted(TWINS)
    assert all(ev.severity.value == "info" for ev in evs)


def test_static_absolute_inputs_only():
    cons = set(IdentityModelEngine.consumes)
    for bad in ("behavior.z", "behavior.zi", "behavior.zr"):
        assert bad not in cons
    for mod in (IM, MI):
        src = inspect.getsource(mod)
        for bad in ('"behavior.z"', '"behavior.zi"', '"behavior.zr"', "'behavior.z'"):
            assert bad not in src
    assert {"feature.vec", "feature.sketch", "behavior.timing"} <= cons
    assert IdentityModelEngine().name == "behavior.identity_model"


# ------------------------------------------------------------------ accessors
def test_accessors_transform_topk_calibration(spec_world):
    store, _eng, now, _n, _ = spec_world
    model = MI.get(store, S)
    rng = np.random.default_rng(99)
    p = spec_personas()
    for e in DISTINCT:
        rows = []
        for _ in range(4):
            vec, sk, tim = p[e].row(rng)
            r = np.full(MI.TICK_DIM, np.nan)
            r[MI.T_VEC], r[MI.T_SK] = vec, sk
            r[MI.T_TIM] = [tim["B"], tim["M"], tim["think_mu"]]
            r[MI.T_CLK] = MI.clock_features(make_tctx(now))
            rows.append(r)
        z = MI.transform(model, MI.window_vector(np.vstack(rows)))
        assert z.shape == (model["pca"]["r"],) and np.all(np.isfinite(z))
        assert MI.top_k(model, z, 1)[0][0] == e
        assert MI.gauss_llr(model, z, e) > 0.0
        assert MI.gauss_llr(model, z, e) > MI.gauss_llr(model, z, [d for d in DISTINCT if d != e][0])
    # calibration: gauss fitted from the CV scores, others default until sampled
    a, b = MI.llr_calib(model, "gauss")
    assert a > 0.0 and math.isfinite(b)
    assert MI.calibrate(model, "gauss", 1e6) == MI.LLR_CAP
    assert MI.calibrate(model, "gauss", -1e6) == -MI.LLR_CAP
    assert math.isnan(MI.calibrate(model, "gauss", math.nan))
    assert MI.llr_calib(model, "vocab") == MI.DEFAULT_CALIB
    # descriptors / distinctive traits in natural units
    d = MI.descriptors(model, DISTINCT[3])
    assert d["distinctive"]["features"], d
    top = d["distinctive"]["features"][0]
    assert top["feature"] in ("bytes_up", "bytes_down", "flows", "http_requests", "dns_queries")
    assert top["dir"] == "higher"
    assert MI.descriptors(model, "nobody") == {}
    assert math.isnan(MI.gauss_loglik(model, np.zeros(model["pca"]["r"]), "nobody"))


def test_modality_share_every_fourth_run(spec_world):
    store, _eng, _now, _n, _ = spec_world
    model = MI.get(store, S)          # first fit (run 0) computes the drop importance
    sh = MI.modality_share(model, DISTINCT[0])
    assert set(sh) == set(MI.BLOCKS)
    tot = sum(v for v in sh.values() if v)
    assert tot == pytest.approx(1.0, abs=1e-3) or tot == 0.0
    assert model["modality_share_ts"] is not None


def test_pure_maths():
    assert MI.eer([1, 2, 3], [-3, -2, -1]) == 0.0
    rng = np.random.default_rng(0)
    g, i = rng.normal(0, 1, 4000), rng.normal(0, 1, 4000)
    assert abs(MI.eer(g, i) - 0.5) < 0.03
    assert abs(MI.eer(rng.normal(2, 1, 4000), i) - 0.1587) < 0.02     # Phi(-1)
    assert math.isnan(MI.eer([], [1.0]))
    # logistic calibration recovers the true LLR slope of two unit Gaussians at +-1
    x = np.concatenate([rng.normal(1, 1, 3000), rng.normal(-1, 1, 3000)])
    y = np.concatenate([np.ones(3000), np.zeros(3000)])
    a, b = MI.logistic_calibration(x, y)
    assert a == pytest.approx(2.0, rel=0.1) and abs(b) < 0.1
    assert MI.logistic_calibration([1.0], [1]) == MI.DEFAULT_CALIB
    assert MI.logistic_calibration(-x, y)[0] == 0.0               # never inverted
    assert MI.union_find_sets(["a", "b", "c", "d"], [("a", "b"), ("c", "b")]) == [["a", "b", "c"]]
    assert MI.bhattacharyya_diag([0], [1], [0], [1]) == pytest.approx(0.0)
    lo = MI.mcq_log_odds({"x": 50, "y": 1}, {"x": 1, "y": 50}, {"x": 1, "y": 1})
    assert lo["x"][1] > 1.96 and lo["y"][1] < -1.96
    assert MI.to_natural(0, math.log1p(3.0)) == pytest.approx(3.0)          # count -> /min
    assert MI.to_natural(14, 0.0) == pytest.approx(0.5)                     # ratio


# ------------------------------------------------------------------ edge cases
def test_empty_store():
    store = make_store()
    eng = IdentityModelEngine()
    assert run_engine(eng, store, T0) == 0
    assert store.get_model(S, "__system__", MI.MODEL) is None
    assert eng.refit(Context(store=store, now=T0, window_s=DT, config={"strict": True})) == 0


def test_silent_entity_and_single_entity():
    store = make_store()
    eng = IdentityModelEngine()
    rng = np.random.default_rng(3)
    p = spec_personas()
    for k in range(60):
        now = T0 + k * DT
        write_tick(store, DISTINCT[0], now, p[DISTINCT[0]], rng)
        write_tick(store, "10.9.9.9", now, p[DISTINCT[1]], rng, active=False)
        run_engine(eng, store, now, training=True)
    idw = store.get_model(S, "__system__", MI.IDWIN)
    assert len(idw["ents"]["10.9.9.9"]["vecs"]) == 0          # silence makes no windows
    assert len(idw["ents"][DISTINCT[0]]["vecs"]) == (60 - 4) // 4
    assert MI.get(store, S) is None                          # one identifiable entity: no metric


def test_nan_inputs_are_imputed_not_fatal():
    store = make_store()
    eng = IdentityModelEngine()
    rng = np.random.default_rng(5)
    per = spec_personas()
    for k in range(64):
        now = T0 + k * DT
        for i, (e, p) in enumerate(per.items()):
            vec, sk, _tim = p.row(rng)
            vec[20:30] = np.nan                        # never observed
            if i == 0 or (k + i) % 3 == 0:
                vec[5:10] = np.nan                     # one entity never, others sometimes
            w = int(DT)
            store.add_vec(S, e, "feature.vec", now, vec.astype(np.float32), window_s=w)
            if k % 2:                                  # sketch sometimes missing
                store.add_vec(S, e, "feature.sketch", now, sk.astype(np.float32), window_s=w)
            store.add_vec(S, e, "feature.active", now, np.ones(1, np.float32), window_s=w)
            store.register_entity(S, e)
        run_engine(eng, store, now, training=True)
    n = eng.refit(Context(store=store, now=now, window_s=DT, config={"strict": True}))
    assert n == 6
    model = MI.get(store, S)
    assert set(model["pca"]["mask_cols"]) >= set(range(5, 10))
    assert not set(model["pca"]["mask_cols"]) & set(range(20, 30))
    for e in per:
        st = MI.stats(model, e)
        assert st["eer_hard"] is not None and 0.0 <= st["eer_hard"] <= 1.0
    z = MI.transform(model, np.full(MI.RAW_DIM, np.nan))       # all-missing window
    assert np.all(np.isfinite(z))


def test_training_mode_emits_no_events():
    store = make_store()
    eng = IdentityModelEngine()
    now = feed(store, eng, spec_personas(), 60, training=True)
    assert eng.refit(Context(store=store, now=now, window_s=DT, training=True,
                             config={"strict": True})) == 6
    assert store.events(S, kinds=["low_identifiability"]) == []
    assert MI.stats(MI.get(store, S), TWINS[0])["eer_hard"] > 0.2     # still measured


def test_spec_fit_is_fast(spec_world):
    *_rest, fit_s = spec_world
    assert fit_s < 2.0


def _vocab_model(tmpl_counts):
    return {"fmt": 1, "kind": "entity", "version": 1,
            "state": {"H": 30 * 86400.0, "t_ref": math.nan, "g": 0.0,
                      "dims": {"tmpl": {v: [float(c), int(c), T0, T0] for v, c in
                                        tmpl_counts.items()}},
                      "N": {"tmpl": float(sum(tmpl_counts.values()))}, "N1": {"tmpl": 0.0}}}


def test_roles_class_identifiability_and_vocab_traits():
    store = make_store()
    roles = {DISTINCT[0]: "r1", DISTINCT[1]: "r1", DISTINCT[2]: "r2", DISTINCT[3]: "r2",
             TWINS[0]: "r3", TWINS[1]: "r3"}
    put_model(store, "__org__", "__org__", "model.class",
              {"assign": {f"{S}|{e}": {"role": r, "prob": 1.0} for e, r in roles.items()},
               "roles": {r: {"name": f"role {r}"} for r in set(roles.values())}, "version": 1})
    for e in roles:
        tm = {"GET erp.corp /orders/view/{num}": 200.0, "POST erp.corp /orders/save": 50.0}
        if e == DISTINCT[0]:
            tm["GET erp.corp /admin/users/{num}"] = 120.0
        put_model(store, S, e, "model.vocab", _vocab_model(tm))
    eng = IdentityModelEngine()
    now = feed(store, eng, spec_personas(), 64)
    assert eng.refit(Context(store=store, now=now, window_s=DT, config={"strict": True})) == 6
    model = MI.get(store, S)
    assert set(model["classes"]) == {"class:r1", "class:r2", "class:r3"}
    for ck, cs in model["classes"].items():
        assert 0.0 <= cs["identifiability"] <= 1.0 and cs["n_members"] == 2
        prof = store.profile(S, ck)
        assert prof.extra["identity"]["identifiability"] == cs["identifiability"]
        assert MI.class_mean(model, ck) is not None
        assert set(model["pca"]["fill_cls"]) == {"class:r1", "class:r2", "class:r3"}
    assert model["classes"]["class:r3"]["identifiability"] > 0.8     # twins' role is distinct
    # class candidate scoring for B16: a twin window is closer to its role than to r1
    z = MI.entity_mean(model, TWINS[0])
    assert MI.gauss_loglik(model, z, "class:r3") > MI.gauss_loglik(model, z, "class:r1")
    voc = MI.distinctive(model, DISTINCT[0])["vocab"]
    assert voc and voc[0]["value"] == "GET erp.corp /admin/users/{num}" and voc[0]["dim"] == "tmpl"
    # relative to its only role peer, DISTINCT[1] over-uses the order templates
    assert all(v["value"] != "GET erp.corp /admin/users/{num}"
               for v in MI.distinctive(model, DISTINCT[1])["vocab"])


def test_drop_importance_finds_the_identifying_block():
    store = make_store()
    rng = np.random.default_rng(21)
    per = {e: Persona(np.zeros(52), _unit_blocks(rng.normal(0, 1, 80))) for e in DISTINCT}
    eng = IdentityModelEngine()
    feed(store, eng, per, 64)                    # first fit (run 0) computes the shares
    model = MI.get(store, S)
    for e in DISTINCT:
        sh = MI.modality_share(model, e)
        assert max(sh, key=lambda k: sh[k] or 0.0) == "sketch", sh
