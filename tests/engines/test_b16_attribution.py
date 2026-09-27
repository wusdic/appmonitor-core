"""B16 AttributionEngine (docs/lib3/engines.md '## B16'): spec unit tests (a)-(e).

The world: four enrolled IPs of one role class, each with its own ABSOLUTE
persona (feature.vec mean, HTTP templates, client stacks), the real lib
sketch of its tokens and stacks, and hand-built model.vocab / model.client
(entity + system tiers, the m_vocab / m_client layouts). model.identity is
fitted here by a mini-B15 on the same window-vector recipe the engine uses
(PCA + within-entity whitening, so within-entity scatter is I).

Spec test mapping:
  (a) test_a_impersonation_with_concurrent_owner_is_high,
      test_a_impersonation_without_concurrent_owner_is_medium
  (b) test_b_own_rows_200_ticks_no_event
  (c) test_c_unseen_distribution_is_unknown_identity (+ same-class MEDIUM)
  (d) test_d_client_stack_change_alone_is_capped
  (e) test_e_consumes_excludes_self_normalised_z
Edge cases live in test_b16_attribution_edges.py (they import World).
"""
from __future__ import annotations

import inspect
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from helpers import DT, T0, add_obs_tick, make_store, put_model, run_engine

from app.engines.behavior import attribution as AT
from app.engines.behavior.attribution import AttributionEngine
from app.engines.behavior.lib import emit
from app.engines.behavior.lib import sketch as SK
from app.engines.behavior.lib import timebins as TB

S = "erp"
A, B, C, D = "10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"
ENTS = (A, B, C, D)
ROLE = "r1"
DIM_PCA = 8


class Persona:
    def __init__(self, rng: np.random.Generator, base: np.ndarray, idx: int,
                 shared_tmpl: Sequence[str]) -> None:
        mask = np.zeros(52)
        mask[rng.choice(52, 14, replace=False)] = 1.0
        self.mean = base + rng.normal(0.0, 1.2, 52) * mask
        self.tmpl = [f"GET erp.corp /p{idx}/r{k}|2xx" for k in range(6)] + list(shared_tmpl)
        self.tw = rng.dirichlet(np.ones(len(self.tmpl)) * 2.0)
        self.stacks = [f"j{idx}{k:02d}|ua{idx}/{100 + k}|windows|128|w64" for k in range(2)]
        self.sw = np.array([0.7, 0.3])

    @staticmethod
    def mixture(parts: Sequence["Persona"]) -> "Persona":
        p = object.__new__(Persona)
        p.mean = np.mean([x.mean for x in parts], axis=0)
        p.tmpl = sorted({t for x in parts for t in x.tmpl})
        p.tw = np.ones(len(p.tmpl)) / len(p.tmpl)
        p.stacks = [s for x in parts for s in x.stacks]
        p.sw = np.ones(len(p.stacks)) / len(p.stacks)
        return p


class World:
    """Synthetic system: personas, model fitting and per-tick feeds."""

    def __init__(self, seed: int = 7, with_class: bool = True) -> None:
        self.rng = np.random.default_rng(seed)
        base = self.rng.normal(0.0, 1.0, 52)
        shared = [f"GET erp.corp /home/{k}|2xx" for k in range(3)]
        self.p = {e: Persona(self.rng, base, i, shared) for i, e in enumerate(ENTS)}
        self.store = make_store()
        self.eng = AttributionEngine()
        self.with_class = with_class
        self.now = T0
        self.alien = Persona(np.random.default_rng(999), base + 6.0, 9, [])

    # -------------------------------------------------------------- one tick
    def draw(self, p: Persona, rng: Optional[np.random.Generator] = None, n_req: int = 30,
             stacks: Optional[Dict[str, float]] = None):
        rng = rng or self.rng
        vec = p.mean + rng.normal(0.0, 0.3, 52)
        cnt = rng.multinomial(n_req, p.tw)
        toks = {t: float(c) for t, c in zip(p.tmpl, cnt) if c > 0}
        if stacks is None:
            sc = rng.multinomial(n_req, p.sw)
            stacks = {t: float(c) for t, c in zip(p.stacks, sc) if c > 0}
        sk = SK.sketch_vector({"act.tokens": toks,
                               "client.stack_set": {t: {"n": n} for t, n in stacks.items()}})
        return vec, toks, stacks, sk

    def feed(self, e: str, now: float, p: Optional[Persona], dt: float = DT,
             vec: Optional[np.ndarray] = None, stacks: Optional[Dict[str, float]] = None,
             n_req: int = 30) -> None:
        st = self.store
        w = int(dt)
        if p is None:                          # idle tick: B01 still writes a row
            st.add_vec(S, e, "feature.vec", now, np.zeros(52, np.float32), window_s=w)
            st.add_vec(S, e, "feature.sketch", now, np.zeros(80, np.float32), window_s=w)
            st.add_vec(S, e, "feature.active", now, np.array([0.0], np.float32), window_s=w)
            return
        v, toks, stk, sk = self.draw(p, n_req=n_req, stacks=stacks)
        if vec is not None:
            v = vec
        st.add_vec(S, e, "feature.vec", now, np.asarray(v, np.float32), window_s=w)
        st.add_vec(S, e, "feature.sketch", now, np.asarray(sk, np.float32), window_s=w)
        st.add_vec(S, e, "feature.active", now, np.array([1.0], np.float32), window_s=w)
        add_obs_tick(st, S, e, now, {"act.tokens": toks, "client.stack_set":
                                     {t: {"n": n} for t, n in stk.items()}})

    def step(self, feeds: Dict[str, Optional[Persona]], dt: float = DT,
             training: bool = False, **kw) -> int:
        self.now += dt
        for e in ENTS:
            if e in feeds:
                self.feed(e, self.now, feeds[e], dt, **kw.get(e, {}))
            else:
                self.feed(e, self.now, self.p[e], dt)
        return run_engine(self.eng, self.store, self.now, training=training, dt=dt)

    # ------------------------------------------------------------ mini-B15
    def fit(self, n_win: int = 50, enrolled: Sequence[str] = ENTS) -> Dict:
        rng = np.random.default_rng(11)
        X, lab = [], []
        for e in enrolled:
            p = self.p[e]
            for w in range(n_win):
                rows, sks = [], []
                for _ in range(AT.K):
                    v, _, _, sk = self.draw(p, rng)
                    rows.append(v)
                    sks.append(sk)
                center = T0 - 86400.0 * 3 + w * 3700.0
                tc = TB.tctx_from_config(center, {}, DT)
                X.append(AT.window_vector(np.vstack(rows), np.vstack(sks), None,
                                          float(tc["hour_local"]), tc["day_type"] == "workday",
                                          None))
                lab.append(e)
        X = np.vstack(X)
        lab = np.array(lab)
        mean = X.mean(axis=0)
        scale = np.maximum(X.std(axis=0), 1e-3)
        Xs = (X - mean) / scale
        _, _, Vt = np.linalg.svd(Xs, full_matrices=False)
        comp = Vt[:DIM_PCA]
        Y = Xs @ comp.T
        Sw = np.zeros((DIM_PCA, DIM_PCA))
        mu = {}
        for e in enrolled:
            Ye = Y[lab == e]
            mu[e] = Ye.mean(axis=0)
            R = Ye - mu[e]
            Sw += R.T @ R
        Sw /= len(Y) - len(enrolled)
        lam, U = np.linalg.eigh(Sw)
        W = U @ np.diag(1.0 / np.sqrt(np.maximum(lam, 1e-3 * lam.max())))
        means = {e: (mu[e] @ W).tolist() for e in enrolled}
        model = {"pca": {"mean": mean.tolist(), "scale": scale.tolist(),
                         "components": comp.tolist()},
                 "W": W.tolist(), "means": means, "class_means": {},
                 "llr_calib": {m: [1.0, 0.0] for m in AT.MODALITIES},
                 "confusion": {e: {j: 0.0 for j in enrolled if j != e} for e in enrolled},
                 "anonymity_sets": [], "version": 1}
        if self.with_class:
            model["class_means"] = {f"class:{ROLE}": np.mean([means[e] for e in enrolled],
                                                             axis=0).tolist()}
        return model

    def setup(self, enrolled: Sequence[str] = ENTS, vocab: bool = True, client: bool = True,
              warm: int = 6) -> "World":
        st = self.store
        put_model(st, S, "__system__", "model.identity", self.fit(enrolled=enrolled), version=1)
        if self.with_class:
            put_model(st, "__org__", "__org__", "model.class",
                      {"assign": {f"{S}|{e}": {"role": ROLE, "prob": 0.9} for e in ENTS},
                       "roles": {ROLE: {"name": "clerks"}}, "version": 1})
        if vocab:
            self._vocab_models()
        if client:
            self._client_models()
        for _ in range(warm):
            self.step({})
        return self

    def _vocab_models(self) -> None:
        st, t = self.store, self.now
        H = 30 * 86400.0
        sys_d: Dict[str, list] = {}
        for e in ENTS:
            p = self.p[e]
            dims = {}
            for tok, w in zip(p.tmpl, p.tw):
                key = tok.split("|", 1)[0]
                dims[key] = [float(w) * 3000.0, 100, t - 86400, t]
                x = sys_d.setdefault(key, [0.0, 0.0, t - 86400, t, 0.0, 100])
                x[0] += float(w) * 3000.0
                x[1] += 1.0
                x[4] += 1.0
            put_model(st, S, e, "model.vocab", {
                "fmt": 1, "kind": "entity", "version": 1, "ts": t, "class_key": None,
                "state": {"H": H, "t_ref": t, "g": 0.0, "dims": {"tmpl": dims},
                          "N": {"tmpl": 3000.0}, "N1": {"tmpl": 0.0}, "clock": t,
                          "first": t - 86400, "days": {}, "jsd": {}, "n_rows": 300}})
        put_model(st, S, "__system__", "model.vocab", {
            "fmt": 1, "kind": "system", "version": 1, "built": t, "H": H,
            "dims": {"tmpl": sys_d}, "N": {"tmpl": 3000.0 * len(ENTS)}, "N1": {"tmpl": 0.0},
            "n_ent": float(len(ENTS)), "members": len(ENTS), "sightings": {}})

    def _client_models(self) -> None:
        st, t = self.store, self.now
        H = 14 * 86400.0
        sys_c: Dict[str, float] = {}
        for e in ENTS:
            p = self.p[e]
            c = {tok: [float(w) * 200.0, 200, t - 86400, t, 0] for tok, w in zip(p.stacks, p.sw)}
            for tok, w in zip(p.stacks, p.sw):
                sys_c[tok] = sys_c.get(tok, 0.0) + float(w) * 200.0
            put_model(st, S, e, "model.client", {
                "fmt": 1, "kind": "entity", "version": 1, "ts": t, "class_key": None,
                "state": {"H": H, "clock": t, "c": c, "N": 200.0, "gaps": {}, "n_rows": 200}})
        put_model(st, S, "__system__", "model.client", {
            "fmt": 1, "kind": "system", "version": 1, "built": t, "H": H, "c": sys_c,
            "N": 200.0 * len(ENTS), "n_ent": len(ENTS), "classes": {}, "cooc": {},
            "ua_N": {}, "acq": {}, "known": {}})

    # --------------------------------------------------------------- reading
    def events(self, e: Optional[str] = None, kinds=("identity_mismatch", "unknown_identity")):
        return self.store.events(system=S, entity=e, kinds=list(kinds), limit=1000)

    def idrow(self, e: str) -> Dict:
        m = self.store.latest_derived(S, e, "behavior.id")
        return dict(m.value) if m is not None else {}


# ================================================================== (a)
def test_a_impersonation_with_concurrent_owner_is_high():
    w = World().setup()
    for _ in range(6):                        # A's persona on B's key; A keeps working
        w.step({B: w.p[A]})
    ev = w.events(B, kinds=("identity_mismatch",))
    assert ev, w.idrow(B)
    e = ev[0]
    assert e.extra["looks_like"] == A
    assert e.extra["posterior"] >= 0.9
    assert e.severity.value == "high" and e.extra["concurrent"] is True
    assert e.axes == ["identity"]
    # the owner of the persona is not accused, nobody is 'unknown'
    assert not w.events(A)
    assert not w.events(kinds=("unknown_identity",))
    # the score is written through emit and reads as anomalous
    sc = emit.read_row(w.store, S, B, emit.SCORE, w.now)["identity"]
    assert sc > 1.0
    assert w.idrow(B)["best_other"]["entity"] == A


def test_a_impersonation_without_concurrent_owner_is_medium():
    w = World().setup()
    for _ in range(6):
        w.step({B: w.p[A], A: None})          # A silent: a swap, not impersonation
    ev = w.events(B, kinds=("identity_mismatch",))
    assert ev
    assert ev[0].extra["looks_like"] == A
    assert ev[0].severity.value == "medium"


# ================================================================== (b)
def test_b_own_rows_200_ticks_no_event():
    w = World().setup()
    for _ in range(200):
        w.step({})
    assert not w.events()
    row = w.idrow(B)
    assert row["posterior_self"] > 0.9
    assert math.isfinite(row["h"]) and row["cusum_other"] < row["h"]


# ================================================================== (c)
def test_c_unseen_distribution_is_unknown_identity():
    w = World().setup()
    for _ in range(6):
        w.step({C: w.alien})
    ev = w.events(C, kinds=("unknown_identity",))
    assert ev, w.idrow(C)
    assert ev[0].severity.value == "high"                 # unlike anyone, class included
    assert ev[0].extra["reason"] == "unlike_anyone"
    assert not w.events(kinds=("identity_mismatch",))


def test_c_same_class_different_individual_is_medium():
    w = World().setup()
    mix = Persona.mixture([w.p[A], w.p[B], w.p[D]])
    for _ in range(8):
        w.step({C: mix})
    ev = w.events(C, kinds=("unknown_identity",))
    assert ev, w.idrow(C)
    assert ev[0].extra["reason"] == "same_class_different_individual"
    assert ev[0].severity.value == "medium"


# ================================================================== (d)
def test_d_client_stack_change_alone_is_capped():
    w = World().setup()
    a_stacks = {t: 200.0 * float(x) for t, x in zip(w.p[A].stacks, w.p[A].sw)}
    for _ in range(12):                       # B's own behaviour on A's client stacks
        w.step({B: w.p[B]}, B={"stacks": a_stacks, "n_req": 30})
    assert not w.events(kinds=("identity_mismatch",))
    att = w.store.profile(S, B).extra["attribution"]
    ll = {c["id"]: c["llr"] for c in att["candidates"]}
    if A in ll:                               # the raw client evidence exceeds the cap
        assert ll[A]["client"] > AT.CAP
    assert w.idrow(B)["posterior_self"] > 0.5


# ================================================================== (e)
def test_e_consumes_excludes_self_normalised_z():
    cons = AttributionEngine.consumes
    for bad in ("behavior.z", "behavior.zr", "behavior.zi", "behavior.pf"):
        assert bad not in cons
    assert not any(c.startswith("behavior.z") for c in cons)
    src = inspect.getsource(AT)
    for bad in ('"behavior.z"', '"behavior.zr"', '"behavior.zi"'):
        assert bad not in src
    for need in ("feature.vec", "feature.sketch", "behavior.timing", "model.identity"):
        assert need in cons
