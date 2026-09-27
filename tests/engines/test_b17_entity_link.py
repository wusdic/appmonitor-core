"""B17 EntityLinkEngine (docs/lib3/engines.md '## B17'): spec unit tests (a)-(d).

The world: four established IPs of one /24 (10.0.0.12 = A and three others),
each with its own ABSOLUTE persona (feature.vec mean, HTTP templates, client
stacks), the real lib sketch of its tokens and stacks, hand-built
model.vocab / model.client (entity + system tiers, the m_vocab / m_client
layouts) and a model.identity fitted by a mini-B15 on the m_identity window
recipe (standardise -> PCA -> within-entity whitening, so within-entity
scatter is I in LDA space). New IPs (B = 10.0.0.112, C = 10.0.0.114) appear
with raw data, so store.first_seen / last_seen behave as in the pipeline.

Spec test mapping:
  (a) test_a_dhcp_move_is_linked_and_new_persona_is_not
  (b) test_b_same_behaviour_other_device_is_possible_impersonation
  (c) test_c_reactivation_retracts_and_baseline_equals_unseeded_fit
  (d) test_d_two_personas_on_disjoint_overlapping_stacks_is_shared_ip
Edge cases live in test_b17_entity_link_edges.py (it imports World).
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np

from helpers import DT, T0, add_obs_tick, make_store, put_model, run_engine, set_trust

from app.engines.behavior.entity_link import EntityLinkEngine
from app.engines.behavior.lib import features as F
from app.engines.behavior.lib import m_identity as MI
from app.engines.behavior.lib import m_link as ML
from app.engines.behavior.lib import sketch as SK
from app.engines.behavior.lib import stack as STK
from app.engines.behavior.lib import timebins as TB
from app.models.schema import DerivedMetric, MetricKind

S = "erp"
A, D1, D2, D3 = "10.0.0.12", "10.0.0.2", "10.0.0.3", "10.0.0.4"
B, C = "10.0.0.112", "10.0.0.114"
ENTS = (A, D1, D2, D3)
DIM_PCA = 8
COUNT_IDX = [i for i, n in enumerate(F.FEATURE_NAMES_V2) if F.FEATURE_KIND[n] == "count"]
LINK_EVENTS = ("entity_resolution", "possible_impersonation", "shared_ip", "identity_moved",
               "link_retracted")


def stack_tok(idx: int, k: int, os_: str = "windows", ttl: int = 128) -> str:
    return f"{idx:02d}{k:02d}{'ab' * 14}|ua{idx}/{100 + k}|{os_}|{ttl}|w64"


class Persona:
    def __init__(self, rng: np.random.Generator, base: np.ndarray, idx: int,
                 shared_tmpl: Sequence[str], rate: float = 10.0) -> None:
        mask = np.zeros(52)
        mask[rng.choice(52, 14, replace=False)] = 1.0
        self.mean = base + rng.normal(0.0, 1.2, 52) * mask
        self.tmpl = [f"GET erp.corp /p{idx}/r{k}|2xx" for k in range(6)] + list(shared_tmpl)
        self.tw = rng.dirichlet(np.ones(len(self.tmpl)) * 2.0)
        self.stacks = [stack_tok(idx, k) for k in range(2)]
        self.sw = np.array([0.7, 0.3])
        self.rate = rate


class World:
    """Synthetic system: personas, model fitting and per-tick feeds."""

    def __init__(self, seed: int = 7, engine: Optional[EntityLinkEngine] = None) -> None:
        self.rng = np.random.default_rng(seed)
        base = self.rng.normal(0.0, 1.0, 52)
        shared = [f"GET erp.corp /home/{k}|2xx" for k in range(3)]
        self.p = {e: Persona(self.rng, base, i, shared, rate=5.0 + 5.0 * i)
                  for i, e in enumerate(ENTS)}
        self.newp = Persona(np.random.default_rng(99), base, 7, shared, rate=30.0)
        self.store = make_store()
        self.eng = engine or EntityLinkEngine()
        self.now = T0
        self.nat_rows: Dict[str, List] = {}

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

    def nat(self, p: Persona) -> np.ndarray:
        x = np.full(F.FEATURE_DIM, np.nan)
        x[COUNT_IDX] = self.rng.poisson(p.rate, len(COUNT_IDX)).astype(np.float64)
        return x

    def feed(self, e: str, now: float, p: Optional[Persona], dt: float = DT,
             vec: Optional[np.ndarray] = None, stacks: Optional[Dict[str, float]] = None,
             s: str = S, nat: Optional[np.ndarray] = None, extra_raw: Optional[Dict] = None
             ) -> Optional[np.ndarray]:
        st, w = self.store, int(dt)
        st.add_derived(DerivedMetric(name="feature.tctx", value=TB.tctx_from_config(now, {}, dt),
                                     ts=now, system=s, entity=e, window_s=w,
                                     kind=MetricKind.CATEGORICAL))
        set_trust(st, s, e, [now], 1.0)
        if p is None:                          # idle tick: B01 still writes a row
            st.add_vec(s, e, "feature.vec", now, np.zeros(52, np.float32), window_s=w)
            st.add_vec(s, e, "feature.sketch", now, np.zeros(80, np.float32), window_s=w)
            st.add_vec(s, e, "feature.active", now, np.array([0.0], np.float32), window_s=w)
            idle = np.full(F.FEATURE_DIM, np.nan)
            idle[COUNT_IDX] = 0.0                # stale counts are a true 0
            st.add_vec(s, e, "feature.nat", now, idle.astype(np.float32), window_s=w)
            return None
        v, toks, stk, sk = self.draw(p, stacks=stacks)
        if vec is not None:
            v = vec
        st.add_vec(s, e, "feature.vec", now, np.asarray(v, np.float32), window_s=w)
        st.add_vec(s, e, "feature.sketch", now, np.asarray(sk, np.float32), window_s=w)
        st.add_vec(s, e, "feature.active", now, np.array([1.0], np.float32), window_s=w)
        n = nat if nat is not None else self.nat(p)
        st.add_vec(s, e, "feature.nat", now, np.asarray(n, np.float32), window_s=w)
        raw = {"act.tokens": toks, "client.stack_set": {t: {"n": c} for t, c in stk.items()}}
        raw.update(extra_raw or {})
        add_obs_tick(st, s, e, now, raw)
        return n

    def step(self, feeds: Dict[str, Optional[Persona]], dt: float = DT, training: bool = False,
             extra: Optional[Dict[str, Dict]] = None, absent: Sequence[str] = (),
             engines: Sequence = ()) -> int:
        """One tick: `feeds` overrides an entity's persona (None = idle); the
        established entities behave as themselves unless listed; entities in
        `absent` get no row at all. Extra engines run after B17."""
        self.now += dt
        extra = extra or {}
        for e in list(ENTS) + [x for x in feeds if x not in ENTS]:
            if e in absent:
                continue
            p = feeds[e] if e in feeds else self.p[e]
            self.feed(e, self.now, p, dt, **extra.get(e, {}))
        n = run_engine(self.eng, self.store, self.now, training=training, dt=dt)
        for eng in engines:
            run_engine(eng, self.store, self.now, training=training, dt=dt)
        return n

    # ------------------------------------------------------------ mini-B15
    def tick_row(self, vec, sk, ts: float) -> np.ndarray:
        tc = TB.tctx_from_config(ts, {}, DT)
        return np.concatenate([vec, sk, np.full(3, np.nan), MI.clock_features(tc)])

    def fit(self, n_win: int = 50, enrolled: Sequence[str] = ENTS) -> Dict:
        rng = np.random.default_rng(11)
        X, lab = [], []
        for e in enrolled:
            p = self.p[e]
            for w in range(n_win):
                t0 = T0 - 86400.0 * 3 + w * 3700.0
                rows = []
                for k in range(MI.K_WIN):
                    v, _, _, sk = self.draw(p, rng)
                    rows.append(self.tick_row(v, sk, t0 + k * DT))
                X.append(MI.window_vector(np.vstack(rows)))
                lab.append(e)
        X = np.vstack(X).astype(np.float64)
        lab = np.array(lab)
        with warnings.catch_warnings():               # all-NaN timing columns
            warnings.simplefilter("ignore", RuntimeWarning)
            fill = np.nan_to_num(np.nanmedian(X, axis=0))
        pca = {"fill": fill.tolist(), "fill_cls": {}, "mask_cols": [], "keep": None}
        Aug = MI.augment({"pca": pca}, X)
        center = Aug.mean(axis=0)
        scale = np.maximum(Aug.std(axis=0), 1e-3)
        U = (Aug - center) / scale
        _, _, Vt = np.linalg.svd(U, full_matrices=False)
        comp = Vt[:DIM_PCA]
        Y = U @ comp.T
        Sw = np.zeros((DIM_PCA, DIM_PCA))
        for e in enrolled:
            R = Y[lab == e] - Y[lab == e].mean(axis=0)
            Sw += R.T @ R
        Sw /= len(Y) - len(enrolled)
        lam, V = np.linalg.eigh(Sw)
        Wh = V @ np.diag(1.0 / np.sqrt(np.maximum(lam, 1e-3 * lam.max())))
        P = comp.T @ Wh
        Z = U @ P
        cov = np.cov(Z.T)
        pca.update(center=center.tolist(), scale=scale.tolist(), P=P.tolist(),
                   d_pca=DIM_PCA, r=DIM_PCA)
        return {"fmt": 1, "version": 1, "fitted_ts": T0, "run": 1,
                "entities": list(enrolled), "roles": {e: None for e in enrolled},
                "pca": pca, "W": Wh.tolist(),
                "means": {e: Z[lab == e].mean(axis=0).tolist() for e in enrolled},
                "class_means": {}, "class_var": {},
                "bg": {"mu": Z.mean(axis=0).tolist(), "prec": np.linalg.inv(cov).tolist(),
                       "logdet": float(np.linalg.slogdet(cov)[1])},
                "llr_calib": {m: [1.0, 0.0] for m in MI.MODALITIES},
                "confusion": {e: {j: 0.0 for j in enrolled if j != e} for e in enrolled},
                "anonymity_sets": [], "stats": {}}

    def setup(self, warm: int = 6, identity: bool = True, engines: Sequence = ()) -> "World":
        st = self.store
        if identity:
            put_model(st, S, "__system__", "model.identity", self.fit(), version=1)
        self._vocab_models()
        self._client_models()
        for _ in range(warm):
            self.step({}, engines=engines)
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
    def events(self, e: Optional[str] = None, kinds: Sequence[str] = LINK_EVENTS):
        return self.store.events(system=S, entity=e, kinds=list(kinds), limit=1000)

    def link(self) -> Dict:
        return self.store.get_model(S, "__system__", ML.MODEL) or {}

    def continuity(self, e: str) -> Dict:
        p = self.store.profile(S, e)
        return dict(p.extra.get("continuity") or {}) if p is not None else {}


# ================================================================== (a)
def test_a_dhcp_move_is_linked_and_new_persona_is_not():
    w = World().setup()
    ts_b: List[float] = []
    for k in range(6):                        # A silent; B = A's persona and stack; C new
        w.step({A: None, B: w.p[A], C: w.newp})
        ts_b.append(w.now)
    ev = w.events(B, kinds=("entity_resolution",))
    assert len(ev) == 1, [e.extra for e in w.events()]
    e = ev[0]
    assert e.ts <= ts_b[3]                     # within 4 active ticks of B
    assert e.extra["linked_from"] == A
    assert e.score >= 0.9 and e.extra["conf"] >= 0.9
    assert e.extra["terms"]["device"] > 0 and e.extra["terms"]["behaviour"] > 0
    assert e.extra["terms"]["topology"] == ML.TOPO_SAME
    moved = w.events(A, kinds=("identity_moved",))
    assert len(moved) == 1 and moved[0].extra["linked_to"] == B
    # the negative control and the bystanders are not linked, nothing else fires
    assert not w.events(C)
    for x in (D1, D2, D3):
        assert not w.events(x)
    lm = w.link()
    act = ML.links(lm)
    assert [(lk["from"], lk["to"]) for lk in act] == [(A, B)]
    assert ML.version(lm) == 1 and w.store.model_version(S, "__system__", ML.MODEL) == 1
    assert ML.linked_from(lm, C) is None
    # continuity on both ends (contract G), aliases and the 2-member actor
    cb, ca = w.continuity(B), w.continuity(A)
    assert cb["linked_from"] == A and cb["aliases"] == [A] and cb["continuity_id"] == f"cid:{A}"
    assert ca["linked_to"] == B and ca["continuity_id"] == f"cid:{A}"
    assert cb["entity_kind"] == "ip" and cb["shared_ip"] is False
    assert ML.actor_members(lm, B) == [A, B]


# ================================================================== (b)
def test_b_same_behaviour_other_device_is_possible_impersonation():
    w = World().setup()
    other = {STK.stack_token("771,4865-4866,0-10-11,29-23,0", "curl/8.1.2", 64, 29200): 30.0}
    assert all(t not in w.p[A].stacks for t in other)
    for _ in range(6):
        w.step({A: None, B: w.p[A]}, extra={B: {"stacks": other}})
    ev = w.events(B, kinds=("possible_impersonation",))
    assert len(ev) == 1, [e.extra for e in w.events()]
    assert ev[0].extra["looks_like"] == A
    assert ev[0].extra["device_lr"] <= ML.IMPERSONATION_LR
    assert ev[0].extra["lo_behaviour"] >= ML.LINK_LO
    assert ev[0].axes == ["identity"]
    assert not w.events(kinds=("entity_resolution", "identity_moved"))
    assert not ML.links(w.link())


# ================================================================== (c)
def test_c_reactivation_retracts_and_baseline_equals_unseeded_fit():
    from app.engines.behavior.baseline import BaselineEngine
    from app.engines.behavior.lib import m_baseline as MB

    b03 = BaselineEngine()
    w = World().setup(warm=12, engines=[b03])         # A has a committed baseline
    CTL = "ctl"                                        # B's unseeded twin (same rows)
    for _ in range(10):
        nat = w.nat(w.p[A])
        w.step({A: None, B: w.p[A]}, extra={B: {"nat": nat}}, engines=[])
        w.feed(B, w.now, w.p[A], s=CTL, nat=nat)
        run_engine(b03, w.store, w.now)
    lk = ML.link_between(w.link(), A, B)
    assert lk is not None
    t_link = lk["ts"]
    cur_b = MB.true_stats(w.store.get_model(S, B, MB.MODEL)["current"])
    cur_twin = MB.true_stats(w.store.get_model(CTL, B, MB.MODEL)["current"])
    assert cur_b.sum() > cur_twin.sum() + 1.0          # B was seeded with 0.5 A
    # A reactivates while B is active: link_retracted (on B), rollback_to = t_link
    nat = w.nat(w.p[A])
    w.step({B: w.p[A]}, extra={B: {"nat": nat}})
    w.feed(B, w.now, w.p[A], s=CTL, nat=nat)
    run_engine(b03, w.store, w.now)
    ev = w.events(B, kinds=("link_retracted",))
    assert len(ev) == 1 and ev[0].extra["rollback_to"] == t_link
    lm = w.link()
    assert not ML.links(lm) and ML.version(lm) == 2
    rt = ML.retractions(lm, B)
    assert len(rt) == 1 and rt[0]["rollback_to"] == t_link and rt[0]["retracted"] is True
    assert w.continuity(B)["linked_from"] is None and w.continuity(A)["linked_to"] is None
    assert ML.actors(lm) == []
    # B28 stand-in: roll the learners back to the link time, release the rows since
    put_model(w.store, S, B, "model.control",
              {"version": 0, "rollback_to": t_link, "release": [t_link, w.now]})
    for _ in range(2):
        nat = w.nat(w.p[A])
        w.step({B: w.p[A]}, extra={B: {"nat": nat}})
        w.feed(B, w.now, w.p[A], s=CTL, nat=nat)
        run_engine(b03, w.store, w.now)
    a = MB.true_stats(w.store.get_model(S, B, MB.MODEL)["current"])
    b = MB.true_stats(w.store.get_model(CTL, B, MB.MODEL)["current"])
    scale = np.abs(b).max(axis=0) + 1e-12
    assert np.all(np.abs(a - b) <= 1e-6 * scale)
    # no second retraction, no relink of B to A (A is active again)
    assert len(w.events(B, kinds=("link_retracted",))) == 1
    assert len(w.events(B, kinds=("entity_resolution",))) == 1


# ================================================================== (d)
def shared_world(two_personas: bool = True, disjoint_stacks: bool = True, same_hours: bool = True,
                 n: int = 170, dt: float = DT):
    """One IP ('10.9.9.9' in system 'nat'): zi rows from one or two personas
    (bimodal in 6 features), per-tick tctx hours, and a stack set / episode
    list with two stacks that interleave within 5 min."""
    st, eng = make_store(), EntityLinkEngine()
    rng = np.random.default_rng(3)
    s, e = "nat", "10.9.9.9"
    mu = np.zeros((2, 52))
    mu[1, [0, 1, 2, 5, 7, 9]] = 3.0
    t1 = stack_tok(1, 0)
    t2 = stack_tok(2, 0, os_="mac", ttl=64) if disjoint_stacks else t1.replace("|w64", "|w32")
    t = T0
    for i in range(n):
        t = T0 + i * dt
        k = (i % 2) if two_personas else 0
        if not same_hours and two_personas:
            k = 0 if TB.tctx_from_config(t, {}, dt)["hour_local"] < 12 else 1
        z = mu[k] + rng.normal(0.0, 1.0, 52)
        z[40:44] = np.nan                                   # stale ratio features
        st.add_vec(s, e, "feature.active", t, np.array([1.0], np.float32), window_s=int(dt))
        st.add_vec(s, e, "behavior.zi", t, z.astype(np.float32), window_s=int(dt))
        st.add_derived(DerivedMetric(name="feature.tctx", value=TB.tctx_from_config(t, {}, dt),
                                     ts=t, system=s, entity=e, window_s=int(dt),
                                     kind=MetricKind.CATEGORICAL))
        ss = {t1: {"n": 10, "first_ts": t - 800, "last_ts": t - 10},
              t2: {"n": 10, "first_ts": t - 700, "last_ts": t - 5}}
        evs = [[STK.stack_id(t1), t - 800, t - 400, 5], [STK.stack_id(t2), t - 700, t - 300, 5]]
        add_obs_tick(st, s, e, t, {"client.stack_set": ss, "client.stack_events": evs})
        run_engine(eng, st, t, dt=dt)
    return st, eng, s, e


def test_d_two_personas_on_disjoint_overlapping_stacks_is_shared_ip():
    st, eng, s, e = shared_world()
    ev = st.events(system=s, entity=e, kinds=["shared_ip"])
    assert len(ev) == 1
    lm = st.get_model(s, "__system__", ML.MODEL)
    rec = lm["shared"][e]
    assert rec["flag"] and rec["streak"] >= 3
    assert all(x > 10.0 for x in rec["bic_delta"][-3:])
    # emitted on the 3rd positive run, not earlier
    assert ev[0].extra["runs"] == 3
    c = st.profile(s, e).extra["continuity"]
    assert c["entity_kind"] == "ip-class" and c["shared_ip"] is True
    assert ML.entity_kind(lm, e) == "ip-class"


def test_d_negative_controls_one_persona_or_one_device_or_split_hours():
    for kw in ({"two_personas": False}, {"disjoint_stacks": False}, {"same_hours": False}):
        st, _, s, e = shared_world(**kw)
        assert not st.events(system=s, entity=e, kinds=["shared_ip"]), kw
        lm = st.get_model(s, "__system__", ML.MODEL)
        assert not ML.shared_ip(lm, e)
