"""B02 PeerGroupEngine: the engines.md unit test (3 roles x 4 individuals + 1
outlier) and the cold-start typing of a new entity.

The World builder writes the committed models B02 reads, as their owners
would: model.baseline (m_baseline anchors folded with make_row / commit_many
on per-tick feature.nat rows), model.vocab (entity layout of m_vocab),
model.rhythm (m_rhythm counts), model.client (entity layout of m_client).
Individuals of one role share template FAMILIES (fixed shares), rhythm shape,
client stack and channel mix; each has its own Dirichlet(0.4) template
preferences inside the families, private paths and a +-10 % individual level.
"""
from __future__ import annotations

import zlib
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score

from helpers import T0, make_store, make_tctx, run_engine

from app.engines.behavior.lib import features as F
from app.engines.behavior.lib import m_baseline as MB
from app.engines.behavior.lib import m_class
from app.engines.behavior.lib import m_client as MC
from app.engines.behavior.lib import m_rhythm as MR
from app.engines.behavior.lib import m_vocab as MV
from app.engines.behavior.lib.classkeys import ORG
from app.engines.behavior.lib.stack import stack_token
from app.engines.behavior.peer_group import MODEL, PeerGroupEngine
from app.models.schema import AcquisitionMethod, RawMetric

S = "erp"
DT = 900.0
DAYS = 3
NOW = T0 + DAYS * 86400.0
UA = {"chrome": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
      "python": "python-requests/2.31.0", "curl": "curl/8.4.0", "tool": "SomeScanner/1.0"}

# role archetypes: families (method, host, first segment) with fixed shares,
# active local hours, UA, per-tick counts and A-components (natural units)
ROLES = [
    dict(name="interactive", fams=[("GET", "erp.corp", "orders", 0.5),
                                   ("POST", "erp.corp", "orders", 0.3),
                                   ("GET", "erp.corp", "reports", 0.2)],
         sni="erp.corp", dport="443", hours=range(9, 18), ua="chrome",
         counts=dict(http_requests=40, dns_queries=10, tls_handshakes=5, flows=20, intensity=60),
         comps=dict(timing_regularity=0.15, periodicity=0.1, path_entropy=0.8, think_time=20.0)),
    dict(name="api", fams=[("POST", "api.corp", "v1", 0.6), ("GET", "api.corp", "health", 0.4)],
         sni="api.corp", dport="8443", hours=range(0, 24), ua="python",
         counts=dict(http_requests=200, dns_queries=2, tls_handshakes=2, flows=50, intensity=210),
         comps=dict(timing_regularity=0.9, periodicity=0.85, path_entropy=0.2, think_time=1.0)),
    dict(name="batch", fams=[("GET", "files.corp", "export", 0.7), ("PUT", "files.corp", "sync", 0.3)],
         sni="files.corp", dport="873", hours=range(0, 6), ua="curl",
         counts=dict(http_requests=15, dns_queries=80, tls_handshakes=30, flows=60, intensity=120),
         comps=dict(timing_regularity=0.7, periodicity=0.6, path_entropy=0.4, think_time=3.0)),
]
OUTLIER = dict(name="odd", fams=[("GET", "odd.example", "x", 1.0)], sni="odd.example",
               dport="22", hours=range(18, 23), ua="tool",
               counts=dict(http_requests=1, dns_queries=0, tls_handshakes=0, flows=300,
                           intensity=300),
               comps=dict(timing_regularity=0.5, periodicity=0.3, path_entropy=0.5,
                          think_time=5.0))
SEGMENTS = {"orders": ["view/{num}", "list", "edit/{num}", "search", "print/{num}"],
            "reports": ["daily", "monthly", "q/{num}", "chart", "export"],
            "v1": ["sync", "items/{num}", "batch", "status", "push"],
            "health": ["ping", "ready", "live", "metrics", "info"],
            "export": ["a.csv", "b.csv", "c.csv", "d.csv", "e.csv"],
            "sync": ["part/{num}", "commit", "begin", "end", "meta"],
            "x": ["1", "2", "3", "4", "5"]}


def ip(r: int, i: int) -> str:
    return f"10.0.{r}.{10 + i}"


def local_hour(ts: float) -> int:
    return int(make_tctx(ts)["hour_local"])


def templates(spec: Dict, i: int, rng: np.random.Generator, scale: float = 1.0
              ) -> Dict[str, float]:
    """tmpl counts: fixed family shares; Dirichlet(0.4) over the family's
    shared templates plus the individual's private path."""
    out: Dict[str, float] = {}
    for m, host, seg, share in spec["fams"]:
        paths = [f"/{seg}/{p}" for p in SEGMENTS[seg]] + [f"/{seg}/private-{i}/{{num}}"]
        w = rng.dirichlet(np.full(len(paths), 0.4))
        for p, x in zip(paths, w):
            c = 1000.0 * share * x * scale
            if c > 0.5:
                out[f"{m} {host} {p}"] = c
    return out


def vocab_model(tmpl: Dict[str, float], spec: Dict, now: float) -> Dict:
    dims = {"tmpl": tmpl, "sni": {spec["sni"]: 300.0}, "dport": {spec["dport"]: 300.0}}
    return {"fmt": 1, "kind": "entity", "version": 0, "ts": now, "class_key": None,
            "state": {"H": MV.HALF_LIFE_S, "t_ref": now, "g": 0.0,
                      "dims": {d: {v: [c, max(1, int(c)), T0, now] for v, c in vals.items()}
                               for d, vals in dims.items()},
                      "N": {d: float(sum(v.values())) for d, v in dims.items()},
                      "N1": {d: 0.0 for d in dims}, "clock": now, "first": T0,
                      "days": {}, "jsd": {}, "n_rows": 100}}


def stack_of(spec: Dict) -> str:
    return stack_token("771,4865-4866,0-10-11,29-23,0", UA[spec["ua"]], 128 if spec["ua"] == "chrome" else 64,
                       65535)


def client_model(spec: Dict, now: float) -> Dict:
    tok = stack_of(spec)
    return {"fmt": 1, "kind": "entity", "version": 0, "ts": now, "class_key": None,
            "state": {"H": MC.HALF_LIFE_S, "clock": now, "c": {tok: [50.0, 200, T0, now, 0]},
                      "N": 50.0, "gaps": {}, "n_rows": 200}}


def nat_row(spec: Dict, level: float, rng: np.random.Generator, dt: float = DT,
            noise: float = 0.05) -> np.ndarray:
    nat = np.full(F.FEATURE_DIM, np.nan)
    for n, v in spec["counts"].items():
        nat[F.FEATURE_INDEX[n]] = float(np.round(v * level * dt / 900.0 * rng.lognormal(0, noise)))
    for n, v in spec["comps"].items():
        x = v * level if n == "think_time" else min(0.99, v * rng.lognormal(0, noise))
        nat[F.FEATURE_INDEX[n]] = x
    return nat


class World:
    """3 roles x 4 individuals + 1 outlier with committed models at NOW."""

    def __init__(self, seed: int = 7, n_per_role: int = 4, system: str = S,
                 roles: Sequence[Dict] = tuple(ROLES), outlier: bool = True) -> None:
        self.store = make_store()
        self.rng = np.random.default_rng(seed)
        self.truth: Dict[str, str] = {}
        self.specs: Dict[str, Dict] = {}
        self.level: Dict[str, float] = {}
        self.system = system
        ents = []
        for r, spec in enumerate(roles):
            for i in range(n_per_role):
                ents.append((ip(r, i), spec, spec["name"]))
        if outlier:
            ents.append(("10.0.9.9", OUTLIER, "odd"))
        for e, spec, truth in ents:
            self.truth[e] = truth
            self.specs[e] = spec
            self.level[e] = float(self.rng.uniform(0.9, 1.1))
        self.build(list(self.truth))

    def build(self, ents: List[str], scale: Optional[Dict[str, float]] = None) -> None:
        scale = scale or {}
        s = self.system
        ticks = [T0 + k * DT for k in range(int(DAYS * 86400 / DT))]
        tcs = [make_tctx(t) for t in ticks]
        models = {e: MB.new_model() for e in ents}
        rh = {e: MR.new_model("entity") for e in ents}
        rngs = {e: np.random.default_rng(zlib.crc32(f"{e}|1".encode())) for e in ents}
        for t, tc in zip(ticks, tcs):
            h = int(tc["hour_local"])
            act, rows = [], []
            for e in ents:
                spec = self.specs[e]
                on = h in spec["hours"]
                c48, c168 = MR.cells_of_tctx(tc)
                st = rh[e]["state"]
                st["N48"][c48] += 1.0
                st["N168"][c168] += 1.0
                if on:
                    st["A48"][c48] += 1.0
                    st["A168"][c168] += 1.0
                    lvl = self.level[e] * scale.get(e, 1.0)
                    act.append(models[e]["current"])
                    rows.append(MB.make_row(t, nat_row(spec, lvl, rngs[e]), DT, tc))
            if act:
                MB.commit_many(act, rows, [1.0] * len(act), [MB.CAP_CURRENT] * len(act),
                               [0.0] * len(act))
        for e in ents:
            spec = self.specs[e]
            m = models[e]
            m.update(fmt=MB.FMT, tier="entity", version=0, branch=0)
            self.store.put_model(s, e, "model.baseline", m)
            st = rh[e]["state"]
            st.update(t_ref=NOW, t_first=T0, t_last=NOW, n_slots=int(st["N48"].sum()))
            self.store.put_model(s, e, "model.rhythm", rh[e])
            trng = np.random.default_rng(zlib.crc32(f"{e}|2".encode()))
            tm = templates(spec, int(e.rsplit(".", 1)[1]), trng, scale.get(e, 1.0))
            self.store.put_model(s, e, "model.vocab", vocab_model(tm, spec, NOW))
            self.store.put_model(s, e, "model.client", client_model(spec, NOW))
            self.store.register_entity(s, e)

    def model(self) -> Dict:
        return self.store.get_model(ORG[0], ORG[1], MODEL) or {}


def new_entity_tick(store, s: str, e: str, spec: Dict, ts: float, rng: np.random.Generator,
                    dt: float = DT, nat: Optional[np.ndarray] = None, raw: bool = True) -> None:
    """One active tick of a brand-new entity, as B01 / R1-R3 write it."""
    row = nat_row(spec, 1.0, rng, dt) if nat is None else nat
    store.add_vec(s, e, "feature.active", ts, np.array([1.0], dtype=np.float32), window_s=int(dt))
    store.add_vec(s, e, "feature.nat", ts, row.astype(np.float32), window_s=int(dt))
    if raw:
        toks = {}
        for m, host, seg, share in spec["fams"]:
            for p in SEGMENTS[seg][:3]:
                toks[f"{m} {host} /{seg}/{p}|2xx"] = max(1.0, round(20 * share * dt / 900.0))
        tok = stack_of(spec)
        for name, val in (("act.tokens", toks), ("tls.sni_etld1_set", {spec["sni"]: 5.0}),
                          ("l4.dport_set", {spec["dport"]: 5.0}),
                          ("client.stack_set", {tok: {"n": 10, "bytes": 1000,
                                                      "first_ts": ts, "last_ts": ts}})):
            store.add_raw(RawMetric(name=name, value=val, ts=ts, system=s, entity=e,
                                    method=AcquisitionMethod.PASSIVE_SPAN))
    else:
        store.register_entity(s, e)


def events(store, kind: str, entity: Optional[str] = None) -> list:
    return [ev for ev in store.events(kinds=[kind], limit=1000)
            if entity is None or ev.entity == entity]


def role_of(mc: Dict, s: str, e: str):
    return (mc.get("assign", {}).get(f"{s}|{e}") or {}).get("role")


# ------------------------------------------------------------ shared fixture
_WORLD: Dict[str, World] = {}


def fitted_world() -> World:
    """World fitted once at NOW (read-only users share it)."""
    if "w" not in _WORLD:
        w = World()
        run_engine(PeerGroupEngine(), w.store, NOW, dt=DT)
        _WORLD["w"] = w
    return _WORLD["w"]


# ======================================================================= tests
def test_level1_ari_is_one_and_outlier_unique():
    w = fitted_world()
    mc = w.model()
    keys = sorted(w.truth)
    pred = []
    for e in keys:
        r = role_of(mc, S, e)
        pred.append(f"noise:{e}" if r in (None, "unique") else str(r))
    truth = [w.truth[e] if w.truth[e] != "odd" else "noise:odd" for e in keys]
    assert adjusted_rand_score(truth, pred) == 1.0
    assert role_of(mc, S, "10.0.9.9") == "unique"
    singletons = sum(1 for p in pred if pred.count(p) == 1)
    assert singletons / len(pred) < 0.10
    # a peer_outlier note for the outlier only
    po = events(w.store, "peer_outlier")
    assert [ev.entity for ev in po] == ["10.0.9.9"]


def test_level2_splits_individuals_inside_a_role():
    w = fitted_world()
    mc = w.model()
    n_subs = {}
    for e in w.truth:
        a = mc["assign"][f"{S}|{e}"]
        if a["role"] != "unique":
            n_subs.setdefault(a["role"], set()).add(a["sub"])
            assert a["sub"] in mc["subs"] and mc["subs"][a["sub"]]["role"] == a["role"]
            assert a["class_path"] == f"{a['super']}/{a['role']}/{a['sub']}"
    assert max(len(v) for v in n_subs.values()) >= 3


def test_super_level_names_and_soft_membership():
    w = fitted_world()
    mc = w.model()
    by_truth = {}
    for e, t in w.truth.items():
        by_truth.setdefault(t, set()).add(role_of(mc, S, e))
    rid_int = by_truth["interactive"].pop()
    rid_api = by_truth["api"].pop()
    assert mc["roles"][rid_int]["super"] == "human"
    assert mc["roles"][rid_api]["super"] == "machine"
    nm = mc["roles"][rid_int]["name"]
    assert nm.startswith("human") and "erp.corp" in nm and "h" in nm
    assert mc["roles"][rid_api]["name"].startswith("automated")
    parts = mc["roles"][rid_api]["name_parts"]
    assert parts["volume_tier"] in ("high", "medium") and parts["window"] == "24h"
    for e in w.truth:
        a = mc["assign"][f"{S}|{e}"]
        if a["role"] != "unique":
            assert a["prob"] >= 0.7
            assert a["super"] == mc["roles"][a["role"]]["super"]
    # medoid / radius are published for cold typing
    for rid in (rid_int, rid_api):
        r = mc["roles"][rid]
        assert r["medoid"] in r["members"] and r["d90"] >= 0.15 and r["desc"]["fam"]


def test_perturbation_keeps_ids():
    w = World()
    eng = PeerGroupEngine()
    run_engine(eng, w.store, NOW, dt=DT)
    before = {k: (a["role"], a["sub"]) for k, a in w.model()["assign"].items()}
    target = ip(0, 1)
    w.build([target], scale={target: 1.05})
    run_engine(eng, w.store, NOW + 6 * 3600.0 + DT, dt=DT)
    after = {k: (a["role"], a["sub"]) for k, a in w.model()["assign"].items()}
    assert {k: v[0] for k, v in before.items()} == {k: v[0] for k, v in after.items()}
    changed = sum(before[k][1] != after[k][1] for k in before)
    assert changed / len(before) <= 0.13
    assert not events(w.store, "class_split") and not events(w.store, "class_merge")
    assert not events(w.store, "class_transition")


def test_new_entity_with_role2_profile_is_matched_on_third_active_tick():
    w = World()
    eng = PeerGroupEngine()
    run_engine(eng, w.store, NOW, dt=DT)
    rid = role_of(w.model(), S, ip(1, 0))
    rng = np.random.default_rng(3)
    new = "10.0.1.99"
    for k in range(1, 4):
        ts = NOW + k * DT
        new_entity_tick(w.store, S, new, ROLES[1], ts, rng)
        run_engine(eng, w.store, ts, dt=DT)
        evs = events(w.store, "new_entity_matched", new) + events(w.store, "new_entity_unmatched", new)
        if k < 3:
            assert not evs
    assert len(evs) == 1 and evs[0].kind == "new_entity_matched"
    ev = evs[0]
    assert ev.extra["prob"] >= 0.8 and ev.extra["role"] == rid
    assert ev.severity.value == "info" and ev.ts == NOW + 3 * DT
    a = w.model()["assign"][f"{S}|{new}"]
    assert a["role"] == rid and a["provisional"] and a["prob"] >= 0.8
    # it now backs off to its role in the consumers' accessor
    assert m_class.role_id(w.store, S, new) == rid
    prof = w.store.profile(S, new)
    assert prof.archetype == a["class_path"] and prof.extra["peer_group"]["role"] == rid
    # announced once only
    ts = NOW + 4 * DT
    new_entity_tick(w.store, S, new, ROLES[1], ts, rng)
    run_engine(eng, w.store, ts, dt=DT)
    assert len(events(w.store, "new_entity_matched", new)) == 1


def test_scanner_like_new_entity_is_unmatched_medium():
    w = World(outlier=False)
    eng = PeerGroupEngine()
    run_engine(eng, w.store, NOW, dt=DT)
    rng = np.random.default_rng(5)
    new = "10.0.7.77"
    for k in range(1, 4):
        ts = NOW + k * DT
        new_entity_tick(w.store, S, new, OUTLIER, ts, rng)
        run_engine(eng, w.store, ts, dt=DT)
    evs = events(w.store, "new_entity_unmatched", new)
    assert len(evs) == 1 and evs[0].severity.value == "medium"
    assert evs[0].extra["D"] > evs[0].extra["d90"]
    a = w.model()["assign"][f"{S}|{new}"]
    assert a["role"] is None and m_class.class_key(w.store, S, new) is None
    assert not events(w.store, "new_entity_matched", new)
