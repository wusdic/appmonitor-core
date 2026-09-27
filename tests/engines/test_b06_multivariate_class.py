"""B06 MultivariateEngine hierarchy and shared models: the young-entity class
shrink (model.density@(s, class:<rid>) fitted on members' committed rows),
the dependence groups model.groups@(s, __system__) consumed by B04, and the
m_density accessors other engines (B14, B29, B30) call."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

from helpers import DT, T0, make_store, run_engine

from app.engines.behavior import multivariate as MV
from app.engines.behavior.lib import emit, m_density
from app.engines.behavior.lib.features import FEATURE_DIM, FEATURE_INDEX
from app.engines.behavior.multivariate import DENSITY, GROUPS_MODEL, ZI, MultivariateEngine

S = "erp"


def put_classes(store, roles: Dict[str, List[str]]) -> None:
    assign = {f"{S}|{ip}": {"role": rid, "sub": None, "prob": 1.0, "static": [], "pool": None,
                            "super": "machine"} for rid, ips in roles.items() for ip in ips}
    store.put_model("__org__", "__org__", "model.class", {
        "assign": assign, "version": 1,
        "roles": {rid: {"name": rid, "members": [f"{S}|{ip}" for ip in ips]}
                  for rid, ips in roles.items()}})


def pair_rows(rng, n, p=4, rho=0.9):
    C = np.eye(p)
    C[0, 1] = C[1, 0] = rho
    X = np.full((n, FEATURE_DIM), np.nan)
    X[:, :p] = rng.standard_normal((n, p)) @ np.linalg.cholesky(C).T
    return X


def run_ticks(eng, store, data: Dict[str, np.ndarray], t0=T0, training=True):
    """data: {entity: rows}; all entities tick together (a row of NaN = absent)."""
    n = max(len(v) for v in data.values())
    for i in range(n):
        t = t0 + i * DT
        for e, X in data.items():
            if i < len(X) and np.isfinite(X[i]).any():
                store.add_vec(S, e, ZI, t, np.asarray(X[i], dtype=np.float32), window_s=int(DT))
                store.register_entity(S, e)
        run_engine(eng, store, t, training=training, dt=DT)
    return t0 + n * DT


# ------------------------------------------------------------------ class shrink
def test_young_entity_backs_off_to_class_model():
    rng = np.random.default_rng(1)
    store = make_store()
    eng = MultivariateEngine()
    put_classes(store, {"r1": ["a", "b", "c", "d"]})
    t = run_ticks(eng, store, {e: pair_rows(rng, 120) for e in "abc"})
    cm = m_density.get(store, S, "class:r1")
    assert cm is not None and cm["members"] == ["a", "b", "c", "d"]
    assert cm["n"] > 200 and cm["box_src"] == "crossfit"            # refits every 8 h
    assert store.profile(S, "class:r1").extra["mv_model"]["fitted"] is True
    # newcomer d: 10 rows, no own fit yet -> the class model alone
    t = run_ticks(eng, store, {"d": pair_rows(rng, 10)}, t0=t)
    m = m_density.get(store, S, "d")
    assert m is not None and m["class_key"] == "class:r1"
    assert m["class_w"] == 1.0 and m["n"] == 0.0 and m["n_pred"] == MV.CLASS_PRIOR_N
    assert m["box_src"] == "class"
    x = np.full(FEATURE_DIM, np.nan)
    x[:4] = (2.5, -2.5, 0.0, 0.0)                                  # the class's pair breaks
    s = m_density.score_model(m, x)
    assert s.p_spe < 1e-3 and s.q == 4 and np.isfinite(s.p_t2)
    # with its own rows the class weight decays as 30 / (n + 30)
    run_ticks(eng, store, {"d": pair_rows(rng, 60)}, t0=t)
    m = m_density.get(store, S, "d")
    assert m["n"] >= 32 and m["class_w"] == pytest.approx(30.0 / (m["n"] + 30.0))
    assert m["n_pred"] == pytest.approx(m["n"] + 30.0)


def test_small_class_is_not_a_backoff_tier():
    rng = np.random.default_rng(2)
    store = make_store()
    eng = MultivariateEngine()
    put_classes(store, {"r1": ["a", "d"]})                          # < 3 members (contract L)
    t = run_ticks(eng, store, {"a": pair_rows(rng, 80)})
    run_ticks(eng, store, {"d": pair_rows(rng, 10)}, t0=t)
    assert store.get_model(S, "class:r1", DENSITY) is None
    assert m_density.get(store, S, "d") is None                     # cold: unscored


# ------------------------------------------------------------------ groups
def test_dependence_groups_model_and_split():
    rng = np.random.default_rng(3)
    store = make_store()
    eng = MultivariateEngine()
    i_up, i_down, i_peers = (FEATURE_INDEX[n] for n in ("bytes_up", "bytes_down",
                                                          "distinct_peers"))
    i_flows, i_dns = FEATURE_INDEX["flows"], FEATURE_INDEX["dns_queries"]

    def rows(n):
        X = np.full((n, FEATURE_DIM), np.nan)
        f = rng.standard_normal(n)
        X[:, i_up] = f
        X[:, i_down] = f + 0.1 * rng.standard_normal(n)
        X[:, i_peers] = -f + 0.1 * rng.standard_normal(n)           # |rho| counts
        X[:, i_flows] = rng.standard_normal(n)
        X[:, i_dns] = rng.standard_normal(n)
        return X

    assert m_density.groups(store, S) == [[i] for i in range(FEATURE_DIM)]   # absent
    run_ticks(eng, store, {e: rows(50) for e in "abc"})
    gm = store.get_model(S, "__system__", GROUPS_MODEL)
    assert gm is not None and gm["n_rows"] >= MV.GROUP_MIN_ROWS
    gs = m_density.groups(store, S)
    assert sorted(i for g in gs for i in g) == list(range(FEATURE_DIM))      # a partition
    big = [g for g in gs if len(g) > 1]
    assert big == [sorted([i_up, i_down, i_peers])]
    assert [i_flows] in gs and [i_dns] in gs
    split = m_density.groups(store, S, split_by_feature_group=True)
    assert sorted([i_up, i_down]) in split and [i_peers] in split
    gof = m_density.group_of(store, S)
    assert gof[i_up] == gof[i_down] == gof[i_peers] != gof[i_flows]


def test_dependence_groups_pure_edges():
    rng = np.random.default_rng(4)
    X = np.full((200, 6), np.nan)
    f = rng.standard_normal(200)
    X[:, 0], X[:, 1] = f, f * 2 + 0.05 * rng.standard_normal(200)
    X[:, 2] = rng.standard_normal(200)
    X[:10, 3] = f[:10]                                              # too few values: alone
    X[:, 4] = 0.6 * f + 0.8 * rng.standard_normal(200)              # rho ~ 0.6: below the cut
    groups, rho = MV.dependence_groups(X)
    assert groups == [[0, 1], [2], [3], [4], [5]]
    assert rho[0, 1] > 0.99 and rho[0, 3] == 0.0


# ------------------------------------------------------------------ accessors
def test_accessors_sigma_loglik_descriptor():
    rng = np.random.default_rng(5)
    store = make_store()
    eng = MultivariateEngine()
    run_ticks(eng, store, {"a": pair_rows(rng, 150)})
    Sg = m_density.sigma(store, S, "a")
    assert Sg.shape == (FEATURE_DIM, FEATURE_DIM)
    # (robustcov's OAS + consistency rescale reads a rho = 0.9 pair ~0.8 at n = 146)
    assert Sg[0, 1] > 0.5 and np.allclose(Sg[10:, 10:], np.eye(FEATURE_DIM - 10))
    mu = m_density.mean(store, S, "a")
    assert mu.shape == (FEATURE_DIM,) and (mu[4:] == 0).all()
    x = np.full(FEATURE_DIM, np.nan)
    x[:2] = (0.5, 0.4)
    ll = m_density.loglik(store, S, "a", x)
    ref = -0.5 * (m_density.score(store, S, "a", x).t2
                  + np.linalg.slogdet(Sg[:2, :2])[1] + 2 * np.log(2 * np.pi))
    assert ll == pytest.approx(ref, rel=1e-9)
    top = m_density.contributions(store, S, "a", x, top=2)
    assert len(top) == 2 and {d["idx"] for d in top} == {0, 1}
    d = store.profile(S, "a").extra["mv_model"]
    assert d == m_density.describe(store, S, "a")
    assert d["fitted"] and d["p"] == 4 and 1 <= d["k"] <= 4 and d["spe_box"]["src"] in (
        "oos", "crossfit")
    assert "bytes_up" in d["pcs"][0]["features"]
    # B14's reader finds the full covariance through the accessor
    assert m_density.sigma(store, S, "nobody") is None
    assert m_density.score(store, S, "nobody", x) is None
    assert emit.read_row(store, S, "a", emit.PM, T0) == {}          # cold at the first tick
