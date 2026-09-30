"""lib/pminhash: signatures, weighted MinHash (ICWS), LSH, mutual kNN (P11, §6.15)."""
import numpy as np
import pytest

from app.engines.behavior.lib import pminhash as MH
from app.engines.behavior.lib import psketch as PS

T0 = 1_790_000_000.0


def _sig(rnd, d):
    keys = sorted(d)
    return MH.icws([MH.h64(k) for k in keys], np.asarray([d[k] for k in keys]), rnd)


def test_icws_estimates_weighted_jaccard():
    rng = np.random.default_rng(0)
    rnd = MH.Randoms(k=512)
    errs = []
    for _ in range(40):
        keys = [f"i{j}" for j in range(12)]
        a = {k: float(rng.uniform(0, 5)) for k in keys if rng.random() < 0.8}
        b = {k: (a.get(k, 0.0) * float(rng.uniform(0.5, 1.5)) if rng.random() < 0.7 else float(rng.uniform(0, 5)))
             for k in keys}
        b = {k: v for k, v in b.items() if v > 0}
        exact = MH.weighted_jaccard(a, b)
        est = MH.jaccard_est(_sig(rnd, a), _sig(rnd, b))
        errs.append(est - exact)
    errs = np.asarray(errs)
    # unbiased, with the binomial spread of 512 samples (sd <= 0.022)
    assert abs(errs.mean()) < 0.02
    assert np.abs(errs).max() < 0.1


def test_icws_consistent_and_deterministic():
    rnd = MH.Randoms()
    a = {"x": 1.0, "y": 2.0}
    s1 = _sig(rnd, a)
    s2 = _sig(MH.Randoms(), dict(a))
    assert np.array_equal(s1, s2)                       # same item hashes -> same samples
    assert MH.jaccard_est(s1, _sig(rnd, {"x": 1.0, "y": 2.0})) == 1.0
    assert MH.jaccard_est(s1, _sig(rnd, {"p": 1.0, "q": 2.0})) < 0.1
    assert np.all(MH.icws([], np.zeros(0), rnd) == 0)


def test_sigstore_space_saving_decay_and_evidence():
    S = MH.SigStore(s_max=10, k=4)
    counts = [30, 20, 5, 5, 5, 5]                        # 6 distinct items into k = 4 slots
    for j, c in enumerate(counts):
        for _ in range(c):
            S.add("10.0.0.1", f"sys|A{j}", T0, 1.0, 0.5)
    ids, w = S.weights("10.0.0.1", T0)
    names = {S.items.key_of(int(i)) for i in ids}
    assert len(ids) == 4
    assert {"sys|A0", "sys|A1"} <= names                 # every item above N / k is kept
    assert S.get("10.0.0.1").ev == pytest.approx(0.5 * sum(counts))
    # forward decay: one half-life later every weight halves
    _, w2 = S.weights("10.0.0.1", T0 + PS.H_M)
    assert np.allclose(w2, w / 2.0, rtol=1e-5)


def test_sigstore_lru_cap_and_prune():
    S = MH.SigStore(s_max=100, k=8)
    for i in range(1000):
        S.add(f"10.1.{i // 250}.{i % 250}", "sys|A", T0 + i, 1.0, 1.0)
    assert len(S) == 100                                 # LRU by last activity
    assert S.dropped == 900
    assert "10.1.3.249" in S and "10.1.0.0" not in S
    n = S.prune(T0 + 8 * 86400, min_ev=10.0, min_age_s=7 * 86400, idle_s=30 * 86400)
    assert n == 100 and len(S) == 0


def test_sigstore_memory_is_per_source_constant():
    S = MH.SigStore(s_max=5000, k=MH.K_SIG)
    for i in range(5000):
        for j in range(40):                              # more items than slots
            S.add(f"s{i}", f"k|{(i + j) % 300}", T0, 1.0, 0.1)
    per = S.nbytes() / len(S)
    assert per < 800                                     # §7.1: S_max x ~0.8 KB at most
    assert len(S.items) == 300


def test_lsh_pairs_find_near_duplicates_and_cap_buckets():
    rnd = MH.Randoms()
    base = {f"i{j}": 1.0 + j for j in range(8)}
    sigs = [_sig(rnd, dict(base, extra=0.1 * i)) for i in range(5)]           # near-duplicates
    sigs += [_sig(rnd, {f"z{i}{j}": 1.0 for j in range(8)}) for i in range(5)]  # unrelated
    M = np.vstack(sigs)
    pairs = MH.lsh_pairs(M)
    assert {(0, 1), (0, 4), (3, 4)} <= pairs
    assert not any(a >= 5 or b >= 5 for a, b in pairs if (a < 5) != (b < 5))
    # a huge bucket of identical signatures links each member to CAP_LINKS others only
    big = np.vstack([sigs[0]] * 500)
    bp = MH.lsh_pairs(big, cap=64, cap_links=8)
    assert len(bp) <= 500 * 8


def test_mutual_knn_graph():
    rnd = MH.Randoms()
    a = [_sig(rnd, {"x": 1.0, "y": 1.0, f"n{i}": 0.2}) for i in range(4)]
    b = [_sig(rnd, {"p": 1.0, "q": 1.0, f"m{i}": 0.2}) for i in range(4)]
    M = np.vstack(a + b)
    edges = MH.knn_graph(M, MH.lsh_pairs(M))
    for (i, j), w in edges.items():
        assert (i < 4) == (j < 4)                       # never across the two populations
        assert w >= MH.J_MIN
    assert len(edges) >= 6
