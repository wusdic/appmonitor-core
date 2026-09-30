"""lib/plouvain: Louvain, similarity merge, stable ids, names, prefix covers (P11, §6.15)."""
import numpy as np

from app.engines.behavior.lib import plouvain as LV
from app.engines.behavior.lib import pminhash as MH


def _cliques(sizes, w_in=1.0, w_out=0.05):
    edges, off, lab = {}, 0, []
    for c, n in enumerate(sizes):
        for i in range(n):
            lab.append(c)
            for j in range(i + 1, n):
                edges[(off + i, off + j)] = w_in
        off += n
    # a weak chain between consecutive cliques
    off = 0
    for n in sizes[:-1]:
        edges[(off, off + n)] = w_out
        off += n
    return sum(sizes), edges, np.asarray(lab)


def test_louvain_recovers_planted_cliques():
    n, edges, truth = _cliques([5, 6, 7, 8])
    levels = LV.louvain(n, edges)
    lab = levels[-1]
    for c in range(4):
        assert len(set(lab[truth == c].tolist())) == 1
    assert len(set(lab.tolist())) == 4
    assert LV.modularity(lab, edges) > 0.6


def test_louvain_deterministic_and_isolated_nodes():
    n, edges, _ = _cliques([4, 4])
    a = LV.louvain(n + 3, edges)[-1]
    b = LV.louvain(n + 3, dict(edges))[-1]
    assert np.array_equal(a, b)
    assert len({int(a[n]), int(a[n + 1]), int(a[n + 2])}) == 3     # isolated -> singletons
    assert LV.louvain(0, {})[0].size == 0


def test_merge_similar_rejoins_a_split_population_only():
    rnd = MH.Randoms()
    # population P: 40 members with the same behaviour and noise; Q: 10 members elsewhere
    rng = np.random.default_rng(1)
    rows = []
    for i in range(40):
        rows.append(MH.icws([MH.h64(k) for k in ("a", "b", "c")],
                            np.asarray([3.0, 2.0, 1.0]) * rng.uniform(0.9, 1.1, 3), rnd))
    for i in range(10):
        rows.append(MH.icws([MH.h64(k) for k in ("a", "x", "y")],
                            np.asarray([1.0, 3.0, 2.0]) * rng.uniform(0.9, 1.1, 3), rnd))
    M = np.vstack(rows)
    lab = np.asarray([0] * 20 + [1] * 20 + [2] * 10)               # P cut in two by modularity
    adj = [(0, 1), (1, 2), (0, 2)]
    out, intra = LV.merge_similar(lab, M, 0.9, adj)
    assert out[0] == out[39]                                       # P re-merged
    assert out[40] != out[0]                                       # Q stays apart
    assert set(intra) == {int(out[0]), int(out[40])}


def test_match_ids_hungarian_and_threshold():
    prev = {"G1": {"a", "b", "c"}, "G2": {"x", "y"}}
    new = [{"x", "y", "z"}, {"a", "b", "c", "d"}, {"q", "r"}]
    m = LV.match_ids(prev, new, 0.3)
    assert m == {0: "G2", 1: "G1", 2: None}
    assert LV.match_ids({}, new) == {0: None, 1: None, 2: None}


def test_match_names():
    groups = {"G1": {"1.1.1.1", "1.1.1.2", "1.1.1.3"}, "G2": {"2.2.2.1", "2.2.2.2"}}
    named = [("综合部", {"1.1.1.1", "1.1.1.2", "1.1.1.3", "1.1.1.9"}), ("财务部", {"3.3.3.3"})]
    assert LV.match_names(groups, named) == {"G1": ("综合部", 0.75)}


def test_prefix_covers_department_and_pool():
    dept = [f"192.168.3.{i}" for i in range(20, 40)]
    others = [f"192.168.1.{i}" for i in (21, 23)] + ["10.168.7.121"]
    act = LV.ActiveIndex(dept + others)
    cv = LV.prefix_covers(dept, act)
    assert len(cv) == 1
    import ipaddress
    net = ipaddress.ip_network(cv[0])
    assert all(ipaddress.ip_address(ip) in net for ip in dept)
    assert net.prefixlen >= 24                                     # never spills into 192.168.1.0/24
    # a DHCP pool spread over a /22: one prefix, at most the /22
    rng = np.random.default_rng(0)
    pool = sorted({f"10.50.{int(rng.integers(0, 4))}.{int(rng.integers(1, 255))}" for _ in range(300)})
    cvp = LV.prefix_covers(pool, LV.ActiveIndex(pool + others))
    assert len(cvp) == 1 and ipaddress.ip_network(cvp[0]).prefixlen >= 22
    # mixed with outsiders so that no pure prefix covers 90 %: no cover
    mixed = [f"10.9.{i}.1" for i in range(40)]
    noise = [f"10.9.{i}.2" for i in range(40)] + [f"10.9.{i}.3" for i in range(40)]
    assert LV.prefix_covers(mixed, LV.ActiveIndex(mixed + noise), max_prefixes=4) == []


def test_prefix_covers_ipv6():
    ips = [f"2001:db8:0:1::{i:x}" for i in range(1, 20)]
    cv = LV.prefix_covers(ips, LV.ActiveIndex(ips))
    assert cv and cv[0].startswith("2001:db8:0:1::")


def test_merge_prefix_needs_behaviour_and_address_plan():
    """Two communities of one subnet merge at the looser bar (the looser one's
    within-similarity); the same pair across two subnets, or a pair below the
    bar, does not."""
    import numpy as np
    from app.engines.behavior.lib import plouvain as LV
    rng = np.random.default_rng(3)
    base = rng.integers(0, 2 ** 62, size=64, dtype=np.uint64)

    def rows(n, flip, seed):
        r = np.random.default_rng(seed)
        out = np.tile(base, (n, 1))
        for i in range(n):
            ix = r.choice(64, size=flip, replace=False)
            out[i, ix] = r.integers(0, 2 ** 62, size=flip, dtype=np.uint64)
        return out
    A, B = rows(6, 8, 1), rows(4, 16, 2)                 # B looser, and a bit apart
    M = np.vstack([A, B])
    lab = np.asarray([0] * 6 + [1] * 4)
    same = [f"192.168.3.{20 + i}" for i in range(10)]
    split = [f"192.168.3.{20 + i}" for i in range(6)] + [f"192.168.7.{20 + i}" for i in range(4)]
    act_same = LV.ActiveIndex(same)
    out = LV.merge_prefix(lab, M, same, act_same, 0.9)
    assert len(set(out.tolist())) == 1
    # the across similarity misses the tighter community's bar (merge_similar keeps them apart)
    ja = LV._mean_j(M, list(range(6)))
    jab = LV._mean_j(M, list(range(6)), list(range(6, 10)))
    assert jab < 0.9 * ja
    # different /24s of one /16: 192.168.0.0/16 is not pure once other subnets are active
    act_split = LV.ActiveIndex(split + [f"192.168.{k}.{i}" for k in (1, 2, 5) for i in range(1, 30)])
    out2 = LV.merge_prefix(lab, M, split, act_split, 0.9)
    assert len(set(out2.tolist())) == 2
    # a pair below even the looser bar
    C = rows(4, 60, 4)
    out3 = LV.merge_prefix(lab, np.vstack([A, C]), same, act_same, 0.9)
    assert len(set(out3.tolist())) == 2
