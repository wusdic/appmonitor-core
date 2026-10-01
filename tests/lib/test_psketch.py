"""lib/psketch.py (progressive core §6.1): error bounds, decay, memory bounds,
merge and persistence of every bounded sketch."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.engines.behavior.lib import psketch as PS
from app.engines.behavior.lib.sketch import HyperLogLog

DAY = PS.DAY


def _zipf_stream(n, n_items, a=1.2, seed=0):
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, n_items + 1) ** a
    p /= p.sum()
    return rng.choice(n_items, size=n, p=p)


# ------------------------------------------------------------ Space-Saving
def test_ss_exact_below_capacity():
    ss = PS.DecayedSpaceSaving(8)
    for i, k in enumerate("aabbbc"):
        ss.add(k, 0.0, 1.0, 1.0)
    assert ss.count("b", 0.0) == pytest.approx(3.0)
    assert ss.guaranteed("b", 0.0) == pytest.approx(3.0)
    assert ss.total(0.0) == pytest.approx(6.0)
    assert ss.untracked(0.0) == pytest.approx(0.0)
    assert [k for k, *_ in ss.items(0.0)] == ["b", "a", "c"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ss_guarantees_under_zipf(seed):
    k = 32
    ss = PS.DecayedSpaceSaving(k, mass_hl=(PS.H_M,), ev_hl=(PS.H_L,), primary=0)
    xs = _zipf_stream(20000, 5000, seed=seed)
    true = np.bincount(xs, minlength=5000).astype(float)
    for x in xs:
        ss.add(int(x), 0.0, 1.0, 1.0)
    N = ss.total(0.0)
    assert N == pytest.approx(20000)
    for key, cnt, guar, _ in ss.items(0.0):
        assert cnt >= true[key] - 1e-9            # overestimate
        assert guar <= true[key] + 1e-9           # guaranteed lower bound
        assert cnt - guar <= N / k + 1e-9         # error <= N / k
    heavy = np.nonzero(true > N / k)[0]
    for h in heavy:
        assert int(h) in ss                       # every item with share > 1/k tracked


def test_ss_decay_halves_after_half_life_and_rescale_is_exact():
    ss = PS.DecayedSpaceSaving(4)
    ss.add("x", 0.0, 8.0, 1.0)
    assert ss.count("x", PS.H_M, PS.CH_M) == pytest.approx(4.0)
    assert ss.count("x", PS.H_S, PS.CH_S) == pytest.approx(4.0)
    assert ss.count("x", PS.H_L, PS.CH_L) == pytest.approx(4.0)
    # daily adds for 200 days force several rescales (> 60 H_s); compare with
    # the closed form sum_j 2^(-(T - t_j)/H)
    ss2 = PS.DecayedSpaceSaving(4)
    T = 200 * DAY
    ts = np.arange(0, 200) * DAY
    for t in ts:
        ss2.add("y", float(t), 1.0, 1.0)
    for ch, h in enumerate(PS.HALF_LIVES):
        want = float(np.sum(2.0 ** (-(T - ts) / h)))
        assert ss2.count("y", T, ch) == pytest.approx(want, rel=1e-9)
    want_ev = float(np.sum(2.0 ** (-(T - ts) / PS.H_L)))
    assert ss2.total_evidence(T) == pytest.approx(want_ev, rel=1e-9)
    assert ss2.landmark > 0                       # it did rescale


def test_ss_unseen_mass_good_turing_and_churn():
    closed = PS.DecayedSpaceSaving(8)
    for d in range(60):
        for ip in ("a", "b", "c"):
            closed.add(ip, d * DAY, 1.0, 1.0)
    u_closed = closed.unseen(60 * DAY)
    assert u_closed < 0.01                        # three sources seen daily: closed
    churn = PS.DecayedSpaceSaving(8)
    for i in range(400):
        churn.add(f"ip{i}", i * 600.0, 1.0, 1.0)  # every source new
    assert churn.unseen(400 * 600.0) > 0.9        # churn never looks closed


def test_ss_confidence_reset_and_cap():
    ss = PS.DecayedSpaceSaving(4)
    for d in range(90):
        ss.add("a", d * DAY, 1.0, 1.0)
    T = 90 * DAY
    n_l, n_m = ss.total_evidence(T, PS.EV_L), ss.total_evidence(T, PS.EV_M)
    assert n_l > n_m * 3                          # confidence channel holds more
    ss.reset_confidence(T)
    assert ss.total_evidence(T, PS.EV_L) == pytest.approx(n_m)
    assert ss.evidence("a", T, PS.EV_L) == pytest.approx(ss.evidence("a", T, PS.EV_M))
    for d in range(90, 120):
        ss.add("a", d * DAY, 1.0, 1.0)
    assert ss.total_evidence(120 * DAY, PS.EV_L) > ss.total_evidence(120 * DAY, PS.EV_M)
    ss.cap_confidence(120 * DAY)
    assert ss.total_evidence(120 * DAY, PS.EV_L) == pytest.approx(
        ss.total_evidence(120 * DAY, PS.EV_M))


def test_ss_merge_bounds_and_roundtrip():
    k = 16
    xs = _zipf_stream(10000, 2000, seed=3)
    a = PS.DecayedSpaceSaving(k)
    b = PS.DecayedSpaceSaving(k)
    for i, x in enumerate(xs):
        (a if i % 2 else b).add(int(x), i * 10.0, 1.0, 1.0)
    true = np.zeros(2000)
    T = len(xs) * 10.0
    for i, x in enumerate(xs):
        true[x] += 2.0 ** (-(T - i * 10.0) / PS.H_M)
    a.merge(b)
    N = a.total(T)
    assert N == pytest.approx(true.sum(), rel=1e-9)
    for key, cnt, guar, _ in a.items(T):
        assert guar <= true[key] + 1e-9
        assert cnt >= true[key] - 1e-9
        assert cnt - guar <= N / k * 2 + 1e-9     # merged error <= (N1 + N2) / k
    c = PS.DecayedSpaceSaving.from_dict(a.to_dict())
    assert c.items(T) == a.items(T)
    assert c.unseen(T) == pytest.approx(a.unseen(T))


def test_ss_memory_is_bounded():
    ss = PS.DecayedSpaceSaving(64)
    for i in range(64):
        ss.add(f"k{i}", float(i), 1.0, 1.0)
    nb_full = ss.nbytes()
    for i in range(64, 50000):
        ss.add(f"k{i}", float(i), 1.0, 1.0)
    assert len(ss) == 64
    assert ss.nbytes() == nb_full and nb_full < 64 * 500


def test_ss_discard_and_distribution():
    ss = PS.DecayedSpaceSaving(4)
    for k, w in (("a", 5), ("b", 3), ("c", 2)):
        ss.add(k, 0.0, w, 1.0)
    keys, sh, other = ss.distribution(0.0)
    assert dict(zip(keys, sh))["a"] == pytest.approx(0.5)
    assert other == pytest.approx(0.0)
    assert ss.discard("b")
    assert "b" not in ss and len(ss) == 2
    keys, sh, other = ss.distribution(0.0)
    assert other == pytest.approx(0.3)


def test_hhh_select_nested_prefixes():
    lv = [PS.DecayedSpaceSaving(16) for _ in range(3)]
    parent = {0: lambda k: k.rsplit(".", 1)[0], 1: lambda k: k.rsplit(".", 1)[0]}

    def add(ip, w):
        p24 = ip.rsplit(".", 1)[0]
        p16 = p24.rsplit(".", 1)[0]
        for ss, key in zip(lv, (ip, p24, p16)):
            ss.add(key, 0.0, w, 1.0)
    add("10.0.1.5", 50)                           # one heavy IP
    for i in range(40):
        add(f"10.0.2.{i}", 1)                     # a /24 of light IPs
    add("10.0.3.1", 10)
    out = PS.hhh_select(lv, lambda l, k: parent[l](k), phi=0.3, t=0.0)
    got = {(l, k) for l, k, _ in out}
    assert (0, "10.0.1.5") in got
    assert (1, "10.0.2") in got
    assert (2, "10.0") not in got                 # its mass is all explained below


# ---------------------------------------------------------------- HLL
def test_hll_matches_lib_sketch_and_error():
    h = PS.HLL(10)
    ref = HyperLogLog()
    for i in range(5000):
        h.add(f"ip{i}")
        ref.add(f"ip{i}")
    assert np.array_equal(h.reg, ref.registers)
    assert abs(h.count() - 5000) / 5000 < 0.1
    h6 = PS.HLL(6)
    errs = []
    for s in range(20):
        hh = PS.HLL(6)
        for i in range(300):
            hh.add(f"{s}:{i}")
        errs.append(hh.count() / 300 - 1)
    assert abs(np.mean(errs)) < 0.1 and np.std(errs) < 0.25     # sigma ~ 1.04/8 = 0.13
    h6.merge(PS.HLL(6))
    assert PS.HLL.from_dict(h.to_dict()).count() == h.count()


def test_epoch_hll_rotation():
    e = PS.EpochHLL(p=10, epoch_s=7 * DAY)
    for i in range(100):
        e.add(f"a{i}", 0.0)
    for i in range(100):
        e.add(f"b{i}", 8 * DAY)                   # one rotation: a's kept as prev
    assert 170 < e.count() < 230
    e.add("c", 15 * DAY)                          # second rotation: a's gone
    assert 80 < e.count() < 120
    e.add("d", 60 * DAY)                          # long gap: both epochs reset
    assert e.count() < 3


# ---------------------------------------------------------------- t-digest
@pytest.mark.parametrize("seed", [0, 1])
def test_tdigest_rank_error_and_size(seed):
    rng = np.random.default_rng(seed)
    xs = rng.lognormal(7.0, 0.5, 20000)
    td = PS.TDigest(delta=50)
    if seed:
        for x in xs:
            td.add(float(x), 0.0)                 # scalar path
    else:
        td.add_many(xs, 0.0)
    srt = np.sort(xs)
    for q in (0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99):
        est = td.quantile(q)
        rank = np.searchsorted(srt, est) / len(xs)
        assert abs(rank - q) <= 0.005, (q, rank)            # measured <= 0.0021 (3 seeds)
        assert abs(td.cdf(np.quantile(xs, q)) - q) < 0.02
    assert td.n_centroids() <= 60
    assert td.quantile(0.0) == pytest.approx(xs.min())
    assert td.quantile(1.0) == pytest.approx(xs.max())


def test_tdigest_decay_follows_new_regime_and_merge():
    rng = np.random.default_rng(5)
    td = PS.TDigest(delta=50, hl=PS.H_M)
    td.add_many(rng.normal(0, 1, 2000), 0.0)
    td.add_many(rng.normal(10, 1, 2000), 10 * PS.H_M)
    assert td.quantile(0.5) > 9.0                 # old regime weighs 2^-10
    assert td.total(10 * PS.H_M) == pytest.approx(2000 * (1 + 2 ** -10), rel=1e-6)
    a, b = PS.TDigest(), PS.TDigest()
    xa, xb = rng.uniform(0, 1, 3000), rng.uniform(1, 2, 1000)
    a.add_many(xa, 0.0)
    b.add_many(xb, 0.0)
    a.merge(b)
    allx = np.concatenate([xa, xb])
    assert a.quantile(0.5) == pytest.approx(np.quantile(allx, 0.5), abs=0.03)
    c = PS.TDigest.from_dict(a.to_dict())
    assert c.quantile(0.9) == pytest.approx(a.quantile(0.9))


# --------------------------------------------------------------- Count-Min
def test_countmin_overestimate_bound():
    cm = PS.CountMin(w=256, d=4)
    xs = _zipf_stream(20000, 5000, seed=7)
    for x in xs:
        cm.add(int(x), 0.0)
    true = np.bincount(xs, minlength=5000)
    bound = cm.error_bound(0.0)
    over = 0
    for key in range(5000):
        est = cm.query(key, 0.0)
        assert est >= true[key] - 1e-9            # never underestimates
        over += est - true[key] > bound
    assert over / 5000 <= math.exp(-4) + 0.01     # P(err > e/w N) <= e^-d
    cm2 = PS.CountMin(w=256, d=4)
    cm2.add(3, 0.0, 5.0)
    cm.merge(cm2)
    assert cm.query(3, 0.0) >= true[3] + 5
    assert PS.CountMin.from_dict(cm.to_dict()).query(3, 0.0) == cm.query(3, 0.0)


def test_countmin_decay():
    cm = PS.CountMin(w=64, d=2, hl=DAY)
    cm.add("k", 0.0, 4.0)
    assert cm.query("k", 2 * DAY) == pytest.approx(1.0)


# ------------------------------------------------------------------- drift
def test_adwin_detects_increase_and_is_quiet_when_stationary():
    rng = np.random.default_rng(0)
    quiet = PS.ADWIN(delta=0.002)
    alarms = sum(quiet.add(x) != 0 for x in np.clip(rng.normal(3, 1, 5000), 0, 20))
    assert alarms <= 1
    a = PS.ADWIN(delta=0.002)
    for x in np.clip(rng.normal(3, 1, 2000), 0, 20):
        a.add(x)
    det = [a.add(x) for x in np.clip(rng.normal(6, 1, 400), 0, 20)]
    assert 1 in det and det.index(1) < 200
    assert a.mean > 5.0                           # old window dropped
    b = PS.ADWIN.from_dict(a.to_dict())
    assert b.width == a.width


def test_page_hinkley_two_sided():
    ph = PS.PageHinkley(lam=5.0, delta=0.5)
    assert all(ph.update(0.0) == 0 for _ in range(100))
    out = [ph.update(2.0, ref=0.0) for _ in range(10)]
    assert 1 in out
    ph.reset()
    out = [ph.update(-2.0, ref=0.0) for _ in range(10)]
    assert -1 in out


# --------------------------------------------------------------- reservoir
def test_weighted_reservoir_inclusion_and_decay():
    hits = np.zeros(4)
    for s in range(400):
        r = PS.WeightedReservoir(1, hl=None, seed=s)
        for i, w in enumerate((1.0, 1.0, 2.0, 4.0)):
            r.offer(i, w)
        hits[r.items()[0][0]] += 1
    p = hits / hits.sum()
    assert p[3] == pytest.approx(0.5, abs=0.08) and p[0] == pytest.approx(0.125, abs=0.06)
    # decayed keys: recent items dominate a sample of old and new equal weights
    r = PS.WeightedReservoir(50, hl=DAY, seed=1)
    for i in range(500):
        r.offer(("old", i), 1.0, 0.0)
    for i in range(500):
        r.offer(("new", i), 1.0, 10 * DAY)
    kinds = [it[0] for it, _, _ in r.items()]
    assert kinds.count("new") >= 48
    assert len(PS.WeightedReservoir.from_dict(r.to_dict())) == 50


# ------------------------------------------------------- LRU and evidence
def test_lru_and_burst_evidence():
    lru = PS.LRU(2)
    lru.put("a", 1)
    lru.put("b", 2)
    lru.get("a")
    assert lru.put("c", 3) == ("b", 2)
    ev = PS.BurstEvidence(cap=4, tau_burst=300)
    units = [ev.unit(("ip", "n"), 1000.0 + i) for i in range(10)]
    assert sum(units) == pytest.approx(sum(1 / (r + 1) for r in range(10)))
    assert all(u <= 1.0 for u in units)
    assert ev.unit(("ip", "n"), 5000.0) == pytest.approx(1.0)   # gap > tau restarts
    assert ev.unit(("ip2", "n"), 5000.0, factor=0.1) == pytest.approx(0.1)
    for i in range(10):
        ev.unit((f"x{i}", "n"), 6000.0)
    assert len(ev) == 4                                          # LRU cap


def test_decayed_vector_list_fast_path_matches_array_path():
    """The per-entry Python path for list weights (P04 NumSummary moments)
    gives the same decayed sums as the numpy path, across a landmark rescale."""
    import numpy as _np
    hl = [PS.H_S] * 3 + [PS.H_M] * 3 + [PS.H_L] * 3
    a, b = PS.DecayedVector(hl), PS.DecayedVector(hl)
    t0 = 1.7e9
    for i in range(400):
        t = t0 + i * 3600.0 * 6
        w = [1.0 + i % 3, 2.0 * i, 0.5] * 3
        a.add(t, w)
        b.add(t, _np.asarray(w))
    t_end = t0 + 400 * 3600.0 * 6
    assert _np.allclose(a.read(t_end), b.read(t_end), rtol=1e-12, atol=0.0)
