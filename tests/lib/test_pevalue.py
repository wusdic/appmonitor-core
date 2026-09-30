"""lib/pevalue.py (§6.5.3-§6.5.5, §6.7): anytime validity of the averaged
e-values under continuous monitoring, power under dependence, value grouping,
tie-breaking and the leave-x-out exception test. The simulations run the
validity check at a small threshold (tau = 2, bound 2^-2 per candidate) so the
bound is measurable in a unit-test budget; tau0 = 10 only scales it."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.engines.behavior.lib import pevalue as PE


def _leaf_pred(counts, kb):
    return (counts + 0.5) / (counts.sum(axis=1, keepdims=True) + 0.5 * kb)


def _run_null(n_leaves, n_events, tau, T=2, dup=1, kb=4, nv=6, seed=0, omega=None):
    """Fraction of leaves whose averaged e-value ever reaches 2^tau (checked
    every 32 events) when the targets are independent of the candidate."""
    rng = np.random.default_rng(seed)
    crossed = 0
    for _ in range(n_leaves):
        theta = rng.dirichlet(np.ones(kb), size=T)
        u = rng.random((n_events, T))
        base = (u[:, :, None] > np.cumsum(theta, axis=1)[None]).sum(axis=2)
        bins = np.tile(base, (1, dup))
        vals = rng.integers(0, nv, n_events)
        ws = np.ones(n_events) if omega is None else omega(rng, n_events)
        ss = PE.SplitStats(T * dup, k_b=kb, C=1)
        ss.set_candidate(0, ("rand", 0), card_hint=nv)
        leaf = np.zeros((T * dup, kb))
        for e in range(n_events):
            ss.update([int(vals[e])], bins[e], _leaf_pred(leaf, kb), float(ws[e]))
            leaf[np.arange(T * dup), bins[e]] += ws[e]
            if (e + 1) % 32 == 0 and ss.log2_e(0) >= tau:
                crossed += 1
                break
    return crossed / n_leaves


def test_null_false_split_rate_within_ville_bound():
    rate = _run_null(120, 320, tau=2.0, seed=1)
    assert rate <= 0.25                               # P(sup e >= 4) <= 1/4
    # fractional evidence units (omega in (0, 1]) keep the bound (Jensen)
    rate_w = _run_null(80, 320, tau=2.0, seed=2,
                       omega=lambda r, n: r.uniform(0.05, 1.0, n))
    assert rate_w <= 0.25


def test_duplicated_targets_do_not_inflate_the_averaged_evalue():
    """(b') three perfectly dependent copies of each target: the average of
    e-processes is still an e-process, so the bound holds unchanged."""
    rate = _run_null(80, 320, tau=2.0, dup=3, seed=3)
    assert rate <= 0.25


def test_evidence_unit_above_one_is_rejected():
    """(b'') a row's mass (HT weight 1000, aggregation 40) must never enter as
    evidence: omega > 1 is a contract violation."""
    ss = PE.SplitStats(1, k_b=2, C=1)
    ss.set_candidate(0, "a")
    with pytest.raises(ValueError):
        ss.update(["x"], [0], np.full((1, 2), 0.5), 40.0)
    ex = PE.ExcTracker(1, k_b=2)
    with pytest.raises(ValueError):
        ex.update("ip", [0], np.full((1, 2), 0.5), 1000.0)


def _power_run(s_mode, n_events=1200):
    rng = np.random.default_rng(0)
    kb = 9
    ss = PE.SplitStats(2, k_b=kb, C=2)
    ss.set_candidate(0, ("net.src", 0), card_hint=2 ** 32)
    ss.set_candidate(1, ("noise", 0), card_hint=16)
    leaf = np.zeros((2, kb))
    first_v = first = None
    for e in range(n_events):
        if rng.random() < 0.6:
            src = ["ip1", "ip2", "ip3"][e % 3]
            b = np.array([{"ip1": 0, "ip2": 1, "ip3": 2}[src], rng.integers(0, 3)])
        else:
            src = f"pop{rng.integers(0, 40)}"
            b = np.array([3 + rng.integers(0, 6), rng.integers(0, 3)])
        ss.update([src, int(rng.integers(0, 16))], b, _leaf_pred(leaf, kb), 1.0, day=e // 30)
        leaf[np.arange(2), b] += 1
        if (e + 1) % 32 == 0:
            dec = ss.check(s_mode=s_mode)
            if dec.pass_v and first_v is None:
                first_v = e
            if dec.accept and first is None:
                first = e
                assert dec.best == 0
                assert {"ip1", "ip2", "ip3"} <= set(v for g in dec.groups for v in g)
    return ss, first_v, first


def test_power_three_ips_with_distinct_usernames():
    """(a)-style power check: the target (username bin) is a function of the
    candidate value for three sources and random for a mixed population.
    Measured: (V) at 32 units, spec-(S) at 352 (range term), rival-valid (S)
    at 64."""
    ss, first_v, first = _power_run("spec")
    assert first_v is not None and first_v < 100
    assert first is not None and first < 500
    assert ss.log2_e(0) > PE.TAU0 + math.log2(ss.C_ever)
    assert ss.log2_e(1) < 2.0
    _, first_v2, first2 = _power_run("rival_valid", 400)
    assert first2 is not None and first2 < 100


def test_value_grouping_one_group_plus_other():
    """(d) three IPs sharing a target distribution unite; the rest is `other`."""
    kb = 4
    ss = PE.SplitStats(1, k_b=kb, k_v=8, C=1)
    ss.set_candidate(0, ("net.src", 0), card_hint=256)
    rng = np.random.default_rng(4)
    leaf = np.zeros((1, kb))
    for e in range(900):
        if e % 3:
            src, b = ["a", "b", "c"][e % 9 // 3], 0
        else:
            src, b = f"p{rng.integers(0, 60)}", int(rng.integers(1, kb))
        ss.update([src], [b], _leaf_pred(leaf, kb), 1.0)
        leaf[0, b] += 1
    groups, oidx, evs = ss.value_groups(0)
    named = [sorted(g) for i, g in enumerate(groups) if i != oidx and g]
    assert ["a", "b", "c"] in named
    assert oidx >= 0


def test_empirical_bernstein_tie_break_is_deterministic():
    """(c) two identical candidates: the deviation of their difference is 0,
    eps <= tau_tie, and the earlier-tracked candidate wins every time."""
    kb = 3
    for _ in range(2):
        ss = PE.SplitStats(1, k_b=kb, C=2)
        ss.set_candidate(0, "first")
        ss.set_candidate(1, "second")
        leaf = np.zeros((1, kb))
        for e in range(200):
            v = e % 2
            b = v
            ss.update([v, v], [b], _leaf_pred(leaf, kb), 1.0, day=e // 50)
            leaf[0, b] += 1
        dec = ss.check()
        assert dec.best == 0 and dec.pass_s and dec.eps <= PE.TAU_TIE


def test_restart_keeps_counting_candidates_and_roundtrip():
    ss = PE.SplitStats(2, C=3)
    for i in range(3):
        ss.set_candidate(i, ("a", i), 4)
    assert ss.C_ever == 3
    ss.update([1, 2, 3], [0, 1], np.full((2, PE.K_B), 1 / PE.K_B), 1.0)
    ss.restart()
    assert ss.C_ever == 6 and ss.n.sum() == 0
    ss.update([1, 2, 3], [0, -1], np.full((2, PE.K_B), 1 / PE.K_B), 0.5)
    ss2 = PE.SplitStats.from_dict(ss.to_dict())
    assert ss2.log2_e(0) == ss.log2_e(0)
    assert ss2.C_ever == 6
    big = PE.SplitStats(10)                          # defaults, T = m_t + 2 = 10
    assert big.nbytes() < 56_000                     # measured 50.8 KB (float64; spec est. 21 KB f32)


def test_exception_for_distinct_sizes_not_for_same_distribution():
    """(g) at a confirmed node: an IP whose size bins differ becomes an
    exception; IPs drawn from the node's own distribution never do."""
    kb = 6

    def run(with_odd, seed):
        rng = np.random.default_rng(seed)
        ex = PE.ExcTracker(1, k_b=kb, k_x=8)
        node = np.zeros((1, kb))
        for ip in ("odd", "n1", "n2"):
            ex.track(ip)
        for e in range(1500):
            ip = ["odd" if with_odd else "n0", "n1", "n2", "pop"][e % 4]
            b = (5 if rng.random() < 0.9 else 4) if ip == "odd" else int(rng.integers(0, 4))
            ex.update(ip, [b], _leaf_pred(node, kb), 1.0, day=e // 200)
            node[0, b] += 1
        return ex
    ex = run(True, 7)
    assert ex.decide("odd", n_conf=50)
    assert ex.per_target_saving("odd")[0] > 2.0
    assert not ex.decide("odd", n_conf=5)             # too little confidence evidence
    for seed in range(5):
        ex = run(False, seed)
        assert not ex.decide("n1", n_conf=50) and not ex.decide("n2", n_conf=50)
