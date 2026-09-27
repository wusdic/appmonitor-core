"""Tests for engines/behavior/lib/ppm.py: PPM-C with exclusion, lazy decay,
the entity -> class -> system backoff chain with Good-Turing novel mass,
entropy-rate bookkeeping, merge and serialisation.

Probabilities are checked three ways: against hand-computed PPM-C values on
tiny models, by normalisation (a single tier sums to exactly 1 over the
vocabulary for any history; a chain sums to <= 1), and by the B10 behaviour
targets from docs/lib3/engines.md (abnormal session excess >= 4 bits, a
normal held-out session < 1 bit, unseen source tokens finite and > 0).
Everything is seeded and small, so the file runs in well under a second.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from app.engines.behavior.lib import ppm as P  # noqa: E402

T0 = 1_700_000_000.0
DAY = 86400.0
HL = P.PPM_HALF_LIFE_S
V = 40                      # "real" vocabulary size (template size) for the chain tests


# ------------------------------------------------------------------ helpers
def _shop_session(rng: random.Random):
    """login -> dashboard -> orders -> view{n} x 1..3 (the B10 unit-test grammar)."""
    s = ["login", "dashboard", "orders"]
    s += [f"view{rng.randrange(5)}" for _ in range(rng.randint(1, 3))]
    return s


def _report_session(rng: random.Random):
    """A second, distinct grammar sharing the login token."""
    s = ["login", "reports"]
    s += [f"report{rng.randrange(4)}" for _ in range(rng.randint(1, 3))]
    return s + ["export", "logout"]


def _train(gen, n=300, seed=0, t0=T0, dt=3600.0, w=1.0, vocab_size=V, model=None,
           backoff=()):
    """Prequential fit: score each session before learning it and fold the
    surprisal into the entropy-rate stats (what B10 does at commit)."""
    rng = random.Random(seed)
    m = model if model is not None else P.PPMModel()
    for i in range(n):
        s = gen(rng)
        P.record_surprisal(m, P.loglik(m, s, backoff=backoff, vocab_size=vocab_size), w)
        P.update(m, s, w=w, ts=t0 + i * dt)
    return m


def _prob(model, sym, history=(), backoff=(), vocab_size=V):
    return 2.0 ** -P.loglik(model, [sym], backoff=backoff, vocab_size=vocab_size,
                            history=history)[0]


def _snapshot(m):
    return json.dumps(m.to_dict(), sort_keys=True, default=str)


# ------------------------------------------------------------------ B10 targets
def test_b10_abnormal_session_excess_and_normal_held_out():
    m = _train(_shop_session, n=300)
    mu, sigma = P.entropy_rate(m)
    assert math.isfinite(mu) and math.isfinite(sigma) and sigma > 0

    ab = P.loglik(m, ["login", "export", "login", "export"], vocab_size=V)
    assert np.all(np.isfinite(ab))
    assert ab.mean() - mu >= 4.0

    rng = random.Random(123)                      # held-out normal sessions
    excess = [P.loglik(m, _shop_session(rng), vocab_size=V).mean() - mu for _ in range(200)]
    assert max(excess) < 1.0
    assert abs(float(np.mean(excess))) < 0.5      # entropy rate tracks the grammar


def test_b10_abnormal_excess_with_backoff_chain():
    """Same targets with the full entity -> class -> system chain, where the
    class knows 'export' (it is not novel to the organisation, only to the
    entity): the entity-level excess must still be large."""
    cls = _train(lambda r: _shop_session(r) if r.random() < 0.5 else _report_session(r),
                 n=400, seed=5)
    sysm = P.PPMModel(order=0)
    rng = random.Random(9)
    for i in range(400):
        P.update(sysm, _report_session(rng) + _shop_session(rng), ts=T0 + i * 3600)
    chain = [cls, sysm]
    m = _train(_shop_session, n=300, backoff=chain)
    mu, _ = P.entropy_rate(m)
    ab = P.loglik(m, ["login", "export", "login", "export"], backoff=chain, vocab_size=V)
    assert ab.mean() - mu >= 4.0
    rng = random.Random(77)
    excess = [P.loglik(m, _shop_session(rng), backoff=chain, vocab_size=V).mean() - mu
              for _ in range(100)]
    assert max(excess) < 1.0


def test_high_entropy_random_walk_excess_is_centred():
    """B10 (b) precondition: for a high-entropy entity the entropy rate tracks
    its own surprisal, so normal 8-token windows have excess ~0 and a thin
    upper tail (the conformal layer then sets the 1 % threshold)."""
    rng = random.Random(0)

    def walk(n):
        x, out = rng.randrange(30), []
        for _ in range(n):
            x = (x + rng.choice([-2, -1, 1, 2, 5])) % 30
            out.append(f"s{x}")
        return out

    m = P.PPMModel()
    for i in range(800):
        s = walk(12)
        P.record_surprisal(m, P.loglik(m, s, vocab_size=V))
        P.update(m, s, ts=T0 + i * 600)
    mu, sigma = P.entropy_rate(m)
    assert 2.0 < mu < 4.0 and sigma > 0.5
    ex = np.array([P.loglik(m, walk(8), vocab_size=V).mean() - mu for _ in range(400)])
    assert abs(ex.mean()) < 0.3
    assert np.quantile(ex, 0.99) < 1.5


def test_unseen_source_token_finite_positive():
    m = _train(_shop_session, n=100)
    sysm = P.PPMModel(order=0)
    P.update(sysm, ["login", "dashboard", "orders", "reports"])
    for chain in ([], [sysm], [P.PPMModel(), sysm]):
        for vs in (0, 5, V, 10_000):
            s = P.loglik(m, ["never-seen-src"], backoff=chain, vocab_size=vs)
            assert s.shape == (1,) and np.isfinite(s[0]) and s[0] > 0.0
    # larger real vocabulary -> each unseen state is rarer
    s_small = P.loglik(m, ["zz"], vocab_size=20)[0]
    s_big = P.loglik(m, ["zz"], vocab_size=2000)[0]
    assert s_big > s_small
    # a known token is far cheaper than an unseen one
    assert P.loglik(m, ["login"], vocab_size=V)[0] < s_small


# ------------------------------------------------------------------ exact PPM-C
def test_hand_computed_single_tier_with_exclusion():
    m = P.PPMModel(order=1)
    P.update(m, ["a", "b", "a", "c"])
    # order0 {a:2,b:1,c:1} N=4, singletons b,c -> N1=2, g=0.5 (inside [0.1, 0.9])
    # order1 (a,)->{b:1,c:1}, (b,)->{a:1}
    assert P.good_turing_unseen(m) == pytest.approx(0.5)
    vs = 10
    # no history: straight to Good-Turing: (1-g) * 2/4
    assert _prob(m, "a", vocab_size=vs) == pytest.approx(0.25)
    # (a,) has b: 1/(n+q) = 1/(2+2)
    assert _prob(m, "b", ["a"], vocab_size=vs) == pytest.approx(0.25)
    # (b,)->{a}: escape q/(n+q)=1/2, exclude a; GT: (1-g) c/(N - c_a) = .5*1/2
    assert _prob(m, "c", ["b"], vocab_size=vs) == pytest.approx(0.5 * 0.25)
    # (a,)->{b,c}: escape 2/4, exclude b,c; GT: .5 * 2/(4-2)
    assert _prob(m, "a", ["a"], vocab_size=vs) == pytest.approx(0.5 * 0.5)
    # unseen after (b,): escape 1/2 then g / (vocab - V_seen) = .5/7
    assert _prob(m, "d", ["b"], vocab_size=vs) == pytest.approx(0.5 * 0.5 / 7)
    # surprisal is -log2 P
    s = P.loglik(m, ["a", "b"], vocab_size=vs)
    np.testing.assert_allclose(s, [2.0, 2.0])


def test_hand_computed_order_ppmc_escape_at_higher_order():
    m = P.PPMModel(order=2)
    P.update(m, ["x", "y", "z"])
    P.update(m, ["x", "y", "w"])
    # (x,y)->{z:1,w:1}: P(z|x,y) = 1/(2+2)
    assert _prob(m, "z", ["x", "y"], vocab_size=8) == pytest.approx(0.25)
    # y after (x,): (x,)->{y:2}: 2/(2+1)
    assert _prob(m, "y", ["x"], vocab_size=8) == pytest.approx(2.0 / 3.0)
    # x after (x,y): escape 2/4 from (x,y) excluding {z,w}; (y,)->{z,w}: all
    # excluded -> escape w.p. 1; order 0: N=6, z,w,x,y  obs: x2 y2 z1 w1 ->
    # N1=2, g=1/3; exclusion removes z,w: (1-g) * 2 / (6-2)
    g = 2.0 / 6.0
    assert _prob(m, "x", ["x", "y"], vocab_size=8) == pytest.approx(0.5 * (1 - g) * 2 / 4)


def test_hand_computed_two_tier_chain():
    ent = P.PPMModel(order=1)
    P.update(ent, ["a", "b"])
    sysm = P.PPMModel(order=0)
    P.update(sysm, ["a", "a", "b", "c"])       # N=4, N1=2 (b, c) -> g=.5
    vs = 10
    # entity: (a,)->{b:1}: 1/(1+1)
    assert _prob(ent, "b", ["a"], [sysm], vs) == pytest.approx(0.5)
    # entity: (b,) absent; order0 {a,b} escape 2/(2+2); system (no cross-tier
    # exclusion): (1-g) c_c / N = .5 * 1/4
    assert _prob(ent, "c", ["b"], [sysm], vs) == pytest.approx(0.5 * 0.125)
    # never seen anywhere: .5 * g / (vocab - 3)
    assert _prob(ent, "d", ["b"], [sysm], vs) == pytest.approx(0.5 * 0.5 / 7)


def test_single_tier_is_exactly_normalised():
    rng = random.Random(3)
    m = P.PPMModel(order=3)
    for i in range(60):
        n = rng.randint(2, 9)
        P.update(m, [f"t{rng.randrange(12)}" for _ in range(n)], w=rng.uniform(0.2, 3.0),
                 ts=T0 + i * 7 * 3600)
    seen = set(P.vocab(m))
    universe = sorted(seen) + [f"new{i}" for i in range(8)]
    vs = len(universe)
    histories = [(), ("t1",), ("t1", "t2"), ("t3", "t4", "t5"), ("new0", "t1"), ("new1",)]
    for h in histories:
        tot = sum(_prob(m, v, h, vocab_size=vs) for v in universe)
        assert tot == pytest.approx(1.0, abs=1e-9), h


def test_chain_is_sub_normalised():
    rng = random.Random(4)
    ent, cls, sysm = P.PPMModel(), P.PPMModel(), P.PPMModel(order=0)
    for i in range(40):
        P.update(ent, [f"t{rng.randrange(6)}" for _ in range(6)], ts=T0 + i)
        P.update(cls, [f"t{rng.randrange(15)}" for _ in range(6)], ts=T0 + i)
        P.update(sysm, [f"t{rng.randrange(25)}" for _ in range(6)], ts=T0 + i)
    universe = [f"t{i}" for i in range(25)] + [f"u{i}" for i in range(5)]
    for h in [(), ("t1",), ("t2", "t3", "t4"), ("t20",)]:
        tot = sum(_prob(ent, v, h, [cls, sysm], len(universe)) for v in universe)
        assert 0.5 < tot <= 1.0 + 1e-12


# ------------------------------------------------------------------ backoff
def test_backoff_chain_lowers_surprisal_of_class_known_tokens():
    ent = _train(_shop_session, n=200)
    cls = _train(lambda r: _shop_session(r) if r.random() < 0.5 else _report_session(r),
                 n=300, seed=1)
    sysm = P.PPMModel(order=0)
    rng = random.Random(2)
    for _ in range(300):
        P.update(sysm, _shop_session(rng) + _report_session(rng) + ["admin"])
    h = ["login"]
    s_alone = P.loglik(ent, ["reports"], vocab_size=V, history=h)[0]
    s_sys = P.loglik(ent, ["reports"], [sysm], vocab_size=V, history=h)[0]
    s_cls = P.loglik(ent, ["reports"], [cls, sysm], vocab_size=V, history=h)[0]
    assert s_cls < s_sys < s_alone
    # the class also knows what follows 'reports' (its own order-1 context)
    s2_cls = P.loglik(ent, ["report1"], [cls, sysm], vocab_size=V, history=["reports"])[0]
    s2_sys = P.loglik(ent, ["report1"], [sysm], vocab_size=V, history=["reports"])[0]
    assert s2_cls < s2_sys
    # system-only token: finite, and cheaper with the system tier present
    s_admin = P.loglik(ent, ["admin"], [cls, sysm], vocab_size=V)[0]
    s_admin_no = P.loglik(ent, ["admin"], [cls], vocab_size=V)[0]
    assert np.isfinite(s_admin) and s_admin < s_admin_no
    # tokens the entity knows are scored by the entity alone
    s_own = P.loglik(ent, ["dashboard"], vocab_size=V, history=h)[0]
    assert P.loglik(ent, ["dashboard"], [cls, sysm], vocab_size=V, history=h)[0] == s_own


def test_good_turing_novel_mass_and_floor():
    m = P.PPMModel(order=0)
    assert P.good_turing_unseen(m) == 1.0                      # empty = immature
    P.update(m, ["a", "b", "c"])
    assert P.good_turing_unseen(m) == pytest.approx(1.0)
    m = P.PPMModel(order=0)
    P.update(m, ["a", "a", "b"])
    assert P.good_turing_unseen(m) == pytest.approx(1.0 / 3.0)
    # mature: no singletons -> the floor 0.5 / (N + 1) is used by loglik
    m = P.PPMModel(order=0)
    P.update(m, ["a", "b"] * 50)
    assert P.good_turing_unseen(m) == 0.0
    N = 100.0
    g = 0.5 / (N + 1)
    assert _prob(m, "zz", vocab_size=12) == pytest.approx(g / 10)
    assert _prob(m, "a", vocab_size=12) == pytest.approx((1 - g) * 0.5)
    # everything seen once: g is capped below 1 so seen tokens keep mass
    m = P.PPMModel(order=0)
    P.update(m, ["only"])
    p = _prob(m, "only", vocab_size=5)
    assert 0.0 < p < 1.0 and np.isfinite(-math.log2(p))
    # the singleton ratio is invariant to a uniform weight (stream_frac, trust)
    a, b = P.PPMModel(order=0), P.PPMModel(order=0)
    toks = ["a", "a", "b", "c", "c", "c", "d"]
    P.update(a, toks, w=1.0)
    P.update(b, toks, w=4.0)
    assert P.good_turing_unseen(a) == pytest.approx(P.good_turing_unseen(b))


def test_empty_chain_and_none_tiers():
    s = P.loglik(P.PPMModel(), ["a", "b"], vocab_size=16)
    np.testing.assert_allclose(s, [4.0, 4.0])
    m = P.PPMModel()
    P.update(m, ["a", "b"])
    base = P.loglik(m, ["a", "c"], vocab_size=16)
    np.testing.assert_array_equal(P.loglik(m, ["a", "c"], [None, P.PPMModel()], vocab_size=16),
                                  base)
    assert P.loglik(m, [], vocab_size=16).shape == (0,)
    assert P.loglik(m, [], vocab_size=16).dtype == np.float64
    # an empty last tier: the last non-empty tier carries the Good-Turing step
    np.testing.assert_array_equal(P.loglik(m, ["a", "c"], [P.PPMModel(order=0)], vocab_size=16),
                                  base)


# ------------------------------------------------------------------ update
def test_update_all_contexts_and_history():
    m = P.PPMModel(order=3)
    P.update(m, ["a", "b", "c", "d"])
    expect = {(), ("a",), ("b",), ("c",), ("a", "b"), ("b", "c"), ("a", "b", "c")}
    assert set(m.counts) == expect
    assert m.counts[("a", "b", "c")] == {"d": 1.0}
    assert m.counts[()] == {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
    assert m.n_tokens == pytest.approx(4.0)
    # a window that continues a session == learning the whole session at once
    toks = ["a", "b", "c", "d", "b", "c", "e"]
    m1, m2 = P.PPMModel(), P.PPMModel()
    P.update(m1, toks, ts=T0)
    P.update(m2, toks[:3], ts=T0)
    P.update(m2, toks[3:], ts=T0, history=["x"] + toks[:3])   # only the last 3 matter
    assert _snapshot(m1) == _snapshot(m2)
    # loglik with history == the tail of the joint loglik
    np.testing.assert_allclose(P.loglik(m1, toks[3:], history=toks[:3], vocab_size=V),
                               P.loglik(m1, toks, vocab_size=V)[3:], rtol=0, atol=1e-12)


def test_weights_scale_counts_and_invalid_weights_are_noops():
    toks = ["login", "orders", "view1", "orders"]
    a, b = P.PPMModel(), P.PPMModel()
    P.update(a, toks, w=1.0, ts=T0)
    P.update(b, toks, w=4.0, ts=T0)                # stream_frac = 0.25 -> x4
    va, vb = P.vocab(a), P.vocab(b)
    assert va.keys() == vb.keys()
    for k in va:
        assert vb[k] == pytest.approx(4.0 * va[k])
    assert b.n_tokens == pytest.approx(4.0 * a.n_tokens)
    snap = _snapshot(a)
    for w in (0.0, -1.0, float("nan"), float("inf")):
        assert P.update(a, toks, w=w, ts=T0 + DAY) is a
        assert _snapshot(a) == snap
    P.update(a, [], ts=T0 + DAY)
    assert _snapshot(a) == snap
    # an effective weight below the 1e-9 "seen" threshold creates no entries
    P.update(a, ["ghost"], w=1e-12, ts=T0)
    P.update(a, ["ghost"], w=1.0, ts=T0 - 40 * HL)
    assert "ghost" not in P.vocab(a) and _snapshot(a) == snap


def test_order_zero_model_and_integer_tokens():
    m = P.PPMModel(order=0)
    P.update(m, [3, 1, 3, 3, 2], history=[9, 9])
    assert set(m.counts) == {()}
    assert P.vocab(m) == {3: 3.0, 1: 1.0, 2: 1.0}
    s = P.loglik(m, [3, 7], vocab_size=10, history=[1])
    assert np.all(np.isfinite(s)) and s[1] > s[0] > 0


def test_hand_built_counts_are_indexed_lazily():
    m = P.PPMModel(order=1, counts={(): {"a": 3.0, "b": 1.0}, ("a",): {"b": 1.0, "a": 2.0}})
    s = P.loglik(m, ["b"], history=["a"], vocab_size=4)
    assert s[0] == pytest.approx(-math.log2(1.0 / 5.0))
    # b inferred as a singleton (count 1) -> g = 1/4
    assert P.good_turing_unseen(m) == pytest.approx(0.25)
    P.update(m, ["a", "b"])
    assert P.vocab(m) == {"a": 4.0, "b": 2.0}
    assert P.good_turing_unseen(m) == pytest.approx(0.0)


# ------------------------------------------------------------------ decay
def test_decay_half_life():
    m = P.PPMModel(order=1)
    P.update(m, ["a"], ts=T0)
    P.update(m, ["b"], ts=T0 + HL)
    v = P.vocab(m)
    assert v["a"] == pytest.approx(0.5) and v["b"] == pytest.approx(1.0)
    assert m.n_tokens == pytest.approx(1.5)
    P.update(m, ["c"], ts=T0 + 2 * HL)
    v = P.vocab(m)
    assert v["a"] == pytest.approx(0.25) and v["b"] == pytest.approx(0.5)
    # the stored scale is lazy: g tracks the clock in half-lives
    assert m.g == pytest.approx(2.0)


def test_out_of_order_rows_equal_in_order_fit():
    rng = random.Random(11)
    rows = [(T0 + rng.uniform(0, 90 * DAY), _shop_session(rng), rng.uniform(0.2, 1.0))
            for _ in range(80)]
    a, b = P.PPMModel(), P.PPMModel()
    for ts, toks, w in sorted(rows, key=lambda r: r[0]):
        P.update(a, toks, w=w, ts=ts)
    for ts, toks, w in rows:                        # arrival order (release after hold)
        P.update(b, toks, w=w, ts=ts)
    va, vb = P.vocab(a), P.vocab(b)
    assert va.keys() == vb.keys()
    for k in va:
        assert vb[k] == pytest.approx(va[k], rel=1e-9)
    probe = ["login", "dashboard", "orders", "view3", "export"]
    np.testing.assert_allclose(P.loglik(a, probe, vocab_size=V), P.loglik(b, probe, vocab_size=V),
                               rtol=1e-9)
    # an old row is decayed, never the state backwards
    c = P.PPMModel(order=0)
    P.update(c, ["x"], ts=T0 + HL)
    P.update(c, ["y"], ts=T0)
    assert P.vocab(c) == pytest.approx({"x": 1.0, "y": 0.5})


def test_renormalisation_prunes_dead_entries():
    m = P.PPMModel(order=2)
    P.update(m, ["old1", "old2", "old3"], ts=T0)
    P.update(m, ["a", "b"], ts=T0 + 15 * HL)
    assert m.g == pytest.approx(15.0)
    P.update(m, ["c", "d"], ts=T0 + 40 * HL)       # g would be 40 > 32 -> renormalise
    assert m.g <= P.PPM_RENORM_G
    assert m.t_ref == pytest.approx(T0 + 40 * HL)
    v = P.vocab(m)
    assert "old1" not in v and ("old1",) not in m.counts   # 2^-40 < 1e-9: dropped
    assert v["a"] == pytest.approx(2.0 ** -25) and v["c"] == pytest.approx(1.0)
    assert m.counts[("a",)]["b"] == pytest.approx(2.0 ** -25)   # stored == unscaled at g=0
    assert set(m.n_obs) == set(v)
    s = P.loglik(m, ["old2", "c", "d"], history=["old1"], vocab_size=V)
    assert np.all(np.isfinite(s)) and s[0] > 0
    # a long idle gap does not overflow and leaves a usable model
    P.update(m, ["e"], ts=T0 + 5000 * HL)
    assert P.vocab(m) == {"e": 1.0}
    assert m.n_tokens == pytest.approx(1.0)


def test_decay_lets_the_model_follow_a_grammar_change():
    m = _train(_shop_session, n=300, t0=T0)
    m = _train(_report_session, n=300, t0=T0 + 180 * DAY, seed=4, model=m)
    rng = random.Random(8)
    s_old = np.mean([P.loglik(m, _shop_session(rng), vocab_size=V).mean() for _ in range(50)])
    s_new = np.mean([P.loglik(m, _report_session(rng), vocab_size=V).mean() for _ in range(50)])
    assert s_new + 1.0 < s_old
    # the entropy rate is dominated by recent (report grammar) surprisal
    mu, _ = P.entropy_rate(m)
    assert abs(mu - s_new) < 1.0


def test_stats_decay_with_the_clock():
    m = P.PPMModel()
    P.update(m, ["a"], ts=T0)
    P.record_surprisal(m, np.array([2.0, 4.0]))
    mu0, sd0 = P.entropy_rate(m)
    P.update(m, ["a"], ts=T0 + HL)
    assert m.stats[0] == pytest.approx(1.0)        # W halved, (mu, sigma) unchanged
    mu1, sd1 = P.entropy_rate(m)
    assert (mu1, sd1) == pytest.approx((mu0, sd0))
    P.update(m, ["a"], ts=T0 + 2 * HL)
    assert np.isnan(P.entropy_rate(m)[0])          # W = 0.5 < 1


def test_nan_ts_means_at_the_clock():
    m = P.PPMModel(order=0)
    P.update(m, ["a"], ts=T0)
    P.update(m, ["b"], ts=T0 + HL)
    P.update(m, ["c"], ts=float("nan"))
    assert P.vocab(m)["c"] == pytest.approx(1.0)


def test_nan_ts_on_an_empty_model_waits_for_the_first_real_clock():
    """Regression: a NaN ts on an EMPTY model used to leave t_ref at epoch 0,
    so the first real ts (~656 half-lives later) renormalised the rows away."""
    m = P.PPMModel(order=1)
    P.update(m, ["a", "b"] * 20, ts=float("nan"))
    assert math.isnan(m.t_ref)                     # unanchored, not epoch 0
    P.update(m, ["c"], ts=T0)                      # the first real ts is the clock
    assert P.vocab(m) == pytest.approx({"a": 20.0, "b": 20.0, "c": 1.0})
    assert m.t_ref == pytest.approx(T0) and m.g == 0.0
    P.update(m, ["c"], ts=T0 + HL)                 # decay runs normally from here
    assert P.vocab(m)["a"] == pytest.approx(10.0)
    # an unanchored model survives JSON and still anchors afterwards
    u = P.PPMModel()
    P.update(u, ["x", "y"], ts=float("nan"))
    r = P.PPMModel.from_dict(json.loads(json.dumps(u.to_dict(), allow_nan=False)))
    assert math.isnan(r.t_ref)
    P.update(r, ["z"], ts=T0)
    assert P.vocab(r) == pytest.approx({"x": 1.0, "y": 1.0, "z": 1.0})
    # merging an unanchored model folds it in at own's clock
    own = P.PPMModel(order=1)
    P.update(own, ["q"], ts=T0)
    P.merge(own, u, w=1.0)
    assert P.vocab(own) == pytest.approx({"q": 1.0, "x": 1.0, "y": 1.0})


# ------------------------------------------------------------------ stats
def test_entropy_rate_and_record_surprisal():
    m = P.PPMModel()
    assert all(np.isnan(P.entropy_rate(m)))
    P.record_surprisal(m, np.array([0.5]), w=0.5)
    assert all(np.isnan(P.entropy_rate(m)))        # W = 0.25 < 1
    m = P.PPMModel()
    x = np.array([1.0, 2.0, 3.0, np.nan, np.inf, 6.0])
    P.record_surprisal(m, x, w=2.0)
    fin = x[np.isfinite(x)]
    mu, sd = P.entropy_rate(m)
    assert m.stats[0] == pytest.approx(2.0 * fin.size)
    assert mu == pytest.approx(fin.mean()) and sd == pytest.approx(fin.std())
    snap = list(m.stats)
    P.record_surprisal(m, np.array([]), w=1.0)
    P.record_surprisal(m, np.array([5.0]), w=0.0)
    P.record_surprisal(m, np.array([5.0]), w=float("nan"))
    assert m.stats == snap
    # constant surprisal: sigma is 0, never NaN from a tiny negative variance
    m = P.PPMModel()
    P.record_surprisal(m, np.full(1000, 0.1))
    assert P.entropy_rate(m) == (pytest.approx(0.1), 0.0)


# ------------------------------------------------------------------ attribution
def test_loglik_scores_candidates_by_their_own_models():
    """B16: a window is best explained by the model of the entity that made it."""
    sysm = P.PPMModel(order=0)
    rng = random.Random(21)
    for _ in range(200):
        P.update(sysm, _shop_session(rng) + _report_session(rng))

    def shop_b(r):          # same alphabet as A, different order / habits
        return ["login", "orders"] + [f"view{r.randrange(5)}" for _ in range(3)] + ["dashboard"]

    A = _train(_shop_session, n=200, seed=1, backoff=[sysm])
    B = _train(shop_b, n=200, seed=2, backoff=[sysm])
    R = _train(_report_session, n=200, seed=3, backoff=[sysm])
    rng = random.Random(99)
    wins = 0
    for _ in range(100):
        s = _shop_session(rng)
        la = P.loglik(A, s, [sysm], V).sum()
        lb = P.loglik(B, s, [sysm], V).sum()
        lr = P.loglik(R, s, [sysm], V).sum()
        assert np.isfinite(la) and np.isfinite(lb) and np.isfinite(lr)
        wins += la < min(lb, lr)
    assert wins >= 98


# ------------------------------------------------------------------ merge
def test_merge_into_empty_equals_other():
    other = _train(_shop_session, n=50)
    own = P.merge(P.PPMModel(), other, w=1.0)
    probe = ["login", "dashboard", "orders", "view2", "zz"]
    np.testing.assert_allclose(P.loglik(own, probe, vocab_size=V),
                               P.loglik(other, probe, vocab_size=V), rtol=1e-12)
    assert P.vocab(own) == pytest.approx(P.vocab(other))
    assert own.stats == pytest.approx(other.stats)
    assert own.n_obs == other.n_obs


def test_merge_weight_and_time_alignment():
    other = P.PPMModel(order=1)
    P.update(other, ["a", "b"], ts=T0)
    own = P.PPMModel(order=1)
    P.update(own, ["c"], ts=T0 + HL)               # own's clock is one half-life newer
    P.merge(own, other, w=0.5)
    v = P.vocab(own)
    assert v["a"] == pytest.approx(0.25) and v["b"] == pytest.approx(0.25)
    assert v["c"] == pytest.approx(1.0)
    assert own.counts[("a",)]["b"] * 2.0 ** -own.g == pytest.approx(0.25)
    assert own.n_obs == {"a": 1, "b": 1, "c": 1}
    # other newer than own: own's clock advances and own decays
    own2 = P.PPMModel(order=1)
    P.update(own2, ["c"], ts=T0)
    other2 = P.PPMModel(order=1)
    P.update(other2, ["a"], ts=T0 + HL)
    P.merge(own2, other2, w=1.0)
    assert P.vocab(own2) == pytest.approx({"c": 0.5, "a": 1.0})
    # contexts beyond own.order are dropped; no-ops
    hi = P.PPMModel(order=3)
    P.update(hi, ["a", "b", "c", "d"])
    lo = P.merge(P.PPMModel(order=1), hi, w=1.0)
    assert max(len(c) for c in lo.counts) == 1
    snap = _snapshot(lo)
    for w in (0.0, -1.0, float("nan")):
        P.merge(lo, hi, w=w)
    P.merge(lo, None)
    P.merge(lo, lo)
    P.merge(lo, P.PPMModel())
    assert _snapshot(lo) == snap


def test_merge_of_an_ancient_model_creates_no_dead_entries():
    """Regression: merge created entries with ~0 mass (2^-61 here) for symbols
    own had never seen; each counted as a distinct symbol in q and inflated
    every escape (P(b | a) fell from 0.95 to 0.87). update() already refuses
    rows below the 1e-9 'seen' threshold; merge now does the same."""
    own = P.PPMModel(order=1)
    P.update(own, ["a", "b"] * 20, ts=T0 + 60 * HL)
    before = P.loglik(own, ["a", "b"], vocab_size=20)
    snap_counts = copy.deepcopy(own.counts)
    old = P.PPMModel(order=1)
    P.update(old, ["a", "x", "a", "y"], ts=T0)
    P.merge(own, old, w=0.5)
    assert "x" not in P.vocab(own) and "y" not in P.vocab(own)
    assert set(own.n_obs) == {"a", "b"} and own.n_obs["a"] == 20 + 2
    assert set(own.counts) == set(snap_counts)
    assert set(own.counts[("a",)]) == {"b"}
    np.testing.assert_allclose(P.loglik(own, ["a", "b"], vocab_size=20), before, rtol=1e-9)


def test_from_dict_with_an_out_of_range_scale_does_not_overflow():
    """Regression: a stored g beyond the renormalisation range made
    loglik's 2**g raise OverflowError; from_dict now renormalises."""
    m = P.PPMModel(order=1)
    P.update(m, ["a", "b", "a"], ts=T0)
    d = m.to_dict()
    d["g"] = 40.0                                  # counts stored at 2^0, clock 40 HL later
    r = P.PPMModel.from_dict(d)
    assert r.g == 0.0 and r.t_ref == pytest.approx(T0 + 40 * HL)
    assert P.vocab(r) == {}                         # 2^-40 < 1e-9: all dead
    d["g"] = 5000.0
    s = P.PPMModel.from_dict(d)
    assert np.all(np.isfinite(P.loglik(s, ["a", "b"], vocab_size=8)))
    nd = P.PPMModel(half_life_s=math.inf)
    P.update(nd, ["a"], ts=T0)
    d = nd.to_dict()
    d["g"] = 33.0
    r = P.PPMModel.from_dict(d)
    assert r.t_ref == T0 and r.g == 0.0            # no decay: t_ref is not moved


def test_merge_of_a_no_decay_model_uses_its_clock():
    """Regression: a no-decay model (half_life_s=inf) never set t_ref, so
    merge() placed it at epoch 0 and a decaying `own` dropped all of it."""
    nd = P.PPMModel(order=1, half_life_s=math.inf)
    P.update(nd, ["a", "b"], ts=T0)
    P.update(nd, ["c"], ts=T0 + 2 * HL)
    P.update(nd, ["d"], ts=T0 + HL)                # older row: the clock stays newest
    assert nd.t_ref == T0 + 2 * HL
    own = P.PPMModel(order=1)
    P.update(own, ["z"], ts=T0 + 2 * HL)
    P.merge(own, nd, w=1.0)
    assert P.vocab(own) == pytest.approx({"z": 1.0, "a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0})


# ------------------------------------------------------------------ serialisation
def test_to_dict_from_dict_round_trip_is_exact_and_json_safe():
    m = _train(_shop_session, n=80)
    P.update(m, [("tls", 443), ("tls", 443), "login"], ts=T0 + 100 * DAY)
    d = m.to_dict()
    js = json.dumps(d, allow_nan=False)
    r = P.PPMModel.from_dict(json.loads(js))
    assert r.order == m.order and r.half_life_s == m.half_life_s
    assert r.t_ref == m.t_ref and r.g == m.g and r.stats == m.stats
    assert r.counts == m.counts and r.n_obs == m.n_obs
    probe = ["login", ("tls", 443), "orders", "view1", "nope"]
    np.testing.assert_allclose(P.loglik(r, probe, vocab_size=V),
                               P.loglik(m, probe, vocab_size=V), rtol=1e-12)
    assert P.good_turing_unseen(r) == pytest.approx(P.good_turing_unseen(m))
    assert P.entropy_rate(r) == P.entropy_rate(m)
    # continued learning from the restored copy matches the original
    P.update(m, ["login", "export"], ts=T0 + 101 * DAY)
    P.update(r, ["login", "export"], ts=T0 + 101 * DAY)
    np.testing.assert_allclose(P.loglik(r, probe, vocab_size=V),
                               P.loglik(m, probe, vocab_size=V), rtol=1e-12)


def test_to_dict_is_a_deep_copy_and_from_dict_edge_cases():
    m = P.PPMModel()
    P.update(m, ["a", "b"])
    d = m.to_dict()
    frozen = copy.deepcopy(d)
    P.update(m, ["a", "c"])
    assert d == frozen
    assert P.PPMModel.from_dict(None) == P.PPMModel()
    assert P.PPMModel.from_dict({}) == P.PPMModel()
    assert P.PPMModel.from_dict(m) is m
    # numpy scalar tokens serialise as plain ints and still match int lookups
    n = P.PPMModel(order=1)
    P.update(n, list(np.array([5, 6, 5], dtype=np.int64)))
    r = P.PPMModel.from_dict(json.loads(json.dumps(n.to_dict(), allow_nan=False)))
    np.testing.assert_allclose(P.loglik(r, [5, 6], vocab_size=10),
                               P.loglik(n, [5, 6], vocab_size=10))
    # no decay (infinite half-life) survives JSON
    nd = P.PPMModel(half_life_s=math.inf)
    P.update(nd, ["a"], ts=T0)
    P.update(nd, ["b"], ts=T0 + 400 * DAY)
    assert P.vocab(nd) == {"a": 1.0, "b": 1.0}
    r = P.PPMModel.from_dict(json.loads(json.dumps(nd.to_dict(), allow_nan=False)))
    assert math.isinf(r.half_life_s) and P.vocab(r) == {"a": 1.0, "b": 1.0}


# ------------------------------------------------------------------ perf
def test_cost_per_token():
    rng = random.Random(0)
    toks = [t for _ in range(1500) for t in _shop_session(rng)]
    m = P.PPMModel()
    t = time.perf_counter()
    P.update(m, toks, ts=T0)
    t_up = (time.perf_counter() - t) / len(toks)
    t = time.perf_counter()
    P.loglik(m, toks, vocab_size=V)
    t_ll = (time.perf_counter() - t) / len(toks)
    # escape-heavy path through three tiers
    rw = [f"r{rng.randrange(200)}" for _ in range(3000)]
    cls, sysm = P.PPMModel(), P.PPMModel(order=0)
    P.update(cls, rw[:1500])
    P.update(sysm, rw[:3000:2] + toks[:3000])
    t = time.perf_counter()
    s = P.loglik(m, rw, [cls, sysm], vocab_size=400)
    t_esc = (time.perf_counter() - t) / len(rw)
    assert np.all(np.isfinite(s))
    # spec: ~3-8 us per token; generous bounds so a loaded CI box stays green
    assert t_up < 25e-6 and t_ll < 25e-6 and t_esc < 60e-6, (t_up, t_ll, t_esc)
