"""Tests for engines/behavior/lib/sketch.py: the signed-hash Hellinger sketch
(feature.sketch, B01 step 5) and the p = 10 HyperLogLog (act.objs, B13 breadth).

Every formula is checked against an independent re-implementation written
from the spec text (blake2b of f'{ns}\\x1f{tok}' read big-endian, bucket =
x % 16, sign = bit 32; HLL idx = top 10 bits, rho = leading zeros of the
remaining 54 bits + 1), plus golden values so a silent change of the hash
(byte order, separator, encoding) cannot slip through: feature.sketch
history and every identity model built on it would be invalidated.

The HLL accuracy targets (< 5 % at n = 2000 and n = 100000) are ~1.5 sigma
of the p = 10 estimator (1.04 / sqrt(1024) = 3.25 %), so they are checked
on the canonical id sets and, statistically, over many independent sets.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import math
import os
import pickle
import subprocess
import sys
import time
from collections import Counter

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..", "..", "backend")
sys.path.insert(0, BACKEND)

from app.engines.behavior.lib import sketch as S  # noqa: E402

NAN = float("nan")
INF = float("inf")


# ------------------------------------------------------------- references
def ref_hash64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def ref_token_hash(ns: str, tok) -> tuple:
    x = ref_hash64(f"{ns}\x1f{tok}")
    return x % 16, (1 if (x >> 32) & 1 else -1)


def ref_block(counts: dict, ns: str) -> np.ndarray:
    """The B01 formula verbatim: v[b] = sum s sqrt(c / C), renormalised."""
    items = [(str(t), float(c)) for t, c in counts.items()
             if str(t) != "__other__" and math.isfinite(float(c)) and float(c) > 0]
    if not items:
        return np.zeros(16)
    total = math.fsum(c for _, c in items)
    v = np.zeros(16)
    u = np.zeros(16)
    for t, c in items:
        b, s = ref_token_hash(ns, t)
        v[b] += s * math.sqrt(c / total)
        u[b] += math.sqrt(c / total)
    n = np.linalg.norm(v)
    if n < 1e-12:
        v, n = u, np.linalg.norm(u)
    return v / n


def ref_registers(items) -> np.ndarray:
    reg = np.zeros(1024, dtype=np.uint8)
    for it in items:
        x = ref_hash64(str(it))
        idx = x >> 54
        bits = format(x & ((1 << 54) - 1), "054b")
        rho = (len(bits) - len(bits.lstrip("0"))) + 1        # leading zeros + 1
        reg[idx] = max(reg[idx], rho)
    return reg


def hll_of(items) -> S.HyperLogLog:
    h = S.HyperLogLog()
    h.add_many(items)
    return h


def rel_err(h: S.HyperLogLog, n: int) -> float:
    return (h.count() - n) / n


def find_pair(ns: str, same_sign: bool) -> tuple:
    """Two tokens that collide in one bucket, with equal or opposite signs."""
    seen = {}
    for i in range(10_000):
        t = f"t{i}"
        b, s = ref_token_hash(ns, t)
        other = seen.get((b, s if same_sign else -s))
        if other is not None:
            return other, t, b
        seen.setdefault((b, s), t)
    raise AssertionError("no colliding pair found")


def distinct_bucket_tokens(ns: str, k: int) -> list:
    out, used = [], set()
    for i in range(10_000):
        t = f"d{i}"
        b, _ = ref_token_hash(ns, t)
        if b not in used:
            used.add(b)
            out.append(t)
            if len(out) == k:
                return out
    raise AssertionError


# ------------------------------------------------------------------ contract
def test_constants_frozen():
    assert S.SKETCH_BLOCK == 16
    assert S.SKETCH_NAMESPACES == ("act.tokens", "client.stack_set", "sni_etld1",
                                   "dns_etld1", "l4.dport_set")
    assert S.SKETCH_DIM == 80
    assert S.OTHER_TOKEN == "__other__"
    assert S.HLL_P == 10 and S.HLL_M == 1024
    assert S.HLL_ALPHA == pytest.approx(0.7213 / (1 + 1.079 / 1024), rel=1e-15)


# -------------------------------------------------------------- token_hash
GOLDEN_TOKENS = [
    ("act.tokens", "GET api.example.com /orders/view/{num}|2xx", (3, -1)),
    ("client.stack_set", "chrome/120|win", (5, -1)),
    ("sni_etld1", "example.com", (2, 1)),
    ("dns_etld1", "example.cn", (9, 1)),
    ("l4.dport_set", "443", (6, -1)),
    ("act.tokens", "", (10, -1)),
]


@pytest.mark.parametrize("ns,tok,expected", GOLDEN_TOKENS)
def test_token_hash_golden(ns, tok, expected):
    assert S.token_hash(ns, tok) == expected
    assert ref_token_hash(ns, tok) == expected


def test_token_hash_matches_spec_reference():
    for ns in S.SKETCH_NAMESPACES + ("other.ns", ""):
        for i in range(300):
            tok = f"tok-{i}-中文" if i % 3 == 0 else f"tok-{i}"
            b, s = S.token_hash(ns, tok)
            assert (b, s) == ref_token_hash(ns, tok)
            assert type(b) is int and 0 <= b < 16 and s in (-1, 1)


def test_token_hash_non_str_tokens_use_str_form():
    # l4.dport_set keys may arrive as ints: 443 and '443' are the same token
    for p in (22, 53, 443, 8443, 65535):
        assert S.token_hash("l4.dport_set", p) == S.token_hash("l4.dport_set", str(p))
    # the cache is keyed on str(tok), so True and 1 (equal and same hash) never alias
    h_true = S.token_hash("act.tokens", True)
    h_one = S.token_hash("act.tokens", 1)
    assert h_true == ref_token_hash("act.tokens", "True")
    assert h_one == ref_token_hash("act.tokens", "1")
    assert S.token_hash("act.tokens", True) == h_true         # cached value, not aliased


def test_token_hash_namespace_is_part_of_the_key():
    toks = [f"x{i}" for i in range(200)]
    a = [S.token_hash("sni_etld1", t) for t in toks]
    b = [S.token_hash("dns_etld1", t) for t in toks]
    assert sum(x != y for x, y in zip(a, b)) > 150           # ~ 1 - 1/32 differ
    # the separator matters: ('ab', 'c') is not ('a', 'bc')
    assert S.token_hash("ab", "c") == ref_token_hash("ab", "c")
    assert S.token_hash("a", "bc") == ref_token_hash("a", "bc")


def test_token_hash_surrogate_does_not_raise():
    b, s = S.token_hash("act.tokens", "bad\udcff")
    assert 0 <= b < 16 and s in (-1, 1)


def test_token_hash_uniform_and_balanced():
    n = 16_000
    hs = [S.token_hash("act.tokens", f"u{i}") for i in range(n)]
    buckets = np.bincount([b for b, _ in hs], minlength=16)
    chi2 = float(((buckets - n / 16) ** 2 / (n / 16)).sum())
    assert chi2 < 45                         # chi2_15 sf(45) ~ 7e-5
    plus = sum(1 for _, s in hs if s > 0)
    assert abs(plus - n / 2) < 4 * math.sqrt(n / 4)
    # sign independent of bucket: every bucket has both signs in similar shares
    for b in range(16):
        sb = [s for bb, s in hs if bb == b]
        assert 0.4 < sum(1 for s in sb if s > 0) / len(sb) < 0.6


def test_token_hash_deterministic_across_processes():
    toks = [("act.tokens", f"t{i}") for i in range(50)] + [("l4.dport_set", "443")]
    here = [S.token_hash(ns, t) for ns, t in toks]
    code = ("import sys; sys.path.insert(0, sys.argv[1]);"
            "from app.engines.behavior.lib import sketch as S;"
            "print([S.token_hash('act.tokens', f't{i}') for i in range(50)]"
            " + [S.token_hash('l4.dport_set', '443')])")
    for seed in ("1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run([sys.executable, "-c", code, os.path.abspath(BACKEND)],
                             capture_output=True, text=True, env=env, check=True)
        assert ast.literal_eval(out.stdout.strip()) == here


def test_hash_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(S, "_CODE_CACHE_MAX", 32)
    monkeypatch.setattr(S, "_NS_CACHE_MAX", 4)
    monkeypatch.setattr(S, "_CODES", {})
    for i in range(500):
        assert S.token_hash("act.tokens", f"c{i}") == ref_token_hash("act.tokens", f"c{i}")
    assert len(S._CODES["act.tokens"]) <= 32
    for j in range(20):
        assert S.token_hash(f"ns{j}", "tok") == ref_token_hash(f"ns{j}", "tok")
    assert len(S._CODES) <= 4
    blk = S.sketch_block({f"c{i}": i + 1 for i in range(100)}, "act.tokens")
    np.testing.assert_allclose(blk, ref_block({f"c{i}": i + 1 for i in range(100)}, "act.tokens"),
                               atol=1e-12)


# ------------------------------------------------------------ sketch_block
def test_block_single_token_is_signed_one_hot():
    for tok in ("GET a.com /x|2xx", "b", "443"):
        blk = S.sketch_block({tok: 7.0}, "act.tokens")
        b, s = ref_token_hash("act.tokens", tok)
        expect = np.zeros(16)
        expect[b] = s
        assert blk.dtype == np.float64 and blk.shape == (16,)
        np.testing.assert_array_equal(blk, expect)


def test_block_without_collisions_is_sqrt_distribution():
    ns = "sni_etld1"
    toks = distinct_bucket_tokens(ns, 6)
    counts = dict(zip(toks, [1.0, 2.0, 3.0, 4.0, 10.0, 80.0]))
    blk = S.sketch_block(counts, ns)
    total = sum(counts.values())
    for t, c in counts.items():
        b, s = ref_token_hash(ns, t)
        assert blk[b] == pytest.approx(s * math.sqrt(c / total), abs=1e-15)
    # Hellinger embedding: the dot product of two blocks is the Bhattacharyya coefficient
    other = dict(zip(toks, [5.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    bc = sum(math.sqrt(counts[t] / total * other[t] / 10.0) for t in toks)
    assert float(S.sketch_block(other, ns) @ blk) == pytest.approx(bc, abs=1e-12)


def test_block_matches_reference_formula_random():
    rng = np.random.default_rng(7)
    for trial in range(300):
        k = int(rng.integers(1, 120))           # spans the pure-Python and numpy paths
        ns = S.SKETCH_NAMESPACES[trial % 5]
        counts = {f"r{trial}-{i}": float(c) for i, c in enumerate(rng.gamma(0.5, 10.0, k))}
        blk = S.sketch_block(counts, ns)
        np.testing.assert_allclose(blk, ref_block(counts, ns), atol=1e-12, rtol=0)
        assert abs(np.linalg.norm(blk) - 1.0) <= 1e-9


def test_block_small_and_large_paths_agree():
    # the same distribution computed below and above the pure-Python cutoff
    base = {f"s{i}": float(i % 5 + 1) for i in range(S._SMALL_BLOCK)}
    blk_small = S.sketch_block(base, "act.tokens")
    padded = dict(base)
    padded.update({f"z{i}": 1e-300 for i in range(40)})   # p ~ 1e-301 each: no effect
    blk_large = S.sketch_block(padded, "act.tokens")
    np.testing.assert_allclose(blk_small, blk_large, atol=1e-14)


def test_block_other_token_excluded_from_numerator_and_total():
    counts = {"a": 3.0, "b": 1.0, "c": 5.0}
    with_other = dict(counts, __other__=1000.0)
    np.testing.assert_array_equal(S.sketch_block(with_other, "act.tokens"),
                                  S.sketch_block(counts, "act.tokens"))
    np.testing.assert_array_equal(S.sketch_block({"__other__": 5.0}, "act.tokens"), np.zeros(16))


@pytest.mark.parametrize("counts", [
    {}, None, {"a": 0.0}, {"a": -3.0, "b": 0}, {"a": NAN}, {"a": INF, "b": -INF},
    {"a": "x"}, {"a": None}, {"a": 10 ** 400}, {"a": {"bytes": 3}}, [],
])
def test_block_empty_or_invalid_counts_give_zeros(counts):
    blk = S.sketch_block(counts, "act.tokens")
    assert blk.shape == (16,) and blk.dtype == np.float64
    np.testing.assert_array_equal(blk, np.zeros(16))


def test_block_invalid_counts_ignored_among_valid():
    clean = {"a": 2.0, "b": 3.0}
    dirty = dict(clean, c=NAN, d=-1.0, e=INF, f=0.0, g="junk", h=None)
    np.testing.assert_array_equal(S.sketch_block(dirty, "dns_etld1"),
                                  S.sketch_block(clean, "dns_etld1"))


def test_block_numeric_text_is_not_a_count():
    # regression: float('3') parses, so text counts used to be accepted despite
    # the contract ("non-numeric counts are ignored"); also via the {'n': ...} shape
    clean = {"a": 2.0}
    dirty = dict(clean, b="3", c=b"4", d=bytearray(b"5"), e={"n": "6"})
    np.testing.assert_array_equal(S.sketch_block(dirty, "act.tokens"),
                                  S.sketch_block(clean, "act.tokens"))
    np.testing.assert_array_equal(S.sketch_block({"a": "3"}, "act.tokens"), np.zeros(16))


def test_block_opposite_sign_collision_gives_no_nan():
    ns = "act.tokens"
    t1, t2, b = find_pair(ns, same_sign=False)
    blk = S.sketch_block({t1: 4.0, t2: 4.0}, ns)            # exact cancellation
    assert np.all(np.isfinite(blk))
    expect = np.zeros(16)
    expect[b] = 1.0                                           # unsigned fallback
    np.testing.assert_array_equal(blk, expect)
    # cancellation plus an unrelated token: the block does not cancel as a whole,
    # the colliding bucket is simply 0 and the rest is renormalised
    t3 = next(f"q{i}" for i in range(1000) if ref_token_hash(ns, f"q{i}")[0] != b)
    blk = S.sketch_block({t1: 4.0, t2: 4.0, t3: 1.0}, ns)
    b3, s3 = ref_token_hash(ns, t3)
    assert np.all(np.isfinite(blk)) and blk[b] == 0.0 and blk[b3] == pytest.approx(s3)
    # partial cancellation keeps the sign of the heavier token
    blk = S.sketch_block({t1: 9.0, t2: 1.0}, ns)
    s1 = ref_token_hash(ns, t1)[1]
    assert blk[b] == pytest.approx(s1) and np.linalg.norm(blk) == pytest.approx(1.0, abs=1e-12)


def test_block_near_cancellation_uses_residual_not_fallback():
    ns = "act.tokens"
    t1, t2, b = find_pair(ns, same_sign=False)
    blk = S.sketch_block({t1: 1.0, t2: 1.0 + 1e-6}, ns)       # residual ~2.5e-7 >> 1e-12
    s2 = ref_token_hash(ns, t2)[1]
    assert blk[b] == pytest.approx(s2)


def test_block_same_sign_collision_renormalised():
    ns = "sni_etld1"
    t1, t2, b = find_pair(ns, same_sign=True)
    blk = S.sketch_block({t1: 1.0, t2: 1.0}, ns)
    s = ref_token_hash(ns, t1)[1]
    expect = np.zeros(16)
    expect[b] = s
    np.testing.assert_allclose(blk, expect, atol=1e-15)


def test_block_norm_fuzz_never_nan():
    rng = np.random.default_rng(11)
    pool = [f"f{i}" for i in range(40)]                       # few tokens: many collisions
    for trial in range(500):
        k = int(rng.integers(1, 30))
        toks = rng.choice(pool, size=k, replace=False)
        vals = rng.choice([0.0, -1.0, NAN, 1.0, 1.0, 2.0, 1e-9, 1e9, INF], size=k)
        counts = dict(zip(toks.tolist(), vals.tolist()))
        blk = S.sketch_block(counts, S.SKETCH_NAMESPACES[trial % 5])
        assert np.all(np.isfinite(blk))
        valid = any(math.isfinite(v) and v > 0 for v in vals)
        if valid:
            assert abs(float(np.linalg.norm(blk)) - 1.0) <= 1e-9
        else:
            assert not blk.any()


def test_block_cadence_and_scale_invariant():
    counts = {f"k{i}": float(i + 1) for i in range(40)}
    base = S.sketch_block(counts, "act.tokens")
    for scale in (15.0, 1e-300, 1e300, 1 / 7):
        scaled = {t: c * scale for t, c in counts.items()}
        np.testing.assert_allclose(S.sketch_block(scaled, "act.tokens"), base, atol=1e-14)
    # counts so large their sum overflows float64 still give the right block
    huge = {"a": 1.5e308, "b": 1.5e308, "c": 0.75e308}
    np.testing.assert_allclose(S.sketch_block(huge, "l4.dport_set"),
                               S.sketch_block({"a": 2.0, "b": 2.0, "c": 1.0}, "l4.dport_set"),
                               atol=1e-14)


def test_block_accepts_store_value_shapes():
    ns = "client.stack_set"
    nested = {"s1": {"n": 3, "bytes": 10, "first_ts": 0.0, "last_ts": 1.0},
              "s2": {"n": 1, "bytes": 99}, "s3": {"bytes": 5}}           # no n: ignored
    np.testing.assert_array_equal(S.sketch_block(nested, ns),
                                  S.sketch_block({"s1": 3.0, "s2": 1.0}, ns))
    # a set of tokens counts each once, a list counts occurrences
    np.testing.assert_array_equal(S.sketch_block({"a", "b", "c"}, ns),
                                  S.sketch_block({"a": 1, "b": 1, "c": 1}, ns))
    np.testing.assert_array_equal(S.sketch_block(["a", "b", "a"], ns),
                                  S.sketch_block(Counter({"a": 2, "b": 1}), ns))
    # numpy scalar and int values, int keys (l4.dport_set)
    ports = {443: np.int64(5), 53: np.float32(2.0), 22: 1}
    np.testing.assert_allclose(S.sketch_block(ports, "l4.dport_set"),
                               S.sketch_block({"443": 5.0, "53": 2.0, "22": 1.0}, "l4.dport_set"),
                               atol=1e-15)
    with pytest.raises(TypeError):
        S.sketch_block("abc", ns)


def test_block_disjoint_sets_are_orthogonal_on_average():
    # random signs make collisions unbiased: E<u, v> = Bhattacharyya = 0 for disjoint
    # token sets. Different namespaces are independent hash functions.
    u = {f"u{i}": 1.0 for i in range(5)}
    v = {f"v{i}": 1.0 for i in range(5)}
    dots = [float(S.sketch_block(u, f"ns{j}") @ S.sketch_block(v, f"ns{j}")) for j in range(300)]
    assert abs(np.mean(dots)) < 0.1
    assert float(S.sketch_block(u, "ns0") @ S.sketch_block(dict(u), "ns0")) == pytest.approx(1.0)


def test_block_returns_fresh_array():
    a = S.sketch_block({"x": 1.0}, "act.tokens")
    a[:] = 99.0
    b = S.sketch_block({"x": 1.0}, "act.tokens")
    assert np.abs(b).max() == 1.0


# ----------------------------------------------------------- sketch_vector
def _nsc() -> dict:
    return {
        "act.tokens": {f"GET h{i % 7}.com /p/{{num}}|2xx{i}": float(i + 1) for i in range(256)}
        | {"__other__": 50.0},
        "client.stack_set": {f"stack{i}": {"n": i + 1, "bytes": 10} for i in range(20)},
        "sni_etld1": {f"d{i}.com": i + 1 for i in range(30)},
        "dns_etld1": {f"q{i}.cn": i + 1 for i in range(30)},
        "l4.dport_set": {p: i + 1 for i, p in enumerate(range(1000, 1064))},
    }


def test_vector_layout_dtype_and_blocks():
    nsc = _nsc()
    vec = S.sketch_vector(nsc)
    assert vec.shape == (80,) and vec.dtype == np.float32
    for i, ns in enumerate(S.SKETCH_NAMESPACES):
        blk = vec[i * 16:(i + 1) * 16]
        ref = S.sketch_block(nsc[ns], ns)
        np.testing.assert_allclose(blk, ref.astype(np.float32), rtol=0, atol=0)
        # float32 storage: unit norm to float32 precision
        assert abs(float(np.linalg.norm(blk.astype(np.float64))) - 1.0) < 1e-6


def test_vector_missing_unknown_and_empty_namespaces():
    vec = S.sketch_vector({"sni_etld1": {"a.com": 1.0}, "unknown.ns": {"z": 5.0},
                           "dns_etld1": {}, "l4.dport_set": None})
    assert vec.shape == (80,)
    assert not vec[:32].any() and not vec[48:].any()           # only sni block is set
    assert abs(float(np.linalg.norm(vec[32:48])) - 1.0) < 1e-6
    np.testing.assert_array_equal(S.sketch_vector({}), np.zeros(80, np.float32))
    np.testing.assert_array_equal(S.sketch_vector(None), np.zeros(80, np.float32))


def test_vector_deterministic_and_order_independent():
    nsc = _nsc()
    a = S.sketch_vector(nsc)
    rev = {ns: dict(reversed(list(c.items()))) for ns, c in reversed(list(nsc.items()))}
    b = S.sketch_vector(rev)
    np.testing.assert_allclose(a, b, atol=1e-7)                # summation order only
    np.testing.assert_array_equal(a, S.sketch_vector(nsc))


def test_vector_performance_budget():
    # B01 budget is 0.3 ms per entity for everything; a heavy entity's sketch
    # (256 act.tokens + 144 other tokens) must stay well inside it once warm.
    nsc = _nsc()
    S.sketch_vector(nsc)
    reps = 50
    t = time.perf_counter()
    for _ in range(reps):
        S.sketch_vector(nsc)
    per_call = (time.perf_counter() - t) / reps
    assert per_call < 2e-3                  # measured ~0.16 ms; generous for loaded CI


# -------------------------------------------------------------- HyperLogLog
def test_hll_empty():
    h = S.HyperLogLog()
    assert h.count() == 0.0
    assert h.to_bytes() == bytes(1024)
    assert h.registers.dtype == np.uint8 and h.registers.shape == (1024,)
    assert S.HyperLogLog().registers is not h.registers        # no shared default


def test_hll_golden_registers():
    for item, idx, rho in (("", 914, 1), ("a", 259, 1), (12345, 465, 2), ("obj-0", 266, 1)):
        h = S.HyperLogLog()
        h.add(item)
        assert np.flatnonzero(h.registers).tolist() == [idx]
        assert int(h.registers[idx]) == rho


def test_hll_add_matches_reference_and_add_many():
    items = ([f"id-{i}" for i in range(1500)] + list(range(700)) + [3.5, ("t", 1), None, b"raw"]
             + ["中文", ""])
    ref = ref_registers(items)
    one = S.HyperLogLog()
    for it in items:
        one.add(it)
    np.testing.assert_array_equal(one.registers, ref)
    np.testing.assert_array_equal(hll_of(items).registers, ref)
    np.testing.assert_array_equal(hll_of(iter(items)).registers, ref)   # generator input
    small = items[:5]                                                    # scalar path
    np.testing.assert_array_equal(hll_of(small).registers, ref_registers(small))


def test_hll_add_many_chunking(monkeypatch):
    items = [f"c{i}" for i in range(1000)]
    full = hll_of(items)
    monkeypatch.setattr(S, "_ADD_CHUNK", 37)                             # 27 chunks + tail
    np.testing.assert_array_equal(hll_of(items).registers, full.registers)
    monkeypatch.setattr(S, "_ADD_CHUNK", 1)                              # all scalar path
    np.testing.assert_array_equal(hll_of(items).registers, full.registers)


def test_hll_bit_length_exact_at_float_boundaries():
    vals = [0, 1, 2, 3, (1 << 27) - 1, 1 << 27, (1 << 27) + 1, (1 << 53) - 1, 1 << 53,
            (1 << 53) + 1, (1 << 54) - 1, (1 << 54) - 2, (1 << 53) + (1 << 52) - 1]
    rng = np.random.default_rng(3)
    vals += [int(v) >> int(s) for v, s in zip(rng.integers(0, 1 << 54, 2000, dtype=np.uint64),
                                              rng.integers(0, 54, 2000))]
    got = S._bit_length_u54(np.array(vals, dtype=np.uint64))
    assert got.tolist() == [v.bit_length() for v in vals]


def test_hll_str_semantics_and_duplicates():
    a, b = S.HyperLogLog(), S.HyperLogLog()
    a.add(5)
    b.add("5")
    assert a == b
    h = S.HyperLogLog()
    for _ in range(1000):
        h.add("same")
    h.add_many(["same"] * 1000)
    assert h == hll_of(["same"])
    assert h.count() == pytest.approx(1024 * math.log(1024 / 1023))   # ~1.0005


@pytest.mark.parametrize("n", [2000, 100_000])
def test_hll_accuracy_targets_canonical(n):
    # the two sizes of the targets, on the id sets an engine test would use
    assert abs(rel_err(hll_of(range(n)), n)) < 0.05
    assert abs(rel_err(hll_of(f"obj-{i}" for i in range(n)), n)) < 0.05


def test_hll_accuracy_n2000_over_many_sets():
    # n = 2000 < 2.5 m: linear counting; theory sd ~3.2 %, so ~12 % of sets exceed 5 %
    errs = np.array([rel_err(hll_of(f"s{s}-{i}" for i in range(2000)), 2000) for s in range(150)])
    assert abs(errs.mean()) < 0.01                  # unbiased (SE of the mean ~0.26 %)
    assert 0.02 < math.sqrt((errs ** 2).mean()) < 0.042
    assert (np.abs(errs) < 0.05).mean() >= 0.8
    assert np.abs(errs).max() < 0.13                # 4 sigma


def test_hll_accuracy_n100000_over_sets():
    # raw HLL regime; theory sd 1.04 / 32 = 3.25 %
    errs = np.array([rel_err(hll_of(range(s * 10 ** 7, s * 10 ** 7 + 100_000)), 100_000)
                     for s in range(6)])
    assert math.sqrt((errs ** 2).mean()) < 0.055
    assert np.abs(errs).max() < 0.12


@pytest.mark.parametrize("n", [10, 100, 700, 2600, 5000, 20_000])
def test_hll_accuracy_across_regimes(n):
    # spans linear counting, the 2.5 m switch and the raw estimator (4 sigma bounds)
    sd = max(1.04 / 32, 0.005)
    for s in range(3):
        e = rel_err(hll_of(f"r{n}-{s}-{i}" for i in range(n)), n)
        assert abs(e) < 4 * sd, (n, s, e)


def test_hll_small_counts_near_exact():
    # linear counting is almost exact while collisions are rare
    for n in (1, 2, 5, 20, 50):
        assert abs(hll_of(f"e{i}" for i in range(n)).count() - n) < max(0.6, 0.03 * n)


def test_hll_merge_equals_union():
    a = hll_of(f"m{i}" for i in range(0, 6000))
    b = hll_of(f"m{i}" for i in range(4000, 9000))
    union = hll_of(f"m{i}" for i in range(0, 9000))
    a_copy = S.HyperLogLog.from_bytes(a.to_bytes())
    out = a.merge(b)
    assert out is a                                             # in place, returns self
    assert a == union                                           # lossless union
    assert abs(rel_err(a, 9000)) < 0.13
    # commutative and idempotent; the argument is not modified
    b_before = b.to_bytes()
    assert S.HyperLogLog.from_bytes(b.to_bytes()).merge(a_copy) == union
    assert b.to_bytes() == b_before
    assert a.merge(a) == union
    assert a.merge(S.HyperLogLog()) == union


def test_hll_merge_type_error():
    with pytest.raises(TypeError):
        S.HyperLogLog().merge(b"\x00" * 1024)


def test_hll_bytes_roundtrip():
    h = hll_of(f"b{i}" for i in range(5000))
    raw = h.to_bytes()
    assert isinstance(raw, bytes) and len(raw) == 1024
    back = S.HyperLogLog.from_bytes(raw)
    assert back == h and back.count() == h.count()
    for form in (bytearray(raw), memoryview(raw)):
        assert S.HyperLogLog.from_bytes(form) == h
    # the decoded sketch owns its registers: writable, not aliasing the payload
    buf = bytearray(raw)
    dec = S.HyperLogLog.from_bytes(buf)
    buf[:] = bytes(1024)
    assert dec == h
    dec.add("new-item")
    dec.merge(h)


@pytest.mark.parametrize("payload,exc", [
    (b"\x00" * 1023, ValueError), (b"\x00" * 1025, ValueError), (b"", ValueError),
    (b"\x38" + b"\x00" * 1023, ValueError),                  # register 56 > 55: corrupt
    ("0" * 1024, TypeError), (list(range(1024)), TypeError),
])
def test_hll_from_bytes_rejects_bad_payload(payload, exc):
    with pytest.raises(exc):
        S.HyperLogLog.from_bytes(payload)


def test_hll_constructor_validation():
    with pytest.raises(ValueError):
        S.HyperLogLog(registers=np.zeros(1000, dtype=np.uint8))
    with pytest.raises(ValueError):
        S.HyperLogLog(registers=np.zeros(1024, dtype=np.float64))
    with pytest.raises(ValueError):
        S.HyperLogLog(registers=np.full(1024, -1, dtype=np.int64))
    with pytest.raises(ValueError):
        S.HyperLogLog(registers=np.full(1024, 60, dtype=np.uint8))
    h = S.HyperLogLog(registers=np.full(1024, 3, dtype=np.int64))
    assert h.registers.dtype == np.uint8 and int(h.registers.sum()) == 3 * 1024
    ro = np.zeros(1024, dtype=np.uint8)
    ro.flags.writeable = False
    h = S.HyperLogLog(registers=ro)
    h.add("x")                                               # read-only input was copied
    assert not ro.any()


def test_hll_count_extremes_finite():
    full = S.HyperLogLog(registers=np.full(1024, 55, dtype=np.uint8))
    c = full.count()
    assert math.isfinite(c) and c > 1e18
    one_zero = np.full(1024, 1, dtype=np.uint8)
    one_zero[0] = 0
    assert math.isfinite(S.HyperLogLog(registers=one_zero).count())


def test_hll_equality_copy_and_pickle():
    h = hll_of(f"p{i}" for i in range(300))
    assert h == S.HyperLogLog.from_bytes(h.to_bytes())
    assert h != S.HyperLogLog()
    assert (h == "not an hll") is False
    for clone in (copy.deepcopy(h), pickle.loads(pickle.dumps(h))):
        assert clone == h and clone.registers is not h.registers
    c = copy.copy(h)
    assert c == h


def test_hll_add_many_speed():
    t = time.perf_counter()
    hll_of(range(100_000))
    assert time.perf_counter() - t < 1.0                     # measured ~0.07 s


# ------------------------------------------------------ spec v2.1 SetSketch
def test_set_sketch_exact_small_and_hll_beyond_256():
    from app.engines.behavior.lib import sketch as SK
    sk = SK.set_sketch("l4.peer_ids", [f"10.0.0.{i}" for i in range(200)] + ["10.0.0.1"])
    assert sk["n"] == 200 and "h" in sk and SK.set_count(sk) == 200.0
    big = SK.set_sketch("l4.peer_ids", [f"h{i}" for i in range(300)])
    assert big["n"] == 300 and "hll" in big and len(big["hll"]) == 1024
    assert abs(SK.set_count(big) - 300) / 300 < 0.13
    # 443 and '443' are one key
    assert SK.set_sketch("ports", [443, "443"])["n"] == 1


def test_set_union_exact_up_to_4096_then_hll():
    from app.engines.behavior.lib import sketch as SK
    parts = [SK.set_sketch("act.template_ids", [f"t{j}" for j in range(i * 100, i * 100 + 150)])
             for i in range(10)]
    u = SK.set_union(parts)
    assert "h" in u and SK.set_count(u) == 1050.0            # exact union with overlaps
    many = [SK.set_sketch("ns", [f"k{j}" for j in range(i * 250, i * 250 + 250)]) for i in range(20)]
    u2 = SK.set_union(many)                                   # 5000 > 4096 distinct -> HLL
    assert "hll" in u2 and abs(SK.set_count(u2) - 5000) / 5000 < 0.13
    # the union of the sketches equals the sketch of the union (exact regime)
    keys_a, keys_b = [f"x{i}" for i in range(0, 120)], [f"x{i}" for i in range(80, 220)]
    ua = SK.set_union([SK.set_sketch("n", keys_a), SK.set_sketch("n", keys_b)])
    direct = SK.set_sketch("n", keys_a + keys_b)
    assert ua["n"] == direct["n"] == 220 and np.array_equal(ua["h"], direct["h"])
    # associativity and HLL members: register max is lossless
    h1 = SK.set_sketch("n", [f"y{i}" for i in range(400)])
    h2 = SK.set_sketch("n", [f"y{i}" for i in range(200, 700)])
    s3 = SK.set_sketch("n", [f"y{i}" for i in range(650, 750)])
    left = SK.set_union([SK.set_union([h1, h2]), s3])
    right = SK.set_union([h1, SK.set_union([h2, s3])])
    assert left["hll"] == right["hll"]
    full = SK.set_sketch("n", [f"y{i}" for i in range(750)])
    assert left["hll"] == full["hll"]
    empty = SK.set_union([])
    assert empty["n"] == 0 and len(empty["h"]) == 0 and SK.set_count(None) == 0.0
