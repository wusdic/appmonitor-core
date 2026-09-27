"""Categorical sketches: signed hashing (feature.sketch) and HyperLogLog (act.objs, B13).

STATUS: implemented. Constants, signatures and maths are frozen
(docs/lib3/helpers_api.md).

feature.sketch[80] turns five categorical namespaces into a fixed-size,
cadence-independent vector for identity (B15/B16) and correlation-aware
scoring without a growing vocabulary. Each namespace owns one block of 16
signed-hash buckets:

    v[ns*16 + b] = sum over tokens tok with h(tok) = b of  s(tok) * sqrt(c_tok / C_ns)
    C_ns = sum of c_tok over the namespace ('__other__' excluded from both)

then each non-empty block is renormalised to unit L2 norm (collisions with
opposite signs change the norm). If a block cancels to norm < 1e-12 it is
recomputed unsigned (s = +1) before normalising, so a non-empty block
always has norm 1 and never NaN; an empty namespace is all zeros.

Why sqrt(c / C) (the Hellinger embedding): without collisions the block is
the square root of the token distribution, so the dot product of two blocks
is the Bhattacharyya coefficient and |u - v|^2 = 2 H^2 (twice the squared
Hellinger distance). It is bounded, cadence invariant (15x the counts give
the same block) and not dominated by one heavy token the way raw frequencies
are. The random signs make collisions unbiased: before the renormalisation,
E[<u, v>] over the hash is still the Bhattacharyya coefficient, because cross
terms of different tokens sharing a bucket cancel in expectation.

Hashing: blake2b(f'{ns}\\x1f{tok}', digest_size=8) -> 64-bit x, read
big-endian (x = int.from_bytes(digest, 'big')); bucket = x % 16,
sign = +1 if (x >> 32) & 1 else -1. Deterministic across processes (never
Python's hash()). Strings are UTF-8 encoded with 'surrogatepass', so a token
carrying a lone surrogate (from a lossy decode upstream) still hashes.

HyperLogLog: act.objs carries 1024 one-byte registers per template when a
tick has more than 256 distinct object ids, and B13 merges them (register-wise
max) into a per-day breadth count. 1 KiB per sketch with ~3.25 % standard
error is the cheapest exact-merge distinct counter; the merge is lossless
(the union sketch equals the sketch of the union).

Implementation notes:
  * Hashes are memoised per namespace in plain dicts keyed by str(tok)
    (<= 16384 tokens per namespace, <= 64 namespaces, cleared when full):
    blake2b is ~1.3 us per token in CPython, which alone would eat B01's
    0.3 ms per-entity budget at 256 act.tokens; a dict hit is ~50 ns. Tokens
    are str()'d first, so 443 and '443' (l4.dport_set keys) hash identically,
    exactly as the f-string in the formula does, and True / 1 cannot alias.
    Per bucket the block is one np.bincount over code = bucket + 16 * (s < 0),
    so the signed and the unsigned block come from the same pass; up to 24
    tokens a pure-Python loop does the same (numpy call overhead dominates
    there). Warm cost for a heavy entity (256 act.tokens + 144 others) is
    ~0.16 ms for the whole sketch_vector.
  * sketch_block is scale-free: counts are divided by their maximum before
    summing, so C_ns cannot overflow even for counts near 1e308 and
    p_tok = c_tok / C_ns is unchanged. The cancellation test (< 1e-12) is on
    the formula-scale block, before renormalisation.
  * sketch_block is lenient about the value shapes the store actually holds:
    a value that is itself a mapping uses its 'n' field (client.stack_set is
    {token: {n, bytes, first_ts, last_ts}}), a non-mapping iterable of tokens
    (a set or list) counts each occurrence once, and None is empty. Values
    that are non-numeric, non-finite or <= 0 are ignored.
  * HyperLogLog.add_many hashes in Python (unavoidable) but does the
    rank / register update vectorised: rho = 55 - bit_length(w) with the
    54-bit remainder w split in two 27-bit halves so float64 frexp gives an
    exact bit length (a direct float64 conversion of w >= 2^53 can round up
    to the next power of two).
  * Registers only ever hold 0..55 (rho <= 64 - p + 1); from_bytes and the
    constructor reject anything else as a corrupt payload rather than let it
    silently bias count().
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np

SKETCH_BLOCK = 16
SKETCH_NAMESPACES: Tuple[str, ...] = ("act.tokens", "client.stack_set", "sni_etld1",
                                      "dns_etld1", "l4.dport_set")
SKETCH_DIM = SKETCH_BLOCK * len(SKETCH_NAMESPACES)   # 80
OTHER_TOKEN = "__other__"

HLL_P = 10
HLL_M = 1 << HLL_P          # 1024 registers = 1024 bytes
HLL_ALPHA = 0.7213 / (1.0 + 1.079 / HLL_M)

_SEP = "\x1f"                       # ASCII unit separator between ns and token
_CANCEL_NORM = 1e-12                # formula-scale norm below which a block is re-hashed unsigned
_CODE_CACHE_MAX = 1 << 14           # tokens memoised per namespace (cleared when full)
_NS_CACHE_MAX = 64                  # namespaces memoised (cleared when exceeded)
_INF = math.inf
_SMALL_BLOCK = 24                   # up to this many tokens the block is built in pure Python

_W_BITS = 64 - HLL_P                # 54 bits left after the register index
_RHO_MAX = _W_BITS + 1              # 55: rank of an all-zero remainder
_W_MASK = (1 << _W_BITS) - 1
_HALF = 27                          # split w into 27-bit halves (exact in float64)
_U_W_MASK = np.uint64(_W_MASK)
_U_HALF_MASK = np.uint64((1 << _HALF) - 1)
_U_W_BITS = np.uint64(_W_BITS)
_U_HALF = np.uint64(_HALF)
_INV_POW2 = np.ldexp(1.0, -np.arange(256))   # 2^-k for every possible uint8 register
_ADD_CHUNK = 1 << 16                # digests buffered per vectorised register update
_SCALAR_ADD_MAX = 8                 # below this, the scalar path beats numpy overhead


# ------------------------------------------------------------------ hashing
def _digest64(s: str) -> int:
    """64-bit blake2b of a string, read big-endian."""
    d = hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=8).digest()
    return int.from_bytes(d, "big")


# ns -> {tok: code}, code = bucket + 16 * (sign < 0). A plain dict per namespace
# (not lru_cache) because the lookup sits in B01's per-token hot loop.
_CODES: Dict[str, Dict[str, int]] = {}


def _ns_cache(ns: str) -> Dict[str, int]:
    cache = _CODES.get(ns)
    if cache is None:
        if len(_CODES) >= _NS_CACHE_MAX:
            _CODES.clear()
        cache = _CODES.setdefault(ns, {})
    return cache


def _code(ns: str, tok: str, cache: Dict[str, int]) -> int:
    code = cache.get(tok)
    if code is None:
        x = _digest64(f"{ns}{_SEP}{tok}")
        code = x % SKETCH_BLOCK + (0 if (x >> 32) & 1 else SKETCH_BLOCK)
        if len(cache) >= _CODE_CACHE_MAX:
            cache.clear()
        cache[tok] = code
    return code


def token_hash(ns: str, tok: str) -> Tuple[int, int]:
    """(bucket in [0, 16), sign in {-1, +1}) for a token in namespace ns."""
    if type(tok) is not str:
        tok = str(tok)
    if type(ns) is not str:
        ns = str(ns)
    code = _code(ns, tok, _ns_cache(ns))
    return (code, 1) if code < SKETCH_BLOCK else (code - SKETCH_BLOCK, -1)


# ------------------------------------------------------------ hashed sketch
def _as_count(c: Any) -> float:
    """A usable count, or 0.0 for anything to ignore (non-numeric, NaN, inf, <= 0)."""
    if isinstance(c, Mapping):
        c = c.get("n")
    if isinstance(c, (str, bytes, bytearray)):   # float('3') would parse: text is not a count
        return 0.0
    try:
        c = float(c)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return c if (c > 0.0 and c < _INF) else 0.0


def _collect(counts: Any, ns: str) -> Tuple[List[int], List[float]]:
    """(codes, counts) of the valid tokens: '__other__' and ignorable counts dropped.

    One pass that also looks the hash codes up, since this loop is the whole
    per-token cost of B01's sketch."""
    if counts is None:
        return [], []
    if isinstance(counts, Mapping):
        items: Iterable[Tuple[Any, Any]] = counts.items()
    elif isinstance(counts, (str, bytes, bytearray)):
        raise TypeError("sketch_block: counts must be a mapping {token: count} "
                        "or an iterable of tokens, not a single string")
    else:
        items = Counter(counts).items()
    cache = _ns_cache(ns)
    get = cache.get
    codes: List[int] = []
    cs: List[float] = []
    for tok, c in items:
        tc = type(c)
        if tc is float:                       # fast path: the common store value types
            if not (c > 0.0 and c < _INF):    # also rejects NaN
                continue
        elif tc is int:
            if c <= 0:
                continue
            try:
                c = float(c)
            except OverflowError:             # > 1e308 is as unusable as inf
                continue
        else:
            c = _as_count(c)
            if c <= 0.0:
                continue
        if type(tok) is not str:
            tok = str(tok)
        if tok == OTHER_TOKEN:
            continue
        code = get(tok)
        if code is None:
            code = _code(ns, tok, cache)
        codes.append(code)
        cs.append(c)
    return codes, cs


def sketch_block(counts: Mapping[str, float], ns: str) -> np.ndarray:
    """float64[16] unit-norm block for one namespace (all zeros if empty /
    all counts <= 0). Non-finite or negative counts are ignored. O(#tokens)."""
    if type(ns) is not str:
        ns = str(ns)
    codes, cs = _collect(counts, ns)
    n = len(cs)
    if n == 0:
        return np.zeros(SKETCH_BLOCK, dtype=np.float64)

    # p = c / C computed as (c / cmax) / sum(c / cmax): identical maths, no overflow.
    # w[code] accumulates sqrt(p): codes < 16 carry s = +1, codes >= 16 carry s = -1.
    if n <= _SMALL_BLOCK:
        # pure Python: a dozen numpy calls cost more than the loop at this size
        cmax = max(cs)
        scaled = [c / cmax for c in cs]
        total = math.fsum(scaled)             # >= 1: the max token contributes 1
        acc = [0.0] * (2 * SKETCH_BLOCK)
        for code, q in zip(codes, scaled):
            acc[code] += math.sqrt(q / total)
        w = np.array(acc, dtype=np.float64)
    else:
        c = np.array(cs, dtype=np.float64)
        c /= c.max()
        a = np.sqrt(c / c.sum())
        w = np.bincount(np.array(codes, dtype=np.intp), weights=a, minlength=2 * SKETCH_BLOCK)
    v = w[:SKETCH_BLOCK] - w[SKETCH_BLOCK:]
    norm = math.sqrt(float(np.dot(v, v)))
    if not norm >= _CANCEL_NORM:
        # opposite-sign collisions cancelled the block: fall back to s = +1, whose
        # norm is >= sqrt(sum p) = 1 because every term is non-negative
        v = w[:SKETCH_BLOCK] + w[SKETCH_BLOCK:]
        norm = math.sqrt(float(np.dot(v, v)))
    return v / norm


def sketch_vector(ns_counts: Mapping[str, Mapping[str, float]]) -> np.ndarray:
    """float32[80]: blocks in SKETCH_NAMESPACES order; a missing namespace is zeros.

    Keys outside SKETCH_NAMESPACES are ignored. The float32 cast keeps each
    non-empty block at unit norm only to float32 precision (|norm - 1| ~ 1e-7);
    the 1e-9 guarantee is sketch_block's (float64)."""
    out = np.zeros(SKETCH_DIM, dtype=np.float32)
    if not ns_counts:
        return out
    for i, ns in enumerate(SKETCH_NAMESPACES):
        counts = ns_counts.get(ns)
        if counts:
            out[i * SKETCH_BLOCK:(i + 1) * SKETCH_BLOCK] = sketch_block(counts, ns)
    return out


# -------------------------------------------------------------- HyperLogLog
def _bit_length_u54(w: np.ndarray) -> np.ndarray:
    """Exact bit_length of uint64 values < 2^54 (0 for 0), vectorised."""
    hi = (w >> _U_HALF).astype(np.float64)
    lo = (w & _U_HALF_MASK).astype(np.float64)
    # frexp(x)[1] = bit_length(x) for integers 0 <= x < 2^53 (frexp(0) = (0, 0))
    return np.where(hi > 0, _HALF + np.frexp(hi)[1], np.frexp(lo)[1])


@dataclass(slots=True)
class HyperLogLog:
    """HLL with p = 10 (1024 uint8 registers), standard error ~3.25%.

    add: x = 64-bit blake2b of str(item); idx = x >> (64 - p);
         rho = leading zeros of the remaining (64 - p) bits + 1; reg[idx] = max.
    count: E = alpha m^2 / sum 2^-reg; if E <= 2.5 m and zeros V > 0 use
         linear counting m ln(m / V). (No large-range correction: 64-bit hash.)
    """
    registers: np.ndarray = field(default_factory=lambda: np.zeros(HLL_M, dtype=np.uint8))

    def __post_init__(self) -> None:
        reg = np.asarray(self.registers)
        if reg.shape != (HLL_M,):
            raise ValueError(f"HyperLogLog: expected {HLL_M} registers, got shape {reg.shape}")
        if reg.dtype != np.uint8:
            if reg.dtype.kind not in "iu":
                raise ValueError(f"HyperLogLog: integer registers required, got {reg.dtype}")
            if reg.min() < 0:
                raise ValueError("HyperLogLog: negative register")
            if reg.max() > _RHO_MAX:
                raise ValueError(f"HyperLogLog: register > {_RHO_MAX} (corrupt payload)")
            reg = reg.astype(np.uint8)
        elif reg.max() > _RHO_MAX:
            raise ValueError(f"HyperLogLog: register > {_RHO_MAX} (corrupt payload)")
        if not (reg.flags.writeable and reg.flags.c_contiguous):
            reg = reg.copy()
        self.registers = reg

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HyperLogLog):
            return NotImplemented
        return bool(np.array_equal(self.registers, other.registers))

    def add(self, item: object) -> None:
        """Add one item (hashed as str(item), so 5 and '5' are the same object)."""
        x = _digest64(item if type(item) is str else str(item))
        idx = x >> _W_BITS
        rho = _RHO_MAX - (x & _W_MASK).bit_length()
        if rho > self.registers[idx]:
            self.registers[idx] = rho

    def add_many(self, items: Iterable[object]) -> None:
        """Same registers as add() per item; the register update is vectorised."""
        blake2b = hashlib.blake2b
        buf: List[bytes] = []
        for it in items:
            s = it if type(it) is str else str(it)
            buf.append(blake2b(s.encode("utf-8", "surrogatepass"), digest_size=8).digest())
            if len(buf) >= _ADD_CHUNK:
                self._add_digests(buf)
                buf = []
        if buf:
            self._add_digests(buf)

    def _add_digests(self, digests: List[bytes]) -> None:
        if len(digests) <= _SCALAR_ADD_MAX:
            reg = self.registers
            for d in digests:
                x = int.from_bytes(d, "big")
                idx = x >> _W_BITS
                rho = _RHO_MAX - (x & _W_MASK).bit_length()
                if rho > reg[idx]:
                    reg[idx] = rho
            return
        x = np.frombuffer(b"".join(digests), dtype=">u8").astype(np.uint64)
        idx = (x >> _U_W_BITS).astype(np.intp)
        rho = (_RHO_MAX - _bit_length_u54(x & _U_W_MASK)).astype(np.uint8)
        np.maximum.at(self.registers, idx, rho)

    def count(self) -> float:
        """Distinct-count estimate (0.0 when empty); never NaN."""
        reg = self.registers
        z = float(_INV_POW2[reg].sum())
        m = float(HLL_M)
        e = HLL_ALPHA * m * m / z
        if e <= 2.5 * m:
            v = int(np.count_nonzero(reg == 0))
            if v > 0:
                return m * math.log(m / v)
        return e

    def merge(self, other: "HyperLogLog") -> "HyperLogLog":
        """In-place register-wise max; returns self."""
        if not isinstance(other, HyperLogLog):
            raise TypeError(f"HyperLogLog.merge: expected HyperLogLog, got {type(other).__name__}")
        np.maximum(self.registers, other.registers, out=self.registers)
        return self

    def to_bytes(self) -> bytes:
        """Exactly 1024 bytes (act.objs 'hll' payload)."""
        reg = self.registers
        if reg.shape != (HLL_M,) or reg.dtype != np.uint8:
            raise ValueError("HyperLogLog: registers were replaced with a non-uint8[1024] array")
        return reg.tobytes()

    @classmethod
    def from_bytes(cls, b: bytes) -> "HyperLogLog":
        """Inverse of to_bytes (copies, so the sketch never aliases the payload).
        ValueError unless exactly 1024 bytes with every register <= 55."""
        if not isinstance(b, (bytes, bytearray, memoryview)):
            raise TypeError(f"HyperLogLog.from_bytes: bytes-like required, got {type(b).__name__}")
        raw = bytes(b)
        if len(raw) != HLL_M:
            raise ValueError(f"HyperLogLog.from_bytes: expected {HLL_M} bytes, got {len(raw)}")
        return cls(registers=np.frombuffer(raw, dtype=np.uint8).copy())
