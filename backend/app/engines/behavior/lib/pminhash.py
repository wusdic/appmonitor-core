"""Behavioural signatures and their similarity for who-discovery
(docs/lib3/progressive.md §6.15 items 1-2, P11).

STATUS: implemented (W-P5, P11 maths). Pure data structures and functions;
no store access. P11 (`behavior.who_groups`) owns the state.

Signatures (§6.15 item 1)
    SigStore(s_max, k)  per source key (an IP, or a /24 for systems in prefix
                        mode) a Space-Saving table of k = 24 items with
                        forward-decayed mass at H_m (one org-wide landmark, so a
                        read is one multiply) and the source's evidence units.
                        Items are interned strings '<tree key>|<action>' with
                        integer ids that are never reused. The store is an LRU
                        by last activity capped at S_max = 50 000 sources.
                        Memory per source: one float32[3, k] array (id, mass,
                        Space-Saving error) plus five scalars, about 0.5 KB, so
                        S_max sources cost about 25 MB whatever the traffic.
Similarity (§6.15 item 2)
    icws(...)           weighted MinHash by Improved Consistent Weighted Sampling
                        (Ioffe 2010), k = 64 samples of the source's weight
                        vector w(a) = log2(1 + decayed mass(a)) / sum (the action
                        mix); two samples agree with probability equal to the
                        weighted (Ruzicka) Jaccard of the two weight vectors.
    lsh_pairs(...)      banding b = 16 x r = 4; a bucket larger than `cap` links
                        each member to its `cap_links` successors in a fixed order
                        (a ring) instead of all pairs, so the candidate count is
                        O(n b cap_links), never O(n^2) (members of one bucket share
                        a whole band and are near-duplicates; the community step
                        re-merges a ring that modularity would cut, see plouvain).
    knn_graph(...)      each source keeps its best `knn` candidates with
                        J >= j_min (a row standing for several identical
                        signatures fills that many places); the graph is the
                        MUTUAL kNN graph weighted by J.
    components(...)     connected components (P11 runs Louvain per component).

Deviation (documented, measured in tests/engines/test_p11_who_groups.py):
    the weight of an item is log2(1 + mass) normalised over the source's items,
    not the mass itself: with raw mass the high-volume shared actions (document
    views every department performs dozens of times a day) dominate the
    weighted Jaccard and make departments look alike, and volume differences
    between sources doing the same thing (a DHCP pool's developers pushing more
    or less code) cut one population into volume bands; the normalised log
    keeps "which actions, in which mix" and drops "how many".
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import psketch as PS

K_SIG = 24                  # items per source signature (§6.15 item 1)
K_MH = 64                   # MinHash samples
BANDS = 16
ROWS = 4
KNN = 10                    # candidates kept per source
J_MIN = 0.2
BUCKET_CAP = 64             # all-pairs inside an LSH bucket up to this size, a ring beyond
CAP_LINKS = 8
S_MAX = 50_000
ITEM_MAX = 65_536           # live interned items (ids are never reused)
RESCALE_EXP = 60.0
ID_LIMIT = float(1 << 24)   # float32 represents every integer below 2**24 exactly
_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_C1 = np.uint64(0x9E3779B97F4A7C15)
_C2 = np.uint64(0xBF58476D1CE4E5B9)
_C3 = np.uint64(0x94D049BB133111EB)


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=8).digest(),
                          "little")


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser on a uint64 array (wrapping arithmetic)."""
    with np.errstate(over="ignore"):
        x = x.astype(np.uint64, copy=True)
        x ^= x >> np.uint64(30)
        x *= _C2
        x ^= x >> np.uint64(27)
        x *= _C3
        x ^= x >> np.uint64(31)
    return x


# ================================================================ items
class ItemDict:
    """Interned items '<tree key>|<action>' -> int id (never reused) with a
    64-bit content hash per id (MinHash is keyed by the hash, so two runs that
    number items differently produce the same samples). Capped: the least
    recently used ids are forgotten beyond `cap` (a forgotten id still in an old
    signature keeps a deterministic hash derived from the id)."""

    __slots__ = ("k2i", "i2h", "i2k", "next_id", "cap")

    def __init__(self, cap: int = ITEM_MAX) -> None:
        self.k2i: PS.LRU = PS.LRU(max(16, int(cap)))
        self.i2h: Dict[int, int] = {}
        self.i2k: Dict[int, str] = {}
        self.next_id = 0
        self.cap = int(cap)

    def id_of(self, key: str, create: bool = True) -> Optional[int]:
        i = self.k2i.get(key)
        if i is not None or not create:
            return i
        if self.next_id >= ID_LIMIT:              # pragma: no cover - 16.7 M distinct actions
            raise OverflowError("ItemDict: id space exhausted")
        i = self.next_id
        self.next_id += 1
        ev = self.k2i.put(key, i)
        self.i2h[i] = h64(key)
        self.i2k[i] = key
        if ev is not None:
            self.i2h.pop(ev[1], None)
            self.i2k.pop(ev[1], None)
        return i

    def hash_of(self, i: int) -> int:
        h = self.i2h.get(int(i))
        if h is None:
            h = int(_mix(np.asarray([int(i) + 1], dtype=np.uint64))[0])
        return h

    def key_of(self, i: int) -> Optional[str]:
        return self.i2k.get(int(i))

    def __len__(self) -> int:
        return len(self.k2i)

    def nbytes(self) -> int:
        return 160 * len(self.k2i) + 64


# ============================================================ signatures
class Sig:
    """One source's signature: a[0] item ids, a[1] forward-decayed mass
    (relative to the store's landmark), a[2] Space-Saving error; n used slots;
    ev evidence units (undecayed total); first / last activity; day of the
    last evidence (for the `days` count of distinct active local days)."""

    __slots__ = ("a", "n", "ev", "first", "last", "days", "lday")

    def __init__(self, k: int, t: float) -> None:
        self.a = np.zeros((3, k), dtype=np.float32)
        self.n = 0
        self.ev = 0.0
        self.first = float(t)
        self.last = float(t)
        self.days = 0
        self.lday = -1


class SigStore:
    """Per-source Space-Saving signatures (see module docstring)."""

    def __init__(self, s_max: int = S_MAX, k: int = K_SIG, hl: float = PS.H_M,
                 item_cap: int = ITEM_MAX) -> None:
        self.k = int(k)
        self.hl = float(hl)
        self.L: Optional[float] = None
        self.sigs: PS.LRU = PS.LRU(max(1, int(s_max)))
        self.items = ItemDict(item_cap)
        self.dropped = 0

    # ------------------------------------------------------------ decay
    def _f(self, t: float) -> float:
        """Forward-decay factor 2^((t - L)/H) of an arrival at t."""
        if self.L is None:
            self.L = float(t)
        e = (float(t) - self.L) / self.hl
        if e > RESCALE_EXP:
            self._rescale(float(t))
            e = 0.0
        return 2.0 ** e

    def _rescale(self, t: float) -> None:
        g = np.float32(2.0 ** (-(t - self.L) / self.hl))
        for _, sg in self.sigs._d.items():
            sg.a[1:, :sg.n] *= g
        self.L = t

    def read_factor(self, t: float) -> float:
        return 0.0 if self.L is None else 2.0 ** (-(float(t) - self.L) / self.hl)

    # ---------------------------------------------------------- updates
    def add(self, src: Hashable, item: str, t: float, mass: float, ev: float,
            day: Optional[int] = None) -> Sig:
        """Count one learned event of `src` on `item` (mass, evidence units)."""
        sg = self.sigs.get(src)
        if sg is None:
            sg = Sig(self.k, t)
            ev_ = self.sigs.put(src, sg)
            if ev_ is not None:
                self.dropped += 1
        i = float(self.items.id_of(item))
        w = float(mass) * self._f(t)
        a = sg.a
        n = sg.n
        hit = np.flatnonzero(a[0, :n] == i) if n else ()
        if len(hit):
            a[1, hit[0]] += w
        elif n < self.k:
            a[0, n] = i
            a[1, n] = w
            a[2, n] = 0.0
            sg.n = n + 1
        else:                                          # Space-Saving replacement
            j = int(np.argmin(a[1]))
            m = a[1, j]
            a[0, j] = i
            a[2, j] = m
            a[1, j] = m + w
        sg.ev += float(ev)
        sg.last = max(sg.last, float(t))
        if day is not None and int(day) != sg.lday:
            sg.days += 1
            sg.lday = int(day)
        return sg

    def drop(self, src: Hashable) -> None:
        self.sigs.pop(src, None)

    def prune(self, t: float, min_ev: float, min_age_s: float, idle_s: float) -> int:
        """Drop sources with < min_ev evidence older than min_age_s, and sources
        idle for idle_s (§6.15 item 1). O(sources)."""
        gone = [k for k, sg in self.sigs._d.items()
                if (sg.ev < min_ev and t - sg.first >= min_age_s) or t - sg.last >= idle_s]
        for k in gone:
            self.sigs.pop(k, None)
        return len(gone)

    def set_cap(self, cap: int) -> None:
        self.sigs.set_cap(max(1, int(cap)))

    # ------------------------------------------------------------ reads
    def get(self, src: Hashable) -> Optional[Sig]:
        return self.sigs.peek(src)

    def __contains__(self, src: Hashable) -> bool:
        return self.sigs.peek(src) is not None

    def __len__(self) -> int:
        return len(self.sigs)

    def keys(self) -> List[Hashable]:
        return list(self.sigs._d.keys())

    def weights(self, src: Hashable, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """(item ids int64[n], decayed mass float64[n]) of a source."""
        sg = self.sigs.peek(src)
        if sg is None or sg.n == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        f = self.read_factor(t)
        return sg.a[0, :sg.n].astype(np.int64), sg.a[1, :sg.n].astype(np.float64) * f

    def profile(self, src: Hashable, t: float) -> Dict[int, float]:
        ids, w = self.weights(src, t)
        return {int(i): float(x) for i, x in zip(ids, w) if x > 0}

    def nbytes(self) -> int:
        per = self.k * 3 * 4 + 112 + 120 + 90      # array + object + LRU entry
        return int(len(self.sigs) * per + self.items.nbytes() + 128)


# =============================================================== MinHash
class Randoms:
    """ICWS random variables per item hash: r, c ~ Gamma(2, 1), beta ~ U(0, 1)
    for each of the k samples, drawn from a generator seeded by the hash (the
    same item always gets the same variables). Cached for one clustering run."""

    def __init__(self, k: int = K_MH, seed: int = 0) -> None:
        self.k = int(k)
        self.seed = int(seed)
        self._c: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def get(self, h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        v = self._c.get(h)
        if v is None:
            g = np.random.Generator(np.random.PCG64([int(h) & 0xFFFFFFFFFFFFFFFF, self.seed]))
            v = self._c[h] = (g.gamma(2.0, 1.0, self.k), g.gamma(2.0, 1.0, self.k),
                              g.uniform(0.0, 1.0, self.k))
        return v

    def stack(self, hs: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rs, cs, bs = zip(*(self.get(int(h)) for h in hs))
        return np.vstack(rs), np.vstack(cs), np.vstack(bs)


def transform(w: np.ndarray) -> np.ndarray:
    """Item weight used for similarity: log2(1 + decayed mass), normalised to
    sum 1 (the source's action MIX: two sources doing the same actions in the
    same proportions are alike whatever their volumes)."""
    x = np.log2(1.0 + np.maximum(np.asarray(w, dtype=np.float64), 0.0))
    s = float(x.sum())
    return x / s if s > 0 else x


def icws(hashes: Sequence[int], w: np.ndarray, rnd: Randoms) -> np.ndarray:
    """Improved Consistent Weighted Sampling (Ioffe 2010): k uint64 samples of
    the weighted set {hashes[j]: w[j]}; P(sample_i(A) == sample_i(B)) =
    sum_j min(A_j, B_j) / sum_j max(A_j, B_j). Empty input -> zeros."""
    w = np.asarray(w, dtype=np.float64)
    ok = w > 0
    if not ok.any():
        return np.zeros(rnd.k, dtype=np.uint64)
    hs = np.asarray([int(h) for h, o in zip(hashes, ok) if o], dtype=np.uint64)
    lw = np.log(w[ok])[:, None]
    R, C, B = rnd.stack(hs.tolist())
    t = np.floor(lw / R + B)
    ln_y = R * (t - B)
    ln_a = np.log(C) - ln_y - R
    j = np.argmin(ln_a, axis=0)
    ti = t[j, np.arange(rnd.k)].astype(np.int64).view(np.uint64)
    with np.errstate(over="ignore"):
        s = hs[j] * _C1 + ti * _C2
    return _mix(s)


def jaccard_est(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a == b)) if a.size else 0.0


def weighted_jaccard(p: Dict[Any, float], q: Dict[Any, float]) -> float:
    """Exact Ruzicka similarity of two sparse non-negative vectors."""
    keys = set(p) | set(q)
    num = sum(min(p.get(k, 0.0), q.get(k, 0.0)) for k in keys)
    den = sum(max(p.get(k, 0.0), q.get(k, 0.0)) for k in keys)
    return num / den if den > 0 else 0.0


# =================================================================== LSH
def band_keys(sigs: np.ndarray, bands: int = BANDS, rows: int = ROWS) -> np.ndarray:
    """uint64[n, bands]: one hash per band of each signature row."""
    n = sigs.shape[0]
    out = np.zeros((n, bands), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for b in range(bands):
            acc = np.full(n, np.uint64(b + 1), dtype=np.uint64)
            for r in range(rows):
                acc = _mix(acc * _C1 + sigs[:, b * rows + r])
            out[:, b] = acc
    return out


def lsh_pairs(sigs: np.ndarray, bands: int = BANDS, rows: int = ROWS, cap: int = BUCKET_CAP,
              cap_links: int = CAP_LINKS) -> Set[Tuple[int, int]]:
    """Candidate pairs (i < j) sharing at least one band. A bucket of more than
    `cap` members links each member to its next `cap_links` members (ring)."""
    n = sigs.shape[0]
    pairs: Set[Tuple[int, int]] = set()
    if n < 2:
        return pairs
    bk = band_keys(sigs, bands, rows)
    for b in range(bands):
        col = bk[:, b]
        order = np.argsort(col, kind="stable")
        sc = col[order]
        cut = np.flatnonzero(np.diff(sc) != 0) + 1
        for grp in np.split(order, cut):
            m = grp.size
            if m < 2:
                continue
            g = np.sort(grp).tolist()
            if m <= cap:
                for x in range(m):
                    gx = g[x]
                    for y in range(x + 1, m):
                        pairs.add((gx, g[y]))
            else:
                for x in range(m):
                    for d in range(1, cap_links + 1):
                        y = g[(x + d) % m]
                        a, c = (g[x], y) if g[x] < y else (y, g[x])
                        pairs.add((a, c))
    return pairs


def knn_graph(sigs: np.ndarray, pairs: Iterable[Tuple[int, int]], knn: int = KNN,
              j_min: float = J_MIN, mutual: bool = True,
              mult: Optional[np.ndarray] = None) -> Dict[Tuple[int, int], float]:
    """Mutual-kNN graph {(i, j): J} (i < j) from candidate pairs. `mult` gives
    the multiplicity of each row (a row standing for several identical
    signatures fills that many of a neighbour list's k places)."""
    best: Dict[int, List[Tuple[float, int]]] = {}
    for i, j in pairs:
        jv = float(np.count_nonzero(sigs[i] == sigs[j])) / sigs.shape[1]
        if jv < j_min:
            continue
        best.setdefault(i, []).append((jv, j))
        best.setdefault(j, []).append((jv, i))
    top: Dict[int, Dict[int, float]] = {}
    for i, lst in best.items():
        lst.sort(key=lambda x: (-x[0], x[1]))
        chosen: Dict[int, float] = {}
        filled = 0
        for jv, j in lst:
            if filled >= knn:
                break
            chosen[j] = jv
            filled += int(mult[j]) if mult is not None else 1
        top[i] = chosen
    edges: Dict[Tuple[int, int], float] = {}
    for i, nb in top.items():
        for j, jv in nb.items():
            if mutual and i not in top.get(j, {}):
                continue
            a, b = (i, j) if i < j else (j, i)
            edges[(a, b)] = jv
    return edges


def components(n: int, edges: Iterable[Tuple[int, int]]) -> List[List[int]]:
    """Connected components (union-find), each sorted, in order of first node."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    out: Dict[int, List[int]] = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    return [out[k] for k in sorted(out)]


def neighbours(top_edges: Dict[Tuple[int, int], float], i: int) -> List[Tuple[int, float]]:
    return [(b if a == i else a, w) for (a, b), w in top_edges.items() if a == i or b == i]
