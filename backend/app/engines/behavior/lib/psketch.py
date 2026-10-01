"""Bounded, forward-decayed, mergeable sketches for the progressive profile core.

STATUS: implemented (W-P0, docs/lib3/progressive.md §6.1). Every structure has a
declared cap and serialises to plain dict / ndarray form (`to_dict` /
`from_dict`); every counting structure is mergeable (`merge`).

Forward decay (Cormode, Shkapenyuk, Srivastava, Xu, ICDE 2009). An item that
arrives at time t with weight w is stored as w * 2^((t - L) / H) relative to a
landmark L; the decayed value at read time T is stored * 2^(-(T - L) / H).
Decay is an add at update time and a multiply at read time, sums stay linear
(merge / subtract / scale are exact), and late events (t < L) are fine. When
(T - L) / min(H) exceeds RESCALE_EXP the structure moves its landmark to T and
multiplies every stored weight by 2^(-(T - L_old) / H).

Mass versus evidence (PPC-9). Counting sketches keep *mass* channels (one per
half-life in `mass_hl`, default H_s, H_m, H_l) and *evidence* channels (one per
half-life in `ev_hl`, default H_m, H_l). The last evidence channel is the
*confidence channel* (§6.9.4): `reset_confidence(t)` sets it to the H_m
evidence state (exactly, because both are kept per counter) and
`cap_confidence(t)` caps it there while an attribute is evolving. Mass drives
distributions, evidence drives every test, bound and confidence.

Structures (all O(cap) memory, never O(#items seen)):

    DecayedVector(hl)      forward-decayed sums (mass / evidence / small counters)
    DecayedSpaceSaving(k)  heavy hitters with guaranteed lower bounds and a
                           Good-Turing unseen-mass estimate with eviction
                           correction (§5.5.3)
    HLL(p) / EpochHLL(p)   distinct counts; EpochHLL rotates two epochs (7 d)
    TDigest(delta)         forward-decayed merging t-digest (quantiles / CDF)
    CountMin(w, d)         decayed point counts, overestimate <= e/w * N w.p. 1 - e^-d
    ADWIN(delta)           adaptive window change detector (Bifet & Gavalda 2007)
    PageHinkley(lam, dlt)  two-sided mean-shift detector
    WeightedReservoir(R)   weighted sampling without replacement with decayed keys
    BurstEvidence(cap)     evidence units per (source, node) run (§6.5.4), LRU
    LRU(cap)               bounded mapping with least-recently-used eviction
    hhh_select(...)        hierarchical heavy hitters over nested levels of SS

Hashing is blake2b (never Python's hash()), so results are reproducible across
processes.
"""
from __future__ import annotations

import hashlib
import heapq
import math
from collections import OrderedDict
from typing import (Any, Callable, Dict, Hashable, Iterable, Iterator, List, Optional,
                    Sequence, Tuple)

import numpy as np

# ------------------------------------------------------------------ constants
DAY = 86400.0
H_S = 1.0 * DAY                  # short half-life
H_M = 7.0 * DAY                  # medium half-life (shape)
H_L = 30.0 * DAY                 # long half-life (confidence channel)
HALF_LIVES: Tuple[float, float, float] = (H_S, H_M, H_L)
CH_S, CH_M, CH_L = 0, 1, 2       # mass channel indices for HALF_LIVES
EV_HALF_LIVES: Tuple[float, float] = (H_M, H_L)
EV_M, EV_L = 0, 1                # evidence channel indices; EV_L = confidence channel
RESCALE_EXP = 60.0               # rescale when (t - L) / min(H) exceeds this
N1_EVIDENCE = 1.5                # "seen about once" threshold for Good-Turing N1

_LN2 = math.log(2.0)


def _as_hl(hl: Sequence[float]) -> np.ndarray:
    a = np.asarray([float(h) for h in hl], dtype=np.float64)
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError(f"half-lives must be positive finite floats, got {hl!r}")
    return a


def decay_factor(dt: float, h: float) -> float:
    """2^(-dt / h): the multiplier that ages a value by dt seconds at half-life h."""
    return 2.0 ** (-float(dt) / float(h))


class _Landmark:
    """Shared forward-decay bookkeeping for a structure (landmark L plus the
    half-lives of every channel family it holds)."""

    __slots__ = ("L", "_hmin")

    def __init__(self, L: Optional[float], hmin: float) -> None:
        self.L = None if L is None else float(L)
        self._hmin = float(hmin)

    def ensure(self, t: float) -> Optional[float]:
        """Set the landmark on first use; return the old landmark if the
        structure must rescale before accepting an item at t, else None."""
        if self.L is None:
            self.L = float(t)
            return None
        if (t - self.L) / self._hmin > RESCALE_EXP:
            return self.L
        return None


def _grow(t: float, L: float, hl: np.ndarray) -> np.ndarray:
    return np.exp2((t - L) / hl)


def _shrink(t: float, L: float, hl: np.ndarray) -> np.ndarray:
    return np.exp2(-(t - L) / hl)


# ========================================================= decayed vector
class DecayedVector:
    """A small vector of forward-decayed sums, one half-life per entry (e.g.
    mass at (H_s, H_m, H_l), or several fields at one half-life)."""

    __slots__ = ("hl", "_lm", "v", "_hs")

    def __init__(self, hl: Sequence[float]) -> None:
        self.hl = _as_hl(hl)
        self._hs = tuple(float(h) for h in self.hl)
        self._lm = _Landmark(None, float(self.hl.min()))
        self.v = np.zeros(self.hl.size)

    def add(self, t: float, w: Any = 1.0) -> None:
        """Add w (scalar, or one value per entry) at time t."""
        t = float(t)
        old = self._lm.ensure(t)
        if old is not None:
            self.v *= _shrink(t, old, self.hl)
            self._lm.L = t
        n = self.v.size
        if n <= 8 and isinstance(w, (int, float)):
            d = t - self._lm.L
            v = self.v
            w = float(w)
            for i, h in enumerate(self._hs):
                v[i] += w * 2.0 ** (d / h)
            return
        if n <= 16 and isinstance(w, (list, tuple)) and len(w) == n:
            # per-entry Python loop: np.exp2 + asarray cost ~3x more at these sizes
            # (P04 NumSummary moments, 9 entries, once per learned numeric target)
            d = t - self._lm.L
            v = self.v
            for i, h in enumerate(self._hs):
                v[i] += float(w[i]) * 2.0 ** (d / h)
            return
        self.v += np.exp2((t - self._lm.L) / self.hl) * np.asarray(w, dtype=np.float64)

    def read(self, t: Optional[float] = None) -> np.ndarray:
        """Decayed values at t (landmark units when t is None or empty)."""
        if self._lm.L is None or t is None:
            return self.v.copy()
        return self.v * _shrink(float(t), self._lm.L, self.hl)

    def get(self, i: int, t: Optional[float] = None) -> float:
        return float(self.read(t)[i])

    def scale(self, f: float) -> None:
        self.v *= float(f)

    def set_entry(self, i: int, value: float, t: float) -> None:
        """Set entry i so that its decayed value at t equals value."""
        if self._lm.L is None:
            self._lm.L = float(t)
        self.v[i] = float(value) * 2.0 ** ((float(t) - self._lm.L) / self.hl[i])

    def merge(self, other: "DecayedVector") -> "DecayedVector":
        if other._lm.L is None:
            return self
        if self._lm.L is None:
            self._lm.L = other._lm.L
        L = max(self._lm.L, other._lm.L)
        self.v = self.v * _shrink(L, self._lm.L, self.hl) + other.v * _shrink(L, other._lm.L, self.hl)
        self._lm.L = L
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {"hl": self.hl.tolist(), "L": self._lm.L, "v": self.v.copy()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DecayedVector":
        x = cls(list(d["hl"]))
        x._lm.L = d.get("L")
        x.v = np.asarray(d["v"], dtype=np.float64).copy()
        return x

    def nbytes(self) -> int:
        return int(self.v.nbytes + self.hl.nbytes + 64)


# ============================================================ Space-Saving
class DecayedSpaceSaving:
    """Space-Saving heavy hitters (Metwally, Agrawal & El Abbadi 2005) with
    forward decay, several mass channels and several evidence channels.

    Guarantees (on the `primary` mass channel, default H_m; exact arithmetic):
      * count(x) >= true decayed mass of x (for tracked x)
      * count(x) - err(x) <= true decayed mass of x, on every mass channel
      * every item whose decayed mass exceeds total / k is tracked
      * max err <= total / k
    Evidence is not inherited on replacement: a newly tracked item's evidence
    is its own, and the evidence of the arrival that evicted an item is added
    to the eviction mass E used by `unseen` (§5.5.3):
        U = (N1 + E + 0.5) / (N + 1)
    with N = total evidence and N1 = tracked items whose evidence < 1.5, both
    on the confidence channel. Under churn E grows, so U over-estimates the
    unseen mass (the conservative direction for closedness).
    Counters are plain Python lists (numpy call overhead dominates at these
    sizes: measured 8.5 -> ~3 us per add); reads return numpy where useful.
    Memory: k x (2 len(mass_hl) + len(ev_hl)) floats plus the keys.
    """

    __slots__ = ("k", "mass_hl", "ev_hl", "primary", "_lm", "_idx", "_keys", "_m", "_err",
                 "_e", "_tot_m", "_tot_e", "_evict_e", "_mh", "_eh")

    def __init__(self, k: int, mass_hl: Sequence[float] = HALF_LIVES,
                 ev_hl: Sequence[float] = EV_HALF_LIVES, primary: int = CH_M) -> None:
        if int(k) < 1:
            raise ValueError("DecayedSpaceSaving: k must be >= 1")
        self.k = int(k)
        self.mass_hl = _as_hl(mass_hl)
        self.ev_hl = _as_hl(ev_hl) if len(ev_hl) else np.zeros(0)
        if not 0 <= int(primary) < self.mass_hl.size:
            raise ValueError("DecayedSpaceSaving: primary channel out of range")
        self.primary = int(primary)
        self._mh = tuple(float(h) for h in self.mass_hl)
        self._eh = tuple(float(h) for h in self.ev_hl)
        hmin = float(min(min(self._mh), min(self._eh) if self._eh else math.inf))
        self._lm = _Landmark(None, hmin)
        self._idx: Dict[Hashable, int] = {}
        self._keys: List[Hashable] = []
        self._m: List[List[float]] = []
        self._err: List[List[float]] = []
        self._e: List[List[float]] = []
        self._tot_m = [0.0] * len(self._mh)
        self._tot_e = [0.0] * len(self._eh)
        self._evict_e = [0.0] * len(self._eh)

    # ------------------------------------------------------------- helpers
    def _fm(self, t: Optional[float]) -> List[float]:
        if self._lm.L is None or t is None:
            return [1.0] * len(self._mh)
        d = float(t) - self._lm.L
        return [2.0 ** (-d / h) for h in self._mh]

    def _fe(self, t: Optional[float]) -> List[float]:
        if self._lm.L is None or t is None:
            return [1.0] * len(self._eh)
        d = float(t) - self._lm.L
        return [2.0 ** (-d / h) for h in self._eh]

    def _scale_all(self, fm: Sequence[float], fe: Sequence[float]) -> None:
        for rows in (self._m, self._err):
            for r in rows:
                for c, f in enumerate(fm):
                    r[c] *= f
        for c, f in enumerate(fm):
            self._tot_m[c] *= f
        for r in self._e:
            for c, f in enumerate(fe):
                r[c] *= f
        for c, f in enumerate(fe):
            self._tot_e[c] *= f
            self._evict_e[c] *= f

    def _rescale_to(self, L: float) -> None:
        if self._lm.L is None:
            self._lm.L = float(L)
            return
        if L == self._lm.L:
            return
        self._scale_all(self._fm(L), self._fe(L))
        self._lm.L = float(L)

    # ------------------------------------------------------------- updates
    def add(self, key: Hashable, t: float, w: float = 1.0, ev: float = 0.0) -> None:
        """Add mass w and evidence ev of `key` at time t."""
        w = float(w)
        ev = float(ev)
        if not (0.0 <= w < math.inf and 0.0 <= ev < math.inf) or (w == 0.0 and ev == 0.0):
            return
        t = float(t)
        L = self._lm.L
        if L is None:
            self._lm.L = L = t
        elif (t - L) / self._lm._hmin > RESCALE_EXP:
            self._rescale_to(t)
            L = t
        dlt = t - L
        gm = [w * 2.0 ** (dlt / h) for h in self._mh]
        ge = [ev * 2.0 ** (dlt / h) for h in self._eh]
        tm = self._tot_m
        for c in range(len(gm)):
            tm[c] += gm[c]
        te = self._tot_e
        for c in range(len(ge)):
            te[c] += ge[c]
        i = self._idx.get(key)
        if i is not None:
            row = self._m[i]
            for c in range(len(gm)):
                row[c] += gm[c]
            erow = self._e[i]
            for c in range(len(ge)):
                erow[c] += ge[c]
            return
        if len(self._keys) < self.k:
            self._idx[key] = len(self._keys)
            self._keys.append(key)
            self._m.append(gm)
            self._err.append([0.0] * len(gm))
            self._e.append(ge)
            return
        p = self.primary
        m = self._m
        i = min(range(len(m)), key=lambda j: m[j][p])
        del self._idx[self._keys[i]]
        self._keys[i] = key
        self._idx[key] = i
        old = m[i]
        self._err[i] = list(old)
        m[i] = [old[c] + gm[c] for c in range(len(gm))]
        self._e[i] = ge
        ee = self._evict_e
        for c in range(len(ge)):
            ee[c] += ge[c]

    def add_many(self, keys: Iterable[Hashable], t: Any, w: Any = 1.0, ev: Any = 0.0) -> None:
        """Vector form of add (t, w, ev scalars or sequences aligned with keys)."""
        keys = list(keys)
        n = len(keys)
        tt = np.broadcast_to(np.asarray(t, dtype=np.float64), (n,)).tolist()
        ww = np.broadcast_to(np.asarray(w, dtype=np.float64), (n,)).tolist()
        ee = np.broadcast_to(np.asarray(ev, dtype=np.float64), (n,)).tolist()
        for j in range(n):
            self.add(keys[j], tt[j], ww[j], ee[j])

    def discard(self, key: Hashable) -> bool:
        """Remove a tracked key (its mass stays in the totals as untracked
        mass). Returns True if it was tracked."""
        i = self._idx.pop(key, None)
        if i is None:
            return False
        last = len(self._keys) - 1
        if i != last:
            k2 = self._keys[last]
            self._keys[i] = k2
            self._idx[k2] = i
            self._m[i], self._err[i], self._e[i] = self._m[last], self._err[last], self._e[last]
        self._keys.pop()
        self._m.pop()
        self._err.pop()
        self._e.pop()
        return True

    def scale(self, f: float) -> None:
        """Multiply every mass and evidence value by f >= 0 (thinning / HT)."""
        f = float(f)
        if not f >= 0.0:
            raise ValueError("scale factor must be >= 0")
        self._scale_all([f] * len(self._mh), [f] * len(self._eh))

    # ---------------------------------------------------------- confidence
    def reset_confidence(self, t: float, src: int = EV_M, dst: int = -1) -> None:
        """Confidence-channel reset (§6.9.4): the decayed value of evidence
        channel `dst` (default the last) becomes that of `src` at time t, for
        every counter, the total and the eviction mass."""
        if not self._eh or self._lm.L is None:
            return
        dst = dst % len(self._eh)
        f = self._fe(t)
        conv = f[src] / f[dst]
        for r in self._e:
            r[dst] = r[src] * conv
        self._tot_e[dst] = self._tot_e[src] * conv
        self._evict_e[dst] = self._evict_e[src] * conv

    def cap_confidence(self, t: float, src: int = EV_M, dst: int = -1) -> None:
        """Cap channel `dst` at the decayed value of `src` (while evolving)."""
        if not self._eh or self._lm.L is None:
            return
        dst = dst % len(self._eh)
        f = self._fe(t)
        conv = f[src] / f[dst]
        for r in self._e:
            r[dst] = min(r[dst], r[src] * conv)
        self._tot_e[dst] = min(self._tot_e[dst], self._tot_e[src] * conv)
        self._evict_e[dst] = min(self._evict_e[dst], self._evict_e[src] * conv)

    # --------------------------------------------------------------- reads
    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._idx

    def keys(self) -> List[Hashable]:
        return list(self._keys)

    def _t_ev(self, t: Optional[float]) -> Optional[float]:
        return t if t is not None else self._lm.L

    def count(self, key: Hashable, t: Optional[float] = None, ch: Optional[int] = None) -> float:
        """Decayed (over-)estimate of key's mass at t on channel ch (0 if untracked).
        t=None returns landmark units (only ratios are meaningful then)."""
        i = self._idx.get(key)
        if i is None:
            return 0.0
        c = self.primary if ch is None else ch
        return float(self._m[i][c] * self._fm(t)[c])

    def guaranteed(self, key: Hashable, t: Optional[float] = None, ch: Optional[int] = None) -> float:
        """Guaranteed lower bound count - err (0 if untracked)."""
        i = self._idx.get(key)
        if i is None:
            return 0.0
        c = self.primary if ch is None else ch
        return float(max(0.0, self._m[i][c] - self._err[i][c]) * self._fm(t)[c])

    def error(self, key: Hashable, t: Optional[float] = None, ch: Optional[int] = None) -> float:
        i = self._idx.get(key)
        if i is None:
            return 0.0
        c = self.primary if ch is None else ch
        return float(self._err[i][c] * self._fm(t)[c])

    def evidence(self, key: Hashable, t: Optional[float] = None, ch: int = -1) -> float:
        i = self._idx.get(key)
        if i is None or not self._eh:
            return 0.0
        return float(self._e[i][ch] * self._fe(self._t_ev(t))[ch])

    def total(self, t: Optional[float] = None, ch: Optional[int] = None) -> float:
        c = self.primary if ch is None else ch
        return float(self._tot_m[c] * self._fm(t)[c])

    def total_evidence(self, t: Optional[float] = None, ch: int = -1) -> float:
        if not self._eh:
            return 0.0
        return float(self._tot_e[ch] * self._fe(self._t_ev(t))[ch])

    def eviction_evidence(self, t: Optional[float] = None, ch: int = -1) -> float:
        if not self._eh:
            return 0.0
        return float(self._evict_e[ch] * self._fe(self._t_ev(t))[ch])

    def untracked(self, t: Optional[float] = None, ch: Optional[int] = None) -> float:
        """Mass not attributable to a tracked key: total minus the guaranteed
        tracked mass (the sum of inherited errors)."""
        c = self.primary if ch is None else ch
        g = sum(max(0.0, self._m[i][c] - self._err[i][c]) for i in range(len(self._keys)))
        return float(max(0.0, self._tot_m[c] - g) * self._fm(t)[c])

    def share(self, key: Hashable, t: Optional[float] = None, ch: Optional[int] = None) -> float:
        i = self._idx.get(key)
        if i is None:
            return 0.0
        c = self.primary if ch is None else ch
        tot = self._tot_m[c]
        return float(self._m[i][c] / tot) if tot > 0 else 0.0

    def n1(self, t: Optional[float] = None, ch: int = -1, thr: float = N1_EVIDENCE) -> int:
        """Tracked items whose evidence on channel ch is < thr (seen about once)."""
        if not self._eh or not self._keys:
            return 0
        f = self._fe(self._t_ev(t))[ch]
        return sum(1 for r in self._e if r[ch] * f < thr)

    def unseen(self, t: Optional[float] = None, ch: int = -1) -> float:
        """Good-Turing unseen mass with eviction correction (N1 + E + 0.5)/(N + 1)
        on evidence channel ch (default: the confidence channel). In [0, 1]."""
        N = self.total_evidence(t, ch)
        u = (self.n1(t, ch) + self.eviction_evidence(t, ch) + 0.5) / (N + 1.0)
        return float(min(1.0, max(0.0, u)))

    def items(self, t: Optional[float] = None, ch: Optional[int] = None,
              n: Optional[int] = None) -> List[Tuple[Hashable, float, float, float]]:
        """[(key, count, guaranteed, evidence_conf)] sorted by count desc."""
        if not self._keys:
            return []
        c = self.primary if ch is None else ch
        fm = self._fm(t)[c]
        fe = self._fe(self._t_ev(t))[-1] if self._eh else 0.0
        order = sorted(range(len(self._keys)), key=lambda j: -self._m[j][c])
        if n is not None:
            order = order[:int(n)]
        out = []
        for i in order:
            m = self._m[i][c]
            ev = float(self._e[i][-1] * fe) if self._eh else 0.0
            out.append((self._keys[i], float(m * fm), float(max(0.0, m - self._err[i][c]) * fm), ev))
        return out

    def distribution(self, t: Optional[float] = None, ch: Optional[int] = None
                     ) -> Tuple[List[Hashable], np.ndarray, float]:
        """(keys, mass shares, untracked share); shares use the guaranteed
        counts, so they sum with the untracked share to 1."""
        c = self.primary if ch is None else ch
        tot = self._tot_m[c]
        if not self._keys or tot <= 0:
            return [], np.zeros(0), 1.0 if tot > 0 else 0.0
        sh = np.asarray([max(0.0, self._m[i][c] - self._err[i][c]) / tot
                         for i in range(len(self._keys))])
        return list(self._keys), sh, float(max(0.0, 1.0 - sh.sum()))

    def min_count(self, t: Optional[float] = None) -> float:
        """Smallest tracked primary count when full (the SS error scale), else 0."""
        if len(self._keys) < self.k:
            return 0.0
        p = self.primary
        return float(min(r[p] for r in self._m) * self._fm(t)[p])

    @property
    def landmark(self) -> Optional[float]:
        return self._lm.L

    # --------------------------------------------------------------- merge
    def merge(self, other: "DecayedSpaceSaving") -> "DecayedSpaceSaving":
        """In-place mergeable-summary union (Agarwal et al. PODS 2012, SS form):
        a key missing on one side is charged that side's minimum count (when
        that side is full) as both count and error, then the top-k by primary
        count are kept. Error stays <= (N1 + N2) / k; returns self."""
        if self._mh != other._mh or self._eh != other._eh:
            raise ValueError("merge: channel half-lives differ")
        if other._lm.L is None:
            return self
        if self._lm.L is None:
            self._lm.L = other._lm.L
        L = max(self._lm.L, other._lm.L)
        self._rescale_to(L)
        dm = L - other._lm.L
        fm = [2.0 ** (-dm / h) for h in self._mh]
        fe = [2.0 ** (-dm / h) for h in self._eh]
        nm = len(self._mh)

        def sc(r: Sequence[float], f: Sequence[float]) -> List[float]:
            return [r[c] * f[c] for c in range(len(f))]
        om = [sc(r, fm) for r in other._m]
        oerr = [sc(r, fm) for r in other._err]
        oe = [sc(r, fe) for r in other._e]
        mins = [min(r[c] for r in self._m) for c in range(nm)] \
            if len(self._keys) >= self.k else [0.0] * nm
        omins = [min(r[c] for r in om) for c in range(nm)] \
            if len(other._keys) >= other.k else [0.0] * nm
        rows: Dict[Hashable, List[List[float]]] = {}
        for i, key in enumerate(self._keys):
            rows[key] = [list(self._m[i]), list(self._err[i]), list(self._e[i])]
        for i, key in enumerate(other._keys):
            if key in rows:
                r = rows[key]
                r[0] = [r[0][c] + om[i][c] for c in range(nm)]
                r[1] = [r[1][c] + oerr[i][c] for c in range(nm)]
                r[2] = [r[2][c] + oe[i][c] for c in range(len(fe))]
            else:
                rows[key] = [[om[i][c] + mins[c] for c in range(nm)],
                             [oerr[i][c] + mins[c] for c in range(nm)], list(oe[i])]
        for key in self._keys:
            if key not in other._idx:
                r = rows[key]
                r[0] = [r[0][c] + omins[c] for c in range(nm)]
                r[1] = [r[1][c] + omins[c] for c in range(nm)]
        p = self.primary
        ranked = sorted(rows.items(), key=lambda kv: (-kv[1][0][p], _sort_key(kv[0])))
        keep, drop = ranked[:self.k], ranked[self.k:]
        self._keys = [k for k, _ in keep]
        self._idx = {k: i for i, k in enumerate(self._keys)}
        self._m = [r[0] for _, r in keep]
        self._err = [r[1] for _, r in keep]
        self._e = [r[2] for _, r in keep]
        for c in range(nm):
            self._tot_m[c] += other._tot_m[c] * fm[c]
        for c in range(len(fe)):
            self._tot_e[c] += other._tot_e[c] * fe[c]
            self._evict_e[c] += other._evict_e[c] * fe[c]
            for _, r in drop:
                self._evict_e[c] += r[2][c]
        return self

    def copy(self) -> "DecayedSpaceSaving":
        return DecayedSpaceSaving.from_dict(self.to_dict())

    # --------------------------------------------------------- persistence
    def to_dict(self) -> Dict[str, Any]:
        nm, ne = len(self._mh), len(self._eh)
        n = len(self._keys)
        return {"fmt": 2, "k": self.k, "mass_hl": list(self._mh), "ev_hl": list(self._eh),
                "primary": self.primary, "L": self._lm.L, "keys": list(self._keys),
                "m": np.asarray(self._m, dtype=np.float64).reshape(n, nm),
                "err": np.asarray(self._err, dtype=np.float64).reshape(n, nm),
                "e": np.asarray(self._e, dtype=np.float64).reshape(n, ne),
                "tot_m": np.asarray(self._tot_m), "tot_e": np.asarray(self._tot_e),
                "evict_e": np.asarray(self._evict_e)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DecayedSpaceSaving":
        s = cls(int(d["k"]), d["mass_hl"], d["ev_hl"], int(d.get("primary", CH_M)))
        s._lm.L = None if d.get("L") is None else float(d["L"])
        keys = [tuple(k) if isinstance(k, list) else k for k in d.get("keys", [])]
        n = len(keys)
        s._keys = keys
        s._idx = {k: i for i, k in enumerate(keys)}
        nm, ne = len(s._mh), len(s._eh)
        s._m = np.asarray(d["m"], dtype=np.float64).reshape(n, nm).tolist() if n else []
        s._err = np.asarray(d["err"], dtype=np.float64).reshape(n, nm).tolist() if n else []
        s._e = np.asarray(d["e"], dtype=np.float64).reshape(n, ne).tolist() if n else []
        s._tot_m = [float(x) for x in d["tot_m"]]
        s._tot_e = [float(x) for x in d["tot_e"]]
        s._evict_e = [float(x) for x in d["evict_e"]]
        return s

    def nbytes(self) -> int:
        per = 8 * (2 * len(self._mh) + len(self._eh)) + 3 * 72 + 120
        return int(len(self._keys) * per + 400)


def _sort_key(k: Any) -> str:
    return repr(k)


# ================================================================== HLL
def _h64(item: Any) -> int:
    s = item if type(item) is str else str(item)
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "surrogatepass"),
                                          digest_size=8).digest(), "big")


class HLL:
    """HyperLogLog with 2^p one-byte registers (4 <= p <= 16); sigma ~ 1.04/sqrt(2^p).
    Same hashing as lib/sketch.HyperLogLog (blake2b-64 of str(item)), so a
    p=10 HLL has identical registers to that class. Merge is lossless."""

    __slots__ = ("p", "reg")

    def __init__(self, p: int = 10, reg: Optional[np.ndarray] = None) -> None:
        if not 4 <= int(p) <= 16:
            raise ValueError("HLL: p must be in [4, 16]")
        self.p = int(p)
        m = 1 << self.p
        self.reg = np.zeros(m, dtype=np.uint8) if reg is None else np.asarray(reg, dtype=np.uint8).copy()
        if self.reg.shape != (m,):
            raise ValueError("HLL: register shape mismatch")

    def add(self, item: Any) -> None:
        x = _h64(item)
        wbits = 64 - self.p
        idx = x >> wbits
        rho = wbits + 1 - (x & ((1 << wbits) - 1)).bit_length()
        if rho > self.reg[idx]:
            self.reg[idx] = rho

    def count(self) -> float:
        m = float(1 << self.p)
        alpha = {16: 0.673, 32: 0.697, 64: 0.709}.get(int(m), 0.7213 / (1.0 + 1.079 / m))
        z = float(np.ldexp(1.0, -self.reg.astype(np.int64)).sum())
        e = alpha * m * m / z
        if e <= 2.5 * m:
            v = int(np.count_nonzero(self.reg == 0))
            if v > 0:
                return m * math.log(m / v)
        return e

    def merge(self, other: "HLL") -> "HLL":
        if other.p != self.p:
            raise ValueError("HLL.merge: p differs")
        np.maximum(self.reg, other.reg, out=self.reg)
        return self

    def copy(self) -> "HLL":
        return HLL(self.p, self.reg)

    def to_dict(self) -> Dict[str, Any]:
        return {"p": self.p, "reg": self.reg.tobytes()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HLL":
        return cls(int(d["p"]), np.frombuffer(d["reg"], dtype=np.uint8))

    def nbytes(self) -> int:
        return int(self.reg.nbytes + 64)


class EpochHLL:
    """Distinct count over roughly the last one to two epochs: two HLLs
    (current, previous) rotated every `epoch_s` (default 7 d, §6.1). No decay."""

    __slots__ = ("epoch_s", "start", "cur", "prev")

    def __init__(self, p: int = 10, epoch_s: float = 7 * DAY) -> None:
        self.epoch_s = float(epoch_s)
        self.start: Optional[float] = None
        self.cur = HLL(p)
        self.prev = HLL(p)

    def _rotate(self, t: float) -> None:
        if self.start is None:
            self.start = float(t)
            return
        k = math.floor((t - self.start) / self.epoch_s)
        if k >= 2:
            self.prev = HLL(self.cur.p)
            self.cur = HLL(self.cur.p)
            self.start += k * self.epoch_s
        elif k == 1:
            self.prev = self.cur
            self.cur = HLL(self.prev.p)
            self.start += self.epoch_s

    def add(self, item: Any, t: float) -> None:
        self._rotate(float(t))
        self.cur.add(item)

    def count(self, t: Optional[float] = None) -> float:
        if t is not None:
            self._rotate(float(t))
        return self.cur.copy().merge(self.prev).count()

    def merge(self, other: "EpochHLL") -> "EpochHLL":
        self.cur.merge(other.cur)
        self.prev.merge(other.prev)
        if self.start is None:
            self.start = other.start
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {"epoch_s": self.epoch_s, "start": self.start, "cur": self.cur.to_dict(),
                "prev": self.prev.to_dict()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EpochHLL":
        e = cls(int(d["cur"]["p"]), float(d["epoch_s"]))
        e.start = d.get("start")
        e.cur = HLL.from_dict(d["cur"])
        e.prev = HLL.from_dict(d["prev"])
        return e

    def nbytes(self) -> int:
        return self.cur.nbytes() + self.prev.nbytes() + 32


# ============================================================== t-digest
class TDigest:
    """Merging t-digest (Dunning & Ertl 2019) with forward-decayed centroid
    weights at one half-life (default H_m). Scale function k1:
    k(q) = delta / (2 pi) * asin(2q - 1); adjacent centroids merge while their
    k-span <= 1, which bounds the centroid count by ~delta and gives rank
    error O(q (1 - q) / delta). min / max are exact over everything seen
    (not decayed; P06 keeps its own daily ring for ranges)."""

    __slots__ = ("delta", "hl", "_lm", "_mu", "_w", "_buf_x", "_buf_w", "_bufcap", "vmin", "vmax",
                 "_tot")

    def __init__(self, delta: float = 50.0, hl: float = H_M) -> None:
        self.delta = float(delta)
        self.hl = float(hl)
        self._lm = _Landmark(None, self.hl)
        self._mu = np.zeros(0)
        self._w = np.zeros(0)
        self._buf_x: List[float] = []
        self._buf_w: List[float] = []
        self._bufcap = max(32, int(4 * self.delta))
        self.vmin = math.inf
        self.vmax = -math.inf
        self._tot = 0.0

    def _rescale(self, t: float) -> None:
        old = self._lm.ensure(t)
        if old is None:
            return
        self._flush()
        f = 2.0 ** (-(t - old) / self.hl)
        self._w *= f
        self._tot *= f
        self._lm.L = float(t)

    def add(self, x: float, t: float, w: float = 1.0) -> None:
        x = float(x)
        w = float(w)
        if not (math.isfinite(x) and w > 0.0 and math.isfinite(w)):
            return
        t = float(t)
        self._rescale(t)
        g = 2.0 ** ((t - self._lm.L) / self.hl) * w
        self._buf_x.append(x)
        self._buf_w.append(g)
        self._tot += g
        if x < self.vmin:
            self.vmin = x
        if x > self.vmax:
            self.vmax = x
        if len(self._buf_x) >= self._bufcap:
            self._flush()

    def add_many(self, xs: Sequence[float], t: Any, w: Any = 1.0) -> None:
        xs = np.asarray(xs, dtype=np.float64)
        n = xs.size
        tt = np.broadcast_to(np.asarray(t, dtype=np.float64), (n,))
        ww = np.broadcast_to(np.asarray(w, dtype=np.float64), (n,))
        ok = np.isfinite(xs) & np.isfinite(ww) & (ww > 0)
        if not ok.any():
            return
        xs, tt, ww = xs[ok], tt[ok], ww[ok]
        tmax = float(tt.max())
        self._rescale(tmax)
        g = np.exp2((tt - self._lm.L) / self.hl) * ww
        self._buf_x.extend(xs.tolist())
        self._buf_w.extend(g.tolist())
        self._tot += float(g.sum())
        self.vmin = min(self.vmin, float(xs.min()))
        self.vmax = max(self.vmax, float(xs.max()))
        if len(self._buf_x) >= self._bufcap:
            self._flush()

    def _k(self, q: np.ndarray) -> np.ndarray:
        return self.delta / (2.0 * math.pi) * np.arcsin(np.clip(2.0 * q - 1.0, -1.0, 1.0))

    def _flush(self) -> None:
        if not self._buf_x:
            return
        mu = np.concatenate([self._mu, np.asarray(self._buf_x)])
        w = np.concatenate([self._w, np.asarray(self._buf_w)])
        self._buf_x = []
        self._buf_w = []
        self._compress(mu, w)

    def _compress(self, mu: np.ndarray, w: np.ndarray) -> None:
        order = np.argsort(mu, kind="stable")
        mu, w = mu[order], w[order]
        tot = float(w.sum())
        if tot <= 0 or mu.size <= 1:
            self._mu, self._w = mu, w
            return
        out_mu: List[float] = []
        out_w: List[float] = []
        cum = 0.0
        cur_mu, cur_w = float(mu[0]), float(w[0])
        k_left = float(self._k(np.asarray(0.0)))
        for i in range(1, mu.size):
            wi = float(w[i])
            q_right = (cum + cur_w + wi) / tot
            if float(self._k(np.asarray(q_right))) - k_left <= 1.0:
                cur_mu = cur_mu + (float(mu[i]) - cur_mu) * wi / (cur_w + wi)
                cur_w += wi
            else:
                out_mu.append(cur_mu)
                out_w.append(cur_w)
                cum += cur_w
                k_left = float(self._k(np.asarray(cum / tot)))
                cur_mu, cur_w = float(mu[i]), wi
        out_mu.append(cur_mu)
        out_w.append(cur_w)
        self._mu = np.asarray(out_mu)
        self._w = np.asarray(out_w)

    def total(self, t: Optional[float] = None) -> float:
        if self._lm.L is None:
            return 0.0
        if t is None:
            return self._tot
        return self._tot * 2.0 ** (-(float(t) - self._lm.L) / self.hl)

    def centroids(self) -> Tuple[np.ndarray, np.ndarray]:
        self._flush()
        return self._mu.copy(), self._w.copy()

    def quantile(self, q: float) -> float:
        """Interpolated q-quantile of the decayed distribution (NaN if empty)."""
        self._flush()
        n = self._mu.size
        if n == 0:
            return math.nan
        q = min(1.0, max(0.0, float(q)))
        if n == 1:
            return float(self._mu[0])
        tot = float(self._w.sum())
        target = q * tot
        # centroid i covers cumulative mass [c_i - w_i/2, c_i + w_i/2] around its mean
        cum = np.cumsum(self._w) - self._w / 2.0
        if target <= cum[0]:
            # between vmin and the first centroid
            lo, hi = self.vmin, float(self._mu[0])
            frac = target / cum[0] if cum[0] > 0 else 0.0
            return lo + (hi - lo) * frac
        if target >= cum[-1]:
            lo, hi = float(self._mu[-1]), self.vmax
            rest = tot - cum[-1]
            frac = (target - cum[-1]) / rest if rest > 0 else 1.0
            return lo + (hi - lo) * frac
        j = int(np.searchsorted(cum, target, side="right"))
        c0, c1 = cum[j - 1], cum[j]
        frac = (target - c0) / (c1 - c0) if c1 > c0 else 0.0
        return float(self._mu[j - 1] + (self._mu[j] - self._mu[j - 1]) * frac)

    def cdf(self, x: float) -> float:
        """Interpolated decayed CDF at x in [0, 1] (NaN if empty)."""
        self._flush()
        n = self._mu.size
        if n == 0:
            return math.nan
        x = float(x)
        if x < self.vmin:
            return 0.0
        if x >= self.vmax:
            return 1.0
        tot = float(self._w.sum())
        if n == 1:
            span = self.vmax - self.vmin
            return (x - self.vmin) / span if span > 0 else 0.5
        cum = np.cumsum(self._w) - self._w / 2.0
        if x < self._mu[0]:
            span = float(self._mu[0]) - self.vmin
            return float(cum[0] * ((x - self.vmin) / span if span > 0 else 1.0) / tot)
        if x >= self._mu[-1]:
            span = self.vmax - float(self._mu[-1])
            frac = (x - float(self._mu[-1])) / span if span > 0 else 1.0
            return float((cum[-1] + (tot - cum[-1]) * frac) / tot)
        j = int(np.searchsorted(self._mu, x, side="right"))
        m0, m1 = float(self._mu[j - 1]), float(self._mu[j])
        frac = (x - m0) / (m1 - m0) if m1 > m0 else 0.5
        return float((cum[j - 1] + (cum[j] - cum[j - 1]) * frac) / tot)

    def merge(self, other: "TDigest") -> "TDigest":
        if other._lm.L is None:
            return self
        other._flush()
        self._flush()
        if self._lm.L is None:
            self._lm.L = other._lm.L
        L = max(self._lm.L, other._lm.L)
        if L != self._lm.L:
            f = 2.0 ** (-(L - self._lm.L) / self.hl)
            self._w *= f
            self._tot *= f
            self._lm.L = L
        fo = 2.0 ** (-(L - other._lm.L) / self.hl)
        self._compress(np.concatenate([self._mu, other._mu]),
                       np.concatenate([self._w, other._w * fo]))
        self._tot += other._tot * fo
        self.vmin = min(self.vmin, other.vmin)
        self.vmax = max(self.vmax, other.vmax)
        return self

    def n_centroids(self) -> int:
        self._flush()
        return int(self._mu.size)

    def to_dict(self) -> Dict[str, Any]:
        self._flush()
        return {"delta": self.delta, "hl": self.hl, "L": self._lm.L, "mu": self._mu.copy(),
                "w": self._w.copy(), "min": self.vmin, "max": self.vmax, "tot": self._tot}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TDigest":
        td = cls(float(d["delta"]), float(d["hl"]))
        td._lm.L = d.get("L")
        td._mu = np.asarray(d["mu"], dtype=np.float64).copy()
        td._w = np.asarray(d["w"], dtype=np.float64).copy()
        td.vmin, td.vmax, td._tot = float(d["min"]), float(d["max"]), float(d["tot"])
        return td

    def nbytes(self) -> int:
        return int(self._mu.nbytes + self._w.nbytes + 16 * len(self._buf_x) + 160)


# ============================================================= Count-Min
class CountMin:
    """Count-Min sketch (Cormode & Muthukrishnan 2005) with forward decay at one
    half-life. Point query overestimates by at most e/w * N with probability
    >= 1 - e^-d (N = decayed total); never underestimates. Mergeable (sum)."""

    __slots__ = ("w", "d", "hl", "_lm", "table", "_tot", "seed")

    def __init__(self, w: int = 1024, d: int = 4, hl: float = H_M, seed: int = 0) -> None:
        if int(w) < 2 or int(d) < 1:
            raise ValueError("CountMin: w >= 2 and d >= 1 required")
        self.w, self.d, self.hl, self.seed = int(w), int(d), float(hl), int(seed)
        self._lm = _Landmark(None, self.hl)
        self.table = np.zeros((self.d, self.w))
        self._tot = 0.0

    def _cols(self, key: Any) -> np.ndarray:
        s = (key if type(key) is str else repr(key)) + f"\x1f{self.seed}"
        h = hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=16).digest()
        h1 = int.from_bytes(h[:8], "big")
        h2 = int.from_bytes(h[8:], "big") | 1
        return np.asarray([(h1 + i * h2) % self.w for i in range(self.d)], dtype=np.intp)

    def _rescale(self, t: float) -> None:
        old = self._lm.ensure(t)
        if old is None:
            return
        f = 2.0 ** (-(t - old) / self.hl)
        self.table *= f
        self._tot *= f
        self._lm.L = float(t)

    def add(self, key: Any, t: float, w: float = 1.0) -> None:
        w = float(w)
        if not (w > 0 and math.isfinite(w)):
            return
        t = float(t)
        self._rescale(t)
        g = 2.0 ** ((t - self._lm.L) / self.hl) * w
        self.table[np.arange(self.d), self._cols(key)] += g
        self._tot += g

    def query(self, key: Any, t: Optional[float] = None) -> float:
        if self._lm.L is None:
            return 0.0
        v = float(self.table[np.arange(self.d), self._cols(key)].min())
        return v if t is None else v * 2.0 ** (-(float(t) - self._lm.L) / self.hl)

    def total(self, t: Optional[float] = None) -> float:
        if self._lm.L is None:
            return 0.0
        return self._tot if t is None else self._tot * 2.0 ** (-(float(t) - self._lm.L) / self.hl)

    def error_bound(self, t: Optional[float] = None) -> float:
        """e / w * N: the additive overestimate bound (w.p. 1 - e^-d)."""
        return math.e / self.w * self.total(t)

    def merge(self, other: "CountMin") -> "CountMin":
        if (other.w, other.d, other.seed) != (self.w, self.d, self.seed):
            raise ValueError("CountMin.merge: shape or seed differs")
        if other._lm.L is None:
            return self
        if self._lm.L is None:
            self._lm.L = other._lm.L
        L = max(self._lm.L, other._lm.L)
        if L != self._lm.L:
            f = 2.0 ** (-(L - self._lm.L) / self.hl)
            self.table *= f
            self._tot *= f
            self._lm.L = L
        fo = 2.0 ** (-(L - other._lm.L) / self.hl)
        self.table += other.table * fo
        self._tot += other._tot * fo
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {"w": self.w, "d": self.d, "hl": self.hl, "seed": self.seed, "L": self._lm.L,
                "table": self.table.copy(), "tot": self._tot}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CountMin":
        cm = cls(int(d["w"]), int(d["d"]), float(d["hl"]), int(d.get("seed", 0)))
        cm._lm.L = d.get("L")
        cm.table = np.asarray(d["table"], dtype=np.float64).copy()
        cm._tot = float(d["tot"])
        return cm

    def nbytes(self) -> int:
        return int(self.table.nbytes + 96)


# ================================================================= ADWIN
class ADWIN:
    """ADWIN2 (Bifet & Gavalda, SDM 2007) with exponential-histogram buckets
    (at most M per row). `add(x)` returns +1 when the recent sub-window's
    mean is significantly higher than the older one's (a cut was made), -1 when
    lower, 0 otherwise. The cut test (checked every `clock` insertions) is
        |mu0 - mu1| >= sqrt(2 / m * var_W * ln(2 / d')) + 2 / (3 m) * ln(2 / d'),
        m = 1 / (1/n0 + 1/n1), d' = delta / ln(n)
    Memory O(M log W)."""

    __slots__ = ("delta", "M", "clock", "min_sub", "rows", "n", "total", "var", "_t",
                 "detections", "last_change")

    def __init__(self, delta: float = 0.002, M: int = 5, clock: int = 32, min_sub: int = 5) -> None:
        self.delta, self.M, self.clock, self.min_sub = float(delta), int(M), int(clock), int(min_sub)
        self.rows: List[List[List[float]]] = []    # row i: buckets [n, total, M2], oldest first
        self.n = 0.0
        self.total = 0.0
        self.var = 0.0                             # sum of squared deviations (M2) of the window
        self._t = 0
        self.detections = 0
        self.last_change = 0

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n > 0 else math.nan

    @property
    def width(self) -> int:
        return int(self.n)

    def add(self, x: float) -> int:
        x = float(x)
        if not math.isfinite(x):
            return 0
        # window moments
        if self.n > 0:
            mu = self.total / self.n
            self.var += self.n / (self.n + 1.0) * (x - mu) ** 2
        self.n += 1.0
        self.total += x
        if not self.rows:
            self.rows.append([])
        self.rows[0].append([1.0, x, 0.0])
        self._compress()
        self._t += 1
        if self._t % self.clock:
            return 0
        return self._detect()

    @staticmethod
    def _combine(a: List[float], b: List[float]) -> List[float]:
        n = a[0] + b[0]
        mu_a, mu_b = a[1] / a[0], b[1] / b[0]
        return [n, a[1] + b[1], a[2] + b[2] + a[0] * b[0] / n * (mu_a - mu_b) ** 2]

    def _compress(self) -> None:
        i = 0
        while i < len(self.rows):
            row = self.rows[i]
            if len(row) <= self.M:
                break
            a, b = row.pop(0), row.pop(0)            # the two oldest of this row
            if i + 1 == len(self.rows):
                self.rows.append([])
            self.rows[i + 1].append(self._combine(a, b))
            i += 1

    def _iter_old_to_new(self) -> Iterator[List[float]]:
        for row in reversed(self.rows):
            for b in row:
                yield b

    def _drop_oldest(self) -> None:
        for i in range(len(self.rows) - 1, -1, -1):
            if self.rows[i]:
                b = self.rows[i].pop(0)
                n0, t0, v0 = b
                n1 = self.n - n0
                if n1 <= 0:
                    self.rows = []
                    self.n = self.total = self.var = 0.0
                    return
                mu_all = self.total / self.n
                mu_b = t0 / n0
                mu_rest = (self.total - t0) / n1
                self.var = max(0.0, self.var - v0 - n0 * n1 / self.n * (mu_b - mu_rest) ** 2)
                self.n = n1
                self.total -= t0
                del mu_all
                while self.rows and not self.rows[-1]:
                    self.rows.pop()
                return

    def _detect(self) -> int:
        change = 0
        cut = True
        while cut and self.n >= 2 * self.min_sub:
            cut = False
            n0 = 0.0
            t0 = 0.0
            var_w = self.var / self.n
            ln = math.log(2.0 * math.log(max(self.n, math.e)) / self.delta)
            for b in self._iter_old_to_new():
                n0 += b[0]
                t0 += b[1]
                n1 = self.n - n0
                if n1 < self.min_sub:
                    break
                if n0 < self.min_sub:
                    continue
                mu0, mu1 = t0 / n0, (self.total - t0) / n1
                m = 1.0 / (1.0 / n0 + 1.0 / n1)
                eps = math.sqrt(2.0 / m * var_w * ln) + 2.0 / (3.0 * m) * ln
                if abs(mu0 - mu1) > eps:
                    change = 1 if mu1 > mu0 else -1
                    cut = True
                    self._drop_oldest()
                    break
        if change:
            self.detections += 1
            self.last_change = change
        return change

    def to_dict(self) -> Dict[str, Any]:
        return {"delta": self.delta, "M": self.M, "clock": self.clock, "min_sub": self.min_sub,
                "rows": [[list(b) for b in r] for r in self.rows], "n": self.n,
                "total": self.total, "var": self.var, "t": self._t,
                "detections": self.detections, "last_change": self.last_change}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ADWIN":
        a = cls(d["delta"], d["M"], d["clock"], d.get("min_sub", 5))
        a.rows = [[list(b) for b in r] for r in d["rows"]]
        a.n, a.total, a.var, a._t = d["n"], d["total"], d["var"], d["t"]
        a.detections, a.last_change = d["detections"], d["last_change"]
        return a

    def nbytes(self) -> int:
        return int(sum(len(r) for r in self.rows) * 100 + 120)


class PageHinkley:
    """Two-sided Page-Hinkley test. With a reference mean mu (the running mean,
    or `ref` supplied per update, e.g. the H_l mean) and allowance delta:
        m_up += x - mu - delta ; alarm up   when m_up - min(m_up) > lam
        m_dn += x - mu + delta ; alarm down when max(m_dn) - m_dn > lam
    `update` returns +1 / -1 / 0; the state resets after an alarm."""

    __slots__ = ("lam", "delta", "n", "mean", "m_up", "min_up", "m_dn", "max_dn")

    def __init__(self, lam: float, delta: float) -> None:
        self.lam, self.delta = float(lam), float(delta)
        self.reset()

    def reset(self) -> None:
        self.n = 0.0
        self.mean = 0.0
        self.m_up = self.min_up = 0.0
        self.m_dn = self.max_dn = 0.0

    def update(self, x: float, ref: Optional[float] = None) -> int:
        x = float(x)
        if not math.isfinite(x):
            return 0
        self.n += 1.0
        self.mean += (x - self.mean) / self.n
        mu = self.mean if ref is None else float(ref)
        self.m_up += x - mu - self.delta
        self.min_up = min(self.min_up, self.m_up)
        self.m_dn += x - mu + self.delta
        self.max_dn = max(self.max_dn, self.m_dn)
        if self.m_up - self.min_up > self.lam:
            self.reset()
            return 1
        if self.max_dn - self.m_dn > self.lam:
            self.reset()
            return -1
        return 0

    def to_dict(self) -> Dict[str, Any]:
        return {"lam": self.lam, "delta": self.delta, "n": self.n, "mean": self.mean,
                "m_up": self.m_up, "min_up": self.min_up, "m_dn": self.m_dn, "max_dn": self.max_dn}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PageHinkley":
        p = cls(d["lam"], d["delta"])
        for k in ("n", "mean", "m_up", "min_up", "m_dn", "max_dn"):
            setattr(p, k, float(d[k]))
        return p


# ===================================================== weighted reservoir
class WeightedReservoir:
    """Weighted random sampling without replacement (Efraimidis & Spirakis
    2006) with time-decayed keys: key = ln(u) / (w * 2^((t - L) / H)), the R
    largest keys are kept. Equivalent in distribution to A-ExpJ; implemented
    as A-Res with a min-heap (O(log R) per offer). Changing the landmark
    multiplies every key by one positive constant, so order is preserved.
    Uniforms come from `u` (e.g. lib/combine.seeded_uniform) or a seeded RNG."""

    __slots__ = ("R", "hl", "L", "_heap", "_seq", "_rng", "offered")

    def __init__(self, R: int, hl: Optional[float] = H_M, seed: int = 0) -> None:
        if int(R) < 1:
            raise ValueError("WeightedReservoir: R >= 1")
        self.R = int(R)
        self.hl = None if hl is None else float(hl)
        self.L: Optional[float] = None
        self._heap: List[Tuple[float, int, Any, float, float]] = []   # (key, seq, item, w, t)
        self._seq = 0
        self._rng = np.random.default_rng(seed)
        self.offered = 0.0

    def offer(self, item: Any, w: float = 1.0, t: float = 0.0, u: Optional[float] = None) -> bool:
        """Offer an item; True if it is (currently) in the sample."""
        w = float(w)
        if not (w > 0 and math.isfinite(w)):
            return False
        t = float(t)
        self.offered += w
        if self.hl is not None:
            if self.L is None:
                self.L = t
            elif (t - self.L) / self.hl > RESCALE_EXP:
                f = 2.0 ** ((t - self.L) / self.hl)      # keys * f keeps order, avoids overflow
                self._heap = [(k * f, s, it, ww, tt) for (k, s, it, ww, tt) in self._heap]
                heapq.heapify(self._heap)
                self.L = t
            g = 2.0 ** ((t - self.L) / self.hl)
        else:
            g = 1.0
        uu = float(self._rng.random()) if u is None else float(u)
        uu = min(max(uu, 1e-300), 1.0 - 1e-16)
        key = math.log(uu) / (w * g)
        self._seq += 1
        entry = (key, self._seq, item, w, t)
        if len(self._heap) < self.R:
            heapq.heappush(self._heap, entry)
            return True
        if key > self._heap[0][0]:
            heapq.heapreplace(self._heap, entry)
            return True
        return False

    def __len__(self) -> int:
        return len(self._heap)

    def items(self) -> List[Tuple[Any, float, float]]:
        """[(item, w, t)] in insertion order."""
        return [(it, w, t) for (_, _, it, w, t) in sorted(self._heap, key=lambda e: e[1])]

    def to_dict(self) -> Dict[str, Any]:
        return {"R": self.R, "hl": self.hl, "L": self.L, "heap": list(self._heap),
                "seq": self._seq, "offered": self.offered}

    @classmethod
    def from_dict(cls, d: Dict[str, Any], seed: int = 0) -> "WeightedReservoir":
        r = cls(d["R"], d["hl"], seed)
        r.L = d["L"]
        r._heap = [tuple(e) for e in d["heap"]]
        heapq.heapify(r._heap)
        r._seq, r.offered = int(d["seq"]), float(d["offered"])
        return r


# ======================================================= LRU and evidence
class LRU:
    """Bounded mapping; inserting beyond `cap` evicts the least recently used.
    `get` refreshes recency. `evicted` counts evictions."""

    __slots__ = ("cap", "_d", "evicted")

    def __init__(self, cap: int) -> None:
        if int(cap) < 1:
            raise ValueError("LRU: cap >= 1")
        self.cap = int(cap)
        self._d: "OrderedDict[Hashable, Any]" = OrderedDict()
        self.evicted = 0

    def get(self, key: Hashable, default: Any = None) -> Any:
        v = self._d.get(key, _MISSING)
        if v is _MISSING:
            return default
        self._d.move_to_end(key)
        return v

    def peek(self, key: Hashable, default: Any = None) -> Any:
        return self._d.get(key, default)

    def put(self, key: Hashable, value: Any) -> Optional[Tuple[Hashable, Any]]:
        """Insert / refresh; returns the evicted (key, value) if any."""
        d = self._d
        if key in d:
            d.move_to_end(key)
            d[key] = value
            return None
        d[key] = value
        if len(d) > self.cap:
            self.evicted += 1
            return d.popitem(last=False)
        return None

    def pop(self, key: Hashable, default: Any = None) -> Any:
        return self._d.pop(key, default)

    def set_cap(self, cap: int) -> List[Tuple[Hashable, Any]]:
        self.cap = max(1, int(cap))
        out = []
        while len(self._d) > self.cap:
            out.append(self._d.popitem(last=False))
            self.evicted += 1
        return out

    def __len__(self) -> int:
        return len(self._d)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._d

    def items(self) -> List[Tuple[Hashable, Any]]:
        return list(self._d.items())


_MISSING = object()
TAU_BURST = 300.0
RUN_MAX = 3600.0              # a burst run restarts after this long however dense it is


class BurstEvidence:
    """Evidence units per learned row (§6.5.4). Keyed by an arbitrary run key,
    typically (ip, sess.key, node): if the key's previous learned row was more
    than tau_burst seconds earlier the run counter restarts, and a row gets
        omega = factor / (r + 1),  then r <- r + 1        (omega <= factor <= 1)
    so a burst of k rows contributes H(k) ~ ln k + 0.58 units. Mass (HT weight,
    aggregation count, sampling rate) never enters omega. LRU-capped.

    A run also restarts once it is older than run_max (default 1 h) however
    dense it is: a source that repeats one action without ever pausing for
    tau_burst (a 60-s health check, a poller) is one run per hour, so it earns
    ~24 H(k) units a day instead of one run for its whole life. Measured on
    pack O (integration, 2026-09-30): the monitor's GET /health node held
    n_c = 7 after 21 days (1 440 rows a day) and was never confirmed."""

    __slots__ = ("tau", "_lru", "run_max")

    def __init__(self, cap: int = 65536, tau_burst: float = TAU_BURST,
                 run_max: float = RUN_MAX) -> None:
        self.tau = float(tau_burst)
        self.run_max = float(run_max)
        self._lru = LRU(cap)

    def unit(self, key: Hashable, t: float, factor: float = 1.0) -> float:
        f = min(1.0, max(0.0, float(factor)))
        st = self._lru.get(key)
        t = float(t)
        run_max = getattr(self, "run_max", RUN_MAX)
        if st is None or t - st[0] > self.tau or t < st[0] - self.tau \
                or (len(st) > 2 and t - st[2] >= run_max):
            r = 0
        else:
            r = st[1]
        start = t if not r else (st[2] if len(st) > 2 else st[0])
        self._lru.put(key, (max(t, st[0]) if st is not None and r else t, r + 1, start))
        return f / (r + 1.0)

    def set_cap(self, cap: int) -> None:
        self._lru.set_cap(cap)

    def __len__(self) -> int:
        return len(self._lru)


# ========================================================== HHH selection
def hhh_select(levels: Sequence[DecayedSpaceSaving], parent: Callable[[int, Hashable], Hashable],
               phi: float, t: Optional[float] = None, ch: Optional[int] = None
               ) -> List[Tuple[int, Hashable, float]]:
    """Hierarchical heavy hitters over a *nested* chain of levels (0 = finest)
    kept as one SpaceSaving per level (Cormode et al. VLDB 2003; Mitzenmacher,
    Steinke & Thaler 2012). parent(l, key) maps a level-l key to its level-(l+1)
    key. Conditioned count of p: count(p) minus the counts of the HHH
    descendants whose nearest HHH ancestor is p. Returns
    [(level, key, conditioned count)] with conditioned count >= phi * N,
    finest first. N is the top level's total."""
    if not levels:
        return []
    N = levels[-1].total(t, ch)
    thr = float(phi) * N
    out: List[Tuple[int, Hashable, float]] = []
    carried: Dict[Hashable, float] = {}         # HHH mass to discount at this level, per key
    for lvl, ss in enumerate(levels):
        nxt: Dict[Hashable, float] = {}
        seen = set()
        for key, cnt, _, _ in ss.items(t, ch):
            seen.add(key)
            cond = cnt - carried.get(key, 0.0)
            if cond >= thr:
                out.append((lvl, key, cond))
                discount = cnt
            else:
                discount = carried.get(key, 0.0)
            if lvl + 1 < len(levels) and discount > 0:
                pk = parent(lvl, key)
                nxt[pk] = nxt.get(pk, 0.0) + discount
        if lvl + 1 < len(levels):
            for key, disc in carried.items():          # HHH mass under an untracked key
                if key not in seen and disc > 0:
                    pk = parent(lvl, key)
                    nxt[pk] = nxt.get(pk, 0.0) + disc
        carried = nxt
    return out


def nbytes_of(obj: Any) -> int:
    """Approximate bytes of a psketch structure or a container of them."""
    fn = getattr(obj, "nbytes", None)
    if callable(fn):
        return int(fn())
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return 64 + sum(64 + nbytes_of(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return 56 + sum(8 + nbytes_of(v) for v in obj)
    return 32
