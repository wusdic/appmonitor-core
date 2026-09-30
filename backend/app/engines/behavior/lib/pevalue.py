"""Anytime-valid split and exception evidence (docs/lib3/progressive.md §6.5.3-§6.5.5, §6.7).

STATUS: implemented (W-P0). Pure accumulators; P04 owns when they are fed and
what happens on a decision.

SplitStats — the split statistics of one learning leaf. For up to C candidate
split attributes c = (attr, level) it keeps, per value slot j (a SpaceSaving of
k_v slots over the candidate's generalised values plus `other`; an evicted
slot folds its counts into `other`), per target t and per bin b, the
evidence-weighted counts n[c, j, t, b]. Slot eviction: a new value takes the
slot of lowest *priority* (the slot's own evidence, decayed with a half-life
of PRI_HALF_UNITS evidence units) and that slot's counts fold into `other`.
This deviates from a plain Space-Saving (where the newcomer inherits the
minimum count). Measured share of checks at which three recurring sources are
all held, among 40 churning values, k_v = 8, 5 seeds
(scratch harness, same update rule): at 20 % of events each, both policies
100 %; at 10 % each, inheritance 53 %, decayed own-evidence 83 %; at 5 % each
both < 7 %. Values below ~1/k_v of the candidate's evidence are therefore not
reliably separable at one level; the hierarchy's coarser levels (/24, grp)
are what isolate them (§6.22). For each learned event with evidence
omega <= 1 (PPC-9: mass never enters) and the leaf's *prequential* predictive
p_leaf[t, b] (computed from the past only):

    l_leaf,t = -log2 p_leaf,t(b_t)
    l_c,t    = -log2 (n[c, j, t, b_t] + alpha p_leaf,t(b_t)) / (n[c, j, t, .] + alpha)
    L1[c, t] += omega * l_c,t                          prequential code under the split
    d_c       = omega * sum_t (l_leaf,t - l_c,t)       bits saved (MDL gain G_c += d_c)
    then n[c, j, t, b_t] += omega

Rule (V), anytime-valid (universal inference + Ville's inequality):
    e_c = (1/T_c) sum_t 2^(L0[c, t] - L1[c, t]),  L0 = pooled ML code length
    split allowed only when log2 e_c >= tau0 + log2 C_ever.
Under the null (t independent of the candidate's value, evidence units
conditionally independent), each 2^(L0 - L1) is bounded by a non-negative
supermartingale started at 1 (because omega <= 1, Jensen), an average of
e-processes is an e-process whatever the dependence between targets, and
P(sup e >= 2^tau) <= 2^-tau however often it is checked. C_ever (candidates
ever tracked, never decreasing) charges the multiplicity per candidate.

Rules (G), (S), (D), (M) and the greedy value grouping follow §6.5.5.

ExcTracker — the leave-x-out exception test of §6.7 at a confirmed node.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from . import pmdl

TAU0 = 10.0               # bits
DELTA = 1e-4              # rule (S) confidence, made time-uniform per check
TAU_TIE = 0.05            # bits per evidence unit (VFDT tie break)
N_CHILD_MIN = 5.0         # evidence units per value group, rule (M)
DIVERSITY_UNITS = 200.0   # rule (D): >= 2 dates or >= 200 units
ALPHA = 2.0
K_V = 8
K_B = 9
C_MAX = 6
PRI_HALF_UNITS = 256.0    # half-life (evidence units) of the value-slot eviction priority
_NEG = -math.inf


def _check_omega(omega: float) -> float:
    w = float(omega)
    if not math.isfinite(w) or w < 0.0:
        raise ValueError(f"evidence unit must be finite and >= 0, got {omega!r}")
    if w > 1.0 + 1e-12:
        raise ValueError(f"evidence unit {w} > 1: mass must not enter evidence (PPC-9)")
    return min(1.0, w)


@dataclass
class SplitDecision:
    best: Optional[int]
    second: Optional[int]
    log2_e: float = _NEG
    threshold: float = math.inf
    pass_v: bool = False
    gain: float = 0.0
    l_split: float = 0.0
    pass_g: bool = False
    eps: float = math.inf
    mu_diff: float = 0.0
    pass_s: bool = False
    pass_d: bool = False
    groups: List[List[Hashable]] = field(default_factory=list)
    other_group: int = -1
    group_evidence: List[float] = field(default_factory=list)
    pass_m: bool = False

    @property
    def accept(self) -> bool:
        return self.pass_v and self.pass_g and self.pass_s and self.pass_d and self.pass_m


OTHER = "__other__"


class SplitStats:
    """Split statistics block of a learning leaf. Memory
    C (k_v + 1) T k_b + O(C^2) floats (about 21 KB at the defaults)."""

    def __init__(self, n_targets: int, k_b: int = K_B, k_v: int = K_V, C: int = C_MAX,
                 alpha: float = ALPHA) -> None:
        if n_targets < 1:
            raise ValueError("SplitStats: at least one target")
        self.T, self.kb, self.kv, self.C, self.alpha = int(n_targets), int(k_b), int(k_v), int(C), float(alpha)
        C_, J = self.C, self.kv + 1
        self.cnt = np.zeros((C_, J, self.T, self.kb))
        self.den = np.zeros((C_, J, self.T))                 # cnt summed over bins
        self.L1 = np.zeros((C_, self.T))
        self.G = np.zeros(C_)
        self.n = np.zeros(C_)
        self.rows = np.zeros(C_)
        self.keys: List[Optional[Hashable]] = [None] * C_
        self.card: np.ndarray = np.ones(C_)
        self.ordinal = np.zeros(C_, dtype=np.int64)
        self.tmask = np.ones((C_, self.T), dtype=bool)
        self.slot_of: List[Dict[Hashable, int]] = [dict() for _ in range(C_)]
        self.slot_val: List[List[Optional[Hashable]]] = [[None] * self.kv for _ in range(C_)]
        self.slot_ev = np.zeros((C_, J))                     # evidence counted in each slot
        self.slot_pri = np.zeros((C_, self.kv))              # decayed slot priority (eviction order)
        self.day0 = np.full(C_, -1, dtype=np.int64)
        self.days2 = np.zeros(C_, dtype=bool)
        self.W = np.zeros((C_, C_))
        self.D1 = np.zeros((C_, C_))
        self.Qaa = np.zeros((C_, C_))
        self.Qab = np.zeros((C_, C_))
        self.Rmin = np.full((C_, C_), math.inf)
        self.Rmax = np.full((C_, C_), -math.inf)
        self.C_ever = 0
        self.checks = 0
        self.since_check = 0.0
        self.total_evidence = 0.0

    # --------------------------------------------------------- candidates
    def set_candidate(self, i: int, key: Hashable, card_hint: float = 2.0,
                      tmask: Optional[Sequence[bool]] = None) -> None:
        """(Re)start slot i with a new candidate: empty statistics, a new
        ordinal (C_ever + 1). tmask excludes targets derived from the same
        source field as the candidate (and the candidate's own hierarchy)."""
        if not 0 <= i < self.C:
            raise IndexError("candidate slot out of range")
        self._clear(i)
        self.keys[i] = key
        self.card[i] = max(1.0, float(card_hint))
        self.C_ever += 1
        self.ordinal[i] = self.C_ever
        if tmask is not None:
            m = np.asarray(tmask, dtype=bool)
            if m.shape != (self.T,):
                raise ValueError("tmask shape")
            self.tmask[i] = m

    def drop_candidate(self, i: int) -> None:
        self._clear(i)
        self.keys[i] = None

    def _clear(self, i: int) -> None:
        self.cnt[i] = 0.0
        self.den[i] = 0.0
        self.L1[i] = 0.0
        self.G[i] = self.n[i] = self.rows[i] = 0.0
        self.tmask[i] = True
        self.slot_of[i] = {}
        self.slot_val[i] = [None] * self.kv
        self.slot_ev[i] = 0.0
        self.slot_pri[i] = 0.0
        self.day0[i] = -1
        self.days2[i] = False
        for M in (self.W, self.D1, self.Qaa, self.Qab):
            M[i, :] = 0.0
            M[:, i] = 0.0
        self.Rmin[i, :] = self.Rmin[:, i] = math.inf
        self.Rmax[i, :] = self.Rmax[:, i] = -math.inf

    def restart(self) -> None:
        """R_learn restart: every current candidate restarts from empty with a
        new ordinal; C_ever keeps counting, so repeated attempts pay."""
        for i, k in enumerate(self.keys):
            if k is not None:
                card, tm = self.card[i], self.tmask[i].copy()
                self.set_candidate(i, k, card, tm)
        self.checks = 0
        self.since_check = 0.0
        self.total_evidence = 0.0

    def active(self) -> List[int]:
        return [i for i, k in enumerate(self.keys) if k is not None]

    def _slot(self, i: int, value: Hashable, omega: float) -> int:
        so = self.slot_of[i]
        j = so.get(value)
        if j is not None:
            return j
        sv = self.slot_val[i]
        for j in range(self.kv):
            if sv[j] is None:
                sv[j] = value
                so[value] = j
                return j
        # full: Space-Saving replacement of the lightest slot; its counts fold into other
        j = int(np.argmin(self.slot_pri[i]))
        old = sv[j]
        if old is not None:
            del so[old]
        o = self.kv
        self.cnt[i, o] += self.cnt[i, j]
        self.den[i, o] += self.den[i, j]
        self.slot_ev[i, o] += self.slot_ev[i, j]
        self.cnt[i, j] = 0.0
        self.den[i, j] = 0.0
        self.slot_ev[i, j] = 0.0
        sv[j] = value
        so[value] = j
        return j

    # ------------------------------------------------------------ update
    def update(self, values: Sequence[Optional[Hashable]], bins: Sequence[int], p_leaf: np.ndarray,
               omega: float, day: Optional[int] = None) -> np.ndarray:
        """Feed one learned event. values[i] = the event's generalised value of
        candidate i (use a sentinel for absence; ignored for empty slots);
        bins[t] in [0, k_b) or -1 when the target is absent; p_leaf[t, b] =
        the leaf's predictive before this event. Returns d (bits saved per
        candidate slot; 0 for inactive ones)."""
        w = _check_omega(omega)
        d_out = np.zeros(self.C)
        if w <= 0.0:
            return d_out
        act = self.active()
        if not act:
            return d_out
        b = np.asarray(bins, dtype=np.int64)
        tp = np.flatnonzero(b >= 0)
        self.total_evidence += w
        self.since_check += w
        a = np.asarray(act, dtype=np.int64)
        jj = np.asarray([self._slot(i, values[i], w) for i in act], dtype=np.int64)
        if day is not None:
            dd = int(day)
            for i in act:
                if self.day0[i] < 0:
                    self.day0[i] = dd
                elif dd != self.day0[i]:
                    self.days2[i] = True
        d = np.zeros(a.size)
        if tp.size:
            bt = b[tp]
            pl = np.maximum(np.asarray(p_leaf, dtype=np.float64)[tp, bt], 1e-12)
            # flat row index of (candidate, slot, target) into cnt / den
            rows = ((a * (self.kv + 1) + jj) * self.T)[:, None] + tp[None, :]     # [A, Tp]
            cells = rows * self.kb + bt[None, :]
            cf = self.cnt.reshape(-1)
            df = self.den.reshape(-1)
            num = cf[cells]
            den = df[rows]
            lc = -np.log2((num + self.alpha * pl) / (den + self.alpha))
            ll = -np.log2(pl)
            m = self.tmask[a[:, None], tp[None, :]]
            contrib = np.where(m, lc, 0.0)
            l1f = self.L1.reshape(-1)
            l1f[(a * self.T)[:, None] + tp[None, :]] += w * contrib
            d = w * (np.where(m, ll, 0.0) - contrib).sum(axis=1)
            cf[cells] += w
            df[rows] += w
        self.G[a] += d
        self.n[a] += w
        self.rows[a] += 1.0
        self.slot_ev[a, jj] += w
        self.slot_pri *= 2.0 ** (-w / PRI_HALF_UNITS)
        self.slot_pri[a, jj] += w
        d_out[a] = d
        # pairwise sums for rule (S) over events where both candidates are active
        if a.size == self.C:                               # every slot active: no masking
            dc = d_out[:, None]
            dr = d_out[None, :]
            self.W += w
            self.D1 += dc
            self.Qaa += dc * dc / w
            self.Qab += dc * dr / w
            x = (dc - dr) / w
            np.minimum(self.Rmin, x, out=self.Rmin)
            np.maximum(self.Rmax, x, out=self.Rmax)
            return d_out
        am = np.zeros(self.C)
        am[a] = 1.0
        both = am[:, None] * am[None, :]
        dc = d_out[:, None]
        self.W += w * both
        self.D1 += dc * am[None, :]
        self.Qaa += (dc * dc / w) * am[None, :]
        self.Qab += dc * d_out[None, :] / w
        x = (dc - d_out[None, :]) / w
        pair = both > 0
        np.minimum(self.Rmin, np.where(pair, x, math.inf), out=self.Rmin)
        np.maximum(self.Rmax, np.where(pair, x, -math.inf), out=self.Rmax)
        return d_out

    # ------------------------------------------------------------ reads
    def l0(self, i: int) -> np.ndarray:
        """Pooled ML code length per target of candidate i."""
        return pmdl.ml_code_length_rows(self.cnt[i].sum(axis=0))

    def log2_e(self, i: int) -> float:
        """log2 of the averaged e-value of candidate i (-inf when empty)."""
        pooled = self.cnt[i].sum(axis=0)
        used = self.tmask[i] & (pooled.sum(axis=-1) > 0)
        if not used.any():
            return _NEG
        x = pmdl.ml_code_length_rows(pooled) - self.L1[i]
        return pmdl.log2_mean_exp2(x[used])

    def per_target_log2_e(self, i: int) -> np.ndarray:
        return pmdl.ml_code_length_rows(self.cnt[i].sum(axis=0)) - self.L1[i]

    def value_groups(self, i: int, n_child_min: float = N_CHILD_MIN
                     ) -> Tuple[List[List[Hashable]], int, List[float]]:
        """Greedy KT merging of candidate i's slots (§6.5.5): the pair whose
        union increases the pooled KT code length least is merged while the
        increase is <= log2 card_hint; slots with < n_child_min evidence join
        `other`. Returns (groups of values, index of the group that contains
        `other` (-1 if none), evidence per group)."""
        m = self.tmask[i]
        J = self.kv + 1
        vals: List[List[Hashable]] = []
        tabs: List[np.ndarray] = []
        evs: List[float] = []
        other_tab = self.cnt[i, self.kv][m].copy()
        other_ev = float(self.slot_ev[i, self.kv])
        for j in range(self.kv):
            v = self.slot_val[i][j]
            if v is None:
                continue
            if self.slot_ev[i, j] < n_child_min:
                other_tab += self.cnt[i, j][m]
                other_ev += float(self.slot_ev[i, j])
                continue
            vals.append([v])
            tabs.append(self.cnt[i, j][m].copy())
            evs.append(float(self.slot_ev[i, j]))
        has_other = other_ev > 0
        if has_other:
            vals.append([OTHER])
            tabs.append(other_tab)
            evs.append(other_ev)
        limit = math.log2(max(1.0, float(self.card[i])))

        def kt(tab: np.ndarray) -> float:
            return sum(pmdl.kt_code_length(r) for r in tab)
        codes = [kt(t) for t in tabs]
        while len(vals) > 1:
            best = None
            for x in range(len(vals)):
                for y in range(x + 1, len(vals)):
                    dl = kt(tabs[x] + tabs[y]) - codes[x] - codes[y]
                    if best is None or dl < best[0]:
                        best = (dl, x, y)
            if best is None or best[0] > limit:
                break
            _, x, y = best
            vals[x] = vals[x] + vals[y]
            tabs[x] = tabs[x] + tabs[y]
            evs[x] += evs[y]
            codes[x] = kt(tabs[x])
            del vals[y], tabs[y], evs[y], codes[y]
        other_idx = -1
        groups: List[List[Hashable]] = []
        for g, vs in enumerate(vals):
            if OTHER in vs:
                other_idx = g
                vs = [v for v in vs if v != OTHER]
            groups.append(vs)
        return groups, other_idx, evs

    def check(self, tau0: float = TAU0, delta: float = DELTA, tau_tie: float = TAU_TIE,
              n_child_min: float = N_CHILD_MIN, diversity_units: float = DIVERSITY_UNITS,
              C_desc: Optional[int] = None, s_mode: str = "spec") -> SplitDecision:
        """Evaluate rules (V), (G), (S), (D), (M) for the best candidate by G.

        s_mode 'spec' applies (S) as written in §6.5.5. 'rival_valid' applies
        (S) only when the runner-up also passes (V) at the same threshold: the
        choice between candidates is in doubt only if both are significant.
        'margin' is 'rival_valid' plus, when (S) fails between two valid
        candidates, the MDL margin of §6.6 (revision / alternates): the best
        saves >= tau0 bits more than the runner-up on their common events.
        Measured (power case of tests/lib/test_pevalue.py, per-event range of
        d_c1 - d_c2 ~ 15 bits): with the three sources at 60 % of events (V)
        passes at 32 units, spec-(S) at 352, rival-valid at 64; at 30 % (V)
        at 96, spec-(S) at ~960. The range term 3 R ln(3/delta_k) / n
        dominates at small n; P04 decides which mode it uses."""
        self.checks += 1
        self.since_check = 0.0
        act = self.active()
        if not act:
            return SplitDecision(None, None)
        order = sorted(act, key=lambda i: (-self.G[i], self.ordinal[i]))
        c1 = order[0]
        c2 = order[1] if len(order) > 1 else None
        dec = SplitDecision(c1, c2)
        dec.log2_e = self.log2_e(c1)
        dec.threshold = float(tau0) + math.log2(max(1, self.C_ever))
        dec.pass_v = dec.log2_e >= dec.threshold
        groups, oidx, gev = self.value_groups(c1, n_child_min)
        dec.groups, dec.other_group, dec.group_evidence = groups, oidx, gev
        n_named = sum(1 for g, vs in enumerate(groups) if vs or g == oidx)
        dec.l_split = pmdl.split_description_length(C_desc or self.C, max(1, len(groups)),
                                                    self.card[c1])
        dec.gain = float(self.G[c1])
        dec.pass_g = dec.gain - dec.l_split >= 0.0
        dec.pass_m = sum(1 for e in gev if e >= n_child_min) >= 2 and n_named >= 2
        dec.pass_d = bool(self.days2[c1]) or self.n[c1] >= diversity_units
        rival_ok = c2 is not None and (s_mode == "spec"
                                       or self.log2_e(c2) >= dec.threshold)
        if not rival_ok:
            dec.eps, dec.mu_diff, dec.pass_s = 0.0, math.inf, True
        else:
            W = self.W[c1, c2]
            if W <= 0:
                dec.pass_s = False
            else:
                mu = (self.D1[c1, c2] - self.D1[c2, c1]) / W
                ex2 = (self.Qaa[c1, c2] - 2.0 * self.Qab[c1, c2] + self.Qaa[c2, c1]) / W
                var = max(0.0, ex2 - mu * mu)
                R = self.Rmax[c1, c2] - self.Rmin[c1, c2]
                R = R if math.isfinite(R) else 0.0
                dk = pmdl.time_uniform_delta(delta, self.checks)
                eps = pmdl.empirical_bernstein_bound(var, R, W, dk)
                dec.eps, dec.mu_diff = eps, mu
                dec.pass_s = mu >= eps or eps <= tau_tie
                if not dec.pass_s and s_mode == "margin":
                    # MDL model-selection margin between two valid splits, the
                    # rule §6.6 uses for revisions and alternates
                    dec.pass_s = (self.D1[c1, c2] - self.D1[c2, c1]) >= float(tau0)
        return dec

    def nbytes(self) -> int:
        arrs = (self.cnt, self.den, self.L1, self.G, self.n, self.rows, self.card, self.ordinal, self.tmask,
                self.slot_ev, self.slot_pri, self.W, self.D1, self.Qaa, self.Qab, self.Rmin, self.Rmax)
        return int(sum(a.nbytes for a in arrs) + 80 * self.C * self.kv + 400)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: getattr(self, k).copy() for k in (
            "cnt", "den", "L1", "G", "n", "rows", "card", "ordinal", "tmask", "slot_ev", "slot_pri", "day0",
            "days2", "W", "D1", "Qaa", "Qab", "Rmin", "Rmax")}
        d.update({"T": self.T, "kb": self.kb, "kv": self.kv, "C": self.C, "alpha": self.alpha,
                  "keys": list(self.keys), "slot_val": [list(s) for s in self.slot_val],
                  "C_ever": self.C_ever, "checks": self.checks, "since_check": self.since_check,
                  "total_evidence": self.total_evidence})
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SplitStats":
        s = cls(d["T"], d["kb"], d["kv"], d["C"], d["alpha"])
        for k in ("cnt", "den", "L1", "G", "n", "rows", "card", "ordinal", "tmask", "slot_ev", "slot_pri", "day0",
                  "days2", "W", "D1", "Qaa", "Qab", "Rmin", "Rmax"):
            setattr(s, k, np.asarray(d[k]).copy())
        s.keys = [tuple(k) if isinstance(k, list) else k for k in d["keys"]]
        s.slot_val = [list(v) for v in d["slot_val"]]
        s.slot_of = [{v: j for j, v in enumerate(sv) if v is not None} for sv in s.slot_val]
        s.C_ever, s.checks = int(d["C_ever"]), int(d["checks"])
        s.since_check, s.total_evidence = float(d["since_check"]), float(d["total_evidence"])
        return s


# ============================================================ exceptions
class _ExcRec:
    __slots__ = ("ordinal", "cN0", "LN0", "cx", "Lx_own", "Lx_N", "n", "day0", "days2")

    def __init__(self, ordinal: int, cN: np.ndarray, LN: np.ndarray) -> None:
        self.ordinal = ordinal
        self.cN0 = cN.copy()
        self.LN0 = LN.copy()
        self.cx = np.zeros_like(cN)
        self.Lx_own = np.zeros_like(LN)
        self.Lx_N = np.zeros_like(LN)
        self.n = 0.0
        self.day0 = -1
        self.days2 = False


class ExcTracker:
    """Leave-x-out exception e-values at a confirmed node (§6.7). For each
    tracked heavy source x (at most k_x), since x's record started:
        L1_x,t = sum_{e from x} omega (-log2 p_x,t)  +  (L_N,t - L_N,t^start - sum_{e from x} omega (-log2 p_N,t))
        L0_x,t = pooled ML code length of the node's counts since start
        log2 e_x = log2 mean_t 2^(L0 - L1)
    p_x backs off to the node predictive p_N with concentration alpha. Coding
    the other sources under N's predictive (which also learned from x) only
    lengthens L1, so the test is conservative."""

    def __init__(self, n_targets: int, k_b: int = K_B, k_x: int = 16, alpha: float = ALPHA) -> None:
        self.T, self.kb, self.k_x, self.alpha = int(n_targets), int(k_b), int(k_x), float(alpha)
        self.cN = np.zeros((self.T, self.kb))
        self.LN = np.zeros(self.T)
        self.recs: Dict[Hashable, _ExcRec] = {}
        self.n_tested = 0

    def track(self, src: Hashable) -> bool:
        if src in self.recs:
            return True
        if len(self.recs) >= self.k_x:
            return False
        self.n_tested += 1
        self.recs[src] = _ExcRec(self.n_tested, self.cN, self.LN)
        return True

    def untrack(self, src: Hashable) -> None:
        self.recs.pop(src, None)

    def update(self, src: Hashable, bins: Sequence[int], p_node: np.ndarray, omega: float,
               day: Optional[int] = None) -> None:
        w = _check_omega(omega)
        if w <= 0.0:
            return
        b = np.asarray(bins, dtype=np.int64)
        tp = np.nonzero(b >= 0)[0]
        if tp.size == 0:
            return
        bt = b[tp]
        pN = np.clip(np.asarray(p_node, dtype=np.float64)[tp, bt], 1e-12, 1.0)
        lN = -np.log2(pN)
        self.LN[tp] += w * lN
        r = self.recs.get(src)
        if r is not None:
            cx = r.cx[tp]
            px = (cx[np.arange(tp.size), bt] + self.alpha * pN) / (cx.sum(axis=1) + self.alpha)
            r.Lx_own[tp] += w * -np.log2(px)
            r.Lx_N[tp] += w * lN
            r.cx[tp, bt] += w
            r.n += w
            if day is not None:
                if r.day0 < 0:
                    r.day0 = int(day)
                elif int(day) != r.day0:
                    r.days2 = True
        self.cN[tp, bt] += w

    def log2_e(self, src: Hashable) -> float:
        r = self.recs.get(src)
        if r is None:
            return _NEG
        since = self.cN - r.cN0
        used = r.cx.sum(axis=1) > 0
        if not used.any():
            return _NEG
        L0 = pmdl.ml_code_length_rows(since)
        L1 = r.Lx_own + (self.LN - r.LN0 - r.Lx_N)
        return pmdl.log2_mean_exp2((L0 - L1)[used])

    def per_target_saving(self, src: Hashable) -> np.ndarray:
        """Bits saved per target by coding x under its own predictive."""
        r = self.recs.get(src)
        if r is None:
            return np.zeros(self.T)
        return r.Lx_N - r.Lx_own

    def decide(self, src: Hashable, n_conf: float, tau0: float = TAU0, n_min: float = 10.0) -> bool:
        """Exception rule: log2 e_x >= tau0 + log2(i_x), x's evidence on the
        confidence channel >= n_min (caller supplies n_conf), >= 2 days."""
        r = self.recs.get(src)
        if r is None:
            return False
        return (self.log2_e(src) >= tau0 + math.log2(max(1, r.ordinal))
                and float(n_conf) >= n_min and r.days2)

    def nbytes(self) -> int:
        per = 2 * self.T * self.kb * 8 + 3 * self.T * 8 + 64
        return int(self.cN.nbytes + self.LN.nbytes + len(self.recs) * per + 200)
