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

Rule (V), anytime-valid (Ville's inequality):
    e_c = (1/T_c) sum_t e_{c,t},  split allowed only when log2 e_c >= tau0 + log2 C_ever,
    e_{c,t} = the blockwise k-sample e-process of close_block (a block ends at
    every split check): per block, the slots' predictors are frozen at its
    start and the block's evidence-weighted outcomes are coded under their
    slot's predictor against pbar = the evidence-weighted mixture of those
    predictors (the RIPr of the alternative onto "no dependence").
Under the null (t independent of the candidate's value, evidence units
conditionally independent) each block's factor has expectation <= 1 for every
null distribution (proof in close_block), an average of e-processes is an
e-process whatever the dependence between targets, and P(sup e >= 2^tau) <=
2^-tau however often it is checked. C_ever (candidates ever tracked, never
decreasing) charges the multiplicity per candidate.

Integration change (2026-09-30): the text's universal-inference e-value
e = 2^(L0 - L1) with L0 = the POOLED maximum-likelihood code length is also
valid, but every candidate first repays the null model's parametric regret
((k_b - 1)/2 log2 n bits per target) before a real dependence counts; on pack
O's shared OA login node the /24 split that separates 综合部 / 财务部 / 销售部
had log2 e = 5 after 101 evidence units (the usernames alone save ~45 bits).
It is kept as log2_e_ui() for diagnosis.

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
        self.Gt = np.zeros((C_, self.T))                     # per-target saving (selective gain)
        # blockwise k-sample e-process (rule V, see _close_block): log2 e per (candidate, target),
        # the open block's evidence-weighted counts and the slot predictors frozen at its start
        self.E = np.zeros((C_, self.T))
        self.bm = np.zeros((C_, J, self.T, self.kb), dtype=np.float32)
        self.S = np.zeros((C_, J, self.T))                   # bits slot j's own predictor saved vs the leaf's
        # e-process wealth per (candidate, target) = 2^(wlog + E); cap = stopped wealth
        # of targets dropped without a successor (retarget); log2_e = log2 of the sum
        self.wlog = np.full((C_, self.T), np.nan)
        self.cap = np.zeros(C_)
        self.blam = np.zeros((C_, J, self.T))                # mixture weights frozen at the block start
        self.blk_open = False
        self.blk_day: Optional[int] = None
        self.blk_pleaf: Optional[np.ndarray] = None
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
        self._gt()[i] = 0.0
        self._blk()
        self.E[i] = 0.0
        self.bm[i] = 0.0                                   # started mid-block: its slots start empty
        self.S[i] = 0.0
        self.blam[i] = 0.0
        self._wl()[i] = np.nan                             # weights set at the candidate's first read
        self.cap[i] = 0.0
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
        if self._blk():
            self.S[i, j] = 0.0                             # a new value starts like the leaf
        if self.blk_open:
            # the evicted value's block events are evaluated under `other`'s frozen
            # predictor, the new value's under the slot's (the event -> predictor map
            # depends on candidate values only, never on target outcomes)
            self.bm[i, o] += self.bm[i, j]
            self.bm[i, j] = 0.0
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
        self._blk()                                         # lazily adds the block state
        if self.blk_open and day is not None and getattr(self, "blk_day", None) is not None \
                and int(day) != self.blk_day:
            self.close_block()                              # a block is one local day
        if not self.blk_open:
            self._open_block(np.asarray(p_leaf, dtype=np.float64))
            self.blk_day = None if day is None else int(day)
        if tp.size:
            bt = b[tp]
            bmf = self.bm.reshape(-1)
            bmf[((((a * (self.kv + 1) + jj) * self.T)[:, None] + tp[None, :]) * self.kb + bt[None, :]).ravel()] += w
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
            dt_ = w * (np.where(m, ll, 0.0) - contrib)
            gtf = self._gt().reshape(-1)
            gtf[(a * self.T)[:, None] + tp[None, :]] += dt_
            sf = self.S.reshape(-1)
            sf[(((a * (self.kv + 1) + jj) * self.T)[:, None] + tp[None, :]).ravel()] += dt_.ravel()
            d = dt_.sum(axis=1)
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

    # ----------------------------------------------- k-sample e-process
    def _blk(self) -> bool:
        """Lazily add the block state to objects built before it existed."""
        if getattr(self, "E", None) is None or getattr(self.E, "shape", None) != (self.C, self.T):
            J = self.kv + 1
            self.E = np.zeros((self.C, self.T))
            self.bm = np.zeros((self.C, J, self.T, self.kb), dtype=np.float32)
            self.S = np.zeros((self.C, J, self.T))
            self.blam = np.zeros((self.C, J, self.T))
            self.blk_open = False
            self.blk_pleaf = None
            return False
        return True

    def _open_block(self, p_leaf: np.ndarray) -> None:
        """Freeze every slot's predictor for the new block: the smoothed slot
        predictive (n[c, j, t, b] + alpha p_leaf(b)) / (n[c, j, t, .] + alpha)
        from the counts BEFORE the block and the leaf predictive of the block's
        first event (both functions of the past only)."""
        pl = np.maximum(np.asarray(p_leaf, dtype=np.float64).reshape(self.T, self.kb), 1e-12)
        pl = pl / pl.sum(axis=1, keepdims=True)
        self.blk_pleaf = pl
        self.bm[:] = 0.0
        # Bayes mixture of "slot j has its own distribution" and "slot j is like
        # the leaf" (prior 1/2 each, their past prequential likelihood ratio 2^S):
        # a slot whose own predictor has not saved bits so far is coded like the
        # leaf, so estimation noise in the many slots that do NOT differ costs
        # (almost) nothing (measured: 2 undifferentiated slots of 26 events a day
        # drew the e-process down by ~18 bits in 10 days without it)
        self.blam = 1.0 / (1.0 + np.exp2(-np.clip(self.S, -60.0, 60.0)))
        self.blk_open = True

    def roll(self, day: Optional[int]) -> None:
        """Close the open block when it belongs to an earlier local day than
        `day` (or when blocks are not dated: unit use without days)."""
        if not self._blk() or not self.blk_open:
            return
        bd = getattr(self, "blk_day", None)
        if bd is None or day is None or int(day) > bd:
            self.close_block()

    def close_block(self) -> None:
        """Close the open block: for every active candidate and target add

            log2 e_blk = sum_{j,b} m[j, b] log2( P_j(b) / pbar(b) ),
            pbar = sum_j (w_j / W) P_j,   w_j = sum_b m[j, b],  W = sum_j w_j,

        m = the block's evidence-weighted counts (omega <= 1), P_j = slot j's
        predictor frozen at the block start. A block is one local day (P04
        passes the day; undated use closes at every check): within a day the
        order of events is not exchangeable (who comes at 09:00, who at 15:00),
        and a sub-day block would test the dependence GIVEN the time of day,
        which has no power where departments differ by their hours (measured
        on the revision test (f): /24 slots of two time-separated groups drove
        the per-check e-process down while the daily one rises). Under the null (the target's
        outcomes independent of the candidate's slots, independent across
        evidence units, one distribution theta within the block) E[e_blk] <= 1
        for EVERY theta: by independence and Jensen (x^omega concave),
        E prod_i (P_ji(X)/pbar(X))^omega_i <= prod_i A_i^omega_i with
        A_i = sum_x theta(x) P_ji(x) / pbar(x), and by concavity of log
        sum_i omega_i log A_i <= W log sum_i (omega_i / W) A_i = W log 1 = 0.
        The product over blocks is therefore a test supermartingale (Ville:
        P(sup e >= 2^tau) <= 2^-tau), like the universal-inference e-value it
        replaces, but its denominator is the RIPr of the split predictors onto
        the null (Turner, Ly & Grunwald's k-sample construction), not the
        pooled maximum-likelihood code: a correct split does not first have to
        repay the null model's parametric regret ((k-1)/2 log2 n bits per
        target), which is what kept a department of 3 IPs from ever splitting
        a shared login node (measured on pack O, see _check_valid_first)."""
        if not self._blk() or not self.blk_open:
            return
        act = self.active()
        if act:
            a = np.asarray(act, dtype=np.int64)
            m = self.bm[a].astype(np.float64)                       # [A, J, T, K]
            w = m.sum(axis=3)                                       # [A, J, T]
            W = w.sum(axis=1)                                       # [A, T]
            # the slots' predictors frozen at the block start: counts before the block
            # (the running counts minus the block's own) backing off to the leaf
            # predictive of the block's first event
            before = np.maximum(self.cnt[a] - m, 0.0)
            pl = self.blk_pleaf if self.blk_pleaf is not None else np.full((self.T, self.kb), 1.0 / self.kb)
            Q = (before + self.alpha * pl[None, None, :, :]) / (before.sum(axis=3, keepdims=True) + self.alpha)
            lam = self.blam[a][..., None]
            P = np.maximum(lam * Q + (1.0 - lam) * pl[None, None, :, :], 1e-300)
            with np.errstate(invalid="ignore", divide="ignore"):
                share = np.where(W[:, None, :] > 0, w / np.maximum(W[:, None, :], 1e-300), 0.0)
            pbar = (share[..., None] * P).sum(axis=1)               # [A, T, K]
            lr = np.log2(P) - np.log2(np.maximum(pbar, 1e-300))[:, None, :, :]
            self.E[a] += np.where(m > 0, m * lr, 0.0).sum(axis=(1, 3))
        self.bm[:] = 0.0
        self.blk_open = False

    def _wl(self) -> np.ndarray:
        w = getattr(self, "wlog", None)
        if w is None or w.shape != (self.C, self.T):
            self.wlog = np.full((self.C, self.T), np.nan)
            self.cap = np.zeros(self.C)
        return self.wlog

    def _weights(self, i: int) -> np.ndarray:
        """log2 weights of candidate i's targets: 1 / (number of tested targets)
        each, fixed at the first read (the plain average of §6.5.5), then moved
        only by retarget (self-financing)."""
        wl = self._wl()
        if np.all(np.isnan(wl[i])):
            m = self.tmask[i]
            k = int(m.sum())
            wl[i] = np.where(m, -math.log2(max(1, k)), -np.inf)
        return wl[i]

    def log2_e(self, i: int) -> float:
        """log2 of the averaged e-value of candidate i (-inf when empty): the
        blockwise k-sample e-process of close_block, averaged over the
        candidate's tested targets (an average of e-processes is one), plus the
        stopped wealth of targets retarget() dropped."""
        self._blk()
        used = self.tmask[i] & (self.cnt[i].sum(axis=(0, 2)) > 0)
        if not used.any() and self.cap[i] <= 0:
            return _NEG
        w = self._weights(i)
        x = (w + self.E[i])[self.tmask[i] & np.isfinite(w)]
        parts = list(x)
        if self.cap[i] > 0:
            parts.append(math.log2(self.cap[i]))
        if not parts:
            return _NEG
        a = np.asarray(parts, dtype=np.float64)
        m = float(a.max())
        return m + math.log2(float(np.exp2(a - m).sum()))

    def retarget(self, keep: Sequence[Optional[int]], tmask_new: np.ndarray) -> None:
        """Change the target set without discarding evidence. keep[t'] = the old
        index of new target t' (None for a new target); tmask_new [C, T'] the
        candidates' target masks. Kept targets keep every statistic; a dropped
        target's e-process wealth (weight x e) is moved to the new targets
        (equal shares; into `cap`, a stopped e-process, when nothing is added),
        and new targets start at e = 1 with that wealth. Moving wealth between
        e-processes at a predictable time keeps the total a nonnegative
        supermartingale (a self-financing portfolio of bets), so rule (V) stays
        anytime-valid. Measured on pack O: P05's node target lists change every
        few days as a node's probe rows change, and each change restarted the
        split statistics of the shared OA login node (n = 49 evidence units at
        day 10, one day block)."""
        self._blk()
        T2 = len(keep)
        C_, J = self.C, self.kv + 1
        old_T = self.T
        wl_old = np.array([self._weights(i) if self.keys[i] is not None else self._wl()[i]
                           for i in range(C_)])
        def remap(a: np.ndarray, axis: int, fill: float = 0.0) -> np.ndarray:
            shape = list(a.shape)
            shape[axis] = T2
            out = np.full(shape, fill, dtype=a.dtype)
            for t2, t1 in enumerate(keep):
                if t1 is not None and 0 <= t1 < old_T:
                    idx_o = [slice(None)] * a.ndim
                    idx_n = [slice(None)] * a.ndim
                    idx_o[axis], idx_n[axis] = t1, t2
                    out[tuple(idx_n)] = a[tuple(idx_o)]
            return out
        E_old = self.E.copy()
        self.cnt = remap(self.cnt, 2)
        self.den = remap(self.den, 2)
        self.L1 = remap(self.L1, 1)
        self.Gt = remap(self._gt(), 1)
        self.E = remap(self.E, 1)
        self.bm = remap(self.bm, 2)
        self.S = remap(self.S, 2)
        self.blam = remap(self.blam, 2)
        if self.blk_pleaf is not None:
            pl = np.full((T2, self.kb), 1.0 / self.kb)
            for t2, t1 in enumerate(keep):
                if t1 is not None and 0 <= t1 < old_T:
                    pl[t2] = self.blk_pleaf[t1]
            self.blk_pleaf = pl
        tm_new = np.asarray(tmask_new, dtype=bool).reshape(C_, T2)
        wl_new = np.full((C_, T2), -np.inf)
        kept_old = {t1 for t1 in keep if t1 is not None}
        for i in range(C_):
            if self.keys[i] is None:
                continue
            wealth_drop = 0.0
            for t1 in range(old_T):
                if t1 not in kept_old and self.tmask[i, t1] and np.isfinite(wl_old[i, t1]):
                    wealth_drop += 2.0 ** (wl_old[i, t1] + float(E_old[i, t1]))
            added = [t2 for t2, t1 in enumerate(keep) if tm_new[i, t2] and
                     (t1 is None or not self.tmask[i, t1] or not np.isfinite(wl_old[i, t1]))]
            for t2, t1 in enumerate(keep):
                if t1 is not None and tm_new[i, t2] and self.tmask[i, t1] and np.isfinite(wl_old[i, t1]):
                    wl_new[i, t2] = wl_old[i, t1]
                elif t1 is not None and self.tmask[i, t1] and not tm_new[i, t2] and np.isfinite(wl_old[i, t1]):
                    wealth_drop += 2.0 ** (wl_old[i, t1] + float(E_old[i, t1]))
            fund = wealth_drop + (self.cap[i] if added else 0.0)
            if added:
                self.cap[i] = 0.0
                share = fund / len(added)
                for t2 in added:
                    wl_new[i, t2] = math.log2(share) if share > 0 else -np.inf
                    self.E[i, t2] = 0.0
                    self.cnt[i, :, t2] = 0.0
                    self.den[i, :, t2] = 0.0
                    self.L1[i, t2] = 0.0
                    self.Gt[i, t2] = 0.0
                    self.bm[i, :, t2] = 0.0
                    self.S[i, :, t2] = 0.0
                    self.blam[i, :, t2] = 0.0
            else:
                self.cap[i] += wealth_drop
        self.T = T2
        self.tmask = tm_new
        self.wlog = wl_new
        self.G = self.Gt.sum(axis=1)

    def log2_e_ui(self, i: int) -> float:
        """The universal-inference e-value of §6.5.5 as written (pooled ML
        denominator), kept for comparison and diagnosis."""
        pooled = self.cnt[i].sum(axis=0)
        used = self.tmask[i] & (pooled.sum(axis=-1) > 0)
        if not used.any():
            return _NEG
        x = pmdl.ml_code_length_rows(pooled) - self.L1[i]
        return pmdl.log2_mean_exp2(x[used])

    def _gt(self) -> np.ndarray:
        g = getattr(self, "Gt", None)
        if g is None or g.shape != (self.C, self.T):
            g = self.Gt = np.zeros((self.C, self.T))
        return g

    def selective_gain(self, i: int) -> float:
        """MDL gain of candidate i when its children specialise only the
        targets the split saves bits on and inherit the others from the leaf
        (the §6.7 exception-child rule applied to splits): sum over tested
        targets of max(0, per-target saving) minus one bit per tested target
        (naming the specialised subset). The plain G sums every target, so on
        a node with many high-entropy targets the split's early prequential
        regret on the targets it does not predict (a password, a duration)
        cancels what it saves on the ones it does. Measured on pack O's OA
        login route node (day 7): /24 had log2 e = 67.6 (user agent, client
        stack, login minute) but G = 22 < L_split, and never split."""
        g = self._gt()[i][self.tmask[i]]
        return float(np.maximum(g, 0.0).sum() - g.size)

    def per_target_log2_e(self, i: int) -> np.ndarray:
        self._blk()
        return self.E[i].copy()

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
        if getattr(self, "blk_day", None) is None:
            self.close_block()                              # undated use: a block per check
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
        dec.gain = self.selective_gain(c1)
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
        self._blk()
        arrs = (self.cnt, self.den, self.L1, self._gt(), self.E, self.bm, self.G, self.n, self.rows, self.card, self.ordinal, self.tmask,
                self.slot_ev, self.slot_pri, self.W, self.D1, self.Qaa, self.Qab, self.Rmin, self.Rmax)
        return int(sum(a.nbytes for a in arrs) + 80 * self.C * self.kv + 400)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: getattr(self, k).copy() for k in (
            "cnt", "den", "L1", "G", "n", "rows", "card", "ordinal", "tmask", "slot_ev", "slot_pri", "day0",
            "days2", "W", "D1", "Qaa", "Qab", "Rmin", "Rmax")}
        d["Gt"] = self._gt().copy()
        self._blk()
        d["E"] = self.E.copy()
        d["wlog"] = self._wl().copy()
        d["cap"] = self.cap.copy()
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
        if "Gt" in d:
            s.Gt = np.asarray(d["Gt"]).copy()
        if "E" in d:
            s.E = np.asarray(d["E"]).copy()
        if "wlog" in d:
            s.wlog = np.asarray(d["wlog"]).copy()
            s.cap = np.asarray(d["cap"]).copy()
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
