"""TimingEngine (B11): a fine-grained inter-arrival profile per IP.

Why: the timing of requests separates humans from scripts and one person from
another even when content is encrypted. People produce heavy-tailed, bursty
gaps with a characteristic think time; scrapers produce a narrow gap
distribution; implants and cron jobs produce strictly periodic trains. v1 had
no timing model at all (think_time was a tick-index artefact), so a 0.8-s
scraper looked like a busy user and identification had no timing modality.

What, per entity and tick (all times are wall-clock seconds, so nothing here
depends on the 60 / 900 / 3600-s cadence):
  1. Gaps come from act.stream timestamps (lib/m_template), and only TRUE gaps
     are used, so sampling never distorts them: consecutive rows of a complete
     tick, plus the gap from the previous tick's last event when both ticks
     were complete (stream_frac = 1; silence in between is data). In a sampled
     tick R2 kept whole sessions, so only gaps <= 30 s (inside a kept block)
     are known and the boundary gap is unknown; its gaps weigh 1/stream_frac.
  2. The tick's gaps are summarised in one row (32-bin log2 histogram from
     10 ms to 1 day, plus the moments of within-session gaps, of their logs
     and of consecutive gap pairs). Rows are kept 8 d in the model: act.stream
     is retained 1 h, but GatedLearner commits a row D ticks later and may
     release or replay it up to 8 d later, so its `fetch` must still find it.
  3. Score against the model as of the last commit: the recent 6-h window
     (running sums of the rows in (now - 6 h, now], recomputed exactly every
     256 ticks) against the committed 7-d decayed histogram. score.timing =
     JSD in bits (B24 turns it into a conformal p). behavior.pm = the
     two-sample G test (G = 2 N JSD_pi) scaled by the entity's learned
     overdispersion phi (bursty, correlated gaps are not multinomial) against
     chi2_df. Accumulator on axis temporal: acc_alarm when pm <= budget ·
     dt / 86400, a union bound over the day's ticks, so the timing budget
     (0.005 alarms per entity-day, lib/detectors) holds at any cadence.
  4. Descriptors of the recent window go to behavior.timing (B15 / B16 / B30):
     burstiness B = (sd - mean)/(sd + mean), memory M = corr(gap_i, gap_i+1)
     and the log-normal think time (mu, sigma of ln gap), all over
     within-session gaps (< model.seq session_gap, default 30 min), because a
     single overnight gap would dominate every raw second moment.
  5. Strict periodicity: an rfft of 5-s binned counts over the last 2 h picks
     a candidate (harmonic sum, so the fundamental beats its harmonics), which
     is confirmed by the window-corrected Rayleigh statistic Z^2_1 at the
     refined period with the Davies trials bound over the scanned band
     (lib/evt). Only strict trains keep phase coherence over 2 h;
     renewal-jittered beacons are B12's job. At most `period_max_per_tick`
     checks per tick, stalest first, so the cost stays bounded.
  6. Learning is trust-gated, delayed, checkpointed and reversible through
     lib/gating.GatedLearner (model.control rollback_to / release /
     rebase_from / frozen honoured; link seeding merges a linked entity's
     statistics at weight 0.5). Training mode learns with trust 1 and raises
     no alarm. No event kinds are emitted (contract F has none for timing).
The pure read side (loglik for identity candidates, quantiles, descriptors,
state layout) lives in lib/m_timing.py.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy import special

from ...core.engine import Context, Engine
from ...models.schema import DerivedMetric, EntityProfile, MetricKind
from .lib import detectors as DET
from .lib import emit
from .lib import evt
from .lib import gating as G
from .lib import m_template as MT
from .lib import m_timing as TM
from .lib import timebins as TB

DETECTOR = "timing"
AXES = ["temporal"]
LEARNER = "timing"
SERIES = "behavior.timing"
SEQ_MODEL = "model.seq"
R2_ENGINE = "raw.action_token"

RECENT_S = 6 * 3600.0             # scored window
RECENT_MIN_N = 16.0               # effective gaps in the window to score
LONG_MIN_N = 64.0                 # effective committed gaps to score against
DESC_MIN_N = 8                    # raw gaps / pairs for a window descriptor
ROW_W_MAX = 64.0                  # cap on 1/stream_frac
ROW_KEEP_S = G.JOURNAL_MAX_AGE_S + 3600.0   # rows must outlive journal and held rows
WIN_RECOMPUTE_EVERY = 256         # exact recompute of the running window sums
SESSION_GAP_DEFAULT_S = 1800.0
SESSION_GAP_CLIP = (120.0, 7200.0)          # B10's clip of G_e
DISP_RATIO_CAP = 25.0             # one anomalous window cannot inflate phi much
P_FLOOR = 1e-300
PROFILE_EVERY_S = 3600.0          # portraits (B30) refresh every 2 h
AXES_PM_MAX = 0.05                # axes are written when the detector contributes

PERIOD_WINDOW_S = 7200.0          # rfft over the last 2 h
PERIOD_BIN_S = 5.0
PERIOD_MIN_S = 10.0               # Nyquist of the 5-s bins
PERIOD_MIN_EVENTS = 12
PERIOD_HARMONICS = 4
PERIOD_CANDIDATES = 8             # harmonic sums ranked exactly
PERIOD_EVERY_S = 600.0            # per entity, when new events arrived
PERIOD_MAX_PER_TICK = 4
PERIOD_P_CONFIRM = 1e-3           # publish the period only below this p
EV_CAP = 1024                     # event times kept for the check

# row statistics (float32; the histogram is a separate uint16[32] array)
S_W, S_NG, S_GM, S_GM2, S_LM, S_LM2 = 0, 1, 2, 3, 4, 5
S_NP, S_PX, S_PY, S_PXX, S_PYY, S_PXY = 6, 7, 8, 9, 10, 11
S_GR, S_DP = 12, 13
N_STATS = 14

# running window sums (float64): [0:32] sum w*counts, then
V_W, V_W2 = 32, 33                # sum w*n, sum w^2*n (per-gap weights)
V_S0, V_S1, V_S2 = 34, 35, 36     # within-session gaps: sum w*ng, *mean, *(m2 + ng*mean^2)
V_L1, V_L2 = 37, 38               # same for ln gaps (weight V_S0)
V_P0, V_PX, V_PY, V_PXX, V_PYY, V_PXY = 39, 40, 41, 42, 43, 44
V_NG, V_NP, V_ROWS = 45, 46, 47   # raw gap / pair / row counts
V_DIM = 48

_OK, _SILENT, _UNKNOWN, _DEGRADED = "ok", "silent", "unknown", "degraded"
_NAN = math.nan
_L2_MIN = math.log2(TM.GAP_MIN_S)
_BUDGET_PER_DAY = float(DET.DETECTOR_INFO[DETECTOR]["budget_per_day"])   # type: ignore[arg-type]
_NO_DESC = {"B": _NAN, "M": _NAN, "think_mu": _NAN, "think_sigma": _NAN}


# ================================================================ row buffer
class _Rows:
    """Per-tick gap summaries in ascending ts, kept ROW_KEEP_S: the rows the
    gated learner folds at commit (t - D), release / rebase (held rows) and
    rollback replay. Grows by doubling and compacts once the pruned head is
    at least half the capacity, so appends are amortised O(1). ~130 B/row."""

    __slots__ = ("ts", "counts", "stats", "h", "n")

    def __init__(self, cap: int = 16) -> None:
        self.ts = np.empty(cap, dtype=np.float64)
        self.counts = np.zeros((cap, TM.N_BINS), dtype=np.uint16)
        self.stats = np.zeros((cap, N_STATS), dtype=np.float32)
        self.h = 0
        self.n = 0

    def __len__(self) -> int:
        return self.n - self.h

    def append(self, ts: float, counts: np.ndarray, stats: np.ndarray) -> bool:
        """Append a row; returns False when it replaced the newest row (an
        idempotent re-run of the same tick)."""
        n = self.n
        if n > self.h:
            last = float(self.ts[n - 1])
            if ts == last:
                self.counts[n - 1] = counts
                self.stats[n - 1] = stats
                return False
            if ts < last:
                raise ValueError(f"timing rows: ts {ts} older than newest row {last}")
        if n == self.ts.shape[0]:
            self._make_room()
            n = self.n
        self.ts[n] = ts
        self.counts[n] = counts
        self.stats[n] = stats
        self.n = n + 1
        return True

    def _make_room(self) -> None:
        cap = self.ts.shape[0]
        k = self.n - self.h
        new_cap = cap if self.h >= cap // 2 else 2 * cap
        ts = np.empty(new_cap, dtype=np.float64)
        counts = np.zeros((new_cap, TM.N_BINS), dtype=np.uint16)
        stats = np.zeros((new_cap, N_STATS), dtype=np.float32)
        ts[:k] = self.ts[self.h:self.n]
        counts[:k] = self.counts[self.h:self.n]
        stats[:k] = self.stats[self.h:self.n]
        self.ts, self.counts, self.stats = ts, counts, stats
        self.h, self.n = 0, k

    def find(self, ts: float) -> int:
        """Index of the row at ts (within 1e-6 s), else -1."""
        h, n = self.h, self.n
        if n <= h:
            return -1
        i = h + int(np.searchsorted(self.ts[h:n], ts - 1e-6, side="left"))
        return i if i < n and abs(float(self.ts[i]) - ts) <= 1e-6 else -1

    def window(self, t0: float, t1: float) -> Tuple[int, int]:
        """Index range [i0, i1) of the rows with t0 < ts <= t1."""
        h, n = self.h, self.n
        v = self.ts[h:n]
        return (h + int(np.searchsorted(v, t0, side="right")),
                h + int(np.searchsorted(v, t1, side="right")))

    def last_ts(self) -> float:
        return float(self.ts[self.n - 1]) if self.n > self.h else -math.inf

    def set_last(self, col: int, value: float) -> None:
        if self.n > self.h:
            self.stats[self.n - 1, col] = value

    def prune(self, cutoff: float) -> None:
        h, n = self.h, self.n
        if n > h and self.ts[h] < cutoff:
            self.h = h + int(np.searchsorted(self.ts[h:n], cutoff, side="left"))

    def nbytes(self) -> int:
        return int(self.ts.nbytes + self.counts.nbytes + self.stats.nbytes)


def new_model() -> Dict[str, Any]:
    return {
        "fmt": TM.FMT,
        "version": 0,
        "state": TM.new_state(),
        "gate": G.GateState(),
        "rows": _Rows(),
        "live": {"last_ev": _NAN, "last_full": False, "last_gap": _NAN,
                 "ev": np.zeros(0, dtype=np.float64), "ev_new": 0,
                 "period": None, "period_eval": -math.inf, "recent": {},
                 "win": np.zeros(V_DIM), "win_lo": -math.inf, "win_hi": -math.inf,
                 "win_rev": 0, "win_age": 0, "score_key": None, "score": None},
    }


# ============================================================== tick summary
def _bins(g: np.ndarray) -> np.ndarray:
    """bin_index for gaps already floored at GAP_MIN_S (the hot path)."""
    return np.minimum(((np.log2(g) - _L2_MIN) / TM.BIN_W).astype(np.intp), TM.N_BINS - 1)


def build_row(t: np.ndarray, frac: float, live: Dict[str, Any],
              session_gap: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Summarise one tick's event times (sorted, finite) into (counts, stats),
    or None when the tick yields no true gap. Updates live's boundary state
    (last event, whether that tick was complete, last within-session gap)."""
    sampled = not frac >= 1.0 - 1e-9
    rw = min(ROW_W_MAX, 1.0 / frac) if sampled else 1.0
    d = np.diff(t)
    last_ev = live.get("last_ev", _NAN)
    bnd = (not sampled) and bool(live.get("last_full")) and last_ev == last_ev \
        and float(t[0]) >= last_ev
    if bnd:
        seq = np.empty(d.size + 1, dtype=np.float64)
        seq[0] = float(t[0]) - last_ev
        seq[1:] = d
    else:
        seq = d
    np.maximum(seq, TM.GAP_MIN_S, out=seq)
    if sampled:
        ok = seq <= MT.SESSION_GAP_S      # inside a kept session block
        within = ok & (seq < session_gap)
    else:
        ok = None
        within = seq < session_gap
    # pairs of consecutive within-session gaps (the boundary pair continues
    # the previous tick's chain)
    if seq.size >= 2:
        pm = within[:-1] & within[1:]
        x_p, y_p = seq[:-1][pm], seq[1:][pm]
    else:
        x_p = y_p = seq[:0]
    lg = live.get("last_gap", _NAN)
    if bnd and within[0] and lg == lg:
        x_p = np.concatenate(([lg], x_p))
        y_p = np.concatenate(([seq[0]], y_p))
    # boundary state for the next tick
    live["last_ev"] = float(t[-1])
    live["last_full"] = not sampled
    live["last_gap"] = float(seq[-1]) if (not sampled and seq.size and within[-1]) else _NAN
    g_ok = seq if ok is None else seq[ok]
    if g_ok.size == 0:
        return None
    counts = np.bincount(_bins(g_ok), minlength=TM.N_BINS)
    st = np.zeros(N_STATS, dtype=np.float32)
    st[S_W] = rw
    st[S_GR] = _NAN
    st[S_DP] = _NAN
    x = seq[within]
    if x.size:
        gm = float(x.mean())
        lx = np.log(x)
        lm = float(lx.mean())
        st[S_NG] = x.size
        st[S_GM] = gm
        st[S_GM2] = float(np.dot(x - gm, x - gm))
        st[S_LM] = lm
        st[S_LM2] = float(np.dot(lx - lm, lx - lm))
    if x_p.size:
        mx, my = float(x_p.mean()), float(y_p.mean())
        dx, dy = x_p - mx, y_p - my
        st[S_NP] = x_p.size
        st[S_PX] = mx
        st[S_PY] = my
        st[S_PXX] = float(np.dot(dx, dx))
        st[S_PYY] = float(np.dot(dy, dy))
        st[S_PXY] = float(np.dot(dx, dy))
    return np.minimum(counts, 65535).astype(np.uint16), st


# ============================================================ window sums
def _contrib_rows(counts: np.ndarray, stats: np.ndarray) -> np.ndarray:
    """Sum over rows of each row's additive window contribution (V_DIM)."""
    S = stats.astype(np.float64)
    w = S[:, S_W]
    ng, gm, lm = S[:, S_NG], S[:, S_GM], S[:, S_LM]
    npr, px, py = S[:, S_NP], S[:, S_PX], S[:, S_PY]
    ntot = counts.sum(axis=1, dtype=np.float64)
    v = np.empty(V_DIM)
    v[:TM.N_BINS] = w @ counts
    v[V_W] = w @ ntot
    v[V_W2] = (w * w) @ ntot
    wg, wp = w * ng, w * npr
    v[V_S0] = wg.sum()
    v[V_S1] = wg @ gm
    v[V_S2] = w @ S[:, S_GM2] + wg @ (gm * gm)
    v[V_L1] = wg @ lm
    v[V_L2] = w @ S[:, S_LM2] + wg @ (lm * lm)
    v[V_P0] = wp.sum()
    v[V_PX] = wp @ px
    v[V_PY] = wp @ py
    v[V_PXX] = w @ S[:, S_PXX] + wp @ (px * px)
    v[V_PYY] = w @ S[:, S_PYY] + wp @ (py * py)
    v[V_PXY] = w @ S[:, S_PXY] + wp @ (px * py)
    v[V_NG] = ng.sum()
    v[V_NP] = npr.sum()
    v[V_ROWS] = float(len(w))
    return v


def _contrib_row(counts: np.ndarray, st: np.ndarray) -> np.ndarray:
    """One row's window contribution (scalar arithmetic; the per-tick path)."""
    (w, ng, gm, gm2, lm, lm2, npr, px, py, pxx, pyy, pxy, _gr, _dp) = st.tolist()
    n = float(counts.sum())
    v = np.empty(V_DIM)
    v[:TM.N_BINS] = w * counts
    wg, wp = w * ng, w * npr
    v[V_W:] = (w * n, w * w * n, wg, wg * gm, w * gm2 + wg * gm * gm, wg * lm,
               w * lm2 + wg * lm * lm, wp, wp * px, wp * py, w * pxx + wp * px * px,
               w * pyy + wp * py * py, w * pxy + wp * px * py, ng, npr, 1.0)
    return v


class WindowStats:
    """Window aggregate read from the running sums: the histogram, the
    effective gap count and the descriptor moments."""

    __slots__ = ("hist", "n_eff", "ng", "npairs", "g_mean", "g_var", "l_mean", "l_var",
                 "cxx", "cyy", "cxy")

    def __init__(self, v: np.ndarray) -> None:
        self.hist = v[:TM.N_BINS]
        W, W2 = float(v[V_W]), float(v[V_W2])
        self.n_eff = W * W / W2 if W2 > 0.0 else 0.0
        self.ng, self.npairs = float(v[V_NG]), float(v[V_NP])
        self.g_mean = self.g_var = self.l_mean = self.l_var = _NAN
        self.cxx = self.cyy = self.cxy = _NAN
        s0 = float(v[V_S0])
        if s0 > 0.0:
            gm = float(v[V_S1]) / s0
            lm = float(v[V_L1]) / s0
            gv = float(v[V_S2]) / s0 - gm * gm
            lv = float(v[V_L2]) / s0 - lm * lm
            # the power sums cancel at ~1e-13 relative: below that a spread is noise
            self.g_mean, self.l_mean = gm, lm
            self.g_var = gv if gv > 1e-12 * gm * gm else 0.0
            self.l_var = lv if lv > 1e-12 * (1.0 + lm * lm) else 0.0
        p0 = float(v[V_P0])
        if p0 > 0.0:
            mx, my = float(v[V_PX]) / p0, float(v[V_PY]) / p0
            cxx = float(v[V_PXX]) - p0 * mx * mx
            cyy = float(v[V_PYY]) - p0 * my * my
            self.cxx = cxx if cxx > 1e-12 * p0 * mx * mx else 0.0
            self.cyy = cyy if cyy > 1e-12 * p0 * my * my else 0.0
            self.cxy = float(v[V_PXY]) - p0 * mx * my

    def descriptors(self) -> Dict[str, float]:
        ok_g = self.ng >= DESC_MIN_N
        return {
            "B": TM.b_from_moments(self.g_mean, self.g_var) if ok_g else _NAN,
            "M": TM.corr_from(self.cxx, self.cyy, self.cxy) if self.npairs >= DESC_MIN_N
            else _NAN,
            "think_mu": self.l_mean if ok_g else _NAN,
            "think_sigma": math.sqrt(self.l_var) if ok_g else _NAN,
        }


def window_summary(rows: _Rows, t0: float, t1: float) -> Optional[WindowStats]:
    """Exact (non-incremental) aggregate of the rows with t0 < ts <= t1."""
    i0, i1 = rows.window(t0, t1)
    if i1 <= i0:
        return None
    return WindowStats(_contrib_rows(rows.counts[i0:i1], rows.stats[i0:i1]))


def _window_advance(live: Dict[str, Any], rows: _Rows, now: float) -> Optional[WindowStats]:
    """Move the running sums to the window (now - RECENT_S, now]: add rows
    newer than the last included one, drop rows that fell out. Exact
    recompute every WIN_RECOMPUTE_EVERY advances (bounds rounding drift) and
    whenever the window empties or the newest row was rewritten."""
    lo = now - RECENT_S
    hi_prev, lo_prev = live["win_hi"], live["win_lo"]
    v = live["win"]
    live["win_age"] += 1
    changed = False
    if live["win_age"] >= WIN_RECOMPUTE_EVERY or lo < lo_prev or now < hi_prev:
        i0, i1 = rows.window(lo, now)
        v = _contrib_rows(rows.counts[i0:i1], rows.stats[i0:i1]) if i1 > i0 else np.zeros(V_DIM)
        live["win_age"] = 0
        changed = True
    else:
        a0, a1 = rows.window(max(hi_prev, lo), now)            # new rows
        if a1 > a0:
            v = v + (_contrib_row(rows.counts[a0], rows.stats[a0]) if a1 - a0 == 1
                     else _contrib_rows(rows.counts[a0:a1], rows.stats[a0:a1]))
            changed = True
        d0, d1 = rows.window(lo_prev, min(lo, hi_prev))       # rows that left
        if d1 > d0:
            v = v - (_contrib_row(rows.counts[d0], rows.stats[d0]) if d1 - d0 == 1
                     else _contrib_rows(rows.counts[d0:d1], rows.stats[d0:d1]))
            changed = True
        if changed and v[V_ROWS] < 0.5:
            v = np.zeros(V_DIM)                                # empty: exactly zero
    live["win"], live["win_lo"], live["win_hi"] = v, lo, now
    if changed:
        live["win_rev"] += 1
    return WindowStats(v) if v[V_ROWS] >= 0.5 else None


def _rewrite_window(live: Dict[str, Any]) -> None:
    """Force an exact recompute on the next advance (newest row rewritten)."""
    live["win_age"] = WIN_RECOMPUTE_EVERY


# ================================================================== scoring
def score_window(summ: WindowStats, model: Mapping[str, Any],
                 daypart: Optional[int]) -> Tuple[float, float, float]:
    """(JSD bits, pm, G/df) of the window against the committed histogram;
    NaN when either side is too small to compare."""
    state = model["state"]
    h_l = state[TM.HIST]
    m_l = float(h_l.sum())
    n_l = m_l * m_l / float(state[TM.W2]) if state[TM.W2] > 0.0 else 0.0
    if summ.n_eff < RECENT_MIN_N or n_l < LONG_MIN_N:
        return _NAN, _NAN, _NAN
    jsd, g, df = _compare(summ.hist, summ.n_eff, h_l, n_l)
    phi = TM.dispersion(model, daypart)
    pm = max(P_FLOOR, float(special.chdtrc(df, g / phi)))
    return jsd, pm, g / df


def _compare(h_r: np.ndarray, n_r: float, h_l: np.ndarray, n_l: float
             ) -> Tuple[float, float, int]:
    """(JSD bits, G nats, df) in one pass; equals (m_timing.jsd_bits,
    m_timing.g_two_sample) on the same inputs (both masses > 0)."""
    p = h_r / float(h_r.sum())
    q = h_l / float(h_l.sum())
    n = n_r + n_l
    a = n_r / n
    lp = np.log(np.where(p > 0.0, p, 1.0))
    lq = np.log(np.where(q > 0.0, q, 1.0))
    m = 0.5 * (p + q)
    lm = np.log(np.where(m > 0.0, m, 1.0))
    jsd = 0.5 * float(p @ (lp - lm) + q @ (lq - lm)) / math.log(2.0)
    mix = a * p + (1.0 - a) * q
    lmix = np.log(np.where(mix > 0.0, mix, 1.0))
    g = 2.0 * n * (a * float(p @ (lp - lmix)) + (1.0 - a) * float(q @ (lq - lmix)))
    df = max(1, int(np.count_nonzero(n * mix >= 0.5)) - 1)
    return min(1.0, max(0.0, jsd)), max(0.0, g), df


# ================================================================ learner maths
def _advance(state: np.ndarray, ts: float) -> None:
    """Move the clock forward to ts, decaying the statistics (never backwards)."""
    t = state[TM.T]
    if t != t:
        state[TM.T] = ts
    elif ts > t:
        d = math.exp(-(ts - t) / TM.TAU_S)
        state[TM.DECAY_IDX] *= d
        state[TM.W2] *= d * d
        state[TM.T] = ts


def _fold(state: np.ndarray, hist: np.ndarray, w2: float, g: Tuple[float, ...],
          p: Tuple[float, ...], disp: Optional[np.ndarray], a: float) -> None:
    """state += a * (a sample of statistics), in place, by Chan's parallel
    combination. g = (W, mean, M2, lmean, lM2) and p = (W, mx, my, Cxx, Cyy,
    Cxy) carry per-gap weights; w2 is a sum of squared per-gap weights (so it
    scales by a^2); disp is [W, S] per daypart (tick weights)."""
    state[TM.HIST] += a * hist
    state[TM.W2] += a * a * w2
    wb = a * g[0]
    if wb > 0.0:
        wa = float(state[TM.GW])
        w = wa + wb
        dm = g[1] - float(state[TM.GM])
        dl = g[3] - float(state[TM.LM])
        state[TM.GM] += dm * wb / w
        state[TM.GM2] += a * g[2] + dm * dm * wa * wb / w
        state[TM.LM] += dl * wb / w
        state[TM.LM2] += a * g[4] + dl * dl * wa * wb / w
        state[TM.GW] = w
    wb = a * p[0]
    if wb > 0.0:
        wa = float(state[TM.PW])
        w = wa + wb
        dx = p[1] - float(state[TM.PX])
        dy = p[2] - float(state[TM.PY])
        f = wa * wb / w
        state[TM.PX] += dx * wb / w
        state[TM.PY] += dy * wb / w
        state[TM.PXX] += a * p[3] + dx * dx * f
        state[TM.PYY] += a * p[4] + dy * dy * f
        state[TM.PXY] += a * p[5] + dx * dy * f
        state[TM.PW] = w
    if disp is not None:
        state[TM.DISP:TM.DISP + 2 * TM.N_DAYPARTS] += a * disp


def _update(state: np.ndarray, row: Tuple[float, np.ndarray, List[float]],
            w: float) -> np.ndarray:
    """GatedLearner update: fold one tick row with trust weight w. A row older
    than the clock is decayed on the way in (release / replay order safe);
    deterministic, so checkpoint + replay equals the original fit."""
    w = float(w)
    if not (w > 0.0 and math.isfinite(w)):
        return state
    ts, counts, st = row
    _advance(state, ts)
    a = w * (math.exp(-(state[TM.T] - ts) / TM.TAU_S) if ts < state[TM.T] else 1.0)
    rw = st[S_W]
    n = float(counts.sum())
    g = (rw * st[S_NG], st[S_GM], rw * st[S_GM2], st[S_LM], rw * st[S_LM2])
    p = (rw * st[S_NP], st[S_PX], st[S_PY], rw * st[S_PXX], rw * st[S_PYY], rw * st[S_PXY])
    # per-gap weight a * rw (histogram, W2, moments)
    _fold(state, rw * counts.astype(np.float64), rw * rw * n, g, p, None, a)
    gr, dp = st[S_GR], st[S_DP]
    if gr == gr and dp == dp and 0 <= int(dp) < TM.N_DAYPARTS:
        # dispersion: tick weight a (trust x decay), G/df capped
        k = TM.DISP + 2 * int(dp)
        state[k] += a
        state[k + 1] += a * min(max(float(gr), 0.0), DISP_RATIO_CAP)
    return state


def _merge(own: np.ndarray, other: np.ndarray, w: float) -> np.ndarray:
    """Link seeding: own + w * other in sufficient-statistic space, both at
    the later of the two clocks."""
    to = other[TM.T]
    if to != to or not float(other[TM.HIST].sum()) > 0.0:
        return own
    _advance(own, float(to))
    a = float(w) * math.exp(-(own[TM.T] - to) / TM.TAU_S)
    g = (float(other[TM.GW]), float(other[TM.GM]), float(other[TM.GM2]),
         float(other[TM.LM]), float(other[TM.LM2]))
    p = (float(other[TM.PW]), float(other[TM.PX]), float(other[TM.PY]),
         float(other[TM.PXX]), float(other[TM.PYY]), float(other[TM.PXY]))
    _fold(own, other[TM.HIST], float(other[TM.W2]), g, p,
          other[TM.DISP:TM.DISP + 2 * TM.N_DAYPARTS], a)
    return own


def _dump(state: np.ndarray) -> np.ndarray:
    return state.copy()


def _load(blob: Any) -> np.ndarray:
    return np.array(blob, dtype=np.float64).reshape(-1)


# ============================================================ periodicity
def _log_q(m: int, x: float) -> float:
    """ln Q(m, x) = ln chi2.sf(2x, 2m) for integer m >= 1, stable at any x:
    -x + ln sum_{j<m} x^j / j!."""
    if x <= 0.0:
        return 0.0
    lx = math.log(x)
    terms = [j * lx - math.lgamma(j + 1.0) for j in range(m)]
    top = max(terms)
    return -x + top + math.log(sum(math.exp(v - top) for v in terms))


def strict_period(times: Any) -> Tuple[float, float, int]:
    """(period_s, p, n) of the best strictly periodic train in `times`, or
    (NaN, NaN, n) when there are too few events or too short a span.

    Candidate: rfft (zero-padded to a power of two) of the mean-removed 5-s
    binned counts, normalised by the band's median power (noise powers are
    ~Exp(1)); per fundamental k the harmonic sum over h = 1..4 (harmonics take
    the max over +-1 bin against scalloping). The largest sums are ranked
    exactly by their chi2_2m tail, so the fundamental beats its sub- and
    super-harmonics; the peak is refined by parabolic interpolation.
    Confirmation: the window-corrected Rayleigh statistic Z^2_1 = 2 n R^2
    (evt.z2_periodogram) at the refined frequency, p from evt.z2_p's Davies
    bound over the whole scanned band [10 s, span/4]: it bounds the maximum
    over the continuous band, hence this data-driven choice. A period below
    10 s aliases in the bins, but then the phases at the alias are not
    coherent and the confirmation fails. ~0.15 ms at 1024 events."""
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    if not (t.size and np.isfinite(t).all() and (t.size < 2 or bool(np.all(t[1:] >= t[:-1])))):
        t = np.sort(t[np.isfinite(t)])
    n = int(t.size)
    if n < PERIOD_MIN_EVENTS:
        return _NAN, _NAN, n
    span = float(t[-1] - t[0])
    p_max = span / 4.0
    if not p_max > 1.5 * PERIOD_MIN_S:
        return _NAN, _NAN, n
    nb = int(math.floor(span / PERIOD_BIN_S)) + 1
    counts = np.bincount(((t - t[0]) * (1.0 / PERIOD_BIN_S)).astype(np.intp), minlength=nb)
    nfft = 1 << max(4, (nb - 1).bit_length())
    spec = np.fft.rfft(counts - counts.mean(), n=nfft)
    pw = spec.real * spec.real + spec.imag * spec.imag
    t_fft = nfft * PERIOD_BIN_S
    k_lo = max(1, int(math.ceil(t_fft / p_max)))
    k_hi = min(pw.size - 2, int(math.floor(t_fft / PERIOD_MIN_S)))
    if k_hi - k_lo < 2:
        return _NAN, _NAN, n
    band = pw[k_lo:k_hi + 1]
    mid = band.size // 2
    scale = float(np.partition(band, mid)[mid]) / math.log(2.0)
    if not scale > 0.0:
        scale = float(band.mean())
    if not scale > 0.0:
        return _NAN, _NAN, n                        # flat counts: no train
    z = pw * (1.0 / scale)
    zmax = z.copy()
    np.maximum(zmax[1:], z[:-1], out=zmax[1:])
    np.maximum(zmax[:-1], z[1:], out=zmax[:-1])
    ks = np.arange(k_lo, k_hi + 1)
    hs = z[k_lo:k_hi + 1].copy()
    m = np.ones(ks.size, dtype=np.intp)
    for h in range(2, PERIOD_HARMONICS + 1):
        top = k_hi // h                             # fundamentals whose h-th harmonic fits
        if top < k_lo:
            break
        sel = slice(0, top - k_lo + 1)
        hs[sel] += zmax[ks[sel] * h]
        m[sel] += 1
    c = PERIOD_CANDIDATES
    cand = np.argpartition(hs, -c)[-c:] if hs.size > c else np.arange(hs.size)
    j = min(cand.tolist(), key=lambda i: (_log_q(int(m[i]), float(hs[i])), i))
    k = int(ks[j])
    a, b0, c1 = float(pw[k - 1]), float(pw[k]), float(pw[k + 1])
    den = a - 2.0 * b0 + c1
    delta = 0.5 * (a - c1) / den if den < 0.0 else 0.0
    f_c = (k + max(-0.5, min(0.5, delta))) / t_fft
    if not (1.0 / p_max < f_c <= 1.0 / PERIOD_MIN_S):
        return _NAN, _NAN, n
    z2 = float(evt.z2_periodogram(t, [1.0 / f_c], m=1)[0])
    if not z2 == z2:
        return _NAN, _NAN, n
    n_eff = (1.0 / PERIOD_MIN_S - 1.0 / p_max) * span
    return 1.0 / f_c, float(evt.z2_p(z2, 1, n_eff)), n


# ================================================================ the engine
class _Rec:
    """One entity's per-tick outcome, written after the periodicity pass."""
    __slots__ = ("s", "e", "model", "status", "summ", "score", "pm", "alarm")

    def __init__(self, s: str, e: str, model: Dict[str, Any], status: str,
                 summ: Optional[WindowStats], score: float, pm: float,
                 alarm: Optional[int]) -> None:
        self.s, self.e, self.model, self.status = s, e, model, status
        self.summ, self.score, self.pm, self.alarm = summ, score, pm, alarm


class TimingEngine(Engine):
    name = "behavior.timing"
    layer = "behavior"
    consumes = ["act.stream", "act.stream_frac", "act.events", "feature.active", "model.seq",
                "behavior.trust", "behavior.trust_prov", "behavior.quarantine",
                "model.control", "model.link"]
    produces = ["model.timing", "behavior.score", "behavior.pm", "behavior.acc_alarm",
                "behavior.axes", "behavior.degraded", "behavior.timing", "profile.extra.timing"]
    description = ("Per-IP inter-arrival profile: 32-bin log-gap histogram (JSD of the "
                   "recent 6 h against the 7-d model), burstiness, memory, log-normal think "
                   "time and strict periodicity (rfft + Rayleigh); trust-gated learning.")
    interval = 1

    def __init__(self, period_max_per_tick: int = PERIOD_MAX_PER_TICK, **params: Any) -> None:
        super().__init__(**params)
        self.period_max_per_tick = int(period_max_per_tick)
        self._rows_cur: Optional[_Rows] = None
        self._learner = G.GatedLearner(
            name=LEARNER, init=TM.new_state, update=_update, fetch=self._fetch,
            dump=_dump, load=_load, merge=_merge)

    # ------------------------------------------------------------------ run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        now, dt = float(ctx.now), float(ctx.window_s)
        self._learner.d_min_s = float(ctx.config.get("D_min_s") or G.D_MIN_S)
        daypart = TB.DAYPARTS.index(TB.tctx_from_config(now, ctx.config, dt)["daypart"])
        alpha = _BUDGET_PER_DAY * dt / 86400.0
        frontier = G.commit_frontier(now, dt, self._learner.d_min_s)
        r2_failed = store.engine_failed(R2_ENGINE, now)
        recs: List[_Rec] = []
        for s in store.systems():
            for e in store.entities(s):
                rec = self._entity(ctx, s, e, now, dt, daypart, alpha, frontier, r2_failed)
                if rec is not None:
                    recs.append(rec)
        self._periodicity(recs, now)
        return sum(self._write(ctx, rec, now, dt) for rec in recs)

    # ------------------------------------------------------------- per entity
    def _entity(self, ctx: Context, s: str, e: str, now: float, dt: float, daypart: int,
                alpha: float, frontier: float, r2_failed: bool) -> Optional[_Rec]:
        store = ctx.store
        model = store.get_model(s, e, TM.MODEL)
        status, t, frac = _read_stream(store, s, e, now, r2_failed)
        if not isinstance(model, dict):
            if status != _OK:
                return None                       # never seen with timestamps: nothing to do
            model = new_model()
        live, rows = model["live"], model["rows"]
        fresh_row = False
        if status == _OK:
            row = build_row(t, frac, live, _session_gap(store, s, e))
            if row is not None:
                if not rows.append(now, row[0], row[1]):
                    _rewrite_window(live)
                fresh_row = True
            _push_events(live, t, now)
        elif status != _SILENT:
            # events at unknown times, or R2 did not deliver: the gap chain breaks
            live["last_ev"], live["last_full"], live["last_gap"] = _NAN, False, _NAN

        # score against the model as of the last commit, then learn
        summ = _window_advance(live, rows, now)
        score = pm = _NAN
        alarm: Optional[int] = None
        if summ is not None and status != _DEGRADED:
            state = model["state"]
            key = (live["win_rev"], daypart, float(state[TM.T]), float(state[TM.HIST].sum()),
                   float(state[TM.W2]), float(state[TM.DISP:].sum()))
            if key == live["score_key"]:
                score, pm, g_ratio = live["score"]          # nothing changed: same answer
            else:
                score, pm, g_ratio = score_window(summ, model, daypart)
                live["score_key"], live["score"] = key, (score, pm, g_ratio)
            if fresh_row and g_ratio == g_ratio:
                rows.set_last(S_GR, g_ratio)
                rows.set_last(S_DP, float(daypart))
            if pm == pm:
                alarm = 0 if ctx.training else int(pm <= alpha)
        elif status == _DEGRADED:
            summ = None
        self._learn(ctx, s, e, model, now, dt, frontier)
        rows.prune(now - ROW_KEEP_S)
        return _Rec(s, e, model, status, summ, score, pm, alarm)

    def _learn(self, ctx: Context, s: str, e: str, model: Dict[str, Any], now: float,
               dt: float, frontier: float) -> None:
        """Gated commit of due rows (+ control, + link seeding). Skipped when
        no row is due and there is no model.control: GatedLearner would only
        advance its cursor over rowless ticks, which the next step does in one
        batch anyway."""
        store, rows, gate = ctx.store, model["rows"], model["gate"]
        i0, i1 = rows.window(gate.last_ts, frontier)
        st = model["state"]
        self._rows_cur = rows
        try:
            if i1 > i0 or store.get_model(s, e, G.CONTROL_MODEL) is not None:
                st, gate = self._learner.step(store, s, e, st.copy(), gate, now, dt,
                                              training=bool(ctx.training))
            st, gate = self._learner.seed_from_link(store, s, e, st, gate,
                                                    lambda a: _other_state(store, s, a))
        finally:
            self._rows_cur = None
        model["state"], model["gate"], model["version"] = st, gate, int(gate.version)

    def _fetch(self, store: Any, s: str, e: str, ts: float
               ) -> Optional[Tuple[float, np.ndarray, List[float]]]:
        """GatedLearner fetch: the tick row at ts from the entity's row buffer
        (None: no true gap that tick, or pruned)."""
        rows = self._rows_cur
        if rows is None:
            return None
        i = rows.find(float(ts))
        if i < 0:
            return None
        return float(rows.ts[i]), rows.counts[i].copy(), rows.stats[i].tolist()

    # ------------------------------------------------------------ periodicity
    def _periodicity(self, recs: List[_Rec], now: float) -> None:
        """Strict-period checks for at most period_max_per_tick due entities,
        stalest first (a fixed entity order would starve the tail)."""
        due = []
        for r in recs:
            lv = r.model["live"]
            if r.status == _DEGRADED or lv["ev_new"] <= 0 or lv["ev"].size < PERIOD_MIN_EVENTS:
                continue
            if now - lv["period_eval"] >= PERIOD_EVERY_S:
                due.append(r)
        due.sort(key=lambda r: r.model["live"]["period_eval"])
        for r in due[:max(0, self.period_max_per_tick)]:
            lv = r.model["live"]
            per, p, n = strict_period(lv["ev"])
            confirmed = p == p and p <= PERIOD_P_CONFIRM
            lv["period"] = {"ts": now, "period": per if confirmed else _NAN, "p": p, "n": n}
            lv["period_eval"] = now
            lv["ev_new"] = 0

    # ------------------------------------------------------------------ write
    def _write(self, ctx: Context, r: _Rec, now: float, dt: float) -> int:
        store = ctx.store
        s, e, model = r.s, r.e, r.model
        store.put_model(s, e, TM.MODEL, model, version=model["version"], ts=now)
        win = int(dt)
        if r.status == _DEGRADED:
            emit.write_scores(store, s, e, now, {DETECTOR: _NAN},
                              degraded={DETECTOR: "stale:act.stream"}, window_s=win)
            return 1
        if r.score == r.score:
            emit.write_scores(
                store, s, e, now, {DETECTOR: r.score},
                pm={DETECTOR: r.pm} if r.pm == r.pm else None,
                axes={DETECTOR: AXES} if (r.pm <= AXES_PM_MAX or r.alarm) else None,
                acc_alarm={DETECTOR: r.alarm} if r.alarm is not None else None,
                window_s=win)
        per, pp = TM.period(model, now)
        if r.summ is None and not pp == pp:
            return 1 if r.score == r.score else 0
        desc = r.summ.descriptors() if r.summ is not None else dict(_NO_DESC)
        desc["period"], desc["period_p"] = per, pp
        model["live"]["recent"] = desc
        store.add_derived(DerivedMetric(name=SERIES, value=dict(desc), ts=now, system=s,
                                        entity=e, window_s=win, kind=MetricKind.CATEGORICAL,
                                        inputs=["act.stream"]))
        if self.entity_due((s, e, "profile"), now, PROFILE_EVERY_S):
            _write_profile(store, s, e, model, desc, now)
        return 1


# ================================================================ store reads
def _read_stream(store: Any, s: str, e: str, now: float, r2_failed: bool
                 ) -> Tuple[str, np.ndarray, float]:
    """(status, sorted finite event times, stream_frac) for tick `now`.

    ok       act.stream rows with finite ts (NaN rows dropped; the tick then
             counts as sampled, since a dropped row would merge two gaps)
    silent   no events (absence is data: the gap chain continues)
    unknown  events this tick without usable timestamps (the chain breaks)
    degraded R2 failed this tick, or did not zero-fill act.events for an
             entity it should have (contract M: NaN + behavior.degraded)"""
    rows = MT.stream_rows(store, s, e, now)
    empty = np.zeros(0, dtype=np.float64)
    if rows.size:
        t = np.asarray(rows["ts"], dtype=np.float64)
        fin = np.isfinite(t)
        frac = MT.stream_frac(store, s, e, now)
        frac = frac if (frac == frac and 0.0 < frac <= 1.0) else 1.0
        if not fin.all():
            frac = min(frac, float(fin.sum()) / t.size, 1.0 - 1e-6)
            t = t[fin]
        if t.size == 0:
            return _UNKNOWN, empty, 1.0
        if t.size > 1 and np.any(t[1:] < t[:-1]):
            t = np.sort(t)
        return _OK, t, frac
    if r2_failed:
        return _DEGRADED, empty, 1.0
    ev = store.latest_fresh(s, e, "act.events", now)
    if isinstance(ev, (int, float)) and ev == ev:
        return (_UNKNOWN if ev > 0 else _SILENT), empty, 1.0
    lw = store.last_write_ts(s, e, "act.events")
    if lw is not None and lw < now:
        ls = store.last_seen(s, e)
        if ls is not None and now - ls <= MT.ZERO_FILL_S:
            return _DEGRADED, empty, 1.0           # R2 zero-fills every entity it knows
    return _SILENT, empty, 1.0


def _session_gap(store: Any, s: str, e: str) -> float:
    """model.seq session_gap (B10; contract C), clipped like B10 does; 30 min
    when absent or invalid."""
    m = store.get_model(s, e, SEQ_MODEL)
    g = m.get("session_gap") if isinstance(m, Mapping) else getattr(m, "session_gap", None)
    if isinstance(g, bool) or not isinstance(g, (int, float, np.integer, np.floating)):
        return SESSION_GAP_DEFAULT_S
    g = float(g)
    if not (g > 0.0 and math.isfinite(g)):
        return SESSION_GAP_DEFAULT_S
    return min(SESSION_GAP_CLIP[1], max(SESSION_GAP_CLIP[0], g))


def _other_state(store: Any, s: str, entity: str) -> Optional[np.ndarray]:
    st = TM.state_of(store.get_model(s, entity, TM.MODEL))
    return st.copy() if st is not None else None


def _push_events(live: Dict[str, Any], t: np.ndarray, now: float) -> None:
    ev = live["ev"]
    ev = np.concatenate((ev, t)) if ev.size else t.copy()
    lo = now - PERIOD_WINDOW_S
    if ev.size and ev[0] < lo:
        ev = ev[int(np.searchsorted(ev, lo, side="left")):]
    if ev.size > EV_CAP:
        ev = ev[-EV_CAP:]
    live["ev"] = ev
    live["ev_new"] = int(live.get("ev_new", 0)) + int(t.size)


def _json(v: float, nd: int = 4) -> Optional[float]:
    v = float(v)
    return round(v, nd) if math.isfinite(v) else None


def _write_profile(store: Any, s: str, e: str, model: Dict[str, Any],
                   recent: Mapping[str, float], now: float) -> None:
    """profile.extra.timing: the committed profile in natural units plus the
    recent-window descriptors (JSON-friendly: NaN -> None)."""
    d = TM.descriptors(model, now)
    p = store.profile(s, e) or EntityProfile(system=s, entity=e)
    out: Dict[str, Any] = {k: _json(v) for k, v in d.items()}
    out["recent"] = {k: _json(v) for k, v in recent.items()}
    out["hist"] = TM.as_float_list(TM.hist_probs(model))
    out["version"] = int(model.get("version", 0))
    out["updated"] = now
    p.extra["timing"] = out
    store.put_profile(p)
