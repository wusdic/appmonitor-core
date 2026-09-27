"""Read accessors and shared maths for model.timing (owner: B11 TimingEngine; contract C).

Why a timing model: the inter-arrival process is one of the few behavioural
traits that survives encryption, NAT-free sampling and template churn. A person
produces heavy-tailed, bursty gaps (reading, thinking, switching tasks); a
script produces a narrow gap distribution or a strictly periodic train. The
shape of the gap distribution, its burstiness, the memory between consecutive
gaps and the in-session think time together separate humans from scripts and
one person from another (B15 identity model, B16 attribution score candidate
windows under each candidate's model with `loglik`; B30 turns `descriptors`
into a portrait). Consumers call these functions instead of reading the model
dict, so the layout can evolve in one place. Everything here is pure: no
store access, nothing mutated.

model.timing@(s, e) is a dict (stored by reference):
    {
      "fmt": 1,
      "version": int,          # model.control version applied (gate.version)
      "state": float64[STATE_DIM],   # committed (trust-gated) statistics, below
      "gate": GateState,       # lib/gating bookkeeping           (engine-internal)
      "rows": object,          # per-tick gap summaries, 8 d      (engine-internal)
      "live": {                # ungated scoring state            (engine-internal
        "last_ev", "last_full", "last_gap", "ev", "ev_new",       #  except below)
        "period": {"ts", "period", "p", "n"},   # last strict-period check
        "recent": {B, M, think_mu, think_sigma, period, period_p},  # = behavior.timing
      },
    }

state layout (all decayed with an e-folding time TAU_S = 7 d; a row older than
the clock is folded with its weight decayed, never the state backwards):
    [0:32]   HIST  per-bin gap weight; bin k covers log2 gaps in
                   [EDGES_S[k], EDGES_S[k+1]), 10 ms .. 1 day (first / last bin
                   also take anything below / above)
    32       W2    sum of squared per-gap weights (n_eff = HIST.sum()^2 / W2)
    33..35   GW, GM, GM2      active gaps (< BURST_MAX_S): weight, mean (s), sum sq. dev.
    36..38   TW, LM, LM2      within-session gaps (< session gap): weight, mean and
                              sum sq. dev. of ln(gap s)
    39..44   PW, PX, PY, PXX, PYY, PXY   consecutive active-gap pairs
                   (x = gap_i, y = gap_i+1): weight, means, co-moments
    45..52   DISP  per daypart (timebins.DAYPARTS order) [W, S]: decayed sum of
                   tick weights and of G/df of the recent-window test on those
                   ticks (overdispersion of the gap histogram, see `dispersion`)
    53       T     clock: newest folded row ts (NaN = empty model)
Two gap populations, because a single overnight gap would dominate every raw
second moment:
  * "active" gaps (< BURST_MAX_S = 30 min, fixed) carry B and M. They include
    the short breaks between a person's sessions, which is what makes human
    activity bursty (Goh & Barabasi); longer silences are the rhythm model's
    business (B07), and the histogram still covers them.
  * "within-session" gaps (< the entity's session gap, model.seq session_gap,
    default 30 min) carry the think-time fit, which is a property of a session.

Descriptor definitions (Goh & Barabasi 2008):
    B = (sd - mean) / (sd + mean) of the active gaps: -1 periodic, 0 Poisson,
        -> 1 bursty. (A pure log-normal renewal with sigma = 1 has CV =
        sqrt(e - 1) = 1.31, B = 0.135; human sessions separated by minutes-long
        breaks give B ~ 0.3 - 0.6.)
    M = corr(gap_i, gap_i+1) over consecutive active gaps.
    think time: log-normal fit (MLE) to within-session gaps, (mu, sigma) of ln s.

Accessor signatures (model may be None or {}: the documented NaN default):
    new_state() -> ndarray[STATE_DIM]
    state_of(model) -> ndarray | None
    is_empty(model) -> bool
    mass(model) -> float                         decayed gap weight in the histogram
    n_eff(model) -> float                        effective number of gaps
    pmf(model, alpha=DIRICHLET_ALPHA) -> ndarray[32]   smoothed predictive over bins
    loglik(model, gaps, alpha=...) -> float      sum ln P(bin(gap)) in nats; NaN for an
                                                 empty model, 0.0 when no valid gap
    loglik_per_gap(model, gaps, alpha=...) -> ndarray  per gap (NaN for invalid gaps)
    quantile(model, q) -> float                  gap (s) at CDF q, log-interpolated
    quantiles(model, qs) -> ndarray              several levels in one pass
    burstiness(model) -> float;  memory(model) -> float      over active gaps
    think_time(model) -> (mu, sigma)             ln-seconds
    think_logpdf(model, gaps) -> ndarray         N(mu, sigma) log-density of ln(gap)
    dispersion(model, daypart=None) -> float >= 1  phi used to scale the G test
    period(model, now=None) -> (period_s, period_p)  last strict-period check
                                                 (NaN if none, or older than 6 h)
    descriptors(model, now=None) -> dict         long-term profile in natural units
    recent(model) -> dict                        last published behavior.timing
    hist_probs(model) -> ndarray[32]             normalised committed histogram (no smoothing)
    as_float_list(x, nd=4) -> list               JSON-friendly rounding (NaN -> None)
Pure maths shared with B11 (and usable by B15/B16 on their own windows):
    bin_index(gaps) -> int ndarray (-1 for non-finite / negative gaps)
    gaps_from_times(times, frac=1.0) -> ndarray  true gaps of ONE tick's act.stream
    b_from_moments(mean, var) -> float;  corr_from(cxx, cyy, cxy) -> float
    jsd_bits(p, q) -> float                      Jensen-Shannon divergence, [0, 1]
    g_two_sample(p1, n1, p2, n2) -> (G, df)      two-sample G statistic (nats), df
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

MODEL = "model.timing"
FMT = 1

N_BINS = 32
GAP_MIN_S = 0.01                  # 10 ms: timestamp resolution / parallel fetches
GAP_MAX_S = 86400.0               # 1 day
_L2_MIN = math.log2(GAP_MIN_S)
BIN_W = (math.log2(GAP_MAX_S) - _L2_MIN) / N_BINS      # ~0.72 octave per bin
EDGES_S = GAP_MIN_S * np.exp2(BIN_W * np.arange(N_BINS + 1))

TAU_S = 7 * 86400.0               # e-folding time of the committed statistics
DIRICHLET_ALPHA = 1.0             # total pseudo-count of the predictive (uniform over bins)
DESC_MIN_W = 8.0                  # min (weighted) gaps / pairs for a descriptor
SESSION_GAP_S = 30.0              # R2 sampling: rows are whole sessions split at > 30 s
BURST_MAX_S = 1800.0              # active gaps (B, M): longer silences belong to rhythm
N_DAYPARTS = 4
PHI_PRIOR = 2.0                   # prior overdispersion of human gap histograms
PHI_PRIOR_W = 8.0                 # its weight, in ticks
PERIOD_STALE_S = 6 * 3600.0       # a strict-period check older than this is not reported

# state layout
HIST = slice(0, N_BINS)
W2 = 32
GW, GM, GM2 = 33, 34, 35
TW, LM, LM2 = 36, 37, 38
PW, PX, PY, PXX, PYY, PXY = 39, 40, 41, 42, 43, 44
DISP = 45                         # DISP + 2*dp -> W, DISP + 2*dp + 1 -> S
T = 53
STATE_DIM = 54
# entries scaled by d when the clock advances by dt (d = exp(-dt/TAU_S)); W2
# scales by d^2 and the means (GM, LM, PX, PY) and T do not scale
DECAY_IDX = np.array(list(range(N_BINS)) + [GW, GM2, TW, LM2, PW, PXX, PYY, PXY]
                     + list(range(DISP, DISP + 2 * N_DAYPARTS)), dtype=np.intp)

_NAN = math.nan


# ================================================================ pure maths
def new_state() -> np.ndarray:
    s = np.zeros(STATE_DIM, dtype=np.float64)
    s[T] = _NAN
    return s


def bin_index(gaps: Any) -> np.ndarray:
    """Histogram bin of each gap (s): log2-spaced, 10 ms .. 1 day, clipped to
    [0, 31]; non-finite or negative gaps -> -1."""
    g = np.asarray(gaps, dtype=np.float64).reshape(-1)
    out = np.full(g.shape, -1, dtype=np.intp)
    ok = np.isfinite(g) & (g >= 0.0)
    if ok.any():
        x = np.log2(np.maximum(g[ok], GAP_MIN_S))
        out[ok] = np.clip(((x - _L2_MIN) / BIN_W).astype(np.intp), 0, N_BINS - 1)
    return out


def gaps_from_times(times: Any, frac: float = 1.0) -> np.ndarray:
    """True inter-event gaps (s) inside ONE tick of act.stream (sorted ts).

    With stream_frac < 1 R2 kept whole sessions (split at gaps > 30 s), so only
    gaps <= SESSION_GAP_S are known to be true gaps; larger ones may span a
    dropped session and are left out. Gaps are floored at GAP_MIN_S. The gap
    to the previous tick is not included (B11 adds it when both ticks are
    complete)."""
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    t = np.sort(t[np.isfinite(t)])
    if t.size < 2:
        return np.zeros(0, dtype=np.float64)
    d = np.diff(t)
    f = float(frac) if frac is not None else 1.0
    if f == f and f < 1.0 - 1e-9:
        d = d[d <= SESSION_GAP_S]
    return np.maximum(d, GAP_MIN_S)


def b_from_moments(mean: float, var: float) -> float:
    """Burstiness (sd - mean)/(sd + mean); NaN when undefined."""
    if not (mean == mean and var == var) or mean <= 0.0 or var < 0.0:
        return _NAN
    sd = math.sqrt(var)
    return (sd - mean) / (sd + mean)


def corr_from(cxx: float, cyy: float, cxy: float) -> float:
    """Pearson correlation from co-moments; NaN when a variance is ~0 (a
    perfectly regular train has no defined memory)."""
    if not (cxx > 0.0 and cyy > 0.0) or not math.isfinite(cxy):
        return _NAN
    den = math.sqrt(cxx * cyy)
    if not den > 0.0:
        return _NAN
    return max(-1.0, min(1.0, cxy / den))


def jsd_bits(p: Any, q: Any) -> float:
    """Jensen-Shannon divergence (equal weights, bits, in [0, 1]) of two
    histograms (normalised here; NaN if either has no mass)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    sp, sq = float(p.sum()), float(q.sum())
    if not (sp > 0.0 and sq > 0.0):
        return _NAN
    p = p / sp
    q = q / sq
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0.0, p * np.log2(p / m), 0.0)
        b = np.where(q > 0.0, q * np.log2(q / m), 0.0)
    return float(min(1.0, max(0.0, 0.5 * (a.sum() + b.sum()))))


def g_two_sample(p1: Any, n1: float, p2: Any, n2: float,
                 min_count: float = 0.5) -> Tuple[float, int]:
    """Two-sample G statistic (nats) of histograms p1, p2 (normalised here)
    with effective sizes n1, n2: G = 2 N JSD_pi(p1, p2), pi = (n1, n2)/N,
    which is the 2 x K contingency likelihood-ratio statistic. df = number of
    bins whose pooled count >= min_count, minus 1 (at least 1). NaN if a
    sample is empty."""
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    s1, s2 = float(p1.sum()), float(p2.sum())
    n1, n2 = float(n1), float(n2)
    if not (s1 > 0.0 and s2 > 0.0 and n1 > 0.0 and n2 > 0.0):
        return _NAN, 0
    p1 = p1 / s1
    p2 = p2 / s2
    n = n1 + n2
    a1, a2 = n1 / n, n2 / n
    mix = a1 * p1 + a2 * p2
    with np.errstate(divide="ignore", invalid="ignore"):
        k1 = np.where(p1 > 0.0, p1 * np.log(p1 / mix), 0.0).sum()
        k2 = np.where(p2 > 0.0, p2 * np.log(p2 / mix), 0.0).sum()
    g = 2.0 * n * (a1 * float(k1) + a2 * float(k2))
    df = max(1, int(np.count_nonzero(n * mix >= min_count)) - 1)
    return max(0.0, g), df


# ============================================================ model accessors
def state_of(model: Any) -> Optional[np.ndarray]:
    if not isinstance(model, Mapping):
        return None
    s = model.get("state")
    if s is None:
        return None
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    return s if s.size == STATE_DIM else None


def mass(model: Any) -> float:
    s = state_of(model)
    return float(s[HIST].sum()) if s is not None else 0.0


def is_empty(model: Any) -> bool:
    return not mass(model) > 0.0


def n_eff(model: Any) -> float:
    s = state_of(model)
    if s is None:
        return 0.0
    m = float(s[HIST].sum())
    return m * m / float(s[W2]) if s[W2] > 0.0 else 0.0


def pmf(model: Any, alpha: float = DIRICHLET_ALPHA) -> np.ndarray:
    """Predictive probability of each bin for a new gap: (h_k + alpha/32) /
    (sum h + alpha). All NaN for an empty model."""
    s = state_of(model)
    if s is None or not s[HIST].sum() > 0.0:
        return np.full(N_BINS, _NAN)
    h = s[HIST]
    a = max(float(alpha), 0.0)
    return (h + a / N_BINS) / (float(h.sum()) + a)


def loglik_per_gap(model: Any, gaps: Any, alpha: float = DIRICHLET_ALPHA) -> np.ndarray:
    """ln P(bin(gap)) per gap in nats (NaN for non-finite / negative gaps or an
    empty model). Candidates share the binning, so comparing their logliks on
    the same gaps needs no Jacobian."""
    b = bin_index(gaps)
    out = np.full(b.shape, _NAN)
    p = pmf(model, alpha)
    if not np.isfinite(p[0]):
        return out
    ok = b >= 0
    out[ok] = np.log(p[b[ok]])
    return out


def loglik(model: Any, gaps: Any, alpha: float = DIRICHLET_ALPHA) -> float:
    """Sum of ln P(bin(gap)) over the valid gaps (nats). NaN for an empty
    model (no evidence either way), 0.0 when no gap is valid."""
    if is_empty(model):
        return _NAN
    v = loglik_per_gap(model, gaps, alpha)
    v = v[np.isfinite(v)]
    return float(v.sum()) if v.size else 0.0


def quantiles(model: Any, qs: Sequence[float]) -> np.ndarray:
    """Gaps (s) at CDF levels qs of the committed histogram, log-linear inside
    the bin. NaN for an empty model or a q outside [0, 1]."""
    q = np.asarray(qs, dtype=np.float64).reshape(-1)
    out = np.full(q.shape, _NAN)
    s = state_of(model)
    if s is None:
        return out
    h = s[HIST]
    tot = float(h.sum())
    if not tot > 0.0:
        return out
    c = np.cumsum(h) / tot
    ok = (q >= 0.0) & (q <= 1.0)
    k = np.minimum(np.searchsorted(c, q[ok], side="left"), N_BINS - 1)
    lo = np.where(k > 0, c[np.maximum(k - 1, 0)], 0.0)
    width = c[k] - lo
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(width > 0.0, (q[ok] - lo) / width, 0.5)
    out[ok] = GAP_MIN_S * np.exp2(BIN_W * (k + np.clip(u, 0.0, 1.0)))
    return out


def quantile(model: Any, q: float) -> float:
    """Gap (s) at CDF level q (see quantiles)."""
    return float(quantiles(model, [q])[0])


def burstiness(model: Any) -> float:
    s = state_of(model)
    if s is None or not s[GW] >= DESC_MIN_W:
        return _NAN
    return b_from_moments(float(s[GM]), float(s[GM2]) / float(s[GW]))


def memory(model: Any) -> float:
    s = state_of(model)
    if s is None or not s[PW] >= DESC_MIN_W:
        return _NAN
    return corr_from(float(s[PXX]), float(s[PYY]), float(s[PXY]))


def think_time(model: Any) -> Tuple[float, float]:
    """(mu, sigma) of ln(within-session gap s): the log-normal MLE."""
    s = state_of(model)
    if s is None or not s[TW] >= DESC_MIN_W:
        return _NAN, _NAN
    return float(s[LM]), math.sqrt(max(0.0, float(s[LM2]) / float(s[TW])))


def think_logpdf(model: Any, gaps: Any) -> np.ndarray:
    """Normal log-density of ln(gap) under the think-time fit (nats per gap;
    density in ln-seconds). NaN for invalid gaps or an unfitted model; sigma
    is floored at 0.05 so a perfectly regular model stays finite."""
    g = np.asarray(gaps, dtype=np.float64).reshape(-1)
    mu, sig = think_time(model)
    out = np.full(g.shape, _NAN)
    if not (mu == mu and sig == sig):
        return out
    sig = max(sig, 0.05)
    ok = np.isfinite(g) & (g >= 0.0)
    z = (np.log(np.maximum(g[ok], GAP_MIN_S)) - mu) / sig
    out[ok] = -0.5 * z * z - math.log(sig) - 0.5 * math.log(2.0 * math.pi)
    return out


def dispersion(model: Any, daypart: Optional[int] = None) -> float:
    """Overdispersion phi >= 1 of the recent-window G test: the decayed mean of
    G/df on committed ticks of this daypart, shrunk towards the pooled mean
    (itself shrunk towards PHI_PRIOR). Bursty, correlated gaps make histogram
    counts overdispersed relative to the multinomial, so G/phi, not G, is
    compared with chi2_df."""
    s = state_of(model)
    if s is None:
        return PHI_PRIOR
    d = s[DISP:DISP + 2 * N_DAYPARTS].reshape(N_DAYPARTS, 2)
    wp, sp = float(d[:, 0].sum()), float(d[:, 1].sum())
    phi_pool = (sp + PHI_PRIOR_W * PHI_PRIOR) / (wp + PHI_PRIOR_W)
    if daypart is None or not (0 <= int(daypart) < N_DAYPARTS):
        return max(1.0, phi_pool)
    w, sm = float(d[int(daypart), 0]), float(d[int(daypart), 1])
    return max(1.0, (sm + PHI_PRIOR_W * phi_pool) / (w + PHI_PRIOR_W))


def _live(model: Any) -> Mapping:
    if not isinstance(model, Mapping):
        return {}
    lv = model.get("live")
    return lv if isinstance(lv, Mapping) else {}


def period(model: Any, now: Optional[float] = None) -> Tuple[float, float]:
    """(period_s, period_p) of the last strict-period check; period is NaN when
    not confirmed. With `now`, a check older than PERIOD_STALE_S reads (NaN, NaN)."""
    pr = _live(model).get("period")
    if not isinstance(pr, Mapping):
        return _NAN, _NAN
    ts = float(pr.get("ts", _NAN))
    if now is not None and not (ts == ts and float(now) - ts <= PERIOD_STALE_S):
        return _NAN, _NAN
    return float(pr.get("period", _NAN)), float(pr.get("p", _NAN))


def recent(model: Any) -> Dict[str, float]:
    """The last behavior.timing dict B11 published for this entity ({} if none)."""
    r = _live(model).get("recent")
    return dict(r) if isinstance(r, Mapping) else {}


def descriptors(model: Any, now: Optional[float] = None) -> Dict[str, float]:
    """Long-term (committed) timing profile in natural units:
    B, M, think_mu, think_sigma (ln s), think_median_s, gap_p10_s, gap_p50_s,
    gap_p90_s, n_eff, period, period_p. Undefined entries are NaN."""
    mu, sig = think_time(model)
    per, pp = period(model, now)
    q10, q50, q90 = quantiles(model, (0.10, 0.50, 0.90)).tolist()
    return {
        "B": burstiness(model),
        "M": memory(model),
        "think_mu": mu,
        "think_sigma": sig,
        "think_median_s": math.exp(mu) if mu == mu else _NAN,
        "gap_p10_s": q10,
        "gap_p50_s": q50,
        "gap_p90_s": q90,
        "n_eff": n_eff(model),
        "period": per,
        "period_p": pp,
    }


def hist_probs(model: Any) -> np.ndarray:
    """Normalised committed histogram (no smoothing); all NaN when empty."""
    s = state_of(model)
    if s is None or not s[HIST].sum() > 0.0:
        return np.full(N_BINS, _NAN)
    return s[HIST] / float(s[HIST].sum())


def as_float_list(x: Sequence[float], nd: int = 4) -> list:
    """JSON-friendly rounded list (NaN -> None) for profile / portrait payloads."""
    return [round(float(v), nd) if math.isfinite(float(v)) else None for v in x]
