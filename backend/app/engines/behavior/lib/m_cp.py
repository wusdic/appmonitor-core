"""model.cp accessors and pure changepoint step functions (owner: B14 ChangepointEngine).

Why this module holds the maths and not only getters: the CUSUM bank and the
Crosier MCUSUM are stateful, and three consumers must reproduce them exactly
(B29 counterfactual replay, B28 onset audits, the unit tests that measure
wall-clock false-alarm rates over thousands of simulated entity-days). So the
step functions live here, pure and batch-vectorised (any leading batch
shape), and B14 itself calls the very same code. No consumer ever mutates
model.cp.

Statistics (docs/lib3/engines.md B14):
  * bank: 12 KEY_FEATURES x 2 sides x k in {0.25, 1.0} = 48 one-sided CUSUMs
    on psi = clip(AR(1)-prewhitened zr, -3, 3); chart c = side*24 + kidx*12 + f
    (side 0 upper, 1 lower). Each chart has ARL 2400 days (family 0.02 per
    entity-day), h = seq.h_gauss(k, arl_ticks(2400, dt)) x the audit multiplier.
  * mcusum: Crosier MCUSUM (k = 0.5) on the whitened 12-vector W psi
    (W = chol(Sigma)^-1, missing dims imputed by conditional expectation),
    h = seq.mcusum_h(12, arl_ticks(100, dt)).
  * latch / onset / reset (shared by both): an alarm latches when a statistic
    crosses h; tau-hat starts from the last tick at which the alarmed
    statistic was at its zero level (0 for a CUSUM; the radial null
    equilibrium (d-1)/(2k) for Crosier's statistic, which is never 0 in 12
    dimensions) and is refined by the step-change MLE over [that tick, now]
    on the alarmed input kept in a 128-tick ring (mle_onset: the chart's
    signed psi from its last zero on; for the MCUSUM, whose zero level is
    only nominal, the direction-free multivariate step MLE over the whole
    ring). The statistics
    reset after 2 (t_alarm - tau-hat) of clean time, a clean tick being one at
    which no alarmed chart increased; any alarmed chart reaching a new peak
    restarts the clean clock, so a persisting shift never resets.
  * State values are rounded to float32 after every tick, so the
    behavior.cusum_state ring (float32) replays bit-identically.

behavior.cusum_state row layout (float32[84]):
  [0:48] bank S, [48:60] MCUSUM state vector, [60:72] zr_prev (key features),
  [72:84] phi used at that tick.

Consumer API (all pure reads; NaN / None when B14 has not run):
  get(store, s, e) -> dict | None                       the raw model.cp
  onset(store, s, e, at=None) -> float                  tau-hat of the episode (NaN = none; exact from model.cp)
  level(store, s, e) -> {cusum, mcusum: S/h}            accumulator level (>= 1 alarms, B28 SUSPECT at 0.5)
  prob(store, s, e) -> float                            latest behavior.cp.prob, P(r <= 3 h)
  alarms(store, s, e) -> {detector: 0|1}                latched accumulator alarms
  descriptor(store, s, e) -> dict                       portrait summary (phi, h_mult, creep, episode)
  replay_state(store, s, e, at_or_before) -> (ts, state) | None   snapshot from behavior.cusum_state
  replay_params(store, s, e, dt=None) -> dict           h, W, Sigma, dt for step_fn
  step_fn(params, which='cusum'|'mcusum') -> StepFn     (state, inputs) -> (state, score >= 1 alarms)
  neutralize(inputs, features) -> inputs                zr of the given feature indices -> 0
  replay_inputs(store, s, e, since, until) -> [(ts, {name: row})]
  replay_rows(store, s, e, since, until, dt=None) -> [(ts, {x, phi, S, mc, adjacent})]
      the bank's OWN inputs per tick (the key residuals after B14's own-support
      mask, recovered from the state row's zr_prev block) for B29
  replay_step(params) -> StepFn           exact B14 tick on replay_rows inputs
      (adjacency gaps, per-tick phi, the recorded resets); score = max S/h,
      state['mc_ratio'] = ||S_mc|| / h_mc
  mark_resets(params, state0, rows, tol=1e-4) -> (rows, info)
      annotate the ticks where B14 restarted / zeroed its charts (control
      release / rebase, latch reset, end of warm-up) so a replay follows
      them, and recover each tick's whitened MCUSUM input (mc_input)
  mc_input(S_prev, S_new, k) -> w         inverse of one Crosier step
  neutralize_key(x12, features, values=None) -> x12   key residuals of the
      given feature indices (0..51) set to `values` (default 0), NaN kept
Engine/test API: bank_h, mcusum_h, new_bank, bank_tick, new_mc, mc_whiten,
mc_tick, peq, mcusum_p, siegmund_arl, whitener, audit_rate, state_row.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import robustcov, seq
from .features import FEATURE_DIM, FEATURE_GROUP, KEY_FEATURE_IDX, KEY_FEATURES

MODEL = "model.cp"
ZR = "behavior.zr"
CP_PROB = "behavior.cp.prob"
CP_ONSET = "behavior.cp.onset"
CUSUM_STATE = "behavior.cusum_state"

N_KEY = len(KEY_FEATURES)                  # 12
KEY_IDX = np.asarray(KEY_FEATURE_IDX, dtype=np.intp)
KEY_GROUPS: List[str] = [FEATURE_GROUP[n] for n in KEY_FEATURES]
KS: Tuple[float, float] = (0.25, 1.0)
N_CHARTS = 2 * len(KS) * N_KEY             # 48
CHART_SIDE = np.repeat([1.0, -1.0], len(KS) * N_KEY)
CHART_K = np.tile(np.repeat(np.asarray(KS), N_KEY), 2)
CHART_FEAT = np.tile(np.arange(N_KEY), 2 * len(KS))
for _a in (CHART_SIDE, CHART_K, CHART_FEAT):
    _a.setflags(write=False)

FAMILY_RATE = 0.02                         # bank false alarms per entity-day
ARL_DAYS = N_CHARTS / FAMILY_RATE          # 2400 d per chart
PSI_CLIP = seq.PSI_CLIP
RHO2 = 2.0 * seq.SIEGMUND_RHO              # b = h + 1.166
MC_K = seq.MCUSUM_K
MC_D = N_KEY
MC_ARL_DAYS = 100.0
MC_ZERO = (MC_D - 1) / (2.0 * MC_K)        # radial equilibrium of Crosier's stat under the null
HMULT_MAX = 1.5
STATE_DIM = N_CHARTS + 3 * N_KEY           # 84
AUDIT_BLOCK = 30
AUDIT_ARL = 300.0                          # audit level: many alarms per bootstrap sample
ONSET_HIST = 128                           # ticks of inputs kept for the onset MLE

_H_CACHE: Dict[Tuple[str, float], Any] = {}


def _f32(x: np.ndarray) -> np.ndarray:
    """Round to float32 and back: live state == stored ring row, bit for bit."""
    return np.asarray(x, dtype=np.float32).astype(np.float64)


# ============================================================ thresholds
def bank_h(dt: float) -> np.ndarray:
    """Per-chart threshold h[48] at cadence dt (s) before the audit multiplier:
    Siegmund h for ARL 2400 d, e.g. 19.4 / 5.35 at 900 s. Cached per dt."""
    key = ("bank", float(dt))
    h = _H_CACHE.get(key)
    if h is None:
        arl = seq.arl_ticks(ARL_DAYS, dt)
        hk = {k: seq.h_gauss(k, arl) for k in KS}
        h = np.array([hk[k] for k in CHART_K], dtype=np.float64)
        h.setflags(write=False)
        _H_CACHE[key] = h
    return h


def mcusum_h(dt: float) -> float:
    """Crosier MCUSUM threshold for d = 12 at ARL 100 d (seq.MCUSUM_H table)."""
    key = ("mc", float(dt))
    h = _H_CACHE.get(key)
    if h is None:
        h = _H_CACHE[key] = seq.mcusum_h(MC_D, seq.arl_ticks(MC_ARL_DAYS, dt))
    return float(h)


def siegmund_arl(k: float, h: np.ndarray) -> np.ndarray:
    """Zero-state ARL (ticks) of a N(0,1) CUSUM, (e^{2kb} - 2kb - 1)/(2k^2), b = h + 1.166."""
    y = 2.0 * float(k) * (np.asarray(h, dtype=np.float64) + RHO2)
    return (np.expm1(y) - y) / (2.0 * float(k) ** 2)


def peq(S: np.ndarray, n_charts: int = N_CHARTS) -> np.ndarray:
    """Per-chart equivalent p = min(1, 48 exp(-2k(S + 0.583)))."""
    return seq.cusum_stationary_p(S, CHART_K, n_charts)


def mcusum_p(stat: float, d: int = MC_D) -> float:
    """Equivalent per-tick p of a Crosier statistic: 1/ARL(h = stat), inverting
    the seq.MCUSUM_H row linearly in ln ARL (the inverse of seq.mcusum_h);
    clipped to [1e-300, 1]. NaN -> NaN."""
    stat = float(stat)
    if math.isnan(stat):
        return math.nan
    row = seq.MCUSUM_H[min(max(int(d), 1), seq.MCUSUM_H.shape[0]) - 1]
    g = np.log(np.asarray(seq.MCUSUM_ARL_GRID))
    j = int(np.searchsorted(row, stat, side="right")) - 1
    j = min(max(j, 0), row.size - 2)
    slope = (g[j + 1] - g[j]) / (row[j + 1] - row[j])
    ln_arl = g[j] + slope * (stat - row[j])
    return float(min(1.0, max(seq.P_FLOOR, math.exp(-ln_arl))))


# ============================================================ latch
def _new_latch(batch: Tuple[int, ...], C: int) -> Dict[str, np.ndarray]:
    return {
        "on": np.zeros(batch, dtype=bool),
        "t_alarm": np.full(batch, np.nan),
        "onset": np.full(batch, np.nan),
        "span": np.zeros(batch),
        "alarmed": np.zeros(batch + (C,), dtype=bool),
        "peak": np.zeros(batch + (C,)),
        "clean": np.zeros(batch),
    }


def _latch_tick(L: Dict[str, np.ndarray], S: np.ndarray, S_old: np.ndarray, h: np.ndarray,
                zts: np.ndarray, now: float, dt: float
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Advance the alarm latch in place. S, S_old, h, zts: (..., C).
    Returns (rise, reset): an alarm opened this tick / the episode ended
    (the caller zeroes its statistics where reset)."""
    over = S >= h
    on = L["on"]
    if not on.any() and not over.any():            # the null steady state: nothing to do
        z = np.zeros_like(on)
        return z, z
    new = ~on & over.any(-1)
    reset = np.zeros_like(on)
    if on.any():
        old = on.copy()
        alarmed = L["alarmed"] | (over & old[..., None])
        grew = (alarmed & (S > L["peak"])).any(-1)
        incr = (alarmed & (S > S_old)).any(-1)
        L["alarmed"] = alarmed
        L["peak"] = np.where(alarmed, np.maximum(L["peak"], S), L["peak"])
        L["clean"] = np.where(old & grew, 0.0, np.where(old & ~incr, L["clean"] + dt, L["clean"]))
        reset = old & (L["clean"] >= 2.0 * L["span"])
    if new.any():
        ratio = np.where(over, S / np.where(h > 0, h, 1.0), -np.inf)
        j = np.argmax(ratio, axis=-1)
        tau = np.take_along_axis(zts, j[..., None], axis=-1)[..., 0]
        tau = np.where(np.isfinite(tau), tau, now)
        L["t_alarm"] = np.where(new, now, L["t_alarm"])
        L["onset"] = np.where(new, tau, L["onset"])
        L["span"] = np.where(new, np.maximum(now - tau, dt), L["span"])
        L["alarmed"] = np.where(new[..., None], over, L["alarmed"])
        L["peak"] = np.where(new[..., None], S, L["peak"])
        L["clean"] = np.where(new, 0.0, L["clean"])
        L["on"] = on | new
    if reset.any():
        L["on"] = L["on"] & ~reset
        L["alarmed"] = np.where(reset[..., None], False, L["alarmed"])
        L["peak"] = np.where(reset[..., None], 0.0, L["peak"])
        L["clean"] = np.where(reset, 0.0, L["clean"])
        L["t_alarm"] = np.where(reset, np.nan, L["t_alarm"])
        L["onset"] = np.where(reset, np.nan, L["onset"])
        L["span"] = np.where(reset, 0.0, L["span"])
    return new, reset


# ============================================================ onset
def _new_hist(batch: Tuple[int, ...]) -> Dict[str, Any]:
    """Trailing ring of the last ONSET_HIST inputs (shared ts: a batch ticks together)."""
    return {"ts": np.full(ONSET_HIST, np.nan), "y": np.full(batch + (ONSET_HIST, N_KEY), np.nan),
            "pos": 0}


def _hist_push(H: Dict[str, Any], v: np.ndarray, now: float) -> None:
    i = int(H["pos"]) % ONSET_HIST
    H["y"][..., i, :] = v
    H["ts"][i] = now
    H["pos"] = int(H["pos"]) + 1


def _hist_order(H: Mapping[str, Any]) -> np.ndarray:
    n = int(H["pos"])
    if n <= ONSET_HIST:
        return np.arange(n)
    return (n + np.arange(ONSET_HIST)) % ONSET_HIST


def mle_onset(y: np.ndarray, ts: np.ndarray, t_lo: float) -> float:
    """Refined tau-hat: the last pre-change tick of the most likely step in y
    (rows in ts order), searched from t_lo on (so tau-hat >= t_lo).

    y 1-D (a CUSUM chart's signed input): argmax_j (sum_{i>j} y_i)^2 / m_j
    over upward steps (sum > 0). y 2-D [n, d] (the MCUSUM's whitened vectors):
    argmax_j ||sum_{i>j} y_i||^2 / m_j, the direction-free multivariate step
    MLE (projecting on the MCUSUM direction instead is biased late, because
    the direction was fitted to the same data).

    Why: 'the last tick the alarmed statistic was 0' is Page's estimate, and
    it is biased early whenever a null excursion was still open when the
    change began (common for k = 0.25; Crosier's 12-dim statistic has no true
    zero at all). When t_lo precedes the buffer, 'every buffered tick is
    post-change' is a candidate and returns t_lo itself. NaN entries count 0
    (a row with any NaN is not counted in m).
    """
    Y = np.asarray(y, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)
    lo = math.isfinite(t_lo)
    if Y.shape[0] < 2:
        return float(t_lo) if lo else math.nan
    vec = Y.ndim == 2
    fin = np.isfinite(Y).all(axis=1) if vec else np.isfinite(Y)
    Yz = np.where(np.isfinite(Y), Y, 0.0)
    suf = np.cumsum(Yz[::-1], axis=0)[::-1]                       # sum_{i>=j}
    total = suf[0]
    post = np.concatenate((suf[1:], np.zeros_like(suf[:1])), axis=0)   # sum_{i>j}
    m = np.concatenate((np.cumsum(fin[::-1])[::-1][1:], [0])).astype(np.float64)
    if vec:
        num = np.sum(post * post, axis=1)
        tot2 = float(np.sum(total * total))
    else:
        num = np.where(post > 0.0, post * post, 0.0)
        tot2 = float(total * total) if total > 0.0 else 0.0
    stat = np.where(m > 0, num / np.maximum(m, 1.0), 0.0)
    if lo:
        stat[ts < t_lo - 1e-6] = 0.0
    best_j = int(np.argmax(stat))
    best = float(stat[best_j])
    n_all = float(fin.sum())
    if lo and t_lo < ts[0] - 1e-6 and n_all > 0 and tot2 > 0.0 and tot2 / n_all >= best:
        return float(t_lo)                             # the change predates the buffer
    if best <= 0.0:
        return float(t_lo) if lo else float(ts[-1])
    return float(ts[best_j])


def _refine_onsets(L: Dict[str, np.ndarray], rise: np.ndarray, H: Mapping[str, Any],
                   series: Callable[[Tuple[int, ...], np.ndarray], np.ndarray],
                   t_lo: np.ndarray, now: float, dt: float) -> None:
    """Replace the latch's last-zero onset by mle_onset for the entities that
    alarmed this tick (rare: a Python loop over them is fine)."""
    order = _hist_order(H)
    ts = H["ts"][order]
    for row in np.argwhere(rise):
        idx = tuple(int(v) for v in row)
        tau = mle_onset(series(idx, order), ts, float(t_lo[idx]))
        if math.isfinite(tau):
            L["onset"][idx] = tau
            L["span"][idx] = max(now - tau, dt)


# ============================================================ CUSUM bank
def new_bank(batch: Tuple[int, ...] = ()) -> Dict[str, Any]:
    """Zero-state bank for a batch of entities / simulated series."""
    b = tuple(batch)
    return {
        "S": np.zeros(b + (N_CHARTS,)),
        "zts": np.full(b + (N_CHARTS,), np.nan),     # last tick each chart was 0
        "prev": np.full(b + (N_KEY,), np.nan),       # zr_{t-1} of the key features
        "latch": _new_latch(b, N_CHARTS),
        "hist": _new_hist(b),                        # psi, for the onset MLE
    }


def prewhiten(x: np.ndarray, prev: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """psi = clip((x - phi prev)/sqrt(1 - phi^2), -3, 3); NaN prev -> x; NaN x -> NaN.
    phi == 0 everywhere is the identity (bit-identical to seq.prewhiten), skipped."""
    ph = np.asarray(phi, dtype=np.float64)
    if not (ph > 0).any():
        return np.clip(np.asarray(x, dtype=np.float64), -PSI_CLIP, PSI_CLIP)
    return np.clip(seq.prewhiten(x, prev, ph), -PSI_CLIP, PSI_CLIP)


def bank_tick(st: Dict[str, Any], x: np.ndarray, now: float, dt: float, phi: np.ndarray,
              h: np.ndarray, adjacent: Any = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """One tick of the 48-chart bank (in place on st; returns (st, out)).

    x: zr of the 12 key features (..., 12), NaN = unobserved (contributes 0,
    no reset). phi: AR(1) coefficients (12,) or (..., 12) at THIS cadence.
    h: thresholds (48,) or (..., 48) including the audit multiplier.
    adjacent False (scalar or (...,)) drops zr_{t-1} (gap: no whitening).
    out: psi, rise (alarm opened), reset (episode ended; S zeroed), on (latched).
    """
    x = np.asarray(x, dtype=np.float64)
    prev = st["prev"]
    if adjacent is not True:
        prev = np.where(np.asarray(adjacent)[..., None], prev, np.nan)
    psi = prewhiten(x, prev, phi)
    S_old = st["S"]
    S = _f32(seq.cusum_step(S_old, psi[..., CHART_FEAT] * CHART_SIDE, CHART_K))
    zts = np.where(S <= 0.0, now, st["zts"])
    H = st.get("hist")
    if H is None:
        H = st["hist"] = _new_hist(np.shape(S)[:-1])
    _hist_push(H, psi, now)
    rise, reset = _latch_tick(st["latch"], S, S_old, h, zts, now, dt)
    if rise.any():                           # tau-hat: MLE on the alarmed chart's own inputs
        hb = np.broadcast_to(h, S.shape)
        c = np.argmax(np.where(S >= hb, S / np.where(hb > 0, hb, 1.0), -np.inf), axis=-1)
        _refine_onsets(st["latch"], rise, H,
                       lambda i, o: CHART_SIDE[c[i]] * H["y"][i][o, CHART_FEAT[c[i]]],
                       np.take_along_axis(zts, c[..., None], axis=-1)[..., 0], now, dt)
    if reset.any():
        S = np.where(reset[..., None], 0.0, S)
        zts = np.where(reset[..., None], now, zts)
    st["S"], st["zts"], st["prev"] = S, zts, x.copy()
    return st, {"psi": psi, "rise": rise, "reset": reset, "on": st["latch"]["on"]}


# ============================================================ MCUSUM
def new_mc(batch: Tuple[int, ...] = ()) -> Dict[str, Any]:
    b = tuple(batch)
    return {"S": np.zeros(b + (MC_D,)), "stat": np.zeros(b), "zts": np.full(b, np.nan),
            "latch": _new_latch(b, 1), "hist": _new_hist(b)}


def whitener(Sigma: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """(Sigma_corr, W) for the key-feature correlation: Sigma is rescaled to
    unit diagonal, eigen-floored, W = chol(Sigma)^-1. None / invalid -> (I, I)."""
    eye = np.eye(MC_D)
    if Sigma is None:
        return eye, eye
    A = np.asarray(Sigma, dtype=np.float64)
    if A.shape != (MC_D, MC_D) or not np.isfinite(A).all():
        return eye, eye
    d = np.sqrt(np.clip(np.diag(A), 0.0, None))
    if not (d > 0).all():
        return eye, eye
    C = robustcov.eigen_floor(A / np.outer(d, d))
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        return eye, eye
    W = np.linalg.solve(L, eye)
    return C, W


def mc_whiten(psi: np.ndarray, W: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """W psi for one 12-vector; missing dims are imputed by their conditional
    expectation first (so the whitened norm equals the observed-dims T^2).
    All-NaN -> all-NaN (the caller skips the MCUSUM update)."""
    psi = np.asarray(psi, dtype=np.float64)
    obs = np.isfinite(psi)
    if obs.all():
        return W @ psi
    if not obs.any():
        return np.full(MC_D, np.nan)
    return W @ robustcov.conditional_impute(psi, Sigma, obs)


def mc_tick(st: Dict[str, Any], w: np.ndarray, now: float, dt: float, h: float
            ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """One Crosier step on the whitened vector w (NaN entries count 0), with
    the shared latch (zero level = MC_ZERO). In place; returns (st, out)."""
    S_old = st["stat"]
    S_vec, stat = seq.mcusum_step(st["S"], w, MC_K)
    S_vec = _f32(S_vec)
    stat = np.sqrt(np.sum(S_vec * S_vec, axis=-1))
    zts = np.where(stat <= MC_ZERO, now, st["zts"])
    H = st.get("hist")
    if H is None:
        H = st["hist"] = _new_hist(np.shape(stat))
    _hist_push(H, np.where(np.isfinite(w), w, 0.0), now)
    hh = np.asarray(h, dtype=np.float64)
    rise, reset = _latch_tick(st["latch"], stat[..., None], np.asarray(S_old)[..., None],
                              np.broadcast_to(hh, np.shape(stat) + (1,)),
                              np.asarray(zts)[..., None], now, dt)
    if rise.any():                           # tau-hat: multivariate step MLE on the buffer
        _refine_onsets(st["latch"], rise, H, lambda i, o: H["y"][i][o],
                       np.full(np.shape(stat), -np.inf), now, dt)
    if reset.any():
        S_vec = np.where(reset[..., None], 0.0, S_vec)
        stat = np.where(reset, 0.0, stat)
        zts = np.where(reset, now, zts)
    st["S"], st["stat"], st["zts"] = S_vec, stat, zts
    return st, {"rise": rise, "reset": reset, "on": st["latch"]["on"]}


# ============================================================ audit
def lindley(y: np.ndarray) -> np.ndarray:
    """Zero-state CUSUM path S_t = max(0, S_{t-1} + y_t) along axis 0 in closed
    form: S_t = C_t - min(0, min_{s<=t} C_s), C = cumsum(y). NaN y counts 0."""
    C = np.cumsum(np.where(np.isnan(y), 0.0, y), axis=0)
    return C - np.minimum.accumulate(np.minimum(C, 0.0), axis=0)


def _arl_emp(S: np.ndarray, h: float) -> Tuple[float, float]:
    """(in-control ticks, alarms) of zero-state runs in Lindley paths S [T, ...]:
    each excursion from 0 that reaches h is one alarm; the ticks from its first
    crossing to the next zero are dropped (a real chart restarts at 0 there)."""
    g = np.cumsum(S <= 1e-12, axis=0)                   # excursion id
    m = np.maximum.accumulate(np.where(S >= h, g, -1), axis=0)
    post = m == g
    n_alarm = float(post[0].sum() + (post[1:] & ~post[:-1]).sum())
    return float((~post).sum()), n_alarm


def audit_rate(x: np.ndarray, gap: np.ndarray, phi: np.ndarray, dt: float, hmult: float,
               rng: np.random.Generator, n_series: int = 8, block: int = AUDIT_BLOCK
               ) -> Dict[str, float]:
    """Realised bank false-alarm rate per entity-day on committed residuals.

    x: committed zr key rows [N, 12] (ts order); gap[N]: True where row i does
    not follow row i-1 (no whitening across it). Moving-block bootstrap
    (blocks of 30 rows, n_series series of N rows) of psi; every chart is run
    in closed form (lindley) at the audit level ARL 300, where alarms are
    plentiful, and the realised / Siegmund ratio r_k is extrapolated to the
    operating threshold as r_k^((h + 1.166)/(h_a + 1.166)) (an exponential-
    tail misfit compounds linearly in b). Returns {'rate', 'r0.25', 'r1.0', 'n'}.
    """
    X = np.asarray(x, dtype=np.float64)
    N = X.shape[0]
    if N < 2 * block:
        return {"rate": math.nan, "n": float(N)}
    prev = np.vstack([np.full((1, N_KEY), np.nan), X[:-1]])
    prev[np.asarray(gap, dtype=bool)] = np.nan
    psi = prewhiten(X, prev, phi)
    nb = int(math.ceil(N / block))
    starts = rng.integers(0, N - block + 1, size=(n_series, nb))
    idx = (starts[..., None] + np.arange(block)).reshape(n_series, -1)[:, :N]
    boot = psi[idx.T]                                   # [N, n_series, 12]
    h_op = bank_h(dt) * float(hmult)
    rate = 0.0
    out: Dict[str, float] = {"n": float(N)}
    for k in KS:
        h_a = seq.h_gauss(k, AUDIT_ARL)
        up = lindley(boot - k)
        lo = lindley(-boot - k)
        t_up, a_up = _arl_emp(up, h_a)
        t_lo, a_lo = _arl_emp(lo, h_a)
        arl_emp = (t_up + t_lo) / max(a_up + a_lo, 0.5)
        ratio = AUDIT_ARL / arl_emp
        hk = float(h_op[np.flatnonzero(CHART_K == k)[0]])
        mult = ratio ** ((hk + RHO2) / (h_a + RHO2))
        # design rate of these 24 charts at the operating h, inflated by the misfit
        rate += 2 * N_KEY / float(siegmund_arl(k, hk)) * mult * seq.SECONDS_PER_DAY / dt
        out[f"r{k}"] = float(ratio)
    out["rate"] = float(rate)
    return out


# ============================================================ ring row / replay
def state_row(bank: Mapping[str, Any], mc: Mapping[str, Any], phi: np.ndarray) -> np.ndarray:
    """behavior.cusum_state row (float32[84]) for one entity."""
    return np.concatenate([np.asarray(bank["S"], dtype=np.float64).reshape(-1),
                           np.asarray(mc["S"], dtype=np.float64).reshape(-1),
                           np.asarray(bank["prev"], dtype=np.float64).reshape(-1),
                           np.asarray(phi, dtype=np.float64).reshape(-1)]).astype(np.float32)


def split_state_row(row: np.ndarray) -> Dict[str, np.ndarray]:
    r = np.asarray(row, dtype=np.float64).reshape(-1)
    return {"S": r[:N_CHARTS].copy(), "mc": r[N_CHARTS:N_CHARTS + MC_D].copy(),
            "prev": r[N_CHARTS + MC_D:N_CHARTS + MC_D + N_KEY].copy(),
            "phi": r[N_CHARTS + MC_D + N_KEY:STATE_DIM].copy()}


def replay_state(store: Any, s: str, e: str, at_or_before: float
                 ) -> Optional[Tuple[float, Dict[str, np.ndarray]]]:
    """(ts, state) of the newest behavior.cusum_state row with ts <= at_or_before
    (the ring keeps 6 h), or None."""
    t, M = store.vec_range(s, e, CUSUM_STATE, -math.inf, float(at_or_before))
    if not len(t):
        return None
    return float(t[-1]), split_state_row(M[-1])


def get(store: Any, s: str, e: str) -> Optional[Dict[str, Any]]:
    return store.get_model(s, e, MODEL)


def replay_params(store: Any, s: str, e: str, dt: Optional[float] = None) -> Dict[str, Any]:
    """Thresholds and whitening of (s, e) as B14 last used them. dt defaults to
    the cadence of B14's last tick."""
    m = get(store, s, e) or {}
    run = m.get("run") or {}
    dt = float(dt if dt is not None else run.get("dt") or 900.0)
    wh = run.get("whiten") or {}
    Sig = wh.get("Sigma")
    W = wh.get("W")
    if Sig is None or W is None:
        Sig, W = whitener(None)
    hm = float(run.get("hmult", 1.0) or 1.0)
    return {"dt": dt, "h": bank_h(dt) * hm, "h_mc": mcusum_h(dt),
            "Sigma": np.asarray(Sig, dtype=np.float64), "W": np.asarray(W, dtype=np.float64)}


def _zr_of(inputs: Any) -> np.ndarray:
    if isinstance(inputs, Mapping):
        inputs = inputs.get(ZR)
    return np.asarray(inputs, dtype=np.float64).reshape(-1)


def step_fn(params: Mapping[str, Any], which: str = "cusum"
            ) -> Callable[[Dict[str, np.ndarray], Any], Tuple[Dict[str, np.ndarray], float]]:
    """StepFn for lib.replay: (state, inputs) -> (state, score). inputs is a zr
    row (52 or 12 values) or {'behavior.zr': row, 'behavior.cusum_state': row};
    the phi recorded in a cusum_state input is used when present (it is the
    phi B14 applied at that tick). score = max_c S_c / h_c ('cusum') or
    ||S_mc|| / h_mc ('mcusum'); >= 1 means the live chart alarmed. Consecutive
    inputs are treated as adjacent ticks. Latch resets are not replayed."""
    if which not in ("cusum", "mcusum"):
        raise ValueError(f"m_cp.step_fn: unknown statistic {which!r}")
    h = np.asarray(params["h"], dtype=np.float64)
    h_mc = float(params["h_mc"])
    W = np.asarray(params["W"], dtype=np.float64)
    Sig = np.asarray(params["Sigma"], dtype=np.float64)

    def step(state: Dict[str, np.ndarray], inputs: Any) -> Tuple[Dict[str, np.ndarray], float]:
        zr = _zr_of(inputs)
        x = zr[KEY_IDX] if zr.size == FEATURE_DIM else zr
        phi = state["phi"]
        if isinstance(inputs, Mapping) and inputs.get(CUSUM_STATE) is not None:
            phi = split_state_row(inputs[CUSUM_STATE])["phi"]
        psi = prewhiten(x, state["prev"], phi)
        S = _f32(seq.cusum_step(state["S"], psi[CHART_FEAT] * CHART_SIDE, CHART_K))
        w = mc_whiten(psi, W, Sig)
        mc = state["mc"]
        if np.isfinite(w).any():
            mc, _ = seq.mcusum_step(mc, w, MC_K)
            mc = _f32(mc)
        new = {"S": S, "mc": mc, "prev": x.copy(), "phi": phi}
        if which == "cusum":
            return new, float(np.max(S / h))
        return new, float(np.sqrt(np.sum(mc * mc)) / h_mc)

    return step


def neutralize(inputs: Any, features: Sequence[int]) -> Any:
    """Counterfactual input: zr of the given feature indices (0..51) set to 0."""
    idx = list(features)
    if isinstance(inputs, Mapping):
        out = dict(inputs)
        zr = np.array(out[ZR], dtype=np.float64, copy=True)
        zr[idx] = 0.0
        out[ZR] = zr
        return out
    zr = np.array(inputs, dtype=np.float64, copy=True)
    zr[idx] = 0.0
    return zr


def replay_inputs(store: Any, s: str, e: str, since: float, until: float
                  ) -> List[Tuple[float, Dict[str, np.ndarray]]]:
    """[(ts, {'behavior.zr': row, 'behavior.cusum_state': row|None})] for since < ts <= until."""
    t, Z = store.vec_range(s, e, ZR, float(since), float(until))
    tc, C = store.vec_range(s, e, CUSUM_STATE, float(since), float(until))
    cs = {float(a): C[i] for i, a in enumerate(tc)}
    return [(float(ts), {ZR: Z[i].astype(np.float64), CUSUM_STATE: cs.get(float(ts))})
            for i, ts in enumerate(t) if ts > since]


def replay_rows(store: Any, s: str, e: str, since: float, until: float,
                dt: Optional[float] = None,
                dt_of: Optional[Callable[[float], float]] = None
                ) -> List[Tuple[float, Dict[str, Any]]]:
    """[(ts, {'x', 'phi', 'S', 'mc', 'adjacent'})] for since < ts <= until from
    the behavior.cusum_state ring alone.

    Why the state ring and not behavior.zr: B14 feeds the bank the key
    residuals of the features its OWN baseline identifies at the bucket
    (others NaN), and writes exactly that input as the row's zr_prev block
    (bank['prev'] = x after the tick). The row therefore holds the true input
    of its tick, float32-exact (zr itself is a float32 ring). 'adjacent'
    follows B14: the previous row is at most 1.5 dt earlier, dt being the
    cadence of the tick itself (dt_of(ts) when given, e.g. from B24's issued
    stratum record, so a window reaching back over a cadence switch is
    judged as B14 judged it; else `dt`, else B14's last cadence); the first
    row is adjacent to the row at or before `since` when that one is close
    enough."""
    if dt is None:
        run = ((get(store, s, e) or {}).get("run") or {})
        dt = float(run.get("dt") or 900.0)
    t, M = store.vec_range(s, e, CUSUM_STATE, -math.inf, float(until))
    out: List[Tuple[float, Dict[str, Any]]] = []
    prev_ts = -math.inf
    for i, ts in enumerate(t):
        ts = float(ts)
        if ts > float(since):
            r = split_state_row(M[i])
            d_t = float(dt)
            if dt_of is not None:
                v = dt_of(ts)
                if v is not None and v == v and v > 0:
                    d_t = float(v)
            out.append((ts, {"x": r["prev"], "phi": r["phi"], "S": r["S"], "mc": r["mc"],
                             "adjacent": bool(ts - prev_ts <= 1.5 * d_t + 1e-6)}))
        prev_ts = ts
    return out


def _replay_tick(state: Mapping[str, np.ndarray], inp: Mapping[str, Any], h: np.ndarray,
                 h_mc: float, W: np.ndarray, Sig: np.ndarray) -> Dict[str, np.ndarray]:
    mode = inp.get("mode") or ()
    if "restart" in mode:                     # charts restarted before the tick (new_bank)
        state = {"S": np.zeros(N_CHARTS), "mc": np.zeros(MC_D),
                 "prev": np.full(N_KEY, np.nan), "phi": state["phi"]}
    x = np.asarray(inp["x"], dtype=np.float64).reshape(-1)
    phi = np.asarray(inp.get("phi", state["phi"]), dtype=np.float64)
    prev = state["prev"] if inp.get("adjacent", True) else np.full(N_KEY, np.nan)
    psi = prewhiten(x, prev, phi)
    S = _f32(seq.cusum_step(state["S"], psi[CHART_FEAT] * CHART_SIDE, CHART_K))
    mc = state["mc"]
    if "w" in inp:
        # the recorded whitened input of this tick (mark_resets recovers it
        # from consecutive states: B14's whitener follows model.density
        # refits and is not stored per tick); a counterfactual input moves
        # it by the change of the whitened residual under today's whitener
        w = inp["w"]
        if w is not None:
            w = np.asarray(w, dtype=np.float64)
            psi_f = inp.get("psi")
            if psi_f is not None and not np.array_equal(psi, psi_f, equal_nan=True):
                d = (np.nan_to_num(mc_whiten(psi, W, Sig))
                     - np.nan_to_num(mc_whiten(psi_f, W, Sig)))
                w = w + d
            mc, _ = seq.mcusum_step(mc, w, MC_K)
            mc = _f32(mc)
    else:
        w = mc_whiten(psi, W, Sig)
        if np.isfinite(w).any():
            mc, _ = seq.mcusum_step(mc, w, MC_K)
            mc = _f32(mc)
    if "zero_S" in mode:                      # the bank latch reset zeroed its charts
        S = np.zeros(N_CHARTS)
    if "zero_mc" in mode:                     # the MCUSUM latch reset (independent)
        mc = np.zeros(MC_D)
    return {"S": S, "mc": np.asarray(mc, dtype=np.float64), "prev": x.copy(), "phi": phi}


def replay_step(params: Mapping[str, Any]
                ) -> Callable[[Dict[str, Any], Mapping[str, Any]], Tuple[Dict[str, Any], float]]:
    """StepFn over replay_rows inputs: one B14 bank + MCUSUM tick (prewhitening
    with the tick's recorded phi, no whitening across a gap, float32 state
    rounding), honouring inp['mode']: a tuple of 'restart' (new bank before
    the tick), 'zero_S' / 'zero_mc' (a latch reset after it), or 'resync'
    (adopt the recorded state; mark_resets sets it only where nothing else
    reproduces B14). Returns score = max_c S_c / h_c; the
    MCUSUM ratio ||S_mc|| / h_mc is left in state['mc_ratio']."""
    h = np.asarray(params["h"], dtype=np.float64)
    h_mc = float(params["h_mc"])
    W = np.asarray(params["W"], dtype=np.float64)
    Sig = np.asarray(params["Sigma"], dtype=np.float64)

    def step(state: Dict[str, Any], inp: Mapping[str, Any]) -> Tuple[Dict[str, Any], float]:
        if "resync" in (inp.get("mode") or ()):
            new = {"S": np.asarray(inp["S"], dtype=np.float64).copy(),
                   "mc": np.asarray(inp["mc"], dtype=np.float64).copy(),
                   "prev": np.asarray(inp["x"], dtype=np.float64).copy(),
                   "phi": np.asarray(inp.get("phi", state["phi"]), dtype=np.float64)}
        else:
            new = _replay_tick(state, inp, h, h_mc, W, Sig)
        new["mc_ratio"] = float(np.sqrt(np.sum(new["mc"] * new["mc"])) / h_mc)
        return new, float(np.max(new["S"] / h))

    return step


def replay_state0(row_or_state: Any) -> Dict[str, np.ndarray]:
    """replay_step state from a cusum_state row (or split_state_row dict)."""
    r = split_state_row(row_or_state) if not isinstance(row_or_state, Mapping) \
        else row_or_state
    return {"S": np.asarray(r["S"], dtype=np.float64).copy(),
            "mc": np.asarray(r["mc"], dtype=np.float64).copy(),
            "prev": np.asarray(r["prev"], dtype=np.float64).copy(),
            "phi": np.asarray(r["phi"], dtype=np.float64).copy(), "mc_ratio": 0.0}


def mc_input(S_prev: np.ndarray, S_new: np.ndarray, k: float = MC_K) -> np.ndarray:
    """The whitened input w of one Crosier step recovered from the states
    before and after it: S' = (S + w)(1 - k/C), ||S'|| = C - k, so
    S + w = S' (||S'|| + k) / ||S'||. When S' = 0 (C <= k) the input is only
    known to lie in a ball; w = -S (C = 0) is returned. A NaN S restarts at 0
    (seq.mcusum_step)."""
    Sp = np.asarray(S_prev, dtype=np.float64)
    if np.isnan(Sp).any():
        Sp = np.zeros_like(Sp)
    Sn = np.asarray(S_new, dtype=np.float64)
    nrm = float(np.sqrt(np.sum(Sn * Sn)))
    if not nrm > 0.0:
        return -Sp
    return Sn * (nrm + float(k)) / nrm - Sp


def _close(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    return bool(np.all(np.abs(np.asarray(a) - np.asarray(b)) <= tol * (1.0 + np.abs(b))))


def mark_resets(params: Mapping[str, Any], state0: Mapping[str, Any],
                rows: Sequence[Tuple[float, Mapping[str, Any]]], tol: float = 1e-4
                ) -> Tuple[List[Tuple[float, Dict[str, Any]]], Dict[str, Any]]:
    """Replay the recorded inputs and compare every tick with the recorded
    state. Where plain replay does not reproduce B14, try a restart before
    the tick (control release / rebase, end of warm-up: new_bank), then a
    reset after it (latch reset: statistics zeroed), else adopt the recorded
    state ('resync'). Returns (rows with 'mode' set, {'n_reset', 'n_resync',
    'max_err'}) where max_err is the largest relative deviation of an
    unannotated tick (0 for a bit-exact replay)."""
    step = replay_step(params)
    st = replay_state0(state0)
    out: List[Tuple[float, Dict[str, Any]]] = []
    n_reset = n_resync = 0
    max_err = 0.0
    prev_x = np.asarray(st["prev"], dtype=np.float64)
    prev_mc = np.asarray(st["mc"], dtype=np.float64)
    for ts, inp in rows:
        d = dict(inp)
        d.pop("mode", None)
        # the MCUSUM's own whitened input, recovered from the recorded states
        xx = np.asarray(d["x"], dtype=np.float64)
        pv = prev_x if d.get("adjacent", True) else np.full(N_KEY, np.nan)
        psi = prewhiten(xx, pv, np.asarray(d.get("phi", st["phi"]), dtype=np.float64))
        d["psi"] = psi
        d["w"] = (mc_input(prev_mc, d["mc"]) if np.isfinite(psi).any()
                  and not np.array_equal(prev_mc, np.asarray(d["mc"], dtype=np.float64))
                  else None)
        prev_x, prev_mc = xx, np.asarray(d["mc"], dtype=np.float64)
        cand, _ = step(st, d)
        rec_S, rec_mc = np.asarray(d["S"]), np.asarray(d["mc"])
        if _close(cand["S"], rec_S, tol) and _close(cand["mc"], rec_mc, tol):
            with np.errstate(invalid="ignore", divide="ignore"):
                err = float(np.max(np.abs(cand["S"] - rec_S) / (1.0 + np.abs(rec_S))))
            max_err = max(max_err, err)
            st = cand
            out.append((ts, d))
            continue
        chosen = None
        for mode in (("restart",), ("zero_S",), ("zero_mc",), ("zero_S", "zero_mc"),
                     ("restart", "zero_S"), ("restart", "zero_mc")):
            d2 = dict(d, mode=mode)
            if "restart" in mode:
                # a new bank: no previous residual to whiten against, and the
                # MCUSUM restarted from 0, so its input is relative to 0
                d2["psi"] = prewhiten(np.asarray(d["x"], dtype=np.float64),
                                      np.full(N_KEY, np.nan),
                                      np.asarray(d.get("phi", st["phi"]), dtype=np.float64))
                if d.get("w") is not None:
                    d2["w"] = mc_input(np.zeros(MC_D), d["mc"])
            c2, _ = step(st, d2)
            if _close(c2["S"], rec_S, tol) and _close(c2["mc"], rec_mc, tol):
                chosen, st = d2, c2
                n_reset += 1
                break
        if chosen is None:
            chosen = dict(d, mode=("resync",))
            st, _ = step(st, chosen)
            n_resync += 1
        out.append((ts, chosen))
    return out, {"n_reset": n_reset, "n_resync": n_resync, "max_err": max_err}


def neutralize_key(x12: Any, features: Sequence[int],
                   values: Optional[Mapping[int, float]] = None) -> np.ndarray:
    """Counterfactual bank input: the key residuals of the given FEATURE
    indices (0..51; non-key features ignored) set to values[f] (default 0 =
    the bucket median's residual). A masked (NaN) input stays NaN: B14 did not
    chart it, so neutralising it changes nothing."""
    x = np.array(x12, dtype=np.float64, copy=True).reshape(-1)
    vals = values or {}
    for f in features:
        j = _KEY_POS.get(int(f))
        if j is not None and math.isfinite(x[j]):
            v = float(vals.get(int(f), 0.0))
            x[j] = v if math.isfinite(v) else 0.0
    return x


_KEY_POS: Dict[int, int] = {int(f): j for j, f in enumerate(KEY_FEATURE_IDX)}


# ============================================================ consumer reads
def _latest1(store: Any, s: str, e: str, name: str) -> float:
    hit = store.vec_latest(s, e, name)
    if hit is None:
        return math.nan
    return float(np.asarray(hit[1]).reshape(-1)[0])


def onset(store: Any, s: str, e: str, at: Optional[float] = None) -> float:
    """tau-hat of the change episode, NaN if none.

    at None: the current episode, exact (float64) from model.cp, falling back
    to the latest behavior.cp.onset row. at given: the behavior.cp.onset row
    written at exactly `at` (NaN if none). Ring rows are float32, which
    quantises epoch seconds to 128 s, so prefer at=None for rollback targets.
    """
    if at is not None:
        row = store.vec_at(s, e, CP_ONSET, float(at))
        return math.nan if row is None else float(np.asarray(row).reshape(-1)[0])
    m = get(store, s, e)
    if isinstance(m, Mapping):
        run = m.get("run") or {}
        if run.get("episode") is not None:
            return float((run.get("episode") or {}).get("onset", math.nan))
    return _latest1(store, s, e, CP_ONSET)


def level(store: Any, s: str, e: str) -> Dict[str, float]:
    """{cusum: max_c S_c / h_c, mcusum: ||S|| / h_mc} as of B14's last tick
    (>= 1 alarms). NaN when B14 has not run. Diagnostic only: on a null the
    max over 48 charts is >= 1/4 on ~88 % and >= 1/2 on ~15 % of ticks at
    900 s, so B27's quiet test and B28's SUSPECT / trust tests use the
    calibrated-p level lib/detectors.acc_level instead."""
    m = get(store, s, e)
    run = (m or {}).get("run") if isinstance(m, Mapping) else None
    if not run or not math.isfinite(float(run.get("dt", math.nan))):
        return {"cusum": math.nan, "mcusum": math.nan}
    dt = float(run["dt"])
    h = bank_h(dt) * float(run.get("hmult", 1.0) or 1.0)
    S = np.asarray((run.get("bank") or {}).get("S", np.zeros(N_CHARTS)), dtype=np.float64)
    stat = float((run.get("mc") or {}).get("stat", 0.0))
    return {"cusum": float(np.max(S / h)), "mcusum": stat / mcusum_h(dt)}


def prob(store: Any, s: str, e: str) -> float:
    """Latest BOCPD P(run length <= 3 h), NaN before the first closed hour."""
    return _latest1(store, s, e, CP_PROB)


def alarms(store: Any, s: str, e: str) -> Dict[str, int]:
    """{cusum, mcusum, bocpd, creep: 0|1} latched state as of B14's last tick."""
    m = get(store, s, e) or {}
    run = m.get("run") or {}
    return {k: int(bool(v)) for k, v in (run.get("alarm") or {}).items()}


def descriptor(store: Any, s: str, e: str) -> Dict[str, Any]:
    """Portrait summary: AR(1) phi per key feature, h multiplier from the
    audit, the current episode and the latest creep verdicts (JSON-safe)."""
    m = get(store, s, e)
    if not m:
        return {}
    run = m.get("run") or {}
    learn = m.get("learn") or {}
    phi = np.asarray(learn.get("phi", np.zeros(N_KEY)), dtype=np.float64)
    ep = run.get("episode") or {}
    return {
        "phi": {n: round(float(v), 3) for n, v in zip(KEY_FEATURES, phi)},
        "h_mult": float(run.get("hmult", 1.0)),
        "audit": dict(run.get("audit") or {}),
        "alarms": alarms(store, s, e),
        "onset": ep.get("onset"),
        "axes": list(ep.get("axes") or []),
        "creep": {g: dict(v) for g, v in ((run.get("creep") or {}).get("groups") or {}).items()},
        "cp_prob": run.get("cp_prob"),
    }
