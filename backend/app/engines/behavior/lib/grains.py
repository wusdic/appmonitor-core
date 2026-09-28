"""Canonical grains, decision clocks, streams, budgets and the scale transfer
(lib-3 v2.1, docs/lib3/cadence.md §2, §6, §7; helpers_api "grains").

Why: every scored feature is defined on a trailing WALL-CLOCK window, not on
"whatever one tick holds". H (3600 s) is the primary grain and is observable
at every supported cadence; Q (900 s) is the sensitivity grain, observable
when the tick is <= 900 s. Grain rows are written, learned and scored only on
DECISION ticks (the tick whose interval contains an epoch multiple of the
grain), the same boundaries for every entity, so consecutive scored rows
never overlap and every sequential chart counts its thresholds in grain
periods instead of ticks.

Pure module: functions of (now, dt, mode, config) only; no store.

Mode. `config['grain_mode']` is 'tick' (v2: G_h := dt, Q never observable,
every grain series name resolves to its v2 name, e_day = q 86400/dt) or
'canonical'. Every function below degenerates to v2 in tick mode, so one
code path serves both (cadence.md D11).
"""
from __future__ import annotations

import datetime as _dt
import math
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import timebins as TB

GRAIN_S: Dict[str, float] = {"h": 3600.0, "q": 900.0}
GRAINS: Tuple[str, ...] = ("h", "q")
M = 4                                   # G_h / G_q
EPS = 1e-6                              # relative slack of `observable`
EPS_S = 1e-3                            # s: slack of epoch-boundary tests
COVER_MIN = 0.95                        # set / map features need cov >= 0.95 G
KAPPA_T = 16.0                          # transfer pseudo-rows in the Q predictive
KAPPA_OMEGA = 24.0                      # pseudo paired-hours of the omega / delta EB shrink
OMEGA_MAX = 2.0 * M                     # omega clip [0, 8] -> v in [1, 7]
C_T_MIN = 5.0                           # floor of a transferred BB concentration
BETA: Dict[str, float] = {"h": 0.5, "q": 0.25, "t": 0.25}     # single-tick budget shares
EVIDENCE_SHARE: Dict[str, float] = {"h": 0.5, "t": 0.5}       # evidence-CUSUM shares
EVIDENCE_ARL_DAYS = 33.0                # the v2 evidence ARL (whole budget)
SPAN_S: Dict[str, float] = {"periodicity": 21600.0, "timing_regularity": 21600.0,
                            "req_per_session": 86400.0, "duty_cycle": 86400.0}
MODES: Tuple[str, ...] = ("tick", "canonical")
TICK, CANONICAL = MODES
TAUS: Tuple[str, ...] = ("h", "q", "t")
DAY = 86400.0


# =================================================================== mode
def mode_of(config: Optional[Mapping[str, Any]]) -> str:
    """config['grain_mode'] ('tick' when absent or unknown)."""
    m = (config or {}).get("grain_mode") if isinstance(config, Mapping) else None
    return CANONICAL if m == CANONICAL else TICK


def canonical(config_or_mode: Any) -> bool:
    if isinstance(config_or_mode, str):
        return config_or_mode == CANONICAL
    return mode_of(config_or_mode) == CANONICAL


def _mode(mode: Any) -> str:
    return mode if isinstance(mode, str) and mode in MODES else mode_of(mode)


# ================================================================= clocks
def grain_s(g: str, dt_s: float, mode: Any = TICK) -> float:
    """Grain length. Tick mode: G_h = dt and Q is undefined (NaN)."""
    if _mode(mode) == TICK:
        return float(dt_s) if g == "h" else math.nan
    return GRAIN_S[g]


def observable(g: str, dt_s: float, mode: Any = TICK) -> bool:
    """dt <= G (1 + EPS). Tick mode: H always, Q never."""
    if _mode(mode) == TICK:
        return g == "h"
    return float(dt_s) <= GRAIN_S[g] * (1.0 + EPS)


def _crosses(now: float, dt: float, G: float, offset: float = 0.0) -> bool:
    """(now - dt, now] contains a multiple of G (shifted by offset)."""
    a = math.floor((now - offset + EPS_S) / G)
    b = math.floor((now - dt - offset + EPS_S) / G)
    return a > b


def decision(now: float, dt_s: float, g: str, mode: Any = TICK) -> bool:
    """The tick's interval (now - dt, now] contains an epoch multiple of G
    (the same boundaries for every entity). Tick mode: H every tick, Q never."""
    md = _mode(mode)
    if md == TICK:
        return g == "h"
    if not observable(g, dt_s, md):
        return False
    return _crosses(float(now), float(dt_s), GRAIN_S[g])


def due(now: float, dt_s: float, mode: Any = TICK) -> Dict[str, bool]:
    return {g: decision(now, dt_s, g, mode) for g in GRAINS}


def last_decision(now: float, dt_s: float, g: str, mode: Any = TICK) -> float:
    """Lower bound of the newest decision tick <= now (the epoch multiple it
    covers), used for staleness: a grain series written at or after it is
    current. Tick mode: now."""
    if _mode(mode) == TICK:
        return float(now)
    G = GRAIN_S[g]
    return math.floor((float(now) + EPS_S) / G) * G


def window(now: float, g: str, dt_s: float, mode: Any = TICK) -> Tuple[float, float]:
    """(lo, hi]: the ticks with ts in (now - G, now]."""
    return float(now) - grain_s(g, dt_s, mode), float(now)


def row_tctx(now: float, g: str, dt_s: float, config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Time context of a grain row: that of its window MIDPOINT now - G/2
    (cadence.md D4). Tick mode: the end-stamp tctx of v2."""
    if not canonical(config):
        return TB.tctx_from_config(float(now), config or {}, float(dt_s))
    key = (float(now), g, float(dt_s), _cfg_key(config))
    tc = _TCTX_CACHE.get(key)
    if tc is None:
        if len(_TCTX_CACHE) > 8192:
            _TCTX_CACHE.clear()
        tc = _TCTX_CACHE[key] = TB.tctx_from_config(float(now) - GRAIN_S[g] / 2.0,
                                                    config or {}, float(dt_s))
    return tc


_TCTX_CACHE: Dict[Tuple, Dict[str, Any]] = {}


def _cfg_key(config: Optional[Mapping[str, Any]]) -> Tuple:
    cfg = config or {}
    cal = cfg.get("calendar") or {}
    return (str(cfg.get("tz") or TB.DEFAULT_TZ),
            tuple(sorted(str(x) for x in (cal.get("holidays") or ()))),
            tuple(sorted(str(x) for x in (cal.get("makeup_workdays") or ()))),
            repr(cfg.get("daypart_day_hours")))


def row_tctx_cfg(now: float, g: str, dt_s: float, config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """row_tctx without the cache (tests)."""
    if not canonical(config):
        return TB.tctx_from_config(float(now), config or {}, float(dt_s))
    return TB.tctx_from_config(float(now) - GRAIN_S[g] / 2.0, config or {}, float(dt_s))


@lru_cache(maxsize=4096)
def _local_offset(ts: float, tz: str) -> float:
    z = TB._zone(tz)
    return float(TB._offset_s(ts, z))


def span_decision(now: float, dt_s: float, feature: str,
                  config: Optional[Mapping[str, Any]]) -> bool:
    """The tick's interval contains a LOCAL-time multiple of the feature's
    span (00/06/12/18 local for 6 h, local midnight for 24 h). Tick mode:
    every tick (v2 scores span features every tick)."""
    if not canonical(config):
        return True
    span = SPAN_S[feature]
    tz = (config or {}).get("tz") or TB.DEFAULT_TZ
    now = float(now)
    lo = now - float(dt_s)
    # candidate boundary: the newest local multiple of span <= now (DST aware:
    # the local wall clock is converted back with the offset in force there)
    off = _local_offset(now, tz)
    loc = now + off
    b_loc = math.floor((loc + EPS_S) / span) * span
    b = b_loc - off
    off_b = _local_offset(b, tz)
    if off_b != off:
        b = b_loc - off_b
    return lo + EPS_S < b <= now + EPS_S


def scored_mask(now: float, dt_s: float, g: str, config: Optional[Mapping[str, Any]]) -> np.ndarray:
    """bool[52]: features scored on this grain row. Span features only on
    their span decision (H only); tick mode: everything."""
    from . import features as F
    out = np.ones(F.FEATURE_DIM, dtype=bool)
    if not canonical(config):
        return out
    for name in SPAN_S:
        i = F.FEATURE_INDEX[name]
        out[i] = g == "h" and span_decision(now, dt_s, name, config)
    return out


def series(base: str, g: str, mode: Any = TICK) -> str:
    """Grain series name. Tick mode: base. Canonical: 'feature.nat' ->
    'feature.nat.h' / '.q'; behaviour names are the H names as is, and
    '<base>.q' for Q."""
    if _mode(mode) == TICK:
        return base
    if base.startswith("feature."):
        return f"{base}.{g}"
    return base if g == "h" else f"{base}.q"


# =========================================================== streams / budgets
def stream_of(detector: str) -> str:
    from .detectors import DETECTOR_INFO
    return str(DETECTOR_INFO[detector]["stream"])


def period_s(detector: str, dt_s: float, mode: Any = TICK) -> float:
    """Stream period of a detector: max(dt, G_h) for H, max(dt, G_q) for Q,
    dt for T. Tick mode: dt."""
    dt = float(dt_s)
    if _mode(mode) == TICK:
        return dt
    st = stream_of(detector)
    if st == "t":
        return dt
    return max(dt, GRAIN_S[st])


def stream_period_s(stream: str, dt_s: float, mode: Any = TICK) -> float:
    dt = float(dt_s)
    if _mode(mode) == TICK or stream == "t":
        return dt
    return max(dt, GRAIN_S[stream])


def tick_type(now: float, dt_s: float, mode: Any = TICK) -> str:
    """'h' on an H decision tick, else 'q' on a Q decision tick, else 't'.
    Tick mode: always 'h'."""
    md = _mode(mode)
    if md == TICK or decision(now, dt_s, "h", md):
        return "h"
    if decision(now, dt_s, "q", md):
        return "q"
    return "t"


def n_per_day(tau: str, dt_s: float, mode: Any = TICK) -> float:
    """Decision ticks of type tau per day; the three sum to 86400/dt."""
    dt = float(dt_s)
    md = _mode(mode)
    if md == TICK:
        return DAY / dt if tau == "h" else 0.0
    nh = DAY / max(GRAIN_S["h"], dt)
    nq = (DAY / GRAIN_S["q"] - nh) if observable("q", dt, md) else 0.0
    if tau == "h":
        return nh
    if tau == "q":
        return nq
    return max(0.0, DAY / dt - nh - nq)


def beta(tau: str, dt_s: float, mode: Any = TICK) -> float:
    """Budget share of tick type tau, renormalised over the types present."""
    md = _mode(mode)
    if md == TICK:
        return 1.0 if tau == "h" else 0.0
    present = [t for t in TAUS if n_per_day(t, dt_s, md) > 1e-9]
    if tau not in present:
        return 0.0
    return BETA[tau] / sum(BETA[t] for t in present)


def e_day_tick(q: float, now: float, dt_s: float, mode: Any = TICK) -> float:
    """Single-tick e_day = q n_tau / beta_tau (tau = this tick's type). Equals
    combine.e_day(q, dt) in tick mode and at dt = 3600."""
    md = _mode(mode)
    if md == TICK:
        from . import combine
        return combine.e_day(q, dt_s)
    tau = tick_type(now, dt_s, md)
    return float(q) * e_day_mult(tau, dt_s, md)


def e_day_mult(tau: str, dt_s: float, mode: Any = TICK) -> float:
    """n_tau / beta_tau (the single-tick multiplier of tick type tau)."""
    b = beta(tau, dt_s, mode)
    if not b > 0.0:
        return math.nan
    return n_per_day(tau, dt_s, mode) / b


def e_day_detector(p: float, detector: str, dt_s: float, mode: Any = TICK) -> float:
    """p * 86400 / period_s(detector)."""
    return float(p) * DAY / period_s(detector, dt_s, mode)


def e_inst(q_by_stream: Mapping[str, float], dt_s: float, mode: Any = TICK) -> float:
    """min_s q_s N_s with N_t = 86400/dt and N_h = 24 (tick mode: q_t 86400/dt;
    NaN streams are skipped; NaN when none is finite)."""
    best = math.nan
    for s, q in q_by_stream.items():
        q = float(q) if q is not None else math.nan
        if not q == q:
            continue
        N = DAY / stream_period_s(s, dt_s, mode)
        v = q * N
        if not best <= v:
            best = v
    return best


def evidence_arl_days(stream: str, mode: Any = TICK) -> float:
    """Wall-clock ARL of a stream's evidence CUSUM: 33 d / share (tick mode:
    the single v2 chart, 33 d)."""
    if _mode(mode) == TICK:
        return EVIDENCE_ARL_DAYS
    return EVIDENCE_ARL_DAYS / EVIDENCE_SHARE[stream]


def evidence_arl_ticks(stream: str, dt_s: float, mode: Any = TICK) -> float:
    """The ARL in the stream's periods."""
    return evidence_arl_days(stream, mode) * DAY / stream_period_s(stream, dt_s, mode)


# =================================================================== transfer
def v_from_omega(omega: Any) -> Any:
    """v = 1 + ((M - 1)/M) clip(omega, 0, 2M)."""
    w = np.clip(np.asarray(omega, dtype=np.float64), 0.0, OMEGA_MAX)
    v = 1.0 + ((M - 1.0) / M) * w
    return float(v) if v.ndim == 0 else v


def transfer_nb(mu: Any, r: Any, v: Any) -> Tuple[Any, Any]:
    """NB rate transfer: same per-minute mu, 1/r_Q = v / r_H."""
    return mu, np.asarray(r, dtype=np.float64) / np.asarray(v, dtype=np.float64)


def transfer_bb(p: Any, c: Any, v: Any) -> Tuple[Any, Any]:
    """BB ratio transfer: same p, 1 + c_Q = (1 + c_H) / v (floored at C_T_MIN)."""
    c = np.asarray(c, dtype=np.float64)
    return p, np.maximum(C_T_MIN, (1.0 + c) / np.asarray(v, dtype=np.float64) - 1.0)


def t_variance(scale: Any, df: Any) -> Any:
    """Predictive variance of a Student-t (scale^2 df/(df-2), scale^2 at df <= 2)."""
    s2 = np.asarray(scale, dtype=np.float64) ** 2
    df = np.asarray(df, dtype=np.float64)
    with np.errstate(all="ignore"):
        return np.where(df > 2.0, s2 * df / (df - 2.0), s2)


def jensen_delta(scale: Any, df: Any, v: Any) -> Any:
    """Default location shift of a log-scale feature: -(v - 1) sigma_H^2 / 2
    (equal expected rates at both grains, lognormal identity)."""
    return -(np.asarray(v, dtype=np.float64) - 1.0) * t_variance(scale, df) / 2.0


def transfer_t(loc: Any, scale: Any, df: Any, v: Any, delta: Any) -> Tuple[Any, Any, Any]:
    """t transfer: loc + delta, scale sqrt(v), same df; |delta| <= scale_T."""
    sc = np.asarray(scale, dtype=np.float64) * np.sqrt(np.asarray(v, dtype=np.float64))
    d = np.clip(np.asarray(delta, dtype=np.float64), -sc, sc)
    return np.asarray(loc, dtype=np.float64) + d, sc, df


def omega_eb(A: Any, B: Any, W: Any, omega_parent: Any, k: float = KAPPA_OMEGA) -> Any:
    """omega_node = (A + k omega_parent bbar) / (B + k bbar), bbar = B / W
    (W = 0 or B <= 0: the parent)."""
    A, B, W = (np.asarray(x, dtype=np.float64) for x in (A, B, W))
    par = np.broadcast_to(np.asarray(omega_parent, dtype=np.float64), A.shape)
    with np.errstate(all="ignore"):
        bbar = B / W
        num = A + k * par * bbar
        den = B + k * bbar
        out = num / den
    ok = (W > 0.0) & (B > 0.0) & np.isfinite(out)
    return np.where(ok, out, par)


def delta_eb(D: Any, W: Any, delta_parent: Any, k: float = KAPPA_OMEGA) -> Any:
    """delta_node = (D + k delta_parent) / (W + k)."""
    D, W = np.asarray(D, dtype=np.float64), np.asarray(W, dtype=np.float64)
    par = np.broadcast_to(np.asarray(delta_parent, dtype=np.float64), D.shape)
    with np.errstate(all="ignore"):
        out = (D + k * par) / (np.maximum(W, 0.0) + k)
    return np.where(np.isfinite(out), out, par)


# ============================================================== local days
def local_date(ts: float, tz: str) -> _dt.date:
    return TB.local_datetime(float(ts), tz).date()


# ========================================================= paired hours (§6.3)
def paired_stats(family: str, q_vals: Any, q_expo: Any, h_val: Any, h_expo: Any,
                 h_pred: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(a, b, d) of one paired hour (an H row holding exactly M Q rows),
    vectorised over features (arrays [..., F]; q_* are [M, ..., F]).

      nb : q_vals = counts x_j, q_expo = minutes e_j; h_val = X_H, h_expo = e_H;
           w_j = x_j / e_j, a = s^2(w) - lam mean(1/e_j) with lam = X_H / e_H
           (the Poisson part of the within-hour variance), b = mu_H^2 / r_H
           (the H predictive's excess rate variance, h_pred {'mean', 'r'}).
      bb : q_vals = k_j, q_expo = n_j; w_j = k_j / n_j, a = s^2(w) -
           p_hat (1 - p_hat) mean(1/n_j), p_hat = K / N; b = p_H (1 - p_H) /
           (1 + c_H) (h_pred {'mean', 'c'}).
      t  : q_vals = y_j (Q vec), a = s^2(y), b = var_H (h_pred {'var'}),
           d = mean(y_j) - y_H (h_val).

    b uses the predictive mean (not the observed hour, whose square is biased
    upward by its own noise); a is unbiased for the excess within-hour
    variance, so omega = E[a] / b and v = 1 + ((M-1)/M) omega is exact for
    exchangeable quarters (cadence.md §6.3)."""
    qv = np.asarray(q_vals, dtype=np.float64)
    qe = np.asarray(q_expo, dtype=np.float64)
    with np.errstate(all="ignore"):
        if family == "t":
            w = qv
        else:
            w = qv / qe
        s2 = np.var(w, axis=0, ddof=1)
        if family == "nb":
            lam = np.asarray(h_val, dtype=np.float64) / np.asarray(h_expo, dtype=np.float64)
            a = s2 - lam * np.mean(1.0 / qe, axis=0)
            mu = np.asarray(h_pred["mean"], dtype=np.float64)
            b = mu * mu / np.asarray(h_pred["r"], dtype=np.float64)
            d = np.zeros_like(a)
        elif family == "bb":
            ph = np.asarray(h_val, dtype=np.float64) / np.asarray(h_expo, dtype=np.float64)
            a = s2 - ph * (1.0 - ph) * np.mean(1.0 / qe, axis=0)
            p = np.asarray(h_pred["mean"], dtype=np.float64)
            b = p * (1.0 - p) / (1.0 + np.asarray(h_pred["c"], dtype=np.float64))
            d = np.zeros_like(a)
        elif family == "t":
            a = s2
            b = np.asarray(h_pred["var"], dtype=np.float64) * np.ones_like(a)
            d = np.mean(w, axis=0) - np.asarray(h_val, dtype=np.float64)
        else:
            raise ValueError(f"paired_stats: unknown family {family!r}")
    b = np.broadcast_to(b, np.shape(a)).astype(np.float64)
    return a, b, d
