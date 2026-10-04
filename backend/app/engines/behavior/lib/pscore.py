"""Pure scoring functions of the progressive core (docs/lib3/progressive.md §6.16).

STATUS: implemented (W-P0). Used by P03 (conformity) and by B29's
counterfactuals against archived models; no store access, no state. Every
count n / N is evidence on the confidence channel (PPC-9, §6.9.4); every
function returns NaN when the model cannot speak (unscored, never 1).

    cat_p(pnode.CatSummary, value, t, parent_pred=None, alpha=2)   HDR p (§6.5.3 predictive)
    who_p(pnode.WhoSummary, key_by_level, t, n_days)               p_who, closed level, U
    when_p(pnode.WhenSummary, daytype, minute, t)                 HDR p over 96 slots (§6.13)
    numeric_p(cdf, n)  /  tail_p(y, u, xi, sigma, tail_mass)  /  conformal_rank_p(k, n)
    invariant_p(n_viol, n);  unseen_p(U);  dual_anchor(p_cur, p_ref)
    content_p(ps) = min(1, m min p);  event_p(ps) = min(1, 5 min p)
    tick_p(p_min, n) (Sidak per tick);  day_p(p_min, n_day) (per-day multiplicity, §6.16.3)
"""
from __future__ import annotations

import math
from typing import Any, Dict, Hashable, Iterable, Optional, Sequence, Tuple

import numpy as np

from . import pmdl
from . import psketch as PS

NAN = float("nan")
ALPHA = 2.0
ALPHA_T = 0.1
N_MIN = 20.0                      # support needed for a type's model (§6.16.1)


def _isnan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


# ============================================================ categorical
def cat_predictive(summary: Any, t: float, parent_pred: Optional[Dict[Hashable, float]] = None,
                   alpha: float = ALPHA) -> Tuple[Dict[Hashable, float], float]:
    """Evidence-scaled hierarchical-Dirichlet predictive of a CatSummary:
    p(v) = (share(v) n + alpha p_parent(v)) / (n + alpha), n = confidence
    evidence. Returns ({value: p}, p of an unseen value). parent_pred None =
    the summary's own Good-Turing unseen mass spread as the escape."""
    ss = summary.ss
    keys, sh, other = ss.distribution(t)
    n = ss.total_evidence(t)
    U = ss.unseen(t)
    if parent_pred is None:
        par = {k: float(s) for k, s in zip(keys, sh)}
        par_unseen = U
    else:
        par = dict(parent_pred)
        par_unseen = max(0.0, 1.0 - sum(par.values()))
    den = n + alpha
    if den <= 0:
        return par, par_unseen
    out = {}
    for k, s in zip(keys, sh):
        out[k] = (float(s) * n + alpha * par.get(k, 0.0)) / den
    for k, p in par.items():
        if k not in out:
            out[k] = alpha * p / den
    unseen = (other * n + alpha * par_unseen) / den
    return out, unseen


def cat_p(summary: Any, value: Hashable, t: float,
          parent_pred: Optional[Dict[Hashable, float]] = None, alpha: float = ALPHA,
          n_min: float = N_MIN) -> float:
    """HDR p-value of `value` under the node's predictive (§6.16.2 content,
    categorical). An unseen value's probability is the escape mass. NaN when
    the node has < n_min evidence."""
    if summary.ss.total_evidence(t) < n_min:
        return NAN
    pred, unseen = cat_predictive(summary, t, parent_pred, alpha)
    probs = np.asarray(list(pred.values()) + [unseen], dtype=np.float64)
    keys = list(pred.keys())
    try:
        idx = keys.index(value)
    except ValueError:
        idx = len(keys)
    return pmdl.hdr_p(probs, idx)


# ==================================================================== who
def who_p(who: Any, keys: Sequence[Any], t: float, n_days: int,
          levels: Optional[Sequence[int]] = None) -> Tuple[float, Optional[int], float]:
    """p_who (§6.16.2): at the finest closed level l* (U <= 0.05, heavy set
    <= 8 items covering >= 95 %, >= 5 normal days): 1 if the event's item is
    in the heavy set, else U_l*. Returns (p, l*, U); (NaN, None, NaN) when no
    level is closed (an open population has no who constraint)."""
    l = who.closed_level(t, n_days, levels=levels)
    if l is None:
        return NAN, None, NAN
    U = who.levels[l].unseen(t)
    g = keys[l] if l < len(keys) else None
    heavy, _ = who.heavy_set(l, t)
    if g is not None and g in heavy:
        return 1.0, l, U
    return float(U), l, U


# =================================================================== when
def when_density(counts: np.ndarray, N: float, alpha_t: float = ALPHA_T) -> np.ndarray:
    """f(s) = (h(s) + alpha_t / 96) / (N + alpha_t), h(s) = share(s) N (§6.13)."""
    c = np.asarray(counts, dtype=np.float64)
    s = c.sum()
    share = c / s if s > 0 else np.zeros_like(c)
    return (share * N + alpha_t / c.size) / (N + alpha_t)


EB_ALPHAS = tuple(float(2.0 ** k) for k in range(-2, 13))     # 0.25 .. 4096


def eb_concentration(counts: np.ndarray, N: float, prior: np.ndarray,
                     alphas: Sequence[float] = EB_ALPHAS) -> float:
    """Empirical-Bayes strength alpha of a PARENT's slot predictive used as the
    Dirichlet prior of a child's slot counts: the alpha maximising the
    Dirichlet-multinomial marginal likelihood of the child's counts
    n_s = share_s N under Dir(alpha prior) (grid over 2^-2 .. 2^12; Minka 2000,
    fixed base measure). A child that keeps its parent's time-of-day law gets
    a large alpha (the parent's longer evidence speaks for it), a child that
    is much more concentrated (one department's login minutes inside the
    route's) a small one."""
    from scipy.special import gammaln
    c = np.asarray(counts, dtype=np.float64)
    s = c.sum()
    if s <= 0 or N <= 0:
        return float(alphas[-1])
    n = c / s * float(N)
    f = np.asarray(prior, dtype=np.float64)
    f = np.maximum(f / max(f.sum(), 1e-300), 1e-300)
    nz = n > 0
    nn, ff = n[nz], f[nz]
    best, arg = -math.inf, float(alphas[-1])
    for a in alphas:
        L = float(gammaln(a) - gammaln(N + a) + np.sum(gammaln(nn + a * ff) - gammaln(a * ff)))
        if L > best:
            best, arg = L, float(a)
    return arg


def when_density_prior(counts: np.ndarray, N: float, prior: np.ndarray,
                       alpha: Optional[float] = None) -> np.ndarray:
    """Hierarchical back-off of the slot density (P03, round 3):
    f(s) = (share(s) N + alpha prior(s)) / (N + alpha), prior = the parent
    node's own predictive (recursively; the root's prior is when_density's
    uniform alpha_t / 96), alpha by eb_concentration. A slot empty at the node
    AND at its ancestors keeps the ancestors' (tiny) mass, so a young node
    (N ~ 40) no longer floors an empty slot's HDR p at ~alpha_t / (N +
    alpha_t); a slot empty at the node but used by its parent's population
    keeps a share of the parent's mass instead of the uniform floor."""
    c = np.asarray(counts, dtype=np.float64)
    pr = np.asarray(prior, dtype=np.float64)
    ps = pr.sum()
    pr = pr / ps if ps > 0 else np.full(c.size, 1.0 / c.size)
    s = c.sum()
    if s <= 0 or N <= 0:
        return pr
    a = eb_concentration(c, N, pr) if alpha is None else float(alpha)
    return (c / s * float(N) + a * pr) / (float(N) + a)


def when_p(when: Any, daytype: int, minute: float, t: float, alpha_t: float = ALPHA_T,
           n_min: float = N_MIN) -> float:
    """HDR p of the event's slot under the node's day-type density; backs off
    to the all-day-type density when the day type has < n_min evidence; NaN
    when neither has n_min (the caller then backs off to the parent)."""
    d = 1 if daytype else 0
    N = when.evidence(d, t)
    h = when.hist[d]
    if N < n_min:
        N = when.evidence(0, t) + when.evidence(1, t)
        h = when.hist.sum(axis=0)
        if N < n_min:
            return NAN
    f = when_density(h, N, alpha_t)
    return pmdl.hdr_p(f, int(float(minute) // 15) % 96)


# ================================================================ numeric
def numeric_p(cdf: float) -> float:
    """Two-sided p from a CDF value: 2 min(F, 1 - F) (mid-rank for the digest)."""
    if _isnan(cdf):
        return NAN
    F = min(1.0, max(0.0, float(cdf)))
    return float(min(1.0, 2.0 * min(F, 1.0 - F)))


def conformal_rank_p(n_as_extreme: float, n: float) -> float:
    """(1 + #{y_i at least as extreme}) / (n + 1)."""
    return float(min(1.0, (1.0 + max(0.0, n_as_extreme)) / (max(0.0, n) + 1.0)))


def tail_p(y: float, u: float, xi: float, sigma: float, tail_mass: float = 0.1,
           k: Optional[float] = None, x_max: Optional[float] = None) -> float:
    """Doubled POT tail probability beyond threshold u: 2 * tail_mass * sf(y - u).

    Without k: the plug-in GPD survival function at the fitted (xi, sigma).
    With k (the number of excesses the fit used): the PREDICTIVE survival
    function, sf averaged over the sampling uncertainty of (sigma, xi)
    (gpd_predictive_sf). x_max (the largest observed value, in the same
    orientation as y) truncates that uncertainty to the parameters whose
    support reaches it."""
    if y <= u:
        return 1.0
    if k is not None and k >= PRED_K_MIN:
        sf = gpd_predictive_sf(y - u, xi, sigma, k, None if x_max is None else x_max - u)
    else:
        from .evt import gpd_sf
        sf = float(gpd_sf(np.asarray([y - u]), xi, sigma)[0])
    return float(min(1.0, 2.0 * tail_mass * sf))


# Predictive GPD tail (round 4, groups_views owner). The plug-in sf at the PWM
# estimate from k <= 64 reservoir excesses treats (xi, sigma) as known: a
# bounded fit (xi < 0) ends at its estimated end point, often BEFORE the
# observed maximum, and a light fitted tail decays exponentially - pack O: a
# portal comment's 222 ms duration (lognormal(3.5, 0.6): true two-sided p
# 1.5e-3) scored p 4.4e-7 against an observed maximum of 200 ms; a git
# net.pkts_down of 2 968 inside the displayed range 4-3 000 scored 1e-9, the
# git / mail TLS durations (lognormal(4, 0.8)) 449 ms p 5e-6 (true 8e-3) and
# 12.7 ms below a range starting at ~13 ms p 1e-9 (true 7e-2). Simulated
# (sim: lognormal, normal, gamma, Poisson, Pareto bodies, n = 300 / 3 000, 64
# excesses): the plug-in p <= 1e-4 fired 4-39x, p <= 1e-6 120-3 600x more
# often than nominal; the predictive below 0.4-1.7x (Pareto 2.8-3.4x, heavy
# tails are where PWM itself is weakest), conservative in bounded tails.
PRED_K_MIN = 3
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(9)       # N(0, 1) quadrature, 9 nodes
_GH_W = _GH_W / _GH_W.sum()
_GZ1 = np.repeat(_GH_X, _GH_X.size)
_GZ2 = np.tile(_GH_X, _GH_X.size)
_GZW = np.outer(_GH_W, _GH_W).ravel()


def pwm_cov(xi: float, sigma: float, k: float) -> Tuple[float, float, float]:
    """Asymptotic (var sigma, cov(sigma, xi), var xi) of the PWM estimates of a
    GPD from k excesses (Hosking & Wallis 1987, Technometrics 29: shape
    kappa = -xi; finite for xi < 1/2 - xi is clipped to [-0.5, 0.4] here)."""
    kk = -min(max(float(xi), -0.5), 0.4)
    c = 1.0 / (max(float(k), 1.0) * (1.0 + 2.0 * kk) * (3.0 + 2.0 * kk))
    vs = sigma * sigma * (7.0 + 18.0 * kk + 11.0 * kk ** 2 + 2.0 * kk ** 3) * c
    csk = sigma * (2.0 + kk) * (2.0 + 6.0 * kk + 7.0 * kk ** 2 + 2.0 * kk ** 3) * c
    vk = (1.0 + kk) * (2.0 + kk) ** 2 * (1.0 + kk + 2.0 * kk ** 2) * c
    return vs, -csk, vk


def gpd_predictive_sf(z: float, xi: float, sigma: float, k: float,
                      z_max: Optional[float] = None) -> float:
    """P(excess > z) under the approximate posterior of (sigma, xi): the
    estimate's asymptotic normal law (pwm_cov) integrated by a 9 x 9
    Gauss-Hermite product rule, restricted to sigma > 0 and - when the
    largest observed excess z_max is given - to the parameters whose support
    reaches it (a bounded GPD ending before an observed value has likelihood
    0). Past the observed maximum the result decays like the heavier members
    of that set instead of jumping to 0; at the maximum it stays near the
    rank probability 1 / (k + 1) of the tail. O(81)."""
    if not (z > 0.0):
        return 1.0
    if not (sigma > 0.0 and math.isfinite(sigma) and math.isfinite(xi)):
        return NAN
    vs, cv, vx = pwm_cov(xi, sigma, k)
    a = math.sqrt(max(vs, 0.0))
    b = cv / a if a > 0 else 0.0
    c = math.sqrt(max(vx - b * b, 0.0))
    sg = sigma + a * _GZ1
    xs = xi + b * _GZ1 + c * _GZ2
    ok = sg > 0.0
    if z_max is not None and z_max > 0.0:
        neg = xs < 0.0
        end = np.where(neg, -sg / np.where(neg, xs, -1.0), np.inf)
        ok &= end >= z_max
    w = _GZW[ok]
    tw = float(w.sum())
    if not tw > 0.0:
        return 1.0 / (float(k) + 1.0)
    sg, xs = sg[ok], xs[ok]
    with np.errstate(all="ignore"):
        zz = z / sg
        small = np.abs(xs) < 1e-9
        xsafe = np.where(small, 1.0, xs)
        t = xsafe * zz
        inside = t > -1.0
        sf = np.where(small, np.exp(-zz),
                      np.where(inside, np.exp(-np.log1p(np.where(inside, t, 0.0)) / xsafe), 0.0))
    return float(min(1.0, float((w * sf).sum()) / tw))


def num_summary_p(num: Any, v: float, t: float, n: float, gpd_hi: Optional[Tuple[float, float, float]] = None,
                  gpd_lo: Optional[Tuple[float, float, float]] = None, n_min: float = N_MIN) -> float:
    """p of a numeric value against a pnode.NumSummary (§6.10): inside the
    [Q(0.01), Q(0.99)] band -> two-sided p from the digest CDF; beyond ->
    the GPD tail (u, xi, sigma) when fitted, else the conformal rank with
    the band's tail mass. n = confidence evidence."""
    if n < n_min or num.td.total() <= 0:
        return NAN
    y = num.y(v)
    if not math.isfinite(y):
        return conformal_rank_p(0, n)
    lo, hi = num.td.quantile(0.01), num.td.quantile(0.99)
    if lo <= y <= hi:
        return numeric_p(num.td.cdf(y))
    if y > hi:
        if gpd_hi is not None:
            u, xi, sigma = gpd_hi
            return tail_p(y, u, xi, sigma, 0.1)
        return conformal_rank_p(0.0 if y > num.td.vmax else 0.01 * n, n)
    if gpd_lo is not None:
        u, xi, sigma = gpd_lo
        return tail_p(-y, -u, xi, sigma, 0.1)
    return conformal_rank_p(0.0 if y < num.td.vmin else 0.01 * n, n)


# ============================================================ combination
def invariant_p(n_viol: float, n: float) -> float:
    """(n_viol + 0.5) / (n + 1) when an invariant is broken."""
    return float(min(1.0, (max(0.0, n_viol) + 0.5) / (max(0.0, n) + 1.0)))


def unseen_p(U: float) -> float:
    return NAN if _isnan(U) else float(min(1.0, max(0.0, U)))


def dual_anchor(p_cur: float, p_ref: Optional[float]) -> float:
    """min(1, 2 min(p_cur, p_ref)); p_cur alone when p_ref is missing / NaN (§6.8.3)."""
    if _isnan(p_ref):
        return p_cur
    if _isnan(p_cur):
        return float(p_ref)
    return float(min(1.0, 2.0 * min(p_cur, p_ref)))


def content_p(ps: Iterable[float]) -> float:
    """min(1, m min p) over the m checked attributes (NaNs skipped)."""
    v = [float(p) for p in ps if not _isnan(p)]
    if not v:
        return NAN
    return float(min(1.0, len(v) * min(v)))


def event_p(ps: Iterable[float], factor: float = 5.0) -> float:
    """p_ev = min(1, 5 min_t p_t) over the typed p-values (NaNs skipped)."""
    v = [float(p) for p in ps if not _isnan(p)]
    if not v:
        return NAN
    return float(min(1.0, factor * min(v)))


def tick_p(p_min: float, n: float) -> float:
    """Sidak over the n scored events of an (s, ip) in a tick (§6.16.3)."""
    return pmdl.sidak(p_min, n)


def day_p(p_min: float, n_day: float) -> float:
    """Per-day multiplicity p_day = 1 - (1 - p_min)^n_day (§6.16.3)."""
    return pmdl.sidak(p_min, n_day)


def vtype_mask(p_by_type: Dict[str, float], thr: float = 1e-3,
               order: Sequence[str] = ("who", "when", "content", "seq", "novel")) -> int:
    """Bitmask of the types with p <= thr (bit i = order[i])."""
    m = 0
    for i, k in enumerate(order):
        p = p_by_type.get(k)
        if not _isnan(p) and p <= thr:
            m |= 1 << i
    return m
