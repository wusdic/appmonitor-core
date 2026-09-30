"""Code lengths, entropies, dependencies and statistical bounds for the
progressive profile core (docs/lib3/progressive.md §6.3-§6.5, §6.12, §6.18).

STATUS: implemented (W-P0). Pure functions on numpy arrays; every count
argument may be fractional (evidence-weighted, PPC-9). All code lengths are in
bits.

Code lengths
    ml_code_length(c)            sum_b c_b log2(n / c_b)   (the best fixed code in hindsight)
    kt_code_length(c)            Krichevsky-Trofimov (Dirichlet(1/2)) sequential code length
    dirichlet_code_length(c, a)  Dirichlet(a * prior) sequential code length
    dirichlet_predictive(...)    evidence-scaled hierarchical-Dirichlet predictive (§6.5.3)
    split_description_length     log2 C + g log2 card_hint (§6.5.3)
    ip_two_part_code             who-level two-part code with escapes (§6.18.2)
Entropies / divergences / dependencies
    entropy_plugin, entropy_miller_madow, entropy_chao_shen, jsd, cond_entropy,
    mutual_information, g3 (Kivinen & Mannila 1995), redundancy_g3
Bounds
    hoeffding_bound, empirical_bernstein_bound (Maurer & Pontil 2009),
    time_uniform_delta (delta / (k (k + 1))), beta_quantile, jeffreys_lower,
    good_turing_unseen, poisson_zero_p, beta_mom_strength, eb_beta_prior
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import special as _sp

LOG2E = 1.0 / math.log(2.0)
_TINY = 1e-300


def _arr(c) -> np.ndarray:
    a = np.asarray(c, dtype=np.float64)
    return np.where(np.isfinite(a) & (a > 0), a, 0.0)


# ============================================================ code lengths
def ml_code_length(counts) -> float:
    """Maximum-likelihood (empirical entropy) code length of a count vector:
    sum_b c_b log2(n / c_b). Never longer than the code under any fixed
    distribution, which is what makes the universal-inference e-value valid."""
    c = _arr(counts).ravel()
    n = c.sum()
    if n <= 0:
        return 0.0
    nz = c[c > 0]
    return float(np.sum(nz * (np.log2(n) - np.log2(nz))))


def ml_code_length_rows(counts) -> np.ndarray:
    """ml_code_length per row of a 2-D (or per leading index of an N-D) array
    whose last axis is the alphabet."""
    c = _arr(counts)
    n = c.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(c > 0, c * (np.log2(np.maximum(n, _TINY)) - np.log2(np.maximum(c, _TINY))), 0.0)
    return t.sum(axis=-1)


def dirichlet_code_length(counts, alpha: float = 0.5, prior=None) -> float:
    """Sequential (order-free) code length of counts under a Dirichlet mixture
    with concentration alpha * prior_b (prior uniform by default):
        -log2 [ Gamma(A) / Gamma(n + A) * prod_b Gamma(c_b + a_b) / Gamma(a_b) ],
    A = sum_b a_b. With alpha = 1/2 per symbol and a uniform prior this is KT.
    Fractional counts are allowed (the Gamma form extends them)."""
    c = _arr(counts).ravel()
    K = c.size
    if K == 0:
        return 0.0
    if prior is None:
        a = np.full(K, float(alpha))
    else:
        p = _arr(prior).ravel()
        p = p / p.sum() if p.sum() > 0 else np.full(K, 1.0 / K)
        a = np.maximum(float(alpha) * K * p, 1e-12)
    A = a.sum()
    n = c.sum()
    lp = (_sp.gammaln(A) - _sp.gammaln(n + A)
          + np.sum(_sp.gammaln(c + a) - _sp.gammaln(a)))
    return float(-lp * LOG2E)


def kt_code_length(counts) -> float:
    """Krichevsky-Trofimov code length (Dirichlet(1/2, ..., 1/2) mixture)."""
    return dirichlet_code_length(counts, 0.5)


def dirichlet_predictive(share, n: float, alpha: float, parent) -> np.ndarray:
    """Evidence-scaled hierarchical-Dirichlet predictive (§6.5.3, PPC-9):
        p(v) = (share(v) * n + alpha * parent(v)) / (n + alpha)
    share = the node's mass shares (sum <= 1; missing mass is carried by
    parent), n = the node's evidence (not mass), parent = the parent's
    predictive over the same alphabet."""
    s = _arr(share)
    par = _arr(parent)
    n = max(0.0, float(n))
    a = max(0.0, float(alpha))
    den = n + a
    if den <= 0:
        return par
    return (s * n + a * par) / den


def split_description_length(C: int, n_groups: int, card_hint: float) -> float:
    """L_split(c) = log2 C + g log2(card_hint(a, l)) bits (§6.5.3)."""
    return float(math.log2(max(1, int(C))) + max(0, int(n_groups)) * math.log2(max(1.0, float(card_hint))))


def ip_two_part_code(p_item: float, item_space_bits: float, seen: bool,
                     unseen_mass: float, escape_item_bits: float) -> float:
    """Who-level two-part code (§6.18.2): a seen item costs
    -log2 p(g) + log2 |g| (item probability plus the address inside it); an
    unseen item costs the escape -log2 U plus its own address bits plus the
    address inside it. Every level is thus a complete code for the IP."""
    if seen and p_item > 0:
        return float(-math.log2(p_item) + item_space_bits)
    u = min(1.0, max(float(unseen_mass), 1e-12))
    return float(-math.log2(u) + escape_item_bits + item_space_bits)


# ======================================================= entropies & deps
def entropy_plugin(counts) -> float:
    """Plug-in entropy in bits."""
    c = _arr(counts).ravel()
    n = c.sum()
    if n <= 0:
        return 0.0
    p = c[c > 0] / n
    return float(-np.sum(p * np.log2(p)))


def entropy_miller_madow(counts) -> float:
    """Plug-in entropy + (K_obs - 1) / (2 n ln 2) bits (Miller-Madow correction)."""
    c = _arr(counts).ravel()
    n = c.sum()
    if n <= 0:
        return 0.0
    k = int(np.count_nonzero(c))
    return entropy_plugin(c) + (k - 1) / (2.0 * n) * LOG2E


def entropy_chao_shen(counts, other_mass: float = 0.0) -> float:
    """Chao-Shen coverage-adjusted entropy (bits). `counts` are the tracked
    item counts (evidence units), `other_mass` untracked mass counted as
    singletons-level uncertainty via the coverage estimate. Falls back to
    Miller-Madow when every item is a singleton (coverage 0)."""
    c = _arr(counts).ravel()
    c = c[c > 0]
    n = c.sum() + max(0.0, float(other_mass))
    if n <= 0 or c.size == 0:
        return 0.0
    f1 = float(np.sum((c > 0) & (c < 1.5))) + max(0.0, float(other_mass))
    cov = 1.0 - f1 / n
    if cov <= 0:
        return entropy_miller_madow(c)
    pa = cov * c / n
    den = 1.0 - (1.0 - pa) ** n
    den = np.where(den > 1e-12, den, 1e-12)
    return float(-np.sum(pa * np.log2(pa) / den))


def jsd(p, q) -> float:
    """Jensen-Shannon divergence in bits (0..1); inputs are normalised."""
    p = _arr(p).ravel()
    q = _arr(q).ravel()
    if p.size != q.size:
        raise ValueError("jsd: size mismatch")
    sp, sq = p.sum(), q.sum()
    if sp <= 0 or sq <= 0:
        return 0.0 if sp == sq else 1.0
    p, q = p / sp, q / sq
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))
    return max(0.0, min(1.0, 0.5 * kl(p, m) + 0.5 * kl(q, m)))


def cond_entropy(joint) -> float:
    """H(Y | X) in bits for a joint count table [x, y] (plug-in)."""
    j = _arr(joint)
    n = j.sum()
    if n <= 0:
        return 0.0
    rows = j.sum(axis=1)
    h = 0.0
    for i in range(j.shape[0]):
        if rows[i] > 0:
            h += rows[i] / n * entropy_plugin(j[i])
    return float(h)


def mutual_information(joint) -> float:
    """I(X; Y) in bits for a joint count table [x, y] (plug-in, >= 0)."""
    j = _arr(joint)
    return max(0.0, entropy_plugin(j.sum(axis=0)) - cond_entropy(j))


def g3(joint) -> float:
    """g3 error of the FD X -> Y (Kivinen & Mannila 1995): the smallest share of
    rows to remove for X to determine Y, 1 - sum_x max_y c(x, y) / n."""
    j = _arr(joint)
    n = j.sum()
    if n <= 0:
        return 0.0
    return float(1.0 - j.max(axis=1).sum() / n)


# ================================================================= bounds
def hoeffding_bound(R: float, n: float, delta: float) -> float:
    """sqrt(R^2 ln(1/delta) / (2 n)): deviation of a mean of n values of range R."""
    if n <= 0:
        return math.inf
    return math.sqrt(R * R * math.log(1.0 / delta) / (2.0 * n))


def empirical_bernstein_bound(var: float, R: float, n: float, delta: float) -> float:
    """sqrt(2 V ln(3/delta) / n) + 3 R ln(3/delta) / n (Maurer & Pontil 2009 form
    used by rule (S), §6.5.5); V the (weighted) sample variance, R the range."""
    if n <= 0:
        return math.inf
    lg = math.log(3.0 / delta)
    return math.sqrt(2.0 * max(0.0, var) * lg / n) + 3.0 * max(0.0, R) * lg / n


def time_uniform_delta(delta: float, k: int) -> float:
    """delta_k = delta / (k (k + 1)), k >= 1: a union bound over checks k = 1, 2, ...
    (sum_k delta_k = delta) that makes a fixed-n bound valid at every check."""
    k = max(1, int(k))
    return float(delta) / (k * (k + 1.0))


def beta_quantile(q: float, a: float, b: float) -> float:
    """q-quantile of Beta(a, b) (scipy betaincinv)."""
    return float(_sp.betaincinv(max(a, 1e-12), max(b, 1e-12), q))


def jeffreys_lower(k: float, n: float, q: float = 0.05) -> float:
    """q-quantile of the Jeffreys posterior Beta(k + 1/2, n - k + 1/2)."""
    return beta_quantile(q, float(k) + 0.5, max(0.0, float(n) - float(k)) + 0.5)


def beta_mom_strength(purities: Sequence[float]) -> float:
    """Method-of-moments strength s = m (1 - m) / v - 1 of a set of rates
    (inf when they are all equal or fewer than two)."""
    x = np.asarray([float(v) for v in purities if np.isfinite(v)])
    if x.size < 2:
        return math.inf
    m, v = float(x.mean()), float(x.var(ddof=1))
    if v <= 1e-12:
        return math.inf
    s = m * (1.0 - m) / v - 1.0
    return s if s > 0 else 1e-6


def eb_beta_prior(k_others: Sequence[float], n_others: Sequence[float],
                  s_cap: float = 20.0) -> Tuple[float, float]:
    """Leave-one-out empirical-Bayes Beta prior of §6.12 from the OTHER heavy
    keys (k successes of n each): mean m = (sum k + 1/2) / (sum n + 1),
    strength s = min(s_cap, s_mom, sum n); Jeffreys (1/2, 1/2) with fewer than
    two others."""
    k = np.asarray(k_others, dtype=float)
    n = np.asarray(n_others, dtype=float)
    if k.size < 2:
        return 0.5, 0.5
    m = (k.sum() + 0.5) / (n.sum() + 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pur = np.where(n > 0, k / n, np.nan)
    s = min(float(s_cap), beta_mom_strength(pur), float(n.sum()))
    s = max(s, 1e-6)
    return float(m * s), float((1.0 - m) * s)


def good_turing_unseen(n1: float, evicted: float, N: float) -> float:
    """(N1 + E + 0.5) / (N + 1), clipped to [0, 1] (§5.5.3 convention)."""
    return float(min(1.0, max(0.0, (float(n1) + float(evicted) + 0.5) / (max(0.0, float(N)) + 1.0))))


def poisson_zero_p(expected: float) -> float:
    """P(0 | Poisson(E)) = e^-E (stale rule of §6.8.1)."""
    return float(math.exp(-max(0.0, float(expected))))


def hdr_p(probs, idx: int) -> float:
    """Highest-density-region p-value of outcome idx under a discrete
    distribution: sum of the probabilities strictly below p[idx] plus half of
    those equal to it (mid-p). In (0, 1] for p[idx] > 0."""
    p = _arr(probs).ravel()
    s = p.sum()
    if s <= 0 or not 0 <= idx < p.size:
        return math.nan
    p = p / s
    f = p[idx]
    return float(p[p < f].sum() + 0.5 * p[p == f].sum())


def sidak(p_min: float, n: float) -> float:
    """1 - (1 - p_min)^n: the probability that the smallest of n valid
    p-values is <= p_min (tick / day multiplicity, §6.16.3)."""
    if not (p_min == p_min):
        return math.nan
    p = min(1.0, max(0.0, float(p_min)))
    n = max(1.0, float(n))
    return float(-math.expm1(n * math.log1p(-p))) if p < 1.0 else 1.0


def log2_mean_exp2(x) -> float:
    """log2( mean_i 2^x_i ), numerically stable (used for averaged e-values)."""
    a = np.asarray(x, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return -math.inf
    m = float(a.max())
    return m + math.log2(float(np.mean(np.exp2(a - m))))
