"""Organisation-level hyperpriors per feature kind (engines.md B03 step 0).

The last tier of the entity -> class -> system -> org backoff. They are
deliberately weak (tiny pseudo-counts) so the very first predictive of a
brand-new entity is finite and *wide*: early p-values are large rather than
garbage and cold start cannot deadlock (B03 test c: an empty store gives
p(10 req/min) > 0.05).

count   : Gamma-Poisson on the per-minute rate, mu0 = 1 event/min, shape a0 = 0.5
          (rate parameter b0 = a0/mu0 minutes); predictive for exposure e min is
          NB(mean = a/b*e, size = a).
ratio   : Beta(0.5, 0.5) (Jeffreys).
others  : Normal-Inverse-Gamma on the transformed (vec) value with a kind
          default m0 (bytes: ln 1e4, i.e. vec = log1p(bytes/min)),
          kappa0 = 0.01, alpha0 = 1, beta0 = 4. Predictive Student-t with
          2*alpha0 = 2 dof and scale sqrt(beta0(1+kappa0)/(alpha0 kappa0)) ~ 20.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Union

from .features import FEATURE_KIND


@dataclass(frozen=True, slots=True)
class GammaPoissonPrior:
    """Gamma(shape a0, rate b0) prior on a per-minute event rate."""
    mu0: float = 1.0          # events per minute
    a0: float = 0.5           # shape (pseudo-events)

    @property
    def b0(self) -> float:
        """Rate parameter in minutes (pseudo-exposure) so that a0/b0 = mu0."""
        return self.a0 / self.mu0


@dataclass(frozen=True, slots=True)
class BetaPrior:
    a0: float = 0.5
    b0: float = 0.5

    @property
    def mean(self) -> float:
        return self.a0 / (self.a0 + self.b0)


@dataclass(frozen=True, slots=True)
class NIGPrior:
    """Normal-Inverse-Gamma(m0, kappa0, alpha0, beta0) on a transformed value."""
    m0: float = 0.0
    kappa0: float = 0.01
    alpha0: float = 1.0
    beta0: float = 4.0

    def predictive_scale(self) -> float:
        """Student-t predictive scale sqrt(beta0 (1 + kappa0) / (alpha0 kappa0))."""
        return math.sqrt(self.beta0 * (1.0 + self.kappa0) / (self.alpha0 * self.kappa0))

    @property
    def df(self) -> float:
        return 2.0 * self.alpha0


Prior = Union[GammaPoissonPrior, BetaPrior, NIGPrior]

KAPPA0 = 0.01
ALPHA0 = 1.0
BETA0 = 4.0

# Kind default m0 on the vec scale (FEATURE_SPEC transforms).
KIND_M0: Dict[str, float] = {
    "bytes": math.log(1e4),   # log1p(bytes/min) ~ 10 kB/min
    "avg": 0.0,
    "bounded": 0.0,           # logit(0.5)
    "clr": 0.0,
    "window": 0.5,            # most window descriptors live in [0, 1]
    "gauge": 0.5,
    "ctx": 0.0,
}

# Per-feature m0 overrides (vec scale). Harmless with kappa0 = 0.01, but they
# keep model_state p50 of an immature entity in a sensible natural range.
FEATURE_M0: Dict[str, float] = {
    "bytes_per_flow": math.log(1e4),
    "updown_log": 0.0,
    "http_latency": math.log(100.0),
    "resp_bytes_avg": math.log(1e4),
    "req_bytes_avg": math.log(1e3),
    "dns_dga_score": math.log(0.1),
    "dns_qname_len": math.log(20.0),
    "tls_handshake_ms": math.log(100.0),
    "think_time": math.log(10.0),
    "rtt": math.log(50.0),
    "flow_duration": math.log(1e3),
    "req_per_session": math.log1p(20.0),
}

COUNT_PRIOR = GammaPoissonPrior()

# Known consequence of the spec values: the hyperprior predictive of a count at
# exposure e is NB(mean e, size 0.5), whose rate marginal is chi2_1. Observing
# 10 events/min gives P(X >= 10e) ~ 1.7e-3 and two-sided mid-p ~ 3.4e-3 at
# e = 15 min. That is finite (no deadlock) but below the 0.05 quoted in B03
# test (c); an empty-store entity is protected by trust = 1 during training
# and by class/system backoff live, not by this prior alone. A shape a0 <= 0.08
# would be needed for p > 0.05 at 10/min.
COLD_START_NOTE = "hyperprior count p(10/min) ~ 3.4e-3 two-sided (a0 = 0.5)"
RATIO_PRIOR = BetaPrior()

# Hierarchy / backoff constants shared by B03, B07, B08, B09.
BACKOFF_KAPPA_MIN = 2.0       # empirical-Bayes backoff pseudo-count clip (B03)
BACKOFF_KAPPA_MAX = 50.0
RHYTHM_CLASS_STRENGTH = 6.0   # Beta(6 pi, 6 (1 - pi)) class prior (B07)
DIRICHLET_BACKOFF = 5.0       # p_e = (c_e + 5 p_c) / (N_e + 5)  (B08, B09)
NB_KAPPA_CLIP = (0.5, 1e3)    # overdispersion kappa-hat clip (B04)
BB_PHI_CLIP = (20.0, 1000.0)  # Beta-Binomial concentration clip (B04)


def prior_for_kind(kind: str) -> Prior:
    """Hyperprior for a FEATURE_SPEC kind."""
    if kind == "count":
        return COUNT_PRIOR
    if kind == "ratio":
        return RATIO_PRIOR
    if kind in KIND_M0:
        return NIGPrior(m0=KIND_M0[kind], kappa0=KAPPA0, alpha0=ALPHA0, beta0=BETA0)
    raise ValueError(f"unknown feature kind {kind!r}")


def prior_for_feature(name: str) -> Prior:
    """Hyperprior for a named feature (kind default plus m0 override)."""
    kind = FEATURE_KIND[name]
    if kind in ("count", "ratio"):
        return prior_for_kind(kind)
    m0 = FEATURE_M0.get(name, KIND_M0[kind])
    return NIGPrior(m0=m0, kappa0=KAPPA0, alpha0=ALPHA0, beta0=BETA0)


def likelihood_family(kind: str) -> str:
    """'nb' (count), 'bb' (ratio) or 't' (NIG Student-t) for a feature kind."""
    if kind == "count":
        return "nb"
    if kind == "ratio":
        return "bb"
    return "t"
