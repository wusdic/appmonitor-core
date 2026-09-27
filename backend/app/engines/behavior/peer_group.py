"""PeerGroupEngine (B02): who is this IP like? A two-level class hierarchy.

Why: an entity is judged against itself AND against its peers. Backing off to
a class gives a new IP a baseline from its first tick, lets B18 watch a whole
role, and tells "this user browses differently from the others" apart from
"this IP no longer behaves like a user at all". v1 (clustering.py) ran a
KMeans silhouette sweep on z-scored fingerprints: labels changed every run,
names came from fixed thresholds, and individual path preferences split one
role into singletons. B02 separates the two questions:

  level 1, ROLE (the class): coarse descriptors that do not depend on the
    individual, clustered org-wide (role ids are global, class state is keyed
    per system as (s, 'class:<rid>'), contract L):
      A      automation index = mean of timing regularity, periodicity,
             non-browser UA share, 1 - think-time percentile, 1 - path
             entropy (components that are not identified are left out);
      CLR    channel mix over http/dns/tls/flows (per-15-min rates + 0.5);
      fam    template-family distribution (m_vocab.family_distribution);
      s48    normalised 48-bin rhythm shape (m_rhythm.shape48);
      dev    device class of the dominant UA family: browser/library/other.
    D_role = 0.3|dA| + 0.2 Aitchison/NORM_AIT + 0.2 sqrt(JSD fam)
             + 0.2 (1 - cos s48) + 0.1 [dev differs]    (all terms in [0, 1];
    a term with a missing side counts NEUTRAL = 0.5).
    HDBSCAN(precomputed, min_cluster_size=2, min_samples=1, eom, single
    cluster allowed), clusters joining below ROLE_EPS = 0.15 merged (the
    cluster_selection_epsilon rule: EOM with min_cluster_size 2 otherwise
    splits a tight role along its sampling noise); noise is the singleton
    role 'unique' (+ peer_outlier);
    > 30 % noise falls back to average linkage cut at the largest gap.
    Super level human / machine by the role's mean A, hysteresis 0.4 / 0.6.
  level 2, SUB-CLASS inside a role, on individual distances:
      D_ind = 0.4 mean_f W1(deciles of the predictive) + 0.3 sqrt(JSD of the
              full templates / SNI / dports) + 0.2 (1 - cos rhythm168)
              + 0.1 [dominant stack differs]
    HDBSCAN leaf (epsilon SUB_EPS = 0.05); noise members are singleton
    sub-classes. The W1 of two
    predictives is taken on their vec-space (median, sd) from
    m_baseline.profile_many at the role's busiest bin48, as the mean over
    the deciles of |d median + d sd z_k|, in units of 3 pooled member sds
    and capped at 1. Individual Dirichlet path preferences therefore split
    sub-classes but never roles.

Stable ids: Hungarian matching on 1 - Jaccard(members) against the previous
run; an id is inherited when J >= 0.3, otherwise minted, with class_split /
class_merge when one old role feeds several new ones or vice versa. A role
retires after 3 runs without a matching cluster. Names are made from data in
natural units (human/automated, dominant family, active window, volume tier,
top-3 class-vs-rest Cohen's d). Soft membership P(c|e) is proportional to
exp(-D_role(e, medoid_c)/d90_c), with d90_c floored at D90_FLOOR: sampling
noise of a few ticks alone moves a descriptor that far, so a tight role
cannot reject its own new members.

Eligible for clustering: >= 48 committed active ticks (model.baseline n_eff),
not quarantined, no open incident, not frozen. The published role of an
established entity only changes through class_transition: the soft argmax
must name the same other role with p >= 0.7 on 3 consecutive runs (a split
or merge re-labels its members directly). Entities that are not eligible
keep their last assignment.

Cold start (every tick): an entity with < 48 commits accumulates a
provisional descriptor from what it sent so far (feature.nat, act.tokens and
the TLS / DNS / port sets, client.stack_set, the bin48 of each active tick),
because raw sets live 1 h and the gated models are still empty. At its 3rd
active tick it is typed once against the role medoids: D_role <= d90 of the
nearest role -> new_entity_matched (INFO, prob = P(c|e)), else
new_entity_unmatched (MEDIUM). With so few ticks the 48-bin shape is not
identified, so its rhythm term asks instead whether the observed slots lie
where the role is active: 1 - sum_b h_b s_c[b] / max s_c.

Static classes come from ctx.config.ip_classes (ipaddress matching); pools
are CIDRs (dhcp_scopes, else /24 | /64) holding >= 5 short-lived IPs that all
share one role and none of which is linkable (model.link).

Cadence: the refit runs when 16 ticks or 6 h have passed, whichever first;
the cold path runs every tick, so the engine itself has interval 1.
ctx.training: everything is learnt, no event is emitted (and what would
have been announced is marked as such, so the end of training is quiet).

Writes model.class@(__org__, __org__) in the layout of lib/m_class.py (the
reader; version bumped and a new dict object on every change, which is what
consumers cache on) plus engine-private '_state'; class profiles at
(s, class:<rid> | class:static:<name> | class:pool:<cidr>); profile.archetype
(= class_path super/role/sub), archetype_confidence, extra.peer_group.
"""
from __future__ import annotations

import ipaddress
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import xlogy

from ...core.engine import Context, Engine
from ...models.schema import BehaviorEvent, EntityProfile, Severity
from .lib import features as F
from .lib import m_baseline as MB
from .lib import m_client as MC
from .lib import m_rhythm as MR
from .lib import m_template as MT
from .lib import m_timing as MTI
from .lib import m_vocab as MV
from .lib import timebins as TB
from .lib.classkeys import ORG, SYSTEM_KEY, assign_key, pool_key, role_key, static_key
from .lib.template import channel_of

MODEL = "model.class"
BASELINE, VOCAB, RHYTHM, CLIENT, TIMING = ("model.baseline", "model.vocab", "model.rhythm",
                                           "model.client", "model.timing")
CONTROL, LINK = "model.control", "model.link"
ACTIVE, NAT = "feature.active", "feature.nat"
QUARANTINE = "behavior.quarantine"
FMT = 1

REFIT_TICKS = 16
REFIT_S = 6 * 3600.0
MIN_COMMITS = 48.0                 # committed active ticks to be clustered
COLD_MIN_ACTIVE = 3                # active ticks before cold typing
COLD_KEEP_S = 8 * 86400.0          # a cold record idle this long is dropped
COLD_DIM_CAP = 256                 # values kept per dimension of a cold vocab
J_INHERIT = 0.3
RETIRE_RUNS = 3
NOISE_MAX = 0.30
SUPER_LO, SUPER_HI = 0.4, 0.6
D90_FLOOR = 0.15
ROLE_EPS = D90_FLOOR               # HDBSCAN cluster_selection_epsilon, level 1
SUB_EPS = 0.05                     # ... level 2
TRANSITION_P = 0.7
TRANSITION_RUNS = 3
NEUTRAL = 0.5                      # distance term with a missing side
NORM_AIT = math.log(100.0) * math.sqrt(2.0)   # one channel x100 up, another x100 down
SD_ID_MAX = 5.0                    # vec-space predictive sd above this: not identified
W1_UNITS = 3.0                     # W1 cap in pooled sds
LINEAGE_KEEP = 16
FAM_KEEP = 48                      # families kept in a stored medoid descriptor
THINK_REF_KEEP = 129
POOL_MIN = 5
POOL_SHORT_S = 86400.0
ROLE_JOIN = 0.35                   # n < 3 eligible: join pairs closer than this
COHEN_MIN = 0.8
W_ROLE = (0.3, 0.2, 0.2, 0.2, 0.1)     # A, channel, families, rhythm, device
W_IND = (0.4, 0.3, 0.2, 0.1)           # W1, templates, rhythm168, stack
UNIQUE = "unique"

CHANNELS = ("http", "dns", "tls", "flows")
CH_IDX = [F.FEATURE_INDEX[n] for n in ("http_requests", "dns_queries", "tls_handshakes", "flows")]
I_INT = F.FEATURE_INDEX["intensity"]
I_REG = F.FEATURE_INDEX["timing_regularity"]
I_PER = F.FEATURE_INDEX["periodicity"]
I_PE = F.FEATURE_INDEX["path_entropy"]
I_THINK = F.FEATURE_INDEX["think_time"]
BROWSERS = frozenset({"chrome", "firefox", "safari", "edge"})
LIBRARIES = frozenset({"python-requests", "go-http-client", "okhttp", "curl", "wget",
                       "postman", "java"})
DEVICES = ("browser", "library", "other")
VOL_TIERS = ((1.0, "idle"), (10.0, "low"), (100.0, "medium"), (1000.0, "high"),
             (math.inf, "very-high"))
FULL_DIMS = ("tmpl", "sni", "dport")
FAM_DIMS = ("tmpl", "sni", "dns", "dport")
RAW_DIMS = (("sni", "tls.sni_etld1_set"), ("dns", "dns.qname_etld1_set"),
            ("dport", "l4.dport_set"))
Z_DEC = np.array([-1.2815515655446004, -0.8416212335729143, -0.5244005127080407,
                  -0.2533471031357997, 0.0, 0.2533471031357997, 0.5244005127080407,
                  0.8416212335729143, 1.2815515655446004])
_LN2 = math.log(2.0)
_NAN = math.nan
OTHER = "__other__"


# ================================================================= descriptors
@dataclass
class _Desc:
    """Role and individual descriptor of one (system, ip)."""
    key: str
    s: str
    e: str
    A: float = _NAN
    comps: Dict[str, float] = field(default_factory=dict)   # A components (think raw)
    clr: np.ndarray = field(default_factory=lambda: np.full(4, _NAN))
    fam: Dict[str, float] = field(default_factory=dict)
    s48: np.ndarray = field(default_factory=lambda: np.full(48, _NAN))
    s168: np.ndarray = field(default_factory=lambda: np.full(168, _NAN))
    dev: str = "other"
    rate15: float = _NAN           # events per 15 min
    med: Optional[np.ndarray] = None
    sd: Optional[np.ndarray] = None
    full: Dict[str, float] = field(default_factory=dict)
    stack: str = ""
    bucket: int = 0


def _f(x: Any, nd: int = 4) -> Optional[float]:
    """JSON-safe rounded float (None for NaN / inf / junk)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _norm_dist(d: Mapping[str, float]) -> Dict[str, float]:
    tot = sum(v for v in d.values() if v > 0.0)
    return {k: v / tot for k, v in d.items() if v > 0.0} if tot > 0.0 else {}


def _clr(rates_per_min: Sequence[float]) -> np.ndarray:
    """CLR of per-15-min rates + 0.5 (NaN when any channel is unknown)."""
    x = np.asarray(rates_per_min, dtype=np.float64) * 15.0
    if not np.all(np.isfinite(x)):
        return np.full(4, _NAN)
    lx = np.log(np.maximum(x, 0.0) + 0.5)
    return lx - lx.mean()


def _device(ua_share: Mapping[str, float]) -> Tuple[str, float]:
    """(device class, non-browser share) from {browser, library, other} shares."""
    tot = sum(ua_share.get(k, 0.0) for k in DEVICES)
    if not tot > 0.0:
        return "other", _NAN
    sh = {k: ua_share.get(k, 0.0) / tot for k in DEVICES}
    return max(DEVICES, key=lambda k: (sh[k], -DEVICES.index(k))), 1.0 - sh["browser"]


def _ua_class(family: str) -> str:
    return "browser" if family in BROWSERS else "library" if family in LIBRARIES else "other"


def _pct(x: float, ref: np.ndarray) -> float:
    """Mid-rank percentile of x in the sorted reference sample (NaN if none)."""
    if not math.isfinite(x) or ref.size == 0:
        return _NAN
    lo = np.searchsorted(ref, x, side="left")
    hi = np.searchsorted(ref, x, side="right")
    return float((lo + 0.5 * (hi - lo)) / ref.size)


def _automation(comps: Mapping[str, float], think_ref: np.ndarray) -> float:
    """A = mean of the identified components; 0.5 when none is."""
    think_p = _pct(comps.get("think", _NAN), think_ref)
    pe = comps.get("pe", _NAN)
    vals = [comps.get("reg", _NAN), comps.get("per", _NAN), comps.get("nonbrowser", _NAN),
            1.0 - think_p if math.isfinite(think_p) else _NAN,
            1.0 - pe if math.isfinite(pe) else _NAN]
    vals = [min(1.0, max(0.0, v)) for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else 0.5


def _vol_tier(rate15: float) -> str:
    if not math.isfinite(rate15):
        return "unknown"
    for lim, name in VOL_TIERS:
        if rate15 < lim:
            return name
    return VOL_TIERS[-1][1]


# ================================================================ distances
def _dense(dists: Sequence[Mapping[str, float]], vocab: Optional[List[str]] = None
           ) -> np.ndarray:
    """Rows = distributions over a shared vocabulary; NaN rows for empty ones."""
    if vocab is None:
        vocab = sorted({k for d in dists for k in d})
    idx = {k: i for i, k in enumerate(vocab)}
    P = np.zeros((len(dists), max(1, len(vocab))))
    for r, d in enumerate(dists):
        if not d:
            P[r] = _NAN
            continue
        for k, v in d.items():
            j = idx.get(k)
            if j is not None:
                P[r, j] = v
        t = P[r].sum()
        P[r] = P[r] / t if t > 0.0 else _NAN
    return P


def _entropy_bits(P: np.ndarray) -> np.ndarray:
    return -xlogy(P, P).sum(axis=-1) / _LN2


def _sqrt_jsd(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """sqrt(JSD in bits) between every row of P and every row of Q, in [0, 1];
    NaN where a row is missing."""
    out = np.full((P.shape[0], Q.shape[0]), _NAN)
    hq = _entropy_bits(Q)
    for i in range(P.shape[0]):
        if not np.isfinite(P[i, 0]):
            continue
        M = 0.5 * (P[i][None, :] + Q)
        j = _entropy_bits(M) - 0.5 * (_entropy_bits(P[i]) + hq)
        out[i] = np.sqrt(np.clip(j, 0.0, 1.0))
    return out


def _one_minus_cos(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """1 - cosine similarity of non-negative rows, in [0, 1]; NaN if missing."""
    def unit(X: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(X, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0.0, X / n, _NAN)
    return np.clip(1.0 - unit(P) @ unit(Q).T, 0.0, 1.0)


def _fill(M: np.ndarray, sym: bool) -> np.ndarray:
    M = np.where(np.isfinite(M), M, NEUTRAL)
    if sym:
        np.fill_diagonal(M, 0.0)
    return M


def _role_terms(a: Sequence[_Desc], b: Sequence[_Desc]) -> Tuple[np.ndarray, ...]:
    Aa = np.array([d.A for d in a])
    Ab = np.array([d.A for d in b])
    tA = np.abs(Aa[:, None] - Ab[None, :])
    Ca = np.stack([d.clr for d in a])
    Cb = np.stack([d.clr for d in b])
    tC = np.minimum(1.0, np.linalg.norm(Ca[:, None, :] - Cb[None, :, :], axis=2) / NORM_AIT)
    vocab = sorted({k for d in list(a) + list(b) for k in d.fam})
    tF = _sqrt_jsd(_dense([d.fam for d in a], vocab), _dense([d.fam for d in b], vocab))
    tR = _one_minus_cos(np.stack([d.s48 for d in a]), np.stack([d.s48 for d in b]))
    tD = np.array([[float(x.dev != y.dev) for y in b] for x in a])
    return tA, tC, tF, tR, tD


def role_distance(a: Sequence[_Desc], b: Optional[Sequence[_Desc]] = None) -> np.ndarray:
    """D_role between two descriptor lists (b None: the square matrix of a)."""
    sym = b is None
    terms = _role_terms(a, a if sym else b)
    D = sum(w * _fill(t, sym) for w, t in zip(W_ROLE, terms))
    return 0.5 * (D + D.T) if sym else D


def ind_distance(ds: Sequence[_Desc]) -> np.ndarray:
    """D_ind among the members of one role."""
    n = len(ds)
    # W1 of the decile vectors of two (roughly normal) vec-space predictives
    tW = np.full((n, n), _NAN)
    if all(d.med is not None for d in ds):
        med = np.stack([d.med for d in ds])
        sd = np.stack([d.sd for d in ds])
        ok = np.isfinite(med) & np.isfinite(sd) & (sd < SD_ID_MAX)
        with warnings.catch_warnings():             # an all-unidentified feature -> NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            scale = np.nanmedian(np.where(ok, sd, np.nan), axis=0)
        scale = np.where(np.isfinite(scale), np.maximum(scale, 1e-3), np.nan)
        for i in range(n):
            dm = med[i][None, :] - med                     # [n, 52]
            ds_ = sd[i][None, :] - sd
            w1 = np.abs(dm[:, :, None] + ds_[:, :, None] * Z_DEC[None, None, :]).mean(axis=2)
            u = np.minimum(1.0, w1 / (W1_UNITS * scale[None, :]))
            good = ok[i][None, :] & ok & np.isfinite(u)
            cnt = good.sum(axis=1)
            tW[i] = np.where(cnt > 0, np.where(good, u, 0.0).sum(axis=1) / np.maximum(cnt, 1), _NAN)
    P = _dense([d.full for d in ds])
    tT = _sqrt_jsd(P, P)
    tR = _one_minus_cos(np.stack([d.s168 for d in ds]), np.stack([d.s168 for d in ds]))
    tS = np.array([[float(bool(x.stack) and bool(y.stack) and x.stack != y.stack)
                    for y in ds] for x in ds])
    D = sum(w * _fill(t, True) for w, t in zip(W_IND, (tW, tT, tR, tS)))
    return 0.5 * (D + D.T)


# ================================================================= clustering
def _hdbscan(D: np.ndarray, method: str, eps: float) -> np.ndarray:
    """HDBSCAN on a precomputed distance matrix, then clusters that join
    below eps are merged (the cluster_selection_epsilon rule; with
    min_samples = 1 the merge height of two clusters is their smallest
    cross distance). Without it EOM with min_cluster_size = 2 splits a tight
    role along its sampling noise (8 members 0.01 apart came out as two
    roles). sklearn's own epsilon search fails on tied / zero distances."""
    from sklearn.cluster import HDBSCAN
    lab = HDBSCAN(metric="precomputed", min_cluster_size=2, min_samples=1,
                  cluster_selection_method=method, allow_single_cluster=True,
                  copy=True).fit(D).labels_.astype(int)
    ids = sorted(set(lab.tolist()) - {-1})
    parent = {c: c for c in ids}

    def find(c: int) -> int:
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c
    for i, a in enumerate(ids):
        ia = lab == a
        for b in ids[i + 1:]:
            if float(D[np.ix_(ia, lab == b)].min()) <= eps:
                parent[find(b)] = find(a)
    root = {c: find(c) for c in ids}
    return np.array([root[x] if x >= 0 else -1 for x in lab.tolist()], dtype=int)


def _gap_linkage(D: np.ndarray) -> np.ndarray:
    """Average linkage cut in the middle of the largest merge-height gap;
    singletons are noise (-1)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    Z = linkage(squareform(D, checks=False), method="average")
    h = Z[:, 2]
    if h.size < 2:
        lab = np.zeros(D.shape[0], dtype=int)
    else:
        k = int(np.argmax(np.diff(h)))
        lab = fcluster(Z, t=0.5 * (h[k] + h[k + 1]), criterion="distance") - 1
    sizes = np.bincount(lab)
    return np.where(sizes[lab] >= 2, lab, -1)


def _components(D: np.ndarray, thr: float) -> np.ndarray:
    """Connected components under D <= thr (tiny populations)."""
    n = D.shape[0]
    lab = -np.ones(n, dtype=int)
    c = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        stack, lab[i] = [i], c
        while stack:
            j = stack.pop()
            for k in np.nonzero((D[j] <= thr) & (lab < 0))[0]:
                lab[k] = c
                stack.append(int(k))
        c += 1
    return lab


def level1(D: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Role labels (-1 = unique) and whether the linkage fallback was used."""
    n = D.shape[0]
    if n < 3:
        lab = _components(D, ROLE_JOIN)
        return (lab if n > 1 else np.zeros(n, dtype=int)), False
    lab = _hdbscan(D, "eom", ROLE_EPS)
    if np.mean(lab < 0) > NOISE_MAX:
        return _gap_linkage(D), True
    return lab, False


def level2(D: np.ndarray) -> np.ndarray:
    """Sub-class labels inside one role; every noise member is its own sub."""
    n = D.shape[0]
    if n < 3:
        return np.zeros(n, dtype=int)
    lab = _hdbscan(D, "leaf", SUB_EPS)
    nxt = int(lab.max()) + 1 if (lab >= 0).any() else 0
    for i in np.nonzero(lab < 0)[0]:
        lab[i] = nxt
        nxt += 1
    return lab


def _medoid(D: np.ndarray, idx: Sequence[int]) -> Tuple[int, float]:
    """(medoid index, d90 of the members' distances to it, floored)."""
    idx = list(idx)
    sub = D[np.ix_(idx, idx)]
    m = idx[int(np.argmin(sub.sum(axis=1)))]
    others = [D[m, j] for j in idx if j != m]
    d90 = float(np.percentile(others, 90)) if others else 0.0
    return m, max(D90_FLOOR, d90)


def match_ids(new: Sequence[Set[str]], old: Mapping[str, Set[str]]
              ) -> Tuple[List[Optional[str]], np.ndarray, List[str]]:
    """Hungarian matching on 1 - Jaccard; an old id is inherited at J >= 0.3."""
    oids = sorted(old)
    out: List[Optional[str]] = [None] * len(new)
    J = np.zeros((len(new), len(oids)))
    for i, a in enumerate(new):
        for j, o in enumerate(oids):
            b = old[o]
            u = len(a | b)
            J[i, j] = len(a & b) / u if u else 0.0
    if len(new) and len(oids):
        r, c = linear_sum_assignment(1.0 - J)
        for i, j in zip(r, c):
            if J[i, j] >= J_INHERIT:
                out[i] = oids[j]
    return out, J, oids


def _window(s48: np.ndarray) -> str:
    """Active window in local hours from a 48-bin shape: '09-18h' / '24h'."""
    if not np.all(np.isfinite(s48)) or not s48.sum() > 0.0:
        return "?"
    d = s48[:24] + s48[24:]
    on = np.nonzero(d >= 0.5 * d.max())[0]
    if on.size >= 22:
        return "24h"
    # the shortest circular arc covering every 'on' hour
    gaps = np.diff(np.r_[on, on[0] + 24])
    k = int(np.argmax(gaps))
    start, end = on[(k + 1) % on.size], on[k] + 1
    return f"{int(start):02d}-{int(end % 24 or 24):02d}h"


# ====================================================================== engine
class PeerGroupEngine(Engine):
    name = "behavior.peer_group"
    layer = "behavior"
    consumes = [BASELINE, VOCAB, RHYTHM, CLIENT, TIMING, CONTROL, LINK, QUARANTINE,
                ACTIVE, NAT, "act.tokens", "tls.sni_etld1_set", "dns.qname_etld1_set",
                "l4.dport_set", "client.stack_set"]
    produces = [MODEL, "profile.archetype", "profile.archetype_confidence",
                "profile.extra.peer_group", "event.new_entity_matched",
                "event.new_entity_unmatched", "event.class_transition", "event.class_split",
                "event.class_merge", "event.peer_outlier"]
    description = ("Two-level class hierarchy: org-wide HDBSCAN roles on coarse role "
                   "descriptors (automation, channel mix, template families, rhythm, "
                   "device), leaf HDBSCAN sub-classes on individual distances; stable ids "
                   "by Hungarian/Jaccard, data-driven names, soft membership, static CIDR "
                   "classes and pools, cold-start typing every tick.")
    # the cold path runs every tick; the refit keeps its own 16-tick / 6-h stride
    interval = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._ip_cache: Tuple[Any, List[Tuple[str, List[Any], List[str], Any]]] = (None, [])

    # ----------------------------------------------------------------- run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        now, dt = float(ctx.now), float(ctx.window_s)
        mc = store.get_model(ORG[0], ORG[1], MODEL, default=None)
        if not (isinstance(mc, dict) and mc.get("fmt") == FMT and isinstance(mc.get("_state"), dict)):
            mc = self._new_model(mc if isinstance(mc, dict) else None)
            store.put_model(ORG[0], ORG[1], MODEL, mc, version=mc["version"], ts=now)
        st = mc["_state"]
        st["ticks"] = int(st.get("ticks", 0)) + 1

        young = self._cold_tick(ctx, st, now, dt)
        n = 0
        new = None
        last = st.get("last_refit")
        if last is None or st["ticks"] - int(st.get("refit_tick", 0)) >= REFIT_TICKS \
                or now - float(last) >= REFIT_S or now < float(last):
            new, k = self._refit(ctx, mc, young, now)
            n += k
        ready = [k for k in young if k not in st["seen"]
                 and int(st["cold"][k]["n"]) >= COLD_MIN_ACTIVE]
        if ready:
            new = new if new is not None else self._copy(mc)
            n += self._announce(ctx, new, ready, now, dt)
        if new is not None:
            new["version"] = int(mc.get("version", 0)) + 1
            new["ts"] = now
            store.put_model(ORG[0], ORG[1], MODEL, new, version=new["version"], ts=now)
        return n

    @staticmethod
    def _new_model(old: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        m: Dict[str, Any] = {"fmt": FMT, "version": int((old or {}).get("version", 0)),
                             "ts": None, "assign": {}, "roles": {}, "subs": {},
                             "statics": {}, "pools": {}}
        m["_state"] = {"ticks": 0, "refit_tick": 0, "last_refit": None, "next_role": 1,
                       "cold": {}, "seen": {}, "outliers": {}, "think_ref": [],
                       "runs": 0}
        return m

    @staticmethod
    def _copy(mc: Mapping[str, Any]) -> Dict[str, Any]:
        """New top-level object (consumers cache on id + version); the private
        state is shared, it is ours."""
        new = dict(mc)
        new["assign"] = {k: dict(v) for k, v in mc.get("assign", {}).items()}
        new["roles"] = {k: dict(v) for k, v in mc.get("roles", {}).items()}
        new["subs"] = {k: dict(v) for k, v in mc.get("subs", {}).items()}
        new["statics"] = dict(mc.get("statics", {}))
        new["pools"] = dict(mc.get("pools", {}))
        return new

    # ------------------------------------------------------------ cold path
    def _cold_tick(self, ctx: Context, st: Dict[str, Any], now: float, dt: float) -> List[str]:
        """Accumulate provisional descriptors of young entities; returns the
        keys of all young entities that have a cold record."""
        store = ctx.store
        cold: Dict[str, Any] = st["cold"]
        young: List[str] = []
        tctx = None
        for s in store.systems():
            for e in store.entities(s):
                key = assign_key(s, e)
                if MB.n_eff(store.get_model(s, e, BASELINE)) >= MIN_COMMITS:
                    continue
                a = store.vec_at(s, e, ACTIVE, now)
                if a is not None and float(a[0]) >= 0.5:
                    if tctx is None:
                        tctx = TB.tctx_from_config(now, ctx.config, dt)
                    cs = cold.get(key)
                    if cs is None:
                        cs = cold[key] = _new_cold(now)
                    _cold_update(cs, store, s, e, now, dt, tctx)
                if key in cold:
                    young.append(key)
        for k in [k for k, cs in cold.items() if now - float(cs["last"]) > COLD_KEEP_S]:
            del cold[k]
        return young

    def _cold_desc(self, key: str, cs: Mapping[str, Any], think_ref: np.ndarray) -> _Desc:
        s, _, e = key.partition("|")
        d = _Desc(key=key, s=s, e=e)
        n = max(1, int(cs["n"]))
        d.clr = _clr([c / n for c in cs["ch"]])
        d.rate15 = float(cs["vol"]) / n * 15.0
        comps = {k: (v[0] / v[1] if v[1] > 0 else _NAN) for k, v in cs["a"].items()}
        dev, nonb = _device(cs["ua"])
        comps["nonbrowser"] = nonb
        d.comps, d.dev = comps, dev
        d.A = _automation(comps, think_ref)
        tier = {"kind": "class", "dims": {dim: {v: [c] for v, c in vals.items()}
                                          for dim, vals in cs["dims"].items()}}
        d.fam = MV.family_distribution(tier)
        bins = np.asarray(cs["bins"], dtype=np.float64)
        d.s48 = bins / bins.sum() if bins.sum() > 0.0 else np.full(48, _NAN)
        return d

    def _type_cold(self, mc: Mapping[str, Any], d: _Desc) -> Optional[Dict[str, Any]]:
        """Nearest role of a provisional descriptor: {role, prob, D, d90, matched, alts}."""
        roles = [(rid, r) for rid, r in sorted(mc.get("roles", {}).items())
                 if r.get("desc") and not r.get("absent")]
        if not roles:
            return None
        meds = [_desc_from_stored(r["desc"]) for _, r in roles]
        tA, tC, tF, _, tD = _role_terms([d], meds)
        # partial-observation rhythm: are the observed slots where the role is active?
        tR = np.full((1, len(meds)), _NAN)
        if np.all(np.isfinite(d.s48)):
            for j, m in enumerate(meds):
                if np.all(np.isfinite(m.s48)) and m.s48.max() > 0.0:
                    tR[0, j] = 1.0 - min(1.0, float(d.s48 @ m.s48) / float(m.s48.max()))
        D = sum(w * _fill(t, False) for w, t in zip(W_ROLE, (tA, tC, tF, tR, tD)))[0]
        d90 = np.array([float(r.get("d90", D90_FLOOR)) for _, r in roles])
        logit = -D / d90
        P = np.exp(logit - logit.max())
        P /= P.sum()
        j = int(np.argmin(D))
        order = np.argsort(-P)[:3]
        return {"role": roles[j][0], "prob": float(P[j]), "D": float(D[j]), "d90": float(d90[j]),
                "matched": bool(D[j] <= d90[j]), "super": roles[j][1].get("super", "human"),
                "alts": [[roles[i][0], round(float(P[i]), 4)] for i in order]}

    def _announce(self, ctx: Context, mc: Dict[str, Any], keys: Sequence[str],
                  now: float, dt: float) -> int:
        st = mc["_state"]
        think_ref = np.asarray(st.get("think_ref") or [], dtype=np.float64)
        n = 0
        for key in keys:
            cs = st["cold"][key]
            prev = mc["assign"].get(key) or {}
            if prev.get("role") is not None and not prev.get("provisional"):
                st["seen"][key] = now        # clustered before (evidence decayed): not new
                continue
            d = self._cold_desc(key, cs, think_ref)
            t = self._type_cold(mc, d)
            if t is None:
                continue                     # no roles yet: try again next tick
            st["seen"][key] = now
            self._assign_provisional(ctx, mc, d, t, now)
            if ctx.training:
                continue
            s, e = d.s, d.e
            role_nm = mc["roles"].get(t["role"], {}).get("name", t["role"])
            matched = t["matched"]
            ctx.store.add_event(BehaviorEvent(
                system=s, entity=e, ts=now,
                kind="new_entity_matched" if matched else "new_entity_unmatched",
                score=float(t["prob"]) if matched else float(min(1.0, t["D"])),
                severity=Severity.INFO if matched else Severity.MEDIUM,
                description=(f"new entity typed as '{role_nm}' (p={t['prob']:.2f}, "
                             f"D={t['D']:.2f} <= d90 {t['d90']:.2f})" if matched else
                             f"new entity unlike every role: nearest '{role_nm}' at "
                             f"D={t['D']:.2f} > d90 {t['d90']:.2f}"),
                extra={"prob": round(t["prob"], 4), "role": t["role"], "role_name": role_nm,
                       "D": round(t["D"], 4), "d90": round(t["d90"], 4),
                       "n_active": int(cs["n"]), "A": _f(d.A), "device": d.dev,
                       "alternatives": t["alts"], "matched": matched},
                axes=["peer"], dedupe_key=f"new_entity|{s}|{e}",
                model_version=int(mc.get("version", 0)), window=(float(cs["first"]), now)))
            n += 1
        return n

    def _assign_provisional(self, ctx: Context, mc: Dict[str, Any], d: _Desc,
                            t: Mapping[str, Any], now: float) -> None:
        prev = mc["assign"].get(d.key) or {}
        role = t["role"] if t["matched"] else None
        sup = t["super"] if t["matched"] else ("machine" if d.A >= 0.5 else "human")
        a = {"role": role, "sub": None, "prob": round(float(t["prob"]), 4),
             "static": list(prev.get("static") or []), "pool": prev.get("pool"),
             "super": sup, "provisional": True, "D": round(float(t["D"]), 4)}
        a["class_path"] = f"{sup}/{role or 'unmatched'}"
        mc["assign"][d.key] = a
        self._entity_profile(ctx, mc, d.s, d.e, a, now)

    # --------------------------------------------------------------- refit
    def _refit(self, ctx: Context, mc: Mapping[str, Any], young: Sequence[str],
               now: float) -> Tuple[Dict[str, Any], int]:
        store = ctx.store
        new = self._copy(mc)
        st = new["_state"]
        st["last_refit"], st["refit_tick"] = now, st["ticks"]
        st["runs"] = int(st.get("runs", 0)) + 1
        events: List[BehaviorEvent] = []
        blocked = self._blocked(store)
        eligible: List[Tuple[str, str, Mapping[str, Any]]] = []
        for s in store.systems():
            for e in store.entities(s):
                bm = store.get_model(s, e, BASELINE)
                if MB.n_eff(bm) < MIN_COMMITS or (s, e) in blocked:
                    continue
                q = store.vec_tail(s, e, QUARANTINE, 1)[1]
                if q.size and float(q[-1, 0]) >= 0.5:
                    continue
                ctl = store.get_model(s, e, CONTROL)
                if isinstance(ctl, Mapping) and ctl.get("frozen"):
                    continue
                eligible.append((s, e, bm))
        fallback_b = int(TB.tctx_from_config(now, ctx.config, ctx.window_s)["bin48"])
        descs = self._mature_descs(store, eligible, now, fallback_b)
        think = np.sort([d.comps["think"] for d in descs if math.isfinite(d.comps.get("think", _NAN))])
        if think.size:
            st["think_ref"] = [float(x) for x in
                               np.quantile(think, np.linspace(0, 1, min(THINK_REF_KEEP, think.size)))]
        think_ref = np.asarray(st["think_ref"], dtype=np.float64)
        for d in descs:
            d.A = _automation(d.comps, think_ref)
        if descs:
            self._cluster(ctx, new, descs, now, events)
        else:
            self._age_roles(new, set())
        # young entities: refresh their provisional typing against the new roles
        for key in young:
            prev = new["assign"].get(key) or {}
            if prev.get("role") is not None and not prev.get("provisional"):
                continue                     # clustered before: keeps its role
            if key in st["seen"] and key in st["cold"]:
                d = self._cold_desc(key, st["cold"][key], think_ref)
                t = self._type_cold(new, d)
                if t is not None:
                    self._assign_provisional(ctx, new, d, t, now)
        self._statics(ctx, new, now)
        self._pools(ctx, new, now)
        self._publish_members(new)
        for key, a in new["assign"].items():
            s, _, e = key.partition("|")
            self._entity_profile(ctx, new, s, e, a, now)
        self._class_profiles(ctx, new, now)
        n = 0
        if not ctx.training:
            for ev in events:
                store.add_event(ev)
                n += 1
        return new, n

    @staticmethod
    def _blocked(store) -> Set[Tuple[str, str]]:
        out: Set[Tuple[str, str]] = set()
        for inc in store.incidents(status="open"):
            out.add((inc.system, inc.entity))
            for m in inc.entities or ():
                out.add((inc.system, str(m)))
        return out

    def _mature_descs(self, store, eligible: Sequence[Tuple[str, str, Mapping[str, Any]]],
                      now: float, fallback_b: int) -> List[_Desc]:
        descs: List[_Desc] = []
        groups: Dict[Tuple[str, int], List[int]] = {}
        models: List[Mapping[str, Any]] = []
        for s, e, bm in eligible:
            d = _Desc(key=assign_key(s, e), s=s, e=e)
            rh = store.get_model(s, e, RHYTHM)
            if isinstance(rh, dict) and isinstance(rh.get("state"), dict):
                d.s48 = MR.shape48(rh)
                d.s168 = MR.shape168(rh)
            d.bucket = int(np.nanargmax(d.s48)) if np.any(np.isfinite(d.s48)) else fallback_b
            vm = MV.get(store, s, e)
            if MV.kind(vm) == "entity":
                d.fam = MV.family_distribution(vm, now=now)
                full: Dict[str, float] = {}
                for dim in FULL_DIMS:
                    for v, c in MV.counts(vm, dim, now).items():
                        if c > 0.0:
                            full[f"{dim}={v}"] = c
                d.full = _norm_dist(full)
            cm = MC.get(store, s, e)
            ua: Dict[str, float] = {}
            if MC.kind(cm) == "entity":
                for tok, sh in MC.shares(cm, now).items():
                    cls = _ua_class(MC.parse(tok).family)
                    ua[cls] = ua.get(cls, 0.0) + sh
                dom = MC.dominant(cm, 1, now)
                d.stack = dom[0][0] if dom else ""
            d.dev, d.comps["nonbrowser"] = _device(ua)
            d.comps["think"] = MTI.think_time(store.get_model(s, e, TIMING))[0]
            groups.setdefault((s, d.bucket), []).append(len(descs))
            descs.append(d)
            models.append(bm)
        for (s, b), idx in groups.items():
            med, sd = MB.profile_many(store, s, [descs[i].e for i in idx],
                                      [models[i] for i in idx], _bucket_tctx(b))
            for r, i in enumerate(idx):
                self._baseline_parts(descs[i], med[r], sd[r])
        return descs

    @staticmethod
    def _baseline_parts(d: _Desc, med: np.ndarray, sd: np.ndarray) -> None:
        """A components, channel mix and volume from the vec-space predictive."""
        d.med, d.sd = med, sd
        ok = np.isfinite(med) & np.isfinite(sd) & (sd < SD_ID_MAX)
        with np.errstate(over="ignore", invalid="ignore"):
            rates = np.expm1(med[CH_IDX])              # count vec = log1p(rate / min)
            d.clr = _clr(rates)
            d.rate15 = float(np.expm1(med[I_INT]) * 15.0)
        if ok[I_REG]:
            d.comps["reg"] = float(med[I_REG])
        if ok[I_PER]:
            d.comps["per"] = float(med[I_PER])
        if ok[I_PE]:
            d.comps["pe"] = float(1.0 / (1.0 + math.exp(-float(med[I_PE]))))
        if ok[I_THINK]:                                 # avg vec = ln(seconds)
            d.comps["think"] = float(med[I_THINK])

    # -------------------------------------------------------- level 1 / 2
    def _cluster(self, ctx: Context, mc: Dict[str, Any], descs: List[_Desc], now: float,
                 events: List[BehaviorEvent]) -> None:
        st = mc["_state"]
        keys = [d.key for d in descs]
        D = role_distance(descs)
        lab, fell_back = level1(D)
        labels = sorted(set(lab.tolist()) - {-1})
        clusters = [sorted(np.nonzero(lab == c)[0].tolist()) for c in labels]
        j_of_label = {c: j for j, c in enumerate(labels)}
        new_sets = [{keys[i] for i in c} for c in clusters]
        old_roles = mc["roles"]
        old_sets = {rid: set(r.get("cluster") or r.get("members") or [])
                    for rid, r in old_roles.items()}
        ids, J, oids = match_ids(new_sets, old_sets)
        # lineage: splits (one old -> several new) and merges (several old -> one new)
        minted: List[int] = []
        for i, rid in enumerate(ids):
            if rid is None:
                ids[i] = f"r{int(st['next_role'])}"
                st["next_role"] = int(st["next_role"]) + 1
                minted.append(i)
        overlap = {(i, o): len(new_sets[i] & old_sets[o]) for i in range(len(new_sets))
                   for o in oids}
        lineage_ev: List[Tuple[str, str, List[str], List[str]]] = []
        for o in oids:
            kids = [i for i in range(len(new_sets))
                    if overlap[(i, o)] >= max(2, 0.25 * len(old_sets[o]))
                    and overlap[(i, o)] >= 0.5 * len(new_sets[i])]
            if len(kids) >= 2 and any(i in minted for i in kids):
                lineage_ev.append(("class_split", o, [ids[i] for i in kids],
                                   sorted(set().union(*(new_sets[i] for i in kids)))))
                for i in kids:
                    if ids[i] != o:
                        old_roles.setdefault(ids[i], {}).setdefault("lineage", [])
                        _lineage(old_roles[ids[i]], {"ts": now, "kind": "split", "from": o})
        for i in range(len(new_sets)):
            pars = [o for o in oids if overlap[(i, o)] >= max(2, 0.5 * len(old_sets[o]))]
            if len(pars) >= 2:
                lineage_ev.append(("class_merge", ids[i], pars, sorted(new_sets[i])))
                old_roles.setdefault(ids[i], {}).setdefault("lineage", [])
                _lineage(old_roles[ids[i]], {"ts": now, "kind": "merge", "from": pars})
        split_kids = {ids[i]: o for kind, o, kids_ids, _ in lineage_ev if kind == "class_split"
                      for i in range(len(ids)) if ids[i] in kids_ids}
        merged_into = {p: ids_i for kind, ids_i, pars, _ in lineage_ev if kind == "class_merge"
                       for p in pars}
        alive = set(ids)
        self._age_roles(mc, alive)

        # medoids, radii, soft membership
        med_idx: List[int] = []
        d90s: List[float] = []
        for c in clusters:
            m, r = _medoid(D, c)
            med_idx.append(m)
            d90s.append(r)
        k_of = {ids[j]: j for j in range(len(ids))}
        if clusters:
            Dm = D[:, med_idx]
            logit = -Dm / np.asarray(d90s)[None, :]
            P = np.exp(logit - logit.max(axis=1, keepdims=True))
            P /= P.sum(axis=1, keepdims=True)
        else:
            Dm = np.zeros((len(descs), 0))
            P = np.zeros((len(descs), 0))

        # roles: stats, super level (hysteresis), descriptor of the medoid, name
        roles: Dict[str, Dict[str, Any]] = mc["roles"]
        for j, c in enumerate(clusters):
            rid = ids[j]
            r = roles.setdefault(rid, {})
            A = float(np.mean([descs[i].A for i in c]))
            r["super"] = _super(r.get("super"), A)
            mdesc = descs[med_idx[j]]
            r.update(A=round(A, 4), d90=round(d90s[j], 4), medoid=mdesc.key,
                     cluster=sorted(new_sets[j]), absent=0,
                     desc=_desc_to_stored(mdesc, [descs[i] for i in c]),
                     version=int(r.get("version", 0)) + (1 if set(r.get("cluster") or []) != new_sets[j] else 0))
            r.setdefault("lineage", [])
            r.setdefault("next_sub", 1)
            r.setdefault("born", now)
        for j, c in enumerate(clusters):
            rest = [i for i in range(len(descs)) if i not in set(c)]
            roles[ids[j]].update(_name(roles[ids[j]], [descs[i] for i in c],
                                       [descs[i] for i in rest]))

        # sub-classes inside each role (on the cluster members)
        sub_of: Dict[str, str] = {}
        for j, c in enumerate(clusters):
            sub_of.update(self._subclasses(ctx.store, mc, ids[j], [descs[i] for i in c], now))

        # publish assignments (transitions need 3 consistent runs)
        assign = mc["assign"]
        outliers: Dict[str, float] = st["outliers"]
        for i, d in enumerate(descs):
            prev = assign.get(d.key) or {}
            cand = ids[j_of_label[int(lab[i])]] if lab[i] >= 0 else UNIQUE
            a = {"static": list(prev.get("static") or []), "pool": prev.get("pool")}
            prole = prev.get("role")
            established = (not prev.get("provisional") and prole not in (None, "", UNIQUE)
                           and prole in alive)
            pend = None
            role = cand
            if P.shape[1]:
                j_arg = int(np.argmax(P[i]))
                arg, p_arg = ids[j_arg], float(P[i, j_arg])
            else:
                arg, p_arg = None, 0.0
            if established and cand != prole and split_kids.get(cand) != prole \
                    and merged_into.get(prole) != cand:
                old_p = prev.get("pend") or [None, 0]
                if arg is not None and arg != prole and p_arg >= TRANSITION_P:
                    pend = [arg, int(old_p[1]) + 1 if old_p[0] == arg else 1]
                if pend is not None and pend[1] >= TRANSITION_RUNS:
                    role = pend[0]
                    events.append(self._event(
                        d.s, d.e, now, "class_transition", Severity.LOW, p_arg,
                        f"role changed '{roles.get(prole, {}).get('name', prole)}' -> "
                        f"'{roles[role].get('name', role)}' (p={p_arg:.2f} on "
                        f"{TRANSITION_RUNS} runs)",
                        {"from": prole, "to": role, "prob": round(p_arg, 4),
                         "runs": TRANSITION_RUNS}, mc))
                    pend = None
                else:
                    role = prole
            elif established and cand != prole:
                role = cand                     # split child / merge: relabel directly
            if role == UNIQUE:
                sup = _super(prev.get("super"), d.A)
                near = float(np.min(Dm[i] / np.asarray(d90s))) if Dm.shape[1] else math.inf
                prob = 1.0 - math.exp(-near) if math.isfinite(near) else 1.0
                sub = None
                if d.key not in outliers:
                    events.append(self._event(
                        d.s, d.e, now, "peer_outlier", Severity.INFO, prob,
                        "entity resembles no role (HDBSCAN noise); typed 'unique'",
                        {"prob": round(prob, 4), "fallback": fell_back,
                         "nearest": ids[int(np.argmin(Dm[i]))] if Dm.shape[1] else None,
                         "D_nearest": _f(np.min(Dm[i])) if Dm.shape[1] else None}, mc))
                outliers[d.key] = now
            else:
                outliers.pop(d.key, None)
                sup = roles[role]["super"]
                prob = float(P[i, k_of[role]])
                sub = sub_of.get(d.key) if role == cand else prev.get("sub")
                if sub is not None and sub not in mc["subs"]:
                    sub = None                  # held in a role whose subs moved on
            a.update(role=role, sub=sub, prob=round(prob, 4), super=sup,
                     class_path=f"{sup}/{role}" + (f"/{sub}" if sub else ""))
            if pend is not None:
                a["pend"] = pend
            assign[d.key] = a

        for kind, rid, others, members in lineage_ev:
            systems = sorted({k.partition("|")[0] for k in members})
            for s in systems:
                ent = role_key(rid)
                events.append(self._event(
                    s, ent, now, kind, Severity.INFO, 1.0,
                    (f"role {rid} split into {', '.join(others)}" if kind == "class_split"
                     else f"roles {', '.join(others)} merged into {rid}"),
                    {"role": rid, ("into" if kind == "class_split" else "from"): others,
                     "members": [k for k in members if k.startswith(s + "|")]}, mc))

    def _subclasses(self, store, mc: Dict[str, Any], rid: str, ds: List[_Desc],
                    now: float) -> Dict[str, str]:
        """Level 2 inside role rid; returns {key: sub id}."""
        role = mc["roles"][rid]
        # the role's busiest bin48: where every member's predictive holds data
        s48 = _nanmean_rows([d.s48 for d in ds])
        b = int(np.nanargmax(s48)) if np.any(np.isfinite(s48)) else ds[0].bucket
        by_sys: Dict[str, List[int]] = {}
        for i, d in enumerate(ds):
            by_sys.setdefault(d.s, []).append(i)
        for s, idx in by_sys.items():
            med, sd = MB.profile_many(store, s, [ds[i].e for i in idx],
                                      [store.get_model(s, ds[i].e, BASELINE) for i in idx],
                                      _bucket_tctx(b))
            for r, i in enumerate(idx):
                ds[i].med, ds[i].sd = med[r], sd[r]
        Dind = ind_distance(ds)
        lab = level2(Dind)
        groups = [sorted(np.nonzero(lab == c)[0].tolist()) for c in sorted(set(lab.tolist()))]
        sets = [{ds[i].key for i in g} for g in groups]
        old = {sid: set(x.get("members") or []) for sid, x in mc["subs"].items()
               if x.get("role") == rid}
        ids, _, _ = match_ids(sets, old)
        for sid in old:
            if sid not in ids:
                del mc["subs"][sid]
        out: Dict[str, str] = {}
        for j, g in enumerate(groups):
            sid = ids[j]
            if sid is None:
                sid = f"{rid}.{int(role.get('next_sub', 1))}"
                role["next_sub"] = int(role.get("next_sub", 1)) + 1
            m, r90 = _medoid(Dind, g)
            prev = mc["subs"].get(sid) or {}
            top = sorted(ds[m].full.items(), key=lambda kv: -kv[1])[:1]
            mc["subs"][sid] = {
                "role": rid, "members": sorted(sets[j]), "medoid": ds[m].key,
                "d90": round(r90, 4), "lineage": list(prev.get("lineage") or []),
                "version": int(prev.get("version", 0)) + (1 if set(prev.get("members") or []) != sets[j] else 0),
                "name": f"{role.get('name', rid)} / {top[0][0] if top else 'individual'}"}
            for i in g:
                out[ds[i].key] = sid
        role["subs"] = sorted(set(out.values()))
        return out

    @staticmethod
    def _age_roles(mc: Dict[str, Any], alive: Set[str]) -> None:
        """Absent roles age; after RETIRE_RUNS absent runs they retire and
        their held members lose the role."""
        for rid in list(mc["roles"]):
            if rid in alive:
                continue
            r = mc["roles"][rid]
            r["absent"] = int(r.get("absent", 0)) + 1
            if r["absent"] >= RETIRE_RUNS:
                del mc["roles"][rid]
                for sid in [s for s, x in mc["subs"].items() if x.get("role") == rid]:
                    del mc["subs"][sid]
                for a in mc["assign"].values():
                    if a.get("role") == rid:
                        a.update(role=None, sub=None, class_path=f"{a.get('super', 'human')}/none")

    @staticmethod
    def _publish_members(mc: Dict[str, Any]) -> None:
        mem: Dict[str, List[str]] = {}
        for k, a in mc["assign"].items():
            r = a.get("role")
            if r not in (None, "", UNIQUE):
                mem.setdefault(str(r), []).append(k)
        for rid, r in mc["roles"].items():
            r["members"] = sorted(mem.get(rid, []))

    # ------------------------------------------------------ statics / pools
    def _ip_classes(self, cfg: Any) -> List[Tuple[str, List[Any], List[str], Any]]:
        sig = repr(cfg)
        if self._ip_cache[0] == sig:
            return self._ip_cache[1]
        out = []
        for c in cfg or []:
            if not isinstance(c, Mapping) or not c.get("name"):
                continue
            nets = []
            for cidr in c.get("cidrs") or []:
                try:
                    nets.append(ipaddress.ip_network(str(cidr), strict=False))
                except ValueError:
                    continue
            out.append((str(c["name"]), nets, [str(x) for x in c.get("systems") or []],
                        c.get("criticality")))
        self._ip_cache = (sig, out)
        return out

    def _statics(self, ctx: Context, mc: Dict[str, Any], now: float) -> None:
        classes = self._ip_classes(ctx.config.get("ip_classes"))
        statics: Dict[str, Any] = {}
        if not classes:                          # config emptied: nobody is static any more
            for a in mc["assign"].values():
                a["static"] = []
            mc["statics"] = statics
            return
        for name, nets, systems, crit in classes:
            statics[name] = {"cidrs": [str(n) for n in nets], "systems": systems,
                             "criticality": crit, "members": []}
        store = ctx.store
        for s in store.systems():
            for e in store.entities(s):
                ip = _ip(e)
                names = []
                if ip is not None:
                    for name, nets, systems, _ in classes:
                        if (not systems or s in systems) and any(ip in n for n in nets
                                                                 if n.version == ip.version):
                            names.append(name)
                            statics[name]["members"].append(assign_key(s, e))
                key = assign_key(s, e)
                a = mc["assign"].get(key)
                if a is None and names:
                    a = mc["assign"][key] = {"role": None, "sub": None, "prob": 0.0,
                                             "static": [], "pool": None, "super": "human",
                                             "class_path": "human/none", "provisional": True}
                if a is not None:
                    a["static"] = names
        mc["statics"] = statics

    def _pools(self, ctx: Context, mc: Dict[str, Any], now: float) -> None:
        store = ctx.store
        scopes = []
        for c in ctx.config.get("dhcp_scopes") or []:
            try:
                scopes.append(ipaddress.ip_network(str(c.get("cidr") if isinstance(c, Mapping)
                                                       else c), strict=False))
            except (ValueError, TypeError):
                continue
        cand: Dict[Tuple[str, str], List[str]] = {}
        linked: Dict[str, Set[str]] = {}
        for s in store.systems():
            linked[s] = _linkable(store.get_model(s, SYSTEM_KEY, LINK))
            for e in store.entities(s):
                ip = _ip(e)
                fs, ls = store.first_seen(s, e), store.last_seen(s, e)
                if ip is None or fs is None or ls is None or ls - fs > POOL_SHORT_S:
                    continue
                net = next((n for n in scopes if n.version == ip.version and ip in n), None)
                if net is None:
                    net = ipaddress.ip_network(f"{ip}/{24 if ip.version == 4 else 64}", strict=False)
                cand.setdefault((s, str(net)), []).append(e)
        pools: Dict[str, Any] = {}
        pooled: Dict[str, str] = {}
        for (s, cidr), ents in sorted(cand.items()):
            if len(ents) < POOL_MIN or any(e in linked[s] for e in ents):
                continue
            roles = {(mc["assign"].get(assign_key(s, e)) or {}).get("role") for e in ents}
            if len(roles) != 1 or next(iter(roles)) in (None, "", UNIQUE):
                continue
            pools[cidr] = {"system": s, "role": next(iter(roles)),
                           "members": sorted(assign_key(s, e) for e in ents), "ts": now}
            for e in ents:
                pooled[assign_key(s, e)] = cidr
        for k, a in mc["assign"].items():
            a["pool"] = pooled.get(k)
        mc["pools"] = pools

    # ------------------------------------------------------------- profiles
    def _entity_profile(self, ctx: Context, mc: Mapping[str, Any], s: str, e: str,
                        a: Mapping[str, Any], now: float) -> None:
        store = ctx.store
        p = store.profile(s, e) or EntityProfile(system=s, entity=e)
        role = a.get("role")
        p.archetype = str(a.get("class_path") or "")
        p.archetype_confidence = float(a.get("prob") or 0.0)
        p.extra["peer_group"] = {
            "role": role, "role_name": mc["roles"].get(str(role), {}).get("name", role or ""),
            "sub": a.get("sub"), "prob": a.get("prob"), "static_classes": list(a.get("static") or []),
            "pool": a.get("pool"), "class_path": a.get("class_path"), "super": a.get("super"),
            "provisional": bool(a.get("provisional", False)), "ts": now}
        store.put_profile(p)

    def _class_profiles(self, ctx: Context, mc: Mapping[str, Any], now: float) -> None:
        store = ctx.store
        per_sys: Dict[Tuple[str, str], List[str]] = {}
        for k, a in mc["assign"].items():
            s, _, ip = k.partition("|")
            r = a.get("role")
            if r not in (None, "", UNIQUE):
                per_sys.setdefault((s, role_key(r)), []).append(ip)
            for name in a.get("static") or []:
                per_sys.setdefault((s, static_key(name)), []).append(ip)
            if a.get("pool"):
                per_sys.setdefault((s, pool_key(a["pool"])), []).append(ip)
        for (s, ck), ips in per_sys.items():
            p = store.profile(s, ck) or EntityProfile(system=s, entity=ck)
            info: Dict[str, Any] = {"members": sorted(ips), "n_members": len(ips), "ts": now}
            if ck.startswith("class:static:"):
                name = ck[len("class:static:"):]
                info.update(kind="static", name=name,
                            criticality=mc["statics"].get(name, {}).get("criticality"),
                            cidrs=mc["statics"].get(name, {}).get("cidrs"))
                p.archetype = f"static/{name}"
                p.archetype_confidence = 1.0
            elif ck.startswith("class:pool:"):
                cidr = ck[len("class:pool:"):]
                info.update(kind="pool", cidr=cidr, role=mc["pools"].get(cidr, {}).get("role"))
                p.archetype = f"pool/{cidr}"
                p.archetype_confidence = 1.0
            else:
                rid = ck[len("class:"):]
                r = mc["roles"].get(rid, {})
                probs = [float((mc["assign"].get(f"{s}|{ip}") or {}).get("prob") or 0.0) for ip in ips]
                info.update(kind="role", role=rid, role_name=r.get("name", rid),
                            super=r.get("super"), medoid=r.get("medoid"), d90=r.get("d90"),
                            A=r.get("A"), subs=r.get("subs", []), lineage=r.get("lineage", []),
                            name_parts=r.get("name_parts"), n_members_org=len(r.get("members", [])))
                p.archetype = f"{r.get('super', 'human')}/{rid}"
                p.archetype_confidence = float(np.mean(probs)) if probs else 0.0
            p.sample_count = len(ips)
            p.extra["peer_group"] = info
            store.put_profile(p)

    @staticmethod
    def _event(s: str, e: str, now: float, kind: str, sev: Severity, score: float, desc: str,
               extra: Dict[str, Any], mc: Mapping[str, Any]) -> BehaviorEvent:
        return BehaviorEvent(system=s, entity=e, ts=now, kind=kind,
                             score=float(min(1.0, max(0.0, score))), severity=sev,
                             description=desc, extra=extra, axes=["peer"],
                             dedupe_key=f"{kind}|{s}|{e}|{int(now)}",
                             model_version=int(mc.get("version", 0)) + 1)


# ================================================================== helpers
def _bucket_tctx(b: int) -> Dict[str, Any]:
    """A time context that selects bin48 bucket b only (an atypical day, so
    no bin168 refinement)."""
    return {"bin48": int(b), "bin168": 0, "dow": 0, "day_type": "nonworkday"}


def _nanmean_rows(rows: Sequence[np.ndarray]) -> np.ndarray:
    """Mean of the fully finite rows (all NaN when there is none)."""
    ok = [r for r in rows if np.all(np.isfinite(r))]
    return np.mean(ok, axis=0) if ok else np.full(len(rows[0]) if len(rows) else 48, _NAN)


def _super(prev: Optional[str], A: float) -> str:
    if prev == "machine":
        return "human" if A <= SUPER_LO else "machine"
    if prev == "human":
        return "machine" if A >= SUPER_HI else "human"
    return "machine" if A >= 0.5 else "human"


def _lineage(r: Dict[str, Any], rec: Dict[str, Any]) -> None:
    r["lineage"] = (list(r.get("lineage") or []) + [rec])[-LINEAGE_KEEP:]


def _ip(e: str) -> Optional[Any]:
    try:
        return ipaddress.ip_address(e)
    except ValueError:
        return None


def _linkable(link: Any) -> Set[str]:
    out: Set[str] = set()
    if not isinstance(link, Mapping):
        return out
    links = link.get("links")
    for lk in (links.values() if isinstance(links, Mapping) else links or ()):
        if isinstance(lk, Mapping):
            for k in ("from", "to"):
                if lk.get(k):
                    out.add(str(lk[k]))
    actors = link.get("actors")
    for act in (actors.values() if isinstance(actors, Mapping) else actors or ()):
        mem = act.get("members") if isinstance(act, Mapping) else act
        if isinstance(mem, (list, tuple, set)) and len(mem) > 1:
            out.update(str(x) for x in mem)
    return out


def _desc_to_stored(m: _Desc, members: Sequence[_Desc]) -> Dict[str, Any]:
    """JSON-safe medoid descriptor for cold typing (the family mix and the
    rhythm are the role's mean, the rest the medoid's)."""
    fam: Dict[str, float] = {}
    for d in members:
        for k, v in d.fam.items():
            fam[k] = fam.get(k, 0.0) + v / len(members)
    top = dict(sorted(fam.items(), key=lambda kv: -kv[1])[:FAM_KEEP])
    s48 = _nanmean_rows([d.s48 for d in members])
    return {"A": _f(float(np.mean([d.A for d in members]))),
            "clr": [_f(x) for x in m.clr], "fam": {k: round(v, 5) for k, v in _norm_dist(top).items()},
            "s48": [_f(x, 5) for x in s48] if np.all(np.isfinite(s48)) else None,
            "dev": m.dev}


def _desc_from_stored(x: Mapping[str, Any]) -> _Desc:
    d = _Desc(key="", s="", e="")
    d.A = float(x.get("A") if x.get("A") is not None else _NAN)
    d.clr = np.array([_NAN if v is None else float(v) for v in (x.get("clr") or [None] * 4)])
    d.fam = dict(x.get("fam") or {})
    s48 = x.get("s48")
    d.s48 = np.array([float(v) for v in s48]) if s48 else np.full(48, _NAN)
    d.dev = str(x.get("dev") or "other")
    return d


def _name(role: Mapping[str, Any], mem: Sequence[_Desc], rest: Sequence[_Desc]) -> Dict[str, Any]:
    """Data-driven role name in natural units."""
    fam: Dict[str, float] = {}
    for d in mem:
        for k, v in d.fam.items():
            fam[k] = fam.get(k, 0.0) + v
    dom = max(fam.items(), key=lambda kv: kv[1])[0] if fam else "?"
    s48s = [d.s48 for d in mem if np.all(np.isfinite(d.s48))]
    win = _window(np.mean(s48s, axis=0)) if s48s else "?"
    rates = [d.rate15 for d in mem if math.isfinite(d.rate15)]
    rate = float(np.median(rates)) if rates else _NAN
    tier = _vol_tier(rate)
    cohen: List[Tuple[str, float]] = []
    if len(mem) >= 2 and rest and all(d.med is not None for d in list(mem) + list(rest)):
        M = np.stack([d.med for d in mem])
        R = np.stack([d.med for d in rest])
        okM = np.stack([d.sd < SD_ID_MAX for d in mem]).all(axis=0)
        okR = np.stack([d.sd < SD_ID_MAX for d in rest]).all(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            pooled = np.sqrt(0.5 * (np.nanvar(M, axis=0) + np.nanvar(R, axis=0)) + 1e-3)
            dd = (np.nanmean(M, axis=0) - np.nanmean(R, axis=0)) / pooled
        dd = np.where(okM & okR & np.isfinite(dd), dd, 0.0)
        for f in np.argsort(-np.abs(dd))[:3]:
            if abs(dd[f]) >= COHEN_MIN:
                cohen.append((F.FEATURE_NAMES_V2[int(f)], round(float(dd[f]), 2)))
    kind = "automated" if role.get("super") == "machine" else "human"
    parts = {"kind": kind, "family": dom, "window": win, "volume_tier": tier,
             "events_per_15min": _f(rate, 1), "cohen_d": cohen}
    tag = ",".join(("+" if v > 0 else "-") + n for n, v in cohen)
    rate_s = f"~{rate:.0f}/15min" if math.isfinite(rate) else "?"
    name = f"{kind} {dom} {win} {tier}({rate_s})" + (f" [{tag}]" if tag else "")
    return {"name": name, "name_parts": parts}


def _new_cold(now: float) -> Dict[str, Any]:
    return {"n": 0, "first": now, "last": now, "bins": [0.0] * 48, "ch": [0.0] * 4,
            "vol": 0.0, "a": {k: [0.0, 0] for k in ("reg", "per", "pe", "think")},
            "dims": {d: {} for d in FAM_DIMS}, "ua": {}}


def _put(dims: Dict[str, Dict[str, float]], dim: str, v: Any, n: Any) -> None:
    try:
        c = float(n)
    except (TypeError, ValueError):
        return
    if not (c > 0.0 and math.isfinite(c)) or v is None or v == OTHER:
        return
    d = dims[dim]
    v = str(v)
    d[v] = d.get(v, 0.0) + c
    if len(d) > COLD_DIM_CAP:
        del d[min(d, key=d.get)]


def _cold_update(cs: Dict[str, Any], store, s: str, e: str, now: float, dt: float,
                 tctx: Mapping[str, Any]) -> None:
    """Fold one active tick of raw evidence into a cold record."""
    cs["n"] = int(cs["n"]) + 1
    cs["last"] = now
    cs["bins"][int(tctx["bin48"])] += 1.0
    nat = store.vec_at(s, e, NAT, now)
    if nat is not None:
        x = np.asarray(nat, dtype=np.float64)
        for i, j in enumerate(CH_IDX):
            v = x[j]
            cs["ch"][i] += float(v) * 60.0 / dt if math.isfinite(v) else 0.0   # stale count = 0
        if math.isfinite(x[I_INT]):
            cs["vol"] += float(x[I_INT]) * 60.0 / dt
        for k, j in (("reg", I_REG), ("per", I_PER), ("pe", I_PE), ("think", I_THINK)):
            v = x[j]
            if math.isfinite(v) and (k != "think" or v > 0.0):
                acc = cs["a"][k]
                acc[0] += math.log(v) if k == "think" else float(v)
                acc[1] += 1
    dims = cs["dims"]
    toks = store.latest_fresh(s, e, "act.tokens", now)
    if isinstance(toks, dict):
        for t, n in toks.items():
            if isinstance(t, str) and t and t[0] != "{" and t != OTHER and channel_of(t) == "http":
                _put(dims, "tmpl", MT.template_key(t), n)
    for dim, name in RAW_DIMS:
        x = store.latest_fresh(s, e, name, now)
        if isinstance(x, dict):
            for v, n in x.items():
                _put(dims, dim, v, n)
    stacks = store.latest_fresh(s, e, "client.stack_set", now)
    if stacks is not None:
        cnt = MC.stack_counts(stacks)
        tot = sum(c for c in cnt.values() if c > 0.0)
        if tot > 0.0:
            for tok, c in cnt.items():
                if c > 0.0:
                    cls = _ua_class(MC.parse(tok).family)
                    cs["ua"][cls] = cs["ua"].get(cls, 0.0) + c / tot
