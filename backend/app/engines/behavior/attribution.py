"""AttributionEngine (B16): who is behind this IP's traffic right now?

Why: an IP is an address, not a person. When a stolen session or credential
is replayed from another desk, when a known host is swapped for another
behind the same address, or when a role account is taken over, the IP's own
detectors (all normalised to the IP itself) may see nothing unusual in
volume or shape. The question has to be asked the other way round: under
whose model is this window of behaviour most likely, and is that the IP
itself, somebody else we know, or nobody we know (open set)?

Identity is ABSOLUTE (architecture §0 rule 7, decisions.md). Every modality
scores the window under each candidate's OWN models on absolute data:
feature.vec, feature.sketch, behavior.timing, act.tokens, act.stream,
client.stack_set and the activity clock. It never reads behavior.z / zr / zi,
which are normalised to each entity and remove exactly the between-entity
information attribution needs (a static test asserts this).

Per system and tick (docs/lib3/engines.md '## B16'):
  1. Window: the entity's last K = 4 active ticks (feature.active), sliding
     by one active tick. Idle ticks abstain: nothing is scored and the
     CUSUMs keep their value; idle slots inside the window span still enter
     the rhythm (presence) term.
  2. Window vector (B15's recipe, absolute): the median of feature.vec (NaN
     filled with the class bucket fill, plus a missing mask), the IQR of 8
     key features, the mean sketch, behavior.timing (B, M, ln think_mu), and
     sin/cos of the local hour at the window centre with a workday flag.
     It is projected with model.identity (PCA -> WCCN/LDA; within-entity
     scatter is the identity in that space).
  3. Candidates: self, the top 5 other enrolled entities by Gaussian
     log-likelihood in that space, the own role class and the best other
     role class.
  4. Modality log-likelihood ratios, each against the system background
     (the "nobody known" reference) and each under the candidate's model:
       gauss   N(z; mu_j, I) vs N(z; mu_0, (1 + tau0^2) I)   (classes: (1 + tau_c^2) I)
       vocab   multinomial over act.tokens (m_vocab chain vs system tier),
               per-token mean x min(n_tok, 20)
       rhythm  Bernoulli presence of the window's 15-min slots (m_rhythm)
       seq     PPM surprisal of the window's act.stream symbols (m_seq),
               per-symbol mean x min(n, 20); computed at window completion
       client  stack mix (m_client chain vs system), per-request mean x the
               number of window ticks with client traffic (slot-equivalents)
       timing  gap histogram (m_timing) vs the pooled system gap pmf,
               per-gap mean x min(n, 20); computed at window completion
     Each is calibrated a_m * llr + b_m (model.identity.llr_calib, B15's
     cross-validated logistic fit) and capped at +-4 nats, so one modality
     (a browser upgrade, a new client library) can never flip identity on
     its own. A modality a candidate has no model for contributes 0 (the
     background), never a guess.
  5. Posterior over the candidates plus an explicit unknown hypothesis
     (log-evidence 0 = the background; prior 0.05; self 0.5; the other
     candidates share 0.45). Linked aliases (B17 continuity) count as self.
     score.identity = -log10 pi_self (instantaneous, axis identity; B24
     calibrates it with (daypart, regime) strata); pm.identity = pi_self, a
     conservative small-sample prior for B24.
  6. Other-identity CUSUM: lambda = clip(max_{j != e} L_j - L_e, -4, 8),
     S = max(0, S + lambda - 0.5), h = seq.h_for('llr', 100 d) counted in the
     entity's windows per day. identity_mismatch when S >= h: MEDIUM, HIGH
     when j* is active elsewhere at the same time (impersonation); INFO
     unless j* won >= 3 of the last 4 windows, pi_j* >= 0.9 and the CV
     confusion(e, j*) < 0.1 (and j* is not in e's anonymity set).
  7. Unknown CUSUM: +1 when max_j p_j < 0.01 (typicality chi2 p of the window
     under each individual candidate), else -0.5; unknown_identity at 2:
     HIGH when the class typicality p < 0.01 ('unlike anyone'), MEDIUM for
     'same class, different individual', INFO for a young (not enrolled) IP.
  8. An entity with continuity.shared_ip (or entity_kind 'ip-class', B17)
     gets every event downgraded to INFO. The common-mode flag is never read:
     it must not suppress identity.

Training mode scores and runs the CUSUMs but emits no events (a chart that
would alarm is reset silently, so warm-up never leaves it primed). B16 owns
no model; its chart state lives in the engine and is mirrored in behavior.id
so a restarted engine resumes it. A failed B01 tick writes NaN + degraded.

Store: reads feature.active, feature.vec, feature.sketch, feature.tctx,
act.tokens, act.stream, act.stream_frac, client.stack_set, behavior.timing,
model.identity, model.vocab, model.rhythm, model.seq, model.client,
model.timing, model.class, model.link, profile.extra.continuity. Writes
behavior.score[identity] / pm / axes / degraded (lib/emit), behavior.id
(dict series), profile.extra.attribution; events identity_mismatch,
unknown_identity.

model.identity layout read here (contract C field names; B15 owns it and its
accessor lib/m_identity.py; when that module provides `window_vector` and
`project`, they are used instead of the local reading below):
    {'pca': {'mean': [D], 'scale': [D] (optional), 'components': [[D]] x d},
     'W': [[k]] x d (projection; within-entity scatter = I after it),
     'means': {entity: [k]}, 'class_means': {'class:<rid>': [k]},
     'class_tau2': {'class:<rid>': float} (optional), 'fill': {key: [52]},
     'llr_calib': {m: [a, b]}, 'confusion': {e: {j: rate}},
     'anonymity_sets': [[entity, ...]], 'version': int}
Missing 'pca' / 'W' mean identity maps; a dimension mismatch between the
window vector and the model degrades the detector instead of guessing.
"""
from __future__ import annotations

import importlib
import math
import warnings
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.special import chdtrc

from ...core.engine import Context, Engine
from ...models.schema import BehaviorEvent, DerivedMetric, EntityProfile, MetricKind, Severity
from .lib import emit
from .lib import features as F
from .lib import m_class
from .lib import m_client as MC
from .lib import m_rhythm as MR
from .lib import m_seq as MS
from .lib import m_template as MT
from .lib import m_timing as MTI
from .lib import m_vocab as MV
from .lib import seq as SQ
from .lib import timebins as TB
from .lib.classkeys import SYSTEM_KEY, is_class, role_key
from .lib.template import channel_of

try:                                   # B15's accessor module (optional until it lands)
    MI: Any = importlib.import_module(f"{__package__}.lib.m_identity")
except ModuleNotFoundError as _exc:    # only the module itself being absent is tolerated
    if not str(getattr(_exc, "name", "")).endswith("m_identity"):
        raise
    MI = None

DETECTOR = "identity"
AXES = ["identity"]
B01_ENGINE = "behavior.feature_vector"
ID_MODEL = "model.identity"
LINK_MODEL = "model.link"
ID_SERIES = "behavior.id"
ACTIVE, VEC, SKETCH = "feature.active", "feature.vec", "feature.sketch"
TIMING = "behavior.timing"

K = 4                                  # active ticks per window
MODALITIES = ("gauss", "vocab", "rhythm", "seq", "client", "timing")
CAP = 4.0                              # nats per modality per window
N_EFF_CAP = 20.0                       # effective sample cap (vocab, seq, timing)
N_TOP = 5                              # other entities by LDA score
UNKNOWN_PRIOR = 0.05
SELF_PRIOR = 0.5
LAMBDA_LO, LAMBDA_HI, CUSUM_K = -4.0, 8.0, 0.5
ARL_DAYS = 100.0
U_UP, U_DOWN, U_H = 1.0, 0.5, 2.0
P_TYPICAL = 0.01
POST_MIN, WINS_MIN, CONF_MAX = 0.9, 3, 0.1
LOOKBACK_S = 86400.0                   # activity history for windows / day (feature.vec 1 d)
RHYTHM_MAX_SLOTS = 8
SEQ_MAX, GAP_MAX = 128, 128
ALERT_HOLD_S = 6 * 3600.0
PROFILE_EVERY_S = 3600.0
AXES_POST_MAX = 0.5                    # pi_self below this -> identity axis contributes
KEY8 = tuple(F.KEY_FEATURE_IDX[:8])
WINDOW_DIM = 2 * F.FEATURE_DIM + len(KEY8) + 80 + 3 + 3
_NAN = math.nan
_LN2 = math.log(2.0)


# ============================================================ small helpers
def _f(x: Any) -> float:
    if isinstance(x, bool) or x is None:
        return _NAN
    try:
        return float(x)
    except (TypeError, ValueError, OverflowError):
        return _NAN


def _int(x: Any) -> int:
    v = _f(x)
    return int(v) if math.isfinite(v) else 0


def _json(v: Any, nd: int = 4) -> Any:
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return round(v, nd) if math.isfinite(v) else None
    return v


def _arr(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        a = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return a


def _chi2_sf(x: float, k: int) -> float:
    if not (x == x) or k < 1:
        return _NAN
    return float(max(chdtrc(k, max(x, 0.0)), 1e-300))


def calibrate(llr: float, ab: Tuple[float, float]) -> float:
    """a_m * llr + b_m capped at +-CAP nats; NaN (no model / no data) -> 0,
    i.e. the modality is neutral (the background), never a guess."""
    if not (llr == llr):
        return 0.0
    v = ab[0] * llr + ab[1]
    return CAP if v > CAP else (-CAP if v < -CAP else v)


def scaled_mean(llr_terms: np.ndarray, cap: float = N_EFF_CAP) -> float:
    """Mean per-item LLR times min(n, cap): items inside a window (tokens,
    symbols, gaps) are not independent draws, so evidence grows with the
    number of items only up to an effective sample of `cap`."""
    t = llr_terms[np.isfinite(llr_terms)]
    if not t.size:
        return _NAN
    return float(t.mean()) * min(float(t.size), cap)


# ============================================================ identity model view
class IdModel:
    """Numeric view of model.identity for one tick (projection, enrolled
    means, class means and scatter, calibration, confusion)."""

    __slots__ = ("raw", "version", "mean", "scale", "comp", "W", "k", "ents", "M",
                 "idx", "classes", "CM", "ctau2", "mu0", "tau0", "calib", "conf",
                 "anon", "fill", "dim")

    def __init__(self) -> None:
        self.raw: Mapping[str, Any] = {}
        self.version = 0
        self.mean = self.scale = self.comp = self.W = None
        self.k = 0
        self.dim: Optional[int] = None
        self.ents: List[str] = []
        self.M = np.zeros((0, 0))
        self.idx: Dict[str, int] = {}
        self.classes: List[str] = []
        self.CM = np.zeros((0, 0))
        self.ctau2: Dict[str, float] = {}
        self.mu0 = np.zeros(0)
        self.tau0 = 0.0
        self.calib: Dict[str, Tuple[float, float]] = {}
        self.conf: Mapping[str, Any] = {}
        self.anon: Dict[str, Set[str]] = {}
        self.fill: Mapping[str, Any] = {}

    @classmethod
    def build(cls, model: Any) -> Optional["IdModel"]:
        if not isinstance(model, Mapping):
            return None
        v = cls()
        v.raw = model
        v.version = _int(model.get("version"))
        pca = model.get("pca")
        if isinstance(pca, Mapping):
            v.mean = _arr(pca.get("mean"))
            v.scale = _arr(pca.get("scale"))
            v.comp = _arr(pca.get("components"))
            if v.comp is not None and v.comp.ndim != 2:
                v.comp = None
        v.W = _arr(model.get("W"))
        if v.W is not None and v.W.ndim != 2:
            v.W = None
        means = model.get("means") or {}
        ents, rows = [], []
        for e, m in (means.items() if isinstance(means, Mapping) else ()):
            a = _arr(m)
            if a is not None and a.ndim == 1 and a.size and np.all(np.isfinite(a)):
                ents.append(str(e))
                rows.append(a)
        if not rows:
            return None
        k = rows[0].size
        keep = [i for i, r in enumerate(rows) if r.size == k]
        v.ents = [ents[i] for i in keep]
        v.M = np.vstack([rows[i] for i in keep])
        v.k = k
        v.idx = {e: i for i, e in enumerate(v.ents)}
        cm = model.get("class_means") or {}
        cks, crow = [], []
        for c, m in (cm.items() if isinstance(cm, Mapping) else ()):
            a = _arr(m)
            if a is not None and a.ndim == 1 and a.size == k and np.all(np.isfinite(a)):
                cks.append(str(c))
                crow.append(a)
        v.classes = cks
        v.CM = np.vstack(crow) if crow else np.zeros((0, k))
        ct = model.get("class_tau2") or {}
        for c in cks:
            t = _f(ct.get(c)) if isinstance(ct, Mapping) else _NAN
            v.ctau2[c] = t if t == t and t >= 0.0 else _NAN
        v.mu0 = v.M.mean(axis=0)
        v.tau0 = float(((v.M - v.mu0) ** 2).sum(axis=1).mean() / k) if len(v.ents) > 1 else 1.0
        cal = model.get("llr_calib") or {}
        for m in MODALITIES:
            ab = cal.get(m) if isinstance(cal, Mapping) else None
            a, b = (_f(ab[0]), _f(ab[1])) if isinstance(ab, (list, tuple)) and len(ab) >= 2 \
                else (1.0, 0.0)
            v.calib[m] = (a if a == a else 1.0, b if b == b else 0.0)
        conf = model.get("confusion")
        v.conf = conf if isinstance(conf, Mapping) else {}
        for grp in model.get("anonymity_sets") or ():
            if isinstance(grp, (list, tuple, set)) and len(grp) > 1:
                g = {str(x) for x in grp}
                for x in g:
                    v.anon.setdefault(x, set()).update(g - {x})
        fill = model.get("fill")
        v.fill = fill if isinstance(fill, Mapping) else {}
        if v.mean is not None:
            v.dim = int(v.mean.size)
        elif v.comp is not None:
            v.dim = int(v.comp.shape[1])
        return v

    # ------------------------------------------------------------ projection
    def project(self, x: np.ndarray) -> Optional[np.ndarray]:
        """Absolute window vector -> identity space (None on a layout mismatch)."""
        if MI is not None and hasattr(MI, "project"):
            z = MI.project(self.raw, x)
            return None if z is None else np.asarray(z, dtype=np.float64)
        y = np.asarray(x, dtype=np.float64)
        if self.mean is not None:
            if self.mean.size != y.size:
                return None
            y = y - self.mean
        if self.scale is not None and self.scale.size == y.size:
            y = y / np.where(self.scale > 0.0, self.scale, 1.0)
        if self.comp is not None:
            if self.comp.shape[1] != y.size:
                return None
            y = self.comp @ y
        if self.W is not None:
            if self.W.shape[0] != y.size:
                return None
            y = y @ self.W
        if y.size != self.k or not np.all(np.isfinite(y)):
            return None
        return y

    def fill_for(self, ck: Optional[str]) -> Optional[np.ndarray]:
        for key in ((ck,) if ck else ()) + (SYSTEM_KEY,):
            a = _arr(self.fill.get(key))
            if a is not None and a.size == F.FEATURE_DIM:
                return a
        return None

    def class_tau2(self, ck: str, members: Sequence[str]) -> float:
        """Member-mean scatter around the class mean (per dimension)."""
        t = self.ctau2.get(ck, _NAN)
        if t == t:
            return t
        ci = self.classes.index(ck)
        rows = [self.idx[m] for m in members if m in self.idx]
        if len(rows) < 2:
            t = self.tau0
        else:
            d = self.M[rows] - self.CM[ci]
            t = float((d ** 2).sum(axis=1).mean() / self.k)
        self.ctau2[ck] = t
        return t

    def confusion(self, e: str, j: str) -> float:
        if j in self.anon.get(e, ()):
            return 1.0
        row = self.conf.get(e)
        v = _f(row.get(j)) if isinstance(row, Mapping) else _NAN
        return v if v == v else 0.0


# ============================================================ window vector
def window_vector(vec_rows: np.ndarray, sketch_rows: Optional[np.ndarray],
                  timing: Optional[Mapping[str, Any]], hour: float, workday: bool,
                  fill: Optional[np.ndarray]) -> np.ndarray:
    """B15's absolute window vector (engines.md '## B15'), WINDOW_DIM long:
    median feature.vec with NaN filled (class bucket fill, else 0) + missing
    mask, IQR of 8 key features, mean sketch, timing (B, M, ln think_mu), and
    sin/cos hour at the window centre + workday flag."""
    V = np.asarray(vec_rows, dtype=np.float64).reshape(-1, F.FEATURE_DIM)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN columns
        med = np.nanmedian(V, axis=0) if V.shape[0] else np.full(F.FEATURE_DIM, np.nan)
        q = np.nanpercentile(V[:, list(KEY8)], [25.0, 75.0], axis=0) if V.shape[0] \
            else np.full((2, len(KEY8)), np.nan)
    miss = ~np.isfinite(med)
    if miss.any():
        fv = np.zeros(F.FEATURE_DIM) if fill is None else np.nan_to_num(fill, nan=0.0)
        med = np.where(miss, fv, med)
    iqr = np.nan_to_num(q[1] - q[0], nan=0.0)
    if sketch_rows is not None and len(sketch_rows):
        S = np.asarray(sketch_rows, dtype=np.float64).reshape(len(sketch_rows), -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sk = np.nan_to_num(np.nanmean(S, axis=0), nan=0.0)
        if sk.size != 80:
            sk = np.resize(sk, 80)
    else:
        sk = np.zeros(80)
    tm = np.zeros(3)
    if isinstance(timing, Mapping):
        mu = _f(timing.get("think_mu"))
        tm = np.array([_f(timing.get("B")), _f(timing.get("M")),
                       math.log(mu) if mu == mu and mu > 0.0 else _NAN])
        tm = np.nan_to_num(tm, nan=0.0, posinf=0.0, neginf=0.0)
    h = hour if hour == hour else 12.0
    tt = np.array([math.sin(2 * math.pi * h / 24.0), math.cos(2 * math.pi * h / 24.0),
                   1.0 if workday else 0.0])
    return np.concatenate([med, miss.astype(np.float64), iqr, sk, tm, tt])


# ============================================================ per-tick system cache
class _SysCache:
    """Candidate models resolved once per system and tick: every entity's
    candidates overlap, so each chain / tier / pmf is built at most once."""

    def __init__(self, store: Any, s: str, now: float, view: IdModel) -> None:
        self.store, self.s, self.now, self.view = store, s, now, view
        self._vocab: Dict[str, Any] = {}
        self._client: Dict[str, Any] = {}
        self._rhythm: Dict[str, Any] = {}
        self._seq: Dict[str, Any] = {}
        self._timing: Dict[str, Any] = {}
        self._ck: Dict[str, Optional[str]] = {}
        self._members: Dict[str, List[str]] = {}
        self.sys_client = MC.get(store, s, SYSTEM_KEY)
        self.sys_vocab = MV.get(store, s, SYSTEM_KEY)
        self._bg_vocab: Optional[MV.Backoff] = None
        self._bg_client: Optional[MC.Backoff] = None
        self._bg_rhythm: Any = False
        self._bg_timing: Any = False
        self._seq_sys: Any = False
        self._vocab_size: Optional[int] = None
        self.smap: Optional[MS.SymbolMap] = None
        self.tz: str = TB.DEFAULT_TZ
        self.calendar: Any = None
        self.healed: Set[int] = set()

    # classes
    def class_key(self, e: str) -> Optional[str]:
        if e not in self._ck:
            self._ck[e] = m_class.class_key(self.store, self.s, e)
        return self._ck[e]

    def role_key(self, e: str) -> Optional[str]:
        rid = m_class.role_id(self.store, self.s, e)
        return role_key(rid) if rid is not None else None

    def members(self, ck: str) -> List[str]:
        if ck not in self._members:
            self._members[ck] = m_class.class_members(self.store, self.s, ck)
        return self._members[ck]

    # vocab
    def vocab(self, j: str) -> Optional[MV.Backoff]:
        if j not in self._vocab:
            ent, cls, sys_ = MV.backoff_models(self.store, self.s, j)
            own = cls if is_class(j) else ent
            self._vocab[j] = MV.Backoff(ent, cls, sys_, now=self.now) if own is not None else None
        return self._vocab[j]

    def bg_vocab(self) -> MV.Backoff:
        if self._bg_vocab is None:
            self._bg_vocab = MV.Backoff(None, None, self.sys_vocab, now=self.now)
        return self._bg_vocab

    # client
    def client(self, j: str) -> Optional[MC.Backoff]:
        if j not in self._client:
            if is_class(j):
                ent, cls = None, MC.class_tier(self.sys_client, j)
                own = cls
            else:
                ent = MC.get(self.store, self.s, j)
                ent = ent if MC.kind(ent) == "entity" else None
                cls = MC.class_tier(self.sys_client, self.class_key(j))
                own = ent
            self._client[j] = MC.Backoff(ent, cls, self.sys_client, now=self.now) \
                if own is not None else None
        return self._client[j]

    def bg_client(self) -> MC.Backoff:
        if self._bg_client is None:
            self._bg_client = MC.Backoff(None, None, self.sys_client, now=self.now)
        return self._bg_client

    # rhythm
    def rhythm(self, j: str) -> Optional[Mapping[str, Any]]:
        if j not in self._rhythm:
            m = self.store.get_model(self.s, j, MR.MODEL)
            self._rhythm[j] = m if isinstance(m, Mapping) and isinstance(m.get("state"), Mapping) \
                else None
        return self._rhythm[j]

    def bg_rhythm(self) -> Optional[Mapping[str, Any]]:
        if self._bg_rhythm is False:
            self._bg_rhythm = self.rhythm(SYSTEM_KEY)
            self.healed = MR.healed_days(self._bg_rhythm)
        return self._bg_rhythm

    # sequence
    def seq_sys(self) -> Any:
        if self._seq_sys is False:
            self._seq_sys = MS.ppm(MS.get(self.store, self.s, SYSTEM_KEY))
            if self._seq_sys is not None and not self._seq_sys.counts:
                self._seq_sys = None
        return self._seq_sys

    def seq(self, j: str) -> Optional[Tuple[Any, List[Any]]]:
        if j not in self._seq:
            p = MS.ppm(MS.get(self.store, self.s, j))
            if p is None or not p.counts:
                self._seq[j] = None
            else:
                tiers = [self.seq_sys()] if is_class(j) else MS.backoff(self.store, self.s, j)
                self._seq[j] = (p, [t for t in tiers if t is not None])
        return self._seq[j]

    def vocab_size(self) -> int:
        if self._vocab_size is None:
            self._vocab_size = MS.vocab_size(self.store, self.s)
        return self._vocab_size

    # timing
    def timing_logp(self, j: str) -> Optional[np.ndarray]:
        """ln pmf of the candidate's gap histogram (entities only)."""
        if j not in self._timing:
            out = None
            if not is_class(j):
                m = self.store.get_model(self.s, j, MTI.MODEL)
                if m is not None and not MTI.is_empty(m):
                    p = MTI.pmf(m)
                    if np.all(np.isfinite(p)):
                        out = np.log(p)
            self._timing[j] = out
        return self._timing[j]

    def bg_timing(self) -> Optional[np.ndarray]:
        """ln of the pooled (mean) gap pmf of the system's timing models."""
        if self._bg_timing is False:
            acc, n = np.zeros(MTI.N_BINS), 0
            for e in self.store.entities(self.s):
                lp = self.timing_logp(e)
                if lp is not None:
                    acc += np.exp(lp)
                    n += 1
            self._bg_timing = np.log(acc / n) if n else None
        return self._bg_timing


# ============================================================ modality helpers
def vocab_llr(bo: Optional[MV.Backoff], bg: MV.Backoff, obs: Mapping[str, Mapping[str, float]]
              ) -> float:
    """Per-token mean ln(p_j / p_bg) x min(n_tok, 20); NaN without a model or tokens."""
    if bo is None or not obs:
        return _NAN
    terms, w = [], []
    for dim, vals in obs.items():
        for v, n in vals.items():
            pj, pb = bo.p(dim, v), bg.p(dim, v)
            terms.append(math.log(max(pj, 1e-300)) - math.log(max(pb, 1e-300)))
            w.append(n)
    if not w:
        return _NAN
    wa = np.asarray(w)
    n_tok = float(wa.sum())
    return float(np.dot(wa, terms) / n_tok) * min(n_tok, N_EFF_CAP)


def client_llr(bo: Optional[MC.Backoff], bg: MC.Backoff, counts: Mapping[str, float],
               n_ticks: int) -> float:
    """Per-request mean ln(p_j / p_bg) x window ticks with client traffic
    (m_client counts are slot-equivalents, not requests)."""
    if bo is None or not counts or n_ticks < 1:
        return _NAN
    tot = sum(counts.values())
    if not tot > 0.0:
        return _NAN
    s = sum(n * (math.log(max(bo.p(t), 1e-300)) - math.log(max(bg.p(t), 1e-300)))
            for t, n in counts.items())
    return s / tot * float(n_ticks)


def rhythm_ll(model: Optional[Mapping[str, Any]], acts: Sequence[float],
              cells: Sequence[Tuple[int, int]]) -> float:
    if model is None or not cells:
        return _NAN
    return MR.loglik(model, acts, cells)


def seq_llr(cand: Optional[Tuple[Any, List[Any]]], sys_ppm: Any, syms: Sequence[str],
            vocab_size: int) -> float:
    if cand is None or not syms:
        return _NAN
    bj = MS.loglik(cand[0], syms, cand[1], vocab_size)
    bb = MS.loglik(sys_ppm, syms, (), vocab_size) if sys_ppm is not None else \
        np.full(len(syms), math.log2(max(1, vocab_size)))
    return scaled_mean((np.asarray(bb) - np.asarray(bj)) * _LN2)


def timing_llr(lp_j: Optional[np.ndarray], lp_bg: Optional[np.ndarray], gaps: np.ndarray
               ) -> float:
    if lp_j is None or lp_bg is None or not gaps.size:
        return _NAN
    b = MTI.bin_index(gaps)
    b = b[b >= 0]
    if not b.size:
        return _NAN
    return scaled_mean(lp_j[b] - lp_bg[b])


# ============================================================ per-entity state
def _new_state() -> Dict[str, Any]:
    return {"S": 0.0, "U": 0.0, "winners": [], "n_act": 0, "ticks": [], "slow": None,
            "hold": {}, "ts": None, "prev": None, "profile_ts": -math.inf}


def _snap(st: Mapping[str, Any]) -> Dict[str, Any]:
    return {"S": st["S"], "U": st["U"], "winners": list(st["winners"]), "n_act": st["n_act"],
            "ticks": list(st["ticks"]), "slow": st["slow"], "hold": dict(st["hold"]),
            "profile_ts": st["profile_ts"]}


def _restore_state(store: Any, s: str, e: str, now: float) -> Dict[str, Any]:
    """Resume the charts from the last behavior.id point before `now` (an
    engine restart must not reset a half-way CUSUM)."""
    st = _new_state()
    for m in reversed(store.derived_tail(s, e, ID_SERIES, 4)):
        if m.ts < now and isinstance(m.value, Mapping):
            v = m.value
            S, U = _f(v.get("cusum_other")), _f(v.get("cusum_new"))
            st["S"] = S if S == S and S >= 0 else 0.0
            st["U"] = U if U == U and U >= 0 else 0.0
            st["winners"] = [str(x) for x in (v.get("winners") or [])][-K:]
            st["n_act"] = _int(v.get("n_act"))
            break
    for ev in store.events(system=s, entity=e, since=now - ALERT_HOLD_S,
                           kinds=["identity_mismatch", "unknown_identity"], limit=50):
        key = ev.extra.get("hold_key") if isinstance(ev.extra, Mapping) else None
        if key:
            st["hold"][str(key)] = max(st["hold"].get(str(key), -math.inf), float(ev.ts))
    return st


def _continuity(store: Any, s: str, e: str) -> Tuple[Set[str], bool]:
    """(aliases of e from B17 continuity / model.link, shared-IP flag)."""
    aliases: Set[str] = set()
    shared = False
    prof = store.profile(s, e)
    c = prof.extra.get("continuity") if prof is not None and isinstance(prof.extra, Mapping) \
        else None
    if isinstance(c, Mapping):
        for k in ("aliases",):
            for a in c.get(k) or ():
                aliases.add(str(a))
        for k in ("linked_from", "linked_to"):
            v = c.get(k)
            if isinstance(v, str) and v:
                aliases.add(v)
            elif isinstance(v, (list, tuple)):
                aliases.update(str(x) for x in v if x)
        shared = bool(c.get("shared_ip")) or c.get("entity_kind") == "ip-class"
    aliases.discard(e)
    return aliases, shared


def _links(store: Any, s: str) -> Dict[str, Set[str]]:
    """Unretracted links of model.link@(s, __system__) as an alias map."""
    out: Dict[str, Set[str]] = {}
    m = store.get_model(s, SYSTEM_KEY, LINK_MODEL)
    links = m.get("links") if isinstance(m, Mapping) else None
    if isinstance(links, Mapping):
        links = list(links.values())
    for ln in links or ():
        a = b = None
        if isinstance(ln, Mapping):
            if ln.get("retracted") or ln.get("status") == "retracted":
                continue
            a = ln.get("from", ln.get("a", ln.get("src")))
            b = ln.get("to", ln.get("b", ln.get("dst")))
        elif isinstance(ln, (list, tuple)) and len(ln) >= 2:
            a, b = ln[0], ln[1]
        if isinstance(a, str) and isinstance(b, str) and a != b:
            out.setdefault(a, set()).add(b)
            out.setdefault(b, set()).add(a)
    return out


# ============================================================ tick inputs
def _vocab_obs(toks: Any) -> Dict[str, Dict[str, float]]:
    """act.tokens -> m_vocab observations: HTTP templates ('tmpl') and TLS
    SNI eTLD+1 ('sni'), the dimensions act.tokens carries."""
    out: Dict[str, Dict[str, float]] = {}
    if not isinstance(toks, Mapping):
        return out
    for t, n in toks.items():
        c = _f(n)
        if not isinstance(t, str) or not t or t[0] == "{" or t == MT.OTHER \
                or not (c > 0.0 and math.isfinite(c)):
            continue
        ch = channel_of(t)
        if ch == "http":
            dim, v = "tmpl", MT.template_key(t)
        elif ch == "tls":
            dim, v = "sni", t.split(":", 1)[0]
        else:
            continue
        d = out.setdefault(dim, {})
        d[v] = d.get(v, 0.0) + c
    return out


def _merge_obs(parts: Sequence[Mapping[str, Mapping[str, float]]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for p in parts:
        for dim, vals in p.items():
            d = out.setdefault(dim, {})
            for v, n in vals.items():
                d[v] = d.get(v, 0.0) + n
    return out


# ============================================================ the engine
class AttributionEngine(Engine):
    name = "behavior.attribution"
    layer = "behavior"
    consumes = [ACTIVE, VEC, SKETCH, "feature.tctx", "act.tokens", "act.stream",
                "act.stream_frac", "client.stack_set", TIMING, ID_MODEL, "model.vocab",
                "model.rhythm", "model.seq", "model.client", "model.timing", "model.class",
                LINK_MODEL, "profile.extra.continuity"]
    produces = ["behavior.score", "behavior.pm", "behavior.axes", "behavior.degraded",
                ID_SERIES, "profile.extra.attribution",
                "event.identity_mismatch", "event.unknown_identity"]
    description = ("Open-set identity attribution per IP: a 4-active-tick absolute window "
                   "scored under self, the nearest known IPs and role classes with calibrated, "
                   "capped modality LLRs (LDA Gaussian, vocab, rhythm, PPM, client, gaps); "
                   "posterior with an unknown hypothesis; other-identity and unknown CUSUMs.")
    interval = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._state: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._view: Dict[str, Tuple[int, Any, Optional[IdModel]]] = {}

    # ------------------------------------------------------------------ run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        now, dt = float(ctx.now), float(ctx.window_s)
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError(f"AttributionEngine: bad ctx.window_s {ctx.window_s!r}")
        b01_failed = store.engine_failed(B01_ENGINE, now)
        return sum(self._system(ctx, s, now, dt, b01_failed) for s in store.systems())

    def _model_view(self, store: Any, s: str) -> Optional[IdModel]:
        m = store.get_model(s, SYSTEM_KEY, ID_MODEL)
        if m is None:
            return None
        ver = store.model_version(s, SYSTEM_KEY, ID_MODEL)
        hit = self._view.get(s)
        if hit is not None and hit[0] == id(m) and hit[1] == ver:
            return hit[2]
        v = IdModel.build(m)
        self._view[s] = (id(m), ver, v)
        return v

    def _system(self, ctx: Context, s: str, now: float, dt: float, b01_failed: bool) -> int:
        store = ctx.store
        view = self._model_view(store, s)
        if view is None:
            return 0                                  # nothing enrolled yet: abstain
        sc = _SysCache(store, s, now, view)
        sc.tz = str(ctx.config.get("tz") or TB.DEFAULT_TZ)
        sc.calendar = TB.parse_calendar(ctx.config.get("calendar"))
        link_map = _links(store, s)
        n = 0
        for e in store.entities(s):
            key = (s, e)
            if b01_failed:
                if key in self._state or store.vec_at(s, e, ACTIVE, now - dt) is not None:
                    emit.write_scores(store, s, e, now, {DETECTOR: _NAN},
                                      degraded={DETECTOR: f"producer_error:{B01_ENGINE}"},
                                      window_s=int(dt))
                    n += 1
                continue
            a = store.vec_at(s, e, ACTIVE, now)
            if a is None or not (float(a[0]) >= 0.5):
                continue                              # no row / idle: abstain
            st = self._state.get(key)
            if st is None:
                st = self._state[key] = _restore_state(store, s, e, now)
            if st["ts"] == now and st["prev"] is not None:
                st.update(_snap(st["prev"]))          # re-run of the same tick
            else:
                st["prev"], st["ts"] = _snap(st), now
            n += self._entity(ctx, sc, e, st, now, dt, link_map)
        return n

    # ------------------------------------------------------------- per entity
    def _entity(self, ctx: Context, sc: _SysCache, e: str, st: Dict[str, Any], now: float,
                dt: float, link_map: Mapping[str, Set[str]]) -> int:
        store, s, view = ctx.store, sc.s, sc.view
        win = int(dt)
        # ---- activity clock: window ticks and windows per day
        ts_a, A = store.vec_since(s, e, ACTIVE, now - LOOKBACK_S)
        ts_a = np.asarray(ts_a, dtype=np.float64)
        act = np.asarray(A, dtype=np.float64).reshape(-1) if len(ts_a) else np.zeros(0)
        on = ts_a[act >= 0.5]
        wts = [float(t) for t in on[-K:]]
        if not wts or wts[-1] != now:
            wts = (wts + [now])[-K:]
        span = max(now - float(ts_a[0]) + dt, dt) if ts_a.size else dt
        wpd = min(max(float(on.size) * 86400.0 / min(span, LOOKBACK_S), 1.0), 86400.0 / dt)
        h = SQ.h_for("llr", ARL_DAYS, 86400.0 / wpd)

        # ---- tick cache (act.tokens / client.stack_set live 1 h)
        toks = _vocab_obs(store.latest_fresh(s, e, "act.tokens", now))
        stacks = MC.stack_counts(store.latest_fresh(s, e, "client.stack_set", now))
        ticks = [t for t in st["ticks"] if t[0] in wts and t[0] != now]
        ticks.append((now, toks, stacks))
        st["ticks"] = ticks
        st["n_act"] = int(st["n_act"]) + 1

        # ---- window vector -> identity space
        rows, sks = [], []
        for t in wts:
            r = store.vec_at(s, e, VEC, t)
            if r is not None:
                rows.append(np.asarray(r, dtype=np.float64))
                sk = store.vec_at(s, e, SKETCH, t)
                if sk is not None:
                    sks.append(np.asarray(sk, dtype=np.float64))
        if not rows:
            emit.write_scores(store, s, e, now, {DETECTOR: _NAN},
                              degraded={DETECTOR: "stale:feature.vec"}, window_s=win)
            return 1
        ck = sc.role_key(e)
        timing = store.latest_derived(s, e, TIMING)
        timing = timing.value if timing is not None and timing.ts >= wts[0] - dt and \
            isinstance(timing.value, Mapping) else None
        center = 0.5 * (wts[0] + wts[-1])
        tc = TB.tctx_from_config(center, ctx.config, dt)
        if MI is not None and hasattr(MI, "window_vector"):
            x = MI.window_vector(store, s, e, list(wts), view.raw)
        else:
            x = window_vector(np.vstack(rows), np.vstack(sks) if sks else None, timing,
                              float(tc["hour_local"]), tc["day_type"] == "workday",
                              view.fill_for(ck))
        z = view.project(np.asarray(x, dtype=np.float64)) if x is not None else None
        if z is None:
            emit.write_scores(store, s, e, now, {DETECTOR: _NAN},
                              degraded={DETECTOR: "layout:model.identity"}, window_s=win)
            return 1

        # ---- Gaussian block: every enrolled entity and class
        k = view.k
        d2 = ((view.M - z) ** 2).sum(axis=1)
        tau0 = view.tau0
        d2_bg = float(((z - view.mu0) ** 2).sum())
        l_bg = -0.5 * d2_bg / (1.0 + tau0) - 0.5 * k * math.log1p(tau0)
        aliases, shared = _continuity(store, s, e)
        aliases |= link_map.get(e, set())
        selfish = {e} | aliases
        order = np.argsort(d2, kind="stable")
        others = [view.ents[i] for i in order if view.ents[i] != e][:N_TOP]
        cands: List[str] = [e] + others + [a for a in sorted(aliases)
                                           if a in view.idx and a not in others]
        gauss: Dict[str, float] = {}
        typ: Dict[str, float] = {}
        for j in cands:
            i = view.idx.get(j)
            if i is None:
                gauss[j], typ[j] = _NAN, _NAN
            else:
                gauss[j] = -0.5 * float(d2[i]) - l_bg
                typ[j] = _chi2_sf(float(d2[i]), k)
        # classes: own role class and the best other role
        cls_ll: Dict[str, float] = {}
        cls_d2: Dict[str, float] = {}
        for ci, c in enumerate(view.classes):
            if not c.startswith("class:") or c.startswith(("class:static:", "class:pool:")):
                continue
            t2 = view.class_tau2(c, sc.members(c))
            dc = float(((z - view.CM[ci]) ** 2).sum()) / (1.0 + t2)
            cls_d2[c] = dc
            cls_ll[c] = -0.5 * dc - 0.5 * k * math.log1p(t2) - l_bg
        own_cls = ck if ck in cls_ll else None
        other_cls = sorted((c for c in cls_ll if c != own_cls), key=lambda c: -cls_ll[c])[:1]
        cls_cands = ([own_cls] if own_cls else []) + other_cls
        for c in cls_cands:
            gauss[c] = cls_ll[c]
        cands = cands + cls_cands

        # ---- modalities
        llr: Dict[str, Dict[str, float]] = {j: {"gauss": gauss[j]} for j in cands}
        self._cheap(sc, e, st, now, dt, wts, ts_a, act, cands, llr)
        self._slow(sc, e, st, now, wts, cands, llr)
        L: Dict[str, float] = {}
        for j in cands:
            L[j] = sum(calibrate(llr[j].get(m, _NAN), view.calib[m]) for m in MODALITIES)

        # ---- posterior with the unknown hypothesis
        n_oth = max(1, len(cands) - 1)
        lp = {j: L[j] + math.log(SELF_PRIOR if j == e else (1.0 - SELF_PRIOR - UNKNOWN_PRIOR)
                                  / n_oth) for j in cands}
        lp_unk = math.log(UNKNOWN_PRIOR)
        mx = max(max(lp.values()), lp_unk)
        z_ = sum(math.exp(v - mx) for v in lp.values()) + math.exp(lp_unk - mx)
        post = {j: math.exp(v - mx) / z_ for j, v in lp.items()}
        p_unknown = math.exp(lp_unk - mx) / z_
        pi_self = sum(post[j] for j in cands if j in selfish)

        # ---- other-identity CUSUM
        rivals = [j for j in cands if j not in selfish and j != own_cls]
        j_star = max(rivals, key=lambda j: (L[j], j)) if rivals else None
        lam = _NAN
        if j_star is not None:
            lam = min(LAMBDA_HI, max(LAMBDA_LO, L[j_star] - L[e]))
            st["S"] = max(0.0, float(st["S"]) + lam - CUSUM_K)
        eligible = [j for j in cands if j != own_cls]
        top = max(eligible, key=lambda j: (L[j], j))
        st["winners"] = (list(st["winners"]) + [e if top in selfish else top])[-K:]

        # ---- unknown CUSUM (typicality of the individual candidates)
        ps = [typ[j] for j in cands if typ.get(j, _NAN) == typ.get(j, _NAN)]
        p_max = max(ps) if ps else _NAN
        if p_max == p_max:
            st["U"] = max(0.0, float(st["U"]) + (U_UP if p_max < P_TYPICAL else -U_DOWN))
        if own_cls is not None:
            class_p = _chi2_sf(cls_d2[own_cls], k)
        else:
            class_p = _chi2_sf(d2_bg / (1.0 + tau0), k)

        # ---- events
        n_ev = 0
        info = {"pi_self": pi_self, "post": post, "L": L, "llr": llr, "j_star": j_star,
                "lam": lam, "h": h, "p_max": p_max, "class_p": class_p, "own_cls": own_cls,
                "p_unknown": p_unknown, "shared": shared, "wts": wts, "wpd": wpd}
        if st["S"] >= h and j_star is not None:
            n_ev += self._mismatch(ctx, sc, e, st, now, dt, info)
            st["S"] = 0.0
        if st["U"] >= U_H:
            n_ev += self._unknown(ctx, sc, e, st, now, dt, info)
            st["U"] = 0.0
        st["hold"] = {kk: t for kk, t in st["hold"].items() if now - t < ALERT_HOLD_S}

        # ---- outputs
        score = -math.log10(max(pi_self, 1e-300))
        emit.write_scores(store, s, e, now, {DETECTOR: score}, pm={DETECTOR: pi_self},
                          axes={DETECTOR: AXES} if pi_self < AXES_POST_MAX else None,
                          window_s=win)
        best = {}
        if j_star is not None:
            best = {"entity": j_star, "posterior": _json(post[j_star]), "L": _json(L[j_star]),
                    "kind": "class" if is_class(j_star) else "entity"}
        store.add_derived(DerivedMetric(
            name=ID_SERIES, ts=now, system=s, entity=e, window_s=win,
            kind=MetricKind.CATEGORICAL, inputs=[VEC, SKETCH, ID_MODEL],
            value={"posterior_self": _json(pi_self), "best_other": best,
                   "p_unknown": _json(p_unknown), "cusum_other": _json(float(st["S"])),
                   "cusum_new": _json(float(st["U"])), "h": _json(h), "lambda": _json(lam),
                   "p_max": _json(p_max), "class_p": _json(class_p),
                   "winners": list(st["winners"]), "n_act": int(st["n_act"]),
                   "n_window": len(wts)}))
        if n_ev or now - float(st["profile_ts"]) >= PROFILE_EVERY_S or now < st["profile_ts"]:
            st["profile_ts"] = now
            self._profile(store, s, e, now, info, st, view)
        return 1

    # ------------------------------------------------------------- modalities
    def _cheap(self, sc: _SysCache, e: str, st: Dict[str, Any], now: float, dt: float,
               wts: List[float], ts_a: np.ndarray, act: np.ndarray, cands: List[str],
               llr: Dict[str, Dict[str, float]]) -> None:
        """vocab, client and rhythm presence: every tick."""
        ticks = st["ticks"]
        obs = _merge_obs([t[1] for t in ticks])
        stacks: Dict[str, float] = {}
        n_cl = 0
        for t in ticks:
            if t[2]:
                n_cl += 1
                for tok, c in t[2].items():
                    stacks[tok] = stacks.get(tok, 0.0) + c
        bgv, bgc = sc.bg_vocab(), sc.bg_client()
        acts, cells = self._slots(sc, now, dt, wts, ts_a, act)
        bgr = sc.bg_rhythm()
        ll_bg_r = rhythm_ll(bgr, acts, cells) if bgr is not None else _NAN
        for j in cands:
            d = llr[j]
            d["vocab"] = vocab_llr(sc.vocab(j), bgv, obs)
            d["client"] = client_llr(sc.client(j), bgc, stacks, n_cl)
            d["rhythm"] = rhythm_ll(sc.rhythm(j), acts, cells) - ll_bg_r \
                if ll_bg_r == ll_bg_r else _NAN

    def _slots(self, sc: _SysCache, now: float, dt: float, wts: List[float],
               ts_a: np.ndarray, act: np.ndarray) -> Tuple[List[float], List[Tuple[int, int]]]:
        """Presence of the local 15-min slots spanned by the window (the most
        recent RHYTHM_MAX_SLOTS): 1 if any tick covering the slot was active,
        0 if all covering ticks were idle; slots no tick covers are left out."""
        t0 = max(wts[0] - dt, now - RHYTHM_MAX_SLOTS * TB.SLOT_S)
        sel = ts_a > t0
        tt, aa = ts_a[sel], act[sel]
        if not tt.size:
            return [], []
        off = TB.local_datetime(now, sc.tz).utcoffset()
        off_s = off.total_seconds() if off is not None else 0.0
        slot_s = TB.SLOT_S
        pres: Dict[int, float] = {}
        prev = float(tt[0]) - dt
        for t, a in zip(tt.tolist(), aa.tolist()):
            lo = max(prev, t - max(dt, 1.0) * 4.0)
            s0 = int(math.floor((lo + off_s) / slot_s + 1e-9))
            s1 = int(math.floor((t + off_s - 1e-6) / slot_s))
            av = 1.0 if a >= 0.5 else 0.0
            for sl in range(max(s0, s1 - 3), s1 + 1):
                pres[sl] = max(pres.get(sl, 0.0), av)
            prev = t
        sl_sorted = sorted(pres)[-RHYTHM_MAX_SLOTS:]
        sc.bg_rhythm()                                 # loads the calendar self-healing days
        cells = [MR.cells_of_slot(sl, sc.calendar, sc.healed) for sl in sl_sorted]
        return [pres[sl] for sl in sl_sorted], cells

    def _slow(self, sc: _SysCache, e: str, st: Dict[str, Any], now: float, wts: List[float],
              cands: List[str], llr: Dict[str, Dict[str, float]]) -> None:
        """PPM and gap terms: refreshed when a window completes (every K
        active ticks) or on the first tick; a candidate that joins between
        completions is scored on the cached window symbols / gaps."""
        slow = st["slow"]
        if slow is None or int(st["n_act"]) % K == 0 or slow.get("syms") is None:
            syms: List[str] = []
            gaps: List[np.ndarray] = []
            if sc.smap is None:
                sc.smap = MS.SymbolMap.from_store(sc.store, sc.s)
            for t in wts:
                _, sy, _, _ = MS.stream_symbols(sc.store, sc.s, e, t, sc.smap)
                syms.extend(sy)
                rows = MT.stream_rows(sc.store, sc.s, e, t)
                if len(rows):
                    gaps.append(MTI.gaps_from_times(rows["ts"], MT.stream_frac(sc.store, sc.s,
                                                                                e, t)))
            g = np.concatenate(gaps) if gaps else np.zeros(0)
            slow = st["slow"] = {"ts": now, "syms": syms[-SEQ_MAX:], "gaps": g[-GAP_MAX:],
                                 "seq": {}, "timing": {}}
        for j in cands:
            if j not in slow["seq"]:
                slow["seq"][j] = seq_llr(sc.seq(j), sc.seq_sys(), slow["syms"], sc.vocab_size()) \
                    if slow["syms"] else _NAN
            if j not in slow["timing"]:
                slow["timing"][j] = timing_llr(sc.timing_logp(j), sc.bg_timing(), slow["gaps"]) \
                    if slow["gaps"].size else _NAN
            llr[j]["seq"] = slow["seq"][j]
            llr[j]["timing"] = slow["timing"][j]

    # ---------------------------------------------------------------- events
    def _mismatch(self, ctx: Context, sc: _SysCache, e: str, st: Dict[str, Any], now: float,
                  dt: float, info: Mapping[str, Any]) -> int:
        store, s, view = ctx.store, sc.s, sc.view
        j = info["j_star"]
        post = float(info["post"][j])
        wins = sum(1 for w in st["winners"] if w == j)
        conf = view.confusion(e, j) if not is_class(j) else 0.0
        concurrent = False
        if not is_class(j):
            a = store.vec_at(s, j, ACTIVE, now)
            concurrent = a is not None and float(a[0]) >= 0.5
        reasons = []
        if wins < WINS_MIN:
            reasons.append(f"wins {wins}/{K}")
        if post < POST_MIN:
            reasons.append(f"posterior {post:.2f}")
        if not conf < CONF_MAX:
            reasons.append(f"confusion {conf:.2f}")
        if info["shared"]:
            reasons.append("shared_ip")
        if reasons:
            sev = Severity.INFO
        else:
            sev = Severity.HIGH if concurrent else Severity.MEDIUM
        hold_key = f"mismatch|{j}|{sev.value}"
        if ctx.training or now - st["hold"].get(hold_key, -math.inf) < ALERT_HOLD_S:
            return 0
        st["hold"][hold_key] = now
        kind = "class" if is_class(j) else "entity"
        what = m_class.role_name(store, j[len("class:"):]) if is_class(j) else j
        desc = (f"window of {e} is attributed to {kind} {what} (posterior {post:.2f}, "
                f"{wins}/{K} windows)" + (", which is active at the same time" if concurrent
                                          else ""))
        store.add_event(BehaviorEvent(
            system=s, entity=e, ts=now, kind="identity_mismatch", score=post, severity=sev,
            description=desc, axes=list(AXES),
            extra={"looks_like": j, "looks_like_kind": kind, "posterior": _json(post),
                   "posterior_self": _json(info["pi_self"]), "p_unknown": _json(info["p_unknown"]),
                   "cusum": _json(float(st["S"])), "h": _json(info["h"]), "wins": wins,
                   "confusion": _json(conf), "concurrent": bool(concurrent),
                   "downgraded": reasons, "shared_ip": bool(info["shared"]),
                   "llr": {m: _json(v) for m, v in info["llr"][j].items()},
                   "llr_self": {m: _json(v) for m, v in info["llr"][e].items()},
                   "hold_key": hold_key},
            dedupe_key=f"identity_mismatch|{s}|{e}|{j}",
            model_version=int(view.version), window=(float(info["wts"][0]) - dt, now)))
        return 1

    def _unknown(self, ctx: Context, sc: _SysCache, e: str, st: Dict[str, Any], now: float,
                 dt: float, info: Mapping[str, Any]) -> int:
        store, s, view = ctx.store, sc.s, sc.view
        young = e not in view.idx
        cp = info["class_p"]
        if young:
            sev, why = Severity.INFO, "young"
        elif cp == cp and cp < P_TYPICAL:
            sev, why = Severity.HIGH, "unlike_anyone"
        else:
            sev, why = Severity.MEDIUM, "same_class_different_individual"
        if info["shared"]:
            sev = Severity.INFO
        hold_key = f"unknown|{sev.value}"
        if ctx.training or now - st["hold"].get(hold_key, -math.inf) < ALERT_HOLD_S:
            return 0
        st["hold"][hold_key] = now
        p_max = info["p_max"]
        desc = {"young": f"{e} is not yet enrolled and resembles no known identity",
                "unlike_anyone": f"window of {e} is unlike any known individual or class",
                "same_class_different_individual":
                    f"window of {e} fits its class but no known individual"}[why]
        store.add_event(BehaviorEvent(
            system=s, entity=e, ts=now, kind="unknown_identity",
            score=1.0 - (p_max if p_max == p_max else 0.0), severity=sev, description=desc,
            axes=list(AXES),
            extra={"reason": why, "p_max": _json(p_max), "class_p": _json(cp),
                   "class_key": info["own_cls"], "p_unknown": _json(info["p_unknown"]),
                   "posterior_self": _json(info["pi_self"]), "cusum": _json(float(st["U"])),
                   "young": bool(young), "shared_ip": bool(info["shared"]),
                   "hold_key": hold_key},
            dedupe_key=f"unknown_identity|{s}|{e}",
            model_version=int(view.version), window=(float(info["wts"][0]) - dt, now)))
        return 1

    # ---------------------------------------------------------------- profile
    def _profile(self, store: Any, s: str, e: str, now: float, info: Mapping[str, Any],
                 st: Mapping[str, Any], view: IdModel) -> None:
        post, L, llr = info["post"], info["L"], info["llr"]
        top = sorted(post, key=lambda j: -post[j])[:4]
        p = store.profile(s, e) or EntityProfile(system=s, entity=e)
        p.extra["attribution"] = {
            "posterior_self": _json(info["pi_self"]), "p_unknown": _json(info["p_unknown"]),
            "looks_like": info["j_star"],
            "looks_like_posterior": _json(post.get(info["j_star"], _NAN))
            if info["j_star"] else None,
            "candidates": [{"id": j, "kind": "class" if is_class(j) else "entity",
                            "posterior": _json(post[j]), "L": _json(L[j]),
                            "llr": {m: _json(v) for m, v in llr[j].items()}} for j in top],
            "cusum_other": _json(float(st["S"])), "cusum_new": _json(float(st["U"])),
            "h": _json(info["h"]), "windows_per_day": _json(info["wpd"]),
            "p_max": _json(info["p_max"]), "class_p": _json(info["class_p"]),
            "enrolled": e in view.idx, "shared_ip": bool(info["shared"]),
            "model_version": int(view.version), "updated": now,
        }
        store.put_profile(p)
