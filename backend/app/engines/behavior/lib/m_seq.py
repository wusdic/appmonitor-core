"""Read accessors for model.seq (owner: B10 SequenceEngine; contract C) and the
shared tokeniser that turns act.stream into sequence symbols.

Why a shared tokeniser: B16 attribution (and B15 / B17) score a window of
tokens under *other* entities' PPM models, so the symbols must be produced
exactly as B10 produced them when it trained those models. Every consumer
therefore goes through `SymbolMap` / `stream_symbols` here instead of
re-deriving tokens from act.stream, and reads model internals only through
the functions below (consumers never mutate a model).

Symbols (stable strings, never ids of a display name):
    'token|outcome'  m_template.seq_token(token, outcome): HTTP tokens already
                     end in their status class, other channels get '|2xx' etc.
    '{rare:<ch>}'    a token whose system count (model.template Space-Saving
                     count, this tick included) is below RARE_MIN_COUNT = 3, or
                     an evicted id; ch in http | tls | dns | l4 | rare
    'cat:<a>+<b>'    stream B: the sorted bag of lib-4 categories of one tick
    '{bos}'          session start: only ever used as history (never predicted),
                     so the first token of a session is predicted from its own
                     context instead of from the previous session's tail
Without model.template (unit pipelines) a token id reads 'id:<tid>|<outcome>'.

model.seq layouts (stored by reference with put_model; objects are live):
  entity  @(s, ip):
    {'kind': 'entity', 'ppm': PPMModel (stream A, order 3, 30 d half-life),
     'ppm_cat': PPMModel (stream B, order 2), 'session_gap': float s,
     'entropy_rate': [mu, sigma] bits/token (NaN until recorded),
     'gap': decayed 32-bin log10-gap histogram {'h','t_ref','g','v'[32]},
     'dwell': decayed run-length moments {'h','t_ref','g','fam': {family: [W, S1, S2]}}
              of x = L - 1 per token family,
     'class_key': 'class:<rid>' | None, 'version': int, 'ts': float,
     '_state' / '_gate' / '_run' / '_rows' / '_held' / '_ntok': B10 private}
  class   @(s, 'class:<rid>'): the same public fields, 'kind': 'class', rebuilt
          by merging the member entity models (order 3 / 2), 'members': n
  system  @(s, '__system__'): 'kind': 'system', order-0 unigrams (the
          Good-Turing tier of the backoff chain), 'members': n
  Decayed accumulators store values at scale 2^g (g = (clock - t_ref) / h);
  read them unscaled through gap_hist() / dwell_moments().

Accessor signatures (pure reads; missing data gives the documented default):
    SymbolMap(templater) / SymbolMap.from_store(store, s)
        .symbol(tid, outcome) -> str; .info(tid, outcome) -> (symbol, family, auth)
        .vocab_size -> int (template vocabulary + the rare symbols)
    stream_symbols(store, s, e, ts, smap=None) -> (ts float64[k], syms[k], fams[k], auth[k])
    session_starts(ts, gap, last_ts=nan) -> (cut indices, first_is_new)
    get(store, s, e) -> dict | None;  ppm(model) / ppm_cat(model) -> PPMModel | None
    session_gap(model, default=1800.0) -> float
    entropy_rate(model) -> (mu, sigma);  maturity(model) -> 1 - Good-Turing unseen mass
    backoff(store, s, e) -> [class PPM?, system PPM?] (non-empty tiers only)
    vocab_size(store, s) -> int
    loglik(model, tokens, backoff=(), vocab_size=0, history=None) -> bits[len]
        history None = session start ('{bos}'); a PPMModel is accepted for model
    session_excess(bits, mu, k=5) -> NLL* - mu with NLL* = (L NLL + k mu)/(L + k)
    gap_hist(model) -> float64[32];  gap_bins(gaps) -> int[];  gap_valley(hist) -> s
    dwell_moments(model, family) -> (W, S1, S2);  dwell_sf(model, family, L, tier=None)
    top_ngrams(model, n=2, k=10) -> [{'ngram': [...], 'count': c}]
    describe(model, k=8) -> dict (portrait descriptor, B30)
    is_auth(family) -> bool
"""
from __future__ import annotations

import heapq
import math
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import m_class
from . import m_template
from . import ppm as _ppm
from .bayes import nb_sf
from .classkeys import SYSTEM_KEY
from .template import RARE_TOKEN, Templater, channel_of, token_family

MODEL = "model.seq"
BOS = "{bos}"
CAT_PREFIX = "cat:"
CAT_MIN_CONF = 0.6
RARE_MIN_COUNT = 3.0
RARE_CHANNELS = ("http", "tls", "dns", "l4", "rare")
N_RARE_SYMBOLS = len(RARE_CHANNELS)

DEFAULT_SESSION_GAP_S = 1800.0
SESSION_GAP_MIN_S = 120.0
SESSION_GAP_MAX_S = 7200.0
GAP_BINS = 32
GAP_LOG10_LO = -1.0             # 0.1 s
GAP_LOG10_HI = 5.5              # ~3.7 d; larger gaps land in the top bin
GAP_BIN_W = (GAP_LOG10_HI - GAP_LOG10_LO) / GAP_BINS
GAP_MIN_WEIGHT = 30.0           # decayed gaps before the valley replaces the default

SHRINK_K = 5.0                  # session NLL* pseudo tokens at mu_e
DWELL_PRIOR_W = 1.0             # one pseudo run of a geometric with mean x = 1
DWELL_PRIOR_MEAN = 1.0
DWELL_TIER_W = 5.0              # at most this much class / system weight per family
_NB_R_MAX = 1e12

# token families that make a sequence axis a credential axis (contract K)
_AUTH_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:log[io]n|logon|logout|sign[-_]?in|signon|auth\w*|oauth\w*|sso|saml\w*|"
    r"cas|token|session|passw\w*|passwd|pwd|mfa|2fa|otp|kerberos|krb5?|ldaps?|adfs|"
    r"radius|ssh|rdp)(?:$|[^a-z0-9])")


def is_auth(family: str) -> bool:
    """True when a token family names an authentication resource or service."""
    return bool(family) and _AUTH_RE.search(family.lower()) is not None


# ================================================================ tokeniser
class SymbolMap:
    """Per-system, per-tick (token_id, outcome) -> symbol map. Build a fresh one
    every tick: the rare floor reads the system count as of this tick."""

    __slots__ = ("tpl", "vocab_size", "_cache")

    def __init__(self, templater: Optional[Templater]) -> None:
        self.tpl = templater
        self.vocab_size = (len(templater.vocab) if templater is not None else 0) + N_RARE_SYMBOLS
        self._cache: Dict[Tuple[int, int], Tuple[str, str, bool]] = {}

    @classmethod
    def from_store(cls, store, system: str) -> "SymbolMap":
        return cls(m_template.templater(store, system))

    def info(self, tid: int, outcome: int) -> Tuple[str, str, bool]:
        key = (tid, outcome)
        hit = self._cache.get(key)
        if hit is None:
            hit = self._cache[key] = self._make(tid, outcome)
        return hit

    def symbol(self, tid: int, outcome: int) -> str:
        return self.info(tid, outcome)[0]

    def _make(self, tid: int, outcome: int) -> Tuple[str, str, bool]:
        tpl = self.tpl
        if tpl is None:
            sym = f"id:{tid}|{m_template.outcome_class(outcome)}"
            fam = "rare|||"
        else:
            tok = tpl.token_of(tid)
            e = tpl.vocab.get(tok) if tok != RARE_TOKEN else None
            if e is None or not float(e[1]) >= RARE_MIN_COUNT:
                sym = "{rare:%s}" % channel_of(tok)
            else:
                sym = m_template.seq_token(tok, outcome)
            fam = token_family(sym)
        sym = sys.intern(sym)
        return sym, sys.intern(fam), is_auth(fam)


def stream_symbols(store, system: str, entity: str, ts: float,
                   smap: Optional[SymbolMap] = None
                   ) -> Tuple[np.ndarray, List[str], List[str], List[bool]]:
    """act.stream of tick ts as (event ts ascending, symbols, families, auth
    flags). Rows with a non-finite ts are dropped; empty when absent."""
    rows = _read_stream(store, system, entity, ts)
    if rows is None:
        return np.zeros(0), [], [], []
    t, tid, oc = rows
    smap = smap if smap is not None else SymbolMap.from_store(store, system)
    info = smap.info
    syms: List[str] = []
    fams: List[str] = []
    auth: List[bool] = []
    for a, b in zip(tid.tolist(), oc.tolist()):
        s_, f_, u_ = info(a, b)
        syms.append(s_)
        fams.append(f_)
        auth.append(u_)
    return t, syms, fams, auth


def _read_stream(store, system: str, entity: str, ts: float
                 ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(ts, token_id, outcome) of the act.stream rows written at tick ts, sorted
    by ts, non-finite ts dropped; None when there are none. The single place
    that knows the row layout (m_template.STREAM_DTYPE)."""
    rows = m_template.stream_rows(store, system, entity, ts)
    if not len(rows):
        return None
    t = np.asarray(rows["ts"], dtype=np.float64)
    tid = np.asarray(rows["token_id"], dtype=np.int64)
    oc = np.asarray(rows["outcome"], dtype=np.int64)
    ok = np.isfinite(t)
    if not ok.all():
        t, tid, oc = t[ok], tid[ok], oc[ok]
        if not t.size:
            return None
    if t.size > 1 and np.any(np.diff(t) < 0.0):
        o = np.argsort(t, kind="stable")
        t, tid, oc = t[o], tid[o], oc[o]
    return t, tid, oc


def session_starts(ts: np.ndarray, gap: float, last_ts: float = math.nan
                   ) -> Tuple[np.ndarray, bool]:
    """Indices i > 0 where ts[i] - ts[i-1] > gap (a new session starts), and
    whether ts[0] starts a new session (no previous event, or a gap > gap)."""
    ts = np.asarray(ts, dtype=np.float64)
    if not ts.size:
        return np.zeros(0, dtype=np.int64), False
    cut = np.flatnonzero(np.diff(ts) > gap) + 1
    first = not math.isfinite(last_ts) or float(ts[0]) - last_ts > gap
    return cut, bool(first)


# ================================================================ model reads
def get(store, system: str, entity: str) -> Optional[Dict[str, Any]]:
    m = store.get_model(system, entity, MODEL, default=None)
    return m if isinstance(m, dict) else None


def _as_ppm(x: Any) -> Optional[_ppm.PPMModel]:
    if x is None:
        return None
    if isinstance(x, _ppm.PPMModel):
        return x
    return _ppm.PPMModel.from_dict(x) if isinstance(x, dict) else None


def ppm(model: Any) -> Optional[_ppm.PPMModel]:
    """Stream-A PPM of a model.seq dict (a PPMModel passes through)."""
    if isinstance(model, _ppm.PPMModel):
        return model
    return _as_ppm(model.get("ppm")) if isinstance(model, dict) else None


def ppm_cat(model: Any) -> Optional[_ppm.PPMModel]:
    return _as_ppm(model.get("ppm_cat")) if isinstance(model, dict) else None


def session_gap(model: Any, default: float = DEFAULT_SESSION_GAP_S) -> float:
    if isinstance(model, dict):
        g = model.get("session_gap")
        try:
            g = float(g)
        except (TypeError, ValueError):
            return default
        if math.isfinite(g) and g > 0.0:
            return g
    return default


def entropy_rate(model: Any) -> Tuple[float, float]:
    """(mu, sigma) bits/token of committed (predictive) surprisal; NaN when unknown."""
    if isinstance(model, dict) and isinstance(model.get("entropy_rate"), (list, tuple)):
        er = model["entropy_rate"]
        if len(er) == 2:
            return float(er[0]), float(er[1])
    p = ppm(model)
    return _ppm.entropy_rate(p) if p is not None else (math.nan, math.nan)


def maturity(model: Any) -> float:
    """1 - Good-Turing unseen mass of the stream-A unigram (0 for an empty model)."""
    p = ppm(model)
    return 0.0 if p is None else 1.0 - _ppm.good_turing_unseen(p)


def backoff(store, system: str, entity: str) -> List[_ppm.PPMModel]:
    """The backoff tiers B10 scores `entity` with: its role class (keyed by id,
    >= 3 members in the system) then the system unigram; empty tiers skipped."""
    out: List[_ppm.PPMModel] = []
    ck = m_class.class_key(store, system, entity)
    for key in ((ck,) if ck else ()) + (SYSTEM_KEY,):
        p = ppm(get(store, system, key))
        if p is not None and p.counts:
            out.append(p)
    return out


def vocab_size(store, system: str) -> int:
    """Real vocabulary size for the Good-Turing unseen share: the system's
    template vocabulary plus the rare symbols (at least the system unigram size)."""
    v = m_template.vocab_size(store, system) + N_RARE_SYMBOLS
    p = ppm(get(store, system, SYSTEM_KEY))
    if p is not None:
        v = max(v, len(p.counts.get((), ())) + 1)
    return v


def loglik(model: Any, tokens: Sequence[str], backoff: Sequence[Any] = (),
           vocab_size: int = 0, history: Optional[Sequence[str]] = None) -> np.ndarray:
    """Per-token surprisal (bits) of `tokens` under model -> backoff tiers
    (ppm.loglik). history None means the tokens start a session. An empty or
    missing model scores through its backoff tiers alone."""
    p = ppm(model)
    tiers = [t for t in (ppm(b) for b in backoff) if t is not None]
    hist = (BOS,) if history is None else tuple(history)
    if p is None:
        if not tiers:
            return np.full(len(tokens), math.log2(max(1, int(vocab_size))))
        p, tiers = tiers[0], tiers[1:]
    return _ppm.loglik(p, tokens, tiers, int(vocab_size), hist)


def session_excess(bits: Sequence[float], mu: float, k: float = SHRINK_K) -> float:
    """NLL* - mu with NLL* = (L NLL + k mu)/(L + k): a short session's mean
    surprisal shrunk toward the entity entropy rate. NaN when undefined."""
    b = np.asarray(bits, dtype=np.float64)
    b = b[np.isfinite(b)]
    if not b.size or not math.isfinite(mu):
        return math.nan
    return float((b.sum() - b.size * mu) / (b.size + k))


# ------------------------------------------------------------ gap histogram
def gap_bins(gaps: Sequence[float]) -> np.ndarray:
    """log10-gap bin of each gap (clipped to [0, GAP_BINS - 1]; <= 1 ms is bin 0)."""
    g = np.maximum(np.asarray(gaps, dtype=np.float64), 1e-3)
    j = np.floor((np.log10(g) - GAP_LOG10_LO) / GAP_BIN_W)
    return np.clip(np.nan_to_num(j, nan=0.0), 0, GAP_BINS - 1).astype(np.int64)


def gap_bin_center_s(j: float) -> float:
    return 10.0 ** (GAP_LOG10_LO + (j + 0.5) * GAP_BIN_W)


def _unscaled(acc: Dict[str, Any]) -> float:
    g = float(acc.get("g", 0.0) or 0.0)
    return 2.0 ** (-g) if math.isfinite(g) else 0.0


def gap_hist(model: Any) -> np.ndarray:
    """Decayed gap histogram (unscaled, at the model clock), float64[GAP_BINS]."""
    acc = model.get("gap") if isinstance(model, dict) else None
    if not isinstance(acc, dict) or acc.get("v") is None:
        return np.zeros(GAP_BINS)
    return np.asarray(acc["v"], dtype=np.float64) * _unscaled(acc)


def gap_valley(hist: Sequence[float], default: float = DEFAULT_SESSION_GAP_S) -> float:
    """Session idle threshold: the density valley between the within-session
    and the between-session modes of the log-gap histogram, clipped to
    [SESSION_GAP_MIN_S, SESSION_GAP_MAX_S].

    The two modes come from Otsu's split of the log-gap mass (the plateau
    middle when an empty stretch makes several splits equivalent); the valley
    is the minimum of the [1, 2, 1]-smoothed density between the two class
    means, ties broken toward the split. `default` below GAP_MIN_WEIGHT gaps or
    when all mass sits in one bin."""
    h = [max(0.0, float(x)) if math.isfinite(float(x)) else 0.0 for x in hist]
    n = len(h)
    tot = sum(h)
    if n < 3 or not tot >= GAP_MIN_WEIGHT:
        return float(default)
    m_tot = sum(j * x for j, x in enumerate(h))
    best, ks = -1.0, []
    w0 = m0 = 0.0
    for k in range(n - 1):
        w0 += h[k]
        m0 += k * h[k]
        w1 = tot - w0
        if w0 <= 0.0 or w1 <= 0.0:
            continue
        d = m0 / w0 - (m_tot - m0) / w1
        v = w0 * w1 * d * d
        if v > best * (1.0 + 1e-12):
            best, ks = v, [k]
        elif v >= best * (1.0 - 1e-12):
            ks.append(k)
    if not ks:
        return float(default)
    k = ks[len(ks) // 2]
    w0 = sum(h[:k + 1])
    mu0 = sum(j * h[j] for j in range(k + 1)) / w0
    mu1 = sum(j * h[j] for j in range(k + 1, n)) / (tot - w0)
    lo, hi = int(math.floor(mu0)), int(math.ceil(mu1))
    split = k + 0.5
    best_j, best_d = k, math.inf
    for j in range(max(0, lo), min(n - 1, hi) + 1):
        d = 0.25 * (h[j - 1] if j > 0 else 0.0) + 0.5 * h[j] + 0.25 * (h[j + 1] if j + 1 < n else 0.0)
        if d < best_d - 1e-12 or (d <= best_d + 1e-12 and abs(j - split) < abs(best_j - split)):
            best_j, best_d = j, d
    g = gap_bin_center_s(best_j)
    return float(min(max(g, SESSION_GAP_MIN_S), SESSION_GAP_MAX_S))


# ------------------------------------------------------------------- dwell
def dwell_moments(model: Any, family: str) -> Tuple[float, float, float]:
    """Decayed (W, S1, S2) of x = run length - 1 for a token family (zeros if unseen)."""
    acc = model.get("dwell") if isinstance(model, dict) else None
    if not isinstance(acc, dict):
        return 0.0, 0.0, 0.0
    v = (acc.get("fam") or {}).get(family)
    if not v:
        return 0.0, 0.0, 0.0
    f = _unscaled(acc)
    return float(v[0]) * f, float(v[1]) * f, float(v[2]) * f


def dwell_params(model: Any, family: str, tier: Any = None) -> Tuple[float, float]:
    """NB (mean, size r) of x = L - 1 for `family`: the entity's decayed moments
    plus up to DWELL_TIER_W runs of the tier's (class / system) moments plus a
    geometric pseudo run (mean DWELL_PRIOR_MEAN), matched by moments."""
    W, S1, S2 = dwell_moments(model, family)
    if tier is not None and W < DWELL_TIER_W:
        tw, t1, t2 = dwell_moments(tier, family)
        if tw > 0.0:
            f = min(tw, DWELL_TIER_W) / tw
            W, S1, S2 = W + f * tw, S1 + f * t1, S2 + f * t2
    m0 = DWELL_PRIOR_MEAN
    W += DWELL_PRIOR_W
    S1 += DWELL_PRIOR_W * m0
    S2 += DWELL_PRIOR_W * (m0 + 2.0 * m0 * m0)     # E x^2 of a geometric with mean m0
    m = S1 / W
    v = S2 / W - m * m
    r = m * m / (v - m) if v > m * (1.0 + 1e-9) else _NB_R_MAX
    return m, min(max(r, 1e-6), _NB_R_MAX)


def dwell_sf(model: Any, family: str, length: int, tier: Any = None) -> float:
    """P(L >= length) of a run of `family` under the NB dwell model (1 for L <= 1)."""
    if length <= 1:
        return 1.0
    m, r = dwell_params(model, family, tier)
    p = nb_sf(float(length - 2), m, r)             # P(X > L - 2) = P(L >= length)
    return float(p) if p == p else math.nan


# ---------------------------------------------------------------- portrait
def top_ngrams(model: Any, n: int = 2, k: int = 10) -> List[Dict[str, Any]]:
    """The k heaviest n-grams (1 <= n <= order + 1) of the stream-A model by
    decayed count; session-start ('{bos}') contexts are skipped."""
    p = ppm(model)
    if p is None or not p.counts or n < 1 or n > p.order + 1:
        return []
    f = 2.0 ** (-p.g)
    cand = []
    for ctx, d in p.counts.items():
        if len(ctx) != n - 1 or BOS in ctx:
            continue
        for sym, c in d.items():
            cand.append((c * f, ctx + (sym,)))
    top = heapq.nlargest(k, cand, key=lambda x: x[0])
    return [{"ngram": list(g), "count": round(float(c), 3)} for c, g in top]


def describe(model: Any, k: int = 8) -> Dict[str, Any]:
    """Compact descriptor for portraits (B30): grammar size, entropy rate,
    session gap, top uni/bigrams and the families with the longest mean runs."""
    p = ppm(model)
    mu, sd = entropy_rate(model)
    fams = []
    acc = model.get("dwell") if isinstance(model, dict) else None
    if isinstance(acc, dict):
        f = _unscaled(acc)
        for fam, v in (acc.get("fam") or {}).items():
            W = float(v[0]) * f
            if W >= 1.0:
                fams.append((1.0 + float(v[1]) * f / W, fam, W))
    fams.sort(key=lambda x: (-x[0], x[1]))
    return {
        "session_gap_s": session_gap(model),
        "entropy_rate_bits": None if not math.isfinite(mu) else round(mu, 4),
        "entropy_sd_bits": None if not math.isfinite(sd) else round(sd, 4),
        "n_tokens": 0.0 if p is None else round(float(p.n_tokens), 3),
        "vocab": 0 if p is None else len(p.counts.get((), ())),
        "maturity": round(maturity(model), 4),
        "top_unigrams": top_ngrams(model, 1, k),
        "top_bigrams": top_ngrams(model, 2, k),
        "long_dwell": [{"family": fam, "mean_run": round(mr, 3), "runs": round(W, 2)}
                       for mr, fam, W in fams[:k]],
    }
