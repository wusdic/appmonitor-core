"""PPM-C sequence model with decay and hierarchical backoff (B10, B15, B16, B20).

STATUS: implemented. Signatures and maths are frozen (docs/lib3/helpers_api.md).
DEVIATION (additive): the dataclass gained `n_obs` (undecayed observation
count per order-0 symbol, needed for the Good-Turing singleton mass under
weighted / decayed counts) and two private caches `_tot` / `_n1`; every
field keeps a default, so positional and keyword construction are unchanged.

Why PPM: with ~1e3-1e4 sessions per entity per week and small templated
alphabets, Prediction by Partial Matching is a near-optimal sequence
predictor (Begleiter, El-Yaniv & Yona 2004), trains incrementally in
microseconds, gives exact log-likelihoods for attribution (B16 scores
candidates under *their* models), and its counts are sufficient statistics
that checkpoint / merge / roll back trivially.

Model (per key: entity, class:<rid> or __system__)
    counts[ctx][sym] for contexts of length 0..order (order <= 3), decayed
    with half-life `half_life_s` (30 d). Decay is lazy: counts are stored
    scaled by 2^(g) with g = (ts - t_ref) / half_life_s and renormalised when
    g > 32, so an update is O(order).
PPM-C probability at one tier, with exclusion inside the tier:
    for k = min(order, len(history)) .. 0:
        ctx = last k symbols; n = total count in ctx, q = #distinct symbols seen
        (count >= 1e-9) excluding already-excluded symbols
        if sym seen in ctx: P *= c(sym) / (n + q); stop
        else if n > 0:     P *= q / (n + q) (escape); exclude ctx's symbols
    falling off order 0 escapes to the next tier in `backoff`.
Last tier (the system unigram model or the last model in the chain):
    Good-Turing novel mass g = N1 / N (N1 = weighted count of symbols seen once,
    floored at 0.5 / (N + 1)); P(seen v) = (1 - g) c_v / N;
    P(unseen v) = g / max(1, vocab_size - V_seen) with vocab_size the REAL
    vocabulary size (model.template size), so unseen source states always
    get finite surprisal > 0.
Surprisal is -log2 P per token (bits).

Implementation notes (the choices the formulas above leave open):
  * Clock. The model's clock is its newest update ts: g = (t_clock - t_ref)/H.
    An update older than the clock is added with weight w * 2^(-(t_clock -
    ts)/H) (the row is decayed, never the state), so folding rows in any
    order gives the same counts (what GatedLearner's release / replay needs).
    loglik reads counts at the clock (it has no ts of its own). An empty
    model re-anchors t_ref at its first update. Renormalisation (g > 32 =
    32 half-lives, ~2.6 y) rescales to t_ref = t_clock and drops entries whose
    unscaled count fell below 1e-9; until then such dead entries still count
    in q (their probability mass is ~0 either way).
  * Exclusion. Update-all keeps the contexts nested (the symbols of a
    context are a subset of those of its suffix), so the excluded set at
    order k-1 is exactly the symbol set of the last non-empty context
    visited; n and q both drop the excluded symbols, which keeps every tier
    exactly normalised. A context whose symbols are all excluded escapes
    with probability 1. Exclusion does not cross tiers (as specified), so a
    multi-tier chain is sub-normalised by esc * P_next(symbols the upper
    tier has seen): novel-to-entity symbols are charged slightly more, never
    less.
  * Good-Turing at the last tier, order 0. "Seen once" is by observation
    count (n_obs == 1), and N1 is the decayed weighted count of those
    symbols, so g = N1/N is invariant to a uniform weight (stream_frac x4 or
    trust 0.5 give the same novel mass). g is clipped to [0.5/(N+1),
    1 - 0.5/(N+1)] so neither seen nor unseen symbols get P = 0. Excluded
    symbols leave the seen part: P(seen v) = (1 - g) c_v / (N - c_excluded),
    and when every seen symbol is excluded the whole mass is novel. N is the
    unscaled total (n_tokens). A chain whose tiers are all empty scores
    log2(max(1, vocab_size)) bits (the uniform code), and empty tiers are
    skipped, so the last NON-EMPTY tier carries the Good-Turing step.
  * Entropy-rate stats [W, sum s, sum s^2] are unscaled and decay with the
    same half-life as the counts whenever the clock advances: early
    immature-model surprisal must not pin mu_e high for months.
  * Weights w <= 0 or non-finite are no-ops (rollback is by checkpoint +
    replay, not by negative counts), and so is a row whose decayed weight
    at the clock is below 1e-9 (it would only create "seen" entries with ~0
    mass). NaN ts means "at the clock"; on an empty model there is no clock
    yet, so t_ref becomes NaN (unanchored, serialised as None) and the first
    finite ts anchors it, instead of epoch 0 renormalising those rows away.
    merge() applies the same 1e-9 threshold to entries it would create. A
    no-decay model (half_life_s = inf) keeps t_ref at its newest ts so merge
    can place it; from_dict renormalises a stored g > PPM_RENORM_G.
  * Cost (CPython 3.11, order 3, templated tokens): update ~1.5 us/token,
    loglik ~1 us/token on the predicted path, a few us on an escape through
    three tiers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

PPM_MAX_ORDER = 3
PPM_HALF_LIFE_S = 30 * 86400.0
PPM_RENORM_G = 32.0

Token = Hashable
Context = Tuple[Token, ...]

_DEAD = 1e-9            # unscaled count below which an entry is dropped at renorm
_FMT = 1                # to_dict format version


@dataclass(slots=True)
class PPMModel:
    """Decayed PPM-C counts. `stats` holds the running (mu, sigma) of committed
    per-token surprisal (the entity's entropy rate, bits/token) as
    [W, sum s, sum s^2].

    `counts` values are stored at scale 2^g (see the module doc); read them
    through vocab() / loglik(). `n_obs[v]` is the undecayed number of
    observations of order-0 symbol v (Good-Turing singletons). `_tot` (stored
    total per context) and `_n1` (stored singleton mass) are caches rebuilt
    automatically if `counts` is replaced wholesale; mutate counts only
    through update() / merge()."""
    order: int = PPM_MAX_ORDER
    half_life_s: float = PPM_HALF_LIFE_S
    counts: Dict[Context, Dict[Token, float]] = field(default_factory=dict)
    t_ref: float = 0.0          # lazy-decay reference ts
    g: float = 0.0              # current log2 scale of stored counts
    n_tokens: float = 0.0       # decayed total of order-0 counts (unscaled)
    stats: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    n_obs: Dict[Token, int] = field(default_factory=dict)
    _tot: Dict[Context, float] = field(default_factory=dict, repr=False, compare=False)
    _n1: float = field(default=0.0, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe deep copy (json.dumps(allow_nan=False) works for str / int /
        float / bool / tuple tokens; numpy scalars are converted). Counts are
        written at their stored scale together with (t_ref, g), so a round trip
        is exact."""
        _ensure_index(self)
        return {
            "fmt": _FMT,
            "order": int(self.order),
            "half_life_s": _enc_f(self.half_life_s),
            "t_ref": _enc_f(self.t_ref),
            "g": _enc_f(self.g),
            "n_tokens": _enc_f(self.n_tokens),
            "stats": [_enc_f(x) for x in self.stats],
            "counts": [[[_enc_tok(t) for t in ctx],
                        [[_enc_tok(s), float(c)] for s, c in d.items()]]
                       for ctx, d in self.counts.items()],
            "n_obs": [[_enc_tok(s), int(n)] for s, n in self.n_obs.items()],
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "PPMModel":
        """Inverse of to_dict. None / {} -> a fresh model; a PPMModel passes
        through unchanged."""
        if isinstance(d, PPMModel):
            return d
        if not d:
            return cls()
        hl = d.get("half_life_s", PPM_HALF_LIFE_S)
        m = cls(order=max(0, int(d.get("order", PPM_MAX_ORDER))),
                half_life_s=math.inf if hl is None else float(hl),
                # None = an unanchored clock (NaN, see _advance), not epoch 0
                t_ref=math.nan if ("t_ref" in d and d["t_ref"] is None)
                else _dec_f(d.get("t_ref"), 0.0),
                g=_dec_f(d.get("g"), 0.0),
                n_tokens=_dec_f(d.get("n_tokens"), 0.0))
        st = list(d.get("stats") or [])
        if len(st) == 3:
            m.stats = [_dec_f(x, 0.0) for x in st]
        counts = m.counts
        for ctx, items in d.get("counts") or ():
            key = tuple(_dec_tok(t) for t in ctx)
            dd = {_dec_tok(s): float(c) for s, c in items if float(c) > 0.0}
            if dd:
                counts[key] = dd
        m.n_obs = {_dec_tok(s): int(n) for s, n in d.get("n_obs") or ()}
        _reindex(m)
        if m.g > PPM_RENORM_G:             # foreign / corrupt scale: 2^g could overflow
            _renorm(m)
        return m


# ================================================================ learning
def update(model: PPMModel, tokens: Sequence[Token], w: float = 1.0, ts: float = 0.0,
           history: Sequence[Token] = ()) -> PPMModel:
    """Add one session / window of tokens with weight w (e.g. trust / stream_frac)
    at time ts. `history` is the preceding context (<= order tokens) when a
    window continues a session. Updates every context length 0..order for each
    token (update-all, not update-exclusion). In place; returns model. O(len * order)."""
    toks = tuple(tokens)
    w = float(w)
    if not toks or not (w > 0.0 and math.isfinite(w)):
        return model
    _ensure_index(model)
    g_ts = _advance(model, ts)
    ws = w * 2.0 ** g_ts                  # stored-scale weight of this row
    if not ws * 2.0 ** (-model.g) >= _DEAD:   # below "seen" (ancient row / ~0 trust)
        return model
    order = max(0, int(model.order))
    hist = tuple(history)[-order:] if order else ()
    seq = hist + toks
    counts, tot, obs = model.counts, model._tot, model.n_obs
    d0 = counts.get(())
    if d0 is None:
        d0 = counts[()] = {}
    tot0 = tot.get((), 0.0)
    n1 = model._n1
    for j in range(len(hist), len(seq)):
        sym = seq[j]
        # order 0 carries the Good-Turing bookkeeping
        prev = d0.get(sym)
        if prev is None:
            d0[sym] = ws
            obs[sym] = 1
            n1 += ws
        else:
            o = obs.get(sym, 0)
            if o == 1:
                n1 -= prev                # leaves the singleton set
            d0[sym] = prev + ws
            obs[sym] = o + 1
        tot0 += ws
        for k in range(1, min(order, j) + 1):
            ctx = seq[j - k:j]
            d = counts.get(ctx)
            if d is None:
                counts[ctx] = {sym: ws}
                tot[ctx] = ws
            else:
                d[sym] = d.get(sym, 0.0) + ws
                tot[ctx] += ws
    tot[()] = tot0
    model._n1 = n1 if n1 > 0.0 else 0.0
    model.n_tokens = tot0 * 2.0 ** (-model.g)
    return model


def merge(own: PPMModel, other: PPMModel, w: float = 0.5) -> PPMModel:
    """own + w * other (counts aligned to own's decay reference). Link seeding.

    In place on `own` (returned). `other` is read at its own clock t_o and
    folded in like an update at ts = t_o: rows newer than own's clock advance
    it, older ones arrive decayed. Contexts longer than own.order are dropped.
    Observation counts and entropy-rate stats are merged with the same
    weight (stats decayed to own's clock)."""
    w = float(w)
    if other is None or other is own or not (w > 0.0 and math.isfinite(w)) or not other.counts:
        return own
    _ensure_index(other)
    _ensure_index(own)
    H_o = other.half_life_s if (other.half_life_s > 0 and math.isfinite(other.half_life_s)) else math.inf
    t_o = other.t_ref + (other.g * H_o if math.isfinite(H_o) else 0.0)
    g_ts = _advance(own, t_o)
    f = w * 2.0 ** (g_ts - other.g)       # other's stored -> own's stored scale
    if not f > 0.0:
        return own
    order = max(0, int(own.order))
    counts, tot = own.counts, own._tot
    # Same "seen" threshold as update(): an entry that would arrive with an
    # unscaled mass < 1e-9 (an ancient or ~0-weight other) is not created,
    # else it would count as a distinct symbol in q and inflate every escape.
    # Counts shrink with context length, so the kept entries stay nested.
    thr = _DEAD * 2.0 ** own.g
    for ctx, d in other.counts.items():
        if len(ctx) > order:
            continue
        dst = counts.get(ctx)
        add = 0.0
        for s, c in d.items():
            v = c * f
            if dst is not None and s in dst:
                dst[s] += v
            elif v >= thr:
                if dst is None:
                    dst = counts[ctx] = {}
                dst[s] = v
            else:
                continue
            add += v
        if dst is not None:
            tot[ctx] = tot.get(ctx, 0.0) + add
    obs = own.n_obs
    d0 = counts.get((), {})
    for s, n in other.n_obs.items():
        if s in d0:
            obs[s] = obs.get(s, 0) + int(n)
    for s in d0:                           # symbols other had without an n_obs row
        if s not in obs:
            obs[s] = 1
    own._n1 = _singleton_mass(d0, obs)
    own.n_tokens = tot.get((), 0.0) * 2.0 ** (-own.g)
    # stats: other's are unscaled at t_o; decay them to own's clock
    fs = w * 2.0 ** min(0.0, g_ts - own.g)
    ost = other.stats
    if len(ost) == 3 and all(math.isfinite(x) for x in ost):
        own.stats = [a + fs * b for a, b in zip(own.stats, ost)]
    return own


# ================================================================ scoring
def loglik(model: PPMModel, tokens: Sequence[Token], backoff: Sequence[PPMModel] = (),
           vocab_size: int = 0, history: Sequence[Token] = ()) -> np.ndarray:
    """Per-token surprisal in bits (float64[len(tokens)]) under the chain
    [model, *backoff] as described in the module doc. Read-only. Never
    returns inf: an unseen token at the last tier gets the Good-Turing novel
    mass. O(len * order * tiers)."""
    toks = tuple(tokens)
    L = len(toks)
    out = np.empty(L, dtype=np.float64)
    if L == 0:
        return out
    V = int(vocab_size) if vocab_size and vocab_size > 0 else 0
    chain = [m for m in (model, *backoff) if m is not None and m.counts]
    if not chain:
        out.fill(math.log2(max(1, V)))
        return out
    tiers = []
    for m in chain:
        _ensure_index(m)
        tiers.append((m.counts, m._tot, max(0, int(m.order)), 2.0 ** m.g))
    maxo = max(t[2] for t in tiers)
    hist = tuple(history)[-maxo:] if maxo else ()
    seq = hist + toks
    off = len(hist)

    # Good-Turing constants of the last (non-empty) tier
    last = chain[-1]
    lcounts, ltot, lorder, lunit = tiers[-1]
    d0 = lcounts.get(())
    N = ltot.get((), 0.0) if d0 else 0.0
    if N > 0.0:
        Nu = N / lunit
        lo = 0.5 / (Nu + 1.0)
        g_nov = min(max(last._n1 / N, lo), 1.0 - lo)
        p_unseen = g_nov / max(1, V - len(d0))
        p_unseen_all = 1.0 / max(1, V - len(d0))   # every seen symbol excluded
    else:
        g_nov = 1.0
        p_unseen = p_unseen_all = 1.0 / max(1, V)
    n_upper = len(tiers) - 1

    for i in range(L):
        j = off + i
        sym = seq[j]
        P = 1.0
        acc = 0.0                               # bits folded out of P (underflow guard)
        for t in range(n_upper + 1):
            counts, tot, order, unit = tiers[t]
            is_last = t == n_upper
            kmin = 1 if is_last else 0
            excl = None
            found = False
            for k in range(min(order, j), kmin - 1, -1):
                ctx = seq[j - k:j]
                d = counts.get(ctx)
                if not d:
                    continue
                n = tot[ctx]
                q = len(d)
                if excl is not None:
                    n, q = _exclude(d, excl, n, q)
                    if q <= 0:                  # nothing new here: escape w.p. 1
                        continue
                c = d.get(sym)
                qu = q * unit
                if c is not None:
                    P *= c / (n + qu)
                    found = True
                    break
                P *= qu / (n + qu)
                excl = d
            if found:
                break
            if is_last:                         # Good-Turing step at order 0
                if d0 and N > 0.0:
                    n, q = (N, len(d0)) if excl is None else _exclude(d0, excl, N, len(d0))
                    c = d0.get(sym)
                    if c is not None and n > 0.0:
                        P *= (1.0 - g_nov) * c / n
                    else:
                        P *= p_unseen if q > 0 else p_unseen_all
                else:
                    P *= p_unseen
            elif P < 1e-150:
                acc -= math.log2(P)
                P = 1.0
        out[i] = acc - math.log2(P) if P > 0.0 else acc + 1074.0
    return out


def _exclude(d: Dict[Token, float], excl: Dict[Token, float], n: float, q: int) -> Tuple[float, int]:
    """(n, q) of context `d` without the symbols of `excl` (a higher context
    already visited). Iterates the smaller excluded set."""
    n_ex = 0.0
    q_ex = 0
    for s in excl:
        c = d.get(s)
        if c is not None:
            n_ex += c
            q_ex += 1
    n -= n_ex
    return (n if n > 0.0 else 0.0), q - q_ex


# ================================================================ stats
def entropy_rate(model: PPMModel) -> Tuple[float, float]:
    """(mu_e, sigma_e) bits/token from model.stats; (NaN, NaN) if W < 1."""
    st = model.stats
    if len(st) != 3:
        return math.nan, math.nan
    W, s1, s2 = (float(x) for x in st)
    if not (W >= 1.0 and math.isfinite(s1) and math.isfinite(s2)):
        return math.nan, math.nan
    mu = s1 / W
    var = s2 / W - mu * mu
    return mu, math.sqrt(var) if var > 0.0 else 0.0


def record_surprisal(model: PPMModel, surprisal: np.ndarray, w: float = 1.0) -> PPMModel:
    """Fold committed per-token surprisal into model.stats (for entropy_rate)."""
    s = np.asarray(surprisal, dtype=np.float64).ravel()
    w = float(w)
    if s.size == 0 or not (w > 0.0 and math.isfinite(w)):
        return model
    s = s[np.isfinite(s)]
    if s.size == 0:
        return model
    st = model.stats if len(model.stats) == 3 else [0.0, 0.0, 0.0]
    model.stats = [st[0] + w * s.size, st[1] + w * float(s.sum()),
                   st[2] + w * float(np.dot(s, s))]
    return model


def good_turing_unseen(model: PPMModel) -> float:
    """Unseen mass N1 / N of the order-0 counts (maturity = 1 - this).

    Raw ratio in [0, 1] (no floor); 1.0 for an empty model."""
    _ensure_index(model)
    N = model._tot.get((), 0.0)
    if not N > 0.0:
        return 1.0
    return min(1.0, max(0.0, model._n1 / N))


def vocab(model: PPMModel) -> Dict[Token, float]:
    """Order-0 decayed counts {token: count} (unscaled)."""
    d0 = model.counts.get(())
    if not d0:
        return {}
    f = 2.0 ** (-model.g)
    return {s: c * f for s, c in d0.items()}


# ================================================================ private
def _advance(model: PPMModel, ts: float) -> float:
    """Move the model clock to ts if newer (decaying stats, renormalising when
    g > PPM_RENORM_G) and return ts's log2 scale relative to t_ref."""
    H = model.half_life_s
    ts = float(ts)
    if not (H > 0.0 and math.isfinite(H)):
        # no decay: every row at the current scale; t_ref still tracks the
        # newest ts so merge() can place this model's clock
        if math.isfinite(ts) and (not model.counts or not ts <= model.t_ref):
            model.t_ref = ts
        return model.g
    if not math.isfinite(ts):
        if not model.counts:             # no clock yet: stay unanchored (t_ref NaN)
            model.t_ref = math.nan
            model.g = 0.0
        return model.g
    if not model.counts:                 # empty: anchor the reference here
        model.t_ref = ts
        model.g = 0.0
        return 0.0
    if math.isnan(model.t_ref):          # rows so far were "at the clock": the
        model.t_ref = ts - model.g * H   # first real ts becomes that clock
        return model.g
    g_ts = (ts - model.t_ref) / H
    if g_ts > model.g:
        _decay_stats(model, g_ts - model.g)
        model.g = g_ts
        if model.g > PPM_RENORM_G:
            g_ts -= model.g
            _renorm(model)
    return g_ts


def _decay_stats(model: PPMModel, dg: float) -> None:
    if dg > 0.0 and len(model.stats) == 3:
        f = 2.0 ** (-dg)
        model.stats = [x * f for x in model.stats]


def _renorm(model: PPMModel) -> None:
    """Rescale stored counts to 2^0 at the clock (t_ref = clock) and drop dead
    entries (unscaled < 1e-9) and their observation counts."""
    g = model.g
    f = 2.0 ** (-g)
    H = model.half_life_s
    new: Dict[Context, Dict[Token, float]] = {}
    tot: Dict[Context, float] = {}
    for ctx, d in model.counts.items():
        nd = {s: c * f for s, c in d.items() if c * f >= _DEAD}
        if nd:
            new[ctx] = nd
            tot[ctx] = math.fsum(nd.values())
    model.counts = new
    model._tot = tot
    d0 = new.get((), {})
    model.n_obs = {s: n for s, n in model.n_obs.items() if s in d0}
    model._n1 = _singleton_mass(d0, model.n_obs)
    if H > 0.0 and math.isfinite(H):
        model.t_ref = model.t_ref + g * H
    model.g = 0.0
    model.n_tokens = tot.get((), 0.0)


def _singleton_mass(d0: Dict[Token, float], obs: Dict[Token, int]) -> float:
    return math.fsum(c for s, c in d0.items() if obs.get(s, 0) == 1)


def _ensure_index(model: PPMModel) -> None:
    """Rebuild the caches when counts were replaced (O(1) check per call)."""
    d0 = model.counts.get(())
    if len(model._tot) != len(model.counts) or len(model.n_obs) != (len(d0) if d0 else 0):
        _reindex(model)


def _reindex(model: PPMModel) -> None:
    """Totals, singleton mass and n_tokens from counts. Symbols without an
    observation count (hand-built counts) get round(unscaled count), >= 1."""
    counts = model.counts
    for ctx in [c for c, d in counts.items() if not d]:
        del counts[ctx]
    model._tot = {ctx: math.fsum(d.values()) for ctx, d in counts.items()}
    d0 = counts.get((), {})
    f = 2.0 ** (-model.g)
    obs = {s: n for s, n in model.n_obs.items() if s in d0 and n >= 1}
    for s, c in d0.items():
        if s not in obs:
            obs[s] = max(1, int(round(c * f)))
    model.n_obs = obs
    model._n1 = _singleton_mass(d0, obs)
    model.n_tokens = model._tot.get((), 0.0) * f


def _enc_f(x: float) -> Optional[float]:
    x = float(x)
    return x if math.isfinite(x) else None


def _dec_f(x: Any, default: float) -> float:
    if x is None:
        return default
    x = float(x)
    return x if math.isfinite(x) else default


def _enc_tok(t: Any) -> Any:
    if isinstance(t, tuple):
        return [_enc_tok(x) for x in t]
    if isinstance(t, np.generic):
        return t.item()
    return t


def _dec_tok(t: Any) -> Any:
    if isinstance(t, list):
        return tuple(_dec_tok(x) for x in t)
    return t
