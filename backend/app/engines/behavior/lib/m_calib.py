"""Read accessors for model.calib (owner: B24 CalibrationEngine; contract C).

Why an accessor module: B24 is the single owner of detector p-values, but B29
(explain) must recompute a p from a ring snapshot to find the counterfactual
score that would not have alarmed, B30 (portrait) describes how much null
history backs each detector, and B25 reads the calibration health weights.
They all go through these pure functions so the layout of model.calib can
evolve in one place and the p a replay recomputes is bit-identical to the one
B24 issued (same float32 rounding, same seeded U, same small-sample blend).

Layout of model.calib@(s, e | class:<id>) (a plain dict; store keeps it by
reference, so B24 mutates it in place and never rebuilds it):

    {
      "layout": 1,
      "version": int,                 # model.control version the rings belong to
      "rings":  {"<detector>@<stratum>": calib.Ring},    # B24 detector rings
                                      # and '<detector>@pm|..': -log10 pm history
                                      # (pm_prior; not a Mondrian stratum)
      "refit":  {"<detector>@<stratum>": int},           # adds since last GPD fit
      "gate":   gating.GateState,     # B24 trust-gated commit bookkeeping
      "pending": {ts: int},           # stratum issued per scored tick (<= 1 d):
                                      # daypart idx + 4 tercile + 12 dt_ms
      "n_own": float,                 # trust-weighted 15-min-equivalent commits
      "n_admit": int,                 # ticks admitted into the rings
      "resets": int, "dt": float, "profile_ts": float,
      "meta": {...}                   # OWNER B25: meta rings 'meta_inst@<st>' /
                                      # 'meta_all@<st>'; B24 never touches it
    }

Multi-writer rule: B24 and B25 both write model.calib. Each must
read-modify-write the SAME dict (m = store.get_model(...); m['meta'] = ...;
store.put_model(..., m)) and only touch its own keys; B24 preserves every key
it does not own. Rings may be live `calib.Ring` objects or their to_dict()
form; every accessor accepts both (`as_ring`). `to_json` gives a JSON-safe
deep copy for export.

At (s, '__system__') the model holds only {"layout", "health"}: the rolling
per-detector health state behind behavior.calib_health.

Strata (Mondrian): daypart(4) x cadence class for every detector except
identity, which uses (daypart, regime tercile, cadence class) - see `stratum_for`. The regime
tercile is 0 settled (NORMAL / RETURNED / unknown), 1 in question (SUSPECT /
DRIFTING / REJECTED), 2 re-learning (ACCEPTED), unless behavior.regime
carries an explicit integer 'tercile' in 0..2 (`regime_tercile`).

p-value (B24, engines.md B24; lib/calib docstring for the tail guards):
    p = calib.p_from_ring(ring, s, u)       # randomised conformal, GPD tail
                                            # (xi >= 0 floor, <= 1/(n+1) cap
                                            # beyond the ring maximum)
    if |ring| < 64: p = calib.blend_small_sample(p, prior, |ring|)
                    p = max(p, #{ring > s} / (|ring| + 1))   # own-history floor
    stored = issued(p)                      # float32, floored at 1.2e-38
with u = uniform(system, entity, detector, ts) and prior the first usable
of: the entity's own rings of the other dayparts (pooled, >= 64), the pm
prior pm_prior(pm ring, behavior.pm[d], u) (pm with its atoms at 1 - the
accumulators' stationary p_eq at a zero statistic - and at the float floor
randomised over their null mass, estimated from the pm ring
'<d>@pm|<cc>' (canonical H / Q detectors '<d>@pm|g:<g>') of the entity's
admitted -log10 pm), the class-pooled
ring p (>= 64 pooled entries), and for identity only the entity's own
settled-regime ring of the same daypart (>= 64 entries).
NaN score -> NaN p, never 1. B24 refreshes a stale GPD fit (>= 16 additions)
just before scoring a tail score, so replay must use the model as it is
after B24's run at that tick (Rings are copy-on-write: keep a shallow copy
of each Ring, or the Ring objects of a new rings dict, as the snapshot).

Signatures (all pure; no store access):
    as_ring(x) -> Ring | None
    stratum_for(detector, daypart, cc, regime_tercile=0) -> str
    ring_key_for(detector, daypart, cc, regime_tercile=0) -> str
    regime_tercile(regime) -> int                 # behavior.regime dict -> 0..2
    rings(model) -> {key: Ring}                   # B24 detector rings only
    ring(model, detector, stratum) -> Ring | None # also finds B25 meta rings
    ring_size(model, detector, stratum) -> int
    tail(model, detector, stratum) -> GPDTail | None
    quantile(model, detector, stratum, q) -> float        # null score quantile
    score_at_p(model, detector, stratum, p) -> float      # score whose p is p
    uniform(system, entity, detector, ts) -> float        # the seeded U of B24
    p_value(ring, score, u, prior=nan) -> float           # B24's p, one ring
    pm_stratum(detector, cc, grain=False) -> str;  pm_ring_key(...) -> str
    pm_score(pm) -> float
    pm_prior(pm_ring, pm, u, min_n=16) -> float           # B24's pm prior, atoms randomised
    issued(p) -> float                                    # as stored in behavior.p
    pooled_p(rings, score, u, min_n=64) -> (p, n)         # conformal over a union
    p_from_snapshot(model, detector, stratum, score, u,
                    pm=None, pooled=None, settled=None) -> float
    issued_stratum(model, ts) -> (daypart, tercile, dt) | None   # B24 'pending' code
    p_replay(model, detector, daypart, cc, score, u, tercile=0,
             pm=None, class_rings=None) -> float   # B24's full prior order (B29)
    version(model) -> int
    describe(model) -> dict                               # portrait summary
    weight_mult(calib_health, detector) -> float          # B25 family weights
    health_of(calib_health, detector) -> dict
    to_json(model) -> dict
"""
from __future__ import annotations

import functools
import math
from collections import deque
from collections.abc import Mapping      # not typing.Mapping: isinstance is on the hot path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from . import calib, combine, evt, timebins
from .detectors import DETECTOR_INFO

MODEL = "model.calib"
LAYOUT = 1
RINGS = "rings"
META = "meta"            # owned by B25
HEALTH = "health"        # at (s, '__system__')
SMALL_N = calib.SMALL_N
# behavior.p is a float32 ring (contract B): a tail p below the smallest
# normal float32 would be stored as 0 (or a denormal) and read downstream as
# "impossible" (-log10 p = inf). Issued p-values are floored here instead of
# at calib.P_FLOOR = 1e-300; every decision threshold (e_day <= 3e-6 at 60 s
# is p ~ 2e-9) is 29 decades above it.
P_ISSUED_FLOOR = float(np.finfo(np.float32).tiny)      # 1.1754944e-38

REGIME_TERCILE = {
    "normal": 0, "returned": 0,
    "suspect": 1, "drifting": 1, "rejected": 1,
    "accepted": 2,
}

_EMPTY_RING = calib.Ring()


def _f(x: Any) -> float:
    if x is None:
        return math.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


# ------------------------------------------------------------------ rings
def as_ring(x: Any) -> Optional[calib.Ring]:
    """A Ring from a live Ring or its to_dict() form; None for anything else."""
    if isinstance(x, calib.Ring):
        return x
    if isinstance(x, Mapping):
        return calib.Ring.from_dict(x)
    return None


def stratum_for(detector: str, daypart: str, cc: int, regime_tercile: int = 0,
                grain: Optional[str] = None, prov: int = 0) -> str:
    """Stratum label of `detector`: 'daypart|cc', or 'daypart|r<k>|cc' for identity.

    spec v2.1: with `grain` ('h' | 'q', canonical mode) an H / Q stream
    detector gets its grain stratum ('daypart|g:h', 'daypart|g:q|p:<prov>',
    identity 'daypart|r<k>|g:h'; calib.grain_stratum_key); T-stream detectors
    ignore `grain` and keep 'daypart|cc'. `daypart` is then the daypart of
    the grain row's midpoint (grains.row_tctx) for H / Q detectors."""
    info = DETECTOR_INFO.get(detector, {})
    is_id = info.get("strata") == "daypart_regime"
    if grain is not None and info.get("stream", "t") != "t":
        g = str(info.get("stream"))
        if is_id:
            return calib.grain_stratum_key(daypart, "h", tercile=regime_tercile)
        return calib.grain_stratum_key(daypart, g, prov=prov if g == "q" else None)
    if is_id:
        return calib.identity_stratum_key(daypart, regime_tercile, cc)
    return calib.stratum_key(daypart, cc)


def ring_key_for(detector: str, daypart: str, cc: int, regime_tercile: int = 0,
                 grain: Optional[str] = None, prov: int = 0) -> str:
    return calib.ring_key(detector, stratum_for(detector, daypart, cc, regime_tercile,
                                                grain=grain, prov=prov))


def regime_tercile(regime: Any) -> int:
    """behavior.regime value -> identity regime stratum 0..2.

    An explicit integral 'tercile' in 0..2 wins; otherwise the governor state
    (case-insensitive) is mapped through REGIME_TERCILE; unknown / missing -> 0.
    """
    if not isinstance(regime, Mapping):
        return 0
    t = regime.get("tercile")
    if t is not None and not isinstance(t, bool):
        v = _f(t)
        if v == v and v.is_integer() and 0 <= v <= 2:
            return int(v)
    st = regime.get("state")
    return REGIME_TERCILE.get(str(st).lower(), 0) if st is not None else 0


def rings(model: Optional[Mapping]) -> Dict[str, calib.Ring]:
    """B24 detector rings {key: Ring} (dict forms converted; meta rings excluded)."""
    if not isinstance(model, Mapping):
        return {}
    raw = model.get(RINGS) or {}
    out: Dict[str, calib.Ring] = {}
    for k, v in raw.items():
        r = as_ring(v)
        if r is not None:
            out[k] = r
    return out


def ring(model: Optional[Mapping], detector: str, stratum: str) -> Optional[calib.Ring]:
    """The ring for (detector, stratum): B24 rings first, then B25's meta
    rings (model['meta'] or a top-level key), else None."""
    if not isinstance(model, Mapping):
        return None
    key = calib.ring_key(detector, stratum)
    for holder in (model.get(RINGS), model.get(META), model):
        if isinstance(holder, Mapping) and key in holder:
            return as_ring(holder[key])
    return None


def ring_size(model: Optional[Mapping], detector: str, stratum: str) -> int:
    r = ring(model, detector, stratum)
    return 0 if r is None else len(r)


def tail(model: Optional[Mapping], detector: str, stratum: str) -> Optional[calib.GPDTail]:
    r = ring(model, detector, stratum)
    return None if r is None else r.gpd


def quantile(model: Optional[Mapping], detector: str, stratum: str, q: float) -> float:
    """Empirical null score quantile of the ring (NaN when empty / absent)."""
    r = ring(model, detector, stratum)
    return math.nan if r is None else r.quantile(q)


def score_at_p(model: Optional[Mapping], detector: str, stratum: str, p: float) -> float:
    """The score at which the ring's p reaches `p` (B29 counterfactual target).

    Inside the body (p >= 1/(n+1)) the empirical (1 - p) quantile; in the
    fitted tail (p < tail.rate) the POT quantile evt.pot_quantile. The
    randomised tie term is ignored (it moves p by at most 1/(n+1)). Empty
    ring or NaN p -> NaN.
    """
    p = _f(p)
    r = ring(model, detector, stratum)
    if r is None or not len(r) or p != p:
        return math.nan
    t = r.gpd
    if t is not None and t.valid() and p < t.rate:
        return evt.pot_quantile(t.u, t.xi, t.sigma, t.rate, p)
    n = len(r)
    if p < 1.0 / (n + 1):
        return float(r.scores[-1])
    return r.quantile(min(1.0, max(0.0, 1.0 - p)))


# ------------------------------------------------------------------ p-values
def uniform(system: str, entity: str, detector: str, ts: float) -> float:
    """The tie-breaking U of B24 at (system, entity, detector, ts): seeded blake2b."""
    return combine.seeded_uniform(system, entity, detector, float(ts))


# ------------------------------------------------------------------ pm prior
PM_STRATUM = "pm"        # '<detector>@pm|<cc>' / '@pm|g:<g>': own history of -log10 pm[d]
PM_RING_MIN = 16         # pm-ring entries before an atom's mass is taken from the ring
PM_ATOM_ONE = 0.0        # pm_score of pm = 1 (a statistic at its minimum: p_eq = 1)
PM_ATOM_FLOOR = calib._r32(-math.log10(P_ISSUED_FLOOR))   # pm_score at the float32 floor (37.93)


def pm_stratum(detector: str, cc: int, grain: bool = False) -> str:
    """Stratum of `detector`'s pm ring: 'pm|<cadence class>', or in canonical
    grain mode 'pm|g:h' / 'pm|g:q' for an H / Q stream detector. Not split
    by daypart (the ring must fill fast), but by cadence: a T-stream pm
    (and every pm in tick mode) changes its null with the tick length, while
    a grain pm is scored on fixed wall-clock rows at every cadence."""
    if grain:
        g = str(DETECTOR_INFO.get(detector, {}).get("stream", "t"))
        if g in ("h", "q"):
            return f"{PM_STRATUM}|g:{g}"
    return f"{PM_STRATUM}|{int(cc)}"


def pm_ring_key(detector: str, cc: int, grain: bool = False) -> str:
    """Ring key of `detector`'s pm ring in model.calib['rings'] (B24)."""
    return calib.ring_key(detector, pm_stratum(detector, cc, grain))


def pm_score(pm: Any) -> float:
    """-log10 pm as stored in a pm ring, with every pm at or below the
    float32 floor (P_ISSUED_FLOOR, incl. a stored 0 / denormal) on the one
    floor atom PM_ATOM_FLOOR; NaN for a missing / out-of-range pm."""
    v = _valid_p(pm)
    if v != v:
        return math.nan
    if v <= P_ISSUED_FLOOR:
        return PM_ATOM_FLOOR
    return calib._r32(-math.log10(v)) if v < 1.0 else PM_ATOM_ONE


def pm_prior(pm_ring: Optional[calib.Ring], pm: Any, u: float,
             min_n: int = PM_RING_MIN) -> float:
    """B24's small-sample prior from behavior.pm[d]: pm, with its two point
    masses randomised (engines.md B24, integration §8.2).

    pm is a p-value with atoms: an accumulator's stationary p_eq is exactly
    1 whenever its statistic is 0, and an over-dispersed or peer-scored pm
    sits at the float floor on every null tick of some entities (a nightly
    backup host scored against its peers: 1e-300 every night). As a logit
    prior an atom puts the issued p back in a point mass at 1, or at the
    floor, on every such tick. The randomised (mid-p) rule for a tie at an
    atom a with null mass pi: p = P(pm < a) + u P(pm = a), i.e.
        pm = 1      -> 1 - pi1 + u pi1
        pm = floor  -> u pi0
    with u the score's own seeded U. pi is the atom's share of the entity's
    admitted pm history (the pm ring of this detector and cadence) once it
    holds `min_n` entries, else the Laplace estimate (k + 1) / (n + 2) at
    the atom 1 (no history: 1 - u/2 ... 1) and, at the floor, pm itself (a
    floor never seen before is evidence, not a routine atom). Any other pm
    (the body) is used as is: pm is exposure-exact there by construction,
    and a ring-conformal pm would be coarse (>= 1/(n+1)) for weeks on a
    sparse stratum. NaN when pm is missing or out of range."""
    x = pm_score(pm)
    if x != x:
        return math.nan
    if x != PM_ATOM_ONE and x != PM_ATOM_FLOOR:
        return _valid_p(pm)
    r = as_ring(pm_ring)
    n = 0 if r is None else r.scores.size
    k = 0
    if n:
        sc = r.scores
        k = int(sc.searchsorted(x, "right") - sc.searchsorted(x, "left"))
    uu = _f(u)
    if uu != uu:
        return math.nan
    if x == PM_ATOM_ONE:
        pi1 = k / n if n >= int(min_n) else (k + 1.0) / (n + 2.0)
        return 1.0 - pi1 + uu * pi1
    if n < int(min_n) or k == 0:
        return _valid_p(pm)                 # an unseen floor: keep the evidence
    return uu * (k / n)


def p_value(r: Optional[calib.Ring], score: float, u: float, prior: float = math.nan) -> float:
    """B24's p for one ring: calib.p_from_ring (conformal + GPD tail), then the
    small-sample logit blend with `prior` (weight n/(n+64)) when |ring| < 64
    and the prior is a finite p. NaN score -> NaN."""
    s = _f(score)
    if s != s:
        return math.nan
    rr = _EMPTY_RING if r is None else r
    p = calib.p_from_ring(rr, s, u)
    n = rr.scores.size
    if n < SMALL_N and prior == prior and prior is not None:
        pr = _f(prior)
        if 0.0 <= pr <= 1.0:
            p = calib.blend_small_sample(p, pr, n)
            # own-history floor: k admitted null scores of this stratum were
            # strictly above s, so the prior may add resolution BEYOND the
            # ring, never contradict it: p >= k / (n + 1). A sparse entity
            # (a nightly backup host: one scored hour a night, ~10 entries
            # after two weeks) scored every night against its peers had
            # pm ~ 1e-18 with the same score in its own ring every night, and
            # the blend (weight n/(n+64) ~ 0.14) issued p ~ 1e-16 every night.
            k = n - int(rr.scores.searchsorted(calib._r32(s), "right")) if n else 0
            if k > 0:
                p = max(p, k / (n + 1.0))
    return p


def issued(p: float) -> float:
    """p exactly as B24 stores it in behavior.p: floored at P_ISSUED_FLOOR and
    rounded to float32 (NaN stays NaN)."""
    v = _f(p)
    if v != v:
        return math.nan
    return float(np.float32(min(1.0, max(P_ISSUED_FLOOR, v))))


def pooled_p(rs: Iterable[Any], score: float, u: float,
             min_n: int = SMALL_N) -> Tuple[float, int]:
    """Randomised conformal p of `score` against the union of several rings
    (class pooling): (sum #{c > s} + u (sum #{c == s} + 1)) / (sum n + 1).
    Returns (p, n_pooled); p is NaN when n_pooled < min_n or score is NaN.
    O(k log M), no concatenation."""
    s = _f(score)
    n = gt = eq = 0
    if s == s:
        x = calib._r32(s)
        for r in rs:
            r = as_ring(r)
            if r is None or not len(r):
                continue
            sc = r.scores
            lo = int(sc.searchsorted(x, "left"))
            hi = int(sc.searchsorted(x, "right"))
            n += sc.size
            gt += sc.size - hi
            eq += hi - lo
    if s != s or n < max(1, int(min_n)):
        return math.nan, n
    uu = _f(u)
    if uu != uu:
        return math.nan, n
    return (gt + uu * (eq + 1)) / (n + 1), n


def p_from_snapshot(model: Optional[Mapping], detector: str, stratum: str, score: float,
                    u: float, pm: Optional[float] = None,
                    pooled: Union[None, float, Iterable[Any]] = None,
                    settled: Union[None, float, calib.Ring, Mapping] = None,
                    pm_ring: Union[None, calib.Ring, Mapping] = None) -> float:
    """Recompute B24's p from a model.calib snapshot (B29 replay); float64,
    equal to the stored behavior.p after issued().

    pm: behavior.pm[d] at that tick, used as is (B25 meta rings: the raw HMP
    p) unless `pm_ring` (B24 detectors: the snapshot's pm ring,
    pm_ring_key) is given, then B24's pm_prior(pm_ring, pm, u); pooled: the
    class-pooled p, or the member rings to pool; settled: (identity only)
    the settled-regime p or ring.
    The prior is the first usable of pm, pooled (>= 64 entries) and settled
    (>= 64 entries), exactly as B24 chooses it.
    """
    r = ring(model, detector, stratum)
    n = 0 if r is None else len(r)
    prior = math.nan
    if n < SMALL_N:
        prior = pm_prior(pm_ring, pm, u) if pm_ring is not None else _valid_p(pm)
        if prior != prior and pooled is not None:
            if isinstance(pooled, (int, float, np.floating)):
                prior = _valid_p(pooled)
            else:
                prior, _ = pooled_p(pooled, score, u)
        if prior != prior and settled is not None:
            if isinstance(settled, (int, float, np.floating)):
                prior = _valid_p(settled)
            else:
                sr = as_ring(settled)
                if sr is not None and len(sr) >= SMALL_N:
                    prior = p_value(sr, score, u)
    return p_value(r, score, u, prior)


@functools.lru_cache(maxsize=1024)
def _stratum_cached(detector: str, daypart: str, cc: int, tercile: int) -> str:
    return stratum_for(detector, daypart, cc, tercile)


def decode_pending(code: Any) -> Tuple[str, int, float, Optional[Dict[str, Any]]]:
    """(daypart, tercile, dt, grain) of a B24 'pending' entry. v2 / tick mode
    is an int (daypart idx + 4 tercile + 12 dt_ms, grain None); spec v2.1
    canonical mode is (code, h daypart idx, q daypart idx, prov bitmask over
    detectors.Q_DETECTORS) with grain = {'dp_h', 'dp_q', 'prov': {d: 0|1}}."""
    from .detectors import Q_DETECTORS
    if isinstance(code, (list, tuple)):
        c = int(code[0])
        dph = timebins.DAYPARTS[int(code[1]) % 4]
        dpq = timebins.DAYPARTS[int(code[2]) % 4]
        mask = int(code[3]) if len(code) > 3 else 0
        g = {"dp_h": dph, "dp_q": dpq,
             "prov": {d: (mask >> i) & 1 for i, d in enumerate(Q_DETECTORS)}}
    else:
        c = int(code)
        g = None
    return timebins.DAYPARTS[c % 4], (c // 4) % 3, (c // 12) / 1000.0, g


def issued_stratum(model: Optional[Mapping], ts: float) -> Optional[Tuple[str, int, float]]:
    """(daypart, regime tercile, dt) B24 recorded when it issued the p of tick
    ts (model['pending'], kept <= 1 d), or None. The code is B24's
    daypart index + 4 tercile + 12 dt_ms (timebins.DAYPARTS order). The
    spec v2.1 grain part is issued_grain(model, ts)."""
    if not isinstance(model, Mapping):
        return None
    pend = model.get("pending")
    if not isinstance(pend, Mapping):
        return None
    code = pend.get(float(ts))
    if code is None:
        return None
    dp, terc, dt, _g = decode_pending(code)
    return dp, terc, dt


def issued_grain(model: Optional[Mapping], ts: float) -> Optional[Dict[str, Any]]:
    """spec v2.1: {'dp_h', 'dp_q', 'prov'} B24 recorded at tick ts in
    canonical mode (None in tick mode or when not recorded)."""
    if not isinstance(model, Mapping):
        return None
    pend = model.get("pending")
    if not isinstance(pend, Mapping):
        return None
    code = pend.get(float(ts))
    if code is None:
        return None
    return decode_pending(code)[3]


def p_replay(model: Optional[Mapping], detector: str, daypart: str, cc: int, score: float,
             u: float, tercile: int = 0, pm: Optional[float] = None,
             class_rings: Optional[Iterable[Any]] = None, grain: Optional[str] = None,
             prov: int = 0) -> float:
    """B24's p of `score` for `detector` exactly as its scoring step chooses it
    (calibration._score / _prior), from a model.calib snapshot:

      |ring| >= 64: calib.p_from_ring (conformal + GPD tail);
      else the logit blend with the first usable prior of
        1. the entity's own rings of the OTHER dayparts at this cadence,
           pooled (>= 64 entries; not for identity),
        2. pm_prior(the pm ring, behavior.pm[d] at that tick, u),
        3. the class-pooled ring (class_rings: the members' rings of the same
           key, the entity itself excluded; >= 64 entries),
        4. identity only: the settled-regime (tercile 0) ring of the same
           daypart (>= 64 entries).
    Returned as issued (float32, floored). NaN score -> NaN.

    spec v2.1: `grain` (canonical mode) selects the grain strata of H / Q
    stream detectors (stratum_for); `daypart` is then the grain row's
    daypart and `prov` the provisional flag of a Q detector. The own-daypart
    prior pools the other dayparts of the same grain stratum.
    """
    s = _f(score)
    if s != s:
        return math.nan
    is_id = DETECTOR_INFO.get(detector, {}).get("strata") == "daypart_regime"
    gr = grain if (grain is not None and DETECTOR_INFO.get(detector, {}).get("stream", "t") != "t") \
        else None
    if gr is not None:
        st = stratum_for(detector, daypart, int(cc), int(tercile), grain=gr, prov=prov)
    else:
        st = _stratum_cached(detector, daypart, int(cc), int(tercile))
    r = ring(model, detector, st)
    n = 0 if r is None else len(r)
    if n >= SMALL_N:
        return issued(calib.p_from_ring(r, s, u))
    prior = math.nan
    if not is_id and (cc or gr is not None):
        if gr is not None:
            keys = [stratum_for(detector, p, int(cc), 0, grain=gr, prov=prov)
                    for p in timebins.DAYPARTS if p != daypart]
        else:
            keys = [_stratum_cached(detector, p, int(cc), 0)
                    for p in timebins.DAYPARTS if p != daypart]
        own = [x for x in (ring(model, detector, k) for k in keys) if x is not None]
        if own:
            prior, _ = pooled_p(own, s, u)
    if prior != prior:
        prior = pm_prior(ring(model, detector, pm_stratum(detector, cc, grain is not None)),
                         pm, u)
    if prior != prior and class_rings is not None:
        prior, _ = pooled_p(class_rings, s, u)
    if prior != prior and is_id and int(tercile) != 0:
        r0 = ring(model, detector, stratum_for(detector, daypart, cc, 0, grain=gr))
        if r0 is not None and len(r0) >= SMALL_N:
            prior = p_value(r0, s, u)
    return issued(p_value(r, s, u, prior))


def _valid_p(x: Any) -> float:
    v = _f(x)
    return v if (v == v and 0.0 <= v <= 1.0) else math.nan


# ------------------------------------------------------------------ summaries
def version(model: Optional[Mapping]) -> int:
    if not isinstance(model, Mapping):
        return 0
    v = _f(model.get("version"))
    return int(v) if v == v else 0


def describe(model: Optional[Mapping]) -> Dict[str, Any]:
    """Portrait / profile summary: ring counts and sizes per detector and stratum."""
    rs = rings(model)
    by_det: Dict[str, Dict[str, int]] = {}
    tails = 0
    for key, r in rs.items():
        d, st = calib.split_ring_key(key)
        by_det.setdefault(d, {})[st] = len(r)
        tails += r.gpd is not None
    m = model if isinstance(model, Mapping) else {}
    na = _f(m.get("n_admit"))
    return {
        "version": version(model),
        "n_rings": len(rs),
        "n_tails": tails,
        "n_admit": int(na) if na == na else 0,
        "n_own": _f(m.get("n_own")),
        "detectors": by_det,
    }


def health_of(calib_health: Optional[Mapping], detector: str) -> Dict[str, Any]:
    """behavior.calib_health entry of `detector` ({} when absent)."""
    if not isinstance(calib_health, Mapping):
        return {}
    h = calib_health.get(detector)
    return dict(h) if isinstance(h, Mapping) else {}


def weight_mult(calib_health: Optional[Mapping], detector: str) -> float:
    """weight_mult of `detector` from a behavior.calib_health value; 1.0 when
    absent or not a finite positive number (health never silences a detector)."""
    h = calib_health.get(detector) if isinstance(calib_health, Mapping) else None
    w = _f(h.get("weight_mult")) if isinstance(h, Mapping) else math.nan
    return w if (w == w and 0.0 < w <= 1.0) else 1.0


def to_json(model: Any) -> Any:
    """JSON-safe deep copy (Rings -> to_dict, arrays / deques -> lists,
    GateState -> to_dict, non-finite floats -> None, keys -> str)."""
    if isinstance(model, calib.Ring):
        return to_json(model.to_dict())
    if isinstance(model, calib.GPDTail):
        return to_json(model.to_dict())
    td = getattr(model, "to_dict", None)
    if callable(td) and not isinstance(model, Mapping):
        return to_json(td())
    if isinstance(model, Mapping):
        return {str(k): to_json(v) for k, v in model.items()}
    if isinstance(model, (list, tuple, deque)):
        return [to_json(v) for v in model]
    if isinstance(model, np.ndarray):
        return [to_json(v) for v in model.tolist()]
    if isinstance(model, (bool, np.bool_)):
        return bool(model)
    if isinstance(model, (int, np.integer)):
        return int(model)
    if isinstance(model, (float, np.floating)):
        v = float(model)
        return v if math.isfinite(v) else None
    return model
