"""Read accessors and pure maths of model.xsys (owner: B21 cross_system; P2).

Layout of model.xsys@('__org__', '__org__') (contract C, "P2 models"):

    {
      "fmt": 1, "version": int, "ts": float,
      "pairs": {"<system>|<ip>": {"state": PairState, "gate": GateState}},
      "live":  {"<system>|<ip>": {"first": ts first observed active (ungated),
                                  "last": ts last observed active,
                                  "wid": start of the last H window scored,
                                  "rows": {ts: (wid, day)} scored rows kept
                                          ROW_KEEP_S for commit / replay}},
      "emitted": {"<system>|<ip>": ts of the last first_access_system},
      "tiers": {"ts": float, "cls": {cls: ClassTier}, "ip_cls": {ip: cls},
                "org": {system: c}, "N_org": float},
    }

PairState (one per (system, IP); the learner state of lib/gating, keyed by
the pair's own (system, ip) so trust, quarantine, model.control and the
checkpoints are the pair's):

    {"t": t_ref, "c": decayed count of committed active H windows at t_ref
     (half-life HALF_LIFE_S), "n": windows committed with weight > 0,
     "first": / "last": ts of the first / last committed window,
     "wins": [window start, ...] committed (dedupe, last WIN_KEEP_S),
     "days": {local-day ordinal: max weight}} (last DAY_KEEP days)

ClassTier: {"c": {system: decayed count}, "N": float, "N1": float (singleton
pair mass), "members": [ip, ...], "df": {system: members using it},
"first": {system: {ip: first observed ts}}, "xbar": mean extra systems per
active member-day, "xdays": member-days}.

Unit of observation (canonical grain semantics, docs/lib3/cadence.md D1/D2):
one (system, IP) pair counts once per epoch-aligned H window (3600 s) in
which it was active, at every cadence; in tick mode the window is the tick.
The daily footprint and the 24-h spread are wall-clock quantities.

Predictive (engines.md B21, three tiers with Good-Turing maturity): for IP i
with committed pair counts c_i(s), N_i = sum, N1_i = singleton mass,

    p_org(s) = (c_org(s) + 1) / (N_org + |S|)                (Laplace base)
    p_cls(s) = w_c c_cls(s) / N_cls + (1 - w_c) p_org(s),   w_c = 1 - (N1_cls + 1/2)/(N_cls + 1)
    p_i(s)   = w_i c_i(s) / N_i     + (1 - w_i) p_cls(s),   w_i = 1 - (N1_i + 1/2)/(N_i + 1)

(1 - w = the Good-Turing unseen mass of the tier; the class counts leave the
IP out). The detector p of a scored pair-window is the upper p-value of s under p_i
(upper_p: the mass of the systems at most as probable as s): 1 for a
habitual system, tiny for a system neither the IP nor its class uses; the
score is -log10 p on a 0.25-decade grid (score_of).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import bayes
from .classkeys import ORG

MODEL = "model.xsys"
HALF_LIFE_S = 30 * 86400.0          # decayed pair counts (as B08's vocab)
H_WINDOW_S = 3600.0                 # canonical unit of observation (the H grain)
WIN_KEEP_S = 9 * 86400.0            # committed window ids kept (release reaches 8 d)
DAY_KEEP = 35                       # committed active local days kept per pair
SPREAD_DAYS = 28                    # daily footprint history used by the NB
SPREAD_KAPPA = 4.0                  # pseudo-days of the class prior
SPREAD_A0, SPREAD_B0 = 0.5, 1.0     # Gamma hyperprior of the extra systems per day
GT_SMOOTH = 0.5                     # Good-Turing (N1 + 1/2) / (N + 1)
ACTIVE_MIN = 0.5                    # a pair "uses" s when its decayed count >= 1/2
P_FLOOR = 1e-300
SCORE_STEP = 0.25                   # score grid, decades (score_of)


# ================================================================ keys
def pair_key(system: str, ip: str) -> str:
    return f"{system}|{ip}"


def split_pair(key: str) -> Tuple[str, str]:
    s, _, ip = key.partition("|")
    return s, ip


def window_start(now: float, dt: float, canonical: bool) -> float:
    """Start of the unit window a tick belongs to: the epoch-aligned H window
    containing the tick's end (canonical), the tick itself (tick mode)."""
    if not canonical:
        return float(now)
    return math.floor((float(now) - 1e-3) / H_WINDOW_S) * H_WINDOW_S


# ================================================================ accessors
def get(store) -> Dict[str, Any]:
    return store.get_model(ORG[0], ORG[1], MODEL, default=None) or {}


def decayed(st: Optional[Mapping[str, Any]], now: float) -> float:
    """Decayed committed window count of a pair state at `now`."""
    if not st:
        return 0.0
    c = float(st.get("c", 0.0))
    t = float(st.get("t", now))
    if not c > 0.0:
        return 0.0
    return c * 2.0 ** (-(float(now) - t) / HALF_LIFE_S) if now > t else c


def footprint(model: Mapping[str, Any], ip: str, now: float) -> Dict[str, Dict[str, float]]:
    """{system: {c, n, first, last}} of an IP's committed pairs."""
    out: Dict[str, Dict[str, float]] = {}
    for key, rec in (model.get("pairs") or {}).items():
        s, e = split_pair(key)
        if e != ip:
            continue
        st = rec.get("state") or {}
        out[s] = {"c": decayed(st, now), "n": float(st.get("n", 0)),
                  "first": float(st.get("first", math.nan)),
                  "last": float(st.get("last", math.nan))}
    return out


def footprints(model: Mapping[str, Any], now: float) -> Dict[str, Dict[str, Dict[str, float]]]:
    """{ip: footprint(model, ip, now)} for every IP, in one pass."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key, rec in (model.get("pairs") or {}).items():
        s, e = split_pair(key)
        st = rec.get("state") or {}
        out.setdefault(e, {})[s] = {"c": decayed(st, now), "n": float(st.get("n", 0)),
                                    "first": float(st.get("first", math.nan)),
                                    "last": float(st.get("last", math.nan))}
    return out


def systems_accessed(store, ip: str, now: float) -> Dict[str, Dict[str, float]]:
    """What profile.extra.systems_accessed shows: the IP's committed footprint."""
    return footprint(get(store), ip, now)


# ================================================================ learner
def new_pair_state() -> Dict[str, Any]:
    return {"t": math.nan, "c": 0.0, "n": 0, "first": math.inf, "last": -math.inf,
            "wins": [], "days": {}}


def pair_update(st: Dict[str, Any], row: Tuple[float, float, int], w: float) -> Dict[str, Any]:
    """Pure, deterministic update with one committed row (ts, window, day).

    A window is counted once (its first scored tick is its only row, but a
    rollback replay or a release may present it again). Rows older than the
    state's reference time are decayed into it (release commits old rows
    after newer ones; lib/gating)."""
    ts, wid, day = float(row[0]), float(row[1]), int(row[2])
    wins = list(st.get("wins") or [])
    if wid in wins:
        return st
    out = dict(st)
    w = max(0.0, min(1.0, float(w)))
    t_ref = float(st.get("t", math.nan))
    c = float(st.get("c", 0.0))
    if not math.isfinite(t_ref):
        t_ref = ts
    if ts >= t_ref:
        c = c * 2.0 ** (-(ts - t_ref) / HALF_LIFE_S) + w
        t_ref = ts
    else:
        c = c + w * 2.0 ** (-(t_ref - ts) / HALF_LIFE_S)
    out["t"], out["c"] = t_ref, c
    if w > 0.0:
        out["n"] = int(st.get("n", 0)) + 1
        out["first"] = min(float(st.get("first", math.inf)), ts)
        out["last"] = max(float(st.get("last", -math.inf)), ts)
        days = dict(st.get("days") or {})
        days[day] = max(float(days.get(day, 0.0)), w)
        lo = max(days) - DAY_KEEP
        out["days"] = {d: v for d, v in days.items() if d > lo}
    wins.append(wid)
    wins.sort()
    cut = max(wins[-1], t_ref) - WIN_KEEP_S
    out["wins"] = [x for x in wins if x >= cut]
    return out


def dump_pair(st: Dict[str, Any]) -> Dict[str, Any]:
    """Checkpoint blob: the lists as float64 arrays (lib/gating deep-copies
    every blob; an array copies in C, a 200-float list element by element)."""
    days = st.get("days") or {}
    return {"t": st.get("t"), "c": st.get("c"), "n": st.get("n"), "first": st.get("first"),
            "last": st.get("last"),
            "wins": np.asarray(st.get("wins") or [], dtype=np.float64),
            "days": np.asarray(sorted(days.items()), dtype=np.float64).reshape(-1, 2)}


def load_pair(blob: Any) -> Dict[str, Any]:
    if not isinstance(blob, Mapping):
        return new_pair_state()
    st = new_pair_state()
    st.update({k: blob[k] for k in blob if k in st})
    st["wins"] = [float(x) for x in np.asarray(st.get("wins") if st.get("wins") is not None
                                                else [], dtype=np.float64).reshape(-1)]
    d = st.get("days")
    if isinstance(d, Mapping):
        st["days"] = {int(k): float(v) for k, v in d.items()}
    else:
        a = np.asarray(d if d is not None else [], dtype=np.float64).reshape(-1, 2)
        st["days"] = {int(k): float(v) for k, v in a.tolist()}
    return st


# ================================================================ predictive
def gt_weight(n: float, n1: float) -> float:
    """Good-Turing maturity w = 1 - (N1 + 1/2)/(N + 1) (0 without data)."""
    if not n > 0.0:
        return 0.0
    return max(0.0, 1.0 - (max(0.0, n1) + GT_SMOOTH) / (n + 1.0))


def predictive(systems: Sequence[str], c_ip: Mapping[str, float], n1_ip: float,
               c_cls: Optional[Mapping[str, float]], n1_cls: float,
               c_org: Mapping[str, float]) -> Dict[str, float]:
    """Three-tier backoff p_i(s) over `systems` (module docstring). Sums to 1."""
    systems = list(systems)
    k = len(systems)
    if not k:
        return {}
    n_org = sum(max(0.0, float(c_org.get(s, 0.0))) for s in systems)
    p = {s: (max(0.0, float(c_org.get(s, 0.0))) + 1.0) / (n_org + k) for s in systems}
    if c_cls:
        n_c = sum(max(0.0, float(c_cls.get(s, 0.0))) for s in systems)
        if n_c > 0.0:
            w = gt_weight(n_c, n1_cls)
            p = {s: w * max(0.0, float(c_cls.get(s, 0.0))) / n_c + (1.0 - w) * p[s]
                 for s in systems}
    n_i = sum(max(0.0, float(c_ip.get(s, 0.0))) for s in systems)
    if n_i > 0.0:
        w = gt_weight(n_i, n1_ip)
        p = {s: w * max(0.0, float(c_ip.get(s, 0.0))) / n_i + (1.0 - w) * p[s] for s in systems}
    return p


def upper_p(pred: Mapping[str, float], s: str) -> float:
    """p-value of system s under the predictive: the mass of the systems at
    most as probable as s (s included). A habitual system reads exactly 1
    (an atom B24 randomises); a system neither the IP nor its class uses
    reads its backoff mass. Not randomised on purpose: a randomised PIT of a
    habitual system is uniform on [1 - p(s), 1], i.e. a slowly drifting
    near-0 score whose B24 ring then fits a degenerate GPD tail (every
    later excursion reads as astronomically rare)."""
    ps = float(pred.get(s, 0.0))
    p = sum(float(v) for v in pred.values() if float(v) <= ps)
    if p >= 1.0 - 1e-9:                      # the most probable system: exactly 1
        return 1.0
    return min(1.0, max(P_FLOOR, p))


def score_of(p: float, step: float = SCORE_STEP) -> float:
    """score = -log10 p floored to a `step`-decade grid. The null support of
    the score is discrete (a few systems per IP, slowly decaying counts);
    the grid makes its null values tie exactly, so B24's conformal ring and
    GPD tail (fitted only on >= 10 values strictly above q_0.90) see the
    atoms as atoms instead of a spuriously continuous, near-zero-width tail."""
    x = -math.log10(min(1.0, max(P_FLOOR, float(p))))
    return math.floor(x / step + 1e-9) * step


def spread_p(x_obs: int, x_days: Iterable[Tuple[float, float]], xbar_cls: float,
             kappa: float = SPREAD_KAPPA) -> Tuple[float, float, float]:
    """Lateral spread: P(X >= x_obs) for X = distinct systems in 24 h - 1
    under a Gamma-Poisson (NB) predictive with a class prior.

    x_days: (X_d, weight) of the IP's committed active days. Posterior on the
    rate: Gamma(a0 + kappa xbar_cls + sum w X, b0 + kappa + sum w); NB with
    size a and mean a / b. Returns (p, mean, size)."""
    sx = sw = 0.0
    for x, w in x_days:
        sx += float(w) * max(0.0, float(x))
        sw += float(w)
    xbar = max(0.0, float(xbar_cls)) if xbar_cls == xbar_cls else 0.0
    a = SPREAD_A0 + kappa * xbar + sx
    b = SPREAD_B0 + kappa + sw
    mean = a / b
    if x_obs <= 0:
        return 1.0, mean, a
    p = float(bayes.nb_sf(float(x_obs) - 1.0, mean, a))
    return (min(1.0, max(P_FLOOR, p)) if p == p else 1.0), mean, a


def combine(p_nov: float, p_spread: float) -> float:
    """Bonferroni over the two tests (novelty of the system, 24-h spread)."""
    return min(1.0, max(P_FLOOR, 2.0 * min(p_nov, p_spread)))


def adamic_adar(fps: Mapping[str, Mapping[str, float]], ip: str, home: Optional[str],
                s: str) -> float:
    """Support of the new bipartite edge (ip, s) from peers: sum over other
    IPs j that use both ip's home system and s of 1 / ln(deg j), deg j = the
    number of systems j uses. 0 = nobody bridges the two systems."""
    if home is None or home == s:
        return 0.0
    out = 0.0
    for j, fp in fps.items():
        if j == ip:
            continue
        used = [x for x, c in fp.items() if c >= ACTIVE_MIN]
        if home in used and s in used:
            out += 1.0 / math.log(max(2, len(used)))
    return out


def daily_extra(pair_states: Iterable[Mapping[str, Any]], before_day: int,
                n_days: int = SPREAD_DAYS) -> List[Tuple[float, float]]:
    """(X_d, 1) over the complete local days d in [before_day - n_days,
    before_day) on which the IP had >= 1 committed pair: X_d = pairs - 1."""
    k: Dict[int, int] = {}
    for st in pair_states:
        for d, w in (st.get("days") or {}).items():
            d = int(d)
            if before_day - n_days <= d < before_day and float(w) >= 0.5:
                k[d] = k.get(d, 0) + 1
    return [(float(v - 1), 1.0) for v in k.values() if v >= 1]
