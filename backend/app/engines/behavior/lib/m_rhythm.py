"""Read accessors for model.rhythm (owner: B07 RhythmEngine; contract C, G, L).

Why a slot clock, and why this module: presence is modelled on a fixed
15-minute LOCAL slot clock (lib/timebins: slot = floor(local_epoch_s / 900)),
never on ticks, so 60-s, 900-s and 3600-s cadences describe the same thing.
Identity (B15/B16/B17), peer typing (B02), the class monitor (B18) and the
portrait (B30) all need "how likely is this entity / class to be active in
this slot", so the maths lives here once, as pure functions over the model
dict. Consumers never mutate the model.

Cells. A bin (contract B: bin48 = hour + 24 * nonworkday, bin168 = dow * 24 +
hour) is split into its four 15-minute quarters, q = slot % 4:
    c48  = bin48  * 4 + q   in [0, 192)
    c168 = bin168 * 4 + q   in [0, 672)   (-1 on an IRREGULAR day)
Sub-hour cells are required: a 40-minute backup is active in 3 of the 4
slots of its hour, so a per-hour P(active) of 0.75 could never reach the
p >= 0.95 that silence needs (B07 test b). A day is irregular when its day
type differs from its weekday's usual type (holiday on a weekday, 调休 or a
self-healed workday on a weekend): such days use (and teach) only the
day-type cells, so a make-up Saturday is never scored against Saturdays.

Model layout, model.rhythm@(s, e | class:<rid> | __system__):
    {
      'fmt': 1, 'kind': 'entity' | 'class' | 'system',
      'version': int,            # regime version (entity: gate.version; pools: rev)
      'rev': int,                # bumped on every change (cache key)
      'updated': ts,
      'state': {                 # decayed sufficient statistics (half-life 28 d)
          'A48', 'N48', 'V48': float64[192],   # active weight, total weight,
                                               # volume on active slots
          'A168', 'N168': float64[672],
          't_ref': float,        # time the counts are decayed to
          't_first', 't_last': float,          # first / last committed slot
          'n_slots': int,        # committed slots (undecayed count)
          'n_commit': int,       # monotone update counter
          'open': list | None,   # engine-internal: slot waiting for its last tick
      },
      'prior': {'tier': 'class:<rid>' | 'system' | 'hyper',
                'pi48': float64[192] | None (None = HYPER_PI), 's': strength},
      'entropy168': float, 'machine_like': bool,       # refreshed hourly
      'automation': float, 'auto_A': float | None,     # index / B02 A used (entity)
      'desc': descriptors(model) as of the last refresh,
      'shifts': [schedule-shift records],
      entity only: 'gate' (lib/gating GateState), 'ledger' (per-slot activity,
                   9 d), 'det' (detector state) -- engine-internal;
      pools only:  'members': [ip], 'n_members': int;
      system only: 'healed': {day_index: ts} (calendar self-healing, B07 step 6)
    }
Class and system models hold POOLED member counts (sum of member statistics
decayed to 'updated') with the hyperprior, so p_cell on them is the expected
fraction of members active in a slot (the class rhythm B18 consumes).

Posterior predictive of a 48-level cell c (own counts A, N; tier prior mean
pi0 = clip(pi_tier, 0.02, 0.98), strength S = 6 class / 2 system / 1 hyper):
    neighbourhood (von Mises across hours, kappa = 4, |dh| <= 2, same quarter
    and day type, circular):  A_nb = sum w_dh A[c + dh h], N_nb likewise
    r_nb  = A_nb / N_nb                   (0 when N_nb = 0; then lam = 0 too)
    rho   = exp(-G/2), G/2 = two-sample Bernoulli log-likelihood ratio of
            (A, N) against (A_nb, N_nb)  (1 when either side is empty)
    lam   = rho * N_nb
    p48   = (A + lam r_nb + S pi0) / (N + lam + S)
The neighbours enter as lam extra observations at their own rate, so the
prior is counted once (a prior-smoothed neighbour rate would count it
twice and hold 20 quiet nights at p ~ 0.023 instead of ~ 0.012).
The neighbourhood is borrowed at full von Mises weight where the cell agrees
with it (sparse or uniformly quiet hours: 20 days of silent nights give
p ~ 0.02, B07 test a) and not at all across a real edge (the backup slots
stay at p ~ 0.97 next to silent hours, test b): plain kernel smoothing at
kappa = 4 (weight 0.87 at 1 h) would erase every sharp schedule.
    p168  = (A168 + 4 p48) / (N168 + 4)       used when the model spans
                                              >= 28 d and the day is regular.

Accessor signatures (pure; missing data gives the documented default):
    new_state() -> dict;  new_model(kind='entity') -> dict
    cell48(bin48, q) -> int;  cell168(bin168, q) -> int
    cells_of_tctx(tctx) -> (c48, c168)          tctx: feature.tctx / timebins.tctx dict
    day_info(day, calendar=None, healed=None) -> (nonwork, regular)
    slot_cells(slot, nonwork, regular) -> (c48, c168)
    cells_of_slot(slot, calendar=None, healed=None) -> (c48, c168)
    window_cells(t0, t1, tz, calendar=None, healed=None) -> [(slot, c48, c168)]
    healed_days(system_model) -> set[int]
    mature168(model) -> bool
    cell_stats(model, c48, c168=-1) -> (p, strength, n_own)
    p_cell(model, c48, c168=-1) -> float        P(slot active); NaN on a bad model
    p_expected(model, tctx) -> float            same for a time context
    class_fraction(class_model, tctx) -> float  expected active fraction (B18)
    profile48(model) -> float64[192];  profile168(model) -> float64[672]
    activity48(model) / activity168(model) -> MLE rates A/N (NaN unobserved)
    hourly48(model) -> float64[48]              expected active slots per bin (0..4)
    shape48(model) -> float64[48];  shape168(model) -> float64[168]   (sum 1; NaN if
                                                never active)
    entropy168(model) -> float
    automation_components(model) -> {offhours, week, regular}   rhythm evidence
    automation_index(model, a_b02=None) -> float       mean with B02's A (NaN: unknown)
    machine_decision(index, previous, regular) -> bool hysteresis 0.6 / 0.4, regular >= 0.5
    machine_like(model) -> bool                        the decision kept by B07
    expected_volume(model, c48) -> float        mean events of an active slot (NaN)
    loglik_terms(model, active_slots, tctx) -> float64[n]   nats per slot
    loglik(model, active_slots, tctx) -> float  sum of the finite terms (NaN if none)
    descriptors(model) -> dict                  mu_h, R, window80, active_window,
                                                wd_we_ratio, entropy168, ...
    det_p(p_hat) -> float                       p_hat clipped to [0.02, 0.98] (detectors)
    offhours_step(W, a, p_hat) -> float         Bernoulli CUSUM step (bits), B07 step 3
    silence_step(s, a, p_hat, machine) -> float silence accumulator step (nats), step 4
    silence_eligible(p_hat, machine) -> bool
    slot_history(model, since=-inf) -> [(slot, a, volume, c48, c168)]  per-slot
                                                activity kept by the engine (9 d; B29 replay)
    offhours_replay(model, neutral_slots=(), since=-inf, W0=0.0, current=None) -> dict
                                                the off-hours CUSUM re-run over the ledger with
                                                the given slots made silent (B29 counterfactual)
    detector_state(model) -> dict               {W_off, s_sil, alarm, shift_explained}
    data_counts(model, at_ts=None) -> dict      decayed copies of the count arrays
    decay_factor(t_from, t_to) -> float
    to_dict(model) -> JSON-safe dict;  from_dict(d) -> live model
"""
from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.special import xlogy

from . import seq as SQ
from . import timebins as TB
from .priors import RHYTHM_CLASS_STRENGTH
from .seq import RHYTHM_P0_MAX

MODEL = "model.rhythm"
FMT = 1

SLOT_S = TB.SLOT_S                 # 900
SLOTS_PER_DAY = 96
QUARTERS = 4
N48 = 48 * QUARTERS                # 192
N168 = 168 * QUARTERS              # 672
HALF_LIFE_S = 28 * 86400.0         # forgetting half-life
MATURE168_S = 28 * 86400.0         # bin168 after >= 4 weeks of history
S168 = 4.0                         # strength of the 48-level cell as prior of a 168 cell
HYPER_PI = 0.5                     # Jeffreys Beta(0.5, 0.5)
HYPER_S = 1.0
CLASS_S = float(RHYTHM_CLASS_STRENGTH)   # Beta(6 pi, 6 (1 - pi)) class prior
SYSTEM_S = 2.0                     # system tier: humans and machines mixed, weaker
PI_CLIP = (0.02, 0.98)
VM_KAPPA = 4.0
VM_MAX_DH = 2
P0_MAX = RHYTHM_P0_MAX             # offhours bins (0.3)
P_SIL_MIN = 0.95                   # silence bins
P_USUAL = 0.5                      # "usual window" / "new window" split (schedule shift)
ENTROPY_MAX = 0.8                  # v2 machine-like rule (entropy <= 0.8); superseded by
                                   # automation_index, kept for reference
N_OBS_MIN = 0.25                   # a cell with less decayed weight counts as unobserved
LL_CLIP = 1e-4                     # p clip inside loglik
ARL_DAYS = 100.0                   # offhours / silence in-control ARL (B07 spec)
ARL_SLOTS = ARL_DAYS * SLOTS_PER_DAY               # 9600 slots
H_OFF = SQ.bernoulli_threshold(ARL_SLOTS)          # 13.73 bits (Wald bound + 0.5)
H_SIL = math.log(ARL_SLOTS)                        # 9.17 nats: alarm at p_eq <= 1/ARL
SIL_P_FLOOR = 1e-9                 # 1 - p_hat floor inside -ln(1 - p_hat)

_EPOCH_ORD = _dt.date(1970, 1, 1).toordinal()
_EPOCH_DOW = 3                     # 1970-01-01 was a Thursday
_LN168 = math.log(168.0)


def _vm_offsets() -> Tuple[Tuple[int, float], ...]:
    """(dh, w) for the von Mises neighbours of an hour, dh != 0 (timebins weights)."""
    out = []
    for b, w in TB.von_mises_weights(0.5, VM_KAPPA, VM_MAX_DH).items():
        dh = ((b + 12) % 24) - 12
        if dh != 0:
            out.append((dh, float(w)))
    return tuple(sorted(out))


VM_OFFSETS = _vm_offsets()         # ((-2, 0.585), (-1, 0.873), (1, 0.873), (2, 0.585))


def _map168_48() -> np.ndarray:
    """168-cell -> 48-cell on a regular day (Mon-Fri workday, Sat-Sun nonworkday)."""
    c = np.arange(N168)
    dow, rem = np.divmod(c, 24 * QUARTERS)
    hour, q = np.divmod(rem, QUARTERS)
    return ((hour + 24 * (dow >= 5)) * QUARTERS + q).astype(np.int64)


MAP168_48 = _map168_48()
MAP168_48.flags.writeable = False


# ============================================================== construction
def new_state() -> Dict[str, Any]:
    """Empty learner state (all counts 0, clock unanchored)."""
    return {
        "A48": np.zeros(N48), "N48": np.zeros(N48), "V48": np.zeros(N48),
        "A168": np.zeros(N168), "N168": np.zeros(N168),
        "t_ref": math.nan, "t_first": math.nan, "t_last": math.nan,
        "n_slots": 0, "n_commit": 0, "open": None,
    }


def new_model(kind: str = "entity") -> Dict[str, Any]:
    if kind not in ("entity", "class", "system"):
        raise ValueError(f"model.rhythm kind must be entity|class|system, got {kind!r}")
    m: Dict[str, Any] = {
        "fmt": FMT, "kind": kind, "version": 0, "rev": 0, "updated": math.nan,
        "state": new_state(),
        "prior": {"tier": "hyper", "pi48": None, "s": HYPER_S},
        "entropy168": math.nan, "machine_like": False, "desc": {}, "shifts": [],
    }
    if kind != "entity":
        m["members"] = []
        m["n_members"] = 0
    if kind == "system":
        m["healed"] = {}
    return m


# ===================================================================== cells
def cell48(bin48: int, q: int) -> int:
    return int(bin48) * QUARTERS + int(q)


def cell168(bin168: int, q: int) -> int:
    return int(bin168) * QUARTERS + int(q)


def _usual_nonwork(dow: int) -> bool:
    return int(dow) >= 5


def cells_of_tctx(t: Mapping[str, Any]) -> Tuple[int, int]:
    """(c48, c168) of a time context (feature.tctx / timebins.tctx dict). c168
    is -1 on an irregular day (day_type differs from the weekday's usual type)."""
    q = int(t["slot"]) % QUARTERS
    c48 = cell48(t["bin48"], q)
    dow = t.get("dow")
    dtype = t.get("day_type")
    if dow is None or dtype is None:
        return c48, cell168(t["bin168"], q)
    regular = (dtype == "nonworkday") == _usual_nonwork(dow)
    return c48, (cell168(t["bin168"], q) if regular else -1)


def day_info(day: int, calendar: Any = None, healed: Optional[Iterable[int]] = None
             ) -> Tuple[bool, bool]:
    """(nonwork, regular) of local day index `day` (= slot // 96): the calendar
    (holidays / 调休) decides, a self-healed day is a workday."""
    day = int(day)
    dow = (day + _EPOCH_DOW) % 7
    if healed is not None and day in healed:
        nonwork = False
    else:
        nonwork = TB.day_type(_dt.date.fromordinal(_EPOCH_ORD + day), calendar) == "nonworkday"
    return nonwork, nonwork == _usual_nonwork(dow)


def slot_cells(slot: int, nonwork: bool, regular: bool) -> Tuple[int, int]:
    """(c48, c168) of a local slot index given its day's (nonwork, regular)."""
    day, rem = divmod(int(slot), SLOTS_PER_DAY)
    hour, q = divmod(rem, QUARTERS)
    dow = (day + _EPOCH_DOW) % 7
    c48 = (hour + (24 if nonwork else 0)) * QUARTERS + q
    return c48, (((dow * 24 + hour) * QUARTERS + q) if regular else -1)


def cells_of_slot(slot: int, calendar: Any = None,
                  healed: Optional[Iterable[int]] = None) -> Tuple[int, int]:
    nonwork, regular = day_info(int(slot) // SLOTS_PER_DAY, calendar, healed)
    return slot_cells(slot, nonwork, regular)


def window_cells(t0: float, t1: float, tz: str = TB.DEFAULT_TZ, calendar: Any = None,
                 healed: Optional[Iterable[int]] = None) -> List[Tuple[int, int, int]]:
    """[(slot, c48, c168)] for every local slot overlapping [t0, t1) (for
    consumers scoring an observed window with loglik)."""
    if not (math.isfinite(t0) and math.isfinite(t1)) or t1 <= t0:
        return []
    s0 = TB.slot_of(t0, tz)
    s1 = TB.slot_of(math.nextafter(t1, -math.inf), tz)
    out = []
    info: Dict[int, Tuple[bool, bool]] = {}
    for j in range(s0, s1 + 1):
        d = j // SLOTS_PER_DAY
        if d not in info:
            info[d] = day_info(d, calendar, healed)
        c48, c168 = slot_cells(j, *info[d])
        out.append((j, c48, c168))
    return out


def healed_days(system_model: Optional[Mapping[str, Any]]) -> Set[int]:
    if not system_model:
        return set()
    return {int(k) for k in (system_model.get("healed") or {})}


# ===================================================================== maths
def decay_factor(t_from: float, t_to: float) -> float:
    """2^(-(t_to - t_from)/H): weight at t_to of a count stored at t_from
    (> 1 when t_to < t_from). Unanchored clocks give 1."""
    if not (math.isfinite(t_from) and math.isfinite(t_to)):
        return 1.0
    return 2.0 ** (-(t_to - t_from) / HALF_LIFE_S)


def _xlogy(x: float, y: float) -> float:
    return x * math.log(y) if x > 0.0 else 0.0


def _lr_weight(x1: float, n1: float, x2: float, n2: float) -> float:
    """exp(-G/2): likelihood ratio of 'same Bernoulli rate' against 'separate
    rates' for (x1 of n1) and (x2 of n2) weighted counts; 1 if a side is empty."""
    if not (n1 > 0.0 and n2 > 0.0):
        return 1.0
    x1 = min(max(x1, 0.0), n1)
    x2 = min(max(x2, 0.0), n2)
    r1, r2 = x1 / n1, x2 / n2
    x, n = x1 + x2, n1 + n2
    r = x / n
    g = (_xlogy(x1, r1) + _xlogy(n1 - x1, 1.0 - r1) + _xlogy(x2, r2)
         + _xlogy(n2 - x2, 1.0 - r2) - _xlogy(x, r) - _xlogy(n - x, 1.0 - r))
    return math.exp(-g) if g > 0.0 else 1.0


def _lr_weight_vec(x1: np.ndarray, n1: np.ndarray, x2: np.ndarray, n2: np.ndarray) -> np.ndarray:
    ok = (n1 > 0.0) & (n2 > 0.0)
    n1s = np.where(ok, n1, 1.0)
    n2s = np.where(ok, n2, 1.0)
    x1 = np.clip(x1, 0.0, n1s)
    x2 = np.clip(x2, 0.0, n2s)
    r1, r2 = x1 / n1s, x2 / n2s
    x, n = x1 + x2, n1s + n2s
    r = x / n
    g = (xlogy(x1, r1) + xlogy(n1s - x1, 1.0 - r1) + xlogy(x2, r2) + xlogy(n2s - x2, 1.0 - r2)
         - xlogy(x, r) - xlogy(n - x, 1.0 - r))
    return np.where(ok & (g > 0.0), np.exp(-np.maximum(g, 0.0)), 1.0)


def _prior(model: Mapping[str, Any]) -> Tuple[Optional[np.ndarray], float]:
    pr = model.get("prior") or {}
    pi = pr.get("pi48")
    s = pr.get("s", HYPER_S)
    try:
        s = float(s)
    except (TypeError, ValueError):
        s = HYPER_S
    if not (math.isfinite(s) and s > 0.0):
        s = HYPER_S
    if pi is not None:
        pi = np.asarray(pi, dtype=np.float64)
        if pi.shape != (N48,):
            pi = None
    return pi, s


def mature168(model: Mapping[str, Any]) -> bool:
    st = model.get("state") or {}
    t0, t1 = st.get("t_first", math.nan), st.get("t_last", math.nan)
    return bool(math.isfinite(t0) and math.isfinite(t1) and t1 - t0 >= MATURE168_S)


def _p48(A: np.ndarray, N: np.ndarray, c: int, pi0: float, s: float) -> Tuple[float, float, float]:
    dtp, rem = divmod(c, 24 * QUARTERS)
    h, q = divmod(rem, QUARTERS)
    base = dtp * 24 * QUARTERS + q
    a_nb = n_nb = 0.0
    for dh, w in VM_OFFSETS:
        k = base + ((h + dh) % 24) * QUARTERS
        a_nb += w * float(A[k])
        n_nb += w * float(N[k])
    a, n = float(A[c]), float(N[c])
    lam = _lr_weight(a, n, a_nb, n_nb) * n_nb
    r_nb = a_nb / n_nb if n_nb > 0.0 else 0.0
    strength = n + lam + s
    return (a + lam * r_nb + s * pi0) / strength, strength, n


def cell_stats(model: Mapping[str, Any], c48: int, c168: int = -1) -> Tuple[float, float, float]:
    """(p, strength, n_own) of a cell: p = posterior predictive P(slot active)
    (the 168 cell when the model is mature and c168 >= 0), strength = own +
    borrowed + prior pseudo-observations of the 48 cell, n_own = its own
    decayed observations. Bad input gives (NaN, 0, 0)."""
    try:
        st = model["state"]
        A, N = st["A48"], st["N48"]
        c48 = int(c48)
        if not 0 <= c48 < N48:
            return math.nan, 0.0, 0.0
        pi, s = _prior(model)
        pi0 = HYPER_PI if pi is None else float(pi[c48])
        p, strength, n = _p48(A, N, c48, pi0, s)
        c168 = int(c168)
        if 0 <= c168 < N168 and mature168(model):
            p = (float(st["A168"][c168]) + S168 * p) / (float(st["N168"][c168]) + S168)
        return p, strength, n
    except (KeyError, TypeError, ValueError, IndexError):
        return math.nan, 0.0, 0.0


def p_cell(model: Mapping[str, Any], c48: int, c168: int = -1) -> float:
    return cell_stats(model, c48, c168)[0]


def p_expected(model: Mapping[str, Any], tctx: Mapping[str, Any]) -> float:
    """P(active) of the slot described by a time context."""
    c48, c168 = cells_of_tctx(tctx)
    return p_cell(model, c48, c168)


def class_fraction(class_model: Mapping[str, Any], tctx: Mapping[str, Any]) -> float:
    """Expected fraction of class members active in the slot (pooled model)."""
    return p_expected(class_model, tctx)


def _vm_matrix() -> np.ndarray:
    """K[c, c'] = von Mises weight of neighbour cell c' for cell c (same day
    type and quarter, |dh| <= 2 hours, circular): a_nb = K @ A."""
    K = np.zeros((N48, N48))
    for c in range(N48):
        dtp, rem = divmod(c, 24 * QUARTERS)
        h, q = divmod(rem, QUARTERS)
        for dh, w in VM_OFFSETS:
            K[c, dtp * 24 * QUARTERS + ((h + dh) % 24) * QUARTERS + q] = w
    return K


VM_MATRIX = _vm_matrix()
VM_MATRIX.flags.writeable = False


def _profile48_arrays(A: np.ndarray, N: np.ndarray, pi: Optional[np.ndarray],
                      s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised _p48 over all 192 cells -> (p, strength)."""
    a = np.asarray(A, dtype=np.float64)
    n = np.asarray(N, dtype=np.float64)
    a_nb = VM_MATRIX @ a
    n_nb = VM_MATRIX @ n
    pi0 = HYPER_PI if pi is None else np.asarray(pi, dtype=np.float64)
    lam = _lr_weight_vec(a, n, a_nb, n_nb) * n_nb
    r_nb = np.divide(a_nb, n_nb, out=np.zeros_like(a_nb), where=n_nb > 0.0)
    strength = n + lam + s
    return (a + lam * r_nb + s * pi0) / strength, strength


def profile48(model: Mapping[str, Any]) -> np.ndarray:
    """Posterior P(active) of every 48-level cell (float64[192])."""
    st = model["state"]
    pi, s = _prior(model)
    return _profile48_arrays(st["A48"], st["N48"], pi, s)[0]


def profile168(model: Mapping[str, Any]) -> np.ndarray:
    """Posterior P(active) of every 168-level cell on a regular day (float64[672]);
    before maturity (< 28 d) this is the mapped 48 profile."""
    st = model["state"]
    p48 = profile48(model)[MAP168_48]
    if not mature168(model):
        return p48
    return (np.asarray(st["A168"]) + S168 * p48) / (np.asarray(st["N168"]) + S168)


def prior_pi48(A: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Tier prior means from pooled counts: the hyperprior posterior of the
    pooled cells (same smoothing as profile48), clipped to PI_CLIP."""
    p, _ = _profile48_arrays(A, N, None, HYPER_S)
    return np.clip(p, PI_CLIP[0], PI_CLIP[1])


def activity48(model: Mapping[str, Any]) -> np.ndarray:
    """MLE active rate A/N per 48 cell; NaN where unobserved (N < 0.25)."""
    st = model["state"]
    A, N = np.asarray(st["A48"], dtype=np.float64), np.asarray(st["N48"], dtype=np.float64)
    ok = N >= N_OBS_MIN
    return np.where(ok, A / np.where(ok, N, 1.0), np.nan)


def activity168(model: Mapping[str, Any]) -> np.ndarray:
    """MLE active rate per 168 cell; unobserved 168 cells (or an immature model)
    fall back to the mapped 48 rate."""
    r48 = activity48(model)[MAP168_48]
    if not mature168(model):
        return r48
    st = model["state"]
    A, N = np.asarray(st["A168"], dtype=np.float64), np.asarray(st["N168"], dtype=np.float64)
    ok = N >= N_OBS_MIN
    return np.where(ok, A / np.where(ok, N, 1.0), r48)


def hourly48(model: Mapping[str, Any]) -> np.ndarray:
    """Expected active slots per bin48 hour (0..4), from the posterior."""
    return profile48(model).reshape(48, QUARTERS).sum(axis=1)


def _norm(m: np.ndarray) -> np.ndarray:
    t = float(np.nansum(m))
    if not t > 0.0:
        return np.full(m.shape, np.nan)
    return np.nan_to_num(m, nan=0.0) / t


def _ever_active(model: Mapping[str, Any]) -> bool:
    st = model.get("state") or {}
    a = st.get("A48")
    return a is not None and float(np.sum(a)) > 0.0


def shape48(model: Mapping[str, Any]) -> np.ndarray:
    """Normalised 48-bin rhythm shape (B02 role descriptor): expected active
    slots per bin48 hour over their total. NaN when nothing was ever active
    (the posterior would only echo the prior)."""
    if not _ever_active(model):
        return np.full(48, np.nan)
    return _norm(hourly48(model))


def shape168(model: Mapping[str, Any]) -> np.ndarray:
    """Normalised 168-bin (hour of week) shape from the posterior (NaN as shape48)."""
    if not _ever_active(model):
        return np.full(168, np.nan)
    return _norm(profile168(model).reshape(168, QUARTERS).sum(axis=1))


def entropy168(model: Mapping[str, Any]) -> float:
    """Normalised entropy of the observed activity over the 168 hour-of-week
    bins (MLE rates, unobserved cells carry no mass): 0 for one bin, 1 for
    uniform. NaN when nothing was ever active."""
    m = np.nan_to_num(activity168(model), nan=0.0).reshape(168, QUARTERS).sum(axis=1)
    t = float(m.sum())
    if not t > 0.0:
        return math.nan
    q = m[m > 0.0] / t
    return float(-(q * np.log(q)).sum() / _LN168)


# ------------------------------------------------------------ automation
BIZ_HOURS = (8, 19)                # local workday business hours [8, 19)
AUTO_ENTER, AUTO_LEAVE = 0.6, 0.4  # machine_like hysteresis (B02's super level)
AUTO_MIN_SPAN_S = 7 * 86400.0      # one week: both day types seen at least once
REGULAR_MIN = 0.5                  # machine_like needs mean 4 r (1 - r) <= 0.5 where active
_BIZ48 = np.zeros(N48, dtype=bool)
_BIZ48[BIZ_HOURS[0] * QUARTERS:BIZ_HOURS[1] * QUARTERS] = True      # workday half only
_BIZ48.flags.writeable = False


def automation_components(model: Mapping[str, Any]) -> Dict[str, float]:
    """Rhythm evidence of automation, each in [0, 1] (1 = machine), NaN when
    not identified (MLE activity rates, activity48):

      offhours  min(1, r_off / r_biz): the active rate outside workday
                business hours relative to inside. An office worker is ~0,
                a 24/7 client ~1, a nightly job (never active in business
                hours) 1.
      week      min(r_wd, r_nwd) / max(r_wd, r_nwd): non-workdays look like
                workdays. A person ~0, a scheduled job or 24/7 client ~1.
      regular   1 - sum_c r_c 4 r_c (1 - r_c) / sum_c r_c over the observed
                cells: presence is all-or-nothing where the entity is
                active (a schedule, a poller: r ~ 1) rather than a coin
                flip (a person's 55 % of daytime slots gives ~0.1). This
                is also what silence needs: cells with p_hat >= 0.95.

    Both need a model spanning AUTO_MIN_SPAN_S (a week holds both day types).
    Normalised entropy is deliberately not used: it is not monotone in
    automation (a 24/7 client is near 1, a nightly job ~0.5 and an office
    worker 0.7 - the old 'entropy <= 0.8' rule called most office workers
    machine-like and no 24/7 client)."""
    out = {"offhours": math.nan, "week": math.nan, "regular": math.nan}
    st = model.get("state") or {}
    t0, t1 = st.get("t_first", math.nan), st.get("t_last", math.nan)
    if not (math.isfinite(t0) and math.isfinite(t1) and t1 - t0 >= AUTO_MIN_SPAN_S):
        return out
    a = activity48(model)
    biz = a[_BIZ48]
    off = a[~_BIZ48]
    r_biz = float(np.nanmean(biz)) if np.isfinite(biz).any() else math.nan
    r_off = float(np.nanmean(off)) if np.isfinite(off).any() else math.nan
    if r_biz == r_biz and r_off == r_off and (r_biz > 0.0 or r_off > 0.0):
        out["offhours"] = 1.0 if r_biz <= 0.0 else min(1.0, r_off / r_biz)
    wd, nwd = a[:N48 // 2], a[N48 // 2:]
    r_wd = float(np.nanmean(wd)) if np.isfinite(wd).any() else math.nan
    r_nwd = float(np.nanmean(nwd)) if np.isfinite(nwd).any() else math.nan
    if r_wd == r_wd and r_nwd == r_nwd and max(r_wd, r_nwd) > 0.0:
        out["week"] = min(r_wd, r_nwd) / max(r_wd, r_nwd)
    r = a[np.isfinite(a)]
    m = float(r.sum())
    if m > 0.0:
        out["regular"] = 1.0 - float(np.sum(r * 4.0 * r * (1.0 - r))) / m
    return out


def automation_index(model: Mapping[str, Any], a_b02: Any = None) -> float:
    """Automation index in [0, 1]: the mean of the identified rhythm
    components (automation_components) and B02's per-IP automation index A
    (timing regularity, periodicity, non-browser share, think time, path
    entropy; m_class assignment 'A') when given. NaN when nothing is
    identified. machine_like applies hysteresis to it."""
    vals = [v for v in automation_components(model).values() if v == v]
    try:
        a = float(a_b02) if a_b02 is not None else math.nan
    except (TypeError, ValueError):
        a = math.nan
    if a == a:
        vals.append(min(1.0, max(0.0, a)))
    return float(np.mean(vals)) if vals else math.nan


def machine_decision(index: float, previous: bool = False, regular: float = math.nan) -> bool:
    """machine_like with hysteresis: enter at index >= AUTO_ENTER, leave at
    index < AUTO_LEAVE (in between, and on a NaN index, keep `previous`).
    A known `regular` (automation_components) below REGULAR_MIN is never
    machine-like: silence and a missed usual window only mean something for
    all-or-nothing presence, not for cells an entity fills like a coin flip
    (an irregular person active every day 07-23 has offhours ~0.6 and week
    ~1, but regular ~0.15)."""
    if regular == regular and regular < REGULAR_MIN:
        return False
    if not index == index:
        return bool(previous)
    if index >= AUTO_ENTER:
        return True
    if index < AUTO_LEAVE:
        return False
    return bool(previous)


def machine_like(model: Mapping[str, Any]) -> bool:
    """The automation decision B07 keeps in the model (model['machine_like'],
    refreshed hourly from automation_index with hysteresis); for a model
    without it, the rhythm-only index against AUTO_ENTER."""
    v = model.get("machine_like") if isinstance(model, Mapping) else None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return machine_decision(automation_index(model),
                            regular=automation_components(model)["regular"])


def expected_volume(model: Mapping[str, Any], c48: int) -> float:
    """Mean volume (act.events) of an active slot in the cell; NaN if unknown."""
    st = model["state"]
    a = float(st["A48"][int(c48)])
    return float(st["V48"][int(c48)]) / a if a > 0.05 else math.nan


def _cells_of(t: Any) -> Tuple[int, int]:
    if isinstance(t, Mapping):
        return cells_of_tctx(t)
    c48, c168 = t
    return int(c48), int(c168)


def loglik_terms(model: Mapping[str, Any], active_slots: Sequence[float],
                 tctx: Sequence[Any]) -> np.ndarray:
    """Per-slot Bernoulli log-likelihood (nats) a ln p + (1 - a) ln(1 - p) of an
    observed window under the model. active_slots: activity in [0, 1] per slot
    (NaN = unobserved -> NaN term). tctx: per-slot time contexts (dicts with
    slot, bin48, bin168, dow, day_type) or (c48, c168) pairs, same length."""
    if len(active_slots) != len(tctx):
        raise ValueError("loglik: active_slots and tctx differ in length")
    out = np.full(len(active_slots), np.nan)
    for i, (a, t) in enumerate(zip(active_slots, tctx)):
        a = float(a)
        if not math.isfinite(a):
            continue
        p = p_cell(model, *_cells_of(t))
        if not math.isfinite(p):
            continue
        p = min(1.0 - LL_CLIP, max(LL_CLIP, p))
        a = min(1.0, max(0.0, a))
        out[i] = a * math.log(p) + (1.0 - a) * math.log1p(-p)
    return out


def loglik(model: Mapping[str, Any], active_slots: Sequence[float], tctx: Sequence[Any]) -> float:
    """Total log-likelihood (nats) of an observed slot window (identity
    modality 'rhythm', B15-B17); NaN when no slot is scorable."""
    t = loglik_terms(model, active_slots, tctx)
    ok = np.isfinite(t)
    return float(t[ok].sum()) if ok.any() else math.nan


# =============================================================== descriptors
def _hhmm(slot_of_day: int) -> str:
    s = int(slot_of_day) % SLOTS_PER_DAY
    return f"{s // 4:02d}:{(s % 4) * 15:02d}"


def _circ_mean(m: np.ndarray) -> Tuple[float, float]:
    """(mean position in hours, resultant length R) of a 96-slot day profile."""
    t = float(m.sum())
    if not t > 0.0:
        return math.nan, math.nan
    th = 2.0 * math.pi * (np.arange(SLOTS_PER_DAY) + 0.5) / SLOTS_PER_DAY
    c, s = float((m * np.cos(th)).sum()) / t, float((m * np.sin(th)).sum()) / t
    mu = (math.atan2(s, c) % (2.0 * math.pi)) * 24.0 / (2.0 * math.pi)
    return mu, math.hypot(c, s)


def _shortest_window(m: np.ndarray, frac: float, mu_h: float) -> Optional[Tuple[int, int]]:
    """(start slot, length) of the shortest circular window holding >= frac of
    the mass; ties go to the window centred nearest the circular mean."""
    t = float(m.sum())
    if not t > 0.0:
        return None
    n = m.size
    c2 = np.concatenate([[0.0], np.cumsum(np.concatenate([m, m]))])
    target = c2[:n] + frac * t * (1.0 - 1e-12)
    ln = np.searchsorted(c2, target, side="left") - np.arange(n)
    best = int(ln.min())
    starts = np.flatnonzero(ln == best)
    if starts.size > 1 and math.isfinite(mu_h):
        centre_h = ((starts + best / 2.0) % n) * 24.0 / n
        d = np.array([abs(TB.hour_distance(h, mu_h)) for h in centre_h])
        return int(starts[int(np.argmin(d))]), best
    return int(starts[0]), best


def _active_window(p: np.ndarray) -> Optional[Tuple[int, int]]:
    """Longest circular run of cells with p >= P_USUAL: (start slot, length)."""
    on = p >= P_USUAL
    n = on.size
    if not on.any():
        return None
    if on.all():
        return 0, n
    k = int(np.flatnonzero(~on)[0])            # rotate so the run cannot wrap
    r = np.roll(on, -k)
    best, best_s, cur, cur_s = 0, 0, 0, 0
    for i, v in enumerate(r.tolist()):
        if v:
            if cur == 0:
                cur_s = i
            cur += 1
            if cur > best:
                best, best_s = cur, cur_s
        else:
            cur = 0
    return (best_s + k) % n, best


def _f(x: float, nd: int = 4) -> Optional[float]:
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def descriptors(model: Mapping[str, Any]) -> Dict[str, Any]:
    """Circadian descriptors (JSON-safe) for profile.extra.rhythm / portraits:
    mu_h and R (circular mean and concentration of workday activity), window80
    (shortest daily window with 80 % of the activity), active_window (longest
    run of slots with P(active) >= 0.5, e.g. '09:00-18:00'), active slots per
    workday / nonworkday and their ratio, entropy168, machine_like, maturity."""
    st = model["state"]
    act = np.nan_to_num(activity48(model), nan=0.0)
    wd, nwd = act[:SLOTS_PER_DAY], act[SLOTS_PER_DAY:]
    mass_wd, mass_nwd = float(wd.sum()), float(nwd.sum())
    day = wd if mass_wd > 0.0 else nwd
    mu, R = _circ_mean(day)
    w80 = _shortest_window(day, 0.8, mu)
    p48 = profile48(model)
    pday = p48[:SLOTS_PER_DAY] if mass_wd > 0.0 or mass_nwd <= 0.0 else p48[SLOTS_PER_DAY:]
    aw = _active_window(pday) if (mass_wd > 0.0 or mass_nwd > 0.0) else None
    h = entropy168(model)
    out: Dict[str, Any] = {
        "day_type": "workday" if mass_wd > 0.0 or mass_nwd <= 0.0 else "nonworkday",
        "mu_h": _f(mu, 3), "R": _f(R),
        "window80": None, "active_window": None,
        "active_slots_wd": _f(mass_wd, 3), "active_slots_nwd": _f(mass_nwd, 3),
        "wd_we_ratio": _f(mass_wd / mass_nwd, 3) if mass_nwd > 0.0 else None,
        "entropy168": _f(h), "machine_like": machine_like(model),
        "automation": _f(automation_index(model, model.get("auto_A") if isinstance(
            model, Mapping) else None), 3),
        "mature168": mature168(model), "n_slots": int(st.get("n_slots", 0) or 0),
        "span_days": _f((st.get("t_last", math.nan) - st.get("t_first", math.nan)) / 86400.0, 2),
        "tier": (model.get("prior") or {}).get("tier", "hyper"),
    }
    if w80 is not None:
        s0, ln = w80
        out["window80"] = {"start": _hhmm(s0), "end": _hhmm(s0 + ln), "start_h": s0 / 4.0,
                           "len_h": ln / 4.0}
    if aw is not None:
        s0, ln = aw
        out["active_window"] = {"start": _hhmm(s0), "end": _hhmm(s0 + ln), "start_h": s0 / 4.0,
                                "len_h": ln / 4.0}
    return out


# ============================================================= detector maths
def det_p(p_hat: float) -> float:
    """p_hat as the detectors use it: clipped to PI_CLIP = [0.02, 0.98], so one
    slot is never worth more than 4.64 bits (off-hours) or 3.9 nats (silence)
    and both alarms need >= 3 slots of evidence whatever the history length
    (B07: the 3rd active slot alarms, 3 x 4.64 = 13.9 >= 13.7)."""
    p = float(p_hat)
    return min(PI_CLIP[1], max(PI_CLIP[0], p)) if p == p else math.nan


def offhours_step(W: float, a: float, p_hat: float) -> float:
    """One slot of the off-hours Bernoulli CUSUM (bits). Only bins with
    p_hat <= 0.3 contribute (p1 = min(0.95, max(0.5, 5 p_hat))); a bin with
    p_hat > 0.3, a NaN p_hat or an unobserved slot (a NaN) leaves W unchanged,
    so p1 > 1 can never occur. p_hat is clipped by det_p: at p_hat <= 0.02 an
    active slot adds 4.64 bits and a silent one -0.97."""
    p0 = det_p(p_hat)
    p1 = SQ.rhythm_p1(p0)                      # NaN above RHYTHM_P0_MAX
    return SQ.bernoulli_cusum_step(W, a, p0, p1)


def silence_eligible(p_hat: float, machine: bool) -> bool:
    """Silence is scored only for machine-like rhythms (normalised 168-bin
    entropy <= 0.8) and only in bins with p_hat >= 0.95."""
    return bool(machine) and p_hat == p_hat and p_hat >= P_SIL_MIN


def silence_step(s: float, a: float, p_hat: float, machine: bool) -> float:
    """One slot of the silence accumulator (nats): an eligible silent slot adds
    -ln(1 - p_hat); an eligible active slot (the scheduled activity happened)
    resets to 0; ineligible or unobserved slots leave s unchanged.
    p_eq = exp(-s); alarm at s >= H_SIL."""
    s = 0.0 if not s == s else float(s)
    a = float(a)
    if not a == a or not silence_eligible(p_hat, machine):
        return s
    if a > 0.5:
        return 0.0
    return s - math.log(max(1.0 - det_p(p_hat), SIL_P_FLOOR))


def slot_history(model: Mapping[str, Any], since: float = -math.inf
                 ) -> List[Tuple[int, float, float, int, int]]:
    """[(slot, a, volume, c48, c168)] of the finalised slots the engine keeps
    (9 d) whose end is >= since, oldest first. a is 1 / 0 / NaN (unobserved)."""
    led = (model or {}).get("ledger") or {}
    out = []
    for j, r in (led.get("slots") or {}).items():
        if r[5] >= since:
            out.append((int(j), float(r[1]), float(r[2]), int(r[3]), int(r[4])))
    out.sort()
    return out


def offhours_replay(model: Mapping[str, Any], neutral_slots: Iterable[int] = (),
                    since: float = -math.inf, W0: float = 0.0,
                    current: Optional[Tuple[int, int, int, float]] = None,
                    until: float = math.inf) -> Dict[str, Any]:
    """Re-run B07's off-hours Bernoulli CUSUM (offhours_step, p_hat from
    p_cell of the model as it is now) over the ledger slots ending in
    [since, until], starting from W0, with the activity of `neutral_slots` set to 0 (a silent
    slot: the counterfactual "this activity did not happen"; unobserved slots
    stay unobserved). current = (slot, c48, c168, a) adds B07's provisional
    step for the open slot when it is active and not neutralised.

    Returns {'W': reported W (provisional step included), 'W_final': W after
    the last finalised slot, 'W_max': max over the path, 'n': slots replayed,
    'active_unusual': [slots with a = 1 at p_hat <= 0.3]} (the candidates B29
    neutralises). Schedule-shift resets are not replayed; the caller compares
    the factual replay with detector_state for fidelity. Pure."""
    neutral = {int(j) for j in neutral_slots}
    W = float(W0) if W0 == W0 else 0.0
    w_max = W
    n = 0
    unusual: List[int] = []
    led = ((model or {}).get("ledger") or {}).get("slots") or {}
    for j, a, _vol, c48, c168 in slot_history(model, since):
        r = led.get(j)
        if r is not None and float(r[5]) > until:
            break
        p = p_cell(model, c48, c168)
        if a == 1.0 and p == p and p <= SQ.RHYTHM_P0_MAX:
            unusual.append(int(j))
        if int(j) in neutral and a == a:
            a = 0.0
        W = offhours_step(W, a, p)
        w_max = max(w_max, W)
        n += 1
    w_fin = W
    if current is not None:
        j, c48, c168, a = current
        p = p_cell(model, int(c48), int(c168))
        if a == 1.0 and int(j) not in neutral:
            if p == p and p <= SQ.RHYTHM_P0_MAX:
                unusual.append(int(j))
            W = offhours_step(W, 1.0, p)
            w_max = max(w_max, W)
    return {"W": W, "W_final": w_fin, "W_max": w_max, "n": n, "active_unusual": unusual}


def detector_state(model: Mapping[str, Any]) -> Dict[str, Any]:
    """Current detector values of an entity model (NaN when absent)."""
    det = (model or {}).get("det") or {}
    return {"W_off": float(det.get("W", math.nan)), "s_sil": float(det.get("s", math.nan)),
            "alarm": dict(det.get("alarm") or {}),
            "shift_explained": bool(det.get("explained_until", -math.inf)
                                    > float((model or {}).get("updated", math.nan) or -math.inf))}


# ============================================================ counts / pooling
def data_counts(model: Mapping[str, Any], at_ts: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Copies of the count arrays decayed to at_ts (default: as stored)."""
    st = model["state"]
    g = 1.0 if at_ts is None else decay_factor(st.get("t_ref", math.nan), float(at_ts))
    return {k: np.asarray(st[k], dtype=np.float64) * g for k in ("A48", "N48", "V48", "A168", "N168")}


# ============================================================= serialisation
_ARRAYS = ("A48", "N48", "V48", "A168", "N168")


def _enc(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return [None if not math.isfinite(v) else v for v in x.astype(np.float64).tolist()] \
            if x.dtype.kind == "f" else x.tolist()
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, dict):
        return {str(k): _enc(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_enc(v) for v in x]
    if hasattr(x, "to_dict"):
        return _enc(x.to_dict())
    return x


def to_dict(model: Mapping[str, Any]) -> Dict[str, Any]:
    """JSON-safe deep copy (arrays -> lists, NaN -> None, GateState -> dict)."""
    return _enc(dict(model))


def _dec_f(v: Any) -> float:
    return math.nan if v is None else float(v)


def from_dict(d: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Inverse of to_dict (a live model passes through). The engine-internal
    'gate', 'ledger' and 'det' parts are restored by the engine."""
    if not d:
        return new_model()
    st = d.get("state") or {}
    if isinstance(st.get("A48"), np.ndarray):
        return dict(d)
    m = new_model(d.get("kind", "entity"))
    for k, v in d.items():
        if k not in ("state", "prior"):
            m[k] = v
    s = new_state()
    for k in _ARRAYS:
        if st.get(k) is not None:
            s[k] = np.array([_dec_f(v) for v in st[k]], dtype=np.float64)
    for k in ("t_ref", "t_first", "t_last"):
        s[k] = _dec_f(st.get(k))
    s["n_slots"] = int(st.get("n_slots") or 0)
    s["n_commit"] = int(st.get("n_commit") or 0)
    s["open"] = st.get("open")
    m["state"] = s
    pr = dict(d.get("prior") or {})
    if pr.get("pi48") is not None:
        pr["pi48"] = np.array([_dec_f(v) for v in pr["pi48"]], dtype=np.float64)
    m["prior"] = {"tier": pr.get("tier", "hyper"), "pi48": pr.get("pi48"),
                  "s": float(pr.get("s", HYPER_S) or HYPER_S), **{k: v for k, v in pr.items()
                                                                  if k not in ("tier", "pi48", "s")}}
    if isinstance(m.get("entropy168"), type(None)):
        m["entropy168"] = math.nan
    if m.get("kind") == "system":
        m["healed"] = {int(k): v for k, v in (m.get("healed") or {}).items()}
    return m
