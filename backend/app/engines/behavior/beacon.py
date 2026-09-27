"""BeaconEngine (B12): C2 beaconing per (entity, rare destination).

Why: an implant calling home is the one behaviour that is both rare in the
population and regular in time. v1 looked for it with binned counts and a
Rayleigh test at a single period, which misses the two things real implants
do: jitter (+-30 % renewal jitter destroys phase coherence; measured Rayleigh
median p = 0.14 at n = 40) and long periods (one call every few hours never
shows up in a tick-level count). This engine works on POINT events instead:

  * Input. act.rare_events (R2): every event of the entity to a destination
    used by <= 20 % of the system's entities, with sub-second times and bytes
    (lib/m_template). Per (entity, destination) the newest 256 events of the
    last 7 d are buffered in model.beacon (act.rare_events is kept 1 h).
    Buffers are dropped for destinations that became common (prevalence >
    20 %) or that are class-shared (at least half, and >= 2, of the other
    members of the entity's role class hold a buffer for it this tick).
  * When. A pair is evaluated once it holds >= 12 events, only when new
    events arrived since its last evaluation, and at most once per 4 ticks
    (4 * ctx.window_s of wall clock, so the throttle means the same at
    60 s and 900 s). At most MAX_EVAL_PER_TICK pairs per tick, stalest first.
  * Tests (lib/evt):
      1) renewal regularity (primary): Gamma shape MLE on the intervals and
         the one-sided LR against kappa = 1 (Poisson), whose p comes from the
         EXACT finite-n null table (beacon_null_p; the chi2_1 asymptotic is
         2-5x anti-conservative at n <= 40). DEVIATION: the p used is taken
         at the least favourable point of the composite null kappa <= 1.5
         (saddlepoint r*, exact to 0.003 decades against the table at
         kappa = 1), because log-normal human gaps (kappa ~ 1.1) reject
         kappa = 1 ever more often as the 256-event buffer fills (see
         renewal_p); the table p is reported and used as a floor;
      2) size constancy: the Gamma shape of the event byte sizes, ranked
         against the same statistic of every other rare-destination pair of
         the system (the destination class's size dispersion), randomised
         at ties -- see DEVIATION below;
      3) strict period: the window-corrected Z^2_2 scan over trial periods
         oversampled at 1/(5T), band [max(10 s, med/2), min(T/4, 2 med)]
         (med = median interval; a 7-d span cannot be scanned at 5x down to
         10 s within 512 trials), with the Davies + union trials bound (the
         peak period is refined on a fine local grid for P_hat only). At
         most Z2_MAX_PER_TICK scans per tick, and none when the renewal p
         already alarms.
      p = min(3 min(p_renewal, p_size, p_Z2), 1) over the finite ones
      (Bonferroni; a skipped test only makes it more conservative).
  * Outputs. score.beacon = -log10 min(1, k p_min) over the k pairs
    evaluated this tick (a Bonferroni over pairs; also behavior.pm), 0 on an
    active tick with nothing to evaluate (absence of evidence is data), no
    row for a silent entity, NaN + behavior.degraded when R2 failed. Axis c2.
    Pairs whitelisted by the population (class-shared at the learned period,
    see below) or allowlisted by analysts (lib/m_feedback, dim 'dest') are
    not scored. acc_alarm = 1 and a 'beacon' event (once per pair per 24 h)
    when p < 1e-5, the destination is used by <= 2 entities (decayed R2
    count or live buffers, whichever is larger) and no other member of the
    class has LEARNED a beacon to the destination at P_hat +- 10 %.
    RITA-style dispersion values (interval MAD / median, Bowley skew, size
    MAD / median) are descriptors in profile.extra.beacons only.
  * Learning (contract H). The learned state is the set of established
    periodic pairs {dest: (P_hat, weight, last_ts)}: a row per tick holds the
    pairs evaluated with p <= 1e-3, folded through lib/gating.GatedLearner
    (delayed by D, weighted by behavior.trust, held under quarantine,
    checkpointed, model.control honoured, link seeding). The class-sharing
    whitelist reads the OTHER members' committed state, so a beacon that
    appears on several members at once is not whitelisted until the governor
    has trusted it. The event buffers themselves are observation windows,
    not learned state. Clock: behavior.score, which this engine writes on
    every tick it produces a row.
  * ctx.training learns and scores but raises no alarm and no event.

DEVIATION (size test): byte sizes of benign traffic to one destination are
often exactly constant (the same API call, the same 304), so a parametric
Gamma-shape LRT against any continuous null would call every such stream a
beacon at p ~ 1e-300 and dominate the Bonferroni minimum. The statistic is
the Gamma shape MLE as specified, but its p is the randomised rank among
the system's other evaluated pairs, which is valid under exchangeability
and floored at 1/(N + 1): size constancy supports, it never alarms alone.
Fewer than SIZE_MIN_POP reference pairs give NaN (the test is dropped).
Prevalence comes from R2's decayed destination counts (m_template) and this
engine's own buffers, through `_Prevalence`, because model.vocab (B08)
has no prevalence accessor yet; switch that one function when it does.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import special

from ...core.engine import Context, Engine
from ...models.schema import BehaviorEvent, EntityProfile, Severity
from .lib import combine
from .lib import emit
from .lib import evt
from .lib import gating as G
from .lib import m_class
from .lib import m_feedback
from .lib import m_template as MT
from .lib import m_vocab as MV

MODEL = "model.beacon"
DETECTOR = "beacon"
AXES = ["c2"]
EVENT_KIND = "beacon"
LEARNER = "beacon"
R2_ENGINE = "raw.action_token"
ALLOW_DIM = "dest"                 # m_feedback allowlist dimension of a beacon event

BUF_MAX = 256                      # events per (entity, dest)
BUF_AGE_S = 7 * 86400.0
MAX_PAIRS = 128                    # destinations buffered per entity (most recent kept)
MIN_EVENTS = 12
MIN_INTERVALS = 8                  # smallest row of the exact null table
RENEWAL_KAPPA0 = 1.5               # composite renewal null: kappa <= 1.5 (see renewal_p)
EVAL_EVERY_TICKS = 4
MAX_EVAL_PER_TICK = 32             # ~0.1 ms each (<= 2 ms/tick with the Z^2 scans)
Z2_MAX_PER_TICK = 2                # ~0.5-0.8 ms each at 256 events x 512 trials
Z2_P_MIN_S = 10.0
Z2_OVERSAMPLE = 5
Z2_HARMONICS = 2
Z2_REFINE = 21                     # fine points between the peak's grid neighbours
N_TESTS = 3                        # Bonferroni factor (renewal, size, Z^2)
RARE_SHARE = MT.RARE_SHARE         # buffer only destinations used by <= 20 % of entities
CLASS_SHARE_MIN = 2                # class-shared: >= 2 other members ...
CLASS_SHARE_FRAC = 0.5             # ... and >= half of them hold a buffer
P_ALERT = 1e-5
PREV_MAX_ENTITIES = 2.0
PERIOD_TOL = 0.10                  # class shares (dest, P_hat +- 10 %)
KNOWN_P = 1e-3                     # evaluated pairs at or below this are learned as periodic
KNOWN_HALF_LIFE_S = 7 * 86400.0
KNOWN_KEEP_S = 14 * 86400.0
SIZE_MIN_POP = 20
EVENT_COOLDOWN_S = 86400.0
ROW_KEEP_S = G.JOURNAL_MAX_AGE_S + 3600.0
ROW_CAP = 16384
PROFILE_TOP = 16
P_FLOOR = 1e-300
CLOCK = emit.SCORE                 # candidate-row clock for the gated learner

_NAN = math.nan
_EMPTY = np.zeros(0, dtype=np.float64)


# ================================================================ pair tests
def _log_ratio_stat(x: np.ndarray) -> float:
    """s = ln(mean x) - mean(ln x) >= 0 without cancellation (lib/evt form)."""
    m = float(x.mean())
    d = x / m - 1.0
    return float(np.mean(d - np.log1p(d)))


def _gamma_lower_p(s_obs: float, n: int, k0: float) -> float:
    """P(s <= s_obs) for n i.i.d. Gamma(k0) intervals (any scale), by the
    Barndorff-Nielsen r* saddlepoint.

    T = n s has the exact CGF (x / sum x is Dirichlet(k0, ..., k0))
        K(u) = -n u ln n + lnG(n k0) - lnG(n z) + n [lnG(z) - lnG(k0)],  z = k0 - u,
    so K'(u) = n [psi(n z) - psi(z) - ln n] and K'' = n psi1(z) - n^2 psi1(n z).
    The lower tail has u < 0 (z > k0); K'(u) = t is solved for ln z by a
    safeguarded Newton. With w = -sqrt(2 (u t - K(u))), v = u sqrt(K''):
        p = Phi(r*),  r* = w + ln(v / w) / w.
    At k0 = 1 this reproduces the exact table of lib/evt (BEACON_NULL_LR) to
    within 0.003 decades from 1e-3 to 1e-9 for 11..255 intervals (tested),
    and matches 2e5 Monte-Carlo draws at k0 = 2. s_obs at or above the null
    mean gives 1 (one-sided); s_obs is floored at 1e-9 (equal intervals),
    which only makes p larger."""
    if not (n >= 2 and s_obs == s_obs):
        return _NAN
    t = n * max(s_obs, 1e-9)
    ln_n = math.log(n)
    if not t < n * (float(special.psi(n * k0)) - float(special.psi(k0)) - ln_n):
        return 1.0

    def h(y: float) -> float:
        z = math.exp(y)
        return n * (float(special.psi(n * z)) - float(special.psi(z)) - ln_n) - t

    lo, hi = math.log(k0), math.log(max(2.0 * k0, (n - 1) / (2.0 * t)))
    while h(hi) > 0.0:                                 # h decreases in y = ln z
        lo, hi = hi, hi + 1.0
    y = hi
    for _ in range(64):
        f = h(y)
        if f > 0.0:
            lo = y
        else:
            hi = y
        if abs(f) <= 1e-12 * t or hi - lo < 1e-13:
            break
        z = math.exp(y)
        d = n * z * (n * float(special.zeta(2.0, n * z)) - float(special.zeta(2.0, z)))
        y_new = y - f / d if d < 0.0 else 0.5 * (lo + hi)
        y = y_new if lo < y_new < hi else 0.5 * (lo + hi)
    z = math.exp(y)
    u = k0 - z
    K = (-n * u * ln_n + math.lgamma(n * k0) - math.lgamma(n * z)
         + n * (math.lgamma(z) - math.lgamma(k0)))
    K2 = n * float(special.zeta(2.0, z)) - n * n * float(special.zeta(2.0, n * z))
    e = u * t - K
    if not (e > 1e-12 and K2 > 0.0):
        return 0.5
    w = -math.sqrt(2.0 * e)
    v = u * math.sqrt(K2)
    r = w + math.log(v / w) / w
    return max(P_FLOOR, min(1.0, math.exp(float(special.log_ndtr(r)))))


def renewal_p(intervals: np.ndarray) -> Tuple[float, float, float, float]:
    """(p, kappa_hat, LR, p_poisson) of the renewal-regularity test.

    kappa_hat and LR are lib/evt's Gamma MLE and one-sided LR against
    kappa = 1; p_poisson = evt.beacon_null_p (the exact finite-n null).
    DEVIATION: the p used is P(s <= s_obs) under the least favourable point
    kappa0 = RENEWAL_KAPPA0 of the composite null kappa <= 1.5 (saddlepoint,
    `_gamma_lower_p`), max'ed with p_poisson. Human gaps are not exponential:
    a log-normal with sigma = 1 has kappa ~ 1.13, and against kappa = 1 its
    p < 1e-4 rate grows with n (0.25 % at 40 intervals, 2.6 % at 255, 0.3 %
    below the 1e-5 alarm level: measured), so a 256-event buffer of a busy
    human would alarm. Against kappa0 = 1.5 it is 0 in 2000 streams at every
    n, while +-30 % jitter (kappa ~ 33) keeps median p = 2e-10 at 20 events."""
    iv = intervals[np.isfinite(intervals)]
    n = int(iv.size)
    kappa, lr = evt.gamma_renewal_lrt(iv)
    if n < MIN_INTERVALS or lr != lr:
        return _NAN, kappa, lr, _NAN
    p_poi = evt.beacon_null_p(lr, n)
    s_obs = _log_ratio_stat(np.maximum(iv, evt.MIN_INTERVAL_S))
    p = _gamma_lower_p(s_obs, n, RENEWAL_KAPPA0)
    if p_poi == p_poi:
        p = max(p, p_poi)
    return p, kappa, lr, p_poi


def _rank_p(k: float, pop: np.ndarray, u: float, own: float = _NAN) -> float:
    """Randomised upper-tail rank p of the size shape k among the population
    (sorted ascending), excluding the pair's own previous value `own`:
    (#{> k} + u (#{== k} + 1)) / (N + 1). NaN below SIZE_MIN_POP references."""
    if not math.isfinite(k):
        return _NAN
    n = int(pop.size)
    lo = int(np.searchsorted(pop, k, side="left"))
    hi = int(np.searchsorted(pop, k, side="right"))
    gt, eq = n - hi, hi - lo
    if math.isfinite(own):
        n -= 1
        if own > k:
            gt -= 1
        elif own == k:
            eq -= 1
    if n < SIZE_MIN_POP:
        return _NAN
    return min(1.0, (gt + u * (eq + 1)) / (n + 1))


def _z2_test(t: np.ndarray, med: float) -> Tuple[float, float]:
    """(p, best period) of the Z^2_2 scan around the median interval."""
    span = float(t[-1] - t[0]) if t.size else 0.0
    if not (span > 0.0 and med > 0.0):
        return _NAN, _NAN
    p_min = max(Z2_P_MIN_S, 0.5 * med)
    p_max = min(0.25 * span, 2.0 * med)
    if not p_max > p_min:
        return _NAN, _NAN
    periods, n_eff = evt.z2_trial_periods(span, p_min, p_max, Z2_OVERSAMPLE)
    if periods.size == 0:
        return _NAN, _NAN
    z = evt.z2_periodogram(t, periods, Z2_HARMONICS)
    if not np.isfinite(z).any():
        return _NAN, _NAN
    j = int(np.nanargmax(z))
    p = evt.z2_p(float(z[j]), Z2_HARMONICS, n_eff, n_trials=int(periods.size))
    return p, _refine_peak(t, periods, j, float(z[j]))


def _refine_peak(t: np.ndarray, periods: np.ndarray, j: int, z_j: float) -> float:
    """Best period on a Z2_REFINE-point frequency grid between the grid
    neighbours of peak j. The scan step 1/(5T) is ~P^2/(5T) in period (12 s
    at P = 1 h over 2.5 d), coarser than a strict train's own precision.
    Only the period estimate uses it; p stays the scanned grid's (the union
    bound counts that grid's trials)."""
    f = 1.0 / periods
    lo, hi = f[max(j - 1, 0)], f[min(j + 1, f.size - 1)]
    if not hi > lo:
        lo, hi = hi, lo
    if not hi > lo:
        return float(periods[j])
    fine = np.linspace(lo, hi, Z2_REFINE)
    zf = evt.z2_periodogram(t, 1.0 / fine, Z2_HARMONICS)
    if not np.isfinite(zf).any() or not float(np.nanmax(zf)) > z_j:
        return float(periods[j])
    return float(1.0 / fine[int(np.nanargmax(zf))])


def _q(xs: np.ndarray, q: float) -> float:
    """Linear-interpolated quantile of an ascending array (np.quantile is ~30 us)."""
    h = (xs.size - 1) * q
    i = int(h)
    return float(xs[i]) if i + 1 >= xs.size else float(xs[i] + (h - i) * (xs[i + 1] - xs[i]))


def _dispersion(x: np.ndarray) -> Tuple[float, float, float]:
    """(median, MAD / median, Bowley skew) of a sample (RITA-style descriptors)."""
    if x.size < 2:
        return (float(x[0]) if x.size else _NAN), _NAN, _NAN
    xs = np.sort(x)
    m = _q(xs, 0.5)
    mad = float(np.median(np.abs(xs - m)))
    q1, q3 = _q(xs, 0.25), _q(xs, 0.75)
    d = q3 - q1
    skew = (q3 + q1 - 2.0 * m) / d if d > 0.0 else 0.0
    return m, (mad / m if m > 0.0 else _NAN), skew


def evaluate_pair(t: np.ndarray, sizes: np.ndarray, size_pop: np.ndarray, u: float,
                  do_z2: bool = True, own_ks: float = _NAN) -> Dict[str, Any]:
    """All B12 tests on one pair's buffered events.

    t: ascending finite event times; sizes: bytes per event (NaN = unknown);
    size_pop: sorted size shapes of the system's other pairs; u: the seeded
    tie-break uniform of the size rank. Returns {p, p_renewal, p_poisson,
    p_size, p_z2, kappa, lr, ks, period, n, span, iv_med, iv_disp, iv_skew,
    size_med, size_disp}; p is NaN when no test could be computed."""
    iv = np.diff(t)
    p_ren, kappa, lr, p_poi = renewal_p(iv)
    sz = sizes[np.isfinite(sizes)]
    ks = evt.gamma_renewal_lrt(np.maximum(sz, 1.0))[0] if sz.size >= 3 else _NAN
    p_size = _rank_p(ks, size_pop, u, own_ks)
    med, iv_disp, iv_skew = _dispersion(iv)
    p_z2, per_z2 = _z2_test(t, med) if do_z2 else (_NAN, _NAN)
    size_med, size_disp, _ = _dispersion(sz)
    ps = [p for p in (p_ren, p_size, p_z2) if p == p]
    p = max(P_FLOOR, min(1.0, N_TESTS * min(ps))) if ps else _NAN
    return {"p": p, "p_renewal": p_ren, "p_poisson": p_poi, "p_size": p_size, "p_z2": p_z2,
            "kappa": kappa, "lr": lr, "ks": ks, "period": _period(med, p_z2, per_z2),
            "n": int(t.size), "span": float(t[-1] - t[0]) if t.size else 0.0,
            "iv_med": med, "iv_disp": iv_disp, "iv_skew": iv_skew,
            "size_med": size_med, "size_disp": size_disp}


def _period(med: float, p_z2: float, per_z2: float) -> float:
    """P_hat: the median interval, refined by the Z^2 peak when that is
    significant and within 20 % of it (the scan's best period can sit on a
    harmonic of a strict train)."""
    if p_z2 == p_z2 and p_z2 <= KNOWN_P and med > 0.0 and abs(per_z2 - med) <= 0.2 * med:
        return per_z2
    return med


# ============================================================ learned state
def _init_state() -> Dict[str, Any]:
    return {"known": {}}


def _fold(known: Dict[int, List[float]], did: int, period: float, w: float, ts: float) -> None:
    """known[did] = [ln P, weight, last_ts]: a decayed weighted mean of ln P.
    A row older than the entry is folded with its weight decayed (the entry
    never moves backwards), so any row order gives the same result."""
    if not (w > 0.0 and period > 0.0 and math.isfinite(period)):
        return
    rec = known.get(did)
    lp = math.log(period)
    if rec is None:
        known[did] = [lp, w, ts]
        return
    lp0, w0, t0 = rec
    if ts >= t0:
        w0 *= 2.0 ** (-(ts - t0) / KNOWN_HALF_LIFE_S)
        t_new = ts
    else:
        w *= 2.0 ** (-(t0 - ts) / KNOWN_HALF_LIFE_S)
        t_new = t0
    wt = w0 + w
    rec[0], rec[1], rec[2] = (w0 * lp0 + w * lp) / wt, wt, t_new


def _update(state: Dict[str, Any], row: Tuple[float, tuple], w: float) -> Dict[str, Any]:
    """GatedLearner update: fold one tick's periodic pairs with trust weight w."""
    ts, items = row
    if not (w > 0.0 and math.isfinite(w)):
        return state
    known = state["known"]
    for did, period in items:
        _fold(known, int(did), float(period), float(w), float(ts))
    cut = float(ts) - KNOWN_KEEP_S
    for did in [d for d, r in known.items() if r[2] < cut]:
        del known[did]
    return state


def _merge(own: Dict[str, Any], other: Dict[str, Any], w: float) -> Dict[str, Any]:
    """own + w * other (link seeding)."""
    for did, (lp, wo, to) in (other.get("known") or {}).items():
        _fold(own["known"], int(did), math.exp(lp), float(w) * float(wo), float(to))
    return own


def _dump(state: Dict[str, Any]) -> Dict[str, Any]:
    return state                      # GatedLearner deep-copies blobs on put and load


def _load(blob: Any) -> Dict[str, Any]:
    return blob if isinstance(blob, dict) and "known" in blob else _init_state()


def known_period(model: Optional[Mapping[str, Any]], did: int, now: float) -> float:
    """Committed period (s) of an established beacon of this entity to did,
    NaN when none within BUF_AGE_S."""
    if not isinstance(model, Mapping):
        return _NAN
    rec = (model.get("state") or {}).get("known", {}).get(int(did))
    if rec is None or not now - rec[2] <= BUF_AGE_S:
        return _NAN
    return math.exp(rec[0])


def new_model() -> Dict[str, Any]:
    return {"fmt": 1, "version": 0, "pairs": {}, "tick": _NAN, "state": _init_state(),
            "gate": G.GateState(), "rows": {}}


def _new_pair() -> Dict[str, Any]:
    return {"t": _EMPTY, "sz": _EMPTY, "new": 0, "last_eval": _NAN, "res": None,
            "alert_ts": _NAN, "z2_ts": _NAN}


# ================================================================== helpers
class _Classes:
    """Role-class membership of one system, read once per tick (m_class
    semantics: prob >= MIN_PROB, a class needs >= MIN_MEMBERS members)."""

    def __init__(self, store: Any, system: str) -> None:
        self.role: Dict[str, str] = {}
        self.members: Dict[str, List[str]] = {}
        for key, a in (m_class.get(store).get("assign") or {}).items():
            s, _, ip = str(key).partition("|")
            if s != system or not isinstance(a, Mapping):
                continue
            r = a.get("role")
            if r in (None, "", "unique") or float(a.get("prob", 1.0)) < m_class.MIN_PROB:
                continue
            self.role[ip] = str(r)
            self.members.setdefault(str(r), []).append(ip)

    def others(self, e: str) -> List[str]:
        r = self.role.get(e)
        if r is None:
            return []
        mem = self.members.get(r, [])
        if len(mem) < m_class.MIN_MEMBERS:
            return []
        return [m for m in mem if m != e]


class _Prevalence:
    """(share of the system's entities, number of entities) using a destination,
    memoised for one system tick.

    The share comes from model.vocab@__system__ (B08, the specified source)
    through m_vocab.dest_prevalence on the destination's name; R2's decayed
    distinct-entity counts (m_template) stand in while B08 has no system
    model or does not know the name (NaN). The entity count is the larger of
    R2's count and the entities holding a live buffer for the destination
    (this engine), which covers a missing or reset model.template
    (population whitelisting errs towards silence)."""

    def __init__(self, store: Any, system: str, now: float,
                 holders: Mapping[int, Set[str]]) -> None:
        self.store, self.system, self.now, self.holders = store, system, now, holders
        self._n_sys: Optional[float] = None
        self._memo: Dict[int, Tuple[float, float]] = {}

    def __call__(self, did: int) -> Tuple[float, float]:
        r = self._memo.get(did)
        if r is None:
            r = self._memo[did] = self._lookup(did)
        return r

    def _lookup(self, did: int) -> Tuple[float, float]:
        st, s, now = self.store, self.system, self.now
        n_ent = max(float(len(self.holders.get(did, ()))), MT.dest_entities(st, s, did, now))
        share = _NAN
        name = MT.dest_name(st, s, did)
        if name:
            share = MV.dest_prevalence(st, s, name, now)
        if share != share:
            share = MT.dest_prevalence(st, s, did, now)
        if share != share:
            if self._n_sys is None:
                self._n_sys = max(MT.system_entities(st, s, now), float(len(st.entities(s))))
            share = n_ent / self._n_sys if self._n_sys > 0.0 else _NAN
        return share, n_ent


def _fresh_pos(store: Any, s: str, e: str, name: str, now: float) -> bool:
    v = store.latest_fresh(s, e, name, now)
    try:
        return v is not None and float(v) > 0.0
    except (TypeError, ValueError):
        return False


def _j(v: Any, nd: int = 4) -> Optional[float]:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _jp(v: Any) -> Optional[float]:
    """p-values keep their magnitude in JSON (rounding would zero them)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return float(f"{v:.4g}") if math.isfinite(v) else None


# =================================================================== engine
class _Ent:
    """One entity's per-tick work item."""
    __slots__ = ("e", "model", "active", "new_model", "evaluated", "alerts", "row")

    def __init__(self, e: str, model: Dict[str, Any], active: bool, new: bool) -> None:
        self.e, self.model, self.active, self.new_model = e, model, active, new
        self.evaluated: List[Tuple[int, Dict[str, Any], bool]] = []   # (did, res, eligible)
        self.alerts: List[Tuple[int, Dict[str, Any], float]] = []     # (did, res, n_ent)
        self.row: List[Tuple[int, float]] = []


class BeaconEngine(Engine):
    name = "behavior.beacon"
    layer = "behavior"
    consumes = ["act.rare_events", "act.events", "model.template", "model.vocab", "model.class",
                "model.feedback", "behavior.trust", "behavior.trust_prov",
                "behavior.quarantine", "model.control", "model.link"]
    produces = [MODEL, "behavior.score", "behavior.pm", "behavior.acc_alarm", "behavior.axes",
                "behavior.degraded", "profile.extra.beacons", "event:beacon"]
    description = ("C2 beacon detection per (entity, rare destination) on point events: "
                   "Gamma renewal LRT with an exact finite-n null, size constancy against "
                   "the population, Z^2_2 strict-period scan, Bonferroni; prevalence, "
                   "class-sharing and allowlist whitelisting.")
    interval = 1

    def __init__(self, max_eval_per_tick: int = MAX_EVAL_PER_TICK,
                 z2_max_per_tick: int = Z2_MAX_PER_TICK, **params: Any) -> None:
        super().__init__(**params)
        self.max_eval_per_tick = int(max_eval_per_tick)
        self.z2_max_per_tick = int(z2_max_per_tick)
        self._rows_cur: Optional[Dict[float, tuple]] = None
        self._learner = G.GatedLearner(
            name=LEARNER, init=_init_state, update=_update, fetch=self._fetch,
            dump=_dump, load=_load, merge=_merge, clock=CLOCK)

    # ------------------------------------------------------------------ run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        now, dt = float(ctx.now), float(ctx.window_s)
        self._learner.d_min_s = float(ctx.config.get("D_min_s") or G.D_MIN_S)
        r2_failed = store.engine_failed(R2_ENGINE, now)
        n = 0
        for s in store.systems():
            n += self._system(ctx, s, now, dt, r2_failed)
        return n

    def _system(self, ctx: Context, s: str, now: float, dt: float, r2_failed: bool) -> int:
        store = ctx.store
        # pass 1: ingest this tick's rare events into every entity's buffers
        ents: List[_Ent] = []
        for e in store.entities(s):
            model = store.get_model(s, e, MODEL)
            if not (isinstance(model, dict) and model.get("fmt") == 1):
                model = None
            rare = {} if r2_failed else MT.rare_events(store, s, e, now)
            active = bool(rare) or _fresh_pos(store, s, e, "act.events", now)
            if model is None and not rare and not active:
                continue
            new = model is None
            if new:
                model = new_model()
            _ingest(model, rare, now)
            ents.append(_Ent(e, model, active, new))
        if not ents:
            return 0
        holders: Dict[int, Set[str]] = {}
        pop_ks: List[float] = []
        for it in ents:
            for did, pr in it.model["pairs"].items():
                holders.setdefault(did, set()).add(it.e)
                res = pr["res"]
                if res is not None and res["ks"] == res["ks"]:
                    pop_ks.append(float(res["ks"]))
        size_pop = np.sort(np.asarray(pop_ks, dtype=np.float64))
        classes = _Classes(store, s)
        prev = _Prevalence(store, s, now, holders)
        # buffer admission: destinations that became common or are class-shared
        # (a class-wide service such as a health check) are not kept
        for it in ents:
            others = classes.others(it.e)
            for did in list(it.model["pairs"]):
                share, _ = prev(did)
                if share == share and share > RARE_SHARE * (1.0 + 1e-9):
                    del it.model["pairs"][did]
                elif others:
                    k = sum(1 for m in others if m in holders[did])
                    if k >= CLASS_SHARE_MIN and k >= CLASS_SHARE_FRAC * len(others):
                        del it.model["pairs"][did]
        # pass 2: evaluate due pairs (stalest first, bounded per tick)
        if not r2_failed:
            self._evaluate(ctx, s, ents, size_pop, classes, prev, now, dt)
        n = 0
        frontier = G.commit_frontier(now, dt, self._learner.d_min_s)
        for it in ents:
            n += self._write(ctx, s, it, now, dt, r2_failed)
            m = it.model
            if it.new_model and not (m["pairs"] or it.row):
                continue                    # active, nothing rare: no model to keep
            self._learn(ctx, s, it, now, dt, frontier)
            store.put_model(s, it.e, MODEL, m, version=int(m["version"]), ts=now)
        return n

    # ----------------------------------------------------------- evaluation
    def _evaluate(self, ctx: Context, s: str, ents: List[_Ent], size_pop: np.ndarray,
                  classes: _Classes, prev: _Prevalence, now: float, dt: float) -> None:
        store = ctx.store
        wait = EVAL_EVERY_TICKS * dt - 1e-6
        due: List[Tuple[float, str, int, _Ent]] = []
        for it in ents:
            for did, pr in it.model["pairs"].items():
                if pr["new"] <= 0 or pr["t"].size < MIN_EVENTS:
                    continue
                le = pr["last_eval"]
                if le == le and now - le < wait:
                    continue
                due.append((le if le == le else -math.inf, it.e, did, it))
        if not due:
            return
        due.sort(key=lambda x: (x[0], x[1], x[2]))
        done: List[Tuple[_Ent, int, Dict[str, Any], Dict[str, Any]]] = []
        for _, e, did, it in due[:max(0, self.max_eval_per_tick)]:
            pr = it.model["pairs"][did]
            own = pr["res"]["ks"] if pr["res"] is not None else _NAN
            u = combine.seeded_uniform("b12-size", s, e, did, now)
            res = evaluate_pair(pr["t"], pr["sz"], size_pop, u, do_z2=False, own_ks=own)
            pr["res"], pr["last_eval"], pr["new"] = res, now, 0
            res["ts"] = now
            done.append((it, did, pr, res))
        # Z^2 scans: bounded per tick, least recently scanned first, and skipped
        # where the renewal test alone already alarms
        budget = self.z2_max_per_tick
        for it, did, pr, res in sorted(done, key=lambda x: (_z2_age(x[2]), x[0].e, x[1])):
            if budget <= 0:
                break
            pren = res["p_renewal"]
            if pren == pren and N_TESTS * pren < P_ALERT:
                continue
            p_z2, per_z2 = _z2_test(pr["t"], res["iv_med"])
            budget -= 1
            pr["z2_ts"] = now
            res["p_z2"] = p_z2
            if p_z2 == p_z2:
                p0 = res["p"] if res["p"] == res["p"] else 1.0
                res["p"] = max(P_FLOOR, min(p0, 1.0, N_TESTS * p_z2))
                res["period"] = _period(res["iv_med"], p_z2, per_z2)
        # population whitelisting and the alert decision
        allow = bool(m_feedback.get(store).get("allowlist"))
        for it, did, pr, res in done:
            e, p = it.e, res["p"]
            name = MT.dest_name(store, s, did) or str(did)
            res["dest"] = name
            listed = allow and (
                m_feedback.allowlisted(store, s, e, ALLOW_DIM, name, now)
                or m_feedback.allowlisted(store, s, e, ALLOW_DIM, str(did), now))
            shared = p == p and _class_shares(store, s, classes.others(e), did,
                                              res["period"], now)
            res["allowlisted"], res["class_shared"] = bool(listed), bool(shared)
            eligible = not listed and not shared
            it.evaluated.append((did, res, eligible))
            if p == p and p <= KNOWN_P:
                it.row.append((did, float(res["period"])))
            if eligible and p == p and p < P_ALERT:
                n_ent = prev(did)[1]
                res["prev_entities"] = n_ent
                if n_ent <= PREV_MAX_ENTITIES + 1e-9:
                    it.alerts.append((did, res, n_ent))

    # --------------------------------------------------------------- writes
    def _write(self, ctx: Context, s: str, it: _Ent, now: float, dt: float,
               r2_failed: bool) -> int:
        store, e, win = ctx.store, it.e, int(dt)
        if r2_failed:
            emit.write_scores(store, s, e, now, {DETECTOR: _NAN},
                              degraded={DETECTOR: "producer_error:" + R2_ENGINE}, window_s=win)
            return 1
        ps = [r["p"] for _, r, ok in it.evaluated if ok and r["p"] == r["p"]]
        if not ps and not it.active:
            return 0                                   # silent entity: nothing to score
        alarm = 0 if ctx.training else int(bool(it.alerts))
        if ps:
            pm = max(P_FLOOR, min(1.0, len(ps) * min(ps)))
            score = -math.log10(pm)
            emit.write_scores(store, s, e, now, {DETECTOR: score}, pm={DETECTOR: pm},
                              axes={DETECTOR: AXES}, acc_alarm={DETECTOR: alarm}, window_s=win)
        else:
            emit.write_scores(store, s, e, now, {DETECTOR: 0.0}, acc_alarm={DETECTOR: 0},
                              window_s=win)
        if alarm:
            for did, res, n_ent in it.alerts:
                self._event(store, s, it, did, res, n_ent, now, dt)
        if it.evaluated:
            _write_profile(store, s, e, it.model, now)
        return 1

    def _event(self, store: Any, s: str, it: _Ent, did: int, res: Dict[str, Any],
               n_ent: float, now: float, dt: float) -> None:
        pr = it.model["pairs"][did]
        last = pr["alert_ts"]
        if last == last and now - last < EVENT_COOLDOWN_S:
            return
        pr["alert_ts"] = now
        p = float(res["p"])
        ed = combine.e_day(p, dt)
        sev = combine.e_day_severity(ed) or "low"
        t = pr["t"]
        store.add_event(BehaviorEvent(
            system=s, entity=it.e, ts=now, kind=EVENT_KIND,
            score=float(min(1.0, -math.log10(max(p, P_FLOOR)) / 10.0)),
            severity=Severity(sev),
            description=(f"Beacon to rare destination {res['dest']}: {res['n']} events, "
                         f"period ~{res['period']:.0f} s (renewal shape {res['kappa']:.3g}), "
                         f"used by {n_ent:.1f} entities, p = {p:.2g}"),
            extra={"dim": ALLOW_DIM, "value": res["dest"], "dest_id": int(did),
                   "period_s": _j(res["period"], 3), "kappa": _j(res["kappa"]),
                   "n": int(res["n"]), "p_renewal": _jp(res["p_renewal"]),
                   "p_size": _jp(res["p_size"]), "p_z2": _jp(res["p_z2"]),
                   "prev_entities": _j(n_ent), "iv_disp": _j(res["iv_disp"]),
                   "size_disp": _j(res["size_disp"])},
            p_value=p, e_day=float(ed), axes=list(AXES), p_by_detector={DETECTOR: p},
            dedupe_key=f"{EVENT_KIND}|{s}|{it.e}|{did}",
            model_version=int(it.model["version"]),
            window=(float(t[0]), float(t[-1])) if t.size else (now, now)))

    # ------------------------------------------------------------- learning
    def _learn(self, ctx: Context, s: str, it: _Ent, now: float, dt: float,
               frontier: float) -> None:
        store, model = ctx.store, it.model
        rows: Dict[float, tuple] = model["rows"]
        if it.row:
            rows[now] = tuple(it.row)
        gate: G.GateState = model["gate"]
        pending = any(gate.last_ts < ts <= frontier for ts in rows)
        state = model["state"]
        self._rows_cur = rows
        try:
            if pending or store.get_model(s, it.e, G.CONTROL_MODEL) is not None:
                state, gate = self._learner.step(store, s, it.e, state, gate, now, dt,
                                                 training=bool(ctx.training))
            state, gate = self._learner.seed_from_link(
                store, s, it.e, state, gate,
                lambda src: _other_state(store, s, src))
        finally:
            self._rows_cur = None
        model["state"], model["gate"], model["version"] = state, gate, int(gate.version)
        cut = now - ROW_KEEP_S
        for ts in [t for t in rows if t < cut]:
            del rows[ts]
        while len(rows) > ROW_CAP:
            del rows[min(rows)]

    def _fetch(self, store: Any, s: str, e: str, ts: float) -> Optional[Tuple[float, tuple]]:
        rows = self._rows_cur
        if rows is None:
            return None
        r = rows.get(float(ts))
        return (float(ts), r) if r is not None else None


# ================================================================ internals
def _ingest(model: Dict[str, Any], rare: Mapping[int, np.ndarray], now: float) -> None:
    """Append this tick's rare events (once per tick), drop events older than
    7 d, keep the newest BUF_MAX per pair and MAX_PAIRS pairs per entity."""
    pairs: Dict[int, Dict[str, Any]] = model["pairs"]
    if rare and model["tick"] != now:
        for did, arr in rare.items():
            a = np.asarray(arr, dtype=np.float64).reshape(-1, 3)
            a = a[np.isfinite(a[:, 0])]
            if not a.shape[0]:
                continue
            pr = pairs.get(int(did))
            if pr is None:
                pr = pairs[int(did)] = _new_pair()
            sz = a[:, 1] + a[:, 2]
            sz = np.where(np.isfinite(sz) & (sz >= 0.0), sz, _NAN)
            t = np.concatenate((pr["t"], a[:, 0]))
            z = np.concatenate((pr["sz"], sz))
            if (pr["t"].size and a[0, 0] < pr["t"][-1]) or np.any(np.diff(a[:, 0]) < 0.0):
                o = np.argsort(t, kind="stable")
                t, z = t[o], z[o]
            pr["t"], pr["sz"] = t, z
            pr["new"] += int(a.shape[0])
    model["tick"] = now if rare else model["tick"]
    lo = now - BUF_AGE_S
    for did in list(pairs):
        pr = pairs[did]
        t = pr["t"]
        i = int(np.searchsorted(t, lo, side="left"))
        i = max(i, t.size - BUF_MAX)
        if i > 0:
            pr["t"], pr["sz"] = t[i:], pr["sz"][i:]
        if not pr["t"].size:
            del pairs[did]
    if len(pairs) > MAX_PAIRS:
        keep = sorted(pairs, key=lambda d: -float(pairs[d]["t"][-1]))[:MAX_PAIRS]
        model["pairs"] = {d: pairs[d] for d in keep}


def _z2_age(pr: Mapping[str, Any]) -> float:
    z = pr["z2_ts"]
    return z if z == z else -math.inf


def _class_shares(store: Any, s: str, others: Sequence[str], did: int, period: float,
                  now: float) -> bool:
    """Another class member has LEARNED a beacon to did at period +- 10 %."""
    if not others or not (period > 0.0 and math.isfinite(period)):
        return False
    for m in others:
        po = known_period(store.get_model(s, m, MODEL), did, now)
        if po == po and abs(po - period) <= PERIOD_TOL * period:
            return True
    return False


def _other_state(store: Any, s: str, entity: str) -> Optional[Dict[str, Any]]:
    m = store.get_model(s, entity, MODEL)
    if not isinstance(m, dict) or not isinstance(m.get("state"), dict):
        return None
    return {"known": {int(k): list(v) for k, v in m["state"].get("known", {}).items()}}


def _write_profile(store: Any, s: str, e: str, model: Dict[str, Any], now: float) -> None:
    """profile.extra.beacons: the most suspicious pairs (JSON-friendly)."""
    items = []
    for did, pr in model["pairs"].items():
        r = pr["res"]
        if r is None:
            continue
        items.append((r["p"] if r["p"] == r["p"] else 2.0, did, pr, r))
    items.sort(key=lambda x: (x[0], x[1]))
    known = model["state"].get("known", {})
    out = []
    for _, did, pr, r in items[:PROFILE_TOP]:
        out.append({
            "dest": r.get("dest", str(did)), "dest_id": int(did), "n": int(r["n"]),
            "period_s": _j(r["period"], 3), "p": _jp(r["p"]), "p_renewal": _jp(r["p_renewal"]),
            "p_size": _jp(r["p_size"]), "p_z2": _jp(r["p_z2"]), "kappa": _j(r["kappa"]),
            "iv_disp": _j(r["iv_disp"]), "iv_skew": _j(r["iv_skew"]),
            "size_med": _j(r["size_med"], 1), "size_disp": _j(r["size_disp"]),
            "first_ts": float(pr["t"][0]) if pr["t"].size else None,
            "last_ts": float(pr["t"][-1]) if pr["t"].size else None,
            "evaluated": _j(r.get("ts")), "established": int(did) in known,
            "allowlisted": bool(r.get("allowlisted")),
            "class_shared": bool(r.get("class_shared")),
            "alerted": pr["alert_ts"] == pr["alert_ts"]})
    p = store.profile(s, e) or EntityProfile(system=s, entity=e)
    p.extra["beacons"] = {"pairs": out, "n_pairs": len(model["pairs"]),
                          "n_established": len(known), "version": int(model["version"]),
                          "updated": now}
    store.put_profile(p)
