"""BudgetEngine (B13): long-horizon cumulative budgets, DLP style.

Why: low-and-slow exfiltration, duty-cycled transfers and slow enumeration
never cross a per-tick threshold. 30 % more upload on every tick is a
per-tick z of 0.9, invisible to B04, but by mid-afternoon the day-to-date
total is many standard errors above what the same hours of every other
workday produced. B13 keeps cumulative quantities per entity and asks, once
per hour of wall clock, whether a budget over a long horizon is exhausted.

Quantities (natural units), by evidence axis:
  volume   bytes_up, bytes_down, writes (http_write_ratio x http_requests of
           feature.nat) and active 15-min slots (from act.stream timestamps,
           feature.active when a tick has no stream);
  exfil    up_novel: bytes up to external destinations the entity first saw
           less than 7 d ago or that are rare in the system (act.stream up per
           dest_id, act.rare_events for rare destinations, which are outside
           the stream cap); dns_label: DNS label bytes (left of eTLD+1, from
           dns.qname_set) under new or rare external eTLD+1s (tunnels);
  breadth  objs: distinct object ids per template per local day (exact sets up
           to 256 ids, then HLL p = 10, from act.objs), templates: distinct
           template keys per day (act.tokens), dests: distinct destinations
           per day. Distinct counts are not additive, so a tick contributes
           the INCREASE of the day's distinct count; horizons within a day are
           then exact, the 7-d horizon is the sum of daily distinct counts.

Storage. Every tick's quantity row goes to
  * live hourly bins (all ticks, ungated, 8 d): the current horizon sums B;
  * a row buffer (8 d + 1 h) that lib/gating.GatedLearner commits D ticks
    later with the trust weight into the committed ring: 29 d x 24 hourly
    bins per quantity that carry their local hour index, so a rollback
    (restore checkpoint + replay the journal) and a release (held rows) put
    every row back into the bin of its own hour. A committed bin holds
    sum w x, sum w dt/3600 and the covered fraction of the hour; its value is
    the trust-weighted rate per full hour (valid with >= 30 min covered).
    Commits are batched (4 due rows, or one waiting an hour): learning later
    than D is allowed and the gate's per-call cost dominates at 60 s.

Horizons (hours ending with the current local hour, which is partial):
1 h, 8 h, day-to-date (from local midnight) and 7 d. Within an hour every
horizon sum only grows, so an alarm at any tick of the hour implies the
end-of-hour value exceeds the phase threshold: the false-alarm budget is
exactly that of 24 hourly evaluations per day, at any cadence (900 -> 60 s).

Thresholds (refit round-robin once per local day, and at once after a
rollback / release / rebase or link seed; the day's fits are spread over
12 h, least recently served entity first), from the committed ring over the
28 past local days (today excluded):
  * window sums W[d, h] for each horizon, day d, local phase h, taken as
    L = log(W + unit), unit = 1 % of the absolute floor: budgets are
    multiplicative, and a phase whose 8-h window is one busy lognormal hour
    is far more skewed than one summing 24 hours; only in logs are the
    standardised residuals of different phases exchangeable;
  * per (horizon, day_type, phase): median and robust scale of L (MAD, mean
    absolute deviation when the MAD is 0; >= 0.02, a 2 % relative spread)
    over the same-phase, same-day_type history (7 d: every day), the median
    falling back to all day types with < 4 samples and the scale to the
    spread pooled over all days with < 10 (weekends), then smoothed over
    +-2 local hours. Same-phase 7-d windows of consecutive days share 6
    days, so their spread understates the scale: it is at least the robust
    log scale of the rolling 24-h sums / sqrt(7), and the 7-d z_q at least
    the normal quantile;
  * the standardised residuals of all phases and days are pooled, and a
    POT/GPD tail is fitted above u = their P98 with the PWM estimators
    (lib/evt). With ~13 exceedances, many from the same day, the raw PWM
    shape is too noisy for a 1e-5 quantile: xi is shrunk n/(n + 30) towards
    0 and clipped to [0, 0.3], sigma = mean excess (1 - xi) >= 0.4 (a
    collapse onto one day's residuals cannot give a razor-thin tail). There
    is no upward drift term.
  * z_q = exp(median + scale * pot_quantile(u, xi, sigma, rate, q)) - unit
    with q = 0.01 / (#Q #H 24): the family false-alarm rate is <= 0.01 per
    entity-day (null runs, 630 entity-days with a diurnal / weekly profile
    and hourly LogNormal(0.5) noise: 0.16 % alarm days; 0 in 84 at 900 s).
  * Fewer than 20 days of history: the peers' pooled GPD (median parameters
    of the mature class members, else of the mature system entities) with
    the peers' log scales, i.e. their relative spread rescaled by the entity
    median; phases without own samples take the peers' median level.
    Without mature peers, an entity with >= 7 days fits its own tail;
    younger ones stay unscored (NaN).
  * Round 4: the 7-d scale used for the threshold and the tail p of a live
    window is multiplied by seven_day_pinf = sqrt(1 + (pi/2) / (n_days/7)):
    the same-phase median of overlapping 7-d windows rests on ~n_days / 7
    independent weeks, an error the in-sample residuals do not contain
    (pack A live: 7-d cells at 11.9x nominal at 1e-3). And on the young
    peer path, a phase with no trusted own sample but an untrusted recurring
    observed pattern (>= RECUR_MIN_DAYS days; rows committed with weight 0,
    kept apart in the ring as 'ox' / 'oc') takes the entity's observed level
    instead of the peers' (_observed_fill: the L15 nightly backup).
Alarm on (Q, H) when B > z_q AND B - median > abs_floor_Q (config
'budget_abs_floor', defaults below) AND, except for exfil, the common-mode
guard passes. Peers are the class members (m_class, >= 3 in the system),
else the system's entities; B / peer_median (current horizon sums of the
peers, median over >= 2) must exceed the P99 of the entity's own history of
that ratio (the peers' committed windows at the same hours) AND the level-q
threshold of the log ratio (same-phase median / robust scale, POT quantile
of the pooled residuals, like z_q). The second half strengthens the spec's
P99 guard: under a class-wide x2.5 day B > z_q holds at every evaluation and
the ratio keeps its null law, so the P99 alone passes 1 % of the hourly
evaluations (measured with the P99 alone: 1 alarming entity in 48
class-entity-days of a x2.5 surge, 2x the whole budget); the level-q ratio
test passes at rate q (0 in the same runs). Without a ratio fit the
threshold is instead rescaled by the peers' current over usual level.

Actors: when model.link@(s, __system__).actors chains IPs, the members'
horizon sums are added and evaluated against the thresholds of the member
seen last (an IP-hopping actor splits its volume below every per-IP floor).

Scores (accumulators): score.budget_vol / budget_exfil / budget_breadth =
-log10 of the axis tail p (Bonferroni minimum over the axis's quantities and
horizons; also written as behavior.pm). The tail p is the POT tail above u
and the empirical body below. acc_alarm per axis; behavior.budget
{'<Q>.<H>': [value, z_q, tail_p]} (values in natural units, also below any
floor, so B18 can aggregate small per-entity exfil across a class);
profile.extra.budget hourly; event budget_exceeded (one per quantity per
local day and scope) with natural-unit text. Training mode learns and
scores but never alarms or emits.

Contract M: a tick whose feature.nat was not written (B01 failed or stale)
scores NaN with behavior.degraded for all three detectors; an R2 failure
degrades exfil and breadth only. A row with a NaN quantity is never learned.
"""
from __future__ import annotations

import datetime as _dt
import math
import weakref
from collections.abc import Mapping
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import BehaviorEvent, DerivedMetric, EntityProfile, MetricKind, Severity
from .lib import combine, emit, evt
from .lib import features as F
from .lib import gating as G
from .lib import m_calib, m_class
from .lib import m_template as MT
from .lib import names as NM
from .lib import timebins as TB
from .lib.classkeys import SYSTEM_KEY
from .lib.sketch import HyperLogLog

try:                                    # B08's accessor; absent until B08 lands
    from .lib import m_vocab as _MV
except ImportError:                     # pragma: no cover
    _MV = None

MODEL = "model.budget"
LEARNER = "budget"
SERIES = "behavior.budget"
EVENT_KIND = "budget_exceeded"
LINK_MODEL = "model.link"
B01_ENGINE = "behavior.feature_vector"
R2_ENGINE = "raw.action_token"
NAT = "feature.nat"
ACTIVE = "feature.active"
FMT = 1

# ------------------------------------------------------------ quantities
Q_NAMES: Tuple[str, ...] = ("bytes_up", "bytes_down", "writes", "slots",
                            "up_novel", "dns_label", "objs", "templates", "dests")
Q_AXIS: Tuple[str, ...] = ("volume",) * 4 + ("exfil",) * 2 + ("breadth",) * 3
Q_UNIT: Tuple[str, ...] = ("bytes", "bytes", "requests", "slots", "bytes", "bytes",
                           "objects", "templates", "destinations")
NQ = len(Q_NAMES)
QI = {q: i for i, q in enumerate(Q_NAMES)}
AXES: Tuple[str, ...] = ("volume", "exfil", "breadth")
AXIS_DETECTOR = {"volume": "budget_vol", "exfil": "budget_exfil", "breadth": "budget_breadth"}
AXIS_Q = {ax: np.array([i for i, a in enumerate(Q_AXIS) if a == ax]) for ax in AXES}
GUARDED = np.array([a != "exfil" for a in Q_AXIS])          # common-mode guard applies
ACT_Q = np.concatenate((AXIS_Q["exfil"], AXIS_Q["breadth"]))  # need R2's act.* series

# absolute floors (natural units per horizon); ctx.config['budget_abs_floor'] overrides
ABS_FLOOR_DEFAULT: Dict[str, float] = {
    "bytes_up": 5e6, "bytes_down": 50e6, "writes": 200.0, "slots": 8.0,
    "up_novel": 20e6, "dns_label": 2e5, "objs": 500.0, "templates": 50.0, "dests": 50.0,
}
UNIT_FRAC = 0.01          # log offset: log(W + 1 % of the absolute floor)
LOG_SCALE_MIN = 0.02      # scale floor: a 2 % relative spread

# -------------------------------------------------------------- horizons
H_NAMES: Tuple[str, ...] = ("1h", "8h", "day", "7d")
H_LEN: Tuple[int, ...] = (1, 8, 0, 168)      # hours; 0 = day-to-date
NH = len(H_NAMES)
H_7D = 3
H_LABEL = {"1h": "last hour", "8h": "last 8 h", "day": "day-to-date", "7d": "last 7 d"}

# ------------------------------------------------------------ rings, fits
HIST_DAYS = 28
NHIST = HIST_DAYS * 24
NB = (HIST_DAYS + 1) * 24          # committed ring: 28 past days + today
LIVE_HOURS = 8 * 24                # live bins: 7 d horizon + the current day
MIN_COV = 0.5                      # covered fraction of an hour for a valid bin
DAY_MIN_HOURS = 12                 # valid hours for a day to count as history
MATURE_DAYS = 20                   # own tail from here on
MIN_OWN_DAYS = 7                   # own (immature) tail when no mature peer exists
MIN_PHASE_N = 4
PHASE_SMOOTH = 2                   # scale: running median over +-2 local hours
MIN_SCALE_N = 10                   # days of one day type for its own scale
MIN_PHASE_N_YOUNG = 2
RECUR_MIN_DAYS = 4                 # untrusted observations that make a phase 'own' (round 4)
MIN_POOL = 96                      # pooled residuals for an own tail
MIN_POOL_IMMATURE = 48
TAIL_U_Q = 0.98
TAIL_U_FALLBACK = 0.90
XI_MAX = 0.3
XI_SHRINK_N = 30.0
SIGMA_MIN = 0.4                    # robust-scale units (a normal tail has ~0.38 above P98)
Q_EVAL = 0.01 / (NQ * NH * 24)     # per (Q, H) hourly evaluation
Z_NORMAL = NormalDist().inv_cdf(1.0 - Q_EVAL)   # 7-d sums are CLT-normal at least
SQRT7 = math.sqrt(7.0)
PQ = np.linspace(0.0, TAIL_U_Q, 50)          # body quantile grid (last = u)
RATIO_Q = 0.99
RATIO_MIN_N = 48
MIN_PEERS = 2
PEER_EXACT_MAX = 8                 # leave-self-out peer medians up to this set size
LOGRATIO_SCALE_MIN = 0.05          # log units: peer ratios are never tighter than 5 %
REFIT_S = 86400.0                  # every quantity is refitted once per local day
FIT_SPREAD_S = 12 * 3600.0         # ... spread over the first half of the day
FIT_MIN_PER_TICK = 4
COMMIT_BATCH = 4
FIT_MAX_PER_TICK = 64
P_FLOOR = 1e-300
PM_STORE_FLOOR = m_calib.P_ISSUED_FLOOR
AXES_PM_MAX = 0.05
SRC_NONE, SRC_OWN, SRC_PEER, SRC_OWN_IMMATURE = 0, 1, 2, 3
SRC_NAME = {SRC_NONE: "none", SRC_OWN: "own", SRC_PEER: "peer", SRC_OWN_IMMATURE: "own_immature"}

# --------------------------------------------------- novelty / day sets
NOVEL_S = 7 * 86400.0
DEST_KEEP_S = 30 * 86400.0
DEST_CAP = 4096
RARE_SHARE = MT.RARE_SHARE
OBJ_SET_CAP = MT.OBJ_IDS_CAP       # exact ids up to this many, then HLL
OBJ_TMPL_CAP = 128
DAY_SET_CAP = 8192
OVERFLOW = "\x00overflow"

ROW_KEEP_S = G.JOURNAL_MAX_AGE_S + 3600.0
PROFILE_EVERY_S = 3600.0
SERIES_KEEP_POINTS = 24            # behavior.budget: the newest points only (36 entries each)
SERIES_KEEP_S = 86400.0

_NAN = math.nan
_I_UP = F.FEATURE_INDEX["bytes_up"]
_I_DOWN = F.FEATURE_INDEX["bytes_down"]
_I_REQ = F.FEATURE_INDEX["http_requests"]
_I_WR = F.FEATURE_INDEX["http_write_ratio"]
_EPOCH = _dt.date(1970, 1, 1)


# ================================================================ rows
class _Rows:
    """Per-tick quantity rows in ascending ts, kept ROW_KEEP_S: what the gated
    learner folds at commit (t - D), release / rebase (held rows) and rollback
    replay. Grows by doubling, compacts once the pruned head is half the
    capacity (amortised O(1) appends). 56 B per row."""

    __slots__ = ("ts", "a", "frac", "x", "h", "n")

    def __init__(self, cap: int = 64) -> None:
        self.ts = np.empty(cap, dtype=np.float64)
        self.a = np.empty(cap, dtype=np.int64)
        self.frac = np.empty(cap, dtype=np.float32)
        self.x = np.empty((cap, NQ), dtype=np.float32)
        self.h = 0
        self.n = 0

    def __len__(self) -> int:
        return self.n - self.h

    def last_ts(self) -> float:
        return float(self.ts[self.n - 1]) if self.n > self.h else -math.inf

    def append(self, ts: float, a: int, frac: float, x: np.ndarray) -> None:
        n = self.n
        if n > self.h:
            last = float(self.ts[n - 1])
            if ts == last:
                self.a[n - 1], self.frac[n - 1], self.x[n - 1] = a, frac, x
                return
            if ts < last:
                raise ValueError(f"budget rows: ts {ts} older than newest row {last}")
        if n == self.ts.shape[0]:
            self._make_room()
            n = self.n
        self.ts[n], self.a[n], self.frac[n], self.x[n] = ts, a, frac, x
        self.n = n + 1

    def _make_room(self) -> None:
        cap = self.ts.shape[0]
        k = self.n - self.h
        new_cap = cap if self.h >= cap // 2 else 2 * cap
        sl = slice(self.h, self.n)
        ts, a = np.empty(new_cap, np.float64), np.empty(new_cap, np.int64)
        frac, x = np.empty(new_cap, np.float32), np.empty((new_cap, NQ), np.float32)
        ts[:k], a[:k], frac[:k], x[:k] = self.ts[sl], self.a[sl], self.frac[sl], self.x[sl]
        self.ts, self.a, self.frac, self.x = ts, a, frac, x
        self.h, self.n = 0, k

    def find(self, ts: float) -> int:
        h, n = self.h, self.n
        if n <= h:
            return -1
        i = h + int(np.searchsorted(self.ts[h:n], ts - 1e-6, side="left"))
        return i if i < n and abs(float(self.ts[i]) - ts) <= 1e-6 else -1

    def count_in(self, t0: float, t1: float) -> int:
        """Rows with t0 < ts <= t1."""
        v = self.ts[self.h:self.n]
        return int(np.searchsorted(v, t1, side="right") - np.searchsorted(v, t0, side="right"))

    def prune(self, cutoff: float) -> None:
        h, n = self.h, self.n
        if n > h and self.ts[h] < cutoff:
            self.h = h + int(np.searchsorted(self.ts[h:n], cutoff, side="left"))


class _Distinct:
    """Distinct ids of one template for one local day: an exact set up to
    OBJ_SET_CAP ids, then HLL(p = 10). `n` only grows (an HLL estimate may
    dip below the exact count it replaced; increments stay >= 0)."""

    __slots__ = ("ids", "hll", "n")

    def __init__(self) -> None:
        self.ids: Optional[Set[str]] = set()
        self.hll: Optional[HyperLogLog] = None
        self.n = 0.0

    def _to_hll(self) -> HyperLogLog:
        if self.hll is None:
            self.hll = HyperLogLog()
            if self.ids:
                self.hll.add_many(self.ids)
            self.ids = None
        return self.hll

    def add(self, ids: Iterable[Any], hll: Optional[bytes]) -> None:
        if hll:
            self._to_hll().merge(HyperLogLog.from_bytes(hll))
        elif self.hll is not None:
            self.hll.add_many(ids)
        else:
            assert self.ids is not None
            self.ids.update(str(i) for i in ids)
            if len(self.ids) > OBJ_SET_CAP:
                self._to_hll()
        cnt = float(len(self.ids)) if self.hll is None else self.hll.count()
        self.n = max(self.n, cnt)


# =============================================================== model
def new_state() -> Dict[str, np.ndarray]:
    """Committed ring: local hour index per slot (-1 empty), sum w x, sum w
    dt/3600, covered fraction of the hour; and (round 4) the rows the gate
    committed with weight 0 (untrusted: e.g. training trust 0 on a HIGH
    lib-4 match), unweighted: sum x and covered fraction ('ox', 'oc'). They
    never enter the fitted history; they only replace the PEERS' level at
    phases without any trusted own sample (see _observed_fill)."""
    return {"hour": np.full(NB, -1, dtype=np.int64), "sx": np.zeros((NB, NQ)),
            "sw": np.zeros(NB), "sc": np.zeros(NB), "ox": np.zeros((NB, NQ)),
            "oc": np.zeros(NB)}


def _copy_state(st: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: v.copy() for k, v in st.items()}


def new_fit() -> Dict[str, Any]:
    nan = np.nan
    return {
        "ts": np.full(NQ, -np.inf), "day": np.full(NQ, -1, dtype=np.int64),
        "src": np.zeros(NQ, dtype=np.int8), "days": np.zeros(NQ, dtype=np.int16),
        "med": np.full((NQ, NH, 2, 24), nan), "scale": np.full((NQ, NH, 2, 24), nan),
        "tail": np.full((NQ, NH, 5), nan),            # u, xi, sigma, rate, z_q (std units)
        "body": np.full((NQ, NH, PQ.size), nan),
        "rp99": np.full((NQ, NH), nan), "pmh": np.full((NQ, NH, 2, 24), nan),
        "rthr": np.full((NQ, NH, 2, 24), nan),        # log(B / peer median) threshold at q
        "unit": np.full(NQ, nan),                     # offset of log(W + unit) at the fit
        # 7-d day-type composition (round 2): per day type the median hourly
        # rate profile, and per phase the usual expected 7-d sum
        "prof": np.full((NQ, 2, 24), nan), "e7bar": np.full((NQ, 24), nan),
        # round 4: predictive inflation of the live 7-d residual per phase
        "pinf": np.ones((NQ, 24)),
    }


def new_live() -> Dict[str, Any]:
    return {
        "bh": np.full(LIVE_HOURS, -1, dtype=np.int64), "bx": np.zeros((LIVE_HOURS, NQ)),
        "day": None, "objs": {}, "tmpl": set(), "dests": set(),
        "dest_first": {}, "last_slot": -1, "dest_sweep": -math.inf,
        "tick_ts": None, "tick_x": None, "reported": {},
        "last": None,
    }


def new_model() -> Dict[str, Any]:
    return {"fmt": FMT, "version": 0, "state": new_state(), "gate": G.GateState(),
            "rows": _Rows(), "live": new_live(), "fit": new_fit()}


# ============================================================== learner
def _update(st: Dict[str, np.ndarray], row: Tuple[float, int, float, np.ndarray],
            w: float) -> Dict[str, np.ndarray]:
    """Fold one tick row into the bin of its own local hour. Deterministic in
    (state, row, w): rows older than the ring slot's hour are dropped, so a
    release of old held rows or a replay lands exactly where it belongs."""
    _ts, a, frac, x = row
    if not (w >= 0.0):
        return st
    sl = int(a) % NB
    hr = st["hour"]
    if hr[sl] != a:
        if hr[sl] > a:
            return st
        hr[sl] = a
        st["sx"][sl] = 0.0
        st["sw"][sl] = 0.0
        st["sc"][sl] = 0.0
        if "ox" in st:
            st["ox"][sl] = 0.0
            st["oc"][sl] = 0.0
    if w == 0.0:
        if "ox" in st:                      # observed but untrusted (round 4)
            st["ox"][sl] += x
            st["oc"][sl] += frac
        return st
    st["sx"][sl] += w * x
    st["sw"][sl] += w * frac
    st["sc"][sl] += frac
    return st


def _merge(own: Dict[str, np.ndarray], other: Dict[str, np.ndarray],
           w: float) -> Dict[str, np.ndarray]:
    """Link seeding B := B_own + w A, bin by bin (same local hour: sums add;
    A newer or own empty: A's bin at weight w)."""
    ho, ha = own["hour"], other["hour"]
    same = (ha >= 0) & (ho == ha)
    newer = (ha >= 0) & (ho < ha)
    own["sx"][same] += w * other["sx"][same]
    own["sw"][same] += w * other["sw"][same]
    own["sc"][same] = np.maximum(own["sc"][same], other["sc"][same])
    ho[newer] = ha[newer]
    own["sx"][newer] = w * other["sx"][newer]
    own["sw"][newer] = w * other["sw"][newer]
    own["sc"][newer] = other["sc"][newer]
    if "ox" in own:                         # a predecessor's untrusted rows stay its own
        own["ox"][newer] = 0.0
        own["oc"][newer] = 0.0
    return own


def _dump(st: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {"hour": st["hour"].copy(), "sx": st["sx"].astype(np.float32),
           "sw": st["sw"].astype(np.float32), "sc": st["sc"].astype(np.float32)}
    if "ox" in st:
        out["ox"], out["oc"] = st["ox"].astype(np.float32), st["oc"].astype(np.float32)
    return out


def _load(blob: Any) -> Dict[str, np.ndarray]:
    st = {"hour": np.array(blob["hour"], dtype=np.int64),
          "sx": np.array(blob["sx"], dtype=np.float64).reshape(NB, NQ),
          "sw": np.array(blob["sw"], dtype=np.float64),
          "sc": np.array(blob["sc"], dtype=np.float64)}
    st["ox"] = (np.array(blob["ox"], dtype=np.float64).reshape(NB, NQ) if "ox" in blob
                else np.zeros((NB, NQ)))
    st["oc"] = np.array(blob["oc"], dtype=np.float64) if "oc" in blob else np.zeros(NB)
    return st


# ============================================================ statistics
def _nanmed(x: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """(median ignoring NaN, finite count) along `axis`, NaN where empty;
    warning-free (np.nanmedian warns on all-NaN slices) and about 3x faster."""
    x = np.moveaxis(np.asarray(x, dtype=np.float64), axis, -1)
    shape = x.shape[:-1]
    n = x.shape[-1]
    x2 = x.reshape(-1, n)
    cnt = np.count_nonzero(~np.isnan(x2), axis=1)
    if n == 0:
        return np.full(shape, np.nan), cnt.reshape(shape)
    s = np.sort(x2, axis=1)                      # NaN last
    rows = np.arange(x2.shape[0])
    lo = np.maximum((cnt - 1) // 2, 0)
    hi = np.minimum(np.maximum(cnt // 2, 0), n - 1)
    med = 0.5 * (s[rows, lo] + s[rows, hi])
    med[cnt == 0] = np.nan
    return med.reshape(shape), cnt.reshape(shape)


def _sorted_quantile(rs: np.ndarray, p: Any) -> Any:
    """np.quantile(rs, p) (linear) for an already sorted 1-d array, without
    np.quantile's per-call overhead."""
    pos = np.asarray(p, dtype=np.float64) * (rs.size - 1)
    lo = np.floor(pos).astype(np.intp)
    hi = np.minimum(lo + 1, rs.size - 1)
    return rs[lo] + (pos - lo) * (rs[hi] - rs[lo])


def hourly_values(st: Mapping[str, np.ndarray], q: int, day0: int
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Committed rate per full hour of quantity q for the NHIST hours from
    local day day0 on (NaN: not covered), and the validity mask."""
    hours = day0 * 24 + np.arange(NHIST, dtype=np.int64)
    sl = hours % NB
    ok = (st["hour"][sl] == hours) & (st["sc"][sl] >= MIN_COV) & (st["sw"][sl] > 0.0)
    v = np.full(NHIST, np.nan)
    v[ok] = st["sx"][sl[ok], q] / st["sw"][sl[ok]]
    return v, ok


def window_sums(v: np.ndarray) -> np.ndarray:
    """W[H, day, phase]: horizon sums ending with each hour of the history
    (NaN when a window reaches before the history or holds an invalid hour).
    Row NH (auxiliary) is the rolling 24 h sum, which sets the 7-d scale."""
    nan = np.isnan(v)
    c = np.concatenate(([0.0], np.cumsum(np.where(nan, 0.0, v))))
    cn = np.concatenate(([0], np.cumsum(nan)))
    end = np.arange(1, NHIST + 1)
    out = np.empty((NH + 1, NHIST))
    for k, L in enumerate(H_LEN + (24,)):
        start = ((end - 1) // 24) * 24 if L == 0 else end - L
        good = start >= 0
        st0 = np.maximum(start, 0)
        w = c[end] - c[st0]
        w[~good | (cn[end] - cn[st0] > 0)] = np.nan
        out[k] = w
    return out.reshape(NH + 1, HIST_DAYS, 24)


def day_type_profile(v: np.ndarray, dtypes: np.ndarray) -> np.ndarray:
    """(2, 24) median hourly rate per day type (0 workday, 1 other) over the
    history; a type with fewer than 2 valid days takes the all-day profile.
    NaN where no day has a value for that hour."""
    vv = v.reshape(HIST_DAYS, 24)
    out = np.full((2, 24), np.nan)
    allp, _ = _nanmed(vv, 0)
    for t in (0, 1):
        sel = dtypes == t
        if sel.sum() >= 2:
            pt, n = _nanmed(vv[sel], 0)
            out[t] = np.where(n >= 2, pt, allp)
        else:
            out[t] = allp
    return out


def expected_7d(prof: np.ndarray, types: np.ndarray, h: int) -> float:
    """Expected 7-d sum ending with local hour h of the LAST day of `types`
    (8 day types, oldest first: the window is hours h+1..23 of the first day,
    the 6 full days between and hours 0..h of the last)."""
    P = np.nan_to_num(prof, nan=0.0)
    t = np.asarray(types, dtype=np.int64)
    return float(P[t[0], h + 1:].sum() + sum(P[t[k]].sum() for k in range(1, 7))
                 + P[t[7], :h + 1].sum())


def composition_7d(v: np.ndarray, dtypes: np.ndarray, unit: float
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Day-type composition of the history's 7-d windows (eval round 2).

    Returns (profile (2, 24), log correction (days, 24) to subtract from the
    7-d log windows, usual expected sum per phase (24,)). The 7-d horizon
    compared a window with the same-phase windows of 28 days, which almost
    always held 5 workdays: a make-up working Saturday (调休) put a 6th
    workday in the next seven 7-d windows, +20 % volume that was a 20-sigma
    residual (pack B: writes / bytes / objs 7-d p < 1e-6 on ~2,400 clean
    control ticks, all within a week of the make-up day). Each window is now
    measured against its own expected composition (the sum of the day-type
    hourly profiles over its hours) rescaled to the usual one, so a window
    with one more workday expects one more workday of volume."""
    prof = day_type_profile(v, dtypes)
    X = np.nan_to_num(prof[np.asarray(dtypes, dtype=np.int64)], nan=0.0).reshape(-1)
    c = np.concatenate(([0.0], np.cumsum(X)))
    end = np.arange(1, NHIST + 1)
    start = end - 168
    good = start >= 0
    E = np.where(good, c[end] - c[np.maximum(start, 0)], np.nan).reshape(HIST_DAYS, 24)
    ebar, _ = _nanmed(E, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.log(E + unit) - np.log(ebar[None, :] + unit)
    corr = np.where(np.isfinite(corr), corr, 0.0)
    return prof, corr, ebar


def _robust(sub: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(median, robust scale, n) along axis 1 of (NH, days, 24)."""
    med, n = _nanmed(sub, 1)
    dev = np.abs(sub - med[:, None, :])
    mad, _ = _nanmed(dev, 1)
    scale = 1.4826 * mad
    fin = ~np.isnan(dev)
    mabs = 1.2533 * np.where(n > 0, np.where(fin, dev, 0.0).sum(1) / np.maximum(n, 1), np.nan)
    scale = np.where(scale > 0.0, scale, mabs)
    return med, scale, n


def phase_stats(L: np.ndarray, dtypes: np.ndarray, min_n: int
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Median and robust scale per (horizon, day_type, phase) of LOG history
    windows log(W + unit) (or log ratios). Unfloored.

    Why logs: budgets are multiplicative (a phase whose 8-h window is one
    busy lognormal hour is far more skewed than a phase summing 24 hours),
    and pooled standardised residuals are only exchangeable across phases if
    the skew is gone; in logs a spread is a relative spread, so a young
    entity can take its peers' scales as they are.

    Median: same day_type with >= min_n samples, else all days (7 d: always
    all days). Scale: a MAD needs days, not phases, so a day type with fewer
    than MIN_SCALE_N days (weekends: ~6 of 20) takes the spread pooled over
    all days of their deviations from their own day type's median; then a
    running median over +-PHASE_SMOOTH neighbouring phases (clamped, not
    circular: day-to-date restarts at midnight). Noisy per-phase scales give
    the pooled residuals a t-like tail (measured: 1-h z_q ~ 15 instead of ~5).

    Same-phase 7-d windows of consecutive days share 6 of their 7 days, so
    their spread across 28 days reflects ~4 independent weeks and grossly
    understates the scale. With the auxiliary rolling-24 h row (L[NH]), the
    7-d log scale is at least the robust log scale of the 24-h sums about
    their day-type median / sqrt(7) (days are the independent units)."""
    aux = L[NH] if L.shape[0] > NH else None
    L = L[:NH]
    med = np.full((NH, 2, 24), np.nan)
    own = np.full((NH, 2, 24), np.nan)
    n_own = np.zeros((NH, 2, 24), dtype=np.int64)
    m_a, s_a, n_a = _robust(L)
    ok_a = n_a >= min_n
    for t in (0, 1):
        sel = dtypes == t
        if sel.any():
            m_t, s_t, n_t = _robust(L[:, sel, :])
            ok_t = n_t >= min_n
            ok_t[H_7D] = False
        else:
            m_t = s_t = np.full((NH, 24), np.nan)
            n_t = np.zeros((NH, 24), dtype=np.int64)
            ok_t = np.zeros((NH, 24), dtype=bool)
        med[:, t] = np.where(ok_t, m_t, np.where(ok_a, m_a, np.nan))
        own[:, t] = np.where(ok_t, s_t, np.where(ok_a, s_a, np.nan))
        n_own[:, t] = np.where(ok_t, n_t, np.where(ok_a, n_a, 0))
    pooled = _robust(L - med[:, dtypes, :])[1][:, None, :]
    scale = np.where(n_own >= MIN_SCALE_N, own, pooled)
    scale = np.where(np.isnan(med), np.nan, scale)
    scale = np.where(np.isnan(scale), np.nan, _nanmed(scale[..., _SMOOTH_IDX], -2)[0])
    if aux is not None:
        dev24 = daily_deviations(aux, dtypes)
        s7 = 1.4826 * _nanmed(np.abs(dev24 - _nanmed(dev24, 0)[0]), 0)[0] / SQRT7
        with np.errstate(invalid="ignore"):
            scale[H_7D] = np.where(np.isnan(s7), scale[H_7D], np.fmax(scale[H_7D], s7))
    return med, scale


def daily_deviations(aux: np.ndarray, dtypes: np.ndarray) -> np.ndarray:
    """dev[days, 24]: rolling 24-h log sums about their day-type median (the
    series phase_stats' 7-d scale floor is built from)."""
    m24 = np.full((2, 24), np.nan)
    for t in (0, 1):
        sel = dtypes == t
        if sel.any():
            m24[t] = _nanmed(aux[sel], 0)[0]
    m24 = np.where(np.isnan(m24), _nanmed(aux, 0)[0], m24)
    return aux - m24[dtypes]


def seven_day_pinf(L: np.ndarray, dtypes: np.ndarray) -> np.ndarray:
    """Predictive inflation [24] of the 7-d scale (round 4): sqrt(1 + (pi/2)
    / n_ind), n_ind = n_days / 7 (>= 1) with n_days the days whose rolling
    24-h sum is valid at that phase. 1 without the auxiliary row.

    Why: the live 7-d residual is standardised by the same-phase median and
    scale of the history's 7-d windows, and its tail p is read off those
    windows' own (in-sample) residuals. Consecutive 7-d windows share 6 days,
    so the median rests on ~n_days / 7 independent weeks and its error
    (variance (pi/2) sigma^2 / n_ind for a median) is part of a new window's
    residual but of none of the in-sample ones. Pack A seed 0 live (16-20
    days of history): the 7-d cells had p < 1e-3 on 11.9x the nominal share
    of clean control cells, the 1-h / 8-h / day cells 1.2-1.4x. Simulated
    daily-regime histories (18 days, day-effect lag-1 correlation 0 / 0.5 /
    0.8, 40 seeds): p < 1e-2 on 2.5 / 2.9 / 4.0x nominal without the term,
    0.6 / 0.6 / 0.7x with it (p < 1e-3: <= 0.3x). Only the 7-d horizon is
    affected (its windows overlap; the others do not)."""
    if L.shape[0] <= NH:
        return np.ones(24)
    dev = daily_deviations(L[NH], dtypes)
    n_ind = np.maximum(np.sum(np.isfinite(dev), axis=0) / 7.0, 1.0)
    return np.sqrt(1.0 + 0.5 * math.pi / n_ind)


_SMOOTH_IDX = np.clip(np.arange(24)[None, :] + np.arange(-PHASE_SMOOTH, PHASE_SMOOTH + 1)[:, None],
                      0, 23)                      # (2k+1, 24) clamped neighbour phases


def floor_scale(med: np.ndarray, scale: np.ndarray, floor: float = LOG_SCALE_MIN,
                count_unit: Optional[float] = None) -> np.ndarray:
    """Log scales never below `floor` (a 2 % relative spread) and, for a
    COUNT quantity (count_unit = its log offset), never below the counting
    noise of its level: L = log(W + unit) of a count W ~ Poisson(m) has
    sd ~ sqrt(m + 1) / (m + 1 + unit) (the +1 keeps a level of 0 finite).

    Why: a count quantity whose history is constant at one phase (0 writes
    at 03:00, 5 destinations in 7 d, 4 active slots in an hour) has a robust
    log scale of ~0, floored at 0.02, so one more request, destination or
    slot was a 20-50 sigma residual and a tail p of 1e-10 ... 1e-38 (pack B:
    budget_breadth / budget_vol p < 1e-6 on ~400 / ~250 clean control ticks,
    against 0.03 expected; the pooled tails were fitted with z_q of 20-270
    robust-scale units). Bytes quantities are unaffected (their level makes
    this floor negligible)."""
    with np.errstate(invalid="ignore", over="ignore"):
        fl = floor
        if count_unit is not None:
            u = float(count_unit)
            m = np.maximum(np.exp(med) - u, 0.0)
            fl = np.maximum(floor, np.sqrt(m + 1.0) / (m + 1.0 + u))
        out = np.where(np.isnan(scale), np.nan, np.maximum(scale, fl))
    return np.where(np.isnan(med), np.nan, out)


def log_windows(W: np.ndarray, unit: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.log(np.maximum(W, 0.0) + unit)


def fit_tail(r: np.ndarray, min_pool: int) -> Optional[np.ndarray]:
    """[u, xi, sigma, rate, z_q] + body quantiles of pooled standardised
    residuals, or None with too few of them (see the module doc)."""
    r = r[np.isfinite(r)]
    n = r.size
    if n < min_pool:
        return None
    rs = np.sort(r)
    u = float(_sorted_quantile(rs, TAIL_U_Q))
    y = rs[rs > u] - u
    if y.size < 3:
        u2 = float(_sorted_quantile(rs, TAIL_U_FALLBACK))
        y2 = rs[rs > u2] - u2
        if y2.size >= 3:
            u, y = u2, y2
    if y.size >= 3:
        xi_raw, _sig = evt.gpd_pwm_fit(y)
        xi = min(XI_MAX, max(0.0, xi_raw * y.size / (y.size + XI_SHRINK_N)))
        sigma = max(float(y.mean()) * (1.0 - xi), SIGMA_MIN)
        rate = y.size / n
    else:                                   # (near-)constant history: exponential, +1 rate
        xi, sigma, rate = 0.0, SIGMA_MIN, (y.size + 1.0) / (n + 1.0)
    zq = evt.pot_quantile(u, xi, sigma, rate, Q_EVAL)
    body = _sorted_quantile(rs, PQ)
    return np.concatenate(([u, xi, sigma, rate, zq], body))


def _min_zq(t: np.ndarray, z_min: float) -> np.ndarray:
    """Widen a fitted tail (sigma) so that its level-q quantile is at least
    z_min, keeping tail_p and z_q consistent (used for the 7-d horizon, whose
    few effectively independent weeks cannot support a thinner tail than
    the CLT normal one)."""
    u, xi, sig, rate, zq = (float(v) for v in t[:5])
    if not (zq < z_min) or not (zq > u) or u >= z_min:
        return t
    t = t.copy()
    t[2] = sig * (z_min - u) / (zq - u)
    t[4] = evt.pot_quantile(u, xi, t[2], rate, Q_EVAL)
    return t


def tail_p(r: np.ndarray, tail: np.ndarray, body: np.ndarray) -> np.ndarray:
    """Tail p of standardised values r (any shape S) under per-cell POT
    parameters tail[S + (5,)] and body quantiles body[S + (K,)]: rate x GPD
    sf above u (xi >= 0 by construction), the empirical body below. NaN where
    r or the fit is NaN."""
    u, xi, sig, rate = tail[..., 0], tail[..., 1], tail[..., 2], tail[..., 3]
    p = np.full(r.shape, np.nan)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        fin = np.isfinite(r) & np.isfinite(u) & (sig > 0.0) & np.isfinite(rate)
        up = fin & (r > u)
        if up.any():
            y, x, s = r[up] - u[up], xi[up], sig[up]
            z = y / s
            sf = np.where(x < 1e-9, np.exp(-z), np.exp(-np.log1p(x * z) / np.maximum(x, 1e-9)))
            p[up] = np.maximum(rate[up] * sf, P_FLOOR)
        lo = fin & ~up
        if lo.any():
            bb, rr = body[lo], r[lo]
            k = np.sum(bb < rr[:, None], axis=1)
            kk = np.clip(k, 1, PQ.size - 1)
            b0 = np.take_along_axis(bb, (kk - 1)[:, None], 1)[:, 0]
            b1 = np.take_along_axis(bb, kk[:, None], 1)[:, 0]
            frac = np.where(b1 > b0, (rr - b0) / np.where(b1 > b0, b1 - b0, 1.0), 1.0)
            P = PQ[kk - 1] + (PQ[kk] - PQ[kk - 1]) * np.clip(frac, 0.0, 1.0)
            P = np.where(k == 0, 0.0, np.where(k >= PQ.size, PQ[-1], P))
            p[lo] = 1.0 - P
    return p


# ================================================================ helpers
def _abs_floors(cfg: Mapping[str, Any]) -> np.ndarray:
    over = cfg.get("budget_abs_floor") or {}
    if not isinstance(over, Mapping):
        raise ValueError(f"config budget_abs_floor must be a mapping, got {type(over).__name__}")
    out = np.empty(NQ)
    for i, q in enumerate(Q_NAMES):
        v = over.get(q, ABS_FLOOR_DEFAULT[q])
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) \
                or not (float(v) >= 0.0 and math.isfinite(float(v))):
            raise ValueError(f"config budget_abs_floor[{q!r}] must be a finite number >= 0")
        out[i] = float(v)
    return out


def _utc_offset(ts: float, tz: str) -> float:
    off = TB.local_datetime(ts, tz).utcoffset()
    return 0.0 if off is None else off.total_seconds()


def _fmt(v: float, unit: str) -> str:
    if not math.isfinite(v):
        return "n/a"
    if unit == "bytes":
        for div, u in ((1e9, "GB"), (1e6, "MB"), (1e3, "KB")):
            if abs(v) >= div:
                return f"{v / div:.1f} {u}"
        return f"{v:.0f} B"
    return f"{v:.0f} {unit}"


def _j(v: Any, nd: int = 4) -> Optional[float]:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _jp(v: Any) -> Optional[float]:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return float(f"{v:.4g}") if math.isfinite(v) else None


_SERIES_KEYS = [f"{q}.{hn}" for q in Q_NAMES for hn in H_NAMES]


def _series_value(res: Mapping[str, np.ndarray]) -> Dict[str, List[Optional[float]]]:
    """behavior.budget {'<Q>.<H>': [value, z_q, tail_p]} (NaN -> None)."""
    with np.errstate(invalid="ignore"):
        b = np.round(res["B"], 3).ravel().tolist()
        z = np.round(res["zthr"], 3).ravel().tolist()
    p = res["p"].ravel().tolist()
    return {k: [bv if bv == bv else None, zv if zv == zv else None,
                float(f"{pv:.4g}") if pv == pv else None]
            for k, bv, zv, pv in zip(_SERIES_KEYS, b, z, p)}


def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
        return None
    return float(v)


def _is_rare(did: int, rare_ids: Set[int], store: Any, s: str, now: float) -> bool:
    """Rare destination (used by <= 20 % of the system's entities). Private on
    purpose, the one place that picks the source: R2 already flags rare
    destinations in act.rare_events; otherwise model.vocab's prevalence
    (B08, lib/m_vocab, when that accessor is present), else R2's decayed
    distinct-entity share (lib/m_template), the same notion."""
    if did in rare_ids:
        return True
    share = math.nan
    if _MV is not None:
        name = MT.dest_name(store, s, did)
        if name:
            share = _MV.dest_prevalence(store, s, name, now)
    if share != share:
        share = MT.dest_prevalence(store, s, did, now)
    return share == share and share <= RARE_SHARE


def _actor_chains(link: Any, s: str) -> List[List[str]]:
    """model.link.actors (B17) as entity lists of system s. Accepted shapes:
    {id: [members] | {'members'|'entities'|'chain': [...]}} or a list of
    either; a member is 'ip', 'sys|ip' or {'system', 'entity'}."""
    if not isinstance(link, Mapping):
        return []
    acts = link.get("actors")
    items: Iterable[Any] = acts.values() if isinstance(acts, Mapping) else (
        acts if isinstance(acts, (list, tuple)) else ())
    out: List[List[str]] = []
    for it in items:
        mem = it
        if isinstance(it, Mapping):
            mem = it.get("members") or it.get("entities") or it.get("chain") or ()
        if not isinstance(mem, (list, tuple)):
            continue
        chain: List[str] = []
        for m in mem:
            if isinstance(m, Mapping):
                if str(m.get("system", s)) != s or not m.get("entity"):
                    continue
                ent = str(m["entity"])
            elif isinstance(m, str):
                if "|" in m:
                    sy, _, ent = m.partition("|")
                    if sy != s:
                        continue
                else:
                    ent = m
            else:
                continue
            if ent and ent not in chain:
                chain.append(ent)
        if len(chain) >= 2:
            out.append(chain)
    return out


def _link_sources(link: Any, e: str) -> List[str]:
    """Entities linked INTO e (their destination memory is not new to e)."""
    if not isinstance(link, Mapping):
        return []
    links = link.get("links")
    items = links.values() if isinstance(links, Mapping) else (links or ())
    out = []
    for lk in items:
        if isinstance(lk, Mapping) and lk.get("to") == e and lk.get("from") \
                and not (lk.get("retracted") or lk.get("status") == "retracted"
                         or lk.get("state") == "retracted" or lk.get("active") is False):
            out.append(str(lk["from"]))
    return out


# ================================================================ engine
class _Rec:
    """One entity's per-tick work item."""
    __slots__ = ("s", "e", "model", "persist", "ok", "B", "qnan", "degraded", "res",
                 "act_res", "peers", "pkey")

    def __init__(self, s: str, e: str, model: Dict[str, Any], persist: bool = True) -> None:
        self.s, self.e, self.model, self.persist = s, e, model, persist
        self.ok = False
        self.B: Optional[np.ndarray] = None          # (NH, NQ) horizon sums
        self.qnan = np.zeros(NQ, dtype=bool)          # quantity unknown this tick
        self.degraded: Dict[str, str] = {}
        self.res: Optional[Dict[str, np.ndarray]] = None
        self.act_res: Optional[Tuple[List[str], Dict[str, np.ndarray]]] = None
        self.peers: List[str] = []
        self.pkey = SYSTEM_KEY


class BudgetEngine(Engine):
    name = "behavior.budget"
    layer = "behavior"
    consumes = [NAT, ACTIVE, "act.objs", "act.stream", "act.stream_frac", "act.rare_events",
                "act.tokens", "dns.qname_set", "model.template", "model.vocab", "model.class",
                LINK_MODEL, "behavior.trust", "behavior.trust_prov", "behavior.quarantine",
                "model.control"]
    produces = [MODEL, "behavior.score", "behavior.pm", "behavior.acc_alarm", "behavior.axes",
                "behavior.degraded", SERIES, "profile.extra.budget", "event.budget_exceeded"]
    description = ("Long-horizon cumulative budgets (volume, exfil to novel destinations, "
                   "breadth via HLL) over 1 h / 8 h / day-to-date / 7 d against same-phase "
                   "hourly history: POT/GPD thresholds, absolute floors, common-mode peer "
                   "guard, actor aggregation; trust-gated, rollback-aware bins.")
    interval = 1

    def __init__(self, fit_max_per_tick: int = FIT_MAX_PER_TICK, **params: Any) -> None:
        super().__init__(**params)
        self.fit_max_per_tick = int(fit_max_per_tick)
        self._rows_cur: Optional[_Rows] = None
        self._learner = G.GatedLearner(
            name=LEARNER, init=new_state, update=_update, fetch=self._fetch,
            dump=_dump, load=_load, merge=_merge)
        self._ret_store: Optional[weakref.ref] = None
        self._dtype_cache: Dict[Tuple[int, Any], int] = {}
        self._pcache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

    # ------------------------------------------------------------------ run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        now, dt = float(ctx.now), float(ctx.window_s)
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError(f"BudgetEngine: bad ctx.window_s {ctx.window_s!r}")
        self._ensure_retention(store)
        self._learner.d_min_s = float(ctx.config.get("D_min_s") or G.D_MIN_S)
        tz = str(ctx.config.get("tz") or TB.DEFAULT_TZ)
        cal = TB.parse_calendar(ctx.config.get("calendar"))
        floors = _abs_floors(ctx.config)
        t_mid = now - 0.5 * dt                      # the tick covers (now - dt, now]
        off = _utc_offset(t_mid, tz)
        a = int(math.floor((t_mid + off) / 3600.0))
        today = a // 24
        tick = {"now": now, "dt": dt, "tz": tz, "cal": cal, "off": off, "a": a,
                "today": today, "h": a % 24, "dti": self._dtype(today, cal),
                "types8": np.array([self._dtype(today - k, cal) for k in range(7, -1, -1)]),
                "floors": floors, "training": bool(ctx.training),
                "org": list(ctx.config.get("org_domains") or []),
                "b01_failed": store.engine_failed(B01_ENGINE, now),
                "r2_failed": store.engine_failed(R2_ENGINE, now)}
        if len(self._pcache) > 4096:
            self._pcache.clear()
        n = 0
        fit_budget = [self.fit_max_per_tick]
        for s in store.systems():
            recs = [r for r in (self._entity(ctx, s, e, tick) for e in store.entities(s))
                    if r is not None]
            if not recs:
                continue
            self._peer_sets(store, s, recs)
            self._fits(store, s, recs, tick, fit_budget)
            self._evaluate_system(store, s, recs, tick)
            for r in recs:
                if r.persist:
                    self._learn(ctx, r, tick)
                n += self._write(ctx, r, tick)
        return n

    def _ensure_retention(self, store: Any) -> None:
        """behavior.budget: the newest points only (36 [value, z_q, p] entries
        each; contract B, store default). The explicit cap stays: it is a
        memory bound on this engine's own output, not an input requirement."""
        if self._ret_store is None or self._ret_store() is not store:
            store.set_retention(SERIES, SERIES_KEEP_POINTS, SERIES_KEEP_S)
            self._ret_store = weakref.ref(store)

    def _dtype(self, day: int, cal: Any) -> int:
        key = (day, id(cal))
        v = self._dtype_cache.get(key)
        if v is None:
            if len(self._dtype_cache) > 4096:
                self._dtype_cache.clear()
            d = _EPOCH + _dt.timedelta(days=int(day))
            v = self._dtype_cache[key] = 0 if TB.day_type(d, cal) == "workday" else 1
        return v

    # ------------------------------------------------------------- per entity
    def _entity(self, ctx: Context, s: str, e: str, tick: Dict[str, Any]) -> Optional[_Rec]:
        store, now = ctx.store, tick["now"]
        model = store.get_model(s, e, MODEL)
        nat = store.vec_at(s, e, NAT, now)
        if not isinstance(model, dict) or model.get("fmt") != FMT:
            if nat is None:
                if tick["b01_failed"] or self._stale(store, s, e, NAT, now):
                    rec = _Rec(s, e, new_model(), persist=False)
                    rec.degraded = {d: self._b01_cause(tick) for d in AXIS_DETECTOR.values()}
                    return rec                            # contract M before the first model
                return None                               # not an entity B01 knows yet
            model = new_model()
        rec = _Rec(s, e, model)
        live, rows = model["live"], model["rows"]
        if nat is None:
            rec.degraded = {d: self._b01_cause(tick) for d in AXIS_DETECTOR.values()}
            return rec
        if live["tick_ts"] == now:                       # re-run of this tick: same row
            x = live["tick_x"].copy()
        else:
            self._roll_day(live, tick["today"])
            x = self._quantities(store, s, e, np.asarray(nat, dtype=np.float64), live, tick)
            fin = np.where(np.isnan(x), 0.0, x)
            self._live_add(live, tick["a"], fin)
            rows.append(now, tick["a"], tick["dt"] / 3600.0, x.astype(np.float32))
            live["tick_ts"], live["tick_x"] = now, x.copy()
        rec.qnan = np.isnan(x)
        if tick["r2_failed"]:
            cause = f"producer_error:{R2_ENGINE}"
            rec.degraded.update({AXIS_DETECTOR["exfil"]: cause, AXIS_DETECTOR["breadth"]: cause})
        for ax in AXES:
            if rec.qnan[AXIS_Q[ax]].any() and AXIS_DETECTOR[ax] not in rec.degraded:
                rec.degraded[AXIS_DETECTOR[ax]] = "nan_input:" + ",".join(
                    Q_NAMES[i] for i in AXIS_Q[ax] if rec.qnan[i])
        rec.B = self._horizons(live, tick["a"])
        rec.ok = True
        return rec

    @staticmethod
    def _stale(store: Any, s: str, e: str, name: str, now: float) -> bool:
        lw = store.last_write_ts(s, e, name)
        return lw is not None and lw < now

    @staticmethod
    def _b01_cause(tick: Dict[str, Any]) -> str:
        return f"producer_error:{B01_ENGINE}" if tick["b01_failed"] else f"stale:{NAT}"

    @staticmethod
    def _roll_day(live: Dict[str, Any], today: int) -> None:
        if live["day"] != today:
            live["day"] = today
            live["objs"], live["tmpl"], live["dests"] = {}, set(), set()

    @staticmethod
    def _live_add(live: Dict[str, Any], a: int, x: np.ndarray) -> None:
        sl = a % LIVE_HOURS
        if live["bh"][sl] != a:
            if live["bh"][sl] > a:
                return
            live["bh"][sl] = a
            live["bx"][sl] = 0.0
        live["bx"][sl] += x

    @staticmethod
    def _horizons(live: Dict[str, Any], a: int) -> np.ndarray:
        """B[H, Q]: live sums over the hours ending with the current hour a."""
        age = a - live["bh"]
        ok = (live["bh"] >= 0) & (age >= 0)
        day0 = (a // 24) * 24
        M = np.empty((NH, LIVE_HOURS), dtype=np.float64)
        M[0] = ok & (age < 1)
        M[1] = ok & (age < 8)
        M[2] = ok & (live["bh"] >= day0)
        M[3] = ok & (age < 168)
        return M @ live["bx"]

    # ------------------------------------------------------------ quantities
    def _quantities(self, store: Any, s: str, e: str, nat: np.ndarray,
                    live: Dict[str, Any], tick: Dict[str, Any]) -> np.ndarray:
        now = tick["now"]
        x = np.zeros(NQ)
        up, down = nat[_I_UP], nat[_I_DOWN]
        x[QI["bytes_up"]] = up
        x[QI["bytes_down"]] = down
        req, wr = nat[_I_REQ], nat[_I_WR]
        # the write ratio is NaN exactly when there were no requests: 0 writes
        if not req == req:
            x[QI["writes"]] = _NAN
        elif req > 0.0:
            x[QI["writes"]] = wr * req if wr == wr else _NAN   # R1 writes both together
        else:
            x[QI["writes"]] = 0.0                     # no requests: the ratio is NaN, 0 writes
        rows = MT.stream_rows(store, s, e, now)
        x[QI["slots"]] = self._slots(store, s, e, rows, live, tick)
        if tick["r2_failed"]:
            x[ACT_Q] = _NAN
            return x
        rare = MT.rare_events(store, s, e, now)
        up_by = self._up_by_dest(store, s, e, rows, rare, now)
        rare_ids = set(rare.keys())
        tick_dests = set(up_by.keys()) | rare_ids
        if rows.size:
            d = rows["dest_id"]
            tick_dests.update(int(v) for v in np.unique(d[d != 0]))
        first = live["dest_first"]
        org = tick["org"]
        novel_up = 0.0
        for did in tick_dests:
            ent = first.get(did)
            if ent is None:
                first[did] = [now, now]
            else:
                ent[1] = now
        for did, b in up_by.items():
            if b <= 0.0:
                continue
            if (now - first[did][0] < NOVEL_S or _is_rare(did, rare_ids, store, s, now)) \
                    and self._external(store, s, did, org):
                novel_up += b
        x[QI["up_novel"]] = novel_up
        x[QI["dns_label"]] = self._dns_label(store, s, e, now, first, rare_ids, org)
        self._sweep_dests(live, now)
        x[QI["objs"]] = self._objs(store, s, e, now, live)
        x[QI["templates"]] = self._templates(store, s, e, now, live)
        ds = live["dests"]
        before = len(ds)
        if len(ds) < DAY_SET_CAP:
            ds.update(tick_dests)
        x[QI["dests"]] = float(len(ds) - before)
        return x

    @staticmethod
    def _slots(store: Any, s: str, e: str, rows: np.ndarray, live: Dict[str, Any],
               tick: Dict[str, Any]) -> float:
        """New active 15-min local slots this tick (event timestamps, else the
        tick's last slot when feature.active = 1)."""
        off, last = tick["off"], live["last_slot"]
        t = rows["ts"] if rows.size else None
        if t is not None:
            t = t[np.isfinite(t)]
        if t is not None and t.size:
            sl = np.unique(np.floor((t + off) / TB.SLOT_S).astype(np.int64))
        else:
            act = store.vec_at(s, e, ACTIVE, tick["now"])
            if act is None or not float(act[0]) > 0.0:
                return 0.0
            sl = np.array([int(math.floor((tick["now"] - 1e-3 + off) / TB.SLOT_S))])
        new = sl[sl > last]
        if new.size:
            live["last_slot"] = int(new[-1])
        return float(new.size)

    @staticmethod
    def _up_by_dest(store: Any, s: str, e: str, rows: np.ndarray,
                    rare: Mapping[int, np.ndarray], now: float) -> Dict[int, float]:
        """Bytes up per destination this tick: rare destinations from
        act.rare_events (complete), the others from act.stream / stream_frac."""
        out: Dict[int, float] = {}
        for did, ev in rare.items():
            if ev.size:
                v = ev[:, 1]
                out[int(did)] = float(np.sum(v[np.isfinite(v) & (v > 0)]))
        if rows.size:
            frac = MT.stream_frac(store, s, e, now)
            w = 1.0 / frac if (frac == frac and 0.0 < frac <= 1.0) else 1.0
            d = rows["dest_id"].astype(np.int64)
            upv = rows["up"].astype(np.float64)
            m = (d != 0) & np.isfinite(upv) & (upv > 0)
            if rare:
                m &= ~np.isin(d, np.fromiter(rare.keys(), dtype=np.int64, count=len(rare)))
            if m.any():
                u, inv = np.unique(d[m], return_inverse=True)
                sums = np.bincount(inv, weights=upv[m]) * w
                for did, b in zip(u.tolist(), sums.tolist()):
                    out[int(did)] = out.get(int(did), 0.0) + b
        return out

    @staticmethod
    def _external(store: Any, s: str, did: int, org: Sequence[str]) -> bool:
        """External destination; a name R2 never recorded counts as external
        (an exfil budget errs towards counting)."""
        name = MT.dest_name(store, s, did)
        return True if not name else NM.is_external(name, org)

    @staticmethod
    def _dns_label(store: Any, s: str, e: str, now: float, first: Dict[int, List[float]],
                   rare_ids: Set[int], org: Sequence[str]) -> float:
        qs = store.latest_fresh(s, e, "dns.qname_set", now)
        if not isinstance(qs, Mapping) or not qs:
            return 0.0
        tot = 0.0
        for qn, n in qs.items():
            c = _num(n.get("n") if isinstance(n, Mapping) else n)
            if qn == MT.OTHER or not isinstance(qn, str) or c is None or not c > 0:
                continue
            left, reg = NM.split_host(qn)
            if not left or not reg:
                continue
            did = MT.dest_id_of(reg)
            ent = first.get(did)
            if ent is None:
                ent = first[did] = [now, now]
            else:
                ent[1] = now
            if (now - ent[0] < NOVEL_S or _is_rare(did, rare_ids, store, s, now)) \
                    and NM.is_external(reg, org):
                tot += len(".".join(left)) * c
        return tot

    @staticmethod
    def _sweep_dests(live: Dict[str, Any], now: float) -> None:
        first = live["dest_first"]
        if len(first) <= DEST_CAP and now - live["dest_sweep"] < 86400.0:
            return
        live["dest_sweep"] = now
        for did in [k for k, v in first.items() if now - v[1] > DEST_KEEP_S]:
            del first[did]
        if len(first) > DEST_CAP:                        # oldest-seen out
            keep = sorted(first.items(), key=lambda kv: kv[1][1])[-DEST_CAP:]
            live["dest_first"] = dict(keep)

    @staticmethod
    def _objs(store: Any, s: str, e: str, now: float, live: Dict[str, Any]) -> float:
        ob = MT.objs(store, s, e, now)
        if not ob:
            return 0.0
        day: Dict[str, _Distinct] = live["objs"]
        inc = 0.0
        for key, entry in ob.items():
            if not isinstance(entry, Mapping):
                continue
            ids = entry.get("ids") or []
            hll = entry.get("hll")
            k = key if (key in day or len(day) < OBJ_TMPL_CAP) else OVERFLOW
            dd = day.get(k)
            if dd is None:
                dd = day[k] = _Distinct()
            n0 = dd.n
            if k == OVERFLOW and not hll:
                ids = [f"{key}\x1f{i}" for i in ids]
            dd.add(ids, hll if hll else None)
            inc += dd.n - n0
        return inc

    @staticmethod
    def _templates(store: Any, s: str, e: str, now: float, live: Dict[str, Any]) -> float:
        toks = store.latest_fresh(s, e, "act.tokens", now)
        if not isinstance(toks, Mapping) or not toks:
            return 0.0
        ts: Set[str] = live["tmpl"]
        before = len(ts)
        for tok in toks:
            if tok != MT.OTHER and isinstance(tok, str) and len(ts) < DAY_SET_CAP:
                ts.add(MT.template_key(tok))
        return float(len(ts) - before)

    # ------------------------------------------------------------------ peers
    def _peer_sets(self, store: Any, s: str, recs: List[_Rec]) -> None:
        """Class members (contract L: role with >= 3 members in s), else the
        system's entities, minus the entity itself."""
        ents = [r.e for r in recs]
        memo: Dict[str, List[str]] = {}
        for r in recs:
            key = m_class.class_key(store, s, r.e)
            if key is None:
                r.pkey, r.peers = SYSTEM_KEY, [x for x in ents if x != r.e]
                continue
            mem = memo.get(key)
            if mem is None:
                mem = memo[key] = m_class.class_members(store, s, key)
            r.pkey, r.peers = key, [x for x in mem if x != r.e]

    def _peer_windows(self, store: Any, s: str, pkey: str, members: Sequence[str], q: int,
                      day0: int, now: float) -> Dict[str, np.ndarray]:
        """{member: W (NH, days, 24)} from the members' committed rings,
        cached per (system, peer set, quantity) for REFIT_S within a day."""
        ck = (s, pkey, q)
        c = self._pcache.get(ck)
        if c is None or c["day0"] != day0 or now - c["ts"] >= REFIT_S or now < c["ts"]:
            W: Dict[str, np.ndarray] = {}
            for m in members:
                mod = store.get_model(s, m, MODEL)
                if isinstance(mod, dict) and mod.get("fmt") == FMT:
                    v, _ok = hourly_values(mod["state"], q, day0)
                    W[m] = window_sums(v)
            c = self._pcache[ck] = {"day0": day0, "ts": now, "W": W, "pmed": None}
        return c["W"]

    def _peer_median(self, s: str, pkey: str, q: int, e: str, peers: Sequence[str],
                     Wp: Mapping[str, np.ndarray]) -> Optional[np.ndarray]:
        """Median of the peers' history windows (NaN below MIN_PEERS). Exact
        leave-self-out for small peer sets; above PEER_EXACT_MAX members the
        all-member median is shared by the whole set (cached with the
        windows; one member moves it by at most one order statistic)."""
        others = [Wp[m] for m in peers if m in Wp]
        if len(others) < MIN_PEERS:
            return None
        if len(Wp) <= PEER_EXACT_MAX:
            pm, cnt = _nanmed(np.stack(others), 0)
            return np.where(cnt >= MIN_PEERS, pm, np.nan)
        c = self._pcache[(s, pkey, q)]
        if c["pmed"] is None:
            pm, cnt = _nanmed(np.stack(list(Wp.values())), 0)
            c["pmed"] = np.where(cnt >= MIN_PEERS + 1, pm, np.nan)
        return c["pmed"]

    # ------------------------------------------------------------------- fits
    def _fits(self, store: Any, s: str, recs: List[_Rec], tick: Dict[str, Any],
              budget: List[int]) -> None:
        """Round-robin refits: the history window moves by one day per day,
        so each quantity is refitted once per local day (and at once after a
        rollback / release / rebase or a link seed). Stalest first, at most
        max(1, ceil(NQ dt / FIT_SPREAD_S)) per entity and a per-tick share
        that spreads the day's fits over about FIT_SPREAD_S, whatever the
        cadence (capped by fit_max_per_tick)."""
        now, today = tick["now"], tick["today"]
        per_ent = max(1, math.ceil(NQ * tick["dt"] / FIT_SPREAD_S))
        work = []
        total = 0
        for r in recs:
            if not r.ok:
                continue
            fit = r.model["fit"]
            due = [q for q in range(NQ) if fit["day"][q] != today
                   or not (0.0 <= now - fit["ts"][q] < REFIT_S)]
            if due:
                due.sort(key=lambda q: fit["ts"][q])
                last = float(np.max(fit["ts"]))           # fairness: least recently served
                work.append(((float(fit["ts"][due[0]]), last), r.e, r, due[:per_ent]))
                total += len(due)
        work.sort(key=lambda w: (w[0], w[1]))
        n_fit = min(budget[0], max(FIT_MIN_PER_TICK,
                                   math.ceil(total * tick["dt"] / FIT_SPREAD_S)))
        dtypes = None
        for _t, _e, r, qs in work:
            for q in qs:
                if n_fit <= 0:
                    return
                n_fit -= 1
                budget[0] -= 1
                if dtypes is None:
                    dtypes = np.array([self._dtype(today - HIST_DAYS + d, tick["cal"])
                                       for d in range(HIST_DAYS)], dtype=np.int64)
                self._fit_q(store, s, r, q, tick, dtypes)

    def _fit_q(self, store: Any, s: str, r: _Rec, q: int, tick: Dict[str, Any],
               dtypes: np.ndarray) -> None:
        now, today = tick["now"], tick["today"]
        day0 = today - HIST_DAYS
        fit = r.model["fit"]
        unit = float(tick["floors"][q]) * UNIT_FRAC
        v, ok = hourly_values(r.model["state"], q, day0)
        n_days = int(np.sum(ok.reshape(HIST_DAYS, 24).sum(1) >= DAY_MIN_HOURS))
        fit["ts"][q], fit["day"][q], fit["days"][q] = now, today, n_days
        src, med, scale, tl = SRC_NONE, None, None, None
        pool = None
        if n_days < MIN_OWN_DAYS:
            pool = self._pool(store, s, r.peers, q, unit)
            if pool is None:                              # nothing to score with yet
                self._clear_fit(fit, q)
                return
        W = window_sums(v)
        L = log_windows(W, unit)
        prof, corr7, e7bar = composition_7d(v, dtypes, unit)
        L[H_7D] = L[H_7D] - corr7                         # each window at the usual composition
        cu = unit if Q_UNIT[q] != "bytes" else None      # counting-noise floor (counts)
        if n_days >= MATURE_DAYS:
            src = SRC_OWN
            med, scale = phase_stats(L, dtypes, MIN_PHASE_N)
            scale = floor_scale(med, scale, count_unit=cu)
            tl = self._own_tail(L, med, scale, dtypes, MIN_POOL)
        if tl is None:
            if pool is None:
                pool = self._pool(store, s, r.peers, q, unit)
            if pool is not None:
                # peers' pooled tail; their log (= relative) scales are the
                # entity's scales rescaled by its own median
                src = SRC_PEER
                m_own, s_own = phase_stats(L, dtypes, MIN_PHASE_N_YOUNG)
                sc_in = np.where(np.isnan(pool["scale"]), s_own, pool["scale"])
                # round 4: a phase the entity has no trusted sample of, but
                # where it was observed (untrusted) on >= RECUR_MIN_DAYS days,
                # takes its own observed level, not the peers'
                obs = self._observed_fill(r.model["state"], q, day0, v, ok, dtypes, unit)
                if obs is not None:
                    m_obs, s_obs = obs
                    use = np.isnan(m_own) & np.isfinite(m_obs)
                    m_own = np.where(use, m_obs, m_own)
                    sc_in = np.where(use, np.fmax(sc_in, s_obs), sc_in)
                med = np.where(np.isnan(m_own), pool["level"], m_own)
                scale = floor_scale(med, sc_in, count_unit=cu)
                tl = np.broadcast_to(pool["tail"], (NH, pool["tail"].shape[-1]))
            elif n_days >= MIN_OWN_DAYS:
                src = SRC_OWN_IMMATURE
                med, scale = phase_stats(L, dtypes, MIN_PHASE_N_YOUNG)
                scale = floor_scale(med, scale, count_unit=cu)
                tl = self._own_tail(L, med, scale, dtypes, MIN_POOL_IMMATURE)
        if tl is None or med is None:
            self._clear_fit(fit, q)
            return
        fit["src"][q], fit["unit"][q] = src, unit
        fit["prof"][q], fit["e7bar"][q] = prof, e7bar
        if "pinf" not in fit:
            fit["pinf"] = np.ones((NQ, 24))
        fit["pinf"][q] = seven_day_pinf(L, dtypes)
        fit["med"][q], fit["scale"][q] = med, scale
        fit["tail"][q], fit["body"][q] = tl[:, :5], tl[:, 5:]
        fit["rp99"][q], fit["pmh"][q], fit["rthr"][q] = np.nan, np.nan, np.nan
        if GUARDED[q] and len(r.peers) >= MIN_PEERS:
            members = sorted(set(r.peers) | {r.e})
            Wp = self._peer_windows(store, s, r.pkey, members, q, day0, now)
            Pmed = self._peer_median(s, r.pkey, q, r.e, r.peers, Wp)
            if Pmed is not None:
                fit["rp99"][q], fit["pmh"][q], fit["rthr"][q] = self._guard_history(
                    W, Pmed, dtypes, unit)

    @staticmethod
    def _observed_fill(st: Mapping[str, np.ndarray], q: int, day0: int, v: np.ndarray,
                       ok: np.ndarray, dtypes: np.ndarray, unit: float
                       ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(median, scale) per (horizon, day type, phase) of the history with
        its untrusted hours filled by what was observed there ('ox' / 'oc'),
        for phases with >= RECUR_MIN_DAYS samples; None when no hour needs it.

        Why (round 4, integration.md §10.7 item 1): a sanctioned nightly
        backup matches a HIGH lib-4 rule every night, so its night rows are
        committed with trust 0; its own ring then holds no night hour (and no
        valid 7-d window at all) and the peer fallback put the PEERS' level
        there: 6.8 GB of 7-d upload against a 20.8 MB 'usual' (pack B,
        10.20.9.5, fit_source peer), a permanent budget alarm. The entity's
        own recurring pattern is better evidence of its usual level than
        other hosts'. Poisoning is bounded: only phases WITHOUT any trusted
        own sample use it, only on the young-entity peer path, and only after
        the same phase recurred on RECUR_MIN_DAYS days; the scale carries the
        median's estimation error (x sqrt(1 + (pi/2) / n))."""
        if "ox" not in st:
            return None
        hours = day0 * 24 + np.arange(NHIST, dtype=np.int64)
        sl = hours % NB
        seen = (st["hour"][sl] == hours) & (st["oc"][sl] >= MIN_COV) & ~ok
        if not seen.any():
            return None
        vf = v.copy()
        vf[seen] = st["ox"][sl[seen], q] / st["oc"][sl[seen]]
        Lf = log_windows(window_sums(vf), unit)
        _prof, corr7, _e = composition_7d(vf, dtypes, unit)
        Lf[H_7D] = Lf[H_7D] - corr7
        med, scale = phase_stats(Lf, dtypes, RECUR_MIN_DAYS)
        n = np.zeros((NH, 2, 24))
        for t in (0, 1):
            sel = dtypes == t
            n_t = np.sum(np.isfinite(Lf[:NH][:, sel, :]), axis=1) if sel.any() else 0
            n[:, t] = np.where(n_t >= RECUR_MIN_DAYS, n_t,
                               np.sum(np.isfinite(Lf[:NH]), axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = scale * np.sqrt(1.0 + 0.5 * math.pi / np.maximum(n, 1.0))
        return med, scale

    @staticmethod
    def _own_tail(L: np.ndarray, med: np.ndarray, scale: np.ndarray, dtypes: np.ndarray,
                  min_pool: int) -> Optional[np.ndarray]:
        """Per horizon: POT fit of the standardised log residuals pooled over
        all phases and days; None if any horizon lacks data (1 h / 8 h /
        day), so a fit is all-or-nothing except the 7-d horizon (young
        history)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            R = (L[:NH] - med[:, dtypes, :]) / scale[:, dtypes, :]
        out = np.full((NH, 5 + PQ.size), np.nan)
        for k in range(NH):
            t = fit_tail(R[k].ravel(), min_pool)
            if t is None:
                if k != H_7D:
                    return None
                continue
            out[k] = _min_zq(t, Z_NORMAL) if k == H_7D else t
        return out

    @staticmethod
    def _clear_fit(fit: Dict[str, Any], q: int) -> None:
        fit["src"][q] = SRC_NONE
        for k in ("med", "scale", "tail", "body", "pmh", "rp99", "rthr", "prof", "e7bar"):
            if k in fit:
                fit[k][q] = np.nan

    @staticmethod
    def _pool(store: Any, s: str, peers: Sequence[str], q: int, unit: float
              ) -> Optional[Dict[str, np.ndarray]]:
        """Pooled GPD of the mature peers, the entity itself excluded (median
        parameters, z_q recomputed), their median log level and log scale per
        phase (fitted with the same unit). Only young entities (or a failed
        own tail) ask, once a day."""
        tails, bodies, levels, scales = [], [], [], []
        for m in peers:
            mod = store.get_model(s, m, MODEL)
            if not (isinstance(mod, dict) and mod.get("fmt") == FMT):
                continue
            f = mod["fit"]
            if int(f["src"][q]) != SRC_OWN or f["unit"][q] != unit:
                continue
            tails.append(f["tail"][q])
            bodies.append(f["body"][q])
            levels.append(f["med"][q])
            scales.append(f["scale"][q])
        if not tails:
            return None
        tail, _ = _nanmed(np.stack(tails), 0)            # (NH, 5)
        for k in range(NH):
            u, xi, sig, rate = tail[k, :4]
            tail[k, 4] = evt.pot_quantile(u, xi, sig, rate, Q_EVAL) if np.isfinite(
                tail[k, :4]).all() else np.nan
        body, _ = _nanmed(np.stack(bodies), 0)
        body = np.maximum.accumulate(np.where(np.isnan(body), -np.inf, body), axis=-1)
        body = np.where(np.isinf(body), np.nan, body)
        level, _ = _nanmed(np.stack(levels), 0)
        scale, _ = _nanmed(np.stack(scales), 0)
        return {"tail": np.concatenate((tail, body), axis=-1), "level": level, "scale": scale}

    @staticmethod
    def _guard_history(W: np.ndarray, Pmed: np.ndarray, dtypes: np.ndarray, unit: float
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """From the entity's committed windows and its peers' median ones:
          rp99  P99 of the entity's own B / peer-median ratio per horizon (spec);
          pmh   the peers' median level per (horizon, day_type, phase), in
                natural units;
          rthr  the level-q threshold of log(B / peer median) per (horizon,
                day_type, phase): same-phase median / robust scale of the log
                ratio and the POT quantile of its pooled standardised
                residuals. Under a class-wide shift the ratio keeps its null
                law, so the guard passes at rate q, not 1 %."""
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = (W + unit) / (Pmed + unit)
            lr = np.log(ratio)
        rp = np.full(NH, np.nan)
        for k in range(NH):
            rk = ratio[k][np.isfinite(ratio[k])]
            if rk.size >= RATIO_MIN_N:
                rp[k] = float(_sorted_quantile(np.sort(rk), RATIO_Q))
        pl, _ = phase_stats(log_windows(Pmed, unit), dtypes, MIN_PHASE_N)
        pmh = np.exp(pl) - unit
        med_l, sc_l = phase_stats(lr, dtypes, MIN_PHASE_N)
        sc_l = floor_scale(med_l, sc_l, LOGRATIO_SCALE_MIN)
        with np.errstate(invalid="ignore", divide="ignore"):
            R = (lr[:NH] - med_l[:, dtypes, :]) / sc_l[:, dtypes, :]
        rthr = np.full((NH, 2, 24), np.nan)
        for k in range(NH):
            t = fit_tail(R[k].ravel(), MIN_POOL_IMMATURE)
            if t is not None:
                if k == H_7D:
                    t = _min_zq(t, Z_NORMAL)
                rthr[k] = med_l[k] + sc_l[k] * t[4]
        return rp, pmh, rthr

    # -------------------------------------------------------------- evaluate
    def _evaluate_system(self, store: Any, s: str, recs: List[_Rec],
                         tick: Dict[str, Any]) -> None:
        byname = {r.e: r for r in recs if r.ok}
        for r in byname.values():
            r.res = self._evaluate(r, r.B, byname, tick)
        link = store.get_model(s, SYSTEM_KEY, LINK_MODEL)
        for chain in _actor_chains(link, s):
            mem = [byname[m] for m in chain if m in byname]
            if len(mem) < 2:
                continue
            cur = max(mem, key=lambda r: (store.last_seen(s, r.e) or -math.inf, r.e))
            Bs = np.stack([m.B for m in mem])
            B_act = np.where(np.all(np.isnan(Bs), 0), np.nan, np.nansum(Bs, 0))
            res = self._evaluate(cur, B_act, byname, tick, exclude=[m.e for m in mem])
            cur.act_res = ([m.e for m in mem], res)

    def _evaluate(self, r: _Rec, B: np.ndarray, byname: Mapping[str, _Rec],
                  tick: Dict[str, Any], exclude: Sequence[str] = ()) -> Dict[str, np.ndarray]:
        fit = r.model["fit"]
        dti, h = tick["dti"], tick["h"]
        floors = tick["floors"]
        Bt = B.T                                                   # (NQ, NH)
        lm = fit["med"][:, :, dti, h].copy()                      # log(W + unit) space
        sc = fit["scale"][:, :, dti, h]
        pinf = fit.get("pinf")
        if pinf is not None:
            # round 4: the 7-d residual is predictive (seven_day_pinf); the
            # threshold and the tail p both use the inflated scale
            sc = sc.copy()
            sc[:, H_7D] = sc[:, H_7D] * pinf[:, h]
        tl = fit["tail"]
        unit = fit["unit"][:, None]
        # the current 7-d window's day-type composition (see composition_7d)
        types = tick.get("types8")
        if types is not None and "prof" in fit:
            u1 = fit["unit"]
            for q in range(NQ):
                eb = fit["e7bar"][q, h]
                if not (eb == eb and u1[q] == u1[q]):
                    continue
                en = expected_7d(fit["prof"][q], types, h)
                with np.errstate(invalid="ignore", divide="ignore"):
                    adj = math.log(en + u1[q]) - math.log(eb + u1[q])
                if math.isfinite(adj):
                    lm[q, H_7D] += adj
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            zthr = np.exp(lm + sc * tl[..., 4]) - unit             # natural units
            m = np.exp(lm) - unit                                  # the usual level
            rz = (np.log(np.maximum(Bt, 0.0) + unit) - lm) / sc
        p = tail_p(rz, tl, fit["body"])
        p[r.qnan] = np.nan
        with np.errstate(invalid="ignore"):
            cand = np.isfinite(p) & np.isfinite(zthr) & (Bt > zthr) & ((Bt - m) > floors[:, None])
        guard = np.ones_like(cand)
        pratio = np.full(cand.shape, np.nan)
        if (cand & GUARDED[:, None]).any():
            skip = set(exclude) | {r.e}
            peers = [byname[x].B for x in r.peers if x in byname and x not in skip]
            if len(peers) >= MIN_PEERS:
                P = np.stack(peers).transpose(0, 2, 1)             # (P, NQ, NH)
                pm_now, cnt = _nanmed(P, 0)
                have = cnt >= MIN_PEERS
                eps = (floors * UNIT_FRAC)[:, None]
                pmh = fit["pmh"][:, :, dti, h]
                rthr = fit["rthr"][:, :, dti, h]
                with np.errstate(invalid="ignore", divide="ignore"):
                    pratio = (Bt + eps) / (pm_now + eps)
                    rp = fit["rp99"]
                    c1 = ~np.isfinite(rp) | (pratio > rp)
                    # level-q test of the ratio; without its fit, the threshold
                    # rescaled by the peers' current over usual level
                    pf = np.where(np.isfinite(pmh), (pm_now + eps) / (pmh + eps), 1.0)
                    c2 = np.where(np.isfinite(rthr), np.log(pratio) > rthr,
                                  Bt > zthr * np.maximum(1.0, pf))
                guard = ~have | (c1 & c2)
                guard[~GUARDED] = True
        alarm = cand & guard
        if tick["training"]:
            alarm[:] = False
        return {"B": Bt, "p": p, "zthr": zthr, "med": m, "alarm": alarm, "ratio": pratio}

    # ------------------------------------------------------------- learning
    def _learn(self, ctx: Context, r: _Rec, tick: Dict[str, Any]) -> None:
        """Gated commit of due rows (+ control, + link seeding); skipped when
        no row is due and there is no model.control. A rollback / release /
        rebase changes the committed history: the fits are then redone."""
        store, model = ctx.store, r.model
        s, e, now, dt = r.s, r.e, tick["now"], tick["dt"]
        rows, gate = model["rows"], model["gate"]
        frontier = G.commit_frontier(now, dt, self._learner.d_min_s)
        st = model["state"]
        before = (gate.version, repr(gate.applied.get("rollback_to")),
                  repr(gate.applied.get("release")), repr(gate.applied.get("rebase_from")))
        lv = gate.link_version
        self._rows_cur = rows
        try:
            # commits are batched (>= COMMIT_BATCH due rows, or the oldest
            # waiting an hour): learning later than D is allowed, and the
            # gate's per-call cost dominates this engine at short cadences
            n_due = rows.count_in(gate.last_ts, frontier)
            if n_due >= COMMIT_BATCH or (n_due and frontier - max(
                    gate.last_ts, rows.ts[rows.h] if len(rows) else frontier) >= 3600.0) \
                    or store.get_model(s, e, G.CONTROL_MODEL) is not None:
                st, gate = self._learner.step(store, s, e, _copy_state(st), gate, now, dt,
                                              training=tick["training"])
            st, gate = self._learner.seed_from_link(store, s, e, st, gate,
                                                    lambda a: self._other_state(store, s, a))
        finally:
            self._rows_cur = None
        after = (gate.version, repr(gate.applied.get("rollback_to")),
                 repr(gate.applied.get("release")), repr(gate.applied.get("rebase_from")))
        if after != before or gate.link_version != lv:
            model["fit"]["ts"][:] = -np.inf                # refit on the corrected history
        if gate.link_version != lv:
            self._inherit_dests(store, s, e, model)
        model["state"], model["gate"], model["version"] = st, gate, int(gate.version)
        rows.prune(now - ROW_KEEP_S)

    def _fetch(self, store: Any, s: str, e: str, ts: float
               ) -> Optional[Tuple[float, int, float, np.ndarray]]:
        """GatedLearner fetch: the tick row at ts (None: pruned, or a quantity
        was unknown that tick — never learn a partial row)."""
        rows = self._rows_cur
        if rows is None:
            return None
        i = rows.find(float(ts))
        if i < 0:
            return None
        x = rows.x[i].astype(np.float64)
        if np.isnan(x).any():
            return None
        return float(rows.ts[i]), int(rows.a[i]), float(rows.frac[i]), x

    @staticmethod
    def _other_state(store: Any, s: str, entity: str) -> Optional[Dict[str, np.ndarray]]:
        m = store.get_model(s, entity, MODEL)
        if isinstance(m, dict) and m.get("fmt") == FMT:
            return _copy_state(m["state"])
        return None

    @staticmethod
    def _inherit_dests(store: Any, s: str, e: str, model: Dict[str, Any]) -> None:
        """Destinations a linked predecessor knew are not new to e."""
        first = model["live"]["dest_first"]
        for src in _link_sources(store.get_model(s, SYSTEM_KEY, LINK_MODEL), e):
            m = store.get_model(s, src, MODEL)
            if not (isinstance(m, dict) and m.get("fmt") == FMT):
                continue
            for did, (f0, l0) in m["live"]["dest_first"].items():
                cur = first.get(did)
                first[did] = [f0, l0] if cur is None else [min(cur[0], f0), max(cur[1], l0)]

    # ---------------------------------------------------------------- write
    def _write(self, ctx: Context, r: _Rec, tick: Dict[str, Any]) -> int:
        store, s, e, now = ctx.store, r.s, r.e, tick["now"]
        win = int(round(tick["dt"]))
        model = r.model
        if not r.ok:
            emit.write_scores(store, s, e, now, {d: _NAN for d in r.degraded},
                              degraded=r.degraded, window_s=win)
            if r.persist:
                store.put_model(s, e, MODEL, model, version=model["version"], ts=now)
            return 1
        store.put_model(s, e, MODEL, model, version=model["version"], ts=now)
        res = r.res
        assert res is not None
        p = res["p"].copy()
        alarm = res["alarm"].copy()
        if r.act_res is not None:
            ares = r.act_res[1]
            p = np.fmin(p, ares["p"])
            alarm |= ares["alarm"]
        scores, pms, axes, accs = {}, {}, {}, {}
        weak: Dict[str, str] = {}
        axis_p: Dict[str, float] = {}
        for ax in AXES:
            d = AXIS_DETECTOR[ax]
            if d in r.degraded:
                scores[d] = _NAN
                continue
            pa = p[AXIS_Q[ax]]
            fin = pa[np.isfinite(pa)]
            if not fin.size:
                scores[d] = _NAN                     # no fit yet (maturity): unscored
                continue
            pax = max(P_FLOOR, min(1.0, fin.size * float(fin.min())))
            axis_p[ax] = pax
            # contract M: an axis scored against the peers' tail / levels,
            # or an immature own history, is a weaker reference than B13's
            # model assumes (its p stays valid; B24 calibrates it)
            srcs = set(int(x) for x in model["fit"]["src"][AXIS_Q[ax]][
                np.isfinite(pa.reshape(-1, NH)).any(axis=1)])
            if SRC_PEER in srcs:
                weak[d] = emit.cause(emit.FALLBACK, "peer")
            elif SRC_OWN_IMMATURE in srcs:
                weak[d] = emit.cause(emit.INSUFFICIENT_SUPPORT, "budget_history")
            scores[d] = -math.log10(pax)
            pms[d] = max(pax, PM_STORE_FLOOR)
            al = int(bool(alarm[AXIS_Q[ax]].any()))
            accs[d] = al
            if al or pax <= AXES_PM_MAX:
                axes[d] = [ax]
        emit.write_scores(store, s, e, now, scores, pm=pms or None, axes=axes or None,
                          acc_alarm=accs or None, degraded={**weak, **r.degraded} or None,
                          window_s=win)
        store.add_derived(DerivedMetric(name=SERIES, value=_series_value(res), ts=now, system=s,
                                        entity=e, window_s=win, kind=MetricKind.CATEGORICAL,
                                        inputs=[NAT, "act.objs", "act.stream"]))
        fired = self._events(store, r, res, tick) if not tick["training"] else False
        model["live"]["last"] = {
            "ts": now, "axis_p": axis_p,
            "alarm": {ax: accs.get(AXIS_DETECTOR[ax], 0) for ax in AXES},
            "alarms": [f"{Q_NAMES[qi]}.{H_NAMES[k]}" for qi, k in zip(*np.nonzero(alarm))]}
        if self.entity_due((s, e, "profile"), now, PROFILE_EVERY_S) or fired:
            self._write_profile(store, r, res, tick, axis_p, accs)
        return 1

    def _events(self, store: Any, r: _Rec, res: Dict[str, np.ndarray],
                tick: Dict[str, Any]) -> bool:
        """budget_exceeded once per (quantity, scope) and local day while the
        within-day horizons (1 h, 8 h, day) alarm; an alarm standing on the
        7-d horizon alone (yesterday's excess still inside the week) is
        reported once per 7 days, not every day it stays in the window."""
        fired = False
        rep = r.model["live"]["reported"]
        today = tick["today"]
        for key in [k for k, d in rep.items() if today - d >= 7]:
            del rep[key]
        todo = [("entity", res, None)]
        if r.act_res is not None:
            todo.append(("actor", r.act_res[1], r.act_res[0]))
        for scope, rs, actor in todo:
            for qi in np.nonzero(rs["alarm"].any(axis=1))[0]:
                key = f"{Q_NAMES[qi]}|{scope}"
                fresh = bool(rs["alarm"][qi, :H_7D].any())
                last = rep.get(key)
                if last == today or (not fresh and last is not None):
                    continue
                other = rep.get(f"{Q_NAMES[qi]}|entity")
                if scope == "actor" and (other == today or (not fresh and other is not None)):
                    continue
                rep[key] = today
                self._event(store, r, int(qi), rs, scope, actor, tick)
                fired = True
        return fired

    def _event(self, store: Any, r: _Rec, qi: int, rs: Dict[str, np.ndarray], scope: str,
               actor: Optional[List[str]], tick: Dict[str, Any]) -> None:
        s, e, now = r.s, r.e, tick["now"]
        q, ax, unit = Q_NAMES[qi], Q_AXIS[qi], Q_UNIT[qi]
        hs = np.nonzero(rs["alarm"][qi])[0]
        k = int(hs[np.argmin(rs["p"][qi, hs])])
        hn = H_NAMES[k]
        B, m, z, tp = (float(rs["B"][qi, k]), float(rs["med"][qi, k]),
                       float(rs["zthr"][qi, k]), float(rs["p"][qi, k]))
        floor = float(tick["floors"][qi])
        p = max(P_FLOOR, min(1.0, NQ * NH * tp))       # the whole family, per hourly evaluation
        ed = combine.e_day(p, 3600.0)
        sev = combine.e_day_severity(ed) or "low"
        d = AXIS_DETECTOR[ax]
        who = f" by actor {'+'.join(actor)}" if actor else ""
        desc = (f"{q} {H_LABEL[hn]}{who}: {_fmt(B, unit)} against a usual {_fmt(m, unit)} "
                f"at this hour (threshold {_fmt(z, unit)}, absolute floor {_fmt(floor, unit)}), "
                f"tail p = {tp:.2g}")
        t0 = {"1h": now - 3600.0, "8h": now - 8 * 3600.0, "7d": now - 168 * 3600.0,
              "day": tick["today"] * 86400.0 - tick["off"]}[hn]
        fit = r.model["fit"]
        ratio = rs["ratio"][qi, k]
        store.add_event(BehaviorEvent(
            system=s, entity=e, ts=now, kind=EVENT_KIND,
            score=float(min(1.0, -math.log10(p) / 10.0)), severity=Severity(sev),
            description=desc,
            extra={"axis": ax, "detector": d, "quantity": q, "horizon": hn,
                   "horizons": [H_NAMES[i] for i in hs], "unit": unit, "value": _j(B, 3),
                   "usual": _j(m, 3), "z_q": _j(z, 3), "abs_floor": floor,
                   "excess": _j(B - m, 3), "tail_p": _jp(tp), "peer_ratio": _j(ratio),
                   "scope": scope, "actor": list(actor) if actor else None,
                   "fit_source": SRC_NAME[int(fit["src"][qi])],
                   "history_days": int(fit["days"][qi])},
            p_value=p, e_day=float(ed), axes=[ax], p_by_detector={d: p},
            dedupe_key=f"{EVENT_KIND}|{s}|{e}|{q}|{scope}|{tick['today']}",
            model_version=int(r.model["version"]), window=(float(t0), now)))

    def _write_profile(self, store: Any, r: _Rec, res: Dict[str, np.ndarray],
                       tick: Dict[str, Any], axis_p: Mapping[str, float],
                       accs: Mapping[str, int]) -> None:
        """profile.extra.budget (JSON-friendly): every quantity's horizons in
        natural units (value, usual at this hour, z_q, tail p), maturity, and
        today's exfil totals below any floor for B18's class aggregation."""
        s, e, now = r.s, r.e, tick["now"]
        fit = r.model["fit"]
        qs: Dict[str, Any] = {}
        for qi, q in enumerate(Q_NAMES):
            qs[q] = {"axis": Q_AXIS[qi], "unit": Q_UNIT[qi],
                     "abs_floor": float(tick["floors"][qi]),
                     "source": SRC_NAME[int(fit["src"][qi])],
                     "h": {hn: {"value": _j(res["B"][qi, k], 3), "usual": _j(res["med"][qi, k], 3),
                                "z_q": _j(res["zthr"][qi, k], 3), "p": _jp(res["p"][qi, k]),
                                "alarm": int(bool(res["alarm"][qi, k]))}
                           for k, hn in enumerate(H_NAMES)}}
        day = _EPOCH + _dt.timedelta(days=int(tick["today"]))
        out = {
            "updated": now, "version": int(r.model["version"]), "day": day.isoformat(),
            "history_days": int(fit["days"].max()) if fit["days"].size else 0,
            "axes": {ax: {"detector": AXIS_DETECTOR[ax], "p": _jp(axis_p.get(ax)),
                          "alarm": int(accs.get(AXIS_DETECTOR[ax], 0))} for ax in AXES},
            "quantities": qs,
            "exfil_today": {"up_novel_bytes": _j(res["B"][QI["up_novel"], 2], 1),
                            "dns_label_bytes": _j(res["B"][QI["dns_label"], 2], 1)},
            "reported_today": sorted(k for k, d in r.model["live"]["reported"].items()
                                     if d == tick["today"]),
        }
        if r.act_res is not None:
            out["actor"] = list(r.act_res[0])
        p = store.profile(s, e) or EntityProfile(system=s, entity=e)
        p.extra["budget"] = out
        store.put_profile(p)
