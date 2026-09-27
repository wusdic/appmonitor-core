"""RhythmEngine (B07) -- when each IP and each class is active.

Why: a compromised workstation used at 02:00 with its owner's normal paths
and volume looks normal to every intensity, shape and vocabulary detector;
a backup server that silently stops running looks like "nothing happened".
Both are only visible against the entity's own daily / weekly rhythm. v1
thresholded per-tick counts with fixed tick-count CUSUMs, which changed
meaning 15-60x between 60-s, 900-s and 3600-s cadences. This engine models
presence on a fixed 15-minute LOCAL slot clock instead, so every cadence
describes (and alarms on) the same wall-clock slots.

  1. Slot clock. A tick covers [now - dt, now) (ctx.window_s is the real dt).
     A slot is active if any act.stream timestamp falls in it; a 3600-s tick
     is resolved into its 4 slots from the timestamps. An active tick without
     stream rows marks every slot it covers (nothing finer is known); in a
     sampled tick (act.stream_frac < 1) a slot without rows is unknown (NaN),
     not inactive. feature.active = 0 is data: the covered slots are
     inactive (absence is data). A slot is finalised by the first tick that
     reaches its end (at 60-s ticks ~15 ticks contribute); a finalised slot
     covered < 450 s and never active is unobserved. At 60-s ticks the
     current slot is scored provisionally as soon as activity appears
     (activity within a slot is monotone), so the off-hours alarm fires in
     the same wall-clock slot, with the same W, at 60 s and at 900 s.
  2. Model (lib/m_rhythm): P(active | cell) ~ Beta per 15-min quarter of
     bin48 (bin168 once >= 4 weeks), adaptive von Mises smoothing across
     hours (kappa 4), class prior Beta(6 pi_class, 6 (1 - pi_class)) (system
     tier Beta(2 pi, 2 (1 - pi)) when the class has < 3 members, contract L),
     half-life 28 d. Updated at slot completion through lib/gating with delay
     D and weight = the slot's trust (minimum over its ticks); trust-gated,
     held while quarantined, checkpointed, reversible and model.control aware
     (rollback_to / release / rebase_from / frozen), link seeded.
  3. Off-hours: Bernoulli CUSUM in bits, once per slot, only over bins with
     p_hat <= 0.3 (p1 = min(0.95, max(0.5, 5 p_hat))); h = log2(ARL) + 0.5
     with ARL = 100 days of slots (13.7 bits). p_eq = 2^-W.
  4. Silence: only for machine-like rhythms (normalised 168-bin entropy <=
     0.8) in bins with p_hat >= 0.95; s += -ln(1 - p_hat) per silent slot, an
     active eligible slot resets it, p_eq = e^-s, alarm at s >= ln(ARL).
     Other entities are unscored (NaN).
  5. Schedule shift: a machine-like entity whose whole usual window
     (contiguous slots with p_hat >= 0.5) stays silent, followed within 24 h
     by a run of unusual activity of the same duration (+-25 %, at least +-1
     slot) and volume (x0.5..x2) whose centre moved by > 1 h, gets a
     schedule_shift event (LOW) when that run ends. While a run may still be
     the moved window (<= its duration + tolerance) its off-hours evidence is
     held (the pre-run W is reported, the 60-s provisional increment waits),
     so an explained move never alarms; a run that outgrows the tolerance or
     ends without matching reports its full W at once. On a shift W and s
     are reset (the evidence is explained) and for 24 h
     behavior.rhythm.shift_explained = 1; B25 caps temporal-only evidence at
     LOW when a schedule_shift event exists. A change of destination or
     content is scored by other detectors and is never capped.
  6. Calendar self-healing: when the fraction of a system's entities active
     in a slot of a nonworkday exceeds 3x the system rhythm's usual level
     (>= 3 entities, >= 25 %), that local day is treated as a workday from
     then on (model.rhythm@(s, __system__).healed): an unconfigured 调休 day
     is not scored as a weekend.
  7. Descriptors (hourly): mu_h, R, the 80 % window, the active window, the
     workday / nonworkday ratio, entropy168 -> profile.extra.rhythm. Class
     and system rhythms (pooled member counts, the expected active fraction
     per slot that B18 consumes through m_rhythm.class_fraction) are rebuilt
     hourly into model.rhythm@(s, class:<rid> | __system__).

Writes: model.rhythm@(s, e | class:<rid> | __system__) (layout and pure
accessors in lib/m_rhythm), behavior.score / pm [offhours, silence],
behavior.acc_alarm, behavior.axes (lib/emit), behavior.rhythm (dict series
{p_expected, W_off, s_sil, ...}), profile.extra.rhythm; event
schedule_shift. ctx.training learns and scores but raises no alarm and emits
no event. A missing feature.active row (B01 stale or failed) gives NaN plus
behavior.degraded.
"""
from __future__ import annotations

import math
import pickle
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import BehaviorEvent, DerivedMetric, EntityProfile, MetricKind, Severity
from .lib import emit
from .lib import gating as G
from .lib import m_class
from .lib import m_rhythm as R
from .lib import m_template as MT
from .lib import timebins as TB
from .lib.classkeys import SYSTEM_KEY

MODEL = R.MODEL
LEARNER = "rhythm"
SERIES = "behavior.rhythm"
ACTIVE = "feature.active"
B01_ENGINE = "behavior.feature_vector"
DETS = ("offhours", "silence")
AXES = ["temporal"]

SLOT_S = float(R.SLOT_S)
MIN_COV_S = 450.0                 # a never-active slot needs this coverage to count as 0
LEDGER_KEEP_S = 9 * 86400.0       # > journal / held horizon (8 d): fetch must find rows
ACC_KEEP_S = 2 * 86400.0          # open slot accumulators older than this are dropped
SHIFT_WINDOW_S = 86400.0          # silence -> new window within 24 h
SHIFT_MIN_H = 1.0                 # circular shift of the window centre
SHIFT_EXPLAIN_S = 86400.0         # shift_explained flag lifetime
SHIFT_DUR_TOL = 0.25
SHIFT_VOL_RANGE = (0.5, 2.0)
SHIFTS_KEEP = 16
TIER_REFIT_S = 3600.0
ENTROPY_REFIT_S = 3600.0          # entropy168 / machine_like (~40 us)
DESC_REFIT_S = 6 * 3600.0         # full descriptors (~0.3 ms): the rhythm moves slowly
PRIOR_REFIT_S = 3600.0
CLASS_MIN_MEMBERS = 3             # contract L: a smaller class backs off to the system tier
SYSTEM_MIN_MEMBERS = 2            # the entity alone is no prior for itself
HEAL_FACTOR = 3.0
HEAL_MIN_ACTIVE = 3
HEAL_MIN_FRAC = 0.25
HEAL_MIN_N = 2.0                  # pooled decayed observations of the system cell
HEAL_KEEP_S = 30 * 86400.0
AXES_PM_MAX = 0.05                # axes are written when the detector is informative
_EPS = 1e-6


class _Slot(NamedTuple):
    """One finalised slot (ledger entry; also the learner's unit)."""
    tc: float        # ts of the tick that finalised it (gating row)
    a: float         # 1 active, 0 inactive, NaN unobserved
    vol: float       # events in the slot (act.stream rows / stream_frac, else act.events share)
    c48: int
    c168: int
    t_end: float     # UTC end of the slot


class _Row(NamedTuple):
    """Learner row of tick `ts`: the slots it finalised plus trust minima."""
    ts: float
    slots: Tuple[_Slot, ...]
    w_eff_min: float     # min behavior.trust over the ticks inside those slots (NaN: none)
    w_prov_min: float
    w_tick_eff: float    # trust / trust_prov at ts (to tell a release from a commit)
    w_tick_prov: float


# ============================================================ learner state
def _init_state() -> Dict[str, Any]:
    return R.new_state()


def _decay_to(st: Dict[str, Any], t: float) -> float:
    """Advance the state clock to t (decaying the counts) and return the
    multiplier of a count observed at t (< 1 for a row older than the clock,
    so released / replayed rows never decay the state backwards)."""
    t_ref = st["t_ref"]
    if not math.isfinite(t_ref):
        st["t_ref"] = t
        return 1.0
    if t > t_ref:
        g = R.decay_factor(t_ref, t)
        for k in ("A48", "N48", "V48", "A168", "N168"):
            st[k] *= g
        st["t_ref"] = t
        return 1.0
    return R.decay_factor(t, t_ref)


def _row_weight(row: _Row, w: float) -> float:
    """Slot weight = min trust over its ticks. The gate passes w_eff on a
    normal commit or journal replay and w_prov on a release / rebase; the
    matching minimum is taken over the slot's other ticks."""
    prov = (row.w_tick_prov == row.w_tick_prov and abs(w - row.w_tick_prov) < 1e-9
            and not abs(w - row.w_tick_eff) < 1e-9)
    m = row.w_prov_min if prov else row.w_eff_min
    return min(w, m) if m == m else w


def _update(st: Dict[str, Any], row: _Row, w: float) -> Dict[str, Any]:
    """Fold the finalised slots of one tick (in place; deterministic in
    (state, row, w), so checkpoint + replay is exact)."""
    ww = _row_weight(row, float(w))
    if not (ww > 0.0 and math.isfinite(ww)):
        return st
    for sl in row.slots:
        a = sl.a
        if not a == a:
            continue                                  # unobserved: no evidence
        x = ww * _decay_to(st, sl.t_end)
        st["A48"][sl.c48] += x * a
        st["N48"][sl.c48] += x
        if a > 0.0 and sl.vol == sl.vol:
            st["V48"][sl.c48] += x * a * sl.vol
        if sl.c168 >= 0:
            st["A168"][sl.c168] += x * a
            st["N168"][sl.c168] += x
        t0 = sl.t_end - SLOT_S
        st["t_first"] = t0 if not st["t_first"] <= t0 else st["t_first"]
        st["t_last"] = sl.t_end if not st["t_last"] >= sl.t_end else st["t_last"]
        st["n_slots"] += 1
    st["n_commit"] += 1
    return st


def _merge(own: Dict[str, Any], other: Dict[str, Any], w: float) -> Dict[str, Any]:
    """own += w * other in sufficient-statistic space (link seeding)."""
    t_o = other.get("t_ref", math.nan)
    if not math.isfinite(t_o):
        return own
    f = w * _decay_to(own, t_o)
    for k in ("A48", "N48", "V48", "A168", "N168"):
        own[k] = own[k] + f * np.asarray(other[k], dtype=np.float64)
    for k, better in (("t_first", min), ("t_last", max)):
        a, b = own.get(k, math.nan), other.get(k, math.nan)
        own[k] = b if not math.isfinite(a) else (a if not math.isfinite(b) else better(a, b))
    return own


def _dump(st: Dict[str, Any]) -> bytes:
    return pickle.dumps(st, protocol=pickle.HIGHEST_PROTOCOL)


def _load(blob: Any) -> Dict[str, Any]:
    return pickle.loads(blob) if isinstance(blob, (bytes, bytearray)) else blob


def _new_det() -> Dict[str, Any]:
    """Per-entity detector state (live, not learned, never rolled back)."""
    return {"W": 0.0, "s": 0.0, "acc": {}, "last_slot": None, "prov": None,
            "win": None, "miss": None, "new": None, "explained_until": -math.inf,
            "alarm": {d: 0 for d in DETS}, "w_slot": None}


# ================================================================= clock
class _Clock:
    """The slot geometry of one tick (shared by every entity of a system)."""

    def __init__(self, now: float, dt: float, tz: str, calendar: Any,
                 healed: Set[int]) -> None:
        self.now, self.dt, self.tz, self.cal, self.healed = now, dt, tz, calendar, healed
        t_start = now - dt
        off1 = _offset(now, tz)
        off0 = _offset(t_start, tz) if dt > 0 else off1
        self.off_now = off1
        self.loc0 = t_start + off0                     # local seconds of the tick start
        self.loc1 = now + off1                         # ... and of its end (exclusive)
        self.cur = int(self.loc1 // SLOT_S)            # slot of `now` (not complete)
        self.lo = int(self.loc0 // SLOT_S)
        self.hi = max(self.lo, int(math.ceil(self.loc1 / SLOT_S)) - 1)
        self._days: Dict[int, Tuple[bool, bool]] = {}
        self._cells: Dict[int, Tuple[int, int]] = {}

    def slots(self) -> List[Tuple[int, float]]:
        """[(slot, covered seconds)] of the tick interval."""
        out = []
        for j in range(self.lo, self.hi + 1):
            cov = min((j + 1) * SLOT_S, self.loc1) - max(j * SLOT_S, self.loc0)
            if cov > 0.0:
                out.append((j, cov))
        return out

    def slot_of_rows(self, ts: np.ndarray) -> np.ndarray:
        """Local slot of each stream timestamp, clamped into the tick's slots
        (a row stamped exactly at `now` belongs to this tick)."""
        j = np.floor((np.asarray(ts, dtype=np.float64) + self.off_now) / SLOT_S)
        return np.clip(j, self.lo, self.hi).astype(np.int64)

    def cells(self, j: int) -> Tuple[int, int]:
        c = self._cells.get(j)
        if c is None:
            d = j // R.SLOTS_PER_DAY
            info = self._days.get(d)
            if info is None:
                info = self._days[d] = R.day_info(d, self.cal, self.healed)
            c = self._cells[j] = R.slot_cells(j, *info)
        return c

    def nonwork_by_calendar(self, j: int) -> bool:
        return R.day_info(j // R.SLOTS_PER_DAY, self.cal, None)[0]

    def t_end(self, j: int) -> float:
        """UTC end of slot j (offset of `now`; exact away from DST changes)."""
        return (j + 1) * SLOT_S - self.off_now

    def heal(self, day: int) -> None:
        self.healed.add(int(day))
        self._days.clear()
        self._cells.clear()


def _offset(ts: float, tz: str) -> float:
    off = TB.local_datetime(ts, tz).utcoffset()
    return off.total_seconds() if off is not None else 0.0


# ================================================================ engine
class RhythmEngine(Engine):
    name = "behavior.rhythm"
    layer = "behavior"
    consumes = ["feature.active", "feature.tctx", "act.stream", "act.stream_frac", "act.events",
                "behavior.trust", "behavior.trust_prov", "behavior.quarantine", "model.class",
                "model.control", "model.link"]
    produces = ["model.rhythm", "behavior.score", "behavior.pm", "behavior.acc_alarm",
                "behavior.axes", "behavior.degraded", SERIES, "profile.extra.rhythm",
                "event.schedule_shift"]
    description = ("Circadian / weekly presence per IP and class on a 15-minute slot clock: "
                   "off-hours Bernoulli CUSUM, silence of scheduled machines, schedule "
                   "shifts, calendar self-healing and rhythm descriptors.")
    interval = 1

    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        self.arl_days = float(params.get("arl_days", R.ARL_DAYS))
        slots = self.arl_days * R.SLOTS_PER_DAY
        self.h_off = math.log2(max(slots, 1.0)) + 0.5
        self.h_sil = math.log(max(slots, 1.0))
        self.tier_refit_s = float(params.get("tier_refit_s", TIER_REFIT_S))
        self._learners: Dict[float, G.GatedLearner] = {}
        self._ledger: Dict[str, Any] = {}

    # ------------------------------------------------------------------ run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        now, dt = float(ctx.now), float(ctx.window_s)
        cfg = ctx.config or {}
        tz = cfg.get("tz") or TB.DEFAULT_TZ
        cal = TB.parse_calendar(cfg.get("calendar"))
        d_min = cfg.get("D_min_s", G.D_MIN_S)
        lrn = self._learner(float(d_min) if d_min else G.D_MIN_S)
        b01_failed = store.engine_failed(B01_ENGINE, now)
        n = 0
        for s in store.systems():
            sysm = self._system_model(store, s)
            clock = _Clock(now, dt, tz, cal, R.healed_days(sysm))
            presence: Dict[int, List[float]] = {}
            used: Set[str] = set()
            for e in store.entities(s):
                n += self._entity(ctx, lrn, s, e, clock, presence, used, b01_failed)
            if presence:
                self._heal(store, s, sysm, clock, presence, now)
            self._refit_tiers(store, s, now, used)
        return n

    def _learner(self, d_min_s: float) -> G.GatedLearner:
        lrn = self._learners.get(d_min_s)
        if lrn is None:
            lrn = self._learners[d_min_s] = G.GatedLearner(
                name=LEARNER, init=_init_state, update=_update, fetch=self._fetch,
                dump=_dump, load=_load, merge=_merge, d_min_s=d_min_s,
                ckpt_every_s=G.CKPT_EVERY_S, clock=ACTIVE)
        return lrn

    def _fetch(self, store, s: str, e: str, ts: float) -> Optional[_Row]:
        """Slots finalised at tick ts, from the ledger of the entity being
        stepped (act.stream is kept 1 h, commits lag up to 4 h and releases
        reach back 8 d), with the trust minima over the slots' ticks."""
        led = self._ledger
        js = led.get("ticks", {}).get(ts)
        if not js:
            return None
        slots = tuple(led["slots"][j] for j in js if j in led["slots"])
        if not slots:
            return None
        t0 = min(sl.t_end for sl in slots) - SLOT_S + _EPS
        return _Row(ts, slots, _ring_min(store, s, e, G.TRUST, t0, ts),
                    _ring_min(store, s, e, G.TRUST_PROV, t0, ts),
                    _ring_at(store, s, e, G.TRUST, ts), _ring_at(store, s, e, G.TRUST_PROV, ts))

    # --------------------------------------------------------------- entity
    def _entity(self, ctx: Context, lrn: G.GatedLearner, s: str, e: str, clock: _Clock,
                presence: Dict[int, List[float]], used: Set[str], b01_failed: bool) -> int:
        store, now, dt = ctx.store, clock.now, clock.dt
        model = store.get_model(s, e, MODEL)
        if not (isinstance(model, dict) and model.get("kind") == "entity" and "gate" in model):
            model = None
        row = store.vec_at(s, e, ACTIVE, now)
        if row is None or b01_failed:
            if model is None and store.vec_latest(s, e, ACTIVE) is None and not b01_failed:
                return 0                                  # never seen by B01: nothing to do
            cause = "producer_error:" + B01_ENGINE if b01_failed else "stale:" + ACTIVE
            emit.write_scores(store, s, e, now, {d: None for d in DETS},
                              degraded={d: cause for d in DETS}, window_s=int(dt))
            if model is not None:                        # keep committing older rows
                self._learn(ctx, lrn, s, e, model)
                store.put_model(s, e, MODEL, model, version=int(model["gate"].version))
            return 0
        if model is None:
            model = R.new_model("entity")
            model.update(gate=G.GateState(), ledger={"slots": {}, "ticks": {}}, det=_new_det())
        det = model["det"]
        self._refresh_prior(store, s, e, model, now)
        ck = m_class.class_key(store, s, e)
        if ck:
            used.add(ck)
        if self.entity_due(("rhythm-entropy", s, e), now, ENTROPY_REFIT_S) or \
                not math.isfinite(model.get("entropy168", math.nan)):
            h = R.entropy168(model)
            model["entropy168"] = h
            model["machine_like"] = bool(math.isfinite(h) and h <= R.ENTROPY_MAX)
            if math.isfinite(h) and (not model.get("desc") or self.entity_due(
                    ("rhythm-desc", s, e), now, DESC_REFIT_S)):
                model["desc"] = R.descriptors(model)

        # 1) this tick's slot activity, merged into the open slot accumulators
        act = float(row[0]) if len(row) else math.nan
        self._observe(store, s, e, clock, det, act)
        # 2) finalise completed slots: detectors, ledger, system presence
        finals = self._finalise(ctx, s, e, clock, model, det, presence)
        # 3) provisional score of the current slot (60-s ticks)
        W_rep, p_cur, a_cur, w_slot = self._provisional(model, det, clock)
        held = _held_W(det)
        if held is not None:
            W_rep = held                    # a possible schedule move: evidence held
        # 4) learn (rows <= now - D; the scores above used the pre-commit model)
        self._learn(ctx, lrn, s, e, model)
        _prune(model, now)
        model["updated"] = now
        model["version"] = int(model["gate"].version)
        model["rev"] = int(model.get("rev", 0)) + 1
        store.put_model(s, e, MODEL, model, version=int(model["gate"].version))
        self._write(ctx, s, e, model, det, W_rep, p_cur, a_cur, w_slot, finals)
        return 1

    def _observe(self, store, s: str, e: str, clock: _Clock, det: Dict[str, Any],
                 act: float) -> None:
        """Merge the tick's per-slot activity [a, volume, covered s] into det['acc']."""
        spans = clock.slots()
        if not spans:
            return
        rows = MT.stream_rows(store, s, e, clock.now)
        frac = MT.stream_frac(store, s, e, clock.now)
        sampled = frac == frac and 0.0 < frac < 1.0
        per_row = 1.0 / frac if sampled else 1.0
        hit: Dict[int, float] = {}
        if len(rows):
            ts = rows["ts"]
            ts = ts[np.isfinite(ts)]
            if ts.size:
                js, cnt = np.unique(clock.slot_of_rows(ts), return_counts=True)
                hit = {int(j): float(c) * per_row for j, c in zip(js, cnt)}
        ev = _fresh_float(store, s, e, "act.events", clock.now)
        acc = det["acc"]
        last = det["last_slot"]
        n_span = len(spans)
        for j, cov in spans:
            if last is not None and j <= last:
                continue                                   # already finalised
            if j in hit:
                a, vol = 1.0, hit[j]
            elif hit:
                a, vol = (math.nan if sampled else 0.0), 0.0
            elif act > 0.5:                                 # active, nothing finer known
                a = 1.0
                vol = ev / n_span if ev == ev and ev > 0.0 else math.nan
            elif act == act:
                a, vol = 0.0, 0.0                           # absence is data
            else:
                a, vol = math.nan, 0.0                      # B01 wrote NaN: unknown
            cur = acc.get(j)
            if cur is None:
                acc[j] = [a, vol, cov if a == a else 0.0]
                continue
            if a == 1.0 or cur[0] == 1.0:
                cur[0] = 1.0
            elif not a == a or not cur[0] == cur[0]:
                cur[0] = math.nan
            if vol == vol:
                cur[1] = vol + cur[1] if cur[1] == cur[1] else vol
            if a == a:
                cur[2] += cov

    def _finalise(self, ctx: Context, s: str, e: str, clock: _Clock, model: Dict[str, Any],
                  det: Dict[str, Any], presence: Dict[int, List[float]]) -> List[Tuple[int, float]]:
        """Close every accumulated slot that ended before now (in slot order)."""
        acc = det["acc"]
        done = sorted(j for j in acc if j < clock.cur)
        if not done:
            return []
        led = model["ledger"]
        machine = bool(model.get("machine_like", False))
        tick_js: List[int] = []
        out: List[Tuple[int, float]] = []
        for j in done:
            a, vol, cov = acc.pop(j)
            if a == 0.0 and cov < MIN_COV_S:
                a = math.nan                                  # barely observed
            c48, c168 = clock.cells(j)
            t_end = clock.t_end(j)
            p = R.p_cell(model, c48, c168)
            w_prev = det["W"]
            det["W"] = R.offhours_step(w_prev, a, p)
            det["s"] = R.silence_step(det["s"], a, p, machine)
            det["w_slot"] = j
            self._track_shift(ctx, s, e, model, det, j, a, vol, p, c48, machine, t_end, w_prev)
            led["slots"][j] = _Slot(clock.now, a, vol if vol == vol else math.nan, c48, c168,
                                    t_end)
            tick_js.append(j)
            det["last_slot"] = j if det["last_slot"] is None else max(det["last_slot"], j)
            if a == a:
                pr = presence.setdefault(j, [0.0, 0.0])
                pr[0] += a
                pr[1] += 1.0
            out.append((j, a))
        led["ticks"][clock.now] = tuple(tick_js)
        return out

    def _provisional(self, model: Dict[str, Any], det: Dict[str, Any], clock: _Clock
                     ) -> Tuple[float, float, float, Optional[int]]:
        """(W to report, p_hat and activity of the current slot, slot W refers
        to). An already active open slot is scored now (activity within a slot
        is monotone, so its final increment is known); an open slot without
        activity yet is left to its completion (it may still become active).
        While a missed usual window is pending (a schedule-shift candidate)
        the unusual run is scored at slot completion only, where the shift is
        checked first, so an explained move never alarms provisionally."""
        j = clock.cur if clock.cur in det["acc"] else clock.hi
        c48, c168 = clock.cells(j)
        p = R.p_cell(model, c48, c168)
        cur = det["acc"].get(j)
        a = cur[0] if cur is not None else math.nan
        if cur is not None and a == 1.0 and not _shift_pending(det, p):
            return R.offhours_step(det["W"], 1.0, p), p, a, j
        return det["W"], p, a, det.get("w_slot")

    def _learn(self, ctx: Context, lrn: G.GatedLearner, s: str, e: str,
               model: Dict[str, Any]) -> None:
        store, now, dt = ctx.store, float(ctx.now), float(ctx.window_s)
        self._ledger = model["ledger"]
        try:
            state, gate = lrn.step(store, s, e, model["state"], model["gate"], now, dt,
                                   training=ctx.training)
            state, gate = lrn.seed_from_link(store, s, e, state, gate,
                                             lambda src: _other_state(store, s, src))
        finally:
            self._ledger = {}
        model["state"], model["gate"] = state, gate

    # --------------------------------------------------------- schedule shift
    def _track_shift(self, ctx: Context, s: str, e: str, model: Dict[str, Any],
                     det: Dict[str, Any], j: int, a: float, vol: float, p: float, c48: int,
                     machine: bool, t_end: float, w_prev: float) -> None:
        """Usual-window misses and unusual-activity runs, per finalised slot."""
        miss = det["miss"]
        if miss is not None and t_end - miss["t_end"] > SHIFT_WINDOW_S + SLOT_S:
            det["miss"] = miss = None
        if not (a == a and p == p):
            det["win"] = None                              # unobserved: runs are unknown
            det["new"] = None
            return
        usual = p >= R.P_USUAL
        win = det["win"]
        if usual:
            if win is None:
                win = det["win"] = {"j0": j, "n": 0, "act": 0.0, "vol_exp": 0.0}
            win["n"] += 1
            win["act"] += a
            v = R.expected_volume(model, c48)
            win["vol_exp"] += v if v == v else math.nan
        elif win is not None:
            if win["act"] == 0.0 and machine:
                det["miss"] = {"j0": win["j0"], "n": win["n"], "vol_exp": win["vol_exp"],
                               "t_end": t_end - SLOT_S}
            det["win"] = None
        new = det["new"]
        if a > 0.5 and not usual:
            if new is None:
                new = det["new"] = {"j0": j, "n": 0, "vol": 0.0, "W0": w_prev}
            new["n"] += 1
            new["vol"] += vol if vol == vol else math.nan
        elif new is not None:
            det["new"] = None
            self._check_shift(ctx, s, e, model, det, new, t_end)

    def _check_shift(self, ctx: Context, s: str, e: str, model: Dict[str, Any],
                     det: Dict[str, Any], new: Dict[str, Any], t_end: float) -> None:
        miss = det["miss"]
        if miss is None or new["j0"] <= miss["j0"]:
            return
        n0, n1 = miss["n"], new["n"]
        if abs(n1 - n0) > max(1.0, SHIFT_DUR_TOL * n0):
            return
        ve, vn = miss["vol_exp"], new["vol"]
        ratio = vn / ve if (ve == ve and ve > 0.0 and vn == vn) else math.nan
        if ratio == ratio and not SHIFT_VOL_RANGE[0] <= ratio <= SHIFT_VOL_RANGE[1]:
            return
        c0 = (miss["j0"] + n0 / 2.0) % R.SLOTS_PER_DAY
        c1 = (new["j0"] + n1 / 2.0) % R.SLOTS_PER_DAY
        shift_h = TB.hour_distance(c1 / 4.0, c0 / 4.0)
        if not abs(shift_h) > SHIFT_MIN_H:
            return
        now = float(ctx.now)
        det["miss"] = None
        det["W"] = 0.0                                   # the evidence is explained
        det["s"] = 0.0
        det["explained_until"] = now + SHIFT_EXPLAIN_S
        rec = {"ts": now, "from": R._hhmm(miss["j0"]), "to": R._hhmm(new["j0"]),
               "from_slot": int(miss["j0"]), "to_slot": int(new["j0"]),
               "shift_h": round(float(shift_h), 3), "n_slots": int(n1),
               "vol_ratio": None if not ratio == ratio else round(float(ratio), 3)}
        shifts = model.setdefault("shifts", [])
        shifts.append(rec)
        del shifts[:-SHIFTS_KEEP]
        if ctx.training:
            return                                       # warm-up: no events
        t_miss = miss["t_end"] - (n0 - 1) * SLOT_S - SLOT_S
        ctx.store.add_event(BehaviorEvent(
            system=s, entity=e, ts=now, kind="schedule_shift", score=0.3,
            severity=Severity.LOW,
            description=(f"schedule moved {rec['from']} -> {rec['to']} "
                         f"({shift_h:+.2f} h, {n1} slots)"),
            extra={**rec, "cap": "low", "shift_explained": True},
            axes=list(AXES), dedupe_key=f"schedule_shift|{s}|{e}|{rec['from']}|{rec['to']}",
            model_version=int(model["gate"].version), window=(float(t_miss), float(t_end))))

    # ---------------------------------------------------------------- writes
    def _write(self, ctx: Context, s: str, e: str, model: Dict[str, Any], det: Dict[str, Any],
               W: float, p_cur: float, a_cur: float, w_slot: Optional[int],
               finals: List[Tuple[int, float]]) -> None:
        store, now, dt = ctx.store, float(ctx.now), float(ctx.window_s)
        machine = bool(model.get("machine_like", False))
        s_sil = det["s"] if machine else math.nan
        pm_off = 2.0 ** (-W)
        pm_sil = math.exp(-s_sil) if s_sil == s_sil else math.nan
        on_off = int(W >= self.h_off)
        on_sil = int(s_sil == s_sil and s_sil >= self.h_sil)
        if ctx.training:
            on_off = on_sil = 0
        explained = det["explained_until"] > now
        axes = {}
        if on_off or pm_off <= AXES_PM_MAX:
            axes["offhours"] = list(AXES)
        if on_sil or (pm_sil == pm_sil and pm_sil <= AXES_PM_MAX):
            axes["silence"] = list(AXES)
        det["alarm"] = {"offhours": on_off, "silence": on_sil}
        emit.write_scores(store, s, e, now, {"offhours": W, "silence": s_sil},
                          pm={"offhours": pm_off, "silence": pm_sil},
                          axes=axes or None, acc_alarm=dict(det["alarm"]), window_s=int(dt))
        val = {"p_expected": _r(p_cur), "W_off": _r(W), "s_sil": _r(s_sil),
               "active": _r(a_cur), "h_off": _r(self.h_off), "h_sil": _r(self.h_sil),
               "shift_explained": int(explained), "machine_like": machine,
               "w_slot": w_slot,
               "slots": [[int(j), _r(a)] for j, a in finals][-8:]}
        store.add_derived(DerivedMetric(name=SERIES, value=val, ts=now, system=s, entity=e,
                                        window_s=int(dt), kind=MetricKind.CATEGORICAL,
                                        inputs=[ACTIVE, "act.stream"]))
        prof = store.profile(s, e)
        if prof is None:
            prof = EntityProfile(system=s, entity=e, updated=now)
            store.put_profile(prof)
        rh = dict(model.get("desc") or {})
        rh.update({"W_off": val["W_off"], "s_sil": val["s_sil"], "p_expected": val["p_expected"],
                   "shift_explained": bool(explained), "version": int(model["gate"].version),
                   "shifts": list(model.get("shifts") or [])[-4:],
                   "prior_tier": (model.get("prior") or {}).get("tier", "hyper")})
        prof.extra["rhythm"] = rh

    # ------------------------------------------------------------ class / system
    @staticmethod
    def _system_model(store, s: str) -> Optional[Dict[str, Any]]:
        m = store.get_model(s, SYSTEM_KEY, MODEL)
        return m if isinstance(m, dict) and m.get("kind") == "system" else None

    def _refresh_prior(self, store, s: str, e: str, model: Dict[str, Any], now: float) -> None:
        """Entity prior from its class (strength 6), else the system tier (2),
        else the Jeffreys hyperprior; refreshed hourly (tier pools move slowly)."""
        if not self.entity_due(("rhythm-prior", s, e), now, PRIOR_REFIT_S):
            return
        ck = m_class.class_key(store, s, e)
        prior = {"tier": "hyper", "pi48": None, "s": R.HYPER_S}
        for key, strength, tier, n_min in ((ck, R.CLASS_S, ck, CLASS_MIN_MEMBERS),
                                           (SYSTEM_KEY, R.SYSTEM_S, "system", SYSTEM_MIN_MEMBERS)):
            if not key:
                continue
            tm = store.get_model(s, key, MODEL)
            pi = tm.get("pi48") if isinstance(tm, dict) else None
            if pi is not None and int(tm.get("n_members", 0) or 0) >= n_min:
                prior = {"tier": tier, "pi48": np.asarray(pi, dtype=np.float64), "s": strength}
                break
        model["prior"] = prior

    def _refit_tiers(self, store, s: str, now: float, used: Set[str]) -> None:
        """Hourly: pooled member rhythms -> model.rhythm@(s, class:<rid>) and
        @(s, __system__) (expected active fraction per slot; class prior)."""
        if self.entity_due(("rhythm-tier", s, SYSTEM_KEY), now, self.tier_refit_s):
            self._build_tier(store, s, SYSTEM_KEY, store.entities(s), now, "system")
        for ck in sorted(used):
            if self.entity_due(("rhythm-tier", s, ck), now, self.tier_refit_s):
                self._build_tier(store, s, ck, m_class.class_members(store, s, ck), now, "class")

    def _build_tier(self, store, s: str, key: str, members: Sequence[str], now: float,
                    kind: str) -> None:
        tm = R.new_model(kind)
        st = tm["state"]
        mem: List[str] = []
        for m in members:
            o = _other_state(store, s, m)
            if o is None or not math.isfinite(o.get("t_ref", math.nan)):
                continue
            g = R.decay_factor(o["t_ref"], now)
            for k in ("A48", "N48", "V48", "A168", "N168"):
                st[k] += g * np.asarray(o[k], dtype=np.float64)
            for k, better in (("t_first", min), ("t_last", max)):
                a, b = st[k], o.get(k, math.nan)
                st[k] = b if not math.isfinite(a) else (a if not math.isfinite(b) else better(a, b))
            st["n_slots"] += int(o.get("n_slots", 0))
            mem.append(m)
        if not mem:
            return
        st["t_ref"] = now
        prev = store.get_model(s, key, MODEL)
        prev = prev if isinstance(prev, dict) else {}
        rev = int(prev.get("rev", 0)) + 1
        h = R.entropy168(tm)
        tm.update(version=rev, rev=rev, updated=now, members=sorted(mem), n_members=len(mem),
                  pi48=R.prior_pi48(st["A48"], st["N48"]), entropy168=h,
                  machine_like=bool(math.isfinite(h) and h <= R.ENTROPY_MAX))
        if math.isfinite(h) and (not prev.get("desc") or self.entity_due(
                ("rhythm-tier-desc", s, key), now, DESC_REFIT_S)):
            tm["desc"] = R.descriptors(tm)
        else:
            tm["desc"] = prev.get("desc") or {}
        if kind == "system":
            tm["healed"] = {int(d): t for d, t in (prev.get("healed") or {}).items()
                            if now - float(t) <= HEAL_KEEP_S}
        store.put_model(s, key, MODEL, tm, version=rev)

    def _heal(self, store, s: str, sysm: Optional[Dict[str, Any]], clock: _Clock,
              presence: Dict[int, List[float]], now: float) -> None:
        """Calendar self-healing: a nonworkday on which the system is present at
        > 3x its usual nonworkday level is treated as a workday."""
        if sysm is None:
            return
        healed = sysm.setdefault("healed", {})
        st = sysm["state"]
        for j, (n_act, n_obs) in sorted(presence.items()):
            day = j // R.SLOTS_PER_DAY
            if day in healed or not clock.nonwork_by_calendar(j):
                continue
            if n_act < HEAL_MIN_ACTIVE or n_obs <= 0.0:
                continue
            c48, c168 = clock.cells(j)
            if not float(st["N48"][c48]) >= HEAL_MIN_N:
                continue
            usual = R.p_cell(sysm, c48, c168)
            frac = n_act / n_obs
            if frac >= HEAL_MIN_FRAC and usual == usual and frac > HEAL_FACTOR * usual:
                healed[int(day)] = now
                clock.heal(day)
                sysm["rev"] = int(sysm.get("rev", 0)) + 1


# ================================================================ helpers
def _held_W(det: Dict[str, Any]) -> Optional[float]:
    """W before the current unusual run while that run may still be the moved
    usual window (<= its duration + tolerance), else None. Its off-hours
    evidence is held until the run either completes the shift (explained,
    W reset) or can no longer be one (then the full W is reported): at most
    n0 + tol slots of delay, and only right after a machine-like entity
    silently missed its window."""
    miss, new = det.get("miss"), det.get("new")
    if miss is None or new is None:
        return None
    if new["n"] > miss["n"] + max(1.0, SHIFT_DUR_TOL * miss["n"]):
        return None
    return float(new.get("W0", det["W"]))


def _shift_pending(det: Dict[str, Any], p: float) -> bool:
    """A missed usual window awaits its moved run, and this unusual slot may
    still complete it (run length <= the window's duration + tolerance)."""
    miss = det.get("miss")
    if miss is None or not (p == p and p < R.P_USUAL):
        return False
    n = (det["new"]["n"] if det.get("new") else 0) + 1
    return n <= miss["n"] + max(1.0, SHIFT_DUR_TOL * miss["n"])


def _other_state(store, s: str, e: str) -> Optional[Dict[str, Any]]:
    m = store.get_model(s, e, MODEL)
    return m.get("state") if isinstance(m, dict) and m.get("kind") == "entity" else None


def _ring_min(store, s: str, e: str, name: str, t0: float, t1: float) -> float:
    _, M = store.vec_range(s, e, name, t0, t1)
    if not len(M):
        return math.nan
    v = np.asarray(M, dtype=np.float64).reshape(len(M), -1)[:, 0]
    v = v[np.isfinite(v)]
    return float(min(1.0, max(0.0, v.min()))) if v.size else math.nan


def _ring_at(store, s: str, e: str, name: str, ts: float) -> float:
    r = store.vec_at(s, e, name, ts)
    if r is None or not len(r):
        return math.nan
    v = float(r[0])
    return min(1.0, max(0.0, v)) if v == v else math.nan


def _fresh_float(store, s: str, e: str, name: str, now: float) -> float:
    v = store.latest_fresh(s, e, name, now)
    try:
        return float(v) if v is not None else math.nan
    except (TypeError, ValueError):
        return math.nan


def _prune(model: Dict[str, Any], now: float) -> None:
    """Ledger kept LEDGER_KEEP_S (both dicts are in insertion = time order);
    stale open accumulators (a gap stranded them) are dropped."""
    led = model["ledger"]
    ticks, slots = led["ticks"], led["slots"]
    cut = now - LEDGER_KEEP_S
    while ticks:
        tc = next(iter(ticks))
        if tc >= cut:
            break
        for j in ticks.pop(tc):
            slots.pop(j, None)
    acc = model["det"]["acc"]
    if len(acc) > 8:
        top = max(acc)
        for j in [j for j in acc if j < top - ACC_KEEP_S / SLOT_S]:
            del acc[j]


def _r(x: Any) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, 6) if math.isfinite(x) else None
