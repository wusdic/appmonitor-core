"""Session engine (D2) — duty cycle, sessions and think time on wall-clock time.

Why v1 was dead: it segmented the last 60 *emitted* http.requests points by
tick index, so idle time (ticks with no point) vanished from the duty cycle,
and "think time" was a count of idle ticks times window_s — a quantity of the
tick cadence, not of the person. v2 (engines.md D2) measures time:

* activity_duty_cycle = share of the last 24 h during which the entity was
  active, on the grid clock of R2's zero-filled act.events (derived/fresh.py:
  a tick with act.events == 0 is a true idle tick). Ticks are weighted by
  their wall-clock length, so a cadence switch 900 s -> 60 s does not let the
  short ticks outvote the long ones; at a uniform cadence this is exactly the
  fraction of active ticks.
* Sessions are built from act.stream event timestamps (sub-second) with the
  idle threshold G = the entity's model.seq session_gap (B10's gap valley),
  else 30 min. session_count / req_per_session describe the sessions that
  COMPLETED in the last 24 h (a session is complete once the entity has been
  idle for more than G). Events are reweighted by 1/act.stream_frac (R2's cap
  drops whole sub-sessions); the session count itself is not, because R2 cuts
  at SESSION_GAP_S = 30 s, far below G, so a dropped sub-session almost never
  removes a whole D2 session.
* think_time_s_avg = median within-session inter-event gap observed THIS tick,
  written only when there are >= 2 such gaps (fresh, kind avg); a tick
  without them writes nothing — absence is data, never a stale re-stamp.
  When R2 sampled the tick (stream_frac < 1) only gaps <= SESSION_GAP_S are
  certain to be true consecutive-event gaps (m_template sampling contract),
  so longer ones are not used as think time (they still link sessions). A
  gap of exactly 0 (records sharing one obs.ts) carries no timing and is
  skipped.

Window outputs are written at ctx.now with dims {span_s, n_active} whenever
R2 wrote the entity's act.events clock this tick (R2 zero-fills every entity
seen in 30 d), so an idle entity keeps reporting a decaying duty cycle and
session count instead of freezing its last busy value. req_per_session is
undefined (not written) while no session completed in the span.

State is per engine instance and incremental (each tick reads only the new
clock points and stream ticks): act.stream is kept 1 h by the store, far
shorter than the 24 h session window. The state is a representation, not a
model of normality, so it is not trust-gated. A clock that goes backwards
(replay) resets the entity.
"""
from __future__ import annotations

import math
import weakref
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import DerivedMetric, MetricKind
from ..behavior.lib import m_seq as MS
from ..behavior.lib import m_template as MT
from .fresh import CLOCK

DAY = 86400.0
SPAN_S = DAY                     # duty / session window
MAX_TICK_S = 3600.0              # longest cadence class: caps a tick's weight
                                 # after a pipeline outage
MIN_THINK_GAPS = 2
IDLE_SWEEP_S = 2 * DAY           # drop state of entities not seen this long
RETAIN_INPUTS = ("act.events",)  # grid clock must reach back SPAN_S

OUT_SESSIONS = "derived.session_count"
OUT_RPS = "derived.req_per_session"
OUT_THINK = "derived.think_time_s_avg"
OUT_DUTY = "derived.activity_duty_cycle"

_RETAINED: "weakref.WeakSet" = weakref.WeakSet()


def ensure_retention(store) -> None:
    """Keep act.events for 24 h (contract B + reviewer correction; D0 sets the
    same rule). Idempotent; done once per store because set_retention clears
    the store's rule cache."""
    if store in _RETAINED:
        return
    for name in RETAIN_INPUTS:
        store.set_retention(name, max_age_s=SPAN_S)
    _RETAINED.add(store)


def session_gap(store, system: str, entity: str) -> float:
    """G: model.seq session_gap (B10) clipped to B10's range; 30 min default."""
    g = MS.session_gap(MS.get(store, system, entity), MS.DEFAULT_SESSION_GAP_S)
    return min(MS.SESSION_GAP_MAX_S, max(MS.SESSION_GAP_MIN_S, g))


class _EntityState:
    """Incremental per-entity state: the 24 h activity clock and sessions."""

    __slots__ = ("last_now", "clock_ts", "clock", "tot", "act", "nact",
                 "stream_ts", "open", "done", "done_w", "last_full")

    def __init__(self) -> None:
        self.last_now = -math.inf
        self.clock_ts = -math.inf              # newest act.events tick consumed
        self.clock: Deque[Tuple[float, float, bool]] = deque()   # (ts, dur, active)
        self.tot = 0.0                          # sum dur over the clock deque
        self.act = 0.0                          # sum dur of active ticks
        self.nact = 0
        self.stream_ts = -math.inf              # newest act.stream tick consumed
        self.open: Optional[List[float]] = None  # [start_ts, last_ts, weight]
        self.done: Deque[Tuple[float, float]] = deque()   # (end_ts, weight)
        self.done_w = 0.0
        self.last_full = False                  # previous stream tick unsampled

    # ---------------------------------------------------------------- clock
    def push_tick(self, ts: float, dur: float, active: bool) -> None:
        self.clock.append((ts, dur, active))
        self.tot += dur
        if active:
            self.act += dur
            self.nact += 1
        self.clock_ts = ts

    def prune(self, now: float) -> None:
        lo = now - SPAN_S
        c = self.clock
        while c and c[0][0] <= lo:
            _, dur, a = c.popleft()
            self.tot -= dur
            if a:
                self.act -= dur
                self.nact -= 1
        if not c:
            self.tot = self.act = 0.0
            self.nact = 0
        d = self.done
        while d and d[0][0] <= lo:
            self.done_w -= d.popleft()[1]
        if not d:
            self.done_w = 0.0

    def duty(self) -> float:
        if not self.tot > 0.0:
            return math.nan
        return min(1.0, max(0.0, self.act / self.tot))

    # ------------------------------------------------------------- sessions
    def close(self, end_ts: float, weight: float) -> None:
        self.done.append((end_ts, weight))
        self.done_w += weight


class SessionEngine(Engine):
    name = "derived.session"
    layer = "derived"
    consumes = ["act.events", "act.stream", "act.stream_frac", "model.seq"]
    produces = [OUT_SESSIONS, OUT_RPS, OUT_THINK, OUT_DUTY]
    description = ("Wall-clock duty cycle (24 h act.events grid), sessions from act.stream "
                   "timestamps split at model.seq session_gap, fresh median think time.")
    interval = 1

    def __init__(self, **p) -> None:
        super().__init__(**p)
        self._state: Dict[Tuple[str, str], _EntityState] = {}
        self._swept: Optional[float] = None

    # ------------------------------------------------------------------- run
    def run(self, ctx: Context, observations=None) -> int:
        store, now = ctx.store, float(ctx.now)
        ensure_retention(store)
        dt = float(ctx.window_s) if ctx.window_s and ctx.window_s > 0 else 60.0
        n = 0
        for system in store.systems():
            for entity in store.entities(system):
                n += self._entity(ctx, system, entity, now, dt)
        self._sweep(now)
        return n

    def _entity(self, ctx: Context, s: str, e: str, now: float, dt: float) -> int:
        key = (s, e)
        st = self._state.get(key)
        if st is not None and now < st.last_now:
            st = None                           # replay: the clock went backwards
        if st is not None and now == st.last_now:
            return 0                            # this tick is already written
        store = ctx.store
        clock_new = _points_after(store, s, e, CLOCK,
                                  st.clock_ts if st is not None else now - SPAN_S, now)
        if st is None:
            if not clock_new and not store.raw_tail(s, e, "act.stream", 1):
                return 0                        # not on R2's grid: nothing to say
            st = self._state[key] = _EntityState()
        st.last_now = now

        # 1) activity clock (the act.events grid)
        prev = st.clock_ts if math.isfinite(st.clock_ts) else None
        for i, (ts, v) in enumerate(clock_new):
            if prev is None:
                dur = (clock_new[i + 1][0] - ts) if i + 1 < len(clock_new) else dt
            else:
                dur = ts - prev
            dur = min(max(dur, 1.0), MAX_TICK_S)
            active = isinstance(v, (int, float)) and math.isfinite(v) and v > 0.0
            st.push_tick(ts, dur, bool(active))
            prev = ts

        # 2) sessions and think time from the new act.stream ticks
        G = session_gap(store, s, e)
        think: Optional[Tuple[float, int]] = None
        since = st.stream_ts if math.isfinite(st.stream_ts) else now - SPAN_S
        head = store.raw_tail(s, e, "act.stream", 1)
        ticks = MT.stream_ticks(store, s, e, since, now) \
            if head and head[-1].ts > st.stream_ts else ()
        for tick_ts, rows, frac in ticks:
            if tick_ts <= st.stream_ts:
                continue
            st.stream_ts = tick_ts
            gaps = self._consume(st, rows, frac, G)
            if tick_ts == now and gaps is not None and gaps.size >= MIN_THINK_GAPS:
                think = (float(np.median(gaps)), int(gaps.size))
        if st.open is not None and now - st.open[1] > G:
            st.close(st.open[1], st.open[2])
            st.open = None
        st.prune(now)

        # 3) outputs
        n = 0
        if think is not None:
            self._emit(ctx, s, e, OUT_THINK, think[0], MetricKind.GAUGE,
                       {"n": think[1], "session_gap_s": G}, ["act.stream", "act.stream_frac"])
            n += 1
        if clock_new and clock_new[-1][0] == now and st.clock:
            dims = {"span_s": SPAN_S, "n_active": int(st.nact)}
            self._emit(ctx, s, e, OUT_DUTY, st.duty(), MetricKind.RATE, dims, [CLOCK])
            k = len(st.done)
            sdims = dict(dims, n_sessions=k, session_gap_s=G)
            src = ["act.stream", "act.stream_frac", "model.seq"]
            self._emit(ctx, s, e, OUT_SESSIONS, float(k), MetricKind.GAUGE, sdims, src)
            n += 2
            if k > 0:
                self._emit(ctx, s, e, OUT_RPS, st.done_w / k, MetricKind.GAUGE, sdims, src)
                n += 1
        return n

    @staticmethod
    def _consume(st: _EntityState, rows: np.ndarray, frac: float,
                 G: float) -> Optional[np.ndarray]:
        """Fold one stream tick into the session state; return its think-time
        gaps (within-session, positive, true consecutive-event gaps)."""
        t = np.asarray(rows["ts"], dtype=np.float64) if len(rows) else np.zeros(0)
        t = t[np.isfinite(t)]
        if not t.size:
            st.last_full = True
            return None
        if t.size > 1 and np.any(np.diff(t) < 0.0):
            t = np.sort(t, kind="stable")
        # a missing / invalid frac means the rows are complete (m_template)
        sampled = bool(frac == frac and 0.0 < frac < 1.0 - 1e-9)
        w = 1.0 / frac if sampled else 1.0
        d = np.diff(t)
        # session cuts inside the tick (gap > G); negative / zero gaps join
        cut = np.flatnonzero(d > G) + 1
        within = d[(d > 0.0) & (d <= G) & ((d <= MT.SESSION_GAP_S) if sampled else True)]
        # boundary with the previous tick's open session
        op = st.open
        g0 = math.nan
        if op is not None:
            g0 = float(t[0]) - op[1]
            if g0 > G:
                st.close(op[1], op[2])
                op = st.open = None
        if op is not None and st.last_full and not sampled and 0.0 < g0 <= G:
            within = np.concatenate(([g0], within))
        st.last_full = not sampled
        bounds = [0, *cut.tolist(), int(t.size)]
        for j in range(len(bounds) - 1):
            a, b = bounds[j], bounds[j + 1]
            wt = (b - a) * w
            if j == 0 and op is not None:          # continues the open session
                op[1] = max(op[1], float(t[b - 1]))
                op[2] += wt
            else:
                st.open = [float(t[a]), float(t[b - 1]), wt]
            if j < len(bounds) - 2:                # a later cut ends this session
                st.close(st.open[1], st.open[2])
                st.open = None
        return within

    @staticmethod
    def _emit(ctx: Context, s: str, e: str, name: str, value: float, kind: MetricKind,
              dims: Dict[str, float], inputs: List[str]) -> None:
        ctx.store.add_derived(DerivedMetric(
            name=name, value=float(value), ts=ctx.now, system=s, entity=e,
            window_s=int(ctx.window_s), kind=kind, inputs=list(inputs), dims=dict(dims)))

    def _sweep(self, now: float) -> None:
        """Forget entities not processed for IDLE_SWEEP_S (bounded memory)."""
        if self._swept is not None and 0.0 <= now - self._swept < DAY:
            return
        self._swept = now
        for k in [k for k, st in self._state.items() if now - st.last_now > IDLE_SWEEP_S]:
            del self._state[k]

    # --------------------------------------------------------------- inspect
    def state_of(self, system: str, entity: str) -> Optional[Dict[str, float]]:
        """Debug / test view of an entity's state."""
        st = self._state.get((system, entity))
        if st is None:
            return None
        return {"duty": st.duty(), "n_active": st.nact, "n_ticks": len(st.clock),
                "sessions": len(st.done), "events": st.done_w,
                "open": None if st.open is None else list(st.open)}


def _points_after(store, s: str, e: str, name: str, after: float,
                  now: float) -> List[Tuple[float, float]]:
    """[(ts, value)] of raw `name` with after < ts <= now, oldest first. Reads
    a short tail and doubles it only while it may still miss points, so the
    usual tick costs one or two islice reads instead of a 24 h copy."""
    n = 4
    while True:
        tail = store.raw_tail(s, e, name, n)
        if not tail or len(tail) < n or tail[0].ts <= after or n >= 1 << 16:
            break
        n *= 4
    out: List[Tuple[float, float]] = []
    for m in tail:
        if after < m.ts <= now:
            v = m.value
            out.append((float(m.ts), float(v) if isinstance(v, (int, float)) else math.nan))
    return out
