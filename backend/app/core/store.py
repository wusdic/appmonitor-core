"""In-memory time-series store shared by every engine.

Deliberately small and dependency-free (numpy aside). In a real deployment this
interface is the seam where you drop in ClickHouse / VictoriaMetrics /
TimescaleDB: engines only ever call the methods below. They never assume a
storage backend, which is what lets the store be swapped without touching a
single engine.

v2 (lib-3, contract B/D) adds, fully backward compatible with v1:

* float32 **vector rings** (`add_vec`) for per-tick feature / score vectors,
  with **virtual scalar views** (`register_vector_names`) so e.g.
  `feature.bytes_up` is a column of `feature.vec`, never a second copy. Storing
  52 scalar DerivedMetric objects per tick is what used to blow memory up.
* O(1) tail / freshness reads (`raw_tail`, `latest_fresh`) instead of copying a
  20000-object deque per call.
* first_seen / last_seen maintained on raw ingest (absence is data), and a
  guard that keeps pseudo-entities (`__system__`, `class:*`) out of raw data.
* ts-indexed events / matches (bisect, O(log n + k)) instead of linear scans.
* labels, incidents, versioned models, geometric checkpoints, profile
  versions, engine health and prefix-based retention.
"""
from __future__ import annotations

import bisect
import itertools
import math
import sys
import threading
from collections import defaultdict, deque
from itertools import islice
from typing import (Any, Deque, Dict, Iterable, Iterator, List, Mapping, NamedTuple, Optional,
                    Sequence, Tuple, Union)

import numpy as np

from ..models.schema import (
    BehaviorEvent,
    DerivedMetric,
    EntityProfile,
    Incident,
    Label,
    MetricKind,
    Observation,
    RawMetric,
    SignatureMatch,
    is_pseudo_entity,
)

HOUR = 3600.0
DAY = 86400.0

# Default retention (contract B). Longest matching name prefix wins; a rule is
# (max_points, max_age_s), None meaning "no limit on that axis". Pruning is
# relative to the newest ts of the series itself, so a quiet series is not
# emptied just because time passed.
DEFAULT_RETENTION: Dict[str, Tuple[Optional[int], Optional[float]]] = {
    # raw categorical sets / streams (raw scalars fall back to 6 h, see below)
    "act.stream": (None, 1 * HOUR),
    "act.stream_frac": (None, 6 * HOUR),
    "act.rare_events": (None, 1 * HOUR),
    "act.tokens": (None, 1 * HOUR),
    "act.objs": (None, 1 * HOUR),
    "client.": (None, 1 * HOUR),
    # derived (lib-2)
    "derived.": (None, 2 * HOUR),
    # lib-3 long rings: needed for rollback and rebase replay (8 d)
    "feature.nat": (None, 8 * DAY),
    "feature.expo": (None, 8 * DAY),
    "feature.active": (None, 8 * DAY),
    "feature.tctx": (None, 8 * DAY),
    "behavior.trust": (None, 8 * DAY),          # also trust_prov
    "behavior.quarantine": (None, 8 * DAY),
    "behavior.regime": (None, 8 * DAY),
    "behavior.risk": (None, 8 * DAY),
    "behavior.q_inst": (None, 8 * DAY),
    "behavior.q_all": (None, 8 * DAY),
    "behavior.evidence": (None, 8 * DAY),
    "behavior.alarm": (None, 8 * DAY),
    # 1 d
    "feature.vec": (None, 1 * DAY),
    "feature.sketch": (None, 1 * DAY),
    "behavior.score": (None, 1 * DAY),
    "behavior.pm": (None, 1 * DAY),
    "behavior.p": (None, 1 * DAY),              # also p_family (explicit below)
    "behavior.p_family": (None, 1 * DAY),
    "behavior.axes": (None, 1 * DAY),
    "ops.engine_health": (None, 1 * DAY),
    # 6 h
    "behavior.z": (None, 6 * HOUR),             # also zr, zi
    "behavior.zr": (None, 6 * HOUR),
    "behavior.zi": (None, 6 * HOUR),
    "behavior.pf": (None, 6 * HOUR),
    "behavior.cusum_state": (None, 6 * HOUR),
    # integration (engine_integration_notes): rules the engines used to set
    # themselves, now part of contract B's table
    "http.requests": (None, 24 * HOUR),         # D0 grid inputs (D0 / D2: 24 h)
    "l4.flows": (None, 24 * HOUR),
    "dns.queries": (None, 24 * HOUR),
    "act.events": (None, 24 * HOUR),
    "l4.bytes_up": (None, 13 * HOUR),           # D0 trend targets (12 h span + 1 h)
    "l4.distinct_peers": (None, 13 * HOUR),
    "http.latency_ms_avg": (None, 13 * HOUR),
    "probe.rtt_ms": (None, 13 * HOUR),
    "behavior.e_day": (None, 8 * DAY),          # as q_all (B25)
    "behavior.budget": (24, 1 * DAY),           # 36 entries per point (B13)
    "behavior.class.agg": (None, 9 * DAY),      # B18 reference replay clock (24 h + 8 d)
    "behavior.calib_health": (None, 8 * DAY),   # B24
    # spec v2.1 grain series (docs/lib3/cadence.md §5.2); written in
    # canonical grain mode only, so tick mode keeps v2's memory
    "l4.peer_ids": (None, 75 * 60.0),           # SetSketches: >= G_h + max dt (<= 900)
    "l4.dport_ids": (None, 75 * 60.0),
    "tls.ja3_ids": (None, 75 * 60.0),
    "act.template_ids": (None, 75 * 60.0),
    "act.slot_events": (None, 25 * HOUR),       # joins the D0 / D2 inputs
    "feature.part": (None, 2 * HOUR),
    "feature.live": (None, 6 * HOUR),
    "feature.meta": (None, 8 * DAY),
    "feature.vec.h": (None, 2 * DAY),           # identity windows look back 48 h
    "feature.vec.q": (None, 2 * DAY),
    "feature.sketch.h": (None, 2 * DAY),
    "behavior.prov": (None, 1 * DAY),
    "behavior.wh.q": (None, 6 * HOUR),
    "behavior.common.q": (None, 6 * HOUR),
    "behavior.q_inst.h": (None, 8 * DAY),
    "behavior.evidence.h": (None, 8 * DAY),
    # B21 (P2): its gating clock, one row per scored (system, IP) H window
    "behavior.xsys": (None, 9 * DAY),
    # retention audit (perf, integration.md §9): lib-3 series that had no rule
    # (20000 points: 208 d at 900 s) get the 8-d lib-3 horizon. Every engine
    # reads them at now or a few points back; behavior.seq.class_llr is B10's
    # gated-learner clock (replay <= 192 h, like feature.nat).
    "behavior.acc_alarm": (None, 8 * DAY),
    "behavior.rhythm": (None, 8 * DAY),
    "behavior.timing": (None, 8 * DAY),
    "behavior.id": (None, 8 * DAY),
    "behavior.class": (None, 8 * DAY),          # behavior.class.agg keeps its 9 d
    "behavior.common.": (None, 8 * DAY),        # behavior.common.q keeps its 6 h
    "behavior.cp.": (None, 8 * DAY),
    "behavior.seq.": (None, 8 * DAY),
}
# Per-tick DICT series (derived space only): a point cap on top of the age
# rule (round 4, gate 14). An age rule holds 15x the points at 60 s that it
# holds at 900 s (8 d = 11520 dicts per series and entity at 60 s, ~163 MB per
# entity extrapolated to 8 d), while every reader of these series looks back
# a bounded number of POINTS. Reader audit (every derived_tail / derived_series
# / latest_* / emit.read_dict call site, round 4):
#   feature.tctx         B24 _tctx_at <= 64, B29 <= 256, B17 <= 4 x 96 (zi-ring
#                        restart), m_identity <= 64; B15's tick-mode fetch and
#                        B24 / B29 recompute the same tctx from the config when
#                        the row is gone (timebins, "the function B01 uses")
#   feature.expo         newest only (B18 latest_fresh; B03 / B04 read the
#                        exposure from feature.nat)
#   behavior.axes        newest only (emit.read_dict <= 4 points; B25, B26)
#   behavior.degraded    newest only (write-merge, B24, API health panel)
#   behavior.timing      m_identity grain / tick rows <= 64 points; B15's
#                        tick-mode fetch within feature.vec's 1 d (96 at 900 s)
#   behavior.acc_alarm   B27 / B28 tail 1, emit.read_dict 4, B29 at t_open
#   behavior.alarm       B26 / B28 newest, B29 at t_open (depth = ticks since)
#   behavior.rhythm      B29 at t_open
#   behavior.regime      B03 incremental scan (points since its last scan),
#                        B18 class keys 256, B27 8, API history (thinned)
#   behavior.id / class / common.* / calib_health / prov: newest few points,
#                        B29 at t_open, the thinned API history
# DICT_POINT_CAP (1536) is >= 8 d at 900 s (768 points), so those series are
# unchanged at dt >= 900 s (the age rule binds first), and >= 1 d at 60 s, so
# B29's replay at an incident's opening tick keeps a day at 60 s. The
# smaller caps are >= every audited lookback of their series at any cadence.
# behavior.p_family is NOT capped: B27's risk trigger, B29 and B23 fold it
# over 24 h at tick resolution (<= 2000 / 1500 points), which its 1-d age
# rule already bounds (1440 points at 60 s).
DICT_POINT_CAP = 1536
DICT_POINT_CAPS: Dict[str, Optional[int]] = {
    "feature.tctx": 512,
    "feature.expo": 64,                     # feature.expo.h / .q: grain rows only
    "behavior.axes": 64,
    "behavior.degraded": 64,
    "behavior.timing": 256,
    "behavior.acc_alarm": DICT_POINT_CAP,
    "behavior.alarm": DICT_POINT_CAP,
    "behavior.rhythm": DICT_POINT_CAP,
    "behavior.regime": DICT_POINT_CAP,
    "behavior.id": DICT_POINT_CAP,
    "behavior.class": DICT_POINT_CAP,
    "behavior.common.": DICT_POINT_CAP,
    "behavior.calib_health": DICT_POINT_CAP,
    "behavior.prov": DICT_POINT_CAP,
    "behavior.p_family": None,              # 1 d at tick resolution (see above)
}
# behavior.degraded had no rule (20000 points = 208 d at 900 s)
DEFAULT_RETENTION["behavior.degraded"] = (None, 8 * DAY)
RAW_SCALAR_MAX_AGE = 6 * HOUR
# timeline(): a vec-ring risk point is listed when it enters a new 10-point band
RISK_TIMELINE_BAND = 10.0
RAW_SET_MAX_AGE = 1 * HOUR
EVENT_MAX_AGE = 30 * DAY
INCIDENT_MAX_AGE = 90 * DAY
PROFILE_VERSIONS_KEPT = 12

# Checkpoint retention: geometric over this horizon, at most this many per key.
CHECKPOINT_BANDS_H = (1, 2, 4, 8, 16, 32, 64, 128, 168)
CHECKPOINT_RECENT_MAX = 7          # checkpoints kept inside the newest 24 h
CHECKPOINT_MAX = 1 + 8 + CHECKPOINT_RECENT_MAX   # floor + daily backbone + recent
CHECKPOINT_HORIZON_S = CHECKPOINT_BANDS_H[-1] * HOUR

_VEC_INIT_CAP = 64


class ProfileVersion(NamedTuple):
    version: Any
    ts: float
    obj: Any


# --------------------------------------------------------------------------- #
# Internal containers
# --------------------------------------------------------------------------- #
class _VecRing:
    """Preallocated float32 ring of fixed dim with a float64 timestamp array.

    Capacity grows by doubling up to `max_cap` (so an entity that is seen for
    an hour does not pay for 8 days), after which the oldest row is
    overwritten. Timestamps are non-decreasing: re-writing the newest ts
    overwrites that row (idempotent re-run of a tick), an older ts is an
    error because every reader bisects on ts."""

    __slots__ = ("dim", "cap", "max_cap", "ts", "data", "start", "n", "window_s")

    def __init__(self, dim: int, max_cap: int) -> None:
        self.dim = dim
        self.max_cap = max(1, int(max_cap))
        self.cap = min(_VEC_INIT_CAP, self.max_cap)
        self.ts = np.empty(self.cap, dtype=np.float64)
        self.data = np.empty((self.cap, dim), dtype=np.float32)
        self.start = 0
        self.n = 0
        self.window_s = 0

    # -- physical/logical mapping
    def _phys(self, i: int) -> int:
        return (self.start + i) % self.cap

    def last_ts(self) -> Optional[float]:
        return float(self.ts[self._phys(self.n - 1)]) if self.n else None

    def first_ts(self) -> Optional[float]:
        return float(self.ts[self.start]) if self.n else None

    def _resize(self, new_cap: int) -> None:
        keep = min(self.n, new_cap)
        idx = self._phys_range(self.n - keep, self.n)
        ts = np.empty(new_cap, dtype=np.float64)
        data = np.empty((new_cap, self.dim), dtype=np.float32)
        ts[:keep] = self.ts[idx]
        data[:keep] = self.data[idx]
        self.ts, self.data, self.cap, self.start, self.n = ts, data, new_cap, 0, keep

    def set_max_cap(self, max_cap: int) -> None:
        max_cap = max(1, int(max_cap))
        self.max_cap = max_cap
        if self.cap > max_cap:
            self._resize(max_cap)

    def append(self, ts: float, row: np.ndarray) -> None:
        if self.n:
            last = self.ts[self._phys(self.n - 1)]
            if ts == last:
                self.data[self._phys(self.n - 1)] = row
                return
            if ts < last:
                raise ValueError(f"vector ring append out of order: ts={ts} < last={last}")
        if self.n == self.cap:
            if self.cap < self.max_cap:
                self._resize(min(self.cap * 2, self.max_cap))
            else:                                  # full: drop the oldest row
                self.start = (self.start + 1) % self.cap
                self.n -= 1
        p = self._phys(self.n)
        self.ts[p] = ts
        self.data[p] = row
        self.n += 1

    def search(self, t: float, side: str = "left") -> int:
        """Logical insertion index of t (np.searchsorted semantics)."""
        if not self.n:
            return 0
        if self.start + self.n <= self.cap:        # not wrapped: one sorted segment
            return int(np.searchsorted(self.ts[self.start:self.start + self.n], t, side=side))
        end1 = min(self.start + self.n, self.cap)
        seg1 = self.ts[self.start:end1]
        i = int(np.searchsorted(seg1, t, side=side))
        if i < len(seg1):
            return i
        seg2 = self.ts[: self.n - len(seg1)]
        return len(seg1) + int(np.searchsorted(seg2, t, side=side))

    def drop_before(self, cutoff: float) -> None:
        if not self.n or self.ts[self.start] >= cutoff:   # common case: nothing expired
            return
        k = self.search(cutoff, "left")
        if k:
            self.start = (self.start + k) % self.cap
            self.n -= k

    def _phys_range(self, i0: int, i1: int) -> Union[slice, np.ndarray]:
        if i1 <= i0:
            return slice(0, 0)
        p0 = self._phys(i0)
        if p0 + (i1 - i0) <= self.cap:
            return slice(p0, p0 + (i1 - i0))
        return (np.arange(i0, i1) + self.start) % self.cap

    def take(self, i0: int, i1: int) -> Tuple[np.ndarray, np.ndarray]:
        i0 = max(0, i0)
        i1 = min(self.n, i1)
        if i1 <= i0:
            return np.empty(0, dtype=np.float64), np.empty((0, self.dim), dtype=np.float32)
        idx = self._phys_range(i0, i1)
        return self.ts[idx].copy(), self.data[idx].copy()

    def row(self, i: int) -> np.ndarray:
        return self.data[self._phys(i)]

    def ts_at(self, i: int) -> float:
        return float(self.ts[self._phys(i)])

    @property
    def nbytes(self) -> int:
        return int(self.ts.nbytes + self.data.nbytes)


class _TsIndex:
    """Time-sorted parallel lists with a lazily advanced head.

    Lists (not deques) because bisect needs O(1) random access; pruning from
    the front just advances `head` and compacts occasionally, so both append
    and prune are amortised O(1) and a `since` query is O(log n + k)."""

    __slots__ = ("ts", "items", "head")

    def __init__(self) -> None:
        self.ts: List[float] = []
        self.items: List[Any] = []
        self.head = 0

    def __len__(self) -> int:
        return len(self.ts) - self.head

    def add(self, ts: float, item: Any) -> None:
        if not self.ts or ts >= self.ts[-1]:
            self.ts.append(ts)
            self.items.append(item)
        else:                                      # late arrival: rare, O(n)
            i = bisect.bisect_right(self.ts, ts, lo=self.head)
            self.ts.insert(i, ts)
            self.items.insert(i, item)

    def oldest(self) -> Tuple[float, Any]:
        return self.ts[self.head], self.items[self.head]

    def remove(self, item: Any) -> None:
        """Remove `item` (by identity); it is expected near the head."""
        i = bisect.bisect_left(self.ts, getattr(item, "ts", -math.inf), lo=self.head)
        n = len(self.ts)
        while i < n and self.items[i] is not item:
            i += 1
        if i >= n:
            return
        if i == self.head:
            self.items[i] = None
            self.head += 1
            if self.head > 1024 and self.head * 2 > len(self.ts):
                del self.ts[: self.head]
                del self.items[: self.head]
                self.head = 0
        else:
            del self.ts[i]
            del self.items[i]

    def newest_first(self, since: Optional[float] = None) -> Iterator[Any]:
        lo = self.head if since is None else bisect.bisect_left(self.ts, since, lo=self.head)
        for i in range(len(self.ts) - 1, lo - 1, -1):
            yield self.items[i]


def _k(system: str, entity: str, name: str) -> str:
    return f"{system}|{entity}|{name}"


_LEAF_TYPES = (float, int, str, bool, type(None))


def _same(a: Any, b: Any, depth: int = 0) -> bool:
    """Exact structural identity of plain values: same types at every level,
    same dict key ORDER, equal leaves (so sharing one object for both can
    change nothing a reader, a JSON dump or a golden hash sees). numpy
    arrays, other objects and nesting deeper than 4 levels are never 'same'
    unless they are the identical object; NaN leaves only if identical."""
    if a is b:
        return True
    ta = type(a)
    if ta is not type(b) or depth > 4:
        return False
    if ta is dict:
        if len(a) != len(b):
            return False
        for (k0, v0), (k1, v1) in zip(a.items(), b.items()):
            if k0 is not k1 and (type(k0) is not type(k1) or k0 != k1):
                return False
            if v0 is not v1 and not _same(v0, v1, depth + 1):
                return False
        return True
    if ta is list or ta is tuple:
        return len(a) == len(b) and all(x is y or _same(x, y, depth + 1) for x, y in zip(a, b))
    if ta in _LEAF_TYPES:
        return a == b
    return False


def _compact(older: Any, row: Any) -> None:
    """Lossless compaction of a derived row that has just stopped being the
    newest of its series (round 4, gate 14): when its dict value (its
    `inputs` / `dims` provenance) equals the previous row's, it shares the
    previous row's object instead of holding a copy. Per-tick state dicts
    repeat tick after tick (measured at 60 s: behavior.regime 98 %,
    behavior.degraded 94 %, calib_health 84 %, acc_alarm 69 %, rhythm 65 %,
    axes 59 %, expo 57 % of rows equal to their predecessor). Only rows that
    are no longer the newest are touched: upsert_dict merges into the newest
    row in place, and no reader mutates a stored value (they copy)."""
    try:
        v0, v1 = older.value, row.value
        if v1 is not v0 and type(v1) is dict and _same(v1, v0):
            row.value = v0
        i0, i1 = older.inputs, row.inputs
        if i1 is not None and i1 is not i0 and _same(i1, i0):
            row.inputs = i0
        d0, d1 = older.dims, row.dims
        if d1 is not None and d1 is not d0 and _same(d1, d0):
            row.dims = d0
    except AttributeError:                  # not a DerivedMetric (tests may store others)
        return


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (bool, int, float, np.integer, np.floating))


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
class MetricStore:
    """Thread-safe bounded in-memory store. Bounded (per-series max_points
    plus age-based retention) so a long-running deployment cannot grow
    without limit."""

    def __init__(self, max_points: int = 20000) -> None:
        self._lock = threading.RLock()
        self._max = max_points
        # keyed scalar/object series
        self._raw: Dict[str, Deque[RawMetric]] = {}
        self._derived: Dict[str, Deque[DerivedMetric]] = {}
        self._raw_names: Dict[Tuple[str, str], set] = defaultdict(set)
        self._derived_names: Dict[Tuple[str, str], set] = defaultdict(set)
        # vector rings + virtual column views
        self._vec: Dict[str, _VecRing] = {}
        self._vec_names: Dict[Tuple[str, str], set] = defaultdict(set)
        self._vec_dim: Dict[str, int] = {}
        self._vec_columns: Dict[str, List[str]] = {}
        self._virtual: Dict[str, Tuple[str, int]] = {}   # virtual name -> (vec name, col)
        # events / matches, ts-indexed
        self._ev_all = _TsIndex()
        self._ev_sys: Dict[str, _TsIndex] = defaultdict(_TsIndex)
        self._ev_ent: Dict[Tuple[str, str], _TsIndex] = defaultdict(_TsIndex)
        self._ev_by_id: Dict[str, BehaviorEvent] = {}
        self._ev_seq = itertools.count(1)
        self._m_all = _TsIndex()
        self._m_sys: Dict[str, _TsIndex] = defaultdict(_TsIndex)
        self._m_ent: Dict[Tuple[str, str], _TsIndex] = defaultdict(_TsIndex)
        # decision objects
        self._labels: List[Label] = []
        self._label_seq = itertools.count(1)
        self._incidents: Dict[str, Incident] = {}
        self._inc_seq = itertools.count(1)
        self._profiles: Dict[str, EntityProfile] = {}
        self._profile_versions: Dict[Tuple[str, str], Deque[ProfileVersion]] = {}
        # models / checkpoints / health
        self._models: Dict[str, Tuple[Any, Any]] = {}      # key -> (obj, version)
        self._model_names: Dict[Tuple[str, str], set] = defaultdict(set)
        self._ckpt: Dict[Tuple[str, str, str], List[Tuple[float, Any]]] = {}
        self._health: Dict[str, Dict[str, Any]] = {}
        self._last_write: Dict[str, float] = {}
        # entity registry
        self._observations: Deque[Observation] = deque(maxlen=max_points)
        self._systems: set[str] = set()
        self._entities: Dict[str, set[str]] = defaultdict(set)
        self._pseudo: Dict[str, set[str]] = defaultdict(set)
        self._is_pseudo: Dict[str, bool] = {}
        self._first_seen: Dict[Tuple[str, str], float] = {}
        self._last_seen: Dict[Tuple[str, str], float] = {}
        # retention
        self._retention: Dict[str, Tuple[Optional[int], Optional[float]]] = dict(DEFAULT_RETENTION)
        self._dict_caps: Dict[str, Optional[int]] = dict(DICT_POINT_CAPS)
        self._ret_cache: Dict[Tuple[str, str], Tuple[int, Optional[float]]] = {}

    # ================================================================ retention
    def set_retention(self, prefix: str, max_points: Optional[int] = None,
                      max_age_s: Optional[float] = None) -> None:
        """Retention for every series (raw, derived, vector) whose name starts
        with `prefix` (longest prefix wins). Applied on the next append."""
        with self._lock:
            self._retention[prefix] = (max_points, max_age_s)
            self._ret_cache.clear()

    def ensure_retention(self, prefix: str, max_points: Optional[int] = None,
                         max_age_s: Optional[float] = None) -> bool:
        """Raise-only retention for `prefix`: keep AT LEAST max_points points
        and max_age_s seconds (None = no requirement), never lowering what the
        store defaults or another engine already set for it (integration R2.1:
        two engines that need the same input must not undo each other in
        registry order). The starting point is the longest-prefix explicit
        rule covering `prefix`; without one the request is taken as is.
        Idempotent and cheap (the rule cache is cleared only on a change).
        Returns True when the rule changed."""
        with self._lock:
            best = ""
            for p in self._retention:
                if prefix.startswith(p) and len(p) > len(best):
                    best = p
            cur = self._retention.get(best) if best else None
            if cur is None:
                new = (int(max_points) if max_points is not None else None,
                       float(max_age_s) if max_age_s is not None else None)
            else:
                mp, age = cur
                if max_points is not None and mp is not None and int(max_points) > mp:
                    mp = int(max_points)
                if max_age_s is not None and age is not None and float(max_age_s) > age:
                    age = float(max_age_s)
                new = (mp, age)
            if best == prefix and cur == new:
                return False
            self._retention[prefix] = new
            self._ret_cache.clear()
            return True

    def _rule(self, space: str, name: str, value: Any = None) -> Tuple[int, Optional[float]]:
        """(max_points, max_age_s) for a series; cached per (space, name)."""
        ck = (space, name)
        hit = self._ret_cache.get(ck)
        if hit is not None:
            return hit
        best = ""
        for p in self._retention:
            if name.startswith(p) and len(p) > len(best):
                best = p
        if best:
            mp, age = self._retention[best]
        else:
            mp, age = None, None
            if space == "raw":
                # raw scalars 6 h; categorical sets / structured values 1 h
                age = RAW_SCALAR_MAX_AGE if (value is None or _is_scalar(value)) else RAW_SET_MAX_AGE
        out_mp = int(mp) if mp else self._max
        if space == "derived":
            cap = self._dict_cap(name)
            if cap is not None:
                out_mp = min(out_mp, int(cap))
        out = (out_mp, age)
        self._ret_cache[ck] = out
        return out

    def _dict_cap(self, name: str) -> Optional[int]:
        """Point cap of a per-tick dict series (DICT_POINT_CAPS, longest
        prefix wins; an explicit None entry exempts the name)."""
        best, cap = "", None
        for p, c in self._dict_caps.items():
            if name.startswith(p) and len(p) > len(best):
                best, cap = p, c
        return cap

    def set_dict_cap(self, prefix: str, max_points: Optional[int]) -> None:
        """Point cap for dict series under `prefix` (None removes the cap for
        that prefix). Applied on the next append."""
        with self._lock:
            self._dict_caps[prefix] = None if max_points is None else int(max_points)
            self._ret_cache.clear()

    def _append_series(self, table: Dict[str, Deque], space: str, key: str, name: str,
                       m: Any) -> None:
        mp, age = self._rule(space, name, m.value)
        dq = table.get(key)
        if dq is None:
            dq = table[key] = deque(maxlen=mp)
        elif dq.maxlen != mp:
            dq = table[key] = deque(dq, maxlen=mp)
        dq.append(m)
        if space == "derived" and len(dq) >= 3:
            _compact(dq[-3], dq[-2])
        if age is not None:
            cutoff = m.ts - age
            while dq and dq[0].ts < cutoff:
                dq.popleft()

    # =================================================================== ingest
    def add_observation(self, obs: Observation) -> None:
        with self._lock:
            self._observations.append(obs)
            self._register(obs.system, obs.entity)

    def register_entity(self, system: str, entity: str) -> None:
        """Make an entity visible to `entities()` without writing data (tests,
        replay). Pseudo-entities go to the pseudo registry."""
        with self._lock:
            self._register(system, entity)

    def _register(self, system: str, entity: str) -> None:
        if is_pseudo_entity(entity) or is_pseudo_entity(system):
            self._pseudo[system].add(entity)
        else:
            self._systems.add(system)
            self._entities[system].add(entity)

    def _register_pseudo(self, system: str, entity: str) -> None:
        pseudo = self._is_pseudo.get(entity)        # memo: called on every write
        if pseudo is None:
            pseudo = self._is_pseudo[entity] = is_pseudo_entity(entity)
        if pseudo:
            self._pseudo[system].add(entity)

    def add_raw(self, m: RawMetric, touch: bool = True) -> None:
        """Append a raw metric. `touch` maintains first_seen/last_seen (raw
        engines stamp ts = ctx.now); zero-fill writes pass touch=False so
        filling a grid does not make an idle entity look active."""
        if is_pseudo_entity(m.entity):
            raise ValueError(f"add_raw: pseudo-entity {m.entity!r} cannot carry raw data")
        with self._lock:
            key = m.key
            self._append_series(self._raw, "raw", key, m.name, m)
            self._raw_names[(m.system, m.entity)].add(m.name)
            self._systems.add(m.system)
            self._entities[m.system].add(m.entity)
            self._bump_write(key, m.ts)
            if touch:
                se = (m.system, m.entity)
                fs = self._first_seen.get(se)
                if fs is None or m.ts < fs:
                    self._first_seen[se] = m.ts
                ls = self._last_seen.get(se)
                if ls is None or m.ts > ls:
                    self._last_seen[se] = m.ts

    def add_derived(self, m: DerivedMetric) -> None:
        with self._lock:
            key = m.key
            self._append_series(self._derived, "derived", key, m.name, m)
            self._derived_names[(m.system, m.entity)].add(m.name)
            self._register_pseudo(m.system, m.entity)
            self._bump_write(key, m.ts)

    def _bump_write(self, key: str, ts: float) -> None:
        lw = self._last_write.get(key)
        if lw is None or ts > lw:
            self._last_write[key] = ts

    # ============================================================ vector rings
    def add_vec(self, system: str, entity: str, name: str, ts: float,
                arr: Union[Sequence[float], np.ndarray], window_s: Optional[int] = None) -> None:
        """Append one float32 row to the (system, entity, name) ring. The dim
        is fixed per name by its first write. NaN is a legal value (it means
        'unscored / degraded' downstream)."""
        row = np.asarray(arr, dtype=np.float32).reshape(-1)
        with self._lock:
            d = self._vec_dim.get(name)
            if d is None:
                self._vec_dim[name] = d = row.shape[0]
            elif row.shape[0] != d:
                raise ValueError(f"add_vec {name}: dim {row.shape[0]} != registered {d}")
            key = _k(system, entity, name)
            mp, age = self._rule("vec", name)
            ring = self._vec.get(key)
            if ring is None:
                ring = self._vec[key] = _VecRing(d, mp)
                self._vec_names[(system, entity)].add(name)
            elif ring.max_cap != mp:
                ring.set_max_cap(mp)
            ring.append(float(ts), row)
            if window_s is not None:
                ring.window_s = int(window_s)
            if age is not None:
                ring.drop_before(float(ts) - age)
            self._register_pseudo(system, entity)
            self._bump_write(key, float(ts))

    def upsert_vec(self, system: str, entity: str, name: str, ts: float,
                   cols: Mapping[int, float], dim: int,
                   window_s: Optional[int] = None) -> None:
        """Set some columns of the row at `ts`, leaving the others untouched.

        Several engines own different slots of one shared vector in the same
        tick (e.g. each detector writes its own column of behavior.score). If
        the newest row already has this ts its other columns are kept;
        otherwise a fresh all-NaN row is started ('not scored this tick')."""
        with self._lock:
            ring = self._vec.get(_k(system, entity, name))
            if ring is not None and ring.n and ring.last_ts() == float(ts):
                row = ring.data[ring._phys(ring.n - 1)].copy()
            else:
                row = np.full(int(dim), np.nan, dtype=np.float32)
            for i, v in cols.items():
                row[int(i)] = np.nan if v is None else v
            self.add_vec(system, entity, name, ts, row, window_s=window_s)

    def upsert_dict(self, system: str, entity: str, name: str, ts: float,
                    items: Mapping[str, Any], window_s: int = 0) -> None:
        """Merge `items` into the dict-valued derived point at `ts` (creating
        it if the newest point is older). Same-tick multi-writer counterpart
        of upsert_vec for dict series such as behavior.axes / acc_alarm."""
        with self._lock:
            last = self.latest_derived(system, entity, name)
            if last is not None and last.ts == float(ts) and isinstance(last.value, dict):
                last.value.update(items)
                self._bump_write(_k(system, entity, name), float(ts))
                return
            self.add_derived(DerivedMetric(
                name=name, value=dict(items), ts=float(ts), system=system, entity=entity,
                window_s=int(window_s), kind=MetricKind.CATEGORICAL))

    def register_vector_names(self, vec_name: str, names: Sequence[str],
                              virtual_prefix: str) -> None:
        """Expose column i of `vec_name` as the scalar series
        `virtual_prefix + names[i]` (e.g. 'feature.' + 'bytes_up'). Views are
        read-only and computed on demand; nothing is stored twice."""
        with self._lock:
            self._vec_columns[vec_name] = list(names)
            for i, n in enumerate(names):
                self._virtual[f"{virtual_prefix}{n}"] = (vec_name, i)

    def vec_columns(self, vec_name: str) -> Optional[List[str]]:
        with self._lock:
            cols = self._vec_columns.get(vec_name)
            return list(cols) if cols is not None else None

    def vec_names(self, system: str, entity: str) -> List[str]:
        with self._lock:
            return sorted(self._vec_names.get((system, entity), ()))

    def vec_dim(self, name: str) -> Optional[int]:
        with self._lock:
            return self._vec_dim.get(name)

    def _empty_vec(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        return (np.empty(0, dtype=np.float64),
                np.empty((0, self._vec_dim.get(name, 0)), dtype=np.float32))

    def vec_tail(self, system: str, entity: str, name: str,
                 n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Last `n` rows as (ts[k], M[k, d]) copies, oldest first (k <= n)."""
        with self._lock:
            ring = self._vec.get(_k(system, entity, name))
            if ring is None or n <= 0:
                return self._empty_vec(name)
            return ring.take(ring.n - n, ring.n)

    def vec_since(self, system: str, entity: str, name: str,
                  since: float) -> Tuple[np.ndarray, np.ndarray]:
        """Rows with ts >= since, oldest first."""
        with self._lock:
            ring = self._vec.get(_k(system, entity, name))
            if ring is None:
                return self._empty_vec(name)
            return ring.take(ring.search(since, "left"), ring.n)

    def vec_range(self, system: str, entity: str, name: str, t0: float,
                  t1: float) -> Tuple[np.ndarray, np.ndarray]:
        """Rows with t0 <= ts <= t1, oldest first (replay windows)."""
        with self._lock:
            ring = self._vec.get(_k(system, entity, name))
            if ring is None:
                return self._empty_vec(name)
            return ring.take(ring.search(t0, "left"), ring.search(t1, "right"))

    def vec_at(self, system: str, entity: str, name: str, ts: float) -> Optional[np.ndarray]:
        """The row written at exactly `ts`, or None."""
        with self._lock:
            ring = self._vec.get(_k(system, entity, name))
            if ring is None or not ring.n:
                return None
            p = (ring.start + ring.n - 1) % ring.cap     # fast path: the newest row
            last = ring.ts[p]
            if ts == last:
                return ring.data[p].copy()
            if ts > last:
                return None
            i = ring.search(ts, "left")
            if i < ring.n and ring.ts_at(i) == ts:
                return ring.row(i).copy()
            return None

    def vec_latest(self, system: str, entity: str,
                   name: str) -> Optional[Tuple[float, np.ndarray]]:
        with self._lock:
            ring = self._vec.get(_k(system, entity, name))
            if ring is None or not ring.n:
                return None
            return ring.ts_at(ring.n - 1), ring.row(ring.n - 1).copy()

    def _virtual_ring(self, system: str, entity: str,
                      name: str) -> Optional[Tuple[_VecRing, int, str]]:
        v = self._virtual.get(name)
        if v is None:
            return None
        ring = self._vec.get(_k(system, entity, v[0]))
        if ring is None or not ring.n:
            return None
        return ring, v[1], v[0]

    def _vview(self, ring: _VecRing, col: int, vec_name: str, i: int, system: str,
               entity: str, name: str) -> DerivedMetric:
        return DerivedMetric(name=name, value=float(ring.row(i)[col]), ts=ring.ts_at(i),
                             system=system, entity=entity, window_s=ring.window_s,
                             kind=MetricKind.GAUGE, inputs=[vec_name])

    def _virtual_rows(self, system: str, entity: str, name: str,
                      n: Optional[int]) -> Optional[List[DerivedMetric]]:
        vr = self._virtual_ring(system, entity, name)
        if vr is None:
            return None
        ring, col, vname = vr
        i0 = 0 if n is None else max(0, ring.n - n)
        return [self._vview(ring, col, vname, i, system, entity, name) for i in range(i0, ring.n)]

    # =================================================================== events
    def add_event(self, e: BehaviorEvent) -> str:
        """Insert an event, assigning `e.id` if empty. Returns the id."""
        with self._lock:
            if not e.id:
                e.id = f"ev{next(self._ev_seq):08d}"
            elif e.id in self._ev_by_id:
                raise ValueError(f"duplicate event id {e.id!r}; use update_event")
            self._ev_by_id[e.id] = e
            self._ev_all.add(e.ts, e)
            self._ev_sys[e.system].add(e.ts, e)
            self._ev_ent[(e.system, e.entity)].add(e.ts, e)
            self._register_pseudo(e.system, e.entity)
            self._prune_indexed(self._ev_all, self._ev_sys, self._ev_ent, e.ts, EVENT_MAX_AGE,
                                self._ev_by_id)
            return e.id

    def get_event(self, event_id: str) -> Optional[BehaviorEvent]:
        with self._lock:
            return self._ev_by_id.get(event_id)

    def update_event(self, event_id: str, **kw: Any) -> BehaviorEvent:
        """Update mutable fields (status, incident_id, ...). ts / system /
        entity / id are index keys and cannot change."""
        frozen = {"id", "ts", "system", "entity"} & set(kw)
        if frozen:
            raise ValueError(f"update_event cannot change index fields {sorted(frozen)}")
        with self._lock:
            e = self._ev_by_id.get(event_id)
            if e is None:
                raise KeyError(event_id)
            for k, v in kw.items():
                if not hasattr(e, k):
                    raise AttributeError(f"BehaviorEvent has no field {k!r}")
                setattr(e, k, v)
            if "status" in kw:
                e.__post_init__()                  # re-validate the vocabulary
            return e

    def add_match(self, m: SignatureMatch) -> None:
        with self._lock:
            self._m_all.add(m.ts, m)
            self._m_sys[m.system].add(m.ts, m)
            self._m_ent[(m.system, m.entity)].add(m.ts, m)
            self._prune_indexed(self._m_all, self._m_sys, self._m_ent, m.ts, EVENT_MAX_AGE, None)

    def _prune_indexed(self, all_ix: _TsIndex, sys_ix: Dict[str, _TsIndex],
                       ent_ix: Dict[Tuple[str, str], _TsIndex], newest: float,
                       max_age: float, by_id: Optional[Dict[str, Any]]) -> None:
        cutoff = max(newest, all_ix.ts[-1]) - max_age
        while len(all_ix) and (len(all_ix) > self._max or all_ix.oldest()[0] < cutoff):
            _, item = all_ix.oldest()
            all_ix.remove(item)
            sys_ix[item.system].remove(item)
            ent_ix[(item.system, item.entity)].remove(item)
            if by_id is not None:
                by_id.pop(getattr(item, "id", ""), None)

    @staticmethod
    def _pick_index(all_ix: _TsIndex, sys_ix: Dict[str, _TsIndex],
                    ent_ix: Dict[Tuple[str, str], _TsIndex], system: Optional[str],
                    entity: Optional[str]) -> Tuple[Optional[_TsIndex], bool]:
        """Returns (index, needs_entity_filter)."""
        if system is not None and entity is not None:
            return ent_ix.get((system, entity)), False
        if system is not None:
            return sys_ix.get(system), False
        return all_ix, entity is not None

    def events(self, system: Optional[str] = None, entity: Optional[str] = None,
               since: Optional[float] = None, kinds: Optional[Iterable[str]] = None,
               limit: int = 200) -> List[BehaviorEvent]:
        """Newest-first events with ts >= since (O(log n + k))."""
        kinds_set = set(kinds) if kinds is not None else None
        with self._lock:
            ix, filt = self._pick_index(self._ev_all, self._ev_sys, self._ev_ent, system, entity)
            if ix is None:
                return []
            out: List[BehaviorEvent] = []
            for e in ix.newest_first(since):
                if len(out) >= limit:
                    break
                if filt and e.entity != entity:
                    continue
                if kinds_set is not None and e.kind not in kinds_set:
                    continue
                out.append(e)
            return out

    def matches(self, system: Optional[str] = None, entity: Optional[str] = None,
                since: Optional[float] = None, categories: Optional[Iterable[str]] = None,
                limit: int = 200) -> List[SignatureMatch]:
        """Newest-first signature matches with ts >= since (O(log n + k))."""
        cats = set(categories) if categories is not None else None
        with self._lock:
            ix, filt = self._pick_index(self._m_all, self._m_sys, self._m_ent, system, entity)
            if ix is None:
                return []
            out: List[SignatureMatch] = []
            for m in ix.newest_first(since):
                if len(out) >= limit:
                    break
                if filt and m.entity != entity:
                    continue
                if cats is not None and m.category not in cats:
                    continue
                out.append(m)
            return out

    # ======================================================= labels / incidents
    def add_label(self, label: Label) -> str:
        """Labels are never pruned (feedback and eval replay need them all)."""
        with self._lock:
            if not label.id:
                label.id = f"lb{next(self._label_seq):08d}"
            self._labels.append(label)
            return label.id

    def labels(self, system: Optional[str] = None, entity: Optional[str] = None,
               since: Optional[float] = None) -> List[Label]:
        """Newest-first labels matching the filters (ts >= since)."""
        with self._lock:
            return [lb for lb in reversed(self._labels)
                    if (system is None or lb.system == system)
                    and (entity is None or lb.entity == entity)
                    and (since is None or lb.ts >= since)]

    def put_incident(self, inc: Incident) -> str:
        """Insert or update (by id) an incident; assigns an id if empty."""
        with self._lock:
            if not inc.id:
                inc.id = f"inc{next(self._inc_seq):07d}"
            self._incidents[inc.id] = inc
            newest = max(inc.last_seen, inc.opened)
            cutoff = newest - INCIDENT_MAX_AGE
            stale = [k for k, v in self._incidents.items() if max(v.last_seen, v.opened) < cutoff]
            for k in stale:
                del self._incidents[k]
            return inc.id

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        with self._lock:
            return self._incidents.get(incident_id)

    def incidents(self, system: Optional[str] = None, entity: Optional[str] = None,
                  status: Optional[Union[str, Iterable[str]]] = None,
                  since: Optional[float] = None) -> List[Incident]:
        """Incidents newest-first by last_seen. `entity` matches the incident
        key or any member of `entities`; `since` filters on last_seen."""
        st = {status} if isinstance(status, str) else (set(status) if status is not None else None)
        with self._lock:
            out = [i for i in self._incidents.values()
                   if (system is None or i.system == system)
                   and (entity is None or i.entity == entity or entity in i.entities)
                   and (st is None or i.status in st)
                   and (since is None or i.last_seen >= since)]
        out.sort(key=lambda i: (i.last_seen, i.opened), reverse=True)
        return out

    # ================================================================ profiles
    def put_profile(self, p: EntityProfile) -> None:
        with self._lock:
            self._profiles[f"{p.system}|{p.entity}"] = p
            self._register_pseudo(p.system, p.entity)

    def put_profile_version(self, system: str, entity: str, version: Any, obj: Any,
                            ts: Optional[float] = None) -> None:
        """Keep the last 12 versions of a portrait / profile snapshot."""
        if ts is None:
            ts = getattr(obj, "updated", None)
            if ts is None and isinstance(obj, dict):
                ts = obj.get("ts", obj.get("updated"))
            ts = float(ts) if ts is not None else self._last_seen.get((system, entity), 0.0)
        with self._lock:
            dq = self._profile_versions.get((system, entity))
            if dq is None:
                dq = self._profile_versions[(system, entity)] = deque(maxlen=PROFILE_VERSIONS_KEPT)
            dq.append(ProfileVersion(version, float(ts), obj))

    def profile_versions(self, system: str, entity: str, n: int = 12) -> List[ProfileVersion]:
        """Newest-first (version, ts, obj) tuples."""
        with self._lock:
            dq = self._profile_versions.get((system, entity))
            return list(islice(reversed(dq), n)) if dq else []

    # ================================================================== models
    def put_model(self, system: str, entity: str, name: str, obj: Any,
                  version: Any = None, ts: Optional[float] = None) -> None:
        """Store a model object (by reference). Version: explicit, else the
        object's own `version` field, else previous + 1."""
        key = _k(system, entity, name)
        with self._lock:
            if version is None:
                version = obj.get("version") if isinstance(obj, dict) else getattr(obj, "version", None)
            if version is None:
                prev = self._models.get(key)
                pv = prev[1] if prev is not None else 0
                version = (pv + 1) if isinstance(pv, (int, np.integer)) else 1
            self._models[key] = (obj, version)
            self._model_names[(system, entity)].add(name)
            self._register_pseudo(system, entity)
            if ts is not None:
                self._bump_write(key, float(ts))

    def get_model(self, system: str, entity: str, name: str, default: Any = None) -> Any:
        with self._lock:
            hit = self._models.get(_k(system, entity, name))
            return hit[0] if hit is not None else default

    def model_version(self, system: str, entity: str, name: str) -> Any:
        with self._lock:
            hit = self._models.get(_k(system, entity, name))
            return hit[1] if hit is not None else None

    def model_names(self, system: str, entity: str) -> List[str]:
        with self._lock:
            return sorted(self._model_names.get((system, entity), ()))

    # ============================================================= checkpoints
    def put_checkpoint(self, system: str, entity: str, learner: str, ts: float,
                       blob: Any) -> None:
        """Keep learner checkpoints so every rollback target is replayable.

        Guarantee (lib/gating.py): for any rollback target T up to 168 h back
        there is a checkpoint in [T - 24 h, T], so the replay never needs
        journal rows older than 192 h (the 8-day rings). Pure geometric
        thinning cannot promise that — thinning only widens gaps, and any gap
        over 24 h eventually straddles the 168 h mark. So retention is:
          * floor:    the newest checkpoint older than the 168 h horizon;
          * backbone: for ages in (24 h, 168 h], the EARLIEST checkpoint of
                      each absolute 24 h day — any 24 h window contains a day
                      start, hence one of these;
          * recent:   ages <= 24 h, thinned greedily (merge the gap pair that
                      is smallest relative to its age) to keep fine
                      resolution where rollbacks are most frequent.
        At most 1 + 8 + CHECKPOINT_RECENT_MAX entries per key."""
        key = (system, entity, learner)
        ts = float(ts)
        with self._lock:
            lst = self._ckpt.setdefault(key, [])
            i = bisect.bisect_left(lst, ts, key=lambda x: x[0])
            if i < len(lst) and lst[i][0] == ts:
                lst[i] = (ts, blob)
            else:
                lst.insert(i, (ts, blob))
            newest = lst[-1][0]
            old = [c for c in lst if newest - c[0] > CHECKPOINT_HORIZON_S]
            inside = [c for c in lst if newest - c[0] <= CHECKPOINT_HORIZON_S]
            backbone: Dict[int, Tuple[float, Any]] = {}
            for c in inside:                         # sorted: first seen = earliest of its day
                backbone.setdefault(int(c[0] // DAY), c)
            bb_ts = {c[0] for c in backbone.values()}
            recent = [c for c in inside
                      if c[0] not in bb_ts and newest - c[0] <= DAY]
            while len(recent) > CHECKPOINT_RECENT_MAX:
                best_i, best_cost = 0, math.inf
                for j in range(0, len(recent) - 1):   # never drop the newest
                    lo = recent[j - 1][0] if j > 0 else newest - DAY
                    cost = (recent[j + 1][0] - lo) / (newest - recent[j + 1][0] + HOUR)
                    if cost < best_cost:
                        best_i, best_cost = j, cost
                del recent[best_i]
            kept = sorted(([old[-1]] if old else []) + list(backbone.values()) + recent,
                          key=lambda c: c[0])
            self._ckpt[key] = kept

    def get_checkpoint(self, system: str, entity: str, learner: str,
                       at_or_before: Optional[float] = None) -> Optional[Tuple[float, Any]]:
        """Latest checkpoint with ts <= at_or_before (newest if None)."""
        with self._lock:
            lst = self._ckpt.get((system, entity, learner))
            if not lst:
                return None
            if at_or_before is None:
                return lst[-1]
            i = bisect.bisect_right(lst, float(at_or_before), key=lambda x: x[0])
            return lst[i - 1] if i else None

    def checkpoint_times(self, system: str, entity: str, learner: str) -> List[float]:
        with self._lock:
            return [c[0] for c in self._ckpt.get((system, entity, learner), ())]

    # ================================================================== health
    def put_health(self, engine: str, record: Dict[str, Any]) -> None:
        with self._lock:
            self._health[engine] = dict(record)

    def health(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self._health.items()}

    def engine_failed(self, engine: str, ts: float) -> bool:
        """True if `engine` raised at tick `ts` (contract M: its consumers
        then write NaN + behavior.degraded instead of trusting stale output)."""
        with self._lock:
            h = self._health.get(engine)
            return bool(h) and h.get("last_error_ts") == ts

    def last_write_ts(self, system: str, entity: str, name: str) -> Optional[float]:
        """Newest ts written to a raw / derived / vector series (a virtual name
        resolves to its vector; models only if put with ts)."""
        with self._lock:
            ts = self._last_write.get(_k(system, entity, name))
            if ts is None and name in self._virtual:
                ts = self._last_write.get(_k(system, entity, self._virtual[name][0]))
            return ts

    # ================================================================= queries
    def systems(self) -> List[str]:
        with self._lock:
            return sorted(self._systems)

    def entities(self, system: str, include_pseudo: bool = False) -> List[str]:
        with self._lock:
            ents = set(self._entities.get(system, ()))
            if include_pseudo:
                ents |= self._pseudo.get(system, set())
            return sorted(ents)

    def pseudo_entities(self, system: str) -> List[str]:
        with self._lock:
            return sorted(self._pseudo.get(system, ()))

    def entities_active(self, system: str, since: float) -> List[str]:
        """Real entities whose last_seen >= since."""
        with self._lock:
            return sorted(e for e in self._entities.get(system, ())
                          if self._last_seen.get((system, e), -math.inf) >= since)

    def first_seen(self, system: str, entity: str) -> Optional[float]:
        with self._lock:
            return self._first_seen.get((system, entity))

    def last_seen(self, system: str, entity: str) -> Optional[float]:
        with self._lock:
            return self._last_seen.get((system, entity))

    def raw_series(self, system: str, entity: str, name: str) -> List[RawMetric]:
        """Full copy (v1 API). Prefer raw_tail in hot paths."""
        with self._lock:
            return list(self._raw.get(_k(system, entity, name), ()))

    def derived_series(self, system: str, entity: str, name: str) -> List[DerivedMetric]:
        """Full copy (v1 API); virtual names return DerivedMetric views."""
        with self._lock:
            dq = self._derived.get(_k(system, entity, name))
            if dq is not None:
                return list(dq)
            rows = self._virtual_rows(system, entity, name, None)
            return rows if rows is not None else []

    def raw_tail(self, system: str, entity: str, name: str, n: int) -> List[RawMetric]:
        """Last n raw points, oldest first, without copying the deque."""
        with self._lock:
            dq = self._raw.get(_k(system, entity, name))
            if not dq or n <= 0:
                return []
            out = list(islice(reversed(dq), n))
        out.reverse()
        return out

    def derived_tail(self, system: str, entity: str, name: str, n: int) -> List[DerivedMetric]:
        """Last n derived points (or virtual views), oldest first."""
        if n <= 0:
            return []
        with self._lock:
            dq = self._derived.get(_k(system, entity, name))
            if dq is None:
                rows = self._virtual_rows(system, entity, name, n)
                return rows if rows is not None else []
            out = list(islice(reversed(dq), n))
        out.reverse()
        return out

    def raw_names(self, system: str, entity: str) -> List[str]:
        with self._lock:
            return sorted(self._raw_names.get((system, entity), ()))

    def derived_names(self, system: str, entity: str) -> List[str]:
        """Stored derived names plus virtual column views whose vector exists."""
        with self._lock:
            names = set(self._derived_names.get((system, entity), ()))
            vecs = self._vec_names.get((system, entity))
            if vecs:
                names.update(v for v, (vec, _) in self._virtual.items() if vec in vecs)
            return sorted(names)

    def names_signature(self, system: str, entity: str) -> Tuple[int, int, int]:
        """O(1) change marker of derived_names / vec_names of an entity: the
        name sets only grow, so equal sizes mean equal lists (callers cache
        name scans on it)."""
        with self._lock:
            return (len(self._derived_names.get((system, entity), ())),
                    len(self._vec_names.get((system, entity), ())), len(self._virtual))

    def latest_raw(self, system: str, entity: str, name: str) -> Optional[RawMetric]:
        with self._lock:
            dq = self._raw.get(_k(system, entity, name))
            return dq[-1] if dq else None

    def latest_derived(self, system: str, entity: str, name: str) -> Optional[DerivedMetric]:
        with self._lock:
            dq = self._derived.get(_k(system, entity, name))
            if dq is not None:
                return dq[-1] if dq else None
            rows = self._virtual_rows(system, entity, name, 1)
            return rows[-1] if rows else None

    def latest_raw_at(self, system: str, entity: str, name: str,
                      ts: float) -> Optional[RawMetric]:
        """The raw point written at exactly ts (scans back from the newest)."""
        with self._lock:
            dq = self._raw.get(_k(system, entity, name))
            if not dq:
                return None
            for m in reversed(dq):
                if m.ts == ts:
                    return m
                if m.ts < ts:
                    break
            return None

    def _latest_any(self, system: str, entity: str, name: str) -> Optional[Any]:
        m = self.latest_raw(system, entity, name)
        if m is None:
            m = self.latest_derived(system, entity, name)
        return m

    def latest_fresh(self, system: str, entity: str, name: str, now: float) -> Any:
        """Latest raw-or-derived value only if it was written at `now`
        (i.e. this tick), else None. Staleness is data (contract D)."""
        with self._lock:
            m = self._latest_any(system, entity, name)
            if m is None or m.ts != now:
                return None
            return m.value

    def snapshot(self, system: str, entity: str, now: Optional[float] = None,
                 names: Optional[Iterable[str]] = None) -> Dict[str, float]:
        """Latest numeric value of every raw+derived metric for an entity.
        This flat name->value map is what the signature engines match against,
        so they need no knowledge of how the values were produced. With `now`,
        only values written at `now` are included (fresh-only). NaN views
        (unscored) are skipped. With `names`, only those metrics are looked
        up (same values as the full snapshot restricted to them): lib-3 adds
        ~300 derived names and vector views per entity that no signature reads
        (integration: the full scan cost lib-4 ~2.5 ms per entity per tick)."""
        snap: Dict[str, float] = {}
        if names is not None:
            with self._lock:
                for name in names:                     # derived wins, as below
                    m = self.latest_derived(system, entity, name)
                    if m is not None and isinstance(m.value, (int, float)) \
                            and (now is None or m.ts == now) and m.value == m.value:
                        snap[name] = float(m.value)
                        continue
                    r = self.latest_raw(system, entity, name)
                    if r is not None and isinstance(r.value, (int, float)) \
                            and (now is None or r.ts == now):
                        snap[name] = float(r.value)
            return snap
        with self._lock:
            for name in self.raw_names(system, entity):
                m = self.latest_raw(system, entity, name)
                if m is not None and isinstance(m.value, (int, float)) \
                        and (now is None or m.ts == now):
                    snap[name] = float(m.value)
            for name in self.derived_names(system, entity):
                m = self.latest_derived(system, entity, name)
                if m is not None and isinstance(m.value, (int, float)) \
                        and (now is None or m.ts == now) and m.value == m.value:
                    snap[name] = float(m.value)
        return snap

    def profile(self, system: str, entity: str) -> Optional[EntityProfile]:
        with self._lock:
            return self._profiles.get(f"{system}|{entity}")

    def all_profiles(self, system: Optional[str] = None) -> List[EntityProfile]:
        with self._lock:
            ps = list(self._profiles.values())
        return [p for p in ps if system is None or p.system == system]

    def recent_observations(self, limit: int = 500) -> List[Observation]:
        with self._lock:
            return list(islice(reversed(self._observations), limit))[::-1]

    # ================================================================ timeline
    def timeline(self, system: str, entity: str, since: Optional[float] = None,
                 limit: int = 200) -> List[Dict[str, Any]]:
        """Newest-first merge of events, matches, incidents, profile versions
        and risk points for one entity (or class). Risk points are included
        only where the risk tier changes (or a scalar risk moves by >= 0.1),
        so a per-tick series cannot flood the timeline."""
        items: List[Dict[str, Any]] = []
        with self._lock:
            for e in self.events(system, entity, since=since, limit=limit):
                items.append({"ts": e.ts, "type": "event", "item": e})
            for m in self.matches(system, entity, since=since, limit=limit):
                items.append({"ts": m.ts, "type": "match", "item": m})
            for inc in self.incidents(system, entity, since=since):
                items.append({"ts": inc.last_seen or inc.opened, "type": "incident", "item": inc})
            for pv in self.profile_versions(system, entity):
                if since is None or pv.ts >= since:
                    items.append({"ts": pv.ts, "type": "profile_version", "item": pv})
            # behavior.risk is a 1-element float32 vec ring (helpers_api 0.1,
            # B26); a derived deque (older writers / tests) is read as well
            ts_r, M_r = self.vec_since(system, entity, "behavior.risk",
                                       -math.inf if since is None else since)
            prev_band: Optional[float] = None
            for t, v in zip(ts_r.tolist(), M_r[:, 0].tolist() if len(ts_r) else ()):
                if v != v:
                    continue
                band = math.floor(v / RISK_TIMELINE_BAND)
                if band != prev_band:
                    items.append({"ts": t, "type": "risk", "item": DerivedMetric(
                        name="behavior.risk", value=float(v), ts=float(t), system=system,
                        entity=entity, window_s=0)})
                    prev_band = band
            dq = self._derived.get(_k(system, entity, "behavior.risk"))
            if dq:
                prev: Any = None
                for m in dq:
                    if since is not None and m.ts < since:
                        prev = self._risk_mark(m.value)
                        continue
                    mark = self._risk_mark(m.value)
                    changed = prev is None or (
                        mark != prev if not isinstance(mark, float) or not isinstance(prev, float)
                        else abs(mark - prev) >= 0.1)
                    if changed:
                        items.append({"ts": m.ts, "type": "risk", "item": m})
                        prev = mark
        items.sort(key=lambda d: d["ts"], reverse=True)
        return items[:limit]

    @staticmethod
    def _risk_mark(v: Any) -> Any:
        if isinstance(v, dict):
            return v.get("tier", v.get("score"))
        try:
            return float(v)
        except (TypeError, ValueError):
            return v

    # ================================================================== memory
    def memory_report(self) -> Dict[str, Any]:
        """Approximate memory by component (bytes). An object series costs its
        rows (row object + deque slot) plus its DISTINCT value objects, each
        sized like the mean of the newest 8 distinct values (shallow size plus
        up to 64 items). Values shared by several rows (a state dict that did
        not change, _compact; B24's calib_health object reused until its
        hourly re-evaluation) are counted once: counting them per row
        reported behavior.calib_health at 93 MB for 3 system keys on a 1-day
        60-s smoke run. O(#points) identity scan, no deep walk."""
        def val_bytes(v: Any) -> int:
            vb = v.nbytes if isinstance(v, np.ndarray) else sys.getsizeof(v)
            if isinstance(v, dict):
                vb += sum(sys.getsizeof(k) + sys.getsizeof(x) for k, x in islice(v.items(), 64))
            return vb

        def obj_bytes(dq: Deque) -> int:
            if not dq:
                return 0
            m = dq[-1]
            row_b = sys.getsizeof(m) + 8
            uniq: Dict[int, Any] = {}
            for x in dq:
                v = x.value
                if id(v) not in uniq:
                    uniq[id(v)] = v
            if len(uniq) == 1:
                return len(dq) * row_b + val_bytes(m.value)
            vals = list(uniq.values())
            samp = vals[-8:]
            vb = sum(val_bytes(v) for v in samp) / len(samp)
            return int(len(dq) * row_b + len(vals) * vb)

        with self._lock:
            raw_pts = sum(len(d) for d in self._raw.values())
            der_pts = sum(len(d) for d in self._derived.values())
            raw_b = sum(obj_bytes(d) for d in self._raw.values())
            der_b = sum(obj_bytes(d) for d in self._derived.values())
            vec_b = sum(r.nbytes for r in self._vec.values())
            vec_rows = sum(r.n for r in self._vec.values())
            ev_n, m_n = len(self._ev_all), len(self._m_all)
            ev_b = (ev_n + m_n) * 600
            ck_n = sum(len(v) for v in self._ckpt.values())
            ck_b = 0
            for v in self._ckpt.values():
                for _, blob in v:
                    ck_b += blob.nbytes if isinstance(blob, np.ndarray) else sys.getsizeof(blob)
            n_ent = sum(len(v) for v in self._entities.values())
            total = raw_b + der_b + vec_b + ev_b + ck_b
            return {
                "raw_series": len(self._raw), "raw_points": raw_pts, "raw_bytes": raw_b,
                "derived_series": len(self._derived), "derived_points": der_pts,
                "derived_bytes": der_b,
                "vec_series": len(self._vec), "vec_rows": vec_rows, "vec_bytes": vec_b,
                "events": ev_n, "matches": m_n, "event_match_bytes": ev_b,
                "incidents": len(self._incidents), "labels": len(self._labels),
                "models": len(self._models), "checkpoints": ck_n, "checkpoint_bytes": ck_b,
                "observations": len(self._observations),
                "entities": n_ent,
                "approx_bytes": total,
                "bytes_per_entity": total / n_ent if n_ent else float(total),
            }
