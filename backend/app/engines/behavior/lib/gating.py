"""Reversible, trust-gated learning (contract H, architecture section 3).

STATUS: implemented. Signatures, store calls and semantics are frozen
(docs/lib3/helpers_api.md); the bookkeeping that the stub left implicit
(checkpoint validity, see below) is documented here. DEVIATION: CommitRow is
an immutable NamedTuple instead of a slots dataclass (same fields and
constructor), so to_dict() / from_dict() share rows instead of rebuilding
the journal every tick (measured 44 + 770 us -> ~10 us at 765 rows).

Why: an attacker who is learned is invisible. Every learner therefore
  * learns LATE: at tick `now` it commits rows up to the commit frontier
    now - D, D = max(4 ticks, D_min_s = 600 s), so detection has D to react;
  * learns from TRUSTED rows: weight w_eff = behavior.trust(row ts);
  * HOLDS instead of learning while the entity is quarantined
    (behavior.quarantine at the latest tick < now == 1);
  * is REVERSIBLE: it checkpoints its state (at most hourly, geometric
    retention in the store) and keeps a journal of committed rows so the
    governor (B28) can rollback_to / release / rebase_from via model.control.

The learner supplies only pure functions; GatedLearner does the bookkeeping.
`update(state, row, w)` MUST be deterministic and depend only on (state, row,
w, row ts) so that restore-checkpoint + replay is bit-identical to the
original fit (B03 test e: equality within 1e-6). Because release commits
held rows *after* newer rows, `update` must accept rows older than the
state's newest ts (apply decay to the row weight, 2^(-(t_state - ts)/H),
rather than decaying the state backwards).

Governance scalars (behavior.trust, behavior.trust_prov,
behavior.quarantine) are 1-element float32 vec rings written by B28 with
store.add_vec; they are read here with store.vec_since / store.vec_at.

Store calls made (and only these):
  read   store.vec_since(s, e, clock, since)                 -> candidate row ts grid
  read   store.vec_since(s, e, 'behavior.trust', since)      -> w_eff per row ts
  read   store.vec_since(s, e, 'behavior.trust_prov', since) -> w_prov per row ts
         (vec_at per row instead when <= 8 rows are due: the per-tick case)
  read   store.vec_at(s, e, 'behavior.quarantine', now - dt) -> gate at t-1 (none = 0)
         (+ vec_since on the same ring when the t-1 row is missing or NaN:
          latest non-NaN value with ts < now)
  read   store.get_model(s, e, 'model.control')              -> governor directives
  read   store.get_model(s, '__system__', 'model.link')      -> link seeding
  read   store.get_checkpoint(s, e, learner, at_or_before)   -> (ts, blob) | None on rollback
  write  store.put_checkpoint(s, e, learner, ts, blob)       -> at most every ckpt_every_s
The learner's own model (which embeds GateState.to_dict()) is written by the
engine with store.put_model; GatedLearner never calls put_model itself.

Checkpoint validity (why the gate keeps barriers). A checkpoint at ts0 is
only useful if it equals "every journal row with ts <= ts0, and nothing
else". Checkpoints are written at ts0 = newest journal ts, but three things
later make an older checkpoint lie:
  * rollback_to tau: checkpoints in (tau, newest] contain rows now held;
  * release / rebase of held rows older than the newest commit: checkpoints
    in [oldest released ts, newest] lack them;
  * a link seed or a rebase hook changes the state without a row. It is
    given a position q = max(newest commit, t_link | rebase_from - eps) and
    a checkpoint contains it iff ts0 > q, so rollback_to = t_link undoes a
    seed (B17 retraction) while later rollbacks keep it.
The store has no delete, and a checkpoint carries no write time, so each of
these records a barrier (lo, hi] in gate.applied['_barriers']: a checkpoint
with lo < ts0 <= hi is never restored (the next older one is used, which only
costs replay) and never written. Rebase hooks are re-applied during replay at
their position; a seed lost by a rollback (not in the restored checkpoint) is
re-seeded by the next seed_from_link if q < tau, and stays undone otherwise.
Private bookkeeping lives under '_' keys of gate.applied, so the frozen
GateState layout is unchanged. ckpt_every_s = inf disables checkpoints
(rollback then replays the journal from init()).

Replay ordering: journal and held are kept sorted by ts and replay folds
rows in ascending ts, i.e. restore + replay equals an offline fit on the
rows <= tau with their recorded weights. A released row is journaled with
w_eff = w_prov, the weight it was actually committed with.
"""
from __future__ import annotations

import copy
import math
from bisect import bisect_left, bisect_right, insort
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import (TYPE_CHECKING, Any, Callable, Dict, Generic, Iterable, List, NamedTuple,
                    Optional, Sequence, Tuple, TypeVar)

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ....core.store import MetricStore

S = TypeVar("S")        # learner state (sufficient statistics)
Row = Any               # whatever the learner's fetch returns for one tick

D_MIN_S = 600.0                     # ctx.config['D_min_s'] default
D_MIN_TICKS = 4
CKPT_EVERY_S = 3600.0               # current-anchor style learners
CKPT_EVERY_REF_S = 86400.0          # reference-anchor style learners
CKPT_AGES_H: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 168)   # store retention
CKPT_MAX_PER_KEY = 10
JOURNAL_MAX_AGE_S = 8 * 86400.0     # committed-row journal kept for replay
HELD_MAX_AGE_S = 8 * 86400.0        # held rows older than this are dropped
ROLLBACK_MAX_DEPTH_S = 7 * 86400.0
ROLLBACK_MIN_INTERVAL_S = 3600.0
LINK_SEED_WEIGHT = 0.5              # B := B_own + 0.5 A
DEFAULT_CLOCK = "feature.active"    # series whose ts define candidate rows

TRUST = "behavior.trust"
TRUST_PROV = "behavior.trust_prov"
QUARANTINE = "behavior.quarantine"
CONTROL_MODEL = "model.control"
LINK_MODEL = "model.link"
SYSTEM_ENTITY = "__system__"

_TS_EPS = 1e-6                      # s; ts equality tolerance (ticks are >= 1 s apart)
_EVENT_KEEP_S = 2 * JOURNAL_MAX_AGE_S   # barriers / hooks outlive every restorable checkpoint
_MAX_BARRIERS = 64
_MAX_SEEDS = 64
_POINT_LOOKUPS = 8                  # <= this many rows: vec_at per row, else one vec_since
_NEG_INF = -math.inf
_CONTROL_KEYS = ("version", "branch", "rebase_from", "rollback_to", "release", "frozen",
                 "allow_drift", "accepted_class_change")


def commit_delay_ticks(dt_s: float, d_min_s: float = D_MIN_S) -> int:
    """D_ticks = max(4, ceil(d_min_s / dt_s))."""
    dt = float(dt_s)
    if not (math.isfinite(dt) and dt > 0.0):
        raise ValueError(f"commit_delay_ticks: dt_s must be finite and > 0, got {dt_s!r}")
    d_min = D_MIN_S if d_min_s is None else float(d_min_s)
    if not math.isfinite(d_min) or d_min <= 0.0:
        return D_MIN_TICKS
    x = d_min / dt
    # relative slack so 600 / 0.1 = 6000.000000000001 does not round up to 6001
    return max(D_MIN_TICKS, int(math.ceil(x - 1e-9 * max(1.0, x))))


def commit_frontier(now: float, dt_s: float, d_min_s: float = D_MIN_S) -> float:
    """Latest row ts that may be committed at `now`: now - D_ticks * dt_s."""
    return float(now) - commit_delay_ticks(dt_s, d_min_s) * float(dt_s)


class CommitRow(NamedTuple):
    """One committed or held row. An immutable NamedTuple (not a slots
    dataclass as in the stub; same fields and constructor): rows are shared
    between gate copies and embedded in to_dict() as is, which keeps the
    per-tick model round trip O(1) in Python instead of O(journal)."""
    ts: float
    w_eff: float        # trust(ts): weight used when committed normally
    w_prov: float       # trust_prov(ts): weight used on release / rebase


def _row_ts(r: CommitRow) -> float:
    return r.ts


def _enc_f(x: float) -> Optional[float]:
    """float -> JSON-safe float (non-finite -> None)."""
    x = float(x)
    return x if math.isfinite(x) else None


def _dec_f(x: Any, default: float = _NEG_INF) -> float:
    if x is None:
        return default
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if not math.isnan(v) else default


def _copy_plain(x: Any) -> Any:
    """Structural copy of the JSON-like `applied` dict (dicts and lists are
    copied, scalars shared). Everything the gate writes there is already
    JSON-safe (non-finite floats are stored as None), so no conversion."""
    if isinstance(x, dict):
        return {k: _copy_plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_copy_plain(v) for v in x]
    return x


def _rows_from(seq: Any) -> List[CommitRow]:
    """Rows from to_dict() (CommitRow tuples: O(1) Python) or from JSON
    ([ts, w_eff, w_prov] lists / {'ts', ...} dicts): converted, non-finite ts
    dropped, sorted by ts."""
    if not seq:
        return []
    if type(seq) is list and type(seq[0]) is CommitRow and type(seq[-1]) is CommitRow:
        # to_dict output: already sorted, and the list is to_dict's own copy.
        # Shared, not copied: GatedLearner copies a gate's lists before any
        # mutation, so the gate and the dict never write to it.
        return seq
    seq = list(seq)
    if type(seq[0]) is CommitRow and type(seq[-1]) is CommitRow:
        return seq
    out: List[CommitRow] = []
    for r in seq:
        if isinstance(r, Mapping):
            out.append(CommitRow(_dec_f(r.get("ts")), _dec_f(r.get("w_eff"), 0.0),
                                 _dec_f(r.get("w_prov"), 0.0)))
        else:
            ts, we, wp = (list(r) + [0.0, 0.0])[:3]
            out.append(CommitRow(_dec_f(ts), _dec_f(we, 0.0), _dec_f(wp, 0.0)))
    out = [r for r in out if math.isfinite(r.ts)]
    out.sort(key=_row_ts)
    return out


@dataclass(slots=True)
class GateState:
    """Bookkeeping persisted inside the learner's model (to_dict / from_dict).

    last_ts          ts of the newest row already processed (committed or held)
    version, branch  mirror of model.control version / branch last applied
    held             rows waiting for the governor, ascending ts
    journal          committed rows (ts, w_eff, w_prov) within JOURNAL_MAX_AGE_S
    last_ckpt_ts     frontier ts of the last checkpoint written
    applied          last applied value of each control directive
                     {'rollback_to', 'release', 'rebase_from', 'frozen', 'version'}
                     so every directive is applied exactly once
    frozen           commits stopped (REJECTED)
    allow_drift      accepted legitimate ramp slope for reference anchors
    link_version     model.link version already seeded from
    last_rollback_ts wall ts of the last rollback (1 per hour limit)

    Private keys of `applied` (see module doc, checkpoint validity):
      '_barriers'  [[lo, hi], ...] checkpoints with lo < ts <= hi are stale
      '_hooks'     [[rebase_from, q], ...] rebase hooks and their position
      '_seeds'     [{'from', 'ts', 'q', 'status'}] link seeds; status is
                   applied | pending (re-seed) | undone | empty (nothing to merge)
      '_last_rollback' diagnostics of the last rollback (eval checks each one;
                   complete = False when the restored checkpoint predates the
                   pruned journal, i.e. rows in between could not be replayed)
      '_jfloor'    newest ts pruned from the journal
    """
    last_ts: float = -math.inf
    version: int = 0
    branch: int = 0
    held: List[CommitRow] = field(default_factory=list)
    journal: List[CommitRow] = field(default_factory=list)
    last_ckpt_ts: float = -math.inf
    applied: Dict[str, Any] = field(default_factory=dict)
    frozen: bool = False
    allow_drift: float = 0.0
    link_version: int = 0
    last_rollback_ts: float = -math.inf

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable (json.dumps(allow_nan=False) works; -inf is None,
        rows serialise as [ts, w_eff, w_prov] arrays). Rows are the immutable
        CommitRow tuples themselves, so this is cheap enough to call per tick."""
        return {
            "last_ts": _enc_f(self.last_ts),
            "version": int(self.version),
            "branch": int(self.branch),
            "held": list(self.held),
            "journal": list(self.journal),
            "last_ckpt_ts": _enc_f(self.last_ckpt_ts),
            "applied": _copy_plain(self.applied),
            "frozen": bool(self.frozen),
            "allow_drift": float(self.allow_drift) if math.isfinite(self.allow_drift) else 0.0,
            "link_version": int(self.link_version),
            "last_rollback_ts": _enc_f(self.last_rollback_ts),
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "GateState":
        """None / {} -> a fresh GateState. A GateState is returned as is
        (GatedLearner never mutates a gate it was given; it returns a new one)."""
        if isinstance(d, GateState):
            return d
        if not d:
            return cls()
        drift = _dec_f(d.get("allow_drift"), 0.0)
        return cls(
            last_ts=_dec_f(d.get("last_ts")),
            version=_as_int(d.get("version")) or 0,
            branch=_as_int(d.get("branch")) or 0,
            held=_rows_from(d.get("held")),
            journal=_rows_from(d.get("journal")),
            last_ckpt_ts=_dec_f(d.get("last_ckpt_ts")),
            applied=_copy_plain(dict(d.get("applied") or {})),
            frozen=bool(d.get("frozen", False)),
            allow_drift=drift if math.isfinite(drift) else 0.0,
            link_version=_as_int(d.get("link_version")) or 0,
            last_rollback_ts=_dec_f(d.get("last_rollback_ts")),
        )

    # ---------------------------------------------------------------- private
    def _copy(self, deep: bool = False, now: Optional[float] = None) -> "GateState":
        """Working copy: new lists (CommitRows are immutable and shared);
        `applied` copied structurally only if asked. With `now`, rows older
        than the journal / held horizons are dropped by the same copy (the
        per-tick prune costs no second O(journal) copy; see _prune)."""
        applied = _copy_plain(self.applied) if deep else dict(self.applied)
        journal, held = self.journal, self.held
        if now is not None and journal and journal[0].ts < now - JOURNAL_MAX_AGE_S:
            k = bisect_left(journal, now - JOURNAL_MAX_AGE_S, key=_row_ts)
            applied["_jfloor"] = journal[k - 1].ts        # replay cannot reach below this
            journal = journal[k:]
        else:
            journal = list(journal)
        if now is not None and held and held[0].ts < now - HELD_MAX_AGE_S:
            held = held[bisect_left(held, now - HELD_MAX_AGE_S, key=_row_ts):]
        else:
            held = list(held)
        return GateState(self.last_ts, self.version, self.branch, held, journal,
                         self.last_ckpt_ts, applied, self.frozen, self.allow_drift,
                         self.link_version, self.last_rollback_ts)

    def _committed_ts(self) -> float:
        """Newest ts folded into the state (-inf when the journal is empty)."""
        return self.journal[-1].ts if self.journal else _NEG_INF


# ============================================================ candidate rows
def _col0(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M)
    if M.ndim == 2:
        return M[:, 0].astype(np.float64) if M.shape[1] else np.full(M.shape[0], np.nan)
    return M.reshape(-1).astype(np.float64)


def _clip01(v: float, default: float) -> float:
    return default if not math.isfinite(v) else (0.0 if v < 0.0 else 1.0 if v > 1.0 else v)


def _lookup(store: "MetricStore", s: str, e: str, name: str, ts: List[float],
            default: float) -> List[float]:
    """Value of the 1-element ring `name` at each ts, `default` where missing
    or NaN, clipped to [0, 1]. The steady state is one row per tick, where a
    vec_at per row beats vec_since + numpy by ~10x; a backlog (first tick,
    catch-up) is matched in one vec_since within _TS_EPS."""
    if len(ts) <= _POINT_LOOKUPS:
        out = []
        for t in ts:
            row = store.vec_at(s, e, name, t)
            out.append(_clip01(float(row[0]), default) if row is not None and len(row)
                       else default)
        return out
    tq = np.asarray(ts, dtype=np.float64)
    t, M = store.vec_since(s, e, name, float(tq[0]) - _TS_EPS)
    t = np.asarray(t, dtype=np.float64)
    out_a = np.full(tq.shape, float(default), dtype=np.float64)
    if t.size:
        v = _col0(M)
        idx = np.searchsorted(t, tq - _TS_EPS, side="left")
        ok = idx < t.size
        ok[ok] = t[idx[ok]] <= tq[ok] + _TS_EPS
        vals = v[idx[ok]]
        out_a[ok] = np.where(np.isfinite(vals), vals, float(default))
    return np.clip(out_a, 0.0, 1.0).tolist()


def commit_candidates(store: "MetricStore", s: str, e: str, learner: str, last_ts: float,
                      now: float, dt_s: float, *, d_min_s: float = D_MIN_S,
                      clock: str = DEFAULT_CLOCK, training: bool = False
                      ) -> List[Tuple[float, float, float]]:
    """Rows eligible at `now`: [(ts, w_eff, w_prov)] ascending, for every clock
    ts in (last_ts, commit_frontier(now)].

    w_eff = behavior.trust at ts, w_prov = behavior.trust_prov at ts, both
    clipped to [0, 1]. Missing trust value: 1.0 when `training`, else 0.0
    (fail safe; the row is still returned so it can be held and released
    later). `learner` is only used for diagnostics. Whether a row is
    committed or held is decided by the caller (see is_quarantined).
    """
    frontier = commit_frontier(now, dt_s, d_min_s)
    lo = _dec_f(last_ts)
    if lo >= frontier:
        return []
    t, _ = store.vec_since(s, e, clock, lo)
    if not len(t):
        return []
    hi = frontier + _TS_EPS
    ts = [x for x in np.asarray(t, dtype=np.float64).tolist() if lo < x <= hi]
    if not ts:
        return []
    default = 1.0 if training else 0.0
    w_eff = _lookup(store, s, e, TRUST, ts, default)
    w_prov = _lookup(store, s, e, TRUST_PROV, ts, default)
    return list(zip(ts, w_eff, w_prov))


def is_quarantined(store: "MetricStore", s: str, e: str, now: float, dt_s: float) -> bool:
    """quarantine(t - 1): latest non-NaN behavior.quarantine with ts < now (> 0.5);
    none within HELD_MAX_AGE_S -> False.

    NaN (a degraded governor tick) counts as missing, never as 0: reading it
    as "not quarantined" would let an attacker who degrades B28 for one tick
    get that tick's row committed in the middle of a quarantine episode."""
    now = float(now)
    row = store.vec_at(s, e, QUARANTINE, now - float(dt_s))
    if row is not None and np.asarray(row).size:
        v = float(np.asarray(row, dtype=np.float64).reshape(-1)[0])
        if not math.isnan(v):
            return bool(v > 0.5)
    # the governor skipped t-1 or wrote NaN (degraded tick, cadence change):
    # the latest defined value before now still holds; a stale quarantine keeps holding
    t, M = store.vec_since(s, e, QUARANTINE, now - HELD_MAX_AGE_S)
    t = np.asarray(t, dtype=np.float64)
    k = int(np.searchsorted(t, now - _TS_EPS, side="left"))            # rows with ts < now
    if k == 0:
        return False
    v = _col0(M)[:k]
    ok = np.flatnonzero(~np.isnan(v))
    return bool(ok.size and v[ok[-1]] > 0.5)


# ================================================================ barriers
def _barriers(g: GateState) -> List[List[float]]:
    return g.applied.get("_barriers") or []


def _barrier_hit(g: GateState, ts0: float) -> Optional[float]:
    """Lowest `lo` of the barriers containing ts0 (lo < ts0 <= hi), else None."""
    hit: Optional[float] = None
    for lo, hi in _barriers(g):
        lo, hi = _dec_f(lo), _dec_f(hi)
        if lo < ts0 <= hi and (hit is None or lo < hit):
            hit = lo
    return hit


def _add_barrier(g: GateState, lo: float, hi: float) -> None:
    """Checkpoints with lo < ts <= hi are stale. Kept sorted; the list is
    bounded by merging the two oldest (a wider barrier is only conservative)."""
    if not hi > lo:
        return
    bs = [[_dec_f(a), _dec_f(b)] for a, b in _barriers(g)]
    bs.append([lo, hi])
    bs.sort()
    while len(bs) > _MAX_BARRIERS:
        a, b = bs.pop(0), bs.pop(0)
        bs.insert(0, [min(a[0], b[0]), max(a[1], b[1])])
    g.applied["_barriers"] = [[_enc_f(a), _enc_f(b)] for a, b in bs]


def _prev(x: float) -> float:
    return math.nextafter(x, _NEG_INF)


def _event_lo(g: GateState, pc: float) -> float:
    """Lower bound of the barrier for a state change without a row (seed,
    rebase hook) made when the newest commit is pc. Checkpoints at pc are
    ambiguous (written before or after the event), older ones predate it.
    With an empty journal (every commit pruned, e.g. > 8 d inactive) every
    checkpoint <= _jfloor predates the event too (later commits have ts >
    _jfloor), so it stays restorable instead of (-inf, q] wiping the model."""
    if math.isfinite(pc):
        return _prev(pc)
    return _dec_f(g.applied.get("_jfloor"))


# ============================================================== the learner
@dataclass
class GatedLearner(Generic[S]):
    """Generic trust-gated, checkpointed, reversible learner.

    name        checkpoint key, e.g. 'baseline.current', 'rhythm', 'vocab'
    init        () -> S, the empty state (hyperpriors only)
    update      (state, row, w) -> state; pure and deterministic (see module doc)
    fetch       (store, s, e, ts) -> row | None; None skips the tick (counts
                as processed, nothing committed or held)
    dump/load   state <-> checkpoint blob (float16 arrays allowed: rollback
                equality is then to float16 precision, so B03 stores float64
                blobs for the current anchor)
    merge       (own, other, w) -> own + w * other in sufficient-stat space;
                required for link seeding, None disables seeding
    on_rebase   (state, rebase_from) -> state hook called before the rebased
                rows are committed into version + 1 (e.g. B03 lifts its rate cap
                and schedules the reference reset)

    Blobs are deep-copied on put and before load, so a learner whose update
    mutates arrays in place cannot corrupt a stored checkpoint.
    """
    name: str
    init: Callable[[], S]
    update: Callable[[S, Row, float], S]
    fetch: Callable[["MetricStore", str, str, float], Optional[Row]]
    dump: Callable[[S], Any]
    load: Callable[[Any], S]
    merge: Optional[Callable[[S, S, float], S]] = None
    on_rebase: Optional[Callable[[S, float], S]] = None
    d_min_s: float = D_MIN_S
    ckpt_every_s: float = CKPT_EVERY_S
    clock: str = DEFAULT_CLOCK

    # ------------------------------------------------------------------ step
    def step(self, store: "MetricStore", s: str, e: str, state: S, gate: GateState,
             now: float, dt_s: float, training: bool = False) -> Tuple[S, GateState]:
        """One tick of learning. Order:

        1. control = store.get_model(s, e, 'model.control'); apply_control(...)
        2. rows = commit_candidates(store, s, e, name, gate.last_ts, now, dt_s)
        3. q = is_quarantined(store, s, e, now, dt_s)
           for each row (ascending ts):
             frozen          -> dropped
             q               -> gate.held.append(row)
             else            -> r = fetch(...); if r is not None:
                                  state = update(state, r, w_eff); journal.append(row)
           gate.last_ts = max processed ts
        4. if frontier - gate.last_ckpt_ts >= ckpt_every_s:
             store.put_checkpoint(s, e, name, ts=last committed ts, blob=dump(state))
        5. prune journal / held older than now - 8 d.
        Returns the new (state, gate); the caller persists both with put_model.
        The input gate is not mutated by the commit loop (an update error
        leaves the caller's (state, gate) pair consistent).
        """
        control = store.get_model(s, e, CONTROL_MODEL)
        state, gate = self.apply_control(store, s, e, state, gate, control, now, dt_s)
        g = gate._copy(now=float(now))
        rows = commit_candidates(store, s, e, self.name, g.last_ts, now, dt_s,
                                 d_min_s=self.d_min_s, clock=self.clock, training=training)
        if rows:
            q = False if g.frozen else is_quarantined(store, s, e, now, dt_s)
            for ts, w_eff, w_prov in rows:
                if g.frozen:
                    pass                                        # REJECTED: dropped
                elif q:
                    g.held.append(CommitRow(ts, w_eff, w_prov))
                else:
                    r = self.fetch(store, s, e, ts)
                    if r is not None:
                        state = self.update(state, r, w_eff)
                        g.journal.append(CommitRow(ts, w_eff, w_prov))
                g.last_ts = max(g.last_ts, ts)
        frontier = commit_frontier(now, dt_s, self.d_min_s)
        if frontier - g.last_ckpt_ts >= self.ckpt_every_s:
            self._checkpoint(store, s, e, state, g, frontier)
        _prune(g, float(now))
        return state, g

    def _checkpoint(self, store: "MetricStore", s: str, e: str, state: S, g: GateState,
                    frontier: float) -> bool:
        """Write dump(state) at the newest committed ts unless that position is
        behind a barrier (it would never be restored). Returns True if written;
        otherwise last_ckpt_ts is left alone so the next tick retries."""
        ts0 = g._committed_ts()
        if (not math.isfinite(ts0) or not math.isfinite(self.ckpt_every_s)
                or _barrier_hit(g, ts0) is not None):
            return False                     # ckpt_every_s = inf: never checkpoint
        store.put_checkpoint(s, e, self.name, ts0, copy.deepcopy(self.dump(state)))
        g.last_ckpt_ts = float(frontier)
        return True

    # --------------------------------------------------------------- control
    def apply_control(self, store: "MetricStore", s: str, e: str, state: S, gate: GateState,
                      control: Optional[Dict[str, Any]], now: float, dt_s: float
                      ) -> Tuple[S, GateState]:
        """Apply each changed model.control directive once, in this order:

        rollback_to (tau):  if tau differs from gate.applied['rollback_to'],
            now - tau <= 7 d and now - gate.last_rollback_ts >= 1 h:
            (ts0, blob) = store.get_checkpoint(s, e, name, at_or_before=tau)
            state = load(blob) (or init() if no checkpoint); replay journal rows
            with ts0 < ts <= tau via fetch/update using their RECORDED w_eff;
            journal rows with ts > tau move to held (keeping w_prov).
        release [t0, t1]: commit held rows with t0 <= ts <= t1 with w = w_prov,
            ascending ts; they join the journal.
        rebase_from (tau): commit held rows with ts < tau with w_prov; bump
            gate.version to control['version'] (or +1); state = on_rebase(state,
            tau); commit held rows with ts >= tau with w_prov.
        frozen: True -> gate.frozen = True and gate.held cleared (REJECTED);
            False -> resume.
        allow_drift: gate.allow_drift = float(control['allow_drift'] or 0).
        A version change without rebase_from only records the version.
        On any replay error the learner sets gate.frozen = True and re-raises
        (the engine's safe_run turns it into pipeline_degraded).

        Details: a rollback deeper than 7 d is recorded as applied and skipped
        (the governor never asks for one); one within the hour of the last is
        deferred to a later tick. frozen = False (resume) is applied first so an
        un-freeze and a release in one control both take effect. A restored
        checkpoint is the newest at or before tau that no barrier marks stale.
        """
        d = control_directives(control)
        if control is None:
            return state, gate
        ap = gate.applied
        now = float(now)
        def todo(key: str) -> bool:                 # changed since last applied
            return d[key] is not None and not _same(d[key], ap.get(key))
        todo_rb, todo_rel, todo_reb = todo("rollback_to"), todo("release"), todo("rebase_from")
        todo_frz = d["frozen"] is not None and d["frozen"] != ap.get("frozen")
        drift = d["allow_drift"] if d["allow_drift"] is not None else 0.0
        if not (todo_rb or todo_rel or todo_reb or todo_frz):
            # fast path (every tick): only the recorded scalars can change
            if (drift == gate.allow_drift and (d["version"] is None or d["version"] == gate.version)
                    and (d["branch"] is None or d["branch"] == gate.branch)):
                return state, gate
            g = gate._copy()
            g.allow_drift = drift
            self._record_version(g, d)
            return state, g

        g = gate._copy(deep=True)
        try:
            if todo_frz and d["frozen"] is False:
                g.frozen = False
                g.applied["frozen"] = False
            if todo_rb:
                tau = float(d["rollback_to"])
                if now - tau > ROLLBACK_MAX_DEPTH_S:
                    g.applied["rollback_to"] = tau                  # never applicable
                    g.applied["_last_rollback"] = {"tau": tau, "wall": now, "skipped": "depth"}
                elif now - g.last_rollback_ts >= ROLLBACK_MIN_INTERVAL_S:
                    state = self._rollback(store, s, e, state, g, tau, now, dt_s)
                    g.applied["rollback_to"] = tau
                    g.last_rollback_ts = now
                # else: rate limited, retried on a later tick
            if todo_rel:
                t0, t1 = d["release"]
                sel = [r for r in g.held if t0 - _TS_EPS <= r.ts <= t1 + _TS_EPS]
                g.held = [r for r in g.held if not (t0 - _TS_EPS <= r.ts <= t1 + _TS_EPS)]
                state = self._commit_held(store, s, e, state, g, sel)
                g.applied["release"] = [t0, t1]
            if todo_reb:
                state = self._rebase(store, s, e, state, g, float(d["rebase_from"]), d["version"])
                g.applied["rebase_from"] = float(d["rebase_from"])
            if todo_frz and d["frozen"] is True:
                g.frozen = True
                g.held = []
                g.applied["frozen"] = True
            g.allow_drift = drift
            if not todo_reb:
                self._record_version(g, d)
            elif d["branch"] is not None:
                g.branch = d["branch"]
        except Exception:
            gate.frozen = True
            raise
        return state, g

    @staticmethod
    def _record_version(g: GateState, d: Dict[str, Any]) -> None:
        if d["version"] is not None:
            g.version = d["version"]
            g.applied["version"] = d["version"]
        if d["branch"] is not None:
            g.branch = d["branch"]

    def _commit_held(self, store: "MetricStore", s: str, e: str, state: S, g: GateState,
                     rows: Sequence[CommitRow]) -> S:
        """Commit (already removed) held rows with w_prov, ascending ts; they
        join the journal in ts order with w_eff = w_prov. Rows older than the
        newest commit make older checkpoints stale -> barrier."""
        if not rows or g.frozen:
            return state
        pc = g._committed_ts()
        lo_ts = math.inf
        for r in sorted(rows, key=_row_ts):
            x = self.fetch(store, s, e, r.ts)
            if x is None:
                continue
            state = self.update(state, x, r.w_prov)
            insort(g.journal, CommitRow(r.ts, r.w_prov, r.w_prov), key=_row_ts)
            lo_ts = min(lo_ts, r.ts)
        if lo_ts <= pc:
            _add_barrier(g, _prev(lo_ts), pc)
        return state

    def _rebase(self, store: "MetricStore", s: str, e: str, state: S, g: GateState,
                tau: float, version: Optional[int]) -> S:
        held, g.held = g.held, []
        pre = [r for r in held if r.ts < tau]
        post = [r for r in held if r.ts >= tau]
        state = self._commit_held(store, s, e, state, g, pre)
        g.version = int(version) if version is not None else g.version + 1
        g.applied["version"] = g.version
        if self.on_rebase is not None:
            pc = g._committed_ts()
            q = max(pc, _prev(tau))
            state = self.on_rebase(state, tau)
            hooks = list(g.applied.get("_hooks") or [])
            hooks.append([tau, q])
            g.applied["_hooks"] = hooks
            _add_barrier(g, _event_lo(g, pc), q)
        return self._commit_held(store, s, e, state, g, post)

    def _restore(self, store: "MetricStore", s: str, e: str, g: GateState,
                 tau: float) -> Optional[Tuple[float, Any]]:
        """Newest checkpoint at or before tau that no barrier marks stale."""
        at = tau
        for _ in range(4 * _MAX_BARRIERS + 4):
            ck = store.get_checkpoint(s, e, self.name, at_or_before=at)
            if ck is None:
                return None
            lo = _barrier_hit(g, float(ck[0]))
            if lo is None:
                return ck
            if not math.isfinite(lo) or lo >= at:
                return None
            at = lo
        return None

    def _rollback(self, store: "MetricStore", s: str, e: str, state: S, g: GateState,
                  tau: float, now: float, dt_s: float) -> S:
        j = g.journal
        k = bisect_right(j, tau, key=_row_ts)
        keep, moved = j[:k], j[k:]
        hooks = [[_dec_f(a), _dec_f(b)] for a, b in (g.applied.get("_hooks") or [])]
        seeds = [dict(sd) for sd in (g.applied.get("_seeds") or [])]
        live_seeds = [sd for sd in seeds if sd.get("status") == "applied"]
        if (not moved and not any(q >= tau for _, q in hooks)
                and not any(_dec_f(sd.get("q")) >= tau for sd in live_seeds)):
            g.applied["_last_rollback"] = {"tau": tau, "wall": now, "noop": True}
            return state                                     # nothing after tau
        pc = g._committed_ts()
        ck = self._restore(store, s, e, g, tau)
        if ck is not None:
            ts0 = float(ck[0])
            st = self.load(copy.deepcopy(ck[1]))
        else:
            ts0 = _NEG_INF
            st = self.init()
        # replay rows (ts0, tau] and the rebase hooks the checkpoint lacks
        # (ts0 <= q < tau), in position order
        rows = keep[bisect_right(keep, ts0, key=_row_ts):]
        replay_hooks = sorted((q, t_r) for t_r, q in hooks if ts0 <= q < tau)
        hi = 0
        n_rows = 0
        for r in rows:
            while hi < len(replay_hooks) and replay_hooks[hi][0] < r.ts:
                st = self.on_rebase(st, replay_hooks[hi][1]) if self.on_rebase else st
                hi += 1
            x = self.fetch(store, s, e, r.ts)
            if x is not None:
                st = self.update(st, x, r.w_eff)
                n_rows += 1
        for q, t_r in replay_hooks[hi:]:
            st = self.on_rebase(st, t_r) if self.on_rebase else st
        # seeds the restored checkpoint lacks: re-seed later if they belong
        # before tau, otherwise they are undone (retraction: tau = t_link)
        for sd in seeds:
            q = _dec_f(sd.get("q"))
            if sd.get("status") == "applied" and q >= ts0:
                sd["status"] = "pending" if q < tau else "undone"
            elif sd.get("status") == "pending" and q >= tau:
                sd["status"] = "undone"
        g.applied["_seeds"] = seeds
        g.applied["_hooks"] = [[t_r, q] for t_r, q in hooks if q < tau]
        g.journal = keep
        if not g.frozen:
            g.held = sorted(g.held + moved, key=_row_ts)
        if pc > tau:
            _add_barrier(g, tau, pc)
        jfloor = _dec_f(g.applied.get("_jfloor"))
        g.applied["_last_rollback"] = {
            "tau": tau, "wall": now, "ckpt_ts": _enc_f(ts0), "replayed": n_rows,
            "moved": len(moved),
            "journal_from": _enc_f(keep[0].ts) if keep else None,
            # rows in (ckpt_ts, jfloor] were pruned: the replay is incomplete
            # (gap None = no checkpoint at all, everything <= jfloor lost)
            "complete": bool(ts0 >= jfloor),
            "history_gap_s": _enc_f(jfloor - ts0) if ts0 < jfloor else 0.0,
        }
        self._checkpoint(store, s, e, st, g, commit_frontier(now, dt_s, self.d_min_s))
        return st

    # ------------------------------------------------------------ link seeding
    def seed_from_link(self, store: "MetricStore", s: str, e: str, state: S, gate: GateState,
                       load_other: Callable[[str], Optional[S]]) -> Tuple[S, GateState]:
        """Link seeding: when model.link@(s, '__system__').version > gate.link_version
        and a link {'from': A, 'to': e, 'ts': t_link, ...} targets this entity,
        state := merge(state, load_other(A), LINK_SEED_WEIGHT) once per link,
        then gate.link_version = link version. A retracted link is undone by the
        governor through rollback_to = t_link, not here.

        Links flagged retracted ('retracted': True, 'status'/'state' ==
        'retracted' or 'active': False) are never seeded. A seed that a
        rollback dropped although it belongs before tau is re-seeded here.
        """
        if self.merge is None:
            return state, gate
        link = store.get_model(s, SYSTEM_ENTITY, LINK_MODEL)
        if not isinstance(link, Mapping):
            return state, gate
        ver = _as_int(link.get("version")) or 0
        seeds = gate.applied.get("_seeds") or []
        pending = any(sd.get("status") == "pending" for sd in seeds)
        if ver <= gate.link_version and not pending:
            return state, gate
        g = gate._copy(deep=True)
        seeds = [dict(sd) for sd in (g.applied.get("_seeds") or [])]
        index = {(sd.get("from"), _seed_ts_key(sd.get("ts"))): sd for sd in seeds}
        live = set()
        for lk in _iter_links(link):
            if lk.get("to") != e or not lk.get("from") or _retracted(lk):
                continue
            src = str(lk["from"])
            t_link = _dec_f(lk.get("ts"))
            key = (src, _seed_ts_key(t_link))
            live.add(key)
            sd = index.get(key)
            if sd is not None and sd.get("status") != "pending":
                continue                                     # once per link
            other = load_other(src)
            if sd is None:
                sd = {"from": src, "ts": _enc_f(t_link)}
                seeds.append(sd)
                index[key] = sd
            if other is None:
                sd.update(q=None, status="empty")
                continue
            pc = g._committed_ts()
            q = max(pc, t_link)
            state = self.merge(state, other, LINK_SEED_WEIGHT)
            sd.update(q=_enc_f(q), status="applied")
            _add_barrier(g, _event_lo(g, pc), q)
            g.last_ckpt_ts = _NEG_INF        # checkpoint the seeded state at the first valid ts
        for key, sd in index.items():                        # link gone or retracted
            if sd.get("status") == "pending" and key not in live:
                sd["status"] = "undone"
        if len(seeds) > _MAX_SEEDS:
            seeds = sorted(seeds, key=lambda sd: _dec_f(sd.get("ts")))[-_MAX_SEEDS:]
        g.applied["_seeds"] = seeds
        g.link_version = max(ver, g.link_version)
        return state, g


# ================================================================== helpers
def _prune(g: GateState, now: float) -> None:
    """Drop journal / held rows older than 8 d and events no restorable
    checkpoint can predate. Replaces lists (never mutates shared ones)."""
    if g.journal and g.journal[0].ts < now - JOURNAL_MAX_AGE_S:
        k = bisect_left(g.journal, now - JOURNAL_MAX_AGE_S, key=_row_ts)
        g.applied["_jfloor"] = g.journal[k - 1].ts        # replay cannot reach below this
        g.journal = g.journal[k:]
    if g.held and g.held[0].ts < now - HELD_MAX_AGE_S:
        g.held = g.held[bisect_left(g.held, now - HELD_MAX_AGE_S, key=_row_ts):]
    cut = now - _EVENT_KEEP_S
    bs = g.applied.get("_barriers")
    if bs and any(_dec_f(hi) < cut for _, hi in bs):
        g.applied["_barriers"] = [b for b in bs if _dec_f(b[1]) >= cut]
    hk = g.applied.get("_hooks")
    if hk and any(_dec_f(q) < cut for _, q in hk):
        g.applied["_hooks"] = [h for h in hk if _dec_f(h[1]) >= cut]


def _iter_links(link: Mapping) -> Iterable[Mapping]:
    links = link.get("links")
    if isinstance(links, Mapping):
        links = links.values()
    for lk in links or ():
        if isinstance(lk, Mapping):
            yield lk


def _retracted(lk: Mapping) -> bool:
    return bool(lk.get("retracted")) or lk.get("status") == "retracted" \
        or lk.get("state") == "retracted" or lk.get("active") is False


def _seed_ts_key(ts: Any) -> Optional[float]:
    v = _dec_f(ts, math.nan)
    return round(v, 3) if math.isfinite(v) else None


def _same(a: Any, b: Any) -> bool:
    """Directive equality (floats within _TS_EPS; release as a pair)."""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        try:
            return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
        except TypeError:
            return False
    try:
        return abs(float(a) - float(b)) <= _TS_EPS
    except (TypeError, ValueError):
        return a == b


def _as_int(v: Any) -> Optional[int]:
    if v is None or isinstance(v, bool):
        return None if v is None else int(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if math.isfinite(f) else None


def _as_ts(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def control_directives(control: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalise a model.control dict to {'version', 'branch', 'rebase_from',
    'rollback_to', 'release', 'frozen', 'allow_drift', 'accepted_class_change'}
    with None for absent keys and release as a (t0, t1) tuple or None."""
    out: Dict[str, Any] = dict.fromkeys(_CONTROL_KEYS)
    if not control:
        return out
    if isinstance(control, Mapping):
        get = control.get
    else:
        def get(k: str, default: Any = None) -> Any:
            return getattr(control, k, default)
    out["version"] = _as_int(get("version"))
    out["branch"] = _as_int(get("branch"))
    out["rebase_from"] = _as_ts(get("rebase_from"))
    out["rollback_to"] = _as_ts(get("rollback_to"))
    rel = get("release")
    if isinstance(rel, Mapping):
        rel = (rel.get("t0"), rel.get("t1"))
    if isinstance(rel, (list, tuple, np.ndarray)) and len(rel) == 2:
        t0, t1 = _as_ts(rel[0]), _as_ts(rel[1])
        if t0 is not None and t1 is not None:
            out["release"] = (min(t0, t1), max(t0, t1))
    fz = get("frozen")
    out["frozen"] = None if fz is None else bool(fz)
    ad = get("allow_drift")
    if ad is not None and not isinstance(ad, bool):
        try:
            f = float(ad)
            out["allow_drift"] = f if math.isfinite(f) else None
        except (TypeError, ValueError):
            out["allow_drift"] = None
    out["accepted_class_change"] = get("accepted_class_change")
    return out


def reference_eligible(ts: float, w_eff: float, incident_or_regime_ts: Sequence[float],
                       window_s: float = 86400.0) -> bool:
    """Reference-anchor admission (B03 step 4): w_eff == 1 and no incident / non-NORMAL
    regime tick within +-window_s of ts."""
    try:
        w = float(w_eff)
        t = float(ts)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(w) and w >= 1.0 - 1e-9 and math.isfinite(t)):
        return False
    if incident_or_regime_ts is None:
        return True
    src = incident_or_regime_ts
    arr = np.asarray(src if isinstance(src, np.ndarray) else list(src),
                     dtype=np.float64).reshape(-1)
    if not arr.size:
        return True
    arr = arr[np.isfinite(arr)]
    return not bool(np.any(np.abs(arr - t) <= float(window_s)))
