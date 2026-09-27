"""Client-stack engine (R3) — passive client / device fingerprint tokens.

Why: an IP says where traffic comes from, not what is sending it. The TLS
ClientHello (JA3 / JA4), the User-Agent and the TCP/IP stack (initial TTL,
window size) describe the client software and the OS, and they move together
only when the device really changes. B09 (client identity / impersonation),
B16 (attribution), B17 (linking, shared-IP detection) and the B01 feature
sketch need them per tick as tokens with timestamps. The token itself is
lib/stack.stack_token, the same pure function R2 uses to stamp act.stream's
stack_id, so the two ids always agree.

Per tick, per real entity with fingerprinted observations (ts = ctx.now, raw
layer, add_raw with touch=True; no zero-fill: absence of a stack is data):

  client.stack_set   {token: {'n': int, 'bytes': float, 'first_ts': float,
                     'last_ts': float}}  top STACK_TOP tokens by n (ties by
                     token) plus '__other__' (the rest summed, first/last as
                     min/max) only when there are more. n = sum of observation
                     weights (extra.count), bytes = up + down (extra
                     bytes_*_total when present, else w * obs bytes), first/last
                     = event times (see below). lib/sketch reads this shape.
  client.stack_events [[stack_id, first_ts, last_ts, n], ...]  plain lists
                     (JSON-safe), sorted by (first_ts, stack_id). One row per
                     EPISODE of a stack: its events sorted by time and split at
                     gaps > EPISODE_GAP_S (5 min). Only stacks listed in
                     stack_set (not '__other__'), so every stack_id joins to a
                     token via stack_id(token). n sums to the stack's n.
  client.ua_set      {'family/major': n}   UA component (e.g. 'chrome/126')
  client.ja3n_set    {ja3n: n}             TLS component: ja3n md5, or 'ja4:<ja4>'
  client.ttl_set     {'64': n}             TTL class component (32/64/128/255)
  client.os_ua_ttl_pairs {'<declared_os>|<ua_family>|<ttl_class>': n}
                     e.g. 'win|chrome|64'; declared_os is what the UA claims
                     ('?' when it claims none), NOT the token's TTL-implied
                     fallback, so B09 can test it with
                     stack.os_ttl_consistent(declared_os, ttl_class). Only for
                     observations that carry both a UA and a TTL.
  The four component sets are top STACK_TOP plus '__other__' like every raw
  set; each holds only KNOWN values (an observation without a UA adds nothing
  to ua_set), so its total is the weight of observations that carried that
  component.

Why episodes split at 5 min: B09's concurrency test is "the two stacks'
intervals overlap or are within 5 minutes, whatever the tick length". With
episodes cut at gaps > 300 s, two stacks pass that test iff some event of one
and some event of the other are <= 300 s apart (an overlap puts an event of
one inside the other's episode, whose internal gaps are <= 300 s; a gap
between episode ends is a gap between two events). Tick boundaries only add
cuts, which keep that property, so the answer is the same at 60, 900 or
3600 s. One [first, last] envelope per tick would not be: at 3600 s, A at
:00 and :59 with B at :30 overlaps, while at 60 s the same events are three
disjoint points. An interleaving at gaps <= 5 min (the NAT / impersonation
signature) always yields overlapping episodes. EPISODES_CAP bounds the rows
per stack (only reachable with ticks > 2.7 h): beyond it the episodes across
the smallest gaps are merged, which can only add overlap.

Event times: obs.ts (non-finite -> ctx.now). An aggregated observation
(extra.count = w) with extra['ts_sample'] contributes its first
min(len, TS_SAMPLE_MAX, w) sample times, parsed exactly as R2 does
(lib/m_template: an entry with |v| < TS_ABSOLUTE_MIN is an offset from
obs.ts, else an epoch; non-finite / unparsable entries are dropped, none
valid -> one event at obs.ts); its w is split over them as integers
(w // k each, the first w % k one more), so n stays integral and exact.
Times are not clipped to the tick: a late record keeps its real time.

Skipped observations (counted in this engine's health record): active
probes (our own traffic: the target's reply TTL is not a client stack),
pseudo / empty entities (add_raw would reject them), weight <= 0, and
observations with no fingerprint field at all (the token
'-|none/0|?|0|w0' says nothing about the client and would only dilute the
shares B09 weights surprise by). Partial tokens are kept: a DNS query with
only a TTL still fingerprints the OS.

Stateless: no model, no learner, nothing to gate or roll back, so cadence
changes (900 -> 60 s) and training mode need no special handling (a raw
engine emits no events in any mode).
"""
from __future__ import annotations

import math
from itertools import chain, islice
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import (AcquisitionMethod, MetricKind, Observation, RawMetric,
                              is_pseudo_entity)
from ..behavior.lib import m_template as MT
from ..behavior.lib.stack import parse_stack_token, stack_id, stack_token, ua_parse

STACK_TOP = 64                # entries per set before '__other__'
OTHER = "__other__"
EPISODE_GAP_S = 300.0         # a stack's events split into episodes at gaps > 5 min
EPISODES_CAP = 32             # episodes per stack per tick; the closest ones merge beyond

_ACTIVE_METHODS = frozenset({AcquisitionMethod.ACTIVE_PROBE, AcquisitionMethod.ACTIVE_DNS,
                             AcquisitionMethod.ACTIVE_TLS})
_PASSIVE_SPAN = AcquisitionMethod.PASSIVE_SPAN
_CAT = MetricKind.CATEGORICAL
_INF = math.inf
_ABS = MT.TS_ABSOLUTE_MIN
_BLANK = stack_token(None, None, None, None)          # '-|none/0|?|0|w0': no evidence

# Fingerprint memo: (ja3, ua, ttl, win, ja4) -> _Fp | None. Bounded, cleared when full;
# inputs repeat heavily (a few dozen stacks per system), a cold build costs ~5-10 us.
_MEMO_MAX = 1 << 14
_FP: Dict[Any, Optional["_Fp"]] = {}
_MISS = object()


class _Fp:
    """Everything R3 derives from one (ja3, ua, ttl, win, ja4) combination."""
    __slots__ = ("token", "sid", "ja3n", "ua", "ttl", "pair")

    def __init__(self, token: str, sid: int, ja3n: Optional[str], ua: Optional[str],
                 ttl: Optional[str], pair: Optional[str]) -> None:
        self.token, self.sid, self.ja3n, self.ua, self.ttl, self.pair = \
            token, sid, ja3n, ua, ttl, pair


def _fingerprint(ja3: Any, ua: Any, ttl: Any, win: Any, ja4: Any) -> Optional[_Fp]:
    """The stack token and its known components; None when nothing is known."""
    token = stack_token(ja3, ua, ttl, win, ja4)
    if token == _BLANK:
        return None
    p = parse_stack_token(token)
    fam, major, ttl_c = p["ua_family"], p["ua_major"], p["ttl_class"]
    has_ua, has_ttl = fam != "none", ttl_c != "0"
    pair = None
    if has_ua and has_ttl:
        declared = ua_parse(ua)[2]
        pair = f"{declared or '?'}|{fam}|{ttl_c}"
    return _Fp(token, stack_id(token),
               p["ja3n"] if p["ja3n"] != "-" else None,
               f"{fam}/{major}" if has_ua else None,
               ttl_c if has_ttl else None,
               pair)


def _fp_of(o: Observation, ex: Optional[Dict[str, Any]]) -> Optional[_Fp]:
    ja4 = ex.get("ja4") if ex else None
    key = (o.ja3, o.user_agent, o.ttl, o.win_size, ja4)
    try:
        fp = _FP.get(key, _MISS)
    except TypeError:                          # unhashable field: same fallback as R2
        return _fingerprint(o.ja3, o.user_agent, o.ttl, o.win_size,
                            str(ja4) if ja4 is not None else None)
    if fp is _MISS:
        fp = _fingerprint(o.ja3, o.user_agent, o.ttl, o.win_size, ja4)
        if len(_FP) >= _MEMO_MAX:
            _FP.clear()
        _FP[key] = fp
    return fp


# ------------------------------------------------------------------ helpers
def _weight(ex: Dict[str, Any]) -> int:
    """Events a record stands for: int(extra.count), default 1. A non-numeric or
    non-finite count is one event; <= 0 is none (the R1 / R2 rule)."""
    c = ex.get("count", 1)
    if c.__class__ is int:
        return c if c > 0 else 0
    try:
        v = float(c)
    except (TypeError, ValueError, OverflowError):
        return 1
    if v != v or v == _INF:
        return 1
    return int(v) if v >= 1.0 else 0


def _bytes(x: Any) -> float:
    """A byte count; NaN, inf, negative or unparsable -> 0."""
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return v if 0.0 <= v < _INF else 0.0


def _ts(t: Any, now: float) -> float:
    """A finite float event time, else ctx.now."""
    try:
        v = float(t)
    except (TypeError, ValueError, OverflowError):
        return now
    return v if -_INF < v < _INF else now


def _sample_head(ex: Dict[str, Any], w: int) -> Optional[Any]:
    """The first min(TS_SAMPLE_MAX, w) entries of extra['ts_sample'] (unparsed),
    or None when there is none (as R2)."""
    sample = ex.get("ts_sample")
    if sample is None or isinstance(sample, (str, bytes, dict)):
        return None
    try:
        k = min(len(sample), MT.TS_SAMPLE_MAX, w)
    except TypeError:
        return None
    if k <= 0:
        return None
    if isinstance(sample, (list, tuple, np.ndarray)):
        return sample[:k]
    return list(islice(sample, k))




def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError, OverflowError):
        return math.nan


class _Stack:
    """One stack of one entity this tick (its events live in the tick's flat lists)."""
    __slots__ = ("fp", "n", "bytes")

    def __init__(self, fp: _Fp) -> None:
        self.fp = fp
        self.n = 0
        self.bytes = 0.0


class _Events:
    """Flat event lists of one tick, over all entities and stacks. Keeping them
    flat lets the episode split run as one vectorised pass per tick instead of
    ~20 numpy calls per stack (hundreds of stacks per tick)."""
    __slots__ = ("pt", "pi", "wt", "ww", "wi", "ch", "ct", "cw", "ci")

    def __init__(self) -> None:
        self.pt: List[float] = []            # plain events: time (weight 1)
        self.pi: List[int] = []              #   stack index
        self.wt: List[float] = []            # aggregated, no ts_sample: time
        self.ww: List[int] = []              #   weight
        self.wi: List[int] = []              #   stack index
        self.ch: List[Any] = []              # aggregated with ts_sample: sample head
        self.ct: List[float] = []            #   obs time (offset origin)
        self.cw: List[int] = []              #   weight
        self.ci: List[int] = []              #   stack index


# ------------------------------------------------------------------ episodes
def _flatten(ev: _Events) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(times float64, weights int64, stack index int64) of every event.

    Sampled records follow R2's rule: |v| < TS_ABSOLUTE_MIN is an offset from the
    obs time, else an epoch; non-finite / unparsable entries are dropped, and a
    record with no valid entry is one event of weight w at its obs time. The
    weight w is split over the k valid entries as integers: w // k each, the
    first w % k (in sample order) one more, so the sum stays exactly w."""
    t_parts = [np.array(ev.pt, dtype=np.float64), np.array(ev.wt, dtype=np.float64)]
    w_parts = [np.ones(len(ev.pt), dtype=np.int64), np.array(ev.ww, dtype=np.int64)]
    i_parts = [np.array(ev.pi, dtype=np.int64), np.array(ev.wi, dtype=np.int64)]
    ch = ev.ch
    if ch:
        lens = np.array([len(h) for h in ch], dtype=np.int64)       # each >= 1
        n = int(lens.sum())
        try:
            flat = np.fromiter(chain.from_iterable(ch), dtype=np.float64, count=n)
        except (TypeError, ValueError, OverflowError):
            flat = np.fromiter((_f(x) for h in ch for x in h), dtype=np.float64, count=n)
        rec = np.repeat(np.arange(len(ch)), lens)
        t0 = np.array(ev.ct, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            ts = np.where(np.abs(flat) < _ABS, flat + t0[rec], flat)
        ok = np.isfinite(ts)
        k = np.bincount(rec[ok], minlength=len(ch))                 # valid entries per record
        w = np.array(ev.cw, dtype=np.int64)
        base, rem = np.divmod(w, np.maximum(k, 1))
        okc = np.cumsum(ok) - ok                                     # valid entries before i
        start = np.cumsum(lens) - lens
        rank = okc - okc[start][rec]                                 # ... inside its record
        wv = base[rec] + (rank < rem[rec])
        ci = np.array(ev.ci, dtype=np.int64)
        t_parts.append(ts[ok])
        w_parts.append(wv[ok])
        i_parts.append(ci[rec[ok]])
        none = k == 0
        if none.any():
            t_parts.append(t0[none])
            w_parts.append(w[none])
            i_parts.append(ci[none])
    return np.concatenate(t_parts), np.concatenate(w_parts), np.concatenate(i_parts)


def _episodes(ev: _Events, n_stacks: int) -> Tuple[List[float], List[float], List[int],
                                                    List[int]]:
    """Episodes of every stack: events sorted by (stack, time) and cut where the
    stack changes or the gap exceeds EPISODE_GAP_S. Returns (firsts, lasts, ns,
    bounds) as lists; stack i's episodes are [bounds[i], bounds[i + 1]), in time
    order, and every stack has at least one."""
    t, w, idx = _flatten(ev)
    order = np.lexsort((t, idx))
    t, w, idx = t[order], w[order], idx[order]
    cut = np.flatnonzero((np.diff(idx) != 0) | (np.diff(t) > EPISODE_GAP_S)) + 1
    starts = np.concatenate(([0], cut))
    ends = np.concatenate((cut, [len(t)]))
    bounds = np.searchsorted(idx[starts], np.arange(n_stacks + 1))
    return (t[starts].tolist(), t[ends - 1].tolist(), np.add.reduceat(w, starts).tolist(),
            bounds.tolist())


def _merge_episodes(firsts: Any, lasts: Any, ns: Any,
                    cap: int = EPISODES_CAP) -> Tuple[List[float], List[float], List[int]]:
    """At most `cap` episodes (time-ordered input): keep the cap - 1 widest gaps as
    cuts (ties: the earlier gap) and merge across the others."""
    f = np.asarray(firsts, dtype=np.float64)
    la = np.asarray(lasts, dtype=np.float64)
    n = np.asarray(ns, dtype=np.int64)
    if len(f) > cap:
        gaps = f[1:] - la[:-1]
        keep = np.sort(np.argsort(-gaps, kind="stable")[:cap - 1]) + 1
        starts = np.concatenate(([0], keep))
        ends = np.concatenate((keep, [len(f)]))
        f, la, n = f[starts], la[ends - 1], np.add.reduceat(n, starts)
    return f.tolist(), la.tolist(), n.tolist()


# ------------------------------------------------------------------ sets
def _top(d: Dict[str, int], k: int = STACK_TOP) -> Dict[str, int]:
    """Top-k values plus '__other__' (the remainder); ties rank by key."""
    if len(d) <= k:
        return dict(d)
    items = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))
    out = dict(items[:k])
    rest = sum(n for _, n in items[k:])
    if rest > 0:
        out[OTHER] = rest
    return out


def _add(d: Dict[str, int], key: Optional[str], w: int) -> None:
    if key is not None:
        d[key] = d.get(key, 0) + w


class ClientStackEngine(Engine):
    name = "raw.client_stack"
    layer = "raw"
    consumes = ["<observations>"]
    produces = [
        "client.stack_set", "client.stack_events", "client.ua_set", "client.ja3n_set",
        "client.ttl_set", "client.os_ua_ttl_pairs",
    ]
    description = ("Passive client/device fingerprint tokens (ja3n|ua/major|os|ttl|win) with "
                   "timestamped per-stack episodes and the UA / JA3n / TTL / OS-consistency "
                   "component sets.")
    interval = 1

    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        self.last_dropped_pseudo = 0
        self.last_dropped_active = 0
        self.last_unfingerprinted = 0       # weight of observations with no fingerprint field

    def health_record(self, ok: bool = True) -> Dict[str, Any]:
        rec = super().health_record(ok)
        rec["dropped_pseudo"] = self.last_dropped_pseudo
        rec["dropped_active"] = self.last_dropped_active
        rec["unfingerprinted"] = self.last_unfingerprinted
        return rec

    # ------------------------------------------------------------------ run
    def run(self, ctx: Context, observations: Optional[List[Observation]] = None) -> int:
        now = float(ctx.now)
        accs: Dict[Tuple[str, str], Dict[str, int]] = {}      # entity -> {token: stack index}
        stacks: List[_Stack] = []
        ev = _Events()
        pt_append, pi_append = ev.pt.append, ev.pi.append
        bad = set()
        drop_pseudo = drop_active = blank = 0
        for o in observations or ():
            m = o.method
            if m is not _PASSIVE_SPAN and m in _ACTIVE_METHODS:
                drop_active += 1                   # our own probes are not a client
                continue
            k = (o.system, o.entity)
            acc = accs.get(k)
            if acc is None:
                if k in bad or not k[0] or not k[1] or is_pseudo_entity(k[1]) \
                        or is_pseudo_entity(k[0]):
                    bad.add(k)
                    drop_pseudo += 1
                    continue
                acc = accs[k] = {}
            ex = o.extra
            if ex:                                 # aggregated / annotated record
                w = _weight(ex)
                if w <= 0:
                    continue
                fp = _fp_of(o, ex)
                if fp is None:
                    blank += w
                    continue
                b = ((_bytes(ex["bytes_up_total"]) if "bytes_up_total" in ex
                      else w * _bytes(o.bytes_up))
                     + (_bytes(ex["bytes_down_total"]) if "bytes_down_total" in ex
                        else w * _bytes(o.bytes_down)))
                t = _ts(o.ts, now)
                head = _sample_head(ex, w)
            else:                                  # one plain event (hot path)
                fp = _fp_of(o, None)
                if fp is None:
                    blank += 1
                    continue
                w, head = 1, None
                up, down = o.bytes_up, o.bytes_down
                if up.__class__ is not int or up < 0:
                    up = _bytes(up)
                if down.__class__ is not int or down < 0:
                    down = _bytes(down)
                b = up + down
                t = o.ts
                if t.__class__ is not float or not -_INF < t < _INF:
                    t = _ts(t, now)
            i = acc.get(fp.token)
            if i is None:
                i = acc[fp.token] = len(stacks)
                stacks.append(_Stack(fp))
            sa = stacks[i]
            sa.n += w
            sa.bytes += b
            if head is not None:
                ev.ch.append(head)
                ev.ct.append(t)
                ev.cw.append(w)
                ev.ci.append(i)
            elif w == 1:
                pt_append(t)
                pi_append(i)
            else:
                ev.wt.append(t)
                ev.ww.append(w)
                ev.wi.append(i)
        self.last_dropped_pseudo, self.last_dropped_active = drop_pseudo, drop_active
        self.last_unfingerprinted = blank
        if not stacks:
            return 0

        eps = _episodes(ev, len(stacks))
        n = 0
        for (s, e), acc in accs.items():
            if acc:
                n += self._emit(ctx.store, s, e, [(i, stacks[i]) for i in acc.values()],
                                eps, now)
        return n

    # ----------------------------------------------------------------- emit
    @staticmethod
    def _emit(store, s: str, e: str, items: List[Tuple[int, _Stack]],
              eps: Tuple[List[float], List[float], List[int], List[int]], now: float) -> int:
        firsts, lasts, ns, bounds = eps
        items.sort(key=lambda it: (-it[1].n, it[1].fp.token))
        kept, rest = items[:STACK_TOP], items[STACK_TOP:]
        stack_set: Dict[str, Dict[str, Any]] = {}
        rows: List[List[Any]] = []
        for i, sa in kept:
            a, z = bounds[i], bounds[i + 1]
            stack_set[sa.fp.token] = {"n": int(sa.n), "bytes": float(sa.bytes),
                                      "first_ts": firsts[a], "last_ts": lasts[z - 1]}
            sid = sa.fp.sid
            if z - a == 1:
                rows.append([sid, firsts[a], lasts[a], ns[a]])
                continue
            f, la, c = firsts[a:z], lasts[a:z], ns[a:z]
            if z - a > EPISODES_CAP:
                f, la, c = _merge_episodes(f, la, c)
            rows.extend([sid, x, y, m] for x, y, m in zip(f, la, c))
        if rest:
            stack_set[OTHER] = {"n": int(sum(sa.n for _, sa in rest)),
                                "bytes": float(sum(sa.bytes for _, sa in rest)),
                                "first_ts": min(firsts[bounds[i]] for i, _ in rest),
                                "last_ts": max(lasts[bounds[i + 1] - 1] for i, _ in rest)}
        rows.sort(key=lambda r: (r[1], r[0]))

        ua: Dict[str, int] = {}
        ja3n: Dict[str, int] = {}
        ttl: Dict[str, int] = {}
        pairs: Dict[str, int] = {}
        for _, sa in items:
            fp, w = sa.fp, sa.n
            _add(ua, fp.ua, w)
            _add(ja3n, fp.ja3n, w)
            _add(ttl, fp.ttl, w)
            _add(pairs, fp.pair, w)

        out: List[Tuple[str, Any]] = [("client.stack_set", stack_set),
                                      ("client.stack_events", rows)]
        for name, d in (("client.ua_set", ua), ("client.ja3n_set", ja3n),
                        ("client.ttl_set", ttl), ("client.os_ua_ttl_pairs", pairs)):
            if d:
                out.append((name, _top(d)))
        add = store.add_raw
        for name, value in out:
            add(RawMetric(name=name, value=value, ts=now, system=s, entity=e, kind=_CAT,
                          method=_PASSIVE_SPAN))
        return len(out)
