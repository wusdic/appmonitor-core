"""Action-token engine (R2) — a fine-grained action vocabulary with no host agent.

Why: per-tick counters say *how much* an IP did, not *what*. Sequence,
novelty, timing, beacon, budget and identity models all need the individual
actions: which templated request, to which destination, with which outcome,
from which client stack, and when (sub-second). Raw paths and qnames carry
ids that make every request unique, so they are templated per system with
Drain-lite (lib/template.py) into (channel, op, resource, outcome) tokens,
while the masked object ids are kept aside for breadth budgets.

Per tick, per real entity with observations (ts = ctx.now):
  act.events (sum of weights), act.tokens, act.stream (<= 512 time-ordered
  rows [ts, token_id, outcome, up, down, dest_id, stack_id]), act.stream_frac,
  act.distinct_templates, act.new_template_ratio (HTTP only), act.objs,
  act.rare_events.
Every other real entity seen in the last 30 d gets act.events = 0 with
touch=False: absence is data, and the zero-filled act.events series is the
wall-clock grid clock of D0/D2 (derived/fresh.py), while last_seen keeps
meaning "last observed on the wire".

State: one Templater per system plus R2's own decayed (half-life 7 d)
distinct-entity count per destination, in model.template@(s, '__system__').
The exact storage forms (stream dtype, outcome codes, destination ids,
ts_sample semantics) are documented in lib/m_template.py, which consumers
use to read them. The vocabulary is a representation, not a model of
normality, so it is not trust-gated: an attacker's tokens must get ids too.
"""
from __future__ import annotations

import heapq
import math
from itertools import chain, islice
from operator import itemgetter
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import (AcquisitionMethod, MetricKind, Observation, RawMetric,
                              is_pseudo_entity)
from ..behavior.lib import m_template as MT
from ..behavior.lib import template as TPL
from ..behavior.lib.classkeys import SYSTEM_KEY
from ..behavior.lib.combine import seeded_uniform
from ..behavior.lib import grains as GR
from ..behavior.lib import sketch as SK
from ..behavior.lib.sketch import HyperLogLog
from ..behavior.lib.stack import stack_id, stack_token

_ACTIVE = (AcquisitionMethod.ACTIVE_PROBE, AcquisitionMethod.ACTIVE_DNS,
           AcquisitionMethod.ACTIVE_TLS)
_ACTIVE_METHODS = frozenset(_ACTIVE) | frozenset(m.value for m in _ACTIVE)   # enum or plain str
_PASSIVE_SPAN = AcquisitionMethod.PASSIVE_SPAN
_TLS_PROTOS = frozenset({"tls", "https", "ssl"})
_DNS_OUTCOME = {"": MT.OUTCOME_UNKNOWN, "noerror": MT.OUTCOME_OK, "0": MT.OUTCOME_OK,
                "nxdomain": MT.OUTCOME_CLIENT, "3": MT.OUTCOME_CLIENT}
_OBJ_MASKS = TPL.OBJ_MASKS
_INF = math.inf

DEST_CAP = 16384              # destinations tracked per system (prevalence state)
PRUNE_AGE_HL = 8.0            # (dest, entity) entries older than 8 half-lives are dropped
SWEEP_EVERY_S = 86400.0       # exact recompute + prune of the decayed sums
_PY_EXPAND_MAX = 256          # up to this many ts_sample rows, expand aggregates in Python

# Module memos (pure functions of their key); bounded, cleared when full.
_MEMO_MAX = 1 << 14
_SID: Dict[Any, int] = {}
_DEST: Dict[str, Tuple[str, int]] = {}
_TKEY: Dict[str, str] = {}


def _memo(d: Dict, k: Any, v: Any) -> None:
    if len(d) >= _MEMO_MAX:
        d.clear()
    d[k] = v


def _weight(extra: Optional[Dict[str, Any]]) -> int:
    """Events an observation stands for: int(extra.count), default 1.
    A non-numeric / non-finite count is one event; <= 0 is none."""
    if not extra:
        return 1
    c = extra.get("count", 1)
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
    """A byte count; NaN, inf, negative or unparsable -> 0 (as template.size_class)."""
    if x.__class__ is not float:
        try:
            x = float(x)
        except (TypeError, ValueError, OverflowError):
            return 0.0
    return x if 0.0 <= x < _INF else 0.0


def _ts(t: Any, now: float) -> float:
    """A finite float event time, else ctx.now (the event happened this tick)."""
    try:
        t = float(t)
    except (TypeError, ValueError, OverflowError):
        return now
    return t if -_INF < t < _INF else now


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError, OverflowError):
        return math.nan


def _sample_head(ex: Dict[str, Any], w: int) -> Optional[Any]:
    """The first min(TS_SAMPLE_MAX, w) entries of extra['ts_sample'] (unparsed;
    _rows_array converts them in bulk), or None when there is none."""
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


def _stack_of(o: Observation) -> int:
    ex = o.extra
    ja4 = ex.get("ja4") if ex else None
    key = (o.ja3, o.user_agent, o.ttl, o.win_size, ja4)
    try:
        sid = _SID.get(key)
    except TypeError:                          # unhashable ja4
        return stack_id(stack_token(o.ja3, o.user_agent, o.ttl, o.win_size, str(ja4)))
    if sid is None:
        sid = stack_id(stack_token(o.ja3, o.user_agent, o.ttl, o.win_size, ja4))
        _memo(_SID, key, sid)
    return sid


def _dest(host: Any) -> Tuple[str, int]:
    """(destination name, dest_id) of a host / SNI / qname / peer."""
    if host.__class__ is not str:
        host = "" if host is None else str(host)
    hit = _DEST.get(host)
    if hit is None:
        name = MT.dest_name_of(host)
        hit = (name, MT.dest_id_of(name))
        if len(host) <= 256:
            _memo(_DEST, host, hit)
    return hit


def _tkey(tok: str) -> str:
    k = _TKEY.get(tok)
    if k is None:
        k = MT.template_key(tok)
        if len(tok) <= 512:
            _memo(_TKEY, tok, k)
    return k


def _http_outcome(status: Any) -> int:
    if status.__class__ is not int:
        try:
            status = int(status)
        except (TypeError, ValueError, OverflowError):
            return MT.OUTCOME_UNKNOWN
    return status // 100 if 100 <= status <= 599 else MT.OUTCOME_UNKNOWN


def _rcode_key(rc: Any) -> str:
    """Normalised DNS rcode key: 'noerror', 'nxdomain', '0', '3', ... ('' = none)."""
    if rc is None:
        return ""
    if rc.__class__ is float and math.isfinite(rc) and rc == int(rc):
        rc = int(rc)
    return str(rc).strip().lower()


def _conn_outcome(down: float, flags: Any) -> int:
    if down > 0.0:
        return MT.OUTCOME_OK
    if flags and "RST" in str(flags).upper():
        return MT.OUTCOME_CLIENT
    return MT.OUTCOME_UNKNOWN


class _Acc:
    """Per-entity accumulator for one tick."""
    __slots__ = ("events", "rows", "chunks", "tokens", "http_n", "http_new", "objs", "dests",
                 "chunk_w", "http_ev")

    def __init__(self) -> None:
        self.events = 0
        self.chunk_w: List[int] = []           # weight of each aggregated chunk (slot tally)
        self.http_ev: List[Tuple[float, str, int]] = []   # spec v2.1: (t, token, w) of HTTP events
        self.rows: List[Tuple[float, int, int, float, float, int, int]] = []
        # aggregated records: (ts_sample head, t0, token_id, outcome, up, down, dest_id, stack_id)
        self.chunks: List[Tuple[Any, float, int, int, float, float, int, int]] = []
        self.tokens: Dict[str, int] = {}
        self.http_n = 0
        self.http_new = 0
        self.objs: Dict[str, Dict[str, None]] = {}     # key -> ordered set of ids
        self.dests: Set[int] = set()


# ------------------------------------------------------------ decayed sums
def _touch(s: float, t_ref: float, last: Dict[str, float], key: str, now: float,
           h: float) -> Tuple[float, float]:
    """Decayed distinct count S = sum_k 2^-((T - t_k)/h) after `key` was seen at
    `now`; O(1). T = max(t_ref, now) so a clock that steps back cannot inflate S."""
    if t_ref == t_ref:
        big_t = t_ref if t_ref > now else now
        if s > 0.0 and big_t > t_ref:
            s *= 2.0 ** (-(big_t - t_ref) / h)
        elif not s > 0.0:
            s = 0.0
    else:
        big_t, s = now, 0.0
    t_old = last.get(key)
    t_new = now if t_old is None or now > t_old else t_old
    if t_old is not None:
        s -= 2.0 ** (-(big_t - t_old) / h)
    s += 2.0 ** (-(big_t - t_new) / h)
    last[key] = t_new
    return (s if s > 0.0 else 0.0), big_t


def _exact(last: Dict[str, float], h: float, max_age: float) -> Tuple[float, float]:
    """Prune entries older than max_age (in place) and recompute (S, T) exactly."""
    if not last:
        return 0.0, math.nan
    big_t = max(last.values())
    for k in [k for k, t in last.items() if big_t - t > max_age]:
        del last[k]
    return sum(2.0 ** (-(big_t - t) / h) for t in last.values()), big_t


def _decay_to(s: float, t_ref: float, now: float, h: float) -> float:
    if not (t_ref == t_ref) or s <= 0.0:
        return 0.0
    return s * 2.0 ** (-(now - t_ref) / h) if now > t_ref else s


class ActionTokenEngine(Engine):
    name = "raw.action_token"
    layer = "raw"
    consumes = ["<observations>", "store.entities", "store.last_seen"]
    produces = [
        "act.events", "act.tokens", "act.stream", "act.stream_frac", "act.distinct_templates",
        "act.new_template_ratio", "act.objs", "act.rare_events", "model.template",
    ]
    description = ("Drain-lite action tokens per system, time-ordered event stream with "
                   "stack/destination ids, object ids, rare-destination events and a "
                   "zero-filled act.events grid clock.")
    interval = 1

    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        self.last_dropped_pseudo = 0
        self.last_dropped_active = 0
        self._canon = False

    def health_record(self, ok: bool = True) -> Dict[str, Any]:
        rec = super().health_record(ok)
        rec["dropped_pseudo"] = self.last_dropped_pseudo
        rec["dropped_active"] = self.last_dropped_active
        return rec

    # ------------------------------------------------------------------ run
    def run(self, ctx: Context, observations: Optional[List[Observation]] = None) -> int:
        store, now = ctx.store, float(ctx.now)
        self._canon = GR.canonical(ctx.config)
        groups: Dict[Tuple[str, str], List[Observation]] = {}
        bad: Set[Tuple[str, str]] = set()
        drop_pseudo = drop_active = 0
        for o in observations or ():
            m = o.method
            if m is not _PASSIVE_SPAN and m in _ACTIVE_METHODS:
                drop_active += 1                     # our own probes are not entity actions
                continue
            k = (o.system, o.entity)
            lst = groups.get(k)
            if lst is None:
                if k in bad or not k[0] or not k[1] or is_pseudo_entity(k[1]) \
                        or is_pseudo_entity(k[0]):
                    bad.add(k)
                    drop_pseudo += 1
                    continue
                lst = groups[k] = []
            lst.append(o)
        self.last_dropped_pseudo, self.last_dropped_active = drop_pseudo, drop_active

        by_sys: Dict[str, Dict[str, List[Observation]]] = {}
        for (s, e), lst in groups.items():
            by_sys.setdefault(s, {})[e] = lst
        n = 0
        active: Set[Tuple[str, str]] = set()
        for s, ents in by_sys.items():
            n += self._run_system(store, s, ents, now, active)
        n += self._zero_fill(store, now, active)
        return n

    # --------------------------------------------------------------- system
    def _run_system(self, store, s: str, ents: Dict[str, List[Observation]], now: float,
                    active: Set[Tuple[str, str]]) -> int:
        model = MT.get(store, s)
        if model is None or not isinstance(model.get("templater"), TPL.Templater):
            model = MT.from_dict(model)
        tm: TPL.Templater = model["templater"]
        if not isinstance(model.get("born"), (int, float)):
            model["born"] = float(now)                 # first tick of this vocabulary
        new_ok = MT.vocab_mature(model, now)
        canon = self._canon
        vocab = tm.vocab
        new_toks: Set[str] = set()
        dest_names: Dict[int, str] = {}
        accs: Dict[str, _Acc] = {}

        http_token, intern, dns_token = tm.http_token, tm.intern, TPL.dns_token
        for e, obs_list in ents.items():
            acc = _Acc()
            rows_append = acc.rows.append
            chunks = acc.chunks
            tokens = acc.tokens
            dests = acc.dests
            objs = acc.objs
            events = http_n = http_new = 0
            for o in obs_list:
                ex = o.extra
                if ex:                                   # aggregated / annotated record
                    w = _weight(ex)
                    if w <= 0:
                        continue
                    up = (_bytes(ex["bytes_up_total"]) / w if "bytes_up_total" in ex
                          else _bytes(o.bytes_up))
                    down = (_bytes(ex["bytes_down_total"]) / w if "bytes_down_total" in ex
                            else _bytes(o.bytes_down))
                    t = _ts(o.ts, now)
                    head = _sample_head(ex, w)
                else:                                    # one plain event (hot path)
                    w, head = 1, None
                    up, down = o.bytes_up, o.bytes_down
                    if up.__class__ is not int or up < 0:
                        up = _bytes(up)
                    if down.__class__ is not int or down < 0:
                        down = _bytes(down)
                    t = o.ts
                    if t.__class__ is not float or not -_INF < t < _INF:
                        t = _ts(t, now)
                proto = o.app_proto
                is_http = False
                if o.http_method or (proto == "http" and (o.http_path or o.http_host)):
                    host = o.http_host or o.peer
                    tok, masked = http_token(o.http_method, host, o.http_path, o.http_status, w)
                    outcome = _http_outcome(o.http_status)
                    is_http = True
                elif proto == "dns" or o.dns_qname:
                    tok, masked, host = dns_token(o.dns_qtype, o.dns_qname), None, o.dns_qname
                    rc = o.dns_rcode
                    outcome = _DNS_OUTCOME.get(rc if rc.__class__ is str and rc.islower()
                                               else _rcode_key(rc), MT.OUTCOME_SERVER)
                elif proto in _TLS_PROTOS or o.tls_sni or o.tls_version:
                    tok = TPL.tls_token(o.tls_sni, o.dst_port, up, down)
                    masked, host = None, (o.tls_sni or o.peer)
                    outcome = _conn_outcome(down, o.tcp_flags)
                else:
                    tok = TPL.l4_token(o.l4_proto or o.l3_proto, o.dst_port, up, down)
                    masked, host = None, o.peer
                    outcome = _conn_outcome(down, o.tcp_flags)

                if tok not in vocab:
                    new_toks.add(tok)
                tid = intern(tok, w)
                if tid > 0x7FFFFFFF:                     # int32 column; never in practice
                    tid = 0
                dname, did = _dest(host)
                if did and did not in dests:
                    dests.add(did)
                    dest_names[did] = dname
                sid = _stack_of(o)
                if head is None:
                    rows_append((t, tid, outcome, up, down, did, sid))
                else:
                    chunks.append((head, t, tid, outcome, up, down, did, sid))
                    acc.chunk_w.append(w)
                events += w
                tokens[tok] = tokens.get(tok, 0) + w
                if is_http:
                    http_n += w
                    if tok in new_toks:
                        http_new += w
                    if canon:
                        acc.http_ev.append((t, tok, w))
                    if masked:
                        vals = [v for m, v in masked if m in _OBJ_MASKS]
                        if vals:
                            _m, h, rest = tok.split(" ", 2)
                            tpl = rest.rsplit("|", 1)[0]
                            q = tpl.find("?")
                            key = h + (tpl[:q] if q >= 0 else tpl)
                            ids = objs.get(key)
                            if ids is None:
                                ids = objs[key] = {}
                            ids[vals[0] if len(vals) == 1 else "/".join(vals)] = None
            if events > 0:
                acc.events, acc.http_n, acc.http_new = events, http_n, http_new
                accs[e] = acc

        if not accs:
            return 0
        if canon:
            self._new_by_window(model, accs, new_toks, now)
        tm.maintain()
        rare = self._update_prevalence(model["prev"], accs, dest_names, now)
        model["version"] = int(model.get("version", 0)) + 1
        store.put_model(s, SYSTEM_KEY, MT.MODEL, model, version=model["version"], ts=now)

        n = 0
        for e, acc in accs.items():
            active.add((s, e))
            n += self._emit(store, s, e, acc, now, rare, new_ok, self._canon)
        return n

    # ------------------------------------------------ spec v2.1 new templates
    @staticmethod
    def _new_by_window(model: Dict[str, Any], accs: Dict[str, "_Acc"], new_toks: Set[str],
                       now: float) -> None:
        """Canonical grain mode: an HTTP event is 'new' when its token was
        first seen by the system less than NEW_WINDOW_S before the event
        (event time, not tick membership), so act.new_template_ratio is an
        additive part whose sum over an hour is the same at 60, 900 and
        3600 s. v2 counts every occurrence in the tick of a token new to the
        vocabulary, i.e. a whole hour of occurrences at 3600 s and one minute
        at 60 s."""
        recent: Dict[str, float] = model.setdefault("tok_first", {})
        first: Dict[str, float] = {}
        for acc in accs.values():
            for t, tok, _w in acc.http_ev:
                if tok in recent:
                    continue
                if tok in new_toks and (tok not in first or t < first[tok]):
                    first[tok] = t
        recent.update(first)
        for acc in accs.values():
            n = 0
            for t, tok, w in acc.http_ev:
                f = recent.get(tok)
                if f is not None and t - f < NEW_WINDOW_S:
                    n += w
            acc.http_new = n
        cut = now - NEW_WINDOW_S - 3600.0
        if recent and len(recent) > 256:
            for k in [k for k, v in recent.items() if v < cut]:
                del recent[k]

    # ----------------------------------------------------------- prevalence
    @staticmethod
    def _update_prevalence(prev: Dict[str, Any], accs: Dict[str, _Acc],
                           dest_names: Dict[int, str], now: float) -> Set[int]:
        """Fold this tick's (entity, destination) pairs into the decayed counts and
        return the destinations used this tick by <= RARE_SHARE of the entities."""
        h = float(prev.get("half_life_s", MT.PREV_HALF_LIFE_S))
        ents: Dict[str, float] = prev["ents"]
        dests: Dict[int, list] = prev["dests"]
        es = prev["ent_sum"]
        for e in accs:
            es[0], es[1] = _touch(es[0], es[1], ents, e, now, h)
        used: Set[int] = set()
        for e, acc in accs.items():
            for did in acc.dests:
                rec = dests.get(did)
                if rec is None:
                    rec = dests[did] = [dest_names.get(did, ""), 0.0, math.nan, {}]
                rec[1], rec[2] = _touch(rec[1], rec[2], rec[3], e, now, h)
                used.add(did)

        max_age = PRUNE_AGE_HL * h
        ls = prev.get("last_sweep", math.nan)
        if not (ls == ls) or now - ls >= SWEEP_EVERY_S or now < ls:
            es[0], es[1] = _exact(ents, h, max_age)
            for did in list(dests):
                rec = dests[did]
                rec[1], rec[2] = _exact(rec[3], h, max_age)
                if not rec[3]:
                    del dests[did]
            prev["last_sweep"] = now
        if len(dests) > DEST_CAP:
            keep = heapq.nlargest(int(DEST_CAP * 0.9), dests.items(),
                                  key=lambda kv: _decay_to(kv[1][1], kv[1][2], now, h))
            prev["dests"] = dests = dict(keep)

        n_ent = _decay_to(es[0], es[1], now, h)
        if not n_ent > 0.0:
            return set()
        lim = MT.RARE_SHARE * n_ent * (1.0 + 1e-9)
        return {did for did in used if did in dests
                and _decay_to(dests[did][1], dests[did][2], now, h) <= lim}

    # ----------------------------------------------------------------- emit
    def _emit(self, store, s: str, e: str, acc: _Acc, now: float, rare: Set[int],
              new_ok: bool = True, canon: bool = False) -> int:
        total = acc.events
        arr = _rows_array(acc)                                   # every sampled event
        order = np.argsort(arr["ts"], kind="stable")
        rare_ev: Optional[Dict[int, List[List[float]]]] = None
        hit = acc.dests & rare if rare else None
        if hit:                                  # every event, outside the stream cap
            dids = arr["dest_id"][order]
            rare_ev = {}
            for d in sorted(hit):
                sub = arr[order[dids == d][-MT.RARE_CAP:]]
                rare_ev[d] = np.column_stack((sub["ts"], sub["up"].astype(np.float64),
                                              sub["down"].astype(np.float64))).tolist()
        if len(order) > MT.STREAM_CAP:
            order = order[_cap_index(arr["ts"][order], s, e, now)]
        arr = arr[order]
        arr.flags.writeable = False

        tok = acc.tokens
        if len(tok) > MT.TOKENS_TOP:
            top = dict(heapq.nlargest(MT.TOKENS_TOP, tok.items(), key=itemgetter(1)))
            other = total - sum(top.values())
            if other > 0:
                top[MT.OTHER] = other
            tok = top

        out: List[Tuple[str, Any, MetricKind]] = [
            ("act.events", float(total), MetricKind.COUNTER),
            ("act.tokens", tok, MetricKind.CATEGORICAL),
            ("act.stream", arr, MetricKind.DISTRIBUTION),
            ("act.stream_frac", len(arr) / total, MetricKind.GAUGE),
            ("act.distinct_templates", float(len({_tkey(t) for t in acc.tokens})),
             MetricKind.GAUGE),
        ]
        if acc.http_n > 0 and new_ok:                  # absent while the vocabulary is young
            out.append(("act.new_template_ratio", acc.http_new / acc.http_n, MetricKind.RATE))
        if acc.objs:
            objs: Dict[str, Dict[str, Any]] = {}
            for key, ids in acc.objs.items():
                k = len(ids)
                ent: Dict[str, Any] = {"n": k, "ids": list(islice(ids, MT.OBJ_IDS_CAP))}
                if k > MT.OBJ_IDS_CAP:
                    hll = HyperLogLog()
                    hll.add_many(ids)
                    ent["hll"] = hll.to_bytes()
                objs[key] = ent
            out.append(("act.objs", objs, MetricKind.CATEGORICAL))
        if rare_ev:
            out.append(("act.rare_events", rare_ev, MetricKind.CATEGORICAL))
        if canon:
            # spec v2.1 (cadence.md §3.2-§3.3): distinct-template sketch of the
            # FULL token set and the tick's events per 15-min slot
            out.append(("act.template_ids",
                        SK.set_sketch("act.template_ids", {_tkey(t) for t in acc.tokens}),
                        MetricKind.CATEGORICAL))
            out.append(("act.slot_events", slot_events(acc, now), MetricKind.CATEGORICAL))

        for name, value, kind in out:
            store.add_raw(RawMetric(name=name, value=value, ts=now, system=s, entity=e,
                                    kind=kind, method=AcquisitionMethod.PASSIVE_SPAN))
        return len(out)

    # ------------------------------------------------------------ zero-fill
    @staticmethod
    def _zero_fill(store, now: float, active: Set[Tuple[str, str]]) -> int:
        horizon = now - MT.ZERO_FILL_S
        n = 0
        for s in store.systems():
            for e in store.entities(s):
                if (s, e) in active:
                    continue
                ls = store.last_seen(s, e)
                if ls is None or ls < horizon:
                    continue
                store.add_raw(RawMetric(name="act.events", value=0.0, ts=now, system=s,
                                        entity=e, kind=MetricKind.COUNTER,
                                        method=AcquisitionMethod.PASSIVE_SPAN), touch=False)
                n += 1
        return n


SLOT_S = 900.0
NEW_WINDOW_S = 900.0           # spec v2.1: an event is new within 15 min of its token's first sighting


def slot_events(acc: _Acc, now: float) -> Dict[float, float]:
    """{slot_start_epoch: events} of one tick (spec v2.1, cadence.md §3.3):
    every event counts, a plain record at its ts, an aggregated record's
    weight split over its ts_sample offsets proportionally (the whole weight
    at t0 when it has no valid sample). Slots are epoch-aligned 15-min slots
    (local offsets are multiples of 15 min, so they are local slots too)."""
    out: Dict[float, float] = {}
    for r in acc.rows:
        t = r[0]
        k = math.floor(t / SLOT_S) * SLOT_S if -_INF < t < _INF else math.floor(now / SLOT_S) * SLOT_S
        out[k] = out.get(k, 0.0) + 1.0
    lo, hi = -MT.TS_ABSOLUTE_MIN, MT.TS_ABSOLUTE_MIN
    for (head, t0, *_rest), w in zip(acc.chunks, acc.chunk_w):
        ts: List[float] = []
        for x in head:
            v = x if x.__class__ is float else _f(x)
            if -_INF < v < _INF:
                ts.append(t0 + v if lo < v < hi else v)
        if not ts:
            ts = [t0]
        share = float(w) / len(ts)
        for t in ts:
            k = math.floor(t / SLOT_S) * SLOT_S
            out[k] = out.get(k, 0.0) + share
    return out


def _expand_chunks(ch: List[tuple]) -> List[tuple]:
    """Row tuples of aggregated chunks (the small-n twin of the numpy path in
    _rows_array; same arithmetic and the same drop / fallback rules)."""
    out: List[tuple] = []
    lo, hi = -MT.TS_ABSOLUTE_MIN, MT.TS_ABSOLUTE_MIN
    for head, t0, tid, oc, up, down, did, sid in ch:
        k0 = len(out)
        for x in head:
            v = x if x.__class__ is float else _f(x)
            if -_INF < v < _INF:
                out.append((t0 + v if lo < v < hi else v, tid, oc, up, down, did, sid))
        if len(out) == k0:
            out.append((t0, tid, oc, up, down, did, sid))
    return out


def _rows_array(acc: _Acc) -> np.ndarray:
    """All sampled events of an entity as one STREAM_DTYPE array (unsorted):
    plain rows via np.array, aggregated chunks column-wise in bulk (or as
    tuples when few). ts_sample entries are offsets from t0 unless
    |v| >= TS_ABSOLUTE_MIN (epochs); non-finite or unparsable entries are
    dropped, and a record left with no valid entry keeps one row at t0."""
    dt = MT.STREAM_DTYPE
    ch = acc.chunks
    lens = [len(c[0]) for c in ch]
    n = sum(lens)
    if n <= _PY_EXPAND_MAX:
        rows = acc.rows + _expand_chunks(ch) if ch else acc.rows
        return np.array(rows, dtype=dt) if rows else np.zeros(0, dtype=dt)
    parts = []
    if acc.rows:
        parts.append(np.array(acc.rows, dtype=dt))
    if ch:
        try:
            flat = np.fromiter(chain.from_iterable(c[0] for c in ch), dtype=np.float64, count=n)
        except (TypeError, ValueError, OverflowError):
            flat = np.fromiter((_f(x) for c in ch for x in c[0]), dtype=np.float64, count=n)
        t0s = np.repeat(np.array([c[1] for c in ch], dtype=np.float64), lens)
        with np.errstate(invalid="ignore"):
            ts = np.where(np.abs(flat) < MT.TS_ABSOLUTE_MIN, flat + t0s, flat)
        ok = np.isfinite(ts)
        if not ok.all():
            offs = np.cumsum([0] + lens[:-1])
            bad = offs[np.add.reduceat(ok.astype(np.int64), offs) == 0]
            ts[bad] = t0s[bad]
            ok[bad] = True
        a = np.empty(n, dtype=dt)
        a["ts"] = ts
        for j, f in enumerate(("token_id", "outcome", "up", "down", "dest_id", "stack_id"), 2):
            a[f] = np.repeat(np.array([c[j] for c in ch]), lens)
        parts.append(a if ok.all() else a[ok])
    if not parts:
        return np.zeros(0, dtype=dt)
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def _cap_index(ts: np.ndarray, s: str, e: str, now: float) -> np.ndarray:
    """Indices of <= STREAM_CAP rows kept as whole sessions (split at gaps >
    SESSION_GAP_S), chosen in a seeded random order and packed greedily
    (reservoir over session blocks: take a block if it still fits, else skip
    it). A session longer than the cap contributes one contiguous chunk, so
    gaps <= SESSION_GAP_S between kept rows are always true gaps. `ts` is
    sorted; the result is ascending. Blocks are few (a session gap is > 30 s,
    so about span / 30 s at most), hence the plain loop."""
    cap = MT.STREAM_CAP
    cut = (np.flatnonzero(np.diff(ts) > MT.SESSION_GAP_S) + 1).tolist()
    rng = np.random.default_rng(int(seeded_uniform(s, e, float(now), "act.stream") * 2.0 ** 53))
    blocks: List[Tuple[int, int]] = []
    for a, b in zip([0] + cut, cut + [len(ts)]):
        if b - a > cap:
            a += int(rng.integers(-(-(b - a) // cap))) * cap
            b = min(a + cap, b)
        blocks.append((a, b))
    budget = cap
    keep: List[int] = []
    for i in rng.permutation(len(blocks)).tolist():
        a, b = blocks[i]
        if b - a <= budget:
            keep.append(i)
            budget -= b - a
            if budget == 0:
                break
    keep.sort()
    return np.concatenate([np.arange(*blocks[i]) for i in keep])
