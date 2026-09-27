"""Read accessors and storage forms for R2's outputs: model.template and the act.* series.

Owner: R2 (engines/raw/action_token.py). Consumers (B08 novelty, B10 sequence,
B11 timing, B12 beacon, B13 budget, B16 attribution, B20, B22, D2 session)
read through these functions instead of reaching into the model or decoding
the series themselves, so the layout can evolve in one place. Consumers never
mutate the model or the arrays returned here (act.stream arrays are stored
read-only).

Why the vocabulary is system-wide: token ids must mean the same thing for
every entity of a system (sequence models back off entity -> class ->
system), so there is exactly one Templater per system, persisted as
model.template@(s, '__system__'). Ids are never reused (lib/template.py):
an evicted token's id reads '{rare}' from then on, so an old act.stream row
never silently changes meaning.

model.template@(s, '__system__') (a dict, stored by reference):
    {
      "fmt": 1,
      "version": int,              # ticks on which the system learned something
      "templater": Templater,      # lib/template.py (vocab, prefix trees)
      "prev": {                    # R2's own destination prevalence (half-life 7 d)
        "half_life_s": float,
        "ents": {entity: last_ts},           # entities R2 saw in the system
        "ent_sum": [S, t_ref],               # sum_e 2^-((t_ref - last_ts_e)/H) at t_ref
        "dests": {dest_id: [name, S, t_ref, {entity: last_ts}]},
        "last_sweep": ts,
      },
    }
  to_dict(model) / from_dict(d) give a JSON-safe form (checkpoints, persistence).

Series written by R2 (all at ts = ctx.now, raw layer, via store.add_raw):
  act.events          float  sum of observation weights (extra.count); ZERO-FILLED
                             (0.0, touch=False) for every real entity seen in the
                             last 30 d with no observation this tick (the D0/D2 grid
                             clock). Written every tick for every such entity.
  act.tokens          {token: n}  top TOKENS_TOP by weighted count plus '__other__'
                             (the remainder, only when > 0). Token strings, not ids.
  act.stream          numpy structured array, dtype STREAM_DTYPE, sorted by ts,
                             at most STREAM_CAP rows, read-only. One row per event:
                               ts        float64  event time (sub-second)
                               token_id  int32    Templater id (0 = '{rare}')
                               outcome   int8     OUTCOME code (see below)
                               up, down  float32  bytes of THIS event (non-finite or
                                                  negative inputs -> 0)
                               dest_id   int32    dest_id_of(destination name), 0 = none
                               stack_id  int32    lib/stack.stack_id(stack_token(...)),
                                                  the same id R3 writes
  act.stream_frac     float  kept rows / sum of weights, in (0, 1]. Counts taken from
                             the stream are reweighted by 1/stream_frac
                             (stream_window gives per-row weights). Below 1 when
                             the cap dropped whole sessions, or when aggregated
                             observations carried fewer ts_sample offsets than
                             their count.
  act.distinct_templates float  distinct template_key(token) this tick.
  act.new_template_ratio float  fraction of the entity's HTTP events (weighted) whose
                             token entered the system vocabulary this tick; written
                             only when the entity had HTTP events (n = http.requests).
  act.objs            {obj_key: {'n': int, 'ids': [str, ...], 'hll'?: bytes}}
                             obj_key = '<host><path template>' without the query
                             (e.g. 'erp.corp/orders/view/{num}'); an id is the value
                             at the {num}/{id}/{uuid} positions of one request, joined
                             with '/' when there are several. n = exact distinct count
                             this tick; ids = the first OBJ_IDS_CAP distinct ids in
                             arrival order; 'hll' = HyperLogLog(p=10).to_bytes() of
                             every id, present only when n > OBJ_IDS_CAP. The key
                             uses the learned template, so an endpoint's first two
                             hits in a system read '{var}' (lib/template.py). An
                             aggregated record contributes the id of its own path.
  act.rare_events     {dest_id: [[ts, up, down], ...]}  every event of the entity to a
                             destination used by <= RARE_SHARE of the system's
                             entities (decayed, half-life 7 d); sorted by ts, the
                             newest RARE_CAP per destination, outside the stream cap
                             (up/down carry the stream's float32 rounding).
  Every series except act.events is written only for entities with events this
  tick; its absence at a tick means "no activity", never "unknown".

Stream sampling contract (consumers computing gaps, e.g. B11 / D2): rows are
whole sessions (split at gaps > SESSION_GAP_S) chosen by seeded random
sampling; a session longer than STREAM_CAP contributes one contiguous chunk.
So a gap <= SESSION_GAP_S between consecutive rows is always a true
inter-event gap; a larger gap may span dropped sessions when stream_frac < 1.

Aggregated observations (extra.count = w > 1): the token is counted with
weight w; stream rows come from extra['ts_sample'] (the first min(64, w)
entries). Entries are offsets in seconds from obs.ts; an entry with
|v| >= 1e8 is taken as an absolute epoch. Non-finite or unparsable entries
are dropped; a record with no usable entry, or without ts_sample, is one row
at obs.ts. up/down per row = extra['bytes_*_total'] / w when present, else
obs.bytes_*. A non-finite obs.ts reads as ctx.now; a non-numeric count is
one event and a count <= 0 is none. Observations of our own active probes
(ACTIVE_PROBE / ACTIVE_DNS / ACTIVE_TLS) and of pseudo-entities are ignored.

OUTCOME codes (int8, uniform across channels):
    0 unknown / no response    2 success (HTTP 2xx, DNS NOERROR, bytes returned)
    1 HTTP 1xx                 3 HTTP 3xx
    4 client-side failure (HTTP 4xx, DNS NXDOMAIN, TCP RST)
    5 server-side failure (HTTP 5xx, DNS SERVFAIL / REFUSED / other rcodes)
  outcome_class(code) -> '2xx' etc.; seq_token(token, code) -> 'token|2xx' for
  non-HTTP tokens (HTTP tokens already end in their status class).

Destinations (dest_id_of(name), a 31-bit blake2b id like stack_id):
    HTTP host / TLS SNI / DNS qname -> eTLD+1 (lib/names.etld1); an IPv4 literal
    -> its /24 ('203.0.113.0/24'), IPv6 -> its /64; L4 -> the peer the same way.

Accessor signatures (all pure reads; missing data gives the documented default):
    get(store, s) -> dict | None
    templater(store, s) -> Templater | None
    version(store, s) -> int                               (0 when absent)
    token_str(store, s, token_id) -> str                   ('{rare}' when unknown)
    token_id(store, s, token) -> int                       (0 when not in the vocab)
    token_count(store, s, token_or_id) -> float            (Space-Saving count, 0.0)
    vocab_size(store, s) -> int
    tokens_of(store, s, ids) -> list[str]
    template_key(token) -> str
    outcome_class(code) -> str;  seq_token(token, code) -> str
    dest_id_of(name) -> int;  dest_name(store, s, dest_id) -> str | None
    dest_entities(store, s, dest_id, now=None) -> float    (decayed distinct entities)
    dest_prevalence(store, s, dest_id, now=None) -> float  (share in [0, 1]; NaN unknown)
    system_entities(store, s, now=None) -> float           (decayed entity count)
    stream_rows(store, s, e, ts) -> ndarray[STREAM_DTYPE]  (empty when absent)
    stream_frac(store, s, e, ts) -> float                  (NaN when absent)
    stream_ticks(store, s, e, since, until=None) -> [(tick_ts, rows, frac)]
    stream_window(store, s, e, since, until=None) -> (rows, w)   w = 1/frac per row
    rare_events(store, s, e, ts) -> {dest_id: ndarray (k, 3) float64}
    objs(store, s, e, ts) -> dict;  objs_hll(entry) -> HyperLogLog
    to_dict(model) -> dict;  from_dict(d) -> model dict
"""
from __future__ import annotations

import hashlib
import ipaddress
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from . import names as _names
from .classkeys import SYSTEM_KEY
from .sketch import HyperLogLog
from .template import RARE_TOKEN, Templater, channel_of

MODEL = "model.template"
FMT = 1

STREAM_DTYPE = np.dtype([
    ("ts", "<f8"), ("token_id", "<i4"), ("outcome", "i1"),
    ("up", "<f4"), ("down", "<f4"), ("dest_id", "<i4"), ("stack_id", "<i4"),
])
STREAM_FIELDS: Tuple[str, ...] = STREAM_DTYPE.names

STREAM_CAP = 512              # rows per entity per tick
SESSION_GAP_S = 30.0          # intra-tick session split for the cap
TOKENS_TOP = 256              # act.tokens entries before '__other__'
OTHER = "__other__"
OBJ_IDS_CAP = 256             # ids listed per act.objs template; hll above this
RARE_CAP = 256                # act.rare_events per destination per tick
RARE_SHARE = 0.2              # rare destination: used by <= 20 % of the entities
PREV_HALF_LIFE_S = 7 * 86400.0
TS_SAMPLE_MAX = 64
TS_ABSOLUTE_MIN = 1e8         # a ts_sample entry this large is an epoch, not an offset
ZERO_FILL_S = 30 * 86400.0

OUTCOME_UNKNOWN, OUTCOME_OK, OUTCOME_CLIENT, OUTCOME_SERVER = 0, 2, 4, 5
_OUTCOME_CLASS = ("0xx", "1xx", "2xx", "3xx", "4xx", "5xx")
_ID_MASK = 0x7FFFFFFF


# ------------------------------------------------------------------ pure helpers
def dest_id_of(name: str) -> int:
    """31-bit id of a destination name (blake2b-8 big-endian & 0x7FFFFFFF, 0 -> 1);
    '' / None -> 0 (no destination)."""
    if not name:
        return 0
    d = hashlib.blake2b(str(name).encode("utf-8", "surrogatepass"), digest_size=8).digest()
    return (int.from_bytes(d, "big") & _ID_MASK) or 1


def dest_name_of(host: str) -> str:
    """Destination name of a host / SNI / qname / peer: eTLD+1, or the /24 (/64)
    of an IP literal; '' when empty."""
    h = _names.normalize_host(host if isinstance(host, str) else str(host or ""))
    if not h:
        return ""
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return _names.etld1(h)
    plen = 24 if ip.version == 4 else 64
    return str(ipaddress.ip_network(f"{ip}/{plen}", strict=False))


def outcome_class(code: Any) -> str:
    """OUTCOME code -> '0xx'..'5xx' (anything else -> '0xx')."""
    try:
        c = int(code)
    except (TypeError, ValueError, OverflowError):
        return "0xx"
    return _OUTCOME_CLASS[c] if 0 <= c <= 5 else "0xx"


def seq_token(token: str, code: Any) -> str:
    """Sequence symbol 'token|outcome' (B10); HTTP tokens already carry it."""
    if channel_of(token) == "http":
        return token
    return f"{token}|{outcome_class(code)}"


def template_key(token: str) -> str:
    """The (channel, op, resource) part of a token: HTTP without '|status class',
    TLS / L4 without the size class, DNS unchanged. act.distinct_templates counts these."""
    ch = channel_of(token)
    if ch == "http":
        return token.rsplit("|", 1)[0]
    if ch in ("tls", "l4"):
        return token.split(" ", 1)[0]
    return token


# ------------------------------------------------------------------ model reads
def new_model() -> Dict[str, Any]:
    return {"fmt": FMT, "version": 0, "templater": Templater(),
            "prev": {"half_life_s": PREV_HALF_LIFE_S, "ents": {}, "ent_sum": [0.0, math.nan],
                     "dests": {}, "last_sweep": math.nan}}


def get(store, system: str) -> Optional[Dict[str, Any]]:
    m = store.get_model(system, SYSTEM_KEY, MODEL, default=None)
    return m if isinstance(m, dict) else None


def templater(store, system: str) -> Optional[Templater]:
    m = get(store, system)
    if m is None:
        return None
    t = m.get("templater")
    return t if isinstance(t, Templater) else (Templater.from_dict(t) if t else None)


def version(store, system: str) -> int:
    m = get(store, system)
    return int(m.get("version", 0)) if m else 0


def token_str(store, system: str, tid: Any) -> str:
    t = templater(store, system)
    return t.token_of(tid) if t is not None else RARE_TOKEN


def tokens_of(store, system: str, ids: Iterable[Any]) -> List[str]:
    t = templater(store, system)
    if t is None:
        return [RARE_TOKEN for _ in ids]
    return [t.token_of(i) for i in ids]


def token_id(store, system: str, token: str) -> int:
    t = templater(store, system)
    if t is None:
        return 0
    e = t.vocab.get(token)
    return int(e[0]) if e is not None else 0


def token_count(store, system: str, token: Any) -> float:
    """Space-Saving count of a token (string or id); 0.0 when unknown / evicted."""
    t = templater(store, system)
    if t is None:
        return 0.0
    if not isinstance(token, str):
        token = t.token_of(token)
    e = t.vocab.get(token)
    return float(e[1]) if e is not None else 0.0


def vocab_size(store, system: str) -> int:
    t = templater(store, system)
    return len(t.vocab) if t is not None else 0


# ---------------------------------------------------------- destination prevalence
def _decayed(s_t: Any, now: Optional[float], h: float) -> float:
    s, t_ref = float(s_t[0]), float(s_t[1])
    if not math.isfinite(t_ref) or s <= 0.0:
        return 0.0
    if now is None or now <= t_ref:
        return s
    return s * 2.0 ** (-(now - t_ref) / h)


def _prev(store, system: str) -> Optional[Dict[str, Any]]:
    m = get(store, system)
    return m.get("prev") if m else None


def dest_name(store, system: str, did: Any) -> Optional[str]:
    p = _prev(store, system)
    if not p:
        return None
    try:
        e = p["dests"].get(int(did))
    except (TypeError, ValueError, OverflowError):
        return None
    return e[0] if e is not None else None


def system_entities(store, system: str, now: Optional[float] = None) -> float:
    """Decayed count of the entities R2 has seen in the system (as of `now`,
    default the last update)."""
    p = _prev(store, system)
    if not p:
        return 0.0
    return _decayed(p["ent_sum"], now, float(p.get("half_life_s", PREV_HALF_LIFE_S)))


def dest_entities(store, system: str, did: Any, now: Optional[float] = None) -> float:
    """Decayed number of distinct entities that used the destination."""
    p = _prev(store, system)
    if not p:
        return 0.0
    try:
        e = p["dests"].get(int(did))
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if e is None:
        return 0.0
    return _decayed((e[1], e[2]), now, float(p.get("half_life_s", PREV_HALF_LIFE_S)))


def dest_prevalence(store, system: str, did: Any, now: Optional[float] = None) -> float:
    """Share of the system's entities using the destination; NaN when unknown."""
    p = _prev(store, system)
    if not p:
        return math.nan
    try:
        known = int(did) in p["dests"]
    except (TypeError, ValueError, OverflowError):
        return math.nan
    if not known:
        return math.nan
    n = system_entities(store, system, now)
    if not n > 0.0:
        return math.nan
    return min(1.0, dest_entities(store, system, did, now) / n)


# ------------------------------------------------------------------ series reads
_EMPTY = np.zeros(0, dtype=STREAM_DTYPE)
_EMPTY.flags.writeable = False


def _raw_at(store, system: str, entity: str, name: str, ts: float) -> Any:
    m = store.latest_raw_at(system, entity, name, ts)
    return m.value if m is not None else None


def stream_rows(store, system: str, entity: str, ts: float) -> np.ndarray:
    """The act.stream rows written at tick ts (read-only), or an empty array."""
    v = _raw_at(store, system, entity, "act.stream", ts)
    if isinstance(v, np.ndarray) and v.dtype == STREAM_DTYPE:
        return v
    return _EMPTY


def stream_frac(store, system: str, entity: str, ts: float) -> float:
    v = _raw_at(store, system, entity, "act.stream_frac", ts)
    try:
        return float(v) if v is not None else math.nan
    except (TypeError, ValueError):
        return math.nan


def stream_ticks(store, system: str, entity: str, since: float,
                 until: Optional[float] = None) -> List[Tuple[float, np.ndarray, float]]:
    """[(tick_ts, rows, stream_frac)] for ticks with since <= tick_ts <= until,
    oldest first. Retention of act.stream is 1 h (contract B)."""
    out: List[Tuple[float, np.ndarray, float]] = []
    tail = [m for m in store.raw_tail(system, entity, "act.stream", 4096)
            if m.ts >= since and (until is None or m.ts <= until)]
    if not tail:
        return out
    fr = {m.ts: m.value for m in store.raw_tail(system, entity, "act.stream_frac", 4096)
          if m.ts >= since}
    for m in tail:
        if isinstance(m.value, np.ndarray) and m.value.dtype == STREAM_DTYPE:
            f = fr.get(m.ts)
            try:
                f = float(f) if f is not None else math.nan
            except (TypeError, ValueError):
                f = math.nan
            out.append((m.ts, m.value, f))
    return out


def stream_window(store, system: str, entity: str, since: float,
                  until: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(rows concatenated oldest first, per-row weight 1/stream_frac). A missing or
    invalid frac weighs 1 (the rows are then taken as complete)."""
    ticks = stream_ticks(store, system, entity, since, until)
    if not ticks:
        return _EMPTY, np.zeros(0, dtype=np.float64)
    rows = np.concatenate([r for _, r, _ in ticks])
    w = np.concatenate([np.full(len(r), 1.0 / f if (f == f and 0.0 < f <= 1.0) else 1.0)
                        for _, r, f in ticks])
    return rows, w


def rare_events(store, system: str, entity: str, ts: float) -> Dict[int, np.ndarray]:
    """{dest_id: float64 array (k, 3) of [ts, up, down]} written at tick ts."""
    v = _raw_at(store, system, entity, "act.rare_events", ts)
    if not isinstance(v, dict):
        return {}
    return {int(k): np.asarray(rows, dtype=np.float64).reshape(-1, 3) for k, rows in v.items()}


def objs(store, system: str, entity: str, ts: float) -> Dict[str, Dict[str, Any]]:
    v = _raw_at(store, system, entity, "act.objs", ts)
    return v if isinstance(v, dict) else {}


def objs_hll(entry: Dict[str, Any]) -> HyperLogLog:
    """HLL of one act.objs entry: its 'hll' registers, else built from its ids."""
    b = entry.get("hll") if isinstance(entry, dict) else None
    if b:
        return HyperLogLog.from_bytes(b)
    h = HyperLogLog()
    ids = entry.get("ids") if isinstance(entry, dict) else None
    if ids:
        h.add_many(ids)
    return h


# ------------------------------------------------------------------ serialisation
def to_dict(model: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-safe copy of a model.template dict."""
    t = model.get("templater")
    p = model.get("prev") or {}
    return {
        "fmt": FMT,
        "version": int(model.get("version", 0)),
        "templater": t.to_dict() if isinstance(t, Templater) else (t or {}),
        "prev": {
            "half_life_s": float(p.get("half_life_s", PREV_HALF_LIFE_S)),
            "ents": {str(k): float(v) for k, v in (p.get("ents") or {}).items()},
            "ent_sum": [float(x) if math.isfinite(float(x)) else None
                        for x in (p.get("ent_sum") or [0.0, math.nan])],
            "dests": {str(k): [e[0], float(e[1]), float(e[2]),
                               {str(a): float(b) for a, b in e[3].items()}]
                      for k, e in (p.get("dests") or {}).items()},
            "last_sweep": (float(p["last_sweep"]) if p.get("last_sweep") is not None
                           and math.isfinite(float(p["last_sweep"])) else None),
        },
    }


def from_dict(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Inverse of to_dict; None / {} -> new_model(). A live model (Templater
    inside) passes through unchanged."""
    if not d:
        return new_model()
    if isinstance(d.get("templater"), Templater):
        return d
    m = new_model()
    m["version"] = int(d.get("version", 0) or 0)
    m["templater"] = Templater.from_dict(d.get("templater"))
    p = d.get("prev") or {}
    nan = math.nan
    es = p.get("ent_sum") or [0.0, None]
    m["prev"] = {
        "half_life_s": float(p.get("half_life_s", PREV_HALF_LIFE_S)),
        "ents": {str(k): float(v) for k, v in (p.get("ents") or {}).items()},
        "ent_sum": [float(es[0] or 0.0), nan if es[1] is None else float(es[1])],
        "dests": {int(k): [str(e[0]), float(e[1]), float(e[2]),
                           {str(a): float(b) for a, b in (e[3] or {}).items()}]
                  for k, e in (p.get("dests") or {}).items()},
        "last_sweep": nan if p.get("last_sweep") is None else float(p["last_sweep"]),
    }
    return m
