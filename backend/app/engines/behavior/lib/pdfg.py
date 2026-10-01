"""Directly-follows graphs, workflows and required predecessors
(docs/lib3/progressive.md §6.14, P10). No store access; P10
(`behavior.workflow`) owns the state, P03 / P14 read it through the pure
functions at the bottom (`seq_scores`, `lookup_scope`).

Actions. act(e) = (r, v): r = the event's route template (`http.route`, ℓ1);
for non-HTTP channels `TLS <sni eTLD+1>`, `DNS <qname eTLD+1>` or `DST <net.dst>`;
v = the pattern-tree node reached through the deepest content split on the
event's path (an action variant, §6.5.2), else 0. The key string is
`r` or `r#v<nid>`. Integer ids come from a per-tree dictionary
(Space-Saving, k = 4096); an evicted action's id is RETIRED with every DFG
statistic that mentions it and is never reused, so a later action never
inherits another's counts.

Sessions (every event of every tick, for scoring and for the row annotations):
per session key (ip, sess.key) an LRU entry
    [sid, last_ts, last_key, earlier, bits, last_learned, last_mass]
`earlier` = the 64-bit hashes of the last E_MAX = 8 distinct actions of the
session (packed bytes, used to count eventually-precedes), `bits` = a 256-bit
two-hash Bloom set of every action seen (membership for the required-
predecessor check; false-positive rate about 0.5 % at 10 distinct actions,
where the 64-bit single-hash set of the text would give about 15 %).
A new session starts when P01's `ctx.sid` changes (P01 applies B10's / its own
gap rule), or, without P01, after the configured gap.

Counting (learned rows of tick t - D only, with trust, §6.9.3): the pending
annotation of each learned row (b, its predecessor a, the delay, start flag,
earlier set, mass) is counted D later with mass m x trust x damp and evidence
omega = trust x damp / (r + 1) (burst run keyed (ip, a, b), tau = 300 s, PPC-9)
into, per scope g in {'*'} U {P11 groups with >= 2 IPs active here, <= 32}:
    edges   Space-Saving k = 4096 over (g, a, b) + a 13-bin log2 delay histogram
            (1 s ... 1 h, then 'long'), H_m-decayed
    out     (g, a) outgoing mass / evidence;  cnt, starts, ends (g, a)
    prec    eventually-precedes (g, b, a) for the TOP_B = 256 most frequent b,
            with c(b) counted alongside (pcnt) so that both counts cover the
            same rows (top_b is refreshed hourly)
Mining (6 h): dep(a => b) = (|a>b| - |b>a|) / (|a>b| + |b>a| + 1) (heuristics
miner, Weijters & van der Aalst 2003); kept edges have dep >= 0.8, >= 10
evidence units on the confidence channel and >= 5 % of a's outgoing mass.
Addition to the text (measured on pack O's approvals, which return to the
next item: list -> item -> approve -> item -> approve ...): a length-two loop
(Flexible Heuristics Miner, Weijters & Ribeiro 2011; `loop2` counts the
patterns a b a) with (|aba| + |bab|) / (|aba| + |bab| + 1) >= 0.8 keeps both
directions; the published `dep` is then max(dep, loop measure) and
`dep_direct` keeps the classic value;
workflows = maximal simple paths (<= 8 actions) along kept edges from start
actions (start share >= 0.1; also any action with kept out-edges and no kept
in-edge, so that every kept edge is in a workflow — an addition to the text);
delay band per edge = [q10, q90] of its histogram; an edge that stopped is
stale and not kept (an addition: the H_l evidence of a renamed step would keep it
for weeks). Staleness counts NORMAL days of the edge's own day type (P01's
model.pcal, the §6.8.1 rule): with p = (occurrence dates + 1/2) / (normal dates of
that type spanned + 1), the edge is stale after k >= 2 missed normal dates with
(1 - p)^k < 0.05 (a daily edge after 2 workdays, a weekly one after ~3 weeks;
weekends, holidays and abnormal days never count). Without a calendar the edge
is stale when silent for >= max(7 d, 3 / its rate); requires(b, a) <=> c(b) >= 15
and the 5 % quantile of Beta(1/2 + c(b with a earlier), 1/2 + c(b without a))
>= 0.85.

Scoring (for P03): p_trans = HDR p of b among the successors of a under the
hierarchical estimate (scope g -> '*' -> action marginal, alpha = 2);
p_req = (c(b without a) + 0.5) / (c(b) + 1) for a required predecessor a absent
from the session; p_seq = min(1, 2 min(p_trans, p_req)). NaN when unscored.

Memory: every structure is a capped sketch or LRU: 5 Space-Saving tables of
k = 4096 plus one histogram per tracked edge, sessions <= S_sess (default
min(65 536, 4 x distinct session keys in 7 d)), burst runs <= S_sess, pending
annotations <= the learning sample of D + dt. Nothing grows with the number of
IPs beyond those caps, nor with the number of attributes (only the route /
sni / qname / dst columns and the tree's content splits are read).
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import re

import numpy as np

from . import pmdl
from . import psketch as PS
from .pevent import ABSENT as _ABSENT

K_ACT = 4096
K_EDGE = 4096
K_PREC = 4096
TOP_B = 256
E_MAX = 8                        # distinct earlier actions enumerated per session
N_BINS = 13                      # 12 log2 bins (1 s ... 1 h) + long
LONG_S = 3600.0
DEP_MIN = 0.8
STALE_MIN_S = 7 * 86400.0        # (no calendar) an edge silent for a week and 3 expected occurrences is stale
STALE_P = 0.05                   # (calendar) P(no occurrence on k normal days) below which an edge is stale
STALE_K_MIN = 2                  # ... and at least 2 missed normal days of its day type
CAL_DAYS = 70                    # calendar days kept per tree
EPOCH_ORD = 719163               # date(1970, 1, 1).toordinal(): local day ordinal = this + (ts + off) // 86400
L2_MIN = 0.8                     # length-two loop measure (the same threshold as dep)
EDGE_MIN = 10.0                  # evidence units (confidence channel)
SHARE_MIN = 0.05
START_MIN = 0.1
PATH_MAX = 8
WF_MAX = 32                      # workflows per scope
REQ_MIN_B = 15.0
REQ_LB = 0.85
REQ_MAX = 4                      # required predecessors kept per action
ALPHA = 2.0
SCOPES_MAX = 32
STAR = "*"
NO_KEY = "∅"
SESS_FLOOR = 256


# ================================================================ hashing
def h64(key: str) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8", "surrogatepass"), digest_size=8).digest(),
                          "little")


def bloom_bits(h: int) -> int:
    return (1 << (h & 255)) | (1 << ((h >> 8) & 255))


def bloom_has(bits: int, h: int) -> bool:
    b = bloom_bits(h)
    return (bits & b) == b


def pack_hashes(hs: Sequence[int]) -> bytes:
    return b"".join(int(h).to_bytes(8, "little") for h in hs)


def unpack_hashes(b: bytes) -> List[int]:
    return [int.from_bytes(b[i:i + 8], "little") for i in range(0, len(b), 8)]


def etld1(host: Any) -> Optional[str]:
    """eTLD+1 approximation (last two labels; no public-suffix list)."""
    if not isinstance(host, str) or not host:
        return None
    parts = [p for p in host.lower().strip(".").split(".") if p]
    if len(parts) <= 2:
        return ".".join(parts)
    return ".".join(parts[-2:])


_DIGITS = re.compile(r"\d+")


def host_key(host: Any) -> Optional[str]:
    """The service a TLS SNI / DNS name stands for: the host name (lower case,
    no port, at most its last five labels) with the digit runs of every label
    left of the registrable domain templated (a12.cdn.example.com ->
    a{n}.cdn.example.com: CDN / pod shards are one service; mail.corp.local
    and git.corp.local stay two). etld1 (two labels) made every internal
    service of one domain one action: measured on pack O, the mail and code
    systems both rendered 'TLS corp.local', so no statement named 'TLS
    mail.corp.local'."""
    if not isinstance(host, str) or not host:
        return None
    h = host.lower().strip(".")
    if h.count(":") == 1:
        h = h.split(":", 1)[0]
    parts = [p for p in h.split(".") if p]
    if not parts:
        return None
    if all(p.isdigit() for p in parts):
        return ".".join(parts)                            # an IPv4 literal
    parts = parts[-5:]
    return ".".join([_DIGITS.sub("{n}", p) for p in parts[:-2]] + parts[-2:])


def route_key(get: Any) -> Optional[str]:
    """The route part r of an action from an attribute getter (None -> skip).
    Absent attributes (pevent.ABSENT, the string '⊥') read as None."""
    raw = get

    def get(a: str) -> Any:
        v = raw(a)
        return None if v is _ABSENT or v == _ABSENT else v
    r = get("http.route")
    if isinstance(r, str) and r:
        return r
    sni = host_key(get("tls.sni"))
    if sni:
        return "TLS " + sni
    q = host_key(get("dns.qname"))
    if q:
        return "DNS " + q
    d = get("net.dst")
    if isinstance(d, str) and d:
        return "DST " + d
    return None


def split_key(key: str) -> Tuple[str, int]:
    """'POST h /x#v12' -> ('POST h /x', 12)."""
    if "#v" in key:
        r, v = key.rsplit("#v", 1)
        try:
            return r, int(v)
        except ValueError:
            return key, 0
    return key, 0


# ============================================================ delay hist
def delay_bin(d: Optional[float]) -> Optional[int]:
    if d is None or not math.isfinite(d) or d < 0:
        return None
    if d >= LONG_S:
        return N_BINS - 1
    if d < 2.0:
        return 0
    return min(N_BINS - 2, int(math.log2(d)))


def _bin_edges(i: int) -> Tuple[float, float]:
    if i == 0:
        return 0.0, 2.0
    if i == N_BINS - 1:
        return LONG_S, 2 * LONG_S
    lo = float(2 ** i)
    hi = float(2 ** (i + 1)) if i < N_BINS - 2 else LONG_S
    return lo, hi


def hist_quantile(hist: Sequence[float], q: float) -> Optional[float]:
    """q-quantile of a delay histogram (log interpolation inside a bin)."""
    h = np.asarray(hist, dtype=np.float64)
    tot = h.sum()
    if tot <= 0:
        return None
    target = q * tot
    c = 0.0
    for i, x in enumerate(h):
        if x <= 0:
            continue
        if c + x >= target:
            f = (target - c) / x
            lo, hi = _bin_edges(i)
            if i == 0:
                return lo + f * (hi - lo)
            return float(lo * (hi / lo) ** f)
        c += x
    return _bin_edges(int(np.flatnonzero(h)[-1]))[1]


def band(hist: Sequence[float], lo: float = 0.1, hi: float = 0.9) -> Optional[List[float]]:
    a, b = hist_quantile(hist, lo), hist_quantile(hist, hi)
    if a is None or b is None:
        return None
    return [round(float(a), 1), round(float(b), 1)]


# ======================================================== tracked sketch
class TrackedSS:
    """DecayedSpaceSaving that reports the key it evicts (O(1) through the
    sketch's slot layout; a set difference fallback otherwise), so dependent
    per-key state (histograms, indexes, retired ids) stays consistent."""

    __slots__ = ("ss", "_mirror")

    def __init__(self, k: int) -> None:
        self.ss = PS.DecayedSpaceSaving(k, PS.HALF_LIVES, PS.EV_HALF_LIVES)
        self._mirror: List[Hashable] = []

    def __contains__(self, key: Hashable) -> bool:
        return key in self.ss

    def __len__(self) -> int:
        return len(self.ss)

    def add(self, key: Hashable, t: float, w: float, ev: float) -> Optional[Hashable]:
        ss = self.ss
        if key in ss:
            ss.add(key, t, w, ev)
            return None
        full = len(ss) >= ss.k
        before = len(ss)
        ss.add(key, t, w, ev)
        if key not in ss:
            return None
        idx = getattr(ss, "_idx", None)
        if len(self._mirror) != before:
            self._mirror = ss.keys()
            return None if not full else self._diff(key)
        if not full:
            self._mirror.append(key)
            return None
        if idx is not None:
            i = idx[key]
            victim = self._mirror[i]
            self._mirror[i] = key
            return victim
        return self._diff(key)

    def _diff(self, key: Hashable) -> Optional[Hashable]:
        cur = set(self.ss.keys())
        gone = [k for k in self._mirror if k not in cur]
        self._mirror = self.ss.keys()
        return gone[0] if gone else None

    def discard(self, key: Hashable) -> bool:
        idx = getattr(self.ss, "_idx", None)
        i = idx.get(key) if idx is not None else None
        ok = self.ss.discard(key)
        if ok:
            if i is not None and len(self._mirror) == len(self.ss) + 1:
                last = len(self._mirror) - 1
                self._mirror[i] = self._mirror[last]
                self._mirror.pop()
            else:
                self._mirror = self.ss.keys()
        return ok

    def ev(self, key: Hashable, t: float) -> float:
        return self.ss.evidence(key, t, -1) if key in self.ss else 0.0

    def mass(self, key: Hashable, t: float) -> float:
        return self.ss.count(key, t) if key in self.ss else 0.0

    def keys(self) -> List[Hashable]:
        return self.ss.keys()

    def nbytes(self) -> int:
        return int(self.ss.nbytes() + 8 * len(self._mirror))


def _ss(k: int) -> PS.DecayedSpaceSaving:
    return PS.DecayedSpaceSaving(k, PS.HALF_LIVES, PS.EV_HALF_LIVES)


def _ev(ss: PS.DecayedSpaceSaving, key: Hashable, t: float) -> float:
    return ss.evidence(key, t, -1) if key in ss else 0.0


# ======================================================= action dictionary
class ActionDict:
    """Action key -> integer id; ids are never reused (§6.14)."""

    __slots__ = ("ss", "k2i", "i2k", "h2i", "next_id", "retired")

    def __init__(self, k: int = K_ACT) -> None:
        self.ss = TrackedSS(k)
        self.k2i: Dict[str, int] = {}
        self.i2k: Dict[int, str] = {}
        self.h2i: Dict[int, int] = {}
        self.next_id = 1
        self.retired = 0

    def _assign(self, key: str) -> int:
        i = self.k2i.get(key)
        if i is None:
            i = self.next_id
            self.next_id += 1
            self.k2i[key] = i
            self.i2k[i] = key
            self.h2i[h64(key)] = i
        return i

    def add(self, key: str, t: float, w: float, ev: float) -> Tuple[int, Optional[int]]:
        """Count an occurrence; returns (id, retired id or None)."""
        victim = self.ss.add(key, t, max(w, 1e-9), ev)
        gone = self._retire(victim) if victim is not None else None
        return self._assign(key), gone

    def ensure(self, key: str, t: float) -> Tuple[int, Optional[int]]:
        """An id for a key seen only as a predecessor (negligible mass)."""
        if key in self.k2i and key in self.ss:
            return self.k2i[key], None
        return self.add(key, t, 1e-9, 0.0)

    def _retire(self, key: Hashable) -> Optional[int]:
        i = self.k2i.pop(key, None)
        if i is None:
            return None
        self.i2k.pop(i, None)
        self.h2i.pop(h64(str(key)), None)
        self.retired += 1
        return i

    def id_of(self, key: str) -> Optional[int]:
        return self.k2i.get(key)

    def id_by_hash(self, h: int) -> Optional[int]:
        return self.h2i.get(h)

    def key_of(self, i: int) -> Optional[str]:
        return self.i2k.get(i)

    def unseen(self, t: float) -> float:
        return self.ss.ss.unseen(t)

    def nbytes(self) -> int:
        return int(self.ss.nbytes() + 3 * 72 * len(self.k2i))


# ============================================================ flow state
class FlowState:
    """Everything P10 keeps between ticks for one tree key (not published)."""

    def __init__(self, s_sess: int = 65536) -> None:
        self.acts = ActionDict()
        self.edges = TrackedSS(K_EDGE)
        self.hist: Dict[Tuple[str, int, int], List[float]] = {}
        self.edge_last: Dict[Tuple[str, int, int], float] = {}
        # per tracked edge [first local day, last local day, dates on workdays, dates on non-workdays]
        self.edge_days: Dict[Tuple[str, int, int], List[int]] = {}
        self.cal_cls: Dict[int, int] = {}       # local day ordinal -> 0 workday / 1 non-workday
        self.cal_norm: Dict[int, bool] = {}     # finished local day -> normal (P01 model.pcal)
        self.today: Optional[int] = None
        self.hL: Optional[float] = None
        self.out = _ss(K_EDGE)
        self.cnt = _ss(K_ACT)
        self.starts = _ss(K_ACT)
        self.ends = _ss(K_ACT)
        self.prec = TrackedSS(K_PREC)
        self.pcnt = _ss(K_ACT)                  # c(b) counted with prec (b in top_b only)
        self.loop2 = TrackedSS(K_EDGE)          # (g, a, b): pattern a b a (length-two loop)
        self.succ: Dict[Tuple[str, int], Set[int]] = {}
        self.by_act: Dict[int, Set[Tuple]] = {}
        self.scopes_seen: Set[str] = {STAR}
        self.sessions = PS.LRU(max(SESS_FLOOR, int(s_sess)))
        self.s_sess_cap = int(s_sess)
        self.sess_hll = PS.EpochHLL(10)
        self.burst = PS.BurstEvidence(max(SESS_FLOOR, int(s_sess)))
        self.pending: Dict[Tuple[str, float], List[Tuple]] = {}
        self.last_batch: Dict[str, float] = {}
        self.grp_ips: PS.LRU = PS.LRU(1024)
        self.grp_ss = _ss(SCOPES_MAX)
        self.top_b: Set[int] = set()
        self.top_t: Optional[float] = None
        self.ev_since_mine = 0.0
        self.gain = [0.0, 0.0]                  # decayed sum of bits saved, of rows (H_m)
        self.gain_t: Optional[float] = None
        self.marg: Optional[Tuple[float, np.ndarray, List[int], Dict[int, int]]] = None
        self.n_rows = 0
        self.cost = [0.0, 0.0, 0.0, 0.0]         # seconds / rows of the session pass, of the counting
        self.n_counted = 0
        self.n_quar = 0

    # ----------------------------------------------------------- caps
    def sess_cap(self, budget_cap: Optional[int]) -> int:
        """min(budget, 4 x distinct session keys in 7 d), floored (§6.2.3)."""
        seen = int(self.sess_hll.count())
        cap = max(SESS_FLOOR, 4 * seen)
        if budget_cap:
            cap = min(cap, int(budget_cap))
        cap = min(cap, self.s_sess_cap)
        if cap != self.sessions.cap:
            self.sessions.set_cap(cap)
            self.burst.set_cap(cap)
        return cap

    # ----------------------------------------------------------- hist
    def _hist_add(self, key: Tuple[str, int, int], b: Optional[int], t: float, m: float) -> None:
        if b is None or m <= 0:
            return
        if self.hL is None:
            self.hL = t
        elif (t - self.hL) / PS.H_M > PS.RESCALE_EXP:
            f = 2.0 ** (-(t - self.hL) / PS.H_M)
            for v in self.hist.values():
                for i in range(len(v)):
                    v[i] *= f
            self.hL = t
        h = self.hist.get(key)
        if h is None:
            h = self.hist[key] = [0.0] * N_BINS
        h[b] += m * 2.0 ** ((t - self.hL) / PS.H_M)

    # ------------------------------------------------------- retirement
    def set_calendar(self, cls: Mapping[int, int], norm: Mapping[int, bool], today: int) -> None:
        """Day classes and normal-day flags (P01's model.pcal), bounded to CAL_DAYS."""
        self.cal_cls.update({int(d): int(c) for d, c in cls.items()})
        self.cal_norm.update({int(d): bool(v) for d, v in norm.items()})
        self.today = int(today)
        lo = int(today) - CAL_DAYS
        for dct in (self.cal_cls, self.cal_norm):
            for d in [d for d in dct if d < lo]:
                del dct[d]

    def _edge_day(self, key: Tuple[str, int, int], day: Optional[int]) -> None:
        if day is None:
            return
        ed = getattr(self, "edge_days", None)
        if ed is None:
            ed = self.edge_days = {}
        rec = ed.get(key)
        dtc = self.cal_cls.get(int(day), 0)
        if rec is None:
            rec = ed[key] = [int(day), int(day), 0, 0]
            rec[2 + dtc] = 1
        elif int(day) > rec[1]:
            rec[1] = int(day)
            rec[2 + dtc] += 1

    def _drop_edge(self, key: Tuple[str, int, int]) -> None:
        self.hist.pop(key, None)
        self.edge_last.pop(key, None)
        getattr(self, "edge_days", {}).pop(key, None)
        g, a, b = key
        s = self.succ.get((g, a))
        if s is not None:
            s.discard(b)
            if not s:
                self.succ.pop((g, a), None)
        for x in (a, b):
            r = self.by_act.get(x)
            if r is not None:
                r.discard(("e",) + key)

    def retire(self, aid: int) -> None:
        """Purge every statistic of a retired action id."""
        for ref in list(self.by_act.pop(aid, ())):
            if ref[0] == "e":
                key = ref[1:]
                self.edges.discard(key)
                self._drop_edge(key)
            else:
                key = ref[1:]
                (self.prec if ref[0] == "p" else self.loop2).discard(key)
                for x in (key[1], key[2]):
                    if x != aid and x in self.by_act:
                        self.by_act[x].discard(ref)
        for g in list(self.scopes_seen):
            for ss in (self.cnt, self.starts, self.ends, self.out, self.pcnt):
                ss.discard((g, aid))
        self.top_b.discard(aid)
        self.marg = None

    def _ref(self, ids: Iterable[int], ref: Tuple) -> None:
        for x in ids:
            self.by_act.setdefault(x, set()).add(ref)

    # ------------------------------------------------------------ counting
    def add_edge(self, g: str, a: int, b: int, t: float, m: float, ev: float, dbin: Optional[int],
                 day: Optional[int] = None) -> None:
        key = (g, a, b)
        victim = self.edges.add(key, t, m, ev)
        if victim is not None:
            self._drop_edge(victim)
        if key in self.edges:
            self._hist_add(key, dbin, t, m)
            self.edge_last[key] = max(t, self.edge_last.get(key, t))
            self._edge_day(key, day)
            self.succ.setdefault((g, a), set()).add(b)
            self._ref((a, b), ("e",) + key)
        self.out.add((g, a), t, m, ev)

    def add_prec(self, g: str, b: int, a: int, t: float, m: float, ev: float) -> None:
        key = (g, b, a)
        victim = self.prec.add(key, t, m, ev)
        if victim is not None:
            ref = ("p",) + tuple(victim)
            for x in (victim[1], victim[2]):
                r = self.by_act.get(x)
                if r is not None:
                    r.discard(ref)
        if key in self.prec:
            self._ref((a, b), ("p",) + key)

    def add_loop(self, g: str, a: int, b: int, t: float, m: float, ev: float) -> None:
        key = (g, a, b)
        victim = self.loop2.add(key, t, m, ev)
        if victim is not None:
            ref = ("l",) + tuple(victim)
            for x in (victim[1], victim[2]):
                r = self.by_act.get(x)
                if r is not None:
                    r.discard(ref)
        if key in self.loop2:
            self._ref((a, b), ("l",) + key)

    def refresh_top(self, t: float) -> None:
        items = [(k, _ev(self.cnt, k, t)) for k in self.cnt.keys() if k[0] == STAR]
        items.sort(key=lambda x: -x[1])
        self.top_b = {k[1] for k, _ in items[:TOP_B]}
        self.top_t = t

    # -------------------------------------------------------------- size
    def nbytes(self) -> int:
        b = self.acts.nbytes() + self.edges.nbytes() + self.prec.nbytes() + self.loop2.nbytes()
        b += sum(ss.nbytes() for ss in (self.out, self.cnt, self.starts, self.ends, self.pcnt, self.grp_ss))
        b += len(self.hist) * (N_BINS * 8 + 120) + len(self.edge_last) * 100
        b += len(getattr(self, "edge_days", {})) * 150 + (len(self.cal_cls) + len(self.cal_norm)) * 70
        b += sum(56 + 28 * len(v) for v in self.succ.values()) + sum(56 + 72 * len(v) for v in self.by_act.values())
        b += len(self.sessions) * SESSION_BYTES + len(self.burst) * 150
        b += sum(len(v) for v in self.pending.values()) * PENDING_BYTES
        b += len(self.grp_ips) * 200 + 1024
        return int(b)


SESSION_BYTES = 420     # measured: key tuple + 7-slot list + packed hashes + bloom int
PENDING_BYTES = 300


class PFlowModel(dict):
    """model.pflow: the published part is the dict (JSON-able: scopes,
    workflows, edges, requires, acts, gain, stats); the counting state lives
    in the attribute `state` (not a key), so snapshots copy only what views
    and eval read."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.state: FlowState = FlowState()
        self.setdefault("fmt", 1)
        self.setdefault("version", 0)
        self.setdefault("updated", None)
        self.setdefault("last_mine", None)
        self.setdefault("scopes", {})
        self.setdefault("acts", {})
        self.setdefault("gain", {})
        self.setdefault("stats", {})

    def __reduce__(self):  # pickling keeps the state
        return (_rebuild, (dict(self), self.state))


def _rebuild(d: Dict[str, Any], state: FlowState) -> PFlowModel:
    m = PFlowModel(d)
    m.state = state
    return m


# ================================================================ mining
def successors(st: FlowState, g: str, a: int, t: float) -> Dict[int, Tuple[float, float]]:
    """{b: (evidence, mass)} of the tracked edges (g, a, b)."""
    out = {}
    for b in st.succ.get((g, a), ()):
        key = (g, a, b)
        if key in st.edges:
            out[b] = (st.edges.ev(key, t), st.edges.mass(key, t))
    return out


def dependency(c_ab: float, c_ba: float) -> float:
    return (c_ab - c_ba) / (c_ab + c_ba + 1.0)


def edge_stale(c_conf: float, last: Optional[float], t: float, c_m: float = 0.0) -> bool:
    """An edge that stopped: silent for >= max(STALE_MIN_S, 3 / rate) with rate
    = max(c_m ln2 / H_m, c_conf ln2 / H_l) evidence units per second (the
    decayed counts of a young edge under-estimate its rate less on H_m), i.e.
    >= 3 expected occurrences missed and at least a week (weekends and
    holidays never make a workday edge stale). The H_l evidence of a renamed
    or abandoned step would otherwise keep it 'kept' for weeks."""
    if last is None or (c_conf <= 0 and c_m <= 0):
        return False
    quiet = max(0.0, t - last)
    # the rates as they were at the last occurrence (the silence itself must not lower them)
    rate = max(c_m * 2.0 ** (quiet / PS.H_M) * math.log(2.0) / PS.H_M,
               c_conf * 2.0 ** (quiet / PS.H_L) * math.log(2.0) / PS.H_L)
    return quiet >= max(STALE_MIN_S, 3.0 / rate)


def edge_stale_days(rec: Optional[Sequence[int]], cls: Mapping[int, int], norm: Mapping[int, bool],
                    today: Optional[int]) -> Optional[bool]:
    """Normal-day staleness (§6.8.1 applied to a DFG edge): None when there is
    no calendar for the edge (the caller falls back to `edge_stale`).
    c = the day type on which the edge occurred most; p = (its occurrence dates
    + 1/2) / (normal dates of type c spanned by first..last occurrence + 1);
    k = normal dates of type c after the last occurrence (finished days only);
    stale <=> k >= STALE_K_MIN and (1 - p)^k < STALE_P."""
    if rec is None or today is None or not norm:
        return None
    first, last, n_wd, n_nwd = (int(x) for x in rec)
    c = 0 if n_wd >= n_nwd else 1
    n_c = n_wd if c == 0 else n_nwd
    span = sum(1 for d in range(first, last + 1) if norm.get(d) and cls.get(d) == c)
    span = max(span, n_c)
    p = (n_c + 0.5) / (span + 1.0)
    k = sum(1 for d in range(last + 1, int(today)) if norm.get(d) and cls.get(d) == c)
    if not any(first <= d < int(today) for d in norm):
        return None
    return bool(k >= STALE_K_MIN and (1.0 - p) ** k < STALE_P)


def loop2_measure(c_aba: float, c_bab: float) -> float:
    """Length-two loop measure of the Flexible Heuristics Miner (Weijters &
    Ribeiro 2011): (|a b a| + |b a b|) / (|a b a| + |b a b| + 1)."""
    return (c_aba + c_bab) / (c_aba + c_bab + 1.0)


def mine_scope(st: FlowState, g: str, t: float) -> Dict[str, Any]:
    """Kept edges, workflows and required predecessors of one scope."""
    edges: List[Dict[str, Any]] = []
    kept: Dict[int, List[Tuple[int, float]]] = {}
    indeg: Dict[int, int] = {}
    stale = 0
    for key in st.edges.keys():
        if key[0] != g:
            continue
        _, a, b = key
        if a == b:
            continue
        c_ab = st.edges.ev(key, t)
        if c_ab < EDGE_MIN:
            continue
        sd = edge_stale_days(getattr(st, "edge_days", {}).get(key), st.cal_cls if hasattr(st, "cal_cls") else {},
                             getattr(st, "cal_norm", {}), getattr(st, "today", None))
        if sd is None:
            sd = edge_stale(c_ab, st.edge_last.get(key), t, st.edges.ss.evidence(key, t, PS.EV_M))
        if sd:
            stale += 1
            continue
        c_ba = st.edges.ev((g, b, a), t)
        dep_d = dependency(c_ab, c_ba)
        l2 = loop2_measure(st.loop2.ev((g, a, b), t), st.loop2.ev((g, b, a), t))
        loop = l2 >= L2_MIN
        dep = max(dep_d, l2) if loop else dep_d
        m_out = st.out.count((g, a), t) if (g, a) in st.out else 0.0
        share = st.edges.mass(key, t) / m_out if m_out > 0 else 0.0
        if dep < DEP_MIN or share < SHARE_MIN:
            continue
        c_out = _ev(st.out, (g, a), t)
        conf = min(dep, pmdl.jeffreys_lower(min(c_ab, c_out), max(c_out, c_ab)))
        edges.append({"a": a, "b": b, "count": round(c_ab, 2), "rev": round(c_ba, 2),
                      "dep": round(dep, 4), "dep_direct": round(dep_d, 4),
                      "loop2": round(l2, 4) if loop else None, "share": round(share, 4),
                      "band": band(st.hist.get(key) or []), "confidence": round(conf, 4)})
        kept.setdefault(a, []).append((b, c_ab))
        indeg[b] = indeg.get(b, 0) + 1
    # workflows: maximal simple paths from start actions along kept edges
    starts = []
    for a in kept:
        c_a = _ev(st.cnt, (g, a), t)
        s_a = _ev(st.starts, (g, a), t) / c_a if c_a > 0 else 0.0
        if s_a >= START_MIN or indeg.get(a, 0) == 0:
            starts.append((a, s_a))
    starts.sort(key=lambda x: (-x[1], x[0]))
    ecount = {(e["a"], e["b"]): e for e in edges}
    flows: List[Dict[str, Any]] = []

    def dfs(path: List[int]) -> None:
        if len(flows) >= WF_MAX:
            return
        a = path[-1]
        nxt = [b for b, _ in sorted(kept.get(a, ()), key=lambda x: -x[1]) if b not in path]
        if len(path) >= PATH_MAX or not nxt:
            if len(path) >= 2:
                es = [ecount[(x, y)] for x, y in zip(path, path[1:])]
                flows.append({"path": list(path), "support": min(e["count"] for e in es),
                              "bands": [e["band"] for e in es],
                              "confidence": min(e["confidence"] for e in es)})
            return
        for b in nxt:
            dfs(path + [b])

    for a, s_a in starts:
        dfs([a])
    # required predecessors
    reqs: List[Dict[str, Any]] = []
    per_b: Dict[int, List[Dict[str, Any]]] = {}
    for key in st.prec.keys():
        if key[0] != g:
            continue
        _, b, a = key
        if a == b:
            continue
        c_b = _ev(st.pcnt, (g, b), t)
        if c_b < REQ_MIN_B:
            continue
        c_with = min(st.prec.ev(key, t), c_b)
        lb = pmdl.beta_quantile(0.05, 0.5 + c_with, 0.5 + max(0.0, c_b - c_with))
        if lb >= REQ_LB:
            per_b.setdefault(b, []).append({"b": b, "a": a, "c_b": round(c_b, 2),
                                            "c_with": round(c_with, 2), "lb": round(lb, 4)})
    for b, lst in per_b.items():
        lst.sort(key=lambda r: -r["lb"])
        reqs.extend(lst[:REQ_MAX])
    n_sess = _ss_total_ev(st.starts, g, t)
    return {"edges": edges, "workflows": flows, "requires": reqs, "sessions": round(n_sess, 2),
            "stale_edges": stale}


def _ss_total_ev(ss: PS.DecayedSpaceSaving, g: str, t: float) -> float:
    return sum(_ev(ss, k, t) for k in ss.keys() if k[0] == g)


def active_scopes(st: FlowState, t: float) -> List[str]:
    """'*' plus the groups with >= 2 IPs seen here among the SCOPES_MAX heaviest."""
    out = [STAR]
    for g, *_ in st.grp_ss.items(t):
        ips = st.grp_ips.peek(g)
        if ips is not None and len(ips) >= 2:
            out.append(str(g))
    return out


# ================================================================ scoring
def _marginal(st: FlowState, t: float) -> Tuple[float, np.ndarray, List[int], Dict[int, int]]:
    """'*' action marginal (unseen U, sorted probs, ids, id -> rank), cached
    until the counts change (P10 invalidates it after each counting pass and
    on retirement), so scoring many events of a tick costs O(1) each."""
    if st.marg is not None:
        return st.marg[1]  # type: ignore[return-value]
    keys = [k for k in st.cnt.keys() if k[0] == STAR]
    ev = np.asarray([_ev(st.cnt, k, t) for k in keys], dtype=np.float64)
    U = st.acts.unseen(t)
    tot = ev.sum()
    p = (1.0 - U) * ev / tot if tot > 0 else np.zeros(len(keys))
    ids = [k[1] for k in keys]
    order = np.argsort(p)
    ps = p[order]
    ids_s = [ids[i] for i in order]
    rank = {a: i for i, a in enumerate(ids_s)}
    res = (float(U), ps, ids_s, rank)
    st.marg = (t, res)  # type: ignore[assignment]
    return res


def p_marginal(st: FlowState, b: Optional[int], t: float) -> float:
    U, ps, ids, rank = _marginal(st, t)
    if b is None or b not in rank:
        return max(U, 1e-12)
    return max(float(ps[rank[b]]), 1e-12)


def trans_dist(st: FlowState, g: str, a: int, t: float) -> Tuple[Dict[int, float], float]:
    """Hierarchical successor predictive of action a (§6.14):
        p_*(b) = (s_*(b) n_* + alpha p_marg(b)) / (n_* + alpha)
        p_g(b) = (s_g(b) n_g + alpha p_*(b))    / (n_g + alpha)
    s = mass share among a's successors, n = their evidence (PPC-9).
    Returns ({b: p} for every successor at either level, w0) where every other
    action x has p(x) = w0 p_marg(x) (and a novel action w0 U)."""
    succ = successors(st, STAR, a, t)
    n_s = sum(e for e, _ in succ.values())
    m_s = sum(m for _, m in succ.values())
    w_s = ALPHA / (n_s + ALPHA)
    dist = {b: (m / m_s if m_s > 0 else 0.0) * n_s / (n_s + ALPHA) + w_s * p_marginal(st, b, t)
            for b, (e, m) in succ.items()}
    w0 = w_s
    if g != STAR:
        sg = successors(st, g, a, t)
        n_g = sum(e for e, _ in sg.values())
        m_g = sum(m for _, m in sg.values())
        w_g = ALPHA / (n_g + ALPHA)
        new = {b: w_g * p for b, p in dist.items()}
        for b, (e, m) in sg.items():
            base = new.get(b, w_g * w_s * p_marginal(st, b, t))
            new[b] = base + (m / m_g if m_g > 0 else 0.0) * n_g / (n_g + ALPHA)
        dist, w0 = new, w_g * w_s
    return dist, w0


def p_next(st: FlowState, a: int, b: int, t: float) -> float:
    """p_*(b | a) of trans_dist at scope '*' in O(1): the successor totals of a
    are read from the `out` sketch (equal to the sum over a's tracked edges
    while none was evicted) instead of iterating a's successors. Used for the
    prequential gain on every counted row."""
    key = (STAR, a)
    n_s = _ev(st.out, key, t)
    m_s = st.out.count(key, t) if key in st.out else 0.0
    e = (STAR, a, b)
    m_ab = st.edges.mass(e, t) if e in st.edges else 0.0
    share = min(1.0, m_ab / m_s) if m_s > 0 else 0.0
    return share * n_s / (n_s + ALPHA) + ALPHA / (n_s + ALPHA) * p_marginal(st, b, t)


def p_trans(st: FlowState, g: str, a: Optional[int], b: Optional[int], t: float,
            n_min: float = 20.0) -> float:
    """HDR p of b among a's successors: the probability of every outcome no
    more probable than b (ties count half). NaN without a predecessor or with
    < n_min evidence units of a's transitions (the model cannot speak)."""
    if a is None:
        return float("nan")
    if sum(e for e, _ in successors(st, STAR, a, t).values()) < n_min:
        return float("nan")
    dist, w0 = trans_dist(st, g, a, t)
    U, ps, ids, rank = _marginal(st, t)
    pb = dist.get(b) if b is not None else None
    if pb is None:
        pb = w0 * (p_marginal(st, b, t) if b is not None and b in rank else U)
    s = sum(p for p in dist.values() if p < pb) + 0.5 * sum(p for p in dist.values() if p == pb)
    thr = pb / w0 if w0 > 0 else 0.0
    k = int(np.searchsorted(ps, thr, side="left"))
    rest = float(ps[:k].sum()) - sum(float(ps[rank[x]]) for x in dist if x in rank and rank[x] < k)
    rest += U if U < thr else 0.0
    return float(min(1.0, s + w0 * max(0.0, rest)))


def p_req(st: FlowState, scope: Mapping[str, Any], acts: ActionDict, b: Optional[int], bits: int,
          t: float, g: str = STAR) -> Tuple[float, List[int]]:
    """(min over required predecessors a of b absent from the session of
    (c(b without a) + 0.5) / (c(b) + 1), [missing a]); NaN when b has none."""
    if b is None:
        return float("nan"), []
    best, missing = float("nan"), []
    for r in scope.get("requires") or ():
        if int(r["b"]) != b:
            continue
        a = int(r["a"])
        k = acts.key_of(a)
        if k is None or bloom_has(bits, h64(k)):
            continue
        c_b = _ev(st.pcnt, (g, b), t)
        c_with = min(st.prec.ev((g, b, a), t), c_b)
        p = (c_b - c_with + 0.5) / (c_b + 1.0)
        missing.append(a)
        best = p if not best == best else min(best, p)
    if missing and bits == 0:
        # b OPENS the session, so its required predecessor is missing because
        # nothing came before it: the same event is also a transition from the
        # session start (the start pseudo-action of the heuristics miner),
        # p_start = (starts(b) + 1/2) / (sessions + 1). Of the two predictive
        # tests the one with the larger reference class is used (chosen from
        # the past counts, before looking at this event, so the p-value stays
        # valid without a multiplicity charge): the KT floor 1/2 / (n + 1)
        # falls with n. Pack O, A5 (a report generated without its form page):
        # c(b) ~ 20 report sessions -> p_req >= 0.024 can never pass P03's
        # SEQ_P = 0.02, while the report action had opened none of ~150 GA sessions.
        n_sess = _ss_total_ev(st.starts, g, t)
        if n_sess > _ev(st.pcnt, (g, b), t):
            best = float((_ev(st.starts, (g, b), t) + 0.5) / (n_sess + 1.0))
    return best, missing


def seq_scores(model: Mapping[str, Any], g: str, a_key: Optional[str], b_key: str, bits: int,
               t: float) -> Dict[str, Any]:
    """P03's sequence scoring of one event: {'p_trans', 'p_req', 'p_seq',
    'missing', 'a', 'b'}; scope g falls back to '*' when not mined."""
    st: FlowState = getattr(model, "state", None)
    nan = float("nan")
    if st is None:
        return {"p_trans": nan, "p_req": nan, "p_seq": nan, "missing": []}
    scopes = model.get("scopes") or {}
    if g not in scopes:
        g = STAR
    a = st.acts.id_of(a_key) if a_key else None
    b = st.acts.id_of(b_key)
    pt = p_trans(st, g, a, b, t)
    pr, miss = p_req(st, scopes.get(g) or {}, st.acts, b, bits, t, g)
    ps = [p for p in (pt, pr) if p == p]
    return {"p_trans": pt, "p_req": pr, "p_seq": float(min(1.0, 2.0 * min(ps))) if ps else nan,
            "missing": miss, "a": a, "b": b}


def lookup_scope(model: Mapping[str, Any], g: str = STAR) -> Dict[str, Any]:
    """The mined scope (edges, workflows, requires) — P13 / P14 accessor."""
    return dict(((model or {}).get("scopes") or {}).get(g) or {})
