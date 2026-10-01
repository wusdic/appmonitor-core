"""P03 ConformityEngine (`behavior.conformity`) — every event scored against the
most specific confident pattern (docs/lib3/progressive.md §6.16, card P03).
Library 3 (behaviour).

Requirement S14 ("如果访问财务系统去审批就是异常") and the anomaly examples of
§11.4: a learned pattern is only useful if a departure from it is reported
with its type, in natural units, at a controlled false-alarm rate. P03 runs
BEFORE P04 in the tick on every event of the tick (not only the learning
sample) against the tree and the fitted constraints as they were at the end of
the previous tick (prequential: an event is judged before anything learns
from it; P03 never writes a model).

Per event (txn events of evt.batch; win events of evt.win for content only):
  route     path = tree.route(e); conf = deepest node in {confirmed, stable,
            evolving, stale}; the IP's exception node of conf for its targets.
  who       deepest node on path[: conf] whose who summary is CLOSED (U <= 0.05,
            heavy set <= 8 items covering 95 %, >= 5 active days): 1 inside the
            heavy set, else U (Good-Turing probability of an unseen member); at
            an ANCESTOR of the covering node (back-off) a source that recurs
            there (>= 3 evidence units) is a member of its mixed population;
            flags outsider_group (the IP's P11 group has no mass there),
            unknown_ip (no group, no history: not at the root, not in P11's
            signatures, never active here), system_new (its group has no mass in
            the whole system), readdress_candidate / concurrent_use (bindings).
  when      HDR p of the local 15-min slot under the node's day-type density
            (evidence on the confidence channel, backs off to the parent);
            flag outside_windows from P09.
  content   per fitted attribute of the nearest node holding a record with
            support: numeric (P06 p_value: band / GPD tail / conformal rank),
            text / set / closed sets (P07 check_*), bindings (P08
            check_forward / check_reverse with the concurrency test on the
            system's last-activity LRU), broken node invariants
            ((0.5)/(n+1)), per-IP intensity (hourly guaranteed count of the
            (IP, node) pair against P06's `rate.ip_h` band, upper tail only,
            and only above N_hour / k_int so sketch error never alarms).
            p_content = min(1, m min_a p_a).
  seq       P10's p_trans / p_req from the session's previous action and
            Bloom set (P10's session LRU as of t-1 plus this tick's own
            events in time order).
  novel     the action is not in P10's dictionary (its unseen mass U, once the
            dictionary holds >= 20 evidence units); without P10, the event fell
            into the `other` branch of a confident route split whose other child
            is not confident and its route is not tracked there (U of the route).
  anchors   p_t = min(1, 2 min(p_cur, p_ref)) against the node's daily
            reference snapshot (P04); for an attribute whose change is
            `evolving` and coordinated (>= max(2, ceil(|who top| / 2)) IPs) p_t =
            max(p_cur, p_ref) and findings on it are capped at LOW.
  calibrate when, content and seq model p-values are calibrated per (covering
            node, type) against the node's own H_m-decayed histogram of past
            scores (CalStore): p = max(p_model, (W(>= s) + p_model) / (W + 1)),
            s = -log10 p_model. Where the node never produced such a score the
            model's tail probability stands (a first 03:05 login keeps its HDR
            p); where the model is misspecified (a coarse quantile grid, a
            bounded GPD tail, an evening slot the density underweights) the
            node's exceedance frequency replaces it. who and novel are already
            probabilities of an unseen event and are not calibrated; the hourly
            intensity statistic bypasses calibration and the per-day rule.
  p_ev = min(1, 5 min_t p_t); vtype bitmask (p_t <= 1e-3); damp = 0.1 when
            p_ev <= 1e-4 at a confirmed / stable node (outlier damping §6.9.3),
            and for a who violation by a source of another group or an unknown
            one (not a readdress candidate): §6.9.2's rule that persistence alone
            never makes a foreign source a member of a who-closed pattern.
Outputs
  pat.assign   row-aligned with the txn batch: leaf, conf, act (P10's action
               key), prev_act, act_node, exc, p_who, p_when, p_content, p_seq,
               p_novel, p_ev, vtype, damp, think (s), readdr (sparse: the bound
               source a readdress candidate replaces), flags (sparse).
  pat.rate     at each local hour close: the finished guaranteed (IP, node)
               counts {'rows': [(kind, nid, ip, count)], 'N', 'untracked'}.
  scores       per (s, ip) with events: p_tick,t = 1 - (1 - min p_t)^n_t (Sidak)
               for t in who, when, content, seq, novel -> detector columns
               conf_<t> of behavior.score (-log10 p) / behavior.pm when the
               detector registry has them (W-P9 appends them, §9.2), and always
               the dict series `behavior.conf` {conf_<t>: {score, p, n, axes}}.
  events       `pattern_violation` for events whose covering node is confirmed
               or stable, typed and severity-ranked by the §6.16.3 table on the
               per-day multiplicity p_day = 1 - (1 - p_min)^n_day of the (IP,
               node, type) cell; dedupe `pv|<node>|<type>|<ip>|<local date>`;
               at most V_MAX = 20 per system and tick (highest first).
Memory  per system: the hourly intensity sketch (k_int = 16 384 pairs), the
        per-day (IP, node, type) cells (LRU 65 536), the last-activity LRU
        (65 536); nothing else is kept per IP. Per event O(depth + checked
        attributes); node-level work (closedness, fitted lookups) is cached
        per node and batch.
Inert unless config['progressive']['enabled'].

Deviations (documented; each forced by a measured failure on pack O, see the
implementation report):
  * who closedness (_who_closed): the heavy set is the union of the mass-heavy
    and the evidence-heavy sets, and a level whose members include grp:∅,
    reg:∅ or * is not a who constraint ("only ungrouped sources" is not a
    population). A 60-s monitor holds most of a node's mass with little
    evidence; a mass-only heavy set excluded the finance approver.
  * calibrated p-values (above) instead of the raw model p for emissions:
    the raw p gave 625 findings on 8 clean days of pack O.
  * `auth route` (cross_binding MEDIUM row) is read as "a write action whose
    node carries a binding": the POST-then-session test needs the next events.
  * a missing required predecessor is emitted at p_day(p_req) <= 0.02; p_req
    is a posterior predictive of a rule that already passed the Jeffreys
    5 % >= 0.85 test, so its p_day would never reach 1e-3.
  * content: each attribute at the deepest fitted node holding it on
    (exception node, covering node, ancestors), the root's system-wide
    records only when the covering node is the root. (Checking only the
    deepest fitted node lost the login's username grammar held one level up:
    A3's injection went unreported on pack O seed 0.)
  * only credential-grade binding evidence has its own severity (cross_binding
    at LB >= 0.9 on a write / sensitive action; concurrent_use); an unbound
    value, a value outside the set or a foreign source is judged by the
    per-day content rule (p_day <= 1e-3) on the credential axis.
  * intensity: the tail rank uses the rate digest's IP-hours (its mass) as n,
    and the hour count is a finding only at p <= 1e-3; the node's intensity
    routine is kept on the count's magnitude (band90 top / count, one unit
    per tail hour at its extreme), since beyond the digest's maximum every
    count has the same rank p; the hour count and the event's content p make
    one content finding, not two.
  * node routine (E_MAX = 1 a day): a when, content, missing-predecessor or
    intensity finding is not emitted where the covering node itself produced
    scores at least that extreme >= once a day over its H_m-decayed history
    (CalStore.per_day; damped outliers count 0.1). A calibrated p-value bounds
    the per-EVENT rate; on a node with 10^4 events a day (a public portal at
    night, a developer's TLS hours) that is still tens of findings a day, all
    part of the node's routine (pack O: 63 portal night visits, 170 intensity
    hours of 370 clean findings on seed 0 without it).
  * who: a source that is not a member at the closed level but whose P11
    group is a member of the node's closed group level is a member (a new
    address of a colleague, §6.9.2).
  * win-kind events are scored for content only and get no pat.assign rows
    (the store has one pat.assign series per system, aligned with evt.batch).
  * the content-attribute test of P10's action variants is duplicated here
    (a shared helper belongs in lib/pdfg, W-P5 / open issue).
  * novel (groups_views round, ActionLedger): an action P10's dictionary does
    not hold is NOT new when it was performed on an earlier local date by >= 2
    distinct sources (a recurring weekly / monthly action - P10's decayed
    dictionary never held SALES' Friday report, flagged MEDIUM for all 20
    members on both Fridays, the largest FAR source); and a new action that
    another member of the source's P11 group performed the same day is a new
    behaviour of the group, capped at LOW (flags recurring_action /
    group_action).
  * who: a source whose group never used the SYSTEM (system_new) and is an
    outsider at a node closed with U <= 0.02 is MEDIUM whatever the node's
    sensitivity (a lateral move into a closed system; the HIGH row's
    `lateral` axis). A9 (sales IP reading finance's approval list) was LOW:
    no incident ever opened.
  * who: a new address inside a configured DHCP pool (dhcp_scopes) whose pool
    (reg level) has standing at the node is a re-addressed member.
  * intensity: the hour count's p is its conformal rank among every IP-hour
    P03 closed at the node (RateTally, binned histogram, H_L decay), not
    P06's rate.ip_h p - that digest is H_m-decayed (rank floor ~1/(mass+1)
    = 2.5e-3: A7's 400 logins an hour was LOW) and its GPD tail, fitted to
    integer counts, put counts of 2-3 at the 1e-9 floor (~40 clean
    intensity findings per run). rate.ip_h's band still gates the test
    (count > band90 top) and is what the finding states as expected.
"""
from __future__ import annotations

import bisect
import datetime as _dt
import heapq
import math
import re
import time
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import ORG, SYSTEM_ENTITY, BehaviorEvent, Severity
from .lib import m_ptree as MP
from .lib import pbounds as PB
from .lib import pdfg as DF
from .lib import pevent as EV
from .lib import pfd as FD
from .lib import pgrammar as PG
from .lib import pmdl
from .lib import pnode as PN
from .lib import psketch as PS
from .lib import pscore as SC
from .lib import pwindows as PW
from .lib.phier import GRP_NONE, REG_NONE, STAR as _STAR

def _route_key(get: Callable[[str], Any]) -> Optional[str]:
    """lib/pdfg.route_key with absent attributes read as None (pdfg tests
    `isinstance(v, str)`, and ABSENT is the string '⊥')."""
    return DF.route_key(lambda a: (lambda v: None if v is EV.ABSENT else v)(get(a)))


TYPES = ("who", "when", "content", "seq", "novel")
DETS = {t: f"conf_{t}" for t in TYPES}
CONF_SERIES = "behavior.conf"
STATE = "model.pconf_state"
P_FLOOR = 1e-9                  # a finite sample never supports p = 0 (bounded GPD tails)
GRP_LEVEL = 3                   # the who level of P11's groups (grp:<id>)
REG_LEVEL = 4                   # the who level of configured / learned regions (reg:<name>)
GROUP_MIN_MEMBERS = 2           # other members with standing that make a node a pattern of their group
MEMBER_EV = 3.0                 # a source recurring at an ANCESTOR population (back-off) is a member
CAL_TYPES = ("when", "content", "seq")
INT_P = 1e-3                    # an hour count is a finding only at this tail (§6.16.4)
E_MAX = 1.0                     # a node's routine: >= this many such scores per day
SEQ_P = 0.02                    # per-day p of a missing required predecessor to emit
CAL_P = 0.05                    # p-values above this are reported uncalibrated (never an emission)
NONE_ITEMS = frozenset({str(GRP_NONE), str(REG_NONE), str(_STAR)})
WG_STATE = "model.who_groups_state"
NAN = float("nan")
N_MIN = 20.0
DAMP = 0.1
DAMP_P = 1e-4
VTYPE_P = 1e-3
V_MAX = 20
K_INT = 16_384
CELLS = 65_536
LAST_ACTIVE = 65_536
DAY = 86400.0
HOUR = 3600.0
WRITE = frozenset({"POST", "PUT", "PATCH", "DELETE"})
ROUTE_ATTRS = ("http.route", "http.path")
SKIP_PREFIX = ("ctx.", "ev.", "net.src", "net.peer_src", "sess.", "http.route", "http.path",
               "http.host", "rate.")
RATE_ATTR = "rate.ip_h"
LB_CROSS = 0.9
SEV_RANK = {Severity.INFO: 0, Severity.LOW: 1, Severity.MEDIUM: 2, Severity.HIGH: 3,
            Severity.CRITICAL: 4}
# P10's content-attribute test for action variants (duplicated: see module docstring)
NON_CONTENT_KINDS = frozenset({"ip", "dst", "route", "path", "tod", "when"})
NON_CONTENT_PREFIX = ("ctx.", "ev.", "sess.", "net.src", "net.peer", "net.dst", "client.",
                      "http.method", "http.host", "http.route", "http.path", "tls.sni", "dns.qname")


def group_members_at(nd: Any, g: str, ip: str, t: float, ip2g: Mapping[str, Any]) -> int:
    """Number of OTHER members of P11 group g that have standing at the node
    (each brought >= MEMBER_EV evidence units to its IP-level who summary).
    Bounded: one pass over the node's level-0 heavy hitters."""
    lv0 = nd.who.levels[0]
    n = 0
    for x, _c, _gu, ev in lv0.items(t):
        if str(x) != str(ip) and ip2g.get(str(x)) == g and ev >= MEMBER_EV:
            n += 1
    return n


def group_outsider(nd: Any, g: str, ip: str, t: float,
                   ip2g: Optional[Mapping[str, Any]] = None) -> bool:
    """The IP's P11 group g has no standing at the node: the group-level
    evidence its OTHER members brought is below MEMBER_EV. Presence of the
    group key alone is not standing - the IP's own earlier (damped) events
    put it there, so a foreign source that persisted became an "insider" after
    its first visit and every later event was learned undamped (pack O, A9:
    the sales address 192.168.3.33 reading finance's approval list daily from
    day 11 was in the node's heavy set by day 21, against §6.9.2 "persistence
    alone never makes a foreign source a member")."""
    lv3 = nd.who.levels[3]
    if lv3.total(t) <= 0:
        return False
    if ip2g is not None:
        # the node is a pattern OF THE GROUP only when >= GROUP_MIN_MEMBERS other
        # members use it; one colleague's individual habit (the finance approver,
        # whom P11 placed in 综合部's group on days 15-17 of pack O seed 0) gives the
        # rest of the group no standing there (A1: 192.168.1.23 approving in
        # finance was scored as a colleague of 192.168.2.10 -> LOW, no incident)
        return group_members_at(nd, g, ip, t, ip2g) < GROUP_MIN_MEMBERS
    key = f"grp:{g}"
    ev_g = lv3.evidence(key, t) if key in lv3 else 0.0
    lv0 = nd.who.levels[0]
    ev_ip = lv0.evidence(ip, t) if ip in lv0 else 0.0
    return ev_g - ev_ip < MEMBER_EV


def _nan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _is_content(attr: str, hier: Any) -> bool:
    if attr.startswith(NON_CONTENT_PREFIX) or attr.startswith("@"):
        return False
    try:
        return hier.kind(attr) not in NON_CONTENT_KINDS
    except Exception:
        return True


# ================================================================ sketches
class HourSS:
    """Space-Saving over (ip, kind, node) pairs for one local hour (§6.16.4):
    exact while fewer than k pairs exist; beyond, count - err is a guaranteed
    lower bound and every pair above N / k is tracked. Min-replacement through
    a lazy heap (entries are re-pushed on update; stale ones are skipped)."""

    __slots__ = ("k", "c", "e", "heap", "N", "hour", "tail")    # tail: pair -> hour's extreme

    def __init__(self, k: int = K_INT, hour: int = -1) -> None:
        self.k = int(k)
        self.c: Dict[Hashable, float] = {}
        self.e: Dict[Hashable, float] = {}
        self.heap: List[Tuple[float, int, Hashable]] = []
        self.N = 0.0
        self.hour = int(hour)
        self.tail: Dict[Hashable, float] = {}     # pairs whose hour count reached the tail

    def add(self, key: Hashable, w: float = 1.0) -> float:
        self.N += w
        c = self.c.get(key)
        if c is not None:
            self.c[key] = c + w
            heapq.heappush(self.heap, (c + w, id(key) & 0xFFFF, key))
        elif len(self.c) < self.k:
            self.c[key] = w
            self.e[key] = 0.0
            heapq.heappush(self.heap, (w, id(key) & 0xFFFF, key))
        else:
            while True:
                m, _, kk = heapq.heappop(self.heap)
                if self.c.get(kk) == m:
                    break
            del self.c[kk]
            del self.e[kk]
            self.c[key] = m + w
            self.e[key] = m
            heapq.heappush(self.heap, (m + w, id(key) & 0xFFFF, key))
        if len(self.heap) > 8 * self.k:
            self.heap = [(v, id(k) & 0xFFFF, k) for k, v in self.c.items()]
            heapq.heapify(self.heap)
        return self.c[key]

    def guaranteed(self, key: Hashable) -> float:
        c = self.c.get(key)
        return 0.0 if c is None else c - self.e.get(key, 0.0)

    def rows(self) -> List[Tuple[Hashable, float]]:
        return [(k, self.c[k] - self.e[k]) for k in self.c]

    def nbytes(self) -> int:
        return int(len(self.c) * 200 + len(self.heap) * 80 + 128)


class CalStore:
    """Calibration of the model p-values (when, content, seq) per (tree key,
    kind, covering node, type): an H_m-decayed histogram of the scores
    s = -log10 p_model of the events scored there (prequential: updated after
    the event is scored, with the event's learning weight trust x damp).
        p_cal = (W(>= s) + alpha p_model) / (W + alpha),  p = max(p_model, p_cal)
    The model's tail probability is kept wherever the node never produced such
    scores (a first 03:05 login keeps its HDR p), and replaced by the node's own
    exceedance frequency where the model is misspecified (a coarse digest, a
    bounded GPD tail, an evening slot the density underweights): calibrated
    p-values whose false-alarm rate follows the data, not the model."""

    NB = 48
    STEP = 0.25
    ALPHA = 1.0

    def __init__(self, cap: int = 32768) -> None:
        self.h = PS.LRU(cap)
        self.L: Optional[float] = None

    def _g(self, t: float) -> float:
        if self.L is None:
            self.L = float(t)
        e = (float(t) - self.L) / PS.H_M
        if e > 60.0:
            f = np.float32(2.0 ** -e)
            for _, a in self.h._d.items():
                a *= f
            self.L = float(t)
            e = 0.0
        return 2.0 ** e

    def _bin(self, p: float) -> Tuple[int, float]:
        sc = -math.log10(max(float(p), 1e-12))
        x = min(sc / self.STEP, self.NB - 1e-9)
        b = int(x)
        return b, x - b

    def p(self, key: Hashable, p_model: float, t: float) -> float:
        if _nan(p_model):
            return p_model
        a = self.h.peek(key)
        if a is None:
            return p_model
        g = self._g(t)
        W = float(a[self.NB]) / g
        if W <= 0:
            return p_model
        b, frac = self._bin(p_model)
        w_ge = (float(a[b + 1:self.NB].sum()) + (1.0 - frac) * float(a[b])) / g
        return float(max(p_model, (w_ge + self.ALPHA * p_model) / (W + self.ALPHA)))

    def add(self, key: Hashable, p_model: float, t: float, w: float = 1.0) -> None:
        if _nan(p_model) or w <= 0:
            return
        a = self.h.get(key)
        if a is None:
            a = np.zeros(self.NB + 2, dtype=np.float32)
            a[self.NB + 1] = float(t) / DAY          # first score (days; float32 is exact to ~1e-3 d)
            self.h.put(key, a)
        g = self._g(t) * w
        b, _ = self._bin(p_model)
        a[b] += g
        a[self.NB] += g

    def move(self, key: Hashable, p_old: float, p_new: float, t: float) -> None:
        """Re-bin one unit scored at p_old as p_new (an hour count that grew
        more extreme within its hour; the hour is one unit, at its extreme)."""
        a = self.h.peek(key)
        if a is None:
            return
        bo, bn = self._bin(p_old)[0], self._bin(p_new)[0]
        if bo == bn:
            return
        g = self._g(t)
        a[bo] = max(0.0, float(a[bo]) - g)
        a[bn] += g

    def per_day(self, key: Hashable, p_model: float, t: float) -> float:
        """Expected number of events per day at this key scoring at least as
        extreme as p_model: the H_m-decayed weight W(>= s) turned into a daily
        rate (W = rate H / ln 2 (1 - 2^(-age / H)) under forward decay). 0 where
        the key never produced such a score."""
        if _nan(p_model):
            return 0.0
        a = self.h.peek(key)
        if a is None:
            return 0.0
        g = self._g(t)
        b, frac = self._bin(p_model)
        w_ge = (float(a[b + 1:self.NB].sum()) + (1.0 - frac) * float(a[b])) / g
        if w_ge <= 0:
            return 0.0
        age = max(float(t) / DAY - float(a[self.NB + 1]), 1.0 / 24.0) if len(a) > self.NB + 1 else 1e9
        hd = PS.H_M / DAY
        return float(w_ge * math.log(2.0) / hd / max(1.0 - 2.0 ** (-age / hd), 1e-9))

    def nbytes(self) -> int:
        return int(len(self.h) * (4 * (self.NB + 2) + 200) + 128)


LEDGER_CAP = 16_384             # (system, action) entries remembered (LRU)
LEDGER_SRC = 4                  # distinct sources kept per action
LEDGER_GRP_SRC = 2              # distinct sources kept per group and action for the current day


class ActionLedger:
    """Which actions a system has seen, by how many sources and on which local
    dates - a long memory next to P10's H_m-decayed action dictionary (whose
    k-slot Space-Saving can lose a weekly or monthly action between two of its
    occurrences, and which learns nothing from quarantined sources).

    Per (system, action route): [first local day, last local day, number of
    local dates, up to LEDGER_SRC distinct sources, the day of `grp`, and grp =
    {group: up to LEDGER_GRP_SRC distinct member sources that performed it on
    that day}]. LRU-bounded (LEDGER_CAP entries); O(1) per scored event.

    Two facts P03's `novel` type reads from it (both computed BEFORE the event
    itself is added, prequential):
      established  performed on an EARLIER local date by >= 2 distinct sources:
                   a recurring action of the system (SALES' Friday report on its
                   second Friday), not a new one - one source alone can never
                   establish an action (an attacker repeating its own probe);
      coordinated  another member of the source's P11 group performed this new
                   action today: a new behaviour of the group (§6.9.2's
                   coordinated change), reported at most LOW."""

    __slots__ = ("d",)

    def __init__(self, cap: int = LEDGER_CAP) -> None:
        self.d = PS.LRU(cap)

    def peek(self, s: str, key: str) -> Optional[List[Any]]:
        return self.d.peek((s, key))

    @staticmethod
    def established(e: Optional[List[Any]], day: int) -> bool:
        return e is not None and int(e[0]) < day and len(e[3]) >= 2

    @staticmethod
    def coordinated(e: Optional[List[Any]], g: Optional[str], ip: str, day: int) -> bool:
        if e is None or g is None or int(e[4]) != day:
            return False
        return any(x != ip for x in (e[5].get(g) or ()))

    def add(self, s: str, key: str, ip: str, g: Optional[str], day: int) -> None:
        e = self.d.get((s, key))
        if e is None:
            e = [day, day, 1, [], day, {}]
            self.d.put((s, key), e)
        elif int(e[1]) != day:
            e[1] = day
            e[2] += 1
        if ip not in e[3] and len(e[3]) < LEDGER_SRC:
            e[3].append(ip)
        if int(e[4]) != day:
            e[4], e[5] = day, {}
        if g is not None:
            m = e[5].setdefault(g, [])
            if ip not in m and len(m) < LEDGER_GRP_SRC:
                m.append(ip)

    def nbytes(self) -> int:
        return int(len(self.d) * 360 + 128)


RATE_TALLY_CAP = 8192
RATE_BINS = 48


def _rate_bin(c: float) -> int:
    """Bin of an hour count: exact for 1-4, half-octave above (5-5, 6-7, 8-11, ...)."""
    c = max(1.0, float(c))
    if c < 5.0:
        return int(c) - 1
    return min(RATE_BINS - 1, 4 + int(2.0 * math.log2(c / 4.0)))


class RateTally:
    """Per (tree key, kind, node): the distribution of the finished hour counts
    of every (IP, node) pair P03 has closed there - a binned histogram (exact
    for 1-4, half-octave bins above) with forward decay at H_L (30 d),
    LRU-bounded by nodes, O(1) per pat.rate row.

    p(c) = (1 + W(>= bin(c))) / (W + 1): the conformal rank of an hour count
    among all IP-hours counted at the node, the whole bin of c counted as "at
    least as extreme" (conservative). Valid for count data under
    exchangeability of IP-hours (prequential: only finished hours enter).

    Why not P06's rate.ip_h p: the digest is H_m-decayed (its mass is a few
    hundred IP-hours on a portal login node however long it is watched), so a
    count beyond its maximum had the rank p 1/(mass + 1) ~ 2.5e-3 and 400
    logins in one hour (pack O A7) was never more than LOW; and its GPD tail,
    fitted to integer counts that are almost all 1, put a count of 2-3 at the
    p floor 1e-9 - 40 clean portal / crm visitors were intensity findings
    (seed 0), most of them MEDIUM incidents."""

    __slots__ = ("d", "L")

    def __init__(self, cap: int = RATE_TALLY_CAP) -> None:
        self.d = PS.LRU(cap)
        self.L: Optional[float] = None

    def _f(self, t: float) -> float:
        if self.L is None:
            self.L = float(t)
        e = (float(t) - self.L) / PS.H_L
        if e > 60.0:
            g = np.float64(2.0 ** -e)
            for _, a in self.d._d.items():
                a *= g
            self.L = float(t)
            e = 0.0
        return 2.0 ** e

    def close_hour(self, key: str, rows: Sequence[Tuple[int, int, str, float]], t: float) -> None:
        f = self._f(t)
        for kind, nid, _ip, g in rows:
            k = (key, int(kind), int(nid))
            a = self.d.get(k)
            if a is None:
                a = np.zeros(RATE_BINS, dtype=np.float64)
                self.d.put(k, a)
            a[_rate_bin(g)] += f

    def p(self, key: str, kind: int, nid: int, c: float, t: float) -> float:
        a = self.d.peek((key, int(kind), int(nid)))
        if a is None:
            return 1.0
        f = self._f(t)
        W = float(a.sum()) / f
        w_ge = float(a[_rate_bin(c):].sum()) / f
        return float(min(1.0, (1.0 + w_ge) / (W + 1.0)))

    def nbytes(self) -> int:
        return int(len(self.d) * (8 * RATE_BINS + 160) + 128)


class ConfState:
    """P03's private state (model.pconf_state@(__org__, __org__); P03 writes no
    pattern model)."""

    def __init__(self, cells: int = CELLS, last_active: int = LAST_ACTIVE) -> None:
        self.cal = CalStore()
        self.hour: Dict[str, HourSS] = {}
        self.cells = PS.LRU(cells)
        self.last_active = PS.LRU(last_active)
        self.ledger = ActionLedger()
        self.last_batch: Dict[Tuple[str, str], float] = {}
        self.stats: Dict[str, float] = {"events": 0, "scored": 0, "violations": 0, "suppressed": 0,
                                        "us": 0.0}

    def nbytes(self) -> int:
        return int(sum(h.nbytes() for h in self.hour.values()) + 120 * len(self.cells)
                   + 100 * len(self.last_active) + self.cal.nbytes() + self.led().nbytes() + 1024)

    def tally(self) -> RateTally:
        r = getattr(self, "rate_tally", None)
        if r is None:                           # a state saved before the tally existed
            r = self.rate_tally = RateTally()
        return r

    def led(self) -> ActionLedger:
        lg = getattr(self, "ledger", None)
        if lg is None:                          # a state saved before the ledger existed
            lg = self.ledger = ActionLedger()
        return lg


# ================================================================ node info
class _NodeInfo:
    """Per node, per batch: everything the event checks read."""

    __slots__ = ("nd", "who_l", "who_heavy", "who_U", "who_grp", "who_ref", "evolving", "when_ok", "wins",
                 "content", "binds", "route", "write", "sens", "rate", "refc")


class _TreeCtx:
    def __init__(self, eng: "ConformityEngine", store: Any, s: str, key: str,
                 config: Mapping[str, Any], now: float) -> None:
        self.store = store
        self.s = s
        self.key = key
        self.now = now
        self.config = config
        self.ptm = MP.get_ptree(store, key)
        self.reg = MP.get_registry(store, key)
        self.hier = MP.hierarchies(store, key, config, self.reg)
        self.gone: Set[str] = {n for n, r in self.reg.records.items() if r.state == "gone"} \
            if self.reg is not None else set()
        self.pb = MP.get_model(store, key, MP.PBOUNDS)
        self.pg = MP.get_model(store, key, MP.PGRAMMAR)
        self.pbind = MP.get_model(store, key, MP.PBIND)
        self.pwin = MP.get_model(store, key, MP.PWIN)
        self.pflow = MP.get_model(store, key, MP.PFLOW)
        self.flow = getattr(self.pflow, "state", None)
        wg = MP.who_groups(store)
        self.ip2g = wg.get("ip2g") or {}
        self.groups = wg.get("groups") or {}
        ws = store.get_model(ORG, ORG, WG_STATE)
        self.sigs = getattr(ws, "sigs", None)
        self.tz = PW.tz_offset(config, now)
        self.sens = [re.compile(p) for p in (config.get("sensitive_patterns") or []) if isinstance(p, str)]
        # configured address POOLS (DHCP scopes): reg-level items whose new
        # addresses are re-addressed members, not strangers
        self.pool_regions: Set[str] = {f"reg:{it.get('name', 'dhcp_scopes')}"
                                       for it in (config.get("dhcp_scopes") or []) if isinstance(it, Mapping)}
        pc = EV.pconfig(config)
        self.gap = float(pc["defaults"].get("session_gap_s", 1800.0))
        self.info: Dict[Tuple[int, int], _NodeInfo] = {}
        self.content_nodes: Dict[int, Set[int]] = {}
        self.whokeys: Dict[str, List[Any]] = {}

    def when_table(self, kind: int, nd: Any, daytype: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """HDR p of every 15-min slot under the node's density for a day type
        (pscore.when_p, vectorised once per node and batch) and under the
        reference snapshot's density; None where the model cannot speak."""
        k = ("w", kind, nd.id, daytype)
        hit = self.info.get(k)
        if hit is not None:
            return hit
        t = self.now
        d = 1 if daytype else 0
        N = nd.when.evidence(d, t)
        h = nd.when.hist[d]
        if N < SC.N_MIN:
            N = nd.when.evidence(0, t) + nd.when.evidence(1, t)
            h = nd.when.hist.sum(axis=0)
        cur = _hdr_table(SC.when_density(h, N)) if N >= SC.N_MIN else None
        ref = None
        refw = ((nd.ref or {}).get("when") or {}).get("nwd" if daytype else "wd")
        if refw is not None:
            Nd = nd.when.evidence(d, t)
            if Nd >= SC.N_MIN:
                ref = _hdr_table(SC.when_density(np.asarray(refw, dtype=np.float64) * Nd, Nd))
        self.info[k] = (cur, ref)
        return cur, ref

    def who_keys(self, ip: str) -> List[Any]:
        k = self.whokeys.get(ip)
        if k is None:
            k = self.whokeys[ip] = [self.hier.gen("net.src", l, ip) for l in range(PN.WHO_LEVELS)]
        return k

    def content_split_nodes(self, kind: int, tree: Any) -> Set[int]:
        c = self.content_nodes.get(kind)
        if c is None:
            c = self.content_nodes[kind] = {nid for nid, nd in tree.nodes.items()
                                            if nd.split is not None and _is_content(nd.split.attr, self.hier)}
        return c

    def sensitive(self, route: Optional[str]) -> bool:
        if not route or not self.sens:
            return False
        path = route.split()[-1]
        return any(p.search(path) for p in self.sens)

    # ------------------------------------------------------------ node info
    def node_info(self, kind: int, tree: Any, nid: int) -> _NodeInfo:
        k = (kind, nid)
        ni = self.info.get(k)
        if ni is not None:
            return ni
        t = self.now
        nd = tree.nodes[nid]
        ni = _NodeInfo()
        ni.nd = nd
        ni.refc = {}
        # who closedness
        ni.who_l, ni.who_heavy, ni.who_U = _who_closed(nd, t) if nd.confident else (None, set(), NAN)
        ni.who_grp = None
        if ni.who_l is not None and ni.who_l < GRP_LEVEL:
            r = _who_level(nd, GRP_LEVEL, t)
            ni.who_grp = r[0] if r is not None else None
        ni.who_ref = set((nd.ref or {}).get("who") or []) if nd.ref else None
        # evolving (coordinated) attributes
        ev = nd.meta.get("evolving") or {}
        who_top = len(nd.who.heavy_set(0, t)[0]) or 1
        need = max(2, math.ceil(0.5 * who_top))
        ni.evolving = {a for a, st in ev.items() if len(st.get("ips") or ()) >= need}
        ni.when_ok = None
        went = PW.lookup(self.pwin, kind, nid)
        ni.wins = (went or {}).get("when") if went and went.get("status") == "fitted" else None
        # content records (own fitted entries; walked up the path at event time)
        recs: List[Tuple[str, str, Any]] = []
        for mdl, typ in ((self.pb, "num"), (self.pg, "pg")):
            ent = PB.lookup(mdl, kind, nid)
            if not ent:
                continue
            for a, rec in (ent.get("attrs") or {}).items():
                if not rec or a.startswith(SKIP_PREFIX) or a == RATE_ATTR:
                    continue
                if any(c[0] == a for c in nd.ctx):
                    continue
                if typ == "num":
                    recs.append((a, "num", _NumFast(rec)))
                else:
                    recs.append((a, str(rec.get("kind", "text")), rec))
        ni.content = recs
        bent = PB.lookup(self.pbind, kind, nid) if isinstance(self.pbind, Mapping) else None
        ni.binds = [rec for rec in ((bent or {}).get("pairs") or {}).values() if rec]
        ni.rate = PB.lookup(self.pb, kind, nid, RATE_ATTR)
        ni.route = None
        for a, lv, vals, neg in nd.ctx:
            if a in ROUTE_ATTRS and not neg and len(vals) == 1:
                ni.route = str(next(iter(vals)))
        ni.write = False
        ni.sens = 1.0
        self.info[k] = ni
        return ni


def _who_closed(nd: Any, t: float) -> Tuple[Optional[int], Set[Any], float]:
    """(level, members, U) at the finest closed who level, else (None, set(), NaN).
    Deviation from pnode.WhoSummary.closed_level (documented): the heavy set is
    the UNION of the mass-heavy set (95 % of mass) and the evidence-heavy set
    (95 % of the confidence-channel evidence). A burst-y automation source
    (a monitor polling every 60 s) holds most of a node's mass with little
    evidence, and a mass-only heavy set then excludes the daily users."""
    if nd.n_days() < PN.CLOSED_DAYS:
        return None, set(), NAN
    for l in range(len(nd.who.levels)):
        r = _who_level(nd, l, t)
        if r is not None:
            return l, r[0], r[1]
    return None, set(), NAN


def _who_level(nd: Any, l: int, t: float) -> Optional[Tuple[Set[Any], float]]:
    """(members, U) when who level l of the node is closed, else None."""
    if l >= len(nd.who.levels):
        return None
    ss = nd.who.levels[l]
    if ss.total(t) <= 0:
        return None
    U = float(ss.unseen(t))
    if U > PN.CLOSED_U:
        return None
    hm, cov = nd.who.heavy_set(l, t)
    if cov < PN.HEAVY_COVER - 1e-9:
        return None
    its = sorted(ss.items(t), key=lambda x: -x[3])
    tot = ss.total_evidence(t)
    he, acc = [], 0.0
    for k, c, g, e in its:
        if acc >= PN.HEAVY_COVER * tot:
            break
        he.append(k)
        acc += e
    mem = set(hm) | set(he)
    if any(str(m) in NONE_ITEMS for m in mem):
        return None                      # "only ungrouped / unregioned sources" is not a who constraint
    if len(mem) > PN.HEAVY_MAX:
        return None
    return mem, U


def _hdr_table(f: np.ndarray) -> np.ndarray:
    """p[s] = sum f[f < f_s] + 0.5 sum f[f == f_s] for every slot s (pmdl.hdr_p)."""
    f = np.asarray(f, dtype=np.float64)
    tot = f.sum()
    if tot <= 0:
        return np.ones_like(f)
    f = f / tot
    srt = np.sort(f)
    cs = np.concatenate([[0.0], np.cumsum(srt)])
    lo = np.searchsorted(srt, f, "left")
    hi = np.searchsorted(srt, f, "right")
    return np.minimum(1.0, cs[lo] + 0.5 * (cs[hi] - cs[lo]))


def _ref_record(nd: Any, name: str, attr: str) -> Optional[Mapping[str, Any]]:
    ref = nd.ref or {}
    ent = (ref.get("fitted") or {}).get(name)
    if not isinstance(ent, Mapping):
        return None
    return (ent.get("attrs") or {}).get(attr)


class _NumFast:
    """lib/pbounds.p_value(rec, v) without a live digest, with the record's
    quantile grid prepared once per record and batch (the per-event cost of
    rebuilding the grid dominated P03's time). Same maths; checked against
    p_value in tests/engines/test_p03_conformity.py."""

    __slots__ = ("rec", "lg", "qg", "ql", "ps", "lo", "hi", "vmin", "vmax", "n", "rng", "th", "tl", "ok")

    def __init__(self, rec: Mapping[str, Any]) -> None:
        self.rec = rec
        self.lg = bool(rec.get("log"))
        qg = rec.get("qgrid") or []
        self.qg = np.asarray(qg, dtype=np.float64)
        self.ok = self.qg.size >= 2 and bool(np.all(np.isfinite(self.qg))) and "band98" in rec
        self.ps = np.linspace(0.0, 1.0, self.qg.size) if self.ok else None
        self.ql = self.qg.tolist()
        self.n = float(rec.get("n_c", NAN))
        self.rng = rec.get("range")
        self.th, self.tl = rec.get("tail_hi"), rec.get("tail_lo")
        if self.ok:
            self.lo = PB._fwd(rec["band98"][0], self.lg)
            self.hi = PB._fwd(rec["band98"][1], self.lg)
            self.vmin, self.vmax = float(self.qg[0]), float(self.qg[-1])

    def p(self, v: Any) -> Tuple[float, List[str]]:
        flags: List[str] = []
        try:
            x = float(v)
        except (TypeError, ValueError):
            return NAN, flags
        if not math.isfinite(x):
            return NAN, flags
        rng = self.rng
        if rng is not None:
            if x > rng[1]:
                flags.append("above_range")
            elif x < rng[0]:
                flags.append("below_range")
        nn = self.n
        if not (nn >= PB.N_MIN) or not self.ok:
            return NAN, flags
        y = PB._fwd(x, self.lg)
        if not math.isfinite(y):
            return SC.conformal_rank_p(0, nn), flags
        if self.lo <= y <= self.hi:
            if y <= self.vmin:
                F = 0.0
            elif y >= self.vmax:
                F = 1.0
            else:
                # np.interp over the grid (quantile levels i / (m - 1)), in pure
                # Python: a scalar np.interp call costs more than the search
                ql = self.ql
                j = bisect.bisect_right(ql, y)             # ql[j-1] <= y < ql[j]
                x0, x1 = ql[j - 1], ql[j]
                m1 = len(ql) - 1
                F = ((j - 1) + ((y - x0) / (x1 - x0) if x1 > x0 else 0.0)) / m1
            return SC.numeric_p(F), flags
        if y > self.hi:
            if self.th:
                return SC.tail_p(x, self.th[0], self.th[1], self.th[2], PB.TAIL_MASS), flags
            return SC.conformal_rank_p(0.0 if y > self.vmax else 0.01 * nn, nn), flags
        if self.tl:
            return SC.tail_p(-x, -self.tl[0], self.tl[1], self.tl[2], PB.TAIL_MASS), flags
        return SC.conformal_rank_p(0.0 if y < self.vmin else 0.01 * nn, nn), flags


def _check(typ: str, rec: Any, v: Any, num: Any) -> Tuple[float, List[str]]:
    if typ == "num":
        if isinstance(rec, _NumFast):
            return rec.p(v)
        return PB.p_value(rec, v, num=num)
    if typ == "set":
        return PG.check_set(rec, v)
    if typ == "cat":
        return PG.check_cat(rec, v)
    return PG.check_text(rec, v)


def _fmt_expected(typ: str, rec: Mapping[str, Any]) -> str:
    if typ == "num":
        d = rec.get("disp90") or {}
        r = rec.get("disp_range") or {}
        return (d.get("text") or "") + (f"；范围 {r.get('text')}" if r.get("text") else "")
    if typ == "set":
        return "必含 " + "、".join(rec.get("required") or [])
    if rec.get("grammar"):
        return f"`{rec['grammar']}`"
    if rec.get("closed") is not None:
        return "{" + ",".join(map(str, rec["closed"][:8])) + "}"
    return ""


# ================================================================== engine
class ConformityEngine(Engine):
    name = "behavior.conformity"
    layer = "behavior"
    consumes = [EV.EVT_BATCH, EV.EVT_WIN, EV.EVT_CTX, MP.PTREE, MP.PBOUNDS, MP.PGRAMMAR, MP.PBIND,
                MP.PWIN, MP.PFLOW, MP.WHO_GROUPS, WG_STATE, MP.SYSPROF, MP.ATTR]
    produces = [EV.PAT_ASSIGN, EV.PAT_RATE, CONF_SERIES, "behavior.score", "behavior.pm",
                "behavior.axes", "event.pattern_violation"]
    description = "P03: typed conformity p-values against the most specific confident pattern"
    interval = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.k_int = int(params.get("k_int", K_INT))
        self.cells = int(params.get("cells", CELLS))
        self.last_active_cap = int(params.get("last_active", LAST_ACTIVE))
        self.v_max = int(params.get("v_max", V_MAX))
        self.last_stats: Dict[str, Any] = {}

    # ================================================================= run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        if not EV.enabled(ctx.config):
            return 0
        store = ctx.store
        now = float(ctx.now)
        dt = float(ctx.window_s or 60.0)
        D = EV.learn_delay_s(dt, ctx.config)
        store.ensure_retention("pat.", max_age_s=D + 2.0 * dt)
        st = store.get_model(ORG, ORG, STATE)
        if not isinstance(st, ConfState):
            st = ConfState(self.cells, self.last_active_cap)
        bud = MP.get_org_model(store, MP.BUDGET)
        k_int = self.k_int
        if isinstance(bud, Mapping) and bud.get("k_int"):
            k_int = int(bud["k_int"])
        t0 = time.perf_counter()
        n = 0
        stats: Dict[str, Any] = {}
        for s in sorted(set(store.batch_systems(EV.EVT_BATCH)) | set(store.batch_systems(EV.EVT_WIN))):
            key = MP.tree_key(store, s)
            tc = _TreeCtx(self, store, s, key, ctx.config, now)
            self._hour_roll(st, store, s, now, tc.tz, k_int)
            agg: Dict[str, Dict[str, Any]] = {}
            cands: List[Tuple[int, float, Callable[[], BehaviorEvent]]] = []
            for series in (EV.EVT_BATCH, EV.EVT_WIN):
                last = st.last_batch.get((s, series), -math.inf)
                for ts_b, b in store.batches_since(s, series, last):
                    st.last_batch[(s, series)] = ts_b
                    if b.n == 0:
                        continue
                    n += self._score_batch(st, tc, store, s, ts_b, b, series, agg, cands)
            self._write_scores(store, s, now, dt, agg)
            cands.sort(key=lambda x: (-x[0], x[1]))  # highest severity, then smallest p
            for _, _, build in cands[:self.v_max]:
                store.add_event(build())
            st.stats["violations"] += min(len(cands), self.v_max)
            st.stats["suppressed"] += max(0, len(cands) - self.v_max)
            stats[s] = {"ips": len(agg), "violations": len(cands)}
        store.put_model(ORG, ORG, STATE, st, ts=now)
        us = (time.perf_counter() - t0) * 1e6
        st.stats["us"] += us
        st.stats["events"] += n
        self.last_stats = {"events": n, "systems": stats,
                           "us_per_event": us / n if n else None,
                           "state_bytes": st.nbytes()}
        return n

    # ----------------------------------------------------------- hour roll
    def _hour_roll(self, st: ConfState, store: Any, s: str, now: float, tz: float, k_int: int) -> None:
        hid = int((now - 1e-6 + tz) // HOUR)
        hs = st.hour.get(s)
        if hs is None:
            st.hour[s] = HourSS(k_int, hid)
            return
        if hs.hour == hid:
            return
        rows = [(k[1], k[2], k[0], g) for k, g in hs.rows() if g > 0]
        # long-run tally of finished IP-hours per node (the rank reference of
        # the hour count, see RateTally)
        st.tally().close_hour(MP.tree_key(store, s), rows, now)
        store.add_batch(s, EV.PAT_RATE, now, {"rows": rows, "hour": hs.hour, "N": hs.N,
                                              "untracked": max(0.0, hs.N - sum(g for *_, g in rows))})
        st.hour[s] = HourSS(k_int, hid)

    # ------------------------------------------------------------ batches
    def _score_batch(self, st: ConfState, tc: _TreeCtx, store: Any, s: str, ts_b: float, b: Any,
                     series: str, agg: Dict[str, Dict[str, Any]],
                     cands: List[Tuple[int, float, Callable[[], BehaviorEvent]]]) -> int:
        kind = getattr(b, "kind", EV.KIND_TXN)
        tree = tc.ptm.kinds.get(kind) if tc.ptm is not None else None
        txn = series == EV.EVT_BATCH
        cb = store.batch_at(s, EV.EVT_CTX, ts_b) if txn else None
        if cb is not None and cb.n != b.n:
            cb = None
        n = b.n
        out_f = {c: np.full(n, NAN) for c in ("leaf", "conf", "act_node", "exc", "p_who", "p_when",
                                                "p_content", "p_seq", "p_novel", "p_ev", "vtype",
                                                "damp", "think")}
        act_col = np.empty(n, dtype=object)
        prev_col = np.empty(n, dtype=object)
        readdr: Dict[int, str] = {}
        flags_col: Dict[int, str] = {}
        overlay: Dict[Tuple[str, str], List[Any]] = {}
        hs = st.hour.get(s)
        order = np.argsort(b.ts, kind="stable").tolist()
        shared = tc.hier.shared
        sk_col = b.dense("sess.key", None) if txn and b.has("sess.key") else None
        for i in order:
            ip = b.ip_of(i)
            ts = float(b.ts[i])

            def get(a: str, i: int = i, ip: str = ip) -> Any:
                v = b.get(a, i)
                if v is EV.ABSENT and cb is not None:
                    v = cb.get(a, i)
                if v is EV.ABSENT and a == "net.src":
                    return ip
                return v
            r = self._score_event(st, tc, kind, tree, s, get, ip, ts, txn, hs, overlay, sk_col, i,
                                  shared, agg, cands)
            st.last_active.put((s, ip), ts)
            if r is None:
                continue
            for c, v in r["num"].items():
                out_f[c][i] = v
            act_col[i] = r.get("act")
            prev_col[i] = r.get("prev")
            if r.get("readdr"):
                readdr[i] = r["readdr"]
            if r.get("flags"):
                flags_col[i] = ",".join(sorted(r["flags"]))
        if txn:
            rows = np.arange(n, dtype=np.int32)
            cols = {c: EV.Col(rows, v) for c, v in out_f.items()}
            cols["act"] = EV.Col(rows, act_col)
            cols["prev_act"] = EV.Col(rows, prev_col)
            if readdr:
                ks = sorted(readdr)
                cols["readdr"] = EV.Col(np.asarray(ks, dtype=np.int32),
                                        np.asarray([readdr[k] for k in ks], dtype=object))
            if flags_col:
                ks = sorted(flags_col)
                cols["flags"] = EV.Col(np.asarray(ks, dtype=np.int32),
                                       np.asarray([flags_col[k] for k in ks], dtype=object))
            store.add_batch(s, EV.PAT_ASSIGN, ts_b, b.aligned(cols, {"kind": kind, "tree_key": tc.key}))
        return n

    # --------------------------------------------------------------- event
    def _score_event(self, st: ConfState, tc: _TreeCtx, kind: int, tree: Any, s: str,
                     get: Callable[[str], Any], ip: str, ts: float, txn: bool, hs: Optional[HourSS],
                     overlay: Dict[Tuple[str, str], List[Any]], sk_col: Any, i: int,
                     shared: Set[str], agg: Dict[str, Dict[str, Any]],
                     cands: List[Tuple[int, float, Callable[[], BehaviorEvent]]]) -> Optional[Dict[str, Any]]:
        t = tc.now
        num: Dict[str, float] = {}
        res: Dict[str, Any] = {"num": num}
        route_key = _route_key(get) if txn else None
        path: List[int] = []
        conf_i = -1
        if tree is not None:
            path = tree.route(get, tc.hier, tc.gone, ts)
            for j in range(len(path) - 1, -1, -1):
                if tree.nodes[path[j]].confident:
                    conf_i = j
                    break
            num["leaf"] = float(path[-1])
            num["conf"] = float(path[conf_i]) if conf_i >= 0 else NAN
        # ---- action, variant, session (seq)
        p_seq, p_novel = NAN, NAN
        novel_coord = False
        flags: Set[str] = set()
        seq_missing: List[str] = []
        if txn and route_key is not None:
            key = route_key
            if tree is not None:
                cs = tc.content_split_nodes(kind, tree)
                v = 0
                for par, child in zip(path, path[1:]):
                    if par in cs:
                        v = child
                if v:
                    key = f"{route_key}#v{v}"
                for nid in path:
                    nd = tree.nodes[nid]
                    if any(c[0] in ROUTE_ATTRS and not c[3] for c in nd.ctx):
                        num["act_node"] = float(nid)
                        break
            res["act"] = key
            sk = (sk_col[i] if sk_col is not None and ip in shared and sk_col[i] not in (None, EV.ABSENT)
                  else DF.NO_KEY)
            sess_k = (ip, sk)
            e = overlay.get(sess_k)
            if e is None and tc.flow is not None:
                e = tc.flow.sessions.peek(sess_k)
            if e is not None and ts - float(e[1]) <= tc.gap:
                a_key, bits = e[2], int(e[4])
                num["think"] = ts - float(e[1]) if ts >= float(e[1]) else NAN
            else:
                a_key, bits = None, 0
            res["prev"] = a_key
            if tc.flow is not None and isinstance(tc.pflow, Mapping):
                g = tc.ip2g.get(ip)
                sc = DF.seq_scores(tc.pflow, str(g) if g is not None else DF.STAR, a_key, key, bits, t)
                p_seq = float(sc.get("p_seq", NAN))
                if sc.get("missing"):
                    seq_missing = [tc.flow.acts.key_of(a) or "?" for a in sc["missing"]]
                    res["p_req"] = sc.get("p_req")
                acts = tc.flow.acts
                k10 = DF.route_key(get)                  # the key P10 itself computes
                led = st.led()
                lday = int((ts + tc.tz) // DAY)
                le = led.peek(s, route_key)
                g_ip = tc.ip2g.get(ip)
                g_ip = str(g_ip) if g_ip is not None else None
                if all(acts.id_of(x) is None for x in {key, route_key, k10} if x):
                    tot_ev = acts.ss.ss.total_evidence(t)
                    if ActionLedger.established(le, lday):
                        # seen on an earlier date by >= 2 sources: a recurring action
                        # (weekly / monthly) P10's decayed dictionary no longer holds
                        flags.add("recurring_action")
                    elif tot_ev >= N_MIN:
                        p_novel = float(acts.unseen(t))
                        flags.add("new_action")
                        if ActionLedger.coordinated(le, g_ip, ip, lday):
                            novel_coord = True
                            flags.add("group_action")
                led.add(s, route_key, ip, g_ip, lday)
            overlay[sess_k] = [None, ts, key, b"", bits | DF.bloom_bits(DF.h64(key))]
        # ---- novelty: `other` branch of a confident route split (without P10's dictionary)
        if tree is not None and tc.flow is None:
            for par, child in zip(path, path[1:]):
                pn = tree.nodes[par]
                if pn.split is None or pn.split.attr not in ROUTE_ATTRS or child != pn.split.other:
                    continue
                if not pn.confident or tree.nodes[child].confident:
                    continue
                tg = pn.targets.get(pn.split.attr)
                if isinstance(tg, PN.CatSummary) and get(pn.split.attr) in tg.ss:
                    continue
                U = float(tg.ss.unseen(t)) if isinstance(tg, PN.CatSummary) else \
                    0.5 / (pn.n_c(t) + 1.0)
                p_novel = U if _nan(p_novel) else min(p_novel, U)
                flags.add("other_branch")
        # ---- no confident node: only seq / novel speak
        conf_nd = tree.nodes[path[conf_i]] if (tree is not None and conf_i >= 0) else None
        p_who = p_when = p_content = NAN
        p_int = NAN
        details: Dict[str, Any] = {}
        cap_low: Set[str] = set()
        if novel_coord:
            cap_low.add("novel")                  # a new behaviour of the group, not of one source
        readdr_src: Optional[str] = None
        bind_fl: Set[str] = set()
        lb_x = NAN
        if conf_nd is not None:
            cinfo = tc.node_info(kind, tree, conf_nd.id)
            # ---- content (bindings first: their flags refine who)
            X = conf_nd
            if conf_nd.exc and ip in conf_nd.exc and conf_nd.exc[ip] in tree.nodes:
                X = tree.nodes[conf_nd.exc[ip]]
                num["exc"] = 1.0
            # each attribute is checked once, at the deepest fitted node holding it
            # (X, the covering node, its ancestors); the root's records (the whole
            # system's mixture) only when the covering node is the root
            chain = [X] + [tree.nodes[path[j]] for j in range(conf_i, -1, -1)
                           if tree.nodes[path[j]] is not X and (j > 0 or conf_i == 0)]
            ps: List[float] = []
            seen_attr: Set[str] = set()
            worst: Tuple[float, str, Any, str, Any] = (2.0, "", None, "", None)
            for nd in chain:
                ni = tc.node_info(kind, tree, nd.id)
                for a, typ, rec in ni.content:
                    if a in seen_attr:
                        continue
                    v = get(a)
                    if v is EV.ABSENT:
                        continue
                    seen_attr.add(a)
                    p_cur, fl = _check(typ, rec, v, None)
                    if _nan(p_cur):
                        continue
                    mname = MP.PBOUNDS if typ == "num" else MP.PGRAMMAR
                    rk = (mname, a)
                    rr = ni.refc.get(rk, False)
                    if rr is False:
                        rr = _ref_record(nd, mname, a)
                        if rr and typ == "num":
                            rr = _NumFast(rr)
                        ni.refc[rk] = rr
                    p_ref = _check(typ, rr, v, None)[0] if rr else NAN
                    if a in ni.evolving:
                        p = max(p_cur, p_ref) if not _nan(p_ref) else p_cur
                        cap_low.add("content")
                    else:
                        p = SC.dual_anchor(p_cur, p_ref)
                    ps.append(p)
                    flags.update(fl)
                    if p < worst[0]:
                        worst = (p, a, v, typ, rec.rec if isinstance(rec, _NumFast) else rec)
                # invariants
                if nd.confident and nd.inv and nd.n_c(t) >= N_MIN:
                    for a, (lv, val) in nd.inv.items():
                        if a in seen_attr or a.startswith(SKIP_PREFIX):
                            continue
                        v = get(a)
                        if v is EV.ABSENT:
                            continue
                        seen_attr.add(a)
                        g = tc.hier.gen(a, lv, v)
                        if g != val and str(g) != str(val):
                            p = SC.invariant_p(0.0, nd.n_c(t))
                            ps.append(p)
                            flags.add("invariant")
                            if p < worst[0]:
                                worst = (p, a, v, "inv", {"value": val})
                # bindings
                for rec in ni.binds:
                    X_, Y_ = rec.get("x"), rec.get("y")
                    if not X_ or not Y_ or ("bind", X_, Y_) in seen_attr:
                        continue
                    seen_attr.add(("bind", X_, Y_))
                    if rec.get("dir") == "rev":
                        y = get(X_)                              # payload value (e.g. a user name)
                        wa, wl = FD.parse_x(Y_)
                        x = tc.hier.gen(wa, wl, get(wa))
                        if y is EV.ABSENT or x is EV.ABSENT:
                            continue
                        known = self._known(tc, st, s, ip)
                        day_start = ts - ((ts + tc.tz) % DAY)
                        p, fl = FD.check_reverse(rec, x, y, ts,
                                                 lambda src: st.last_active.peek((s, str(src))),
                                                 known, day_start)
                        if "readdress_candidate" in fl:
                            ent = (rec.get("table") or {}).get(str(FD._jv(y))) or {}
                            readdr_src = str(ent.get("top")) if ent.get("bound") else \
                                (str((ent.get("set") or [None])[0]) if ent.get("set") else None)
                    else:
                        xa, xl = FD.parse_x(X_)
                        x = tc.hier.gen(xa, xl, get(xa))
                        y = get(Y_)
                        if y is EV.ABSENT or x is EV.ABSENT:
                            continue
                        p, fl = FD.check_forward(rec, x, y)
                        ent = (rec.get("table") or {}).get(str(x)) or {}
                        if ent.get("LB") is not None:
                            lb_x = float(ent["LB"])
                    if _nan(p):
                        continue
                    ps.append(p)
                    bind_fl.update(fl)
                    if p < worst[0]:
                        worst = (p, str(Y_ if rec.get("dir") != "rev" else X_), None, "bind", rec)
            # per-IP intensity (upper tail, guaranteed count, heavy only)
            if hs is not None and txn:
                ck = (ip, kind, conf_nd.id)
                hs.add(ck)
                g_cnt = hs.guaranteed(ck)
                if g_cnt > hs.N / hs.k:
                    rrec = None
                    for j in range(conf_i, -1, -1):
                        rrec = tc.node_info(kind, tree, path[j]).rate
                        if rrec:
                            break
                    if rrec and rrec.get("band90") and g_cnt > float(rrec["band90"][1]):
                        # the rank of the count among every IP-hour P03 closed at the
                        # node (RateTally), not P06's decayed continuous fit
                        p = st.tally().p(tc.key, kind, conf_nd.id, g_cnt, ts)
                        if not _nan(p):
                            # the hour count is one cumulative statistic, not a new test per
                            # event: it bypasses calibration and the per-day multiplicity
                            p_int = max(float(p), P_FLOOR)
                            if p_int <= INT_P:
                                # the node's intensity routine is kept on the MAGNITUDE of
                                # its tail hours (x = band90 top / count, binned like a p):
                                # beyond the digest's maximum every count has the same
                                # rank p, and a crawler's 20 an hour must not make a
                                # 400-an-hour burst routine
                                ik = (tc.key, kind, conf_nd.id, "intensity")
                                x_int = min(1.0, max(float(rrec["band90"][1]), 1.0) / g_cnt)
                                routine = st.cal.per_day(ik, x_int, ts) >= E_MAX
                                if not isinstance(getattr(hs, "tail", None), dict):
                                    hs.tail = {}               # a sketch restored from an older state
                                # one unit per tail hour of the pair, at the hour's most
                                # extreme magnitude
                                x_old = hs.tail.get(ck)
                                if x_old is None:
                                    hs.tail[ck] = x_int
                                    st.cal.add(ik, x_int, ts)
                                elif x_int < x_old:
                                    hs.tail[ck] = x_int
                                    st.cal.move(ik, x_old, x_int, ts)
                                if routine:
                                    p_int = NAN
                                else:
                                    flags.add("intensity")
                                    details["intensity"] = (p_int, g_cnt, rrec)
                            else:
                                p_int = NAN
            flags |= bind_fl
            p_content = SC.content_p(ps)
            if not _nan(p_content):
                details["content"] = worst
            # ---- who
            def _is_member(ni_: Any, keys_: Sequence[Any]) -> bool:
                item_ = keys_[ni_.who_l]
                if item_ in ni_.who_heavy:
                    return True
                # a source with standing of its own (>= MEMBER_EV evidence units,
                # i.e. recurring over runs and days, damped rows counting 0.1)
                # is a member though it holds < 5 % of the mass or evidence:
                # at an ancestor (back-off) whose population mixes the
                # children's, and equally at the covering node itself (pack O:
                # the finance approver, ~3 % of finance's evidence, was outside
                # the root's heavy set, flagged on every login - 8 HIGH incidents
                # - and damped, so its own approval nodes stalled at n_c = 12
                # and never confirmed)
                ss_ = ni_.nd.who.levels[ni_.who_l]
                if item_ in ss_ and ss_.evidence(item_, t) >= MEMBER_EV:
                    return True
                if ni_.who_grp and keys_[GRP_LEVEL] in ni_.who_grp:
                    # a new address of a member group at a node whose group level is
                    # closed too: a colleague, not an outsider (§6.9.2) - provided the
                    # node is a pattern of the group (>= GROUP_MIN_MEMBERS other
                    # members use it), not one member's individual pattern
                    g0 = tc.ip2g.get(ip)
                    return g0 is None or group_members_at(ni_.nd, g0, ip, t, tc.ip2g) >= GROUP_MIN_MEMBERS
                # a new address inside a CONFIGURED address pool (dhcp_scopes /
                # ip_classes: the operator's statement that these addresses are one
                # population) whose pool has standing at the node: a re-addressed
                # member, not an unknown source. Pack O: on the make-up Saturday
                # five fresh 研发 DHCP leases logging in to OA were MEDIUM unknown_ip
                # who findings against a /24-closed login node whose fourth /24 of the
                # pool had not been drawn yet
                if ni_.who_l < REG_LEVEL and len(keys_) > REG_LEVEL:
                    rg = keys_[REG_LEVEL]
                    if rg is not None and str(rg) not in NONE_ITEMS and tc.pool_regions \
                            and str(rg) in tc.pool_regions:
                        lv = ni_.nd.who.levels[REG_LEVEL] if len(ni_.nd.who.levels) > REG_LEVEL else None
                        if lv is not None and rg in lv and lv.evidence(rg, t) >= MEMBER_EV:
                            return True
                return False

            for j in range(conf_i, -1, -1):
                ni = tc.node_info(kind, tree, path[j])
                if ni.who_l is None:
                    continue
                keys = tc.who_keys(ip)
                member = _is_member(ni, keys)
                p_cur = 1.0 if member else ni.who_U
                if not member:
                    # foreign at every closed ancestor too (up to the system root):
                    # the most confident of those closed populations says how
                    # unlikely the source is - a young single-user node (U ~ 1/n_c)
                    # must not hide that the source never used the closed SYSTEM
                    # (pack O A1: 192.168.1.23 approving in finance, node U 0.029,
                    # root U 0.0013). Bonferroni over the closed levels tested.
                    u_min, n_t = ni.who_U, 1
                    for j2 in range(j - 1, -1, -1):
                        ni2 = tc.node_info(kind, tree, path[j2])
                        if ni2.who_l is None:
                            continue
                        if _is_member(ni2, keys):
                            break
                        n_t += 1
                        u_min = min(u_min, ni2.who_U)
                    p_cur = min(ni.who_U, u_min * n_t)
                p_ref = NAN
                if ni.who_ref is not None and ni.who_l == 0:
                    p_ref = 1.0 if str(ip) in ni.who_ref else ni.who_U
                p_who = SC.dual_anchor(p_cur, p_ref)
                if "@who" in ni.evolving:
                    p_who = max(p_cur, p_ref) if not _nan(p_ref) else p_cur
                    cap_low.add("who")
                details["who"] = (ni, j)
                if p_who < 1.0:
                    g = tc.ip2g.get(ip)
                    nd = ni.nd
                    if g is not None and group_outsider(nd, g, ip, t, tc.ip2g):
                        flags.add("outsider_group")
                    root = tree.nodes[tree.root]
                    if g is not None and root.who.levels[3].total(t) > 0 and f"grp:{g}" not in root.who.levels[3]:
                        flags.add("system_new")
                    if g is None and not self._known(tc, st, s, ip):
                        flags.add("unknown_ip")
                break
            # ---- when
            dt_raw = get("ctx.daytype")
            daytype = 0 if dt_raw in ("workday", "wd", 0, "makeup") else 1 if dt_raw is not EV.ABSENT else None
            minute = get("ctx.tod_min")
            if minute is EV.ABSENT:
                minute = ((ts + tc.tz) % DAY) / 60.0
            if daytype is None:
                daytype = 0 if ((int((ts + tc.tz) // DAY) + 3) % 7) < 5 else 1
            minute = float(minute)
            slot = int(minute // 15) % 96
            for j in range(conf_i, -1, -1):
                nd = tree.nodes[path[j]]
                cur_t, ref_t = tc.when_table(kind, nd, daytype)
                if cur_t is None:
                    continue
                p_cur = float(cur_t[slot])
                p_ref = float(ref_t[slot]) if ref_t is not None else NAN
                if "@when" in tc.node_info(kind, tree, nd.id).evolving:
                    p_when = max(p_cur, p_ref) if not _nan(p_ref) else p_cur
                    cap_low.add("when")
                else:
                    p_when = SC.dual_anchor(p_cur, p_ref)
                wins = tc.node_info(kind, tree, nd.id).wins
                if wins:
                    iv = wins.get("nonworkday" if daytype else "workday") or []
                    if iv and not PW.in_windows(minute, iv):
                        flags.add("outside_windows")
                details["when"] = (nd, minute, daytype, wins)
                break
        if seq_missing:
            flags.add("missing_predecessor")
        # ---- combine
        pt = {"who": p_who, "when": p_when, "content": p_content, "seq": p_seq, "novel": p_novel}
        pm = {k: (v if _nan(v) else max(float(v), P_FLOOR)) for k, v in pt.items()}
        pt = dict(pm)
        if conf_nd is not None:
            for k in CAL_TYPES:
                if not _nan(pm[k]) and pm[k] <= CAL_P:
                    pt[k] = st.cal.p((tc.key, kind, conf_nd.id, k), pm[k], ts)
        details["pm"] = pm
        if not _nan(p_int):
            pt["content"] = p_int if _nan(pt["content"]) else min(1.0, 2.0 * min(pt["content"], p_int))
        for k, v in pt.items():
            num[f"p_{k}"] = v
        p_ev = SC.event_p(pt.values())
        num["p_ev"] = p_ev
        num["vtype"] = float(SC.vtype_mask(pt, VTYPE_P))
        damp = 1.0
        if conf_nd is not None and conf_nd.state in ("confirmed", "stable"):
            if not _nan(p_ev) and p_ev <= DAMP_P:
                damp = DAMP
            elif (not _nan(p_who) and p_who < 1.0 and "readdress_candidate" not in bind_fl
                  and ("outsider_group" in flags or tc.ip2g.get(ip) is None)):
                # §6.9.2: persistence alone never makes a foreign source a member of
                # a who-closed pattern; its events are learned damped until it
                # qualifies (a colleague's group already there, a readdress, a label)
                damp = DAMP
            elif "concurrent_use" in bind_fl or ("cross_binding" in bind_fl and lb_x >= LB_CROSS):
                # a value credibly bound to ANOTHER source (or in use there right now)
                # is not learned at full weight either: undamped, a borrowed
                # credential used for five days became the borrower's own binding
                # (pack O, A2: 192.168.1.21 -> {jack, rose} on day 21, PG5
                # non-adoption). A new value bound nowhere (a legitimate rename,
                # D2 mike -> mike.w) carries no such flag and is learned as before.
                damp = DAMP
        num["damp"] = damp
        if conf_nd is not None:                     # prequential: the calibration learns after scoring
            for k in CAL_TYPES:
                st.cal.add((tc.key, kind, conf_nd.id, k), pm[k], ts, damp)
        res["flags"] = flags
        if readdr_src is not None and "readdress_candidate" in bind_fl:
            res["readdr"] = readdr_src
        # ---- per (s, ip) aggregation for the tick scores
        a = agg.setdefault(ip, {"p": {t_: 1.0 for t_ in TYPES}, "n": {t_: 0 for t_ in TYPES},
                                "axes": {t_: set() for t_ in TYPES}})
        for k, v in pt.items():
            if _nan(v):
                continue
            a["n"][k] += 1
            if v < a["p"][k]:
                a["p"][k] = v
        # ---- discrete findings
        if conf_nd is not None and conf_nd.state in ("confirmed", "stable"):
            self._findings(st, tc, s, kind, tree, conf_nd, path, conf_i, ip, ts, pt, flags, bind_fl,
                           lb_x, details, cap_low, route_key, seq_missing, res, a, cands)
        return res

    @staticmethod
    def _known(tc: _TreeCtx, st: ConfState, s: str, ip: str) -> bool:
        if ip in tc.ip2g:
            return True
        if tc.sigs is not None:
            try:
                if ip in tc.sigs:
                    return True
            except TypeError:
                pass
        if st.last_active.peek((s, ip)) is not None:
            return True
        tree = tc.ptm.kinds.get(EV.KIND_TXN) if tc.ptm is not None else None
        if tree is not None and ip in tree.nodes[tree.root].who.levels[0]:
            return True
        return False

    # ------------------------------------------------------------ findings
    def _findings(self, st: ConfState, tc: _TreeCtx, s: str, kind: int, tree: Any, conf_nd: Any,
                  path: List[int], conf_i: int, ip: str, ts: float, pt: Mapping[str, float],
                  flags: Set[str], bind_fl: Set[str], lb_x: float, details: Mapping[str, Any],
                  cap_low: Set[str], route_key: Optional[str], seq_missing: List[str],
                  res: Mapping[str, Any], a: Dict[str, Any],
                  cands: List[Tuple[int, float, Callable[[], BehaviorEvent]]]) -> None:
        t = tc.now
        method = route_key.split()[0] if route_key and route_key.split()[0].isupper() else ""
        write = method in WRITE
        sensitive = tc.sensitive(route_key)
        who_d = details.get("who")
        few = bool(who_d and len(who_d[0].who_heavy) <= 3)
        sigma = min(3.0, 1.0 + float(write) + 0.5 * float(few) + float(sensitive))
        local_day = int((ts + tc.tz) // DAY)
        hour_local = ((ts + tc.tz) % DAY) / 3600.0
        night = not (8.0 <= hour_local < 20.0)
        out: List[Tuple[str, Severity, List[str], float, float, Dict[str, Any]]] = []
        ck = (s, ip, conf_nd.id)
        C = st.cells.get(ck)
        if C is None or C[0] != local_day:
            C = [local_day, {}]
            st.cells.put(ck, C)

        def cell(typ: str, p: float) -> Tuple[float, int, List[Any]]:
            """Per (IP, node, type) and local day: [n_day, p_min, emitted rank]."""
            e = C[1].get(typ)
            if e is None:
                e = C[1][typ] = [0, 1.0, -1]
            e[0] += 1
            e[1] = min(e[1], p)
            return SC.day_p(e[1], e[0]), e[0], e
        pm = details.get("pm") or {}

        def routine(typ: str) -> bool:
            """The node itself produced such scores >= E_MAX times a day (its
            H_m-decayed history, damped outliers at 0.1): part of its routine."""
            return st.cal.per_day((tc.key, kind, conf_nd.id, typ), pm.get(typ, NAN), ts) >= E_MAX
        # who
        p = pt["who"]
        if not _nan(p) and p < 1.0:
            U = p
            sev = None
            if U <= 0.01 and sigma >= 2 and "outsider_group" in flags:
                sev = Severity.HIGH
            elif U <= 0.02 and sigma >= 2 and ("outsider_group" in flags or "unknown_ip" in flags):
                sev = Severity.LOW if "readdress_candidate" in bind_fl else Severity.MEDIUM
            elif U <= 0.02 and {"outsider_group", "system_new"} <= flags and "readdress_candidate" not in bind_fl:
                # a lateral move: the source's group never used this SYSTEM at all and
                # the node is closed to others - MEDIUM on a read too (the §6.16.3
                # HIGH row's `lateral` axis, without its write / sensitivity
                # condition). Pack O A9: a sales address reading finance's approval
                # list (GET, sigma 1.5) was LOW, so no incident ever opened
                sev = Severity.MEDIUM
            elif U <= 0.05:
                sev = Severity.INFO if "readdress_candidate" in bind_fl else Severity.LOW
            if "concurrent_use" in bind_fl and sev is not None:
                sev = Severity.HIGH
            if sev is not None:
                axes = ["privilege"] + (["lateral"] if "system_new" in flags else [])
                pd, nd_, c = cell("who", p)
                out.append(("who", sev, axes, p, pd, {"cell": c}))
        # content / binding
        p = pt["content"]
        if not _nan(p):
            pd, nd_, c = cell("content", p)
            sev = None
            axes = ["content"]
            wd = details.get("content")
            attr = wd[1] if wd else ""
            if bind_fl & {"cross_binding", "concurrent_use", "unbound_value", "outside_set",
                          "foreign_source", "readdress_candidate"}:
                axes = ["credential"]
                # only credential-grade evidence has its own severity: a value bound to
                # another source on a write / sensitive action, or concurrent use; an
                # unbound value, a value outside the set or a foreign source is judged
                # by the ordinary per-day content rule below
                if "cross_binding" in bind_fl and (lb_x >= LB_CROSS) and (write or sensitive):
                    sev = Severity.HIGH if nd_ >= 2 else Severity.MEDIUM
                elif "concurrent_use" in bind_fl:
                    sev = Severity.HIGH if (not _nan(pt["who"]) and pt["who"] < 1.0) else Severity.MEDIUM
            rt = pd <= 1e-3 and routine("content")
            if "injection_shape" in flags or (pd <= 1e-4 and write and not rt):
                sev = max(sev or Severity.INFO, Severity.MEDIUM, key=SEV_RANK.get)
            elif pd <= 1e-3 and not rt:
                sev = max(sev or Severity.INFO, Severity.LOW, key=SEV_RANK.get)
            if attr.startswith("m.") or attr == RATE_ATTR:
                axes = sorted(set(axes) | {"volume"})
                if attr == RATE_ATTR:
                    axes = sorted(set(axes) | {"credential"} if write else set(axes))
            if attr in ("body.len", "net.bytes_up") and "above_range" in flags:
                axes = sorted(set(axes) | {"exfil"})
            if sev is not None:
                out.append(("content", sev, axes, p, pd, {"cell": c}))
        # per-IP intensity (§6.16.4): the guaranteed hour count against rate.ip_h
        it = details.get("intensity")
        if it is not None:
            p_i, g_cnt, rrec = it
            if p_i <= 1e-3:
                _, _, c = cell("intensity", p_i)
                sev = Severity.MEDIUM if (p_i <= 1e-4 and (write or sensitive)) else Severity.LOW
                axes = ["volume"] + (["credential"] if write else [])
                # the event's content p already carries the hour count: one finding
                for o in [o for o in out if o[0] == "content"]:
                    out.remove(o)
                    sev = max(sev, o[1], key=SEV_RANK.get)
                    axes = sorted(set(axes) | set(o[2]))
                out.append(("content", sev, axes, p_i, p_i, {"cell": c, "intensity": g_cnt}))
        # when
        p = pt["when"]
        if not _nan(p):
            pd, nd_, c = cell("when", p)
            if pd <= 1e-3 and "outside_windows" in flags and not routine("when"):
                sev = Severity.MEDIUM if (write and night) else Severity.LOW
                out.append(("when", sev, ["temporal"], p, pd, {"cell": c}))
        # seq
        p = pt["seq"]
        if seq_missing and not _nan(res.get("p_req", NAN)):
            p_req = float(res["p_req"])
            pd, nd_, c = cell("seq", p_req)
            rk = (tc.key, kind, conf_nd.id, "seq_req")
            rt = st.cal.per_day(rk, p_req, ts) >= E_MAX       # missing predecessors are routine here
            st.cal.add(rk, p_req, ts)
            if pd <= SEQ_P and not rt:
                out.append(("seq", Severity.MEDIUM if write else Severity.LOW, ["sequence"],
                            float(res["p_req"]), pd, {"cell": c, "missing": seq_missing}))
        # novel
        p = pt["novel"]
        if not _nan(p) and p <= 0.05:
            pd, nd_, c = cell("novel", p)
            sev = Severity.MEDIUM if (sensitive or write) else Severity.INFO
            axes = ["categorical"] + (["privilege"] if sensitive else [])
            out.append(("novel", sev, axes, p, pd, {"cell": c}))
        for typ, sev, axes, p, pd, extra in out:
            if typ in cap_low and SEV_RANK[sev] > SEV_RANK[Severity.LOW]:
                sev = Severity.LOW
            c = extra["cell"]
            if SEV_RANK[sev] <= c[2]:
                continue                                   # already emitted today at this level
            c[2] = SEV_RANK[sev]
            pid = PN.pattern_id(tc.key, kind, conf_nd.id, conf_nd.version, conf_nd.cver)
            a["axes"][typ].update(axes)
            date = _dt.datetime.utcfromtimestamp(ts + tc.tz).strftime("%Y-%m-%d")
            fl_all = sorted(flags | bind_fl)
            U_who = details["who"][0].who_U if details.get("who") else None
            n_c = float(conf_nd.n_c(t))

            def build(typ=typ, sev=sev, axes=list(axes), p=float(p), pd=float(pd), extra=extra,
                      pid=pid, date=date, fl_all=fl_all, U_who=U_who, n_c=n_c,
                      details=dict(details)) -> BehaviorEvent:
                # built only for the <= V_MAX findings a tick emits
                zh, en, observed, expected = self._describe(tc, typ, ip, route_key, details, flags,
                                                            bind_fl, extra, pt, ts)
                return BehaviorEvent(
                    system=s, entity=ip, ts=tc.now, kind="pattern_violation",
                    score=float(min(1.0, -math.log10(max(p, 1e-12)) / 10.0)), severity=sev,
                    description=zh, axes=axes,
                    p_value=p, p_by_detector={DETS[typ]: p},
                    dedupe_key=f"pv|{conf_nd.id}|{typ}|{ip}|{date}",
                    window=(ts, ts),
                    extra={"pattern_id": pid, "statement_zh": zh, "statement_en": en, "type": typ,
                           "flags": fl_all, "observed": observed, "expected": expected,
                           "p": p, "p_day": pd, "U": U_who, "sensitivity": sigma, "n_c": n_c,
                           "route": route_key, "node": conf_nd.id, "tree_key": tc.key,
                           "event_ts": ts, "high_candidate": sev == Severity.HIGH})
            cands.append((SEV_RANK[sev], float(p), build))

    # ------------------------------------------------------------ describe
    def _describe(self, tc: _TreeCtx, typ: str, ip: str, route: Optional[str], details: Mapping[str, Any],
                  flags: Set[str], bind_fl: Set[str], extra: Mapping[str, Any],
                  pt: Mapping[str, float], ts: float) -> Tuple[str, str, Any, Any]:
        r = " ".join(p for p in (route or "?").split() if not ("." in p and p.islower() and "/" not in p))
        g = tc.ip2g.get(ip)
        gname = str((tc.groups.get(g) or {}).get("name") or g) if g is not None else None
        who_txt = f"{ip}（{gname}）" if gname else ip
        if typ == "who":
            ni, _ = details["who"]
            exp = sorted(str(x) for x in ni.who_heavy)[:8]
            fl = sorted(flags & {"outsider_group", "unknown_ip", "system_new"} | bind_fl &
                        {"readdress_candidate", "concurrent_use"})
            zh = (f"{who_txt} 在 {tc.s} 执行 {r}：该模式的来源封闭于 {'、'.join(exp)}"
                  f"（未见来源概率 U = {pt['who']:.3g}；{','.join(fl)}）")
            en = (f"{ip} performed {r} on {tc.s}; the pattern's sources are closed on {', '.join(exp)} "
                  f"(U = {pt['who']:.3g}; {','.join(fl)})")
            return zh, en, ip, exp
        if typ == "when":
            nd, minute, daytype, wins = details["when"]
            obs = PW.hhmm(minute)
            iv = (wins or {}).get("nonworkday" if daytype else "workday") or []
            exp = "、".join(f"{PW.hhmm(a)}–{PW.hhmm(b)}" for a, b in iv) or "?"
            zh = f"{who_txt} 于 {obs} 执行 {r}，不在该模式的时间窗 {exp} 内（p = {pt['when']:.3g}）"
            en = f"{ip} performed {r} at {obs}, outside the pattern's windows {exp} (p = {pt['when']:.3g})"
            return zh, en, obs, exp
        if typ == "content":
            wd = details.get("content")
            if wd is None:
                return f"{who_txt} 执行 {r} 的内容异常", f"{ip}: unusual content on {r}", None, None
            p, a, v, kind, rec = wd
            if kind == "bind":
                fl = ",".join(sorted(bind_fl))
                zh = f"{who_txt} 执行 {r}：{a} 与已学到的绑定不符（{fl}，p = {p:.3g}）"
                en = f"{ip} performed {r}: {a} contradicts a learned binding ({fl}, p = {p:.3g})"
                return zh, en, a, fl
            exp = _fmt_expected(kind, rec) if kind != "inv" else str(rec.get("value"))
            obs = v if isinstance(v, (int, float, str)) else str(v)
            zh = f"{who_txt} 执行 {r}：{a} = {obs}，与该模式的约束 {exp} 不符（p = {p:.3g}）"
            en = f"{ip} performed {r}: {a} = {obs} does not fit the pattern's constraint {exp} (p = {p:.3g})"
            return zh, en, obs, exp
        if typ == "seq":
            miss = "、".join(PR_route(m) for m in extra.get("missing") or [])
            zh = f"{who_txt} 执行 {r} 前缺少必需的前置动作 {miss}"
            en = f"{ip} performed {r} without the required predecessor {miss}"
            return zh, en, r, miss
        zh = f"{who_txt} 执行了 {tc.s} 上的新动作 {r}（未见动作概率 {pt['novel']:.3g}）"
        en = f"{ip} performed a new action {r} on {tc.s} (unseen-action probability {pt['novel']:.3g})"
        return zh, en, r, None

    # -------------------------------------------------------------- scores
    @staticmethod
    def _write_scores(store: Any, s: str, now: float, dt: float, agg: Mapping[str, Dict[str, Any]]) -> None:
        if not agg:
            return
        try:
            from .lib.detectors import DETECTOR_INDEX
            registered = all(d in DETECTOR_INDEX for d in DETS.values())
        except Exception:                           # pragma: no cover
            registered = False
        for ip, a in agg.items():
            rec: Dict[str, Any] = {}
            scores: Dict[str, float] = {}
            pms: Dict[str, float] = {}
            axes: Dict[str, List[str]] = {}
            for t_ in TYPES:
                n = a["n"][t_]
                if n <= 0:
                    continue
                p = float(SC.tick_p(a["p"][t_], n))
                sc = -math.log10(max(p, 1e-300))
                rec[DETS[t_]] = {"score": sc, "p": p, "n": n, "axes": sorted(a["axes"][t_])}
                scores[DETS[t_]] = sc
                pms[DETS[t_]] = p
                if a["axes"][t_]:
                    axes[DETS[t_]] = sorted(a["axes"][t_])
            if not rec:
                continue
            store.upsert_dict(s, ip, CONF_SERIES, now, rec, int(dt))
            if registered:
                from .lib import emit as EM
                EM.write_scores(store, s, ip, now, scores, pm=pms, axes=axes or None, window_s=int(dt))


def PR_route(key: str) -> str:
    r = DF.split_key(str(key))[0]
    p = r.split()
    if len(p) >= 2 and p[0].isupper():
        return f"{p[0]} {p[-1]}"
    return r
