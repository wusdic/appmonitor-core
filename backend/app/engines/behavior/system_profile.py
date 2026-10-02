"""P12 SystemProfile (`behavior.system_profile`) — system characteriser,
scenario-adaptive strategy selector and system families
(docs/lib3/progressive.md §6.18, §6.20, §6.21, card P12). Library 3 (behaviour).

Requirement S20 ("要考虑真实环境下各种情况的适用性，不同的场景怎么自动适配哪些算法引擎")
and S12 (a portal whose IPs follow no rule is profiled by prefix / region or
without IP at all): each business system is MEASURED, and the engines and
granularities it runs with are chosen from declared preconditions plus the
measured prequential utility of each option, in bits per event, net of its
CPU cost (lib/pstrategy). S1 for servers (PPC-10): systems that run the same
application share one pattern tree (a system family, lib/pfamily), so the
state does not grow with the number of servers.

Per tick (light, O(rows of the tick) numpy + O(distinct IPs of the tick)):
    per system s with an evt.batch at this tick, a bounded tracker
    (model.sysprof_state@(s, '__system__')) is updated from the batch:
      daily HLLs of source IPs (7-day ring) + a cumulative HLL -> population, churn
      per-IP LRU (cap 256): last ts, dominant client stack -> stack concentration
      pooled decayed histogram of log10 inter-event gaps -> session identifiability
      decayed channel masses (http / body-or-query / opaque tls / sess.key)
      decayed signature sketches (route prefix, host, SNI, dst, port, stack)
      heavy IPs (SpaceSaving 8, H_s); for the dominant IPs the distinct client
      stacks per day and the concurrent session keys per 15 min -> snat_suspect
      non-workday mass (evt.ctx), DHCP-scope mass (config dhcp_scopes)
Per day and tree key (entity_due, crc32 phase):
    characteristics (§6.18.1) from the trackers of the tree's member systems,
    model.attr (P02), model.attrsel (P05 ip_info / who_mode), model.ptree (root
    who levels, P04's per-level who code lengths summed over all nodes),
    model.who_groups (P11), B17 shared_ip events, B23 labels, config;
    measurements: who bits/event per level (full information), fitter gains
    and costs from model.pbounds / pgrammar / pbind / pflow;
    lib/pstrategy.decide -> chosen arms (Hedge, budgeted UCB, hysteresis).
Per day (org, once): system families (lib/pfamily): weighted MinHash
    signatures, LSH candidates, 2-day matching, stable ids; a new family adopts
    the tree of its member with the most evidence (the others are checkpointed
    under 'ptree' and released); a detached member gets a copy of the family's
    tree (lineage kept).

Writes  model.sysprof@(tree key, '__system__') and a reference at every member
        system's own key: {'fmt', 'version', 't', 'day', 'tree_key', 'members',
        'characteristics', 'arms', 'chosen', 'reasons', 'probe', 'hints',
        'history', 'state'}; consumers read `chosen` (dims who, P06-P10 (+ p07-p10
        aliases), win, content, tier, e_max, B09-B15 applicability).
        model.sysfam@('__org__', '__org__') = {'version', 'member': {s: 'fam:<id>'},
        'families': {fid: [s]}, 'dst_members': {fid: {peer:port: s}}, 'state', ...}.
        Events strategy_changed / family_changed (INFO) at (s, '__system__').
Budget  per tick O(events of the tick) numpy + O(distinct IPs of the tick) for
        the who code: measured 7-15 µs per event with recurring IPs, <= ~75 µs
        per event when every event is a new IP (tests/engines/test_p12_*: 100 vs
        50 000 IPs, 0 vs 300 extra attributes); daily O(#nodes) for the fitter
        gains, O(A) registry reads; family pass O(#systems . k) (LSH, no
        all-pairs scan). Memory per system 80-250 KB (tracker: LRU 256 IPs,
        SpaceSaving-capped who code, 8 192 per-day IP counts), flat in the
        number of attributes and capped in the number of IPs.
Deviations from §6.18 (measured reasons in lib/pstrategy and below): churn is
        new IPs / distinct IPs of the day; the who measurement is the system-level
        code (P04's per-leaf sums are kept as 'who_tree'); fitter gains are
        recomputed from the per-node records; SNAT evidence is a full day's share
        plus >= 10 stacks or >= 5 overlapping sessions behind the source.
Inert unless config['progressive']['enabled'].
"""
from __future__ import annotations

import copy
import ipaddress
import math
import zlib
from collections import Counter, OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import ORG, SYSTEM_ENTITY, BehaviorEvent, Severity
from .lib import detectors as DET
from .lib import m_ptree as MP
from .lib import pevent as EV
from .lib import pdfg as PD
from .lib import pfamily as PF
from .lib import psketch as PS
from .lib import pstrategy as PSt
from .lib.phier import GRP_NONE, REG_NONE, STAR, Regions, ip_prefix

STATE = "model.sysprof_state"
DAY = PS.DAY
PERIOD_S = DAY
MIN_DAYS = 1                               # full local days measured before the first decision
FRESH_S = 1.5 * DAY                        # a fitter gain older than this is not a measurement
COST_CAP_US = 1e5                          # µs/event cap of a periodic cost estimate
LRU_IPS = 256
GAP_BINS = 48
GAP_LO, GAP_HI = 0.0, 6.0                  # log10 seconds
SNAT_SHARE = 0.9
SNAT_STACKS = 10.0                         # distinct client stacks behind one source in a day
SNAT_CONC = 5.0                            # overlapping sessions (keys) behind one source at one time
STACK_CONC = 0.9
AUTOMATION_SCORE = 0.8
SIG_K = {"route": 64, "host": 16, "sni": 16, "dst": 16, "port": 8, "stack": 16}
CRIT = {"high": 2.0, "critical": 3.0, "normal": 1.0, "low": 0.5}
FITTERS = {"P06": (MP.PBOUNDS, 24.0), "P07": (MP.PGRAMMAR, 24.0), "P08": (MP.PBIND, 24.0),
           "P09": (MP.PWIN, 4.0), "P10": (MP.PFLOW, 4.0)}
COPY_MODELS = (MP.PTREE, MP.ATTR, MP.ATTRSEL, MP.PWANT, MP.PBOUNDS, MP.PGRAMMAR, MP.PBIND,
               MP.PWIN, MP.PFLOW)
HINT_SNAT = ("来源地址被代理转换，未配置可信代理，IP 不作为特征",
             "source addresses are translated by a proxy and no trusted proxy is configured; "
             "IP is not used as a feature")
HINT_OPAQUE = ("流量加密，内容约束（语法、绑定）不可用", "traffic is encrypted; content constraints "
               "(grammar, bindings) are unavailable")


def _dv(n: int, hl: float = PS.H_M) -> PS.DecayedVector:
    return PS.DecayedVector([hl] * n)


def _ss(k: int, hl: float = PS.H_M) -> PS.DecayedSpaceSaving:
    return PS.DecayedSpaceSaving(k, [hl], [hl], 0)


DAY_COUNTS_MAX = 8192                       # IPs with a per-day evidence count (beyond: counted as new)


class WhoCode:
    """System-level prequential two-part code of the source IPs at each who
    level (/32, /24, /16, grp, reg; §6.18.2), the full-information measurement
    of the who dimension. For an IP whose events of the day so far carry n
    and now c more evidence units (a day's c events of one IP count H(c),
    harmonic: a burst is not c observations), at level l with item g:

        c x addr_bits(g)                                   the item's own address space
      + [g unseen]  -log2(alpha / (N + alpha)) + id_bits   escape + naming the item
      + Dirichlet block  -log2 Gamma(n+c)Gamma(N+alpha) / (Gamma(n)Gamma(N+alpha+c))

    with n, N from a SpaceSaving per level (H_l-decayed, k bounded: memory is
    independent of the population; an evicted item pays its escape again).
    addr_bits: 0 for /32, 8 for /24, 16 for /16, log2 |group| (32 without a
    group), log2 |region| (32 outside every region); id_bits 32 / 24 / 16 /
    log2(#groups + 1) / log2(#regions + 1); the 'none' arm is 32 bits.
    (P04's per-leaf sums charge an IP's introduction once per LEAF it visits,
    so they penalise fine levels by the tree's own splits; the system-level
    code answers P12's question, and is the one PG8 compares against.)"""

    K = (128, 128, 64, 64, 32)
    ALPHA = 1.0
    # behaviour given who (held-out gain of each level, the who utility of
    # lib/pstrategy.who_utilities): the same pair budget at every level, so a
    # level whose (item, behaviour) statistics do not fit is charged by its
    # evictions (gains compared at equal memory)
    KB_PAIR = 256
    KB_ITEM = 128
    KB_MARG = 256
    BEH_TICK = 384             # distinct (ip, behaviour) pairs coded per tick (bottom-k by IP hash)
    BDAY_MAX = 4096
    BETA = 1.0                 # prior weight of the marginal in p(b | item)
    NB_ESC = 65536.0           # alphabet size of an unseen behaviour (escape)

    def __init__(self) -> None:
        self.ss = [PS.DecayedSpaceSaving(k, [PS.H_L], [PS.H_L], 0) for k in self.K]
        self.day: Optional[int] = None
        self.day_counts: Dict[str, float] = {}
        self.cur = np.zeros(5)
        self.cur_ev = 0.0
        self.ring: List[Tuple[int, List[float], float]] = []
        self._beh_init()

    def _beh_init(self) -> None:
        self.bp = [PS.DecayedSpaceSaving(self.KB_PAIR, [PS.H_L], [], 0) for _ in range(5)]
        self.bi = [PS.DecayedSpaceSaving(self.KB_ITEM, [PS.H_L], [], 0) for _ in range(5)]
        self.bm = PS.DecayedSpaceSaving(self.KB_MARG, [PS.H_L], [], 0)
        self.bday: Dict[Tuple[str, Any], float] = {}
        self.cur_b = np.zeros(5)          # bits of behaviour coded given each level's item
        self.cur_bm = 0.0                 # bits of behaviour coded by the marginal
        self.cur_bev = 0.0
        self.bring: List[Tuple[int, List[float], float, float]] = []

    def roll(self, day: int) -> None:
        if not hasattr(self, "bp"):
            self._beh_init()                      # a tracker pickled before the behaviour code
        if self.day is None:
            self.day = day
        if day > self.day:
            self.ring.append((self.day, [float(x) for x in self.cur], float(self.cur_ev)))
            self.ring = self.ring[-7:]
            self.bring.append((self.day, [float(x) for x in self.cur_b], float(self.cur_bm),
                               float(self.cur_bev)))
            self.bring = self.bring[-7:]
            self.cur = np.zeros(5)
            self.cur_ev = 0.0
            self.cur_b = np.zeros(5)
            self.cur_bm = 0.0
            self.cur_bev = 0.0
            self.day_counts = {}
            self.bday = {}
            self.day = day

    @staticmethod
    def items(ip: str, ip2g: Mapping[str, Any], gsize: Mapping[Any, int], regions: Any
              ) -> Tuple[str, str, str, str, str]:
        """The who item of an address at each level (/32, /24, /16, grp, reg)."""
        g = ip2g.get(ip)
        return (ip, ip_prefix(ip, 1), ip_prefix(ip, 2), GRP_NONE if g is None else f"grp:{g}",
                regions.of(ip) if len(regions) else REG_NONE)

    def observe_beh(self, counts: Mapping[Tuple[str, Any], float], t: float, day: int,
                    ip2g: Mapping[str, Any], gsize: Mapping[Any, int], regions: Any,
                    rsize: Mapping[str, float]) -> None:
        """Prequential code of the behaviour b = (action, local hour) of learned
        events, by the marginal predictive and given the who item at each level:

            p_m(b)     = (n_b + alpha / NB_ESC) / (M + alpha)
            p(b | g_l) = (n_{g,b} + BETA p_m(b)) / (n_g + BETA)

        every block coded BEFORE it is learned (held out by construction); an
        (ip, b) pair's c events of a day count H(c) evidence units (harmonic, as
        the who code). n_{g,b} is the Space-Saving guaranteed count (an evicted
        or new pair falls back to the marginal: no gain, no loss). Gain of level
        l = (bits_marginal - bits_l) / evidence."""
        self.roll(day)
        a, beta = self.ALPHA, self.BETA
        Mt = self.bm.total(t)
        ln2 = math.log(2.0)
        keys = sorted(counts, key=lambda x: (_iphash(x[0]), x[0], str(x[1])))
        if len(keys) > self.BEH_TICK:
            # bottom-k by a hash of the SOURCE: a sampled address is sampled with
            # all its behaviours and, under a steady load, at every tick (its
            # per-item statistics are learned, not fragments of them); the gain
            # is a per-event average, which a source sample estimates unbiasedly
            keys = keys[: self.BEH_TICK]
        for (ip, bk) in keys:
            c0 = float(counts[(ip, bk)])
            n0 = self.bday.get((ip, bk), 0.0)
            n1 = n0 + c0
            if len(self.bday) < self.BDAY_MAX or (ip, bk) in self.bday:
                self.bday[(ip, bk)] = n1
            c = _harm(n1) - _harm(n0)
            if c <= 0:
                continue
            pm = (self.bm.guaranteed(bk, t) + a / self.NB_ESC) / (Mt + a)
            self.cur_bm += -c * math.log(pm) / ln2
            its = self.items(ip, ip2g, gsize, regions)
            for l in range(5):
                g = its[l]
                ng = self.bi[l].count(g, t) if g in self.bi[l] else 0.0
                ngb = min(ng, self.bp[l].guaranteed((g, bk), t)) if ng > 0 else 0.0
                p = (ngb + beta * pm) / (ng + beta)
                self.cur_b[l] += -c * math.log(p) / ln2
                self.bp[l].add((g, bk), t, c)
                self.bi[l].add(g, t, c)
            self.bm.add(bk, t, c)
            Mt += c
            self.cur_bev += c

    def day_values(self) -> Optional[Tuple[List[float], float, Optional[List[float]], float]]:
        """The last COMPLETED day's (who bits/event, evidence, behaviour gain
        per level, behaviour evidence): one non-overlapping round for Hedge
        (the 7-day sums re-count each day seven times). None before a day ended."""
        if not self.ring:
            return None
        d, b, e = self.ring[-1]
        if e <= 0:
            return None
        bits = [float(x) / e for x in b]
        gain, bev = None, 0.0
        if getattr(self, "bring", None):
            dd, cb, cm, ce = self.bring[-1]
            if dd == d and ce > 0:
                gain, bev = [float((cm - x) / ce) for x in cb], float(ce)
        return bits, float(e), gain, bev

    def beh_gain(self) -> Tuple[Optional[List[float]], float]:
        """Held-out behaviour gain per level (bits/event) over the last 7 days + today."""
        if not hasattr(self, "bp"):
            return None, 0.0
        tot = self.cur_b.copy()
        m = float(self.cur_bm)
        ev = float(self.cur_bev)
        for _, b, bm, e in self.bring:
            tot += np.asarray(b)
            m += bm
            ev += e
        if ev <= 0:
            return None, 0.0
        return [float((m - x) / ev) for x in tot], float(ev)

    def observe(self, counts: Mapping[str, float], t: float, day: int, ip2g: Mapping[str, Any],
                gsize: Mapping[Any, int], regions: Any, rsize: Mapping[str, float]) -> None:
        self.roll(day)
        n_groups = len(gsize)
        n_reg = len(rsize)
        a = self.ALPHA
        Ns = [ss.total(t) for ss in self.ss]
        idg, idr = math.log2(n_groups + 1.0), math.log2(n_reg + 1.0)
        ln2 = math.log(2)
        for ip, c0 in counts.items():
            n0 = self.day_counts.get(ip, 0.0)
            n1 = n0 + float(c0)
            if len(self.day_counts) < DAY_COUNTS_MAX or ip in self.day_counts:
                self.day_counts[ip] = n1
            c = _harm(n1) - _harm(n0)
            if c <= 0:
                continue
            v6 = ":" in ip
            keys, addr, idb = [], [], []
            keys.append(ip); addr.append(0.0); idb.append(128.0 if v6 else 32.0)
            keys.append(ip_prefix(ip, 1)); addr.append(64.0 if v6 else 8.0); idb.append(64.0 if v6 else 24.0)
            keys.append(ip_prefix(ip, 2)); addr.append(80.0 if v6 else 16.0); idb.append(48.0 if v6 else 16.0)
            g = ip2g.get(ip)
            if g is None:
                keys.append(GRP_NONE); addr.append(32.0)
            else:
                keys.append(f"grp:{g}"); addr.append(math.log2(max(1, gsize.get(g, 1))))
            idb.append(idg)
            r = regions.of(ip) if len(regions) else REG_NONE
            keys.append(r); addr.append(32.0 if r == REG_NONE else math.log2(max(1.0, rsize.get(r, 1.0))))
            idb.append(idr)
            for l in range(5):
                ss = self.ss[l]
                key = keys[l]
                N = Ns[l]
                n = ss.count(key, t) if key in ss else 0.0
                bits = c * addr[l]
                rem = c
                if n <= 1e-12:
                    w = min(1.0, rem)
                    bits += w * (math.log2((N + a) / a) + idb[l])
                    n, N, rem = w, N + w, rem - w
                if rem > 0:
                    bits += -(math.lgamma(n + rem) - math.lgamma(n) - math.lgamma(N + a + rem)
                              + math.lgamma(N + a)) / ln2
                ss.add(key, t, c, c)
                Ns[l] += c
                self.cur[l] += bits
            self.cur_ev += c

    def bits(self) -> Tuple[Optional[List[float]], float]:
        tot = self.cur.copy()
        ev = self.cur_ev
        for _, b, e in self.ring:
            tot += np.asarray(b)
            ev += e
        if ev <= 0:
            return None, 0.0
        return [float(x) for x in tot / ev], float(ev)

    def nbytes(self) -> int:
        n = int(sum(x.nbytes() for x in self.ss) + 24 * len(self.day_counts) + 400)
        if hasattr(self, "bp"):
            n += int(sum(x.nbytes() for x in self.bp) + sum(x.nbytes() for x in self.bi)
                     + self.bm.nbytes() + 48 * len(self.bday))
        return n


def _iphash(ip: str) -> int:
    return zlib.crc32(ip.encode("utf-8"))


def _action_of(col: str, v: Any) -> Optional[str]:
    if col == "http.route":
        return str(v)
    if col == "tls.sni":
        h = PD.host_key(v)
        return ("TLS " + h) if h else None
    if col == "dns.qname":
        h = PD.host_key(v)
        return ("DNS " + h) if h else None
    return "DST " + str(v)


def behaviour_counts(b: EV.EventBatch, cb: Optional[EV.EventBatch]) -> Dict[Tuple[str, Any], float]:
    """(source IP, behaviour) counts of the batch's learned rows, behaviour =
    (action, workday?, local hour): the action as P10 names it (route template,
    else TLS / DNS host key, else destination), the hour and day type from P01's
    context batch. O(learned rows); the action is resolved per distinct value."""
    lr = b.learned_rows()
    if not len(lr):
        return {}
    n = int(b.n)
    act = np.full(n, None, dtype=object)
    for col in ("http.route", "tls.sni", "dns.qname", "net.dst"):
        c = b.cols.get(col)
        if c is None or not len(c.rows):
            continue
        free = act[c.rows] == None   # noqa: E711  (object array compare)
        if not free.any():
            continue
        rows = c.rows[free]
        vals = c.vals[free]
        memo: Dict[Any, Optional[str]] = {}
        out = np.empty(len(rows), dtype=object)
        for i, v in enumerate(vals.tolist()):
            if v is EV.ABSENT or v == EV.ABSENT:
                out[i] = None
                continue
            a = memo.get(v, memo)
            if a is memo:
                a = memo[v] = _action_of(col, v)
            out[i] = a
        act[rows] = out
    hour = np.full(n, -1, dtype=np.int64)
    wd = np.full(n, "", dtype=object)
    if cb is not None and cb.n == n:
        if cb.has("ctx.tod_min"):
            tm = cb.dense("ctx.tod_min", fill=np.nan)
            tm = np.asarray([x if isinstance(x, (int, float)) else np.nan for x in tm.tolist()],
                            dtype=np.float64)
            ok = np.isfinite(tm)
            hour[ok] = (tm[ok] // 60.0).astype(np.int64)
        if cb.has("ctx.daytype"):
            wd = cb.dense("ctx.daytype")
    out: Counter = Counter()
    ipi, ips = b.ip, b.ips
    for i in lr.tolist():
        a = act[i]
        if a is None:
            continue
        out[(ips[int(ipi[i])], (a, str(wd[i]) == "workday", int(hour[i])))] += 1.0
    return dict(out)


def _harm(n: float) -> float:
    """Harmonic number H(n) for real n >= 0 (digamma(n + 1) + gamma)."""
    if n <= 0:
        return 0.0
    if n < 64:
        k = int(n)
        h = sum(1.0 / j for j in range(1, k + 1))
        return h + (n - k) / (k + 1.0)
    return math.log(n) + 0.5772156649 + 1.0 / (2.0 * n)


class SysTracker:
    """Bounded per-system measurement state (module docstring). Memory is
    O(LRU_IPS + sketch sizes), independent of the number of IPs and attributes."""

    def __init__(self, system: str, t: float) -> None:
        self.system = system
        self.created = float(t)
        self.day: Optional[int] = None
        self.days_seen = 0
        self.hll_cum = PS.HLL(p=10)
        self.hll_day = PS.HLL(p=10)
        self.day_ring: List[Tuple[int, Any, float, float]] = []   # (day, HLL, mass, new_ips)
        self.new_ips: List[Tuple[int, float, float]] = []         # (day, new IPs, distinct IPs of the day)
        self.route_hll = PS.HLL(p=8)
        self.heaps: List[Tuple[float, float]] = []                # (events cum, routes cum)
        self.events_cum = 0.0
        self.lru: "OrderedDict[str, List[Any]]" = OrderedDict()   # ip -> [last_ts, {stack: n}, n, dhcp]
        self.gaps = _dv(GAP_BINS)
        self.chan = _dv(5)                                         # total, http, body|q, tls, sess
        self.dayt = _dv(2)                                         # total, non-workday
        self.dhcp = _dv(2)                                         # total, in dhcp scope
        self.sig = {g: _ss(k) for g, k in SIG_K.items()}
        self.heavy = _ss(8, PS.H_S)
        self.day_heavy = _ss(8, 1e12)                  # today's sources (no decay; reset daily)
        self.prev_top: Optional[Tuple[str, float]] = None   # (ip, share) of the previous full day
        self.behind: Dict[str, List[Any]] = {}        # dominant ip -> [HLL stacks today, max conc. sessions today]
        self.behind_prev: Dict[str, Tuple[float, float]] = {}
        self.ev_day = [0.0, 0.0]                                  # [mass, learned rows] of the current day
        self.ev_ring: List[Tuple[int, float, float]] = []          # (day, mass, learned)
        self.last_t: Optional[float] = None
        self.who = WhoCode()

    # ------------------------------------------------------------ day roll
    def roll(self, day: int) -> None:
        if self.day is None:
            self.day = day
            return
        if day <= self.day:
            return
        pop_before = self.hll_cum.count()
        merged = self.hll_cum.copy().merge(self.hll_day)
        new = max(0.0, merged.count() - pop_before)
        self.hll_cum = merged
        self.day_ring.append((self.day, self.hll_day, self.ev_day[0], new))
        self.day_ring = self.day_ring[-7:]
        distinct = float(self.hll_day.count())
        if self.days_seen >= 2 and distinct > 0:
            self.new_ips.append((self.day, new, distinct))
            self.new_ips = self.new_ips[-7:]
        self.ev_ring.append((self.day, self.ev_day[0], self.ev_day[1]))
        self.ev_ring = self.ev_ring[-14:]
        self.events_cum += self.ev_day[1]
        self.heaps.append((self.events_cum, self.route_hll.count()))
        self.heaps = self.heaps[-60:]
        self.behind_prev = {ip: (float(v[0].count()), float(v[1])) for ip, v in self.behind.items()}
        self.prev_top = self._top_share(self.day_heavy, None) if self.day == day - 1 else None
        self.day_heavy = _ss(8, 1e12)
        self.behind = {}
        self.hll_day = PS.HLL(p=10)
        self.ev_day = [0.0, 0.0]
        self.days_seen += 1
        self.day = day

    def population(self) -> float:
        """Distinct source IPs over the last 7 days (incl. today)."""
        acc = PS.HLL(p=10)
        for _, h, _, _ in self.day_ring[-6:]:
            acc.merge(h)
        acc.merge(self.hll_day)
        return float(acc.count())

    def churn(self) -> Optional[float]:
        """Mean over the last 7 days of (IPs never seen before) / (distinct IPs
        of the day): ~0 for a stable department, ~1 for one-shot public
        sources. (Relative to the 7-day population, as §6.18.1 reads, it could
        not exceed 1/7 in steady state even for fully random sources.)"""
        if len(self.new_ips) < 2:
            return None
        v = [n / p for _, n, p in self.new_ips if p > 0]
        return float(np.mean(v)) if v else None

    # -------------------------------------------------------------- update
    def observe(self, b: EV.EventBatch, cb: Optional[EV.EventBatch], now: float, day: int,
                dhcp: Sequence[Any], who_ctx: Optional[Tuple[Any, ...]] = None) -> None:
        self.roll(day)
        self.last_t = float(now)
        n = int(b.n)
        if n == 0:
            return
        if who_ctx is not None:
            cnt = np.bincount(b.ip, minlength=len(b.ips))
            self.who.observe({b.ips[j]: float(cnt[j]) for j in np.flatnonzero(cnt > 0).tolist()},
                             float(now), day, *who_ctx)
            bc = behaviour_counts(b, cb)
            if bc:
                self.who.observe_beh(bc, float(now), day, *who_ctx)
        for ip in b.ips:
            self.hll_day.add(ip)
        w = np.asarray(b.w, dtype=np.float64)
        lr = b.learned_rows()
        mass = b.mass()
        self.ev_day[0] += float(w.sum())
        self.ev_day[1] += float(len(lr))
        t = float(now)
        # channel masses on learned rows (HT mass: unbiased)
        if len(lr):
            m = mass[lr]
            tot = float(m.sum())
            ch = b.dense("ev.ch")[lr]
            http = float(m[ch == "http"].sum())
            tls = float(m[ch == "tls"].sum())
            bq = np.zeros(n, dtype=bool)
            for nm, c in b.cols.items():
                if nm.startswith("body.") or nm.startswith("q."):
                    bq[c.rows] = True
            body = float(m[bq[lr]].sum())
            sk = b.cols.get("sess.key")
            sess = 0.0
            if sk is not None:
                has = np.zeros(n, dtype=bool)
                has[sk.rows] = True
                sess = float(m[has[lr]].sum())
            self.chan.add(t, np.array([tot, http, body, tls, sess]))
            if cb is not None and cb.n == n and cb.has("ctx.daytype"):
                dt = cb.dense("ctx.daytype")[lr]
                self.dayt.add(t, np.array([tot, float(m[dt == "nonworkday"].sum())]))
            # signature sketches (aggregated per distinct value first)
            for g, col, fn in (("route", "http.route", PF.route_prefix), ("host", "http.host", None),
                               ("sni", "tls.sni", PF.etld1), ("dst", "net.dst", None),
                               ("port", "net.dport", None), ("stack", "client.stack", None)):
                c = b.cols.get(col)
                if c is None:
                    continue
                agg: Counter = Counter()
                pos = np.full(n, -1, dtype=np.int64)
                pos[c.rows] = np.arange(len(c.rows))
                for i in lr[pos[lr] >= 0].tolist():
                    v = c.vals[pos[i]]
                    if isinstance(v, np.floating):
                        v = int(v) if float(v).is_integer() else float(v)
                    agg[fn(v) if fn is not None else v] += float(mass[i])
                    if g == "route":
                        self.route_hll.add(v)
                for v, mv in agg.items():
                    self.sig[g].add(v, t, mv, 1.0)
        # heavy sources (all rows, raw mass), stacks / session keys behind them
        ipw = np.bincount(b.ip, weights=w, minlength=len(b.ips))
        for j in np.flatnonzero(ipw > 0).tolist():
            self.heavy.add(b.ips[j], t, float(ipw[j]), 1.0)
            self.day_heavy.add(b.ips[j], t, float(ipw[j]), 1.0)
        # what is behind the dominant sources: distinct client stacks (per day) and
        # concurrent session keys (distinct keys per 15 min of the tick) - one user
        # has one stack and few concurrent sessions, a translating proxy has many
        top = [k for k, _, _, _ in self.heavy.items(t)[:2]]
        if top:
            for ip in top:
                try:
                    j = b.ips.index(ip)
                except ValueError:
                    continue
                rows_ip = np.flatnonzero(b.ip == j)
                ent = self.behind.get(ip)
                if ent is None:
                    ent = self.behind[ip] = [PS.HLL(p=6), 0.0]
                c = b.cols.get("client.stack")
                if c is not None:
                    for v in set(c.vals[np.isin(c.rows, rows_ip)].tolist()):
                        ent[0].add(v)
                c = b.cols.get("sess.key")
                if c is not None:
                    mask = np.isin(c.rows, rows_ip)
                    ent[1] = max(ent[1], _max_overlap(c.vals[mask], np.asarray(b.ts)[c.rows[mask]]))
            if len(self.behind) > 4:
                for k in [k for k in self.behind if k not in top][: len(self.behind) - 4]:
                    self.behind.pop(k, None)
        # per-IP gaps and stack concentration (all rows; LRU of LRU_IPS sources)
        order = np.lexsort((b.ts, b.ip))
        ips_s, ts_s = b.ip[order], np.asarray(b.ts, dtype=np.float64)[order]
        same = ips_s[1:] == ips_s[:-1]
        gaps = list(np.diff(ts_s)[same])
        first = np.concatenate(([True], ~same))
        last = np.concatenate((~same, [True]))
        stk = b.cols.get("client.stack")
        spos = None
        if stk is not None:
            spos = np.full(n, -1, dtype=np.int64)
            spos[stk.rows] = np.arange(len(stk.rows))
        stacks_by_ip: Dict[int, Counter] = {}
        if spos is not None:
            for i in np.flatnonzero(spos >= 0).tolist():
                stacks_by_ip.setdefault(int(b.ip[i]), Counter())[stk.vals[spos[i]]] += 1
        cnt = np.bincount(b.ip, minlength=len(b.ips))
        dmass = [0.0, 0.0]
        for k in np.flatnonzero(first).tolist():
            j = int(ips_s[k])
            ip = b.ips[j]
            ent = self.lru.get(ip)
            if ent is not None:
                if ent[0] is not None and ts_s[k] > ent[0]:
                    gaps.append(float(ts_s[k] - ent[0]))
                self.lru.move_to_end(ip)
            else:
                ent = [None, Counter(), 0, _in_dhcp(ip, dhcp)]
                self.lru[ip] = ent
                if len(self.lru) > LRU_IPS:
                    self.lru.popitem(last=False)
            ent[2] += int(cnt[j])
            sc = stacks_by_ip.get(j)
            if sc:
                ent[1].update(sc)
                if len(ent[1]) > 4:
                    ent[1] = Counter(dict(ent[1].most_common(4)))
            dmass[0] += float(ipw[j])
            if ent[3]:
                dmass[1] += float(ipw[j])
        for k in np.flatnonzero(last).tolist():
            ent = self.lru.get(b.ips[int(ips_s[k])])
            if ent is not None:
                ent[0] = float(ts_s[k])
        if dhcp:
            self.dhcp.add(t, np.array(dmass))
        g = np.asarray([x for x in gaps if x > 0], dtype=np.float64)
        if g.size:
            h, _ = np.histogram(np.clip(np.log10(g), GAP_LO, GAP_HI - 1e-9), bins=GAP_BINS,
                                range=(GAP_LO, GAP_HI))
            self.gaps.add(t, h.astype(np.float64))

    # --------------------------------------------------------------- reads
    @staticmethod
    def _top_share(ss: Any, t: Optional[float]) -> Optional[Tuple[str, float]]:
        tot = ss.total(t)
        it = ss.items(t)
        if tot <= 0 or not it:
            return None
        return str(it[0][0]), float(it[0][1] / tot)

    def snat_suspect(self, t: float) -> Tuple[bool, Optional[str]]:
        """§5.1.4: one source carries >= 90 % of the traffic over a full day (the
        previous local day AND today so far) with many users behind it (>= 10
        client stacks in a day or >= 5 sessions open at the same time - one user
        opens many short sessions a day, but few at once). A one-day burst of
        one IP (a bulk export) is not a proxy."""
        cur = self._top_share(self.day_heavy, t)
        if self.prev_top is None or cur is None or self.prev_top[0] != cur[0]:
            return False, None
        ip0 = cur[0]
        if min(self.prev_top[1], cur[1]) < SNAT_SHARE:
            return False, None
        ent = self.behind.get(ip0)
        prev = self.behind_prev.get(ip0, (0.0, 0.0))
        stacks = max(prev[0], ent[0].count() if ent else 0.0)
        conc = max(prev[1], ent[1] if ent else 0.0)
        return (stacks >= SNAT_STACKS or conc >= SNAT_CONC), ip0

    def signature_groups(self, t: float) -> Dict[str, Dict[Any, float]]:
        out: Dict[str, Dict[Any, float]] = {}
        for g, ss in self.sig.items():
            if g == "dst":
                continue
            out[g] = {str(k): float(c) for k, c, _, _ in ss.items(t)}
        return out

    def stack_concentration(self) -> Optional[float]:
        vals = [max(e[1].values()) / sum(e[1].values()) >= STACK_CONC
                for e in self.lru.values() if e[1] and sum(e[1].values()) >= 5]
        return float(np.mean(vals)) if len(vals) >= 3 else None

    def gap_bimodality(self, t: float) -> Optional[float]:
        edges = np.linspace(GAP_LO, GAP_HI, GAP_BINS + 1)
        return PSt.bimodality_coefficient(self.gaps.read(t), 0.5 * (edges[1:] + edges[:-1]))

    def nbytes(self) -> int:
        b = self.hll_cum.nbytes() + self.hll_day.nbytes() + sum(h.nbytes() for _, h, _, _ in self.day_ring)
        b += sum(s.nbytes() for s in self.sig.values()) + self.heavy.nbytes()
        b += len(self.lru) * 160 + 3 * 8 * GAP_BINS + 2000 + self.who.nbytes()
        return int(b)


def _max_overlap(keys: np.ndarray, ts: np.ndarray) -> float:
    """Largest number of session keys whose [first, last] event intervals in
    the tick overlap at one instant (a sweep over interval ends)."""
    if len(keys) == 0:
        return 0.0
    span: Dict[Any, List[float]] = {}
    for k, t in zip(keys.tolist(), ts.tolist()):
        v = span.get(k)
        if v is None:
            span[k] = [t, t]
        else:
            v[0], v[1] = min(v[0], t), max(v[1], t)
    ev = sorted([(a, 1) for a, _ in span.values()] + [(b, -1) for _, b in span.values()],
                key=lambda x: (x[0], -x[1]))
    cur = best = 0
    for _, d in ev:
        cur += d
        best = max(best, cur)
    return float(best)


def _in_dhcp(ip: str, nets: Sequence[Any]) -> bool:
    if not nets:
        return False
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(a.version == n.version and a in n for n in nets)


def _dhcp_nets(config: Mapping[str, Any]) -> List[Any]:
    out = []
    for r in (config or {}).get("dhcp_scopes") or ():
        c = r.get("cidr") if isinstance(r, Mapping) else r
        try:
            out.append(ipaddress.ip_network(str(c), strict=False))
        except ValueError:
            continue
    return out


def _local_day(t: float, config: Mapping[str, Any]) -> int:
    return int(math.floor((float(t) + MP._tz_offset(config)) / DAY))


def _criticality(config: Mapping[str, Any], s: str) -> float:
    per = ((EV.pconfig(config).get("budget") or {}).get("per_system") or {}).get(s) or {}
    c = per.get("criticality") if isinstance(per, Mapping) else None
    if c is None:
        for r in (config or {}).get("ip_classes") or ():
            if isinstance(r, Mapping) and s in (r.get("systems") or ()) and r.get("criticality"):
                c = r["criticality"]
                break
    if isinstance(c, (int, float)):
        return float(c)
    return CRIT.get(str(c or "normal").lower(), 1.0)


# ======================================================================
class SystemProfileEngine(Engine):
    name = "behavior.system_profile"
    layer = "behavior"
    consumes = [EV.EVT_BATCH, EV.EVT_CTX, MP.ATTR, MP.ATTRSEL, MP.PTREE, MP.WHO_GROUPS, MP.PBOUNDS,
                MP.PGRAMMAR, MP.PBIND, MP.PFLOW, MP.BUDGET, "event.shared_ip", "labels"]
    produces = [MP.SYSPROF, MP.SYSFAM, STATE, "event.strategy_changed", "event.family_changed"]
    description = ("P12: system characteriser, scenario-adaptive strategy selector (Hedge / budgeted "
                   "UCB on bits/event net of cost) and system families")
    interval = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.day_period_s = float(params.get("day_period_s", PERIOD_S))
        self.last_stats: Dict[str, Any] = {}

    # ----------------------------------------------------------------- run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        if not EV.enabled(ctx.config):
            return 0
        store = ctx.store
        now = float(ctx.now)
        day = _local_day(now, ctx.config)
        dhcp = _dhcp_nets(ctx.config)
        n = 0
        # per tick: trackers of systems with a batch at this tick
        wctx = self._who_ctx(store, ctx.config)
        for s in store.batch_systems(EV.EVT_BATCH):
            b = store.batch_at(s, EV.EVT_BATCH, now)
            if b is None:
                continue
            tr = self._tracker(store, s, now)
            cb = store.batch_at(s, EV.EVT_CTX, now)
            tr.observe(b, cb, now, day, dhcp, wctx)
            n += 1
        # daily: families first (they decide the tree keys), then strategies
        stats: Dict[str, Any] = {}
        if self.entity_due(("p12fam",), now, self.day_period_s):
            stats["families"] = self.families(ctx, now, day)
        keys: Dict[str, List[str]] = {}
        for s in sorted(set(store.batch_systems(EV.EVT_BATCH))):
            keys.setdefault(MP.tree_key(store, s), []).append(s)
        for key, members in sorted(keys.items()):
            if not self._ready(store, members):
                continue
            if not self.entity_due(("p12", key), now, self.day_period_s):
                continue
            stats[key] = self.profile(ctx, key, members, now, day)
        self.last_stats = stats
        return n

    def _who_ctx(self, store: Any, config: Mapping[str, Any]) -> Tuple[Any, ...]:
        """(ip2g, group sizes, regions, region sizes) for the who code: P11's
        groups, config ip_classes / dhcp_scopes regions, else P11's covers.
        Cached while model.who_groups and the config are unchanged."""
        wg = MP.who_groups(store)
        ck = (id(wg), store.model_version(ORG, ORG, MP.WHO_GROUPS), id(config))
        hit = getattr(self, "_wctx", None)
        if hit is not None and hit[0] == ck:
            return hit[1]
        out = self._who_ctx_build(wg, config)
        self._wctx = (ck, out)
        return out

    @staticmethod
    def _who_ctx_build(wg: Mapping[str, Any], config: Mapping[str, Any]) -> Tuple[Any, ...]:
        ip2g = wg.get("ip2g") or {}
        gsize = Counter(ip2g.values())
        regions = Regions.from_config(config)
        if not len(regions) and wg.get("covers"):
            regions = Regions([(str(g), list(c)) for g, c in (wg.get("covers") or {}).items()])
        rsize: Dict[str, float] = {}
        for net, name in getattr(regions, "_nets", ()):
            rsize[f"reg:{name}"] = rsize.get(f"reg:{name}", 0.0) + float(net.num_addresses)
        return ip2g, gsize, regions, rsize

    @staticmethod
    def _ready(store: Any, members: Sequence[str]) -> bool:
        """A tree is profiled once one of its systems has a full local day of
        measurements (a decision on the first tick would rest on nothing)."""
        return any(isinstance(t, SysTracker) and t.days_seen >= MIN_DAYS
                   for t in (store.get_model(s, SYSTEM_ENTITY, STATE) for s in members))

    def _tracker(self, store: Any, s: str, now: float) -> SysTracker:
        tr = store.get_model(s, SYSTEM_ENTITY, STATE)
        if not isinstance(tr, SysTracker):
            tr = SysTracker(s, now)
            store.put_model(s, SYSTEM_ENTITY, STATE, tr, version=1, ts=now)
        return tr

    # ------------------------------------------------------ characteristics
    def characteristics(self, ctx: Context, key: str, members: Sequence[str], now: float
                        ) -> Dict[str, Any]:
        store, cfg = ctx.store, ctx.config
        trs = [t for t in (store.get_model(s, SYSTEM_ENTITY, STATE) for s in members)
               if isinstance(t, SysTracker)]
        days_seen = max([int(t.days_seen) for t in trs] or [0])
        reg = MP.get_registry(store, key)
        sel = MP.get_model(store, key, MP.ATTRSEL) or {}
        ptm = MP.get_ptree(store, key)
        ch: Dict[str, Any] = {"members": list(members)}
        pops = [t.population() for t in trs]
        ch["population"] = float(sum(pops))
        churns = [c for c in (t.churn() for t in trs) if c is not None]
        ch["churn"] = float(np.mean(churns)) if churns else None
        dm = sum((t.dhcp.read(now) for t in trs), np.zeros(2))
        ch["dhcp_share"] = float(dm[1] / dm[0]) if dm[0] > 0 else 0.0
        info = sel.get("ip_info") if isinstance(sel, Mapping) else None
        ch["ip_info"] = {int(k): float(v) for k, v in (info or {}).items()}
        # P05 evaluates a level only while it has <= 64 distinct values: /32 then
        # reads 0.0 on any system with > 64 IPs. Exactly 0 at /32 with
        # information at /24 is 'not evaluated', not 'no information'.
        if ch["ip_info"].get(0) == 0.0 and ch["ip_info"].get(1, 0.0) >= PSt.IP_INFO_MIN:
            ch["ip_info"].pop(0)
            ch["ip_info_32"] = "unevaluated"
        if ch["ip_info"]:
            lv = max(ch["ip_info"], key=ch["ip_info"].get)
            ch["ip_info_level"], ch["ip_info_best"] = lv, ch["ip_info"][lv]
        ch["who_mode"] = sel.get("who_mode") if isinstance(sel, Mapping) else None
        # snat suspect: one source carries >= 90 % of a day's mass with >= 50 stacks / session keys behind it
        snat, snat_ip = False, None
        for t in trs:
            sn, ip0 = t.snat_suspect(now)
            if sn:
                snat, snat_ip = True, ip0
        pc = EV.pconfig(cfg)
        if pc.get("trusted_proxies") and snat_ip is not None and _in_nets(snat_ip, pc["trusted_proxies"]):
            snat = False
        ch["snat"], ch["snat_ip"] = snat, snat_ip
        wg = MP.who_groups(store)
        shared = set(wg.get("shared") or ())
        # root who: population by levels, group / region coverage
        root = None
        if ptm is not None and EV.KIND_TXN in ptm.kinds:
            tr_ = ptm.kinds[EV.KIND_TXN]
            root = tr_.nodes.get(tr_.root)
        if root is not None:
            lv = root.who.levels
            tot0 = lv[0].total(now)
            ch["grp_cover"] = _cover(lv[3], 3, tot0, now)
            ch["reg_cover"] = _cover(lv[4], 4, tot0, now)
            ch["distinct_7d"] = float(root.who.hll.count(now))
            sh = sum(c for k, c, _, _ in lv[0].items(now) if str(k).startswith("shared:") or k in shared)
            ch["shared_share"] = float(sh / tot0) if tot0 > 0 else 0.0
        else:
            ch["grp_cover"] = ch["reg_cover"] = None
        # registry-derived characteristics
        if reg is not None:
            rr = reg.get("http.route")
            ch["route_card"] = float(rr.card_estimate()) if rr is not None else 0.0
            ch["stack_vis"] = max([reg.coverage(a, now) for a in ("client.stack", "tls.ja3", "http.ua")
                                   if a in reg] or [0.0])
            ch["sess_key_cov"] = reg.coverage("sess.key", now) if "sess.key" in reg else 0.0
            ch["numeric_targets"] = any(r.type == "numeric" for r in reg.records.values())
            meth = reg.get("http.method")
            ch["upload_routes"] = bool(meth is not None and any(
                str(k).upper() in ("POST", "PUT", "PATCH") for k, _, _, _ in meth.top.items(now)))
        else:
            ch.update(route_card=0.0, stack_vis=0.0, sess_key_cov=0.0, numeric_targets=False,
                      upload_routes=False)
        betas = [PSt.heaps_beta(t.heaps) for t in trs]
        betas = [b for b in betas if b is not None]
        ch["growth_beta"] = float(np.mean(betas)) if betas else None
        chan = sum((t.chan.read(now) for t in trs), np.zeros(5))
        tot = float(chan[0])
        age = max((now - t.created for t in trs), default=DAY)
        fill = max(1e-3, 1.0 - 2.0 ** (-max(age, 3600.0) / PS.H_M))     # young decayed sums
        ch["payload_vis"] = {"http": float(chan[1] / tot) if tot > 0 else 0.0,
                             "body_day": float(chan[2]) / fill * math.log(2) / PS.H_M * DAY,
                             "body": float(chan[2] / tot) if tot > 0 else 0.0,
                             "tls_opaque": float(max(0.0, chan[3] - chan[1]) / tot) if tot > 0 else 0.0}
        if ch["sess_key_cov"] == 0.0 and tot > 0:
            ch["sess_key_cov"] = float(chan[4] / tot)
        # session identifiability (§6.18.1)
        concs = [c for c in (t.stack_concentration() for t in trs) if c is not None]
        bcs = [x for x in (t.gap_bimodality(now) for t in trs) if x is not None]
        conc = float(np.mean(concs)) if concs else 0.0
        bc = float(np.mean(bcs)) if bcs else None
        pop = max(1.0, ch["population"])
        n_shared = len({e.entity for s in members for e in store.events(
            system=s, since=now - 7 * DAY, kinds=["shared_ip"], limit=1000)})
        no_nat = max(0.0, 1.0 - n_shared / pop) * (0.0 if snat else 1.0)
        bimodal = 1.0 if (bc is not None and bc > PSt.BC_BIMODAL) else 0.0
        ch["stack_conc"], ch["gap_bc"] = conc, bc
        ch["sess_ident"] = float(max(ch["sess_key_cov"], conc * no_nat * bimodal))
        # volume, automation, calendar
        evd = [x for t in trs for x in t.ev_ring[-7:]]
        days = max(1, len({d for d, _, _ in evd}))
        ev_day = sum(m for _, m, _ in evd) / days if evd else sum(t.ev_day[0] for t in trs)
        lr_day = sum(l for _, _, l in evd) / days if evd else sum(t.ev_day[1] for t in trs)
        ch["volume"] = {"events_day": float(ev_day), "learned_day": float(lr_day),
                        "learned_share": float(lr_day / ev_day) if ev_day > 0 else 0.0}
        per = []
        for t, s in ((t, t.system) for t in trs):
            for ip in list(t.lru.keys())[-64:]:
                v = store.latest_derived(s, ip, "derived.periodicity_score")
                if v is not None and isinstance(getattr(v, "value", None), (int, float)):
                    per.append(float(v.value) >= AUTOMATION_SCORE)
        ch["automation"] = float(np.mean(per)) if per else 0.0
        ch["periodic_ips"] = int(sum(per))
        dt = sum((t.dayt.read(now) for t in trs), np.zeros(2))
        ch["calendar"] = {"nonworkday_share": float(dt[1] / dt[0]) if dt[0] > 0 else 0.0,
                          "monthly": _monthly(sel)}
        fam = MP.get_org_model(store, MP.SYSFAM) or {}
        ch["family"] = key if key.startswith(PF.PREFIX) else None
        ch["criticality"] = max(_criticality(cfg, s) for s in members) if members else 1.0
        ch["win_events"] = any(store.batch_times(s, EV.EVT_WIN) for s in members)
        ch["vetoed"] = self._veto(store, members, now)
        ch["tracker_bytes"] = int(sum(t.nbytes() for t in trs))
        return ch

    def _veto(self, store: Any, members: Sequence[str], now: float) -> List[str]:
        counts: Dict[str, List[int]] = {}
        for s in members:
            for lb in store.labels(system=s, since=now - 30 * DAY):
                if lb.verdict not in ("tp", "fp") or lb.target_type != "event":
                    continue
                ev = store.get_event(lb.target_id)
                if ev is None or not ev.p_by_detector:
                    continue
                det = min(ev.p_by_detector, key=lambda d: ev.p_by_detector[d])
                try:
                    fam = DET.family_of(det)
                except Exception:
                    continue
                c = counts.setdefault(fam, [0, 0])
                c[0 if lb.verdict == "tp" else 1] += 1
        return PSt.label_veto({f: (c[0], c[1]) for f, c in counts.items()})

    # ------------------------------------------------------- measurements
    def measurements(self, ctx: Context, key: str, members: Sequence[str], ch: Mapping[str, Any],
                     now: float) -> Dict[str, Any]:
        store = ctx.store
        ptm = MP.get_ptree(store, key)
        meas: Dict[str, Any] = {"who": None, "who_n": 0.0, "n_nodes": 0.0}
        if ptm is not None:
            meas["n_nodes"] = float(sum(len(t.nodes) for t in ptm.kinds.values()))
            tr = ptm.kinds.get(EV.KIND_TXN)
            if tr is not None:
                bits, n = who_level_bits(tr, now)
                meas["who_tree"], meas["who_tree_n"] = bits, n
        trs = [t for t in (store.get_model(s, SYSTEM_ENTITY, STATE) for s in members)
               if isinstance(t, SysTracker)]
        days_seen = max([int(t.days_seen) for t in trs] or [0])
        tot, ev = np.zeros(5), 0.0
        for t in trs:
            b, e = t.who.bits()
            if b is not None:
                tot += np.asarray(b) * e
                ev += e
        if ev > 0:
            meas["who"], meas["who_n"] = [float(x) for x in tot / ev], float(ev)
        # held-out behaviour gain of each who level (evidence-weighted over members)
        gtot, gev = np.zeros(5), 0.0
        for t in trs:
            g, e = t.who.beh_gain()
            if g is not None:
                gtot += np.asarray(g) * e
                gev += e
        if gev > 0:
            meas["who_pred"], meas["who_pred_n"] = [float(x) for x in gtot / gev], float(gev)
        # the last completed day alone (one Hedge round, evidence-weighted over members)
        db, dg, dbe, dge = np.zeros(5), np.zeros(5), 0.0, 0.0
        for t in trs:
            dv = t.who.day_values()
            if dv is None:
                continue
            bits, e, gain, ge = dv
            db += np.asarray(bits) * e
            dbe += e
            if gain is not None and ge > 0:
                dg += np.asarray(gain) * ge
                dge += ge
        if dbe > 0:
            meas["who_day"], meas["who_day_n"] = [float(x) for x in db / dbe], float(dbe)
        if dge > 0:
            meas["who_pred_day"], meas["who_pred_day_n"] = [float(x) for x in dg / dge], float(dge)
        vol = ch.get("volume") or {}
        ev_day = float(vol.get("events_day", 0.0) or 0.0)
        ld = float(vol.get("learned_day", 0.0) or 0.0)
        # evidence per day ~ learned rows per day (<= 1 unit each)
        meas["ev_day"] = ld
        eng_cost = _engine_costs(store, now)
        bud = MP.get_org_model(store, MP.BUDGET) or {}
        usage = (bud.get("usage") or {}) if isinstance(bud, Mapping) else {}
        share = (bud.get("budget") or {}).get("pcore_cpu_share") if isinstance(bud, Mapping) else None
        if usage.get("pcore_cpu_share") is not None and share:
            meas["cpu_usage_share"] = float(usage["pcore_cpu_share"]) / float(share)
        for dim, (name, runs) in FITTERS.items():
            m = MP.get_model(store, key, name)
            g = (m or {}).get("gain") if isinstance(m, Mapping) else None
            if not isinstance(g, Mapping) or g.get("bits_per_event") is None:
                continue
            ts = [m.get(k) for k in ("updated", "last_run", "last_mine")]
            ts = [float(x) for x in ts if isinstance(x, (int, float))]
            if not ts or now - max(ts) > FRESH_S:
                continue                                  # not run lately (arm off): unmeasured
            if dim in NODE_GAIN and ptm is not None:
                # per-node records only: the fitter's own figure is 0 whenever no
                # node was refitted. Nothing judged YET is not a gain of 0 while the
                # system is young (JUDGE_WAIT_DAYS: a daily user needs ~5 workdays for
                # the n_bind events of a binding); after that, nothing judged means
                # nothing to bind (a portal of one-off visitors) and the gain is 0
                gain = fitted_gain(m, ptm, now)
                if gain is None:
                    if days_seen < JUDGE_WAIT_DAYS or (dim == "P08" and bindings_pending(m)):
                        continue
                    gain = 0.0
            else:                                         # no tree: the fitter's own figure
                gain = float(g["bits_per_event"])
            cost = eng_cost.get(ENGINE_OF[dim])
            if cost is None:
                ms = g.get("ms")
                cost = (float(ms) * 1000.0 * runs / max(ev_day, 1.0)) if (ms is not None and ev_day > 0) \
                    else g.get("us_per_event")
            cost = min(float(cost or 0.0), COST_CAP_US)
            meas[dim] = {"gain": float(gain), "cost": cost}
        meas["n_earned"] = self._n_earned(store, members)
        return meas

    @staticmethod
    def _n_earned(store: Any, members: Sequence[str]) -> int:
        b = MP.get_org_model(store, MP.BUDGET) or {}
        cands = 0
        for s in members:
            rec = ((b.get("systems") or {}).get(s) or {}) if isinstance(b, Mapping) else {}
            cands += int(rec.get("n_earned_candidates", len(rec.get("earned") or ())))
        return cands

    # ------------------------------------------------------------ profile
    def profile(self, ctx: Context, key: str, members: Sequence[str], now: float, day: int
                ) -> Dict[str, Any]:
        store = ctx.store
        ch = self.characteristics(ctx, key, members, now)
        meas = self.measurements(ctx, key, members, ch, now)
        old = MP.get_model(store, key, MP.SYSPROF)
        state = copy.deepcopy((old or {}).get("state")) if isinstance(old, Mapping) else None
        if not isinstance(state, dict):
            state = PSt.new_state()
        bud = MP.get_org_model(store, MP.BUDGET) or {}
        explore = not bool((bud.get("ladder") or {}).get("no_explore")) if isinstance(bud, Mapping) else True
        dec = PSt.decide(state, ch, meas, day, key, explore=explore)
        hints = []
        if ch.get("snat"):
            hints.append({"kind": "snat", "text_zh": HINT_SNAT[0], "text_en": HINT_SNAT[1],
                          "ip": ch.get("snat_ip")})
        pv = ch.get("payload_vis") or {}
        if float(pv.get("tls_opaque", 0.0)) >= 0.5 and float(pv.get("body", 0.0)) < PSt.PAYLOAD_MIN:
            hints.append({"kind": "opaque", "text_zh": HINT_OPAQUE[0], "text_en": HINT_OPAQUE[1]})
        version = int((old or {}).get("version", 0)) + 1 if isinstance(old, Mapping) else 1
        model = {"fmt": 1, "version": version, "t": now, "day": day, "tree_key": key,
                 "members": list(members), "characteristics": _jsonable(ch),
                 "measurements": _jsonable(meas), "arms": _jsonable(dec["arms"]),
                 "chosen": dict(dec["chosen"]), "reasons": dict(dec["reasons"]),
                 "probe": list(dec["probe"]), "hints": hints, "lambda": dec.get("lambda"),
                 "history": list(state.get("history") or []), "state": state}
        store.put_model(key, SYSTEM_ENTITY, MP.SYSPROF, model, version=version, ts=now)
        for s in members:
            if s != key:
                store.put_model(s, SYSTEM_ENTITY, MP.SYSPROF, model, version=version, ts=now)
        for dim, a, b, why in dec["changed"]:
            for s in members:
                store.add_event(BehaviorEvent(
                    system=s, entity=SYSTEM_ENTITY, ts=now, kind="strategy_changed", score=0.0,
                    severity=Severity.INFO, description=f"strategy {dim}: {a} -> {b} ({why})",
                    extra={"dim": dim, "from": a, "to": b, "why": why, "tree_key": key},
                    dedupe_key=f"strategy_changed|{s}|{dim}|{day}"))
        return {"chosen": dict(dec["chosen"]), "changed": len(dec["changed"])}

    # ----------------------------------------------------------- families
    def families(self, ctx: Context, now: float, day: int) -> Dict[str, Any]:
        store, cfg = ctx.store, ctx.config
        fam = MP.get_org_model(store, MP.SYSFAM)
        fam = dict(fam) if isinstance(fam, Mapping) else {"version": 0, "member": {}, "families": {},
                                                          "dst_members": {}, "state": {}}
        state = fam.setdefault("state", {})
        sigs: Dict[str, Dict[str, Any]] = {}
        for s in store.batch_systems(EV.EVT_BATCH):
            tr = store.get_model(s, SYSTEM_ENTITY, STATE)
            if not isinstance(tr, SysTracker) or tr.last_t is None or now - tr.last_t > DAY:
                continue
            groups = tr.signature_groups(now)
            reg = MP.get_registry(store, MP.tree_key(store, s))
            if reg is not None:
                groups["name"] = {n: 1.0 for n in reg.names()}
            fs = PF.feature_set(groups)
            chan = tr.chan.read(now)
            sigs[s] = {"fs": fs, "sig": PF.weighted_minhash(fs), "informative": PF.informative(groups),
                       "payload_vis": float(chan[2] / chan[0]) if chan[0] > 0 else 0.0}
        shares = self._detach_shares(store, fam, now)
        old_member = dict(fam.get("member") or {})
        res = PF.update_families(state, day, sigs, EV.pconfig(cfg).get("system_families") or (), shares)
        member = res["member"]
        # tree adoption for new families, release of members' own trees
        for fid, mem in res["families"].items():
            if MP.get_ptree(store, fid) is None:
                seed = max(mem, key=lambda x: self._tree_weight(store, x, now))
                self._adopt(store, seed, fid, now)
        for s, fid in res["joined"]:
            if old_member.get(s) != fid:
                self._release(store, s, now)
        for s, fid, why in res["left"]:
            self._detach(store, s, fid, now)
        dst: Dict[str, Dict[str, str]] = {}
        for s, fid in member.items():
            tr = store.get_model(s, SYSTEM_ENTITY, STATE)
            if isinstance(tr, SysTracker):
                for k, _, _, _ in tr.sig["dst"].items(now):
                    dst.setdefault(fid, {})[str(k)] = s
        fam.update(version=int(fam.get("version", 0)) + 1, t=now, member=member,
                   families=res["families"], dst_members=dst, state=state,
                   last={"pairs": res["pairs"], "matched": res["matched"], "signatures": len(sigs)})
        store.put_model(ORG, ORG, MP.SYSFAM, fam, version=fam["version"], ts=now)
        for s, fid in res["joined"]:
            self._event(store, s, now, day, "joined", fid, "similar signature on 2 consecutive days")
        for s, fid, why in res["left"]:
            self._event(store, s, now, day, "left", fid, why)
        return {"families": {f: len(m) for f, m in res["families"].items()}, "joined": len(res["joined"]),
                "left": len(res["left"]), "pairs": res["pairs"]}

    @staticmethod
    def _event(store: Any, s: str, now: float, day: int, op: str, fid: str, why: str) -> None:
        store.add_event(BehaviorEvent(
            system=s, entity=SYSTEM_ENTITY, ts=now, kind="family_changed", score=0.0,
            severity=Severity.INFO, description=f"system {s} {op} family {fid}: {why}",
            extra={"op": op, "family": fid, "why": why}, dedupe_key=f"family_changed|{s}|{op}|{day}"))

    @staticmethod
    def _tree_weight(store: Any, s: str, now: float) -> float:
        m = MP.get_ptree(store, s)
        if m is None or EV.KIND_TXN not in m.kinds:
            return 0.0
        tr = m.kinds[EV.KIND_TXN]
        root = tr.nodes.get(tr.root)
        return float(root.n_c(now)) if root is not None else 0.0

    @staticmethod
    def _adopt(store: Any, seed: str, fid: str, now: float) -> None:
        """The family's tree, registry, selection, fitted models and strategy
        start as the seed member's (by reference; the seed's own key is
        released after)."""
        for name in COPY_MODELS + (MP.SYSPROF,):
            obj = store.get_model(seed, SYSTEM_ENTITY, name)
            if obj is None:
                continue
            if name == MP.PTREE:
                obj.tree_key = fid
            store.put_model(fid, SYSTEM_ENTITY, name, obj, ts=now)

    @staticmethod
    def _release(store: Any, s: str, now: float) -> None:
        """A system that joined a family no longer uses its own tree: keep a
        checkpoint (lineage, detach restore) and drop the in-memory models."""
        m = MP.get_ptree(store, s)
        if m is None:
            return
        store.put_checkpoint(s, SYSTEM_ENTITY, MP.CHECKPOINT, now, m)
        for name in COPY_MODELS:
            if store.get_model(s, SYSTEM_ENTITY, name) is not None:
                store.put_model(s, SYSTEM_ENTITY, name, None, ts=now)

    @staticmethod
    def _detach(store: Any, s: str, fid: str, now: float) -> None:
        """A leaving member gets a copy of the family's tree and registry
        (lineage kept; nodes of other members go stale and are pruned by P04)."""
        for name in (MP.PTREE, MP.ATTR, MP.ATTRSEL):
            obj = store.get_model(fid, SYSTEM_ENTITY, name)
            if obj is None:
                continue
            cp = copy.deepcopy(obj)
            if name == MP.PTREE:
                cp.tree_key = s
            store.put_model(s, SYSTEM_ENTITY, name, cp, ts=now)

    @staticmethod
    def _detach_shares(store: Any, fam: Mapping[str, Any], now: float) -> Dict[str, float]:
        """Per member: share of its family's txn nodes lying in a branch whose
        net.dst constraint admits only that member (§6.20 detach rule)."""
        out: Dict[str, float] = {}
        dstm = fam.get("dst_members") or {}
        for fid, mem in (fam.get("families") or {}).items():
            m = MP.get_ptree(store, fid)
            if m is None or EV.KIND_TXN not in m.kinds:
                continue
            tr = m.kinds[EV.KIND_TXN]
            total = max(1, len(tr.nodes))
            dmap = dstm.get(fid) or {}
            cnt: Counter = Counter()
            for nd in tr.nodes.values():
                owner = None
                for attr, level, vals, neg in nd.ctx:
                    if attr != "net.dst" or neg:
                        continue
                    who = {dmap.get(str(v), str(v)) if level == 0 else str(v) for v in vals}
                    if len(who) == 1:
                        owner = next(iter(who))
                if owner in mem:
                    cnt[owner] += 1
            for s in mem:
                out[s] = cnt.get(s, 0) / total
        return out


def _uncovered(level: int, key: Any) -> bool:
    """True for an item that is not a real group (level 3) / region (level 4):
    lib/phier maps an ungrouped IP to 'grp:∅', one outside every region to
    'reg:∅', and falls through to a coarser level ('*', or a region at the
    grp level) when the level has no model at all."""
    k = str(key)
    if level == 3:
        return not k.startswith("grp:") or k == GRP_NONE
    return k in (REG_NONE, STAR) or k.startswith("grp:")


def _none_share(ss: Any, level: int, t: float, evidence: bool) -> float:
    tot = ss.total_evidence(t, PS.EV_L) if evidence else ss.total(t)
    if tot <= 0:
        return 1.0
    none = 0.0
    for key, c, _, _ in ss.items(t):
        if _uncovered(level, key):
            none += ss.evidence(key, t, PS.EV_L) if evidence else c
    return float(min(1.0, none / tot))


def _cover(ss: Any, level: int, tot0: float, t: float) -> float:
    """Share of the node's mass whose IP has a real item at a grp / reg level."""
    if tot0 <= 0:
        return 0.0
    return float(max(0.0, 1.0 - _none_share(ss, level, t, False)) * min(1.0, ss.total(t) / tot0))


NODE_GAIN = ("P06", "P07", "P08")
JUDGE_WAIT_DAYS = 7                        # fitter records unjudged this long: nothing to fit (gain 0)
ENGINE_OF = {"P06": "behavior.content_bounds", "P07": "behavior.payload_grammar",
             "P08": "behavior.binding", "P09": "behavior.time_window", "P10": "behavior.workflow"}


def fitted_gain(model: Mapping[str, Any], ptm: Any, t: float) -> Optional[float]:
    """Bits per SYSTEM event a fitter's constraints save, from its per-node
    records (P06 / P07: attrs[a]['gain'], P08: pairs[p]['gain']; each the
    prequential saving at its node): sum over the deepest fitted node of every
    path of mass(node) x node gain, over the root's mass. (The fitters' own
    'gain' covers only the nodes refitted in their last run, 0 when none was
    dirty, which is not a measurement of the arm.) None when no record could be
    judged yet (no node fitted, or bindings without a source of n_bind events)."""
    if ptm is None or not isinstance(model, Mapping):
        return None
    nodes = model.get("nodes") or {}
    num = den = 0.0
    any_fit = any_rec = False
    for kind, tree in getattr(ptm, "kinds", {}).items():
        ents = nodes.get(kind) or nodes.get(str(kind)) or {}
        if not ents:
            continue
        root = tree.nodes.get(tree.root)
        if root is None:
            continue
        den += root.mass_at(t)
        gains: Dict[int, float] = {}
        for nid, ent in ents.items():
            nid = int(nid)
            if nid not in tree.nodes or not isinstance(ent, Mapping):
                continue
            # a binding record is a measurement once it judged a source (n_x >=
            # n_bind): before that its gain is 0 for lack of evidence, not of value
            recs = list((ent.get("attrs") or {}).values()) + \
                [r for r in (ent.get("pairs") or {}).values()
                 if isinstance(r, Mapping) and int(((r.get("fd") or {}).get("judged", 1)) or 0) > 0
                 and not _constant_pair(r)]
            recs = [r for r in recs if isinstance(r, Mapping)]
            if recs:
                any_rec = True
            g = sum(max(0.0, float(r.get("gain") or 0.0)) for r in recs)
            if g > 0:
                gains[nid] = g
        covered = set()
        for nid in gains:
            p = tree.nodes[nid].parent
            while p is not None and p in tree.nodes:
                covered.add(p)
                p = tree.nodes[p].parent
        for nid, g in gains.items():
            if nid not in covered:
                num += tree.nodes[nid].mass_at(t) * g
                any_fit = True
    if den <= 0 or not any_rec:
        return None                     # nothing judged yet: unmeasured
    return float(num / den) if any_fit else 0.0


def _pair_tops(rec: Mapping[str, Any], n_min: float = 0.0) -> List[str]:
    return [str(e.get("top")) for e in (rec.get("table") or {}).values()
            if isinstance(e, Mapping) and e.get("top") is not None and float(e.get("n") or 0.0) >= n_min]


def _constant_pair(rec: Mapping[str, Any]) -> bool:
    """A pair whose sources all hold the same value (body format, content
    type): a constant of the action, which binds nothing - its gain of 0 is
    no measurement of what bindings are worth (evaluator round 3: finance's
    'net.src -> body.fmt' record, judged with gain 0, switched P08 off on day
    10 while the three users' user-name pair was still gathering n_bind)."""
    tops = _pair_tops(rec)
    return len(tops) >= 2 and len(set(tops)) == 1


def bindings_pending(model: Any) -> bool:
    """True while a P08 pair that could bind is still gathering evidence: no
    source judged yet, but >= 2 RECURRING sources (>= 2 events each) holding
    different values (three users logging in once a day, before their 5th
    login). One-off visitors (n = 1 each) are not pending: nothing to bind."""
    if not isinstance(model, Mapping):
        return False
    for ents in (model.get("nodes") or {}).values():
        for ent in (ents or {}).values():
            for r in ((ent or {}).get("pairs") or {}).values() if isinstance(ent, Mapping) else ():
                if not isinstance(r, Mapping) or int(((r.get("fd") or {}).get("judged", 1)) or 0) > 0:
                    continue
                tops = _pair_tops(r, 2.0)
                if len(tops) >= 2 and len(set(tops)) >= 2:
                    return True
    return False


def _engine_costs(store: Any, now: float) -> Dict[str, float]:
    """µs per event of each engine over the last hour: P15's measured engine
    time (model.budget_state) over the org's events of that hour ({} before
    P15 ran)."""
    gs = store.get_model(ORG, ORG, "model.budget_state")
    em = getattr(gs, "engine_ms", None)
    rates = getattr(gs, "rate", None)
    if not em or not rates:
        return {}
    ev_h = 0.0
    for dv in rates.values():
        try:
            ev_h += float(dv.read(now)[0]) * math.log(2) / PS.H_S * 3600.0
        except Exception:
            continue
    if ev_h <= 0:
        return {}
    return {e: sum(x[1] for x in q) * 1000.0 / ev_h for e, q in em.items() if q}


def who_level_bits(tree: Any, t: float) -> Tuple[Optional[List[float]], float]:
    """Bits per event of the learned events' source IPs at each who level (/32,
    /24, /16, grp, reg), summed over every node's prequential who code (P04
    accumulates it at the leaf an event reached). A level whose key is missing
    for part of the events (an IP without a group or region) was coded as 0
    bits by P04; those events are charged the escape of a never-modelled
    address (32 bits) here, so every level is a complete code."""
    code = np.zeros(5)
    n = 0.0
    miss = np.zeros(5)
    for nd in tree.nodes.values():
        w = nd.who
        if w.code_n <= 0:
            continue
        code += np.asarray(w.code[:5], dtype=np.float64)
        n += float(w.code_n)
        t0 = w.levels[0].total_evidence(t, PS.EV_L)
        if t0 > 0:
            for l in (3, 4):
                ss = w.levels[l]
                cov = ss.total_evidence(t, PS.EV_L) * (1.0 - _none_share(ss, l, t, True))
                miss[l] += (1.0 - min(1.0, max(0.0, cov) / t0)) * float(w.code_n)
    if n <= 0:
        return None, 0.0
    bits = (code + miss * PSt.BITS_NONE) / n
    return [float(x) for x in bits], float(n)


def _monthly(sel: Any) -> Optional[float]:
    if not isinstance(sel, Mapping):
        return None
    us = sel.get("ustat") or {}
    vals = [((us.get(a) or {}).get("U_s")) for a in ("ctx.mend", "ctx.dom")]
    vals = [float(v) for v in vals if isinstance(v, (int, float))]
    return max(vals) if vals else None


def _in_nets(ip: str, cidrs: Iterable[str]) -> bool:
    try:
        a = ipaddress.ip_address(str(ip))
    except ValueError:
        return False
    for c in cidrs:
        try:
            n = ipaddress.ip_network(str(c), strict=False)
        except ValueError:
            continue
        if a.version == n.version and a in n:
            return True
    return False


def _jsonable(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {(k if isinstance(k, (str, int, float, bool)) or k is None else str(k)): _jsonable(v)
                for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x
