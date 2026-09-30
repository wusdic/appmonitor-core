"""P04 PatternTreeEngine (`behavior.pattern_tree`) — the progressive pattern lattice.

docs/lib3/progressive.md §6.5-§6.9, card P04. Library 3 (behaviour).

Requirement (S1-S4): not an enumeration of users and servers, but a model
that becomes more precise with observation time, from "a class of behaviour"
down to "one IP", and follows behaviour as it changes. Mechanism: one pattern
tree per (tree key, event kind) — a greedy, evidence-guided path through the
generalisation lattice of all contexts — grown only where evidence pays.

Per tick (learning is delayed by D = max(4 ticks, 600 s), §6.9.3):
  learn     every LEARNED row of the evt.batch / evt.win batches of tick
            t' <= t - D (with the aligned evt.ctx columns and P03's pat.assign
            `damp`), with mass m = w/pi x trust x damp and evidence
            omega = trust x damp / (r + 1) (burst run r per (ip, sess.key, leaf),
            §6.5.4; mass never enters evidence, PPC-9). Quarantined IPs' rows are
            held (<= 256 per IP, <= H_max per tree), learned on release with
            trust_prov, discarded on reject or after 7 d (model.control, B28).
  count     every node on the event's path: mass, evidence, who, when, dates;
            targets on the leaf and its parent always, on busier ancestors on a
            thinned stream (rho = min(1, r_target / rate_Hs), mass / rho).
  evidence  a learning leaf (<= L_max per tree) keeps pevalue.SplitStats over
            up to C = 6 candidates (P05's split candidates not constant at the
            leaf) and its targets U {who@l_w, when@slot}, with the leaf's
            prequential predictive (hierarchical Dirichlet backing off to the
            leaf's own summaries at learning start).
  split     every n_g = 32 evidence units: rules (V) anytime-valid averaged
            e-value >= 2^(tau0 + log2 C_ever), (G) MDL gain >= L_split, (S)
            stability, (D) diversity, (M) mass; value grouping (KT merge);
            children start as candidates, their categorical targets seeded from
            the split statistics. (S) runs in mode `split_s_mode` (config
            progressive.defaults, default 'margin': the empirical-Bernstein test
            applies between two VALID candidates and is settled by the tau0-bit
            MDL margin of §6.6 when it cannot separate them; see open issues).
  exception confirmed leaves track their heavy sources (share >= 0.02, <= 16,
            `shared:` excluded) with the leave-x-out e-value (pevalue.ExcTracker);
            an IP whose distribution differs gets its own node N@ip=x holding
            only the targets it differs on (saving >= 2 bits).
  drift     ADWIN on each confident leaf's clipped per-event log-loss
            (structural), Page-Hinkley on numeric targets and on the circular
            arrival minute (content / time). A change is ACCEPTED only when it
            persists T_persist normal days and is shown by
            >= max(2, ceil(0.5 |who top|)) IPs (a coordinated change) or, on a
            single-IP node, for 5 days; then the confidence channel of the
            changed attribute (whole node for structural) restarts from its
            H_m state, cver + 1, `pattern_drift`.
Daily (local date rollover; dirty work only):
  lifecycle candidate -> confirmed (n_c >= 20, >= 3 dates, fitters not pending)
            -> stable (7 d, 5 dates, no alarm) ; stale (expected arrivals >= 3 on
            normal active days with none observed, `pattern_absent`) ; retired
            after 30 d stale (dormant when its dates recur regularly, revived
            without novelty when its context comes back, `pattern_revived`).
  generalise prune (split saving < 0 on 3 checks), sibling merge (JSD < 0.02
            bits on 3 checks), splits on `gone` attributes collapse at once,
            EFDT revision (R_max internal nodes keep alternative candidates on
            the same events; an alternative that beats the current split by
            tau0 bits and passes (V) replaces it: e.g. a /24 split replaced by
            the learned group level `grp` once P11 has groups), budget N_max
            (lowest-utility leaves first, stale before anything else).
  exceptions tracked / decided / removed; invariants; reference snapshots
            at 04:00 local (P03's dual anchor).

Deviations from the text of §6.5-§6.9 (each measured or forced by the data; see
the implementation report):
  * rule (S) runs in 'margin' mode by default (pevalue.SplitStats.check): with
    spec-(S) the range term blocked every split of pack O's OA root (redundant
    valid rivals such as route vs method are always present).
  * prune / sibling merge use a two-part MDL estimate from the H_m summaries
    (sum_children n_c KL(child || parent) - parameter cost; mass-weighted JSD)
    at the daily check instead of a per-event prequential saving per internal
    node (the per-event form costs one extra predictive per ancestor and event).
  * EFDT revision runs only at internal nodes whose children are all leaves
    (its statistics compare two one-level splits and say nothing about the
    children's own subtrees; a root revision discarded a 20-node subtree on
    pack O), and compares only non-nested levels of the split attribute (IP /24
    -> grp / reg, value -> value group) and, for identity proxies published by
    P05 (`who_proxies`), the who levels; the replacement is immediate (the old
    subtree is retired, not kept scoring until the new children confirm).
    Unrestricted revision replaced pack O's route split by a client-stack split
    and discarded the route subtree daily.
  * content drift (Page-Hinkley per numeric target and per arrival minute)
    runs on leaves and exception nodes only (§6.5.1: an internal node's
    targets are a mixture of its children) and is fed ONE value per (node,
    detector, day type) and local day, the day's mean, with sigma = max(spread
    of past daily means, per-event sd / sqrt(n)); the history is winsorised
    at mu +- 3 sigma. Per-event PH on a time-ordered stream alarmed on every
    stationary node that mixes sources active at different hours (measured:
    5 of 7 confident nodes of pack O's OA tree `evolving` at day 14 without
    any drift); acceptance (T_persist days, coordinated IPs) is unchanged.
  * structural drift: one ADWIN per day type (weekday / weekend mixes are
    seasonality); a structural change is accepted when the leaf's daily mean
    loss stays >= 0.5 bit above its pre-alarm H_m level on 3 normal days (no
    Hoeffding-Adaptive-Tree alternate subtree is grown); split statistics restart
    on acceptance, not on the alarm. The Bernoulli CUSUM on "IP new to this node"
    is not implemented (who changes follow the who-closed rules of P03 / §6.9.2).
  * burst evidence is keyed (ip, leaf); the session key splits sources only
    behind a shared address (`shared:<ip>`), because per-request cookies (health
    checks) otherwise made every row an independent unit.
  * the @when target's bin width (15 min, 1 h, 4 h) is chosen per learning episode
    from the node's past arrivals; a root starts learning after one full day.
  * exceptions: targets that behave as a functional dependency of the source
    (bound user names) and P08's binding targets are masked from the e-value.
  * confirmation waits for a fitter only when its model lists the node under
    `pending`; model.ptree checkpoints (put_checkpoint) are not written.

Trust gating follows the lib-3 learner rule (lib/gating): a row's weight is
behavior.trust of its own tick, the hold decision is the quarantine gate at
t-1, so an IP quarantined during the learning delay D is held, not learned.
P04's working state (batch cursors, burst runs, held rows) lives in a private
attribute of the model object and is not part of the published pattern model.

Writes model.ptree@(key, '__system__') and the events pattern_confirmed,
pattern_retired, pattern_replaced, pattern_drift, pattern_absent,
pattern_revived. Reads evt.*, pat.assign, pat.rate, model.attr (P02),
model.attrsel (P05, bootstrap before its first run), model.pwant (P06-P10
requests: pairs, minute reservoirs, extra targets), model.pbounds / pgrammar /
pbind / pwin (confirmation gate, reference snapshots), model.budget (P15),
model.who_groups / sysfam (hierarchies), model.sysprof (who level), B28
trust / quarantine / model.control. Inert unless config['progressive']['enabled'].
"""
from __future__ import annotations

import datetime as _dt
import math
import zlib
from collections import Counter, deque
from typing import Any, Callable, Deque, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import SYSTEM_ENTITY, BehaviorEvent, Severity
from .lib import m_governor as MG
from .lib import m_ptree as MP
from .lib import pevalue as PE
from .lib import pevent as EV
from .lib import pmdl
from .lib import pnode as PN
from .lib import pselect as SEL
from .lib import psketch as PS
from .lib import ptree as PT
from .lib.combine import seeded_uniform
from .lib.phier import STAR

DAY = PS.DAY
N_G = 32.0                     # evidence units between split checks
N_G_MIN = 8.0                  # ... or at least daily once this many units arrived
LEARN_MIN = 16.0               # evidence a root needs before its bins / prior are chosen
                               # (and one full day: a daily cycle of values and arrival times;
                               # rule (D) needs two dates before any split anyway)
R_LEARN_S = 14 * DAY           # restart split statistics after 14 d ...
R_LEARN_UNITS = 2000.0         # ... or 2 000 units without an accepted split
L_MAX = 64                     # learning leaves per tree (tier M)
R_MAX = 16                     # internal nodes with revision statistics
REV_MIN_EV = 200.0             # n_eff (H_m) a node needs to keep revision statistics
C_MAX = PE.C_MAX
K_B = PE.K_B
K_V = PE.K_V
M_T = 8
R_TARGET = 1.0                 # events / s of H_s mass above which ancestors thin targets
N_CONF = 20.0
CONF_DATES = {EV.KIND_TXN: 3, EV.KIND_WIN: 2}
STABLE_S = 7 * DAY
STABLE_DATES = 5
STALE_E = 3.0
STALE_RETIRE_S = 30 * DAY
DORMANT_CV = 0.2
DORMANT_KEEP_S = 400 * DAY
PHI_X = 0.02
K_X = 16
EXC_SHARE = 0.10               # exceptions <= 10 % of N_max
EXC_SAVE_BITS = 2.0
EXC_N_CONF = 10.0
PRUNE_RUNS = 3
TAU_MERGE = 0.02
DAMP = 0.1
HELD_PER_IP = 256
HELD_MAX = 65536
HELD_EXPIRE_S = 7 * DAY
HELD_COLS = 96
ADWIN_DELTA = 0.002
LOSS_CLIP = 20.0
PH_MIN_N = 30.0
PH_MIN_DAYS = 5                # daily means a node needs before its Page-Hinkley tests run
DRIFT_EXPIRE_S = 14 * DAY
T_PERSIST_DAYS = {"num": 1, "when": 3, "structural": 3}
SINGLE_IP_DAYS = 5
SNAPSHOT_HOUR = 4
ORDINAL0 = 719163              # date(1970, 1, 1).toordinal()
WHO_LEVEL = {"ip": 0, "prefix": 1, "grp": 3, "reg": 4}
PSEUDO_SOURCE = {"@who": "net.src", "@when": "ctx.when"}
FITTERS = (MP.PBOUNDS, MP.PGRAMMAR, MP.PBIND, MP.PWIN)
EVENT_KINDS = ("pattern_confirmed", "pattern_retired", "pattern_replaced", "pattern_drift",
               "pattern_absent", "pattern_revived")


def _h(v: Any) -> Hashable:
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    try:
        hash(v)
        return v
    except TypeError:
        return repr(v)


def _local_day(ts: float, off: float) -> int:
    return int((float(ts) + off) // DAY) + ORDINAL0


def ctx_text(ctx: Sequence[Tuple[str, int, frozenset, bool]], hier: Any = None, max_vals: int = 4) -> str:
    """Readable context conjunction, e.g. 'http.route=POST /login ∧ net.src@grp∈{grp:3}'."""
    parts = []
    for a, l, vals, neg in ctx:
        lv = ""
        if hier is not None:
            try:
                lv = "" if l == 0 else "@" + hier.levels(a)[l]
            except Exception:
                lv = f"@{l}"
        vs = sorted(map(str, vals))
        txt = ",".join(vs[:max_vals]) + ("…" if len(vs) > max_vals else "")
        if neg:
            parts.append(f"{a}{lv}∉{{{txt}}}")
        elif len(vs) == 1:
            parts.append(f"{a}{lv}={txt}")
        else:
            parts.append(f"{a}{lv}∈{{{txt}}}")
    return " ∧ ".join(parts) if parts else "*"


# ================================================================== coder
def _stable_bucket(v: Hashable, k: int) -> int:
    """A process-independent hash bucket (Python's str hash is salted)."""
    return zlib.crc32(repr(v).encode("utf-8", "replace")) % k


class Coder:
    """Binning and prequential predictive of a leaf's targets (§6.5.3): per
    target k_b bins (the leaf's top values at learning start, then first come,
    the last bin = other), evidence counts since start and a prior p0 from the
    node's own summaries at start (a predictable backing-off distribution).
    p(b) = (lc[b] + alpha p0[b]) / (lc[.] + alpha).

    Flat high-cardinality targets are *hashed*: when the seeded top k_b - 1
    values cover < HASH_COVER of the node's mass, every value goes to one of
    k_b hash buckets instead (a fixed function, so the code stays prequential).
    Measured on pack O's OA login node: 29 usernames with no dominant value put
    ~70 % of the logins into `other`, so the target could not tell 综合部's two
    names from 销售部's twenty and no who split could pass rule (V)."""

    __slots__ = ("targets", "bmap", "p0", "lc", "hver", "wdiv", "hashed")

    def __init__(self, targets: Sequence[str], seeds: Sequence[Sequence[Tuple[Hashable, float]]],
                 hver: Tuple = (), wdiv: int = 15) -> None:
        self.wdiv = int(wdiv)                   # minutes per @when bin (15, 60 or 240)
        self.targets = list(targets)
        T = len(self.targets)
        self.bmap: List[Dict[Hashable, int]] = [dict() for _ in range(T)]
        self.p0 = np.full((T, K_B), 1.0 / K_B)
        self.lc = np.zeros((T, K_B))
        self.hver = tuple(hver)
        self.hashed = [False] * T
        for t, sd in enumerate(seeds):
            sd = list(sd)
            top = sum(max(0.0, float(s)) for _, s in sd[:K_B - 1])
            if (len(sd) >= K_B - 1 and top < HASH_COVER and not self.targets[t].startswith("@")):
                self.hashed[t] = True
                sh = np.zeros(K_B)
                for v, s in sd:
                    sh[_stable_bucket(v, K_B)] += max(0.0, float(s))
                if sh.sum() > 0:
                    self.p0[t] = 0.9 * sh / sh.sum() + 0.1 / K_B
                continue
            sh = np.zeros(K_B)
            for v, s in sd[:K_B - 1]:
                j = len(self.bmap[t])
                self.bmap[t][v] = j
                sh[j] = max(0.0, float(s))
            rest = max(0.0, 1.0 - sh.sum())
            sh[K_B - 1] += rest
            if sh.sum() > 0:
                self.p0[t] = 0.9 * sh / sh.sum() + 0.1 / K_B

    def bins(self, values: Sequence[Any]) -> List[int]:
        out = []
        hashed = getattr(self, "hashed", None) or ()
        for t, v in enumerate(values):
            if v is None:
                out.append(-1)
                continue
            if t < len(hashed) and hashed[t]:
                out.append(_stable_bucket(v, K_B))
                continue
            bm = self.bmap[t]
            j = bm.get(v)
            if j is None:
                if len(bm) < K_B - 1:
                    j = bm[v] = len(bm)
                else:
                    j = K_B - 1
            out.append(j)
        return out

    def pred(self) -> np.ndarray:
        n = self.lc.sum(axis=1, keepdims=True)
        return (self.lc + PE.ALPHA * self.p0) / (n + PE.ALPHA)

    def add(self, bins: Sequence[int], w: float) -> None:
        for t, b in enumerate(bins):
            if b >= 0:
                self.lc[t, b] += w

    def value_of(self, t: int, b: int) -> Optional[Hashable]:
        hashed = getattr(self, "hashed", None) or ()
        if t < len(hashed) and hashed[t]:
            return None                            # a bucket names no single value
        for v, j in self.bmap[t].items():
            if j == b:
                return v
        return None

    def nbytes(self) -> int:
        return int(self.p0.nbytes + self.lc.nbytes + 64 * sum(len(b) for b in self.bmap) + 100)


# ============================================================ learn context
class _LC:
    """Everything one tree needs during a tick (read once per tick)."""

    def __init__(self, engine: "PatternTreeEngine", store: Any, key: str, now: float,
                 config: Mapping[str, Any], off: float) -> None:
        self.eng = engine
        self.store = store
        self.key = key
        self.now = now
        self.config = config
        self.off = off
        self.reg = MP.get_registry(store, key)
        self.hier = MP.hierarchies(store, key, config, self.reg)
        self.budget = MP.budget_for(store, key)
        self.gone: Set[str] = {n for n, r in self.reg.records.items() if r.state == "gone"} \
            if self.reg is not None else set()
        pc = EV.pconfig(config)
        d = pc.get("defaults") or {}
        self.s_mode = str(d.get("split_s_mode", "margin"))
        self.tau0 = float(d.get("tau0", PE.TAU0))
        self.sel: Dict[int, Dict[str, Any]] = {}
        sp = MP.get_model(store, key, MP.SYSPROF)
        who = ((sp or {}).get("chosen") or {}).get("who") if isinstance(sp, Mapping) else None
        self.who_level: Optional[int] = WHO_LEVEL.get(who, 0) if who != "none" else None
        wg = MP.who_groups(store)
        self.n_groups = len(wg.get("groups") or {}) or len(set((wg.get("ip2g") or {}).values()))
        gsize = Counter((wg.get("ip2g") or {}).values())
        self.gsize = gsize
        n_reg = len(self.hier.regions)
        self.space_bits = [0.0, 8.0, 16.0, 0.0, 8.0]
        self.escape_bits = [32.0, 24.0, 16.0, math.log2(self.n_groups + 1.0), math.log2(n_reg + 1.0)]
        self.pairs, self.minutes, self.extra_targets = _read_pwant(store, key)
        self.tinfo: Dict[str, Tuple[str, str, bool]] = {}
        self.whokeys: Dict[str, List[Any]] = {}
        self.trust: Dict[Tuple[str, str, float], Tuple[float, bool]] = {}
        self.n_learned = 0
        self.row_factor = 1.0
        self.m: Any = None
        self.aux: Dict[str, Any] = {}

    def selection(self, kind: int) -> Dict[str, Any]:
        s = self.sel.get(kind)
        if s is None:
            s = self.sel[kind] = SEL.selection_for(self.store, self.key, kind, self.reg)
        return s

    def kind_of(self, a: str) -> Tuple[str, str, bool]:
        r = self.tinfo.get(a)
        if r is None:
            rec = self.reg.get(a) if self.reg is not None else None
            typ = getattr(rec, "type", "unknown") if rec is not None else "unknown"
            k = "num" if typ in ("numeric", "time") else typ if typ in ("text", "set") else \
                "?" if typ == "unknown" else "cat"
            pol = getattr(rec, "policy", "clear") if rec is not None else "clear"
            lg = bool((getattr(rec, "hier", {}) or {}).get("log", False)) if rec is not None else False
            r = self.tinfo[a] = (k, pol, lg)
        return r

    def who_keys(self, ip: str) -> List[Any]:
        k = self.whokeys.get(ip)
        if k is None:
            h = self.hier
            k = self.whokeys[ip] = [h.gen("net.src", l, ip) for l in range(PN.WHO_LEVELS)]
            if len(self.whokeys) > 65536:
                self.whokeys.clear()
        return k


HASH_COVER = 0.8             # a target whose top k_b - 1 values cover less is hash-binned
ROUTE_ATTRS = frozenset({"http.route", "http.path", "tls.sni", "dns.qname"})


def _read_pwant(store: Any, key: str) -> Tuple[List[Tuple[str, int, str, Optional[Set[int]], Optional[int]]],
                                              Dict[int, Set[int]], Dict[Tuple[int, int], List[str]]]:
    """model.pwant: each fitter writes its own sub-key (§5.6). Accepted shapes:
      {'p08': {'pairs': [{'x': 'net.src', 'x_level': 0, 'y': 'body.kv.username',
                          'nodes': [nid...] | None, 'kind': 0}, ...]},
       'p09': {'minute_reservoir': [nid...] | {kind: [nid...]}},
       'p06': {'targets': {nid: [attr...]} | {kind: {nid: [attr...]}}}}
    Pair tuples (x, y) / (x, y, nodes) are accepted as well."""
    pw = MP.get_model(store, key, MP.PWANT)
    pairs: List[Tuple[str, int, str, Optional[Set[int]], Optional[int]]] = []
    minutes: Dict[int, Set[int]] = {}
    extra: Dict[Tuple[int, int], List[str]] = {}
    if not isinstance(pw, Mapping):
        return pairs, minutes, extra
    subs = [v for v in pw.values() if isinstance(v, Mapping)]
    if any(k in pw for k in ("pairs", "minute_reservoir", "targets")):
        subs.append(pw)
    def xname(x: str) -> Tuple[str, int]:
        if "@" in x:
            a, lv = x.rsplit("@", 1)
            try:
                return a, int(lv)
            except ValueError:
                return x, 0
        return x, 0
    for sub in subs:
        prs = sub.get("pairs")
        if isinstance(prs, Mapping):
            # P08's shape: {'fmt', 'updated', 'by_kind': {kind: {nid: [[X, Y], ...]}}}
            grouped: Dict[Tuple[str, int, str, int], Set[int]] = {}
            for kd, bynode in (prs.get("by_kind") or {}).items():
                for nid, lst in (bynode or {}).items():
                    for xy in lst or ():
                        if len(xy) >= 2:
                            x, lv = xname(str(xy[0]))
                            grouped.setdefault((x, lv, str(xy[1]), int(kd)), set()).add(int(nid))
            for (x, lv, y, kd), nodes in grouped.items():
                pairs.append((x, lv, y, nodes, kd))
            continue
        for p in prs or ():
            if isinstance(p, Mapping):
                x, y = p.get("x") or p.get("X"), p.get("y") or p.get("Y")
                lv = int(p.get("x_level", p.get("level", 0)) or 0)
                nodes = p.get("nodes")
                kind = p.get("kind")
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                x, y = p[0], p[1]
                lv, nodes, kind = 0, (p[2] if len(p) > 2 else None), None
            else:
                continue
            if x and y:
                x, lv2 = xname(str(x))
                pairs.append((x, lv or lv2, str(y), set(int(n) for n in nodes) if nodes else None,
                              None if kind is None else int(kind)))
        mr = sub.get("minute_reservoir")
        if isinstance(mr, Mapping):
            for k, v in mr.items():
                minutes.setdefault(int(k), set()).update(int(n) for n in v or ())
        elif mr:
            minutes.setdefault(EV.KIND_TXN, set()).update(int(n) for n in mr)
        tg = sub.get("targets")
        if isinstance(tg, Mapping):
            for k, v in tg.items():
                if isinstance(v, Mapping):
                    for nid, attrs in v.items():
                        extra.setdefault((int(k), int(nid)), []).extend(map(str, attrs))
                else:
                    extra.setdefault((EV.KIND_TXN, int(k)), []).extend(map(str, v or ()))
    return pairs[:64], minutes, extra


# ================================================================== engine
class PatternTreeEngine(Engine):
    name = "behavior.pattern_tree"
    layer = "behavior"
    consumes = [EV.EVT_BATCH, EV.EVT_CTX, EV.EVT_WIN, EV.PAT_ASSIGN, EV.PAT_RATE, MP.ATTR, MP.ATTRSEL,
                MP.PWANT, MP.PBOUNDS, MP.PGRAMMAR, MP.PBIND, MP.PWIN, MP.BUDGET, MP.WHO_GROUPS,
                MP.SYSPROF, "behavior.trust", "behavior.quarantine", "model.control"]
    produces = [MP.PTREE] + ["event." + k for k in EVENT_KINDS]
    description = ("P04: progressive pattern lattice — evidence-driven specialisation / "
                   "generalisation, exceptions, budget, lifecycle, drift")

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.last_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------ helpers
    @staticmethod
    def aux(m: PT.PTreeModel) -> Dict[str, Any]:
        """P04's private per-tree working state (batch cursors, burst runs, held
        rows, learning / revision sets). Kept under a private attribute of the
        model object: it is not part of the published pattern model (plain-data
        copies of model.ptree skip it)."""
        a = getattr(m, "_paux", None)
        if a is None:
            a = m._paux = {"last": {}, "burst": PS.BurstEvidence(65536), "held": {}, "held_n": 0,
                         "day": None, "snap_day": None, "learning": {}, "revising": {},
                         "systems": set(), "seen": PS.LRU(65536), "n_alt": 0, "stats": Counter()}
        return a

    def _tree(self, lc: _LC, m: PT.PTreeModel, kind: int) -> PT.Tree:
        tr = m.tree(kind, lc.now)
        b = lc.budget
        if b.get("tier") in PT.TIERS and tr.budget.get("tier") != b["tier"]:
            tr.set_tier(b["tier"])
        elif b.get("n_max"):
            tr.budget["n_max"] = int(b["n_max"])
        return tr

    def _l_max(self, lc: _LC, tr: PT.Tree) -> int:
        if lc.budget.get("l_max") is not None:
            return int(lc.budget["l_max"])
        n_max = int(tr.budget.get("n_max", PT.TIERS["M"]))
        return max(1, min(L_MAX, n_max // 16)) if n_max < PT.TIERS["M"] else L_MAX

    # ---------------------------------------------------------------- run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        if not EV.enabled(ctx.config):
            return 0
        store = ctx.store
        now = float(ctx.now)
        dt = float(ctx.window_s)
        D = EV.learn_delay_s(dt, ctx.config)
        store.ensure_retention("pat.", max_age_s=D + 2.0 * dt)
        store.ensure_retention("evt.", max_age_s=D + 2.0 * dt)
        off = MP._tz_offset(ctx.config)
        systems = sorted(set(store.batch_systems(EV.EVT_BATCH)) | set(store.batch_systems(EV.EVT_WIN)))
        by_key: Dict[str, List[str]] = {}
        for s in systems:
            by_key.setdefault(MP.tree_key(store, s), []).append(s)
        n = 0
        stats = Counter()
        for key, ss in by_key.items():
            m = MP.ensure_ptree(store, key, now)
            aux = self.aux(m)
            lc = _LC(self, store, key, now, ctx.config, off)
            lc.m, lc.aux = m, aux
            for s in ss:
                aux["systems"].add(s)
                for kind, name in ((EV.KIND_TXN, EV.EVT_BATCH), (EV.KIND_WIN, EV.EVT_WIN)):
                    for ts_b, b in MP.learnable_batches(store, s, name, aux["last"].get((s, kind)), now, D):
                        n += self._learn_batch(lc, m, s, kind, ts_b, b)
                        aux["last"][(s, kind)] = ts_b
                self._rates(lc, m, s, aux, D)
                n += self._held(lc, m, s)
            self._maintain(lc, m, ss[0])
            stats.update(aux["stats"])
            aux["stats"] = Counter()
            MP.put_ptree(store, key, m, now)
        self.last_stats = dict(stats, learned=n)
        return n

    # ------------------------------------------------------------- batches
    def _trust(self, lc: _LC, s: str, ip: str, at: float) -> Tuple[float, bool]:
        """(trust of the row's tick, quarantined NOW). The lib-3 learner rule
        (lib/gating): the weight is behavior.trust of the row's own tick, the
        hold decision is the quarantine gate at t-1 (latest value before now;
        B28 runs after P04 in a tick) - an IP quarantined during the learning
        delay D is held, which is the purpose of learning late (§6.9.3)."""
        k = (s, ip, at)
        r = lc.trust.get(k)
        if r is None:
            q = MG.is_quarantined(lc.store, s, ip, lc.now)
            row = lc.store.vec_at(s, ip, MG.TRUST, float(at))
            if row is None:
                tr = 1.0                                    # governor not running for this key
            else:
                tr = float(np.asarray(row).reshape(-1)[0])
                tr = 0.0 if not tr == tr else min(1.0, max(0.0, tr))
            r = lc.trust[k] = (tr, q)
        return r

    def _learn_batch(self, lc: _LC, m: PT.PTreeModel, s: str, kind: int, ts_b: float,
                     b: EV.EventBatch) -> int:
        rr = b.learned_rows()
        if rr.size == 0:
            return 0
        tr = self._tree(lc, m, kind)
        cb = lc.store.batch_at(s, EV.EVT_CTX, ts_b) if kind == EV.KIND_TXN else None
        if cb is not None and cb.n != b.n:
            cb = None
        asg = lc.store.batch_at(s, EV.PAT_ASSIGN, ts_b)
        damp_col = None
        if asg is not None and asg.n == b.n and asg.has("damp"):
            damp_col = asg.dense("damp", 1.0)
        mass = b.mass()
        n = 0
        aux = self.aux(m)
        for i in rr.tolist():
            ip = b.ip_of(i)
            ts = float(b.ts[i])

            def get(nm: str, i: int = i, ip: str = ip) -> Any:
                v = b.get(nm, i)
                if v is EV.ABSENT and cb is not None:
                    v = cb.get(nm, i)
                if v is EV.ABSENT and nm == "net.src":
                    return ip
                return v
            trust, quar = self._trust(lc, s, ip, ts_b)
            if quar:
                self._hold(m, s, ip, ts, kind, b, cb, i, float(mass[i]))
                continue
            damp = 1.0
            if damp_col is not None:
                d = float(damp_col[i])
                damp = d if d == d else 1.0
            f = trust * damp
            if f <= 0.0:
                continue
            sess = b.get("sess.key", i, "∅")
            if ip in aux.get("watch", ()):                  # heavy sources of confident nodes only
                aux["seen"].put(ip, ts)
            self._learn_one(lc, tr, s, kind, get, ip, ts, float(mass[i]) * f, f, sess, b.rid[i])
            n += 1
        _expire_runs(aux["burst"], float(b.t0) - PS.TAU_BURST)
        return n

    # ---------------------------------------------------------- one event
    def _learn_one(self, lc: _LC, tr: PT.Tree, s: str, kind: int, get: Callable[[str], Any], ip: str,
                   ts: float, mass: float, factor: float, sess: Any, rid: Any = 0) -> None:
        hier = lc.hier
        path = tr.route(get, hier, lc.gone, ts)
        leaf = tr.nodes[path[-1]]
        if leaf.parent is not None and tr.dormant:
            rev = self._maybe_revive(lc, tr, leaf, get, ts, s)
            if rev:
                path = tr.route(get, hier, lc.gone, ts)
                leaf = tr.nodes[path[-1]]
        # burst runs: consecutive learned rows of one source on one leaf (§6.5.4); the
        # session key separates sources only behind a shared address (NAT, VDI),
        # an unshared IP's rows are one source whatever cookies they carry
        k0 = lc.who_keys(ip)[0]
        src = (ip, sess) if isinstance(k0, str) and k0.startswith("shared:") else (ip,)
        omega = lc.aux["burst"].unit(src + (leaf.id,), ts, factor)
        if omega <= 0.0:
            return
        lc.row_factor = float(factor)       # trust x damp: < 1 keeps the row out of hard ranges
        day = _local_day(ts, lc.off)
        dt_raw = get("ctx.daytype")
        daytype = None if dt_raw is EV.ABSENT else (0 if dt_raw in ("workday", "wd", 0) else 1)
        minute = get("ctx.tod_min")
        if minute is EV.ABSENT:
            minute = ((ts + lc.off) % DAY) / 60.0
            if daytype is None:
                daytype = 0 if ((int((ts + lc.off) // DAY) + 3) % 7) < 5 else 1
        minute = float(minute)
        if daytype is None:
            daytype = 0
        keys = lc.who_keys(ip)
        code = None
        if leaf.who.levels[0].total_evidence(ts) > 0:
            code = leaf.who.code_lengths(keys, ts, lc.space_bits, lc.escape_bits)
        sel = lc.selection(kind)
        nlast = len(path) - 1
        for d, nid in enumerate(path):
            nd = tr.nodes[nid]
            nd.update_core(ts, mass, omega, keys, ip, daytype, minute, day,
                           code if d == nlast else None)
            if nd.state == "stale":
                nd.state = "confirmed"
                nd.meta.pop("stale_at", None)
            if d >= nlast - 1:
                rho = 1.0
            else:
                rate = nd.rate_hs(ts)
                rho = 1.0 if rate <= R_TARGET else R_TARGET / rate
                if rho < 1.0 and seeded_uniform(lc.key, nid, ts, ip, rid) >= rho:
                    rev = nd.meta.get("R")
                    if rev is not None:
                        self._rev_update(lc, tr, nd, path[d + 1], get, ip, ts, omega, day, kind)
                    continue
            self._update_targets(lc, tr, nd, kind, sel, get, ts, mass / rho, omega, day)
            if d < nlast and nd.meta.get("R") is not None:
                self._rev_update(lc, tr, nd, path[d + 1], get, ip, ts, omega, day, kind)
        # exception node of the leaf's heavy source
        xid = leaf.exc.get(ip) if leaf.exc else None
        if xid is not None and xid in tr.nodes:
            xn = tr.nodes[xid]
            xn.update_core(ts, mass, omega, keys, ip, daytype, minute, day)
            self._update_targets(lc, tr, xn, kind, sel, get, ts, mass, omega, day)
        if not leaf.is_exc:
            self._leaf_learning(lc, tr, leaf, kind, sel, get, ip, ts, omega, day, daytype, minute)
        if lc.pairs:
            self._pairs(lc, tr, path, kind, get, ts, mass, omega)
        lc.n_learned += 1

    # --------------------------------------------------------- targets
    def _targets_of(self, lc: _LC, tr: PT.Tree, nd: PN.Node, kind: int, sel: Mapping[str, Any]) -> List[str]:
        own = nd.meta.get("targets")
        extra = lc.extra_targets.get((kind, nd.id)) if lc is not None else None
        if own is not None:
            # a node's frozen target list still takes the fitters' requests
            # (model.pwant, P06/P07: content attributes P05 gave only the split
            # role must still be fitted, §6.5.2 item 3)
            return list(own) + [a for a in (extra or ()) if a not in own]
        ovs = (sel.get("node_overrides") or {}).get(kind) or (sel.get("node_overrides") or {}).get(str(kind)) or {}
        ov = None
        cur: Optional[PN.Node] = nd
        while cur is not None and ov is None:           # nearest ancestor's override
            ov = ovs.get(cur.id)
            cur = tr.nodes.get(cur.parent) if cur.parent is not None else None
        base = list(ov) if ov else list((sel.get("targets_sys") or {}).get(kind) or [])[:M_T]
        if extra:
            base = base + [a for a in extra if a not in base]
        return base

    @staticmethod
    def _tvalue(lc: _LC, a: str, v: Any) -> Tuple[Any, str, str, bool]:
        k, pol, lg = lc.kind_of(a)
        return v, k, pol, lg

    def _update_targets(self, lc: _LC, tr: PT.Tree, nd: PN.Node, kind: int, sel: Mapping[str, Any],
                        get: Callable[[str], Any], ts: float, mass: float, omega: float, day: int) -> None:
        for a in self._targets_of(lc, tr, nd, kind, sel):
            k, pol, lg = lc.kind_of(a)
            if k == "?":
                continue                                   # not typed yet (P02)
            v = get(a)
            cur = nd.targets.get(a)
            if cur is not None and getattr(cur, "kind", k) != k:
                del nd.targets[a]                          # the registry re-typed it
            if v is EV.ABSENT:
                if k != "cat":
                    continue
            elif k == "cat":
                v = _h(v)
            tpl = None
            if k == "set" and v is not EV.ABSENT:
                tpl = lc.hier.gen(a, 1, v)
                if not isinstance(tpl, frozenset):
                    tpl = None
            try:
                nd.update_target(a, v, ts, mass, omega, k, pol, lg, day, tpl,
                                 lc.row_factor >= 0.999)
            except (TypeError, ValueError):
                continue
            # content drift detectors run on leaves (and exception nodes) only: an
            # internal node's targets are a mixture of its children, whose
            # composition changes are not content drift (§6.5.1)
            if k == "num" and nd.split is None and nd.state in PN.CONFIDENT_STATES:
                self._ph_num(lc, nd, a, v, ts, day, get("net.src"), get("ctx.daytype"))

    # ------------------------------------------------------ leaf learning
    def _coder_values(self, lc: _LC, coder: Coder, get: Callable[[str], Any], ip: str,
                      daytype: int, minute: float) -> List[Any]:
        out: List[Any] = []
        hier = lc.hier
        for a in coder.targets:
            if a == "@who":
                out.append(_h(lc.who_keys(ip)[lc.who_level or 0]))
            elif a == "@when":
                out.append((daytype, int(minute // coder.wdiv)))
            else:
                v = get(a)
                k = lc.kind_of(a)[0]
                if v is EV.ABSENT:
                    out.append(EV.ABSENT if k == "cat" else None)
                elif k == "num":
                    b = hier.gen(a, 1, v)
                    out.append(None if b is STAR else _h(b))
                elif k == "set":
                    out.append(_h(hier.gen(a, 1, v)))
                else:
                    out.append(_h(v))
        return out

    def _seed(self, lc: _LC, nd: PN.Node, a: str, t: float) -> List[Tuple[Hashable, float]]:
        """Top values and shares of a target at a node (its prior at learning start)."""
        src = nd
        if a == "@who":
            ss = src.who.levels[lc.who_level or 0]
            keys, sh, _ = ss.distribution(t)
            return sorted(zip(keys, sh.tolist()), key=lambda x: -x[1])
        if a == "@when":
            div = self._when_div(src)
            k = div // 15
            out = []
            tot = src.when.hist.sum()
            for d in (0, 1):
                if tot <= 0:
                    continue
                h = src.when.hist[d].reshape(-1, k).sum(axis=1)
                for j in np.argsort(-h)[:K_B]:
                    if h[j] > 0:
                        out.append(((d, int(j)), float(h[j] / tot)))
            return sorted(out, key=lambda x: -x[1])
        s = src.targets.get(a)
        if s is None:
            return []
        ss = getattr(s, "ss", None) or getattr(s, "values", None)
        if ss is not None and hasattr(ss, "distribution"):
            keys, sh, _ = ss.distribution(t)
            return sorted(zip([_h(k) for k in keys], sh.tolist()), key=lambda x: -x[1])
        return []

    @staticmethod
    def _hver(lc: _LC, targets: Sequence[str]) -> Tuple:
        """The value kinds of the coder's targets. A kind change (re-typing)
        changes what a bin means and rebuilds the coder; a refreshed numeric
        bin edge does not (every bin is still a function of the past only, so
        the prequential code and the e-value stay valid)."""
        return tuple("pseudo" if a.startswith("@") else lc.kind_of(a)[0] for a in targets)

    @staticmethod
    def _cver(lc: _LC, a: str) -> int:
        reg = lc.reg
        return int(getattr(reg.get(a), "version", 0)) if reg is not None and a in reg else 0

    @staticmethod
    def _when_div(nd: PN.Node) -> int:
        """Resolution of the @when target at a coder's start: the finest of 15-min
        slots, hours or 4-hour parts whose top k_b - 1 values cover >= 80 % of the
        node's arrival mass with >= 4 evidence units per occupied bin (a function of
        the past only, so the prequential code stays valid); 15 min when the node
        has no history yet."""
        h = nd.when.hist
        tot = h.sum()
        if tot <= 0:
            return 15
        t = nd.last_seen or 0.0
        ev = nd.when.evidence(0, t, conf=False) + nd.when.evidence(1, t, conf=False)
        for div in (15, 60, 240):
            g = h.reshape(2, -1, div // 15).sum(axis=2).ravel()
            top = np.sort(g)[::-1][:K_B - 1].sum()
            # a finer width only when the node saw >= 4 units per occupied bin: with
            # a handful of arrivals every width looks concentrated
            if top >= 0.8 * tot and ev >= 4.0 * int((g > 0).sum()):
                return div
        return 240

    def _new_coder(self, lc: _LC, nd: PN.Node, targets: Sequence[str], t: float,
                   tr: Optional[PT.Tree] = None) -> Coder:
        """Bins and prior from the node's own summaries, or from its parent's
        while the node has < LEARN_MIN evidence units (a new child)."""
        src = nd
        if nd.n_m(t) < LEARN_MIN and nd.parent is not None and tr is not None and nd.parent in tr.nodes:
            src = tr.nodes[nd.parent]
        seeds = [self._seed(lc, src, a, t) for a in targets]
        return Coder(targets, seeds, self._hver(lc, targets), self._when_div(src))

    def _coder_targets(self, lc: _LC, tr: PT.Tree, nd: PN.Node, kind: int, sel: Mapping[str, Any]) -> List[str]:
        tg = [a for a in self._targets_of(lc, tr, nd, kind, sel) if lc.kind_of(a)[0] != "?"][:M_T]
        if lc.who_level is not None:
            tg.append("@who")
        tg.append("@when")
        return tg

    @staticmethod
    def _revisable(lc: _LC, a1: str, l1: int, a2: str, l2: int) -> bool:
        """Revision alternatives of a split on (a1, l1): other, non-nested levels of
        the same attribute (IP /24 -> learned group or region; a categorical value
        -> its learned value group). A different attribute enters the tree by
        splitting below instead: a fully general revision replaced pack O's
        route split by a client-stack split and discarded the route subtree
        every day (measured), whereas re-coarsening the same attribute keeps
        the meaning of the node."""
        if a1 != a2 or l1 == l2:
            return False
        return not PatternTreeEngine._nested(lc, a1, l1, a2, l2) or lc.hier.kind(a1) == "cat"

    @staticmethod
    def _nested(lc: _LC, a1: str, l1: int, a2: str, l2: int) -> bool:
        if a1 != a2:
            return False
        if lc.hier.kind(a1) == "ip":
            return l1 <= 2 and l2 <= 2 or l1 == l2
        return True

    def _leaf_cands(self, lc: _LC, tr: PT.Tree, nd: PN.Node, kind: int, sel: Mapping[str, Any],
                    for_revision: bool = False) -> List[Tuple[str, int]]:
        if nd.depth + 1 > PT.D_MAX and not for_revision:
            return []
        hier = lc.hier
        out: List[Tuple[str, int]] = []
        t = lc.now
        sc = [(str(a), int(l)) for a, l in (sel.get("split_cands") or {}).get(kind) or []]
        const = nd.meta.get("const") or set()
        split_attrs = {a for a, _ in sc}
        # an attribute constrained in the context contributes its next finer levels
        # first (route group -> route prefix -> route template, §6.5.2 rule 2)
        finer: List[Tuple[str, int]] = []
        # the latest constraint per (attr, level): a value *group* ({3 /24s} or the
        # negated `other` bucket) can still be divided at its own level. Measured on
        # pack O: the OA root split put 192.168.1/2/3.0/24 into one group, after
        # which /24 was never offered again below it, so 综合部's, 财务部's and
        # 销售部's logins stayed one node for good.
        multi: Set[Tuple[str, int]] = set()
        for ca, cl, vals, neg in nd.ctx:
            if neg or len(vals) >= 2:
                multi.add((ca, cl))
            else:
                multi.discard((ca, cl))
        for ca, cl, vals, neg in nd.ctx:
            if ca in split_attrs and hier.kind(ca) != "ip":
                finer += [(ca, x) for x in ((cl,) if (ca, cl) in multi else ()) + (cl - 1, cl - 2)
                          if x >= 0]
        # slots are reserved by facet (§7.1 C = 6: who x2, when x2, route or content
        # x2): P05 ranks by system-wide U_s, where the many near-duplicate client /
        # content attributes would otherwise crowd out the who and when levels
        who = [(a, l) for a, l in sc if a in SEL.WHO_ATTRS][:2]
        when = [(a, l) for a, l in sc if a in SEL.WHEN_ATTRS][:2]
        # one slot is the route facet's (§7.1: "route or content x2"): ranked by
        # system-wide U_s the route family came third behind a size bin and a
        # user-agent shape on pack O's OA, so no tree ever split on the action
        # and every pattern mixed login, documents and approvals
        route = [(a, l) for a, l in sc if a in ROUTE_ATTRS or hier.kind(a) in ("route", "path")][:1]
        rest = [(a, l) for a, l in sc if (a, l) not in set(who) | set(when) | set(route)]
        seen: Set[Tuple[str, int]] = set()
        for a, l in finer + who + when + route + rest:
            if (a, l) in seen:
                continue
            seen.add((a, l))
            if a in lc.gone or a in nd.inv or (a, l) in const:
                continue
            ok = True
            ipk = hier.kind(a) == "ip"
            for ca, cl, vals, neg in nd.ctx:
                if ca != a:
                    continue
                same_ok = l == cl and (a, l) in multi   # a group divided at its own level
                if ipk:
                    # grp / reg are alternative coarsenings, not nested in the prefixes
                    if (l == cl and not same_ok) or (cl <= 2 and l <= 2 and l > cl):
                        ok = False
                elif l > cl or (l == cl and not same_ok):
                    ok = False                  # only levels finer than the constrained one
            if not ok:
                continue
            if a == "net.src" and l < len(nd.who.levels):
                ss = nd.who.levels[l]
                if ss.total_evidence(t) >= 5 and len(ss) == 1:
                    continue                    # constant at this leaf
            out.append((a, l))
            if len(out) >= C_MAX:
                break
        return out

    def _start_learning(self, lc: _LC, tr: PT.Tree, nd: PN.Node, kind: int, sel: Mapping[str, Any]) -> bool:
        cands = self._leaf_cands(lc, tr, nd, kind, sel)
        if not cands:
            return False
        coder = nd.meta.get("C")
        if coder is None:
            coder = nd.meta["C"] = self._new_coder(lc, nd, self._coder_targets(lc, tr, nd, kind, sel), lc.now, tr)
        ss = PE.SplitStats(len(coder.targets), k_b=K_B, k_v=K_V, C=C_MAX)
        for i, (a, l) in enumerate(cands):
            ss.set_candidate(i, (a, l), self._card(lc, a, l), self._tmask(coder, a))
        nd.split_stats = ss
        nd.meta["cands"] = cands
        nd.meta["lstart"] = lc.now
        self.aux_learning(lc, kind).add(nd.id)
        return True

    @staticmethod
    def _card(lc: _LC, a: str, l: int) -> float:
        """Number of values a split's value groups are named among (L_split and
        the value-grouping merge limit of §6.5.3 / §6.5.5). For IP levels the
        hierarchy's hint is the address space (2^32 at /32, 2^24 at /24), which
        charges 24-32 bits per named group: measured on pack O's OA login node,
        the three department /24s were merged into one group for weeks because
        each separate group had to repay 24 bits. A group only has to be named
        among the sources that exist, so an IP level's card is bounded by the
        registry's distinct-source estimate (HLL). The e-value test (V) does not
        depend on the grouping, so the false-split bound is unchanged."""
        c = float(lc.hier.card_hint(a, l))
        if lc.reg is not None and lc.hier.kind(a) == "ip":
            rec = lc.reg.get(a)
            try:
                est = float(rec.card_estimate()) if rec is not None else 0.0
            except Exception:
                est = 0.0
            if est > 0:
                c = min(c, max(2.0, est))
        return c

    @staticmethod
    def _tmask(coder: Coder, a: str) -> List[bool]:
        return [not SEL.same_source(a, PSEUDO_SOURCE.get(t, t)) for t in coder.targets]

    def aux_learning(self, lc: _LC, kind: int) -> Set[int]:
        return lc.aux["learning"].setdefault(kind, set())

    def _leaf_learning(self, lc: _LC, tr: PT.Tree, leaf: PN.Node, kind: int, sel: Mapping[str, Any],
                       get: Callable[[str], Any], ip: str, ts: float, omega: float, day: int,
                       daytype: int, minute: float) -> None:
        learning = self.aux_learning(lc, kind)
        if leaf.split_stats is None and len(learning) < self._l_max(lc, tr) \
                and not leaf.meta.get("no_learn") and (leaf.parent is not None or (
                    leaf.n_m(ts) >= LEARN_MIN and leaf.first_seen is not None
                    and ts - leaf.first_seen >= DAY)):
            self._start_learning(lc, tr, leaf, kind, sel)
        coder: Optional[Coder] = leaf.meta.get("C")
        confident = leaf.state in PN.CONFIDENT_STATES
        if coder is None:
            if not confident:
                return
            coder = leaf.meta["C"] = self._new_coder(lc, leaf, self._coder_targets(lc, tr, leaf, kind, sel), ts, tr)
        vals = self._coder_values(lc, coder, get, ip, daytype, minute)
        bins = coder.bins(vals)
        p = coder.pred()
        ss = leaf.split_stats
        if ss is not None:
            cands = leaf.meta.get("cands") or []
            cvals: List[Any] = [None] * ss.C
            for i, c in enumerate(cands):
                if c is not None:
                    cvals[i] = _h(lc.hier.gen(c[0], c[1], get(c[0])))
            ss.update(cvals, bins, p, omega, day)
            _note_value_days(leaf, ss, cvals, day)
        if confident:
            xs = leaf.xstats
            if xs is not None and xs.T == len(coder.targets):
                # an IP trivially "differs" in its own identity: the who target is masked
                xb = [(-1 if a == "@who" else b_) for a, b_ in zip(coder.targets, bins)]
                xs.update(ip, xb, p, omega, day)
            ls = [-math.log2(max(p[t, b], 1e-12)) for t, b in enumerate(bins) if b >= 0]
            if ls:
                # one ADWIN per day type: the weekday / weekend mix of a node changes
                # every week, which is seasonality, not drift
                loss = min(LOSS_CLIP, float(np.mean(ls)))
                if daytype:
                    ad = leaf.meta.get("adwin_nwd")
                    if ad is None:
                        ad = leaf.meta["adwin_nwd"] = PS.ADWIN(ADWIN_DELTA)
                else:
                    if leaf.adwin is None:
                        leaf.adwin = PS.ADWIN(ADWIN_DELTA)
                    ad = leaf.adwin
                dr = leaf.meta.get("drift")
                if dr is not None:
                    dr["dsum"][day] = dr["dsum"].get(day, 0.0) + loss
                    dr["dn"][day] = dr["dn"].get(day, 0) + 1
                else:
                    le = leaf.meta.get("loss_ew")
                    if le is None:
                        le = leaf.meta["loss_ew"] = PS.DecayedVector([PS.H_M, PS.H_M])
                    le.add(ts, [loss, 1.0])
                if ad.add(loss) > 0:
                    self._structural_alarm(lc, tr, leaf, ts)
            self._ph_when(lc, leaf, minute, ts, day, ip, daytype)
        coder.add(bins, omega)
        # every n_g units, and at least daily once N_G_MIN units arrived: (V) is
        # anytime-valid, so the cadence only bounds CPU; a daily check lets a
        # 3-login-a-day pattern be tested daily instead of every ten days
        if ss is not None and (ss.since_check >= N_G or (
                ss.since_check >= N_G_MIN and ts - float(leaf.meta.get("last_check", -math.inf)) >= DAY)):
            leaf.meta["last_check"] = ts
            self._try_split(lc, tr, leaf, kind, sel, ts)

    # ------------------------------------------------------------ split
    def _try_split(self, lc: _LC, tr: PT.Tree, leaf: PN.Node, kind: int, sel: Mapping[str, Any],
                   ts: float) -> bool:
        ss = leaf.split_stats
        cands = leaf.meta.get("cands") or []
        coder0: Optional[Coder] = leaf.meta.get("C")
        # the target set of a learning episode is fixed, except while it is young
        # (< 2 n_g units), when it has no real target yet (attributes not typed at
        # its start), or when P05's list has differed for 8 consecutive checks
        # (a young episode may adopt the current list once: a restart makes a
        # new young episode, which must not restart again, or it never ages)
        young = ss.total_evidence < 2.0 * N_G and not leaf.meta.get("young_rebuilt")
        # the target *set* matters, not P05's ranking order: its hourly h x cov order
        # of near-equal attributes flips, and every flip restarted young episodes
        differs = coder0 is not None and set(coder0.targets) != set(self._coder_targets(lc, tr, leaf, kind, sel))
        leaf.meta["tdiff"] = leaf.meta.get("tdiff", 0) + 1 if differs else 0
        no_real = coder0 is not None and not any(not a.startswith("@") for a in coder0.targets)
        if coder0 is not None and (coder0.hver != self._hver(lc, coder0.targets)
                                   or (differs and (young or no_real or leaf.meta["tdiff"] >= 8))):
            leaf.meta["young_rebuilt"] = bool(young) or leaf.meta.get("young_rebuilt", False)
            leaf.meta["tdiff"] = 0
            # a target was re-typed or its hierarchy (bins) refreshed: its bins
            # changed meaning, so the statistics restart (C_ever keeps counting)
            leaf.meta["C"] = None
            self._restart_learning(lc, tr, leaf, kind, sel)
            return False
        # a candidate whose hierarchy was refreshed (P02) is re-keyed: its statistics
        # restart, the other candidates keep theirs (§6.3)
        cv = leaf.meta.setdefault("cver", {})
        for i, c in enumerate(cands):
            if c is None:
                continue
            a, l = c
            v = self._cver(lc, a)
            if cv.get(a, v) != v and i < ss.C and ss.keys[i] is not None:
                ss.set_candidate(i, (a, l), self._card(lc, a, l), self._tmask(coder0, a))
            cv[a] = v
        dec = _check_valid_first(ss, lc.tau0, lc.s_mode)
        aux = lc.aux
        aux["stats"]["checks"] += 1
        if dec.accept and dec.best is not None and ss.keys[dec.best] is not None:
            if self._do_split(lc, tr, leaf, kind, dec, tuple(ss.keys[dec.best]), ts):
                return True
        # R_learn restart, candidate re-ranking (a replaced candidate starts empty)
        if ts - float(leaf.meta.get("lstart", ts)) >= R_LEARN_S or ss.total_evidence >= R_LEARN_UNITS:
            self._restart_learning(lc, tr, leaf, kind, sel)
            return False
        # a candidate that is constant at this leaf (one occupied value slot after
        # n_g units: e.g. a daypart level when the leaf's events all fall in one
        # daypart, a size bin that holds every login) can never split it; it
        # yields its slot to the next ranked candidate until the next restart
        const = leaf.meta.setdefault("const", set())
        for i in ss.active():
            if ss.n[i] >= N_G and sum(1 for v in ss.slot_val[i] if v is not None) <= 1:
                const.add(tuple(ss.keys[i]))
        # candidate re-ranking with hysteresis: a tracked candidate keeps its
        # statistics while P05 still proposes it (order changes are ignored); one
        # that left the list is replaced by the best new one and starts empty
        new = self._leaf_cands(lc, tr, leaf, kind, sel)
        cur = [tuple(c) if c is not None else None for c in cands] + [None] * (ss.C - len(cands))
        new_t = [tuple(c) for c in new]
        if set(x for x in cur if x is not None) != set(new_t):
            coder = leaf.meta.get("C")
            waiting = [c for c in new_t if c not in cur]
            for i in range(ss.C):
                have = cur[i]
                if have is not None and have in new_t:
                    continue
                want = waiting.pop(0) if waiting else None
                if want is None:
                    if have is not None:
                        ss.drop_candidate(i)
                    cur[i] = None
                else:
                    ss.set_candidate(i, want, self._card(lc, *want), self._tmask(coder, want[0]))
                    cur[i] = want
            leaf.meta["cands"] = cur
        return False

    def _restart_learning(self, lc: _LC, tr: PT.Tree, leaf: PN.Node, kind: int, sel: Mapping[str, Any]) -> None:
        """R_learn restart (§6.5.2): statistics restart from empty, C_ever and the
        candidate ordinals keep counting so repeated attempts pay. A changed
        target set or hierarchy version rebuilds the coder too."""
        st_ = lc.aux["stats"]
        st_["restarts"] = st_.get("restarts", 0) + 1
        tg = self._coder_targets(lc, tr, leaf, kind, sel)
        hv = self._hver(lc, tg)
        coder = leaf.meta.get("C")
        old = leaf.split_stats
        leaf.meta.pop("const", None)
        if coder is not None and set(coder.targets) == set(tg) and coder.hver == self._hver(lc, coder.targets) \
                and old is not None:
            old.restart()
            leaf.meta["lstart"] = lc.now
            return
        leaf.meta["C"] = self._new_coder(lc, leaf, tg, lc.now, tr)
        leaf.xstats = None
        leaf.split_stats = None
        self.aux_learning(lc, kind).discard(leaf.id)
        if self._start_learning(lc, tr, leaf, kind, sel) and old is not None:
            ss = leaf.split_stats
            for i in range(ss.C):
                if ss.keys[i] is not None:
                    ss.ordinal[i] += old.C_ever
            ss.C_ever += old.C_ever

    def _do_split(self, lc: _LC, tr: PT.Tree, leaf: PN.Node, kind: int, dec: PE.SplitDecision,
                  cand: Tuple[str, int], ts: float) -> bool:
        a, l = cand
        groups = [list(g) for gi, g in enumerate(dec.groups) if gi != dec.other_group and g]
        groups = [[v for v in g if v != PE.OTHER] for g in groups]
        # a named child must recur: at least one of its values was seen on >= 2 local
        # dates (else the "group" is one day, e.g. a day-of-month or a one-off burst,
        # and its events stay in `other`)
        sd = leaf.meta.get("sdays") or {}
        o = int(leaf.split_stats.ordinal[dec.best])
        groups = [g for g in groups if g and any(len(sd.get((o, v), ())) >= 2 for v in g)]
        if not groups or leaf.depth + 1 > PT.D_MAX:
            return False
        need = len(groups) + 1
        if len(tr.nodes) + need > int(tr.budget.get("n_max", PT.TIERS["M"])):
            self._make_room(lc, tr, need, protect={leaf.id})
            if len(tr.nodes) + need > int(tr.budget.get("n_max", PT.TIERS["M"])):
                return False
        ss = leaf.split_stats
        coder: Coder = leaf.meta.get("C")
        detail = {"attr": a, "level": int(l), "log2_e": round(dec.log2_e, 2),
                  "threshold": round(dec.threshold, 2), "gain": round(dec.gain, 2),
                  "l_split": round(dec.l_split, 2), "evidence": round(float(ss.n[dec.best]), 1),
                  "groups": len(groups)}
        sp = tr.split(leaf.id, a, l, groups, ts, detail)
        # seed categorical targets of the named children from the split statistics
        i = dec.best
        slot_of = ss.slot_of[i]
        ratio = leaf.mass_at(ts) / max(leaf.n_m(ts), 1e-9)
        for g, cid in zip(sp.groups, sp.children):
            child = tr.nodes[cid]
            js = [slot_of[v] for v in g if v in slot_of]
            if not js or coder is None:
                continue
            cnt = ss.cnt[i, js].sum(axis=0)          # [T, k_b]
            for t, name in enumerate(coder.targets):
                if name.startswith("@") or lc.kind_of(name)[0] != "cat":
                    continue
                for b in range(K_B - 1):
                    c = float(cnt[t, b])
                    if c <= 0:
                        continue
                    v = coder.value_of(t, b)
                    if v is None:
                        continue
                    k, pol, lg = lc.kind_of(name)
                    # mass = evidence x the leaf's mass / evidence ratio; the evidence
                    # of a seeded value is its split-statistics evidence
                    child.update_target(name, v, ts, c * ratio, c, k, pol, lg)
        leaf.split_stats = None
        leaf.meta.pop("cands", None)
        leaf.meta.pop("C", None)
        leaf.xstats = None
        leaf.adwin = None
        self.aux_learning(lc, kind).discard(leaf.id)
        lc.aux["stats"]["splits"] += 1
        return True

    # --------------------------------------------------------- revision
    def _rev_update(self, lc: _LC, tr: PT.Tree, nd: PN.Node, child_id: int, get: Callable[[str], Any],
                    ip: str, ts: float, omega: float, day: int, kind: int) -> None:
        R = nd.meta.get("R")
        if R is None or nd.split is None:
            return
        coder: Coder = R["coder"]
        dtr = get("ctx.daytype")
        daytype = 0 if dtr in ("workday", "wd", 0) else 1
        minute = get("ctx.tod_min")
        minute = float(minute) if minute is not EV.ABSENT else ((ts + lc.off) % DAY) / 60.0
        vals = self._coder_values(lc, coder, get, ip, daytype, minute)
        bins = coder.bins(vals)
        p = coder.pred()
        ss: PE.SplitStats = R["ss"]
        cvals: List[Any] = [None] * ss.C
        cvals[0] = int(child_id)
        for i, (a, l) in enumerate(R["cands"]):
            if i == 0:
                continue
            cvals[i] = _h(lc.hier.gen(a, l, get(a)))
        ss.update(cvals, bins, p, omega, day)
        coder.add(bins, omega)

    def _start_revision(self, lc: _LC, tr: PT.Tree, nd: PN.Node, kind: int, sel: Mapping[str, Any]) -> bool:
        sp = nd.split
        if sp is None:
            return False
        # alternatives that are not a nested refinement / coarsening of the current
        # split (a finer level of the same attribute is reached by splitting the
        # children; revision exists for non-nested alternatives such as /24 -> grp)
        hier = lc.hier
        levels = range(hier.n_levels(sp.attr) - 1)
        alts = [(sp.attr, l) for l in levels if self._revisable(lc, sp.attr, sp.level, sp.attr, l)
                and hier.card_hint(sp.attr, l) > 1]
        # a split on an identity proxy (a department's client stack, user agent,
        # TTL: P05's who_proxies) may be replaced by the who levels themselves
        if sp.attr in set(sel.get("who_proxies") or ()) and sel.get("who_mode") != "none":
            alts += [("net.src", l) for l in (3, 1, 0) if hier.card_hint("net.src", l) > 1
                     and not (l == 3 and not hier.ip2g)]
        alts = alts[:C_MAX - 1]
        if not alts:
            return False
        who_alt = any(a == "net.src" for a, _ in alts)
        targets = [t for t in self._coder_targets(lc, tr, nd, kind, sel)
                   if not SEL.same_source(sp.attr, PSEUDO_SOURCE.get(t, t))
                   and not (who_alt and t == "@who")]
        if not targets:
            return False
        coder = self._new_coder(lc, nd, targets, lc.now, tr)
        ss = PE.SplitStats(len(targets), k_b=K_B, k_v=max(K_V, len(sp.children) + 1), C=C_MAX)
        cands = [(sp.attr, sp.level)] + alts
        for i, (a, l) in enumerate(cands):
            ss.set_candidate(i, (a, l), self._card(lc, a, l),
                             self._tmask(coder, a) if i else [True] * len(targets))
        nd.meta["R"] = {"ss": ss, "coder": coder, "cands": cands, "start": lc.now}
        return True

    def _check_revision(self, lc: _LC, tr: PT.Tree, nd: PN.Node, kind: int, sel: Mapping[str, Any],
                        s: str) -> bool:
        R = nd.meta.get("R")
        if R is None or nd.split is None:
            return False
        ss: PE.SplitStats = R["ss"]
        ss.checks += 1                                      # daily checks (time-uniform delta_k)
        best, best_margin = None, -math.inf
        thr = lc.tau0 + math.log2(max(1, ss.C_ever))
        for i in range(1, ss.C):
            if ss.keys[i] is None or ss.W[i, 0] <= 0:
                continue
            margin = float(ss.D1[i, 0] - ss.D1[0, i])       # bits saved on the common events
            if margin >= lc.tau0 and ss.log2_e(i) >= thr and margin > best_margin \
                    and _stability_ok(ss, i, 0):
                best, best_margin = i, margin
        if best is None:
            if lc.now - R["start"] >= R_LEARN_S:
                nd.meta.pop("R", None)                      # restart on the next daily pass
            return False
        a, l = R["cands"][best]
        groups, oidx, gev = ss.value_groups(best)
        groups = [[v for v in g if v != PE.OTHER] for gi, g in enumerate(groups) if gi != oidx]
        groups = [g for g in groups if g]
        if not groups or sum(1 for e in gev if e >= PE.N_CHILD_MIN) < 2:
            return False
        old = {"attr": nd.split.attr, "level": nd.split.level}
        retired = tr.collapse(nd.id, lc.now, op="replace", reason={"by": (a, l), "margin": round(best_margin, 1)})
        for c in retired:
            for kset in lc.aux["learning"].values():
                kset.discard(c)
        sp = tr.split(nd.id, a, l, groups, lc.now, {"attr": a, "level": int(l), "revision_of": old,
                                                    "margin": round(best_margin, 1)})
        nd.meta.pop("R", None)
        nd.meta.pop("prune_neg", None)
        self._emit(lc, s, "pattern_replaced", tr, nd, Severity.INFO,
                   f"模式 {ctx_text(nd.ctx, lc.hier)} 的细分由 {old['attr']}@{old['level']} 改为 "
                   f"{a}@{l}（同一批事件上多节省 {best_margin:.0f} 比特）",
                   {"old": old, "new": {"attr": a, "level": int(l)}, "retired": list(retired),
                    "children": sp.all_children()})
        return True

    # ------------------------------------------------------- exceptions
    def _exceptions_daily(self, lc: _LC, tr: PT.Tree, nd: PN.Node, kind: int, t: float) -> None:
        coder: Optional[Coder] = nd.meta.get("C")
        if coder is None or nd.is_exc or nd.split is not None:
            return
        if nd.xstats is None or nd.xstats.T != len(coder.targets):
            nd.xstats = PE.ExcTracker(len(coder.targets), K_B, K_X)
        xs: PE.ExcTracker = nd.xstats
        ss0 = nd.who.levels[0]
        tot = ss0.total(t)
        if tot <= 0:
            return
        for k, c, g, ev in ss0.items(t, n=K_X):
            if isinstance(k, str) and k.startswith("shared:"):
                continue
            if g / tot >= PHI_X:
                xs.track(k)
        n_max = int(tr.budget.get("n_max", PT.TIERS["M"]))
        mask = _binding_like(xs, coder, {y for _, _, y, _, _ in lc.pairs})
        for src in list(xs.recs):
            if src in nd.exc:
                continue
            n_conf = ss0.evidence(src, t)
            r = xs.recs[src]
            if n_conf < EXC_N_CONF or not r.days2:
                continue
            if _exc_log2_e(xs, src, mask) < lc.tau0 + math.log2(max(1, r.ordinal)):
                continue
            if tr.exceptions_count() >= max(1, int(EXC_SHARE * n_max)) or len(tr.nodes) + 1 > n_max:
                break
            save = xs.per_target_saving(src)
            names = [a for a, sv, mk in zip(coder.targets, save, mask)
                     if sv >= EXC_SAVE_BITS and mk and not a.startswith("@")]
            if not names:
                continue
            xid = tr.add_exception(nd.id, src, t)
            xn = tr.nodes[xid]
            xn.meta["targets"] = names
            xn.meta["e"] = round(float(xs.log2_e(src)), 2)
            lc.aux["stats"]["exceptions"] += 1
        # removal: the IP returns to its group when its saving is < 0 on 3 checks
        for src, xid in list(nd.exc.items()):
            xn = tr.nodes.get(xid)
            if xn is None:
                continue
            sv = _pair_saving(nd, xn, t, xn.meta.get("targets") or list(xn.targets))
            if sv < 0:
                xn.meta["neg"] = xn.meta.get("neg", 0) + 1
                if xn.meta["neg"] >= PRUNE_RUNS:
                    tr.remove_exception(nd.id, src, t)
                    xs.untrack(src)
            else:
                xn.meta["neg"] = 0

    # ------------------------------------------------------------- drift
    def _structural_alarm(self, lc: _LC, tr: PT.Tree, leaf: PN.Node, ts: float) -> None:
        dr = leaf.meta.get("drift")
        if dr is None:
            le = leaf.meta.get("loss_ew")
            ref = le.read(ts) if le is not None else np.zeros(2)
            leaf.meta["drift"] = {"t0": ts, "last": ts, "kind": "structural", "dsum": {}, "dn": {},
                                  "ref": float(ref[0] / ref[1]) if ref[1] > 0 else math.nan,
                                  "prev_state": leaf.state}
            leaf.state = "evolving"
        else:
            dr["last"] = ts
        leaf.meta["last_alarm"] = ts
        lc.aux["stats"]["adwin_alarms"] += 1

    @staticmethod
    def _evolve_obs(st: Dict[str, Any], dev: float, ip: Any, day: int) -> None:
        """Record an event of an attribute under change: its signed deviation from
        the pre-change mean (in the alarm's direction) per local day; IPs whose
        events are beyond one sd in that direction show the change."""
        x = dev * st["dir"]
        st["dsum"][day] = st["dsum"].get(day, 0.0) + x
        st["dn"][day] = st["dn"].get(day, 0) + 1
        if x > st["sd"] and len(st["ips"]) < 64:
            st["ips"].add(str(ip))

    @staticmethod
    def _shifted_days(st: Mapping[str, Any]) -> List[int]:
        """Days whose mean deviation shows the change: > sd x max(0.5, 1.5/sqrt(n))."""
        out = []
        for d, sm in st["dsum"].items():
            n = st["dn"].get(d, 0)
            if n and sm / n > st["sd"] * max(0.5, 1.5 / math.sqrt(n)):
                out.append(d)
        return out

    def _alarm(self, nd: PN.Node, key: str, ts: float, r: int, mu: float, sd: float, kind: str) -> None:
        ev = nd.meta.setdefault("evolving", {})
        ev[key] = {"t0": ts, "dir": int(r), "mu": mu, "sd": sd, "ips": set(), "dsum": {}, "dn": {},
                   "kind": kind}
        if nd.state in ("confirmed", "stable"):
            nd.meta.setdefault("prev_state", nd.state)
            nd.state = "evolving"
        nd.meta["last_alarm"] = ts

    @staticmethod
    def _dph(nd: PN.Node, key: str, x: float, day: int, ts: float, sd_ev: float
             ) -> Tuple[int, int, float, int]:
        """Daily-mean Page-Hinkley (§6.9.1, per numeric target and per when).
        Events reach a node in time order, so a per-event test on a node that
        mixes sources active at different hours sees a trend every day
        (measured on pack O's OA tree: most confident nodes were `evolving`
        after two weeks without any drift). Each (node, detector, day type)
        therefore feeds ONE value per local day, the day's mean, into PH with
        lambda = 5 sd, allowance 0.5 sd, sd = max(spread of past daily means
        (H_l), per-event sd / sqrt(n of the day)), reference = the H_l mean
        of past daily means; the day is tested before it is added
        (prequential). Returns (+1 / -1 / 0, closed day, its mean, its n)
        when a new day starts, else (0, -1, nan, 0)."""
        D = nd.meta.get("dph")
        if D is None:
            D = nd.meta["dph"] = {}
        r = D.get(key)
        if r is None:
            r = D[key] = [int(day), 0.0, 0, PS.DecayedVector([PS.H_L] * 3), None]
        out: Tuple[int, int, float, int] = (0, -1, math.nan, 0)
        if int(day) != r[0]:
            if r[2] > 0:
                xm, n = r[1] / r[2], r[2]
                m = r[3].read(ts)
                res = 0
                xa = xm
                if m[2] >= PH_MIN_DAYS:
                    mu = m[0] / m[2]
                    sd = math.sqrt(max(0.0, m[1] / m[2] - mu * mu))
                    if sd_ev == sd_ev and sd_ev > 0:
                        sd = max(sd, sd_ev / math.sqrt(n))
                    if sd > 0:
                        ph = r[4]
                        if ph is None or abs(ph.lam - 5.0 * sd) > 0.5 * ph.lam:
                            ph = r[4] = PS.PageHinkley(5.0 * sd, 0.5 * sd)
                        res = ph.update(xm, mu)
                        # winsorised at mu +- 3 sd: the days of a change must not
                        # inflate the spread they are tested against
                        xa = min(max(xm, mu - 3.0 * sd), mu + 3.0 * sd)
                r[3].add(ts, [xa, xa * xa, 1.0])
                out = (res, r[0], xm, n)
            r[0], r[1], r[2] = int(day), 0.0, 0
        r[1] += float(x)
        r[2] += 1
        return out

    @staticmethod
    def _seed_evolving(st: Dict[str, Any], prev_day: int, dev_mean: float, n: int) -> None:
        """The day whose mean raised the alarm is the first observed day of the change."""
        if prev_day >= 0 and n > 0 and dev_mean == dev_mean:
            st["dsum"][prev_day] = st["dsum"].get(prev_day, 0.0) + dev_mean * st["dir"] * n
            st["dn"][prev_day] = st["dn"].get(prev_day, 0) + n

    def _ph_num(self, lc: _LC, nd: PN.Node, a: str, v: Any, ts: float, day: int, ip: Any,
                daytype: Any = None) -> None:
        s = nd.targets.get(a)
        if not isinstance(s, PN.NumSummary):
            return
        try:
            y = s.y(v)
        except (TypeError, ValueError):
            return
        if not math.isfinite(y):
            return
        ev = nd.meta.setdefault("evolving", {})
        st = ev.get(a)
        if st is not None:
            self._evolve_obs(st, y - st["mu"], ip, day)
            return
        w, mu, var = s.moments(ts, PS.CH_L)
        if not (w >= 5.0 and var == var and var > 0):
            return
        sd = math.sqrt(var)
        dt_key = "nwd" if daytype not in (None, EV.ABSENT, "workday", "wd", 0) else "wd"
        r, pday, xm, n = self._dph(nd, f"{a}|{dt_key}", y, day, ts, sd)
        if r != 0:
            self._alarm(nd, a, ts, r, mu, sd, "num")
            self._seed_evolving(ev[a], pday, xm - mu, n)
            self._evolve_obs(ev[a], y - mu, ip, day)

    def _ph_when(self, lc: _LC, nd: PN.Node, minute: float, ts: float, day: int, ip: str,
                 daytype: int = 0) -> None:
        """Page-Hinkley on the signed circular difference of the arrival minute
        from the H_l circular mean (sigma from the H_l second moment of those
        differences, accumulated only once the mean is defined), tested on
        daily means per day type (see _dph)."""
        wc = nd.meta.get("wc")
        if wc is None:
            wc = nd.meta["wc"] = PS.DecayedVector([PS.H_L] * 5)
        ang = 2.0 * math.pi * float(minute) / 1440.0
        x = wc.read(ts)
        n = x[2]
        add = [math.cos(ang), math.sin(ang), 1.0, 0.0, 0.0]
        if n >= 2:
            mu = math.atan2(x[1] / n, x[0] / n) % (2 * math.pi) * 1440.0 / (2 * math.pi)
            dmin = (float(minute) - mu + 720.0) % 1440.0 - 720.0
            add[3], add[4] = dmin * dmin, 1.0
            ev = nd.meta.setdefault("evolving", {})
            st = ev.get("@when")
            if st is not None:
                self._evolve_obs(st, dmin, ip, day)
            elif x[4] >= 2.0:
                sd = math.sqrt(max(1.0, x[3] / x[4]))
                r, pday, xm, nd_n = self._dph(nd, f"@when|{int(daytype or 0)}", dmin, day, ts, sd)
                if r != 0:
                    self._alarm(nd, "@when", ts, r, mu, sd, "when")
                    self._seed_evolving(ev["@when"], pday, xm, nd_n)
                    self._evolve_obs(ev["@when"], dmin, ip, day)
        wc.add(ts, add)

    def _drift_daily(self, lc: _LC, tr: PT.Tree, nd: PN.Node, t: float, day: int, s: str,
                     normal_days: Callable[[int, int], int]) -> None:
        who_top = len(nd.who.heavy_set(0, t)[0]) or 1
        need_ips = max(2, math.ceil(0.5 * who_top))
        ev: Dict[str, Any] = nd.meta.get("evolving") or {}
        for a, st in list(ev.items()):
            persist = T_PERSIST_DAYS.get(st.get("kind", "num"), 1)
            shifted = self._shifted_days(st)
            nd_days = normal_days(min(shifted), day + 1) if shifted else 0
            coordinated = len(st["ips"]) >= need_ips
            single = who_top <= 1 and nd_days >= SINGLE_IP_DAYS
            quarantined = any(MG.is_quarantined(lc.store, s, ip, t) for ip in list(st["ips"])[:8])
            if len(shifted) >= persist + 1 and (coordinated or single) and not quarantined:
                if a == "@when":
                    nd.when.reset_confidence(t)
                else:
                    nd.reset_confidence(t, [a], day)
                nd.cver += 1
                del ev[a]
                # the daily-mean detector restarts from the new regime
                for k in [k for k in (nd.meta.get("dph") or {}) if k.split("|", 1)[0] == a]:
                    del nd.meta["dph"][k]
                self._emit(lc, s, "pattern_drift", tr, nd, Severity.INFO,
                           f"模式 {ctx_text(nd.ctx, lc.hier)} 的 {a} 发生了已确认的合法变化"
                           f"（{len(st['ips'])} 个 IP、{len(shifted)} 天），置信度从新状态重新累积",
                           {"attr": a, "dir": st["dir"], "ips": sorted(st["ips"])[:16],
                            "days": len(shifted), "kind": st.get("kind")})
            elif t - st["t0"] >= DRIFT_EXPIRE_S:
                del ev[a]
            else:
                # while evolving the confidence channel of the attribute is capped at its H_m state
                sm = nd.targets.get(a)
                inner = getattr(sm, "ss", None) or getattr(sm, "values", None)
                if inner is not None and hasattr(inner, "cap_confidence"):
                    inner.cap_confidence(t)
        dr = nd.meta.get("drift")
        if dr is not None:
            # accepted when the loss stayed above its pre-alarm level by >= 0.5 bit
            # on T_persist normal days (a real change of the node's distribution);
            # an alarm without such persistence expires without touching confidence
            ref = dr.get("ref", math.nan)
            higher = [d for d, sm in dr["dsum"].items()
                      if dr["dn"].get(d) and ref == ref and sm / dr["dn"][d] >= ref + 0.5
                      and normal_days(d, d + 1)]
            if len(higher) >= T_PERSIST_DAYS["structural"]:
                nd.reset_confidence(t)
                nd.cver += 1
                nd.meta.pop("drift", None)
                if nd.split_stats is not None:             # do not mix regimes (§6.5.2)
                    nd.split_stats.restart()
                    nd.meta["lstart"] = t
                self._emit(lc, s, "pattern_drift", tr, nd, Severity.INFO,
                           f"模式 {ctx_text(nd.ctx, lc.hier)} 的整体分布发生了已确认的变化，置信度从新状态重新累积",
                           {"kind": "structural", "since": dr["t0"]})
            elif t - dr["t0"] >= DRIFT_EXPIRE_S:
                nd.meta.pop("drift", None)
            else:
                ne = nd.n_eff
                ne.set_entry(PS.CH_L, min(ne.get(PS.CH_L, t), ne.get(PS.CH_M, t)), t)
        if nd.state == "evolving" and not nd.meta.get("drift") and not ev:
            nd.state = nd.meta.pop("prev_state", None) or (dr or {}).get("prev_state") or "confirmed"
            if nd.state not in ("confirmed", "stable"):
                nd.state = "confirmed"

    # -------------------------------------------------------------- pairs
    def _pairs(self, lc: _LC, tr: PT.Tree, path: Sequence[int], kind: int, get: Callable[[str], Any],
               ts: float, mass: float, omega: float) -> None:
        for x, lv, y, nodes, pk in lc.pairs:
            if pk is not None and pk != kind:
                continue
            xv = get(x)
            yv = get(y)
            if xv is EV.ABSENT or yv is EV.ABSENT:
                continue
            xg = _h(lc.hier.gen(x, lv, xv))
            yv = _h(yv)
            name = x if lv == 0 else f"{x}@{lv}"
            targets = [n for n in path if n in nodes] if nodes else [path[0], path[-1]]
            for nid in dict.fromkeys(targets):
                tr.nodes[nid].pair(name, y).update(xg, yv, ts, mass, omega)

    # --------------------------------------------------------------- rates
    def _rates(self, lc: _LC, m: PT.PTreeModel, s: str, aux: Dict[str, Any], D: float) -> None:
        """pat.rate (P03): finished per-(ip, node) hourly counts -> node rate.ip_h."""
        for ts_b, obj in MP.learnable_batches(lc.store, s, EV.PAT_RATE, aux["last"].get((s, "rate")), lc.now, D):
            aux["last"][(s, "rate")] = ts_b
            for kind, nid, cnt in _rate_rows(obj):
                tr = m.kinds.get(kind)
                nd = tr.nodes.get(nid) if tr is not None else None
                if nd is None:
                    continue
                if nd.rate_iph is None:
                    nd.rate_iph = PS.TDigest(50.0, PS.H_M)
                nd.rate_iph.add(float(cnt), ts_b, 1.0)

    # ---------------------------------------------------------------- held
    def _hold(self, m: PT.PTreeModel, s: str, ip: str, ts: float, kind: int, b: EV.EventBatch,
              cb: Optional[EV.EventBatch], i: int, mass: float) -> None:
        aux = self.aux(m)
        row: Dict[str, Any] = {}
        for nm in b.names():
            v = b.get(nm, i)
            if v is not EV.ABSENT:
                row[nm] = v
            if len(row) >= HELD_COLS:
                break
        if cb is not None:
            for nm in ("ctx.tod_min", "ctx.daytype", "ctx.when"):
                v = cb.get(nm, i)
                if v is not EV.ABSENT:
                    row[nm] = v
        row["net.src"] = ip
        q = aux["held"].get((s, ip))
        if q is None:
            q = aux["held"][(s, ip)] = deque(maxlen=HELD_PER_IP)
        if len(q) == q.maxlen:
            aux["held_n"] -= 1
        q.append((ts, kind, row, mass))
        aux["held_n"] += 1
        while aux["held_n"] > HELD_MAX:                    # FIFO over the tree
            oldest = min(aux["held"], key=lambda k: aux["held"][k][0][0] if aux["held"][k] else math.inf)
            aux["held"][oldest].popleft()
            aux["held_n"] -= 1
            if not aux["held"][oldest]:
                del aux["held"][oldest]

    def _held(self, lc: _LC, m: PT.PTreeModel, s: str) -> int:
        aux = self.aux(m)
        n = 0
        for (hs, ip), q in list(aux["held"].items()):
            if hs != s:
                continue
            ctl = MG.control(lc.store, s, ip)
            if ctl.get("frozen"):
                aux["held_n"] -= len(q)
                del aux["held"][(hs, ip)]
                continue
            rel = ctl.get("release")
            keep: Deque = deque(maxlen=HELD_PER_IP)
            for ts, kind, row, mass in q:
                if lc.now - ts > HELD_EXPIRE_S:
                    aux["held_n"] -= 1
                    continue
                if rel is not None and rel[0] <= ts <= rel[1]:
                    tp = MG.trust_prov(lc.store, s, ip, ts)
                    f = 1.0 if not tp == tp else min(1.0, max(0.0, tp))
                    aux["held_n"] -= 1
                    if f > 0:
                        tr = self._tree(lc, m, kind)
                        get = (lambda nm, row=row: row.get(nm, EV.ABSENT))
                        self._learn_one(lc, tr, s, kind, get, ip, ts, mass * f, f,
                                        row.get("sess.key", "∅"))
                        n += 1
                    continue
                keep.append((ts, kind, row, mass))
            if keep:
                aux["held"][(hs, ip)] = keep
            else:
                del aux["held"][(hs, ip)]
        return n

    # ------------------------------------------------------- maintenance
    def _maintain(self, lc: _LC, m: PT.PTreeModel, s: str) -> None:
        aux = self.aux(m)
        now = lc.now
        day = _local_day(now, lc.off)
        # minute reservoirs requested by P09 (cheap, every tick)
        for kind, nids in lc.minutes.items():
            tr = m.kinds.get(kind)
            if tr is None:
                continue
            for nid in nids:
                nd = tr.nodes.get(nid)
                if nd is not None and nd.when.res is None:
                    nd.when.want_minutes(True, seed=nid)
        if aux["day"] is None:
            aux["day"] = day
            return
        if day != aux["day"]:
            aux["day"] = day
            for kind, tr in list(m.kinds.items()):
                self._daily(lc, m, tr, kind, s, day)
        hour = ((now + lc.off) % DAY) / 3600.0
        if aux["snap_day"] != day and hour >= SNAPSHOT_HOUR:
            aux["snap_day"] = day
            for kind, tr in m.kinds.items():
                self._snapshots(lc, tr, kind, s)

    def _normal_days_fn(self, lc: _LC, systems: Iterable[str]) -> Callable[[int, int], int]:
        cal: Dict[int, Any] = {}
        for s in systems:
            pc = lc.store.get_model(s, SYSTEM_ENTITY, "model.pcal")
            if isinstance(pc, Mapping):
                for d, ok in (pc.get("normal") or {}).items():
                    cal[int(d)] = bool(ok) or cal.get(int(d), False)

        def count(d0: int, d1: int) -> int:
            """Normal local days in [d0, d1) (unknown days count as normal)."""
            return sum(1 for d in range(int(d0), int(d1)) if cal.get(d, True))
        return count

    def _daily(self, lc: _LC, m: PT.PTreeModel, tr: PT.Tree, kind: int, s: str, day: int) -> None:
        t = lc.now
        sel = lc.selection(kind)
        aux = self.aux(m)
        normal_days = self._normal_days_fn(lc, aux["systems"] or [s])
        learning = aux["learning"].setdefault(kind, set())
        learning.intersection_update(tr.nodes.keys())
        # 1. schema change: splits on gone attributes collapse at once
        for nid in [n for n, nd in tr.nodes.items() if nd.split is not None and nd.split.attr in lc.gone]:
            if nid in tr.nodes:
                self._collapse(lc, tr, nid, "prune", {"gone": tr.nodes[nid].split.attr})
        # 2. prune (split saving < 0 on 3 daily checks) and sibling merge
        for nid in sorted([n for n, nd in tr.nodes.items() if nd.split is not None],
                          key=lambda n: -tr.nodes[n].depth):
            nd = tr.nodes.get(nid)
            if nd is None or nd.split is None:
                continue
            kids = [tr.nodes[c] for c in nd.split.all_children() if c in tr.nodes]
            if any(k.split is not None for k in kids):
                continue                                   # prune bottom-up only
            sv = _split_saving(nd, kids, t, lc.who_level)
            nd.meta["saving"] = round(sv, 2)
            if sv < 0 and nd.n_m(t) >= N_CONF:
                nd.meta["prune_neg"] = nd.meta.get("prune_neg", 0) + 1
                if nd.meta["prune_neg"] >= PRUNE_RUNS:
                    self._collapse(lc, tr, nid, "prune", {"saving": round(sv, 2)})
                    continue
            else:
                nd.meta["prune_neg"] = 0
            self._merge_siblings(lc, tr, nd, t)
        # 3. lifecycle, invariants, drift, exceptions
        conf_dates = CONF_DATES.get(kind, 3)
        for nid in list(tr.nodes.keys()):
            nd = tr.nodes.get(nid)
            if nd is None:
                continue
            self._invariants(nd, t)
            cd = nd.meta.get("C")
            if cd is not None and nd.split_stats is None and cd.hver != self._hver(lc, cd.targets):
                nd.meta["C"] = self._new_coder(lc, nd, self._coder_targets(lc, tr, nd, kind, sel), t, tr)
                nd.xstats = None
            if len(nd.targets) > M_T + 4:              # bounded: drop summaries no longer targeted
                keep = set(self._targets_of(lc, tr, nd, kind, sel)) | set(lc.extra_targets.get((kind, nid), ()))
                for a in [a for a in nd.targets if a not in keep][:len(nd.targets) - M_T]:
                    del nd.targets[a]
            if nd.state == "candidate":
                if nd.n_c(t) >= N_CONF and nd.n_days() >= conf_dates and not self._fitter_pending(lc, kind, nid):
                    nd.state = "confirmed"
                    nd.meta["confirmed_at"] = t
                    self._emit(lc, s, "pattern_confirmed", tr, nd, Severity.INFO,
                               f"模式已确认：{ctx_text(nd.ctx, lc.hier)}（证据 {nd.n_c(t):.0f}，{nd.n_days()} 天）",
                               {"n_c": round(nd.n_c(t), 1), "days": nd.n_days()})
            elif nd.state == "confirmed":
                if (t - float(nd.meta.get("confirmed_at", t)) >= STABLE_S and nd.n_days() >= STABLE_DATES
                        and t - float(nd.meta.get("last_alarm", -math.inf)) >= STABLE_S):
                    nd.state = "stable"
            if nd.state in ("confirmed", "stable") and nd.last_seen is not None and nd.parent is not None:
                self._stale_check(lc, tr, nd, t, day, s, normal_days)
            elif nd.state == "stale":
                if t - float(nd.meta.get("stale_at", t)) >= STALE_RETIRE_S:
                    self._retire(lc, tr, nd, t, s)
                    continue
            self._drift_daily(lc, tr, nd, t, day, s, normal_days)
            if nd.state in ("confirmed", "stable", "evolving") and nd.split is None and not nd.is_exc:
                self._exceptions_daily(lc, tr, nd, kind, t)
        # 4. EFDT revision on the R_max busiest internal nodes
        # only nodes whose children are all leaves: the revision statistics compare
        # the current ONE-LEVEL split with an alternative one-level split on the
        # same events, which says nothing about what the children's own subtrees
        # learned since (measured on pack O: a root revision replaced a split whose
        # subtree held 20 nodes, confirmed ones included, and discarded it)
        internal = sorted([nd for nd in tr.nodes.values() if nd.split is not None and nd.n_m(t) >= REV_MIN_EV
                           and all(tr.nodes[c].split is None for c in nd.split.all_children() if c in tr.nodes)],
                          key=lambda x: -x.mass_at(t))[:R_MAX]
        keep = {nd.id for nd in internal}
        for nd in tr.nodes.values():
            if nd.meta.get("R") is not None and nd.id not in keep:
                nd.meta.pop("R", None)
        for nd in internal:
            if nd.id not in tr.nodes:
                continue
            if nd.meta.get("R") is None:
                self._start_revision(lc, tr, nd, kind, sel)
            else:
                self._check_revision(lc, tr, nd, kind, sel, s)
        # 5. budget: learning leaves (largest mass x (1 - purity)) and node count
        self._budget(lc, tr, kind)
        # the sources whose daily activity pattern_absent needs (heavy sets of
        # confident nodes: <= 8 per node, never the population)
        watch = aux.setdefault("watch", set())
        if kind == min(m.kinds):
            watch.clear()
        for nd in tr.nodes.values():
            if nd.state in PN.CONFIDENT_STATES:
                watch.update(str(x) for x in nd.who.heavy_set(0, t)[0][:PN.HEAVY_MAX])
        # 6. dormant memory expiry
        for nid in [n for n, r in tr.dormant.items() if t - r.get("since", t) >= DORMANT_KEEP_S]:
            tr.dormant.pop(nid, None)

    def _collapse(self, lc: _LC, tr: PT.Tree, nid: int, op: str, reason: Any) -> None:
        gone = tr.collapse(nid, lc.now, op=op, reason=reason)
        for kset in lc.aux["learning"].values():
            for c in gone:
                kset.discard(c)
        nd = tr.nodes[nid]
        nd.meta.pop("prune_neg", None)
        nd.meta.pop("R", None)
        lc.aux["stats"]["collapses"] += 1

    def _merge_siblings(self, lc: _LC, tr: PT.Tree, parent: PN.Node, t: float) -> None:
        sp = parent.split
        kids = [c for c in sp.all_children() if c in tr.nodes and tr.nodes[c].split is None
                and tr.nodes[c].n_m(t) >= PE.N_CHILD_MIN]
        runs = parent.meta.setdefault("merge_runs", {})
        best = None
        for i in range(len(kids)):
            for j in range(i + 1, len(kids)):
                d = _node_jsd(tr.nodes[kids[i]], tr.nodes[kids[j]], t, lc.who_level)
                key = f"{min(kids[i], kids[j])}-{max(kids[i], kids[j])}"
                if d < TAU_MERGE:
                    runs[key] = runs.get(key, 0) + 1
                    if runs[key] >= PRUNE_RUNS and (best is None or d < best[0]):
                        best = (d, kids[i], kids[j], key)
                else:
                    runs.pop(key, None)
        if best is not None:
            _, a, b, key = best
            if b == sp.other:
                a, b = b, a
            try:
                tr.merge_siblings(parent.id, a, b, t)
            except ValueError:
                return
            runs.pop(key, None)
            for kset in lc.aux["learning"].values():
                kset.discard(b)
            if len(sp.children) == 0:
                self._collapse(lc, tr, parent.id, "merge", {"all_merged": True})
            lc.aux["stats"]["merges"] += 1

    @staticmethod
    def _invariants(nd: PN.Node, t: float) -> None:
        if nd.n_c(t) < 30:
            return
        for a, s in nd.targets.items():
            v = None
            if isinstance(s, PN.CatSummary):
                v = s.invariant(t)
            elif isinstance(s, PN.TextSummary) and s.values is not None:
                items = s.values.items(t, PS.CH_M, 1)
                tot = s.values.total(t)
                if items and tot > 0 and items[0][2] / tot >= 0.995 and s.values.total_evidence(t) >= 30:
                    v = items[0][0]
            if v is not None:
                nd.inv[a] = (0, v)
            else:
                nd.inv.pop(a, None)

    def _fitter_pending(self, lc: _LC, kind: int, nid: int) -> bool:
        """A fitter (P06-P09) blocks confirmation only when it explicitly lists
        the node as pending (model.<fit>['pending'] = [nid...] or {kind: [nid...]});
        fitters that do not run, or that declared the node done / not
        applicable, do not block."""
        for name in FITTERS:
            mdl = MP.get_model(lc.store, lc.key, name)
            if not isinstance(mdl, Mapping):
                continue
            pend = mdl.get("pending")
            if isinstance(pend, Mapping):
                pend = pend.get(kind) or pend.get(str(kind))
            if pend and int(nid) in {int(x) for x in pend}:
                return True
        return False

    def _stale_check(self, lc: _LC, tr: PT.Tree, nd: PN.Node, t: float, day: int, s: str,
                     normal_days: Callable[[int, int], int]) -> None:
        last_day = _local_day(nd.last_seen, lc.off)
        if last_day >= day - 1:
            return
        wd_mass = nd.when.mass(0, t)
        nwd_mass = nd.when.mass(1, t)
        tot = wd_mass + nwd_mass
        if tot <= 0:
            return
        active_types = {d for d, mm in ((0, wd_mass), (1, nwd_mass)) if mm / tot >= 0.1}
        pcal: Dict[int, Any] = {}
        for sys_ in lc.aux["systems"] or [s]:
            pc = lc.store.get_model(sys_, SYSTEM_ENTITY, "model.pcal")
            if isinstance(pc, Mapping):
                pcal.update(pc.get("days") or {})
        n_active = 0
        for d in range(last_day + 1, day):
            rec = pcal.get(d) or pcal.get(str(d))
            cls = rec.get("class") if isinstance(rec, Mapping) else None
            if cls == "holiday" or normal_days(d, d + 1) == 0:
                continue
            if cls is None:
                cls = "workday" if _dt.date.fromordinal(d).weekday() < 5 else "weekend"
            dtp = 0 if cls in ("workday", "makeup") else 1
            if dtp in active_types:
                n_active += 1
        n_dates = max(1, nd.n_days(43))
        rate = nd.n_c(t) / n_dates
        E = rate * n_active
        if E >= STALE_E and pmdl.poisson_zero_p(E) < 0.05:
            stable = nd.state == "stable"
            nd.state = "stale"
            nd.meta["stale_at"] = t
            sev = Severity.INFO
            if stable and nd.n_days(7) >= 5:
                seen = lc.aux["seen"]
                heavy, _ = nd.who.heavy_set(0, t)
                if heavy and all((seen.peek(ip) or 0.0) >= t - DAY for ip in heavy):
                    sev = Severity.LOW
            self._emit(lc, s, "pattern_absent", tr, nd, sev,
                       f"预期出现的模式 {ctx_text(nd.ctx, lc.hier)} 未出现（期望 {E:.1f} 次，实际 0）",
                       {"expected": round(E, 2), "active_days": n_active})

    def _retire(self, lc: _LC, tr: PT.Tree, nd: PN.Node, t: float, s: str) -> None:
        par = tr.nodes.get(nd.parent) if nd.parent is not None else None
        if nd.split is not None or par is None:
            return
        if nd.is_exc:
            tr.retire_leaf(nd.id, t, "stale")
            return
        sp = par.split
        if sp is None or nd.id == sp.other:
            return
        grp = sp.groups[sp.children.index(nd.id)]
        rec_ctx = nd.ctx
        # dormant when its dates recur regularly (monthly, quarterly)
        dates = [i for i in range(64) if (nd.days_bits >> i) & 1]
        regular = False
        if len(dates) >= 3:
            gaps = np.diff(sorted(dates))
            regular = bool(gaps.mean() > 0 and gaps.std() / gaps.mean() <= DORMANT_CV and gaps.mean() >= 5)
        ref = nd.ref
        tr.retire_leaf(nd.id, t, "stale")
        self._emit(lc, s, "pattern_retired", tr, nd, Severity.INFO,
                   f"模式 {ctx_text(rec_ctx, lc.hier)} 已连续 30 天未出现，退役", {"regular": regular})
        if regular:
            tr.make_dormant(nd.id, t)
            rec = tr.dormant.get(nd.id)
            if rec is not None:
                rec.update({"parent": par.id, "attr": sp.attr, "level": sp.level, "group": grp,
                            "state": "confirmed", "ref": ref})

    def _maybe_revive(self, lc: _LC, tr: PT.Tree, leaf: PN.Node, get: Callable[[str], Any], ts: float,
                      s: str) -> bool:
        par = tr.nodes.get(leaf.parent)
        if par is None or par.split is None or par.split.other != leaf.id:
            return False
        for nid, rec in list(tr.dormant.items()):
            if rec.get("parent") != par.id or rec.get("attr") != par.split.attr \
                    or rec.get("level") != par.split.level:
                continue
            g = _h(lc.hier.gen(par.split.attr, par.split.level, get(par.split.attr)))
            if g not in rec.get("group", ()):
                continue
            n_max = int(tr.budget.get("n_max", PT.TIERS["M"]))
            if len(tr.nodes) + 1 > n_max:
                return False
            new = _add_group_child(tr, par, rec["group"], ts)
            nd = tr.nodes[new]
            nd.state = "confirmed"
            nd.meta["confirmed_at"] = ts
            nd.meta["revived_from"] = nid
            nd.ref = rec.get("ref")
            nd.days_bits = rec.get("days_bits", 0)
            tr.dormant.pop(nid, None)
            tr._log(ts, "revive", new, (nid,), (new,), None)
            self._emit(lc, s, "pattern_revived", tr, nd, Severity.INFO,
                       f"休眠模式重新出现：{ctx_text(nd.ctx, lc.hier)}（周期性模式，不按新行为处理）",
                       {"dormant": nid})
            return True
        return False

    def _budget(self, lc: _LC, tr: PT.Tree, kind: int) -> None:
        t = lc.now
        learning = self.aux_learning(lc, kind)
        l_max = self._l_max(lc, tr)
        if len(learning) > l_max:
            def prio(nid: int) -> float:
                nd = tr.nodes[nid]
                pur = []
                for s_ in nd.targets.values():
                    inner = getattr(s_, "ss", None) or getattr(s_, "values", None)
                    if inner is not None and hasattr(inner, "items"):
                        it = inner.items(t, None, 1)
                        tot = inner.total(t)
                        if it and tot > 0:
                            pur.append(it[0][2] / tot)
                purity = float(np.mean(pur)) if pur else 0.0
                return nd.mass_at(t) * (1.0 - purity)
            ranked = sorted(learning, key=prio, reverse=True)
            for nid in ranked[l_max:]:
                nd = tr.nodes[nid]
                nd.split_stats = None
                nd.meta["no_learn"] = True
                learning.discard(nid)
            for nid in ranked[:l_max]:
                tr.nodes[nid].meta.pop("no_learn", None)
        else:
            for nd in tr.nodes.values():
                nd.meta.pop("no_learn", None)
        over = tr.over_budget()
        if over > 0:
            self._make_room(lc, tr, over)

    def _make_room(self, lc: _LC, tr: PT.Tree, need: int, protect: Set[int] = frozenset()) -> None:
        """Prune the lowest-utility leaves (stale first; utility = H_m evidence per
        day since creation x 2 when confirmed, a proxy of the leaf's share of its
        parent's saving) until `need` node slots are free; a parent whose named
        children are all gone collapses."""
        t = lc.now
        n_max = int(tr.budget.get("n_max", PT.TIERS["M"]))
        target = max(0, len(tr.nodes) + need - n_max)
        if target <= 0:
            return
        freed = 0
        cands = []
        for nid, nd in tr.nodes.items():
            if nd.split is not None or nid in protect or nd.parent is None:
                continue
            par = tr.nodes.get(nd.parent)
            if par is None:
                continue
            if par.split is not None and nid == par.split.other:
                continue
            age = max(1.0, (t - nd.created) / DAY)
            u = nd.n_m(t) / age * (2.0 if nd.state in PN.CONFIDENT_STATES else 1.0)
            cands.append((0 if nd.state == "stale" else 1, u, nid))
        cands.sort()
        for _, _, nid in cands:
            if freed >= target:
                break
            nd = tr.nodes.get(nid)
            if nd is None or nd.split is not None:
                continue
            par = tr.nodes.get(nd.parent)
            try:
                tr.retire_leaf(nid, t, "budget")
            except ValueError:
                continue
            freed += 1
            for kset in lc.aux["learning"].values():
                kset.discard(nid)
            if par is not None and par.split is not None and not par.split.children:
                self._collapse(lc, tr, par.id, "prune", {"budget": True})
                freed += 1
        lc.aux["stats"]["budget_pruned"] += freed

    def _snapshots(self, lc: _LC, tr: PT.Tree, kind: int, s: str) -> None:
        """Daily reference snapshot (§6.8.3) of confident nodes without a drift
        alarm or held events in the previous day."""
        t = lc.now
        fitted = {name: MP.get_model(lc.store, lc.key, name) for name in FITTERS}
        aux = lc.aux
        held_ips = {ip for (_, ip) in aux["held"].keys()}
        for nd in tr.nodes.values():
            if nd.state not in ("confirmed", "stable"):
                continue
            if t - float(nd.meta.get("last_alarm", -math.inf)) < DAY:
                continue
            heavy, _ = nd.who.heavy_set(0, t)
            if held_ips and any(ip in held_ips for ip in heavy):
                continue
            nd.ref = {"t": t, "version": nd.version, "cver": nd.cver,
                      "targets": {a: _compact(sm, t) for a, sm in nd.targets.items()},
                      "who": [str(x) for x in heavy[:16]],
                      "when": {"wd": nd.when.density(0).astype(np.float32),
                               "nwd": nd.when.density(1).astype(np.float32)},
                      "fitted": {name: _fitted_entry(mdl, kind, nd.id) for name, mdl in fitted.items()
                                 if _fitted_entry(mdl, kind, nd.id) is not None}}

    # -------------------------------------------------------------- events
    def _emit(self, lc: _LC, s: str, kind: str, tr: PT.Tree, nd: PN.Node, sev: Severity, desc: str,
              extra: Dict[str, Any]) -> None:
        pid = PN.pattern_id(lc.key, tr.kind, nd.id, nd.version, nd.cver)
        lc.store.add_event(BehaviorEvent(
            system=s, entity=SYSTEM_ENTITY, ts=lc.now, kind=kind, score=0.0, severity=sev,
            description=desc,
            extra=dict(extra, pattern_id=pid, node=nd.id, tree_key=lc.key, event_kind=tr.kind,
                       context=ctx_text(nd.ctx, lc.hier), state=nd.state),
            dedupe_key=f"{kind}|{lc.key}|{tr.kind}|{nd.id}|{_local_day(lc.now, lc.off)}"))
        lc.aux["stats"][kind] += 1


# ================================================================ helpers
def _add_group_child(tr: PT.Tree, par: PN.Node, group: Iterable[Any], t: float) -> int:
    """Re-add a named child for `group` under par's split (dormant revival)."""
    sp = par.split
    g = frozenset(group)
    cid = tr._new(par.id, par.depth + 1, par.ctx + ((sp.attr, sp.level, g, False),), t)
    sp.groups.append(g)
    sp.children.append(cid)
    sp.reindex()
    oth = tr.nodes.get(sp.other)
    if oth is not None:
        union = frozenset().union(*sp.groups)
        oth.ctx = oth.ctx[:-1] + ((sp.attr, sp.level, union, True),)
    par.version += 1
    tr.version += 1
    return cid


def _rate_rows(obj: Any) -> List[Tuple[int, int, float]]:
    """pat.rate rows as (kind, nid, count). Accepted shapes: {'rows': [(nid, ip,
    count) | (kind, nid, ip, count) | {'kind', 'nid', 'count'}]}, a list of the
    same, or an EventBatch with columns nid / count (/ kind)."""
    out: List[Tuple[int, int, float]] = []
    rows = obj.get("rows") if isinstance(obj, Mapping) else obj
    if isinstance(obj, EV.EventBatch):
        nid = obj.dense("nid", -1)
        cnt = obj.dense("count", 0.0)
        knd = obj.dense("kind", 0) if obj.has("kind") else np.zeros(obj.n)
        for i in range(obj.n):
            if nid[i] is not EV.ABSENT and float(nid[i]) >= 0:
                out.append((int(knd[i]), int(nid[i]), float(cnt[i])))
        return out
    for r in rows or ():
        try:
            if isinstance(r, Mapping):
                out.append((int(r.get("kind", 0)), int(r["nid"]), float(r["count"])))
            elif len(r) == 3:
                out.append((0, int(r[0]), float(r[2])))
            elif len(r) >= 4:
                out.append((int(r[0]), int(r[1]), float(r[3])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _check_valid_first(ss: PE.SplitStats, tau0: float, s_mode: str) -> PE.SplitDecision:
    """Rules (V), (G), (S), (D), (M) of §6.5.5, with two deviations measured on
    pack O's OA tree (both keep the false-split bound: every candidate is
    tested at tau0 + log2 C_ever and its ordinal is charged in C_ever):

    * c1 is chosen AMONG THE CANDIDATES THAT PASS (V). The text takes the best
      by G overall; a candidate whose significance test has not passed then
      blocks every valid one (a login node whose route split had log2 e = 126
      against a threshold of 13.8 waited on a /24 candidate with a larger,
      target-duplicated G but log2 e = 8.4).
    * in 'margin' mode, (S) between valid candidates is the tau0-bit MDL margin
      on their COMMON events (§6.6): c1 = the first valid candidate by G that
      no other valid candidate beats by >= tau0 bits on the events where both
      were tracked; (S) holds when it is unbeaten. Two candidates within tau0
      bits of each other are equivalent at the MDL scale (a route and a
      response-size bin that separate the same events), where the VFDT tie
      eps <= tau_tie needs ~1e4 units at a range of ~15 bits; and a candidate
      tracked since earlier can lead on G while losing on the common events
      (measured: the size bin led by 5 bits on G, the route won by 34 bits
      on the common events)."""
    act = ss.active()
    thr = float(tau0) + math.log2(max(1, ss.C_ever))
    valid = [i for i in act if ss.log2_e(i) >= thr]
    if not valid:
        return ss.check(tau0=tau0, s_mode=s_mode)
    keep = set(valid)
    chosen = None
    unbeaten = True
    if s_mode == "margin":
        valid.sort(key=lambda i: (-float(ss.G[i]), int(ss.ordinal[i])))

        def beaten(c: int) -> bool:
            return any(ss.W[o, c] > 0 and float(ss.D1[o, c] - ss.D1[c, o]) >= float(tau0)
                       for o in valid if o != c)
        chosen = next((c for c in valid if not beaten(c)), None)
        if chosen is None:
            chosen, unbeaten = valid[0], False
        keep = {chosen}
    masked = [i for i in act if i not in keep]
    saved = ss.G[masked].copy()
    ss.G[masked] = -1e300
    try:
        dec = ss.check(tau0=tau0, s_mode=s_mode)
    finally:
        ss.G[masked] = saved
    if chosen is not None:
        dec.pass_s = unbeaten
    return dec


def _stability_ok(ss: PE.SplitStats, c1: int, c2: int) -> bool:
    """Rule (S) of §6.5.5 between two tracked candidates (time-uniform empirical
    Bernstein on the per-unit difference of their savings, or a tie)."""
    W = ss.W[c1, c2]
    if W <= 0:
        return False
    mu = (ss.D1[c1, c2] - ss.D1[c2, c1]) / W
    ex2 = (ss.Qaa[c1, c2] - 2.0 * ss.Qab[c1, c2] + ss.Qaa[c2, c1]) / W
    var = max(0.0, ex2 - mu * mu)
    R = ss.Rmax[c1, c2] - ss.Rmin[c1, c2]
    R = R if math.isfinite(R) else 0.0
    eps = pmdl.empirical_bernstein_bound(var, R, W, pmdl.time_uniform_delta(PE.DELTA, max(1, ss.checks)))
    return mu >= eps


def _note_value_days(leaf: PN.Node, ss: PE.SplitStats, cvals: Sequence[Any], day: int) -> None:
    """Up to two local dates per (candidate ordinal, value) of a learning leaf,
    for the recurrence condition on named children (bounded: stale ordinals and
    values that lost their slot are dropped when the map grows)."""
    sd = leaf.meta.get("sdays")
    if sd is None:
        sd = leaf.meta["sdays"] = {}
    for i, v in enumerate(cvals):
        if v is None or ss.keys[i] is None or v not in ss.slot_of[i]:
            continue
        k = (int(ss.ordinal[i]), v)
        cur = sd.get(k)
        if cur is None:
            sd[k] = (day,)
        elif len(cur) < 2 and day not in cur:
            sd[k] = cur + (day,)
    if len(sd) > 4 * ss.C * (ss.kv + 1):
        live = {(int(ss.ordinal[i]), v) for i in range(ss.C) if ss.keys[i] is not None
                for v in ss.slot_of[i]}
        for k in [k for k in sd if k not in live]:
            del sd[k]


def _expire_runs(burst: PS.BurstEvidence, before: float) -> None:
    """Drop burst-run state older than tau_burst: a run whose last row is more
    than tau_burst old restarts anyway, so the state kept is proportional to the
    sources active in the last few minutes, not to the population."""
    d = getattr(getattr(burst, "_lru", None), "_d", None)
    if d is None:
        return
    while d:
        k = next(iter(d))
        v = d[k]
        if isinstance(v, tuple) and v and float(v[0]) < before:
            del d[k]
        else:
            break


def _binding_like(xs: PE.ExcTracker, coder: Coder, pair_ys: Set[str]) -> np.ndarray:
    """Targets excluded from the exception test (§6.7: bindings are not
    exceptions): the who target, targets that are the Y of a binding pair
    requested by P08, and targets that behave as a functional dependency of the
    source at this node (every tracked source with >= 3 units is >= 80 % pure
    on its own modal value and the modal values differ): 'each of the three IPs
    submits its own username' is one FD at the group node, not three nodes."""
    T = len(coder.targets)
    mask = np.ones(T, dtype=bool)
    for t, a in enumerate(coder.targets):
        if a == "@who" or a in pair_ys:
            mask[t] = False
            continue
        modes, pure = set(), []
        for r in xs.recs.values():
            row = r.cx[t]
            n = float(row.sum())
            if n < 3.0:
                continue
            j = int(np.argmax(row))
            modes.add(j)
            pure.append(row[j] / n)
        if len(pure) >= 2 and min(pure) >= 0.8 and len(modes) >= 2:
            mask[t] = False
    return mask


def _exc_log2_e(xs: PE.ExcTracker, src: Hashable, mask: np.ndarray) -> float:
    """pevalue.ExcTracker.log2_e restricted to the unmasked targets."""
    r = xs.recs.get(src)
    if r is None:
        return -math.inf
    since = xs.cN - r.cN0
    used = (r.cx.sum(axis=1) > 0) & mask
    if not used.any():
        return -math.inf
    L0 = pmdl.ml_code_length_rows(since)
    L1 = r.Lx_own + (xs.LN - r.LN0 - r.Lx_N)
    return pmdl.log2_mean_exp2((L0 - L1)[used])


def _dist(obj: Any, t: float) -> Optional[Dict[Hashable, float]]:
    ss = getattr(obj, "ss", None) or getattr(obj, "values", None) or getattr(obj, "shapes", None) \
        or getattr(obj, "tpl", None)
    if ss is None or not hasattr(ss, "distribution"):
        return None
    keys, sh, other = ss.distribution(t)
    d = {k: float(v) for k, v in zip(keys, sh)}
    d["__other__"] = float(other)
    return d


def _num_occ(par: Any, child: Any, t: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    td_p, td_c = getattr(par, "td", None), getattr(child, "td", None)
    if td_p is None or td_c is None or td_p.total(t) <= 0 or td_c.total(t) <= 0:
        return None
    qs = [td_p.quantile(k / 8.0) for k in range(1, 8)]
    cp = np.asarray([td_p.cdf(q) for q in qs])
    cc = np.asarray([td_c.cdf(q) for q in qs])
    op = np.maximum(np.diff(np.r_[0.0, cp, 1.0]), 1e-9)
    oc = np.maximum(np.diff(np.r_[0.0, cc, 1.0]), 0.0)
    return oc, op


def _dists_pairs(a: PN.Node, b: PN.Node, t: float, who_level: Optional[int],
                 attrs: Optional[Iterable[str]] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Aligned (child-like, parent-like) distributions over the targets both
    nodes track, plus who (at the system's who level) and when."""
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    names = set(a.targets) & set(b.targets) if attrs is None else [x for x in attrs if x in a.targets and x in b.targets]
    for x in names:
        sa, sb = a.targets[x], b.targets[x]
        if isinstance(sa, PN.NumSummary) and isinstance(sb, PN.NumSummary):
            r = _num_occ(sb, sa, t)
            if r is not None:
                out.append(r)
            continue
        da, db = _dist(sa, t), _dist(sb, t)
        if da is None or db is None:
            continue
        keys = list(set(da) | set(db))
        out.append((np.asarray([da.get(k, 0.0) for k in keys]), np.asarray([db.get(k, 0.0) for k in keys])))
    if who_level is not None and attrs is None:
        ka, sha, oa = a.who.levels[who_level].distribution(t)
        kb, shb, ob = b.who.levels[who_level].distribution(t)
        if len(ka) and len(kb):
            da = dict(zip(ka, sha.tolist()))
            db = dict(zip(kb, shb.tolist()))
            keys = list(set(da) | set(db))
            out.append((np.r_[[da.get(k, 0.0) for k in keys], oa], np.r_[[db.get(k, 0.0) for k in keys], ob]))
    if attrs is None:
        ha, hb = a.when.hist.ravel(), b.when.hist.ravel()
        if ha.sum() > 0 and hb.sum() > 0:
            out.append((ha / ha.sum(), hb / hb.sum()))
    return out


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.maximum(np.asarray(p, dtype=float), 0.0)
    q = np.asarray(q, dtype=float)
    if p.sum() <= 0:
        return 0.0
    p = p / p.sum()
    q = (q + 1e-3) / (q + 1e-3).sum()
    m = p > 0
    return float((p[m] * np.log2(p[m] / q[m])).sum())


def _split_saving(par: PN.Node, kids: Sequence[PN.Node], t: float, who_level: Optional[int]) -> float:
    """Two-part MDL estimate of the H_m-decayed saving of a split (§6.6 prune):
    sum over children of n_c x sum_t KL(p_child,t || p_parent,t) minus the
    children's parameter cost sum_t (K - 1)/2 log2(n_c + 1), in bits (evidence units)."""
    tot = 0.0
    for c in kids:
        n = c.n_m(t)
        if n <= 0:
            continue
        for pc, pp in _dists_pairs(c, par, t, who_level):
            k = int((pc > 0).sum())
            tot += n * _kl(pc, pp) - 0.5 * max(0, k - 1) * math.log2(n + 1.0)
    return tot


def _pair_saving(par: PN.Node, x: PN.Node, t: float, attrs: Iterable[str]) -> float:
    n = x.n_m(t)
    if n <= 0:
        return 0.0
    tot = 0.0
    for pc, pp in _dists_pairs(x, par, t, None, attrs):
        k = int((pc > 0).sum())
        tot += n * _kl(pc, pp) - 0.5 * max(0, k - 1) * math.log2(n + 1.0)
    return tot


def _node_jsd(a: PN.Node, b: PN.Node, t: float, who_level: Optional[int]) -> float:
    pairs = _dists_pairs(a, b, t, who_level)
    if not pairs:
        return 1.0
    return float(np.mean([pmdl.jsd(p, q) for p, q in pairs]))


def _compact(sm: Any, t: float) -> Dict[str, Any]:
    if isinstance(sm, PN.NumSummary):
        td = sm.td
        if td.total(t) <= 0:
            return {"kind": "num"}
        return {"kind": "num", "log": sm.log,
                "q": [float(td.quantile(q)) for q in (0.01, 0.05, 0.5, 0.95, 0.99)],
                "moments": list(sm.moments(t))}
    d = _dist(sm, t)
    if d is None:
        return {"kind": getattr(sm, "kind", "?")}
    top = sorted(((k, v) for k, v in d.items() if k != "__other__"), key=lambda x: -x[1])[:8]
    return {"kind": getattr(sm, "kind", "cat"), "top": top, "other": d.get("__other__", 0.0)}


def _fitted_entry(mdl: Any, kind: int, nid: int) -> Any:
    if not isinstance(mdl, Mapping):
        return None
    for key in ("nodes", "by_node"):
        nodes = mdl.get(key)
        if isinstance(nodes, Mapping):
            sub = nodes.get(kind) if isinstance(nodes.get(kind), Mapping) else nodes
            v = sub.get(nid, sub.get(str(nid))) if isinstance(sub, Mapping) else None
            if v is not None:
                return v
    return None
