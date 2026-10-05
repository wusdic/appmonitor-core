"""P15 ResourceGovernor (`behavior.resource_governor`) — budgets, activity-
proportional caps, degradation ladder, idle-tree checkpointing, and the active
and earned sets of the bounded B-library (docs/lib3/progressive.md §6.19,
§6.20, §10.1-§10.2, card P15). Library 3 (behaviour), first behaviour engine.

Requirement S1 ("不能是遍历所有的用户和服务器的行为(一旦参数多或数据量大就会把资源消耗死)"):
memory and CPU are allocated from a global budget in proportion to what each
pattern tree (a system or a system family) is worth and how active it is; an
idle tree shrinks to tier XS and, after 30 idle days, leaves memory entirely
(checkpoint) until its next event; the B-library processes only the active and
earned IPs of a system in bounded mode.

Per tick (O(#trees + events of the tick + |active sets|), never O(#known IPs)):
  1. costs: each engine's latest run (engine health, counted once per run)
     into a 1-hour window -> P-core and lib-3 CPU shares. The cost is the
     deterministic price of the run's counted work (lib/pcost; config
     progressive.budget.cost_model 'counted', the default), so two identical
     runs take the same ladder steps and P12 charges its arms the same costs;
     the measured durations are kept beside it for operations (usage.*_wall_*,
     ops.budget pcore_wall_ms_h; cost_model 'wall' decides on them instead);
     every 15 min the P-core memory (model nbytes per tree) and
     store.memory_report().
  2. idle trees: a tree without events for 30 d is checkpointed
     (store.put_checkpoint(key, '__system__', 'ptree', t, {models})) and its
     models released; the first new batch of one of its systems restores it
     before P02/P04 run (P15 is the first behaviour engine).
  3. allocation (§6.19 step 1): every tree starts at XS; the remaining memory
     budget is water-filled, one tier step at a time, to the tree with the
     largest weight per MB of the step, weight = utility x criticality x
     learned-event rate (utility and criticality from P12's model.sysprof),
     up to the tier P12 recommends (demand). Caps follow the tier: n_max,
     l_max, e_rate, w_max, a_win, r_p, a_max, and LRU caps sized to the
     sources seen: s_sess = k_int = h_max = min(cap, 4 x sources in 7 d).
  4. ladder (§6.19 step 2): a budget exceeded on 3 consecutive ticks engages
     one step (at most one step per 3 ticks); 24 h under 70 % undoes one:
       1 halve e_rate  2 stop exploration (P12)  3 halve l_max  4 lower the
       tier  5 lower E_max (bounded mode only; skipped otherwise)  6 B29/B30/
       P13/P14 refresh on read only  7 P00 skips body parsing, P03 scores a
       1-in-k sample of events on IP-agnostic read-only nodes.
  5. sets (§10.1, bounded mode or progressive on): A_t(s) = IPs observed within
     the linger window (config lib3.linger_s, default 24 h, a per-system LRU of
     recently active IPs), IPs of open incidents, IPs with a B28 regime event
     in 7 d (quarantine / release); E_t(s) = the top-E_max IPs of B04's
     earned-gain records (model.earned) by gain per row x criticality, forced
     IPs first, demotion after 3 daily checks below tau_earn / 2.
Writes  model.budget@('__org__', '__org__') = {'version', 't', 'trees': {key: caps},
        'systems': {s: {'active', 'earned', 'e_max', 'ts', 'n_active',
        'n_earned_candidates'}}, 'ladder': {'step', 'no_explore', 'refresh_on_read',
        'skip_body_parsing', 'score_sample_k'}, 'usage': {...}, 'evicted': {key: ts}}
        ops.budget@(s, '__system__') every 15 min for active systems:
        {'tier', 'tree_bytes', 'events_h', 'pcore_ms_h', 'share'}.
Inert unless config['progressive']['enabled'] or lib3.resource_mode == 'bounded'.
"""
from __future__ import annotations

import ipaddress
import math
from collections import OrderedDict, deque
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ...core.engine import Context, Engine
from ...models.schema import ORG, SYSTEM_ENTITY, DerivedMetric, MetricKind, is_pseudo_entity
from .lib import m_ptree as MP
from .lib import pactive as PA
from .lib import pcost as PC
from .lib import pevent as EV
from .lib import psketch as PS
from .lib import pstrategy as PSt

STATE = "model.budget_state"
DAY = PS.DAY
HOUR = 3600.0
MEM_EVERY_S = 900.0
MEM_TREE_S = 3600.0              # a changed tree's deep size is re-measured at most hourly
STORE_REPORT_S = 6 * 3600.0     # store.memory_report cadence (it is O(stored points))
IDLE_XS_S = DAY                  # no event for a day -> tier XS
IDLE_EVICT_S = 30 * DAY          # no event for 30 days -> checkpoint and release
LINGER_S = DAY
RELEASE_IDLE_S = 7 * DAY         # bounded mode: per-IP decision-chain state of an IP idle this long is released
RELEASE_MODELS = ("model.calib", "model.governor", "model.control")
OVER_TICKS = 3
UNDER_FRAC = 0.7
UNDER_S = DAY
MAX_STEP = 7
TIER_MB = {"XS": 1.0, "S": 5.0, "M": 30.0, "L": 60.0}
TIER_CAPS = {                    # per tier: l_max, w_max, a_win, r_p, a_max
    "XS": (2, 64, 32, 256, 128),
    "S": (16, 256, 64, 1024, 256),
    "M": (64, 512, 96, 4096, 512),
    "L": (128, 1024, 96, 8192, 512),
}
SRC_CAPS = {"s_sess": 65536, "k_int": 16384, "h_max": 65536}
TAU_EARN = 2.0                   # bits per row (§10.2)
N_ROWS_EARN = 48
DEMOTE_CHECKS = 3
AUTOMATION_FORCE = 0.6            # machine clients are few and distinct (§10.2)
SCORE_SAMPLE_K = 4
P_ENGINES = frozenset({"raw.event", "derived.event_context", "behavior.resource_governor",
                       "behavior.attr_registry", "behavior.attr_select", "behavior.conformity",
                       "behavior.pattern_tree", "behavior.content_bounds", "behavior.payload_grammar",
                       "behavior.binding", "behavior.time_window", "behavior.workflow",
                       "behavior.who_groups", "behavior.system_profile", "behavior.facets",
                       "behavior.views"})
TREE_MODELS = (MP.PTREE, MP.ATTR, MP.ATTRSEL, MP.PWANT, MP.PBOUNDS, MP.PGRAMMAR, MP.PBIND,
               MP.PWIN, MP.PFLOW)


class GovState:
    """P15's private state (bounded: O(#engines x ticks in 1 h + #trees + the
    linger sets of recently active IPs))."""

    def __init__(self) -> None:
        self.health_seen: Dict[str, float] = {}
        self.cpu: Deque[Tuple[float, float, float]] = deque()      # (ts, pcore ms, lib3 ms)
        self.t_first: Optional[float] = None
        self.mem_t: Optional[float] = None
        self.mem_tree: Dict[str, int] = {}
        self.mem_meas: Dict[str, float] = {}                         # tree key -> ts of its last deep size
        self.mem_pcore = 0
        self.mem_store: Dict[str, Any] = {}
        self.store_t: Optional[float] = None
        self.step = 0
        self.over = 0
        self.last_change: Optional[float] = None
        self.under_since: Optional[float] = None
        self.ticks_since_change = 0
        self.last_event: Dict[str, float] = {}                     # tree key -> ts
        self.rate: Dict[str, PS.DecayedVector] = {}                # tree key -> (events, learned) at H_s
        self.linger: Dict[str, "OrderedDict[str, float]"] = {}     # system -> ip -> last ts
        self.evicted: Dict[str, float] = {}
        self.earn_low: Dict[str, Dict[str, int]] = {}
        self.earned: Dict[str, List[str]] = {}
        self.earn_day: Dict[str, int] = {}
        self.engine_ms: Dict[str, Deque[Tuple[float, float]]] = {}      # cost the decisions see (lib/pcost)
        self.engine_wall_ms: Dict[str, Deque[Tuple[float, float]]] = {}  # measured, for operations
        self.cpu_wall: Deque[Tuple[float, float, float]] = deque()       # (ts, pcore ms, lib3 ms) measured
        self.tick_ev: "OrderedDict[float, float]" = OrderedDict()        # tick ts -> org events (2 h)
        self.regime: Dict[str, Tuple[float, Any]] = {}             # system -> (ts, IPs with a B28 regime event in 7 d)
        self.keep: Dict[str, "OrderedDict[str, float]"] = {}       # system -> ip -> last ts (7 d, bounded mode)
        self.released = 0

    def nbytes(self) -> int:
        n = 200 + 24 * len(self.cpu) + 64 * len(self.last_event) + 100 * len(self.rate)
        n += sum(80 * len(v) for v in self.linger.values())
        n += sum(80 * len(v) for v in (getattr(self, "keep", None) or {}).values())
        n += sum(24 * len(v) for v in self.engine_ms.values())
        n += sum(24 * len(v) for v in (getattr(self, "engine_wall_ms", None) or {}).values())
        n += 24 * len(getattr(self, "cpu_wall", ()) or ()) + 32 * len(getattr(self, "tick_ev", ()) or ())
        return int(n)


def _enabled(config: Mapping[str, Any]) -> bool:
    return EV.enabled(config) or PA.bounded(config)


class ResourceGovernorEngine(Engine):
    name = "behavior.resource_governor"
    layer = "behavior"
    consumes = ["engine.health", MP.SYSPROF, MP.SYSFAM, PA.EARNED, EV.EVT_BATCH, "observations",
                "incidents", "event.regime"]
    produces = [MP.BUDGET, MP.OPS_BUDGET, STATE]
    description = ("P15: budgets, activity-proportional caps per tree, degradation ladder, "
                   "idle-tree checkpointing, active / earned sets of the bounded B-library")
    interval = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.last_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------ run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        cfg = ctx.config
        if not _enabled(cfg):
            return 0
        store = ctx.store
        now, dt = float(ctx.now), float(ctx.window_s)
        st = store.get_model(ORG, ORG, STATE)
        if not isinstance(st, GovState):
            st = GovState()
            store.put_model(ORG, ORG, STATE, st, version=1, ts=now)
        if st.t_first is None:
            st.t_first = now
        bud = _budget_cfg(cfg)
        self._costs(store, st, now, bud)
        sources = self._tick_sources(store, now, dt)
        restored = self._restore(store, st, sources, now)
        self._activity(store, st, sources, now)
        if st.mem_t is None or now - st.mem_t >= MEM_EVERY_S:
            self._memory(store, st, now)
        over, frac = self._over_budget(st, bud, now)
        self._ladder(st, over, frac, now, PA.bounded(cfg))
        evicted = self._evict_idle(store, st, now)
        trees = self._allocate(store, st, bud, now)
        systems = self._sets(store, st, sources, now, cfg) if (PA.bounded(cfg) or EV.enabled(cfg)) else {}
        released = self._release_idle(store, st, sources, systems, now) if PA.bounded(cfg) else 0
        ladder = self._ladder_flags(st, PA.bounded(cfg))
        old = store.get_model(ORG, ORG, MP.BUDGET)
        version = int((old or {}).get("version", 0)) + 1 if isinstance(old, Mapping) else 1
        cpu_p, cpu_l, span = self._cpu_share(st, now)
        wall_p, wall_l, _ = self._cpu_share(st, now, wall=True)
        model = {"fmt": 1, "version": version, "t": now, "trees": trees, "systems": systems,
                 "ladder": ladder,
                 "usage": {"pcore_cpu_share": cpu_p, "lib3_cpu_share": cpu_l, "window_s": span,
                           "cost_model": PC.mode(bud), "pcore_wall_share": wall_p,
                           "lib3_wall_share": wall_l,
                           "pcore_mem_mb": st.mem_pcore / 1e6, "over": over, "frac": frac,
                           "store": dict(st.mem_store), "gov_bytes": st.nbytes()},
                 "budget": dict(bud), "evicted": dict(st.evicted)}
        store.put_model(ORG, ORG, MP.BUDGET, model, version=version, ts=now)
        if st.mem_t == now:
            self._ops_budget(store, st, trees, now, dt)
        self.last_stats = {"trees": len(trees), "systems": len(systems), "step": st.step,
                           "restored": restored, "evicted": evicted, "pcore_cpu_share": cpu_p,
                           "released": released}
        return len(trees)

    # ---------------------------------------------------------------- costs
    def _costs(self, store: Any, st: GovState, now: float, bud: Optional[Mapping[str, Any]] = None) -> None:
        """Each engine's latest run (engine health, counted once per run) into
        a 1-hour window: the cost the decisions see (lib/pcost: modelled from
        the run's own count and the events of its tick in mode 'counted', the
        default; the measured duration in mode 'wall') and, for operations,
        the measured duration. Engines that ran before P15 in this tick carry
        this tick's timestamp, the others the previous tick's."""
        if getattr(st, "engine_wall_ms", None) is None:          # state from an older version
            st.engine_wall_ms, st.cpu_wall, st.tick_ev = {}, deque(), OrderedDict()
        ev = 0.0
        for s in store.batch_systems(EV.EVT_BATCH):
            b = store.batch_at(s, EV.EVT_BATCH, now)
            if b is not None:
                ev += float(b.n)
        st.tick_ev[now] = ev
        while st.tick_ev and next(iter(st.tick_ev)) <= now - 2 * HOUR:
            st.tick_ev.popitem(last=False)
        pc = lc = pw = lw = 0.0
        for eng, rec in store.health().items():
            ts = rec.get("ts")
            if ts is None or st.health_seen.get(eng) == ts:
                continue
            st.health_seen[eng] = ts
            wall = float(rec.get("duration_ms") or 0.0)
            ms = PC.run_cost_ms(eng, rec, st.tick_ev.get(ts, 0.0), bud)
            if eng in P_ENGINES:
                pc += ms
                pw += wall
            if eng.startswith("behavior."):
                lc += ms
                lw += wall
            for book, v in ((st.engine_ms, ms), (st.engine_wall_ms, wall)):
                q = book.setdefault(eng, deque())
                q.append((now, v))
                while q and q[0][0] <= now - HOUR:
                    q.popleft()
        for cpu, p, l in ((st.cpu, pc, lc), (st.cpu_wall, pw, lw)):
            cpu.append((now, p, l))
            while cpu and cpu[0][0] <= now - HOUR:
                cpu.popleft()

    @staticmethod
    def _cpu_share(st: GovState, now: float, wall: bool = False) -> Tuple[float, float, float]:
        span = min(HOUR, max(1.0, now - (st.t_first or now)))
        cpu = (getattr(st, "cpu_wall", None) or ()) if wall else st.cpu
        if not cpu:
            return 0.0, 0.0, span
        span = max(span, 1.0)
        p = sum(x[1] for x in cpu) / 1000.0 / span
        l = sum(x[2] for x in cpu) / 1000.0 / span
        return float(p), float(l), float(span)

    def _memory(self, store: Any, st: GovState, now: float) -> None:
        """P-core memory per tree. A tree's deep size is re-measured when it
        changed (an event since its last measure) and its measure is older
        than MEM_TREE_S, or when it is new: walking every tree every 15 min
        was 83 % of P15's time on pack O (ptree / registry nbytes, §16.12) for
        a figure that moves slowly (the ladder needs 3 ticks over budget)."""
        keys = set(st.last_event) | {MP.tree_key(store, s) for s in store.batch_systems(EV.EVT_BATCH)}
        if getattr(st, "mem_meas", None) is None:
            st.mem_meas = {}
        tot = 0
        old = st.mem_tree
        st.mem_tree = {}
        for k in sorted(keys):
            if k in st.evicted:
                continue
            t_m = st.mem_meas.get(k)
            if k in old and t_m is not None and (st.last_event.get(k, -math.inf) <= t_m
                                                 or now - t_m < MEM_TREE_S):
                b = old[k]
            else:
                b = _tree_bytes(store, k)
                st.mem_meas[k] = now
            st.mem_tree[k] = b
            tot += b
        for k in [k for k in st.mem_meas if k not in st.mem_tree]:
            st.mem_meas.pop(k, None)
        # store.memory_report() walks every stored point (O(all series), i.e. it
        # grows with the known IPs' retained data): taken every STORE_REPORT_S
        # only, its batch bytes carried in between
        if st.store_t is None or now - st.store_t >= STORE_REPORT_S:
            try:
                rep = store.memory_report()
                st.mem_store = {"approx_bytes": rep.get("approx_bytes"),
                                "batch_bytes": rep.get("batch_bytes"), "entities": rep.get("entities")}
            except Exception:
                st.mem_store = {}
            st.store_t = now
        tot += int(st.mem_store.get("batch_bytes") or 0)
        st.mem_pcore = tot
        st.mem_t = now

    # ------------------------------------------------------------- activity
    @staticmethod
    def _tick_sources(store: Any, now: float, dt: float) -> Dict[str, Set[str]]:
        """Source IPs observed at this tick per system: the evt.batch IPs when P00
        ran, else the newest observations (walked back to the tick start; the
        observation deque is bounded, so a saturated deque falls back to the
        store's last_seen index)."""
        out: Dict[str, Set[str]] = {}
        for s in store.batch_systems(EV.EVT_BATCH):
            b = store.batch_at(s, EV.EVT_BATCH, now)
            if b is not None:
                out.setdefault(s, set()).update(b.ips)
        lim = 1024
        while True:
            obs = store.recent_observations(lim)
            if not obs:
                break
            for o in obs:
                if o.ts > now - dt - 1e-6 and not is_pseudo_entity(o.entity):
                    out.setdefault(o.system, set()).add(o.entity)
            if len(obs) < lim or obs[0].ts <= now - dt:
                break
            if lim >= 1 << 20:
                for s in store.systems():
                    out.setdefault(s, set()).update(store.entities_active(s, now))
                break
            lim *= 4
        return out

    def _activity(self, store: Any, st: GovState, sources: Mapping[str, Set[str]], now: float) -> None:
        for s in store.batch_systems(EV.EVT_BATCH):
            b = store.batch_at(s, EV.EVT_BATCH, now)
            if b is None or b.n == 0:
                continue
            k = MP.tree_key(store, s)
            st.last_event[k] = now
            dv = st.rate.get(k)
            if dv is None:
                dv = st.rate[k] = PS.DecayedVector([PS.H_S, PS.H_S])
            dv.add(now, [float(b.n), float(len(b.learned_rows()))])
        for s, ips in sources.items():
            lru = st.linger.setdefault(s, OrderedDict())
            for ip in ips:
                if ip in lru:
                    lru.move_to_end(ip)
                lru[ip] = now

    def _release_idle(self, store: Any, st: GovState, sources: Mapping[str, Set[str]],
                      systems: Mapping[str, Any], now: float) -> int:
        """Bounded mode (§10.1-§10.3): the per-IP decision-chain state of an IP
        that has been idle for RELEASE_IDLE_S (7 d) is released - B24/B25's
        model.calib (bookkeeping, meta rings, CUSUM), B28's model.governor /
        model.control - unless the IP is earned, in an open incident or had a
        B28 regime event within 7 d (P15's active set). The IPs come from a
        per-system LRU of the sources seen in the last 7 days, updated from the
        tick's sources, so the work is O(sources of the tick + expiries), never
        a scan of the known IPs; memory of the chain is O(|E_t| + |A_7d|)
        instead of O(every IP ever seen). A returning IP starts from its
        class's pooled rings (B24) and an empty CUSUM, as a new IP does."""
        if getattr(st, "keep", None) is None:
            st.keep = {}
        n = 0
        for s, ips in sources.items():
            lru = st.keep.setdefault(s, OrderedDict())
            for ip in ips:
                if ip in lru:
                    lru.move_to_end(ip)
                lru[ip] = now
        for s in list(st.keep):
            lru = st.keep[s]
            rec = systems.get(s) or {}
            protect = set(rec.get("earned") or ()) | set(rec.get("active") or ())
            while lru:
                ip, ts = next(iter(lru.items()))
                if ts >= now - RELEASE_IDLE_S:
                    break
                lru.popitem(last=False)
                if ip in protect or is_pseudo_entity(ip):
                    lru[ip] = now                      # protected: re-examined after another window
                    continue
                for name in RELEASE_MODELS:
                    if store.get_model(s, ip, name) is not None:
                        store.put_model(s, ip, name, None, ts=now)
                        n += 1
            if not lru:
                st.keep.pop(s, None)
        st.released = int(getattr(st, "released", 0)) + n
        return n

    def _restore(self, store: Any, st: GovState, sources: Mapping[str, Set[str]], now: float) -> List[str]:
        out = []
        for s in store.batch_systems(EV.EVT_BATCH):
            if store.batch_at(s, EV.EVT_BATCH, now) is None:
                continue
            k = MP.tree_key(store, s)
            if k not in st.evicted:
                continue
            ck = store.get_checkpoint(k, SYSTEM_ENTITY, MP.CHECKPOINT)
            if ck is not None and isinstance(ck[1], Mapping) and ck[1].get("p15_evicted"):
                for name, obj in (ck[1].get("models") or {}).items():
                    store.put_model(k, SYSTEM_ENTITY, name, obj, ts=now)
            st.evicted.pop(k, None)
            out.append(k)
        return out

    def _evict_idle(self, store: Any, st: GovState, now: float) -> List[str]:
        out = []
        for k, t in list(st.last_event.items()):
            if k in st.evicted or now - t < IDLE_EVICT_S:
                continue
            models = {n: store.get_model(k, SYSTEM_ENTITY, n) for n in TREE_MODELS}
            models = {n: m for n, m in models.items() if m is not None}
            if models:
                store.put_checkpoint(k, SYSTEM_ENTITY, MP.CHECKPOINT, now,
                                     {"p15_evicted": True, "models": models, "last_event": t})
                for n in models:
                    store.put_model(k, SYSTEM_ENTITY, n, None, ts=now)
            st.evicted[k] = now
            st.rate.pop(k, None)
            out.append(k)
        return out

    # ------------------------------------------------------------ allocation
    def _allocate(self, store: Any, st: GovState, bud: Mapping[str, Any], now: float) -> Dict[str, Any]:
        # a key whose tree was released (its system joined a family) is forgotten
        for k in [k for k in st.last_event if k not in st.evicted and MP.get_ptree(store, k) is None
                  and st.last_event[k] < now]:
            st.last_event.pop(k, None)
            st.rate.pop(k, None)
        keys = [k for k in st.last_event if k not in st.evicted]
        info: Dict[str, Dict[str, Any]] = {}
        for k in keys:
            sp = MP.get_model(store, k, MP.SYSPROF)
            ch = (sp or {}).get("chosen") if isinstance(sp, Mapping) else None
            chars = (sp or {}).get("characteristics") if isinstance(sp, Mapping) else None
            rate = st.rate.get(k)
            ev_day, learned_day = (0.0, 0.0)
            if rate is not None:
                r = rate.read(now)
                f = math.log(2) / PS.H_S * DAY            # decayed sum -> per day
                ev_day, learned_day = float(r[0]) * f, float(r[1]) * f
            idle = now - st.last_event.get(k, now) >= IDLE_XS_S
            demand = "XS" if idle else (str((ch or {}).get("tier")) if (ch or {}).get("tier") in TIER_MB
                                        else PSt.recommend_tier(_n_nodes(store, k), learned_day))
            util = _utility(sp)
            crit = float((chars or {}).get("criticality", 1.0) or 1.0) if isinstance(chars, Mapping) else 1.0
            info[k] = {"demand": demand, "weight": max(1e-6, util * crit * max(learned_day, 1.0)),
                       "ev_day": ev_day, "learned_day": learned_day, "idle": idle,
                       "sources": float((chars or {}).get("population", 0.0) or 0.0)
                       if isinstance(chars, Mapping) else 0.0, "chosen": ch or {}}
        total = float(bud.get("mem_mb_total", 2048.0) or 2048.0) * 0.8
        tier = {k: "XS" for k in info}
        left = total - TIER_MB["XS"] * len(info)
        order = list(PSt.TIER_ORDER)
        while left > 0:
            best, best_v = None, 0.0
            for k, v in info.items():
                i = order.index(tier[k])
                if i >= order.index(v["demand"]) or i + 1 >= len(order):
                    continue
                step_mb = TIER_MB[order[i + 1]] - TIER_MB[order[i]]
                if step_mb > left:
                    continue
                val = v["weight"] / step_mb
                if val > best_v:
                    best, best_v = k, val
            if best is None:
                break
            i = order.index(tier[best])
            left -= TIER_MB[order[i + 1]] - TIER_MB[order[i]]
            tier[best] = order[i + 1]
        out: Dict[str, Any] = {}
        e_rate0 = float(EV.PROGRESSIVE_DEFAULTS["defaults"]["e_rate"])
        for k, v in info.items():
            t = tier[k]
            if st.step >= 4 and t != "XS":                    # ladder 4: one tier down
                t = order[order.index(t) - 1]
            l_max, w_max, a_win, r_p, a_max = TIER_CAPS[t]
            if st.step >= 3:
                l_max = max(1, l_max // 2)
            e_rate = e_rate0 / (2.0 if st.step >= 1 else 1.0)
            src = max(64.0, 4.0 * v["sources"])
            caps = {"tier": t, "n_max": PSt.TIER_NODES[t], "l_max": l_max, "e_rate": e_rate,
                    "w_max": w_max, "a_win": a_win, "r_p": r_p, "a_max": a_max,
                    "demand": v["demand"], "idle": v["idle"], "ev_day": round(v["ev_day"], 1),
                    "learned_day": round(v["learned_day"], 1), "weight": v["weight"],
                    "bytes": st.mem_tree.get(k)}
            for name, cap in SRC_CAPS.items():
                caps[name] = int(min(cap, src))
            if st.step >= 7:
                caps["skip_body_parsing"] = True
                caps["score_sample_k"] = SCORE_SAMPLE_K
            out[k] = caps
        return out

    # --------------------------------------------------------------- ladder
    def _over_budget(self, st: GovState, bud: Mapping[str, Any], now: float) -> Tuple[bool, float]:
        cpu_p, cpu_l, span = self._cpu_share(st, now)
        fr = []
        share = bud.get("pcore_cpu_share")
        if share and span >= 0.25 * HOUR:
            fr.append(cpu_p / float(share))
        lshare = bud.get("lib3_cpu_share")
        if lshare and span >= 0.25 * HOUR:
            fr.append(cpu_l / float(lshare))
        mem = bud.get("mem_mb_total")
        if mem and st.mem_pcore:
            fr.append(st.mem_pcore / 1e6 / float(mem))
        frac = max(fr) if fr else 0.0
        return frac > 1.0, float(frac)

    @staticmethod
    def _ladder(st: GovState, over: bool, frac: float, now: float, bounded: bool) -> None:
        st.ticks_since_change += 1
        if over:
            st.over += 1
            st.under_since = None
            if st.over >= OVER_TICKS and st.ticks_since_change >= OVER_TICKS and st.step < MAX_STEP:
                st.step += 1
                if st.step == 5 and not bounded:             # step 5 only in bounded mode
                    st.step = 6
                st.ticks_since_change = 0
                st.last_change = now
        else:
            st.over = 0
            if frac < UNDER_FRAC:
                if st.under_since is None:
                    st.under_since = now
                if st.step > 0 and now - st.under_since >= UNDER_S:
                    st.step -= 1
                    if st.step == 5 and not bounded:
                        st.step = 4
                    st.under_since = now
                    st.ticks_since_change = 0
                    st.last_change = now
            else:
                st.under_since = None

    @staticmethod
    def _ladder_flags(st: GovState, bounded: bool) -> Dict[str, Any]:
        return {"step": st.step, "no_explore": st.step >= 2, "lower_e_max": st.step >= 5 and bounded,
                "refresh_on_read": st.step >= 6, "skip_body_parsing": st.step >= 7,
                "score_sample_k": SCORE_SAMPLE_K if st.step >= 7 else 1,
                "last_change": st.last_change}

    # ------------------------------------------------------------------ sets
    def _sets(self, store: Any, st: GovState, sources: Mapping[str, Set[str]], now: float,
              cfg: Mapping[str, Any]) -> Dict[str, Any]:
        linger = PA.linger_s(cfg)
        out: Dict[str, Any] = {}
        systems = set(st.linger) | set(sources)
        for s in systems:
            lru = st.linger.get(s) or OrderedDict()
            while lru and next(iter(lru.values())) < now - linger:
                lru.popitem(last=False)
            if not lru and s not in sources:
                st.linger.pop(s, None)
                continue
            act = set(lru)
            for inc in store.incidents(system=s, status="open"):
                act.add(inc.entity)
                act.update(getattr(inc, "entities", None) or ())
            reg = st.regime.get(s)
            if reg is None or now - reg[0] >= HOUR or now < reg[0]:
                reg = st.regime[s] = (now, frozenset(
                    ev.entity for ev in store.events(system=s, since=now - 7 * DAY, kinds=["regime"],
                                                     limit=10000)))
            act |= set(reg[1])
            act = {e for e in act if e and not is_pseudo_entity(e)}
            earned, n_cand = self._earned(store, st, s, now, cfg)
            out[s] = {"ts": now, "active": sorted(act), "earned": earned, "n_active": len(act),
                      "n_earned_candidates": n_cand, "e_max": _e_max(store, s, st.step, PA.bounded(cfg))}
        return out

    def _earned(self, store: Any, st: GovState, s: str, now: float, cfg: Mapping[str, Any]
                ) -> Tuple[List[str], int]:
        """E_t(s) from B04's shadow records (model.earned@(s,'__system__')):
        {ip: {'g': gain bits, 'n': rows, 'forced': bool, 'crit': float}}; re-evaluated
        once per local day, with demotion hysteresis."""
        rec = store.get_model(s, SYSTEM_ENTITY, PA.EARNED)
        day = int(now // DAY)
        if st.earn_day.get(s) == day and s in st.earned:
            n_cand = sum(1 for r in ((rec or {}).get("ips") or {}).values()
                         if float(r.get("n", 0) or 0) >= N_ROWS_EARN
                         and float(r.get("g", 0) or 0) / max(float(r.get("n", 1) or 1), 1e-9) >= TAU_EARN) \
                if isinstance(rec, Mapping) else 0
            return st.earned[s], n_cand
        forced = self._forced(store, st, s, now, cfg)
        ips = dict((rec or {}).get("ips") or {}) if isinstance(rec, Mapping) else {}
        for ip in forced:
            ips.setdefault(ip, {"g": 0.0, "n": 0.0})
        cands = []
        for ip, r in ips.items():
            n = float(r.get("n", 0.0) or 0.0)
            g = float(r.get("g", 0.0) or 0.0)
            gpr = g / n if n > 0 else 0.0
            fz = bool(r.get("forced")) or ip in forced
            ok = fz or (n >= N_ROWS_EARN and gpr >= TAU_EARN)
            cands.append((ip, gpr, fz, ok, float(r.get("crit", 1.0) or 1.0), n))
        n_cand = sum(1 for c in cands if c[3])
        st.earn_day[s] = day
        prev = set(st.earned.get(s, []))
        low = st.earn_low.setdefault(s, {})
        keep = []
        for ip, gpr, forced, ok, crit, n in cands:
            if ok:
                low.pop(ip, None)
                keep.append((ip, gpr, forced, crit))
            elif ip in prev:
                if gpr < TAU_EARN / 2.0:
                    low[ip] = low.get(ip, 0) + 1
                if low.get(ip, 0) < DEMOTE_CHECKS:
                    keep.append((ip, gpr, forced, crit))
                else:
                    low.pop(ip, None)
        emax = _e_max(store, s, st.step, PA.bounded(cfg))
        # forced IPs first, then incumbents (a slot is not lost to a newcomer
        # while its holder is within its demotion hysteresis), then by gain
        keep.sort(key=lambda x: (not x[2], x[0] not in prev, -x[1] * x[3], x[0]))
        st.earned[s] = sorted(x[0] for x in keep[:emax])
        return st.earned[s], n_cand

    @staticmethod
    def _forced(store: Any, st: GovState, s: str, now: float, cfg: Mapping[str, Any]) -> Set[str]:
        """IPs whose per-entity models are kept whatever their gain (§10.2):
        an open incident; a P04 single-IP exception (P04-distinctive); an IP of
        an ip_class with criticality 'high' among the recently active; an
        automation index >= 0.6 (derived.periodicity_score) among the recently
        active. O(active + incidents + exceptions), never O(#known IPs)."""
        out: Set[str] = set()
        for inc in store.incidents(system=s, status="open"):
            if inc.entity and not is_pseudo_entity(inc.entity):
                out.add(inc.entity)
        m = MP.get_ptree(store, MP.tree_key(store, s))
        if m is not None:
            for tr in m.kinds.values():
                for nd in tr.nodes.values():
                    for ip in (nd.exc or {}):
                        out.add(str(ip))
        recent = list((st.linger.get(s) or {}).keys())
        crit_nets = []
        for r in (cfg.get("ip_classes") or ()):
            if isinstance(r, Mapping) and str(r.get("criticality", "")).lower() in ("high", "critical"):
                for c in r.get("cidrs") or ():
                    try:
                        crit_nets.append(ipaddress.ip_network(str(c), strict=False))
                    except ValueError:
                        continue
        for ip in recent[-4096:]:
            if crit_nets:
                try:
                    a = ipaddress.ip_address(ip)
                    if any(a.version == n.version and a in n for n in crit_nets):
                        out.add(ip)
                        continue
                except ValueError:
                    pass
            v = store.latest_derived(s, ip, "derived.periodicity_score")
            val = getattr(v, "value", None)
            if isinstance(val, (int, float)) and val >= AUTOMATION_FORCE:
                out.add(ip)
        return out

    # --------------------------------------------------------------- ops
    def _ops_budget(self, store: Any, st: GovState, trees: Mapping[str, Any], now: float, dt: float) -> None:
        store.ensure_retention(MP.OPS_BUDGET, max_age_s=8 * DAY)
        eng = {e: round(sum(x[1] for x in q), 3) for e, q in st.engine_ms.items() if q}
        pms = sum(v for e, v in eng.items() if e in P_ENGINES)
        wms = sum(sum(x[1] for x in q) for e, q in (getattr(st, "engine_wall_ms", None) or {}).items()
                  if e in P_ENGINES)
        tot_ev = sum(float(c.get("ev_day") or 0.0) for c in trees.values()) or 1.0
        for s in store.batch_systems(EV.EVT_BATCH):
            k = MP.tree_key(store, s)
            c = trees.get(k)
            if c is None or c.get("idle"):
                continue
            share = float(c.get("ev_day") or 0.0) / tot_ev
            store.add_derived(DerivedMetric(
                name=MP.OPS_BUDGET, value={"tier": c["tier"], "tree_bytes": st.mem_tree.get(k),
                                           "events_day": c.get("ev_day"), "share": round(share, 4),
                                           "pcore_ms_h": round(pms * share, 3),
                                           "pcore_wall_ms_h": round(wms * share, 3), "step": st.step},
                ts=now, system=s, entity=SYSTEM_ENTITY, window_s=int(dt), kind=MetricKind.CATEGORICAL))


def _budget_cfg(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(EV.pconfig(cfg).get("budget") or {})


def _tree_bytes(store: Any, key: str) -> int:
    tot = 0
    for n in TREE_MODELS:
        m = store.get_model(key, SYSTEM_ENTITY, n)
        if m is None:
            continue
        fn = getattr(m, "nbytes", None)
        if callable(fn):
            try:
                tot += int(fn())
                continue
            except Exception:
                pass
        tot += 2048
    return tot


def _n_nodes(store: Any, key: str) -> float:
    m = MP.get_ptree(store, key)
    return float(sum(len(t.nodes) for t in m.kinds.values())) if m is not None else 0.0


def _utility(sp: Any) -> float:
    """Utility of a tree for the allocation: 1 + the positive utilities (bits /
    event) of its chosen arms as P12 measured them (1 before P12 ran)."""
    if not isinstance(sp, Mapping):
        return 1.0
    arms, ch = sp.get("arms") or {}, sp.get("chosen") or {}
    u = 0.0
    for dim, arm in ch.items():
        rec = (arms.get(dim) or {}).get(arm) if isinstance(arms.get(dim), Mapping) else None
        v = (rec or {}).get("U") if isinstance(rec, Mapping) else None
        if isinstance(v, (int, float)) and math.isfinite(v) and v > 0 and dim != "who":
            u += float(v)
    return 1.0 + u


def _e_max(store: Any, s: str, step: int, bounded: bool) -> int:
    sp = store.get_model(s, SYSTEM_ENTITY, MP.SYSPROF)
    v = ((sp or {}).get("chosen") or {}).get("e_max") if isinstance(sp, Mapping) else None
    try:
        e = int(v) if v is not None else 32
    except (TypeError, ValueError):
        e = 32
    if bounded and step >= 5:
        e = max(PSt.E_MAX_ARMS[0], e // 4)
    return e
