"""P11 WhoGroupsEngine (`behavior.who_groups`) — who belongs together
(docs/lib3/progressive.md §6.15, card P11). Library 3 (behaviour).

Requirement S5/S11/S12/S14/S15 ("综合部的3个用户", "某几个IP段或者某几个IP区域，或者
直接IP不作为特征"): the members of a behavioural group are LEARNED from what the
IPs do, never listed in code; only its NAME needs a source (operator config,
IPAM / DHCP-scope / asset import). Groups sharpen with observation time (more
evidence per signature, stable ids, two-run move rule) and follow behaviour
(H_m-decayed signatures, daily re-clustering).

Per tick (O(learned rows of tick t - D); nothing iterates over IPs):
  signatures  every LEARNED, trusted row of evt.batch of tick t' <= t - D adds
              two items to the signature of its source: the ACTION
              '<tree key>|<route>' (P03's pat.assign `act`, else lib/pdfg.route_key)
              and the PATTERN '<tree key>|@<node>' of its covering confident node
              (P03's `conf`; the IP x pattern co-occurrence: as P04 specialises
              a department's login into its own node, its members' signatures
              diverge from other departments'). The pattern is the nearest
              node of the covering node's path reached by a who / when split
              (_pattern_node; none without one): payload-bin and action-split
              nodes (net.bytes_up bins, GET / POST) split SALES on pack O
              (ARI 1.0 -> 0.53 -> 0.49 on days 9 / 11 / 21).
              Mass w/pi x trust x damp and
              evidence units trust x damp / (r + 1) (burst run (source, item), PPC-9).
              Source = the IP for systems whose who mode is ip / grp, the /24
              for prefix mode; reg / none systems contribute nothing (§6.15.1).
              `shared:` IPs (B17 shared_ip) get no signature. Quarantined IPs'
              rows add nothing. Store: lib/pminhash.SigStore, LRU S_max.
  readdress   P03's pat.assign `readdr` (a never-seen IP submitting the bound
              value of a member that went silent, §6.12) joins the old IP's group
              at once, flagged provisional (§6.15 item 3).
  joins       an ungrouped IP that reaches 10 evidence units joins the group
              holding >= 60 % of the J-weight of its 10 nearest group
              representatives when its mean J to them reaches RHO x the group's
              cohesion (one label-propagation step, same rule as the merge);
              <= 256 per tick, never-tried sources first, then the longest
              waiting (an address-ordered cut starved a SALES member isolated
              by one clustering run behind thousands of public visitors).
Daily (entity_due, 24 h) — the clustering:
  eligible    signatures with >= EV_CLUSTER evidence units; idle 30 d or < 10
              units after 7 d are dropped (§6.15.1).
  similarity  weighted MinHash (ICWS, k = 64) of the normalised log2(1 + mass)
              item weights (the action mix), LSH 16 x 4, mutual 10-NN graph with
              J >= 0.2 (lib/pminhash); identical signatures are one node with a
              self-loop for their multiplicity.
  groups      Louvain (resolution 1, IP order) per connected component, its
              first level (local moves) as the base communities (= the display
              sub-groups); communities are then merged
              (a) when their members are as similar across as within: J(A, B) >=
                  RHO x max(J(A, A), J(B, B)) (lib/plouvain.merge_similar), and
              (b) when both are ephemeral-address populations (addresses active on
                  <= 1/3 of the days since first seen) inside one pure prefix
                  (a DHCP / VPN pool, lib/plouvain.merge_local);
              (c) when J(A, B) >= RHO x min(J(A, A), J(B, B)) AND one pure prefix
                  (>= /16) covers 90 % of A u B: the address plan corroborates a
                  looser behavioural bar (lib/plouvain.merge_prefix). Measured on
                  pack O: 5 of SALES' 20 members (no document comments) split off
                  from day 17 on both seeds (ARI 0.48 on day 21); they remain
                  a display sub-group;
              singletons stay ungrouped (grp:∅) - after the join rule below
              was applied to them in the same run (groups_views round: the
              mutual-kNN graph cannot link a member slightly off a large
              tight group, whose members fill each other's 10 nearest; pack
              O's A9 sales IP was ungrouped on every run, ARI 0.97 -> 0.86).
              Deviation (measured, tests/eval/who_convergence.py): the text's
              "final Louvain level" suffers modularity's resolution limit on an
              organisation with a large public population (a 3-IP department
              joins any neighbour); (a) and (b) replace aggregation.
  stable ids  Hungarian on 1 - Jaccard(members), inherit at J >= 0.3; an IP
              moves only after 2 consecutive runs put it outside its group
              (a community that forms anew gets a new id each run, so the runs
              "elsewhere" are counted); events group_formed / group_changed (INFO,
              at (__org__, __org__)).
  names       config who_group_names [{name, ips | cidrs}] and dhcp_scopes
              [{cidr, name}] -> Hungarian on Jaccard >= 0.5 over the groups'
              ADDRESSES (prefix-mode pool sources are not people of the
              department); a group whose addresses lie >= 80 % inside a configured
              name that went to another group is one of its ROLES, named
              '<name>·<top action label>' (rec 'dept' = the configured name; P14
              composes a department view from its roles); else the auto name
              'G<id>·<top-2 action labels>'.
  labels      top-5 actions by lift x sqrt(support), lift >= 2.
  actions     per system the group's action mix (items >= 2 % of the group's
              mass there, <= 8, with the members that perform an action only
              some of them do): the user view's "what the group does where"
              (P14 group view).
  covers      smallest CIDR set covering >= 90 % of members with purity >= 0.8
              (lib/plouvain.prefix_covers) — the "某几个 IP 段" rendering and the
              `reg` level of the IP hierarchy when no configured region matches;
              plus the group's prefix-mode pool sources whose recurring addresses
              are >= 80 % the group's own (_pool_pure).
  members     the group's ADDRESSES; its prefix-mode pool sources (one-shot
              addresses pooled per /24) are published apart as 'pools' (pack O:
              财务部's 192.168.2.0/24 pool was listed as a member of 综合部's group
              and rendered in the login statement's who).
Who mode per system (the IP-agnostic decision, published in `mode`):
  model.sysprof chosen['who'] (P12) when present; else P05's attrsel who_mode
  'none' (IP carries no information about the targets); else the level with
  the shortest prequential two-part who code over the txn tree's leaves
  (§6.18.2; P04 accumulates it per level), 'none' when even the best level
  costs >= the 32 bits of an unmodelled IPv4 address. Views (P14) and P03 read
  it; signatures change their key only for P12's explicit strategy (prefix ->
  /24 sources, reg / none -> no signature): measured on pack O, keying OA by
  /24 because its DHCP-pool users make /24 the cheapest who code dissolved
  综合部 and 财务部 into their subnets' mixtures (ARI 0.97 -> 0.46).

Writes  model.who_groups@(__org__, __org__):
          {'fmt': 1, 'version', 'updated', 'last_run',
           'groups': {gid: {'id', 'name', 'name_source', 'auto_name', 'dept', 'members',
                            'n', 'pools', 'sub', 'covers', 'labels', 'systems', 'actions',
                            'first_seen', 'changed', 'provisional', 'cohesion'}},
           'ip2g': {ip: gid}, 'covers': {gid: [cidr]}, 'shared': [ip],
           'mode': {system: {'mode', 'source', 'bits'}}, 'stats'}
        model.who_groups_state@(__org__, __org__) (private: signatures, cursors,
        representatives, move bookkeeping).
        events group_formed, group_changed.
Budget  per learned row ~ 10 us; daily O(n k) MinHash + O(n b) LSH + Louvain on
        <= 10 n edges; memory S_max x ~0.5 KB + G_max x 32 representatives.
Inert unless config['progressive']['enabled'].
"""
from __future__ import annotations

import ipaddress
import math
import time
from typing import Callable, Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import ORG, SYSTEM_ENTITY, BehaviorEvent, Severity
from .lib import m_governor as MG
from .lib import m_ptree as MP
from .lib import pdfg as DF
from .lib import pevent as EV
from .lib import plouvain as LV
from .lib import pminhash as MH
from .lib import psketch as PS
from .lib import pwindows as PW

def _route_key(get: Callable[[str], Any]) -> Optional[str]:
    """lib/pdfg.route_key with absent attributes read as None (pdfg tests
    `isinstance(v, str)`, and ABSENT is the string '⊥')."""
    return DF.route_key(lambda a: (lambda v: None if v is EV.ABSENT else v)(get(a)))


STATE = "model.who_groups_state"
DAY = 86400.0
CLUSTER_PERIOD_S = DAY
EV_CLUSTER = 3.0            # evidence units a signature needs to enter the daily clustering
EV_JOIN = 10.0              # ... to join a group between runs (§6.15 item 3)
JOIN_MAJ = 0.6
JOIN_PER_TICK = 256
DROP_EV = 10.0              # dropped after 7 d with fewer units (§6.15 item 1)
DROP_AGE_S = 7 * DAY
IDLE_S = 30 * DAY
RHO = 0.9                   # merge / join when J(across) >= RHO x J(within)
MOVE_RUNS = 2
EPHEMERAL_SHARE = 1.0 / 3.0   # an address active on <= 1/3 of the days since it was first seen ...
EPHEMERAL_AGE_S = 2 * DAY       # ... (and seen >= 2 days ago) is an ephemeral (re-addressed) address
J_INHERIT = 0.3
J_NAME = 0.5
LIFT_MIN = 2.0
N_LABELS = 5
REPS = 32                   # representative MinHash samples kept per group
G_MAX = 256
CHANGED_J = 0.9
SHARED_MAX = 4096
WHO_LEVEL_MODE = {0: "ip", 1: "prefix", 2: "prefix", 3: "grp", 4: "reg"}
SIG_MODES = ("ip", "grp", "prefix")
MODE_PERIOD_S = 3600.0


def _ip_key(s: str) -> Tuple[int, int, str]:
    """Deterministic order of sources: IPs as integers (v4 before v6), then prefixes."""
    try:
        a = ipaddress.ip_address(s)
        return (a.version, int(a), s)
    except ValueError:
        try:
            n = ipaddress.ip_network(s, strict=False)
            return (n.version + 10, int(n.network_address), s)
        except ValueError:
            return (99, 0, s)


RECUR_DAYS = 2              # prefix mode: an address seen on >= 2 local days keeps its own signature


def _source_prefix_mode(st: "WGState", ip: str, ts: float, off: float) -> Optional[str]:
    """Signature source of an address on a system whose chosen who level is a
    prefix (P12). One-shot addresses (a public visitor, a DHCP lease) have too
    little evidence of their own, so they are pooled per /24; an address that
    recurs on >= RECUR_DAYS local days is a stable client and keeps its own
    signature. Measured on pack O: P12 chose 'prefix' for OA (its 60-person
    DHCP pool makes /24 the cheapest who code), and keying every OA source by
    /24 put 192.168.1.0/24, .2.0/24 and .3.0/24 into 综合部's and 财务部's
    groups as members and split 综合部 in two (ARI 0.60 on day 14)."""
    day = int((ts + off) // DAY)
    if getattr(st, "addr_days", None) is None:          # state saved before the field existed
        st.addr_days = PS.LRU(65536)
    rec = st.addr_days.get(ip)
    if rec is None:
        rec = (day, 1)
    elif rec[0] != day:
        rec = (day, rec[1] + 1)
    st.addr_days.put(ip, rec)
    return ip if rec[1] >= RECUR_DAYS else _p24(ip)


def _p24(ip: str) -> Optional[str]:
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return None
    n = ipaddress.ip_network(f"{ip}/{24 if a.version == 4 else 64}", strict=False)
    return str(n)


def _route_part(act: Any) -> Optional[str]:
    if not isinstance(act, str) or not act:
        return None
    return DF.split_key(act)[0]


def item_label(item: str) -> str:
    """'oa|POST oa.corp.local /login' -> 'oa POST /login' (host dropped)."""
    key, _, act = item.partition("|")
    parts = act.split()
    if len(parts) >= 3 and parts[0].isupper():
        return f"{key} {parts[0]} {parts[-1]}"
    return f"{key} {act}"


class WGState:
    """P11's private state (not published)."""

    def __init__(self, s_max: int = MH.S_MAX) -> None:
        self.sigs = MH.SigStore(s_max)
        self.burst = PS.BurstEvidence(min(65536, 4 * s_max))
        self.last_batch: Dict[str, float] = {}
        self.last_asg: Dict[str, float] = {}
        self.prev_members: Dict[str, Set[str]] = {}
        self.pending_move: Dict[str, Tuple[Optional[str], int]] = {}
        self.reps: Dict[str, np.ndarray] = {}          # gid -> uint64[REPS, K_MH]
        self.next_gid = 1
        self.runs = 0
        self.shared: PS.LRU = PS.LRU(SHARED_MAX)
        self.shared_since = -math.inf
        self.join_queue: Set[str] = set()
        self.join_tried: PS.LRU = PS.LRU(s_max)        # src -> last join attempt (fair order)
        self.ms: Dict[str, float] = {}
        self.modes: Dict[str, Dict[str, Any]] = {}
        self.cohesion: Dict[str, float] = {}
        # prefix mode: address -> (last local day, distinct days); recurrent
        # addresses keep their own signature (see _source)
        self.addr_days: PS.LRU = PS.LRU(min(65536, 4 * s_max))

    def nbytes(self) -> int:
        return int(self.sigs.nbytes() + sum(r.nbytes for r in self.reps.values())
                   + 64 * sum(len(v) for v in self.prev_members.values())
                   + 100 * len(getattr(self, "join_tried", ()) or ()) + 1024)


def empty_model() -> Dict[str, Any]:
    return {"fmt": 1, "version": 0, "updated": None, "last_run": None, "groups": {}, "ip2g": {},
            "covers": {}, "shared": [], "mode": {}, "stats": {}}


class WhoGroupsEngine(Engine):
    name = "behavior.who_groups"
    layer = "behavior"
    consumes = [EV.EVT_BATCH, EV.PAT_ASSIGN, MP.SYSPROF, MP.ATTRSEL, MP.PTREE, MP.BUDGET,
                "event.shared_ip", "config.who_group_names", "config.dhcp_scopes"]
    produces = [MP.WHO_GROUPS, STATE, "event.group_formed", "event.group_changed"]
    description = "P11: behavioural groups (IP x action co-occurrence), prefix covers, names, who mode"
    interval = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.cluster_period_s = float(params.get("cluster_period_s", CLUSTER_PERIOD_S))
        self.s_max = int(params.get("s_max", MH.S_MAX))
        self.ev_cluster = float(params.get("ev_cluster", EV_CLUSTER))
        self.rho = float(params.get("rho", RHO))
        self.last_stats: Dict[str, Any] = {}

    # ================================================================== run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        if not EV.enabled(ctx.config):
            return 0
        store = ctx.store
        now = float(ctx.now)
        D = EV.learn_delay_s(float(ctx.window_s or 60.0), ctx.config)
        st = store.get_model(ORG, ORG, STATE)
        if not isinstance(st, WGState):
            st = WGState(self.s_max)
        bud = MP.get_org_model(store, MP.BUDGET)
        cap = (bud or {}).get("s_max") if isinstance(bud, Mapping) else None
        if cap:
            st.sigs.set_cap(min(self.s_max, int(cap)))
        model = MP.get_org_model(store, MP.WHO_GROUPS)
        model = dict(model) if isinstance(model, Mapping) else empty_model()
        t0 = time.perf_counter()
        self._shared(st, store, now)
        modes = self._modes(st, store, now)
        n = self._signatures(st, store, now, D, modes, model, PW.tz_offset(ctx.config, now))
        changed = self._readdress(st, store, now, model)
        changed |= self._joins(st, now, model)
        ran = False
        if len(st.sigs) and self.entity_due(("p11", ORG), now, self.cluster_period_s):
            self._cluster(ctx, st, model, now)
            ran = True
            changed = True
        model["shared"] = sorted(st.shared._d.keys())
        model["mode"] = modes
        model["updated"] = now
        st.ms["tick"] = st.ms.get("tick", 0.0) + (time.perf_counter() - t0) * 1000.0
        model["stats"] = dict(model.get("stats") or {}, signatures=len(st.sigs),
                              items=len(st.sigs.items), dropped=st.sigs.dropped,
                              state_bytes=st.nbytes(), runs=st.runs,
                              ms_tick_total=round(st.ms.get("tick", 0.0), 1))
        if changed or ran:
            model["version"] = int(model.get("version") or 0) + 1
        store.put_model(ORG, ORG, STATE, st, version=st.runs + 1, ts=now)
        store.put_model(ORG, ORG, MP.WHO_GROUPS, model, version=int(model.get("version") or 0), ts=now)
        self.last_stats = {"rows": n, "clustered": ran, "signatures": len(st.sigs)}
        return n

    # ------------------------------------------------------------ shared
    def _shared(self, st: WGState, store: Any, now: float) -> None:
        """B17 shared_ip findings (NAT / VDI / terminal servers): several users
        behind one address, so the address gets no signature (§5.4.1)."""
        for e in store.events(since=st.shared_since if st.shared_since > -math.inf else None,
                              kinds=("shared_ip",), limit=1000):
            if e.entity and e.entity != SYSTEM_ENTITY:
                st.shared.put(str(e.entity), float(e.ts))
                st.sigs.drop(str(e.entity))
        st.shared_since = now

    # ------------------------------------------------------------- modes
    def _modes(self, st: WGState, store: Any, now: float) -> Dict[str, Dict[str, Any]]:
        """Who mode per system, refreshed hourly (reads O(nodes) of the tree)."""
        out: Dict[str, Dict[str, Any]] = {}
        for s in sorted(store.batch_systems(EV.EVT_BATCH)):
            cur = st.modes.get(s)
            if cur is None or now - float(cur.get("t", -math.inf)) >= MODE_PERIOD_S:
                cur = dict(decide_mode(store, MP.tree_key(store, s)), t=now)
                st.modes[s] = cur
            out[s] = {k: v for k, v in cur.items() if k != "t"}
        return out

    # -------------------------------------------------------- signatures
    def _signatures(self, st: WGState, store: Any, now: float, D: float,
                    modes: Mapping[str, Mapping[str, Any]], model: Mapping[str, Any],
                    off: float) -> int:
        n = 0
        tcache: Dict[Tuple[str, str, float], Tuple[float, bool]] = {}
        for s in sorted(store.batch_systems(EV.EVT_BATCH)):
            m = modes.get(s) or {}
            # signatures follow an explicit strategy (P12's chosen who); the
            # self-decided mode is advisory (views, P03): a system whose /24 level
            # codes its users more cheaply because of a DHCP pool still has
            # departments whose per-IP signatures are what groups them (the pool is
            # joined by the address plan, lib/plouvain.merge_local)
            mode = m.get("mode", "ip") if m.get("source") == "sysprof" else "ip"
            key = MP.tree_key(store, s)
            last = st.last_batch.get(s)
            for ts_b, b in MP.learnable_batches(store, s, EV.EVT_BATCH, last, now, D):
                st.last_batch[s] = ts_b
                if mode not in SIG_MODES or getattr(b, "kind", EV.KIND_TXN) != EV.KIND_TXN or b.n == 0:
                    continue
                asg = store.batch_at(s, EV.PAT_ASSIGN, ts_b)
                if asg is not None and asg.n != b.n:
                    asg = None
                act = asg.dense("act", None) if asg is not None and asg.has("act") else None
                cnode = asg.dense("conf", np.nan) if asg is not None and asg.has("conf") else None
                root = _root_id(store, key)
                tree = _tree(store, key)
                pcache: Dict[int, int] = {}
                damp = asg.dense("damp", 1.0) if asg is not None and asg.has("damp") else None
                mass = b.mass()
                rows = b.learned_rows()
                for i in rows.tolist():
                    ip = b.ip_of(i)
                    if st.shared.peek(ip) is not None:
                        continue
                    r = _route_part(act[i]) if act is not None else None
                    if r is None:
                        r = _route_key(lambda a, i=i: b.get(a, i))
                    if r is None:
                        continue
                    tr, q = _trust(store, s, ip, ts_b, tcache)
                    if q or tr <= 0.0:
                        continue
                    dm = 1.0
                    if damp is not None:
                        x = damp[i]
                        dm = float(x) if x is not None and x is not EV.ABSENT and x == x else 1.0
                    f = tr * dm
                    src = ip if mode in ("ip", "grp") else _source_prefix_mode(st, ip, ts_b, off)
                    if src is None:
                        continue
                    item = f"{key}|{r}"
                    ts = float(b.ts[i])
                    ev = st.burst.unit((src, item), ts, f)
                    sg = st.sigs.add(src, item, ts, float(mass[i]) * f, ev, int((ts + off) // DAY))
                    if cnode is not None:
                        c = cnode[i]
                        pn = _pattern_node(tree, int(c), pcache) \
                            if (c is not None and c is not EV.ABSENT and c == c) else root
                        if pn != root:
                            # IP x pattern co-occurrence: the covering confident pattern
                            # (its context part, see _pattern_node)
                            st.sigs.add(src, f"{key}|@{pn}", ts, float(mass[i]) * f, 0.0)
                    if sg.ev >= EV_JOIN and "/" not in src and src not in (model.get("ip2g") or {}):
                        st.join_queue.add(src)
                    n += 1
        return n

    # ---------------------------------------------------------- readdress
    def _readdress(self, st: WGState, store: Any, now: float, model: Dict[str, Any]) -> bool:
        """P03's readdress_candidate rows: the new IP joins the old IP's group
        at once, flagged provisional (the next daily run confirms or releases)."""
        ip2g = model.setdefault("ip2g", {})
        groups = model.setdefault("groups", {})
        changed = False
        for s in sorted(store.batch_systems(EV.PAT_ASSIGN)):
            last = st.last_asg.get(s, -math.inf)
            for ts_b, asg in store.batches_since(s, EV.PAT_ASSIGN, last):
                st.last_asg[s] = ts_b
                if not asg.has("readdr"):
                    continue
                col = asg.cols["readdr"]
                b = store.batch_at(s, EV.EVT_BATCH, ts_b)
                for r, old in zip(col.rows.tolist(), col.vals.tolist()):
                    if not isinstance(old, str) or b is None or r >= b.n:
                        continue
                    new = b.ip_of(int(r))
                    g = ip2g.get(old)
                    if g is None or new in ip2g or g not in groups:
                        continue
                    ip2g[new] = g
                    gr = groups[g]
                    gr["members"] = sorted(set(gr.get("members") or []) | {new}, key=_ip_key)
                    gr["n"] = len(gr["members"])
                    gr["provisional"] = sorted(set(gr.get("provisional") or []) | {new}, key=_ip_key)
                    changed = True
        return changed

    # --------------------------------------------------------------- joins
    def _joins(self, st: WGState, now: float, model: Dict[str, Any]) -> bool:
        if not st.join_queue or not st.reps:
            st.join_queue.clear()
            return False
        ip2g = model.setdefault("ip2g", {})
        groups = model.setdefault("groups", {})
        rnd = MH.Randoms()
        gids = sorted(st.reps)
        R = np.vstack([st.reps[g] for g in gids])
        owner = np.concatenate([[i] * st.reps[g].shape[0] for i, g in enumerate(gids)])
        changed = False
        # fair order: never-tried sources first, then the longest-waiting ones (an
        # address-ordered cut starved 192.168.x behind thousands of public 10.x
        # visitors: a SALES member isolated by one clustering run never rejoined)
        tried = getattr(st, "join_tried", None)
        if tried is None:
            tried = st.join_tried = PS.LRU(MH.S_MAX)
        queue = sorted((src for src in st.join_queue if src not in ip2g),
                       key=lambda x: (tried.peek(x) or -math.inf, _ip_key(x)))
        for src in queue[:JOIN_PER_TICK]:
            tried.put(src, float(now))
            mh = self._minhash(st, src, now, rnd)
            if mh is None:
                continue
            J = (R == mh[None, :]).mean(axis=1)
            top = np.argsort(-J, kind="stable")[:MH.KNN]
            top = [int(i) for i in top if J[i] >= MH.J_MIN]
            if not top:
                continue
            votes: Dict[int, float] = {}
            for i in top:
                votes[int(owner[i])] = votes.get(int(owner[i]), 0.0) + float(J[i])
            g_i, v = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))
            g = gids[g_i]
            jg = float(J[owner == g_i].mean())
            # label propagation with the merge rule: the IP must be as similar to
            # the group as its members are to each other
            if v / sum(votes.values()) >= JOIN_MAJ and jg >= self.rho * getattr(st, "cohesion", {}).get(g, 1.0) \
                    and g in groups:
                ip2g[src] = g
                gr = groups[g]
                gr["members"] = sorted(set(gr.get("members") or []) | {src}, key=_ip_key)
                gr["n"] = len(gr["members"])
                gr.setdefault("joined", []).append(src)
                changed = True
        st.join_queue.clear()
        return changed

    @staticmethod
    def _minhash(st: WGState, src: str, t: float, rnd: MH.Randoms) -> Optional[np.ndarray]:
        ids, w = st.sigs.weights(src, t)
        if ids.size == 0:
            return None
        return MH.icws([st.sigs.items.hash_of(int(i)) for i in ids], MH.transform(w), rnd)

    # ============================================================ cluster
    def _cluster(self, ctx: Context, st: WGState, model: Dict[str, Any], now: float) -> None:
        t0 = time.perf_counter()
        store = ctx.store
        sigs = st.sigs
        sigs.prune(now, DROP_EV, DROP_AGE_S, IDLE_S)
        srcs = sorted((k for k in sigs.keys() if sigs.get(k).ev >= self.ev_cluster), key=_ip_key)
        rnd = MH.Randoms()
        mats, prof, keep = [], [], []
        for src in srcs:
            ids, w = sigs.weights(src, now)
            tw = MH.transform(w)
            if not (tw > 0).any():
                continue
            mats.append(MH.icws([sigs.items.hash_of(int(i)) for i in ids], tw, rnd))
            prof.append({int(i): float(x) for i, x in zip(ids, tw) if x > 0})
            keep.append(src)
        srcs = keep
        n = len(srcs)
        M = np.vstack(mats) if mats else np.zeros((0, MH.K_MH), dtype=np.uint64)
        # identical signatures (a pool whose members behave alike) are one node with
        # a self-loop for their multiplicity: ties among exact duplicates would
        # otherwise make the mutual-kNN graph arbitrary
        if n:
            U, inv, cnt = np.unique(M, axis=0, return_inverse=True, return_counts=True)
            inv = np.asarray(inv).reshape(-1)
        else:
            U, inv, cnt = M, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        pairs = MH.lsh_pairs(U)
        edges = MH.knn_graph(U, pairs, mult=cnt)
        for u, c in enumerate(cnt.tolist()):
            if c > 1:
                edges[(u, u)] = c * (c - 1) / 2.0
        # Louvain per connected component (modularity's null model then uses the
        # component's own weight, not the whole organisation's: with a large
        # public population in the graph a 3-IP department otherwise merges with
        # any neighbour, the resolution limit), and its FIRST level (local moves)
        # as the base; communities are joined afterwards only by the within /
        # across rule and the address plan
        nU = U.shape[0]
        lab_u = np.arange(nU)
        sub0_u = np.arange(nU)
        n_levels = 0
        for comp in MH.components(nU, (e for e in edges if e[0] != e[1])):
            if len(comp) < 2:
                continue
            pos = {u: k for k, u in enumerate(comp)}
            ce = {(pos[a], pos[b]): w for (a, b), w in edges.items() if a in pos and b in pos}
            lv = LV.louvain(len(comp), ce)
            n_levels = max(n_levels, len(lv))
            base = lv[0]
            for k, u in enumerate(comp):
                lab_u[u] = comp[_first(base, int(base[k]))]
                sub0_u[u] = lab_u[u]
        levels_u = [sub0_u]
        # communities as similar across as within are one group (lib/plouvain)
        adj = {(int(lab_u[a]), int(lab_u[b])) for (a, b) in edges if lab_u[a] != lab_u[b]}
        lab_u, intra = LV.merge_similar(lab_u, U, self.rho, adj, mult=cnt)
        lab = lab_u[inv] if n else np.zeros(0, dtype=np.int64)
        levels = [lv[inv] for lv in levels_u] if n else []
        # re-addressed pools (DHCP / VPN): ephemeral addresses in one pure prefix
        active = LV.ActiveIndex(k for k in sigs.keys() if "/" not in str(k))
        if n:
            eph = [_ephemeral(sigs.get(src), now) and "/" not in src for src in srcs]
            lab = LV.merge_local(lab, M, srcs, eph, active)
            # a department split by one optional activity, inside its own subnet
            lab = LV.merge_prefix(lab, M, srcs, active, self.rho)
        comm: Dict[int, List[str]] = {}
        for i, c in enumerate(lab.tolist()):
            comm.setdefault(c, []).append(srcs[i])
        new_groups = [set(v) for c, v in sorted(comm.items()) if len(v) >= 2]
        sub0 = levels[0] if levels else np.arange(n)
        # ---- stable ids with the two-run move rule
        prev = {g: set(m) for g, m in st.prev_members.items()}
        inherit = LV.match_ids(prev, new_groups, J_INHERIT)
        old_of: Dict[str, str] = {}
        for g, m in prev.items():
            for ip in m:
                old_of[ip] = g
        final: Dict[str, Set[str]] = {}
        for i, members in enumerate(new_groups):
            gid = inherit.get(i)
            if gid is None:
                gid = f"G{st.next_gid}"
                st.next_gid += 1
            final[gid] = set(members)
        where = {ip: g for g, m in final.items() for ip in m}
        pend: Dict[str, Tuple[Optional[str], int]] = {}
        for ip, g_old in old_of.items():
            g_new = where.get(ip)
            if g_new == g_old:
                continue
            if ip not in sigs:                           # dropped / evicted: leaves at once
                continue
            # consecutive runs that put the IP somewhere other than its group (a
            # community that forms anew gets a new id every run, so "elsewhere"
            # is counted, not "the same elsewhere")
            p = st.pending_move.get(ip)
            cnt = (p[1] + 1) if p is not None else 1
            if cnt >= MOVE_RUNS:
                continue                                  # the move is accepted
            pend[ip] = (g_new, cnt)
            if g_new is not None:                         # keep it where it was for now
                final[g_new].discard(ip)
            if g_old in final:
                final[g_old].add(ip)
            elif g_old in prev and len(prev[g_old]) >= 2:
                final.setdefault(g_old, set()).add(ip)
        st.pending_move = pend
        final = {g: m for g, m in final.items() if len(m) >= 2}
        # ---- provisional members (readdress) are confirmed by the clustering or released
        prov_keep: Dict[str, Set[str]] = {}
        for g, gr in (model.get("groups") or {}).items():
            for ip in gr.get("provisional") or []:
                if ip in final.get(g, set()):
                    continue
                sg = sigs.get(ip)
                if sg is not None and sg.ev < self.ev_cluster and g in final:
                    prov_keep.setdefault(g, set()).add(ip)       # too little evidence to judge yet
        for g, ips in prov_keep.items():
            final[g] |= ips
        # ---- group records
        glob: Dict[int, float] = {}
        for p in prof:
            s = sum(p.values())
            for k, v in p.items():
                glob[k] = glob.get(k, 0.0) + v / s
        gsum = sum(glob.values()) or 1.0
        idx = {src: i for i, src in enumerate(srcs)}
        named = _named_sets(ctx.config, sigs.keys())
        # names are matched on the groups' ADDRESSES: prefix-mode pool sources
        # (one-shot visitors pooled per /24) are not people of the department -
        # pack O: 综合部's learned group {.23, .121} carried 10.168.7.0/24 and
        # 192.168.2.0/24 pool sources, Jaccard 2/5 < J_NAME, and stayed unnamed
        ipsets = {g: {m for m in final[g] if "/" not in m} for g in final}
        names = LV.match_names(ipsets, named, J_NAME) if named else {}
        subnames = _sub_names(ipsets, named, names)
        old_groups = model.get("groups") or {}
        groups: Dict[str, Dict[str, Any]] = {}
        ip2g: Dict[str, str] = {}
        covers: Dict[str, List[str]] = {}
        reps: Dict[str, np.ndarray] = {}
        events: List[Tuple[str, str, Dict[str, Any]]] = []
        order = sorted(final, key=lambda g: -len(final[g]))
        for rank, g in enumerate(order):
            members = sorted(final[g], key=_ip_key)
            mi = [idx[m] for m in members if m in idx]
            gp: Dict[int, float] = {}
            has: Dict[int, int] = {}
            for i in mi:
                p = prof[i]
                s = sum(p.values())
                for k, v in p.items():
                    gp[k] = gp.get(k, 0.0) + v / s
                    has[k] = has.get(k, 0) + 1
            tot = sum(gp.values()) or 1.0
            labels = []
            for k, v in gp.items():
                if (sigs.items.key_of(k) or "").partition("|")[2].startswith("@"):
                    continue                          # pattern items are not labels
                lift = (v / tot) / max(glob.get(k, 0.0) / gsum, 1e-12)
                sup = has[k] / max(1, len(mi))
                if lift >= LIFT_MIN:
                    key = sigs.items.key_of(k) or f"#{k}"
                    labels.append({"item": key, "label": item_label(key), "lift": round(lift, 3),
                                   "support": round(sup, 3), "score": lift * math.sqrt(sup)})
            labels.sort(key=lambda x: (-x["score"], x["item"]))
            labels = labels[:N_LABELS]
            systems: Dict[str, float] = {}
            for k, v in gp.items():
                key = sigs.items.key_of(k) or ""
                sk = key.partition("|")[0]
                systems[sk] = systems.get(sk, 0.0) + v / tot
            acts = _group_actions(gp, tot, mi, prof, srcs, sigs.items)
            ips = [m for m in members if "/" not in m]
            pools = [m for m in members if "/" in m]
            cv = LV.prefix_covers(ips, active)
            # a prefix-mode pool source covers the group only where the group's
            # addresses are the pool's recurring population (purity, as for
            # prefix_covers): 192.168.2.0/24 (财务部's subnet, whose members'
            # first-day OA rows were pooled) is not a 综合部 address range
            cv += [m for m in pools if _pool_pure(m, set(ips), active)]
            subs: Dict[int, List[str]] = {}
            for i in mi:
                subs.setdefault(int(sub0[i]), []).append(srcs[i])
            sub = [sorted(v, key=_ip_key) for _, v in sorted(subs.items())] if len(subs) > 1 else []
            top = labels[:2] or [{"label": item_label(sigs.items.key_of(k) or f"#{k}")}
                                 for k, _ in sorted(gp.items(), key=lambda kv: -kv[1])[:2]
                                 if not (sigs.items.key_of(k) or "").partition("|")[2].startswith("@")]
            auto = f"{g}·" + "+".join(x["label"] for x in top) if top else g
            nm = names.get(g) or subnames.get(g)
            old = old_groups.get(g) or {}
            prov = [ip for ip in old.get("provisional") or [] if ip in prov_keep.get(g, set())]
            if nm is not None and g in subnames and g not in names:
                # a learned role inside a configured department (another group got
                # the department's name): '综合部·<its top action>'
                nm = (f"{nm[0]}·" + "+".join(x["label"] for x in top[:1]) if top else nm[0], nm[1])
            rec = {"id": g, "name": nm[0] if nm else auto, "name_source": "config" if nm else "auto",
                   "name_jaccard": round(nm[1], 3) if nm else None, "auto_name": auto,
                   "dept": (names.get(g) or subnames.get(g) or (None,))[0],
                   "members": ips, "n": len(ips), "pools": pools, "sub": sub, "covers": cv,
                   "labels": labels, "systems": {k: round(v, 4) for k, v in sorted(systems.items())
                                                  if v >= 0.01},
                   "actions": acts,
                   "first_seen": old.get("first_seen", now), "changed": now, "provisional": prov,
                   "materialise": rank < G_MAX or bool(nm)}
            if not old:
                events.append(("group_formed", g, {"members": members[:32], "n": len(members),
                                                   "name": rec["name"]}))
                rec["changed"] = now
            else:
                jm = LV._jac(set(old.get("members") or []), set(members))
                if jm < CHANGED_J or old.get("name") != rec["name"]:
                    events.append(("group_changed", g, {"jaccard": round(jm, 3), "n": len(members),
                                                        "name": rec["name"],
                                                        "old_name": old.get("name")}))
                else:
                    rec["changed"] = old.get("changed", now)
            groups[g] = rec
            for m in members:
                if "/" not in m:
                    ip2g[m] = g
            if cv:
                covers[g] = cv
            if mi:
                sel = mi if len(mi) <= REPS else [mi[int(j)] for j in
                                                  np.linspace(0, len(mi) - 1, REPS).round().astype(int)]
                reps[g] = M[sel]
                cl = {int(lab[i]) for i in mi}
                rec["cohesion"] = round(float(np.mean([intra.get(c, 1.0) for c in cl])), 4)
        st.prev_members = {g: set(final[g]) for g in groups}
        st.reps = reps
        st.cohesion = {g: float(r.get("cohesion", 1.0)) for g, r in groups.items()}
        st.runs += 1
        model.update({"groups": groups, "ip2g": ip2g, "covers": covers, "last_run": now,
                      "stats": {"eligible": n, "pairs": len(pairs), "edges": len(edges),
                                "levels": n_levels, "groups": len(groups),
                                "grouped": len(ip2g), "pending_moves": len(pend),
                                "ms_cluster": round((time.perf_counter() - t0) * 1000.0, 1)}})
        # the join rule (one label-propagation step) for the eligible sources the
        # run left ungrouped. The mutual-kNN graph drops a member that is only
        # slightly off a large tight group: with 20 near-identical SALES
        # members every member's 10 nearest are other members, so an address
        # with one extra item (A9's sales IP, J 0.92 to its colleagues at
        # cohesion 0.95) was in nobody's top 10 and left ungrouped on every run
        # (pack O seeds 0-1, day 21, ARI 0.97 -> 0.86); between runs it joined
        # and the next run dropped it again
        st.join_queue = {src for src in srcs if "/" not in src and src not in ip2g
                         and sigs.get(src) is not None and sigs.get(src).ev >= EV_JOIN}
        if self._joins(st, now, model):
            for g, gr in model["groups"].items():
                st.prev_members[g] = set(gr.get("members") or []) | set(gr.get("pools") or [])
        for kind, g, extra in events:
            store.add_event(BehaviorEvent(
                system=ORG, entity=ORG, ts=now, kind=kind, score=0.0, severity=Severity.INFO,
                description=(f"行为群组 {groups[g]['name']} 形成（{groups[g]['n']} 个成员）" if kind == "group_formed"
                             else f"行为群组 {groups[g]['name']} 的成员发生变化（{groups[g]['n']} 个成员）"),
                extra=dict(extra, group=g), dedupe_key=f"{kind}|{g}|{int(now // DAY)}"))


# ====================================================================== helpers
def _trust(store: Any, s: str, ip: str, at: float,
           cache: Dict[Tuple[str, str, float], Tuple[float, bool]]) -> Tuple[float, bool]:
    k = (s, ip, at)
    r = cache.get(k)
    if r is None:
        q = MG.is_quarantined(store, s, ip, at)
        row = store.vec_at(s, ip, MG.TRUST, float(at))
        if row is None:
            tr = 1.0
        else:
            tr = float(np.asarray(row).reshape(-1)[0])
            tr = 0.0 if not tr == tr else min(1.0, max(0.0, tr))
        r = cache[k] = (tr, q)
    return r


ACT_SHARE = 0.02            # an action is part of a group's activity in a system at this share
ACT_MEMBER_W = 0.02         # ... and a member "does" it at this share of its own signature
ACTS_PER_SYSTEM = 8
ACT_MEMBERS_MAX = 16       # members listed per action (a subset of a large group is a share only)


def _group_actions(gp: Mapping[int, float], tot: float, mi: Sequence[int],
                   prof: Sequence[Mapping[int, float]], srcs: Sequence[str],
                   items: Any) -> Dict[str, List[Dict[str, Any]]]:
    """What the group does in each system (the user view's "综合部 访问 OA：登录、
    审批、生成报告"): per system key the action items of the group profile holding
    >= ACT_SHARE of the group's mass IN THAT SYSTEM, largest first, each with
    its share and - when only some members do it - the members that do (a
    member does an action when it holds >= ACT_MEMBER_W of its own normalised
    signature; 综合部's approvals are one member's, its reports two others').
    Read from the signatures the clustering already holds: O(group items)."""
    per_sys: Dict[str, List[Tuple[str, int, float]]] = {}
    for k, v in gp.items():
        key = items.key_of(k) or ""
        sk, _, act = key.partition("|")
        if not act or act.startswith("@"):
            continue
        per_sys.setdefault(sk, []).append((act, int(k), float(v)))
    out: Dict[str, List[Dict[str, Any]]] = {}
    for sk, lst in sorted(per_sys.items()):
        s_tot = sum(v for _, _, v in lst)
        if s_tot <= 0:
            continue
        rows = []
        for act, k, v in sorted(lst, key=lambda x: (-x[2], x[0])):
            share = v / s_tot
            if share < ACT_SHARE:
                continue
            # members are the group's ADDRESSES (prefix-mode pool sources are
            # not people: '提交报告[10.168.7.121、192.168.1.23、10.168.7.0/24]')
            who = []
            n_ip = 0
            for i in mi:
                if "/" in str(srcs[i]):
                    continue
                n_ip += 1
                p = prof[i]
                s = sum(p.values()) or 1.0
                if p.get(k, 0.0) / s >= ACT_MEMBER_W:
                    who.append(srcs[i])
            rows.append({"action": act, "label": item_label(f"{sk}|{act}"), "share": round(share, 4),
                         "support": round(len(who) / max(1, n_ip), 3),
                         "members": sorted(who, key=_ip_key) if len(who) < min(n_ip, ACT_MEMBERS_MAX + 1)
                         else []})
            if len(rows) >= ACTS_PER_SYSTEM:
                break
        if rows:
            out[sk] = rows
    return out


def _first(labels: np.ndarray, c: int) -> int:
    """Index of the first element of `labels` equal to c (a stable community id)."""
    return int(np.flatnonzero(labels == c)[0])


def _ephemeral(sg: Any, now: float) -> bool:
    age = (now - sg.first) / DAY
    return age >= EPHEMERAL_AGE_S / DAY and sg.days <= max(1.0, EPHEMERAL_SHARE * age)


# split attributes that partition WHO does an action (address, client stack) or
# WHEN (calendar / time of day); not which action (the action item carries it),
# not its payload, not per-session measures (think time, position in session)
# Pattern items come from TIME-context splits only. A split on the source
# address (net.src at any level) or on a client-stack proxy of it encodes WHERE a
# source is, not what it does: measured on pack O (integration 2026-09-30), as
# P04 split shared nodes by /24 (10.168.7.0/24 = one of 综合部's three members,
# 192.168.1.0/24 = the other two) those items pulled a department's members
# apart, and ARI against the departments fell from 1.0 on day 7 to 0.87 / 0.59
# on day 21 (seeds 0 / 1) - the opposite of "longer is more precise".
CONTEXT_PREFIX = ("ctx.when", "ctx.daytype", "ctx.dayclass", "ctx.dow",
                  "ctx.tod", "ctx.mend", "ctx.dom")


def _tree(store: Any, key: str) -> Any:
    ptm = MP.get_ptree(store, key)
    return ptm.kinds.get(EV.KIND_TXN) if ptm is not None else None


def _pattern_node(tree: Any, nid: int, cache: Dict[int, int]) -> int:
    """The context pattern of a covering node: the nearest node on its path
    reached by a who / when split (net.src, client.*, calendar / time of day),
    climbing out of exception nodes, CONTENT splits (net.bytes_up bins, a body
    field's values) and ACTION splits (http.method / path / route, SNI: the
    action item already carries those) and per-session measures (think
    time bins: a 3.33 of SALES ended on its own think-time node and stayed
    ungrouped). Returns the root id when no such
    split is on the path (no pattern item). Measured on pack O: payload-bin
    items split SALES (20 IPs) into three communities (ARI 1.0 -> 0.53, day
    9 -> 11); a GET/POST child counted every crm event twice and split it
    again by crm share (ARI 0.49 on day 21)."""
    r = cache.get(nid)
    if r is not None:
        return r
    out = nid
    if tree is not None:
        x = tree.nodes.get(nid)
        while x is not None and x.parent is not None:
            par = tree.nodes.get(x.parent)
            if par is None:
                break
            attr = str(getattr(par.split, "attr", "") or "") if par.split is not None else ""
            if not getattr(x, "is_exc", False) and attr.startswith(CONTEXT_PREFIX):
                break
            x = par
        out = int(x.id) if x is not None else nid
    cache[nid] = out
    return out


def _root_id(store: Any, key: str) -> int:
    ptm = MP.get_ptree(store, key)
    tree = ptm.kinds.get(EV.KIND_TXN) if ptm is not None else None
    return int(tree.root) if tree is not None else -1


SUB_NAME_IN = 0.8           # a group whose addresses lie >= 80 % inside a configured department


def _sub_names(ipsets: Mapping[str, Set[str]], named: Sequence[Tuple[str, Set[str]]],
               names: Mapping[str, Tuple[str, float]]) -> Dict[str, Tuple[str, float]]:
    """{group: (department name, share inside)} for learned groups that are not
    the Hungarian match of a configured name but whose addresses lie >= SUB_NAME_IN
    inside one: the department's other ROLES. Pack O: 综合部's approver
    (192.168.1.21) and its two report writers behave differently enough to be two
    learned groups (approvals vs reports, weighted Jaccard 0.47); the address
    list names both as 综合部, the learned split is its role structure. Names
    only - membership stays learned."""
    out: Dict[str, Tuple[str, float]] = {}
    for g, ips in ipsets.items():
        if g in names or not ips:
            continue
        best = None
        for nm, ss in named:
            share = len(ips & ss) / len(ips)
            if share >= SUB_NAME_IN and (best is None or share > best[1]):
                best = (nm, share)
        if best is not None:
            out[g] = best
    return out


def _pool_pure(pool: str, ips: Set[str], active: Any) -> bool:
    """A prefix-mode pool source belongs to a group's address range when the
    group's own addresses are >= PURITY of the recurring addresses inside it."""
    try:
        n = ipaddress.ip_network(pool, strict=False)
    except ValueError:
        return False
    lo = int(n.network_address)
    hi = lo + n.num_addresses - 1
    tot = active.count(n.version, lo, hi)
    own = sum(1 for ip in ips if _in_net(ip, n))
    return tot == 0 or own / tot >= LV.PURITY


def _in_net(ip: str, n: Any) -> bool:
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return a.version == n.version and a in n


def _named_sets(config: Mapping[str, Any], sources: Iterable[str]) -> List[Tuple[str, Set[str]]]:
    """Configured / imported group names as member sets over the tracked
    sources: who_group_names [{name, ips | cidrs}], dhcp_scopes [{cidr, name}]."""
    srcs = [s for s in sources if "/" not in str(s)]
    addr = {}
    for s in srcs:
        try:
            addr[s] = ipaddress.ip_address(s)
        except ValueError:
            continue
    out: List[Tuple[str, Set[str]]] = []

    def inside(cidrs: Iterable[str]) -> Set[str]:
        nets = []
        for c in cidrs:
            try:
                nets.append(ipaddress.ip_network(str(c), strict=False))
            except ValueError:
                continue
        return {s for s, a in addr.items() if any(a.version == n.version and a in n for n in nets)}
    for it in (config or {}).get("who_group_names") or []:
        if not isinstance(it, Mapping) or not it.get("name"):
            continue
        m = {str(x) for x in it.get("ips") or []} | inside(it.get("cidrs") or [])
        if m:
            out.append((str(it["name"]), m))
    have = {n for n, _ in out}
    for it in (config or {}).get("dhcp_scopes") or []:
        if not isinstance(it, Mapping) or not it.get("name") or not it.get("cidr"):
            continue
        if str(it["name"]) in have:
            continue
        m = inside([it["cidr"]])
        if m:
            out.append((str(it["name"]), m))
    return out


def who_code_bits(store: Any, key: str) -> Dict[str, float]:
    """Per-level prequential two-part who code (bits per learned event) summed
    over the leaves of a tree's txn tree (§6.18.2; P04 accumulates it on leaves)."""
    ptm = MP.get_ptree(store, key)
    tree = ptm.kinds.get(EV.KIND_TXN) if ptm is not None else None
    if tree is None:
        return {}
    code = None
    n = 0.0
    for nd in tree.nodes.values():
        if nd.split is not None or nd.is_exc or nd.who.code_n <= 0:
            continue
        c = np.asarray(nd.who.code, dtype=np.float64)
        code = c.copy() if code is None else code + c
        n += float(nd.who.code_n)
    if code is None or n <= 0:
        return {}
    names = ("ip", "p24", "p16", "grp", "reg")
    return {names[i]: float(code[i] / n) for i in range(min(len(names), code.size))
            if code[i] > 0}


def decide_mode(store: Any, key: str) -> Dict[str, Any]:
    """The who mode of a tree (§6.15.1 / §6.18.2): P12's chosen arm, else P05's
    'IP carries no information' verdict, else the shortest who code."""
    sp = MP.get_model(store, key, MP.SYSPROF)
    ch = (sp or {}).get("chosen") if isinstance(sp, Mapping) else None
    if isinstance(ch, Mapping) and ch.get("who"):
        return {"mode": str(ch["who"]), "source": "sysprof"}
    sel = MP.get_model(store, key, MP.ATTRSEL)
    wm = sel.get("who_mode") if isinstance(sel, Mapping) and not sel.get("bootstrap") else None
    bits = who_code_bits(store, key)
    if wm == "none":
        return {"mode": "none", "source": "attrsel", "bits": bits}
    if bits:
        best = min(bits, key=bits.get)
        if bits[best] >= 32.0:
            return {"mode": "none", "source": "code", "bits": bits}
        lv = {"ip": 0, "p24": 1, "p16": 2, "grp": 3, "reg": 4}[best]
        return {"mode": WHO_LEVEL_MODE[lv], "source": "code", "bits": bits}
    return {"mode": "ip", "source": "default"}
