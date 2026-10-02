"""P10 Workflow (`behavior.workflow`) — directly-follows graphs, workflows and
required predecessors per system and behavioural group
(docs/lib3/progressive.md §6.14, card P10). Library 3 (behaviour).

Requirement S10 ("后续192.168.1.21访问业务审批页面 … 下午5点提交报告数据到生成报告页面"):
the ORDER in which a class of IPs (or one group) performs actions is learned
from sessions, not configured: "登录 → 首页", "审批列表 → 审批详情 → 批准（间隔 30 s–5 min）",
"报告表单 → 生成报告（先有表单，后有提交）". Edges, workflows and required
predecessors all carry evidence on the confidence channel, so they appear
only once enough independent sessions support them and become more certain
the longer the behaviour is observed; counts are H_m / H_l-decayed and mined
again every 6 h, so they follow the behaviour when it changes.

Per tick (all of it O(events of the tick); nothing iterates over IPs):
  sessions  every event of the tick's evt.batch (not only the learning sample),
            in time order, updates its session (ip, sess.key) in a bounded LRU:
            previous action, delay, earlier actions, Bloom set of actions seen.
            A LEARNED row leaves a pending annotation (b, a, delay, start, earlier).
            A session boundary is the configured gap (defaults.session_gap_s,
            30 min) by default; session_mode='ctx' follows P01's ctx.sid.
            Deviation from §6.14, measured (tests/eval/temporal_convergence.py,
            pack O OA, seed 0): P01's Otsu valley of the log gap histogram drops
            to its 60-s clamp on OA from day 3, below the 30-300 s think times of
            the approval and report steps, so ctx sessions cut real workflows:
            truth-edge recall 0.4 / 0.4 / 0.8 at days 2 / 5 / 10 with ctx.sid
            against 0.4 / 0.8 / 1.0 with the 30-min gap (program precision 1.0
            and 0.89 at day 10). Switch the default once P01's gap is fixed.
  counting  annotations of tick t' <= t - D are counted with trust (B28) and
            outlier damping (P03's pat.assign `damp` when present), so an
            attacker's session is never learned before it could be judged
            (RC16); a quarantined IP's rows add nothing. Scopes: '*' and every P11
            group (model.who_groups ip2g) with >= 2 IPs active here (<= 32).
  mining    every 6 h per tree (entity_due), when new evidence arrived: kept
            edges, workflows (with P09 time anchors of each action), required
            predecessors (lib/pdfg.mine_scope). An edge that stopped is stale
            after missed NORMAL days of its own day type (P01 model.pcal; 2
            workdays for a daily edge, ~3 weeks for a weekly one; weekends and
            holidays never count), so a renamed step (D3) leaves the view
            within 2 workdays instead of the 7 calendar days of the time rule.

  renames   (round 3, lib/pdfg.detect_renames / adopt_renames) the session pass
            also feeds a route ledger (every event, trusted or not: first / last
            day, dates, <= 4 sources, <= 4 previous routes); once a local day, a new
            route that replaced an established one of the same sources (the old
            one not used since, one literal path segment changed, the same
            workflow position, >= 2 dates) takes the old action's id: its edges,
            workflows and requirements carry over under the new name, P03 no
            longer scores it as new, and model.pflow['renamed'] lists it (D3:
            /approval/... -> /flow/... of 192.168.1.21 was flagged new every day
            and its source held, so the learned counts never saw the new pages).
Reads   evt.batch (+ evt.ctx for ctx.sid), pat.assign (damp), model.ptree
        (action variants: the node reached through the deepest content split;
        act_node for rendering and P09 anchors), model.who_groups (ip2g),
        model.pwin (anchors), model.pcal (P01 normal days, edge staleness), model.sysprof (arm 'p10' / 'workflow' off -> skip),
        model.budget (session cap), behavior.trust / behavior.quarantine (B28).
Writes  model.pflow@(tree key, '__system__') = lib/pdfg.PFlowModel:
          {'fmt': 1, 'version', 'updated', 'last_mine',
           'scopes': {g: {'edges': [{'a', 'b', 'from', 'to', 'count', 'rev', 'dep',
                                     'share', 'band': [q10_s, q90_s], 'confidence'}],
                          'workflows': [{'path': [ids], 'routes': [...], 'support',
                                         'bands', 'anchors': [when | None], 'confidence'}],
                          'requires': [{'b', 'a', 'to', 'from', 'c_b', 'c_with', 'lb'}],
                          'sessions'}},
           'acts': {id: {'key', 'route', 'variant', 'act_node', 'n'}},
           'gain': {'bits_per_event', 'rows'}, 'stats': {...}}
        and the counting state in its attribute `.state` (lib/pdfg.FlowState),
        which P03 scores against through lib/pdfg.seq_scores (p_trans, p_req,
        p_seq). The edge dicts' 'from'/'to' are route strings, i.e. directly
        the statement contract's `evidence.workflow` entries (eval/pmetrics).
Budget  ~10 us per event (session pass) + ~40 us per learned row (counting);
        mining O(tracked edges + prec keys) per tree every 6 h. Memory: capped
        sketches and LRUs only (lib/pdfg docstring).
Inert unless config['progressive']['enabled'].
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import SYSTEM_ENTITY
from .lib import m_governor as MG
from .lib import m_ptree as MP
from .lib import pdfg as DF
from .lib import pevent as EV
from .lib import psketch as PS
from .lib import pwindows as PW

MINE_PERIOD_S = 6 * 3600.0
PCAL = "model.pcal"              # P01's per-system calendar (day classes, normal-day flags)
SESSION_MODE = "gap"             # 'gap': configured gap; 'ctx': P01's ctx.sid (see "sessions" above)
TOP_REFRESH_S = 3600.0
GAIN_ROWS = 64                   # prequential-gain evaluations per tick and tree (systematic subsample)
NON_CONTENT_KINDS = frozenset({"ip", "dst", "route", "path", "tod", "when"})
NON_CONTENT_PREFIX = ("ctx.", "ev.", "sess.", "net.src", "net.peer", "net.dst", "client.",
                      "http.method", "http.host", "http.route", "http.path", "tls.sni", "dns.qname")


def _arm_off(store: Any, key: str) -> bool:
    sp = store.get_model(key, SYSTEM_ENTITY, MP.SYSPROF)
    ch = (sp or {}).get("chosen") if isinstance(sp, Mapping) else None
    if isinstance(ch, Mapping):
        for n in ("P10", "p10", "workflow"):
            if str(ch.get(n, "on")).lower() == "off":
                return True
    return False


def _is_content(attr: str, hier: Any) -> bool:
    if attr.startswith(NON_CONTENT_PREFIX) or attr.startswith("@"):
        return False
    try:
        return hier.kind(attr) not in NON_CONTENT_KINDS
    except Exception:
        return True


class WorkflowEngine(Engine):
    name = "behavior.workflow"
    layer = "behavior"
    consumes = [EV.EVT_BATCH, EV.EVT_CTX, EV.PAT_ASSIGN, MP.PTREE, MP.WHO_GROUPS, MP.PWIN,
                MP.SYSPROF, MP.BUDGET, PCAL]
    produces = [MP.PFLOW]
    description = "P10: directly-follows graphs, workflows and required predecessors per system and group"
    interval = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.mine_period_s = float(params.get("mine_period_s", MINE_PERIOD_S))
        self.session_mode = str(params.get("session_mode", SESSION_MODE))
        self.last_stats: Dict[str, Any] = {}

    # ----------------------------------------------------------------- run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        if not EV.enabled(ctx.config):
            return 0
        store = ctx.store
        now = float(ctx.now)
        pc = EV.pconfig(ctx.config)
        D = EV.learn_delay_s(float(ctx.window_s or 60.0), ctx.config)
        gap = float(pc["defaults"].get("session_gap_s", 1800.0))
        by_key: Dict[str, List[str]] = {}
        for s in sorted(store.batch_systems(EV.EVT_BATCH)):
            by_key.setdefault(MP.tree_key(store, s), []).append(s)
        wg = MP.who_groups(store)
        ip2g = wg.get("ip2g") or {}
        shared = set(wg.get("shared") or ())
        n = 0
        stats: Dict[str, Any] = {}
        for key, systems in by_key.items():
            if _arm_off(store, key):
                stats[key] = {"off": True}
                continue
            t0 = time.perf_counter()
            model = MP.get_model(store, key, MP.PFLOW)
            if not isinstance(model, DF.PFlowModel):
                model = DF.PFlowModel()
                model.state = DF.FlowState(int(pc["defaults"].get("s_sess", 65536)))
            st = model.state
            bud = MP.budget_for(store, key)
            st.sess_cap(bud.get("s_sess") or bud.get("sessions"))
            off = self._calendar(st, store, systems, now, ctx.config)
            router = self._router(store, key, ctx.config)
            rows = 0
            ta = time.perf_counter()
            for s in systems:
                for ts_b, b in store.batches_since(s, EV.EVT_BATCH, st.last_batch.get(s, -math.inf)):
                    if getattr(b, "kind", EV.KIND_TXN) != EV.KIND_TXN:
                        continue
                    rows += self._sessions(st, store, s, ts_b, b, router, gap, shared, off)
                    st.last_batch[s] = ts_b
            tb = time.perf_counter()
            counted = self._count(st, store, now, D, ip2g, off)
            if st.today is not None and st.ren_day != st.today:
                # once a local day: confirmed route renames are adopted (lib/pdfg.adopt_renames)
                st.ren_day = st.today
                if DF.adopt_renames(st, int(st.today), now):
                    st.ev_since_mine += 1.0             # re-mine: the edges read the new names
            model["renamed"] = dict(st.renamed)
            t1 = time.perf_counter()
            st.cost[0] += tb - ta
            st.cost[1] += rows
            st.cost[2] += t1 - tb
            st.cost[3] += counted
            mined = False
            if st.ev_since_mine > 0 and self.entity_due(("p10", key), now, self.mine_period_s):
                self._mine(model, store, key, now)
                mined = True
            model["updated"] = now
            model["stats"] = {"rows": st.n_rows, "counted": st.n_counted, "quarantined": st.n_quar,
                              "sessions": len(st.sessions), "sess_cap": st.sessions.cap,
                              "pending": sum(len(v) for v in st.pending.values()),
                              "actions": len(st.acts.k2i), "retired": st.acts.retired,
                              "edges": len(st.edges), "prec": len(st.prec),
                              "us_per_row": st.cost[0] * 1e6 / st.cost[1] if st.cost[1] else None,
                              "us_per_counted": st.cost[2] * 1e6 / st.cost[3] if st.cost[3] else None}
            store.put_model(key, SYSTEM_ENTITY, MP.PFLOW, model, version=int(model["version"]), ts=now)
            n += counted
            stats[key] = {"rows": rows, "counted": counted, "mined": mined,
                          "ms": (time.perf_counter() - t0) * 1000.0}
        self.last_stats = stats
        return n

    # ------------------------------------------------------------- calendar
    @staticmethod
    def _calendar(st: DF.FlowState, store: Any, systems: List[str], now: float,
                  config: Mapping[str, Any]) -> float:
        """Day classes and normal-day flags of the tree's systems from P01's
        model.pcal (a day is normal when any member system had a normal day),
        so edge staleness counts normal days of the edge's day type (§6.8.1).
        Returns the local UTC offset used for day ordinals."""
        off = PW.tz_offset(config, now)
        cls: Dict[int, int] = {}
        norm: Dict[int, bool] = {}
        for s in systems:
            pc = store.get_model(s, SYSTEM_ENTITY, PCAL)
            if not isinstance(pc, Mapping):
                continue
            for d, rec in (pc.get("days") or {}).items():
                c = rec.get("class") if isinstance(rec, Mapping) else None
                if c is not None:
                    cls[int(d)] = 0 if c in ("workday", "makeup") else 1
            for d, ok in (pc.get("normal") or {}).items():
                norm[int(d)] = bool(ok) or norm.get(int(d), False)
        st.set_calendar(cls, norm, DF.EPOCH_ORD + int((now + off) // 86400.0))
        return off

    # ------------------------------------------------------------- variants
    def _router(self, store: Any, key: str, config: Mapping[str, Any]) -> Optional[Callable[[Callable], int]]:
        """Maps an event to its action variant (the node reached through the
        deepest content split on its path); None when the tree has none."""
        ptm = MP.get_ptree(store, key)
        tree = ptm.kinds.get(EV.KIND_TXN) if ptm is not None else None
        if tree is None:
            return None
        hier = MP.hierarchies(store, key, config)
        content = {nid for nid, nd in tree.nodes.items()
                   if nd.split is not None and _is_content(nd.split.attr, hier)}
        if not content:
            return None

        def route(get: Callable[[str], Any]) -> int:
            path = tree.route(get, hier)
            v = 0
            for par, child in zip(path, path[1:]):
                if par in content:
                    v = child
            return v
        return route

    # ------------------------------------------------------------- sessions
    def _sessions(self, st: DF.FlowState, store: Any, s: str, ts_b: float, b: Any,
                  router: Optional[Callable], gap: float, shared: frozenset = frozenset(),
                  off: float = 8 * 3600.0) -> int:
        if b.n == 0:
            return 0
        cb = store.batch_at(s, EV.EVT_CTX, ts_b)
        if cb is not None and cb.n != b.n:
            cb = None
        route = b.dense("http.route", None)
        sid = cb.dense("ctx.sid", None) if (self.session_mode == "ctx" and cb is not None
                                            and cb.has("ctx.sid")) else None
        skey = b.dense("sess.key", None) if b.has("sess.key") else None
        mass = b.mass()
        learn = b.learn
        pend = st.pending.setdefault((s, float(ts_b)), [])
        hcache: Dict[str, int] = {}
        order = np.argsort(b.ts, kind="stable")
        n = 0
        for i in order.tolist():
            if route[i] is not None and route[i] is not EV.ABSENT:
                rk: Optional[str] = str(route[i])
            else:
                rk = DF.route_key(lambda a, i=i: b.get(a, i))
            if rk is None:
                continue
            key = rk
            if router is not None:
                def get(a: str, i: int = i) -> Any:
                    v = b.get(a, i)
                    if v is EV.ABSENT and cb is not None:
                        v = cb.get(a, i)
                    if v is EV.ABSENT and a == "net.src":
                        return b.ip_of(i)
                    return v
                v = router(get)
                if v:
                    key = f"{rk}#v{v}"
            ip = b.ip_of(i)
            # the session key separates users only behind a shared address (NAT,
            # VDI: P11 / B17 `shared`); elsewhere per-request cookies (health
            # checks) would turn every request into its own session
            sk = (skey[i] if skey is not None and ip in shared and skey[i] not in (None, EV.ABSENT)
                  else DF.NO_KEY)
            k = (ip, sk)
            ts = float(b.ts[i])
            st.sess_hll.add(k, ts)
            e = st.sessions.get(k)
            sd = sid[i] if sid is not None and sid[i] not in (None, EV.ABSENT) else None
            if e is None:
                new = True
            elif sd is not None:
                new = e[0] != sd
            else:
                new = ts - e[1] > gap
            h = hcache.get(key)
            if h is None:
                h = hcache[key] = DF.h64(key)
            if new:
                if e is not None and e[5]:
                    pend.append(("end", e[1], ip, e[2], e[6]))
                prev, prev2, delay, earlier, bits, start = None, None, None, b"", 0, True
            else:
                prev, prev2, earlier, bits, start = e[2], e[7], e[3], e[4], False
                delay = ts - e[1] if ts >= e[1] else None
            # the route ledger sees every event (rename detection, lib/pdfg.detect_renames)
            st.see_route(rk, ip, DF.EPOCH_ORD + int((ts + off) // 86400.0),
                         DF.split_key(prev)[0] if prev else None)
            if learn[i]:
                pend.append(("row", ts, ip, key, prev, delay, start, earlier, float(mass[i]), int(i),
                             prev2 == key and prev != key))
            hs = DF.unpack_hashes(earlier)
            if h in hs:
                hs.remove(h)
            hs.append(h)
            st.sessions.put(k, [sd, max(ts, e[1]) if (e is not None and not new) else ts, key,
                                DF.pack_hashes(hs[-DF.E_MAX:]), bits | DF.bloom_bits(h),
                                bool(learn[i]), float(mass[i]), prev])
            n += 1
        st.n_rows += n
        return n

    # ------------------------------------------------------------- counting
    def _trust(self, store: Any, s: str, ip: str, at: float,
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

    def _count(self, st: DF.FlowState, store: Any, now: float, D: float,
               ip2g: Mapping[str, Any], off: float = 8 * 3600.0) -> int:
        due = sorted(k for k in st.pending if k[1] <= now - D)
        if not due:
            return 0
        tcache: Dict[Tuple[str, str, float], Tuple[float, bool]] = {}
        if st.top_t is None or now - st.top_t >= TOP_REFRESH_S:
            st.refresh_top(now)
        admitted = set(DF.active_scopes(st, now))
        # the prequential gain (P12's utility of the arm) is estimated on a
        # systematic 1-in-k subsample of at most GAIN_ROWS rows per tick, weighted
        # by k: its cost is O(successors of a) per scored row, the rest of the
        # counting is O(1) per row
        n_due = sum(1 for k in due for a in st.pending[k] if a[0] == "row")
        k_gain = max(1, -(-n_due // GAIN_ROWS))
        j_gain = 0
        n = 0
        for s, ts_b in due:
            anns = st.pending.pop((s, ts_b))
            asg = store.batch_at(s, EV.PAT_ASSIGN, ts_b)
            damp = asg.dense("damp", 1.0) if asg is not None and asg.has("damp") else None
            for ann in anns:
                if ann[0] == "end":
                    _, ts, ip, last_key, m = ann
                    tr, q = self._trust(store, s, ip, ts_b, tcache)
                    aid = st.acts.id_of(last_key)
                    if q or tr <= 0 or aid is None:
                        continue
                    for g in self._scopes(st, ip, ip2g, admitted):
                        st.ends.add((g, aid), ts, m * tr, tr)
                    continue
                _, ts, ip, bk, ak, delay, start, earlier, m, row, aba = ann
                tr, q = self._trust(store, s, ip, ts_b, tcache)
                if q:
                    st.n_quar += 1
                    continue
                dm = float(damp[row]) if damp is not None and row < len(damp) else 1.0
                f = max(0.0, min(1.0, tr * dm))
                if f <= 0:
                    continue
                omega = st.burst.unit((ip, ak, bk), ts, f)
                mass = m * f
                grp = ip2g.get(ip)
                if grp is not None:
                    g = str(grp)
                    ips = st.grp_ips.get(g) or set()
                    if len(ips) < 2:
                        ips = set(ips) | {ip}
                        st.grp_ips.put(g, ips)
                    st.grp_ss.add(g, ts, mass, omega)
                # prequential gain of the transition model ('*'), before learning
                a_id = st.acts.id_of(ak) if ak else None
                b_id0 = st.acts.id_of(bk)
                if a_id is not None and b_id0 is not None:
                    if j_gain % k_gain == 0:
                        self._gain(st, a_id, b_id0, ts, omega * k_gain)
                    j_gain += 1
                b_id, gone = st.acts.add(bk, ts, mass, omega)
                if gone is not None:
                    st.retire(gone)
                if ak:
                    a_id, gone = st.acts.ensure(ak, ts)
                    if gone is not None:
                        st.retire(gone)
                dbin = DF.delay_bin(delay)
                day = DF.EPOCH_ORD + int((ts + off) // 86400.0)
                for g in self._scopes(st, ip, ip2g, admitted):
                    st.scopes_seen.add(g)
                    st.cnt.add((g, b_id), ts, mass, omega)
                    if start:
                        st.starts.add((g, b_id), ts, mass, omega)
                    if ak and a_id is not None:
                        st.add_edge(g, a_id, b_id, ts, mass, omega, dbin, day)
                        if aba:                         # pattern b a b: a length-two loop
                            st.add_loop(g, b_id, a_id, ts, mass, omega)
                    if b_id not in st.top_b and len(st.top_b) < DF.TOP_B:
                        st.top_b.add(b_id)              # fewer than TOP_B actions: all qualify
                    if b_id in st.top_b:
                        st.pcnt.add((g, b_id), ts, mass, omega)
                        for h in DF.unpack_hashes(earlier):
                            x = st.acts.id_by_hash(h)
                            if x is not None and x != b_id:
                                st.add_prec(g, b_id, x, ts, mass, omega)
                st.ev_since_mine += omega
                st.n_counted += 1
                n += 1
        st.marg = None
        return n

    @staticmethod
    def _scopes(st: DF.FlowState, ip: str, ip2g: Mapping[str, Any], admitted: set) -> List[str]:
        g = ip2g.get(ip)
        if g is not None and str(g) in admitted:
            return [DF.STAR, str(g)]
        return [DF.STAR]

    @staticmethod
    def _gain(st: DF.FlowState, a: int, b: int, t: float, w: float) -> None:
        """Decayed (H_m) mean of log2 p_trans(b | a) - log2 p_marg(b): the
        bits per transition the DFG saves over the action marginal, scored
        before the row is learned (prequential; P12's utility of the arm)."""
        pm = DF.p_marginal(st, b, t)
        pt = DF.p_next(st, a, b, t)
        bits = math.log2(max(pt, 1e-12)) - math.log2(max(pm, 1e-12))
        if st.gain_t is not None:
            f = 2.0 ** (-(t - st.gain_t) / PS.H_M) if t > st.gain_t else 1.0
            st.gain[0] *= f
            st.gain[1] *= f
        st.gain_t = max(t, st.gain_t or t)
        st.gain[0] += w * bits
        st.gain[1] += w

    # --------------------------------------------------------------- mining
    def _mine(self, model: DF.PFlowModel, store: Any, key: str, now: float) -> None:
        st = model.state
        ptm = MP.get_ptree(store, key)
        tree = ptm.kinds.get(EV.KIND_TXN) if ptm is not None else None
        pwin = MP.get_model(store, key, MP.PWIN)
        hier = MP.hierarchies(store, key, None) if tree is not None else None
        acts_pub: Dict[int, Dict[str, Any]] = {}

        def act(aid: int) -> Dict[str, Any]:
            rec = acts_pub.get(aid)
            if rec is None:
                k = st.acts.key_of(aid) or f"#{aid}"
                r, v = DF.split_key(k)
                node = self._act_node(tree, hier, r, v)
                ent = PW.lookup(pwin, EV.KIND_TXN, v or node) if node is not None else None
                if ent is None and node is not None and v:
                    ent = PW.lookup(pwin, EV.KIND_TXN, node)
                rec = acts_pub[aid] = {"key": k, "route": r, "variant": v, "act_node": node,
                                       "n": round(DF._ev(st.cnt, (DF.STAR, aid), now), 2),
                                       "when": (ent or {}).get("when") if ent else None}
            return rec

        scopes: Dict[str, Any] = {}
        for g in DF.active_scopes(st, now):
            sc = DF.mine_scope(st, g, now)
            if g != DF.STAR and not sc["edges"] and not sc["requires"]:
                continue
            for e in sc["edges"]:
                e["from"], e["to"] = act(e["a"])["route"], act(e["b"])["route"]
                e["from_key"], e["to_key"] = act(e["a"])["key"], act(e["b"])["key"]
            for w in sc["workflows"]:
                w["routes"] = [act(x)["route"] for x in w["path"]]
                w["anchors"] = [act(x)["when"] for x in w["path"]]
            for r in sc["requires"]:
                r["to"], r["from"] = act(r["b"])["route"], act(r["a"])["route"]
            scopes[g] = sc
        for aid in sorted(st.top_b, key=lambda x: -DF._ev(st.cnt, (DF.STAR, x), now))[:64]:
            act(aid)
        model["scopes"] = scopes
        model["acts"] = acts_pub
        model["gain"] = {"bits_per_event": st.gain[0] / st.gain[1] if st.gain[1] > 0 else 0.0,
                         "rows": st.gain[1]}
        model["last_mine"] = now
        model["version"] = int(model.get("version", 0)) + 1
        st.ev_since_mine = 0.0

    @staticmethod
    def _act_node(tree: Any, hier: Any, route: str, variant: int) -> Optional[int]:
        """The shallowest node whose context constrains the route (rendering
        and P09 anchors); the variant node itself when it still exists."""
        if tree is None:
            return None
        if variant and variant in tree.nodes:
            return variant
        attr = "http.route"
        get_val: Dict[str, Any] = {"http.route": route}
        if route.startswith(("TLS ", "DNS ", "DST ")):
            attr = {"TLS": "tls.sni", "DNS": "dns.qname", "DST": "net.dst"}[route[:3]]
            get_val = {attr: route[4:]}
        try:
            path = tree.route(lambda a: get_val.get(a, EV.ABSENT), hier)
        except Exception:
            return None
        for nid in path:
            nd = tree.nodes[nid]
            if any(c[0] == attr and not c[3] for c in nd.ctx):
                return nid
        return None
