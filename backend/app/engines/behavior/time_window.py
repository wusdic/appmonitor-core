"""P09 TimeWindow (`behavior.time_window`) — activity windows per pattern node
and day type (docs/lib3/progressive.md §6.13, card P09). Library 3 (behaviour).

Requirement S5/S10 ("早上9点到9点21分 … 访问登录页面", "下午5点提交报告"): the time
windows of a behaviour are LEARNED from its arrivals, per pattern (a route, a
route of one group, one IP's exception …) and per day type, not configured.
They become more precise with observation time (slot edges first, minute
edges once P04 keeps a minute reservoir for the node; coverage, distinct dates
and day stability grow with evidence) and follow behaviour when it changes
(the densities are H_m-decayed; a node is refitted when new evidence arrived).

Reads   model.ptree (every txn node's when summary: hist96 per day type at
        H_m, evidence per day type at H_m / on the confidence channel, the
        optional minute reservoir), model.sysprof (arm 'p09' / 'time_window':
        'off' skips the tree).
Writes  model.pwin@(tree key, '__system__'):
          {'fmt': 1, 'version', 'updated', 'last_run',
           'root':  {'all': [(s, e, label)], 'by_daytype': {'wd': [...], 'nwd': [...]},
                     'version', 'updated'}      <- time level l2 of lib/phier (P04 splits,
                                                   P03 routing), changed only when an
                                                   endpoint moves > 10 min (hysteresis)
           'nodes': {kind: {nid: entry}},       entry = {'status': 'fitted'|'none',
                      'by_daytype': {'wd'|'nwd': rec | None}, 'when': {'workday':
                      [[m0, m1]], 'nonworkday': [...], 'coverage', 'confidence'},
                      'n_c', 'dates', 'cver', 'fit_t', 'state', 'text_zh', 'text_en'}
                      rec = lib/pwindows.fit_daytype(...) + {'n_c', 'confidence',
                      'text_zh', 'text_en'}; 'when' is the statement-contract block
                      (eval/pmetrics) P14 copies into `evidence.when`.
           'fit':   {kind: {nid: mark}}          dirty-node bookkeeping
           'pending': {}                         P09 never blocks confirmation
           'gain':  {'bits_per_event', 'us_per_event', 'nodes_fitted', 'ms'}}
        model.pwant['p09'] = {'minute_reservoir': {kind: [nid, ...]}}: nodes with
        >= 50 % of a day type's mass in <= 4 slots (§5.5.4), sticky, <= R_NODES
        per tree (largest mass first), so minute memory is bounded per tree.
Cadence 6 h per tree (entity_due, crc32 phase). A tree without learned events
        since the last run is skipped; inside a tree only dirty nodes are refitted:
        decay-corrected new evidence >= 20 units or >= 10 % of the evidence at the
        last fit, or a state / version change (§6.20). Periodic cost therefore
        follows the evidence that arrived, not #systems x N_max; it never depends
        on the number of IPs or attributes (a node's when summary has a fixed size).
Change  while P04 has an open Page-Hinkley alarm on the node's arrival time
        (node.meta['evolving']['@when']), windows are PROVISIONAL and fitted
        from the reservoir arrivals since the alarm; when P04 accepts the change
        (the alarm closes with cver + 1: persistence over T_persist normal days,
        >= 2 IPs of the node or one IP for 5 days, no quarantine, §6.9.2) the
        constraints SWITCH to the new regime's arrivals; an expired alarm
        restores the full data. The H_m hist cannot be split in time, so a
        slot-mode node follows a change with the H_m half-life only.
Budget  per dirty node and day type: one Bayesian-Blocks DP on 96 slot cells
        (~1.5 ms) or <= 513 minute cells (~6 ms); memory O(nodes x windows).
Inert unless config['progressive']['enabled'].
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Tuple


from ...core.engine import Context, Engine
from ...models.schema import SYSTEM_ENTITY
from .lib import m_ptree as MP
from .lib import pevent as EV
from .lib import psketch as PS
from .lib import pwindows as PW

N_FIT_MIN = 5.0                 # H_m evidence units of a day type before windows are fitted
DIRTY_UNITS = 20.0
DIRTY_FRAC = 0.10
R_NODES = 256                   # minute reservoirs requested per tree (x ~2.5 KB each)
PERIOD_S = 6 * 3600.0
DT_KEYS = PW.DAYTYPES           # ('wd', 'nwd'); index 0 = workday, 1 = non-workday


def empty_model() -> Dict[str, Any]:
    return {"fmt": 1, "version": 0, "updated": None, "last_run": None,
            "root": {"all": [], "by_daytype": {}, "version": 0, "updated": None},
            "nodes": {}, "fit": {}, "pending": {}, "gain": {}}


def _arm_off(store: Any, key: str) -> bool:
    sp = store.get_model(key, SYSTEM_ENTITY, MP.SYSPROF)
    ch = (sp or {}).get("chosen") if isinstance(sp, Mapping) else None
    if isinstance(ch, Mapping):
        for n in ("p09", "time_window", "when"):
            if str(ch.get(n, "on")).lower() == "off":
                return True
    return False


def _mark(node: Any, n_tot: float, now: float, regime: Mapping[str, Any]) -> Dict[str, Any]:
    return dict({"n": float(n_tot), "t": float(now), "state": node.state, "version": int(node.version),
                 "cver": int(node.cver)}, **regime)


def _when_alarm(node: Any) -> Optional[float]:
    """Start of an open P04 alarm on the arrival time (node.meta['evolving']
    ['@when']['t0'], set by P04's Page-Hinkley on the circular minute), else None."""
    meta = getattr(node, "meta", None)
    ev = meta.get("evolving") if isinstance(meta, Mapping) else None
    st = ev.get("@when") if isinstance(ev, Mapping) else None
    t0 = st.get("t0") if isinstance(st, Mapping) else None
    return float(t0) if t0 is not None else None


def _regime(mark: Optional[Mapping[str, Any]], node: Any) -> Dict[str, Any]:
    """Which arrivals describe the node's CURRENT time behaviour (§6.9.2):
      alarm open (P04 evolving on '@when')  -> provisional windows from the
                                               arrivals since the alarm
      alarm closed with cver + 1 (accepted) -> the new regime: arrivals since
                                               the alarm, from then on
      alarm closed otherwise (expired)      -> all arrivals again.
    The H_m hist cannot be split by time, so this applies to minute mode."""
    mk = dict(mark or {})
    t_alarm = _when_alarm(node)
    out = {"when_t0": mk.get("when_t0"), "alarm_cver": mk.get("alarm_cver"),
           "regime_t0": mk.get("regime_t0")}
    if t_alarm is not None:
        if out["when_t0"] is None:
            out["alarm_cver"] = int(node.cver)
        out["when_t0"] = t_alarm
    elif out["when_t0"] is not None:
        ac = out.get("alarm_cver")
        if ac is not None and int(node.cver) > int(ac):
            out["regime_t0"] = out["when_t0"]
        out["when_t0"], out["alarm_cver"] = None, None
    return out


def _dirty(mark: Optional[Mapping[str, Any]], node: Any, n_tot: float, now: float) -> bool:
    """New evidence since the last fit (decay-corrected) >= 20 units or >= 10 %
    of what the fit saw, or a structural / lifecycle change of the node."""
    if not mark:
        return True
    if _changed(mark, node):
        return True
    old = float(mark.get("n", 0.0)) * 2.0 ** (-(now - float(mark.get("t", now))) / PS.H_M)
    new = n_tot - old
    return new > 0.5 and (new >= DIRTY_UNITS or new >= DIRTY_FRAC * max(old, 1e-9))


def _changed(mark: Optional[Mapping[str, Any]], node: Any) -> bool:
    """A lifecycle / structural / content-version change or an alarm since the
    last fit (O(1); lets an idle tree skip without touching any summary)."""
    if not mark:
        return node.last_seen is not None
    return (mark.get("state") != node.state or int(mark.get("version", -1)) != int(node.version)
            or int(mark.get("cver", -1)) != int(node.cver) or _when_alarm(node) != mark.get("when_t0"))


def _points(node: Any, d: int, since: Optional[float] = None) -> List[Tuple[float, float]]:
    res = node.when.res
    if res is None or not len(res):
        return []
    return [(float(it[1]), float(t)) for it, _w, t in res.items()
            if int(it[0]) == d and (since is None or t >= since)]


class TimeWindowEngine(Engine):
    name = "behavior.time_window"
    layer = "behavior"
    consumes = [MP.PTREE, MP.SYSPROF]
    produces = [MP.PWIN, MP.PWANT]
    description = "P09: activity windows per pattern node and day type (Bayesian Blocks on the circular day)"
    period_s = PERIOD_S
    interval = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.last_stats: Dict[str, Any] = {}

    # ----------------------------------------------------------------- run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        if not EV.enabled(ctx.config):
            return 0
        store = ctx.store
        now = float(ctx.now)
        keys = sorted({MP.tree_key(store, s) for s in store.systems()} |
                      {MP.tree_key(store, s) for s in store.batch_systems(EV.EVT_BATCH)})
        fitted = 0
        stats: Dict[str, Any] = {}
        for key in keys:
            ptm = MP.get_ptree(store, key)
            if ptm is None or EV.KIND_TXN not in ptm.kinds:
                continue
            if not self.entity_due(("p09", key), now):
                continue
            if _arm_off(store, key):
                stats[key] = {"off": True}
                continue
            n, st = self.fit_tree(ctx, key, ptm, now)
            fitted += n
            stats[key] = st
        self.last_stats = stats
        return fitted

    # ------------------------------------------------------------- per tree
    def fit_tree(self, ctx: Context, key: str, ptm: Any, now: float) -> Tuple[int, Dict[str, Any]]:
        store = ctx.store
        t0 = time.perf_counter()
        model = MP.get_model(store, key, MP.PWIN)
        if not isinstance(model, dict) or model.get("fmt") != 1:
            model = empty_model()
        kind = EV.KIND_TXN
        tree = ptm.kinds[kind]
        root = tree.nodes.get(tree.root)
        last_run = model.get("last_run")
        fits0 = model["fit"].get(kind) or {}
        idle = last_run is not None and root is not None and root.last_seen is not None \
            and root.last_seen <= last_run and bool(model["nodes"].get(kind)) \
            and not any(_changed(fits0.get(nid), nd) for nid, nd in tree.nodes.items())
        if root is None or root.last_seen is None or idle:
            model["last_run"] = now
            store.put_model(key, SYSTEM_ENTITY, MP.PWIN, model, version=int(model["version"]), ts=now)
            return 0, {"fitted": 0, "skipped": True}
        off = PW.tz_offset(ctx.config, now)
        fits = model["fit"].setdefault(kind, {})
        outn = model["nodes"].setdefault(kind, {})
        for nid in [n for n in list(outn) if n not in tree.nodes]:          # retired nodes
            outn.pop(nid, None)
            fits.pop(nid, None)
        n_fit, gain_num, gain_den, new_ev = 0, 0.0, 0.0, 0.0
        for nid, node in tree.nodes.items():
            if node.last_seen is None:
                continue
            n_tot = node.when.evidence(0, now, conf=False) + node.when.evidence(1, now, conf=False)
            mk = fits.get(nid)
            if not _dirty(mk, node, n_tot, now):
                continue
            if mk:
                new_ev += max(0.0, n_tot - float(mk["n"]) * 2.0 ** (-(now - float(mk["t"])) / PS.H_M))
            else:
                new_ev += n_tot
            regime = _regime(mk, node)
            entry = self.fit_node(node, now, off, outn.get(nid), regime)
            outn[nid] = entry
            fits[nid] = _mark(node, n_tot, now, regime)
            n_fit += 1
            if entry["status"] == "fitted":
                for d, dk in enumerate(DT_KEYS):
                    if entry["by_daytype"].get(dk):
                        gain_num += node.when.mass(d, now) * PW.entropy_gain_bits(node.when.hist[d])
                        gain_den += node.when.mass(d, now)
        self._root(model, root, now, off)
        want = self._want(store, key, tree, now)
        ms = (time.perf_counter() - t0) * 1000.0
        model["last_run"] = now
        model["updated"] = now
        model["version"] = int(model.get("version", 0)) + (1 if n_fit else 0)
        model["pending"] = {}
        if n_fit or not model.get("gain"):
            model["gain"] = {"bits_per_event": gain_num / gain_den if gain_den > 0 else 0.0,
                             "us_per_event": ms * 1000.0 / new_ev if new_ev > 0 else None,
                             "nodes_fitted": n_fit, "ms": ms}
        store.put_model(key, SYSTEM_ENTITY, MP.PWIN, model, version=model["version"], ts=now)
        return n_fit, {"fitted": n_fit, "ms": ms, "minute_nodes": want}

    # ------------------------------------------------------------- per node
    def fit_node(self, node: Any, now: float, off: float, old: Optional[Mapping[str, Any]],
                 regime: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        by: Dict[str, Optional[Dict[str, Any]]] = {}
        rg = regime or {}
        since = rg.get("when_t0") if rg.get("when_t0") is not None else rg.get("regime_t0")
        for d, dk in enumerate(DT_KEYS):
            n_m = node.when.evidence(d, now, conf=False)
            if n_m < N_FIT_MIN:
                by[dk] = None
                continue
            pts = _points(node, d)
            recent = _points(node, d, since) if since is not None else pts
            use_recent = since is not None and len(recent) >= PW.MIN_POINTS
            rec = PW.fit_daytype(node.when.hist[d], n_m, recent if use_recent else pts, tz_offset_s=off)
            if rec is None:
                by[dk] = None
                continue
            if since is not None:
                rec["since"] = float(since)
                rec["provisional"] = rg.get("when_t0") is not None
                rec["regime"] = "new" if use_recent else "mixed"
            if rec.get("dates") is None:
                rec["dates"] = node.n_days()
            rec["n_c"] = node.when.evidence(d, now, conf=True)
            rec["confidence"] = PW.confidence(rec)
            rec["text_zh"] = PW.render_zh(dk, rec)
            rec["text_en"] = PW.render_en(dk, rec)
            by[dk] = rec
        fitted = [r for r in by.values() if r]
        prev = (old or {}).get("by_daytype") or {}
        cv = int((old or {}).get("cver", 0))
        if old and any(bool(prev.get(dk)) != bool(by.get(dk)) or
                       (prev.get(dk) and by.get(dk) and PW.moved(prev[dk]["windows"], by[dk]["windows"]))
                       for dk in DT_KEYS):
            cv += 1
        masses = [node.when.mass(d, now) for d in range(2)]
        cov_num = sum(masses[d] * by[dk]["coverage"] for d, dk in enumerate(DT_KEYS) if by.get(dk))
        cov_den = sum(masses[d] for d, dk in enumerate(DT_KEYS) if by.get(dk))
        when = {PW.DT_LONG[dk]: (PW.as_intervals(by[dk]["windows"]) if by.get(dk) else [])
                for dk in DT_KEYS}
        when["coverage"] = cov_num / cov_den if cov_den > 0 else None
        when["confidence"] = min([r["confidence"] for r in fitted]) if fitted else None
        return {"status": "fitted" if fitted else "none", "by_daytype": by, "when": when,
                "n_c": float(node.n_c(now)), "dates": int(node.n_days()), "cver": cv,
                "fit_t": float(now), "state": node.state,
                "text_zh": "；".join(r["text_zh"] for r in fitted),
                "text_en": "; ".join(r["text_en"] for r in fitted)}

    # ------------------------------------------------------------ root / l2
    def _root(self, model: Dict[str, Any], root: Any, now: float, off: float) -> None:
        """System-root windows = the learned time level l2 (`w:0900-0921`);
        replaced only on a material change (hysteresis: P04's split statistics
        keyed by window labels must not be reset every 6 h)."""
        cur = model.setdefault("root", {"all": [], "by_daytype": {}, "version": 0, "updated": None})
        new_by: Dict[str, List[Tuple[int, int, str]]] = {}
        for d, dk in enumerate(DT_KEYS):
            n_m = root.when.evidence(d, now, conf=False)
            if n_m < N_FIT_MIN:
                continue
            rec = PW.fit_daytype(root.when.hist[d], n_m, _points(root, d), tz_offset_s=off)
            if rec is not None and not rec.get("all_day"):
                new_by[dk] = [(int(s), int(e), lab) for (s, e), lab in zip(rec["windows"], rec["labels"])]
        n_all = root.when.evidence(0, now, conf=False) + root.when.evidence(1, now, conf=False)
        new_all: List[Tuple[int, int, str]] = []
        if n_all >= N_FIT_MIN:
            pts = _points(root, 0) + _points(root, 1)
            rec = PW.fit_daytype(root.when.hist[0] + root.when.hist[1], n_all, pts, tz_offset_s=off)
            if rec is not None and not rec.get("all_day"):
                new_all = [(int(s), int(e), lab) for (s, e), lab in zip(rec["windows"], rec["labels"])]
        changed = PW.moved([w[:2] for w in cur.get("all") or []], [w[:2] for w in new_all])
        for dk in DT_KEYS:
            changed = changed or PW.moved([w[:2] for w in (cur.get("by_daytype") or {}).get(dk) or []],
                                          [w[:2] for w in new_by.get(dk) or []])
        if changed:
            cur["all"] = new_all
            cur["by_daytype"] = new_by
            cur["version"] = int(cur.get("version", 0)) + 1
            cur["updated"] = now

    # ------------------------------------------------------ minute requests
    def _want(self, store: Any, key: str, tree: Any, now: float) -> int:
        """Ask P04 for minute reservoirs on concentrated nodes (sticky, capped)."""
        want = MP.get_model(store, key, MP.PWANT)
        if not isinstance(want, dict):
            want = {}
        sub = want.get("p09") if isinstance(want.get("p09"), dict) else {}
        mr = sub.get("minute_reservoir") if isinstance(sub.get("minute_reservoir"), Mapping) else {}
        kind = EV.KIND_TXN
        prev = {int(n) for n in (mr.get(kind) or mr.get(str(kind)) or []) if int(n) in tree.nodes}
        cand = set(prev)
        for nid, node in tree.nodes.items():
            if nid in cand or node.last_seen is None:
                continue
            for d in range(2):
                if node.when.evidence(d, now, conf=False) >= N_FIT_MIN and PW.concentrated(node.when.hist[d]):
                    cand.add(nid)
                    break
        if len(cand) > R_NODES:
            cand = set(sorted(cand, key=lambda n: -tree.nodes[n].mass_at(now))[:R_NODES])
        nids = sorted(cand)
        if sorted(prev) != nids or "p09" not in want:
            want["p09"] = {"minute_reservoir": {kind: nids}}
            store.put_model(key, SYSTEM_ENTITY, MP.PWANT, want, ts=now)
        return len(nids)

