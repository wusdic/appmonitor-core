"""P05 AttributeSelectionEngine (`behavior.attr_select`) — which metrics are used,
kept, and at which detail, per system and per node.

docs/lib3/progressive.md §6.4, card P05. Library 3 (behaviour). Maths in lib/pselect.

Every tick: the learned rows of the tick's evt.batch (with its aligned evt.ctx
columns) and evt.win enter a stratified probe per (tree key, kind)
(lib/pselect.StratifiedProbe, R_p rows, stratum = (ev.ch, the tree root
split's value) or the bootstrap key before the root has split).

Hourly per tree (entity_due, deterministic phase): evaluate every attribute
holding a role other than `dropped` plus a rotating crc32 slice of A_probe = 64
others (dropped ones are re-probed every 7 days), so the run costs
O((n_kept + 64) x levels x targets x R_p) numpy work and never sweeps the
registry. Roles with hysteresis: invariant / split / target / shape /
redundant / dropped; `net.src` leaves the split candidates when it carries no
information at any IP level (who_mode none, the "直接 IP 不作为特征" case).
Per node: node_overrides from the node summaries in model.ptree. Daily:
categorical value groups (registry level 1). Writes model.attrsel and
registry.set_role; every role change emits `attribute_role` INFO.

Inert unless config['progressive']['enabled'].
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import ORG, SYSTEM_ENTITY, BehaviorEvent, Severity
from .lib import m_ptree as MP
from .lib import pevent as EV
from .lib import pselect as SEL
from .lib.combine import seeded_uniform

PERIOD_S = 3600.0
A_ROW = 128                # attributes kept per probe row
STABLE_RUNS = 6            # runs without a role change before the cadence relaxes
STABLE_PERIOD_S = 6 * 3600.0
ROLE_ZH = {"invariant": "不变式", "split": "细分候选", "target": "约束目标", "shape": "仅保留形状",
           "redundant": "冗余", "dropped": "丢弃", "probe": "待评估"}


class ProbeState:
    """P05's probe reservoirs inside model.attrsel ({kind: StratifiedProbe}).
    A slotted holder: memory reports count it (deep size follows __slots__),
    while plain-data copies of the published model (eval snapshots, API
    dumps) render it as one line instead of copying R_p rows per kind."""

    __slots__ = ("probes",)

    def __init__(self, probes: Dict[int, SEL.StratifiedProbe]) -> None:
        self.probes = probes

    def __repr__(self) -> str:
        return "<P05 probes " + ", ".join(f"kind {k}: {len(p)} rows" for k, p in self.probes.items()) + ">"


class AttributeSelectionEngine(Engine):
    name = "behavior.attr_select"
    layer = "behavior"
    consumes = [EV.EVT_BATCH, EV.EVT_CTX, EV.EVT_WIN, MP.ATTR, MP.PTREE]
    produces = [MP.ATTRSEL, "event.attribute_role"]
    description = "P05: attribute roles, kept sets, per-node target overrides, redundancy"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.probes: Dict[Tuple[str, int], SEL.StratifiedProbe] = {}
        self.run_index: Dict[str, int] = {}
        self.last_vg: Dict[str, float] = {}
        self.last_stats: Dict[str, Any] = {}
        self.eval_period_s = float(params.get("eval_period_s", PERIOD_S))
        self._want_cache: Dict[str, Tuple[int, Optional[set]]] = {}
        self._stab: Dict[str, Dict[str, Any]] = {}
        self._last_fp: Dict[str, Tuple] = {}

    # ----------------------------------------------------------- probe
    def _stratum_fn(self, store: Any, key: str, kind: int, config: Mapping[str, Any]):
        """(tree, hierarchies) when the tree has split: probe rows are stratified by
        the leaf they route to. The root split alone (the first version) left a
        rare action without rows: on pack O's OA the 60-s health monitor filled
        the staff stratum with /health rows (435 rows, none of them logins), so
        the login node never got node-local targets (username, body size)."""
        m = MP.get_ptree(store, key)
        tree = m.kinds.get(kind) if m is not None else None
        sp = tree.nodes[tree.root].split if tree is not None else None
        if sp is None:
            return None, None
        return tree, MP.hierarchies(store, key, config)

    def _offer(self, store: Any, s: str, key: str, kind: int, b: EV.EventBatch,
               cb: Optional[EV.EventBatch], now: float, config: Mapping[str, Any], R: int) -> int:
        rr = b.learned_rows()
        if rr.size == 0:
            return 0
        pk = (key, kind)
        pr = self.probes.get(pk)
        if pr is None:
            pr = self.probes[pk] = SEL.StratifiedProbe(R)
        root, hier = self._stratum_fn(store, key, kind, config)
        mass = b.mass()
        want = self._wanted(store, key, now, root)
        names = [nm for nm in b.names() if want is None or nm in want]
        cnames = [nm for nm in (cb.names() if cb is not None and cb.n == b.n else [])
                  if want is None or nm in want]
        if want is None and len(names) + len(cnames) > A_ROW:
            names = names[:A_ROW]
            cnames = cnames[:max(0, A_ROW - len(names))]
            want = set(names) | set(cnames)
        wid = pr.want_id(want)
        # row-aligned value lists, read once per batch (EventBatch._values; the
        # same values and dict order as b.get / cb.get per row and name)
        ABSENT = EV.ABSENT
        bl = [(nm, b._values(nm)) for nm in names]
        cl = [(nm, cb._values(nm)) for nm in cnames]
        for i in rr.tolist():
            row = {}
            for nm, lst in bl:
                if lst is not None:
                    v = lst[i]
                    if v is not ABSENT:
                        row[nm] = v
            for nm, lst in cl:
                if nm not in row and lst is not None:
                    v = lst[i]
                    if v is not ABSENT:
                        row[nm] = v
            row.setdefault("net.src", b.ip_of(i))
            if root is not None:
                try:
                    leaf = root.route(lambda nm, row=row: row.get(nm, EV.ABSENT), hier, (), float(b.ts[i]))[-1]
                except Exception:
                    leaf = root.root
                st = (row.get("ev.ch", ""), int(leaf))
            else:
                st = EV.bootstrap_stratum(b, i)
            u = seeded_uniform(s, float(b.t1), "p05", int(b.rid[i]))
            pr.offer(st, row, float(mass[i]), float(b.ts[i]), u, wid)
        return int(rr.size)

    def _wanted(self, store: Any, key: str, now: float, root: Any) -> Optional[set]:
        """Attributes a probe row keeps (bounded memory, O(R_p x A_ROW)): every
        attribute holding a role other than dropped, the rotating slice the NEXT
        hourly run will evaluate, the context seeds and the tree's split
        attributes. None before the first run (then the first A_ROW columns)."""
        hit = self._want_cache.get(key)
        hour = int(now // self.eval_period_s)
        if hit is not None and hit[0] == hour:
            return hit[1]
        reg = MP.get_registry(store, key)
        prev = store.get_model(key, SYSTEM_ENTITY, MP.ATTRSEL)
        if reg is None or not isinstance(prev, Mapping):
            self._want_cache[key] = (hour, None)
            return None
        roles = prev.get("roles") or {}
        todo = SEL.names_to_evaluate(reg.names(), roles, prev.get("dropped_at") or {}, now,
                                     self.run_index.get(key, 0))
        want = set(todo[:A_ROW]) | set(SEL.CONTEXT_SEEDS) | {"net.src", "ev.ch"}
        pt = MP.get_ptree(store, key)
        if pt is not None:
            for tr in pt.kinds.values():
                want.update(nd.split.attr for nd in tr.nodes.values() if nd.split is not None)
        self._want_cache[key] = (hour, want)
        return want

    # ------------------------------------------------------------------ run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        if not EV.enabled(ctx.config):
            return 0
        store = ctx.store
        now = float(ctx.now)
        systems = sorted(set(store.batch_systems(EV.EVT_BATCH)) | set(store.batch_systems(EV.EVT_WIN)))
        keys: Dict[str, str] = {}
        n = 0
        for s in systems:
            key = MP.tree_key(store, s)
            keys.setdefault(key, s)
            R = int(MP.budget_for(store, key).get("r_p", SEL.R_P))
            b = store.batch_at(s, EV.EVT_BATCH, now)
            if b is not None:
                n += self._offer(store, s, key, EV.KIND_TXN, b, store.batch_at(s, EV.EVT_CTX, now),
                                 now, ctx.config, R)
            w = store.batch_at(s, EV.EVT_WIN, now)
            if w is not None:
                n += self._offer(store, s, key, EV.KIND_WIN, w, None, now, ctx.config, R)
        runs = 0
        skipped = 0
        skip_unchanged = bool((EV.pconfig(ctx.config).get("defaults") or {}).get("p05_skip_unchanged", False))
        for key, s in keys.items():
            # hourly while the selection is moving; once the roles have not changed
            # for STABLE_RUNS runs and no attribute appeared, every STABLE_PERIOD_S
            # (any change returns to hourly)
            reg = MP.get_registry(store, key)
            rv = getattr(reg, "version", 0) if reg is not None else 0
            stv = self._stab.setdefault(key, {"stable": 0, "regv": rv})
            if rv != stv["regv"] and reg is not None and len(reg) != stv.get("nattr"):
                stv["stable"] = 0
            stv["regv"], stv["nattr"] = rv, (len(reg) if reg is not None else 0)
            period = self.eval_period_s if stv["stable"] < STABLE_RUNS else STABLE_PERIOD_S
            if not self.entity_due(("attrsel", key, period), now, period):
                continue
            # (round 5, opt-in: progressive.defaults.p05_skip_unchanged) an
            # evaluation reads the probe, the registry, the tree's structure and
            # the learned hierarchies (groups, windows, families); when none of
            # them changed since the key's last evaluation it would re-measure
            # the same rows (the probe weights only decay, uniformly per stratum:
            # every entropy and gain is scale-free), so it is skipped and does
            # not count as a run. Measured on PG4's servers pack (100 systems,
            # 1 day): 536 of 1 442 due evaluations skipped (P05 was 63 s of the
            # P-core's 66 s). Off by default: on pack O seed 0 the slower
            # rotation of the evaluated slice and the hysteresis counted in
            # evaluations of new data changed the mail tree's first split
            # (region instead of /16) and lost the finance / 综合部 mail parts
            # (recall@21 0.82 -> 0.79; round-5 runs r5b vs r5c).
            fp = self._fingerprint(store, key, reg) if skip_unchanged else None
            if fp is not None and self._last_fp.get(key) == fp:
                skipped += 1
                continue
            if self.evaluate_key(store, key, s, now, ctx.config):
                runs += 1
                if fp is not None:
                    self._last_fp[key] = fp
        self.last_stats = {"offered": n, "evaluations": runs, "skipped_unchanged": skipped}
        return n + runs

    def _fingerprint(self, store: Any, key: str, reg: Any) -> Tuple:
        """What an evaluation of `key` depends on, as change counters: rows
        offered to its probes, the registry version and size, each tree's
        structure (node ids, last lineage entry) and the versions of the models
        the hierarchies are built from (P11 groups, P09 windows, families)."""
        offered = tuple(sorted((kd, p.n_offered, len(p)) for (k, kd), p in self.probes.items() if k == key))
        pt = MP.get_ptree(store, key)
        trees = ()
        if pt is not None:
            trees = tuple(sorted((kd, tr.next_id, len(tr.nodes), tuple(tr.lineage[-1][:3]) if tr.lineage else ())
                                 for kd, tr in pt.kinds.items()))
        mv = (store.model_version(ORG, ORG, MP.WHO_GROUPS), store.model_version(ORG, ORG, MP.SYSFAM),
              store.model_version(key, SYSTEM_ENTITY, MP.PWIN))
        return (offered, getattr(reg, "version", None), len(reg) if reg is not None else 0, trees, mv)

    # -------------------------------------------------------------- evaluate
    def evaluate_key(self, store: Any, key: str, s: str, now: float,
                     config: Mapping[str, Any]) -> bool:
        reg = MP.get_registry(store, key)
        if reg is None:
            return False
        prev = store.get_model(key, SYSTEM_ENTITY, MP.ATTRSEL)
        prev = dict(prev) if isinstance(prev, Mapping) else {}
        run_idx = self.run_index.get(key, 0)
        self.run_index[key] = run_idx + 1
        hier = MP.hierarchies(store, key, config, reg)
        roles_prev = dict(prev.get("roles") or {})
        stats: Dict[str, Dict[str, Any]] = {}
        ipinfo: Dict[int, float] = {}
        any_rows = False
        for kind in (EV.KIND_TXN, EV.KIND_WIN):
            pr = self.probes.get((key, kind))
            if pr is None or not len(pr):
                continue
            any_rows = True
            pr.rebalance(now)
            names = [a for a in reg.names() if kind in reg.records[a].kinds and reg.records[a].state != "gone"]
            present = set(pr.names())
            names = [a for a in names if a in present]
            # the rotating slice covers every attribute at least once a day whatever the cadence
            stable = self._stab.get(key, {}).get("stable", 0) >= STABLE_RUNS
            a_probe = SEL.A_PROBE * (int(STABLE_PERIOD_S // self.eval_period_s) if stable else 1)
            todo = SEL.names_to_evaluate(names, roles_prev, prev.get("dropped_at") or {}, now, run_idx,
                                         a_probe)
            tprev = list((prev.get("targets_sys") or {}).get(kind) or [])
            if not tprev:
                tprev = SEL.bootstrap_selection(reg, kind, names)["targets_sys"][kind]
            sprev = [a for a, _ in (prev.get("split_cands") or {}).get(kind) or []]
            st = SEL.evaluate(pr, now, hier, todo, reg, tprev, sprev,
                              coverage=lambda a: reg.coverage(a, now),
                              stability=lambda a: float(reg.records[a].stability) if a in reg.records else 1.0,
                              cost_us=lambda a: float(getattr(reg.records.get(a), "cost_us", 0.0) or 0.0),
                              proxies=prev.get("who_proxies") or ())
            stats.update(st)
            if kind == EV.KIND_TXN and "net.src" in present:
                ipinfo = SEL.ip_information(pr, now, hier, tprev)
        if not any_rows:
            return False
        # redundancy among kept (target-grade) attributes of the txn probe
        pr0 = self.probes.get((key, EV.KIND_TXN))
        cand = [a for a, v in stats.items() if SEL.targetable(a) and v["U_t"] >= SEL.U_HI
                and v["H"] >= SEL.RED_HB]
        cand = sorted(cand, key=lambda a: -stats[a]["U_t"])[:3 * SEL.M_T]
        # the members of last run's redundant pairs stay in the comparison while
        # they are still evaluated (a pair must not dissolve because its keeper
        # dropped out of the top-U_t list for one run)
        prev_red = dict(prev.get("redundant") or {})
        cand += [a for a in dict.fromkeys(list(prev_red) + list(prev_red.values()))
                 if a in stats and a not in cand and stats[a]["H"] >= SEL.RED_HB]
        red = {}
        if pr0 is not None and len(pr0):
            red = SEL.redundancy(pr0, now, hier, cand,
                                 cost=lambda a: float(getattr(reg.records.get(a), "cost_us", 0.0) or 0.0),
                                 cov=lambda a: reg.coverage(a, now), prev=prev_red,
                                 distinct=lambda a: float(stats[a].get("distinct0", 0.0)))
        out = SEL.assign_roles(stats, prev, red, ipinfo, now,
                               lambda a: reg.records[a].kinds if a in reg.records else ())
        proxies: List[str] = []
        if pr0 is not None and len(pr0):
            # every kept attribute: a source property that is only a target (a client
            # stack P05 kept as a target) must not pay for splits either (§6.5.3)
            proxies = SEL.who_proxies(pr0, now, hier,
                                      [a for a, r in out["roles"].items() if r in ("split", "target", "shape")])
        # per-node overrides from the node summaries
        pt = MP.get_ptree(store, key)
        overrides: Dict[int, Dict[int, List[str]]] = {}
        if pt is not None:
            gone = [a for a, r in reg.records.items() if r.state == "gone"]
            for kind, tree in pt.kinds.items():
                prk = self.probes.get((key, kind))
                if prk is not None and len(prk):
                    tsys = out["targets_sys"].get(kind, [])
                    local = [a for a, r in out["roles"].items()
                             if r in ("split", "target", "shape") and a not in tsys
                             and SEL.targetable(a) and a in reg.records
                             and kind in set(reg.records[a].kinds)]
                    # n_min 12 (not 32): a rare action's node (a department's
                    # login, 3 rows a workday in the stratified probe) inherited
                    # its ancestor's targets for ~11 days, so its username /
                    # opinion fields were fitted from day ~17 (measured on pack O)
                    overrides[kind] = SEL.node_targets_from_probe(
                        tree, prk, now, hier, tsys, gone, n_min=12, local_pool=local, proxies=proxies)
        # daily: categorical value groups (registry level 1)
        if now - self.last_vg.get(key, -math.inf) >= 86400.0 and pr0 is not None and len(pr0):
            self.last_vg[key] = now
            tg = out["targets_sys"].get(EV.KIND_TXN, [])
            for a, r in reg.records.items():
                if r.type != "categorical" or a in SEL.WHO_ATTRS or not SEL.targetable(a):
                    continue
                if out["roles"].get(a) not in ("split", "target"):
                    continue
                vg = SEL.value_groups(pr0, now, hier, a, tg)
                if vg != (r.hier.get("groups") or {}):
                    reg.set_value_groups(a, vg)
        changes = []
        stv = self._stab.setdefault(key, {"stable": 0})
        for a, role in out["roles"].items():
            if roles_prev.get(a, "probe") != role:
                changes.append((a, roles_prev.get(a, "probe"), role))
                reg.set_role(a, role)
        stv["stable"] = 0 if changes else stv.get("stable", 0) + 1
        version = int(prev.get("version", 0)) + (1 if changes or not prev else 0)
        model = {"version": version, "t": now, "roles": out["roles"], "levels": out["levels"],
                 "targets_sys": out["targets_sys"], "split_cands": out["split_cands"],
                 "redundant": out["redundant"], "node_overrides": overrides,
                 "who_mode": out["who_mode"], "who_proxies": proxies, "ip_info": {int(k): float(v) for k, v in ipinfo.items()},
                 "low": out["low"], "dropped_at": out["dropped_at"], "ustat": out["ustat"],
                 "runs": run_idx + 1,
                 # the probes are part of P05's state (counted by memory reports)
                 "probes": ProbeState({k: self.probes[(key, k)] for k in (EV.KIND_TXN, EV.KIND_WIN)
                                       if (key, k) in self.probes})}
        store.put_model(key, SYSTEM_ENTITY, MP.ATTRSEL, model, version=version, ts=now)
        for prb in model["probes"].probes.values():
            prb.release()
        for a, old, new in changes[:64]:
            store.add_event(BehaviorEvent(
                system=s, entity=SYSTEM_ENTITY, ts=now, kind="attribute_role", score=0.0,
                severity=Severity.INFO,
                description=f"属性 {a} 的角色：{ROLE_ZH.get(old, old)} → {ROLE_ZH.get(new, new)}",
                extra={"attribute": a, "old": old, "new": new,
                       "U_t": out["ustat"].get(a, {}).get("U_t"), "U_s": out["ustat"].get(a, {}).get("U_s")},
                dedupe_key=f"attribute_role|{key}|{a}|{new}|{int(now // 3600)}"))
        return True
