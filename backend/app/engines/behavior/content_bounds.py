"""P06 ContentBounds (`behavior.content_bounds`) — numeric bands and hard
bounds per pattern node (docs/lib3/progressive.md §6.10, card P06).

Reads   model.ptree (node numeric target summaries and rate.ip_h digests, as P04
        maintains them), model.attr (log / approx share), model.sysprof (chosen
        arms), config progressive.content_pins / model.cpins (operator pins).
Writes  model.pbounds@(tree key, '__system__'):
          {'fmt': 1, 'version', 'updated',
           'nodes': {kind: {nid: {'status': 'fitted'|'none', 'attrs': {attr: record},
                                  'cver', 'n_c', 'fit_t'}}},
           'fit':   {kind: {nid: fit mark}}          (dirty-node bookkeeping)
           'gain':  {'bits_per_event', 'us_per_event', 'nodes_fitted', 'ms'}}
        record = lib/pbounds.fit_numeric (band90, band98, range, n_rng, cover, hard,
        tail_hi / tail_lo, qgrid, disp90 / disp_range, n_eff, n_c, approx, confidence,
        cver). A node with no numeric target gets status 'none' (P04's confirmation
        rule needs every applicable fitter to have spoken, §6.8.1).
        model.pwant['p06'] = {'targets': {kind: {nid: [attr...]}}}: numeric attributes that
        P05 found informative (role split / target) but that the node does not model,
        <= 3 per node at the 32 busiest nodes, in P05's ranking (lib/pbounds.
        request_targets). Deviation: §8 says "none by default"; pack O's real
        lattice never made the login body size a target (P05 gives it the split
        role only), so "90 % of submissions are 1-2 KB" could not be fitted.
Hygiene Every tick the numeric values of rows P03 judged violations (any typed
        p <= 1e-3, damped, or a content flag; pat.assign) go to a FIFO ledger
        (model.pbounds_state, <= 1024 rows per tree); the hard range of a node
        is read from its daily ring without them (lib/pbounds.clean_range): a
        day whose extreme is a violating value takes its next extreme from the
        exceedance reservoirs, or leaves the range with its observations.
        (2026-10-01: A3's undamped 12 KB injection login was the OA login
        node's stated maximum, "0-12 KB".)
Cadence 1 h per tree (entity_due); a tree that learned nothing since the last
        run is skipped, and inside a tree only dirty nodes are refitted (evidence
        grew >= 10 % or >= 20 units, or version / cver / state changed, §6.20), so
        periodic cost follows the evidence that arrived, not #systems x N_max.
Inert unless config['progressive']['enabled'].
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np
from typing import Any, Dict, List, Mapping, Optional

from ...core.engine import Context, Engine
from ...models.schema import SYSTEM_ENTITY
from .lib import m_ptree as MP
from .lib import pbounds as PB
from .lib import pevent as EV
from .lib import pnode as PN

RATE_ATTR = "rate.ip_h"
STATE = "model.pbounds_state"
LEDGER_MAX = 1024               # violating rows remembered per tree (FIFO, <= RING_DAYS old)
VIOL_FLAGS = frozenset({"above_range", "below_range", "injection_shape", "grammar", "length"})
NUM_TYPES = ("numeric",)
pins_for, chosen_arm, local_day, empty_model = PB.pins_for, PB.chosen_arm, PB.local_day, PB.empty_model
CPINS = PB.CPINS


class ContentBoundsEngine(Engine):
    name = "behavior.content_bounds"
    layer = "behavior"
    consumes = [MP.PTREE, MP.ATTR, MP.SYSPROF, CPINS]
    produces = [MP.PBOUNDS, MP.PWANT]
    description = "P06: numeric bands (90 %/98 %) and hard bounds with coverage per pattern node"
    period_s = 3600.0
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
        for s in store.batch_systems(EV.PAT_ASSIGN):
            self._ledger(ctx, s, now)
        fitted = 0
        stats: Dict[str, Any] = {}
        for key in keys:
            ptm = MP.get_ptree(store, key)
            if ptm is None or not self.entity_due(("p06", key), now):
                continue
            n, st = self.fit_tree(ctx, key, ptm, now)
            fitted += n
            stats[key] = st
        self.last_stats = stats
        return fitted

    # -------------------------------------------------------- ledger
    @staticmethod
    def _state(store: Any, key: str) -> Dict[str, Any]:
        st = store.get_model(key, SYSTEM_ENTITY, STATE)
        if not isinstance(st, dict):
            st = {"last": {}, "led": deque(maxlen=LEDGER_MAX), "attrs": []}
            store.put_model(key, SYSTEM_ENTITY, STATE, st)
        return st

    def _ledger(self, ctx: Context, s: str, now: float) -> int:
        """Remember the numeric values of the rows P03 judged violations
        (§6.9.3: learners do not absorb violations). P04 learns such rows a
        learning delay later and, unless P03 also damped them, at full weight,
        so they reach the daily (min, max) ring that the hard range is read
        from (pack O, A3: the 12 KB injection login on day 18 became the OA
        login node's stated maximum, 0-12 KB). A row counts as a violation
        when any typed p-value is <= 1e-3 (P03's `vtype` mask), when it was
        damped, or when it carries a content flag. Every tick, O(rows of the
        tick) over the scored batches; FIFO of LEDGER_MAX rows per tree."""
        store = ctx.store
        key = MP.tree_key(store, s)
        st = self._state(store, key)
        attrs = set(st.get("attrs") or ())
        if not attrs:
            return 0
        n = 0
        last = st["last"].get(s)
        for ts_b, asg in MP.batches_since(store, s, EV.PAT_ASSIGN, -1e18 if last is None else last, now):
            st["last"][s] = max(float(ts_b), float(st["last"].get(s, -1e18)))
            b = store.batch_at(s, EV.EVT_BATCH, ts_b)
            if b is None or asg.n != b.n or not asg.has("leaf"):
                continue
            leaf = asg.dense("leaf", float("nan"))
            cols = [a for a in attrs if b.has(a)]
            if not cols:
                continue
            # violating rows, vectorised: any typed p <= 1e-3, damped, or a content flag
            mask = np.zeros(b.n, dtype=bool)
            if asg.has("vtype"):
                vt = np.asarray(asg.dense("vtype", 0.0), dtype=np.float64)
                mask |= np.nan_to_num(vt, nan=0.0) != 0.0
            if asg.has("damp"):
                dm = np.asarray(asg.dense("damp", 1.0), dtype=np.float64)
                mask |= np.nan_to_num(dm, nan=1.0) < 1.0
            fc = asg.cols.get("flags") if asg.has("flags") else None
            if fc is not None:
                for r_, f_s in zip(fc.rows.tolist(), fc.vals.tolist()):
                    if isinstance(f_s, str) and VIOL_FLAGS & set(f_s.split(",")):
                        mask[int(r_)] = True
            mask &= np.isfinite(np.asarray(leaf, dtype=np.float64))
            if not mask.any():
                continue
            kind = int((getattr(asg, "meta", None) or {}).get("kind", getattr(b, "kind", 0)) or 0)
            for i in np.flatnonzero(mask).tolist():
                lf = float(leaf[i])
                vals = {}
                for a in cols:
                    v = b.get(a, i)
                    try:
                        x = float(v)
                    except (TypeError, ValueError):
                        continue
                    if x == x:
                        vals[a] = x
                if vals:
                    st["led"].append((kind, int(lf), local_day(float(b.ts[i]), ctx.config), vals))
                    n += 1
        if n or st["last"]:
            store.put_model(key, SYSTEM_ENTITY, STATE, st, ts=now)
        return n

    @staticmethod
    def _by_node(st: Mapping[str, Any], tree: Any, kind: int, day_now: int
                 ) -> Dict[int, List[Any]]:
        """The ledger's rows by node: a row belongs to its leaf and to every
        ancestor (P04 updates the targets along the whole path). O(rows x depth)."""
        out: Dict[int, List[Any]] = {}
        for k, lf, d, vals in st.get("led") or ():
            if k != kind or d <= day_now - PN.RING_DAYS:
                continue
            cur = tree.nodes.get(lf)
            while cur is not None:
                out.setdefault(cur.id, []).append((d, vals))
                cur = tree.nodes.get(cur.parent) if cur.parent is not None else None
        return out

    @staticmethod
    def _excl(rows: List[Any], summ: Any, a: str) -> Dict[int, List[float]]:
        """{local day: [y]} of attribute a among a node's violating rows."""
        out: Dict[int, List[float]] = {}
        for d, vals in rows:
            if a not in vals:
                continue
            try:
                y = summ.y(vals[a])
            except (TypeError, ValueError):
                continue
            out.setdefault(int(d), []).append(float(y))
        return out

    def fit_tree(self, ctx: Context, key: str, ptm: Any, now: float) -> tuple:
        store = ctx.store
        t0 = time.perf_counter()
        model = MP.get_model(store, key, MP.PBOUNDS)
        if not isinstance(model, dict):
            model = empty_model()
        reg = MP.get_registry(store, key)
        day = local_day(now, ctx.config)
        byte_globs = tuple((((ctx.config.get("progressive") or {}).get("units") or {})
                            .get("bytes")) or PB.BYTE_GLOBS)
        n_fit, gain_num, gain_den, new_ev = 0, 0.0, 0.0, 0.0
        skipped = 0
        st = self._state(store, key)
        num_attrs = sorted({a for tree in ptm.kinds.values() for nd in tree.nodes.values()
                            for a, sm in nd.targets.items() if isinstance(sm, PN.NumSummary)})
        if num_attrs != list(st.get("attrs") or []):
            st["attrs"] = num_attrs
            store.put_model(key, SYSTEM_ENTITY, STATE, st, ts=now)
        for kind, tree in ptm.kinds.items():
            root = tree.nodes.get(tree.root)
            marks = model.setdefault("tree_mark", {})
            if root is None or (not PB.tree_changed(marks, kind, root, now)
                                and model["nodes"].get(kind)):
                skipped += 1
                continue
            fits = model["fit"].setdefault(kind, {})
            outn = model["nodes"].setdefault(kind, {})
            for nid in [n for n in list(outn) if n not in tree.nodes]:     # retired nodes
                outn.pop(nid, None)
                fits.pop(nid, None)
            if root is not None and kind in model["fit"] and tree.root in fits:
                new_ev += max(0.0, PB.new_evidence(fits[tree.root], root, now))
            led = self._by_node(st, tree, kind, day)
            for nid, node in tree.nodes.items():
                if node.last_seen is None or not PB.is_dirty(fits.get(nid), node, now):
                    continue
                entry = self.fit_node(store, key, node, now, day, reg, ctx.config, byte_globs,
                                      outn.get(nid), led.get(nid))
                outn[nid] = entry
                fits[nid] = PB.fit_mark(node, now)
                n_fit += 1
                m = node.mass_at(now)
                for a, rec in entry["attrs"].items():
                    g = rec.get("gain")
                    if g is not None:
                        gain_num += m * g
                        gain_den += m
        ms = (time.perf_counter() - t0) * 1000.0
        model["last_run"] = now
        model["updated"] = now
        model["version"] = int(model.get("version", 0)) + (1 if n_fit else 0)
        model["gain"] = {"bits_per_event": gain_num / gain_den if gain_den > 0 else 0.0,
                         "us_per_event": ms * 1000.0 / new_ev if new_ev > 0 else None,
                         "nodes_fitted": n_fit, "ms": ms}
        req = PB.request_targets(store, key, ptm, now, NUM_TYPES, model.setdefault("req", {}))
        want = MP.get_model(store, key, MP.PWANT)
        if not isinstance(want, dict):
            want = {}
        sub = {"targets": req} if req else {}
        if want.get("p06") != sub:
            want["p06"] = sub
            store.put_model(key, SYSTEM_ENTITY, MP.PWANT, want, ts=now)
        store.put_model(key, SYSTEM_ENTITY, MP.PBOUNDS, model, version=model["version"], ts=now)
        return n_fit, {"fitted": n_fit, "skipped_kinds": skipped, "ms": ms,
                       "requested": sum(len(v) for kv in req.values() for v in kv.values())}

    def fit_node(self, store: Any, key: str, node: Any, now: float, day: int, reg: Any,
                 config: Mapping[str, Any], byte_globs: tuple,
                 old: Optional[Mapping[str, Any]], viol: Optional[List[Any]] = None
                 ) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {}
        node_mass = node.mass_at(now)
        n_c, n_m = node.n_c(now), node.n_m(now)
        old_attrs = (old or {}).get("attrs") or {}
        for a, summ in node.targets.items():
            if not isinstance(summ, PN.NumSummary):
                continue
            pres = min(1.0, summ.td.total(now) / node_mass) if node_mass > 0 else 0.0
            rr = reg.get(a) if reg is not None else None
            approx = float(getattr(rr, "approx_share", 0.0) or 0.0) if rr is not None else 0.0
            excl = self._excl(viol, summ, a) if viol else None
            rec = PB.fit_numeric(summ, now, day, n_c * pres, n_m * pres, approx,
                                 PB.unit_of(a, byte_globs), pins_for(config, store, key, a),
                                 excl=excl, day_of=lambda ts: local_day(ts, config))
            if rec is None:
                continue
            sysd = getattr(rr, "num", None) if rr is not None else None
            rec["gain"] = PB.bin_gain(summ, sysd)
            prev = old_attrs.get(a)
            cv = int((prev or {}).get("cver", 0))
            rec["cver"] = cv + 1 if PB.material_change(prev, rec) and prev else cv
            attrs[a] = rec
        if node.rate_iph is not None:
            rec = PB.fit_digest(node.rate_iph, now, n=node.rate_iph.total(now))
            if rec is not None:
                rec["n_c"] = n_c
                attrs[RATE_ATTR] = rec
        return {"status": "fitted" if attrs else "none", "attrs": attrs, "n_c": float(n_c),
                "fit_t": float(now), "state": node.state,
                "cver": max([r.get("cver", 0) for r in attrs.values()] or [0])}

