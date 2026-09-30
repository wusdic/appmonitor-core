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
Cadence 1 h per tree (entity_due); a tree that learned nothing since the last
        run is skipped, and inside a tree only dirty nodes are refitted (evidence
        grew >= 10 % or >= 20 units, or version / cver / state changed, §6.20), so
        periodic cost follows the evidence that arrived, not #systems x N_max.
Inert unless config['progressive']['enabled'].
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional

from ...core.engine import Context, Engine
from ...models.schema import SYSTEM_ENTITY
from .lib import m_ptree as MP
from .lib import pbounds as PB
from .lib import pevent as EV
from .lib import pnode as PN

RATE_ATTR = "rate.ip_h"
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
            for nid, node in tree.nodes.items():
                if node.last_seen is None or not PB.is_dirty(fits.get(nid), node, now):
                    continue
                entry = self.fit_node(store, key, node, now, day, reg, ctx.config, byte_globs,
                                      outn.get(nid))
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
                 old: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
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
            rec = PB.fit_numeric(summ, now, day, n_c * pres, n_m * pres, approx,
                                 PB.unit_of(a, byte_globs), pins_for(config, store, key, a))
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

