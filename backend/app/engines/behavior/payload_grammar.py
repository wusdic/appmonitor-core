"""P07 PayloadGrammar (`behavior.payload_grammar`) — key sets, value grammars
and closed value sets per pattern node (docs/lib3/progressive.md §6.11, card P07).

Reads   model.ptree (node text / set / categorical target summaries), model.attr
        (system shape distributions for the gain), model.sysprof (arm 'p07' on/off),
        config progressive.content_pins / model.cpins (operator length pins).
Writes  model.pgrammar@(tree key, '__system__'):
          {'fmt': 1, 'version', 'updated', 'applicable': bool,
           'nodes': {kind: {nid: {'status': 'fitted'|'none'|'off', 'attrs': {attr: record},
                                  'cver', 'n_c', 'fit_t'}}},
           'fit': {kind: {nid: fit mark}}, 'gain': {...}}
        record (lib/pgrammar): text -> grammar (regex), charset, len, len_cover, c_g, U_s,
        closed / U (closed value set), top, confidence; set -> required, optional,
        presence, p_new_key, p_missing; categorical -> closed / U.
        model.pwant['p07'] = {'targets': {kind: {nid: [attr...]}}}: key sets and payload
        text attributes P05 found informative that the node does not model (<= 3 per
        node, 32 busiest nodes; lib/pbounds.request_targets), so key sets and value
        grammars exist where the requirement needs them.
Adaptive value capacity (round 3, lib/pgrammar.adapt_values): a text target's
        exact-value sketch (pnode.TEXT_VALUES_K = 16) is doubled (<= 64) when the
        values P07 found tracked at its fits stopped growing while the sketch kept
        evicting (a finite population larger than the sketch: 销售部's 20
        usernames), and the closed set is then judged on the arrivals since the
        growth; random tokens never grow it. The state is kept per node in
        'vstate' {attr: {'k', 'seg', 'seen' (<= 128 hashes) | 'open', 'ck', 'new'}}.
Closure (round 4, lib/pgrammar._closed): a value set is closed when its
        Good-Turing unseen mass U is below its rarest member's share (every
        member repeated, nothing evicted), not at fixed U / n thresholds:
        finance's three users close once each logged in twice (round 3: U <=
        0.02 at n_c >= 20, ~day 16-21 of pack O).
Cadence 1 h per tree, dirty nodes only (§6.20).
Nothing is pre-set: "后边内容不超过 10 个字符" is an output only when observed
lengths reach 10 (or an operator pins it; pins only widen).
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
from .lib import pgrammar as PG
from .lib import pnode as PN

ARM = ("p07", "P07", "grammar", "payload_grammar")
TEXT_TYPES = ("set", "text")


class PayloadGrammarEngine(Engine):
    name = "behavior.payload_grammar"
    layer = "behavior"
    consumes = [MP.PTREE, MP.ATTR, MP.SYSPROF, PB.CPINS]
    produces = [MP.PGRAMMAR, MP.PWANT]
    description = "P07: required keys, value grammars and closed value sets per pattern node"
    period_s = 3600.0
    interval = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.last_stats: Dict[str, Any] = {}

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
            if ptm is None or not self.entity_due(("p07", key), now):
                continue
            n, st = self.fit_tree(ctx, key, ptm, now)
            fitted += n
            stats[key] = st
        self.last_stats = stats
        return fitted

    def fit_tree(self, ctx: Context, key: str, ptm: Any, now: float) -> tuple:
        store = ctx.store
        t0 = time.perf_counter()
        model = MP.get_model(store, key, MP.PGRAMMAR)
        if not isinstance(model, dict):
            model = PB.empty_model()
        on = PB.chosen_arm(store, key, ARM) != "off"
        switched = model.get("applicable", True) != bool(on)
        model["applicable"] = bool(on)
        reg = MP.get_registry(store, key)
        pc = EV.pconfig(ctx.config)
        dflt = pc.get("defaults") or {}
        closed_n = float(dflt.get("closed_n", PG.CLOSED_N))
        closed_u = float(dflt.get("closed_u", PG.CLOSED_U))
        n_fit, gain_num, gain_den, new_ev, skipped = 0, 0.0, 0.0, 0.0, 0
        for kind, tree in ptm.kinds.items():
            root = tree.nodes.get(tree.root)
            fits = model["fit"].setdefault(kind, {})
            outn = model["nodes"].setdefault(kind, {})
            for nid in [n for n in list(outn) if n not in tree.nodes]:
                outn.pop(nid, None)
                fits.pop(nid, None)
            marks = model.setdefault("tree_mark", {})
            if root is None or (not PB.tree_changed(marks, kind, root, now) and outn
                                and not switched):
                skipped += 1
                continue
            if tree.root in fits:
                new_ev += max(0.0, PB.new_evidence(fits[tree.root], root, now))
            for nid, node in tree.nodes.items():
                if node.last_seen is None:
                    continue
                prev = outn.get(nid)
                was_off = prev is not None and prev.get("status") == "off"
                if on and not was_off and not PB.is_dirty(fits.get(nid), node, now):
                    continue
                if not on:
                    if not was_off:
                        outn[nid] = {"status": "off", "attrs": {}, "fit_t": now, "cver": 0}
                    continue
                entry = self.fit_node(store, key, node, now, reg, ctx.config, closed_n, prev,
                                      closed_u, lambda nd, a, tree=tree, outn=outn:
                                      _ancestor_grammar(tree, outn, nd, a),
                                      lambda nd, a, tree=tree, outn=outn:
                                      _ancestor_capacity(tree, outn, nd, a))
                outn[nid] = entry
                fits[nid] = PB.fit_mark(node, now)
                n_fit += 1
                m = node.mass_at(now)
                for rec in entry["attrs"].values():
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
        req = PB.request_targets(store, key, ptm, now, TEXT_TYPES, model.setdefault("req", {})) \
            if on else {}
        want = MP.get_model(store, key, MP.PWANT)
        if not isinstance(want, dict):
            want = {}
        sub = {"targets": req} if req else {}
        if want.get("p07") != sub:
            want["p07"] = sub
            store.put_model(key, SYSTEM_ENTITY, MP.PWANT, want, ts=now)
        store.put_model(key, SYSTEM_ENTITY, MP.PGRAMMAR, model, version=model["version"], ts=now)
        return n_fit, {"fitted": n_fit, "skipped_kinds": skipped, "ms": ms, "on": on,
                       "requested": sum(len(v) for kv in req.values() for v in kv.values())}

    def fit_node(self, store: Any, key: str, node: Any, now: float, reg: Any,
                 config: Mapping[str, Any], closed_n: float,
                 old: Optional[Mapping[str, Any]], closed_u: float = PG.CLOSED_U,
                 inherit: Any = None, capacity_of: Any = None) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {}
        old_attrs = (old or {}).get("attrs") or {}
        old_vstate = (old or {}).get("vstate") or {}
        vstate: Dict[str, Any] = {}
        for a, summ in node.targets.items():
            rr = reg.get(a) if reg is not None else None
            old_st = old_vstate.get(a) or {}
            if isinstance(summ, PN.TextSummary):
                # adaptive value capacity (lib/pgrammar.adapt_values): a finite
                # population of values that outgrew the exact-value sketch grows it
                # (bounded), the closure is then judged on the arrivals since
                vcap = PG.adapt_values(summ, now, old_st or None,
                                       inherit_k=capacity_of(node, a) if capacity_of else None)
                if vcap:
                    vstate[a] = vcap
                rec = PG.fit_text(summ, now, closed_n=closed_n, closed_u=closed_u,
                                  pin=PB.pins_for(config, store, key, a), vcap=vcap)
                if rec is not None:
                    rec["gain"] = PG.shape_gain(summ.shapes, getattr(rr, "top", None), now)
                elif inherit is not None:
                    rec = PG.inherit_text(inherit(node, a), summ, now)
            elif isinstance(summ, PN.SetSummary):
                rec = PG.fit_set(summ, now)
            elif isinstance(summ, PN.CatSummary):
                rec = PG.fit_cat(summ, now, closed_n=closed_n, closed_u=closed_u)
                if rec is not None and "closed" not in rec:
                    rec = None                    # nothing to constrain beyond P03's predictive
            else:
                rec = None
            if rec is None:
                continue
            prev = old_attrs.get(a)
            cv = int((prev or {}).get("cver", 0))
            rec["cver"] = cv + 1 if prev and PG.material_change(prev, rec) else cv
            attrs[a] = rec
        out = {"status": "fitted" if attrs else "none", "attrs": attrs,
               "n_c": float(node.n_c(now)), "fit_t": float(now), "state": node.state,
               "cver": max([r.get("cver", 0) for r in attrs.values()] or [0])}
        if vstate:
            out["vstate"] = vstate                # adaptive value capacity (lib/pgrammar.adapt_values)
        return out


def _ancestor_grammar(tree: Any, outn: Mapping[int, Any], node: Any, a: str) -> Optional[Dict[str, Any]]:
    """The nearest ancestor's own (not inherited) grammar record of a."""
    cur = node
    while cur.parent is not None:
        cur = tree.nodes.get(cur.parent)
        if cur is None:
            return None
        rec = ((outn.get(cur.id) or {}).get("attrs") or {}).get(a)
        if rec and rec.get("kind") == "text" and rec.get("grammar") and not rec.get("inherited"):
            return dict(rec, _from=int(cur.id))
    return None


def _ancestor_capacity(tree: Any, outn: Mapping[int, Any], node: Any, a: str) -> Optional[int]:
    """The largest value-sketch capacity P07 gave attribute a at an ancestor
    of the node (lib/pgrammar.adapt_values inherit_k), None without one."""
    best = None
    cur = node
    while cur.parent is not None:
        cur = tree.nodes.get(cur.parent)
        if cur is None:
            break
        st = (((outn.get(cur.id) or {}).get("vstate") or {}).get(a)) or {}
        k = st.get("k")
        if k is not None and (best is None or int(k) > best):
            best = int(k)
    return best
