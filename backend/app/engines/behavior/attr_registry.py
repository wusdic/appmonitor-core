"""P02 AttributeRegistryEngine (`behavior.attr_registry`) — the open attribute space.

docs/lib3/progressive.md §5.3, §5.4, §6.3, card P02. Library 3 (behaviour).

Every attribute name an event carries is registered on first sight (current
tick; `attribute_new` INFO), typed from decayed evidence, summarised with
bounded sketches and given a generalisation-hierarchy model (lib/pregistry).
No code lists which attributes exist (PPC-4, requirement S16).

Per tick, per system (registry per tree key: a system or its family):
  * registration of every column of the tick's evt.batch / evt.ctx (kind txn)
    and evt.win (kind win);
  * statistics from the LEARNED rows of the batches of tick t' <= t - D (the
    poisoning rule of §6.9.3: an attacker's first hour must not move bin edges
    or value groups), at most A_ev = 32 attribute updates per learned event:
    every attribute P05 keeps (split / target roles) plus each other attribute
    column with inclusion probability q = (A_ev - n_kept) / n_rest (mass 1/q,
    Horvitz-Thompson);
  * hourly: types and statistics (entropy, stability, approx share, log flag);
  * daily (and once after the first hour of data): hierarchy refresh (numeric
    bins, ordinal medians, set templates; replaced only above the JSD
    threshold) and schema-change detection (`attribute_gone` INFO when an
    attribute's coverage over the previous full NORMAL local day fell below
    5 % of its H_l coverage; P01's evt.ctx meta carries the normal-day flag).

Writes model.attr@(key, '__system__') (the AttrRegistry object).
Inert unless config['progressive']['enabled'].
"""
from __future__ import annotations

import math
import zlib
from typing import Any, Dict, List, Mapping, Optional, Set

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import SYSTEM_ENTITY, BehaviorEvent, Severity
from .lib import m_ptree as MP
from .lib import pevent as EV
from .lib import pselect as SEL

A_EV = 32
REST_MIN = 4               # attributes of the rest updated per event even when A_ev is used up
ROW_CAP = 48               # rows per (attribute, batch); beyond, a uniform HT subsample
HOUR = 3600.0
FIRST_REFRESH_S = 3600.0


class AttributeRegistryEngine(Engine):
    name = "behavior.attr_registry"
    layer = "behavior"
    consumes = [EV.EVT_BATCH, EV.EVT_CTX, EV.EVT_WIN, MP.ATTRSEL]
    produces = [MP.ATTR, "event.attribute_new", "event.attribute_gone"]
    description = "P02: open attribute registry (schema inference, statistics, hierarchies, schema change)"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.last_stats: Dict[str, Any] = {}

    # ---------------------------------------------------------------- run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        if not EV.enabled(ctx.config):
            return 0
        store = ctx.store
        now = float(ctx.now)
        D = EV.learn_delay_s(ctx.window_s, ctx.config)
        systems = sorted(set(store.batch_systems(EV.EVT_BATCH)) | set(store.batch_systems(EV.EVT_WIN)))
        n_obs = n_new = 0
        touched: Dict[str, str] = {}
        for s in systems:
            key = MP.tree_key(store, s)
            reg = MP.ensure_registry(store, key, ctx.config)
            touched.setdefault(key, s)
            # 1. registration (current tick)
            for name, kind in ((EV.EVT_BATCH, EV.KIND_TXN), (EV.EVT_CTX, EV.KIND_TXN),
                               (EV.EVT_WIN, EV.KIND_WIN)):
                b = store.batch_at(s, name, now)
                if b is None:
                    continue
                for nm in b.names():
                    st = reg.register(nm, now, kind)
                    if st in ("new", "revived"):
                        n_new += 1
                        self._event(store, s, now, "attribute_new", nm,
                                    f"新属性 {nm}（{EV.KIND_NAMES.get(kind)}）已登记" if st == "new"
                                    else f"属性 {nm} 重新出现", {"kind": kind, "status": st})
            # 2. statistics from learned rows of t - D
            meta = reg.__dict__.setdefault("_p02", {"last": {}, "hour": None, "refresh0": None})
            sel = store.get_model(key, SYSTEM_ENTITY, MP.ATTRSEL) or {}
            roles = sel.get("roles") or {} if isinstance(sel, Mapping) else {}
            for name, kind, ctxname in ((EV.EVT_BATCH, EV.KIND_TXN, EV.EVT_CTX),
                                        (EV.EVT_WIN, EV.KIND_WIN, None)):
                lastk = (s, name)
                for ts_b, b in MP.learnable_batches(store, s, name, meta["last"].get(lastk), now, D):
                    meta["last"][lastk] = ts_b
                    cb = store.batch_at(s, ctxname, ts_b) if ctxname else None
                    n_obs += self._observe(reg, b, cb, ts_b, roles, kind, s)
            # normal-day flag of the previous day (P01)
            cbn = store.batch_at(s, EV.EVT_CTX, now)
            if cbn is not None:
                meta["normal_prev"] = bool(cbn.meta.get("normal_prev_day", True))
        # 3. hourly / daily maintenance per registry
        for key, s in touched.items():
            reg = MP.get_registry(store, key)
            meta = reg.__dict__["_p02"]
            h = int(now // HOUR)
            stale_scale = False
            if meta["hour"] != h:
                meta["hour"] = h
                reg.update_types(now)
                reg.update_stats(now)
                # lib/phier bins a value on the scale of hier['log'] while the edges
                # were computed on hier['edges_log']: when update_stats flips the log
                # flag, the edges must be recomputed at once (else every value of
                # the attribute falls into one bin until the next daily refresh)
                stale_scale = any("edges" in r.hier and bool(r.hier.get("log", False))
                                  != bool(r.hier.get("edges_log", False)) for r in reg.records.values())
            if meta["refresh0"] is None and reg.cur_day is not None:
                meta["refresh0"] = now
            first = (reg.last_refresh is None and meta["refresh0"] is not None
                     and now - meta["refresh0"] >= FIRST_REFRESH_S)
            # a newly typed numeric attribute gets its bins at the next hour, not the next day
            if not first and meta["hour"] == h and meta.get("binned_hour") != h:
                meta["binned_hour"] = h
                first = any(r.type in ("numeric", "time") and r.num is not None and "edges" not in r.hier
                            for r in reg.records.values())
            if first or stale_scale or (reg.last_refresh is not None and now - reg.last_refresh >= 86400.0):
                reg.update_types(now)
                reg.refresh_hierarchies(now, force=first or stale_scale)
            for nm in reg.check_gone(now, bool(meta.get("normal_prev", True))):
                self._event(store, s, now, "attribute_gone", nm,
                            f"属性 {nm} 在一个正常工作日内几乎消失（模式结构变化，而非行为变化）",
                            {"coverage_l": reg.coverage(nm, now, 2)})
        self.last_stats = {"observed": n_obs, "new": n_new}
        return n_obs + n_new

    # --------------------------------------------------------- statistics
    def _observe(self, reg: Any, b: EV.EventBatch, cb: Optional[EV.EventBatch], ts_b: float,
                 roles: Mapping[str, str], kind: int, s: str) -> int:
        rr = b.learned_rows()
        if rr.size == 0:
            return 0
        mass_all = b.mass()
        reg.observe_events(kind, ts_b, float(mass_all[rr].sum()))
        sel = np.zeros(b.n, dtype=bool)
        sel[rr] = True
        policy = (b.meta or {}).get("policy", {})
        cols = dict(b.cols)
        if cb is not None and cb.n == b.n:
            for nm, c in cb.cols.items():
                cols.setdefault(nm, c)
        kept = [nm for nm in cols if roles.get(nm) in ("split", "target", "invariant")]
        kept_set = set(kept)
        rest = [nm for nm in cols if nm not in kept_set]
        n_kept = len(kept)
        # the rest: a uniform subset with q = (A_ev - n_kept) / n_rest, and at least
        # REST_MIN of them per event so that a dropped attribute keeps its statistics
        # (its re-probe and a role change need them)
        q = 1.0 if len(rest) + n_kept <= A_EV else \
            min(1.0, max(REST_MIN, A_EV - n_kept) / max(1, len(rest)))
        used = 0
        for nm in kept + rest:
            c = cols[nm]
            m = sel[c.rows]
            if not m.any():
                continue
            r = c.rows[m]
            vals = c.vals[m]
            w = mass_all[r]
            qa = q if nm not in kept_set else 1.0
            if r.size * qa > ROW_CAP:
                qa = ROW_CAP / r.size
            if qa < 1.0:
                if qa <= 0.0:
                    continue
                seed = zlib.crc32(f"{s}|{ts_b!r}|{nm}".encode("utf-8", "surrogatepass"))
                u = np.random.default_rng(seed).random(r.size)          # reproducible per batch
                keep = u < qa
                if not keep.any():
                    continue
                r, vals, w = r[keep], vals[keep], w[keep] / qa
            used += reg.observe(nm, list(vals), ts_b, w, None, (b.flags[r] & 1) > 0, kind,
                                policy.get(nm))
        return used

    # ------------------------------------------------------------- events
    @staticmethod
    def _event(store: Any, s: str, now: float, kind: str, attr: str, desc: str,
               extra: Dict[str, Any]) -> None:
        store.add_event(BehaviorEvent(
            system=s, entity=SYSTEM_ENTITY, ts=now, kind=kind, score=0.0, severity=Severity.INFO,
            description=desc, extra=dict(extra, attribute=attr),
            dedupe_key=f"{kind}|{s}|{attr}|{int(now // 86400)}"))
