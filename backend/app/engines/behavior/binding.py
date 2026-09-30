"""P08 Binding (`behavior.binding`) — approximate functional dependencies and
value bindings per pattern node (docs/lib3/progressive.md §6.12, card P08).

Every tick (cheap, bounded by the learned rows of tick t - D):
    * the learned, trusted rows (quarantined IPs excluded, §6.9.3) feed a
      stratified screening probe (lib/pfd.ProbeReservoir: <= S_max strata x
      R_k rows, candidate columns only);
    * rows that reach a node holding a fitted binding update that node's
      rebinding tracker (last 5 values of each bound source, candidate new
      values with their normal days, first / last seen).
Hourly per tree (entity_due), trees whose arm 'p08' is not 'off':
    * screening (lib/pfd.screen) -> <= Q_pairs pairs (X, Y) and the nodes where
      Y (reverse: X) is a target, <= 8 nodes per pair and <= 32 pair-nodes per
      tree, written to model.pwant['pairs'] for P04, which keeps a
      pnode.PairSketch at node.pairs[(X, Y)] (X = 'net.src' or 'net.src@<level>',
      x = gen(attr, level, value));
    * fit of every dirty node's pair sketches (lib/pfd.fit_pair) with the
      per-source segment baselines of accepted rebindings; rebinding
      (a renamed account, trusted, >= 5 events over >= 2 normal days, old value
      absent from the last 5) moves y*_x, restarts x's confidence segment and
      emits `binding_changed` (INFO).
Writes  model.pbind@(tree key, '__system__'):
          {'fmt': 1, 'version', 'updated', 'applicable',
           'nodes': {kind: {nid: {'status', 'pairs': {'X->Y': record}, 'fit_t', 'n_c'}}},
           'fit': {kind: {nid: fit mark}}, 'screen': [pair specs], 'gain': {...}}
        record = lib/pfd.fit_pair + {'x', 'y', 'dir': 'fwd'|'rev'}; table entries carry
        first / last seen from the tracker.
        model.pwant['pairs'] = {'fmt': 1, 'updated', 'by_kind': {kind: {nid: [[X, Y], ...]}},
                                'specs': [...]}
        model.pbind_state@(tree key, '__system__') = private bookkeeping (probe,
        trackers, segment baselines, ingestion marks).
Inert unless config['progressive']['enabled'].
"""
from __future__ import annotations

import datetime as _dt
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ...core.engine import Context, Engine
from ...models.schema import BehaviorEvent, Severity, SYSTEM_ENTITY
from .lib import m_governor as MG
from .lib import m_ptree as MP
from .lib import pbounds as PB
from .lib import pevent as EV
from .lib import pfd as FD
from .lib import timebins as TB
from .lib.combine import seeded_uniform

ARM = ("p08", "P08", "bindings", "binding")
WHO_ARM = ("who", "who_granularity")
STATE = "model.pbind_state"
PCAL = "model.pcal"
TRUST_MIN = 0.5
P_NODES = 8                   # nodes per pair
SCREEN_FRESH = 0.25          # re-screen when >= 25 % new probe offers (vs rows kept)
SCREEN_MIN_AGE = 6 * 3600.0   # ... at most every 6 h once pairs exist
SCREEN_MAX_AGE = 86400.0      # ... and at least daily
Y_MAX = 32                    # payload candidates screened per tree
PAIR_NODES = 32               # pair-nodes per tree (§7.2 memory row)
Y_EXCLUDE_NS = ("ctx", "ev", "net", "m", "rate")
X_ATTRS = ("client.stack", "sess.key")
Y_EXCLUDE = ("http.route", "http.path", "http.host", "http.method", "hdr.referer")   # routing, not content
WHO_LEVELS = {"ip": (0,), "grp": (3,), "prefix": (1,), "reg": (), "none": ()}
DEFAULT_WHO_LEVELS = (0, 1, 3)
STRATUM_KEYS = ("http.route", "tls.sni", "dns.qname", "net.dst")


def _stratum(get) -> str:
    for a in STRATUM_KEYS:
        v = get(a)
        if v is not EV.ABSENT and v is not None:
            return f"{a}={v}"
    v = get("ev.ch")
    return "ch=" + (str(v) if v is not EV.ABSENT else "?")


def _card(reg: Any, a: str) -> float:
    rec = reg.get(a) if reg is not None else None
    if rec is None:
        return 0.0
    try:
        return float(rec.card_estimate())
    except Exception:
        return 0.0


def pair_key(x: str, y: str) -> str:
    return f"{x}->{y}"


class BindingEngine(Engine):
    name = "behavior.binding"
    layer = "behavior"
    consumes = [EV.EVT_BATCH, EV.EVT_CTX, MP.PTREE, MP.ATTR, MP.ATTRSEL, MP.SYSPROF, PCAL,
                "behavior.quarantine"]
    produces = [MP.PBIND, MP.PWANT, STATE, "binding_changed"]
    description = "P08: approximate functional dependencies / value bindings (IP -> username, ...)"
    interval = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.fit_period_s = float(params.get("fit_period_s", 3600.0))
        self.last_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _state(store: Any, key: str) -> Dict[str, Any]:
        st = store.get_model(key, SYSTEM_ENTITY, STATE)
        if not isinstance(st, dict):
            st = {"probe": FD.ProbeReservoir(), "trackers": {}, "seg": {}, "last": {},
                  "pairs_at": {}}
            store.put_model(key, SYSTEM_ENTITY, STATE, st)
        return st

    @staticmethod
    def _y_candidates(store: Any, key: str, reg: Any) -> set:
        sel = MP.get_model(store, key, MP.ATTRSEL)
        names: Iterable[str] = ()
        tsys = (sel or {}).get("targets_sys") if isinstance(sel, Mapping) else None
        roles = (sel or {}).get("roles") if isinstance(sel, Mapping) else None
        if (isinstance(tsys, Mapping) and tsys) or (isinstance(roles, Mapping) and roles):
            # every attribute P05 found informative (target or split role): a
            # bound username is usually a split candidate (it separates
            # departments), not a system target; P05's target list alone
            # rotates and lost it on pack O's real lattice
            names = [a for v in (tsys or {}).values() for a in (v or [])]
            names += [a for a, r in (roles or {}).items() if r in ("split", "target")]
        elif reg is not None:
            names = reg.names()
        out = set()
        for a in names:
            rec = reg.get(a) if reg is not None else None
            if rec is None:
                continue
            if getattr(rec, "type", None) not in ("categorical", "text"):
                continue
            if getattr(rec, "policy", "clear") == "shape" or getattr(rec, "state", "active") == "gone":
                continue
            if a.split(".", 1)[0] in Y_EXCLUDE_NS or a in X_ATTRS or a in Y_EXCLUDE:
                continue
            out.add(a)
        if len(out) > Y_MAX and reg is not None:
            # bounded screening width: the Y_MAX best-covered candidates
            # (P05's target list, when present, is already bounded)
            cov = {a: float(reg.coverage(a)) for a in out}
            out = set(sorted(out, key=lambda a: (-cov[a], a))[:Y_MAX])
        return out

    @staticmethod
    def _who_levels(store: Any, key: str) -> Tuple[int, ...]:
        """X levels of the source address screened for bindings: the IP itself
        always (a binding "192.168.1.21 -> username=jack" is about one address
        whatever level P12 chose to describe a pattern's WHO at), plus the
        levels of the chosen who arm. Measured on pack O: P12 chose `prefix`
        for OA, so only /24 -> username was screened and no OA login binding
        was ever fitted (PG1 bindings 0 / 6); `none` (the portal) still
        screens nothing."""
        arm = PB.chosen_arm(store, key, WHO_ARM, default="")
        lv = WHO_LEVELS.get(arm, DEFAULT_WHO_LEVELS)
        if arm == "none":
            return lv
        return tuple(dict.fromkeys((0,) + tuple(lv)))

    # ----------------------------------------------------------------- run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        if not EV.enabled(ctx.config):
            return 0
        store = ctx.store
        now = float(ctx.now)
        D = EV.learn_delay_s(ctx.window_s, ctx.config)
        tz = ctx.config.get("tz") or TB.DEFAULT_TZ
        cal = TB.parse_calendar(ctx.config.get("calendar"))
        ingested = 0
        keys = set()
        qcache: Dict[Tuple[str, str], bool] = {}
        for s in store.batch_systems(EV.EVT_BATCH):
            key = MP.tree_key(store, s)
            keys.add(key)
            if PB.chosen_arm(store, key, ARM) == "off":
                continue
            st = self._state(store, key)
            batches = MP.learnable_batches(store, s, EV.EVT_BATCH, st["last"].get(s), now, D)
            if not batches:
                continue
            reg = MP.get_registry(store, key)
            ycand = self._y_candidates(store, key, reg)
            ptm = MP.get_ptree(store, key)
            hier = None
            if st["pairs_at"] and ptm is not None:
                hier = MP.hierarchies(store, key, ctx.config, reg)
            pcal = store.get_model(s, SYSTEM_ENTITY, PCAL)
            normal = (pcal or {}).get("normal", {}) if isinstance(pcal, Mapping) else {}
            for ts_b, b in batches:
                cb = store.batch_at(s, EV.EVT_CTX, ts_b)
                ingested += self._ingest(store, s, key, b, cb, st, ycand, ptm, hier, tz, cal,
                                         normal, qcache)
                st["last"][s] = max(float(ts_b), float(st["last"].get(s, -1e18)))
        for key in sorted(keys | {MP.tree_key(store, s) for s in store.systems()}):
            ptm = MP.get_ptree(store, key)
            if ptm is None or PB.chosen_arm(store, key, ARM) == "off":
                continue
            if not self.entity_due(("p08", key), now, self.fit_period_s):
                continue
            self.fit_tree(ctx, key, ptm, now)
        self.last_stats["ingested"] = ingested
        return ingested

    # -------------------------------------------------------------- ingest
    def _ingest(self, store: Any, s: str, key: str, b: Any, cb: Any, st: Dict[str, Any],
                ycand: set, ptm: Any, hier: Any, tz: str, cal: Any, normal: Mapping[int, bool],
                qcache: Dict[Tuple[str, str], bool]) -> int:
        probe: FD.ProbeReservoir = st["probe"]
        rows = b.learned_rows()
        cols = [a for a in b.cols if a in ycand]
        pairs_at = st["pairs_at"].get(b.kind) or {}
        tree = ptm.tree(b.kind, b.t1, create=False) if (ptm is not None and pairs_at) else None
        need_route = set(y for prs in pairs_at.values() for (_, y) in prs) | \
            set(x for prs in pairs_at.values() for (x, _) in prs)
        mass = b.mass()
        dmemo: Dict[int, int] = {}
        n = 0
        dense = {a: b.dense(a) for a in list(cols) + [a for a in X_ATTRS if a in b.cols]}
        for i in rows:
            i = int(i)
            ip = b.ip_of(i)
            q = qcache.get((s, ip))
            if q is None:
                # held / untrusted sources teach nothing (§6.9.3): quarantined, or
                # B28 trust below TRUST_MIN (no trust row = trusted, bounded mode)
                tr_v = MG.trust(store, s, ip)
                q = qcache[(s, ip)] = bool(MG.is_quarantined(store, s, ip)
                                           or (tr_v == tr_v and tr_v < TRUST_MIN))
            if q:
                continue

            def get(a: str, i: int = i) -> Any:
                if a in b.cols:
                    return b.get(a, i)
                if cb is not None and a in cb.cols:
                    return cb.get(a, i)
                if a == "net.src":
                    return b.ip_of(i)
                return EV.ABSENT
            row = {"net.src": ip}
            for a, arr in dense.items():
                v = arr[i]
                if v is not EV.ABSENT:
                    row[a] = v
            ts = float(b.ts[i])
            if len(row) > 1:
                probe.offer(_stratum(get), row, float(mass[i]), ts,
                            seeded_uniform("p08", s, ts, int(b.rid[i])))
                n += 1
            if tree is None or not (need_route & set(row)):
                continue
            path = tree.route(get, hier)
            mk = int(ts // 60)
            day = dmemo.get(mk)
            if day is None:
                day = dmemo[mk] = TB.local_datetime(mk * 60.0, tz).date().toordinal()
            is_normal = bool(normal.get(day, _dt.date.fromordinal(day) not in cal.holidays))
            for nid in path:
                for (X, Y) in pairs_at.get(nid, ()):
                    xa, xl = FD.parse_x(X)
                    xv, yv = get(xa), get(Y)
                    if xv is EV.ABSENT or yv is EV.ABSENT:
                        continue
                    xg = hier.gen(xa, xl, xv) if hier is not None else xv
                    if xg is None:
                        continue
                    tk = (b.kind, nid, pair_key(X, Y))
                    tr = st["trackers"].get(tk)
                    if tr is None:
                        tr = st["trackers"][tk] = FD.RebindTracker()
                    bound = st.get("bound", {}).get(tk, {}).get(str(xg))
                    tr.observe(str(xg), yv, ts, day, bound, is_normal)
        return n

    # ----------------------------------------------------------------- fit
    def fit_tree(self, ctx: Context, key: str, ptm: Any, now: float) -> int:
        store = ctx.store
        t0 = time.perf_counter()
        st = self._state(store, key)
        model = MP.get_model(store, key, MP.PBIND)
        if not isinstance(model, dict):
            model = PB.empty_model()
            model["screen"] = None
        model["applicable"] = True
        reg = MP.get_registry(store, key)
        hier = MP.hierarchies(store, key, ctx.config, reg)
        # ---- screening -> pwant.pairs
        probe: FD.ProbeReservoir = st["probe"]
        fresh = probe.offered - int(st.get("screened_at", -1))
        age = now - float(st.get("screened_t", -1e18))
        if not model.get("screen") or age >= SCREEN_MAX_AGE \
                or (age >= SCREEN_MIN_AGE and fresh >= max(1, SCREEN_FRESH * probe.n_rows())):
            # hourly while nothing is found; afterwards when the probe changed
            # materially and at most every SCREEN_MIN_AGE (at least daily)
            rows, w, strata = probe.rows(now, with_strata=True)
            xc = [("net.src", l) for l in self._who_levels(store, key)] + [(a, 0) for a in X_ATTRS]
            specs = FD.screen(rows, w, xc, sorted(self._y_candidates(store, key, reg)),
                              lambda a, l, v: hier.gen(a, l, v), absent=None, strata=strata,
                              card=lambda a: _card(reg, a))
            model["screen"] = [{k: v for k, v in d.items()} for d in specs]
            st["screened_at"] = probe.offered
            st["screened_t"] = now
        specs = model["screen"]
        by_kind: Dict[int, Dict[int, List[List[str]]]] = {}
        budget = PAIR_NODES
        for kind, tree in ptm.kinds.items():
            if kind != EV.KIND_TXN:
                continue
            for d in specs:
                payload = d["y"] if d["dir"] == "fwd" else d["x"]
                nodes = [nd for nd in tree.nodes.values() if payload in nd.targets and not nd.is_exc]
                if not nodes:
                    nodes = [tree.nodes[tree.root]]
                nodes.sort(key=lambda nd: -nd.mass_at(now))
                for nd in nodes[:P_NODES]:
                    if budget <= 0:
                        break
                    by_kind.setdefault(kind, {}).setdefault(nd.id, []).append([d["x"], d["y"]])
                    budget -= 1
        want = MP.get_model(store, key, MP.PWANT)
        if not isinstance(want, dict):
            want = {}
        want["pairs"] = {"fmt": 1, "updated": now, "by_kind": by_kind,
                         "specs": [{"x": d["x"], "y": d["y"], "dir": d["dir"]} for d in specs]}
        store.put_model(key, SYSTEM_ENTITY, MP.PWANT, want, ts=now)
        # ---- fits
        n_fit, gnum, gden = 0, 0.0, 0.0
        pairs_at: Dict[int, Dict[int, List[Tuple[str, str]]]] = {}
        bound_map: Dict[Tuple[int, int, str], Dict[str, str]] = {}
        dirs = {(d["x"], d["y"]): d["dir"] for d in specs}
        for kind, tree in ptm.kinds.items():
            fits = model["fit"].setdefault(kind, {})
            outn = model["nodes"].setdefault(kind, {})
            for nid in [n for n in list(outn) if n not in tree.nodes]:
                outn.pop(nid, None)
                fits.pop(nid, None)
            for nid, node in tree.nodes.items():
                if not node.pairs:
                    continue
                prev = outn.get(nid)
                dirty = PB.is_dirty(fits.get(nid), node, now)
                entry = prev if (prev and not dirty) else {"status": "none", "pairs": {}}
                if dirty:
                    entry = {"status": "none", "pairs": {}, "fit_t": now,
                             "n_c": float(node.n_c(now)), "state": node.state}
                    for (X, Y), ps in node.pairs.items():
                        pk = pair_key(X, Y)
                        segs = st["seg"].setdefault((kind, nid, pk), {})
                        rec = FD.fit_pair(ps, now, segs)
                        if rec is None:
                            continue
                        rec.update({"x": X, "y": Y,
                                    "dir": dirs.get((X, Y), "rev" if "net.src" in Y else "fwd")})
                        rec = self._rebind(ctx, key, kind, node, X, Y, ps, rec, st, now, segs)
                        tr = st["trackers"].get((kind, nid, pk))
                        if tr is not None:
                            for x, ent in rec["table"].items():
                                fl = tr.seen(x)
                                if fl is not None:
                                    ent["first"], ent["last"] = fl
                        old = ((prev or {}).get("pairs") or {}).get(pk)
                        cv = int((old or {}).get("cver", 0))
                        rec["cver"] = cv + 1 if old and FD.material_change(old, rec) else cv
                        entry["pairs"][pk] = rec
                        gnum += node.mass_at(now) * rec["gain"]
                        gden += node.mass_at(now)
                    entry["status"] = "fitted" if entry["pairs"] else "none"
                    outn[nid] = entry
                    fits[nid] = PB.fit_mark(node, now)
                    n_fit += 1
                for pk, rec in (entry.get("pairs") or {}).items():
                    # sticky: a source stays bound to its last bound value while it
                    # is tracked (a rename first lowers its purity; rebinding needs
                    # the old binding to recognise the new value)
                    prev_b = (st.get("bound") or {}).get((kind, nid, pk)) or {}
                    bm = {x: y for x, y in prev_b.items() if x in rec["table"]}
                    bm.update({x: str(e["top"]) for x, e in rec["table"].items() if e.get("bound")})
                    if bm:
                        pairs_at.setdefault(kind, {}).setdefault(nid, []).append((rec["x"], rec["y"]))
                        bound_map[(kind, nid, pk)] = bm
        st["pairs_at"] = pairs_at
        st["bound"] = bound_map
        for tk in [k for k in st["trackers"] if k not in bound_map]:
            del st["trackers"][tk]
        live = {(k, n) for k, tr in ptm.kinds.items() for n in tr.nodes}
        for sk in [k for k in st["seg"] if (k[0], k[1]) not in live or not st["seg"][k]]:
            del st["seg"][sk]
        ms = (time.perf_counter() - t0) * 1000.0
        model["last_run"] = now
        model["updated"] = now
        model["version"] = int(model.get("version", 0)) + (1 if n_fit else 0)
        model["gain"] = {"bits_per_event": gnum / gden if gden > 0 else 0.0, "ms": ms,
                         "nodes_fitted": n_fit, "pairs": len(specs)}
        store.put_model(key, SYSTEM_ENTITY, MP.PBIND, model, version=model["version"], ts=now)
        store.put_model(key, SYSTEM_ENTITY, STATE, st, ts=now)
        return n_fit

    def _rebind(self, ctx: Context, key: str, kind: int, node: Any, X: str, Y: str, ps: Any,
                rec: Dict[str, Any], st: Dict[str, Any], now: float,
                segs: Dict[Any, Any]) -> Dict[str, Any]:
        tk = (kind, node.id, pair_key(X, Y))
        tr = st["trackers"].get(tk)
        if tr is None:
            return rec
        changed = []
        sticky = (st.get("bound") or {}).get(tk) or {}
        for xk in list(ps.x.keys()):
            ent = rec["table"].get(str(xk))
            if not ent:
                continue
            old_y = str(ent["top"]) if ent.get("bound") else sticky.get(str(xk))
            if old_y is None:
                continue
            new = tr.rebind_candidate(str(xk), old_y)
            if new is None:
                continue
            ymatch = [y for y in (ps.y.get(xk).keys() if ps.y.get(xk) is not None else [])
                      if str(y) == new]
            if not ymatch:
                continue
            segs[xk] = FD.rebind_baseline(ps, xk, ymatch[0], now)
            tr.clear_candidates(str(xk))
            sticky[str(xk)] = new
            changed.append((str(xk), old_y, new))
        if not changed:
            return rec
        rec2 = FD.fit_pair(ps, now, segs) or rec
        rec2.update({"x": rec["x"], "y": rec["y"], "dir": rec["dir"]})
        for x, old, new in changed:
            ent = rec2["table"].get(x, {})
            ent["rebound"] = {"from": old, "to": new, "t": now}
            entity = x if FD.parse_x(X) == ("net.src", 0) else SYSTEM_ENTITY
            ctx.store.add_event(BehaviorEvent(
                system=key, entity=entity, ts=now, kind="binding_changed", score=0.0,
                severity=Severity.INFO,
                description=f"binding {X} -> {Y} of {x} changed from {old} to {new}",
                extra={"node": int(node.id), "kind": int(kind), "x_attr": X, "y_attr": Y,
                       "x": x, "old": old, "new": new},
                dedupe_key=f"binding_changed|{key}|{node.id}|{X}|{Y}|{x}|{new}"))
        return rec2
