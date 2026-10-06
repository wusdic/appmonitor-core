"""P08 Binding (`behavior.binding`) — approximate functional dependencies and
value bindings per pattern node (docs/lib3/progressive.md §6.12, card P08).

Every tick (cheap, bounded by the learned rows of tick t - D):
    * the learned, trusted rows (quarantined IPs excluded, §6.9.3) feed a
      stratified screening probe (lib/pfd.ProbeReservoir: <= S_max strata x
      R_k rows, candidate columns only);
    * for every screened pair (X, Y), rows of a source some node's pair sketch
      tracks update the tree-level value history (lib/pfd.ValueHistory: per
      source and value first / last seen, clean events and normal days, the
      last 5 values). "Clean" = not damped by P03 (pat.assign `damp`).
Hourly per tree (entity_due), trees whose arm 'p08' is not 'off':
    * screening (lib/pfd.screen) -> <= Q_pairs pairs (X, Y) and the nodes where
      Y (reverse: X) is a target, <= 8 nodes per pair and <= 32 pair-nodes per
      tree, written to model.pwant['pairs'] for P04, which keeps a
      pnode.PairSketch at node.pairs[(X, Y)] (X = 'net.src' or 'net.src@<level>',
      x = gen(attr, level, value));
    * fit of every node whose sketch grew or whose sources' value histories
      changed (lib/pfd.fit_pair) with the history's verdicts (lib/pfd.classify):
      a value superseded by a rename (the current value has >= 5 clean events
      over >= 2 normal days, the old one is absent from the last 5) and a
      minority newcomer still pending confirmation (>= 5 clean events, >= 2
      normal days, >= 5 days, no governor episode of the source, and not the
      established value of another source: a borrowed credential is never
      adopted by persistence, §6.9.2) are kept out of the source's counts and
      listed on its table entry ('superseded' / 'pending'). A source whose
      bound value moves to the value that superseded it emits
      `binding_changed` (INFO) and carries 'rebound'.
      Deviation (2026-10-01, measured on pack O): the per-node rebinding
      trackers and segment baselines were replaced by the tree-level history:
      A2 (192.168.1.21 logging in as rose for five days) was adopted as
      {jack, rose}, and at the 综合部 login node created by a late split the
      D2 rename read {mike, mike.w} because the rebinding lived in the parent
      node's baselines only.
      Deviation (round 3, measured on pack O finance): hierarchical pooling.
      A newly screened pair's value history is seeded from the probe's rows of
      it (_seed_histories: the rows kept since the payload became a candidate,
      <= 64 sources seen on >= 2 days), and a source is judged on its tree-level
      clean normal DAYS (lib/pfd.history_support) when they hold more evidence
      than the node's young sketch and name the same value, with the node's
      leave-one-out empirical-Bayes prior as before: 财务部's once-a-day users
      were screened on day 8, their sketch started on day 9 and no binding was
      ever stated (P12 switched the arm off on day 10 for want of a judged one).
Writes  model.pbind@(tree key, '__system__'):
          {'fmt': 1, 'version', 'updated', 'applicable',
           'nodes': {kind: {nid: {'status', 'pairs': {'X->Y': record}, 'fit_t', 'n_c'}}},
           'fit': {kind: {nid: fit mark + 'hv'}}, 'screen': [pair specs], 'gain': {...}}
        record = lib/pfd.fit_pair + {'x', 'y', 'dir': 'fwd'|'rev'}; table entries carry
        first / last seen from the value history.
        model.pwant['pairs'] = {'fmt': 1, 'updated', 'by_kind': {kind: {nid: [[X, Y], ...]}},
                                'specs': [...]}
        model.pbind_state@(tree key, '__system__') = private bookkeeping (probe,
        value histories, tracked sources, last bound values, rebinding marks,
        ingestion marks).
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
CLEAN_COL = "__clean"         # probe rows: P03 did not damp the row (never a screened column)
HIST_IDLE_S = 7 * 86400.0     # an untracked source's value history is kept while seen within a week ...
SEED_X = 64                   # ... (seeded: <= 64 sources per pair, each seen on >= 2 days)


def _stratum(get) -> str:
    for a in STRATUM_KEYS:
        v = get(a)
        if v is not EV.ABSENT and v is not None:
            return f"{a}={v}"
    v = get("ev.ch")
    return "ch=" + (str(v) if v is not EV.ABSENT else "?")


def _stratum_of(s_cols: List[Tuple[str, Optional[List[Any]]]], ch_col: Optional[List[Any]], i: int,
                ip: str, memo: Dict[Any, str]) -> str:
    """_stratum of row i from the columns of STRATUM_KEYS (None: not a batch
    or context attribute) and of ev.ch."""
    for a, col in s_cols:
        v = col[i] if col is not None else (ip if a == "net.src" else EV.ABSENT)
        if v is not EV.ABSENT and v is not None:
            if v.__class__ is not str:
                return f"{a}={v}"
            k = (a, v)
            out = memo.get(k)
            if out is None:
                out = memo[k] = f"{a}={v}"
            return out
    v = ch_col[i] if ch_col is not None else EV.ABSENT
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
            st = {"probe": FD.ProbeReservoir(), "hist": {}, "last": {}, "bound": {}, "rebound": {}}
            store.put_model(key, SYSTEM_ENTITY, STATE, st)
        st.setdefault("hist", {})
        st.setdefault("bound", {})
        st.setdefault("rebound", {})
        st.setdefault("track", {})
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
            pm = MP.get_model(store, key, MP.PBIND)
            specs = [(d["x"], d["y"]) for d in ((pm or {}).get("screen") or [])] \
                if isinstance(pm, Mapping) else []
            hier = MP.hierarchies(store, key, ctx.config, reg) if specs else None
            pcal = store.get_model(s, SYSTEM_ENTITY, PCAL)
            normal = (pcal or {}).get("normal", {}) if isinstance(pcal, Mapping) else {}
            for ts_b, b in batches:
                cb = store.batch_at(s, EV.EVT_CTX, ts_b)
                ingested += self._ingest(store, s, key, ts_b, b, cb, st, ycand, specs, hier, tz,
                                         cal, normal, qcache)
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
    def _ingest(self, store: Any, s: str, key: str, ts_b: float, b: Any, cb: Any,
                st: Dict[str, Any], ycand: set, specs: List[Tuple[str, str]], hier: Any, tz: str,
                cal: Any, normal: Mapping[int, bool], qcache: Dict[Tuple[str, str], bool]) -> int:
        probe: FD.ProbeReservoir = st["probe"]
        rows = b.learned_rows()
        cols = [a for a in b.cols if a in ycand]
        mass = b.mass()
        # P03's outlier damping of the row (§6.9.3): a damped row (an extreme
        # outlier, a value credibly bound to another source, a who outsider)
        # never counts toward the confirmation of a new value
        asg = store.batch_at(s, EV.PAT_ASSIGN, ts_b)
        damp = asg.dense("damp", 1.0) if (asg is not None and asg.n == b.n and asg.has("damp")) else None
        dmemo: Dict[int, int] = {}
        n = 0
        dense = {a: b.dense(a).tolist() for a in list(cols) + [a for a in X_ATTRS if a in b.cols]}
        dense_items = list(dense.items())
        # column-wise forms of the per-row lookups (round 5; the values the
        # former per-row `get` closure returned, tests/engines/
        # test_p08_ingest_equivalence.py): an attribute of the batch, else of
        # the row-aligned evt.ctx, else net.src = the row's address, else absent
        n_b = int(b.n)
        colcache: Dict[str, Optional[List[Any]]] = {}

        def column(a: str) -> Optional[List[Any]]:
            if a in colcache:
                return colcache[a]
            if a in b.cols:
                out = [b.get(a, i) for i in range(n_b)]
            elif cb is not None and a in cb.cols:
                out = [cb.get(a, i) for i in range(n_b)]
            else:
                out = None                     # net.src (the row's address) or absent
            colcache[a] = out
            return out
        s_cols = [(a, column(a)) for a in STRATUM_KEYS]
        ch_col = column("ev.ch")
        smemo: Dict[Any, str] = {}
        pspecs = []
        if specs and hier is not None:
            for (X, Y) in specs:
                xa, xl = FD.parse_x(X)
                ya, yl = FD.parse_x(Y)
                pspecs.append((xa, xl, ya, yl, pair_key(X, Y), column(xa), column(ya)))
        ts_all = b.ts.tolist()
        rid_all = b.rid.tolist()
        damp_l = damp.tolist() if damp is not None else None
        hist_all, track_all = st["hist"], st["track"]
        for i in rows.tolist():
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
            clean_i = damp_l is None or not (float(damp_l[i]) < 1.0)
            row = {"net.src": ip, CLEAN_COL: bool(clean_i)}
            for a, arr in dense_items:
                v = arr[i]
                if v is not EV.ABSENT:
                    row[a] = v
            ts = float(ts_all[i])
            if len(row) > 2:
                probe.offer(_stratum_of(s_cols, ch_col, i, ip, smemo), row, float(mass[i]), ts,
                            seeded_uniform("p08", s, ts, int(rid_all[i])))
                n += 1
            if not pspecs:
                continue
            day = None
            for (xa, xl, ya, yl, pk, xcol, ycol) in pspecs:
                xv = xcol[i] if xcol is not None else (ip if xa == "net.src" else EV.ABSENT)
                yv = ycol[i] if ycol is not None else (ip if ya == "net.src" else EV.ABSENT)
                if xv is EV.ABSENT or yv is EV.ABSENT or xv is None or yv is None:
                    continue
                xg = hier.gen(xa, xl, xv)
                yg = hier.gen(ya, yl, yv) if yl else yv
                if xg is None or yg is None:
                    continue
                if day is None:
                    mk = int(ts // 60)
                    day = dmemo.get(mk)
                    if day is None:
                        day = dmemo[mk] = TB.local_datetime(mk * 60.0, tz).date().toordinal()
                    is_normal = bool(normal.get(day, _dt.date.fromordinal(day) not in cal.holidays))
                    clean = clean_i
                hist = hist_all.get(pk)
                tr = track_all.get(pk)
                if tr is not None:
                    if str(xg) not in tr:
                        continue                # history only for sources some node tracks
                elif hist is None or str(xg) not in hist.x:
                    continue                    # a screened pair no sketch holds yet: its seeded sources
                if hist is None:
                    hist = hist_all[pk] = FD.ValueHistory()
                hist.observe(str(xg), yg, ts, day, is_normal, clean)
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
            self._seed_histories(ctx, key, st, model["screen"], hier)
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
                hv = self._hist_version(st, node)
                dirty = PB.is_dirty(fits.get(nid), node, now) or \
                    int((fits.get(nid) or {}).get("hv", -1)) != hv
                entry = prev if (prev and not dirty) else {"status": "none", "pairs": {}}
                if dirty:
                    entry = {"status": "none", "pairs": {}, "fit_t": now,
                             "n_c": float(node.n_c(now)), "state": node.state}
                    for (X, Y), ps in node.pairs.items():
                        pk = pair_key(X, Y)
                        rec = self._fit_one(ctx, key, kind, node, X, Y, ps, st, now)
                        if rec is None:
                            continue
                        rec.update({"x": X, "y": Y,
                                    "dir": dirs.get((X, Y), "rev" if "net.src" in Y else "fwd")})
                        old = ((prev or {}).get("pairs") or {}).get(pk)
                        cv = int((old or {}).get("cver", 0))
                        rec["cver"] = cv + 1 if old and FD.material_change(old, rec) else cv
                        entry["pairs"][pk] = rec
                        gnum += node.mass_at(now) * rec["gain"]
                        gden += node.mass_at(now)
                    entry["status"] = "fitted" if entry["pairs"] else "none"
                    outn[nid] = entry
                    fits[nid] = dict(PB.fit_mark(node, now), hv=hv)
                    n_fit += 1
                for pk, rec in (entry.get("pairs") or {}).items():
                    # a source keeps its last bound value while it is tracked (a
                    # rename passes through a few unbound fits before the new value
                    # has the evidence of a binding; `binding_changed` compares with it)
                    prev_b = (st.get("bound") or {}).get((kind, nid, pk)) or {}
                    bm = {x: y for x, y in prev_b.items() if x in rec["table"]}
                    bm.update({x: str(e["top"]) for x, e in rec["table"].items() if e.get("bound")})
                    if bm:
                        bound_map[(kind, nid, pk)] = bm
        st["bound"] = bound_map
        # the sources whose value history is kept: those some node's pair
        # sketch tracks (<= PAIR_NODES x 64), so the history is bounded by the
        # sketches and never by the number of IPs seen
        track: Dict[str, set] = {}
        for tree in ptm.kinds.values():
            for node in tree.nodes.values():
                for (X, Y), ps in (node.pairs or {}).items():
                    track.setdefault(pair_key(X, Y), set()).update(str(x) for x in ps.x.keys())
        st["track"] = track
        screened = {pair_key(d["x"], d["y"]) for d in specs}
        for pk in list(st["hist"]):
            keep = track.get(pk)
            if not keep:
                if pk not in screened:
                    del st["hist"][pk]          # a seeded history waits for its sketches while screened
                continue
            h = st["hist"][pk]
            # an untracked source leaves the history once it has been silent for
            # HIST_IDLE_S: a source seeded from the probe is often not yet in any
            # sketch when the first fit after the sketch's start runs (finance:
            # .10 / .11 had not logged in yet at 09:00, kate had - the seeded days
            # of the other two were deleted and they started again from 1)
            idle = [x for x in h.x if x not in keep]
            for x in idle:
                seen = h.seen(x)
                multi = max((len(r[3]) for r in h.x[x]["v"].values()), default=0) >= FD.REBIND_DAYS
                if seen is None or not multi or now - seen[1] > HIST_IDLE_S:
                    del h.x[x]
            extra = [x for x in h.x if x not in keep]
            for x in extra[:max(0, len(extra) - SEED_X)]:
                del h.x[x]                      # (LRU order: the least recently seen first)
        live = {(k, n) for k, tr in ptm.kinds.items() for n in tr.nodes}
        st["rebound"] = {k: v for k, v in st["rebound"].items() if (k[0], k[1]) in live}
        ms = (time.perf_counter() - t0) * 1000.0
        model["last_run"] = now
        model["updated"] = now
        model["version"] = int(model.get("version", 0)) + (1 if n_fit else 0)
        model["gain"] = {"bits_per_event": gnum / gden if gden > 0 else 0.0, "ms": ms,
                         "nodes_fitted": n_fit, "pairs": len(specs)}
        store.put_model(key, SYSTEM_ENTITY, MP.PBIND, model, version=model["version"], ts=now)
        store.put_model(key, SYSTEM_ENTITY, STATE, st, ts=now)
        return n_fit

    @staticmethod
    def _seed_histories(ctx: Context, key: str, st: Dict[str, Any], specs: Iterable[Mapping[str, Any]],
                        hier: Any) -> int:
        """A newly screened pair's value history starts from the probe's rows of
        it (the learned rows P08 kept since the payload became a candidate,
        oldest first, with their clean flags and normal days), not from the
        first row after a node's sketch exists: pack O's finance logins (three
        users, one login a workday) were screened on day 8 and their sketch
        started on day 9, so no source reached n_bind before P12 judged the arm.
        Bounded by the probe (<= R_TOTAL rows per tree)."""
        probe: FD.ProbeReservoir = st["probe"]
        new = [d for d in specs if pair_key(d["x"], d["y"]) not in st["hist"]]
        if not new or hier is None:
            return 0
        tz = ctx.config.get("tz") or TB.DEFAULT_TZ
        cal = TB.parse_calendar(ctx.config.get("calendar"))
        pcal = ctx.store.get_model(key, SYSTEM_ENTITY, PCAL)
        normal = (pcal or {}).get("normal", {}) if isinstance(pcal, Mapping) else {}
        rows = probe.timed_rows()
        n = 0
        for d in new:
            X, Y = d["x"], d["y"]
            xa, xl = FD.parse_x(X)
            ya, yl = FD.parse_x(Y)
            h = FD.ValueHistory()
            for row, ts in rows:
                xv, yv = row.get(xa), row.get(ya)
                if xv is None or yv is None:
                    continue
                xg = hier.gen(xa, xl, xv)
                yg = hier.gen(ya, yl, yv) if yl else yv
                if xg is None or yg is None:
                    continue
                day = TB.local_datetime(ts, tz).date().toordinal()
                is_normal = bool(normal.get(day, _dt.date.fromordinal(day) not in cal.holidays))
                h.observe(str(xg), yg, ts, day, is_normal, bool(row.get(CLEAN_COL, True)))
                n += 1
            # only sources the probe saw on >= REBIND_DAYS days can be pooled (a
            # one-off visitor never could), at most SEED_X of them (most days first):
            # the history stays bounded like the sketches (<= 64 sources a node tracks)
            keep = sorted(((max((len(r[3]) for r in v["v"].values()), default=0), x)
                           for x, v in h.x.items()), reverse=True)
            keep = {x for d_, x in keep[:SEED_X] if d_ >= FD.REBIND_DAYS}
            for x in [x for x in h.x if x not in keep]:
                del h.x[x]
            if h.x:
                st["hist"][pair_key(X, Y)] = h
        return n

    @staticmethod
    def _hist_version(st: Mapping[str, Any], node: Any) -> int:
        """Observation counter of the value histories of the node's tracked
        sources: a source's new value can be confirmed or a rename recognised
        by a row that adds too little evidence to make the node dirty
        (< 10 % / 20 units; a department's login node grows ~3 units a day)."""
        hv = 0
        for (X, Y), ps in (node.pairs or {}).items():
            h = st["hist"].get(pair_key(X, Y))
            if h is not None:
                hv += h.version(ps.x.keys())
        return hv

    def _fit_one(self, ctx: Context, key: str, kind: int, node: Any, X: str, Y: str, ps: Any,
                 st: Dict[str, Any], now: float) -> Optional[Dict[str, Any]]:
        """Fit one pair sketch of a node with the value history's verdicts
        (lib/pfd.classify): values superseded by a rename and minority values
        still pending confirmation are kept out of their source's counts.
        Two passes: the established value of every source first (without the
        cross-source rule), then each source's verdicts knowing the values
        established at the node's other sources. A source whose bound value
        moved to the value that superseded it emits `binding_changed` (INFO)."""
        pk = pair_key(X, Y)
        hist: Optional[FD.ValueHistory] = st["hist"].get(pk)
        excl: Dict[str, Dict[str, str]] = {}
        if hist is not None:
            xs = list(ps.x.keys())
            ip_x = FD.parse_x(X) == ("net.src", 0)
            values = {str(x): [y for y, _, _ in rows] for x, rows in ps.table(now).items()}

            def episode(x: str) -> Optional[Any]:
                if not ip_x:
                    return None
                return lambda t0: bool(MG.episodes(ctx.store, key, x, since=float(t0)))
            first = {str(x): FD.classify(hist.get(str(x)), values.get(str(x), ()), now,
                                         (), episode(str(x))) for x in xs}
            cnt = FD.pair_counts(ps, now, None, first)
            est: Dict[str, str] = {}
            for x, c in cnt.items():
                if c["n"] >= FD.N_BIND and c["y"] and not FD.is_shared(x):
                    est[str(x)] = str(FD._jv(max(c["y"].items(), key=lambda kv: kv[1])[0]))
            for x in xs:
                others = [v for xx, v in est.items() if xx != str(x)]
                ex = FD.classify(hist.get(str(x)), values.get(str(x), ()), now, others, episode(str(x)))
                if ex:
                    excl[str(x)] = ex
            # hierarchical pooling (deviation, §6.12): each source's tree-level days
            # (every node of the pair, seeded with the probe rows from before the
            # sketch existed), the node's leave-one-out prior over its other sources
            sup = {str(x): FD.history_support(hist.get(str(x)), excl.get(str(x))) for x in xs}
        else:
            sup = {}
        rec = FD.fit_pair(ps, now, exclude=excl, support=sup)
        if rec is None:
            return None
        prev_b = (st.get("bound") or {}).get((kind, node.id, pk)) or {}
        for x, ent in rec["table"].items():
            if hist is not None:
                fl = hist.seen(x)
                if fl is not None:
                    ent["first"], ent["last"] = fl
            rk = (kind, node.id, pk, x)
            old = prev_b.get(x)
            if ent.get("bound") and old is not None and old != str(ent["top"]) \
                    and old in (ent.get("superseded") or ()):
                st["rebound"][rk] = {"from": old, "to": str(ent["top"]), "t": now}
                entity = x if FD.parse_x(X) == ("net.src", 0) else SYSTEM_ENTITY
                ctx.store.add_event(BehaviorEvent(
                    system=key, entity=entity, ts=now, kind="binding_changed", score=0.0,
                    severity=Severity.INFO,
                    description=f"binding {X} -> {Y} of {x} changed from {old} to {ent['top']}",
                    extra={"node": int(node.id), "kind": int(kind), "x_attr": X, "y_attr": Y,
                           "x": x, "old": old, "new": str(ent["top"])},
                    dedupe_key=f"binding_changed|{key}|{node.id}|{X}|{Y}|{x}|{ent['top']}"))
            rb = st["rebound"].get(rk)
            if rb is not None and rb["to"] == str(ent["top"]):
                ent["rebound"] = dict(rb)
        return rec
