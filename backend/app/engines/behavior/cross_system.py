"""CrossSystemEngine (B21, P2) — an IP-centric view across business systems.

Why: every other detector scores one (system, IP) key against its own past.
An actor that moves laterally - a user workstation of oa-portal that starts
calling api-gateway and erp-prod /admin (eval T20) - appears in the other
systems as a brand-new key with no past of its own, whose single-key
detectors can only compare it with the target system's population. What is
anomalous is the IP's FOOTPRINT: which systems it touches, against its own
history, its class peers and the organisation. B21 keeps that footprint and
scores every (system, IP) activity by how expected the system is for the IP.

Unit of observation (canonical grain semantics, docs/lib3/cadence.md D1/D2):
a (system, IP) pair counts once per epoch-aligned H window (3600 s) in which
it is active (feature.active), at every cadence; tick mode (G_h := dt) counts
every active tick. A pair is scored at its FIRST active tick of each window,
so a first access is scored the moment it happens and a pair yields one score
per active hour whatever the tick (a 60-s tick does not repeat the evidence
60 times). The daily footprint and the 24-h spread are wall-clock.

  * State (model.xsys@(__org__, __org__), layout and maths: lib/m_xsys.py):
    per pair a decayed count of committed active windows (half-life 30 d),
    first / last committed ts and the committed active local days. Learning
    is lib/gating per pair key (system, IP): delayed D, trust-weighted, held
    under quarantine, reversible through model.control and checkpoints
    (contract H). The clock is B21's own 1-element ring behavior.xsys (one
    row per scored pair-window), so gating only visits scored rows.
  * Tiers (rebuilt hourly from the committed pair states): class = the IP's
    class IN ITS HOME SYSTEM (the system with most committed windows;
    contract L): its static CIDR class (ctx.config.ip_classes via
    model.class), else its role there when it has >= 3 members, else the
    home system's IPs (system tier); org = all pairs. B02 role ids are
    org-wide, so a role pooled over systems would mix per-system
    footprints. The class counts leave the IP out.
  * Novelty: three-tier Dirichlet backoff IP -> class -> org with Good-Turing
    maturity (lib/m_xsys.predictive); p_nov = the upper p-value of the
    system under the IP's predictive (1 for a habitual system; tiny for a
    system neither the IP nor its class uses; lib/m_xsys.upper_p).
  * Lateral spread: X = distinct systems of the IP in the last 24 h - 1
    under a Gamma-Poisson (NB) predictive of the IP's committed active days
    with a class prior (lib/m_xsys.spread_p).
  * p = min(1, 2 min(p_nov, p_spread)) (Bonferroni over the two tests);
    score.cross_system = -log10 p on a 0.25-decade grid (m_xsys.score_of:
    the null support is discrete, the grid keeps B24's ring from fitting a
    near-zero-width tail), behavior.pm[cross_system] = p unrounded (B24's
    prior while the key's ring is small: a new target key has none).
    Instantaneous, family xsys, axis 'lateral' (lib/stages: stage lateral).
  * Class adoption: a system new to the IP that max(3, 30 %) of its class
    first reached within 7 d is an adoption: p -> p^0.1 (B08's discount of
    the score) and no event.
  * Tier of a system new to the IP: 'entity' when its class uses it (not
    class-rare: df + 1 > (members + 1) / 5), else 'org' when no other IP
    bridges the IP's home system and it (Adamic-Adar support 0), else
    'class'. Event first_access_system (kind of contract F; LOW, MEDIUM when
    another system of the IP alarmed in the last 24 h - the precursor of a
    lateral move) when the tier >= class and p <= 0.005, once per pair per
    30 d; extra = {system, ip, home, tier, p_nov, p_spread, support_aa,
    k_24h, precursor, adopted}; axes ['lateral'].
  * profile.extra.systems_accessed {system: {hours, first, last}} on every
    key of an active IP, hourly.
  * ctx.training learns and scores but emits no events. A producer error of
    B01 (feature.active) writes NaN + behavior.degraded for the pairs active
    in the last tick; an idle pair is not scored (absence is not lateral).

Store: reads feature.active (every real key), model.class, behavior.alarm
(other keys of the IP, t-1: B25 runs after B21), behavior.trust /
trust_prov / quarantine and model.control (lib/gating); writes model.xsys,
behavior.xsys (gating clock), behavior.score / pm / axes / degraded
[cross_system] (lib/emit), profile.extra.systems_accessed, events
first_access_system.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import BehaviorEvent, Severity
from .lib import combine
from .lib import emit
from .lib import gating as G
from .lib import grains as GR
from .lib import m_class
from .lib import m_xsys as MX
from .lib.classkeys import ORG

DETECTOR = "cross_system"
MODEL = MX.MODEL
ACTIVE = "feature.active"
CLOCK = "behavior.xsys"             # own gating clock: one row per scored pair-window
ALARM = "behavior.alarm"
B01 = "behavior.feature_vector"
LEARNER = "xsys"
XSYS_DIM = "xsys"                  # token dimension of a first access (event extra)
AXES = ["lateral"]

DAY = 86400.0
TIER_REFIT_S = 3600.0
PROFILE_EVERY_S = 3600.0
ROW_KEEP_S = 8 * DAY                # scored rows kept for commit / rollback replay
ROW_CAP = 256
LIVE_KEEP_S = 35 * DAY              # a pair's live record outlives the 30-d dedupe
PAIR_DROP_C = 0.01                  # committed count below which an old pair is dropped
EMIT_KEEP_S = 30 * DAY
P_EVENT = 0.005
ADOPT_WINDOW_S = 7 * DAY
ADOPT_MIN = 3
ADOPT_FRAC = 0.3
ADOPT_POWER = 0.1                   # p -> p^0.1: B08's 0.1 x score discount
CLASS_RARE_DIV = 5.0                # IDF >= ln 5 + 1 (B08)
PRECURSOR_S = DAY
TIERS = ("entity", "class", "org")
_ONE = np.ones(1, dtype=np.float32)


def _ctl_sig(control: Any) -> Optional[Tuple]:
    if not isinstance(control, Mapping):
        return None
    return tuple(repr(control.get(k)) for k in G._CONTROL_KEYS)


def new_model() -> Dict[str, Any]:
    return {"fmt": 1, "version": 0, "ts": -math.inf, "pairs": {}, "live": {}, "emitted": {},
            "tiers": {"ts": -math.inf, "cls": {}, "ip_cls": {}, "org": {}, "home": {}}}


class _IP:
    """Per-IP scoring context of one tick."""
    __slots__ = ("ip", "fp", "home", "cls", "pred", "n_ip", "k24", "p_spread", "spread_mean")

    def __init__(self, ip: str) -> None:
        self.ip = ip
        self.fp: Dict[str, float] = {}
        self.home: Optional[str] = None
        self.cls: Optional[str] = None
        self.pred: Dict[str, float] = {}
        self.n_ip = 0.0
        self.k24 = 0
        self.p_spread = 1.0
        self.spread_mean = math.nan


class CrossSystemEngine(Engine):
    name = "behavior.cross_system"
    layer = "behavior"
    consumes = [ACTIVE, "model.class", ALARM, G.TRUST, G.TRUST_PROV, G.QUARANTINE,
                G.CONTROL_MODEL]
    produces = [MODEL, CLOCK, emit.SCORE, emit.PM, emit.AXES, emit.DEGRADED,
                "profile.extra.systems_accessed", "event:first_access_system"]
    description = ("IP-centric cross-system footprint: first access of an IP to a business "
                   "system under an IP -> class -> org Dirichlet backoff with Good-Turing "
                   "maturity, 24-h lateral spread under an NB predictive with a class prior, "
                   "Adamic-Adar support of the new edge, class-adoption discount; "
                   "first_access_system events (stage lateral).")
    # interval 1 (spec: 4): the per-window rule makes one score per active
    # pair-hour at any cadence, and the first active tick of a window is
    # the tick of a first access; skipping ticks would delay it by up to 4.
    interval = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._fps: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None
        self._rows_cur: Optional[Dict[float, Tuple[float, float, int]]] = None
        self._learner = G.GatedLearner(
            name=LEARNER, init=MX.new_pair_state, update=MX.pair_update, fetch=self._fetch,
            dump=MX.dump_pair, load=MX.load_pair, merge=None, clock=CLOCK)

    # ------------------------------------------------------------------ run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        now, dt = float(ctx.now), float(ctx.window_s)
        self._learner.d_min_s = float(ctx.config.get("D_min_s") or G.D_MIN_S)
        self._ensure_retention(store)
        systems = store.systems()
        if not systems:
            return 0
        model = store.get_model(ORG[0], ORG[1], MODEL)
        if not (isinstance(model, dict) and model.get("fmt") == 1):
            model = new_model()
        canon = GR.canonical(ctx.config)
        wid = MX.window_start(now, dt, canon)
        tz = str(ctx.config.get("tz") or "Asia/Shanghai")
        day = GR.local_date(now, tz).toordinal()
        live: Dict[str, Dict[str, Any]] = model["live"]

        if store.engine_failed(B01, now):
            n = self._degraded(store, live, now, dt)
            store.put_model(ORG[0], ORG[1], MODEL, model, version=int(model["version"]), ts=now)
            return n

        # 1) observe: the systems every IP is active in at this tick
        act: Dict[str, List[str]] = {}
        for s in systems:
            for e in store.entities(s):
                row = store.vec_at(s, e, ACTIVE, now)
                if row is not None and len(row) and float(row[0]) > 0.5:
                    act.setdefault(e, []).append(s)
        for ip, ss in act.items():
            for s in ss:
                pk = MX.pair_key(s, ip)
                lv = live.get(pk)
                if lv is None:
                    lv = live[pk] = {"first": now, "last": now, "wid": math.nan, "rows": {}}
                lv["last"] = now

        # 2) class / org tiers from the committed pair states (hourly)
        tiers = model["tiers"]
        if now - float(tiers.get("ts", -math.inf)) >= TIER_REFIT_S or now < float(tiers["ts"]):
            model["tiers"] = tiers = self._tiers(store, model, systems, now, day)

        # 3) score each pair at its first active tick of the window
        n = 0
        self._fps = None                   # committed footprints, built once when needed
        for ip in sorted(act):
            due = [s for s in act[ip] if live[MX.pair_key(s, ip)]["wid"] != wid]
            if not due:
                continue
            if self._fps is None:
                self._fps = MX.footprints(model, now)
            ipc = self._ip_context(store, model, tiers, systems, ip, now, day)
            for s in sorted(due):
                n += self._score(ctx, model, tiers, ipc, s, ip, now, dt, wid, day)

        # 4) learning, per pair key (lib/gating)
        frontier = G.commit_frontier(now, dt, self._learner.d_min_s)
        for pk in sorted(live):
            self._learn(ctx, model, pk, now, dt, frontier)

        # 5) profiles of the active keys, hourly
        for ip, ss in act.items():
            for s in ss:
                if self.entity_due((s, ip), now, PROFILE_EVERY_S):
                    self._profile(store, model, s, ip, now)

        self._prune(model, now)
        model["ts"] = now
        store.put_model(ORG[0], ORG[1], MODEL, model, version=int(model["version"]), ts=now)
        return n

    # ------------------------------------------------------------ retention
    def _ensure_retention(self, store) -> None:
        if getattr(self, "_ret_store", None) is store:
            return
        store.ensure_retention(CLOCK, max_age_s=ROW_KEEP_S + DAY)
        self._ret_store = store

    # ----------------------------------------------------------------- tiers
    def _tiers(self, store, model: Dict[str, Any], systems: List[str], now: float,
               day: int) -> Dict[str, Any]:
        """Class and org counts from the committed pair states (IP left in;
        the scorer subtracts the IP's own counts)."""
        by_ip: Dict[str, Dict[str, Tuple[float, float, Mapping]]] = {}
        for pk, rec in model["pairs"].items():
            s, ip = MX.split_pair(pk)
            st = rec.get("state") or {}
            c = MX.decayed(st, now)
            n1 = c if int(st.get("n", 0)) == 1 else 0.0
            by_ip.setdefault(ip, {})[s] = (c, n1, st)
        org: Dict[str, float] = {}
        home: Dict[str, str] = {}
        ip_cls: Dict[str, str] = {}
        cls: Dict[str, Dict[str, Any]] = {}
        for ip, fp in by_ip.items():
            for s, (c, _, _) in fp.items():
                org[s] = org.get(s, 0.0) + c
            h = self._home(fp, None)
            if h is None:
                continue
            home[ip] = h
            ck = self._class_of(store, h, ip)
            ip_cls[ip] = ck
            t = cls.setdefault(ck, {"c": {}, "N": 0.0, "N1": 0.0, "members": [], "df": {},
                                    "xs": 0.0, "xd": 0.0})
            t["members"].append(ip)
            for s, (c, n1, _) in fp.items():
                t["c"][s] = t["c"].get(s, 0.0) + c
                t["N"] += c
                t["N1"] += n1
                if c >= MX.ACTIVE_MIN:
                    t["df"][s] = t["df"].get(s, 0) + 1
            for x, w in MX.daily_extra([st for _, _, st in fp.values()], day):
                t["xs"] += w * x
                t["xd"] += w
        for t in cls.values():
            t["members"].sort()
        return {"ts": now, "cls": cls, "ip_cls": ip_cls, "org": org, "home": home}

    @staticmethod
    def _home(fp: Mapping[str, Tuple[float, float, Any]], fallback: Optional[str]) -> Optional[str]:
        best, bc = fallback, 0.0
        for s in sorted(fp):
            c = fp[s][0]
            if c > bc:
                best, bc = s, c
        return best

    @staticmethod
    def _class_of(store, system: str, ip: str) -> str:
        """The IP's class, keyed per HOME system (contract L: class state is
        instantiated per system; role ids are org-wide, so one role holds
        e.g. the interactive users of every system, each confined to its
        own - pooled across systems it would read "members use erp-prod 50 %
        of the time" and hide an oa-portal user's first erp access):
        '<home>|class:static:<name>' (ctx.config.ip_classes, as B02 assigns
        it), else '<home>|class:<rid>' when the role has >= 3 members there
        (m_class.class_key), else '<home>|__system__' (the system tier: the
        IPs homed in that system; never org)."""
        a = m_class.assignment(store, system, ip)
        if a:
            st = a.get("static") or []
            if st:
                return f"{system}|class:static:{sorted(str(x) for x in st)[0]}"
        ck = m_class.class_key(store, system, ip)
        return f"{system}|{ck}" if ck else f"{system}|__system__"

    # --------------------------------------------------------------- scoring
    def _ip_context(self, store, model: Dict[str, Any], tiers: Mapping[str, Any],
                    systems: List[str], ip: str, now: float, day: int) -> _IP:
        ipc = _IP(ip)
        fp_full = (self._fps or {}).get(ip) or {}
        ipc.fp = {s: v["c"] for s, v in fp_full.items()}
        n1_ip = sum(v["c"] for v in fp_full.values() if v["n"] == 1)
        ipc.n_ip = sum(ipc.fp.values())
        ipc.home = tiers["home"].get(ip)
        if ipc.home is None:
            ipc.home = self._home({s: (c, 0.0, None) for s, c in ipc.fp.items()}, None)
        ipc.cls = tiers["ip_cls"].get(ip)
        if ipc.cls is None and ipc.home is not None:
            ipc.cls = self._class_of(store, ipc.home, ip)
        c_cls: Optional[Dict[str, float]] = None
        n1_cls = 0.0
        t = tiers["cls"].get(ipc.cls) if ipc.cls else None
        own_in = t is not None and ip in t["members"]
        if t is not None:
            c_cls = dict(t["c"])
            n1_cls = float(t["N1"])
            if own_in:                          # leave the IP out of its class
                for s, v in fp_full.items():
                    c_cls[s] = max(0.0, c_cls.get(s, 0.0) - v["c"])
                    if v["n"] == 1:
                        n1_cls = max(0.0, n1_cls - v["c"])
        universe = sorted(set(systems) | set(ipc.fp))
        ipc.pred = MX.predictive(universe, ipc.fp, n1_ip, c_cls, n1_cls, tiers["org"])
        # 24-h spread (observed, ungated) against the committed daily history
        live = model["live"]
        ipc.k24 = sum(1 for s in universe
                      if float((live.get(MX.pair_key(s, ip)) or {}).get("last", -math.inf))
                      > now - DAY)
        st_ip = [(model["pairs"].get(MX.pair_key(s, ip)) or {}).get("state") or {}
                 for s in fp_full]
        x_days = MX.daily_extra(st_ip, day)
        xbar = math.nan
        if t is not None:
            xs, xd = float(t["xs"]), float(t["xd"])
            if own_in:
                xs -= sum(w * x for x, w in x_days)
                xd -= sum(w for _, w in x_days)
            if xd > 0.0:
                xbar = max(0.0, xs) / xd
        ipc.p_spread, ipc.spread_mean, _ = MX.spread_p(ipc.k24 - 1, x_days, xbar)
        return ipc

    def _score(self, ctx: Context, model: Dict[str, Any], tiers: Mapping[str, Any], ipc: _IP,
               s: str, ip: str, now: float, dt: float, wid: float, day: int) -> int:
        store = ctx.store
        pk = MX.pair_key(s, ip)
        lv = model["live"][pk]
        p_nov = MX.upper_p(ipc.pred, s)
        p = MX.combine(p_nov, ipc.p_spread)
        new_to_ip = ipc.fp.get(s, 0.0) < MX.ACTIVE_MIN
        adopted = new_to_ip and self._adopted(model, tiers, ipc, s, ip, now)
        if adopted:
            p = min(1.0, max(MX.P_FLOOR, p ** ADOPT_POWER))
        score = MX.score_of(p)
        degraded = None
        if ipc.n_ip < MX.ACTIVE_MIN:
            # no committed footprint of its own: judged by the class / org tier
            degraded = {DETECTOR: emit.cause(emit.FALLBACK, "class" if ipc.cls else "org")}
        emit.write_scores(store, s, ip, now, {DETECTOR: score}, pm={DETECTOR: p},
                          axes={DETECTOR: AXES}, degraded=degraded, window_s=int(dt))
        store.add_vec(s, ip, CLOCK, now, _ONE, window_s=int(dt))    # the row's ts is the clock
        lv["wid"] = wid
        rows = lv["rows"]
        rows[now] = (now, wid, day)
        if new_to_ip and not ctx.training:
            self._maybe_event(store, model, tiers, ipc, s, ip, now, dt, p, p_nov, adopted)
        return 1

    def _adopted(self, model: Dict[str, Any], tiers: Mapping[str, Any], ipc: _IP, s: str,
                 ip: str, now: float) -> bool:
        t = tiers["cls"].get(ipc.cls) if ipc.cls else None
        if t is None:
            return False
        members = [m for m in t["members"] if m != ip]
        if not members:
            return False
        live = model["live"]
        k = 0
        for m in members:
            lv = live.get(MX.pair_key(s, m))
            if lv is not None and float(lv.get("first", -math.inf)) >= now - ADOPT_WINDOW_S:
                k += 1
        return k >= max(ADOPT_MIN, ADOPT_FRAC * (len(members) + 1))

    def _tier(self, model: Dict[str, Any], tiers: Mapping[str, Any], ipc: _IP, s: str,
              ip: str, now: float) -> Tuple[str, float]:
        t = tiers["cls"].get(ipc.cls) if ipc.cls else None
        if t is not None:
            n_m = len(t["members"])
            df = int(t["df"].get(s, 0)) - (1 if ip in t["members"] and
                                             ipc.fp.get(s, 0.0) >= MX.ACTIVE_MIN else 0)
            if (df + 1.0) > (n_m + 1.0) / CLASS_RARE_DIV:
                return "entity", math.nan
        fps = {j: {sy: v["c"] for sy, v in fp.items()} for j, fp in (self._fps or {}).items()}
        aa = MX.adamic_adar(fps, ip, ipc.home, s)
        return ("org" if aa <= 0.0 else "class"), aa

    def _precursor(self, store, ip: str, s: str, systems: Iterable[str],
                   now: float) -> Optional[Dict[str, Any]]:
        """An alarm on another system of the IP in the last 24 h (behavior.alarm
        of t-1 and before: B25 runs after B21)."""
        best: Optional[Dict[str, Any]] = None
        for sy in systems:
            if sy == s:
                continue
            m = store.latest_derived(sy, ip, ALARM)
            if m is None or not isinstance(m.value, Mapping) or m.ts < now - PRECURSOR_S:
                continue
            if best is None or m.ts > best["ts"]:
                best = {"system": sy, "ts": float(m.ts),
                        "severity": str(m.value.get("severity") or ""),
                        "axes": list(m.value.get("axes") or [])}
        return best

    def _maybe_event(self, store, model: Dict[str, Any], tiers: Mapping[str, Any], ipc: _IP,
                     s: str, ip: str, now: float, dt: float, p: float, p_nov: float,
                     adopted: bool) -> None:
        if adopted or not p <= P_EVENT:
            return
        pk = MX.pair_key(s, ip)
        last = model["emitted"].get(pk)
        if last is not None and now - float(last) < EMIT_KEEP_S:
            return
        tier, aa = self._tier(model, tiers, ipc, s, ip, now)
        if TIERS.index(tier) < TIERS.index("class"):
            return
        pre = self._precursor(store, ip, s, sorted(set(ipc.fp) | set(tiers["org"])), now)
        sev = Severity.MEDIUM if pre is not None else Severity.LOW
        home = ipc.home or "?"
        store.add_event(BehaviorEvent(
            system=s, entity=ip, ts=now, kind="first_access_system",
            score=float(min(1.0, -math.log10(max(p, 1e-300)) / 10.0)), severity=sev,
            description=(f"{ip} (home {home}) reached {s} for the first time "
                         f"({tier} tier, p={p:.1e})"
                         + (f" after an alarm on {pre['system']}" if pre else "")),
            extra={"system": s, "ip": ip, "home": ipc.home, "class": ipc.cls, "tier": tier,
                   "p_nov": p_nov, "p_spread": ipc.p_spread, "k_24h": int(ipc.k24),
                   "spread_mean": ipc.spread_mean, "support_aa": aa,
                   "footprint": {k: round(v, 2) for k, v in sorted(ipc.fp.items())},
                   "precursor": pre, "adopted": bool(adopted), "stage": "lateral",
                   # the new categorical value the finding rests on (m_feedback
                   # token 'xsys=<system>'): B23 policies / B29's counterfactual
                   # candidate 'token:xsys=<system>' (round 4, evaluator)
                   "dim": XSYS_DIM, "value": s},
            p_value=float(p), e_day=float(combine.e_day(p, dt)), axes=list(AXES),
            p_by_detector={DETECTOR: float(p)}, dedupe_key=f"first_access_system|{pk}",
            window=(now - dt, now)))
        model["emitted"][pk] = now

    # -------------------------------------------------------------- learning
    def _learn(self, ctx: Context, model: Dict[str, Any], pk: str, now: float, dt: float,
               frontier: float) -> None:
        store = ctx.store
        s, ip = MX.split_pair(pk)
        lv = model["live"][pk]
        rec = model["pairs"].get(pk)
        rows: Dict[float, Tuple[float, float, int]] = lv["rows"]
        if rec is None:
            if not rows:
                return
            rec = model["pairs"][pk] = {"state": MX.new_pair_state(), "gate": G.GateState()}
        gate: G.GateState = rec["gate"]
        pending = any(gate.last_ts < ts <= frontier for ts in rows)
        control = store.get_model(s, ip, G.CONTROL_MODEL)
        sig = _ctl_sig(control)
        if not pending and sig == lv.get("ctl"):
            return
        self._rows_cur = rows
        try:
            state, gate = self._learner.step(store, s, ip, rec["state"], gate, now, dt,
                                             training=bool(ctx.training))
        finally:
            self._rows_cur = None
        lv["ctl"] = sig
        if int(gate.version) != int(getattr(rec["gate"], "version", 0)):
            model["version"] = int(model["version"]) + 1
        rec["state"], rec["gate"] = state, gate
        cut = now - ROW_KEEP_S
        for ts in [t for t in rows if t < cut]:
            del rows[ts]
        while len(rows) > ROW_CAP:
            del rows[min(rows)]

    def _fetch(self, store: Any, s: str, e: str, ts: float) -> Optional[Tuple[float, float, int]]:
        rows = self._rows_cur
        if rows is None:
            return None
        return rows.get(float(ts))

    # --------------------------------------------------------------- outputs
    def _profile(self, store, model: Dict[str, Any], s: str, ip: str, now: float) -> None:
        prof = store.profile(s, ip)
        if prof is None:
            return
        fp = MX.footprint(model, ip, now)
        prof.extra["systems_accessed"] = {
            sy: {"hours": round(v["c"], 2),
                 "first": None if not math.isfinite(v["first"]) else v["first"],
                 "last": None if not math.isfinite(v["last"]) else v["last"]}
            for sy, v in sorted(fp.items())}
        store.put_profile(prof)

    def _degraded(self, store, live: Mapping[str, Mapping[str, Any]], now: float,
                  dt: float) -> int:
        n = 0
        cause = emit.cause(emit.PRODUCER_ERROR, B01)
        for pk, lv in live.items():
            if float(lv.get("last", -math.inf)) >= now - 1.5 * dt:
                s, ip = MX.split_pair(pk)
                emit.write_scores(store, s, ip, now, {DETECTOR: math.nan},
                                  degraded={DETECTOR: cause}, window_s=int(dt))
                n += 1
        return n

    def _prune(self, model: Dict[str, Any], now: float) -> None:
        live, pairs = model["live"], model["pairs"]
        for pk in [k for k, lv in live.items()
                   if float(lv.get("last", now)) < now - LIVE_KEEP_S and not lv.get("rows")]:
            del live[pk]
            rec = pairs.get(pk)
            if rec is not None and MX.decayed(rec.get("state"), now) < PAIR_DROP_C:
                del pairs[pk]
        em = model["emitted"]
        for pk in [k for k, t in em.items() if float(t) < now - EMIT_KEEP_S]:
            del em[pk]
