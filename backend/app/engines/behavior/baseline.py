"""BaselineEngine (B03) — per-entity seasonal, exposure-aware baseline with
two anchors and a hierarchy of priors, learned only through trust-gated,
delayed, checkpointed and reversible commits.

Why (v1 -> v2): the v1 baseline was a median/MAD over whatever the store
held, so an attacker present for a day became "normal", a quiet entity could
not be scored at all until it had 12 samples, and exposure (15 min vs 1 min
ticks, 3 requests vs 3000) was ignored. Now:

  * Exact predictives per feature family (lib/m_baseline): counts are
    Gamma-Poisson / NB with exposure in minutes, ratios Beta-Binomial on
    (k, n), everything else NIG / Student-t on the FEATURE_SPEC transform.
    Values are rebuilt from feature.nat (8-d retention), so replay and
    rollback never depend on the 1-d feature.vec ring.
  * Seasonal buckets: bin48 (hour x day type, holidays and 调休 aware), plus
    hour-of-week location cells used after 4 weeks of commits. Each commit
    is spread over neighbouring hours with von Mises weights.
  * Two anchors. The CURRENT anchor commits row t - D (D = max(4 ticks,
    D_min_s)) with weight trust(t - D), and each mature bucket's mean may move
    at most 0.1 sigma15 per band-day (sigma15 = predictive sd of a 15-min
    exposure, so the cap does not depend on cadence; a row that would break
    the band is folded with the weight that lands on its edge). The REFERENCE anchor commits with
    a 24-h delay, only rows with trust == 1 and no incident or abnormal
    regime within +-24 h, capped at 0.03 sigma15 per day plus
    model.control.allow_drift; once a golden anchor exists (median of up to 4
    weekly reference snapshots from weeks whose max behavior.risk < 30) the
    reference predictive IS the golden anchor. B04 combines both
    (p = 2 min(p_cur, p_ref)) and B14 measures change against the reference,
    so a creeping current anchor cannot cancel the evidence.
  * Hierarchy: entity -> role class (s, class:<rid>) when the class has >= 3
    members in the system, else system -> org -> hyperprior. Tier statistics
    are the sums of their members' committed current statistics (hence they
    "update from the same gated rows" and follow every member rollback),
    recomputed every tier_period_s; each tier's backoff strength kappa is an
    empirical-Bayes moment estimate from its children, clipped to [2, 50],
    and every link is leave-one-out. A new entity is therefore scored from
    its first tick with its class's predictive.
  * Learning goes through lib/gating.GatedLearner (one learner per anchor):
    delayed commits, holds while quarantined, hourly (current) / daily
    (reference) float32 checkpoints whose unchanged bucket blocks are shared,
    and model.control rollback_to / release / rebase_from / frozen /
    allow_drift. A rebase lifts the current anchor's cap for the new
    regime's rows and resets the reference one day into the new regime.
    Half-lives (7 / 14 / 28 d) are re-chosen per feature every 96 commits by
    the p5/p95 pinball loss.

Per tick: for every real entity step the current learner (rows due at
t - D), the reference learner (hourly batch of rows due at t - 24 h, all
entities in the same tick, or at once on a new model.control directive) and
link seeding; the learners only QUEUE rows on the anchors, and one
m_baseline.flush_many folds them all, vectorised across entities (per-entity
numpy calls would cost ~1 ms per entity). Then the weekly golden snapshot,
put_model, and the legacy profile (baseline_median / mad / seasonal in
feature.vec space at the current bucket, extra.maturity) when the bucket
changes, computed per system in one batch. Tier models (class, system, org)
are refreshed every tier_period_s and whenever model.class or the set of
systems changes.

B03 scores nothing and emits no event, so ctx.training only changes the
gating default (missing trust -> 1). Absence is data: inactive ticks are not
committed (B04 scores active ticks against these predictives; silence is
B07's), but they still advance the commit cursor.

Store:
  reads   feature.active, feature.nat (vec rings), ctx.config (tz, calendar:
          the same tctx B01 writes to feature.tctx), behavior.trust /
          trust_prov / quarantine (via lib/gating), behavior.risk,
          behavior.regime, model.class, model.control, model.link,
          store.incidents
  writes  model.baseline@(s, e), @(s, class:<rid>), @(s, __system__),
          @(__org__, __org__); checkpoints 'baseline.current' and
          'baseline.reference'; profile.baseline_median / baseline_mad /
          seasonal / sample_count and profile.extra.maturity
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import EntityProfile
from .lib import gating as G
from .lib import m_baseline as MB
from .lib import m_class
from .lib import timebins as TB
from .lib.classkeys import CLASS_PREFIX, ORG, POOL_PREFIX, STATIC_PREFIX, SYSTEM_KEY

# legacy: anomaly.py (v1) imports this list; profile.seasonal is filled for it
SEASONAL_FEATURES = ["http_requests", "bytes_up", "flows", "dns_queries", "duty_cycle"]

LEARNER_CUR = "baseline.current"
LEARNER_REF = "baseline.reference"
ACTIVE = "feature.active"
NAT = "feature.nat"
RISK = "behavior.risk"
REGIME = "behavior.regime"

DAY = MB.DAY
WEEK = MB.WEEK
TIER_PERIOD_S = 3600.0           # tier sums, EB kappas and backoff chains
REF_PERIOD_S = 3600.0            # reference learner batch cadence
PROFILE_PERIOD_S = 3600.0
SEASONAL_PERIOD_S = 6 * 3600.0   # legacy seasonal curves and maturity change slowly
REF_WINDOW_S = DAY               # no incident / abnormal regime within +-24 h
REF_ELIG_KEEP_S = G.JOURNAL_MAX_AGE_S + DAY
REGIME_KEEP_S = 10 * DAY
ABNORMAL_REGIMES = frozenset({"suspect", "drifting", "rejected", "rollback"})
GOLDEN_MAX_RISK = 30.0
GOLDEN_SNAPS = 4
GOLDEN_MIN_NEFF = 96.0           # the reference must hold >= 1 day at 900 s
ROW_DT_MAX = 3600.0              # a row's exposure is the gap to its previous tick
_TCTX_CACHE_MAX = 4096


def new_model() -> Dict[str, Any]:
    """Empty model.baseline@(s, e) (hyperprior predictives)."""
    return {
        "fmt": MB.FMT, "tier": "entity", "version": 0, "branch": 0,
        "current": MB.new_anchor(week=True, select=True),
        "reference": MB.new_anchor(week=False, select=False, hl_days=MB.HL_REF_DAYS),
        "gate": G.GateState(), "gate_ref": G.GateState(),
        "golden": {"week": None, "snaps": [], "stats": None, "n_reset": 0},
        "ref_elig": {}, "reg": {"scan": -math.inf, "since": None, "iv": []},
        "ctl_sig": None, "n_eff": 0.0, "allow_drift": 0.0, "held": [], "pb": None,
        "pb_ts": -math.inf,
    }


def _valid(model: Any) -> bool:
    return (isinstance(model, dict) and model.get("fmt") == MB.FMT
            and model.get("tier") == "entity" and isinstance(model.get("current"), MB.Anchor))


def _update_cur(state: MB.Anchor, row: MB.Row, w: float) -> MB.Anchor:
    """GatedLearner update: queue the row (folded at the end of the tick,
    vectorised across entities; see m_baseline.queue)."""
    return MB.queue(state, row, w, cap=MB.CAP_CURRENT)


def _update_ref(state: MB.Anchor, row: MB.Row, w: float) -> MB.Anchor:
    """Reference admission: trust == 1 and eligible (decided at first fetch)."""
    if not row.elig or not (float(w) >= 1.0 - 1e-6):
        return state
    return MB.queue(state, row, 1.0, cap=MB.CAP_REFERENCE, drift=row.drift)


def _init_cur() -> MB.Anchor:
    return MB.new_anchor(week=True, select=True)


def _init_ref() -> MB.Anchor:
    return MB.new_anchor(week=False, select=False, hl_days=MB.HL_REF_DAYS)


def _control_sig(control: Any) -> Optional[Tuple]:
    """Directives whose change must reach the reference learner promptly."""
    if control is None:
        return None
    d = G.control_directives(control)
    return (d["rollback_to"], d["release"], d["rebase_from"], d["frozen"], d["version"])


def _abnormal(value: Any) -> bool:
    if isinstance(value, Mapping):
        st = value.get("state")
        return st is not None and str(st).lower() in ABNORMAL_REGIMES
    return False


def _eligible(ts: float, intervals: Sequence[Tuple[float, float]]) -> bool:
    """No incident / abnormal-regime interval intersects [ts - 24 h, ts + 24 h]."""
    lo, hi = ts - REF_WINDOW_S, ts + REF_WINDOW_S
    return not any(a <= hi and b >= lo for a, b in intervals)


def _finite_list(v: np.ndarray) -> List[float]:
    v = np.asarray(v, dtype=np.float64)
    return np.where(np.isfinite(v), v, 0.0).tolist()


class BaselineEngine(Engine):
    name = "behavior.baseline"
    layer = "behavior"
    consumes = [NAT, ACTIVE, "feature.tctx", "feature.expo", "behavior.trust",
                "behavior.trust_prov", "behavior.quarantine", RISK, REGIME, "model.class",
                "model.control", "model.link", "incidents"]
    produces = [MB.MODEL, "checkpoint:" + LEARNER_CUR, "checkpoint:" + LEARNER_REF,
                "profile.baseline_median", "profile.baseline_mad", "profile.seasonal",
                "profile.extra.maturity"]
    description = ("Two-anchor (rate-capped current, 24-h delayed reference / golden) seasonal "
                   "conjugate baseline per feature family with entity -> class -> system -> org "
                   "empirical-Bayes backoff; trust-gated, checkpointed, reversible learning.")
    interval = 1

    def __init__(self, tier_period_s: float = TIER_PERIOD_S, ref_period_s: float = REF_PERIOD_S,
                 **params: Any) -> None:
        super().__init__(**params)
        self.tier_period_s = float(tier_period_s)
        self.ref_period_s = float(ref_period_s)
        self._cfg: Dict[str, Any] = {}
        self._dt = 900.0
        self._now = 0.0
        self._tctx_cache: Dict[float, Dict[str, Any]] = {}
        self._ref_ctx: Optional[Tuple[Dict[str, Any], List[Tuple[float, float]], float]] = None
        self._tier_sig: Optional[Tuple] = None
        self._tier_ver = 0
        self._cur = G.GatedLearner(
            name=LEARNER_CUR, init=_init_cur, update=_update_cur, fetch=self._fetch_cur,
            dump=MB.dump, load=MB.load, merge=MB.merge, on_rebase=self._on_rebase_cur,
            ckpt_every_s=G.CKPT_EVERY_S)
        self._ref = G.GatedLearner(
            name=LEARNER_REF, init=_init_ref, update=_update_ref, fetch=self._fetch_ref,
            dump=MB.dump, load=MB.load, on_rebase=MB.on_rebase_reference, d_min_s=DAY,
            ckpt_every_s=G.CKPT_EVERY_REF_S)

    # ------------------------------------------------------------------- run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        now, dt = float(ctx.now), float(ctx.window_s)
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError(f"BaselineEngine: bad ctx.window_s {ctx.window_s!r}")
        self._cfg, self._dt, self._now = ctx.config, dt, now
        self._cur.d_min_s = float(ctx.config.get("D_min_s") or G.D_MIN_S)
        if len(self._tctx_cache) > _TCTX_CACHE_MAX:
            self._tctx_cache.clear()
        tc_now = self._tctx(now, dt)
        # the reference learner runs in hourly batches, all entities in the
        # same tick so its rows are folded vectorised across entities
        ref_due = self.entity_due(("__reference__",), now, self.ref_period_s)
        recs: List[Tuple[str, str, Dict[str, Any]]] = []
        for s in store.systems():
            ents = store.entities(s)
            if not ents:
                continue
            incs = [(i.entity, tuple(i.entities), float(i.opened), float(i.last_seen), i.status)
                    for i in store.incidents(system=s, since=now - 2.0 * REF_WINDOW_S)]
            for e in ents:
                recs.append((s, e, self._learn(ctx, s, e, now, dt, incs, ref_due)))
        # fold every queued row: one round per queued row, all anchors at once
        MB.flush_many([m[k] for _, _, m in recs for k in ("current", "reference")])
        for s, e, model in recs:
            self._publish(store, s, e, model, now)
        self._profiles(store, recs, tc_now, now)
        if self._tiers_due(store, now):
            self._refresh_tiers(store, now)
        return len(recs)

    # ------------------------------------------------------------ per entity
    def _learn(self, ctx: Context, s: str, e: str, now: float, dt: float,
               incs: List[Tuple], ref_due: bool) -> Dict[str, Any]:
        """Step both gated learners (their rows are queued on the anchors)."""
        store = ctx.store
        model = store.get_model(s, e, MB.MODEL)
        if not _valid(model):
            model = new_model()
        training = bool(ctx.training)
        control = store.get_model(s, e, G.CONTROL_MODEL)
        lw = store.last_write_ts(s, e, ACTIVE)

        # current anchor: every tick, rows t - D
        cur, gate = model["current"], model["gate"]
        front = G.commit_frontier(now, dt, self._cur.d_min_s)
        if control is not None or (lw is not None and gate.last_ts < min(lw, front)):
            cur, gate = self._cur.step(store, s, e, cur, gate, now, dt, training)
        cur, gate = self._cur.seed_from_link(store, s, e, cur, gate,
                                             lambda a: _other_current(store, s, a))
        model["current"], model["gate"] = cur, gate

        # reference anchor: hourly batches of rows t - 24 h (a new directive at once)
        sig = _control_sig(control)
        gref = model["gate_ref"]
        if ref_due or sig != model["ctl_sig"]:
            model["ctl_sig"] = sig
            front_r = G.commit_frontier(now, dt, DAY)
            if control is not None or (lw is not None and gref.last_ts < min(lw, front_r)):
                ivs = self._ref_marks(store, s, e, model, now, incs)
                self._ref_ctx = (model, ivs, float(gate.allow_drift))
                try:
                    ref, gref = self._ref.step(store, s, e, model["reference"], gref, now, dt,
                                               training)
                finally:
                    self._ref_ctx = None
                model["reference"], model["gate_ref"] = ref, gref
                el = model["ref_elig"]
                if el and min(el) < now - REF_ELIG_KEEP_S:
                    model["ref_elig"] = {k: v for k, v in el.items() if k >= now - REF_ELIG_KEEP_S}
        return model

    def _publish(self, store: Any, s: str, e: str, model: Dict[str, Any], now: float) -> None:
        """Golden snapshot, contract fields and put_model."""
        gate = model["gate"]
        self._golden(store, s, e, model, now)
        model["version"], model["branch"] = int(gate.version), int(gate.branch)
        model["allow_drift"] = float(gate.allow_drift)
        model["held"] = gate.held                 # rows (ts, w_eff, w_prov) held for B28
        model["n_eff"] = MB.n_eff(model["current"])
        store.put_model(s, e, MB.MODEL, model, version=int(gate.version), ts=now)

    # --------------------------------------------------------------- fetch
    def _tctx(self, ts: float, dt: float) -> Dict[str, Any]:
        """feature.tctx of tick ts (same function and config B01 uses)."""
        tc = self._tctx_cache.get(ts)
        if tc is None:
            tc = self._tctx_cache[ts] = TB.tctx_from_config(ts, self._cfg, dt)
        return tc

    def _row_dt(self, store: Any, s: str, e: str, ts: float) -> float:
        """Real exposure of the row: the gap to the previous feature tick
        (the cadence may switch 900 -> 60), else this tick's dt."""
        if store.vec_at(s, e, ACTIVE, ts - self._dt) is not None:
            return self._dt                     # steady cadence: the common case
        t, _ = store.vec_range(s, e, ACTIVE, ts - ROW_DT_MAX - 1.0, ts - 1e-6)
        if len(t):
            gap = ts - float(t[-1])
            if 0.0 < gap <= ROW_DT_MAX:
                return gap
        return self._dt

    def _fetch_cur(self, store: Any, s: str, e: str, ts: float) -> Optional[MB.Row]:
        """Row of an active tick (inactive / missing ticks are skipped)."""
        a = store.vec_at(s, e, ACTIVE, ts)
        if a is None or not float(a[0]) > 0.5:
            return None
        nat = store.vec_at(s, e, NAT, ts)
        if nat is None:
            return None
        dt = self._row_dt(store, s, e, ts)
        return MB.make_row(ts, nat, dt, self._tctx(ts, dt))

    def _fetch_ref(self, store: Any, s: str, e: str, ts: float) -> Optional[MB.Row]:
        """Current row plus the reference admission decision (no incident or
        abnormal regime within +-24 h), cached per row ts so a replay makes
        the same decision; an ineligible row is skipped outright."""
        if self._ref_ctx is None:
            return self._fetch_cur(store, s, e, ts)
        model, ivs, drift = self._ref_ctx
        el = model["ref_elig"]
        ok = el.get(ts)
        if ok is None:
            ok = el[ts] = _eligible(ts, ivs)
        if not ok:
            return None
        row = self._fetch_cur(store, s, e, ts)
        return row._replace(drift=drift) if row is not None and drift else row

    def _on_rebase_cur(self, state: MB.Anchor, tau: float) -> MB.Anchor:
        """ACCEPTED: the new regime's rows (held ones and the next day's) are
        committed without the rate cap."""
        return MB.on_rebase_current(state, tau, until=self._now + DAY)

    # ------------------------------------------------------ reference marks
    def _ref_marks(self, store: Any, s: str, e: str, model: Dict[str, Any], now: float,
                   incs: List[Tuple]) -> List[Tuple[float, float]]:
        """Intervals that exclude rows from the reference: incidents of the
        entity (or of a class incident it belongs to) and abnormal regimes."""
        ivs: List[Tuple[float, float]] = []
        for ent, ents, opened, last, status in incs:
            if ent == e or e in ents:
                b = now if status == "open" else last
                ivs.append((opened, max(b, opened)))
        reg = model["reg"]
        self._scan_regime(store, s, e, reg, now)
        ivs.extend((float(a), float(b)) for a, b in reg["iv"])
        if reg["since"] is not None:
            ivs.append((float(reg["since"]), now))
        return ivs

    @staticmethod
    def _scan_regime(store: Any, s: str, e: str, reg: Dict[str, Any], now: float) -> None:
        """Fold behavior.regime points written since the last scan into
        abnormal intervals (the regime is a step function of its points)."""
        last = reg["scan"]
        n = 64
        pts = store.derived_tail(s, e, REGIME, n)
        while pts and pts[0].ts > last and len(pts) == n and n < 65536:
            n *= 4
            pts = store.derived_tail(s, e, REGIME, n)
        for p in pts:
            if p.ts <= last:
                continue
            abn = _abnormal(p.value)
            if abn and reg["since"] is None:
                reg["since"] = float(p.ts)
            elif not abn and reg["since"] is not None:
                reg["iv"].append([reg["since"], float(p.ts)])
                reg["since"] = None
        if pts:
            reg["scan"] = max(last, float(pts[-1].ts))
        if reg["iv"] and reg["iv"][0][1] < now - REGIME_KEEP_S:
            reg["iv"] = [iv for iv in reg["iv"] if iv[1] >= now - REGIME_KEEP_S]

    # ---------------------------------------------------------------- golden
    @staticmethod
    def _golden(store: Any, s: str, e: str, model: Dict[str, Any], now: float) -> None:
        """Weekly: snapshot the reference when the finished week's max risk
        stays below 30; golden = cell-wise median of the last 4 snapshots."""
        g = model["golden"]
        ref = model["reference"]
        if ref.n_reset != g["n_reset"]:                 # reset by a rebase: new regime
            g.update(snaps=[], stats=None, n_reset=ref.n_reset)
        wk = int(now // WEEK)
        if g["week"] is None:
            g["week"] = wk
            return
        if wk <= g["week"]:
            return
        g["week"] = wk
        if MB.n_eff(ref) < GOLDEN_MIN_NEFF:
            return
        _, M = store.vec_since(s, e, RISK, now - WEEK)
        r = np.asarray(M, dtype=np.float64).reshape(-1)
        r = r[np.isfinite(r)]
        if r.size and float(r.max()) >= GOLDEN_MAX_RISK:
            return
        St = MB.true_stats(ref)
        if St is None:
            return
        g["snaps"] = (g["snaps"] + [{"ts": now, "stats": St.astype(np.float32)}])[-GOLDEN_SNAPS:]
        g["stats"] = MB.median_select([sn["stats"] for sn in g["snaps"]])

    # --------------------------------------------------------------- profile
    def _profiles(self, store: Any, recs: List[Tuple[str, str, Dict[str, Any]]],
                  tc: Mapping[str, Any], now: float) -> None:
        """Legacy profile fields at the current bucket (feature.vec space) and
        extra.maturity, when the bucket or the version changes and at least
        hourly; the predictives are computed per system in one batch."""
        key_b = int(tc["bin48"])
        due: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        for s, e, model in recs:
            if model["pb"] != (key_b, model["version"]) or now - model["pb_ts"] >= PROFILE_PERIOD_S:
                due.setdefault(s, []).append((e, model))
        for s, items in due.items():
            med, sd = MB.profile_many(store, s, [e for e, _ in items], [m for _, m in items], tc)
            for i, (e, model) in enumerate(items):
                prof = store.profile(s, e) or EntityProfile(system=s, entity=e, updated=now)
                prof.baseline_median = _finite_list(med[i])
                prof.baseline_mad = _finite_list(sd[i])
                if (now - model.get("ps_ts", -math.inf) >= SEASONAL_PERIOD_S or not prof.seasonal
                        or "maturity" not in prof.extra):
                    prof.seasonal = MB.seasonal_curves(model, str(tc["day_type"]), SEASONAL_FEATURES)
                    prof.extra["maturity"] = MB.maturity(model)
                    model["ps_ts"] = now
                prof.sample_count = int(round(model["n_eff"]))
                store.put_profile(prof)
                model["pb"], model["pb_ts"] = (key_b, model["version"]), now

    # ----------------------------------------------------------------- tiers
    def _tiers_due(self, store: Any, now: float) -> bool:
        sig = (m_class.version(store), tuple(store.systems()))
        due = self.entity_due(("__tiers__",), now, self.tier_period_s)
        if sig != self._tier_sig:
            self._tier_sig = sig
            return True
        return due

    def _refresh_tiers(self, store: Any, now: float) -> None:
        """Class, system and org tiers: sums of members' current statistics at
        now, EB kappas from their children and leave-one-out backoff chains
        from the org down (lib/m_baseline.join)."""
        self._tier_ver += 1
        ver = self._tier_ver
        zeros = np.zeros((48, MB.L_STATS))
        per_sys: Dict[str, Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, List[str]], float]] = {}
        for s in store.systems():
            ents: Dict[str, np.ndarray] = {}
            neff = 0.0
            for e in store.entities(s):
                m = store.get_model(s, e, MB.MODEL)
                if _valid(m):
                    St = MB.true_stats(m["current"], now)
                    if St is not None:
                        ents[e] = St
                        neff += MB.n_eff(m)
            S_sys = sum(ents.values()) if ents else zeros
            classes: Dict[str, List[str]] = {}
            for key in m_class.all_class_keys(store, s, min_members=m_class.MIN_MEMBERS):
                if key.startswith(CLASS_PREFIX) and not key.startswith((STATIC_PREFIX, POOL_PREFIX)):
                    classes[key] = m_class.class_members(store, s, key)
            per_sys[s] = (ents, S_sys, classes, neff)
        S_org = sum(v[1] for v in per_sys.values()) if per_sys else zeros
        kap_org = MB.eb_kappa([v[1].sum(axis=0) for v in per_sys.values()])
        h1 = np.ones((48, MB.NF))
        store.put_model(ORG[0], ORG[1], MB.MODEL, _tier_dict(
            "org", S_org, S_org, h1, kap_org, now, ver, sorted(per_sys),
            sum(v[3] for v in per_sys.values())), version=ver, ts=now)
        for s, (ents, S_sys, classes, neff) in per_sys.items():
            E_s, h_s = MB.join(S_sys, S_org, h1, kap_org, loo=True)
            kap_ent = MB.eb_kappa([S.sum(axis=0) for S in ents.values()])
            cstats = {k: [ents[x] for x in mem if x in ents] for k, mem in classes.items()}
            kap_cls = MB.eb_kappa([sum(v).sum(axis=0) for v in cstats.values() if v])
            store.put_model(s, SYSTEM_KEY, MB.MODEL, _tier_dict(
                "system", S_sys, E_s, h_s, kap_ent, now, ver, sorted(ents), neff,
                kappa_cls=kap_cls), version=ver, ts=now)
            for key, mem in classes.items():
                sts = cstats[key]
                S_c = sum(sts) if sts else zeros
                E_c, h_c = MB.join(S_c, E_s, h_s, kap_cls, loo=True)
                store.put_model(s, key, MB.MODEL, _tier_dict(
                    "class", S_c, E_c, h_c, MB.eb_kappa([S.sum(axis=0) for S in sts]), now, ver,
                    list(mem), sum(MB.n_eff(store.get_model(s, x, MB.MODEL)) for x in mem
                                   if x in ents)), version=ver, ts=now)


def _tier_dict(tier: str, S: np.ndarray, E: np.ndarray, h: np.ndarray, kappa: np.ndarray,
               now: float, ver: int, members: List[str], n_eff: float,
               kappa_cls: Optional[np.ndarray] = None) -> Dict[str, Any]:
    return {"fmt": MB.FMT, "tier": tier, "ts": float(now), "version": int(ver),
            "stats": S, "E": E, "h": h, "kappa": kappa,
            "kappa_cls": kappa if kappa_cls is None else kappa_cls,
            "members": members, "n_eff": float(n_eff)}


def _other_current(store: Any, s: str, entity: str) -> Optional[MB.Anchor]:
    """Link seeding source: the linked entity's current anchor (read only)."""
    m = store.get_model(s, entity, MB.MODEL)
    return m["current"] if _valid(m) and not m["current"].empty else None
