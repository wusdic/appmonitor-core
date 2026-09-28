"""SequenceEngine (B10) — each IP's own grammar over fine-grained action tokens.

Why: behaviour has grammar. A clerk logs in, opens the dashboard, lists
orders and views a few; a bulk export straight after login, a replayed
credential loop or a scripted walk over admin pages is out of grammar even
when every volume metric is normal. v1 modelled one dominant signature
category per tick as a first-order Markov chain keyed by the archetype NAME:
unseen source states scored 0 (novel-state blindness), a fixed 4-bit
threshold fired on high-entropy users, and renaming a role reset the model.
This engine replaces it with:

  * Streams. (A) act.stream symbols 'template|outcome' (lib/m_seq.SymbolMap;
    tokens with a system count < 3 become '{rare:<channel>}'), counts
    reweighted by 1/act.stream_frac. (B) the bag of lib-4 categories of each
    earlier tick (confidence >= 0.6, one-tick lag) as one 'cat:a+b' symbol.
  * Sessions. The idle threshold G_e is the valley of a decayed 32-bin
    log-gap histogram (m_seq.gap_valley), clipped to [2 min, 2 h] and
    published as model.seq.session_gap (D2, B11, B20 read it). Gaps longer
    than R2's 30-s cut are skipped on sampled ticks (they may span dropped
    sessions). A session starts from the '{bos}' context.
  * Model. PPM-C order 3 per entity, 30-d decay (lib/ppm). On escape it backs
    off to the role-class PPM model.seq@(s, 'class:<rid>') -- a stable id from
    m_class.class_key, never a name -- then to the system unigram with
    Good-Turing novel mass N1/N over the REAL vocabulary (model.template
    size), so unseen tokens and unseen source states always get surprisal
    > 0. Class and system tiers are rebuilt hourly by merging the member
    entity models, so they inherit the members' trust gating and rollbacks.
  * Scores, per tick with tokens, against the model as of the last commit
    (scores are relative to the entity's own entropy rate mu_e, so a
    high-entropy user is not a permanent false positive):
      E     = max over (a) complete W = 8-token blocks: mean surprisal - mu_e,
              (b) every session touched this tick: NLL* - mu_e with
              NLL* = (L NLL + 5 mu_e)/(L + 5) (short sessions shrink to mu_e),
              (c) stream B: the same shrunk excess over its entropy rate;
      CUSUM = within-session max(0, C + s_i - (mu_e + 0.5 sigma_e)), h = 8 bits;
      score.seq = max(E, CUSUM / h * 8)            (instantaneous)
      score.dwell = -log10 min(1, n * min_run P(L >= l)): semi-Markov dwell,
              a moment-matched NB per token family on run lengths
      behavior.seq.class_llr = mean_i (log2 P_e - log2 P_class) bits/token.
    Axes: 'sequence', plus 'credential' when a surprising token (or the
    longest-tail run) belongs to an auth family (login, sso, token, ssh ...).
  * Learning is delayed, trust-gated and reversible (lib/gating.GatedLearner,
    contract H) and honours model.control. act.stream is kept 1 h while
    commits lag D (up to 4 h at 3600-s ticks) and releases / rollbacks reach
    back days, so each tick's compact row (symbols, recorded surprisal, gap
    bins, completed dwell runs, stream-B bags) is kept in the model: pending
    rows until committed, held rows while held, committed rows for 26 h so a
    rollback replays exactly from its checkpoint (deeper replays restore the
    checkpoint and skip the rows no longer kept); <= 64k tokens per entity.
    The clock of candidate rows is this engine's own behavior.seq.class_llr
    ring, written at every tick that produced a row.

B10 emits no events (contract F retired the legacy 'sequence' kind): it
writes detector scores that B24 calibrates per entity. ctx.training learns
as usual (missing trust counts as 1). A stale act.stream on an active tick,
or an R2 failure this tick, gives NaN plus behavior.degraded.
"""
from __future__ import annotations

import math
import pickle
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import EntityProfile
from .lib import emit
from .lib import gating as G
from .lib import m_class
from .lib import m_seq
from .lib import m_template
from .lib import ppm as P
from .lib.classkeys import SYSTEM_KEY, role_key

MODEL = m_seq.MODEL
CLASS_LLR = "behavior.seq.class_llr"
R2_ENGINE = "raw.action_token"
LEARNER = "seq"

ORDER = 3
CAT_ORDER = 2
PPM_HALF_LIFE_S = P.PPM_HALF_LIFE_S       # 30 d
GAP_HALF_LIFE_S = 7 * 86400.0
DWELL_HALF_LIFE_S = P.PPM_HALF_LIFE_S
W_TOKENS = 8                              # excess window
CUSUM_H = 8.0                             # bits
MIN_TOK_STATS = 50.0                      # entity tokens before its surprisal enters mu_e
MIN_STATS_W = 20.0                        # recorded tokens before an entropy rate is used
SAMPLED_GAP_MAX_S = 30.0                  # R2's intra-tick session cut (m_template.SESSION_GAP_S)
REPLAY_KEEP_S = 26 * 3600.0
ROW_TOKEN_CAP = 65536
MAX_FAMILIES = 256
TIER_REFIT_S = 3600.0
GAP_REFRESH_S = 3600.0                    # session_gap re-derived at most hourly (row time)
PORTRAIT_REFIT_S = 3600.0
_RENORM_G = 32.0


class _Row(NamedTuple):
    """One tick of stream data, kept until committed (gating fetch)."""
    ts: float
    w: float                                    # 1 / act.stream_frac
    segs: Tuple[Tuple[tuple, tuple], ...]       # (history, symbols) per session segment
    surpr: np.ndarray                           # float32 bits per token, NaN = not recorded
    gaps: Tuple[Tuple[int, float], ...]         # (log-gap bin, count)
    runs: Tuple[Tuple[str, int], ...]           # completed dwell runs (family, length)
    cats: tuple                                 # stream-B symbols
    cat_hist: tuple
    cat_surpr: np.ndarray
    n_tok: int


# ============================================================ learner state
def _new_acc(h: float, bins: int = 0) -> Dict[str, Any]:
    acc: Dict[str, Any] = {"h": h, "t_ref": math.nan, "g": 0.0}
    if bins:
        acc["v"] = np.zeros(bins)
    else:
        acc["fam"] = {}
    return acc


def _acc_scale(acc: Dict[str, Any], ts: float) -> float:
    """Stored-scale multiplier of a row at ts (lazy decay, as lib/ppm): the
    clock only moves forward, an older row arrives decayed."""
    h = acc["h"]
    if not math.isfinite(acc["t_ref"]):
        acc["t_ref"], acc["g"] = ts, 0.0
        return 1.0
    g_ts = (ts - acc["t_ref"]) / h
    if g_ts > acc["g"]:
        acc["g"] = g_ts
        if g_ts > _RENORM_G:                      # rescale to the clock
            f = 2.0 ** (-g_ts)
            if "v" in acc:
                acc["v"] = acc["v"] * f
            else:
                acc["fam"] = {k: [x * f for x in v] for k, v in acc["fam"].items()
                              if v[0] * f > 1e-12}
            acc["t_ref"] += g_ts * h
            acc["g"] = g_ts = 0.0
    return 2.0 ** g_ts


def _acc_merge(own: Dict[str, Any], other: Dict[str, Any], w: float) -> None:
    """own += w * other, other folded at its own clock (decayed if older)."""
    if not math.isfinite(other.get("t_ref", math.nan)):
        return
    t_o = other["t_ref"] + other["g"] * other["h"]
    f = w * _acc_scale(own, t_o) * 2.0 ** (-other["g"])
    if "v" in own:
        own["v"] = own["v"] + f * np.asarray(other["v"], dtype=np.float64)
    else:
        fam = own["fam"]
        for k, v in other["fam"].items():
            cur = fam.get(k)
            if cur is None:
                fam[k] = [f * x for x in v]
            else:
                for i in range(3):
                    cur[i] += f * v[i]
        _cap_families(fam)


def _cap_families(fam: Dict[str, List[float]]) -> None:
    if len(fam) > MAX_FAMILIES:
        keep = sorted(fam.items(), key=lambda kv: -kv[1][0])[:MAX_FAMILIES]
        fam.clear()
        fam.update(keep)


def _init_state(order: int = ORDER, cat_order: int = CAT_ORDER) -> Dict[str, Any]:
    return {
        "ppm": P.PPMModel(order=order, half_life_s=PPM_HALF_LIFE_S),
        "ppm_cat": P.PPMModel(order=cat_order, half_life_s=PPM_HALF_LIFE_S),
        "gap": _new_acc(GAP_HALF_LIFE_S, m_seq.GAP_BINS),
        "dwell": _new_acc(DWELL_HALF_LIFE_S),
        "session_gap": m_seq.DEFAULT_SESSION_GAP_S,
    }


def _update(state: Dict[str, Any], row: _Row, w: float) -> Dict[str, Any]:
    """Fold one committed row with weight w_eff * (1 / stream_frac). In place;
    deterministic in (state, row, w) so checkpoint + replay is exact."""
    ww = float(w) * row.w
    if not (ww > 0.0 and math.isfinite(ww)):
        return state
    ts = row.ts
    p = state["ppm"]
    for hist, syms in row.segs:
        P.update(p, syms, ww, ts, history=hist)
    if row.n_tok:
        P.record_surprisal(p, row.surpr, ww)
    if row.cats:
        P.update(state["ppm_cat"], row.cats, ww, ts, history=row.cat_hist)
        P.record_surprisal(state["ppm_cat"], row.cat_surpr, ww)
    if row.gaps:
        acc = state["gap"]
        sc = ww * _acc_scale(acc, ts)
        v = acc["v"]
        for j, c in row.gaps:
            v[j] += sc * c
        # the decayed histogram moves slowly: re-derive the valley hourly (in row
        # time, so replay is exact) and on every row while it is still young
        h = v * 2.0 ** (-acc["g"])
        if ts - state.get("gap_at", -math.inf) >= GAP_REFRESH_S or \
                h.sum() < 4.0 * m_seq.GAP_MIN_WEIGHT:
            state["session_gap"] = m_seq.gap_valley(h)
            state["gap_at"] = ts
    if row.runs:
        acc = state["dwell"]
        sc = ww * _acc_scale(acc, ts)
        fam = acc["fam"]
        for f, L in row.runs:
            x = float(L - 1)
            cur = fam.get(f)
            if cur is None:
                fam[f] = [sc, sc * x, sc * x * x]
            else:
                cur[0] += sc
                cur[1] += sc * x
                cur[2] += sc * x * x
        _cap_families(fam)
    return state


def _merge_state(own: Dict[str, Any], other: Dict[str, Any], w: float) -> Dict[str, Any]:
    """own + w * other in sufficient-statistic space (link seeding, tier builds)."""
    P.merge(own["ppm"], other["ppm"], w)
    P.merge(own["ppm_cat"], other["ppm_cat"], w)
    _acc_merge(own["gap"], other["gap"], w)
    _acc_merge(own["dwell"], other["dwell"], w)
    acc = own["gap"]
    own["session_gap"] = m_seq.gap_valley(acc["v"] * 2.0 ** (-acc["g"]))
    return own


def _dump(state: Dict[str, Any]) -> bytes:
    # bytes: GatedLearner deep-copies blobs, which is O(1) for bytes
    return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)


def _load(blob: Any) -> Dict[str, Any]:
    return pickle.loads(blob) if isinstance(blob, (bytes, bytearray)) else blob


def _rate(p: Optional[P.PPMModel]) -> Tuple[float, float]:
    """Entropy rate once at least MIN_STATS_W tokens were recorded."""
    if p is None or len(p.stats) != 3 or not p.stats[0] >= MIN_STATS_W:
        return math.nan, math.nan
    return P.entropy_rate(p)


def _new_run() -> Dict[str, Any]:
    """Per-entity scoring state (not learned, not rolled back)."""
    return {"last_ts": math.nan, "last_full": False, "hist": (m_seq.BOS,), "sn": 0,
            "ss": 0.0, "blk": [], "c": 0.0, "fam": None, "flen": 0, "fevald": 0,
            "cat_ts": math.nan, "chist": ()}


class _Tiers:
    """Per-system tier models and role sizes read once per tick."""

    def __init__(self, store, system: str) -> None:
        self.store, self.system = store, system
        self.sys = m_seq.get(store, system, SYSTEM_KEY)
        self._cls: Dict[str, Optional[Dict[str, Any]]] = {}
        self._n_members: Dict[str, int] = {}

    def class_key(self, entity: str) -> Optional[str]:
        """m_class.class_key with the member count cached per role (O(1) per entity)."""
        rid = m_class.role_id(self.store, self.system, entity)
        if rid is None:
            return None
        n = self._n_members.get(rid)
        if n is None:
            n = self._n_members[rid] = len(m_class.members(self.store, self.system, rid))
        return role_key(rid) if n >= m_class.MIN_MEMBERS else None

    def cls(self, key: Optional[str]) -> Optional[Dict[str, Any]]:
        if not key:
            return None
        if key not in self._cls:
            self._cls[key] = m_seq.get(self.store, self.system, key)
        return self._cls[key]


def _nonempty(p: Optional[P.PPMModel]) -> Optional[P.PPMModel]:
    return p if p is not None and p.counts else None


# ================================================================== engine
class SequenceEngine(Engine):
    name = "behavior.sequence"
    layer = "behavior"
    consumes = ["act.stream", "act.stream_frac", "act.events", "match.*", "model.template",
                "model.class", "behavior.trust", "behavior.trust_prov", "behavior.quarantine",
                "model.control", "model.link"]
    produces = ["model.seq", "behavior.score", "behavior.pm", "behavior.axes",
                "behavior.degraded", CLASS_LLR, "profile.extra.sequence"]
    description = ("PPM-C grammar over action tokens per IP with role-class and system "
                   "backoff; excess surprisal over the entity's entropy rate, session "
                   "CUSUM and semi-Markov dwell.")
    interval = 1

    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        self.window = int(params.get("window", W_TOKENS))
        self.cusum_h = float(params.get("cusum_h", CUSUM_H))
        self.tier_refit_s = float(params.get("tier_refit_s", TIER_REFIT_S))
        self._learners: Dict[float, G.GatedLearner] = {}
        self._rows: Dict[float, _Row] = {}
        self._held: Dict[float, _Row] = {}
        # perf: tier rebuilds whose members did not change are reused (_build_tier)
        self._mark: Dict[Tuple[str, str], Tuple[Any, ...]] = {}
        self._rev: Dict[Tuple[str, str], int] = {}
        self._tier_sig: Dict[Tuple[str, str], Tuple[Any, ...]] = {}

    # ----------------------------------------------------------------- run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        now, dt = float(ctx.now), float(ctx.window_s)
        d_min = ctx.config.get("D_min_s", G.D_MIN_S)
        lrn = self._learner(float(d_min) if d_min else G.D_MIN_S)
        r2_failed = store.engine_failed(R2_ENGINE, now)
        n = 0
        for s in store.systems():
            smap = m_seq.SymbolMap.from_store(store, s)
            tiers = _Tiers(store, s)
            used: set = set()
            for e in store.entities(s):
                n += self._entity(ctx, lrn, s, e, smap, tiers, used, r2_failed)
            self._refit_tiers(store, s, now, used)
        return n

    def _learner(self, d_min_s: float) -> G.GatedLearner:
        lrn = self._learners.get(d_min_s)
        if lrn is None:
            lrn = self._learners[d_min_s] = G.GatedLearner(
                name=LEARNER, init=_init_state, update=_update, fetch=self._fetch,
                dump=_dump, load=_load, merge=_merge_state, d_min_s=d_min_s,
                ckpt_every_s=G.CKPT_EVERY_S, clock=CLASS_LLR)
        return lrn

    def _fetch(self, store, s: str, e: str, ts: float) -> Optional[_Row]:
        # rows of the entity being stepped (set by _entity before step)
        r = self._rows.get(ts)
        return r if r is not None else self._held.get(ts)

    # -------------------------------------------------------------- entity
    def _entity(self, ctx: Context, lrn: G.GatedLearner, s: str, e: str,
                smap: m_seq.SymbolMap, tiers: _Tiers, used: set, r2_failed: bool) -> int:
        store, now, dt = ctx.store, float(ctx.now), float(ctx.window_s)
        model = m_seq.get(store, s, e)
        if model is not None and "_state" not in model:
            model = None                              # not ours / foreign layout
        run = model["_run"] if model is not None else _new_run()
        tst, syms, fams, auth = m_seq.stream_symbols(store, s, e, now, smap)
        cats = self._cat_tokens(store, s, e, now, dt, run)
        degraded = not syms and (r2_failed or _fresh_pos(store, s, e, "act.events", now))
        if model is None and not syms and not cats:
            if degraded:
                self._write_degraded(store, s, e, now, dt, r2_failed)
            return 0
        if model is None:
            model = {"kind": "entity", "_state": _init_state(), "_gate": G.GateState(),
                     "_run": run, "_rows": {}, "_held": {}, "_ntok": 0}
        state, gate = model["_state"], model["_gate"]
        ck = tiers.class_key(e)
        if ck:
            used.add(ck)
        n = 0
        out: Optional[Dict[str, Any]] = None
        if syms or cats:
            row, out = self._score(state, run, tiers, ck, smap, now, _frac(store, s, e, now),
                                   tst, syms, fams, auth, cats)
            model["_rows"][now] = row
            model["_ntok"] += row.n_tok + len(row.cats)
            if degraded:
                self._write_degraded(store, s, e, now, dt, r2_failed)
            else:
                self._write(store, s, e, now, dt, out)
            store.add_vec(s, e, CLASS_LLR, now, [out["class_llr"]], window_s=int(dt))
            n = 1
        elif degraded:
            self._write_degraded(store, s, e, now, dt, r2_failed)
        # learn: commit rows <= now - D (scores above used the pre-commit model)
        self._rows, self._held = model["_rows"], model["_held"]
        try:
            state, gate = lrn.step(store, s, e, state, gate, now, dt, training=ctx.training)
            state, gate = lrn.seed_from_link(store, s, e, state, gate,
                                             lambda src: _other_state(store, s, src))
        finally:
            self._rows, self._held = {}, {}
        mark = _state_mark(state, gate)
        if self._mark.get((s, e)) != mark:     # the learned state may have changed
            self._mark[(s, e)] = mark
            self._rev[(s, e)] = self._rev.get((s, e), 0) + 1
        _prune_rows(model, gate, now)
        self._publish(store, s, e, model, state, gate, run, ck, now, out)
        return n

    # ------------------------------------------------------------- scoring
    def _score(self, state: Dict[str, Any], run: Dict[str, Any], tiers: _Tiers,
               ck: Optional[str], smap: m_seq.SymbolMap, now: float, frac: float,
               tst: np.ndarray, syms: List[str], fams: List[str], auth: List[bool], cats: tuple
               ) -> Tuple[_Row, Dict[str, Any]]:
        ent = state["ppm"]
        cm, sm = tiers.cls(ck), tiers.sys
        c_ppm, s_ppm = _nonempty(m_seq.ppm(cm)), _nonempty(m_seq.ppm(sm))
        chain = [t for t in (c_ppm, s_ppm) if t is not None]
        # the real vocabulary: template size + rare symbols, at least every symbol seen
        V = max(smap.vocab_size, len(ent.counts.get((), ())) + 1,
                len(s_ppm.counts.get((), ())) + 1 if s_ppm else 0)
        mu, sd = _rate(ent)
        for tier in (c_ppm, s_ppm):
            if not math.isfinite(mu) and tier is not None:
                mu, sd = _rate(tier)
        w_frac = 1.0 / frac
        out: Dict[str, Any] = {"seq": math.nan, "dwell": math.nan, "pm_dwell": math.nan,
                               "class_llr": math.nan, "excess": math.nan, "cusum": math.nan,
                               "cred_seq": False, "cred_dwell": False, "n": len(syms)}
        segs: List[Tuple[tuple, tuple]] = []
        runs: List[Tuple[str, int]] = []
        gaps: Tuple[Tuple[int, float], ...] = ()
        surpr = np.zeros(0, dtype=np.float32)
        stats: List[float] = []
        if syms:
            G_e = float(state["session_gap"])
            dw_tier = cm if cm is not None else sm
            surpr, stats, gaps = self._score_a(run, tst, syms, fams, auth, frac < 1.0, ent,
                                               chain, c_ppm, s_ppm, V, mu, sd, G_e, segs,
                                               runs, state, dw_tier, out)
        cat_hist = tuple(run["chist"])
        cat_surpr = np.zeros(0, dtype=np.float32)
        if cats:
            e_cat, cat_surpr = self._score_b(run, state, cm, sm, cats)
            stats.append(e_cat)
        vals = [x for x in stats if math.isfinite(x)]
        out["seq"] = max(vals) if vals else math.nan
        row = _Row(now, w_frac, tuple(segs), surpr, gaps, tuple(runs), tuple(cats), cat_hist,
                   cat_surpr, len(syms))
        return row, out

    def _score_a(self, run: Dict[str, Any], tst: np.ndarray, syms: List[str], fams: List[str],
                 auth: List[bool], sampled: bool, ent: P.PPMModel, chain: List[P.PPMModel],
                 c_ppm: Optional[P.PPMModel], s_ppm: Optional[P.PPMModel], V: int, mu: float,
                 sd: float, G_e: float, segs: List, runs: List, state: Dict[str, Any],
                 dw_tier: Optional[Dict[str, Any]], out: Dict[str, Any]):
        """Stream A: surprisal, excess blocks / sessions, CUSUM, dwell, class LLR."""
        n = len(syms)
        W, K = self.window, m_seq.SHRINK_K
        have_mu = math.isfinite(mu)
        sd = sd if math.isfinite(sd) else 0.0
        ref = mu + 0.5 * sd
        cut, first_new = m_seq.session_starts(tst, G_e, run["last_ts"])
        bounds = [0] + cut.tolist() + [n]
        surpr = np.empty(n, dtype=np.float64)
        emax, cmax = -math.inf, 0.0
        llr_sum, llr_n = 0.0, 0
        dwell_p: List[Tuple[float, str]] = []
        dmodel = {"dwell": state["dwell"]}
        cls_chain = [s_ppm] if s_ppm is not None else []
        for si in range(len(bounds) - 1):
            a, b = bounds[si], bounds[si + 1]
            if si > 0 or first_new:
                self._close_run(run, runs, dwell_p, dmodel, dw_tier)
                run.update(hist=(m_seq.BOS,), sn=0, ss=0.0, blk=[], c=0.0)
            hist = tuple(run["hist"])
            seg = syms[a:b]
            bits = m_seq.loglik(ent, seg, chain, V, hist)
            surpr[a:b] = bits
            if c_ppm is not None:
                cb = P.loglik(c_ppm, seg, cls_chain, V, hist)
                llr_sum += float(np.sum(cb - bits))
                llr_n += b - a
            segs.append((hist, tuple(seg)))
            sn, ss, blk, c = run["sn"], run["ss"], run["blk"], run["c"]
            for i, x in enumerate(bits.tolist()):
                j = a + i
                ss += x
                sn += 1
                if have_mu:
                    c += x - ref
                    if c < 0.0:
                        c = 0.0
                    elif c > cmax:
                        cmax = c
                    blk.append(x)
                    if len(blk) >= W:
                        eb = sum(blk) / len(blk) - mu
                        if eb > emax:
                            emax = eb
                        blk = []
                f = fams[j]
                if f == run["fam"]:
                    run["flen"] += 1
                else:
                    self._close_run(run, runs, dwell_p, dmodel, dw_tier)
                    run["fam"], run["flen"] = f, 1
            if have_mu:
                es = (ss - sn * mu) / (sn + K)
                if es > emax:
                    emax = es
            run.update(sn=sn, ss=ss, blk=blk, c=c, hist=(hist + tuple(seg))[-ORDER:])
        # the open run at the tick end (evaluated, learned only once complete)
        if run["fam"] is not None and run["flen"] >= 2:
            p = m_seq.dwell_sf(dmodel, run["fam"], run["flen"], dw_tier)
            if p == p:
                dwell_p.append((p, run["fam"]))
            run["fevald"] = run["flen"]
        # outputs
        if have_mu:
            out["excess"] = emax if emax > -math.inf else math.nan
            out["cusum"] = cmax
            thr = mu + max(3.0 * sd, 2.0)
            out["cred_seq"] = any(u and x >= thr for u, x in zip(auth, surpr.tolist()))
        if dwell_p:
            pmin, fam = min(dwell_p)
            p = min(1.0, len(dwell_p) * pmin)
            out["pm_dwell"] = max(p, 1e-300)
            out["dwell"] = max(0.0, -math.log10(max(p, 1e-300)))
            out["cred_dwell"] = m_seq.is_auth(fam) and p < 0.5
        else:
            out["dwell"] = 0.0                     # evaluated: no run longer than 1
        if llr_n:
            out["class_llr"] = llr_sum / llr_n
        stats = [out["excess"], out["cusum"] / self.cusum_h * 8.0] if have_mu else []
        # gaps for the session histogram (learned at commit)
        gl: List[float] = []
        last = run["last_ts"]
        if math.isfinite(last) and not sampled and run["last_full"]:
            gl.append(float(tst[0]) - last)
        d = np.diff(tst)
        gl.extend((d[d <= SAMPLED_GAP_MAX_S] if sampled else d).tolist())
        gaps: Tuple[Tuple[int, float], ...] = ()
        if gl:
            cnt = np.bincount(m_seq.gap_bins(gl), minlength=m_seq.GAP_BINS)
            gaps = tuple((int(j), float(cnt[j])) for j in np.flatnonzero(cnt))
        run["last_ts"], run["last_full"] = float(tst[-1]), not sampled
        # recorded surprisal feeds mu_e only once the entity model is mature enough
        rec = surpr.astype(np.float32)
        if not ent.n_tokens >= MIN_TOK_STATS:
            rec[:] = np.nan
        return rec, stats, gaps

    @staticmethod
    def _close_run(run: Dict[str, Any], runs: List, dwell_p: List, dmodel: Dict[str, Any],
                   tier: Optional[Dict[str, Any]]) -> None:
        """Complete the open run: learned at commit; evaluated unless it was
        already evaluated at this length at the previous tick end."""
        f, L = run["fam"], run["flen"]
        if f is None or L < 1:
            return
        runs.append((f, int(L)))
        if L >= 2 and L > run.get("fevald", 0):
            p = m_seq.dwell_sf(dmodel, f, L, tier)
            if p == p:
                dwell_p.append((p, f))
        run["fam"], run["flen"], run["fevald"] = None, 0, 0

    def _score_b(self, run: Dict[str, Any], state: Dict[str, Any], cm, sm, cats: tuple
                 ) -> Tuple[float, np.ndarray]:
        """Stream B: shrunk excess of this tick's category bags."""
        ent = state["ppm_cat"]
        c_p, s_p = _nonempty(m_seq.ppm_cat(cm)), _nonempty(m_seq.ppm_cat(sm))
        chain = [t for t in (c_p, s_p) if t is not None]
        Vc = max(16, (len(s_p.counts.get((), ())) if s_p else len(ent.counts.get((), ()))) + 8)
        hist = tuple(run["chist"])
        bits = m_seq.loglik(ent, list(cats), chain, Vc, hist)
        mu, _ = _rate(ent)
        for tier in (c_p, s_p):
            if not math.isfinite(mu) and tier is not None:
                mu, _ = _rate(tier)
        run["chist"] = (hist + tuple(cats))[-CAT_ORDER:]
        rec = bits.astype(np.float32)
        if not ent.n_tokens >= MIN_TOK_STATS:
            rec[:] = np.nan
        return m_seq.session_excess(bits, mu), rec

    @staticmethod
    def _cat_tokens(store, s: str, e: str, now: float, dt: float, run: Dict[str, Any]) -> tuple:
        """Stream B: one 'cat:a+b' symbol per earlier tick with lib-4 matches of
        confidence >= 0.6 (one-tick lag: matches at ts >= now are left for later)."""
        last = run["cat_ts"]
        since = last if math.isfinite(last) else now - dt
        ms = store.matches(s, e, since=since, limit=1000)
        if not ms:
            return ()
        bags: Dict[float, set] = {}
        for m in ms:
            if m.ts >= now or (math.isfinite(last) and m.ts <= last):
                continue
            conf = m.confidence
            if conf is None or not conf >= m_seq.CAT_MIN_CONF or not m.category:
                continue
            bags.setdefault(m.ts, set()).add(str(m.category))
        if not bags:
            return ()
        run["cat_ts"] = max(bags)
        return tuple(m_seq.CAT_PREFIX + "+".join(sorted(bags[t])) for t in sorted(bags))

    # -------------------------------------------------------------- writes
    def _write(self, store, s: str, e: str, now: float, dt: float, out: Dict[str, Any]) -> None:
        axes_seq = ["sequence"] + (["credential"] if out["cred_seq"] else [])
        axes_dw = ["sequence"] + (["credential"] if out["cred_dwell"] else [])
        pm = {"dwell": out["pm_dwell"]} if math.isfinite(out["pm_dwell"]) else None
        emit.write_scores(store, s, e, now, {"seq": out["seq"], "dwell": out["dwell"]}, pm=pm,
                          axes={"seq": axes_seq, "dwell": axes_dw}, window_s=int(dt))

    @staticmethod
    def _write_degraded(store, s: str, e: str, now: float, dt: float, r2_failed: bool) -> None:
        cause = "producer_error:" + R2_ENGINE if r2_failed else "stale:act.stream"
        emit.write_scores(store, s, e, now, {"seq": math.nan, "dwell": math.nan},
                          degraded={"seq": cause, "dwell": cause}, window_s=int(dt))

    def _publish(self, store, s: str, e: str, model: Dict[str, Any], state: Dict[str, Any],
                 gate: G.GateState, run: Dict[str, Any], ck: Optional[str], now: float,
                 out: Optional[Dict[str, Any]]) -> None:
        mu, sd = P.entropy_rate(state["ppm"])
        model.update(ppm=state["ppm"], ppm_cat=state["ppm_cat"], gap=state["gap"],
                     dwell=state["dwell"], session_gap=float(state["session_gap"]),
                     entropy_rate=[mu, sd], class_key=ck, version=int(gate.version), ts=now,
                     _state=state, _gate=gate, _run=run)
        store.put_model(s, e, MODEL, model, version=int(gate.version))
        if out is None:
            return
        prof = store.profile(s, e)
        if prof is None:
            prof = EntityProfile(system=s, entity=e, updated=now)
            store.put_profile(prof)
        seq = dict(prof.extra.get("sequence") or {})
        seq.update({
            "session_gap_s": float(state["session_gap"]),
            "entropy_rate_bits": None if not math.isfinite(mu) else round(mu, 4),
            "entropy_sd_bits": None if not math.isfinite(sd) else round(sd, 4),
            "n_tokens": round(float(state["ppm"].n_tokens), 3),
            "maturity": round(m_seq.maturity(model), 4),
            "class_key": ck, "version": int(gate.version),
            "last": {"ts": now, "excess": _r(out.get("excess")), "cusum": _r(out.get("cusum")),
                     "seq": _r(out.get("seq")), "dwell": _r(out.get("dwell")),
                     "class_llr": _r(out.get("class_llr")), "n_tokens": out.get("n", 0)},
        })
        if self.entity_due(("seq-portrait", s, e), now, PORTRAIT_REFIT_S):
            d = m_seq.describe(model, k=8)
            seq.update(top_bigrams=d["top_bigrams"], top_unigrams=d["top_unigrams"],
                       long_dwell=d["long_dwell"])
        prof.extra["sequence"] = seq

    # --------------------------------------------------------------- tiers
    def _refit_tiers(self, store, s: str, now: float, used: set) -> int:
        """Rebuild the system unigram and the role-class models from the member
        entity models (hourly per key, deterministic phase; first call at once)."""
        n = 0
        if self.entity_due(("seq-tier", s, SYSTEM_KEY), now, self.tier_refit_s):
            n += self._build_tier(store, s, SYSTEM_KEY, store.entities(s), now, 0, 0, "system")
        for ck in sorted(used):
            if self.entity_due(("seq-tier", s, ck), now, self.tier_refit_s):
                n += self._build_tier(store, s, ck, m_class.class_members(store, s, ck), now,
                                      ORDER, CAT_ORDER, "class")
        return n

    def _build_tier(self, store, s: str, key: str, members: Sequence[str], now: float,
                    order: int, cat_order: int, kind: str) -> int:
        """Rebuild a tier from its members' learned states. The build is a
        pure function of the members' states (in member order), so when no
        member's state changed since this key's last build (_state_mark,
        tracked per entity tick) the previous tier's statistics are
        republished as they are, as a new version at now (perf: the full
        rebuild merges every member PPM, ~half of B10's cost on pack A)."""
        states = [(m, _other_state(store, s, m)) for m in members]
        sig = (order, cat_order, kind,
               tuple((m, id(o), self._rev.get((s, m))) for m, o in states if o is not None))
        prev = m_seq.get(store, s, key)
        last = self._tier_sig.get((s, key))
        if (last is not None and last[0] == sig and prev is not None
                and prev.get("ppm") is last[1] and prev.get("kind") == kind):
            ver = int(prev.get("version", 0)) + 1
            model = dict(prev)
            model.update(version=ver, ts=now)
            store.put_model(s, key, MODEL, model, version=ver)
            return 1
        st = _init_state(order, cat_order)
        k = 0
        for m, o in states:
            if o is not None:
                _merge_state(st, o, 1.0)
                k += 1
        if not k:
            self._tier_sig.pop((s, key), None)
            return 0
        ver = int(prev.get("version", 0)) + 1 if prev else 1
        mu, sd = P.entropy_rate(st["ppm"])
        store.put_model(s, key, MODEL, {
            "kind": kind, "ppm": st["ppm"], "ppm_cat": st["ppm_cat"], "gap": st["gap"],
            "dwell": st["dwell"], "session_gap": float(st["session_gap"]),
            "entropy_rate": [mu, sd], "members": k, "version": ver, "ts": now}, version=ver)
        self._tier_sig[(s, key)] = (sig, st["ppm"])
        return 1


# ================================================================= helpers
def _state_mark(state: Dict[str, Any], gate: G.GateState) -> Tuple[Any, ...]:
    """A marker that changes whenever the gated learner may have changed the
    learned state: every update is a committed journal row (the journal's
    newest row or its length moves), a rollback / release / rebase / freeze /
    link seed moves version, branch, last_rollback_ts, link_version, frozen
    or the applied directives, and a reload replaces the state object."""
    j = gate.journal
    return (id(state), gate.version, gate.branch, gate.last_ts, len(j), j[-1] if j else None,
            len(gate.held), gate.link_version, gate.frozen, gate.last_rollback_ts,
            repr(sorted(gate.applied.items(), key=lambda kv: kv[0])))


def _other_state(store, s: str, e: str) -> Optional[Dict[str, Any]]:
    m = m_seq.get(store, s, e)
    return m.get("_state") if m is not None and m.get("kind") == "entity" else None


def _fresh_pos(store, s: str, e: str, name: str, now: float) -> bool:
    v = store.latest_fresh(s, e, name, now)
    try:
        return v is not None and float(v) > 0.0
    except (TypeError, ValueError):
        return False


def _frac(store, s: str, e: str, now: float) -> float:
    """act.stream_frac of this tick in (0, 1]; missing or invalid -> 1 (rows complete)."""
    f = m_template.stream_frac(store, s, e, now)
    return f if (f == f and 0.0 < f <= 1.0) else 1.0


def _r(x: Any) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, 4) if math.isfinite(x) else None


def _prune_rows(model: Dict[str, Any], gate: G.GateState, now: float) -> None:
    """Keep pending rows (ts > gate.last_ts), held rows, and committed rows for
    REPLAY_KEEP_S; at most ROW_TOKEN_CAP tokens (oldest committed rows first,
    then the oldest held)."""
    rows, held = model["_rows"], model["_held"]
    held_ts = {r.ts for r in gate.held} if gate.held else set()
    cut = now - REPLAY_KEEP_S
    last = gate.last_ts
    while rows:
        ts0 = next(iter(rows))
        if ts0 >= cut and model["_ntok"] <= ROW_TOKEN_CAP:
            break
        r = rows.pop(ts0)
        if ts0 > last or ts0 in held_ts:
            held[ts0] = r                          # still needed: keep aside
        else:
            model["_ntok"] -= r.n_tok + len(r.cats)
    if held:
        for ts0 in [t for t in held if t <= last and t not in held_ts]:
            r = held.pop(ts0)                      # released, rebased or dropped
            model["_ntok"] -= r.n_tok + len(r.cats)
        while held and model["_ntok"] > ROW_TOKEN_CAP:
            r = held.pop(next(iter(held)))
            model["_ntok"] -= r.n_tok + len(r.cats)
