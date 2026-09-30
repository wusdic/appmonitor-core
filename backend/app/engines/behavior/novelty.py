"""NoveltyEngine (B08) — first-seen, rarity and distribution shift over everything an IP touches.

Why: a stolen credential, a compromised host or an insider rarely moves the
volume first; it touches something NEW — an admin template, an external
upload host, a TXT query to a fresh domain, a port nobody in the role uses.
But "new" alone is useless: explorers find new pages daily and a role rollout
makes every member "novel" on the same morning. So novelty is judged at three
tiers (entity, role class, system), against the entity's own novelty rate,
and discounted when the class adopts the value together — while the adoption
itself is recorded for the class monitor (B18), so a coordinated risky
adoption cannot hide behind the discount.

  * Dimensions (lib/m_vocab.DIMS): HTTP method+host+template (act.tokens),
    content type, SNI eTLD+1, DNS domain eTLD+1 and qtype, dport, peer /24 and
    the lib-4 category (store.matches, one tick late). '__other__' is skipped.
  * State, per (s, e): decayed counts (half-life 30 d), n_acc, first/last
    seen per value, Space-Saving cap 256 per dim; class (s, 'class:<rid>')
    and system (s, '__system__') tiers are rebuilt hourly from the member
    entity models (cap 2048; decayed document frequency df over 30 d), so they
    inherit the members' trust gating and rollbacks.
  * Hierarchical Dirichlet backoff (m_vocab.Backoff): p_e = (c_e + 5 p_c)/(N_e + 5),
    p_c = (c_c + 5 p_s)/(N_c + 5), p_s = (c_s + U)/(N_s + 1); I_v = -log2 p_e.
  * Tier of a value new to the entity: 'system' when the (mature) system has
    never seen it; 'class' when it is new or rare in the class (IDF =
    ln((N_class+1)/(df+1)) + 1 >= ln 5 + 1, df counting members that sighted
    it since the last rebuild) — the system stands in for an entity without a
    class; else 'entity'. Re-emergence after 30 d counts as new.
  * score.novelty = max(-log10 p_rate, sum_d max_v omega_d w_v I_v / 12)
    (instantaneous):
      p_rate  Good-Turing novelty-rate test: p_new = (N1 + 0.5)/(N + 1) (N1 the
              decayed mass of singleton values), p = betainc(k, n - k + 1, p_new)
              for k first-seen values among the n distinct values of the tick;
      w_v     sensitivity weight in [1, 3]: +1 sensitive (config
              sensitive_patterns) or admin, +h for a low-prevalence value that
              moved heavy bytes (h = 1 at 1 MB, or 250 KB up);
      omega_d 1 - Good-Turing unseen mass of the dimension (pooled while the
              dimension holds < 20 accesses).
    Values in the sum: new to the entity (sighted but not yet committed
    included) and rare sensitive ones (class-rare, I >= 12 bits).
  * Adoption: when max(3, 30 %) of the class first saw v within 7 d the
    per-entity score is max(score without adopted values, 0.1 x the full
    score), first_seen is marked adopted (INFO) and class_adopted (INFO) is
    emitted once at the class key. The adoption record (members + first-seen
    ts, flags external / upload / sensitive / new_eTLD1_org) is ALWAYS written
    to model.vocab@class for B18. Values inherited through model.link are not
    new at the entity tier; allowlisted values (m_feedback) are learned but
    never scored or reported.
  * score.novelty_rate (accumulator): class-rare first sightings in the last
    24 h against a Gamma-Poisson NB predictive of the entity's committed
    active-day counts (method-of-moments overdispersion); >= 2 complete days.
  * score.jsd (accumulator): max over dimensions of the JSD between decayed
    1-h / 24-h windows and the smoothed profile p_e; randomised conformal p
    (+ GPD tail) against the entity's committed JSD rings, both windows,
    Bonferroni x 2; the 1-h window at most every 5 min, the 24-h one hourly,
    once the entity holds >= 100 accesses and its ring >= 32 entries.
    acc_alarm when e_day(p) <= the detector's daily budget (lib/detectors).
  * Axes: categorical; exfil for an external upload-dominant value; privilege
    for a sensitive or admin one. Events: first_seen when tier >= class or
    omega_d I >= 12 bits (class / entity tier need a mature entity: >= 100
    accesses and omega >= 0.5; system tier always), rare_access for sensitive values,
    deduplicated by (entity, dim, value) for 30 d, <= 5 per entity-tick;
    extra = {dim, value, tier, bits, idf, flags, adopted, ...}.
  * Learning is delayed, trust-gated and reversible (lib/gating, contract H,
    model.control, link seeding); rows are kept in the model until committed
    (and 26 h after for rollback replay). Clock: behavior.score.
  * ctx.training learns and scores but emits no events. A raw producer error
    on an active tick, or activity without categorical data, writes NaN plus
    behavior.degraded. Silent entities are not scored (absence is not novelty).

Interpretation (spec test a): n counts the DISTINCT values of the tick, not
raw accesses: accesses inside a tick are bursty (one page view hits a
template many times), so the opportunities for a first sighting are the
distinct values; with raw accesses the test would reduce to ~1/(2T) after T
ticks whatever the entity's history. The per-dimension max in the surprisal
sum keeps one burst of fresh CDN peers from reading as dozens of findings
(breadth is novelty_rate's job).
"""
from __future__ import annotations

import math
import pickle
import re
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import special

from ...core.engine import Context, Engine
from ...models.schema import BehaviorEvent, EntityProfile, Severity
from .lib import bayes
from .lib import pactive as PA
from .lib import calib
from .lib import combine
from .lib import emit
from .lib import gating as G
from .lib import m_class
from .lib import m_feedback
from .lib import m_template as MT
from .lib import m_vocab as V
from .lib import grains as GR
from .lib.classkeys import SYSTEM_KEY, role_key
from .lib.detectors import DETECTOR_INFO
from .lib.names import etld1, is_external
from .lib.template import channel_of

MODEL = V.MODEL
LEARNER = "vocab"
CLOCK = emit.SCORE
DETECTORS = ("novelty", "novelty_rate", "jsd")
RAW_PRODUCERS = ("raw.action_token", "raw.http", "raw.tls", "raw.dns", "raw.l4flow")
OTHER = "__other__"

H = V.HALF_LIFE_S
DAY = 86400.0
REEMERGE_S = 30 * DAY
ADOPT_WINDOW_S = 7 * DAY
ADOPT_MIN = 3
ADOPT_FRAC = 0.3
ADOPT_DISCOUNT = 0.1
IDF_RARE = math.log(5.0) + 1.0            # df + 1 <= (N_class + 1) / 5
BITS_EVENT = 12.0
BITS_SCALE = 12.0
RATE_MIN_N = 32.0                          # accesses before the Good-Turing test
OMEGA_DIM_MIN_N = 20.0
MATURE_N = 100.0
MATURE_OMEGA = 0.5
SYS_MIN_N = 50.0                           # system tier "mature" (total accesses)
MAX_EVENTS = 5
CAT_GRAIN_S = 900.0                        # lib-4 matches are per 15-min grain (rule_match)
LOW_PREV = 0.2
UPLOAD_MIN = 1024.0
WIN_H = (3600.0, DAY)                      # JSD window half-lives
WIN_CAP = 48
JSD_MIN_MASS = (3.0, 10.0)
JSD_EVAL_S = 300.0
JSD24_EVAL_S = 3600.0                      # the 24-h window moves slowly: hourly
JSD_RING = calib.RING_M
JSD_MIN_RING = 32
JSD_REFIT = 16
NR_MIN_DAYS = 2
NR_KEEP_DAYS = 28
NR_A0, NR_M0 = 0.5, 1.0
REPLAY_KEEP_S = 26 * 3600.0
ROW_ENTRY_CAP = 20000
SEEN_KEEP_S = 30 * DAY
SEEN_CAP = 4096
EMIT_KEEP_S = 30 * DAY
LEDGER_KEEP_S = 8 * DAY
LEDGER_CAP = 2048
TIER_REFIT_S = 3600.0
PORTRAIT_REFIT_S = 3600.0
_RENORM_G = 32.0
_REST = "\x00rest"
_NAN = math.nan
_P_FLOOR = 1e-300

DEFAULT_SENSITIVE = (r"salary", r"payroll", r"passw", r"secret", r"credential")
_ADMIN_RE = re.compile(r"(^|[/ .])(admin|administrator|root|sudo|superuser|sysadmin|"
                       r"manage|management|console|phpmyadmin|wp-admin)([/ .?|{]|$)", re.I)
ADMIN_PORTS = frozenset({"22", "23", "135", "445", "3389", "5900", "5985", "5986"})


def peer_rare(model: Optional[Dict[str, Any]], dim: str, value: Any,
              extra_df: float = 0.0) -> bool:
    """Is `value` rare among the scored entity's PEERS (round 4, evaluator)?

    The posterior median of the peers' usage share under a Jeffreys prior,
    Beta(df + 1/2, n - df + 1/2) with df the (decayed) peers that used the
    value and n = n_ent - 1 the peers (the scored entity, a class member,
    has not used it: a new value, or rare_access's own bits test), is at
    most LOW_PREV (20 %). Why: the tier rule's smoothed IDF,
    ln((N + 1)/(df + 1)) + 1 >= ln 5 + 1, cannot call a value used by ONE
    peer rare in a class of fewer than 9 members - (1 + 1)/(8 + 1) = 22 %
    > 20 % - although its share among the peers is 1/7. Pack A's ERP users
    form a class of 8 in which only the HR persona uses /hr/salary/export,
    so T7 (another user exporting salaries once per tick for a day, the
    spec's 'B08 rare_access' case) never produced rare_access and was
    detected only while q_inst was anti-conservative (fix agent, round 4).
    The Jeffreys median is ~ (df + 1/6) / (n + 1/3): one of 7 peers gives
    0.16 (rare), one of 3 gives 0.35, two of 8 gives 0.27 (not rare).
    Fewer than 2 peers: undecidable, False."""
    if not isinstance(model, dict) or model.get("kind") not in ("class", "system"):
        return False
    n = float(model.get("n_ent", 0.0)) - 1.0
    if not n >= 2.0:
        return False
    x = ((model.get("dims") or {}).get(dim) or {}).get(value)
    df = (float(x[1]) if x is not None else 0.0) + max(0.0, float(extra_df))
    df = min(max(df, 0.0), n)
    return float(special.betaincinv(df + 0.5, n - df + 0.5, 0.5)) <= LOW_PREV


class _Row(NamedTuple):
    """One tick of categorical observations, kept until committed (gating fetch)."""
    ts: float
    items: Tuple[Tuple[str, str, float], ...]      # (dim, value, count)
    k_rare: float                                  # class-rare first sightings (novelty_rate)
    j1: float                                      # JSD statistics (NaN = not evaluated)
    j24: float


# ============================================================ learner state
def _init_state() -> Dict[str, Any]:
    return {"H": H, "t_ref": _NAN, "g": 0.0, "dims": {}, "N": {}, "N1": {},
            "clock": -math.inf, "first": math.inf, "days": {},
            "jsd": {"h1": calib.Ring(cap=JSD_RING), "h24": calib.Ring(cap=JSD_RING)},
            "jsd_n": 0, "n_rows": 0}


def _renorm(st: Dict[str, Any], f: float) -> None:
    for tab in st["dims"].values():
        for x in tab.values():
            x[0] *= f
    for key in ("N", "N1"):
        st[key] = {d: v * f for d, v in st[key].items()}


def _scale(st: Dict[str, Any], ts: float) -> float:
    """Stored-scale multiplier of a row at ts (lazy decay: the clock only moves
    forward, an older row arrives decayed)."""
    if not math.isfinite(st["t_ref"]):
        st["t_ref"], st["g"] = ts, 0.0
        return 1.0
    g_ts = (ts - st["t_ref"]) / st["H"]
    if g_ts > st["g"]:
        st["g"] = g_ts
        if g_ts > _RENORM_G:
            _renorm(st, 2.0 ** (-g_ts))
            st["t_ref"] += g_ts * st["H"]
            st["g"] = g_ts = 0.0
    return 2.0 ** g_ts


def _add_value(st: Dict[str, Any], dim: str, v: str, inc: float, n: float,
               first: float, last: float) -> None:
    """Fold one value (stored-scale increment) with Space-Saving eviction."""
    tab = st["dims"].get(dim)
    if tab is None:
        tab = st["dims"][dim] = {}
    N1 = st["N1"]
    x = tab.get(v)
    if x is not None:
        if x[1] == 1.0:
            N1[dim] = N1.get(dim, 0.0) - x[0]
        x[0] += inc
        x[1] += n
        if first < x[2]:
            x[2] = first
        if last > x[3]:
            x[3] = last
        if x[1] == 1.0:
            N1[dim] = N1.get(dim, 0.0) + x[0]
    else:
        base = 0.0
        if len(tab) >= V.ENTITY_CAP:
            vmin = min(tab, key=lambda k: tab[k][0])
            xm = tab.pop(vmin)
            if xm[1] == 1.0:
                N1[dim] = N1.get(dim, 0.0) - xm[0]
            base = xm[0]
        tab[v] = x = [base + inc, n, first, last]
        if n == 1.0:
            N1[dim] = N1.get(dim, 0.0) + x[0]


def _update(st: Dict[str, Any], row: _Row, w: float) -> Dict[str, Any]:
    """Commit one row with trust weight w. In place, deterministic in
    (state, row, w) so checkpoint + replay is exact."""
    ww = float(w)
    if not (ww > 0.0 and math.isfinite(ww)):
        return st
    ts = row.ts
    sc = ww * _scale(st, ts)
    N = st["N"]
    for dim, v, n in row.items:
        inc = sc * n
        _add_value(st, dim, v, inc, float(n), ts, ts)
        N[dim] = N.get(dim, 0.0) + inc
    if ts > st["clock"]:
        st["clock"] = ts
    if ts < st["first"]:
        st["first"] = ts
    cd = int(st["clock"] // DAY)
    day = int(ts // DAY)
    days = st["days"]
    if day >= cd - NR_KEEP_DAYS:
        days[day] = days.get(day, 0.0) + ww * float(row.k_rare)
    for d in [d for d in days if d < cd - NR_KEEP_DAYS]:
        del days[d]
    if ww >= 0.5:
        rings = st["jsd"]
        if row.j1 == row.j1:
            rings["h1"].add(row.j1, ts)
        if row.j24 == row.j24:
            rings["h24"].add(row.j24, ts)
        if row.j1 == row.j1 or row.j24 == row.j24:
            st["jsd_n"] += 1
    st["n_rows"] += 1
    return st


def _merge_state(own: Dict[str, Any], other: Dict[str, Any], w: float) -> Dict[str, Any]:
    """own + w * other in sufficient-statistic space (link seeding)."""
    if not math.isfinite(other.get("t_ref", _NAN)):
        return own
    t_o = other["t_ref"] + other["g"] * other["H"]
    f = float(w) * _scale(own, t_o) * 2.0 ** (-other["g"])
    for dim, tab in other["dims"].items():
        for v, x in tab.items():
            _add_value(own, dim, v, f * x[0], float(x[1]), x[2], x[3])
    for dim, tot in other["N"].items():
        own["N"][dim] = own["N"].get(dim, 0.0) + f * tot
    own["clock"] = max(own["clock"], other["clock"])
    own["first"] = min(own["first"], other["first"])
    for d, c in other["days"].items():
        own["days"][d] = own["days"].get(d, 0.0) + float(w) * c
    for k in ("h1", "h24"):
        own["jsd"][k] = calib.seed_ring(own["jsd"][k], other["jsd"][k], G.LINK_SEED_WEIGHT)
    own["n_rows"] += other["n_rows"]
    return own


def _dump(st: Dict[str, Any]) -> bytes:
    # bytes: GatedLearner deep-copies blobs, which is O(1) for bytes
    return pickle.dumps(st, protocol=pickle.HIGHEST_PROTOCOL)


def _load(blob: Any) -> Dict[str, Any]:
    return pickle.loads(blob) if isinstance(blob, (bytes, bytearray)) else blob


def _new_run() -> Dict[str, Any]:
    """Per-entity observation bookkeeping (not learned, not rolled back)."""
    return {"seen": {}, "emitted": {}, "nov": [], "cat_ts": _NAN, "jsd_t": -math.inf,
            "win": [_new_win(), _new_win()], "tail": {}, "last": {}, "recent": []}


def _new_win() -> Dict[str, Any]:
    return {"t0": _NAN, "d": {}, "tot": {}}


# ================================================================ windows
def _win_add(win: Dict[str, Any], h: float, ts: float, obs: Dict[str, Dict[str, float]]) -> None:
    """Add a tick to a decayed window (lazy scale; smallest values fold into _REST)."""
    if not math.isfinite(win["t0"]):
        win["t0"] = ts
    g = (ts - win["t0"]) / h
    if g > 30.0:
        f = 2.0 ** (-g)
        for dd in win["d"].values():
            for v in dd:
                dd[v] *= f
        win["tot"] = {d: x * f for d, x in win["tot"].items()}
        win["t0"], g = ts, 0.0
    sc = 2.0 ** g
    for dim, vals in obs.items():
        dd = win["d"].setdefault(dim, {})
        s = 0.0
        for v, n in vals.items():
            m = sc * n
            dd[v] = dd.get(v, 0.0) + m
            s += m
        win["tot"][dim] = win["tot"].get(dim, 0.0) + s
        if len(dd) > WIN_CAP:
            extra = sorted((x, v) for v, x in dd.items() if v != _REST)[:len(dd) - WIN_CAP]
            rest = dd.get(_REST, 0.0)
            for x, v in extra:
                rest += x
                del dd[v]
            dd[_REST] = rest


def _jsd_dim(dd: Dict[str, float], tot: float, probs: Dict[str, float]) -> float:
    """JSD (bits) between the window shares and the profile over the window
    support plus one 'rest' bin (the profile mass outside the support)."""
    q_rest = 0.0
    p_sum = 0.0
    js = 0.0
    log2 = math.log2
    for v, m in dd.items():
        q = m / tot
        if v == _REST:
            q_rest += q
            continue
        p = probs[v]
        p_sum += p
        mid = q + p
        if q > 0.0:
            js += q * log2(2.0 * q / mid)
        if p > 0.0:
            js += p * log2(2.0 * p / mid)
    p_rest = max(0.0, 1.0 - p_sum)
    mid = q_rest + p_rest
    if q_rest > 0.0:
        js += q_rest * log2(2.0 * q_rest / mid)
    if p_rest > 0.0:
        js += p_rest * log2(2.0 * p_rest / mid)
    return min(1.0, max(0.0, 0.5 * js))


# ================================================================ helpers
def _num(x: Any) -> float:
    if isinstance(x, dict):
        x = x.get("n", 0.0)
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return v if (v > 0.0 and math.isfinite(v)) else 0.0


def _fresh_pos(store, s: str, e: str, name: str, now: float) -> bool:
    v = store.latest_fresh(s, e, name, now)
    try:
        return v is not None and float(v) > 0.0
    except (TypeError, ValueError):
        return False


def _r(x: Any, nd: int = 4) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


def _jp(x: Any) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return float(f"{x:.4g}") if math.isfinite(x) else None


def _clip_p(p: float) -> float:
    return min(1.0, max(_P_FLOOR, p))


def _score_of(p: float) -> float:
    return -math.log10(_clip_p(p)) + 0.0 if p == p else _NAN     # + 0.0: no -0.0


def _host_of(dim: str, v: str) -> str:
    if dim == "tmpl":
        parts = v.split(" ", 2)
        return parts[1] if len(parts) > 1 else ""
    if dim in ("sni", "dns"):
        return v
    if dim == "peer":
        return v.split("/", 1)[0]
    return ""


def _new_ledger_rec(dim: str, v: str, now: float, tier: str) -> Dict[str, Any]:
    return {"dim": dim, "value": v, "members": {}, "first_ts": now, "last_ts": now,
            "flags": {"external": False, "upload": False, "sensitive": False,
                      "new_eTLD1_org": False},
            "adopted": False, "adopted_ts": None, "n_class": 0, "tier": tier}


class _Val:
    """One scored value of one entity-tick (a candidate: new or sensitive)."""
    __slots__ = ("dim", "v", "n", "key", "new", "first", "reemerg", "sens", "admin", "ext",
                 "up", "down", "upload", "tier", "idf", "idf_s", "share_s", "bits", "w",
                 "allow", "adopted", "contrib", "axes", "rare_sens", "new_org", "omega")

    def __init__(self, dim: str, v: str, n: float) -> None:
        self.dim, self.v, self.n = dim, v, n
        self.key = V.value_key(dim, v)
        self.new = self.first = self.reemerg = False
        self.sens = self.admin = self.ext = self.upload = False
        self.up = self.down = 0.0
        self.tier = "entity"
        self.idf = self.idf_s = self.share_s = self.bits = _NAN
        self.w = 1.0
        self.allow = self.adopted = self.rare_sens = self.new_org = False
        self.contrib = 0.0
        self.omega = 1.0
        self.axes: List[str] = ["categorical"]


class _Ent:
    """One entity's per-tick work item (pass 1 fills, pass 2 scores)."""
    __slots__ = ("e", "model", "obs", "active", "degraded", "ck", "vals", "n_distinct",
                 "bo", "bytes", "first_keys")

    def __init__(self, e: str, model: Dict[str, Any]) -> None:
        self.e, self.model = e, model
        self.obs: Dict[str, Dict[str, float]] = {}
        self.active = False
        self.degraded = ""
        self.ck: Optional[str] = None
        self.vals: List[_Val] = []
        self.n_distinct = 0
        self.bo: Optional[V.Backoff] = None
        self.bytes: Optional[Tuple[Dict[int, List[float]], Dict[str, List[float]]]] = None
        self.first_keys: List[_Val] = []


class _Sys:
    """Per-system tick context: tier models, role sizes, links, allowlist, org names."""

    def __init__(self, engine: "NoveltyEngine", ctx: Context, s: str) -> None:
        self.store, self.s = ctx.store, s
        self.now, self.dt, self.training = float(ctx.now), float(ctx.window_s), ctx.training
        self.sys = V.get(self.store, s, SYSTEM_KEY)
        self._cls: Dict[str, Optional[Dict[str, Any]]] = {}
        self._n_members: Dict[str, int] = {}
        self.touched: Dict[str, Dict[str, Any]] = {}
        al = m_feedback.get(self.store).get("allowlist") or {}
        self.has_allow = bool(al)
        self.links: Dict[str, List[str]] = {}
        link = self.store.get_model(s, SYSTEM_KEY, G.LINK_MODEL)
        if isinstance(link, dict):
            for lk in _iter_links(link):
                if lk.get("to") and lk.get("from") and not _retracted(lk):
                    self.links.setdefault(str(lk["to"]), []).append(str(lk["from"]))
        self._org: Optional[Set[str]] = None
        cfg = ctx.config or {}
        self.org_domains = tuple(cfg.get("org_domains") or ())
        self.sens_re = engine._patterns(cfg.get("sensitive_patterns"))
        n_sys = sum(float(x) for x in (self.sys or {}).get("N", {}).values()) if self.sys else 0.0
        self.sys_ok = bool(self.sys) and int(self.sys.get("members", 0)) >= 1 and n_sys >= SYS_MIN_N
        # canonical grain mode (round 4, evaluator): a lib-4 category counts
        # as the grains its match covers, dt / 900 (CAT_GRAIN_S)
        self.cat_w = self.dt / CAT_GRAIN_S if GR.canonical(cfg) else 1.0

    def class_key(self, e: str) -> Optional[str]:
        rid = m_class.role_id(self.store, self.s, e)
        if rid is None:
            return None
        n = self.n_members(rid)
        return role_key(rid) if n >= m_class.MIN_MEMBERS else None

    def n_members(self, rid: str) -> int:
        n = self._n_members.get(rid)
        if n is None:
            n = self._n_members[rid] = len(m_class.members(self.store, self.s, rid))
        return n

    def cls(self, key: Optional[str]) -> Optional[Dict[str, Any]]:
        if not key:
            return None
        if key not in self._cls:
            self._cls[key] = V.get(self.store, self.s, key)
        return self._cls[key]

    def cls_for_ledger(self, key: str) -> Dict[str, Any]:
        m = self.cls(key)
        if m is None:
            m = {"fmt": 1, "kind": "class", "version": 0, "built": _NAN, "H": H, "dims": {},
                 "N": {}, "N1": {}, "n_ent": 0.0, "members": 0, "adoption": {}}
            self._cls[key] = m
        m.setdefault("adoption", {})
        self.touched[key] = m
        return m

    def sightings(self) -> Optional[Dict[str, list]]:
        if self.sys is None:
            return None
        sg = self.sys.setdefault("sightings", {})
        self.touched[SYSTEM_KEY] = self.sys
        return sg

    def org_names(self) -> Set[str]:
        """eTLD+1 names any system of the organisation has committed (sni / dns)."""
        if self._org is None:
            names: Set[str] = set()
            for s2 in self.store.systems():
                m = V.get(self.store, s2, SYSTEM_KEY)
                if m is None:
                    continue
                for d in ("sni", "dns"):
                    names.update((m.get("dims") or {}).get(d, {}).keys())
            self._org = names
        return self._org


# ================================================================== engine
class NoveltyEngine(Engine):
    name = "behavior.novelty"
    layer = "behavior"
    consumes = ["act.tokens", "act.stream", "act.stream_frac", "act.events",
                "http.content_types", "tls.sni_etld1_set", "tls.sni_set",
                "dns.qname_etld1_set", "dns.qname_set", "dns.qtype_set", "l4.dport_set",
                "l4.peer_set", "match.*", "model.template", "model.class", "model.feedback",
                "model.link", "behavior.trust", "behavior.trust_prov", "behavior.quarantine",
                "model.control"]
    produces = [MODEL, "behavior.score", "behavior.pm", "behavior.axes", "behavior.acc_alarm",
                "behavior.degraded", "profile.extra.categorical", "event:first_seen",
                "event:rare_access", "event:class_adopted"]
    description = ("First-seen, rarity and distribution shift at entity, class and system "
                   "tiers: hierarchical Dirichlet surprisal, Good-Turing novelty-rate test, "
                   "class IDF x sensitivity, adoption discount with class adoption records, "
                   "daily NB novelty rate and conformal JSD accumulators.")
    interval = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.tier_refit_s = float(params.get("tier_refit_s", TIER_REFIT_S))
        self._learners: Dict[float, G.GatedLearner] = {}
        self._rows: Dict[float, _Row] = {}
        self._held: Dict[float, _Row] = {}
        self._pat_key: Any = None
        self._pat: List[re.Pattern] = []
        self._sens_cache: Dict[Tuple[str, str], Tuple[bool, bool]] = {}
        self._ext_cache: Dict[Tuple[Any, str, str], bool] = {}

    # ----------------------------------------------------------------- run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        self._pa_cfg = ctx.config
        store = ctx.store
        now = float(ctx.now)
        d_min = ctx.config.get("D_min_s", G.D_MIN_S)
        lrn = self._learner(float(d_min) if d_min else G.D_MIN_S)
        failed = [p for p in RAW_PRODUCERS if store.engine_failed(p, now)]
        n = 0
        for s in store.systems():
            n += self._system(ctx, lrn, s, failed)
        return n

    def _learner(self, d_min_s: float) -> G.GatedLearner:
        lrn = self._learners.get(d_min_s)
        if lrn is None:
            lrn = self._learners[d_min_s] = G.GatedLearner(
                name=LEARNER, init=_init_state, update=_update, fetch=self._fetch,
                dump=_dump, load=_load, merge=_merge_state, d_min_s=d_min_s,
                ckpt_every_s=G.CKPT_EVERY_S, clock=CLOCK)
        return lrn

    def _fetch(self, store, s: str, e: str, ts: float) -> Optional[_Row]:
        r = self._rows.get(ts)
        return r if r is not None else self._held.get(ts)

    def _patterns(self, pats: Any) -> List[re.Pattern]:
        key = tuple(str(p) for p in pats) if pats else ()
        if key != self._pat_key:
            src = key or DEFAULT_SENSITIVE
            out = []
            for p in src:
                try:
                    out.append(re.compile(p, re.I))
                except re.error:
                    out.append(re.compile(re.escape(p), re.I))
            self._pat_key, self._pat = key, out
            self._sens_cache.clear()
        return self._pat

    def _sensitive(self, sc: _Sys, dim: str, v: str) -> Tuple[bool, bool]:
        k = (dim, v)
        hit = self._sens_cache.get(k)
        if hit is None:
            if dim in ("tmpl", "sni", "dns", "ctype"):
                sens = any(p.search(v) for p in sc.sens_re)
                admin = dim == "tmpl" and bool(_ADMIN_RE.search(v))
            elif dim == "dport":
                sens, admin = False, v in ADMIN_PORTS
            elif dim == "cat":
                sens, admin = False, v.lower() == "admin"
            else:
                sens = admin = False
            if len(self._sens_cache) > 65536:
                self._sens_cache.clear()
            hit = self._sens_cache[k] = (sens, admin)
        return hit

    def _external(self, sc: _Sys, dim: str, v: str) -> bool:
        k = (sc.org_domains, dim, v)
        hit = self._ext_cache.get(k)
        if hit is None:
            host = _host_of(dim, v)
            hit = bool(host) and is_external(host, sc.org_domains)
            if len(self._ext_cache) > 65536:
                self._ext_cache.clear()
            self._ext_cache[k] = hit
        return hit

    # -------------------------------------------------------------- system
    def _system(self, ctx: Context, lrn: G.GatedLearner, s: str, failed: List[str]) -> int:
        store = ctx.store
        now = float(ctx.now)
        self._refit_tiers(store, s, now)
        sc = _Sys(self, ctx, s)
        items: List[_Ent] = []
        for e in PA.entities(store, s, now, ctx.config):
            it = self._gather(ctx, sc, e, failed)
            if it is not None:
                items.append(it)
        self._ledgers(ctx, sc, items)
        n = 0
        for it in items:
            n += self._entity(ctx, lrn, sc, it)
        for key, m in sc.touched.items():
            store.put_model(s, key, MODEL, m, version=int(m.get("version", 0)))
        return n

    # ------------------------------------------------------------- pass 1
    def _gather(self, ctx: Context, sc: _Sys, e: str, failed: List[str]) -> Optional[_Ent]:
        store, now = ctx.store, sc.now
        model = V.get(store, sc.s, e)
        if model is not None and model.get("kind") != "entity":
            model = None
        run = model["run"] if model is not None else _new_run()
        obs = self._observe(store, sc.s, e, now, sc.dt, run, sc.cat_w)
        active = bool(obs) or _fresh_pos(store, sc.s, e, "act.events", now)
        if model is None and not active:
            return None
        if model is None:
            model = {"fmt": 1, "kind": "entity", "version": 0, "ts": now, "class_key": None,
                     "state": _init_state(), "gate": G.GateState(), "run": run, "rows": {},
                     "held_rows": {}, "n_entries": 0}
        it = _Ent(e, model)
        it.active = active
        if active and failed:
            it.degraded = "producer_error:" + failed[0]
        elif active and not obs:
            it.degraded = "stale:act.tokens"
        if it.degraded or not obs:
            return it
        it.obs = obs
        it.ck = sc.class_key(e)
        self._classify(sc, it)
        return it

    def _observe(self, store, s: str, e: str, now: float, dt: float,
                 run: Dict[str, Any], cat_w: float = 1.0) -> Dict[str, Dict[str, float]]:
        """The tick's categorical observations per dimension. Every dimension
        is an additive count except the lib-4 categories ('cat'), one per
        match: a match states what the entity does over a 15-min grain (its
        counter and ratio clauses are read per grain, signature/rule_match),
        so at 60 s a sustained activity matched on each of the grain's 15
        ticks and weighed 15x its 900-s count against the profile and the
        JSD rings learnt at 900 s. cat_w = dt / 900 in canonical mode (round
        4, evaluator: pack E seed 0, jsd p < 0.01 on 15 % of clean control
        ticks at 60 s, 941 accumulator alarms on four machine personas)."""
        out: Dict[str, Dict[str, float]] = {}

        def put(dim: str, v: Any, n: Any) -> None:
            c = _num(n)
            if c > 0.0 and v is not None:
                v = str(v)
                if v and v != OTHER:
                    d = out.setdefault(dim, {})
                    d[v] = d.get(v, 0.0) + c

        toks = store.latest_fresh(s, e, "act.tokens", now)
        if isinstance(toks, dict):
            for t, n in toks.items():
                if isinstance(t, str) and t and t[0] != "{" and t != OTHER \
                        and channel_of(t) == "http":
                    put("tmpl", MT.template_key(t), n)
        ct = store.latest_fresh(s, e, "http.content_types", now)
        if isinstance(ct, dict):
            for v, n in ct.items():
                if isinstance(v, str) and v != OTHER:
                    put("ctype", v.split(";", 1)[0].strip().lower(), n)
        for dim, reg, full in (("sni", "tls.sni_etld1_set", "tls.sni_set"),
                               ("dns", "dns.qname_etld1_set", "dns.qname_set")):
            x = store.latest_fresh(s, e, reg, now)
            if isinstance(x, dict):
                for v, n in x.items():
                    put(dim, v, n)
            else:
                x = store.latest_fresh(s, e, full, now)
                if isinstance(x, dict):
                    for v, n in x.items():
                        if isinstance(v, str) and v != OTHER:
                            put(dim, etld1(v), n)
        for dim, name in (("qtype", "dns.qtype_set"), ("dport", "l4.dport_set"),
                          ("peer", "l4.peer_set")):
            x = store.latest_fresh(s, e, name, now)
            if isinstance(x, dict):
                for v, n in x.items():
                    put(dim, str(v).upper() if dim == "qtype" else v, n)
        # lib-4 categories, one tick late (signature runs after behaviour)
        last = run.get("cat_ts", _NAN)
        since = last if math.isfinite(last) else now - dt
        newest = last
        for m in store.matches(s, e, since=since, limit=1000):
            if m.ts >= now or (math.isfinite(last) and m.ts <= last) or not m.category:
                continue
            put("cat", str(m.category).lower(), cat_w)
            newest = m.ts if not math.isfinite(newest) else max(newest, m.ts)
        run["cat_ts"] = newest
        return out

    def _classify(self, sc: _Sys, it: _Ent) -> None:
        """Mark new / first-sighted / sensitive values and compute their tier,
        rarity, surprisal and flags against the pre-tick ledgers."""
        model, now = it.model, sc.now
        st = model["state"]
        run = model["run"]
        seen = run["seen"]
        dims = st["dims"]
        cls = sc.cls(it.ck)
        it.bo = V.Backoff(model, cls, sc.sys, now=now)
        srcs = [V.get(sc.store, sc.s, a) for a in sc.links.get(it.e, ())]
        srcs = [m["state"] for m in srcs if m is not None and m.get("kind") == "entity"]
        allow: Dict[str, Set[str]] = {}
        if sc.has_allow:
            for dim in it.obs:
                allow[dim] = m_feedback.allowlist_values(sc.store, sc.s, it.e, dim, now)
        n_distinct = 0
        for dim, vals in it.obs.items():
            tab = dims.get(dim) or {}
            al = allow.get(dim)
            for v, n in vals.items():
                if al and v in al:
                    allowed = True
                else:
                    allowed = False
                    n_distinct += 1
                x = tab.get(v)
                known = x is not None and now - x[3] <= REEMERGE_S
                if known:
                    sens, admin = self._sensitive(sc, dim, v)
                    if not (sens or admin):
                        continue
                    val = _Val(dim, v, n)
                    val.sens, val.admin = sens, admin
                else:
                    if any(v in (s2["dims"].get(dim) or ()) for s2 in srcs):
                        continue                          # inherited through a link
                    val = _Val(dim, v, n)
                    val.new = True
                    val.reemerg = x is not None
                    t_seen = seen.get(val.key)
                    val.first = t_seen is None or t_seen >= now
                    val.sens, val.admin = self._sensitive(sc, dim, v)
                val.allow = allowed
                self._describe(sc, it, val, cls)
                it.vals.append(val)
                if val.first:
                    it.first_keys.append(val)
        it.n_distinct = n_distinct

    def _bytes(self, sc: _Sys, it: _Ent) -> Tuple[Dict[int, List[float]], Dict[str, List[float]]]:
        """Per-destination and per-template (up, down) bytes of this tick from act.stream."""
        if it.bytes is None:
            dest: Dict[int, List[float]] = {}
            tpl: Dict[str, List[float]] = {}
            rows = MT.stream_rows(sc.store, sc.s, it.e, sc.now)
            if rows.size:
                f = MT.stream_frac(sc.store, sc.s, it.e, sc.now)
                wr = 1.0 / f if (f == f and 0.0 < f <= 1.0) else 1.0
                up = rows["up"].astype(np.float64)
                dn = rows["down"].astype(np.float64)
                for key_arr, out, conv in ((rows["dest_id"], dest, None),
                                           (rows["token_id"], tpl, "tpl")):
                    ids, inv = np.unique(key_arr, return_inverse=True)
                    su = np.bincount(inv, weights=up) * wr
                    sd = np.bincount(inv, weights=dn) * wr
                    for i, k in enumerate(ids.tolist()):
                        if conv is None:
                            if k:
                                out[int(k)] = [float(su[i]), float(sd[i])]
                        else:
                            tk = MT.template_key(MT.token_str(sc.store, sc.s, k))
                            cur = out.get(tk)
                            if cur is None:
                                out[tk] = [float(su[i]), float(sd[i])]
                            else:
                                cur[0] += float(su[i])
                                cur[1] += float(sd[i])
            it.bytes = (dest, tpl)
        return it.bytes

    def _describe(self, sc: _Sys, it: _Ent, val: _Val, cls: Optional[Dict[str, Any]]) -> None:
        dim, v = val.dim, val.v
        val.bits = it.bo.bits(dim, v)
        val.ext = self._external(sc, dim, v)
        if dim in ("tmpl", "sni", "dns", "peer"):
            dest, tpl = self._bytes(sc, it)
            b = tpl.get(v) if dim == "tmpl" else dest.get(MT.dest_id_of(v))
            if b is not None:
                val.up, val.down = b
        val.upload = val.up >= max(UPLOAD_MIN, val.down)
        # rarity at the class (or, without a class, the system) tier
        key = val.key
        extra = 0.0
        if cls is not None:
            rec = (cls.get("adoption") or {}).get(key)
            built = cls.get("built", _NAN)
            if rec:
                extra = sum(1.0 for m, t in rec["members"].items()
                            if m != it.e and (not math.isfinite(built) or t > built))
            val.idf = V.idf(cls, dim, v, extra) if cls.get("members") else _NAN
            x = (cls.get("dims") or {}).get(dim, {}).get(v)
            cls_new = (x is None or x[1] < 0.5) and extra < 0.5 and bool(cls.get("members"))
        else:
            cls_new = False
        if sc.sys is not None and sc.sys.get("members"):
            val.idf_s = V.idf(sc.sys, dim, v)
            x = (sc.sys.get("dims") or {}).get(dim, {}).get(v)
            n_ent = float(sc.sys.get("n_ent", 0.0))
            val.share_s = (x[1] / n_ent if x is not None else 0.0) if n_ent > 0.0 else _NAN
        low_prev = not (val.share_s == val.share_s) or val.share_s <= LOW_PREV
        if val.new:
            sg = (sc.sys or {}).get("sightings") or {}
            x = (sc.sys.get("dims") or {}).get(dim, {}).get(v) if sc.sys else None
            s_new = sc.sys_ok and x is None and (key not in sg or sg[key][0] >= sc.now)
            if s_new:
                val.tier = "system"
            elif cls is not None and cls.get("members"):
                if cls_new or (val.idf == val.idf and val.idf >= IDF_RARE):
                    val.tier = "class"
            elif sc.sys_ok and val.idf_s == val.idf_s and val.idf_s >= IDF_RARE:
                val.tier = "class"
        rare = val.tier != "entity" or (val.idf == val.idf and val.idf >= IDF_RARE) or \
            (cls is None and val.idf_s == val.idf_s and val.idf_s >= IDF_RARE)
        if (val.sens or val.admin) and not rare:
            # round 4 (evaluator): a sensitive value is rare among the peers
            # when their estimated usage share is below LOW_PREV (peer_rare);
            # the smoothed IDF cannot certify it in a small class
            ref = cls if (cls is not None and cls.get("members")) else sc.sys
            rare = peer_rare(ref, dim, v, extra if ref is cls else 0.0)
        if (val.sens or val.admin) and rare and (val.new or val.bits >= BITS_EVENT):
            val.rare_sens = True
            if not val.new:
                val.tier = "class"             # known to the entity, rare among its peers
        # sensitivity weight in [1, 3]
        w = 1.0 + (1.0 if (val.sens or val.admin) else 0.0)
        heavy = max(4.0 * val.up, val.up + val.down)
        if low_prev and heavy > 1e4:
            w += min(1.0, math.log10(heavy / 1e4) / 2.0)
        val.w = min(3.0, w)
        if val.new and val.ext and dim in ("sni", "dns", "tmpl"):
            val.new_org = etld1(_host_of(dim, v)) not in sc.org_names()
        ax = []
        if val.sens or val.admin:
            ax.append("privilege")
        if val.ext and val.upload:
            ax.append("exfil")
        val.axes = ax or ["categorical"]

    # ------------------------------------------------------------ ledgers
    def _ledgers(self, ctx: Context, sc: _Sys, items: List[_Ent]) -> None:
        """System sightings and class adoption records from this tick's first
        sightings (all entities first, so same-tick adopters see each other)."""
        now = sc.now
        touched: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for it in items:
            for val in it.first_keys:
                if sc.sys is not None:
                    x = (sc.sys.get("dims") or {}).get(val.dim, {}).get(val.v)
                    if x is None:
                        sg = sc.sightings()
                        if val.key not in sg:
                            sg[val.key] = [now, it.e]
                if not it.ck:
                    continue
                cm = sc.cls_for_ledger(it.ck)
                ad = cm["adoption"]
                rec = ad.get(val.key)
                if rec is None:
                    rec = ad[val.key] = _new_ledger_rec(val.dim, val.v, now, val.tier)
                t0 = rec["members"].get(it.e)
                rec["members"][it.e] = now if t0 is None else min(float(t0), now)
                rec["last_ts"] = max(float(rec["last_ts"]), now)
                fl = rec["flags"]
                fl["external"] = fl["external"] or val.ext
                fl["upload"] = fl["upload"] or bool(val.upload and val.up > 0.0)
                fl["sensitive"] = fl["sensitive"] or val.sens or val.admin
                fl["new_eTLD1_org"] = fl["new_eTLD1_org"] or val.new_org
                touched[(it.ck, val.key)] = rec
        for (ck, key), rec in touched.items():
            n_class = sc.n_members(ck[len("class:"):])
            rec["n_class"] = n_class
            m7 = sum(1 for t in rec["members"].values() if float(t) >= now - ADOPT_WINDOW_S)
            if not rec["adopted"] and m7 >= max(ADOPT_MIN, math.ceil(ADOPT_FRAC * n_class)):
                rec["adopted"], rec["adopted_ts"] = True, now
                if not sc.training:
                    self._class_adopted(sc, ck, rec, m7)
        for ck, m in list(sc.touched.items()):
            if ck != SYSTEM_KEY:
                _prune_ledger(m.get("adoption") or {}, now)

    def _class_adopted(self, sc: _Sys, ck: str, rec: Dict[str, Any], m7: int) -> None:
        fl = rec["flags"]
        sc.store.add_event(BehaviorEvent(
            system=sc.s, entity=ck, ts=sc.now, kind="class_adopted", score=0.0,
            severity=Severity.INFO,
            description=(f"{m7} of {rec['n_class']} members of {ck} started using "
                         f"{rec['dim']} {rec['value']} within 7 d (class adoption)"),
            extra={"dim": rec["dim"], "value": rec["value"], "tier": "class",
                   "members": sorted(rec["members"]), "n_adopters": m7,
                   "n_class": rec["n_class"], "flags": dict(fl),
                   "external": fl["external"], "upload_dominant": fl["upload"],
                   "sensitive": fl["sensitive"], "new_external_domain": fl["new_eTLD1_org"]},
            axes=["categorical"],
            dedupe_key=f"class_adopted|{sc.s}|{ck}|{rec['dim']}|{rec['value']}",
            window=(sc.now - ADOPT_WINDOW_S, sc.now)))

    # ------------------------------------------------------------- pass 2
    def _entity(self, ctx: Context, lrn: G.GatedLearner, sc: _Sys, it: _Ent) -> int:
        store, now, dt = ctx.store, sc.now, sc.dt
        s, e, model = sc.s, it.e, it.model
        st, run = model["state"], model["run"]
        n = 0
        out: Optional[Dict[str, Any]] = None
        if it.degraded:
            cause = it.degraded
            emit.write_scores(store, s, e, now, {d: _NAN for d in DETECTORS},
                              degraded={d: cause for d in DETECTORS}, window_s=int(dt))
        elif it.obs:
            out = self._score(ctx, sc, it)
            self._write(store, s, e, now, dt, out)
            if not sc.training:
                self._events(sc, it, out)
            row = _Row(now, tuple((d, v, c) for d, vals in it.obs.items() for v, c in vals.items()),
                       float(out["k_rare"]), out["j1"], out["j24"])
            model["rows"][now] = row
            model["n_entries"] += len(row.items)
            seen = run["seen"]
            for val in it.vals:
                if val.new and val.key not in seen:
                    seen[val.key] = now
            n = 1
        # learn: commit rows <= now - D (the scores above used the pre-commit model)
        gate = model["gate"]
        self._rows, self._held = model["rows"], model["held_rows"]
        try:
            st, gate = lrn.step(store, s, e, st, gate, now, dt, training=ctx.training)
            st, gate = lrn.seed_from_link(store, s, e, st, gate,
                                          lambda src: _other_state(store, s, src))
        finally:
            self._rows, self._held = {}, {}
        model["state"], model["gate"] = st, gate
        _prune_run(run, st, now)
        _prune_rows(model, gate, now)
        model.update(version=int(gate.version), ts=now, class_key=it.ck)
        store.put_model(s, e, MODEL, model, version=int(gate.version))
        if out is not None:
            self._profile(store, s, e, model, out, now)
        return n

    def _score(self, ctx: Context, sc: _Sys, it: _Ent) -> Dict[str, Any]:
        model, now, dt = it.model, sc.now, sc.dt
        st, run = model["state"], model["run"]
        cls = sc.cls(it.ck)
        # adoption status of this entity's new values
        adoption = (cls or {}).get("adoption") or {}
        for val in it.vals:
            if val.new:
                rec = adoption.get(val.key)
                t = rec["members"].get(it.e) if rec else None
                val.adopted = bool(rec and rec.get("adopted") and t is not None
                                   and float(t) >= now - ADOPT_WINDOW_S)
        # maturity (Good-Turing unseen mass), pooled and per dimension
        f = V.factor(model, now)
        N_all = sum(st["N"].values()) * f
        N1_all = max(0.0, sum(st["N1"].values()) * f)
        omega = 1.0 - (min(1.0, N1_all / N_all) if N_all > 0.0 else 1.0)
        # surprisal term: per dimension the most surprising scored value
        best_all: Dict[str, float] = {}
        best_na: Dict[str, float] = {}
        for val in it.vals:
            if val.allow or not (val.new or val.rare_sens):
                continue
            Nd = st["N"].get(val.dim, 0.0) * f
            if Nd >= OMEGA_DIM_MIN_N:
                od = 1.0 - min(1.0, max(0.0, st["N1"].get(val.dim, 0.0) * f) / Nd)
            else:
                od = omega
            val.omega = od
            val.contrib = od * val.w * val.bits / BITS_SCALE
            if val.contrib > best_all.get(val.dim, -1.0):
                best_all[val.dim] = val.contrib
            if not val.adopted and val.contrib > best_na.get(val.dim, -1.0):
                best_na[val.dim] = val.contrib
        s_surp_all = sum(best_all.values())
        s_surp_na = sum(best_na.values())
        # Good-Turing novelty-rate test over the distinct values of the tick
        firsts = [v for v in it.vals if v.first and not v.allow]
        k_all = len(firsts)
        k_na = sum(1 for v in firsts if not v.adopted)
        n_d = max(it.n_distinct, k_all)
        p_new = (N1_all + 0.5) / (N_all + 1.0)
        if N_all >= RATE_MIN_N:
            p_all = float(special.betainc(k_all, n_d - k_all + 1, p_new)) if k_all else 1.0
            p_na = float(special.betainc(k_na, n_d - k_na + 1, p_new)) if k_na else 1.0
        else:
            p_all = p_na = _NAN
        s_rate_all, s_rate_na = _score_of(p_all), _score_of(p_na)
        raw = max(x for x in (s_rate_all, s_surp_all, 0.0) if x == x)
        score = max(x for x in (s_rate_na, s_surp_na, ADOPT_DISCOUNT * raw, 0.0) if x == x)
        # axes of what drove the score
        if s_rate_na == s_rate_na and s_rate_na >= s_surp_na and k_na:
            drivers = [v for v in firsts if not v.adopted]
        else:
            top = max(best_na.values()) if best_na else 0.0
            drivers = [v for v in it.vals if v.contrib > 0.0 and not v.adopted and not v.allow
                       and v.contrib >= 0.25 * top] if top > 0.0 else []
        axes = sorted({a for v in drivers for a in v.axes}) or ["categorical"]
        # novelty_rate: class-rare first sightings over the last 24 h
        k_rare = float(sum(1 for v in firsts if v.tier != "entity" and not v.adopted))
        nov = run["nov"]
        if k_rare:
            nov.append([now, k_rare])
        while nov and nov[0][0] <= now - DAY:
            nov.pop(0)
        nr_score, nr_alarm, nr_p = self._novelty_rate(st, nov, now, dt)
        # jsd: decayed windows against the smoothed profile
        for win, h in zip(run["win"], WIN_H):
            _win_add(win, h, now, it.obs)
        j1 = j24 = _NAN
        jsd_score, jsd_alarm, jsd_p = _NAN, None, _NAN
        # an immature profile is not a null: its JSD (~1 against the hyperprior)
        # would poison the conformal rings, so the statistic waits for maturity
        if N_all >= MATURE_N and now - run["jsd_t"] >= JSD_EVAL_S - 1e-6:
            run["jsd_t"] = now
            do24 = now - run.get("jsd24_t", -math.inf) >= JSD24_EVAL_S - 1e-6
            if do24:
                run["jsd24_t"] = now
            j1, j24 = self._jsd_stats(run, it.bo, now, do24)
            jsd_score, jsd_alarm, jsd_p = self._jsd_p(sc, it, st, run, j1, j24, dt)
        return {"novelty": score, "raw": raw, "pm": _clip_p(10.0 ** (-score)),
                "p_rate": p_na, "s_surp": s_surp_na, "omega": omega, "p_new": p_new,
                "axes": axes, "k_new": k_all, "k_adopted": k_all - k_na, "n": n_d,
                "k_rare": k_rare, "novelty_rate": nr_score, "nr_alarm": nr_alarm, "nr_p": nr_p,
                "jsd": jsd_score, "jsd_alarm": jsd_alarm, "jsd_p": jsd_p, "j1": j1, "j24": j24,
                "mature": N_all >= MATURE_N and omega >= MATURE_OMEGA, "N": N_all}

    @staticmethod
    def _novelty_rate(st: Dict[str, Any], nov: List[list], now: float,
                      dt: float) -> Tuple[float, Optional[int], float]:
        today = int(now // DAY)
        first_day = int(st["first"] // DAY) if math.isfinite(st["first"]) else today
        hist = [c for d, c in st["days"].items() if first_day < d < today]
        if len(hist) < NR_MIN_DAYS:
            return _NAN, None, _NAN
        x = int(round(sum(k for _, k in nov)))
        n_days = float(len(hist))
        a = NR_A0 + float(sum(hist))
        b = NR_A0 / NR_M0 + n_days
        mean = a / b
        kappa = 1e3
        if n_days >= 4:
            mu = float(np.mean(hist))
            var = float(np.var(hist, ddof=1))
            if var > mu > 0.0:
                kappa = mu * mu / (var - mu)
        r = bayes.nb_size(kappa, a)
        p = float(bayes.nb_sf(x - 1, mean, r)) if x >= 1 else 1.0
        p = _clip_p(p)
        budget = float(DETECTOR_INFO["novelty_rate"]["budget_per_day"])
        return _score_of(p), int(combine.e_day(p, dt) <= budget), p

    @staticmethod
    def _jsd_stats(run: Dict[str, Any], bo: V.Backoff, now: float,
                   do24: bool) -> Tuple[float, float]:
        """max over dimensions of the window-vs-profile JSD, for the 1-h window
        and (when due) the 24-h window; NaN where no dimension has enough mass."""
        probs: Dict[str, Dict[str, float]] = {}
        out = []
        for i, (win, h, min_mass) in enumerate(zip(run["win"], WIN_H, JSD_MIN_MASS)):
            if i == 1 and not do24:
                out.append(_NAN)
                continue
            f = 2.0 ** (-(now - win["t0"]) / h) if math.isfinite(win["t0"]) else 0.0
            best = _NAN
            for dim, dd in win["d"].items():
                tot = win["tot"].get(dim, 0.0)
                if not tot * f >= min_mass:
                    continue
                pd = probs.get(dim)
                if pd is None:
                    pd = probs[dim] = {}
                miss = [v for v in dd if v not in pd and v != _REST]
                if miss:
                    pd.update(zip(miss, bo.probs(dim, miss)))
                j = _jsd_dim(dd, tot, pd)
                if not j <= best:
                    best = j
            out.append(best)
        return out[0], out[1]

    def _jsd_p(self, sc: _Sys, it: _Ent, st: Dict[str, Any], run: Dict[str, Any],
               j1: float, j24: float, dt: float) -> Tuple[float, Optional[int], float]:
        ps = []
        for key, j in (("h1", j1), ("h24", j24)):
            ring = st["jsd"][key]
            if j != j or len(ring) < JSD_MIN_RING:
                continue
            tc = run["tail"].get(key)
            if tc is None or st["jsd_n"] - tc[0] >= JSD_REFIT or tc[0] > st["jsd_n"]:
                tc = run["tail"][key] = (st["jsd_n"], calib.fit_tail(ring, sc.now))
            u = combine.seeded_uniform(sc.s, it.e, "jsd", key, sc.now)
            ps.append(calib.p_from_ring(ring, j, u, tc[1]))
        ps = [p for p in ps if p == p]
        if not ps:
            return _NAN, None, _NAN
        p = _clip_p(min(1.0, len(ps) * min(ps)))
        budget = float(DETECTOR_INFO["jsd"]["budget_per_day"])
        return _score_of(p), int(combine.e_day(p, dt) <= budget), p

    # -------------------------------------------------------------- writes
    @staticmethod
    def _write(store, s: str, e: str, now: float, dt: float, out: Dict[str, Any]) -> None:
        scores = {"novelty": out["novelty"]}
        pm = {"novelty": out["pm"]}
        axes = {"novelty": out["axes"]}
        acc: Dict[str, int] = {}
        for d, pk, ak in (("novelty_rate", "nr_p", "nr_alarm"), ("jsd", "jsd_p", "jsd_alarm")):
            if out[d] == out[d]:
                scores[d] = out[d]
                pm[d] = out[pk]
                axes[d] = list(DETECTOR_INFO[d]["axes"])
                acc[d] = int(out[ak])
        emit.write_scores(store, s, e, now, scores, pm=pm, axes=axes, acc_alarm=acc or None,
                          window_s=int(dt))

    def _events(self, sc: _Sys, it: _Ent, out: Dict[str, Any]) -> None:
        run = it.model["run"]
        emitted = run["emitted"]
        cands: List[Tuple[float, str, _Val]] = []
        for val in it.vals:
            if val.allow:
                continue
            # entity tier: maturity-weighted bits, so an explorer's routine
            # discoveries (omega ~ 0.7) do not each become a finding
            if val.first and (val.tier == "system" or (out["mature"] and (
                    val.tier == "class" or val.omega * val.bits >= BITS_EVENT))):
                cands.append((val.contrib, "first_seen", val))
            if val.rare_sens:
                cands.append((val.contrib + 1e-9, "rare_access", val))
        if not cands:
            return
        cands.sort(key=lambda c: -c[0])
        n = 0
        dt = sc.dt
        for _, kind, val in cands:
            dk = f"{kind}|{sc.s}|{it.e}|{val.dim}|{val.v}"
            if dk in emitted:
                continue
            if n >= MAX_EVENTS:
                break
            emitted[dk] = sc.now
            n += 1
            self._emit(sc, it, out, kind, val, dk, dt)

    def _emit(self, sc: _Sys, it: _Ent, out: Dict[str, Any], kind: str, val: _Val,
              dk: str, dt: float) -> None:
        risky = val.sens or val.admin or (val.ext and val.upload)
        if kind == "rare_access":
            sev = Severity.LOW if val.adopted else Severity.MEDIUM
        elif val.adopted or val.tier == "entity":
            sev = Severity.INFO
        else:
            sev = Severity.MEDIUM if risky else Severity.LOW
        p_val = _clip_p(10.0 ** (-max(val.contrib, 0.0)))
        p_tick = out["pm"]
        low_prev = val.tier != "entity" or (val.share_s == val.share_s and val.share_s <= LOW_PREV)
        flags = {"sensitive": val.sens, "admin": val.admin, "external": val.ext,
                 "upload_dominant": bool(val.upload and val.up > 0.0),
                 "new_external_domain": bool(val.ext and val.new_org),
                 "low_prevalence": bool(low_prev)}
        tier_txt = {"system": "the system", "class": "the class", "entity": "this IP"}[val.tier]
        what = "first seen" if kind == "first_seen" else "rare sensitive access"
        desc = (f"{what}: {val.dim} {val.v} (new to {tier_txt}, {val.bits:.1f} bits"
                + (f", class IDF {val.idf:.2f}" if val.idf == val.idf else "")
                + (", adopted by the class" if val.adopted else "") + ")")
        sc.store.add_event(BehaviorEvent(
            system=sc.s, entity=it.e, ts=sc.now, kind=kind,
            score=float(min(1.0, val.contrib / 10.0)), severity=sev, description=desc,
            extra={"dim": val.dim, "value": val.v, "tier": val.tier, "bits": _r(val.bits),
                   "idf": _r(val.idf), "idf_system": _r(val.idf_s),
                   "share_system": _r(val.share_s),
                   "n_system": _r((sc.sys or {}).get("n_ent")),
                   "w": _r(val.w), "adopted": bool(val.adopted),
                   "discount": ADOPT_DISCOUNT if val.adopted else 1.0,
                   "reemergent": bool(val.reemerg), "class_key": it.ck,
                   "bytes_up": _r(val.up, 1), "bytes_down": _r(val.down, 1),
                   "flags": flags, **flags},
            p_value=p_val, e_day=float(combine.e_day(p_tick, dt)), axes=list(val.axes),
            p_by_detector={"novelty": p_tick}, dedupe_key=dk,
            model_version=int(it.model["gate"].version), window=(sc.now - dt, sc.now)))
        rec = {"ts": sc.now, "kind": kind, "dim": val.dim, "value": val.v, "tier": val.tier,
               "bits": _r(val.bits), "adopted": bool(val.adopted)}
        recent = it.model["run"]["recent"]
        recent.append(rec)
        del recent[:-10]

    def _profile(self, store, s: str, e: str, model: Dict[str, Any], out: Dict[str, Any],
                 now: float) -> None:
        prof = store.profile(s, e)
        if prof is None:
            prof = EntityProfile(system=s, entity=e, updated=now)
            store.put_profile(prof)
        cat = dict(prof.extra.get("categorical") or {})
        cat.update({
            "novelty_rate_gt": _r(V.novelty_rate(model)), "maturity": _r(out["omega"]),
            "n_accesses": _r(out["N"], 2), "class_key": model.get("class_key"),
            "version": int(model.get("version", 0)),
            "recent_new": list(model["run"]["recent"]),
            "last": {"ts": now, "novelty": _r(out["novelty"]), "novelty_raw": _r(out["raw"]),
                     "p_rate": _jp(out["p_rate"]), "surprisal_term": _r(out["s_surp"]),
                     "k_new": out["k_new"], "k_adopted": out["k_adopted"], "n": out["n"],
                     "novelty_rate": _r(out["novelty_rate"]), "jsd": _r(out["jsd"]),
                     "jsd_1h": _r(out["j1"]), "jsd_24h": _r(out["j24"]),
                     "axes": list(out["axes"])},
        })
        if self.entity_due(("vocab-portrait", s, e), now, PORTRAIT_REFIT_S):
            d = V.descriptors(model, k=5, now=now)
            cat.update(dims=d["dims"], families=d["families"])
        prof.extra["categorical"] = cat

    # --------------------------------------------------------------- tiers
    def _refit_tiers(self, store, s: str, now: float) -> int:
        """Rebuild the system tier and every live role-class tier from the member
        entity models (hourly per key, deterministic phase; first call at once)."""
        n = 0
        if self.entity_due(("vocab-tier", s, SYSTEM_KEY), now, self.tier_refit_s):
            n += self._build_tier(store, s, SYSTEM_KEY,
                                  PA.entities(store, s, now, getattr(self, "_pa_cfg", None)), now, "system")
        for ck in m_class.all_class_keys(store, s, min_members=m_class.MIN_MEMBERS):
            if not ck.startswith("class:") or ck.startswith(("class:static:", "class:pool:")):
                continue
            if self.entity_due(("vocab-tier", s, ck), now, self.tier_refit_s):
                n += self._build_tier(store, s, ck, m_class.class_members(store, s, ck), now,
                                      "class")
        return n

    @staticmethod
    def _build_tier(store, s: str, key: str, members: Sequence[str], now: float,
                    kind: str) -> int:
        dims: Dict[str, Dict[str, list]] = {}
        N: Dict[str, float] = {}
        n_ent = 0.0
        k = 0
        for m in members:
            em = V.get(store, s, m)
            if em is None or em.get("kind") != "entity":
                continue
            st = em.get("state") or {}
            if not st.get("n_rows") or not math.isfinite(st.get("t_ref", _NAN)):
                continue
            hl = float(st["H"])
            f = 2.0 ** (-(now - st["t_ref"]) / hl)
            k += 1
            n_ent += 2.0 ** (-max(0.0, now - st["clock"]) / hl)
            for dim, tab in st["dims"].items():
                out = dims.setdefault(dim, {})
                for v, x in tab.items():
                    c = x[0] * f
                    dfw = 2.0 ** (-max(0.0, now - x[3]) / hl)
                    y = out.get(v)
                    if y is None:
                        out[v] = [c, dfw, x[2], x[3], 1, x[1]]
                    else:
                        y[0] += c
                        y[1] += dfw
                        y[2] = min(y[2], x[2])
                        y[3] = max(y[3], x[3])
                        y[4] += 1
                        y[5] += x[1]
            for dim, tot in st["N"].items():
                N[dim] = N.get(dim, 0.0) + tot * f
        prev = V.get(store, s, key)
        if not k:
            return 0
        for dim, tab in dims.items():
            if len(tab) > V.TIER_CAP:
                keep = sorted(tab.items(), key=lambda kv: -kv[1][0])[:V.TIER_CAP]
                dims[dim] = dict(keep)
        N1 = {d: sum(y[0] for y in tab.values() if y[5] == 1.0) for d, tab in dims.items()}
        ver = int(prev.get("version", 0)) + 1 if prev else 1
        model = {"fmt": 1, "kind": kind, "version": ver, "built": now, "H": H, "dims": dims,
                 "N": N, "N1": N1, "n_ent": n_ent, "members": k}
        if kind == "class":
            ad = dict((prev or {}).get("adoption") or {})
            _prune_ledger(ad, now)
            model["adoption"] = ad
        else:
            sg = {key2: x for key2, x in ((prev or {}).get("sightings") or {}).items()
                  if now - float(x[0]) <= ADOPT_WINDOW_S
                  and V.split_key(key2)[1] not in dims.get(V.split_key(key2)[0], {})}
            if len(sg) > LEDGER_CAP:
                sg = dict(sorted(sg.items(), key=lambda kv: -float(kv[1][0]))[:LEDGER_CAP])
            model["sightings"] = sg
        store.put_model(s, key, MODEL, model, version=ver)
        return 1


# ================================================================= helpers
def _other_state(store, s: str, e: str) -> Optional[Dict[str, Any]]:
    m = V.get(store, s, e)
    return m.get("state") if m is not None and m.get("kind") == "entity" else None


def _iter_links(link: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    links = link.get("links")
    if isinstance(links, dict):
        links = links.values()
    for lk in links or ():
        if isinstance(lk, dict):
            yield lk


def _retracted(lk: Dict[str, Any]) -> bool:
    """The same retraction markers lib/gating honours for link seeding."""
    return bool(lk.get("retracted")) or lk.get("status") == "retracted" \
        or lk.get("state") == "retracted" or lk.get("active") is False


def _prune_ledger(ad: Dict[str, Dict[str, Any]], now: float) -> None:
    """Drop adoption records whose newest member sighting is older than 8 d; cap."""
    for key in [k for k, r in ad.items() if now - float(r.get("last_ts", now)) > LEDGER_KEEP_S]:
        del ad[key]
    if len(ad) > LEDGER_CAP:
        for key, _ in sorted(ad.items(), key=lambda kv: float(kv[1].get("last_ts", 0.0))
                             )[:len(ad) - LEDGER_CAP]:
            del ad[key]


def _prune_run(run: Dict[str, Any], st: Dict[str, Any], now: float) -> None:
    """Forget sightings the model has committed (or older than 30 d) and
    expired dedupe keys."""
    seen = run["seen"]
    if seen:
        dims = st["dims"]
        for key in list(seen):
            d, v = V.split_key(key)
            x = (dims.get(d) or {}).get(v)
            if (x is not None and now - x[3] <= REEMERGE_S) or now - seen[key] > SEEN_KEEP_S:
                del seen[key]
        if len(seen) > SEEN_CAP:
            for key, _ in sorted(seen.items(), key=lambda kv: kv[1])[:len(seen) - SEEN_CAP]:
                del seen[key]
    em = run["emitted"]
    if em:
        for key in [k for k, t in em.items() if now - t > EMIT_KEEP_S]:
            del em[key]


def _prune_rows(model: Dict[str, Any], gate: G.GateState, now: float) -> None:
    """Keep pending rows (ts > gate.last_ts), held rows, and committed rows for
    REPLAY_KEEP_S; at most ROW_ENTRY_CAP entries (oldest committed rows first,
    then the oldest held)."""
    rows, held = model["rows"], model["held_rows"]
    held_ts = {r.ts for r in gate.held} if gate.held else set()
    cut = now - REPLAY_KEEP_S
    last = gate.last_ts
    while rows:
        ts0 = next(iter(rows))
        if ts0 >= cut and model["n_entries"] <= ROW_ENTRY_CAP:
            break
        r = rows.pop(ts0)
        if ts0 > last or ts0 in held_ts:
            held[ts0] = r                          # still needed: keep aside
        else:
            model["n_entries"] -= len(r.items)
    if held:
        for ts0 in [t for t in held if t <= last and t not in held_ts]:
            r = held.pop(ts0)                      # released, rebased or dropped
            model["n_entries"] -= len(r.items)
        while held and model["n_entries"] > ROW_ENTRY_CAP:
            r = held.pop(next(iter(held)))
            model["n_entries"] -= len(r.items)
