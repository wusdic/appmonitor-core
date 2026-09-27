"""PortraitEngine (B30) — the human-readable behaviour library: a versioned
portrait with diffs for every IP and every class (role, static CIDR, pool).

Why: every other engine keeps its knowledge in a model tuned for scoring
(conjugate statistics, PPM tries, slot clocks, LDA spaces). An analyst, the
UI and the evaluation need the same knowledge as a description in natural
units: "who is this, when is it active, how much does it do, with what, how
distinctive is it, how settled is its model and how risky is it now", plus
"what changed since the last time we described it". This engine only READS
the owners' models, always through their lib/m_*.py accessors, so a portrait
can never disagree with what the detectors score against.

Per IP (every 8 ticks or 2 h; each key is refreshed at most once per
`refresh_s` with a per-key phase, Engine.entity_due, so a 60-s cadence does
not multiply the work):
  identity    system, IP, class path + membership probability, static
              classes / pool (m_class), identifiability = separability
              (1 - 2 EER_hard, B15) with a Wilson CI over its held-out
              windows, confusable_with, anonymity set, continuity / aliases /
              shared_ip (profile.extra.continuity from B17, attribution B16)
  role        automation index (mean of timing regularity (1 - B)/2, a
              confirmed strict period, the non-browser share of the client
              UA mix; unidentified components are left out), super class,
              activity mix from action-token families (m_vocab) and the lib-4
              categories matched over 7 d (store.matches)
  rhythm      the active window (m_rhythm.descriptors: longest run of slots
              with P(active) >= 0.5, else the 80 % window), weekday/weekend
              ratio, machine-likeness, periodicity (m_timing), schedule-shift
              history (the rhythm model's shift records)
  workload    per feature (requests, flows, bytes up/down) p10/p50/p90 per 15
              min on an ACTIVE tick per day type, and p5/p50/p95 per tick of
              the current cadence over all ticks (absence is data: the
              rhythm's inactive share is a point mass at 0). Each bin48 bucket
              of model.baseline's current anchor that holds committed data is
              one mixture component, read from m_baseline.bucket_means (mean
              and 15-min sd): a negative binomial for counts (its size r
              follows exactly from the two moments), a normal on the
              log1p(rate / min) scale for bytes (the t family's location and
              spread). The day-type mixture weighs the buckets by P(active)
              (m_rhythm.hourly48, else the bucket's committed weight). The
              mixture CDF is evaluated once per feature and exposure on a
              shared grid and inverted by interpolation between grid points.
  client      dominant stacks, UA / OS mix (m_client.descriptors)
  top         templates, SNI, ports, peers (m_vocab.top_values), top-3
              bigrams and trigrams (m_seq.top_ngrams)
  timing      burstiness, memory, think time, gap quantiles (m_timing)
  distinctive B15's Fisher / MCQ traits, else MCQ log-odds of the template
              vocabulary against the system tier
  stability   regime state / version / branch / history / rollbacks
              (m_governor.descriptor), baseline maturity and n_eff per bin48
              bucket, maturity = 1 - Good-Turing unseen mass per vocabulary
              dimension and for the sequence model
  risk        score, tier, trend and top reasons (profile.extra.risk, B26)
  timeline    the last 5 items of store.timeline (ids and kinds only)
Per class additionally: members with probabilities, cohesion (1 - mean JSD of
the members' family mixes to their centroid), B18's coherence, common tokens
(held by >= 50 % of the members), class-vs-system distinctive tokens (MCQ),
outlier members (robust JSD outliers and recent peer_outlier events), the
class_monitor aggregate bands and active-fraction heatmap, the adoption
history, class identifiability and class risk.

Versions and diffs: a compact signature of the portrait (the workday / non-
workday active windows, the per-day-type workload quantiles and the
categorical mixes: templates, activity families, client stacks, members) is
kept with the portrait. A new version is cut, and put_profile_version called,
when against the signature of the current version a categorical JSD exceeds
0.1 bits, a workload quantile moves by more than 25 % or an active-window edge
moves by more than 1 h (or the class path changes). Between versions the
json and texts are refreshed in place and the diff of the current version is
kept. Comparing against the version's signature, not the last run, means a
slow drift is reported once it has accumulated.

Privacy: path and query VALUES never reach a portrait. Templates are already
masked by R2; every HTTP token shown is re-masked here (placeholders kept,
every other path segment through lib/template.mask_segment, query strings
reduced to parameter names), and timeline items carry kinds and ids, never
descriptions written by other engines.

NaN policy: an unknown quantity is None in the json and is left out of the
texts; it is never replaced by a default that looks like a measurement.
ctx.training changes nothing (the engine emits no events). ctx.window_s sets
the exposure of the per-tick bands, so a 900 -> 60 s switch changes the tick
band's exposure but not the 15-min day-type bands the diffs compare.

Store:
  reads   profile.extra.* (peer_group, identity, continuity, attribution,
          risk, class_monitor), model.baseline, model.classagg, model.vocab,
          model.rhythm, model.client, model.seq, model.timing, model.class,
          model.identity, model.governor (all through lib/m_*),
          behavior.class (dict), store.matches, store.events (peer_outlier),
          store.timeline
  writes  profile.extra.portrait {json, text_zh, text_en, version, diff,
          updated, sig} for real entities and class keys;
          store.put_profile_version(s, key, version, {kind: 'portrait', ...})
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from scipy import special as sp

from ...core.engine import Context, Engine
from ...models.schema import EntityProfile
from .lib import features as F
from .lib import m_baseline as MB
from .lib import m_class
from .lib import m_client
from .lib import m_governor
from .lib import m_identity as MI
from .lib import m_rhythm as MR
from .lib import m_seq
from .lib import m_timing as MT
from .lib import m_vocab as MV
from .lib.classkeys import class_id, class_kind
from .lib.template import channel_of, mask_segment

PORTRAIT = "portrait"
CLASSAGG = "model.classagg"
CLASS_SERIES = "behavior.class"
REFRESH_S = 7200.0                 # per-key refresh (spec: every 8 ticks or 2 h)
DAY = 86400.0
CATS_WINDOW_S = 7 * DAY            # lib-4 categories and peer_outlier look-back
TIMELINE_N = 5
TIMELINE_SINCE_S = 30 * DAY

# workload bands (NB count / bytes features of FEATURE_SPEC v2)
WORKLOAD_FEATURES = ("http_requests", "flows", "bytes_up", "bytes_down")
HAS_DATA_W = 0.25                  # committed (decayed) weight for a bucket to count
WORKLOAD_IDX = tuple(F.FEATURE_INDEX[n] for n in WORKLOAD_FEATURES)
Q_DAYTYPE = (0.10, 0.50, 0.90)
Q_TICK = (0.05, 0.50, 0.95)
BAND_EXPOSURE_S = 900.0            # day-type bands: per 15 min ("每刻")
GRID_N = 40                        # log-spaced grid points of the mixture CDF
R_POISSON = 1e12                   # NB size standing for a Poisson bucket (bayes.nb_cdf)
MIN_BUCKET_W = 1e-3                # buckets below this share of the weight are dropped

# diff thresholds (engines.md B30)
DIFF_JSD = 0.1
DIFF_QSHIFT = 0.25
DIFF_WINDOW_H = 1.0
CAT_TOP = 20                       # values kept per categorical signature

# display sizes
TOP_TEMPLATES = 8
TOP_DEST = 5
TOP_NGRAMS = 3
COMMON_DF = 0.5                    # common token: held by >= 50 % of the members
MEMBER_TOP = 32                    # member values considered for df
MEMBER_MIN_SHARE = 0.01
OUTLIER_JSD_MIN = 0.2
WILSON_Z = 1.96

BROWSERS = frozenset({"chrome", "edge", "firefox", "safari"})
LIBRARIES = frozenset({"python-requests", "curl", "go-http-client", "java", "okhttp",
                       "wget", "postman"})
_PLACEHOLDER = re.compile(r"^\{[a-z0-9_]+\}$")

SUPER_ZH = {"human": "人工交互用户", "machine": "自动化程序"}
SUPER_EN = {"human": "human-interactive user", "machine": "automated client"}
TIER_ZH = {"low": "低", "medium": "中", "high": "高", "critical": "严重", "info": "提示"}
KIND_ZH = {"role": "角色类", "static": "静态网段", "pool": "地址池"}
KIND_EN = {"role": "role class", "static": "static CIDR class", "pool": "address pool"}
FEATURE_ZH = {"http_requests": "请求", "flows": "连接", "bytes_up": "上行字节",
              "bytes_down": "下行字节"}
FEATURE_EN = {"http_requests": "requests", "flows": "flows", "bytes_up": "bytes up",
              "bytes_down": "bytes down"}
DAYTYPE_ZH = {"workday": "工作日", "nonworkday": "休息日"}
LABELS = {  # diff field -> (zh, en)
    "rhythm.window": ("活跃时段", "active window"),
    "workload": ("工作负载", "workload"),
    "top_templates": ("常用模板", "top templates"),
    "activity_mix": ("活动构成", "activity mix"),
    "client": ("终端栈", "client stacks"),
    "members": ("成员", "members"),
    "class_path": ("类别", "class path"),
}
_NAN = math.nan


class PortraitEngine(Engine):
    name = "behavior.portrait"
    layer = "behavior"
    consumes = ["profile.extra", "model.baseline", "model.classagg", "model.vocab",
                "model.rhythm", "model.client", "model.seq", "model.timing", "model.class",
                "model.identity", "model.governor", "behavior.class", "store.timeline",
                "store.matches"]
    produces = ["profile.extra.portrait", "profile.version"]
    description = ("Versioned natural-unit portraits of every IP and class (identity, role, "
                   "rhythm window, workload bands per day type, client stacks, top "
                   "templates / destinations / n-grams, timing, distinctive traits, "
                   "stability, risk, timeline; class members, cohesion, bands, heatmap, "
                   "adoptions) with zh/en texts and diffs (JSD > 0.1, quantile shift > 25 %, "
                   "window change > 1 h).")
    interval = 8
    period_s = 7200.0

    def __init__(self, refresh_s: float = REFRESH_S, **params: Any) -> None:
        super().__init__(**params)
        self.refresh_s = float(refresh_s)

    # ------------------------------------------------------------------ run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store, now = ctx.store, float(ctx.now)
        dt = float(ctx.window_s) if ctx.window_s and ctx.window_s > 0 else BAND_EXPOSURE_S
        n = 0
        for s in store.systems():
            sysc = _SysCtx(store, s, now, dt)
            for e in store.entities(s):
                if not self.entity_due(("portrait", s, e), now, self.refresh_s):
                    continue
                js, sig = entity_portrait(sysc, e)
                _publish(store, s, e, js, sig, now)
                n += 1
            for ck in m_class.all_class_keys(store, s):
                if not self.entity_due(("portrait", s, ck), now, self.refresh_s):
                    continue
                js, sig = class_portrait(sysc, ck)
                _publish(store, s, ck, js, sig, now)
                n += 1
        return n


# ====================================================================== context
class _SysCtx:
    """Per-(system, run) reads shared by every portrait of the system."""

    def __init__(self, store: Any, s: str, now: float, dt: float) -> None:
        self.store, self.s, self.now, self.dt = store, s, now, dt
        self.identity = MI.get(store, s)
        self.sys_vocab = MV.get(store, s, "__system__")
        self._sys_tmpl: Optional[Dict[str, float]] = None
        self._role_n: Dict[str, int] = {}

    def role_size(self, rid: Optional[str]) -> int:
        """Members of a role in this system (cached: m_class.members is O(N))."""
        if rid is None:
            return 0
        if rid not in self._role_n:
            self._role_n[rid] = len(m_class.members(self.store, self.s, rid))
        return self._role_n[rid]

    def sys_tmpl(self) -> Dict[str, float]:
        if self._sys_tmpl is None:
            self._sys_tmpl = MV.counts(self.sys_vocab, "tmpl", self.now) if self.sys_vocab else {}
        return self._sys_tmpl


# ================================================================= entity
def entity_portrait(c: _SysCtx, e: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(json, signature) of one real entity."""
    store, s, now = c.store, c.s, c.now
    prof = store.profile(s, e)
    extra = prof.extra if prof is not None and isinstance(prof.extra, dict) else {}
    vm = MV.get(store, s, e)
    rm = _rhythm_model(store, s, e)
    tm = store.get_model(s, e, MT.MODEL, default=None)
    tm = tm if isinstance(tm, Mapping) and not MT.is_empty(tm) else None
    cm = m_client.get(store, s, e)
    sm = m_seq.get(store, s, e)
    bm = store.get_model(s, e, MB.MODEL, default=None)

    p48 = MR.profile48(rm) if rm is not None else None
    rhythm = _rhythm_block(rm, p48)
    timing = _timing_block(tm, now)
    if rhythm and timing:
        rhythm["periodicity"] = {"period_s": timing.get("period"), "p": timing.get("period_p")}
    client = _client_block(cm, now)
    js: Dict[str, Any] = {
        "kind": "entity", "system": s, "entity": e, "ts": now,
        "last_seen": _num(store.last_seen(s, e)), "first_seen": _num(store.first_seen(s, e)),
        "identity": _identity_block(c, e, extra),
        "role": _role_block(c, e, vm, timing, client),
        "rhythm": rhythm,
        "active_hours": (rhythm.get("workday") or rhythm.get("any") or {}).get("window"),
        "workload": _workload_block(_anchor(bm), _p_active_hourly(p48), c.dt, per_tick=True),
        "client": client,
        "top_templates": _top_templates(vm, now),
        "top_sni": _top_values(vm, "sni", now),
        "top_ports": _top_values(vm, "dport", now),
        "top_peers": _top_values(vm, "peer", now),
        "top_ngrams": _ngrams(sm),
        "timing": timing,
        "distinctive": _distinctive(c, e, vm),
        "stability": _stability(store, s, e, bm, vm, sm, rm),
        "risk": _risk(extra),
        "timeline": _timeline(store, s, e, now),
    }
    js["status"] = "active" if (vm or rm or _anchor(bm) is not None or tm or cm) else "no_model"
    return _clean(js), _signature(js)


def _identity_block(c: _SysCtx, e: str, extra: Mapping[str, Any]) -> Dict[str, Any]:
    store, s = c.store, c.s
    a = m_class.assignment(store, s, e) or {}
    pg = extra.get("peer_group") if isinstance(extra.get("peer_group"), Mapping) else {}
    rid = m_class.role_id(store, s, e)
    role_n = c.role_size(rid)
    out: Dict[str, Any] = {
        "system": s, "ip": e,
        "role": rid, "role_name": m_class.role_name(store, rid) if rid is not None else None,
        "class_path": a.get("class_path") or pg.get("class_path"),
        "class_prob": _num(a.get("prob", pg.get("prob"))),
        "super": a.get("super") or pg.get("super"),
        "class_size": role_n or None,
        "static_classes": list(a.get("static") or pg.get("static_classes") or []),
        "pool": a.get("pool") or pg.get("pool"),
        "provisional": bool(a.get("provisional", False)),
    }
    st = MI.stats(c.identity, e) if c.identity is not None else {}
    idx = extra.get("identity") if isinstance(extra.get("identity"), Mapping) else {}
    src = st or idx
    sep = _num(src.get("separability"))
    n_w = src.get("n_windows")
    out.update({
        "identifiability": sep,
        "identifiability_ci": _wilson(sep, n_w),
        "recall1": _num(src.get("recall1")),
        "eer_hard": _num(src.get("eer_hard")),
        "n_windows": int(n_w) if isinstance(n_w, (int, float)) and n_w == n_w else None,
        "confusable_with": list(src.get("confusable_with") or [])[:5],
        "anonymity_set": (MI.anonymity_set(c.identity, e) if c.identity is not None
                          else list(idx.get("anonymity_set") or [e])),
    })
    cont = extra.get("continuity") if isinstance(extra.get("continuity"), Mapping) else {}
    att = extra.get("attribution") if isinstance(extra.get("attribution"), Mapping) else {}
    out["continuity"] = {
        "continuity_id": cont.get("continuity_id"),
        "aliases": list(cont.get("aliases") or []),
        "linked_from": cont.get("linked_from"), "linked_to": cont.get("linked_to"),
        "entity_kind": cont.get("entity_kind"),
    }
    out["shared_ip"] = bool(cont.get("shared_ip") or att.get("shared_ip") or False)
    return out


def _role_block(c: _SysCtx, e: str, vm: Optional[Mapping], timing: Mapping[str, Any],
                client: Mapping[str, Any]) -> Dict[str, Any]:
    parts: Dict[str, float] = {}
    b = timing.get("B")
    if isinstance(b, float) and math.isfinite(b):
        parts["regularity"] = min(1.0, max(0.0, (1.0 - b) / 2.0))
    if timing.get("n_eff"):
        per, pp = timing.get("period"), timing.get("period_p")
        parts["periodic"] = 1.0 if (per is not None and pp is not None and pp <= 0.01) else 0.0
    br = lib = 0.0
    for ua, sh in (client.get("ua_mix") or {}).items():
        fam = str(ua).split("/", 1)[0]
        if fam in BROWSERS:
            br += float(sh)
        elif fam in LIBRARIES:
            lib += float(sh)
    if br + lib > 0.0:
        parts["nonbrowser"] = lib / (br + lib)
    auto = float(np.mean(list(parts.values()))) if parts else None
    fam = MV.family_distribution(vm, now=c.now) if vm else {}
    cats: Dict[str, float] = {}
    for m in c.store.matches(c.s, e, since=c.now - CATS_WINDOW_S, limit=1000):
        cats[str(m.category)] = cats.get(str(m.category), 0.0) + 1.0
    tot = sum(cats.values())
    return {
        "automation_index": auto, "automation_parts": parts,
        "activity_mix": [[_safe_family(f), v] for f, v in _top_items(fam, 6)],
        "categories_7d": {k: v / tot for k, v in sorted(cats.items())} if tot > 0 else {},
        "n_matches_7d": int(tot),
    }


# ================================================================== class
def class_portrait(c: _SysCtx, ck: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(json, signature) of one class key."""
    store, s, now = c.store, c.s, c.now
    prof = store.profile(s, ck)
    extra = prof.extra if prof is not None and isinstance(prof.extra, dict) else {}
    kind = class_kind(ck) or "role"
    cid = class_id(ck)
    mem = m_class.class_members(store, s, ck)
    probs = {}
    for ip in mem:
        a = m_class.assignment(store, s, ip) or {}
        probs[ip] = _num(a.get("prob", 1.0)) if kind == "role" else 1.0
    name = m_class.role_name(store, cid) if kind == "role" else cid

    # members' vocabularies: family mixes (cohesion / outliers) and tmpl df
    fams: Dict[str, Dict[str, float]] = {}
    tmpl_sum: Dict[str, float] = {}
    df: Dict[str, int] = {}
    for ip in mem:
        vm = MV.get(store, s, ip)
        if vm is None:
            continue
        f = MV.family_distribution(vm, now=now)
        if f:
            fams[ip] = f
        for v, sh in MV.top_values(vm, "tmpl", MEMBER_TOP, now):
            if sh >= MEMBER_MIN_SHARE:
                df[v] = df.get(v, 0) + 1
        for v, x in MV.counts(vm, "tmpl", now).items():
            tmpl_sum[v] = tmpl_sum.get(v, 0.0) + x
    cohesion, outliers = _cohesion(fams)
    for ip in mem:
        if ip not in outliers and store.events(s, ip, since=now - CATS_WINDOW_S,
                                               kinds=["peer_outlier"], limit=1):
            outliers.append(ip)
    n_mem = len(mem)
    common = sorted(((v, k) for v, k in df.items() if n_mem and k / n_mem >= COMMON_DF),
                    key=lambda t: (-t[1], -tmpl_sum.get(t[0], 0.0), t[0]))
    tot = sum(tmpl_sum.values())
    cm = extra.get("class_monitor") if isinstance(extra.get("class_monitor"), Mapping) else {}
    agg_model = store.get_model(s, ck, CLASSAGG, default=None)
    rm = _rhythm_model(store, s, ck)
    rhythm = _rhythm_block(rm, MR.profile48(rm) if rm is not None else None)
    heat = _heatmap(cm, rm)
    if not rhythm.get("workday") and heat is not None:
        rhythm = dict(rhythm, **_window_from_frac(heat))
    cls_series = store.latest_derived(s, ck, CLASS_SERIES)
    cval = cls_series.value if cls_series is not None and isinstance(cls_series.value, Mapping) \
        else {}
    idc = MI.class_stats(c.identity, ck) if c.identity is not None else {}
    if not idc and isinstance(extra.get("identity"), Mapping):
        idc = dict(extra["identity"])
    sep = _num(idc.get("separability", idc.get("identifiability")))
    js: Dict[str, Any] = {
        "kind": "class", "class_kind": kind, "system": s, "entity": ck, "name": name,
        "ts": now,
        "members": [{"ip": ip, "prob": probs[ip]} for ip in mem],
        "n_members": n_mem,
        "cohesion": cohesion,
        "coherence": _num(cval.get("coherence")),
        "active_frac_now": _num(cval.get("active_frac")),
        "common_tokens": [{"template": _safe_token(v), "df": k / n_mem,
                           "share": (tmpl_sum.get(v, 0.0) / tot) if tot > 0 else None}
                          for v, k in common[:TOP_TEMPLATES]],
        "distinctive_tokens": _class_distinctive(c, tmpl_sum),
        "outlier_members": sorted(outliers),
        "top_templates": [{"template": _safe_token(v), "share": x / tot}
                          for v, x in _top_items(tmpl_sum, TOP_TEMPLATES)] if tot > 0 else [],
        "rhythm": rhythm,
        "active_hours": (rhythm.get("workday") or rhythm.get("any") or {}).get("window"),
        "bands": _class_bands(cm),
        "workload": _workload_block(_anchor(agg_model), None, c.dt, per_tick=False),
        "active_frac_heatmap": heat,
        "adoption": _adoption(store, s, ck, cm, kind),
        "identity": {"identifiability": sep,
                     "identifiability_ci": _wilson(sep, idc.get("n_windows")),
                     "recall1": _num(idc.get("recall1")),
                     "confusable_with": list(idc.get("confusable_with") or [])[:5],
                     "n_windows": idc.get("n_windows")},
        "client": _class_client(store, s, ck, mem, now),
        "stability": {"regime": m_governor.descriptor(store, s, ck),
                      "classagg_version": _int(agg_model.get("version"))
                      if isinstance(agg_model, Mapping) else None,
                      "n_eff": _num(cm.get("n_eff")) if cm else
                      (_num(MB.n_eff(_anchor(agg_model))) if _anchor(agg_model) is not None
                       else None)},
        "risk": _risk(extra),
        "timeline": _timeline(store, s, ck, now),
    }
    return _clean(js), _signature(js)


def _cohesion(fams: Mapping[str, Mapping[str, float]]) -> Tuple[Optional[float], List[str]]:
    """1 - mean JSD (bits) of the members' family mixes to their centroid, and
    the robust outliers (JSD > max(0.2, median + 3 * 1.4826 MAD))."""
    if len(fams) < 2:
        return None, []
    keys = sorted({k for f in fams.values() for k in f})
    ix = {k: i for i, k in enumerate(keys)}
    ips = sorted(fams)
    P = np.zeros((len(ips), len(keys)))
    for r, ip in enumerate(ips):
        for k, v in fams[ip].items():
            P[r, ix[k]] = v
    P /= np.maximum(P.sum(axis=1, keepdims=True), 1e-300)
    cen = P.mean(axis=0)
    d = np.array([MT.jsd_bits(P[r], cen) for r in range(len(ips))])
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    thr = max(OUTLIER_JSD_MIN, med + 3.0 * 1.4826 * mad)
    return 1.0 - float(d.mean()), [ip for ip, x in zip(ips, d) if x > thr]


def _class_distinctive(c: _SysCtx, tmpl_sum: Mapping[str, float]) -> List[Dict[str, Any]]:
    sysc = c.sys_tmpl()
    if not tmpl_sum or not sysc:
        return []
    rest = {v: max(0.0, x - tmpl_sum.get(v, 0.0)) for v, x in sysc.items()}
    if not sum(rest.values()) > 0.0:
        return []
    out = [{"template": _safe_token(v), "z": z, "log_odds": dl}
           for v, (dl, z) in MI.mcq_log_odds(tmpl_sum, rest, sysc).items() if z >= WILSON_Z]
    out.sort(key=lambda d: -d["z"])
    return out[:5]


def _class_bands(cm: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """B18's aggregate p5/p50/p95 for the current bucket (15-min exposure)."""
    agg = cm.get("aggregate") if isinstance(cm, Mapping) else None
    if not isinstance(agg, Mapping) or not isinstance(agg.get("features"), Mapping):
        return None
    feats = {}
    for f, q in agg["features"].items():
        if isinstance(q, (list, tuple)) and len(q) == 3:
            feats[f] = {"p5": _num(q[0]), "p50": _num(q[1]), "p95": _num(q[2])}
    return {"bucket": cm.get("bucket"), "exposure_s": agg.get("exposure_s", BAND_EXPOSURE_S),
            "features": feats}


def _heatmap(cm: Mapping[str, Any], rm: Optional[Mapping]) -> Optional[Dict[str, List]]:
    """Active fraction by hour and day type: B18's members-active fraction,
    else the class rhythm's P(active) per hour."""
    fr = cm.get("active_frac_by_bin") if isinstance(cm, Mapping) else None
    if isinstance(fr, (list, tuple)) and len(fr) == 48:
        v = [_num(x) for x in fr]
    elif rm is not None:
        v = [_num(x) for x in (MR.hourly48(rm) / MR.QUARTERS).tolist()]
    else:
        return None
    return {"workday": v[:24], "nonworkday": v[24:]}


def _window_from_frac(heat: Mapping[str, List]) -> Dict[str, Any]:
    """Workday window (longest run of hours with an active fraction >= 0.5)
    when the class has no rhythm model of its own."""
    out: Dict[str, Any] = {}
    for dtp in ("workday", "nonworkday"):
        v = np.array([x if x is not None else 0.0 for x in heat.get(dtp) or []], dtype=float)
        if v.size != 24:
            continue
        run = _longest_run(v >= 0.5)
        if run is not None:
            s0, ln = run
            out[dtp] = {"window": _fmt_window(s0, s0 + ln), "start_h": float(s0),
                        "len_h": float(ln), "source": "active_frac"}
    return out


def _adoption(store: Any, s: str, ck: str, cm: Mapping[str, Any], kind: str) -> Dict[str, Any]:
    ad = cm.get("adoption") if isinstance(cm, Mapping) else None
    out: Dict[str, Any] = {"rate": _num(ad.get("rate")) if isinstance(ad, Mapping) else None,
                           "history": []}
    recs = MV.adoption_records(store, s, ck) if kind == "role" else []
    for r in recs[:5]:
        v = r.get("value")
        out["history"].append({
            "dim": r.get("dim"), "value": _safe_token(v) if r.get("dim") == "tmpl" else v,
            "n_members": len(r.get("members") or {}), "first_ts": _num(r.get("first_ts")),
            "adopted": bool(r.get("adopted")), "flags": dict(r.get("flags") or {})})
    if not out["history"] and isinstance(ad, Mapping):
        for h in list(ad.get("recent") or [])[-5:]:
            if isinstance(h, Mapping):
                d = dict(h)
                if d.get("dim") == "tmpl" and "value" in d:
                    d["value"] = _safe_token(d["value"])
                out["history"].append(d)
    return out


def _class_client(store: Any, s: str, ck: str, mem: Sequence[str], now: float
                  ) -> Dict[str, Any]:
    """Class stack mix: the system model's class tier, else the members' mean shares."""
    tier = m_client.class_tier(m_client.get(store, s, "__system__"), ck)
    sh: Dict[str, float] = {}
    if tier is not None:
        sh = m_client.shares(tier, now)
    else:
        n = 0
        for ip in mem:
            x = m_client.shares(m_client.get(store, s, ip), now)
            if x:
                n += 1
                for t, v in x.items():
                    sh[t] = sh.get(t, 0.0) + v
        sh = {t: v / n for t, v in sh.items()} if n else {}
    dom = []
    for t, v in _top_items(sh, 3):
        p = m_client.parse(t)
        dom.append({"token": t, "ua": p.ua, "os": p.os, "share": v})
    return {"dominant": dom, "n_stacks": len(sh), "shares": dict(_top_items(sh, CAT_TOP))}


# ============================================================ shared blocks
def _rhythm_model(store: Any, s: str, key: str) -> Optional[Mapping[str, Any]]:
    m = store.get_model(s, key, MR.MODEL, default=None)
    return m if isinstance(m, Mapping) and isinstance(m.get("state"), Mapping) else None


def _p_active_hourly(p48: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """P(active) per bin48 hour (m_rhythm.hourly48 / 4) from a profile48."""
    if p48 is None:
        return None
    return np.clip(np.asarray(p48).reshape(48, MR.QUARTERS).mean(axis=1), 0.0, 1.0)


def _rhythm_block(rm: Optional[Mapping], p48: Optional[np.ndarray]) -> Dict[str, Any]:
    if rm is None or p48 is None:
        return {}
    d = MR.descriptors(rm)
    win = d.get("active_window") or d.get("window80")
    out: Dict[str, Any] = {
        "wd_we_ratio": d.get("wd_we_ratio"), "machine_like": d.get("machine_like"),
        "entropy168": d.get("entropy168"), "mu_h": d.get("mu_h"), "R": d.get("R"),
        "active_slots_wd": d.get("active_slots_wd"), "active_slots_nwd": d.get("active_slots_nwd"),
        "window80": _win_str(d.get("window80")), "mature168": d.get("mature168"),
        "n_slots": d.get("n_slots"), "span_days": d.get("span_days"),
        "shifts": [dict(x) for x in list(rm.get("shifts") or [])[-4:]],
    }
    if win:
        blk = {"window": _win_str(win), "start": win.get("start"), "end": win.get("end"),
               "start_h": win.get("start_h"), "len_h": win.get("len_h"),
               "source": "active_window" if d.get("active_window") else "window80"}
        out[d.get("day_type") or "workday"] = blk
        out["any"] = blk
    # the other day type's window from the posterior profile, only where that
    # day type was ever seen active (else the posterior only echoes the prior)
    act = np.nan_to_num(MR.activity48(rm), nan=0.0)
    for dtp, sl in (("workday", slice(0, 96)), ("nonworkday", slice(96, 192))):
        if dtp in out or not float(act[sl].sum()) > 0.0:
            continue
        run = _longest_run(p48[sl] >= 0.5)
        if run is not None:
            s0, ln = run
            out[dtp] = {"window": _fmt_window(s0 / 4.0, (s0 + ln) / 4.0),
                        "start_h": s0 / 4.0, "len_h": ln / 4.0, "source": "active_window"}
    return out


def _timing_block(tm: Optional[Mapping], now: float) -> Dict[str, Any]:
    if tm is None:
        return {}
    d = MT.descriptors(tm, now)
    return {k: _num(v) for k, v in d.items()}


def _client_block(cm: Optional[Mapping], now: float) -> Dict[str, Any]:
    if cm is None or m_client.kind(cm) != "entity":
        return {}
    d = m_client.descriptors(cm, k=3, now=now)
    return {
        "dominant": [{"token": x["token"], "ua": x["ua"], "os": x["os"], "share": x["share"],
                      "first_ts": x.get("first_ts"), "last_ts": x.get("last_ts")}
                     for x in d.get("dominant") or []],
        "n_stacks": d.get("n_stacks"), "entropy_bits": _num(d.get("entropy_bits")),
        "ua_mix": d.get("ua_mix") or {}, "os_mix": d.get("os_mix") or {},
        "maturity": _num(d.get("maturity")),
        "shares": dict(_top_items(m_client.shares(cm, now), CAT_TOP)),
    }


def _top_templates(vm: Optional[Mapping], now: float) -> List[Dict[str, Any]]:
    return [{"template": _safe_token(v), "share": sh}
            for v, sh in MV.top_values(vm, "tmpl", TOP_TEMPLATES, now)] if vm else []


def _top_values(vm: Optional[Mapping], dim: str, now: float) -> List[Dict[str, Any]]:
    return [{"value": v, "share": sh} for v, sh in MV.top_values(vm, dim, TOP_DEST, now)] \
        if vm else []


def _ngrams(sm: Optional[Mapping]) -> List[Dict[str, Any]]:
    if sm is None:
        return []
    out = []
    for n in (2, 3):
        for g in m_seq.top_ngrams(sm, n, TOP_NGRAMS):
            out.append({"n": n, "ngram": [_safe_token(x) for x in g["ngram"]],
                        "count": g["count"]})
    return out


def _distinctive(c: _SysCtx, e: str, vm: Optional[Mapping]) -> Dict[str, Any]:
    d = MI.distinctive(c.identity, e) if c.identity is not None else {}
    if d:
        voc = []
        for x in d.get("vocab") or []:
            x = dict(x)
            if x.get("dim") == "tmpl":
                x["value"] = _safe_token(x.get("value"))
            voc.append(x)
        return {"source": "identity", "features": list(d.get("features") or []), "vocab": voc}
    own = MV.counts(vm, "tmpl", c.now) if vm else {}
    sysc = c.sys_tmpl()
    if not own or not sysc:
        return {"source": None, "features": [], "vocab": []}
    rest = {v: max(0.0, x - own.get(v, 0.0)) for v, x in sysc.items()}
    if not sum(rest.values()) > 0.0:
        return {"source": None, "features": [], "vocab": []}
    voc = [{"dim": "tmpl", "value": _safe_token(v), "z": z, "log_odds": dl}
           for v, (dl, z) in MI.mcq_log_odds(own, rest, sysc).items() if z >= WILSON_Z]
    voc.sort(key=lambda x: -x["z"])
    return {"source": "vocab_vs_system", "features": [], "vocab": voc[:5]}


def _stability(store: Any, s: str, e: str, bm: Any, vm: Optional[Mapping],
               sm: Optional[Mapping], rm: Optional[Mapping]) -> Dict[str, Any]:
    anc = _anchor(bm)
    mat: Dict[str, Any] = {}
    if vm:
        for dim in MV.DIMS:
            if MV.total(vm, dim) > 0.0:
                mat[dim] = MV.maturity(vm, dim)
    if sm is not None:
        mat["seq"] = m_seq.maturity(sm)
    neff = None
    if anc is not None:
        St = MB.true_stats(anc)
        if St is not None:
            neff = [round(float(x), 2) for x in St[:, MB.FULL.W[F.FEATURE_INDEX["flows"]]]]
    return {
        "regime": m_governor.descriptor(store, s, e),
        "baseline": MB.maturity(bm) if isinstance(bm, Mapping) and anc is not None else None,
        "maturity": mat,
        "n_eff_by_bin48": neff,
        "rhythm": ({"n_slots": int((rm.get("state") or {}).get("n_slots", 0) or 0),
                    "mature168": MR.mature168(rm)} if rm is not None else None),
    }


def _risk(extra: Mapping[str, Any]) -> Dict[str, Any]:
    r = extra.get("risk") if isinstance(extra.get("risk"), Mapping) else None
    if not r:
        return {}
    reasons = [{k: x.get(k) for k in ("key", "source", "name", "share", "L")}
               for x in list(r.get("top_reasons") or [])[:5] if isinstance(x, Mapping)]
    return {"score": _num(r.get("score")), "tier": r.get("tier"), "trend": _num(r.get("trend")),
            "stages": list(r.get("stages") or []), "top_reasons": reasons,
            "degraded": bool(r.get("degraded", False))}


def _timeline(store: Any, s: str, key: str, now: float) -> List[Dict[str, Any]]:
    out = []
    for it in store.timeline(s, key, since=now - TIMELINE_SINCE_S, limit=TIMELINE_N):
        x, typ = it.get("item"), it.get("type")
        d: Dict[str, Any] = {"ts": _num(it.get("ts")), "type": typ}
        if typ == "event":
            d.update(kind=x.kind, severity=_sev(x.severity), status=x.status, id=x.id,
                     score=_num(x.score))
        elif typ == "match":
            d.update(category=x.category, signature_id=x.signature_id, severity=_sev(x.severity))
        elif typ == "incident":
            d.update(id=x.id, status=x.status, severity=_sev(x.severity),
                     kinds=list(x.kinds or []))
        elif typ == "profile_version":
            obj = x.obj
            d.update(version=x.version, kind=obj.get("kind") if isinstance(obj, Mapping) else None)
        elif typ == "risk":
            v = x.value
            d.update(value=v.get("tier", v.get("score")) if isinstance(v, Mapping) else _num(v))
        out.append(d)
    return out


# ================================================================ workload
def _anchor(model: Any) -> Optional[Any]:
    a = model.get("current") if isinstance(model, Mapping) else None
    return a if isinstance(a, MB.Anchor) and not a.empty else None


def _workload_block(anc: Optional[Any], pa: Optional[np.ndarray], dt: float,
                    per_tick: bool) -> Dict[str, Any]:
    """Natural-unit workload bands (see module doc). Only buckets holding
    committed data (weight >= HAS_DATA_W) enter a mixture: an unvisited
    bucket's predictive is the hyperprior and would read like a measurement."""
    if anc is None:
        return {}
    mean, sd15 = MB.bucket_means(anc)
    St = MB.true_stats(anc)
    w_data = np.maximum(St[:, MB.FULL.W[F.FEATURE_INDEX["flows"]]], 0.0) if St is not None \
        else np.zeros(48)
    has = w_data >= HAS_DATA_W
    w_act = (pa if pa is not None else w_data) * has
    occ = np.r_[np.full(24, 5.0 / 7.0 / 24.0), np.full(24, 2.0 / 7.0 / 24.0)]
    if pa is not None:
        w_tick, w0 = occ * pa * has, float(np.sum(occ * (1.0 - pa)))
    else:
        w_tick, w0 = w_data * has, 0.0
    out: Dict[str, Any] = {}
    for name, f in zip(WORKLOAD_FEATURES, WORKLOAD_IDX):
        ok = has & np.isfinite(mean[:, f]) & np.isfinite(sd15[:, f])
        if not ok.any():
            continue
        blk: Dict[str, Any] = {"unit": f"per {int(BAND_EXPOSURE_S // 60)} min"}
        g15 = _cdf_grid(int(MB.FAMILY[f]), mean[:, f], sd15[:, f], BAND_EXPOSURE_S / 60.0, ok)
        for dtp, lo in (("workday", 0), ("nonworkday", 24)):
            w = np.zeros(48)
            w[lo:lo + 24] = w_act[lo:lo + 24]
            q = _grid_quantiles(g15, w, 0.0, Q_DAYTYPE)
            blk[dtp] = None if q is None else {"p10": q[0], "p50": q[1], "p90": q[2]}
        if per_tick:
            gt = g15 if abs(dt - BAND_EXPOSURE_S) < 1e-6 else \
                _cdf_grid(int(MB.FAMILY[f]), mean[:, f], sd15[:, f], dt / 60.0, ok)
            q = _grid_quantiles(gt, w_tick, w0, Q_TICK)
            if q is not None:
                blk.update(p5=q[0], p50=q[1], p95=q[2], exposure_s=float(dt),
                           includes_inactive=pa is not None)
        out[name] = blk
    return out


class _Grid(NamedTuple):
    """CDF of every kept bucket on one shared grid: C[k, j] = P(X_b(k) <= x_j),
    natural values x, interpolation coordinate u (log1p(x) for counts, the
    transformed value y for t features) and the exposure in minutes."""
    idx: np.ndarray
    x: np.ndarray
    u: np.ndarray
    C: np.ndarray
    integer: bool
    e_min: float


def _cdf_grid(fam: int, mean: np.ndarray, sd15: np.ndarray, e_min: float,
              ok: np.ndarray) -> _Grid:
    """NB family: mean = rate / min and sd15 = sd of the 15-min rate, so the
    15-min count moments give the NB size r exactly (r is exposure-free in the
    Gamma-Poisson predictive). t family (bytes): y = log1p(rate / min) with
    location `mean` and the 15-min spread sd15 (sqrt(15 / e) at exposure e),
    taken as normal; y <= 0 is a natural 0."""
    idx = np.flatnonzero(ok)
    m, sd = mean[idx], sd15[idx]
    if fam == MB.FAM_NB:
        mu = np.maximum(m, 0.0) * e_min
        m15, v15 = 15.0 * np.maximum(m, 0.0), (15.0 * sd) ** 2
        with np.errstate(all="ignore"):
            r = np.where(v15 > m15 * (1.0 + 1e-9), m15 * m15 / (v15 - m15), R_POISSON)
        r = np.where(np.isfinite(r) & (r > 0.0), np.minimum(r, R_POISSON), R_POISSON)
        hi = float(np.max(mu + 8.0 * np.sqrt(mu + mu * mu / r)))
        hi = hi if math.isfinite(hi) and hi >= 1.0 else 1.0
        k = np.unique(np.floor(np.geomspace(1.0, hi + 1.0, GRID_N)))
        x = np.concatenate([[0.0], k[k > 0.0]])
        C = _nb_cdf_rows(x, mu, r)
        return _Grid(idx, x, np.log1p(x), C, True, e_min)
    s = np.maximum(sd * math.sqrt(15.0 / e_min), 1e-6)
    lo, hi = float(np.min(m - 6.0 * s)), float(np.max(m + 6.0 * s))
    y = np.unique(np.concatenate([[0.0], np.linspace(max(0.0, lo), max(hi, 1e-6), GRID_N)]))
    C = sp.ndtr((y[None, :] - m[:, None]) / s[:, None])
    return _Grid(idx, np.expm1(y) * e_min, y, C, False, e_min)


def _nb_cdf_rows(x: np.ndarray, mu: np.ndarray, r: np.ndarray) -> np.ndarray:
    """P(X_b <= x_j) of NB(mu_b, r_b) on an integer grid, [B, J]: the
    regularised incomplete beta I_q(r, x + 1), q = r / (r + mu) (as
    bayes.nb_cdf, without its scalar / validation overhead), Poisson at
    r >= R_POISSON and a point mass at 0 for mu <= 0."""
    X = x[None, :]
    with np.errstate(all="ignore"):
        q = (r / (r + mu))[:, None]
        C = sp.betainc(r[:, None], X + 1.0, q)
        C = np.where((r >= R_POISSON)[:, None], sp.pdtr(X, mu[:, None]), C)
    C = np.where((mu <= 0.0)[:, None], 1.0, C)
    return np.clip(np.nan_to_num(C, nan=1.0), 0.0, 1.0)


def _grid_quantiles(g: _Grid, w: np.ndarray, w0: float, qs: Sequence[float]
                    ) -> Optional[List[float]]:
    """Quantiles of sum_b w_b X_b + w0 delta_0 (weights normalised over the
    grid's buckets), inverted linearly in the grid's coordinate. None when no
    bucket carries weight."""
    ww = np.asarray(w, dtype=np.float64)[g.idx]
    ww = np.where(np.isfinite(ww) & (ww > 0.0), ww, 0.0)
    if not float(ww.sum()) > 0.0:
        return None
    ww = np.where(ww >= MIN_BUCKET_W * float(ww.max()), ww, 0.0)
    tot = float(ww.sum()) + max(0.0, float(w0))
    Fm = max(0.0, float(w0)) / tot + (ww[:, None] / tot * g.C).sum(axis=0)
    Fm = np.maximum.accumulate(np.minimum(Fm, 1.0))
    q = np.asarray(qs, dtype=np.float64)
    j = np.searchsorted(Fm, q - 1e-12, side="left")
    jj = np.clip(j, 1, g.x.size - 1)
    f0, f1 = Fm[jj - 1], Fm[jj]
    with np.errstate(all="ignore"):
        t = np.where(f1 > f0, (q - f0) / (f1 - f0), 1.0)
    u = g.u[jj - 1] + np.clip(t, 0.0, 1.0) * (g.u[jj] - g.u[jj - 1])
    if g.integer:
        v = np.minimum(g.x[jj], np.maximum(g.x[jj - 1] + 1.0, np.ceil(np.expm1(u) - 1e-9)))
    else:
        v = np.maximum(0.0, np.expm1(u) * g.e_min)
    v = np.where(j <= 0, max(0.0, float(g.x[0])), np.where(j >= g.x.size, g.x[-1], v))
    return [float(x) if g.integer else float(f"{x:.4g}") for x in v]


def _mixture_quantiles(means: np.ndarray, r: np.ndarray, w: np.ndarray, w0: float,
                       qs: Sequence[float]) -> Optional[List[float]]:
    """Quantiles of sum_b w_b NB(means_b, r_b) + w0 delta_0 with the means given
    per exposure (a direct entry to the NB grid maths; tests and B29)."""
    means = np.asarray(means, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if not float(np.sum(w)) > 0.0:
        return None if not w0 > 0.0 else [0.0 for _ in qs]
    # an exposure of 15 min with rate = mean / 15 reproduces NB(mean, r)
    m15 = means / 15.0
    sd15 = np.sqrt(means + means * means / r) / 15.0
    g = _cdf_grid(MB.FAM_NB, m15, sd15, 15.0, np.isfinite(means) & (w > 0.0))
    return _grid_quantiles(g, w, w0, qs)


# ================================================================ versioning
def _signature(js: Mapping[str, Any]) -> Dict[str, Any]:
    """Compact state the diffs compare (see module doc)."""
    rh = js.get("rhythm") or {}
    win = {}
    for dtp in ("workday", "nonworkday"):
        b = rh.get(dtp)
        if isinstance(b, Mapping) and b.get("start_h") is not None and b.get("len_h") is not None:
            win[dtp] = [float(b["start_h"]), float(b["len_h"]), b.get("window")]
    wl = {}
    for f, b in (js.get("workload") or {}).items():
        wl[f] = {dtp: [b[dtp]["p10"], b[dtp]["p50"], b[dtp]["p90"]]
                 for dtp in ("workday", "nonworkday") if isinstance(b.get(dtp), Mapping)}
    cats: Dict[str, Dict[str, float]] = {
        "top_templates": {d["template"]: d["share"] for d in js.get("top_templates") or []
                          if d.get("share") is not None},
        "client": dict((js.get("client") or {}).get("shares") or {}),
    }
    role = js.get("role") or {}
    if role.get("activity_mix"):
        cats["activity_mix"] = {f: v for f, v in role["activity_mix"]}
    if js.get("kind") == "class":
        n = max(1, len(js.get("members") or []))
        cats["members"] = {m["ip"]: 1.0 / n for m in js.get("members") or []}
    return {"window": win, "workload": wl, "cats": cats,
            "class_path": (js.get("identity") or {}).get("class_path")
            if js.get("kind") == "entity" else None}


def diff_signatures(old: Mapping[str, Any], new: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Material changes between two signatures (engines.md B30 thresholds)."""
    out: List[Dict[str, Any]] = []
    ow, nw = old.get("window") or {}, new.get("window") or {}
    for dtp in ("workday", "nonworkday"):
        a, b = ow.get(dtp), nw.get(dtp)
        if a is None and b is None:
            continue
        if a is None or b is None:
            sh = math.inf
        else:
            d0 = abs(_hdist(a[0], b[0]))
            d1 = abs(_hdist(a[0] + a[1], b[0] + b[1]))
            sh = max(d0, d1)
        if sh > DIFF_WINDOW_H:
            out.append(_item("rhythm.window", kind="window", day_type=dtp,
                             before=a[2] if a else None, after=b[2] if b else None,
                             shift_h=None if math.isinf(sh) else round(sh, 2)))
    for f, nb in (new.get("workload") or {}).items():
        ob = (old.get("workload") or {}).get(f) or {}
        for dtp, qn in nb.items():
            qo = ob.get(dtp)
            if qo is None:
                continue
            rel = max(abs(x - y) / max(abs(y), 1.0) for x, y in zip(qn, qo))
            if rel > DIFF_QSHIFT:
                out.append(_item("workload", kind="quantile", feature=f, day_type=dtp,
                                 before=list(qo), after=list(qn), max_rel_shift=round(rel, 3)))
    for key, nd in (new.get("cats") or {}).items():
        od = (old.get("cats") or {}).get(key)
        if od is None or (not od and not nd):
            continue
        j = _jsd_dicts(od, nd)
        if j > DIFF_JSD:
            added = [v for v, _ in _top_items(nd, 5) if od.get(v, 0.0) < MEMBER_MIN_SHARE]
            removed = [v for v, _ in _top_items(od, 5) if nd.get(v, 0.0) < MEMBER_MIN_SHARE]
            inc = sorted(((x - od.get(v, 0.0), v) for v, x in nd.items()), reverse=True)
            gained = [v for dx, v in inc[:3] if dx > 0.05]
            out.append(_item(key, kind="categorical", jsd=round(j, 4), added=added,
                             removed=removed, gained=gained,
                             before=[v for v, _ in _top_items(od, 3)],
                             after=[v for v, _ in _top_items(nd, 3)]))
    if old.get("class_path") != new.get("class_path") and new.get("class_path") is not None \
            and old.get("class_path") is not None:
        out.append(_item("class_path", kind="categorical", before=old.get("class_path"),
                         after=new.get("class_path")))
    return out


def _item(field: str, **kw: Any) -> Dict[str, Any]:
    zh, en = LABELS[field]
    return {"field": field, "label_zh": zh, "label_en": en, **kw}


def _publish(store: Any, s: str, key: str, js: Dict[str, Any], sig: Dict[str, Any],
             now: float) -> None:
    prof = store.profile(s, key) or EntityProfile(system=s, entity=key, updated=now)
    if not isinstance(prof.extra, dict):
        prof.extra = {}
    old = prof.extra.get(PORTRAIT)
    old = old if isinstance(old, Mapping) else {}
    v_old = int(old.get("version") or 0)
    prev_sig = old.get("sig")
    diff = diff_signatures(prev_sig, sig) if isinstance(prev_sig, Mapping) else []
    bump = v_old == 0 or bool(diff)
    v = v_old + 1 if bump else v_old
    if not bump:
        diff, sig = list(old.get("diff") or []), prev_sig      # keep the version's reference
    js["version"] = v
    text_zh, text_en = render_zh(js, diff), render_en(js, diff)
    prof.extra[PORTRAIT] = {"json": js, "text_zh": text_zh, "text_en": text_en, "version": v,
                            "diff": diff, "updated": now, "sig": sig}
    prof.updated = now
    store.put_profile(prof)
    if bump:
        store.put_profile_version(s, key, v, {"kind": PORTRAIT, "version": v, "ts": now,
                                              "json": js, "text_zh": text_zh,
                                              "text_en": text_en, "diff": diff}, ts=now)


# ================================================================== texts
def render_zh(js: Mapping[str, Any], diff: Sequence[Mapping[str, Any]] = ()) -> str:
    """Chinese narrative from the json alone (templates only, no raw values)."""
    parts: List[str] = []
    if js.get("kind") == "class":
        kind = js.get("class_kind") or "role"
        head = f"{js.get('name') or js.get('entity')}（{KIND_ZH.get(kind, kind)}，" \
               f"{js.get('n_members', 0)}个成员"
        if js.get("cohesion") is not None:
            head += f"，内聚度{js['cohesion']:.2f}"
        head += "）"
    else:
        idn = js.get("identity") or {}
        head = f"{js.get('entity')}："
        sup = SUPER_ZH.get(str(idn.get("super")), "")
        if idn.get("role_name") or sup:
            head += f"{idn.get('role_name') or ''} {sup}".strip()
            det = []
            if idn.get("class_size"):
                det.append(f"同类{idn['class_size']}个")
            if idn.get("class_prob") is not None:
                det.append(f"置信{idn['class_prob']:.2f}")
            if det:
                head += "(" + ",".join(det) + ")"
        elif js.get("status") == "no_model":
            head += "尚无行为模型"
    parts.append(head)
    rh = js.get("rhythm") or {}
    for dtp in ("workday", "nonworkday"):
        b = rh.get(dtp)
        if isinstance(b, Mapping) and b.get("window"):
            parts.append(f"{DAYTYPE_ZH[dtp]}{b['window']}活跃")
            break
    wl = (js.get("workload") or {}).get("http_requests") or {}
    wd = wl.get("workday") or wl.get("nonworkday")
    if isinstance(wd, Mapping) and wd.get("p10") is not None:
        agg = "聚合" if js.get("kind") == "class" else ""
        parts.append(f"{agg}典型每刻{_n(wd['p10'])}–{_n(wd['p90'])}请求")
    tt = js.get("top_templates") or []
    if tt:
        parts.append("常用" + "、".join(d["template"] for d in tt[:2]))
    if js.get("kind") == "class" and js.get("common_tokens"):
        parts.append("共同模板" + "、".join(d["template"] for d in js["common_tokens"][:2]))
    cl = (js.get("client") or {}).get("dominant") or []
    if cl:
        parts.append(f"终端{cl[0]['ua']}({cl[0]['os']})")
    tm = js.get("timing") or {}
    if tm.get("think_median_s") is not None:
        parts.append(f"思考时间中位{_n(tm['think_median_s'])}秒")
    idn = js.get("identity") or {}
    if idn.get("identifiability") is not None:
        s = f"可辨识度{idn['identifiability']:.2f}"
        if idn.get("confusable_with"):
            s += f"(易混淆:{','.join(idn['confusable_with'][:2])})"
        parts.append(s)
    if js.get("kind") == "class" and js.get("outlier_members"):
        parts.append("离群成员" + "、".join(js["outlier_members"][:3]))
    rk = js.get("risk") or {}
    if rk.get("tier"):
        parts.append(f"风险{TIER_ZH.get(str(rk['tier']), rk['tier'])}"
                     + (f"({rk['score']:.2f})" if rk.get("score") is not None else ""))
    txt = _join(parts, "：", "，")
    if diff:
        txt += "；较上一版变化：" + "、".join(_diff_zh(d) for d in diff[:4])
    return txt + "。"


def render_en(js: Mapping[str, Any], diff: Sequence[Mapping[str, Any]] = ()) -> str:
    parts: List[str] = []
    if js.get("kind") == "class":
        kind = js.get("class_kind") or "role"
        head = f"{js.get('name') or js.get('entity')} ({KIND_EN.get(kind, kind)}, " \
               f"{js.get('n_members', 0)} members"
        if js.get("cohesion") is not None:
            head += f", cohesion {js['cohesion']:.2f}"
        head += ")"
    else:
        idn = js.get("identity") or {}
        head = f"{js.get('entity')}:"
        sup = SUPER_EN.get(str(idn.get("super")), "")
        if idn.get("role_name") or sup:
            head += " " + f"{idn.get('role_name') or ''} {sup}".strip()
            det = []
            if idn.get("class_size"):
                det.append(f"{idn['class_size']} in class")
            if idn.get("class_prob") is not None:
                det.append(f"confidence {idn['class_prob']:.2f}")
            if det:
                head += " (" + ", ".join(det) + ")"
        elif js.get("status") == "no_model":
            head += " no behaviour model yet"
    parts.append(head)
    rh = js.get("rhythm") or {}
    for dtp in ("workday", "nonworkday"):
        b = rh.get(dtp)
        if isinstance(b, Mapping) and b.get("window"):
            parts.append(f"active on {dtp}s {b['window']}")
            break
    wl = (js.get("workload") or {}).get("http_requests") or {}
    wd = wl.get("workday") or wl.get("nonworkday")
    if isinstance(wd, Mapping) and wd.get("p10") is not None:
        agg = "aggregate " if js.get("kind") == "class" else ""
        parts.append(f"typically {agg}{_n(wd['p10'])}-{_n(wd['p90'])} requests per 15 min")
    tt = js.get("top_templates") or []
    if tt:
        parts.append("top " + ", ".join(d["template"] for d in tt[:2]))
    if js.get("kind") == "class" and js.get("common_tokens"):
        parts.append("common " + ", ".join(d["template"] for d in js["common_tokens"][:2]))
    cl = (js.get("client") or {}).get("dominant") or []
    if cl:
        parts.append(f"client {cl[0]['ua']} ({cl[0]['os']})")
    tm = js.get("timing") or {}
    if tm.get("think_median_s") is not None:
        parts.append(f"median think time {_n(tm['think_median_s'])} s")
    idn = js.get("identity") or {}
    if idn.get("identifiability") is not None:
        s = f"identifiability {idn['identifiability']:.2f}"
        if idn.get("confusable_with"):
            s += f" (confusable with: {', '.join(idn['confusable_with'][:2])})"
        parts.append(s)
    if js.get("kind") == "class" and js.get("outlier_members"):
        parts.append("outliers " + ", ".join(js["outlier_members"][:3]))
    rk = js.get("risk") or {}
    if rk.get("tier"):
        parts.append(f"risk {rk['tier']}"
                     + (f" ({rk['score']:.2f})" if rk.get("score") is not None else ""))
    txt = _join(parts, ":", ", ")
    if diff:
        txt += "; changed since the previous version: " + ", ".join(_diff_en(d) for d in diff[:4])
    return txt + "."


def _join(parts: Sequence[str], colon: str, sep: str) -> str:
    """Head plus clauses; no separator right after a bare 'key：' head."""
    head, body = parts[0], [p for p in parts[1:] if p]
    if not body:
        return head.rstrip(colon).rstrip()
    glue = (" " if colon == ":" else "") if head.endswith(colon) else sep
    return head + glue + sep.join(body)


def _diff_zh(d: Mapping[str, Any]) -> str:
    lab = d.get("label_zh", "")
    if d.get("kind") == "window":
        return f"{lab}({DAYTYPE_ZH.get(d.get('day_type'), '')}{d.get('before') or '无'}→" \
               f"{d.get('after') or '无'})"
    if d.get("kind") == "quantile":
        return f"{lab}({FEATURE_ZH.get(d.get('feature'), d.get('feature'))}," \
               f"{DAYTYPE_ZH.get(d.get('day_type'), '')})"
    if d.get("added"):
        return f"{lab}(新增{'、'.join(map(str, d['added'][:2]))})"
    return lab


def _diff_en(d: Mapping[str, Any]) -> str:
    lab = d.get("label_en", "")
    if d.get("kind") == "window":
        return (f"{lab} ({d.get('day_type')} {d.get('before') or 'none'} -> "
                f"{d.get('after') or 'none'})")
    if d.get("kind") == "quantile":
        return f"{lab} ({FEATURE_EN.get(d.get('feature'), d.get('feature'))}, {d.get('day_type')})"
    if d.get("added"):
        return f"{lab} (new: {', '.join(map(str, d['added'][:2]))})"
    return lab


# ================================================================== helpers
def _safe_token(tok: Any) -> str:
    """Re-mask an HTTP token / template so no path or query VALUE survives
    (placeholders kept, other segments through mask_segment, query -> names)."""
    if not isinstance(tok, str):
        return "" if tok is None else str(tok)
    if not tok or tok[:1] == "{" or channel_of(tok) != "http":
        return tok
    parts = tok.split(" ", 2)
    if len(parts) < 3:
        return tok
    meth, host, rest = parts
    path, sep, status = rest.partition("|")
    return f"{meth} {host} {_safe_path(path)}" + (sep + status if sep else "")


def _safe_path(path: str) -> str:
    p, _, query = path.partition("?")
    segs = []
    for seg in p.split("/"):
        if not seg:
            continue
        segs.append(seg if _PLACEHOLDER.match(seg) else mask_segment(seg)[0])
    out = "/" + "/".join(segs)
    if query:
        names = sorted({n if _PLACEHOLDER.match(n) else mask_segment(n)[0]
                        for n in (x.split("=", 1)[0] for x in query.split("&")) if n})
        if names:
            out += "?" + "&".join(names)
    return out


def _safe_family(fam: str) -> str:
    """'http|read|host|seg0': seg0 is the first template segment; re-masked."""
    ps = str(fam).split("|")
    if len(ps) == 4 and ps[0] == "http" and ps[3] and not _PLACEHOLDER.match(ps[3]):
        ps[3] = mask_segment(ps[3])[0]
    return "|".join(ps)


def _wilson(p: Any, n: Any, z: float = WILSON_Z) -> Optional[List[float]]:
    """Wilson score interval of a proportion estimated from n held-out windows."""
    p, n = _num(p), _num(n)
    if p is None or n is None or not n >= 1.0:
        return None
    p = min(1.0, max(0.0, p))
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def _jsd_dicts(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """JSD (bits) of two top-k share maps, each completed with an '__other__' mass."""
    keys = sorted(set(a) | set(b))
    pa = [max(0.0, float(a.get(k, 0.0))) for k in keys]
    pb = [max(0.0, float(b.get(k, 0.0))) for k in keys]
    pa.append(max(0.0, 1.0 - sum(pa)))
    pb.append(max(0.0, 1.0 - sum(pb)))
    j = MT.jsd_bits(pa, pb)
    return j if j == j else (0.0 if not (sum(pa) or sum(pb)) else 1.0)


def _top_items(d: Mapping[str, float], k: int) -> List[Tuple[str, float]]:
    return sorted(((str(v), float(x)) for v, x in d.items() if x == x),
                  key=lambda t: (-t[1], t[0]))[:k]


def _longest_run(on: np.ndarray) -> Optional[Tuple[int, int]]:
    """(start, length) of the longest circular run of True."""
    on = np.asarray(on, dtype=bool)
    n = on.size
    if n == 0 or not on.any():
        return None
    if on.all():
        return 0, n
    k = int(np.flatnonzero(~on)[0])
    r = np.roll(on, -k)
    best = best_s = cur = cur_s = 0
    for i, v in enumerate(r.tolist()):
        if v:
            if cur == 0:
                cur_s = i
            cur += 1
            if cur > best:
                best, best_s = cur, cur_s
        else:
            cur = 0
    return (best_s + k) % n, best


def _fmt_window(h0: float, h1: float) -> str:
    return f"{_hhmm(h0)}–{_hhmm(h1)}"


def _hhmm(h: float) -> str:
    m = int(round(float(h) * 60.0)) % (24 * 60)
    if float(h) >= 24.0 and m == 0:
        return "24:00"
    return f"{m // 60:02d}:{m % 60:02d}"


def _win_str(w: Any) -> Optional[str]:
    if not isinstance(w, Mapping) or w.get("start_h") is None or w.get("len_h") is None:
        return None
    h0 = float(w["start_h"])
    return _fmt_window(h0, h0 + float(w["len_h"]))


def _hdist(a: float, b: float) -> float:
    return ((float(b) - float(a) + 12.0) % 24.0) - 12.0


def _sev(x: Any) -> Any:
    return getattr(x, "value", x)


def _n(x: float) -> str:
    x = float(x)
    if abs(x) >= 100 or float(x).is_integer():
        return f"{x:.0f}"
    return f"{x:.1f}"


def _num(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _int(x: Any) -> Optional[int]:
    v = _num(x)
    return int(v) if v is not None else None


def _clean(x: Any, nd: int = 4) -> Any:
    """JSON-safe copy: numpy scalars / arrays to Python, non-finite -> None,
    floats below 1e6 rounded to nd decimals (timestamps and byte counts kept).
    Exact type checks first: this walks every node of every portrait."""
    t = type(x)
    if t is float:
        if x != x or x in (math.inf, -math.inf):
            return None
        return round(x + 0.0, nd) if -1e6 < x < 1e6 else x
    if t is str or t is int or t is bool or x is None:
        return x
    if t is dict:
        return {str(k): _clean(v, nd) for k, v in x.items()}
    if t is list or t is tuple:
        return [_clean(v, nd) for v in x]
    if isinstance(x, np.ndarray):
        return [_clean(v, nd) for v in x.tolist()]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return _clean(float(x), nd)
    if isinstance(x, Mapping):
        return {str(k): _clean(v, nd) for k, v in x.items()}
    return x
